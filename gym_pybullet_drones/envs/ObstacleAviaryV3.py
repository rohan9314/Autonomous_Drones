"""ObstacleAviaryV3 — multi-waypoint drone navigation with lidar and procedural obstacles.

Key differences from V2:
  - Sequential multi-waypoint task: visit N waypoints in order
  - Observation = base_kin | 48-ray lidar | current_wp_vec(3) | next_wp_vec(3) | final_wp_vec(3) | progress_fraction(1)
  - Difficulty controls both obstacle count AND waypoint count simultaneously
  - Curriculum metric: fraction_completed (continuous) instead of binary success
  - Visual waypoint markers in GUI mode (green=done, orange=active, yellow=future)

Observation layout (58 extra dims beyond base):
  [0 : base_dim]          kinematic state + action buffer (inherited)
  [base_dim : +48]        48-ray lidar, normalized [0,1]
  [base_dim+48 : +51]     current waypoint vector  (WP[i] - drone_pos)
  [base_dim+51 : +54]     next waypoint vector     (WP[i+1] - drone_pos, zeros if none)
  [base_dim+54 : +57]     final waypoint vector    (WP[N-1] - drone_pos)
  [base_dim+57]           waypoint progress fraction  i / N  in [0, 1]

Difficulty table:
  0 — 2 waypoints, 0 obstacles    (pure waypoint warmup)
  1 — 3 waypoints, 1–3 obstacles
  2 — 4 waypoints, 3–6 obstacles
  3 — 5 waypoints, 5–10 obstacles
  4 — 6 waypoints, 8–15 obstacles  (hardest)
"""

import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


LIDAR_RANGE    = 3.0
N_LIDAR_HORIZ  = 36
N_LIDAR_VERT   = 12
N_LIDAR_TOTAL  = N_LIDAR_HORIZ + N_LIDAR_VERT   # 48

# (n_waypoints, obstacle_count_lo, obstacle_count_hi)
DIFFICULTY_CONFIG = {
    0: (2, 0,  0),
    1: (3, 1,  3),
    2: (4, 3,  6),
    3: (5, 5, 10),
    4: (6, 8, 15),
}

SUCCESS_RADIUS_INTER = 0.25   # metres — intermediate waypoints (looser)
SUCCESS_RADIUS_FINAL = 0.15   # metres — final waypoint (tighter)
MIN_WP_SEPARATION    = 0.6    # metres — min gap between consecutive waypoints
OBSTACLE_CLEARANCE   = 0.35   # metres — reject waypoints inside this radius of any obstacle

WP_BONUS     = 3.0    # reward for reaching each intermediate waypoint
FINAL_BONUS  = 8.0    # reward for reaching the final waypoint
TIME_PENALTY = -0.005 # per control step (prevents indefinite hovering)


class ObstacleAviaryV3(BaseRLAviary):
    """Single-agent RL task: visit N sequential waypoints while avoiding obstacles.

    The observation space is fixed-size regardless of N (current/next/final WP vectors
    plus a progress fraction) so the same policy generalises across all difficulty levels.
    """

    def __init__(
        self,
        drone_model: DroneModel  = DroneModel.CF2X,
        initial_xyzs             = None,
        initial_rpys             = None,
        physics: Physics         = Physics.PYB,
        pyb_freq: int            = 240,
        ctrl_freq: int           = 30,
        gui: bool                = False,
        record: bool             = False,
        obs: ObservationType     = ObservationType.KIN,
        act: ActionType          = ActionType.PID,
        difficulty: int          = 1,
    ):
        self.difficulty = difficulty
        n_wp, obs_lo, obs_hi = DIFFICULTY_CONFIG.get(difficulty, (3, 1, 3))
        self.n_waypoints  = n_wp
        self._obs_lo      = obs_lo
        self._obs_hi      = obs_hi
        self.EPISODE_LEN_SEC = self._episode_len(difficulty, n_wp)

        # Episode state — fully reset each episode
        self.waypoints       = np.zeros((n_wp, 3), dtype=np.float64)
        self.current_wp_idx  = 0
        self.prev_dist_to_wp = 0.0
        self._last_lidar     = np.ones(N_LIDAR_TOTAL, dtype=np.float32)
        self.obstacle_ids    = []
        self._wp_visual_ids  = []
        self._all_complete   = False

        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=act,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _episode_len(difficulty: int, n_waypoints: int) -> float:
        per_wp = 3.0 if difficulty >= 3 else 4.0
        return 4.0 + per_wp * n_waypoints

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        # Apply any difficulty change pushed by curriculum callback
        n_wp, obs_lo, obs_hi = DIFFICULTY_CONFIG.get(self.difficulty, (3, 1, 3))
        self.n_waypoints     = n_wp
        self._obs_lo         = obs_lo
        self._obs_hi         = obs_hi
        self.EPISODE_LEN_SEC = self._episode_len(self.difficulty, n_wp)

        self.current_wp_idx  = 0
        self._all_complete   = False
        self._last_lidar     = np.ones(N_LIDAR_TOTAL, dtype=np.float32)
        self._wp_visual_ids  = []
        self.waypoints       = np.zeros((n_wp, 3), dtype=np.float64)

        # super().reset() → _housekeeping() → _addObstacles() uses self._obs_lo/_obs_hi
        obs_out, info = super().reset(seed=seed, options=options)

        # Generate waypoints after obstacles are placed (so we can check clearance)
        self._generate_waypoints()
        self._place_wp_markers()

        state = self._getDroneStateVector(0)
        self.prev_dist_to_wp = float(np.linalg.norm(self.waypoints[0] - state[0:3]))

        return obs_out, info

    # ------------------------------------------------------------------
    # Waypoint generation
    # ------------------------------------------------------------------

    def _generate_waypoints(self):
        """Sample n_waypoints positions that clear obstacles and prior waypoints."""
        rng       = self.np_random
        placed    = []
        attempts  = 0
        max_tries = self.n_waypoints * 300

        while len(placed) < self.n_waypoints and attempts < max_tries:
            attempts += 1
            pos = np.array([
                float(rng.uniform(0.3, 2.7)),
                float(rng.uniform(-1.3, 1.3)),
                float(rng.uniform(0.3, 2.2)),
            ])

            # Must be far enough from drone start (origin)
            if np.linalg.norm(pos) < MIN_WP_SEPARATION:
                continue

            # Must be far enough from previously placed waypoints
            if any(np.linalg.norm(pos - w) < MIN_WP_SEPARATION for w in placed):
                continue

            # Must not be inside an obstacle
            in_obs = False
            for oid in self.obstacle_ids:
                obs_pos, _ = p.getBasePositionAndOrientation(oid, physicsClientId=self.CLIENT)
                if np.linalg.norm(pos - np.array(obs_pos)) < OBSTACLE_CLEARANCE:
                    in_obs = True
                    break
            if in_obs:
                continue

            placed.append(pos)

        # Fallback grid if sampling fails (e.g. very dense obstacles)
        while len(placed) < self.n_waypoints:
            idx = len(placed)
            placed.append(np.array([0.5 + idx * 0.5, 0.0, 1.0]))

        self.waypoints = np.array(placed, dtype=np.float64)

    # ------------------------------------------------------------------
    # GUI waypoint markers
    # ------------------------------------------------------------------

    def _place_wp_markers(self):
        if not self.GUI:
            return
        self._wp_visual_ids = []
        for wp in self.waypoints:
            vis = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.07,
                rgbaColor=[1.0, 1.0, 0.0, 0.6],
                physicsClientId=self.CLIENT,
            )
            bid = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=vis,
                basePosition=wp.tolist(),
                physicsClientId=self.CLIENT,
            )
            self._wp_visual_ids.append(bid)
        self._update_wp_markers()

    def _update_wp_markers(self):
        if not self.GUI or not self._wp_visual_ids:
            return
        for i, vid in enumerate(self._wp_visual_ids):
            if i < self.current_wp_idx:
                color = [0.0, 0.8, 0.0, 0.4]   # green — completed
            elif i == self.current_wp_idx:
                color = [1.0, 0.5, 0.0, 0.9]   # orange — active target
            else:
                color = [1.0, 1.0, 0.0, 0.5]   # yellow — future
            p.changeVisualShape(vid, -1, rgbaColor=color, physicsClientId=self.CLIENT)

    # ------------------------------------------------------------------
    # Action preprocessing (same step-size cap as V2)
    # ------------------------------------------------------------------

    def _preprocessAction(self, action):
        self.action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES, 4))
        for k in range(action.shape[0]):
            target   = action[k, :]
            state    = self._getDroneStateVector(k)
            next_pos = self._calculateNextStep(
                current_position=state[0:3],
                destination=target,
                step_size=0.10,
            )
            rpm_k, _, _ = self.ctrl[k].computeControl(
                control_timestep=self.CTRL_TIMESTEP,
                cur_pos=state[0:3],
                cur_quat=state[3:7],
                cur_vel=state[10:13],
                cur_ang_vel=state[13:16],
                target_pos=next_pos,
            )
            rpm[k, :] = rpm_k
        return rpm

    # ------------------------------------------------------------------
    # Obstacle placement
    # ------------------------------------------------------------------

    def _addObstacles(self):
        """Spawn randomised obstacles. Called by BaseAviary._housekeeping() on reset."""
        self.obstacle_ids = []
        lo, hi = self._obs_lo, self._obs_hi
        if hi == 0:
            return

        n_obs = int(self.np_random.integers(lo, hi + 1))
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        spawned  = 0
        attempts = 0
        while spawned < n_obs and attempts < n_obs * 20:
            attempts += 1
            ox = float(self.np_random.uniform(0.15, 2.85))
            oy = float(self.np_random.uniform(-1.3,  1.3))
            oz = float(self.np_random.uniform(0.15,  2.0))
            he = float(self.np_random.uniform(0.05,  0.20))

            if np.linalg.norm([ox, oy, oz]) < 0.35:
                continue

            use_cylinder = (spawned % 2 == 1)
            if use_cylinder:
                col = p.createCollisionShape(
                    p.GEOM_CYLINDER, radius=he, height=he * 2.0,
                    physicsClientId=self.CLIENT)
                vis = p.createVisualShape(
                    p.GEOM_CYLINDER, radius=he, length=he * 2.0,
                    rgbaColor=[0.2, 0.4, 0.9, 1.0], physicsClientId=self.CLIENT)
            else:
                col = p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=[he, he, he],
                    physicsClientId=self.CLIENT)
                vis = p.createVisualShape(
                    p.GEOM_BOX, halfExtents=[he, he, he],
                    rgbaColor=[0.8, 0.2, 0.2, 1.0], physicsClientId=self.CLIENT)

            body_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[ox, oy, oz],
                physicsClientId=self.CLIENT,
            )
            self.obstacle_ids.append(body_id)
            spawned += 1

    # ------------------------------------------------------------------
    # Lidar
    # ------------------------------------------------------------------

    def _computeLidar(self) -> np.ndarray:
        """Cast 48 rays and return hit fractions (1.0=clear, 0.0=contact)."""
        state     = self._getDroneStateVector(0)
        drone_pos = state[0:3]

        h_angles = np.linspace(0.0, 2.0 * np.pi, N_LIDAR_HORIZ, endpoint=False)
        h_dirs   = np.column_stack([
            np.cos(h_angles), np.sin(h_angles), np.zeros(N_LIDAR_HORIZ),
        ])

        v_angles = np.linspace(-np.pi / 4.0, np.pi / 4.0, N_LIDAR_VERT)
        v_dirs   = np.column_stack([
            np.cos(v_angles), np.zeros(N_LIDAR_VERT), np.sin(v_angles),
        ])

        all_dirs  = np.vstack([h_dirs, v_dirs])
        ray_froms = [drone_pos.tolist()] * N_LIDAR_TOTAL
        ray_tos   = [(drone_pos + d * LIDAR_RANGE).tolist() for d in all_dirs]

        results   = p.rayTestBatch(ray_froms, ray_tos, physicsClientId=self.CLIENT)
        return np.array([r[2] for r in results], dtype=np.float32)

    # ------------------------------------------------------------------
    # Observation space
    # ------------------------------------------------------------------

    def _observationSpace(self):
        base_space = super()._observationSpace()
        n_extra    = N_LIDAR_TOTAL + 10  # 48 lidar + 3 curr_wp + 3 next_wp + 3 final_wp + 1 progress
        low  = np.hstack([base_space.low,  np.full((1, n_extra), -np.inf, dtype=np.float32)])
        high = np.hstack([base_space.high, np.full((1, n_extra),  np.inf, dtype=np.float32)])
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _computeObs(self):
        """base | lidar(48) | curr_wp(3) | next_wp(3) | final_wp(3) | progress(1)."""
        base_obs  = super()._computeObs()
        state     = self._getDroneStateVector(0)
        drone_pos = state[0:3]

        lidar = self._computeLidar()
        self._last_lidar = lidar

        curr_wp  = self.waypoints[self.current_wp_idx] - drone_pos
        next_wp  = (self.waypoints[self.current_wp_idx + 1] - drone_pos
                    if self.current_wp_idx + 1 < self.n_waypoints
                    else np.zeros(3))
        final_wp = self.waypoints[-1] - drone_pos
        progress = np.array([self.current_wp_idx / max(self.n_waypoints, 1)], dtype=np.float32)

        extra = np.concatenate([
            lidar,
            curr_wp.astype(np.float32),
            next_wp.astype(np.float32),
            final_wp.astype(np.float32),
            progress,
        ]).reshape(1, -1)

        return np.hstack([base_obs, extra]).astype(np.float32)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _computeReward(self):
        """Potential-based progress + waypoint bonuses + clearance + crash + time penalty.

        Waypoint pointer is advanced HERE (before distance delta computation) so that
        prev_dist and curr_dist always refer to the same target waypoint.
        """
        state     = self._getDroneStateVector(0)
        drone_pos = state[0:3]
        velocity  = state[10:13]
        r_total   = 0.0

        if self.current_wp_idx < self.n_waypoints:
            curr_dist = float(np.linalg.norm(
                self.waypoints[self.current_wp_idx] - drone_pos))

            is_final   = (self.current_wp_idx == self.n_waypoints - 1)
            radius     = SUCCESS_RADIUS_FINAL if is_final else SUCCESS_RADIUS_INTER
            arrived    = curr_dist < radius

            if arrived:
                # Waypoint reached — give bonus and advance pointer
                r_total += FINAL_BONUS if is_final else WP_BONUS
                if is_final:
                    self._all_complete = True
                else:
                    self.current_wp_idx += 1
                    curr_dist = float(np.linalg.norm(
                        self.waypoints[self.current_wp_idx] - drone_pos))
                self.prev_dist_to_wp = curr_dist
                # Update GUI markers after pointer advance
                self._update_wp_markers()
            else:
                # Potential-based progress: velocity component toward goal
                goal_dir  = self.waypoints[self.current_wp_idx] - drone_pos
                goal_norm = np.linalg.norm(goal_dir)
                if goal_norm > 1e-6:
                    r_progress = float(np.dot(velocity, goal_dir / goal_norm))
                else:
                    r_progress = 0.0

                r_delta          = 1.5 * (self.prev_dist_to_wp - curr_dist)
                self.prev_dist_to_wp = curr_dist
                r_total += r_progress + r_delta

        # Obstacle clearance bonus (incentivise keeping distance from walls)
        r_total += 0.01 * float(np.min(self._last_lidar))

        # Crash penalty
        for oid in self.obstacle_ids:
            if p.getContactPoints(self.DRONE_IDS[0], oid, physicsClientId=self.CLIENT):
                r_total += -3.0
                break

        # Time penalty (discourages hovering near current waypoint)
        r_total += TIME_PENALTY

        return r_total

    # ------------------------------------------------------------------
    # Termination / truncation
    # ------------------------------------------------------------------

    def _computeTerminated(self):
        if not self._all_complete:
            return False
        # Final waypoint reached — but only count as success if not in contact
        for oid in self.obstacle_ids:
            if p.getContactPoints(self.DRONE_IDS[0], oid, physicsClientId=self.CLIENT):
                self._all_complete = False
                return False
        return True

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        # Out-of-bounds
        if abs(state[0]) > 3.5 or abs(state[1]) > 2.0 or state[2] > 3.0 or state[2] < 0.05:
            return True
        # Excessive tilt
        if abs(state[7]) > 0.4 or abs(state[8]) > 0.4:
            return True
        # Obstacle contact (crash)
        for oid in self.obstacle_ids:
            if p.getContactPoints(self.DRONE_IDS[0], oid, physicsClientId=self.CLIENT):
                return True
        # Timeout
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeInfo(self):
        state     = self._getDroneStateVector(0)
        drone_pos = state[0:3]
        dist      = float(np.linalg.norm(
            self.waypoints[min(self.current_wp_idx, self.n_waypoints - 1)] - drone_pos))
        frac      = self.current_wp_idx / max(self.n_waypoints, 1)
        return {
            "dist_to_current_wp":  dist,
            "waypoints_completed": self.current_wp_idx,
            "fraction_completed":  frac,
            "n_waypoints":         self.n_waypoints,
            "difficulty":          self.difficulty,
            "n_obstacles":         len(self.obstacle_ids),
            "success":             self._all_complete,
        }
