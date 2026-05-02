"""ObstacleAviaryV3 — multi-waypoint drone navigation with lidar and procedural obstacles.

Key improvements over the original V3:
  - Goal-aligned horizontal lidar: ray 0 always faces current waypoint.
    The policy sees "gap left/right of goal" at fixed indices — eliminates need
    to learn the yaw-to-goal mapping, cutting sample complexity ~2-3x.
  - Body-frame waypoint vectors: goal distances expressed in drone's local frame.
    Translation-invariant to global heading; easier for the policy to generalise.
  - Goal-directed rays (3 new obs dims): explicit path-clearance signal to/around target.
  - Clearance-weighted progress: r_delta scaled by goal line-of-sight fraction.
    When path blocked → approach reward drops → detour moves become relatively rewarded.
  - Guided detour shaping: when path blocked, reward lateral movement toward more open side.
  - Survival bonus (+0.01/step): breaks ties in favour of staying alive.
  - Fixed step size 0.05 m: matches V2 baseline, prevents PID over-tilt at high speeds.

Observation layout (62 extra dims beyond base):
  [0 : base_dim]          kinematic state + action buffer (inherited)
  [base_dim : +48]        48-ray GOAL-ALIGNED lidar — ray 0 faces current WP, [0,1]
  [base_dim+48 : +51]     current waypoint vector  in BODY frame
  [base_dim+51 : +54]     next waypoint vector     in BODY frame (zeros if none)
  [base_dim+54 : +57]     final waypoint vector    in BODY frame
  [base_dim+57]           waypoint progress fraction  i / N  in [0, 1]
  [base_dim+58]           turn sharpness: cos(current-WP dir, next segment)
  [base_dim+59]           goal ray fraction (1=clear path to WP, 0=blocked)
  [base_dim+60]           left ray fraction (90° left of goal, 1=clear)
  [base_dim+61]           right ray fraction (90° right of goal, 1=clear)

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

DANGER_THRESH       = 0.3    # lidar fraction below which smooth danger penalty fires (0.9 m)
GOAL_BLOCKED_THRESH = 0.5    # goal ray fraction below which path is considered blocked
SURVIVAL_BONUS      = 0.01   # per-step reward for staying alive and not crashing

# (n_waypoints, obstacle_count_lo, obstacle_count_hi)
DIFFICULTY_CONFIG = {
    0: (2, 0,  0),
    1: (3, 1,  3),
    2: (4, 3,  6),
    3: (5, 5, 10),
    4: (6, 8, 15),
}

SUCCESS_RADIUS_INTER = 0.25   # metres — intermediate waypoints
SUCCESS_RADIUS_FINAL = 0.15   # metres — final waypoint (tighter)
MIN_WP_SEPARATION    = 0.6    # metres — min gap between consecutive waypoints
OBSTACLE_CLEARANCE   = 0.35   # metres — reject waypoints inside this radius of any obstacle

WP_BONUS     = 3.0    # reward for reaching each intermediate waypoint
FINAL_BONUS  = 8.0    # reward for reaching the final waypoint
TIME_PENALTY = -0.005 # per control step


class ObstacleAviaryV3(BaseRLAviary):
    """Single-agent RL task: visit N sequential waypoints while avoiding obstacles.

    Fixed observation size regardless of N so the same policy generalises across
    all difficulty levels.
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
        self.waypoints        = np.zeros((n_wp, 3), dtype=np.float64)
        self.current_wp_idx   = 0
        self.prev_dist_to_wp  = 0.0
        self._last_lidar      = np.ones(N_LIDAR_TOTAL, dtype=np.float32)
        self._last_goal_rays  = np.ones(3, dtype=np.float32)   # (goal, left, right) fractions
        self.obstacle_ids     = []
        self._wp_visual_ids   = []
        self._all_complete    = False

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
        # More time per waypoint at higher difficulties (more obstacles, longer paths)
        per_wp = 5.0 + difficulty * 1.0
        return 4.0 + per_wp * n_waypoints

    @staticmethod
    def _world_to_body(v_world: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """Rotate a world-frame vector into the drone's body frame.

        quat = [qx, qy, qz, qw] (PyBullet convention).
        R is the body→world rotation matrix; R.T maps world→body.
        """
        qx, qy, qz, qw = quat
        R = np.array([
            [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
            [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
        ], dtype=np.float64)
        return (R.T @ v_world).astype(np.float32)

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

        self.current_wp_idx   = 0
        self._all_complete    = False
        self._last_lidar      = np.ones(N_LIDAR_TOTAL, dtype=np.float32)
        self._last_goal_rays  = np.ones(3, dtype=np.float32)
        self._wp_visual_ids   = []
        self.waypoints        = np.zeros((n_wp, 3), dtype=np.float64)

        obs_out, info = super().reset(seed=seed, options=options)

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

            if np.linalg.norm(pos) < MIN_WP_SEPARATION:
                continue
            if any(np.linalg.norm(pos - w) < MIN_WP_SEPARATION for w in placed):
                continue

            in_obs = False
            for oid in self.obstacle_ids:
                obs_pos, _ = p.getBasePositionAndOrientation(oid, physicsClientId=self.CLIENT)
                if np.linalg.norm(pos - np.array(obs_pos)) < OBSTACLE_CLEARANCE:
                    in_obs = True
                    break
            if in_obs:
                continue

            placed.append(pos)

        # Fallback grid — try several y/z offsets to clear obstacles
        while len(placed) < self.n_waypoints:
            idx    = len(placed)
            chosen = None
            for y in [0.0, 0.4, -0.4, 0.8, -0.8]:
                for z in [1.0, 1.4, 0.6, 1.8]:
                    cand = np.array([0.5 + idx * 0.5, y, z])
                    clearances = [
                        np.linalg.norm(cand - np.array(
                            p.getBasePositionAndOrientation(oid, physicsClientId=self.CLIENT)[0]))
                        for oid in self.obstacle_ids
                    ]
                    if not clearances or min(clearances) >= OBSTACLE_CLEARANCE:
                        chosen = cand
                        break
                if chosen is not None:
                    break
            placed.append(chosen if chosen is not None else np.array([0.5 + idx * 0.5, 0.0, 1.0]))

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
                p.GEOM_SPHERE, radius=0.07,
                rgbaColor=[1.0, 1.0, 0.0, 0.6],
                physicsClientId=self.CLIENT,
            )
            bid = p.createMultiBody(
                baseMass=0, baseVisualShapeIndex=vis,
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
                color = [0.0, 0.8, 0.0, 0.4]
            elif i == self.current_wp_idx:
                color = [1.0, 0.5, 0.0, 0.9]
            else:
                color = [1.0, 1.0, 0.0, 0.5]
            p.changeVisualShape(vid, -1, rgbaColor=color, physicsClientId=self.CLIENT)

    # ------------------------------------------------------------------
    # Action preprocessing — adaptive step size
    # ------------------------------------------------------------------

    def _preprocessAction(self, action):
        self.action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES, 4))
        for k in range(action.shape[0]):
            target = action[k, :]
            state  = self._getDroneStateVector(k)
            next_pos = self._calculateNextStep(
                current_position=state[0:3],
                destination=target,
                step_size=0.05,
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
    # Lidar — goal-aligned horizontal ring
    # ------------------------------------------------------------------

    def _computeLidar(self) -> np.ndarray:
        """Cast 48 rays and return hit fractions (1.0=clear, 0.0=contact).

        Horizontal ring is rotated so ray 0 always faces the current waypoint.
        This gives the policy a goal-relative sensor frame: it consistently sees
        'obstacle left of goal' at low indices and 'obstacle right' at high indices,
        removing the need to learn the yaw-to-goal mapping from scratch.
        """
        state     = self._getDroneStateVector(0)
        drone_pos = state[0:3]

        # Compute goal azimuth to align ray 0 with the goal direction
        goal_vec  = self.waypoints[self.current_wp_idx] - drone_pos
        goal_dist = np.linalg.norm(goal_vec)
        goal_az   = float(np.arctan2(goal_vec[1], goal_vec[0])) if goal_dist > 1e-6 else 0.0

        h_angles = np.linspace(0.0, 2.0 * np.pi, N_LIDAR_HORIZ, endpoint=False) + goal_az
        h_dirs   = np.column_stack([
            np.cos(h_angles), np.sin(h_angles), np.zeros(N_LIDAR_HORIZ),
        ])

        # Vertical rays in the goal-azimuth plane (up/down obstacle detection)
        v_elevs = np.linspace(-np.pi / 4.0, np.pi / 4.0, N_LIDAR_VERT)
        v_dirs  = np.column_stack([
            np.cos(goal_az) * np.cos(v_elevs),
            np.sin(goal_az) * np.cos(v_elevs),
            np.sin(v_elevs),
        ])

        all_dirs  = np.vstack([h_dirs, v_dirs])
        ray_froms = [drone_pos.tolist()] * N_LIDAR_TOTAL
        ray_tos   = [(drone_pos + d * LIDAR_RANGE).tolist() for d in all_dirs]

        results = p.rayTestBatch(ray_froms, ray_tos, physicsClientId=self.CLIENT)
        return np.array([r[2] for r in results], dtype=np.float32)

    def _compute_goal_rays(self) -> np.ndarray:
        """Cast 3 goal-directed rays; return (goal_frac, left_frac, right_frac).

        goal_frac : hit fraction directly toward current waypoint (1=clear path)
        left_frac : hit fraction 90° left of goal in horizontal plane
        right_frac: hit fraction 90° right of goal

        These explicitly encode whether the direct path is open and which side
        has more room for a detour.
        """
        state     = self._getDroneStateVector(0)
        drone_pos = state[0:3]
        goal_vec  = self.waypoints[self.current_wp_idx] - drone_pos
        goal_dist = float(np.linalg.norm(goal_vec))

        if goal_dist < 1e-6:
            return np.ones(3, dtype=np.float32)

        goal_unit = goal_vec / goal_dist
        ray_len   = min(goal_dist, LIDAR_RANGE)

        # Lateral unit vector: 90° CCW in horizontal plane
        lat = np.array([-goal_unit[1], goal_unit[0], 0.0])
        lat_norm = np.linalg.norm(lat)
        lat = lat / lat_norm if lat_norm > 1e-6 else lat

        froms = [drone_pos.tolist()] * 3
        tos   = [
            (drone_pos + goal_unit * ray_len).tolist(),
            (drone_pos + lat * LIDAR_RANGE).tolist(),
            (drone_pos - lat * LIDAR_RANGE).tolist(),
        ]
        results = p.rayTestBatch(froms, tos, physicsClientId=self.CLIENT)
        return np.array([r[2] for r in results], dtype=np.float32)

    # ------------------------------------------------------------------
    # Observation space
    # ------------------------------------------------------------------

    def _observationSpace(self):
        base_space = super()._observationSpace()
        # 48 lidar + 3 curr_wp + 3 next_wp + 3 final_wp + 1 progress + 1 turn_cos + 3 goal_rays
        n_extra = N_LIDAR_TOTAL + 14
        low  = np.hstack([base_space.low,  np.full((1, n_extra), -np.inf, dtype=np.float32)])
        high = np.hstack([base_space.high, np.full((1, n_extra),  np.inf, dtype=np.float32)])
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _computeObs(self):
        """base | lidar(48,goal-aligned) | curr_wp(3,body) | next_wp(3,body) |
           final_wp(3,body) | progress(1) | turn_cos(1) | goal_rays(3)."""
        base_obs  = super()._computeObs()
        state     = self._getDroneStateVector(0)
        drone_pos = state[0:3]
        quat      = state[3:7]   # [qx, qy, qz, qw]

        lidar = self._computeLidar()
        self._last_lidar = lidar

        goal_rays = self._compute_goal_rays()
        self._last_goal_rays = goal_rays

        # Waypoint vectors in drone body frame — translation-invariant to heading
        curr_wp_w  = self.waypoints[self.current_wp_idx] - drone_pos
        next_wp_w  = (self.waypoints[self.current_wp_idx + 1] - drone_pos
                      if self.current_wp_idx + 1 < self.n_waypoints
                      else np.zeros(3))
        final_wp_w = self.waypoints[-1] - drone_pos

        curr_wp  = self._world_to_body(curr_wp_w,  quat) / LIDAR_RANGE
        next_wp  = self._world_to_body(next_wp_w,  quat) / LIDAR_RANGE
        final_wp = self._world_to_body(final_wp_w, quat) / LIDAR_RANGE

        progress = np.array([self.current_wp_idx / max(self.n_waypoints, 1)], dtype=np.float32)

        # Turn sharpness: cos(drone→WP[i] direction, WP[i]→WP[i+1] direction)
        if self.current_wp_idx + 1 < self.n_waypoints:
            dir_curr = curr_wp_w
            dir_next = self.waypoints[self.current_wp_idx + 1] - self.waypoints[self.current_wp_idx]
            nc, nn   = np.linalg.norm(dir_curr), np.linalg.norm(dir_next)
            turn_cos = float(np.dot(dir_curr / nc, dir_next / nn)) if nc > 1e-6 and nn > 1e-6 else 1.0
        else:
            turn_cos = 1.0

        extra = np.concatenate([
            lidar,
            curr_wp,
            next_wp,
            final_wp,
            progress,
            np.array([turn_cos], dtype=np.float32),
            goal_rays,
        ]).reshape(1, -1)

        return np.hstack([base_obs, extra]).astype(np.float32)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _computeReward(self):
        """Multi-component reward for complex obstacle-course navigation.

        Reward hierarchy (strongest to weakest signal):
          +8.0 / +3.0   waypoint bonuses (final / intermediate)
          +0–1.5        clearance-weighted distance progress toward goal
          +0–1.0        velocity shaping toward goal (capped, clearance-scaled)
          +0–0.4        guided detour: lateral reward toward open side when blocked
          +0–0.2        deceleration bonus before sharp corners
          +0.01         survival bonus per step
          −0–0.5        smooth danger penalty approaching obstacles
          −3.0          terminal crash penalty
          −0.005        time penalty per step
        """
        state     = self._getDroneStateVector(0)
        drone_pos = state[0:3]
        velocity  = state[10:13]
        r_total   = 0.0

        goal_frac  = float(self._last_goal_rays[0])   # 1=clear path, 0=blocked
        left_frac  = float(self._last_goal_rays[1])
        right_frac = float(self._last_goal_rays[2])

        # Clearance weight: suppresses approach reward when direct path is blocked
        clearance_w = float(np.clip(goal_frac / GOAL_BLOCKED_THRESH, 0.0, 1.0))

        if self.current_wp_idx < self.n_waypoints:
            curr_dist = float(np.linalg.norm(
                self.waypoints[self.current_wp_idx] - drone_pos))

            is_final = (self.current_wp_idx == self.n_waypoints - 1)
            radius   = SUCCESS_RADIUS_FINAL if is_final else SUCCESS_RADIUS_INTER
            arrived  = curr_dist < radius

            if arrived:
                r_total += FINAL_BONUS if is_final else WP_BONUS
                if is_final:
                    self._all_complete = True
                else:
                    self.current_wp_idx += 1
                    curr_dist = float(np.linalg.norm(
                        self.waypoints[self.current_wp_idx] - drone_pos))
                self.prev_dist_to_wp = curr_dist
                self._update_wp_markers()
            else:
                goal_dir  = self.waypoints[self.current_wp_idx] - drone_pos
                goal_norm = np.linalg.norm(goal_dir)
                goal_unit = goal_dir / goal_norm if goal_norm > 1e-6 else goal_dir

                # Clearance-weighted velocity shaping (capped ±1)
                r_progress = float(np.clip(np.dot(velocity, goal_unit), -1.0, 1.0))
                r_progress *= clearance_w

                # Clearance-weighted distance delta
                r_delta = 1.5 * (self.prev_dist_to_wp - curr_dist) * max(clearance_w, 0.2)
                self.prev_dist_to_wp = curr_dist
                r_total += r_progress + r_delta

                # Guided detour: when path blocked, steer toward the more open side
                if goal_frac < GOAL_BLOCKED_THRESH and goal_norm > 1e-6:
                    goal_flat  = np.array([goal_unit[0], goal_unit[1], 0.0])
                    gf_norm    = np.linalg.norm(goal_flat)
                    if gf_norm > 1e-6:
                        lat     = np.array([-goal_flat[1], goal_flat[0], 0.0]) / gf_norm
                        lat_vel = float(np.dot(velocity, lat))
                        blockage = 1.0 - goal_frac
                        # Positive lat_vel = left, negative = right
                        if left_frac >= right_frac:
                            r_detour = 0.4 * max(0.0, lat_vel) * blockage
                        else:
                            r_detour = 0.4 * max(0.0, -lat_vel) * blockage
                        r_total += r_detour

                # Deceleration bonus before a sharp turn
                if curr_dist < 0.4 and self.current_wp_idx + 1 < self.n_waypoints:
                    dir_next = (self.waypoints[self.current_wp_idx + 1]
                                - self.waypoints[self.current_wp_idx])
                    nn = np.linalg.norm(dir_next)
                    if nn > 1e-6:
                        turn_cos = float(np.dot(goal_unit, dir_next / nn))
                        if turn_cos < 0.5:
                            speed = float(np.linalg.norm(velocity))
                            r_total += 0.2 * (1.0 - turn_cos) * (0.4 - curr_dist) / 0.4 * max(0.0, 1.0 - speed)

        # Survival bonus — small per-step incentive to stay alive
        r_total += SURVIVAL_BONUS

        # Smooth danger shaping — continuous gradient before contact
        min_lidar = float(np.min(self._last_lidar))
        if min_lidar < DANGER_THRESH:
            r_total += -0.5 * (DANGER_THRESH - min_lidar) / DANGER_THRESH

        # Terminal crash penalty (fires once at the collision step)
        for oid in self.obstacle_ids:
            if p.getContactPoints(self.DRONE_IDS[0], oid, physicsClientId=self.CLIENT):
                r_total += -3.0
                break

        r_total += TIME_PENALTY
        return r_total

    # ------------------------------------------------------------------
    # Termination / truncation
    # ------------------------------------------------------------------

    def _computeTerminated(self):
        if not self._all_complete:
            return False
        for oid in self.obstacle_ids:
            if p.getContactPoints(self.DRONE_IDS[0], oid, physicsClientId=self.CLIENT):
                self._all_complete = False
                return False
        return True

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        if abs(state[0]) > 3.5 or abs(state[1]) > 2.0 or state[2] > 3.0 or state[2] < 0.05:
            return True
        if abs(state[7]) > 0.4 or abs(state[8]) > 0.4:
            return True
        for oid in self.obstacle_ids:
            if p.getContactPoints(self.DRONE_IDS[0], oid, physicsClientId=self.CLIENT):
                return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeInfo(self):
        state     = self._getDroneStateVector(0)
        drone_pos = state[0:3]
        dist      = float(np.linalg.norm(
            self.waypoints[min(self.current_wp_idx, self.n_waypoints - 1)] - drone_pos))
        # On a successful episode current_wp_idx stays at n-1 (pointer not incremented
        # after the final waypoint), so we explicitly set fraction to 1.0 on success.
        frac      = 1.0 if self._all_complete else (self.current_wp_idx / max(self.n_waypoints, 1))
        return {
            "dist_to_current_wp":  dist,
            "waypoints_completed": self.current_wp_idx,
            "fraction_completed":  frac,
            "n_waypoints":         self.n_waypoints,
            "difficulty":          self.difficulty,
            "n_obstacles":         len(self.obstacle_ids),
            "success":             self._all_complete,
        }
