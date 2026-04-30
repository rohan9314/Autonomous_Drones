"""ObstacleAviaryV2 — drone obstacle avoidance with ray-cast lidar and procedural obstacles.

Key differences from v1:
  - 48-ray lidar ring replaces oracle obstacle positions (forces real perception)
  - Procedural randomized obstacles on every reset (1–15 depending on difficulty)
  - Velocity-direction progress reward + clearance bonus (removes quadratic proximity repulsion)
  - Difficulty controls obstacle count, not fixed obstacle positions
"""

import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


LIDAR_RANGE   = 3.0   # metres — rays beyond this are treated as clear
N_LIDAR_HORIZ = 36    # rays in horizontal ring
N_LIDAR_VERT  = 12    # rays in forward vertical fan
N_LIDAR_TOTAL = N_LIDAR_HORIZ + N_LIDAR_VERT  # 48

# Obstacle count range per difficulty level.
# Difficulty controls how many random obstacles are spawned each episode.
DIFFICULTY_OBSTACLE_COUNTS = {
    0: (0, 0),    # no obstacles
    1: (1, 3),    # 1–3 obstacles
    2: (3, 6),    # 3–6 obstacles
    3: (5, 10),   # 5–10 obstacles
    4: (8, 15),   # 8–15 obstacles
}


class ObstacleAviaryV2(BaseRLAviary):
    """Single-agent RL task: fly from origin to [1,0,1] while avoiding obstacles.

    Curriculum via `difficulty` (0–4):
      0 — no obstacles
      1 — 1–3 randomly placed obstacles
      2 — 3–6 randomly placed obstacles
      3 — 5–10 randomly placed obstacles
      4 — 8–15 randomly placed obstacles (hardest)

    Observation vector layout:
      [0:base_dim]           kinematic state + action buffer (inherited)
      [base_dim:base_dim+48] 48-ray lidar distances, normalized [0,1]
      [base_dim+48:base_dim+51] goal vector (TARGET_POS − drone_pos)

    Lidar ring:
      - 36 horizontal rays covering full 360° at drone height
      - 12 forward vertical rays covering ±45° pitch forward
      - Value 1.0 = clear (no hit within LIDAR_RANGE), 0.0 = contact

    Reward:
      r_progress  = v · normalize(goal_dir)   (velocity component toward goal)
      r_delta     = 0.5 * (prev_dist - curr_dist)  (distance delta, secondary)
      r_clearance = 0.01 * min(lidar)         (incentivize keeping distance from walls)
      r_crash     = -3.0 on obstacle contact  (one-time, same step as truncation)
    """

    def __init__(self,
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
        self.TARGET_POS      = np.array([1.0, 0.0, 1.0])
        self.difficulty      = difficulty
        self.EPISODE_LEN_SEC = self._episode_len(difficulty)
        self.prev_dist       = 0.0
        self.obstacle_ids    = []
        self._last_lidar     = np.ones(N_LIDAR_TOTAL, dtype=np.float32)
        self._terminated     = False

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
    def _episode_len(difficulty: int) -> int:
        if difficulty <= 1:
            return 8
        elif difficulty <= 3:
            return 10
        else:
            return 12

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        self.EPISODE_LEN_SEC = self._episode_len(self.difficulty)
        self._terminated     = False
        obs, info = super().reset(seed=seed, options=options)
        state = self._getDroneStateVector(0)
        self.prev_dist = float(np.linalg.norm(self.TARGET_POS - state[0:3]))
        self._last_lidar = np.ones(N_LIDAR_TOTAL, dtype=np.float32)
        return obs, info

    # ------------------------------------------------------------------
    # Action preprocessing (same step-size cap as v1)
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
    # Lidar
    # ------------------------------------------------------------------

    def _computeLidar(self) -> np.ndarray:
        """Cast 48 rays and return hit fractions (1.0=clear, 0.0=contact)."""
        state     = self._getDroneStateVector(0)
        drone_pos = state[0:3]

        # 36 horizontal rays — full ring at drone height
        h_angles = np.linspace(0.0, 2.0 * np.pi, N_LIDAR_HORIZ, endpoint=False)
        h_dirs   = np.column_stack([
            np.cos(h_angles),
            np.sin(h_angles),
            np.zeros(N_LIDAR_HORIZ),
        ])

        # 12 vertical rays — forward-pointing fan (pitch ±45°)
        v_angles = np.linspace(-np.pi / 4.0, np.pi / 4.0, N_LIDAR_VERT)
        v_dirs   = np.column_stack([
            np.cos(v_angles),          # X (forward)
            np.zeros(N_LIDAR_VERT),    # Y
            np.sin(v_angles),          # Z (up/down)
        ])

        all_dirs   = np.vstack([h_dirs, v_dirs])  # (48, 3)
        ray_froms  = [drone_pos.tolist()] * N_LIDAR_TOTAL
        ray_tos    = [(drone_pos + d * LIDAR_RANGE).tolist() for d in all_dirs]

        results   = p.rayTestBatch(ray_froms, ray_tos, physicsClientId=self.CLIENT)
        fractions = np.array([r[2] for r in results], dtype=np.float32)
        # hitFraction: 0.0 = contact at ray origin, 1.0 = no hit within range
        return fractions

    # ------------------------------------------------------------------
    # Obstacle placement
    # ------------------------------------------------------------------

    def _addObstacles(self):
        """Spawn randomized obstacles for the current difficulty level.

        Called by BaseAviary._housekeeping() on every reset(). Obstacles are
        placed randomly each episode — the policy must learn to sense and avoid
        rather than memorize fixed dodge maneuvers.
        """
        self.obstacle_ids = []
        lo, hi = DIFFICULTY_OBSTACLE_COUNTS.get(self.difficulty, (0, 0))
        if hi == 0:
            return

        n_obs = int(self.np_random.integers(lo, hi + 1))

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        spawned = 0
        attempts = 0
        while spawned < n_obs and attempts < n_obs * 20:
            attempts += 1

            ox = float(self.np_random.uniform(0.15, 0.90))
            oy = float(self.np_random.uniform(-0.55, 0.55))
            oz = float(self.np_random.uniform(0.15, 1.15))
            he = float(self.np_random.uniform(0.05, 0.20))  # half-extent

            # reject if too close to start or goal
            dist_start = np.linalg.norm([ox, oy, oz])
            dist_goal  = np.linalg.norm([ox - 1.0, oy, oz - 1.0])
            if dist_start < 0.35 or dist_goal < 0.35:
                continue

            use_cylinder = (spawned % 2 == 1)

            if use_cylinder:
                col = p.createCollisionShape(
                    p.GEOM_CYLINDER,
                    radius=he,
                    height=he * 2.0,
                    physicsClientId=self.CLIENT,
                )
                vis = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=he,
                    length=he * 2.0,
                    rgbaColor=[0.2, 0.4, 0.9, 1.0],
                    physicsClientId=self.CLIENT,
                )
            else:
                col = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=[he, he, he],
                    physicsClientId=self.CLIENT,
                )
                vis = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[he, he, he],
                    rgbaColor=[0.8, 0.2, 0.2, 1.0],
                    physicsClientId=self.CLIENT,
                )

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
    # Observation space
    # ------------------------------------------------------------------

    def _observationSpace(self):
        base_space = super()._observationSpace()
        n_extra    = N_LIDAR_TOTAL + 3   # 48 lidar + 3 goal vector

        low  = np.hstack([base_space.low,
                          np.full((1, n_extra), -np.inf, dtype=np.float32)])
        high = np.hstack([base_space.high,
                          np.full((1, n_extra),  np.inf, dtype=np.float32)])
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _computeObs(self):
        """Base obs | 48-ray lidar | goal vector."""
        base_obs  = super()._computeObs()
        drone_pos = self._getDroneStateVector(0)[0:3]

        lidar = self._computeLidar()
        self._last_lidar = lidar  # cache for reward computation

        goal_vec = self.TARGET_POS - drone_pos
        extra    = np.concatenate([lidar, goal_vec]).astype(np.float32).reshape(1, -1)

        return np.hstack([base_obs, extra]).astype(np.float32)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _computeReward(self):
        """Velocity-direction progress + distance delta + clearance bonus + crash penalty."""
        state     = self._getDroneStateVector(0)
        drone_pos = state[0:3]
        velocity  = state[10:13]

        curr_dist = float(np.linalg.norm(self.TARGET_POS - drone_pos))

        # progress: velocity component in the direction of the goal
        goal_dir  = self.TARGET_POS - drone_pos
        goal_norm = np.linalg.norm(goal_dir)
        if goal_norm > 1e-6:
            r_progress = float(np.dot(velocity, goal_dir / goal_norm))
        else:
            r_progress = 0.0

        # secondary distance-delta term
        r_delta        = 0.5 * (self.prev_dist - curr_dist)
        self.prev_dist = curr_dist

        # clearance bonus — reward staying far from walls/obstacles
        r_clearance = 0.01 * float(np.min(self._last_lidar))

        # crash penalty on obstacle contact
        r_crash = 0.0
        for oid in self.obstacle_ids:
            if p.getContactPoints(self.DRONE_IDS[0], oid, physicsClientId=self.CLIENT):
                r_crash = -3.0
                break

        return r_progress + r_delta + r_clearance + r_crash

    # ------------------------------------------------------------------
    # Termination / truncation
    # ------------------------------------------------------------------

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS - state[0:3]) < 0.15:
            for oid in self.obstacle_ids:
                if p.getContactPoints(self.DRONE_IDS[0], oid, physicsClientId=self.CLIENT):
                    self._terminated = False
                    return False
            self._terminated = True
            return True
        self._terminated = False
        return False

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        if abs(state[0]) > 2.5 or abs(state[1]) > 2.5 or state[2] > 2.5 or state[2] < 0.05:
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
        state = self._getDroneStateVector(0)
        dist  = float(np.linalg.norm(self.TARGET_POS - state[0:3]))
        return {
            "dist_to_goal": dist,
            "difficulty":   self.difficulty,
            "n_obstacles":  len(self.obstacle_ids),
            "success":      self._terminated,
        }
