import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class ObstacleAviary(BaseRLAviary):
    """Single-agent RL: fly to a target while avoiding randomly placed box obstacles.

    Observation (per step):
        [0:3]   position (x, y, z)
        [3:6]   orientation (roll, pitch, yaw)
        [6:9]   linear velocity
        [9:12]  angular velocity
        [12:15] relative target position (target - pos), unnormalized
        [15:39] 24 LiDAR hit-fractions (0=obstacle at ray origin, 1=no obstacle within range)
        [39:]   action buffer (ACTION_BUFFER_SIZE × 4 values)

    Action: VEL type — [vx, vy, vz, speed] all in [-1, 1].

    Reward:
        - Dense shaping: (prev_dist - curr_dist) × 10  (reward forward progress)
        - Success bonus:  +20 when within TARGET_RADIUS of target  (terminates)
        - Collision:      -10 on any obstacle contact              (terminates)
        - Time penalty:   -0.01 per control step
    """

    N_LIDAR_RAYS = 24   # 16 horizontal + 8 downward-angled
    LIDAR_RANGE = 3.0   # metres
    TARGET_RADIUS = 0.15
    ARENA_HALF = 2.0    # ±2 m in x and y
    EPISODE_LEN_SEC = 15

    # ---------------------------------------------------------------------------

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 30,
        gui: bool = False,
        record: bool = False,
        obs: ObservationType = ObservationType.KIN,
        act: ActionType = ActionType.VEL,
        num_obstacles: int = 4,
    ):
        self.NUM_OBSTACLES = num_obstacles
        self._obstacle_ids = []
        self.TARGET_POS = np.array([1.5, 0.0, 1.0])

        # Random state — will be properly seeded in reset()
        self._rng = np.random.default_rng(0)
        self._start_xy = np.array([0.0, 0.0])
        self._prev_dist = None

        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            initial_xyzs=initial_xyzs if initial_xyzs is not None
                         else np.array([[0.0, 0.0, 0.5]]),
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=act,
        )
        # Modest speed increase — 2x the default
        self.SPEED_LIMIT = 0.06 * self.MAX_SPEED_KMH * (1000 / 3600)

    # ---------------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """Randomize start position, target, and obstacles, then reset simulation."""
        self._rng = np.random.default_rng(seed)

        # Random start in arena
        self._start_xy = self._rng.uniform(-1.5, 1.5, size=2)
        self.INIT_XYZS = np.array([[self._start_xy[0], self._start_xy[1], 0.5]])

        # Random target at least 2 m from start
        for _ in range(200):
            tgt_xy = self._rng.uniform(-1.5, 1.5, size=2)
            if np.linalg.norm(tgt_xy - self._start_xy) >= 2.0:
                break
        self.TARGET_POS = np.array([tgt_xy[0], tgt_xy[1], 1.0])

        obs_out, info = super().reset(seed=seed, options=options)

        state = self._getDroneStateVector(0)
        self._prev_dist = np.linalg.norm(self.TARGET_POS - state[0:3])
        return obs_out, info

    # ---------------------------------------------------------------------------

    def _addObstacles(self):
        """Spawn random box obstacles; called by _housekeeping() on every reset."""
        self._obstacle_ids = []

        # Green sphere to mark the target (GUI only)
        if self.GUI:
            vis = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=self.TARGET_RADIUS,
                rgbaColor=[0.1, 0.9, 0.1, 0.7],
                physicsClientId=self.CLIENT,
            )
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=vis,
                basePosition=self.TARGET_POS.tolist(),
                physicsClientId=self.CLIENT,
            )

        occupied = [self._start_xy.copy(), self.TARGET_POS[:2].copy()]

        for _ in range(self.NUM_OBSTACLES):
            for _attempt in range(100):
                xy = self._rng.uniform(-1.8, 1.8, size=2)
                too_close = any(np.linalg.norm(xy - o) < 0.7 for o in occupied)
                if too_close:
                    continue
                hw = self._rng.uniform(0.1, 0.35, size=2)   # half-width x/y
                hz = self._rng.uniform(0.25, 0.7)            # half-height z
                half_ext = [float(hw[0]), float(hw[1]), float(hz)]
                col = p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=half_ext, physicsClientId=self.CLIENT
                )
                viz = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=half_ext,
                    rgbaColor=[0.8, 0.3, 0.1, 1.0],
                    physicsClientId=self.CLIENT,
                )
                body = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=viz,
                    basePosition=[float(xy[0]), float(xy[1]), float(hz)],
                    physicsClientId=self.CLIENT,
                )
                self._obstacle_ids.append(body)
                occupied.append(xy)
                break

    # ---------------------------------------------------------------------------

    def _observationSpace(self):
        """Custom obs space: kinematics + relative target + LiDAR + action buffer."""
        lo, hi = -np.inf, np.inf
        act_size = 4  # VEL action has 4 components

        core_dim = 12 + 3 + self.N_LIDAR_RAYS   # 39
        buf_dim = self.ACTION_BUFFER_SIZE * act_size
        total = core_dim + buf_dim

        obs_lo = np.full((1, total), lo, dtype=np.float32)
        obs_hi = np.full((1, total), hi, dtype=np.float32)
        # LiDAR fractions are in [0, 1]
        obs_lo[0, 15:15 + self.N_LIDAR_RAYS] = 0.0
        obs_hi[0, 15:15 + self.N_LIDAR_RAYS] = 1.0
        # Action buffer in [-1, 1]
        obs_lo[0, core_dim:] = -1.0
        obs_hi[0, core_dim:] = +1.0

        return spaces.Box(low=obs_lo, high=obs_hi, dtype=np.float32)

    # ---------------------------------------------------------------------------

    def _computeObs(self):
        """Assemble the observation vector."""
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:10]
        vel = state[10:13]
        ang_v = state[13:16]

        rel_target = self.TARGET_POS - pos          # 3D, unnormalized
        lidar = self._castLidarRays(pos)            # N_LIDAR_RAYS values in [0,1]

        core = np.hstack([pos, rpy, vel, ang_v, rel_target, lidar]).astype(np.float32)

        buf = np.hstack(
            [self.action_buffer[i][0, :] for i in range(self.ACTION_BUFFER_SIZE)]
        ).astype(np.float32)

        return np.hstack([core, buf]).reshape(1, -1)

    # ---------------------------------------------------------------------------

    def _castLidarRays(self, pos):
        """Cast rays from the drone; return hit fractions (1.0 = no hit in range)."""
        dirs = []
        # 16 horizontal rays, evenly spread 360°
        for k in range(16):
            az = 2 * np.pi * k / 16
            dirs.append([np.cos(az), np.sin(az), 0.0])
        # 8 downward-angled rays (30° below horizontal)
        for k in range(8):
            az = 2 * np.pi * k / 8
            el = -np.pi / 6
            dirs.append([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])

        ray_froms = [pos.tolist()] * len(dirs)
        ray_tos = [(pos + self.LIDAR_RANGE * np.asarray(d)).tolist() for d in dirs]

        hits = p.rayTestBatch(ray_froms, ray_tos, physicsClientId=self.CLIENT)
        return np.array([h[2] for h in hits], dtype=np.float32)  # hit fraction

    # ---------------------------------------------------------------------------

    def _isColliding(self):
        """Return True if the drone is in contact with any obstacle."""
        for oid in self._obstacle_ids:
            if p.getContactPoints(
                bodyA=self.DRONE_IDS[0], bodyB=oid, physicsClientId=self.CLIENT
            ):
                return True
        return False

    # ---------------------------------------------------------------------------

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        dist = np.linalg.norm(self.TARGET_POS - state[0:3])

        # Potential-based progress shaping
        prev = self._prev_dist if self._prev_dist is not None else dist
        reward = (prev - dist) * 10.0
        self._prev_dist = dist

        # Moderate time penalty — encourages speed without destabilizing
        reward -= 0.02

        # Proximity reward — when very close, reward staying close to stop orbiting
        if dist < 0.5:
            reward += 0.5 * (0.5 - dist)

        # Terminal bonuses/penalties
        if dist < self.TARGET_RADIUS:
            # Early completion bonus: more reward for finishing faster
            steps_remaining = max(0, self.EPISODE_LEN_SEC * self.CTRL_FREQ - self.step_counter)
            reward += 20.0 + 10.0 * (steps_remaining / (self.EPISODE_LEN_SEC * self.CTRL_FREQ))
        elif self._isColliding():
            reward -= 10.0

        return float(reward)

    # ---------------------------------------------------------------------------

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        dist = np.linalg.norm(self.TARGET_POS - state[0:3])
        if dist < self.TARGET_RADIUS:
            return True
        if self._isColliding():
            return True
        return False

    # ---------------------------------------------------------------------------

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:10]

        out_of_bounds = (
            abs(pos[0]) > self.ARENA_HALF
            or abs(pos[1]) > self.ARENA_HALF
            or pos[2] > 3.0
            or pos[2] < 0.05
        )
        too_tilted = abs(rpy[0]) > np.pi / 3 or abs(rpy[1]) > np.pi / 3
        timeout = self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC

        return bool(out_of_bounds or too_tilted or timeout)

    # ---------------------------------------------------------------------------

    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        dist = np.linalg.norm(self.TARGET_POS - state[0:3])
        return {
            "dist_to_target": float(dist),
            "colliding": self._isColliding(),
            "success": bool(dist < self.TARGET_RADIUS),
        }
