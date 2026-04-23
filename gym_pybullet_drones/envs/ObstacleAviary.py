import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


OBSTACLE_HALF_EXTENT = 0.10   # 20 cm cube side length
MAX_OBSTACLES        = 2      # obs vector always padded to this many obstacles

# Base obstacle positions per phase.
# Direct path: (0,0,0) → (1,0,1), parameterised as t*(1,0,1) for t ∈ [0,1].
#   Phase 1 — one obstacle 0.3 m off the path in Y: easy first encounter.
#   Phase 2 — one obstacle dead-centre on the path: forced detour.
#   Phase 3 — two obstacles in the corridor (original hard layout).
PHASE_OBSTACLES = {
    1: [(0.50,  0.10, 0.50)],
    2: [(0.50,  0.00, 0.50)],
    3: [(0.33,  0.00, 0.50), (0.66, 0.15, 0.80)],
}


class ObstacleAviary(BaseRLAviary):
    """Single-agent RL task: fly from origin to [1,0,1] while avoiding obstacles.

    Four-phase curriculum (controlled by `difficulty`):
      0 — no obstacles, open corridor
      1 — one obstacle placed off the direct flight path (easy avoidance)
      2 — one obstacle placed directly on the flight path (forced detour)
      3 — two obstacles in the corridor, randomised each episode

    Observation vector layout (total 66 dims with PID / ctrl_freq=30):
      [0:12]   kinematic state  (pos, rpy, vel, ang_v)
      [12:57]  action buffer    (last 15 actions × 3 dims each)
      [57:63]  obstacle slots   (MAX_OBSTACLES × 3 relative XYZ, padded)
      [63:66]  goal vector      (TARGET_POS − drone_pos)

    Including the goal vector means the policy has direct access to the
    direction it needs to fly — critical for PID mode where the policy
    outputs 3-D target positions.
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
                 difficulty: int          = 0,
                 ):
        self.TARGET_POS      = np.array([1.0, 0.0, 1.0])
        self.EPISODE_LEN_SEC = 8
        self.difficulty      = difficulty
        self.prev_dist       = 0.0   # seeded in reset(); used by progress reward
        self.obstacle_ids    = []    # populated by _addObstacles() on every reset()

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
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        state = self._getDroneStateVector(0)
        self.prev_dist = float(np.linalg.norm(self.TARGET_POS - state[0:3]))
        return obs, info

    # ------------------------------------------------------------------
    # Action preprocessing
    # ------------------------------------------------------------------

    def _preprocessAction(self, action):
        """Override PID waypoint step size for physically safe navigation.

        BaseRLAviary uses step_size=1, meaning the PID controller receives a
        waypoint up to 1 m away per control step (33 ms at 30 Hz).  Covering
        1 m in 33 ms requires ~30 m/s — far beyond the Crazyflie's ~8 m/s
        limit — causing extreme tilt and immediate truncation.

        step_size=0.10 limits the intermediate waypoint to 10 cm per step,
        equivalent to a maximum commanded speed of ~3 m/s.  The policy
        still outputs an absolute target position in [-1,1]³; we just cap
        how far the drone attempts to move in a single timestep.
        """
        self.action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES, 4))
        for k in range(action.shape[0]):
            target = action[k, :]
            state  = self._getDroneStateVector(k)
            next_pos = self._calculateNextStep(
                current_position=state[0:3],
                destination=target,
                step_size=0.10,           # 10 cm/step ≈ 3 m/s max commanded speed
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
        """Spawn static box obstacles for the current difficulty level.

        Called by BaseAviary._housekeeping() on every reset(), so the world
        is rebuilt from scratch each episode.  Changing self.difficulty before
        a natural episode reset is all that is needed to advance the curriculum.
        """
        self.obstacle_ids = []
        if self.difficulty == 0:
            return

        base = PHASE_OBSTACLES.get(self.difficulty, PHASE_OBSTACLES[3])

        if self.difficulty < 3:
            positions = list(base)
        else:
            rng = np.random.default_rng(self.np_random.integers(0, 2**31))
            positions = [
                (x + rng.uniform(-0.10, 0.10),
                 y + rng.uniform(-0.15, 0.15),
                 z)
                for x, y, z in base
            ]

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        for (ox, oy, oz) in positions:
            col = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[OBSTACLE_HALF_EXTENT] * 3,
                physicsClientId=self.CLIENT,
            )
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[OBSTACLE_HALF_EXTENT] * 3,
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

    # ------------------------------------------------------------------
    # Observation space
    # ------------------------------------------------------------------

    def _observationSpace(self):
        """Extends the kinematic+buffer base space with obstacle slots and goal.

        Extra features appended (9 total):
          MAX_OBSTACLES × 3  relative obstacle positions  (padded with [10,10,10])
          1 × 3              goal relative position       (TARGET_POS − drone_pos)

        Why include the goal vector:
          With PID the policy outputs a 3-D target position each step.  Without
          knowing where the goal is the policy must guess a fixed point in
          [-1,1]³ that happens to be [1,0,1] — learnable but slow.  Providing
          (goal − drone) directly gives the policy the direction to fly from
          step one, cutting early training time significantly.
        """
        base_space = super()._observationSpace()
        n_extra    = 3 * MAX_OBSTACLES + 3   # obstacle slots + goal vector

        low  = np.hstack([base_space.low,
                          np.full((1, n_extra), -np.inf, dtype=np.float32)])
        high = np.hstack([base_space.high,
                          np.full((1, n_extra),  np.inf, dtype=np.float32)])
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _computeObs(self):
        """Kinematic+buffer base | obstacle relative positions | goal vector."""
        base_obs  = super()._computeObs()                    # (1, base_dim)
        drone_pos = self._getDroneStateVector(0)[0:3]

        # ── obstacle slots (padded to MAX_OBSTACLES) ──────────────────
        obs_features = []
        for i in range(MAX_OBSTACLES):
            if i < len(self.obstacle_ids):
                pos, _ = p.getBasePositionAndOrientation(
                    self.obstacle_ids[i], physicsClientId=self.CLIENT
                )
                obs_features.extend(np.array(pos) - drone_pos)
            else:
                obs_features.extend([0.0, 0.0, 0.0])      # absent-obstacle pad

        # ── goal vector ───────────────────────────────────────────────
        goal_vec = self.TARGET_POS - drone_pos
        obs_features.extend(goal_vec)

        extra = np.array(obs_features, dtype=np.float32).reshape(1, -1)
        return np.hstack([base_obs, extra]).astype(np.float32)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _computeReward(self):
        """Progress reward: positive when closing distance to goal.

        r_progress = dist_prev − dist_curr   (positive → approaching goal)
        r_crash    = −3 on contact           (keeps obstacles genuinely costly)

        Hovering yields exactly 0; crashing is only −3 plus lost future
        progress — no longer worth doing deliberately.
        """
        state     = self._getDroneStateVector(0)
        curr_dist = float(np.linalg.norm(self.TARGET_POS - state[0:3]))

        r_progress     = self.prev_dist - curr_dist
        self.prev_dist = curr_dist

        r_crash = 0.0
        for oid in self.obstacle_ids:
            if p.getContactPoints(self.DRONE_IDS[0], oid, physicsClientId=self.CLIENT):
                r_crash = -3.0
                break

        return r_progress + r_crash

    # ------------------------------------------------------------------
    # Termination / truncation
    # ------------------------------------------------------------------

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS - state[0:3]) < 0.05:
            for oid in self.obstacle_ids:
                if p.getContactPoints(self.DRONE_IDS[0], oid, physicsClientId=self.CLIENT):
                    return False
            return True
        return False

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        if abs(state[0]) > 2.0 or abs(state[1]) > 2.0 or state[2] > 2.5:
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
        return {"dist_to_goal": dist, "difficulty": self.difficulty}
