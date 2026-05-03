import numpy as np
import pybullet as p
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class TargetAviary(BaseRLAviary):
    """Single agent RL problem: fly to target while avoiding obstacles."""

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 TARGET_POS=None,
                 OBSTACLE_POSITIONS=None
                 ):
        self.TARGET_POS = TARGET_POS if TARGET_POS is not None else np.array([0.5, 0.5, 0.5])
        self.OBSTACLE_POSITIONS = OBSTACLE_POSITIONS if OBSTACLE_POSITIONS is not None else np.array([
            [0.25, 0.25, 0.5],
            [0.0,  0.5,  0.3],
        ])
        self.OBSTACLE_RADIUS = 0.1
        self.EPISODE_LEN_SEC = 8
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )
        self._addObstacles()

    def _addObstacles(self):
        for pos in self.OBSTACLE_POSITIONS:
            visual = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=self.OBSTACLE_RADIUS,
                rgbaColor=[1, 0, 0, 0.7],
                physicsClientId=self.CLIENT
            )
            collision = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=self.OBSTACLE_RADIUS,
                physicsClientId=self.CLIENT
            )
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=pos,
                physicsClientId=self.CLIENT
            )

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        distance = np.linalg.norm(self.TARGET_POS - pos)
        reward = np.exp(-distance)
        for obs_pos in self.OBSTACLE_POSITIONS:
            obs_dist = np.linalg.norm(obs_pos - pos)
            reward -= np.exp(-obs_dist / self.OBSTACLE_RADIUS)
        return max(0.0, reward)

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        return bool(np.linalg.norm(self.TARGET_POS - state[0:3]) < 0.0001)

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        margin = 1.5
        if (any(abs(state[i]) > margin + abs(self.TARGET_POS[i]) for i in range(3))
                or abs(state[7]) > .4 or abs(state[8]) > .4):
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeInfo(self):
        return {"answer": 42}

    def reset(self, seed=None, options={}):
        rng = np.random.default_rng(seed)
        # self.INIT_XYZS = np.array([[
        #     rng.uniform(-0.5, 0.5),
        #     rng.uniform(-0.5, 0.5),
        #     rng.uniform(0.1, 0.5)
        # ]])
        self.INIT_XYZS = np.array([[0., 0., 0.1]])
        result = super().reset(seed=seed, options=options)
        self._addObstacles()
        return result

    def _computeObs(self):
        obs = super()._computeObs()
        state = self._getDroneStateVector(0)
        rel_target = (self.TARGET_POS - state[0:3]).reshape(1, 3)
        rel_obstacles = (self.OBSTACLE_POSITIONS - state[0:3]).reshape(1, -1)
        return np.hstack([obs, rel_target, rel_obstacles]).astype('float32')

    def _observationSpace(self):
        obs_space = super()._observationSpace()
        lo, hi = -np.inf, np.inf
        n_extra = 3 + 3 * len(self.OBSTACLE_POSITIONS)
        new_low  = np.hstack([obs_space.low,  np.full((1, n_extra), lo)])
        new_high = np.hstack([obs_space.high, np.full((1, n_extra), hi)])
        return spaces.Box(low=new_low, high=new_high, dtype=np.float32)
