import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class HoverAviary(BaseRLAviary):
    """Single drone with PID position control."""

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=True,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.PID
                 ):

        self.EPISODE_LEN_SEC = 6
        self.TARGET_POS = np.array([0.0, 0.0, 1.0])

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
            act=act
        )

    ############################################################

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)

        # Resample until target is far enough from the drone's start position
        # to avoid immediately terminating on the first step.
        MIN_DIST = 0.3
        while True:
            candidate = np.random.uniform(low=[-1, -1, 0.5], high=[1, 1, 1.5])
            if np.linalg.norm(candidate - self.INIT_XYZS[0]) >= MIN_DIST:
                self.TARGET_POS = candidate
                break

        print("New target:", self.TARGET_POS)
        return obs, info

    ############################################################

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        dist = np.linalg.norm(self.TARGET_POS - state[0:3])
        return -dist

    ############################################################

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        dist = np.linalg.norm(self.TARGET_POS - state[0:3])
        return dist < 0.1

    ############################################################

    def _computeTruncated(self):
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    ############################################################

    def _computeInfo(self):
        return {"target": self.TARGET_POS}


# ============================================================
# RUN PID CONTROL (NO RL)
# ============================================================

if __name__ == "__main__":

    env = HoverAviary(gui=True)
    obs, _ = env.reset()

    for _ in range(3000):
        action = np.array([env.TARGET_POS])   # shape (1, 3) for 1 drone
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()
