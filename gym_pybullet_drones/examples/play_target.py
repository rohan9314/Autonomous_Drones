import os
import time
import argparse
import numpy as np
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.TargetAviary import TargetAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.Logger import Logger

DEFAULT_MODEL_PATH = "results/best_model.zip"
DEFAULT_GUI = True
DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('pid')

TARGET_POS = np.array([0.5, 0.5, 0.5])
OBSTACLE_POSITIONS = np.array([
    [0.25, 0.25, 0.5],
    [0.0,  0.5,  0.3],
])

def play(model_path=DEFAULT_MODEL_PATH, gui=DEFAULT_GUI):
    if not os.path.isfile(model_path):
        print(f"[ERROR] Model file not found at: {model_path}")
        return

    model = PPO.load(model_path)
    print(f"[INFO] Loaded model from {model_path}")

    env = TargetAviary(gui=gui, obs=DEFAULT_OBS, act=DEFAULT_ACT,
                       TARGET_POS=TARGET_POS,
                       OBSTACLE_POSITIONS=OBSTACLE_POSITIONS)

    logger = Logger(logging_freq_hz=int(env.CTRL_FREQ),
                    num_drones=1,
                    output_folder="logs_playback/",
                    colab=False)

    obs, _ = env.reset(seed=42, options={})
    start = time.time()

    for i in range((env.EPISODE_LEN_SEC+2)*env.CTRL_FREQ):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        obs2 = obs.squeeze()
        act2 = action.squeeze()

        if DEFAULT_OBS == ObservationType.KIN:
            logger.log(drone=0,
                       timestamp=i/env.CTRL_FREQ,
                       state=np.hstack([obs2[0:3], np.zeros(4), obs2[3:15], act2]),
                       control=np.zeros(12))

        env.render()
        sync(i, start, env.CTRL_TIMESTEP)
        if terminated or truncated:
            break

    env.close()
    logger.plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play trained TargetAviary policy")
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument('--gui', type=str2bool, default=DEFAULT_GUI)
    args = parser.parse_args()
    play(**vars(args))
