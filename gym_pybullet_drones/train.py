import numpy as np
from gym_pybullet_drones.envs.TargetAviary import TargetAviary
from gym_pybullet_drones.utils.enums import ActionType
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.HoverAviary import HoverAviary

    # in TargetAviary.py, modify reward function to become reaching target
    # do something similar to 'ObstacleAviary.py' to define 'obstacles' and modify the reward function to avoid those

if __name__ == "__main__":
    env = HoverAviary(gui=True)
    model = PPO.load("results/save-04.25.2026_15.02.54/best_model.zip")

    obs, _ = env.reset()
    for _ in range(3000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    # env = TargetAviary(gui=True, act=ActionType.PID)
    # obs, _ = env.reset()
    # for _ in range(3000):
    #     # PID target position command
    #     action = np.array([env.TARGET_POS])   # shape (1, 3)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     if terminated or truncated:
    #         obs, _ = env.reset()