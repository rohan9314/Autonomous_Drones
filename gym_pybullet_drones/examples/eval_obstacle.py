"""Evaluate and visualise a trained obstacle-avoidance drone policy.

Usage
-----
    # With PyBullet GUI (default)
    python eval_obstacle.py --model results/obstacle_<timestamp>/best_model.zip

    # With prettier Vispy renderer (headless physics, custom 3-D window)
    python eval_obstacle.py --model ... --vispy true

    # Headless (fast, prints stats only)
    python eval_obstacle.py --model ... --gui false --episodes 50

    # Test with more obstacles than the model was trained on
    python eval_obstacle.py --model ... --num_obstacles 6
"""
import time
import argparse

import numpy as np
from stable_baselines3 import PPO

from gym_pybullet_drones.envs.ObstacleAviary import ObstacleAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync, str2bool


def run(model_path: str, gui: bool = True, vispy: bool = False,
        num_episodes: int = 10, num_obstacles: int = 4):
    model = PPO.load(model_path)
    print(f"[INFO] Loaded model from: {model_path}")

    if vispy:
        from stream_viz_obstacle import DroneVizObstacle
        env = ObstacleAviary(
            gui=False,
            obs=ObservationType.KIN,
            act=ActionType.VEL,
            num_obstacles=num_obstacles,
        )
        viz = DroneVizObstacle(env, title=f"Obstacle Avoidance  |  {num_obstacles} obstacles")
        try:
            viz.run(model=model, seed=0, n_episodes=num_episodes)
        finally:
            env.close()
        return

    env = ObstacleAviary(
        gui=gui,
        obs=ObservationType.KIN,
        act=ActionType.VEL,
        num_obstacles=num_obstacles,
    )

    successes = 0
    collisions = 0
    episode_rewards = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep * 17)
        start = time.time()
        done = False
        step = 0
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

            if gui:
                sync(step, start, env.CTRL_TIMESTEP)
            step += 1

        episode_rewards.append(ep_reward)
        success = info["success"]
        collided = info["colliding"]
        successes += int(success)
        collisions += int(collided)

        status = "SUCCESS  " if success else ("COLLISION" if collided else "TIMEOUT  ")
        print(
            f"  Ep {ep+1:3d}/{num_episodes}  [{status}]"
            f"  reward={ep_reward:7.2f}"
            f"  dist={info['dist_to_target']:.3f} m"
            f"  steps={step}"
        )

    env.close()

    print("\n" + "=" * 55)
    print(f"  Success rate:    {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
    print(f"  Collision rate:  {collisions}/{num_episodes} ({100*collisions/num_episodes:.1f}%)")
    print(f"  Mean reward:     {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate obstacle-avoidance drone policy")
    parser.add_argument(
        "--model", required=True, type=str,
        help="Path to best_model.zip or final_model.zip"
    )
    parser.add_argument(
        "--gui", default=True, type=str2bool,
        help="Show PyBullet GUI (default: True)"
    )
    parser.add_argument(
        "--vispy", default=False, type=str2bool,
        help="Use Vispy renderer instead of PyBullet GUI (default: False)"
    )
    parser.add_argument(
        "--episodes", default=10, type=int,
        help="Number of evaluation episodes (default: 10)"
    )
    parser.add_argument(
        "--num_obstacles", default=4, type=int,
        help="Number of box obstacles per episode (default: 4)"
    )
    args = parser.parse_args()

    run(
        model_path=args.model,
        gui=args.gui,
        vispy=args.vispy,
        num_episodes=args.episodes,
        num_obstacles=args.num_obstacles,
    )
