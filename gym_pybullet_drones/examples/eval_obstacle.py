"""Evaluate and visualise a trained obstacle-avoidance drone policy.

Usage
-----
    # With GUI (visual)
    python eval_obstacle.py --model results/obstacle_<timestamp>/best_model.zip

    # Headless (fast, prints stats only)
    python eval_obstacle.py --model results/obstacle_<timestamp>/best_model.zip --gui false --episodes 50

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


def run(model_path: str, gui: bool = True, num_episodes: int = 2, num_obstacles: int = 4, log_path: str = None):
    model = PPO.load(model_path)

    env = ObstacleAviary(
        gui=gui,
        obs=ObservationType.KIN,
        act=ActionType.VEL,
        num_obstacles=num_obstacles,
    )

    successes = 0
    collisions = 0
    episode_rewards = []
    success_efficiencies = []
    lines = [f"Loaded model from: {model_path}\n"]

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
        if success:
            success_efficiencies.append(info["efficiency"])

        status = "SUCCESS  " if success else ("COLLISION" if collided else "TIMEOUT  ")
        line = (
            f"  Ep {ep+1:3d}/{num_episodes}  [{status}]"
            f"  episode reward={ep_reward:7.2f}"
            f"  final dist to target={info['dist_to_target']:.3f} m"
            f"  efficiency (average velocity)={info['efficiency']:.3f} m/s"
            f"  steps taken={step}"
        )
        lines.append(line)

    env.close()

    mean_eff = np.mean(success_efficiencies) if success_efficiencies else float("nan")
    summary = (
        f"\n{'=' * 55}\n"
        f"Statistics across {num_episodes} episodes:\n"
        f"  Success rate:    {successes}/{num_episodes}\n"
        f"  Collision rate:  {collisions}/{num_episodes}\n"
        f"  Episode reward:  {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}\n"
        f"  Mean efficiency (successful episodes): {mean_eff:.4f} m/s\n"
        f"{'=' * 55}"
    )
    lines.append(summary)

    output = "\n".join(lines)
    print(output)

    if log_path:
        with open(log_path, "w") as f:
            f.write(output + "\n")
        print(f"[INFO] Results saved to {log_path}")


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
        "--episodes", default=2, type=int,
        help="Number of evaluation episodes (default: 2)"
    )
    parser.add_argument(
        "--num_obstacles", default=4, type=int,
        help="Number of box obstacles per episode (default: 4)"
    )
    parser.add_argument(
        "--log", default=None, type=str,
        help="Optional path to save results to a text file"
    )
    args = parser.parse_args()

    run(
        model_path=args.model,
        gui=args.gui,
        num_episodes=args.episodes,
        num_obstacles=args.num_obstacles,
        log_path=args.log,
    )