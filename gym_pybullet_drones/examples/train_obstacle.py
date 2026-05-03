"""Train a single drone to navigate to a target while avoiding obstacles.

Uses PPO (Proximal Policy Optimization) from Stable-Baselines3.
Observations include simulated LiDAR so the agent can sense obstacles.

Usage
-----
    # Quick test (100k steps, 1 obstacle)
    python train_obstacle.py --timesteps 100000 --num_obstacles 1

    # Full training run
    python train_obstacle.py --timesteps 5000000 --num_obstacles 4 --n_envs 4

    # Headless (no GUI) — much faster, recommended for training
    python train_obstacle.py --gui false

After training, the best checkpoint is saved to results/obstacle_<timestamp>/best_model.zip.
Use eval_obstacle.py to visualise the learned policy.
"""
import os
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)

from gym_pybullet_drones.envs.ObstacleAviary import ObstacleAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import str2bool

DEFAULT_OUTPUT = "gym_pybullet_drones/examples/results"
DEFAULT_TIMESTEPS = 5_000_000
DEFAULT_N_ENVS = 4
DEFAULT_N_OBS = 4

# n_envs * n_steps_per_env (per PPO update := 'run' := 'batch'), scaled to reach total_timesteps

def run(
    output_folder=DEFAULT_OUTPUT,
    total_timesteps=DEFAULT_TIMESTEPS,
    n_envs=DEFAULT_N_ENVS,
    num_obstacles=DEFAULT_N_OBS,
    gui=False,
):
    timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    save_dir = os.path.join(output_folder, f"obstacle_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    env_kwargs = dict(
        obs=ObservationType.KIN,
        act=ActionType.VEL,
        num_obstacles=num_obstacles,
    )

    train_env = make_vec_env(
        ObstacleAviary,
        env_kwargs=env_kwargs,
        n_envs=n_envs,
        seed=0,
    )
    eval_env = ObstacleAviary(**env_kwargs)

    print("[INFO] Action space:      ", train_env.action_space)
    print("[INFO] Observation space: ", train_env.observation_space)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,          # entropy bonus encourages exploration
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log=os.path.join(save_dir, "tb"),
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1),
        save_path=save_dir,
        name_prefix="ckpt",
        verbose=0,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=max(20_000 // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList([checkpoint_cb, eval_cb]),
        log_interval=20,
    )

    model.save(os.path.join(save_dir, "final_model"))
    train_env.close()
    eval_env.close()
    print(f"\n[INFO] Training complete. Files saved to: {save_dir}")

    # Plot and save training curve
    eval_path = os.path.join(save_dir, "evaluations.npz")
    if os.path.isfile(eval_path):
        with np.load(eval_path) as data:
            ts = data["timesteps"]
            rewards = data["results"][:, 0]

        plt.figure(figsize=(10, 4))
        plt.plot(ts, rewards, marker="o", markersize=3, linewidth=1)
        plt.xlabel("Timesteps")
        plt.ylabel("Mean Episode Reward")
        plt.title(f"Obstacle Navigation — {num_obstacles} obstacle(s)")
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        curve_path = os.path.join(save_dir, "training_curve.png")
        plt.savefig(curve_path, dpi=150)
        print(f"[INFO] Training curve saved to: {curve_path}")
        plt.show()

    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train drone obstacle-avoidance policy with PPO"
    )
    parser.add_argument(
        "--timesteps", default=DEFAULT_TIMESTEPS, type=int,
        help="Total training timesteps (default: 5,000,000)"
    )
    parser.add_argument(
        "--n_envs", default=DEFAULT_N_ENVS, type=int,
        help="Number of parallel training environments (default: 4)"
    )
    parser.add_argument(
        "--num_obstacles", default=DEFAULT_N_OBS, type=int,
        help="Number of random box obstacles per episode (default: 4)"
    )
    parser.add_argument(
        "--output_folder", default=DEFAULT_OUTPUT, type=str,
        help="Directory to save models and logs (default: results/)"
    )
    parser.add_argument(
        "--gui", default=False, type=str2bool,
        help="Show PyBullet GUI during training — very slow, not recommended (default: False)"
    )
    args = parser.parse_args()

    run(
        output_folder=args.output_folder,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        num_obstacles=args.num_obstacles,
        gui=args.gui,
    )