"""Evaluate and visualize a trained ObstacleAviaryV3 (multi-waypoint) model.

Usage:
    python play_obstacle_v3.py                              # auto-find latest v3 model
    python play_obstacle_v3.py --model_path results/obstacle_v3-.../best_model.zip
    python play_obstacle_v3.py --difficulty 2 --n_episodes 20 --gui false
    python play_obstacle_v3.py --gui true --difficulty 1 --seed 0
"""

import os
import argparse
from glob import glob
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from gym_pybullet_drones.envs.ObstacleAviaryV3 import ObstacleAviaryV3, DIFFICULTY_CONFIG
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync, str2bool


DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('pid')


def find_latest_model(base_dir="results"):
    """Find the most recent best_model.zip in any obstacle_v3-* results directory."""
    candidates = sorted(
        glob(os.path.join(base_dir, "obstacle_v3-*", "best_model.zip")),
        key=os.path.getmtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    # Fall back to any best_model
    candidates = sorted(
        glob(os.path.join(base_dir, "obstacle_v3-*", "final_model.zip")),
        key=os.path.getmtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def run_evaluation(model, difficulty, n_episodes, gui, seed):
    """Run n_episodes and return per-episode stats."""
    env = ObstacleAviaryV3(
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
        difficulty=difficulty,
        gui=gui,
    )
    n_wp = DIFFICULTY_CONFIG.get(difficulty, (3, 0, 0))[0]

    fracs    = []
    successes = []
    crashes  = []
    timeouts = []
    wp_counts = []
    trajectories = []   # list of xyz arrays

    for ep in range(n_episodes):
        ep_seed = seed + ep if seed is not None else None
        obs, info = env.reset(seed=ep_seed)
        traj     = []
        ep_frac  = 0.0
        ep_done  = False
        ep_crash = False
        ep_success = False
        max_steps = int((env.EPISODE_LEN_SEC + 4) * env.CTRL_FREQ)

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            state = env._getDroneStateVector(0)
            traj.append(state[0:3].copy())

            if gui:
                sync(step, 0, env.CTRL_TIMESTEP)

            if terminated or truncated:
                ep_frac    = info.get("fraction_completed", 0.0)
                ep_success = info.get("success", False)
                ep_crash   = (not ep_success and truncated
                              and info.get("dist_to_current_wp", 999) > 0.1)
                ep_done    = True
                break

        fracs.append(ep_frac)
        successes.append(ep_success)
        crashes.append(ep_crash and not ep_success)
        timeouts.append(ep_done and not ep_success and not (ep_crash and not ep_success))
        wp_counts.append(info.get("waypoints_completed", 0))
        trajectories.append(np.array(traj))

    env.close()
    return fracs, successes, crashes, timeouts, wp_counts, trajectories, n_wp


def plot_trajectories(trajectories, waypoints_sample, difficulty, save_dir, n_plot=6):
    """Plot XY and XZ projections of up to n_plot trajectories."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, min(n_plot, len(trajectories))))

    for i, (traj, c) in enumerate(zip(trajectories[:n_plot], colors)):
        axes[0].plot(traj[:, 0], traj[:, 1], alpha=0.7, linewidth=1.2, color=c)
        axes[1].plot(traj[:, 0], traj[:, 2], alpha=0.7, linewidth=1.2, color=c)

    axes[0].scatter([0], [0], color='lime', s=80, zorder=5, label='Start')
    axes[1].scatter([0], [1], color='lime', s=80, zorder=5, label='Start')

    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].set_title(f"XY trajectories — difficulty {difficulty}")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.4)
    axes[0].set_aspect('equal')

    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Z (m)")
    axes[1].set_title(f"XZ trajectories — difficulty {difficulty}")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.4)
    axes[1].set_aspect('equal')

    fig.tight_layout()
    path = os.path.join(save_dir, f"trajectories_v3_diff{difficulty}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Trajectory plot saved to {path}")


def run(
    model_path=None,
    difficulty=1,
    n_episodes=20,
    gui=False,
    seed=0,
    output_folder="results",
    plot=True,
):
    # ── find model ────────────────────────────────────────────────────────────
    if model_path is None:
        model_path = find_latest_model(output_folder)
        if model_path is None:
            print("[ERROR] No trained V3 model found. Run learn_obstacle_v3.py first.")
            return
        print(f"[INFO] Auto-selected model: {model_path}")
    else:
        print(f"[INFO] Using model: {model_path}")

    run_dir = os.path.dirname(model_path)

    # ── load model ────────────────────────────────────────────────────────────
    model = PPO.load(model_path)
    print(f"[INFO] Model loaded. Evaluating at difficulty={difficulty}, {n_episodes} episodes.")

    n_wp = DIFFICULTY_CONFIG.get(difficulty, (3, 0, 0))[0]
    print(f"[INFO] Difficulty {difficulty}: {n_wp} waypoints, "
          f"{DIFFICULTY_CONFIG.get(difficulty, (0,0,0))[1]}–"
          f"{DIFFICULTY_CONFIG.get(difficulty, (0,0,0))[2]} obstacles")

    # ── evaluate ──────────────────────────────────────────────────────────────
    fracs, successes, crashes, timeouts, wp_counts, trajectories, n_wp = run_evaluation(
        model, difficulty, n_episodes, gui, seed
    )

    mean_frac    = float(np.mean(fracs))
    success_rate = float(np.mean(successes))
    crash_rate   = float(np.mean(crashes))
    timeout_rate = float(np.mean(timeouts))
    mean_wps     = float(np.mean(wp_counts))

    print(f"\n{'='*55}")
    print(f"  V3 Eval — difficulty={difficulty}  ({n_episodes} episodes)")
    print(f"{'='*55}")
    print(f"  Mean fraction completed : {mean_frac:.1%}")
    print(f"  Full success rate       : {success_rate:.1%}  ({int(sum(successes))}/{n_episodes})")
    print(f"  Crash rate              : {crash_rate:.1%}")
    print(f"  Timeout rate            : {timeout_rate:.1%}")
    print(f"  Mean waypoints reached  : {mean_wps:.1f} / {n_wp}")
    print(f"{'='*55}\n")

    if plot:
        plot_trajectories(trajectories, None, difficulty, run_dir)

    # ── per-difficulty sweep ───────────────────────────────────────────────────
    print("[INFO] Running 5-episode sweep across all difficulty levels ...")
    print(f"\n{'Diff':>5}  {'N_WP':>5}  {'Frac%':>8}  {'Success':>8}  {'Crash':>8}")
    print("-" * 45)
    for d in range(0, 5):
        fr, su, cr, to, wc, _, _ = run_evaluation(model, d, 5, False, seed)
        print(f"  {d:>3d}     {DIFFICULTY_CONFIG[d][0]:>3d}    {np.mean(fr)*100:6.1f}%   "
              f"{np.mean(su)*100:6.1f}%   {np.mean(cr)*100:6.1f}%")
    print()

    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ObstacleAviaryV3 PPO model."
    )
    parser.add_argument("--model_path",    default=None,   type=str,      metavar="")
    parser.add_argument("--difficulty",    default=1,      type=int,      metavar="")
    parser.add_argument("--n_episodes",    default=20,     type=int,      metavar="")
    parser.add_argument("--gui",           default=False,  type=str2bool, metavar="")
    parser.add_argument("--seed",          default=0,      type=int,      metavar="")
    parser.add_argument("--output_folder", default="results", type=str,   metavar="")
    parser.add_argument("--plot",          default=True,   type=str2bool, metavar="")
    args = parser.parse_args()
    run(**vars(args))
