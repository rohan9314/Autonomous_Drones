"""Evaluate and visualise a trained ObstacleAviary policy.

Loads a saved PPO model, runs N evaluation episodes per difficulty level,
prints a statistics table, optionally plots 3-D flight trajectories, and
optionally opens the PyBullet GUI so you can watch the drone fly.

Usage
-----
    # auto-find the latest run in results/
    python play_obstacle.py

    # point to a specific model
    python play_obstacle.py --model_path results/obstacle-04.21.2026_10.00.00/best_model.zip

    # watch it fly in the GUI (one episode, difficulty 1)
    python play_obstacle.py --gui true --difficulty 1

    # run 20 quantitative episodes per phase, no GUI
    python play_obstacle.py --n_episodes 20 --gui false
"""

import os
import glob
import time
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless-safe default; switched to TkAgg if GUI requested
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

from stable_baselines3 import PPO

from gym_pybullet_drones.envs.ObstacleAviary import ObstacleAviary, OBSTACLE_HALF_EXTENT
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_OBS          = ObservationType('kin')
DEFAULT_ACT          = ActionType('pid')
DEFAULT_N_EPISODES   = 10
DEFAULT_GUI          = False
DEFAULT_PLOT         = True
DEFAULT_OUTPUT       = 'results'
DEFAULT_MODEL_PATH   = None   # None → auto-find latest run
DEFAULT_DIFFICULTY   = None   # None → evaluate all three difficulties


# ── helpers ────────────────────────────────────────────────────────────────────

def _find_latest_model(results_dir: str) -> str:
    """Return the best_model.zip from the most-recently created obstacle run."""
    pattern = os.path.join(results_dir, "obstacle-*", "best_model.zip")
    candidates = sorted(glob.glob(pattern))          # sorted by name → chronological
    if not candidates:
        raise FileNotFoundError(
            f"No trained obstacle model found under '{results_dir}'. "
            "Run learn_obstacle.py first, or pass --model_path explicitly."
        )
    return candidates[-1]


def _episode_outcome(terminated: bool, truncated: bool, env: ObstacleAviary) -> str:
    """Classify why an episode ended: success, crash, out-of-bounds, or timeout."""
    if terminated:
        return "success"
    if truncated:
        state = env._getDroneStateVector(0)
        # check collision first (most informative)
        import pybullet as p
        for oid in env.obstacle_ids:
            if p.getContactPoints(env.DRONE_IDS[0], oid, physicsClientId=env.CLIENT):
                return "crash"
        if abs(state[0]) > 2.0 or abs(state[1]) > 2.0 or state[2] > 2.5:
            return "out-of-bounds"
        if abs(state[7]) > 0.4 or abs(state[8]) > 0.4:
            return "flip"
    return "timeout"


# ── quantitative evaluation ────────────────────────────────────────────────────

def evaluate_difficulty(model, difficulty: int, n_episodes: int) -> dict:
    """Run n_episodes headlessly and return aggregated statistics.

    Returns a dict with keys:
        mean_reward, std_reward      — episode totals
        success_rate                 — fraction of terminated (reached goal)
        outcome_counts               — {"success": N, "crash": N, ...}
        mean_steps, std_steps        — steps per episode
        mean_dist_final              — mean dist-to-goal when episode ended
    """
    env = ObstacleAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT, difficulty=difficulty)

    rewards, steps_list, final_dists = [], [], []
    outcome_counts = {"success": 0, "crash": 0, "timeout": 0,
                      "out-of-bounds": 0, "flip": 0}

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        ep_reward  = 0.0
        ep_steps   = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps  += 1
            if terminated or truncated:
                outcome = _episode_outcome(terminated, truncated, env)
                outcome_counts[outcome] += 1
                final_dists.append(info["dist_to_goal"])
                break

        rewards.append(ep_reward)
        steps_list.append(ep_steps)

    env.close()

    return {
        "mean_reward":    float(np.mean(rewards)),
        "std_reward":     float(np.std(rewards)),
        "success_rate":   outcome_counts["success"] / n_episodes,
        "outcome_counts": outcome_counts,
        "mean_steps":     float(np.mean(steps_list)),
        "std_steps":      float(np.std(steps_list)),
        "mean_dist_final": float(np.mean(final_dists)),
    }


# ── trajectory collection ──────────────────────────────────────────────────────

def collect_trajectory(model, difficulty: int, seed: int = 0) -> dict:
    """Run one episode and record the drone's XYZ path + per-step rewards.

    Returns a dict:
        positions  — (T, 3) array of [x, y, z]
        rewards    — (T,) array
        outcome    — string ("success" / "crash" / ...)
        obstacle_positions — list of (x,y,z) tuples actually used in the episode
    """
    env = ObstacleAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT, difficulty=difficulty)
    obs, _ = env.reset(seed=seed)

    # grab the real obstacle positions *after* reset so randomised positions
    # (difficulty=2) are captured correctly
    import pybullet as p
    obs_positions = []
    for oid in env.obstacle_ids:
        pos, _ = p.getBasePositionAndOrientation(oid, physicsClientId=env.CLIENT)
        obs_positions.append(pos)

    positions, rewards = [], []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        state = env._getDroneStateVector(0)
        positions.append(state[0:3].copy())
        rewards.append(reward)
        if terminated or truncated:
            outcome = _episode_outcome(terminated, truncated, env)
            break

    env.close()
    return {
        "positions":           np.array(positions),
        "rewards":             np.array(rewards),
        "outcome":             outcome,
        "obstacle_positions":  obs_positions,
    }


# ── 3-D trajectory plot ────────────────────────────────────────────────────────

def plot_trajectory(traj: dict, difficulty: int, save_path: str):
    """Draw the drone's flight path in 3-D with obstacles and goal marked.

    Design note: we colour the path by normalised time (blue=start, red=end)
    so you can see the direction of travel without needing arrows, which clutter
    3-D plots.  Obstacles are drawn as semi-transparent red cubes.
    """
    pos     = traj["positions"]          # (T, 3)
    outcome = traj["outcome"]
    goal    = np.array([1.0, 0.0, 1.0])

    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")

    # --- flight path coloured by time
    T       = len(pos)
    colours = plt.cm.coolwarm(np.linspace(0, 1, T))
    for i in range(T - 1):
        ax.plot(pos[i:i+2, 0], pos[i:i+2, 1], pos[i:i+2, 2],
                color=colours[i], linewidth=1.2, alpha=0.85)

    ax.scatter(*pos[0],  color="blue",  s=60, zorder=5, label="start")
    ax.scatter(*pos[-1], color="red",   s=60, zorder=5, label=f"end ({outcome})")
    ax.scatter(*goal,    color="green", s=120, marker="*", zorder=5, label="goal")

    # --- obstacles as wireframe cubes
    h = OBSTACLE_HALF_EXTENT
    for (ox, oy, oz) in traj["obstacle_positions"]:
        # draw 12 edges of the cube
        corners = np.array([
            [ox-h, oy-h, oz-h], [ox+h, oy-h, oz-h],
            [ox+h, oy+h, oz-h], [ox-h, oy+h, oz-h],
            [ox-h, oy-h, oz+h], [ox+h, oy-h, oz+h],
            [ox+h, oy+h, oz+h], [ox-h, oy+h, oz+h],
        ])
        edges = [(0,1),(1,2),(2,3),(3,0),   # bottom face
                 (4,5),(5,6),(6,7),(7,4),   # top face
                 (0,4),(1,5),(2,6),(3,7)]   # verticals
        for a, b in edges:
            ax.plot(*zip(corners[a], corners[b]), color="darkred",
                    linewidth=1.5, alpha=0.6)

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(f"Obstacle Avoidance Trajectory  |  difficulty={difficulty}  |  {outcome}")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[INFO] Trajectory plot saved to {save_path}")
    plt.close(fig)


# ── GUI visual episode ─────────────────────────────────────────────────────────

def run_visual_episode(model, difficulty: int):
    """Open the PyBullet GUI and run one episode in real-time so you can watch.

    sync() throttles the loop to wall-clock time so the simulation doesn't
    blur past invisibly — it runs at the same speed as real life.
    """
    env = ObstacleAviary(
        obs=DEFAULT_OBS, act=DEFAULT_ACT,
        difficulty=difficulty, gui=True
    )
    obs, _ = env.reset(seed=0)

    print(f"\n[GUI] Watching one episode at difficulty={difficulty} ...")
    print(f"{'Step':>5}  {'x':>7} {'y':>7} {'z':>7}  {'dist':>6}  {'reward':>8}  outcome")

    start_t    = time.time()
    ep_reward  = 0.0
    max_steps  = (env.EPISODE_LEN_SEC + 2) * env.CTRL_FREQ

    for i in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward

        state = env._getDroneStateVector(0)
        print(f"{i:5d}  {state[0]:7.3f} {state[1]:7.3f} {state[2]:7.3f}"
              f"  {info['dist_to_goal']:6.3f}  {reward:8.3f}", end="")

        if terminated or truncated:
            outcome = _episode_outcome(terminated, truncated, env)
            print(f"  ← {outcome}  (total={ep_reward:.2f})")
            break
        else:
            print()

        sync(i, start_t, env.CTRL_TIMESTEP)

    env.close()


# ── stats table ────────────────────────────────────────────────────────────────

def print_stats_table(all_stats: dict):
    """Pretty-print per-difficulty statistics."""
    print("\n" + "═" * 72)
    print(f"  {'Difficulty':<12} {'Mean Reward':>12} {'Success%':>9} "
          f"{'Crash%':>7} {'Timeout%':>9} {'MeanDist':>9}")
    print("─" * 72)
    for diff, s in all_stats.items():
        n  = sum(s["outcome_counts"].values())
        sr = 100 * s["success_rate"]
        cr = 100 * s["outcome_counts"]["crash"]   / n
        tr = 100 * s["outcome_counts"]["timeout"] / n
        print(f"  {diff:<12} {s['mean_reward']:>+12.2f} {sr:>8.1f}% "
              f"{cr:>6.1f}% {tr:>8.1f}% {s['mean_dist_final']:>9.3f}m")
    print("═" * 72 + "\n")


# ── main ───────────────────────────────────────────────────────────────────────

def run(
    model_path=DEFAULT_MODEL_PATH,
    difficulty=DEFAULT_DIFFICULTY,
    n_episodes=DEFAULT_N_EPISODES,
    gui=DEFAULT_GUI,
    plot=DEFAULT_PLOT,
    output_folder=DEFAULT_OUTPUT,
):
    # ── locate model ──────────────────────────────────────────────────────────
    if model_path is None:
        model_path = _find_latest_model(output_folder)
    print(f"[INFO] Loading model: {model_path}")
    model = PPO.load(model_path)

    # derive the save directory from the model path for storing plots
    save_dir = os.path.dirname(model_path)

    # ── decide which difficulties to evaluate ─────────────────────────────────
    # If the user passed --difficulty, evaluate only that level.
    # Otherwise evaluate all three so we get a full comparison table.
    difficulties = [difficulty] if difficulty is not None else [0, 1, 2]

    # ── quantitative evaluation ───────────────────────────────────────────────
    all_stats = {}
    for diff in difficulties:
        print(f"[INFO] Evaluating difficulty={diff} over {n_episodes} episodes ...")
        stats = evaluate_difficulty(model, diff, n_episodes)
        all_stats[diff] = stats
        counts = stats["outcome_counts"]
        print(f"       mean_reward={stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}  "
              f"success={counts['success']}  crash={counts['crash']}  "
              f"timeout={counts['timeout']}  oob={counts['out-of-bounds']}")

    print_stats_table(all_stats)

    # ── trajectory plots ──────────────────────────────────────────────────────
    if plot:
        for diff in difficulties:
            print(f"[INFO] Collecting trajectory for plot (difficulty={diff}) ...")
            traj      = collect_trajectory(model, diff, seed=0)
            plot_path = os.path.join(save_dir, f"trajectory_diff{diff}.png")
            plot_trajectory(traj, diff, plot_path)

    # ── GUI episode ───────────────────────────────────────────────────────────
    if gui:
        diff_for_gui = difficulty if difficulty is not None else max(difficulties)
        run_visual_episode(model, diff_for_gui)

    return all_stats


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained ObstacleAviary policy.")
    parser.add_argument("--model_path",    default=DEFAULT_MODEL_PATH,  type=str,      metavar="")
    parser.add_argument("--difficulty",    default=DEFAULT_DIFFICULTY,  type=int,      metavar="")
    parser.add_argument("--n_episodes",    default=DEFAULT_N_EPISODES,  type=int,      metavar="")
    parser.add_argument("--gui",           default=DEFAULT_GUI,         type=str2bool, metavar="")
    parser.add_argument("--plot",          default=DEFAULT_PLOT,        type=str2bool, metavar="")
    parser.add_argument("--output_folder", default=DEFAULT_OUTPUT,      type=str,      metavar="")
    args = parser.parse_args()
    run(**vars(args))
