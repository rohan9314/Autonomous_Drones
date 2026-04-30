"""Evaluate and demo a trained ObstacleAviaryV2 policy.

Usage
-----
    # headless quantitative eval across all difficulties (auto-finds latest model)
    python play_obstacle_v2.py

    # watch it fly in the PyBullet GUI at difficulty 2
    python play_obstacle_v2.py --gui true --difficulty 2

    # watch with the Vispy 3-D visualiser (prettier, headless physics)
    python play_obstacle_v2.py --vispy true --difficulty 2

    # run 20 episodes per difficulty, no GUI, save trajectory plots
    python play_obstacle_v2.py --n_episodes 20

    # point to a specific model
    python play_obstacle_v2.py --model_path results/obstacle_v2-04.29.2026_10.39.08/best_model.zip
"""

import os
import glob
import time
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gym_pybullet_drones.envs.ObstacleAviaryV2 import (
    ObstacleAviaryV2,
    DIFFICULTY_OBSTACLE_COUNTS,
)
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_OBS         = ObservationType("kin")
DEFAULT_ACT         = ActionType("pid")
DEFAULT_N_EPISODES  = 10
DEFAULT_GUI         = False
DEFAULT_VISPY       = False
DEFAULT_PLOT        = True
DEFAULT_OUTPUT      = "results"
DEFAULT_MODEL_PATH  = None
DEFAULT_DIFFICULTY  = None   # None → evaluate all difficulties


# ── model / vecnorm loading ────────────────────────────────────────────────────

def _find_latest_model(results_dir: str) -> str:
    candidates = sorted(glob.glob(os.path.join(results_dir, "obstacle_v2-*", "best_model.zip")))
    if not candidates:
        raise FileNotFoundError(
            f"No v2 model found under '{results_dir}'. "
            "Train one with learn_obstacle_v2.py first, or pass --model_path."
        )
    return candidates[-1]


def _load_model_and_vecnorm(model_path: str):
    """Return (PPO model, vec_normalize_path or None)."""
    model = PPO.load(model_path)
    vec_norm_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
    if not os.path.exists(vec_norm_path):
        vec_norm_path = None
    return model, vec_norm_path


def _make_venv(difficulty: int, gui: bool, vec_norm_path):
    """Wrap ObstacleAviaryV2 in DummyVecEnv + VecNormalize for proper inference."""
    def _make():
        return ObstacleAviaryV2(obs=DEFAULT_OBS, act=DEFAULT_ACT,
                                difficulty=difficulty, gui=gui)
    venv = DummyVecEnv([_make])
    if vec_norm_path:
        venv = VecNormalize.load(vec_norm_path, venv)
        venv.training = False
        venv.norm_reward = False
    return venv


# ── obstacle geometry helpers ──────────────────────────────────────────────────

def _get_obstacle_half_extent(oid: int, client_id: int) -> float:
    """Query PyBullet for actual obstacle size (handles boxes and cylinders)."""
    import pybullet as p
    data = p.getCollisionShapeData(oid, -1, physicsClientId=client_id)
    if not data:
        return 0.15
    shape_type = data[0][2]
    dims = data[0][3]
    if shape_type == p.GEOM_BOX:
        return float(dims[0])       # half-extent
    elif shape_type == p.GEOM_CYLINDER:
        return float(dims[1])       # radius
    return 0.15


# ── outcome classification ─────────────────────────────────────────────────────

def _episode_outcome(terminated: bool, truncated: bool, env: ObstacleAviaryV2) -> str:
    if terminated:
        return "success"
    if truncated:
        import pybullet as p
        for oid in env.obstacle_ids:
            if p.getContactPoints(env.DRONE_IDS[0], oid, physicsClientId=env.CLIENT):
                return "crash"
        state = env._getDroneStateVector(0)
        if abs(state[0]) > 2.5 or abs(state[1]) > 2.5 or state[2] > 2.5 or state[2] < 0.05:
            return "out-of-bounds"
        if abs(state[7]) > 0.4 or abs(state[8]) > 0.4:
            return "flip"
    return "timeout"


# ── quantitative headless evaluation ──────────────────────────────────────────

def evaluate_difficulty(model, vec_norm_path, difficulty: int, n_episodes: int) -> dict:
    venv = _make_venv(difficulty, gui=False, vec_norm_path=vec_norm_path)
    inner_env: ObstacleAviaryV2 = venv.envs[0] if hasattr(venv, "envs") else venv.venv.envs[0]

    rewards, steps_list, final_dists = [], [], []
    outcome_counts = {"success": 0, "crash": 0, "timeout": 0, "out-of-bounds": 0, "flip": 0}

    for ep in range(n_episodes):
        obs = venv.reset()
        ep_reward, ep_steps = 0.0, 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = venv.step(action)
            ep_reward += float(reward[0])
            ep_steps  += 1

            if done[0]:
                terminated = info[0].get("success", False)
                truncated  = not terminated
                outcome    = _episode_outcome(terminated, truncated, inner_env)
                outcome_counts[outcome] += 1
                final_dists.append(info[0].get("dist_to_goal", 0.0))
                break

        rewards.append(ep_reward)
        steps_list.append(ep_steps)

    venv.close()
    return {
        "mean_reward":     float(np.mean(rewards)),
        "std_reward":      float(np.std(rewards)),
        "success_rate":    outcome_counts["success"] / n_episodes,
        "outcome_counts":  outcome_counts,
        "mean_steps":      float(np.mean(steps_list)),
        "mean_dist_final": float(np.mean(final_dists)),
    }


# ── trajectory collection ──────────────────────────────────────────────────────

def collect_trajectory(model, vec_norm_path, difficulty: int, seed: int = 0) -> dict:
    import pybullet as p

    venv = _make_venv(difficulty, gui=False, vec_norm_path=vec_norm_path)
    inner_env: ObstacleAviaryV2 = venv.envs[0] if hasattr(venv, "envs") else venv.venv.envs[0]
    obs = venv.reset()

    obs_geom = []  # list of (position, half_extent) tuples
    for oid in inner_env.obstacle_ids:
        pos, _ = p.getBasePositionAndOrientation(oid, physicsClientId=inner_env.CLIENT)
        he = _get_obstacle_half_extent(oid, inner_env.CLIENT)
        obs_geom.append((pos, he))

    positions, rewards = [], []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = venv.step(action)
        state = inner_env._getDroneStateVector(0)
        positions.append(state[0:3].copy())
        rewards.append(float(reward[0]))
        if done[0]:
            terminated = info[0].get("success", False)
            truncated  = not terminated
            outcome    = _episode_outcome(terminated, truncated, inner_env)
            break

    venv.close()
    return {
        "positions":  np.array(positions),
        "rewards":    np.array(rewards),
        "outcome":    outcome,
        "obs_geom":   obs_geom,
    }


# ── 3-D trajectory plot ────────────────────────────────────────────────────────

def plot_trajectory(traj: dict, difficulty: int, save_path: str):
    pos     = traj["positions"]
    outcome = traj["outcome"]
    goal    = np.array([1.0, 0.0, 1.0])

    fig = plt.figure(figsize=(9, 6))
    ax  = fig.add_subplot(111, projection="3d")

    T       = len(pos)
    colours = plt.cm.coolwarm(np.linspace(0, 1, T))
    for i in range(T - 1):
        ax.plot(pos[i:i+2, 0], pos[i:i+2, 1], pos[i:i+2, 2],
                color=colours[i], linewidth=1.4, alpha=0.85)

    ax.scatter(*pos[0],  color="blue",  s=70,  zorder=5, label="start")
    ax.scatter(*pos[-1], color="red",   s=70,  zorder=5, label=f"end ({outcome})")
    ax.scatter(*goal,    color="lime",  s=140, marker="*", zorder=5, label="goal")

    # obstacles — boxes as wireframe cubes, cylinders as approximate cubes
    for (ox, oy, oz), he in traj["obs_geom"]:
        corners = np.array([
            [ox-he, oy-he, oz-he], [ox+he, oy-he, oz-he],
            [ox+he, oy+he, oz-he], [ox-he, oy+he, oz-he],
            [ox-he, oy-he, oz+he], [ox+he, oy-he, oz+he],
            [ox+he, oy+he, oz+he], [ox-he, oy+he, oz+he],
        ])
        for a, b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
            ax.plot(*zip(corners[a], corners[b]), color="darkred", linewidth=1.5, alpha=0.6)

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    n_obs = len(traj["obs_geom"])
    lo, hi = DIFFICULTY_OBSTACLE_COUNTS.get(difficulty, (0, 0))
    ax.set_title(
        f"V2 Obstacle Avoidance  |  difficulty={difficulty} ({lo}–{hi} obstacles)  |  {outcome}"
    )
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[INFO] Trajectory plot saved → {save_path}")
    plt.close(fig)


# ── stats table ────────────────────────────────────────────────────────────────

def print_stats_table(all_stats: dict):
    print("\n" + "═" * 76)
    print(f"  {'Diff':<6} {'Obstacles':<12} {'MeanReward':>12} {'Success%':>9} "
          f"{'Crash%':>7} {'Timeout%':>9} {'MeanDist':>9}")
    print("─" * 76)
    for diff, s in all_stats.items():
        n   = sum(s["outcome_counts"].values())
        sr  = 100 * s["success_rate"]
        cr  = 100 * s["outcome_counts"]["crash"]   / n
        tr  = 100 * s["outcome_counts"]["timeout"] / n
        lo, hi = DIFFICULTY_OBSTACLE_COUNTS.get(diff, (0, 0))
        obs_str = f"{lo}–{hi}"
        print(f"  {diff:<6} {obs_str:<12} {s['mean_reward']:>+12.2f} {sr:>8.1f}% "
              f"{cr:>6.1f}% {tr:>8.1f}% {s['mean_dist_final']:>9.3f}m")
    print("═" * 76 + "\n")


# ── PyBullet GUI live demo ─────────────────────────────────────────────────────

def run_pybullet_gui(model, vec_norm_path, difficulty: int, seed: int = 0,
                     n_episodes: int = 3):
    """Live episode(s) in the PyBullet GUI window — real-time, real physics."""
    venv = _make_venv(difficulty, gui=True, vec_norm_path=vec_norm_path)
    inner_env: ObstacleAviaryV2 = venv.envs[0] if hasattr(venv, "envs") else venv.venv.envs[0]

    lo, hi = DIFFICULTY_OBSTACLE_COUNTS.get(difficulty, (0, 0))
    print(f"\n[GUI] difficulty={difficulty}  obstacles={lo}–{hi}  seed={seed}  episodes={n_episodes}")
    print(f"{'Step':>5}  {'x':>7} {'y':>7} {'z':>7}  {'dist':>6}  {'reward':>8}")

    # Reset once — SB3 auto-resets after each done, so we must NOT reset again
    # between episodes or the scene will flicker from a double-reset.
    obs = venv.reset()

    for ep in range(n_episodes):
        print(f"\n── Episode {ep + 1}/{n_episodes} ──")
        start_t   = time.time()
        ep_reward = 0.0
        max_steps = int((inner_env.EPISODE_LEN_SEC + 2) * inner_env.CTRL_FREQ)

        for i in range(max_steps):
            prev_state = inner_env._getDroneStateVector(0)

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = venv.step(action)
            ep_reward += float(reward[0])

            # After done, SB3 already auto-reset — use prev_state to show
            # where the episode actually ended, not the new initial position.
            state = prev_state if done[0] else inner_env._getDroneStateVector(0)
            dist  = info[0].get("dist_to_goal", 0.0)
            print(f"{i:5d}  {state[0]:7.3f} {state[1]:7.3f} {state[2]:7.3f}"
                  f"  {dist:6.3f}  {float(reward[0]):8.3f}", end="")

            if done[0]:
                success = info[0].get("success", False)
                if success:
                    outcome = "success"
                elif float(reward[0]) < -2.0:
                    outcome = "crash"
                else:
                    outcome = "timeout"
                print(f"  ← {outcome}  (total={ep_reward:.2f})")
                break
            print()
            sync(i, start_t, inner_env.CTRL_TIMESTEP)

        # Pause so you can see the final state before the next episode resets.
        print(f"[GUI] Episode {ep + 1} done — pausing 3 s …")
        time.sleep(3.0)

    venv.close()


# ── Vispy live demo ────────────────────────────────────────────────────────────

def run_vispy_demo(model, vec_norm_path, difficulty: int, seed: int = 0):
    """Live episode rendered by Vispy — headless physics, custom 3-D window."""
    from stream_viz_v2 import DroneVizV2

    venv = _make_venv(difficulty, gui=False, vec_norm_path=vec_norm_path)
    inner_env: ObstacleAviaryV2 = venv.envs[0] if hasattr(venv, "envs") else venv.venv.envs[0]

    lo, hi = DIFFICULTY_OBSTACLE_COUNTS.get(difficulty, (0, 0))
    viz = DroneVizV2(inner_env, venv,
                     title=f"V2 Obstacle Avoidance  |  difficulty {difficulty}  ({lo}–{hi} obstacles)")
    try:
        viz.run(model=model, seed=seed)
    finally:
        venv.close()


# ── main entry point ───────────────────────────────────────────────────────────

def run(
    model_path=DEFAULT_MODEL_PATH,
    difficulty=DEFAULT_DIFFICULTY,
    n_episodes=DEFAULT_N_EPISODES,
    gui=DEFAULT_GUI,
    vispy=DEFAULT_VISPY,
    plot=DEFAULT_PLOT,
    output_folder=DEFAULT_OUTPUT,
    seed=0,
):
    if model_path is None:
        model_path = _find_latest_model(output_folder)
    print(f"[INFO] Model: {model_path}")

    model, vec_norm_path = _load_model_and_vecnorm(model_path)
    if vec_norm_path:
        print(f"[INFO] VecNormalize: {vec_norm_path}")
    else:
        print("[WARN] vec_normalize.pkl not found — running without obs normalization")

    save_dir = os.path.dirname(model_path)
    difficulties = [difficulty] if difficulty is not None else [1, 2, 3]

    # ── quantitative eval ─────────────────────────────────────────────────────
    if not gui and not vispy:
        all_stats = {}
        for diff in difficulties:
            print(f"[INFO] Evaluating difficulty={diff} × {n_episodes} episodes ...")
            stats = evaluate_difficulty(model, vec_norm_path, diff, n_episodes)
            all_stats[diff] = stats
            c = stats["outcome_counts"]
            print(f"       reward={stats['mean_reward']:.2f}±{stats['std_reward']:.2f}  "
                  f"success={c['success']}  crash={c['crash']}  timeout={c['timeout']}")
        print_stats_table(all_stats)

    # ── trajectory plots ──────────────────────────────────────────────────────
    if plot and not gui and not vispy:
        for diff in difficulties:
            print(f"[INFO] Collecting trajectory for plot (difficulty={diff}) ...")
            traj      = collect_trajectory(model, vec_norm_path, diff, seed=seed)
            plot_path = os.path.join(save_dir, f"trajectory_v2_diff{diff}.png")
            plot_trajectory(traj, diff, plot_path)

    # ── live GUI demo ─────────────────────────────────────────────────────────
    if gui:
        diff_for_gui = difficulty if difficulty is not None else 2
        run_pybullet_gui(model, vec_norm_path, diff_for_gui, seed=seed,
                         n_episodes=n_episodes)

    # ── vispy live demo ───────────────────────────────────────────────────────
    if vispy:
        diff_for_viz = difficulty if difficulty is not None else 2
        run_vispy_demo(model, vec_norm_path, diff_for_viz, seed=seed)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and demo a trained V2 obstacle policy.")
    parser.add_argument("--model_path",    default=DEFAULT_MODEL_PATH,  type=str,      metavar="")
    parser.add_argument("--difficulty",    default=DEFAULT_DIFFICULTY,  type=int,      metavar="")
    parser.add_argument("--n_episodes",    default=DEFAULT_N_EPISODES,  type=int,      metavar="")
    parser.add_argument("--gui",           default=DEFAULT_GUI,         type=str2bool, metavar="")
    parser.add_argument("--vispy",         default=DEFAULT_VISPY,       type=str2bool, metavar="")
    parser.add_argument("--plot",          default=DEFAULT_PLOT,        type=str2bool, metavar="")
    parser.add_argument("--output_folder", default=DEFAULT_OUTPUT,      type=str,      metavar="")
    parser.add_argument("--seed",          default=0,                   type=int,      metavar="")
    args = parser.parse_args()
    run(**vars(args))
