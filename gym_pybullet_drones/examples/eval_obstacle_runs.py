"""Aggregate and rank obstacle-navigation checkpoints across run folders.

Supports:
  - V2 runs: obstacle_v2-*, obstacle_v2_sac-*, obstacle_v2_td3-*
  - V3 runs: obstacle_v3-*, obstacle_v3_sac-*, obstacle_v3_td3-*
"""

import argparse
import json
import os
from glob import glob

import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gym_pybullet_drones.envs.ObstacleAviaryV2 import ObstacleAviaryV2
from gym_pybullet_drones.envs.ObstacleAviaryV3 import ObstacleAviaryV3
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from gym_pybullet_drones.utils.utils import str2bool

DEFAULT_OBS = ObservationType("kin")
DEFAULT_ACT = ActionType("pid")


def _infer_algo(path):
    p = path.lower()
    if "_sac-" in p or "/sac" in p:
        return "sac"
    if "_td3-" in p or "/td3" in p:
        return "td3"
    return "ppo"


def _load_model(model_path, algo):
    if algo == "ppo":
        return PPO.load(model_path)
    if algo == "sac":
        return SAC.load(model_path)
    if algo == "td3":
        return TD3.load(model_path)
    raise ValueError(f"Unsupported algo: {algo}")


def _make_eval_venv(task, difficulty, vec_norm_path):
    if task == "v2":
        def _make():
            return ObstacleAviaryV2(obs=DEFAULT_OBS, act=DEFAULT_ACT, difficulty=difficulty)
    else:
        def _make():
            return ObstacleAviaryV3(obs=DEFAULT_OBS, act=DEFAULT_ACT, difficulty=difficulty)

    venv = DummyVecEnv([_make])
    if vec_norm_path and os.path.exists(vec_norm_path):
        venv = VecNormalize.load(vec_norm_path, venv)
        venv.training = False
        venv.norm_reward = False
    return venv


def _evaluate_one(task, model, model_path, difficulty, n_episodes, seed):
    vec_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
    if not os.path.exists(vec_path):
        vec_path = os.path.join(os.path.dirname(model_path), "vec_normalize_final.pkl")

    venv = _make_eval_venv(task, difficulty, vec_path if os.path.exists(vec_path) else None)
    inner_env = venv.envs[0] if hasattr(venv, "envs") else venv.venv.envs[0]
    max_steps = int((inner_env.EPISODE_LEN_SEC + (4 if task == "v3" else 2)) * inner_env.CTRL_FREQ)

    successes = 0
    crashes = 0
    timeouts = 0
    rewards = []
    fractions = []

    for ep in range(n_episodes):
        obs = venv.reset()
        ep_reward = 0.0
        ep_seed = seed + ep if seed is not None else None
        try:
            inner_env.reset(seed=ep_seed)
        except Exception:
            pass

        last_info = {}
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = venv.step(action)
            ep_reward += float(reward[0])
            last_info = info[0]
            if bool(done[0]):
                break

        success = bool(last_info.get("success", False))
        dist = float(last_info.get("dist_to_goal", last_info.get("dist_to_current_wp", 999.0)))
        if success:
            successes += 1
        else:
            if dist > 0.1:
                crashes += 1
            else:
                timeouts += 1

        rewards.append(ep_reward)
        if task == "v3":
            fractions.append(float(last_info.get("fraction_completed", 0.0)))

    venv.close()
    out = {
        "success_rate": successes / n_episodes,
        "crash_rate": crashes / n_episodes,
        "timeout_rate": timeouts / n_episodes,
        "mean_reward": float(np.mean(rewards)),
    }
    if task == "v3":
        out["mean_fraction_completed"] = float(np.mean(fractions)) if fractions else 0.0
    return out


def _find_models(results_dir, task):
    if task == "v2":
        pats = ["obstacle_v2-*", "obstacle_v2_sac-*", "obstacle_v2_td3-*"]
    else:
        pats = ["obstacle_v3-*", "obstacle_v3_sac-*", "obstacle_v3_td3-*"]
    dirs = []
    for pat in pats:
        dirs.extend(glob(os.path.join(results_dir, pat)))
    dirs = sorted(set(dirs), key=os.path.getmtime, reverse=True)

    model_paths = []
    for d in dirs:
        for name in ("best_model.zip", "final_model.zip"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                model_paths.append(p)
                break
    return model_paths


def run(results_dir="results", task="v3", difficulty=3, n_episodes=20, top_k=10, seed=0, output_json=True):
    model_paths = _find_models(results_dir, task)
    if not model_paths:
        print("[ERROR] No model checkpoints found.")
        return None

    rows = []
    for mp in model_paths:
        algo = _infer_algo(mp)
        print(f"[INFO] Evaluating {algo.upper()} | {mp}")
        model = _load_model(mp, algo)
        metrics = _evaluate_one(task=task, model=model, model_path=mp, difficulty=difficulty, n_episodes=n_episodes, seed=seed)
        score = metrics["success_rate"] - 0.5 * metrics["crash_rate"]
        if task == "v3":
            score += 0.25 * metrics.get("mean_fraction_completed", 0.0)
        rows.append(
            {
                "task": task,
                "algo": algo,
                "run_dir": os.path.dirname(mp),
                "model_path": mp,
                "difficulty": difficulty,
                "n_episodes": n_episodes,
                "score": score,
                **metrics,
            }
        )

    rows.sort(key=lambda x: x["score"], reverse=True)
    print("\nRank  Algo   Score   Success   Crash   Timeout   Reward   Run")
    print("-" * 95)
    for i, r in enumerate(rows[:top_k], start=1):
        print(
            f"{i:>4}  {r['algo']:<4}  {r['score']:>6.3f}  "
            f"{r['success_rate']*100:>6.1f}%   {r['crash_rate']*100:>6.1f}%   "
            f"{r['timeout_rate']*100:>7.1f}%   {r['mean_reward']:>7.2f}   {os.path.basename(r['run_dir'])}"
        )

    if output_json:
        out_path = os.path.join(results_dir, f"obstacle_{task}_leaderboard.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"\n[INFO] Wrote leaderboard JSON: {out_path}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and rank obstacle run folders.")
    parser.add_argument("--results_dir", default="results", type=str, metavar="")
    parser.add_argument("--task", default="v3", choices=["v2", "v3"], type=str, metavar="")
    parser.add_argument("--difficulty", default=3, type=int, metavar="")
    parser.add_argument("--n_episodes", default=20, type=int, metavar="")
    parser.add_argument("--top_k", default=10, type=int, metavar="")
    parser.add_argument("--seed", default=0, type=int, metavar="")
    parser.add_argument("--output_json", default=True, type=str2bool, metavar="")
    args = parser.parse_args()
    run(**vars(args))
