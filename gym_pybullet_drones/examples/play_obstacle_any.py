"""Run a trained obstacle policy in the PyBullet GUI (PPO, SAC, or TD3).

Use this for off-policy runs (SAC/TD3) and VecNormalize; play_obstacle_v3.py is PPO-only.

Examples
--------
  python play_obstacle_any.py --task v3 --model_path ~/drone-results/obstacle_v3_sac-.../best_model.zip --gui true
  python play_obstacle_any.py --task v3 --pointer_json ~/drone-results/best_checkpoint_v3.json --gui true
"""

import argparse
import json
import os
import sys
import time

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gym_pybullet_drones.envs.ObstacleAviaryV2 import ObstacleAviaryV2
from gym_pybullet_drones.envs.ObstacleAviaryV3 import ObstacleAviaryV3, DIFFICULTY_CONFIG
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from gym_pybullet_drones.utils.utils import str2bool, sync

DEFAULT_OBS = ObservationType("kin")
DEFAULT_ACT = ActionType("pid")


def _infer_algo(model_path: str) -> str:
    p = model_path.lower()
    if "_sac-" in p or "sac_" in p or "/sac-" in p:
        return "sac"
    if "_td3-" in p or "td3_" in p or "/td3-" in p:
        return "td3"
    return "ppo"


def _load_policy(model_path: str, algo: str):
    if algo == "ppo":
        return PPO.load(model_path)
    if algo == "sac":
        return SAC.load(model_path)
    if algo == "td3":
        return TD3.load(model_path)
    raise ValueError(algo)


def _vec_paths(run_dir: str):
    for name in ("vec_normalize.pkl", "vec_normalize_final.pkl"):
        p = os.path.join(run_dir, name)
        if os.path.isfile(p):
            return p
    return None


def _make_venv(task: str, difficulty: int, gui: bool, vec_path: str | None):
    if task == "v2":

        def _make():
            return ObstacleAviaryV2(obs=DEFAULT_OBS, act=DEFAULT_ACT, difficulty=difficulty, gui=gui)

    else:

        def _make():
            return ObstacleAviaryV3(obs=DEFAULT_OBS, act=DEFAULT_ACT, difficulty=difficulty, gui=gui)

    venv = DummyVecEnv([_make])
    if vec_path:
        venv = VecNormalize.load(vec_path, venv)
        venv.training = False
        venv.norm_reward = False
    return venv


def _inner_env(venv):
    v = venv
    while hasattr(v, "venv"):
        v = v.venv
    return v.envs[0]


def run(
    task: str,
    model_path: str | None,
    pointer_json: str | None,
    difficulty: int,
    gui: bool,
    seed: int | None,
    episodes: int,
):
    if pointer_json:
        with open(pointer_json, encoding="utf-8") as f:
            payload = json.load(f)
        best = payload.get("best") or {}
        model_path = best.get("model_path") or model_path
        if not model_path:
            print("[ERROR] pointer_json missing best.model_path", file=sys.stderr)
            sys.exit(1)

    if not model_path or not os.path.isfile(model_path):
        print(f"[ERROR] model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    run_dir = os.path.dirname(model_path)
    algo = _infer_algo(model_path)
    vec_path = _vec_paths(run_dir)

    print(f"[INFO] model_path={model_path}")
    vn = vec_path if vec_path else "(none; inference may be wrong without VecNormalize)"
    print(f"[INFO] inferred algo={algo}  vec_normalize={vn}")

    model = _load_policy(model_path, algo)
    venv = _make_venv(task, difficulty, gui, vec_path)
    inner = _inner_env(venv)

    if task == "v3":
        n_wp = DIFFICULTY_CONFIG.get(difficulty, (3, 0, 0))[0]
        print(f"[INFO] difficulty={difficulty}  waypoints≈{n_wp}")

    for ep in range(episodes):
        obs = venv.reset()
        if seed is not None:
            try:
                inner.reset(seed=seed + ep)
            except TypeError:
                pass
        ep_reward = 0.0
        max_steps = int((inner.EPISODE_LEN_SEC + 4) * inner.CTRL_FREQ)
        start_t = time.time()
        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = venv.step(action)
            ep_reward += float(rewards[0])
            if gui:
                sync(step, start_t, inner.CTRL_TIMESTEP)
            if bool(dones[0]):
                info = infos[0]
                ok = info.get("success", False)
                frac = info.get("fraction_completed")
                print(f"[EP {ep + 1}] reward={ep_reward:.2f} success={ok} fraction_completed={frac}")
                break
        else:
            print(f"[EP {ep + 1}] reward={ep_reward:.2f} (timeout)")

    venv.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play obstacle policy (PPO/SAC/TD3) with VecNormalize.")
    parser.add_argument("--task", choices=["v2", "v3"], default="v3")
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--pointer_json", default=None, type=str, help="best_checkpoint_v*.json from autopilot")
    parser.add_argument("--difficulty", default=1, type=int)
    parser.add_argument("--gui", default=True, type=str2bool)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--episodes", default=1, type=int)
    args = parser.parse_args()
    run(
        task=args.task,
        model_path=args.model_path,
        pointer_json=args.pointer_json,
        difficulty=args.difficulty,
        gui=args.gui,
        seed=args.seed,
        episodes=args.episodes,
    )
