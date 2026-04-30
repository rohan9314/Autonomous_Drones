"""Train off-policy RL agents on ObstacleAviaryV2 (SAC/TD3).

This script mirrors the existing PPO curriculum training flow but swaps in
off-policy algorithms that are often better sample-efficient in continuous
control tasks.
"""

import argparse
import glob
import json
import os
import re
import time
from collections import deque
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from gym_pybullet_drones.envs.ObstacleAviaryV2 import ObstacleAviaryV2
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from gym_pybullet_drones.utils.utils import str2bool, sync

DEFAULT_OBS = ObservationType("kin")
DEFAULT_ACT = ActionType("pid")
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_GUI = False
DEFAULT_DIFFICULTY = 1
DEFAULT_TOTAL_STEPS = int(5e6)
DEFAULT_N_ENVS = 8
DEFAULT_BATCH_SIZE = 1024
DEFAULT_BUFFER_SIZE = 1_000_000
DEFAULT_LEARNING_STARTS = 10_000
DEFAULT_NET_ARCH = "512,512,256"
DEFAULT_ALGO = "sac"

ADVANCE_THRESHOLD = 0.70
RETREAT_THRESHOLD = 0.35
ADVANCE_CONSEC = 3
RETREAT_CONSEC = 2
MAX_DIFFICULTY = 4

CURRICULUM_STATE_NAME = "curriculum_state.json"


class SuccessRateCurriculumCallback(BaseCallback):
    """Success-rate curriculum over training rollouts."""

    def __init__(self, eval_freq, save_path, starting_difficulty=1, max_difficulty=MAX_DIFFICULTY, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.state_path = os.path.join(save_path, CURRICULUM_STATE_NAME)
        self.max_difficulty = max_difficulty

        self.difficulty = starting_difficulty
        if os.path.isfile(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.difficulty = int(data.get("difficulty", starting_difficulty))
            except (OSError, ValueError, json.JSONDecodeError):
                self.difficulty = starting_difficulty
        self.best_success_rate = 0.0
        self.episode_outcomes = deque(maxlen=400)
        self.consec_above = 0
        self.consec_below = 0
        self.plot_data = []
        self.transitions = []

    def _persist_difficulty(self):
        try:
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump({"difficulty": self.difficulty}, f)
        except OSError:
            pass

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for done, info in zip(dones, infos):
            if done:
                self.episode_outcomes.append(1 if info.get("success", False) else 0)

        if self.n_calls % self.eval_freq != 0:
            return True
        if len(self.episode_outcomes) < 30:
            return True

        success_rate = float(np.mean(self.episode_outcomes))
        success_pct = success_rate * 100.0
        self.plot_data.append((self.num_timesteps, success_pct, self.difficulty))

        if self.verbose:
            print(
                f"[CURR] step={self.num_timesteps:,} diff={self.difficulty} "
                f"success={success_pct:.1f}% n={len(self.episode_outcomes)}"
            )

        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
            self.model.save(os.path.join(self.save_path, "best_model"))
            if hasattr(self.training_env, "save"):
                self.training_env.save(os.path.join(self.save_path, "vec_normalize.pkl"))

        if success_rate > ADVANCE_THRESHOLD:
            self.consec_above += 1
            self.consec_below = 0
        elif success_rate < RETREAT_THRESHOLD:
            self.consec_below += 1
            self.consec_above = 0
        else:
            self.consec_above = 0
            self.consec_below = 0

        if self.consec_above >= ADVANCE_CONSEC and self.difficulty < self.max_difficulty:
            self.consec_above = 0
            old_diff = self.difficulty
            self.difficulty += 1
            self.training_env.set_attr("difficulty", self.difficulty)
            self.transitions.append((self.num_timesteps, old_diff, self.difficulty))
            self.model.save(os.path.join(self.save_path, f"model_diff{old_diff}_complete"))
            self._persist_difficulty()
            if self.verbose:
                print(f"[CURR] advance {old_diff} -> {self.difficulty}")
        elif self.consec_below >= RETREAT_CONSEC and self.difficulty > 1:
            self.consec_below = 0
            old_diff = self.difficulty
            self.difficulty -= 1
            self.training_env.set_attr("difficulty", self.difficulty)
            self.transitions.append((self.num_timesteps, old_diff, self.difficulty))
            self._persist_difficulty()
            if self.verbose:
                print(f"[CURR] retreat {old_diff} -> {self.difficulty}")

        return True


def _make_model(algo, env, net_arch, learning_rate, buffer_size, learning_starts, batch_size, tb_log):
    policy_kwargs = dict(net_arch=net_arch)
    lr = learning_rate if learning_rate is not None else (2e-4 if algo == "sac" else 1e-4)
    if algo == "sac":
        return SAC(
            "MlpPolicy",
            env,
            learning_rate=lr,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            train_freq=(1, "step"),
            gradient_steps=1,
            tau=0.02,
            gamma=0.99,
            ent_coef="auto_0.2",
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_log,
            verbose=1,
        )
    if algo == "td3":
        return TD3(
            "MlpPolicy",
            env,
            learning_rate=lr,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            train_freq=(1, "step"),
            gradient_steps=1,
            tau=0.02,
            gamma=0.99,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_log,
            verbose=1,
        )
    raise ValueError(f"Unsupported algo: {algo}")


def _read_start_difficulty(save_dir, default_difficulty):
    path = os.path.join(save_dir, CURRICULUM_STATE_NAME)
    if not os.path.isfile(path):
        return default_difficulty
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("difficulty", default_difficulty))
    except (OSError, ValueError, json.JSONDecodeError):
        return default_difficulty


def _resolve_resume_checkpoint(save_dir):
    candidates = []
    final_p = os.path.join(save_dir, "final_model.zip")
    if os.path.isfile(final_p):
        candidates.append((10**18, final_p))
    for p in glob.glob(os.path.join(save_dir, "checkpoint_*_steps.zip")):
        m = re.search(r"checkpoint_(\d+)_steps\.zip$", os.path.basename(p))
        if m:
            candidates.append((int(m.group(1)), p))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]
    best_p = os.path.join(save_dir, "best_model.zip")
    if os.path.isfile(best_p):
        return best_p
    return None


def _plot_training_curve(save_dir, cb):
    if not cb.plot_data:
        return
    steps = [d[0] for d in cb.plot_data]
    sr = [d[1] for d in cb.plot_data]
    window = min(10, len(sr))
    rolled = np.convolve(sr, np.ones(window) / window, mode="valid")
    rolled_steps = steps[window - 1 :]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(steps, sr, alpha=0.35, linewidth=1, color="steelblue", label="Success rate (raw %)")
    ax.plot(rolled_steps, rolled, linewidth=2, color="steelblue", label=f"Rolling avg (window={window})")
    ax.axhline(ADVANCE_THRESHOLD * 100, color="green", linestyle="--", alpha=0.7)
    ax.axhline(RETREAT_THRESHOLD * 100, color="red", linestyle="--", alpha=0.7)

    for ts, from_d, to_d in cb.transitions:
        color = "green" if to_d > from_d else "red"
        ax.axvline(ts, color=color, linestyle=":", alpha=0.8)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Episode Success Rate (%)")
    ax.set_title("ObstacleAviaryV2 Off-policy Curriculum Training Curve")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_curve_offpolicy.png"), dpi=150)
    plt.close(fig)


def run(
    algo=DEFAULT_ALGO,
    difficulty=DEFAULT_DIFFICULTY,
    output_folder=DEFAULT_OUTPUT_FOLDER,
    gui=DEFAULT_GUI,
    total_timesteps=DEFAULT_TOTAL_STEPS,
    n_envs=DEFAULT_N_ENVS,
    batch_size=DEFAULT_BATCH_SIZE,
    buffer_size=DEFAULT_BUFFER_SIZE,
    learning_starts=DEFAULT_LEARNING_STARTS,
    net_arch=DEFAULT_NET_ARCH,
    learning_rate=None,
    run_dir=None,
    load_model=None,
    resume_latest=False,
):
    algo = algo.lower()
    if algo not in {"sac", "td3"}:
        raise ValueError("algo must be one of: sac, td3")

    net_arch_list = [int(x.strip()) for x in net_arch.split(",")] if isinstance(net_arch, str) else list(net_arch)
    if run_dir:
        save_dir = os.path.abspath(run_dir)
    else:
        timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
        save_dir = os.path.join(output_folder, f"obstacle_v2_{algo}-{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] Saving results to: {save_dir}")

    ckpt_path = load_model
    if resume_latest:
        ckpt_path = _resolve_resume_checkpoint(save_dir)
        if ckpt_path:
            print(f"[INFO] resume_latest -> {ckpt_path}")
    start_difficulty = _read_start_difficulty(save_dir, difficulty)

    train_env = make_vec_env(
        ObstacleAviaryV2,
        env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT, difficulty=start_difficulty),
        n_envs=n_envs,
        seed=0,
    )
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=5.0, clip_reward=10.0)

    try:
        import tensorboard  # noqa: F401

        tb_log = os.path.join(save_dir, "tb")
    except ImportError:
        tb_log = None
        print("[WARN] tensorboard not installed; skipping TB logging")

    loaded = False
    if ckpt_path and os.path.isfile(ckpt_path):
        vec_norm_path = os.path.join(os.path.dirname(ckpt_path), "vec_normalize.pkl")
        if not os.path.isfile(vec_norm_path):
            vec_norm_path = os.path.join(os.path.dirname(ckpt_path), "vec_normalize_final.pkl")
        if os.path.isfile(vec_norm_path):
            print(f"[INFO] Loading VecNormalize from {vec_norm_path}")
            train_env = VecNormalize.load(vec_norm_path, train_env.venv)
            train_env.training = True
        print(f"[INFO] Loading checkpoint: {ckpt_path}")
        if algo == "sac":
            model = SAC.load(ckpt_path, env=train_env, tensorboard_log=tb_log)
        else:
            model = TD3.load(ckpt_path, env=train_env, tensorboard_log=tb_log)
        if learning_rate is not None:
            model.learning_rate = learning_rate
        loaded = True
    else:
        if ckpt_path:
            print(f"[WARN] Checkpoint path invalid or missing; training from scratch. ({ckpt_path})")
        model = _make_model(
            algo=algo,
            env=train_env,
            net_arch=net_arch_list,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tb_log=tb_log,
        )

    curriculum_cb = SuccessRateCurriculumCallback(
        eval_freq=1000, save_path=save_dir, starting_difficulty=start_difficulty, verbose=1
    )
    train_env.set_attr("difficulty", curriculum_cb.difficulty)
    curriculum_cb._persist_difficulty()

    chk = CheckpointCallback(
        save_freq=max(50_000 // max(n_envs, 1), 1000),
        save_path=save_dir,
        name_prefix="checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    print(f"[INFO] Training {algo.upper()} for {int(total_timesteps):,} steps over {n_envs} envs")
    model.learn(
        total_timesteps=int(total_timesteps),
        callback=[curriculum_cb, chk],
        log_interval=50,
        reset_num_timesteps=not loaded,
    )

    model.save(os.path.join(save_dir, "final_model"))
    train_env.save(os.path.join(save_dir, "vec_normalize_final.pkl"))
    if not os.path.exists(os.path.join(save_dir, "best_model.zip")):
        model.save(os.path.join(save_dir, "best_model"))
        train_env.save(os.path.join(save_dir, "vec_normalize.pkl"))

    _plot_training_curve(save_dir, curriculum_cb)

    eval_env = ObstacleAviaryV2(obs=DEFAULT_OBS, act=DEFAULT_ACT, difficulty=curriculum_cb.difficulty)
    success_flags = []
    rewards = []
    steps_to_goal = []
    max_steps = (eval_env.EPISODE_LEN_SEC + 2) * eval_env.CTRL_FREQ
    for _ in range(20):
        obs, _ = eval_env.reset()
        ep_reward = 0.0
        ep_steps = 0
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += reward
            ep_steps += 1
            if terminated or truncated:
                succ = info.get("success", False)
                success_flags.append(succ)
                if succ:
                    steps_to_goal.append(ep_steps)
                break
        rewards.append(ep_reward)
    eval_env.close()

    print(f"[RESULT] mean reward: {float(np.mean(rewards)):.2f} +/- {float(np.std(rewards)):.2f}")
    print(f"[RESULT] success rate: {float(np.mean(success_flags)):.1%} ({int(sum(success_flags))}/20)")
    if steps_to_goal:
        print(f"[RESULT] mean steps to goal: {float(np.mean(steps_to_goal)):.1f}")

    test_env = ObstacleAviaryV2(obs=DEFAULT_OBS, act=DEFAULT_ACT, difficulty=curriculum_cb.difficulty, gui=gui)
    obs, _ = test_env.reset(seed=42)
    ep_reward = 0.0
    start_t = time.time()
    max_steps = (test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ
    for i in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        ep_reward += reward
        if gui:
            sync(i, start_t, test_env.CTRL_TIMESTEP)
        if terminated or truncated:
            status = "SUCCESS" if info.get("success", False) else "FAIL"
            print(f"[TEST] Episode ended: {status} total_reward={ep_reward:.2f}")
            break
    test_env.close()
    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC/TD3 on ObstacleAviaryV2 with curriculum.")
    parser.add_argument("--algo", default=DEFAULT_ALGO, type=str, metavar="")
    parser.add_argument("--difficulty", default=DEFAULT_DIFFICULTY, type=int, metavar="")
    parser.add_argument("--output_folder", default=DEFAULT_OUTPUT_FOLDER, type=str, metavar="")
    parser.add_argument("--gui", default=DEFAULT_GUI, type=str2bool, metavar="")
    parser.add_argument("--total_timesteps", default=DEFAULT_TOTAL_STEPS, type=float, metavar="")
    parser.add_argument("--n_envs", default=DEFAULT_N_ENVS, type=int, metavar="")
    parser.add_argument("--batch_size", default=DEFAULT_BATCH_SIZE, type=int, metavar="")
    parser.add_argument("--buffer_size", default=DEFAULT_BUFFER_SIZE, type=int, metavar="")
    parser.add_argument("--learning_starts", default=DEFAULT_LEARNING_STARTS, type=int, metavar="")
    parser.add_argument("--net_arch", default=DEFAULT_NET_ARCH, type=str, metavar="")
    parser.add_argument("--learning_rate", default=None, type=float, metavar="")
    parser.add_argument("--run_dir", default=None, type=str, metavar="", help="Fixed run directory for chained SLURM jobs.")
    parser.add_argument("--load_model", default=None, type=str, metavar="", help="Path to .zip checkpoint to resume.")
    parser.add_argument("--resume_latest", default=False, type=str2bool, metavar="", help="Resume from latest checkpoint in run_dir.")
    args = parser.parse_args()
    run(**vars(args))
