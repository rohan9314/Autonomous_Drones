"""Train off-policy RL agents on ObstacleAviaryV3 (SAC/TD3).

V3 is the multi-waypoint task; curriculum is based on mean fraction completed.
"""

import argparse
import os
from collections import deque
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from gym_pybullet_drones.envs.ObstacleAviaryV3 import ObstacleAviaryV3
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from gym_pybullet_drones.utils.utils import str2bool

DEFAULT_OBS = ObservationType("kin")
DEFAULT_ACT = ActionType("pid")
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_DIFFICULTY = 1
DEFAULT_TOTAL_STEPS = int(10e6)
DEFAULT_N_ENVS = 8
DEFAULT_BATCH_SIZE = 1024
DEFAULT_BUFFER_SIZE = 1_000_000
DEFAULT_LEARNING_STARTS = 20_000
DEFAULT_NET_ARCH = "512,512,256"
DEFAULT_ALGO = "sac"

ADVANCE_THRESHOLD = 0.85
RETREAT_THRESHOLD = 0.35
ADVANCE_CONSEC = 3
RETREAT_CONSEC = 2
MAX_DIFFICULTY = 4


class FractionCurriculumCallback(BaseCallback):
    def __init__(self, eval_freq, save_path, starting_difficulty=1, max_difficulty=MAX_DIFFICULTY, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.difficulty = starting_difficulty
        self.max_difficulty = max_difficulty
        self.best_mean_fraction = 0.0
        self.episode_fractions = deque(maxlen=300)
        self.consec_above = 0
        self.consec_below = 0
        self.plot_data = []
        self.transitions = []

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for done, info in zip(dones, infos):
            if done:
                frac = float(info.get("fraction_completed", 0.0))
                if frac == 0.0 and info.get("success", False):
                    frac = 1.0
                self.episode_fractions.append(frac)

        if self.n_calls % self.eval_freq != 0:
            return True
        if len(self.episode_fractions) < 30:
            return True

        mean_frac = float(np.mean(self.episode_fractions))
        self.plot_data.append((self.num_timesteps, mean_frac * 100.0, self.difficulty))
        if self.verbose:
            print(f"[CURR] step={self.num_timesteps:,} diff={self.difficulty} mean_frac={mean_frac:.1%}")

        if mean_frac > self.best_mean_fraction:
            self.best_mean_fraction = mean_frac
            self.model.save(os.path.join(self.save_path, "best_model"))
            if hasattr(self.training_env, "save"):
                self.training_env.save(os.path.join(self.save_path, "vec_normalize.pkl"))

        if mean_frac > ADVANCE_THRESHOLD:
            self.consec_above += 1
            self.consec_below = 0
        elif mean_frac < RETREAT_THRESHOLD:
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
        elif self.consec_below >= RETREAT_CONSEC and self.difficulty > 0:
            self.consec_below = 0
            old_diff = self.difficulty
            self.difficulty -= 1
            self.training_env.set_attr("difficulty", self.difficulty)
            self.transitions.append((self.num_timesteps, old_diff, self.difficulty))
        return True


def _build_model(algo, env, net_arch, learning_rate, buffer_size, learning_starts, batch_size, tb_log):
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
            gamma=0.99,
            tau=0.02,
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
            gamma=0.99,
            tau=0.02,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_log,
            verbose=1,
        )
    raise ValueError(f"Unsupported algo: {algo}")


def _plot_curve(save_dir, cb):
    if not cb.plot_data:
        return
    steps = [d[0] for d in cb.plot_data]
    frac = [d[1] for d in cb.plot_data]
    window = min(10, len(frac))
    rolled = np.convolve(frac, np.ones(window) / window, mode="valid")
    rolled_steps = steps[window - 1 :]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(steps, frac, alpha=0.35, linewidth=1, color="steelblue", label="Mean fraction (raw %)")
    ax.plot(rolled_steps, rolled, linewidth=2, color="steelblue", label=f"Rolling avg (window={window})")
    ax.axhline(ADVANCE_THRESHOLD * 100, color="green", linestyle="--", alpha=0.7)
    ax.axhline(RETREAT_THRESHOLD * 100, color="red", linestyle="--", alpha=0.7)
    for ts, from_d, to_d in cb.transitions:
        ax.axvline(ts, color="green" if to_d > from_d else "red", linestyle=":", alpha=0.8)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Fraction Completed (%)")
    ax.set_ylim(0, 105)
    ax.set_title("ObstacleAviaryV3 Off-policy Curriculum Training Curve")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_curve_v3_offpolicy.png"), dpi=150)
    plt.close(fig)


def run(
    algo=DEFAULT_ALGO,
    difficulty=DEFAULT_DIFFICULTY,
    output_folder=DEFAULT_OUTPUT_FOLDER,
    gui=False,
    total_timesteps=DEFAULT_TOTAL_STEPS,
    n_envs=DEFAULT_N_ENVS,
    batch_size=DEFAULT_BATCH_SIZE,
    buffer_size=DEFAULT_BUFFER_SIZE,
    learning_starts=DEFAULT_LEARNING_STARTS,
    net_arch=DEFAULT_NET_ARCH,
    learning_rate=None,
):
    _ = gui
    algo = algo.lower()
    if algo not in {"sac", "td3"}:
        raise ValueError("algo must be one of: sac, td3")

    net_arch_list = [int(x.strip()) for x in net_arch.split(",")] if isinstance(net_arch, str) else list(net_arch)
    timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    save_dir = os.path.join(output_folder, f"obstacle_v3_{algo}-{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] Saving results to: {save_dir}")

    train_env = make_vec_env(
        ObstacleAviaryV3,
        env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT, difficulty=difficulty),
        n_envs=n_envs,
        seed=0,
    )
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=5.0, clip_reward=10.0)

    try:
        import tensorboard  # noqa: F401
        tb_log = os.path.join(save_dir, "tb")
    except ImportError:
        tb_log = None

    model = _build_model(
        algo=algo,
        env=train_env,
        net_arch=net_arch_list,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tb_log=tb_log,
    )

    cb = FractionCurriculumCallback(eval_freq=1000, save_path=save_dir, starting_difficulty=difficulty, verbose=1)
    model.learn(total_timesteps=int(total_timesteps), callback=cb, log_interval=50)

    model.save(os.path.join(save_dir, "final_model"))
    train_env.save(os.path.join(save_dir, "vec_normalize_final.pkl"))
    if not os.path.exists(os.path.join(save_dir, "best_model.zip")):
        model.save(os.path.join(save_dir, "best_model"))
        train_env.save(os.path.join(save_dir, "vec_normalize.pkl"))
    _plot_curve(save_dir, cb)

    eval_env = ObstacleAviaryV3(obs=DEFAULT_OBS, act=DEFAULT_ACT, difficulty=cb.difficulty)
    fracs, succ = [], []
    for _ in range(20):
        obs, _ = eval_env.reset()
        max_steps = int((eval_env.EPISODE_LEN_SEC + 4) * eval_env.CTRL_FREQ)
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = eval_env.step(action)
            if terminated or truncated:
                fracs.append(float(info.get("fraction_completed", 0.0)))
                succ.append(bool(info.get("success", False)))
                break
    eval_env.close()
    print(f"[RESULT] mean fraction: {float(np.mean(fracs)):.1%}")
    print(f"[RESULT] success rate : {float(np.mean(succ)):.1%} ({int(sum(succ))}/20)")
    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC/TD3 on ObstacleAviaryV3.")
    parser.add_argument("--algo", default=DEFAULT_ALGO, type=str, metavar="")
    parser.add_argument("--difficulty", default=DEFAULT_DIFFICULTY, type=int, metavar="")
    parser.add_argument("--output_folder", default=DEFAULT_OUTPUT_FOLDER, type=str, metavar="")
    parser.add_argument("--gui", default=False, type=str2bool, metavar="")
    parser.add_argument("--total_timesteps", default=DEFAULT_TOTAL_STEPS, type=float, metavar="")
    parser.add_argument("--n_envs", default=DEFAULT_N_ENVS, type=int, metavar="")
    parser.add_argument("--batch_size", default=DEFAULT_BATCH_SIZE, type=int, metavar="")
    parser.add_argument("--buffer_size", default=DEFAULT_BUFFER_SIZE, type=int, metavar="")
    parser.add_argument("--learning_starts", default=DEFAULT_LEARNING_STARTS, type=int, metavar="")
    parser.add_argument("--net_arch", default=DEFAULT_NET_ARCH, type=str, metavar="")
    parser.add_argument("--learning_rate", default=None, type=float, metavar="")
    args = parser.parse_args()
    run(**vars(args))
