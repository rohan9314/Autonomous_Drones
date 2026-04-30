"""Training script for ObstacleAviaryV3: multi-waypoint navigation with lidar.

Key differences from v2:
  - Uses ObstacleAviaryV3: sequential N-waypoint task
  - Curriculum metric: fraction_completed (continuous) not binary success
  - Difficulty simultaneously advances obstacle count and waypoint count
  - Advance threshold: mean fraction_completed > 0.85 for 3 checks
  - Retreat threshold: mean fraction_completed < 0.35 for 2 checks

Usage:
    python learn_obstacle_v3.py                              # default: 8M steps, diff 1
    python learn_obstacle_v3.py --total_timesteps 12e6       # longer run
    python learn_obstacle_v3.py --difficulty 2               # start at diff 2
    python learn_obstacle_v3.py --load_model path/best_model.zip --learning_rate 1e-4
"""

import os
import time
import argparse
from collections import deque
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from gym_pybullet_drones.envs.ObstacleAviaryV3 import ObstacleAviaryV3
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync, str2bool

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_OBS           = ObservationType('kin')
DEFAULT_ACT           = ActionType('pid')
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_GUI           = False
DEFAULT_DIFFICULTY    = 1
DEFAULT_TOTAL_STEPS   = int(8e6)
DEFAULT_N_ENVS        = 16
DEFAULT_N_STEPS       = 2048
DEFAULT_BATCH_SIZE    = 512
DEFAULT_NET_ARCH      = "256,256,128"

# Curriculum thresholds on fraction_completed (continuous 0–1)
ADVANCE_THRESHOLD = 0.85   # advance when rolling mean fraction > this
RETREAT_THRESHOLD = 0.35   # retreat when rolling mean fraction < this
ADVANCE_CONSEC    = 3      # consecutive eval periods required to advance
RETREAT_CONSEC    = 2      # consecutive eval periods required to retreat
MAX_DIFFICULTY    = 4


# ── curriculum callback ────────────────────────────────────────────────────────

class CurriculumCallbackV3(BaseCallback):
    """Fraction-completed based curriculum callback for ObstacleAviaryV3.

    Tracks episode `fraction_completed` from info dicts across all parallel envs.
    fraction_completed = current_wp_idx / n_waypoints at episode end.

    Curriculum rules:
      - Advance when rolling mean fraction_completed > 0.85 for 3 consecutive checks.
      - Retreat when rolling mean fraction_completed < 0.35 for 2 consecutive checks
        (minimum difficulty = 0, but training starts at 1 by default).
    """

    def __init__(self, eval_freq, save_path, starting_difficulty=1,
                 max_difficulty=MAX_DIFFICULTY, verbose=1,
                 stability_window=ADVANCE_CONSEC):
        super().__init__(verbose)
        self.eval_freq        = eval_freq
        self.save_path        = save_path
        self.max_difficulty   = max_difficulty
        self.stability_window = stability_window

        self.difficulty            = starting_difficulty
        self.best_mean_fraction    = 0.0

        # Rolling deque of fraction_completed values from completed episodes
        self.episode_fractions = deque(maxlen=200)

        self.consec_above = 0
        self.consec_below = 0

        # (timestep, mean_fraction_pct, difficulty) for plotting
        self.plot_data   = []
        self.transitions = []   # (timestep, from_diff, to_diff)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for done, info in zip(dones, infos):
            if done:
                frac = info.get("fraction_completed", 0.0)
                # Treat binary success as fraction = 1.0 if present and fraction missing
                if frac == 0.0 and info.get("success", False):
                    frac = 1.0
                self.episode_fractions.append(frac)

        if self.n_calls % self.eval_freq != 0:
            return True

        if len(self.episode_fractions) < 20:
            if self.verbose:
                print(f"[V3] step={self.num_timesteps:,}  diff={self.difficulty}  "
                      f"waiting for episodes... ({len(self.episode_fractions)}/20)")
            return True

        mean_frac     = float(np.mean(self.episode_fractions))
        mean_frac_pct = mean_frac * 100.0
        n_tracked     = len(self.episode_fractions)

        if self.verbose:
            print(f"[V3] step={self.num_timesteps:,}  diff={self.difficulty}  "
                  f"mean_frac={mean_frac_pct:.1f}%  ({n_tracked} episodes tracked)")

        self.plot_data.append((self.num_timesteps, mean_frac_pct, self.difficulty))

        # Save best model whenever mean fraction improves
        if mean_frac > self.best_mean_fraction:
            self.best_mean_fraction = mean_frac
            self.model.save(os.path.join(self.save_path, "best_model"))
            if hasattr(self.training_env, "save"):
                self.training_env.save(os.path.join(self.save_path, "vec_normalize.pkl"))
            if self.verbose:
                print(f"[V3] New best model saved  (mean_frac={mean_frac_pct:.1f}%)")

        # Advance / retreat logic
        if mean_frac > ADVANCE_THRESHOLD:
            self.consec_above += 1
            self.consec_below  = 0
            if self.verbose:
                print(f"[V3] Above advance threshold for {self.consec_above}/{self.stability_window} evals")
        elif mean_frac < RETREAT_THRESHOLD:
            self.consec_below += 1
            self.consec_above  = 0
            if self.verbose:
                print(f"[V3] Below retreat threshold for {self.consec_below}/{RETREAT_CONSEC} evals")
        else:
            self.consec_above = 0
            self.consec_below = 0

        if self.consec_above >= self.stability_window and self.difficulty < self.max_difficulty:
            self.consec_above = 0
            old_diff = self.difficulty
            self.difficulty += 1
            if self.verbose:
                print(f"[V3] *** ADVANCING to difficulty {self.difficulty} ***")
            self.model.save(os.path.join(self.save_path, f"model_diff{old_diff}_complete"))
            self.training_env.set_attr("difficulty", self.difficulty)
            self.transitions.append((self.num_timesteps, old_diff, self.difficulty))

        elif self.consec_below >= RETREAT_CONSEC and self.difficulty > 0:
            self.consec_below = 0
            old_diff = self.difficulty
            self.difficulty -= 1
            if self.verbose:
                print(f"[V3] *** RETREATING to difficulty {self.difficulty} ***")
            self.training_env.set_attr("difficulty", self.difficulty)
            self.transitions.append((self.num_timesteps, old_diff, self.difficulty))

        return True


# ── main training function ─────────────────────────────────────────────────────

def run(
    difficulty=DEFAULT_DIFFICULTY,
    output_folder=DEFAULT_OUTPUT_FOLDER,
    gui=DEFAULT_GUI,
    total_timesteps=DEFAULT_TOTAL_STEPS,
    n_envs=DEFAULT_N_ENVS,
    n_steps=DEFAULT_N_STEPS,
    batch_size=DEFAULT_BATCH_SIZE,
    net_arch=DEFAULT_NET_ARCH,
    load_model=None,
    learning_rate=None,
):
    if isinstance(net_arch, str):
        net_arch_list = [int(x.strip()) for x in net_arch.split(",")]
    else:
        net_arch_list = list(net_arch)

    timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    save_dir  = os.path.join(output_folder, f"obstacle_v3-{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] Saving results to: {save_dir}")

    train_env = make_vec_env(
        ObstacleAviaryV3,
        env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT, difficulty=difficulty),
        n_envs=n_envs,
        seed=0,
    )
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                             clip_obs=5.0, clip_reward=10.0)

    print(f"[INFO] Observation space: {train_env.observation_space}")
    print(f"[INFO] Action space:      {train_env.action_space}")
    print(f"[INFO] Starting difficulty: {difficulty}  (n_envs={n_envs})")

    try:
        import tensorboard  # noqa: F401
        tb_log = os.path.join(save_dir, "tb")
    except ImportError:
        tb_log = None

    if load_model:
        print(f"[INFO] Loading model from checkpoint: {load_model}")
        vec_norm_path = os.path.join(os.path.dirname(load_model), "vec_normalize.pkl")
        if os.path.exists(vec_norm_path):
            print(f"[INFO] Loading VecNormalize stats from: {vec_norm_path}")
            train_env = VecNormalize.load(vec_norm_path, train_env.venv)
            train_env.training = True
        model = PPO.load(load_model, env=train_env, tensorboard_log=tb_log)
        model.target_kl = 0.01
        if learning_rate is not None:
            for g in model.policy.optimizer.param_groups:
                g["lr"] = learning_rate
            print(f"[INFO] Overriding LR → {learning_rate}")
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,
            learning_rate=learning_rate if learning_rate is not None else 3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            target_kl=0.01,
            policy_kwargs=dict(net_arch=net_arch_list),
            verbose=1,
            tensorboard_log=tb_log,
        )

    curriculum_cb = CurriculumCallbackV3(
        eval_freq=1000,
        save_path=save_dir,
        starting_difficulty=difficulty,
        max_difficulty=MAX_DIFFICULTY,
        verbose=1,
        stability_window=ADVANCE_CONSEC,
    )

    print(f"\n[INFO] Training for {int(total_timesteps):,} steps across {n_envs} envs ...\n")
    model.learn(
        total_timesteps=int(total_timesteps),
        callback=curriculum_cb,
        log_interval=50,
    )

    model.save(os.path.join(save_dir, "final_model"))
    train_env.save(os.path.join(save_dir, "vec_normalize_final.pkl"))
    if not os.path.exists(os.path.join(save_dir, "best_model.zip")):
        model.save(os.path.join(save_dir, "best_model"))
        if hasattr(train_env, "save"):
            train_env.save(os.path.join(save_dir, "vec_normalize.pkl"))
    print(f"\n[INFO] Training complete. Models saved to {save_dir}")

    # ── training curve ────────────────────────────────────────────────────────
    if curriculum_cb.plot_data:
        steps    = [d[0] for d in curriculum_cb.plot_data]
        frac_pcts = [d[1] for d in curriculum_cb.plot_data]

        window = min(10, len(frac_pcts))
        rolled = np.convolve(frac_pcts, np.ones(window) / window, mode="valid")
        rolled_steps = steps[window - 1:]

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(steps, frac_pcts, alpha=0.35, linewidth=1, color="steelblue",
                label="Mean fraction_completed (raw %)")
        ax.plot(rolled_steps, rolled, linewidth=2, color="steelblue",
                label=f"Rolling avg (window={window})")
        ax.axhline(ADVANCE_THRESHOLD * 100, color="green", linestyle="--", alpha=0.7,
                   label=f"Advance threshold ({ADVANCE_THRESHOLD:.0%})")
        ax.axhline(RETREAT_THRESHOLD * 100, color="red", linestyle="--", alpha=0.7,
                   label=f"Retreat threshold ({RETREAT_THRESHOLD:.0%})")

        for ts, from_d, to_d in curriculum_cb.transitions:
            direction = "advance" if to_d > from_d else "retreat"
            colour    = "green" if direction == "advance" else "red"
            ax.axvline(ts, color=colour, linestyle=":", alpha=0.8)
            ax.text(ts, 95, f"d{from_d}→d{to_d}", fontsize=7, color=colour,
                    rotation=90, va="top", ha="right")

        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Mean Fraction Completed (%)")
        ax.set_title("ObstacleAviaryV3 — Multi-Waypoint Curriculum Training Curve")
        ax.set_ylim(0, 105)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.4)
        fig.tight_layout()
        curve_path = os.path.join(save_dir, "training_curve_v3.png")
        fig.savefig(curve_path, dpi=150)
        print(f"[INFO] Training curve saved to {curve_path}")
        plt.close(fig)

    # ── quick evaluation at final difficulty ──────────────────────────────────
    best_path = os.path.join(save_dir, "best_model.zip")
    if os.path.exists(best_path):
        model = PPO.load(best_path)

    final_difficulty = curriculum_cb.difficulty
    print(f"\n[INFO] Running 20 evaluation episodes at difficulty={final_difficulty} ...")

    eval_env = ObstacleAviaryV3(obs=DEFAULT_OBS, act=DEFAULT_ACT, difficulty=final_difficulty)
    total_fracs = []
    success_flags = []

    for ep in range(20):
        obs, info = eval_env.reset()
        ep_frac   = 0.0
        ep_success = False
        max_steps = int((eval_env.EPISODE_LEN_SEC + 4) * eval_env.CTRL_FREQ)
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            if terminated or truncated:
                ep_frac    = info.get("fraction_completed", 0.0)
                ep_success = info.get("success", False)
                break
        total_fracs.append(ep_frac)
        success_flags.append(ep_success)
    eval_env.close()

    mean_frac    = float(np.mean(total_fracs))
    success_rate = float(np.mean(success_flags))
    print(f"\n[RESULT] 20-episode evaluation (difficulty={final_difficulty}):")
    print(f"  Mean fraction_completed : {mean_frac:.1%}")
    print(f"  Full success rate       : {success_rate:.1%}  ({int(sum(success_flags))}/20)\n")

    return save_dir


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on ObstacleAviaryV3 (multi-waypoint)."
    )
    parser.add_argument("--difficulty",       default=DEFAULT_DIFFICULTY,    type=int,      metavar="")
    parser.add_argument("--output_folder",    default=DEFAULT_OUTPUT_FOLDER, type=str,      metavar="")
    parser.add_argument("--gui",              default=DEFAULT_GUI,           type=str2bool, metavar="")
    parser.add_argument("--total_timesteps",  default=DEFAULT_TOTAL_STEPS,   type=float,    metavar="")
    parser.add_argument("--n_envs",           default=DEFAULT_N_ENVS,        type=int,      metavar="")
    parser.add_argument("--n_steps",          default=DEFAULT_N_STEPS,       type=int,      metavar="")
    parser.add_argument("--batch_size",       default=DEFAULT_BATCH_SIZE,    type=int,      metavar="")
    parser.add_argument("--net_arch",         default=DEFAULT_NET_ARCH,      type=str,      metavar="")
    parser.add_argument("--load_model",       default=None,                  type=str,      metavar="")
    parser.add_argument("--learning_rate",    default=None,                  type=float,    metavar="")
    args = parser.parse_args()
    run(**vars(args))
