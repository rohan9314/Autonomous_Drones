"""Training script for ObstacleAviaryV2: drone navigation with lidar perception.

Key differences from v1:
  - Uses ObstacleAviaryV2 with 48-ray lidar and procedural randomized obstacles
  - Success-rate curriculum (advance at 70%, retreat at 35%) instead of reward threshold
  - 16 parallel envs, n_steps=2048, deeper MLP [256,256,128]
  - No entropy boost on phase transition

Usage:
    python learn_obstacle_v2.py                            # default: 5M steps, diff 1
    python learn_obstacle_v2.py --total_timesteps 10e6     # longer run
    python learn_obstacle_v2.py --difficulty 2             # start at diff 2
    python learn_obstacle_v2.py --load_model path/best_model.zip --learning_rate 1e-4
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
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from gym_pybullet_drones.envs.ObstacleAviaryV2 import ObstacleAviaryV2
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync, str2bool

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_OBS            = ObservationType('kin')
DEFAULT_ACT            = ActionType('pid')
DEFAULT_OUTPUT_FOLDER  = 'results'
DEFAULT_GUI            = False
DEFAULT_DIFFICULTY     = 1       # start at difficulty 1 (min 1-3 obstacles)
DEFAULT_TOTAL_STEPS    = int(5e6)
DEFAULT_N_ENVS         = 16      # more parallel envs for diversity
DEFAULT_N_STEPS        = 2048    # more steps per update
DEFAULT_BATCH_SIZE     = 512
DEFAULT_NET_ARCH       = "256,256,128"

# Curriculum thresholds
ADVANCE_THRESHOLD  = 0.70   # success rate required to advance difficulty
RETREAT_THRESHOLD  = 0.35   # success rate that triggers a retreat
ADVANCE_CONSEC     = 3      # consecutive eval periods above advance threshold
RETREAT_CONSEC     = 2      # consecutive eval periods below retreat threshold
MAX_DIFFICULTY     = 4


# ── curriculum callback ────────────────────────────────────────────────────────

class CurriculumCallbackV2(BaseCallback):
    """Success-rate based curriculum callback for ObstacleAviaryV2.

    Tracks episode success/failure from the training envs directly (no separate
    eval env). Success is defined as info["success"] == True at episode end,
    which the environment sets when the drone reaches the goal without crashing.

    Curriculum rules:
      - Advance difficulty when success_rate > 0.70 for 3 consecutive checks.
      - Retreat difficulty when success_rate < 0.35 for 2 consecutive checks
        (minimum difficulty = 1).
    """

    def __init__(self, eval_freq, save_path, max_difficulty=MAX_DIFFICULTY,
                 verbose=1, stability_window=ADVANCE_CONSEC):
        super().__init__(verbose)
        self.eval_freq        = eval_freq
        self.save_path        = save_path
        self.max_difficulty   = max_difficulty
        self.stability_window = stability_window

        self.difficulty       = 1
        self.best_success_rate = 0.0

        # deque of 1 (success) / 0 (fail) for recent episodes across all envs
        self.episode_outcomes = deque(maxlen=200)

        # consecutive eval counters
        self.consec_above     = 0
        self.consec_below     = 0

        # for the training curve plot: (timestep, success_rate_pct, difficulty)
        self.plot_data        = []

        # track difficulty transition timesteps for vertical plot lines
        self.transitions      = []   # [(timestep, from_diff, to_diff), ...]

    def _on_step(self) -> bool:
        # Record episode outcomes from all parallel envs.
        # dones is a numpy array of shape (n_envs,); infos is a list of dicts.
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for done, info in zip(dones, infos):
            if done:
                success = info.get("success", False)
                self.episode_outcomes.append(1 if success else 0)

        # Only run the periodic evaluation logic at the right cadence.
        if self.n_calls % self.eval_freq != 0:
            return True

        # Need a minimum sample before making curriculum decisions.
        if len(self.episode_outcomes) < 20:
            if self.verbose:
                print(f"[V2] step={self.num_timesteps:,}  diff={self.difficulty}  "
                      f"waiting for episodes ... ({len(self.episode_outcomes)}/20 so far)")
            return True

        success_rate = float(np.mean(self.episode_outcomes))
        success_pct  = success_rate * 100.0
        n_tracked    = len(self.episode_outcomes)

        if self.verbose:
            print(f"[V2] step={self.num_timesteps:,}  diff={self.difficulty}  "
                  f"success_rate={success_pct:.1f}%  ({n_tracked} episodes tracked)")

        # Record for plot
        self.plot_data.append((self.num_timesteps, success_pct, self.difficulty))

        # Save best model whenever we hit a new best success rate
        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
            self.model.save(os.path.join(self.save_path, "best_model"))
            if hasattr(self.training_env, "save"):
                self.training_env.save(os.path.join(self.save_path, "vec_normalize.pkl"))
            if self.verbose:
                print(f"[V2] New best model saved  (success_rate={success_pct:.1f}%)")

        # ── advance / retreat logic ────────────────────────────────────────────
        if success_rate > ADVANCE_THRESHOLD:
            self.consec_above += 1
            self.consec_below  = 0
            if self.verbose:
                print(f"[V2] Above advance threshold {ADVANCE_THRESHOLD:.0%} for "
                      f"{self.consec_above}/{self.stability_window} evals")
        elif success_rate < RETREAT_THRESHOLD:
            self.consec_below += 1
            self.consec_above  = 0
            if self.verbose:
                print(f"[V2] Below retreat threshold {RETREAT_THRESHOLD:.0%} for "
                      f"{self.consec_below}/{RETREAT_CONSEC} evals")
        else:
            # In the middle band — reset both counters
            self.consec_above = 0
            self.consec_below = 0

        # Advance
        if self.consec_above >= self.stability_window and self.difficulty < self.max_difficulty:
            self.consec_above = 0
            old_diff = self.difficulty
            self.difficulty += 1
            if self.verbose:
                print(f"[V2] *** ADVANCING to difficulty {self.difficulty} ***")

            # Checkpoint model at the completed phase
            self.model.save(
                os.path.join(self.save_path, f"model_diff{old_diff}_complete")
            )

            # Push new difficulty to all training envs (they pick it up on next reset)
            self.training_env.set_attr("difficulty", self.difficulty)

            # Record transition for plot
            self.transitions.append((self.num_timesteps, old_diff, self.difficulty))

        # Retreat
        elif self.consec_below >= RETREAT_CONSEC and self.difficulty > 1:
            self.consec_below = 0
            old_diff = self.difficulty
            self.difficulty -= 1
            if self.verbose:
                print(f"[V2] *** RETREATING to difficulty {self.difficulty} ***")

            self.training_env.set_attr("difficulty", self.difficulty)

            # Record transition for plot
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
    # ── parse net_arch ────────────────────────────────────────────────────────
    if isinstance(net_arch, str):
        net_arch_list = [int(x.strip()) for x in net_arch.split(",")]
    else:
        net_arch_list = list(net_arch)

    # ── output directory ──────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    save_dir  = os.path.join(output_folder, f"obstacle_v2-{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] Saving results to: {save_dir}")

    # ── training environments ─────────────────────────────────────────────────
    # 16 parallel envs provide diverse experience and reduce variance.
    train_env = make_vec_env(
        ObstacleAviaryV2,
        env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT, difficulty=difficulty),
        n_envs=n_envs,
        seed=0,
    )
    # Normalise observations and rewards using running statistics.
    # clip_obs=5.0: lidar distances are bounded [0,1], so a tight clip avoids
    # inflating the normalisation range with rare out-of-range values.
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                             clip_obs=5.0, clip_reward=10.0)

    print(f"[INFO] Observation space: {train_env.observation_space}")
    print(f"[INFO] Action space:      {train_env.action_space}")
    print(f"[INFO] Starting difficulty: {difficulty}")
    print(f"[INFO] n_envs={n_envs}  n_steps={n_steps}  batch_size={batch_size}")
    print(f"[INFO] net_arch={net_arch_list}")

    # ── TensorBoard log dir ───────────────────────────────────────────────────
    try:
        import tensorboard  # noqa: F401
        tb_log = os.path.join(save_dir, "tb")
    except ImportError:
        tb_log = None
        print("[WARN] tensorboard not installed — skipping TB logging. "
              "Install with: pip install tensorboard")

    # ── PPO model ─────────────────────────────────────────────────────────────
    if load_model:
        print(f"[INFO] Loading model from checkpoint: {load_model}")

        # Try to detect obs space mismatch between checkpoint and current env
        try:
            import zipfile, json
            with zipfile.ZipFile(load_model, "r") as zf:
                if "data" in zf.namelist():
                    with zf.open("data") as f:
                        ckpt_data = json.loads(f.read().decode("utf-8"))
                    saved_obs_dim = ckpt_data.get("observation_space", {}).get("n", None)
                    current_obs_dim = train_env.observation_space.shape[0]
                    if saved_obs_dim is not None and saved_obs_dim != current_obs_dim:
                        print(f"[WARN] Obs space mismatch: checkpoint has dim={saved_obs_dim}, "
                              f"current env has dim={current_obs_dim}. "
                              f"This may be a v1 checkpoint (dim=66) loaded into v2 (dim>=108).")
        except Exception:
            pass  # if introspection fails, just continue loading

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

    # ── curriculum callback ───────────────────────────────────────────────────
    # eval_freq=1000 → check every 1000 _on_step calls.
    # Each call processes one rollout step across all envs simultaneously,
    # so 1000 checks = 1000 * n_envs environment steps = 16,000 env steps.
    curriculum_cb = CurriculumCallbackV2(
        eval_freq=1000,
        save_path=save_dir,
        max_difficulty=MAX_DIFFICULTY,
        verbose=1,
        stability_window=ADVANCE_CONSEC,
    )
    # Initialise callback's difficulty to match the starting difficulty arg
    curriculum_cb.difficulty = difficulty

    # ── train ─────────────────────────────────────────────────────────────────
    print(f"\n[INFO] Training for {int(total_timesteps):,} steps across {n_envs} envs ...\n")
    model.learn(
        total_timesteps=int(total_timesteps),
        callback=curriculum_cb,
        log_interval=50,
    )

    # Save final weights
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
        sr_pcts  = [d[1] for d in curriculum_cb.plot_data]
        diffs    = [d[2] for d in curriculum_cb.plot_data]

        # rolling average (window=10 eval points) for smoother curve
        window = min(10, len(sr_pcts))
        rolled = np.convolve(sr_pcts, np.ones(window) / window, mode="valid")
        rolled_steps = steps[window - 1:]

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(steps, sr_pcts, alpha=0.35, linewidth=1, color="steelblue",
                label="Success rate (raw %)")
        ax.plot(rolled_steps, rolled, linewidth=2, color="steelblue",
                label=f"Rolling avg (window={window})")

        # Horizontal threshold lines
        ax.axhline(ADVANCE_THRESHOLD * 100, color="green", linestyle="--", alpha=0.7,
                   label=f"Advance threshold ({ADVANCE_THRESHOLD:.0%})")
        ax.axhline(RETREAT_THRESHOLD * 100, color="red", linestyle="--", alpha=0.7,
                   label=f"Retreat threshold ({RETREAT_THRESHOLD:.0%})")

        # Vertical lines at difficulty transitions
        transition_colours = {
            "advance": "green",
            "retreat": "red",
        }
        for ts, from_d, to_d in curriculum_cb.transitions:
            direction = "advance" if to_d > from_d else "retreat"
            colour = transition_colours[direction]
            ax.axvline(ts, color=colour, linestyle=":", alpha=0.8)
            ax.text(ts, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 95,
                    f"d{from_d}→d{to_d}", fontsize=7, color=colour,
                    rotation=90, va="top", ha="right")

        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Episode Success Rate (%)")
        ax.set_title("ObstacleAviaryV2 — Success-Rate Curriculum Training Curve")
        ax.set_ylim(0, 105)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.4)
        fig.tight_layout()
        curve_path = os.path.join(save_dir, "training_curve_v2.png")
        fig.savefig(curve_path, dpi=150)
        print(f"[INFO] Training curve saved to {curve_path}")
        plt.close(fig)

    # ── load best model for evaluation ───────────────────────────────────────
    best_path = os.path.join(save_dir, "best_model.zip")
    if os.path.exists(best_path):
        print(f"\n[INFO] Loading best model from {best_path}")
        model = PPO.load(best_path)
    else:
        print("[WARN] No best_model.zip found; using final model weights.")

    final_difficulty = curriculum_cb.difficulty

    # ── 20-episode quantitative evaluation ───────────────────────────────────
    print(f"\n[INFO] Running 20 evaluation episodes at difficulty={final_difficulty} ...")
    eval_env = ObstacleAviaryV2(
        obs=DEFAULT_OBS, act=DEFAULT_ACT, difficulty=final_difficulty
    )

    total_rewards   = []
    success_flags   = []
    steps_to_goal   = []

    for ep in range(20):
        obs, info = eval_env.reset()
        ep_reward  = 0.0
        ep_steps   = 0
        ep_success = False
        max_steps  = (eval_env.EPISODE_LEN_SEC + 2) * eval_env.CTRL_FREQ
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += reward
            ep_steps  += 1
            if terminated or truncated:
                ep_success = info.get("success", False)
                break
        total_rewards.append(ep_reward)
        success_flags.append(ep_success)
        if ep_success:
            steps_to_goal.append(ep_steps)

    eval_env.close()

    mean_reward  = float(np.mean(total_rewards))
    std_reward   = float(np.std(total_rewards))
    success_rate = float(np.mean(success_flags))
    mean_steps   = float(np.mean(steps_to_goal)) if steps_to_goal else float("nan")

    print(f"\n[RESULT] 20-episode evaluation (difficulty={final_difficulty}):")
    print(f"  Mean episode reward : {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Success rate        : {success_rate:.1%}  ({int(sum(success_flags))}/20 episodes)")
    print(f"  Mean steps to goal  : {mean_steps:.1f}  (successful episodes only)\n")

    # ── step-by-step test rollout ─────────────────────────────────────────────
    test_env = ObstacleAviaryV2(
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
        difficulty=final_difficulty,
        gui=gui,
    )
    obs, info = test_env.reset(seed=42)

    # Heuristic: base kinematic state is 12 values; lidar follows immediately.
    # ObstacleAviaryV2 observation layout: [kin_state (12) | lidar (48) | goal (3) | ...]
    # We try to infer base_dim from the observation space and fall back to 12.
    obs_dim   = obs.shape[0]
    base_dim  = 12   # standard kin obs from MultirotorRateLaw or similar

    print("[INFO] Running one test episode (step-by-step output):")
    print(f"{'Step':>5}  {'x':>7} {'y':>7} {'z':>7}  "
          f"{'dist_goal':>9}  {'min_lidar':>9}  {'reward':>8}  {'done':>5}")

    ep_reward = 0.0
    start_t   = time.time()
    max_steps  = (test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ

    for i in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        ep_reward += reward
        done = terminated or truncated

        state = test_env._getDroneStateVector(0)

        # Parse min lidar distance from obs vector
        lidar_end = min(base_dim + 48, obs_dim)
        if base_dim < obs_dim:
            min_lidar = float(np.min(obs[base_dim:lidar_end]))
        else:
            min_lidar = float("nan")

        print(f"{i:5d}  {state[0]:7.3f} {state[1]:7.3f} {state[2]:7.3f}  "
              f"{info.get('dist_to_goal', float('nan')):9.3f}  "
              f"{min_lidar:9.3f}  "
              f"{reward:8.3f}  {str(done):>5}")

        if gui:
            sync(i, start_t, test_env.CTRL_TIMESTEP)

        if done:
            ep_success = info.get("success", False)
            status = "SUCCESS" if ep_success else "FAIL (crash/timeout)"
            print(f"\n[TEST] Episode ended — {status}  total_reward={ep_reward:.2f}\n")
            break

    test_env.close()
    return save_dir


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on ObstacleAviaryV2 with success-rate curriculum."
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
