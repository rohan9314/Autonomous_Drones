"""Training script for ObstacleAviary: fly from origin to goal while avoiding obstacles.

Five-phase curriculum:
  difficulty=0  no obstacles, straight flight (learn basic navigation)
  difficulty=1  one obstacle 0.10 m off path (easy, narrow miss)
  difficulty=2  one obstacle 0.05 m off path (nearly centred, small gap)
  difficulty=3  one obstacle dead-centre on path (forced real detour)
  difficulty=4  two obstacles in the corridor, randomised each episode

Usage
-----
    python learn_obstacle.py                        # default settings
    python learn_obstacle.py --difficulty 1         # skip phase 0, start with obstacles
    python learn_obstacle.py --total_timesteps 3e6  # longer training run
    python learn_obstacle.py --gui true             # open PyBullet window for the test rollout
"""

import os
import time
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from gym_pybullet_drones.envs.ObstacleAviary import ObstacleAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync, str2bool

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_OBS            = ObservationType('kin')
DEFAULT_ACT            = ActionType('pid')
DEFAULT_OUTPUT_FOLDER  = 'results'
DEFAULT_GUI            = False
DEFAULT_DIFFICULTY     = 0
DEFAULT_TOTAL_STEPS    = int(2e6)
DEFAULT_N_ENVS         = 4

# Reward thresholds that trigger a difficulty advance.
# Calibrated for progress-based reward with PID action type.
# A perfect episode scores ≈ 1.345 (start→goal Euclidean distance).
# Phase 0 → 1: drone reaches goal reliably in open space          (~82% optimal)
# Phase 1 → 2: reaches goal around the barely-off-path obstacle   (~74% optimal)
# Phase 2 → 3: reaches goal around the nearly-centred obstacle    (~67% optimal)
# Phase 3 → 4: routes around the dead-centre obstacle             (~52% optimal)
CURRICULUM_THRESHOLDS = {0: 1.1, 1: 0.9, 2: 0.8, 3: 0.7}


# ── curriculum callback ────────────────────────────────────────────────────────

class CurriculumCallback(BaseCallback):
    """Evaluates the policy periodically and advances difficulty when ready.

    Design note: this wraps around a single evaluation env (not the training
    VecEnv).  When the threshold is crossed we use `set_attr` to push the new
    difficulty to every parallel training env; they pick it up on their next
    natural episode reset so there is no forced interruption.
    """

    def __init__(self, eval_env, thresholds, eval_freq, save_path, verbose=1,
                 stability_window=3):
        super().__init__(verbose)
        self.eval_env              = eval_env
        self.thresholds            = thresholds   # dict: {difficulty -> reward threshold}
        self.eval_freq             = eval_freq
        self.save_path             = save_path
        self.difficulty            = eval_env.difficulty
        self.best_mean             = -np.inf
        self.eval_results          = []           # [(timestep, mean_reward), ...]
        self.stability_window      = stability_window
        self.evals_above_threshold = 0            # consecutive evals that beat threshold
        self._explore_steps_left   = 0            # steps remaining at boosted entropy

    # called every training step
    def _on_step(self) -> bool:
        # decay entropy boost after a phase transition
        if self._explore_steps_left > 0:
            self._explore_steps_left -= 1
            if self._explore_steps_left == 0:
                self.model.ent_coef = 0.01
                if self.verbose:
                    print("[Curriculum] Entropy boost expired, ent_coef → 0.01")

        if self.n_calls % self.eval_freq != 0:
            return True

        mean_reward, std_reward = evaluate_policy(
            self.model, self.eval_env, n_eval_episodes=10, deterministic=True
        )
        self.eval_results.append((self.num_timesteps, mean_reward))

        if self.verbose:
            print(f"\n[Curriculum] step={self.num_timesteps:,}  "
                  f"difficulty={self.difficulty}  "
                  f"mean_reward={mean_reward:.2f} ± {std_reward:.2f}")

        # save best model so far (also snapshot VecNormalize stats if available)
        if mean_reward > self.best_mean:
            self.best_mean = mean_reward
            self.model.save(os.path.join(self.save_path, "best_model"))
            if hasattr(self.training_env, "save"):
                self.training_env.save(os.path.join(self.save_path, "vec_normalize.pkl"))
            if self.verbose:
                print(f"[Curriculum] New best model saved  (reward={mean_reward:.2f})")

        # check whether to advance difficulty (require stability_window consecutive hits)
        threshold = self.thresholds.get(self.difficulty)
        if threshold is not None and mean_reward > threshold:
            self.evals_above_threshold += 1
            if self.verbose:
                print(f"[Curriculum] Above threshold {threshold} for "
                      f"{self.evals_above_threshold}/{self.stability_window} evals")
        else:
            self.evals_above_threshold = 0   # reset if we dip below threshold

        if (threshold is not None
                and self.evals_above_threshold >= self.stability_window):
            self.evals_above_threshold = 0
            self.difficulty += 1
            if self.verbose:
                print(f"[Curriculum] *** Advancing to difficulty {self.difficulty} ***")

            # push new difficulty to all parallel training envs
            self.training_env.set_attr("difficulty", self.difficulty)

            # update the eval env and reset it so it immediately reflects the change
            self.eval_env.difficulty = self.difficulty
            self.eval_env.reset()

            # checkpoint the model at each phase transition
            self.model.save(
                os.path.join(self.save_path, f"model_phase_{self.difficulty - 1}_complete")
            )

            # Restore LR to base rate; keep Adam state intact so momentum carries over.
            # Entropy boost removed: ent_coef=0.05 caused policy collapse at diff 2
            # by making the policy too stochastic, overwhelming the avoidance gradient.
            for g in self.model.policy.optimizer.param_groups:
                g["lr"] = 1e-4
            if self.verbose:
                print(f"[Curriculum] LR → 1e-04")

        return True  # returning False would stop training early


# ── main training function ─────────────────────────────────────────────────────

def run(
    difficulty=DEFAULT_DIFFICULTY,
    output_folder=DEFAULT_OUTPUT_FOLDER,
    gui=DEFAULT_GUI,
    total_timesteps=DEFAULT_TOTAL_STEPS,
    n_envs=DEFAULT_N_ENVS,
    load_model=None,
):
    # ── output directory ──────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    save_dir  = os.path.join(output_folder, f"obstacle-{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] Saving results to: {save_dir}")

    # ── environments ──────────────────────────────────────────────────────────
    # n_envs parallel envs collect experience simultaneously.
    # Each one is a separate PyBullet instance running headless (no GUI).
    train_env = make_vec_env(
        ObstacleAviary,
        env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT, difficulty=difficulty),
        n_envs=n_envs,
        seed=0,
    )
    # Normalise observations and rewards using running statistics.
    # clip_obs=10 keeps outlier obstacle/goal values bounded;
    # clip_reward=10 keeps the -3 crash penalty from dominating early gradients.
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                             clip_obs=10.0, clip_reward=10.0)

    # a single env just for evaluation (never used for training)
    eval_env = ObstacleAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT, difficulty=difficulty)

    print(f"[INFO] Observation space: {train_env.observation_space}")
    print(f"[INFO] Action space:      {train_env.action_space}")
    print(f"[INFO] Starting difficulty: {difficulty}")

    # ── PPO model ─────────────────────────────────────────────────────────────
    # MlpPolicy = fully-connected neural net (2 hidden layers of 64 by default).
    # n_steps: how many steps each env collects before a gradient update.
    # batch_size: mini-batch size for the gradient step.
    # n_epochs: how many times we iterate over the collected data per update.
    try:
        import tensorboard  # noqa: F401
        tb_log = os.path.join(save_dir, "tb")
    except ImportError:
        tb_log = None
        print("[WARN] tensorboard not installed — skipping TB logging. Install with: pip install tensorboard")

    if load_model:
        print(f"[INFO] Loading model from checkpoint: {load_model}")
        vec_norm_path = os.path.join(os.path.dirname(load_model), "vec_normalize.pkl")
        if os.path.exists(vec_norm_path):
            print(f"[INFO] Loading VecNormalize stats from: {vec_norm_path}")
            train_env = VecNormalize.load(vec_norm_path, train_env.venv)
            train_env.training = True
        model = PPO.load(load_model, env=train_env, tensorboard_log=tb_log)
        model.target_kl = 0.01
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            n_steps=512,
            batch_size=128,
            n_epochs=10,
            learning_rate=1e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            target_kl=0.01,
            verbose=1,
            tensorboard_log=tb_log,
        )

    # ── curriculum callback ───────────────────────────────────────────────────
    # eval_freq=2000 means we evaluate every 2000 *calls to _on_step*,
    # which equals 2000 * n_envs = 8000 environment steps.
    curriculum_cb = CurriculumCallback(
        eval_env=eval_env,
        thresholds=CURRICULUM_THRESHOLDS,
        eval_freq=2000,
        save_path=save_dir,
        verbose=1,
    )

    # ── train ─────────────────────────────────────────────────────────────────
    print(f"\n[INFO] Training for {total_timesteps:,} steps across {n_envs} envs ...\n")
    model.learn(
        total_timesteps=int(total_timesteps),
        callback=curriculum_cb,
        log_interval=50,      # print SB3's own loss/entropy stats every 50 updates
    )

    # save final weights regardless of whether best_model was updated
    model.save(os.path.join(save_dir, "final_model"))
    # if the callback never fired (very short runs), promote final to best
    if not os.path.exists(os.path.join(save_dir, "best_model.zip")):
        model.save(os.path.join(save_dir, "best_model"))
    print(f"\n[INFO] Training complete. Models saved to {save_dir}")

    # ── training curve ────────────────────────────────────────────────────────
    if curriculum_cb.eval_results:
        steps, rewards = zip(*curriculum_cb.eval_results)
        plt.figure(figsize=(9, 4))
        plt.plot(steps, rewards, marker="o", markersize=3, linewidth=1)
        colours = ["orange", "goldenrod", "red", "darkred"]
        labels  = ["phase 0→1", "phase 1→2", "phase 2→3", "phase 3→4"]
        for (phase, thresh), colour, label in zip(CURRICULUM_THRESHOLDS.items(), colours, labels):
            plt.axhline(thresh, color=colour, linestyle="--",
                        label=f"{label} threshold ({thresh})")
        plt.xlabel("Training Steps")
        plt.ylabel("Mean Episode Reward (5 eval episodes)")
        plt.title("Obstacle Avoidance — Curriculum Training Curve")
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        curve_path = os.path.join(save_dir, "training_curve.png")
        plt.savefig(curve_path, dpi=150)
        print(f"[INFO] Training curve saved to {curve_path}")
        plt.close()

    # ── load best model and evaluate ─────────────────────────────────────────
    best_path = os.path.join(save_dir, "best_model.zip")
    if os.path.exists(best_path):
        print(f"\n[INFO] Loading best model from {best_path}")
        model = PPO.load(best_path)
    else:
        print("[WARN] No best_model.zip found; using final model weights.")

    # quantitative eval: 10 episodes, no rendering
    eval_env_nogui = ObstacleAviary(
        obs=DEFAULT_OBS, act=DEFAULT_ACT, difficulty=curriculum_cb.difficulty
    )
    mean_r, std_r = evaluate_policy(model, eval_env_nogui, n_eval_episodes=10)
    print(f"\n[RESULT] Mean reward over 10 episodes: {mean_r:.2f} ± {std_r:.2f}\n")
    eval_env_nogui.close()

    # ── test rollout ──────────────────────────────────────────────────────────
    # one episode with verbose step-by-step output so you can see what the
    # trained policy is actually doing
    test_env = ObstacleAviary(
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
        difficulty=curriculum_cb.difficulty,
        gui=gui,
    )
    obs, info = test_env.reset(seed=0)
    print("[INFO] Running one test episode (step-by-step output):")
    print(f"{'Step':>5}  {'drone_x':>8} {'drone_y':>8} {'drone_z':>8}  "
          f"{'dist_goal':>9}  {'reward':>8}  {'done':>5}")

    ep_reward = 0.0
    start_t   = time.time()
    for i in range((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        ep_reward += reward

        state = test_env._getDroneStateVector(0)
        done  = terminated or truncated
        print(f"{i:5d}  {state[0]:8.3f} {state[1]:8.3f} {state[2]:8.3f}  "
              f"{info['dist_to_goal']:9.3f}  {reward:8.3f}  {str(done):>5}")

        if gui:
            sync(i, start_t, test_env.CTRL_TIMESTEP)

        if done:
            status = "SUCCESS" if terminated else "FAIL (crash/timeout)"
            print(f"\n[TEST] Episode ended — {status}  total_reward={ep_reward:.2f}\n")
            break

    test_env.close()
    return save_dir


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on ObstacleAviary with curriculum learning."
    )
    parser.add_argument("--difficulty",        default=DEFAULT_DIFFICULTY,    type=int,      metavar="")
    parser.add_argument("--output_folder",     default=DEFAULT_OUTPUT_FOLDER, type=str,      metavar="")
    parser.add_argument("--gui",               default=DEFAULT_GUI,           type=str2bool, metavar="")
    parser.add_argument("--total_timesteps",   default=DEFAULT_TOTAL_STEPS,   type=float,    metavar="")
    parser.add_argument("--n_envs",            default=DEFAULT_N_ENVS,        type=int,      metavar="")
    parser.add_argument("--load_model",        default=None,                  type=str,      metavar="")
    args = parser.parse_args()
    run(**vars(args))
