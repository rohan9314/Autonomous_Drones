import os
import time
from datetime import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.TargetAviary import TargetAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = False
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results_target'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('pid')

TARGET_POS = np.array([0.5, 0.5, 0.5])
OBSTACLE_POSITIONS = np.array([
    [0.25, 0.25, 0.5],
    [0.0,  0.5,  0.3],
])

def run(output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True):

    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    train_env = make_vec_env(TargetAviary,
                             env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT,
                                             TARGET_POS=TARGET_POS,
                                             OBSTACLE_POSITIONS=OBSTACLE_POSITIONS),
                             n_envs=1,
                             seed=0
                             )
    eval_env = TargetAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT,
                            TARGET_POS=TARGET_POS,
                            OBSTACLE_POSITIONS=OBSTACLE_POSITIONS)

    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    model = PPO('MlpPolicy',
                train_env,
                learning_rate=0.0001,
                clip_range=0.1,
                verbose=1)

    target_reward = 180.  # exp(-distance) per step, max ~240, obstacles subtract
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward, verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=False)
    model.learn(total_timesteps=int(1e7) if local else int(1e2),
                callback=eval_callback,
                log_interval=100)

    model.save(filename+'/final_model.zip')
    print(filename)

    with np.load(filename+'/evaluations.npz') as data:
        timesteps = data['timesteps']
        results = data['results'][:, 0]
        print("Data from evaluations.npz")
        for j in range(timesteps.shape[0]):
            print(f"{timesteps[j]},{results[j]}")
        if local:
            plt.plot(timesteps, results, marker='o', linestyle='-', markersize=4)
            plt.xlabel('Training Steps')
            plt.ylabel('Episode Reward')
            plt.grid(True, alpha=0.6)
            plt.savefig(filename+'/reward_plot.png')

    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
        return
    model = PPO.load(path)

    test_env = TargetAviary(gui=gui, obs=DEFAULT_OBS, act=DEFAULT_ACT,
                            TARGET_POS=TARGET_POS,
                            OBSTACLE_POSITIONS=OBSTACLE_POSITIONS,
                            record=record_video)
    test_env_nogui = TargetAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT,
                                  TARGET_POS=TARGET_POS,
                                  OBSTACLE_POSITIONS=OBSTACLE_POSITIONS)

    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                    num_drones=1,
                    output_folder=output_folder,
                    colab=colab)

    mean_reward, std_reward = evaluate_policy(model, test_env_nogui, n_eval_episodes=10)
    print(f"\n\nMean reward {mean_reward} +- {std_reward}\n\n")

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        if DEFAULT_OBS == ObservationType.KIN:
            logger.log(drone=0,
                       timestamp=i/test_env.CTRL_FREQ,
                       state=np.hstack([obs2[0:3], np.zeros(4), obs2[3:15], act2]),
                       control=np.zeros(12))
        test_env.render()
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated or truncated:
            break
    test_env.close()

    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TargetAviary RL training with obstacles')
    parser.add_argument('--gui',           default=DEFAULT_GUI,           type=str2bool, metavar='')
    parser.add_argument('--record_video',  default=DEFAULT_RECORD_VIDEO,  type=str2bool, metavar='')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,      metavar='')
    parser.add_argument('--colab',         default=DEFAULT_COLAB,         type=bool,     metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
