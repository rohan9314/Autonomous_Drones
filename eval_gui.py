import time
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.ObstacleAviary import ObstacleAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync

MODEL_PATH = 'results/obstacle-04.27.2026_21.37.47/best_model.zip'
DIFFICULTY  = 3

print(f'Loading {MODEL_PATH}')
model = PPO.load(MODEL_PATH)

env = ObstacleAviary(
    obs=ObservationType('kin'),
    act=ActionType('pid'),
    difficulty=DIFFICULTY,
    gui=True,
)

obs, info = env.reset(seed=42)
print(f'Running live episode — difficulty {DIFFICULTY} (obstacle dead-centre on path)')
print(f'{"Step":>5}  {"x":>7} {"y":>7} {"z":>7}  {"dist_goal":>9}  {"reward":>8}')

ep_reward = 0.0
start_t   = time.time()
for i in range((env.EPISODE_LEN_SEC + 2) * env.CTRL_FREQ):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    ep_reward += reward

    state = env._getDroneStateVector(0)
    print(f'{i:5d}  {state[0]:7.3f} {state[1]:7.3f} {state[2]:7.3f}  '
          f'{info["dist_to_goal"]:9.3f}  {reward:8.3f}')

    sync(i, start_t, env.CTRL_TIMESTEP)

    if terminated or truncated:
        status = 'SUCCESS' if terminated else 'TIMEOUT'
        print(f'\n[RESULT] {status}  total_reward={ep_reward:.2f}')
        break

time.sleep(3)
env.close()
