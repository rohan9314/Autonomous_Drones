# Obstacle Avoidance Drone — Team Guide

A single drone learns to fly from a random start position to a target (green sphere) while avoiding randomly placed box obstacles. It uses Reinforcement Learning (PPO) and simulated LiDAR to sense its environment.

---

## Setup

### 1. Install Miniconda (one-time)
```sh
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh  # M-chip Mac
bash Miniconda3-latest-MacOSX-arm64.sh
# Close and reopen terminal after install
```

> For Intel Mac replace `arm64` with `x86_64` in the URL.

### 2. Create the environment (one-time)
```sh
conda create -n drones python=3.10
conda activate drones
conda install -c conda-forge pybullet
pip install numpy scipy transforms3d matplotlib gymnasium "stable-baselines3>=2.0" control Pillow tensorboard
cd /path/to/gym-pybullet-drones
pip install -e . --no-deps
```

### 3. Every new terminal session
```sh
conda activate drones
cd /path/to/gym-pybullet-drones
```

---

## Running the trained model

### Watch the drone navigate (GUI)
```sh
/Users/<you>/miniconda3/envs/drones/bin/python3 gym_pybullet_drones/examples/eval_obstacle.py \
  --model results_cluster/obstacle_04.25.2026_14.17.24/best_model.zip
```

### Change number of obstacles
```sh
# 2 obstacles
python3 gym_pybullet_drones/examples/eval_obstacle.py \
  --model results_cluster/obstacle_04.25.2026_14.17.24/best_model.zip \
  --num_obstacles 2

# 6 obstacles
python3 gym_pybullet_drones/examples/eval_obstacle.py \
  --model results_cluster/obstacle_04.25.2026_14.17.24/best_model.zip \
  --num_obstacles 6
```

### Get statistics without GUI (fast)
```sh
python3 gym_pybullet_drones/examples/eval_obstacle.py \
  --model results_cluster/obstacle_04.25.2026_14.17.24/best_model.zip \
  --gui false \
  --episodes 50
```

### All eval options
| Flag | Default | Description |
|---|---|---|
| `--model` | required | Path to best_model.zip |
| `--gui` | true | Show PyBullet window |
| `--episodes` | 10 | Number of episodes to run |
| `--num_obstacles` | 4 | Number of box obstacles |

---

## Performance results (best model, 50 episodes each)

| Obstacles | Success Rate | Collision Rate | Timeout Rate |
|---|---|---|---|
| 2 | 84% | 10% | 6% |
| 4 (trained on) | 50% | 0% | 50% |
| 6 | 70% | 26% | 4% |
| 8 | 46% | 42% | 12% |

---

## Training a new model

### Quick test (local, ~5 min)
```sh
python3 gym_pybullet_drones/examples/train_obstacle.py \
  --timesteps 50000 \
  --num_obstacles 1
```

### Full training (use the cluster — see below)
```sh
python3 gym_pybullet_drones/examples/train_obstacle.py \
  --timesteps 5000000 \
  --num_obstacles 4 \
  --n_envs 4
```

### All training options
| Flag | Default | Description |
|---|---|---|
| `--timesteps` | 5,000,000 | Total training steps |
| `--num_obstacles` | 4 | Obstacles per episode |
| `--n_envs` | 4 | Parallel environments (more = faster) |
| `--output_folder` | results/ | Where to save model |
| `--gui` | false | Show GUI during training (very slow) |

Models are saved to `results/obstacle_<timestamp>/best_model.zip`

---

## Running on the MIT ORCD Cluster (much faster)

### SSH in
```sh
ssh <kerb>@orcd-login.mit.edu
```

### Copy project to cluster (run on your Mac)
```sh
scp -r /path/to/gym-pybullet-drones <kerb>@orcd-login.mit.edu:~/
```

### Install on cluster (first time only)
```sh
module load deprecated-modules
module load anaconda3/2022.05-x86_64
conda create -n drones python=3.10
/home/<kerb>/.conda/envs/drones/bin/pip install pybullet numpy scipy transforms3d matplotlib gymnasium "stable-baselines3>=2.0" control Pillow tensorboard
cd ~/gym-pybullet-drones
/home/<kerb>/.conda/envs/drones/bin/pip install -e . --no-deps
```

### Submit a training job
```sh
cat > train.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=drone_rl
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=train_%j.log

cd ~/gym-pybullet-drones
/home/<kerb>/.conda/envs/drones/bin/python3 gym_pybullet_drones/examples/train_obstacle.py \
  --timesteps 5000000 \
  --num_obstacles 4 \
  --n_envs 16
EOF

sbatch train.sh
```

### Monitor the job
```sh
squeue -u <kerb>          # check status (PD=pending, R=running, CG=done)
tail -f train_<jobid>.log  # stream live output
```

### Copy results back to Mac
```sh
scp -r <kerb>@orcd-login.mit.edu:~/gym-pybullet-drones/results ./results_cluster
```

---

## File structure

```
gym_pybullet_drones/
├── envs/
│   └── ObstacleAviary.py      ← Custom RL environment (edit this to change behavior)
└── examples/
    ├── train_obstacle.py       ← PPO training script
    └── eval_obstacle.py        ← Evaluation and visualization
```

---

## How to customize

### Change the reward function
Edit `_computeReward()` in `ObstacleAviary.py`:

```python
def _computeReward(self):
    dist = distance(drone, target)

    reward = (prev_dist - dist) * 10.0   # progress toward target
    reward -= 0.02                        # time penalty (increase to go faster)

    if dist < 0.5:
        reward += 0.5 * (0.5 - dist)     # proximity bonus (stops orbiting)

    if dist < 0.15:
        reward += 20.0 + early_bonus      # success!
    if collision:
        reward -= 10.0                    # hit obstacle
```

### Change obstacle appearance
Edit `_addObstacles()` in `ObstacleAviary.py`. Change `rgbaColor` for color, `halfExtents` for size:
```python
viz = p.createVisualShape(
    p.GEOM_BOX,
    halfExtents=[0.2, 0.2, 0.4],
    rgbaColor=[0.8, 0.3, 0.1, 1.0],   # [R, G, B, Alpha]
    physicsClientId=self.CLIENT,
)
```

### Change target appearance
Also in `_addObstacles()`, change the target sphere:
```python
vis = p.createVisualShape(
    p.GEOM_SPHERE,
    radius=0.15,
    rgbaColor=[0.1, 0.9, 0.1, 0.7],   # green
    physicsClientId=self.CLIENT,
)
```

### Change arena size
In `ObstacleAviary.py`, modify:
```python
ARENA_HALF = 2.0      # ±2m in x and y
EPISODE_LEN_SEC = 15  # max episode length
TARGET_RADIUS = 0.15  # success distance from target
```

### Change LiDAR
In `_castLidarRays()`, modify:
```python
N_LIDAR_RAYS = 24    # total rays (16 horizontal + 8 downward)
LIDAR_RANGE = 3.0    # max sensing distance in metres
```

### Add rendering / visual effects
In `eval_obstacle.py`, after each `env.step()`:
```python
obs, reward, terminated, truncated, info = env.step(action)

# Add your rendering code here
# e.g. draw debug lines, add text overlays, etc.
p.addUserDebugText(
    f"dist: {info['dist_to_target']:.2f}m",
    textPosition=[0, 0, 2],
    textColorRGB=[1, 1, 1],
    physicsClientId=env.CLIENT
)
```

---

## How it works (quick summary)

**Observation (what the drone sees):**
- Its own position, orientation, velocity (12 values)
- Direction to target (3 values)
- 24 LiDAR rays showing distance to nearest obstacle in each direction
- Last 15 actions it took

**Action (what the drone controls):**
- Velocity direction vector [vx, vy, vz, speed] — all in [-1, 1]
- A PID controller converts this to motor RPMs

**Learning:**
- PPO algorithm runs millions of episodes
- Each episode: random start, random target, random obstacles
- Drone learns from the reward signal over time

**Key insight:** The drone never sees obstacle positions directly — it only senses them through LiDAR, just like a real drone would.
