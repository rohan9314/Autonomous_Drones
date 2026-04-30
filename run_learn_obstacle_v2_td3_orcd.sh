#!/bin/bash
#SBATCH --job-name=drone-v2-td3
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --partition=mit_normal

module load deprecated-modules
module load anaconda3/2022.05-x86_64
source activate drones

cd ~/gym-pybullet-drones
mkdir -p logs

python gym_pybullet_drones/examples/learn_obstacle_v2_offpolicy.py \
  --algo td3 \
  --difficulty 1 \
  --n_envs 8 \
  --total_timesteps 8000000 \
  --batch_size 1024 \
  --buffer_size 1000000 \
  --learning_starts 20000 \
  --learning_rate 1e-4 \
  --net_arch 512,512,256 \
  --output_folder ~/drone-results
