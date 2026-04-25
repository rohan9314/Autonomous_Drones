#!/bin/bash
#SBATCH --job-name=drone-obstacle
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=mit_normal

module load deprecated-modules
module load anaconda3/2022.05-x86_64
source activate drones

cd ~/gym-pybullet-drones
mkdir -p logs

python gym_pybullet_drones/examples/learn_obstacle.py \
    --n_envs 8 \
    --total_timesteps 5e6 \
    --output_folder ~/drone-results
