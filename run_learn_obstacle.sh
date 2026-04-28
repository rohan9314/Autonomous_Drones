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
    --total_timesteps 6000000 \
    --difficulty 3 \
    --load_model /home/eybo/drone-results/obstacle-04.27.2026_21.37.47/best_model.zip \
    --learning_rate 3e-5 \
    --output_folder ~/drone-results
