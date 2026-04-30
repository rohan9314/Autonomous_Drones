#!/bin/bash
set -euo pipefail

mkdir -p logs

ALGOS=("sac" "td3")
LRS=("2e-4" "1e-4")
SEEDS=("0" "1" "2")

for i in "${!ALGOS[@]}"; do
  algo="${ALGOS[$i]}"
  lr="${LRS[$i]}"
  for seed in "${SEEDS[@]}"; do
    job_name="drone-v3-${algo}-s${seed}"
    sbatch --job-name="${job_name}" \
      --output="logs/%x-%j.out" \
      --error="logs/%x-%j.err" \
      --time=12:00:00 \
      --ntasks=1 \
      --cpus-per-task=8 \
      --mem=24G \
      --partition=mit_normal \
      --wrap="module load deprecated-modules && module load anaconda3/2022.05-x86_64 && source activate drones && cd ~/gym-pybullet-drones && python gym_pybullet_drones/examples/learn_obstacle_v3_offpolicy.py --algo ${algo} --difficulty 1 --n_envs 8 --total_timesteps 12000000 --batch_size 1024 --buffer_size 1000000 --learning_starts 20000 --learning_rate ${lr} --net_arch 512,512,256 --output_folder ~/drone-results"
  done
done

echo "Submitted ${#ALGOS[@]} algorithms x ${#SEEDS[@]} seeds for V3."
