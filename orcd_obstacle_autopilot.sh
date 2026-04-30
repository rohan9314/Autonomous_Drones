#!/bin/bash
set -euo pipefail

# ORCD autopilot:
# 1) submit SAC/TD3 sweeps
# 2) poll until all jobs finish
# 3) run leaderboard evaluation
# 4) write best-checkpoint pointer JSON for deployment/testing

TASK="v3"
RESULTS_DIR="${HOME}/drone-results"
DIFFICULTY=3
N_EPISODES=30
TOP_K=10
POLL_SECS=60
SEEDS="0 1 2"
DRY_RUN=0
PROMOTE_BEST=0
DEPLOY_ROOT="${RESULTS_DIR}/deploy"

usage() {
  cat <<EOF
Usage: $0 [options]
  --task v2|v3
  --results_dir PATH
  --difficulty INT
  --n_episodes INT
  --top_k INT
  --poll_secs INT
  --seeds "0 1 2"
  --dry_run true|false
  --promote_best true|false
  --deploy_root PATH
EOF
}

to_bool() {
  local v
  v="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  case "${v}" in
    1|true|yes|y) echo "1" ;;
    0|false|no|n) echo "0" ;;
    *) echo "Invalid boolean: $1" >&2; exit 1 ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --results_dir) RESULTS_DIR="$2"; shift 2 ;;
    --difficulty) DIFFICULTY="$2"; shift 2 ;;
    --n_episodes) N_EPISODES="$2"; shift 2 ;;
    --top_k) TOP_K="$2"; shift 2 ;;
    --poll_secs) POLL_SECS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --dry_run) DRY_RUN="$(to_bool "$2")"; shift 2 ;;
    --promote_best) PROMOTE_BEST="$(to_bool "$2")"; shift 2 ;;
    --deploy_root) DEPLOY_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "$TASK" != "v2" && "$TASK" != "v3" ]]; then
  echo "--task must be v2 or v3" >&2
  exit 1
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found. Run this script on ORCD login node." >&2
  exit 1
fi
if ! command -v squeue >/dev/null 2>&1; then
  echo "squeue not found. Run this script on ORCD login node." >&2
  exit 1
fi

mkdir -p logs
mkdir -p "${RESULTS_DIR}"
AUTOPILOT_DIR="${RESULTS_DIR}/autopilot_${TASK}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${AUTOPILOT_DIR}"
JOB_IDS_FILE="${AUTOPILOT_DIR}/job_ids.txt"
SUBMISSION_LOG="${AUTOPILOT_DIR}/submission.log"
POLL_LOG="${AUTOPILOT_DIR}/poll.log"

ALGOS=("sac" "td3")
if [[ "$TASK" == "v2" ]]; then
  TIMESTEPS=8000000
else
  TIMESTEPS=12000000
fi

echo "[INFO] Autopilot dir: ${AUTOPILOT_DIR}" | tee -a "${SUBMISSION_LOG}"
echo "[INFO] Submitting ${TASK} sweep: algos=${ALGOS[*]} seeds=${SEEDS}" | tee -a "${SUBMISSION_LOG}"

submit_one() {
  local algo="$1"
  local lr="$2"
  local seed="$3"
  local job_name="drone-${TASK}-${algo}-s${seed}"
  local train_script="gym_pybullet_drones/examples/learn_obstacle_${TASK}_offpolicy.py"
  local cmd="module load deprecated-modules && module load anaconda3/2022.05-x86_64 && source activate drones && cd ~/gym-pybullet-drones && python ${train_script} --algo ${algo} --difficulty 1 --n_envs 8 --total_timesteps ${TIMESTEPS} --batch_size 1024 --buffer_size 1000000 --learning_starts 20000 --learning_rate ${lr} --net_arch 512,512,256 --output_folder ${RESULTS_DIR} && echo AUTOPILOT_DONE"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] sbatch --job-name=${job_name} --wrap='${cmd}'" | tee -a "${SUBMISSION_LOG}"
    return 0
  fi

  out=$(sbatch --job-name="${job_name}" \
    --output="logs/%x-%j.out" \
    --error="logs/%x-%j.err" \
    --time=24:00:00 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=24G \
    --partition=mit_normal \
    --wrap="${cmd}")

  echo "${out}" | tee -a "${SUBMISSION_LOG}"
  job_id=$(echo "${out}" | awk '{print $4}')
  if [[ -z "${job_id}" ]]; then
    echo "[ERROR] Could not parse job ID from sbatch output: ${out}" >&2
    exit 1
  fi
  echo "${job_id}" >> "${JOB_IDS_FILE}"
}

for algo in "${ALGOS[@]}"; do
  if [[ "$algo" == "sac" ]]; then
    lr="2e-4"
  else
    lr="1e-4"
  fi
  for seed in ${SEEDS}; do
    submit_one "${algo}" "${lr}" "${seed}"
  done
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[INFO] Dry run complete. No jobs submitted."
  exit 0
fi

echo "[INFO] Submitted jobs:"
cat "${JOB_IDS_FILE}"

all_done=0
while [[ "${all_done}" == "0" ]]; do
  remaining=0
  ts="$(date +%Y-%m-%dT%H:%M:%S)"
  while read -r jid; do
    [[ -z "${jid}" ]] && continue
    if squeue -j "${jid}" -h | rg -q .; then
      remaining=$((remaining + 1))
    fi
  done < "${JOB_IDS_FILE}"

  echo "[${ts}] remaining_jobs=${remaining}" | tee -a "${POLL_LOG}"
  if [[ "${remaining}" -eq 0 ]]; then
    all_done=1
    break
  fi
  sleep "${POLL_SECS}"
done

echo "[INFO] All jobs completed. Running leaderboard evaluation..."
python gym_pybullet_drones/examples/eval_obstacle_runs.py \
  --results_dir "${RESULTS_DIR}" \
  --task "${TASK}" \
  --difficulty "${DIFFICULTY}" \
  --n_episodes "${N_EPISODES}" \
  --top_k "${TOP_K}" \
  --seed 0 \
  --output_json true

LEADERBOARD_JSON="${RESULTS_DIR}/obstacle_${TASK}_leaderboard.json"
POINTER_JSON="${RESULTS_DIR}/best_checkpoint_${TASK}.json"
if [[ ! -f "${LEADERBOARD_JSON}" ]]; then
  echo "[ERROR] Expected leaderboard file not found: ${LEADERBOARD_JSON}" >&2
  exit 1
fi

python - "${LEADERBOARD_JSON}" "${POINTER_JSON}" "${AUTOPILOT_DIR}" <<'PY'
import json
import os
import sys
from datetime import datetime

leaderboard_path, pointer_path, autopilot_dir = sys.argv[1:4]
with open(leaderboard_path, "r", encoding="utf-8") as f:
    rows = json.load(f)
if not rows:
    raise SystemExit(f"No rows in leaderboard: {leaderboard_path}")

best = rows[0]
payload = {
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "leaderboard_json": leaderboard_path,
    "autopilot_run_dir": autopilot_dir,
    "best": {
        "task": best.get("task"),
        "algo": best.get("algo"),
        "model_path": best.get("model_path"),
        "run_dir": best.get("run_dir"),
        "difficulty": best.get("difficulty"),
        "n_episodes": best.get("n_episodes"),
        "score": best.get("score"),
        "success_rate": best.get("success_rate"),
        "crash_rate": best.get("crash_rate"),
        "timeout_rate": best.get("timeout_rate"),
        "mean_reward": best.get("mean_reward"),
        "mean_fraction_completed": best.get("mean_fraction_completed"),
    },
}

os.makedirs(os.path.dirname(pointer_path), exist_ok=True)
with open(pointer_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
print(f"[INFO] Wrote pointer file: {pointer_path}")
print(f"[INFO] Best model: {best.get('model_path')}")
PY

echo "[INFO] Autopilot complete."

if [[ "${PROMOTE_BEST}" == "1" ]]; then
  echo "[INFO] Promoting best model to stable deploy path..."
  python - "${POINTER_JSON}" "${DEPLOY_ROOT}" "${TASK}" <<'PY'
import json
import os
import shutil
import sys
from datetime import datetime

pointer_path, deploy_root, task = sys.argv[1:4]
with open(pointer_path, "r", encoding="utf-8") as f:
    payload = json.load(f)

best = payload.get("best", {})
model_path = best.get("model_path")
run_dir = best.get("run_dir")
if not model_path or not os.path.exists(model_path):
    raise SystemExit(f"Best model not found: {model_path}")
if not run_dir or not os.path.isdir(run_dir):
    raise SystemExit(f"Best run dir not found: {run_dir}")

vec_candidates = [
    os.path.join(run_dir, "vec_normalize.pkl"),
    os.path.join(run_dir, "vec_normalize_final.pkl"),
]
vec_path = next((p for p in vec_candidates if os.path.exists(p)), None)

task_root = os.path.join(deploy_root, task)
staging = os.path.join(task_root, "current.tmp")
current = os.path.join(task_root, "current")
os.makedirs(staging, exist_ok=True)
os.makedirs(task_root, exist_ok=True)

model_dst = os.path.join(staging, "best_model.zip")
shutil.copy2(model_path, model_dst)

if vec_path:
    vec_dst = os.path.join(staging, "vec_normalize.pkl")
    shutil.copy2(vec_path, vec_dst)
else:
    print("[WARN] No VecNormalize stats found for best model.")

manifest = {
    "promoted_at": datetime.utcnow().isoformat() + "Z",
    "source_pointer": pointer_path,
    "source_model_path": model_path,
    "source_run_dir": run_dir,
    "copied_files": ["best_model.zip"] + (["vec_normalize.pkl"] if vec_path else []),
    "task": task,
}
with open(os.path.join(staging, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

if os.path.exists(current):
    backup = os.path.join(task_root, f"previous_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
    os.replace(current, backup)
os.replace(staging, current)
print(f"[INFO] Promoted deploy path: {current}")
print(f"[INFO] Deploy model: {os.path.join(current, 'best_model.zip')}")
PY
fi
