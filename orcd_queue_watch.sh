#!/bin/bash
# Poll ORCD until your SLURM queue is empty. Uses orcd-login.mit.edu (recommended); login00x
# nodes often break non-interactive SSH exec from automation.
set -euo pipefail

ORCD_USER="${ORCD_USER:-eybo}"
ORCD_SSH_HOST="${ORCD_SSH_HOST:-orcd-login.mit.edu}"
POLL_SECS="${POLL_SECS:-180}"
MAX_HOURS="${MAX_HOURS:-48}"

usage() {
  echo "Usage: ORCD_USER=$ORCD_USER ORCD_SSH_HOST=$ORCD_SSH_HOST $0"
  echo "  Env: POLL_SECS (default 180), MAX_HOURS (default 48)"
}

while [[ "${1:-}" == -h || "${1:-}" == --help ]]; do usage; exit 0; shift; done

SSH_CMD=(
  ssh -o BatchMode=yes -o ConnectTimeout=25
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
  "${ORCD_USER}@${ORCD_SSH_HOST}"
)

deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))
echo "[watch] host=${ORCD_SSH_HOST} user=${ORCD_USER} poll=${POLL_SECS}s max=${MAX_HOURS}h"

while true; do
  now=$(date +%s)
  if (( now >= deadline )); then
    echo "[watch] TIMEOUT after ${MAX_HOURS}h — queue:"
    "${SSH_CMD[@]}" squeue -u "${ORCD_USER}" || true
    exit 0
  fi
  # SLURM lines only; wc -l counts running/pending jobs for this user
  if ! out="$("${SSH_CMD[@]}" bash -lc "squeue -u '${ORCD_USER}' -h 2>/dev/null | wc -l | tr -d ' '" 2>&1)"; then
    echo "[watch] $(date -Is) ssh/squeue error: ${out}"
    sleep 60
    continue
  fi
  n="${out//$'\r'/}"
  n="${n//[^0-9]/}"
  n="${n:-0}"
  echo "[watch] $(date -Is) active_jobs=${n}"
  if [[ "${n}" -eq 0 ]]; then
    echo "[watch] QUEUE_EMPTY"
    "${SSH_CMD[@]}" sacct -u "${ORCD_USER}" --starttime=today --format=JobID,JobName,State,ExitCode,Elapsed,End -X | tail -n 40 || true
    exit 0
  fi
  sleep "${POLL_SECS}"
done
