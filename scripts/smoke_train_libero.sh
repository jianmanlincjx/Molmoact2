#!/usr/bin/env bash
# 50-step smoke test on LIBERO + local loss curve plot.
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export JOB_NAME="${JOB_NAME:-molmoact2-libero-smoke}"
export STEPS="${STEPS:-50}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export LOG_FREQ="${LOG_FREQ:-1}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
# Smoke: skip mid-run checkpoints unless overridden
export SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export RUN_ID
OUTPUT_DIR="${OUTPUT_DIR:-${WS}/lerobot/outputs/${JOB_NAME}/${RUN_ID}}"
export OUTPUT_DIR
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}.log}"
export LOG_FILE

mkdir -p "$(dirname "${OUTPUT_DIR}")" "$(dirname "${LOG_FILE}")"

echo "[smoke] starting ${STEPS}-step run → ${OUTPUT_DIR}"
echo "[smoke] log → ${LOG_FILE}"

# Train (tee to log). Do not use exec so we can plot afterwards.
set +e
bash "${WS}/scripts/train_libero_molmoact2.sh" "$@" 2>&1 | tee "${LOG_FILE}"
train_rc=${PIPESTATUS[0]}
set -e

echo "[smoke] train exit code=${train_rc}"

# Prefer plotting against the run output dir if it exists; else log parent.
PLOT_DIR="${OUTPUT_DIR}"
if [[ ! -d "${PLOT_DIR}" ]]; then
  PLOT_DIR="$(dirname "${LOG_FILE}")"
fi

source "${WS}/scripts/activate_train_env.sh"
python "${WS}/scripts/plot_train_loss.py" \
  --log "${LOG_FILE}" \
  --out-dir "${PLOT_DIR}" \
  --title "MolmoAct2 LIBERO smoke (${STEPS} steps, bs=${BATCH_SIZE}/gpu)"

exit "${train_rc}"
