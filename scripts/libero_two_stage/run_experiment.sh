#!/usr/bin/env bash
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${WS}/scripts/activate_train_env.sh"

: "${RUN_TYPE:?RUN_TYPE must be set}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"
: "${JOB_NAME:?JOB_NAME must be set}"

SEED="${SEED:-1000}"
DATASET_ROOT="${DATASET_ROOT:-/data2/JM/dataset/libero_lerobot_format}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}.log}"
MANIFEST_PATH="${OUTPUT_DIR}.manifest.json"
RESUME_MODE="${RESUME_MODE:-auto}"
if [[ -z "${RESUME_CHECKPOINT:-}" && "${RESUME_MODE}" == "auto" ]]; then
  candidate="${OUTPUT_DIR}/checkpoints/last"
  if [[ -f "${candidate}/pretrained_model/train_config.json" ]]; then
    RESUME_CHECKPOINT="${candidate}"
  fi
fi
export SEED DATASET_ROOT LOG_FILE RESUME_MODE
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

TRAIN_CMD=(bash "${WS}/scripts/train_libero_molmoact2.sh" "$@")
python "${WS}/scripts/libero_two_stage/write_manifest.py" \
  --workspace "${WS}" \
  --dataset-root "${DATASET_ROOT}" \
  --output "${MANIFEST_PATH}" \
  --run-type "${RUN_TYPE}" \
  --seed "${SEED}" \
  --command "${TRAIN_CMD[@]}"

set +e
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  "${TRAIN_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
else
  "${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
fi
train_rc=${PIPESTATUS[0]}
set -e

if [[ -d "${OUTPUT_DIR}" ]]; then
  cp "${MANIFEST_PATH}" "${OUTPUT_DIR}/experiment_manifest.json"
  python "${WS}/scripts/plot_train_loss.py" \
    --log "${LOG_FILE}" \
    --out-dir "${OUTPUT_DIR}" \
    --title "MolmoAct2 LIBERO ${RUN_TYPE}"
fi

exit "${train_rc}"
