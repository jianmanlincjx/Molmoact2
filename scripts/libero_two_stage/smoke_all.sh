#!/usr/bin/env bash
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEED="${SEED:-1000}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${WS}/lerobot/outputs/libero_two_stage/seed_${SEED}}"
SMOKE_ROOT="${SMOKE_ROOT:-${EXPERIMENT_ROOT}/smoke/${RUN_ID}}"
CANONICAL_INIT_DIR="${CANONICAL_INIT_DIR:-${EXPERIMENT_ROOT}/00_canonical_init}"
COMMON_ENV=(
  SEED="${SEED}"
  STEPS=50
  SAVE_FREQ=50
  LOG_FREQ=1
  SCHEDULER_WARMUP_STEPS=10
  SCHEDULER_VLM_WARMUP_STEPS=10
  SCHEDULER_VIT_WARMUP_STEPS=10
  SCHEDULER_CONNECTOR_WARMUP_STEPS=10
  SCHEDULER_ACTION_EXPERT_WARMUP_STEPS=5
)

if [[ ! -f "${CANONICAL_INIT_DIR}/config.json" ]]; then
  SEED="${SEED}" EXPERIMENT_ROOT="${EXPERIMENT_ROOT}" CANONICAL_INIT_DIR="${CANONICAL_INIT_DIR}" \
    bash "${WS}/scripts/libero_two_stage/create_canonical_init.sh"
fi

env "${COMMON_ENV[@]}" \
  CANONICAL_INIT_DIR="${CANONICAL_INIT_DIR}" \
  OUTPUT_DIR="${SMOKE_ROOT}/01_baseline_50step" \
  BATCH_SIZE=32 \
  bash "${WS}/scripts/libero_two_stage/train_baseline_35k.sh"

env "${COMMON_ENV[@]}" \
  CANONICAL_INIT_DIR="${CANONICAL_INIT_DIR}" \
  OUTPUT_DIR="${SMOKE_ROOT}/02_stage1_50step" \
  BATCH_SIZE=256 \
  bash "${WS}/scripts/libero_two_stage/train_stage1_10k.sh"

env "${COMMON_ENV[@]}" \
  STAGE1_OUTPUT_DIR="${SMOKE_ROOT}/02_stage1_50step" \
  OUTPUT_DIR="${SMOKE_ROOT}/03_stage2_from_stage1_50step" \
  BATCH_SIZE=32 \
  SCHEDULER_ACTION_EXPERT_WARMUP_STEPS=3 \
  bash "${WS}/scripts/libero_two_stage/train_stage2_25k.sh"

python "${WS}/scripts/libero_two_stage/validate_smokes.py" \
  --smoke-root "${SMOKE_ROOT}" \
  --canonical-dir "${CANONICAL_INIT_DIR}" \
  --expected-steps 50
