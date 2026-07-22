#!/usr/bin/env bash
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${WS}/scripts/activate_train_env.sh"

SEED="${SEED:-1000}"
EVAL_SEED="${EVAL_SEED:-1000}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${WS}/lerobot/outputs/libero_two_stage/seed_${SEED}}"
BASELINE_DIR="${BASELINE_DIR:-${EXPERIMENT_ROOT}/01_baseline_35k}"
STAGE2_DIR="${STAGE2_DIR:-${EXPERIMENT_ROOT}/02_two_stage/stage2_25k}"
BASE_EVAL_ROOT="${EVAL_ROOT:-${EXPERIMENT_ROOT}/03_evaluation/eval_seed_${EVAL_SEED}}"

CHECKPOINT_LABELS=(
  baseline_total20k baseline_total30k baseline_total35k
  two_stage_total20k two_stage_total30k two_stage_total35k
)
CHECKPOINT_PATHS=(
  "${BASELINE_DIR}/checkpoints/020000/pretrained_model"
  "${BASELINE_DIR}/checkpoints/030000/pretrained_model"
  "${BASELINE_DIR}/checkpoints/035000/pretrained_model"
  "${STAGE2_DIR}/checkpoints/010000/pretrained_model"
  "${STAGE2_DIR}/checkpoints/020000/pretrained_model"
  "${STAGE2_DIR}/checkpoints/025000/pretrained_model"
)

mkdir -p "${BASE_EVAL_ROOT}"

for index in "${!CHECKPOINT_LABELS[@]}"; do
  label="${CHECKPOINT_LABELS[$index]}"
  policy_path="${CHECKPOINT_PATHS[$index]}"
  if [[ ! -f "${policy_path}/config.json" ]]; then
    echo "Missing aligned checkpoint: ${policy_path}" >&2
    exit 2
  fi
  CHECKPOINT_LABEL="${label}" \
  EVAL_ROOT="${BASE_EVAL_ROOT}/${label}" \
  EVAL_SEED="${EVAL_SEED}" \
  EPISODES_PER_TASK="${EPISODES_PER_TASK}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    bash "${WS}/scripts/libero_eval/eval_libero_checkpoint.sh" "${policy_path}"
done

python "${WS}/scripts/libero_two_stage/summarize_eval.py" \
  --eval-root "${BASE_EVAL_ROOT}" \
  --output-dir "${BASE_EVAL_ROOT}/summary"
