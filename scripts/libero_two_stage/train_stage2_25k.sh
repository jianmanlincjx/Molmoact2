#!/usr/bin/env bash
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEED="${SEED:-1000}"
export RUN_TYPE=two_stage_stage2_25k
export JOB_NAME="${JOB_NAME:-molmoact2-libero-stage2-25k}"
export EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${WS}/lerobot/outputs/libero_two_stage/seed_${SEED}}"
export STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${EXPERIMENT_ROOT}/02_two_stage/stage1_10k}"
export POLICY_PATH="${POLICY_PATH:-${STAGE1_OUTPUT_DIR}/checkpoints/last/pretrained_model}"
export OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_ROOT}/02_two_stage/stage2_25k}"
export RESUME_MODE="${RESUME_MODE:-auto}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export STEPS="${STEPS:-25000}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export SAVE_FREQ="${SAVE_FREQ:-10000}"
export LOG_FREQ="${LOG_FREQ:-20}"
export ACTION_MODE=continuous
export TRAIN_ACTION_EXPERT_ONLY=false
export DISABLE_VISUAL_INPUT=false
export IMAGE_TRANSFORMS_ENABLE=true
export SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-1000}"
export SCHEDULER_VLM_WARMUP_STEPS="${SCHEDULER_VLM_WARMUP_STEPS:-1000}"
export SCHEDULER_VIT_WARMUP_STEPS="${SCHEDULER_VIT_WARMUP_STEPS:-1000}"
export SCHEDULER_CONNECTOR_WARMUP_STEPS="${SCHEDULER_CONNECTOR_WARMUP_STEPS:-1000}"
export SCHEDULER_ACTION_EXPERT_WARMUP_STEPS="${SCHEDULER_ACTION_EXPERT_WARMUP_STEPS:-300}"
export SEED

if [[ ! -f "${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json" \
  && ! -f "${POLICY_PATH}/config.json" ]]; then
  echo "Stage-1 model checkpoint is missing: ${POLICY_PATH}" >&2
  exit 2
fi

exec bash "${WS}/scripts/libero_two_stage/run_experiment.sh" "$@"
