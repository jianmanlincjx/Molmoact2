#!/usr/bin/env bash
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEED="${SEED:-1000}"
export RUN_TYPE=two_stage_stage1_10k
export JOB_NAME="${JOB_NAME:-molmoact2-libero-stage1-10k}"
export EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${WS}/lerobot/outputs/libero_two_stage/seed_${SEED}}"
export CANONICAL_INIT_DIR="${CANONICAL_INIT_DIR:-${EXPERIMENT_ROOT}/00_canonical_init}"
export POLICY_PATH="${POLICY_PATH:-${CANONICAL_INIT_DIR}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_ROOT}/02_two_stage/stage1_10k}"
export RESUME_MODE="${RESUME_MODE:-auto}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export STEPS="${STEPS:-10000}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export SAVE_FREQ="${SAVE_FREQ:-10000}"
export LOG_FREQ="${LOG_FREQ:-20}"
export ACTION_MODE=continuous
export TRAIN_ACTION_EXPERT_ONLY=true
export DISABLE_VISUAL_INPUT=true
export IMAGE_TRANSFORMS_ENABLE=false
export SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-500}"
export SCHEDULER_VLM_WARMUP_STEPS="${SCHEDULER_VLM_WARMUP_STEPS:-500}"
export SCHEDULER_VIT_WARMUP_STEPS="${SCHEDULER_VIT_WARMUP_STEPS:-500}"
export SCHEDULER_CONNECTOR_WARMUP_STEPS="${SCHEDULER_CONNECTOR_WARMUP_STEPS:-500}"
export SCHEDULER_ACTION_EXPERT_WARMUP_STEPS="${SCHEDULER_ACTION_EXPERT_WARMUP_STEPS:-500}"
export SEED

if [[ ! -f "${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json" \
  && ! -f "${POLICY_PATH}/config.json" ]]; then
  echo "Canonical initialization is missing: ${POLICY_PATH}" >&2
  exit 2
fi

exec bash "${WS}/scripts/libero_two_stage/run_experiment.sh" "$@"
