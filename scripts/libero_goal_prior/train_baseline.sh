#!/usr/bin/env bash
# Baseline for the goal-pose prior experiment: plain MolmoAct2 fine-tuning.
#
# Same clean bootstrap as the goal-prior runs (Molmo2-ER VLM weights + randomly
# re-initialized action expert), but WITHOUT any goal-pose machinery: vision is
# on, the whole model trains, loss is flow matching only. This is the
# apples-to-apples comparison point for exp1.
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SEED="${SEED:-1000}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
export JOB_NAME="${JOB_NAME:-molmoact2-baseline}"
export OUTPUT_DIR="${OUTPUT_DIR:-${WS}/lerobot/outputs/libero_goal_prior/seed_${SEED}/baseline}"
export RESUME_MODE="${RESUME_MODE:-auto}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
# ~273K LIBERO frames; bs 32/GPU x 7 GPUs = 224 effective -> ~1221 steps/epoch.
export STEPS="${STEPS:-40000}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export SAVE_FREQ="${SAVE_FREQ:-5000}"
export LOG_FREQ="${LOG_FREQ:-20}"
export SEED

# Normal full training: vision on, train the whole model, flow loss only.
export ACTION_MODE=continuous
export TRAIN_ACTION_EXPERT_ONLY=false
export DISABLE_VISUAL_INPUT=false
export IMAGE_TRANSFORMS_ENABLE=true

# Clean bootstrap: load Molmo2-ER VLM weights, then randomly re-initialize the
# action expert (do NOT inherit the released MolmoAct2 action-expert weights).
USE_ER_BOOTSTRAP="${USE_ER_BOOTSTRAP:-true}"
VLM_CHECKPOINT_PATH="${VLM_CHECKPOINT_PATH:-${WS}/Checkpoint/Molmo2-ER}"

BASELINE_ARGS=(
  --policy.chunk_size="${CHUNK_SIZE}"
  --policy.n_action_steps="${CHUNK_SIZE}"
)

if [[ "${USE_ER_BOOTSTRAP}" == "true" ]]; then
  BASELINE_ARGS+=(
    --policy.vlm_checkpoint_path="${VLM_CHECKPOINT_PATH}"
    --policy.randomize_action_expert=true
    --policy.audit_bootstrap=true
  )
fi

exec bash "${WS}/scripts/train_libero_molmoact2.sh" "${BASELINE_ARGS[@]}" "$@"
