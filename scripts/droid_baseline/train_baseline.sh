#!/usr/bin/env bash
# DROID baseline: plain MolmoAct2 continuous policy, NO goal-pose prior.
#
# This is the control for the two-stage goal-pose experiment. It holds three
# things fixed against that experiment and changes exactly one:
#
#   held fixed  weights init  -- identical to Stage 1: the released MolmoAct2
#                               template, VLM tensors replaced by raw Molmo2-ER,
#                               action expert randomized. NOT the main-README
#                               init (which starts from MolmoAct2-DROID); the
#                               point is a like-for-like comparison, not the
#                               strongest possible baseline.
#   held fixed  data          -- the same derived dataset and the same anchor
#                               manifest, so both arms see the identical sample
#                               set. observation.ee_pose is present in the
#                               parquet but unused here.
#   held fixed  optimization  -- Stage 2's budget: 100k optimizer steps, global
#                               batch 768, Stage 2's LRs, warmups and schedule.
#   CHANGED     architecture  -- enable_goal_pose is left at its default false,
#                               so there is no SE(3) encoder, no learnable goal
#                               queries, no semantic-visual recurrent module and
#                               no pose reconstruction loss.
#
# Stage 2 ran 8 GPUs x batch 96. crane7 has 4, so GRAD_ACCUM_STEPS=2 restores
# the same global batch of 768. STEPS stays in optimizer-step units, so the LR
# schedule is identical to Stage 2's; the loop just pulls 200k micro-batches.
#
# Note this baseline does NOT set mask_image_from_action_expert. Stage 2 set it
# true because its semantic-visual module carried the visual signal; here the
# action expert must see the images itself, which makes its sequence longer than
# Stage 2's. That is the intended architecture, but it means Stage 2's memory
# measurements are not an upper bound -- probe before committing a long run.
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SEED="${SEED:-1000}"
CHUNK_SIZE=15
export JOB_NAME="${JOB_NAME:-molmoact2-droid-baseline}"
export RESUME_MODE="${RESUME_MODE:-auto}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export SEED

if [[ "${SMOKE_RUN:-false}" == "true" ]]; then
  export OUTPUT_DIR="${OUTPUT_DIR:-${WS}/lerobot/outputs/droid_baseline/seed_${SEED}/smoke}"
  export STEPS="${SMOKE_STEPS:-100}"
  export BATCH_SIZE="${SMOKE_BATCH_SIZE:-1}"
  export GRAD_ACCUM_STEPS="${SMOKE_GRAD_ACCUM_STEPS:-2}"
  export NUM_WORKERS="${SMOKE_NUM_WORKERS:-0}"
  export SAVE_FREQ="${SMOKE_SAVE_FREQ:-100}"
  export LOG_FREQ="${SMOKE_LOG_FREQ:-1}"
  export SAVE_CHECKPOINT="${SMOKE_SAVE_CHECKPOINT:-true}"
else
  export OUTPUT_DIR="${OUTPUT_DIR:-${WS}/lerobot/outputs/droid_baseline/seed_${SEED}/formal}"
  # Stage2 (droid_stage2_omp): 100000 optimizer steps at global batch 768.
  export STEPS="${STEPS:-100000}"
  export BATCH_SIZE="${BATCH_SIZE:-96}"
  export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
  export NUM_WORKERS="${NUM_WORKERS:-12}"
  export SAVE_FREQ="${SAVE_FREQ:-5000}"
  export LOG_FREQ="${LOG_FREQ:-20}"
fi

export DATASET_REPO_ID="${DATASET_REPO_ID:-lerobot/droid_1.0.1}"
export DATASET_ROOT="${DATASET_ROOT:?set DATASET_ROOT to the derived goal_pose dataset}"
export SAMPLE_MANIFEST_PATH="${SAMPLE_MANIFEST_PATH:-${DATASET_ROOT}/valid_anchor_indices.parquet}"

# Full-model training with vision, same as Stage 2.
export ACTION_MODE=continuous
export TRAIN_ACTION_EXPERT_ONLY=false
export DISABLE_VISUAL_INPUT=false
export IMAGE_TRANSFORMS_ENABLE=true
# Every frame is decoded here, so the same float32 timestamp-rounding guard
# Stage 2 needed applies (see train_stage2.sh).
export TOLERANCE_S="${TOLERANCE_S:-0.001}"

# Stage 2's optimizer settings, verbatim.
export OPTIMIZER_LR="${OPTIMIZER_LR:-1e-4}"
export OPTIMIZER_VIT_LR="${OPTIMIZER_VIT_LR:-1e-4}"
export OPTIMIZER_CONNECTOR_LR="${OPTIMIZER_CONNECTOR_LR:-1e-4}"
export OPTIMIZER_ACTION_EXPERT_LR="${OPTIMIZER_ACTION_EXPERT_LR:-3e-4}"
export OPTIMIZER_BETAS="${OPTIMIZER_BETAS:-[0.9,0.95]}"
export OPTIMIZER_EPS="${OPTIMIZER_EPS:-1e-6}"
export OPTIMIZER_WEIGHT_DECAY="${OPTIMIZER_WEIGHT_DECAY:-0.0}"
export OPTIMIZER_GRAD_CLIP_NORM="${OPTIMIZER_GRAD_CLIP_NORM:-1.0}"
export SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-5000}"
export SCHEDULER_VLM_WARMUP_STEPS="${SCHEDULER_VLM_WARMUP_STEPS:-5000}"
export SCHEDULER_VIT_WARMUP_STEPS="${SCHEDULER_VIT_WARMUP_STEPS:-5000}"
export SCHEDULER_CONNECTOR_WARMUP_STEPS="${SCHEDULER_CONNECTOR_WARMUP_STEPS:-5000}"
export SCHEDULER_ACTION_EXPERT_WARMUP_STEPS="${SCHEDULER_ACTION_EXPERT_WARMUP_STEPS:-5000}"

# Stage 1's initialization, byte for byte.
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-${WS}/Checkpoint/MolmoAct2}"
export CHECKPOINT_REVISION="${CHECKPOINT_REVISION:-e432d85f6e039edca44afb93c262f3084ab72a9c}"
VLM_CHECKPOINT_PATH="${VLM_CHECKPOINT_PATH:-${WS}/Checkpoint/Molmo2-ER}"
VLM_CHECKPOINT_REVISION="${VLM_CHECKPOINT_REVISION:-dab22564403d2607855bb1fffb0721285b445081}"

"${WS}/lerobot/.venv/bin/python" - \
  "${CHECKPOINT_PATH}" "${CHECKPOINT_REVISION}" \
  "${VLM_CHECKPOINT_PATH}" "${VLM_CHECKPOINT_REVISION}" <<'PY'
import sys
from pathlib import Path


def verify_download(label: str, raw_path: str, expected_revision: str) -> None:
    path = Path(raw_path).expanduser().resolve()
    if not path.is_dir():
        raise SystemExit(f"REFUSING: missing pinned {label} checkpoint: {path}")
    marker = path / ".cache/huggingface/download/config.json.metadata"
    if not marker.is_file():
        raise SystemExit(f"REFUSING: cannot verify {label} revision: {marker}")
    actual = marker.read_text(encoding="utf-8").splitlines()[0].strip()
    if actual != expected_revision:
        raise SystemExit(
            f"REFUSING: {label} revision is {actual}, expected {expected_revision}"
        )


verify_download("MolmoAct2", sys.argv[1], sys.argv[2])
verify_download("Molmo2-ER", sys.argv[3], sys.argv[4])
PY

# No goal-pose flags at all. enable_goal_pose defaults to false
# (configuration_molmoact2.py), which disables the SE(3) encoder, the learnable
# goal queries, the semantic-visual module and the pose reconstruction loss in
# one go. The bootstrap trio below is what makes this Stage 1's init:
# vlm_checkpoint_path swaps the VLM to raw Molmo2-ER, and the config layer
# requires randomize_action_expert=true alongside it so no released action
# expert can leak in.
BASELINE_ARGS=(
  --policy.vlm_checkpoint_path="${VLM_CHECKPOINT_PATH}"
  --policy.vlm_checkpoint_revision="${VLM_CHECKPOINT_REVISION}"
  --policy.randomize_action_expert=true
  --policy.audit_bootstrap=true
)

echo "[droid-baseline] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[droid-baseline] VLM=${VLM_CHECKPOINT_PATH} revision=${VLM_CHECKPOINT_REVISION}"
echo "[droid-baseline] no goal encoder; full-model training; vision enabled"
echo "[droid-baseline] chunk_size=${CHUNK_SIZE} (driver pins chunk/n_action_steps to 15)"

exec bash "${WS}/scripts/train_droid_molmoact2.sh" "${BASELINE_ARGS[@]}" "$@"
