#!/bin/bash
#SBATCH --job-name=droid-baseline-memprobe
#SBATCH --partition=debug
#SBATCH --nodelist=crane7
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=734000M
# ~95% of crane7's RealMemory=773574 MB. Slurm's node default is
# DefMemPerNode=16000 (16 GB), which OOM-kills the job during checkpoint load:
# each of the 4 ranks pulls MolmoAct2 (21.8 GB) plus the Molmo2-ER bootstrap
# (19.4 GB) into host RAM before anything reaches the GPU.
#SBATCH --output=/data/home/shailesh/logs/slurm-%j-memprobe.log
#
# Short probe to find the largest per-GPU batch that fits, holding the global
# batch at 768. Submit with PROBE_BS / PROBE_ACCUM such that
# PROBE_BS * 4 GPUs * PROBE_ACCUM == 768:
#   sbatch --export=ALL,PROBE_BS=96,PROBE_ACCUM=2 droid_baseline_memprobe.sbatch
# Valid pairs: 96x2, 48x4, 32x6, 24x8, 16x12, 12x16.
set -uo pipefail
cd /data/home/shailesh/Molmoact2_jianman

PROBE_BS="${PROBE_BS:-96}"
PROBE_ACCUM="${PROBE_ACCUM:-2}"
if (( PROBE_BS * 4 * PROBE_ACCUM != 768 )); then
  echo "REFUSING: PROBE_BS($PROBE_BS) * 4 * PROBE_ACCUM($PROBE_ACCUM) != 768" >&2
  exit 2
fi

export CACHE_ROOT=/data/home/shailesh/.cache/Molmoact2
# Memory depends on batch shape, not dataset size, so the smoke view is enough.
export DATASET_ROOT=/data/home/shailesh/datasets/droid_1.0.1_goal_pose_smoke
export SAMPLE_MANIFEST_PATH="${DATASET_ROOT}/valid_anchor_indices.parquet"
export CHECKPOINT_PATH=/data/home/shailesh/checkpoints/MolmoAct2
export VLM_CHECKPOINT_PATH=/data/home/shailesh/checkpoints/Molmo2-ER
export OUTPUT_DIR=/data/home/shailesh/outputs/droid_baseline/probe_bs${PROBE_BS}
export CUDA_VISIBLE_DEVICES=0,1,2,3

export SMOKE_RUN=true
export SMOKE_STEPS=12
export SMOKE_BATCH_SIZE="${PROBE_BS}"
export SMOKE_GRAD_ACCUM_STEPS="${PROBE_ACCUM}"
export SMOKE_NUM_WORKERS=8
export SMOKE_SAVE_CHECKPOINT=false
export SMOKE_LOG_FREQ=1
export WANDB_ENABLE=false

# Match the formal run's allocator settings exactly, or the probe is not a probe.
export PYTORCH_ALLOC_CONF=expandable_segments:True

rm -rf "${OUTPUT_DIR}"
echo "[probe] batch=${PROBE_BS}/gpu x 4 gpus x ${PROBE_ACCUM} accum = $((PROBE_BS*4*PROBE_ACCUM))"
bash scripts/droid_baseline/train_baseline.sh
