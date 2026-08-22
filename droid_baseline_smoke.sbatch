#!/bin/bash
#SBATCH --job-name=droid-baseline-smoke
#SBATCH --partition=debug
#SBATCH --nodelist=crane7
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=734000M
# ~95% of crane7's RealMemory=773574 MB. Slurm's node default is
# DefMemPerNode=16000 (16 GB), which OOM-kills the job during checkpoint load:
# each of the 4 ranks pulls MolmoAct2 (21.8 GB) plus the Molmo2-ER bootstrap
# (19.4 GB) into host RAM before anything reaches the GPU.
#SBATCH --output=/data/home/shailesh/logs/slurm-%j-baseline-smoke.log
set -uo pipefail
cd /data/home/shailesh/Molmoact2_jianman

# env.sh defaults its caches to /scratch, which does not exist on this cluster.
export CACHE_ROOT=/data/home/shailesh/.cache/Molmoact2

export DATASET_ROOT=/data/home/shailesh/datasets/droid_1.0.1_goal_pose_smoke
export SAMPLE_MANIFEST_PATH="${DATASET_ROOT}/valid_anchor_indices.parquet"
export CHECKPOINT_PATH=/data/home/shailesh/checkpoints/MolmoAct2
export VLM_CHECKPOINT_PATH=/data/home/shailesh/checkpoints/Molmo2-ER
export OUTPUT_DIR=/data/home/shailesh/outputs/droid_baseline/seed_1000/smoke
export CUDA_VISIBLE_DEVICES=0,1,2,3

export SMOKE_RUN=true
export SMOKE_STEPS=100
export SMOKE_BATCH_SIZE=1
export SMOKE_GRAD_ACCUM_STEPS=2
export WANDB_ENABLE=false

# Gate before the long run: 100 optimizer steps on 4 GPUs must show
# "Effective batch size: 1 x 4 procs x 2 accum = 8" and complete without OOM.
bash scripts/droid_baseline/train_baseline.sh
