#!/bin/bash
#SBATCH --job-name=droid-baseline
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
#SBATCH --output=/data/home/shailesh/logs/slurm-%j-baseline.log
#
# Baseline control for the DROID goal-pose experiment: Stage 1's weight
# initialization, Stage 2's optimization budget, no goal encoder.
#
# crane7 has 4 GPUs against Stage 2's 8, so GRAD_ACCUM_STEPS=2 restores the
# global batch of 768 (96 x 4 x 2). STEPS counts optimizer steps, so 100000 here
# means the same 100000 optimizer steps Stage 2 took -- the loop consumes 200000
# micro-batches to get there.
#
# The partition has no time limit, but RESUME_MODE=auto means re-submitting this
# same file after any interruption picks up from the last checkpoint.
set -uo pipefail
cd /data/home/shailesh/Molmoact2_jianman

# env.sh defaults its caches to /scratch, which does not exist on this cluster.
export CACHE_ROOT=/data/home/shailesh/.cache/Molmoact2

export DATASET_ROOT=/data/home/shailesh/datasets/droid_1.0.1_goal_pose
export SAMPLE_MANIFEST_PATH="${DATASET_ROOT}/valid_anchor_indices.parquet"
export CHECKPOINT_PATH=/data/home/shailesh/checkpoints/MolmoAct2
export VLM_CHECKPOINT_PATH=/data/home/shailesh/checkpoints/Molmo2-ER
export OUTPUT_DIR=/data/home/shailesh/outputs/droid_baseline/seed_1000/formal
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RESUME_MODE=auto

export STEPS=100000
export BATCH_SIZE=96
export GRAD_ACCUM_STEPS=2
export SAVE_FREQ=5000
export NUM_WORKERS=12

# Same fragmentation guard Stage 2 needed at this batch size. The baseline feeds
# images to the action expert (Stage 2 masked them out), so its activation
# footprint is not bounded by Stage 2's measurements -- see the memory probe.
export PYTORCH_ALLOC_CONF=expandable_segments:True

export WANDB_ENABLE=true
export WANDB_PROJECT=molmoact2-droid
export WANDB_DISABLE_ARTIFACT=true

bash scripts/droid_baseline/train_baseline.sh
