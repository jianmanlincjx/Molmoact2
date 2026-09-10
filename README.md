# LIT on MolmoAct2

The **Latent Interface Training (LIT)** instantiation of *Breaking the Vision–Action Shortcut: Latent
Interface Training for Generalizable Robot Foundation Models* on MolmoAct2.
Cross-framework hub, project page and released checkpoints: https://github.com/jianmanlincjx/LIT ·
https://jianmanlincjx.github.io/LIT/ · https://huggingface.co/linjianman/LIT

This is a fork of [MolmoAct2](https://github.com/allenai/molmoact2); the original README is kept as [`README_upstream.md`](./README_upstream.md)
(installation of the base framework lives there). Everything below is what this fork adds.


Everything below is MolmoAct2. The same two-stage recipe on π0.5, FAST-WAM and ImageWAM
lives in their own repositories; see the table at the bottom.

## Which branch, and why it says v4

Use **`feat/libero-goal-prior-v4`**. The paper's method is the *v3* recipe — 100 latents with
8 pose-supervised (`num_semantic_visual_tokens=100`, `num_semantic_visual_pose_tokens=8`),
under `scripts/libero_goal_prior_v3/`. The v4 directory is a different, harder bottleneck
(latents == pose tokens == 8) that the paper does **not** use; the branch is named after it
only because that recipe was added later on the same line of development.

Do not check out `feat/goal-pose-prior-v3`. It predates a revision to the v3 recipe's learning
rates that every reported number depends on:

| | that branch | what the paper uses |
| --- | --- | --- |
| action-expert LR | 1e-5 | **1e-4** |
| action-expert warmup | 1000 | **5000** |
| semantic-visual aggregator LR | 1e-5 | **1e-4** |
| semantic-visual warmup | 1000 | **5000** |

VLM, ViT and connector stay at 1e-5. Every ablation arm copies this same group of six learning
rates and seven warmups, read from the reference checkpoint's `train_config.json` rather than
from a script default.

## 0. What you need

| | |
| --- | --- |
| Backbone | `Checkpoint/MolmoAct2` (released weights) |
| VLM bootstrap | `Checkpoint/Molmo2-ER` (`allenai/Molmo2-ER`) |
| Data | LIBERO, LeRobot format, all four suites, `no_noops` — 1,693 episodes / 273K frames |
| Benchmarks | LIBERO (in-distribution) and [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) (10,030 perturbed tasks over 7 axes) |

The dataset's `meta/stats.json` must be the recomputed one: an earlier version had bad
QUANTILES that clipped 82% of the state Z range. `stats.v3_recompute_note.json` records the fix.

**LIBERO-Plus language bug.** Upstream `grab_language_from_filename` derives the instruction
from the perturbed file name, so for every non-language axis the policy is fed strings like
`... view 0 0 100 2 352 initstate 0`. Every number here is produced with `LIBERO_PLUS_FIX_LANG=1`,
which strips the perturbation suffix. Numbers taken without it are not comparable.

## 1. Baseline

Plain MolmoAct2 fine-tuning: vision on, whole model trains, flow-matching loss only.
Same clean bootstrap as LIT (Molmo2-ER VLM weights + randomly re-initialised action expert).

```bash
SEED=1000 STEPS=30000 BATCH_SIZE=32 \
  bash scripts/libero_goal_prior/train_baseline.sh
```

## 2. LIT (ours)

Two runs, in order. Stage 1 is vision-free; Stage 2 restores vision but routes it through
the latent interface.

```bash
# Stage 1 — language + state + chunk-end SE(3) -> action prior, no images, VLM frozen
SEED=1000 STEPS=10000 BATCH_SIZE=128 \
  bash scripts/libero_goal_prior/train_stage1.sh

# Stage 2 — 100 learnable latents aggregate the backbone; raw image tokens are masked
# out of the action expert; 8 latents are supervised to reconstruct the same SE(3) pose
SEED=1000 STEPS=30000 BATCH_SIZE=32 \
  STAGE1_OUTPUT_DIR=lerobot/outputs/libero_goal_prior/seed_1000/stage1 \
  bash scripts/libero_goal_prior/train_stage2.sh
```

Stage 1's SE(3) encoder is training-time scaffolding: Stage 2 discards it (it shows up as
6 unexpected keys at load) and the latents predict the pose from vision instead, so **no
privileged pose input is needed at inference**.

## 3. Evaluation

### LIBERO (in-distribution)

Official protocol: 50 episodes per task, official horizons, 4 suites x 10 tasks = 2,000 episodes.

```bash
CHECKPOINT_LABEL=my_run \
EVAL_ROOT=lerobot/outputs/libero_eval/my_run/libero_official50_seed_1000 \
EVAL_GPU_IDS='0 1 2 3 4 5 6 7' EVAL_SEED=1000 EVAL_BATCH_SIZE=10 \
  bash scripts/libero_eval/eval_libero_v4_official_8gpu.sh /path/to/pretrained_model
```

`EVAL_BATCH_SIZE=10` matters: 8 shards x 50 async envs oversubscribes a 128-core host and
runs ~3x slower per episode, with workers dropping out on broken pipes.

### LIBERO-Plus (out of distribution)

Full set, 10,030 tasks, one episode each. `EVAL_TASK_SHARD i/n` splits the manifest index-wise
so several instances cover disjoint work; four instances x 8 GPUs finishes in about 5 hours.

```bash
for i in 0 1 2 3; do
  LIBERO_PLUS_FIX_LANG=1 PLUS_PROTOCOL=full EVAL_SEED=1000 \
  EVAL_SUITES="libero_object libero_10 libero_goal libero_spatial" \
  EVAL_TASK_SHARD="$i/4" EVAL_GPU_IDS='0 1 2 3 4 5 6 7' \
  EVAL_ROOT=lerobot/outputs/libero_plus_eval/my_run/slice$i \
    bash scripts/libero_eval/eval_libero_plus_checkpoint_suites.sh /path/to/pretrained_model &
  sleep 20
done; wait
```

Ablation checkpoints deliberately differ from the full method's config, so they need
`EVAL_SKIP_CONFIG_GUARD=1 EVAL_VARIANT=v3`; the "direct visual conditioning" arm additionally
needs `LIT_ALLOW_OPEN_VISUAL_PATH=1`.

Then aggregate per axis, paired task-by-task against the reference runs:

```bash
python scripts/libero_ablation/collect_abl.py <arm_name>   # LIBERO-Plus, 7 axes
python scripts/libero_ablation/collect_id.py  <arm_name>   # LIBERO, 4 suites
```

## 4. Ablations

Each launcher in `scripts/libero_ablation/` pins the six learning-rate groups and seven
warmups read from the reference checkpoint's `train_config.json`, and audits the weight load
before committing GPU hours.

| Arm | Launcher | What changes vs. Full LIT |
| --- | --- | --- |
| Vanilla staged training | `sw_stage1.sh` then `sw_stage2.sh` | whole latent interface removed: no Stage-1 SE(3), no latents, no pose loss, no firewall |
| LIT w/o Stage-1 | `abl_wo_stage1.sh` | no Stage 1 at all; random action expert straight into the full LIT Stage 2 |
| LIT w/ direct visual conditioning | `abl_open_path.sh` | one switch: `mask_image_from_action_expert=false`, so image tokens reach the action expert alongside the latents |
| LIT w/o pose supervision | `abl_wo_pose.sh` | one switch: `enable_pose_reconstruction=false` |
| Latent interface only | `abl_latent_only.sh` | both spatial supervisions removed; interface and firewall kept |
| Baseline + pose head | `abl_ae_pose_head.sh` | baseline architecture; the chunk-end pose is regressed from pooled action-expert hidden states instead |

## 5. Checkpoints

Released directories (https://huggingface.co/linjianman/LIT). Each is a self-contained LeRobot policy
(`model.safetensors`, `config.json`, `train_config.json`, normaliser tensors) — pass the directory to
`--policy.path`.

| Table row | Directory |
| --- | --- |
| Baseline | `molmoact2/baseline` |
| Full LIT | `molmoact2/lit_stage1` (action prior) -> `molmoact2/lit_stage2` (reported model) |

Ablation checkpoints are not released.

## 6. The same method on other backbones

| Backbone | Repository |
| --- | --- |
| π0.5 | `jianmanlincjx/pi05` |
| FAST-WAM | `jianmanlincjx/fastwam` |
| ImageWAM | `jianmanlincjx/ImageWAM` |
