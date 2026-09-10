# LIT on MolmoAct2

The **Latent Interface Training (LIT)** instantiation of *Breaking the Vision–Action Shortcut: Latent
Interface Training for Generalizable Robot Foundation Models* on MolmoAct2.
Hub, project page, checkpoints: https://github.com/jianmanlincjx/LIT · https://jianmanlincjx.github.io/LIT/ ·
https://huggingface.co/linjianman/LIT (public, no login needed)

This is a fork of [MolmoAct2](https://github.com/allenai/molmoact2) (the original README is
[`README_upstream.md`](./README_upstream.md)). Use branch **`feat/libero-goal-prior-v4`** and clone with
`--recursive` — the policy code lives in the `lerobot` submodule.

```bash
git clone --recursive -b feat/libero-goal-prior-v4 https://github.com/jianmanlincjx/Molmoact2.git
cd Molmoact2 && source scripts/activate_train_env.sh      # the LeRobot venv under lerobot/.venv
```

Two external things are needed for either path below:

| | |
| --- | --- |
| `LIBERO_PLUS_ROOT` | a [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) checkout (10,030 perturbed tasks over 7 axes) |
| `LIBERO_RESOURCE_ROOT` | the directory that contains `third_party/LIBERO-plus`; defaults to this repo |

Every number in the paper was produced with **`LIBERO_PLUS_FIX_LANG=1`**. Upstream LIBERO-Plus derives the
instruction from the perturbed file name, so without the fix the policy is fed strings like
`... view 0 0 100 2 352 initstate 0` on every non-language axis. **Overall** in the tables is the arithmetic
mean over the seven perturbation axes.

---

## 1. Evaluate the released checkpoint

```bash
hf download linjianman/LIT --include "molmoact2/*" --local-dir ./LIT_ckpt
CK=./LIT_ckpt/molmoact2/lit_stage2          # a self-contained LeRobot policy dir (config.json + model.safetensors)
```

**LIBERO (in-distribution)** — 4 suites × 10 tasks × 50 episodes, official horizons, seed 1000:

```bash
CHECKPOINT_LABEL=lit EVAL_ROOT=lerobot/outputs/libero_eval/lit/libero_official50_seed_1000 \
EVAL_GPU_IDS='0 1 2 3 4 5 6 7' EVAL_SEED=1000 EVAL_BATCH_SIZE=10 \
  bash scripts/libero_eval/eval_libero_v4_official_8gpu.sh "$CK"
```

`EVAL_BATCH_SIZE=10` matters: 8 shards × 50 async environments oversubscribes a 128-core host and
runs ~3× slower with workers dropping on broken pipes.

**LIBERO-Plus (out of distribution)** — all 10,030 tasks, one episode each. `EVAL_TASK_SHARD i/n`
splits the manifest so several instances cover disjoint work; four instances × 8 GPUs takes ~5 h:

```bash
for i in 0 1 2 3; do
  LIBERO_PLUS_FIX_LANG=1 PLUS_PROTOCOL=full EVAL_SEED=1000 \
  EVAL_SUITES="libero_object libero_10 libero_goal libero_spatial" \
  EVAL_TASK_SHARD="$i/4" EVAL_GPU_IDS='0 1 2 3 4 5 6 7' \
  EVAL_ROOT=lerobot/outputs/libero_plus_eval/lit/slice$i \
    bash scripts/libero_eval/eval_libero_plus_checkpoint_suites.sh "$CK" &
  sleep 20
done; wait
```

Aggregate per axis with `python scripts/aggregate.py lerobot/outputs/libero_plus_eval/lit` from the
[LIT hub](https://github.com/jianmanlincjx/LIT) (also `--libero` for the ID run, `--pair a b` to compare two runs).

Quick check (two episodes, one task, ~1 min):

```bash
python -m lerobot.scripts.lerobot_eval --policy.path="$CK" --policy.inference_action_mode=continuous \
  --env.type=libero --env.task=libero_spatial --env.task_ids="[0]" --eval.n_episodes=2 --eval.batch_size=1 --seed=1000
```

Verified on a fresh clone on 2026-09-10 (LIBERO and LIBERO-Plus rollouts, including the pose-overlay
videos on the project page).

---

## 2. Train, then evaluate

**What you need**

| | |
| --- | --- |
| Backbone | `Checkpoint/MolmoAct2` — the released MolmoAct2 weights |
| VLM bootstrap | `Checkpoint/Molmo2-ER` ([`allenai/Molmo2-ER`](https://huggingface.co/allenai/Molmo2-ER)) |
| Data | LIBERO in LeRobot format, all four suites, `no_noops` — 1,693 episodes / 273K frames; set `DATASET_ROOT` |
| Real-robot data | [`chinchinati/yam_bimanual_manipulation`](https://huggingface.co/datasets/chinchinati/yam_bimanual_manipulation) — the paper's YAM dual-arm demonstrations: three tasks, 292 episodes, LeRobot v3.0, 4.8 GB, public |

The dataset's `meta/stats.json` must be the recomputed one: an earlier version had bad quantiles that clipped
82% of the state-Z range (`stats.v3_recompute_note.json` records the fix).

**Baseline** — plain MolmoAct2 fine-tuning: vision on, whole model trains, flow-matching loss only, from the
same clean bootstrap as LIT (Molmo2-ER VLM weights + randomly re-initialised action expert):

```bash
SEED=1000 STEPS=30000 BATCH_SIZE=32 bash scripts/libero_goal_prior/train_baseline.sh
```

**LIT** — two runs, in order:

```bash
# Stage 1 — language + state + chunk-end SE(3) -> action prior; no images; VLM frozen
SEED=1000 STEPS=10000 BATCH_SIZE=128 bash scripts/libero_goal_prior/train_stage1.sh

# Stage 2 — 100 learnable latents aggregate the backbone; image tokens are masked out of the action
# expert; 8 latents are supervised to reconstruct the same SE(3) pose
SEED=1000 STEPS=30000 BATCH_SIZE=32 \
  STAGE1_OUTPUT_DIR=lerobot/outputs/libero_goal_prior/seed_1000/stage1 \
  bash scripts/libero_goal_prior/train_stage2.sh
```

To skip Stage 1, start Stage 2 from the released prior: `POLICY_PATH=./LIT_ckpt/molmoact2/lit_stage1` in place of `STAGE1_OUTPUT_DIR`.

Stage 1's SE(3) encoder is training-time scaffolding: Stage 2 discards it (6 unexpected keys at load) and the
latents predict the pose from vision instead, so no privileged pose input exists at inference. Then evaluate
`lerobot/outputs/libero_goal_prior/seed_1000/stage2/checkpoints/030000/pretrained_model` exactly as in §1.

**Learning rates that the reported numbers depend on.** Do not use the older `feat/goal-pose-prior-v3`
branch — it predates this revision:

| | old branch | what the paper uses |
| --- | --- | --- |
| action-expert LR / warmup | 1e-5 / 1000 | **1e-4 / 5000** |
| spatial-cue aggregator LR / warmup | 1e-5 / 1000 | **1e-4 / 5000** |
| VLM, ViT, connector | 1e-5 | 1e-5 |

**Ablations** (Table III) are launched from `scripts/libero_ablation/`; each launcher pins the same LR groups,
read from the reference checkpoint's `train_config.json`. Ablation checkpoints are not released.

| Arm | Launcher | Change vs. full LIT |
| --- | --- | --- |
| Vanilla staged training | `sw_stage1.sh` → `sw_stage2.sh` | whole interface removed: no Stage-1 SE(3), no latents, no pose loss, no firewall |
| LIT w/o Stage 1 | `abl_wo_stage1.sh` | random action expert straight into Stage 2 |
| LIT w/ direct visual access | `abl_open_path.sh` | `mask_image_from_action_expert=false` |
| LIT w/o pose supervision | `abl_wo_pose.sh` | `enable_pose_reconstruction=false` |
| Latent interface only | `abl_latent_only.sh` | both spatial supervisions removed; interface + firewall kept |
| Baseline w/ pose supervision | `abl_ae_pose_head.sh` | pose regressed from pooled action-expert states instead |

Ablation checkpoints deliberately differ from the full config: evaluate them with
`EVAL_SKIP_CONFIG_GUARD=1 EVAL_VARIANT=v3`, plus `LIT_ALLOW_OPEN_VISUAL_PATH=1` for the direct-visual-access arm.

---

## 3. How LIT is integrated in MolmoAct2

MolmoAct2 couples a Molmo2 VLM with a flow-matching action expert that reads the VLM's token sequence
(images, instruction, state) through layer-wise conditioning. LIT changes only that conditioning interface;
everything lives in the policy under `lerobot/src/lerobot/policies/molmoact2/`:

| Piece | Where | What it does |
| --- | --- | --- |
| Firewall | `mask_image_from_action_expert` (`modeling_molmoact2.py`) | image tokens are masked out of every action-expert attention; the action expert never reads backbone visual tokens directly |
| Latent interface | `semantic_visual_recurrent`, `num_semantic_visual_tokens=100` | 100 learnable tokens appended to the VLM sequence, contextualised by the backbone layer by layer, and exposed to the action expert as its only visual pathway |
| Spatial supervision | `num_semantic_visual_pose_tokens=8`, pose decoder, `pose_recon_loss_weight=0.3` | 8 of the latents are decoded by a small MLP to the chunk-end SE(3) pose; MSE against the same target Stage 1 was conditioned on |
| Stage-1 conditioning | SE(3) encoder (`goal_*` modules) | `Linear(8→512)→GELU→Linear(512→512)→GELU→Linear(512→4×2560)`; discarded after Stage 1 |
| Target | `target_pose_delta_index=10` (`processor_molmoact2.py`) | `observation.state` at t+10: EE position (3, world frame) + axis-angle (3) + gripper (2), quantile-normalised to [-1, 1] |
| Optimiser groups | `train_stage2.sh` | `semantic_visual_*` 1e-4 · `goal_*` 5e-5 · `action_expert` 1e-4 · VLM/ViT/connector 1e-5; warmup 5000 |

Stage 1 trains the SE(3) encoder and the action expert with the VLM frozen and no images; Stage 2 restores
vision, initialises the action expert from Stage 1, and fine-tunes everything jointly with
`L = L_flow + 0.3 · L_pose`. The action representation, chunk length (10), horizon and generation objective
are unchanged from upstream.
