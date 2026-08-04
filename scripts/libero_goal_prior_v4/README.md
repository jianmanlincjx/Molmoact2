# Goal-Pose Prior v4 — pose-only hard bottleneck

v4 is an **additive** recipe on top of v3: only the hard bottleneck differs
(`8/8` latents). Stage2 optimizer/warmup is locked to the v3 `025000` run
(StarVLA-aligned: VLM `1e-5`, AE + aggregator `1e-4`).

| | v3 (kept) | v4 (this dir) |
| --- | --- | --- |
| Stage1 goal tokens | 4 | **8** |
| Stage2 latents | 100 = 8 pose + 92 context | **8 pose only** |
| L_pose | first 8 | **all 8** |
| QUANTILES | recomputed | **reuse v3 stats** |
| Stage2 optimizer | VLM `1e-5`；AE / aggregator `1e-4`（warmup 5k） | **同 v3 025000 / StarVLA 对齐** |
| OUTPUT_DIR | `libero_goal_prior_v3/` | `libero_goal_prior_v4/` |

Architecture detail: [ARCHITECTURE.md](ARCHITECTURE.md).

## Train

```bash
# stats (once, if not already done for v3)
bash scripts/libero_goal_prior_v3/fix_stats.sh

bash scripts/libero_goal_prior_v4/train_stage1.sh
bash scripts/libero_goal_prior_v4/train_stage2.sh
```

Smoke:

```bash
CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=1 STEPS=2 SAVE_CHECKPOINT=false \
OUTPUT_DIR=lerobot/outputs/libero_goal_prior_v4/smoke_s1 \
bash scripts/libero_goal_prior_v4/train_stage1.sh
```

Eval wrapper: `scripts/libero_eval/eval_libero_v4_checkpoint.sh`.
