# MolmoAct2 LIBERO controlled experiments

All runs use seed `1000`, GPUs `0-6`, continuous flow-matching loss, the full
local LIBERO dataset, and the same canonical initialization:

- MolmoAct2 supplies the model/action-expert structure and robot tokens.
- Molmo2-ER supplies every compatible VLM tensor.
- The released MolmoAct2 action-expert tensors are discarded and reset.
- `bootstrap_audit.json` records the source coverage and action-expert fingerprint.

Run the preflight suite first:

```bash
bash scripts/libero_two_stage/smoke_all.sh
```

The smoke suite contains exactly three 50-step runs:

1. `01_baseline_50step`
2. `02_stage1_50step`
3. `03_stage2_from_stage1_50step`

The formal matrix is:

- Baseline: 35k steps, batch size 32/GPU, visual input enabled, joint VLM + action-expert training.
- Stage 1: 10k steps, batch size 256/GPU, visual input disabled, action expert only.
- Stage 2: load Stage-1 model weights (not optimizer/scheduler), then 25k steps at batch size 32/GPU with visual input and joint training.
- All formal runs save every 10k steps and also save the final step.

Launch each phase manually from the repository root. Do not use `run_all.sh` for
formal experiments.

First, create the shared canonical initialization once:

```bash
bash scripts/libero_two_stage/create_canonical_init.sh
```

Launch the baseline independently:

```bash
bash scripts/libero_two_stage/train_baseline_35k.sh
```

Launch Stage 1 independently:

```bash
bash scripts/libero_two_stage/train_stage1_10k.sh
```

Only after Stage 1 has completed and written
`02_two_stage/stage1_10k/checkpoints/last/pretrained_model`, launch Stage 2:

```bash
bash scripts/libero_two_stage/train_stage2_25k.sh
```

The baseline and Stage 1 both read the canonical checkpoint and do not depend
on each other. Stage 2 depends only on the completed Stage-1 model weights.
Each command runs in the foreground; start and monitor it separately before
launching the next command.

All three training commands use automatic resume. If their own output directory
contains `checkpoints/last`, rerunning the same command restores model,
optimizer, scheduler, RNG, and training step from that checkpoint and appends
to the existing log. If no checkpoint exists, the command starts a fresh run.
Set `RESUME_MODE=false` only when intentionally starting from scratch in a new
output directory, or set `RESUME_CHECKPOINT=/path/to/checkpoint` to select an
explicit checkpoint.

Evaluate step-aligned checkpoints (`20k`, `30k`, `35k` total updates):

```bash
bash scripts/libero_two_stage/eval_aligned_checkpoints.sh
```

Evaluation uses 10 episodes per task for each of
`libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`, with
`per_episode_seed=true`, `eval_seed=1000`, and LIBERO's 50-step settling default.
Each run writes a sibling manifest before launch and copies it into the output
directory after successful initialization.

Evaluate one checkpoint on vanilla LIBERO (40 tasks, 400 rollouts):

```bash
bash scripts/libero_eval/eval_libero_checkpoint.sh \
  /path/to/checkpoint/pretrained_model
```

Evaluate one checkpoint on the balanced LIBERO-Plus visual-generalization
protocol (360 tasks, 3600 rollouts):

```bash
bash scripts/libero_eval/eval_libero_plus_checkpoint.sh \
  /path/to/checkpoint/pretrained_model
```

The Plus protocol covers Background Textures, Camera Viewpoints, Light
Conditions, Objects Layout, Robot Initial States, and Sensor Noise. It excludes
all `Language Instructions` variants. Both commands use GPUs 0-3 by default;
override with `EVAL_GPU_IDS="4 5 6 7"`.

Each command displays the success rate over completed rollouts while it runs.
The monitor can be reattached from another terminal without interrupting eval:

```bash
python scripts/libero_eval/monitor_eval.py \
  --eval-root /path/printed/by/the/eval/script
```

Live results are written atomically to `live_summary.json` and
`live_summary.csv`. Videos are disabled by default to avoid excessive disk
usage; set `MAX_EPISODES_RENDERED=1` to retain one rollout video per task.

Outputs use this fixed hierarchy:

```text
lerobot/outputs/libero_two_stage/seed_1000/
├── 00_canonical_init/
├── 01_baseline_35k/
├── 02_two_stage/
│   ├── stage1_10k/
│   └── stage2_25k/
├── 03_evaluation/
└── smoke/<timestamp>/
    ├── 01_baseline_50step/
    ├── 02_stage1_50step/
    └── 03_stage2_from_stage1_50step/
```
