# Bimanual YAM Goal-Pose Prior

A third instance of the two-stage goal-pose-prior method, alongside
[`scripts/libero_goal_prior_v3/`](../libero_goal_prior_v3/) and
[`scripts/droid_goal_prior/`](../droid_goal_prior/). The architecture is
unchanged — Molmo2-ER VLM bootstrap with a randomized continuous Action Expert,
a Stage-1 vision-free pose-conditioned prior, and a Stage-2 semantic-visual
recurrent aggregator (100 tokens = 8 pose + 92 context, 6 layer groups,
`L_flow + 0.3·L_pose`). Only the dataset contract differs.

No changes to the `lerobot` submodule are required: the policy is already
embodiment-agnostic and everything here is driven through `goal_pose_feature_key`,
`image_keys`, `target_pose_delta_index`, and `setup_type`/`control_mode`.

## Dataset contract

| | |
| --- | --- |
| Source | `real_robot_datasets/yam_3task` — three tasks merged by `merge_datasets.py` (292 episodes, 283,886 frames) |
| | LeRobot **v3.0**, `robot_type: bi_yam_follower`, **30 fps** |
| `observation.state` | 16-D absolute EEF, from `observation.state_eef_absolute` |
| `action` | 16-D absolute EEF, from `action_eef_absolute` |
| Goal target | `observation.state` at `t+30` — the LIBERO arrangement, **no separate feature** |
| Cameras | `observation.images.{top,left,right}` (AV1; resolutions are not pinned) |
| Horizon | `chunk_size = n_action_steps = target_pose_delta_index = 30` |

State/action layout, in order:

```
L_x L_y L_z  L_qw L_qx L_qy L_qz  L_gripper   R_x R_y R_z  R_qw R_qx R_qy R_qz  R_gripper
 0   1   2    3    4    5    6        7        8   9  10   11   12   13   14       15
```

Left-arm xyz stays at dims 0:3 so the existing goal-pose visualization in
`lerobot_eval.py` keeps working. The two gripper dimensions are *named*
`*.gripper` so MolmoAct2's name-based normalization mask
(`"gripper" not in name.lower()`, with `normalize_gripper=false`) leaves them raw.
Their range is [0, 1], already inside [-1, 1], so passing them through is safe.

### Why the goal is just `observation.state`

State is itself the absolute EEF pose, so the goal at `t+30` is the same feature
at a later frame — exactly LIBERO's arrangement, where one tensor carries a time
axis and the processor slices `[:, 0]` for the current state and `[:, -1]` for
the goal. There is no separate goal feature and no forward kinematics.

### Why quaternions, not axis-angle

LIBERO and DROID both use axis-angle (`[xyz(3), axis-angle(3), gripper]`). That
does not transfer here. The YAM grippers point down at the table, roughly π away
from identity, so converting these recordings to axis-angle puts them right on
the antipodal boundary where the 3-parameter encoding flips sign:

| | rotation angle > 3.0 rad | > 3.1 rad (π = 3.1416) |
| --- | ---: | ---: |
| Left arm | **17.0%** | 4.8% |
| Right arm | **13.3%** | 3.5% |

Roughly one frame in six, and it would land in the *action* space, under the
flow-matching loss. Quaternions are smooth in exactly that regime — near-π
rotations are where `qw ≈ 0`, which is unremarkable for a 4-component encoding.
The source already records quaternions, so keeping them also means no conversion
at train or deploy time.

### Quaternion sign canonicalization

The only quaternion hazard is the double cover: `q` and `−q` are the same
rotation but differ by `|Δq| = 2`. The recordings contain **88 such flips** on
the right arm, each about **697× a normal step** — unfixed they would inject
enormous spurious jumps into the action loss.

`prepare_dataset.py` fixes this per episode: each state quaternion is kept on the
same side of the 4-sphere as its predecessor, and the action quaternion of the
same frame is then tied to the state's hemisphere so the two can never diverge.
Lossless, and measured to leave **zero** residual flips (max step 0.049). All 97
episodes land in one hemisphere naturally, with `qy > 0.59` throughout, so
nothing sits near the boundary.

**Deploy note:** the flow-matching head emits an unconstrained 4-vector, so the
policy bridge must renormalize the quaternion to unit norm before sending it.

### Why the horizon is 30

LIBERO used H=10 at 10 Hz and DROID H=15 at 15 Hz — both 1.0 s. Measured EEF
translation on this dataset, against a 0.0092 m servo tracking lag:

| H | left arm | right arm | vs. lag |
| ---: | ---: | ---: | --- |
| 10 (0.33 s) | 0.0123 m | 0.0280 m | 1.3× — the goal is close to noise |
| **30 (1.00 s)** | **0.0335 m** | **0.0745 m** | **3.6× / 8×** |

## The three-task merge

`blocks_filtered`, `dustpan_filtered` and `transfer_filtered` were recorded
independently. Their feature schemas are byte-identical, but each numbers its
episodes, rows and tasks from zero, so `merge_datasets.py` builds one legitimate
v3.0 dataset from them before `prepare_dataset.py` ever runs. Doing it in that
order keeps the validated quaternion, anchor and verification code in
`prepare_dataset.py` completely untouched.

| Source | Task | `task_index` | Episodes | Frames |
| --- | --- | ---: | ---: | ---: |
| `blocks_filtered` | "Put all blocks into the box." | 0 | 97 | 136,489 |
| `dustpan_filtered` | "Clean the table using the dust pan." | 1 | 100 | 58,018 |
| `transfer_filtered` | "Transfer the egg from the pan into the bowl." | 2 | 95 | 89,379 |
| **merged** | | | **292** | **283,886** |

Five things collide on a naive concatenation, and only one of them would raise:

1. **`task_index` is `0` in all three** despite three different instructions.
   Nothing checks this, so the merged set would silently train a language-blind
   policy mapping one token sequence to three incompatible behaviours. This is
   the dangerous one.
2. `episode_index` restarts, so episodes collide and the per-episode quaternion
   canonicalization would run straight across a seam between two tasks.
3. The global `index` restarts, breaking the contiguity `anchor_mask` asserts.
4. `data/file_index` is `0` everywhere.
5. Video `file_index` restarts, and the shard split points differ per camera
   (blocks and transfer ship 2 shards per camera, dustpan 1), so frames would be
   read from the wrong recording.

Videos are never decoded or re-encoded — each source shard is symlinked under a
remapped `file_index` (5 per camera after the merge). That is precisely why the
per-episode `from_timestamp`/`to_timestamp` stay valid: they are relative to
their own shard, and the shard contents do not change.

The merge verifies itself the same way the build does. `--verify-only` re-hashes
every artifact, re-checks each source's content digest, compares every non-remapped
column against its source **element for element**, confirms each episode's
`task_index` round-trips to the same instruction string its metadata carries,
resolves every video symlink back to its source shard, and recomputes
`meta/stats.json`.

## Anchor rules

An anchor is a frame `t` that is a valid training start. Rules 1 and 4 differ
from DROID because the source columns do not exist:

1. the episode is not in `--exclude-episodes` (there is no
   `is_episode_successful` column; the list is how a bad demonstration is
   dropped once you review the full set);
2. `task_index` resolves through `meta/tasks.parquet` to non-empty task text;
3. the current state, all 30 actions and the `t+30` goal stay contiguous within
   one episode;
4. idle segmentation over the consecutive-command delta of the 16-D absolute
   EEF `action` (DROID read `action.joint_velocity`, which is absent here),
   rescaled for 30 fps: `action_delta=5e-4` (half DROID's 1e-3, since a 30 Hz
   step is about half as large), `min_idle_len=14` (≈0.47 s, the same wall clock
   as DROID's 7 at 15 Hz), `min_non_idle_len=31`;
5. the final 30 frames of each retained run are trimmed so the full action/goal
   horizon stays non-idle.

**Measured: rule 4 barely does anything** — 275,067 anchors survive out of the
275,126 the horizon trim alone would leave, so idle filtering removes 59 across
all three tasks. Idle runs rarely reach 14 consecutive frames in this
teleoperated data. It is kept for contract symmetry, not because it is
load-bearing.

## Normalization — the LIBERO v3 lesson

LeRobot v3.0 builds `meta/stats.json` by **aggregating per-episode statistics**,
which is not the true global quantile. That defect sank LIBERO v2b and is live in
two of the three sources here:

| Source | worst-dimension clipping under its own shipped stats |
| --- | ---: |
| `blocks_filtered` | 2.00% (clean) |
| `dustpan_filtered` | **24.19%** (`observation.state` `right_eef.gripper`) |
| `transfer_filtered` | **15.95%** (`observation.state` `right_eef.gripper`) |

It reaches non-gripper dimensions too: `dustpan_filtered`'s shipped quantiles
clip 18.6% of `action` `left_eef.qz`, and `transfer_filtered`'s clip 13.2% of
`action` `left_eef.x`. Those dimensions *are* normalized, so this would have gone
straight into the flow-matching loss.

`merge_datasets.py` therefore writes an exact global `meta/stats.json` rather
than aggregating the three, which brings the merged source down to 2.00%.

`prepare_dataset.py` recomputes `observation.state` and `action` statistics
directly from parquet, **over every frame of the included episodes**. The scope
is all-frames rather than anchors-only (DROID's choice) because the goal target
*is* `observation.state` at `t+H`: current states and goal states share one set
of quantiles, so the statistic has to cover every frame either role can reach.

After recomputation the worst clip is **2.06%** on both features — essentially
the q01/q99 floor (a q01/q99 pair excludes 2% by construction). `audit_normalization.py` is the gate and refuses anything
above 5%; it also checks the normalize→clamp→denormalize round trip (measured
1.1e-16) and that both raw gripper dims stay inside [-1, 1].

## Files

| File | Purpose |
| --- | --- |
| `merge_datasets.py` | Merge the three task datasets into one v3.0 dataset, and verify it. |
| `prepare_dataset.py` | Build and strictly verify the derived dataset. |
| `audit_normalization.py` | Measure the clipping training will actually apply. |
| `validate_dataloader.py` | Golden-check real batches through the MolmoAct2 processor. |
| `train_stage1.sh` | Vision-free pose-conditioned prior. |
| `train_stage2.sh` | Visual semantic-visual recurrent steering. |
| [`../train_yam_molmoact2.sh`](../train_yam_molmoact2.sh) | Shared launcher and dataset preflight. |

## Build

```bash
# Step 1 — merge the three sources into real_robot_datasets/yam_3task.
lerobot/.venv/bin/python scripts/yam_goal_prior/merge_datasets.py
lerobot/.venv/bin/python scripts/yam_goal_prior/merge_datasets.py --verify-only

# Step 2 — build the derived goal-pose view (292 episodes, 275,067 anchors).
lerobot/.venv/bin/python scripts/yam_goal_prior/prepare_dataset.py \
  --source-root real_robot_datasets/yam_3task \
  --output-root real_robot_datasets/yam_3task_goal_pose

# Independent re-verification: re-hashes every artifact, recomputes the anchor
# mask per episode against the manifest, recomputes all statistics to 1e-7,
# and re-derives every pose to exact equality.
lerobot/.venv/bin/python scripts/yam_goal_prior/prepare_dataset.py \
  --source-root real_robot_datasets/yam_3task \
  --output-root real_robot_datasets/yam_3task_goal_pose --verify-only
```

`--verify-only` needs the same `--source-root` the build used; it re-checks the
source content digest, so omitting it makes the default apply and the check fail
against the wrong dataset.

The build is fail-closed and atomic: it stages into a sibling directory and
`os.replace`s only after verification passes. An existing completed output is
reused only when its configuration fingerprint matches; an incomplete or
differently configured output is never overwritten.

`artifact_sha256` covers **every derived data parquet**, not just metadata.
The DROID recipe hashed only metadata, which is why 33 truncated parquet files
passed every provenance check and killed a run inside the dataloader after it
had already claimed the GPUs. The launcher re-verifies those hashes before
`accelerate launch`.

To pin the source, capture its content digest and pass it back:

```bash
DIGEST=$(lerobot/.venv/bin/python scripts/yam_goal_prior/prepare_dataset.py \
  --source-root real_robot_datasets/yam_3task --print-source-digest)
EXPECTED_SOURCE_DIGEST="$DIGEST" bash scripts/yam_goal_prior/train_stage1.sh
```

## Validate

```bash
lerobot/.venv/bin/python scripts/yam_goal_prior/audit_normalization.py \
  --dataset-root real_robot_datasets/yam_3task_goal_pose

lerobot/.venv/bin/python scripts/yam_goal_prior/validate_dataloader.py \
  --dataset-root real_robot_datasets/yam_3task_goal_pose
```

`validate_dataloader.py` needs no Stage-1 checkpoint — it builds the policy
config directly so it can run before the first launch. It checks the goal time
axis is present (if it is not, `_extract_goal_pose` returns `None` and Stage 2
**silently drops `L_pose`** with no warning), compares the processor against a
hand-written normalization reference, and reports the real tokenized sequence
length.

## Train

Both stages take `SMOKE_RUN=true` for a 100-step end-to-end check.

```bash
# Stage 1 smoke
SMOKE_RUN=true SMOKE_STEPS=100 \
DATASET_ROOT=$PWD/real_robot_datasets/yam_3task_goal_pose \
bash scripts/yam_goal_prior/train_stage1.sh

# Stage 2 smoke (initializes strictly from the Stage-1 smoke checkpoint)
SMOKE_RUN=true STAGE1_SMOKE_STEPS=100 SMOKE_STEPS=100 \
DATASET_ROOT=$PWD/real_robot_datasets/yam_3task_goal_pose \
bash scripts/yam_goal_prior/train_stage2.sh
```

Formal runs use the defaults:

```bash
DATASET_ROOT=/path/to/yam_goal_pose bash scripts/yam_goal_prior/train_stage1.sh
DATASET_ROOT=/path/to/yam_goal_pose bash scripts/yam_goal_prior/train_stage2.sh
```

### Schedule

The invariant is `steps × batch_size_per_gpu × num_gpus / anchors`. This dataset
yields **275,067 anchors** from 292 episodes, split across the three tasks in
proportion to their frames (48.5% / 20.0% / 31.5% — natural sampling, no
rebalancing).

Defaults are a deliberately **short first pass** — enough to read the loss curve
and time a step on real hardware before committing:

| | Global batch | Steps | Samples | Epochs |
| --- | ---: | ---: | ---: | ---: |
| Stage 1 | 1024 | 2,000 | 2.05M | ~7.4 |
| Stage 2 | 256 | 5,000 | 1.28M | ~4.7 |

For reference, matching LIBERO v3's actual exposure (37 / 28 epochs) over this
anchor count would be ~10,000 and ~30,000 steps — which is, coincidentally,
exactly LIBERO's own step count. At 275,067 anchors the merged set is close
enough in size to LIBERO's 273,465 frames that the two budgets converge. That is
the natural target once the short pass reads healthy.

Stage 1 uses `5e-5` for the Action Expert and goal encoder. Stage 2 uses `1e-4`
for VLM/ViT/connector and `3e-4` for the Action Expert and semantic-visual
modules. (The DROID README quotes an older `1e-5`/`1e-4` pair that its own
script no longer uses; these are the script values, which have a real run behind
them.)

Reach the target global batch with `GRAD_ACCUM_STEPS` on fewer GPUs — Stage 2
defaults to `BATCH_SIZE=8 × GRAD_ACCUM_STEPS=8`, because DROID OOMed in the
first DDP backward at `bs=32` with three 180×320 streams and YAM feeds three at
360–480p. **Run a memory probe before committing to a long job.**

### Known: six video shards are one frame short

Six of the 15 merged shards (`file-000` and `file-002`, all three cameras) report
one fewer frame than the rows assigned to them. This is **inherited**, not caused
by the merge — `blocks_filtered` and `dustpan_filtered` both show it on their own
shard 0 before any merging, and the merge never re-encodes or re-cuts video. It
is an encoder off-by-one on the last frame of a shard.

It is harmless here, and structurally so: anchor rule 5 trims the last 30 frames
of every retained run, so the last image any anchor requests sits **29–30 frames
before the end of its shard** on all 15 shards. Verified directly — zero anchors
read past the end, minimum margin 29 frames. Anything that widens the horizon or
relaxes the trim should re-check this.

## Notes

- `scripts/env.sh` defaults `CACHE_ROOT` to `/scratch/...`, which does not exist
  on every host. Override it (`CACHE_ROOT=/some/writable/path`) where it is
  missing.
- Checkpoints resolve from `${WS}/Checkpoint/<name>` and fall back to
  `${HOME}/checkpoints/<name>`. Both stages refuse to start unless the pinned
  revisions match: `MolmoAct2@e432d85f`, `Molmo2-ER@dab22564`.
- `MAX_SEQUENCE_LENGTH` defaults to 896. Measured with three cameras at
  360–480p and a 14-D state, the real tokenized length is **689**, so there is
  ample headroom; re-check with `validate_dataloader.py` if the instruction set
  grows much longer.
- Stage-2 lineage is enforced three ways: the checkpoint path must end with
  `yam_goal_prior/<seed>/[smoke/]stage1/checkpoints/<NNNNNN>/pretrained_model`,
  the Stage-1 `train_config.json` must match this recipe field for field, and
  the processor's stored stats must be bit-identical to the dataset's current
  `meta/stats.json` with a mask that preserves both raw grippers. The Action
  Expert fingerprint check is enforced inside `lerobot` at load time.
