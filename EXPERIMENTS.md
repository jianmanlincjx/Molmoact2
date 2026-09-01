# 实验记录（Goal-Pose 动作先验）

数据：本地 LIBERO（273,465 帧、40 任务、fps 10）；
公共配置：seed 1000、GPU 0-6、chunk 10、continuous action。

## 版本

| 实验 | 主仓分支 | 主仓 commit | 子模块分支 | 子模块 commit | tag |
| --- | --- | --- | --- | --- | --- |
| v1（4-token goal-pose） | `feat/goal-pose-prior` | `d288f5d` | `feat/goal-pose-prior` | `f888ad14` | 10k inference-fixed |
| v2b（depth-grouped） | `feat/goal-pose-prior-v2` | `de781fa` | `feat/goal-pose-prior-v2` | `e13fe164` | in-dist 20k + plus 评测 |
| 当前 v3（fix QUANTILES） | `feat/goal-pose-prior-v3` | `dbd5696` | `feat/goal-pose-prior-v3` | `b3a70086` | stats fixed → Stage2 |

---

## v1 — 4-token Goal-Pose Prior

- 初始化：Molmo2-ER VLM + 随机 AE；Stage1 用 `SE3Encoder(s_t+10) → 4 tokens`，无视觉，仅 `L_flow`。
- Stage2：4 个 causal learnable queries；AE 屏蔽 raw image；`L = L_flow + L_pose`。
- 收敛：Stage1 flow=0.065；Stage2 10k flow=0.121、pose=0.016。
- 评测：旧 0/40 因推理 bug 无效；修复后 10k Spatial task0 为 30/32（93.75%）。

---

## v2b — 6-group Semantic-Visual Prior

- 初始化：加载 v1 Stage1 10k；VLM/AE 继承，新增模块随机初始化。
- 架构：100×768 tokens（8 pose + 92 context）；每层 `self-attn → semantic cross-attn → visual cross-attn`；36 层分 6 组，全部 synthetic KV 进入 AE。
- 损失：前 8 tokens 经 LayerNorm + concat decoder 预测 8D pose；`L = L_flow + 0.3 L_pose`。
- 训练：7×A800，bs32/卡，30k，每 5k 保存；聚合器 173.6M 参数。
- 推理：non-RTC learned-context 与实时准确率路径已由测试覆盖。
- In-dist（每任务 32 eps）：ours 20k **82.50%** vs baseline 30k **69.45%**（+13.05 pp）；Goal +25.6、Spatial +14.4、10 +9.4、Object +2.8。
- LIBERO-Plus：`base_category` 协议（排除 Language Instructions）；主对比 checkpoint = v2b 20k vs baseline 30k。
- 已知问题：数据集 `observation.state` QUANTILES 偏窄（Z 约 94% 被 clip），pose 可视化 meter 误差被放大；见 v3。

---

## 当前 v3 — 同 v2b 结构 + 修正 QUANTILES

- 动机：旧 `stats.json` 的 state Z q01/q99 ≈ `[0.64, 0.88]`，真实高度约 `[0.04, 1.27]`；归一化把 pose 目标压成 ±1。
- 做法：**不改训练代码**；`fix_stats.sh` 从 parquet 重算 `observation.state`/`action` 分位数并备份，再训 Stage1/Stage2。
- 已执行：state Z outside 率 **93.9% → 2.0%**；action any-dim outside **71% → 8%**。备份：`meta/stats.pre_v3_bad_quantiles.json`。
- 架构 / 超参：Stage2 与 v2b 相同（100 tokens / 8 pose / 6-group / weight 0.3）。
- 输出：`lerobot/outputs/libero_goal_prior_v3/seed_1000/{stage1,stage2}`。
- Stage2：**只**从 v3 Stage1 `010000` 初始化；启动时校验 NEW state Z 分位数（拒绝旧 clip stats）；无 legacy Stage1 回退。
- 注意：本机原 v1 Stage1 目录已缺失；改 stats 后不要 resume v2b。

---

## v3-highLR — 高学习率 Stage2

- 状态：**已完成 30k**；实际运行覆盖了原 canonical v3 Stage2 目录，因此该目录下各 checkpoint 均属于 high-LR recipe。
- 动机：参考 StarVLA 从 VL backbone + 随机 Action Head 训练 LIBERO 时为 Action Head 使用 `1e-4`；检验当前 Stage2 的 `1e-5` 是否限制 Action Expert 与新聚合模块的适配速度和最终性能。
- 初始化与数据：沿用 corrected QUANTILES，并严格从 v3 Stage1 `010000` 初始化；模型结构、损失、batch size 和 seed 均不变。
- 学习率：VLM / ViT / connector 保持 `1e-5`；Action Expert 与 semantic-visual 模块提高至 `1e-4`。
- 调度：Action Expert 与 semantic-visual warmup 提高至 5k；总步数与 decay 均为 30k。
- 实际输出：`lerobot/outputs/libero_goal_prior_v3/seed_1000/stage2`。
- 解释边界：该实验用于寻找更强的 v3 训练 recipe，不与 canonical v3 混报；StarVLA 的架构与归一化不同，因此这里只参考优化尺度，不视为严格对齐。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
OUTPUT_DIR=/data2/JM/Code/molmoact2/lerobot/outputs/libero_goal_prior_v3/seed_1000/stage2 \
LOG_FILE=/data2/JM/Code/molmoact2/lerobot/outputs/libero_goal_prior_v3/seed_1000/stage2.console.log \
RESUME_MODE=fresh \
STEPS=30000 \
SCHEDULER_DECAY_STEPS=30000 \
BATCH_SIZE=32 \
SAVE_FREQ=5000 \
OPTIMIZER_LR=1e-5 \
OPTIMIZER_VIT_LR=1e-5 \
OPTIMIZER_CONNECTOR_LR=1e-5 \
OPTIMIZER_ACTION_EXPERT_LR=1e-4 \
OPTIMIZER_SEMANTIC_VISUAL_LR=1e-4 \
SCHEDULER_ACTION_EXPERT_WARMUP_STEPS=5000 \
SCHEDULER_SEMANTIC_VISUAL_WARMUP_STEPS=5000 \
bash scripts/libero_goal_prior_v3/train_stage2.sh
```

---

## DROID Goal-Pose Prior 适配 — 后续计划

- 状态：**待实现**；先在本地小规模打通数据、两阶段训练与推理，再交由协作者在完整 DROID 上运行。
- 数据集：优先使用 Hugging Face [`lerobot/droid_1.0.1`](https://huggingface.co/datasets/lerobot/droid_1.0.1)（LeRobot v3.0，95,658 episodes、27,630,375 frames、15 FPS、三路相机，约 412 GB）；正式开始前与协作者固定同一 revision。
- 本地调试：只选择约 10 个 episodes 下载相关 v3 shards；小子集统计量仅用于 smoke test，不用于正式训练。
- 当前状态输入：保留 `observation.state = [joint_position(7), gripper(1)]`，与现有 MolmoAct2-DROID 接口一致。
- 独立目标位姿：由数据集已有的 `observation.state.cartesian_position = [xyz, roll, pitch, yaw]` 与 `gripper_position` 生成 `observation.ee_pose = [xyz(3), axis-angle(3), gripper(1)]`，无需 FK；该 pose 对应 DROID wrist attachment site，而非夹爪尖端 TCP。
- 最小代码改动：增加可配置的 `goal_pose_feature_key`；Stage1/Stage2 从未来 `observation.ee_pose` 读取 7D target；pose encoder/decoder 与 `observation.state` 维度解耦。8 pose tokens、92 context tokens、6-group aggregator 和 Action Expert 接口保持不变。
- 时间跨度：第一版保持 `chunk_size=10`，在 15 FPS 下对应约 0.67 秒；流程稳定后再对比 15-step（约 1 秒）目标。
- 归一化：在完整 train split 上分别重算 joint state、7D EE pose 和 action 的统计量；不得复用 LIBERO stats。检查每维 q01/q99、clip 比例、gripper 语义及 normalize→unnormalize round trip。
- Smoke test：验证 future target 不跨 episode、RPY→axis-angle 转换正确、Stage1 可训练、Stage2 可从 Stage1 初始化、Stage2 推理不依赖 future pose，以及输出 action 与 DROID 真机控制格式一致。
- 交付：核心模型/processor 修改提交到 `lerobot` 独立分支；DROID 数据准备、Stage1/Stage2 启动脚本和说明提交到主仓库，再由协作者执行全量统计与训练。
---

## Bimanual YAM Goal-Pose Prior Adaptation — Data and Two-Stage Scripts Wired Up

- Status: **the three-task merged dataset is built and verified**; the Stage-1/Stage-2 launch
  scripts are ready. The smoke runs must be **re-run against the merged dataset** on a node with
  a GPU (the earlier smoke runs used the old single-task data).
- Dataset: `real_robot_datasets/yam_3task` — merged from three independently recorded datasets
  (LeRobot v3.0, `bi_yam_follower`, **292 episodes, 283,886 frames**, **3 tasks**, **30 fps**,
  three AV1 camera streams: top 360×640 / left 480×640 / right 480×640):

  | Source dataset | Task | `task_index` | Episodes | Frames |
  | --- | --- | ---: | ---: | ---: |
  | `blocks_filtered` | "Put all blocks into the box." | 0 | 97 | 136,489 |
  | `dustpan_filtered` | "Clean the table using the dust pan." | 1 | 100 | 58,018 |
  | `transfer_filtered` | "Transfer the egg from the pan into the bowl." | 2 | 95 | 89,379 |

- **Merge first, then derive.** `merge_datasets.py` combines the three recordings into one valid
  v3.0 dataset, and `prepare_dataset.py` then runs on top of it **unchanged** — the validated
  quaternion canonicalization, anchor construction and verification code are untouched. The three
  recordings are byte-for-byte schema-identical but each numbers from 0, so a naive concatenation
  collides in five places. Only one of them raises an error:
  1. **All three use `task_index` 0** while their instructions differ. No check catches this; the
     result is a language-blind policy mapping one token span onto three mutually incompatible
     behaviors — the most dangerous of the five;
  2. `episode_index` restarts, so episodes collide and per-episode quaternion canonicalization
     would run across task seams;
  3. the global `index` restarts, breaking the contiguity that `anchor_mask` asserts;
  4. `data/file_index` is 0 everywhere;
  5. video `file_index` restarts, and each camera splits at different shard boundaries
     (blocks/transfer 2 shards per camera, dustpan 1), so frames would be pulled from the wrong
     recording.

  Videos are **neither decoded nor re-encoded**: each source shard is symlinked under its remapped
  `file_index` (5 shards per camera after merging). This is exactly why each episode's
  `from_timestamp`/`to_timestamp` remains valid — they are relative to their own shard, and shard
  contents are unchanged.
- **Both observation and action use absolute EEF pose**: `observation.state = action = 16-D`,
  per arm `[xyz(3), quat wxyz(4), gripper(1)]`. The dataset also records `action_eef_delta` and
  joint angles, but the delta columns are currently unreliable, so the derived view publishes only
  the two canonical columns `observation.state` / `action` — otherwise all three `action_*` keys
  would be mapped to policy action features by `dataset_to_policy_features`.
- **The goal pose is simply `observation.state` at t+30**, matching the shared-key arrangement used
  for LIBERO: the tensor carries a time axis and the processor reads `[:, 0]` as the current state
  and `[:, -1]` as the goal. There is no separate goal feature and **no FK is required** (unlike the
  20-D FK approach from the cubesv3 era, which is now retired).
- **Quaternions rather than axis-angle for rotation.** The YAM grippers point down at the table, and
  measured **17.0% (left) / 13.3% (right)** of frames have a rotation angle > 3.0 rad, sitting right
  against the ±π antipode where axis-angle is discontinuous. `L_pose` (MSE in normalized space) is
  not learnable on those frames.
- **Quaternion double cover is handled** (q ≡ −q). The merged raw data contains **374 sign flips**
  (284 left arm, 90 right arm) with a single-step |Δq| = 2.0, roughly **31×** the true maximum step.
  `QuaternionCanonicalizer` aligns hemispheres in episode order (each frame's action follows its
  state's hemisphere; state continuity carries across parquet file boundaries). After processing,
  **zero residual flips** remain and the maximum single step drops to **0.0639**. Merging makes
  episode numbering globally unique, so canonicalization never runs across a task seam.
- **Temporal horizon H=30** (1.0 s at 30 fps, equivalent to LIBERO 10@10Hz and DROID 15@15Hz).
  Measured chunk displacement at H=10 is 0.0123 m (left arm), only 1.3× the 0.0092 m servo tracking
  lag — the goal condition is close to noise. At H=30 displacement is 0.0335 m (left) / 0.0745 m
  (right), or 3.6× / 8× the lag.
- **Normalization (the trap LIBERO fell into; gated here).** LeRobot v3.0 aggregates per-episode
  statistics into a global `meta/stats.json`, which are not true global quantiles; on LIBERO that
  defect clipped roughly 94% of state-Z targets to ±1. **Two of the three source recordings really
  do hit it**: computed against their own bundled stats, the worst-dimension clip rate is
  `blocks_filtered` 2.00% (clean), `dustpan_filtered` **24.19%**, `transfer_filtered` **15.95%**.
  And it is not confined to gripper dimensions — `dustpan_filtered` clips 18.6% of `action`
  `left_eef.qz`, and `transfer_filtered` clips 13.2% of `action` `left_eef.x`. Both of those
  dimensions **participate in normalization** and feed directly into the flow-matching loss.
  `merge_datasets.py` therefore writes **exact global statistics** rather than an aggregate of the
  three, bringing the rate down to 2.00%; `prepare_dataset.py` then recomputes once more over the
  scope "all frames of the included episodes". Measured worst clip on the published data is
  **2.06%** (the theoretical floor for q01/q99), with round-trip error 1.1e-16. The gate remains in
  place: `audit_normalization.py` uses a 5% threshold, verified to actually fail (exit 1) via a
  negative control that injects bad quantiles.
- Valid anchors: **275,067 / 283,886 frames (96.9%)**. Idle filtering (derived from consecutive
  action differences, threshold 5e-4 and `min_idle_len` 14 at 30 fps) removes 59 frames in total on
  the merged set, retained for contract alignment.
- **Known issue (harmless, and not introduced by merging).** Six of the 15 merged shards (`file-000`
  and `file-002` across all three cameras) report one fewer container frame than the number of rows
  assigned to them. `blocks_filtered` and `dustpan_filtered` each showed this on their own shard 0
  before merging; merging never re-encodes or re-splits video, so this is an encoder off-by-one on
  the final frame of a shard. It is structurally safe: anchor rule 5 trims the last 30 frames of
  every retained run, so on all 15 shards the last image any anchor requests sits **29–30 frames
  before the end of its shard**. Verified directly: zero anchors out of range, minimum margin 29
  frames. Re-check this if the horizon is increased or the trimming relaxed in future.
- Data integrity: `artifact_sha256` covers **every derived data parquet**, not just metadata (DROID
  previously hashed only metadata, letting 33 truncated parquet files pass verification and crash
  only after occupying GPUs). Videos reuse the source data by symlink, so the derived view is only
  14 MB.
- **CUDA gate.** The first Stage-1 attempt ran at 195.62 s/step — the node driver reported CUDA 11.6
  while the venv had torch 2.10.0+cu128 (which needs driver ≥525), and LeRobot silently fell back to
  CPU (the tells in the log were `mem_gb:0.0` and an effective batch size of 1). The launch scripts
  now carry a `REQUIRE_CUDA` precondition check, and refuse to start rather than silently degrade.
- Verified on the merged dataset (runs to completion without a GPU node):
  - `merge_datasets.py --verify-only`: re-hashes every artifact, re-checks the content digests of
    all three sources, compares every non-remapped column against the source **element by element**,
    confirms each episode's `task_index` round-trips to the same instruction string in metadata,
    resolves every video symlink back to its source shard, and recomputes `meta/stats.json`
  - `prepare_dataset.py --verify-only`: 275,067 anchors, all hashes/statistics/derived columns
    recomputed consistently, asserting that canonicalization changed signs only (non-rotation
    dimensions bitwise unchanged) and that residual flips in the published data are zero
  - `audit_normalization.py`: worst_clip of **2.06%** on both features, round-trip error 1.1e-16
  - Video-path spot check: 9 episodes × 3 cameras across all three sources, actually decoded with
    ffmpeg at `from_timestamp`, all landing in the correct shard
- To be re-run on a GPU node (earlier results were based on the old single-task data and are now
  invalid):
  - `validate_dataloader.py` (on the old single-task data it measured a tokenized length of 691 / 896
    with 205 headroom; after merging, the longest instruction "Transfer the egg from the pan into the
    bowl." adds roughly 4 tokens, still within headroom, but this needs to be measured — and confirm
    that instructions genuinely vary across samples within a batch)
  - Stage-1 / Stage-2 smoke runs: this is **the first time the language pathway does real work**
  - the `train_stage1.sh` dry run and the `train_stage2.sh` lineage gate have both passed previously
- Training budget: start with a short run to check health — **Stage 1 at 2,000 steps, Stage 2 at
  5,000 steps**, 4 GPUs on crane7, Stage 2 at bs 8/GPU × 4 GPUs × accum 8 = global 256. Extend if
  healthy.
- Architecture and losses are **identical to LIBERO v3 / DROID** (100 tokens = 8 pose + 92 context,
  6 layer groups, `mask_image_from_action_expert=true`, `L_flow + 0.3 L_pose`), and the `lerobot`
  submodule **needs no changes at all**.
- TODO: on a GPU node, re-run the Stage-1/Stage-2 smoke runs (20 steps each) against
  `yam_3task_goal_pose` plus a Stage-2 memory probe (bs=8/GPU), then start formal training on
  crane7. A single-GPU smoke run **does not cover DDP or gradient accumulation**, so verify those
  separately on crane7 before the formal run. The existing Stage-1 checkpoint under
  `outputs/yam_goal_prior/seed_1000/smoke/` was trained on the old single-task data; its
  normalization statistics are invalidated by the dataset change (Stage 2's lineage precondition
  will reject it) and it should be deleted.
