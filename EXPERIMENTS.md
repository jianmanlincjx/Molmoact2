# 实验记录（Goal-Pose 动作先验）

数据：本地 LIBERO（273,465 帧、40 任务、fps 10）；
公共配置：seed 1000、GPU 0-6、chunk 10、continuous action。

## 版本

| 实验 | 主仓分支 | 主仓 commit | 子模块分支 | 子模块 commit | tag |
| --- | --- | --- | --- | --- | --- |
| v1（4-token goal-pose） | `feat/goal-pose-prior` | `d288f5d` | `feat/goal-pose-prior` | `f888ad14` | 10k inference-fixed |
| v2b（depth-grouped） | `feat/goal-pose-prior-v2` | `de781fa` | `feat/goal-pose-prior-v2` | `e13fe164` | in-dist 20k + plus 评测 |
| v3（fix QUANTILES） | `feat/goal-pose-prior-v3` | `dbd5696` | `feat/goal-pose-prior-v3` | `b3a70086` | stats fixed → Stage2 |
| 当前 v4（硬瓶颈 8/8） | `feat/libero-goal-prior-v4` | `7ab958e` | `feat/libero-goal-prior-v4` | `56e5e36d` | pose-only AE KV |

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
