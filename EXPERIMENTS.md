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
