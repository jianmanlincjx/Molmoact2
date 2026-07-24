# 实验记录（Goal-Pose 动作先验）

只保留 v1 与当前 v2b。数据：本地 LIBERO（273,465 帧、40 任务、fps 10）；
公共配置：seed 1000、GPU 0-6、chunk 10、continuous action。

## 版本

| 实验 | 主仓分支 | 主仓 commit | 子模块分支 | 子模块 commit | tag |
| --- | --- | --- | --- | --- | --- |
| v1（4-token goal-pose） | `feat/goal-pose-prior` | `d288f5d` | `feat/goal-pose-prior` | `f888ad14` | 10k inference-fixed |
| 当前 v2b（depth-grouped） | `feat/goal-pose-prior-v2` | `f008856` | `feat/goal-pose-prior-v2` | `40810a64` | 20k 评测 |

---

## v1 — 4-token Goal-Pose Prior

- 初始化：Molmo2-ER VLM + 随机 AE；Stage1 用 `SE3Encoder(s_t+10) → 4 tokens`，无视觉，仅 `L_flow`。
- Stage2：4 个 causal learnable queries；AE 屏蔽 raw image；`L = L_flow + L_pose`。
- 收敛：Stage1 flow=0.065；Stage2 10k flow=0.121、pose=0.016。
- 评测：旧 0/40 因推理 bug 无效；修复后 10k Spatial task0 为 30/32（93.75%）。

---

## 当前 v2b — 6-group Semantic-Visual Prior

- 初始化：加载 v1 Stage1 10k；VLM/AE 继承，新增模块随机初始化。
- 架构：100×768 tokens（8 pose + 92 context）；每层 `self-attn → semantic cross-attn → visual cross-attn`；36 层分 6 组，全部 synthetic KV 进入 AE。
- 损失：前 8 tokens 经 LayerNorm + concat decoder 预测 8D pose；`L = L_flow + 0.3 L_pose`。
- 训练：7×A800，bs32/卡，30k，每 5k 保存；聚合器 173.6M 参数。
- 推理：non-RTC learned-context 与实时准确率路径已由 59 个测试覆盖。
- 状态：主训练继续；10k Spatial=82.5%、Object=80.63%；20k checkpoint 正在评测。v1/v2b 统一每任务 32 rollouts。
