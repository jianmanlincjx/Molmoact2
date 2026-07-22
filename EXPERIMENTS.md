# 实验记录（Goal-Pose 动作先验）

用于快速回顾每组实验的配置与收敛情况，只记关键信息。新实验往下追加即可。

代码与方法细节见 `scripts/libero_goal_prior/IMPLEMENTATION.md`；训练脚本在 `scripts/libero_goal_prior/`。
数据集：本地 LIBERO（`/data2/JM/dataset/libero_lerobot_format`，273,465 帧、40 任务、fps 10）。
公共默认：seed 1000、GPU 0-6、`chunk_size=10`、`action_mode=continuous`。

## 代码分支索引

每版实验单开一个特性分支，主仓与 lerobot 子模块**用同名分支**，方便对照。规则：先推 lerobot 子模块，再推主仓（主仓里记录的是子模块 commit 指针）。

- 主仓 remote：`origin = git@github.com-jianman:jianmanlincjx/Molmoact2.git`（`upstream = allenai/molmoact2`）
- 子模块 remote：`origin = git@github.com-jianman:jianmanlincjx/lerobot.git`（`upstream = allenai/lerobot`）
- 基线分支：主仓 `main` / 子模块 `molmoact2-policy`（保持干净，不直接堆实验）

| 实验 | 主仓分支 | 主仓 commit | 子模块分支 | 子模块 commit | tag |
| --- | --- | --- | --- | --- | --- |
| exp1（goal-pose v1） | `feat/goal-pose-prior` | `c564ee8` | `feat/goal-pose-prior` | `a41a1ff` | v1 代码快照（10k 评测） |
| exp2（semantic-visual v2） | `feat/goal-pose-prior-v2` | `ed45544` | `feat/goal-pose-prior-v2` | `bd1d2b58` | Stage1 沿用 exp1 |

记录格式（模板）：

```
## expN — 名称 — 日期
- 初始化：基于 xxx / xxx
- Stage1：xxx
- Stage2：xxx
- 收敛：stage1 final loss=xxx；stage2 flow/pose=xxx
- 评测：vanilla LIBERO=xx%；LIBERO-Plus=xx%
- 备注：xxx
```

---

## exp1 — goal-pose v1（se3 先验 → 视觉 query steering） — 2026-07-21

- 代码：分支 `feat/goal-pose-prior`（主仓 `c564ee8` / 子模块 `a41a1ff`）。
- 初始化：MolmoAct2（仅模板：结构/processor/机器人 token）+ Molmo2-ER VLM 权重 + **随机初始化 action expert**（clean bootstrap，`USE_ER_BOOTSTRAP=true`）；`norm_tag=null`（用数据集统计量）。
- Stage1：无视觉（`disable_visual_input=true`）、冻结 VLM（`train_action_expert_only=true`）；goal token = `se3_encoder(目标位姿 s_{t+H})`，K=4；仅 `L_flow`；bs128/卡（有效 896）、10000 步、warmup(AE/goal)=500。
- Stage2：加载 Stage1 `checkpoints/010000`；goal token = learnable_queries（VLM 上下文化，**causal**，非 full-attn）；对 AE 屏蔽 raw image（`mask_image_from_action_expert=true`）；开 `L_pose`（`pose_recon_loss_weight=1.0`，目标=归一化 8D state 的 MSE）；VLM+AE 全量 LR 不下调 + warmup（VLM/AE/goal=2000，ViT=4000）；bs32/卡（有效 224）、30000 步。
- 收敛：stage1 final loss(=flow)=**0.065**；stage2 走势 step20 flow1.36/pose0.68 → 1k flow0.25/pose0.09 → 3k flow0.15/pose0.04 → 6k flow≈0.14/pose≈0.03 → **10k flow=0.121、pose=0.016**。训练损失继续缓慢下降，但闭环行为没有同步改善。
- 评测：Stage2 10k in-dist LIBERO Object 初始两项任务 **0/40 success**（每任务 20 rollouts）；视频中机械臂有明显运动，但方向/终点错误且出现漂移。完整四套件结果未作为 v1 结论继续等待。
- 备注：v1 的 4 个 causal query 主要受 `L_pose` 约束，只屏蔽 AE 的 raw image KV；没有显式保留更丰富的视觉上下文。低 pose MSE 并未转化为闭环成功，后续 v2 改为 100-token semantic→visual 聚合，并保留 language/state KV。

---

## exp2 — semantic-visual recurrent prior v2 — 2026-07-22

- 代码：分支 `feat/goal-pose-prior-v2`（主仓 `ed45544` / 子模块 `bd1d2b58`）。
- 初始化：直接加载 exp1 Stage1 `checkpoints/010000`；Stage1 结构与权重不变，VLM/AE 完整继承；v2 `semantic_visual_*` 模块随机初始化。
- Stage2：100 个 768-D learnable queries 跨 36 层递归；每层先 cross-attend vision-fused language/state hidden，再 cross-attend image patch hidden。AE 保留原 language/state KV、mask raw image KV，并追加对应层 100-token synthetic KV；AE 只单向读取，不写回 query。
- 位姿监督：最终层 100 tokens 通过 attention pooling 解码 normalized 8D target pose；`L = L_flow + L_pose`。
- 优化：VLM/ViT/connector/AE/semantic 均为 LR `1e-5`，不降低 AE LR；warmup VLM/connector/AE/semantic=1000、ViT=2000。
- 预算：7 GPU，bs32/卡（有效 224），**30000 步，每 5000 步保存**；输出 `lerobot/outputs/libero_goal_prior_v2/seed_1000/stage2`。
- 验证：52 个 MolmoAct2 单测通过；Stage1→v2 真实 2-step smoke 通过（AE fingerprint=`ec1e80...`，仅 `semantic_visual_*` 为预期 missing keys，flow/pose/grad 有限）；保存 v2 checkpoint 后的 LIBERO 1-step 端到端推理 smoke 通过；bs1/bs4 单卡峰值显存约 43.6GB。
- 评测：TBD（优先 5k/10k/15k in-dist）。
