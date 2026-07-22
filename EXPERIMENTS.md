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
| exp1（goal-pose v1） | `feat/goal-pose-prior` | `a982375` | `feat/goal-pose-prior` | `a41a1ff` | 待 Stage2 跑完后打 `v1-goal-prior` |

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

- 代码：分支 `feat/goal-pose-prior`（主仓 `a982375` / 子模块 `a41a1ff`）。
- 初始化：MolmoAct2（仅模板：结构/processor/机器人 token）+ Molmo2-ER VLM 权重 + **随机初始化 action expert**（clean bootstrap，`USE_ER_BOOTSTRAP=true`）；`norm_tag=null`（用数据集统计量）。
- Stage1：无视觉（`disable_visual_input=true`）、冻结 VLM（`train_action_expert_only=true`）；goal token = `se3_encoder(目标位姿 s_{t+H})`，K=4；仅 `L_flow`；bs128/卡（有效 896）、10000 步、warmup(AE/goal)=500。
- Stage2：加载 Stage1 `checkpoints/010000`；goal token = learnable_queries（VLM 上下文化，**causal**，非 full-attn）；对 AE 屏蔽 raw image（`mask_image_from_action_expert=true`）；开 `L_pose`（`pose_recon_loss_weight=1.0`，目标=归一化 8D state 的 MSE）；VLM+AE 全量 LR 不下调 + warmup（VLM/AE/goal=2000，ViT=4000）；bs32/卡（有效 224）、30000 步。
- 收敛：stage1 final loss(=flow)=**0.065**；stage2 走势 step20 flow1.36/pose0.68 → 1k flow0.25/pose0.09 → 3k flow0.15/pose0.04 → **~6k flow≈0.14/pose≈0.03**（30k 中，进行中）。pose 已远低于均值基线，说明 query 确实从视觉学到目标位姿。
- 评测：TBD。
- 备注：起步 flow 高（step20≈1.36）属预期——goal 从"真值位姿"换成随机 query + raw image 被屏蔽；flow 下限受 pose 质量制约。warmup 比旧 baseline（200）长，判断为不影响最终、且对保护 VLM 泛化偏有利，故未重启。
