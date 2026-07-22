# 实验记录（Goal-Pose 动作先验）

用于快速回顾每组实验的配置与收敛情况，只记关键信息。新实验往下追加即可。

代码与方法细节见 `scripts/libero_goal_prior/IMPLEMENTATION.md`；训练脚本在 `scripts/libero_goal_prior/`。
数据集：本地 LIBERO（`/data2/JM/dataset/libero_lerobot_format`，273,465 帧、40 任务、fps 10）。
公共默认：seed 1000、GPU 0-6、`chunk_size=10`、`action_mode=continuous`。

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

- 初始化：MolmoAct2（仅模板：结构/processor/机器人 token）+ Molmo2-ER VLM 权重 + **随机初始化 action expert**（clean bootstrap，`USE_ER_BOOTSTRAP=true`）；`norm_tag=null`（用数据集统计量）。
- Stage1：无视觉（`disable_visual_input=true`）、冻结 VLM（`train_action_expert_only=true`）；goal token = `se3_encoder(目标位姿 s_{t+H})`，K=4；仅 `L_flow`；bs128/卡（有效 896）、10000 步、warmup(AE/goal)=500。
- Stage2：加载 Stage1 `checkpoints/010000`；goal token = learnable_queries（VLM 上下文化，**causal**，非 full-attn）；对 AE 屏蔽 raw image（`mask_image_from_action_expert=true`）；开 `L_pose`（`pose_recon_loss_weight=1.0`，目标=归一化 8D state 的 MSE）；VLM+AE 全量 LR 不下调 + warmup（VLM/AE/goal=2000，ViT=4000）；bs32/卡（有效 224）、30000 步。
- 收敛：stage1 final loss(=flow)=**0.065**；stage2 @step1k：flow≈0.28、pose≈0.12（warmup 到 2000，进行中）。
- 评测：TBD。
- 备注：起步 flow 高（step20≈1.36）属预期——goal 从"真值位姿"换成随机 query + raw image 被屏蔽；flow 下限受 pose 质量制约。warmup 比旧 baseline（200）长，判断为不影响最终、且对保护 VLM 泛化偏有利，故未重启。
