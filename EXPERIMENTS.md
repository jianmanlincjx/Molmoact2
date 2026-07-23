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
| exp2（semantic-visual v2） | `feat/goal-pose-prior-v2` | `ed45544` | `feat/goal-pose-prior-v2` | `f903be74` | Stage1 沿用 exp1 |

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
- 评测（已判定无效）：Stage2 10k 曾记录 LIBERO Object 初始两项任务 **0/40 success**。2026-07-23 排查发现 non-RTC continuous inference 错误绕过 policy-owned prefill，导致 learnable goal queries 与 image mask 均未进入 AE；该结果不能用于评价 v1 架构，修复后需重测。
- 备注：v1 的 4 个 causal query 主要受 `L_pose` 约束，只屏蔽 AE 的 raw image KV；没有显式保留更丰富的视觉上下文。后续 v2 改为 100-token semantic→visual 聚合并保留 language/state KV，但“低 pose MSE 未转化为闭环成功”的旧结论因上述推理 bug 暂不成立。

---

## exp2 — semantic-visual recurrent prior v2 — 2026-07-22

- 代码：分支 `feat/goal-pose-prior-v2`（主仓 `ed45544` / 子模块 `f903be74`）。
- 初始化：直接加载 exp1 Stage1 `checkpoints/010000`；Stage1 结构与权重不变，VLM/AE 完整继承；v2 `semantic_visual_*` 模块随机初始化。
- Stage2：100 个 768-D learnable queries 跨 36 层递归；每层先 cross-attend vision-fused language/state hidden，再 cross-attend image patch hidden。AE 保留原 language/state KV、mask raw image KV，并追加对应层 100-token synthetic KV；AE 只单向读取，不写回 query。
- 位姿监督：最终层 100 tokens 通过 attention pooling 解码 normalized 8D target pose；`L = L_flow + L_pose`。
- 优化：VLM/ViT/connector/AE/semantic 均为 LR `1e-5`，不降低 AE LR；warmup VLM/connector/AE/semantic=1000、ViT=2000。
- 预算：7 GPU，bs32/卡（有效 224），**30000 步，每 5000 步保存**；输出 `lerobot/outputs/libero_goal_prior_v2/seed_1000/stage2`。
- 验证：53 个 MolmoAct2 单测通过；Stage1→v2 真实训练 smoke 通过（AE fingerprint=`ec1e80...`，`semantic_visual_*` 为预期 missing keys；旧 Stage1 checkpoint 的未训练 legacy goal keys 可由 `strict=False` 忽略；flow/pose/grad 有限）；旧 LIBERO 1-step inference smoke 后续确认只验证了 upstream fallback、未验证 semantic query 路径，不能作为有效推理验证；bs1/bs4 单卡峰值显存约 43.6GB。
- 收敛：step600 flow=0.269/pose=0.537，之后 pose 明显加速；step1000 flow=0.213/pose=0.164。
- 状态：已在约 step1000 主动停止，未继续到 5k；转向结构更明确的 exp2a/v2b。
- 评测：未执行。

---

## exp2a — two-group semantic-visual prior v2b — 2026-07-22

- 定位：替代已停止的 exp2，作为下一版主方案；同时解决 pose 读出瓶颈、token 间无直接通信和单套聚合器跨全部深度共享的问题。
- 初始化：同 exp2，直接加载 exp1 Stage1 `checkpoints/010000`；VLM/AE 完整继承，`semantic_visual_*` 随机初始化。
- Stage2：总数仍为 100×768 latent tokens；前 8 个定义为 pose 组，后 92 个定义为 context 组。每层先做全局 latent self-attention，再依次 cross-attend language/state 与 image。36 层分成连续 6 组，每组拥有独立 self/semantic/visual attention 和 latent→AE K/V projection，并在组内 6 层复用；token 跨组连续递归、不重置。全部 100 个 synthetic KV 仍进入 AE。
- 位姿监督：仅前 8 个 pose tokens 通过 v1-style concat decoder（`8×768 -> MLP -> 8D`）预测 normalized target pose；移除 exp2 的单向量 attention pooling；`L = L_flow + 0.3 * L_pose`。
- 参数：初始 learnable query bank 仅一份；六组聚合器约 170M 参数，相比原单组 v2 增加约 149M，远低于逐层完全独立的约 1B。
- 预算：若启动主跑，仍为 7 GPU、bs32/卡、30k、每 5k 保存；输出隔离到 `lerobot/outputs/libero_goal_prior_v2b/seed_1000/stage2`。
- 脚本：`scripts/libero_goal_prior_v2b/train_stage2.sh`；评测 `scripts/libero_eval/eval_libero_v2b_checkpoint.sh`。
- 验证：55 个 MolmoAct2 单测通过，覆盖六组映射、self→semantic→visual 顺序、跨 token 通信、pose→context 梯度耦合及旧 v2 state-dict 路径兼容。Stage1→v2b 单卡 2-step smoke 通过：AE fingerprint=`ec1e80...`，新增六组/pose 模块为预期 missing keys，legacy goal keys 为预期 unexpected keys；加入 pose-token LayerNorm 后随机初始 pose=0.689/0.832（修正前无 norm 为 5.602/3.893），flow=0.698/0.714，gradient 有限，峰值显存 44.7GB；semantic-visual 参数 173.6M；console log 成功落地。
- 推理修复（2026-07-23）：发现 `rtc_config=None` 时 `predict_action_chunk` 直接调用 upstream generation，绕过六组 semantic-visual query、synthetic KV 与 image mask。现已让所有 learnable goal-context continuous inference 走 policy-owned prefill/denoise，并新增 non-RTC 路由回归测试；完整 MolmoAct2 单测 57 个通过。5k checkpoint 在修复后完成 LIBERO Spatial task0 单回合 smoke，**1/1 success**（修复前同任务 0/20），确认 checkpoint 权重、六组 query、相对动作与后处理链路正确工作。
