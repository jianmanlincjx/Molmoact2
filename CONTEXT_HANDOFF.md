# Goal-Pose Prior 项目快速交接

> 用途：在新对话窗口中快速恢复上下文。  
> 最近更新：2026-08-02 09:25（UTC+8）。动态训练与评测状态请按下文命令重新核对。

## 新窗口启动方式

将下面这句话发给新窗口：

> 请先阅读 `CONTEXT_HANDOFF.md`、`EXPERIMENTS.md` 和 `README.md`，再检查文档中列出的动态状态文件与 tmux；以 `CONTEXT_HANDOFF.md` 为当前项目上下文，不要从旧版本重新推断。

建议优先读取：

```text
CONTEXT_HANDOFF.md
EXPERIMENTS.md
README.md
lerobot/outputs/libero_eval/v2b_20k_vs_v3_15k_task_comparison.md
lerobot/outputs/libero_goal_prior_v3/seed_1000/baseline_vs_v2b_vs_v3_live.latest.json
```

---

## 一句话研究主线

**先通过显式目标位姿条件学习可控的 Action Prior，再从当前图像、语言和机器人状态中聚合等价的 Visual Steering Condition，引导该 Action Prior 完成场景相关的动作生成。**

研究标题：

> **Learning Visually Steerable Action Priors for Generalizable Robot Manipulation**

---

## 方法结构

### Stage 1：Pose-conditioned Action Prior

- 初始化：Molmo2-ER VLM + 随机初始化 Action Expert。
- 冻结 VLM，关闭视觉输入。
- 使用未来时刻 `t + H` 的末端位姿作为显式条件。
- Action Expert 学习“给定目标位姿，如何生成到达该目标的动作轨迹”。
- 只优化 flow-matching loss。

### Stage 2：Visual Steering

- 严格继承 Stage 1 的 Action Expert。
- 使用 100 个 `768D` learnable tokens：
  - 前 8 个 pose tokens；
  - 后 92 个 context tokens。
- 递归注意力顺序：
  1. latent self-attention；
  2. semantic cross-attention；
  3. visual cross-attention。
- 36 个 VLM layers 划分为 6 个连续 depth groups，每组复用一套聚合器。
- 前 8 个 pose tokens 经 concat decoder 重建未来 8D pose。
- 全部 100 个 synthetic K/V 条件化 Action Expert。
- Raw image tokens 被禁止直接进入 Action Expert。
- Stage 2 损失：

```text
L = L_flow + 0.3 * L_pose
```

### 推理边界

- 推理时不使用 future target pose。
- Visual Steering Condition 只从当前 image-language-state context 聚合。
- continuous non-RTC inference 必须经过 policy-owned prefill，否则 learnable queries 和 image mask 不会生效；当前代码已包含并校验该修复。

---

## 版本关系

### v2b

- 100 tokens（8 pose + 92 context）。
- 6-group depth-specific aggregator。
- Self → semantic cross → visual cross。
- 旧 QUANTILES。
- Action Expert / aggregator 学习率约 `1e-5`。

### 当前 v3

- **架构与 v2b 不变。**
- 修复 `observation.state` 和 `action` 的 QUANTILES。
- Stage 1 使用修复后统计量重新训练。
- Stage 2 从 v3 Stage 1 `010000` 初始化。
- 当前实际运行还采用 high-LR recipe：
  - VLM / ViT / connector：`1e-5`；
  - Action Expert：`1e-4`；
  - semantic-visual：`1e-4`；
  - AE / semantic-visual warmup：5k；
  - 总步数和 scheduler decay：30k。

重要解释边界：

- 当前 v3 相比 v2b 同时包含 corrected QUANTILES 和 high LR。
- 二者的精确贡献无法用现有实验拆分。
- QUANTILES 修复应视为数据纠错，而不是方法优化。
- 若论文需要公平证明架构贡献，最终应补一个 corrected QUANTILES、相同优化预算的 baseline。

---

## QUANTILES 问题

旧数据统计严重错误：

- State Z 旧 q01/q99 约为 `[0.64, 0.88]`；
- 实际约为 `[0.04, 1.27]`；
- 约 94% 的 State Z 被 clamp。

修复结果：

- State Z outside ratio：`93.9% → 2.0%`；
- Action any-dim outside ratio：`71% → 8%`；
- 原始统计备份：

```text
/data2/JM/dataset/libero_lerobot_format/meta/stats.pre_v3_bad_quantiles.json
```

- 当前正确统计：

```text
/data2/JM/dataset/libero_lerobot_format/meta/stats.json
```

- v3 checkpoint 保存的 preprocessor/postprocessor 已核对，与正确 q01/q99 一致。
- 旧 v2b 与新 v3 的 loss 绝对值不可直接解释为同尺度性能差异；v3 内部随 step 的下降趋势可以比较。

---

## 当前训练状态

实际 high-LR Stage 2 输出在：

```text
lerobot/outputs/libero_goal_prior_v3/seed_1000/stage2
```

而不是原计划文档中的 `stage2_lr1e4`。

截至最近快照：

- step：29,860 / 30,000；
- epoch：24.46；
- total loss：0.019；
- flow loss：0.018；
- pose loss：0.003；
- grad norm：0.936；
- AE / semantic-visual LR：约 `1e-5`。

动态状态：

```bash
tmux attach -t trian
```

或者读取：

```text
lerobot/outputs/libero_goal_prior_v3/seed_1000/stage2.console.log
lerobot/outputs/libero_goal_prior_v3/seed_1000/baseline_vs_v2b_vs_v3_live.latest.json
lerobot/outputs/libero_goal_prior_v3/seed_1000/baseline_vs_v2b_vs_v3_live.png
```

训练原则：

- 先完成当前 30k，不中途修改 scheduler。
- success rate 不保证随训练 loss 单调提升。
- 最终在 15k / 25k / 30k 中按完整闭环评测选择 checkpoint。

---

## LIBERO 已完成结果

### Baseline 30k（旧 QUANTILES）

```text
Overall: 69.45% (889/1280)
```

### v2b 20k（旧 QUANTILES）

```text
Spatial: 89.38% (286/320)
Object:  87.19% (279/320)
Long:    73.12% (234/320)
Goal:    80.31% (257/320)
Overall: 82.50% (1056/1280)
```

### v3 15k（corrected QUANTILES + high LR）

```text
Spatial: 96.56% (309/320)
Object:  96.25% (308/320)
Long:    87.81% (281/320)
Goal:    95.94% (307/320)
Overall: 94.14% (1205/1280)
```

相对 v2b 20k：

```text
+11.64 percentage points
```

关键困难任务：

- LIBERO-10 task 8：`3.12% → 71.88%`；
- LIBERO-Goal task 9：`50.00% → 96.88%`。

当前判断：

- v3 15k 已处于强模型区间；
- 30k 有希望进入约 95%–97% 区间；
- StarVLA 常见 30k 结果约 95.4%–96.6%，更新的报告可达 98% 左右；
- 正式声称 SOTA 前应使用相同协议，最好补每任务 50 episodes。

---

## v3 25k 评测状态

输出目录：

```text
lerobot/outputs/libero_eval/goal_prior_v3_025000/libero_seed_1000
```

最近快照：

```text
Spatial: 95.94% (307/320)
Object:  95.31% (305/320)
LIBERO-10: 评测中
Goal: 未开始
```

当前 Markdown 中显示的 v3 25k overall 只统计已完成任务，不能在四套件结束前与 v3 15k overall 直接比较。

实时对比表：

```text
lerobot/outputs/libero_eval/v2b_20k_vs_v3_15k_task_comparison.md
```

该文件现在包含：

- v2b 20k；
- v3 15k；
- v3 25k；
- `v3-15k − v2b-20k`；
- `v3-25k − v3-15k`。

监控 tmux：

```bash
tmux attach -t v3_eval_compare
```

---

## v3 评测命令

评测入口：

```text
scripts/libero_eval/eval_libero_v3_checkpoint.sh
```

当前 wrapper 的默认 `LIBERO_RESOURCE_ROOT` 有问题，必须显式复用已有 LIBERO 配置：

```bash
cd /data2/JM/Code/molmoact2

LIBERO_RESOURCE_ROOT=/data2/JM/Code/molmo_serious/molmoact2-main \
EVAL_GPU_IDS=7 \
EPISODES_PER_TASK=32 \
EVAL_BATCH_SIZE=32 \
MAX_EPISODES_RENDERED=32 \
bash scripts/libero_eval/eval_libero_v3_checkpoint.sh \
  lerobot/outputs/libero_goal_prior_v3/seed_1000/stage2/checkpoints/030000/pretrained_model
```

必须传入 `pretrained_model` 目录，而不是只传 checkpoint step 目录。

四套件执行顺序：

```text
libero_spatial → libero_object → libero_10 → libero_goal
```

单 GPU 时顺序执行；四 GPU 可通过 `EVAL_GPU_IDS="0 1 2 3"` 并行。

---

## 已知文档/代码待清理项

1. `EXPERIMENTS.md` 仍将 `v3-highLR` 写成“待启动”。
2. `EXPERIMENTS.md` 写的 highLR 输出是 `stage2_lr1e4`，实际运行目录是 `stage2`。
3. 原计划保留的 corrected-QUANTILES + low-LR canonical 5k 已被覆盖；当前 `stage2/checkpoints/005000` 也是 high-LR。
4. `eval_libero_v3_checkpoint.sh` 默认 resource root 会触发 LIBERO 交互询问并产生 `EOFError`，目前通过显式环境变量规避。
5. 当前对比 Markdown 文件名仍写 `v2b_20k_vs_v3_15k`，但内容已包含 v3 25k。
6. `monitor_checkpoint_task_comparison.py` 已改为自动生成相邻 checkpoint 的差值列。

---

## DROID 数据与迁移计划

完整 LeRobot DROID 数据已下载：

```text
/data0/JM/dataset/droid_1.0.1
```

固定 revision：

```text
0eabc778f959c54b8c5aa3626cc1128d2d2e54d4
```

数据规模：

- 95,658 episodes；
- 27,630,375 frames；
- 15 FPS；
- 三路相机；
- 约 412 GB。

适配方案：

- 保留 `observation.state = [joint_position(7), gripper(1)]`；
- 从已有 Cartesian pose 构建独立 `observation.ee_pose`；
- `RPY → axis-angle`；
- DROID target pose：

```text
[xyz(3), axis-angle(3), gripper(1)] = 7D
```

- 增加可配置 `goal_pose_feature_key`；
- pose encoder/decoder 与 state 维度解耦；
- 无需 FK；
- 第一版保持 `chunk_size=10`，15 FPS 下约 0.67 秒；
- 在完整 train split 上分别重算 state、ee_pose、action 统计量；
- 先本地小规模跑通，再交协作者全量训练。

---

## 代码与复现定位

主仓与子模块目标分支：

```text
molmoact2: feat/goal-pose-prior-v3
lerobot:   feat/goal-pose-prior-v3
```

文档中记录的固定提交：

```text
molmoact2: dbd5696
lerobot:   b3a70086
```

主要文档：

```text
README.md
README_GOAL_POSE_PRIOR_EN.md
EXPERIMENTS.md
scripts/libero_goal_prior_v3/README.md
```

主要训练脚本：

```text
scripts/libero_goal_prior_v3/train_stage1.sh
scripts/libero_goal_prior_v3/train_stage2.sh
```

主要评测/监控脚本：

```text
scripts/libero_eval/eval_libero_v3_checkpoint.sh
scripts/libero_eval/eval_libero_v2b_checkpoint.sh
scripts/monitor_checkpoint_task_comparison.py
scripts/monitor_baseline_vs_v2b_vs_v3_loss.py
```

---

## 下一步优先级

1. 确认 v3 Stage 2 30k checkpoint 完整保存。
2. 等待 v3 25k 四套件评测完成。
3. 启动 v3 30k 四套件评测。
4. 比较 15k / 25k / 30k，按 success rate 选择最佳 checkpoint。
5. 修正 `EXPERIMENTS.md` 和 v3 eval wrapper 的已知不一致。
6. 使用最佳 checkpoint 进行 LIBERO-Plus / LIBERO-PRO。
7. 开始 DROID 小规模数据读取、`ee_pose` 构建和两阶段 smoke test。
8. 论文阶段补 corrected QUANTILES + 相同训练预算的公平 baseline。
