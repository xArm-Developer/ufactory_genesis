# 固定布局 xArm6 + Gripper G2 强化学习

这条纯仿真主线只训练固定 `+Y` 抓放布局，不宣称随机布局泛化、CPU/Windows 训练或
RL 真机部署。

[English](README.md)

## 参考环境与工件状态

参考环境为 Linux、NVIDIA GPU、Python 3.12/3.13、Genesis World 1.4.0、
Quadrants 1.3.0、PyTorch 2.10 和 RSL-RL 5.4.2；仓库锁文件是唯一参考：

```bash
uv sync --extra sim --extra rl
export NUMBA_CACHE_DIR="$HOME/.cache/numba"
```

环境只从 `simulation.substeps` 解析 32 个刚体子步，已删除的任务级字段属于非法配置。
0.625 ms 刚体步长是本项目为正式运行保留的保守裕量，并不表示 Genesis 1.3.3 必须
使用 32 子步。主要的非数值修复是给 G2 的 5 个联动等式统一使用柔顺求解参数；同一
参数在 8 子步下也通过了 128 个扰动接触环境的检查。

释放意图迟滞（配方 `release_command_margin_m`，18 mm）要求指令间隙显著越过闭合间隙
才认定松爪。此变更前保存的检查点评估时请传 `--release-command-margin-m 0.0005`。

内置的 seed 7 `model_299_g2stable.pt`（SHA-256 `f2142a63…f783f2`）在现行
`g2_stable_v1_3_3` 物理配置下训练，是默认评估目标。上一版接触模型训练的
`model_299.pt` 不再随仓库发布：它没有绑定 `g2_stable_v1_3_3` 物理配置，
当前评估器会按设计拒绝它。

## 分层引导披露

观测中包含 4 维脚本引导动作列（44→48，配方开关 `include_scripted_action_hint`）。
引导与行为克隆打标签用的是同一份哈希脚本专家配置，观测与标签不可能互相矛盾。
按 v0.2.12 先例，这属于公开披露的分层/仿真器引导策略：发布工件不得表述为脱离引导
仍能泛化的学习策略。2026-08-21 的诊断（`rl/reports/`）表明 44 维观测无法区分
"在取件点夹着立方体"与"在目标点夹着立方体"，克隆因此会在取件点松爪；引导列正是
用来消解这一歧义的。

## 从零训练路径

先确认脚本专家能够达到不可变的 `contact_v1` 目标：

```bash
python -m examples.rl.pick_place.evaluate --expert --headless -B 8 --episodes 8 \
  --acceptance-profile contact_v1 --require-target robustness
```

重新采集示教并在不加载任何旧策略的情况下做行为克隆。标准采集档为
256 环境 × 8000 步（约 205 万样本）；更小的档会把动作均方误差拟合得更低，
但闭环行为反而会崩：

```bash
python -m examples.rl.pick_place.pretrain_bc -B 256 --rollout-steps 8000 \
  --far-open-penalty 20.0 --seed 1 -e v0213g-fixed-bc-seed1
```

从这个新 BC actor 启动三个相互独立的 300 轮 PPO：

```bash
for seed in 1 7 17; do
  python -m examples.rl.pick_place.train -B 256 --max-iterations 300 \
    --seed "$seed" \
    --warm-start outputs/rl/pick_place/v0213g-fixed-bc-seed1/model_0.pt \
    -e "v0213g-fixed-ppo-seed${seed}"
done
```

产物只写入已忽略的 `outputs/rl/pick_place/`。不要把 `pretrained/` 当训练目录，
也不要加载旧物理基线训练出的检查点。

## 评估专家或显式指定的现行物理候选

完整工件包包含 `model_N.pt`、`config.yaml` 和
`model_N.checkpoint_manifest.json`。评估器先检查哈希、任务/机器人身份及完整运行时契约，
再使用 PyTorch 安全的 weights-only 模式加载。

```bash
# 现行纯物理接触基线
python -m examples.rl.pick_place.evaluate --expert \
  --headless -B 8 --episodes 8 --acceptance-profile contact_v1

# 本地完整工件包
python -m examples.rl.pick_place.evaluate \
  --checkpoint outputs/rl/pick_place/<run>/model_N.pt \
  --headless -B 8 --episodes 8 --acceptance-profile contact_v1
```

当前固定库绑定运行时配置 `654b538e…7ed`，SHA-256 为 `2238202c…eb0`：

```bash
python -m examples.rl.pick_place.evaluate \
  --checkpoint outputs/rl/pick_place/<run>/model_N.pt \
  --scenario-bank examples/rl/pick_place/scenarios/fixed_seed17000_n512.json \
  --headless -B 64 --episodes 64 --acceptance-profile contact_v1 \
  --require-target standard
```

正式动作噪声试验使用标准差 `0.02` 和新种子 `20260817`：

```bash
python -m examples.rl.pick_place.evaluate \
  --checkpoint outputs/rl/pick_place/<run>/model_N.pt \
  --scenario-bank examples/rl/pick_place/scenarios/fixed_seed17000_n512.json \
  --headless -B 64 --episodes 512 --acceptance-profile contact_v1 \
  --action-noise-std 0.02 --action-noise-bank 20260817 \
  --require-target robustness
```

## 发布要求

候选必须是非零 PPO 检查点，并在动作裁剪、IK 失败和 IK 跳变拒绝均为零的前提下满足：

- 种子 `1/7/17` × 并行数 `1/8/64`：9/9 组、219/219 回合；
- 独立固定库：完整与质量成功 64/64；
- 固定布局动作噪声 512 回合：完整与质量成功率均不低于 99%；
- 最终 XY、夹起前拖动、松爪后漂移的 P99 分别不超过 10、5、3 mm。

`Full success` 同时要求抓取、抬升、有效放下/松爪、最终位姿和速度、保持时间及全部
质量限制。即使训练指标或局部阶段看起来较好，筛选失败也不会发布工件。

## 发布结果

内置的 `model_299_g2stable.pt` 在 Genesis World 1.3.3 的 `g2_stable_v1_3_3`
物理配置下完成全部训练。48 维观测修复了 2026-08-21 定位的阶段标签歧义。
通过验收的行为克隆从收起姿势运行 8/8 成功，随后种子 1、7、17 分别完成
300 轮 PPO。最终选择 seed 7 检查点，训练使用 `place_phase_reset_frac: 0.40`
和学习率 `1e-5`。

同一个候选通过全部 9 组评估种子/并行数组合（219/219 回合）、独立固定场景库
（完整与质量成功 64/64），以及固定场景 512 回合、标准差 0.02 的动作噪声检查
（完整与质量成功 512/512）。噪声检查的最终 XY 误差、夹起前拖动、松爪后漂移
P99 分别为 0.93、0.35、0.07 mm；动作裁剪、IK 失败、IK 跳变拒绝和松爪后再接触
均为零。机器可读证据见 `pretrained/evaluation_summary.json`。

**Genesis 1.4.0 迁移说明：** 内置策略在 Genesis 1.3.3 下训练。迁移到 1.4.0 后，
由于上游求解器变更（IK 延迟分配、质量 API 重命名等），物理行为可能有所不同。
在 1.4.0 上使用前**必须重新验证或重新训练**。此次迁移未执行 GPU 仿真或策略评估。
