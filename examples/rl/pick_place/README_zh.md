# 2. Pick & Place：桌面抓放

xArm6 搭配 Gripper G2，在仿真中抓取随机位置的方块，搬运到固定目标点并放下。策略使用 PPO 从零训练。

[English](README.md) · [RL 示例](../README_zh.md) · [前一例：Reach](../reach/README_zh.md)

## 准备

在项目根目录运行以下命令。需要 Linux、NVIDIA GPU；交互式推理还需要桌面显示环境。

```bash
pip install -e ".[sim,rl]"
```

## 任务说明

- 每回合从末端位置 `(300, 0, 300) mm`、夹爪开度 `50 mm` 开始。`30 mm` 方块的起点在 `x ∈ [300, 500] mm`、`y ∈ [-200, 200] mm` 范围内随机采样，目标点为 `(300, 300) mm`。
- 控制周期 `20 ms`，最长 `400` 步。策略使用 `70` 维观测，输出末端 XYZ 和夹爪开度的 `4` 维增量动作；每步最大增量分别为 `5 mm` 和 `3 mm`。
- 成功要求：起抓拖行 ≤10 mm、倾角 ≤15°；方块最低点抬升至离桌 ≥65 mm；落点误差 ≤20 mm、放置倾角 ≤15°；在离桌 0–3 mm 且夹爪开度 ≥35 mm 时脱爪，触桌后末端额外下压 ≤1 mm，并稳定保持 0.2 s。机械臂碰桌判为失败。

## 训练

训练输出位于 `outputs/rl/pick_place/`。以下命令均在项目根目录运行。

### 一键训练与验收

```bash
python -m examples.rl.pick_place.run_seeds \
  --seeds 1 2 3 -B 2048 --max-iterations 2000
```

对三个种子依次训练、独立评估 64 回合，并录制单环境回放。每 50 轮评估（采用 `seed=17000` 64 回合 + `seed=17500` 128 回合共 192 回合双验证集，要求合并成功率 ≥99.4%、最大下压 ≤0.8 mm 且近期训练曲线稳定）并更新 `model_best.pt`；连续两次达标后提前停止。结果保存在各自的 `pick_place_e2e_s<种子>/` 目录，汇总见 `e2e_validation/summary.json`。验收未通过时命令返回非零退出码。

### 单独训练或续训

```bash
python -m examples.rl.pick_place.train \
  -e pick_place_e2e_s1 --seed 1 -B 2048 --max-iterations 2000
```

续训时指定原检查点和新的实验名，`-B` 必须与原训练一致：

```bash
python -m examples.rl.pick_place.train \
  --resume outputs/rl/pick_place/pick_place_e2e_s1/model_best.pt \
  -e pick_place_e2e_s1_resume --seed 1 -B 2048 --max-iterations 200
```

`-B` 是并行环境数，显存不足时可减小。`--max-iterations` 是训练轮数上限；`--eval-start-iteration` 可推迟周期性严格评估起始轮次；`--early-stop-evaluations 0` 可关闭提前停止。已有实验目录不会被覆盖，单独训练可通过 `-e` 指定新目录。

## 基准与预训练结果

在 `v0.3.11` 的 10 种子 `C10` 基准（`seeds 1..10`，`-B 2048`，上限 `1200` 轮）中，**8/10** 个种子通过独立盲测（`seed = 18000 + s`，64 回合成功率 ≥98%、最大下压 ≤1.0 mm）与单环境 `GLB` 回放，通过种子合计盲测成功率 **`508/512 (99.22%)`**，全种子 `ik_rejects = 0.00`（其中 4 个种子 `64/64 = 100.0%`，4 个种子 `63/64 = 98.44%`）。

托管于 Hugging Face 的预训练检查点（`pick_place/xarm6-pick-place/seed-7/model_best.pt`，第 `650` 轮早停）已在 Colab L4 与本地 **NVIDIA GeForce RTX 4060 Ti (8 GB)** 上双重验证通过：

| 机械臂 + 夹爪 | 观测 / 动作维 | 检查点 | 双验证集 (`17000` / `17500`) | 独立盲测 (`18007`, 64 回合) | 最大下压 (`18007`) | 落点误差均值 / 中位 (`18007`) | GLB `B=1` 回放 |
|---|---:|---|---|---|---|---|---|
| `xarm6` + `gripper_g2` | 70 / 4 | `seed-7/model_best.pt` | **100%** (64/64) / **100%** (128/128) | **100%** (64/64) | `0.477 mm` (≤1.0 mm) | `6.98` / `6.14 mm` (≤20 mm) | **PASS** |

## 评估与推理

如果不需要在本地从零训练，可先从 Hugging Face（[`UFACTORY-Developer/ufactory_genesis_rl`](https://huggingface.co/UFACTORY-Developer/ufactory_genesis_rl)）下载预训练检查点到 `outputs/rl/pick_place/`：

```bash
# 如未安装 Hugging Face CLI，先执行：pip install huggingface_hub
hf download UFACTORY-Developer/ufactory_genesis_rl --include "pick_place/**" --local-dir outputs/rl
```

以下命令以下载后的 `outputs/rl/pick_place/xarm6-pick-place/seed-7/model_best.pt` 为例（若使用本地训练结果，将 `--checkpoint` 替换为 `outputs/rl/pick_place/pick_place_e2e_s1/model_best.pt` 或传入 `-e pick_place_e2e_s1`）。请先评估检查点，再用于推理。

### 评估

以下示例评估预训练种子 7 的最佳检查点，要求 64 回合成功率 ≥98%、最大下压 ≤1 mm：

```bash
python -m examples.rl.pick_place.evaluate \
  --checkpoint outputs/rl/pick_place/xarm6-pick-place/seed-7/model_best.pt \
  -B 32 --episodes 64 --seed 18007 \
  --min-success-rate 0.98 --max-press-mm 1.0 \
  --quiet-episodes --out outputs/rl/pick_place/xarm6-pick-place/seed-7/test.json
```

`--out` 同时生成 JSON 汇总和 CSV 逐回合结果；未达到门限时命令返回退出码 `1`。

### 实时推理与录像

```bash
# 打开交互窗口
python -m examples.rl.pick_place.visualize \
  --checkpoint outputs/rl/pick_place/xarm6-pick-place/seed-7/model_best.pt -B 1

# 无头录制单回合视频
python -m examples.rl.pick_place.visualize \
  --checkpoint outputs/rl/pick_place/xarm6-pick-place/seed-7/model_best.pt \
  -B 1 --episodes 1 --headless \
  --record outputs/rl/pick_place/xarm6-pick-place/seed-7/replay.mp4
```

`-e` 指定 `outputs/rl/pick_place/` 下的实验名，默认加载对应目录中的 `model_best.pt`；也可使用 `--checkpoint` 显式指定检查点文件。`--headless` 无需桌面窗口，`--record` 指定 MP4 输出路径。检查点、`config.yaml` 与 `model_best.checkpoint_manifest.json` 需放在同一目录。

单独运行 `train`、`evaluate` 或 `visualize` 时，可用 `--recipe`、`--runtime-config` 指定训练配方和场景配置。检查点会校验配置；修改任务配置后需重新训练。
