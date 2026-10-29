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

对三个种子依次训练、独立评估 64 回合，并录制单环境回放。每 50 轮评估并更新 `model_best.pt`；连续两次达到成功率 ≥98%、最大下压 ≤1 mm 后提前停止。结果保存在各自的 `pick_place_e2e_s<种子>/` 目录，汇总见 `e2e_validation/summary.json`。验收未通过时命令返回非零退出码。

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

`-B` 是并行环境数，显存不足时可减小。`--max-iterations` 是训练轮数上限；`--early-stop-evaluations 0` 可关闭提前停止。已有实验目录不会被覆盖，单独训练可通过 `-e` 指定新目录。

## 本地检查点评估与推理

以下命令使用本地训练生成的 `model_best.pt`。请先评估检查点，再用于推理。

### 评估

以下示例评估种子 1 的最佳检查点，要求 64 回合成功率 ≥98%、最大下压 ≤1 mm：

```bash
python -m examples.rl.pick_place.evaluate \
  --checkpoint outputs/rl/pick_place/pick_place_e2e_s1/model_best.pt \
  -B 32 --episodes 64 --seed 18001 \
  --min-success-rate 0.98 --max-press-mm 1.0 \
  --quiet-episodes --out outputs/rl/pick_place/pick_place_e2e_s1/test.json
```

`--out` 同时生成 JSON 汇总和 CSV 逐回合结果；未达到门限时命令返回退出码 `1`。

### 实时推理与录像

```bash
# 打开交互窗口
python -m examples.rl.pick_place.visualize -e pick_place_e2e_s1 -B 1

# 无头录制单回合视频
python -m examples.rl.pick_place.visualize \
  -e pick_place_e2e_s1 -B 1 --episodes 1 --headless \
  --record outputs/rl/pick_place/pick_place_e2e_s1/replay.mp4
```

`-e` 指定实验名，默认加载对应目录中的 `model_best.pt`；也可使用 `--checkpoint` 指定文件。`--headless` 无需桌面窗口，`--record` 指定 MP4 输出路径。

单独运行 `train`、`evaluate` 或 `visualize` 时，可用 `--recipe`、`--runtime-config` 指定训练配方和场景配置。检查点会校验配置；修改任务配置后需重新训练。
