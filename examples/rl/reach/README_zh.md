# 1. Reach：机械臂目标到达

不安装末端工具的机械臂从默认关节姿态出发，末端到达目标空间内的随机目标点。使用 PPO 在仿真中从零训练，末端与目标距离严格小于 **5 mm** 判为成功。支持 `xarm5`、`xarm6`（默认）、`xarm7`、`uf850`、`lite6` 共 5 款机械臂。

[English](README.md) · [RL 示例顺序](../README_zh.md)

## 运行环境

- Linux + NVIDIA GPU，安装 `pip install -e ".[sim,rl]"`（见[示例总览](../../README_zh.md)）。
- 在项目根目录以模块方式运行，例如 `python -m examples.rl.reach.train`。
- GLB 实时推理需要桌面显示环境。

## 任务定义

1. **回合**：控制周期 20 ms，每回合最长 5 s。每回合均从末端法兰 TCP `(300, 0, 300) mm`、姿态 `RPY = (180°, 0°, 0°)` 对应的初始关节角开始，目标点在目标空间内均匀采样；距离小于 5 mm 时判成功并结束回合，超时判失败。
2. **观测**（`3 * dof + 6` 维，按顺序拼接）：
   - 关节角度 `joint_pos`（`dof` 维，rad）
   - 关节角速度 `joint_vel`（`dof` 维，rad/s）
   - 末端在世界坐标系下的位置 `ee_pos`（3 维，m）
   - 目标相对末端的位移 `target_pos - ee_pos`（3 维，m）
   - 关节位置指令与实测角度之差 `position_command - joint_pos`（`dof` 维，rad）
3. **动作**（`dof` 维）：`[-1, 1]` 归一化关节位置增量，线性映射到每步最大 0.01 rad（关节速度上限 0.5 rad/s × 20 ms）。累加后的关节位置指令限制在 URDF 关节限位内。
4. **奖励**：距离奖励 `exp(-10 d)`（权重 1.0）与关节速度惩罚 `-Σ q̇²`（权重 0.001），两项按控制周期缩放；成功时一次性奖励 2.0。
5. **训练算法**：rsl-rl PPO，Actor 与 Critic 均为 `[256, 256, 128]` ELU MLP，动作采用 Beta 分布。每轮每个环境采样 24 步，默认训练 300 轮、种子 7，每 25 轮保存一次检查点，最后为 `model_299.pt`。

## 机型与实测数据

| `--robot` | 自由度 (`dof`) | 观测 / 动作维 | 目标空间 (m) |
|---|---|---|---|
| `xarm5` | 5 | 21 / 5 | x `0.15–0.55`、y `-0.30–0.30`、z `0.05–0.50` |
| `xarm6`（默认） | 6 | 24 / 6 | x `0.15–0.55`、y `-0.30–0.30`、z `0.05–0.50` |
| `xarm7` | 7 | 27 / 7 | x `0.15–0.55`、y `-0.30–0.30`、z `0.05–0.50` |
| `uf850` | 6 | 24 / 6 | x `0.15–0.55`、y `-0.30–0.30`、z `0.05–0.50` |
| `lite6` | 6 | 24 / 6 | x `0.15–0.35`、y `-0.20–0.20`、z `0.05–0.35` |

以下数据基于 **NVIDIA GeForce RTX 4060 Ti (8 GB)**，使用[一键运行](#方式一一键运行)命令（`-B 4096`、种子 7、300 轮）实测：

| `--robot` | 显存占用 | 300 轮训练时间（总 / PPO 迭代） | 选中检查点 | 验收成功率（17000 / 17001） | 平均终止距离（17000 / 17001） |
|---|---|---|---|---|---|
| `xarm5` | 1.82 GB | 192 s / 159 s | `model_299.pt` | 99.6% / 99.8% | 4.48 / 4.50 mm |
| `xarm6` | 2.16 GB | 257 s / 240 s | `model_299.pt` | 100% / 100% | 4.23 / 4.24 mm |
| `xarm7` | 2.60 GB | 272 s / 255 s | `model_299.pt` | 99.8% / 99.2% | 4.39 / 4.52 mm |
| `uf850` | 2.32 GB | 297 s / 280 s | `model_299.pt` | 99.6% / 99.6% | 4.26 / 4.34 mm |
| `lite6` | 1.94 GB | 200 s / 167 s | `model_299.pt` | 99.8% / 99.6% | 4.66 / 4.71 mm |

- 显存占用为训练进程峰值；总时间包含 Genesis 初始化与场景构建。
- 显存不足时减小 `-B`（并行环境数），续训时 `-B` 须与原训练一致。

## 训练、筛选与验收

一键运行和分步运行的输出目录结构相同，二选一即可。以下以 `xarm6`、输出目录 `outputs/rl/reach/xarm6-reach` 为例；训练其他机型时同时替换 `--robot` 和目录名。

### 方式一：一键运行

```bash
python -m examples.rl.reach.run_experiments \
  --robot xarm6 \
  --output outputs/rl/reach/xarm6-reach \
  --seeds 7 -B 4096 --max-iterations 300
```

依次执行以下步骤，任一步失败即停止：

1. **冒烟测试**：64 个环境训练 2 轮、续训 1 轮并评估 8 个回合。
2. **训练**：每个种子输出到 `<output>/seed-<种子>/`。
3. **筛选**：用种子 17002 的 128 个目标评估全部检查点，按成功率（降序）、平均终止距离（升序）、平均成功用时（升序）选出最佳检查点，写入 `<output>/seed-<种子>/checkpoint-selection.json`。
4. **验收**：用种子 17000、17001 各 512 个目标评估选中的检查点，每组要求成功率 ≥95%、平均终止距离 ≤10 mm，结果写入 `<output>/seed-<种子>/acceptance-result.json`。

- `--output` 目录须不存在；省略时为 `outputs/rl/reach/reach-<时间戳>`。
- 多种子训练将 `--seeds 7` 改为 `--seeds 1 7 17`。
- 各步骤的日志、进程记录和 GPU 采样保存在 `<output>/`，总体状态见 `experiment-status.json`。
- 评估目标由固定种子在 CPU 上生成，与并行环境数和 GPU 无关，可复现。

### 方式二：分步运行

```bash
# 1. 训练
python -m examples.rl.reach.train \
  --robot xarm6 \
  --log-dir outputs/rl/reach/xarm6-reach/seed-7 \
  --seed 7 -B 4096 --max-iterations 300

# 2. 筛选：种子 17002，128 个目标
python -m examples.rl.reach.run_experiments \
  --stage select --robot xarm6 --output outputs/rl/reach/xarm6-reach --seeds 7

# 3. 验收：种子 17000、17001，各 512 个目标
python -m examples.rl.reach.run_experiments \
  --stage accept --robot xarm6 --output outputs/rl/reach/xarm6-reach --seeds 7
```

- 筛选与验收规则同方式一；`--robot` 默认为 `xarm6`，须与训练机型一致。
- 筛选与验收按 `<output>/seed-<种子>/` 查找检查点，`--log-dir` 需按此结构命名。
- 训练目录已存在时报错，可更换名称，或加 `--overwrite` 删除原目录后重新训练。
- 训练曲线：`tensorboard --logdir outputs/rl/reach/xarm6-reach`。

### 续训（可选）

```bash
python -m examples.rl.reach.train \
  --resume outputs/rl/reach/xarm6-reach/seed-7/model_299.pt \
  --log-dir outputs/rl/reach/xarm6-reach/seed-7-resume \
  --seed 7 -B 4096 --max-iterations 100
```

机型从检查点目录的 `config.yaml` 自动识别；`--max-iterations 100` 表示追加 100 轮，检查点编号接续原训练（`model_300.pt` 至 `model_399.pt`）；`-B` 须与原训练一致。

## 评估与推理

如无需本地从零训练，可直接从 Hugging Face（[`UFACTORY-Developer/ufactory_genesis_rl`](https://huggingface.co/UFACTORY-Developer/ufactory_genesis_rl)）下载预训练检查点到 `outputs/rl/reach/`：

```bash
# 如尚未安装 hf 命令行工具：pip install huggingface_hub
# 下载全部 5 款机型预训练检查点（约 15 MB）
hf download UFACTORY-Developer/ufactory_genesis_rl --include "reach/**" --local-dir outputs/rl

# 或仅下载单款机型（以 xarm6 为例）
hf download UFACTORY-Developer/ufactory_genesis_rl --include "reach/xarm6-reach/**" --local-dir outputs/rl
```

筛选选中的检查点不一定是最后一个，可先通过以下命令查看：

```bash
# 输出示例："selected_checkpoint": "model_299.pt",
grep selected_checkpoint outputs/rl/reach/xarm6-reach/seed-7/checkpoint-selection.json
```

以下以 `outputs/rl/reach/xarm6-reach/seed-7/model_299.pt` 为例，按实际查到的文件名替换 `--checkpoint`。

### 单独评估

```bash
python -m examples.rl.reach.evaluate \
  --checkpoint outputs/rl/reach/xarm6-reach/seed-7/model_299.pt \
  -B 128 --episodes 512 --seed 17000 \
  --min-success-rate 0.95 --max-mean-distance-m 0.010 \
  --quiet-episodes --summary-json outputs/rl/reach/xarm6-reach/seed-7/evaluation-17000.json
```

- 机型从检查点目录的 `config.yaml` 自动识别。
- 未达到 `--min-success-rate` 或 `--max-mean-distance-m` 门限时退出码为 1；报告文件已存在时报错。
- 第二组将 `--seed` 改为 `17001`，报告文件名改为 `evaluation-17001.json`。

### GLB 实时推理

```bash
# 单臂
python -m examples.rl.reach.visualize \
  --checkpoint outputs/rl/reach/xarm6-reach/seed-7/model_299.pt

# 四臂
python -m examples.rl.reach.visualize \
  --checkpoint outputs/rl/reach/xarm6-reach/seed-7/model_299.pt -b 4
```

- 使用 GLB 外观模型显示；每台机械臂成功后独立保持 1 s 再开始下一回合。关闭窗口或按 Ctrl+C 退出。
- 检查点、`config.yaml` 和对应的 `*.checkpoint_manifest.json` 须在同一目录（训练输出目录默认满足）。

## 配置与文件

| 文件 | 说明 |
|---|---|
| `recipe.yaml` | 并行环境数、奖励权重、PPO 超参数 |
| `assets/configs/runtime/tasks/reach.yaml` | 公共任务参数：控制周期、回合时长、成功距离、动作缩放、默认目标空间 |
| `assets/configs/runtime/tasks/robots/*_reach.yaml` | 各机型裸臂 URDF 与初始关节角；`lite6` 另设较小目标空间 |
| `env.py` | 训练与评估环境：状态初始化、动作执行、奖励和回合统计 |
| `realtime_env.py`、`realtime_viewer.py` | 实时推理：复用训练环境，另设场景布局、成功保持和实时步进 |
| `train.py`、`evaluate.py`、`visualize.py`、`run_experiments.py` | 命令行入口 |

- `--recipe`、`--runtime-config` 可指定自定义文件。
- 检查点加载时会校验任务配置与动作契约，修改任务配置后需重新训练。
- 训练日志、检查点清单和缓存设置与 Pick & Place 共用 `ufactory/training` 实现。
