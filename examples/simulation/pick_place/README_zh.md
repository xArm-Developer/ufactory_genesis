# 抓放仿真

[English](README.md) · [仿真索引](../README_zh.md) · [示例指南](../../README_zh.md)

多机型抓放的 Genesis 物理仿真与离线轨迹、碰撞预检。安装方式及运行目录见[前置条件](../../README_zh.md#前置条件)。

`examples/simulation/pick_place/run.py` 委托给稳定的 `ufactory-pick-place` 控制台命令，两者接受相同参数。

## 参数

| 参数               | 取值 / 说明                                                          |
| ---------------- | ---------------------------------------------------------------- |
| `--robot`        | 必选：`xarm5` / `xarm6` / `xarm7` / `uf850` / `lite6`               |
| `--gripper`      | 可选：`default` / `g2` / `bio_gripper_g2` / `lite6` / `robotiq_2f_85` / `robotiq_2f_140` / `vacuum_gripper`（`bio_gripper_g2`、`robotiq_2f_*`、`vacuum_gripper` 支持 `xarm5`、`xarm6`、`xarm7`、`uf850`；`bio_gripper_g2` 在 `sim` 下运行微孔板任务，非 `sim` 模式（`dry-run` / `sdk-sim` / `real`）支持 `--executor servo_j` / `servo_cartesian` 且真机模式仅放行 `xarm6`；`vacuum_gripper` 支持 `dry-run` / `sdk-sim` / `real`（`servo_j` / `servo_cartesian`，可配 `--visual` 空吸镜像，不含物理负压 `sim`），已在 `xarm5`（1 代直插线）上完成真机验证；`robotiq_2f_*` 连接真实控制器（`sdk-sim` / `real`）目前只支持 `xarm6` + `robotiq_2f_85`，已在硬件上验证（含 11 mm 法兰转接件）；其他 Robotiq 组合只支持 `sim` / `dry-run`） |
| `--vacuum-hardware-version` | 可选：`1`（1 代插线 `TGPIO 0/1`，默认）或 `2`（2 代触点 `TGPIO 3/4`），配合 `--gripper vacuum_gripper` 使用 |
| `--mode`         | 仿真必选：`sim`（Genesis 物理仿真）、`dry-run`（离线轨迹与碰撞预检）                    |
| `--executor`     | `servo_j` 或 `servo_cartesian`；Bio G2 在 `sim` 下默认 `servo_cartesian`，非 `sim` 模式及其他夹爪必选 |
| `--backend`      | 可选：`cpu` / `gpu`；覆盖 `simulation.backend`（无 Genesis 支持显卡时用 `cpu`） |
| `--visual`       | 可选：`sim` 模式下强制打开 Genesis 查看器视窗；`real` 模式下打开异步运动学镜像视窗（Bio G2 为空夹，不搬运方块） |
| `--config`       | 可选：严格的局部覆盖 YAML 路径                                               |
| `--print-config` | 可选：打印解析后的运行时 YAML 后退出                                            |
| `--report`       | 可选：报告输出路径                                                        |

## 运行指令

```bash
# xArm6 + Gripper G2：打印解析后的抓放运行时配置并退出
python examples/simulation/pick_place/run.py \
  --robot xarm6 --mode dry-run --executor servo_j --print-config

# xArm6 + Gripper G2：抓放离线轨迹与碰撞预检（servo_j）
python examples/simulation/pick_place/run.py \
  --robot xarm6 --mode dry-run --executor servo_j

# Lite6 + Lite6 Gripper：抓放物理仿真并打开查看器（servo_cartesian）
python examples/simulation/pick_place/run.py \
  --robot lite6 --mode sim --executor servo_cartesian --visual

# xArm6 + Bio Gripper G2：微孔板抓放物理仿真并打开查看器（默认 servo_cartesian）
python examples/simulation/pick_place/run.py \
  --robot xarm6 --gripper bio_gripper_g2 --mode sim --visual

# xArm6 + Bio Gripper G2：抓放离线轨迹与碰撞预检（支持 servo_j / servo_cartesian）
python examples/simulation/pick_place/run.py \
  --robot xarm6 --gripper bio_gripper_g2 --mode dry-run --executor servo_j
python examples/simulation/pick_place/run.py \
  --robot xarm6 --gripper bio_gripper_g2 --mode dry-run --executor servo_cartesian

# xArm6 + Robotiq 2F-85 / 2F-140：抓放物理仿真
python examples/simulation/pick_place/run.py \
  --robot xarm6 --gripper robotiq_2f_85 --mode sim --executor servo_j
python examples/simulation/pick_place/run.py \
  --robot xarm6 --gripper robotiq_2f_140 --mode sim --executor servo_cartesian

# Lite6 + Lite6 Gripper：使用 CPU 后端运行抓放物理仿真（无支持 GPU 时）
python examples/simulation/pick_place/run.py \
  --robot lite6 --mode sim --executor servo_cartesian --backend cpu

# xArm6 + Gripper G2：使用局部配置覆盖文件运行抓放离线预检
python examples/simulation/pick_place/run.py \
  --robot xarm6 --mode dry-run --executor servo_j \
  --config examples/simulation/pick_place/runtime.example.yaml
```

## 覆盖配置

`runtime.example.yaml` 是严格的局部覆盖文件；未写出的机器人、几何、运动、安全和仿真值仍由 `assets/configs/runtime` 解析。

## 真机

真实机械臂抓放部署请参见[真机操作与硬件校验](../../README_zh.md#真机操作与硬件校验)。
