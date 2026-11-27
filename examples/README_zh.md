# 按任务组织的示例

[English](README.md)

## 前置条件

已验证 Python 3.12/3.13、Genesis World 1.4.7，Genesis 最低版本为 1.4.2。1.4.2～1.4.6 会警告一次；1.4.7 同时为最高版本，更高版本会被拒绝。详见[版本策略](../ROADMAP.md)。

1. 克隆本仓库，并在**仓库根目录**运行命令。
2. 使用 editable install（仅支持源码树；不支持 wheel/sdist）：

```bash
git clone https://github.com/xArm-Developer/ufactory_genesis.git
cd ufactory_genesis
pip install -e ".[sim]"

# 可选：
#   pip install -e ".[sim,rl]"        # Linux/NVIDIA 到达与抓放
#   pip install -e ".[sim,real]"      # xArm SDK + Pinocchio/Coal（dry-run / sdk-sim / real 安全预检）
#   pip install -e ".[sim,dynamics]"  # 动力学参考检查

# 默认：PyPI 上的 torch 2.10.0（CUDA 12.8）。已验证构建：2.10.0+cu128。
# 需要其他构建时，先安装同版本 torch（cu126、cu130、cpu 未验证）：
#   pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126   # 或 cu130 / cpu
#   pip install -e ".[sim]"
```

`[sim]` 固定 `torch==2.10.0`；需要其他 torch 版本的包（如 lerobot）请使用独立虚拟环境。

3. 运行依赖 Genesis 的流程前，设置可写的 Numba 缓存目录：

```bash
# Linux / macOS
export NUMBA_CACHE_DIR=~/.cache/numba

# Windows PowerShell
# $env:NUMBA_CACHE_DIR="$env:USERPROFILE\.cache\numba"
```

4. 支持的 `--robot`：`xarm5`、`xarm6`、`xarm7`、`uf850`、`lite6`。
5. **Windows / 无可用 GPU：** 先从 CPU 索引（`--index-url https://download.pytorch.org/whl/cpu`）安装 `torch==2.10.0`，再 `pip install -e ".[sim]"`。可视化、抓放、装箱命令加 `--backend cpu`（有显卡但不被 Genesis 支持时，不要依赖 GPU 自动回退）。强化学习需要 Linux/NVIDIA GPU。
6. **Windows 下查看器无报错退出：** 若 `view_robot.py` 在 `Building visualizer...` 之后无窗口直接回到提示符：

```powershell
$env:MKL_THREADING_LAYER = "SEQUENTIAL"
python examples/visualization/view_robot.py --robot xarm6
```

长期生效：`conda env config vars set MKL_THREADING_LAYER=SEQUENTIAL -n <env>`。`--headless` 不受影响。

本版公开目录：


| 目录 | 用途 |
|---|---|
| `visualization/` | GLB 机型与末端工具查看器，见[可视化](#可视化) |
| `kinematics/` | FK/IK / 机型校验包装，见[运动学与动力学仿真](#运动学与动力学仿真) |
| `simulation/` | 轨迹控制任务仿真（[抓放](simulation/pick_place/README_zh.md)、[装箱](simulation/packaging/README_zh.md)），见[仿真索引](simulation/README_zh.md) |
| `rl/` | 裸臂 Reach → 桌面 Pick & Place → 装箱（待实现）；Linux/NVIDIA，见 [RL 索引](rl/README_zh.md) |

学习顺序及训练、推理指南见 [RL 索引](rl/README_zh.md)。

## 支持机型与末端

| 机械臂 | 可用末端 | 相机支架适配 | 仿真与预检 | 真机装箱 |
|---|---|---|---|---|
| `xarm5` | Gripper G2、Bio Gripper G2、Robotiq 2F-85、Robotiq 2F-140、Vacuum Gripper | 裸臂、Gripper G2、Robotiq 2F-85、Robotiq 2F-140、Vacuum Gripper（不适配 Bio Gripper G2） | 支持 | — |
| `xarm6` | Gripper G2、Bio Gripper G2、Robotiq 2F-85、Robotiq 2F-140、Vacuum Gripper | 裸臂、Gripper G2、Robotiq 2F-85、Robotiq 2F-140、Vacuum Gripper（不适配 Bio Gripper G2） | 支持 | Gripper G2 |
| `xarm7` | Gripper G2、Bio Gripper G2、Robotiq 2F-85、Robotiq 2F-140、Vacuum Gripper | 裸臂、Gripper G2、Robotiq 2F-85、Robotiq 2F-140、Vacuum Gripper（不适配 Bio Gripper G2） | 支持 | — |
| `uf850` | Gripper G2、Bio Gripper G2、Robotiq 2F-85、Robotiq 2F-140、Vacuum Gripper | 裸臂、Gripper G2、Robotiq 2F-85、Robotiq 2F-140、Vacuum Gripper（不适配 Bio Gripper G2） | 支持 | — |
| `lite6` | Lite6 Gripper、Lite6 Vacuum Gripper | 裸臂、Lite6 Gripper、Lite6 Vacuum Gripper | 支持 | Lite6 Gripper |

所有机械臂均支持加装[相机支架](../assets/urdf/mounts/camera_stand/README.md)（可单独安装或搭载一台视野沿工具轴方向的深度相机）；装上支架后，末端及其 TCP 沿工具轴前移 3 mm：

| 配件 key | 名称 | 说明 |
|---|---|---|
| `camera_stand` | [相机支架](../assets/urdf/mounts/camera_stand/README.md) | 3 mm 铝合金支架（48 g） |
| `d405` | [Intel RealSense D405](../assets/urdf/cameras/d405/README.md) | 支架 + D405 短距双目深度相机（58 g） |
| `d435` | [Intel RealSense D435](../assets/urdf/cameras/d435/README.md) | 支架 + D435 双目深度相机（75 g） |
| `gemini_305` | [奥比中光 Gemini 305](../assets/urdf/cameras/gemini_305/README.md) | 支架 + Gemini 305 双目深度相机（68 g） |
| `zed_x_nano` | [Stereolabs ZED X Nano](../assets/urdf/cameras/zed_x_nano/README.md) | 支架 + ZED X Nano 双目相机（75 g） |

- 配置中的 URDF 引用示例：`assembly:gripper_g2+d435`、`assembly:gripper_g2+ft_sensor_ai1500+d435`、`assembly:lite6_gripper+d435`、`assembly:bare+ft_sensor_ai1500`、`assembly:bare+camera_stand`、`assembly:bare+mounting_plate` 或 `assembly:gripper_g2+d435+mounting_plate`。
- 真机说明与安全约束见[真机操作与硬件校验](#真机操作与硬件校验)；G2 仿真使用 `g2_calibrated_v1`，策略训练与当前 v3 检查点契约见[抓放 RL 指南](rl/pick_place/README_zh.md)。

`xarm5`、`xarm6`、`xarm7` 与 `uf850` 均支持在末端法兰上加装 [AI1500 六维力矩传感器](../assets/urdf/accessories/ft_sensor_ai1500/README.md)（沿法兰 `+Z` 抬升 `58.6 mm`、`0.6 kg`，位于相机支架、转接板及末端执行器之前）。

全部 5 款机械臂均支持加装对应铝合金安装底板（安装在 `link_base` 下方，上安装面 `z = 0`，不改变机械臂运动学坐标）：

| 安装底板 | 适配机型 | 外形包络（mm） | 独立查看 `--tool` |
|---|---|---|---|
| [xArm 安装底板](../assets/urdf/mounts/xarm_mounting_plate/README.md) | `xarm5`、`xarm6`、`xarm7` | `136 × 185 × 8` | `xarm_mounting_plate` |
| [UF850 安装底板](../assets/urdf/mounts/uf850_mounting_plate/README.md) | `uf850` | `190 × 260 × 10` | `uf850_mounting_plate` |
| [Lite 6 安装底板](../assets/urdf/mounts/lite6_mounting_plate/README.md) | `lite6` | `130 × 190 × 5` | `lite6_mounting_plate` |

## 可视化

```bash
# xArm6 裸臂静态查看
python examples/visualization/view_robot.py --robot xarm6

# xArm6 + Gripper G2 静态查看（固定张开位姿）
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2

# xArm6 + Gripper G2 开合演示
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2 --gripper-demo

# xArm6 + Robotiq 2F-85 / 2F-140 开合演示（自动装入 11 mm robotiq_mounting 转接件）
python examples/visualization/view_robot.py --robot xarm6 --robotiq-2f-85 --gripper-demo
python examples/visualization/view_robot.py --robot xarm6 --robotiq-2f-140 --camera d435 --gripper-demo

# UF850 + Bio Gripper G2 静态查看（固定张开位姿）
python examples/visualization/view_robot.py --robot uf850 --bio-gripper-g2

# xArm6 + Bio Gripper G2 开合演示
python examples/visualization/view_robot.py --robot xarm6 --bio-gripper-g2 --gripper-demo

# xArm6 + Vacuum Gripper 静态查看
python examples/visualization/view_robot.py --robot xarm6 --vacuum-gripper

# xArm6 + 相机支架 + ZED X Nano + Gripper G2 开合演示
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2 --gripper-demo --camera zed_x_nano

# UF850 + 相机支架 + D435 + Vacuum Gripper 静态查看
python examples/visualization/view_robot.py --robot uf850 --vacuum-gripper --camera d435

# Lite6 + Lite6 Gripper 开合演示
python examples/visualization/view_robot.py --robot lite6 --lite6-gripper --gripper-demo

# Lite6 + 相机支架 + D435 + Lite6 Gripper 开合演示
python examples/visualization/view_robot.py --robot lite6 --lite6-gripper --gripper-demo --camera d435

# Lite6 + Lite6 Vacuum Gripper 静态查看
python examples/visualization/view_robot.py --robot lite6 --lite6-vacuum-gripper

# xArm6 + Gripper G2 + D435 + 安装底板
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2 --camera d435 --mounting-plate

# xArm6 + AI1500 六维力矩传感器 + D435 + Gripper G2
python examples/visualization/view_robot.py --robot xarm6 --ft-sensor --gripper-g2 --camera d435 --gripper-demo

# 末端工具、转接件、力矩传感器与安装底板独立查看
python examples/visualization/view_tool.py --tool gripper_g2
python examples/visualization/view_tool.py --tool robotiq_2f_85
python examples/visualization/view_tool.py --tool robotiq_2f_140
python examples/visualization/view_tool.py --tool robotiq_mounting
python examples/visualization/view_tool.py --tool ft_sensor_ai1500
python examples/visualization/view_tool.py --tool bio_gripper_g2
python examples/visualization/view_tool.py --tool lite6_gripper
python examples/visualization/view_tool.py --tool vacuum_gripper
python examples/visualization/view_tool.py --tool lite6_vacuum_gripper
python examples/visualization/view_tool.py --tool xarm_mounting_plate
python examples/visualization/view_tool.py --tool uf850_mounting_plate
python examples/visualization/view_tool.py --tool lite6_mounting_plate
```

`view_robot.py` 参数：

| 参数                                                                                 | 说明                                    |
| ---------------------------------------------------------------------------------- | ------------------------------------- |
| `--robot`                                                                          | 必选机型 key                              |
| `--gripper-g2` / `--bio-gripper-g2` / `--robotiq-2f-85` / `--robotiq-2f-140` / `--vacuum-gripper` / `--lite6-gripper` / `--lite6-vacuum-gripper` | 末端组合（最多选一个）                           |
| `--camera`                                                                         | `camera_stand`、`d405`、`d435`、`gemini_305` 或 `zed_x_nano`：加装相机支架，可单独装支架，也可再装对应相机（适配所有机型，支持裸臂及除 `--bio-gripper-g2` 外的末端） |
| `--ft-sensor`                                                                      | 在末端法兰加装六维力矩传感器（适配 `xarm5`/`xarm6`/`xarm7`/`uf850`；不填值时默认 `ft_sensor_ai1500`） |
| `--mounting-plate`                                                                 | 在 `link_base` 下方加装对应铝合金安装底板             |
| `--movable`                                                                        | 兼容保留（所有夹爪默认均使用逐连杆可动 GLB）              |
| `--gripper-demo`                                                                   | 循环开合，到位后保持 0.5 s 切换（需配合夹爪标志）          |
| `--pd`                                                                             | 平滑关节运动演示（约 0.873 rad/s），仅视觉           |
| `--show-tcp`                                                                       | 法兰 TCP 红色标记（默认隐藏）                     |
| `--diagnose`                                                                       | 无窗口 STL/GLB 连杆位姿诊断                    |
| `--headless`                                                                       | 不打开查看器窗口                              |
| `--backend`                                                                        | `cpu` / `gpu`（默认 `gpu`；无支持显卡时用 `cpu`） |

`view_tool.py` 支持 `--tool {gripper_g2,robotiq_2f_85,robotiq_2f_140,robotiq_mounting,ft_sensor_ai1500,bio_gripper_g2,lite6_gripper,vacuum_gripper,lite6_vacuum_gripper,xarm_mounting_plate,uf850_mounting_plate,lite6_mounting_plate}`、`--headless`、`--backend`，以及标定夹爪调参选项（`--position-mm`、`--speed-percent`、`--force-percent`，其中 `bio_gripper_g2` 另支持 `--speed-mm-s`）。




## 运动学与动力学仿真

```bash
# xArm6 离线 URDF 模型、FK 与 PD 控制自检（不连控制器）
python examples/kinematics/verify_robot.py --robot xarm6

# xArm6 动力学仿真校验（默认姿态 + 5 组随机姿态）
dynamics-sim-check --robot xarm6 --random-count 5
```

> 如需与真实控制器的 FK/IK 对比或硬件动力学检查，请参见 [真机操作与硬件校验](#真机操作与硬件校验)。



## 抓放仿真

多机型抓放的 Genesis 物理仿真与离线预检。参数、更多指令及覆盖配置见[抓放指南](simulation/pick_place/README_zh.md)。

```bash
# xArm6 + Gripper G2：抓放离线轨迹与碰撞预检（servo_j）
python examples/simulation/pick_place/run.py \
  --robot xarm6 --mode dry-run --executor servo_j

# Lite6 + Lite6 Gripper：抓放物理仿真并打开查看器（servo_cartesian）
python examples/simulation/pick_place/run.py \
  --robot lite6 --mode sim --executor servo_cartesian --visual
```

## 装箱展示仿真

多机型轨迹控制装箱。参数及更多指令见[装箱指南](simulation/packaging/README_zh.md)。

```bash
# xArm6 + Gripper G2：单轮装箱物理仿真并打开查看器（servo_j）
python examples/simulation/packaging/run.py \
  --robot xarm6 --mode sim --executor servo_j --visual
```

## 强化学习

Linux/NVIDIA 上的仿真示例按以下顺序运行：

1. [Reach](rl/reach/README_zh.md)：多机型（`xarm5`、`xarm6`、`xarm7`、`uf850`、`lite6`）裸臂到达，成功阈值为 5 mm。
2. [Pick & Place](rl/pick_place/README_zh.md)：xArm6 + Gripper G2 将方块搬运到桌面目标位置。
3. 装箱：将物体放入箱内，RL 示例待实现。

训练、推理指令见 [RL 索引](rl/README_zh.md)。上面的装箱展示采用轨迹控制。

## 真机操作与硬件校验

> [!CAUTION]
> **真机安全规范**：
>
> 1. 执行前需安装真机扩展依赖：`pip install -e ".[real]"`（包含 xArm SDK 与 Pinocchio/Coal 碰撞检测）。
> 2. 真实机械臂物理运动具有碰撞与损坏风险。驱动真机运动必须提供实机逐台标定文件（`--calibration`）并显式传入安全确认门（`--confirm-real`）。缺少任一项均会在连接前直接拒绝执行。
> 3. 真机模式单次运行强制限制为 **1 轮（cycle）**。



### 1. 硬件只读与状态对比指令（不驱动运动）

通过网络读取控制器数据或进行计算对比，真实关节不会产生位移：

```bash
# xArm6：Genesis URDF 与真实控制器 SDK 正运动学（FK）对比
python examples/kinematics/verify_fk.py --robot xarm6 --ip <ip>

# Lite6：Genesis URDF 与真实控制器 SDK 逆运动学（IK）对比
python examples/kinematics/verify_ik.py --robot lite6 --ip <ip>

# xArm6：在控制器虚拟仿真模式下，对预设测试点位进行自碰撞检查（真实关节不动作）
dynamics-sim-collision-check --robot xarm6 --ip <ip>

# Lite6 + Lite6 Gripper：装箱控制器内部虚拟仿真（sdk-sim，真实关节不动作）
python examples/simulation/packaging/run.py \
  --robot lite6 --mode sdk-sim --executor servo_cartesian \
  --ip <ip> --calibration path/to/exact.yaml

# xArm6 + Bio Gripper G2：抓放控制器内部虚拟仿真（sdk-sim，支持 servo_j / servo_cartesian）
python examples/simulation/pick_place/run.py \
  --robot xarm6 --gripper bio_gripper_g2 --mode sdk-sim --executor servo_j \
  --ip <ip> --calibration path/to/exact.yaml
python examples/simulation/pick_place/run.py \
  --robot xarm6 --gripper bio_gripper_g2 --mode sdk-sim --executor servo_cartesian \
  --ip <ip> --calibration path/to/exact.yaml
```



### 2. 真机物理运动指令（实际驱动机械臂与夹爪）

实际下发目标位姿与轨迹控制指令至真实机械臂：

```bash
# xArm6：驱动真机运动至验证姿态并对比关节力矩（需 --confirm-real）
dynamics-hardware-check --robot xarm6 --ip <ip> --confirm-real

# xArm6 + Gripper G2：抓放真机物理运动（需实机标定文件与 --confirm-real）
XARM_IP=<ip> python examples/simulation/pick_place/run.py \
  --robot xarm6 --mode real --executor servo_j \
  --calibration path/to/exact.yaml --confirm-real

# xArm6 + Bio Gripper G2：抓放真机物理运动（支持 servo_j / servo_cartesian，可选 --visual 镜像；需实机标定文件与 --confirm-real）
XARM_IP=<ip> python examples/simulation/pick_place/run.py \
  --robot xarm6 --gripper bio_gripper_g2 --mode real --executor servo_j --visual \
  --calibration path/to/exact.yaml --confirm-real
XARM_IP=<ip> python examples/simulation/pick_place/run.py \
  --robot xarm6 --gripper bio_gripper_g2 --mode real --executor servo_cartesian \
  --calibration path/to/exact.yaml --confirm-real

# Lite6 + Lite6 Gripper：装箱真机物理运动（当前仅支持 xArm6 + Gripper G2、Lite6 + Lite6 Gripper；需实机标定文件与 --confirm-real）
XARM_IP=<ip> python examples/simulation/packaging/run.py \
  --robot lite6 --mode real --executor servo_j \
  --calibration path/to/exact.yaml --confirm-real
```

> 注：xArm5、xArm7、UF850 的装箱 `--mode real` 会在连接控制器前自动拦截退出，直到对应机型启用真机夹爪支持。

