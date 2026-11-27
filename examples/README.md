# Task-oriented examples

[中文](README_zh.md)

## Prerequisites

Validated with Python 3.12/3.13 and Genesis World 1.4.7; the minimum Genesis version is 1.4.2. Versions 1.4.2 to 1.4.6 warn once; 1.4.7 is also the maximum, and newer versions are rejected. See the [version policy](../ROADMAP.md).

1. Clone this repository and run commands from the repository root.
2. Editable install (source-only; wheels/sdists are unsupported):

```bash
git clone https://github.com/xArm-Developer/ufactory_genesis.git
cd ufactory_genesis
pip install -e ".[sim]"

# Optional:
#   pip install -e ".[sim,rl]"        # Linux/NVIDIA reach and pick-place
#   pip install -e ".[sim,real]"      # xArm SDK + Pinocchio/Coal for dry-run/sdk-sim/real safety
#   pip install -e ".[sim,dynamics]"  # dynamics reference checks

# Default: PyPI build of torch 2.10.0 (CUDA 12.8). Validated build: 2.10.0+cu128.
# For another build, install the same torch version first (not validated: cu126, cu130, cpu):
#   pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126   # or cu130 / cpu
#   pip install -e ".[sim]"
```

`[sim]` pins `torch==2.10.0`; use a separate virtual environment for packages that require another torch version (for example lerobot).

3. Set a writable Numba cache directory before Genesis-backed runs:

```bash
# Linux / macOS
export NUMBA_CACHE_DIR=~/.cache/numba

# Windows PowerShell
# $env:NUMBA_CACHE_DIR="$env:USERPROFILE\.cache\numba"
```

4. Supported `--robot` keys: `xarm5`, `xarm6`, `xarm7`, `uf850`, `lite6`.

5. **Windows / no supported GPU:** install `torch==2.10.0` from the CPU index (`--index-url https://download.pytorch.org/whl/cpu`) first, then `pip install -e ".[sim]"`. Pass `--backend cpu` on visualization, pick-place, and packaging commands (do not rely on GPU auto-fallback when an unsupported card still enumerates under CUDA). RL requires Linux/NVIDIA GPUs.

6. **Windows, viewer exits with no error:** if `view_robot.py` returns to the prompt with no window after `Building visualizer...`:

```powershell
$env:MKL_THREADING_LAYER = "SEQUENTIAL"
python examples/visualization/view_robot.py --robot xarm6
```

Permanent: `conda env config vars set MKL_THREADING_LAYER=SEQUENTIAL -n <env>`. `--headless` is unaffected.

Public directories in this release:

| Directory | Purpose |
|---|---|
| `visualization/` | GLB robot and tool viewers; see [Visualization](#visualization) |
| `kinematics/` | FK/IK / robot verification wrappers; see [Kinematics & Dynamics Simulation](#kinematics--dynamics-simulation) |
| `simulation/` | Trajectory-controlled task simulations ([pick-place](simulation/pick_place/README.md), [packaging](simulation/packaging/README.md)); see [simulation index](simulation/README.md) |
| `rl/` | Bare-arm Reach → tabletop Pick & Place → box packing (planned); Linux/NVIDIA; see [RL index](rl/README.md) |

See the [RL index](rl/README.md) for the learning sequence and training/inference guides.

## Supported robots and end effectors

| Robot | End effectors | Camera stand compatibility | Sim & preflight | Real packaging |
|---|---|---|---|---|
| `xarm5` | Gripper G2, Bio Gripper G2, Robotiq 2F-85, Robotiq 2F-140, Vacuum Gripper | Bare arm, Gripper G2, Robotiq 2F-85, Robotiq 2F-140, Vacuum Gripper (not Bio Gripper G2) | Yes | — |
| `xarm6` | Gripper G2, Bio Gripper G2, Robotiq 2F-85, Robotiq 2F-140, Vacuum Gripper | Bare arm, Gripper G2, Robotiq 2F-85, Robotiq 2F-140, Vacuum Gripper (not Bio Gripper G2) | Yes | Gripper G2 |
| `xarm7` | Gripper G2, Bio Gripper G2, Robotiq 2F-85, Robotiq 2F-140, Vacuum Gripper | Bare arm, Gripper G2, Robotiq 2F-85, Robotiq 2F-140, Vacuum Gripper (not Bio Gripper G2) | Yes | — |
| `uf850` | Gripper G2, Bio Gripper G2, Robotiq 2F-85, Robotiq 2F-140, Vacuum Gripper | Bare arm, Gripper G2, Robotiq 2F-85, Robotiq 2F-140, Vacuum Gripper (not Bio Gripper G2) | Yes | — |
| `lite6` | Lite6 Gripper, Lite6 Vacuum Gripper | Bare arm, Lite6 Gripper, Lite6 Vacuum Gripper | Yes | Lite6 Gripper |

All robots accept the [camera stand](../assets/urdf/mounts/camera_stand/README.md) (alone or with one depth camera looking along the tool axis), which moves the end effector and its TCP 3 mm along the tool axis:

| Accessory key | Name | Notes |
|---|---|---|
| `camera_stand` | [Camera Stand](../assets/urdf/mounts/camera_stand/README.md) | 3 mm aluminium stand (48 g) |
| `d405` | [Intel RealSense D405](../assets/urdf/cameras/d405/README.md) | Stand + D405 short-range stereo depth camera (58 g) |
| `d435` | [Intel RealSense D435](../assets/urdf/cameras/d435/README.md) | Stand + D435 stereo depth camera (75 g) |
| `gemini_305` | [Orbbec Gemini 305](../assets/urdf/cameras/gemini_305/README.md) | Stand + Gemini 305 stereo depth camera (68 g) |
| `zed_x_nano` | [Stereolabs ZED X Nano](../assets/urdf/cameras/zed_x_nano/README.md) | Stand + ZED X Nano stereo camera (75 g) |

- Configuration URDF references: `assembly:gripper_g2+d435`, `assembly:gripper_g2+ft_sensor_ai1500+d435`, `assembly:lite6_gripper+d435`, `assembly:bare+ft_sensor_ai1500`, `assembly:bare+camera_stand`, `assembly:bare+mounting_plate`, or `assembly:gripper_g2+d435+mounting_plate`.
- Real packaging details: [Hardware & Real Robot](#hardware--real-robot). G2 simulations use `g2_calibrated_v1`; see the [pick-place RL guide](rl/pick_place/README.md) for training and the current v3 checkpoint contract.

`xarm5`, `xarm6`, `xarm7`, and `uf850` also accept the [AI1500 six-axis force/torque sensor](../assets/urdf/accessories/ft_sensor_ai1500/README.md) directly on the tool flange (`+58.6 mm` along flange `+Z`, `0.6 kg`), below any camera stand, adapter, or end effector.

All robots also accept their official aluminium mounting plate (mounted below `link_base`, top face at `z = 0`, keeping kinematics unchanged):

| Mounting plate | Robots | Envelope (mm) | Standalone `--tool` |
|---|---|---|---|
| [xArm Mounting Plate](../assets/urdf/mounts/xarm_mounting_plate/README.md) | `xarm5`, `xarm6`, `xarm7` | `136 × 185 × 8` | `xarm_mounting_plate` |
| [UF850 Mounting Plate](../assets/urdf/mounts/uf850_mounting_plate/README.md) | `uf850` | `190 × 260 × 10` | `uf850_mounting_plate` |
| [Lite 6 Mounting Plate](../assets/urdf/mounts/lite6_mounting_plate/README.md) | `lite6` | `130 × 190 × 5` | `lite6_mounting_plate` |

## Visualization

```bash
# xArm6 bare arm static view
python examples/visualization/view_robot.py --robot xarm6

# xArm6 + Gripper G2 static view (fixed open pose)
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2

# xArm6 + Gripper G2 open/close demo
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2 --gripper-demo

# xArm6 + Robotiq 2F-85 / 2F-140 open/close demo (automatically mounts the 11 mm robotiq_mounting adapter)
python examples/visualization/view_robot.py --robot xarm6 --robotiq-2f-85 --gripper-demo
python examples/visualization/view_robot.py --robot xarm6 --robotiq-2f-140 --camera d435 --gripper-demo

# UF850 + Bio Gripper G2 static view (fixed open pose)
python examples/visualization/view_robot.py --robot uf850 --bio-gripper-g2

# xArm6 + Bio Gripper G2 open/close demo
python examples/visualization/view_robot.py --robot xarm6 --bio-gripper-g2 --gripper-demo

# xArm6 + Vacuum Gripper static view
python examples/visualization/view_robot.py --robot xarm6 --vacuum-gripper

# xArm6 + camera stand + ZED X Nano + Gripper G2 open/close demo
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2 --gripper-demo --camera zed_x_nano

# UF850 + camera stand + D435 + Vacuum Gripper static view
python examples/visualization/view_robot.py --robot uf850 --vacuum-gripper --camera d435

# Lite6 + Lite6 Gripper open/close demo
python examples/visualization/view_robot.py --robot lite6 --lite6-gripper --gripper-demo

# Lite6 + camera stand + D435 + Lite6 Gripper open/close demo
python examples/visualization/view_robot.py --robot lite6 --lite6-gripper --gripper-demo --camera d435

# Lite6 + Lite6 Vacuum Gripper static view
python examples/visualization/view_robot.py --robot lite6 --lite6-vacuum-gripper

# xArm6 + Gripper G2 + D435 + mounting plate
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2 --camera d435 --mounting-plate

# xArm6 + AI1500 F/T sensor + D435 + Gripper G2
python examples/visualization/view_robot.py --robot xarm6 --ft-sensor --gripper-g2 --camera d435 --gripper-demo

# Standalone tool, adapter, F/T sensor, and mounting-plate preview
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

`view_robot.py` flags:

| Flag | Meaning |
|------|---------|
| `--robot` | Required profile key |
| `--gripper-g2` / `--bio-gripper-g2` / `--robotiq-2f-85` / `--robotiq-2f-140` / `--vacuum-gripper` / `--lite6-gripper` / `--lite6-vacuum-gripper` | End-effector combo (at most one) |
| `--camera` | `camera_stand`, `d405`, `d435`, `gemini_305` or `zed_x_nano`: the camera stand, alone or with that camera (all robots; bare arm or any end effector except `--bio-gripper-g2`) |
| `--ft-sensor` | Mount `ft_sensor_ai1500` on the tool flange (`xarm5`, `xarm6`, `xarm7`, `uf850`; optional value defaults to `ft_sensor_ai1500`) |
| `--mounting-plate` | Mount the robot's official aluminium mounting plate below `link_base` |
| `--movable` | Compatibility no-op (grippers always use per-link movable GLBs) |
| `--gripper-demo` | Cycle open/close with a 0.5 s hold after reaching each endpoint (requires a gripper flag) |
| `--pd` | Smooth joint-motion demo (~0.873 rad/s), visual only |
| `--show-tcp` | Red flange TCP marker (hidden by default) |
| `--diagnose` | Headless STL/GLB link-pose diagnostic |
| `--headless` | No viewer window |
| `--backend` | `cpu` / `gpu` (default `gpu`; use `cpu` without a supported GPU) |

`view_tool.py` accepts `--tool {gripper_g2,robotiq_2f_85,robotiq_2f_140,robotiq_mounting,ft_sensor_ai1500,bio_gripper_g2,lite6_gripper,vacuum_gripper,lite6_vacuum_gripper,xarm_mounting_plate,uf850_mounting_plate,lite6_mounting_plate}`, `--headless`, `--backend`, and calibrated gripper tuning flags (`--position-mm`, `--speed-percent`, `--force-percent`, plus `--speed-mm-s` for `bio_gripper_g2`).

## Kinematics & Dynamics Simulation

```bash
# xArm6 offline URDF, FK, and PD control sanity check (no controller)
python examples/kinematics/verify_robot.py --robot xarm6

# xArm6 dynamics simulation check (default poses + 5 random poses)
dynamics-sim-check --robot xarm6 --random-count 5
```

> For FK/IK comparison against a physical controller or hardware dynamics verification, see [Hardware & Real Robot](#hardware--real-robot).

## Pick-place

Multi-robot pick-place in Genesis physics simulation or offline preflight. Flags, more commands, and the configuration overlay are in the [pick-place guide](simulation/pick_place/README.md).

```bash
# xArm6 + Gripper G2: offline trajectory and collision preflight (servo_j)
python examples/simulation/pick_place/run.py \
  --robot xarm6 --mode dry-run --executor servo_j

# Lite6 + Lite6 Gripper: pick-place physics simulation with viewer (servo_cartesian)
python examples/simulation/pick_place/run.py \
  --robot lite6 --mode sim --executor servo_cartesian --visual
```

## Packaging Showcase Simulation

Multi-robot box packing with trajectory control. Flags and more commands are in the [packaging guide](simulation/packaging/README.md).

```bash
# xArm6 + Gripper G2: 1-cycle packaging physics simulation with viewer (servo_j)
python examples/simulation/packaging/run.py \
  --robot xarm6 --mode sim --executor servo_j --visual
```

## Reinforcement Learning

The Linux/NVIDIA simulation examples follow this order:

1. [Reach](rl/reach/README.md): multi-robot (`xarm5`, `xarm6`, `xarm7`, `uf850`, `lite6`) bare-arm reaching, with a 5 mm success threshold.
2. [Pick & Place](rl/pick_place/README.md): xArm6 + Gripper G2 transfers a cube to a tabletop target.
3. Box packing: place an object inside a box; the RL example is planned.

See the [RL index](rl/README.md) for training and inference. The packaging showcase above uses trajectory control.

## Hardware & Real Robot

> [!CAUTION]
> **Safety Requirements**:
> 1. Requires the real hardware extra: `pip install -e ".[real]"` (includes xArm SDK and Pinocchio/Coal collision checks).
> 2. Physical motion carries collision and equipment damage risks. Motion commands require exact per-unit kinematics calibration (`--calibration`) and an explicit confirmation gate (`--confirm-real`). Omitting either rejects before connection.
> 3. Real-robot commands are strictly capped at **1 cycle**.

### 1. Hardware Read-only & Inspection Commands (No Motion)

Connects to a physical controller over the network for validation without commanding joint motion:

```bash
# xArm6: compare Genesis URDF FK against physical controller SDK
python examples/kinematics/verify_fk.py --robot xarm6 --ip <ip>

# Lite6: compare Genesis URDF IK against physical controller SDK
python examples/kinematics/verify_ik.py --robot lite6 --ip <ip>

# xArm6: self-collision check across preset test poses in controller simulation mode (joints do not move)
dynamics-sim-collision-check --robot xarm6 --ip <ip>

# Lite6 + Lite6 Gripper: packaging controller-internal simulation (sdk-sim, joints do not move)
python examples/simulation/packaging/run.py \
  --robot lite6 --mode sdk-sim --executor servo_cartesian \
  --ip <ip> --calibration path/to/exact.yaml

# xArm6 + Bio Gripper G2: pick-place controller-internal simulation (sdk-sim, servo_j or servo_cartesian)
python examples/simulation/pick_place/run.py \
  --robot xarm6 --gripper bio_gripper_g2 --mode sdk-sim --executor servo_j \
  --ip <ip> --calibration path/to/exact.yaml
python examples/simulation/pick_place/run.py \
  --robot xarm6 --gripper bio_gripper_g2 --mode sdk-sim --executor servo_cartesian \
  --ip <ip> --calibration path/to/exact.yaml
```

### 2. Physical Motion Commands (Drives Robot & Gripper)

Sends actual trajectory and servo setpoints to physical hardware:

```bash
# xArm6: move physical robot to validation poses and compare joint torques (requires --confirm-real)
dynamics-hardware-check --robot xarm6 --ip <ip> --confirm-real

# xArm6 + Gripper G2: real pick-place motion (requires per-unit calibration + --confirm-real)
XARM_IP=<ip> python examples/simulation/pick_place/run.py \
  --robot xarm6 --mode real --executor servo_j \
  --calibration path/to/exact.yaml --confirm-real

# xArm6 + Bio Gripper G2: real empty pick-place motion (servo_j or servo_cartesian, optional --visual mirror; requires per-unit calibration + --confirm-real)
XARM_IP=<ip> python examples/simulation/pick_place/run.py \
  --robot xarm6 --gripper bio_gripper_g2 --mode real --executor servo_j --visual \
  --calibration path/to/exact.yaml --confirm-real
XARM_IP=<ip> python examples/simulation/pick_place/run.py \
  --robot xarm6 --gripper bio_gripper_g2 --mode real --executor servo_cartesian \
  --calibration path/to/exact.yaml --confirm-real

# Lite6 + Lite6 Gripper: real packaging motion (enabled for xArm6 + Gripper G2 and Lite6 + Lite6 Gripper; requires per-unit calibration + --confirm-real)
XARM_IP=<ip> python examples/simulation/packaging/run.py \
  --robot lite6 --mode real --executor servo_j \
  --calibration path/to/exact.yaml --confirm-real
```

> Packaging `--mode real` on xArm5, xArm7, and UF850 aborts before connecting until real gripper support is enabled. Real mode never loops.
