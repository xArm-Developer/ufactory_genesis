# Pick-place

[中文](README_zh.md) · [Simulation index](../README.md) · [Examples guide](../../README.md)

Multi-robot pick-place in Genesis physics simulation or offline trajectory and collision preflight. Install the package and run commands from the repository root as described in [Prerequisites](../../README.md#prerequisites).

`examples/simulation/pick_place/run.py` delegates to the stable `ufactory-pick-place` console command. Both forms accept the same flags.

## Flags

| Flag | Values / notes |
|------|----------------|
| `--robot` | Required: `xarm5` / `xarm6` / `xarm7` / `uf850` / `lite6` |
| `--gripper` | Optional: `default` / `g2` / `bio_gripper_g2` / `lite6` / `robotiq_2f_85` / `robotiq_2f_140` / `vacuum_gripper` (`bio_gripper_g2`, `robotiq_2f_*`, and `vacuum_gripper` support `xarm5`, `xarm6`, `xarm7`, `uf850`; `bio_gripper_g2` runs the microplate task in `sim`, and `dry-run` / `sdk-sim` / `real` with `--executor servo_j` or `servo_cartesian` (`sdk-sim` / `real` validated on `xarm6` only); `vacuum_gripper` supports `dry-run` / `sdk-sim` / `real` with `--executor servo_j` or `servo_cartesian` (optional `--visual` empty-grasp mirror; no suction physics `sim`), hardware-validated on `xarm5` (plug-in hardware version 1); `robotiq_2f_*` runs on a real controller (`sdk-sim` / `real`) only as `robotiq_2f_85` on `xarm6`, validated on hardware with the 11 mm flange adapter; every other Robotiq combination is `sim` / `dry-run` only) |
| `--vacuum-hardware-version` | Optional: `1` (plug-in `TGPIO 0/1`, default) or `2` (contact `TGPIO 3/4`) for `--gripper vacuum_gripper` |
| `--mode` | Required for simulation: `sim` (Genesis physics), `dry-run` (offline trajectory and collision preflight) |
| `--executor` | `servo_j` or `servo_cartesian`; Bio G2 defaults to `servo_cartesian` in `sim`, required for non-`sim` modes and other grippers |
| `--backend` | Optional: `cpu` / `gpu`; overrides `simulation.backend` (use `cpu` without a Genesis-supported GPU) |
| `--visual` | Optional: open the Genesis viewer in `sim` mode, or an async kinematic mirror viewer in `real` mode (Bio G2 runs as empty grasp without carrying the cube) |
| `--config` | Optional strict partial overlay YAML |
| `--print-config` | Print resolved runtime YAML and exit |
| `--report` | Optional report output path |

## Commands

```bash
# xArm6 + Gripper G2: print resolved pick-place runtime YAML and exit
python examples/simulation/pick_place/run.py \
  --robot xarm6 --mode dry-run --executor servo_j --print-config

# xArm6 + Gripper G2: offline trajectory and collision preflight (servo_j)
python examples/simulation/pick_place/run.py \
  --robot xarm6 --mode dry-run --executor servo_j

# Lite6 + Lite6 Gripper: pick-place physics simulation with viewer (servo_cartesian)
python examples/simulation/pick_place/run.py \
  --robot lite6 --mode sim --executor servo_cartesian --visual

# xArm6 + Bio Gripper G2: microplate pick-place physics simulation with viewer (servo_cartesian by default)
python examples/simulation/pick_place/run.py \
  --robot xarm6 --gripper bio_gripper_g2 --mode sim --visual

# xArm6 + Bio Gripper G2: offline preflight (servo_j or servo_cartesian)
python examples/simulation/pick_place/run.py \
  --robot xarm6 --gripper bio_gripper_g2 --mode dry-run --executor servo_j
python examples/simulation/pick_place/run.py \
  --robot xarm6 --gripper bio_gripper_g2 --mode dry-run --executor servo_cartesian

# xArm6 + Robotiq 2F-85 / 2F-140: pick-place physics simulation
python examples/simulation/pick_place/run.py \
  --robot xarm6 --gripper robotiq_2f_85 --mode sim --executor servo_j
python examples/simulation/pick_place/run.py \
  --robot xarm6 --gripper robotiq_2f_140 --mode sim --executor servo_cartesian

# Lite6 + Lite6 Gripper: pick-place physics simulation on CPU backend (without a supported GPU)
python examples/simulation/pick_place/run.py \
  --robot lite6 --mode sim --executor servo_cartesian --backend cpu

# xArm6 + Gripper G2: offline preflight with partial configuration overlay
python examples/simulation/pick_place/run.py \
  --robot xarm6 --mode dry-run --executor servo_j \
  --config examples/simulation/pick_place/runtime.example.yaml
```

## Configuration overlay

`runtime.example.yaml` is a strict partial overlay. Omitted robot, geometry, motion, safety, and simulation values continue to resolve from `assets/configs/runtime`.

## Real robot

For physical robot pick-place, see [Hardware & Real Robot](../../README.md#hardware--real-robot).
