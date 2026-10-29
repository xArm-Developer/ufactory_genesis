# ufactory_genesis

![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Version](https://img.shields.io/badge/version-0.3.6-orange) ![Genesis](https://img.shields.io/badge/genesis-1.4.5-lightgrey) ![CI](https://github.com/xArm-Developer/ufactory_genesis/actions/workflows/ci.yml/badge.svg)

UFACTORY robot simulation and reinforcement learning examples built on Genesis.

[中文](README_zh.md) | [Simulation](#simulation) | [Reinforcement learning](#reinforcement-learning)

## Simulation

[![Packaging simulation demo](https://i.ytimg.com/vi/Dib244y-zUY/maxresdefault.jpg)](https://youtu.be/Dib244y-zUY)

Supports xArm5, xArm6, xArm7, UF850 and Lite6. See the [robot and gripper combinations](examples/README.md#supported-robots-and-end-effectors).

- [Robot and gripper visualization](examples/README.md#visualization): GLB models, movable fingers and gripper opening/closing demos.
- [Kinematics and contact simulation](examples/README.md#kinematics--dynamics-simulation): FK/IK verification and physics checks.
- [Trajectory-controlled pick & place](examples/pick_place/README.md): pick up, transfer and place objects.
- [Box packing simulation](examples/packaging/README.md): YAML-configured trajectories and contact simulation.

Installation and complete commands are in the [examples guide](examples/README.md).

## Reinforcement learning

[![Reach evaluation demo](https://i.ytimg.com/vi/FRZopwFl-aM/maxresdefault.jpg)](https://youtu.be/FRZopwFl-aM)

Simulation-only training and inference on Linux/NVIDIA GPUs. See the [RL index](examples/rl/README.md) for the task progression and guides.

1. **[Reach: target reaching](examples/rl/reach/README.md)** — Move the end effector of a bare-arm robot (`xarm5`, `xarm6`, `xarm7`, `uf850`, `lite6`) to a target with a success distance strictly below **5 mm**. Supports single-arm and four-arm real-time GLB inference, with [pretrained checkpoints on Hugging Face](https://huggingface.co/UFACTORY-Developer/ufactory_genesis_rl).
2. **[Pick & Place: tabletop](examples/rl/pick_place/README.md)** — xArm6 + Gripper G2 grasps and transfers a cube sampled uniformly in `x ∈ [300, 500] mm`, `y ∈ [-200, 200] mm` to a fixed target at `(300, 300) mm`, trained from scratch with pure PPO.



## Documentation

- [Hardware and real-robot operation](examples/README.md#hardware--real-robot)
- [Dynamics checks](ufactory/dynamics/README.md)
- [Contributing and project checks](CONTRIBUTING.md)
- [Current entry points](ROADMAP.md#current-entry-points) · [Roadmap](ROADMAP.md) · [Changelog](CHANGELOG.md)

Real motion requires per-unit calibration, predictive preflight and explicit confirmation. See the [safety requirements](SECURITY.md).

## License and citation

Code is MIT licensed; robot URDF and mesh assets include upstream material with separate attribution in [NOTICE](NOTICE).

For research use, cite “genesis-ufactory: UFACTORY Robot Models for Genesis Simulation,” UFACTORY, 2026, `https://github.com/xArm-Developer/ufactory_genesis`.
