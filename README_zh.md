# ufactory_genesis

![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Version](https://img.shields.io/badge/version-0.3.6-orange) ![Genesis](https://img.shields.io/badge/genesis-1.4.5-lightgrey) ![CI](https://github.com/xArm-Developer/ufactory_genesis/actions/workflows/ci.yml/badge.svg)

基于 Genesis 的 UFACTORY 机械臂仿真与强化学习示例。

[English](README.md) | [仿真](#仿真) | [强化学习](#强化学习)

## 仿真

[![装箱仿真演示](https://i0.hdslb.com/bfs/archive/d23d45c6454ae51a396a056312fd101f2b7b8380.jpg@672w_378h_1c.webp)](https://www.bilibili.com/video/BV1NqWc6gEWB/)

支持 xArm5、xArm6、xArm7、UF850 和 Lite6。[机械臂与夹爪组合](examples/README_zh.md#支持机型与末端)见示例指南。

- [机械臂与夹爪可视化](examples/README_zh.md#可视化)：GLB 模型、可动手指与夹爪开合演示。
- [运动学与接触仿真](examples/README_zh.md#运动学与动力学仿真)：FK/IK 校验与物理仿真检查。
- [轨迹控制抓放](examples/pick_place/README_zh.md)：抓取、搬运并放置物体。
- [装箱仿真](examples/packaging/README_zh.md)：通过 YAML 配置轨迹，模拟装箱与接触过程。

安装方式与完整运行指令见[示例指南](examples/README_zh.md)。

## 强化学习

[![Reach 评估演示](https://i.ytimg.com/vi/FRZopwFl-aM/maxresdefault.jpg)](https://www.bilibili.com/video/BV12rpk69ENc/)

在 Linux/NVIDIA GPU 上进行仿真训练与推理。学习顺序及各任务指南见 [RL 索引](examples/rl/README_zh.md)。

1. **[Reach：目标到达](examples/rl/reach/README_zh.md)**：让未安装夹爪的机械臂（`xarm5`、`xarm6`、`xarm7`、`uf850`、`lite6`）末端到达目标点，成功距离严格小于 **5 mm**。支持单臂、四臂 GLB 实时推理，提供 [Hugging Face 预训练检查点](https://huggingface.co/UFACTORY-Developer/ufactory_genesis_rl)。
2. **[Pick & Place：桌面抓放](examples/rl/pick_place/README_zh.md)**：xArm6 + Gripper G2 抓取在 `x ∈ [300, 500] mm`、`y ∈ [-200, 200] mm` 内均匀随机采样的方块并搬运到桌面固定目标位置 `(300, 300) mm`，以纯 PPO 从零训练策略。



## 文档

- [真机操作与硬件校验](examples/README_zh.md#真机操作与硬件校验)
- [动力学校验](ufactory/dynamics/README.md)
- [贡献与项目检查](CONTRIBUTING.md)
- [当前入口](ROADMAP.md#current-entry-points) · [路线图](ROADMAP.md) · [变更记录](CHANGELOG.md)

真机运动需要逐台标定、预测安全预检和显式确认，详见[安全说明](SECURITY.md)。

## 许可与引用

代码采用 MIT 许可；机器人 URDF 和网格资产包含上游材料，其独立归属说明见 [NOTICE](NOTICE)。

研究使用时请引用：“genesis-ufactory: UFACTORY Robot Models for Genesis Simulation”，UFACTORY，2026，`https://github.com/xArm-Developer/ufactory_genesis`。
