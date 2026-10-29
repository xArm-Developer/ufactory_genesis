# Gripper G2 Assets

标准 xArm/UF850 平行夹爪。

## 模板 URDF

`gripper_g2.urdf` 是从 xarm_ros2 的 xacro 生成的 standalone 模板文件；v0.1.6 中 mesh 路径已改为本仓库内的 `meshes/collision/*.stl`，可用于基础结构解析。它保留 vendor 的外部父 link，不作为 Genesis standalone 查看入口。

Genesis standalone 查看使用 `gripper_g2_movable_visual.urdf`（`python examples/visualization/view_gripper_g2.py`）。实际机械臂搭配优先使用 `robot_visual_glb_urdf(..., with_gripper_g2=True)` 返回的组合 URDF。

## 目录结构

```
gripper_g2/
├── gripper_g2.urdf                   # standalone 模板 URDF
├── gripper_g2_movable_visual.urdf    # 仅夹爪可动视觉 URDF（调试用）
├── meshes/
│   ├── collision/                    # STL 碰撞网格（保持原密度，不做抽面）
│   │   ├── base_link.stl
│   │   ├── left_finger.stl / right_finger.stl
│   │   ├── left_inner_knuckle.stl / right_inner_knuckle.stl
│   │   └── left_outer_knuckle.stl / right_outer_knuckle.stl
│   └── visual/                       # GLB 视觉网格（法兰中心 canonical + 原生 Draco）
│       ├── gripper_g2_static.glb     # 合并静态高模（全机型共用）
│       └── visual_glb/               # 可动分体 GLB（base + fingers/knuckles）
```

视觉面数保持不变，用原生 `draco_encoder` 压 **视觉 GLB**（合计目标 **< 1 MiB**）。所有视觉 GLB 与碰撞 STL 均以官方 SolidWorks 模型（`Gripper G2 AG1200 20250710.STEP`）法兰安装面 `Z = 0`、法兰回转轴 `(0, 0, 0)` 为基准（定位凸台底面 `Z = -5.0 mm`，底座顶面 `Z = 104.5 mm`，外指节销孔 `Z = 0.0606 m`，内指节销孔 `Z = 0.075602 m`，`joint_tcp` `Z = 0.1735 m`）；全机型组合 URDF 的 `<visual><origin>` 与 `<collision><origin>` 均为 `0 0 0`。

## 加载方式

机械臂场景使用组合 URDF：

```python
from ufactory.robots.paths import robot_visual_glb_urdf

# 静态 GLB
urdf = robot_visual_glb_urdf("xarm6_1305", with_gripper_g2=True)

# 可动 GLB
urdf = robot_visual_glb_urdf("xarm6_1305", with_gripper_g2=True, movable=True)
```

## Source / License

- 模板与网格源自上游 xArm ROS / xarm_ros2 家族（见仓库根 [NOTICE](../../../NOTICE)）。
- 本仓库维护 standalone / 组合 URDF、共享 `visual_glb/`（法兰安装面基准）与 Draco 压缩 GLB。
