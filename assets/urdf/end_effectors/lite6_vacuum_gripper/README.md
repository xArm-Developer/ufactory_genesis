# Lite6 Vacuum Gripper

Lite6 vacuum gripper for the `lite6` tool flange. Static part, no joints.

## Files

```
lite6_vacuum_gripper/
├── lite6_vacuum_gripper.urdf                  # part URDF (vendor xacro output): 0.14 kg, link_tcp at z = 61 mm
└── meshes/visual/
    ├── vacuum_gripper_lite.stl                # STL visual and collision
    └── lite6_vacuum_gripper_visual_link6.glb  # Draco-compressed GLB visual, 0.16 MB (test budget < 1 MB)
```

## Usage

`compose_robot_urdf("lite6", "lite6_vacuum_gripper")` replaces the STL visual with the GLB (`glb_visuals` in [`assembly.yaml`](../../assembly.yaml)); `visual="stl"` keeps the STL. The collision mesh is always `vacuum_gripper_lite.stl`.

```bash
python examples/visualization/view_tool.py --tool lite6_vacuum_gripper    # standalone (GLB, flange up)
python examples/visualization/view_robot.py --robot lite6 --lite6-vacuum-gripper
```

## Source / License

- URDF and STL come from upstream [xarm_ros2](https://github.com/xArm-Developer/xarm_ros2) (see [NOTICE](../../../../NOTICE)).
- This repository maintains the Draco-compressed GLB.
