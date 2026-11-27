# Lite6 Gripper

Lite6 parallel gripper for the `lite6` tool flange.

## Files

```
lite6_gripper/
├── lite6_gripper_movable_visual.urdf       # part URDF: composition source and standalone preview
└── meshes/
    ├── collision/
    │   ├── shell.stl                       # shell collision
    │   └── finger1.stl / finger2.stl       # finger visuals (finger collision uses boxes)
    └── visual/                             # Draco-compressed GLBs, 0.09 MB in total (test budget < 1 MB)
        └── visual_glb/shell.glb            # shell visual
```

## Model

- Joints: `finger_joint1` (prismatic along +Y, 0–8.9 mm) drives the gripper; `finger_joint2` mimics it along −Y. Both joint frames are the vendor frame at z = 54.3 mm; `link_tcp` is at z = 83.6 mm.
- Finger collision: two boxes per finger form an L (inner pad + outer root plate) and leave the notch empty. Genesis convexifies each geom separately, so the notch is never filled in, and a grasped object gets no false contact on its top face.
- Gap: the large inner faces of `finger1.stl` / `finger2.stl` are 22.4 mm apart when closed and 40.2 mm when open. The real gripper is installed reversed and has a 20–38 mm gap; `ufactory/grippers/lite6.py` models this as a fixed 20 mm offset plus the 0–8.9 mm per-finger travel.
- The `world` link raises the base 0.12 m for standalone preview; composition drops it.

## Usage

```python
from ufactory.robots.assembly import compose_robot_urdf

urdf = compose_robot_urdf("lite6", "lite6_gripper")  # per-link visuals, movable
```

The runtime configuration uses `assembly:lite6_gripper:stl` (STL visuals, same collision).

```bash
python examples/visualization/view_tool.py --tool lite6_gripper    # standalone (flange up, fingertips down)
python examples/visualization/view_robot.py --robot lite6 --lite6-gripper --gripper-demo
```

## Source / License

- Meshes derive from the upstream xArm ROS / xarm_ros2 family (see [NOTICE](../../../../NOTICE)).
- This repository maintains the part URDF and the Draco-compressed GLBs.
