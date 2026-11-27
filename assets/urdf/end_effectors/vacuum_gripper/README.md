# Vacuum Gripper (xArm / UF850)

Five-cup vacuum gripper for the `xarm` tool flange (xArm5/6/7, UF850). Static part, no joints.

## Parameters

Frame: `xarm_vacuum_gripper_link`, origin at the flange mounting face, axes aligned with the flange.

- Mass 0.61 kg; centre of mass at (0, 0, 53) mm.
- Inertia at the centre of mass (kg·m²): `ixx = 0.00047106`, `iyy = 0.00033072`, `izz = 0.00025642`, `ixy = 3.9292e-7`, `ixz = 2.6537e-6`, `iyz = -1.0975e-5`.
- TCP: `vacuum_tcp` at (0, 0, 126) mm through `vacuum_joint_tcp`, oriented like the flange.
- Maximum suction payload: 5.0 kg, excluding the gripper's own mass (vendor spec, not modelled).

## Files

```
vacuum_gripper/
├── vacuum_gripper.urdf                # part URDF: GLB visual + 33 collision hulls; composition source
└── meshes/
    ├── collision/convex/              # 33 watertight convex hulls, 11,358 faces
    │   └── hull_000_flange.stl … hull_032_cup_05.stl
    └── visual/
        ├── vacuum_gripper.glb         # Draco-compressed GLB: 209,604 triangles, 5 materials, 0.55 MB (test budget < 1 MB)
        └── vacuum_gripper_visual.stl  # upstream visual STL, 20,544 faces; used by visual="stl"
```

The hulls are cut from the upstream visual STL by part: flange (1), body (6), cover (1) and the five cup columns (5 each). Cutting by part keeps the gaps between the cups. `test_vacuum_gripper_static_urdfs_and_parameters` checks the parameters and the hull list.

## Usage

```python
from ufactory.robots.assembly import compose_robot_urdf
from ufactory.robots.paths import vacuum_gripper_urdf

urdf = vacuum_gripper_urdf()                               # standalone part
urdf = compose_robot_urdf("xarm6_1305", "vacuum_gripper")  # on an arm; visual="stl" uses the STL visual
```

```bash
python examples/visualization/view_tool.py --tool vacuum_gripper    # standalone (flange up, cups down)
python examples/visualization/view_robot.py --robot xarm6 --vacuum-gripper
python examples/visualization/view_robot.py --robot uf850 --vacuum-gripper
```

## Source / License

- URDF parameters and STL come from upstream [xArm-Developer/xarm_ros2](https://github.com/xArm-Developer/xarm_ros2) (BSD-3-Clause; see [NOTICE](../../../../NOTICE)).
- This repository maintains the Draco-compressed GLB and the 33 collision hulls.
