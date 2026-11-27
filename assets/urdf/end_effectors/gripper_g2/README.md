# Gripper G2

UFACTORY Gripper G2 parallel gripper for the `xarm` tool flange (xArm5/6/7, UF850).

## Files

```
gripper_g2/
├── gripper_g2_movable_visual.urdf    # part URDF: composition source and standalone preview
└── meshes/
    ├── collision/
    │   ├── base_link_col_00.stl … base_link_col_09.stl   # base: 10 convex hulls
    │   ├── left_finger.stl / right_finger.stl             # fingers and knuckles: full-density STLs
    │   ├── left_inner_knuckle.stl / right_inner_knuckle.stl
    │   └── left_outer_knuckle.stl / right_outer_knuckle.stl
    └── visual/
        ├── base_link.stl             # full base mesh, the base's visual="stl" mesh
        └── visual_glb/               # per-link GLBs: base, fingers, knuckles
```

The GLBs are Draco-compressed, 0.21 MB in total (test budget < 1 MB). With `visual="stl"` the base shows `visual/base_link.stl` (`stl_visuals` in [`assembly.yaml`](../../assembly.yaml)); the fingers and knuckles show their collision STLs.

## Model

- Frame: all meshes are in the flange frame of the official SolidWorks model `Gripper G2 AG1200 20250710.STEP`, with the flange mounting face at Z = 0 and the flange axis through the origin. Every `<visual>` and `<collision>` origin is `0 0 0`.
- Reference heights: locating boss bottom −5.0 mm, base top 104.5 mm, outer / inner knuckle pivots 60.6 / 75.602 mm, `joint_tcp` 173.5 mm.
- Materials of `visual_glb/base.glb`, one primitive each: housing white; flange, centring boss and locating pin in the xArm / UF850 flange material (anodised aluminium: base colour 0.953, metallic 0.8, roughness 0.64); the +Y connector cover in the fingers' black (the contact-type cover for XX1305 and UF850 arm ends); the bearing cover between the fingers and its four M3×8 screws in matte black anodised aluminium (base colour 0.12, metallic 0.8, roughness 0.62). With `visual="stl"` the base is one `White` mesh. `visual_glb/{left,right}_finger.glb` split into 2 primitives: black finger body (base colour 0.077, metallic 0, roughness 0.55) and white translucent silicone inner pad (base colour 0.92/0.92/0.90, alpha 0.75 BLEND, metallic 0, roughness 0.30); with `visual="stl"`, fingers and knuckles still show their collision STLs (`Silver`).
- Base below the flange face: the locating boss and the Ø6 slotted round locating pin at (0, −25) mm. In both visual meshes (`visual_glb/base.glb` and `visual/base_link.stl`), the pin is an analytic watertight mesh with a `r = 2.98 mm` cylindrical body (`z = -2.50..2.00 mm`), a `2.00 mm` dome fillet to `z = -4.50 mm`, and a `1.00 mm × 2.00 mm` slot at the STEP yaw (`-7.04°`). Only two of the 10 base hulls reach there: `base_link_col_00` (boss) and `base_link_col_01` (pin). The hulls are byte-identical to ufactory_mujoco's.
- Joints: `drive_joint` (revolute, 0.0572–0.8651 rad) drives the gripper; the other five finger and knuckle joints mimic it (multiplier 1).
- The `world` link raises the base 0.12 m for standalone preview. Composition drops it and attaches `xarm_gripper_base_link` to the flange with the fixed joint `gripper_fix`. `examples/visualization/view_tool.py --tool gripper_g2` turns the base 180° about X (flange up, fingertips down) at `base_pos=(0, 0, 0.35)` to match the mounted home pose.

## Usage

```python
from ufactory.robots.assembly import compose_robot_urdf

urdf = compose_robot_urdf("xarm6_1305", "gripper_g2")               # per-link GLBs
urdf = compose_robot_urdf("xarm6_1305", "gripper_g2", visual="stl") # STL visuals
```

Composed URDFs are written to `.cache/ufactory/urdf/composed/`. Runtime configurations reference this assembly as `urdf: "assembly:gripper_g2"`.

```bash
python examples/visualization/view_tool.py --tool gripper_g2    # standalone (flange up, fingertips down)
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2 --gripper-demo
```

## Source / License

- Meshes derive from the upstream xArm ROS / xarm_ros2 family (see [NOTICE](../../../../NOTICE)).
- This repository maintains the part URDF, the flange-frame meshes and the Draco-compressed GLBs.
