# Camera Stand

UFACTORY 3 mm aluminium camera stand. It is clamped between the ISO 9409-1-50-4-M6 tool flange (xArm5/6/7, UF850, Lite 6) and the end effector, and carries one camera: [D405](../../cameras/d405/README.md), [D435](../../cameras/d435/README.md), [Gemini 305](../../cameras/gemini_305/README.md) or [ZED X Nano](../../cameras/zed_x_nano/README.md). Static part, no joints.

## Parameters

Frame: `camera_stand`, origin at the flange centre on the flange face. The plate spans y = 0…3 mm, so the plate normal is +Y. The composer mounts it with `rpy = (π/2, 0, 0)`, which turns +Y onto flange +Z.

- Mass 48 g (weighed); centre of mass at (29.49, 1.49, 0.40) mm.
- Inertia at the centre of mass (kg·m², uniform density over the visual mesh): `ixx = 1.8923e-5`, `iyy = 7.7626e-5`, `izz = 5.8775e-5`, `ixy = -1.73e-8`, `ixz = 5.61e-7`, `iyz = 3.2e-9`.
- Extents 126 × 3 × 75 mm. Flange end: Ø34.5 mm centre bore, Ø6.10 mm locating-pin hole, 4 × M6 at PCD 50 mm.
- Camera end, around `stand_camera_mount` at (67.5, 3, 0) mm with +Z out of the camera face:
  - `stand_hole_20_{0,1}`: 2 × M3 at 20 mm (D405, Gemini 305).
  - `stand_hole_45_{0,1}`: 2 × M3 at 45 mm (D435).
  - `stand_hole_20x20_{0..3}`: 20 × 20 mm square of 4 × M3 at x = 57.5/77.5, z = ±10 mm (ZED X Nano).
  - A 20 × 23.2 mm centre opening; the ZED X Nano GMSL2 connector passes through it.
- The stand adds 3 mm between the flange and the end effector (or flange adapter). The end effector mount, and with it the TCP, moves 3 mm along flange +Z: Gripper G2 TCP goes from 173.5 to 176.5 mm, vacuum gripper TCP from 126 to 129 mm, Robotiq 2F-85 (via 11 mm `robotiq_mounting` adapter) from 173.253 to 176.253 mm, Robotiq 2F-140 from 245.466 to 248.466 mm, Lite6 Gripper TCP from 83.6 to 86.6 mm, and Lite6 Vacuum Gripper TCP from 61 to 64 mm. The bare arm's `link_eef` (when present) moves the same 3 mm. **On the real robot, set the controller TCP offset to match.**
- Fits: bare arm, `gripper_g2`, `vacuum_gripper`, `robotiq_2f_85`, `robotiq_2f_140`, `lite6_gripper`, `lite6_vacuum_gripper`. Not `bio_gripper_g2` (its body reaches x = 59.22 mm and hits the D405/D435).

## Files

```
camera_stand/
├── camera_stand.urdf                 # part URDF: GLB visual + 45 collision prisms + hole frames; composition source
└── meshes/
    ├── collision/piece_000…044.stl   # 45 convex 2D prisms (3 mm), 888 faces
    └── visual/
        ├── camera_stand.glb          # Draco-compressed GLB, 7,428 triangles, 1 material, 16 KB
        └── camera_stand.stl          # same mesh in metres; visual="stl"
```

The collision prisms are slices of the plate outline, merged greedily while keeping the bore, the pin hole, the pogo cut-out, the camera-end opening and a positive gap to the Gripper G2 base hulls (+0.014 mm). Generic convex decomposition chords across the round holes and overlapped the G2 base by 0.35–1.45 mm.

## Usage

```python
from ufactory.robots.assembly import compose_robot_urdf

urdf = compose_robot_urdf("xarm6_1305", "gripper_g2", accessory="camera_stand")  # stand only
urdf = compose_robot_urdf("xarm6_1305", "gripper_g2", accessory="d435")          # stand + D435
urdf = compose_robot_urdf("lite6", "lite6_gripper", accessory="d435")            # Lite 6 + stand + D435
urdf = compose_robot_urdf("uf850", None, accessory="zed_x_nano")                 # bare arm + stand + ZED X Nano
```

Runtime configurations accept the same as `urdf: assembly:gripper_g2+d435`, `urdf: assembly:lite6_gripper+d435`, or `urdf: assembly:bare+camera_stand`.

```bash
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2 --gripper-demo --camera d435
python examples/visualization/view_robot.py --robot lite6 --lite6-gripper --gripper-demo --camera d435
```

## Source

UFACTORY stand CAD `405_435_zedX.STEP`. This repository maintains the GLB, the STL and the collision prisms.
