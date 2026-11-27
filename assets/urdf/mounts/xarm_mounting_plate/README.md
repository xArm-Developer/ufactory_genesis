# xArm Mounting Plate

Official UFACTORY 8 mm aluminium mounting plate for **xArm5**, **xArm6**, and **xArm7**. Static part, no joints.

## Parameters

Frame: `xarm_mounting_plate`, origin at the centre of the Ø90 mm central bore on the top mounting plane (`z = 0`). The plate spans `z ∈ [-8, 0] mm`, with its long edge parallel to `Y` and its two-chamfered rear tongue facing `+X`. The runtime composer attaches it below `link_base` at `xyz = (0, 0, 0)` with `mounting_plate_fix`.

- Estimated mass `0.3554 kg` (uniform 6061 aluminium density `2700 kg/m³` integrated over the watertight CAD mesh; fixed-base arm dynamics are unaffected); centre of mass at `(5.15, 0.01, -4.00) mm`.
- Inertia at the centre of mass (kg·m²): `ixx = 1.2285e-3`, `iyy = 6.3206e-4`, `izz = 1.8568e-3`, `ixy = 1.68e-7`, `ixz = -1.53e-9`, `iyz = 6.98e-12`.
- Envelope `136 × 185 × 8 mm`: `x ∈ [-63.0, +73.0] mm`, `y ∈ [-92.5, +92.5] mm`, `z ∈ [-8.0, 0.0] mm`.
- Features: Ø90 mm central through-bore, 12 × Ø4.2 mm mounting holes on PCD 110 mm every 30° (five aligned with xArm `link_base` mounting holes at `xarm_mounting_plate_hole_0..4`), 4 longitudinal slots at `(±83.0, ±37.5)` mm rotated into `(±37.5, ±83.0)` mm in plate XY, and 4 transverse mounting slots.
- Fits: `xarm5_1305`, `xarm6_1305`, `xarm7_1305` (bare arm or any compatible end effector / camera assembly).

## Files

```text
xarm_mounting_plate/
├── xarm_mounting_plate.urdf                          # part URDF: GLB visual + 17 collision prisms + hole frames
└── meshes/
    ├── collision/xarm_mounting_plate_col_00…16.stl   # 17 convex 2D prisms (hole-filled, 0.5 mm top taper)
    └── visual/xarm_mounting_plate.glb                # Draco-compressed GLB, 98,494 triangles, 117 KB
```

## Usage

```python
from ufactory.robots.assembly import compose_robot_urdf

urdf = compose_robot_urdf("xarm6_1305", None, mounting_plate=True)
urdf = compose_robot_urdf("xarm6_1305", "gripper_g2", accessory="d435", mounting_plate=True)
```

Runtime configurations accept `urdf: assembly:bare+mounting_plate` or `urdf: assembly:gripper_g2+d435+mounting_plate`.

```bash
python examples/visualization/view_robot.py --robot xarm6 --mounting-plate
python examples/visualization/view_tool.py --tool xarm_mounting_plate
```

## Source

Derived from the published UFACTORY xArm mounting plate mechanical STEP and validated meshes in `ufactory_mujoco` (`xarm_base_plate`). This repository maintains the Draco GLB, prefixed collision hulls, and URDF packaging.
