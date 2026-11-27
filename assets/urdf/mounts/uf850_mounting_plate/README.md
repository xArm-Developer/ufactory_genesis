# UF850 Mounting Plate

Official UFACTORY 10 mm aluminium mounting plate for **UF850**. Static part, no joints.

## Parameters

Frame: `uf850_mounting_plate`, origin at the centre of the Ø130 mm central bore on the top mounting plane (`z = 0`). The plate spans `z ∈ [-10, 0] mm`, with its long edge parallel to `Y`. The runtime composer attaches it below `link_base` at `xyz = (0, 0, 0)` with `mounting_plate_fix`.

- Measured mass `1.056 kg` (uniformly scaled over the watertight CAD mesh volume `338.578 cm³`, effective density `3118.92 kg/m³`; fixed-base arm dynamics are unaffected); centre of mass at `(0.00, 0.00, -5.00) mm`.
- Inertia at the centre of mass (kg·m²): `ixx = 7.4944e-3`, `iyy = 3.7860e-3`, `izz = 1.1263e-2`, `ixy = 2.83e-9`, `ixz = 8.91e-11`, `iyz = -2.16e-11`.
- Envelope `190 × 260 × 10 mm`: `x ∈ [-95.0, +95.0] mm`, `y ∈ [-130.0, +130.0] mm`, `z ∈ [-10.0, 0.0] mm` (corner radius 20 mm).
- Features: Ø130 mm central through-bore, 4 × Ø6.8 mm arm mounting holes at `(±60.104, ±60.104) mm` (`uf850_mounting_plate_hole_0..3`), and 8 outer mounting slots (`(±70, ±115)` mm and `(±80, ±85)` mm).
- Fits: `uf850` (bare arm or any compatible end effector / camera assembly).

## Files

```text
uf850_mounting_plate/
├── uf850_mounting_plate.urdf                         # part URDF: GLB visual + 1 convex collision prism + hole frames
└── meshes/
    ├── collision/uf850_mounting_plate_col_00.stl     # 1 convex 2D prism (hole-filled)
    └── visual/uf850_mounting_plate.glb               # Draco-compressed GLB, 89,916 triangles, 102 KB
```

## Usage

```python
from ufactory.robots.assembly import compose_robot_urdf

urdf = compose_robot_urdf("uf850", None, mounting_plate=True)
urdf = compose_robot_urdf("uf850", "gripper_g2", accessory="d435", mounting_plate=True)
```

Runtime configurations accept `urdf: assembly:bare+mounting_plate` or `urdf: assembly:gripper_g2+d435+mounting_plate`.

```bash
python examples/visualization/view_robot.py --robot uf850 --mounting-plate
python examples/visualization/view_tool.py --tool uf850_mounting_plate
```

## Source

Derived from the published UFACTORY UF850 mounting plate mechanical STEP and validated meshes in `ufactory_mujoco` (`uf850_base_plate`). This repository maintains the Draco GLB, prefixed collision hull, and URDF packaging.
