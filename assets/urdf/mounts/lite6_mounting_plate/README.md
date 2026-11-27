# Lite 6 Mounting Plate

Official UFACTORY 5 mm aluminium mounting plate for **Lite 6**. Static part, no joints.

## Parameters

Frame: `lite6_mounting_plate`, origin at the centre of the Ø90 mm central bore on the top mounting plane (`z = 0`). The plate spans `z ∈ [-5, 0] mm`, with its long edge parallel to `Y`. Because Lite 6's four base mounting holes are centred at `(-3, 0) mm` in `link_base`, the runtime composer attaches `lite6_mounting_plate` below `link_base` at `xyz = (-0.003, 0, 0)` with `mounting_plate_fix`.

- Estimated mass `0.2431 kg` (uniform 6061 aluminium density `2700 kg/m³` integrated over the watertight CAD mesh; fixed-base arm dynamics are unaffected); centre of mass at `(0.00, 0.00, -2.50) mm`.
- Inertia at the centre of mass (kg·m²): `ixx = 9.2852e-4`, `iyy = 4.1146e-4`, `izz = 1.3390e-3`, `ixy = -3.47e-12`, `ixz = -1.05e-12`, `iyz = 3.22e-12`.
- Envelope `130 × 190 × 5 mm`: `x ∈ [-65.0, +65.0] mm`, `y ∈ [-95.0, +95.0] mm`, `z ∈ [-5.0, 0.0] mm` (corner radius 15 mm).
- Features: Ø90 mm central through-bore, 4 × Ø4.2 mm arm mounting holes at `(±57.0, ±57.0) mm` (`lite6_mounting_plate_hole_0..3`), and 4 outer Ø5.2 mm mounting holes at `(±50.0, ±80.0) mm`.
- Fits: `lite6` (bare arm or any compatible end effector / camera assembly).

## Files

```text
lite6_mounting_plate/
├── lite6_mounting_plate.urdf                         # part URDF: GLB visual + 1 convex collision prism + hole frames
└── meshes/
    ├── collision/lite6_mounting_plate_col_00.stl     # 1 convex 2D prism (hole-filled)
    └── visual/lite6_mounting_plate.glb               # Draco-compressed GLB, 34,806 triangles, 42 KB
```

## Usage

```python
from ufactory.robots.assembly import compose_robot_urdf

urdf = compose_robot_urdf("lite6", None, mounting_plate=True)
urdf = compose_robot_urdf("lite6", "lite6_gripper", accessory="d435", mounting_plate=True)
```

Runtime configurations accept `urdf: assembly:bare+mounting_plate` or `urdf: assembly:lite6_gripper+d435+mounting_plate`.

```bash
python examples/visualization/view_robot.py --robot lite6 --mounting-plate
python examples/visualization/view_tool.py --tool lite6_mounting_plate
```

## Source

Derived from the published UFACTORY Lite 6 mounting plate mechanical STEP and validated meshes in `ufactory_mujoco` (`lite6_base_plate`). This repository maintains the Draco GLB, prefixed collision hull, and URDF packaging.
