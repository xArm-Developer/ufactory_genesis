# Robotiq Flange Adapter (`robotiq_mounting`)

UFACTORY 11 mm silver anodized aluminium coupling adapter for mounting **Robotiq 2F-85** and **Robotiq 2F-140** grippers onto the UFACTORY ISO 9409-1-50-4-M6 tool flange (**xArm5**, **xArm6**, **xArm7**, and **UF850**; not compatible with **Lite 6**). Static part, no joints.

## Parameters

Frame: `robotiq_mounting`, origin at the flange centre on the xArm/UF850 mating plane (`z = 0`), with `+Z` pointing toward the Robotiq gripper.

- Measured mass `0.118 kg` (uniformly scaled over the exact B-Rep volume `40.638 cm³`, effective density `2903.72 kg/m³`); centre of mass at `(0.00, -0.80, 4.46) mm`.
- Inertia at the centre of mass (kg·m²): `ixx = 4.4271e-5`, `iyy = 4.9980e-5`, `izz = 9.0184e-5`, `ixy = 0`, `ixz = 0`, `iyz = 7.2081e-8`.
- Envelope: outer diameter `Ø75.0 mm` (`x, y ∈ [-37.5, +37.5] mm`), axial extent `z ∈ [-5.6, +13.9] mm`.
  - **xArm / UF850 side (`z = 0`)**: `Ø31.5 × 5.6 mm` centring boss (`z ∈ [-5.6, 0.0] mm`, fits inside the 6 mm arm flange recess and passes through the 3 mm `camera_stand` `Ø34.5 mm` centre bore), 4 × M6 counterbored bolt holes on PCD 50 mm at `(±17.68, ±17.68) mm` (`robotiq_mounting_xarm_hole_0..3`), `Ø6.0 mm` locating-pin hole at `(0, -25.0) mm` (`robotiq_mounting_xarm_pin`), and tool Pogo-Pin pass-through slot at `(0, +23.0) mm`.
  - **Robotiq side (`z = +11.0 mm`)**: flat seating plane at `z = +11.0 mm` (`robotiq_mounting_tool_mount`) surrounded by a `Ø71.0 mm` inner / `Ø75.0 mm` outer centring rim (`z ∈ [+11.0, +13.9] mm`), 4 × M4 clearance through-holes at `(±16.0, ±27.0) mm` (`robotiq_mounting_robotiq_hole_0..3`), and a `Ø4.0 mm` blind locating-pin hole at `(0, -30.5) mm` (`robotiq_mounting_robotiq_pin`).
- Axial stack height: adds **11.0 mm** along flange `+Z` between the arm flange (or `camera_stand`) and the Robotiq base. Assembled Robotiq TCPs from the arm flange are `173.253 mm` (2F-85) and `245.466 mm` (2F-140), or `176.253 mm` / `248.466 mm` when stacked with the 3 mm `camera_stand`. **On the real robot, include this 11 mm offset (and 3 mm if `camera_stand` is installed) in the controller TCP offset and payload.**
- Fits: `xarm5_1305`, `xarm6_1305`, `xarm7_1305`, `uf850` with `robotiq_2f_85` or `robotiq_2f_140` (bare or with `camera_stand` + any supported depth camera). Automatically inserted by `ufactory.robots.assembly`.

## Files

```text
robotiq_mounting/
├── robotiq_mounting.urdf                       # part URDF: GLB visual + 2 convex collision hulls + hole/mount frames
└── meshes/
    ├── collision/
    │   ├── robotiq_mounting_col_00.stl         # convex hull of Ø75 mm main disc (z ∈ [0, 13.9] mm)
    │   └── robotiq_mounting_col_01.stl         # convex hull of Ø31.5 mm centring boss (z ∈ [-5.6, 0] mm)
    └── visual/
        ├── robotiq_mounting.glb                # Draco-compressed GLB, arm end-flange metal PBR (FLANGE_METAL)
        └── robotiq_mounting.stl                # watertight visual STL in metres (for visual="stl")
```

## Usage

```python
from ufactory.robots.assembly import compose_robot_urdf

urdf = compose_robot_urdf("xarm6_1305", "robotiq_2f_85")                  # arm + adapter + 2F-85
urdf = compose_robot_urdf("uf850", "robotiq_2f_140", accessory="d435")    # arm + stand + D435 + adapter + 2F-140
```

```bash
python examples/visualization/view_tool.py --tool robotiq_mounting
python examples/visualization/view_robot.py --robot xarm6 --robotiq-2f-85 --gripper-demo
python examples/visualization/view_robot.py --robot uf850 --robotiq-2f-140 --camera d435 --gripper-demo
```

## Source

Derived from UFACTORY mechanical CAD `robotiq_mounting.STEP` (`robotiq coupling 转接件-V02.STEP`). This repository maintains the Draco GLB, visual STL, convex collision hulls, and URDF packaging.
