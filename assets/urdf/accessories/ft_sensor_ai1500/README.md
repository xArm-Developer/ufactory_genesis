# UFACTORY 6-Axis Force/Torque Sensor (AI1500)

UFACTORY AI1500 six-axis force/torque sensor for the **xArm5 / xArm6 / xArm7 / UF850** tool flange (`flange: xarm`, ISO 9409-1-50-4-M6; **not compatible with Lite 6**).

## Mechanical & Inertial Parameters

Frame: `ft_sensor_ai1500`, origin at the arm tool flange centre on the mating plane (`z = 0`, `+Z` pointing outward along the tool axis).

- **Tool lift (`ft_sensor_ai1500_output`)**: `+58.6 mm` along flange `+Z` (matching the default UFACTORY Studio TCP Z offset `(0, 0, 58.6 mm, 0, 0, 0)`).
- **Mass**: `0.6 kg` (official specification including flange mounting hardware).
- **Centre of mass**: `(0, 0, 22) mm` in the flange frame.
- **Inertia at CoM** (uniform cylinder approximation `R = 38.8 mm`, `H = 63.6 mm`, matching the `ufactory_mujoco` AI1500 sandbox):
  `ixx = 0.000428064`, `iyy = 0.000428064`, `izz = 0.000451632 kg·m²`, `ixy = ixz = iyz = 0`.
- **Outer envelope**: `Ø77.6 mm` (`r_max = 38.8 mm`), `z ∈ [-5.0, +58.6] mm` (bottom locating spigot reaches `-5.0 mm`; mounting skirt covers `z ∈ [-3.0, 0.0] mm` around the arm flange).
- **Reference frames**:
  - `ft_sensor_ai1500_flange_mount`: `(0, 0, 0)` mm
  - `ft_sensor_ai1500_output`: `(0, 0, 58.6)` mm (downstream parts attach here)
  - `ft_sensor_ai1500_measurement`: `(0, 0, 58.6)` mm (nominal six-axis sensing frame)

## Stacking Order

Because the AI1500 bottom skirt (`r ∈ [31.5, 38.8] mm`, `z ∈ [-3.0, 0.0] mm`) wraps around the arm flange, the camera stand cannot sit below the sensor. Downstream parts always stack onto `ft_sensor_ai1500_output` in this order:

`arm flange -> ft_sensor_ai1500 (+58.6 mm) -> [camera_stand (+3.0 mm) + camera] -> [robotiq_mounting (+11.0 mm)] -> end_effector`

> **Real-robot note**: When mounting a camera stand/camera downstream of the AI1500, its mass is supported by the sensor and included during `iden_ft_sensor_load_offset`; secure the camera cable to minimize tugging disturbances.

## Sensor Specifications (Reference)

- **Sampling rate**: `200 Hz`
- **Rated range**: `±400 N` (`Fx/Fy/Fz`), `±20 N·m` (`Tx/Ty/Tz`)
- **Overload limit**: `±600 N`, `±30 N·m`
- **Resolution**: `0.4 N` (force), `0.01 N·m` (torque)

> **Hardware verification status (`xArm6`, controller `v2.8.2`, AI1500 `SN=AI15000626001F`, `v1.1.1`)**: Bare-sensor communication, `enable`/`disable`, SN/firmware readback, stationary zeroing (`±0.05 N` / `±0.001 N·m` noise), ±45° tilt gravity sign convention (external force applied to distal flange in tool frame, `+Z` outward; bare distal cap mass ~154 g), admittance hold (`ft_mode=1`), and constant-force closed-loop sign/gain (`ft_mode=2`, `coord=TOOL`, `f_ref=-2 N` moves along tool `-Z` at `2.0 mm/s` for `kp=0.005`) have been verified via `scripts/ft_sensor_check.py`. Mounted-tool payload identification (`identify-load`, which requires a non-trivial external payload downstream of the sensor) and dynamic inertia calibration remain pending.

## Files

```
ft_sensor_ai1500/
├── ft_sensor_ai1500.urdf                  # Part URDF (GLB visual + 1 convex collision hull + reference frames)
└── meshes/
    ├── collision/ft_sensor_ai1500_col_00.stl   # Watertight convex hull (48 vertices, 92 faces)
    └── visual/
        ├── ft_sensor_ai1500.glb           # Multi-material Draco-compressed GLB
        └── ft_sensor_ai1500.stl           # Merged meter-scale visual STL (visual="stl")
```

## Source

Exported via [dev/accessories/ai1500/export_ft_sensor_ai1500.py](../../../dev/accessories/ai1500/export_ft_sensor_ai1500.py) from the frozen source snapshots (`model.glb`, `collision.json`, derived from `6 Axis Force Torque Sensor(AI1500)_20260413.STEP`, SHA-256 verified) copied from `ufactory_mujoco`.
