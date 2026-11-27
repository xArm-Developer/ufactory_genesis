# Lite6

Lite6 arm description.

## Files

| Path | Content |
|---|---|
| `lite6.urdf` | Default URDF, STL visuals |
| `lite6_visual.glb.urdf` | GLB visuals. Same physics, except that the `link_base` inertia tensor is zero (fixed base, no effect on dynamics) |
| `meshes/lite6/visual/*.stl` | STL visuals (7) |
| `meshes/lite6/visual_glb/*.glb` | Draco-compressed GLB visuals (7), 0.75 MB in total (test budget < 1 MB); `link_base.glb` sits flush at `z = 0` matching `link_base.stl` |
| `meshes/lite6/collision/lite6_link*_col_*.stl` | Convex collision hulls (19) |
| `kinematics/user/` | Per-unit calibration `<prefix>_kinematics_<suffix>.yaml` (Git-ignored), read by `ufactory/kinematics/calibration.py` |

End effectors for the `lite6` flange: [`lite6_gripper`](../../end_effectors/lite6_gripper/README.md) and [`lite6_vacuum_gripper`](../../end_effectors/lite6_vacuum_gripper/README.md). No arm + end-effector URDFs are stored: `ufactory.robots.assembly` composes them at runtime from [`assembly.yaml`](../../assembly.yaml), e.g. `compose_robot_urdf("lite6", "lite6_gripper")`. The runtime configuration uses `assembly:lite6_gripper:stl` (STL visuals).

## Collision meshes

| Link | link_base | link1 | link2 | link3 | link4 | link5 | link6 |
|---|---|---|---|---|---|---|---|
| Hulls | 3 | 2 | 4 | 2 | 4 | 1 | 3 |

`link6` is split into three exact convex hulls from `visual/link6.stl` (flange body `col_00`, recessed neck `col_01`, and side connector boss `col_02`) so the convex envelope does not bridge across the concave neck/connector steps and falsely reduce the `link4`/`link6` wrist clearance below 5 mm under per-unit calibration.

Why the other links use several hulls: see [xArm6 collision meshes](../xarm6/README.md#collision-meshes).

## Source / License

- URDF and STL derive from the upstream xArm ROS / xarm_ros2 family (see [NOTICE](../../../../NOTICE)).
- This repository maintains the GLB visuals, the collision hulls and the Draco compression. Code is MIT; asset attribution is in NOTICE.
