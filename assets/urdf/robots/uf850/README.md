# UF850

UF850 arm description.

## Files

| Path | Content |
|---|---|
| `uf850.urdf` | Default URDF, STL visuals |
| `uf850_visual.glb.urdf` | Same physics, GLB visuals |
| `meshes/uf850/visual/*.stl` | STL visuals (7) |
| `meshes/uf850/visual_glb/*.glb` | Draco-compressed GLB visuals (7), 0.74 MB in total (test budget < 1 MB) |
| `meshes/uf850/collision/uf850_link*_col_*.stl` | Convex collision hulls (27) |
| `kinematics/user/` | Per-unit calibration `<prefix>_kinematics_<suffix>.yaml` (Git-ignored), read by `ufactory/kinematics/calibration.py` |

The UF850 has the same `xarm` tool flange as xArm5/6/7, so it takes the same end effectors: [`gripper_g2`](../../end_effectors/gripper_g2/README.md), [`bio_gripper_g2`](../../end_effectors/bio_gripper_g2/README.md) and [`vacuum_gripper`](../../end_effectors/vacuum_gripper/README.md). No arm + end-effector URDFs are stored: `ufactory.robots.assembly` composes them at runtime from [`assembly.yaml`](../../assembly.yaml), e.g. `compose_robot_urdf("uf850", "gripper_g2")`.

## Collision meshes

| Link | link_base | link1 | link2 | link3 | link4 | link5 | link6 |
|---|---|---|---|---|---|---|---|
| Hulls | 3 | 4 | 7 | 5 | 5 | 2 | 1 |

Why the links use several hulls: see [xArm6 collision meshes](../xarm6/README.md#collision-meshes).

## Base inertial

`uf850.urdf` and `uf850_visual.glb.urdf` share the `link_base` inertial parameters evaluated in SolidWorks on the `Arm850_P01` base assembly (`5.869631 kg`, CoM `(-0.762, -0.044, 117.610) mm`, standard URDF negative off-diagonal product-of-inertia convention; total bare-arm mass `19.844631 kg`).

## Source / License

- URDF and STL derive from the upstream UFACTORY / xArm ROS descriptions (see [NOTICE](../../../../NOTICE)).
- This repository maintains the GLB visuals, the collision hulls and the Draco compression. Code is MIT; asset attribution is in NOTICE.
