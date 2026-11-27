# xArm7 1305

xArm7 arm description (1305 variant).

## Files

| Path | Content |
|---|---|
| `xarm7_1305.urdf` | Default URDF, STL visuals |
| `xarm7_1305_visual.glb.urdf` | Same physics, GLB visuals |
| `meshes/xarm7_1305/visual/*.stl` | STL visuals of link1–link6 |
| `meshes/xarm7_1305/visual_glb/*.glb` | Draco-compressed GLB visuals of link1–link6, 0.64 MB in total (test budget < 1 MB) |
| `meshes/xarm7_1305/collision/xarm7_link*_col_*.stl` | Convex collision hulls of link1–link6 (23) |
| `kinematics/user/` | Per-unit calibration `<prefix>_kinematics_<suffix>.yaml` (Git-ignored), read by `ufactory/kinematics/calibration.py` |

`link_base` and the flange link `link7` use xArm6 meshes (see [Shared parts](#shared-parts)). No arm + end-effector URDFs are stored: `ufactory.robots.assembly` composes them at runtime from [`assembly.yaml`](../../assembly.yaml), e.g. `compose_robot_urdf("xarm7_1305", "gripper_g2")`.

## Collision meshes

| Link | link_base | link1 | link2 | link3 | link4 | link5 | link6 | link7 |
|---|---|---|---|---|---|---|---|---|
| Hulls | 2 | 2 | 5 | 4 | 5 | 5 | 2 | 1 |

26 hulls in total; the 3 of `link_base` and `link7` are xArm6 files. Why the links use several hulls: see [xArm6 collision meshes](../xarm6/README.md#collision-meshes).

## Shared parts

`link_base` and `link7` use the visual and collision meshes of xArm6 `link_base` and `link6` (see [xArm6 shared parts](../xarm6/README.md#shared-parts)): the production base (version A, also sharing the unified `2.788411 kg` / CoM `(-2.587, -0.269, 103.569) mm` `link_base` inertial parameters), and the flange link that xArm5/6/7 share.

## Source / License

- URDF and STL derive from the upstream xArm ROS descriptions (see [NOTICE](../../../../NOTICE); typically BSD-3-Clause).
- This repository maintains the GLB visuals, the collision hulls and the Draco compression. Code is MIT; asset attribution is in NOTICE.
