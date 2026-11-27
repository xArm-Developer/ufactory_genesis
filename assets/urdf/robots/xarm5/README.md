# xArm5 1305

xArm5 arm description (1305 variant). The xArm5 is an xArm6 whose J4 harmonic joint is replaced by a fixed aluminium key, i.e. an xArm6 with J4 locked at 0. This folder stores no meshes: every visual and collision mesh references the [xArm6](../xarm6/README.md) files in `../xarm6/meshes/xarm6_1305/`. Kinematics, moving-link inertials, and link and joint names are the xArm5's own; `link_base` shares the unified xArm5/6/7 inertial parameters.

## Files

| Path | Content |
|---|---|
| `xarm5_1305.urdf` | Default URDF, STL visuals |
| `xarm5_1305_visual.glb.urdf` | Same physics, GLB visuals |
| `kinematics/user/` | Per-unit calibration `<prefix>_kinematics_<suffix>.yaml` (Git-ignored), read by `ufactory/kinematics/calibration.py` |

No arm + end-effector URDFs are stored: `ufactory.robots.assembly` composes them at runtime from [`assembly.yaml`](../../assembly.yaml), e.g. `compose_robot_urdf("xarm5_1305", "gripper_g2")`.

## Link mapping

| xArm5 link | xArm6 part | Collision hulls |
|---|---|---|
| link_base / link1 / link2 | link of the same name | 2 / 2 / 5 |
| link3 | link3 + link4, link4 placed at J4 = 0 (xyz `0.0775 0.3425 0`, rpy `-1.5708 0 0`) | 5 + 5 |
| link4 | link5 | 2 |
| link5 (flange) | link6 | 1 |

- 22 hulls in total. xArm5 J4/J5 are xArm6 J5/J6.
- `test_xarm5_is_xarm6_with_j4_locked` (`tests/robots/test_asset_integrity.py`) checks that the link poses and mesh placements equal the xArm6's at J4 = 0.
- Moving-link inertials are not shared: xArm5 link3 is 2.121 kg, 1.07 kg lighter than xArm6 link3 + link4 (no J4 module). `link_base` shares the unified xArm5/6/7 inertial parameters (`2.788411 kg`; see [xArm6 shared parts](../xarm6/README.md#shared-parts)).
- Why the links use several hulls: see [xArm6 collision meshes](../xarm6/README.md#collision-meshes).

## Source / License

- URDF derives from the upstream xArm ROS descriptions (see [NOTICE](../../../../NOTICE); typically BSD-3-Clause).
- Meshes: see [xArm6](../xarm6/README.md). Code is MIT; asset attribution is in NOTICE.
