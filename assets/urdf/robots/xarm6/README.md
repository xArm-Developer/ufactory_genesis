# xArm6 1305

xArm6 arm description. 1305 is the only xArm6 variant in this tree.

## Files

| Path | Content |
|---|---|
| `xarm6_1305.urdf` | Default URDF, STL visuals |
| `xarm6_1305_visual.glb.urdf` | Same physics, GLB visuals |
| `meshes/xarm6_1305/visual/*.stl` | STL visuals (7) |
| `meshes/xarm6_1305/visual_glb/*.glb` | Draco-compressed GLB visuals (7), 1.04 MB in total (test budget < 1.5 MB) |
| `meshes/xarm6_1305/collision/xarm6_link*_col_*.stl` | Convex collision hulls (22) |
| `kinematics/user/` | Per-unit calibration `<prefix>_kinematics_<suffix>.yaml` (Git-ignored), read by `ufactory/kinematics/calibration.py` |

No arm + end-effector URDFs are stored: `ufactory.robots.assembly` composes them at runtime from [`assembly.yaml`](../../assembly.yaml), e.g. `compose_robot_urdf("xarm6_1305", "gripper_g2")`.

## Collision meshes

| Link | link_base | link1 | link2 | link3 | link4 | link5 | link6 |
|---|---|---|---|---|---|---|---|
| Hulls | 2 | 2 | 5 | 5 | 5 | 2 | 1 |

By default Genesis turns each collision mesh of a robot into one convex hull (`convexify=True` for rigid entities, `decompose_robot_error_threshold=inf`; checked in Genesis 1.4.7). One hull fills the concave links: for link2–link4 it lies 30–40 mm (P95) outside the visual surface, and with J5 near 165–180° (e.g. `[90, -90, -60, 0, 175, -90]°`) it gives a false link4–link6 self-collision and torque saturation. The links are therefore pre-split into watertight convex hulls that stay within 5–7 mm (P95) of the visual surface, so Genesis keeps its pure-convex collision path.

## Shared parts

Other arms reference these meshes directly, so editing them changes those arms too. The mapping is `_ARM_PART_SOURCES` in `tests/robots/test_asset_integrity.py`.

- [xArm5](../xarm5/README.md): every link. The xArm5 is an xArm6 with J4 locked at 0.
- [xArm7](../xarm7/README.md): `link_base`, and `link6` as the xArm7 flange link `link7`.

The base is the production version A: flat top at z = 150.35 mm, body Ø95.4 mm. Its STL matches upstream `xarm6_1305`; its GLB comes from the CAD export `1305-xarm5-P01-95-3`. The prototype version B (body Ø98.0 mm, top rising to 152.55 mm) is not used. All three arms also share the unified `link_base` inertial parameters evaluated in SolidWorks on the production base assembly `Xarm+_P01` (`2.788411 kg`, CoM `(-2.587, -0.269, 103.569) mm`, standard URDF negative off-diagonal product-of-inertia convention).

## Source / License

- URDF and STL derive from the upstream xArm ROS descriptions (see [NOTICE](../../../../NOTICE); typically BSD-3-Clause).
- This repository maintains the GLB visuals, the collision hulls and the Draco compression. Code is MIT; asset attribution is in NOTICE.
