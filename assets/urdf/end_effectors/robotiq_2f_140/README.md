# Robotiq 2F-140 Adaptive Gripper URDF & Meshes

This directory holds the standalone movable URDF, collision STL meshes, and visual GLB/STL assets for the **Robotiq 2F-140** adaptive gripper.

## Source & License

- **Upstream:** [Robotiq/robotiq-ros](https://github.com/Robotiq/robotiq-ros) (`49167c4f44a1da2b564715fba3e729c8ce958b63`)
- **License:** BSD-3-Clause (see [`LICENSE`](LICENSE))

## Files

- `robotiq_2f_140_movable_visual.urdf` — standalone movable Genesis URDF with Draco-compressed visual GLBs (`meshes/visual/visual_glb/*.glb`) and split/convex collision STLs (`meshes/collision/*.stl`).
- `meshes/visual/robotiq_2f_140_base.stl` — base visual STL used when building combo STL-collision URDFs via `assets/urdf/assembly.yaml`.
- `meshes/visual/visual_glb/` — Draco-compressed per-link visual GLBs (`base.glb`, `{left,right}_{outer_knuckle,finger,inner_knuckle}.glb`).
- `meshes/collision/` — CoACD base convex hull (`robotiq_2f_140_base_col_00.stl`), split finger pad & body hulls, and linkage collision meshes.

## Kinematics & Frames

- Master actuated joint: `robotiq_2f_140_drive_joint` (`0.003859` rad open at 140 mm gap to `0.766594` rad closed at 0 mm gap).
- Five coupled mimic joints (`multiplier=1`, `offset=0`) drive the four-bar parallel finger linkage symmetrically.
- `robotiq_2f_140_tcp_link` sits at `(0, 0, 0.234466)` m relative to `robotiq_2f_140_base_link` (base mounting face at `z = 0`).
- When composed onto `xarm5`/`xarm6`/`xarm7`/`uf850` via `assets/urdf/assembly.yaml`, the 11 mm [`robotiq_mounting`](../../mounts/robotiq_mounting/README.md) flange adapter is inserted automatically (not supported on `lite6`), placing `robotiq_2f_140_tcp_link` at `245.466 mm` from the arm flange (`248.466 mm` with the 3 mm `camera_stand`).
