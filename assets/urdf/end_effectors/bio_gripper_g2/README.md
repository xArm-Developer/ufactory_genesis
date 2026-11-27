# Bio Gripper G2

UFACTORY Bio Gripper G2 parallel gripper for the `xarm` tool flange (xArm5/6/7, UF850). It also runs standalone.

## Stroke and drive

- Gap between the gripping faces: 71 mm closed to 150 mm open.
- `bio_gripper_g2_right_finger_joint` (prismatic, 0–0.0395 m) drives the gripper; `bio_gripper_g2_left_finger_joint` mimics it (multiplier −1). Joint zero is closed: gap = 71 mm + 2q.
- URDF limits: `effort="40"` (N, twice the 20 N maximum finger force), `velocity="0.333333333333333"` (m/s).
- `ufactory/simulation/bio_g2_model.py` maps SDK units to the model: finger speed 41.67–333.33 mm/s (500–4000 rpm; closing speed 83.33–666.67 mm/s), finger force 4–20 N. `ufactory/grippers/bio_g2.py` holds the joint-space constants.

## Files

```
bio_gripper_g2/
├── bio_gripper_g2_movable_visual.urdf       # part URDF
├── bio_gripper_g2.urdf                      # vendor xacro output
└── meshes/
    ├── collision/                           # convex hulls, in the link frames
    │   ├── bio_gripper_g2_base_link_convex_0.stl
    │   ├── bio_gripper_g2_left_finger_convex_0.stl … _4.stl
    │   └── bio_gripper_g2_right_finger_convex_0.stl … _4.stl
    └── visual/
        ├── visual_glb/bio_gripper_g2_*.glb  # per-link meshes, ~258k triangles in total
        └── bio_gripper_g2_*.stl             # upstream STL visuals, used by bio_gripper_g2.urdf
```

- `bio_gripper_g2_movable_visual.urdf` is the composition source, the standalone preview model (its `world` link raises the base 0.12 m) and the default model of `BioGripperG2Model.load()`. Masses: base with adapter 0.765827 kg, fingers 0.012105 / 0.012068 kg, 0.79 kg in total.
- `bio_gripper_g2.urdf` has the same inertials, joints and hulls, with the STL visuals. Its parent link `link_tool` is not defined in the file, so it is not a Genesis standalone model. Bio G2 replay reports record its SHA-256 (`ufactory/trajectory/bio_g2_replay.py`).
- The GLBs are compressed with the native `draco_encoder`: 0.27 MB in total (test budget < 1 MB). Collision STLs keep full density.
- Collision STLs need unique `bio_gripper_g2_*` basenames. Genesis reads URDF collision meshes through MuJoCo, which keys mesh assets by basename, and the arms also have a `link_base.stl` (`test_bio_gripper_g2_collision_mesh_basenames_are_unique_in_combos`).

## Frames and mount

- The part frame is not at the adapter. In `bio_gripper_g2_base_link` the adapter centre is at (−22.636, −0.956, −2.516) mm with rpy (0.031, 0.254, 0.360)°. Joints, TCP, inertials and hulls are defined in this frame; the mount compensates.
- Mount (`bio_gripper_g2.mount` in [`assembly.yaml`](../../assembly.yaml), the same on every robot of the flange): xyz (22.642, 0.814, 2.516) mm, rpy (0, 0, −0.00628537) rad. It puts the adapter centre on the flange axis and the adapter face at z = 0, with the adapter's two M6 through-holes over the flange's 135°/225° holes (Ø50 mm PCD, < 0.04 mm off).
- Yaw only. The GLB is tilted 0.25° in the part frame, a placement artifact (two flat faces in contact cannot tilt). Compensating it would make TCP-down poses unreachable on xArm5 and near-singular on xArm6. The adapter-face edge therefore shows ±0.16 mm of unevenness.
- In the flange frame: TCP (`bio_gripper_g2_tcp_link`) at (157.6, 0.0, 57.5) mm; grasp point (`GRASP_OFFSET_IN_BASE_LINK`) at (150.6, 0.0, 52.5) mm; visual fingertips 157.9 (right) / 158.5 (left) mm from the flange axis (CAD: 157.5 mm).
- The vendor xarm_ros2 model has its origin at the adapter, but its fingers sit about 12 mm off the CAD (tips at 170.5 mm). This repository follows the CAD.

## Collision hulls

Built from the visual GLBs, in the link frames (every `<collision><origin>` is zero).

- Base: one hull of the GLB vertices above the adapter face, without the M6 screw tips. It fills the ~20 mm adapter overhang behind the housing, up to 13 mm outside the visual. It predates the adapter cleanup; the rounded rim lies up to 0.25 mm outside it.
- Fingers: five hulls each, split in the finger frame (u = |y|, the distance from the gripper mid-plane):

| File | Part | Range (mm) |
|---|---|---|
| `*_finger_convex_0` | pad, inner layer | x ≥ 31.5, u ≤ 40.3; inner face flattened to u = 35.5 |
| `*_finger_convex_1` | pad, outer layer | x ≥ 31.5, u ≥ 40.1, z ≤ 25.9 |
| `*_finger_convex_2` | rod | −3 ≤ x ≤ 31.5 |
| `*_finger_convex_3` | knuckle plate | −25 ≤ x ≤ −3, u ≥ 39.6 |
| `*_finger_convex_4` | knuckle block | −25 ≤ x ≤ −3, z ≤ 6.8 |

- The inner face of `convex_0` is the gripping plane: u = 35.5 mm at joint zero (71 mm gap), normal to the joint axis, covering x 33–76 mm and z 11–29 mm. It must stay the first `<collision>` of each finger, because `collision_gripping_planes()` (`ufactory/simulation/bio_g2_adapter.py`) reads it there. The visual gripping face has a 0.36° yaw residual and lies within 0.3 mm of this plane.
- The slider (x < −25 mm) stays inside the base hull over the full stroke, so it has no hull.
- Hull tips match the visual tips. The hulls fill the cavities of the ribbed pads; elsewhere they lie at most 3.9 mm outside the visual, at the knuckle-block neck inside the housing.
- The per-link GLB fingers carry Y offsets (`GLB_FINGER_VISUAL_Y_OFFSET_M`), so the visual and collision gripping planes align over 71–150 mm.

## Adapter plate

Adapter frame: contact face at z = 0, +Z toward the housing; the plate is 5 mm thick. In `bio_gripper_g2_base.glb` and `bio_gripper_g2_visual.glb`:

- The two M6 countersunk screws in the 135°/225° holes are removed; the real part has through-holes there.
- The curved faces (rim, centre bore, the four M6 holes, the lugs with their holes and screws, the stepped pin hole at −90°) lie on their analytic shapes within 0.015 mm, with analytic normals.
- The contact face and the plate top end at round hole rims. All other vertices are unchanged from the export.
- Checks: `test_bio_gripper_g2_adapter_circles_are_round`, `test_bio_gripper_g2_adapter_has_through_holes_not_screws`, `test_bio_gripper_g2_adapter_faces_stop_at_hole_rims`. History: [CHANGELOG](../../../../CHANGELOG.md).

## Usage

```python
from ufactory.grippers.bio_g2 import BioGripperG2
from ufactory.robots.assembly import compose_robot_urdf
from ufactory.robots.paths import bio_gripper_g2_movable_visual_urdf

urdf = bio_gripper_g2_movable_visual_urdf()                # standalone
urdf = compose_robot_urdf("xarm6_1305", "bio_gripper_g2")  # on an arm

# After scene.build(); robot is the Genesis entity loaded from a Bio G2 URDF.
gripper = BioGripperG2(robot)
if gripper.found:
    gripper.setup_pd()                            # calibrated servo (BioG2GenesisController); starts open
    for _ in range(100):
        gripper.control_pose(gripper.CLOSE_POS)   # close to 71 mm; OPEN_POS opens to 150 mm
        scene.step()
```

```bash
python examples/visualization/view_tool.py --tool bio_gripper_g2    # standalone, as mounted: adapter up, TCP down
python examples/visualization/view_robot.py --robot xarm6 --bio-gripper-g2 --gripper-demo
```

## Source / License

- The vendor URDF and the meshes derive from the upstream xArm ROS / xarm_ros2 family (see [NOTICE](../../../../NOTICE)).
- This repository maintains the part URDF, the mount, the collision hulls and the Draco-compressed GLBs.
