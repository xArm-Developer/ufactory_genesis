# Stereolabs ZED X Nano

Stereo camera for the [camera stand](../../mounts/camera_stand/README.md). Static part, no joints.

## Parameters

Frame: `zed_x_nano_body`, the CAD frame. The lenses face −Y, +Z is the top, and the base is at z = 0. The rear mounting face is at y = 20.57 mm.

- Mass 75 g (vendor spec); centre of mass at the volume-weighted centre of the two collision boxes.
- Extents 42 × 44.54 × 30.5 mm, including the rear GMSL2 connector.
- Collision: two boxes, the body (42 × 33.81 × 30.5 mm) and the connector (7.4 × 10.73 × 10 mm).
- Inertia: two uniform boxes.
- Mount: rear 20 × 20 mm square of 4 × M3 (`zed_x_nano_hole_{0..3}`, centre `zed_x_nano_mount` at (0, 20.57, 15.263) mm), on `stand_hole_20x20_{0..3}`. The GMSL2 connector and the two locating bosses pass through the stand's centre opening. The closest is 0.17 mm from the opening edge.
- Optical frame: `zed_x_nano_optical_frame` (OpenCV: +X right, +Y down, +Z forward) between the lenses, on the front face. It is a geometric estimate, not a calibration.
- On the stand the camera looks along flange +Z (down when the TCP points down) with its top towards flange +X. The optical frame sits 67.5 mm off the flange axis and 36.81 mm above the flange face.
- With Gripper G2: ≥ 14.7 mm to the G2 base, ≥ 45 mm to the fingers over the full stroke, ≥ 12 mm from the connector to link 6.

## Files

```
zed_x_nano/
├── zed_x_nano.urdf                 # part URDF: GLB visual, 2 box collisions, mount/hole/optical frames
└── meshes/visual/zed_x_nano.glb    # Draco-compressed GLB, 272,418 triangles, housing + glass materials, 0.47 MB
```

## Usage

```python
from ufactory.robots.assembly import compose_robot_urdf

urdf = compose_robot_urdf("xarm6_1305", "gripper_g2", accessory="zed_x_nano")
```

```bash
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2 --movable --gripper-demo --camera zed_x_nano
```

## Source

- Mesh: [Stereolabs ZED X Nano STEP](https://www.stereolabs.com/3dmodels), body and rear connector; the two field-of-view solids are left out.
- Mass and size: Stereolabs ZED X Nano datasheet.
