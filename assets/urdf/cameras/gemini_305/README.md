# Orbbec Gemini 305

Stereo depth camera for the [camera stand](../../mounts/camera_stand/README.md). Static part, no joints.

## Parameters

Frame: `gemini_305_body`, the CAD frame. The camera faces −Y; the rear mounting face is at y = 23.08 mm.

- Mass 68 g (vendor spec); centre of mass at the centre of the collision box.
- Extents 42 × 23 × 42 mm. Collision: one box, 42.03 × 23 × 42.03 mm.
- Inertia: uniform box.
- Mount: rear 2 × M3 at 20 mm (`gemini_305_hole_{0,1}`), on `stand_hole_20_{0,1}`.
- Optical frame: `gemini_305_optical_frame` (OpenCV: +X right, +Y down, +Z forward) at the front-face centre. It is a geometric estimate, not a calibration.
- On the stand the camera looks along flange +Z with its top towards flange +X. The optical frame sits 67.5 mm off the flange axis and 26 mm above the flange face.

## Files

```
gemini_305/
├── gemini_305.urdf                 # part URDF: GLB visual, box collision, mount/hole/optical frames
└── meshes/visual/gemini_305.glb    # Draco-compressed GLB, 192,620 triangles, housing + glass materials, 0.33 MB
```

## Usage

```python
from ufactory.robots.assembly import compose_robot_urdf

urdf = compose_robot_urdf("xarm6_1305", "gripper_g2", accessory="gemini_305")
```

```bash
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2 --movable --camera gemini_305
```

## Source

- Mesh: the three body solids of the [Orbbec Gemini 305 CAD](https://doc.orbbec.com/attachments/18) (`3D__00_mb2502_asm(G305)_ V1.1.stp`).
- Mass and size: [Orbbec store page](https://store.orbbec.com/products/gemini-305) and the G305 2D drawing V1.1.
