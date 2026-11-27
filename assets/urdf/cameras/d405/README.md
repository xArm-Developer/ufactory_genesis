# Intel RealSense D405

Short-range stereo depth camera for the [camera stand](../../mounts/camera_stand/README.md). Static part, no joints.

## Parameters

Frame: `d405_body`, from the realsense-ros mesh; the front glass faces +Z and the rear mounting face is at z = −23 mm.

- Mass 58 g (datasheet); centre of mass at the centre of the collision box.
- Extents 42 × 42 × 23 mm. Collision: one box, 42.09 × 42 × 23 mm.
- Inertia: uniform box.
- Mount: rear 2 × M3 at 20 mm (`d405_hole_{0,1}`), on `stand_hole_20_{0,1}`. `d405_mount` is the rear-face centre with +Z out of the face.
- Optical frame: `d405_optical_frame` (OpenCV: +X right, +Y down, +Z forward) at the front-face centre. It is a geometric estimate, not a calibration.
- On the stand the camera looks along flange +Z with its top towards flange +X. The optical frame sits 67.5 mm off the flange axis and 26 mm above the flange face.

## Files

```
d405/
├── d405.urdf                  # part URDF: GLB visual, box collision, mount/hole/optical frames
├── LICENSE                    # realsense-ros (Apache-2.0)
└── meshes/visual/d405.glb     # Draco-compressed GLB, 3,892 triangles, housing + glass materials, 14 KB
```

## Usage

```python
from ufactory.robots.assembly import compose_robot_urdf

urdf = compose_robot_urdf("xarm6_1305", "gripper_g2", accessory="d405")  # visual="stl" draws the collision box
```

```bash
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2 --movable --camera d405
```

## Source / License

- Mesh: [realsense-ros `d405.stl`](https://github.com/realsenseai/realsense-ros/tree/ros2-master/realsense2_description/meshes) (Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](../../../../NOTICE)).
- Mass and size: Intel RealSense D400 series datasheet.
