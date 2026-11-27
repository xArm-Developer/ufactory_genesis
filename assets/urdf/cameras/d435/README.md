# Intel RealSense D435

Stereo depth camera for the [camera stand](../../mounts/camera_stand/README.md). Static part, no joints.

## Parameters

Frame: `d435_body`, from the realsense-ros mesh; the front glass faces +Z and the nominal rear face is at z = −25 mm.

- Mass 75 g (datasheet); centre of mass at the centre of the collision box.
- Extents 90 × 25 × 25 mm. Collision: one box, 90 × 25 × 25 mm.
- Inertia: uniform box.
- Mount: rear 2 × M3 at 45 mm (`d435_hole_{0,1}`), on `stand_hole_45_{0,1}`. The ROS mesh pokes 0.0547 mm past the nominal rear face, so the camera sits 0.055 mm off the stand.
- Optical frame: `d435_optical_frame` (OpenCV: +X right, +Y down, +Z forward) at the front-face centre. It is a geometric estimate, not a calibration.
- On the stand the camera looks along flange +Z with its top towards flange +X. The optical frame sits 67.5 mm off the flange axis and 28.06 mm above the flange face. The USB-C port is on the flange −Y side.

## Files

```
d435/
├── d435.urdf                  # part URDF: GLB visual, box collision, mount/hole/optical frames
├── LICENSE                    # realsense-ros (Apache-2.0)
└── meshes/visual/d435.glb     # Draco-compressed GLB, 231,127 triangles, housing + glass materials, 0.50 MB
```

## Usage

```python
from ufactory.robots.assembly import compose_robot_urdf

urdf = compose_robot_urdf("xarm6_1305", "gripper_g2", accessory="d435")  # visual="stl" draws the collision box
```

```bash
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2 --movable --camera d435
```

## Source / License

- Mesh: [realsense-ros `d435.dae`](https://github.com/realsenseai/realsense-ros/tree/ros2-master/realsense2_description/meshes) (Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](../../../../NOTICE)).
- Mass and size: Intel RealSense D400 series datasheet.
