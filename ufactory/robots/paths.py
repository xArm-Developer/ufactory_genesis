"""Project path helpers for UFACTORY URDF and configuration assets."""

from pathlib import Path

from ufactory.robots.assembly import (
    compose_robot_urdf,
    compose_standalone_adapter_urdf,
    compose_standalone_end_effector_urdf,
    compose_standalone_ft_sensor_urdf,
    compose_standalone_mounting_plate_urdf,
    parse_assembly_ref,
)
from ufactory.robots.registry import PROJECT_ROOT, URDF_ROOT, get_robot_profile

DYNAMICS_VALIDATION_POSES_YAML = PROJECT_ROOT / "assets" / "configs" / "dynamics_validation_pose.yaml"


def _existing_path(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return str(path.resolve())


def _urdf_path(assets: Path, name: str) -> str:
    return _existing_path(assets / name)


def xarm5_urdf(name: str = "xarm5_1305.urdf") -> str:
    """Return absolute path to an xArm 5 URDF file in project assets."""
    return _urdf_path(get_robot_profile("xarm5_1305").assets_dir, name)


def xarm6_urdf(name: str = "xarm6_1305.urdf") -> str:
    """Return absolute path to an xArm 6 URDF file in project assets."""
    return _urdf_path(get_robot_profile("xarm6_1305").assets_dir, name)


def xarm7_urdf(name: str = "xarm7_1305.urdf") -> str:
    """Return absolute path to an xArm 7 URDF file in project assets."""
    return _urdf_path(get_robot_profile("xarm7_1305").assets_dir, name)


def xarm6_1305_urdf() -> str:
    """Return absolute path to the xArm6 XI1305 simulation URDF."""
    return xarm6_urdf()


def robot_assets(robot_name: str) -> Path:
    return get_robot_profile(robot_name).assets_dir


def kinematics_user_dir(robot_name: str) -> Path:
    return get_robot_profile(robot_name).assets_dir / "kinematics" / "user"


def robot_urdf(robot_key: str, name: str | None = None) -> str:
    """Return the absolute path of a robot URDF by profile key or robot name.

    ``name`` is a file in the arm asset directory (default: the dynamics URDF) or an assembly reference such as
    ``assembly:gripper_g2``, ``assembly:gripper_g2+d435``, or ``assembly:bare+mounting_plate`` (see
    :mod:`ufactory.robots.assembly`), which is composed on demand.
    """
    profile = get_robot_profile(robot_key)
    urdf_name = name or profile.default_urdf
    ref = parse_assembly_ref(urdf_name)
    if ref is not None:
        return compose_robot_urdf(
            profile.key,
            ref.end_effector,
            visual=ref.visual,
            accessory=ref.accessory,
            mounting_plate=ref.mounting_plate,
            ft_sensor=ref.ft_sensor,
        )
    return _urdf_path(profile.assets_dir, urdf_name)


_ACCESSORY_FLAGS = (
    ("with_bio_gripper_g2", "bio_gripper_g2", "Bio Gripper G2"),
    ("with_gripper_g2", "gripper_g2", "Gripper G2"),
    ("with_robotiq_2f_85", "robotiq_2f_85", "Robotiq 2F-85"),
    ("with_robotiq_2f_140", "robotiq_2f_140", "Robotiq 2F-140"),
    ("with_vacuum_gripper", "vacuum_gripper", "Vacuum Gripper"),
    ("with_lite6_gripper", "lite6_gripper", "Lite6 Gripper"),
    ("with_lite6_vacuum_gripper", "lite6_vacuum_gripper", "Lite6 Vacuum Gripper"),
)


def robot_visual_glb_urdf(
    robot_key: str,
    *,
    with_bio_gripper_g2: bool = False,
    with_gripper_g2: bool = False,
    with_robotiq_2f_85: bool = False,
    with_robotiq_2f_140: bool = False,
    with_vacuum_gripper: bool = False,
    with_lite6_gripper: bool = False,
    with_lite6_vacuum_gripper: bool = False,
    movable: bool = False,
    accessory: str | None = None,
    mounting_plate: bool = False,
    ft_sensor: str | None = None,
) -> str:
    """GLB-visual URDF of an arm, optionally composed with an F/T sensor, end effector, accessory, and/or mounting plate.

    Without parts this is the arm's ``*_visual.glb.urdf``. With any, the arm and parts are assembled at runtime using
    per-link visuals; ``movable`` is kept for compatibility (grippers always use per-link movable visuals).
    ``accessory`` is ``camera_stand`` or a camera on it (``d405``, ``d435``, ``gemini_305``, ``zed_x_nano``). New code
    should call :func:`ufactory.robots.assembly.compose_robot_urdf` directly.
    """
    profile = get_robot_profile(robot_key)
    flags = {
        "with_bio_gripper_g2": with_bio_gripper_g2,
        "with_gripper_g2": with_gripper_g2,
        "with_robotiq_2f_85": with_robotiq_2f_85,
        "with_robotiq_2f_140": with_robotiq_2f_140,
        "with_vacuum_gripper": with_vacuum_gripper,
        "with_lite6_gripper": with_lite6_gripper,
        "with_lite6_vacuum_gripper": with_lite6_vacuum_gripper,
    }
    selected = [(name, label) for flag, name, label in _ACCESSORY_FLAGS if flags[flag]]
    if len(selected) > 1:
        raise ValueError(
            "with_bio_gripper_g2, with_gripper_g2, with_robotiq_2f_85, with_robotiq_2f_140, "
            "with_vacuum_gripper, with_lite6_gripper, and with_lite6_vacuum_gripper are mutually exclusive"
        )
    if movable and not (
        with_gripper_g2 or with_robotiq_2f_85 or with_robotiq_2f_140 or with_lite6_gripper or with_bio_gripper_g2
    ):
        raise ValueError(
            "movable=True requires with_gripper_g2=True, with_robotiq_2f_85=True, "
            "with_robotiq_2f_140=True, with_lite6_gripper=True, or with_bio_gripper_g2=True"
        )
    if ft_sensor is not None and not profile.supports_ft_sensor(ft_sensor):
        raise ValueError(f"Robot {robot_key} does not support F/T sensor {ft_sensor}")
    if not selected:
        if accessory is None and not mounting_plate and ft_sensor is None:
            return _urdf_path(profile.assets_dir, profile.visual_glb_urdf)
        if accessory is not None and not profile.supports_accessory(accessory):
            raise ValueError(f"Robot {robot_key} does not support {accessory}")
        return compose_robot_urdf(
            profile.key,
            None,
            visual="glb",
            accessory=accessory,
            mounting_plate=mounting_plate,
            ft_sensor=ft_sensor,
        )
    end_effector, label = selected[0]
    if not profile.supports(end_effector):
        raise ValueError(f"Robot {robot_key} does not support {label}")
    if accessory is not None and not profile.supports_accessory(accessory, end_effector):
        raise ValueError(f"{accessory} does not fit {label} on {robot_key}")
    return compose_robot_urdf(
        profile.key,
        end_effector,
        visual="glb",
        accessory=accessory,
        mounting_plate=mounting_plate,
        ft_sensor=ft_sensor,
    )


def lite6_gripper_movable_visual_urdf() -> str:
    """Standalone Lite6 gripper-only movable visual URDF (no arm)."""
    return _existing_path(URDF_ROOT / "end_effectors" / "lite6_gripper" / "lite6_gripper_movable_visual.urdf")


def gripper_g2_movable_visual_urdf() -> str:
    """Standalone gripper-only movable visual URDF (no arm)."""
    return _existing_path(URDF_ROOT / "end_effectors" / "gripper_g2" / "gripper_g2_movable_visual.urdf")


def robotiq_2f_85_movable_visual_urdf() -> str:
    """Standalone Robotiq 2F-85 gripper-only movable visual URDF (no arm)."""
    return _existing_path(URDF_ROOT / "end_effectors" / "robotiq_2f_85" / "robotiq_2f_85_movable_visual.urdf")


def robotiq_2f_140_movable_visual_urdf() -> str:
    """Standalone Robotiq 2F-140 gripper-only movable visual URDF (no arm)."""
    return _existing_path(URDF_ROOT / "end_effectors" / "robotiq_2f_140" / "robotiq_2f_140_movable_visual.urdf")


def robotiq_mounting_urdf() -> str:
    """Standalone xArm-to-Robotiq flange adapter URDF with GLB visual resolved from assembly.yaml."""
    return compose_standalone_adapter_urdf("robotiq_mounting", visual="glb")


def ft_sensor_ai1500_urdf() -> str:
    """Standalone UFACTORY AI1500 six-axis F/T sensor URDF with GLB visual resolved from assembly.yaml."""
    return compose_standalone_ft_sensor_urdf("ft_sensor_ai1500", visual="glb")


def bio_gripper_g2_urdf() -> str:
    """Standalone Bio Gripper G2 URDF (no arm)."""
    return _existing_path(URDF_ROOT / "end_effectors" / "bio_gripper_g2" / "bio_gripper_g2.urdf")


def bio_gripper_g2_movable_visual_urdf() -> str:
    """Standalone Bio Gripper G2 movable visual URDF (no arm)."""
    return _existing_path(URDF_ROOT / "end_effectors" / "bio_gripper_g2" / "bio_gripper_g2_movable_visual.urdf")


def vacuum_gripper_urdf() -> str:
    """Standalone xArm / UF850 vacuum gripper URDF (no arm)."""
    return _existing_path(URDF_ROOT / "end_effectors" / "vacuum_gripper" / "vacuum_gripper.urdf")


def lite6_vacuum_gripper_visual_glb_urdf() -> str:
    """Standalone Lite6 vacuum gripper URDF with per-link GLB visual resolved from assembly.yaml."""
    return compose_standalone_end_effector_urdf("lite6_vacuum_gripper", visual="glb")


def xarm_mounting_plate_urdf() -> str:
    """Standalone xArm mounting plate URDF with GLB visual resolved from assembly.yaml."""
    return compose_standalone_mounting_plate_urdf("xarm_mounting_plate", visual="glb")


def uf850_mounting_plate_urdf() -> str:
    """Standalone UF850 mounting plate URDF with GLB visual resolved from assembly.yaml."""
    return compose_standalone_mounting_plate_urdf("uf850_mounting_plate", visual="glb")


def lite6_mounting_plate_urdf() -> str:
    """Standalone Lite 6 mounting plate URDF with GLB visual resolved from assembly.yaml."""
    return compose_standalone_mounting_plate_urdf("lite6_mounting_plate", visual="glb")


def microplate_sbs_96_well_urdf() -> str:
    """SBS 96-well microplate URDF for Bio Gripper G2 pick-and-place."""
    return _existing_path(URDF_ROOT / "objects" / "microplate" / "sbs_96_well.urdf")
