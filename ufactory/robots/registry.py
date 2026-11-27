"""Robot model metadata for multi-arm support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
URDF_ROOT = PROJECT_ROOT / "assets" / "urdf"


@dataclass(frozen=True)
class RobotModelSpec:
    """Metadata for a supported UFACTORY arm variant."""

    key: str
    robot_name: str
    variant: str
    dof: int
    ee_link: str
    assets_dir: Path
    default_urdf: str
    visual_glb_urdf: str
    kinematics_prefix: str
    # Model id the controller reports as XArmAPI ``device_type`` (with ``axis == dof``).
    controller_device_type: int

    # End-effector compatibility comes from assets/urdf/assembly.yaml (single source of truth).
    def supports(self, end_effector: str) -> bool:
        from ufactory.robots.assembly import supports_end_effector

        return supports_end_effector(self.key, end_effector)

    @property
    def supports_gripper_g2(self) -> bool:
        return self.supports("gripper_g2")

    @property
    def supports_robotiq_2f_85(self) -> bool:
        return self.supports("robotiq_2f_85")

    @property
    def supports_robotiq_2f_140(self) -> bool:
        return self.supports("robotiq_2f_140")

    @property
    def supports_bio_gripper_g2(self) -> bool:
        return self.supports("bio_gripper_g2")

    @property
    def supports_vacuum_gripper(self) -> bool:
        return self.supports("vacuum_gripper")

    @property
    def supports_lite6_gripper(self) -> bool:
        return self.supports("lite6_gripper")

    @property
    def supports_lite6_vacuum_gripper(self) -> bool:
        return self.supports("lite6_vacuum_gripper")

    def supports_accessory(self, accessory: str, end_effector: str | None = None) -> bool:
        """Whether a camera stand or camera (with ``end_effector``; ``None`` = bare arm) fits this arm."""
        from ufactory.robots.assembly import supports_accessory

        return supports_accessory(self.key, accessory, end_effector)

    @property
    def supports_camera_stand(self) -> bool:
        return self.supports_accessory("camera_stand")

    def supports_ft_sensor(self, ft_sensor: str) -> bool:
        """Whether a six-axis force/torque sensor fits this arm's tool flange."""
        from ufactory.robots.assembly import supports_ft_sensor

        return supports_ft_sensor(self.key, ft_sensor)

    @property
    def supports_ft_sensor_ai1500(self) -> bool:
        return self.supports_ft_sensor("ft_sensor_ai1500")

    @property
    def supports_mounting_plate(self) -> bool:
        return True

    @property
    def mounting_plate(self) -> str:
        """Canonical mounting plate name for this arm from ``assets/urdf/assembly.yaml``."""
        from ufactory.robots.assembly import mounting_plate_for

        return mounting_plate_for(self.key).name


def _assets(robot: str) -> Path:
    return URDF_ROOT / "robots" / robot


def _xarm_profile(dof: int) -> RobotModelSpec:
    robot = f"xarm{dof}"
    key = f"{robot}_1305"
    return RobotModelSpec(
        key=key,
        robot_name=robot,
        variant="1305",
        dof=dof,
        ee_link=f"link{dof}",
        assets_dir=_assets(robot),
        default_urdf=f"{key}.urdf",
        visual_glb_urdf=f"{key}_visual.glb.urdf",
        kinematics_prefix=robot,
        controller_device_type=dof,
    )


ROBOT_PROFILES: dict[str, RobotModelSpec] = {
    "xarm5_1305": _xarm_profile(5),
    "xarm7_1305": _xarm_profile(7),
    "lite6": RobotModelSpec(
        key="lite6",
        robot_name="lite6",
        variant="",
        dof=6,
        ee_link="link6",
        assets_dir=_assets("lite6"),
        default_urdf="lite6.urdf",
        visual_glb_urdf="lite6_visual.glb.urdf",
        kinematics_prefix="lite6",
        controller_device_type=9,
    ),
    "uf850": RobotModelSpec(
        key="uf850",
        robot_name="uf850",
        variant="",
        dof=6,
        ee_link="link6",
        assets_dir=_assets("uf850"),
        default_urdf="uf850.urdf",
        visual_glb_urdf="uf850_visual.glb.urdf",
        kinematics_prefix="uf850",
        controller_device_type=12,
    ),
    "xarm6_1305": _xarm_profile(6),
}


def get_robot_profile(key: str) -> RobotModelSpec:
    return ROBOT_PROFILES[get_profile_key_for_robot_name(key)]


def joint_names(profile: RobotModelSpec) -> tuple[str, ...]:
    return tuple(f"joint{i}" for i in range(1, profile.dof + 1))


def get_profile_key_for_robot_name(robot_name: str) -> str:
    """Resolve a robot name or profile key to a canonical profile key.

    Accepts both profile keys (``xarm6_1305``) and short robot names (``xarm6``).
    Short xArm names resolve to the XI1305 profile (``xarm6`` -> ``xarm6_1305``).
    """
    if robot_name in ROBOT_PROFILES:
        return robot_name
    preferred = f"{robot_name}_1305"
    if preferred in ROBOT_PROFILES:
        return preferred
    matches = [key for key, profile in ROBOT_PROFILES.items() if profile.robot_name == robot_name]
    if matches:
        return sorted(matches)[0]
    raise KeyError(f"Unknown robot: {robot_name}")


def robot_name_for_controller(axis: int, device_type: int) -> str | None:
    """Robot name of a controller reporting ``(axis, device_type)``; ``None`` if no single model matches."""
    names = {
        profile.robot_name
        for profile in ROBOT_PROFILES.values()
        if (profile.dof, profile.controller_device_type) == (axis, device_type)
    }
    return names.pop() if len(names) == 1 else None


def robot_cli_choices() -> list[str]:
    """Sorted ``--robot`` choices: profile keys plus short robot-name aliases."""
    keys = set(ROBOT_PROFILES)
    aliases = {profile.robot_name for profile in ROBOT_PROFILES.values()}
    return sorted(keys | aliases)
