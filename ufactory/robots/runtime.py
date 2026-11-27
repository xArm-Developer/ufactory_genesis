"""Typed runtime parameters for supported UFACTORY robot profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import xml.etree.ElementTree as ET

from ufactory.robots.registry import RobotModelSpec, get_robot_profile, joint_names, robot_cli_choices
from ufactory.grippers.bio_g2 import (
    BIO_G2_JOINT_NAMES,
    BIO_GRIPPER_G2_CLOSED_GAP_M,
    BIO_GRIPPER_G2_OPEN_GAP_M,
    BIO_GRIPPER_G2_SIM_CLOSED_DRIVE,
    BIO_GRIPPER_G2_SIM_OPEN_DRIVE,
)
from ufactory.grippers.g2 import G2_JOINT_NAMES, GRIPPER_G2_SIM_OPEN_DRIVE, GRIPPER_G2_SIM_CLOSE_DRIVE
from ufactory.grippers.robotiq import (
    ROBOTIQ_2F_85_JOINT_NAMES,
    ROBOTIQ_2F_85_OPEN_GAP_M,
    ROBOTIQ_2F_85_SIM_CLOSE_DRIVE,
    ROBOTIQ_2F_85_SIM_OPEN_DRIVE,
    ROBOTIQ_2F_140_JOINT_NAMES,
    ROBOTIQ_2F_140_OPEN_GAP_M,
    ROBOTIQ_2F_140_SIM_CLOSE_DRIVE,
    ROBOTIQ_2F_140_SIM_OPEN_DRIVE,
)


FloatTuple = tuple[float, ...]
NamedPoseTuple = tuple[tuple[str, FloatTuple], ...]

# ≈ 40°/s; all real-robot motion APIs use rad / rad/s (see real_robot_session).
DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S = math.radians(40.0)


@dataclass(frozen=True)
class ArmControlParams:
    """Runtime arm-control parameters derived from a robot profile."""

    joint_names: tuple[str, ...]
    ee_link: str
    home_qpos: FloatTuple
    default_qpos: FloatTuple
    kp: FloatTuple
    kv: FloatTuple
    force_lower: FloatTuple
    force_upper: FloatTuple
    effort_limits: FloatTuple


@dataclass(frozen=True)
class DynamicsValidationParams:
    """Per-robot dynamics validation parameters."""

    default_configs: NamedPoseTuple
    stress_configs: NamedPoseTuple = ()
    abs_err_limits: FloatTuple = ()
    l2_err_limit: float = 5.0
    rel_err_limit: float = 0.15
    default_z_min_mm: float = 50.0
    default_move_speed_rad_s: float = DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S
    supports_hardware_validation: bool = False
    # Layered static analysis (L2/L3); L1 uses abs_err_limits / l2_err_limit.
    genesis_internal_abs_limits: FloatTuple = ()
    genesis_internal_l2_limit: float = 0.8
    pd_vs_pin_abs_limits: FloatTuple = ()
    pd_vs_pin_l2_limit: float = 5.0
    mass_rel_fro_limit_warn: float = 0.05
    mass_rel_fro_limit_fail: float = 0.15
    pin_vs_real_abs_scale: float = 1.5


@dataclass(frozen=True)
class GripperControlParams:
    """Optional gripper-control parameters for task and showcase examples.

    ``open_pos``/``close_pos`` are the Genesis drive-DOF values (native units:
    radians for Gripper G2's angled ``drive_joint``, metres for Lite6's
    prismatic ``finger_joint1``) at the fully-open/fully-closed physical
    two-finger gap. ``closed_gap_m``/``open_gap_m`` describe that physical gap
    range. G2 uses its linkage geometry; Lite6 uses linear prismatic travel.
    Increasing the joint value closes G2 and opens Lite6.
    """

    family: str
    drive_joint: str | None
    all_joint_names: tuple[str, ...]
    finger_link_names: tuple[str, ...]
    open_pos: float
    close_pos: float
    closed_gap_m: float
    open_gap_m: float
    kp: float
    kv: float
    force_lower: float
    force_upper: float
    damping: float
    frictionloss: float
    tool_tip_link: str | None = None
    tcp_offset_mm_deg: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    @property
    def has_actuated_joints(self) -> bool:
        return bool(self.drive_joint and self.all_joint_names)

    @property
    def has_finger_links(self) -> bool:
        return len(self.finger_link_names) >= 2


@dataclass(frozen=True)
class TaskProfile:
    """Task-level defaults and capability flags."""

    pick_place_supported: bool = False
    showcase_supported: bool = False


@dataclass(frozen=True)
class RobotRuntimeProfile:
    """Resolved profile plus runtime parameters used by examples and validators."""

    model: RobotModelSpec
    arm: ArmControlParams
    dynamics: DynamicsValidationParams
    gripper: GripperControlParams | None = None
    task: TaskProfile = field(default_factory=TaskProfile)


XARM6_KP: FloatTuple = (3000.0, 3000.0, 2000.0, 2000.0, 1000.0, 1000.0)
XARM6_KV: FloatTuple = (300.0, 300.0, 200.0, 200.0, 100.0, 100.0)
# Wrist output is about 1.4 N·m. Shared xArm gains saturate J4/J6 inside the settle window.
LITE6_KP: FloatTuple = (3000.0, 3000.0, 2000.0, 40.0, 80.0, 40.0)
LITE6_KV: FloatTuple = (300.0, 300.0, 200.0, 8.0, 16.0, 8.0)
# xArm 6 1305 official max torque. Used only when a URDF joint effort cannot be read.
_EFFORT_PARSE_FALLBACK: FloatTuple = (179.4445, 179.4445, 92.0, 81.558, 30.6, 30.6)
ABS_ERR_FRACTION = 0.10

_HARDWARE_DYNAMICS_KEYS = frozenset({"xarm5_1305", "xarm6_1305", "uf850", "lite6", "xarm7_1305"})

G2_GRIPPER_PARAMS = GripperControlParams(
    family="g2",
    drive_joint="drive_joint",
    all_joint_names=G2_JOINT_NAMES,
    finger_link_names=("left_finger", "right_finger"),
    open_pos=GRIPPER_G2_SIM_OPEN_DRIVE,
    close_pos=GRIPPER_G2_SIM_CLOSE_DRIVE,
    closed_gap_m=0.0,
    open_gap_m=0.084,
    kp=20.0,
    kv=5.0,
    force_lower=-5.0,
    force_upper=5.0,
    damping=0.1,
    frictionloss=0.0,
    tcp_offset_mm_deg=(0.0, 0.0, 178.6, 0.0, 0.0, 0.0),
)

LITE6_GRIPPER_PARAMS = GripperControlParams(
    family="lite6",
    drive_joint="finger_joint1",
    all_joint_names=("finger_joint1", "finger_joint2"),
    finger_link_names=("uflite_finger1", "uflite_finger2"),
    open_pos=0.0089,
    close_pos=0.0,
    closed_gap_m=0.020,
    open_gap_m=0.038,
    kp=500.0,
    kv=50.0,
    force_lower=-20.0,
    force_upper=20.0,
    damping=0.05,
    frictionloss=0.0,
    tcp_offset_mm_deg=(0.0, 0.0, 75.3, 0.0, 0.0, 0.0),
)

ROBOTIQ_2F_85_GRIPPER_PARAMS = GripperControlParams(
    family="robotiq_2f_85",
    drive_joint="robotiq_2f_85_drive_joint",
    all_joint_names=ROBOTIQ_2F_85_JOINT_NAMES,
    finger_link_names=("robotiq_2f_85_left_finger", "robotiq_2f_85_right_finger"),
    open_pos=ROBOTIQ_2F_85_SIM_OPEN_DRIVE,
    close_pos=ROBOTIQ_2F_85_SIM_CLOSE_DRIVE,
    closed_gap_m=0.0,
    open_gap_m=ROBOTIQ_2F_85_OPEN_GAP_M,
    kp=120.0,
    kv=10.0,
    force_lower=-30.0,
    force_upper=30.0,
    damping=0.1,
    frictionloss=0.0,
    tcp_offset_mm_deg=(0.0, 0.0, 173.8, 0.0, 0.0, 0.0),
)

ROBOTIQ_2F_140_GRIPPER_PARAMS = GripperControlParams(
    family="robotiq_2f_140",
    drive_joint="robotiq_2f_140_drive_joint",
    all_joint_names=ROBOTIQ_2F_140_JOINT_NAMES,
    finger_link_names=("robotiq_2f_140_left_finger", "robotiq_2f_140_right_finger"),
    open_pos=ROBOTIQ_2F_140_SIM_OPEN_DRIVE,
    close_pos=ROBOTIQ_2F_140_SIM_CLOSE_DRIVE,
    closed_gap_m=0.0,
    open_gap_m=ROBOTIQ_2F_140_OPEN_GAP_M,
    kp=120.0,
    kv=10.0,
    force_lower=-30.0,
    force_upper=30.0,
    damping=0.1,
    frictionloss=0.0,
    tcp_offset_mm_deg=(0.0, 0.0, 246.1, 0.0, 0.0, 0.0),
)

BIO_GRIPPER_G2_GRIPPER_PARAMS = GripperControlParams(
    family="bio_gripper_g2",
    drive_joint="bio_gripper_g2_right_finger_joint",
    all_joint_names=BIO_G2_JOINT_NAMES,
    finger_link_names=("bio_gripper_g2_left_finger", "bio_gripper_g2_right_finger"),
    open_pos=BIO_GRIPPER_G2_SIM_OPEN_DRIVE,
    close_pos=BIO_GRIPPER_G2_SIM_CLOSED_DRIVE,
    closed_gap_m=BIO_GRIPPER_G2_CLOSED_GAP_M,
    open_gap_m=BIO_GRIPPER_G2_OPEN_GAP_M,
    kp=16000.0,
    kv=40.0,
    force_lower=-20.0,
    force_upper=20.0,
    damping=0.1,
    frictionloss=0.0,
    tool_tip_link="bio_gripper_g2_tcp_link",
    tcp_offset_mm_deg=(157.64, 0.0, 59.5, 0.0, 0.0, 0.0),
)

VACUUM_GRIPPER_PARAMS = GripperControlParams(
    family="vacuum_gripper",
    drive_joint=None,
    all_joint_names=(),
    finger_link_names=(),
    open_pos=0.0,
    close_pos=0.0,
    closed_gap_m=0.0,
    open_gap_m=0.0,
    kp=0.0,
    kv=0.0,
    force_lower=0.0,
    force_upper=0.0,
    damping=0.0,
    frictionloss=0.0,
    tool_tip_link="vacuum_tcp",
    tcp_offset_mm_deg=(0.0, 0.0, 126.0, 0.0, 0.0, 0.0),
)

LITE6_VACUUM_GRIPPER_PARAMS = GripperControlParams(
    family="lite6_vacuum_gripper",
    drive_joint=None,
    all_joint_names=(),
    finger_link_names=(),
    open_pos=0.0,
    close_pos=0.0,
    closed_gap_m=0.0,
    open_gap_m=0.0,
    kp=0.0,
    kv=0.0,
    force_lower=0.0,
    force_upper=0.0,
    damping=0.0,
    frictionloss=0.0,
    tool_tip_link="link_tcp",
    tcp_offset_mm_deg=(0.0, 0.0, 61.0, 0.0, 0.0, 0.0),
)


def _tuple(values) -> FloatTuple:
    return tuple(map(float, values))


def abs_err_limits_from_efforts(
    efforts: FloatTuple,
    *,
    fraction: float = ABS_ERR_FRACTION,
) -> FloatTuple:
    """Per-joint absolute torque error limits (Nm) as a fraction of URDF effort."""
    return tuple(float(e) * fraction for e in efforts)


def _l2_err_limit_from_abs(abs_limits: FloatTuple) -> float:
    return max(2.0, sum(v * v for v in abs_limits) ** 0.5)


def _fit(values: tuple[float, ...], n: int, *, fill: float = 0.0) -> FloatTuple:
    if len(values) >= n:
        return _tuple(values[:n])
    return _tuple((*values, *([fill] * (n - len(values)))))


def _parse_effort_limits(profile: RobotModelSpec) -> FloatTuple:
    urdf = profile.assets_dir / profile.default_urdf
    fallback = _fit(_EFFORT_PARSE_FALLBACK, profile.dof, fill=_EFFORT_PARSE_FALLBACK[-1])
    try:
        root = ET.parse(str(urdf)).getroot()
    except Exception:
        return fallback
    by_name = {joint.get("name"): joint for joint in root.findall("joint")}
    efforts: list[float] = []
    for name in joint_names(profile):
        joint = by_name.get(name)
        limit = joint.find("limit") if joint is not None else None
        try:
            efforts.append(abs(float(limit.get("effort"))) if limit is not None else fallback[len(efforts)])
        except Exception:
            efforts.append(fallback[len(efforts)])
    return _tuple(efforts)


def _default_qpos(profile: RobotModelSpec) -> FloatTuple:
    if profile.key == "xarm6_1305":
        return (0.0, -0.5, 0.0, 0.0, 0.5, 0.0)
    if profile.key == "lite6":
        return (0.0, -0.6, 0.0, 0.0, 0.6, 0.0)
    return _fit((0.0, -0.5, 0.0, 0.0, 0.5, 0.0, 0.0), profile.dof)


def _generic_default_configs(profile: RobotModelSpec, default_qpos: FloatTuple) -> NamedPoseTuple:
    home = tuple(0.0 for _ in range(profile.dof))
    small_pos = list(home)
    small_neg = list(home)
    for idx in range(profile.dof):
        small_pos[idx] = 0.15 if idx % 2 == 0 else -0.15
        small_neg[idx] = -small_pos[idx]
    return (
        ("0", _tuple(home)),
        ("1", _tuple(default_qpos)),
        ("2", _tuple(small_pos)),
        ("3", _tuple(small_neg)),
    )


def _arm_params(profile: RobotModelSpec) -> ArmControlParams:
    efforts = _parse_effort_limits(profile)
    if profile.key == "xarm6_1305":
        kp = XARM6_KP
        kv = XARM6_KV
    elif profile.key == "lite6":
        kp = LITE6_KP
        kv = LITE6_KV
    else:
        kp = _fit((3000.0, 3000.0, 2000.0, 2000.0, 1000.0, 1000.0, 800.0), profile.dof, fill=800.0)
        kv = _fit((300.0, 300.0, 200.0, 200.0, 100.0, 100.0, 80.0), profile.dof, fill=80.0)
    return ArmControlParams(
        joint_names=joint_names(profile),
        ee_link=profile.ee_link,
        home_qpos=tuple(0.0 for _ in range(profile.dof)),
        default_qpos=_default_qpos(profile),
        kp=_fit(kp, profile.dof),
        kv=_fit(kv, profile.dof),
        force_lower=tuple(-abs(v) for v in efforts),
        force_upper=efforts,
        effort_limits=efforts,
    )


def _dynamics_params(profile: RobotModelSpec, arm: ArmControlParams) -> DynamicsValidationParams:
    from ufactory.dynamics.poses import default_configs_named_tuple, stress_configs_for_robot

    internal_abs = tuple(0.5 for _ in arm.effort_limits)
    yaml_default = default_configs_named_tuple(profile.key)
    yaml_stress = stress_configs_for_robot(profile.key)
    default_configs = yaml_default or _generic_default_configs(profile, arm.default_qpos)
    stress_configs = yaml_stress
    abs_limits = abs_err_limits_from_efforts(arm.effort_limits)
    l2_limit = _l2_err_limit_from_abs(abs_limits)
    return DynamicsValidationParams(
        default_configs=default_configs,
        stress_configs=stress_configs,
        abs_err_limits=abs_limits,
        l2_err_limit=l2_limit,
        rel_err_limit=0.15,
        default_z_min_mm=50.0,
        supports_hardware_validation=profile.key in _HARDWARE_DYNAMICS_KEYS,
        genesis_internal_abs_limits=internal_abs,
        genesis_internal_l2_limit=0.8,
        pd_vs_pin_abs_limits=abs_limits,
        pd_vs_pin_l2_limit=l2_limit,
    )


def _task_profile(profile: RobotModelSpec) -> TaskProfile:
    return TaskProfile(
        pick_place_supported=profile.supports_gripper_g2 or profile.supports_lite6_gripper,
        showcase_supported=profile.supports_gripper_g2 or profile.supports_lite6_gripper,
    )


def _gripper_params_for(profile: RobotModelSpec, *, gripper: str | None = None) -> GripperControlParams | None:
    choice = None if gripper in (None, "default") else str(gripper)
    if choice is None:
        if profile.supports_gripper_g2:
            return G2_GRIPPER_PARAMS
        if profile.supports_lite6_gripper:
            return LITE6_GRIPPER_PARAMS
        return None
    if choice in {"g2", "gripper_g2"}:
        if not profile.supports_gripper_g2:
            raise ValueError(f"Robot {profile.key} does not support gripper {choice!r}")
        return G2_GRIPPER_PARAMS
    if choice == "lite6":
        if not profile.supports_lite6_gripper:
            raise ValueError(f"Robot {profile.key} does not support gripper {choice!r}")
        return LITE6_GRIPPER_PARAMS
    if choice == "robotiq_2f_85":
        if not profile.supports_robotiq_2f_85:
            raise ValueError(f"Robot {profile.key} does not support gripper {choice!r}")
        return ROBOTIQ_2F_85_GRIPPER_PARAMS
    if choice == "robotiq_2f_140":
        if not profile.supports_robotiq_2f_140:
            raise ValueError(f"Robot {profile.key} does not support gripper {choice!r}")
        return ROBOTIQ_2F_140_GRIPPER_PARAMS
    if choice == "bare":
        return None
    if choice in {"bio_gripper_g2", "bio_g2"}:
        if not profile.supports_bio_gripper_g2:
            raise ValueError(f"Robot {profile.key} does not support gripper {choice!r}")
        return BIO_GRIPPER_G2_GRIPPER_PARAMS
    if choice == "vacuum_gripper":
        if not profile.supports_vacuum_gripper:
            raise ValueError(f"Robot {profile.key} does not support gripper {choice!r}")
        return VACUUM_GRIPPER_PARAMS
    if choice == "lite6_vacuum_gripper":
        if not profile.supports_lite6_vacuum_gripper:
            raise ValueError(f"Robot {profile.key} does not support gripper {choice!r}")
        return LITE6_VACUUM_GRIPPER_PARAMS
    raise ValueError(f"Unknown gripper {choice!r} for robot {profile.key}")


def _build_runtime_profile(profile: RobotModelSpec, *, gripper: str | None = None) -> RobotRuntimeProfile:
    arm = _arm_params(profile)
    return RobotRuntimeProfile(
        model=profile,
        arm=arm,
        dynamics=_dynamics_params(profile, arm),
        gripper=_gripper_params_for(profile, gripper=gripper),
        task=_task_profile(profile),
    )


def get_robot_runtime_profile(robot_key: str, *, gripper: str | None = None) -> RobotRuntimeProfile:
    """Resolve a robot key or alias to typed runtime parameters."""
    return _build_runtime_profile(get_robot_profile(robot_key), gripper=gripper)


def robot_runtime_cli_choices() -> list[str]:
    return robot_cli_choices()
