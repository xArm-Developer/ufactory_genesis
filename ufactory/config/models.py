"""Versioned immutable runtime configuration models."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Mapping

from ufactory.safety.models import SafetyPolicy


FloatTuple = tuple[float, ...]
TcpOffsetTuple = tuple[float, float, float, float, float, float]
G2_PHYSICS_PROFILE = "g2_calibrated_v1"
ROBOTIQ_2F_85_PHYSICS_PROFILE = "robotiq_2f_85_calibrated_v1"
ROBOTIQ_2F_140_PHYSICS_PROFILE = "robotiq_2f_140_calibrated_v1"
# Pyramidal friction with DEFAULT_IMPRATIO=100.0 in make_rigid_options() stiffens
# tangential regularization in the main Newton solver so a held cube does not
# creep under gravity, keeping the post-solve no-slip PGS pass off globally.
G2_NOSLIP_ITERATIONS = 0


def _immutable_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _immutable_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class RobotSpec:
    """Robot identity, assets, joints, and declared capabilities."""

    key: str
    robot_name: str
    variant: str
    dof: int
    joint_names: tuple[str, ...]
    ee_link: str
    assets_dir: str
    urdf: str
    capabilities: frozenset[str]
    environment_contact_pairs: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.dof < 1 or len(self.joint_names) != self.dof:
            raise ValueError("RobotSpec joint_names must match dof")
        if len(set(self.joint_names)) != self.dof:
            raise ValueError("RobotSpec joint_names must be unique")
        if not self.key or not self.robot_name or not self.ee_link or not self.urdf:
            raise ValueError("RobotSpec identity and asset fields cannot be empty")
        if any(len(pair) != 2 or pair[0] == pair[1] for pair in self.environment_contact_pairs):
            raise ValueError("each environment contact pair must contain two distinct names")


@dataclass(frozen=True)
class ArmControlProfile:
    """Joint-space controller defaults, with explicit SI-unit field names."""

    default_qpos_rad: FloatTuple
    kp: FloatTuple
    kv: FloatTuple
    force_lower_nm: FloatTuple
    force_upper_nm: FloatTuple
    effort_limits_nm: FloatTuple

    def validate_dof(self, dof: int) -> None:
        for name in (
            "default_qpos_rad",
            "kp",
            "kv",
            "force_lower_nm",
            "force_upper_nm",
            "effort_limits_nm",
        ):
            if len(getattr(self, name)) != dof:
                raise ValueError(f"arm.{name} expected {dof} values")


@dataclass(frozen=True)
class GripperProfile:
    """Gripper mapping and capabilities consumed through an adapter."""

    adapter: str
    drive_joint: str
    all_joint_names: tuple[str, ...]
    finger_link_names: tuple[str, ...]
    open_drive: float
    closed_drive: float
    open_gap_m: float
    closed_gap_m: float
    kp: float
    kv: float
    force_lower_n: float
    force_upper_n: float
    damping: float
    frictionloss: float
    real_command: bool
    feedback: bool
    closed_loop: bool
    allowed_contact_links: frozenset[str]
    tool_tip_offset_z_m: float
    finger_pad_below_center_m: float
    finger_close_descent_m: float
    grasp_table_clearance_m: float
    grasp_height_extra_m: float
    # URDF collision geometries (``<link>_<collision index>``) whose mutual contact Genesis physics ignores.
    excluded_geometry_pairs: tuple[tuple[str, str], ...] = ()
    # Flange-to-TCP 6D offset [x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]
    # matching xarm-python-sdk default (is_radian=False) and UFACTORY Studio.
    tcp_offset: TcpOffsetTuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    @property
    def has_actuated_joints(self) -> bool:
        return bool(self.drive_joint and self.all_joint_names)

    def __post_init__(self) -> None:
        if not self.adapter:
            raise ValueError("gripper adapter is required")
        is_zero_dof = not self.drive_joint and not self.all_joint_names
        if not is_zero_dof and not self.drive_joint:
            raise ValueError("gripper adapter and drive joint are required")
        tcp = tuple(float(value) for value in self.tcp_offset)
        if len(tcp) != 6 or not all(math.isfinite(value) for value in tcp):
            raise ValueError("gripper tcp_offset must contain 6 finite values [x_mm, y_mm, z_mm, r_deg, p_deg, y_deg]")
        object.__setattr__(self, "tcp_offset", (tcp[0], tcp[1], tcp[2], tcp[3], tcp[4], tcp[5]))
        pairs = self.excluded_geometry_pairs
        if any(
            len(pair) != 2 or any(not isinstance(name, str) or not name.strip() for name in pair) or pair[0] == pair[1]
            for pair in pairs
        ):
            raise ValueError("excluded_geometry_pairs must contain pairs of distinct geometry names")
        if len({frozenset(pair) for pair in pairs}) != len(pairs):
            raise ValueError("excluded_geometry_pairs cannot contain duplicate pairs")
        object.__setattr__(self, "excluded_geometry_pairs", tuple(tuple(pair) for pair in pairs))
        if not self.closed_gap_m >= 0.0:
            raise ValueError("gripper gaps cannot be negative")
        if is_zero_dof:
            if self.open_gap_m < self.closed_gap_m:
                raise ValueError("gripper open_gap_m cannot be below closed_gap_m")
            if self.force_lower_n > self.force_upper_n:
                raise ValueError("gripper force_lower_n cannot exceed force_upper_n")
            if not math.isfinite(float(self.tool_tip_offset_z_m)) or self.tool_tip_offset_z_m < 0.0:
                raise ValueError("gripper tool_tip_offset_z_m must be finite and non-negative")
        else:
            if self.open_gap_m <= self.closed_gap_m:
                raise ValueError("gripper open_gap_m must be greater than closed_gap_m")
            if self.force_lower_n >= self.force_upper_n:
                raise ValueError("gripper force_lower_n must be below force_upper_n")
            if self.tool_tip_offset_z_m <= 0.0:
                raise ValueError("gripper tool_tip_offset_z_m must be positive")
        for name in (
            "finger_pad_below_center_m",
            "finger_close_descent_m",
            "grasp_table_clearance_m",
            "grasp_height_extra_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"gripper {name} must be finite and non-negative")


@dataclass(frozen=True)
class TaskProfile:
    """Task geometry, timing, and explicitly allowed contact phases."""

    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    allowed_contacts: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _immutable_mapping(self.parameters))
        object.__setattr__(self, "allowed_contacts", _immutable_mapping(self.allowed_contacts))
        if not self.name:
            raise ValueError("task name cannot be empty")


@dataclass(frozen=True)
class GraspObjectSpec:
    """Shared physical specification for the pick-place reference object."""

    size_m: tuple[float, float, float]
    mass_kg: float

    def __post_init__(self) -> None:
        size_m = tuple(float(value) for value in self.size_m)
        mass_kg = float(self.mass_kg)
        if len(size_m) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in size_m):
            raise ValueError("GraspObjectSpec size_m must contain three positive finite values")
        if not math.isfinite(mass_kg) or mass_kg <= 0.0:
            raise ValueError("GraspObjectSpec mass_kg must be finite and positive")
        object.__setattr__(self, "size_m", (size_m[0], size_m[1], size_m[2]))
        object.__setattr__(self, "mass_kg", mass_kg)

    @property
    def rest_center_z_m(self) -> float:
        """Object-center height when the box rests on the base-frame z=0 table."""

        return float(self.size_m[2]) / 2.0


def resolve_manipulation_object_spec(config: ResolvedRuntimeConfig | TaskProfile) -> GraspObjectSpec:
    """Resolve the shared manipulation-object contract from task configuration."""

    task = config.task if isinstance(config, ResolvedRuntimeConfig) else config
    if task.name not in {"pick_place", "packaging_showcase"}:
        raise ValueError(f"task {task.name!r} has no manipulation-object specification")
    size = tuple(float(value) for value in task.parameters["object_size_m"])
    return GraspObjectSpec(
        size_m=(size[0], size[1], size[2]),
        mass_kg=float(task.parameters["object_mass_kg"]),
    )


def resolve_pick_place_object_spec(config: ResolvedRuntimeConfig | TaskProfile) -> GraspObjectSpec:
    """Resolve the pick-place reference-object specification."""

    task = config.task if isinstance(config, ResolvedRuntimeConfig) else config
    if task.name != "pick_place":
        raise ValueError(f"task {task.name!r} has no pick-place object specification")
    return resolve_manipulation_object_spec(task)


@dataclass(frozen=True)
class MotionConfig:
    """Trajectory sampling and kinematic limits in SI units."""

    rate_hz: float
    joint_speed_rad_s: float
    joint_acceleration_rad_s2: float
    cartesian_speed_m_s: float
    cartesian_acceleration_m_s2: float
    gripper_duration_s: float
    allow_approx_motion: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.allow_approx_motion, bool):
            raise ValueError("motion.allow_approx_motion must be a boolean")


@dataclass(frozen=True)
class G2SimulationConfig:
    """SDK position and percentage commands for the calibrated G2 servo."""

    position_mm: float = 0.0
    speed_percent: float = 50.0
    force_percent: float = 50.0

    def __post_init__(self) -> None:
        for name, low, high in (("position_mm", 0, 84), ("speed_percent", 0, 100), ("force_percent", 1, 100)):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"simulation.g2.{name} must be finite and within {low}–{high}")


@dataclass(frozen=True)
class RobotiqSimulationConfig:
    """SDK position and percentage commands for the calibrated Robotiq 2F-85/2F-140 servo."""

    position_mm: float = 0.0
    speed_percent: float = 50.0
    force_percent: float = 50.0

    def __post_init__(self) -> None:
        for name, low, high in (("position_mm", 0, 140), ("speed_percent", 0, 100), ("force_percent", 1, 100)):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"simulation.robotiq.{name} must be finite and within {low}–{high}")


CONSTRAINT_SOLVERS = ("newton",)
FRICTION_CONES = ("pyramidal", "elliptic")
CONTACT_RESOLUTIONS = ("convex", "signorini")


def validate_rigid_physics(
    *,
    constraint_solver: str,
    friction_cone: str,
    contact_resolution: str,
    noslip_iterations: int,
) -> None:
    """Reject unsupported or internally inconsistent rigid-contact settings."""

    if constraint_solver not in CONSTRAINT_SOLVERS:
        raise ValueError(f"constraint_solver must be one of {CONSTRAINT_SOLVERS}")
    if friction_cone not in FRICTION_CONES:
        raise ValueError(f"friction_cone must be one of {FRICTION_CONES}")
    if contact_resolution not in CONTACT_RESOLUTIONS:
        raise ValueError(f"contact_resolution must be one of {CONTACT_RESOLUTIONS}")
    if noslip_iterations < 0:
        raise ValueError("noslip_iterations must be non-negative")
    if friction_cone == "elliptic" and noslip_iterations:
        raise ValueError("elliptic friction is incompatible with the noslip solver")
    if contact_resolution == "signorini" and (
        constraint_solver != "newton" or friction_cone != "elliptic" or noslip_iterations != 0
    ):
        raise ValueError("signorini contact requires Newton, elliptic friction, and noslip_iterations=0")


@dataclass(frozen=True)
class SimulationConfig:
    """Validated Genesis process and rigid-solver settings."""

    backend: str
    precision: str
    seed: int
    substeps: int
    constraint_solver: str
    solver_iterations: int
    noslip_iterations: int
    friction_cone: str
    contact_resolution: str
    constraint_time_constant_s: float
    use_gjk_collision: bool | None
    show_viewer: bool
    g2: G2SimulationConfig = G2SimulationConfig()
    robotiq: RobotiqSimulationConfig = RobotiqSimulationConfig()
    rate_hz: float = 50.0

    def __post_init__(self) -> None:
        if self.backend not in {"cpu", "gpu"}:
            raise ValueError("simulation backend must be cpu or gpu")
        if self.precision not in {"32", "64"}:
            raise ValueError("simulation precision must be 32 or 64")
        if isinstance(self.rate_hz, bool) or not math.isfinite(float(self.rate_hz)) or float(self.rate_hz) <= 0.0:
            raise ValueError("simulation rate_hz must be finite and positive")
        if self.substeps < 1 or self.solver_iterations < 1 or self.noslip_iterations < 0:
            raise ValueError("simulation iteration counts are invalid")
        if self.constraint_time_constant_s <= 0.0:
            raise ValueError("simulation constraint time constant must be positive")
        validate_rigid_physics(
            constraint_solver=self.constraint_solver,
            friction_cone=self.friction_cone,
            contact_resolution=self.contact_resolution,
            noslip_iterations=self.noslip_iterations,
        )


@dataclass(frozen=True)
class ResolvedRuntimeConfig:
    """Fully merged configuration plus reproducibility provenance."""

    schema_version: int
    robot: RobotSpec
    arm: ArmControlProfile
    gripper: GripperProfile | None
    task: TaskProfile
    motion: MotionConfig
    simulation: SimulationConfig
    safety: SafetyPolicy
    sources: tuple[str, ...]
    sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported runtime schema_version: {self.schema_version}")
        self.arm.validate_dof(self.robot.dof)
        if len(self.sha256) != 64:
            raise ValueError("resolved configuration sha256 must contain 64 hexadecimal characters")
