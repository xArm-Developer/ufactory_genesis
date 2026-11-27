"""Dependency inversion ports used by the safety and execution domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ufactory.types import FloatArray


@dataclass(frozen=True)
class ArmState:
    q_rad: FloatArray
    monotonic_ns: int
    ready: bool
    error_code: int = 0
    state_code: int = 0
    serial_number: str = ""
    robot_key: str = ""


@dataclass(frozen=True)
class CollisionResult:
    colliding: bool
    min_distance_m: float
    link_a: str = ""
    link_b: str = ""
    environment: bool = False
    geometry_a: str = ""
    geometry_b: str = ""


@runtime_checkable
class ArmTransport(Protocol):
    def read_state(self) -> ArmState: ...
    def send_joint_target(self, q_rad: FloatArray) -> int: ...
    def send_cartesian_target(self, pose: FloatArray) -> int: ...
    def pause(self) -> int: ...
    def stop(self) -> int: ...
    def disconnect(self) -> None: ...


@runtime_checkable
class KinematicsBackend(Protocol):
    def forward(self, q_rad: FloatArray) -> FloatArray:
        """Return base-frame end-effector pose as xyz or xyz+quaternion."""

    def inverse(self, pose: FloatArray, seed_q_rad: FloatArray) -> FloatArray:
        """Solve xyz or xyz+quaternion_xyzw while preserving seed continuity."""


@runtime_checkable
class JacobianKinematicsBackend(KinematicsBackend, Protocol):
    """Kinematics backend exposing base-frame spatial geometric Jacobian."""

    def jacobian(self, q_rad: FloatArray) -> FloatArray:
        """Return 6xN spatial Jacobian ``[v_m; w_rad]`` in base frame."""


@runtime_checkable
class CartesianStepOutcome(Protocol):
    """Result of one Cartesian control step returned by a step controller."""

    @property
    def q_rad(self) -> FloatArray: ...

    @property
    def used_approx(self) -> bool: ...

    @property
    def position_error_m(self) -> float: ...

    @property
    def orientation_error_rad(self) -> float: ...


@runtime_checkable
class CartesianStepController(KinematicsBackend, Protocol):
    """Unified controller capability supporting single-step Cartesian IK and approx motion."""

    @property
    def is_approx_motion(self) -> bool: ...

    def solve_step(
        self,
        target_pose: FloatArray,
        seed_q_rad: FloatArray,
        *,
        dt_s: float | None = None,
        include_tcp: bool = False,
    ) -> CartesianStepOutcome: ...


@runtime_checkable
class CollisionBackend(Protocol):
    def check(self, q_rad: FloatArray, *, stage: str, gripper_drive: float | None = None) -> CollisionResult: ...
    def check_all(
        self, q_rad: FloatArray, *, stage: str, gripper_drive: float | None = None
    ) -> tuple[CollisionResult, ...]: ...


@runtime_checkable
class MarginCollisionBackend(Protocol):
    """Optional capability returning only pairs at or inside a safety margin."""

    def check_all_within_margin(
        self,
        q_rad: FloatArray,
        *,
        stage: str,
        margin_m: float,
        gripper_drive: float | None = None,
    ) -> tuple[CollisionResult, ...]: ...


@runtime_checkable
class Clock(Protocol):
    def monotonic_ns(self) -> int: ...
    def wait_until_ns(self, deadline_ns: int) -> None: ...
