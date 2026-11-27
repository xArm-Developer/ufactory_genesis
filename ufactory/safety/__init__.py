"""Stable public API for predictive trajectory safety."""

from typing import TYPE_CHECKING, Any

from ufactory.safety.approved import ApprovedProgram
from ufactory.safety.clock import SystemClock
from ufactory.safety.interfaces import (
    ArmState,
    ArmTransport,
    CartesianStepController,
    CartesianStepOutcome,
    Clock,
    CollisionBackend,
    CollisionResult,
    JacobianKinematicsBackend,
    KinematicsBackend,
    MarginCollisionBackend,
)
from ufactory.safety.models import (
    CollisionExemption,
    FaultClass,
    PreflightCheck,
    PreflightReport,
    SafetyPolicy,
    SafetyViolation,
    ViolationType,
)
from ufactory.safety.sdk_sim import (
    SdkSimulationEvidence,
    load_sdk_simulation_evidence,
    validate_sdk_simulation,
)

if TYPE_CHECKING:
    from ufactory.safety.gate import SafetyGate


def __getattr__(name: str) -> Any:
    if name == "SafetyGate":
        from ufactory.safety.gate import SafetyGate

        return SafetyGate
    raise AttributeError(name)


__all__ = [
    "ApprovedProgram",
    "ArmState",
    "ArmTransport",
    "CartesianStepController",
    "CartesianStepOutcome",
    "Clock",
    "CollisionBackend",
    "CollisionExemption",
    "CollisionResult",
    "FaultClass",
    "JacobianKinematicsBackend",
    "KinematicsBackend",
    "MarginCollisionBackend",
    "PreflightCheck",
    "PreflightReport",
    "SafetyGate",
    "SafetyPolicy",
    "SafetyViolation",
    "SdkSimulationEvidence",
    "SystemClock",
    "ViolationType",
    "validate_sdk_simulation",
    "load_sdk_simulation_evidence",
]
