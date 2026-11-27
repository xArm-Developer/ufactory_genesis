"""Firmware SDK-simulation evidence for Cartesian real execution."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields
import hashlib
import math
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np

from ufactory.safety.interfaces import CollisionBackend, CollisionResult, KinematicsBackend
from ufactory.safety.models import SafetyPolicy
from ufactory.types import FloatArray


SDK_SIMULATION_EVIDENCE_MAX_AGE_S = 300.0
SDK_SIMULATION_EVIDENCE_SCHEMA_VERSION = 2
# servo_cartesian commands a full pose (6 DoF). An arm with more joints is
# redundant: its firmware may resolve the extra DoF (xArm7 elbow) differently
# from the host IK while the end effector stays on the commanded pose.
TASK_SPACE_DOF = 6


def _quaternion_angle(a: FloatArray, b: FloatArray) -> float:
    """Rotation angle (rad) between two xyzw quaternions, robust for small angles."""

    qa = np.asarray(a, dtype=np.float64) / np.linalg.norm(a)
    qb = np.asarray(b, dtype=np.float64) / np.linalg.norm(b)
    if float(qa @ qb) < 0.0:
        qb = -qb
    # |qa - qb| = 2 sin(phi/2), |qa + qb| = 2 cos(phi/2) with cos(phi) = qa.qb; the rotation angle is 2 phi.
    return 4.0 * math.atan2(float(np.linalg.norm(qa - qb)), float(np.linalg.norm(qa + qb)))


def stream_sha256(stream: FloatArray) -> str:
    values = np.asarray(stream, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(tuple(values.shape)).encode())
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class SdkSimulationEvidence:
    schema_version: int
    passed: bool
    robot_key: str
    serial_number: str
    program_sha256: str
    config_sha256: str
    start_q_sha256: str
    shadow_stream_sha256: str
    firmware_stream_sha256: str
    created_at_unix_s: float
    samples: int
    max_joint_error_rad: float
    max_ee_error_m: float
    max_ee_rotation_error_rad: float
    self_motion_allowed: bool
    min_distance_m: float
    failures: tuple[str, ...] = ()


def validate_sdk_simulation(
    *,
    robot_key: str,
    serial_number: str,
    program_sha256: str,
    config_sha256: str,
    shadow_joint_stream_rad: FloatArray,
    firmware_joint_stream_rad: FloatArray,
    stages: tuple[str, ...],
    policy: SafetyPolicy,
    kinematics: KinematicsBackend,
    collision: CollisionBackend,
    allowed_collision: Callable[[CollisionResult, str], bool] | None = None,
    joint_lower_rad: FloatArray | None = None,
    joint_upper_rad: FloatArray | None = None,
    created_at_unix_s: float | None = None,
) -> SdkSimulationEvidence:
    """Compare host shadow and firmware feedback and recheck every sample.

    Both streams must reach the same end-effector pose: position within
    ``max_shadow_ee_error_m``, rotation within ``max_shadow_joint_error_rad``.
    A non-redundant arm must also match joint by joint. A redundant arm (more
    joints than the 6-DoF pose) may differ by self-motion, e.g. the xArm7
    elbow angle, so its joints are not compared; the firmware stream is then
    bounded by the joint limits (less ``joint_limit_margin_rad``), which are
    required, as well as by the collision checks every sample gets.
    """

    shadow = np.asarray(shadow_joint_stream_rad, dtype=np.float64)
    firmware = np.asarray(firmware_joint_stream_rad, dtype=np.float64)
    failures: list[str] = []
    if shadow.ndim != 2 or firmware.shape != shadow.shape or len(stages) != len(shadow):
        failures.append("shadow, firmware, and stage timelines must have identical dimensions")
    if not np.all(np.isfinite(shadow)) or not np.all(np.isfinite(firmware)):
        failures.append("SDK simulation streams contain NaN or infinity")
    self_motion_allowed = shadow.ndim == 2 and shadow.shape[1] > TASK_SPACE_DOF
    max_joint_error = math.inf
    max_ee_error = math.inf
    max_ee_rotation_error = math.inf
    min_distance = math.inf
    if not failures:
        max_joint_error = float(np.max(np.abs(firmware - shadow), initial=0.0))
        if not self_motion_allowed and max_joint_error > policy.max_shadow_joint_error_rad:
            failures.append(
                f"firmware/shadow joint error {max_joint_error:.6g} rad exceeds "
                f"{policy.max_shadow_joint_error_rad:.6g} rad"
            )
        if joint_lower_rad is not None and joint_upper_rad is not None:
            lower = np.asarray(joint_lower_rad, dtype=np.float64).reshape(-1) + policy.joint_limit_margin_rad
            upper = np.asarray(joint_upper_rad, dtype=np.float64).reshape(-1) - policy.joint_limit_margin_rad
            if lower.shape != (shadow.shape[1],) or upper.shape != (shadow.shape[1],):
                failures.append("joint limits must have one value per joint")
            else:
                outside = np.argwhere((firmware < lower) | (firmware > upper))
                if outside.size:
                    sample, joint = (int(v) for v in outside[0])
                    failures.append(
                        f"firmware stream exceeds joint {joint + 1} limit at sample {sample}: "
                        f"{firmware[sample, joint]:.6g} rad outside [{lower[joint]:.6g}, {upper[joint]:.6g}] "
                        f"({len(outside)} sample-joints)"
                    )
        elif self_motion_allowed:
            failures.append("joint limits are required to accept self-motion of a redundant arm")
        ee_errors: list[float] = []
        rotation_errors: list[float] = []
        orientation_missing = False
        distances: list[float] = []
        for index, (host_q, firmware_q, stage) in enumerate(zip(shadow, firmware, stages, strict=True)):
            host_pose = np.asarray(kinematics.forward(host_q), dtype=np.float64).reshape(-1)
            firmware_pose = np.asarray(kinematics.forward(firmware_q), dtype=np.float64).reshape(-1)
            if host_pose.size < 3 or firmware_pose.size < 3:
                failures.append(f"FK returned an invalid pose at sample {index}")
                continue
            ee_errors.append(float(np.linalg.norm(host_pose[:3] - firmware_pose[:3])))
            if host_pose.size >= 7 and firmware_pose.size >= 7:
                rotation_errors.append(_quaternion_angle(host_pose[3:7], firmware_pose[3:7]))
            else:
                orientation_missing = True
            check_all = getattr(collision, "check_all", None)
            results = (
                tuple(check_all(firmware_q, stage=stage))
                if check_all is not None
                else (collision.check(firmware_q, stage=stage),)
            )
            for result in results:
                distances.append(float(result.min_distance_m))
                unsafe = result.colliding or result.min_distance_m < policy.min_collision_distance_m
                if unsafe and not (allowed_collision and allowed_collision(result, stage)):
                    failures.append(
                        f"firmware stream unsafe at sample {index}: "
                        f"{result.link_a}/{result.link_b} distance={result.min_distance_m:.6g} m"
                    )
        max_ee_error = max(ee_errors, default=math.inf)
        min_distance = min(distances, default=math.inf)
        if max_ee_error > policy.max_shadow_ee_error_m:
            failures.append(
                f"firmware/shadow FK error {max_ee_error:.6g} m exceeds {policy.max_shadow_ee_error_m:.6g} m"
            )
        if orientation_missing:
            if self_motion_allowed:
                failures.append("FK must return xyz+quaternion to compare a redundant arm by end-effector pose")
        else:
            max_ee_rotation_error = max(rotation_errors, default=math.inf)
            if max_ee_rotation_error > policy.max_shadow_joint_error_rad:
                failures.append(
                    f"firmware/shadow FK rotation error {max_ee_rotation_error:.6g} rad exceeds "
                    f"{policy.max_shadow_joint_error_rad:.6g} rad"
                )
    start = shadow[:1] if shadow.ndim == 2 and len(shadow) else np.empty((0, 0))
    created_at = time.time() if created_at_unix_s is None else float(created_at_unix_s)
    if not math.isfinite(created_at) or created_at <= 0.0:
        failures.append("SDK simulation evidence creation time is invalid")
    return SdkSimulationEvidence(
        schema_version=SDK_SIMULATION_EVIDENCE_SCHEMA_VERSION,
        passed=not failures,
        robot_key=robot_key,
        serial_number=serial_number.strip(),
        program_sha256=program_sha256,
        config_sha256=config_sha256,
        start_q_sha256=stream_sha256(start),
        shadow_stream_sha256=stream_sha256(shadow),
        firmware_stream_sha256=stream_sha256(firmware),
        created_at_unix_s=created_at,
        samples=len(shadow) if shadow.ndim == 2 else 0,
        max_joint_error_rad=max_joint_error,
        max_ee_error_m=max_ee_error,
        max_ee_rotation_error_rad=max_ee_rotation_error,
        self_motion_allowed=self_motion_allowed,
        min_distance_m=min_distance,
        failures=tuple(failures),
    )


def load_sdk_simulation_evidence(path: str | Path) -> SdkSimulationEvidence:
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read SDK simulation evidence: {exc}") from exc
    expected = {item.name for item in fields(SdkSimulationEvidence)}
    if not isinstance(data, dict) or set(data) != expected:
        raise ValueError(f"SDK evidence fields must be exactly {sorted(expected)}")
    data["failures"] = tuple(data["failures"])
    evidence = SdkSimulationEvidence(**data)
    if evidence.schema_version != SDK_SIMULATION_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported SDK simulation evidence schema")
    return evidence
