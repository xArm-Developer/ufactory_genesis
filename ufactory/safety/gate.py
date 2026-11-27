"""Predictive, fail-closed safety preflight and approval for mixed trajectory programs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from collections.abc import Iterable

import numpy as np

from ufactory.config.models import ResolvedRuntimeConfig
from ufactory.grippers.g2 import gripper_g2_gap_m_to_sim_drive
from ufactory.grippers.robotiq import robotiq_gap_m_to_sim_drive
from ufactory.kinematics.orientation import GRIPPER_DOWN_QUAT_XYZW
from ufactory.safety.approved import ApprovedProgram, issue_approved_program
from ufactory.safety.interfaces import (
    CartesianStepController,
    CollisionBackend,
    CollisionResult,
    KinematicsBackend,
)
from ufactory.safety.models import (
    PreflightCheck,
    PreflightReport,
    SafetyViolation,
    ViolationType,
)
from ufactory.safety.statistics import compute_motion_statistics
from ufactory.safety.sdk_sim import (
    SDK_SIMULATION_EVIDENCE_MAX_AGE_S,
    SdkSimulationEvidence,
    stream_sha256,
)
from ufactory.trajectory.segments import Program, Segment
from ufactory.types import FloatArray

# Stationary-seed projection only absorbs FK disagreement between IK backends
# (measured <= 1.4e-4 rad). Larger offsets are real motion and stay in the timeline.
_STATIONARY_SEED_MAX_PROJECTION_RAD = 1e-3


@dataclass(frozen=True)
class _Timeline:
    q_rad: FloatArray
    xyz_m: FloatArray
    stages: tuple[str, ...]
    gripper_drive: FloatArray
    quaternion_xyzw: FloatArray
    approx_steps: int = 0
    max_approx_position_error_m: float = 0.0
    max_approx_orientation_error_rad: float = 0.0


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def program_sha256(program: Program) -> str:
    """Hash exact samples and metadata, independent of NumPy formatting."""

    digest = hashlib.sha256()
    header = {
        "schema_version": 1,
        "rate_hz": float(program.rate),
        "robot_key": program.robot_key,
        "metadata": program.metadata,
    }
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":"), default=str).encode())
    for segment in program.segments:
        samples, _ = segment.samples(program.rate)
        descriptor = {
            "kind": segment.kind,
            "label": segment.label,
            "duration_s": float(segment.duration),
            "v_max": float(segment.v_max),
            "a_max": float(segment.a_max),
            "shape": list(samples.shape),
        }
        digest.update(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode())
        digest.update(np.ascontiguousarray(samples, dtype="<f8").tobytes())
        for endpoint in (segment.q_start, segment.q_end, segment.pose_start, segment.pose_end):
            if endpoint is not None:
                digest.update(np.ascontiguousarray(endpoint, dtype="<f8").tobytes())
        if segment.gap_start is not None:
            digest.update(np.asarray((segment.gap_start, segment.gap_end), dtype="<f8").tobytes())
    return digest.hexdigest()


def _violation(
    kind: ViolationType,
    stage: str,
    message: str,
    *,
    sample_index: int | None = None,
    joint: str | None = None,
    link: str | None = None,
    actual: float | None = None,
    limit: float | None = None,
) -> SafetyViolation:
    return SafetyViolation(
        type=kind,
        stage=stage,
        message=message,
        sample_index=sample_index,
        joint=joint,
        link=link,
        actual=actual,
        limit=limit,
    )


class SafetyGate:
    """Run every safety rule against every command sample."""

    def __init__(
        self,
        config: ResolvedRuntimeConfig,
        *,
        joint_lower_rad: Iterable[float],
        joint_upper_rad: Iterable[float],
        kinematics: KinematicsBackend | None,
        collision: CollisionBackend | None,
        urdf_sha256: str,
        calibration_sha256: str,
        scene_sha256: str,
    ) -> None:
        self.config = config
        self.joint_lower_rad: FloatArray = np.asarray(tuple(joint_lower_rad), dtype=np.float64)
        self.joint_upper_rad: FloatArray = np.asarray(tuple(joint_upper_rad), dtype=np.float64)
        self.kinematics = kinematics
        self.collision = collision
        self.urdf_sha256 = str(urdf_sha256)
        self.calibration_sha256 = str(calibration_sha256)
        self.scene_sha256 = str(scene_sha256)
        expected = (config.robot.dof,)
        if self.joint_lower_rad.shape != expected or self.joint_upper_rad.shape != expected:
            raise ValueError(f"joint bounds must have shape {expected}")
        if not np.all(np.isfinite(self.joint_lower_rad)) or not np.all(np.isfinite(self.joint_upper_rad)):
            raise ValueError("joint bounds must be finite")
        if np.any(self.joint_lower_rad >= self.joint_upper_rad):
            raise ValueError("joint lower bounds must be below upper bounds")

    def _project_stationary_seed(
        self,
        program: Program,
        segment: Segment,
        start_q: FloatArray,
        samples: FloatArray,
    ) -> FloatArray | None:
        """Put a held joint seed on the following Cartesian start.

        Genesis and Pinocchio can disagree by a few micrometres on the same
        joint vector. Leaving that point in the finite-difference stream makes
        the first planned acceleration step look faster than the Cartesian
        command. A stationary seed is re-solved onto ``pose_start``; a segment
        that already moves is left unchanged.

        The executor still sends the original seed, so the projection is
        bounded by ``_STATIONARY_SEED_MAX_PROJECTION_RAD``. A larger offset is
        not projected and is checked as real motion.
        """

        if self.kinematics is None or segment.pose_start is not None:
            return None
        if start_q.shape != (self.config.robot.dof,) or not np.all(np.isfinite(start_q)):
            return None
        if samples.ndim != 2 or samples.shape[1] != self.config.robot.dof:
            return None
        if float(np.max(np.abs(samples - start_q))) > 1e-8:
            return None
        if segment.q_end is not None:
            end_q = np.asarray(segment.q_end, dtype=np.float64).reshape(-1)
            if end_q.shape != start_q.shape or float(np.max(np.abs(end_q - start_q))) > 1e-8:
                return None
        pose_start: FloatArray | None = None
        seen = False
        for later in program.segments:
            if not seen:
                if later is segment:
                    seen = True
                continue
            if later.kind == "movel" and later.pose_start is not None:
                pose_start = np.asarray(later.pose_start, dtype=np.float64).reshape(3)
                break
            if later.kind == "movej":
                return None
        if pose_start is None or pose_start.shape != (3,) or not np.all(np.isfinite(pose_start)):
            return None
        pose = np.concatenate((pose_start, np.asarray(GRIPPER_DOWN_QUAT_XYZW, dtype=np.float64)))
        try:
            projected = np.asarray(self.kinematics.inverse(pose, start_q), dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if projected.shape != start_q.shape or not np.all(np.isfinite(projected)):
            return None
        if float(np.max(np.abs(projected - start_q))) > _STATIONARY_SEED_MAX_PROJECTION_RAD:
            return None
        return projected

    def _build_timeline(self, program: Program) -> tuple[_Timeline | None, list[SafetyViolation]]:
        violations: list[SafetyViolation] = []
        if self.kinematics is None and self.config.safety.require_kinematics:
            violations.append(
                _violation(ViolationType.BACKEND_UNAVAILABLE, "preflight", "kinematics backend is required")
            )
            return None, violations
        step_controller = self.kinematics if isinstance(self.kinematics, CartesianStepController) else None
        if self.config.motion.allow_approx_motion:
            if step_controller is None:
                violations.append(
                    _violation(
                        ViolationType.BACKEND_UNAVAILABLE,
                        "preflight",
                        "allow_approx_motion requires a CartesianStepController kinematics backend",
                    )
                )
                return None, violations
            if not step_controller.is_approx_motion:
                violations.append(
                    _violation(
                        ViolationType.INVALID_INPUT,
                        "preflight",
                        "controller allow_approx_motion does not match runtime configuration",
                    )
                )
                return None, violations
        elif step_controller is not None and step_controller.is_approx_motion:
            violations.append(
                _violation(
                    ViolationType.INVALID_INPUT,
                    "preflight",
                    "controller allow_approx_motion is enabled while runtime configuration disables it",
                )
            )
            return None, violations
        current_q: FloatArray | None = None
        q_rows: list[FloatArray] = []
        xyz_rows: list[FloatArray] = []
        stages: list[str] = []
        drive_rows: list[float] = []
        quaternion_rows: list[FloatArray] = []
        approx_steps = 0
        max_approx_pos_err_m = 0.0
        max_approx_rot_err_rad = 0.0
        current_drive = self.config.gripper.open_drive if self.config.gripper is not None else 0.0
        sample_index = 0
        for segment in program.segments:
            stage = segment.label or segment.kind
            try:
                samples, _ = segment.samples(program.rate)
            except (AssertionError, TypeError, ValueError) as exc:
                violations.append(_violation(ViolationType.INVALID_INPUT, stage, str(exc)))
                continue
            if samples.ndim != 2 or samples.shape[0] < 1 or not np.all(np.isfinite(samples)):
                violations.append(
                    _violation(ViolationType.INVALID_INPUT, stage, "segment samples must be a finite non-empty matrix")
                )
                continue
            if segment.kind == "movej":
                if samples.shape[1] != self.config.robot.dof:
                    violations.append(
                        _violation(
                            ViolationType.INVALID_INPUT,
                            stage,
                            f"expected {self.config.robot.dof} joints, got {samples.shape[1]}",
                        )
                    )
                    continue
                q_segment = samples
                if current_q is None and segment.q_start is not None:
                    start_q = np.asarray(segment.q_start, dtype=np.float64).reshape(-1)
                    projected = self._project_stationary_seed(program, segment, start_q, samples)
                    if projected is not None:
                        start_q = projected
                        q_segment = np.repeat(projected[None, :], len(samples), axis=0)
                    if start_q.shape == (self.config.robot.dof,) and np.all(np.isfinite(start_q)):
                        q_segment = np.concatenate((start_q[None, :], q_segment), axis=0)
                drive_segment: FloatArray = np.full(len(q_segment), current_drive, dtype=np.float64)
            elif segment.kind == "movel":
                if samples.shape[1] != 3:
                    violations.append(_violation(ViolationType.INVALID_INPUT, stage, "Cartesian samples need xyz"))
                    continue
                if current_q is None:
                    violations.append(
                        _violation(ViolationType.INVALID_INPUT, stage, "MoveL requires a preceding joint state")
                    )
                    continue
                if self.kinematics is None:
                    violations.append(
                        _violation(ViolationType.BACKEND_UNAVAILABLE, stage, "MoveL requires inverse kinematics")
                    )
                    continue
                shadow: list[FloatArray] = []
                seed = current_q.copy()
                step_dt = 1.0 / float(program.rate)
                for offset, row in enumerate(samples):
                    row_index = sample_index + offset
                    try:
                        pose = np.concatenate((row, np.asarray(GRIPPER_DOWN_QUAT_XYZW, dtype=np.float64)))
                        if step_controller is not None:
                            step_result = step_controller.solve_step(pose, seed, dt_s=step_dt, include_tcp=False)
                            q = np.asarray(step_result.q_rad, dtype=np.float64).reshape(-1)
                            if bool(step_result.used_approx):
                                approx_steps += 1
                                max_approx_pos_err_m = max(max_approx_pos_err_m, float(step_result.position_error_m))
                                max_approx_rot_err_rad = max(
                                    max_approx_rot_err_rad, float(step_result.orientation_error_rad)
                                )
                        else:
                            q = np.asarray(self.kinematics.inverse(pose, seed), dtype=np.float64).reshape(-1)
                    except Exception as exc:
                        violations.append(
                            _violation(ViolationType.INVALID_INPUT, stage, f"IK failed: {exc}", sample_index=row_index)
                        )
                        break
                    if q.shape != (self.config.robot.dof,) or not np.all(np.isfinite(q)):
                        violations.append(
                            _violation(
                                ViolationType.INVALID_INPUT,
                                stage,
                                "IK returned an invalid joint vector",
                                sample_index=row_index,
                            )
                        )
                        break
                    jump = float(np.max(np.abs(q - seed)))
                    if jump > self.config.safety.max_ik_jump_rad:
                        violations.append(
                            _violation(
                                ViolationType.IK_DISCONTINUITY,
                                stage,
                                "IK branch jump exceeds policy",
                                sample_index=row_index,
                                actual=jump,
                                limit=self.config.safety.max_ik_jump_rad,
                            )
                        )
                    shadow.append(q)
                    seed = q
                if len(shadow) != len(samples):
                    continue
                q_segment = np.stack(shadow)
                drive_segment = np.full(len(q_segment), current_drive, dtype=np.float64)
            else:
                if current_q is None:
                    violations.append(
                        _violation(
                            ViolationType.INVALID_INPUT, stage, "gripper motion requires a preceding joint state"
                        )
                    )
                    continue
                q_segment = np.repeat(current_q[None, :], len(samples), axis=0)
                if self.config.gripper is None:
                    violations.append(_violation(ViolationType.INVALID_INPUT, stage, "robot has no gripper profile"))
                    continue
                gaps = samples.reshape(-1)
                closed_gap = self.config.gripper.closed_gap_m
                gap_span = self.config.gripper.open_gap_m - closed_gap
                if gap_span == 0.0:
                    if np.any(np.abs(gaps - closed_gap) > 1e-12):
                        violations.append(
                            _violation(ViolationType.INVALID_INPUT, stage, "gripper gap exceeds configured range")
                        )
                        continue
                    ratios = np.zeros_like(gaps, dtype=np.float64)
                else:
                    ratios = (gaps - closed_gap) / gap_span
                    if np.any(ratios < -1e-12) or np.any(ratios > 1.0 + 1e-12):
                        violations.append(
                            _violation(ViolationType.INVALID_INPUT, stage, "gripper gap exceeds configured range")
                        )
                        continue
                if self.config.gripper.adapter == "g2":
                    drive_segment = np.asarray([gripper_g2_gap_m_to_sim_drive(gap) for gap in gaps])
                elif self.config.gripper.adapter in {"robotiq_2f_85", "robotiq_2f_140"}:
                    variant = self.config.gripper.adapter
                    drive_segment = np.asarray([robotiq_gap_m_to_sim_drive(variant, float(gap)) for gap in gaps])
                else:
                    drive_segment = self.config.gripper.closed_drive + ratios * (
                        self.config.gripper.open_drive - self.config.gripper.closed_drive
                    )
                current_drive = float(drive_segment[-1])
            for q, drive in zip(q_segment, drive_segment, strict=True):
                if not np.all(np.isfinite(q)):
                    violations.append(
                        _violation(
                            ViolationType.INVALID_INPUT, stage, "joint sample is non-finite", sample_index=sample_index
                        )
                    )
                    sample_index += 1
                    continue
                try:
                    xyz = (
                        np.asarray(self.kinematics.forward(q), dtype=np.float64).reshape(-1)
                        if self.kinematics is not None
                        else np.full(3, np.nan)
                    )
                except Exception as exc:
                    violations.append(
                        _violation(ViolationType.INVALID_INPUT, stage, f"FK failed: {exc}", sample_index=sample_index)
                    )
                    sample_index += 1
                    continue
                if xyz.size < 3 or not np.all(np.isfinite(xyz[:3])):
                    violations.append(
                        _violation(
                            ViolationType.INVALID_INPUT, stage, "FK returned invalid xyz", sample_index=sample_index
                        )
                    )
                    sample_index += 1
                    continue
                quaternion = xyz[3:7] if xyz.size >= 7 else np.asarray((0.0, 0.0, 0.0, 1.0))
                norm = float(np.linalg.norm(quaternion))
                if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)) or norm < 1e-12:
                    violations.append(
                        _violation(
                            ViolationType.INVALID_INPUT,
                            stage,
                            "FK returned invalid orientation quaternion",
                            sample_index=sample_index,
                        )
                    )
                    sample_index += 1
                    continue
                quaternion = quaternion / norm
                q_rows.append(q.copy())
                xyz_rows.append(xyz[:3].copy())
                stages.append(stage)
                drive_rows.append(float(drive))
                quaternion_rows.append(quaternion)
                current_q = q
                sample_index += 1
        if not q_rows:
            violations.append(
                _violation(ViolationType.INVALID_INPUT, "program", "program has no valid command samples")
            )
            return None, violations
        return _Timeline(
            np.stack(q_rows),
            np.stack(xyz_rows),
            tuple(stages),
            np.asarray(drive_rows, dtype=np.float64),
            np.stack(quaternion_rows),
            approx_steps=approx_steps,
            max_approx_position_error_m=max_approx_pos_err_m,
            max_approx_orientation_error_rad=max_approx_rot_err_rad,
        ), violations

    def _allowed_contact(self, result: CollisionResult, stage: str) -> bool:
        if frozenset((result.link_a, result.link_b)) in {
            frozenset(pair) for pair in self.config.robot.environment_contact_pairs
        }:
            return True
        declared = self.config.task.allowed_contacts.get(stage, ())
        if "finger_links_to_object" in declared and self.config.gripper is not None:
            pair = {result.link_a, result.link_b}
            if "object" in pair and pair.intersection(self.config.gripper.allowed_contact_links):
                return True
        if "object_to_table" in declared and {result.link_a, result.link_b} == {"object", "table"}:
            return True
        return False

    def collision_allowed(self, result: CollisionResult, stage: str) -> bool:
        """Return whether a task contact or versioned exemption permits a pair."""

        return self._allowed_contact(result, stage) or self._exempt(result, stage)

    def _collision_results_for_margin(
        self,
        q_rad: FloatArray,
        *,
        stage: str,
        gripper_drive: float | None,
    ) -> tuple[CollisionResult, ...]:
        """Query all unsafe candidates, preferring a backend margin capability."""

        if self.collision is None:
            return ()
        effective_drive = (
            gripper_drive if self.config.gripper is not None and self.config.gripper.has_actuated_joints else None
        )
        within_margin = getattr(self.collision, "check_all_within_margin", None)
        if callable(within_margin):
            return tuple(
                within_margin(
                    q_rad,
                    stage=stage,
                    gripper_drive=effective_drive,
                    margin_m=self.config.safety.min_collision_distance_m,
                )
            )
        check_all = getattr(self.collision, "check_all", None)
        if callable(check_all):
            return tuple(check_all(q_rad, stage=stage, gripper_drive=effective_drive))
        return (self.collision.check(q_rad, stage=stage, gripper_drive=effective_drive),)

    def _collision_query_mode(self) -> str:
        if self.collision is None:
            return "unavailable"
        if callable(getattr(self.collision, "check_all_within_margin", None)):
            return "security-margin-candidates"
        return "full-distance-fallback"

    def _exempt(self, result: CollisionResult, stage: str) -> bool:
        return any(
            exemption.matches(result.link_a, result.link_b, stage, self.urdf_sha256)
            for exemption in self.config.safety.exemptions
        )

    def preflight(self, program: Program, *, executor: str) -> PreflightReport:
        started = time.perf_counter_ns()
        violations: list[SafetyViolation] = []
        checks: list[PreflightCheck] = []
        if executor not in {"servo_j", "servo_cartesian"}:
            violations.append(_violation(ViolationType.INVALID_INPUT, "program", f"unknown executor: {executor}"))
        if program.robot_key and program.robot_key != self.config.robot.key:
            violations.append(
                _violation(
                    ViolationType.IDENTITY,
                    "program",
                    f"program robot {program.robot_key} does not match {self.config.robot.key}",
                )
            )
        if not math.isfinite(float(program.rate)) or float(program.rate) <= 0.0:
            violations.append(_violation(ViolationType.INVALID_INPUT, "program", "rate must be finite and positive"))
        elif not math.isclose(float(program.rate), self.config.motion.rate_hz, rel_tol=0.0, abs_tol=1e-9):
            violations.append(
                _violation(
                    ViolationType.TIMING,
                    "program",
                    "program rate differs from approved runtime configuration",
                    actual=float(program.rate),
                    limit=self.config.motion.rate_hz,
                )
            )
        build_started = time.perf_counter_ns()
        timeline, build_violations = self._build_timeline(program)
        violations.extend(build_violations)
        warnings: list[str] = []
        timeline_details = {
            "allow_approx_motion": bool(self.config.motion.allow_approx_motion),
            "approx_steps": 0 if timeline is None else timeline.approx_steps,
            "max_approx_position_error_m": 0.0 if timeline is None else timeline.max_approx_position_error_m,
            "max_approx_orientation_error_rad": 0.0 if timeline is None else timeline.max_approx_orientation_error_rad,
        }
        if timeline is not None and timeline.approx_steps > 0:
            warnings.append(
                f"allow_approx_motion activated on {timeline.approx_steps} sample(s) "
                f"(max_pos_err={timeline.max_approx_position_error_m * 1e3:.3f} mm, "
                f"max_rot_err={math.degrees(timeline.max_approx_orientation_error_rad):.3f} deg)"
            )
        checks.append(
            PreflightCheck(
                "timeline",
                timeline is not None and not build_violations,
                0 if timeline is None else len(timeline.q_rad),
                (time.perf_counter_ns() - build_started) / 1e6,
                timeline_details,
            )
        )
        if timeline is not None:
            bounds_started = time.perf_counter_ns()
            lower = self.joint_lower_rad + self.config.safety.joint_limit_margin_rad
            upper = self.joint_upper_rad - self.config.safety.joint_limit_margin_rad
            before = len(violations)
            for index, q in enumerate(timeline.q_rad):
                for joint_index, value in enumerate(q):
                    if value < lower[joint_index] or value > upper[joint_index]:
                        limit = lower[joint_index] if value < lower[joint_index] else upper[joint_index]
                        violations.append(
                            _violation(
                                ViolationType.JOINT_LIMIT,
                                timeline.stages[index],
                                "joint position exceeds margin-adjusted limit",
                                sample_index=index,
                                joint=self.config.robot.joint_names[joint_index],
                                actual=float(value),
                                limit=float(limit),
                            )
                        )
            checks.append(
                PreflightCheck(
                    "joint_limits",
                    len(violations) == before,
                    len(timeline.q_rad),
                    (time.perf_counter_ns() - bounds_started) / 1e6,
                )
            )
            orientation_started = time.perf_counter_ns()
            before = len(violations)
            if len(timeline.quaternion_xyzw) >= 2:
                dots = np.sum(timeline.quaternion_xyzw[:-1] * timeline.quaternion_xyzw[1:], axis=1)
                steps = 2.0 * np.arccos(np.clip(np.abs(dots), 0.0, 1.0))
                bad = np.flatnonzero(steps > self.config.safety.max_orientation_step_rad)
                for raw_index in bad:
                    index = int(raw_index)
                    violations.append(
                        _violation(
                            ViolationType.ORIENTATION,
                            timeline.stages[index + 1],
                            "orientation step exceeds configured limit",
                            sample_index=index + 1,
                            actual=float(steps[index]),
                            limit=self.config.safety.max_orientation_step_rad,
                        )
                    )
            checks.append(
                PreflightCheck(
                    "orientation",
                    len(violations) == before,
                    len(timeline.q_rad),
                    (time.perf_counter_ns() - orientation_started) / 1e6,
                )
            )
            workspace_started = time.perf_counter_ns()
            before = len(violations)
            lo = np.asarray(self.config.safety.workspace_lower_m)
            hi = np.asarray(self.config.safety.workspace_upper_m)
            for index, xyz in enumerate(timeline.xyz_m):
                if xyz[2] < self.config.safety.z_min_m:
                    violations.append(
                        _violation(
                            ViolationType.Z_MIN,
                            timeline.stages[index],
                            "end effector is below z minimum",
                            sample_index=index,
                            actual=float(xyz[2]),
                            limit=self.config.safety.z_min_m,
                        )
                    )
                if np.any(xyz < lo) or np.any(xyz > hi):
                    violations.append(
                        _violation(
                            ViolationType.WORKSPACE,
                            timeline.stages[index],
                            f"end effector is outside workspace: {xyz.tolist()}",
                            sample_index=index,
                        )
                    )
            checks.append(
                PreflightCheck(
                    "workspace",
                    len(violations) == before,
                    len(timeline.q_rad),
                    (time.perf_counter_ns() - workspace_started) / 1e6,
                )
            )
            motion_started = time.perf_counter_ns()
            before = len(violations)
            try:
                stats = compute_motion_statistics(timeline.q_rad, timeline.xyz_m, rate_hz=program.rate)
                limits: tuple[tuple[ViolationType, str, float, float], ...]
                if executor == "servo_j":
                    # servo_j streams joints (often Genesis-IK + joint LSPB retime).
                    # Cartesian speed/accel were enforced on the source MoveL plan;
                    # Pinocchio FK of the retimed joint polyline is not the same curve.
                    limits = (
                        (
                            ViolationType.VELOCITY,
                            "joint velocity",
                            stats.max_joint_speed_rad_s,
                            self.config.motion.joint_speed_rad_s,
                        ),
                        (
                            ViolationType.ACCELERATION,
                            "joint acceleration",
                            stats.max_joint_acceleration_rad_s2,
                            self.config.motion.joint_acceleration_rad_s2,
                        ),
                    )
                else:
                    limits = (
                        (
                            ViolationType.VELOCITY,
                            "joint velocity",
                            stats.max_joint_speed_rad_s,
                            self.config.motion.joint_speed_rad_s,
                        ),
                        (
                            ViolationType.ACCELERATION,
                            "joint acceleration",
                            stats.max_joint_acceleration_rad_s2,
                            self.config.motion.joint_acceleration_rad_s2,
                        ),
                        (
                            ViolationType.VELOCITY,
                            "Cartesian velocity",
                            stats.max_cartesian_speed_m_s,
                            self.config.motion.cartesian_speed_m_s,
                        ),
                        (
                            ViolationType.ACCELERATION,
                            "Cartesian acceleration",
                            stats.max_cartesian_acceleration_m_s2,
                            self.config.motion.cartesian_acceleration_m_s2,
                        ),
                    )
                for kind, label, actual, limit in limits:
                    # FK/IK backends have finite convergence tolerance.  The
                    # 0.1% comparison band only absorbs numerical projection
                    # noise; planned samples themselves remain strictly bound.
                    if actual > limit * 1.001 + 1e-12:
                        violations.append(
                            _violation(kind, "program", f"{label} exceeds configured limit", actual=actual, limit=limit)
                        )
                details = stats.__dict__
            except ValueError as exc:
                violations.append(_violation(ViolationType.INVALID_INPUT, "program", str(exc)))
                details = {}
            checks.append(
                PreflightCheck(
                    "motion_limits",
                    len(violations) == before,
                    len(timeline.q_rad),
                    (time.perf_counter_ns() - motion_started) / 1e6,
                    details,
                )
            )
            collision_started = time.perf_counter_ns()
            before = len(violations)
            collision_query_mode = self._collision_query_mode()
            returned_pair_samples = 0
            pair_count = getattr(self.collision, "collision_pair_count", None)
            potential_pair_samples = (
                int(pair_count) * len(timeline.q_rad) if isinstance(pair_count, int) and pair_count >= 0 else None
            )
            if self.collision is None:
                if self.config.safety.require_collision:
                    violations.append(
                        _violation(ViolationType.BACKEND_UNAVAILABLE, "preflight", "collision backend is required")
                    )
            else:
                for index, q in enumerate(timeline.q_rad):
                    stage = timeline.stages[index]
                    try:
                        results = self._collision_results_for_margin(
                            q,
                            stage=stage,
                            gripper_drive=float(timeline.gripper_drive[index]),
                        )
                        returned_pair_samples += len(results)
                    except Exception as exc:
                        violations.append(
                            _violation(
                                ViolationType.BACKEND_UNAVAILABLE,
                                stage,
                                f"collision backend failed: {exc}",
                                sample_index=index,
                            )
                        )
                        continue
                    for result in results:
                        if not math.isfinite(result.min_distance_m):
                            violations.append(
                                _violation(
                                    ViolationType.INVALID_INPUT,
                                    stage,
                                    "collision backend returned non-finite distance",
                                    sample_index=index,
                                )
                            )
                            continue
                        unsafe = result.colliding or result.min_distance_m < self.config.safety.min_collision_distance_m
                        if unsafe and not self._allowed_contact(result, stage) and not self._exempt(result, stage):
                            kind = (
                                ViolationType.ENVIRONMENT_COLLISION
                                if result.environment
                                else ViolationType.SELF_COLLISION
                            )
                            if not result.colliding:
                                kind = ViolationType.CLEARANCE
                            violations.append(
                                _violation(
                                    kind,
                                    stage,
                                    f"unsafe link pair {result.link_a}/{result.link_b}",
                                    sample_index=index,
                                    link=f"{result.link_a}/{result.link_b}",
                                    actual=float(result.min_distance_m),
                                    limit=self.config.safety.min_collision_distance_m,
                                )
                            )
            checks.append(
                PreflightCheck(
                    "collision",
                    len(violations) == before,
                    len(timeline.q_rad),
                    (time.perf_counter_ns() - collision_started) / 1e6,
                    {
                        "query_mode": collision_query_mode,
                        "margin_m": self.config.safety.min_collision_distance_m,
                        "returned_pair_samples": returned_pair_samples,
                        **(
                            {"potential_pair_samples": potential_pair_samples}
                            if potential_pair_samples is not None
                            else {}
                        ),
                    },
                )
            )
        digest = program_sha256(program)
        checks.append(
            PreflightCheck(
                "total",
                not violations,
                0 if timeline is None else len(timeline.q_rad),
                (time.perf_counter_ns() - started) / 1e6,
            )
        )
        return PreflightReport(
            schema_version=1,
            passed=not violations and all(check.passed for check in checks),
            program_sha256=digest,
            robot_key=self.config.robot.key,
            executor=executor,
            config_sha256=self.config.sha256,
            urdf_sha256=self.urdf_sha256,
            calibration_sha256=self.calibration_sha256,
            scene_sha256=self.scene_sha256,
            checks=tuple(checks),
            violations=tuple(violations),
            warnings=tuple(warnings),
            exemption_ids=tuple(
                f"{item.link_a}/{item.link_b}:{item.stage}:{item.expires_version}"
                for item in self.config.safety.exemptions
            ),
        )

    def shadow_joint_stream(self, program: Program) -> tuple[FloatArray, tuple[str, ...]]:
        """Return the exact host IK joint stream used by this SafetyGate."""

        timeline, violations = self._build_timeline(program)
        if timeline is None or violations:
            messages = "; ".join(item.message for item in violations)
            raise ValueError(f"cannot build shadow joint stream: {messages}")
        return timeline.q_rad.copy(), timeline.stages

    def runtime_monitor(
        self,
        q_rad: FloatArray,
        stage: str,
        *,
        gripper_drive: float | None = None,
        include_collision: bool = True,
    ) -> SafetyViolation | None:
        """Fail-closed feedback check suitable for ``execute_real``.

        Full Coal/Pinocchio collision checks are typically tens of milliseconds and
        cannot meet a 50 Hz servo budget. Callers that already preflighted the
        approved timeline may pass ``include_collision=False`` for stream ticks.
        """

        q = np.asarray(q_rad, dtype=np.float64).reshape(-1)
        if q.shape != (self.config.robot.dof,) or not np.all(np.isfinite(q)):
            return _violation(ViolationType.FEEDBACK, stage, "feedback joint vector is invalid")
        lower = self.joint_lower_rad + self.config.safety.joint_limit_margin_rad
        upper = self.joint_upper_rad - self.config.safety.joint_limit_margin_rad
        if np.any(q < lower) or np.any(q > upper):
            index = int(np.flatnonzero((q < lower) | (q > upper))[0])
            return _violation(
                ViolationType.JOINT_LIMIT,
                stage,
                "feedback exceeds joint limit",
                joint=self.config.robot.joint_names[index],
                actual=float(q[index]),
                limit=float(lower[index] if q[index] < lower[index] else upper[index]),
            )
        if self.kinematics is None:
            return _violation(ViolationType.BACKEND_UNAVAILABLE, stage, "runtime safety backend unavailable")
        try:
            xyz = np.asarray(self.kinematics.forward(q), dtype=np.float64).reshape(-1)[:3]
            if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
                raise ValueError("FK returned invalid xyz")
            lo, hi = np.asarray(self.config.safety.workspace_lower_m), np.asarray(self.config.safety.workspace_upper_m)
            if xyz[2] < self.config.safety.z_min_m or np.any(xyz < lo) or np.any(xyz > hi):
                return _violation(ViolationType.WORKSPACE, stage, "feedback FK is outside safe workspace")
        except Exception as exc:
            return _violation(ViolationType.BACKEND_UNAVAILABLE, stage, f"runtime safety check failed: {exc}")
        if not include_collision:
            return None
        if self.collision is None:
            return _violation(ViolationType.BACKEND_UNAVAILABLE, stage, "runtime safety backend unavailable")
        try:
            results = self._collision_results_for_margin(
                q,
                stage=stage,
                gripper_drive=gripper_drive,
            )
        except Exception as exc:
            return _violation(ViolationType.BACKEND_UNAVAILABLE, stage, f"runtime safety check failed: {exc}")
        for result in results:
            unsafe = result.colliding or result.min_distance_m < self.config.safety.min_collision_distance_m
            if unsafe and not self.collision_allowed(result, stage):
                return _violation(
                    ViolationType.ENVIRONMENT_COLLISION if result.environment else ViolationType.SELF_COLLISION,
                    stage,
                    f"runtime collision/clearance violation {result.link_a}/{result.link_b}",
                    link=f"{result.link_a}/{result.link_b}",
                    actual=float(result.min_distance_m),
                    limit=self.config.safety.min_collision_distance_m,
                )
        return None

    def approve(
        self,
        program: Program,
        report: PreflightReport,
        *,
        expected_serial_number: str,
        sdk_evidence: SdkSimulationEvidence | None = None,
    ) -> ApprovedProgram:
        """Issue an opaque approval after rechecking all binding hashes."""

        expected = {
            "program": (report.program_sha256, program_sha256(program)),
            "config": (report.config_sha256, self.config.sha256),
            "urdf": (report.urdf_sha256, self.urdf_sha256),
            "calibration": (report.calibration_sha256, self.calibration_sha256),
            "scene": (report.scene_sha256, self.scene_sha256),
        }
        mismatches = [name for name, (actual, current) in expected.items() if actual != current]
        if mismatches:
            raise ValueError(f"cannot approve: changed bindings: {', '.join(mismatches)}")
        if report.robot_key != self.config.robot.key or not report.passed:
            raise ValueError("cannot approve a failed or cross-robot preflight")
        if not expected_serial_number.strip():
            raise ValueError("approval must bind a complete robot serial number")
        if report.executor == "servo_cartesian":
            if sdk_evidence is None or not sdk_evidence.passed:
                raise ValueError("servo_cartesian approval requires passing current SDK simulation evidence")
            evidence_age_s = time.time() - float(sdk_evidence.created_at_unix_s)
            if (
                not math.isfinite(evidence_age_s)
                or evidence_age_s < -1.0
                or evidence_age_s > SDK_SIMULATION_EVIDENCE_MAX_AGE_S
            ):
                raise ValueError("SDK simulation evidence is expired or has an invalid creation time")
            shadow, _ = self.shadow_joint_stream(program)
            evidence_expected = {
                "robot": (sdk_evidence.robot_key, self.config.robot.key),
                "serial": (sdk_evidence.serial_number, expected_serial_number.strip()),
                "program": (sdk_evidence.program_sha256, report.program_sha256),
                "config": (sdk_evidence.config_sha256, self.config.sha256),
                "shadow": (sdk_evidence.shadow_stream_sha256, stream_sha256(shadow)),
            }
            stale = [name for name, (actual, current) in evidence_expected.items() if actual != current]
            if stale:
                raise ValueError(f"SDK simulation evidence is stale or mismatched: {', '.join(stale)}")
        return issue_approved_program(program, report, expected_serial_number.strip())

    def approve_simulation(self, program: Program, report: PreflightReport) -> ApprovedProgram:
        """Issue a hash-bound approval for offline simulation only.

        Firmware evidence remains mandatory in :meth:`approve` for hardware
        ``servo_cartesian`` execution. The synthetic serial prevents this
        approval from matching a physical transport binding.
        """

        expected = {
            "program": (report.program_sha256, program_sha256(program)),
            "config": (report.config_sha256, self.config.sha256),
            "urdf": (report.urdf_sha256, self.urdf_sha256),
            "calibration": (report.calibration_sha256, self.calibration_sha256),
            "scene": (report.scene_sha256, self.scene_sha256),
        }
        mismatches = [name for name, (actual, current) in expected.items() if actual != current]
        if mismatches:
            raise ValueError(f"cannot approve simulation: changed bindings: {', '.join(mismatches)}")
        if report.robot_key != self.config.robot.key or not report.passed:
            raise ValueError("cannot approve a failed or cross-robot simulation preflight")
        return issue_approved_program(program, report, f"SIMULATED-{self.config.robot.key}")
