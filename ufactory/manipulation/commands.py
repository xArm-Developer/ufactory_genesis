"""Shared model, report and hardware operations for manipulation commands."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Sequence

import numpy as np

from ufactory.config import RepositoryAssetStore
from ufactory.control import ArmMotionController
from ufactory.grippers import create_gripper_adapter
from ufactory.hardware import XArmTransport
from ufactory.hardware.xarm import verify_controller_robot, wait_for_servo_motion_ready
from ufactory.kinematics import build_calibrated_urdf, load_kinematics_calibration
from ufactory.robots.assembly import resolve_robot_urdf
from ufactory.kinematics.orientation import GRIPPER_DOWN_RPY_RAD
from ufactory.kinematics.tcp_offset import pose_flange_to_tcp, read_tcp_offset
from ufactory.safety.adapters import EnvironmentObstacle, PinocchioCollisionBackend
from ufactory.safety.adapters.pinocchio import StageAwareObjectCollisionBackend
from ufactory.safety.gate import sha256_file


def build_arm_controller(
    config: Any,
    urdf: Path,
    *,
    tcp_offset: Sequence[float] | np.ndarray | None = None,
) -> ArmMotionController:
    """Construct the unified pure-NumPy ArmMotionController for runtime config + URDF."""

    return ArmMotionController.from_robot(
        config.robot.key,
        urdf_path=urdf,
        tcp_offset_mm_rpy=tcp_offset,
        servo_rate_hz=float(config.motion.rate_hz),
        allow_approx_motion=bool(getattr(config.motion, "allow_approx_motion", False)),
    )


def build_manipulation_backends(
    config: Any,
    urdf: Path,
    *,
    obstacles: Sequence[EnvironmentObstacle],
    spawn_center_m: Sequence[float],
    place_center_m: Sequence[float],
    relocated_stages: Sequence[str],
    object_width_m: float,
    calibrated_g2: bool = False,
) -> tuple[Any, Any]:
    passive = (
        {config.gripper.drive_joint: config.gripper.open_drive}
        if config.gripper is not None and config.gripper.drive_joint
        else {}
    )
    kinematics = build_arm_controller(config, urdf)
    collision: Any = StageAwareObjectCollisionBackend(
        PinocchioCollisionBackend(
            urdf,
            joint_names=config.robot.joint_names,
            ee_link=config.robot.ee_link,
            passive_joint_positions=passive,
            obstacles=tuple(obstacles),
            tool_root_link=config.robot.ee_link,
        ),
        spawn_center_m=tuple(map(float, spawn_center_m)),
        place_center_m=tuple(map(float, place_center_m)),
        relocated_stages=tuple(relocated_stages),
    )
    if calibrated_g2 and config.gripper is not None:
        if config.gripper.adapter == "g2":
            from ufactory.simulation.g2_adapter import G2ContactCollisionBackend

            collision = G2ContactCollisionBackend(collision, object_width_m=float(object_width_m))
        elif config.gripper.adapter in {"robotiq_2f_85", "robotiq_2f_140"}:
            from ufactory.simulation.robotiq_adapter import RobotiqContactCollisionBackend

            collision = RobotiqContactCollisionBackend(
                collision,
                variant=config.gripper.adapter,
                object_width_m=float(object_width_m),
            )
    return kinematics, collision


def build_packaging_backends(config: Any, urdf: Path, *, calibrated_g2: bool = False) -> tuple[Any, Any]:
    from ufactory.manipulation.packaging.core import (
        OBJECT_RELOCATED_STAGES,
        packaging_layout,
        packaging_obstacles,
    )

    layout = packaging_layout(config)
    return build_manipulation_backends(
        config,
        urdf,
        obstacles=packaging_obstacles(layout),
        spawn_center_m=layout.object_position_m,
        place_center_m=layout.target_position_m,
        relocated_stages=OBJECT_RELOCATED_STAGES,
        object_width_m=layout.object_size_m[1],
        calibrated_g2=calibrated_g2,
    )


def paced_start_hold(mirror: Any, hold_s: float) -> None:
    if hold_s <= 0.0:
        return
    tick_s = 1.0 / float(mirror.rate)
    steps = max(1, int(round(float(hold_s) / tick_s)))
    deadline = time.monotonic()
    for _ in range(steps):
        mirror.hold_step()
        deadline += tick_s
        sleep_s = deadline - time.monotonic()
        if sleep_s > 0.0:
            time.sleep(sleep_s)


def prime_first_movej_sample(
    transport: XArmTransport,
    program: Any,
    *,
    executor: str,
    missing_message: str = "approved servo_j program has no MoveJ sample to prime MODE_SERVO",
) -> None:
    first_joint = None
    for segment in program.segments:
        if segment.kind != "movej":
            continue
        samples, _ = segment.samples(program.rate)
        if len(samples) == 0:
            continue
        first_joint = np.asarray(samples[0], dtype=np.float64)
        break
    if first_joint is not None:
        transport.prime_joint_stream(first_joint)
    elif executor == "servo_j":
        raise RuntimeError(missing_message)


def prepare_real_gripper_sender(
    arm: Any,
    transport: XArmTransport,
    gripper_config: Any,
    *,
    hardware_version: int = 1,
) -> tuple[list[float], Callable[[float], int] | None]:
    gripper_adapter = create_gripper_adapter(gripper_config, hardware_version=hardware_version)
    if gripper_adapter.capabilities.real_command:
        prepare_code = gripper_adapter.prepare_real(arm)
        if prepare_code != 0:
            transport.stop()
            raise RuntimeError(f"gripper preparation failed with SDK code {prepare_code}; controller stopped")
    current_drive = [float(gripper_config.open_drive)]

    def send_gripper(gap_m: float) -> int:
        current_drive[0] = gripper_adapter.gap_to_drive(gap_m)
        return gripper_adapter.send_real_gap(arm, gap_m)

    gripper_sender = send_gripper if gripper_adapter.capabilities.real_command else None
    return current_drive, gripper_sender


def shutdown_real_gripper(
    arm: Any,
    gripper_config: Any,
    *,
    hardware_version: int = 1,
) -> int:
    """Safely release stateless real end effectors (e.g. pneumatic vacuum suction) on exit."""
    if gripper_config is None or not bool(getattr(gripper_config, "real_command", False)):
        return 0
    adapter = create_gripper_adapter(gripper_config, hardware_version=hardware_version)
    shutdown = getattr(adapter, "shutdown_real", None)
    if callable(shutdown):
        return int(shutdown(arm))
    return 0


def print_real_execution_summary(execution: Any) -> None:
    print(f"real_execution={'PASS' if execution.completed else 'FAIL'} sent_samples={execution.sent_samples}")
    if execution.fault is not None:
        print(
            f"fault={execution.fault.fault_class}: {execution.fault.message} "
            f"sample_index={execution.fault.sample_index}"
        )
    print(
        f"timing_minor_misses={execution.minor_lateness_count} "
        f"timing_max_lateness_ms={execution.max_lateness_ns / 1_000_000.0:.3f}"
    )


def print_sdk_sim_failures(evidence: Any, evidence_path: Path | None) -> None:
    """Print every SDK simulation failure reason so ``sdk_sim=FAIL`` is actionable."""

    print(
        f"sdk_sim_failures={len(evidence.failures)} max_joint_error_rad={float(evidence.max_joint_error_rad):.6g} "
        f"max_ee_error_m={float(evidence.max_ee_error_m):.6g} "
        f"max_ee_rotation_error_rad={float(evidence.max_ee_rotation_error_rad):.6g} "
        f"self_motion_allowed={bool(evidence.self_motion_allowed)}",
        file=sys.stderr,
    )
    for failure in evidence.failures:
        print(f"  {failure}", file=sys.stderr)
    if evidence_path is not None:
        print(f"  evidence: {evidence_path}", file=sys.stderr)


def model_and_hashes(config: Any, calibration_path: Path | None, serial_number: str | None) -> tuple[Path, str, str]:
    store = RepositoryAssetStore.discover()
    base = resolve_robot_urdf(config.robot.key, config.robot.urdf, assets_dir=config.robot.assets_dir, store=store)
    if calibration_path is None:
        calibration_hash = hashlib.sha256(f"NOMINAL:{config.robot.key}".encode()).hexdigest()
        return base, sha256_file(base), calibration_hash
    calibration = load_kinematics_calibration(
        str(calibration_path),
        robot_key=config.robot.key,
        serial_number=serial_number,
        joint_names=config.robot.joint_names,
    )
    output = Path(
        build_calibrated_urdf(
            str(base),
            calibration,
            suffix=serial_number or "offline",
            joint_count=config.robot.dof,
        )
    )
    return output, sha256_file(output), calibration.sha256


def write_json(path: Path | None, value: Any) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def connect(ip: str) -> Any:
    from xarm.wrapper import XArmAPI

    arm = XArmAPI(ip, is_radian=True)
    if not arm.connected:
        connect = getattr(arm, "connect", None)
        if callable(connect):
            connect()
    if not arm.connected:
        raise RuntimeError(f"cannot connect to {ip}")
    return arm


def read_q(arm: Any, dof: int) -> np.ndarray:
    code, values = arm.get_servo_angle(is_radian=True)
    q = np.asarray(values, dtype=np.float64).reshape(-1)[:dof]
    if code != 0 or q.shape != (dof,) or not np.all(np.isfinite(q)):
        raise RuntimeError(f"invalid joint feedback (SDK code={code})")
    return q


def ensure_program_start(
    transport: XArmTransport,
    q_start: np.ndarray,
    *,
    allow_preposition: bool,
    speed_rad_s: float = 0.35,
    mvacc_rad_s2: float = 2.0,
    tolerance_rad: float = 0.02,
) -> np.ndarray:
    """Require the arm at the planned joint start; optionally preposition with set_servo_angle."""

    current = transport.read_state().q_rad
    target = np.asarray(q_start, dtype=np.float64).reshape(-1)
    if current.size < target.size or not np.all(np.isfinite(current[: target.size])):
        raise RuntimeError("invalid joint feedback before program start")
    err = float(np.max(np.abs(current[: target.size] - target)))
    if err <= tolerance_rad:
        print(f"  [preposition] already at program start max_err={err:.4f}rad")
        return current[: target.size].copy()
    if not allow_preposition:
        raise RuntimeError(
            f"arm is {err:.4f} rad from program start (limit {tolerance_rad:.4f} rad); "
            "move there manually or pass --confirm-real to allow MODE_POSITION set_servo_angle preposition"
        )
    reached_err = transport.preposition_joints(
        target,
        speed_rad_s=speed_rad_s,
        mvacc_rad_s2=mvacc_rad_s2,
        tolerance_rad=tolerance_rad,
    )
    print(f"  [preposition] mode=0 joint target reached max_err={reached_err:.4f}rad")
    return transport.read_state().q_rad[: target.size].copy()


def ensure_cartesian_program_start(
    transport: XArmTransport,
    xyz_m: np.ndarray,
    *,
    allow_preposition: bool,
    speed_mm_s: float = 100.0,
    mvacc_mm_s2: float = 500.0,
    tolerance_mm: float = 2.0,
) -> None:
    """Require the arm at Cartesian home (gripper-down); optionally preposition with set_position.

    ``xyz_m`` is the flange position; it is compared with the controller TCP
    feedback after applying ``transport.tcp_offset``.
    """

    code, reported = transport.arm.get_position(is_radian=True)
    target = np.asarray(xyz_m, dtype=np.float64).reshape(3)
    if int(code) != 0:
        raise RuntimeError(f"invalid Cartesian feedback before program start (SDK code={code})")
    expected_tcp = np.asarray(transport.flange_to_command_pose(target)[:3], dtype=np.float64)
    reported_arr = np.asarray(reported, dtype=np.float64).reshape(-1)
    xyz_err = float(np.linalg.norm(reported_arr[:3] - expected_tcp))
    if xyz_err <= tolerance_mm:
        print(f"  [preposition] already at Cartesian home xyz_err={xyz_err:.2f}mm")
        return
    if not allow_preposition:
        raise RuntimeError(
            f"arm is {xyz_err:.2f} mm from Cartesian home (limit {tolerance_mm:.2f} mm); "
            "move there manually or pass --confirm-real to allow MODE_POSITION set_position preposition"
        )
    xyz_err, rpy_err = transport.preposition_cartesian(
        target,
        speed_mm_s=speed_mm_s,
        mvacc_mm_s2=mvacc_mm_s2,
        tolerance_mm=tolerance_mm,
    )
    print(f"  [preposition] mode=0 Cartesian home reached xyz_err={xyz_err:.2f}mm rpy_err={rpy_err:.4f}rad")


def sdk_sim_feedback(
    arm: Any,
    program: Any,
    config: Any,
    expected_start_q: np.ndarray,
    *,
    tcp_offset: Sequence[float] | np.ndarray | None = None,
    allow_approx_motion: bool | None = None,
) -> np.ndarray:
    """Replay against the controller's virtual robot from the approved start.

    The virtual preposition happens only after SDK simulation mode is enabled.
    This keeps ``sdk-sim`` independent of the physical arm pose and prevents a
    validation-only command from moving the real robot. The controller is
    always returned to non-simulation mode (and its original
    ``allow_approx_motion`` setting restored), including on replay failures.

    MoveL samples are flange poses; they are converted to the controller TCP
    frame with ``tcp_offset`` (read from the controller when omitted).
    """

    target = np.asarray(expected_start_q, dtype=np.float64).reshape(-1)
    if target.shape != (config.robot.dof,) or not np.all(np.isfinite(target)):
        raise ValueError(f"SDK simulation start must contain {config.robot.dof} finite joint values")
    offset = read_tcp_offset(arm) if tcp_offset is None else np.asarray(tcp_offset, dtype=np.float64).reshape(-1)
    if offset.shape != (6,) or not np.all(np.isfinite(offset)):
        raise ValueError("tcp_offset must contain 6 finite values")
    approx_flag = (
        bool(getattr(getattr(config, "motion", None), "allow_approx_motion", False))
        if allow_approx_motion is None
        else bool(allow_approx_motion)
    )
    has_approx_api = callable(getattr(arm, "get_allow_approx_motion", None)) and callable(
        getattr(arm, "set_allow_approx_motion", None)
    )
    if approx_flag and not has_approx_api:
        raise RuntimeError("SDK simulation allow_approx_motion=True requires controller get/set_allow_approx_motion")

    def require_ok(label: str, code: int) -> None:
        if int(code) != 0:
            raise RuntimeError(f"SDK simulation {label} failed with code {code}")

    simulation_requested = bool(getattr(arm, "is_simulation_robot", False))
    approx_prev: bool | None = None
    approx_changed = False
    try:
        if has_approx_api:
            read_code, prev_val = arm.get_allow_approx_motion()
            require_ok("get_allow_approx_motion", int(read_code))
            approx_prev = bool(prev_val)
        if not simulation_requested:
            # The SDK deliberately suppresses motion_enable once its report
            # stream says simulation mode is active. Prepare readiness first,
            # then switch modes and wait for that report bit to arrive.
            require_ok("motion_enable", arm.motion_enable(enable=True))
            require_ok("set_mode(position)", arm.set_mode(0))
            require_ok("set_state", arm.set_state(0))
            require_ok("enable mode", arm.set_simulation_robot(True))
            simulation_requested = True
        deadline = time.monotonic() + 2.0
        while not bool(getattr(arm, "is_simulation_robot", False)) and time.monotonic() < deadline:
            time.sleep(0.05)
        if not bool(getattr(arm, "is_simulation_robot", False)):
            raise RuntimeError("SDK simulation mode was not confirmed by the controller report stream")
        if has_approx_api:
            require_ok("set_allow_approx_motion", arm.set_allow_approx_motion(approx_flag))
            approx_changed = True

        require_ok("set_mode(position)", arm.set_mode(0))
        require_ok("set_state", arm.set_state(0))
        require_ok(
            "virtual preposition",
            arm.set_servo_angle(
                angle=target.tolist(),
                speed=0.35,
                mvacc=2.0,
                wait=True,
                is_radian=True,
            ),
        )
        virtual_start = read_q(arm, config.robot.dof)
        start_error = float(np.max(np.abs(virtual_start - target)))
        if start_error > 0.02:
            raise RuntimeError(f"SDK simulation virtual start error {start_error:.4f} rad exceeds 0.0200 rad")
        require_ok("set_mode(servo)", arm.set_mode(1))
        require_ok("set_state", arm.set_state(0))
        wait_for_servo_motion_ready(arm)
        print(f"  [sdk-sim] virtual start reached max_err={start_error:.4f}rad; physical arm was not prepositioned")
        print(f"  [sdk-sim] tcp_offset={offset.tolist()} allow_approx_motion={approx_flag} (flange targets -> TCP)")
        rows = [virtual_start]
        orientation = GRIPPER_DOWN_RPY_RAD
        dt = 1.0 / program.rate
        for segment in program.segments:
            samples, _ = segment.samples(program.rate)
            for target in samples:
                if segment.kind == "movej":
                    code = arm.set_servo_angle_j(angles=target.tolist(), speed=0, mvacc=0, mvtime=0, is_radian=True)
                elif segment.kind == "movel":
                    flange_pose = [*(target * 1000.0).tolist(), *orientation]
                    pose = pose_flange_to_tcp(flange_pose, offset).tolist()
                    code = arm.set_servo_cartesian(mvpose=pose, speed=0, mvacc=0, mvtime=0, is_radian=True)
                else:
                    code = 0
                if int(code) != 0:
                    err_warn: object = None
                    state: object = None
                    try:
                        if callable(getattr(arm, "get_err_warn_code", None)):
                            err_warn = arm.get_err_warn_code()
                        if callable(getattr(arm, "get_state", None)):
                            state = arm.get_state()
                    except Exception:
                        pass
                    arm.set_state(4)
                    raise RuntimeError(
                        f"SDK simulation command failed with code {code} "
                        f"(segment={segment.label}, err_warn={err_warn}, state={state})"
                    )
                time.sleep(dt)
                rows.append(read_q(arm, config.robot.dof))
        return np.stack(rows)
    finally:
        cleanup_failures: list[str] = []
        if approx_changed and approx_prev is not None:
            restore_code = int(arm.set_allow_approx_motion(approx_prev))
            if restore_code != 0:
                cleanup_failures.append(f"restore allow_approx_motion={approx_prev} failed with code {restore_code}")
        if simulation_requested:
            pause_code = int(arm.set_state(3))
            if pause_code != 0:
                cleanup_failures.append(f"pause failed with code {pause_code}")
            disable_code = int(arm.set_simulation_robot(False))
            if disable_code != 0:
                cleanup_failures.append(f"disable mode failed with code {disable_code}")
            deadline = time.monotonic() + 2.0
            while bool(getattr(arm, "is_simulation_robot", False)) and time.monotonic() < deadline:
                time.sleep(0.05)
            if bool(getattr(arm, "is_simulation_robot", False)):
                cleanup_failures.append("controller report still says simulation mode is active")
        if cleanup_failures:
            raise RuntimeError("SDK simulation cleanup failed: " + "; ".join(cleanup_failures))


def validate_robot_identity(arm: Any, config: Any, serial: str) -> None:
    if not serial or len(serial) < 8:
        raise RuntimeError("controller did not provide a complete serial number")
    verify_controller_robot(arm, config.robot.key)
    if int(getattr(arm, "error_code", 0)) != 0:
        raise RuntimeError("controller has an active error; inspect it manually in UFACTORY Studio")


def print_summary(config: Any) -> None:
    print("configuration_sources:")
    for source in config.sources:
        print(f"  - {source}")
    print(
        f"robot={config.robot.key} rate_hz={config.motion.rate_hz} "
        f"sim_rate_hz={config.simulation.rate_hz} "
        f"allow_approx_motion={config.motion.allow_approx_motion} "
        f"z_min_m={config.safety.z_min_m}"
    )
    print(f"config_sha256={config.sha256}")


def program_sample_count(program: Any) -> int:
    count = 0
    has_joint_state = False
    for segment in program.segments:
        count += int(segment.samples(program.rate)[1])
        if segment.kind == "movej":
            if not has_joint_state and segment.q_start is not None:
                count += 1
            has_joint_state = True
    return count


def print_ik_compile_complete(program: Any, *, elapsed_s: float) -> None:
    print(
        f"[ik-compile] complete samples={program_sample_count(program)} elapsed_s={elapsed_s:.2f}",
        flush=True,
    )


def print_preflight_start(program: Any, config: Any) -> None:
    print(
        f"[preflight] checking samples={program_sample_count(program)} "
        f"collision_margin_mm={float(config.safety.min_collision_distance_m) * 1000.0:.1f}...",
        flush=True,
    )


def print_preflight_complete(preflight: Any) -> None:
    checks = {check.name: check for check in preflight.checks}
    collision_check = checks.get("collision")
    total_check = checks.get("total")
    collision_s = float(collision_check.duration_ms) / 1000.0 if collision_check is not None else 0.0
    total_s = float(total_check.duration_ms) / 1000.0 if total_check is not None else 0.0
    print(
        f"[preflight] complete status={'PASS' if preflight.passed else 'FAIL'} "
        f"collision_s={collision_s:.3f} total_s={total_s:.3f}",
        flush=True,
    )
    for warning in getattr(preflight, "warnings", ()):
        print(f"[preflight] warning: {warning}", flush=True)


def print_preflight_violation_summary(preflight: Any, report_path: Path | None) -> None:
    """Print every violation group without flooding the terminal per sample."""

    groups: dict[tuple[str, str, str], list[Any]] = {}
    for violation in preflight.violations:
        subject = violation.link or violation.joint or "-"
        key = (violation.type.value, violation.stage, subject)
        groups.setdefault(key, []).append(violation)
    print(
        f"violation_summary=grouped total={len(preflight.violations)} groups={len(groups)}",
        file=sys.stderr,
    )
    for (kind, stage, subject), items in sorted(groups.items()):
        samples = [item.sample_index for item in items if item.sample_index is not None]
        actuals = [float(item.actual) for item in items if item.actual is not None]
        limits = [float(item.limit) for item in items if item.limit is not None]
        sample_text = "n/a" if not samples else f"{min(samples)}-{max(samples)}"
        actual_text = "n/a" if not actuals else f"{min(actuals):.9g}"
        limit_text = "n/a" if not limits else f"{limits[0]:.9g}"
        if kind in {"clearance", "self_collision", "environment_collision"}:
            measurement_text = f"min_distance_m={actual_text} threshold_m={limit_text}"
        else:
            measurement_text = f"min_actual={actual_text} limit={limit_text}"
        print(
            f"  {kind}: {stage}: {subject} count={len(items)} samples={sample_text} {measurement_text}",
            file=sys.stderr,
        )
    if report_path is None:
        print("  full per-sample details: rerun with --report PATH", file=sys.stderr)
    else:
        print(f"  full per-sample details: {report_path}", file=sys.stderr)
