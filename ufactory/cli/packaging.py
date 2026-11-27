"""Safe configuration-driven packaging showcase command."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import os
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from ufactory.manipulation.commands import (
    build_packaging_backends as _backends,
    connect as _connect,
    ensure_cartesian_program_start as _ensure_cartesian_program_start,
    ensure_program_start as _ensure_program_start,
    model_and_hashes as _model_and_hashes,
    prepare_real_gripper_sender,
    prime_first_movej_sample,
    print_ik_compile_complete as _print_ik_compile_complete,
    print_preflight_complete as _print_preflight_complete,
    print_preflight_start as _print_preflight_start,
    print_preflight_violation_summary as _print_preflight_violation_summary,
    print_real_execution_summary,
    print_sdk_sim_failures,
    print_summary as _print_summary,
    read_q as _read_q,
    sdk_sim_feedback as _sdk_sim_feedback,
    validate_robot_identity as _validate_robot_identity,
    write_json as _write_json,
)
from ufactory.config import dump_runtime_config, load_runtime_config
from ufactory.grippers.robotiq import ROBOTIQ_ADAPTERS, robotiq_controller_mode_error
from ufactory.hardware import XArmTransport
from ufactory.kinematics import get_robot_sn, read_tcp_offset
from ufactory.manipulation.packaging.core import (
    build_packaging_program,
    packaging_layout,
    packaging_scene_sha256,
)
from ufactory.manipulation.tcp import check_controller_tcp_offset, resolve_tool_tcp_spec
from ufactory.robots.registry import robot_cli_choices
from ufactory.safety import validate_sdk_simulation
from ufactory.safety.gate import program_sha256
from ufactory.simulation import override_simulation_backend
from ufactory.simulation.compat import require_genesis_capabilities
from ufactory.trajectory.execution import ExecutionBindings, execute_real
from ufactory.trajectory.preflight import create_safety_gate


def _build_packaging_context(config: Any, urdf: Path, *, show_viewer: bool = False) -> Any:
    from ufactory.manipulation.packaging.scene import build_packaging_scene
    from ufactory.manipulation.packaging.simulation import (
        _trajectory_context,
        init_showcase_robot,
    )

    scene, robot, block, display_layout = build_packaging_scene(
        sim_dt=1.0 / float(config.simulation.rate_hz),
        show_viewer=show_viewer,
        robot_urdf_path=str(urdf),
        runtime_config=config,
    )
    standard_offset = (0.0, 0.0)
    showcase_ctx = init_showcase_robot(
        robot,
        display_layout,
        scene,
        runtime_config=config,
        cartesian_xy_offset_m=standard_offset,
        arm_kp_scale=1.0,
        calibrated_g2=False,
    )
    traj_ctx = _trajectory_context(
        scene,
        robot,
        block,
        display_layout,
        showcase_ctx,
        config,
        cartesian_xy_offset_m=standard_offset,
    )
    return traj_ctx


def _run_sim(args: argparse.Namespace, config: Any) -> int:
    from ufactory.manipulation.packaging.simulation import PackagingSimulationOptions, run_packaging_simulation

    options = PackagingSimulationOptions(
        speed=args.speed,
        executor=args.executor,
        cycles=args.cycles,
        table_height=args.table_height,
        visual=args.visual,
        csv=args.csv,
        sim_report=args.sim_report,
        report=args.report,
    )
    return run_packaging_simulation(options, config)


def _positive_cycles(value: str) -> int:
    cycles = int(value)
    if cycles < 1:
        raise argparse.ArgumentTypeError("cycles must be at least 1")
    return cycles


def _sdk_evidence_path(report: Path | None, robot_key: str) -> Path:
    """Resolve the SDK-simulation evidence path without an xArm6 default."""

    return report or Path("reports") / f"sdk_sim_{robot_key}_packaging.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ufactory-packaging-showcase")
    parser.add_argument("--robot", default="xarm6", choices=robot_cli_choices())
    parser.add_argument(
        "--gripper",
        choices=("default", "g2", "lite6", "robotiq_2f_85", "robotiq_2f_140"),
        default="default",
        help="Select end-effector variant (robotiq_2f_85, robotiq_2f_140 support xarm5/6/7 and uf850)",
    )
    parser.add_argument("--mode", default="sim", choices=("sim", "dry-run", "sdk-sim", "real"))
    parser.add_argument("--executor", default="servo_j", choices=("servo_j", "servo_cartesian"))
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--backend",
        choices=("cpu", "gpu"),
        default=None,
        help="Override simulation.backend for Genesis (use cpu without a supported GPU)",
    )
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--ip", default=None)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--confirm-real", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--sim-report", type=Path)
    parser.add_argument("--position-mm", type=float)
    parser.add_argument("--speed-percent", type=float)
    parser.add_argument("--force-percent", type=float)
    parser.add_argument(
        "--allow-approx-motion",
        action="store_true",
        help="Enable firmware-aligned DLS singularity avoidance (mirrors set_allow_approx_motion(True))",
    )
    parser.add_argument(
        "--sim-rate-hz",
        type=float,
        default=None,
        help="Override Genesis physics simulation step rate in Hz (default: 50.0)",
    )
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0, help="Simulation-only playback multiplier")
    parser.add_argument("--cycles", type=_positive_cycles, default=None, help="Simulation cycle count (default: 1)")
    parser.add_argument("--table-height", type=float, default=None)
    args = parser.parse_args(argv)
    robotiq_error = robotiq_controller_mode_error("packaging", args.robot, args.gripper, args.mode)
    if robotiq_error is not None:
        parser.error(robotiq_error)
    sim_section = "robotiq" if args.gripper in ROBOTIQ_ADAPTERS else "g2"
    gripper_overrides = {
        f"simulation.{sim_section}.{name}": getattr(args, name)
        for name in ("position_mm", "speed_percent", "force_percent")
        if getattr(args, name) is not None
    }
    if args.mode != "sim" and (gripper_overrides or args.csv or args.sim_report):
        parser.error("percentage commands and execution records require --mode sim")

    if args.speed <= 0.0:
        parser.error("--speed must be positive")
    if args.mode != "sim" and (args.speed != 1.0 or args.cycles is not None or args.table_height is not None):
        parser.error("--speed/--cycles/--table-height are simulation-only")
    if args.mode == "real" and not args.confirm_real:
        parser.error("real packaging motion requires --confirm-real")
    if args.visual and args.mode in {"dry-run", "sdk-sim"}:
        parser.error("--visual is only supported with --mode sim or --mode real")
    selected_gripper = None if args.gripper == "default" else args.gripper
    cli_overrides: dict[str, Any] = dict(gripper_overrides)
    if args.allow_approx_motion:
        cli_overrides["motion.allow_approx_motion"] = True
    if args.sim_rate_hz is not None:
        cli_overrides["simulation.rate_hz"] = args.sim_rate_hz
    try:
        config = load_runtime_config(
            args.robot,
            task="packaging_showcase",
            gripper=selected_gripper,
            config_path=args.config,
            overrides=cli_overrides,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.backend is not None:
        config = override_simulation_backend(config, args.backend)
    if args.print_config:
        print(dump_runtime_config(config), end="")
        return 0
    if args.mode == "real" and (config.gripper is None or not config.gripper.real_command):
        parser.error(
            f"{config.robot.key} has no enabled real gripper; full real packaging is unsupported for this profile"
        )
    if args.mode != "sim" and args.visual:
        require_genesis_capabilities(
            pbr=True,
            deferred_viewer=True,
        )
    if args.mode == "sim":
        if config.gripper is not None and config.gripper.adapter == "g2" and config.simulation.g2.speed_percent <= 0:
            parser.error("automatic G2 manipulation requires --speed-percent > 0")
        if (
            config.gripper is not None
            and config.gripper.adapter in {"robotiq_2f_85", "robotiq_2f_140"}
            and config.simulation.robotiq.speed_percent <= 0
        ):
            parser.error("automatic Robotiq manipulation requires --speed-percent > 0")
        return _run_sim(args, config)
    _print_summary(config)

    ip = args.ip or os.environ.get("XARM_IP")
    arm = None
    serial = None
    if args.mode in {"sdk-sim", "real"}:
        if not ip:
            parser.error("--ip or XARM_IP is required for sdk-sim/real")
        if args.calibration is None:
            parser.error("--calibration is required for sdk-sim/real")
        arm = _connect(ip)
        serial = get_robot_sn(arm)
        _validate_robot_identity(arm, config, serial)
        print(
            "[hardware] identity verified; motion remains disabled until preflight PASS",
            flush=True,
        )

    bridge = None
    try:
        urdf, urdf_hash, calibration_hash = _model_and_hashes(config, args.calibration, serial)
        kinematics, collision = _backends(config, urdf)
        q_home: np.ndarray | None = None
        if args.executor == "servo_j":
            from ufactory.trajectory.ik import (
                ArmIKContext,
                compile_cartesian_program_to_joint_stream,
                solve_arm_home_q,
            )

            print(
                "[ik-compile] compiling servo_j trajectory with ArmMotionController...",
                flush=True,
            )
            ik_started = time.perf_counter()
            layout = packaging_layout(config)
            q_home = solve_arm_home_q(
                kinematics,
                layout.home_position_m,
                default_qpos_rad=config.arm.default_qpos_rad,
            )
            ik_ctx = ArmIKContext(
                arm_controller=kinematics,
                robot_key=config.robot.key,
                robot_urdf_path=str(urdf),
                kinematics_yaml_path=str(args.calibration) if args.calibration is not None else None,
                kinematics_suffix=serial,
            )
            program = compile_cartesian_program_to_joint_stream(
                build_packaging_program(config, q_home=q_home),
                ik_ctx,
            )
            _print_ik_compile_complete(program, elapsed_s=time.perf_counter() - ik_started)
        else:
            program = build_packaging_program(config, kinematics=kinematics)

        scene_hash = packaging_scene_sha256(config)
        gate = create_safety_gate(
            config,
            kinematics=kinematics,
            collision=collision,
            calibration_sha256=calibration_hash,
            scene_sha256=scene_hash,
            urdf_path=urdf,
        )
        _print_preflight_start(program, config)
        preflight = gate.preflight(program, executor=args.executor)
        _write_json(args.report, preflight.to_dict())
        print(f"program_sha256={preflight.program_sha256}")
        print(f"preflight={'PASS' if preflight.passed else 'FAIL'}")
        _print_preflight_complete(preflight)
        if not preflight.passed:
            _print_preflight_violation_summary(preflight, args.report)
            return 1
        if args.mode == "dry-run":
            return 0

        assert arm is not None and serial is not None
        layout = packaging_layout(config)
        # Read once so SDK simulation and real execution share the same TCP conversion.
        tcp_offset = read_tcp_offset(arm)
        print(f"[hardware] controller tcp_offset={tcp_offset.tolist()} (xyz mm, rpy rad)", flush=True)
        check_controller_tcp_offset(
            tcp_offset,
            resolve_tool_tcp_spec(config).tcp_offset_mm_deg,
            mode=args.mode,
        )
        transport = XArmTransport(arm, robot_key=config.robot.key, serial_number=serial, tcp_offset=tcp_offset)
        shadow = None
        stages = None
        if args.mode == "sdk-sim":
            shadow, stages = gate.shadow_joint_stream(program)
            start_q = shadow[0].copy()
        else:
            print(
                "[hardware] preflight approved; --confirm-real now permits start prepositioning",
                flush=True,
            )
            if args.executor == "servo_j":
                assert q_home is not None
                start_q = _ensure_program_start(transport, q_home, allow_preposition=bool(args.confirm_real))
            else:
                _ensure_cartesian_program_start(
                    transport,
                    np.asarray(layout.home_position_m, dtype=np.float64),
                    allow_preposition=bool(args.confirm_real),
                )
                start_q = transport.read_state().q_rad[: config.robot.dof].copy()

        evidence = None
        if args.executor == "servo_cartesian" or args.mode == "sdk-sim":
            if shadow is None or stages is None:
                shadow, stages = gate.shadow_joint_stream(program)
            firmware = _sdk_sim_feedback(
                arm,
                program,
                config,
                start_q,
                tcp_offset=tcp_offset,
                allow_approx_motion=config.motion.allow_approx_motion,
            )
            evidence = validate_sdk_simulation(
                robot_key=config.robot.key,
                serial_number=serial,
                program_sha256=program_sha256(program),
                config_sha256=config.sha256,
                shadow_joint_stream_rad=shadow,
                firmware_joint_stream_rad=firmware,
                stages=stages,
                policy=config.safety,
                kinematics=kinematics,
                collision=collision,
                allowed_collision=gate.collision_allowed,
                joint_lower_rad=gate.joint_lower_rad,
                joint_upper_rad=gate.joint_upper_rad,
            )
            evidence_path = _sdk_evidence_path(args.report, config.robot.key)
            _write_json(evidence_path, asdict(evidence))
            print(f"sdk_sim={'PASS' if evidence.passed else 'FAIL'}")
            if not evidence.passed:
                print_sdk_sim_failures(evidence, evidence_path)
                return 1
            if args.mode == "sdk-sim":
                return 0
            time.sleep(0.2)
            physical_q = _read_q(arm, config.robot.dof)
            if float(np.max(np.abs(physical_q - start_q))) > 0.01:
                raise RuntimeError("cannot prove SDK simulation and real run share the same physical start")

        approved = gate.approve(
            program,
            preflight,
            expected_serial_number=serial,
            sdk_evidence=evidence,
        )
        if args.visual:
            from ufactory.trajectory.mirror_process import PackagingMirrorProcess

            bridge = PackagingMirrorProcess(
                approved.program,
                robot_key=config.robot.key,
                config_path=args.config,
                urdf_path=urdf,
                start_hold_s=0.5,
            )
            bridge.start()

        transport.authorize_motion(mode=1)
        prime_first_movej_sample(
            transport,
            approved.program,
            executor=args.executor,
            missing_message="approved servo_j packaging program has no MoveJ sample",
        )
        current_drive, gripper_sender = prepare_real_gripper_sender(arm, transport, config.gripper)

        switch_ctx = (
            kinematics.sync_hardware_switches(arm)
            if args.executor == "servo_cartesian"
            and callable(getattr(kinematics, "sync_hardware_switches", None))
            and callable(getattr(arm, "set_allow_approx_motion", None))
            else nullcontext()
        )
        with switch_ctx:
            execution = execute_real(
                approved,
                transport=transport,
                bindings=ExecutionBindings(
                    robot_key=config.robot.key,
                    serial_number=serial,
                    config_sha256=config.sha256,
                    urdf_sha256=urdf_hash,
                    calibration_sha256=calibration_hash,
                    scene_sha256=scene_hash,
                ),
                confirm_real=True,
                runtime_monitor=lambda q, stage: gate.runtime_monitor(
                    q,
                    stage,
                    gripper_drive=current_drive[0],
                    include_collision=False,
                ),
                gripper_sender=gripper_sender,
                on_tick=bridge.on_tick if bridge is not None else None,
                not_ready_codes=frozenset((9,)),
            )
        print_real_execution_summary(execution)
        if bridge is not None:
            bridge.hold_until_closed()
        return 0 if execution.completed else 1
    finally:
        if bridge is not None:
            bridge.close()
        if arm is not None and getattr(arm, "connected", False):
            arm.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
