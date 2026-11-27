"""Unified, configuration-driven pick-place command."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from ufactory.config import (
    dump_runtime_config,
    load_runtime_config,
    resolve_pick_place_object_spec,
)
from ufactory.manipulation.commands import (
    build_arm_controller,
    build_manipulation_backends,
    model_and_hashes as _model_and_hashes,
    paced_start_hold as _paced_start_hold,
    prepare_real_gripper_sender,
    prime_first_movej_sample,
    print_real_execution_summary,
    print_sdk_sim_failures,
    shutdown_real_gripper,
    write_json as _write_json,
    connect as _connect,
    read_q as _read_q,
    ensure_program_start as _ensure_program_start,
    ensure_cartesian_program_start as _ensure_cartesian_program_start,
    sdk_sim_feedback as _sdk_sim_feedback,
    validate_robot_identity as _validate_robot_identity,
    print_summary as _print_summary,
    print_ik_compile_complete as _print_ik_compile_complete,
    print_preflight_start as _print_preflight_start,
    print_preflight_complete as _print_preflight_complete,
    print_preflight_violation_summary as _print_preflight_violation_summary,
)
from ufactory.grippers.bio_g2 import bio_g2_controller_mode_error
from ufactory.grippers.robotiq import robotiq_controller_mode_error
from ufactory.grippers.vacuum import vacuum_controller_mode_error
from ufactory.manipulation.tcp import check_controller_tcp_offset, resolve_tool_tcp_spec
from ufactory.hardware import XArmTransport
from ufactory.kinematics import get_robot_sn, read_tcp_offset
from ufactory.kinematics.orientation import GRIPPER_DOWN_QUAT_XYZW
from ufactory.safety import validate_sdk_simulation
from ufactory.safety.adapters import EnvironmentObstacle
from ufactory.safety.gate import program_sha256, sha256_file
from ufactory.simulation import GenesisRuntimeManager, override_simulation_backend
from ufactory.simulation.compat import require_genesis_capabilities
from ufactory.trajectory.execution import ExecutionBindings, execute_real, execute_sim
from ufactory.trajectory.ik import (
    ArmIKContext,
    compile_cartesian_program_to_joint_stream,
    solve_arm_home_q,
)
from ufactory.trajectory.planner import (
    CartesianWaypoint,
    JointWaypoint,
    TrajectoryPlannerConfig,
    plan_mixed_waypoints,
)
from ufactory.trajectory.preflight import create_safety_gate


G2_GRIP_SETTLE_S = 0.5
LITE6_GRIPPER_DURATION_S = 0.5
LITE6_PLACE_SETTLE_S = 0.18
# Keep in sync with ufactory.trajectory.scene grasp-gap defaults.
DEFAULT_GRIPPER_GRASP_GAP_M = 0.022
LITE6_DEFAULT_GRIPPER_GRASP_GAP_M = 0.020
OBJECT_RELOCATED_STAGES = (
    "place-descend",
    "place-settle",
    "release",
    "retreat",
    "return-home",
)


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=list).encode()).hexdigest()


def _backends(config: Any, urdf: Path, *, calibrated_g2: bool = False) -> tuple[Any, Any]:
    params = config.task.parameters
    object_size = tuple(map(float, params["object_size_m"]))
    object_position = tuple(map(float, params["fixed_object_position_m"]))
    target_position = tuple(map(float, params["fixed_target_position_m"]))
    obstacles = (
        EnvironmentObstacle("table", (1.2, 1.2, 0.05), (0.3, 0.0, -0.025)),
        EnvironmentObstacle("object", object_size, object_position),
    )
    return build_manipulation_backends(
        config,
        urdf,
        obstacles=obstacles,
        spawn_center_m=object_position,
        place_center_m=target_position,
        relocated_stages=OBJECT_RELOCATED_STAGES,
        object_width_m=object_size[1],
        calibrated_g2=calibrated_g2,
    )


def _build_program(config: Any, kinematics: Any | None = None, *, q_home: np.ndarray | None = None) -> Any:
    """Build the v0.2.4-style pick-place program starting at Cartesian home.

    ``start_xyz`` is always the flange position above the pick point at ``home_z``
    (not FK of ``default_qpos``). Task pick/place coordinates are expressed in
    TCP space and converted to flange (``link6``) waypoints via
    :func:`~ufactory.manipulation.tcp.resolve_tool_tcp_spec`.
    When ``q_home`` is provided, a zero-length MoveJ seed is prepended for
    ``servo_j`` compilation/preposition. Otherwise, if ``kinematics`` is
    provided, exact IK seeds a MoveJ at home for ``servo_cartesian`` preflight.
    """

    params = config.task.parameters
    object_pos = np.asarray(params["fixed_object_position_m"], dtype=np.float64)
    target_pos = np.asarray(params["fixed_target_position_m"], dtype=np.float64)
    tool_tcp = resolve_tool_tcp_spec(config)
    grasp_tcp_z = tool_tcp.grasp_tcp_z_m
    grasp_flange = tool_tcp.tcp_to_flange_xyz([float(object_pos[0]), float(object_pos[1]), grasp_tcp_z])
    place_flange = tool_tcp.tcp_to_flange_xyz([float(target_pos[0]), float(target_pos[1]), grasp_tcp_z])
    is_lite6 = str(getattr(config.gripper, "adapter", "")) == "lite6"
    obj_x, obj_y = float(grasp_flange[0]), float(grasp_flange[1])
    place_x, place_y = float(place_flange[0]), float(place_flange[1])
    grasp_z = float(grasp_flange[2])
    pre_grasp_z = grasp_z + 0.10
    default_ee_z = float(params["default_ee_position_m"][2])
    lift_z = default_ee_z
    home_z = default_ee_z
    open_gap = float(config.gripper.open_gap_m)
    close_gap = _close_gap_m(config)
    gripper_duration_s = LITE6_GRIPPER_DURATION_S if is_lite6 else float(config.motion.gripper_duration_s)

    pre_grasp = [obj_x, obj_y, pre_grasp_z]
    grasp = [obj_x, obj_y, grasp_z]
    lift = [obj_x, obj_y, lift_z]
    place_top = [place_x, place_y, lift_z]
    place_grasp = [place_x, place_y, grasp_z]
    retreat = [place_x, place_y, lift_z]
    home = [obj_x, obj_y, home_z]

    grip_head: list[Any] = [
        {
            "type": "gripper",
            "gap_start": open_gap,
            "gap_end": close_gap,
            "duration": gripper_duration_s,
            "label": "grip",
        },
    ]
    if not is_lite6:
        grip_head.append(
            {
                "type": "gripper",
                "gap_start": close_gap,
                "gap_end": close_gap,
                "duration": G2_GRIP_SETTLE_S,
                "label": "grip-settle",
            }
        )

    place_tail: list[Any] = [
        CartesianWaypoint(place_grasp, label="place-descend"),
    ]
    if is_lite6:
        place_tail.append(
            {
                "type": "gripper",
                "gap_start": close_gap,
                "gap_end": close_gap,
                "duration": LITE6_PLACE_SETTLE_S,
                "label": "place-settle",
            }
        )
    place_tail.extend(
        [
            {
                "type": "gripper",
                "gap_start": close_gap,
                "gap_end": open_gap,
                "duration": gripper_duration_s,
                "label": "release",
            },
            CartesianWaypoint(retreat, label="retreat"),
            CartesianWaypoint(home, label="return-home"),
        ]
    )

    planner = TrajectoryPlannerConfig(
        robot_key=config.robot.key,
        rate=config.motion.rate_hz,
        speed_rad_s=config.motion.joint_speed_rad_s,
        mvacc_rad_s2=config.motion.joint_acceleration_rad_s2,
        linear_speed_m_s=config.motion.cartesian_speed_m_s,
        linear_acc_m_s2=config.motion.cartesian_acceleration_m_s2,
        z_min_m=config.safety.z_min_m,
        runtime_config=config,
    )
    waypoints: list[Any] = [
        CartesianWaypoint(pre_grasp, label="home->pregrasp"),
        CartesianWaypoint(grasp, label="descend"),
        *grip_head,
        CartesianWaypoint(lift, label="lift"),
        CartesianWaypoint(place_top, label="transit"),
        *place_tail,
    ]
    start_q = None
    if q_home is not None:
        q_home_arr = np.asarray(q_home, dtype=np.float64).reshape(-1)
        waypoints.insert(0, JointWaypoint(q_home_arr, label="start"))
        start_q = q_home_arr
    elif kinematics is not None:
        seed = np.asarray(config.arm.default_qpos_rad, dtype=np.float64).reshape(-1)
        home_pose = np.concatenate(
            (np.asarray(home, dtype=np.float64), np.asarray(GRIPPER_DOWN_QUAT_XYZW, dtype=np.float64))
        )
        q_at_home = np.asarray(kinematics.inverse(home_pose, seed), dtype=np.float64)
        waypoints.insert(0, JointWaypoint(q_at_home, label="start"))
        start_q = q_at_home
    return plan_mixed_waypoints(
        planner,
        waypoints,
        start_q=start_q,
        start_xyz=home,
    )


def _home_xyz_m(config: Any) -> np.ndarray:
    params = config.task.parameters
    object_pos = np.asarray(params["fixed_object_position_m"], dtype=np.float64)
    tool_tcp = resolve_tool_tcp_spec(config)
    flange_xy = tool_tcp.tcp_to_flange_xyz([float(object_pos[0]), float(object_pos[1]), 0.0])[:2]
    home_z = float(params["default_ee_position_m"][2])
    return np.asarray([float(flange_xy[0]), float(flange_xy[1]), home_z], dtype=np.float64)


def _arm_q_from_scene(ctx: Any) -> np.ndarray:
    q_full = np.asarray(ctx.home_qpos, dtype=np.float64).reshape(-1)
    arm_idx = list(ctx.arm_dof_idx)
    return q_full[arm_idx].copy()


def _build_ik_scene(
    config: Any,
    *,
    calibration: Path | None,
    show_viewer: bool = False,
    calibrated_g2: bool = False,
    arm_controller: Any | None = None,
) -> Any:
    from ufactory.trajectory.scene import build_scene

    gripper_name = config.gripper.adapter if config.gripper is not None else None
    tool_tcp = resolve_tool_tcp_spec(config)
    has_xy_offset = any(abs(v) > 1e-9 for v in tool_tcp.tcp_offset_flange_m[:2])
    home_xy_base = tuple(float(v) for v in _home_xyz_m(config)[:2]) if has_xy_offset else None
    return build_scene(
        robot_key=config.robot.key,
        rate=config.simulation.rate_hz,
        show_viewer=show_viewer,
        substeps=config.simulation.substeps,
        constraint_solver=config.simulation.constraint_solver,
        solver_iterations=config.simulation.solver_iterations,
        noslip_iterations=config.simulation.noslip_iterations,
        friction_cone=config.simulation.friction_cone,
        contact_resolution=config.simulation.contact_resolution,
        constraint_timeconst=config.simulation.constraint_time_constant_s,
        use_gjk_collision=config.simulation.use_gjk_collision,
        kinematics_yaml=str(calibration) if calibration is not None else None,
        calibrated_g2=calibrated_g2,
        g2_settings=config.simulation.g2,
        robotiq_settings=config.simulation.robotiq,
        gripper=gripper_name,
        excluded_geometry_pairs=config.gripper.excluded_geometry_pairs if config.gripper is not None else (),
        arm_controller=arm_controller,
        home_xy_base=home_xy_base,
        **_scene_layout_kwargs(config),
    )


def _mirror_attach_dist_m(config: Any) -> float | None:
    """Return mirror carry latch distance override, disabling carry for empty Bio G2 / vacuum runs."""
    if str(getattr(config.gripper, "adapter", "")) in {"bio_g2", "vacuum_gripper"}:
        return 0.0
    return None


def _open_sim_viewer(ctx: Any, *, warmup_steps: int = 3) -> None:
    """Settle the scene, then open the interactive viewer at the home pose."""

    from ufactory.visualization import start_deferred_viewer

    scene = ctx.scene
    for _ in range(max(0, int(warmup_steps))):
        scene.step()
    start_deferred_viewer(scene)


def _compile_servo_j_offline(
    config: Any,
    *,
    kinematics: Any,
    urdf: Path,
    calibration: Path | None = None,
    serial: str | None = None,
) -> tuple[Any, np.ndarray]:
    """Compile the Cartesian pick-place program to a joint stream without starting Genesis."""

    q_home = solve_arm_home_q(
        kinematics,
        _home_xyz_m(config),
        default_qpos_rad=config.arm.default_qpos_rad,
    )
    seeded = _build_program(config, q_home=q_home)
    ik_ctx = ArmIKContext(
        arm_controller=kinematics,
        robot_key=config.robot.key,
        robot_urdf_path=str(urdf),
        kinematics_yaml_path=str(calibration) if calibration is not None else None,
        kinematics_suffix=serial,
    )
    compiled = compile_cartesian_program_to_joint_stream(seeded, ik_ctx)
    return compiled, q_home


def _compile_servo_j_program(
    config: Any,
    *,
    calibration: Path | None,
    show_viewer: bool = False,
    arm_controller: Any | None = None,
) -> tuple[Any, Any, np.ndarray]:
    """Build Cartesian program, compile with down_quat, return (program, ctx, q_home)."""

    ctx = _build_ik_scene(
        config,
        calibration=calibration,
        show_viewer=show_viewer,
        arm_controller=arm_controller,
    )
    q_home = _arm_q_from_scene(ctx)
    seeded = _build_program(config, q_home=q_home)
    compiled = compile_cartesian_program_to_joint_stream(seeded, ctx)
    return compiled, ctx, q_home


def _close_gap_m(config: Any) -> float:
    adapter = str(getattr(config.gripper, "adapter", ""))
    if adapter == "lite6":
        return LITE6_DEFAULT_GRIPPER_GRASP_GAP_M
    if adapter == "vacuum_gripper":
        return float(getattr(config.gripper, "closed_gap_m", 0.0))
    closed_gap = float(getattr(config.gripper, "closed_gap_m", 0.0))
    return max(DEFAULT_GRIPPER_GRASP_GAP_M, closed_gap)


def _scene_layout_kwargs(config: Any) -> dict[str, Any]:
    """Map runtime task layout into ``build_scene`` kwargs (base frame)."""

    params = config.task.parameters
    object_pos = tuple(map(float, params["fixed_object_position_m"]))
    target_pos = tuple(map(float, params["fixed_target_position_m"]))
    object_spec = resolve_pick_place_object_spec(config)
    return {
        "obj_pos_base": object_pos,
        "place_pos_base": target_pos,
        "obj_size": object_spec.size_m,
        "obj_mass_kg": object_spec.mass_kg,
    }


def _hold_mirror_at_program_start(mirror: Any, *, hold_s: float) -> None:
    _paced_start_hold(mirror, hold_s)


def _warm_real_visual_tracker(tracker: Any) -> None:
    """Warm lazy Genesis mirror kernels before hardware motion is enabled."""

    print("[visual-warmup] compiling kinematic mirror object updates...")
    started = time.monotonic()
    tracker.warm_up()
    print(f"[visual-warmup] complete elapsed_s={time.monotonic() - started:.2f}")


def _hold_viewer(ctx: Any, on_step: Any = None) -> None:
    """Keep the viewer open until Ctrl+C; re-teleport when ``on_step`` is set."""

    visualizer = getattr(ctx.scene, "visualizer", None)
    viewer = getattr(visualizer, "viewer", None)
    if viewer is None:
        raise RuntimeError("viewer is unavailable")

    print("Viewer open. Press Ctrl+C to exit...")
    try:
        while viewer.is_alive():
            started = time.monotonic()
            try:
                if on_step is not None:
                    on_step()
                else:
                    ctx.scene.step()
            except Exception:
                if not viewer.is_alive():
                    break
                raise
            if on_step is not None:
                # Kinematic real mirrors deliberately disable Genesis' own
                # scene-time pacer. Keep the post-run hold responsive without
                # hammering the renderer as fast as the CPU allows.
                sleep_s = (1.0 / 30.0) - (time.monotonic() - started)
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
    except KeyboardInterrupt:
        pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ufactory-pick-place")
    parser.add_argument("--robot", required=True, choices=("xarm5", "xarm6", "xarm7", "uf850", "lite6"))
    parser.add_argument(
        "--gripper",
        choices=("default", "g2", "bio_gripper_g2", "lite6", "robotiq_2f_85", "robotiq_2f_140", "vacuum_gripper"),
        default="default",
        help="Select end-effector variant (bio_gripper_g2, robotiq_2f_85, robotiq_2f_140, vacuum_gripper support xarm5/6/7 and uf850)",
    )
    parser.add_argument(
        "--vacuum-hardware-version",
        type=int,
        choices=(1, 2),
        default=1,
        help="Vacuum gripper hardware version (1: plug-in TGPIO 0/1, 2: contact TGPIO 3/4)",
    )
    parser.add_argument("--mode", required=True, choices=("sim", "dry-run", "sdk-sim", "real"))
    parser.add_argument("--executor", choices=("servo_j", "servo_cartesian"))
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
    parser.add_argument("--csv", type=Path, help="Simulation execution samples")
    parser.add_argument("--sim-report", type=Path, help="Simulation execution report")
    parser.add_argument("--position-mm", type=float)
    parser.add_argument("--speed-percent", type=float)
    parser.add_argument("--speed-mm-s", type=float)
    parser.add_argument("--transport-speed-mm-s", type=float)
    parser.add_argument("--substeps", type=int, choices=(8, 32))
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
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Sim: force Genesis viewer; real: open kinematic mirror viewer (async, off servo path)",
    )
    parser.add_argument(
        "--visual-start-hold-s",
        type=float,
        default=0.5,
        help="With --visual on real mode: hold the initial mirror pose before streaming (seconds)",
    )
    args = parser.parse_args(argv)
    if args.visual_start_hold_s < 0.0:
        parser.error("--visual-start-hold-s must be non-negative")
    if args.visual and args.mode in {"dry-run", "sdk-sim"}:
        parser.error("--visual is only supported with --mode sim or --mode real")

    robotiq_error = robotiq_controller_mode_error("pick_place", args.robot, args.gripper, args.mode)
    if robotiq_error is not None:
        parser.error(robotiq_error)

    bio_error = bio_g2_controller_mode_error("pick_place", args.robot, args.gripper, args.mode)
    if bio_error is not None:
        parser.error(bio_error)

    vacuum_error = vacuum_controller_mode_error("pick_place", args.robot, args.gripper, args.mode)
    if vacuum_error is not None:
        parser.error(vacuum_error)

    if args.gripper == "bio_gripper_g2":
        if args.robot not in {"xarm5", "xarm6", "xarm7", "uf850"}:
            parser.error("--gripper bio_gripper_g2 requires --robot xarm5, xarm6, xarm7 or uf850")
        if args.mode == "sim":
            if args.speed_mm_s is not None and args.speed_percent is not None:
                parser.error("Provide at most one of --speed-mm-s or --speed-percent")
            from ufactory.simulation.bio_g2_model import (
                MAX_SPEED_MM_S,
                MIN_SPEED_MM_S,
                bio_g2_speed_mm_s_from_percent,
                bounded,
                finger_force_n,
            )
            from ufactory.trajectory.bio_g2_replay import (
                DEFAULT_PICK_PLACE_SPEED_MM_S,
                DEFAULT_SUBSTEPS,
                DEFAULT_TRANSPORT_SPEED_MM_S,
                run_bio_g2_pick_place,
            )

            try:
                if args.speed_mm_s is not None:
                    speed_mm_s = bounded(args.speed_mm_s, MIN_SPEED_MM_S, MAX_SPEED_MM_S, "speed_mm_s")
                elif args.speed_percent is not None:
                    if args.speed_percent <= 0:
                        raise ValueError("automatic Bio Gripper G2 manipulation requires --speed-percent > 0")
                    speed_mm_s = bio_g2_speed_mm_s_from_percent(args.speed_percent)
                else:
                    speed_mm_s = DEFAULT_PICK_PLACE_SPEED_MM_S
                force_percent = 50.0 if args.force_percent is None else float(args.force_percent)
                finger_force_n(force_percent)
                transport_speed_mm_s = (
                    DEFAULT_TRANSPORT_SPEED_MM_S
                    if args.transport_speed_mm_s is None
                    else float(args.transport_speed_mm_s)
                )
                if not np.isfinite(transport_speed_mm_s) or transport_speed_mm_s <= 0:
                    raise ValueError("transport_speed_mm_s must be finite and positive")
            except ValueError as exc:
                parser.error(str(exc))

            summary = run_bio_g2_pick_place(
                robot=args.robot,
                backend=args.backend or "gpu",
                headless=not bool(args.visual),
                executor=args.executor or "servo_cartesian",
                speed_mm_s=speed_mm_s,
                transport_speed_mm_s=transport_speed_mm_s,
                substeps=DEFAULT_SUBSTEPS if args.substeps is None else int(args.substeps),
                force_percent=force_percent,
                csv_path=args.csv,
                summary_path=args.sim_report or args.report,
            )
            metrics = summary["metrics"]
            if "place_xy_error_mm" in metrics and "home_return_error_mm" in metrics:
                print(
                    f"place_error_mm={metrics['place_xy_error_mm']:.3f} "
                    f"home_drift_mm={metrics['home_return_error_mm']:.3f} "
                    f"completed={summary['passed']}"
                )
            return 0 if summary["passed"] else 1
        if (
            any(
                getattr(args, name) is not None
                for name in (
                    "position_mm",
                    "speed_percent",
                    "force_percent",
                    "speed_mm_s",
                    "transport_speed_mm_s",
                    "substeps",
                )
            )
            or args.csv
            or args.sim_report
        ):
            parser.error("Bio Gripper G2 simulation speed/force/record flags require --mode sim")

    if args.executor is None:
        parser.error("--executor is required for this gripper")
    sim_section = "robotiq" if args.gripper in {"robotiq_2f_85", "robotiq_2f_140"} else "g2"
    gripper_overrides = {
        f"simulation.{sim_section}.{name}": getattr(args, name)
        for name in ("position_mm", "speed_percent", "force_percent")
        if getattr(args, name) is not None
    }
    if args.mode != "sim" and (gripper_overrides or args.csv or args.sim_report):
        parser.error("percentage commands and execution records require --mode sim")
    cli_overrides: dict[str, Any] = dict(gripper_overrides)
    if args.allow_approx_motion:
        cli_overrides["motion.allow_approx_motion"] = True
    if args.sim_rate_hz is not None:
        cli_overrides["simulation.rate_hz"] = args.sim_rate_hz
    selected_gripper = None if args.gripper == "default" else args.gripper
    try:
        config = load_runtime_config(
            args.robot,
            gripper=selected_gripper,
            config_path=args.config,
            overrides=cli_overrides,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.mode == "sim" and config.gripper.adapter == "g2" and config.simulation.g2.speed_percent <= 0:
        parser.error("automatic G2 manipulation requires --speed-percent > 0")
    if (
        args.mode == "sim"
        and config.gripper.adapter in {"robotiq_2f_85", "robotiq_2f_140"}
        and config.simulation.robotiq.speed_percent <= 0
    ):
        parser.error("automatic Robotiq manipulation requires --speed-percent > 0")
    if args.backend is not None:
        config = override_simulation_backend(config, args.backend)
    if args.print_config:
        print(dump_runtime_config(config), end="")
        return 0
    if args.mode == "sim" or args.visual:
        require_genesis_capabilities(
            pbr=True,
            deferred_viewer=bool(args.visual) or bool(config.simulation.show_viewer),
        )
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
    try:
        urdf, urdf_hash, calibration_hash = _model_and_hashes(config, args.calibration, serial)
        arm_controller = build_arm_controller(config, urdf) if urdf.is_file() else None
        ik_ctx = None
        q_home: np.ndarray | None = None
        genesis_for_ik = None
        try:
            if args.mode == "sim":
                from ufactory.trajectory.g2_confirmation import prepare_g2_program, prepare_robotiq_program

                genesis_for_ik = GenesisRuntimeManager(config.simulation)
                genesis_for_ik.__enter__()
                ik_ctx = _build_ik_scene(
                    config,
                    calibration=args.calibration,
                    calibrated_g2=True,
                    arm_controller=arm_controller,
                )
                urdf = Path(ik_ctx.robot_urdf_path)
                urdf_hash = sha256_file(urdf)
                kinematics, collision = _backends(config, urdf, calibrated_g2=True)
                ik_ctx.arm_controller = kinematics
                q_home = _arm_q_from_scene(ik_ctx)
                program = _build_program(config, q_home=q_home)
                if config.gripper.adapter == "g2":
                    program = prepare_g2_program(program, config.simulation.g2)
                elif config.gripper.adapter in {"robotiq_2f_85", "robotiq_2f_140"}:
                    program = prepare_robotiq_program(program, config.simulation.robotiq, config.gripper.adapter)
                if args.executor == "servo_j":
                    program = compile_cartesian_program_to_joint_stream(program, ik_ctx)
            elif args.executor == "servo_j":
                kinematics, collision = _backends(config, urdf)
                print(
                    "[ik-compile] compiling servo_j trajectory with ArmMotionController...",
                    flush=True,
                )
                ik_started = time.perf_counter()
                program, q_home = _compile_servo_j_offline(
                    config,
                    kinematics=kinematics,
                    urdf=urdf,
                    calibration=args.calibration,
                    serial=serial,
                )
                _print_ik_compile_complete(program, elapsed_s=time.perf_counter() - ik_started)
            else:
                kinematics, collision = _backends(config, urdf)
                program = _build_program(config, kinematics)
            scene_hash = _json_sha256(
                {"task": dict(config.task.parameters), "contacts": dict(config.task.allowed_contacts)}
            )
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
            if args.mode == "sim":
                assert ik_ctx is not None
                approved = gate.approve_simulation(program, preflight)
                from ufactory.trajectory.sim_executor import replay_sim

                show_viewer = bool(args.visual) or bool(config.simulation.show_viewer)
                if show_viewer:
                    _open_sim_viewer(ik_ctx)
                sim_report = execute_sim(
                    approved,
                    lambda checked: replay_sim(checked, ik_ctx, csv_path=args.csv, sim_report_path=args.sim_report),
                )
                if show_viewer:
                    _hold_viewer(ik_ctx)
                print(
                    f"place_error_mm={sim_report.place_error_mm:.3f} "
                    f"home_drift_mm={sim_report.home_drift_mm:.3f} "
                    f"metric_settle_ticks={sim_report.metric_settle_ticks}"
                )
                calibrated_adaptive = config.gripper.adapter in {"g2", "robotiq_2f_85", "robotiq_2f_140"}
                if calibrated_adaptive:
                    print(
                        f"completed={sim_report.completed} timing={'PASS' if sim_report.timing_passed else 'FAIL'} "
                        f"transport={'PASS' if sim_report.transport_passed else 'FAIL'}"
                    )
                if sim_report.failure_reason:
                    print(sim_report.failure_reason)
                return (
                    0
                    if sim_report.completed
                    and sim_report.timing_passed
                    and (not calibrated_adaptive or sim_report.transport_passed)
                    and sim_report.place_error_mm < 50.0
                    and sim_report.home_drift_mm < 10.0
                    else 1
                )

            assert arm is not None and serial is not None
            home_xyz = _home_xyz_m(config)
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
                    start_q = _ensure_program_start(
                        transport,
                        q_home,
                        allow_preposition=bool(args.confirm_real),
                    )
                else:
                    _ensure_cartesian_program_start(
                        transport,
                        home_xyz,
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
                evidence_path = args.report or Path("reports") / f"sdk_sim_{config.robot.key}.json"
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

            if not args.confirm_real:
                raise RuntimeError("real motion requires --confirm-real")
            approved = gate.approve(
                program,
                preflight,
                expected_serial_number=serial,
                sdk_evidence=evidence,
            )

            bridge = None
            tracker = None
            mirror = None
            mirror_ctx = None
            try:
                if args.visual:
                    from ufactory.trajectory.mirror_executor import (
                        AsyncMirrorBridge,
                        KinematicCarryTracker,
                        TrajKinematicMirror,
                    )
                    from ufactory.visualization import start_deferred_viewer

                    if ik_ctx is not None:
                        mirror_ctx = ik_ctx
                    else:
                        if genesis_for_ik is None:
                            genesis_for_ik = GenesisRuntimeManager(config.simulation)
                            genesis_for_ik.__enter__()
                        mirror_ctx = _build_ik_scene(
                            config,
                            calibration=args.calibration,
                            show_viewer=False,
                            arm_controller=kinematics,
                        )
                    attach_dist_m = _mirror_attach_dist_m(config)
                    carry_note = (
                        "empty grasp; cube not carried"
                        if attach_dist_m == 0.0
                        else "cube kinematically carried while gripped"
                    )
                    base_mm = [v * 1000.0 for v in mirror_ctx.base_pos_world]
                    print(
                        f"\n[mirror] robot={mirror_ctx.robot_key} gripper={mirror_ctx.gripper.family} "
                        f"rate={config.motion.rate_hz}Hz visual_model={mirror_ctx.visual_model} "
                        f"base=[{base_mm[0]:.0f},{base_mm[1]:.0f},{base_mm[2]:.0f}]mm "
                        f"kinematic open-loop; {carry_note} "
                        f"(no contact physics; async mirror off servo critical path)"
                    )
                    mirror = TrajKinematicMirror(mirror_ctx, approved.program)
                    mirror.prime_to_home()
                    start_deferred_viewer(mirror_ctx.scene, kinematic_mirror=True)
                    _hold_mirror_at_program_start(mirror, hold_s=args.visual_start_hold_s)
                    tracker_kwargs: dict[str, Any] = {
                        "grasp_gap_m": _close_gap_m(config),
                        "grasp_segment_label": "grip",
                        "release_segment_label": "release",
                        "approach_freeze_labels": ("descend",),
                    }
                    if attach_dist_m is not None:
                        tracker_kwargs["attach_dist_m"] = attach_dist_m
                    tracker = KinematicCarryTracker(mirror, **tracker_kwargs)
                    _warm_real_visual_tracker(tracker)
                    bridge = AsyncMirrorBridge(tracker)

                transport.authorize_motion(mode=1)
                prime_first_movej_sample(transport, approved.program, executor=args.executor)
                if mirror is not None:
                    mirror.prime_to_first_arm_segment_start(approved.program)
                current_drive, gripper_sender = prepare_real_gripper_sender(
                    arm,
                    transport,
                    config.gripper,
                    hardware_version=int(args.vacuum_hardware_version),
                )
                switch_ctx = (
                    kinematics.sync_hardware_switches(arm)
                    if args.executor == "servo_cartesian"
                    and callable(getattr(kinematics, "sync_hardware_switches", None))
                    and callable(getattr(arm, "set_allow_approx_motion", None))
                    else nullcontext()
                )
                try:
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
                finally:
                    shutdown_real_gripper(
                        arm,
                        config.gripper,
                        hardware_version=int(args.vacuum_hardware_version),
                    )
                print_real_execution_summary(execution)
                if args.visual and mirror_ctx is not None and tracker is not None:
                    _hold_viewer(mirror_ctx, on_step=tracker.hold_step)
                return 0 if execution.completed else 1
            finally:
                if bridge is not None:
                    bridge.close()
        finally:
            if genesis_for_ik is not None:
                genesis_for_ik.__exit__(None, None, None)
    finally:
        if arm is not None and getattr(arm, "connected", False):
            arm.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
