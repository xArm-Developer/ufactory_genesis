"""State-confirmed trajectory replay for the calibrated G2 drive."""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from ufactory.config.models import G2_PHYSICS_PROFILE, G2SimulationConfig, RobotiqSimulationConfig
from ufactory.grippers.robotiq import get_robotiq_spec
from ufactory.simulation.g2_adapter import (
    G2GenesisController,
    object_finger_contact_forces_n,
    pad_centers,
    pad_normal,
)
from ufactory.simulation.g2_model import g2_physics_summary, sdk_position_counts
from ufactory.simulation.robotiq_adapter import RobotiqGenesisController, robotiq_pad_centers
from ufactory.simulation.robotiq_model import robotiq_physics_summary
from ufactory.trajectory.g2_confirmation import (
    G2Telemetry,
    GripForceGate,
    ReleaseGate,
    grip_timeout_s,
    release_timeout_s,
    robotiq_grip_timeout_s,
    robotiq_release_timeout_s,
)


def save_sim_report(report, csv_path=None, sim_report_path=None, *, failure_subdir: str = "g2") -> None:
    if not report.completed and (csv_path is None or sim_report_path is None):
        from datetime import datetime
        from ufactory.config.assets import discover_asset_store

        stem = (
            discover_asset_store().root
            / f"reports/{failure_subdir}/failures"
            / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        csv_path = csv_path or stem.with_suffix(".csv")
        sim_report_path = sim_report_path or stem.with_suffix(".json")
    report.record_paths = {
        "csv": None if csv_path is None else str(csv_path),
        "sim_report": None if sim_report_path is None else str(sim_report_path),
    }
    if csv_path is not None:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            fields = list(report.records[0]) if report.records else ["time_s", "phase"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(report.records)
    if sim_report_path is not None:
        path = Path(sim_report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(report)
        payload.pop("records")
        payload["csv"] = None if csv_path is None else str(csv_path)

        def finite(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, dict):
                return {key: finite(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [finite(item) for item in value]
            return value

        path.write_text(
            json.dumps(finite(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
        )


def _scalar(value) -> float:
    return float(value.detach().reshape(-1)[0].item())


def _record(ctx, adapter, output, phase: str, time_s: float, previous_gap: float, q_arm, dt: float) -> dict:
    from genesis.utils.geom import transform_by_quat

    c, model = adapter.controller, adapter.model
    variant: str | None = getattr(adapter, "variant", None)
    lookup = c.force_lookup()
    position_mm = _scalar(c.position_mm)
    gap = adapter.gap_mm()
    q = _scalar(ctx.robot.get_dofs_position(ctx.gripper_dof_idx))
    torque = _scalar(ctx.robot.get_dofs_control_force(ctx.gripper_dof_idx))
    left, right = object_finger_contact_forces_n(
        ctx.obj.get_contacts(with_entity=ctx.robot, exclude_self_contact=True),
        left_link_idx=int(ctx.left_finger.idx),
        right_link_idx=int(ctx.right_finger.idx),
        left_normal=pad_normal(ctx.left_finger),
        right_normal=pad_normal(ctx.right_finger),
    )
    left, right = _scalar(left), _scalar(right)
    if variant is None:
        raw_counts = sdk_position_counts(position_mm)
        motor_rpm = int(_scalar(c.requested_speed_rpm))
        cal_rpm = lookup.calibration_rpm
        outside_cal = lookup.outside_calibration
        static_force = model.static_force_n(lookup, max(0, min(84, gap)))
    else:
        spec = get_robotiq_spec(variant)
        raw_counts = spec.gap_m_to_register(position_mm / 1000.0)
        motor_rpm = 0
        cal_rpm = 0.0
        outside_cal = lookup.outside_calibration
        static_force = model.static_force_n(lookup, max(0.0, min(spec.open_gap_mm, gap)))
    row = {
        "time_s": time_s,
        "phase": phase,
        "position_mm": position_mm,
        "position_raw_counts": raw_counts,
        "speed_percent": _scalar(c.speed_percent),
        "force_percent": _scalar(c.force_percent),
        "speed_command_mm_s": _scalar(c.requested_speed_mm_s),
        "motor_speed_command_rpm": motor_rpm,
        "calibration_speed_rpm": cal_rpm,
        "outside_calibration": outside_cal,
        "curve_force_n": lookup.force_n,
        "expected_static_force_n": static_force,
        "reference_gap_mm": _scalar(output.reference_gap_mm),
        "gap_mm": gap,
        "gap_velocity_mm_s": (gap - previous_gap) / dt,
        "drive_rad": q,
        "drive_velocity_rad_s": _scalar(ctx.robot.get_dofs_velocity(ctx.gripper_dof_idx)),
        "drive_torque_nm": torque,
        "torque_limit_nm": _scalar(output.torque_limit_nm),
        "integral_torque_nm": _scalar(output.integral_torque_nm),
        "motor_torque_estimate_nm": model.motor_torque_estimate_nm(torque),
        "left_normal_n": left,
        "right_normal_n": right,
        "sensor_force_n": (left + right) / 2,
    }
    # Enforce the shared record schema before the arm-specific keys are added.
    row = asdict(G2Telemetry(**row))
    position, tool = ctx.obj.get_pos(), ctx.ik_link.get_pos()
    inverse = ctx.ik_link.get_quat().clone()
    inverse[..., 1:] *= -1
    relative = transform_by_quat(position - tool, inverse).detach().cpu().numpy().reshape(3)
    if variant is None:
        a, b = pad_centers(model, ctx.left_finger, ctx.right_finger)
    else:
        a, b = robotiq_pad_centers(model, ctx.left_finger, ctx.right_finger)
    center = (a + b) / 2
    lq, rq = ctx.left_finger.get_quat(), ctx.right_finger.get_quat()
    sign = ((lq * rq).sum(dim=-1, keepdim=True) >= 0).to(lq.dtype) * 2 - 1
    quat = lq + sign * rq
    quat = quat / quat.norm(dim=-1, keepdim=True)
    quat[..., 1:] *= -1
    pad_relative = transform_by_quat(position - center, quat).detach().cpu().numpy().reshape(3)
    pad_in_tool = transform_by_quat(center - tool, inverse).detach().cpu().numpy().reshape(3)
    object_base = ctx.world_to_base(position.detach().cpu().numpy().reshape(3))
    tool_base = ctx.world_to_base(tool.detach().cpu().numpy().reshape(3))
    for name, values in (
        ("object_base", object_base),
        ("tool_base", tool_base),
        ("object_in_tool", relative),
        ("object_in_pad", pad_relative),
        ("pad_in_tool", pad_in_tool),
    ):
        row.update({f"{name}_{axis}_mm": float(values[i] * 1000) for i, axis in enumerate("xyz")})
    row["object_z_mm"] = _scalar(position[..., 2]) * 1000
    row["object_speed_mm_s"] = _scalar(ctx.obj.get_vel().norm(dim=-1)) * 1000
    row["home_error_mm"] = float(np.linalg.norm(np.asarray(tool_base) - ctx.home_pos_base) * 1000)
    actual = ctx.robot.get_dofs_position(ctx.arm_dof_idx).detach().cpu().numpy().reshape(-1)
    for i, (target, measured) in enumerate(zip(q_arm, actual, strict=True), start=1):
        row[f"arm_target_q{i}_rad"], row[f"arm_q{i}_rad"] = float(target), float(measured)
    return row


def _replay_calibrated(
    program,
    ctx,
    *,
    adapter,
    settings,
    open_mm: float,
    physics_profile: str,
    physics_summary: dict,
    grip_deadline_fn,
    release_deadline_fn,
    variant: str | None = None,
    failure_subdir: str = "g2",
    final_metric_settle_s: float = 0.04,
    on_phase=None,
    on_tick=None,
    csv_path=None,
    sim_report_path=None,
):
    from ufactory.trajectory.g2_confirmation import summarize
    from ufactory.trajectory.mirror_executor import resolve_tick_arm_q
    from ufactory.trajectory.sim_executor import (
        PhaseStatus,
        SimReport,
        _assert_finite_replay_state,
        _resolve_sim_rate_hz,
        physics_steps_for_tick,
    )

    if not math.isfinite(final_metric_settle_s) or final_metric_settle_s < 0:
        raise ValueError("final_metric_settle_s must be finite and non-negative")
    adapter.reset(max(0.0, min(open_mm, adapter.gap_mm())))
    adapter.command(open_mm, settings.speed_percent, settings.force_percent)
    report = SimReport(physics_profile=physics_profile, physics_parameters=physics_summary)
    report.physics_parameters["command"] = asdict(settings)
    report.physics_parameters.update(getattr(ctx, "simulation_parameters", {}))
    rate, dt = float(program.rate), 1 / float(program.rate)
    sim_rate_hz = _resolve_sim_rate_hz(ctx, rate)
    sim_dt = 1.0 / sim_rate_hz
    report.physics_parameters["rate_hz"] = program.rate
    report.physics_parameters["sim_rate_hz"] = sim_rate_hz
    sim_time_acc = 0.0
    q_arm = ctx.home_qpos[ctx.arm_dof_idx].astype(np.float64).copy()
    previous_gap = adapter.gap_mm()
    pending_confirmation = None

    def tick(segment, tick_index):
        nonlocal previous_gap, pending_confirmation, sim_time_acc
        actual_q = ctx.robot.get_dofs_position(ctx.arm_dof_idx)
        ctx.robot.control_dofs_position(actual_q.new_tensor(q_arm).reshape(actual_q.shape), ctx.arm_dof_idx)
        n_phys, sim_time_acc = physics_steps_for_tick(sim_rate_hz, rate, sim_time_acc)
        output = None
        for _ in range(max(1, n_phys)):
            output = adapter.apply(sim_dt)
            ctx.scene.step()
        _assert_finite_replay_state(ctx, phase=segment.label, tick=tick_index)
        row = _record(
            ctx,
            adapter,
            output,
            segment.label or segment.kind,
            round((len(report.records) + 1) * dt, 12),
            previous_gap,
            q_arm,
            dt,
        )
        previous_gap = row["gap_mm"]
        report.records.append(row)
        if pending_confirmation is not None:
            confirmation = report.confirmations[pending_confirmation]
            confirmation["next_phase"] = row["phase"]
            confirmation["next_action_time_s"] = row["time_s"]
            confirmation["next_action_delay_s"] = round(row["time_s"] - confirmation["confirmed_time_s"], 12)
            pending_confirmation = None
        if on_tick is not None:
            on_tick(segment, tick_index)
        return row

    try:
        for si, segment in enumerate(program.segments):
            samples, count = segment.samples(rate)
            if count == 0:
                continue
            status = PhaseStatus(segment.label or segment.kind, si, 0, 0, segment.v_max, segment.a_max)
            gate = None
            if segment.kind == "gripper" and segment.label in {"grip", "release"}:
                reference = _scalar(adapter.controller.reference_gap_mm)
                if segment.label == "grip":
                    deadline = grip_deadline_fn(reference, settings.position_mm, settings.speed_percent)
                    gate = GripForceGate(rate)
                    target = settings.position_mm
                else:
                    deadline = release_deadline_fn(reference, settings.speed_percent)
                    gate = ReleaseGate(rate, open_position_mm=open_mm)
                    target = open_mm
                # The approved duration is the maximum reference stroke deadline.
                # Reversal can only reduce the remaining reference travel.
                count = int(math.ceil(min(deadline, segment.duration) * rate - 1e-9))
                adapter.command(target, settings.speed_percent, settings.force_percent)
            for t in range(count):
                if segment.kind != "gripper":
                    started = time.perf_counter_ns()
                    try:
                        q_arm = resolve_tick_arm_q(ctx, segment, samples, t, init_qpos=q_arm, dt_s=dt)
                    except Exception:
                        status.ik_failures += 1
                        raise
                    finally:
                        if segment.kind == "movel":
                            status.ik_duration_ms += (time.perf_counter_ns() - started) / 1e6
                row = tick(segment, t)
                status.ticks += 1
                if gate is not None and gate.observe(row):
                    report.confirmations[segment.label] = {
                        "confirmed_time_s": row["time_s"],
                        "deadline_s": count / rate,
                        "samples": [
                            {
                                key: item[key]
                                for key in (
                                    "time_s",
                                    "gap_mm",
                                    "expected_static_force_n",
                                    "left_normal_n",
                                    "right_normal_n",
                                )
                            }
                            for item in report.records[-2:]
                        ],
                    }
                    pending_confirmation = segment.label
                    break
            status.duration = status.ticks / rate
            status.obj_pos_mm = tuple(float(v) * 1000 for v in ctx.obj.get_pos().detach().cpu().numpy().reshape(3))
            status.link6_mm = tuple(float(v) * 1000 for v in ctx.ik_link.get_pos().detach().cpu().numpy().reshape(3))
            if segment.kind == "movel" and segment.pose_end is not None:
                status.eside_arm_mm = float(
                    np.linalg.norm(np.asarray(status.link6_mm) - np.asarray(ctx.base_to_world(segment.pose_end)) * 1000)
                )
            report.phases.append(status)
            if on_phase is not None:
                on_phase(status, segment)
            if gate is not None and gate.confirmed_time_s is None:
                report.completed = False
                report.timeout_phase = segment.label
                report.failure_reason = f"{segment.label} confirmation timeout after {status.duration:.2f} s"
                break
        if report.completed:
            from ufactory.trajectory.segments import Segment

            settle = Segment("gripper", final_metric_settle_s, 0, 0, label="final-settle")
            for t in range(int(math.ceil(final_metric_settle_s * rate))):
                tick(settle, t)
                report.metric_settle_ticks += 1
    except Exception as exc:
        report.completed = False
        report.failure_reason = f"{type(exc).__name__}: {exc}"

    report.total_ticks = len(report.records)
    report.total_duration = report.total_ticks / rate
    report.place_error_mm = float(
        np.linalg.norm(ctx.obj.get_pos().detach().cpu().numpy().reshape(3) - ctx.base_to_world(ctx.place_pos_base))
        * 1000
    )
    report.home_drift_mm = float(
        np.linalg.norm(ctx.ik_link.get_pos().detach().cpu().numpy().reshape(3) - ctx.base_to_world(ctx.home_pos_base))
        * 1000
    )
    if report.records:
        config = SimpleNamespace(
            motion=SimpleNamespace(rate_hz=rate),
            task=SimpleNamespace(
                parameters={
                    "fixed_target_position_m": ctx.place_pos_base,
                    "fixed_object_position_m": ctx.obj_pos_base,
                    "object_size_m": ctx.obj_size,
                }
            ),
        )
        packaging = any(s.label == "post-release-settle" for s in program.segments)
        report.acceptance = summarize(report.records, config, packaging=packaging, open_position_mm=open_mm)
        report.timing_passed = report.completed and report.acceptance["timing_passed"]
        report.transport_passed = report.completed and report.acceptance["transport_passed"]
    else:
        report.timing_passed = report.transport_passed = False
    save_sim_report(report, csv_path, sim_report_path, failure_subdir=failure_subdir)
    return report


def replay_g2(
    program, ctx, *, final_metric_settle_s=0.04, on_phase=None, on_tick=None, csv_path=None, sim_report_path=None
):
    settings = ctx.g2_settings or G2SimulationConfig()
    if settings.speed_percent <= 0:
        raise ValueError("automatic G2 manipulation requires speed_percent > 0")
    adapter = ctx.g2_controller
    if adapter is None:
        adapter = ctx.g2_controller = G2GenesisController(
            ctx.robot, ctx.left_finger, ctx.right_finger, axis_link=ctx.ik_link
        )
    return _replay_calibrated(
        program,
        ctx,
        adapter=adapter,
        settings=settings,
        open_mm=84.0,
        physics_profile=G2_PHYSICS_PROFILE,
        physics_summary=g2_physics_summary(),
        grip_deadline_fn=grip_timeout_s,
        release_deadline_fn=release_timeout_s,
        variant=None,
        failure_subdir="g2",
        final_metric_settle_s=final_metric_settle_s,
        on_phase=on_phase,
        on_tick=on_tick,
        csv_path=csv_path,
        sim_report_path=sim_report_path,
    )


def replay_robotiq(
    program, ctx, *, final_metric_settle_s=0.04, on_phase=None, on_tick=None, csv_path=None, sim_report_path=None
):
    variant = getattr(ctx.gripper, "family", "robotiq_2f_85")
    settings = getattr(ctx, "robotiq_settings", None) or RobotiqSimulationConfig()
    if settings.speed_percent <= 0:
        raise ValueError("automatic Robotiq manipulation requires speed_percent > 0")
    adapter = getattr(ctx, "robotiq_controller", None)
    if adapter is None:
        adapter = ctx.robotiq_controller = RobotiqGenesisController(
            ctx.robot, ctx.left_finger, ctx.right_finger, variant=variant, axis_link=ctx.ik_link
        )
    summary = robotiq_physics_summary(variant)
    return _replay_calibrated(
        program,
        ctx,
        adapter=adapter,
        settings=settings,
        open_mm=adapter.model.spec.open_gap_mm,
        physics_profile=summary["physics_profile"],
        physics_summary=summary,
        grip_deadline_fn=lambda ref, pos, spd: robotiq_grip_timeout_s(variant, ref, pos, spd),
        release_deadline_fn=lambda ref, spd: robotiq_release_timeout_s(variant, ref, spd),
        variant=variant,
        failure_subdir="robotiq",
        final_metric_settle_s=final_metric_settle_s,
        on_phase=on_phase,
        on_tick=on_tick,
        csv_path=csv_path,
        sim_report_path=sim_report_path,
    )
