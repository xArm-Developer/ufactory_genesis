"""Host-side IK compilation for trajectory programs.

The pick-place / packaging ``servo_j`` path keeps the Cartesian waypoint program
as the source of truth, then compiles each MoveL tick into an explicit joint
target using the unified pure-NumPy :class:`~ufactory.control.ArmMotionController`
(falling back to Genesis link6 IK only when no ``arm_controller`` is provided).

After per-tick IK, the joint stream is plateau-collapsed and LSPB-retimed along
the joint polyline so finite-difference ``servo_j`` acceleration stays within
joint limits (especially at 100 Hz, where fine Cartesian sampling creates
near-duplicate IK solutions).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ufactory.kinematics.orientation import GRIPPER_DOWN_QUAT_XYZW
from ufactory.trajectory.profile import (
    _EPS,
    _lspb_alpha_for_duration,
    _lspb_displacement,
    _sample_grid,
    lspb_duration,
)
from ufactory.trajectory.segments import Program, Segment
from ufactory.trajectory.validation import validate_program

# Collapse IK samples closer than this (rad, Euclidean in joint space).
_IK_JOINT_PLATEAU_EPS_RAD = 1e-6
_DEFAULT_IK_DAMPING = 0.01
_UF850_IK_DAMPING = 0.05
_MAX_RETIME_SAMPLES = 1_000_000

HOME_IK_POS_TOL_M = 1e-6
HOME_IK_ROT_TOL_RAD = 1e-5
HOME_IK_MAX_POS_ERR_M = 1e-5
HOME_IK_MAX_ROT_ERR_RAD = 5e-3


@dataclass(frozen=True)
class ArmIKContext:
    """Genesis-free context for compiling MoveL streams with ``ArmMotionController``."""

    arm_controller: Any
    robot_key: str
    robot_urdf_path: str = ""
    kinematics_yaml_path: str | None = None
    kinematics_suffix: str | None = None
    include_tcp: bool = False


def solve_arm_home_q(
    arm_controller: Any,
    target_pos_base: Sequence[float] | np.ndarray,
    *,
    default_qpos_rad: Sequence[float] | np.ndarray | None = None,
    include_tcp: bool = False,
    pos_tol_m: float = HOME_IK_POS_TOL_M,
    rot_tol_rad: float = HOME_IK_ROT_TOL_RAD,
    max_pos_err_m: float = HOME_IK_MAX_POS_ERR_M,
    max_rot_err_rad: float = HOME_IK_MAX_ROT_ERR_RAD,
) -> np.ndarray:
    """Solve a gripper-down home pose to micrometres using pure-NumPy ``ArmMotionController``."""
    from ufactory.control.kinematics_chain import pose_to_se3
    from ufactory.control.solvers import solve_exact_ik

    base_xyz = np.asarray(target_pos_base, dtype=np.float64).reshape(3)
    target_se3 = pose_to_se3(
        np.concatenate((base_xyz, np.asarray(GRIPPER_DOWN_QUAT_XYZW, dtype=np.float64))),
        position_in_mm=False,
    )
    seed_q = None if default_qpos_rad is None else np.asarray(default_qpos_rad, dtype=np.float64).reshape(-1)
    effective_rot_tol = max_rot_err_rad if getattr(arm_controller, "dof", 6) == 5 else rot_tol_rad
    res = solve_exact_ik(
        arm_controller.chain,
        target_se3,
        seed_q_rad=seed_q,
        include_tcp=bool(include_tcp),
        pos_tol_m=float(pos_tol_m),
        rot_tol_rad=float(effective_rot_tol),
        cold_start_fallback=True,
    )
    if not (np.isfinite(res.position_error_m) and np.isfinite(res.orientation_error_rad)) or (
        res.position_error_m > max_pos_err_m or res.orientation_error_rad > max_rot_err_rad
    ):
        raise ValueError(
            f"ArmMotionController home IK residual {res.position_error_m * 1e3:.4f} mm / "
            f"{res.orientation_error_rad * 1e3:.4f} mrad exceeds "
            f"{max_pos_err_m * 1e3:.3f} mm / {max_rot_err_rad * 1e3:.3f} mrad "
            f"for target {base_xyz.round(6).tolist()}"
        )
    return np.asarray(res.q_rad, dtype=np.float64).reshape(-1).copy()


def compile_cartesian_program_to_joint_stream(program: Program, ctx) -> Program:
    """Return a Program whose MoveL segments are explicit MoveJ target streams.

    Gripper segments are copied unchanged. Existing MoveJ segments are copied
    unchanged and update the IK seed. For each MoveL segment, Cartesian samples
    are solved tick-for-tick with Genesis IK (previous solution as seed), then
    the joint stream is plateau-collapsed and LSPB-retimed so finite-difference
    ``servo_j`` acceleration stays within joint limits (critical at 100 Hz).
    """
    rate = float(program.rate)
    segments: list[Segment] = []
    last_q: np.ndarray | None = None
    compiled_movel = 0
    compiled_ticks = 0
    plateau_collapsed = 0
    any_retimed = False
    ik_damping = _ik_damping_for_context(ctx)

    for seg in program.segments:
        if seg.kind == "gripper":
            segments.append(seg)
            continue

        if seg.kind == "movej":
            samples, n = seg.samples(rate)
            if n:
                last_q = np.asarray(samples[-1], dtype=np.float64).reshape(-1)
            segments.append(seg)
            continue

        if seg.kind != "movel":
            raise ValueError(f"unsupported segment kind for IK compile: {seg.kind!r}")
        if seg.pose_start is None:
            raise ValueError(f"MoveL segment {seg.label!r} has no pose_start")

        samples, n = seg.samples(rate)
        q_start = _solve_base_xyz(ctx, seg.pose_start, init_qpos=last_q, damping=ik_damping)
        last_q = q_start
        q_rows = np.empty((n, q_start.size), dtype=np.float64)
        for tick, xyz_base in enumerate(samples):
            q = _solve_base_xyz(ctx, xyz_base, init_qpos=last_q, damping=ik_damping)
            q_rows[tick, :] = q
            last_q = q

        v_max = _joint_vmax(program)
        a_max = _joint_amax(program)
        q_retimed, n_out, duration_out, n_keys = retime_ik_joint_stream(
            q_start,
            q_rows,
            rate=rate,
            duration_s=float(seg.duration),
            v_max=v_max,
            a_max=a_max,
        )
        collapsed = max(0, (n + 1) - n_keys)
        plateau_collapsed += collapsed
        if abs(duration_out - float(seg.duration)) > 1e-9 or n_out != n:
            any_retimed = True
        q_end = q_retimed[-1].copy()
        last_q = q_end
        segments.append(
            Segment(
                kind="movej",
                duration=duration_out,
                v_max=v_max,
                a_max=a_max,
                label=seg.label,
                q_start=q_start,
                q_end=q_end,
                q_samples=q_retimed,
                pose_start=seg.pose_start,
                pose_end=seg.pose_end,
                samples_count=n_out,
            )
        )
        compiled_movel += 1
        compiled_ticks += n_out

    metadata = dict(program.metadata)
    metadata.update(
        {
            "kind": "joint-from-cartesian-ik",
            "source_kind": program.metadata.get("kind", ""),
            "ik_compiler": "ufactory.trajectory.ik.compile_cartesian_program_to_joint_stream",
            "ik_compiled_movel_segments": compiled_movel,
            "ik_compiled_ticks": compiled_ticks,
            "ik_timing_policy": "joint-lspb-retime",
            "ik_damping": ik_damping,
            "ik_joint_retimed": any_retimed,
            "ik_plateau_collapsed_samples": plateau_collapsed,
            "ik_robot_urdf": getattr(ctx, "robot_urdf_path", None),
            "ik_kinematics_yaml": getattr(ctx, "kinematics_yaml_path", None),
            "ik_kinematics_suffix": getattr(ctx, "kinematics_suffix", None),
        }
    )
    compiled = Program(
        segments=segments,
        rate=rate,
        limits=program.limits,
        robot_key=program.robot_key or getattr(ctx, "robot_key", None),
        metadata=metadata,
    )
    validate_program(compiled)
    return compiled


def collapse_joint_keypoints(
    q_start: np.ndarray,
    q_rows: np.ndarray,
    *,
    eps_rad: float = _IK_JOINT_PLATEAU_EPS_RAD,
) -> np.ndarray:
    """Return unique joint keypoints ``[q_start, ...]`` ending at ``q_rows[-1]``."""
    start = np.asarray(q_start, dtype=np.float64).reshape(-1)
    rows = np.asarray(q_rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 1:
        raise ValueError("q_rows must be a non-empty 2D array")
    if rows.shape[1] != start.size:
        raise ValueError(f"q_start/q_rows dof mismatch: {start.size} vs {rows.shape[1]}")
    pts = [start.copy()]
    for q in rows:
        if float(np.linalg.norm(q - pts[-1])) > eps_rad:
            pts.append(np.asarray(q, dtype=np.float64).copy())
    end = rows[-1]
    if float(np.linalg.norm(end - pts[-1])) > eps_rad:
        pts.append(np.asarray(end, dtype=np.float64).copy())
    return np.asarray(pts, dtype=np.float64)


def retime_ik_joint_stream(
    q_start: np.ndarray,
    q_rows: np.ndarray,
    *,
    rate: float,
    duration_s: float,
    v_max: float,
    a_max: float,
    eps_rad: float = _IK_JOINT_PLATEAU_EPS_RAD,
) -> tuple[np.ndarray, int, float, int]:
    """Collapse IK plateaus and LSPB-retime along the joint polyline.

    Returns ``(q_samples, n, duration_s, n_keypoints)``. ``q_samples`` are the
    streamed targets after ``q_start`` (same convention as MoveJ ``q_samples``).
    """
    keys = collapse_joint_keypoints(q_start, q_rows, eps_rad=eps_rad)
    n_keys = int(keys.shape[0])
    rate = float(rate)
    if rate <= 0.0:
        raise ValueError("rate must be positive")

    if n_keys == 1:
        n = max(1, int(round(max(float(duration_s), 0.0) * rate)))
        duration = float(n) / rate
        return np.tile(keys[-1], (n, 1)), n, duration, n_keys

    seg_lens = np.linalg.norm(np.diff(keys, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    path_len = float(cum[-1])
    if path_len < _EPS:
        n = max(1, int(round(max(float(duration_s), 0.0) * rate)))
        duration = float(n) / rate
        return np.tile(keys[-1], (n, 1)), n, duration, n_keys

    duration = float(duration_s)
    v_lim = float(v_max)
    a_lim = float(a_max)
    if v_lim > 0.0 and a_lim > 0.0:
        needed = lspb_duration(path_len, v_lim, a_lim)
        if needed <= _EPS:
            needed = max(_EPS, path_len / max(v_lim, _EPS))
    elif v_lim > 0.0:
        needed = max(_EPS, path_len / v_lim)
    else:
        needed = 0.0
    # Never compress the joint polyline into a window shorter than the LSPB
    # bound; Cartesian duration is a lower bound only when it is already safe.
    if duration < needed:
        duration = needed
    if duration <= _EPS:
        duration = max(_EPS, needed)
    n = max(1, int(round(duration * rate)))
    if n > _MAX_RETIME_SAMPLES:
        raise ValueError(f"retimed IK joint stream sample count {n} exceeds safety cap {_MAX_RETIME_SAMPLES}")
    duration = float(n) / rate
    u = _sample_grid(n)
    alpha = _lspb_alpha_for_duration(path_len, v_lim, a_lim, duration)
    s = _lspb_displacement(u, path_len, alpha)

    q_out = np.empty((n, keys.shape[1]), dtype=np.float64)
    for i, si in enumerate(s):
        j = int(np.searchsorted(cum, si, side="right") - 1)
        j = min(max(j, 0), seg_lens.size - 1)
        if seg_lens[j] < _EPS:
            q_out[i] = keys[j + 1]
        else:
            t = (float(si) - float(cum[j])) / float(seg_lens[j])
            t = min(1.0, max(0.0, t))
            q_out[i] = keys[j] + t * (keys[j + 1] - keys[j])
    q_out[-1] = keys[-1]
    return q_out, n, duration, n_keys


def _solve_base_xyz(
    ctx,
    xyz_base,
    *,
    init_qpos: np.ndarray | None,
    damping: float | None = None,
) -> np.ndarray:
    controller = getattr(ctx, "arm_controller", None)
    if controller is not None:
        include_tcp = bool(getattr(ctx, "include_tcp", False))
        xyz = np.asarray(xyz_base, dtype=np.float64).reshape(3)
        pose = np.concatenate((xyz, np.asarray(GRIPPER_DOWN_QUAT_XYZW, dtype=np.float64)))
        if init_qpos is None:
            from ufactory.control.kinematics_chain import pose_to_se3
            from ufactory.control.solvers import solve_exact_ik

            res = solve_exact_ik(
                controller.chain,
                pose_to_se3(pose, position_in_mm=False),
                include_tcp=include_tcp,
                cold_start_fallback=True,
            )
            return np.asarray(res.q_rad, dtype=np.float64).reshape(-1)
        res = controller.solve_step(
            pose,
            np.asarray(init_qpos, dtype=np.float64).reshape(-1),
            include_tcp=include_tcp,
        )
        return np.asarray(res.q_rad, dtype=np.float64).reshape(-1)
    from ufactory.kinematics.genesis import solve_link6_ik

    return solve_link6_ik(
        ctx.robot,
        ctx.ik_link,
        ctx.base_to_world(xyz_base),
        arm_dof_idx=ctx.arm_dof_idx,
        quat=ctx.down_quat,
        init_qpos=init_qpos,
        damping=damping,
    )


def _ik_damping_for_context(ctx) -> float:
    robot_key = str(getattr(ctx, "robot_key", "")).strip().lower()
    if robot_key == "uf850":
        return _UF850_IK_DAMPING
    return _DEFAULT_IK_DAMPING


def _joint_vmax(program: Program) -> float:
    if program.limits is None:
        return 0.0
    return float(program.limits.v_max_rad_s)


def _joint_amax(program: Program) -> float:
    if program.limits is None:
        return 0.0
    return float(program.limits.a_max_rad_s2)
