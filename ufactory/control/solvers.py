"""Pure-NumPy inverse kinematics solvers for all 5 UFACTORY arms + firmware-aligned DLS.

* **Lite6 / UF850**: Closed-form Pieper spherical-wrist analytical IK on the
  nominal geometry (enumerating all <= 8 branches, wrapping multi-turn joints to
  ``seed_q``, filtering by joint limits), followed by a 2-6 step damped
  Newton-Raphson polish on the compensated chain whenever per-unit calibration
  is active (or to absorb tiny URDF ``1.5708`` rounding).
* **xArm6**: 6-DOF damped SE(3) Levenberg-Marquardt / Newton-Raphson IK.
* **xArm7**: 7-DOF redundant minimum-norm damped IK with nullspace seed attraction.
* **xArm5**: 5-DOF position-priority hierarchical IK (hard position convergence
  < 1 um; secondary orientation projection inside the 2-DOF position nullspace,
  tolerating irreducible roll residuals caused by non-coplanar calibration).
* **Singularity-robust approximate step (``set_allow_approx_motion(True)``)**:
  Native ``[mm, rad]`` single-step Damped Least Squares (DLS, ``lambda_max = 1.0``)
  reverse-engineered from UF850 v2.8.100 and Lite6 v2.8.2 controllers, paired
  with Chiaverini C1 smooth damping ramp and fail-closed pose error ceilings.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np

from ufactory.control.kinematics_chain import (
    CalibratedKinematicsChain,
    so3_log,
)
from ufactory.kinematics.controller_fk import axis_angle_matrix
from ufactory.kinematics.tcp_offset import rpy_matrix
from ufactory.types import FloatArray

SPHERICAL_WRIST_ROBOTS = frozenset({"lite6", "uf850"})
DEFAULT_POS_TOL_M = 1e-7
DEFAULT_ROT_TOL_RAD = 1e-5
DEFAULT_XARM5_ROT_TOL_RAD = 5e-3


@dataclass(frozen=True)
class ApproxMotionConfig:
    """Tuning and fail-closed bounds for ``allow_approx_motion=True``."""

    damping: float = 1.0
    singular_value_threshold: float = 0.08
    trigger_joint_speed_rad_s: float | None = None
    max_position_error_m: float = 0.002
    max_orientation_error_rad: float = 0.05
    position_unit_scale: float = 1000.0

    def __post_init__(self) -> None:
        for name in (
            "damping",
            "singular_value_threshold",
            "max_position_error_m",
            "max_orientation_error_rad",
            "position_unit_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"ApproxMotionConfig.{name} must be finite")
        if self.damping <= 0.0:
            raise ValueError("ApproxMotionConfig.damping must be positive")
        if self.singular_value_threshold < 0.0:
            raise ValueError("ApproxMotionConfig.singular_value_threshold must be non-negative")
        if self.trigger_joint_speed_rad_s is not None and (
            not math.isfinite(self.trigger_joint_speed_rad_s) or self.trigger_joint_speed_rad_s <= 0.0
        ):
            raise ValueError("ApproxMotionConfig.trigger_joint_speed_rad_s must be finite and positive")
        if self.max_position_error_m <= 0.0 or self.max_orientation_error_rad <= 0.0:
            raise ValueError("ApproxMotionConfig error ceilings must be positive")
        if self.position_unit_scale <= 0.0:
            raise ValueError("ApproxMotionConfig.position_unit_scale must be positive")


@dataclass(frozen=True)
class IKStepResult:
    """Result and diagnostics of one inverse-kinematics step."""

    q_rad: FloatArray
    used_approx: bool
    effective_damping: float
    min_singular_value: float
    position_error_m: float
    orientation_error_rad: float
    solver: str


def _wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _wrap_near(angle: float, reference: float, lower: float, upper: float) -> float | None:
    """Choose ``angle + 2*pi*k`` inside ``[lower, upper]`` closest to ``reference``."""

    k_center = round((reference - angle) / (2.0 * math.pi))
    best: float | None = None
    best_dist = math.inf
    for dk in (-2, -1, 0, 1, 2):
        cand = angle + 2.0 * math.pi * (k_center + dk)
        if lower - 1e-9 <= cand <= upper + 1e-9:
            cand = float(np.clip(cand, lower, upper))
            dist = abs(cand - reference)
            if dist < best_dist:
                best_dist = dist
                best = cand
    return best


def _within_limits(q: FloatArray, lower: FloatArray, upper: FloatArray, tol: float = 1e-7) -> bool:
    return bool(np.all(q >= lower - tol) and np.all(q <= upper + tol))


def _clip_to_limits(q: FloatArray, lower: FloatArray, upper: FloatArray) -> FloatArray:
    return np.asarray(np.clip(q, lower, upper), dtype=np.float64)


# ------------------------------------------------------------------ Lite6 / UF850 closed-form analytical Pieper IK


def solve_spherical_wrist_analytical(
    chain: CalibratedKinematicsChain,
    target_flange_se3: FloatArray,
    *,
    seed_q_rad: Sequence[float] | FloatArray | None = None,
) -> list[FloatArray]:
    """Enumerate all valid closed-form Pieper IK solutions (up to 8 branches) for nominal Lite6 / UF850.

    Returns solutions sorted by weighted joint distance to ``seed_q_rad`` (closest first).
    """

    if chain.robot_key not in SPHERICAL_WRIST_ROBOTS or chain.dof != 6:
        raise ValueError(f"analytical spherical-wrist IK only supports Lite6 and UF850, got {chain.robot_key!r}")

    seed = np.zeros(6, dtype=np.float64) if seed_q_rad is None else chain.validate_q(seed_q_rad)
    nom = chain.nominal_origins  # (6, 6): [x, y, z, r, p, y]
    d1 = float(nom[0, 2])
    l2 = float(nom[2, 0])
    a4 = float(nom[3, 0])
    b4 = abs(float(nom[3, 1]))
    d6 = abs(float(nom[5, 1]))
    l3 = math.hypot(a4, b4)
    phi = math.atan2(a4, b4)

    # Lite6 vs UF850 planar sign convention:
    # Lite6: r =  L2*sin(q2) + a4*cos(q2-q3) - b4*sin(q2-q3); wrist Rz(q4)*Ry(-q5)*Rz(q6)
    # UF850: r = -L2*sin(q2) + a4*cos(q3-q2) - b4*sin(q3-q2) => identical with (u2, u3) = (-q2, -q3);
    #        wrist Rz(q4)*Ry(+q5)*Rz(q6).
    is_uf850 = chain.robot_key == "uf850"
    planar_sign = -1.0 if is_uf850 else 1.0
    wrist_q5_sign = 1.0 if is_uf850 else -1.0

    r_flange = np.asarray(target_flange_se3[:3, :3], dtype=np.float64)
    p_flange = np.asarray(target_flange_se3[:3, 3], dtype=np.float64)
    # Wrist center W = p_flange - d6 * z_flange
    w_pos = p_flange - d6 * r_flange[:, 2]
    wx, wy, wz = float(w_pos[0]), float(w_pos[1]), float(w_pos[2])
    r_xy = math.hypot(wx, wy)
    dz = wz - d1

    # Step 1: q1 candidates (2 shoulder branches)
    if r_xy < 1e-9:
        q1_branches: list[tuple[float, float]] = [(_wrap_to_pi(float(seed[0])), 0.0)]
    else:
        q1_fwd = math.atan2(wy, wx)
        q1_branches = [(q1_fwd, r_xy), (_wrap_to_pi(q1_fwd + math.pi), -r_xy)]

    # Precompute nominal origin rotations for exact R_04(q1, q2, q3)
    r_orig = [rpy_matrix(float(nom[i, 3]), float(nom[i, 4]), float(nom[i, 5])) for i in range(6)]
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    solutions: list[FloatArray] = []
    for q1_raw, r_planar in q1_branches:
        q1 = _wrap_near(q1_raw, float(seed[0]), float(chain.lower[0]), float(chain.upper[0]))
        if q1 is None:
            continue

        # Step 2: u3, u2 in canonical Lite6 planar coordinates
        dist_sq = r_planar * r_planar + dz * dz
        cos_u3_plus_phi = (l2 * l2 + l3 * l3 - dist_sq) / (2.0 * l2 * l3)
        if abs(cos_u3_plus_phi) > 1.0 + 1e-7:
            continue
        cos_clamped = float(np.clip(cos_u3_plus_phi, -1.0, 1.0))
        acos_val = math.acos(cos_clamped)
        u3_candidates = [acos_val - phi] if abs(acos_val) < 1e-10 else [acos_val - phi, -acos_val - phi]

        for u3 in u3_candidates:
            q3_raw = planar_sign * u3
            q3 = _wrap_near(q3_raw, float(seed[2]), float(chain.lower[2]), float(chain.upper[2]))
            if q3 is None:
                continue
            u3_eff = planar_sign * q3
            a_coeff = a4 * math.cos(u3_eff) + b4 * math.sin(u3_eff)
            b_coeff = l2 + a4 * math.sin(u3_eff) - b4 * math.cos(u3_eff)
            u2 = math.atan2(r_planar, dz) - math.atan2(a_coeff, b_coeff)
            q2_raw = planar_sign * u2
            q2 = _wrap_near(q2_raw, float(seed[1]), float(chain.lower[1]), float(chain.upper[1]))
            if q2 is None:
                continue

            # Step 3: Exact nominal R_04(q1, q2, q3, q4=0) -> R_wrist = R_04^T @ R_flange
            r01 = r_orig[0] @ axis_angle_matrix(z_axis, q1)
            r02 = r01 @ r_orig[1] @ axis_angle_matrix(z_axis, q2)
            r03 = r02 @ r_orig[2] @ axis_angle_matrix(z_axis, q3)
            r04_zero = r03 @ r_orig[3]
            r_wrist = r04_zero.T @ r_flange

            # Decompose R_wrist = Rz(q4) @ Ry(wrist_q5_sign * q5) @ Rz(q6)
            m22 = float(np.clip(r_wrist[2, 2], -1.0, 1.0))
            beta_mag = math.acos(m22)
            wrist_pairs: list[tuple[float, float, float]] = []
            if math.sin(beta_mag) < 1e-7:
                # Wrist singularity: only q4 +/- q6 is determined; keep q4 = seed[3].
                alpha = float(seed[3])
                if m22 >= 0.0:
                    gamma = math.atan2(float(r_wrist[1, 0]), float(r_wrist[0, 0])) - alpha
                    wrist_pairs.append((alpha, 0.0, gamma))
                else:
                    gamma = alpha - math.atan2(float(r_wrist[1, 0]), -float(r_wrist[0, 0]))
                    wrist_pairs.append((alpha, wrist_q5_sign * math.pi, gamma))
            else:
                for beta in (beta_mag, -beta_mag):
                    sb = math.sin(beta)
                    alpha = math.atan2(float(r_wrist[1, 2]) / sb, float(r_wrist[0, 2]) / sb)
                    gamma = math.atan2(float(r_wrist[2, 1]) / sb, -float(r_wrist[2, 0]) / sb)
                    wrist_pairs.append((alpha, wrist_q5_sign * beta, gamma))

            for q4_raw, q5_raw, q6_raw in wrist_pairs:
                q4 = _wrap_near(q4_raw, float(seed[3]), float(chain.lower[3]), float(chain.upper[3]))
                q5 = _wrap_near(q5_raw, float(seed[4]), float(chain.lower[4]), float(chain.upper[4]))
                q6 = _wrap_near(q6_raw, float(seed[5]), float(chain.lower[5]), float(chain.upper[5]))
                if q4 is None or q5 is None or q6 is None:
                    continue
                cand = np.array([q1, q2, q3, q4, q5, q6], dtype=np.float64)
                if not any(float(np.max(np.abs(cand - prev))) < 1e-6 for prev in solutions):
                    solutions.append(cand)

    # Sort closest to seed_q (prioritize avoiding large shoulder/elbow flips)
    weights = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0], dtype=np.float64)
    solutions.sort(key=lambda q: float(np.sum(weights * np.square(q - seed))))
    return solutions


# ------------------------------------------------------------------ Iterative IK solvers (all 5 robots)


def _pose_residual_vec(target_se3: FloatArray, current_se3: FloatArray) -> FloatArray:
    dp = target_se3[:3, 3] - current_se3[:3, 3]
    dw = so3_log(target_se3[:3, :3] @ current_se3[:3, :3].T)
    return np.concatenate((dp, dw))


def solve_iterative_6dof_or_7dof(
    chain: CalibratedKinematicsChain,
    target_se3: FloatArray,
    seed_q_rad: FloatArray,
    *,
    include_tcp: bool = True,
    max_iterations: int = 120,
    pos_tol_m: float = DEFAULT_POS_TOL_M,
    rot_tol_rad: float = DEFAULT_ROT_TOL_RAD,
    damping: float = 1e-6,
    max_step_rad: float = 0.15,
    nullspace_gain: float = 0.05,
) -> tuple[FloatArray, float, float]:
    """Damped SE(3) Newton-Raphson IK for 6-DOF (xArm6, calibrated Lite6/UF850 polish) and 7-DOF (xArm7)."""

    q = _clip_to_limits(chain.validate_q(seed_q_rad).copy(), chain.lower, chain.upper)
    seed_ref = q.copy()
    n = chain.dof
    is_redundant = n > 6
    pos_err = math.inf
    rot_err = math.inf

    for _ in range(max_iterations):
        cur_se3, jac = chain.forward_and_jacobian(q, include_tcp=include_tcp)
        e = _pose_residual_vec(target_se3, cur_se3)
        pos_err = float(np.linalg.norm(e[:3]))
        rot_err = float(np.linalg.norm(e[3:]))
        if pos_err <= pos_tol_m and rot_err <= rot_tol_rad:
            return q, pos_err, rot_err

        # Scale position to mm internally only for well-conditioned SVD check, keep SI solve with adaptive damping
        s_min = float(np.min(np.linalg.svd(jac, compute_uv=False)))
        scale = min(10.0, max(1.0, 1e-2 / max(s_min, 1e-12)))
        reg = (damping * (scale**2)) * np.eye(6, dtype=np.float64)
        jjt = jac @ jac.T + reg
        dq_task = jac.T @ np.linalg.solve(jjt, e)

        if is_redundant and nullspace_gain > 0.0 and (pos_err > 1e-4 or rot_err > 1e-3):
            # Soft attraction toward seed_ref in the nullspace during coarse approach;
            # disable inside the fine <0.1 mm / 1 mrad polish basin so second-order
            # nonlinear chord error cannot stall final Newton convergence.
            j_pinv = jac.T @ np.linalg.inv(jjt)
            null_proj = np.eye(n, dtype=np.float64) - j_pinv @ jac
            dq = dq_task + nullspace_gain * (null_proj @ (seed_ref - q))
        else:
            dq = dq_task

        max_step = float(np.max(np.abs(dq), initial=0.0))
        if max_step > max_step_rad:
            dq *= max_step_rad / max_step
        q = _clip_to_limits(q + dq, chain.lower, chain.upper)

    raise ValueError(
        f"{chain.robot_key} iterative IK did not converge after {max_iterations} iterations "
        f"(pos_err={pos_err * 1e3:.4f} mm, rot_err={rot_err * 1e3:.4f} mrad)"
    )


def solve_iterative_5dof_position_priority(
    chain: CalibratedKinematicsChain,
    target_se3: FloatArray,
    seed_q_rad: FloatArray,
    *,
    include_tcp: bool = True,
    max_iterations: int = 120,
    pos_tol_m: float = DEFAULT_POS_TOL_M,
    rot_tol_rad: float = DEFAULT_XARM5_ROT_TOL_RAD,
    max_step_rad: float = 0.15,
) -> tuple[FloatArray, float, float]:
    """Hierarchical 5-DOF IK for xArm5: exact position priority + nullspace orientation projection.

    A calibrated 5-DOF arm has non-coplanar link offsets, so full 6-DOF pose is
    over-constrained by ~0.3-2 mrad of irreducible roll residual. Solving 3-DOF
    position first and projecting orientation into the 2-DOF position nullspace
    achieves < 1 um position error while minimizing total rotation error.
    """

    if chain.dof != 5:
        raise ValueError(f"5-DOF position-priority IK requires dof=5, got {chain.dof}")

    q = _clip_to_limits(chain.validate_q(seed_q_rad).copy(), chain.lower, chain.upper)
    pos_err = math.inf
    rot_err = math.inf
    prev_rot_err = math.inf
    stalled = 0

    for _ in range(max_iterations):
        cur_se3, jac = chain.forward_and_jacobian(q, include_tcp=include_tcp)
        dp = target_se3[:3, 3] - cur_se3[:3, 3]
        dw = so3_log(target_se3[:3, :3] @ cur_se3[:3, :3].T)
        pos_err = float(np.linalg.norm(dp))
        rot_err = float(np.linalg.norm(dw))

        jp = jac[:3, :]  # (3, 5)
        jr = jac[3:, :]  # (3, 5)
        jp_jjt = jp @ jp.T + 1e-10 * np.eye(3, dtype=np.float64)
        jp_pinv = jp.T @ np.linalg.inv(jp_jjt)
        dq_pos = jp_pinv @ dp

        null_p = np.eye(5, dtype=np.float64) - jp_pinv @ jp
        jr_null = jr @ null_p  # (3, 5), rank <= 2
        # Damped least-squares projection of remaining orientation error in position nullspace
        gram = jr_null.T @ jr_null + 1e-6 * np.eye(5, dtype=np.float64)
        rhs = jr_null.T @ (dw - jr @ dq_pos)
        dq_ori = null_p @ np.linalg.solve(gram, rhs)

        dq = dq_pos + dq_ori
        if pos_err <= pos_tol_m and float(np.linalg.norm(dq_ori)) < 1e-7:
            if rot_err <= rot_tol_rad:
                return q, pos_err, rot_err
            break
        if pos_err <= pos_tol_m and abs(prev_rot_err - rot_err) < 1e-9:
            stalled += 1
            if stalled >= 4 and rot_err <= rot_tol_rad:
                return q, pos_err, rot_err
        else:
            stalled = 0
        prev_rot_err = rot_err

        max_step = float(np.max(np.abs(dq), initial=0.0))
        if max_step > max_step_rad:
            dq *= max_step_rad / max_step
        q = _clip_to_limits(q + dq, chain.lower, chain.upper)

    if pos_err <= pos_tol_m and rot_err <= rot_tol_rad:
        return q, pos_err, rot_err
    raise ValueError(
        f"{chain.robot_key} 5-DOF position-priority IK did not converge after {max_iterations} iterations "
        f"(pos_err={pos_err * 1e3:.4f} mm, rot_err={rot_err * 1e3:.4f} mrad, rot_tol={rot_tol_rad * 1e3:.3f} mrad)"
    )


# ------------------------------------------------------------------ Unified exact IK dispatcher (all 5 robots)


def solve_exact_ik(
    chain: CalibratedKinematicsChain,
    target_se3: FloatArray,
    seed_q_rad: Sequence[float] | FloatArray | None = None,
    *,
    include_tcp: bool = True,
    pos_tol_m: float = DEFAULT_POS_TOL_M,
    rot_tol_rad: float | None = None,
    cold_start_fallback: bool = False,
) -> IKStepResult:
    """Solve exact IK for any of the 5 UFACTORY arms (analytical + polish for Lite6/UF850, iterative for xArm5/6/7)."""

    seed = np.zeros(chain.dof, dtype=np.float64) if seed_q_rad is None else chain.validate_q(seed_q_rad)
    target = np.asarray(target_se3, dtype=np.float64).reshape(4, 4)

    if chain.dof == 5:
        r_tol = DEFAULT_XARM5_ROT_TOL_RAD if rot_tol_rad is None else float(rot_tol_rad)
        seeds_5 = [seed]
        if cold_start_fallback:
            mid5 = 0.5 * (chain.lower + chain.upper)
            seeds_5.extend(
                [
                    np.array([0.0, -0.5, -0.8, 1.3, 0.0], dtype=np.float64),
                    np.array([0.0, 0.2, -1.2, 1.0, 0.0], dtype=np.float64),
                    mid5,
                ]
            )
        last_exc_5: Exception | None = None
        for cand_seed in seeds_5:
            try:
                q_sol, pos_err, rot_err = solve_iterative_5dof_position_priority(
                    chain,
                    target,
                    cand_seed,
                    include_tcp=include_tcp,
                    pos_tol_m=pos_tol_m,
                    rot_tol_rad=r_tol,
                )
                jac = chain.jacobian(q_sol, include_tcp=include_tcp)
                jac_mm = jac.copy()
                jac_mm[:3, :] *= 1000.0
                s_min = float(np.min(np.linalg.svd(jac_mm, compute_uv=False)))
                return IKStepResult(q_sol, False, 0.0, s_min, pos_err, rot_err, "iterative-5dof-pos-priority")
            except ValueError as exc:
                last_exc_5 = exc
        raise ValueError(f"{chain.robot_key} 5-DOF iterative IK failed: {last_exc_5}")

    r_tol = DEFAULT_ROT_TOL_RAD if rot_tol_rad is None else float(rot_tol_rad)

    if chain.robot_key in SPHERICAL_WRIST_ROBOTS:
        flange_target = chain.tcp_target_to_flange(target) if include_tcp else target
        analytical_branches = solve_spherical_wrist_analytical(chain, flange_target, seed_q_rad=seed)
        # Near wrist singularity (|q5| < 0.05 rad) during tight continuous tracking,
        # URDF 4-decimal 1.5708 rounding (~3.7e-6 rad) gets amplified by 1/sin(q5) in
        # closed-form atan2; prefer local damped Newton polish from seed first there,
        # while using global Pieper analytical branches everywhere else.
        seed_pos_err, seed_rot_err = chain.pose_error(target, chain.forward_matrix(seed, include_tcp=include_tcp))
        near_wrist_tracking = (
            not cold_start_fallback and seed_pos_err < 0.015 and seed_rot_err < 0.1 and abs(float(seed[4])) < 0.05
        )
        last_exc: Exception | None = None
        candidates: list[tuple[FloatArray, str]] = []
        if near_wrist_tracking:
            candidates.append((seed, "iterative-6dof"))
            candidates.extend((b, "analytical-pieper+polish") for b in analytical_branches)
        else:
            candidates.extend((b, "analytical-pieper+polish") for b in analytical_branches)
            candidates.append((seed, "iterative-6dof"))
        for cand_seed, polish_label in candidates:
            cur_se3 = chain.forward_matrix(cand_seed, include_tcp=include_tcp)
            pos_err, rot_err = chain.pose_error(target, cur_se3)
            if pos_err <= pos_tol_m and rot_err <= r_tol and _within_limits(cand_seed, chain.lower, chain.upper):
                jac = chain.jacobian(cand_seed, include_tcp=include_tcp)
                jac_mm = jac.copy()
                jac_mm[:3, :] *= 1000.0
                s_min = float(np.min(np.linalg.svd(jac_mm, compute_uv=False)))
                exact_label = "analytical-pieper" if polish_label.startswith("analytical") else "iterative-6dof"
                return IKStepResult(cand_seed, False, 0.0, s_min, pos_err, rot_err, exact_label)
            try:
                q_sol, pos_err, rot_err = solve_iterative_6dof_or_7dof(
                    chain,
                    target,
                    cand_seed,
                    include_tcp=include_tcp,
                    max_iterations=60 if polish_label.startswith("analytical") else 120,
                    pos_tol_m=pos_tol_m,
                    rot_tol_rad=r_tol,
                )
                jac = chain.jacobian(q_sol, include_tcp=include_tcp)
                jac_mm = jac.copy()
                jac_mm[:3, :] *= 1000.0
                s_min = float(np.min(np.linalg.svd(jac_mm, compute_uv=False)))
                return IKStepResult(q_sol, False, 0.0, s_min, pos_err, rot_err, polish_label)
            except ValueError as exc:
                last_exc = exc
        raise ValueError(f"{chain.robot_key} analytical+polish IK failed: {last_exc}")

    # xArm6 / xArm7 iterative IK (with optional multi-seed fallback for cold-start queries)
    seeds_to_try = [seed]
    if cold_start_fallback:
        mid = 0.5 * (chain.lower + chain.upper)
        if chain.dof == 6:
            seeds_to_try.extend(
                [
                    np.array([0.0, -0.5, -0.8, 0.0, 1.3, 0.0], dtype=np.float64),
                    np.array([0.0, 0.2, -1.2, 0.0, 1.0, 0.0], dtype=np.float64),
                    mid,
                ]
            )
        elif chain.dof == 7:
            seeds_to_try.extend(
                [
                    np.array([0.0, -0.5, 0.0, 0.0, 0.5, 0.0, 0.0], dtype=np.float64),
                    np.array([0.0, -0.4, 0.0, 0.8, 0.0, 1.2, 0.0], dtype=np.float64),
                    np.array([0.0, 0.2, 0.0, 1.2, 0.0, 1.0, 0.0], dtype=np.float64),
                    mid,
                ]
            )
    last_exc = None
    for cand_seed in seeds_to_try:
        try:
            q_sol, pos_err, rot_err = solve_iterative_6dof_or_7dof(
                chain,
                target,
                cand_seed,
                include_tcp=include_tcp,
                pos_tol_m=pos_tol_m,
                rot_tol_rad=r_tol,
            )
            jac = chain.jacobian(q_sol, include_tcp=include_tcp)
            jac_mm = jac.copy()
            jac_mm[:3, :] *= 1000.0
            s_min = float(np.min(np.linalg.svd(jac_mm, compute_uv=False)))
            label = "iterative-7dof-redundant" if chain.dof == 7 else "iterative-6dof"
            return IKStepResult(q_sol, False, 0.0, s_min, pos_err, rot_err, label)
        except ValueError as exc:
            last_exc = exc
    raise ValueError(f"{chain.robot_key} iterative IK failed: {last_exc}")


# ------------------------------------------------------------------ Firmware-aligned native [mm, rad] DLS step


def solve_cartesian_step(
    chain: CalibratedKinematicsChain,
    target_se3: FloatArray,
    seed_q_rad: Sequence[float] | FloatArray,
    *,
    allow_approx_motion: bool,
    approx_config: ApproxMotionConfig | None = None,
    dt_s: float | None = None,
    include_tcp: bool = True,
    force_approx: bool = False,
) -> IKStepResult:
    """Solve one Cartesian control tick with optional firmware-aligned DLS singularity avoidance.

    When ``allow_approx_motion`` is ``False``, this strictly executes exact IK
    and never relaxes orientation or position. When ``True``, entering the
    Chiaverini singularity neighborhood (``sigma_min(J_mm) < sigma_0``), exact
    IK non-convergence, or exact single-step joint speed exceeding
    ``trigger_joint_speed_rad_s`` switches smoothly to the native ``[mm, rad]``
    single-step DLS law ``dq = (J_mm^T J_mm + lam_eff^2 I)^-1 J_mm^T e_mm``
    and enforces fail-closed position/orientation ceilings.
    """

    seed = chain.validate_q(seed_q_rad)
    target = np.asarray(target_se3, dtype=np.float64).reshape(4, 4)

    if not allow_approx_motion:
        return solve_exact_ik(chain, target, seed, include_tcp=include_tcp)

    cfg = approx_config or ApproxMotionConfig()
    cur_se3, jac = chain.forward_and_jacobian(seed, include_tcp=include_tcp)
    jac_mm = jac.copy()
    jac_mm[:3, :] *= cfg.position_unit_scale
    s_min = float(np.min(np.linalg.svd(jac_mm, compute_uv=False)))

    in_singular_band = cfg.singular_value_threshold > 0.0 and s_min < cfg.singular_value_threshold
    use_dls = bool(force_approx) or in_singular_band
    exact_res: IKStepResult | None = None

    if not use_dls:
        try:
            exact_res = solve_exact_ik(chain, target, seed, include_tcp=include_tcp)
            if (
                cfg.trigger_joint_speed_rad_s is not None
                and dt_s is not None
                and dt_s > 0.0
                and float(np.max(np.abs(exact_res.q_rad - seed))) / float(dt_s) > cfg.trigger_joint_speed_rad_s
            ):
                use_dls = True
        except ValueError:
            use_dls = True

    if not use_dls and exact_res is not None:
        return exact_res

    # Compute effective damping: Chiaverini C1 ramp inside singular band, full damping on speed/failure fallback
    if not force_approx and in_singular_band and exact_res is None and cfg.singular_value_threshold > 0.0:
        ratio = min(1.0, max(0.0, s_min / cfg.singular_value_threshold))
        eff_lam = cfg.damping * math.sqrt(max(0.0, 1.0 - ratio * ratio))
    else:
        eff_lam = cfg.damping

    dp_mm = (target[:3, 3] - cur_se3[:3, 3]) * cfg.position_unit_scale
    dw = so3_log(target[:3, :3] @ cur_se3[:3, :3].T)
    e_mm = np.concatenate((dp_mm, dw))

    n = chain.dof
    gram = jac_mm.T @ jac_mm + (eff_lam * eff_lam) * np.eye(n, dtype=np.float64)
    dq = np.linalg.solve(gram, jac_mm.T @ e_mm)
    q_next = seed + dq

    if not _within_limits(q_next, chain.lower, chain.upper):
        raise ValueError(f"{chain.robot_key} approximate DLS step exceeded joint position limits")
    q_next = _clip_to_limits(q_next, chain.lower, chain.upper)

    next_se3 = chain.forward_matrix(q_next, include_tcp=include_tcp)
    pos_err, rot_err = chain.pose_error(target, next_se3)
    if pos_err > cfg.max_position_error_m or rot_err > cfg.max_orientation_error_rad:
        raise ValueError(
            f"{chain.robot_key} approximate DLS residual {pos_err * 1e3:.3f} mm / {math.degrees(rot_err):.3f} deg "
            f"exceeds safety ceiling ({cfg.max_position_error_m * 1e3:.2f} mm / "
            f"{math.degrees(cfg.max_orientation_error_rad):.2f} deg)"
        )
    return IKStepResult(
        q_rad=q_next,
        used_approx=True,
        effective_damping=eff_lam,
        min_singular_value=s_min,
        position_error_m=pos_err,
        orientation_error_rad=rot_err,
        solver="approx-dls-mm-rad",
    )
