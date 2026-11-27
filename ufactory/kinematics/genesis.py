"""Link6 inverse-kinematics wrapper for the trajectory pipeline.

Wraps :meth:`gs.RigidEntity.inverse_kinematics` so sim/replay code streams
Cartesian (MoveL) targets one tick at a time and gets back per-tick arm joint
angles. The orientation is fixed gripper-down (roll=pi) for v1 (``tcp_offset=0``),
matching the pick-place demo.
"""

from __future__ import annotations

import numpy as np
import torch

import genesis as gs
from ufactory.kinematics.orientation import GRIPPER_DOWN_QUAT_XYZW


# Gripper-down orientation quaternion (roll=180deg), reused across MoveL ticks.
def down_quat(device=None, dtype=None) -> torch.Tensor:
    dev = device or gs.device
    dt = dtype or gs.tc_float
    x, y, z, w = GRIPPER_DOWN_QUAT_XYZW
    # Genesis uses WXYZ at its IK boundary; the public kinematics contract is
    # XYZW, so convert explicitly instead of defining another orientation.
    return torch.tensor([[w, x, y, z]], device=dev, dtype=dt)


DEFAULT_IK_MAX_POS_ERR_M = 2e-3

# Genesis IK stops as soon as it is inside its default tolerances (0.5 mm, 5 mrad). A held start pose that far from the
# program's first Cartesian point is a step in the planned stream: 0.445 mm at the xArm6 + Robotiq 2F-140 home read as
# 1.40 m/s^2 against the 0.8 m/s^2 MoveL limit in preflight. Start poses are therefore solved to micrometres.
HOME_IK_POS_TOL_M = 1e-6
HOME_IK_ROT_TOL_RAD = 1e-5
HOME_IK_REFINE_ITERS = 100
HOME_IK_MAX_POS_ERR_M = 1e-5
# Orientation stays at Genesis's default tolerance: a 5-axis xArm5 cannot always reach exactly gripper-down with a
# per-unit calibration (1.47 mrad left on XF130505D43A02), and orientation did not cause the preflight step.
HOME_IK_MAX_ROT_ERR_RAD = 5e-3


def solve_home_ik(
    robot,
    ik_link,
    pos,
    quat,
    *,
    arm_dof_idx,
    init_qpos=None,
    arm_controller=None,
    target_pos_base=None,
    default_qpos_rad=None,
):
    """Solve a held start pose to micrometres on the joint branch Genesis's default IK picks.

    When ``arm_controller`` and ``target_pos_base`` are provided, uses the unified
    pure-NumPy :class:`~ufactory.control.ArmMotionController` (Pieper analytical +
    polish for Lite6/UF850, 5-DOF position-priority IK for xArm5, iterative SE(3)
    for xArm6/xArm7) so preflight, simulation, and hardware start on the identical
    exact joint configuration. Otherwise falls back to two-pass Genesis IK.
    """
    if arm_controller is not None and target_pos_base is not None:
        from ufactory.trajectory.ik import solve_arm_home_q

        seed_q = None
        if default_qpos_rad is not None:
            seed_q = np.asarray(default_qpos_rad, dtype=np.float64).reshape(-1)
        elif init_qpos is not None:
            seed_arr = init_qpos.detach().cpu().numpy() if hasattr(init_qpos, "detach") else np.asarray(init_qpos)
            seed_flat = np.asarray(seed_arr, dtype=np.float64).reshape(-1)
            if seed_flat.size == len(arm_dof_idx):
                seed_q = seed_flat
            elif seed_flat.size > max(arm_dof_idx):
                seed_q = seed_flat[list(arm_dof_idx)]
        q_rad = solve_arm_home_q(
            arm_controller,
            target_pos_base,
            default_qpos_rad=seed_q,
            include_tcp=False,
            pos_tol_m=HOME_IK_POS_TOL_M,
            rot_tol_rad=HOME_IK_ROT_TOL_RAD,
            max_pos_err_m=HOME_IK_MAX_POS_ERR_M,
            max_rot_err_rad=HOME_IK_MAX_ROT_ERR_RAD,
        )
        device = pos.device if hasattr(pos, "device") else gs.device
        dtype = pos.dtype if hasattr(pos, "dtype") else gs.tc_float
        n_dofs = int(getattr(robot, "n_dofs", getattr(robot, "n_qs", max(arm_dof_idx) + 1)))
        refined = torch.zeros((1, n_dofs), device=device, dtype=dtype)
        for i, idx in enumerate(arm_dof_idx):
            refined[0, idx] = float(q_rad[i])
        return refined
    seed = {} if init_qpos is None else {"init_qpos": init_qpos}
    coarse = robot.inverse_kinematics(link=ik_link, pos=pos, quat=quat, dofs_idx_local=arm_dof_idx, **seed)
    refined, error = robot.inverse_kinematics(
        link=ik_link,
        pos=pos,
        quat=quat,
        dofs_idx_local=arm_dof_idx,
        init_qpos=coarse,
        max_samples=1,
        max_solver_iters=HOME_IK_REFINE_ITERS,
        pos_tol=HOME_IK_POS_TOL_M,
        rot_tol=HOME_IK_ROT_TOL_RAD,
        return_error=True,
    )
    if hasattr(error, "detach"):
        error = error.detach().cpu().numpy()
    residual = np.asarray(error, dtype=np.float64).reshape(-1, 6)
    pos_err = float(np.max(np.linalg.norm(residual[:, :3], axis=1)))
    rot_err = float(np.max(np.linalg.norm(residual[:, 3:], axis=1)))
    if not (np.isfinite(pos_err) and np.isfinite(rot_err)) or (
        pos_err > HOME_IK_MAX_POS_ERR_M or rot_err > HOME_IK_MAX_ROT_ERR_RAD
    ):
        target = pos.detach().cpu().numpy() if hasattr(pos, "detach") else pos
        raise ValueError(
            f"Genesis home IK residual {pos_err * 1e3:.4f} mm / {rot_err * 1e3:.4f} mrad exceeds "
            f"{HOME_IK_MAX_POS_ERR_M * 1e3:.3f} mm / {HOME_IK_MAX_ROT_ERR_RAD * 1e3:.3f} mrad "
            f"for target {np.asarray(target, dtype=np.float64).reshape(-1).round(6).tolist()}"
        )
    return refined


def solve_link6_ik(
    robot,
    ik_link,
    pos_world=None,
    *,
    arm_dof_idx,
    quat=None,
    init_qpos=None,
    damping: float | None = None,
    max_pos_err_m: float | None = DEFAULT_IK_MAX_POS_ERR_M,
    pos_base=None,
) -> np.ndarray:
    """Solve IK for a single link6 world-frame xyz target.

    ``pos_world`` may be a length-3 sequence (m) in the Genesis world frame.
    Returns the arm joint angles (rad) as a numpy array. ``quat`` defaults to
    gripper-down. When ``max_pos_err_m`` is set and ``robot`` supports forward
    kinematics, raises :class:`ValueError` if ``|FK(q) - pos_world|`` exceeds
    ``max_pos_err_m``.
    """
    if pos_world is None:
        if pos_base is None:
            raise TypeError("solve_link6_ik() missing required argument: 'pos_world'")
        pos_world = pos_base
    elif pos_base is not None:
        raise TypeError("solve_link6_ik() got both 'pos_world' and legacy 'pos_base'")
    if quat is None:
        quat = down_quat()
    target_xyz = np.asarray(list(pos_world), dtype=np.float64).reshape(3)
    pos_t = torch.as_tensor([target_xyz.tolist()], device=gs.device, dtype=gs.tc_float)
    if init_qpos is not None:
        init_qpos = _normalize_init_qpos(robot, init_qpos, arm_dof_idx)
    ik_kwargs = {}
    if damping is not None:
        ik_kwargs["damping"] = float(damping)
    sol = robot.inverse_kinematics(
        link=ik_link,
        pos=pos_t,
        quat=quat,
        dofs_idx_local=arm_dof_idx,
        init_qpos=init_qpos,
        **ik_kwargs,
    )
    if max_pos_err_m is not None:
        fk_pos = _eval_ik_link_pos(robot, sol, ik_link)
        if fk_pos is not None:
            pos_err = float(np.linalg.norm(fk_pos - target_xyz))
            if not np.isfinite(pos_err) or pos_err > float(max_pos_err_m):
                raise ValueError(
                    f"Genesis IK position residual {pos_err * 1000.0:.3f} mm exceeds "
                    f"limit {float(max_pos_err_m) * 1000.0:.3f} mm for target {target_xyz.tolist()}"
                )
    sol_vec = sol[0] if getattr(sol, "ndim", 1) == 2 else sol
    return sol_vec[arm_dof_idx].detach().cpu().numpy().astype(np.float64)


def _eval_ik_link_pos(robot, qpos, ik_link) -> np.ndarray | None:
    idx_local = getattr(ik_link, "idx_local", None)
    if idx_local is None:
        return None
    has_fk = (
        getattr(robot, "solver", None) is not None
        or getattr(robot, "_solver", None) is not None
        or callable(getattr(robot, "forward_kinematics", None))
    )
    if not has_fk:
        return None
    from ufactory.simulation.compat import forward_kinematics

    links_pos, _ = forward_kinematics(robot, qpos)
    idx = int(idx_local)
    ee_pos = links_pos[idx] if links_pos.ndim == 2 else links_pos[0, idx]
    if hasattr(ee_pos, "detach"):
        ee_pos = ee_pos.detach().cpu().numpy()
    return np.asarray(ee_pos, dtype=np.float64).reshape(3)


def _normalize_init_qpos(robot, init_qpos, arm_dof_idx) -> torch.Tensor:
    """Accept either full-robot qpos or arm-only qpos as an IK seed."""
    q = torch.as_tensor(init_qpos, device=gs.device, dtype=gs.tc_float)
    if q.ndim == 2 and q.shape[0] == 1:
        q = q.reshape(-1)
    arm_n = len(arm_dof_idx)
    if q.ndim == 1 and q.numel() == arm_n and getattr(robot, "n_qs", arm_n) != arm_n:
        try:
            full = robot.get_qpos()[0].detach().clone().to(device=gs.device, dtype=gs.tc_float)
        except Exception:
            full = torch.zeros(int(robot.n_qs), device=gs.device, dtype=gs.tc_float)
        full[arm_dof_idx] = q
        return full
    return q
