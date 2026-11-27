"""Lite6 reversed parallel gripper command-space conversions.

Unlike Gripper G2 (angled linkage, ``drive_joint`` in radians), the Lite6
gripper is a simple parallel-jaw mechanism: ``finger_joint1`` is a prismatic
joint (metres) with ``finger_joint2`` mirroring it via a ``mimic`` constraint.
The installed real gripper is reversed and has a 20-38 mm physical two-finger
gap. The trajectory sim models that as a fixed 20 mm closed offset plus the
vendor 0-8.9 mm per-finger URDF travel.

Real hardware note: the xArm Python SDK exposes this gripper only as two
digital-IO commands (``open_lite6_gripper`` / ``close_lite6_gripper``), not a
continuous position API. The gap-based conversions below are for the Genesis
sim/mirror path; the real executor maps any gripper segment to the nearer of
open/close.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

LITE6_GRIPPER_FINGER_TRAVEL_M = 0.0089
LITE6_GRIPPER_CLOSED_GAP_M = 0.020
LITE6_GRIPPER_OPEN_GAP_M = 0.038

LITE6_GRIPPER_SIM_CLOSED_DRIVE = 0.0
LITE6_GRIPPER_SIM_OPEN_DRIVE = LITE6_GRIPPER_FINGER_TRAVEL_M
LITE6_GRIPPER_DEMO_HOLD_STEPS = 200

LITE6_GRIPPER_KP = 700.0
LITE6_GRIPPER_KV = 20.0
LITE6_GRIPPER_DRIVE_LIMIT_N = 3.0
LITE6_GRIPPER_DAMPING = 0.05
LITE6_GRIPPER_FRICTIONLOSS = 0.0
LITE6_GRIPPER_HOLD_BIAS_GAP_M = 0.018

_FINGER_MESH_DIR = (
    Path(__file__).resolve().parents[2] / "assets" / "urdf" / "end_effectors" / "lite6_gripper" / "meshes" / "collision"
)
_FINGER1_VISUAL_ORIGIN_M = np.asarray((0.0, 0.0, 0.0), dtype=np.float64)
_FINGER2_VISUAL_ORIGIN_M = np.asarray((0.0, 0.0, 0.0), dtype=np.float64)
_FINGER1_VERTS: np.ndarray | None = None
_FINGER2_VERTS: np.ndarray | None = None


def clamp_lite6_gripper_gap_m(gap_m: float) -> float:
    """Clamp a two-finger gap in metres to the Lite6 gripper's physical range."""
    return max(LITE6_GRIPPER_CLOSED_GAP_M, min(LITE6_GRIPPER_OPEN_GAP_M, float(gap_m)))


def lite6_gripper_sim_drive_to_gap_m(drive: float) -> float:
    """Map Genesis ``finger_joint1`` position to physical two-finger gap in metres."""
    clipped = max(LITE6_GRIPPER_SIM_CLOSED_DRIVE, min(LITE6_GRIPPER_SIM_OPEN_DRIVE, float(drive)))
    span = LITE6_GRIPPER_SIM_OPEN_DRIVE - LITE6_GRIPPER_SIM_CLOSED_DRIVE
    if span <= 0.0:
        return LITE6_GRIPPER_CLOSED_GAP_M
    open_fraction = (clipped - LITE6_GRIPPER_SIM_CLOSED_DRIVE) / span
    return LITE6_GRIPPER_CLOSED_GAP_M + open_fraction * (LITE6_GRIPPER_OPEN_GAP_M - LITE6_GRIPPER_CLOSED_GAP_M)


def lite6_gripper_demo_drive(step: int, *, hold_steps: int = LITE6_GRIPPER_DEMO_HOLD_STEPS) -> float:
    """Pure open/closed keyframe law shared by viewers and diagnostics."""
    if int(step) < 0 or int(hold_steps) < 1:
        raise ValueError("step must be non-negative and hold_steps must be positive")
    phase = (int(step) // int(hold_steps)) % 2
    return LITE6_GRIPPER_SIM_CLOSED_DRIVE if phase else LITE6_GRIPPER_SIM_OPEN_DRIVE


def lite6_gripper_demo_label(step: int, *, hold_steps: int = LITE6_GRIPPER_DEMO_HOLD_STEPS) -> str:
    return (
        "closed" if lite6_gripper_demo_drive(step, hold_steps=hold_steps) == LITE6_GRIPPER_SIM_CLOSED_DRIVE else "open"
    )


def configure_lite6_gripper_drive(
    robot,
    gripper_dof_idx: list[int],
    all_gripper_dof_idx: list[int],
    *,
    kp: float = LITE6_GRIPPER_KP,
    kv: float = LITE6_GRIPPER_KV,
    force_lower_n: float = -LITE6_GRIPPER_DRIVE_LIMIT_N,
    force_upper_n: float = LITE6_GRIPPER_DRIVE_LIMIT_N,
    damping: float = LITE6_GRIPPER_DAMPING,
    frictionloss: float = LITE6_GRIPPER_FRICTIONLOSS,
) -> None:
    """Configure master ``finger_joint1`` PD drive and passive ``finger_joint2`` mimic mode."""
    pd_idx = list(gripper_dof_idx or all_gripper_dof_idx)
    damp_idx = list(all_gripper_dof_idx or gripper_dof_idx)
    if pd_idx:
        robot.set_dofs_kp(np.full(len(pd_idx), float(kp)), pd_idx)
        robot.set_dofs_kv(np.full(len(pd_idx), float(kv)), pd_idx)
        robot.set_dofs_force_range(
            np.full(len(pd_idx), float(force_lower_n)),
            np.full(len(pd_idx), float(force_upper_n)),
            pd_idx,
        )
    if damp_idx:
        robot.set_dofs_damping(np.full(len(damp_idx), float(damping)), damp_idx)
        robot.set_dofs_frictionloss(np.full(len(damp_idx), float(frictionloss)), damp_idx)
    passive_idx = [idx for idx in damp_idx if idx not in pd_idx]
    if passive_idx and callable(getattr(robot, "control_dofs_force", None)):
        robot.control_dofs_force(np.zeros(len(passive_idx)), passive_idx)


def _finger_mesh_vertices() -> tuple[np.ndarray, np.ndarray]:
    global _FINGER1_VERTS, _FINGER2_VERTS
    if _FINGER1_VERTS is None or _FINGER2_VERTS is None:
        import trimesh

        _FINGER1_VERTS = (
            np.asarray(
                trimesh.load(_FINGER_MESH_DIR / "finger1.stl", force="mesh").vertices,
                dtype=np.float64,
            )
            + _FINGER1_VISUAL_ORIGIN_M
        )
        _FINGER2_VERTS = (
            np.asarray(
                trimesh.load(_FINGER_MESH_DIR / "finger2.stl", force="mesh").vertices,
                dtype=np.float64,
            )
            + _FINGER2_VISUAL_ORIGIN_M
        )
    return _FINGER1_VERTS, _FINGER2_VERTS


def _link_world_vertices(link, mesh_vertices: np.ndarray) -> np.ndarray:
    pos = link.get_pos()[0].detach().cpu().numpy()
    quat = np.asarray(link.get_quat()[0].detach().cpu().numpy(), dtype=np.float64)
    w, x, y, z = (float(v) for v in quat / np.linalg.norm(quat))
    rot = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return mesh_vertices @ rot.T + pos


def _finger_close_axis(ctx) -> np.ndarray:
    left = ctx.left_finger.get_pos()[0].detach().cpu().numpy()
    right = ctx.right_finger.get_pos()[0].detach().cpu().numpy()
    axis = right - left
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        return np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    return axis / norm


def _lite6_glb_inner_contact_geometry(ctx, obj_pos=None) -> tuple[float, float, float, float]:
    if obj_pos is None:
        obj_pos = ctx.obj.get_pos()[0].detach().cpu().numpy()
    else:
        obj_pos = np.asarray(obj_pos, dtype=np.float64)
    half_y = float(ctx.obj_size[1]) / 2.0
    half_z = float(ctx.obj_size[2]) / 2.0
    z0, z1 = obj_pos[2] - half_z, obj_pos[2] + half_z
    finger1, finger2 = _finger_mesh_vertices()
    w1 = _link_world_vertices(ctx.left_finger, finger1)
    w2 = _link_world_vertices(ctx.right_finger, finger2)
    band1 = w1[(w1[:, 2] >= z0) & (w1[:, 2] <= z1)]
    band2 = w2[(w2[:, 2] >= z0) & (w2[:, 2] <= z1)]
    if len(band1) == 0 or len(band2) == 0:
        return 0.0, 0.0, 0.0, 0.0
    axis = _finger_close_axis(ctx)
    rel1 = (band1 - obj_pos) @ axis
    rel2 = (band2 - obj_pos) @ axis
    if float(rel1.mean()) <= float(rel2.mean()):
        neg_proj, pos_proj = rel1, rel2
    else:
        neg_proj, pos_proj = rel2, rel1
    inner_neg = float(neg_proj.max())
    inner_pos = float(pos_proj.min())
    gap_neg = float(-half_y - inner_neg)
    gap_pos = float(inner_pos - half_y)
    return inner_neg, inner_pos, gap_neg, gap_pos


def snap_object_to_lite6_glb_grasp(ctx) -> float:
    """Center the block between measured GLB inner pad Y coordinates.

    A single Y translation cannot close both air gaps when the pad span exceeds
      the cube width; centering removes left/right asymmetry so the residual
      clearance is ``(gap_neg + gap_pos) / 2`` per face.

      Returns the applied Y shift in metres.
    """
    _, _, gap_neg, gap_pos = _lite6_glb_inner_contact_geometry(ctx)
    shift_y = 0.5 * (gap_pos - gap_neg)
    if abs(shift_y) <= 1e-6:
        return 0.0
    pos = ctx.obj.get_pos()[0].clone()
    old_y = float(pos[1].item())
    pos[1] = pos[1] + shift_y
    ctx.obj.set_pos(pos.unsqueeze(0), zero_velocity=True)
    return float(pos[1].item() - old_y)
