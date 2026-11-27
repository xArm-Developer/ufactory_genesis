"""Unified tool TCP offset geometry and real-controller consistency checks.

Runtime YAML configs store ``gripper.tcp_offset`` as a 6-element vector
``[x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]`` matching the default
``xarm-python-sdk`` (``is_radian=False``) and UFACTORY Studio conventions.
Internal kinematics (:mod:`ufactory.kinematics.tcp_offset`,
:class:`~ufactory.control.ArmMotionController`) use ``[mm, mm, mm, rad, rad, rad]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from ufactory.kinematics.orientation import GRIPPER_DOWN_RPY_RAD
from ufactory.kinematics.tcp_offset import pose_flange_to_tcp, pose_tcp_to_flange, rpy_matrix
from ufactory.types import FloatArray

DEFAULT_REAL_TCP_POS_TOL_MM = 10.0
DEFAULT_REAL_TCP_ROT_TOL_DEG = 2.0

_ADAPTER_TOOL_TIP_LINKS: dict[str, str] = {
    "bio_g2": "bio_gripper_g2_tcp_link",
    "bio_gripper_g2": "bio_gripper_g2_tcp_link",
    "vacuum_gripper": "vacuum_tcp",
    "lite6_vacuum_gripper": "link_tcp",
}


def _as_vec6(values: Sequence[float] | np.ndarray[Any, Any], *, name: str) -> FloatArray:
    arr: FloatArray = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size != 6 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain 6 finite values")
    out: FloatArray = arr.copy()
    return out


def tcp_offset_mm_deg_to_mm_rad(tcp_offset_mm_deg: Sequence[float] | np.ndarray[Any, Any]) -> FloatArray:
    """Convert ``[x_mm, y_mm, z_mm, r_deg, p_deg, y_deg]`` to ``[mm, mm, mm, rad, rad, rad]``."""
    arr = _as_vec6(tcp_offset_mm_deg, name="tcp_offset_mm_deg")
    arr[3:] = np.deg2rad(arr[3:])
    return arr


def tcp_offset_mm_rad_to_mm_deg(tcp_offset_mm_rad: Sequence[float] | np.ndarray[Any, Any]) -> FloatArray:
    """Convert ``[x_mm, y_mm, z_mm, r_rad, p_rad, y_rad]`` to ``[mm, mm, mm, deg, deg, deg]``."""
    arr = _as_vec6(tcp_offset_mm_rad, name="tcp_offset_mm_rad")
    arr[3:] = np.rad2deg(arr[3:])
    return arr


@dataclass(frozen=True)
class ToolTcpSpec:
    """Flange-to-tool-TCP specification used for task planning and mirroring."""

    tcp_offset_mm_deg: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    tool_tip_link: str | None = None
    grasp_table_clearance_m: float = 0.0
    grasp_height_extra_m: float = 0.0

    def __post_init__(self) -> None:
        vec = _as_vec6(self.tcp_offset_mm_deg, name="tcp_offset_mm_deg")
        object.__setattr__(
            self,
            "tcp_offset_mm_deg",
            (float(vec[0]), float(vec[1]), float(vec[2]), float(vec[3]), float(vec[4]), float(vec[5])),
        )
        for name in ("grasp_table_clearance_m", "grasp_height_extra_m"):
            val = float(getattr(self, name))
            if not math.isfinite(val) or val < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, val)

    @property
    def tcp_offset_mm_rad(self) -> FloatArray:
        """Return ``[x_mm, y_mm, z_mm, r_rad, p_rad, y_rad]`` for SDK / ArmMotionController."""
        return tcp_offset_mm_deg_to_mm_rad(self.tcp_offset_mm_deg)

    @property
    def tcp_offset_flange_m(self) -> tuple[float, float, float]:
        """Return ``(x_m, y_m, z_m)`` of the TCP expressed in the flange frame."""
        return (
            self.tcp_offset_mm_deg[0] / 1000.0,
            self.tcp_offset_mm_deg[1] / 1000.0,
            self.tcp_offset_mm_deg[2] / 1000.0,
        )

    @property
    def grasp_tcp_z_m(self) -> float:
        """Configured TCP height above table (m) at grasp/place contact."""
        return self.grasp_table_clearance_m + self.grasp_height_extra_m

    def tcp_to_flange_xyz(
        self,
        tcp_xyz_base_m: Sequence[float] | np.ndarray[Any, Any],
        *,
        orientation_rpy_rad: Sequence[float] = GRIPPER_DOWN_RPY_RAD,
    ) -> FloatArray:
        """Convert base-frame TCP xyz (m) to flange (link6) xyz (m) at the given tool orientation."""
        xyz = np.asarray(tcp_xyz_base_m, dtype=np.float64).reshape(3)
        rpy = tuple(float(v) for v in orientation_rpy_rad)
        tcp_pose_mm_rpy = [xyz[0] * 1000.0, xyz[1] * 1000.0, xyz[2] * 1000.0, rpy[0], rpy[1], rpy[2]]
        flange_pose_mm_rpy = pose_tcp_to_flange(tcp_pose_mm_rpy, self.tcp_offset_mm_rad)
        result: FloatArray = np.asarray(flange_pose_mm_rpy[:3], dtype=np.float64) / 1000.0
        return result

    def flange_to_tcp_xyz(
        self,
        flange_xyz_base_m: Sequence[float] | np.ndarray[Any, Any],
        *,
        orientation_rpy_rad: Sequence[float] = GRIPPER_DOWN_RPY_RAD,
    ) -> FloatArray:
        """Convert base-frame flange (link6) xyz (m) to TCP xyz (m) at the given tool orientation."""
        xyz = np.asarray(flange_xyz_base_m, dtype=np.float64).reshape(3)
        rpy = tuple(float(v) for v in orientation_rpy_rad)
        flange_pose_mm_rpy = [xyz[0] * 1000.0, xyz[1] * 1000.0, xyz[2] * 1000.0, rpy[0], rpy[1], rpy[2]]
        tcp_pose_mm_rpy = pose_flange_to_tcp(flange_pose_mm_rpy, self.tcp_offset_mm_rad)
        result: FloatArray = np.asarray(tcp_pose_mm_rpy[:3], dtype=np.float64) / 1000.0
        return result


def resolve_tool_tcp_spec(config_or_gripper: Any) -> ToolTcpSpec:
    """Resolve a :class:`ToolTcpSpec` from runtime config, GripperProfile, GripperControlParams, or ``None``."""
    if config_or_gripper is None:
        return ToolTcpSpec()
    if hasattr(config_or_gripper, "gripper") and hasattr(config_or_gripper, "robot"):
        return resolve_tool_tcp_spec(config_or_gripper.gripper)
    if hasattr(config_or_gripper, "tcp_offset"):
        adapter = str(getattr(config_or_gripper, "adapter", ""))
        vec = _as_vec6(config_or_gripper.tcp_offset, name="gripper.tcp_offset")
        return ToolTcpSpec(
            tcp_offset_mm_deg=(
                float(vec[0]),
                float(vec[1]),
                float(vec[2]),
                float(vec[3]),
                float(vec[4]),
                float(vec[5]),
            ),
            tool_tip_link=_ADAPTER_TOOL_TIP_LINKS.get(adapter),
            grasp_table_clearance_m=float(getattr(config_or_gripper, "grasp_table_clearance_m", 0.0)),
            grasp_height_extra_m=float(getattr(config_or_gripper, "grasp_height_extra_m", 0.0)),
        )
    if hasattr(config_or_gripper, "tcp_offset_mm_deg"):
        vec = _as_vec6(config_or_gripper.tcp_offset_mm_deg, name="gripper.tcp_offset_mm_deg")
        family = str(getattr(config_or_gripper, "family", ""))
        tip_link = getattr(config_or_gripper, "tool_tip_link", None) or _ADAPTER_TOOL_TIP_LINKS.get(family)
        return ToolTcpSpec(
            tcp_offset_mm_deg=(
                float(vec[0]),
                float(vec[1]),
                float(vec[2]),
                float(vec[3]),
                float(vec[4]),
                float(vec[5]),
            ),
            tool_tip_link=tip_link,
        )
    raise TypeError(f"unsupported object for resolve_tool_tcp_spec: {type(config_or_gripper).__name__}")


def tcp_offset_difference(
    controller_tcp_mm_rad: Sequence[float] | np.ndarray[Any, Any],
    expected_tcp_mm_deg: Sequence[float] | np.ndarray[Any, Any],
) -> tuple[float, float]:
    """Return ``(pos_diff_mm, rot_diff_deg)`` between live controller TCP (mm, rad) and YAML TCP (mm, deg)."""
    ctrl = _as_vec6(controller_tcp_mm_rad, name="controller_tcp_mm_rad")
    exp = tcp_offset_mm_deg_to_mm_rad(expected_tcp_mm_deg)
    pos_diff_mm = float(np.linalg.norm(ctrl[:3] - exp[:3]))
    r_ctrl = rpy_matrix(float(ctrl[3]), float(ctrl[4]), float(ctrl[5]))
    r_exp = rpy_matrix(float(exp[3]), float(exp[4]), float(exp[5]))
    cos_angle = float(np.clip((np.trace(r_ctrl.T @ r_exp) - 1.0) / 2.0, -1.0, 1.0))
    rot_diff_deg = math.degrees(math.acos(cos_angle))
    return pos_diff_mm, rot_diff_deg


def check_controller_tcp_offset(
    controller_tcp_mm_rad: Sequence[float] | np.ndarray[Any, Any],
    expected_tcp_mm_deg: Sequence[float] | np.ndarray[Any, Any],
    *,
    mode: str,
    pos_tol_mm: float = DEFAULT_REAL_TCP_POS_TOL_MM,
    rot_tol_deg: float = DEFAULT_REAL_TCP_ROT_TOL_DEG,
) -> tuple[float, float]:
    """Verify controller TCP offset against YAML; fail closed in ``real`` mode on mismatch."""
    pos_diff_mm, rot_diff_deg = tcp_offset_difference(controller_tcp_mm_rad, expected_tcp_mm_deg)
    ctrl_deg = np.round(tcp_offset_mm_rad_to_mm_deg(controller_tcp_mm_rad), 3).tolist()
    exp_deg = np.round(_as_vec6(expected_tcp_mm_deg, name="expected_tcp_mm_deg"), 3).tolist()
    if pos_diff_mm > float(pos_tol_mm) or rot_diff_deg > float(rot_tol_deg):
        detail = (
            f"controller tcp_offset {ctrl_deg} (mm, deg) differs from YAML tcp_offset {exp_deg} (mm, deg) "
            f"by pos={pos_diff_mm:.2f} mm / rot={rot_diff_deg:.2f} deg "
            f"(limits: {float(pos_tol_mm):.1f} mm / {float(rot_tol_deg):.1f} deg)"
        )
        if str(mode) == "real":
            raise RuntimeError(
                f"real motion refused: {detail}; verify the physically mounted end effector "
                "and UFACTORY Studio TCP offset match the selected --gripper"
            )
        print(f"[hardware] WARNING: {detail}", flush=True)
        return pos_diff_mm, rot_diff_deg
    print(
        f"[hardware] tcp_offset check PASS pos_diff={pos_diff_mm:.2f}mm rot_diff={rot_diff_deg:.2f}deg",
        flush=True,
    )
    return pos_diff_mm, rot_diff_deg
