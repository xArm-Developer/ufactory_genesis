"""Pure-NumPy rigid kinematics chain with per-unit parameter compensation and TCP offset.

Reuses the repository's nominal URDF parser (:func:`load_nominal_chain`),
schema-v1 calibration loader (:func:`load_kinematics_calibration`), and SDK TCP
offset conventions (:mod:`ufactory.kinematics.tcp_offset`). No dependency on
Pinocchio, Genesis, Torch, or xArm SDK.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from ufactory.config.assets import RepositoryAssetStore
from ufactory.kinematics.calibration import KinematicsCalibration, load_kinematics_calibration
from ufactory.kinematics.controller_fk import (
    CALIBRATION_FIELDS,
    NominalChain,
    axis_angle_matrix,
    load_nominal_chain,
    matrix_rpy,
    rotation_angle,
)
from ufactory.kinematics.tcp_offset import pose_to_matrix, rpy_matrix
from ufactory.robots.assembly import resolve_robot_urdf
from ufactory.robots.registry import PROJECT_ROOT, get_robot_profile
from ufactory.types import FloatArray


def rotation_to_quat_xyzw(rotation: FloatArray) -> FloatArray:
    """Convert a 3x3 proper rotation matrix to a unit quaternion ``[x, y, z, w]``."""

    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(r[0, 0] + r[1, 1] + r[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(max(1.0 + float(r[0, 0] - r[1, 1] - r[2, 2]), 0.0)) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(max(1.0 + float(r[1, 1] - r[0, 0] - r[2, 2]), 0.0)) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(max(1.0 + float(r[2, 2] - r[0, 0] - r[1, 1]), 0.0)) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < 1e-15:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    q /= norm
    if q[3] < 0.0:
        q = -q
    return q


def quat_xyzw_to_rotation(quat_xyzw: Sequence[float] | FloatArray) -> FloatArray:
    """Convert quaternion ``[x, y, z, w]`` to a 3x3 rotation matrix."""

    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(-1)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("quaternion must contain 4 finite values [x, y, z, w]")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("quaternion must have non-zero norm")
    x, y, z, w = (float(v) / norm for v in q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def so3_log(rotation: FloatArray) -> FloatArray:
    """Robust SO(3) logarithm returning the rotation vector ``omega`` (rad), valid up to pi."""

    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cos_theta = float(np.clip((float(np.trace(r)) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-12:
        return 0.5 * np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]], dtype=np.float64)
    if math.pi - theta < 1e-5:
        # Near pi: extract unit axis from the largest diagonal element of (R + I)/2.
        diag = np.maximum(0.5 * (np.diag(r) + 1.0), 0.0)
        idx = int(np.argmax(diag))
        axis: FloatArray = np.zeros(3, dtype=np.float64)
        axis[idx] = math.sqrt(float(diag[idx]))
        if axis[idx] > 1e-12:
            for j in range(3):
                if j != idx:
                    axis[j] = 0.5 * float(r[idx, j] + r[j, idx]) / (2.0 * axis[idx])
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-12:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis /= axis_norm
        # Fix sign using skew part when non-zero.
        vee = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]], dtype=np.float64)
        if float(np.dot(axis, vee)) < 0.0:
            axis = -axis
        return axis * theta
    sin_theta = math.sin(theta)
    vee = 0.5 * np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]], dtype=np.float64)
    return vee * (theta / sin_theta)


def slerp_rotation(rot_a: FloatArray, rot_b: FloatArray, alpha: float) -> FloatArray:
    """Geodesic SLERP on SO(3) between ``rot_a`` (alpha=0) and ``rot_b`` (alpha=1)."""

    a = float(np.clip(alpha, 0.0, 1.0))
    ra = np.asarray(rot_a, dtype=np.float64).reshape(3, 3)
    rb = np.asarray(rot_b, dtype=np.float64).reshape(3, 3)
    if a <= 0.0:
        return ra.copy()
    if a >= 1.0:
        return rb.copy()
    omega = so3_log(ra.T @ rb)
    theta = float(np.linalg.norm(omega))
    if theta < 1e-15:
        return ra.copy()
    return np.asarray(ra @ axis_angle_matrix(omega / theta, theta * a), dtype=np.float64)


def pose_to_se3(pose: Sequence[float] | FloatArray, *, position_in_mm: bool = False) -> FloatArray:
    """Convert ``xyz+rpy`` (len 6) or ``xyz+quat_xyzw`` (len 7) to a 4x4 SE(3) matrix in metres."""

    arr = np.asarray(pose, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError("pose must contain only finite values")
    scale = 1e-3 if position_in_mm else 1.0
    mat: FloatArray = np.eye(4, dtype=np.float64)
    if arr.size == 6:
        mat[:3, 3] = arr[:3] * scale
        mat[:3, :3] = rpy_matrix(float(arr[3]), float(arr[4]), float(arr[5]))
        return mat
    if arr.size == 7:
        mat[:3, 3] = arr[:3] * scale
        mat[:3, :3] = quat_xyzw_to_rotation(arr[3:7])
        return mat
    raise ValueError(f"pose must have length 6 (xyz+rpy) or 7 (xyz+quat_xyzw), got {arr.size}")


def se3_to_pose_m_quat(transform: FloatArray) -> FloatArray:
    """Convert a 4x4 SE(3) matrix (metres) to ``[x, y, z, qx, qy, qz, qw]``."""

    t = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    quat = rotation_to_quat_xyzw(t[:3, :3])
    return np.concatenate((t[:3, 3], quat))


def se3_to_pose_mm_rpy(transform: FloatArray) -> FloatArray:
    """Convert a 4x4 SE(3) matrix (metres) to SDK ``[x_mm, y_mm, z_mm, roll, pitch, yaw]``."""

    t = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    roll, pitch, yaw = matrix_rpy(t[:3, :3])
    return np.array(
        [t[0, 3] * 1000.0, t[1, 3] * 1000.0, t[2, 3] * 1000.0, roll, pitch, yaw],
        dtype=np.float64,
    )


def _origin_matrix(row: FloatArray) -> FloatArray:
    mat: FloatArray = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rpy_matrix(float(row[3]), float(row[4]), float(row[5]))
    mat[:3, 3] = row[:3]
    return mat


def _tcp_offset_matrix_m(tcp_offset_mm_rpy: Sequence[float] | FloatArray) -> tuple[FloatArray, FloatArray]:
    arr = np.asarray(tcp_offset_mm_rpy, dtype=np.float64).reshape(-1)
    if arr.shape != (6,) or not np.all(np.isfinite(arr)):
        raise ValueError("tcp_offset must contain 6 finite values [x_mm, y_mm, z_mm, roll, pitch, yaw]")
    mat_mm = pose_to_matrix(arr)
    mat_m = mat_mm.copy()
    mat_m[:3, 3] *= 1e-3
    return arr.copy(), mat_m


@dataclass(frozen=True)
class CalibratedKinematicsChain:
    """Pure-NumPy revolute kinematic chain supporting nominal URDF + per-unit calibration + TCP offset."""

    robot_key: str
    joint_names: tuple[str, ...]
    nominal_origins: FloatArray  # (N, 6) [x, y, z, r, p, y] in m/rad from nominal URDF
    origins: FloatArray  # (N, 6) active origins (nominal or compensated)
    origin_transforms: FloatArray  # (N, 4, 4) precomputed T_origin per joint
    axes: FloatArray  # (N, 3) unit axis in local joint frame
    lower: FloatArray  # (N,) rad
    upper: FloatArray  # (N,) rad
    ee_link: str
    is_calibrated: bool
    calibration_sha256: str | None
    serial_number: str | None
    tcp_offset_mm_rpy: FloatArray  # (6,) [mm, mm, mm, rad, rad, rad]
    tcp_transform_m: FloatArray  # (4, 4) T_flange_to_tcp in metres

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    @classmethod
    def from_nominal_chain(
        cls,
        robot_key: str,
        nominal: NominalChain,
        *,
        calibration: KinematicsCalibration | None = None,
        tcp_offset_mm_rpy: Sequence[float] | FloatArray | None = None,
    ) -> CalibratedKinematicsChain:
        nominal_origins = np.asarray(nominal.origins, dtype=np.float64).copy()
        if calibration is not None:
            active_origins = np.asarray(
                [[calibration.joints[name][field] for field in CALIBRATION_FIELDS] for name in nominal.joint_names],
                dtype=np.float64,
            )
            is_calibrated = True
            cal_sha = calibration.sha256
            serial = calibration.serial_number
        else:
            active_origins = nominal_origins.copy()
            is_calibrated = False
            cal_sha = None
            serial = None

        origin_transforms = np.stack([_origin_matrix(row) for row in active_origins], axis=0)
        resolved_tcp = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0) if tcp_offset_mm_rpy is None else tcp_offset_mm_rpy
        tcp_arr, tcp_mat = _tcp_offset_matrix_m(resolved_tcp)
        return cls(
            robot_key=robot_key,
            joint_names=tuple(nominal.joint_names),
            nominal_origins=nominal_origins,
            origins=active_origins,
            origin_transforms=origin_transforms,
            axes=np.asarray(nominal.axes, dtype=np.float64).copy(),
            lower=np.asarray(nominal.lower, dtype=np.float64).copy(),
            upper=np.asarray(nominal.upper, dtype=np.float64).copy(),
            ee_link=nominal.ee_link,
            is_calibrated=is_calibrated,
            calibration_sha256=cal_sha,
            serial_number=serial,
            tcp_offset_mm_rpy=tcp_arr,
            tcp_transform_m=tcp_mat,
        )

    @classmethod
    def from_robot(
        cls,
        robot_key: str,
        *,
        urdf_path: str | Path | None = None,
        calibration_path: str | Path | None = None,
        serial_number: str | None = None,
        tcp_offset_mm_rpy: Sequence[float] | FloatArray | None = None,
        asset_store: RepositoryAssetStore | None = None,
    ) -> CalibratedKinematicsChain:
        """Build a pure-NumPy kinematic chain for one of the 5 UFACTORY arms."""

        profile = get_robot_profile(robot_key)
        joint_names = tuple(f"joint{i}" for i in range(1, profile.dof + 1))
        if urdf_path is None:
            store = asset_store or RepositoryAssetStore.discover()
            assets_rel = profile.assets_dir.relative_to(PROJECT_ROOT).as_posix()
            resolved_urdf = resolve_robot_urdf(
                profile.key,
                profile.default_urdf,
                assets_dir=assets_rel,
                store=store,
            )
        else:
            resolved_urdf = Path(urdf_path).expanduser().resolve()
        nominal = load_nominal_chain(resolved_urdf, joint_names)
        calibration: KinematicsCalibration | None = None
        if calibration_path is not None:
            calibration = load_kinematics_calibration(
                str(calibration_path),
                robot_key=profile.key,
                serial_number=serial_number,
                joint_names=joint_names,
            )
        return cls.from_nominal_chain(
            profile.key,
            nominal,
            calibration=calibration,
            tcp_offset_mm_rpy=tcp_offset_mm_rpy,
        )

    def with_tcp_offset(self, tcp_offset_mm_rpy: Sequence[float] | FloatArray) -> CalibratedKinematicsChain:
        """Return a new chain with updated TCP offset (mm + rad)."""

        tcp_arr, tcp_mat = _tcp_offset_matrix_m(tcp_offset_mm_rpy)
        return CalibratedKinematicsChain(
            robot_key=self.robot_key,
            joint_names=self.joint_names,
            nominal_origins=self.nominal_origins,
            origins=self.origins,
            origin_transforms=self.origin_transforms,
            axes=self.axes,
            lower=self.lower,
            upper=self.upper,
            ee_link=self.ee_link,
            is_calibrated=self.is_calibrated,
            calibration_sha256=self.calibration_sha256,
            serial_number=self.serial_number,
            tcp_offset_mm_rpy=tcp_arr,
            tcp_transform_m=tcp_mat,
        )

    def validate_q(self, q_rad: Sequence[float] | FloatArray) -> FloatArray:
        q = np.asarray(q_rad, dtype=np.float64).reshape(-1)
        if q.shape != (self.dof,) or not np.all(np.isfinite(q)):
            raise ValueError(f"joint vector must contain {self.dof} finite values, got shape {q.shape}")
        return q

    def forward_matrix(self, q_rad: Sequence[float] | FloatArray, *, include_tcp: bool = True) -> FloatArray:
        """Return 4x4 SE(3) end-effector transform in robot base frame (metres)."""

        q = self.validate_q(q_rad)
        pose: FloatArray = np.eye(4, dtype=np.float64)
        for i in range(self.dof):
            pose = pose @ self.origin_transforms[i]
            rot = axis_angle_matrix(self.axes[i], float(q[i]))
            pose[:3, :3] = pose[:3, :3] @ rot
        if include_tcp:
            pose = pose @ self.tcp_transform_m
        return pose

    def forward_pose_m_quat(self, q_rad: Sequence[float] | FloatArray, *, include_tcp: bool = True) -> FloatArray:
        """Return ``[x, y, z, qx, qy, qz, qw]`` in base frame (metres)."""

        return se3_to_pose_m_quat(self.forward_matrix(q_rad, include_tcp=include_tcp))

    def forward_pose_mm_rpy(self, q_rad: Sequence[float] | FloatArray, *, include_tcp: bool = True) -> FloatArray:
        """Return ``[x_mm, y_mm, z_mm, roll, pitch, yaw]`` in base frame (SDK units)."""

        return se3_to_pose_mm_rpy(self.forward_matrix(q_rad, include_tcp=include_tcp))

    def forward_and_jacobian(
        self,
        q_rad: Sequence[float] | FloatArray,
        *,
        include_tcp: bool = True,
    ) -> tuple[FloatArray, FloatArray]:
        """Compute 4x4 FK transform and 6xN base-aligned geometric Jacobian ``[v_m; w_rad]``."""

        q = self.validate_q(q_rad)
        n = self.dof
        joint_positions: FloatArray = np.empty((n, 3), dtype=np.float64)
        joint_axes: FloatArray = np.empty((n, 3), dtype=np.float64)
        pose: FloatArray = np.eye(4, dtype=np.float64)
        for i in range(n):
            pose = pose @ self.origin_transforms[i]
            joint_positions[i] = pose[:3, 3]
            joint_axes[i] = pose[:3, :3] @ self.axes[i]
            rot = axis_angle_matrix(self.axes[i], float(q[i]))
            pose[:3, :3] = pose[:3, :3] @ rot
        if include_tcp:
            pose = pose @ self.tcp_transform_m
        p_ee = pose[:3, 3]
        jac: FloatArray = np.empty((6, n), dtype=np.float64)
        for i in range(n):
            z_i = joint_axes[i]
            jac[:3, i] = np.cross(z_i, p_ee - joint_positions[i])
            jac[3:, i] = z_i
        return pose, jac

    def jacobian(self, q_rad: Sequence[float] | FloatArray, *, include_tcp: bool = True) -> FloatArray:
        """Return 6xN spatial geometric Jacobian ``[v_m; w_rad]`` in base (LOCAL_WORLD_ALIGNED) frame."""

        _, jac = self.forward_and_jacobian(q_rad, include_tcp=include_tcp)
        return jac

    def tcp_target_to_flange(self, target_tcp_se3: FloatArray) -> FloatArray:
        """Strip ``tcp_offset`` from a 4x4 TCP target matrix to obtain the flange target matrix."""

        t = np.asarray(target_tcp_se3, dtype=np.float64).reshape(4, 4)
        rot_t = self.tcp_transform_m[:3, :3].T
        inv_tcp: FloatArray = np.eye(4, dtype=np.float64)
        inv_tcp[:3, :3] = rot_t
        inv_tcp[:3, 3] = -rot_t @ self.tcp_transform_m[:3, 3]
        return np.asarray(t @ inv_tcp, dtype=np.float64)

    def pose_error(self, target_se3: FloatArray, current_se3: FloatArray) -> tuple[float, float]:
        """Return ``(position_error_m, orientation_error_rad)`` between two 4x4 transforms."""

        pos_err = float(np.linalg.norm(target_se3[:3, 3] - current_se3[:3, 3]))
        rot_err = float(rotation_angle(current_se3[:3, :3], target_se3[:3, :3]))
        return pos_err, rot_err
