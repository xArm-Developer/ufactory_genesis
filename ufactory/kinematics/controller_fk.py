"""Per-unit joint origins identified from the controller's forward kinematics.

The controller's FK is a rigid chain: every joint is a fixed screw axis. With
all other joints at 0, moving one joint shows its axis directly, so the zero
pose plus a few single-joint sweeps identify the whole chain (product of
exponentials). This module turns those axes into schema-v1 calibration joint
origins (``joint<i>: {x, y, z, roll, pitch, yaw}``) for the project URDF:

* joints 1..N-1: the nominal joint frame, minimally rotated onto the
  identified axis, at the foot of the perpendicular from the nominal origin;
* joint N: the controller flange frame at q = 0, rotating about its own z.

The controller puts the flange slightly off its last axis (measured
38-240 um, 8-22 mdeg tilt). Joint origins cannot express that, so the model is
exact whenever the last joint is at 0, and its error grows with |q_N| up to
about twice the offset. The register-8 export uses the same last-joint form.
Every identification is checked against controller FK at random
configurations it did not use.

No SDK import: callers pass ``flange_fk(q) -> 4x4`` (metres, base frame).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from ufactory.kinematics.tcp_offset import pose_to_matrix, rpy_matrix
from ufactory.types import FloatArray

CALIBRATION_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")
SWEEP_ANGLES_RAD = (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5)
SAMPLE_SPAN_RAD = 2.5
VERIFICATION_SAMPLES = 100
VERIFICATION_SEED = 20261126

# Acceptance: the same limits as the FK validation against the controller
# (ufactory.kinematics.validation PASS_POS_MM / PASS_RPY_DEG).
MAX_MODEL_POSITION_ERROR_M = 1.0e-3
MAX_MODEL_ROTATION_ERROR_RAD = math.radians(0.5)
# The controller FK must be a rigid chain; measured <= 1.55 um on four arms.
MAX_RIGID_CHAIN_ERROR_M = 10.0e-6
# Plausibility guard against a wrong URDF, not an accuracy limit: identified
# axes and origins, accumulated along the chain, versus the nominal URDF.
# Measured: four arms <= 0.30 deg / 2.5 mm, xArm7 XS130502F56A0C 1.46 deg / 5.4 mm.
MAX_AXIS_DEVIATION_RAD = math.radians(5.0)
MAX_ORIGIN_DEVIATION_M = 20.0e-3


class CalibrationIdentificationError(ValueError):
    """The controller FK cannot be turned into trustworthy joint origins."""


@dataclass(frozen=True)
class NominalChain:
    """Revolute joint chain ``joint1..N`` of a URDF, in the base link frame."""

    joint_names: tuple[str, ...]
    origins: FloatArray  # (N, 6): xyz (m) + rpy (rad), URDF convention
    axes: FloatArray  # (N, 3): unit axis in each joint frame
    lower: FloatArray
    upper: FloatArray
    ee_link: str


@dataclass(frozen=True)
class ScrewAxis:
    """Joint axis at q = 0 in the base frame: unit direction and closest point to the origin."""

    direction: FloatArray
    point: FloatArray


@dataclass(frozen=True)
class ControllerFkIdentification:
    """Joint origins plus the evidence that they reproduce the controller."""

    origins: FloatArray
    identification_samples: int
    verification_samples: int
    rigid_chain_error_m: float
    mean_position_error_m: float
    max_position_error_m: float
    max_rotation_error_rad: float
    flange_axis_offset_m: float
    flange_axis_tilt_rad: float
    max_axis_deviation_rad: float
    max_origin_deviation_m: float

    def joints(self, joint_names: Sequence[str]) -> dict[str, dict[str, float]]:
        return {
            name: {field: float(value) for field, value in zip(CALIBRATION_FIELDS, row, strict=True)}
            for name, row in zip(joint_names, self.origins, strict=True)
        }


# ---------------------------------------------------------------- geometry


def _transform(
    rotation: FloatArray | None = None, translation: Sequence[float] | FloatArray = (0.0, 0.0, 0.0)
) -> FloatArray:
    matrix: FloatArray = np.eye(4, dtype=np.float64)
    if rotation is not None:
        matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def _skew(vector: FloatArray) -> FloatArray:
    x, y, z = (float(v) for v in vector)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def axis_angle_matrix(axis: FloatArray, angle: float) -> FloatArray:
    """Rotation by ``angle`` about the unit vector ``axis`` (Rodrigues)."""

    k = _skew(np.asarray(axis, dtype=np.float64))
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def rotation_vector(rotation: FloatArray) -> FloatArray:
    """Inverse of ``axis_angle_matrix`` for angles below pi."""

    r = np.asarray(rotation, dtype=np.float64)
    vee = 0.5 * np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    sin_angle = float(np.linalg.norm(vee))
    angle = math.atan2(sin_angle, (float(np.trace(r)) - 1.0) / 2.0)
    if sin_angle < 1e-15:
        return np.zeros(3)
    return vee / sin_angle * angle


def rotation_angle(a: FloatArray, b: FloatArray) -> float:
    """Angle (rad) of the rotation taking ``a`` to ``b``."""

    return float(np.linalg.norm(rotation_vector(np.asarray(a).T @ np.asarray(b))))


def _vector_angle(a: FloatArray, b: FloatArray) -> float:
    return math.atan2(float(np.linalg.norm(np.cross(a, b))), float(np.dot(a, b)))


def _minimal_rotation(a: FloatArray, b: FloatArray) -> FloatArray:
    """Smallest rotation taking unit vector ``a`` onto unit vector ``b``."""

    cross = np.cross(a, b)
    norm = float(np.linalg.norm(cross))
    if norm < 1e-15:
        if float(np.dot(a, b)) > 0.0:
            return np.eye(3)
        raise CalibrationIdentificationError("identified joint axis is opposite to the nominal axis")
    return axis_angle_matrix(cross / norm, math.atan2(norm, float(np.dot(a, b))))


def matrix_rpy(rotation: FloatArray) -> tuple[float, float, float]:
    """URDF roll/pitch/yaw of ``rotation`` (R = Rz(yaw) Ry(pitch) Rx(roll)), exact at gimbal lock."""

    r = np.asarray(rotation, dtype=np.float64)
    cos_pitch = math.hypot(float(r[0, 0]), float(r[1, 0]))
    pitch = math.atan2(-float(r[2, 0]), cos_pitch)
    if cos_pitch > 1e-9:
        return math.atan2(float(r[2, 1]), float(r[2, 2])), pitch, math.atan2(float(r[1, 0]), float(r[0, 0]))
    # pitch = +-pi/2: only roll - yaw (or roll + yaw) is defined; take yaw = 0.
    return math.atan2(-float(r[1, 2]), float(r[1, 1])), pitch, 0.0


def chain_pose(origins: FloatArray, axes: FloatArray, q: Sequence[float] | FloatArray) -> FloatArray:
    """URDF chain FK: per joint, fixed origin then rotation ``q_i`` about the local axis."""

    pose = np.eye(4)
    for row, axis, angle in zip(np.asarray(origins), np.asarray(axes), np.asarray(q, dtype=np.float64), strict=True):
        pose = (
            pose
            @ _transform(rpy_matrix(float(row[3]), float(row[4]), float(row[5])), row[:3])
            @ _transform(axis_angle_matrix(axis, float(angle)))
        )
    return pose


def screw_pose(zero_pose: FloatArray, axes: Sequence[ScrewAxis], q: Sequence[float] | FloatArray) -> FloatArray:
    """Product-of-exponentials FK of identified screw axes."""

    pose = np.eye(4)
    for axis, angle in zip(axes, np.asarray(q, dtype=np.float64), strict=True):
        rotation = axis_angle_matrix(axis.direction, float(angle))
        pose = pose @ _transform(rotation, (np.eye(3) - rotation) @ axis.point)
    return pose @ zero_pose


def pose_errors(model: Sequence[FloatArray], reference: Sequence[FloatArray]) -> tuple[FloatArray, FloatArray]:
    """Position (m) and rotation (rad) error per sample."""

    position = np.array([np.linalg.norm(a[:3, 3] - b[:3, 3]) for a, b in zip(model, reference, strict=True)])
    rotation = np.array([rotation_angle(a[:3, :3], b[:3, :3]) for a, b in zip(model, reference, strict=True)])
    return position, rotation


# ---------------------------------------------------------------- nominal model


def _floats(text: str | None, default: tuple[float, float, float]) -> FloatArray:
    return np.asarray(default if text is None else [float(v) for v in text.split()], dtype=np.float64)


def load_nominal_chain(urdf_path: str | Path, joint_names: Sequence[str]) -> NominalChain:
    """Read ``joint_names`` as a direct revolute chain (each joint's parent is the previous joint's child)."""

    root = ET.parse(str(urdf_path)).getroot()
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    origins, axes, lower, upper = [], [], [], []
    previous_child: str | None = None
    for name in joint_names:
        joint = joints.get(name)
        if joint is None or joint.get("type") not in {"revolute", "continuous"}:
            raise CalibrationIdentificationError(f"{urdf_path}: {name} is not a revolute joint")
        parent, child = joint.find("parent"), joint.find("child")
        if parent is None or child is None:
            raise CalibrationIdentificationError(f"{urdf_path}: {name} lacks parent or child")
        if previous_child is not None and parent.get("link") != previous_child:
            raise CalibrationIdentificationError(f"{urdf_path}: {name} does not follow the previous joint directly")
        previous_child = child.get("link")
        origin = joint.find("origin")
        xyz = _floats(None if origin is None else origin.get("xyz"), (0.0, 0.0, 0.0))
        rpy = _floats(None if origin is None else origin.get("rpy"), (0.0, 0.0, 0.0))
        axis_element = joint.find("axis")
        axis = _floats(None if axis_element is None else axis_element.get("xyz"), (1.0, 0.0, 0.0))
        limit = joint.find("limit")
        lo = -math.pi if limit is None or limit.get("lower") is None else float(limit.get("lower", ""))
        hi = math.pi if limit is None or limit.get("upper") is None else float(limit.get("upper", ""))
        origins.append([*xyz, *rpy])
        axes.append(axis / np.linalg.norm(axis))
        lower.append(lo)
        upper.append(hi)
    assert previous_child is not None
    return NominalChain(
        joint_names=tuple(joint_names),
        origins=np.asarray(origins, dtype=np.float64),
        axes=np.asarray(axes, dtype=np.float64),
        lower=np.asarray(lower, dtype=np.float64),
        upper=np.asarray(upper, dtype=np.float64),
        ee_link=previous_child,
    )


# ---------------------------------------------------------------- sampling


def sweep_angles(lower: float, upper: float) -> tuple[float, ...]:
    """Non-zero angles for one joint's sweep, within its limits and +-SAMPLE_SPAN_RAD."""

    lo, hi = max(lower, -SAMPLE_SPAN_RAD), min(upper, SAMPLE_SPAN_RAD)
    values = [value for value in SWEEP_ANGLES_RAD if lo <= value <= hi]
    if len(values) < 4:  # one-sided joint (e.g. xArm5 J3): spread over its range
        values = [float(value) for value in np.linspace(lo, hi, 6) if abs(value) > 0.04]
    if len(values) < 2:
        raise CalibrationIdentificationError(f"joint range [{lower:.3f}, {upper:.3f}] rad is too small to sweep")
    return tuple(values)


def verification_configurations(chain: NominalChain, count: int, seed: int) -> FloatArray:
    lo = np.maximum(chain.lower, -SAMPLE_SPAN_RAD)
    hi = np.minimum(chain.upper, SAMPLE_SPAN_RAD)
    return np.random.default_rng(seed).uniform(lo, hi, size=(count, len(chain.joint_names)))


# ---------------------------------------------------------------- identification


def identify_screw_axes(
    zero_pose: FloatArray,
    sweeps: Sequence[Sequence[tuple[float, FloatArray]]],
) -> list[ScrewAxis]:
    """Screw axis of each joint from ``sweeps[j] = [(angle, controller pose with only joint j moved), ...]``."""

    axes: list[ScrewAxis] = []
    zero_inverse = np.linalg.inv(zero_pose)
    for samples in sweeps:
        directions, rows, rhs = [], [], []
        for angle, pose in samples:
            motion = pose @ zero_inverse  # = exp([S_j] angle)
            directions.append(rotation_vector(motion[:3, :3]) / angle)
            rows.append(np.eye(3) - motion[:3, :3])
            rhs.append(motion[:3, 3])
        direction = np.mean(directions, axis=0)
        direction /= np.linalg.norm(direction)
        point = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)[0]
        axes.append(ScrewAxis(direction=direction, point=point - float(point @ direction) * direction))
    return axes


def joint_origins_from_axes(
    chain: NominalChain,
    zero_pose: FloatArray,
    axes: Sequence[ScrewAxis],
) -> tuple[FloatArray, float, float]:
    """Joint origins for identified axes; returns (origins, max axis deviation rad, max origin deviation m)."""

    count = len(chain.joint_names)
    nominal_frames, frame = [], np.eye(4)
    for row in chain.origins:
        frame = frame @ _transform(rpy_matrix(float(row[3]), float(row[4]), float(row[5])), row[:3])
        nominal_frames.append(frame.copy())
    frames: list[FloatArray] = []
    max_axis = max_origin = 0.0
    for index, (nominal, axis) in enumerate(zip(nominal_frames, axes, strict=True)):
        nominal_axis = nominal[:3, :3] @ chain.axes[index]
        axis_deviation = _vector_angle(nominal_axis, axis.direction)
        if index < count - 1:
            rotation = _minimal_rotation(nominal_axis, axis.direction) @ nominal[:3, :3]
            offset = nominal[:3, 3] - axis.point
            position = axis.point + float(offset @ axis.direction) * axis.direction
            origin_deviation = float(np.linalg.norm(position - nominal[:3, 3]))
        else:
            rotation, position = zero_pose[:3, :3], zero_pose[:3, 3]
            origin_deviation = float(np.linalg.norm(position - nominal[:3, 3]))
            axis_deviation = max(axis_deviation, rotation_angle(nominal[:3, :3], rotation))
        if axis_deviation > MAX_AXIS_DEVIATION_RAD or origin_deviation > MAX_ORIGIN_DEVIATION_M:
            raise CalibrationIdentificationError(
                f"{chain.joint_names[index]}: controller axis is {math.degrees(axis_deviation):.3f} deg / "
                f"{origin_deviation * 1e3:.3f} mm from the nominal URDF (limits "
                f"{math.degrees(MAX_AXIS_DEVIATION_RAD):.1f} deg / {MAX_ORIGIN_DEVIATION_M * 1e3:.0f} mm); "
                "is the controller the same robot model as the URDF?"
            )
        max_axis, max_origin = max(max_axis, axis_deviation), max(max_origin, origin_deviation)
        frames.append(_transform(rotation, position))
    origins, previous = [], np.eye(4)
    for frame in frames:
        relative = np.linalg.inv(previous) @ frame
        origins.append([*relative[:3, 3], *matrix_rpy(relative[:3, :3])])
        previous = frame
    result = np.asarray(origins, dtype=np.float64)
    rebuilt = chain_pose(result, chain.axes, np.zeros(count))
    if (
        float(np.linalg.norm(rebuilt[:3, 3] - zero_pose[:3, 3])) > 1e-9
        or rotation_angle(rebuilt[:3, :3], zero_pose[:3, :3]) > 1e-9
    ):
        raise CalibrationIdentificationError("joint origins do not rebuild the controller zero pose")
    return result, max_axis, max_origin


def identify_from_controller_fk(
    flange_fk: Callable[[FloatArray], FloatArray],
    chain: NominalChain,
    *,
    verification_samples: int = VERIFICATION_SAMPLES,
    seed: int = VERIFICATION_SEED,
) -> ControllerFkIdentification:
    """Identify joint origins from controller FK and verify them on held-out configurations.

    Raises ``CalibrationIdentificationError`` when the controller FK is not a
    rigid chain, the geometry is far from the nominal URDF, or the result
    misses the FK validation limits on the held-out configurations.
    """

    count = len(chain.joint_names)
    zero_pose = np.asarray(flange_fk(np.zeros(count)), dtype=np.float64)
    sweeps: list[list[tuple[float, FloatArray]]] = []
    for index in range(count):
        samples = []
        for angle in sweep_angles(float(chain.lower[index]), float(chain.upper[index])):
            q = np.zeros(count)
            q[index] = angle
            samples.append((angle, np.asarray(flange_fk(q), dtype=np.float64)))
        sweeps.append(samples)
    axes = identify_screw_axes(zero_pose, sweeps)
    origins, max_axis, max_origin = joint_origins_from_axes(chain, zero_pose, axes)

    configurations = verification_configurations(chain, verification_samples, seed)
    controller = [np.asarray(flange_fk(q), dtype=np.float64) for q in configurations]
    rigid_position, rigid_rotation = pose_errors([screw_pose(zero_pose, axes, q) for q in configurations], controller)
    position, rotation = pose_errors([chain_pose(origins, chain.axes, q) for q in configurations], controller)

    last = axes[-1]
    flange_offset = zero_pose[:3, 3] - last.point
    flange_offset -= float(flange_offset @ last.direction) * last.direction
    result = ControllerFkIdentification(
        origins=origins,
        identification_samples=1 + sum(len(samples) for samples in sweeps),
        verification_samples=len(configurations),
        rigid_chain_error_m=float(rigid_position.max(initial=0.0)),
        mean_position_error_m=float(position.mean()) if position.size else 0.0,
        max_position_error_m=float(position.max(initial=0.0)),
        max_rotation_error_rad=float(rotation.max(initial=0.0)),
        flange_axis_offset_m=float(np.linalg.norm(flange_offset)),
        flange_axis_tilt_rad=_vector_angle(zero_pose[:3, :3] @ chain.axes[-1], last.direction),
        max_axis_deviation_rad=max_axis,
        max_origin_deviation_m=max_origin,
    )
    if result.rigid_chain_error_m > MAX_RIGID_CHAIN_ERROR_M:
        raise CalibrationIdentificationError(
            f"controller FK is not a rigid chain: screw model misses held-out samples by "
            f"{result.rigid_chain_error_m * 1e6:.2f} um (limit {MAX_RIGID_CHAIN_ERROR_M * 1e6:.0f} um, "
            f"rotation {math.degrees(float(rigid_rotation.max(initial=0.0))) * 1e3:.3f} mdeg)"
        )
    if (
        result.max_position_error_m > MAX_MODEL_POSITION_ERROR_M
        or result.max_rotation_error_rad > MAX_MODEL_ROTATION_ERROR_RAD
    ):
        raise CalibrationIdentificationError(
            f"joint origins miss controller FK by {result.max_position_error_m * 1e3:.4f} mm / "
            f"{math.degrees(result.max_rotation_error_rad):.4f} deg (limits {MAX_MODEL_POSITION_ERROR_M * 1e3:.1f} mm / "
            f"{math.degrees(MAX_MODEL_ROTATION_ERROR_RAD):.1f} deg); flange is "
            f"{result.flange_axis_offset_m * 1e6:.0f} um off the last axis"
        )
    return result


# ---------------------------------------------------------------- SDK adapter


def controller_flange_fk(arm: object, tcp_offset: Sequence[float] | FloatArray) -> Callable[[FloatArray], FloatArray]:
    """``q -> flange pose`` from ``arm.get_forward_kinematics`` with the controller TCP offset removed.

    ``tcp_offset`` is ``[x, y, z, roll, pitch, yaw]`` (mm, rad), as read from the
    controller; the SDK returns the TCP pose ``T_flange @ T_offset``.
    """

    remove_offset = np.linalg.inv(pose_to_matrix(tcp_offset))
    get_fk = getattr(arm, "get_forward_kinematics")

    def flange_fk(q: FloatArray) -> FloatArray:
        code, pose = get_fk([float(v) for v in q], input_is_radian=True, return_is_radian=True)
        values = np.asarray(pose, dtype=np.float64).reshape(-1)
        if int(code) != 0 or values.shape != (6,) or not np.all(np.isfinite(values)):
            raise RuntimeError(f"controller FK failed (SDK code={code})")
        flange = pose_to_matrix(values) @ remove_offset
        flange[:3, 3] /= 1000.0
        return flange

    return flange_fk
