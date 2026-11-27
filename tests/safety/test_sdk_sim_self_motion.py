"""SDK-simulation comparison of redundant arms: self-motion is accepted, end-effector disagreement is not."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ufactory.config import load_runtime_config
from ufactory.safety import CollisionResult
from ufactory.safety.sdk_sim import _quaternion_angle, validate_sdk_simulation


class SelfMotionKin:
    """FK double: only q1 + q2 moves the end effector (q1 - q2 is self-motion); q7 turns the flange about z."""

    def __init__(self, *, with_orientation: bool = True):
        self.with_orientation = with_orientation

    def forward(self, q):
        q = np.asarray(q, dtype=np.float64)
        xyz = (0.3 + 0.1 * (q[0] + q[1]), 0.0, 0.3 + 0.1 * q[2])
        if not self.with_orientation:
            return np.asarray(xyz)
        half = 0.5 * q[-1]
        return np.asarray((*xyz, 0.0, 0.0, math.sin(half), math.cos(half)))

    def inverse(self, pose, seed):
        return np.asarray(seed, dtype=np.float64)


class Clear:
    def check(self, q, *, stage, gripper_drive=None):
        return CollisionResult(False, 1.0, "link1", "table", environment=True)

    def check_all(self, q, *, stage, gripper_drive=None):
        return (self.check(q, stage=stage),)


def _validate(firmware_offset, *, dof=7, kinematics=None, limits=(-3.0, 3.0)):
    config = load_runtime_config("xarm7" if dof == 7 else "xarm6")
    shadow = np.zeros((3, dof))
    firmware = shadow + np.asarray(firmware_offset, dtype=np.float64)
    return validate_sdk_simulation(
        robot_key=config.robot.key,
        serial_number="XS130502XXXXXX",
        program_sha256="p" * 64,
        config_sha256=config.sha256,
        shadow_joint_stream_rad=shadow,
        firmware_joint_stream_rad=firmware,
        stages=("a", "b", "c"),
        policy=config.safety,
        kinematics=kinematics or SelfMotionKin(),
        collision=Clear(),
        joint_lower_rad=None if limits is None else np.full(dof, limits[0]),
        joint_upper_rad=None if limits is None else np.full(dof, limits[1]),
    )


ELBOW = [0.17, -0.17, 0.0, 0.0, 0.0, 0.0, 0.0]  # 10 deg apart, same end effector


def test_redundant_arm_accepts_a_different_elbow_when_the_end_effector_matches():
    evidence = _validate(ELBOW)

    assert evidence.passed, evidence.failures
    assert evidence.self_motion_allowed
    assert evidence.max_joint_error_rad == pytest.approx(0.17)
    assert evidence.max_ee_error_m == pytest.approx(0.0, abs=1e-12)
    assert evidence.max_ee_rotation_error_rad == pytest.approx(0.0, abs=1e-12)


def test_redundant_arm_rejects_end_effector_position_or_rotation_mismatch():
    moved = _validate([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 10 mm along x
    turned = _validate([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06])  # flange only: position unchanged

    assert not moved.passed
    assert any("FK error" in failure for failure in moved.failures)
    assert not turned.passed
    assert turned.max_ee_error_m == pytest.approx(0.0, abs=1e-12)
    assert turned.max_ee_rotation_error_rad == pytest.approx(0.06)
    assert any("rotation error" in failure for failure in turned.failures)


def test_redundant_arm_self_motion_needs_joint_limits_and_stays_inside_them():
    without_limits = _validate(ELBOW, limits=None)
    near_limit = _validate(ELBOW, limits=(-0.18, 0.18))  # 0.17 is inside 0.18 but not inside the 0.02 margin

    assert not without_limits.passed
    assert any("joint limits are required" in failure for failure in without_limits.failures)
    assert not near_limit.passed
    assert any("exceeds joint 1 limit" in failure for failure in near_limit.failures)


def test_redundant_arm_self_motion_needs_fk_orientation():
    evidence = _validate(ELBOW, kinematics=SelfMotionKin(with_orientation=False))

    assert not evidence.passed
    assert any("xyz+quaternion" in failure for failure in evidence.failures)


def test_non_redundant_arm_still_compares_joint_by_joint():
    evidence = _validate(ELBOW[:6], dof=6)

    assert not evidence.self_motion_allowed
    assert not evidence.passed
    assert any("joint error" in failure for failure in evidence.failures)


def test_quaternion_angle_is_the_rotation_angle():
    angle = 0.1
    q = np.asarray((0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)))
    identity = np.asarray((0.0, 0.0, 0.0, 1.0))

    assert _quaternion_angle(identity, q) == pytest.approx(angle)
    assert _quaternion_angle(identity, -q) == pytest.approx(angle)
    assert _quaternion_angle(q, q) == pytest.approx(0.0, abs=1e-15)
