"""Pinocchio mimic fallback for URDFs where <mimic> precedes its parent joint."""

from __future__ import annotations

import numpy as np
import pytest

from ufactory.robots.assembly import compose_robot_urdf
from ufactory.safety.adapters.pinocchio import PinocchioCollisionBackend


def test_bio_g2_mimic_fallback_and_symmetric_fingers() -> None:
    pytest.importorskip("pinocchio")

    bio_urdf = compose_robot_urdf("xarm6_1305", "bio_gripper_g2")
    backend = PinocchioCollisionBackend(
        bio_urdf,
        joint_names=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
        ee_link="link6",
        passive_joint_positions={"bio_gripper_g2_right_finger_joint": 0.0395},
    )
    assert backend._mimic_fallback is True
    left_id = int(backend.model.getJointId("bio_gripper_g2_left_finger_joint"))
    right_id = int(backend.model.getJointId("bio_gripper_g2_right_finger_joint"))
    left_idx = int(backend.model.joints[left_id].idx_q)
    right_idx = int(backend.model.joints[right_id].idx_q)

    q0 = np.zeros(6, dtype=np.float64)
    q_open = backend.full_q(q0)
    assert q_open[right_idx] == pytest.approx(0.0395, abs=1e-9)
    assert q_open[left_idx] == pytest.approx(-0.0395, abs=1e-9)

    q_closed = backend.full_q(q0, gripper_drive=0.0)
    assert q_closed[right_idx] == pytest.approx(0.0, abs=1e-9)
    assert q_closed[left_idx] == pytest.approx(0.0, abs=1e-9)

    # Verify the two finger frames are symmetric across the bio_gripper_g2_base_link Y midplane.
    base_id = backend.model.getFrameId("bio_gripper_g2_base_link")
    for drive, expected_half_y in ((0.0395, 0.0395), (0.0, 0.0)):
        q_full = backend.full_q(q0, gripper_drive=drive)
        backend.pin.forwardKinematics(backend.model, backend.data, q_full)
        backend.pin.updateFramePlacements(backend.model, backend.data)
        base = backend.data.oMf[base_id]
        left_pos = base.actInv(backend.data.oMf[backend.model.getFrameId("bio_gripper_g2_left_finger")]).translation
        right_pos = base.actInv(backend.data.oMf[backend.model.getFrameId("bio_gripper_g2_right_finger")]).translation
        assert left_pos[1] == pytest.approx(-expected_half_y, abs=1e-6)
        assert right_pos[1] == pytest.approx(expected_half_y, abs=1e-6)


def test_default_g2_keeps_native_pinocchio_mimic_path() -> None:
    pytest.importorskip("pinocchio")

    g2_urdf = compose_robot_urdf("xarm6_1305", "gripper_g2")
    backend = PinocchioCollisionBackend(
        g2_urdf,
        joint_names=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
        ee_link="link6",
        passive_joint_positions={"drive_joint": 0.057210812728026743},
    )
    assert backend._mimic_fallback is False
