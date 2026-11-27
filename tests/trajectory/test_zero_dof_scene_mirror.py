"""Tests for 0-DOF end-effector (vacuum gripper / bare arm) support in scene and mirror."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ufactory.config.models import GripperProfile
from ufactory.robots.runtime import (
    LITE6_VACUUM_GRIPPER_PARAMS,
    VACUUM_GRIPPER_PARAMS,
    get_robot_runtime_profile,
)


def test_zero_dof_gripper_profile_and_runtime_params() -> None:
    profile = GripperProfile(
        adapter="vacuum_gripper",
        drive_joint="",
        all_joint_names=(),
        finger_link_names=(),
        open_drive=0.0,
        closed_drive=0.0,
        open_gap_m=0.0,
        closed_gap_m=0.0,
        kp=0.0,
        kv=0.0,
        force_lower_n=0.0,
        force_upper_n=0.0,
        damping=0.0,
        frictionloss=0.0,
        real_command=True,
        feedback=True,
        closed_loop=False,
        allowed_contact_links=frozenset({"xarm_vacuum_gripper_link"}),
        tool_tip_offset_z_m=0.126,
        finger_pad_below_center_m=0.0,
        finger_close_descent_m=0.0,
        grasp_table_clearance_m=0.010,
        grasp_height_extra_m=0.0,
        tcp_offset=(0.0, 0.0, 126.0, 0.0, 0.0, 0.0),
    )
    assert profile.has_actuated_joints is False
    assert profile.tcp_offset == (0.0, 0.0, 126.0, 0.0, 0.0, 0.0)

    assert VACUUM_GRIPPER_PARAMS.has_actuated_joints is False
    assert VACUUM_GRIPPER_PARAMS.has_finger_links is False
    assert LITE6_VACUUM_GRIPPER_PARAMS.has_actuated_joints is False
    assert LITE6_VACUUM_GRIPPER_PARAMS.has_finger_links is False

    rt_vac = get_robot_runtime_profile("xarm6", gripper="vacuum_gripper")
    assert rt_vac.gripper == VACUUM_GRIPPER_PARAMS
    rt_bare = get_robot_runtime_profile("xarm6", gripper="bare")
    assert rt_bare.gripper is None


def test_zero_dof_drive_gap_and_tool_center_fallbacks() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("genesis")
    from ufactory.trajectory.mirror_executor import KinematicCarryTracker, gap_m_from_drive
    from ufactory.trajectory.scene import drive_for_gap_m, gripper_dof_multipliers

    assert drive_for_gap_m(0.05, VACUUM_GRIPPER_PARAMS) == 0.0
    assert drive_for_gap_m(0.05, None) == 0.0
    assert gap_m_from_drive(0.0, VACUUM_GRIPPER_PARAMS) == 0.0
    assert gap_m_from_drive(0.0, None) == 0.0
    assert gripper_dof_multipliers(VACUUM_GRIPPER_PARAMS) == ()
    assert gripper_dof_multipliers(None) == ()

    # Test _tool_center fallback to tool_tip_link, then ik_link + rotated tcp_offset_flange_m.
    obj = SimpleNamespace(
        get_pos=lambda: torch.tensor([[0.3, 0.0, 0.415]], dtype=torch.float32),
        get_quat=lambda: torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
    )
    tip = SimpleNamespace(get_pos=lambda: torch.tensor([[0.3, 0.0, 0.50]], dtype=torch.float32))
    ik = SimpleNamespace(
        get_pos=lambda: torch.tensor([[0.3, 0.0, 0.626]], dtype=torch.float32),
        get_quat=lambda: torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32),  # Rx(pi) gripper-down WXYZ
    )

    mirror_tip = SimpleNamespace(
        ctx=SimpleNamespace(left_finger=None, right_finger=None, tool_tip_link=tip, ik_link=ik, obj=obj),
        rate=50.0,
    )
    tracker_tip = KinematicCarryTracker(mirror_tip, grasp_gap_m=0.0)
    np.testing.assert_allclose(tracker_tip._tool_center().numpy(), [0.3, 0.0, 0.50], atol=1e-6)

    mirror_offset = SimpleNamespace(
        ctx=SimpleNamespace(
            left_finger=None,
            right_finger=None,
            tool_tip_link=None,
            ik_link=ik,
            tcp_offset_flange_m=(0.0, 0.0, 0.126),
            obj=obj,
        ),
        rate=50.0,
    )
    tracker_offset = KinematicCarryTracker(mirror_offset, grasp_gap_m=0.0)
    np.testing.assert_allclose(tracker_offset._tool_center().numpy(), [0.3, 0.0, 0.50], atol=1e-6)


@pytest.mark.gpu
@pytest.mark.parametrize("gripper", ["vacuum_gripper", "bare"])
def test_build_scene_and_mirror_support_zero_dof_end_effectors(gripper: str) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("genesis")
    from ufactory.config import load_runtime_config
    from ufactory.simulation import GenesisRuntimeManager
    from ufactory.trajectory.mirror_executor import KinematicCarryTracker, TrajKinematicMirror
    from ufactory.trajectory.scene import build_scene
    from ufactory.trajectory.segments import JointLimits, Program, make_gripper, make_movel

    cfg = load_runtime_config("xarm6")
    with GenesisRuntimeManager(cfg.simulation):
        ctx = build_scene(robot_key="xarm6", gripper=gripper, show_viewer=False, calibrated_g2=False)
        assert ctx.left_finger is None
        assert ctx.right_finger is None
        assert ctx.gripper_dof_idx == []
        assert ctx.all_gripper_dof_idx == []
        if gripper == "vacuum_gripper":
            assert ctx.tcp_offset_flange_m == pytest.approx((0.0, 0.0, 0.126), abs=1e-6)
            assert ctx.finger_z_offset == pytest.approx(0.126, abs=2e-3)
        else:
            assert ctx.gripper is None
            assert ctx.tcp_offset_flange_m == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
            assert ctx.finger_z_offset == pytest.approx(0.0, abs=1e-6)

        limits = JointLimits(0.35, 2.0, 0.14, 0.8)
        seg_move = make_movel(
            np.array(ctx.home_pos_base, dtype=np.float64),
            np.array([ctx.home_pos_base[0], ctx.home_pos_base[1], ctx.home_pos_base[2] - 0.03], dtype=np.float64),
            rate=50.0,
            limits=limits,
            label="descend",
        )
        seg_grip = make_gripper(0.0, 0.0, rate=50.0, duration_s=0.04, label="grip")
        program = Program(rate=50.0, robot_key="xarm6_1305", limits=limits, segments=[seg_move, seg_grip])

        mirror = TrajKinematicMirror(ctx, program)
        mirror.prime_to_home()
        tracker = KinematicCarryTracker(mirror, grasp_gap_m=0.0, attach_dist_m=0.0)
        for seg in program.segments:
            _, n = seg.samples(program.rate)
            for i in range(n):
                tracker.on_tick(seg, i)
        assert tracker.attached is False
        flange_z = float(ctx.ik_link.get_pos()[0][2])
        tool_z = float(tracker._tool_center()[2])
        expected_dz = 0.126 if gripper == "vacuum_gripper" else 0.0
        assert (flange_z - tool_z) == pytest.approx(expected_dz, abs=2e-3)
