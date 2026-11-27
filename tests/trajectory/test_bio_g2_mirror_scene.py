"""Scene builder and kinematic mirror helpers for Bio Gripper G2."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("genesis")

from ufactory.config import load_runtime_config
from ufactory.robots.assembly import compose_robot_urdf
from ufactory.robots.runtime import BIO_GRIPPER_G2_GRIPPER_PARAMS, G2_GRIPPER_PARAMS, get_robot_runtime_profile
from ufactory.simulation import GenesisRuntimeManager
from ufactory.trajectory.mirror_executor import (
    KinematicCarryTracker,
    TrajKinematicMirror,
    _set_kinematic_pose,
    gap_m_from_drive,
)
from ufactory.trajectory.scene import _robot_urdf_for_visual_model, drive_for_gap_m, gripper_dof_multipliers


class _RecordingRobot:
    def __init__(self) -> None:
        self.calls: list[tuple[list[float], list[int], bool]] = []

    def set_dofs_position(self, values, dof_idx, *, zero_velocity: bool = False) -> None:
        self.calls.append(([float(v) for v in values], list(dof_idx), bool(zero_velocity)))


def test_gripper_dof_multipliers_bio_and_g2() -> None:
    assert gripper_dof_multipliers(BIO_GRIPPER_G2_GRIPPER_PARAMS) == (1.0, -1.0)
    assert gripper_dof_multipliers(G2_GRIPPER_PARAMS) == (1.0,) * len(G2_GRIPPER_PARAMS.all_joint_names)
    assert gripper_dof_multipliers(None) == ()


def test_robot_urdf_for_visual_model_supports_bio_gripper_g2() -> None:
    runtime = get_robot_runtime_profile("xarm6", gripper="bio_gripper_g2")
    urdf_path, use_glb = _robot_urdf_for_visual_model(runtime, "glb")
    assert use_glb is True
    assert urdf_path == compose_robot_urdf("xarm6_1305", "bio_gripper_g2")
    assert Path(urdf_path).is_file()


def test_set_kinematic_pose_applies_grip_multipliers() -> None:
    q_arm = [0.1, -0.2, 0.3, 0.0, 0.4, 0.5]
    arm_idx = [0, 1, 2, 3, 4, 5]
    grip_idx = [6, 7]
    drive = 0.02

    robot_with_mult = _RecordingRobot()
    _set_kinematic_pose(robot_with_mult, q_arm, arm_idx, drive, grip_idx, grip_multipliers=(1.0, -1.0))
    assert len(robot_with_mult.calls) == 2
    assert robot_with_mult.calls[0] == (pytest.approx(q_arm), arm_idx, True)
    assert robot_with_mult.calls[1] == (pytest.approx([drive, -drive]), grip_idx, True)

    robot_without_mult = _RecordingRobot()
    _set_kinematic_pose(robot_without_mult, q_arm, arm_idx, drive, grip_idx, grip_multipliers=None)
    assert len(robot_without_mult.calls) == 2
    assert robot_without_mult.calls[1] == (pytest.approx([drive, drive]), grip_idx, True)


@pytest.mark.parametrize(
    ("gap_m", "expected_drive"),
    [
        (0.071, 0.0),
        (0.150, 0.0395),
        (0.1105, 0.01975),
    ],
)
def test_bio_g2_drive_gap_roundtrip(gap_m: float, expected_drive: float) -> None:
    drive = drive_for_gap_m(gap_m, BIO_GRIPPER_G2_GRIPPER_PARAMS)
    assert drive == pytest.approx(expected_drive, abs=1e-9)
    assert gap_m_from_drive(drive, BIO_GRIPPER_G2_GRIPPER_PARAMS) == pytest.approx(gap_m, abs=1e-9)


@pytest.mark.gpu
def test_bio_g2_headless_kinematic_mirror_symmetry_and_empty_grasp() -> None:
    cfg = load_runtime_config("xarm6", gripper="bio_gripper_g2")
    module = importlib.import_module("ufactory.cli.pick_place")

    with GenesisRuntimeManager(cfg.simulation):
        program, ctx, _q_home = module._compile_servo_j_program(cfg, calibration=None)
        mirror = TrajKinematicMirror(ctx, program)
        mirror.prime_to_home()
        tracker = KinematicCarryTracker(
            mirror,
            grasp_gap_m=module._close_gap_m(cfg),
            attach_dist_m=module._mirror_attach_dist_m(cfg),
            grasp_segment_label="grip",
            release_segment_label="release",
            approach_freeze_labels=("descend",),
        )

        obj0 = ctx.obj.get_pos()[0].clone()
        saw_grip_mid = False
        saw_grip_end = False
        saw_release_end = False

        for seg in program.segments:
            _, n = seg.samples(program.rate)
            for i in range(n):
                tracker.on_tick(seg, i)
                if seg.label == "grip" and i == n // 2:
                    mid_q = ctx.robot.get_dofs_position(ctx.all_gripper_dof_idx).detach().cpu().numpy().reshape(-1)
                    assert mid_q[0] > 0.005
                    assert mid_q[1] == pytest.approx(-mid_q[0], abs=1e-6)
                    saw_grip_mid = True
                if seg.label == "grip" and i == n - 1:
                    grip_q = ctx.robot.get_dofs_position(ctx.all_gripper_dof_idx).detach().cpu().numpy().reshape(-1)
                    assert grip_q[0] == pytest.approx(0.0, abs=1e-6)
                    assert grip_q[1] == pytest.approx(-grip_q[0], abs=1e-6)
                    assert gap_m_from_drive(mirror.grip_drive, ctx.gripper) == pytest.approx(0.071, abs=1e-6)
                    assert tracker.attached is False
                    saw_grip_end = True
                if seg.label == "release" and i == n - 1:
                    release_q = ctx.robot.get_dofs_position(ctx.all_gripper_dof_idx).detach().cpu().numpy().reshape(-1)
                    assert release_q[0] == pytest.approx(0.0395, abs=1e-6)
                    assert release_q[1] == pytest.approx(-0.0395, abs=1e-6)
                    assert tracker.attached is False
                    saw_release_end = True

        assert saw_grip_mid and saw_grip_end and saw_release_end
        obj_end = ctx.obj.get_pos()[0]
        assert float(torch.linalg.norm(obj_end - obj0)) < 1e-3
