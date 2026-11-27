"""Tests for host-side IK compilation of Cartesian trajectory programs."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ufactory.trajectory.ik import compile_cartesian_program_to_joint_stream
from ufactory.trajectory.segments import JointLimits, Program, make_gripper, make_movel


def test_compile_cartesian_program_to_joint_stream_preserves_ticks_labels_and_uses_continuous_seed(monkeypatch):
    calls: list[np.ndarray | None] = []

    def fake_solve(_ctx, xyz_base, *, init_qpos, damping=None):
        init_copy = None if init_qpos is None else np.asarray(init_qpos, dtype=np.float64).copy()
        calls.append((init_copy, damping))
        xyz = np.asarray(xyz_base, dtype=np.float64).reshape(3)
        return np.array([xyz[0], xyz[1], xyz[2], len(calls) * 0.01, 0.0, 0.0], dtype=np.float64)

    monkeypatch.setattr("ufactory.trajectory.ik._solve_base_xyz", fake_solve)
    ctx = SimpleNamespace(
        robot_key="xarm6_1305",
        robot_urdf_path="/tmp/xarm6_g2.urdf",
        kinematics_yaml_path="/tmp/xarm6_kinematics_F56A14.yaml",
        kinematics_suffix="F56A14",
    )
    limits = JointLimits(0.35, 2.0, 0.14, 0.8)
    move_seg = make_movel(
        np.array([0.30, 0.0, 0.30]),
        np.array([0.30, 0.0, 0.20]),
        rate=50.0,
        limits=limits,
        label="descend",
    )
    grip_seg = make_gripper(
        0.084,
        0.022,
        rate=50.0,
        duration_s=0.02,
        label="grip",
    )
    source = Program(
        rate=50.0,
        robot_key="xarm6_1305",
        limits=limits,
        metadata={"kind": "mixed"},
        segments=[move_seg, grip_seg],
    )
    _move_samples, move_ticks = move_seg.samples(source.rate)

    compiled = compile_cartesian_program_to_joint_stream(source, ctx)

    assert compiled.metadata["kind"] == "joint-from-cartesian-ik"
    assert compiled.metadata["ik_compiled_movel_segments"] == 1
    assert compiled.metadata["ik_timing_policy"] == "joint-lspb-retime"
    assert compiled.metadata["ik_damping"] == 0.01
    assert compiled.metadata["ik_kinematics_suffix"] == "F56A14"
    assert [seg.kind for seg in compiled.segments] == ["movej", "gripper"]
    assert [seg.label for seg in compiled.segments] == ["descend", "grip"]

    move = compiled.segments[0]
    q_samples, n = move.samples(compiled.rate)
    assert n == compiled.metadata["ik_compiled_ticks"]
    assert move.samples_count == n
    assert move.q_start.shape == (6,)
    assert move.q_end.shape == (6,)
    np.testing.assert_allclose(move.q_end, q_samples[-1])
    assert calls[0][0] is None
    assert calls[0][1] == 0.01
    np.testing.assert_allclose(calls[1][0], move.q_start)
    assert calls[1][1] == 0.01
    # IK still runs once per Cartesian sample (+ start); output may be retimed.
    assert len(calls) == move_ticks + 1


def test_compile_cartesian_program_to_joint_stream_uses_higher_uf850_ik_damping(monkeypatch):
    dampings: list[float | None] = []

    def fake_solve(_ctx, xyz_base, *, init_qpos, damping=None):
        del init_qpos
        dampings.append(damping)
        xyz = np.asarray(xyz_base, dtype=np.float64).reshape(3)
        return np.array([xyz[0], xyz[1], xyz[2], 0.0, 0.0, 0.0], dtype=np.float64)

    monkeypatch.setattr("ufactory.trajectory.ik._solve_base_xyz", fake_solve)
    ctx = SimpleNamespace(
        robot_key="uf850",
        robot_urdf_path="/tmp/uf850_g2.urdf",
        kinematics_yaml_path="/tmp/uf850_kinematics_XXXXXX.yaml",
        kinematics_suffix="XXXXXX",
    )
    limits = JointLimits(1.0, 12.0, 0.15, 0.8)
    move_seg = make_movel(
        np.array([0.30, 0.0, 0.30]),
        np.array([0.30, 0.0, 0.20]),
        rate=50.0,
        limits=limits,
        label="descend",
    )
    source = Program(
        rate=50.0,
        robot_key="uf850",
        limits=limits,
        metadata={"kind": "mixed"},
        segments=[move_seg],
    )

    compiled = compile_cartesian_program_to_joint_stream(source, ctx)

    assert compiled.metadata["ik_damping"] == 0.05
    assert dampings
    assert all(value == 0.05 for value in dampings)


def test_collapse_joint_keypoints_drops_plateau():
    from ufactory.trajectory.ik import collapse_joint_keypoints

    q0 = np.zeros(6)
    rows = np.vstack(
        [
            np.zeros(6),
            np.zeros(6),
            np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
    )
    keys = collapse_joint_keypoints(q0, rows)
    assert keys.shape == (3, 6)
    np.testing.assert_allclose(keys[0], 0.0)
    np.testing.assert_allclose(keys[1, 0], 0.1)
    np.testing.assert_allclose(keys[2, 0], 0.2)


def test_retime_ik_joint_stream_bounds_finite_diff_acc():
    from ufactory.trajectory.ik import retime_ik_joint_stream
    from ufactory.safety.statistics import signed_velocity_and_acceleration

    q0 = np.zeros(6)
    # Leading plateau then a jump — the failure mode seen at 100 Hz.
    rows = np.vstack([np.zeros((4, 6)), np.linspace(0.0, 0.05, 20)[:, None] * np.array([0, 1, 0, 0, 1, 0])])
    q_out, n, dur, n_keys = retime_ik_joint_stream(q0, rows, rate=100.0, duration_s=0.25, v_max=0.35, a_max=2.0)
    assert n_keys < rows.shape[0] + 1
    # Short Cartesian duration is extended to the joint LSPB bound.
    assert dur >= 0.25 - 1e-9
    vel, acc = signed_velocity_and_acceleration(np.vstack([q0, q_out]), rate_hz=100.0)
    assert float(np.max(np.abs(acc))) <= 2.0 + 1e-3
    assert float(np.max(np.abs(vel))) <= 0.35 + 1e-3


def test_retime_ik_joint_stream_and_compile_handle_unconstrained_limits_without_oom(monkeypatch):
    import pytest
    from ufactory.trajectory.ik import _MAX_RETIME_SAMPLES, retime_ik_joint_stream

    rng = np.random.default_rng(0)
    q_start = np.zeros(6, dtype=np.float64)
    q_rows = q_start + np.cumsum(rng.uniform(-0.02, 0.02, (50, 6)), axis=0)

    q_out, n, dur, n_keys = retime_ik_joint_stream(
        q_start,
        q_rows,
        rate=250.0,
        duration_s=0.2,
        v_max=0.0,
        a_max=0.0,
    )
    assert n == 50
    assert dur == pytest.approx(0.2)
    assert q_out.shape == (50, 6)
    assert n_keys >= 2
    np.testing.assert_allclose(q_out[-1], q_rows[-1])

    monkeypatch.setattr(
        "ufactory.trajectory.ik._solve_base_xyz",
        lambda _ctx, xyz_base, *, init_qpos, damping=None: np.array(
            [xyz_base[0], xyz_base[1], xyz_base[2], 0.0, 0.0, 0.0], dtype=np.float64
        ),
    )
    ctx = SimpleNamespace(robot_key="xarm6_1305")
    limits = JointLimits(0.35, 2.0, 0.14, 0.8)
    move_seg = make_movel(
        np.array([0.30, 0.0, 0.30]),
        np.array([0.30, 0.0, 0.20]),
        rate=50.0,
        limits=limits,
        label="descend",
    )
    unconstrained = Program(
        rate=50.0,
        robot_key="xarm6_1305",
        limits=None,
        segments=[move_seg],
    )
    compiled = compile_cartesian_program_to_joint_stream(unconstrained, ctx)
    assert len(compiled.segments) == 1
    assert compiled.segments[0].samples_count == move_seg.samples_count

    with pytest.raises(ValueError, match="exceeds safety cap"):
        retime_ik_joint_stream(
            q_start,
            q_rows,
            rate=float(_MAX_RETIME_SAMPLES + 100),
            duration_s=2.0,
            v_max=0.0,
            a_max=0.0,
        )


def test_arm_ik_context_and_solve_arm_home_q_pure_numpy():
    from ufactory.control import ArmMotionController
    from ufactory.trajectory.ik import ArmIKContext, solve_arm_home_q

    controller = ArmMotionController.from_robot(
        "xarm6",
        tcp_offset_mm_rpy=(157.64, 0.0, 59.5, 0.0, 0.0, 0.0),
    )
    q_flange = solve_arm_home_q(controller, [0.22236, -0.12, 0.30], include_tcp=False)
    q_tcp = solve_arm_home_q(controller, [0.38, -0.12, 0.30 - 0.0595], include_tcp=True)
    np.testing.assert_allclose(q_flange, q_tcp, atol=1e-5)

    limits = JointLimits(0.35, 2.0, 0.14, 0.8)
    move_seg = make_movel(
        np.array([0.22236, -0.12, 0.30]),
        np.array([0.22236, -0.12, 0.20]),
        rate=50.0,
        limits=limits,
        label="descend",
    )
    source = Program(rate=50.0, robot_key="xarm6_1305", limits=limits, segments=[move_seg])
    ctx = ArmIKContext(
        arm_controller=controller,
        robot_key="xarm6_1305",
        robot_urdf_path="/tmp/xarm6.urdf",
        kinematics_yaml_path="/tmp/cal.yaml",
        kinematics_suffix="L56A01",
        include_tcp=False,
    )
    compiled = compile_cartesian_program_to_joint_stream(source, ctx)
    assert compiled.segments[0].kind == "movej"
    assert compiled.metadata["ik_kinematics_suffix"] == "L56A01"
    np.testing.assert_allclose(compiled.segments[0].q_start, q_flange, atol=1e-4)
