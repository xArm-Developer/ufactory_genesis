"""CPU regressions for manipulation preposition and SDK simulation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from ufactory.manipulation.commands import (
    ensure_program_start as _ensure_program_start,
    ensure_cartesian_program_start as _ensure_cartesian_program_start,
    sdk_sim_feedback as _sdk_sim_feedback,
)
from ufactory.config import load_runtime_config
from ufactory.hardware.transport import XArmTransport
from ufactory.trajectory.segments import Program, Segment


class _FakeArm:
    def __init__(self, q: np.ndarray, *, xyz_mm: np.ndarray | None = None):
        self.q = np.asarray(q, dtype=np.float64).copy()
        self.connected = True
        self.error_code = 0
        self.warn_code = 0
        self.state = 2
        self.mode = 0
        self.axis = int(self.q.size)
        self.calls: list[tuple] = []
        # Default far from Cartesian home so preposition tests exercise set_position.
        self.xyz_mm = (
            np.asarray(xyz_mm, dtype=np.float64).copy()
            if xyz_mm is not None
            else np.asarray([100.0, 0.0, 100.0], dtype=np.float64)
        )
        self.rpy_rad = np.asarray([np.pi, 0.0, 0.0], dtype=np.float64)

    def get_servo_angle(self, is_radian=True):
        return 0, self.q.tolist()

    def get_position(self, is_radian=True):
        return 0, [*self.xyz_mm.tolist(), *self.rpy_rad.tolist()]

    def get_state(self):
        return 0, self.state

    def set_servo_angle(self, angle, speed, mvacc, wait=True, is_radian=True):
        self.calls.append(("set_servo_angle", list(angle), float(speed), float(mvacc), bool(wait)))
        self.q = np.asarray(angle, dtype=np.float64)
        return 0

    def set_position(self, *pose, speed=100.0, mvacc=500.0, wait=True, is_radian=True):
        self.calls.append(("set_position", list(pose), float(speed), float(mvacc), bool(wait)))
        self.xyz_mm = np.asarray(pose[:3], dtype=np.float64)
        self.rpy_rad = np.asarray(pose[3:6], dtype=np.float64)
        return 0

    def motion_enable(self, enable=True):
        self.calls.append(("motion_enable", bool(enable)))
        return 0

    def set_mode(self, mode):
        self.mode = int(mode)
        self.calls.append(("set_mode", int(mode)))
        return 0

    def set_state(self, state):
        self.state = int(state)
        self.calls.append(("set_state", int(state)))
        return 0

    def disconnect(self):
        self.connected = False


class _FakeSdkSimArm(_FakeArm):
    def __init__(self, physical_q: np.ndarray, *, tcp_offset=(0.0, 0.0, 173.5, 0.0, 0.0, 0.0)):
        super().__init__(physical_q)
        self.physical_q = np.asarray(physical_q, dtype=np.float64).copy()
        self.virtual_q = self.physical_q.copy()
        self.simulation_enabled = False
        self.tcp_offset = list(tcp_offset)

    @property
    def is_simulation_robot(self):
        return self.simulation_enabled

    def set_simulation_robot(self, enabled):
        self.simulation_enabled = bool(enabled)
        self.calls.append(("set_simulation_robot", self.simulation_enabled))
        return 0

    def get_servo_angle(self, is_radian=True):
        q = self.virtual_q if self.simulation_enabled else self.physical_q
        return 0, q.tolist()

    def set_servo_angle(self, angle, speed, mvacc, wait=True, is_radian=True):
        assert self.simulation_enabled
        self.calls.append(("set_servo_angle", list(angle), float(speed), float(mvacc), bool(wait)))
        self.virtual_q = np.asarray(angle, dtype=np.float64)
        return 0

    def set_servo_angle_j(self, angles, speed, mvacc, mvtime, is_radian=True):
        assert self.simulation_enabled
        self.calls.append(("set_servo_angle_j", list(angles)))
        self.virtual_q = np.asarray(angles, dtype=np.float64)
        return 0

    def set_servo_cartesian(self, mvpose, speed, mvacc, mvtime, is_radian=True):
        assert self.simulation_enabled
        self.calls.append(("set_servo_cartesian", list(mvpose)))
        return 0

    def set_state(self, state):
        requested = int(state)
        self.state = 2 if requested == 0 else requested
        self.calls.append(("set_state", requested))
        return 0


def test_preposition_joints_uses_blocking_set_servo_angle_not_servo_stream():
    start = np.asarray([0.1, -0.2, 0.0, 0.0, 0.3, 0.0])
    target = np.asarray([0.0, -0.5, 0.0, 0.0, 0.5, 0.0])
    arm = _FakeArm(start)
    transport = XArmTransport(arm, robot_key="xarm6_1305", serial_number="XI130506XXXXXX")
    err = transport.preposition_joints(target, speed_rad_s=0.2, mvacc_rad_s2=1.5, tolerance_rad=0.02)
    assert err <= 0.02
    assert any(call[0] == "set_servo_angle" for call in arm.calls)
    assert not any(call[0] == "set_servo_angle_j" for call in arm.calls)
    assert np.allclose(arm.q, target)


def test_preposition_cartesian_uses_blocking_set_position_gripper_down():
    arm = _FakeArm(np.zeros(6), xyz_mm=np.asarray([100.0, 50.0, 80.0]))
    transport = XArmTransport(arm, robot_key="xarm6_1305", serial_number="XI130506XXXXXX")
    target = np.asarray([0.30, 0.00, 0.30])
    xyz_err, rpy_err = transport.preposition_cartesian(target, speed_mm_s=80.0, mvacc_mm_s2=400.0)
    assert xyz_err <= 2.0
    assert rpy_err <= 0.05
    assert any(call[0] == "set_position" for call in arm.calls)
    assert np.allclose(arm.xyz_mm, target * 1000.0)
    assert np.allclose(arm.rpy_rad, [np.pi, 0.0, 0.0])


def test_preposition_cartesian_wraps_plus_minus_pi_roll_branch_cut():
    arm = _FakeArm(np.zeros(6), xyz_mm=np.asarray([100.0, 50.0, 80.0]))
    orig_set_position = arm.set_position

    def set_position_reporting_minus_pi(*pose, **kwargs):
        rc = orig_set_position(*pose, **kwargs)
        # Firmware atan2 commonly reports roll=-pi for a commanded +pi gripper-down pose.
        arm.rpy_rad = np.asarray([-np.pi, 0.0, 0.0], dtype=np.float64)
        return rc

    arm.set_position = set_position_reporting_minus_pi
    transport = XArmTransport(arm, robot_key="xarm6_1305", serial_number="XI130506XXXXXX")
    xyz_err, rpy_err = transport.preposition_cartesian(np.asarray([0.30, 0.00, 0.30]))
    assert xyz_err == pytest.approx(0.0, abs=1e-9)
    assert rpy_err == pytest.approx(0.0, abs=1e-9)


def test_leave_software_stop_clears_state_4_without_clean_error():
    arm = _FakeArm(np.zeros(6))
    arm.state = 4
    transport = XArmTransport(arm, robot_key="xarm6_1305", serial_number="XI130506XXXXXX")
    transport.leave_software_stop()
    assert arm.state == 0
    assert ("set_state", 0) in arm.calls
    assert not any(call[0] == "clean_error" for call in arm.calls)


def test_read_state_truncates_padded_seven_vector_to_axis():
    arm = _FakeArm(np.zeros(6))
    arm.q = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 9.9])
    transport = XArmTransport(arm, robot_key="xarm6_1305", serial_number="XI130506XXXXXX")
    state = transport.read_state()
    assert state.q_rad.shape == (6,)
    assert np.allclose(state.q_rad, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


def test_ensure_program_start_requires_confirm_when_far_from_default():
    start = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    target = np.asarray([0.0, -0.5, 0.0, 0.0, 0.5, 0.0])
    transport = XArmTransport(_FakeArm(start), robot_key="xarm6_1305", serial_number="XI130506XXXXXX")
    with pytest.raises(RuntimeError, match="--confirm-real"):
        _ensure_program_start(transport, target, allow_preposition=False)


def test_ensure_program_start_prepositions_when_allowed():
    start = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    target = np.asarray([0.0, -0.5, 0.0, 0.0, 0.5, 0.0])
    arm = _FakeArm(start)
    transport = XArmTransport(arm, robot_key="xarm6_1305", serial_number="XI130506XXXXXX")
    reached = _ensure_program_start(transport, target, allow_preposition=True)
    assert np.allclose(reached, target)
    assert any(call[0] == "set_servo_angle" for call in arm.calls)


def test_ensure_cartesian_program_start_requires_confirm_when_far():
    transport = XArmTransport(_FakeArm(np.zeros(6)), robot_key="xarm6_1305", serial_number="XI130506XXXXXX")
    with pytest.raises(RuntimeError, match="--confirm-real"):
        _ensure_cartesian_program_start(
            transport,
            np.asarray([0.30, 0.00, 0.30]),
            allow_preposition=False,
        )


def test_ensure_cartesian_program_start_prepositions_when_allowed():
    arm = _FakeArm(np.zeros(6), xyz_mm=np.asarray([100.0, 0.0, 100.0]))
    transport = XArmTransport(arm, robot_key="xarm6_1305", serial_number="XI130506XXXXXX")
    target = np.asarray([0.30, 0.00, 0.30])
    _ensure_cartesian_program_start(transport, target, allow_preposition=True)
    assert any(call[0] == "set_position" for call in arm.calls)
    assert np.allclose(arm.xyz_mm, target * 1000.0)


def test_sdk_sim_feedback_prepositions_only_virtual_robot(monkeypatch):
    monkeypatch.setattr("ufactory.manipulation.commands.time.sleep", lambda _seconds: None)
    physical_start = np.asarray([0.3, -0.2, 0.1, 0.0, 0.2, -0.1])
    virtual_start = np.asarray([0.0, -0.5, 0.0, 0.0, 0.5, 0.0])
    virtual_end = virtual_start + np.asarray([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    segment = Segment(
        kind="movej",
        duration=0.02,
        v_max=1.0,
        a_max=2.0,
        label="move",
        q_start=virtual_start,
        q_end=virtual_end,
        q_samples=np.stack((virtual_start, virtual_end)),
        samples_count=2,
    )
    program = Program([segment], rate=50.0, robot_key="xarm6_1305")
    config = load_runtime_config("xarm6")
    arm = _FakeSdkSimArm(physical_start)

    feedback = _sdk_sim_feedback(arm, program, config, virtual_start)

    assert np.allclose(feedback, np.stack((virtual_start, virtual_start, virtual_end)))
    assert np.allclose(arm.physical_q, physical_start)
    assert ("set_simulation_robot", True) in arm.calls
    assert ("set_simulation_robot", False) in arm.calls
    assert not arm.simulation_enabled
    assert next(call for call in arm.calls if call[0] == "set_servo_angle")[1] == pytest.approx(virtual_start)
    assert any(call[0] == "set_servo_angle_j" for call in arm.calls)
    sim_index = arm.calls.index(("set_simulation_robot", True))
    virtual_move_index = next(i for i, call in enumerate(arm.calls) if call[0] == "set_servo_angle")
    disable_index = arm.calls.index(("set_simulation_robot", False))
    assert sim_index < virtual_move_index < disable_index


def test_sdk_sim_feedback_disables_simulation_after_replay_failure(monkeypatch):
    monkeypatch.setattr("ufactory.manipulation.commands.time.sleep", lambda _seconds: None)
    start = np.zeros(6, dtype=np.float64)
    segment = Segment(
        kind="movej",
        duration=0.02,
        v_max=1.0,
        a_max=2.0,
        label="move",
        q_start=start,
        q_end=start,
        q_samples=start[None, :],
        samples_count=1,
    )
    arm = _FakeSdkSimArm(np.ones(6, dtype=np.float64))
    arm.set_servo_angle_j = lambda **_kwargs: 9

    with pytest.raises(RuntimeError, match="SDK simulation command failed with code 9"):
        _sdk_sim_feedback(
            arm,
            Program([segment], rate=50.0, robot_key="xarm6_1305"),
            load_runtime_config("xarm6"),
            start,
        )

    assert not arm.simulation_enabled
    assert ("set_simulation_robot", False) in arm.calls


G2_TCP_OFFSET = (0.0, 0.0, 173.5, 0.0, 0.0, 0.0)


class _RecordingCartesianArm(_FakeArm):
    def set_servo_cartesian(self, mvpose, speed, mvacc, mvtime, is_radian=True):
        self.calls.append(("set_servo_cartesian", list(mvpose)))
        return 0


def test_send_cartesian_target_zero_offset_is_unchanged():
    arm = _RecordingCartesianArm(np.zeros(6))
    transport = XArmTransport(arm, robot_key="xarm6_1305", serial_number="XI130506XXXXXX")
    assert transport.send_cartesian_target(np.asarray([0.30, 0.05, 0.25])) == 0
    sent = arm.calls[-1][1]
    assert sent[:3] == pytest.approx([300.0, 50.0, 250.0])
    assert abs(sent[3]) == pytest.approx(np.pi)
    assert sent[4:] == pytest.approx([0.0, 0.0], abs=1e-12)


def test_send_cartesian_target_applies_tcp_offset_gripper_down():
    arm = _RecordingCartesianArm(np.zeros(6))
    transport = XArmTransport(
        arm,
        robot_key="xarm6_1305",
        serial_number="XI130506XXXXXX",
        tcp_offset=G2_TCP_OFFSET,
    )
    transport.send_cartesian_target(np.asarray([0.30, 0.05, 0.25]))
    sent = arm.calls[-1][1]
    # Gripper-down flips the tool z axis, so the TCP is 173.5 mm below the flange.
    assert sent[:3] == pytest.approx([300.0, 50.0, 250.0 - 173.5])
    assert abs(sent[3]) == pytest.approx(np.pi)


def test_send_cartesian_target_matches_pose_flange_to_tcp_with_rotated_offset():
    from ufactory.kinematics.tcp_offset import pose_flange_to_tcp

    offset = (10.0, -5.0, 120.0, 0.0, 0.0, np.pi / 2.0)
    rpy = (np.pi, 0.1, -0.3)
    arm = _RecordingCartesianArm(np.zeros(6))
    transport = XArmTransport(arm, robot_key="xarm6_1305", serial_number="XI130506XXXXXX", tcp_offset=offset)
    transport.send_cartesian_target(np.asarray([0.30, 0.05, 0.25, *rpy]))
    expected = pose_flange_to_tcp([300.0, 50.0, 250.0, *rpy], offset)
    assert arm.calls[-1][1] == pytest.approx(expected.tolist())


def test_transport_rejects_invalid_tcp_offset():
    with pytest.raises(ValueError, match="tcp_offset"):
        XArmTransport(_FakeArm(np.zeros(6)), robot_key="xarm6_1305", serial_number="X" * 14, tcp_offset=[0.0, 1.0])


def test_preposition_cartesian_commands_and_checks_tcp_pose():
    arm = _FakeArm(np.zeros(6), xyz_mm=np.asarray([100.0, 50.0, 80.0]))
    transport = XArmTransport(
        arm,
        robot_key="xarm6_1305",
        serial_number="XI130506XXXXXX",
        tcp_offset=G2_TCP_OFFSET,
    )
    xyz_err, rpy_err = transport.preposition_cartesian(np.asarray([0.30, 0.00, 0.30]))
    assert xyz_err == pytest.approx(0.0, abs=1e-9)
    assert rpy_err == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(arm.xyz_mm, [300.0, 0.0, 300.0 - 173.5])


def test_ensure_cartesian_program_start_compares_tcp_feedback():
    # Controller reports the TCP pose; the flange is at z=300 mm when TCP z=126.5 mm.
    arm = _FakeArm(np.zeros(6), xyz_mm=np.asarray([300.0, 0.0, 126.5]))
    transport = XArmTransport(
        arm,
        robot_key="xarm6_1305",
        serial_number="XI130506XXXXXX",
        tcp_offset=G2_TCP_OFFSET,
    )
    _ensure_cartesian_program_start(transport, np.asarray([0.30, 0.00, 0.30]), allow_preposition=False)
    assert not any(call[0] == "set_position" for call in arm.calls)


def _movel_program(start_q: np.ndarray, xyz_m: np.ndarray) -> Program:
    movej = Segment(
        kind="movej",
        duration=0.02,
        v_max=1.0,
        a_max=2.0,
        label="start",
        q_start=start_q,
        q_end=start_q,
        q_samples=start_q[None, :],
        samples_count=1,
    )
    movel = Segment(
        kind="movel",
        duration=0.02,
        v_max=0.1,
        a_max=1.0,
        label="line",
        pose_start=xyz_m,
        pose_end=xyz_m,
    )
    return Program([movej, movel], rate=50.0, robot_key="xarm6_1305")


@pytest.mark.parametrize("explicit_offset", [True, False])
def test_sdk_sim_feedback_converts_movel_targets_to_tcp(monkeypatch, explicit_offset):
    monkeypatch.setattr("ufactory.manipulation.commands.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("ufactory.kinematics.tcp_offset.time.sleep", lambda _seconds: None)
    start = np.asarray([0.0, -0.5, 0.0, 0.0, 0.5, 0.0])
    xyz = np.asarray([0.30, 0.05, 0.25])
    program = _movel_program(start, xyz)
    arm = _FakeSdkSimArm(start, tcp_offset=G2_TCP_OFFSET)
    kwargs = {"tcp_offset": np.asarray(G2_TCP_OFFSET)} if explicit_offset else {}

    _sdk_sim_feedback(arm, program, load_runtime_config("xarm6"), start, **kwargs)

    cartesian = [call[1] for call in arm.calls if call[0] == "set_servo_cartesian"]
    assert cartesian
    for pose in cartesian:
        assert pose[:3] == pytest.approx([300.0, 50.0, 250.0 - 173.5])
        assert abs(pose[3]) == pytest.approx(np.pi)
    assert not arm.simulation_enabled
