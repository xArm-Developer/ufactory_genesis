"""Phase 2 integration unit tests (transport, sdk_sim switch sync, sim-rate stepping, CLIs, IK)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ufactory.cli import packaging, pick_place
from ufactory.config import load_runtime_config
from ufactory.control import ArmMotionController
from ufactory.hardware.transport import XArmTransport
from ufactory.manipulation.commands import sdk_sim_feedback
from ufactory.trajectory.ik import compile_cartesian_program_to_joint_stream
from ufactory.trajectory.segments import JointLimits, Program, Segment, make_movel


class _ApproxFakeArm:
    def __init__(self, q: np.ndarray, *, initial_approx: int = 0) -> None:
        self.physical_q = np.asarray(q, dtype=np.float64).copy()
        self.virtual_q = self.physical_q.copy()
        self.axis = int(self.physical_q.size)
        self.connected = True
        self.state = 2
        self.mode = 0
        self.error_code = 0
        self.warn_code = 0
        self.is_simulation_robot = False
        self.approx_state = int(initial_approx)
        self.tcp_offset = [0.0] * 6
        self.calls: list[tuple] = []
        self.fail_on_servo = False

    def get_allow_approx_motion(self) -> tuple[int, int]:
        return 0, self.approx_state

    def set_allow_approx_motion(self, on_off: bool) -> int:
        self.approx_state = 1 if on_off else 0
        self.calls.append(("set_allow_approx_motion", bool(on_off)))
        return 0

    def motion_enable(self, enable: bool = True) -> int:
        self.calls.append(("motion_enable", bool(enable)))
        return 0

    def set_mode(self, mode: int) -> int:
        self.mode = int(mode)
        return 0

    def set_state(self, state: int) -> int:
        self.state = 2 if int(state) == 0 else int(state)
        return 0

    def get_state(self) -> tuple[int, int]:
        return 0, self.state

    def set_simulation_robot(self, on_off: bool) -> int:
        self.is_simulation_robot = bool(on_off)
        self.calls.append(("set_simulation_robot", bool(on_off)))
        return 0

    def set_servo_angle(self, angle, **_kwargs) -> int:
        self.virtual_q = np.asarray(angle, dtype=np.float64).copy()
        return 0

    def set_servo_angle_j(self, angles, **_kwargs) -> int:
        if self.fail_on_servo:
            return 9
        self.virtual_q = np.asarray(angles, dtype=np.float64).copy()
        return 0

    def get_servo_angle(self, is_radian: bool = True):
        del is_radian
        return 0, (self.virtual_q if self.is_simulation_robot else self.physical_q).tolist()


def test_xarm_transport_approx_motion_methods() -> None:
    arm = _ApproxFakeArm(np.zeros(6), initial_approx=0)
    transport = XArmTransport(arm, robot_key="uf850", serial_number="FX851005XXXXXX")
    assert transport.get_allow_approx_motion() == (0, 0)
    assert transport.is_approx_motion is False
    assert transport.set_allow_approx_motion(True) == 0
    assert transport.get_allow_approx_motion() == (0, 1)
    assert transport.is_approx_motion is True
    with pytest.raises(TypeError, match="bool"):
        transport.set_allow_approx_motion(1)  # type: ignore[arg-type]


def test_sdk_sim_feedback_syncs_and_restores_approx_motion_on_success_and_failure(monkeypatch) -> None:
    monkeypatch.setattr("ufactory.manipulation.commands.time.sleep", lambda _s: None)
    cfg = load_runtime_config("uf850", overrides={"motion.allow_approx_motion": True})
    q0 = np.asarray(cfg.arm.default_qpos_rad, dtype=np.float64)
    seg = Segment(
        kind="movej",
        duration=0.02,
        v_max=1.0,
        a_max=12.0,
        label="move",
        q_start=q0,
        q_end=q0,
        q_samples=q0[None, :],
        samples_count=1,
    )
    prog = Program([seg], rate=50.0, robot_key=cfg.robot.key)

    arm = _ApproxFakeArm(q0, initial_approx=0)
    sdk_sim_feedback(arm, prog, cfg, q0, tcp_offset=np.zeros(6))
    assert arm.approx_state == 0
    assert ("set_allow_approx_motion", True) in arm.calls
    assert arm.calls[-2] == ("set_allow_approx_motion", False)
    assert not arm.is_simulation_robot

    # Must also restore on failure
    arm_fail = _ApproxFakeArm(q0, initial_approx=0)
    arm_fail.fail_on_servo = True
    with pytest.raises(RuntimeError, match="code 9"):
        sdk_sim_feedback(arm_fail, prog, cfg, q0, tcp_offset=np.zeros(6))
    assert arm_fail.approx_state == 0
    assert not arm_fail.is_simulation_robot


def test_physics_steps_for_tick_rates() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("genesis")
    from ufactory.trajectory.sim_executor import physics_steps_for_tick

    assert physics_steps_for_tick(50.0, 50.0) == (1, 0.0)
    assert physics_steps_for_tick(100.0, 50.0) == (2, 0.0)
    assert physics_steps_for_tick(250.0, 50.0) == (5, 0.0)
    # Non-integer ratio (60 Hz physics / 50 Hz control -> 6 steps every 5 control ticks)
    acc = 0.0
    counts = []
    for _ in range(5):
        n, acc = physics_steps_for_tick(60.0, 50.0, acc)
        counts.append(n)
    assert sum(counts) == 6
    assert abs(acc) < 1e-12
    with pytest.raises(ValueError, match="sim_rate_hz"):
        physics_steps_for_tick(0.0, 50.0)


def test_cli_print_config_accepts_allow_approx_motion_and_sim_rate_hz(capsys) -> None:
    rc1 = pick_place.main(
        [
            "--robot",
            "uf850",
            "--mode",
            "dry-run",
            "--executor",
            "servo_cartesian",
            "--allow-approx-motion",
            "--sim-rate-hz",
            "250",
            "--print-config",
        ]
    )
    out1 = capsys.readouterr().out
    assert rc1 == 0
    assert "allow_approx_motion: true" in out1
    assert "rate_hz: 250.0" in out1

    rc2 = packaging.main(
        [
            "--robot",
            "lite6",
            "--mode",
            "sim",
            "--allow-approx-motion",
            "--sim-rate-hz",
            "100",
            "--print-config",
        ]
    )
    out2 = capsys.readouterr().out
    assert rc2 == 0
    assert "allow_approx_motion: true" in out2
    assert "rate_hz: 100.0" in out2


def test_mirror_and_ik_compile_use_injected_arm_controller() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("genesis")
    from ufactory.trajectory.mirror_executor import resolve_segment_start_arm_q, resolve_tick_arm_q

    ctrl = ArmMotionController.from_robot("uf850", allow_approx_motion=True)
    _, q_a_list = ctrl.get_inverse_kinematics(
        [300.0, 0.0, 288.6, np.pi, 0.0, 0.0], input_is_radian=True, return_is_radian=True
    )
    q_a = np.asarray(q_a_list, dtype=np.float64)
    ctx = SimpleNamespace(
        robot_key="uf850",
        arm_controller=ctrl,
        arm_dof_idx=list(range(6)),
        home_qpos=q_a.copy(),
        robot_urdf_path="",
        kinematics_yaml_path=None,
        kinematics_suffix=None,
    )
    limits = JointLimits(1.0, 12.0, 0.15, 0.8)
    seg = make_movel(
        np.array([0.300, 0.0, 0.2886]),
        np.array([0.300, 0.0, 0.1886]),
        rate=50.0,
        limits=limits,
        label="descend",
    )
    q_start = resolve_segment_start_arm_q(ctx, seg, init_qpos=q_a)
    np.testing.assert_allclose(q_start, q_a, atol=1e-4)

    samples, _ = seg.samples(50.0)
    q_tick0 = resolve_tick_arm_q(ctx, seg, samples, 0, init_qpos=q_a, dt_s=0.02)
    assert q_tick0.shape == (6,)

    prog = Program([seg], rate=50.0, limits=limits, robot_key="uf850")
    compiled = compile_cartesian_program_to_joint_stream(prog, ctx)
    assert compiled.segments[0].kind == "movej"
    assert compiled.segments[0].q_samples.shape[1] == 6
