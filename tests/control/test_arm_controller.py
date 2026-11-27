from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from ufactory.control import (
    DEFAULT_CONTROL_RATE_HZ,
    DEFAULT_SERVO_RATE_HZ,
    ERROR_INVALID_ARG,
    ERROR_INVALID_MODE,
    ERROR_NONE,
    ERROR_OVERSPEED,
    MODE_POSITION,
    MODE_SERVO,
    STATE_READY,
    STATE_STOP,
    ApproxMotionConfig,
    ArmMotionController,
)


class _FakeXArm:
    def __init__(self, initial_approx: int = 0) -> None:
        self.approx = int(initial_approx)
        self.calls: list[tuple[str, object]] = []

    def get_allow_approx_motion(self) -> tuple[int, int]:
        return (0, self.approx)

    def set_allow_approx_motion(self, on_off: bool) -> int:
        self.approx = 1 if on_off else 0
        self.calls.append(("set_allow_approx_motion", bool(on_off)))
        return 0

    def set_mode(self, mode: int) -> int:
        self.calls.append(("set_mode", mode))
        return 0

    def set_state(self, state: int) -> int:
        self.calls.append(("set_state", state))
        return 0

    def clean_error(self) -> int:
        self.calls.append(("clean_error", None))
        return 0

    def set_tcp_offset(self, offset: list[float], *, is_radian: bool = True) -> int:
        self.calls.append(("set_tcp_offset", (tuple(offset), is_radian)))
        return 0


def test_controller_defaults_rates_and_sdk_switch_parity() -> None:
    ctrl = ArmMotionController.from_robot("uf850")
    assert ctrl.control_rate_hz == DEFAULT_CONTROL_RATE_HZ == 250.0
    assert ctrl.servo_rate_hz == DEFAULT_SERVO_RATE_HZ == 50.0
    assert ctrl.is_approx_motion is False
    assert ctrl.get_allow_approx_motion() == (0, 0)

    assert ctrl.set_allow_approx_motion(True) == 0
    assert ctrl.is_approx_motion is True
    assert ctrl.get_allow_approx_motion() == (0, 1)

    with pytest.raises(TypeError):
        ctrl.set_allow_approx_motion(1)  # type: ignore[arg-type]

    # Servo rate allows 30-250 Hz, rejects out of range
    ctrl.set_servo_rate(100.0)
    assert ctrl.servo_rate_hz == 100.0
    ctrl.set_servo_rate(250.0)
    with pytest.raises(ValueError, match="servo_rate_hz"):
        ctrl.set_servo_rate(20.0)
    with pytest.raises(ValueError, match="servo_rate_hz"):
        ctrl.set_servo_rate(300.0)


def test_sync_hardware_switches_restores_on_normal_and_error() -> None:
    fake = _FakeXArm(initial_approx=0)
    ctrl = ArmMotionController.from_robot("lite6", allow_approx_motion=True)

    with ctrl.sync_hardware_switches(fake):
        assert fake.approx == 1
    assert fake.approx == 0

    with pytest.raises(RuntimeError, match="boom"):
        with ctrl.sync_hardware_switches(fake):
            assert fake.approx == 1
            raise RuntimeError("boom")
    assert fake.approx == 0


def test_mode0_250hz_movej_and_movel_and_tcp_offset() -> None:
    ctrl = ArmMotionController.from_robot("uf850")
    assert ctrl.set_tcp_offset([0.0, 0.0, 100.0, 0.0, 0.0, 0.0], is_radian=True) == ERROR_NONE
    assert ctrl.set_mode(MODE_POSITION) == ERROR_NONE
    assert ctrl.set_state(STATE_READY) == ERROR_NONE

    q_start = np.array([0.0, -0.3, -0.7, 0.0, 1.0, 0.0], dtype=np.float64)
    assert ctrl.set_servo_angle(q_start, speed=1.0, mvacc=8.0, is_radian=True) == ERROR_NONE
    assert ctrl.last_motion is not None
    assert ctrl.last_motion.rate_hz == 250.0
    code_p, pos0 = ctrl.get_position(is_radian=True)
    assert code_p == 0

    # MoveL 60 mm in +X at 250 Hz
    code_m = ctrl.set_position(x=pos0[0] + 60.0, speed=150.0, mvacc=800.0, is_radian=True)
    assert code_m == ERROR_NONE
    assert ctrl.last_motion is not None
    assert ctrl.last_motion.kind == "movel"
    assert ctrl.last_motion.rate_hz == 250.0
    assert len(ctrl.last_motion.q_samples) > 50
    _, pos1 = ctrl.get_position(is_radian=True)
    assert pos1[0] == pytest.approx(pos0[0] + 60.0, abs=1e-3)
    assert pos1[1] == pytest.approx(pos0[1], abs=1e-3)
    assert pos1[2] == pytest.approx(pos0[2], abs=1e-3)

    # FK/IK SDK roundtrip (degree mode too)
    code_a, ang_deg = ctrl.get_servo_angle(is_radian=False)
    assert code_a == 0
    code_fk, fk_deg = ctrl.get_forward_kinematics(ang_deg, input_is_radian=False, return_is_radian=False)
    assert code_fk == 0
    code_ik, ik_deg = ctrl.get_inverse_kinematics(fk_deg, input_is_radian=False, return_is_radian=False)
    assert code_ik == 0
    assert np.allclose(ik_deg, ang_deg, atol=1e-2)


def test_mode0_and_mode1_singularity_c24_vs_approx_motion_with_calibrated_uf850() -> None:
    cal = Path("assets/urdf/robots/uf850/kinematics/user/uf850_kinematics_F54B07.yaml")
    if not cal.is_file():
        pytest.skip("local UF850 calibration YAML not present")

    pose_a = [300.0, 0.0, 288.6, math.pi, 0.0, 0.0]
    pose_b = [300.0, 0.0, 188.6, math.pi, 0.0, 0.0]

    # Mode 0 (250 Hz): approx=False aborts with C24 overspeed, approx=True passes cleanly
    ctrl = ArmMotionController.from_robot("uf850", calibration_path=cal, allow_approx_motion=False)
    code_ik, q_a = ctrl.get_inverse_kinematics(pose_a, input_is_radian=True, return_is_radian=True)
    assert code_ik == 0
    ctrl.reset_state(q_a)

    code_fail = ctrl.set_position(*pose_b, speed=150.0, mvacc=800.0, is_radian=True)
    assert code_fail == ERROR_OVERSPEED
    assert ctrl.state == STATE_STOP
    assert ctrl.error_code == ERROR_OVERSPEED

    # Recover, enable allow_approx_motion, re-run Mode 0 (250 Hz) -> PASS with tiny orientation error
    ctrl.clean_error()
    ctrl.reset_state(q_a)
    assert ctrl.set_allow_approx_motion(True) == 0
    code_ok = ctrl.set_position(*pose_b, speed=150.0, mvacc=800.0, is_radian=True)
    assert code_ok == ERROR_NONE
    assert ctrl.last_motion is not None
    assert ctrl.last_motion.used_approx_steps > 0
    assert ctrl.last_motion.max_position_error_m < 1e-4  # < 0.1 mm
    assert ctrl.last_motion.max_orientation_error_rad < math.radians(0.5)
    _, end_pos = ctrl.get_position(is_radian=True)
    assert end_pos[2] == pytest.approx(188.6, abs=0.05)

    # Mode 1 (50 Hz): set_servo_cartesian with approx=True passes smoothly
    ctrl.reset_state(q_a)
    assert ctrl.set_mode(MODE_SERVO) == 0
    assert ctrl.set_position(*pose_b) == ERROR_INVALID_MODE  # Mode 0 API rejected in Mode 1
    zs = np.linspace(288.6, 188.6, 45)
    for z in zs:
        assert ctrl.set_servo_cartesian([300.0, 0.0, float(z), math.pi, 0.0, 0.0], is_radian=True) == 0
    _, end_pos_m1 = ctrl.get_position(is_radian=True)
    assert end_pos_m1[2] == pytest.approx(188.6, abs=0.05)


def test_mode1_servo_angle_j_and_invalid_inputs() -> None:
    ctrl = ArmMotionController.from_robot("lite6")
    assert ctrl.set_servo_angle_j([0.0] * 6) == ERROR_INVALID_MODE  # Still in Mode 0
    assert ctrl.set_mode(MODE_SERVO) == 0
    assert ctrl.set_servo_angle_j([0.1, 0.2, 0.5, 0.0, 0.2, 0.0]) == 0
    assert ctrl.set_servo_angle_j([0.1, 0.2]) == ERROR_INVALID_ARG
    assert ctrl.set_servo_angle_j([0.0, 0.0, 99.0, 0.0, 0.0, 0.0]) == ERROR_INVALID_ARG
    assert ctrl.state == STATE_STOP
    ctrl.clean_error()
    assert ctrl.set_servo_cartesian([200.0, 0.0, float("nan"), math.pi, 0.0, 0.0]) == ERROR_INVALID_ARG
    assert ctrl.set_mode(99) == ERROR_INVALID_MODE
    assert ctrl.set_state(99) == ERROR_INVALID_ARG
    assert ctrl.set_tcp_offset([1.0, 2.0]) == ERROR_INVALID_ARG
    assert ctrl.set_allow_approx_motion(False) == 0
    assert ctrl.approx_config == ApproxMotionConfig()
