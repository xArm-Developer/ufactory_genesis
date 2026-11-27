"""Offline unit tests for ufactory.hardware.ft_sensor and scripts/ft_sensor_check.py."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from ufactory.hardware import (
    AI1500_MASS_KG,
    AI1500_OVERLOAD,
    AI1500_RANGE,
    FT_COORD_BASE,
    FT_COORD_TOOL,
    FT_MODE_ADMITTANCE,
    FT_MODE_FORCE,
    FT_MODE_OFF,
    FTAdmittanceParams,
    FTCollisionConfig,
    FTForceControlParams,
    FTLoadOffset,
    FTSensorConfigSnapshot,
    FTSensorReading,
    XArmFTSensor,
    XArmTransport,
    describe_ft_sensor_error,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts import ft_sensor_check  # noqa: E402


def _make_ft_arm(
    *, state: int = 0, mode: int = 0, error_code: int = 0, warn_code: int = 0, axis: int = 6, device_type: int = 6
) -> MagicMock:
    arm = MagicMock()
    arm.connected = True
    arm.state = state
    arm.mode = mode
    arm.error_code = error_code
    arm.warn_code = warn_code
    arm.axis = axis
    arm.device_type = device_type
    arm.get_state.return_value = (0, state)
    arm.motion_enable.return_value = 0
    arm.set_mode.return_value = 0
    arm.set_state.return_value = 0
    arm.set_ft_sensor_enable.return_value = 0
    arm.set_ft_sensor_mode.return_value = 0
    arm.get_ft_sensor_mode.return_value = (0, 0)
    arm.set_ft_sensor_zero.return_value = 0
    arm.iden_ft_sensor_load_offset.return_value = (0, [0.8, 1.0, -2.0, 45.0, 0.1, -0.2, -7.8, 0.01, -0.02, 0.005])
    arm.set_ft_sensor_load_offset.return_value = 0
    arm.set_ft_sensor_admittance_parameters.return_value = 0
    arm.set_ft_sensor_force_parameters.return_value = 0
    arm.get_ft_sensor_data.return_value = (0, [0.5, -0.5, -9.8, 0.01, -0.02, 0.03])
    arm.ft_raw_force = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
    arm.ft_ext_force = [0.2, -0.1, -5.0, 0.01, 0.0, -0.02]
    arm.get_ft_sensor_error.return_value = (0, 0)
    arm.get_ft_sensor_sn.return_value = (0, "AI150000000001")
    arm.get_ft_sensor_version.return_value = (0, "1.2.3")
    arm.get_ft_sensor_config.return_value = (
        0,
        [
            0,
            1,
            1,
            1,
            200,
            0.6,
            0.0,
            [0.0, 0.0, 22.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1,
            [1, 1, 1, 0, 0, 0],
            [0.1, 0.1, 0.1, 0.002, 0.002, 0.002],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [20.0, 20.0, 20.0, 1.0, 1.0, 1.0],
            1,
            [0, 0, 1, 0, 0, 0],
            [0.0, 0.0, -5.0, 0.0, 0.0, 0.0],
            [50.0, 50.0, 50.0, 10.0, 10.0, 10.0],
            [0.001] * 6,
            [0.0] * 6,
            [0.0] * 6,
            [50.0] * 6,
        ],
    )
    arm.set_ft_collision_detection.return_value = 0
    arm.get_ft_collision_detection.return_value = (0, 1)
    arm.set_ft_collision_threshold.return_value = 0
    arm.get_ft_collision_threshold.return_value = (0, [50.0, 50.0, 50.0, 1.0, 1.0, 1.0])
    arm.set_ft_collision_rebound.return_value = 0
    arm.get_ft_collision_rebound.return_value = (0, 1)
    arm.set_ft_collision_reb_distance.return_value = 0
    arm.get_ft_collision_reb_distance.return_value = (0, [10.0, 10.0, 10.0, 0.1, 0.1, 0.1])
    arm.set_ft_admittance_ctrl_threshold.return_value = 0
    arm.get_ft_admittance_ctrl_threshold.return_value = (0, [1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
    return arm


def test_dataclass_validation_and_bounds() -> None:
    reading = FTSensorReading.from_wrench([400.1, -10.0, 600.0, 20.0, 20.1, -30.0], is_raw=False, monotonic_ns=123)
    assert reading.overrange == (True, False, True, False, True, True)
    assert reading.overload == (False, False, True, False, False, True)
    assert reading.any_overrange and reading.any_overload
    assert reading.force_n == (400.1, -10.0, 600.0)
    assert reading.torque_nm == (20.0, 20.1, -30.0)
    np.testing.assert_allclose(reading.as_array(), [400.1, -10.0, 600.0, 20.0, 20.1, -30.0])
    assert AI1500_RANGE == (400.0, 400.0, 400.0, 20.0, 20.0, 20.0)
    assert AI1500_OVERLOAD == (600.0, 600.0, 600.0, 30.0, 30.0, 30.0)
    assert AI1500_MASS_KG == pytest.approx(0.6)

    with pytest.raises(ValueError, match="finite"):
        FTSensorReading.from_wrench([0, 0, float("nan"), 0, 0, 0], is_raw=False)
    with pytest.raises(ValueError, match="length 6"):
        FTSensorReading.from_wrench([0, 0, 0], is_raw=True)

    load = FTLoadOffset.from_sdk_list([0.8, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03])
    assert load.to_sdk_list() == [0.8, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03]
    with pytest.raises(ValueError, match="non-negative"):
        FTLoadOffset(-0.1, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    adm = FTAdmittanceParams(
        coord=FT_COORD_TOOL,
        c_axis=(1, 1, 1, 0, 0, 0),
        M=(0.02, 0.5, 1.0, 0.0001, 0.005, 0.01),
        K=(0.0, 500.0, 2000.0, 0.0, 5.0, 20.0),
        B=(0.0, 10.0, 20.0, 0.1, 1.0, 2.0),
    )
    assert adm.coord == 1
    with pytest.raises(ValueError, match="coord"):
        FTAdmittanceParams(2, (1, 0, 0, 0, 0, 0), (0.1, 0.1, 0.1, 0.001, 0.001, 0.001), (0,) * 6, (1,) * 6)
    with pytest.raises(ValueError, match="integer 0 or 1"):
        FTAdmittanceParams(0, (True, 0, 0, 0, 0, 0), (0.1, 0.1, 0.1, 0.001, 0.001, 0.001), (0,) * 6, (1,) * 6)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"M\[0\]"):
        FTAdmittanceParams(0, (1, 0, 0, 0, 0, 0), (0.01, 0.1, 0.1, 0.001, 0.001, 0.001), (0,) * 6, (1,) * 6)
    with pytest.raises(ValueError, match=r"K\[3\]"):
        FTAdmittanceParams(0, (1, 0, 0, 0, 0, 0), (0.1, 0.1, 0.1, 0.001, 0.001, 0.001), (0, 0, 0, 25.0, 0, 0), (1,) * 6)
    with pytest.raises(ValueError, match=r"B\[0\]"):
        FTAdmittanceParams(0, (1, 0, 0, 0, 0, 0), (0.1, 0.1, 0.1, 0.001, 0.001, 0.001), (0,) * 6, (-1,) * 6)

    fc = FTForceControlParams(
        coord=FT_COORD_BASE,
        c_axis=(0, 0, 1, 0, 0, 0),
        f_ref=(0, 0, -20.0, 0, 0, 0),
        limits=(50, 50, 50, 10, 10, 10),
        kp=(0.05,) * 6,
        ki=(0.0005,) * 6,
        kd=(0.05,) * 6,
        xe_limit=(200.0,) * 6,
    )
    assert fc.coord == 0
    with pytest.raises(ValueError, match=r"f_ref\[2\]"):
        FTForceControlParams(
            0, (0, 0, 1, 0, 0, 0), (0, 0, 201, 0, 0, 0), (10,) * 6, (0.01,) * 6, (0,) * 6, (0,) * 6, (10,) * 6
        )
    with pytest.raises(ValueError, match=r"ki\[0\]"):
        FTForceControlParams(
            0, (0, 0, 1, 0, 0, 0), (0, 0, -5, 0, 0, 0), (10,) * 6, (0.01,) * 6, (0.001,) * 6, (0,) * 6, (10,) * 6
        )


def test_robot_compatibility_and_error_descriptions() -> None:
    arm = _make_ft_arm()
    for robot in ("xarm5", "xarm6_1305", "xarm7", "uf850"):
        assert XArmFTSensor(arm, robot_key=robot).robot_key is not None
    with pytest.raises(ValueError, match="does not support F/T sensor"):
        XArmFTSensor(arm, robot_key="lite6")

    assert describe_ft_sensor_error(0) == "Normal"
    assert "Communication" in describe_ft_sensor_error(64)
    assert "Fx" in describe_ft_sensor_error(66)
    assert "code=99" in describe_ft_sensor_error(99)


def test_read_methods_and_fail_closed_errors() -> None:
    arm = _make_ft_arm()
    ft = XArmFTSensor(arm, robot_key="xarm6")

    snap = ft.read_config()
    assert isinstance(snap, FTSensorConfigSnapshot)
    assert snap.ft_freq == 200
    assert snap.ft_mass_kg == pytest.approx(0.6)
    assert snap.ft_centroid == (0.0, 0.0, 22.0)

    status = ft.read_status(include_identity=True)
    assert status.mode == 0
    assert status.error_code == 0
    assert status.enabled is True
    assert status.serial_number == "AI150000000001"
    assert status.firmware_version == "1.2.3"

    r_ext = ft.read_wrench()
    assert not r_ext.is_raw
    assert r_ext.wrench == (0.5, -0.5, -9.8, 0.01, -0.02, 0.03)
    r_rep = ft.read_wrench(is_raw=True, from_report=True)
    assert r_rep.is_raw
    assert r_rep.wrench == (1.0, 2.0, 3.0, 0.1, 0.2, 0.3)

    col_cfg = ft.read_collision_config(is_radian=True)
    assert isinstance(col_cfg, FTCollisionConfig)
    assert col_cfg.enabled and col_cfg.rebound_enabled
    assert ft.read_admittance_threshold() == (1.0, 1.0, 1.0, 0.1, 0.1, 0.1)

    # Unplugged / comm error fails closed cleanly.
    arm.get_ft_sensor_data.return_value = (1, [0.0] * 6)
    with pytest.raises(RuntimeError, match="get_ft_sensor_data failed"):
        ft.read_wrench()
    arm.get_ft_sensor_sn.return_value = (0, "**************")
    with pytest.raises(RuntimeError, match="Invalid F/T sensor serial number"):
        ft.read_serial_number()

    # Fallback to arm._arm for xarm-python-sdk 1.18.4 where XArmAPI omits SN/version wrappers.
    inner = SimpleNamespace(
        get_ft_sensor_sn=lambda: (0, "AI15000626001F"),
        get_ft_sensor_version=lambda: (0, "1.1.1"),
    )
    wrapper_like = SimpleNamespace(_arm=inner)
    ft_wrapped = XArmFTSensor(wrapper_like, robot_key="xarm6")
    assert ft_wrapped.read_serial_number() == "AI15000626001F"
    assert ft_wrapped.read_firmware_version() == "1.1.1"
    with pytest.raises(RuntimeError, match="does not expose get_ft_sensor_sn"):
        XArmFTSensor(SimpleNamespace(), robot_key="xarm6").read_serial_number()


def test_safety_gates_never_clean_errors_and_always_stop_in_finally(monkeypatch: pytest.MonkeyPatch) -> None:
    import ufactory.hardware.ft_sensor as ft_mod

    monkeypatch.setattr(ft_mod.time, "sleep", lambda _s: None)
    arm = _make_ft_arm()
    ft = XArmFTSensor(arm, robot_key="xarm6")

    # confirm_real gate blocks motion-capable operations.
    with pytest.raises(RuntimeError, match="confirm_real=True"):
        ft.identify_load_offset()
    with pytest.raises(RuntimeError, match="confirm_real=True"):
        ft.set_mode(FT_MODE_ADMITTANCE)
    with pytest.raises(RuntimeError, match="confirm_real=True"):
        ft.set_mode(FT_MODE_FORCE)
    # Turning mode OFF never requires confirm_real and does not re-enter motion state.
    arm.set_state.reset_mock()
    ft.set_mode(FT_MODE_OFF)
    arm.set_ft_sensor_mode.assert_called_with(0)
    arm.set_state.assert_not_called()

    # Controller error / stop / warn / FT error refuse active ops without calling clean_error/clean_warn.
    adm = FTAdmittanceParams(1, (1, 1, 1, 0, 0, 0), (0.1,) * 3 + (0.002,) * 3, (300.0,) * 3 + (4.0,) * 3, (0.0,) * 6)
    arm.error_code = 10
    with pytest.raises(RuntimeError, match="active error"):
        ft.set_mode(FT_MODE_ADMITTANCE, confirm_real=True)
    arm.error_code = 0
    arm.warn_code = 3
    with pytest.raises(RuntimeError, match="active warning"):
        ft.zero()
    arm.warn_code = 0
    arm.get_ft_sensor_error.return_value = (0, 64)
    with pytest.raises(RuntimeError, match="64"):
        with ft.admittance_session(adm, confirm_real=True):
            pass
    arm.clean_error.assert_not_called()
    arm.clean_warn.assert_not_called()

    # Context managers prepare motion state, call set_state(0) after mode activation,
    # and guarantee mode=0 teardown on both normal exit and body exception.
    arm.get_ft_sensor_error.return_value = (0, 0)
    arm.set_ft_sensor_mode.reset_mock()
    arm.set_state.reset_mock()
    arm.motion_enable.reset_mock()
    with pytest.raises(ValueError, match="boom"):
        with ft.admittance_session(
            adm,
            confirm_real=True,
            reaction_threshold=(1.0, 1.0, 1.0, 0.1, 0.1, 0.1),
            zero_on_enter=True,
            disable_on_exit=True,
        ):
            raise ValueError("boom")
    mode_calls = [call.args[0] for call in arm.set_ft_sensor_mode.call_args_list]
    assert mode_calls == [FT_MODE_ADMITTANCE, FT_MODE_OFF]
    arm.motion_enable.assert_called_with(enable=True)
    assert arm.set_state.call_count == 2  # prepare_arm_for_motion + post-mode activation
    arm.set_ft_sensor_enable.assert_any_call(True)
    arm.set_ft_sensor_enable.assert_any_call(False)

    fc = FTForceControlParams(
        1,
        (0, 0, 1, 0, 0, 0),
        (0, 0, -2, 0, 0, 0),
        (0,) * 6,
        (0.001,) * 6,
        (0,) * 6,
        (0,) * 6,
        (10, 10, 10, 0.2, 0.2, 0.2),
    )
    arm.set_ft_sensor_mode.reset_mock()
    arm.set_state.reset_mock()
    with ft.force_control_session(fc, confirm_real=True):
        pass
    assert [c.args[0] for c in arm.set_ft_sensor_mode.call_args_list] == [FT_MODE_FORCE, FT_MODE_OFF]
    assert arm.set_state.call_count == 2

    # RET_IS_INVALID (code=10) from identify_load_offset yields actionable hint.
    arm.iden_ft_sensor_load_offset.return_value = (10, [0.5, -1898.0, 3085.0, -34954.0, 0, 0, 0, 0, 0, 0])
    with pytest.raises(RuntimeError, match="RET_IS_INVALID"):
        ft.identify_load_offset(confirm_real=True)


def test_collision_and_load_configuration_calls() -> None:
    arm = _make_ft_arm()
    ft = XArmFTSensor(arm, robot_key="uf850")
    offset = ft.identify_load_offset(confirm_real=True)
    ft.set_load_offset(offset, update_tcp_load=True)
    arm.set_ft_sensor_load_offset.assert_called_once_with(offset.to_sdk_list(), association_setting_tcp_load=True)

    ft.configure_collision_detection(
        enabled=True,
        thresholds=(20.0, 20.0, 20.0, 1.0, 1.0, 1.0),
        rebound=True,
        rebound_distance=(10.0, 10.0, 10.0, math.radians(5.0), math.radians(5.0), math.radians(5.0)),
        is_radian=True,
    )
    arm.set_ft_collision_threshold.assert_called_once()
    arm.set_ft_collision_reb_distance.assert_called_once()
    arm.set_ft_collision_rebound.assert_called_once_with(True)
    arm.set_ft_collision_detection.assert_called_once_with(True)

    with pytest.raises(ValueError, match="collision_thresholds"):
        ft.configure_collision_detection(thresholds=(1.0, 20.0, 20.0, 1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="rebound_distance"):
        ft.configure_collision_detection(rebound_distance=(1.0, 10.0, 10.0, 5.0, 5.0, 5.0), is_radian=False)

    # XArmTransport.ft_sensor factory integrates cleanly.
    transport = XArmTransport(arm, robot_key="xarm6_1305", serial_number="XI1305000001")
    assert isinstance(transport.ft_sensor(), XArmFTSensor)
    assert transport.ft_sensor().robot_key == "xarm6_1305"


def test_ft_sensor_check_cli_mock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import ufactory.hardware.ft_sensor as ft_mod

    arm = _make_ft_arm()
    monkeypatch.setattr(ft_sensor_check.time, "sleep", lambda _s: None)
    monkeypatch.setattr(ft_mod.time, "sleep", lambda _s: None)

    assert (
        ft_sensor_check.main(
            ["status", "--robot", "xarm6", "--ip", "mock", "--include-identity", "--json"], arm_factory=lambda _ip: arm
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert out["status"]["serial_number"] == "AI150000000001"
    arm.disconnect.assert_called()

    arm.disconnect.reset_mock()
    arm.set_ft_sensor_enable.reset_mock()
    arm.set_ft_sensor_mode.reset_mock()
    assert (
        ft_sensor_check.main(["enable", "--robot", "xarm6", "--ip", "mock", "--json"], arm_factory=lambda _ip: arm) == 0
    )
    enable_out = json.loads(capsys.readouterr().out)
    assert enable_out["command"] == "enable"
    assert enable_out["status"]["enabled"] is True
    arm.set_ft_sensor_enable.assert_called_once_with(True)

    arm.set_ft_sensor_enable.reset_mock()
    arm.set_ft_sensor_mode.reset_mock()
    assert (
        ft_sensor_check.main(["disable", "--robot", "xarm6", "--ip", "mock", "--json"], arm_factory=lambda _ip: arm)
        == 0
    )
    disable_out = json.loads(capsys.readouterr().out)
    assert disable_out["command"] == "disable"
    arm.set_ft_sensor_mode.assert_called_once_with(FT_MODE_OFF)
    arm.set_ft_sensor_enable.assert_called_once_with(False)

    arm.disconnect.reset_mock()
    assert (
        ft_sensor_check.main(["zero", "--robot", "xarm6", "--ip", "mock", "--json"], arm_factory=lambda _ip: arm) == 0
    )
    arm.set_ft_sensor_zero.assert_called_once()

    # Motion commands refuse without --confirm-real BEFORE connecting.
    called = False

    def fail_if_connected(_ip: str):
        nonlocal called
        called = True
        return SimpleNamespace()

    with pytest.raises(RuntimeError, match="--confirm-real"):
        ft_sensor_check.main(["identify-load", "--robot", "xarm6", "--ip", "mock"], arm_factory=fail_if_connected)
    with pytest.raises(RuntimeError, match="--confirm-real"):
        ft_sensor_check.main(["admittance-test", "--robot", "xarm6", "--ip", "mock"], arm_factory=fail_if_connected)
    with pytest.raises(RuntimeError, match="--confirm-real"):
        ft_sensor_check.main(["force-test", "--robot", "xarm6", "--ip", "mock"], arm_factory=fail_if_connected)
    assert not called

    with pytest.raises(SystemExit):
        ft_sensor_check.main(["status", "--robot", "lite6", "--ip", "mock"], arm_factory=lambda _ip: arm)

    out_json = tmp_path / "load.json"
    assert (
        ft_sensor_check.main(
            [
                "identify-load",
                "--robot",
                "xarm6",
                "--ip",
                "mock",
                "--confirm-real",
                "--apply",
                "--output",
                str(out_json),
                "--json",
            ],
            arm_factory=lambda _ip: arm,
        )
        == 0
    )
    assert out_json.is_file()

    arm.set_ft_sensor_mode.reset_mock()
    assert (
        ft_sensor_check.main(
            [
                "admittance-test",
                "--robot",
                "xarm6",
                "--ip",
                "mock",
                "--confirm-real",
                "--duration",
                "0",
                "--zero",
                "--json",
            ],
            arm_factory=lambda _ip: arm,
        )
        == 0
    )
    assert [c.args[0] for c in arm.set_ft_sensor_mode.call_args_list] == [FT_MODE_ADMITTANCE, FT_MODE_OFF]

    arm.set_ft_sensor_mode.reset_mock()
    assert (
        ft_sensor_check.main(
            ["force-test", "--robot", "xarm6", "--ip", "mock", "--confirm-real", "--duration", "0", "--json"],
            arm_factory=lambda _ip: arm,
        )
        == 0
    )
    assert [c.args[0] for c in arm.set_ft_sensor_mode.call_args_list] == [FT_MODE_FORCE, FT_MODE_OFF]
