"""Mock contract tests for VacuumGripperAdapter and TGPIO pin mapping."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ufactory.config import load_runtime_config
from ufactory.grippers import (
    VacuumGripperAdapter,
    VacuumProfileAdapter,
    create_gripper_adapter,
    vacuum_controller_mode_error,
    vacuum_io_pins,
)


@pytest.mark.parametrize(("version", "expected"), [(1, (0, 1)), (2, (3, 4))])
def test_vacuum_io_pins_mapping(version: int, expected: tuple[int, int]) -> None:
    assert vacuum_io_pins(version) == expected


@pytest.mark.parametrize("bad_version", [0, 3, -1, True, False, 1.5, "1"])
def test_vacuum_io_pins_rejects_invalid_version(bad_version) -> None:
    with pytest.raises(ValueError):
        vacuum_io_pins(bad_version)
    with pytest.raises(ValueError):
        VacuumGripperAdapter(hardware_version=bad_version)


@pytest.mark.parametrize("version", [1, 2])
def test_vacuum_adapter_sdk_contract(version: int) -> None:
    adapter = VacuumGripperAdapter(hardware_version=version)
    arm = MagicMock()
    arm.get_vacuum_gripper.return_value = (0, 0)
    arm.set_vacuum_gripper.return_value = 0

    assert adapter.prepare_real(arm) == 0
    arm.get_vacuum_gripper.assert_called_with(hardware_version=version)

    assert adapter.send_real_suction(arm, True) == 0
    arm.set_vacuum_gripper.assert_called_with(True, wait=False, hardware_version=version)

    assert adapter.send_real_suction(arm, False) == 0
    arm.set_vacuum_gripper.assert_called_with(False, wait=False, hardware_version=version)

    arm.get_vacuum_gripper.return_value = (0, 1)
    assert adapter.read_real(arm) == (0, 1)


def test_vacuum_adapter_prepare_real_requires_sdk_methods() -> None:
    adapter = VacuumGripperAdapter()

    class IncompleteArm:
        def get_vacuum_gripper(self, **_kwargs):
            return (0, 0)

    with pytest.raises(RuntimeError, match="set_vacuum_gripper"):
        adapter.prepare_real(IncompleteArm())


def test_vacuum_adapter_rejects_non_boolean_suction_argument() -> None:
    adapter = VacuumGripperAdapter()
    arm = MagicMock()
    with pytest.raises(ValueError, match="boolean"):
        adapter.send_real_suction(arm, "on")  # type: ignore[arg-type]
    arm.set_vacuum_gripper.assert_not_called()


@pytest.mark.parametrize(("version", "pins"), [(1, (0, 1)), (2, (3, 4))])
def test_vacuum_adapter_sync_false_and_read_real_output(version: int, pins: tuple[int, int]) -> None:
    adapter = VacuumGripperAdapter(hardware_version=version)
    arm = MagicMock()
    arm.set_vacuum_gripper.return_value = 0
    outputs = [0, 0, 0, 0, 0]
    outputs[pins[0]] = 1
    arm.get_tgpio_output_digital.return_value = (0, outputs)

    assert adapter.send_real_suction(arm, True, sync=False) == 0
    arm.set_vacuum_gripper.assert_called_with(True, wait=False, sync=False, hardware_version=version)
    assert adapter.read_real_output(arm) == (0, (1, 0))


@pytest.mark.parametrize("version", [1, 2])
def test_vacuum_profile_adapter_deduplicates_and_shuts_down(version: int) -> None:
    cfg = load_runtime_config("xarm5", gripper="vacuum_gripper")
    assert cfg.gripper is not None
    adapter = create_gripper_adapter(cfg.gripper, hardware_version=version)
    assert isinstance(adapter, VacuumProfileAdapter)
    assert adapter.gap_to_drive(1.0) == 0.0
    assert adapter.gap_to_drive(0.0) == 0.0
    assert adapter.drive_to_gap(0.0) == 1.0

    arm = MagicMock()
    arm.set_vacuum_gripper.return_value = 0
    outputs_off = [0, 0, 0, 0, 0]
    outputs_off[adapter._vacuum.off_pin] = 1
    arm.get_tgpio_output_digital.return_value = (0, outputs_off)

    # prepare_real releases suction without calling check_on=True get_vacuum_gripper.
    assert adapter.prepare_real(arm) == 0
    arm.get_vacuum_gripper.assert_not_called()
    arm.set_vacuum_gripper.assert_called_once_with(False, wait=False, hardware_version=version)

    # Repeated open/off ticks deduplicate; crossing midpoint switches suction ON once.
    assert adapter.send_real_gap(arm, 1.0) == 0
    assert adapter.send_real_gap(arm, 0.8) == 0
    assert arm.set_vacuum_gripper.call_count == 1

    assert adapter.send_real_gap(arm, 0.2) == 0
    assert adapter.send_real_gap(arm, 0.0) == 0
    assert arm.set_vacuum_gripper.call_count == 2
    arm.set_vacuum_gripper.assert_called_with(True, wait=False, hardware_version=version)

    # Crossing back above midpoint switches suction OFF once.
    assert adapter.send_real_gap(arm, 0.6) == 0
    assert adapter.send_real_gap(arm, 1.0) == 0
    assert arm.set_vacuum_gripper.call_count == 3
    arm.set_vacuum_gripper.assert_called_with(False, wait=False, hardware_version=version)

    # shutdown_real skips blocking SDK call when TGPIO output is already OFF (0, 1).
    assert adapter.shutdown_real(arm) == 0
    assert arm.set_vacuum_gripper.call_count == 3

    # When suction is still ON and arm is not paused, shutdown_real turns it OFF via SDK.
    outputs_on = [0, 0, 0, 0, 0]
    outputs_on[adapter._vacuum.on_pin] = 1
    arm.get_tgpio_output_digital.return_value = (0, outputs_on)
    arm.state = 2
    assert adapter.shutdown_real(arm) == 0
    assert arm.set_vacuum_gripper.call_count == 4

    # When suction is still ON while paused (state=3), shutdown_real never calls blocking set_vacuum_gripper.
    arm.state = 3
    arm._arm.arm_cmd.tgpio_set_digital.return_value = [0]
    assert adapter.shutdown_real(arm) == 0
    assert arm.set_vacuum_gripper.call_count == 4
    assert arm._arm.arm_cmd.tgpio_set_digital.call_count == 2


def test_vacuum_controller_mode_error_matrix() -> None:
    for robot in ("xarm5", "xarm6", "xarm7", "uf850"):
        for mode in ("dry-run", "sdk-sim", "real"):
            assert vacuum_controller_mode_error("pick_place", robot, "vacuum_gripper", mode) is None
        sim_err = vacuum_controller_mode_error("pick_place", robot, "vacuum_gripper", "sim")
        assert sim_err is not None and "--mode sim" in sim_err
    lite_err = vacuum_controller_mode_error("pick_place", "lite6", "vacuum_gripper", "dry-run")
    assert lite_err is not None and "xarm5" in lite_err
    assert vacuum_controller_mode_error("pick_place", "lite6", "default", "sim") is None
