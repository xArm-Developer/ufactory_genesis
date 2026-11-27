from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import xml.etree.ElementTree as ET

import pytest

from ufactory.config import load_runtime_config
from ufactory.grippers import create_gripper_adapter
from ufactory.grippers.g2 import (
    GRIPPER_G2_SIM_OPEN_DRIVE,
    GRIPPER_G2_SIM_CLOSE_DRIVE,
    gripper_g2_gap_m_to_sim_drive,
)
from ufactory.grippers.lite6 import (
    LITE6_GRIPPER_CLOSED_GAP_M,
    LITE6_GRIPPER_DEMO_HOLD_STEPS,
    LITE6_GRIPPER_DRIVE_LIMIT_N,
    LITE6_GRIPPER_HOLD_BIAS_GAP_M,
    LITE6_GRIPPER_KP,
    LITE6_GRIPPER_KV,
    LITE6_GRIPPER_OPEN_GAP_M,
    LITE6_GRIPPER_SIM_CLOSED_DRIVE,
    LITE6_GRIPPER_SIM_OPEN_DRIVE,
    configure_lite6_gripper_drive,
    lite6_gripper_demo_drive,
    lite6_gripper_demo_label,
    lite6_gripper_sim_drive_to_gap_m,
)
from ufactory.robots.paths import lite6_gripper_movable_visual_urdf


@pytest.mark.parametrize("robot", ["xarm5", "xarm6", "xarm7", "uf850"])
def test_g2_mapping_and_real_sdk_contract(robot):
    adapter = create_gripper_adapter(load_runtime_config(robot).gripper)
    assert adapter.profile.open_drive == GRIPPER_G2_SIM_OPEN_DRIVE
    assert adapter.profile.closed_drive == GRIPPER_G2_SIM_CLOSE_DRIVE
    arm = MagicMock()
    arm.get_gripper_err_code.return_value = (0, 0)
    arm.set_gripper_enable.return_value = 0
    arm.set_gripper_mode.return_value = 0
    arm.set_gripper_g2_position.return_value = 0
    assert adapter.prepare_real(arm) == 0
    for gap in (0, 0.010, 0.02415587200800565, 0.030, 0.060, 0.084):
        q = adapter.gap_to_drive(gap)
        assert q == pytest.approx(gripper_g2_gap_m_to_sim_drive(gap), abs=1e-14)
        assert adapter.drive_to_gap(q) == pytest.approx(gap, abs=1e-12)
        if adapter.capabilities.real_command:
            assert adapter.send_real_gap(arm, gap) == 0
            arm.set_gripper_g2_position.assert_called_with(pos=gap * 1000, wait=False)
    if not adapter.capabilities.real_command:
        with pytest.raises(RuntimeError, match="not enabled"):
            adapter.send_real_gap(arm, 0.030)
        arm.set_gripper_g2_position.assert_not_called()


def test_g2_never_cleans_existing_error():
    adapter = create_gripper_adapter(load_runtime_config("xarm6").gripper)
    arm = MagicMock()
    arm.get_gripper_err_code.return_value = (0, 7)
    with pytest.raises(RuntimeError, match="recover manually"):
        adapter.prepare_real(arm)
    arm.clean_gripper_error.assert_not_called()


def test_lite6_quantizes_gap_to_binary_commands():
    adapter = create_gripper_adapter(load_runtime_config("lite6").gripper)
    arm = MagicMock()
    arm.open_lite6_gripper.return_value = 0
    arm.close_lite6_gripper.return_value = 0
    assert adapter.prepare_real(arm) == 0
    assert adapter.send_real_gap(arm, adapter.profile.open_gap_m) == 0
    assert adapter.send_real_gap(arm, adapter.profile.closed_gap_m) == 0
    arm.open_lite6_gripper.assert_called_once_with(sync=False)
    arm.close_lite6_gripper.assert_called_once_with(sync=False)


def test_lite6_real_precheck_requires_all_sdk_commands():
    adapter = create_gripper_adapter(load_runtime_config("lite6").gripper)

    class IncompleteArm:
        def open_lite6_gripper(self, **_kwargs):
            return 0

        def close_lite6_gripper(self, **_kwargs):
            return 0

    with pytest.raises(RuntimeError, match="stop_lite6_gripper"):
        adapter.prepare_real(IncompleteArm())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_gripper_mapping_rejects_invalid_gap(value):
    adapter = create_gripper_adapter(load_runtime_config("xarm6").gripper)
    with pytest.raises(ValueError):
        adapter.gap_to_drive(value)


def test_standalone_lite6_gripper_collision_assets_and_gap_mapping():
    urdf = Path(lite6_gripper_movable_visual_urdf()).resolve()
    root = ET.parse(urdf).getroot()
    for link_name in ("uflite_finger1", "uflite_finger2"):
        link = root.find(f".//link[@name='{link_name}']")
        assert link is not None
        collisions = link.findall("collision")
        assert len(collisions) == 2
        for collision in collisions:
            assert collision.find("./geometry/box") is not None
            assert collision.find("./geometry/mesh") is None
        visual_mesh = link.find("./visual/geometry/mesh")
        assert visual_mesh is not None
        mesh_path = (urdf.parent / str(visual_mesh.get("filename"))).resolve()
        assert mesh_path.is_file()
    assert lite6_gripper_sim_drive_to_gap_m(LITE6_GRIPPER_SIM_CLOSED_DRIVE) == pytest.approx(LITE6_GRIPPER_CLOSED_GAP_M)
    assert lite6_gripper_sim_drive_to_gap_m(LITE6_GRIPPER_SIM_OPEN_DRIVE) == pytest.approx(LITE6_GRIPPER_OPEN_GAP_M)


def test_lite6_collision_demo_drive_matches_glb_keyframes():
    cases = (
        (0, LITE6_GRIPPER_SIM_OPEN_DRIVE),
        (LITE6_GRIPPER_DEMO_HOLD_STEPS - 1, LITE6_GRIPPER_SIM_OPEN_DRIVE),
        (LITE6_GRIPPER_DEMO_HOLD_STEPS, LITE6_GRIPPER_SIM_CLOSED_DRIVE),
        (2 * LITE6_GRIPPER_DEMO_HOLD_STEPS - 1, LITE6_GRIPPER_SIM_CLOSED_DRIVE),
    )
    for step, expected in cases:
        assert lite6_gripper_demo_drive(step) == pytest.approx(expected)


def test_lite6_collision_demo_labels_snap_at_hold_boundary():
    assert lite6_gripper_demo_label(0) == "open"
    assert lite6_gripper_demo_label(LITE6_GRIPPER_DEMO_HOLD_STEPS - 1) == "open"
    assert lite6_gripper_demo_label(LITE6_GRIPPER_DEMO_HOLD_STEPS) == "closed"
    assert lite6_gripper_demo_label(2 * LITE6_GRIPPER_DEMO_HOLD_STEPS - 1) == "closed"


def test_lite6_runtime_config_and_drive_setup_match_calibrated_pneumatic_dynamics():
    config = load_runtime_config("lite6")
    assert list(config.arm.kv) == pytest.approx([300.0, 300.0, 200.0, 4.0, 8.0, 4.0])
    assert config.gripper is not None
    assert config.gripper.kp == pytest.approx(LITE6_GRIPPER_KP)
    assert config.gripper.kv == pytest.approx(LITE6_GRIPPER_KV)
    assert config.gripper.force_lower_n == pytest.approx(-LITE6_GRIPPER_DRIVE_LIMIT_N)
    assert config.gripper.force_upper_n == pytest.approx(LITE6_GRIPPER_DRIVE_LIMIT_N)
    assert LITE6_GRIPPER_HOLD_BIAS_GAP_M == pytest.approx(0.018)

    robot = MagicMock()
    configure_lite6_gripper_drive(robot, [6], [6, 7])
    robot.set_dofs_kp.assert_called_once()
    assert list(robot.set_dofs_kp.call_args.args[1]) == [6]
    robot.set_dofs_kv.assert_called_once()
    assert list(robot.set_dofs_kv.call_args.args[1]) == [6]
    robot.set_dofs_force_range.assert_called_once()
    assert list(robot.set_dofs_force_range.call_args.args[2]) == [6]
    robot.control_dofs_force.assert_called_once()
    assert list(robot.control_dofs_force.call_args.args[0].shape) == [1]
    assert list(robot.control_dofs_force.call_args.args[1]) == [7]


def _bio_g2_test_profile():
    from ufactory.config.models import GripperProfile
    from ufactory.grippers.bio_g2 import (
        BIO_GRIPPER_G2_CLOSED_GAP_M,
        BIO_GRIPPER_G2_OPEN_GAP_M,
        BIO_GRIPPER_G2_SIM_CLOSED_DRIVE,
        BIO_GRIPPER_G2_SIM_OPEN_DRIVE,
    )

    return GripperProfile(
        adapter="bio_g2",
        drive_joint="bio_gripper_g2_right_finger_joint",
        all_joint_names=("bio_gripper_g2_right_finger_joint", "bio_gripper_g2_left_finger_joint"),
        finger_link_names=("bio_gripper_g2_left_finger", "bio_gripper_g2_right_finger"),
        open_drive=BIO_GRIPPER_G2_SIM_OPEN_DRIVE,
        closed_drive=BIO_GRIPPER_G2_SIM_CLOSED_DRIVE,
        open_gap_m=BIO_GRIPPER_G2_OPEN_GAP_M,
        closed_gap_m=BIO_GRIPPER_G2_CLOSED_GAP_M,
        kp=16000.0,
        kv=40.0,
        force_lower_n=-40.0,
        force_upper_n=40.0,
        damping=0.0,
        frictionloss=0.0,
        real_command=True,
        feedback=True,
        closed_loop=False,
        allowed_contact_links=frozenset({"bio_gripper_g2_left_finger", "bio_gripper_g2_right_finger"}),
        tool_tip_offset_z_m=0.0595,
        finger_pad_below_center_m=0.0,
        finger_close_descent_m=0.0,
        grasp_table_clearance_m=0.05,
        grasp_height_extra_m=0.0,
    )


def test_bio_g2_adapter_mapping_and_sdk_contract():
    from ufactory.grippers.adapters import BioG2Adapter
    from ufactory.grippers.bio_g2 import bio_gripper_g2_gap_m_to_sim_drive

    profile = _bio_g2_test_profile()
    adapter = create_gripper_adapter(profile)
    assert isinstance(adapter, BioG2Adapter)
    arm = MagicMock()
    arm.get_bio_gripper_error.return_value = (0, 0)
    arm.set_bio_gripper_enable.return_value = 0
    arm.get_bio_gripper_g2_position.return_value = (0, 150)
    arm.set_bio_gripper_g2_position.return_value = 0
    assert adapter.prepare_real(arm) == 0
    arm.get_bio_gripper_g2_position.assert_called_once_with()

    for gap in (0.071, 0.08548, 0.110, 0.150):
        q = adapter.gap_to_drive(gap)
        assert q == pytest.approx(bio_gripper_g2_gap_m_to_sim_drive(gap), abs=1e-14)
        assert adapter.drive_to_gap(q) == pytest.approx(gap, abs=1e-12)
        assert adapter.send_real_gap(arm, gap) == 0
        arm.set_bio_gripper_g2_position.assert_called_with(
            pos=gap * 1000.0,
            speed=2000,
            force=50,
            wait=False,
            wait_motion=False,
        )

    # Deduplication skips sub-0.05 mm repeats; prepare_real resets deduplication state.
    calls_before = arm.set_bio_gripper_g2_position.call_count
    assert adapter.send_real_gap(arm, 0.15004) == 0
    assert arm.set_bio_gripper_g2_position.call_count == calls_before
    assert adapter.prepare_real(arm) == 0
    assert adapter.send_real_gap(arm, 0.150) == 0
    assert arm.set_bio_gripper_g2_position.call_count == calls_before + 1

    # Non-zero error / enable / pre-read codes propagate cleanly.
    arm.get_bio_gripper_error.return_value = (11, 0)
    assert adapter.prepare_real(arm) == 11
    arm.get_bio_gripper_error.return_value = (0, 0)
    arm.set_bio_gripper_enable.return_value = 12
    assert adapter.prepare_real(arm) == 12
    arm.set_bio_gripper_enable.return_value = 0
    arm.get_bio_gripper_g2_position.return_value = (13, 0)
    assert adapter.prepare_real(arm) == 13

    arm.get_bio_gripper_error.return_value = (0, 3)
    with pytest.raises(RuntimeError, match="recover manually"):
        adapter.prepare_real(arm)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"speed_rpm": 499}, "speed_rpm"),
        ({"speed_rpm": 4001}, "speed_rpm"),
        ({"speed_rpm": float("nan")}, "speed_rpm"),
        ({"speed_rpm": True}, "speed_rpm"),
        ({"force_percent": 0}, "force_percent"),
        ({"force_percent": 101}, "force_percent"),
        ({"force_percent": float("inf")}, "force_percent"),
        ({"force_percent": False}, "force_percent"),
    ],
)
def test_bio_g2_adapter_rejects_out_of_range_parameters(kwargs, match):
    from ufactory.grippers.adapters import BioG2Adapter

    with pytest.raises(ValueError, match=match):
        BioG2Adapter(_bio_g2_test_profile(), **kwargs)


def test_bio_g2_tcp_in_flange_constant_matches_composed_urdf_fk():
    pin = pytest.importorskip("pinocchio")
    import numpy as np

    from ufactory.grippers.bio_g2 import BIO_GRIPPER_G2_TCP_IN_FLANGE_M
    from ufactory.robots.assembly import compose_robot_urdf

    urdf = compose_robot_urdf("xarm6_1305", "bio_gripper_g2")
    model = pin.buildModelFromUrdf(str(urdf))
    data = model.createData()
    q = pin.neutral(model)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    flange = data.oMf[model.getFrameId("link6")]
    tcp = data.oMf[model.getFrameId("bio_gripper_g2_tcp_link")]
    measured = flange.actInv(tcp).translation
    err_mm = float(np.linalg.norm(measured - np.asarray(BIO_GRIPPER_G2_TCP_IN_FLANGE_M)) * 1000.0)
    assert err_mm <= 0.05


@pytest.mark.parametrize(("variant", "open_m"), [("robotiq_2f_85", 0.085), ("robotiq_2f_140", 0.140)])
def test_robotiq_adapter_mapping_and_sdk_safety_contract(variant: str, open_m: float) -> None:
    config = load_runtime_config("xarm6", gripper=variant)
    assert config.gripper is not None
    adapter = create_gripper_adapter(config.gripper)

    q_open = adapter.gap_to_drive(open_m)
    q_close = adapter.gap_to_drive(0.0)
    assert adapter.drive_to_gap(q_open) == pytest.approx(open_m, abs=1e-4)
    assert adapter.drive_to_gap(q_close) == pytest.approx(0.0, abs=1e-4)

    arm = MagicMock()
    arm.robotiq_status = {"gFLT": 0, "gSTA": 3}
    arm.robotiq_is_activated = True
    arm.robotiq_get_status.return_value = (0, [0] * 9)
    arm.robotiq_set_position.return_value = (0, [0] * 6)
    assert adapter.prepare_real(arm) == 0
    arm.robotiq_reset.assert_not_called()
    assert adapter.send_real_gap(arm, open_m) == 0
    arm.robotiq_set_position.assert_called_with(
        0,
        speed=128,
        force=128,
        wait=False,
        wait_motion=False,
        auto_enable=False,
    )
    assert adapter.send_real_gap(arm, 0.0) == 0
    arm.robotiq_set_position.assert_called_with(
        255,
        speed=128,
        force=128,
        wait=False,
        wait_motion=False,
        auto_enable=False,
    )

    # Faulted gripper raises cleanly without calling robotiq_reset().
    arm.robotiq_status = {"gFLT": 9, "gSTA": 0}
    arm.robotiq_is_activated = False
    with pytest.raises(RuntimeError, match="gFLT=9"):
        adapter.prepare_real(arm)
    arm.robotiq_reset.assert_not_called()

    # Unactivated healthy gripper activates explicitly without calling robotiq_reset().
    arm.robotiq_status = {"gFLT": 0, "gSTA": 0}
    arm.robotiq_set_activate.return_value = (0, [0] * 6)
    assert adapter.prepare_real(arm) == 0
    arm.robotiq_set_activate.assert_called_once_with(wait=True)
    arm.robotiq_reset.assert_not_called()


class _RobotiqStatusArm:
    """Fake SDK arm: each robotiq_get_status() publishes the next gFLT value in robotiq_status."""

    def __init__(self, faults: list[int], *, gsta: int = 3) -> None:
        self._faults = list(faults)
        self._gsta = gsta
        self.reads = 0
        self.robotiq_status: dict[str, int] = {}
        self.robotiq_is_activated = False
        self.activations = 0

    def robotiq_get_status(self, number_of_registers: int = 3) -> tuple[int, list[int]]:
        fault = self._faults[min(self.reads, len(self._faults) - 1)]
        self.reads += 1
        self.robotiq_status = {"gSTA": self._gsta, "gFLT": fault}
        return 0, [0] * 9

    def robotiq_set_activate(self, wait: bool = True) -> tuple[int, list[int]]:
        self.activations += 1
        return 0, [0] * 6


@pytest.mark.parametrize("variant", ["robotiq_2f_85", "robotiq_2f_140"])
def test_robotiq_prepare_real_rereads_stale_communication_timeout(variant: str, monkeypatch) -> None:
    from ufactory.grippers.adapters import RobotiqAdapter

    monkeypatch.setattr("ufactory.grippers.adapters.time.sleep", lambda _s: None)
    config = load_runtime_config("xarm6", gripper=variant)
    assert config.gripper is not None
    adapter = create_gripper_adapter(config.gripper)

    # First poll after an idle period reports 0x09; it clears once polling resumes (observed on a real 2F-85).
    arm = _RobotiqStatusArm([0x09, 0])
    assert adapter.prepare_real(arm) == 0
    assert (arm.reads, arm.activations) == (2, 0)

    # A communication fault that persists through every re-read is still refused.
    arm = _RobotiqStatusArm([0x09])
    with pytest.raises(RuntimeError, match="gFLT=9"):
        adapter.prepare_real(arm)
    assert arm.reads == 1 + RobotiqAdapter.COMM_TIMEOUT_REREADS

    # Any other fault is refused on the first read, without re-reading.
    arm = _RobotiqStatusArm([0x0E, 0])
    with pytest.raises(RuntimeError, match="gFLT=14"):
        adapter.prepare_real(arm)
    assert arm.reads == 1
