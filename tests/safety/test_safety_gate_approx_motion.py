"""Phase 2 SafetyGate integration tests for ArmMotionController and allow_approx_motion."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from ufactory.cli.pick_place import _build_program
from ufactory.config import load_runtime_config
from ufactory.control import ArmMotionController
from ufactory.manipulation.commands import build_arm_controller, model_and_hashes
from ufactory.safety import (
    CartesianStepController,
    CollisionResult,
    JacobianKinematicsBackend,
    ViolationType,
)
from ufactory.trajectory.preflight import create_safety_gate
from ufactory.trajectory.segments import JointLimits, Program, make_movel, make_movej


class _PassCollision:
    def check(self, q, *, stage, gripper_drive=None):
        return CollisionResult(False, 1.0, "link1", "link3")

    def check_all(self, q, *, stage, gripper_drive=None):
        return (CollisionResult(False, 1.0, "link1", "link3"),)


class _PlainKinematics:
    def forward(self, q):
        return np.asarray((0.3, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0), dtype=np.float64)

    def inverse(self, pose, seed):
        return np.asarray(seed, dtype=np.float64)


def test_arm_motion_controller_satisfies_safety_protocols() -> None:
    ctrl = ArmMotionController.from_robot("uf850")
    assert isinstance(ctrl, JacobianKinematicsBackend)
    assert isinstance(ctrl, CartesianStepController)
    assert ctrl.is_approx_motion is False


def test_safety_gate_uf850_wrist_singularity_fails_off_and_passes_on() -> None:
    cal = Path("assets/urdf/robots/uf850/kinematics/user/uf850_kinematics_F54B07.yaml")
    cfg_off = load_runtime_config("uf850")
    urdf, _, cal_hash = model_and_hashes(cfg_off, cal, None)
    ctrl_off = build_arm_controller(cfg_off, urdf)

    # Start right above the UF850 wrist singularity and descend through z=208.6 mm
    p_a = np.array([0.300, 0.0, 0.2886], dtype=np.float64)
    p_b = np.array([0.300, 0.0, 0.1886], dtype=np.float64)
    _, q_a_list = ctrl_off.get_inverse_kinematics(
        [300.0, 0.0, 288.6, np.pi, 0.0, 0.0], input_is_radian=True, return_is_radian=True
    )
    q_a = np.asarray(q_a_list, dtype=np.float64)
    limits = JointLimits(
        cfg_off.motion.joint_speed_rad_s,
        cfg_off.motion.joint_acceleration_rad_s2,
        cfg_off.motion.cartesian_speed_m_s,
        cfg_off.motion.cartesian_acceleration_m_s2,
    )
    prog = Program(
        segments=[
            make_movej(q_a, q_a, rate=50.0, limits=limits, label="start"),
            make_movel(p_a, p_b, rate=50.0, limits=limits, label="descend"),
        ],
        rate=50.0,
        robot_key=cfg_off.robot.key,
    )

    gate_off = create_safety_gate(
        cfg_off,
        kinematics=ctrl_off,
        collision=_PassCollision(),
        calibration_sha256=cal_hash,
        scene_sha256="0" * 64,
        urdf_path=urdf,
    )
    rep_off = gate_off.preflight(prog, executor="servo_cartesian")
    assert not rep_off.passed
    violation_types = {v.type for v in rep_off.violations}
    assert ViolationType.VELOCITY in violation_types
    assert ViolationType.ACCELERATION in violation_types
    tl_off = next(c for c in rep_off.checks if c.name == "timeline")
    assert tl_off.details["allow_approx_motion"] is False
    assert tl_off.details["approx_steps"] == 0
    assert rep_off.warnings == ()

    cfg_on = load_runtime_config("uf850", overrides={"motion.allow_approx_motion": True})
    assert cfg_on.sha256 != cfg_off.sha256
    ctrl_on = build_arm_controller(cfg_on, urdf)
    gate_on = create_safety_gate(
        cfg_on,
        kinematics=ctrl_on,
        collision=_PassCollision(),
        calibration_sha256=cal_hash,
        scene_sha256="0" * 64,
        urdf_path=urdf,
    )
    rep_on = gate_on.preflight(prog, executor="servo_cartesian")
    assert rep_on.passed, rep_on.violations
    tl_on = next(c for c in rep_on.checks if c.name == "timeline")
    assert tl_on.details["allow_approx_motion"] is True
    assert tl_on.details["approx_steps"] > 0
    assert 0.0 <= tl_on.details["max_approx_position_error_m"] < 0.002
    assert 0.0 < tl_on.details["max_approx_orientation_error_rad"] < 0.05
    assert any("allow_approx_motion activated" in w for w in rep_on.warnings)


def test_safety_gate_fails_closed_on_approx_motion_mismatch() -> None:
    cfg_on = load_runtime_config("uf850", overrides={"motion.allow_approx_motion": True})
    cfg_off = load_runtime_config("uf850")
    q0 = np.asarray(cfg_off.arm.default_qpos_rad, dtype=np.float64)
    limits = JointLimits(1.0, 12.0, 0.15, 0.8)
    prog = Program(
        [make_movej(q0, q0, rate=50.0, limits=limits, label="start")], rate=50.0, robot_key=cfg_off.robot.key
    )

    # 1. allow_approx_motion=True but kinematics is not a CartesianStepController
    gate_plain = create_safety_gate(
        cfg_on,
        kinematics=_PlainKinematics(),
        collision=_PassCollision(),
        calibration_sha256="0" * 64,
        scene_sha256="0" * 64,
    )
    rep_plain = gate_plain.preflight(prog, executor="servo_cartesian")
    assert not rep_plain.passed
    assert ViolationType.BACKEND_UNAVAILABLE in {v.type for v in rep_plain.violations}

    # 2. config enables approx motion, controller disables it
    ctrl_off = ArmMotionController.from_robot("uf850", allow_approx_motion=False)
    gate_mismatch_1 = create_safety_gate(
        cfg_on,
        kinematics=ctrl_off,
        collision=_PassCollision(),
        calibration_sha256="0" * 64,
        scene_sha256="0" * 64,
    )
    rep_m1 = gate_mismatch_1.preflight(prog, executor="servo_cartesian")
    assert not rep_m1.passed
    assert ViolationType.INVALID_INPUT in {v.type for v in rep_m1.violations}

    # 3. config disables approx motion, controller enables it
    ctrl_on = ArmMotionController.from_robot("uf850", allow_approx_motion=True)
    gate_mismatch_2 = create_safety_gate(
        cfg_off,
        kinematics=ctrl_on,
        collision=_PassCollision(),
        calibration_sha256="0" * 64,
        scene_sha256="0" * 64,
    )
    rep_m2 = gate_mismatch_2.preflight(prog, executor="servo_cartesian")
    assert not rep_m2.passed
    assert ViolationType.INVALID_INPUT in {v.type for v in rep_m2.violations}


def test_safety_gate_calibrated_xarm5_pick_place_passes_with_position_priority_ik() -> None:
    cal = Path("assets/urdf/robots/xarm5/kinematics/user/xarm5_kinematics_D43A02.yaml")
    cfg = load_runtime_config("xarm5")
    urdf, _, cal_hash = model_and_hashes(cfg, cal, None)
    ctrl = build_arm_controller(cfg, urdf)
    prog = _build_program(cfg, kinematics=ctrl)
    gate = create_safety_gate(
        cfg,
        kinematics=ctrl,
        collision=_PassCollision(),
        calibration_sha256=cal_hash,
        scene_sha256="0" * 64,
        urdf_path=urdf,
    )
    rep = gate.preflight(prog, executor="servo_cartesian")
    assert rep.passed, rep.violations
    # Verify default config uses allow_approx_motion=False
    cfg_no_req = replace(cfg, safety=replace(cfg.safety, require_collision=False))
    assert cfg_no_req.motion.allow_approx_motion is False
