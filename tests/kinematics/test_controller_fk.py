"""Calibration export: joint origins identified from (synthetic) controller FK."""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from ufactory.kinematics import build_calibrated_urdf, load_kinematics_calibration
from ufactory.kinematics.controller_fk import (
    CalibrationIdentificationError,
    chain_pose,
    controller_flange_fk,
    identify_from_controller_fk,
    load_nominal_chain,
    matrix_rpy,
    rotation_angle,
    sweep_angles,
)
from ufactory.kinematics.tcp_offset import matrix_to_pose, pose_to_matrix, rpy_matrix
from ufactory.robots.registry import ROBOT_PROFILES, get_robot_profile, joint_names

ROOT = Path(__file__).resolve().parents[2]
ROBOTS = ("xarm5", "xarm6", "xarm7", "uf850", "lite6")


def _chain(robot: str):
    profile = get_robot_profile(robot)
    return load_nominal_chain(profile.assets_dir / profile.default_urdf, joint_names(profile))


def _flange(offset_m: float, tilt_rad: float) -> np.ndarray:
    flange = np.eye(4)
    flange[:3, :3] = rpy_matrix(tilt_rad, 0.5 * tilt_rad, 0.25 * tilt_rad)
    flange[:3, 3] = (0.6 * offset_m, -0.8 * offset_m, 0.0)
    return flange


def _controller(chain, *, seed=1, offset_m=200e-6, tilt_rad=math.radians(0.02), perturb=True):
    """Rigid controller chain: perturbed joint origins, flange off the last axis."""

    rng = np.random.default_rng(seed)
    origins = chain.origins.copy()
    if perturb:
        origins[:, :3] += rng.uniform(-1e-3, 1e-3, size=(len(origins), 3))
        origins[:, 3:] += rng.uniform(-math.radians(0.2), math.radians(0.2), size=(len(origins), 3))
    flange = _flange(offset_m, tilt_rad)
    return lambda q: chain_pose(origins, chain.axes, q) @ flange


@pytest.mark.parametrize("robot", ROBOTS)
def test_identified_origins_reproduce_the_controller(robot):
    chain = _chain(robot)
    controller = _controller(chain)
    result = identify_from_controller_fk(controller, chain, verification_samples=40)

    assert result.rigid_chain_error_m < 1e-9
    assert result.flange_axis_offset_m == pytest.approx(200e-6, abs=1e-9)
    assert result.flange_axis_tilt_rad == pytest.approx(math.hypot(1.0, 0.5) * math.radians(0.02), rel=1e-3)
    # Exact whenever the last joint is at 0; otherwise within twice the flange offset.
    rng = np.random.default_rng(7)
    for q in rng.uniform(-1.0, 1.0, size=(20, len(chain.joint_names))):
        q[-1] = 0.0
        model, reference = chain_pose(result.origins, chain.axes, q), controller(q)
        assert np.linalg.norm(model[:3, 3] - reference[:3, 3]) < 1e-9
        assert rotation_angle(model[:3, :3], reference[:3, :3]) < 1e-9
    assert result.max_position_error_m <= 2 * 200e-6 + 1e-9


@pytest.mark.parametrize("robot", ROBOTS)
def test_nominal_controller_gives_the_nominal_urdf_origins(robot):
    chain = _chain(robot)
    result = identify_from_controller_fk(
        _controller(chain, perturb=False, offset_m=0.0, tilt_rad=0.0), chain, verification_samples=10
    )

    assert result.max_position_error_m < 1e-9
    np.testing.assert_allclose(result.origins[:, :3], chain.origins[:, :3], atol=1e-9)
    for identified, nominal in zip(result.origins, chain.origins, strict=True):
        assert rotation_angle(rpy_matrix(*identified[3:]), rpy_matrix(*nominal[3:])) < 1e-9


def test_non_rigid_controller_wrong_model_and_large_flange_offset_are_refused():
    chain = _chain("xarm6")
    rigid = _controller(chain)

    def sagging(q):
        pose = rigid(q)
        pose[2, 3] -= 1e-4 * q[1] * q[2]
        return pose

    with pytest.raises(CalibrationIdentificationError, match="not a rigid chain"):
        identify_from_controller_fk(sagging, chain, verification_samples=20)
    with pytest.raises(CalibrationIdentificationError, match="nominal URDF"):
        identify_from_controller_fk(_controller(_chain("uf850")), chain, verification_samples=20)
    with pytest.raises(CalibrationIdentificationError, match="miss controller FK"):
        identify_from_controller_fk(_controller(chain, offset_m=0.8e-3), chain, verification_samples=40)


def test_sweep_angles_cover_one_sided_joints_and_reject_tiny_ranges():
    lower, upper = _chain("xarm5").lower[2], _chain("xarm5").upper[2]
    angles = sweep_angles(float(lower), float(upper))

    assert len(angles) >= 4 and all(lower <= a <= upper and abs(a) > 0.04 for a in angles)
    assert sweep_angles(-3.0, 3.0) == (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5)
    with pytest.raises(CalibrationIdentificationError):
        sweep_angles(-0.01, 0.01)


@pytest.mark.parametrize("rpy", [(0.3, -0.2, 1.1), (1.2, math.pi / 2, 0.4), (-0.7, -math.pi / 2, 0.0)])
def test_matrix_rpy_round_trips_including_gimbal_lock(rpy):
    rotation = rpy_matrix(*rpy)

    assert rotation_angle(rpy_matrix(*matrix_rpy(rotation)), rotation) < 1e-12


class FakeArm:
    """Controller double: SDK FK returns the TCP pose (mm, rad) of a synthetic xArm6."""

    axis, device_type = 6, 6

    def __init__(self, *, world_offset=(0.0,) * 6, fail=False):
        chain = _chain("xarm6")
        self.flange = _controller(chain)
        self.tcp_offset = [1.0, -2.0, 174.0, 0.0, 0.01, 0.0]
        self.world_offset = list(world_offset)
        self.fail = fail
        self.calls = 0

    def get_robot_sn(self):
        return 0, "XI130505TEST01"

    def get_forward_kinematics(self, angles, input_is_radian=None, return_is_radian=None):
        assert input_is_radian and return_is_radian
        self.calls += 1
        if self.fail:
            return 1, []
        flange = self.flange(np.asarray(angles))
        flange[:3, 3] *= 1000.0
        return 0, matrix_to_pose(flange @ pose_to_matrix(self.tcp_offset)).tolist()


def test_controller_flange_fk_removes_the_tcp_offset():
    arm = FakeArm()
    q = np.asarray([0.1, -0.2, 0.3, 0.4, -0.5, 0.6])
    flange = controller_flange_fk(arm, arm.tcp_offset)(q)

    np.testing.assert_allclose(flange, arm.flange(q), atol=1e-12)
    with pytest.raises(RuntimeError, match="controller FK failed"):
        controller_flange_fk(FakeArm(fail=True), arm.tcp_offset)(q)


def _script():
    spec = importlib.util.spec_from_file_location(
        "gen_kinematics_params", ROOT / "scripts" / "gen_kinematics_params.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_writes_a_loadable_calibration_that_reproduces_the_controller(tmp_path, monkeypatch):
    script = _script()
    monkeypatch.setattr(script, "_read_world_offset", lambda arm: np.asarray(arm.world_offset))
    arm = FakeArm()
    args = argparse.Namespace(kinematics_suffix=None, output_dir=str(tmp_path), force=False)

    assert script._export(arm, args) == 0
    path = tmp_path / "xarm6_kinematics_TEST01.yaml"
    calibration = load_kinematics_calibration(str(path), robot_key="xarm6", serial_number="XI130505TEST01")
    profile = ROBOT_PROFILES["xarm6_1305"]
    urdf = build_calibrated_urdf(
        str(profile.assets_dir / profile.default_urdf), calibration, joint_count=6, output_dir=str(tmp_path)
    )
    calibrated = load_nominal_chain(urdf, joint_names(profile))
    q = np.asarray([0.2, -0.4, 0.3, 1.0, -0.6, 0.0])
    np.testing.assert_allclose(chain_pose(calibrated.origins, calibrated.axes, q), arm.flange(q), atol=1e-9)


def test_export_refuses_a_world_offset_and_writes_nothing(tmp_path, capsys):
    script = _script()
    arm = FakeArm(world_offset=(10.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    args = argparse.Namespace(kinematics_suffix=None, output_dir=str(tmp_path), force=False)

    assert script._export(arm, args) == 1
    assert "world_offset" in capsys.readouterr().out
    assert arm.calls == 0 and not list(tmp_path.iterdir())
