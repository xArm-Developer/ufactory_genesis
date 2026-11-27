from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from ufactory.control import (
    CalibratedKinematicsChain,
    pose_to_se3,
    quat_xyzw_to_rotation,
    rotation_to_quat_xyzw,
    se3_to_pose_m_quat,
    se3_to_pose_mm_rpy,
    slerp_rotation,
    so3_log,
)
from ufactory.kinematics.controller_fk import axis_angle_matrix

ALL_ROBOTS = ("xarm5", "xarm6", "xarm7", "uf850", "lite6")


def test_so3_quaternion_and_slerp_roundtrip() -> None:
    rng = np.random.default_rng(42)
    for angle in (0.0, 1e-9, 0.35, 1.8, math.pi - 1e-6, math.pi - 1e-10):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        rot = axis_angle_matrix(axis, angle)
        q = rotation_to_quat_xyzw(rot)
        assert np.allclose(quat_xyzw_to_rotation(q), rot, atol=1e-10)
        omega = so3_log(rot)
        assert np.allclose(np.linalg.norm(omega), angle, atol=1e-6)
        if angle > 1e-6:
            rebuilt = axis_angle_matrix(omega / np.linalg.norm(omega), float(np.linalg.norm(omega)))
            assert np.allclose(rebuilt, rot, atol=1e-6)

    # SLERP midpoint check
    ra = axis_angle_matrix(np.array([0.0, 0.0, 1.0]), 0.0)
    rb = axis_angle_matrix(np.array([0.0, 0.0, 1.0]), 1.2)
    rmid = slerp_rotation(ra, rb, 0.5)
    assert np.allclose(rmid, axis_angle_matrix(np.array([0.0, 0.0, 1.0]), 0.6), atol=1e-12)
    assert np.allclose(slerp_rotation(ra, rb, 0.0), ra, atol=1e-12)
    assert np.allclose(slerp_rotation(ra, rb, 1.0), rb, atol=1e-12)

    # Pose conversions
    pose_mm = np.array([300.0, -50.0, 220.0, math.pi, 0.1, -0.3])
    t = pose_to_se3(pose_mm, position_in_mm=True)
    assert np.allclose(se3_to_pose_mm_rpy(t)[:3], pose_mm[:3], atol=1e-9)
    p7 = se3_to_pose_m_quat(t)
    assert np.allclose(pose_to_se3(p7), t, atol=1e-12)


@pytest.mark.parametrize("robot", ALL_ROBOTS)
def test_analytical_jacobian_matches_finite_difference_all_5_robots(robot: str) -> None:
    chain = CalibratedKinematicsChain.from_robot(robot, tcp_offset_mm_rpy=[12.0, -8.0, 45.0, 0.1, -0.2, 0.3])
    rng = np.random.default_rng(abs(hash(robot)) % (2**32))
    eps = 1e-6
    for _ in range(5):
        lo = np.maximum(chain.lower + 0.1, -1.5)
        hi = np.minimum(chain.upper - 0.1, 1.5)
        q = rng.uniform(lo, hi)
        t0, jac = chain.forward_and_jacobian(q, include_tcp=True)
        jac_fd = np.zeros_like(jac)
        for j in range(chain.dof):
            qp = q.copy()
            qm = q.copy()
            qp[j] += eps
            qm[j] -= eps
            tp = chain.forward_matrix(qp, include_tcp=True)
            tm = chain.forward_matrix(qm, include_tcp=True)
            jac_fd[:3, j] = (tp[:3, 3] - tm[:3, 3]) / (2.0 * eps)
            jac_fd[3:, j] = so3_log(tp[:3, :3] @ tm[:3, :3].T) / (2.0 * eps)
        assert np.allclose(jac, jac_fd, atol=1e-6, rtol=1e-5)
        assert np.allclose(chain.tcp_target_to_flange(t0), chain.forward_matrix(q, include_tcp=False), atol=1e-12)


@pytest.mark.parametrize("robot", ALL_ROBOTS)
def test_pure_numpy_fk_and_jacobian_match_pinocchio(robot: str) -> None:
    pin = pytest.importorskip("pinocchio")
    from ufactory.config import RepositoryAssetStore
    from ufactory.robots.assembly import resolve_robot_urdf
    from ufactory.robots.registry import PROJECT_ROOT, get_robot_profile

    profile = get_robot_profile(robot)
    store = RepositoryAssetStore.discover()
    assets_rel = profile.assets_dir.relative_to(PROJECT_ROOT).as_posix()
    urdf = resolve_robot_urdf(profile.key, profile.default_urdf, assets_dir=assets_rel, store=store)
    chain = CalibratedKinematicsChain.from_robot(robot, urdf_path=urdf)

    model = pin.buildModelFromUrdf(str(urdf))
    data = model.createData()
    fid = model.getFrameId(chain.ee_link)

    rng = np.random.default_rng(1234 + chain.dof)
    lo = np.maximum(chain.lower + 0.1, -1.5)
    hi = np.minimum(chain.upper - 0.1, 1.5)
    for _ in range(5):
        q = rng.uniform(lo, hi)
        t_np, j_np = chain.forward_and_jacobian(q, include_tcp=False)
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)
        oMf = data.oMf[fid]
        j_pin = pin.getFrameJacobian(model, data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        assert np.allclose(t_np[:3, 3], oMf.translation, atol=1e-11)
        assert np.allclose(t_np[:3, :3], oMf.rotation, atol=1e-11)
        assert np.allclose(j_np, j_pin, atol=1e-11)


def test_parameter_compensation_yaml_updates_fk(tmp_path: Path) -> None:
    nom = CalibratedKinematicsChain.from_robot("uf850")
    joints_dict = {}
    for idx, name in enumerate(nom.joint_names):
        row = nom.nominal_origins[idx].copy()
        if idx == 3:
            row[0] += 0.0025  # +2.5 mm perturbation
            row[4] += 0.004  # +4 mrad pitch tilt
        joints_dict[name] = {
            "x": float(row[0]),
            "y": float(row[1]),
            "z": float(row[2]),
            "roll": float(row[3]),
            "pitch": float(row[4]),
            "yaw": float(row[5]),
        }
    cal_file = tmp_path / "uf850_kinematics_TEST1234.yaml"
    cal_file.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "robot_key": "uf850",
                "serial_number": "FX851005TEST1234",
                "units": {"position": "m", "angle": "rad"},
                "joints": joints_dict,
            }
        ),
        encoding="utf-8",
    )
    comp = CalibratedKinematicsChain.from_robot("uf850", calibration_path=cal_file)
    assert comp.is_calibrated
    assert comp.serial_number == "FX851005TEST1234"
    q = np.array([0.1, -0.4, -0.5, 0.2, 0.3, -0.1])
    t_nom = nom.forward_matrix(q)
    t_comp = comp.forward_matrix(q)
    pos_diff, rot_diff = nom.pose_error(t_nom, t_comp)
    assert pos_diff > 0.0015
    assert rot_diff > 0.002
