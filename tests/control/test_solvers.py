from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from ufactory.control import (
    ApproxMotionConfig,
    CalibratedKinematicsChain,
    pose_to_se3,
    solve_cartesian_step,
    solve_exact_ik,
    solve_spherical_wrist_analytical,
)


@pytest.mark.parametrize("robot", ("lite6", "uf850"))
def test_analytical_pieper_ik_recovers_all_branches_and_exact_pose(robot: str) -> None:
    chain = CalibratedKinematicsChain.from_robot(robot)
    rng = np.random.default_rng(77 if robot == "lite6" else 88)
    tested = 0
    while tested < 25:
        q_true = np.array(
            [
                rng.uniform(-2.0, 2.0),
                rng.uniform(-1.2, 1.2),
                rng.uniform(0.3, 2.5) if robot == "lite6" else rng.uniform(-2.5, -0.3),
                rng.uniform(-2.5, 2.5),
                rng.uniform(-1.5, 1.5),
                rng.uniform(-2.5, 2.5),
            ],
            dtype=np.float64,
        )
        if abs(q_true[4]) < 0.08:
            continue
        target = chain.forward_matrix(q_true, include_tcp=False)
        branches = solve_spherical_wrist_analytical(chain, target, seed_q_rad=q_true + rng.normal(0.0, 0.03, size=6))
        assert len(branches) >= 2
        for cand in branches:
            pos_err, rot_err = chain.pose_error(target, chain.forward_matrix(cand, include_tcp=False))
            # Nominal URDF uses rounded 1.5708 for pi/2 (~3.7e-6 rad -> ~2 um residual before polish)
            assert pos_err < 2e-5
            assert rot_err < 5e-5
        # Closest branch to slightly perturbed seed must recover q_true closely
        # (URDF 1.5708 rounding introduces up to ~2e-4 rad before polish when r_xy is small)
        assert float(np.max(np.abs(branches[0] - q_true))) < 5e-4

        # Full exact solver (analytical + polish) reaches < 1e-7 m / 1e-5 rad
        res = solve_exact_ik(chain, target, seed_q_rad=q_true + rng.normal(0.0, 0.05, size=6), include_tcp=False)
        assert res.position_error_m < 1e-7
        assert res.orientation_error_rad < 1e-5
        assert float(np.max(np.abs(res.q_rad - q_true))) < 1e-4
        assert res.solver.startswith("analytical-pieper")
        tested += 1


@pytest.mark.parametrize("robot", ("lite6", "uf850"))
def test_analytical_ik_handles_wrist_singularity_and_calibrated_polish(robot: str, tmp_path: Path) -> None:
    nom = CalibratedKinematicsChain.from_robot(robot)
    # Exact wrist singularity q5 = 0
    q_sing = np.array([0.2, -0.3, 1.0 if robot == "lite6" else -1.0, 0.5, 0.0, -0.4], dtype=np.float64)
    target_sing = nom.forward_matrix(q_sing, include_tcp=False)
    branches = solve_spherical_wrist_analytical(nom, target_sing, seed_q_rad=q_sing)
    assert len(branches) >= 1
    pos_err, rot_err = nom.pose_error(target_sing, nom.forward_matrix(branches[0], include_tcp=False))
    assert pos_err < 2e-5 and rot_err < 5e-5

    # Synthetic calibration perturbation (breaks exact spherical wrist by ~1 mm & 3 mrad)
    joints = {}
    for idx, name in enumerate(nom.joint_names):
        row = nom.nominal_origins[idx].copy()
        if idx in (3, 4, 5):
            row[0] += 0.0004 * (idx - 2)
            row[1] -= 0.0003
            row[4] += 0.0012
        joints[name] = {k: float(v) for k, v in zip(("x", "y", "z", "roll", "pitch", "yaw"), row, strict=True)}
    cal_path = tmp_path / f"{robot}_kinematics_CAL00001.yaml"
    cal_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "robot_key": nom.robot_key,
                "serial_number": "TESTSERIAL0001",
                "units": {"position": "m", "angle": "rad"},
                "joints": joints,
            }
        ),
        encoding="utf-8",
    )
    comp = CalibratedKinematicsChain.from_robot(
        robot,
        calibration_path=cal_path,
        tcp_offset_mm_rpy=[0.0, 0.0, 80.0, 0.0, 0.0, 0.2],
    )
    q_ref = np.array([0.1, -0.4, 0.9 if robot == "lite6" else -0.9, 0.2, 0.5, -0.3], dtype=np.float64)
    target = comp.forward_matrix(q_ref, include_tcp=True)
    res = solve_exact_ik(comp, target, seed_q_rad=q_ref + np.array([0.1, -0.1, 0.1, -0.1, 0.1, 0.1]), include_tcp=True)
    assert res.position_error_m < 1e-7
    assert res.orientation_error_rad < 1e-5


@pytest.mark.parametrize("robot", ("xarm6", "xarm7"))
def test_iterative_ik_xarm6_and_xarm7_converge_with_tcp_offset(robot: str) -> None:
    chain = CalibratedKinematicsChain.from_robot(robot, tcp_offset_mm_rpy=[0.0, 0.0, 174.0, 0.0, 0.0, 0.0])
    q_true = (
        np.array([0.2, -0.4, -0.6, 0.3, 1.0, -0.2], dtype=np.float64)
        if robot == "xarm6"
        else np.array([0.2, -0.4, 0.1, 0.8, -0.1, 1.0, -0.2], dtype=np.float64)
    )
    target = chain.forward_matrix(q_true, include_tcp=True)
    seed = q_true + 0.08 * np.ones_like(q_true)
    res = solve_exact_ik(chain, target, seed_q_rad=seed, include_tcp=True)
    assert res.position_error_m < 1e-7
    assert res.orientation_error_rad < 1e-5
    if robot == "xarm7":
        # Redundant arm stays close to seed (no wild elbow drift)
        assert float(np.linalg.norm(res.q_rad - seed)) < 0.35


def test_xarm7_cold_start_converges_without_nullspace_chord_stall() -> None:
    chain = CalibratedKinematicsChain.from_robot("xarm7")
    target = pose_to_se3([300.0, 0.0, 260.0, math.pi, 0.0, 0.0], position_in_mm=True)
    seed = np.array([0.0, -0.5, 0.0, 0.0, 0.5, 0.0, 0.0], dtype=np.float64)
    res = solve_exact_ik(chain, target, seed_q_rad=seed, include_tcp=False, cold_start_fallback=True)
    assert res.solver == "iterative-7dof-redundant"
    assert res.position_error_m < 1e-7
    assert res.orientation_error_rad < 1e-5


def test_xarm5_position_priority_ik_tolerates_non_coplanar_calibration(tmp_path: Path) -> None:
    nom = CalibratedKinematicsChain.from_robot("xarm5")
    # Introduce realistic non-coplanar ~1.5 mrad roll tilt on joint 2/3
    joints = {}
    for idx, name in enumerate(nom.joint_names):
        row = nom.nominal_origins[idx].copy()
        if idx == 1:
            row[4] += 0.0015
        if idx == 4:
            row[0] += 0.0005
        joints[name] = {k: float(v) for k, v in zip(("x", "y", "z", "roll", "pitch", "yaw"), row, strict=True)}
    cal_path = tmp_path / "xarm5_kinematics_TEST0005.yaml"
    cal_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "robot_key": nom.robot_key,
                "serial_number": "XF130505TEST05",
                "units": {"position": "m", "angle": "rad"},
                "joints": joints,
            }
        ),
        encoding="utf-8",
    )
    comp = CalibratedKinematicsChain.from_robot("xarm5", calibration_path=cal_path)
    target = pose_to_se3([300.0, 0.0, 200.0, math.pi, 0.0, 0.0], position_in_mm=True)
    seed = np.array([0.0, -0.5, -0.8, 1.3, 0.0], dtype=np.float64)
    res = solve_exact_ik(comp, target, seed_q_rad=seed, include_tcp=False)
    assert res.solver == "iterative-5dof-pos-priority"
    assert res.position_error_m < 1e-6  # Position priority achieves < 1 um
    assert res.orientation_error_rad < 5e-3  # Accepts irreducible ~1.5 mrad roll


def test_approx_dls_step_passes_wrist_singularity_and_fails_closed_on_huge_error() -> None:
    chain = CalibratedKinematicsChain.from_robot("uf850")
    # Put arm right next to wrist singularity (q5 = 0.003 rad)
    q_start = np.array([0.0, -0.4, -0.4, 0.02, 0.003, -0.02], dtype=np.float64)
    t0 = chain.forward_matrix(q_start, include_tcp=False)
    t_next = t0.copy()
    t_next[2, 3] -= 0.003  # 3 mm vertical step (50 Hz at 150 mm/s)

    cfg = ApproxMotionConfig(damping=1.0, singular_value_threshold=0.08, trigger_joint_speed_rad_s=0.63)
    res = solve_cartesian_step(
        chain,
        t_next,
        q_start,
        allow_approx_motion=True,
        approx_config=cfg,
        dt_s=0.02,
        include_tcp=False,
    )
    assert res.used_approx
    assert res.position_error_m < 0.0005
    assert res.orientation_error_rad < math.radians(1.0)
    # Wrist J4/J6 stay damped instead of jumping wildly
    assert abs(res.q_rad[3] - q_start[3]) < 0.02
    assert abs(res.q_rad[5] - q_start[5]) < 0.02

    # Unreachable giant orientation jump must fail closed even with allow_approx_motion=True
    t_wild = t0.copy()
    t_wild[:3, :3] = t0[:3, :3] @ axis_angle_matrix_y(0.6)
    with pytest.raises(ValueError, match="exceeds safety ceiling"):
        solve_cartesian_step(
            chain,
            t_wild,
            q_start,
            allow_approx_motion=True,
            approx_config=cfg,
            dt_s=0.02,
            include_tcp=False,
        )


def axis_angle_matrix_y(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def test_nominal_uf850_continuous_tracking_across_q5_zero_avoids_urdf_rounding_spike() -> None:
    from ufactory.control import ArmMotionController
    from ufactory.kinematics.orientation import GRIPPER_DOWN_QUAT_XYZW
    from ufactory.safety.statistics import compute_motion_statistics
    from ufactory.trajectory.profile import linear_cartesian_samples

    ctrl = ArmMotionController.from_robot("uf850", allow_approx_motion=False)
    _, q0_list = ctrl.get_inverse_kinematics(
        [300.0, 0.0, 288.6, math.pi, 0.0, 0.0], input_is_radian=True, return_is_radian=True
    )
    q = np.asarray(q0_list, dtype=np.float64)
    pts, _, _ = linear_cartesian_samples(
        np.array([0.300, 0.0, 0.2886]),
        np.array([0.300, 0.0, 0.1886]),
        rate=50.0,
        v_max=0.15,
        a_max=0.8,
    )
    qs = [q.copy()]
    xyzs = [ctrl.forward(q)[:3]]
    for row in pts:
        pose = np.concatenate((row, np.asarray(GRIPPER_DOWN_QUAT_XYZW)))
        q = ctrl.solve_step(pose, q, dt_s=0.02, include_tcp=False).q_rad
        qs.append(q.copy())
        xyzs.append(ctrl.forward(q)[:3])
    stats = compute_motion_statistics(np.stack(qs), np.stack(xyzs), rate_hz=50.0)
    assert stats.max_joint_speed_rad_s < 1.0
    assert stats.max_joint_acceleration_rad_s2 < 12.0
