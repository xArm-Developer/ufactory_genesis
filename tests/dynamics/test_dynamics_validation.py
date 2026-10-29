"""Unit tests for enterprise dynamics validation helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ufactory.dynamics.analysis import (
    build_dynamics_sample,
    classify_torque_result,
    validate_urdf_dynamics,
)
from ufactory.dynamics.cli import run_sim_collision_chain
from ufactory.dynamics.plot import torque_plot_layout, write_torque_plot
from ufactory.dynamics.poses import dynamics_default_configs, xarm6_default_dynamics_configs
from ufactory.dynamics.report import (
    ABS_ERR_LIMITS,
    REPORT_SCHEMA_VERSION,
    DynamicsRunConfig,
    GenesisDynamicsSample,
    SafePose,
    ValidationStatus,
    compare_report_records,
    dynamics_report_paths,
    joint_torque_rows,
    read_report_records,
    sanitize_report_identity,
    torque_summary,
    write_csv_report,
    write_jsonl_report,
)
from ufactory.dynamics.cli import print_compare_table
from ufactory.kinematics.calibration import build_calibrated_urdf
from ufactory.robots.paths import robot_urdf, xarm6_1305_urdf
from ufactory.robots.runtime import DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S, get_robot_runtime_profile
from ufactory.hardware.session import RealRobotSession


def test_xarm6_urdf_dynamics_static_checks_have_no_errors():
    issues = validate_urdf_dynamics(xarm6_1305_urdf())
    assert [issue for issue in issues if issue.severity == "ERROR"] == []


def test_uf850_urdf_dynamics_static_checks_have_no_errors():
    issues = validate_urdf_dynamics(robot_urdf("uf850"))
    assert [issue for issue in issues if issue.severity == "ERROR"] == []


def test_sim_only_robot_urdf_dynamics_static_checks_have_no_errors():
    for robot_key in ("lite6", "xarm5", "xarm7"):
        issues = validate_urdf_dynamics(robot_urdf(robot_key))
        assert [issue for issue in issues if issue.severity == "ERROR"] == []


@pytest.mark.parametrize(
    ("robot_key", "expected_dof"),
    [
        ("xarm7", 7),
        ("uf850", 6),
        ("lite6", 6),
        ("xarm5", 5),
    ],
)
def test_non_xarm6_default_pose_set_is_balanced_hardware_profile(robot_key: str, expected_dof: int):
    runtime = get_robot_runtime_profile(robot_key)
    configs = dynamics_default_configs(robot_key, include_stress=True)
    names = {name for name, _ in configs}

    assert runtime.dynamics.supports_hardware_validation
    assert len(configs) == 20
    assert names == {str(i) for i in range(20)}
    assert {len(q) for _, q in configs} == {expected_dof}
    assert len(runtime.dynamics.abs_err_limits) == runtime.model.dof
    assert len(runtime.arm.effort_limits) == runtime.model.dof
    assert len(runtime.arm.kp) == runtime.model.dof
    assert len(runtime.arm.kv) == runtime.model.dof


def test_default_pose_set_is_enterprise_scale_and_excludes_stress_pose():
    configs = xarm6_default_dynamics_configs()
    names = {name for name, _ in configs}
    assert len(configs) == 20
    assert names == {str(i) for i in range(20)}
    stress_names = {name for name, _ in xarm6_default_dynamics_configs(include_stress=True)}
    assert stress_names == {str(i) for i in range(20)}


def test_torque_classification_distinguishes_bias_like_single_joint_failure():
    abs_err = np.array([0.1, 0.2, 0.1, 0.1, 0.1, ABS_ERR_LIMITS[5] + 0.05])
    rel_err = abs_err / (ABS_ERR_LIMITS / 0.10)
    status = classify_torque_result(
        settled=True,
        saturated=False,
        tau_real=np.zeros(6),
        abs_err=abs_err,
        rel_err=rel_err,
        l2_err=float(np.linalg.norm(abs_err)),
    )
    assert status == ValidationStatus.FAIL_BIAS


def test_report_jsonl_and_csv_roundtrip(tmp_path: Path):
    pose = SafePose("0", np.zeros(6), 108.0)
    genesis = GenesisDynamicsSample(
        q_actual=np.zeros(6),
        qvel=np.zeros(6),
        pd_hold_tau=np.ones(6),
        actual_dof_force=np.zeros(6),
        mass_matrix=np.eye(6),
        settled=True,
        saturated=False,
        pos_err=0.0,
        vel_mag=0.0,
    )
    sample = build_dynamics_sample(pose, genesis, tau_real=np.ones(6), n_real_samples=3)
    run_config = DynamicsRunConfig(robot_key="xarm6_1305", urdf_path=xarm6_1305_urdf(), mode="unit")

    jsonl = tmp_path / "report.jsonl"
    csv = tmp_path / "report.csv"
    write_jsonl_report([sample], jsonl, run_config=run_config)
    write_csv_report([sample], csv)

    assert read_report_records(jsonl)[0]["pose"] == "0"
    assert read_report_records(csv)[0]["pose"] == "0"
    csv_text = csv.read_text(encoding="utf-8")
    assert "schema_version,pose" in csv_text
    assert "torque_l2_err_nm" in csv_text
    assert "genesis_tau_J1_nm" in csv_text
    assert "sdk_tau_mean_J1_nm" in csv_text
    assert "l2a_l2_err_nm" in csv_text
    assert "l3a_mass_rel_fro" in csv_text
    assert read_report_records(csv)[0]["schema_version"] == REPORT_SCHEMA_VERSION


def test_default_dynamics_report_paths_use_identity_and_run_stamps(tmp_path: Path):
    csv_path, jsonl_path, plot_path = dynamics_report_paths(
        identity="XI1305/ABC 123",
        report_root=tmp_path,
        stamp="20260702_103201",
    )

    assert sanitize_report_identity("XI1305/ABC 123") == "XI1305_ABC_123"
    assert (
        csv_path == tmp_path / "dyn_ver_XI1305_ABC_123" / "20260702_1032" / "dyn_ver_XI1305_ABC_123_20260702_103201.csv"
    )
    assert jsonl_path == csv_path.with_suffix(".jsonl")
    assert plot_path == csv_path.with_name("dyn_ver_XI1305_ABC_123_20260702_103201_torque.png")


def test_explicit_report_paths_take_precedence(tmp_path: Path):
    csv_path, jsonl_path, plot_path = dynamics_report_paths(
        identity="XI1305",
        report=tmp_path / "manual.csv",
        jsonl_report=tmp_path / "manual-report.jsonl",
        stamp="20260702_103201",
    )

    assert csv_path == tmp_path / "manual.csv"
    assert jsonl_path == tmp_path / "manual-report.jsonl"
    assert plot_path == tmp_path / "manual_torque.png"


def test_v3_report_rows_expose_human_readable_torque_fields(tmp_path: Path):
    runtime = get_robot_runtime_profile("xarm7")
    pose = SafePose("5", np.zeros(7), 500.0)
    genesis = GenesisDynamicsSample(
        q_actual=np.zeros(7),
        qvel=np.zeros(7),
        pd_hold_tau=np.array([0.0, 20.824854, 0.0, 0.0, 0.0, 0.0, 0.0]),
        actual_dof_force=np.zeros(7),
        mass_matrix=np.eye(7),
        settled=True,
        saturated=False,
        pos_err=0.0,
        vel_mag=0.0,
    )
    sample = build_dynamics_sample(
        pose,
        genesis,
        runtime_profile=runtime,
        tau_real=np.array([0.0, 34.159194, 0.0, 0.0, 0.0, 0.0, 0.0]),
        tau_real_std=np.array([0.0, 0.298257, 0.0, 0.0, 0.0, 0.0, 0.0]),
        n_real_samples=31,
    )

    csv = tmp_path / "report.csv"
    write_csv_report([sample], csv, runtime_profile=runtime)
    text = csv.read_text(encoding="utf-8")

    assert "genesis_tau_J2_nm" in text
    assert "sdk_tau_mean_J2_nm" in text
    assert "abs_limit_J2_nm" in text
    assert "20.824854" in text
    assert "34.159194" in text
    summary = torque_summary(sample, runtime_profile=runtime)
    assert summary["worst_joint"] == "J2"
    assert summary["torque_l2_limit_nm"] == runtime.dynamics.l2_err_limit
    rows = joint_torque_rows(sample, runtime_profile=runtime)
    assert rows[1]["genesis_tau_nm"] == 20.824854
    assert rows[1]["sdk_tau_mean_nm"] == 34.159194


def test_read_report_records_supports_legacy_v2_csv(tmp_path: Path):
    csv = tmp_path / "legacy.csv"
    csv.write_text(
        "schema_version,pose,status,l2_err,q1,pd_hold_tau1,tau_real1,abs_err1\n2,home,PASS,1.0,0.0,3.0,1.0,2.0\n",
        encoding="utf-8",
    )

    record = read_report_records(csv)[0]
    assert record["schema_version"] == "2"
    assert record["signed_err"] == [2.0]
    assert record["abs_err"] == [2.0]


def test_failure_detail_prints_theory_sdk_and_thresholds(capsys):
    runtime = get_robot_runtime_profile("xarm7")
    pose = SafePose("5", np.zeros(7), 500.0)
    genesis = GenesisDynamicsSample(
        q_actual=np.zeros(7),
        qvel=np.zeros(7),
        pd_hold_tau=np.array([0.0, 20.824854, 0.0, 0.0, 0.0, 0.0, 0.0]),
        actual_dof_force=np.zeros(7),
        mass_matrix=np.eye(7),
        settled=True,
        saturated=False,
        pos_err=0.0,
        vel_mag=0.0,
    )
    sample = build_dynamics_sample(
        pose,
        genesis,
        runtime_profile=runtime,
        tau_real=np.array([0.0, 40.824854, 0.0, 0.0, 0.0, 0.0, 0.0]),
        tau_real_std=np.array([0.0, 0.298257, 0.0, 0.0, 0.0, 0.0, 0.0]),
        n_real_samples=31,
    )

    print_compare_table([sample], runtime_profile=runtime)
    out = capsys.readouterr().out

    assert "genesis_tau_nm" in out
    assert "sdk_tau_mean_nm" in out
    assert "abs_limit_nm" in out
    assert "20.824854" in out
    assert "40.824854" in out


def test_torque_plot_layouts_and_generation(tmp_path: Path):
    pytest.importorskip("matplotlib")
    runtime6 = get_robot_runtime_profile("xarm6")
    runtime7 = get_robot_runtime_profile("xarm7")
    assert torque_plot_layout(6) == (2, 3)
    assert torque_plot_layout(7) == (3, 3)

    pose6 = SafePose("0", np.zeros(6), 100.0)
    genesis6 = GenesisDynamicsSample(
        q_actual=np.zeros(6),
        qvel=np.zeros(6),
        pd_hold_tau=np.ones(6),
        actual_dof_force=np.zeros(6),
        mass_matrix=np.eye(6),
        settled=True,
        saturated=False,
        pos_err=0.0,
        vel_mag=0.0,
    )
    sample6 = build_dynamics_sample(pose6, genesis6, runtime_profile=runtime6, tau_real=np.ones(6), n_real_samples=3)
    out6 = tmp_path / "xarm6_torque.png"
    assert write_torque_plot([sample6], out6, runtime_profile=runtime6)
    assert out6.exists() and out6.stat().st_size > 0

    pose7 = SafePose("0", np.zeros(7), 100.0)
    genesis7 = GenesisDynamicsSample(
        q_actual=np.zeros(7),
        qvel=np.zeros(7),
        pd_hold_tau=np.ones(7),
        actual_dof_force=np.zeros(7),
        mass_matrix=np.eye(7),
        settled=True,
        saturated=False,
        pos_err=0.0,
        vel_mag=0.0,
    )
    sample7 = build_dynamics_sample(pose7, genesis7, runtime_profile=runtime7, tau_real=np.ones(7), n_real_samples=3)
    out7 = tmp_path / "xarm7_torque.png"
    assert write_torque_plot([sample7], out7, runtime_profile=runtime7)
    assert out7.exists() and out7.stat().st_size > 0

    dry_run_sample = build_dynamics_sample(pose6, genesis6, runtime_profile=runtime6, skip_reason="dry-run")
    assert not write_torque_plot([dry_run_sample], tmp_path / "dry_run.png", runtime_profile=runtime6)


def test_compare_report_records_uses_signed_residuals():
    old = [{"signed_err": [1, 2, 3, 4, 5, 6]}]
    new = [{"signed_err": [2, 2, 3, 4, 5, 9]}]
    stats = compare_report_records(old, new)
    assert stats[0]["bias_delta"] == 1
    assert stats[5]["new_bias"] == 9


def test_report_compare_supports_seven_dof():
    old = [{"signed_err": [1, 2, 3, 4, 5, 6, 7]}]
    new = [{"signed_err": [1, 2, 3, 4, 5, 6, 9]}]
    stats = compare_report_records(old, new)
    assert len(stats) == 7
    assert stats[6]["bias_delta"] == 2


def test_calibrated_urdf_defaults_to_cache(tmp_path: Path):
    kinematics = {
        f"joint{i}": {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0} for i in range(1, 7)
    }
    out = Path(build_calibrated_urdf(xarm6_1305_urdf(), kinematics, suffix="unit", output_dir=str(tmp_path)))
    assert out.parent == tmp_path
    assert out.name.endswith("_calib.urdf")
    assert out.exists()


def test_real_robot_collect_hold_samples_uses_statistics():
    session = object.__new__(RealRobotSession)
    arm = MagicMock()
    arm.get_joint_states.side_effect = [
        (0, [np.zeros(7).tolist(), np.zeros(7).tolist(), np.ones(7).tolist()]),
        (0, [np.zeros(7).tolist(), np.zeros(7).tolist(), (np.ones(7) * 3).tolist()]),
    ]
    arm.get_joints_torque.side_effect = [
        (0, np.ones(7).tolist()),
        (0, (np.ones(7) * 3).tolist()),
    ]
    session.arm = arm
    session.dof = 6
    session.home_qpos = np.zeros(6)

    sample = session.collect_hold_samples(duration_s=0.0)
    assert sample.n_samples == 1
    assert np.allclose(sample.tau, np.ones(6))
    assert np.allclose(sample.tau_direct, np.ones(6))


def test_run_sim_collision_chain_moves_through_poses_in_order(monkeypatch):
    monkeypatch.setattr("ufactory.hardware.xarm.time.sleep", lambda _: None)
    arm = MagicMock()
    arm.error_code = 0
    session = object.__new__(RealRobotSession)
    session.arm = arm
    session.dof = 6
    session.home_qpos = np.zeros(6)

    calls: list[tuple[str, np.ndarray]] = []

    def move_to(q, *, speed_rad_s, wait, move_strategy):
        del speed_rad_s, wait, move_strategy
        calls.append(("move", np.asarray(q, dtype=np.float64)))
        return 0

    session.move_to = move_to

    poses = [
        ("0", np.zeros(6)),
        ("pose_a", np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ("pose_b", np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])),
    ]
    results = run_sim_collision_chain(
        session,
        poses,
        speed_rad_s=DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S,
        move_strategy="direct",
    )

    assert len(results) == 3
    assert all(r.passed for r in results)
    assert [name for name, _ in poses] == [r.pose_name for r in results]
    assert len(calls) == 3
    assert np.allclose(calls[0][1], np.zeros(6))
    assert np.allclose(calls[1][1], poses[1][1])
    assert np.allclose(calls[2][1], poses[2][1])


def test_run_sim_collision_chain_records_failure_and_recovers(monkeypatch):
    from ufactory.hardware.session import RobotMotionError

    monkeypatch.setattr("ufactory.hardware.xarm.time.sleep", lambda _: None)
    arm = MagicMock()
    arm.error_code = 0
    session = object.__new__(RealRobotSession)
    session.arm = arm
    session.dof = 6
    session.home_qpos = np.zeros(6)

    def move_to(q, *, speed_rad_s, wait, move_strategy):
        del speed_rad_s, wait, move_strategy
        if float(np.asarray(q)[0]) == 0.2:
            raise RobotMotionError(
                "collision",
                code=22,
                waypoint_index=1,
                waypoint_count=1,
                target_q=q,
                waypoint_q=q,
            )
        return 0

    session.move_to = move_to

    poses = [
        ("0", np.zeros(6)),
        ("bad", np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ("next", np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0])),
    ]
    results = run_sim_collision_chain(
        session,
        poses,
        speed_rad_s=DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S,
        move_strategy="direct",
    )

    assert results[0].passed
    assert not results[1].passed
    assert results[1].error_code == 22
    assert results[2].passed


def test_check_controller_tcp_offset_z_blocks_nonzero_offset_below_table():
    from ufactory.dynamics.cli import _check_controller_tcp_offset_z

    session = object.__new__(RealRobotSession)
    arm = MagicMock()
    arm.tcp_offset = [0.0, 0.0, 173.5, 0.0, 0.0, 0.0]

    def fk(angles, input_is_radian=True, return_is_radian=True):
        del input_is_radian, return_is_radian
        if np.allclose(angles, np.zeros(6)):
            return 0, [207.0, 0.0, -58.5, 3.14, 0.0, 0.0]
        return 0, [300.0, 0.0, 200.0, 3.14, 0.0, 0.0]

    arm.get_forward_kinematics.side_effect = fk
    session.arm = arm

    poses = [
        SafePose("0", np.array([0.5, -0.5, 0.0, 0.0, 0.0, 0.0]), 200.0),
        SafePose("9", np.zeros(6), 113.5),
    ]
    reasons = _check_controller_tcp_offset_z(session, poses, np.zeros(6))
    assert reasons == ["home: EE z -58.50 mm <= 0.00 mm"]

    arm.tcp_offset = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert _check_controller_tcp_offset_z(session, poses, np.zeros(6)) == []


def test_cli_hardware_check_aborts_before_motion_when_tcp_offset_fails_z_precheck(monkeypatch, capsys):
    from ufactory.dynamics import cli as dyn_cli

    monkeypatch.setattr(
        "ufactory.kinematics.calibration.resolve_kinematics_suffix_from_ip",
        lambda *args, **kwargs: (None, "XI130505L56A01"),
    )
    monkeypatch.setattr(
        "ufactory.kinematics.calibration.prepare_robot_model_for_verification",
        lambda *args, **kwargs: (xarm6_1305_urdf(), None),
    )
    monkeypatch.setattr(dyn_cli, "validate_urdf_dynamics", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        dyn_cli, "build_genesis_scene", lambda *args, **kwargs: ("scene", "robot", "ee", [0, 1, 2, 3, 4, 5])
    )
    monkeypatch.setattr(dyn_cli, "compute_ee_z_table_from_sim", lambda *args, **kwargs: {"0": 200.0, "9": 113.5})
    monkeypatch.setattr(dyn_cli, "load_reference_backend", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        dyn_cli,
        "_run_genesis_samples",
        lambda *args, **kwargs: {
            "0": MagicMock(q_actual=np.zeros(6)),
            "9": MagicMock(q_actual=np.zeros(6)),
        },
    )

    fake_session = MagicMock()
    fake_session.arm.tcp_offset = [0.0, 0.0, 173.5, 0.0, 0.0, 0.0]
    fake_session.arm.get_forward_kinematics.return_value = (0, [207.0, 0.0, -58.5, 3.14, 0.0, 0.0])
    monkeypatch.setattr("ufactory.hardware.session.RealRobotSession", lambda *args, **kwargs: fake_session)

    rc = dyn_cli.cli_hardware_check(["--robot", "xarm6", "--ip", "192.168.1.60", "--confirm-real", "--poses", "0,9"])
    out = capsys.readouterr().out

    assert rc == 1
    assert "Controller has non-zero tcp_offset and failed EE z > 0 precheck" in out
    assert "home: EE z -58.50 mm <= 0.00 mm" in out
    fake_session.configure_for_dynamics.assert_not_called()
    fake_session.sample_at_hold.assert_not_called()
    fake_session.return_home.assert_not_called()
    fake_session.disconnect.assert_called_once()
