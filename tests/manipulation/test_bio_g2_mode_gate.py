"""Mode gates, flange offset, and dry-run preflight for Bio Gripper G2 in pick_place."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from ufactory.config import load_runtime_config
from ufactory.grippers.bio_g2 import (
    BIO_G2_CONTROLLER_VALIDATED,
    BIO_GRIPPER_G2_TCP_IN_FLANGE_M,
    bio_g2_controller_mode_error,
)
from ufactory.manipulation.commands import build_arm_controller
from ufactory.robots.assembly import compose_robot_urdf


def _pick_place_module():
    pytest.importorskip("torch")
    pytest.importorskip("genesis")
    return importlib.import_module("ufactory.cli.pick_place")


def test_bio_g2_controller_validation_matrix() -> None:
    assert BIO_G2_CONTROLLER_VALIDATED == frozenset({("pick_place", "xarm6")})
    for mode in ("sim", "dry-run"):
        assert bio_g2_controller_mode_error("pick_place", "xarm5", "bio_gripper_g2", mode) is None
    assert bio_g2_controller_mode_error("pick_place", "xarm6", "bio_gripper_g2", "sdk-sim") is None
    assert bio_g2_controller_mode_error("pick_place", "xarm6", "bio_gripper_g2", "real") is None
    assert bio_g2_controller_mode_error("pick_place", "xarm6", "g2", "real") is None
    for robot in ("xarm5", "xarm7", "uf850"):
        for mode in ("sdk-sim", "real"):
            err = bio_g2_controller_mode_error("pick_place", robot, "bio_gripper_g2", mode)
            assert err is not None and "pick_place on xarm6" in err


@pytest.mark.parametrize(
    "argv",
    [
        ["--robot", "xarm6", "--gripper", "bio_gripper_g2", "--mode", "dry-run"],
        [
            "--robot",
            "xarm6",
            "--gripper",
            "bio_gripper_g2",
            "--mode",
            "sdk-sim",
            "--executor",
            "servo_j",
            "--visual",
        ],
        [
            "--robot",
            "xarm5",
            "--gripper",
            "bio_gripper_g2",
            "--mode",
            "sdk-sim",
            "--executor",
            "servo_cartesian",
        ],
        [
            "--robot",
            "xarm7",
            "--gripper",
            "bio_gripper_g2",
            "--mode",
            "real",
            "--executor",
            "servo_cartesian",
        ],
    ],
)
def test_bio_g2_non_sim_gates_reject_unsupported_combinations(argv: list[str]) -> None:
    module = _pick_place_module()
    with pytest.raises(SystemExit) as exc:
        module.main(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize("executor", ["servo_j", "servo_cartesian"])
def test_bio_g2_real_visual_passes_gate_and_reaches_ip_check(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    executor: str,
) -> None:
    module = _pick_place_module()
    monkeypatch.delenv("XARM_IP", raising=False)
    with pytest.raises(SystemExit) as exc:
        module.main(
            [
                "--robot",
                "xarm6",
                "--gripper",
                "bio_gripper_g2",
                "--mode",
                "real",
                "--executor",
                executor,
                "--visual",
            ]
        )
    assert exc.value.code == 2
    err = capsys.readouterr().err
    error_line = err.splitlines()[-1]
    assert "--ip or XARM_IP" in error_line
    assert "servo_cartesian" not in error_line
    assert "requires --executor servo_cartesian" not in err


@pytest.mark.gpu
def test_bio_g2_servo_j_compile_and_dry_run_pass(tmp_path: Path) -> None:
    import numpy as np

    from ufactory.simulation import GenesisRuntimeManager

    module = _pick_place_module()
    cfg = load_runtime_config("xarm6", gripper="bio_gripper_g2")
    with GenesisRuntimeManager(cfg.simulation):
        program, ctx, q_home = module._compile_servo_j_program(cfg, calibration=None)
        home_pregrasp = next(seg for seg in program.segments if seg.label == "home->pregrasp")
        assert home_pregrasp.q_start is not None
        assert float(np.max(np.abs(np.asarray(home_pregrasp.q_start, dtype=np.float64) - q_home))) < 2e-3
        assert ctx.gripper_dof_multipliers == (1.0, -1.0)

    report = tmp_path / "bio_dry_run_servo_j.json"
    rc = module.main(
        [
            "--robot",
            "xarm6",
            "--gripper",
            "bio_gripper_g2",
            "--mode",
            "dry-run",
            "--executor",
            "servo_j",
            "--report",
            str(report),
        ]
    )
    assert rc == 0
    assert report.is_file()


def test_bio_g2_build_program_applies_flange_xy_offset_and_closed_gap() -> None:
    module = _pick_place_module()
    cfg = load_runtime_config("xarm6", gripper="bio_gripper_g2")
    urdf = Path(compose_robot_urdf("xarm6_1305", "bio_gripper_g2"))
    kinematics = build_arm_controller(cfg, urdf)
    program = module._build_program(cfg, kinematics)

    expected_obj_x = 0.38 - BIO_GRIPPER_G2_TCP_IN_FLANGE_M[0]
    descend = next(seg for seg in program.segments if seg.label == "descend")
    samples, _ = descend.samples(program.rate)
    assert samples[-1, 0] == pytest.approx(expected_obj_x, abs=1e-6)
    assert samples[-1, 1] == pytest.approx(-0.12, abs=1e-6)
    assert samples[-1, 2] == pytest.approx(0.1595, abs=1e-6)

    grip = next(seg for seg in program.segments if seg.label == "grip")
    grip_samples, _ = grip.samples(program.rate)
    assert grip.gap_start == pytest.approx(0.150, abs=1e-6)
    assert 0.149 < float(grip_samples[0, 0]) <= 0.150
    assert grip_samples[-1, 0] == pytest.approx(0.071, abs=1e-6)


def test_bio_g2_xarm6_dry_run_servo_cartesian_passes(tmp_path: Path) -> None:
    module = _pick_place_module()
    report = tmp_path / "bio_dry_run.json"
    rc = module.main(
        [
            "--robot",
            "xarm6",
            "--gripper",
            "bio_gripper_g2",
            "--mode",
            "dry-run",
            "--executor",
            "servo_cartesian",
            "--report",
            str(report),
        ]
    )
    assert rc == 0
    assert report.is_file()
