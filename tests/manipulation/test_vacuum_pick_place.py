"""Config overlay, mode gates, and offline dry-run preflight for vacuum_gripper in pick_place."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from ufactory.config import ConfigError, load_runtime_config
from ufactory.manipulation.commands import build_arm_controller
from ufactory.robots.assembly import compose_robot_urdf


def _pick_place_module():
    return importlib.import_module("ufactory.cli.pick_place")


@pytest.mark.parametrize("robot", ["xarm5", "xarm6", "xarm7", "uf850"])
def test_vacuum_gripper_runtime_config_overlay(robot: str) -> None:
    cfg = load_runtime_config(robot, gripper="vacuum_gripper")
    assert cfg.robot.urdf == "assembly:vacuum_gripper"
    assert cfg.gripper is not None
    assert cfg.gripper.adapter == "vacuum_gripper"
    assert cfg.gripper.has_actuated_joints is False
    assert cfg.gripper.real_command is True
    assert cfg.gripper.tcp_offset == pytest.approx((0.0, 0.0, 126.0, 0.0, 0.0, 0.0))
    assert cfg.gripper.open_gap_m == pytest.approx(1.0)
    assert cfg.gripper.closed_gap_m == pytest.approx(0.0)


def test_vacuum_gripper_overlay_rejects_lite6_and_reach() -> None:
    with pytest.raises(ConfigError):
        load_runtime_config("lite6", gripper="vacuum_gripper")
    with pytest.raises(ConfigError):
        load_runtime_config("xarm5", task="reach", gripper="vacuum_gripper")


@pytest.mark.parametrize(
    "argv",
    [
        ["--robot", "xarm5", "--gripper", "vacuum_gripper", "--mode", "sim", "--executor", "servo_j"],
        ["--robot", "lite6", "--gripper", "vacuum_gripper", "--mode", "dry-run", "--executor", "servo_j"],
        ["--robot", "xarm5", "--gripper", "vacuum_gripper", "--mode", "dry-run"],
    ],
)
def test_vacuum_pick_place_cli_rejects_invalid_combinations(argv: list[str]) -> None:
    module = _pick_place_module()
    with pytest.raises(SystemExit) as exc:
        module.main(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize("executor", ["servo_j", "servo_cartesian"])
def test_vacuum_real_visual_passes_gate_to_ip_check(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    executor: str,
) -> None:
    module = _pick_place_module()
    monkeypatch.delenv("XARM_IP", raising=False)
    monkeypatch.setattr(module, "require_genesis_capabilities", lambda **_kw: None)
    with pytest.raises(SystemExit) as exc:
        module.main(
            [
                "--robot",
                "xarm5",
                "--gripper",
                "vacuum_gripper",
                "--mode",
                "real",
                "--executor",
                executor,
                "--visual",
            ]
        )
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--ip or XARM_IP" in err.splitlines()[-1]


def test_vacuum_xarm5_program_applies_126mm_tcp_offset() -> None:
    module = _pick_place_module()
    cfg = load_runtime_config("xarm5", gripper="vacuum_gripper")
    urdf = Path(compose_robot_urdf("xarm5_1305", "vacuum_gripper"))
    kinematics = build_arm_controller(cfg, urdf)
    program = module._build_program(cfg, kinematics)

    descend = next(seg for seg in program.segments if seg.label == "descend")
    samples, _ = descend.samples(program.rate)
    # grasp_table_clearance_m (0.050) + vacuum TCP Z (0.126) = 0.176 m flange Z
    assert samples[-1, 0] == pytest.approx(0.30, abs=1e-6)
    assert samples[-1, 1] == pytest.approx(0.00, abs=1e-6)
    assert samples[-1, 2] == pytest.approx(0.176, abs=1e-6)

    grip = next(seg for seg in program.segments if seg.label == "grip")
    grip_samples, _ = grip.samples(program.rate)
    assert grip.gap_start == pytest.approx(1.0, abs=1e-6)
    assert grip_samples[-1, 0] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("executor", ["servo_j", "servo_cartesian"])
def test_vacuum_xarm5_dry_run_passes(tmp_path: Path, executor: str) -> None:
    pytest.importorskip("pinocchio")
    pytest.importorskip("coal")
    module = _pick_place_module()
    report = tmp_path / f"xarm5_vacuum_{executor}.json"
    rc = module.main(
        [
            "--robot",
            "xarm5",
            "--gripper",
            "vacuum_gripper",
            "--mode",
            "dry-run",
            "--executor",
            executor,
            "--report",
            str(report),
        ]
    )
    assert rc == 0
    assert report.is_file()
