from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ufactory.cli.packaging import _backends, _model_and_hashes, _sdk_evidence_path
from ufactory.config import ConfigError, load_runtime_config
from ufactory.robots.runtime import get_robot_runtime_profile
from ufactory.manipulation.packaging import (
    build_packaging_program,
    packaging_layout,
    packaging_obstacles,
    packaging_scene_sha256,
)
from ufactory.trajectory.preflight import create_safety_gate


ROBOTS = ("xarm5", "xarm6", "xarm7", "uf850", "lite6")
G2_ROBOTS = ("xarm5", "xarm6", "xarm7", "uf850")


@pytest.mark.parametrize("robot", ROBOTS)
def test_packaging_profiles_are_available_for_all_supported_robots(robot: str) -> None:
    config = load_runtime_config(robot, task="packaging_showcase")
    layout = packaging_layout(config)

    assert get_robot_runtime_profile(robot).task.showcase_supported is True
    assert config.gripper is not None
    assert layout.object_size_m == pytest.approx((0.030, 0.030, 0.030))
    assert layout.object_mass_kg == pytest.approx(0.017)
    assert config.simulation.substeps == 32
    assert "simulation_substeps" not in config.task.parameters


@pytest.mark.parametrize("robot", G2_ROBOTS)
def test_g2_packaging_profiles_preserve_reference_geometry(robot: str) -> None:
    layout = packaging_layout(load_runtime_config(robot, task="packaging_showcase"))

    assert layout.object_position_m == pytest.approx((0.300, 0.000, 0.015))
    assert layout.target_position_m == pytest.approx((0.300, 0.300, 0.018))
    assert layout.grasp_link6_z_m == pytest.approx(0.1886)
    assert layout.release_link6_z_m == pytest.approx(0.3916)


def test_g2_packaging_scene_rejects_unsafe_global_substeps_before_build() -> None:
    pytest.importorskip("genesis")
    from ufactory.manipulation.packaging.scene import build_packaging_scene

    config = load_runtime_config(
        "xarm6",
        task="packaging_showcase",
        overrides={"simulation.substeps": 31},
    )
    with pytest.raises(ValueError, match=r"0\.625 ms.*substeps=31.*at least 32"):
        build_packaging_scene(show_viewer=False, runtime_config=config)


def test_lite6_packaging_profile_uses_lite6_gripper_geometry_and_safe_box_center() -> None:
    config = load_runtime_config("lite6", task="packaging_showcase")
    layout = packaging_layout(config)

    assert "assets/configs/runtime/tasks/robots/lite6_packaging_showcase.yaml" in config.sources
    assert layout.object_position_m == pytest.approx((0.200, 0.000, 0.015))
    assert layout.target_position_m == pytest.approx((0.200, 0.280, 0.018))
    assert layout.box_center_xy_m == pytest.approx((0.200, 0.280))
    assert layout.home_position_m == pytest.approx((0.200, 0.000, 0.200))
    assert layout.grasp_gap_m == pytest.approx(0.020)
    assert layout.grasp_link6_z_m == pytest.approx(0.0963)
    assert layout.release_link6_z_m == pytest.approx(0.2993)
    assert layout.simulation_pre_release_relax_duration_s == pytest.approx(0.0)
    assert layout.simulation_release_duration_s == pytest.approx(0.500)


@pytest.mark.parametrize(
    ("robot", "expected"),
    [("xarm6", True), ("lite6", True), ("xarm5", False), ("xarm7", False), ("uf850", False)],
)
def test_real_packaging_capability_matches_installed_end_effector(robot: str, expected: bool) -> None:
    config = load_runtime_config(robot, task="packaging_showcase")

    assert config.gripper is not None
    assert config.gripper.real_command is expected


@pytest.mark.parametrize("robot", ROBOTS)
def test_packaging_program_transits_to_configured_target(robot: str) -> None:
    config = load_runtime_config(robot, task="packaging_showcase")
    layout = packaging_layout(config)
    program = build_packaging_program(config)
    transit = next(segment for segment in program.segments if segment.label == "transit")

    assert transit.pose_end is not None
    assert transit.pose_end == pytest.approx(
        (layout.target_position_m[0], layout.target_position_m[1], layout.release_link6_z_m)
    )


def test_box_floor_top_is_consumed_as_physical_floor_geometry() -> None:
    config = load_runtime_config(
        "xarm6",
        task="packaging_showcase",
        overrides={
            "task.parameters.box_floor_top_z_m": 0.006,
            "task.parameters.fixed_target_position_m": [0.300, 0.300, 0.021],
        },
    )
    floor = next(obstacle for obstacle in packaging_obstacles(packaging_layout(config)) if obstacle.name == "box_floor")

    assert floor.size_m[2] == pytest.approx(0.006)
    assert floor.center_m[2] == pytest.approx(0.003)


def test_box_floor_thickness_is_relative_to_nonzero_table_height() -> None:
    config = load_runtime_config(
        "xarm6",
        task="packaging_showcase",
        overrides={
            "task.parameters.table_center_m": [0.337, 0.000, 0.180],
            "task.parameters.fixed_object_position_m": [0.300, 0.000, 0.215],
            "task.parameters.box_floor_top_z_m": 0.203,
            "task.parameters.fixed_target_position_m": [0.300, 0.300, 0.218],
        },
    )
    layout = packaging_layout(config)
    floor = next(obstacle for obstacle in packaging_obstacles(layout) if obstacle.name == "box_floor")

    assert layout.table_top_z_m == pytest.approx(0.200)
    assert floor.size_m[2] == pytest.approx(0.003)
    assert floor.center_m[2] == pytest.approx(0.2015)


def test_packaging_scene_hash_covers_effective_gripper_configuration() -> None:
    baseline = load_runtime_config("xarm6", task="packaging_showcase")
    assert baseline.gripper is not None
    tuned = load_runtime_config(
        "xarm6",
        task="packaging_showcase",
        overrides={"gripper.kp": baseline.gripper.kp + 1.0},
    )

    assert packaging_scene_sha256(baseline) != packaging_scene_sha256(tuned)


@pytest.mark.slow
@pytest.mark.parametrize("robot", ROBOTS)
def test_servo_cartesian_packaging_preflight_passes_for_all_robots(robot: str) -> None:
    config = load_runtime_config(robot, task="packaging_showcase")
    urdf, _urdf_hash, calibration_hash = _model_and_hashes(config, None, None)
    kinematics, collision = _backends(config, urdf)
    program = build_packaging_program(config, kinematics=kinematics)
    gate = create_safety_gate(
        config,
        kinematics=kinematics,
        collision=collision,
        calibration_sha256=calibration_hash,
        scene_sha256=packaging_scene_sha256(config),
        urdf_path=urdf,
    )

    report = gate.preflight(program, executor="servo_cartesian")

    assert report.passed, [violation.message for violation in report.violations[:10]]


@pytest.mark.parametrize(
    ("robot", "body", "message"),
    [
        ("xarm6", "grasp_gap_m: 0.100", "gripper gap range"),
        ("xarm6", "simulation_pre_release_relax_gap_m: 0.100", "gripper gap range"),
        ("lite6", "grasp_gap_m: 0.019", "gripper gap range"),
        ("xarm6", "box_outer_size_m: [0.040, 0.040, 0.150]", "safety margin"),
        ("xarm6", "fixed_target_position_m: [0.300, 0.300, 0.019]", "fixed target Z"),
        ("xarm6", "box_center_xy_m: [0.000, 0.000]", "outside the box opening"),
        ("xarm6", "table_size_m: [0.100, 0.100, 0.040]", "outside the table"),
    ],
)
def test_invalid_packaging_yaml_is_rejected(
    tmp_path: Path,
    robot: str,
    body: str,
    message: str,
) -> None:
    path = tmp_path / "invalid_packaging.yaml"
    path.write_text(f"schema_version: 1\ntask:\n  parameters:\n    {body}\n", encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_runtime_config(robot, task="packaging_showcase", config_path=path)


@pytest.mark.parametrize("robot", ("xarm5", "xarm7", "uf850"))
def test_real_packaging_rejects_profiles_without_enabled_real_gripper(
    monkeypatch: pytest.MonkeyPatch,
    robot: str,
) -> None:
    from ufactory.cli import packaging

    connected = False

    def fail_if_connected(_ip: str) -> None:
        nonlocal connected
        connected = True
        raise AssertionError("controller connection must not be attempted")

    monkeypatch.setattr(packaging, "_connect", fail_if_connected)

    with pytest.raises(SystemExit):
        packaging.main(["--robot", robot, "--mode", "real", "--confirm-real"])
    assert connected is False


def test_packaging_sdk_report_name_is_robot_specific() -> None:
    assert _sdk_evidence_path(None, "xarm6") == Path("reports/sdk_sim_xarm6_packaging.json")
    assert _sdk_evidence_path(None, "lite6") == Path("reports/sdk_sim_lite6_packaging.json")
    explicit = Path("reports/custom.json")
    assert _sdk_evidence_path(explicit, "lite6") is explicit


@pytest.mark.parametrize(
    "robot,extra", [("lite6", ["--cycles", "3"]), ("xarm6", ["--backend", "cpu"]), ("xarm6", ["--visual"])]
)
def test_packaging_simulation_receives_resolved_options_once(monkeypatch, robot, extra) -> None:
    pytest.importorskip("genesis")
    from ufactory.cli import packaging
    from ufactory.manipulation.packaging import simulation

    received = []
    load_calls = []
    original_load = packaging.load_runtime_config

    def load(*args, **kwargs):
        load_calls.append((args, kwargs))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(packaging, "load_runtime_config", load)
    monkeypatch.setattr(
        simulation, "load_runtime_config", lambda *args, **kwargs: pytest.fail("configuration reloaded")
    )
    monkeypatch.setattr(
        simulation, "run_packaging_simulation", lambda options, config: received.append((options, config)) or 0
    )
    assert packaging.main(["--mode", "sim", "--robot", robot, *extra]) == 0
    assert len(load_calls) == len(received) == 1
    options, config = received[0]
    assert config.robot.key == load_runtime_config(robot, task="packaging_showcase").robot.key
    assert options.cycles == (3 if "--cycles" in extra else None)
    assert options.visual == ("--visual" in extra)
    if "--backend" in extra:
        assert config.simulation.backend == "cpu"


@pytest.mark.parametrize("speed", [0.25, 1.0, 1.5])
def test_packaging_speed_scaling_preserves_resolved_overrides_and_hash(monkeypatch, speed) -> None:
    pytest.importorskip("genesis")
    from ufactory.config import runtime_config_dict
    from ufactory.manipulation.packaging import simulation

    overrides = {
        "simulation.backend": "cpu",
        "simulation.g2.speed_percent": 90,
        "simulation.g2.force_percent": 80,
        "task.parameters.object_mass_kg": 0.025,
        "motion.cartesian_speed_m_s": 0.12,
        "safety.min_collision_distance_m": 0.004,
    }
    base = load_runtime_config("xarm6", task="packaging_showcase", overrides=overrides)
    expected = load_runtime_config(
        "xarm6",
        task="packaging_showcase",
        overrides={
            **overrides,
            "motion.joint_speed_rad_s": base.motion.joint_speed_rad_s * speed,
            "motion.joint_acceleration_rad_s2": base.motion.joint_acceleration_rad_s2 * speed,
            "motion.cartesian_speed_m_s": base.motion.cartesian_speed_m_s * speed,
            "motion.cartesian_acceleration_m_s2": base.motion.cartesian_acceleration_m_s2 * speed,
            "motion.gripper_duration_s": base.motion.gripper_duration_s / speed,
            "task.parameters.grasp_settle_s": base.task.parameters["grasp_settle_s"] / speed,
            "task.parameters.pre_release_settle_s": base.task.parameters["pre_release_settle_s"] / speed,
            "task.parameters.post_release_settle_s": base.task.parameters["post_release_settle_s"] / speed,
        },
    )
    monkeypatch.setattr(
        simulation, "load_runtime_config", lambda *args, **kwargs: pytest.fail("configuration reloaded")
    )
    scaled = simulation._speed_config("xarm6", speed, base_config=base)
    assert runtime_config_dict(scaled) == runtime_config_dict(expected)
    assert base.task.parameters["object_mass_kg"] == 0.025
    assert base.motion.cartesian_speed_m_s == 0.12


def _run_packaging_physics_cycles(robot: str, executor: str, cycles: int) -> None:
    from ufactory.manipulation.packaging.scene import build_packaging_scene
    from ufactory.manipulation.packaging.simulation import (
        _cycle_failure_reason,
        init_showcase_robot,
        prepare_packaging_cycle,
        run_pick_place_cycle,
    )
    from ufactory.simulation import GenesisRuntimeManager

    config = load_runtime_config(robot, task="packaging_showcase")
    task_layout = packaging_layout(config)
    with GenesisRuntimeManager(config.simulation):
        scene, robot_entity, block, display_layout = build_packaging_scene(
            show_viewer=False,
            runtime_config=config,
        )
        context = init_showcase_robot(
            robot_entity,
            display_layout,
            scene,
            runtime_config=config,
        )
        for cycle in range(1, cycles + 1):
            prepare_packaging_cycle(scene, robot_entity, block, display_layout, context)
            report = run_pick_place_cycle(
                scene,
                robot_entity,
                block,
                display_layout,
                ctx=context,
                robot_key=config.robot.key,
                executor=executor,
            )

            assert report is not None
            assert _cycle_failure_reason(report, display_layout, task_layout) is None, (
                f"{robot}/{executor} cycle {cycle} failed"
            )
            phases = {phase.label: phase for phase in report.phases}
            assert phases["lift"].obj_pos_mm[2] > (display_layout.table_top_z + 0.100) * 1000.0
            # High release and rebound can move the cube away from the target
            # center. Check its entire final footprint and height inside the box.
            bounds = block.get_AABB().detach().cpu().numpy().reshape(2, 3)
            half_inner = np.asarray(display_layout.box_outer[:2]) / 2.0 - display_layout.box_wall
            center = np.asarray(display_layout.box_center_xy)
            inner_bounds = np.array(
                [
                    [*(center - half_inner), display_layout.box_inner_floor_z],
                    [*(center + half_inner), display_layout.box_inner_floor_z + display_layout.box_outer[2]],
                ]
            )
            contact_tolerance_m = 0.001
            assert np.isfinite(bounds).all()
            assert np.all(bounds[0] >= inner_bounds[0] - contact_tolerance_m) and np.all(
                bounds[1] <= inner_bounds[1] + contact_tolerance_m
            ), (
                f"{robot}/{executor} cycle {cycle}: object bounds {bounds.tolist()} "
                f"outside box interior {inner_bounds.tolist()}"
            )
            assert report.home_drift_mm < 10.0


@pytest.mark.gpu
def test_packaging_physics_representative_two_cycles() -> None:
    """Default project-check sim check: one robot, one executor, two cycles."""
    _run_packaging_physics_cycles("xarm6", "servo_j", cycles=2)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("robot", ROBOTS)
def test_multi_robot_packaging_physics_one_cycle(robot: str) -> None:
    """Deep GPU matrix: five robots, servo_j, one cycle (serial; one GPU)."""
    _run_packaging_physics_cycles(robot, "servo_j", cycles=1)


@pytest.mark.gpu
@pytest.mark.slow
def test_packaging_physics_servo_cartesian_xarm6_one_cycle() -> None:
    """Deep keeps one cartesian executor sample; full 5×2 matrix is not required."""
    _run_packaging_physics_cycles("xarm6", "servo_cartesian", cycles=1)
