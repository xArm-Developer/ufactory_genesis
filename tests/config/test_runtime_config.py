from __future__ import annotations

from pathlib import Path

import pytest

from ufactory.config import (
    ConfigError,
    GraspObjectSpec,
    RepositoryAssetStore,
    dump_runtime_config,
    load_runtime_config,
    resolve_pick_place_object_spec,
)
from ufactory.config.models import G2_NOSLIP_ITERATIONS


ROBOTS = ("xarm5", "xarm6", "xarm7", "uf850", "lite6")


@pytest.mark.parametrize(
    "value",
    ["[ab]", "[[a]]", "[[a, a]]", "[[a, b], [b, a]]", "[[a, 1]]", "[[a, '']]"],
)
def test_invalid_geometry_exclusions_are_rejected(tmp_path: Path, value: str) -> None:
    path = tmp_path / "bad_pairs.yaml"
    path.write_text(f"schema_version: 1\ngripper:\n  excluded_geometry_pairs: {value}\n")
    with pytest.raises(ValueError, match="excluded_geometry_pairs"):
        load_runtime_config("lite6", config_path=path)


def test_collision_pair_settings_left_preflight(tmp_path: Path) -> None:
    """Preflight skips every pair below the flange; the old per-robot pair lists are no longer configuration."""
    lite6 = load_runtime_config("lite6")
    assert lite6.gripper is not None
    assert lite6.gripper.excluded_geometry_pairs == (("uflite_finger1_0", "uflite_finger2_0"),)
    assert load_runtime_config("xarm6", gripper="robotiq_2f_85").gripper.excluded_geometry_pairs == ()
    for robot, text in (
        ("xarm6", "robot:\n  adjacent_collision_pairs: [[left_finger, left_inner_knuckle]]\n"),
        ("lite6", "safety:\n  excluded_geometry_pairs: [[uflite_finger1_0, uflite_finger2_0]]\n"),
    ):
        path = tmp_path / f"{robot}_old_pairs.yaml"
        path.write_text(f"schema_version: 1\n{text}")
        with pytest.raises(ValueError, match="unknown configuration field"):
            load_runtime_config(robot, config_path=path)


def test_all_default_robot_configs_are_stable_and_assets_exist():
    store = RepositoryAssetStore.discover()
    first = {name: load_runtime_config(name).sha256 for name in ROBOTS}
    second = {name: load_runtime_config(name).sha256 for name in ROBOTS}
    assert first == second
    assert len(set(first.values())) == len(ROBOTS)
    store.validate_manifest()


def test_unknown_config_field_is_rejected(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: 1\nmotion:\n  mystery: 3\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown configuration field"):
        load_runtime_config("xarm6", config_path=path)


@pytest.mark.parametrize("value", [".nan", ".inf", "-.inf"])
def test_non_finite_config_is_rejected(tmp_path: Path, value: str):
    path = tmp_path / "bad.yaml"
    path.write_text(
        f"schema_version: 1\nmotion:\n  joint_speed_rad_s: {value}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="finite"):
        load_runtime_config("xarm6", config_path=path)


def test_cli_scalar_override_changes_hash_and_xarm_ip_does_not(monkeypatch: pytest.MonkeyPatch):
    base = load_runtime_config("xarm6")
    changed = load_runtime_config("xarm6", overrides={"motion.rate_hz": 100.0})
    monkeypatch.setenv("XARM_IP", "192.0.2.10")
    with_ip = load_runtime_config("xarm6")
    assert changed.sha256 != base.sha256
    assert with_ip.sha256 == base.sha256


def test_dump_config_contains_sources_and_hash_without_importing_genesis():
    text = dump_runtime_config(load_runtime_config("xarm6"))
    assert "schema_version: 1" in text
    assert "sources:" in text
    assert "sha256:" in text


def test_default_calibrated_g2_contact_profile_is_explicit() -> None:
    simulation = load_runtime_config("xarm6").simulation
    assert simulation.substeps == 32
    assert simulation.constraint_solver == "newton"
    assert simulation.friction_cone == "pyramidal"
    assert simulation.contact_resolution == "convex"
    assert simulation.solver_iterations == 100
    assert simulation.noslip_iterations == G2_NOSLIP_ITERATIONS == 0
    assert simulation.constraint_time_constant_s == pytest.approx(0.00125)
    assert simulation.use_gjk_collision is True


@pytest.mark.parametrize("robot", ["xarm5", "xarm6", "xarm7", "uf850"])
def test_g2_robots_use_unified_noslip_default(robot: str) -> None:
    assert load_runtime_config(robot).simulation.noslip_iterations == G2_NOSLIP_ITERATIONS == 0
    assert (
        load_runtime_config(robot, task="packaging_showcase").simulation.noslip_iterations == G2_NOSLIP_ITERATIONS == 0
    )
    assert load_runtime_config("lite6").simulation.noslip_iterations == 0


def test_g2_noslip_default_can_be_overridden_by_user_config(tmp_path: Path) -> None:
    path = tmp_path / "noslip.yaml"
    path.write_text("schema_version: 1\nsimulation:\n  noslip_iterations: 5\n", encoding="utf-8")
    assert load_runtime_config("xarm6", config_path=path).simulation.noslip_iterations == 5


@pytest.mark.parametrize("value", [0, -1, 31.5, 32.0, True, "many", "32"])
def test_simulation_substeps_must_be_a_positive_integer(tmp_path: Path, value: object) -> None:
    path = tmp_path / "bad_substeps.yaml"
    yaml_value = f"'{value}'" if isinstance(value, str) else str(value).lower()
    path.write_text(
        f"schema_version: 1\nsimulation:\n  substeps: {yaml_value}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=r"simulation\.substeps must be a positive integer"):
        load_runtime_config("xarm6", config_path=path)


@pytest.mark.parametrize(
    ("task", "legacy_field"),
    [("pick_place", "substeps"), ("packaging_showcase", "simulation_substeps")],
)
def test_legacy_task_substep_fields_are_rejected(tmp_path: Path, task: str, legacy_field: str) -> None:
    path = tmp_path / "legacy_substeps.yaml"
    path.write_text(
        f"schema_version: 1\ntask:\n  parameters:\n    {legacy_field}: 32\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=rf"unknown configuration field: task\.parameters\.{legacy_field}"):
        load_runtime_config("xarm6", task=task, config_path=path)


@pytest.mark.parametrize(
    "body,match",
    [
        ("friction_cone: future", "friction_cone"),
        ("contact_resolution: future", "contact_resolution"),
        ("constraint_solver: cg", "constraint_solver"),
        ("friction_cone: elliptic\n  noslip_iterations: 5", "elliptic friction"),
        ("contact_resolution: signorini", "signorini contact"),
    ],
)
def test_invalid_rigid_contact_configuration_is_rejected(tmp_path: Path, body: str, match: str) -> None:
    path = tmp_path / "bad_physics.yaml"
    path.write_text(f"schema_version: 1\nsimulation:\n  {body}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_runtime_config("xarm6", config_path=path)


@pytest.mark.parametrize("robot", ROBOTS)
def test_all_robots_share_the_30mm_17g_table_object(robot: str):
    config = load_runtime_config(robot)
    spec = resolve_pick_place_object_spec(config)
    assert spec.size_m == pytest.approx((0.030, 0.030, 0.030))
    assert spec.mass_kg == pytest.approx(0.017)
    assert spec.rest_center_z_m == pytest.approx(0.015)
    params = config.task.parameters
    for name in (
        "fixed_object_position_m",
        "fixed_target_position_m",
    ):
        assert float(params[name][2]) == pytest.approx(0.015)


def test_grasp_object_spec_defensively_freezes_values():
    mutable_size = [0.030, 0.030, 0.030]
    spec = GraspObjectSpec(mutable_size, 0.017)  # type: ignore[arg-type]
    mutable_size[2] = 1.0

    assert spec.size_m == pytest.approx((0.030, 0.030, 0.030))
    assert spec.rest_center_z_m == pytest.approx(0.015)


@pytest.mark.parametrize(
    "body,match",
    [
        ("object_size_m: [0.03, 0.03]", "expected 3 values"),
        ("object_size_m: [0.03, -0.03, 0.03]", "must be positive"),
        ("object_mass_kg: 0", "must be finite and positive"),
        ("fixed_object_position_m: [0.3, 0.0]", "expected 3 values"),
        ("default_ee_position_m: [.nan, 0.0, 0.3]", "must be finite"),
    ],
)
def test_invalid_grasp_object_configuration_is_rejected(tmp_path: Path, body: str, match: str):
    path = tmp_path / "bad_object.yaml"
    path.write_text(f"schema_version: 1\ntask:\n  parameters:\n    {body}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=match):
        load_runtime_config("xarm6", config_path=path)


def test_object_mass_override_changes_config_hash():
    base = load_runtime_config("xarm6")
    changed = load_runtime_config("xarm6", overrides={"task.parameters.object_mass_kg": 0.018})
    assert resolve_pick_place_object_spec(changed).mass_kg == pytest.approx(0.018)
    assert changed.sha256 != base.sha256


@pytest.mark.parametrize(
    ("variant", "open_m", "default_z"),
    [("robotiq_2f_85", 0.085, 0.30), ("robotiq_2f_140", 0.140, 0.36)],
)
def test_robotiq_gripper_overlay_resolution_and_validation(variant: str, open_m: float, default_z: float):
    cfg = load_runtime_config("xarm6", gripper=variant)
    assert cfg.gripper is not None
    assert cfg.gripper.adapter == variant
    assert cfg.gripper.open_gap_m == pytest.approx(open_m)
    assert cfg.robot.urdf == f"assembly:{variant}"
    assert float(cfg.task.parameters["default_ee_position_m"][2]) == pytest.approx(default_z)

    pack_cfg = load_runtime_config("uf850", task="packaging_showcase", gripper=variant)
    assert pack_cfg.gripper is not None and pack_cfg.gripper.adapter == variant

    with pytest.raises(ConfigError, match="not supported"):
        load_runtime_config("lite6", gripper=variant)

    with pytest.raises(ConfigError, match="reach"):
        load_runtime_config("xarm6", task="reach", gripper=variant)


def test_robotiq_2f_85_rejects_position_mm_above_85():
    with pytest.raises(ConfigError, match="85"):
        load_runtime_config("xarm6", gripper="robotiq_2f_85", overrides={"simulation.robotiq.position_mm": 100.0})


@pytest.mark.parametrize("robot", ["xarm5", "xarm6", "xarm7", "uf850"])
@pytest.mark.parametrize("alias", ["bio_gripper_g2", "bio_g2"])
def test_bio_g2_overlay_resolution_for_supported_arms(robot: str, alias: str):
    cfg = load_runtime_config(robot, gripper=alias)
    assert cfg.gripper is not None
    assert cfg.gripper.adapter == "bio_g2"
    assert cfg.gripper.closed_gap_m == pytest.approx(0.071)
    assert cfg.gripper.open_gap_m == pytest.approx(0.150)
    assert cfg.robot.urdf == "assembly:bio_gripper_g2"
    assert list(cfg.task.parameters["fixed_object_position_m"]) == pytest.approx([0.38, -0.12, 0.015])
    assert list(cfg.task.parameters["fixed_target_position_m"]) == pytest.approx([0.38, 0.12, 0.015])
    assert list(cfg.task.parameters["default_ee_position_m"]) == pytest.approx([0.38, -0.12, 0.30])


def test_bio_g2_overlay_rejects_lite6_and_reach():
    with pytest.raises(ConfigError, match="not supported"):
        load_runtime_config("lite6", gripper="bio_gripper_g2")
    with pytest.raises(ConfigError, match="reach"):
        load_runtime_config("xarm6", task="reach", gripper="bio_gripper_g2")


def test_default_g2_and_robotiq_hashes_are_unchanged():
    g2_cfg = load_runtime_config("xarm6")
    robotiq_cfg = load_runtime_config("xarm6", gripper="robotiq_2f_85")
    bio_cfg = load_runtime_config("xarm6", gripper="bio_gripper_g2")
    assert g2_cfg.gripper is not None and g2_cfg.gripper.tcp_offset == pytest.approx((0.0, 0.0, 178.6, 0.0, 0.0, 0.0))
    assert robotiq_cfg.gripper is not None and robotiq_cfg.gripper.tcp_offset == pytest.approx(
        (0.0, 0.0, 173.8, 0.0, 0.0, 0.0)
    )
    assert bio_cfg.gripper is not None and bio_cfg.gripper.tcp_offset == pytest.approx(
        (157.64, 0.0, 59.5, 0.0, 0.0, 0.0)
    )
    assert g2_cfg.sha256 == "f4ecfc73c965d1163e44c68a42867cfe8157e569912ec8d2ccc6efd5be95b88c"
    assert robotiq_cfg.sha256 == "119a0c44439e3f6495f50209c05457b41eae78318ccb5105fe839d1b97bc20e0"
    assert bio_cfg.sha256 == "154ddbc4baf6485c00da7cfc11612887953d97b7629a54e9c64de2594a458454"

    overridden = load_runtime_config("xarm6", overrides={"gripper.tcp_offset": [0.0, 0.0, 180.0, 0.0, 0.0, 0.0]})
    assert overridden.sha256 != g2_cfg.sha256
    with pytest.raises(ConfigError, match="gripper.tcp_offset"):
        load_runtime_config("xarm6", overrides={"gripper.tcp_offset": [0.0, 0.0, 180.0]})


@pytest.mark.parametrize("robot", ROBOTS)
def test_default_approx_motion_and_simulation_rate_defaults(robot: str) -> None:
    cfg = load_runtime_config(robot)
    assert cfg.motion.allow_approx_motion is False
    assert cfg.motion.rate_hz == pytest.approx(50.0)
    assert cfg.simulation.rate_hz == pytest.approx(50.0)


def test_allow_approx_motion_and_sim_rate_overrides_change_hash_and_validate(tmp_path: Path) -> None:
    base = load_runtime_config("uf850")
    approx_cfg = load_runtime_config("uf850", overrides={"motion.allow_approx_motion": True})
    rate_cfg = load_runtime_config("uf850", overrides={"simulation.rate_hz": 250.0})
    assert approx_cfg.motion.allow_approx_motion is True
    assert rate_cfg.simulation.rate_hz == pytest.approx(250.0)
    assert approx_cfg.sha256 != base.sha256
    assert rate_cfg.sha256 != base.sha256
    assert approx_cfg.sha256 != rate_cfg.sha256

    for bad_approx in (1, "true", 0):
        with pytest.raises(ConfigError, match=r"motion\.allow_approx_motion must be a boolean"):
            load_runtime_config("uf850", overrides={"motion.allow_approx_motion": bad_approx})

    for bad_rate in (0.0, -50.0, float("nan"), float("inf"), True):
        with pytest.raises(ConfigError, match=r"simulation\.rate_hz must be finite"):
            load_runtime_config("uf850", overrides={"simulation.rate_hz": bad_rate})

    bad_yaml = tmp_path / "bad_approx.yaml"
    bad_yaml.write_text("schema_version: 1\nmotion:\n  allow_approx_motion: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="boolean"):
        load_runtime_config("uf850", config_path=bad_yaml)
