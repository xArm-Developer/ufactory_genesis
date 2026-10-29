"""Strict RL recipe and runtime-binding tests."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from conftest import PROJECT_ROOT
from ufactory.config import load_runtime_config
from ufactory.training import (
    PICK_PLACE_RL_NOSLIP_ITERATIONS,
    ArtifactError,
    build_pick_place_task_configs,
    build_train_config,
    load_pick_place_rl_runtime_config,
    load_training_recipe,
)

GRASP_RECIPE = PROJECT_ROOT / "examples" / "rl" / "pick_place" / "recipe.yaml"


def test_pick_place_binds_the_resolved_runtime_hash() -> None:
    pytest.importorskip("trimesh")
    grasp_env, _reward, _robot = build_pick_place_task_configs("xarm6", recipe_path=GRASP_RECIPE)

    rl_config = load_pick_place_rl_runtime_config("xarm6")
    assert grasp_env["runtime_config_sha256"] == rl_config.sha256
    assert grasp_env["noslip_iterations"] == rl_config.simulation.noslip_iterations == PICK_PLACE_RL_NOSLIP_ITERATIONS
    assert rl_config.sha256 == load_runtime_config("xarm6", task="pick_place").sha256
    assert grasp_env["num_obs"] == 70
    assert grasp_env["num_actions"] == 4
    assert grasp_env["spawn_xy_lower"] == pytest.approx([0.30, -0.20])
    assert grasp_env["spawn_xy_upper"] == pytest.approx([0.50, 0.20])
    assert grasp_env["workspace_lower"] == pytest.approx([0.10, -0.35, -0.02])
    assert grasp_env["workspace_upper"] == pytest.approx([0.65, 0.35, 0.55])


def test_pick_place_rl_rejects_non_xarm6_robots() -> None:
    with pytest.raises(ValueError, match="only xArm6"):
        build_pick_place_task_configs("lite6", recipe_path=GRASP_RECIPE)


@pytest.mark.parametrize(
    "overlay",
    [
        {"arm": {"kp": [1] * 6}},
        {"robot": {"urdf": "xarm6_1305.urdf"}},
        {"task": {"parameters": {"default_ee_position_m": [0.3, 0.0, 0.25]}}},
        {"task": {"parameters": {"object_size_m": [0.04] * 3}}},
        {"task": {"parameters": {"fixed_object_position_m": [0.3, 0.1, 0.015]}}},
        {"task": {"parameters": {"episode_length_s": 12}}},
        {"simulation": {"noslip_iterations": 5}},
        {"simulation": {"g2": {"position_mm": 10}}},
    ],
)
def test_pick_place_rejects_unused_runtime_overrides(tmp_path, overlay) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(_yaml({"schema_version": 1, **overlay}))
    with pytest.raises(ValueError, match="unsupported pick-place RL runtime override"):
        load_pick_place_rl_runtime_config("xarm6", config_path=path)


@pytest.mark.parametrize(
    "field,value", [("randomize_target", True), ("fixed_demo_layout", False), ("base_pos", [1, 0, 0])]
)
def test_pick_place_rejects_unused_recipe_fields(tmp_path, field, value) -> None:
    recipe = load_training_recipe(GRASP_RECIPE)
    recipe["environment"][field] = value
    path = tmp_path / "recipe.yaml"
    path.write_text(_yaml(recipe))
    with pytest.raises(ValueError, match="unsupported pick-place RL environment fields"):
        build_pick_place_task_configs("xarm6", recipe_path=path)


def test_pick_place_supported_overrides_reach_the_environment(tmp_path) -> None:
    pytest.importorskip("trimesh")
    path = tmp_path / "runtime.yaml"
    path.write_text(
        _yaml(
            {
                "schema_version": 1,
                "task": {"parameters": {"object_mass_kg": 0.02, "workspace_upper_m": [0.6, 0.35, 0.55]}},
                "gripper": {"damping": 0.2, "frictionloss": 0.1},
                "simulation": {"g2": {"speed_percent": 90, "force_percent": 80}},
            }
        )
    )
    env, _, robot = build_pick_place_task_configs("xarm6", recipe_path=GRASP_RECIPE, runtime_config_path=path)
    assert env["obj_mass_kg"] == 0.02
    assert env["workspace_upper"] == [0.6, 0.35, 0.55]
    assert env["g2"] == {"speed_percent": 90, "force_percent": 80}
    assert robot == {"gripper_damping": 0.2, "gripper_frictionloss": 0.1}
    assert not {"table_height", "obj_size", "obj_spawn_lower", "fixed_obj_pos", "default_ee_pos"} & env.keys()


def test_command_line_train_settings_override_recipe() -> None:
    train = build_train_config(GRASP_RECIPE, experiment_name="override", max_iterations=3)
    assert train["runner"]["experiment_name"] == "override"
    assert train["runner"]["max_iterations"] == 3
    assert train["algorithm"]["gamma"] == pytest.approx(0.995)


def test_recipe_rejects_unknown_top_level_and_runtime_owned_fields(tmp_path: Path) -> None:
    recipe = load_training_recipe(GRASP_RECIPE)
    recipe["unknown"] = {}
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text(_yaml(recipe), encoding="utf-8")
    with pytest.raises(ArtifactError, match="fields must be exactly"):
        load_training_recipe(unknown)

    recipe.pop("unknown")
    recipe["environment"]["object_mass_kg"] = 0.1
    physical = tmp_path / "physical.yaml"
    physical.write_text(_yaml(recipe), encoding="utf-8")
    with pytest.raises(ArtifactError, match="ResolvedRuntimeConfig"):
        load_training_recipe(physical)


def test_recipe_rejects_non_finite_values(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        "schema_version: 1\nenvironment: {num_envs: 1}\nreward: {keypoints: .nan}\ntrain: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ArtifactError, match="must be finite"):
        load_training_recipe(invalid)


def test_reach_configuration_resolves_without_torch() -> None:
    code = """
import importlib.abc, sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("torch deliberately unavailable")
sys.meta_path.insert(0, BlockTorch())
from ufactory.training import build_reach_task_configs, build_train_config
env, _, _ = build_reach_task_configs("xarm6", recipe_path="examples/rl/reach/recipe.yaml")
assert env["num_obs"] == 24
build_train_config("examples/rl/reach/recipe.yaml", experiment_name="test", max_iterations=1)
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def _yaml(value: object) -> str:
    import yaml

    return yaml.safe_dump(value, sort_keys=False)
