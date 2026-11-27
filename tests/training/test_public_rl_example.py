"""Repository-level contracts for the pure-PPO fixed-layout pick-place RL example."""

from __future__ import annotations

from pathlib import Path
import re

import pytest

from ufactory.training import (
    build_pick_place_task_configs,
    load_pick_place_rl_runtime_config,
    load_training_recipe,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXAMPLE = PROJECT_ROOT / "examples" / "rl" / "pick_place"


@pytest.mark.parametrize(
    ("key_path", "expected"),
    (
        (("environment", "env_version"), 3),
        (("environment", "control"), "cartesian_and_g2_gap_increment"),
        (("environment", "num_envs"), 2048),
        (("environment", "num_obs"), 70),
        (("environment", "num_actions"), 4),
        (("environment", "action_scale"), 0.005),
        (("environment", "gap_action_scale"), 0.003),
        (("environment", "pre_grasp_gap_m"), 0.050),
        (("environment", "close_floor_gap_m"), 0.012),
        (("environment", "place_success_dist_m"), 0.020),
        (("environment", "success_hold_steps"), 10),
        (("reward", "approach_potential"), 8.0),
        (("reward", "grasp_potential"), 4.0),
        (("reward", "lift_drive_potential"), 1.2),
        (("reward", "place_potential"), 9.0),
        (("reward", "release_potential"), 8.0),
        (("reward", "hold_potential"), 8.0),
        (("reward", "pre_lift_stall_penalty"), 0.04),
        (("reward", "lift_potential"), 40.0),
        (("reward", "transit_potential"), 8.0),
        (("reward", "success"), 40.0),
        (("train", "algorithm", "class_name"), "PPO"),
        (("train", "algorithm", "learning_rate"), 3e-4),
        (("train", "actor", "class_name"), "MLPModel"),
        (("train", "actor", "distribution_cfg", "std_range"), [0.16, 0.65]),
        (("train", "critic", "class_name"), "MLPModel"),
        (("train", "runner", "max_iterations"), 2000),
        (("train", "runner", "early_stop_evaluations"), 2),
        (("train", "num_steps_per_env"), 32),
    ),
)
def test_canonical_recipe_contract(key_path: tuple[str, ...], expected: object) -> None:
    value = load_training_recipe(EXAMPLE / "recipe.yaml")
    for key in key_path:
        value = value[key]
    assert value == expected


def test_public_rl_tree_contains_only_pure_ppo_files() -> None:
    assert sorted(path.name for path in EXAMPLE.glob("*.yaml")) == ["recipe.yaml"]
    assert not (EXAMPLE / "pretrained").exists()
    assert not (EXAMPLE / "scenarios").exists()
    assert not (EXAMPLE / "random_start").exists()
    assert not any(EXAMPLE.rglob("*.sh"))


def test_public_entries_are_module_only_and_never_use_unsafe_rsl_loading() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in EXAMPLE.glob("*.py"))
    assert "sys.path" not in combined
    assert "weights_only=False" not in combined
    assert ".load(str(" not in combined
    assert 'ROOT / "outputs" / "rl" / "pick_place"' in combined


def test_pick_place_task_configs_bind_current_runtime_hash() -> None:
    pytest.importorskip("trimesh")
    env_cfg, reward_cfg, robot_cfg = build_pick_place_task_configs("xarm6", recipe_path=EXAMPLE / "recipe.yaml")
    rl_config = load_pick_place_rl_runtime_config("xarm6")
    assert env_cfg["runtime_config_sha256"] == rl_config.sha256
    assert env_cfg["num_obs"] == 70
    assert env_cfg["num_actions"] == 4
    assert reward_cfg["release_potential"] == pytest.approx(8.0)
    assert set(robot_cfg) == {"gripper_damping", "gripper_frictionloss"}


def test_training_provenance_hashes_shared_simulation_and_training_policy() -> None:
    source = (EXAMPLE / "train.py").read_text(encoding="utf-8")
    for relative_path in (
        "ufactory/simulation/__init__.py",
        "ufactory/simulation/compat.py",
        "ufactory/simulation/g2_adapter.py",
        "ufactory/simulation/g2_controller.py",
        "ufactory/simulation/g2_model.py",
        "ufactory/simulation/physics.py",
        "ufactory/training/logic/__init__.py",
        "ufactory/training/artifacts.py",
        "ufactory/training/runner.py",
        "ufactory/training/runtime.py",
        "ufactory/training/tasks.py",
    ):
        assert f'repo_root / "{relative_path}"' in source


def test_training_provenance_sources_exist() -> None:
    source = (EXAMPLE / "train.py").read_text(encoding="utf-8")
    paths = re.findall(r'repo_root / "([^"]+)"', source)
    assert paths
    missing = [path for path in paths if not (PROJECT_ROOT / path).is_file()]
    assert missing == []


def test_no_legacy_checkpoint_or_random_start_contract_is_published() -> None:
    for readme in (EXAMPLE / "README.md", EXAMPLE / "README_zh.md"):
        text = readme.read_text(encoding="utf-8")
        assert "model_199" not in text
        assert "model_299_g2stable" not in text
        assert "Genesis World 1.3.1" not in text
        assert "random_start/" not in text
