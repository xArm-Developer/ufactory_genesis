"""Control and artifact regression tests for end-to-end pick-place PPO."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from examples.rl.pick_place.contacts import table_contact_loads
from ufactory.config.models import G2_PHYSICS_PROFILE
from ufactory.training import (
    ArtifactError,
    load_training_config,
    validate_checkpoint_artifacts,
    write_checkpoint_manifest,
    write_training_config,
)
from ufactory.training.pick_place_config import PICK_PLACE_ACTION_CONTRACT


def controller(monkeypatch):
    from examples.rl.pick_place.runtime import configure_runtime

    configure_runtime()
    pytest.importorskip("genesis")
    from examples.rl.pick_place import env as module
    from examples.rl.pick_place.env import XArm6PickPlaceEnv

    env = XArm6PickPlaceEnv.__new__(XArm6PickPlaceEnv)
    env.num_envs, env.num_actions = 2, 4
    env.action_scale, env.gap_action_scale, env.setpoint_leash_m = 0.005, 0.003, 0.025
    env.close_floor_gap_m, env.g2_speed_percent, env.g2_force_percent, env.ctrl_dt = 0.012, 50, 50, 0.02
    env.arm_dof_idx, env.gripper_dof_idx = list(range(6)), [6]
    env.ee_setpoint = torch.tensor([[0.3, 0.0, 0.3], [0.3, 0.3, 0.19]])
    env.gap_cmd = torch.tensor([0.04, 0.025])
    env.q_target = torch.zeros(2, 6)
    env.workspace_lower = torch.tensor([0.1, -0.35, -0.02])
    env.workspace_upper = torch.tensor([0.65, 0.35, 0.55])
    env.ee_link = SimpleNamespace(get_pos=lambda: env.ee_setpoint.clone())
    env.robot = SimpleNamespace(control_dofs_position=lambda *args: None)
    env.g2_controller = SimpleNamespace(command=lambda *args: None)
    env.last_actions = torch.zeros(2, 4)
    env._solve_ik = lambda target: (env.q_target.clone(), torch.zeros(2, dtype=torch.bool))
    monkeypatch.setattr(module, "apply_batch_control", lambda *args: None)
    return env


def test_all_four_channels_work_at_pickup_and_placement(monkeypatch):
    env = controller(monkeypatch)
    old = env.ee_setpoint.clone()
    gap = env.gap_cmd.clone()
    actions = torch.tensor([[0.5, -0.5, -0.5, -0.5], [-0.5, 0.5, 0.5, 0.5]])
    env._apply_action(actions, torch.tensor([0.05, 0.03]))
    torch.testing.assert_close(env.ee_setpoint, old + actions[:, :3] * 0.005)
    torch.testing.assert_close(env.gap_cmd, gap + actions[:, 3] * 0.003)
    env._apply_action(-actions, torch.tensor([0.04, 0.04]))
    torch.testing.assert_close(env.ee_setpoint, old)
    torch.testing.assert_close(env.gap_cmd, gap)


def test_zero_actions_hold_commands_despite_measured_drift(monkeypatch):
    env = controller(monkeypatch)
    old = env.ee_setpoint.clone()
    gap, q = env.gap_cmd.clone(), env.q_target.clone()
    env.ee_link = SimpleNamespace(get_pos=lambda: old + 0.04)
    env._apply_action(torch.zeros(2, 4), torch.tensor([0.08, 0.01]))
    torch.testing.assert_close(env.ee_setpoint, old, rtol=0, atol=0)
    torch.testing.assert_close(env.gap_cmd, gap, rtol=0, atol=0)
    torch.testing.assert_close(env.q_target, q, rtol=0, atol=0)


def test_approach_potential_prefers_open_gap_until_grasp_alignment(monkeypatch):
    env = controller(monkeypatch)
    from examples.rl.pick_place import env as module

    obj = torch.tensor([[0.3, 0.0, 0.015]]).repeat(2, 1)
    pads = obj.clone()
    pads[:, 2] = 0.05
    env.obj = SimpleNamespace(get_pos=lambda: obj, get_quat=lambda: torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1))
    monkeypatch.setattr(module, "pad_centers", lambda *args: (pads, pads))
    env.g2_model, env.left_finger_link, env.right_finger_link = None, None, None
    env.pre_grasp_gap_m, env.reward_scales = 0.05, {}
    env.target_xy = torch.tensor([[0.3, 0.3]])
    env.forces = torch.zeros(2, 2)
    env.place_rewarded = torch.zeros(2, dtype=torch.bool)
    env.metrics = SimpleNamespace(
        max_held_z=torch.full((2,), 0.015),
        ever_lifted_clear=torch.zeros(2, dtype=torch.bool),
        release_confirmed=torch.zeros(2, dtype=torch.bool),
        touchdown_seen=torch.zeros(2, dtype=torch.bool),
        touchdown_ee_z=torch.zeros(2),
        hold_steps=torch.zeros(2, dtype=torch.long),
        required_hold_steps=10,
        place_clearance=0.002,
        release_max_height=0.003,
    )
    env.measured_gap = lambda: torch.tensor([0.05, 0.012])
    potential = env._potential()
    assert potential[0] > potential[1] + 1.0
    # Once aligned, starting to close (50 -> 44 mm) and grasping must both increase potential monotonically.
    pads[:, 2] = 0.027
    env.gap_cmd = torch.tensor([0.050, 0.044])
    env.measured_gap = lambda: torch.tensor([0.050, 0.044])
    closing = env._potential()
    assert closing[1] > closing[0] + 0.1
    env.measured_gap = lambda: torch.tensor([0.05, 0.03])
    env.forces[1] = 2.0
    potential = env._potential()
    assert potential[1] > potential[0] + 2.0
    # A held cube at transport height still needs a usable descent signal.
    # Move pads with the cube so this tests placement rather than approach.
    obj[:, 1] = 0.3
    obj[:, 2] = torch.tensor([0.36, 0.35])
    pads[:] = obj
    env.metrics.ever_lifted_clear[:] = True
    env.measured_gap = lambda: torch.full((2,), 0.03)
    env.forces[:] = 2.0
    potential = env._potential()
    assert potential[1] > potential[0] + 0.001
    # Even at the closed command floor (gap_cmd=12 mm), less-negative gripper
    # actions must increase potential so PPO can escape negative saturation.
    obj[:, 2] = 0.017
    pads[:] = obj
    env.gap_cmd = torch.tensor([0.012, 0.012])
    env.last_actions = torch.tensor([[0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, -0.2]])
    potential = env._potential()
    assert potential[1] > potential[0] + 0.1
    # Opening must provide progress while the command traverses the preload
    # below measured gap; otherwise PPO cannot discover release from a hold.
    env.last_actions = torch.zeros(2, 4)
    env.gap_cmd = torch.tensor([0.012, 0.015])
    potential = env._potential()
    assert potential[1] > potential[0] + 0.01
    # Unclamping one or both fingers at the target table before release
    # confirmation must continue increasing potential rather than dipping.
    env.forces = torch.tensor([[20.0, 20.0], [0.0, 20.0]])
    env.gap_cmd = torch.tensor([0.030, 0.033])
    potential = env._potential()
    assert potential[1] > potential[0] + 0.01
    # Detaching fingers and opening to 50 mm at the target table must strictly
    # beat re-closing onto the resting cube after release confirmation.
    obj[:, 2] = 0.015
    pads[:, 2] = torch.tensor([0.027, 0.040])
    env.place_rewarded[:] = True
    env.metrics.release_confirmed[:] = True
    env.forces = torch.tensor([[2.0, 2.0], [0.0, 0.0]])
    env.gap_cmd = torch.tensor([0.015, 0.050])
    env.measured_gap = lambda: torch.tensor([0.030, 0.050])
    potential = env._potential()
    assert potential[1] > potential[0] + 5.0


def test_distant_spawn_lift_has_no_negative_transit_potential_jump(monkeypatch):
    env = controller(monkeypatch)
    from examples.rl.pick_place import env as module

    spawns = torch.tensor([[0.50, -0.20], [0.30, 0.20]])
    obj = torch.tensor([[0.50, -0.20, 0.080], [0.30, 0.20, 0.080]])
    pads = obj.clone()
    env.obj = SimpleNamespace(get_pos=lambda: obj, get_quat=lambda: torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1))
    monkeypatch.setattr(module, "pad_centers", lambda *args: (pads, pads))
    env.g2_model, env.left_finger_link, env.right_finger_link = None, None, None
    env.pre_grasp_gap_m, env.reward_scales = 0.05, {}
    env.target_xy = torch.tensor([[0.3, 0.3]])
    env.forces = torch.full((2, 2), 2.0)
    env.place_rewarded = torch.zeros(2, dtype=torch.bool)
    env.metrics = SimpleNamespace(
        spawn_xy=spawns.clone(),
        max_held_z=torch.full((2,), 0.080),
        ever_lifted_clear=torch.zeros(2, dtype=torch.bool),
        release_confirmed=torch.zeros(2, dtype=torch.bool),
        touchdown_seen=torch.zeros(2, dtype=torch.bool),
        touchdown_ee_z=torch.zeros(2),
        hold_steps=torch.zeros(2, dtype=torch.long),
        required_hold_steps=10,
        place_clearance=0.002,
        release_max_height=0.003,
    )
    env.measured_gap = lambda: torch.full((2,), 0.03)
    before_lift_flag = env._potential()
    env.metrics.ever_lifted_clear[:] = True
    after_lift_flag = env._potential()
    assert torch.all(after_lift_flag >= before_lift_flag)
    # Moving 50 mm closer to target from any spawn increases potential equally in transit term.
    obj[:, 1] += 0.05
    pads[:] = obj
    moved_closer = env._potential()
    assert torch.all(moved_closer > after_lift_flag + 0.35)
    # When horizontally far from a resting cube, hovering at z=65 mm beats diving to z=25 mm.
    env.metrics.ever_lifted_clear[:] = False
    env.metrics.max_held_z[:] = 0.015
    env.forces[:] = 0.0
    env.gap_cmd = torch.full((2,), 0.05)
    env.measured_gap = lambda: torch.full((2,), 0.05)
    obj[:] = torch.tensor([[0.50, -0.20, 0.015], [0.50, -0.20, 0.015]])
    env.metrics.spawn_xy[:] = obj[:, :2]
    pads[:] = torch.tensor([[0.30, 0.00, 0.065], [0.30, 0.00, 0.025]])
    far_potential = env._potential()
    assert far_potential[0] > far_potential[1] + 0.3
    # Once horizontally aligned over the spawn (even at 8 mm offset), open-gripper
    # descent from hover (65 mm -> 45 mm -> 25 mm) must be strictly monotonic.
    pads[:] = torch.tensor([[0.508, -0.20, 0.065], [0.508, -0.20, 0.045]])
    hover_vs_mid = env._potential()
    assert hover_vs_mid[1] > hover_vs_mid[0] + 0.8
    pads[:] = torch.tensor([[0.508, -0.20, 0.045], [0.508, -0.20, 0.025]])
    mid_vs_grasp = env._potential()
    assert mid_vs_grasp[1] > mid_vs_grasp[0] + 0.9
    # Even at an 8 mm horizontal offset at grasp height, closing from 50 -> 44 mm
    # must increase potential while opening wider (50 -> 60 mm) must decrease it.
    pads[:] = torch.tensor([[0.508, -0.20, 0.025], [0.508, -0.20, 0.025]])
    env.gap_cmd = torch.tensor([0.050, 0.044])
    env.measured_gap = lambda: torch.tensor([0.050, 0.044])
    close_at_8mm = env._potential()
    assert close_at_8mm[1] > close_at_8mm[0] + 0.1
    env.gap_cmd = torch.tensor([0.050, 0.060])
    env.measured_gap = lambda: torch.tensor([0.050, 0.060])
    open_wider_at_8mm = env._potential()
    assert open_wider_at_8mm[0] > open_wider_at_8mm[1] + 0.3
    # Lowering a held cube across release_max_height over the target with a 12 mm
    # horizontal pad grasp offset must strictly increase potential without a cliff.
    env.metrics.ever_lifted_clear[:] = True
    env.metrics.max_held_z[:] = 0.080
    env.forces[:] = 2.0
    env.gap_cmd = torch.full((2,), 0.012)
    env.measured_gap = lambda: torch.full((2,), 0.030)
    obj[:] = torch.tensor([[0.30, 0.30, 0.022], [0.30, 0.30, 0.017]])
    pads[:] = obj.clone()
    pads[:, 0] += 0.012
    descent_across_at_table = env._potential()
    assert descent_across_at_table[1] > descent_across_at_table[0] + 0.5
    # Before ever_lifted_clear, lowering a held cube from 35 mm back to 15 mm must lose lift potential.
    env.metrics.ever_lifted_clear[:] = False
    env.metrics.max_held_z[:] = 0.035
    obj[:] = torch.tensor([[0.50, -0.20, 0.035], [0.50, -0.20, 0.015]])
    pads[:] = obj.clone()
    partial_lift = env._potential()
    assert partial_lift[0] > partial_lift[1] + 0.5


def test_make_spawns_is_deterministic_and_within_configured_bounds():
    import numpy as np
    from examples.rl.pick_place.evaluate import make_spawns

    first = make_spawns(64, seed=17000)
    second = make_spawns(64, seed=17000)
    other = make_spawns(64, seed=18001)
    assert first.shape == (64, 2)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert np.all((first[:, 0] >= 0.30) & (first[:, 0] <= 0.50))
    assert np.all((first[:, 1] >= -0.20) & (first[:, 1] <= 0.20))
    assert float(np.std(first[:, 0])) > 0.03
    assert float(np.std(first[:, 1])) > 0.06


def test_rejected_ik_keeps_command_and_gap_remains_active(monkeypatch):
    env = controller(monkeypatch)
    old = env.ee_setpoint.clone()
    env._solve_ik = lambda target: (env.q_target, torch.ones(2, dtype=torch.bool))
    env._apply_action(torch.ones(2, 4), torch.tensor([0.03, 0.03]))
    torch.testing.assert_close(env.ee_setpoint, old)
    torch.testing.assert_close(env.gap_cmd, torch.tensor([0.043, 0.028]))


def test_identical_clipped_actions_have_a_gradient_out_of_saturation(monkeypatch):
    env = controller(monkeypatch)
    env.action_clip, env.reward_scales = 1.0, {}
    raw = torch.tensor([[-1.7, -0.6, -1.0, -2.5], [-1.0, -0.6, -1.0, -1.0]])
    torch.testing.assert_close(raw[0].clamp(-1, 1), raw[1].clamp(-1, 1))
    penalty = env._action_penalty(raw)
    assert penalty[0] > penalty[1] + 0.25
    # Within the command bounds, the regularizer remains the existing small
    # action cost; all four channels still control the environment directly.
    torch.testing.assert_close(penalty[1], 0.001 * raw[1].square().sum())


def test_table_contact_forces_filter_padding_pairs_and_side_order():
    contacts = {
        "link_a": torch.tensor([[1, 2, 3, 2, 2], [2, 1, 3, 2, 2]]),
        "link_b": torch.tensor([[2, 1, 2, 3, 1], [1, 2, 2, 3, 1]]),
        "force_a": torch.tensor(
            [
                [[0.0, 0.0, 0.1], [0.0, 0.0, -0.2], [0.0, 0.0, 4.0], [0.0, 0.0, -8.0], [0.0, 0.0, -99.0]],
                [[0.0, 0.0, -0.3], [0.0, 0.0, 0.4], [0.0, 0.0, 5.0], [0.0, 0.0, -6.0], [0.0, 0.0, -99.0]],
            ]
        ),
        "valid_mask": torch.tensor([[True, True, True, False, False], [True, True, False, True, False]]),
    }
    contacts["force_b"] = -contacts["force_a"]
    support, robot = table_contact_loads(contacts, table_link_idx=2, object_link_idx=1)
    torch.testing.assert_close(support, torch.tensor([0.3, 0.7]))
    torch.testing.assert_close(robot, torch.tensor([4.0, 6.0]))
    unbatched = {k: v[0, :3] for k, v in contacts.items() if k != "valid_mask"}
    torch.testing.assert_close(table_contact_loads(unbatched, table_link_idx=2, object_link_idx=1)[0], support[:1])
    empty = {k: v[:, :0] for k, v in contacts.items()}
    support, robot = table_contact_loads(empty, table_link_idx=2, object_link_idx=1)
    assert support.tolist() == robot.tolist() == [0.0, 0.0]


def artifact(tmp_path: Path, env: dict, contract: str):
    config = tmp_path / "config.yaml"
    write_training_config(config, task="pick_place", robot_key="xarm6_1305", env=env, reward={}, robot={}, train={})
    checkpoint = tmp_path / "model_0.pt"
    torch.save({"actor_state_dict": {}, "critic_state_dict": {}}, checkpoint)
    write_checkpoint_manifest(
        checkpoint, training_config=load_training_config(config), executor_action_contract=contract
    )
    return checkpoint, config


def test_old_phase_checkpoint_requires_retraining(tmp_path):
    env = {"env_version": 2, "num_obs": 42, "num_actions": 4, "physics_profile": G2_PHYSICS_PROFILE}
    checkpoint, config = artifact(tmp_path, env, "phase_guarded_cartesian_and_g2_gap_delta")
    with pytest.raises(ArtifactError, match="v3.*retrain"):
        validate_checkpoint_artifacts(
            checkpoint,
            config,
            expected_task="pick_place",
            expected_runtime_env={"env_version": 3},
        )


@pytest.mark.parametrize("field,value", [("action_scale", 0.01), ("press_limit_m", 0.002), ("control", "guarded")])
def test_v3_control_contract_is_validated(tmp_path, field, value):
    env = {
        "task_name": "pick_place",
        "env_version": 3,
        "num_obs": 70,
        "num_actions": 4,
        "action_scale": 0.005,
        "press_limit_m": 0.001,
        "control": "cartesian_and_g2_gap_increment",
        "physics_profile": G2_PHYSICS_PROFILE,
    }
    checkpoint, config = artifact(tmp_path, env, PICK_PLACE_ACTION_CONTRACT)
    expected = deepcopy(env)
    expected[field] = value
    with pytest.raises(ArtifactError, match="runtime config body"):
        validate_checkpoint_artifacts(checkpoint, config, expected_task="pick_place", expected_runtime_env=expected)


def test_v3_manifest_rejects_wrong_executor(tmp_path):
    env = {
        "task_name": "pick_place",
        "env_version": 3,
        "num_obs": 70,
        "num_actions": 4,
        "physics_profile": G2_PHYSICS_PROFILE,
    }
    checkpoint, config = artifact(tmp_path, env, "guarded")
    with pytest.raises(ArtifactError, match="action contract"):
        validate_checkpoint_artifacts(checkpoint, config)


def test_v3_checkpoint_loads_when_contract_matches(tmp_path):
    from ufactory.training import build_pick_place_task_configs, validate_and_load_rsl_checkpoint

    recipe = Path(__file__).resolve().parents[2] / "examples/rl/pick_place/recipe.yaml"
    expected, _, _ = build_pick_place_task_configs("xarm6", recipe_path=recipe)
    checkpoint, config = artifact(tmp_path, expected, PICK_PLACE_ACTION_CONTRACT)
    state, saved, manifest = validate_and_load_rsl_checkpoint(
        checkpoint,
        config,
        expected_task="pick_place",
        expected_runtime_env=expected,
        expected_runtime_config_sha256=expected["runtime_config_sha256"],
    )
    assert state == {"actor_state_dict": {}, "critic_state_dict": {}}
    assert saved["env"] == expected
    assert manifest.executor_action_contract == PICK_PLACE_ACTION_CONTRACT
    with checkpoint.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ArtifactError, match="checkpoint hash"):
        validate_checkpoint_artifacts(checkpoint, config, expected_runtime_env=expected)


def test_public_pick_place_policy_samples_stay_finite_and_bounded() -> None:
    from importlib import metadata
    from packaging.version import Version
    from tensordict import TensorDict
    from ufactory.training import load_training_recipe

    pytest.importorskip("rsl_rl")
    try:
        version = Version(metadata.version("rsl-rl-lib"))
    except metadata.PackageNotFoundError:
        pytest.skip("rsl-rl-lib is not installed")
    if version < Version("5.3.0"):
        pytest.skip("requires rsl-rl-lib >= 5.3")

    from rsl_rl.models import MLPModel

    recipe = load_training_recipe(Path(__file__).resolve().parents[2] / "examples/rl/pick_place/recipe.yaml")
    obs = TensorDict({"policy": torch.randn(4096, 70)}, batch_size=[4096])
    actor_cfg = deepcopy(recipe["train"]["actor"])
    assert actor_cfg.pop("class_name") == "MLPModel"
    actor = MLPModel(
        obs,
        recipe["train"]["obs_groups"],
        "actor",
        output_dim=4,
        **actor_cfg,
    )
    stochastic = actor(obs, stochastic_output=True).clamp(-1.0, 1.0)
    deterministic = actor(obs, stochastic_output=False).clamp(-1.0, 1.0)
    assert stochastic.shape == (4096, 4)
    assert torch.isfinite(stochastic).all()
    assert torch.isfinite(deterministic).all()
    assert torch.all((stochastic >= -1.0) & (stochastic <= 1.0))
    assert torch.all((deterministic >= -1.0) & (deterministic <= 1.0))


def test_fixed_learning_rate_guard_checks_every_update() -> None:
    from examples.rl.pick_place.train import _install_fixed_learning_rate_guard

    parameter = torch.nn.Parameter(torch.tensor([0.0]))
    optimizer = torch.optim.Adam([parameter], lr=3e-5)
    algorithm = SimpleNamespace(
        learning_rate=3e-5,
        optimizer=optimizer,
        update=lambda: "ok",
    )
    runner = SimpleNamespace(alg=algorithm)
    cfg = {"algorithm": {"schedule": "fixed", "learning_rate": 3e-5}}
    _install_fixed_learning_rate_guard(runner, cfg)
    assert runner.alg.update() == "ok"

    def changes_rate() -> None:
        runner.alg.learning_rate = 1e-3
        optimizer.param_groups[0]["lr"] = 1e-3

    runner.alg.update = changes_rate
    _install_fixed_learning_rate_guard(runner, cfg)
    with pytest.raises(RuntimeError, match="fixed learning rate changed after"):
        runner.alg.update()


def test_checkpoint_ranking_and_acceptance_use_pressing():
    from examples.rl.pick_place.evaluate import acceptance_passed, checkpoint_score

    def summary(success, press, error, touchdown=1.0, high_release=0.0):
        return {
            "success_rate": success,
            "touchdown_seen_rate": touchdown,
            "high_release_rate": high_release,
            "press_mm": {"max": press},
            "target_error_mm": {"mean": error},
        }

    assert checkpoint_score(summary(1.0, 0.2, 5.0)) > checkpoint_score(summary(1.0, 0.8, 1.0))
    assert checkpoint_score(summary(1.0, 0.8, 5.0)) > checkpoint_score(summary(0.98, 0.2, 1.0))
    assert checkpoint_score(summary(0.0, 1.04, 3.3, touchdown=1.0)) > checkpoint_score(
        summary(0.0, 0.0, 3.5, touchdown=0.0)
    )
    assert acceptance_passed(summary(63 / 64, 1.0, 5.0))
    assert not acceptance_passed(summary(62 / 64, 0.5, 5.0))
    assert not acceptance_passed(summary(1.0, 1.01, 5.0))


def test_three_seed_runner_preserves_failed_evaluation_and_still_records(monkeypatch, tmp_path):
    import json
    import subprocess
    from examples.rl.pick_place import run_seeds

    monkeypatch.setattr(run_seeds, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["run_seeds"])
    commands = []

    def run(command, log):
        commands.append(command)
        module = command[3]
        if module.endswith(".train"):
            assert command[command.index("--max-iterations") + 1] == "2000"
            assert command[command.index("-B", 4) + 1] == "2048"
            assert command[command.index("--eval-interval") + 1] == "50"
            assert "--resume" not in command
            name = command[command.index("-e") + 1]
            (tmp_path / name).mkdir()
        else:
            out = Path(command[command.index("--out") + 1])
            failed = "s2" in str(out) and module.endswith(".evaluate")
            out.write_text(json.dumps({"passed": not failed}))
            if module.endswith(".evaluate"):
                assert command[command.index("--episodes") + 1] == "64"
            else:
                assert command[command.index("-B", 4) + 1] == "1"
                assert "--record" in command
            if failed:
                raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(run_seeds, "run", run)
    with pytest.raises(SystemExit) as error:
        run_seeds.main()
    assert error.value.code == 1
    assert len(commands) == 9
    report = json.loads((tmp_path / "e2e_validation" / "summary.json").read_text())
    assert set(report["seeds"]) == {"1", "2", "3"}
    assert not report["passed"]
    assert report["seeds"]["1"]["passed"] and report["seeds"]["3"]["passed"]
    assert "glb" in report["seeds"]["2"]
