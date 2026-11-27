"""Train the xArm6 + Gripper G2 tabletop pick-and-place task with pure PPO."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
from typing import Any

from .runtime import (
    DEFAULT_RECIPE,
    OUTPUT_ROOT,
    configure_runtime,
    require_backend,
    require_checkpoint,
    require_rsl_runtime,
)

configure_runtime()
require_rsl_runtime()

import genesis as gs
import torch

from ufactory.robots.paths import robot_urdf
from ufactory.simulation.compat import require_genesis_runtime
from ufactory.training import (
    PICK_PLACE_RL_ROBOTS,
    build_pick_place_task_configs,
    build_train_config,
    load_pick_place_rl_runtime_config,
    load_training_config,
    load_training_recipe,
    validate_and_load_rsl_checkpoint,
    write_artifact_inventory,
    write_run_provenance,
    write_training_config,
)
from ufactory.training.pick_place_config import PICK_PLACE_ACTION_CONTRACT, PICK_PLACE_ROBOT, PICK_PLACE_URDF
from ufactory.training.runner import ArtifactOnPolicyRunner
from ufactory.training.runtime import prepare_log_dir

from .env import XArm6PickPlaceEnv
from .evaluate import acceptance_passed, checkpoint_score, evaluate, save_results

EXECUTOR_ACTION_CONTRACT = PICK_PLACE_ACTION_CONTRACT


def _install_fixed_learning_rate_guard(runner: Any, train_cfg: dict[str, Any]) -> None:
    """Reject accidental learning-rate schedule changes when schedule == 'fixed'."""
    alg_cfg = train_cfg.get("algorithm", {})
    if str(alg_cfg.get("schedule", "")).lower() != "fixed":
        return
    expected_lr = float(alg_cfg["learning_rate"])
    original_update = runner.alg.update

    def guarded_update(*args: Any, **kwargs: Any) -> Any:
        result = original_update(*args, **kwargs)
        if hasattr(runner.alg, "actor"):
            dist = getattr(runner.alg.actor, "distribution", None)
            with torch.no_grad():
                if dist is not None and hasattr(dist, "std_param"):
                    dist.std_param.clamp_(min=0.16, max=0.65)
                elif dist is not None and hasattr(dist, "log_std_param"):
                    dist.log_std_param.clamp_(min=math.log(0.16), max=math.log(0.65))
                elif hasattr(runner.alg.actor, "std"):
                    runner.alg.actor.std.clamp_(min=0.16, max=0.65)
                elif hasattr(runner.alg.actor, "log_std"):
                    runner.alg.actor.log_std.clamp_(min=math.log(0.16), max=math.log(0.65))
        actual_lr = float(runner.alg.learning_rate)
        group_lrs = [float(group["lr"]) for group in runner.alg.optimizer.param_groups]
        if not math.isclose(actual_lr, expected_lr, rel_tol=1e-9, abs_tol=1e-12) or any(
            not math.isclose(lr, expected_lr, rel_tol=1e-9, abs_tol=1e-12) for lr in group_lrs
        ):
            raise RuntimeError(
                f"fixed learning rate changed after update: expected {expected_lr:g}, got alg={actual_lr:g}, optimizer={group_lrs}"
            )
        return result

    runner.alg.update = guarded_update


def _recent_training_stable(log_dir: Path, window: int = 15) -> bool:
    metrics_path = log_dir / "training_metrics.jsonl"
    if not metrics_path.is_file():
        return True
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    valid = [r.get("episode", {}) for r in rows if int(r.get("completed_episodes", 0)) >= 32]
    recent = valid[-window:] if valid else [r.get("episode", {}) for r in rows[-window:]]
    if not recent:
        return True
    mean_sr = sum(float(ep.get("success_rate", 0.0)) for ep in recent) / len(recent)
    mean_col_press = sum(
        float(ep.get("collision_failed", 0.0)) + float(ep.get("press_failed", 0.0)) for ep in recent
    ) / len(recent)
    return mean_sr >= 0.88 and mean_col_press <= 0.035


def _provenance_sources(repo_root: Path) -> list[Path]:
    return [
        Path(__file__).resolve(),
        Path(__file__).with_name("env.py"),
        Path(__file__).with_name("metrics.py"),
        Path(__file__).with_name("contacts.py"),
        Path(__file__).with_name("runtime.py"),
        Path(__file__).with_name("evaluate.py"),
        Path(__file__).with_name("recipe.yaml"),
        repo_root / "ufactory/simulation/__init__.py",
        repo_root / "ufactory/simulation/compat.py",
        repo_root / "ufactory/simulation/g2_adapter.py",
        repo_root / "ufactory/simulation/g2_controller.py",
        repo_root / "ufactory/simulation/g2_model.py",
        repo_root / "ufactory/simulation/physics.py",
        repo_root / "ufactory/training/logic/__init__.py",
        repo_root / "ufactory/training/artifacts.py",
        repo_root / "ufactory/training/runner.py",
        repo_root / "ufactory/training/runtime.py",
        Path(robot_urdf(PICK_PLACE_ROBOT, PICK_PLACE_URDF)),
        repo_root / "ufactory/training/tasks.py",
        repo_root / "ufactory/training/pick_place_config.py",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="xarm6", choices=PICK_PLACE_RL_ROBOTS)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("-e", "--exp-name", default="pick_place_e2e_s1")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("-v", "--vis", action="store_true")
    parser.add_argument("-B", "--num-envs", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-start-iteration", type=int, default=0)
    parser.add_argument(
        "--early-stop-evaluations", type=int, help="Consecutive passing evaluations before stopping; 0 disables"
    )
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    if args.num_envs is not None and args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if args.max_iterations is not None and args.max_iterations < 1:
        parser.error("--max-iterations must be positive")
    if args.eval_episodes < 1:
        parser.error("--eval-episodes must be positive")
    if args.eval_interval < 1:
        parser.error("--eval-interval must be positive")
    if args.early_stop_evaluations is not None and args.early_stop_evaluations < 0:
        parser.error("--early-stop-evaluations must be non-negative")

    num_threads = max(1, int(os.environ.get("OMP_NUM_THREADS", "4")))
    torch.set_num_threads(num_threads)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    recipe = load_training_recipe(args.recipe)
    early_stop_evaluations = int(
        args.early_stop_evaluations
        if args.early_stop_evaluations is not None
        else recipe["train"]["runner"].get("early_stop_evaluations", 2)
    )
    num_envs = int(args.num_envs if args.num_envs is not None else recipe["environment"]["num_envs"])
    max_iterations = int(
        args.max_iterations if args.max_iterations is not None else recipe["train"]["runner"]["max_iterations"]
    )
    runtime_config = load_pick_place_rl_runtime_config(args.robot, config_path=args.runtime_config)
    env_cfg, reward_cfg, robot_cfg = build_pick_place_task_configs(
        args.robot,
        recipe_path=args.recipe,
        runtime_config_path=args.runtime_config,
    )
    env_cfg["num_envs"] = num_envs
    performance_mode = bool(recipe["environment"].get("genesis_performance_mode", False))
    env_cfg["genesis_performance_mode"] = performance_mode
    train_cfg = build_train_config(
        args.recipe,
        experiment_name=args.exp_name,
        max_iterations=max_iterations,
    )
    if args.seed is not None:
        train_cfg["seed"] = int(args.seed)
    train_cfg["runner"]["early_stop_evaluations"] = early_stop_evaluations
    train_cfg["runner"]["resume"] = args.resume is not None
    train_cfg["runner"]["resume_path"] = None if args.resume is None else str(args.resume)

    resume_state = None
    if args.resume is not None:
        args.resume = require_checkpoint(args.resume)
        resume_state, resume_config, _resume_manifest = validate_and_load_rsl_checkpoint(
            args.resume,
            args.resume.parent / "config.yaml",
            map_location="cpu",
            expected_task="pick_place",
            expected_robot_key=runtime_config.robot.key,
            expected_runtime_config_sha256=runtime_config.sha256,
            expected_runtime_env=env_cfg,
        )
        if int(resume_config["env"]["num_envs"]) != num_envs:
            parser.error("resume requires the original --num-envs; episodes restart from home")

    log_dir = args.log_dir or (OUTPUT_ROOT / args.exp_name)
    prepare_log_dir(log_dir, overwrite=args.overwrite)
    repo_root = Path(__file__).resolve().parents[3]

    write_training_config(
        log_dir / "config.yaml",
        task="pick_place",
        robot_key=runtime_config.robot.key,
        env=env_cfg,
        reward=reward_cfg,
        robot=robot_cfg,
        train=train_cfg,
    )
    training_config = load_training_config(log_dir / "config.yaml")
    write_run_provenance(
        log_dir / "provenance.json",
        training_config=training_config,
        source_paths=_provenance_sources(repo_root),
    )

    require_genesis_runtime(gs)
    gs.init(
        backend=gs.cpu if args.backend == "cpu" else gs.gpu,
        precision="32",
        logging_level="warning",
        seed=int(train_cfg["seed"]),
        performance_mode=performance_mode,
    )
    require_backend(gs, args.backend)
    env = XArm6PickPlaceEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=args.vis,
    )

    runner_cfg = deepcopy(train_cfg)
    runner_cfg["runner"]["resume"] = False
    runner_cfg["runner"]["resume_path"] = None
    runner = ArtifactOnPolicyRunner(
        env, runner_cfg, log_dir=str(log_dir), device=gs.device, executor_action_contract=EXECUTOR_ACTION_CONTRACT
    )
    runner.checkpoint_training_config = training_config
    runner.checkpoint_env = env
    _install_fixed_learning_rate_guard(runner, runner_cfg)

    if resume_state is not None:
        runner.alg.load(resume_state, None, strict=True)
        runner.current_learning_iteration = int(resume_state.get("iter", 0)) + 1
        infos = resume_state.get("infos") or {}
        if "ufactory_env_state" in infos:
            env.load_training_state_dict(infos["ufactory_env_state"])

    best_score = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, float("-inf"), float("-inf"))
    passing_evaluations = 0

    def validate(tag: str) -> dict[str, Any]:
        nonlocal best_score, passing_evaluations
        actor = runner.get_inference_policy(device=gs.device)

        def _policy(obs: Any) -> torch.Tensor:
            return actor(obs, stochastic_output=False)

        summary, rows = evaluate(env, _policy, episodes=args.eval_episodes, seed=17000)
        summary["passed"] = acceptance_passed(summary)
        holdout_passed = False
        combined_sr = float(summary["success_rate"])
        combined_press = float(summary["press_mm"]["max"])
        if summary["passed"] and tag != "initial":
            holdout_eps = max(128, int(args.eval_episodes) * 2)
            holdout_summary, _ = evaluate(env, _policy, episodes=holdout_eps, seed=17500)
            holdout_passed = acceptance_passed(holdout_summary)
            total_eps = float(args.eval_episodes + holdout_eps)
            combined_sr = (
                float(summary["success_rate"]) * args.eval_episodes
                + float(holdout_summary["success_rate"]) * holdout_eps
            ) / total_eps
            combined_press = max(combined_press, float(holdout_summary["press_mm"]["max"]))
            summary["holdout_17500"] = {
                "episodes": holdout_eps,
                "success_rate": float(holdout_summary["success_rate"]),
                "max_press_mm": float(holdout_summary["press_mm"]["max"]),
                "passed": holdout_passed,
            }
        both_passed = bool(summary["passed"] and holdout_passed)
        stable_pass = bool(
            both_passed and combined_sr >= 0.994 and combined_press <= 0.8 and _recent_training_stable(log_dir)
        )
        summary["both_passed"] = both_passed
        summary["stable_pass"] = stable_pass
        summary["combined_sr"] = combined_sr
        summary["combined_press_mm"] = combined_press
        if tag != "initial":
            passing_evaluations = passing_evaluations + 1 if stable_pass else 0
        save_results(log_dir / f"evaluation_{tag}.json", summary, rows)
        print(f"STRICT EVAL {tag}: {json.dumps(summary)}", flush=True)
        streak_tier = min(passing_evaluations, early_stop_evaluations) if early_stop_evaluations else int(stable_pass)
        base_cs = checkpoint_score(summary)
        score = (
            float(both_passed),
            float(stable_pass),
            float(streak_tier),
            combined_sr,
            base_cs[1],
            base_cs[2],
            -combined_press,
            base_cs[4],
        )
        if score > best_score:
            best_score = score
            runner.save(str(log_dir / "model_best.pt"), infos={"validation": summary, "stage": tag})
        env.fixed_spawns = None
        env.reset()
        if hasattr(runner.logger, "cur_reward_sum") and runner.logger.cur_reward_sum is not None:
            runner.logger.cur_reward_sum.zero_()
            runner.logger.cur_episode_length.zero_()
        return summary

    if args.eval_start_iteration <= 0:
        validate("initial")
    remaining = max_iterations
    completed = 0
    while remaining > 0:
        count = min(args.eval_interval, remaining)
        runner.learn(num_learning_iterations=count, init_at_random_ep_len=False)
        remaining -= count
        completed += count
        if completed >= args.eval_start_iteration or remaining == 0:
            validate(f"ppo_{completed}")
        if early_stop_evaluations and passing_evaluations >= early_stop_evaluations:
            break
        if remaining > 0:
            runner.current_learning_iteration += 1

    final_ckpt = log_dir / "model_final.pt"
    runner.save(str(final_ckpt), infos={"stage": "final"})
    checkpoints = sorted(log_dir.glob("model_*.pt"))
    selected = log_dir / "model_best.pt" if (log_dir / "model_best.pt").is_file() else final_ckpt
    write_artifact_inventory(
        log_dir / "artifacts.yaml",
        training_config=training_config,
        checkpoints=checkpoints,
        selected_checkpoint=selected,
    )
    (log_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "completed_iterations": completed,
                "max_iterations": max_iterations,
                "early_stopped": remaining > 0,
                "passing_evaluations": passing_evaluations,
                "best_success_rate": best_score[-5],
                "best_press_mm": -best_score[-2],
                "selected_checkpoint": selected.name,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    gs.destroy()
    print(f"Completed {completed} iterations: best strict success_rate={best_score[-5]:.3f}; {selected}", flush=True)


if __name__ == "__main__":
    main()
