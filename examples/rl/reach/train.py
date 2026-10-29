"""Train the bare-arm UFACTORY reach task with PPO."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from .runtime import (
    DEFAULT_RECIPE,
    OUTPUT_ROOT,
    configure_runtime,
    require_checkpoint,
    require_rsl_runtime,
    resolve_reach_robot,
)

configure_runtime()
require_rsl_runtime()

import genesis as gs
from ufactory.config import load_runtime_config
from ufactory.simulation.compat import require_genesis_runtime
from ufactory.training import (
    REACH_RL_ROBOTS,
    build_reach_task_configs,
    build_train_config,
    load_training_config,
    load_training_recipe,
    validate_and_load_rsl_checkpoint,
    write_artifact_inventory,
    write_run_provenance,
    write_training_config,
)

from ufactory.training.runner import ArtifactOnPolicyRunner
from ufactory.training.runtime import prepare_log_dir

from .env import ReachEnv

EXECUTOR_ACTION_CONTRACT = "normalized_joint_position_delta"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the bare-arm UFACTORY reach task")
    parser.add_argument("--robot", default=None, choices=REACH_RL_ROBOTS)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("-e", "--exp-name", default=None)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("-v", "--vis", action="store_true")
    parser.add_argument("-B", "--num-envs", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    if args.num_envs is not None and args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if args.max_iterations is not None and args.max_iterations < 1:
        parser.error("--max-iterations must be positive")

    robot = resolve_reach_robot(
        args.robot,
        None if args.resume is None else args.resume.parent / "config.yaml",
    )

    recipe = load_training_recipe(args.recipe)
    num_envs = int(args.num_envs if args.num_envs is not None else recipe["environment"]["num_envs"])
    max_iterations = int(
        args.max_iterations if args.max_iterations is not None else recipe["train"]["runner"]["max_iterations"]
    )
    runtime_config = load_runtime_config(robot, task="reach", config_path=args.runtime_config)
    exp_name = args.exp_name or f"{runtime_config.robot.robot_name}-reach"
    env_cfg, reward_cfg, robot_cfg = build_reach_task_configs(
        robot,
        recipe_path=args.recipe,
        runtime_config_path=args.runtime_config,
    )
    env_cfg["num_envs"] = num_envs
    performance_mode = bool(recipe["environment"].get("genesis_performance_mode", False))
    randomize_initial_episode_length = bool(recipe["environment"].get("randomize_initial_episode_length", False))
    env_cfg["genesis_performance_mode"] = performance_mode
    env_cfg["randomize_initial_episode_length"] = randomize_initial_episode_length
    train_cfg = build_train_config(
        args.recipe,
        experiment_name=exp_name,
        max_iterations=max_iterations,
    )
    if args.seed is not None:
        train_cfg["seed"] = int(args.seed)
    train_cfg["runner"]["resume"] = args.resume is not None
    train_cfg["runner"]["resume_path"] = None if args.resume is None else str(args.resume)

    resume_state = None
    if args.resume is not None:
        args.resume = require_checkpoint(args.resume)
        resume_state, resume_config, _resume_manifest = validate_and_load_rsl_checkpoint(
            args.resume,
            args.resume.parent / "config.yaml",
            map_location="cpu",
            expected_task="reach",
            expected_robot_key=runtime_config.robot.key,
            expected_runtime_env=env_cfg,
        )

        if int(resume_config["env"]["num_envs"]) != num_envs:
            parser.error("resume requires the original --num-envs; the checkpoint restores every environment state")

    log_dir = args.log_dir or OUTPUT_ROOT / exp_name
    prepare_log_dir(log_dir, overwrite=args.overwrite)
    write_training_config(
        log_dir / "config.yaml",
        task="reach",
        robot_key=runtime_config.robot.key,
        env=env_cfg,
        reward=reward_cfg,
        robot=robot_cfg,
        train=train_cfg,
    )
    artifact = load_training_config(log_dir / "config.yaml")
    repo_root = Path(__file__).resolve().parents[3]
    provenance_sources = [
        Path(__file__),
        Path(__file__).with_name("env.py"),
        Path(__file__).with_name("runtime.py"),
        repo_root / "pyproject.toml",
        repo_root / "uv.lock",
        args.recipe,
        repo_root / "ufactory/training/logic/__init__.py",
        repo_root / "ufactory/training/logic/common.py",
        repo_root / "ufactory/training/logic/reach.py",
        repo_root / "ufactory/training/tasks.py",
        repo_root / "ufactory/training/artifacts.py",
        repo_root / "ufactory/training/runner.py",
        repo_root / "ufactory/training/runtime.py",
        repo_root / "ufactory/training/reach_evaluation.py",
        repo_root / "ufactory/training/reach_config.py",
        repo_root / "ufactory/config/loader.py",
        Path(__file__).with_name("evaluate.py"),
        Path(robot_cfg["urdf_path"]),
        repo_root / "ufactory/simulation/__init__.py",
        repo_root / "ufactory/simulation/compat.py",
        repo_root / "ufactory/simulation/physics.py",
    ]
    provenance_sources.extend(repo_root / source for source in runtime_config.sources)
    for source in set(path.resolve() for path in provenance_sources):
        relative = source.relative_to(repo_root) if source.is_relative_to(repo_root) else Path("recipe.yaml")
        target = log_dir / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    write_run_provenance(
        log_dir / "run_provenance.json",
        training_config=artifact,
        source_paths=provenance_sources,
    )

    require_genesis_runtime(gs)
    gs.init(
        backend=gs.gpu,
        precision="32",
        logging_level="warning",
        seed=int(train_cfg["seed"]),
        performance_mode=performance_mode,
    )
    env = ReachEnv(env_cfg=env_cfg, reward_cfg=reward_cfg, robot_cfg=robot_cfg, show_viewer=args.vis)
    runner = ArtifactOnPolicyRunner(
        env, train_cfg, log_dir=str(log_dir), device=gs.device, executor_action_contract=EXECUTOR_ACTION_CONTRACT
    )
    runner.checkpoint_training_config = artifact
    runner.checkpoint_env = env
    if resume_state is not None:
        print(f"Resuming full training state from {args.resume}")
        load_iteration = runner.alg.load(resume_state, None, strict=True)
        if load_iteration:
            runner.current_learning_iteration = int(resume_state["iter"])
        resume_infos = resume_state.get("infos")
        if isinstance(resume_infos, dict) and "ufactory_env_state" in resume_infos:
            env.load_training_state_dict(resume_infos["ufactory_env_state"])
        runner.current_learning_iteration = int(runner.current_learning_iteration) + 1
    status = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "stop_reason": "running"}
    status_path = log_dir / "training_status.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    try:
        runner.learn(
            num_learning_iterations=max_iterations,
            init_at_random_ep_len=randomize_initial_episode_length and resume_state is None,
        )
        status["stop_reason"] = "completed_iterations"
    except BaseException as exc:
        status["stop_reason"] = type(exc).__name__
        status["error"] = str(exc)
        raise
    finally:
        status["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        status["last_iteration"] = int(runner.current_learning_iteration)
        status_path.write_text(json.dumps(status, indent=2) + "\n")
    write_artifact_inventory(
        log_dir / "artifacts.yaml",
        training_config=artifact,
        checkpoints=sorted(log_dir.glob("model_*.pt")),
    )
    print(f"Training artifacts written to {log_dir}")


if __name__ == "__main__":
    main()
