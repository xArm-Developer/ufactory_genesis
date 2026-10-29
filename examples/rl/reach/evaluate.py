"""Evaluate a bare-arm UFACTORY reach checkpoint."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

from .runtime import DEFAULT_RECIPE, configure_runtime, require_checkpoint, require_rsl_runtime, resolve_reach_robot

configure_runtime()
require_rsl_runtime()

from rsl_rl.runners import OnPolicyRunner

import genesis as gs
import torch
from ufactory.config import load_runtime_config
from ufactory.robots.paths import robot_urdf
from ufactory.simulation.compat import require_genesis_runtime
from ufactory.training import (
    REACH_RL_ROBOTS,
    build_reach_task_configs,
    validate_and_load_rsl_checkpoint,
)
from ufactory.training.reach_evaluation import evaluate_fixed_targets, sample_reach_targets, summarize_reach_reports

from .env import ReachEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a bare-arm UFACTORY reach checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--robot", default=None, choices=REACH_RL_ROBOTS)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("-B", "--num-envs", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17002, help="Fixed target and simulation seed")
    parser.add_argument("--min-success-rate", type=float)
    parser.add_argument("--summary-json", type=Path, help="Save summary, every episode, and artifact hashes")
    parser.add_argument(
        "--quiet-episodes", action="store_true", help="Only print the summary; JSON retains every episode"
    )
    parser.add_argument(
        "--max-mean-distance-m",
        type=float,
        default=None,
        help="Exit with status 1 when the mean terminal distance exceeds this value",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if args.max_mean_distance_m is not None and (
        not math.isfinite(args.max_mean_distance_m) or args.max_mean_distance_m < 0.0
    ):
        parser.error("--max-mean-distance-m must be finite and non-negative")
    if args.min_success_rate is not None and not 0.0 <= args.min_success_rate <= 1.0:
        parser.error("--min-success-rate must be between 0 and 1")

    checkpoint = require_checkpoint(args.checkpoint)
    config_path = checkpoint.parent / "config.yaml"
    if args.summary_json is not None and args.summary_json.exists():
        raise FileExistsError(f"evaluation report already exists: {args.summary_json}")
    robot = resolve_reach_robot(args.robot, config_path)
    runtime_config = load_runtime_config(robot, task="reach", config_path=args.runtime_config)
    expected_env, _reward, _robot = build_reach_task_configs(
        robot,
        recipe_path=args.recipe,
        runtime_config_path=args.runtime_config,
    )
    _loaded, artifact, manifest = validate_and_load_rsl_checkpoint(
        checkpoint,
        config_path,
        map_location="cpu",
        expected_task="reach",
        expected_robot_key=runtime_config.robot.key,
        expected_runtime_env=expected_env,
    )
    env_cfg = deepcopy(artifact["env"])
    reward_cfg = deepcopy(artifact["reward"])
    robot_cfg = deepcopy(artifact["robot"])
    train_cfg = deepcopy(artifact["train"])
    env_cfg["num_envs"] = int(args.num_envs)
    robot_cfg["urdf_path"] = robot_urdf(runtime_config.robot.key, runtime_config.robot.urdf)
    if args.seed is not None:
        train_cfg["seed"] = int(args.seed)
    train_cfg["runner"]["max_iterations"] = 0
    train_cfg["runner"]["resume"] = False
    performance_mode = bool(env_cfg.get("genesis_performance_mode", False))

    require_genesis_runtime(gs)
    gs.init(
        backend=gs.gpu,
        precision="32",
        logging_level="warning",
        seed=int(train_cfg["seed"]),
        performance_mode=performance_mode,
    )
    env = ReachEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=False,
    )
    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=gs.device)
    runner.alg.load(_loaded, None, strict=True)
    policy = runner.get_inference_policy(device=gs.device)

    targets = sample_reach_targets(env_cfg["target_pos_lower"], env_cfg["target_pos_upper"], args.episodes, args.seed)
    with torch.inference_mode():
        reports = evaluate_fixed_targets(env, policy, targets)
    gs.destroy()
    summary = summarize_reach_reports(reports)
    passed = (args.min_success_rate is None or summary["success_rate"] >= args.min_success_rate) and (
        args.max_mean_distance_m is None or summary["mean_m"] <= args.max_mean_distance_m
    )
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_json.open("x", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "schema_version": 1,
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": manifest.checkpoint_sha256,
                        "config_sha256": artifact["config_sha256"],
                        "urdf_sha256": hashlib.sha256(Path(robot_cfg["urdf_path"]).read_bytes()).hexdigest(),
                        "target_sha256": hashlib.sha256(targets.numpy().tobytes()).hexdigest(),
                        "seed": args.seed,
                        "thresholds": {
                            "min_success_rate": args.min_success_rate,
                            "max_mean_distance_m": args.max_mean_distance_m,
                        },
                        "passed": passed,
                        "summary": summary,
                        "episodes": reports,
                    },
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            )
    if not args.quiet_episodes:
        for report in reports:
            print(
                f"target={report['index']:04d} success={int(report['success'])} "
                f"distance={report['terminal_distance_m'] * 1000:.1f} mm time={report['elapsed_s']:.2f} s"
            )
    print(
        f"episodes={int(summary['episodes'])} "
        f"success={summary['success_rate']:.3f} "
        f"mean={summary['mean_m'] * 1000:.1f} mm "
        f"median={summary['median_m'] * 1000:.1f} mm "
        f"p90={summary['p90_m'] * 1000:.1f} mm "
        f"p99={summary['p99_m'] * 1000:.1f} mm"
    )
    if not passed:
        print("Checkpoint failed the requested success rate or mean terminal distance threshold")
        sys.exit(1)


if __name__ == "__main__":
    main()
