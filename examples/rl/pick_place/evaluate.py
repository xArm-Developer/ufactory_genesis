"""Strict tabletop pick-and-place evaluation for heuristic expert, zero policy, or PPO checkpoints."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

from .runtime import DEFAULT_RECIPE, configure_runtime, require_backend, require_checkpoint, require_rsl_runtime

configure_runtime()
require_rsl_runtime()

import genesis as gs
import numpy as np
from rsl_rl.runners import OnPolicyRunner
import torch

from ufactory.robots.paths import robot_urdf
from ufactory.training.pick_place_config import PICK_PLACE_ROBOT, PICK_PLACE_URDF
from ufactory.simulation.compat import require_genesis_runtime
from ufactory.training import (
    PICK_PLACE_RL_ROBOTS,
    build_pick_place_task_configs,
    load_pick_place_rl_runtime_config,
    validate_and_load_rsl_checkpoint,
)

from .env import SUBSTEPS, XArm6PickPlaceEnv
from .expert import PickPlaceExpert
from .metrics import SPAWN_XY_LOWER, SPAWN_XY_UPPER


def make_spawns(
    episodes: int,
    seed: int = 17000,
    lower: Sequence[float] = SPAWN_XY_LOWER,
    upper: Sequence[float] = SPAWN_XY_UPPER,
) -> np.ndarray:
    """Use a dedicated CPU generator so evaluation spawns do not depend on batch size or GPU."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    lo = torch.tensor(lower, dtype=torch.float32, device="cpu")
    hi = torch.tensor(upper, dtype=torch.float32, device="cpu")
    return (lo + torch.rand(int(episodes), 2, generator=generator, device="cpu") * (hi - lo)).numpy()


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise RuntimeError("evaluation finished no episodes")
    summary: dict[str, Any] = {"episodes": len(records)}
    for key in (
        "success",
        "drag_failed",
        "tilt_failed",
        "dropped",
        "press_failed",
        "high_release",
        "collision_failed",
        "timeout",
        "release_confirmed",
        "ever_carried",
        "ever_lifted_clear",
        "touchdown_seen",
    ):
        count = sum(int(bool(r[key])) for r in records)
        summary[key] = count
        summary[f"{key}_rate"] = count / len(records)
    for key in (
        "max_drag_mm",
        "max_tilt_deg",
        "target_error_mm",
        "steps",
        "press_mm",
        "release_height_mm",
        "max_table_force_n",
        "max_place_table_force_n",
        "max_held_table_force_n",
        "max_robot_table_force_n",
    ):
        values = [float(r[key]) for r in records]
        summary[key] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }
    return summary


def checkpoint_score(summary: dict[str, Any]) -> tuple[float, float, float, float, float]:
    """Prefer strict success, then low release, then target touchdown, then less pressing, then landing error."""
    no_high = 1.0 - float(summary.get("high_release_rate", 0.0))
    low_release = float(summary.get("release_confirmed_rate", 0.0)) * no_high
    low_touchdown_or_success = max(float(summary["success_rate"]), float(summary.get("touchdown_seen_rate", 1.0)))
    target_touchdown = low_touchdown_or_success * no_high * float(summary["target_error_mm"]["mean"] <= 20.0)
    return (
        float(summary["success_rate"]),
        low_release,
        target_touchdown,
        -float(summary["press_mm"]["max"]),
        -float(summary["target_error_mm"]["mean"]),
    )


def acceptance_passed(summary: dict[str, Any], *, min_success_rate: float = 0.98, max_press_mm: float = 1.0) -> bool:
    return bool(summary["success_rate"] >= min_success_rate and summary["press_mm"]["max"] <= max_press_mm)


def evaluate(
    env: XArm6PickPlaceEnv,
    policy: Callable[[Any], torch.Tensor] | None,
    episodes: int = 64,
    seed: int = 17000,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spawns = make_spawns(
        episodes,
        seed=seed,
        lower=env.cfg.get("spawn_xy_lower", SPAWN_XY_LOWER),
        upper=env.cfg.get("spawn_xy_upper", SPAWN_XY_UPPER),
    )
    obs, _ = env.set_evaluation_spawns(spawns)
    if policy is None:
        policy = PickPlaceExpert(env)
    records: dict[int, dict[str, Any]] = {}
    start = time.monotonic()
    start_steps = env.total_env_steps
    with torch.inference_mode():
        while len(records) < episodes:
            actions = policy(obs)
            obs, _, done, info = env.step(actions)
            idx = done.nonzero(as_tuple=False).squeeze(-1)
            if len(idx):
                terminal = {k: v[idx].cpu().tolist() for k, v in info["terminal"].items()}
                for j in range(len(idx)):
                    record = {k: v[j] for k, v in terminal.items()}
                    sid = int(record["scenario_id"])
                    if 0 <= sid < episodes and sid not in records:
                        records[sid] = record
    rows = [records[i] for i in range(episodes)]
    summary = summarize(rows)
    elapsed = max(1e-9, time.monotonic() - start)
    summary["seconds"] = elapsed
    summary["transitions_per_second"] = (env.total_env_steps - start_steps) / elapsed
    summary["seed"] = int(seed)
    summary["spawn_sha256"] = hashlib.sha256(spawns.tobytes()).hexdigest()
    summary["device"] = str(env.device)
    summary["env_version"] = int(env.cfg["env_version"])
    return summary, rows


def save_results(
    path: str | Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(summary)
    if extra_metadata:
        payload.update(extra_metadata)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    if rows:
        csv_rows = [{k: (json.dumps(v) if isinstance(v, list | dict) else v) for k, v in row.items()} for row in rows]
        with target.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert", action="store_true", help="Run the independent scripted physics baseline")
    parser.add_argument("--zero-policy", action="store_true", help="Run an all-zero policy baseline")
    parser.add_argument("--checkpoint", "--ckpt", dest="checkpoint", type=Path, help="Path to trained model_*.pt")
    parser.add_argument("--robot", default="xarm6", choices=PICK_PLACE_RL_ROBOTS)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("-B", "--num-envs", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17000)
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--substeps", type=int, default=SUBSTEPS)
    parser.add_argument("--min-success-rate", type=float)
    parser.add_argument("--max-target-error-mm", type=float)
    parser.add_argument("--max-press-mm", type=float, default=1.0)
    parser.add_argument("--summary-json", "--out", dest="summary_json", type=Path, help="Output JSON/CSV path")
    parser.add_argument("--quiet-episodes", action="store_true")
    args = parser.parse_args()

    mode_count = sum(bool(x) for x in (args.expert, args.zero_policy, args.checkpoint is not None))
    if mode_count != 1:
        parser.error("Specify exactly one of --expert, --zero-policy, or --checkpoint")
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.max_press_mm < 0 or (args.min_success_rate is not None and not 0 <= args.min_success_rate <= 1):
        parser.error("require --max-press-mm >= 0 and --min-success-rate in [0, 1]")

    runtime_config = load_pick_place_rl_runtime_config(args.robot, config_path=args.runtime_config)
    torch.set_num_threads(4)
    expected_env, expected_reward, expected_robot = build_pick_place_task_configs(
        args.robot,
        recipe_path=args.recipe,
        runtime_config_path=args.runtime_config,
    )

    loaded_state = None
    extra_meta: dict[str, Any] = {}
    if args.checkpoint is not None:
        checkpoint = require_checkpoint(args.checkpoint)
        config_path = checkpoint.parent / "config.yaml"
        loaded_state, artifact, manifest = validate_and_load_rsl_checkpoint(
            checkpoint,
            config_path,
            map_location="cpu",
            expected_task="pick_place",
            expected_robot_key=runtime_config.robot.key,
            expected_runtime_config_sha256=runtime_config.sha256,
            expected_runtime_env=expected_env,
        )
        env_cfg = deepcopy(artifact["env"])
        reward_cfg = deepcopy(artifact["reward"])
        robot_cfg = deepcopy(artifact["robot"])
        train_cfg = deepcopy(artifact["train"])
        extra_meta = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": manifest.checkpoint_sha256,
            "config_sha256": artifact["config_sha256"],
            "urdf_sha256": hashlib.sha256(Path(robot_urdf(PICK_PLACE_ROBOT, PICK_PLACE_URDF)).read_bytes()).hexdigest(),
        }
    else:
        env_cfg = deepcopy(expected_env)
        reward_cfg = deepcopy(expected_reward)
        robot_cfg = deepcopy(expected_robot)
        train_cfg = None

    env_cfg["num_envs"] = int(args.num_envs)
    env_cfg["substeps"] = int(args.substeps)

    require_genesis_runtime(gs)
    gs.init(
        backend=gs.cpu if args.backend == "cpu" else gs.gpu,
        precision="32",
        logging_level="warning",
        seed=int(args.seed),
        performance_mode=bool(env_cfg.get("genesis_performance_mode", False)),
    )
    require_backend(gs, args.backend)
    env = XArm6PickPlaceEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=False,
    )

    policy: Callable[[Any], torch.Tensor] | None
    if args.zero_policy:

        def _zero_policy(_obs: Any) -> torch.Tensor:
            return torch.zeros((env.num_envs, env.num_actions), device=env.device, dtype=gs.tc_float)

        policy = _zero_policy
    elif args.expert:
        policy = None
    else:
        assert train_cfg is not None and loaded_state is not None
        train_cfg["seed"] = int(args.seed)
        train_cfg["runner"]["max_iterations"] = 0
        train_cfg["runner"]["resume"] = False
        runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=gs.device)
        runner.alg.load(loaded_state, None, strict=True)
        actor = runner.get_inference_policy(device=gs.device)

        def _det_policy(obs: Any) -> torch.Tensor:
            return actor(obs, stochastic_output=False)

        policy = _det_policy

    summary, rows = evaluate(env, policy, episodes=args.episodes, seed=args.seed)
    gs.destroy()

    passed = acceptance_passed(
        summary, min_success_rate=args.min_success_rate or 0, max_press_mm=args.max_press_mm
    ) and (args.max_target_error_mm is None or summary["target_error_mm"]["max"] <= args.max_target_error_mm)
    summary["passed"] = passed
    if args.summary_json is not None:
        save_results(args.summary_json, summary, rows, extra_metadata=extra_meta)

    if not args.quiet_episodes:
        for row in rows:
            print(
                f"scenario={int(row['scenario_id']):04d} success={int(bool(row['success']))} "
                f"target_err={float(row['target_error_mm']):.2f}mm "
                f"drag={float(row['max_drag_mm']):.2f}mm tilt={float(row['max_tilt_deg']):.2f}deg "
                f"press={float(row['press_mm']):.3f}mm release={float(row['release_height_mm']):.2f}mm "
                f"steps={int(row['steps'])}"
            )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
