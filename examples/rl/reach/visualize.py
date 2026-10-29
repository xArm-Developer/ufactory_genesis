"""Real-time paced viewer for the 5 mm reach policy.

Steps the simulation at wall-clock 1:1 with the control rate (20 ms per step),
so the interactive viewer shows continuous servo motion instead of fast-forward
hops. Closes cleanly when the viewer window is closed.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from .runtime import (
    DEFAULT_RECIPE,
    configure_runtime,
    patch_pyglet_screen,
    require_checkpoint,
    require_rsl_runtime,
    resolve_reach_robot,
)

# GLX is only for the interactive command. Importing this module from tests must
# leave the headless platform alone, or later Genesis teardowns lose the context.
if __name__ == "__main__":
    configure_runtime(viewer=True)
else:
    configure_runtime()
    patch_pyglet_screen()
require_rsl_runtime()

import torch
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from .realtime_env import RealtimeReachEnv
from .realtime_viewer import RealtimeClock, shutdown, start_viewer, warmup
from ufactory.config import load_runtime_config
from ufactory.robots.paths import robot_visual_glb_urdf
from ufactory.simulation.compat import require_genesis_runtime
from ufactory.training import (
    REACH_RL_ROBOTS,
    build_reach_task_configs,
    validate_and_load_rsl_checkpoint,
)
from ufactory.visualization.glb import glb_pbr_surfaces, glb_view_surface


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time viewer for the 5 mm reach policy")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--robot", default=None, choices=REACH_RL_ROBOTS)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("-b", "--num-envs", type=int, default=1, help="parallel environments shown in the viewer")
    parser.add_argument("--seed", type=int, default=17002)
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")

    checkpoint = require_checkpoint(args.checkpoint)
    config_path = checkpoint.parent / "config.yaml"
    robot = resolve_reach_robot(args.robot, config_path)
    runtime_config = load_runtime_config(robot, task="reach", config_path=args.runtime_config)
    expected_env, _reward, robot_cfg = build_reach_task_configs(
        robot, recipe_path=args.recipe, runtime_config_path=args.runtime_config
    )
    _loaded, artifact, _manifest = validate_and_load_rsl_checkpoint(
        checkpoint,
        config_path,
        map_location="cpu",
        expected_task="reach",
        expected_robot_key=runtime_config.robot.key,
        expected_runtime_env=expected_env,
    )
    env_cfg = dict(artifact["env"])
    env_cfg["num_envs"] = args.num_envs
    robot_cfg["urdf_path"] = robot_visual_glb_urdf(runtime_config.robot.key)

    require_genesis_runtime(gs)
    completed = 0
    successes = 0
    env = None
    try:
        with glb_pbr_surfaces():
            print("Preparing GLB simulation and policy; the window opens when the scene is ready.", flush=True)
            gs.init(
                backend=gs.gpu,
                precision="32",
                logging_level="warning",
                seed=args.seed,
                performance_mode=bool(env_cfg.get("genesis_performance_mode", False)),
            )
            env = RealtimeReachEnv(
                env_cfg=env_cfg,
                reward_cfg=dict(artifact["reward"]),
                robot_cfg=robot_cfg,
                robot_surface=glb_view_surface(),
            )
            train_cfg = copy.deepcopy(artifact["train"])
            train_cfg["runner"]["max_iterations"] = 0
            train_cfg["runner"]["resume"] = False
            runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=gs.device)
            runner.alg.load(_loaded, None, strict=True)
            policy = runner.get_inference_policy(device=gs.device)

            env.reset()
            warmup(env, policy)
            observation = env.get_observations()
            viewer = start_viewer(env)
            clock = RealtimeClock(env.ctrl_dt)
            print(
                f"real-time GLB viewer: environments={args.num_envs}; close the window (or Ctrl+C) to stop", flush=True
            )
            with torch.inference_mode():
                while viewer.is_alive():
                    observation, _reward, dones, infos = env.step(policy(observation))
                    viewer.update(auto_refresh=True)
                    if bool(dones.any()):
                        successes += int(infos["reach_done_success"].sum())
                        for env_idx, distance, ok in zip(
                            infos["reach_done_envs_idx"].tolist(),
                            infos["reach_done_distance_m"].tolist(),
                            infos["reach_done_success"].tolist(),
                        ):
                            completed += 1
                            print(
                                f"episode {completed:4d} env={env_idx} success={int(ok)} distance={distance * 1000:.2f} mm",
                                flush=True,
                            )
                    clock.wait()
    except KeyboardInterrupt:
        pass
    finally:
        shutdown(env)
        print(f"total episodes={completed} success_rate={successes / max(1, completed):.3f}", flush=True)


if __name__ == "__main__":
    main()
