"""Resolved task configuration builders shared by RL entry points and tests."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from ufactory.config import ConfigError, load_runtime_config
from ufactory.robots.paths import robot_urdf
from ufactory.robots.registry import robot_cli_choices
from ufactory.training.reach_config import REACH_ENV_VERSION, REACH_SUCCESS_DISTANCE_M, effective_max_joint_delta_rad
from ufactory.training.recipe import load_training_recipe
from ufactory.training.pick_place_config import (
    PICK_PLACE_CONTROL,
    PICK_PLACE_ENV_VERSION,
    PICK_PLACE_NUM_ACTIONS,
    PICK_PLACE_NUM_OBS,
    SPAWN_XY_LOWER,
    SPAWN_XY_UPPER,
)

PICK_PLACE_RL_ROBOTS = ("xarm6", "xarm6_1305")
REACH_RL_ROBOTS = tuple(robot_cli_choices())
# RL keeps the post-solve no-slip PGS pass off (noslip_iterations=5 slows a 512-env
# batch ~3.5x on an RTX 5090). Tangential contact stability is handled directly by
# the elliptic Signorini Newton solver configuration.
PICK_PLACE_RL_NOSLIP_ITERATIONS = 0


_PICK_PLACE_RUNTIME_OVERRIDES = frozenset(
    {
        "task.parameters.control_dt_s",
        "task.parameters.workspace_lower_m",
        "task.parameters.workspace_upper_m",
        "task.parameters.object_mass_kg",
        "gripper.damping",
        "gripper.frictionloss",
        "simulation.substeps",
        "simulation.constraint_solver",
        "simulation.solver_iterations",
        "simulation.friction_cone",
        "simulation.contact_resolution",
        "simulation.constraint_time_constant_s",
        "simulation.use_gjk_collision",
        "simulation.g2.speed_percent",
        "simulation.g2.force_percent",
    }
)
_PICK_PLACE_RECIPE_FIELDS = frozenset(
    {
        "task_name",
        "env_version",
        "num_envs",
        "num_obs",
        "num_actions",
        "control",
        "genesis_performance_mode",
        "high_fidelity",
        "action_scale",
        "action_clip",
        "gap_action_scale",
        "setpoint_leash_m",
        "pre_grasp_gap_m",
        "close_floor_gap_m",
        "reward_gamma",
        "place_clearance_m",
        "release_max_height_m",
        "press_limit_m",
        "near_table_m",
        "max_near_table_descent_m_s",
        "table_overload_margin_n",
        "place_success_dist_m",
        "drag_limit_m",
        "tilt_limit_deg",
        "success_hold_steps",
    }
)


def _validate_pick_place_runtime_overrides(config_path: str | Path | None) -> None:
    if config_path is None:
        return
    import yaml

    try:
        overlay = yaml.safe_load(Path(config_path).expanduser().read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read pick-place RL runtime configuration: {exc}") from exc
    if not isinstance(overlay, dict) or overlay.get("schema_version") != 1:
        raise ConfigError("pick-place RL runtime configuration requires schema_version 1")

    def validate(mapping: dict[str, Any], prefix: str = "") -> None:
        for key, value in mapping.items():
            if not prefix and key == "schema_version":
                continue
            field = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                if not any(allowed.startswith(field + ".") for allowed in _PICK_PLACE_RUNTIME_OVERRIDES):
                    raise ConfigError(f"unsupported pick-place RL runtime override: {field}")
                validate(value, field)
            elif field not in _PICK_PLACE_RUNTIME_OVERRIDES:
                raise ConfigError(f"unsupported pick-place RL runtime override: {field}")

    validate(overlay)


def load_pick_place_rl_runtime_config(robot: str, *, config_path: str | Path | None = None) -> Any:
    """Resolve the runtime configuration shared by pick-place RL training and evaluation."""

    _validate_pick_place_runtime_overrides(config_path)
    return load_runtime_config(
        robot,
        task="pick_place",
        config_path=config_path,
        overrides={
            "simulation.noslip_iterations": PICK_PLACE_RL_NOSLIP_ITERATIONS,
        },
    )


def build_train_config(
    recipe_path: str | Path,
    *,
    experiment_name: str,
    max_iterations: int,
) -> dict[str, Any]:
    """Resolve PPO settings with command-line run settings taking precedence."""

    train_cfg = deepcopy(load_training_recipe(recipe_path)["train"])
    train_cfg["runner"]["experiment_name"] = experiment_name
    train_cfg["runner"]["max_iterations"] = int(max_iterations)
    return train_cfg


def build_pick_place_task_configs(
    robot: str,
    *,
    recipe_path: str | Path,
    runtime_config_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve the retained xArm6 + Gripper G2 pick-place RL task."""

    from ufactory.simulation import G2_PHYSICS_PROFILE
    from ufactory.simulation.g2_model import g2_physics_summary

    if robot not in PICK_PLACE_RL_ROBOTS:
        raise ValueError("the pick-place RL environment supports only xArm6 + Gripper G2")
    config = load_pick_place_rl_runtime_config(robot, config_path=runtime_config_path)
    if config.gripper is None or config.gripper.adapter != "g2":
        raise ValueError("the pick-place RL environment requires Gripper G2")
    params = config.task.parameters
    recipe = load_training_recipe(recipe_path)
    env_cfg = deepcopy(recipe["environment"])
    unsupported = sorted(set(env_cfg) - _PICK_PLACE_RECIPE_FIELDS)
    if unsupported:
        raise ValueError(f"unsupported pick-place RL environment fields: {', '.join(unsupported)}")
    if env_cfg.get("task_name", "pick_place") != "pick_place" or (
        env_cfg.get("env_version"),
        env_cfg.get("num_obs"),
        env_cfg.get("num_actions"),
        env_cfg.get("control"),
    ) != (PICK_PLACE_ENV_VERSION, PICK_PLACE_NUM_OBS, PICK_PLACE_NUM_ACTIONS, PICK_PLACE_CONTROL):
        raise ValueError("pick-place v3 requires 70 observations and 4 continuous command increments")
    env_cfg["task_name"] = "pick_place"
    if float(env_cfg.get("reward_gamma", 0.995)) != float(recipe["train"]["algorithm"]["gamma"]):
        raise ValueError("pick-place reward_gamma must match PPO gamma for potential shaping")
    for key in ("place_clearance_m", "release_max_height_m", "press_limit_m"):
        if float(env_cfg.get(key, 0.0)) <= 0.0:
            raise ValueError(f"pick-place v3 requires positive {key}")
    env_cfg.update(
        {
            "runtime_config_sha256": config.sha256,
            "physics_profile": G2_PHYSICS_PROFILE,
            "g2": {
                "speed_percent": config.simulation.g2.speed_percent,
                "force_percent": config.simulation.g2.force_percent,
            },
            "g2_physics": g2_physics_summary(simulation=config.simulation, rate_hz=1 / float(params["control_dt_s"])),
            "episode_length_s": float(params["episode_length_s"]),
            "ctrl_dt": float(params["control_dt_s"]),
            "spawn_xy_lower": list(SPAWN_XY_LOWER),
            "spawn_xy_upper": list(SPAWN_XY_UPPER),
            "workspace_lower": list(params["workspace_lower_m"]),
            "workspace_upper": list(params["workspace_upper_m"]),
            "substeps": int(config.simulation.substeps),
            "constraint_solver": config.simulation.constraint_solver,
            "solver_iterations": config.simulation.solver_iterations,
            "noslip_iterations": config.simulation.noslip_iterations,
            "friction_cone": config.simulation.friction_cone,
            "contact_resolution": config.simulation.contact_resolution,
            "constraint_time_constant_s": config.simulation.constraint_time_constant_s,
            "use_gjk_collision": config.simulation.use_gjk_collision,
            "obj_mass_kg": float(params["object_mass_kg"]),
        }
    )
    robot_cfg = {
        "gripper_damping": config.gripper.damping,
        "gripper_frictionloss": config.gripper.frictionloss,
    }
    return env_cfg, deepcopy(recipe["reward"]), robot_cfg


def build_reach_task_configs(
    robot: str,
    *,
    recipe_path: str | Path,
    runtime_config_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve the bare-arm reach task for any supported UFACTORY robot."""

    if robot not in REACH_RL_ROBOTS:
        raise ValueError(f"unsupported reach RL robot {robot!r}; expected one of {REACH_RL_ROBOTS}")
    config = load_runtime_config(robot, task="reach", config_path=runtime_config_path)
    params = config.task.parameters
    if float(params["success_distance_m"]) != REACH_SUCCESS_DISTANCE_M:
        raise ValueError("reach success_distance_m must be 0.005 (5 mm)")
    recipe = load_training_recipe(recipe_path)
    env_cfg = deepcopy(recipe["environment"])
    ctrl_dt = float(params["control_dt_s"])
    action_scale = float(params["action_scale_rad"])
    action_clip = float(params["action_clip"])
    servo_speed = float(params["servo_speed_rad_s"])
    max_joint_delta = effective_max_joint_delta_rad(
        action_scale=action_scale,
        action_clip=action_clip,
        ctrl_dt=ctrl_dt,
        servo_speed_rad_s=servo_speed,
    )
    num_obs = int(config.robot.dof) * 3 + 6
    num_actions = int(config.robot.dof)
    if "num_obs" in env_cfg and int(env_cfg["num_obs"]) != num_obs:
        raise ValueError(f"reach recipe num_obs must be {num_obs}, got {env_cfg['num_obs']}")
    if "num_actions" in env_cfg and int(env_cfg["num_actions"]) != num_actions:
        raise ValueError(f"reach recipe num_actions must be {num_actions}, got {env_cfg['num_actions']}")
    env_cfg.update(
        {
            "task_name": "reach",
            "env_version": REACH_ENV_VERSION,
            "num_obs": num_obs,
            "num_actions": num_actions,
            "runtime_config_sha256": config.sha256,
            "episode_length_s": float(params["episode_length_s"]),
            "ctrl_dt": ctrl_dt,
            "success_distance_m": float(params["success_distance_m"]),
            "action_scale": action_scale,
            "action_clip": action_clip,
            "servo_speed_rad_s": servo_speed,
            "max_joint_delta_rad": max_joint_delta,
            "target_pos_lower": list(params["target_position_lower_m"]),
            "target_pos_upper": list(params["target_position_upper_m"]),
            "substeps": int(config.simulation.substeps),
            "constraint_solver": config.simulation.constraint_solver,
            "solver_iterations": config.simulation.solver_iterations,
            "noslip_iterations": config.simulation.noslip_iterations,
            "friction_cone": config.simulation.friction_cone,
            "contact_resolution": config.simulation.contact_resolution,
            "constraint_time_constant_s": config.simulation.constraint_time_constant_s,
            "use_gjk_collision": config.simulation.use_gjk_collision,
        }
    )
    robot_cfg = {
        "urdf_path": robot_urdf(config.robot.key, config.robot.urdf),
        "base_pos": [0.0, 0.0, 0.0],
        "ee_link_name": config.robot.ee_link,
        "joint_names": list(config.robot.joint_names),
        "default_qpos": list(config.arm.default_qpos_rad),
        "kp": list(config.arm.kp),
        "kv": list(config.arm.kv),
        "force_lower": list(config.arm.force_lower_nm),
        "force_upper": list(config.arm.force_upper_nm),
    }
    return env_cfg, deepcopy(recipe["reward"]), robot_cfg
