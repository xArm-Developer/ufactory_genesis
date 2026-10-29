"""Strict layered YAML loader for runtime configuration."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, cast

import yaml

from ufactory.config.assets import RepositoryAssetStore
from ufactory.config.models import (
    ArmControlProfile,
    G2_NOSLIP_ITERATIONS,
    GripperProfile,
    G2SimulationConfig,
    MotionConfig,
    ResolvedRuntimeConfig,
    RobotSpec,
    SimulationConfig,
    TaskProfile,
)
from ufactory.robots.registry import get_profile_key_for_robot_name
from ufactory.safety.models import CollisionExemption, SafetyPolicy


class ConfigError(ValueError):
    """A source configuration failed strict validation."""


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"failed to read configuration {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"configuration root must be a mapping: {path}")
    if raw.get("schema_version") != 1:
        raise ConfigError(f"{path}: schema_version must be 1")
    return cast(dict[str, Any], raw)


def _merge_strict(base: dict[str, Any], overlay: Mapping[str, Any], *, path: str = "") -> None:
    for key, value in overlay.items():
        if key == "schema_version":
            continue
        dotted = f"{path}.{key}" if path else str(key)
        if key not in base:
            raise ConfigError(f"unknown configuration field: {dotted}")
        original = base[key]
        if isinstance(original, dict):
            if not isinstance(value, Mapping):
                raise ConfigError(f"{dotted} must be a mapping")
            _merge_strict(original, value, path=dotted)
        else:
            base[key] = value


def _validate_finite(value: Any, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite(item, f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _validate_finite(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ConfigError(f"{path} must be finite")


def _tuple_floats(value: Any, *, field: str, size: int | None = None) -> tuple[float, ...]:
    if not isinstance(value, list | tuple):
        raise ConfigError(f"{field} must be a sequence")
    result = tuple(float(item) for item in value)
    if size is not None and len(result) != size:
        raise ConfigError(f"{field} expected {size} values, got {len(result)}")
    if not all(math.isfinite(item) for item in result):
        raise ConfigError(f"{field} must contain only finite values")
    return result


def _positive(value: Any, *, field: str, allow_zero: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number) or (number < 0.0 if allow_zero else number <= 0.0):
        relation = "non-negative" if allow_zero else "positive"
        raise ConfigError(f"{field} must be finite and {relation}")
    return number


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, frozenset | set):
        return sorted(_plain(item) for item in value)
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def _canonical_payload(config: ResolvedRuntimeConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, dict):
        return config
    payload = cast(dict[str, Any], _plain(config))
    payload.pop("sources", None)
    payload.pop("sha256", None)
    payload["robot"]["capabilities"] = sorted(config.robot.capabilities)
    if config.gripper is not None:
        payload["gripper"]["allowed_contact_links"] = sorted(config.gripper.allowed_contact_links)
    payload["task"]["parameters"] = dict(config.task.parameters)
    payload["task"]["allowed_contacts"] = dict(config.task.allowed_contacts)
    return payload


def config_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _build_resolved(data: dict[str, Any], sources: tuple[str, ...]) -> ResolvedRuntimeConfig:
    _validate_finite(data)
    robot_data = data["robot"]
    dof = int(robot_data["dof"])
    robot = RobotSpec(
        key=str(robot_data["key"]),
        robot_name=str(robot_data["robot_name"]),
        variant=str(robot_data["variant"]),
        dof=dof,
        joint_names=tuple(map(str, robot_data["joint_names"])),
        ee_link=str(robot_data["ee_link"]),
        assets_dir=str(robot_data["assets_dir"]),
        urdf=str(robot_data["urdf"]),
        capabilities=frozenset(map(str, robot_data["capabilities"])),
        adjacent_collision_pairs=tuple((str(pair[0]), str(pair[1])) for pair in robot_data["adjacent_collision_pairs"]),
        environment_contact_pairs=tuple(
            (str(pair[0]), str(pair[1])) for pair in robot_data["environment_contact_pairs"]
        ),
    )
    arm_data = data["arm"]
    arm = ArmControlProfile(
        default_qpos_rad=_tuple_floats(arm_data["default_qpos_rad"], field="arm.default_qpos_rad", size=dof),
        kp=_tuple_floats(arm_data["kp"], field="arm.kp", size=dof),
        kv=_tuple_floats(arm_data["kv"], field="arm.kv", size=dof),
        force_lower_nm=_tuple_floats(arm_data["force_lower_nm"], field="arm.force_lower_nm", size=dof),
        force_upper_nm=_tuple_floats(arm_data["force_upper_nm"], field="arm.force_upper_nm", size=dof),
        effort_limits_nm=_tuple_floats(arm_data["effort_limits_nm"], field="arm.effort_limits_nm", size=dof),
    )
    if any(lo >= hi for lo, hi in zip(arm.force_lower_nm, arm.force_upper_nm, strict=True)):
        raise ConfigError("each arm force lower bound must be below its upper bound")
    task_data = data["task"]
    task_name = str(task_data["name"])
    gripper_data = None if task_name == "reach" else data.get("gripper")
    gripper = None
    if gripper_data is not None:
        gripper = GripperProfile(
            adapter=str(gripper_data["adapter"]),
            drive_joint=str(gripper_data["drive_joint"]),
            all_joint_names=tuple(map(str, gripper_data["all_joint_names"])),
            finger_link_names=tuple(map(str, gripper_data["finger_link_names"])),
            open_drive=float(gripper_data["open_drive"]),
            closed_drive=float(gripper_data["closed_drive"]),
            open_gap_m=float(gripper_data["open_gap_m"]),
            closed_gap_m=float(gripper_data["closed_gap_m"]),
            kp=float(gripper_data["kp"]),
            kv=float(gripper_data["kv"]),
            force_lower_n=float(gripper_data["force_lower_n"]),
            force_upper_n=float(gripper_data["force_upper_n"]),
            damping=float(gripper_data["damping"]),
            frictionloss=float(gripper_data["frictionloss"]),
            real_command=bool(gripper_data["real_command"]),
            feedback=bool(gripper_data["feedback"]),
            closed_loop=bool(gripper_data["closed_loop"]),
            allowed_contact_links=frozenset(map(str, gripper_data["allowed_contact_links"])),
            tool_tip_offset_z_m=float(gripper_data["tool_tip_offset_z_m"]),
            finger_pad_below_center_m=float(gripper_data["finger_pad_below_center_m"]),
            finger_close_descent_m=float(gripper_data["finger_close_descent_m"]),
            grasp_table_clearance_m=float(gripper_data["grasp_table_clearance_m"]),
            grasp_height_extra_m=float(gripper_data["grasp_height_extra_m"]),
        )
    task_parameters = task_data["parameters"]
    if not isinstance(task_parameters, Mapping):
        raise ConfigError("task.parameters must be a mapping")
    if task_name in {"pick_place", "packaging_showcase"}:
        object_size = _tuple_floats(
            task_parameters["object_size_m"],
            field="task.parameters.object_size_m",
            size=3,
        )
        if any(value <= 0.0 for value in object_size):
            raise ConfigError("task.parameters.object_size_m values must be positive")
        _positive(task_parameters["object_mass_kg"], field="task.parameters.object_mass_kg")
        for field_name in ("fixed_object_position_m", "fixed_target_position_m", "default_ee_position_m"):
            _tuple_floats(
                task_parameters[field_name],
                field=f"task.parameters.{field_name}",
                size=3,
            )
    if task_name == "pick_place":
        if gripper is None:
            raise ConfigError("pick_place requires a configured gripper")
        _positive(task_parameters["episode_length_s"], field="task.parameters.episode_length_s")
        _positive(task_parameters["control_dt_s"], field="task.parameters.control_dt_s")
        ws_lower = _tuple_floats(
            task_parameters["workspace_lower_m"],
            field="task.parameters.workspace_lower_m",
            size=3,
        )
        ws_upper = _tuple_floats(
            task_parameters["workspace_upper_m"],
            field="task.parameters.workspace_upper_m",
            size=3,
        )
        if any(lower >= upper for lower, upper in zip(ws_lower, ws_upper, strict=True)):
            raise ConfigError("task.parameters.workspace_lower_m must be below workspace_upper_m on every axis")
    if task_name == "reach":
        _positive(task_parameters["episode_length_s"], field="task.parameters.episode_length_s")
        _positive(task_parameters["control_dt_s"], field="task.parameters.control_dt_s")
        _positive(task_parameters["success_distance_m"], field="task.parameters.success_distance_m")
        _positive(task_parameters["action_scale_rad"], field="task.parameters.action_scale_rad")
        _positive(task_parameters["action_clip"], field="task.parameters.action_clip")
        _positive(task_parameters["servo_speed_rad_s"], field="task.parameters.servo_speed_rad_s")
        target_lower = _tuple_floats(
            task_parameters["target_position_lower_m"],
            field="task.parameters.target_position_lower_m",
            size=3,
        )
        target_upper = _tuple_floats(
            task_parameters["target_position_upper_m"],
            field="task.parameters.target_position_upper_m",
            size=3,
        )
        if any(lower >= upper for lower, upper in zip(target_lower, target_upper, strict=True)):
            raise ConfigError(
                "task.parameters.target_position_lower_m must be below target_position_upper_m on every axis"
            )
    if task_name == "packaging_showcase":
        table_size = _tuple_floats(task_parameters["table_size_m"], field="task.parameters.table_size_m", size=3)
        box_size = _tuple_floats(task_parameters["box_outer_size_m"], field="task.parameters.box_outer_size_m", size=3)
        if any(value <= 0.0 for value in (*table_size, *box_size)):
            raise ConfigError("packaging table and box dimensions must be positive")
        table_center = _tuple_floats(task_parameters["table_center_m"], field="task.parameters.table_center_m", size=3)
        box_center = _tuple_floats(task_parameters["box_center_xy_m"], field="task.parameters.box_center_xy_m", size=2)
        wall = _positive(task_parameters["box_wall_thickness_m"], field="task.parameters.box_wall_thickness_m")
        if wall * 2.0 >= min(box_size[0], box_size[1]):
            raise ConfigError("box wall thickness leaves no inner opening")
        for field_name in (
            "box_floor_top_z_m",
            "pregrasp_clearance_m",
            "release_object_bottom_clearance_m",
            "grasp_gap_m",
            "grasp_settle_s",
            "pre_release_settle_s",
            "post_release_settle_s",
            "simulation_release_duration_s",
            "simulation_home_settle_s",
            "lift_success_clearance_m",
            "home_success_distance_m",
        ):
            _positive(task_parameters[field_name], field=f"task.parameters.{field_name}", allow_zero=False)
        _positive(
            task_parameters["simulation_pre_release_relax_duration_s"],
            field="task.parameters.simulation_pre_release_relax_duration_s",
            allow_zero=True,
        )
        collision_margin = float(data["safety"]["min_collision_distance_m"])
        if (
            object_size[0] + 2.0 * (wall + collision_margin) >= box_size[0]
            or object_size[1] + 2.0 * (wall + collision_margin) >= box_size[1]
        ):
            raise ConfigError("configured object plus safety margin does not fit through the box opening")
        if gripper is None:
            raise ConfigError("packaging_showcase requires a configured gripper")
        grasp_gap = float(task_parameters["grasp_gap_m"])
        relax_gap = float(task_parameters["simulation_pre_release_relax_gap_m"])
        for field_name, gap in (("grasp_gap_m", grasp_gap), ("simulation_pre_release_relax_gap_m", relax_gap)):
            if not gripper.closed_gap_m <= gap <= gripper.open_gap_m:
                raise ConfigError(
                    f"task.parameters.{field_name} must be within the configured gripper gap range "
                    f"[{gripper.closed_gap_m}, {gripper.open_gap_m}]"
                )

        object_position = _tuple_floats(
            task_parameters["fixed_object_position_m"],
            field="task.parameters.fixed_object_position_m",
            size=3,
        )
        target_position = _tuple_floats(
            task_parameters["fixed_target_position_m"],
            field="task.parameters.fixed_target_position_m",
            size=3,
        )
        table_top_z = table_center[2] + table_size[2] / 2.0
        object_support_z = object_position[2] - object_size[2] / 2.0
        if not math.isclose(object_support_z, table_top_z, rel_tol=0.0, abs_tol=1e-9):
            raise ConfigError("fixed object must rest on the configured table top")
        floor_top_z = _positive(task_parameters["box_floor_top_z_m"], field="task.parameters.box_floor_top_z_m")
        if floor_top_z <= table_top_z:
            raise ConfigError("box floor top must be above the configured table top")
        floor_thickness = floor_top_z - table_top_z
        if floor_thickness >= box_size[2]:
            raise ConfigError("box floor thickness must be smaller than the box height")
        expected_target_z = floor_top_z + object_size[2] / 2.0
        if not math.isclose(target_position[2], expected_target_z, rel_tol=0.0, abs_tol=1e-9):
            raise ConfigError("fixed target Z must equal box floor top plus half the object height")

        inner_x = box_size[0] - 2.0 * wall
        inner_y = box_size[1] - 2.0 * wall
        if abs(target_position[0] - box_center[0]) + object_size[0] / 2.0 + collision_margin >= inner_x / 2.0:
            raise ConfigError("fixed target X footprint is outside the box opening")
        if abs(target_position[1] - box_center[1]) + object_size[1] / 2.0 + collision_margin >= inner_y / 2.0:
            raise ConfigError("fixed target Y footprint is outside the box opening")

        table_min_x = table_center[0] - table_size[0] / 2.0
        table_max_x = table_center[0] + table_size[0] / 2.0
        table_min_y = table_center[1] - table_size[1] / 2.0
        table_max_y = table_center[1] + table_size[1] / 2.0

        def require_on_table(name: str, center_x: float, center_y: float, size_x: float, size_y: float) -> None:
            if (
                center_x - size_x / 2.0 < table_min_x
                or center_x + size_x / 2.0 > table_max_x
                or center_y - size_y / 2.0 < table_min_y
                or center_y + size_y / 2.0 > table_max_y
            ):
                raise ConfigError(f"configured {name} footprint is outside the table")

        require_on_table("object", object_position[0], object_position[1], object_size[0], object_size[1])
        require_on_table("box", box_center[0], box_center[1], box_size[0], box_size[1])
    task = TaskProfile(
        name=task_name,
        parameters=task_parameters,
        allowed_contacts=task_data["allowed_contacts"],
    )
    motion_data = data["motion"]
    motion = MotionConfig(
        rate_hz=_positive(motion_data["rate_hz"], field="motion.rate_hz"),
        joint_speed_rad_s=_positive(motion_data["joint_speed_rad_s"], field="motion.joint_speed_rad_s"),
        joint_acceleration_rad_s2=_positive(
            motion_data["joint_acceleration_rad_s2"], field="motion.joint_acceleration_rad_s2"
        ),
        cartesian_speed_m_s=_positive(motion_data["cartesian_speed_m_s"], field="motion.cartesian_speed_m_s"),
        cartesian_acceleration_m_s2=_positive(
            motion_data["cartesian_acceleration_m_s2"], field="motion.cartesian_acceleration_m_s2"
        ),
        gripper_duration_s=_positive(motion_data["gripper_duration_s"], field="motion.gripper_duration_s"),
    )
    simulation_data = data["simulation"]
    raw_simulation_substeps = simulation_data["substeps"]
    if isinstance(raw_simulation_substeps, bool) or not isinstance(raw_simulation_substeps, int):
        raise ConfigError("simulation.substeps must be a positive integer")
    simulation_substeps = int(raw_simulation_substeps)
    if simulation_substeps < 1:
        raise ConfigError("simulation.substeps must be a positive integer")
    simulation = SimulationConfig(
        backend=str(simulation_data["backend"]),
        precision=str(simulation_data["precision"]),
        seed=int(simulation_data["seed"]),
        substeps=simulation_substeps,
        constraint_solver=str(simulation_data["constraint_solver"]),
        solver_iterations=int(simulation_data["solver_iterations"]),
        noslip_iterations=int(simulation_data["noslip_iterations"]),
        friction_cone=str(simulation_data["friction_cone"]),
        contact_resolution=str(simulation_data["contact_resolution"]),
        constraint_time_constant_s=_positive(
            simulation_data["constraint_time_constant_s"],
            field="simulation.constraint_time_constant_s",
        ),
        use_gjk_collision=(
            None if simulation_data["use_gjk_collision"] is None else bool(simulation_data["use_gjk_collision"])
        ),
        show_viewer=bool(simulation_data.get("show_viewer", False)),
        g2=G2SimulationConfig(**simulation_data["g2"]),
    )
    safety_data = data["safety"]
    geometry_pairs = safety_data.get("excluded_geometry_pairs", [])
    if not isinstance(geometry_pairs, list) or any(not isinstance(pair, list | tuple) for pair in geometry_pairs):
        raise ConfigError("safety.excluded_geometry_pairs must be a list of geometry pairs")
    exemptions = tuple(CollisionExemption(**item) for item in safety_data.get("exemptions", []))
    workspace_lower = _tuple_floats(safety_data["workspace_lower_m"], field="safety.workspace_lower_m", size=3)
    workspace_upper = _tuple_floats(safety_data["workspace_upper_m"], field="safety.workspace_upper_m", size=3)
    safety = SafetyPolicy(
        schema_version=int(safety_data["schema_version"]),
        joint_limit_margin_rad=float(safety_data["joint_limit_margin_rad"]),
        workspace_lower_m=(workspace_lower[0], workspace_lower[1], workspace_lower[2]),
        workspace_upper_m=(workspace_upper[0], workspace_upper[1], workspace_upper[2]),
        z_min_m=float(safety_data["z_min_m"]),
        min_collision_distance_m=float(safety_data["min_collision_distance_m"]),
        max_ik_jump_rad=float(safety_data["max_ik_jump_rad"]),
        max_orientation_step_rad=float(safety_data["max_orientation_step_rad"]),
        max_shadow_joint_error_rad=float(safety_data["max_shadow_joint_error_rad"]),
        max_shadow_ee_error_m=float(safety_data["max_shadow_ee_error_m"]),
        minor_lateness_ratio=float(safety_data["minor_lateness_ratio"]),
        minor_lateness_limit_per_s=int(safety_data["minor_lateness_limit_per_s"]),
        feedback_stale_periods=float(safety_data["feedback_stale_periods"]),
        require_kinematics=bool(safety_data["require_kinematics"]),
        require_collision=bool(safety_data["require_collision"]),
        exemptions=exemptions,
        excluded_geometry_pairs=tuple(tuple(pair) for pair in geometry_pairs),
    )
    provisional = ResolvedRuntimeConfig(
        schema_version=1,
        robot=robot,
        arm=arm,
        gripper=gripper,
        task=task,
        motion=motion,
        simulation=simulation,
        safety=safety,
        sources=sources,
        sha256="0" * 64,
    )
    digest = config_sha256(_canonical_payload(provisional))
    return ResolvedRuntimeConfig(**{**provisional.__dict__, "sha256": digest})


def load_runtime_config(
    robot_key: str,
    *,
    task: str = "pick_place",
    config_path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    asset_store: RepositoryAssetStore | None = None,
) -> ResolvedRuntimeConfig:
    """Load defaults, an optional user YAML, and explicit dotted overrides.

    ``XARM_IP`` is intentionally absent: connection details are runtime input
    and never influence the reproducibility hash.
    """

    store = asset_store or RepositoryAssetStore.discover()
    canonical = get_profile_key_for_robot_name(robot_key)
    base_paths = (
        store.require(f"assets/configs/runtime/robots/{canonical}.yaml"),
        store.require(f"assets/configs/runtime/tasks/{task}.yaml"),
        store.require("assets/configs/runtime/motion/default.yaml"),
        store.require("assets/configs/runtime/simulation/default.yaml"),
        store.require("assets/configs/runtime/safety/default.yaml"),
    )
    data: dict[str, Any] = {"schema_version": 1}
    sources: list[str] = []
    for path in base_paths:
        layer = _read_yaml(path)
        if "simulation" in layer and data.get("gripper", {}).get("adapter") == "g2":
            layer["simulation"]["constraint_time_constant_s"] = 0.00125
            layer["simulation"]["noslip_iterations"] = G2_NOSLIP_ITERATIONS
        for key, value in layer.items():
            if key != "schema_version":
                if key in data:
                    raise ConfigError(f"duplicate default configuration section: {key}")
                data[key] = value
        sources.append(str(path.relative_to(store.root)))
    for relative in (
        f"assets/configs/runtime/tasks/robots/{canonical}_{task}.yaml",
        f"assets/configs/runtime/safety/robots/{canonical}.yaml",
    ):
        path = store.root / relative
        if path.is_file():
            _merge_strict(data, _read_yaml(path))
            sources.append(relative)
    if config_path is not None:
        user_path = Path(config_path).expanduser().resolve()
        _merge_strict(data, _read_yaml(user_path))
        sources.append(str(user_path))
    for dotted, value in (overrides or {}).items():
        keys = str(dotted).split(".")
        target: dict[str, Any] = data
        for key in keys[:-1]:
            current = target.get(key)
            if not isinstance(current, dict):
                raise ConfigError(f"unknown configuration override: {dotted}")
            target = current
        leaf = keys[-1]
        if leaf not in target or isinstance(target[leaf], dict):
            raise ConfigError(f"unknown scalar configuration override: {dotted}")
        target[leaf] = value
    resolved = _build_resolved(data, tuple(sources))
    if resolved.robot.key != canonical:
        raise ConfigError(f"requested robot {canonical}, but configuration identifies {resolved.robot.key}")
    store.require(Path(resolved.robot.assets_dir) / resolved.robot.urdf)
    return resolved


def runtime_config_dict(config: ResolvedRuntimeConfig, *, include_provenance: bool = True) -> dict[str, Any]:
    payload = _canonical_payload(config)
    if include_provenance:
        payload["sources"] = list(config.sources)
        payload["sha256"] = config.sha256
    return payload


def dump_runtime_config(config: ResolvedRuntimeConfig) -> str:
    """Return stable, human-readable YAML without initializing any backend."""

    return yaml.safe_dump(runtime_config_dict(config), sort_keys=False, allow_unicode=True)
