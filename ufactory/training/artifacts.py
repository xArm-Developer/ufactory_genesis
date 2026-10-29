"""Strict YAML configuration and integrity manifests for RL checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

import yaml

from ufactory.training.reach_config import REACH_SUCCESS_DISTANCE_M
from ufactory.training.pick_place_config import PICK_PLACE_ACTION_CONTRACT, PICK_PLACE_ENV_VERSION
from ufactory.config.models import G2_PHYSICS_PROFILE


class ArtifactError(ValueError):
    pass


_PICK_PLACE_RUNTIME_ENV_CONTRACT_KEYS = (
    "runtime_config_sha256",
    "physics_profile",
    "g2",
    "g2_physics",
    "episode_length_s",
    "ctrl_dt",
    "spawn_xy_lower",
    "spawn_xy_upper",
    "workspace_lower",
    "workspace_upper",
    "place_success_dist_m",
    "success_hold_steps",
    "substeps",
    "constraint_solver",
    "solver_iterations",
    "noslip_iterations",
    "friction_cone",
    "contact_resolution",
    "constraint_time_constant_s",
    "use_gjk_collision",
    "obj_mass_kg",
    "task_name",
    "env_version",
    "num_obs",
    "num_actions",
    "control",
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
    "drag_limit_m",
    "tilt_limit_deg",
)
_REACH_RUNTIME_ENV_CONTRACT_KEYS = (
    "task_name",
    "env_version",
    "num_obs",
    "num_actions",
    "episode_length_s",
    "ctrl_dt",
    "success_distance_m",
    "action_scale",
    "action_clip",
    "servo_speed_rad_s",
    "max_joint_delta_rad",
    "target_pos_lower",
    "target_pos_upper",
    "substeps",
    "constraint_solver",
    "solver_iterations",
    "noslip_iterations",
    "friction_cone",
    "contact_resolution",
    "constraint_time_constant_s",
    "use_gjk_collision",
)


def runtime_env_contract_keys(env: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the runtime environment contract keys for the given task."""

    if str(env.get("task_name", "")) == "reach":
        return _REACH_RUNTIME_ENV_CONTRACT_KEYS
    return _PICK_PLACE_RUNTIME_ENV_CONTRACT_KEYS


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise ArtifactError(f"training configuration contains unsupported type: {type(value).__name__}")


def _finite(value: Any, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ArtifactError(f"{path} must be finite")


def _canonical_sha256(data: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_env_contract(env: Mapping[str, Any]) -> dict[str, Any]:
    """Return the task fields that must agree with the resolved runtime YAML."""

    return {key: _plain(env.get(key)) for key in runtime_env_contract_keys(env)}


def _git_provenance(cwd: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    diff = run("diff", "--binary", "HEAD")
    return {
        "commit": None if commit is None else commit.strip(),
        "dirty": None if status is None else bool(status.strip()),
        "status_sha256": None if status is None else hashlib.sha256(status.encode()).hexdigest(),
        "diff_sha256": None if diff is None else hashlib.sha256(diff.encode()).hexdigest(),
    }


def write_run_provenance(
    path: str | Path,
    *,
    training_config: Mapping[str, Any],
    source_paths: list[str | Path],
) -> Path:
    """Write machine, dependency, Git, config and source hashes."""

    target = Path(path)
    resolved_sources: list[dict[str, str]] = []
    for source_path in source_paths:
        source = Path(source_path).resolve()
        if not source.is_file():
            raise ArtifactError(f"provenance source does not exist: {source}")
        resolved_sources.append({"path": str(source), "sha256": _file_sha256(source)})

    package_versions: dict[str, str | None] = {}
    for name in (
        "genesis-world",
        "quadrants",
        "rsl-rl-lib",
        "torch",
        "tensordict",
        "numpy",
        "pyyaml",
    ):
        try:
            package_versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            package_versions[name] = None

    gpu_query = None
    try:
        gpu_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass

    body = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": package_versions,
        "gpu": gpu_query,
        "git": _git_provenance(Path.cwd()),
        "training_config_sha256": str(training_config["config_sha256"]),
        "runtime_env_contract_sha256": _canonical_sha256(runtime_env_contract(training_config["env"])),
        "sources": resolved_sources,
    }
    target.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def write_training_config(
    path: str | Path,
    *,
    task: str,
    robot_key: str,
    env: Mapping[str, Any],
    reward: Mapping[str, Any],
    robot: Mapping[str, Any],
    train: Mapping[str, Any],
) -> str:
    target = Path(path)
    body = {
        "schema_version": 1,
        "task": str(task),
        "robot_key": str(robot_key),
        "env": _plain(env),
        "reward": _plain(reward),
        "robot": _plain(robot),
        "train": _plain(train),
    }
    _finite(body)
    body["config_sha256"] = _canonical_sha256(body)
    target.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return str(body["config_sha256"])


def load_training_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ArtifactError(f"cannot read training config {source}: {exc}") from exc
    required = {"schema_version", "task", "robot_key", "env", "reward", "robot", "train", "config_sha256"}
    if not isinstance(data, dict) or set(data) != required:
        raise ArtifactError(f"training config fields must be exactly {sorted(required)}")
    if data["schema_version"] != 1:
        raise ArtifactError("unsupported training config schema")
    _finite(data)
    claimed = str(data.pop("config_sha256"))
    actual = _canonical_sha256(data)
    data["config_sha256"] = claimed
    if claimed != actual:
        raise ArtifactError("training config hash mismatch")
    for section in ("env", "reward", "robot", "train"):
        if not isinstance(data[section], dict):
            raise ArtifactError(f"training config {section} must be a mapping")
    return data


@dataclass(frozen=True)
class CheckpointManifest:
    schema_version: int
    task: str
    robot_key: str
    dof: int
    observation_dim: int
    action_dim: int
    executor_action_contract: str
    config_sha256: str
    checkpoint_sha256: str
    code_version: str
    checkpoint_file: str


def _code_version() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_checkpoint_manifest(
    checkpoint_path: str | Path,
    *,
    training_config: Mapping[str, Any],
    executor_action_contract: str,
    output_path: str | Path | None = None,
) -> Path:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise ArtifactError(f"checkpoint does not exist: {checkpoint}")
    env = training_config["env"]
    robot = training_config["robot"]
    manifest = CheckpointManifest(
        schema_version=1,
        task=str(training_config["task"]),
        robot_key=str(training_config["robot_key"]),
        dof=int(env.get("num_actions", len(robot.get("joint_names", ())))),
        observation_dim=int(env["num_obs"]),
        action_dim=int(env["num_actions"]),
        executor_action_contract=str(executor_action_contract),
        config_sha256=str(training_config["config_sha256"]),
        checkpoint_sha256=_file_sha256(checkpoint),
        code_version=_code_version(),
        checkpoint_file=checkpoint.name,
    )
    target = Path(output_path) if output_path is not None else checkpoint.with_suffix(".checkpoint_manifest.json")
    target.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def write_artifact_inventory(
    path: str | Path,
    *,
    training_config: Mapping[str, Any],
    checkpoints: list[Path],
    selected_checkpoint: Path | None = None,
) -> Path:
    """Write the complete resolved config and deterministic checkpoint inventory."""

    body = {
        "schema_version": 1,
        "training_config": _plain(training_config),
        "checkpoints": [
            {
                "file": checkpoint.name,
                "sha256": _file_sha256(checkpoint),
                "manifest": checkpoint.with_suffix(".checkpoint_manifest.json").name,
            }
            for checkpoint in checkpoints
        ],
        "selected_checkpoint": None if selected_checkpoint is None else selected_checkpoint.name,
    }
    _finite(body)
    target = Path(path)
    target.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return target


def load_checkpoint_manifest(path: str | Path) -> CheckpointManifest:
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read checkpoint manifest {source}: {exc}") from exc
    required = set(CheckpointManifest.__dataclass_fields__)
    if not isinstance(data, dict) or set(data) != required:
        raise ArtifactError(f"checkpoint manifest fields must be exactly {sorted(required)}")
    manifest = CheckpointManifest(**data)
    if manifest.schema_version != 1 or len(manifest.checkpoint_sha256) != 64:
        raise ArtifactError("invalid checkpoint manifest schema or hash")
    return manifest


def validate_checkpoint_artifacts(
    checkpoint_path: str | Path,
    config_path: str | Path,
    manifest_path: str | Path | None = None,
    *,
    expected_task: str | None = None,
    expected_robot_key: str | None = None,
    expected_runtime_config_sha256: str | None = None,
    expected_runtime_env: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], CheckpointManifest]:
    checkpoint = Path(checkpoint_path)
    config = load_training_config(config_path)
    if (
        expected_task == "pick_place"
        and expected_runtime_env is not None
        and expected_runtime_env.get("env_version") == PICK_PLACE_ENV_VERSION
        and config["env"].get("env_version") != PICK_PLACE_ENV_VERSION
    ):
        raise ArtifactError("pick-place environment v3 changed observations and continuous control; retrain the policy")
    path = Path(manifest_path) if manifest_path else checkpoint.with_suffix(".checkpoint_manifest.json")
    manifest = load_checkpoint_manifest(path)
    failures: list[str] = []
    if _file_sha256(checkpoint) != manifest.checkpoint_sha256:
        failures.append("checkpoint hash")
    if config["config_sha256"] != manifest.config_sha256:
        failures.append("config hash")
    if manifest.checkpoint_file != checkpoint.name:
        failures.append("checkpoint filename")
    if expected_task is not None and manifest.task != expected_task:
        failures.append("task")
    if expected_robot_key is not None and manifest.robot_key != expected_robot_key:
        failures.append("robot")
    env = config["env"]
    if env.get("task_name") == "pick_place" and env.get("env_version") == PICK_PLACE_ENV_VERSION:
        if manifest.executor_action_contract != PICK_PLACE_ACTION_CONTRACT:
            failures.append("pick-place v3 action contract (retrain the policy)")
    if config["task"] == "pick_place" and env.get("physics_profile") != G2_PHYSICS_PROFILE:
        failures.append("G2 physics version (retrain the policy)")
    if config["task"] == "reach" and env.get("success_distance_m") != REACH_SUCCESS_DISTANCE_M:
        failures.append("reach success_distance_m (5 mm required)")
    if expected_runtime_config_sha256 is not None and env.get("runtime_config_sha256") != str(
        expected_runtime_config_sha256
    ):
        failures.append("runtime config hash")
    if expected_runtime_env is not None:
        if expected_task == "reach" and env.get("env_version") != expected_runtime_env.get("env_version"):
            raise ArtifactError(
                "reach environment version changed (3 * dof + 6 observations, accumulated commands); retrain the policy"
            )
        saved_keys = runtime_env_contract_keys(env)
        expected_keys = runtime_env_contract_keys(expected_runtime_env)
        if saved_keys != expected_keys:
            failures.append("runtime config body (task_name)")
        else:
            saved_contract = runtime_env_contract(env)
            expected_contract = runtime_env_contract(expected_runtime_env)
            mismatched_keys = [key for key in saved_keys if saved_contract.get(key) != expected_contract.get(key)]
            if mismatched_keys:
                failures.append(f"runtime config body ({', '.join(mismatched_keys)})")
    if int(env["num_obs"]) != manifest.observation_dim or int(env["num_actions"]) != manifest.action_dim:
        failures.append("observation/action dimensions")
    if failures:
        raise ArtifactError(f"checkpoint artifact mismatch: {', '.join(failures)}")
    return config, manifest


def validate_and_load_rsl_checkpoint(
    checkpoint_path: str | Path,
    config_path: str | Path,
    manifest_path: str | Path | None = None,
    *,
    map_location: Any = "cpu",
    expected_task: str | None = None,
    expected_robot_key: str | None = None,
    expected_runtime_config_sha256: str | None = None,
    expected_runtime_env: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], CheckpointManifest]:
    """Validate a complete artifact bundle before safely loading RSL-RL weights.

    PyTorch's ``weights_only`` loader rejects arbitrary pickle globals.  Integrity
    validation deliberately happens first so a public entry point never asks
    PyTorch to deserialize an unverified checkpoint.
    """

    config, manifest = validate_checkpoint_artifacts(
        checkpoint_path,
        config_path,
        manifest_path,
        expected_task=expected_task,
        expected_robot_key=expected_robot_key,
        expected_runtime_config_sha256=expected_runtime_config_sha256,
        expected_runtime_env=expected_runtime_env,
    )

    # Keep torch optional for configuration-only users and lightweight CI
    # collection.  It is required only when weights are actually requested.
    import torch

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ArtifactError("RSL-RL checkpoint payload must be a mapping")
    required = {"actor_state_dict", "critic_state_dict"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ArtifactError(f"RSL-RL checkpoint is missing fields: {missing}")
    return checkpoint, config, manifest
