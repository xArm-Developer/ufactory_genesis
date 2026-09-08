"""Genesis version and private-hook compatibility checks.

Genesis 1.4.0 is both the minimum and the reference pinned physics baseline for
the contact-v1 pick-place campaign (local and training server aligned).

Migration from 1.3.3 to 1.4.0:
- RigidEntity.forward_kinematics() was removed; use get_links_pos/get_links_quat
  after set_qpos instead, or rely on IK's internal FK.
- RigidEntity.set_links_inertial_mass() was renamed to set_links_mass().
- IK scratch fields (_IK_qpos_orig etc.) are now allocated lazily inside the
  solver; the ensure_ik_scratch() helper is retained for API compatibility but
  is a no-op on 1.4.0+.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
import inspect
import threading
from types import ModuleType
from typing import Any, Callable
import warnings

from packaging.version import InvalidVersion, Version


MIN_GENESIS_VERSION = Version("1.4.0")
VALIDATED_GENESIS_VERSION = Version("1.4.0")  # reference / pinned baseline alias

_WARNING_LOCK = threading.Lock()
_WARNED_UNVALIDATED = False


class GenesisCompatibilityError(RuntimeError):
    """Genesis is missing or does not satisfy the project's runtime contract."""


@dataclass(frozen=True)
class DeferredViewerAPI:
    viewer_type: type
    default_aspect_ratio: float
    default_height_ratio: float


def require_genesis_version() -> Version:
    """Return the installed Genesis version after enforcing the minimum."""

    try:
        raw_version = metadata.version("genesis-world")
    except metadata.PackageNotFoundError as exc:
        raise GenesisCompatibilityError(
            f"Genesis is not installed; install genesis-world>={MIN_GENESIS_VERSION}."
        ) from exc
    try:
        version = Version(raw_version)
    except InvalidVersion as exc:
        raise GenesisCompatibilityError(f"Cannot parse installed Genesis version: {raw_version!r}.") from exc
    if version < MIN_GENESIS_VERSION:
        raise GenesisCompatibilityError(f"Genesis>={MIN_GENESIS_VERSION} is required; found {raw_version}.")
    if version != VALIDATED_GENESIS_VERSION:
        _warn_unvalidated_version(version)
    return version


def _warn_unvalidated_version(version: Version) -> None:
    global _WARNED_UNVALIDATED
    with _WARNING_LOCK:
        if _WARNED_UNVALIDATED:
            return
        warnings.warn(
            f"Genesis {version} satisfies the minimum version, but only {VALIDATED_GENESIS_VERSION} "
            "is the project's reference baseline with maintainer sim/hardware verification. "
            "Compatibility hooks will be checked before use.",
            RuntimeWarning,
            stacklevel=3,
        )
        _WARNED_UNVALIDATED = True


def _require_callable(owner: Any, name: str, context: str) -> Callable[..., Any]:
    value = getattr(owner, name, None)
    if not callable(value):
        raise GenesisCompatibilityError(f"Genesis {context} requires callable {name!r}.")
    return value


def _require_parameters(function: Callable[..., Any], required: set[str], context: str) -> None:
    try:
        parameters = set(inspect.signature(function).parameters)
    except (TypeError, ValueError) as exc:
        raise GenesisCompatibilityError(f"Cannot inspect Genesis {context} signature.") from exc
    missing = sorted(required - parameters)
    if missing:
        raise GenesisCompatibilityError(f"Genesis {context} is incompatible; missing parameters: {', '.join(missing)}.")


def require_genesis_runtime(gs_module: ModuleType | Any | None = None) -> Any:
    """Validate the public runtime and kinematics entry points used by the project."""

    require_genesis_version()
    if gs_module is None:
        try:
            import genesis as gs_module
        except ImportError as exc:
            raise GenesisCompatibilityError("The genesis package cannot be imported.") from exc

    for name in ("init", "destroy", "Scene"):
        _require_callable(gs_module, name, "runtime")
    for name in ("options", "morphs", "surfaces"):
        if getattr(gs_module, name, None) is None:
            raise GenesisCompatibilityError(f"Genesis runtime requires module attribute {name!r}.")
    for name in ("constraint_solver", "friction_cone", "contact_resolution"):
        if getattr(gs_module, name, None) is None:
            raise GenesisCompatibilityError(f"Genesis runtime requires module attribute {name!r}.")
    rigid_options = _require_callable(gs_module.options, "RigidOptions", "rigid options")
    _require_parameters(
        rigid_options,
        {"constraint_solver", "friction_cone", "contact_resolution", "noslip_iterations"},
        "RigidOptions",
    )

    try:
        from genesis.engine.entities.rigid_entity.rigid_entity import RigidEntity
    except ImportError as exc:
        raise GenesisCompatibilityError("Genesis rigid-entity kinematics API is unavailable.") from exc
    inverse = _require_callable(RigidEntity, "inverse_kinematics", "inverse kinematics")
    _require_parameters(
        inverse,
        {"link", "pos", "quat", "init_qpos", "dofs_idx_local", "damping"},
        "RigidEntity.inverse_kinematics",
    )
    _require_callable(RigidEntity, "get_links_pos", "link position accessor")
    _require_callable(RigidEntity, "get_links_quat", "link orientation accessor")
    _require_callable(RigidEntity, "set_links_mass", "link mass setter")
    return gs_module


def require_genesis_capabilities(
    gs_module: ModuleType | Any | None = None,
    *,
    pbr: bool = False,
    deferred_viewer: bool = False,
) -> Any:
    """Validate requested Genesis capabilities without initializing a scene.

    Hardware-facing commands call this before connecting to a controller so a
    future Genesis release cannot fail late after external state is involved.
    """

    gs_module = require_genesis_runtime(gs_module)
    if pbr:
        try:
            import genesis.utils.gltf as gltf_utils
            import genesis.utils.mesh as mesh_utils
        except ImportError as exc:
            raise GenesisCompatibilityError("Genesis GLB PBR utility modules are unavailable.") from exc
        require_pbr_hooks(gs_module, gltf_utils, mesh_utils)
    if deferred_viewer:
        load_deferred_viewer_api(gs_module)
    return gs_module


def require_pbr_hooks(gs_module: Any, gltf_utils: Any, mesh_utils: Any) -> None:
    """Validate every Genesis hook patched by the scoped GLB PBR integration."""

    require_genesis_version()
    parse_mesh_glb = _require_callable(gltf_utils, "parse_mesh_glb", "GLB parser")
    surface_to_visual = _require_callable(mesh_utils, "surface_uvs_to_trimesh_visual", "surface-to-trimesh conversion")
    mesh_class = getattr(gs_module, "Mesh", None)
    from_trimesh = getattr(mesh_class, "from_trimesh", None) if mesh_class is not None else None
    if not callable(from_trimesh) or "from_trimesh" not in getattr(mesh_class, "__dict__", {}):
        raise GenesisCompatibilityError("Genesis Mesh.from_trimesh classmethod hook is unavailable.")
    _require_parameters(
        parse_mesh_glb,
        {"path", "group_by_material", "scale", "is_mesh_zup", "surface"},
        "parse_mesh_glb",
    )
    _require_parameters(
        from_trimesh,
        {
            "mesh",
            "scale",
            "convexify",
            "decimate",
            "decimate_face_num",
            "decimate_aggressiveness",
            "metadata",
            "surface",
            "is_mesh_zup",
        },
        "Mesh.from_trimesh",
    )
    _require_parameters(surface_to_visual, {"surface", "uvs", "n_verts"}, "surface_uvs_to_trimesh_visual")


def load_deferred_viewer_api(gs_module: Any) -> DeferredViewerAPI:
    """Load and validate the private Viewer entry points used for late startup."""

    require_genesis_version()
    if not isinstance(getattr(gs_module, "_scene_registry", None), list):
        raise GenesisCompatibilityError("Genesis deferred Viewer requires the private _scene_registry list.")
    try:
        from genesis.vis.viewer import Viewer
        from genesis.vis.visualizer import VIEWER_DEFAULT_ASPECT_RATIO, VIEWER_DEFAULT_HEIGHT_RATIO
    except ImportError as exc:
        raise GenesisCompatibilityError("Genesis deferred Viewer modules are unavailable.") from exc
    _require_parameters(Viewer, {"options", "context"}, "Viewer")
    try:
        aspect_ratio = float(VIEWER_DEFAULT_ASPECT_RATIO)
        height_ratio = float(VIEWER_DEFAULT_HEIGHT_RATIO)
    except (TypeError, ValueError) as exc:
        raise GenesisCompatibilityError("Genesis Viewer default ratios are invalid.") from exc
    if aspect_ratio <= 0.0 or height_ratio <= 0.0:
        raise GenesisCompatibilityError("Genesis Viewer default ratios must be positive.")
    return DeferredViewerAPI(Viewer, aspect_ratio, height_ratio)


def ensure_ik_scratch(robot: Any, *, gs_module: Any | None = None, qd_module: Any | None = None) -> None:
    """No-op retained for API compatibility.

    Genesis 1.4.0 allocates IK scratch fields lazily inside the solver when
    inverse_kinematics is first called. Manual pre-allocation is no longer
    required, so this function now does nothing beyond a version check.
    """
    require_genesis_version()
