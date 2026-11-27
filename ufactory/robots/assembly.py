"""Runtime URDF assembly: arm URDF [+ mounting plate] [+ camera stand / camera] [+ end-effector part].

Flanges, parts and mounts live in ``assets/urdf/assembly.yaml``. Robots sharing a flange share each part's mount. An
accessory is a spacer (``camera_stand``) or a camera on a spacer; the spacer sits between the flange and the end
effector, so the end effector (or the arm's ``link_eef``) moves by the spacer's ``tool_lift_m``. A mounting plate sits
below ``link_base`` with its top mounting surface at ``z = 0``. Composed files are written under
``<repo>/.cache/ufactory/urdf/composed/`` with mesh paths relative to that directory, so identical inputs give
identical bytes (and ``sha256``) on every checkout. File names are content-addressed and writes are atomic.

Runtime configurations and :func:`ufactory.robots.paths.robot_urdf` accept an assembly reference in place of a
URDF file name::

    assembly:<end_effector>[+<ft_sensor>][+<accessory>][+mounting_plate][:<visual>]
    assembly:bare+<ft_sensor>[+<accessory>][+mounting_plate][:<visual>]
    assembly:bare+<accessory>[+mounting_plate][:<visual>]
    assembly:bare+mounting_plate[:<visual>]

``bare`` means no end effector. ``visual`` is ``glb`` (default, per-link GLB arm and gripper visuals) or ``stl`` (the
arm's dynamics URDF; end-effector GLB visuals are replaced by their collision geometry; cameras, F/T sensors, and
mounting plates draw their STL/collision geometry).
"""

from __future__ import annotations

import copy
import hashlib
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Protocol

import yaml

from ufactory.robots.registry import PROJECT_ROOT, ROBOT_PROFILES, get_profile_key_for_robot_name, get_robot_profile

ASSEMBLY_PREFIX = "assembly:"
ASSEMBLY_YAML = Path("assets") / "urdf" / "assembly.yaml"
COMPOSED_DIR = Path(".cache") / "ufactory" / "urdf" / "composed"
VISUALS = ("glb", "stl")
# Bump when the composition rules change so cached files are regenerated.
COMPOSER_VERSION = "1"


BARE = "bare"
MOUNTING_PLATE = "mounting_plate"
FT_SENSOR_PREFIX = "ft_sensor_"


class AssemblyError(ValueError):
    """Raised for unknown parts, unsupported robot/end-effector pairs, or malformed references."""


class _AssetStore(Protocol):
    @property
    def root(self) -> Path: ...

    def require(self, relative_path: str | Path) -> Path: ...


Pose = tuple[tuple[float, float, float], tuple[float, float, float]]
IDENTITY: Pose = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


@dataclass(frozen=True)
class EndEffector:
    name: str
    source: str
    base_link: str
    mount_joint: str
    flange: str
    robots: tuple[str, ...]
    mount: Pose
    glb_visuals: Mapping[str, str]
    stl_visuals: Mapping[str, str]
    stl_materials: Mapping[str, str]
    adapter: str | None = None


@dataclass(frozen=True)
class Adapter:
    name: str
    source: str
    base_link: str
    mount_joint: str
    flange: str
    robots: tuple[str, ...]
    mount: Pose
    tool_lift_m: float
    glb_visuals: Mapping[str, str]
    stl_visuals: Mapping[str, str]
    stl_materials: Mapping[str, str]


@dataclass(frozen=True)
class FTSensor:
    name: str
    source: str
    base_link: str
    output_link: str
    mount_joint: str
    flange: str
    robots: tuple[str, ...]
    mount: Pose
    tool_lift_m: float
    glb_visuals: Mapping[str, str]
    stl_visuals: Mapping[str, str]
    stl_materials: Mapping[str, str]


@dataclass(frozen=True)
class Spacer:
    name: str
    source: str
    base_link: str
    mount_joint: str
    flanges: tuple[str, ...]
    robots: tuple[str, ...]
    mount: Pose
    tool_lift_m: float
    end_effectors: tuple[str, ...]
    glb_visuals: Mapping[str, str]
    stl_visuals: Mapping[str, str]
    stl_materials: Mapping[str, str]


@dataclass(frozen=True)
class Camera:
    name: str
    source: str
    base_link: str
    mount_joint: str
    spacer: str
    mount: Pose
    optical_frame: str
    glb_visuals: Mapping[str, str]
    stl_visuals: Mapping[str, str]
    stl_materials: Mapping[str, str]


@dataclass(frozen=True)
class MountingPlate:
    name: str
    source: str
    base_link: str
    mount_joint: str
    robots: tuple[str, ...]
    thickness_m: float
    mount: Pose
    glb_visuals: Mapping[str, str]
    stl_visuals: Mapping[str, str]
    stl_materials: Mapping[str, str]


class _VisualParts(Protocol):
    @property
    def glb_visuals(self) -> Mapping[str, str]: ...

    @property
    def stl_visuals(self) -> Mapping[str, str]: ...

    @property
    def stl_materials(self) -> Mapping[str, str]: ...


@dataclass(frozen=True)
class _Catalog:
    end_effectors: dict[str, EndEffector]
    adapters: dict[str, Adapter]
    ft_sensors: dict[str, FTSensor]
    spacers: dict[str, Spacer]
    cameras: dict[str, Camera]
    mounting_plates: dict[str, MountingPlate]


@dataclass(frozen=True)
class AssemblyRef:
    end_effector: str | None
    visual: str = "glb"
    accessory: str | None = None
    mounting_plate: bool = False
    ft_sensor: str | None = None

    def __str__(self) -> str:
        return assembly_ref(
            self.end_effector,
            visual=self.visual,
            accessory=self.accessory,
            mounting_plate=self.mounting_plate,
            ft_sensor=self.ft_sensor,
        )


def _triplet(value: Any, *, where: str) -> tuple[float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise AssemblyError(f"{where} must be a list of three numbers")
    return (float(value[0]), float(value[1]), float(value[2]))


def _load_flanges(data: Mapping[str, Any], path: Path) -> dict[str, tuple[str, ...]]:
    flanges: dict[str, tuple[str, ...]] = {}
    owner: dict[str, str] = {}
    for name, spec in (data.get("flanges") or {}).items():
        robots = tuple(get_profile_key_for_robot_name(str(r)) for r in spec["robots"])
        for key in robots:
            if key in owner:
                raise AssemblyError(f"{path}: {key} listed under flanges {owner[key]!r} and {name!r}")
            owner[key] = str(name)
        flanges[str(name)] = robots
    return flanges


def _pose(spec: Mapping[str, Any], name: str) -> Pose:
    m = spec.get("mount") or {}
    return (
        _triplet(m.get("xyz", (0, 0, 0)), where=f"{name}.mount.xyz"),
        _triplet(m.get("rpy", (0, 0, 0)), where=f"{name}.mount.rpy"),
    )


def _str_map(value: Any) -> dict[str, str]:
    return {str(k): str(v) for k, v in (value or {}).items()}


@lru_cache(maxsize=8)
def _load_catalog(path: Path, mtime_ns: int) -> _Catalog:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") not in (3, 4, 5, 6):
        raise AssemblyError(f"{path}: unsupported assembly schema")
    flanges = _load_flanges(data, path)

    def flange_of(name: str, spec: Mapping[str, Any]) -> str:
        flange = str(spec["flange"])
        if flange not in flanges:
            raise AssemblyError(f"{name}: unknown flange {flange!r}; expected one of {sorted(flanges)}")
        return flange

    def flanges_of(name: str, spec: Mapping[str, Any]) -> tuple[str, ...]:
        if "flanges" not in spec:
            return (flange_of(name, spec),)
        raw = spec["flanges"]
        if not isinstance(raw, list | tuple) or not raw:
            raise AssemblyError(f"{name}: flanges must be a non-empty list")
        seen: list[str] = []
        for item in raw:
            flange = str(item)
            if flange not in flanges:
                raise AssemblyError(f"{name}: unknown flange {flange!r}; expected one of {sorted(flanges)}")
            if flange not in seen:
                seen.append(flange)
        return tuple(seen)

    adapter_map: dict[str, Adapter] = {}
    for name, spec in (data.get("adapters") or {}).items():
        adapter_name = str(name)
        flange = flange_of(adapter_name, spec)
        lift_m = float(spec.get("tool_lift_m", 0.0))
        if lift_m < 0.0:
            raise AssemblyError(f"{adapter_name}: tool_lift_m must be non-negative")
        adapter_map[adapter_name] = Adapter(
            name=adapter_name,
            source=str(spec["source"]),
            base_link=str(spec["base_link"]),
            mount_joint=str(spec["mount_joint"]),
            flange=flange,
            robots=flanges[flange],
            mount=_pose(spec, adapter_name),
            tool_lift_m=lift_m,
            glb_visuals=_str_map(spec.get("glb_visuals")),
            stl_visuals=_str_map(spec.get("stl_visuals")),
            stl_materials=_str_map(spec.get("stl_materials")),
        )

    ft_sensor_map: dict[str, FTSensor] = {}
    for name, spec in (data.get("ft_sensors") or {}).items():
        sensor_name = str(name)
        if not sensor_name.startswith(FT_SENSOR_PREFIX):
            raise AssemblyError(f"{sensor_name}: ft_sensors entry name must start with {FT_SENSOR_PREFIX!r}")
        flange = flange_of(sensor_name, spec)
        lift_m = float(spec.get("tool_lift_m", 0.0))
        if lift_m < 0.0:
            raise AssemblyError(f"{sensor_name}: tool_lift_m must be non-negative")
        output_link = str(spec.get("output_link", "")).strip()
        if not output_link:
            raise AssemblyError(f"{sensor_name}: output_link is required")
        ft_sensor_map[sensor_name] = FTSensor(
            name=sensor_name,
            source=str(spec["source"]),
            base_link=str(spec["base_link"]),
            output_link=output_link,
            mount_joint=str(spec["mount_joint"]),
            flange=flange,
            robots=flanges[flange],
            mount=_pose(spec, sensor_name),
            tool_lift_m=lift_m,
            glb_visuals=_str_map(spec.get("glb_visuals")),
            stl_visuals=_str_map(spec.get("stl_visuals")),
            stl_materials=_str_map(spec.get("stl_materials")),
        )

    out: dict[str, EndEffector] = {}
    for name, spec in (data.get("end_effectors") or {}).items():
        ee_name = str(name)
        flange = flange_of(ee_name, spec)
        mount = _pose(spec, ee_name)
        raw_adapter = spec.get("adapter")
        ee_adapter = None if raw_adapter is None else str(raw_adapter)
        if ee_adapter is not None:
            if ee_adapter not in adapter_map:
                raise AssemblyError(f"{ee_name}: unknown adapter {ee_adapter!r}; expected one of {sorted(adapter_map)}")
            if adapter_map[ee_adapter].flange != flange:
                raise AssemblyError(
                    f"{ee_name}: adapter {ee_adapter!r} is on flange {adapter_map[ee_adapter].flange!r}, not {flange!r}"
                )
        out[ee_name] = EndEffector(
            name=ee_name,
            source=str(spec["source"]),
            base_link=str(spec["base_link"]),
            mount_joint=str(spec["mount_joint"]),
            flange=flange,
            robots=flanges[flange],
            mount=mount,
            glb_visuals=_str_map(spec.get("glb_visuals")),
            stl_visuals=_str_map(spec.get("stl_visuals")),
            stl_materials=_str_map(spec.get("stl_materials")),
            adapter=ee_adapter,
        )
    spacers: dict[str, Spacer] = {}
    for name, spec in (data.get("spacers") or {}).items():
        spacer_flanges = flanges_of(str(name), spec)
        robots = tuple(r for f in spacer_flanges for r in flanges[f])
        fits = tuple(str(e) for e in spec.get("end_effectors") or ())
        for ee_name in fits:
            if ee_name not in out or out[ee_name].flange not in spacer_flanges:
                raise AssemblyError(f"{name}: end effector {ee_name!r} is not on flanges {list(spacer_flanges)}")
        spacers[str(name)] = Spacer(
            name=str(name),
            source=str(spec["source"]),
            base_link=str(spec["base_link"]),
            mount_joint=str(spec["mount_joint"]),
            flanges=spacer_flanges,
            robots=robots,
            mount=_pose(spec, str(name)),
            tool_lift_m=float(spec.get("tool_lift_m", 0.0)),
            end_effectors=fits,
            glb_visuals=_str_map(spec.get("glb_visuals")),
            stl_visuals=_str_map(spec.get("stl_visuals")),
            stl_materials=_str_map(spec.get("stl_materials")),
        )
    cameras: dict[str, Camera] = {}
    for name, spec in (data.get("cameras") or {}).items():
        spacer = str(spec["spacer"])
        if spacer not in spacers:
            raise AssemblyError(f"{name}: unknown spacer {spacer!r}; expected one of {sorted(spacers)}")
        cameras[str(name)] = Camera(
            name=str(name),
            source=str(spec["source"]),
            base_link=str(spec["base_link"]),
            mount_joint=str(spec["mount_joint"]),
            spacer=spacer,
            mount=_pose(spec, str(name)),
            optical_frame=str(spec["optical_frame"]),
            glb_visuals=_str_map(spec.get("glb_visuals")),
            stl_visuals=_str_map(spec.get("stl_visuals")),
            stl_materials=_str_map(spec.get("stl_materials")),
        )
    plates: dict[str, MountingPlate] = {}
    plate_owner: dict[str, str] = {}
    for name, spec in (data.get("mounting_plates") or {}).items():
        plate_name = str(name)
        robots = tuple(get_profile_key_for_robot_name(str(r)) for r in spec.get("robots") or ())
        if not robots:
            raise AssemblyError(f"{plate_name}: robots must be a non-empty list")
        for key in robots:
            if key in plate_owner:
                raise AssemblyError(
                    f"{path}: {key} listed under mounting_plates {plate_owner[key]!r} and {plate_name!r}"
                )
            plate_owner[key] = plate_name
        thickness_m = float(spec["thickness_m"])
        if thickness_m <= 0.0:
            raise AssemblyError(f"{plate_name}: thickness_m must be positive")
        plates[plate_name] = MountingPlate(
            name=plate_name,
            source=str(spec["source"]),
            base_link=str(spec["base_link"]),
            mount_joint=str(spec["mount_joint"]),
            robots=robots,
            thickness_m=thickness_m,
            mount=_pose(spec, plate_name),
            glb_visuals=_str_map(spec.get("glb_visuals")),
            stl_visuals=_str_map(spec.get("stl_visuals")),
            stl_materials=_str_map(spec.get("stl_materials")),
        )
    sections = (set(out), set(adapter_map), set(ft_sensor_map), set(spacers), set(cameras), set(plates))
    seen: set[str] = set()
    clash: set[str] = set()
    for sec in sections:
        clash.update(seen & sec)
        clash.update(sec & {BARE, MOUNTING_PLATE})
        seen.update(sec)
    if clash:
        raise AssemblyError(
            f"{path}: part names must be unique across sections and not {BARE!r}/{MOUNTING_PLATE!r}: {sorted(clash)}"
        )
    return _Catalog(out, adapter_map, ft_sensor_map, spacers, cameras, plates)


def _catalog(root: Path | None = None) -> _Catalog:
    path = (root or PROJECT_ROOT) / ASSEMBLY_YAML
    return _load_catalog(path, path.stat().st_mtime_ns)


def end_effectors(root: Path | None = None) -> dict[str, EndEffector]:
    """End-effector parts declared in ``assets/urdf/assembly.yaml``."""
    return _catalog(root).end_effectors


def adapters(root: Path | None = None) -> dict[str, Adapter]:
    """Flange adapters declared in ``assets/urdf/assembly.yaml``."""
    return _catalog(root).adapters


def get_adapter(name: str, root: Path | None = None) -> Adapter:
    """Return a declared flange adapter by name."""
    parts = adapters(root)
    if name not in parts:
        raise AssemblyError(f"unknown adapter {name!r}; expected one of {sorted(parts)}")
    return parts[name]


def ft_sensors(root: Path | None = None) -> dict[str, FTSensor]:
    """Six-axis force/torque sensors declared in ``assets/urdf/assembly.yaml``."""
    return _catalog(root).ft_sensors


def get_ft_sensor(name: str, root: Path | None = None) -> FTSensor:
    """Return a declared six-axis force/torque sensor by name."""
    parts = ft_sensors(root)
    if name not in parts:
        raise AssemblyError(f"unknown F/T sensor {name!r}; expected one of {sorted(parts)}")
    return parts[name]


def supported_ft_sensors(robot_key: str, root: Path | None = None) -> tuple[str, ...]:
    """F/T sensors compatible with ``robot_key``."""
    key = get_profile_key_for_robot_name(robot_key)
    return tuple(name for name, sensor in ft_sensors(root).items() if key in sensor.robots)


def supports_ft_sensor(robot_key: str, ft_sensor: str, root: Path | None = None) -> bool:
    """Whether ``ft_sensor`` fits ``robot_key``."""
    return ft_sensor in supported_ft_sensors(robot_key, root)


def spacers(root: Path | None = None) -> dict[str, Spacer]:
    """Spacers (camera stands) declared in ``assets/urdf/assembly.yaml``."""
    return _catalog(root).spacers


def cameras(root: Path | None = None) -> dict[str, Camera]:
    """Cameras declared in ``assets/urdf/assembly.yaml``."""
    return _catalog(root).cameras


def mounting_plates(root: Path | None = None) -> dict[str, MountingPlate]:
    """Mounting plates declared in ``assets/urdf/assembly.yaml``."""
    return _catalog(root).mounting_plates


def get_mounting_plate(name: str, root: Path | None = None) -> MountingPlate:
    """Return a declared mounting plate by name."""
    plates = mounting_plates(root)
    if name not in plates:
        raise AssemblyError(f"unknown mounting plate {name!r}; expected one of {sorted(plates)}")
    return plates[name]


def mounting_plate_for(robot_key: str, root: Path | None = None) -> MountingPlate:
    """Return the mounting plate matching ``robot_key``."""
    key = get_profile_key_for_robot_name(robot_key)
    for plate in mounting_plates(root).values():
        if key in plate.robots:
            return plate
    raise AssemblyError(f"no mounting plate configured for {key}")


def accessories(root: Path | None = None) -> tuple[str, ...]:
    """Accessory names: every spacer, then every camera."""
    catalog = _catalog(root)
    return (*catalog.spacers, *catalog.cameras)


def resolve_accessory(name: str, root: Path | None = None) -> tuple[Spacer, Camera | None]:
    """The spacer and (for a camera) the camera an accessory name stands for."""
    catalog = _catalog(root)
    if name in catalog.cameras:
        camera = catalog.cameras[name]
        return catalog.spacers[camera.spacer], camera
    if name in catalog.spacers:
        return catalog.spacers[name], None
    raise AssemblyError(f"unknown accessory {name!r}; expected one of {sorted(accessories(root))}")


def get_end_effector(name: str, root: Path | None = None) -> EndEffector:
    parts = end_effectors(root)
    if name not in parts:
        raise AssemblyError(f"unknown end effector {name!r}; expected one of {sorted(parts)}")
    return parts[name]


def supported_end_effectors(robot_key: str, root: Path | None = None) -> tuple[str, ...]:
    key = get_profile_key_for_robot_name(robot_key)
    return tuple(name for name, ee in end_effectors(root).items() if key in ee.robots)


def supports_end_effector(robot_key: str, end_effector: str, root: Path | None = None) -> bool:
    return end_effector in supported_end_effectors(robot_key, root)


def supported_accessories(robot_key: str, end_effector: str | None = None, root: Path | None = None) -> tuple[str, ...]:
    """Accessories that fit ``robot_key`` (with ``end_effector``; ``None`` = bare arm)."""
    key = get_profile_key_for_robot_name(robot_key)
    if end_effector is not None and not supports_end_effector(key, end_effector, root):
        return ()
    out = []
    for name in accessories(root):
        spacer, _ = resolve_accessory(name, root)
        if key in spacer.robots and (end_effector is None or end_effector in spacer.end_effectors):
            out.append(name)
    return tuple(out)


def supports_accessory(
    robot_key: str, accessory: str, end_effector: str | None = None, root: Path | None = None
) -> bool:
    return accessory in supported_accessories(robot_key, end_effector, root)


def assembly_ref(
    end_effector: str | None,
    *,
    visual: str = "glb",
    accessory: str | None = None,
    mounting_plate: bool = False,
    ft_sensor: str | None = None,
) -> str:
    """Canonical reference string, omitting default fields."""
    if end_effector is None and ft_sensor is None and accessory is None and not mounting_plate:
        raise AssemblyError("an assembly needs an end effector or an accessory, an F/T sensor, or a mounting plate")
    tokens = [end_effector or BARE]
    if ft_sensor:
        tokens.append(ft_sensor)
    if accessory:
        tokens.append(accessory)
    if mounting_plate:
        tokens.append(MOUNTING_PLATE)
    parts = ["+".join(tokens)]
    if visual != "glb":
        parts.append(visual)
    return ASSEMBLY_PREFIX + ":".join(parts)


def parse_assembly_ref(value: str) -> AssemblyRef | None:
    """Parse ``assembly:<end_effector|bare>[+<ft_sensor>][+<accessory>][+mounting_plate][:<visual>]``; ``None`` for a file name."""
    if not value.startswith(ASSEMBLY_PREFIX):
        return None
    fields = value[len(ASSEMBLY_PREFIX) :].split(":")
    if not fields[0] or len(fields) > 2:
        raise AssemblyError(f"malformed assembly reference {value!r}")
    tokens = fields[0].split("+")
    if any(not tok for tok in tokens):
        raise AssemblyError(f"malformed assembly reference {value!r}")
    head, *extras = tokens
    if head == MOUNTING_PLATE or head.startswith(FT_SENSOR_PREFIX) or extras.count(MOUNTING_PLATE) > 1:
        raise AssemblyError(f"malformed assembly reference {value!r}")
    has_mounting_plate = MOUNTING_PLATE in extras
    ft_tokens = [tok for tok in extras if tok.startswith(FT_SENSOR_PREFIX)]
    acc_tokens = [tok for tok in extras if tok != MOUNTING_PLATE and not tok.startswith(FT_SENSOR_PREFIX)]
    if len(ft_tokens) > 1 or len(acc_tokens) > 1:
        raise AssemblyError(f"malformed assembly reference {value!r}")
    ft_sensor = ft_tokens[0] if ft_tokens else None
    accessory = acc_tokens[0] if acc_tokens else None
    if head == BARE and ft_sensor is None and accessory is None and not has_mounting_plate:
        raise AssemblyError(f"malformed assembly reference {value!r}")
    visual = fields[1] if len(fields) > 1 else "glb"
    if visual not in VISUALS:
        raise AssemblyError(f"{value!r}: visual must be one of {VISUALS}")
    return AssemblyRef(None if head == BARE else head, visual, accessory, has_mounting_plate, ft_sensor)


def validate_assembly(
    robot_key: str,
    end_effector: str | None,
    *,
    visual: str = "glb",
    accessory: str | None = None,
    mounting_plate: bool = False,
    ft_sensor: str | None = None,
    root: Path | None = None,
) -> EndEffector | None:
    """Check that the assembly can be composed, without writing anything."""
    root = root or PROJECT_ROOT
    key = get_profile_key_for_robot_name(robot_key)
    if visual not in VISUALS:
        raise AssemblyError(f"visual must be one of {VISUALS}")
    if end_effector is None and ft_sensor is None and accessory is None and not mounting_plate:
        raise AssemblyError("an assembly needs an end effector or an accessory, an F/T sensor, or a mounting plate")
    urdf_root = root / "assets" / "urdf"
    sources = [_arm_source(key, visual, root)]
    if mounting_plate:
        plate = mounting_plate_for(key, root)
        sources.append(urdf_root / plate.source)
    if ft_sensor is not None:
        ft = get_ft_sensor(ft_sensor, root)
        if key not in ft.robots:
            raise AssemblyError(f"{ft_sensor} is not supported on {key}; supported robots: {list(ft.robots)}")
        sources.append(urdf_root / ft.source)
    ee = None
    if end_effector is not None:
        ee = get_end_effector(end_effector, root)
        if key not in ee.robots:
            raise AssemblyError(f"{end_effector} is not supported on {key}; supported robots: {list(ee.robots)}")
        if ee.adapter is not None:
            sources.append(urdf_root / get_adapter(ee.adapter, root).source)
        sources.append(urdf_root / ee.source)
    if accessory is not None:
        spacer, camera = resolve_accessory(accessory, root)
        if key not in spacer.robots:
            raise AssemblyError(f"{accessory} is not supported on {key}; supported robots: {list(spacer.robots)}")
        if end_effector is not None and end_effector not in spacer.end_effectors:
            raise AssemblyError(
                f"{accessory} does not fit with {end_effector}; supported end effectors: "
                f"{[BARE, *spacer.end_effectors]}"
            )
        sources.append(urdf_root / spacer.source)
        if camera is not None:
            sources.append(urdf_root / camera.source)
    for path in sources:
        if not path.is_file():
            raise AssemblyError(f"missing assembly source {path}")
    return ee


def _arm_source(robot_key: str, visual: str, root: Path) -> Path:
    profile = ROBOT_PROFILES[robot_key]
    name = str(profile.visual_glb_urdf if visual == "glb" else profile.default_urdf)
    return root / "assets" / "urdf" / "robots" / str(profile.assets_dir.name) / name


def _fmt(values: tuple[float, float, float]) -> str:
    return " ".join(repr(v) if v else "0" for v in values)


def _rebase_meshes(elem: ET.Element, src_dir: Path, out_dir: Path) -> None:
    for mesh in elem.iter("mesh"):
        name = mesh.get("filename")
        if not name or name.startswith(("package://", "file://")):
            continue
        target = Path(os.path.normpath(Path(name) if os.path.isabs(name) else src_dir / name))
        mesh.set("filename", Path(os.path.relpath(target, out_dir)).as_posix())


def _link_ref(joint: ET.Element, tag: str) -> str | None:
    node = joint.find(tag)
    return node.get("link") if node is not None else None


def _tool_elements(
    ee: EndEffector | Adapter | FTSensor | Spacer | Camera | MountingPlate, source: Path
) -> tuple[list[ET.Element], dict[str, ET.Element]]:
    root = ET.parse(source).getroot()
    attach = [j for j in root.findall("joint") if _link_ref(j, "child") == ee.base_link]
    if len(attach) != 1:
        raise AssemblyError(f"{source}: expected exactly one joint into {ee.base_link}")
    stub = _link_ref(attach[0], "parent")
    elems = [
        copy.deepcopy(e)
        for e in root
        if e.tag in ("link", "joint") and e is not attach[0] and not (e.tag == "link" and e.get("name") == stub)
    ]
    materials = {str(m.get("name")): m for m in root.findall("material")}
    return elems, materials


def _apply_visual_mode(elems: list[ET.Element], ee: _VisualParts, visual: str, urdf_root: Path) -> None:
    for link in (e for e in elems if e.tag == "link"):
        name = link.get("name") or ""
        override = (ee.glb_visuals if visual == "glb" else ee.stl_visuals).get(name)
        material = ee.stl_materials.get(name, ee.stl_materials.get("*")) if visual == "stl" else None
        if override is not None:
            for v in link.findall("visual"):
                mesh = v.find("geometry/mesh")
                if mesh is not None:
                    mesh.set("filename", str(urdf_root / override))
                mat = v.find("material")
                if visual == "glb" and mat is not None:
                    v.remove(mat)
                elif material and mat is None:
                    ET.SubElement(v, "material", {"name": material})
            continue
        if visual == "stl":
            for v in link.findall("visual"):
                mesh = v.find("geometry/mesh")
                if mesh is None or not (mesh.get("filename") or "").lower().endswith(".glb"):
                    continue
                index = list(link).index(v)
                link.remove(v)
                for offset, col in enumerate(link.findall("collision")):
                    nv = ET.Element("visual")
                    for child in col:
                        if child.tag in ("origin", "geometry"):
                            nv.append(copy.deepcopy(child))
                    if material:
                        ET.SubElement(nv, "material", {"name": material})
                    link.insert(index + offset, nv)


def _merge_part_materials(robot: ET.Element, elems: list[ET.Element], part_materials: Mapping[str, ET.Element]) -> None:
    """Insert the part materials its elements use that the robot does not define yet."""
    defined = {m.get("name") for m in robot.findall("material")}
    used = {name for e in elems for m in e.iter("material") if (name := m.get("name")) is not None}
    insert_at = max((i for i, e in enumerate(robot) if e.tag == "material"), default=0) + 1
    for name in sorted(n for n in used - defined if n in part_materials):
        robot.insert(insert_at, copy.deepcopy(part_materials[name]))
        insert_at += 1


def _fixed_joint(robot: ET.Element, name: str, parent: str, child: str, origin: Mapping[str, str]) -> None:
    joint = ET.SubElement(robot, "joint", {"name": name, "type": "fixed"})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    ET.SubElement(joint, "origin", dict(origin))


def _attach_part(
    robot: ET.Element,
    part: Adapter | FTSensor | Spacer | Camera | MountingPlate,
    parent: str,
    visual: str,
    urdf_root: Path,
    out_dir: Path,
    *,
    z_offset: float = 0.0,
) -> None:
    """Append an adapter, F/T sensor, spacer, camera, or mounting-plate part, fixed to ``parent`` at its mount pose."""
    source = urdf_root / part.source
    elems, materials = _tool_elements(part, source)
    _apply_visual_mode(elems, part, visual, urdf_root)
    for e in elems:
        _rebase_meshes(e, source.parent, out_dir)
    _merge_part_materials(robot, elems, materials)
    xyz, rpy = part.mount
    if z_offset:
        xyz = (xyz[0], xyz[1], xyz[2] + z_offset)
    _fixed_joint(robot, part.mount_joint, parent, part.base_link, {"xyz": _fmt(xyz), "rpy": _fmt(rpy)})
    robot.extend(elems)


def _lift_link_eef(robot: ET.Element, lift_m: float) -> None:
    """Move the arm's ``link_eef`` (when it has one) along its parent's +Z by ``lift_m``."""
    for joint in robot.findall("joint"):
        if _link_ref(joint, "child") != "link_eef":
            continue
        origin = joint.find("origin")
        if origin is None:
            origin = ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        x, y, z = (float(v) for v in (origin.get("xyz") or "0 0 0").split())
        origin.set("xyz", _fmt((x, y, z + lift_m)))


def _assembly_label(
    end_effector: str | None,
    accessory: str | None,
    mounting_plate: bool = False,
    ft_sensor: str | None = None,
) -> str:
    if ft_sensor is None:
        label = f"{end_effector or BARE}__{accessory}" if accessory else (end_effector or BARE)
        return f"{label}__{MOUNTING_PLATE}" if mounting_plate else label
    parts = [end_effector or BARE, ft_sensor]
    if accessory:
        parts.append(accessory)
    if mounting_plate:
        parts.append(MOUNTING_PLATE)
    return "__".join(parts)


def build_assembly_tree(
    robot_key: str,
    end_effector: str | None,
    *,
    visual: str = "glb",
    accessory: str | None = None,
    mounting_plate: bool = False,
    ft_sensor: str | None = None,
    root: Path | None = None,
    out_dir: Path | None = None,
) -> ET.Element:
    """Return the composed ``<robot>`` element with mesh paths relative to ``out_dir``."""
    root = (root or PROJECT_ROOT).resolve()
    out_dir = (out_dir or root / COMPOSED_DIR).resolve()
    key = get_profile_key_for_robot_name(robot_key)
    ee = validate_assembly(
        key,
        end_effector,
        visual=visual,
        accessory=accessory,
        mounting_plate=mounting_plate,
        ft_sensor=ft_sensor,
        root=root,
    )
    urdf_root = root / "assets" / "urdf"
    arm_file = _arm_source(key, visual, root)
    flange = get_robot_profile(key).ee_link
    plate = mounting_plate_for(key, root) if mounting_plate else None
    ft = get_ft_sensor(ft_sensor, root) if ft_sensor is not None else None
    spacer, camera = resolve_accessory(accessory, root) if accessory is not None else (None, None)
    adapter = get_adapter(ee.adapter, root) if ee is not None and ee.adapter is not None else None

    sources = [arm_file]
    if plate is not None:
        sources.append(urdf_root / plate.source)
    if ft is not None:
        sources.append(urdf_root / ft.source)
    if spacer is not None:
        sources.append(urdf_root / spacer.source)
    if camera is not None:
        sources.append(urdf_root / camera.source)
    if adapter is not None:
        sources.append(urdf_root / adapter.source)
    if ee is not None:
        sources.append(urdf_root / ee.source)
    label = _assembly_label(end_effector, accessory, mounting_plate, ft_sensor)
    robot = ET.Element("robot", {"name": f"{key}__{label}__{visual}"})
    robot.append(
        ET.Comment(
            f" Generated by ufactory.robots.assembly v{COMPOSER_VERSION} from "
            f"{' + '.join(p.relative_to(root).as_posix() for p in sources)}; do not edit. "
        )
    )
    for e in ET.parse(arm_file).getroot():
        e = copy.deepcopy(e)
        _rebase_meshes(e, arm_file.parent, out_dir)
        robot.append(e)

    if plate is not None:
        _attach_part(robot, plate, "link_base", visual, urdf_root, out_dir)

    tool_parent = flange
    local_lift = 0.0
    total_lift = 0.0
    if ft is not None:
        _attach_part(robot, ft, flange, visual, urdf_root, out_dir)
        if robot.find(f"./link[@name='{ft.output_link}']") is None:
            raise AssemblyError(f"{ft.name}: output_link {ft.output_link!r} missing from {ft.source}")
        tool_parent = ft.output_link
        total_lift += ft.tool_lift_m

    if spacer is not None:
        _attach_part(robot, spacer, tool_parent, visual, urdf_root, out_dir, z_offset=local_lift)
        local_lift += spacer.tool_lift_m
        total_lift += spacer.tool_lift_m
        if camera is not None:
            _attach_part(robot, camera, spacer.base_link, visual, urdf_root, out_dir)

    if adapter is not None:
        _attach_part(robot, adapter, tool_parent, visual, urdf_root, out_dir, z_offset=local_lift)
        local_lift += adapter.tool_lift_m
        total_lift += adapter.tool_lift_m

    if ee is not None:
        tool_file = urdf_root / ee.source
        elems, tool_materials = _tool_elements(ee, tool_file)
        _apply_visual_mode(elems, ee, visual, urdf_root)
        for e in elems:
            _rebase_meshes(e, tool_file.parent, out_dir)
        _merge_part_materials(robot, elems, tool_materials)

        xyz, rpy = ee.mount
        if local_lift:
            xyz = (xyz[0], xyz[1], xyz[2] + local_lift)
        origin = {"xyz": _fmt(xyz), "rpy": _fmt(rpy)}
        _fixed_joint(robot, ee.mount_joint, tool_parent, ee.base_link, origin)
        robot.extend(elems)
    elif total_lift:
        _lift_link_eef(robot, total_lift)

    names = [e.get("name") for e in robot if e.tag in ("link", "joint")]
    duplicates = sorted({n for n in names if names.count(n) > 1 and n is not None})
    if duplicates:
        raise AssemblyError(f"{key} + {label}: duplicate link/joint names {duplicates}")
    return robot


def serialize_assembly(robot: ET.Element) -> bytes:
    tree = ET.ElementTree(robot)
    ET.indent(tree, space="  ")
    body: bytes = ET.tostring(robot, encoding="utf-8")
    return b'<?xml version="1.0" encoding="utf-8"?>\n' + body + b"\n"


def _write_composed_urdf(out_dir: Path, stem: str, tree: ET.Element) -> str:
    data = serialize_assembly(tree)
    digest = hashlib.sha256(data).hexdigest()[:12]
    path = out_dir / f"{stem}.{digest}.urdf"
    if not path.is_file() or path.read_bytes() != data:
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
    return str(path)


def _build_standalone_part_tree(
    part: EndEffector | Adapter | FTSensor | MountingPlate,
    *,
    mount_pose: Pose,
    visual: str = "glb",
    root: Path | None = None,
    out_dir: Path | None = None,
) -> ET.Element:
    """Return a standalone ``<robot>`` element (rooted at ``world``) for one part."""
    root = (root or PROJECT_ROOT).resolve()
    out_dir = (out_dir or root / COMPOSED_DIR).resolve()
    if visual not in VISUALS:
        raise AssemblyError(f"visual must be one of {VISUALS}")
    urdf_root = root / "assets" / "urdf"
    part_file = urdf_root / part.source
    if not part_file.is_file():
        raise AssemblyError(f"missing assembly source {part_file}")

    elems, part_materials = _tool_elements(part, part_file)
    _apply_visual_mode(elems, part, visual, urdf_root)
    for elem in elems:
        _rebase_meshes(elem, part_file.parent, out_dir)

    robot = ET.Element("robot", {"name": f"standalone__{part.name}__{visual}"})
    robot.append(
        ET.Comment(
            f" Generated by ufactory.robots.assembly v{COMPOSER_VERSION} from "
            f"{part_file.relative_to(root).as_posix()}; do not edit. "
        )
    )
    _merge_part_materials(robot, elems, part_materials)
    ET.SubElement(robot, "link", {"name": "world"})
    xyz, rpy = mount_pose
    _fixed_joint(robot, part.mount_joint, "world", part.base_link, {"xyz": _fmt(xyz), "rpy": _fmt(rpy)})
    robot.extend(elems)
    return robot


def build_standalone_end_effector_tree(
    end_effector: str,
    *,
    visual: str = "glb",
    root: Path | None = None,
    out_dir: Path | None = None,
) -> ET.Element:
    """Return a standalone ``<robot>`` element (rooted at ``world``) for one end effector."""
    resolved_root = (root or PROJECT_ROOT).resolve()
    ee = get_end_effector(end_effector, resolved_root)
    return _build_standalone_part_tree(
        ee,
        mount_pose=ee.mount,
        visual=visual,
        root=resolved_root,
        out_dir=out_dir,
    )


def compose_standalone_end_effector_urdf(
    end_effector: str,
    *,
    visual: str = "glb",
    root: Path | None = None,
    out_dir: Path | None = None,
) -> str:
    """Compose (or reuse) a standalone end-effector URDF with visuals resolved from ``assembly.yaml``."""
    resolved_root = (root or PROJECT_ROOT).resolve()
    resolved_out = (out_dir or resolved_root / COMPOSED_DIR).resolve()
    tree = build_standalone_end_effector_tree(
        end_effector,
        visual=visual,
        root=resolved_root,
        out_dir=resolved_out,
    )
    ee = get_end_effector(end_effector, resolved_root)
    return _write_composed_urdf(resolved_out, f"standalone__{ee.name}__{visual}", tree)


def build_standalone_adapter_tree(
    name: str,
    *,
    visual: str = "glb",
    root: Path | None = None,
    out_dir: Path | None = None,
) -> ET.Element:
    """Return a standalone ``<robot>`` element (rooted at ``world``) for one flange adapter."""
    resolved_root = (root or PROJECT_ROOT).resolve()
    adapter = get_adapter(name, resolved_root)
    return _build_standalone_part_tree(
        adapter,
        mount_pose=adapter.mount,
        visual=visual,
        root=resolved_root,
        out_dir=out_dir,
    )


def compose_standalone_adapter_urdf(
    name: str,
    *,
    visual: str = "glb",
    root: Path | None = None,
    out_dir: Path | None = None,
) -> str:
    """Compose (or reuse) a standalone flange-adapter URDF with visuals resolved from ``assembly.yaml``."""
    resolved_root = (root or PROJECT_ROOT).resolve()
    resolved_out = (out_dir or resolved_root / COMPOSED_DIR).resolve()
    tree = build_standalone_adapter_tree(
        name,
        visual=visual,
        root=resolved_root,
        out_dir=resolved_out,
    )
    adapter = get_adapter(name, resolved_root)
    return _write_composed_urdf(resolved_out, f"standalone__{adapter.name}__{visual}", tree)


def build_standalone_ft_sensor_tree(
    name: str,
    *,
    visual: str = "glb",
    root: Path | None = None,
    out_dir: Path | None = None,
) -> ET.Element:
    """Return a standalone ``<robot>`` element (rooted at ``world``) for one F/T sensor."""
    resolved_root = (root or PROJECT_ROOT).resolve()
    sensor = get_ft_sensor(name, resolved_root)
    return _build_standalone_part_tree(
        sensor,
        mount_pose=sensor.mount,
        visual=visual,
        root=resolved_root,
        out_dir=out_dir,
    )


def compose_standalone_ft_sensor_urdf(
    name: str,
    *,
    visual: str = "glb",
    root: Path | None = None,
    out_dir: Path | None = None,
) -> str:
    """Compose (or reuse) a standalone F/T sensor URDF with visuals resolved from ``assembly.yaml``."""
    resolved_root = (root or PROJECT_ROOT).resolve()
    resolved_out = (out_dir or resolved_root / COMPOSED_DIR).resolve()
    tree = build_standalone_ft_sensor_tree(
        name,
        visual=visual,
        root=resolved_root,
        out_dir=resolved_out,
    )
    sensor = get_ft_sensor(name, resolved_root)
    return _write_composed_urdf(resolved_out, f"standalone__{sensor.name}__{visual}", tree)


def build_standalone_mounting_plate_tree(
    name: str,
    *,
    visual: str = "glb",
    root: Path | None = None,
    out_dir: Path | None = None,
) -> ET.Element:
    """Return a standalone ``<robot>`` element (rooted at ``world``) for one mounting plate."""
    resolved_root = (root or PROJECT_ROOT).resolve()
    plate = get_mounting_plate(name, resolved_root)
    return _build_standalone_part_tree(
        plate,
        mount_pose=IDENTITY,
        visual=visual,
        root=resolved_root,
        out_dir=out_dir,
    )


def compose_standalone_mounting_plate_urdf(
    name: str,
    *,
    visual: str = "glb",
    root: Path | None = None,
    out_dir: Path | None = None,
) -> str:
    """Compose (or reuse) a standalone mounting-plate URDF with visuals resolved from ``assembly.yaml``."""
    resolved_root = (root or PROJECT_ROOT).resolve()
    resolved_out = (out_dir or resolved_root / COMPOSED_DIR).resolve()
    tree = build_standalone_mounting_plate_tree(
        name,
        visual=visual,
        root=resolved_root,
        out_dir=resolved_out,
    )
    plate = get_mounting_plate(name, resolved_root)
    return _write_composed_urdf(resolved_out, f"standalone__{plate.name}__{visual}", tree)


def compose_robot_urdf(
    robot_key: str,
    end_effector: str | None,
    *,
    visual: str = "glb",
    accessory: str | None = None,
    mounting_plate: bool = False,
    ft_sensor: str | None = None,
    root: Path | None = None,
    out_dir: Path | None = None,
) -> str:
    """Compose (or reuse) the assembled URDF and return its absolute path.

    ``end_effector`` may be ``None`` (bare arm) when ``ft_sensor``, ``accessory``, or ``mounting_plate=True`` is given.
    """
    root = (root or PROJECT_ROOT).resolve()
    out_dir = (out_dir or root / COMPOSED_DIR).resolve()
    key = get_profile_key_for_robot_name(robot_key)
    tree = build_assembly_tree(
        key,
        end_effector,
        visual=visual,
        accessory=accessory,
        mounting_plate=mounting_plate,
        ft_sensor=ft_sensor,
        root=root,
        out_dir=out_dir,
    )
    return _write_composed_urdf(
        out_dir,
        f"{key}__{_assembly_label(end_effector, accessory, mounting_plate, ft_sensor)}__{visual}",
        tree,
    )


def resolve_robot_urdf(
    robot_key: str,
    urdf: str,
    *,
    assets_dir: str | Path | None = None,
    store: _AssetStore | None = None,
) -> Path:
    """Resolve a configured URDF value: an assembly reference is composed, a file name is looked up."""
    root = store.root if store is not None else PROJECT_ROOT
    ref = parse_assembly_ref(urdf)
    if ref is not None:
        return Path(
            compose_robot_urdf(
                robot_key,
                ref.end_effector,
                visual=ref.visual,
                accessory=ref.accessory,
                mounting_plate=ref.mounting_plate,
                ft_sensor=ref.ft_sensor,
                root=root,
            )
        )
    directory = Path(assets_dir) if assets_dir is not None else get_robot_profile(robot_key).assets_dir
    if store is not None and not directory.is_absolute():
        return store.require(directory / urdf)
    path = (directory if directory.is_absolute() else root / directory) / urdf
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def validate_robot_urdf(
    robot_key: str,
    urdf: str,
    *,
    assets_dir: str | Path,
    store: _AssetStore,
) -> None:
    """Validate a configured URDF value without composing (no files are written)."""
    ref = parse_assembly_ref(urdf)
    if ref is None:
        store.require(Path(assets_dir) / urdf)
        return
    validate_assembly(
        robot_key,
        ref.end_effector,
        visual=ref.visual,
        accessory=ref.accessory,
        mounting_plate=ref.mounting_plate,
        ft_sensor=ref.ft_sensor,
        root=store.root,
    )
