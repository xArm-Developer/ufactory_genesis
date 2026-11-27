"""Real UFACTORY six-axis Force/Torque sensor (AI1500) SDK control and safety wrapper."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import math
import time
from typing import Any, Protocol

import numpy as np

from ufactory.hardware.xarm import MODE_POSITION, STATE_MOTION, assert_motion_ready, prepare_arm_for_motion
from ufactory.robots.assembly import supports_ft_sensor
from ufactory.robots.registry import get_profile_key_for_robot_name

# F/T controller application modes (XArmAPI.set_ft_sensor_mode).
FT_MODE_OFF = 0
FT_MODE_ADMITTANCE = 1
FT_MODE_FORCE = 2
FT_MODES: tuple[int, ...] = (FT_MODE_OFF, FT_MODE_ADMITTANCE, FT_MODE_FORCE)

# F/T task coordinate frames.
FT_COORD_BASE = 0
FT_COORD_TOOL = 1
FT_COORDS: tuple[int, ...] = (FT_COORD_BASE, FT_COORD_TOOL)

# AI1500 official hardware specifications & Studio defaults.
AI1500_NAME = "ft_sensor_ai1500"
AI1500_TOOL_OFFSET_Z_MM = 58.6
AI1500_MASS_KG = 0.6
AI1500_COM_FLANGE_MM: tuple[float, float, float] = (0.0, 0.0, 22.0)
AI1500_SAMPLE_RATE_HZ = 200.0
AI1500_RESOLUTION: tuple[float, float, float, float, float, float] = (0.4, 0.4, 0.4, 0.01, 0.01, 0.01)
AI1500_RANGE: tuple[float, float, float, float, float, float] = (400.0, 400.0, 400.0, 20.0, 20.0, 20.0)
AI1500_OVERLOAD: tuple[float, float, float, float, float, float] = (600.0, 600.0, 600.0, 30.0, 30.0, 30.0)

# xArm Python SDK (v1.18.4) parameter bounds.
ADMITTANCE_M_LINEAR_BOUNDS: tuple[float, float] = (0.02, 1.0)
ADMITTANCE_M_ANGULAR_BOUNDS: tuple[float, float] = (0.0001, 0.01)
ADMITTANCE_K_LINEAR_BOUNDS: tuple[float, float] = (0.0, 2000.0)
ADMITTANCE_K_ANGULAR_BOUNDS: tuple[float, float] = (0.0, 20.0)
ADMITTANCE_THRESHOLD_FORCE_BOUNDS: tuple[float, float] = (0.1, 50.0)
ADMITTANCE_THRESHOLD_TORQUE_BOUNDS: tuple[float, float] = (0.01, 2.0)

FORCE_REF_LIMITS: tuple[float, float, float, float, float, float] = (150.0, 150.0, 200.0, 4.0, 4.0, 4.0)
FORCE_KP_BOUNDS: tuple[float, float] = (0.0, 0.05)
FORCE_KI_BOUNDS: tuple[float, float] = (0.0, 0.0005)
FORCE_KD_BOUNDS: tuple[float, float] = (0.0, 0.05)
FORCE_XE_LIMIT_BOUNDS: tuple[float, float] = (0.0, 200.0)

COLLISION_FORCE_THRESHOLD_BOUNDS: tuple[float, float] = (5.0, 200.0)
COLLISION_TORQUE_THRESHOLD_BOUNDS: tuple[float, float] = (0.1, 4.0)
COLLISION_REBOUND_POS_MM_BOUNDS: tuple[float, float] = (2.0, 500.0)
COLLISION_REBOUND_ROT_DEG_BOUNDS: tuple[float, float] = (0.2, 50.0)

# Mirrors xarm.core.config.x_code.FtSensorErrorCodeMap.
FT_SENSOR_ERROR_DESCRIPTIONS: dict[int, str] = {
    0: "Normal",
    64: "Six-axis Force Torque Sensor Communication Failure",
    65: "The Data Collected by the Six-axis Force Torque Sensor is Abnormal",
    66: "Six-axis Force Torque Sensor Fx Exceeds Limit",
    67: "Six-axis Force Torque Sensor Fy Exceeds Limit",
    68: "Six-axis Force Torque Sensor Fz Exceeds Limit",
    69: "Six-axis Force Torque Sensor Tx Exceeds Limit",
    70: "Six-axis Force Torque Sensor Ty Exceeds Limit",
    71: "Six-axis Force Torque Sensor Tz Exceeds Limit",
    73: "Six-axis Force Torque Sensor Failed to Initialize",
}


class FTSensorSDK(Protocol):
    """Minimal structural protocol covering XArmAPI F/T sensor attributes and methods."""

    state: int
    mode: int
    error_code: int
    warn_code: int

    @property
    def ft_raw_force(self) -> Sequence[float]: ...

    @property
    def ft_ext_force(self) -> Sequence[float]: ...

    def get_state(self) -> tuple[int, int]: ...

    def motion_enable(self, enable: bool = True, **kwargs: Any) -> int: ...

    def set_mode(self, mode: int = 0, **kwargs: Any) -> int: ...

    def set_state(self, state: int = 0) -> int: ...

    def set_ft_sensor_enable(self, on_off: int | bool) -> int: ...

    def set_ft_sensor_mode(self, mode: int, **kwargs: Any) -> int: ...

    def get_ft_sensor_mode(self) -> tuple[int, int]: ...

    def set_ft_sensor_zero(self) -> int: ...

    def iden_ft_sensor_load_offset(self) -> tuple[int, Sequence[float]]: ...

    def set_ft_sensor_load_offset(
        self,
        iden_result_list: list[float],
        association_setting_tcp_load: bool = False,
        **kwargs: Any,
    ) -> int: ...

    def set_ft_sensor_admittance_parameters(
        self,
        coord: int | None = None,
        c_axis: Sequence[int] | None = None,
        M: Sequence[float] | None = None,
        K: Sequence[float] | None = None,
        B: Sequence[float] | None = None,
        **kwargs: Any,
    ) -> int: ...

    def set_ft_sensor_force_parameters(
        self,
        coord: int | None = None,
        c_axis: Sequence[int] | None = None,
        f_ref: Sequence[float] | None = None,
        limits: Sequence[float] | None = None,
        kp: Sequence[float] | None = None,
        ki: Sequence[float] | None = None,
        kd: Sequence[float] | None = None,
        xe_limit: Sequence[float] | None = None,
        **kwargs: Any,
    ) -> int: ...

    def get_ft_sensor_data(self, is_raw: bool = False) -> tuple[int, Sequence[float]]: ...

    def get_ft_sensor_config(self) -> tuple[int, Sequence[Any]]: ...

    def get_ft_sensor_error(self) -> tuple[int, int]: ...

    def get_ft_sensor_sn(self) -> tuple[int, str]: ...

    def get_ft_sensor_version(self) -> tuple[int, str]: ...

    def set_ft_collision_detection(self, on_off: int | bool) -> int: ...

    def get_ft_collision_detection(self) -> tuple[int, int]: ...

    def set_ft_collision_threshold(self, thresholds: list[float]) -> int: ...

    def get_ft_collision_threshold(self) -> tuple[int, Sequence[float]]: ...

    def set_ft_collision_rebound(self, on_off: int | bool) -> int: ...

    def get_ft_collision_rebound(self) -> tuple[int, int]: ...

    def set_ft_collision_reb_distance(self, distances: list[float], is_radian: bool | None = None) -> int: ...

    def get_ft_collision_reb_distance(self, is_radian: bool | None = None) -> tuple[int, Sequence[float]]: ...

    def set_ft_admittance_ctrl_threshold(self, thresholds: list[float]) -> int: ...

    def get_ft_admittance_ctrl_threshold(self) -> tuple[int, Sequence[float]]: ...


def describe_ft_sensor_error(code: int) -> str:
    """Return a human-readable message for an xArm six-axis F/T sensor error code."""
    if isinstance(code, bool) or not isinstance(code, int):
        raise ValueError(f"error code must be an int, got {code!r}")
    if code in FT_SENSOR_ERROR_DESCRIPTIONS:
        return FT_SENSOR_ERROR_DESCRIPTIONS[code]
    return f"Six-axis Force Torque Sensor Error (code={code})"


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, not bool")
    num = float(value)
    if not math.isfinite(num):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return num


def _finite_vector(values: Sequence[Any], length: int, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of {length} finite floats")
    seq = tuple(_finite_float(v, f"{name}[{i}]") for i, v in enumerate(values))
    if len(seq) != length:
        raise ValueError(f"{name} must have length {length}, got {len(seq)}")
    return seq


def _bounded_vector(
    values: Sequence[Any],
    bounds: Sequence[tuple[float, float]],
    name: str,
) -> tuple[float, ...]:
    vec = _finite_vector(values, len(bounds), name)
    for i, (val, (lo, hi)) in enumerate(zip(vec, bounds, strict=True)):
        if not (lo <= val <= hi):
            raise ValueError(f"{name}[{i}]={val} outside [{lo}, {hi}]")
    return vec


def _binary_axis_mask(values: Sequence[Any], name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a 6-element binary mask")
    items = list(values)
    if len(items) != 6:
        raise ValueError(f"{name} must have length 6, got {len(items)}")
    out: list[int] = []
    for i, val in enumerate(items):
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
            raise ValueError(f"{name}[{i}] must be integer 0 or 1, got {val!r}")
        ival = int(val)
        if ival not in (0, 1):
            raise ValueError(f"{name}[{i}] must be 0 or 1, got {ival}")
        out.append(ival)
    return tuple(out)


def _check_int_choice(value: Any, allowed: tuple[int, ...], name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer in {allowed}, got {value!r}")
    ival = int(value)
    if ival not in allowed:
        raise ValueError(f"{name} must be in {allowed}, got {ival}")
    return ival


def _check_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _check_sdk_code(label: str, code: Any) -> None:
    if int(code) != 0:
        raise RuntimeError(f"{label} failed with SDK code {int(code)}")


def _unwrap_sdk_tuple(label: str, result: Any) -> Any:
    if not isinstance(result, (tuple, list)) or len(result) < 2:
        raise RuntimeError(f"{label} returned unexpected response {result!r}")
    _check_sdk_code(label, result[0])
    return result[1]


@dataclass(frozen=True)
class FTSensorReading:
    """Six-axis wrench reading ``[Fx, Fy, Fz, Tx, Ty, Tz]`` in ``[N, Nm]``."""

    wrench: tuple[float, float, float, float, float, float]
    is_raw: bool
    monotonic_ns: int
    overrange: tuple[bool, bool, bool, bool, bool, bool]
    overload: tuple[bool, bool, bool, bool, bool, bool]

    def __post_init__(self) -> None:
        vec = _finite_vector(self.wrench, 6, "wrench")
        object.__setattr__(self, "wrench", vec)
        if len(self.overrange) != 6 or not all(isinstance(v, bool) for v in self.overrange):
            raise ValueError("overrange must contain 6 booleans")
        if len(self.overload) != 6 or not all(isinstance(v, bool) for v in self.overload):
            raise ValueError("overload must contain 6 booleans")

    @classmethod
    def from_wrench(
        cls,
        wrench: Sequence[float],
        *,
        is_raw: bool,
        monotonic_ns: int | None = None,
    ) -> FTSensorReading:
        vec = _finite_vector(wrench, 6, "wrench")
        overrange = tuple(abs(val) > limit for val, limit in zip(vec, AI1500_RANGE, strict=True))
        overload = tuple(abs(val) >= limit for val, limit in zip(vec, AI1500_OVERLOAD, strict=True))
        return cls(
            wrench=(vec[0], vec[1], vec[2], vec[3], vec[4], vec[5]),
            is_raw=bool(is_raw),
            monotonic_ns=time.monotonic_ns() if monotonic_ns is None else int(monotonic_ns),
            overrange=(overrange[0], overrange[1], overrange[2], overrange[3], overrange[4], overrange[5]),
            overload=(overload[0], overload[1], overload[2], overload[3], overload[4], overload[5]),
        )

    @property
    def force_n(self) -> tuple[float, float, float]:
        return self.wrench[:3]

    @property
    def torque_nm(self) -> tuple[float, float, float]:
        return self.wrench[3:]

    @property
    def any_overrange(self) -> bool:
        return any(self.overrange)

    @property
    def any_overload(self) -> bool:
        return any(self.overload)

    def as_array(self) -> np.ndarray:
        arr: np.ndarray = np.asarray(self.wrench, dtype=np.float64)
        return arr


@dataclass(frozen=True)
class FTLoadOffset:
    """Identified or configured tool load & sensor zero-offset parameters."""

    mass_kg: float
    centroid_mm: tuple[float, float, float]
    wrench_offset: tuple[float, float, float, float, float, float]

    def __post_init__(self) -> None:
        mass = _finite_float(self.mass_kg, "mass_kg")
        if mass < 0.0:
            raise ValueError(f"mass_kg must be non-negative, got {mass}")
        centroid = _finite_vector(self.centroid_mm, 3, "centroid_mm")
        offset = _finite_vector(self.wrench_offset, 6, "wrench_offset")
        object.__setattr__(self, "mass_kg", mass)
        object.__setattr__(self, "centroid_mm", (centroid[0], centroid[1], centroid[2]))
        object.__setattr__(
            self,
            "wrench_offset",
            (offset[0], offset[1], offset[2], offset[3], offset[4], offset[5]),
        )

    @classmethod
    def from_sdk_list(cls, values: Sequence[float]) -> FTLoadOffset:
        vec = _finite_vector(values, 10, "load_offset")
        return cls(
            mass_kg=vec[0],
            centroid_mm=(vec[1], vec[2], vec[3]),
            wrench_offset=(vec[4], vec[5], vec[6], vec[7], vec[8], vec[9]),
        )

    def to_sdk_list(self) -> list[float]:
        return [self.mass_kg, *self.centroid_mm, *self.wrench_offset]


@dataclass(frozen=True)
class FTAdmittanceParams:
    """Parameters for ``XArmAPI.set_ft_sensor_admittance_parameters`` (``mode=1``)."""

    coord: int
    c_axis: tuple[int, ...]
    M: tuple[float, ...]
    K: tuple[float, ...]
    B: tuple[float, ...]

    def __post_init__(self) -> None:
        coord = _check_int_choice(self.coord, FT_COORDS, "coord")
        c_axis = _binary_axis_mask(self.c_axis, "c_axis")
        m_bounds = (ADMITTANCE_M_LINEAR_BOUNDS,) * 3 + (ADMITTANCE_M_ANGULAR_BOUNDS,) * 3
        k_bounds = (ADMITTANCE_K_LINEAR_BOUNDS,) * 3 + (ADMITTANCE_K_ANGULAR_BOUNDS,) * 3
        b_bounds = ((0.0, math.inf),) * 6
        object.__setattr__(self, "coord", coord)
        object.__setattr__(self, "c_axis", c_axis)
        object.__setattr__(self, "M", _bounded_vector(self.M, m_bounds, "M"))
        object.__setattr__(self, "K", _bounded_vector(self.K, k_bounds, "K"))
        object.__setattr__(self, "B", _bounded_vector(self.B, b_bounds, "B"))


@dataclass(frozen=True)
class FTForceControlParams:
    """Parameters for ``XArmAPI.set_ft_sensor_force_parameters`` (``mode=2``)."""

    coord: int
    c_axis: tuple[int, ...]
    f_ref: tuple[float, ...]
    limits: tuple[float, ...]
    kp: tuple[float, ...]
    ki: tuple[float, ...]
    kd: tuple[float, ...]
    xe_limit: tuple[float, ...]

    def __post_init__(self) -> None:
        coord = _check_int_choice(self.coord, FT_COORDS, "coord")
        c_axis = _binary_axis_mask(self.c_axis, "c_axis")
        f_ref_bounds = tuple((-lim, lim) for lim in FORCE_REF_LIMITS)
        limit_bounds = ((0.0, math.inf),) * 6
        object.__setattr__(self, "coord", coord)
        object.__setattr__(self, "c_axis", c_axis)
        object.__setattr__(self, "f_ref", _bounded_vector(self.f_ref, f_ref_bounds, "f_ref"))
        object.__setattr__(self, "limits", _bounded_vector(self.limits, limit_bounds, "limits"))
        object.__setattr__(self, "kp", _bounded_vector(self.kp, (FORCE_KP_BOUNDS,) * 6, "kp"))
        object.__setattr__(self, "ki", _bounded_vector(self.ki, (FORCE_KI_BOUNDS,) * 6, "ki"))
        object.__setattr__(self, "kd", _bounded_vector(self.kd, (FORCE_KD_BOUNDS,) * 6, "kd"))
        object.__setattr__(self, "xe_limit", _bounded_vector(self.xe_limit, (FORCE_XE_LIMIT_BOUNDS,) * 6, "xe_limit"))


@dataclass(frozen=True)
class FTCollisionConfig:
    """F/T collision detection and rebound configuration snapshot."""

    enabled: bool
    thresholds: tuple[float, ...]
    rebound_enabled: bool
    rebound_distance: tuple[float, ...]
    is_radian: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "rebound_enabled", bool(self.rebound_enabled))
        object.__setattr__(self, "thresholds", _finite_vector(self.thresholds, 6, "collision_thresholds"))
        object.__setattr__(self, "rebound_distance", _finite_vector(self.rebound_distance, 6, "rebound_distance"))
        object.__setattr__(self, "is_radian", _check_bool(self.is_radian, "is_radian"))


@dataclass(frozen=True)
class FTSensorConfigSnapshot:
    """Typed snapshot of the 22-element list returned by ``get_ft_sensor_config()``."""

    ft_mode: int
    ft_enabled: bool
    ft_type: int
    ft_id: int
    ft_freq: int
    ft_mass_kg: float
    ft_dir_bias: float
    ft_centroid: tuple[float, float, float]
    ft_zero: tuple[float, float, float, float, float, float]
    imp_coord: int
    imp_c_axis: tuple[int, ...]
    M: tuple[float, ...]
    K: tuple[float, ...]
    B: tuple[float, ...]
    f_coord: int
    f_c_axis: tuple[int, ...]
    f_ref: tuple[float, ...]
    f_limits: tuple[float, ...]
    kp: tuple[float, ...]
    ki: tuple[float, ...]
    kd: tuple[float, ...]
    xe_limit: tuple[float, ...]

    @classmethod
    def from_sdk_config(cls, config: Sequence[Any]) -> FTSensorConfigSnapshot:
        if len(config) < 22:
            raise ValueError(f"expected at least 22 elements from get_ft_sensor_config(), got {len(config)}")
        centroid = _finite_vector(config[7], 3, "ft_centroid")
        zero = _finite_vector(config[8], 6, "ft_zero")
        return cls(
            ft_mode=int(config[0]),
            ft_enabled=bool(config[1]),
            ft_type=int(config[2]),
            ft_id=int(config[3]),
            ft_freq=int(config[4]),
            ft_mass_kg=_finite_float(config[5], "ft_mass"),
            ft_dir_bias=_finite_float(config[6], "ft_dir_bias"),
            ft_centroid=(centroid[0], centroid[1], centroid[2]),
            ft_zero=(zero[0], zero[1], zero[2], zero[3], zero[4], zero[5]),
            imp_coord=int(config[9]),
            imp_c_axis=tuple(int(v) for v in config[10][:6]),
            M=_finite_vector(config[11][:6], 6, "M"),
            K=_finite_vector(config[12][:6], 6, "K"),
            B=_finite_vector(config[13][:6], 6, "B"),
            f_coord=int(config[14]),
            f_c_axis=tuple(int(v) for v in config[15][:6]),
            f_ref=_finite_vector(config[16][:6], 6, "f_ref"),
            f_limits=_finite_vector(config[17][:6], 6, "f_limits"),
            kp=_finite_vector(config[18][:6], 6, "kp"),
            ki=_finite_vector(config[19][:6], 6, "ki"),
            kd=_finite_vector(config[20][:6], 6, "kd"),
            xe_limit=_finite_vector(config[21][:6], 6, "xe_limit"),
        )


@dataclass(frozen=True)
class FTSensorStatus:
    """High-level status of the attached F/T sensor."""

    mode: int
    error_code: int
    error_message: str
    enabled: bool | None = None
    serial_number: str | None = None
    firmware_version: str | None = None


def validate_admittance_threshold(thresholds: Sequence[float]) -> tuple[float, ...]:
    bounds = (ADMITTANCE_THRESHOLD_FORCE_BOUNDS,) * 3 + (ADMITTANCE_THRESHOLD_TORQUE_BOUNDS,) * 3
    return _bounded_vector(thresholds, bounds, "admittance_threshold")


def validate_collision_thresholds(thresholds: Sequence[float]) -> tuple[float, ...]:
    bounds = (COLLISION_FORCE_THRESHOLD_BOUNDS,) * 3 + (COLLISION_TORQUE_THRESHOLD_BOUNDS,) * 3
    return _bounded_vector(thresholds, bounds, "collision_thresholds")


def validate_collision_rebound_distance(distances: Sequence[float], *, is_radian: bool = True) -> tuple[float, ...]:
    lo_deg, hi_deg = COLLISION_REBOUND_ROT_DEG_BOUNDS
    rot_bounds = (math.radians(lo_deg), math.radians(hi_deg)) if is_radian else (lo_deg, hi_deg)
    bounds = (COLLISION_REBOUND_POS_MM_BOUNDS,) * 3 + (rot_bounds,) * 3
    return _bounded_vector(distances, bounds, "rebound_distance")


class XArmFTSensor:
    """Fail-closed wrapper around XArmAPI six-axis Force/Torque sensor commands.

    Physical activation (``identify_load_offset`` and entering active force/admittance
    modes ``1`` or ``2``) requires ``confirm_real=True`` and a fault-free controller
    plus fault-free F/T sensor. It never calls ``clean_error()`` or ``clean_warn()``.
    """

    def __init__(
        self,
        arm: FTSensorSDK | Any,
        *,
        robot_key: str | None = None,
        sensor_name: str = AI1500_NAME,
    ) -> None:
        if robot_key is not None:
            resolved_key = get_profile_key_for_robot_name(robot_key)
            if not supports_ft_sensor(resolved_key, sensor_name):
                raise ValueError(f"Robot {resolved_key} does not support F/T sensor {sensor_name}")
            self.robot_key: str | None = resolved_key
        else:
            self.robot_key = None
        self.arm = arm
        self.sensor_name = sensor_name

    def _assert_healthy_for_active_ft(self) -> None:
        assert_motion_ready(self.arm)
        if int(getattr(self.arm, "warn_code", 0)) != 0:
            raise RuntimeError(
                f"Controller has active warning warn_code={getattr(self.arm, 'warn_code', None)}; "
                "inspect and recover manually in UFACTORY Studio"
            )
        err_code, err_msg = self.read_error()
        if err_code != 0:
            raise RuntimeError(f"F/T sensor has active error {err_code}: {err_msg}")

    # --- Read-only queries (no motion, no confirm_real needed) ---

    def read_config(self) -> FTSensorConfigSnapshot:
        raw_cfg = _unwrap_sdk_tuple("get_ft_sensor_config", self.arm.get_ft_sensor_config())
        return FTSensorConfigSnapshot.from_sdk_config(raw_cfg)

    def read_mode(self) -> int:
        mode = _unwrap_sdk_tuple("get_ft_sensor_mode", self.arm.get_ft_sensor_mode())
        return _check_int_choice(mode, FT_MODES, "ft_mode")

    def read_error(self) -> tuple[int, str]:
        err = _unwrap_sdk_tuple("get_ft_sensor_error", self.arm.get_ft_sensor_error())
        code = int(err)
        return code, describe_ft_sensor_error(code)

    def _resolve_sdk_method(self, name: str) -> Any:
        """Resolve an F/T SDK method on ``self.arm`` or fallback to ``self.arm._arm``.

        ``xarm-python-sdk`` 1.18.4 implements ``get_ft_sensor_sn`` and
        ``get_ft_sensor_version`` on the underlying ``XArm`` instance (``arm._arm``)
        without forwarding them on ``XArmAPI``.
        """
        fn = getattr(self.arm, name, None)
        if callable(fn):
            return fn
        inner = getattr(self.arm, "_arm", None)
        inner_fn = getattr(inner, name, None) if inner is not None else None
        if callable(inner_fn):
            return inner_fn
        raise RuntimeError(f"Connected xArm SDK object does not expose {name}()")

    def read_serial_number(self) -> str:
        sn = _unwrap_sdk_tuple("get_ft_sensor_sn", self._resolve_sdk_method("get_ft_sensor_sn")())
        text = str(sn).strip()
        if not text or set(text) == {"*"}:
            raise RuntimeError(f"Invalid F/T sensor serial number readback: {sn!r}")
        return text

    def read_firmware_version(self) -> str:
        ver = _unwrap_sdk_tuple("get_ft_sensor_version", self._resolve_sdk_method("get_ft_sensor_version")())
        text = str(ver).strip()
        if not text:
            raise RuntimeError(f"Invalid F/T sensor version readback: {ver!r}")
        return text

    def read_status(self, *, include_identity: bool = False) -> FTSensorStatus:
        mode = self.read_mode()
        err_code, err_msg = self.read_error()
        enabled: bool | None = None
        config_getter = getattr(self.arm, "get_ft_sensor_config", None)
        if callable(config_getter):
            enabled = self.read_config().ft_enabled
        serial_number = self.read_serial_number() if include_identity else None
        firmware_version = self.read_firmware_version() if include_identity else None
        return FTSensorStatus(
            mode=mode,
            error_code=err_code,
            error_message=err_msg,
            enabled=enabled,
            serial_number=serial_number,
            firmware_version=firmware_version,
        )

    def read_wrench(self, *, is_raw: bool = False, from_report: bool = False) -> FTSensorReading:
        raw_flag = _check_bool(is_raw, "is_raw")
        report_flag = _check_bool(from_report, "from_report")
        if report_flag:
            values = self.arm.ft_raw_force if raw_flag else self.arm.ft_ext_force
        else:
            values = _unwrap_sdk_tuple("get_ft_sensor_data", self.arm.get_ft_sensor_data(is_raw=raw_flag))
        return FTSensorReading.from_wrench(values, is_raw=raw_flag)

    def read_admittance_threshold(self) -> tuple[float, ...]:
        values = _unwrap_sdk_tuple(
            "get_ft_admittance_ctrl_threshold",
            self.arm.get_ft_admittance_ctrl_threshold(),
        )
        return _finite_vector(values[:6], 6, "admittance_threshold")

    def read_collision_config(self, *, is_radian: bool = True) -> FTCollisionConfig:
        rad_flag = _check_bool(is_radian, "is_radian")
        enabled = bool(_unwrap_sdk_tuple("get_ft_collision_detection", self.arm.get_ft_collision_detection()))
        thresholds = _unwrap_sdk_tuple("get_ft_collision_threshold", self.arm.get_ft_collision_threshold())
        rebound_enabled = bool(_unwrap_sdk_tuple("get_ft_collision_rebound", self.arm.get_ft_collision_rebound()))
        rebound_distance = _unwrap_sdk_tuple(
            "get_ft_collision_reb_distance",
            self.arm.get_ft_collision_reb_distance(is_radian=rad_flag),
        )
        return FTCollisionConfig(
            enabled=enabled,
            thresholds=_finite_vector(thresholds[:6], 6, "collision_thresholds"),
            rebound_enabled=rebound_enabled,
            rebound_distance=_finite_vector(rebound_distance[:6], 6, "rebound_distance"),
            is_radian=rad_flag,
        )

    # --- Sensor enable, zeroing & load identification ---

    def set_enable(self, enabled: bool) -> None:
        flag = _check_bool(enabled, "enabled")
        _check_sdk_code(f"set_ft_sensor_enable({flag})", self.arm.set_ft_sensor_enable(flag))

    def zero(self, *, settle_s: float = 0.2) -> None:
        """Zero static F/T bias and wait briefly for controller write-through.

        Note: In xArm firmware, ``set_ft_sensor_zero`` overwrites prior sensor zero
        offset and payload compensation; official SDK examples wait ``0.2 s`` after
        calling it so the updated offset takes effect before reading or activating
        force modes.
        """
        wait_s = _finite_float(settle_s, "settle_s")
        if wait_s < 0.0:
            raise ValueError(f"settle_s must be non-negative, got {wait_s}")
        self._assert_healthy_for_active_ft()
        _check_sdk_code("set_ft_sensor_zero", self.arm.set_ft_sensor_zero())
        if wait_s > 0.0:
            time.sleep(wait_s)

    def identify_load_offset(self, *, confirm_real: bool = False) -> FTLoadOffset:
        if not confirm_real:
            raise RuntimeError("identify_load_offset drives physical robot poses; pass confirm_real=True to execute")
        self._assert_healthy_for_active_ft()
        prepare_arm_for_motion(self.arm, mode=MODE_POSITION)
        raw = self.arm.iden_ft_sensor_load_offset()
        if isinstance(raw, (tuple, list)) and len(raw) >= 1 and int(raw[0]) == 10:
            raise RuntimeError(
                "iden_ft_sensor_load_offset failed with SDK code 10 (RET_IS_INVALID); "
                "ensure a rigid external payload is mounted (bare sensor mass ~0 kg can make centroid "
                "identification ill-conditioned) and the arm has workspace clearance"
            )
        values = _unwrap_sdk_tuple("iden_ft_sensor_load_offset", raw)
        return FTLoadOffset.from_sdk_list(values)

    def set_load_offset(self, offset: FTLoadOffset, *, update_tcp_load: bool = False) -> None:
        """Write identified F/T load/zero offset to the controller.

        Note: ``xarm-python-sdk`` 1.18.4's ``association_setting_tcp_load=True`` uses
        hardcoded legacy-sensor mass/CoM defaults (0.270 kg) with mixed m/mm units; for
        AI1500 keep ``update_tcp_load=False`` and configure controller TCP load/offset
        explicitly.
        """
        flag = _check_bool(update_tcp_load, "update_tcp_load")
        code = self.arm.set_ft_sensor_load_offset(offset.to_sdk_list(), association_setting_tcp_load=flag)
        _check_sdk_code("set_ft_sensor_load_offset", code)

    # --- Force / admittance / collision configuration & mode switching ---

    def configure_admittance(
        self,
        params: FTAdmittanceParams,
        *,
        reaction_threshold: Sequence[float] | None = None,
    ) -> None:
        validated_threshold = (
            validate_admittance_threshold(reaction_threshold) if reaction_threshold is not None else None
        )
        _check_sdk_code(
            "set_ft_sensor_admittance_parameters",
            self.arm.set_ft_sensor_admittance_parameters(
                coord=params.coord,
                c_axis=list(params.c_axis),
                M=list(params.M),
                K=list(params.K),
                B=list(params.B),
            ),
        )
        if validated_threshold is not None:
            _check_sdk_code(
                "set_ft_admittance_ctrl_threshold",
                self.arm.set_ft_admittance_ctrl_threshold(list(validated_threshold)),
            )

    def configure_force_control(self, params: FTForceControlParams) -> None:
        _check_sdk_code(
            "set_ft_sensor_force_parameters",
            self.arm.set_ft_sensor_force_parameters(
                coord=params.coord,
                c_axis=list(params.c_axis),
                f_ref=list(params.f_ref),
                limits=list(params.limits),
                kp=list(params.kp),
                ki=list(params.ki),
                kd=list(params.kd),
                xe_limit=list(params.xe_limit),
            ),
        )

    def configure_collision_detection(
        self,
        *,
        enabled: bool | None = None,
        thresholds: Sequence[float] | None = None,
        rebound: bool | None = None,
        rebound_distance: Sequence[float] | None = None,
        is_radian: bool = True,
    ) -> None:
        rad_flag = _check_bool(is_radian, "is_radian")
        validated_thresholds = validate_collision_thresholds(thresholds) if thresholds is not None else None
        validated_distance = (
            validate_collision_rebound_distance(rebound_distance, is_radian=rad_flag)
            if rebound_distance is not None
            else None
        )
        enabled_flag = _check_bool(enabled, "enabled") if enabled is not None else None
        rebound_flag = _check_bool(rebound, "rebound") if rebound is not None else None

        if validated_thresholds is not None:
            _check_sdk_code(
                "set_ft_collision_threshold", self.arm.set_ft_collision_threshold(list(validated_thresholds))
            )
        if validated_distance is not None:
            _check_sdk_code(
                "set_ft_collision_reb_distance",
                self.arm.set_ft_collision_reb_distance(list(validated_distance), is_radian=rad_flag),
            )
        if rebound_flag is not None:
            _check_sdk_code("set_ft_collision_rebound", self.arm.set_ft_collision_rebound(rebound_flag))
        if enabled_flag is not None:
            _check_sdk_code("set_ft_collision_detection", self.arm.set_ft_collision_detection(enabled_flag))

    def set_mode(self, mode: int, *, confirm_real: bool = False) -> None:
        target_mode = _check_int_choice(mode, FT_MODES, "mode")
        if target_mode != FT_MODE_OFF:
            if not confirm_real:
                raise RuntimeError(
                    f"Entering active F/T mode {target_mode} allows physical motion; pass confirm_real=True"
                )
            self._assert_healthy_for_active_ft()
            prepare_arm_for_motion(self.arm, mode=MODE_POSITION)
        _check_sdk_code(f"set_ft_sensor_mode({target_mode})", self.arm.set_ft_sensor_mode(target_mode))
        if target_mode != FT_MODE_OFF:
            # Switching into admittance/force mode leaves controller in state=5 until set_state(0).
            _check_sdk_code("set_state(0)", self.arm.set_state(STATE_MOTION))

    def stop_force_mode(self, *, disable_sensor: bool = False) -> None:
        disable_flag = _check_bool(disable_sensor, "disable_sensor")
        mode_code = self.arm.set_ft_sensor_mode(FT_MODE_OFF)
        disable_code = 0
        if disable_flag:
            disable_code = self.arm.set_ft_sensor_enable(False)
        _check_sdk_code("set_ft_sensor_mode(0)", mode_code)
        if disable_flag:
            _check_sdk_code("set_ft_sensor_enable(False)", disable_code)

    @contextmanager
    def admittance_session(
        self,
        params: FTAdmittanceParams,
        *,
        confirm_real: bool = False,
        reaction_threshold: Sequence[float] | None = None,
        zero_on_enter: bool = False,
        disable_on_exit: bool = False,
    ) -> Iterator[XArmFTSensor]:
        """Enter admittance control safely and always exit back to ``FT_MODE_OFF``."""
        if not confirm_real:
            raise RuntimeError("admittance_session enables physical motion; pass confirm_real=True")
        zero_flag = _check_bool(zero_on_enter, "zero_on_enter")
        disable_flag = _check_bool(disable_on_exit, "disable_on_exit")
        self._assert_healthy_for_active_ft()
        self.set_enable(True)
        time.sleep(0.2)
        try:
            if zero_flag:
                self.zero()
            self.configure_admittance(params, reaction_threshold=reaction_threshold)
            self.set_mode(FT_MODE_ADMITTANCE, confirm_real=True)
            yield self
        finally:
            self.stop_force_mode(disable_sensor=disable_flag)

    @contextmanager
    def force_control_session(
        self,
        params: FTForceControlParams,
        *,
        confirm_real: bool = False,
        zero_on_enter: bool = False,
        disable_on_exit: bool = False,
    ) -> Iterator[XArmFTSensor]:
        """Enter force control safely and always exit back to ``FT_MODE_OFF``."""
        if not confirm_real:
            raise RuntimeError("force_control_session enables physical motion; pass confirm_real=True")
        zero_flag = _check_bool(zero_on_enter, "zero_on_enter")
        disable_flag = _check_bool(disable_on_exit, "disable_on_exit")
        self._assert_healthy_for_active_ft()
        self.set_enable(True)
        time.sleep(0.2)
        try:
            if zero_flag:
                self.zero()
            self.configure_force_control(params)
            self.set_mode(FT_MODE_FORCE, confirm_real=True)
            yield self
        finally:
            self.stop_force_mode(disable_sensor=disable_flag)
