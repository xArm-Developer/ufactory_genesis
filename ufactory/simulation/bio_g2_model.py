"""Bio Gripper G2 SDK units, calibrated rigid-body parameters, and force/speed mappings.

This module has no Genesis or hardware connection. Simulations load the project URDF directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

BIO_G2_PHYSICS_PROFILE = "bio_g2_calibrated_v1"

BIO_G2_KP = 16000.0
BIO_G2_KV = 40.0
BIO_G2_KI = 4000.0
BIO_G2_RATE_HZ = 50
BIO_G2_DT = 0.02
BIO_G2_SUBSTEPS = 32
BIO_G2_CONSTRAINT_TIMECONST = 0.005

MASS_KG = 0.79
TOTAL_MASS_KG = MASS_KG
BASE_MASS_KG = 0.765827
LEFT_FINGER_MASS_KG = 0.012105
RIGHT_FINGER_MASS_KG = 0.012068
BASE_INERTIA_SCALE = BASE_MASS_KG / 0.38837

MIN_GAP_MM = 71.0
MAX_GAP_MM = 150.0
CLOSED_GAP_MM = MIN_GAP_MM
OPEN_GAP_MM = MAX_GAP_MM
STROKE_M = (MAX_GAP_MM - MIN_GAP_MM) / 2000.0  # 0.0395 m per finger
STROKE_PER_FINGER_M = STROKE_M

MIN_SPEED_RPM = 500.0
MAX_SPEED_RPM = 4000.0
DEFAULT_SPEED_RPM = 2000.0

MIN_SPEED_MM_S = MIN_SPEED_RPM * 5.0 / 60.0  # 41.67 mm/s per finger
MAX_SPEED_MM_S = MAX_SPEED_RPM * 5.0 / 60.0  # 333.33 mm/s per finger
DEFAULT_SPEED_MM_S = DEFAULT_SPEED_RPM * 5.0 / 60.0  # 166.67 mm/s per finger
MIN_GAP_SPEED_MM_S = 2.0 * MIN_SPEED_MM_S  # 83.33 mm/s bilateral jaw speed
MAX_GAP_SPEED_MM_S = 2.0 * MAX_SPEED_MM_S  # 666.67 mm/s bilateral jaw speed
DEFAULT_GAP_SPEED_MM_S = 2.0 * DEFAULT_SPEED_MM_S  # 333.33 mm/s bilateral jaw speed
MAX_JOINT_SPEED_M_S = MAX_SPEED_MM_S / 1000.0
SPEEDS_MM_S: tuple[float, ...] = tuple(rpm * 5.0 / 60.0 for rpm in (500, 1000, 2000, 3000, 4000))

MIN_FORCE_N = 4.0
MAX_FORCE_N = 20.0
FORCE_CURVE_PERCENTS: tuple[float, ...] = (1.0, 25.0, 50.0, 75.0, 100.0)
FORCE_CURVE_N: tuple[float, ...] = (4.0, 8.0, 12.0, 16.0, 20.0)

RIGHT = "bio_gripper_g2_right_finger_joint"
LEFT = "bio_gripper_g2_left_finger_joint"
RIGHT_JOINT = RIGHT
LEFT_JOINT = LEFT
BIO_G2_JOINT_NAMES: tuple[str, str] = (RIGHT_JOINT, LEFT_JOINT)

# Y-axis visual offsets aligning the vendor GLB finger inner gripping planes to +/-35.5 mm (71 mm closed gap).
GLB_FINGER_VISUAL_Y_OFFSET_M: dict[str, float] = {
    "bio_gripper_g2_left_finger": 0.0069565922021865845,
    "bio_gripper_g2_right_finger": -0.008117303252220154,
}

# In bio_gripper_g2_base_link frame, the validated planar gripping ridge center is at (0.128, 0, 0.050).
GRASP_OFFSET_IN_BASE_LINK: tuple[float, float, float] = (0.128, 0.0, 0.050)


def bounded(value: float, lo: float, hi: float, name: str) -> float:
    """Validate that *value* is a finite float within ``[lo, hi]``."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite, in [{lo}, {hi}]")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite, in [{lo}, {hi}]") from exc
    if not math.isfinite(numeric) or not lo <= numeric <= hi:
        raise ValueError(f"{name} must be finite, in [{lo}, {hi}], got {numeric}")
    return numeric


checked = bounded


def gap_to_q(position_mm: float) -> float:
    """Convert two-finger jaw gap (mm) in ``[71, 150]`` to right-finger prismatic joint position (m)."""
    return (bounded(position_mm, MIN_GAP_MM, MAX_GAP_MM, "position_mm") - MIN_GAP_MM) / 2000.0


def q_to_gap(q_right: float, q_left: float | None = None) -> float:
    """Convert finger prismatic joint position(s) (m) to two-finger jaw gap (mm)."""
    return MIN_GAP_MM + (2.0 * float(q_right) if q_left is None else float(q_right) - float(q_left)) * 1000.0


def finger_force_n(force_percent: float) -> float:
    """Map SDK force percentage ``[1, 100]`` via the calibrated 5-point curve to single-finger force ``[4, 20]`` N."""
    pct = bounded(force_percent, 1.0, 100.0, "force_percent")
    if pct <= 25.0:
        return MIN_FORCE_N + (pct - 1.0) * (4.0 / 24.0)
    return 8.0 + (pct - 25.0) * 0.16


def drive_limit_n(force_percent: float) -> float:
    """Total symmetric drive force limit (N) via virtual work: ``dW = F*dq_right - F*dq_left = 2F*dq_s``."""
    return 2.0 * finger_force_n(force_percent)


def speed_mm_s_from_rpm(speed_rpm: float) -> float:
    """Convert SDK motor speed (rpm) in ``[500, 4000]`` to single-finger speed (mm/s)."""
    rpm = bounded(speed_rpm, MIN_SPEED_RPM, MAX_SPEED_RPM, "speed_rpm")
    return rpm * 5.0 / 60.0


def speed_rpm_from_mm_s(speed_mm_s: float) -> float:
    """Convert single-finger speed (mm/s) in ``[41.666..., 333.333...]`` to SDK motor speed (rpm)."""
    speed = bounded(speed_mm_s, MIN_SPEED_MM_S, MAX_SPEED_MM_S, "speed_mm_s")
    return speed * 60.0 / 5.0


def bio_g2_speed_mm_s_from_percent(speed_percent: float) -> float:
    """Convert percentage speed ``[0, 100]`` to single-finger speed (mm/s).

    ``0%`` pauses reference motion (``0.0 mm/s``); ``(0, 100]%`` scales linearly
    with ``MAX_SPEED_MM_S`` (so ``50%`` equals ``DEFAULT_SPEED_MM_S = 166.666... mm/s``
    per finger / ``333.333... mm/s`` bilateral jaw speed / ``2000 rpm``) and clamps to
    ``MIN_SPEED_MM_S`` (``500 rpm``) at the low end.
    """
    percent = bounded(speed_percent, 0.0, 100.0, "speed_percent")
    if percent == 0.0:
        return 0.0
    return max(MIN_SPEED_MM_S, (percent / 100.0) * MAX_SPEED_MM_S)


def bio_g2_speed_percent_from_mm_s(speed_mm_s: float) -> float:
    """Convert single-finger speed (mm/s) to percentage ``[0, 100]``."""
    if isinstance(speed_mm_s, bool):
        raise ValueError("speed_mm_s must be finite")
    numeric = float(speed_mm_s)
    if numeric == 0.0:
        return 0.0
    speed = bounded(numeric, MIN_SPEED_MM_S, MAX_SPEED_MM_S, "speed_mm_s")
    return min(100.0, max(1.0, (speed / MAX_SPEED_MM_S) * 100.0))


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class BioGripperG2Model:
    """Loaded Bio Gripper G2 URDF metadata and calibrated physical parameters."""

    urdf_path: Path
    urdf_sha256: str
    total_mass_kg: float
    link_masses_kg: dict[str, float]
    min_gap_mm: float = MIN_GAP_MM
    max_gap_mm: float = MAX_GAP_MM
    min_speed_mm_s: float = MIN_SPEED_MM_S
    max_speed_mm_s: float = MAX_SPEED_MM_S
    default_speed_mm_s: float = DEFAULT_SPEED_MM_S
    min_force_n: float = MIN_FORCE_N
    max_force_n: float = MAX_FORCE_N

    @classmethod
    @lru_cache(maxsize=4)
    def load(cls, urdf_path: str | Path | None = None) -> BioGripperG2Model:
        if urdf_path is None:
            default_path = (
                Path(__file__).resolve().parents[2]
                / "assets"
                / "urdf"
                / "end_effectors"
                / "bio_gripper_g2"
                / "bio_gripper_g2_movable_visual.urdf"
            )
            path = default_path.resolve()
        else:
            path = Path(urdf_path).resolve()
        root = ET.parse(path).getroot()
        masses: dict[str, float] = {}
        for link in root.findall("link"):
            name = link.get("name") or ""
            mass_node = link.find("inertial/mass")
            if name.startswith("bio_gripper_g2_") and mass_node is not None:
                masses[name] = float(mass_node.get("value", "0"))
        return cls(
            urdf_path=path,
            urdf_sha256=sha256(path),
            total_mass_kg=sum(masses.values()),
            link_masses_kg=masses,
        )


def bio_g2_physics_summary() -> dict:
    model = BioGripperG2Model.load()
    payload = asdict(model)
    payload["urdf_path"] = str(model.urdf_path)
    payload["profile"] = BIO_G2_PHYSICS_PROFILE
    payload["kp_n_m"] = BIO_G2_KP
    payload["kv_n_s_m"] = BIO_G2_KV
    payload["ki_n_m_s"] = BIO_G2_KI
    return payload
