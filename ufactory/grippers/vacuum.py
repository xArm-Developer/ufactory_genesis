"""xArm / UF850 vacuum gripper SDK adapter and TGPIO pin mapping."""

from __future__ import annotations

_VACUUM_IO_PINS: dict[int, tuple[int, int]] = {
    1: (0, 1),
    2: (3, 4),
}

VACUUM_GRIPPER_SUPPORTED_ROBOTS: frozenset[str] = frozenset({"xarm5", "xarm6", "xarm7", "uf850"})
VACUUM_GRIPPER_TCP_OFFSET_MM_DEG: tuple[float, float, float, float, float, float] = (
    0.0,
    0.0,
    126.0,
    0.0,
    0.0,
    0.0,
)
VACUUM_GRIPPER_MASS_KG: float = 0.61
VACUUM_GRIPPER_COM_FLANGE_MM: tuple[float, float, float] = (0.0, 0.0, 53.0)


def vacuum_io_pins(hardware_version: int) -> tuple[int, int]:
    """Return ``(on_pin, off_pin)`` TGPIO numbers for vacuum gripper ``hardware_version`` (1 or 2)."""
    if isinstance(hardware_version, bool) or not isinstance(hardware_version, int):
        raise ValueError(f"hardware_version must be integer 1 or 2, got {hardware_version!r}")
    try:
        return _VACUUM_IO_PINS[hardware_version]
    except KeyError as exc:
        raise ValueError(f"unsupported vacuum gripper hardware_version={hardware_version}; expected 1 or 2") from exc


def vacuum_controller_mode_error(task: str, robot: str, gripper: str, mode: str) -> str | None:
    """Return an error message when a vacuum gripper CLI mode/robot combination is unsupported."""
    if gripper != "vacuum_gripper":
        return None
    if robot not in VACUUM_GRIPPER_SUPPORTED_ROBOTS:
        return "--gripper vacuum_gripper requires --robot xarm5, xarm6, xarm7 or uf850"
    if mode == "sim":
        return (
            f"vacuum_gripper {task} does not model suction physics for --mode sim; "
            "use --mode dry-run, --mode sdk-sim, or --mode real (optionally with --visual)"
        )
    return None


class VacuumGripperAdapter:
    """Adapter for xArm / UF850 pneumatic vacuum gripper driven over TGPIO."""

    def __init__(self, *, hardware_version: int = 1) -> None:
        self.on_pin, self.off_pin = vacuum_io_pins(hardware_version)
        self.hardware_version = int(hardware_version)

    def prepare_real(self, arm: object) -> int:
        missing = [
            name for name in ("set_vacuum_gripper", "get_vacuum_gripper") if not callable(getattr(arm, name, None))
        ]
        if missing:
            raise RuntimeError(f"Vacuum gripper SDK capability is unavailable: {', '.join(missing)}")
        code, _state = getattr(arm, "get_vacuum_gripper")(hardware_version=self.hardware_version)
        return int(code)

    def send_real_suction(self, arm: object, on: bool, *, sync: bool | None = None) -> int:
        if isinstance(on, bool):
            suction_on = on
        elif isinstance(on, (int, float)) and on in (0, 1):
            suction_on = bool(on)
        else:
            raise ValueError(f"on must be a boolean, got {on!r}")
        method = getattr(arm, "set_vacuum_gripper")
        if sync is None:
            return int(method(suction_on, wait=False, hardware_version=self.hardware_version))
        return int(method(suction_on, wait=False, sync=bool(sync), hardware_version=self.hardware_version))

    def read_real(self, arm: object) -> tuple[int, int]:
        code, state = getattr(arm, "get_vacuum_gripper")(hardware_version=self.hardware_version)
        return int(code), int(state)

    def read_real_output(self, arm: object) -> tuple[int, tuple[int, int]]:
        """Read ``(on_pin, off_pin)`` TGPIO output levels without relying on object-suction detection."""
        code, outputs = getattr(arm, "get_tgpio_output_digital")()
        if int(code) != 0:
            return int(code), (-1, -1)
        values = list(outputs)
        max_pin = max(self.on_pin, self.off_pin)
        if len(values) <= max_pin:
            raise RuntimeError(f"get_tgpio_output_digital returned {len(values)} pins; expected > {max_pin}")
        return 0, (int(values[self.on_pin]), int(values[self.off_pin]))
