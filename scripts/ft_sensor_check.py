#!/usr/bin/env python3
"""Diagnostic and post-installation verification CLI for the xArm six-axis F/T sensor (AI1500).

Enable / disable F/T sampling (no physical arm motion):
    python scripts/ft_sensor_check.py enable --robot xarm6 --ip <ROBOT_IP>
    python scripts/ft_sensor_check.py disable --robot xarm6 --ip <ROBOT_IP>

Read-only status & wrench streaming:
    python scripts/ft_sensor_check.py status --robot xarm6 --ip <ROBOT_IP> --duration 5

Stationary zeroing:
    python scripts/ft_sensor_check.py zero --robot xarm6 --ip <ROBOT_IP>

Load & zero-offset identification (moves the physical arm -> requires --confirm-real):
    python scripts/ft_sensor_check.py identify-load --robot xarm6 --ip <ROBOT_IP> --confirm-real

Conservative admittance / force-control verification (requires --confirm-real):
    python scripts/ft_sensor_check.py admittance-test --robot xarm6 --ip <ROBOT_IP> --confirm-real
    python scripts/ft_sensor_check.py force-test --robot xarm6 --ip <ROBOT_IP> --confirm-real
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

from ufactory.hardware.ft_sensor import (
    FTAdmittanceParams,
    FTForceControlParams,
    FTSensorReading,
    XArmFTSensor,
)
from ufactory.hardware.xarm import verify_controller_robot
from ufactory.robots.assembly import supported_ft_sensors
from ufactory.robots.registry import ROBOT_PROFILES, get_profile_key_for_robot_name


def _supported_robot_choices() -> list[str]:
    choices: set[str] = set()
    for key, profile in ROBOT_PROFILES.items():
        if supported_ft_sensors(key):
            choices.add(key)
            choices.add(profile.robot_name)
    return sorted(choices)


def _default_arm_factory(ip: str) -> Any:
    from xarm.wrapper import XArmAPI  # type: ignore[import-untyped]

    arm = XArmAPI(ip, is_radian=True)
    time.sleep(0.5)
    if not getattr(arm, "connected", False):
        raise ConnectionError(f"Failed to connect to controller at {ip}")
    return arm


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--robot", required=True, choices=_supported_robot_choices(), help="Supported robot model")
    parser.add_argument("--ip", required=True, help="xArm controller IP address")
    parser.add_argument("--json", action="store_true", help="Print structured JSON output")


def _sample_stream(
    sensor: XArmFTSensor,
    *,
    duration_s: float,
    rate_hz: float,
    is_raw: bool,
    from_report: bool,
    as_json: bool,
) -> list[dict[str, Any]]:
    if duration_s < 0.0:
        raise ValueError("duration must be non-negative")
    if rate_hz <= 0.0:
        raise ValueError("rate_hz must be positive")
    period_s = 1.0 / rate_hz
    start = time.monotonic()
    samples: list[dict[str, Any]] = []
    step = 0
    while True:
        reading: FTSensorReading = sensor.read_wrench(is_raw=is_raw, from_report=from_report)
        row = {
            "t_s": round(time.monotonic() - start, 4),
            "wrench": list(reading.wrench),
            "overrange": list(reading.overrange),
            "overload": list(reading.overload),
        }
        samples.append(row)
        if not as_json:
            flags = " OVERLOAD" if reading.any_overload else (" OVERRANGE" if reading.any_overrange else "")
            vals = ", ".join(f"{v:8.3f}" for v in reading.wrench)
            print(f"[{row['t_s']:6.2f}s] [{vals}]{flags}")
        if duration_s <= 0.0 or (time.monotonic() - start) >= duration_s:
            break
        step += 1
        target = start + step * period_s
        remaining = target - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)
    return samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Read-only F/T status, config, and optional wrench stream")
    _add_common_args(p_status)
    p_status.add_argument("--duration", type=float, default=0.0, help="Stream duration in seconds (0 = single reading)")
    p_status.add_argument("--rate-hz", type=float, default=20.0, help="Polling frequency in Hz")
    p_status.add_argument("--raw", action="store_true", help="Read raw uncompensated wrench instead of external wrench")
    p_status.add_argument("--from-report", action="store_true", help="Read cached socket report attributes")
    p_status.add_argument("--include-identity", action="store_true", help="Query F/T sensor SN and firmware version")

    p_enable = sub.add_parser("enable", help="Enable F/T sensor data sampling (no physical arm motion)")
    _add_common_args(p_enable)

    p_disable = sub.add_parser("disable", help="Exit force mode and disable F/T sensor sampling")
    _add_common_args(p_disable)

    p_zero = sub.add_parser("zero", help="Enable sensor and zero current static offset (stationary)")
    _add_common_args(p_zero)

    p_id = sub.add_parser("identify-load", help="Identify tool load and offset (drives robot motion)")
    _add_common_args(p_id)
    p_id.add_argument("--confirm-real", action="store_true", help="Required confirmation to move physical robot")
    p_id.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    p_id.add_argument("--apply", action="store_true", help="Write identified load offset back to controller")
    p_id.add_argument("--update-tcp-load", action="store_true", help="Also update controller TCP load when applying")

    p_adm = sub.add_parser("admittance-test", help="Conservative admittance compliance check (requires --confirm-real)")
    _add_common_args(p_adm)
    p_adm.add_argument("--confirm-real", action="store_true", help="Required confirmation to enable F/T motion")
    p_adm.add_argument("--coord", type=int, choices=(0, 1), default=1, help="0=base frame, 1=tool frame")
    p_adm.add_argument("--c-axis", type=int, nargs=6, default=[1, 1, 1, 0, 0, 0])
    p_adm.add_argument("--mass", type=float, nargs=6, default=[0.1, 0.1, 0.1, 0.002, 0.002, 0.002])
    p_adm.add_argument("--stiffness", type=float, nargs=6, default=[300.0, 300.0, 300.0, 4.0, 4.0, 4.0])
    p_adm.add_argument("--damping", type=float, nargs=6, default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    p_adm.add_argument("--duration", type=float, default=5.0, help="Hold duration in seconds")
    p_adm.add_argument("--rate-hz", type=float, default=20.0, help="Logging rate in Hz")
    p_adm.add_argument("--zero", action="store_true", help="Zero sensor before entering admittance mode")
    p_adm.add_argument("--disable-on-exit", action="store_true", help="Disable sensor sampling after exiting mode")

    p_force = sub.add_parser("force-test", help="Conservative constant-force check (requires --confirm-real)")
    _add_common_args(p_force)
    p_force.add_argument("--confirm-real", action="store_true", help="Required confirmation to enable F/T motion")
    p_force.add_argument("--coord", type=int, choices=(0, 1), default=1, help="0=base frame, 1=tool frame")
    p_force.add_argument("--c-axis", type=int, nargs=6, default=[0, 0, 1, 0, 0, 0])
    p_force.add_argument("--f-ref", type=float, nargs=6, default=[0.0, 0.0, -2.0, 0.0, 0.0, 0.0])
    p_force.add_argument("--limits", type=float, nargs=6, default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    p_force.add_argument("--kp", type=float, nargs=6, default=[0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
    p_force.add_argument("--ki", type=float, nargs=6, default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    p_force.add_argument("--kd", type=float, nargs=6, default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    p_force.add_argument("--xe-limit", type=float, nargs=6, default=[10.0, 10.0, 10.0, 0.2, 0.2, 0.2])
    p_force.add_argument("--duration", type=float, default=2.0, help="Hold duration in seconds")
    p_force.add_argument("--rate-hz", type=float, default=20.0, help="Logging rate in Hz")
    p_force.add_argument("--zero", action="store_true", help="Zero sensor before entering force mode")
    p_force.add_argument("--disable-on-exit", action="store_true", help="Disable sensor sampling after exiting mode")

    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    arm_factory: Callable[[str], Any] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    robot_key = get_profile_key_for_robot_name(args.robot)

    if args.command in ("identify-load", "admittance-test", "force-test") and not args.confirm_real:
        raise RuntimeError(f"Subcommand '{args.command}' requires --confirm-real before connecting to the controller")

    # Pre-validate force/admittance parameters before opening a hardware socket.
    adm_params: FTAdmittanceParams | None = None
    force_params: FTForceControlParams | None = None
    if args.command == "admittance-test":
        adm_params = FTAdmittanceParams(
            coord=args.coord,
            c_axis=tuple(args.c_axis),
            M=tuple(args.mass),
            K=tuple(args.stiffness),
            B=tuple(args.damping),
        )
    elif args.command == "force-test":
        force_params = FTForceControlParams(
            coord=args.coord,
            c_axis=tuple(args.c_axis),
            f_ref=tuple(args.f_ref),
            limits=tuple(args.limits),
            kp=tuple(args.kp),
            ki=tuple(args.ki),
            kd=tuple(args.kd),
            xe_limit=tuple(args.xe_limit),
        )

    factory = _default_arm_factory if arm_factory is None else arm_factory
    arm = factory(args.ip)
    try:
        verify_controller_robot(arm, robot_key)
        sensor = XArmFTSensor(arm, robot_key=robot_key)
        payload: dict[str, Any] = {"robot_key": robot_key, "ip": args.ip, "command": args.command}

        if args.command == "status":
            status = sensor.read_status(include_identity=bool(args.include_identity))
            cfg = sensor.read_config() if callable(getattr(arm, "get_ft_sensor_config", None)) else None
            payload["status"] = asdict(status)
            if cfg is not None:
                payload["config"] = asdict(cfg)
            if not args.json:
                print(
                    f"robot={robot_key} mode={status.mode} enabled={status.enabled} "
                    f"error={status.error_code} ({status.error_message})"
                )
                if status.serial_number or status.firmware_version:
                    print(f"sn={status.serial_number} version={status.firmware_version}")
            payload["samples"] = _sample_stream(
                sensor,
                duration_s=args.duration,
                rate_hz=args.rate_hz,
                is_raw=args.raw,
                from_report=args.from_report,
                as_json=args.json,
            )

        elif args.command == "enable":
            sensor.set_enable(True)
            time.sleep(0.2)
            status = sensor.read_status()
            reading = sensor.read_wrench()
            payload["status"] = asdict(status)
            payload["wrench"] = list(reading.wrench)
            if not args.json:
                print(
                    f"Enabled F/T sensor: robot={robot_key} mode={status.mode} enabled={status.enabled} "
                    f"error={status.error_code} ({status.error_message}) wrench={reading.wrench}"
                )

        elif args.command == "disable":
            sensor.stop_force_mode(disable_sensor=True)
            time.sleep(0.1)
            status = sensor.read_status()
            payload["status"] = asdict(status)
            if not args.json:
                print(
                    f"Disabled F/T sensor: robot={robot_key} mode={status.mode} enabled={status.enabled} "
                    f"error={status.error_code} ({status.error_message})"
                )

        elif args.command == "zero":
            sensor.set_enable(True)
            time.sleep(0.2)
            before = sensor.read_wrench()
            sensor.zero()
            after = sensor.read_wrench()
            payload["before"] = list(before.wrench)
            payload["after"] = list(after.wrench)
            if not args.json:
                print(f"Zeroed F/T sensor: before={before.wrench} after={after.wrench}")

        elif args.command == "identify-load":
            sensor.set_enable(True)
            offset = sensor.identify_load_offset(confirm_real=True)
            if args.apply:
                sensor.set_load_offset(offset, update_tcp_load=bool(args.update_tcp_load))
            payload["load_offset"] = asdict(offset)
            payload["applied"] = bool(args.apply)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            if not args.json:
                print(f"Identified load offset: {offset}")

        elif args.command == "admittance-test":
            assert adm_params is not None
            with sensor.admittance_session(
                adm_params,
                confirm_real=True,
                zero_on_enter=bool(args.zero),
                disable_on_exit=bool(args.disable_on_exit),
            ):
                payload["samples"] = _sample_stream(
                    sensor,
                    duration_s=args.duration,
                    rate_hz=args.rate_hz,
                    is_raw=False,
                    from_report=False,
                    as_json=args.json,
                )

        elif args.command == "force-test":
            assert force_params is not None
            with sensor.force_control_session(
                force_params,
                confirm_real=True,
                zero_on_enter=bool(args.zero),
                disable_on_exit=bool(args.disable_on_exit),
            ):
                payload["samples"] = _sample_stream(
                    sensor,
                    duration_s=args.duration,
                    rate_hz=args.rate_hz,
                    is_raw=False,
                    from_report=False,
                    as_json=args.json,
                )

        if args.json:
            print(json.dumps(payload, indent=2))
        return 0
    finally:
        arm.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
