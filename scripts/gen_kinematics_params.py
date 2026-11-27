#!/usr/bin/env python3
"""Export per-unit kinematic calibration from an xArm controller.

The joint origins are identified from the controller's own forward kinematics
(read-only ``get_forward_kinematics`` queries: the zero pose and single-joint
sweeps, then 100 random configurations to verify the result). Nothing is
enabled or moved. The file is written only if it reproduces the controller FK
within the FK validation limits (1 mm, 0.5 deg).

Output: the strict calibration schema, bound to the full robot serial number:
  joints.joint1..N: {x, y, z, roll, pitch, yaw}  (meters, radians)

SN eligibility (no compensation file expected):
  - xArm 5/6/7: SN positions 3-6 < 1304
  - Lite6: SN positions 3-6 < 1006
  - UF850: all units have compensation

Usage:
    python scripts/gen_kinematics_params.py <robot-ip> [suffix] [output_dir]
    python scripts/gen_kinematics_params.py <robot-ip> [suffix] --force

When suffix is omitted, defaults to the last 6 characters of the robot SN.
"""

from __future__ import annotations

import argparse
import ipaddress
import math
import sys
import time
from pathlib import Path

import numpy as np
from yaml import dump

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ufactory.kinematics.calibration import (  # noqa: E402
    get_robot_sn,
    has_per_unit_kinematics_calibration,
    kinematics_suffix_from_sn,
    log_kinematics_sn_status,
    parse_sn_model_code,
)
from ufactory.kinematics.controller_fk import (  # noqa: E402
    CalibrationIdentificationError,
    controller_flange_fk,
    identify_from_controller_fk,
    load_nominal_chain,
)
from ufactory.kinematics.tcp_offset import read_tcp_offset  # noqa: E402
from ufactory.robots.registry import get_robot_profile, joint_names, robot_name_for_controller  # noqa: E402


def _parse_robot_ipv4(value: str) -> str:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid IPv4 address {value!r}; use dotted-decimal form, e.g. 192.168.1.65"
        ) from None
    if address.version != 4:
        raise argparse.ArgumentTypeError(
            f"invalid robot IPv4 address {value!r}; IPv6 is not supported by the xArm controller"
        )
    return str(address)


def _output_dir_for_robot(robot_name: str) -> Path:
    return _REPO_ROOT / "assets" / "urdf" / "robots" / robot_name / "kinematics" / "user"


def _read_world_offset(arm, timeout_s: float = 2.0, poll_s: float = 0.05) -> np.ndarray:
    """``arm.world_offset`` once the rich report has delivered it (all-zero until then)."""
    deadline = time.monotonic() + timeout_s
    offset = np.zeros(6)
    while time.monotonic() < deadline:
        offset = np.asarray(arm.world_offset, dtype=np.float64).reshape(-1)
        if np.any(np.abs(offset) > 1e-6):
            return offset
        time.sleep(poll_s)
    return offset


def _export(arm, args: argparse.Namespace) -> int:
    axis, device_type = int(arm.axis), int(arm.device_type)
    robot_name = robot_name_for_controller(axis, device_type)
    if robot_name is None:
        print(f"[Failed] controller reports axis={axis}, device_type={device_type}: not a registered robot model")
        return 1
    profile = get_robot_profile(robot_name)
    sn = get_robot_sn(arm)
    if not sn:
        print("[Failed] controller did not report a serial number")
        return 1
    if not args.kinematics_suffix:
        args.kinematics_suffix = kinematics_suffix_from_sn(sn)
        print(f"kinematics_suffix: {args.kinematics_suffix} (auto from SN {sn})")

    print(f"robot_name     : {robot_name}")
    log_kinematics_sn_status(
        sn,
        robot_name,
        kinematics_suffix=args.kinematics_suffix,
        allow_sn_override=args.force,
    )
    if not args.force and not has_per_unit_kinematics_calibration(sn, robot_name):
        print(
            "[Skipped] SN model code {} indicates no per-unit kinematics compensation. "
            "Use nominal URDF without --kinematics-suffix. Pass --force to export anyway.".format(
                parse_sn_model_code(sn)
            )
        )
        return 2

    tcp_offset = read_tcp_offset(arm)
    world_offset = _read_world_offset(arm)
    if np.any(np.abs(world_offset) > 1e-6):
        print(
            f"[Failed] controller world_offset {world_offset.tolist()} is not zero; the export reads FK in the "
            "robot base frame. Set the world offset to 0 in UFACTORY Studio, export, then restore it."
        )
        return 1
    print(f"tcp_offset     : {tcp_offset.tolist()} (removed from controller FK)")

    chain = load_nominal_chain(profile.assets_dir / profile.default_urdf, joint_names(profile))
    if chain.ee_link != profile.ee_link:
        print(f"[Failed] URDF chain ends at {chain.ee_link}, expected {profile.ee_link}")
        return 1
    try:
        result = identify_from_controller_fk(controller_flange_fk(arm, tcp_offset), chain)
    except (CalibrationIdentificationError, RuntimeError) as exc:
        print(f"[Failed] {exc}; no file written")
        return 1
    print(
        f"controller FK  : rigid chain to {result.rigid_chain_error_m * 1e6:.2f} um "
        f"({result.identification_samples} identification + {result.verification_samples} held-out samples)"
    )
    print(
        f"calibration    : held-out error mean {result.mean_position_error_m * 1e3:.4f} mm, "
        f"max {result.max_position_error_m * 1e3:.4f} mm / {math.degrees(result.max_rotation_error_rad):.4f} deg"
    )
    print(
        f"flange         : {result.flange_axis_offset_m * 1e6:.1f} um off and "
        f"{math.degrees(result.flange_axis_tilt_rad) * 1e3:.2f} mdeg tilted from the last axis "
        "(exact at last joint 0)"
    )

    output_dir = Path(args.output_dir).resolve() if args.output_dir else _output_dir_for_robot(robot_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "{}_kinematics_{}.yaml".format(profile.kinematics_prefix, args.kinematics_suffix)
    data = {
        "schema_version": 1,
        "robot_key": profile.key,
        "serial_number": sn,
        "units": {"position": "m", "angle": "rad"},
        "joints": result.joints(chain.joint_names),
    }
    with open(output_file, "w", encoding="utf-8") as f:
        dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print("[Success] save to {}".format(output_file))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Export per-unit kinematics YAML from the controller FK")
    parser.add_argument(
        "robot_ip",
        type=_parse_robot_ipv4,
        help="Robot IPv4 address in dotted-decimal form (e.g. 192.168.1.65)",
    )
    parser.add_argument(
        "kinematics_suffix",
        nargs="?",
        default=None,
        help="Suffix for output YAML filename (default: last 6 characters of SN)",
    )
    parser.add_argument("output_dir", nargs="?", default=None, help="Optional output directory")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Export even when SN indicates no factory compensation (e.g. POE calibration in firmware)",
    )
    args = parser.parse_args()

    from xarm.wrapper import XArmAPI

    arm = None
    try:
        try:
            arm = XArmAPI(args.robot_ip, is_radian=True)
        except Exception as exc:
            print(
                f"[Failed] cannot connect to {args.robot_ip}: {exc}. "
                "Check the IP, controller power, and host network settings."
            )
            return 1
        if not arm.connected:
            print(
                f"[Failed] cannot connect to {args.robot_ip}. "
                "Check the IP, controller power, and host network settings."
            )
            return 1
        return _export(arm, args)
    finally:
        if arm is not None:
            try:
                arm.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
