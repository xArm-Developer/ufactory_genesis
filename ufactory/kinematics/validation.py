"""Reusable FK/IK validation helpers for UFACTORY Genesis models."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from ufactory.kinematics.calibration import (
    get_robot_sn,
    has_per_unit_kinematics_calibration,
    log_kinematics_sn_status,
    prepare_robot_model_for_verification,
    resolve_kinematics_suffix,
    validate_kinematics_calibration_request,
)
from ufactory.kinematics.tcp_offset import pose_flange_to_tcp, pose_tcp_to_flange, read_tcp_offset
from ufactory.robots.paths import robot_urdf
from ufactory.robots.runtime import RobotRuntimeProfile, get_robot_runtime_profile, robot_runtime_cli_choices
from ufactory.simulation import make_rigid_options
from ufactory.simulation.compat import ensure_ik_scratch, require_genesis_runtime

PASS_POS_MM = 1.0
PASS_RPY_DEG = 0.5


def quat_to_rpy(quat: Sequence[float]) -> tuple[float, float, float]:
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def angle_diff_deg(a: float, b: float) -> float:
    diff = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(diff) * 180.0 / math.pi


def validation_configs(runtime: RobotRuntimeProfile) -> list[tuple[str, np.ndarray]]:
    dof = runtime.model.dof
    configs = [("home", np.asarray(runtime.arm.home_qpos, dtype=np.float64))]
    configs.append(("default", np.asarray(runtime.arm.default_qpos, dtype=np.float64)))
    if dof >= 5:
        configs.append(
            ("A", np.asarray([0.5, -0.3, 0.0, 0.0, 0.3, *([0.0] * max(0, dof - 5))], dtype=np.float64)[:dof])
        )
    if dof >= 6:
        configs.append(
            ("B", np.asarray([0.0, -0.5, -0.1, 0.5, 0.5, 0.0, *([0.0] * max(0, dof - 6))], dtype=np.float64)[:dof])
        )
    return configs


def build_genesis_robot(urdf_path: str, *, backend: str = "cpu", show_viewer: bool = False):
    gs = require_genesis_runtime()

    gs.init(backend=gs.cpu if backend == "cpu" else gs.gpu)
    scene = gs.Scene(show_viewer=show_viewer, rigid_options=make_rigid_options(gs))
    robot = scene.add_entity(gs.morphs.URDF(file=urdf_path, fixed=True, requires_jac_and_IK=True))
    scene.build()
    return scene, robot


def genesis_fk(robot, q: np.ndarray, ee_link_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute forward kinematics for a given configuration.

    Genesis 1.4.0 removed RigidEntity.forward_kinematics(); use set_qpos and
    then get_links_pos/get_links_quat on the link objects.
    """
    import genesis as gs

    ensure_ik_scratch(robot, gs_module=gs)
    robot.set_qpos(q.astype(np.float32))
    links_pos = robot.get_links_pos(links_idx_local=[ee_link_idx])
    links_quat = robot.get_links_quat(links_idx_local=[ee_link_idx])
    if links_pos.ndim == 2:
        return links_pos[0].cpu().numpy(), links_quat[0].cpu().numpy()
    return links_pos[0, 0].cpu().numpy(), links_quat[0, 0].cpu().numpy()


def _connect_sdk(ip: str):
    from xarm.wrapper import XArmAPI

    arm = XArmAPI(ip, is_radian=True)
    connect = getattr(arm, "connect", None)
    if connect is not None:
        connect()
    return arm


def _prepare_sdk_and_urdf(args, runtime: RobotRuntimeProfile) -> tuple[object, str, str | None, np.ndarray]:
    arm = _connect_sdk(args.ip)
    sn = get_robot_sn(arm)
    cli_suffix = args.kinematics_suffix
    resolved_suffix = resolve_kinematics_suffix(
        kinematics_suffix=cli_suffix,
        kinematics_yaml=args.kinematics_yaml,
        sn=sn,
        robot_name=runtime.model.robot_name,
        kinematics_yaml_dir=args.kinematics_yaml_dir,
    )
    if resolved_suffix and not cli_suffix and args.kinematics_yaml is None:
        if has_per_unit_kinematics_calibration(sn, runtime.model.robot_name):
            print(f"kinematics_suffix: {resolved_suffix} (auto from SN {sn})")
        else:
            print(f"kinematics_suffix: {resolved_suffix} (auto from existing user YAML for SN {sn})")
    args.kinematics_suffix = resolved_suffix
    validate_kinematics_calibration_request(
        sn,
        runtime.model.robot_name,
        kinematics_yaml=args.kinematics_yaml,
        kinematics_suffix=args.kinematics_suffix,
        allow_sn_override=getattr(args, "force_kinematics", False),
        kinematics_yaml_dir=args.kinematics_yaml_dir,
    )
    log_kinematics_sn_status(
        sn,
        runtime.model.robot_name,
        kinematics_yaml=args.kinematics_yaml,
        kinematics_suffix=args.kinematics_suffix,
        allow_sn_override=getattr(args, "force_kinematics", False),
        kinematics_yaml_dir=args.kinematics_yaml_dir,
    )
    tcp_offset = read_tcp_offset(arm)
    print(f"tcp_offset     : {tcp_offset.tolist()}  (xyz mm, rpy rad; applied in FK/IK checks)")
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)

    robot_model_arg = args.robot_model or args.urdf
    urdf_path, kinematics_yaml_path = prepare_robot_model_for_verification(
        robot_model_arg,
        args.kinematics_yaml,
        args.kinematics_suffix,
        args.kinematics_yaml_dir,
        default_base_urdf=robot_urdf(runtime.model.key),
        robot_name=runtime.model.robot_name,
        joint_count=runtime.model.dof,
    )
    return arm, urdf_path, kinematics_yaml_path, tcp_offset


def run_fk_validation(args) -> int:
    runtime = get_robot_runtime_profile(args.robot)
    arm, urdf_path, kinematics_yaml_path, tcp_offset = _prepare_sdk_and_urdf(args, runtime)
    print(f"Robot: {runtime.model.key}")
    print(f"URDF : {Path(urdf_path).resolve()}")
    if kinematics_yaml_path:
        print(f"Calib: {kinematics_yaml_path}")

    _, robot = build_genesis_robot(urdf_path, backend=args.backend, show_viewer=args.vis)
    ee_link = next(l for l in robot.links if l.name.split("/")[-1] == runtime.arm.ee_link)

    failed = 0
    for name, q in validation_configs(runtime):
        code, pose = arm.get_forward_kinematics(q.tolist(), input_is_radian=True, return_is_radian=True)
        if code != 0:
            raise RuntimeError(f"SDK FK failed for {name}: code={code}")
        sdk_pose = np.asarray(pose[:6], dtype=np.float64)
        g_pos, g_quat = genesis_fk(robot, q, int(ee_link.idx_local))
        g_pos_mm = np.asarray(g_pos, dtype=np.float64) * 1000.0
        g_rpy = np.asarray(quat_to_rpy(g_quat), dtype=np.float64)
        flange_pose = np.concatenate([g_pos_mm, g_rpy])
        genesis_tcp = pose_flange_to_tcp(flange_pose, tcp_offset)
        pos_mm = float(np.linalg.norm(genesis_tcp[:3] - sdk_pose[:3]))
        rpy_deg = max(angle_diff_deg(a, b) for a, b in zip(genesis_tcp[3:6], sdk_pose[3:6]))
        ok = pos_mm < PASS_POS_MM and rpy_deg < PASS_RPY_DEG
        print(f"{'PASS' if ok else 'FAIL'} {name}: pos={pos_mm:.2f}mm rpy={rpy_deg:.2f}deg")
        failed += 0 if ok else 1

    arm.disconnect()
    if failed:
        return 1
    print("All FK checks passed")
    return 0


def run_ik_validation(args) -> int:
    import genesis as gs

    runtime = get_robot_runtime_profile(args.robot)
    arm, urdf_path, kinematics_yaml_path, tcp_offset = _prepare_sdk_and_urdf(args, runtime)
    print(f"Robot: {runtime.model.key}")
    print(f"URDF : {Path(urdf_path).resolve()}")
    if kinematics_yaml_path:
        print(f"Calib: {kinematics_yaml_path}")

    _, robot = build_genesis_robot(urdf_path, backend=args.backend, show_viewer=args.vis)
    ee_link = next(l for l in robot.links if l.name.split("/")[-1] == runtime.arm.ee_link)
    rng = np.random.default_rng(args.seed)
    failed = 0

    for i in range(args.samples):
        q_seed = rng.uniform(-0.5, 0.5, runtime.model.dof)
        code, pose = arm.get_forward_kinematics(q_seed.tolist(), input_is_radian=True, return_is_radian=True)
        if code != 0:
            print(f"SKIP ik_{i}: SDK FK code={code}")
            continue
        target_tcp = np.asarray(pose[:6], dtype=np.float64)
        flange_pose = pose_tcp_to_flange(target_tcp, tcp_offset)
        target_pos = flange_pose[:3] / 1000.0
        target_quat = np.asarray(rpy_to_quat(*flange_pose[3:6]), dtype=np.float64)
        init_q = torch.tensor(q_seed, dtype=torch.float32, device=gs.device)
        result = robot.inverse_kinematics(
            link=ee_link,
            pos=torch.tensor(target_pos, dtype=gs.tc_float, device=gs.device),
            quat=torch.tensor(target_quat, dtype=gs.tc_float, device=gs.device),
            init_qpos=init_q,
            respect_joint_limit=True,
            return_error=True,
            max_solver_iters=args.max_iters,
        )
        if result is None:
            print(f"FAIL ik_{i}: no solution")
            failed += 1
            continue
        q_sol = result[0] if isinstance(result, tuple) else result
        q_np = q_sol.cpu().numpy().reshape(-1)[: runtime.model.dof]
        code2, pose2 = arm.get_forward_kinematics(q_np.tolist(), input_is_radian=True, return_is_radian=True)
        if code2 != 0:
            print(f"FAIL ik_{i}: SDK FK(solution) code={code2}")
            failed += 1
            continue
        pose2_tcp = np.asarray(pose2[:6], dtype=np.float64)
        pos_mm = float(np.linalg.norm(pose2_tcp[:3] - target_tcp[:3]))
        rpy_deg = max(angle_diff_deg(a, b) for a, b in zip(pose2_tcp[3:6], target_tcp[3:6]))
        ok = pos_mm < PASS_POS_MM and rpy_deg < PASS_RPY_DEG
        print(f"{'PASS' if ok else 'FAIL'} ik_{i}: pos={pos_mm:.2f}mm rpy={rpy_deg:.2f}deg")
        failed += 0 if ok else 1

    arm.disconnect()
    if failed:
        return 1
    print("IK checks passed")
    return 0


def _add_common_args(parser: argparse.ArgumentParser, *, default_robot: str, require_robot: bool) -> None:
    parser.add_argument("--robot", default=default_robot, required=require_robot, choices=robot_runtime_cli_choices())
    parser.add_argument("--ip", required=True)
    parser.add_argument("--robot-model", default=None)
    parser.add_argument("--urdf", default=None, help="Alias for --robot-model")
    parser.add_argument("--kinematics-suffix", default=None)
    parser.add_argument("--kinematics-yaml", default=None)
    parser.add_argument("--kinematics-yaml-dir", default=None)
    parser.add_argument(
        "--force-kinematics",
        action="store_true",
        help="Use kinematics YAML/suffix even when SN rules out factory compensation",
    )
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("-v", "--vis", action="store_true")


def cli_fk(argv: Sequence[str] | None = None, *, default_robot: str = "xarm6", require_robot: bool = True) -> int:
    parser = argparse.ArgumentParser(description="FK verification: Genesis URDF vs xArm Python SDK")
    _add_common_args(parser, default_robot=default_robot, require_robot=require_robot)
    args = parser.parse_args(argv)
    return run_fk_validation(args)


def cli_ik(argv: Sequence[str] | None = None, *, default_robot: str = "xarm6", require_robot: bool = True) -> int:
    parser = argparse.ArgumentParser(description="IK verification: Genesis URDF vs xArm Python SDK")
    _add_common_args(parser, default_robot=default_robot, require_robot=require_robot)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--max-iters", type=int, default=20)
    args = parser.parse_args(argv)
    return run_ik_validation(args)
