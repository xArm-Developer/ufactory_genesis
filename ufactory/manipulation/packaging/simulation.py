"""Configuration-driven UFACTORY packaging showcase physics execution.

Build a robot-specific movable-gripper scene, physically grasp a red block,
and drop it into the configured open cardboard box.

Usage:
    export NUMBA_CACHE_DIR=~/.cache/numba
    # Windows PowerShell: $env:NUMBA_CACHE_DIR="$env:USERPROFILE\\.cache\\numba"
    ufactory-packaging-showcase --robot xarm6 --mode sim
    ufactory-packaging-showcase --robot lite6 --mode sim --cycles 3
    # Without a Genesis-supported GPU:
    ufactory-packaging-showcase --robot xarm6 --mode sim --backend cpu
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
import math
import time
from pathlib import Path

import numpy as np
import torch

import genesis as gs
from genesis.utils.geom import transform_quat_by_quat, xyz_to_quat

from ufactory.config import load_runtime_config, runtime_config_dict
from ufactory.config.models import ROBOTIQ_2F_85_PHYSICS_PROFILE, ROBOTIQ_2F_140_PHYSICS_PROFILE
from ufactory.config.loader import config_sha256
from ufactory.grippers import create_gripper_adapter
from ufactory.kinematics.genesis import solve_home_ik
from ufactory.grippers.g2 import (
    GRIPPER_G2_SIM_OPEN_DRIVE as GRIPPER_OPEN,
    GRIPPER_G2_SIM_CLOSE_DRIVE as GRIPPER_CLOSE,
)
from ufactory.manipulation.packaging.core import build_packaging_program, packaging_layout
from ufactory.manipulation.packaging.scene import (
    FINGER_FRICTION_MU,
    HOME_RPY_DEG,
    ROBOT_BASE_YAW_DEG,
    build_packaging_scene,
    make_layout,
)
from ufactory.robots.runtime import get_robot_runtime_profile
from ufactory.simulation import G2_PHYSICS_PROFILE, GenesisRuntimeManager
from ufactory.simulation.compat import require_genesis_capabilities
from ufactory.trajectory.ik import compile_cartesian_program_to_joint_stream
from ufactory.trajectory.scene import TrajSceneContext
from ufactory.trajectory.sim_executor import SimReport, replay_sim
from ufactory.visualization.viewer import start_deferred_viewer

SIM_DT = 0.02
SETTLE_STEPS = 40


@dataclass
class ShowcaseRobotCtx:
    ik_link: object
    left_finger: object
    right_finger: object
    arm_dof_idx: list[int]
    gripper_dof_idx: list[int]
    all_gripper_dof_idx: list[int]
    down_quat: torch.Tensor
    home_pos: list[float]
    home_qpos_saved: torch.Tensor
    finger_z_offset: float = 0.0
    grasp_drive: float = GRIPPER_CLOSE
    gripper_open_drive: float = GRIPPER_OPEN
    gripper_open_dof_idx: list[int] | None = None
    gripper_initial_open_dof_idx: list[int] | None = None
    gripper_initial_open_clearance_m: float = 0.0
    home_settle_steps: int = SETTLE_STEPS
    g2_controller: object | None = None
    robotiq_controller: object | None = None
    sim_dt: float = SIM_DT
    arm_controller: object | None = None


def _scale_steps(steps: int, speed: float) -> int:
    return max(1, int(round(steps / max(0.25, speed))))


def _world_home(
    layout,
    *,
    cartesian_xy_offset_m: tuple[float, float] | None = None,
) -> list[float]:
    rx, ry = layout.robot_xy
    bx, by, bz = layout.home_position_base
    offset_x, offset_y = (
        layout.simulation_grasp_center_compensation_xy if cartesian_xy_offset_m is None else cartesian_xy_offset_m
    )
    bx += offset_x
    by += offset_y
    yaw = math.radians(ROBOT_BASE_YAW_DEG)
    c, s = math.cos(yaw), math.sin(yaw)
    wx = bx * c - by * s
    wy = bx * s + by * c
    return [rx + wx, ry + wy, layout.table_top_z + bz]


def _world_down_quat() -> torch.Tensor:
    """TCP RPY in base frame → world quat (accounts for ROBOT_BASE_YAW_DEG)."""
    base_yaw_quat = xyz_to_quat(
        torch.tensor([[0.0, 0.0, ROBOT_BASE_YAW_DEG]], device=gs.device, dtype=gs.tc_float),
        rpy=True,
        degrees=True,
    )
    tcp_base_quat = xyz_to_quat(
        torch.tensor([list(HOME_RPY_DEG)], device=gs.device, dtype=gs.tc_float),
        rpy=True,
        degrees=True,
    )
    return transform_quat_by_quat(tcp_base_quat, base_yaw_quat)


def _setup_robot(robot, layout, config, *, arm_kp_scale: float | None = None, calibrated_g2: bool = True):
    gripper = config.gripper
    if gripper is None:
        raise RuntimeError(f"{config.robot.key} packaging showcase requires a configured gripper")
    ik_link = robot.get_link(config.robot.ee_link)
    left_finger = robot.get_link(gripper.finger_link_names[0])
    right_finger = robot.get_link(gripper.finger_link_names[1])
    left_finger.set_friction(float(FINGER_FRICTION_MU))
    right_finger.set_friction(float(FINGER_FRICTION_MU))

    arm_dof_idx = [robot.get_joint(name).dofs_idx_local[0] for name in config.robot.joint_names]
    gripper_dof_idx = [robot.get_joint(gripper.drive_joint).dofs_idx_local[0]]
    all_gripper_dof_idx = [robot.get_joint(name).dofs_idx_local[0] for name in gripper.all_joint_names]
    opening_gripper_dof_idx = (
        list(gripper_dof_idx)
        if calibrated_g2 or gripper.adapter not in {"g2", "robotiq_2f_85", "robotiq_2f_140"}
        else [gripper_dof_idx[0], all_gripper_dof_idx[3]]
    )
    applied_arm_kp_scale = layout.simulation_arm_kp_scale if arm_kp_scale is None else float(arm_kp_scale)
    robot.set_dofs_kp(
        torch.tensor(
            np.asarray(config.arm.kp) * applied_arm_kp_scale,
            device=gs.device,
            dtype=gs.tc_float,
        ),
        arm_dof_idx,
    )
    robot.set_dofs_kv(
        torch.tensor(config.arm.kv, device=gs.device, dtype=gs.tc_float),
        arm_dof_idx,
    )
    robot.set_dofs_force_range(
        torch.tensor(config.arm.force_lower_nm, device=gs.device, dtype=gs.tc_float),
        torch.tensor(config.arm.force_upper_nm, device=gs.device, dtype=gs.tc_float),
        arm_dof_idx,
    )
    robot.set_dofs_kp(
        torch.full((len(opening_gripper_dof_idx),), gripper.kp, device=gs.device, dtype=gs.tc_float),
        opening_gripper_dof_idx,
    )
    robot.set_dofs_kv(
        torch.full((len(opening_gripper_dof_idx),), gripper.kv, device=gs.device, dtype=gs.tc_float),
        opening_gripper_dof_idx,
    )
    robot.set_dofs_force_range(
        torch.full((len(opening_gripper_dof_idx),), gripper.force_lower_n, device=gs.device, dtype=gs.tc_float),
        torch.full((len(opening_gripper_dof_idx),), gripper.force_upper_n, device=gs.device, dtype=gs.tc_float),
        opening_gripper_dof_idx,
    )
    n_grip = len(all_gripper_dof_idx)
    robot.set_dofs_damping(
        torch.full((n_grip,), gripper.damping, device=gs.device, dtype=gs.tc_float),
        all_gripper_dof_idx,
    )
    robot.set_dofs_frictionloss(
        torch.full((n_grip,), gripper.frictionloss, device=gs.device, dtype=gs.tc_float),
        all_gripper_dof_idx,
    )

    if calibrated_g2 and gripper.adapter == "g2":
        from ufactory.simulation.g2_adapter import configure_g2_drive

        configure_g2_drive(robot, damping=gripper.damping, frictionloss=gripper.frictionloss)
    elif calibrated_g2 and gripper.adapter in {"robotiq_2f_85", "robotiq_2f_140"}:
        from ufactory.simulation.robotiq_adapter import configure_robotiq_drive

        configure_robotiq_drive(
            robot,
            gripper.adapter,
            damping=gripper.damping,
            frictionloss=gripper.frictionloss,
        )
    elif gripper.adapter == "lite6":
        from ufactory.grippers.lite6 import configure_lite6_gripper_drive

        configure_lite6_gripper_drive(
            robot,
            gripper_dof_idx,
            all_gripper_dof_idx,
            kp=gripper.kp,
            kv=gripper.kv,
            force_lower_n=gripper.force_lower_n,
            force_upper_n=gripper.force_upper_n,
            damping=gripper.damping,
            frictionloss=gripper.frictionloss,
        )
    down_quat = _world_down_quat()

    return ik_link, left_finger, right_finger, arm_dof_idx, gripper_dof_idx, all_gripper_dof_idx, down_quat


def _init_home_qpos(
    robot,
    ik_link,
    arm_dof_idx,
    all_gripper_dof_idx,
    down_quat,
    home_pos,
    *,
    gripper_open_drive: float,
    arm_controller=None,
    target_pos_base=None,
    default_qpos_rad=None,
):
    home_link6_pos = torch.tensor([home_pos], device=gs.device, dtype=gs.tc_float)
    init_qpos = torch.zeros(1, robot.n_dofs, device=gs.device, dtype=gs.tc_float)
    home_qpos_result = solve_home_ik(
        robot,
        ik_link,
        home_link6_pos,
        down_quat,
        arm_dof_idx=arm_dof_idx,
        init_qpos=init_qpos,
        arm_controller=arm_controller,
        target_pos_base=target_pos_base,
        default_qpos_rad=default_qpos_rad,
    )
    for i, idx in enumerate(arm_dof_idx):
        init_qpos[:, idx] = home_qpos_result[0, arm_dof_idx[i]]
    for idx in all_gripper_dof_idx:
        init_qpos[:, idx] = gripper_open_drive
    robot.set_qpos(init_qpos)
    return init_qpos.clone()


def _measure_finger_offset(ik_link, left_finger, right_finger):
    link6_pos = ik_link.get_pos()[0]
    fc_pos = ((left_finger.get_pos() + right_finger.get_pos()) / 2)[0]
    return (link6_pos[2] - fc_pos[2]).item()


def init_showcase_robot(
    robot,
    layout,
    scene,
    *,
    runtime_config=None,
    cartesian_xy_offset_m: tuple[float, float] | None = None,
    arm_kp_scale: float | None = None,
    calibrated_g2: bool = True,
) -> ShowcaseRobotCtx:
    """Apply PD gains, set home qpos via IK, and prime finger geometry."""
    config = runtime_config or load_runtime_config(layout.robot_key, task="packaging_showcase")
    task_layout = packaging_layout(config)
    gripper = config.gripper
    if gripper is None:
        raise RuntimeError(f"{config.robot.key} packaging showcase requires a configured gripper")
    ik_link, left_finger, right_finger, arm_dof_idx, gripper_dof_idx, all_gripper_dof_idx, down_quat = _setup_robot(
        robot,
        layout,
        config,
        arm_kp_scale=arm_kp_scale,
        calibrated_g2=calibrated_g2,
    )
    from ufactory.simulation.g2_model import G2Model
    from ufactory.simulation.robotiq_model import RobotiqModel

    is_g2 = calibrated_g2 and gripper.adapter == "g2"
    is_robotiq = calibrated_g2 and gripper.adapter in {"robotiq_2f_85", "robotiq_2f_140"}
    if is_g2:
        open_drive = G2Model.load().linkage.drive_for_gap_m(0.084)
    elif is_robotiq:
        open_drive = RobotiqModel.load(gripper.adapter).linkage.drive_for_gap_m(gripper.open_gap_m)
    else:
        open_drive = float(gripper.open_drive)
    home_pos = _world_home(layout, cartesian_xy_offset_m=cartesian_xy_offset_m)
    offset_x, offset_y = (
        layout.simulation_grasp_center_compensation_xy if cartesian_xy_offset_m is None else cartesian_xy_offset_m
    )
    home_pos_base = (
        float(layout.home_position_base[0]) + float(offset_x),
        float(layout.home_position_base[1]) + float(offset_y),
        float(layout.home_position_base[2]),
    )
    urdf_model_path = getattr(robot, "_ufactory_model_path", "")
    arm_controller = None
    if urdf_model_path:
        from ufactory.control import ArmMotionController

        arm_controller = ArmMotionController.from_robot(
            config.robot.key,
            urdf_path=urdf_model_path,
            servo_rate_hz=float(config.motion.rate_hz),
            allow_approx_motion=bool(config.motion.allow_approx_motion),
        )
    home_qpos_saved = _init_home_qpos(
        robot,
        ik_link,
        arm_dof_idx,
        all_gripper_dof_idx,
        down_quat,
        home_pos,
        gripper_open_drive=open_drive,
        arm_controller=arm_controller,
        target_pos_base=home_pos_base,
        default_qpos_rad=config.arm.default_qpos_rad,
    )
    scene.step()
    finger_z_offset = _measure_finger_offset(ik_link, left_finger, right_finger)
    grasp_drive = create_gripper_adapter(gripper).gap_to_drive(task_layout.grasp_gap_m)
    from ufactory.simulation.g2_adapter import G2GenesisController
    from ufactory.simulation.robotiq_adapter import RobotiqGenesisController

    adapter = G2GenesisController(robot, left_finger, right_finger, axis_link=ik_link) if is_g2 else None
    robotiq_adapter = (
        RobotiqGenesisController(robot, left_finger, right_finger, variant=gripper.adapter, axis_link=ik_link)
        if is_robotiq
        else None
    )
    return ShowcaseRobotCtx(
        ik_link=ik_link,
        left_finger=left_finger,
        right_finger=right_finger,
        arm_dof_idx=arm_dof_idx,
        gripper_dof_idx=gripper_dof_idx,
        all_gripper_dof_idx=all_gripper_dof_idx,
        down_quat=down_quat,
        home_pos=home_pos,
        home_qpos_saved=home_qpos_saved,
        finger_z_offset=finger_z_offset,
        grasp_drive=grasp_drive,
        gripper_open_drive=open_drive,
        gripper_open_dof_idx=list(gripper_dof_idx),
        gripper_initial_open_dof_idx=[],
        gripper_initial_open_clearance_m=0.0,
        g2_controller=adapter,
        robotiq_controller=robotiq_adapter,
        home_settle_steps=max(1, int(round(task_layout.simulation_home_settle_s * config.simulation.rate_hz))),
        sim_dt=1.0 / float(config.simulation.rate_hz),
        arm_controller=arm_controller,
    )


def hold_robot_home(robot, scene, ctx: ShowcaseRobotCtx, *, steps: int = 1) -> None:
    target_qpos = ctx.home_qpos_saved
    gripper_open = torch.full((1, 1), ctx.gripper_open_drive, device=gs.device, dtype=gs.tc_float)
    step_dt = float(getattr(ctx, "sim_dt", SIM_DT))
    for _ in range(steps):
        robot.control_dofs_position(target_qpos[:, ctx.arm_dof_idx], ctx.arm_dof_idx)
        if ctx.g2_controller is not None:
            ctx.g2_controller.command(84)
            ctx.g2_controller.apply(step_dt)
        elif ctx.robotiq_controller is not None:
            ctx.robotiq_controller.command(ctx.robotiq_controller.model.spec.open_gap_mm)
            ctx.robotiq_controller.apply(step_dt)
        else:
            robot.control_dofs_position(gripper_open, ctx.gripper_dof_idx)
        scene.step()


def _reset_block(block, layout) -> None:
    pos = torch.tensor(
        [[layout.obj_spawn_xy[0], layout.obj_spawn_xy[1], layout.table_top_z + layout.obj_spawn_center_z]],
        device=gs.device,
        dtype=gs.tc_float,
    )
    block.set_pos(pos, zero_velocity=True)
    block.set_quat(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=gs.device, dtype=gs.tc_float),
        zero_velocity=True,
    )


def prepare_packaging_cycle(
    scene,
    robot,
    block,
    layout,
    ctx: ShowcaseRobotCtx,
    *,
    speed: float = 1.0,
) -> None:
    """Restore the cube and passive release joint, then settle at home."""
    release_only_dof_idx = [idx for idx in ctx.all_gripper_dof_idx if idx not in ctx.gripper_dof_idx]
    robot.control_dofs_force(
        torch.zeros((1, len(release_only_dof_idx)), device=gs.device, dtype=gs.tc_float),
        release_only_dof_idx,
    )
    if getattr(ctx, "g2_controller", None) is not None:
        ctx.g2_controller.reset(84)
    elif getattr(ctx, "robotiq_controller", None) is not None:
        ctx.robotiq_controller.reset(ctx.robotiq_controller.model.spec.open_gap_mm)
    _reset_block(block, layout)
    hold_robot_home(robot, scene, ctx, steps=_scale_steps(getattr(ctx, "home_settle_steps", SETTLE_STEPS), speed))


def _with_sim_release_timing(program, *, speed: float, task_layout=None):
    """Return a copy with YAML-configured packaging-simulation release timing."""
    if task_layout is None:
        task_layout = packaging_layout(load_runtime_config("xarm6", task="packaging_showcase"))
    speed_scale = max(0.25, float(speed))
    duration = task_layout.simulation_release_duration_s / speed_scale
    relax_duration = task_layout.simulation_pre_release_relax_duration_s / speed_scale
    relax_gap = task_layout.simulation_pre_release_relax_gap_m
    segments = []
    for segment in program.segments:
        if segment.label == "pre-release-settle" and relax_duration > 0.0:
            # Preserve the configured stationary hold so the cube can settle
            # after the long transit.  Then ease the clamp preload in a
            # separate Genesis-only phase before the full release target.
            segments.append(segment)
            tuned = replace(
                segment,
                label="pre-release-relax",
                gap_end=relax_gap,
                duration=relax_duration,
            )
        elif segment.label == "release":
            tuned = replace(
                segment,
                gap_start=relax_gap if relax_duration > 0.0 else segment.gap_start,
                duration=duration,
            )
        else:
            segments.append(segment)
            continue
        _, samples_count = tuned.samples(program.rate)
        segments.append(replace(tuned, samples_count=samples_count))
    return replace(program, segments=segments)


def _speed_config(robot_key: str, speed: float, config_path: Path | None = None, *, base_config=None):
    """Scale the resolved task without reloading or losing user overrides."""
    mult = max(0.25, float(speed))
    base = base_config or load_runtime_config(robot_key, task="packaging_showcase", config_path=config_path)
    p = base.task.parameters
    scaled = replace(
        base,
        motion=replace(
            base.motion,
            joint_speed_rad_s=base.motion.joint_speed_rad_s * mult,
            joint_acceleration_rad_s2=base.motion.joint_acceleration_rad_s2 * mult,
            cartesian_speed_m_s=base.motion.cartesian_speed_m_s * mult,
            cartesian_acceleration_m_s2=base.motion.cartesian_acceleration_m_s2 * mult,
            gripper_duration_s=base.motion.gripper_duration_s / mult,
        ),
        task=replace(
            base.task,
            parameters={
                **p,
                "grasp_settle_s": float(p["grasp_settle_s"]) / mult,
                "pre_release_settle_s": float(p["pre_release_settle_s"]) / mult,
                "post_release_settle_s": float(p["post_release_settle_s"]) / mult,
            },
        ),
    )
    return replace(scaled, sha256=config_sha256(runtime_config_dict(scaled, include_provenance=False)))


def _trajectory_context(
    scene,
    robot,
    block,
    layout,
    ctx: ShowcaseRobotCtx,
    config,
    *,
    cartesian_xy_offset_m: tuple[float, float] | None = None,
) -> TrajSceneContext:
    task_layout = packaging_layout(config)
    gripper_name = config.gripper.adapter if config.gripper is not None else None
    runtime = get_robot_runtime_profile(config.robot.key, gripper=gripper_name)
    gripper = runtime.gripper
    if gripper is None:
        raise RuntimeError(f"{config.robot.key} packaging showcase requires a configured gripper")
    left = ctx.left_finger.get_pos()[0]
    right = ctx.right_finger.get_pos()[0]
    finger_span = max(abs(float(left[0] - right[0])), abs(float(left[1] - right[1])))
    offset_x, offset_y = (
        task_layout.simulation_grasp_center_compensation_xy_m
        if cartesian_xy_offset_m is None
        else cartesian_xy_offset_m
    )
    home_pos_base = list(task_layout.home_position_m)
    home_pos_base[0] += offset_x
    home_pos_base[1] += offset_y
    return TrajSceneContext(
        scene=scene,
        robot=robot,
        obj=block,
        target_marker=None,
        ik_link=ctx.ik_link,
        left_finger=ctx.left_finger,
        right_finger=ctx.right_finger,
        arm_dof_idx=ctx.arm_dof_idx,
        gripper_dof_idx=ctx.gripper_dof_idx,
        all_gripper_dof_idx=ctx.all_gripper_dof_idx,
        down_quat=ctx.down_quat,
        home_qpos=ctx.home_qpos_saved[0].detach().cpu().numpy().astype(np.float64),
        base_pos_world=(layout.robot_xy[0], layout.robot_xy[1], layout.table_top_z),
        base_yaw_rad=math.radians(ROBOT_BASE_YAW_DEG),
        gripper_open_dof_idx=list(ctx.gripper_open_dof_idx or ctx.gripper_dof_idx),
        gripper_initial_open_dof_idx=list(ctx.gripper_initial_open_dof_idx or ()),
        gripper_initial_open_clearance_m=ctx.gripper_initial_open_clearance_m,
        gripper_binary_commands=gripper.family == "lite6",
        visual_model="glb",
        finger_z_offset=ctx.finger_z_offset,
        finger_y_gap=finger_span,
        grasp_link6_z=task_layout.grasp_link6_z_m,
        pre_grasp_link6_z=task_layout.grasp_link6_z_m + task_layout.pregrasp_clearance_m,
        lift_link6_z=task_layout.release_link6_z_m,
        robot_key=config.robot.key,
        gripper=gripper,
        home_pos_base=home_pos_base,
        table_height=layout.table_top_z,
        obj_xy=task_layout.object_position_m[:2],
        place_xy=task_layout.target_position_m[:2],
        obj_size=task_layout.object_size_m,
        obj_mass_kg=task_layout.object_mass_kg,
        obj_pos_base=task_layout.object_position_m,
        place_pos_base=task_layout.target_position_m,
        robot_urdf_path=getattr(robot, "_ufactory_model_path", ""),
        g2_settings=config.simulation.g2,
        g2_controller=ctx.g2_controller,
        robotiq_settings=config.simulation.robotiq,
        robotiq_controller=ctx.robotiq_controller,
        simulation_parameters={
            "rate_hz": config.motion.rate_hz,
            "sim_rate_hz": float(config.simulation.rate_hz),
            "substeps": config.simulation.substeps,
            "constraint_time_constant_s": config.simulation.constraint_time_constant_s,
            "constraint_solver": config.simulation.constraint_solver,
            "solver_iterations": config.simulation.solver_iterations,
            "noslip_iterations": config.simulation.noslip_iterations,
            "friction_cone": config.simulation.friction_cone,
            "contact_resolution": config.simulation.contact_resolution,
            "use_gjk_collision": config.simulation.use_gjk_collision,
        },
        sim_rate_hz=float(config.simulation.rate_hz),
        arm_controller=getattr(ctx, "arm_controller", None),
    )


def run_pick_place_cycle(
    scene,
    robot,
    block,
    layout,
    *,
    speed: float = 1.0,
    ctx: ShowcaseRobotCtx | None = None,
    stop_after_phase0: bool = False,
    capture_hook: callable | None = None,
    config_path: Path | None = None,
    robot_key: str | None = None,
    executor: str = "servo_cartesian",
    runtime_config=None,
    csv_path: Path | None = None,
    sim_report_path: Path | None = None,
    preflight_report_path: Path | None = None,
) -> SimReport | None:
    """Replay the shared physical trajectory without kinematic object carry."""
    selected_robot = robot_key or layout.robot_key
    if ctx is None:
        base_config = runtime_config or load_runtime_config(
            selected_robot, task="packaging_showcase", config_path=config_path
        )
        ctx = init_showcase_robot(robot, layout, scene, runtime_config=base_config)
        runtime_config = base_config
    if stop_after_phase0:
        hold_robot_home(robot, scene, ctx, steps=_scale_steps(SETTLE_STEPS, speed))
        if capture_hook is not None:
            capture_hook("Home")
        return None

    config = _speed_config(selected_robot, speed, config_path, base_config=runtime_config)
    task_layout = packaging_layout(config)
    traj_ctx = _trajectory_context(scene, robot, block, layout, ctx, config)
    sim_offset = task_layout.simulation_grasp_center_compensation_xy_m
    if executor == "servo_j":
        q_home = traj_ctx.home_qpos[traj_ctx.arm_dof_idx]
        source = build_packaging_program(
            config,
            q_home=q_home,
            cartesian_xy_offset_m=sim_offset,
            place_xy_offset_m=(0.0, 0.0),
        )
        program = compile_cartesian_program_to_joint_stream(source, traj_ctx)
    elif executor == "servo_cartesian":
        program = build_packaging_program(
            config,
            q_home=traj_ctx.home_qpos[traj_ctx.arm_dof_idx],
            cartesian_xy_offset_m=sim_offset,
            place_xy_offset_m=(0.0, 0.0),
        )
    else:
        raise ValueError(f"unknown packaging executor: {executor}")
    if config.gripper.adapter == "g2":
        from ufactory.trajectory.g2_confirmation import prepare_g2_program

        program = prepare_g2_program(program, config.simulation.g2)
    elif config.gripper.adapter in {"robotiq_2f_85", "robotiq_2f_140"}:
        from ufactory.trajectory.g2_confirmation import prepare_robotiq_program

        program = prepare_robotiq_program(program, config.simulation.robotiq, config.gripper.adapter)
    else:
        program = _with_sim_release_timing(program, speed=speed, task_layout=task_layout)

    # Preflight the exact built model and executable program.
    from ufactory.manipulation.commands import build_packaging_backends, write_json
    from ufactory.manipulation.packaging.core import packaging_scene_sha256
    from ufactory.trajectory.preflight import create_safety_gate
    from ufactory.trajectory.g2_replay import save_sim_report

    urdf = Path(traj_ctx.robot_urdf_path)
    kinematics, collision = build_packaging_backends(
        config,
        urdf,
        calibrated_g2=(ctx.g2_controller is not None or ctx.robotiq_controller is not None),
    )
    gate = create_safety_gate(
        config,
        kinematics=kinematics,
        collision=collision,
        calibration_sha256="0" * 64,
        scene_sha256=packaging_scene_sha256(config),
        urdf_path=urdf,
    )
    preflight = gate.preflight(program, executor=executor)
    write_json(preflight_report_path, preflight.to_dict())
    if not preflight.passed:
        report = SimReport(completed=False, failure_reason="preflight failed", timing_passed=False)
        save_sim_report(report, csv_path, sim_report_path)
        return report
    approved = gate.approve_simulation(program, preflight)

    def _on_phase(status, segment) -> None:
        obj = tuple(value / 1000.0 for value in status.obj_pos_mm)
        link6 = tuple(value / 1000.0 for value in status.link6_mm)
        print(
            f"  [{status.label:20s}] link6=[{link6[0]:.3f}, {link6[1]:.3f}, {link6[2]:.3f}]  "
            f"block=[{obj[0]:.3f}, {obj[1]:.3f}, {obj[2]:.3f}]  "
            f"end_err={status.eside_arm_mm:.1f} mm"
        )
        if capture_hook is not None:
            capture_hook(segment.label or segment.kind)

    print("\n[Physical packaging trajectory]")
    print(
        f"  release object-bottom clearance={task_layout.release_object_bottom_clearance_m * 1000.0:.0f} mm  "
        f"grasp gap={task_layout.grasp_gap_m * 1000.0:.0f} mm"
    )
    from ufactory.trajectory.execution import execute_sim

    report = execute_sim(
        approved,
        lambda checked: replay_sim(
            checked,
            traj_ctx,
            stabilize_grasp_weld=False,
            on_phase=_on_phase,
            csv_path=csv_path,
            sim_report_path=sim_report_path,
        ),
    )
    ok = _cycle_failure_reason(report, layout, task_layout) is None
    print(
        f"\n  Place error: {report.place_error_mm:.1f} mm  "
        f"Home drift: {report.home_drift_mm:.1f} mm  {'OK' if ok else 'CHECK'}"
    )
    if config.gripper.adapter in {"g2", "robotiq_2f_85", "robotiq_2f_140"}:
        print(
            f"  completed={report.completed} timing={'PASS' if report.timing_passed else 'FAIL'} "
            f"transport={'PASS' if report.transport_passed else 'FAIL'}"
        )
    return report


def _cycle_failure_reason(report: SimReport, layout, task_layout) -> str | None:
    if not report.completed:
        return report.failure_reason or "simulation incomplete"
    if not report.timing_passed:
        return "timing acceptance failed"
    if (
        report.physics_profile in {G2_PHYSICS_PROFILE, ROBOTIQ_2F_85_PHYSICS_PROFILE, ROBOTIQ_2F_140_PHYSICS_PROFILE}
        and not report.transport_passed
    ):
        return "transport acceptance failed"
    phases = {phase.label: phase for phase in report.phases}
    lift = phases.get("lift")
    lift_clearance_m = float(getattr(task_layout, "lift_success_clearance_m", 0.100))
    min_lift_z_mm = (layout.table_top_z + lift_clearance_m) * 1000.0
    if lift is None or lift.obj_pos_mm[2] <= min_lift_z_mm:
        actual_z_mm = float("nan") if lift is None else lift.obj_pos_mm[2]
        return f"grasp/lift failed: block_z={actual_z_mm:.1f} mm, required>{min_lift_z_mm:.1f} mm"
    home_limit_mm = float(getattr(task_layout, "home_success_distance_m", 0.010)) * 1000.0
    if report.home_drift_mm >= home_limit_mm:
        return f"home drift {report.home_drift_mm:.1f} mm exceeds {home_limit_mm:.1f} mm"
    return None


def _hold_final_view(scene) -> None:
    visualizer = getattr(scene, "visualizer", None)
    viewer = getattr(visualizer, "viewer", None) if visualizer is not None else None
    if viewer is None:
        return
    try:
        while True:
            scene.step()
            time.sleep(SIM_DT)
    except gs.GenesisException as exc:
        if "Viewer closed" not in str(exc):
            raise


def _cycle_limit(cycles: int | None) -> int:
    """Return the finite cycle count. The default is one cycle."""

    return 1 if cycles is None else cycles


def _cycle_indices(cycle_limit: int) -> Iterator[int]:
    yield from range(1, cycle_limit + 1)


@dataclass(frozen=True)
class PackagingSimulationOptions:
    speed: float = 1.0
    executor: str = "servo_j"
    cycles: int | None = None
    table_height: float | None = None
    visual: bool = False
    csv: Path | None = None
    sim_report: Path | None = None
    report: Path | None = None


def run_packaging_simulation(options: PackagingSimulationOptions, runtime_config) -> int:
    """Execute a resolved command without parsing arguments or reloading YAML."""
    if not math.isfinite(options.speed) or options.speed <= 0:
        raise ValueError("speed must be finite and positive")
    if options.cycles is not None and options.cycles < 1:
        raise ValueError("cycles must be at least 1")
    if runtime_config.gripper is None:
        raise ValueError("packaging requires a configured gripper")
    if runtime_config.gripper.adapter == "g2" and runtime_config.simulation.g2.speed_percent <= 0:
        raise ValueError("automatic G2 manipulation requires speed_percent > 0")
    if (
        runtime_config.gripper.adapter in {"robotiq_2f_85", "robotiq_2f_140"}
        and runtime_config.simulation.robotiq.speed_percent <= 0
    ):
        raise ValueError("automatic Robotiq manipulation requires speed_percent > 0")
    require_genesis_capabilities(gs, pbr=True, deferred_viewer=True)
    with GenesisRuntimeManager(runtime_config.simulation):
        return _run_packaging_sim_body(options, runtime_config)


def _cycle_path(path: Path | None, cycle: int, limit: int) -> Path | None:
    return path.with_name(f"{path.stem}.cycle-{cycle}{path.suffix}") if path is not None and limit > 1 else path


def _run_packaging_sim_body(args: PackagingSimulationOptions, runtime_config) -> int:
    table_top_z = (
        args.table_height if args.table_height is not None else make_layout(runtime_config=runtime_config).table_top_z
    )
    scene, robot, block, layout = build_packaging_scene(
        table_top_z,
        sim_dt=1.0 / float(runtime_config.simulation.rate_hz),
        build_scene=True,
        show_viewer=False,
        runtime_config=runtime_config,
    )

    print(f"{runtime_config.robot.key} packaging showcase — Ctrl+C to exit")
    print(f"  simulation.backend={runtime_config.simulation.backend}")
    cycle_limit = _cycle_limit(args.cycles)
    task_layout = packaging_layout(runtime_config)
    print(f"  table_top_z={layout.table_top_z:.2f}m  speed={args.speed}  cycles={cycle_limit}")

    ctx = init_showcase_robot(robot, layout, scene, runtime_config=runtime_config)
    hold_robot_home(robot, scene, ctx, steps=_scale_steps(SETTLE_STEPS, args.speed))

    if args.visual:
        start_deferred_viewer(scene)

    failed = False
    cycle = 0
    try:
        for cycle in _cycle_indices(cycle_limit):
            total = str(cycle_limit)
            print(f"\n=== Cycle {cycle}/{total} ===")
            prepare_packaging_cycle(scene, robot, block, layout, ctx, speed=args.speed)
            report = run_pick_place_cycle(
                scene,
                robot,
                block,
                layout,
                speed=args.speed,
                ctx=ctx,
                robot_key=runtime_config.robot.key,
                executor=args.executor,
                runtime_config=runtime_config,
                csv_path=_cycle_path(args.csv, cycle, cycle_limit),
                sim_report_path=_cycle_path(args.sim_report, cycle, cycle_limit),
                preflight_report_path=_cycle_path(args.report, cycle, cycle_limit),
            )
            assert report is not None
            failure_reason = _cycle_failure_reason(report, layout, task_layout)
            if failure_reason is not None:
                failed = True
                print(f"\nCycle {cycle}/{total} FAILED: {failure_reason}")
                break
            print(f"Cycle {cycle}/{total} complete.")

        if args.visual:
            if failed:
                print("No further cycles will run. Viewer stays open for inspection.")
            else:
                print(f"\nCompleted {cycle_limit} cycle(s). Viewer stays open.")
            _hold_final_view(scene)
        elif failed:
            print("No further cycles will run.")
        else:
            print(f"\nCompleted {cycle_limit} cycle(s).")
    except KeyboardInterrupt:
        return 1 if failed else 0
    return 1 if failed else 0
