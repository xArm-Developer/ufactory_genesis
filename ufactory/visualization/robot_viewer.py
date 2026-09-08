"""Shared Genesis GLB viewer helpers for UFACTORY robots."""

from __future__ import annotations

import time

import numpy as np

import genesis as gs
from ufactory.grippers.bio_g2 import BioGripperG2
from ufactory.visualization.glb import enable_glb_pbr_surfaces, glb_view_surface
from ufactory.robots.paths import robot_urdf, robot_visual_glb_urdf
from ufactory.robots.runtime import get_robot_runtime_profile
from ufactory.robots.registry import RobotModelSpec, joint_names
from ufactory.simulation import make_rigid_options
from ufactory.simulation.compat import ensure_ik_scratch, require_genesis_runtime

from ufactory.visualization.bio_gripper_g2_viewer import (
    BIO_GRIPPER_G2_OPEN,
    bio_gripper_g2_demo_target,
)
from ufactory.visualization.gripper_g2_viewer import (
    GRIPPER_OPEN,
    control_gripper_pose,
    gripper_demo_target,
    gripper_dof_indices,
    set_gripper_pose,
    setup_gripper_pd,
)
from ufactory.visualization.lite6_gripper_viewer import (
    LITE6_GRIPPER_OPEN,
    control_lite6_gripper_pose,
    lite6_gripper_demo_target,
    lite6_gripper_dof_indices,
    set_lite6_gripper_pose,
    setup_lite6_gripper_pd,
)

TCP_MARKER_RADIUS = 0.008


def resolve_robot_link(robot, name: str):
    available = {link.name.split("/")[-1]: link for link in robot.links}
    if name not in available:
        raise KeyError(f"Link not found: {name}. Available: {sorted(available)}")
    return available[name]


def add_tcp_marker(scene):
    """Red sphere at DH TCP (EE flange); visual only, no collision."""
    return scene.add_entity(
        gs.morphs.Sphere(
            radius=TCP_MARKER_RADIUS,
            fixed=True,
            collision=False,
        ),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.0, 0.0)),
        ),
    )


def add_base_axes(scene, *, length: float = 0.25, radius: float = 0.004) -> list:
    """Temporary robot-base RGB axes at the origin (X red / Y green / Z blue)."""
    half = 0.5 * float(length)
    specs = (
        ((half, 0.0, 0.0), (0.0, 90.0, 0.0), (1.0, 0.0, 0.0)),  # +X
        ((0.0, half, 0.0), (-90.0, 0.0, 0.0), (0.0, 1.0, 0.0)),  # +Y
        ((0.0, 0.0, half), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),  # +Z
    )
    markers = []
    for pos, euler, color in specs:
        markers.append(
            scene.add_entity(
                gs.morphs.Cylinder(
                    height=float(length),
                    radius=float(radius),
                    pos=pos,
                    euler=euler,
                    fixed=True,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=color),
                ),
            )
        )
    return markers


def update_tcp_marker(marker, ee_link) -> None:
    marker.set_pos(ee_link.get_pos())


def start_deferred_viewer(scene) -> None:
    """Open the interactive viewer after the scene has been initialized and warmed up."""
    from ufactory.visualization.viewer import start_deferred_viewer as _start_deferred_viewer

    _start_deferred_viewer(scene)


def setup_arm_pd(robot, dof_idx: list[int], profile: RobotModelSpec) -> None:
    runtime = get_robot_runtime_profile(profile.key)
    robot.set_dofs_kp(np.array(runtime.arm.kp), dof_idx)
    robot.set_dofs_kv(np.array(runtime.arm.kv), dof_idx)
    robot.set_dofs_force_range(
        np.array(runtime.arm.force_lower),
        np.array(runtime.arm.force_upper),
        dof_idx,
    )


def _to_numpy3(pos) -> np.ndarray:
    if hasattr(pos, "cpu"):
        pos = pos.cpu().numpy()
    return np.asarray(pos).reshape(-1)[:3]


def _link_world_positions(robot, link_names: tuple[str, ...]) -> dict[str, list[float]]:
    available = {link.name.split("/")[-1]: link for link in robot.links}
    out = {}
    for name in link_names:
        if name not in available:
            continue
        arr = _to_numpy3(available[name].get_pos())
        out[name] = [float(x) for x in arr]
    return out


def _fk_link_pos(robot, ee_link, qpos_np: np.ndarray) -> np.ndarray:
    """Compute the end-effector link position via Genesis FK.

    Genesis 1.4.0 removed RigidEntity.forward_kinematics(); use set_qpos and
    get_links_pos instead. The qpos is temporarily set and then the link
    position is read; this is accurate for kinematic queries.
    """
    ensure_ik_scratch(robot, gs_module=gs)
    robot.set_qpos(qpos_np.astype(np.float32))
    link_pos = ee_link.get_pos()
    if hasattr(link_pos, "cpu"):
        link_pos = link_pos.cpu().numpy()
    return np.asarray(link_pos).reshape(-1)[:3]


def run_glb_diagnose(
    profile: RobotModelSpec,
    *,
    with_gripper_g2: bool = False,
    backend: str = "gpu",
) -> None:
    """Headless GLB/STL link pose diagnostic for any supported arm."""
    from ufactory.simulation import genesis_backend_constant

    require_genesis_runtime(gs)
    enable_glb_pbr_surfaces()
    gs.init(backend=genesis_backend_constant(gs, backend))
    stl_path = robot_urdf(profile.key)
    glb_path = robot_visual_glb_urdf(profile.key, with_gripper_g2=with_gripper_g2)
    link_names = tuple(["link_base"] + [f"link{i}" for i in range(1, profile.dof + 1)])

    def load_robot(urdf_path: str, use_glb: bool = False):
        scene = gs.Scene(
            show_viewer=False,
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=make_rigid_options(gs),
        )
        morph = gs.morphs.URDF(file=urdf_path, pos=(0.0, 0.0, 0.0), fixed=True, requires_jac_and_IK=True)
        robot = scene.add_entity(morph, surface=glb_view_surface()) if use_glb else scene.add_entity(morph)
        scene.build()
        return robot, scene, _link_world_positions(robot, link_names)

    _, _, stl_pos = load_robot(stl_path)
    robot, scene, glb_pos = load_robot(glb_path, use_glb=True)

    max_delta_mm = 0.0
    for link in set(stl_pos) & set(glb_pos):
        delta = float(np.linalg.norm(np.array(stl_pos[link]) - np.array(glb_pos[link])) * 1000)
        max_delta_mm = max(max_delta_mm, delta)
    print(f"max link pose delta STL vs GLB: {max_delta_mm:.3f} mm")

    runtime = get_robot_runtime_profile(profile.key)
    joint_map = {j.name.split("/")[-1]: j for j in robot.joints}
    arm_dof_idx = [joint_map[n].dofs_idx_local[0] for n in runtime.arm.joint_names if n in joint_map]
    if arm_dof_idx:
        home = np.asarray(runtime.arm.home_qpos, dtype=np.float64)
        robot.set_dofs_position(home[: len(arm_dof_idx)], arm_dof_idx)
        for _ in range(5):
            scene.step()
    ee_link = resolve_robot_link(robot, runtime.arm.ee_link)
    qpos = robot.get_dofs_position()
    if hasattr(qpos, "cpu"):
        qpos = qpos.cpu().numpy()
    fk_pos = _fk_link_pos(robot, ee_link, np.asarray(qpos).reshape(-1))
    ee_pos = _to_numpy3(ee_link.get_pos())
    fk_delta_mm = float(np.linalg.norm(fk_pos - ee_pos) * 1000)
    print(f"{runtime.arm.ee_link} get_pos vs forward_kinematics: {fk_delta_mm:.4f} mm")


def _set_gripper_kinematic(robot, all_gripper_dof_idx: list[int], value: float) -> None:
    if not all_gripper_dof_idx:
        return
    robot.set_dofs_position(
        np.full(len(all_gripper_dof_idx), value),
        all_gripper_dof_idx,
        zero_velocity=True,
    )


def _disable_robot_pd(robot, dof_idx: list[int]) -> None:
    """Zero PD gains so scene.step() does not apply position control forces."""
    if not dof_idx:
        return
    n = len(dof_idx)
    zeros = np.zeros(n)
    robot.set_dofs_kp(zeros, dof_idx)
    robot.set_dofs_kv(zeros, dof_idx)
    robot.set_dofs_force_range(zeros, zeros, dof_idx)


def _apply_kinematic_hold(
    robot,
    arm_dof_idx: list[int],
    arm_q: np.ndarray,
    *,
    hold_arm: bool,
    hold_gripper: bool,
    all_gripper_dof_idx: list[int],
    all_lite6_gripper_dof_idx: list[int],
    all_bio_gripper_g2_dof_idx: list[int],
    bio_gripper: BioGripperG2 | None = None,
) -> None:
    """Visual-only hold: teleport joints after physics step, no PD."""
    if hold_arm and arm_dof_idx:
        robot.set_dofs_position(arm_q[: len(arm_dof_idx)], arm_dof_idx, zero_velocity=True)
    if not hold_gripper:
        return
    if all_gripper_dof_idx:
        _set_gripper_kinematic(robot, all_gripper_dof_idx, GRIPPER_OPEN)
    elif all_lite6_gripper_dof_idx:
        _set_gripper_kinematic(robot, all_lite6_gripper_dof_idx, LITE6_GRIPPER_OPEN)
    elif all_bio_gripper_g2_dof_idx and bio_gripper is not None:
        # Bio Gripper G2 mirrors the left finger (-1), so hold the open pose through the
        # controller rather than writing the same value to both finger DOFs.
        bio_gripper.set_pose(BIO_GRIPPER_G2_OPEN)


def _kinematic_step(
    scene,
    robot,
    *,
    arm_kinematic_hold: bool,
    idle_gripper_kinematic_hold: bool,
    arm_dof_idx: list[int],
    home: np.ndarray,
    all_gripper_dof_idx: list[int],
    all_lite6_gripper_dof_idx: list[int],
    all_bio_gripper_g2_dof_idx: list[int],
    bio_gripper: BioGripperG2 | None = None,
    arm_target: np.ndarray | None = None,
) -> None:
    """Advance sim one step; held DOFs are teleported immediately before and after."""
    hold_needed = arm_kinematic_hold or idle_gripper_kinematic_hold
    if hold_needed:
        _apply_kinematic_hold(
            robot,
            arm_dof_idx,
            home if arm_target is None else arm_target,
            hold_arm=arm_kinematic_hold,
            hold_gripper=idle_gripper_kinematic_hold,
            all_gripper_dof_idx=all_gripper_dof_idx,
            all_lite6_gripper_dof_idx=all_lite6_gripper_dof_idx,
            all_bio_gripper_g2_dof_idx=all_bio_gripper_g2_dof_idx,
            bio_gripper=bio_gripper,
        )
        scene.step(update_visualizer=False)
        _apply_kinematic_hold(
            robot,
            arm_dof_idx,
            home if arm_target is None else arm_target,
            hold_arm=arm_kinematic_hold,
            hold_gripper=idle_gripper_kinematic_hold,
            all_gripper_dof_idx=all_gripper_dof_idx,
            all_lite6_gripper_dof_idx=all_lite6_gripper_dof_idx,
            all_bio_gripper_g2_dof_idx=all_bio_gripper_g2_dof_idx,
            bio_gripper=bio_gripper,
        )
        visualizer = getattr(scene, "visualizer", None)
        if getattr(visualizer, "viewer", None) is not None:
            visualizer.update(force=False)
    else:
        scene.step()


# Genesis viewer joint demo: smooth playback at 50°/s equivalent in rad/s.
_ARM_DEMO_SPEED_RAD_S = float(np.radians(50.0))
_ARM_DEMO_DT = 0.01


def _step_joint_toward(current: np.ndarray, goal: np.ndarray, max_step: float) -> np.ndarray:
    delta = goal - current
    max_abs = float(np.max(np.abs(delta)))
    if max_abs <= max_step:
        return goal.copy()
    return current + delta * (max_step / max_abs)


class _ArmJointDemo:
    """Smooth joint-space waypoint playback (kinematic), not stiff PD tracking."""

    def __init__(self, poses: list[np.ndarray], n_dof: int) -> None:
        self._poses = poses
        self._n = n_dof
        self._idx = 0
        self._max_step = _ARM_DEMO_SPEED_RAD_S * _ARM_DEMO_DT
        self.current = poses[0][:n_dof].copy()
        self.goal = poses[0][:n_dof].copy()
        self.finished = len(poses) <= 1

    def step(self) -> tuple[np.ndarray, np.ndarray, bool]:
        if self.finished:
            return self.current.copy(), self.current.copy(), True
        prev = self.current.copy()
        self.current = _step_joint_toward(self.current, self.goal, self._max_step)
        if np.allclose(self.current, self.goal, atol=1e-4):
            if self._idx >= len(self._poses) - 1:
                self.finished = True
                return prev, self.current.copy(), True
            self._idx += 1
            self.goal = self._poses[self._idx][: self._n].copy()
        return prev, self.current.copy(), False


def run_glb_viewer(
    profile: RobotModelSpec,
    urdf_path: str,
    *,
    headless: bool = False,
    pd_demo: bool = False,
    gripper_demo: bool = False,
    show_tcp: bool = False,
    show_base_axes: bool = False,
    backend: str = "gpu",
) -> None:
    from ufactory.simulation import genesis_backend_constant

    require_genesis_runtime(gs)
    enable_glb_pbr_surfaces()
    gs.init(backend=genesis_backend_constant(gs, backend))
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -1.5, 1.5),
            camera_lookat=(0.0, 0.0, 0.4),
            camera_fov=40,
            refresh_rate=60,
        ),
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=make_rigid_options(gs),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_path,
            pos=(0.0, 0.0, 0.0),
            fixed=True,
            requires_jac_and_IK=True,
        ),
        surface=glb_view_surface(),
    )
    tcp_marker = None
    if show_tcp and not headless:
        tcp_marker = add_tcp_marker(scene)
    if show_base_axes and not headless:
        add_base_axes(scene)
        print("Base axes: X=red, Y=green, Z=blue (robot base origin)")
    scene.build()

    jnames = joint_names(profile)
    home = np.zeros(profile.dof)
    joint_map = {j.name.split("/")[-1]: j for j in robot.joints}
    arm_dof_idx = [joint_map[n].dofs_idx_local[0] for n in jnames if n in joint_map]
    gripper_dof_idx, all_gripper_dof_idx = gripper_dof_indices(robot)
    lite6_gripper_dof_idx, all_lite6_gripper_dof_idx = lite6_gripper_dof_indices(robot)
    # Use BioGripperG2 class for discovery and control (preferred API).
    bio_gripper = BioGripperG2(robot)
    bio_gripper_g2_dof_idx = bio_gripper.drive_dof_idx
    all_bio_gripper_g2_dof_idx = bio_gripper.all_dof_idx
    arm_kinematic_hold = not pd_demo
    idle_gripper_kinematic_hold = not gripper_demo

    ee_link = None
    if tcp_marker is not None:
        ee_link = resolve_robot_link(robot, profile.ee_link)
        print(f"TCP marker: {profile.ee_link} (DH flange, no tool)")

    held_dof_idx: list[int] = []
    if arm_kinematic_hold:
        held_dof_idx.extend(arm_dof_idx)
    if idle_gripper_kinematic_hold:
        held_dof_idx.extend(all_gripper_dof_idx or all_lite6_gripper_dof_idx or all_bio_gripper_g2_dof_idx)
    if held_dof_idx:
        _disable_robot_pd(robot, sorted(set(held_dof_idx)))
    if arm_kinematic_hold or idle_gripper_kinematic_hold:
        _apply_kinematic_hold(
            robot,
            arm_dof_idx,
            home,
            hold_arm=arm_kinematic_hold,
            hold_gripper=idle_gripper_kinematic_hold,
            all_gripper_dof_idx=all_gripper_dof_idx,
            all_lite6_gripper_dof_idx=all_lite6_gripper_dof_idx,
            all_bio_gripper_g2_dof_idx=all_bio_gripper_g2_dof_idx,
            bio_gripper=bio_gripper,
        )
    if pd_demo and arm_dof_idx:
        setup_arm_pd(robot, arm_dof_idx, profile)
        robot.set_dofs_position(home[: len(arm_dof_idx)], arm_dof_idx)
        robot.control_dofs_position(home[: len(arm_dof_idx)], arm_dof_idx)
    if gripper_demo and gripper_dof_idx:
        setup_gripper_pd(robot, gripper_dof_idx, all_gripper_dof_idx)
        set_gripper_pose(robot, gripper_dof_idx, all_gripper_dof_idx, GRIPPER_OPEN)
    elif gripper_demo and lite6_gripper_dof_idx:
        setup_lite6_gripper_pd(robot, lite6_gripper_dof_idx, all_lite6_gripper_dof_idx)
        set_lite6_gripper_pose(
            robot,
            lite6_gripper_dof_idx,
            all_lite6_gripper_dof_idx,
            LITE6_GRIPPER_OPEN,
        )
    elif gripper_demo and bio_gripper_g2_dof_idx:
        bio_gripper.setup_pd()
        bio_gripper.set_pose(BIO_GRIPPER_G2_OPEN)

    warmup_steps = 3 if arm_kinematic_hold else 100
    for _ in range(warmup_steps):
        if pd_demo and arm_dof_idx:
            robot.control_dofs_position(home[: len(arm_dof_idx)], arm_dof_idx)
        if tcp_marker is not None and ee_link is not None:
            update_tcp_marker(tcp_marker, ee_link)
        _kinematic_step(
            scene,
            robot,
            arm_kinematic_hold=arm_kinematic_hold,
            idle_gripper_kinematic_hold=idle_gripper_kinematic_hold,
            arm_dof_idx=arm_dof_idx,
            home=home,
            all_gripper_dof_idx=all_gripper_dof_idx,
            all_lite6_gripper_dof_idx=all_lite6_gripper_dof_idx,
            all_bio_gripper_g2_dof_idx=all_bio_gripper_g2_dof_idx,
            bio_gripper=bio_gripper,
        )

    if headless:
        return
    start_deferred_viewer(scene)

    if gripper_demo:
        print(f"Viewer: {profile.key} — gripper open/close demo")
    elif pd_demo:
        print(f"Viewer: {profile.key} — joint motion demo ({_ARM_DEMO_SPEED_RAD_S:.4f} rad/s, once)")
    else:
        print(f"Viewer: {profile.key} ({profile.dof} DOF). Close window or Ctrl+C to exit.")

    joint_demo = _ArmJointDemo(_demo_poses(profile), len(arm_dof_idx)) if pd_demo and arm_dof_idx else None
    demo_done_announced = False
    step = 0
    last_gripper_phase = -1
    while True:
        arm_target = None
        if joint_demo is not None and not joint_demo.finished:
            _, q, _ = joint_demo.step()
            robot.set_dofs_position(q, arm_dof_idx, zero_velocity=True)
            arm_target = q
        elif joint_demo is not None:
            if not demo_done_announced:
                print("  Joint motion demo complete.")
                demo_done_announced = True
            robot.set_dofs_position(joint_demo.current, arm_dof_idx, zero_velocity=True)
            arm_target = joint_demo.current
        elif pd_demo and arm_dof_idx:
            robot.control_dofs_position(home[: len(arm_dof_idx)], arm_dof_idx)
        if gripper_demo and gripper_dof_idx:
            grip_phase = (step // 200) % 2
            if grip_phase != last_gripper_phase:
                label = "closed" if grip_phase else "open"
                print(f"  Gripper target: {label}")
                last_gripper_phase = grip_phase
            control_gripper_pose(
                robot,
                gripper_dof_idx,
                all_gripper_dof_idx,
                gripper_demo_target(step),
            )
        elif gripper_demo and lite6_gripper_dof_idx:
            grip_phase = (step // 200) % 2
            if grip_phase != last_gripper_phase:
                label = "closed" if grip_phase else "open"
                print(f"  Lite6 gripper target: {label}")
                last_gripper_phase = grip_phase
            control_lite6_gripper_pose(
                robot,
                lite6_gripper_dof_idx,
                all_lite6_gripper_dof_idx,
                lite6_gripper_demo_target(step),
            )
        elif gripper_demo and bio_gripper_g2_dof_idx:
            grip_phase = (step // 200) % 2
            if grip_phase != last_gripper_phase:
                label = "closed" if grip_phase else "open"
                print(f"  Bio Gripper G2 target: {label}")
                last_gripper_phase = grip_phase
            bio_gripper.control_pose(bio_gripper_g2_demo_target(step))
        if tcp_marker is not None and ee_link is not None:
            update_tcp_marker(tcp_marker, ee_link)
        _kinematic_step(
            scene,
            robot,
            arm_kinematic_hold=arm_kinematic_hold,
            idle_gripper_kinematic_hold=idle_gripper_kinematic_hold,
            arm_dof_idx=arm_dof_idx,
            home=home,
            all_gripper_dof_idx=all_gripper_dof_idx,
            all_lite6_gripper_dof_idx=all_lite6_gripper_dof_idx,
            all_bio_gripper_g2_dof_idx=all_bio_gripper_g2_dof_idx,
            bio_gripper=bio_gripper,
            arm_target=arm_target,
        )
        step += 1
        time.sleep(0.01)


def _lite6_demo_poses_deg() -> list[list[float]]:
    """Lite6 PD preview: J1 cycles 0→-90→0°, then J3 cycles 0→90→0° (other joints at 0)."""
    home = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return [
        home,
        [-90.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        home,
        [0.0, 0.0, 90.0, 0.0, 0.0, 0.0],
        home,
    ]


def _sdk_move_joint_poses_deg(dof: int) -> list[list[float]]:
    """Joint targets from xArm-Python-SDK example/wrapper/*/2001-move_joint.py (degrees)."""
    if dof == 5:
        return [
            [90, 0, 0, 0, 0],
            [90, 0, -60, 0, 0],
            [90, -30, -60, 0, 0],
            [0, -30, -60, 0, 0],
            [0, 0, -60, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    if dof == 7:
        return [
            [90, 0, 0, 0, 0, 0, 0],
            [90, -60, 0, 0, 0, 0, 0],
            [90, -60, -30, 0, 0, 0, 0],
            [0, -60, -30, 0, 0, 0, 0],
            [0, -60, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    return [
        [90, 0, 0, 0, 0, 0],
        [90, 0, -60, 0, 0, 0],
        [90, -30, -60, 0, 0, 0],
        [0, -30, -60, 0, 0, 0],
        [0, 0, -60, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]


def _demo_poses(profile: RobotModelSpec) -> list[np.ndarray]:
    """PD preview joint targets (radians). Lite6 uses J1/J3 sweep; others use SDK 2001-move_joint."""
    if profile.key == "lite6":
        rows = _lite6_demo_poses_deg()
    else:
        rows = _sdk_move_joint_poses_deg(profile.dof)
    return [np.radians(row[: profile.dof], dtype=float) for row in rows]
