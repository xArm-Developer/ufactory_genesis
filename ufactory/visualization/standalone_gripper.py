"""Shared Genesis loop and tool registry for standalone end-effector previews."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import count

import genesis as gs

from ufactory.robots.paths import (
    bio_gripper_g2_movable_visual_urdf,
    ft_sensor_ai1500_urdf,
    gripper_g2_movable_visual_urdf,
    lite6_gripper_movable_visual_urdf,
    lite6_mounting_plate_urdf,
    lite6_vacuum_gripper_visual_glb_urdf,
    robotiq_2f_85_movable_visual_urdf,
    robotiq_2f_140_movable_visual_urdf,
    robotiq_mounting_urdf,
    uf850_mounting_plate_urdf,
    vacuum_gripper_urdf,
    xarm_mounting_plate_urdf,
)
from ufactory.simulation import make_rigid_options
from ufactory.simulation.collision import configure_lite6_collision_filters
from ufactory.simulation.compat import require_genesis_runtime
from ufactory.visualization.glb import glb_pbr_surfaces, glb_view_surface
from ufactory.visualization.viewer import start_deferred_viewer

CAMERA_POS = (0.22, -0.28, 0.18)
CAMERA_LOOKAT = (0.0, 0.0, 0.14)
CAMERA_FOV = 35
PLATE_CAMERA_POS = (0.24, -0.30, 0.22)
PLATE_CAMERA_LOOKAT = (0.0, 0.0, 0.005)
ADAPTER_CAMERA_POS = (0.14, -0.18, 0.13)
ADAPTER_CAMERA_LOOKAT = (0.0, 0.0, 0.008)
# Same as the view_robot.py warm-up for calibrated grippers.
VIEWER_WARMUP_STEPS = 3
# Mounted at home on an arm: flange / adapter turned 180 deg about X (flange up, tool tip down).
MOUNTED_QUAT_WXYZ = (0.0, 1.0, 0.0, 0.0)
# Mounting plates sit upright on the floor (top mounting surface facing +Z).
UPRIGHT_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class StandaloneToolSpec:
    """Configuration for previewing one standalone flange tool or mounting plate."""

    name: str
    description: str
    urdf_loader: Callable[[], str]
    base_pos: tuple[float, float, float]
    base_quat: tuple[float, float, float, float] = MOUNTED_QUAT_WXYZ
    controller: Callable[[object], Callable[[int], None]] | None = None
    calibrated_g2: bool = False
    calibrated_bio_g2: bool = False
    calibrated_robotiq: str | None = None
    camera_pos: tuple[float, float, float] = CAMERA_POS
    camera_lookat: tuple[float, float, float] = CAMERA_LOOKAT


def _gripper_g2_controller(robot: object) -> Callable[[int], None]:
    from ufactory.visualization.gripper_g2_viewer import (
        control_gripper_pose,
        gripper_dof_indices,
        setup_gripper_pd,
        step_gripper_demo_target,
    )

    drive_idx, all_idx = gripper_dof_indices(robot)
    setup_gripper_pd(robot, drive_idx, all_idx)
    return lambda _step: control_gripper_pose(robot, drive_idx, all_idx, step_gripper_demo_target(robot)[0])


def _bio_gripper_g2_controller(robot: object) -> Callable[[int], None]:
    from ufactory.grippers.bio_g2 import BioGripperG2

    gripper = BioGripperG2(robot)
    gripper.setup_pd()
    return lambda _step: gripper.control_pose(gripper.step_demo_target()[0])


def _lite6_gripper_controller(robot: object) -> Callable[[int], None]:
    from ufactory.visualization.lite6_gripper_viewer import (
        LITE6_GRIPPER_OPEN,
        control_lite6_gripper_pose,
        lite6_gripper_dof_indices,
        set_lite6_gripper_pose,
        setup_lite6_gripper_pd,
        step_lite6_gripper_demo_target,
    )

    drive_idx, all_idx = lite6_gripper_dof_indices(robot)
    setup_lite6_gripper_pd(robot, drive_idx, all_idx)
    set_lite6_gripper_pose(robot, drive_idx, all_idx, LITE6_GRIPPER_OPEN)
    return lambda _step: control_lite6_gripper_pose(
        robot,
        drive_idx,
        all_idx,
        step_lite6_gripper_demo_target(robot, drive_idx)[0],
    )


def _robotiq_controller(variant: str) -> Callable[[object], Callable[[int], None]]:
    def _setup(robot: object) -> Callable[[int], None]:
        from ufactory.visualization.robotiq_viewer import (
            control_robotiq_pose,
            robotiq_dof_indices,
            setup_robotiq_pd,
            step_robotiq_demo_target,
        )

        resolved, drive_idx, all_idx = robotiq_dof_indices(robot, variant)
        assert resolved is not None
        setup_robotiq_pd(robot, resolved, drive_idx, all_idx)
        return lambda _step: control_robotiq_pose(
            robot,
            resolved,
            drive_idx,
            all_idx,
            step_robotiq_demo_target(robot, resolved)[0],
        )

    return _setup


STANDALONE_TOOL_SPECS: dict[str, StandaloneToolSpec] = {
    "gripper_g2": StandaloneToolSpec(
        name="gripper_g2",
        description="Gripper G2 open/close demo (standalone)",
        urdf_loader=gripper_g2_movable_visual_urdf,
        base_pos=(0.0, 0.0, 0.35),
        controller=_gripper_g2_controller,
        calibrated_g2=True,
    ),
    "robotiq_2f_85": StandaloneToolSpec(
        name="robotiq_2f_85",
        description="Robotiq 2F-85 open/close demo (standalone)",
        urdf_loader=robotiq_2f_85_movable_visual_urdf,
        base_pos=(0.0, 0.0, 0.35),
        controller=_robotiq_controller("robotiq_2f_85"),
        calibrated_robotiq="robotiq_2f_85",
    ),
    "robotiq_2f_140": StandaloneToolSpec(
        name="robotiq_2f_140",
        description="Robotiq 2F-140 open/close demo (standalone)",
        urdf_loader=robotiq_2f_140_movable_visual_urdf,
        base_pos=(0.0, 0.0, 0.40),
        controller=_robotiq_controller("robotiq_2f_140"),
        calibrated_robotiq="robotiq_2f_140",
    ),
    "robotiq_mounting": StandaloneToolSpec(
        name="robotiq_mounting",
        description="xArm/UF850-to-Robotiq 11 mm flange adapter static preview (standalone)",
        urdf_loader=robotiq_mounting_urdf,
        base_pos=(0.0, 0.0, 0.0056),
        base_quat=UPRIGHT_QUAT_WXYZ,
        camera_pos=ADAPTER_CAMERA_POS,
        camera_lookat=ADAPTER_CAMERA_LOOKAT,
    ),
    "ft_sensor_ai1500": StandaloneToolSpec(
        name="ft_sensor_ai1500",
        description="UFACTORY AI1500 six-axis force/torque sensor static preview (standalone)",
        urdf_loader=ft_sensor_ai1500_urdf,
        base_pos=(0.0, 0.0, 0.005),
        base_quat=UPRIGHT_QUAT_WXYZ,
        camera_pos=(0.18, -0.22, 0.14),
        camera_lookat=(0.0, 0.0, 0.03),
    ),
    "bio_gripper_g2": StandaloneToolSpec(
        name="bio_gripper_g2",
        description="Bio Gripper G2 open/close demo (standalone)",
        urdf_loader=bio_gripper_g2_movable_visual_urdf,
        base_pos=(0.0, 0.0, 0.29),
        controller=_bio_gripper_g2_controller,
        calibrated_bio_g2=True,
    ),
    "lite6_gripper": StandaloneToolSpec(
        name="lite6_gripper",
        description="Lite6 gripper open/close demo (standalone)",
        urdf_loader=lite6_gripper_movable_visual_urdf,
        base_pos=(0.0, 0.0, 0.30),
        controller=_lite6_gripper_controller,
    ),
    "vacuum_gripper": StandaloneToolSpec(
        name="vacuum_gripper",
        description="Vacuum gripper static preview (standalone)",
        urdf_loader=vacuum_gripper_urdf,
        base_pos=(0.0, 0.0, 0.20),
    ),
    "lite6_vacuum_gripper": StandaloneToolSpec(
        name="lite6_vacuum_gripper",
        description="Lite6 vacuum gripper static preview (standalone)",
        urdf_loader=lite6_vacuum_gripper_visual_glb_urdf,
        base_pos=(0.0, 0.0, 0.17),
    ),
    "xarm_mounting_plate": StandaloneToolSpec(
        name="xarm_mounting_plate",
        description="xArm5/6/7 mounting plate static preview (standalone)",
        urdf_loader=xarm_mounting_plate_urdf,
        base_pos=(0.0, 0.0, 0.008),
        base_quat=UPRIGHT_QUAT_WXYZ,
        camera_pos=PLATE_CAMERA_POS,
        camera_lookat=PLATE_CAMERA_LOOKAT,
    ),
    "uf850_mounting_plate": StandaloneToolSpec(
        name="uf850_mounting_plate",
        description="UF850 mounting plate static preview (standalone)",
        urdf_loader=uf850_mounting_plate_urdf,
        base_pos=(0.0, 0.0, 0.010),
        base_quat=UPRIGHT_QUAT_WXYZ,
        camera_pos=PLATE_CAMERA_POS,
        camera_lookat=PLATE_CAMERA_LOOKAT,
    ),
    "lite6_mounting_plate": StandaloneToolSpec(
        name="lite6_mounting_plate",
        description="Lite 6 mounting plate static preview (standalone)",
        urdf_loader=lite6_mounting_plate_urdf,
        base_pos=(0.0, 0.0, 0.005),
        base_quat=UPRIGHT_QUAT_WXYZ,
        camera_pos=PLATE_CAMERA_POS,
        camera_lookat=PLATE_CAMERA_LOOKAT,
    ),
}


def _open_viewer(scene) -> None:
    """Warm up, then open the window through the deferred viewer.

    A window opened at scene construction shows black or undefined frames until the scene renders.
    The deferred viewer only presents complete frames, and the first shows the initial tool pose.
    The warm-up steps leave the demo unchanged: the gripper holds the PD target set by its setup.
    """
    for _ in range(VIEWER_WARMUP_STEPS):
        scene.step()
    start_deferred_viewer(scene)


def _add_calibrated_args(parser: argparse.ArgumentParser, *, bio_g2: bool) -> None:
    parser.add_argument("--position-mm", type=float, default=71 if bio_g2 else 0)
    parser.add_argument("--speed-percent", type=float, default=50)
    if bio_g2:
        parser.add_argument("--speed-mm-s", type=float, default=None)
    parser.add_argument("--force-percent", type=float, default=50)


def _validate_calibrated_args(
    args: argparse.Namespace,
    *,
    calibrated_g2: bool,
    calibrated_bio_g2: bool,
    calibrated_robotiq: str | None = None,
) -> None:
    if calibrated_g2:
        from ufactory.simulation.g2_model import checked

        for name, low, high in (("position_mm", 0, 84), ("speed_percent", 0, 100), ("force_percent", 1, 100)):
            checked(getattr(args, name), low, high, name)
    elif calibrated_bio_g2:
        from ufactory.simulation.bio_g2_model import MAX_SPEED_MM_S, MIN_SPEED_MM_S, checked

        for name, low, high in (("position_mm", 71, 150), ("speed_percent", 0, 100), ("force_percent", 1, 100)):
            checked(getattr(args, name), low, high, name)
        if args.speed_mm_s is not None:
            checked(args.speed_mm_s, MIN_SPEED_MM_S, MAX_SPEED_MM_S, "speed_mm_s")
    elif calibrated_robotiq is not None:
        from ufactory.grippers.robotiq import get_robotiq_spec
        from ufactory.simulation.g2_model import checked

        open_mm = get_robotiq_spec(calibrated_robotiq).open_gap_mm
        for name, low, high in (("position_mm", 0, open_mm), ("speed_percent", 0, 100), ("force_percent", 1, 100)):
            checked(getattr(args, name), low, high, name)


def parse_standalone_tool_args(argv: Sequence[str] | None = None) -> tuple[StandaloneToolSpec, argparse.Namespace]:
    """Parse CLI arguments for the unified standalone tool viewer and resolve the matching tool spec."""
    parser = argparse.ArgumentParser(description="View any standalone flange tool or mounting plate in Genesis")
    parser.add_argument(
        "--tool",
        required=True,
        choices=tuple(STANDALONE_TOOL_SPECS),
        help="Flange tool (mounted orientation) or mounting plate (upright on floor) to preview",
    )
    parser.add_argument("--headless", action="store_true", help="Run without viewer window")
    parser.add_argument(
        "--backend",
        choices=("cpu", "gpu"),
        default="gpu",
        help="Genesis backend (use cpu without a supported GPU)",
    )
    parser.add_argument("--position-mm", type=float, default=None, help="Target jaw opening (mm)")
    parser.add_argument("--speed-percent", type=float, default=None, help="Gripper speed (%%)")
    parser.add_argument(
        "--speed-mm-s",
        type=float,
        default=None,
        help="Bio Gripper G2 speed (mm/s, overrides --speed-percent)",
    )
    parser.add_argument("--force-percent", type=float, default=None, help="Gripper force limit (%%)")
    args = parser.parse_args(argv)

    spec = STANDALONE_TOOL_SPECS[args.tool]
    gripper_opts = ("--position-mm", "--speed-percent", "--speed-mm-s", "--force-percent")
    if not (spec.calibrated_g2 or spec.calibrated_bio_g2 or spec.calibrated_robotiq):
        bad = [flag for flag in gripper_opts if getattr(args, flag.lstrip("-").replace("-", "_")) is not None]
        if bad:
            parser.error(f"{', '.join(bad)} not supported for --tool {spec.name}")
    else:
        if (spec.calibrated_g2 or spec.calibrated_robotiq) and args.speed_mm_s is not None:
            parser.error("--speed-mm-s is only supported for --tool bio_gripper_g2")
        if args.position_mm is None:
            args.position_mm = 71.0 if spec.calibrated_bio_g2 else 0.0
        if args.speed_percent is None:
            args.speed_percent = 50.0
        if args.force_percent is None:
            args.force_percent = 50.0
        try:
            _validate_calibrated_args(
                args,
                calibrated_g2=spec.calibrated_g2,
                calibrated_bio_g2=spec.calibrated_bio_g2,
                calibrated_robotiq=spec.calibrated_robotiq,
            )
        except ValueError as exc:
            parser.error(str(exc))
    return spec, args


def _run_standalone_viewer(
    urdf_path: str,
    controller: Callable[[object], Callable[[int], None]] | None,
    *,
    backend: str,
    headless: bool,
    headless_steps: int,
    calibrated_g2: bool = False,
    calibrated_bio_g2: bool = False,
    calibrated_robotiq: str | None = None,
    base_pos: tuple[float, float, float] | None = None,
    base_quat: tuple[float, float, float, float] | None = None,
    position_mm: float | None = None,
    speed_percent: float = 50.0,
    speed_mm_s: float | None = None,
    force_percent: float = 50.0,
    camera_pos: tuple[float, float, float] = CAMERA_POS,
    camera_lookat: tuple[float, float, float] = CAMERA_LOOKAT,
) -> dict | None:
    from ufactory.simulation import genesis_backend_constant

    calibrated = calibrated_g2 or calibrated_bio_g2 or calibrated_robotiq is not None
    require_genesis_runtime(gs)
    gs.init(backend=genesis_backend_constant(gs, backend), logging_level="error")
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=0.02, substeps=32) if calibrated else gs.options.SimOptions(dt=0.01),
        rigid_options=(
            make_rigid_options(gs, constraint_timeconst=0.00125 if (calibrated_g2 or calibrated_robotiq) else 0.002)
            if calibrated
            else make_rigid_options(gs)
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos, camera_lookat=camera_lookat, camera_fov=CAMERA_FOV
        ),
    )
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    pose: dict[str, tuple[float, ...]] = {}
    if base_pos is not None:
        pose["pos"] = base_pos
    if base_quat is not None:
        pose["quat"] = base_quat
    with glb_pbr_surfaces():
        robot = scene.add_entity(gs.morphs.URDF(file=urdf_path, fixed=True, **pose), surface=glb_view_surface())
    configure_lite6_collision_filters(robot)
    scene.build()
    if calibrated_g2:
        robot._ufactory_g2_demo_settings = (0.0 if position_mm is None else position_mm, speed_percent, force_percent)
    elif calibrated_robotiq is not None:
        robot._ufactory_robotiq_demo_settings = (
            0.0 if position_mm is None else position_mm,
            speed_percent,
            force_percent,
        )
    elif calibrated_bio_g2:
        robot._ufactory_bio_g2_demo_settings = (
            71.0 if position_mm is None else position_mm,
            speed_percent,
            force_percent,
            speed_mm_s,
        )

    step_controller = controller(robot) if controller is not None else None
    if not headless:
        _open_viewer(scene)
    gaps: list[float] = []
    target_steps = headless_steps if controller is not None else min(headless_steps, 10)
    for step in count():
        if step_controller is not None:
            step_controller(step)
        scene.step()
        if calibrated_g2:
            gaps.append(robot._ufactory_g2_controller.gap_mm())
        elif calibrated_robotiq is not None:
            gaps.append(robot._ufactory_robotiq_controller.gap_mm())
        elif calibrated_bio_g2:
            gaps.append(robot._ufactory_bio_g2_controller.gap_mm())
        demo_cycle = (
            getattr(robot, "_ufactory_g2_demo_cycle", None)
            or getattr(robot, "_ufactory_robotiq_demo_cycle", None)
            or getattr(robot, "_ufactory_bio_g2_demo_cycle", None)
            or getattr(robot, "_ufactory_lite6_demo_cycle", None)
        )
        if demo_cycle is not None:
            demo_cycle.on_step_completed(robot)
        if headless and (
            step + 1 >= target_steps
            or (
                controller is not None
                and headless_steps == 600
                and demo_cycle is not None
                and demo_cycle.completed_cycles >= 1
            )
        ):
            break
    if calibrated:
        return {"min_gap_mm": min(gaps), "max_gap_mm": max(gaps), "final_gap_mm": gaps[-1], "ticks": len(gaps)}
    return None


def run_standalone_tool_viewer(argv: Sequence[str] | None = None, *, headless_steps: int = 600) -> dict | None:
    """Run the unified standalone tool viewer from CLI flags or ``argv``."""
    spec, args = parse_standalone_tool_args(argv)
    return _run_standalone_viewer(
        spec.urdf_loader(),
        spec.controller,
        backend=args.backend,
        headless=args.headless,
        headless_steps=headless_steps,
        calibrated_g2=spec.calibrated_g2,
        calibrated_bio_g2=spec.calibrated_bio_g2,
        calibrated_robotiq=spec.calibrated_robotiq,
        base_pos=spec.base_pos,
        base_quat=spec.base_quat,
        position_mm=args.position_mm,
        speed_percent=50.0 if args.speed_percent is None else args.speed_percent,
        speed_mm_s=args.speed_mm_s,
        force_percent=50.0 if args.force_percent is None else args.force_percent,
        camera_pos=spec.camera_pos,
        camera_lookat=spec.camera_lookat,
    )


def run_standalone_gripper_viewer(
    description: str,
    urdf_path: str,
    controller: Callable[[object], Callable[[int], None]] | None = None,
    *,
    headless_steps: int = 600,
    calibrated_g2: bool = False,
    calibrated_bio_g2: bool = False,
    base_pos: tuple[float, float, float] | None = None,
    base_quat: tuple[float, float, float, float] | None = None,
) -> dict | None:
    """Run a tool demo, with the URDF root placed at ``base_pos`` / ``base_quat`` (w, x, y, z) when given."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--headless", action="store_true", help="Run without viewer window")
    parser.add_argument(
        "--backend",
        choices=("cpu", "gpu"),
        default="gpu",
        help="Genesis backend (use cpu without a supported GPU)",
    )
    if calibrated_g2 or calibrated_bio_g2:
        _add_calibrated_args(parser, bio_g2=calibrated_bio_g2)
    args = parser.parse_args()
    _validate_calibrated_args(args, calibrated_g2=calibrated_g2, calibrated_bio_g2=calibrated_bio_g2)
    return _run_standalone_viewer(
        urdf_path,
        controller,
        backend=args.backend,
        headless=args.headless,
        headless_steps=headless_steps,
        calibrated_g2=calibrated_g2,
        calibrated_bio_g2=calibrated_bio_g2,
        base_pos=base_pos,
        base_quat=base_quat,
        position_mm=getattr(args, "position_mm", None),
        speed_percent=getattr(args, "speed_percent", 50.0),
        speed_mm_s=getattr(args, "speed_mm_s", None),
        force_percent=getattr(args, "force_percent", 50.0),
    )
