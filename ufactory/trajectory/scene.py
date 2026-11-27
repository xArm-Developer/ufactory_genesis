"""Scene construction for the trajectory pick-place pipeline.

Robot-generic Genesis scene builder for the pick-place trajectory pipeline
(originally written for xArm6 scripted pick-place demos). Returns a typed
context (entities, dof indices, PD gains, finger offsets, key heights) reused
across xArm5/6/7, UF850 (Gripper G2 / Bio Gripper G2) and Lite6 (parallel gripper). Single
environment, robot base at table height.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

import genesis as gs
from genesis.utils.geom import xyz_to_quat
from ufactory.config import load_runtime_config, resolve_pick_place_object_spec
from ufactory.kinematics.calibration import prepare_robot_model_for_verification
from ufactory.kinematics.genesis import solve_home_ik
from ufactory.kinematics.orientation import GRIPPER_DOWN_RPY_RAD
from ufactory.visualization.glb import glb_pbr_surfaces, glb_view_surface
from ufactory.simulation import (
    DEFAULT_CONSTRAINT_SOLVER,
    DEFAULT_CONSTRAINT_TIMECONST,
    DEFAULT_CONTACT_RESOLUTION,
    DEFAULT_FRICTION_CONE,
    DEFAULT_NOSLIP_ITERATIONS,
    DEFAULT_SOLVER_ITERATIONS,
    GenesisRuntimeManager,
    configure_g2_mimic_constraints,
    configure_robotiq_mimic_constraints,
    make_rigid_options,
    validate_g2_contact_substeps,
    validate_robotiq_contact_substeps,
)
from ufactory.simulation.collision import apply_collision_geometry_filters
from ufactory.robots.assembly import compose_robot_urdf
from ufactory.robots.registry import joint_names
from ufactory.robots.runtime import GripperControlParams, RobotRuntimeProfile, get_robot_runtime_profile

TABLE_HEIGHT = 0.4  # meters; robot base sits on the table surface
DEFAULT_ROBOT_BASE_POS = (0.30, 0.00, TABLE_HEIGHT)
# Lite6 has a 440 mm reach (vs. 700-850 mm for the other arms). The installed
# reversed Lite6 gripper has a 20-38 mm physical opening, so the default task
# uses the shared cube with closer-in configured workspace coordinates.

# Default grasp gaps intentionally close past the 30 mm cube faces so Genesis
# rigid contact and friction decide whether the block can be carried. Gripper
# G2 uses a 22 mm target: enough preload for the 30 mm cube while avoiding the
# visible contact chatter caused by stronger 15 mm over-closure. Reversed Lite6
# is limited by its physical 20 mm minimum opening.
DEFAULT_GRIPPER_GRASP_GAP_M = 0.022
LITE6_DEFAULT_GRIPPER_GRASP_GAP_M = 0.020

RIGID_SOLVER_ITERATIONS = DEFAULT_SOLVER_ITERATIONS
RIGID_NOSLIP_ITERATIONS = DEFAULT_NOSLIP_ITERATIONS
RIGID_CONSTRAINT_TIMECONST = DEFAULT_CONSTRAINT_TIMECONST
# Painted wood block and silicone fingertip pads. Size and mass come from the
# shared pick-place runtime configuration; contact stiffness remains code policy.
OBJ_FRICTION = 1.0
FINGER_FRICTION = 1.2


@dataclass
class TrajSceneContext:
    scene: object
    robot: object
    obj: object
    target_marker: object
    ik_link: object
    left_finger: object | None
    right_finger: object | None
    arm_dof_idx: list[int]
    gripper_dof_idx: list[int]
    all_gripper_dof_idx: list[int]
    down_quat: torch.Tensor
    home_qpos: np.ndarray
    base_pos_world: tuple[float, float, float]
    visual_model: str
    finger_z_offset: float
    finger_y_gap: float
    # Key link6 heights in the **robot base frame** (z above the base / table).
    grasp_link6_z: float
    pre_grasp_link6_z: float
    lift_link6_z: float
    base_yaw_rad: float = 0.0
    control_all_gripper_dofs: bool = False
    control_all_gripper_dofs_on_open: bool = False
    gripper_open_dof_idx: list[int] = field(default_factory=list)
    # Optional staged opening: use these DOFs until the measured horizontal
    # finger span grows by ``gripper_initial_open_clearance_m``.
    gripper_initial_open_dof_idx: list[int] = field(default_factory=list)
    gripper_initial_open_clearance_m: float = 0.0
    # Binary grippers receive their final open/close target at command start;
    # the segment duration is retained as their mechanical settle window.
    gripper_binary_commands: bool = False
    robot_urdf_path: str = ""
    kinematics_yaml_path: str | None = None
    kinematics_suffix: str | None = None
    robot_key: str = "xarm6_1305"
    gripper: GripperControlParams | None = None
    home_pos_base: list[float] = field(default_factory=list)
    table_height: float = TABLE_HEIGHT
    obj_xy: tuple[float, float] = (0.0, 0.0)
    place_xy: tuple[float, float] = (0.0, 0.0)
    obj_size: tuple[float, float, float] = (0.0, 0.0, 0.0)
    obj_mass_kg: float = 0.0
    obj_pos_base: tuple[float, float, float] = (0.0, 0.0, 0.0)
    place_pos_base: tuple[float, float, float] = (0.0, 0.0, 0.0)
    g2_settings: object | None = None
    g2_controller: object | None = None
    robotiq_settings: object | None = None
    robotiq_controller: object | None = None
    simulation_parameters: dict = field(default_factory=dict)
    sim_rate_hz: float = 50.0
    arm_controller: object | None = None
    gripper_dof_multipliers: tuple[float, ...] = ()
    tool_tip_link: object | None = None
    tcp_offset_flange_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def base_to_world(self, pos_base) -> list[float]:
        """Convert a base-frame xyz (m) to the Genesis world frame (m)."""
        p = [float(pos_base[0]), float(pos_base[1]), float(pos_base[2])]
        c = math.cos(self.base_yaw_rad)
        s = math.sin(self.base_yaw_rad)
        return [
            self.base_pos_world[0] + c * p[0] - s * p[1],
            self.base_pos_world[1] + s * p[0] + c * p[1],
            self.base_pos_world[2] + p[2],
        ]

    def world_to_base(self, pos_world) -> list[float]:
        """Convert a Genesis world-frame xyz (m) to the robot base frame."""
        dx = float(pos_world[0]) - self.base_pos_world[0]
        dy = float(pos_world[1]) - self.base_pos_world[1]
        dz = float(pos_world[2]) - self.base_pos_world[2]
        c = math.cos(self.base_yaw_rad)
        s = math.sin(self.base_yaw_rad)
        return [c * dx + s * dy, -s * dx + c * dy, dz]


def _base_to_world(base_pos_world: tuple[float, float, float], pos_base) -> list[float]:
    p = [float(pos_base[0]), float(pos_base[1]), float(pos_base[2])]
    return [p[0] + base_pos_world[0], p[1] + base_pos_world[1], p[2] + base_pos_world[2]]


def _robot_urdf_for_visual_model(runtime: RobotRuntimeProfile, visual_model: str) -> tuple[str, bool]:
    """Resolve the sim URDF for ``robot_key`` at the requested visual fidelity.

    ``glb`` (default): GLB visuals + STL collision, movable gripper included.
    ``stl``: legacy all-STL visuals + collision (xArm6 only; kept for geometry
    alignment checks against the original demo).
    """
    model = str(visual_model).strip().lower()
    profile = runtime.model
    gripper = runtime.gripper
    if model == "stl":
        if profile.key != "xarm6_1305":
            raise ValueError(f"visual_model='stl' is only supported for xarm6_1305, got {profile.key!r}")
        return compose_robot_urdf(profile.key, "gripper_g2", visual="stl"), False
    if model != "glb":
        raise ValueError("visual_model must be 'glb' or 'stl'")
    if gripper is None:
        from ufactory.robots.paths import robot_visual_glb_urdf

        return robot_visual_glb_urdf(profile.key), True
    if gripper.family == "lite6":
        return compose_robot_urdf(profile.key, "lite6_gripper"), True
    if gripper.family == "g2":
        return compose_robot_urdf(profile.key, "gripper_g2"), True
    if gripper.family == "bio_gripper_g2":
        return compose_robot_urdf(profile.key, "bio_gripper_g2"), True
    if gripper.family in {"robotiq_2f_85", "robotiq_2f_140", "vacuum_gripper", "lite6_vacuum_gripper"}:
        return compose_robot_urdf(profile.key, gripper.family), True
    raise ValueError(f"{profile.key} has no supported gripper for the trajectory scene")


def gripper_dof_multipliers(gripper: GripperControlParams | None) -> tuple[float, ...]:
    """Return per-DOF sign multipliers aligned with ``gripper.all_joint_names``.

    In ``bio_gripper_g2``, ``bio_gripper_g2_left_finger_joint`` mimics the right
    finger with ``multiplier="-1"`` and joint limits ``[-0.0395, 0]``.
    """
    if gripper is None or not gripper.all_joint_names:
        return ()
    if gripper.family == "bio_gripper_g2":
        from ufactory.grippers.bio_g2 import LEFT_JOINT

        return tuple(-1.0 if name == LEFT_JOINT else 1.0 for name in gripper.all_joint_names)
    return tuple(1.0 for _ in gripper.all_joint_names)


def _robot_defaults(robot_key: str, *, gripper: str | None = None) -> dict:
    """Per-robot default scene geometry (object/place xy, key heights)."""
    from ufactory.manipulation.tcp import resolve_tool_tcp_spec

    runtime = get_robot_runtime_profile(robot_key, gripper=gripper)
    is_zero_dof_choice = gripper in {"bare", "vacuum_gripper", "lite6_vacuum_gripper"}
    config = load_runtime_config(runtime.model.key, gripper=None if is_zero_dof_choice else gripper)
    params = config.task.parameters
    object_spec = resolve_pick_place_object_spec(config)
    obj_pos_base = tuple(float(value) for value in params["fixed_object_position_m"])
    place_pos_base = tuple(float(value) for value in params["fixed_target_position_m"])
    default_ee = tuple(float(value) for value in params["default_ee_position_m"])
    is_lite6 = runtime.model.key == "lite6"
    if is_zero_dof_choice:
        tool_tcp = resolve_tool_tcp_spec(runtime.gripper)
        return {
            "base_pos": DEFAULT_ROBOT_BASE_POS,
            "obj_xy": obj_pos_base[:2],
            "place_xy": place_pos_base[:2],
            "obj_pos_base": obj_pos_base,
            "place_pos_base": place_pos_base,
            "obj_size": object_spec.size_m,
            "obj_mass_kg": object_spec.mass_kg,
            "lift_z": default_ee[2],
            "home_z": default_ee[2],
            "grasp_gap_m": 0.0,
            "tool_tip_offset_z_m": tool_tcp.tcp_offset_flange_m[2],
            "finger_pad_below_center_m": 0.0,
            "finger_close_descent_m": 0.0,
            "grasp_table_clearance_m": 0.010,
            "grasp_height_extra_m": 0.0,
            "excluded_geometry_pairs": (),
            "default_qpos_rad": config.arm.default_qpos_rad,
            "tcp_offset_flange_m": tool_tcp.tcp_offset_flange_m,
        }
    gripper_profile = config.gripper
    if gripper_profile is None:
        raise ValueError(f"{runtime.model.key} has no configured gripper geometry")
    tool_tcp = resolve_tool_tcp_spec(config)
    return {
        "base_pos": DEFAULT_ROBOT_BASE_POS,
        "obj_xy": obj_pos_base[:2],
        "place_xy": place_pos_base[:2],
        "obj_pos_base": obj_pos_base,
        "place_pos_base": place_pos_base,
        "obj_size": object_spec.size_m,
        "obj_mass_kg": object_spec.mass_kg,
        "lift_z": default_ee[2],
        "home_z": default_ee[2],
        "grasp_gap_m": LITE6_DEFAULT_GRIPPER_GRASP_GAP_M if is_lite6 else DEFAULT_GRIPPER_GRASP_GAP_M,
        "tool_tip_offset_z_m": gripper_profile.tool_tip_offset_z_m,
        "finger_pad_below_center_m": gripper_profile.finger_pad_below_center_m,
        "finger_close_descent_m": gripper_profile.finger_close_descent_m,
        "grasp_table_clearance_m": gripper_profile.grasp_table_clearance_m,
        "grasp_height_extra_m": gripper_profile.grasp_height_extra_m,
        "excluded_geometry_pairs": gripper_profile.excluded_geometry_pairs,
        "default_qpos_rad": config.arm.default_qpos_rad,
        "tcp_offset_flange_m": tool_tcp.tcp_offset_flange_m,
    }


@dataclass
class DryHeights:
    """Off-sim height/geometry estimate for the real-path dry-run/mirror flow.

    Mirrors the physically-derived constants :func:`build_scene` measures at
    runtime (finger offset, key link6 heights) without needing a live Genesis
    scene. The real arm uses per-unit calibrated forward kinematics at deploy
    time, so these base-frame heights match the sim plan up to the per-arm
    kinematics offset.
    """

    finger_z_offset: float
    home_pos_base: list[float]
    obj_xy: tuple[float, float]
    place_xy: tuple[float, float]
    grasp_link6_z: float
    pre_grasp_link6_z: float
    lift_link6_z: float
    gripper: GripperControlParams | None


def dry_heights(robot_key: str, *, gripper: str | None = None) -> DryHeights:
    """Build a :class:`DryHeights` for ``robot_key`` without a Genesis scene."""
    runtime = get_robot_runtime_profile(robot_key, gripper=gripper)
    resolved_gripper = runtime.gripper
    if resolved_gripper is None and gripper != "bare":
        raise ValueError(f"{runtime.model.key} has no supported gripper for the trajectory scene")
    defaults = _robot_defaults(runtime.model.key, gripper=gripper)
    obj_xy = defaults["obj_xy"]
    finger_z_offset = float(defaults["tool_tip_offset_z_m"])
    pad_below_fc = float(defaults["finger_pad_below_center_m"])
    close_descent = float(defaults["finger_close_descent_m"])
    table_clearance = float(defaults["grasp_table_clearance_m"])
    grasp_extra = float(defaults["grasp_height_extra_m"])
    grasp_link6_z = table_clearance + close_descent + pad_below_fc + finger_z_offset + grasp_extra
    return DryHeights(
        finger_z_offset=finger_z_offset,
        home_pos_base=[obj_xy[0], obj_xy[1], defaults["home_z"]],
        obj_xy=tuple(obj_xy),
        place_xy=tuple(defaults["place_xy"]),
        grasp_link6_z=grasp_link6_z,
        pre_grasp_link6_z=grasp_link6_z + 0.10,
        lift_link6_z=defaults["lift_z"],
        gripper=resolved_gripper,
    )


def build_scene(
    *,
    robot_key: str = "xarm6",
    rate: float = 50.0,
    show_viewer: bool = False,
    substeps: int = 32,
    constraint_solver: str = DEFAULT_CONSTRAINT_SOLVER,
    solver_iterations: int = RIGID_SOLVER_ITERATIONS,
    noslip_iterations: int = RIGID_NOSLIP_ITERATIONS,
    friction_cone: str = DEFAULT_FRICTION_CONE,
    contact_resolution: str = DEFAULT_CONTACT_RESOLUTION,
    constraint_timeconst: float | None = None,
    use_gjk_collision: bool | None = True,
    contact_sol_params: tuple[float, float, float, float, float, float, float] | None = None,
    base_pos: tuple[float, float, float] | None = None,
    visual_model: Literal["glb", "stl"] = "glb",
    kinematics_yaml: str | None = None,
    kinematics_suffix: str | None = None,
    kinematics_yaml_dir: str | None = None,
    obj_xy: tuple[float, float] | None = None,
    place_xy: tuple[float, float] | None = None,
    obj_pos_base: tuple[float, float, float] | None = None,
    place_pos_base: tuple[float, float, float] | None = None,
    obj_size: tuple[float, float, float] | None = None,
    obj_mass_kg: float | None = None,
    lift_z: float | None = None,
    home_z: float | None = None,
    calibrated_g2: bool = True,
    g2_settings: object | None = None,
    robotiq_settings: object | None = None,
    gripper: str | None = None,
    excluded_geometry_pairs: tuple[tuple[str, str], ...] | None = None,
    arm_controller: object | None = None,
    home_xy_base: tuple[float, float] | None = None,
) -> TrajSceneContext:
    """Build the single-env Genesis scene and configure PD gains + finger offsets.

    ``robot_key`` selects the arm profile (``xarm5``/``xarm6``/``xarm7``/``lite6``/``uf850``);
    the gripper is resolved from that robot's runtime profile or explicit ``gripper`` overlay.
    Full base-frame object positions are preferred. Legacy ``obj_xy``/``place_xy`` inputs
    remain supported and use half the configured object height as their table-resting z coordinate.
    """
    runtime = get_robot_runtime_profile(robot_key, gripper=gripper)
    profile = runtime.model
    resolved_gripper = runtime.gripper
    if resolved_gripper is None and gripper != "bare":
        raise ValueError(f"{profile.key} has no supported gripper for the trajectory scene")
    defaults = _robot_defaults(profile.key, gripper=gripper)
    gripper_spec = resolved_gripper
    gripper_family = gripper_spec.family if gripper_spec is not None else None
    # Keep local name ``gripper`` bound cleanly below.

    dt = 1.0 / float(rate)
    if gripper_family == "g2":
        validate_g2_contact_substeps(dt=dt, substeps=substeps)
    elif gripper_family in {"robotiq_2f_85", "robotiq_2f_140"}:
        validate_robotiq_contact_substeps(dt=dt, substeps=substeps, variant=gripper_family)
    substeps = int(substeps)

    robot_base = tuple(float(v) for v in (base_pos if base_pos is not None else defaults["base_pos"]))
    obj_size = tuple(float(value) for value in (obj_size if obj_size is not None else defaults["obj_size"]))
    if len(obj_size) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in obj_size):
        raise ValueError("obj_size must contain three positive finite values")
    if obj_pos_base is not None and obj_xy is not None:
        raise ValueError("obj_pos_base and obj_xy are mutually exclusive")
    if place_pos_base is not None and place_xy is not None:
        raise ValueError("place_pos_base and place_xy are mutually exclusive")
    rest_center_z = obj_size[2] / 2.0
    if obj_pos_base is None:
        selected_xy = tuple(obj_xy) if obj_xy is not None else defaults["obj_xy"]
        obj_pos_base = (float(selected_xy[0]), float(selected_xy[1]), rest_center_z)
    else:
        obj_pos_base = tuple(float(value) for value in obj_pos_base)
    if place_pos_base is None:
        selected_xy = tuple(place_xy) if place_xy is not None else defaults["place_xy"]
        place_pos_base = (float(selected_xy[0]), float(selected_xy[1]), rest_center_z)
    else:
        place_pos_base = tuple(float(value) for value in place_pos_base)
    if len(obj_pos_base) != 3 or not all(math.isfinite(value) for value in obj_pos_base):
        raise ValueError("obj_pos_base must contain three finite values")
    if len(place_pos_base) != 3 or not all(math.isfinite(value) for value in place_pos_base):
        raise ValueError("place_pos_base must contain three finite values")
    obj_mass_kg = float(defaults["obj_mass_kg"] if obj_mass_kg is None else obj_mass_kg)
    if not math.isfinite(obj_mass_kg) or obj_mass_kg <= 0.0:
        raise ValueError("obj_mass_kg must be finite and positive")
    obj_xy = obj_pos_base[:2]
    place_xy = place_pos_base[:2]
    if home_xy_base is not None:
        if len(home_xy_base) != 2 or not all(math.isfinite(float(value)) for value in home_xy_base):
            raise ValueError("home_xy_base must contain two finite values")
        resolved_home_xy = (float(home_xy_base[0]), float(home_xy_base[1]))
    else:
        resolved_home_xy = obj_xy
    lift_z = float(lift_z) if lift_z is not None else defaults["lift_z"]
    home_z = float(home_z) if home_z is not None else defaults["home_z"]

    GenesisRuntimeManager.active()
    robot_urdf_path, use_glb_visual = _robot_urdf_for_visual_model(runtime, visual_model)
    kinematics_yaml_path = None
    if kinematics_yaml is not None or kinematics_suffix is not None:
        robot_urdf_path, kinematics_yaml_path = prepare_robot_model_for_verification(
            None,
            kinematics_yaml,
            kinematics_suffix,
            kinematics_yaml_dir,
            default_base_urdf=robot_urdf_path,
            robot_name=profile.robot_name,
            joint_count=profile.dof,
        )
    is_robotiq = gripper_family in {"robotiq_2f_85", "robotiq_2f_140"}
    is_calibrated_g2 = calibrated_g2 and gripper_family == "g2"
    is_calibrated_robotiq = calibrated_g2 and is_robotiq
    is_calibrated_adaptive = is_calibrated_g2 or is_calibrated_robotiq
    has_finger_links = bool(gripper_spec is not None and gripper_spec.has_finger_links)
    has_actuated_joints = bool(gripper_spec is not None and gripper_spec.has_actuated_joints)
    if constraint_timeconst is None:
        constraint_timeconst = 0.00125 if is_calibrated_adaptive else RIGID_CONSTRAINT_TIMECONST
    substep_dt = dt / int(substeps)
    constraint_timeconst = max(float(constraint_timeconst), 2.0 * substep_dt)
    camera_lookat = _base_to_world(robot_base, (0.30, 0.15, 0.10))
    camera_pos = (camera_lookat[0] + 0.90, camera_lookat[1] - 1.35, camera_lookat[2] + 0.60)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=substeps),
        rigid_options=make_rigid_options(
            gs,
            constraint_solver=constraint_solver,
            friction_cone=friction_cone,
            contact_resolution=contact_resolution,
            enable_collision=True,
            enable_joint_limit=True,
            iterations=int(solver_iterations),
            noslip_iterations=int(noslip_iterations),
            constraint_timeconst=float(constraint_timeconst),
            use_gjk_collision=use_gjk_collision,
        ),
        viewer_options=gs.options.ViewerOptions(
            refresh_rate=50,
            camera_pos=camera_pos,
            camera_lookat=tuple(camera_lookat),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    scene.add_entity(
        gs.morphs.Box(
            size=(0.5, 0.8, TABLE_HEIGHT),
            pos=(0.45, 0.0, TABLE_HEIGHT / 2),
            fixed=True,
        ),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ColorTexture(color=(0.6, 0.6, 0.6)),
        ),
    )

    robot_morph = gs.morphs.URDF(
        file=robot_urdf_path,
        pos=robot_base,
        fixed=True,
        requires_jac_and_IK=True,
    )
    if use_glb_visual:
        with glb_pbr_surfaces():
            robot = scene.add_entity(robot_morph, surface=glb_view_surface())
    else:
        robot = scene.add_entity(robot_morph)

    obj_pos_world = _base_to_world(robot_base, obj_pos_base)
    place_pos_world = _base_to_world(robot_base, place_pos_base)
    obj = scene.add_entity(
        gs.morphs.Box(
            size=tuple(obj_size),
            pos=tuple(obj_pos_world),
            fixed=False,
        ),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ColorTexture(color=(0.9, 0.1, 0.1)),
        ),
    )
    target_marker = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.02,
            pos=tuple(place_pos_world),
            fixed=True,
            collision=False,
        ),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ColorTexture(color=(0.0, 1.0, 0.0)),
        ),
    )

    apply_collision_geometry_filters(
        robot, defaults["excluded_geometry_pairs"] if excluded_geometry_pairs is None else excluded_geometry_pairs
    )
    scene.build(n_envs=1)

    ee_link_name = profile.ee_link
    ik_link = robot.get_link(ee_link_name)
    left_finger = robot.get_link(gripper_spec.finger_link_names[0]) if has_finger_links and gripper_spec else None
    right_finger = robot.get_link(gripper_spec.finger_link_names[1]) if has_finger_links and gripper_spec else None
    tool_tip_link = None
    if gripper_spec is not None and gripper_spec.tool_tip_link:
        try:
            tool_tip_link = robot.get_link(gripper_spec.tool_tip_link)
        except Exception:
            tool_tip_link = None

    if gripper_family == "g2":
        configure_g2_mimic_constraints(robot)
    elif is_robotiq and gripper_family is not None:
        configure_robotiq_mimic_constraints(robot, gripper_family)
    if contact_sol_params is not None:
        sol_targets = tuple(item for item in (obj, left_finger, right_finger) if item is not None)
        _set_contact_sol_params(sol_targets, contact_sol_params)

    # Configured painted-wood cube + silicone fingertip pads (μ=1.2).
    # Stiffness: leave default rigid contact sol_params. Set before any stepping.
    obj.set_friction(float(OBJ_FRICTION))
    if left_finger is not None and right_finger is not None:
        left_finger.set_friction(float(FINGER_FRICTION))
        right_finger.set_friction(float(FINGER_FRICTION))
    if hasattr(obj, "set_links_mass"):
        obj.set_links_mass(torch.tensor([obj_mass_kg], device=gs.device, dtype=gs.tc_float))
    else:
        obj.set_links_inertial_mass(torch.tensor([obj_mass_kg], device=gs.device, dtype=gs.tc_float))

    jnames = joint_names(profile)
    arm_dof_idx = [robot.get_joint(n).dofs_idx_local[0] for n in jnames]
    if has_actuated_joints and gripper_spec is not None and gripper_spec.drive_joint is not None:
        gripper_dof_idx = [robot.get_joint(gripper_spec.drive_joint).dofs_idx_local[0]]
        all_gripper_dof_idx = [robot.get_joint(n).dofs_idx_local[0] for n in gripper_spec.all_joint_names]
    else:
        gripper_dof_idx = []
        all_gripper_dof_idx = []

    runtime_arm = runtime.arm
    robot.set_dofs_kp(
        torch.tensor(runtime_arm.kp, device=gs.device, dtype=gs.tc_float),
        arm_dof_idx,
    )
    robot.set_dofs_kv(
        torch.tensor(runtime_arm.kv, device=gs.device, dtype=gs.tc_float),
        arm_dof_idx,
    )
    robot.set_dofs_force_range(
        torch.tensor(runtime_arm.force_lower, device=gs.device, dtype=gs.tc_float),
        torch.tensor(runtime_arm.force_upper, device=gs.device, dtype=gs.tc_float),
        arm_dof_idx,
    )
    if has_actuated_joints and gripper_spec is not None:
        robot.set_dofs_kp(
            torch.tensor([gripper_spec.kp], device=gs.device, dtype=gs.tc_float),
            gripper_dof_idx,
        )
        robot.set_dofs_kv(
            torch.tensor([gripper_spec.kv], device=gs.device, dtype=gs.tc_float),
            gripper_dof_idx,
        )
        robot.set_dofs_force_range(
            torch.tensor([gripper_spec.force_lower], device=gs.device, dtype=gs.tc_float),
            torch.tensor([gripper_spec.force_upper], device=gs.device, dtype=gs.tc_float),
            gripper_dof_idx,
        )
        n_grip = len(all_gripper_dof_idx)
        robot.set_dofs_damping(
            torch.full((n_grip,), gripper_spec.damping, device=gs.device, dtype=gs.tc_float),
            all_gripper_dof_idx,
        )
        robot.set_dofs_frictionloss(
            torch.full((n_grip,), gripper_spec.frictionloss, device=gs.device, dtype=gs.tc_float),
            all_gripper_dof_idx,
        )
        if is_calibrated_g2:
            from ufactory.simulation.g2_adapter import configure_g2_drive

            configure_g2_drive(robot, damping=gripper_spec.damping, frictionloss=gripper_spec.frictionloss)
        elif is_calibrated_robotiq:
            from ufactory.simulation.robotiq_adapter import configure_robotiq_drive

            configure_robotiq_drive(
                robot,
                gripper_spec.family,
                damping=gripper_spec.damping,
                frictionloss=gripper_spec.frictionloss,
            )
        elif gripper_spec.family == "lite6":
            from ufactory.grippers.lite6 import configure_lite6_gripper_drive

            configure_lite6_gripper_drive(
                robot,
                gripper_dof_idx,
                all_gripper_dof_idx,
                kp=gripper_spec.kp,
                kv=gripper_spec.kv,
                force_lower_n=gripper_spec.force_lower,
                force_upper_n=gripper_spec.force_upper,
                damping=gripper_spec.damping,
                frictionloss=gripper_spec.frictionloss,
            )

    dq = xyz_to_quat(
        torch.tensor([GRIPPER_DOWN_RPY_RAD], device=gs.device, dtype=gs.tc_float),
        rpy=True,
        degrees=False,
    )
    home_pos_base = [resolved_home_xy[0], resolved_home_xy[1], home_z]
    home_pos = _base_to_world(robot_base, home_pos_base)
    home_link6_pos = torch.tensor([home_pos], device=gs.device, dtype=gs.tc_float)
    home_controller = arm_controller
    if home_controller is None and profile.dof == 5:
        from ufactory.control import ArmMotionController

        home_controller = ArmMotionController.from_robot(profile.key, urdf_path=robot_urdf_path)
    home_qpos_result = solve_home_ik(
        robot,
        ik_link,
        home_link6_pos,
        dq,
        arm_dof_idx=arm_dof_idx,
        arm_controller=home_controller,
        target_pos_base=home_pos_base,
        default_qpos_rad=defaults["default_qpos_rad"],
    )
    init_qpos = torch.zeros(1, robot.n_dofs, device=gs.device, dtype=gs.tc_float)
    for i, idx in enumerate(arm_dof_idx):
        init_qpos[:, idx] = home_qpos_result[0, arm_dof_idx[i]]
    multipliers = gripper_dof_multipliers(gripper_spec)
    if has_actuated_joints and gripper_spec is not None:
        for i, idx in enumerate(all_gripper_dof_idx):
            if is_calibrated_g2:
                from ufactory.simulation.g2_model import G2Model

                init_qpos[:, idx] = G2Model.load().linkage.drive_for_gap_m(0.084)
            elif is_calibrated_robotiq:
                from ufactory.simulation.robotiq_model import RobotiqModel

                init_qpos[:, idx] = RobotiqModel.load(gripper_spec.family).linkage.drive_for_gap_m(
                    gripper_spec.open_gap_m
                )
            else:
                init_qpos[:, idx] = gripper_spec.open_pos * multipliers[i]
    robot.set_qpos(init_qpos)
    scene.step()

    link6_pos = ik_link.get_pos()[0]
    tcp_offset_flange_m = tuple(float(v) for v in defaults["tcp_offset_flange_m"])
    if left_finger is not None and right_finger is not None:
        lf_pos = left_finger.get_pos()[0]
        rf_pos = right_finger.get_pos()[0]
        fc_pos = (lf_pos + rf_pos) / 2
        finger_z_offset = (link6_pos[2] - fc_pos[2]).item()
        # Finger-span across the open gap: fingers separate mainly along y when
        # the gripper points down, so use the larger of |dy|/|dx| as the span.
        finger_y_gap = max(abs(lf_pos[1] - rf_pos[1]).item(), abs(lf_pos[0] - rf_pos[0]).item())
    elif tool_tip_link is not None:
        tip_pos = tool_tip_link.get_pos()[0]
        finger_z_offset = (link6_pos[2] - tip_pos[2]).item()
        finger_y_gap = 0.0
    else:
        finger_z_offset = float(tcp_offset_flange_m[2])
        finger_y_gap = 0.0

    pad_below_fc = float(defaults["finger_pad_below_center_m"])
    close_descent = float(defaults["finger_close_descent_m"])
    table_clearance = float(defaults["grasp_table_clearance_m"])
    grasp_extra = float(defaults["grasp_height_extra_m"])

    grasp_link6_z = table_clearance + close_descent + pad_below_fc + finger_z_offset + grasp_extra
    if is_calibrated_adaptive and gripper_spec is not None:
        grasp_link6_z = dry_heights(profile.key, gripper=gripper_spec.family).grasp_link6_z
    pre_grasp_link6_z = grasp_link6_z + 0.10
    lift_link6_z = lift_z
    home_qpos_np = init_qpos[0].detach().cpu().numpy().astype(np.float64).copy()

    ctx = TrajSceneContext(
        scene=scene,
        robot=robot,
        obj=obj,
        target_marker=target_marker,
        ik_link=ik_link,
        left_finger=left_finger,
        right_finger=right_finger,
        arm_dof_idx=arm_dof_idx,
        gripper_dof_idx=gripper_dof_idx,
        all_gripper_dof_idx=all_gripper_dof_idx,
        down_quat=dq,
        home_qpos=home_qpos_np,
        base_pos_world=robot_base,
        visual_model=str(visual_model).strip().lower(),
        robot_urdf_path=robot_urdf_path,
        kinematics_yaml_path=kinematics_yaml_path,
        kinematics_suffix=kinematics_suffix,
        finger_z_offset=finger_z_offset,
        finger_y_gap=finger_y_gap,
        grasp_link6_z=grasp_link6_z,
        pre_grasp_link6_z=pre_grasp_link6_z,
        lift_link6_z=lift_link6_z,
        robot_key=profile.key,
        gripper=gripper_spec,
        home_pos_base=home_pos_base,
        table_height=TABLE_HEIGHT,
        obj_xy=obj_xy,
        place_xy=place_xy,
        obj_size=obj_size,
        obj_mass_kg=obj_mass_kg,
        obj_pos_base=obj_pos_base,
        place_pos_base=place_pos_base,
        g2_settings=g2_settings,
        robotiq_settings=robotiq_settings,
        simulation_parameters={
            "rate_hz": rate,
            "sim_rate_hz": float(rate),
            "substeps": substeps,
            "constraint_time_constant_s": constraint_timeconst,
            "constraint_solver": constraint_solver,
            "solver_iterations": solver_iterations,
            "noslip_iterations": noslip_iterations,
            "friction_cone": friction_cone,
            "contact_resolution": contact_resolution,
            "use_gjk_collision": use_gjk_collision,
        },
        sim_rate_hz=float(rate),
        arm_controller=arm_controller if arm_controller is not None else home_controller,
        gripper_dof_multipliers=multipliers,
        tool_tip_link=tool_tip_link,
        tcp_offset_flange_m=(tcp_offset_flange_m[0], tcp_offset_flange_m[1], tcp_offset_flange_m[2]),
    )
    if is_calibrated_g2:
        from ufactory.simulation.g2_adapter import G2GenesisController

        robot.set_qpos(init_qpos, zero_velocity=True)
        robot.control_dofs_position(init_qpos[:, arm_dof_idx], arm_dof_idx)
        obj.set_pos(init_qpos.new_tensor([obj_pos_world]), zero_velocity=True)
        ctx.g2_controller = G2GenesisController(robot, left_finger, right_finger, axis_link=ik_link)
    elif is_calibrated_robotiq:
        from ufactory.simulation.robotiq_adapter import RobotiqGenesisController

        robot.set_qpos(init_qpos, zero_velocity=True)
        robot.control_dofs_position(init_qpos[:, arm_dof_idx], arm_dof_idx)
        obj.set_pos(init_qpos.new_tensor([obj_pos_world]), zero_velocity=True)
        ctx.robotiq_controller = RobotiqGenesisController(
            robot,
            left_finger,
            right_finger,
            variant=gripper_spec.family,
            axis_link=ik_link,
        )
    return ctx


def drive_for_gap_m(gap_m: float, gripper: GripperControlParams | None) -> float:
    """Convert pad gap (m) using G2/Robotiq geometry or linear prismatic travel."""
    if gripper is None or not getattr(gripper, "has_actuated_joints", bool(getattr(gripper, "all_joint_names", ()))):
        return 0.0
    if gripper.family == "g2":
        from ufactory.grippers.g2 import gripper_g2_gap_m_to_sim_drive

        return gripper_g2_gap_m_to_sim_drive(gap_m)
    if gripper.family == "bio_gripper_g2":
        from ufactory.grippers.bio_g2 import bio_gripper_g2_gap_m_to_sim_drive

        return bio_gripper_g2_gap_m_to_sim_drive(gap_m)
    if gripper.family in {"robotiq_2f_85", "robotiq_2f_140"}:
        from ufactory.grippers.robotiq import robotiq_gap_m_to_sim_drive

        return robotiq_gap_m_to_sim_drive(gripper.family, gap_m)
    closed_gap = float(getattr(gripper, "closed_gap_m", 0.0))
    open_gap = float(gripper.open_gap_m)
    span = open_gap - closed_gap
    if span <= 0.0:
        return float(gripper.close_pos)
    open_fraction = max(0.0, min(1.0, (float(gap_m) - closed_gap) / span))
    return gripper.close_pos + open_fraction * (gripper.open_pos - gripper.close_pos)


def _set_contact_sol_params(entities_or_links, sol_params) -> None:
    """Apply contact solver parameters to selected entities or links."""
    sol = np.asarray(sol_params, dtype=np.float64)
    for item in entities_or_links:
        links = getattr(item, "links", None)
        if links is None:
            links = (item,)
        for link in links:
            for geom in link.geoms:
                geom.set_sol_params(sol)
