"""Scene construction for the trajectory pick-place pipeline.

Robot-generic Genesis scene builder for the pick-place trajectory pipeline
(originally written for xArm6 scripted pick-place demos). Returns a typed
context (entities, dof indices, PD gains, finger offsets, key heights) reused
across xArm5/6/7, UF850 (Gripper G2) and Lite6 (parallel gripper). Single
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
    make_rigid_options,
    validate_g2_contact_substeps,
)
from ufactory.robots.paths import robot_visual_glb_urdf, robot_urdf
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

# Finger geometry constants measured/URDF-derived (same as the demo). These
# describe the Gripper G2 pad geometry; Lite6 uses its own LITE6_* constants.
FINGER_PAD_BELOW_FC = 0.061
FINGER_CLOSE_DESCENT = 0.015
GRASP_TABLE_CLEARANCE = 0.010

# Lite6 parallel-jaw fingers: pad extends ~27 mm below the joint/finger-center
# Lite6 reversed parallel-jaw finger geometry (URDF/mesh-derived). Each finger
# is an L: a mounting boss near link6 that used to over-reach the cube, plus a
# long fingertip plate (the large flat inner pad) that is the intended gripping
# surface. The finger URDFs place both fingers on the raw mesh (no synthetic
# outward offset), so the fingertip plate closes onto the 30 mm cube while the
# boss stays clear. ``PAD_BELOW_FC`` is the fc->fingertip distance; the grasp
# clearance keeps the fingertip plate spanning the cube's upper body with the
# boss a few mm above the cube top.
# fc->lowest inner-pad Z at grasp (collision STL, gripper-down); tuned so the flat
# pad spans the 30 mm cube mid-height after typical MoveL end-side error on descend.
LITE6_FINGER_PAD_BELOW_FC = 0.021
LITE6_FINGER_CLOSE_DESCENT = 0.0
# Slightly below the G2 default so the fingertip plate (not the mounting boss)
# reaches the cube sides once descend tracking error is accounted for.
LITE6_GRASP_TABLE_CLEARANCE = 0.006
# Lift the Lite6 grasp slightly so the low mounting boss clears the cube top
# and the large flat inner pad, not the stop/boss region, carries side contact.
LITE6_GRASP_LINK6_Z_EXTRA_M = 0.015
# Lite6 opens at the same grasp height after a short closed-gripper settle in
# the trajectory wrapper; keep this deprecated standoff at zero so parallel-jaw
# release does not lift the cube before opening.
LITE6_PLACE_RELEASE_STANDOFF_M = 0.0

# Empirical link6->finger-center z offset (gripper open, "down" orientation),
# measured once per gripper family by build_scene and stable across xArm5/6/7/
# UF850 (all Gripper G2) since the arm's ee_link and the G2 mount are identical
# geometry. Used by real-path dry-run height estimates (no sim scene needed).
FINGER_Z_OFFSET_G2 = 0.1011
FINGER_Z_OFFSET_LITE6 = 0.0543

RIGID_SOLVER_ITERATIONS = DEFAULT_SOLVER_ITERATIONS
RIGID_NOSLIP_ITERATIONS = DEFAULT_NOSLIP_ITERATIONS
RIGID_CONSTRAINT_TIMECONST = DEFAULT_CONSTRAINT_TIMECONST
# Painted wood block and silicone fingertip pads. Size and mass come from the
# shared pick-place runtime configuration; contact stiffness remains code policy.
OBJ_FRICTION = 1.0
FINGER_FRICTION = 1.2
LITE6_OBJ_FRICTION = OBJ_FRICTION
LITE6_FINGER_FRICTION = FINGER_FRICTION


@dataclass
class TrajSceneContext:
    scene: object
    robot: object
    obj: object
    target_marker: object
    ik_link: object
    left_finger: object
    right_finger: object
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
        return robot_urdf(profile.key, "xarm6_with_gripper.urdf"), False
    if model != "glb":
        raise ValueError("visual_model must be 'glb' or 'stl'")
    if gripper is not None and gripper.family == "lite6":
        return robot_visual_glb_urdf(profile.key, with_lite6_gripper=True, movable=True), True
    if gripper is not None and gripper.family == "g2":
        return robot_visual_glb_urdf(profile.key, with_gripper_g2=True, movable=True), True
    raise ValueError(f"{profile.key} has no supported gripper for the trajectory scene")


def _robot_defaults(robot_key: str) -> dict:
    """Per-robot default scene geometry (object/place xy, key heights)."""
    runtime = get_robot_runtime_profile(robot_key)
    config = load_runtime_config(runtime.model.key)
    params = config.task.parameters
    object_spec = resolve_pick_place_object_spec(config)
    gripper_profile = config.gripper
    if gripper_profile is None:
        raise ValueError(f"{runtime.model.key} has no configured gripper geometry")
    obj_pos_base = tuple(float(value) for value in params["fixed_object_position_m"])
    place_pos_base = tuple(float(value) for value in params["fixed_target_position_m"])
    default_ee = tuple(float(value) for value in params["default_ee_position_m"])
    is_lite6 = runtime.model.key == "lite6"
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
    }


def default_grasp_gap_m(robot_key: str) -> float:
    """Tuned default two-finger grasp gap (m) for ``robot_key``'s default object."""
    return _robot_defaults(get_robot_runtime_profile(robot_key).model.key)["grasp_gap_m"]


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
    gripper: GripperControlParams


def dry_heights(robot_key: str) -> DryHeights:
    """Build a :class:`DryHeights` for ``robot_key`` without a Genesis scene."""
    runtime = get_robot_runtime_profile(robot_key)
    gripper = runtime.gripper
    if gripper is None:
        raise ValueError(f"{runtime.model.key} has no supported gripper for the trajectory scene")
    defaults = _robot_defaults(runtime.model.key)
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
        gripper=gripper,
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
    constraint_timeconst: float = RIGID_CONSTRAINT_TIMECONST,
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
) -> TrajSceneContext:
    """Build the single-env Genesis scene and configure PD gains + finger offsets.

    ``robot_key`` selects the arm profile (``xarm5``/``xarm6``/``xarm7``/``lite6``/``uf850``);
    the gripper (Gripper G2 or Lite6 parallel gripper) is resolved from that
    robot's runtime profile. Full base-frame object positions are preferred.
    Legacy ``obj_xy``/``place_xy`` inputs remain supported and use half the
    configured object height as their table-resting z coordinate.
    """
    runtime = get_robot_runtime_profile(robot_key)
    profile = runtime.model
    gripper = runtime.gripper
    if gripper is None:
        raise ValueError(f"{profile.key} has no supported gripper for the trajectory scene")
    defaults = _robot_defaults(profile.key)

    dt = 1.0 / float(rate)
    if gripper.family == "g2":
        validate_g2_contact_substeps(dt=dt, substeps=substeps)
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
    substep_dt = dt / int(substeps)
    constraint_timeconst = max(float(constraint_timeconst), 2.0 * substep_dt)
    camera_lookat = _base_to_world(robot_base, (0.30, 0.15, 0.10))
    camera_pos = (camera_lookat[0] + 0.90, camera_lookat[1] - 1.35, camera_lookat[2] + 0.60)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=substeps),
        rigid_options=make_rigid_options(
            gs,
            dt=dt,
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

    scene.build(n_envs=1)

    ee_link_name = profile.ee_link
    ik_link = robot.get_link(ee_link_name)
    left_finger = robot.get_link(gripper.finger_link_names[0])
    right_finger = robot.get_link(gripper.finger_link_names[1])

    if gripper.family == "g2":
        configure_g2_mimic_constraints(robot)
    if contact_sol_params is not None:
        _set_contact_sol_params((obj, left_finger, right_finger), contact_sol_params)

    # Configured painted-wood cube + silicone fingertip pads (μ=1.2).
    # Stiffness: leave default rigid contact sol_params. Set before any stepping.
    obj.set_friction(float(OBJ_FRICTION))
    left_finger.set_friction(float(FINGER_FRICTION))
    right_finger.set_friction(float(FINGER_FRICTION))
    obj.set_links_mass(
        torch.tensor([obj_mass_kg], device=gs.device, dtype=gs.tc_float),
    )

    jnames = joint_names(profile)
    arm_dof_idx = [robot.get_joint(n).dofs_idx_local[0] for n in jnames]
    gripper_dof_idx = [robot.get_joint(gripper.drive_joint).dofs_idx_local[0]]
    all_gripper_dof_idx = [robot.get_joint(n).dofs_idx_local[0] for n in gripper.all_joint_names]

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
    robot.set_dofs_kp(
        torch.tensor([gripper.kp], device=gs.device, dtype=gs.tc_float),
        gripper_dof_idx,
    )
    robot.set_dofs_kv(
        torch.tensor([gripper.kv], device=gs.device, dtype=gs.tc_float),
        gripper_dof_idx,
    )
    robot.set_dofs_force_range(
        torch.tensor([gripper.force_lower], device=gs.device, dtype=gs.tc_float),
        torch.tensor([gripper.force_upper], device=gs.device, dtype=gs.tc_float),
        gripper_dof_idx,
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

    dq = xyz_to_quat(
        torch.tensor([GRIPPER_DOWN_RPY_RAD], device=gs.device, dtype=gs.tc_float),
        rpy=True,
        degrees=False,
    )
    home_pos_base = [obj_xy[0], obj_xy[1], home_z]
    home_pos = _base_to_world(robot_base, home_pos_base)
    home_link6_pos = torch.tensor([home_pos], device=gs.device, dtype=gs.tc_float)
    home_qpos_result = robot.inverse_kinematics(
        link=ik_link,
        pos=home_link6_pos,
        quat=dq,
        dofs_idx_local=arm_dof_idx,
    )
    init_qpos = torch.zeros(1, robot.n_dofs, device=gs.device, dtype=gs.tc_float)
    for i, idx in enumerate(arm_dof_idx):
        init_qpos[:, idx] = home_qpos_result[0, arm_dof_idx[i]]
    for idx in all_gripper_dof_idx:
        init_qpos[:, idx] = gripper.open_pos
    robot.set_qpos(init_qpos)
    scene.step()

    link6_pos = ik_link.get_pos()[0]
    lf_pos = left_finger.get_pos()[0]
    rf_pos = right_finger.get_pos()[0]
    fc_pos = (lf_pos + rf_pos) / 2
    finger_z_offset = (link6_pos[2] - fc_pos[2]).item()
    # Finger-span across the open gap: fingers separate mainly along y when
    # the gripper points down, so use the larger of |dy|/|dx| as the span.
    finger_y_gap = max(abs(lf_pos[1] - rf_pos[1]).item(), abs(lf_pos[0] - rf_pos[0]).item())

    pad_below_fc = float(defaults["finger_pad_below_center_m"])
    close_descent = float(defaults["finger_close_descent_m"])
    table_clearance = float(defaults["grasp_table_clearance_m"])
    grasp_extra = float(defaults["grasp_height_extra_m"])

    grasp_link6_z = table_clearance + close_descent + pad_below_fc + finger_z_offset + grasp_extra
    pre_grasp_link6_z = grasp_link6_z + 0.10
    lift_link6_z = lift_z
    home_qpos_np = init_qpos[0].detach().cpu().numpy().astype(np.float64).copy()

    return TrajSceneContext(
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
        gripper=gripper,
        home_pos_base=home_pos_base,
        table_height=TABLE_HEIGHT,
        obj_xy=obj_xy,
        place_xy=place_xy,
        obj_size=obj_size,
        obj_mass_kg=obj_mass_kg,
        obj_pos_base=obj_pos_base,
        place_pos_base=place_pos_base,
    )


def drive_for_gap_m(gap_m: float, gripper: GripperControlParams) -> float:
    """Convert a physical two-finger gap (m) to the Genesis drive-DOF value.

    Interpolates linearly between ``gripper.close_pos``
    (gap=``closed_gap_m``) and ``gripper.open_pos``
    (gap=``open_gap_m``); works for both Gripper G2 (open=0.0 < close=0.85)
    and Lite6 (close=0.0 < open=0.0089) sign conventions.
    """
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
