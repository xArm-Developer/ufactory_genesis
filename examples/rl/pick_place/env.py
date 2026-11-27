"""End-to-end xArm6 + Gripper G2 tabletop pick-and-place environment."""

from __future__ import annotations

import math
from typing import Any

import genesis as gs
from tensordict import TensorDict
import torch

from ufactory.dynamics.poses import parse_joint_limits
from ufactory.grippers.g2 import GRIPPER_G2_OPEN_GAP_M
from ufactory.kinematics.orientation import GRIPPER_DOWN_QUAT_XYZW
from ufactory.robots.paths import robot_urdf, robot_visual_glb_urdf
from ufactory.robots.registry import get_robot_profile, joint_names
from ufactory.simulation import (
    G2_PHYSICS_PROFILE,
    configure_g2_mimic_constraints,
    make_rigid_options,
    object_finger_contact_forces_n,
    validate_g2_contact_substeps,
)
from ufactory.simulation.g2_adapter import apply_batch_control, configure_g2_drive, measured_gap_mm, pad_centers
from ufactory.simulation.g2_controller import G2Controller
from ufactory.simulation.g2_model import G2Model
from ufactory.training.logic.common import action_penalty as _quadratic_penalty
from ufactory.training.pick_place_config import (
    HOME_EE_POS,
    PICK_PLACE_ACTION_CONTRACT,
    PICK_PLACE_CONTROL,
    PICK_PLACE_ENV_VERSION,
    PICK_PLACE_NUM_ACTIONS,
    PICK_PLACE_NUM_OBS,
    PICK_PLACE_ROBOT,
    PICK_PLACE_URDF,
    PRE_LIFT_PAD_FLOOR_Z,
    TABLE_POS,
    TABLE_SIZE,
    TARGET_PAD_POS,
    TARGET_PAD_SIZE,
    TOUCHDOWN_REF_MARGIN_Z,
)
from ufactory.visualization import start_deferred_viewer
from ufactory.visualization.glb import glb_pbr_surfaces, glb_view_surface
from ufactory.visualization.robot_viewer import setup_arm_pd

from .contacts import table_contact_loads
from .metrics import (
    CUBE_REST_Z,
    CUBE_SIZE,
    DRAG_LIMIT_M,
    PLACE_CLEARANCE_M,
    PRESS_LIMIT_M,
    RELEASE_MAX_HEIGHT_M,
    SAFE_LIFT_Z,
    SPAWN_X,
    SPAWN_XY_LOWER,
    SPAWN_XY_UPPER,
    SPAWN_Y,
    SUCCESS_HOLD_STEPS,
    SUPPORT_FORCE_N,
    TARGET_X,
    TARGET_Y,
    TILT_LIMIT_DEG,
    PickPlaceMetrics,
    cube_clearance,
    grasp_distance,
)

ENV_VERSION = PICK_PLACE_ENV_VERSION
NUM_OBS = PICK_PLACE_NUM_OBS
DT = 0.02
SUBSTEPS = 32
EPISODE_LENGTH_S = 8.0
ACTION_SCALE = 0.005
GAP_ACTION_SCALE = 0.003
SETPOINT_LEASH_M = 0.025
PRE_GRASP_GAP_M = 0.050
CLOSE_FLOOR_GAP_M = 0.012


class XArm6PickPlaceEnv:
    """Parallel GPU pick-and-place environment with calibrated Gripper G2 control."""

    def __init__(
        self,
        env_cfg: dict[str, Any],
        reward_cfg: dict[str, Any],
        robot_cfg: dict[str, Any],
        show_viewer: bool = False,
        *,
        capture: bool = False,
        high_fidelity: bool = False,
    ) -> None:
        self.num_envs = int(env_cfg["num_envs"])
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.num_obs = int(env_cfg.get("num_obs", NUM_OBS))
        self.num_actions = int(env_cfg.get("num_actions", PICK_PLACE_NUM_ACTIONS))
        if (
            env_cfg.get("env_version") != ENV_VERSION
            or self.num_obs != NUM_OBS
            or self.num_actions != PICK_PLACE_NUM_ACTIONS
            or env_cfg.get("control") != PICK_PLACE_CONTROL
        ):
            raise ValueError(
                "pick-place environment v3 requires 70 observations and 4 continuous increments; retrain the policy"
            )
        self.num_privileged_obs = None
        self.device = gs.device
        self.cfg = env_cfg
        self.env_cfg = env_cfg
        self.robot_cfg = robot_cfg
        self.reward_scales = dict(reward_cfg)
        self.high_fidelity = bool(high_fidelity or env_cfg.get("high_fidelity", False))
        self.total_env_steps = 0

        self.physics_profile = str(env_cfg.get("physics_profile", G2_PHYSICS_PROFILE))
        if self.physics_profile != G2_PHYSICS_PROFILE:
            raise ValueError(
                f"xArm6 G2 pick-place requires physics_profile={G2_PHYSICS_PROFILE!r}, got {self.physics_profile!r}"
            )

        self.ctrl_dt = float(env_cfg.get("ctrl_dt", DT))
        self.substeps = int(env_cfg.get("substeps", SUBSTEPS))
        substep_dt = validate_g2_contact_substeps(dt=self.ctrl_dt, substeps=self.substeps)
        self.max_episode_length = math.ceil(float(env_cfg.get("episode_length_s", EPISODE_LENGTH_S)) / self.ctrl_dt)

        self.action_scale = float(env_cfg.get("action_scale", ACTION_SCALE))
        self.action_clip = float(env_cfg.get("action_clip", 1.0))
        self.gap_action_scale = float(env_cfg.get("gap_action_scale", GAP_ACTION_SCALE))
        self.setpoint_leash_m = float(env_cfg.get("setpoint_leash_m", SETPOINT_LEASH_M))
        self.pre_grasp_gap_m = float(env_cfg.get("pre_grasp_gap_m", PRE_GRASP_GAP_M))
        self.close_floor_gap_m = float(env_cfg.get("close_floor_gap_m", CLOSE_FLOOR_GAP_M))
        self.workspace_lower = torch.tensor(env_cfg["workspace_lower"], device=self.device, dtype=gs.tc_float)
        self.workspace_upper = torch.tensor(env_cfg["workspace_upper"], device=self.device, dtype=gs.tc_float)
        self.spawn_xy_lower = torch.tensor(
            env_cfg.get("spawn_xy_lower", SPAWN_XY_LOWER), device=self.device, dtype=gs.tc_float
        )
        self.spawn_xy_upper = torch.tensor(
            env_cfg.get("spawn_xy_upper", SPAWN_XY_UPPER), device=self.device, dtype=gs.tc_float
        )
        if (
            self.spawn_xy_lower.shape != (2,)
            or self.spawn_xy_upper.shape != (2,)
            or (self.spawn_xy_lower > self.spawn_xy_upper).any()
            or (self.spawn_xy_lower < self.workspace_lower[:2]).any()
            or (self.spawn_xy_upper > self.workspace_upper[:2]).any()
        ):
            raise ValueError("spawn_xy bounds must be 2D, ordered, and within workspace XY bounds")
        self.reward_gamma = float(env_cfg.get("reward_gamma", 0.995))
        self.near_table_m = float(env_cfg.get("near_table_m", 0.020))
        self.max_near_table_descent_m_s = float(env_cfg.get("max_near_table_descent_m_s", 0.020))
        self.table_overload_margin_n = float(env_cfg.get("table_overload_margin_n", 1.0))
        self.object_mass_kg = float(env_cfg.get("obj_mass_kg", 0.017))

        self.g2_model = G2Model.load()
        g2_command = env_cfg.get("g2", {})
        self.g2_speed_percent: float | torch.Tensor = float(g2_command.get("speed_percent", 50.0))
        self.g2_force_percent: float | torch.Tensor = float(g2_command.get("force_percent", 50.0))
        self.g2_controller = G2Controller(
            self.g2_model,
            num_envs=self.num_envs,
            device=self.device,
            dtype=gs.tc_float,
            initial_position_mm=self.pre_grasp_gap_m * 1000.0,
        )
        self.g2_controller.command(
            self.pre_grasp_gap_m * 1000.0,
            self.g2_speed_percent,
            self.g2_force_percent,
        )

        constraint_timeconst = max(
            float(env_cfg.get("constraint_time_constant_s", 0.00125)),
            2.0 * substep_dt,
        )
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.05, -0.75, 0.70),
                camera_lookat=(0.30, 0.15, 0.12),
                camera_fov=38,
            ),
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt, substeps=self.substeps),
            rigid_options=make_rigid_options(
                gs,
                constraint_solver=str(env_cfg.get("constraint_solver", "newton")),
                friction_cone=str(env_cfg.get("friction_cone", "elliptic")),
                contact_resolution=str(env_cfg.get("contact_resolution", "signorini")),
                enable_collision=True,
                enable_joint_limit=True,
                batch_dofs_info=True,
                iterations=int(env_cfg.get("solver_iterations", 50)),
                noslip_iterations=int(env_cfg.get("noslip_iterations", 0)),
                constraint_timeconst=constraint_timeconst,
                use_gjk_collision=env_cfg.get("use_gjk_collision"),
            ),
            show_viewer=False,
        )

        self.scene.add_entity(gs.morphs.Plane(collision=False))
        table_surf = (
            gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.22, 0.24, 0.27)))
            if self.high_fidelity
            else None
        )
        self.table = self.scene.add_entity(
            gs.morphs.Box(size=TABLE_SIZE, pos=TABLE_POS, fixed=True),
            surface=table_surf,
        )
        self.target_pad = self.scene.add_entity(
            gs.morphs.Box(
                size=TARGET_PAD_SIZE,
                pos=TARGET_PAD_POS,
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=(0.15, 0.75, 0.35)),
            ),
        )

        if self.high_fidelity:
            urdf_file = robot_visual_glb_urdf(PICK_PLACE_ROBOT, with_gripper_g2=True, movable=True)
            with glb_pbr_surfaces():
                self.robot = self.scene.add_entity(
                    gs.morphs.URDF(file=urdf_file, fixed=True),
                    surface=glb_view_surface(),
                )
        else:
            urdf_file = robot_urdf(PICK_PLACE_ROBOT, PICK_PLACE_URDF)
            self.robot = self.scene.add_entity(gs.morphs.URDF(file=urdf_file, fixed=True))

        self.obj = self.scene.add_entity(
            gs.morphs.Box(size=(CUBE_SIZE,) * 3, pos=(SPAWN_X, SPAWN_Y, CUBE_REST_Z)),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=(0.90, 0.15, 0.08)),
            ),
        )

        self.camera = (
            self.scene.add_camera(
                res=(1280, 720),
                pos=(1.05, -0.75, 0.70),
                lookat=(0.30, 0.15, 0.12),
                fov=38,
            )
            if capture
            else None
        )

        self.scene.build(n_envs=self.num_envs)
        configure_g2_mimic_constraints(self.robot)
        self.obj.set_mass(self.object_mass_kg)
        self.obj.set_friction(1.0)
        self.table.set_friction(1.0)

        self.left_finger_link = self.robot.get_link("left_finger")
        self.right_finger_link = self.robot.get_link("right_finger")
        self.left_finger_link.set_friction(1.2)
        self.right_finger_link.set_friction(1.2)

        profile = get_robot_profile(PICK_PLACE_ROBOT)
        joint_map = {j.name.split("/")[-1]: j for j in self.robot.joints}
        self.arm_dof_idx = [joint_map[n].dofs_idx_local[0] for n in joint_names(profile)]
        setup_arm_pd(self.robot, self.arm_dof_idx, profile)
        lower, upper = parse_joint_limits(urdf_file, joint_names(profile))
        self.joint_lower = torch.tensor(lower, device=self.device, dtype=gs.tc_float)
        self.joint_upper = torch.tensor(upper, device=self.device, dtype=gs.tc_float)
        self.gripper_dof_idx, self.all_gripper_dof_idx = configure_g2_drive(
            self.robot,
            damping=float(robot_cfg.get("gripper_damping", 0.05)),
            frictionloss=float(robot_cfg.get("gripper_frictionloss", 0.0)),
        )

        self.ee_link = self.robot.get_link(profile.ee_link)
        x, y, z, w = GRIPPER_DOWN_QUAT_XYZW
        self.ee_quat_wxyz = torch.tensor([[w, x, y, z]], device=self.device, dtype=gs.tc_float).expand(self.num_envs, 4)
        home_ee = torch.tensor([HOME_EE_POS], device=self.device, dtype=gs.tc_float).expand(self.num_envs, 3)
        home_q = self.robot.inverse_kinematics(
            link=self.ee_link,
            pos=home_ee,
            quat=self.ee_quat_wxyz,
            dofs_idx_local=self.arm_dof_idx,
        )
        self.home_q = home_q[0, self.arm_dof_idx].detach().clone()

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_id = torch.full_like(self.episode_length_buf, -1)
        self.scenario_id = torch.full_like(self.episode_length_buf, -1)

        self.ee_setpoint = home_ee.clone()
        self.q_target = self.home_q.repeat(self.num_envs, 1)
        self.gap_cmd = torch.full((self.num_envs,), self.pre_grasp_gap_m, device=self.device, dtype=gs.tc_float)
        self.forces = torch.zeros(self.num_envs, 2, device=self.device, dtype=gs.tc_float)
        self.table_force = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        self.prev_raw_table_force = torch.zeros_like(self.table_force)
        self.robot_table_force = torch.zeros_like(self.table_force)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=gs.tc_float)
        self.metrics = PickPlaceMetrics(
            self.num_envs,
            self.device,
            ctrl_dt=self.ctrl_dt,
            target_tol=float(env_cfg.get("place_success_dist_m", 0.020)),
            drag_limit=float(env_cfg.get("drag_limit_m", DRAG_LIMIT_M)),
            tilt_limit=float(env_cfg.get("tilt_limit_deg", TILT_LIMIT_DEG)),
            hold_steps=int(env_cfg.get("success_hold_steps", SUCCESS_HOLD_STEPS)),
            place_clearance=float(env_cfg.get("place_clearance_m", PLACE_CLEARANCE_M)),
            release_max_height=float(env_cfg.get("release_max_height_m", RELEASE_MAX_HEIGHT_M)),
            press_limit=float(env_cfg.get("press_limit_m", PRESS_LIMIT_M)),
        )
        self.ep_return = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        self.contact_rewarded = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.touchdown_rewarded = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.place_rewarded = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.pre_lift_grip_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.ik_rejects = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)

        self.default_spawn_xy = torch.tensor([[SPAWN_X, SPAWN_Y]], device=self.device, dtype=gs.tc_float)
        self.target_xy = torch.tensor([[TARGET_X, TARGET_Y]], device=self.device, dtype=gs.tc_float)
        self.fixed_spawns: torch.Tensor | None = None
        self.fixed_cursor = 0
        self.extras: dict[str, Any] = {"observations": {}}
        self.reset()
        self._derive_flange_height_limits()
        if show_viewer:
            start_deferred_viewer(self.scene)

    def _derive_flange_height_limits(self) -> None:
        """Derive flange-height limits directly from the loaded Gripper G2 pad geometry."""
        left, right = pad_centers(self.g2_model, self.left_finger_link, self.right_finger_link)
        flange_to_pad_z = float((self.ee_link.get_pos()[:, 2] - (left[:, 2] + right[:, 2]) / 2).mean())
        self.pre_lift_flange_floor_z = PRE_LIFT_PAD_FLOOR_Z + flange_to_pad_z
        self.touchdown_ref_ee_z = self.pre_lift_flange_floor_z + TOUCHDOWN_REF_MARGIN_Z

    def training_state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "action_contract": PICK_PLACE_ACTION_CONTRACT,
            "total_env_steps": int(self.total_env_steps),
        }

    def load_training_state_dict(self, state: dict[str, Any]) -> None:
        if (
            not isinstance(state, dict)
            or state.get("schema_version") != 3
            or state.get("action_contract") != PICK_PLACE_ACTION_CONTRACT
        ):
            raise ValueError("checkpoint environment state is incompatible with v3; retrain the policy")
        self.total_env_steps = int(state.get("total_env_steps", 0))

    def measured_gap(self) -> torch.Tensor:
        """Measured fingertip pad separation in meters via 4-bar linkage geometry."""
        return measured_gap_mm(self.g2_model, self.left_finger_link, self.right_finger_link) / 1000.0

    def reset(self) -> tuple[TensorDict, dict[str, Any]]:
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.get_observations(), self.extras

    def set_evaluation_spawns(self, xy: Any) -> tuple[TensorDict, dict[str, Any]]:
        self.fixed_spawns = torch.as_tensor(xy, device=self.device, dtype=gs.tc_float)
        self.fixed_cursor = 0
        return self.reset()

    @torch.inference_mode(False)
    @torch.no_grad()
    def reset_idx(self, idx: torch.Tensor) -> None:
        n = len(idx)
        if n == 0:
            return
        home = self.home_q.repeat(n, 1)
        self.robot.set_dofs_position(home, self.arm_dof_idx, envs_idx=idx, zero_velocity=True)
        self.robot.control_dofs_position(home, self.arm_dof_idx, envs_idx=idx)

        drive = float(self.g2_model.linkage.drive_for_gap_m(self.pre_grasp_gap_m))
        self.robot.set_dofs_position(
            torch.full((n, len(self.all_gripper_dof_idx)), drive, device=self.device, dtype=gs.tc_float),
            self.all_gripper_dof_idx,
            envs_idx=idx,
            zero_velocity=True,
        )
        self.robot.control_dofs_position(
            torch.full((n, 1), drive, device=self.device, dtype=gs.tc_float),
            self.gripper_dof_idx,
            envs_idx=idx,
        )
        self.g2_controller.reset(idx, position_mm=self.pre_grasp_gap_m * 1000.0)
        self.q_target[idx] = home

        if self.fixed_spawns is None:
            u = torch.rand(n, 2, device=self.device, dtype=gs.tc_float)
            xy = self.spawn_xy_lower + u * (self.spawn_xy_upper - self.spawn_xy_lower)
            self.scenario_id[idx] = -1
        else:
            sample = torch.arange(self.fixed_cursor, self.fixed_cursor + n, device=self.device)
            xy = self.fixed_spawns[sample % len(self.fixed_spawns)]
            self.scenario_id[idx] = sample
            self.fixed_cursor += n

        pos = torch.cat([xy, torch.full((n, 1), CUBE_REST_Z, device=self.device, dtype=gs.tc_float)], dim=-1)
        quat = torch.zeros(n, 4, device=self.device, dtype=gs.tc_float)
        quat[:, 0] = 1.0
        self.obj.set_pos(pos, envs_idx=idx, zero_velocity=True)
        self.obj.set_quat(quat, envs_idx=idx, zero_velocity=True)
        self.metrics.reset(idx, xy, self.ee_link.get_pos()[idx, 2])

        for buf in (
            self.episode_length_buf,
            self.ep_return,
            self.contact_rewarded,
            self.touchdown_rewarded,
            self.place_rewarded,
            self.pre_lift_grip_steps,
            self.ik_rejects,
        ):
            buf[idx] = 0
        self.episode_id[idx] += 1
        self.gap_cmd[idx] = self.pre_grasp_gap_m
        self.forces[idx] = 0.0
        self.table_force[idx] = 0.0
        self.prev_raw_table_force[idx] = 0.0
        self.robot_table_force[idx] = 0.0
        self.last_actions[idx] = 0.0
        self.ee_setpoint[idx] = self.ee_link.get_pos()[idx]

    def _solve_ik(self, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        full_q = self.robot.get_qpos().detach().clone()
        full_q[:, self.arm_dof_idx] = self.q_target
        sol, error = self.robot.inverse_kinematics(
            link=self.ee_link,
            pos=target,
            quat=self.ee_quat_wxyz,
            dofs_idx_local=self.arm_dof_idx,
            init_qpos=full_q,
            damping=0.01,
            respect_joint_limit=True,
            return_error=True,
            max_samples=1,
        )
        q = sol[:, self.arm_dof_idx]
        error = error.reshape(self.num_envs, -1)
        reject = (
            (~torch.isfinite(q).all(-1))
            | (~torch.isfinite(error).all(-1))
            | ((q - self.q_target).abs().amax(-1) > 0.5)
            | (q < self.joint_lower - 1e-5).any(-1)
            | (q > self.joint_upper + 1e-5).any(-1)
            | (torch.linalg.vector_norm(error[:, :3], dim=-1) > 0.001)
            | (torch.linalg.vector_norm(error[:, 3:], dim=-1) > 0.01)
        )
        self.ik_rejects += reject.float()
        return torch.where(reject[:, None], self.q_target, q), reject

    def _apply_action(self, actions: torch.Tensor, current_measured_gap: torch.Tensor) -> None:
        """Integrate every action channel into commands; zero actions hold them."""
        ee = self.ee_link.get_pos()
        base_setpoint = self.ee_setpoint
        at_release_table = None
        near_place = None
        table_sp_z = None
        if hasattr(self, "metrics") and hasattr(self, "obj"):
            obj_pos = self.obj.get_pos()
            height = cube_clearance(obj_pos, self.obj.get_quat()).clamp(min=0)
            in_target_xy = torch.linalg.vector_norm(obj_pos[:, :2] - self.target_xy, dim=-1) <= getattr(
                self.metrics, "target_tol", 0.020
            )
            at_release_table = (
                self.metrics.ever_lifted_clear
                & in_target_xy
                & (height <= self.metrics.release_max_height)
                & (
                    self.metrics.touchdown_seen
                    | self.metrics.release_confirmed
                    | (height <= self.metrics.place_clearance)
                )
            )
            pre_lift = ~self.metrics.ever_lifted_clear
            if pre_lift.any():
                contact_steps = getattr(self.metrics, "contact_steps", torch.zeros_like(self.metrics.hold_steps))
                ever_carried = getattr(self.metrics, "ever_carried", torch.zeros_like(self.metrics.ever_lifted_clear))
                gripping_pre_lift = pre_lift & ((contact_steps >= 3) | ever_carried)
                pre_lift_floor = getattr(self, "pre_lift_flange_floor_z", 0.1833)
                pre_lift_min_sp_z = torch.where(
                    gripping_pre_lift,
                    torch.maximum(base_setpoint[:, 2], ee[:, 2] + 0.0036).clamp(min=pre_lift_floor),
                    torch.full_like(ee[:, 2], pre_lift_floor),
                )
                base_setpoint = base_setpoint.clone()
                base_setpoint[:, 2] = torch.where(
                    pre_lift,
                    torch.maximum(base_setpoint[:, 2], pre_lift_min_sp_z),
                    base_setpoint[:, 2],
                )
                actions = actions.clone()
                actions[:, 2] = torch.where(gripping_pre_lift, actions[:, 2].clamp(min=0.0), actions[:, 2])
                lift_off_xy_damp = gripping_pre_lift & (height < 0.015)
                actions[:, :2] = torch.where(lift_off_xy_damp[:, None], actions[:, :2] * 0.35, actions[:, :2])
            near_place = self.metrics.ever_lifted_clear & (height <= self.near_table_m)
            if near_place.any():
                table_ee_z = torch.maximum(self.metrics.touchdown_ee_z, ee[:, 2] - height)
                table_sp_z = table_ee_z + 0.0036
                curr_eq_sp_z = ee[:, 2] + 0.0036
                sp_z = base_setpoint[:, 2]
                sp_z = torch.where(
                    near_place,
                    torch.maximum(sp_z, torch.maximum(table_sp_z, curr_eq_sp_z - 0.0012)),
                    sp_z,
                )
                base_setpoint = base_setpoint.clone()
                base_setpoint[:, 2] = sp_z
                near_dz_limit = 0.015 * self.ctrl_dt
                dz_cmd = (actions[:, 2] * self.action_scale).clamp(min=-near_dz_limit) / self.action_scale
                dz_cmd = torch.where(
                    at_release_table,
                    dz_cmd.clamp(max=near_dz_limit / self.action_scale),
                    dz_cmd,
                )
                dz_cmd = torch.where(at_release_table & (actions[:, 2] <= 0.3), torch.zeros_like(dz_cmd), dz_cmd)
                actions = actions.clone()
                actions[:, 2] = torch.where(near_place, dz_cmd, actions[:, 2])
                hold_xy = at_release_table & (~self.metrics.release_confirmed)
                actions[:, :2] = torch.where(hold_xy[:, None], actions[:, :2] * 0.25, actions[:, :2])
        if hasattr(self, "metrics") and hasattr(self, "obj"):
            # Zero-penalty outer-margin linear spring keeps targets well inside the
            # reachable IK envelope with full-rank action Jacobian (no clamp dead zones).
            base_setpoint = base_setpoint.clone()
            dx_lo = (0.275 - base_setpoint[:, 0]).clamp(min=0.0)
            dx_hi = (base_setpoint[:, 0] - 0.515).clamp(min=0.0)
            base_setpoint[:, 0] = base_setpoint[:, 0] + 0.16 * dx_lo - 0.16 * dx_hi
            dy_lo = (-0.215 - base_setpoint[:, 1]).clamp(min=0.0)
            dy_hi = (base_setpoint[:, 1] - 0.315).clamp(min=0.0)
            base_setpoint[:, 1] = base_setpoint[:, 1] + 0.16 * dy_lo - 0.16 * dy_hi
            r_sp = torch.linalg.vector_norm(base_setpoint[:, :2], dim=-1, keepdim=True).clamp(min=1e-9)
            r_pull = (r_sp - 0.542).clamp(min=0.0) * 0.22
            base_setpoint[:, :2] = base_setpoint[:, :2] * ((r_sp - r_pull).clamp(min=0.0) / r_sp)
            z_soft_ceil = torch.where(~self.metrics.ever_lifted_clear, 0.304, 0.262)
            dz_hi = (base_setpoint[:, 2] - z_soft_ceil).clamp(min=0.0)
            base_setpoint[:, 2] = base_setpoint[:, 2] - 0.16 * dz_hi
        target = base_setpoint + actions[:, :3] * self.action_scale
        if hasattr(self, "metrics") and hasattr(self, "obj"):
            pre_lift_floor = getattr(self, "pre_lift_flange_floor_z", 0.1833)
            target[:, 0] = target[:, 0].clamp(min=0.225, max=0.550)
            target[:, 1] = target[:, 1].clamp(min=-0.252, max=0.350)
            r_xy = torch.linalg.vector_norm(target[:, :2], dim=-1, keepdim=True).clamp(min=1e-9)
            target[:, :2] = target[:, :2] * (0.568 / r_xy).clamp(max=1.0)
            target[:, 2] = torch.where(
                ~self.metrics.ever_lifted_clear,
                target[:, 2].clamp(min=pre_lift_floor, max=0.340),
                target[:, 2].clamp(max=0.305),
            )
        if near_place is not None and table_sp_z is not None:
            target[:, 2] = torch.where(near_place, torch.maximum(target[:, 2], table_sp_z), target[:, 2])
        target = torch.maximum(torch.minimum(target, self.workspace_upper), self.workspace_lower)
        delta = target - ee
        scale = (self.setpoint_leash_m / torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp(min=1e-9)).clamp(
            max=1
        )
        moving = (actions[:, :3] != 0).any(-1) | (base_setpoint != self.ee_setpoint).any(-1)
        target = torch.where(moving[:, None], ee + delta * scale, base_setpoint)
        q, reject = self._solve_ik(target)
        accepted = moving & (~reject)
        self.ee_setpoint = torch.where(accepted[:, None], target, base_setpoint)
        self.q_target = torch.where(accepted[:, None], q, self.q_target)
        self.robot.control_dofs_position(self.q_target, self.arm_dof_idx)
        base_gap = self.gap_cmd
        gap_act = actions[:, 3]
        if at_release_table is not None:
            opening_at_table = at_release_table & (actions[:, 3] > 0) & (self.gap_cmd < current_measured_gap)
            base_gap = torch.where(opening_at_table, current_measured_gap, base_gap)
            unclamping_at_table = at_release_table & (
                (base_gap >= current_measured_gap - 5e-4)
                | (self.metrics.release_confirmed & (~self.metrics.high_release))
            )
            gap_act = torch.where(unclamping_at_table, gap_act.clamp(min=0.0), gap_act)
        self.gap_cmd = (base_gap + gap_act * self.gap_action_scale).clamp(self.close_floor_gap_m, GRIPPER_G2_OPEN_GAP_M)
        self.g2_controller.command(self.gap_cmd * 1000, self.g2_speed_percent, self.g2_force_percent)
        apply_batch_control(
            self.robot,
            self.g2_controller,
            self.gripper_dof_idx,
            current_measured_gap * 1000,
            self.ctrl_dt,
        )

    def _potential(self) -> torch.Tensor:
        obj = self.obj.get_pos()
        left, right = pad_centers(self.g2_model, self.left_finger_link, self.right_finger_link)
        pad_mid = (left + right) / 2
        distance = grasp_distance(pad_mid, obj)
        pad_xy_err = torch.linalg.vector_norm((pad_mid - obj)[:, :2], dim=-1)
        dz_band = ((pad_mid[:, 2] - obj[:, 2]).abs() - CUBE_SIZE * 0.4).clamp(min=0)
        height = cube_clearance(obj, self.obj.get_quat()).clamp(min=0)
        target_error = torch.linalg.vector_norm(obj[:, :2] - self.target_xy, dim=-1)
        current_gap = self.measured_gap()
        grasp_near = torch.exp(-distance / 0.025)
        placed = self.place_rewarded
        grip_force = torch.where(
            self.metrics.ever_lifted_clear,
            (self.forces.amin(-1) / 2).clamp(0, 1),
            0.75 * (self.forces.amin(-1) / 2).clamp(0, 1) + 0.25 * (self.forces.amin(-1) / 20).clamp(0, 1),
        )
        spawn_xy = getattr(self.metrics, "spawn_xy", None)
        grip = grip_force * grasp_near * (~placed).float()
        outside_xy = (pad_xy_err - 0.015).clamp(min=0)
        z_floor = obj[:, 2] + CUBE_SIZE * 0.4 + 0.038 * (1.0 - torch.exp(-(outside_xy / 0.015).square()))
        below_floor = torch.where(
            self.metrics.ever_lifted_clear | (self.forces > 0.2).any(-1),
            0.0,
            (z_floor - pad_mid[:, 2]).clamp(min=0),
        )
        floor_gate = torch.exp(-below_floor / 0.015)
        near_cube = torch.where(
            self.metrics.ever_lifted_clear | (self.forces > 0.2).any(-1),
            1.0,
            torch.exp(-(distance / 0.012).square()),
        )
        at_table = self.metrics.ever_lifted_clear & (
            (height <= self.metrics.release_max_height) | self.metrics.touchdown_seen | self.metrics.release_confirmed
        )
        at_target_table = at_table & (target_error <= getattr(self.metrics, "target_tol", 0.020))
        gap_cmd_ref = getattr(self, "gap_cmd", current_gap)
        over_open = torch.where(
            at_target_table,
            0.0,
            0.65 * (current_gap - self.pre_grasp_gap_m).clamp(min=0)
            + 0.35 * (gap_cmd_ref - self.pre_grasp_gap_m).clamp(min=0),
        )
        early_close = (
            (self.pre_grasp_gap_m - torch.minimum(current_gap, gap_cmd_ref))
            / (self.pre_grasp_gap_m - self.close_floor_gap_m)
        ).clamp(0, 1)
        cmd_close = ((self.pre_grasp_gap_m - gap_cmd_ref) / (self.pre_grasp_gap_m - self.close_floor_gap_m)).clamp(0, 1)
        meas_close = ((self.pre_grasp_gap_m - current_gap) / (self.pre_grasp_gap_m - CUBE_SIZE)).clamp(0, 1)
        close_progress = torch.where(
            at_target_table,
            1.0,
            torch.where(self.metrics.ever_lifted_clear, meas_close, 0.65 * meas_close + 0.35 * cmd_close),
        )
        pregrasp_gap = (
            0.62 * (1.0 - (1.0 - near_cube).square() * early_close) + 0.38 * near_cube * close_progress
        ) * torch.exp(-over_open / 0.010)
        approach_kernel = (
            0.35 * torch.exp(-pad_xy_err / 0.22 - dz_band / 0.06) + 0.65 * torch.exp(-distance / 0.045)
        ) * floor_gate
        pregrasp_kernel = 1.125 * torch.exp(-distance / 0.040) * pregrasp_gap * floor_gate
        target_near = torch.exp(-target_error / 0.04)
        place_approach = 0.70 * target_near + 0.30 * torch.exp(-target_error / 0.24)
        target_zone = torch.exp(-(target_error - 0.015).clamp(min=0) / 0.04)
        lifted = self.metrics.ever_lifted_clear.float()
        place_height_err = (height - self.metrics.place_clearance).clamp(min=0)
        both_clear = (self.forces < 0.1).all(-1).float()
        released_clear = self.metrics.release_confirmed.float() * both_clear
        meas_open = ((current_gap - CUBE_SIZE) / (self.pre_grasp_gap_m - CUBE_SIZE)).clamp(0, 1)
        if hasattr(self, "ee_link"):
            ee_z = self.ee_link.get_pos()[:, 2]
            ref_ee_z = torch.where(
                self.metrics.touchdown_ee_z > 0,
                self.metrics.touchdown_ee_z,
                getattr(self, "touchdown_ref_ee_z", 0.1855),
            )
            has_td_ref = ((self.metrics.touchdown_ee_z > 0) | self.metrics.release_confirmed) & at_table
            ee_sink = torch.where(
                has_td_ref,
                (ref_ee_z - ee_z - 0.0025).clamp(min=0),
                0.0,
            )
            post_td_pose = torch.exp(-ee_sink / 0.0015)
        else:
            post_td_pose = torch.ones_like(target_error)
        table_guard = post_td_pose * torch.where(placed, released_clear, 1.0)
        place_zone = lifted * target_zone * torch.exp(-place_height_err / 0.005) * table_guard
        cmd_open = ((self.gap_cmd - self.close_floor_gap_m) / (self.pre_grasp_gap_m - self.close_floor_gap_m)).clamp(
            0, 1
        )
        action_open = (0.5 * (self.last_actions[:, 3] + 1.0)).clamp(0, 1)
        open_drive = 0.65 * cmd_open + 0.35 * action_open
        release_zone = place_zone * torch.maximum(open_drive, meas_open)
        approach_term = torch.maximum(
            approach_kernel * torch.where(at_table, table_guard, 1.0),
            release_zone,
        )
        pregrasp = torch.maximum(
            pregrasp_kernel * torch.where(at_table, table_guard, 1.0),
            1.125 * release_zone,
        )
        grasp_or_place = torch.maximum(
            grip * torch.where(at_table, post_td_pose, 1.0),
            place_zone,
        )
        hold_progress = (self.metrics.hold_steps.float() / self.metrics.required_hold_steps).clamp(0, 1)
        init_target_dist = (
            torch.linalg.vector_norm(spawn_xy - self.target_xy, dim=-1)
            if spawn_xy is not None
            else torch.full_like(target_error, 0.3)
        )
        max_lift = (self.metrics.max_held_z - CUBE_REST_Z).clamp(0, SAFE_LIFT_Z - CUBE_REST_Z)
        current_lift = torch.where((self.forces > 0.05).all(-1), height.clamp(0, SAFE_LIFT_Z - CUBE_REST_Z), 0.0)
        lift_progress = torch.where(self.metrics.ever_lifted_clear, max_lift, 0.75 * current_lift + 0.25 * max_lift)
        contact_steps = getattr(self.metrics, "contact_steps", torch.zeros_like(self.metrics.hold_steps))
        ever_carried = getattr(self.metrics, "ever_carried", torch.zeros_like(self.metrics.ever_lifted_clear))
        gripping_pre_lift = (~self.metrics.ever_lifted_clear) & ((contact_steps >= 3) | ever_carried)
        lift_action_up = (0.5 * (self.last_actions[:, 2] + 1.0)).clamp(0, 1)
        lift_drive = torch.where(self.metrics.ever_lifted_clear, 1.0, gripping_pre_lift.float() * lift_action_up)
        transit_xy_progress = (init_target_dist - target_error).clamp(min=-0.50)
        excess_height = (height - (SAFE_LIFT_Z - CUBE_REST_Z)).clamp(min=0)
        s = self.reward_scales
        return (
            float(s.get("approach_potential", 8)) * approach_term
            + float(s.get("pregrasp_potential", 4)) * pregrasp
            + float(s.get("grasp_potential", 4)) * grasp_or_place
            + float(s.get("lift_potential", 40)) * lift_progress
            + float(s.get("lift_drive_potential", 1.2)) * lift_drive
            + float(s.get("transit_potential", 8)) * lifted * (transit_xy_progress - 1.8 * excess_height)
            + float(s.get("place_potential", 9))
            * lifted
            * (
                place_approach / (1 + place_height_err / 0.08)
                + 0.45 * target_zone * torch.exp(-place_height_err / 0.045)
            )
            + float(s.get("release_potential", 4)) * place_zone * open_drive
            + float(s.get("hold_potential", 6)) * (0.5 * place_zone * meas_open + 0.5 * hold_progress)
        )

    def _lift_only_potential(self) -> torch.Tensor:
        """Lift-height potential alone; zero on the table so pre-lift timeout keeps partial lift credit only."""
        height = cube_clearance(self.obj.get_pos(), self.obj.get_quat()).clamp(min=0)
        max_lift = (self.metrics.max_held_z - CUBE_REST_Z).clamp(0, SAFE_LIFT_Z - CUBE_REST_Z)
        current_lift = torch.where((self.forces > 0.05).all(-1), height.clamp(0, SAFE_LIFT_Z - CUBE_REST_Z), 0.0)
        lift_progress = torch.where(self.metrics.ever_lifted_clear, max_lift, 0.75 * current_lift + 0.25 * max_lift)
        return float(self.reward_scales.get("lift_potential", 40)) * lift_progress

    def _action_penalty(self, raw_actions: torch.Tensor) -> torch.Tensor:
        bounded = raw_actions.clamp(-self.action_clip, self.action_clip)
        excess = (raw_actions - bounded).clamp(-5.0, 5.0)
        return float(self.reward_scales.get("action_penalty", 0.001)) * (-_quadratic_penalty(bounded)) + float(
            self.reward_scales.get("action_saturation_penalty", 0.1)
        ) * (-_quadratic_penalty(excess))

    @torch.inference_mode(False)
    @torch.no_grad()
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, Any]]:
        if actions.shape != (self.num_envs, self.num_actions) or not torch.isfinite(actions).all():
            raise ValueError("actions must be finite with shape (num_envs, 4)")
        action_penalty = self._action_penalty(actions)
        actions = actions.clamp(-self.action_clip, self.action_clip)
        old_potential = self._potential()
        self._apply_action(actions, self.measured_gap())
        self.scene.step()
        self.episode_length_buf += 1
        self.total_env_steps += self.num_envs
        self.last_actions = actions.clone()
        ee, obj = self.ee_link.get_pos(), self.obj.get_pos()
        contacts = self.obj.get_contacts(with_entity=self.robot, exclude_self_contact=True)
        fl, fr = object_finger_contact_forces_n(
            contacts,
            left_link_idx=int(self.left_finger_link.idx),
            right_link_idx=int(self.right_finger_link.idx),
        )
        self.forces = torch.stack([fl, fr], -1).to(self.device)
        raw_table_force, self.robot_table_force = table_contact_loads(
            self.table.get_contacts(),
            table_link_idx=int(self.table.links[0].idx),
            object_link_idx=int(self.obj.links[0].idx),
        )
        obj_quat = self.obj.get_quat()
        obj_vel = self.obj.get_vel()
        substep_alias = (
            (raw_table_force == 0)
            & (self.prev_raw_table_force > SUPPORT_FORCE_N)
            & (cube_clearance(obj, obj_quat) <= 0.0)
            & (obj_vel[:, 2].abs() < 0.01)
        )
        self.table_force = torch.where(substep_alias, self.prev_raw_table_force, raw_table_force)
        self.prev_raw_table_force = raw_table_force.clone()
        status = self.metrics.update(
            obj,
            obj_quat,
            self.forces,
            obj_vel,
            self.measured_gap(),
            ee_z=ee[:, 2],
            table_force=self.table_force,
            robot_table_force=self.robot_table_force,
        )
        success = status["success"]
        failed = (
            status["drag_failed"]
            | status["tilt_failed"]
            | status["dropped"]
            | status["press_failed"]
            | status["high_release"]
            | status["collision_failed"]
        )
        timeout = self.episode_length_buf >= self.max_episode_length
        done = success | failed | timeout
        s = self.reward_scales
        left, right = pad_centers(self.g2_model, self.left_finger_link, self.right_finger_link)
        pad_mid = (left + right) / 2
        in_vertical_grasp_band = (pad_mid[:, 2] - obj[:, 2]).abs() <= CUBE_SIZE * 0.4 + 1e-4
        first_contact = (
            status["stable_contact"] & (~self.contact_rewarded) & status["quality_ok"] & in_vertical_grasp_band
        )
        self.contact_rewarded |= first_contact
        first_place = (
            status["first_release"] & status["in_target_xy"] & (~self.metrics.high_release) & (~self.place_rewarded)
        )
        self.place_rewarded |= first_place
        raw_potential = self._potential()
        lift_only_potential = self._lift_only_potential()
        at_table_target_touchdown = (
            self.metrics.ever_lifted_clear
            & (self.metrics.touchdown_seen | self.metrics.release_confirmed | self.place_rewarded)
            & (self.metrics.target_error <= 0.060)
            & (~status["collision_failed"])
        )
        post_lift_timeout = (
            timeout & self.metrics.ever_lifted_clear & (~failed) & (~success) & (~at_table_target_touchdown)
        )
        post_lift_noncol_fail = (
            failed & self.metrics.ever_lifted_clear & (~status["collision_failed"]) & (~at_table_target_touchdown)
        )
        pre_lift_timeout = timeout & (~failed) & (~success) & (~self.metrics.ever_lifted_clear)
        potential = torch.where(
            done & (~success),
            torch.where(
                at_table_target_touchdown,
                torch.where(
                    self.metrics.release_confirmed | self.place_rewarded,
                    (raw_potential - 0.5).clamp(min=0.0),
                    (raw_potential - 3.5).clamp(min=0.0),
                ),
                torch.where(
                    post_lift_timeout,
                    (raw_potential - 1.5).clamp(min=0.0),
                    torch.where(
                        post_lift_noncol_fail,
                        torch.minimum(raw_potential, torch.full_like(raw_potential, 6.0)),
                        torch.where(pre_lift_timeout, lift_only_potential, 0.0),
                    ),
                ),
            ),
            raw_potential,
        )
        reward = self.reward_gamma * potential - old_potential - float(s.get("step_penalty", 0.005))
        reward += first_contact.float() * float(s.get("first_contact", 2))
        reward += status["first_carry"].float() * float(s.get("first_carry", 3))
        reward += status["first_lift_clear"].float() * float(s.get("first_lift", 6))
        first_target_touchdown = (
            status["first_touchdown"]
            & (~self.touchdown_rewarded)
            & (self.metrics.target_error <= 0.060)
            & status["quality_ok"]
            & (~status["collision_failed"])
        )
        self.touchdown_rewarded |= first_target_touchdown
        td_xy_quality = torch.exp(-(self.metrics.target_error - 0.015).clamp(min=0) / 0.025)
        reward += first_target_touchdown.float() * float(s.get("first_touchdown", 8)) * (0.25 + 0.75 * td_xy_quality)
        reward += first_place.float() * float(s.get("first_place", 10))
        at_release_table = (
            (self.metrics.clearance <= 0.0005) | self.metrics.touchdown_seen | self.metrics.release_confirmed
        )
        near_held = (
            self.metrics.ever_lifted_clear
            & (self.metrics.clearance < self.near_table_m)
            & (status["any_contact"] | at_release_table)
        )
        ee_vel = self.ee_link.get_vel()
        descent_limit = torch.where(at_release_table, 0.0, self.max_near_table_descent_m_s)
        fast_descent = (-ee_vel[:, 2] - descent_limit).clamp(min=0)
        lateral_excess = torch.where(
            at_release_table,
            (torch.linalg.vector_norm(ee_vel[:, :2], dim=-1) - 0.02).clamp(min=0),
            0.0,
        )
        reward -= float(s.get("fast_descent_penalty", 10)) * torch.where(near_held, fast_descent + lateral_excess, 0)
        capped_press = self.metrics.max_press.clamp(max=0.0028)
        prev_press = (self.metrics.max_press - status["press_increment"]).clamp(min=0.0, max=0.0028)
        capped_inc = (capped_press - prev_press).clamp(min=0.0)
        excess_press_inc = (capped_press - 0.0005).clamp(min=0.0) - (prev_press - 0.0005).clamp(min=0.0)
        reward -= float(s.get("press_penalty", 3000)) * (0.25 * capped_inc + 0.75 * excess_press_inc)
        reward -= float(s.get("recontact_penalty", 0)) * status["recontact"].float()
        overload = (self.table_force - self.object_mass_kg * 9.81 - self.table_overload_margin_n).clamp(
            min=0.0, max=40.0
        )
        reward -= float(s.get("table_overload_penalty", 0.1)) * torch.where(near_held, overload, 0.0) * self.ctrl_dt
        reward -= (
            float(s.get("high_release_penalty", 10)) * (status["first_release"] & self.metrics.high_release).float()
        )
        reward -= float(s.get("robot_table_penalty", 1)) * self.robot_table_force.clamp(max=150.0) * self.ctrl_dt
        reward -= action_penalty
        unreleased_timeout = timeout & (~success) & (~self.metrics.ever_lifted_clear)
        stalled_pre_lift_grip = (
            (~self.metrics.ever_carried) & status["stable_contact"] & (self.metrics.clearance < 0.015)
        )
        self.pre_lift_grip_steps = torch.where(stalled_pre_lift_grip, self.pre_lift_grip_steps + 1, 0)
        stall_grace_steps = int(self.env_cfg.get("pre_lift_stall_grace_steps", 30))
        reward -= float(s.get("pre_lift_stall_penalty", 0.04)) * (self.pre_lift_grip_steps > stall_grace_steps).float()
        pre_lift_clawback = (
            (failed | unreleased_timeout) & (~self.metrics.ever_carried) & self.contact_rewarded
        ).float() * float(s.get("first_contact", 2))
        reward -= pre_lift_clawback
        fail_pen = torch.where(
            self.metrics.ever_lifted_clear,
            torch.where(
                status["collision_failed"],
                float(s.get("failure_penalty", 10)),
                float(s.get("post_lift_failure_penalty", 6.0)),
            ),
            float(s.get("pre_lift_failure_penalty", 4.0)),
        )
        penalized_fail = failed | unreleased_timeout
        reward += float(s.get("success", 40)) * success.float() - fail_pen * penalized_fail.float()
        self.ep_return += reward
        terminal = {
            "success": success.clone(),
            "drag_failed": status["drag_failed"].clone(),
            "tilt_failed": status["tilt_failed"].clone(),
            "dropped": status["dropped"].clone(),
            "press_failed": status["press_failed"].clone(),
            "high_release": self.metrics.high_release.clone(),
            "collision_failed": status["collision_failed"].clone(),
            "timeout": (timeout & ~success & ~failed).clone(),
            "max_drag_mm": self.metrics.max_drag.clone() * 1000,
            "max_tilt_deg": self.metrics.max_tilt_deg.clone(),
            "target_error_mm": self.metrics.target_error.clone() * 1000,
            "release_height_mm": self.metrics.release_height.clone() * 1000,
            "release_confirmed": self.metrics.release_confirmed.clone(),
            "ever_carried": self.metrics.ever_carried.clone(),
            "ever_lifted_clear": self.metrics.ever_lifted_clear.clone(),
            "max_held_clearance_mm": (self.metrics.max_held_z.clone() - CUBE_REST_Z) * 1000,
            "touchdown_seen": self.metrics.touchdown_seen.clone(),
            "touchdown_ee_z_mm": self.metrics.touchdown_ee_z.clone() * 1000,
            "min_ee_z_before_release_mm": self.metrics.min_ee_z.clone() * 1000,
            "press_mm": self.metrics.max_press.clone() * 1000,
            "max_table_force_n": self.metrics.max_table_force.clone(),
            "max_place_table_force_n": self.metrics.max_place_table_force.clone(),
            "max_held_table_force_n": self.metrics.max_held_table_force.clone(),
            "max_robot_table_force_n": self.metrics.max_robot_table_force.clone(),
            "clearance_mm": self.metrics.clearance.clone() * 1000,
            "obj_x": obj[:, 0].clone(),
            "obj_y": obj[:, 1].clone(),
            "obj_z": obj[:, 2].clone(),
            "hold_steps": self.metrics.hold_steps.clone(),
            "return": self.ep_return.clone(),
            "steps": self.episode_length_buf.clone(),
            "spawn_xy": self.metrics.spawn_xy.clone(),
            "episode_id": self.episode_id.clone(),
            "scenario_id": self.scenario_id.clone(),
            "ik_rejects": self.ik_rejects.clone(),
        }
        # Remaining time is explicitly observed (dim 69), so finite-horizon timeouts
        # are true terminal states and must not be bootstrapped by RSL-RL.
        info: dict[str, Any] = {"episode": {}, "terminal": terminal, "time_outs": torch.zeros_like(timeout)}
        if done.any():
            idx = done.nonzero(as_tuple=False).squeeze(-1)
            info["episode"] = {
                k: v[idx].float().mean().item()
                for k, v in terminal.items()
                if v.ndim == 1 and k not in {"episode_id", "scenario_id"}
            }
            info["episode"]["success_rate"] = info["episode"].pop("success")
            self.reset_idx(idx)
        self.extras = info
        return self.get_observations(), reward, done, info

    def get_observations(self) -> TensorDict:
        """70 policy inputs: measured state, command error, contacts and history."""
        ee, obj = self.ee_link.get_pos(), self.obj.get_pos()
        left, right = pad_centers(self.g2_model, self.left_finger_link, self.right_finger_link)
        target = obj.new_tensor((TARGET_X, TARGET_Y, CUBE_REST_Z)).expand(self.num_envs, 3)
        m = self.metrics
        at_release = m.touchdown_seen | m.release_confirmed
        ee_obs = torch.cat(
            [
                torch.where(m.ever_lifted_clear[:, None], (ee[:, :2] - self.target_xy) / 0.3, 0.0),
                ee[:, 2:3],
            ],
            dim=-1,
        )
        obj_obs = torch.cat([obj[:, :2] - ee[:, :2], obj[:, 2:3]], dim=-1)
        target_obs = torch.where(m.ever_lifted_clear[:, None], (target - obj) / 0.3, 0.0)
        target_obs[:, 2] = torch.where(at_release, 0.0, target_obs[:, 2])
        obs = torch.cat(
            [
                self.robot.get_dofs_position(self.arm_dof_idx) / math.pi,  # 6
                self.robot.get_dofs_velocity(self.arm_dof_idx) / 2,  # 6
                ee_obs,
                self.ee_link.get_vel() / 0.2,  # 6
                obj_obs,
                self.obj.get_quat(),
                self.obj.get_vel() / 0.2,
                self.obj.get_ang() / 2,  # 13
                target_obs,
                (left - obj) / 0.1,
                (right - obj) / 0.1,  # 9
                self.gap_cmd[:, None] / GRIPPER_G2_OPEN_GAP_M,
                self.measured_gap()[:, None] / GRIPPER_G2_OPEN_GAP_M,  # 2
                (self.ee_setpoint - ee) / self.setpoint_leash_m,  # 3
                (self.forces / 20).clamp(0, 1),
                m.finger_contact.float(),  # 4
                torch.where(m.ever_lifted_clear, (self.table_force / 20).clamp(0, 1), 0.0)[:, None],
                (self.robot_table_force / 20).clamp(0, 1)[:, None],  # 2
                self.last_actions,  # 4
                (m.max_drag / DRAG_LIMIT_M)[:, None],
                (m.max_tilt_deg / TILT_LIMIT_DEG)[:, None],
                (m.hold_steps.float() / m.required_hold_steps)[:, None],  # 3
                m.ever_carried[:, None].float(),
                m.ever_lifted_clear[:, None].float(),  # 2
                (m.max_press / PRESS_LIMIT_M)[:, None],
                m.touchdown_seen[:, None].float(),
                torch.where(m.touchdown_seen, (m.touchdown_ee_z - ee[:, 2]) / 0.01, 0)[:, None],  # 3
                (m.release_height / RELEASE_MAX_HEIGHT_M)[:, None],
                m.release_confirmed[:, None].float(),
                m.high_release[:, None].float(),  # 3
                self.contact_rewarded[:, None].float(),
                self.place_rewarded[:, None].float(),  # 2
                (1 - self.episode_length_buf.float() / self.max_episode_length)[:, None],  # 1
                ((m.max_held_z - CUBE_REST_Z).clamp(0, SAFE_LIFT_Z - CUBE_REST_Z) / (SAFE_LIFT_Z - CUBE_REST_Z))[
                    :, None
                ],  # 1
            ],
            dim=-1,
        )
        if obs.shape != (self.num_envs, self.num_obs):
            raise RuntimeError(f"pick-place observation contract mismatch: {tuple(obs.shape)}")
        return TensorDict({"policy": obs}, batch_size=[self.num_envs])
