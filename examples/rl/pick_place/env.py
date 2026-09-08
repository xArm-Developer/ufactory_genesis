"""xArm6 + Gripper G2 pick-place RL environment.

Observation (30-dim legacy prefix, up to 47-dim controller-aware policy):
    q(6) + qd(6) + finger_center_base(3) + gripper_gap_m(1)
    + obj_pos_base(3) + target_pos_base(3)
    + ee_to_obj(3) + obj_to_target(3) + grasped(1) + ever_grasped(1)
    + commanded_gripper_gap(1) + previous_action(4)
    + normalized_object_velocity(3) + holding(1)
    + ever_carried_near(1) + release_started(1)
    + bilateral_contact_state(3) + normalized_setpoint_residual_xyz(3)

Action (4-dim):
    normalized base-frame Δxyz for the EE (link6 IK, fixed gripper-down RPY)
    + normalized Gripper G2 gap delta.

Home pose (locked): default_ee_position_m=[0.30, 0.00, 0.30] with RPY=(180°,0°,0°).

The environment keeps object and target observations in robot base frame even
though Genesis entities live in world frame. The robot base is mounted at the
table surface height, so the current conversion is translation-only.
"""

from __future__ import annotations

import csv
import math
import os

import numpy as np
import torch
from tensordict import TensorDict

import genesis as gs
from ufactory.kinematics.orientation import GRIPPER_DOWN_QUAT_XYZW
from ufactory.manipulation.frames import base_to_world_pos, world_to_base_pos
from ufactory.grippers.g2 import (
    GRIPPER_G2_OPEN_GAP_M,
)
from ufactory.trajectory.scene import (
    FINGER_FRICTION,
    OBJ_FRICTION,
    RIGID_CONSTRAINT_TIMECONST,
    RIGID_NOSLIP_ITERATIONS,
    RIGID_SOLVER_ITERATIONS,
)
from ufactory.simulation import (
    G2_PHYSICS_PROFILE,
    G2ContactHoldController,
    configure_g2_mimic_constraints,
    make_rigid_options,
    object_finger_contact_forces_n,
    validate_g2_contact_substeps,
)
from ufactory.visualization.glb import enable_glb_pbr_surfaces, glb_pbr_surfaces, glb_view_surface
from .trace_utils import action_noise_episode_mask, scheduled_action_noise_std
from ufactory.training.logic import (
    arm_target_ee_pos,
    annealed_frac,
    desired_grasp_pos_base as logic_desired_grasp_pos_base,
    drive_to_gap_m,
    gap_m_to_drive,
    gripper_target_gap,
    leashed_ee_setpoint,
    next_curriculum_stage,
    normalized_ee_setpoint_residual,
    normalized_pick_place_contact_features,
    normalized_pick_place_layout_offsets,
    normalized_pick_place_quality_features,
    pick_place_observation,
    reward_action_penalty,
    reward_align,
    reward_approach_potential,
    reward_clean_lift_bonus,
    reward_clean_lift_quality,
    reward_grasp_centering,
    reward_close_gripper,
    reward_descent_progress,
    reward_drop_far,
    reward_grasp,
    reward_grasp_bonus,
    reward_grasp_gap_progress,
    reward_grasp_lift_action,
    reward_grasp_lift_progress,
    reward_grasp_ready_closing,
    reward_grasp_settle_action,
    reward_hard_landing,
    reward_holding_table,
    reward_invalid_release,
    reward_keypoints,
    reward_landing_quality,
    reward_lift,
    reward_lower,
    reward_near_table_down_speed_margin,
    reward_near_table_xy_action,
    reward_near_table_xy_speed_margin,
    reward_near_target_speed,
    reward_place_xy,
    reward_place_z,
    reward_post_release_clearance_progress,
    reward_post_release_contact,
    reward_post_release_recontact,
    reward_premature_opening,
    reward_precision_progress,
    reward_pre_lift_xy_progress,
    reward_push_after_release,
    reward_push_before_grasp,
    reward_release_clearance_opening,
    reward_ready_opening,
    reward_reach,
    reward_release,
    reward_release_quality,
    reward_release_readiness_progress,
    reward_setdown_action,
    reward_success,
    reward_table_collision,
    reward_throw_release,
    reward_transport_progress,
    reward_valid_release,
    reward_workspace_violation,
    sample_object_and_target,
    sample_reset_phase_masks,
    update_task_state,
)

# Reward keys NOT multiplied by ctrl_dt: sparse events (success, grasp_bonus) and the
# telescoping approach potential (a per-step distance delta that must sum to a fixed
# total regardless of control frequency, not a per-second hover rate).
_UNSCALED_REWARD_KEYS = frozenset(
    {
        "success",
        "grasp_bonus",
        "approach_potential",
        "transport_progress",
        "descent_progress",
        "pre_lift_xy_progress",
        "clean_lift_bonus",
        "clean_lift_quality",
        "grasp_centering",
        "grasp_ready_closing",
        "grasp_settle_action",
        "grasp_gap_progress",
        "grasp_lift_action",
        "grasp_lift_progress",
        "valid_release",
        "invalid_release",
        "release_quality",
        "release_readiness_progress",
        "premature_opening",
        "release_clearance_opening",
        "setdown_action",
        "landing_quality",
        "hard_landing",
        "precision_progress",
        "post_release_clearance_progress",
        "post_release_recontact",
    }
)


class XArm6PickPlaceEnv:
    def __init__(
        self,
        env_cfg: dict,
        reward_cfg: dict,
        robot_cfg: dict,
        show_viewer: bool = False,
    ) -> None:
        self.num_envs = int(env_cfg["num_envs"])
        self.num_obs = int(env_cfg["num_obs"])
        self.include_commanded_gap = bool(env_cfg.get("include_commanded_gap", False))
        self.include_previous_action = bool(env_cfg.get("include_previous_action", False))
        self.include_normalized_layout_offsets = bool(env_cfg.get("include_normalized_layout_offsets", False))
        self.include_scripted_action_hint = bool(env_cfg.get("include_scripted_action_hint", False))
        self.include_quality_observations = bool(env_cfg.get("include_quality_observations", False))
        self.include_ee_setpoint_residual = bool(env_cfg.get("include_ee_setpoint_residual", False))
        self.quality_pre_release_shaping_only = bool(env_cfg.get("quality_pre_release_shaping_only", False))
        self.privileged_critic_obs = bool(env_cfg.get("privileged_critic_obs", False))
        self.num_privileged_obs = 6 if self.privileged_critic_obs else None
        self.num_actions = int(env_cfg["num_actions"])
        self.device = gs.device
        enable_glb_pbr_surfaces()

        self.ctrl_dt = float(env_cfg["ctrl_dt"])
        self.max_episode_length = math.ceil(float(env_cfg["episode_length_s"]) / self.ctrl_dt)

        self.env_cfg = env_cfg
        self.physics_profile = str(env_cfg.get("physics_profile", G2_PHYSICS_PROFILE))
        if self.physics_profile != G2_PHYSICS_PROFILE:
            raise ValueError(
                f"xArm6 G2 pick-place requires physics_profile={G2_PHYSICS_PROFILE!r}, got {self.physics_profile!r}"
            )
        # rsl-rl >=5 Logger stores env.cfg alongside the train config.
        self.cfg = env_cfg
        self.reward_scales = reward_cfg.copy()
        self.action_scale = float(env_cfg["action_scale"])
        self.action_clip = float(env_cfg.get("action_clip", 1.0))
        self.train_action_noise_std = float(env_cfg.get("train_action_noise_std", 0.0))
        if self.train_action_noise_std < 0.0:
            raise ValueError("train_action_noise_std must be non-negative")
        # Noise curriculum: anneal the execution-noise std from the start value to
        # `train_action_noise_std_end` over `noise_anneal_steps` batch steps.  When
        # unset the std stays fixed at `train_action_noise_std` (round-10 behavior).
        self.train_action_noise_std_end = float(env_cfg.get("train_action_noise_std_end", self.train_action_noise_std))
        if self.train_action_noise_std_end < 0.0:
            raise ValueError("train_action_noise_std_end must be non-negative")
        self.noise_anneal_steps = int(env_cfg.get("noise_anneal_steps", 0))
        if self.noise_anneal_steps < 0:
            raise ValueError("noise_anneal_steps must be non-negative")
        # When true, each step samples its noise magnitude uniformly from [0, current];
        # when false every step uses the current (possibly annealed) std exactly.
        self.noise_std_uniform_sample = bool(env_cfg.get("noise_std_uniform_sample", False))
        self.train_action_noise_clean_episode_frac = float(env_cfg.get("train_action_noise_clean_episode_frac", 0.0))
        if not 0.0 <= self.train_action_noise_clean_episode_frac <= 1.0:
            raise ValueError("train_action_noise_clean_episode_frac must be in [0, 1]")
        self.strict_action_bounds = bool(env_cfg.get("strict_action_bounds", False))
        self.max_cartesian_delta_m = float(env_cfg["max_cartesian_delta_m"])
        self.max_ik_jump_rad = float(env_cfg.get("max_ik_jump_rad", 0.5))
        self.gripper_delta_m = float(env_cfg["gripper_delta_mm"]) / 1000.0
        self._ik_damping = 0.01
        self.action_response_exponent = float(env_cfg.get("action_response_exponent", 1.0))
        if self.action_response_exponent <= 0.0:
            raise ValueError("action_response_exponent must be positive")
        # "measured" adds each delta to where the arm actually is, so the standing PD
        # droop is subtracted from every command and a zero action sinks the wrist.
        # "commanded" integrates a setpoint instead, making a zero action hold still.
        self.ee_command_integration = str(env_cfg.get("ee_command_integration", "measured"))
        if self.ee_command_integration not in ("measured", "commanded"):
            raise ValueError("ee_command_integration must be 'measured' or 'commanded'")
        self.ee_setpoint_leash_m = float(env_cfg.get("ee_setpoint_leash_m", 0.02))
        if self.ee_setpoint_leash_m <= 0.0:
            raise ValueError("ee_setpoint_leash_m must be positive")

        self.table_height = float(env_cfg["table_height"])
        self.obj_size = tuple(float(v) for v in env_cfg["obj_size"])
        self.obj_mass_kg = float(env_cfg["obj_mass_kg"])
        if not math.isfinite(self.obj_mass_kg) or self.obj_mass_kg <= 0.0:
            raise ValueError("obj_mass_kg must be finite and positive")
        self.obj_size_t = torch.tensor(self.obj_size, device=self.device, dtype=gs.tc_float)
        self.obj_rest_z_base = self.obj_size[2] / 2.0
        self.grasp_center_offset_z = float(env_cfg.get("grasp_center_offset_z", 0.065))
        # link6 -> finger-center z offset (gripper-down). Bootstrap resets solve IK for
        # link6, so a finger-center grasp target must be raised by this to place link6.
        self.tool_tip_offset_z_m = float(env_cfg.get("tool_tip_offset_z_m", 0.1011))
        self.lift_height_m = float(env_cfg.get("lift_height_m", 0.08))
        self.place_success_dist_m = float(env_cfg.get("place_success_dist_m", 0.04))
        self.carry_near_dist_m = float(env_cfg.get("carry_near_dist_m", 0.04))
        self.place_shaping_dist_m = float(env_cfg.get("place_shaping_dist_m", self.place_success_dist_m))
        self.release_success_dist_m = float(env_cfg.get("release_success_dist_m", self.place_success_dist_m))
        if self.carry_near_dist_m <= 0.0:
            raise ValueError("carry_near_dist_m must be positive")
        if (
            min(
                self.place_success_dist_m,
                self.place_shaping_dist_m,
                self.release_success_dist_m,
            )
            <= 0.0
        ):
            raise ValueError("place, shaping, and release distances must be positive")
        self.place_lower_near_factor = float(env_cfg.get("place_lower_near_factor", 1.5))
        if self.place_lower_near_factor <= 0.0:
            raise ValueError("place_lower_near_factor must be positive")
        self.success_hold_steps = int(env_cfg.get("success_hold_steps", 10))
        self.success_table_z_tolerance_m = float(env_cfg.get("success_table_z_tolerance_m", 0.025))
        self.success_max_obj_speed_m_s = float(env_cfg.get("success_max_obj_speed_m_s", 0.15))
        self.release_height_tolerance_m = float(env_cfg.get("release_height_tolerance_m", 0.005))
        self.release_max_obj_speed_m_s = float(env_cfg.get("release_max_obj_speed_m_s", 0.02))
        self.pre_lift_max_drag_m = float(env_cfg.get("pre_lift_max_drag_m", 0.005))
        self.post_release_max_drift_m = float(env_cfg.get("post_release_max_drift_m", 0.003))
        self.post_release_clearance_max_m = float(env_cfg.get("post_release_clearance_max_m", 0.050))
        self.post_release_clearance_gap_margin_m = float(env_cfg.get("post_release_clearance_gap_margin_m", 0.008))
        self.landing_near_table_height_m = float(env_cfg.get("landing_near_table_height_m", 0.02))
        self.landing_speed_margin_height_m = float(
            env_cfg.get(
                "landing_speed_margin_height_m",
                self.landing_near_table_height_m,
            )
        )
        self.landing_max_xy_speed_m_s = float(env_cfg.get("landing_max_xy_speed_m_s", 0.03))
        self.landing_max_down_speed_m_s = float(env_cfg.get("landing_max_down_speed_m_s", 0.05))
        # Fraction of the landing speed limit that is free (no margin cost) so the
        # policy is pushed to keep a real safety margin under the acceptance limit.
        self.landing_xy_speed_safety_zone_frac = float(env_cfg.get("landing_xy_speed_safety_zone_frac", 0.0))
        self.landing_down_speed_safety_zone_frac = float(env_cfg.get("landing_down_speed_safety_zone_frac", 0.0))
        self.near_table_xy_action_height_m = float(env_cfg.get("near_table_xy_action_height_m", 0.025))
        self.release_action_threshold = float(env_cfg.get("release_action_threshold", 0.05))
        self.release_command_margin_m = float(env_cfg.get("release_command_margin_m", 0.0005))
        self.quality_velocity_obs_scale_m_s = float(env_cfg.get("quality_velocity_obs_scale_m_s", 0.25))
        raw_speed_height = env_cfg.get("near_target_speed_height_m")
        self.near_target_speed_height_m = None if raw_speed_height is None else float(raw_speed_height)
        self.clean_lift_reward_scale_m = float(env_cfg.get("clean_lift_reward_scale_m", 0.04))
        self.grasp_centering_reward_scale_m = float(env_cfg.get("grasp_centering_reward_scale_m", 0.005))
        self.release_quality_xy_scale_m = float(env_cfg.get("release_quality_xy_scale_m", 0.04))
        self.release_quality_height_scale_m = float(env_cfg.get("release_quality_height_scale_m", 0.04))
        self.release_quality_speed_scale_m_s = float(env_cfg.get("release_quality_speed_scale_m_s", 0.10))
        if (
            min(
                self.release_height_tolerance_m,
                self.release_max_obj_speed_m_s,
                self.pre_lift_max_drag_m,
                self.post_release_max_drift_m,
                self.post_release_clearance_max_m,
                self.post_release_clearance_gap_margin_m,
                self.landing_near_table_height_m,
                self.landing_speed_margin_height_m,
                self.landing_max_xy_speed_m_s,
                self.landing_max_down_speed_m_s,
                self.near_table_xy_action_height_m,
                self.quality_velocity_obs_scale_m_s,
                self.clean_lift_reward_scale_m,
                self.grasp_centering_reward_scale_m,
                self.release_quality_xy_scale_m,
                self.release_quality_height_scale_m,
                self.release_quality_speed_scale_m_s,
            )
            <= 0.0
        ):
            raise ValueError("quality thresholds and velocity observation scale must be positive")
        if self.near_target_speed_height_m is not None and self.near_target_speed_height_m <= 0.0:
            raise ValueError("near_target_speed_height_m must be positive when set")
        for zone_frac in (
            self.landing_xy_speed_safety_zone_frac,
            self.landing_down_speed_safety_zone_frac,
        ):
            if not (0.0 <= zone_frac < 1.0):
                raise ValueError("landing speed safety zone fractions must be in [0, 1)")
        # Soft-land: release credit height / speed shaping; throw penalty threshold.
        self.release_height_m = float(env_cfg.get("release_height_m", 0.05))
        self.release_speed_k = float(env_cfg.get("release_speed_k", 20.0))
        self.throw_speed_m_s = float(env_cfg.get("throw_speed_m_s", 0.15))
        self.push_before_grasp_dist_m = float(env_cfg.get("push_before_grasp_dist_m", 0.02))
        if self.release_height_m <= 0.0:
            raise ValueError("release_height_m must be positive")
        if self.release_speed_k < 0.0:
            raise ValueError("release_speed_k must be non-negative")
        if self.throw_speed_m_s <= 0.0:
            raise ValueError("throw_speed_m_s must be positive")
        if self.push_before_grasp_dist_m <= 0.0:
            raise ValueError("push_before_grasp_dist_m must be positive")

        self.gripper_open_gap_m = float(env_cfg["gripper_open_mm"]) / 1000.0
        self.gripper_close_gap_m = float(env_cfg["gripper_close_mm"]) / 1000.0
        # The command floor, not the nominal grasp gap, decides how hard the pads can
        # squeeze: once the drive stalls on the cube the remaining command margin is the
        # only thing left driving PD force. Defaults to the grasp gap for older recipes.
        self.gripper_min_command_gap_m = float(env_cfg.get("gripper_min_command_gap_m", self.gripper_close_gap_m))
        if not 0.0 <= self.gripper_min_command_gap_m <= self.gripper_close_gap_m:
            raise ValueError("gripper_min_command_gap_m must be in [0, gripper_close_gap_m]")
        self.contact_force_scale_n = float(env_cfg.get("contact_force_scale_n", 5.0))
        self.contact_force_threshold_n = float(env_cfg.get("contact_force_threshold_n", 0.05))
        self.include_contact_observations = bool(env_cfg.get("include_contact_observations", False))
        self.use_contact_holding = True
        # A held reset must begin at the cube-contact equilibrium, not at the
        # no-load close command. Teleporting every mimic joint to the latter
        # over-penetrates the cube and the first constraint solve ejects it.
        self.held_gripper_initial_drive = float(env_cfg.get("held_gripper_initial_drive", 0.60))
        if not math.isclose(self.gripper_open_gap_m, GRIPPER_G2_OPEN_GAP_M, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("Gripper G2 open gap must match the 84 mm SDK contract")
        if not 0.0 <= self.held_gripper_initial_drive <= 0.85:
            raise ValueError("held_gripper_initial_drive must be in [0.0, 0.85]")

        self.base_pos_world = torch.tensor(
            robot_cfg.get("base_pos", [0.30, 0.0, self.table_height]),
            device=self.device,
            dtype=gs.tc_float,
        )
        # Match trajectory pick-place viewer framing (base-relative lookat + offset).
        cam_lookat_base = (0.30, 0.15, 0.10)
        cam_lookat = (
            float(self.base_pos_world[0]) + cam_lookat_base[0],
            float(self.base_pos_world[1]) + cam_lookat_base[1],
            float(self.base_pos_world[2]) + cam_lookat_base[2],
        )
        cam_pos = (cam_lookat[0] + 0.90, cam_lookat[1] - 1.35, cam_lookat[2] + 0.60)

        # Match trajectory build_scene: same substeps / Newton rigid solver settings.
        self.substeps = int(env_cfg["substeps"])
        substep_dt = validate_g2_contact_substeps(dt=self.ctrl_dt, substeps=self.substeps)
        self.constraint_solver = str(env_cfg.get("constraint_solver", "newton"))
        self.solver_iterations = int(env_cfg.get("solver_iterations", RIGID_SOLVER_ITERATIONS))
        self.noslip_iterations = int(env_cfg.get("noslip_iterations", RIGID_NOSLIP_ITERATIONS))
        self.friction_cone = str(env_cfg.get("friction_cone", "elliptic"))
        self.contact_resolution = str(env_cfg.get("contact_resolution", "signorini"))
        self.use_gjk_collision = env_cfg.get("use_gjk_collision")
        constraint_timeconst = max(
            float(env_cfg.get("constraint_time_constant_s", RIGID_CONSTRAINT_TIMECONST)),
            2.0 * substep_dt,
        )
        self.fixed_demo_layout = bool(env_cfg.get("fixed_demo_layout", True))
        self.randomize_target = bool(env_cfg.get("randomize_target", True))
        self.place_phase_reset_frac = float(env_cfg.get("place_phase_reset_frac", 0.25))
        self.place_phase_hover_z_m = float(env_cfg.get("place_phase_hover_z_m", 0.07))
        self.place_phase_table_reset_frac = float(env_cfg.get("place_phase_table_reset_frac", 0.0))
        self.place_phase_table_hover_z_m = float(env_cfg.get("place_phase_table_hover_z_m", 0.0))
        if not 0.0 <= self.place_phase_table_reset_frac <= 1.0:
            raise ValueError("place_phase_table_reset_frac must be in [0, 1]")
        if not 0.0 <= self.place_phase_table_hover_z_m <= self.place_phase_hover_z_m:
            raise ValueError("place_phase_table_hover_z_m must be between 0 and place_phase_hover_z_m")
        # Carry-phase reverse-curriculum bootstrap: start already grasping the cube
        # LIFTED above the pickup, so the policy must learn transport->lower->release
        # (the transition a target-only place bootstrap skips). Kept static.
        self.carry_phase_reset_frac = float(env_cfg.get("carry_phase_reset_frac", 0.0))
        self.carry_phase_hover_z_m = float(env_cfg.get("carry_phase_hover_z_m", 0.10))
        # Grasp-phase reverse-curriculum bootstrap: start EE at the grasp pose (gripper
        # open, cube on the table) so the policy only has to learn close+lift. Annealed
        # to grasp_phase_reset_frac_final over grasp_phase_anneal_steps step() calls so
        # the from-home grasp skill takes over.
        self.grasp_phase_reset_frac_init = float(env_cfg.get("grasp_phase_reset_frac", 0.0))
        self.grasp_phase_reset_frac_final = float(env_cfg.get("grasp_phase_reset_frac_final", 0.0))
        self.grasp_phase_anneal_steps = int(env_cfg.get("grasp_phase_anneal_steps", 0))
        self.curriculum_min_count = int(env_cfg.get("curriculum_min_count", 1024))
        self.curriculum_min_stage_steps = int(env_cfg.get("curriculum_min_stage_steps", 0))
        self.curriculum_stage0_grasp_rate = float(env_cfg.get("curriculum_stage0_grasp_rate", 0.95))
        self.curriculum_stage1_grasp_rate = float(env_cfg.get("curriculum_stage1_grasp_rate", 0.90))
        self.curriculum_stage2_place_rate = float(env_cfg.get("curriculum_stage2_place_rate", 0.90))
        self.curriculum_stage3_place_rate = float(env_cfg.get("curriculum_stage3_place_rate", 0.85))
        self.curriculum_initial_stage = int(env_cfg.get("curriculum_initial_stage", 0))
        self.curriculum_max_stage = int(env_cfg.get("curriculum_max_stage", 4))
        self.curriculum_edge_fraction = float(env_cfg.get("curriculum_edge_fraction", 0.0))
        if not 0 <= self.curriculum_initial_stage <= self.curriculum_max_stage <= 4:
            raise ValueError("curriculum stages must satisfy 0 <= initial <= max <= 4")
        if self.curriculum_min_stage_steps < 0:
            raise ValueError("curriculum_min_stage_steps must be non-negative")
        if not 0.0 <= self.curriculum_edge_fraction <= 1.0:
            raise ValueError("curriculum_edge_fraction must be in [0, 1]")
        self.total_env_steps = 0

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt, substeps=self.substeps),
            rigid_options=make_rigid_options(
                gs,
                dt=self.ctrl_dt,
                constraint_solver=self.constraint_solver,
                friction_cone=self.friction_cone,
                contact_resolution=self.contact_resolution,
                enable_collision=True,
                enable_joint_limit=True,
                iterations=self.solver_iterations,
                noslip_iterations=self.noslip_iterations,
                constraint_timeconst=float(constraint_timeconst),
                use_gjk_collision=self.use_gjk_collision,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(min(10, self.num_envs)))),
            viewer_options=gs.options.ViewerOptions(
                refresh_rate=50,
                camera_pos=cam_pos,
                camera_lookat=cam_lookat,
                camera_fov=40,
            ),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        self.table = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.5, 0.8, self.table_height),
                pos=(0.45, 0.0, self.table_height / 2.0),
                fixed=True,
            ),
            surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.6, 0.6, 0.6))),
        )
        robot_morph = gs.morphs.URDF(
            file=robot_cfg["urdf_path"],
            pos=tuple(float(v) for v in self.base_pos_world.cpu().tolist()),
            fixed=True,
            requires_jac_and_IK=True,
        )
        self.obj = self.scene.add_entity(
            gs.morphs.Box(
                size=self.obj_size,
                pos=tuple(float(v) for v in self.base_to_world(self._cfg_vec("fixed_obj_pos")).cpu().tolist()),
                fixed=False,
            ),
            surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.9, 0.1, 0.1))),
        )
        self.target_marker = self.scene.add_entity(
            gs.morphs.Sphere(radius=0.02, fixed=True, collision=False),
            surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.0, 1.0, 0.0))),
        )

        self.capture_cam = None
        if env_cfg.get("capture_camera"):
            self.capture_cam = self.scene.add_camera(
                res=tuple(env_cfg.get("capture_res", (960, 720))),
                pos=tuple(env_cfg.get("capture_pos", cam_pos)),
                lookat=tuple(env_cfg.get("capture_lookat", cam_lookat)),
                fov=float(env_cfg.get("capture_fov", 40)),
            )

        # Match trajectory/packaging: preserve GLB PBR on import (enable_* keeps patch for the process).
        with glb_pbr_surfaces():
            self.robot = self.scene.add_entity(robot_morph, surface=glb_view_surface())
        self.scene.build(n_envs=self.num_envs)

        self.ik_link = self.robot.get_link(robot_cfg["ik_link_name"])
        self.left_finger_link = self.robot.get_link(robot_cfg["gripper_link_names"][0])
        self.right_finger_link = self.robot.get_link(robot_cfg["gripper_link_names"][1])
        self.obj.set_friction(float(OBJ_FRICTION))
        self.left_finger_link.set_friction(float(FINGER_FRICTION))
        self.right_finger_link.set_friction(float(FINGER_FRICTION))
        self.obj.set_links_mass(
            torch.tensor([self.obj_mass_kg], device=self.device, dtype=gs.tc_float),
        )
        self.collision_monitor_links = [
            self.robot.get_link(name) for name in robot_cfg.get("collision_monitor_links", [])
        ]
        self.arm_joint_names = robot_cfg["arm_joint_names"]
        self.arm_dof_idx = [self.robot.get_joint(name).dofs_idx_local[0] for name in self.arm_joint_names]
        self.gripper_joint_name = robot_cfg["gripper_joint_name"]
        self.gripper_dof_idx = [self.robot.get_joint(self.gripper_joint_name).dofs_idx_local[0]]
        self.all_dof_idx = self.arm_dof_idx + self.gripper_dof_idx

        self._setup_robot_control(robot_cfg)
        configure_g2_mimic_constraints(self.robot)

        self.default_gripper_drive = self.gap_m_to_drive_t(
            torch.full((self.num_envs,), self.gripper_open_gap_m, device=self.device, dtype=gs.tc_float)
        )[0].detach()
        self.default_arm_qpos = self._default_arm_qpos_from_ik(env_cfg)

        self.fixed_obj_pos = self._cfg_vec("fixed_obj_pos")
        self.fixed_target_pos = self._cfg_vec("fixed_target_pos")
        self.obj_spawn_lower = self._cfg_vec("obj_spawn_lower")
        self.obj_spawn_upper = self._cfg_vec("obj_spawn_upper")
        self.target_spawn_lower = self._cfg_vec("target_spawn_lower")
        self.target_spawn_upper = self._cfg_vec("target_spawn_upper")
        self.workspace_lower = self._cfg_vec("workspace_lower")
        self.workspace_upper = self._cfg_vec("workspace_upper")
        raw_scenarios = env_cfg.get("evaluation_scenarios")
        self._scenario_cursor = 0
        self._scenario_objects = None
        self._scenario_targets = None
        self._scenario_ids = None
        if raw_scenarios is not None:
            if not isinstance(raw_scenarios, list) or not raw_scenarios:
                raise ValueError("evaluation_scenarios must be a non-empty list")
            self._scenario_objects = torch.tensor(
                [scenario["object_position_m"] for scenario in raw_scenarios],
                device=self.device,
                dtype=gs.tc_float,
            )
            self._scenario_targets = torch.tensor(
                [scenario["target_position_m"] for scenario in raw_scenarios],
                device=self.device,
                dtype=gs.tc_float,
            )
            self._scenario_ids = torch.tensor(
                [int(scenario["id"]) for scenario in raw_scenarios],
                device=self.device,
                dtype=torch.int64,
            )

        self.reward_functions, self.episode_sums = {}, {}
        for name in self.reward_scales.keys():
            # Sparse events and telescoping potentials must not be crushed by ctrl_dt
            # (a hover would otherwise farm dense credit; a potential would be scaled away).
            if name not in _UNSCALED_REWARD_KEYS:
                self.reward_scales[name] *= self.ctrl_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)

        self.csv_log_path = None
        self._csv_file = None
        self._csv_writer = None

        self._init_buffers()
        self._scripted_action_guide = None
        self._scripted_action_hint = torch.zeros(
            self.num_envs,
            self.num_actions,
            device=self.device,
            dtype=gs.tc_float,
        )
        self._scripted_action_hint_step = -1
        if self.include_scripted_action_hint:
            from .expert import scripted_pick_place_expert_from_env

            self._scripted_action_guide = scripted_pick_place_expert_from_env(self)
        self.curriculum_stage = self.curriculum_initial_stage
        self.curriculum_stage_enter_step = self.total_env_steps
        self._initial_reset_done = False

        hist_len = int(env_cfg.get("success_history_len", 2000))
        self.grasp_success_history = torch.zeros(hist_len, device=self.device)
        self.lift_success_history = torch.zeros(hist_len, device=self.device)
        self.place_success_history = torch.zeros(hist_len, device=self.device)
        self.success_history = torch.zeros(hist_len, device=self.device)
        # Learned = from-home episodes only (exclude bootstrap resets). Honest skill signal.
        self.learned_grasp_success_history = torch.zeros(hist_len, device=self.device)
        self.learned_lift_success_history = torch.zeros(hist_len, device=self.device)
        self.learned_place_success_history = torch.zeros(hist_len, device=self.device)
        self.learned_success_history = torch.zeros(hist_len, device=self.device)
        self.grasp_history_idx = 0
        self.grasp_history_count = 0
        self.lift_history_idx = 0
        self.lift_history_count = 0
        self.place_history_idx = 0
        self.place_history_count = 0
        self.success_history_idx = 0
        self.success_history_count = 0
        self.learned_grasp_history_idx = 0
        self.learned_grasp_history_count = 0
        self.learned_lift_history_idx = 0
        self.learned_lift_history_count = 0
        self.learned_place_history_idx = 0
        self.learned_place_history_count = 0
        self.learned_success_history_idx = 0
        self.learned_success_history_count = 0
        self._metric_cohorts = (
            "learned_clean",
            "learned_noisy",
            "bootstrap_clean",
            "bootstrap_noisy",
        )
        self.cohort_grasp_histories = {
            label: torch.zeros(hist_len, device=self.device) for label in self._metric_cohorts
        }
        self.cohort_success_histories = {
            label: torch.zeros(hist_len, device=self.device) for label in self._metric_cohorts
        }
        self.cohort_history_indices = {label: 0 for label in self._metric_cohorts}
        self.cohort_history_counts = {label: 0 for label in self._metric_cohorts}

        self.reset()

    def _cfg_vec(self, key: str) -> torch.Tensor:
        return torch.tensor(self.env_cfg[key], device=self.device, dtype=gs.tc_float)

    def base_to_world(self, pos: torch.Tensor) -> torch.Tensor:
        return base_to_world_pos(pos, self.base_pos_world)

    def world_to_base(self, pos: torch.Tensor) -> torch.Tensor:
        return world_to_base_pos(pos, self.base_pos_world)

    def _setup_robot_control(self, robot_cfg: dict) -> None:
        self.robot.set_dofs_kp(torch.tensor(robot_cfg["kp"], device=self.device, dtype=gs.tc_float), self.arm_dof_idx)
        self.robot.set_dofs_kv(torch.tensor(robot_cfg["kv"], device=self.device, dtype=gs.tc_float), self.arm_dof_idx)
        self.robot.set_dofs_force_range(
            torch.tensor(robot_cfg["force_lower"], device=self.device, dtype=gs.tc_float),
            torch.tensor(robot_cfg["force_upper"], device=self.device, dtype=gs.tc_float),
            self.arm_dof_idx,
        )

        self.robot.set_dofs_kp(
            torch.tensor([robot_cfg["gripper_kp"]], device=self.device, dtype=gs.tc_float), self.gripper_dof_idx
        )
        self.robot.set_dofs_kv(
            torch.tensor([robot_cfg["gripper_kv"]], device=self.device, dtype=gs.tc_float), self.gripper_dof_idx
        )
        self.robot.set_dofs_force_range(
            torch.tensor([robot_cfg["gripper_force_lower"]], device=self.device, dtype=gs.tc_float),
            torch.tensor([robot_cfg["gripper_force_upper"]], device=self.device, dtype=gs.tc_float),
            self.gripper_dof_idx,
        )

        self.all_gripper_dof_idx = [
            self.robot.get_joint(name).dofs_idx_local[0] for name in robot_cfg["all_gripper_joint_names"]
        ]
        n_grip = len(self.all_gripper_dof_idx)
        self.robot.set_dofs_damping(
            torch.full((n_grip,), robot_cfg["gripper_damping"], device=self.device, dtype=gs.tc_float),
            self.all_gripper_dof_idx,
        )
        self.robot.set_dofs_frictionloss(
            torch.full((n_grip,), robot_cfg["gripper_frictionloss"], device=self.device, dtype=gs.tc_float),
            self.all_gripper_dof_idx,
        )

    def _default_arm_qpos_from_ik(self, env_cfg: dict) -> torch.Tensor:
        default_ee_base = torch.tensor(
            [env_cfg.get("default_ee_pos", [0.3, 0.0, 0.3])],
            device=self.device,
            dtype=gs.tc_float,
        ).expand(self.num_envs, 3)
        default_ee_world = self.base_to_world(default_ee_base)
        x, y, z, w = GRIPPER_DOWN_QUAT_XYZW
        down_quat = torch.tensor([[w, x, y, z]], device=self.device, dtype=gs.tc_float).expand(self.num_envs, 4)
        init_qpos = self.robot.inverse_kinematics(
            link=self.ik_link,
            pos=default_ee_world,
            quat=down_quat,
            dofs_idx_local=self.arm_dof_idx,
        )
        return init_qpos[0, self.arm_dof_idx].detach()

    def _bootstrap_link6_target(self, obj_base: torch.Tensor) -> torch.Tensor:
        return logic_desired_grasp_pos_base(
            obj_base,
            self.grasp_center_offset_z + self.tool_tip_offset_z_m,
        )

    def _init_buffers(self) -> None:
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_int)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=gs.tc_float)
        self.initial_obj_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=gs.tc_float)
        # Immutable task-layout context. Unlike initial_obj_pos, this stays at the
        # sampled pickup pose even when a bootstrap reset teleports the cube elsewhere.
        self.episode_layout_obj_pos = torch.zeros(
            self.num_envs,
            3,
            device=self.device,
            dtype=gs.tc_float,
        )
        self.episode_layout_target_pos = torch.zeros_like(self.episode_layout_obj_pos)
        self.scenario_id = torch.full(
            (self.num_envs,),
            -1,
            device=self.device,
            dtype=torch.int64,
        )
        # Stage at episode start. After a curriculum transition, late completions
        # from the previous envelope must not repopulate the freshly cleared gate.
        self.episode_curriculum_stage = torch.full(
            (self.num_envs,),
            self.curriculum_initial_stage,
            device=self.device,
            dtype=torch.int64,
        )
        self.holding = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.carry = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ever_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ever_carried_near = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ever_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # True for episodes that started from a bootstrap reset (place- or grasp-phase),
        # so learned_* metrics can exclude them and report honest from-home skill.
        self.is_bootstrap_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_action_noise_enabled = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.episode_place_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.place_stable_steps = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_int)
        self.release_started = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.release_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.release_violation = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.near_table_entered = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.hard_landing_violation = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.quality_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.max_pre_lift_xy_m = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=gs.tc_float,
        )
        self.max_landing_xy_speed_m_s = torch.zeros_like(self.max_pre_lift_xy_m)
        self.max_landing_down_speed_m_s = torch.zeros_like(self.max_pre_lift_xy_m)
        self.release_xy_dist_m = torch.zeros_like(self.max_pre_lift_xy_m)
        self.release_height_error_m = torch.zeros_like(self.max_pre_lift_xy_m)
        self.release_speed_m_s = torch.zeros_like(self.max_pre_lift_xy_m)
        self.post_release_drift_m = torch.zeros_like(self.max_pre_lift_xy_m)
        self.post_release_clearance_m = torch.zeros_like(self.max_pre_lift_xy_m)
        self.prev_post_release_clearance_m = torch.zeros_like(self.max_pre_lift_xy_m)
        self.release_clearance_achieved = torch.zeros_like(self.release_started)
        self.post_release_recontact = torch.zeros_like(self.release_started)
        self.post_release_recontact_event = torch.zeros_like(self.release_started)
        self.left_contact_force_n = torch.zeros_like(self.max_pre_lift_xy_m)
        self.right_contact_force_n = torch.zeros_like(self.max_pre_lift_xy_m)
        self.grasp_offset_xy_m = torch.zeros_like(self.max_pre_lift_xy_m)
        self.release_obj_xy = torch.zeros(
            self.num_envs,
            2,
            device=self.device,
            dtype=gs.tc_float,
        )
        self.prev_release_started = torch.zeros_like(self.release_started)
        self.prev_near_table_entered = torch.zeros_like(self.near_table_entered)
        self.prev_hard_landing_violation = torch.zeros_like(self.hard_landing_violation)
        self.prev_max_pre_lift_xy_m = torch.zeros_like(self.max_pre_lift_xy_m)
        self.prev_target_xy_dist_m = torch.zeros_like(self.max_pre_lift_xy_m)
        self.prev_obj_speed_m_s = torch.zeros_like(self.max_pre_lift_xy_m)
        self.episode_action_sat_sum = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        self.episode_action_near_bound_sum = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=gs.tc_float,
        )
        self.episode_action_clip_sum = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        self.episode_delta_sat_sum = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        self.episode_gripper_bound_sum = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        self.episode_ik_failure_sum = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        self.episode_ik_jump_reject_sum = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        self.prev_obj_z = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        self.prev_gripper_gap = torch.full(
            (self.num_envs,),
            self.gripper_open_gap_m,
            device=self.device,
            dtype=gs.tc_float,
        )
        # Previous finger-center -> grasp-pose distance, for telescoping approach shaping.
        self.prev_grasp_dist = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        self.prev_finger_center_base = torch.zeros(
            self.num_envs,
            3,
            device=self.device,
            dtype=gs.tc_float,
        )
        self.ee_velocity_base = torch.zeros_like(self.prev_finger_center_base)
        # ever_grasped snapshot from before the current step's task-state update, so the
        # one-shot grasp bonus fires exactly on the first grasp-and-lift of the episode.
        self.prev_ever_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Commanded gripper gap (integrated position command, NOT the measured gap). The
        # policy nudges this each step; driving to it (rather than to measured+delta) lets
        # the gripper build full PD force against a rigid cube instead of stalling on contact.
        self.commanded_gap = torch.full(
            (self.num_envs,), self.gripper_open_gap_m, device=self.device, dtype=gs.tc_float
        )
        self.gripper_contact_hold = G2ContactHoldController(
            self.num_envs,
            device=self.device,
            dtype=gs.tc_float,
            initial_gap_m=self.gripper_open_gap_m,
        )
        self.previous_action = torch.zeros(
            self.num_envs,
            self.num_actions,
            device=self.device,
            dtype=gs.tc_float,
        )
        # Integrated Cartesian setpoint for the IK link, used when the recipe selects
        # commanded-pose integration. Re-latched to the measured pose on every reset.
        self.ee_setpoint_base = torch.zeros(
            self.num_envs,
            3,
            device=self.device,
            dtype=gs.tc_float,
        )
        self.extras = {"observations": {}}

    def reset(self) -> tuple[TensorDict, dict]:
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs = self.get_observations()
        return obs, self.extras

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        if len(envs_idx) == 0:
            return
        self._scripted_action_hint_step = -1

        has_episode_steps = bool(torch.any(self.episode_length_buf[envs_idx] > 0).item())
        if self._initial_reset_done and has_episode_steps:
            self._record_episode_outcomes(envs_idx)
            self._maybe_update_curriculum()

        self._write_episode_extras(envs_idx)
        self.episode_curriculum_stage[envs_idx] = self.curriculum_stage

        self.episode_length_buf[envs_idx] = 0
        self.holding[envs_idx] = False
        self.carry[envs_idx] = False
        self.grasped[envs_idx] = False
        self.ever_grasped[envs_idx] = False
        self.ever_carried_near[envs_idx] = False
        self.ever_lifted[envs_idx] = False
        self.is_bootstrap_episode[envs_idx] = False
        noise_available = (
            max(
                self.train_action_noise_std,
                self.train_action_noise_std_end,
            )
            > 0.0
        )
        self.episode_action_noise_enabled[envs_idx] = action_noise_episode_mask(
            torch.rand(len(envs_idx), device=self.device),
            clean_episode_frac=self.train_action_noise_clean_episode_frac,
            noise_available=noise_available,
        )
        # Reset the integrated gripper command to fully open; place bootstrap overrides to
        # closed below since it starts already holding the cube.
        self.commanded_gap[envs_idx] = self.gripper_open_gap_m
        self.gripper_contact_hold.reset(envs_idx, gap_m=self.commanded_gap[envs_idx])
        self.previous_action[envs_idx] = 0.0
        self.episode_place_success[envs_idx] = False
        self.episode_success[envs_idx] = False
        self.place_stable_steps[envs_idx] = 0
        self.release_started[envs_idx] = False
        self.release_valid[envs_idx] = False
        self.release_violation[envs_idx] = False
        self.near_table_entered[envs_idx] = False
        self.hard_landing_violation[envs_idx] = False
        self.quality_ok[envs_idx] = False
        self.max_pre_lift_xy_m[envs_idx] = 0.0
        self.max_landing_xy_speed_m_s[envs_idx] = 0.0
        self.max_landing_down_speed_m_s[envs_idx] = 0.0
        self.release_xy_dist_m[envs_idx] = 0.0
        self.release_height_error_m[envs_idx] = 0.0
        self.release_speed_m_s[envs_idx] = 0.0
        self.post_release_drift_m[envs_idx] = 0.0
        self.post_release_clearance_m[envs_idx] = 0.0
        self.prev_post_release_clearance_m[envs_idx] = 0.0
        self.release_clearance_achieved[envs_idx] = False
        self.post_release_recontact[envs_idx] = False
        self.post_release_recontact_event[envs_idx] = False
        self.left_contact_force_n[envs_idx] = 0.0
        self.right_contact_force_n[envs_idx] = 0.0
        self.grasp_offset_xy_m[envs_idx] = 0.0
        self.release_obj_xy[envs_idx] = 0.0
        self.prev_release_started[envs_idx] = False
        self.prev_near_table_entered[envs_idx] = False
        self.prev_hard_landing_violation[envs_idx] = False
        self.prev_max_pre_lift_xy_m[envs_idx] = 0.0
        self.episode_action_sat_sum[envs_idx] = 0.0
        self.episode_action_near_bound_sum[envs_idx] = 0.0
        self.episode_action_clip_sum[envs_idx] = 0.0
        self.episode_delta_sat_sum[envs_idx] = 0.0
        self.episode_gripper_bound_sum[envs_idx] = 0.0
        self.episode_ik_failure_sum[envs_idx] = 0.0
        self.episode_ik_jump_reject_sum[envs_idx] = 0.0

        n = len(envs_idx)
        default_qpos = torch.zeros(n, self.robot.n_dofs, device=self.device, dtype=gs.tc_float)
        for i, idx in enumerate(self.arm_dof_idx):
            default_qpos[:, idx] = self.default_arm_qpos[i]
        # Genesis does not expand URDF mimic joints during a direct state reset. Set
        # the complete G2 linkage consistently so the first physics step does not snap
        # passive fingers from zero and eject a bootstrapped cube.
        for dof in self.all_gripper_dof_idx:
            default_qpos[:, dof] = self.default_gripper_drive

        obj_base, target_base, scenario_ids = self._sample_object_and_target_base(n)
        obj_base = obj_base.clone()
        layout_obj_base = obj_base.clone()
        layout_target_base = target_base.clone()
        u = torch.rand(n, device=self.device)
        place_mask, carry_mask, grasp_mask = sample_reset_phase_masks(
            u,
            self.place_phase_reset_frac,
            self._current_grasp_phase_reset_frac(),
            self.carry_phase_reset_frac,
        )
        bootstrap_mask = place_mask | carry_mask | grasp_mask
        held_mask = place_mask | carry_mask  # bootstraps that start already holding the cube

        if bootstrap_mask.any():
            # One full-env IK pass (Genesis batch size = num_envs) shared by all phases.
            full_ee = self.world_to_base(self.ik_link.get_pos()).detach().clone()
            full_init = self.robot.get_dofs_position(self.arm_dof_idx).detach().clone()

            if place_mask.any():
                # Place bootstrap: cube already held above the place target. An optional
                # table-ready subset supplies valid-release experience in the same PPO
                # batch as high starts that penalize premature opening.
                hover_offsets = torch.full(
                    (n,),
                    self.place_phase_hover_z_m,
                    device=self.device,
                    dtype=gs.tc_float,
                )
                if self.place_phase_table_reset_frac > 0.0:
                    table_ready_mask = place_mask & (
                        torch.rand(n, device=self.device) < self.place_phase_table_reset_frac
                    )
                    hover_offsets[table_ready_mask] = self.place_phase_table_hover_z_m
                obj_base[place_mask, 0] = target_base[place_mask, 0]
                obj_base[place_mask, 1] = target_base[place_mask, 1]
                obj_base[place_mask, 2] = self.obj_rest_z_base + hover_offsets[place_mask]
                global_place = envs_idx[place_mask]
                # IK targets link6; raise the finger-center grasp target by the
                # link6->finger-center offset so the closed gripper actually holds the cube.
                full_ee[global_place] = self._bootstrap_link6_target(obj_base[place_mask])

            if carry_mask.any():
                # Carry bootstrap: cube already held LIFTED above the PICKUP; the policy
                # must learn transport -> lower -> release (the skill a target-only place
                # bootstrap never exercises). XY stays at the sampled pickup.
                carry_z = self.obj_rest_z_base + self.carry_phase_hover_z_m
                obj_base[carry_mask, 2] = carry_z
                global_carry = envs_idx[carry_mask]
                full_ee[global_carry] = self._bootstrap_link6_target(obj_base[carry_mask])

            if grasp_mask.any():
                # Grasp bootstrap: EE pre-posed at the grasp pose over the on-table cube;
                # gripper stays open and the cube is NOT pre-grasped (learn close+lift).
                # IK targets link6; raise the finger-center grasp target by the
                # link6->finger-center offset so the fingers straddle the cube (not the table).
                global_grasp = envs_idx[grasp_mask]
                full_ee[global_grasp] = self._bootstrap_link6_target(obj_base[grasp_mask])

            # A reset is an instantaneous state assignment, not a control step. On the
            # constructor's first reset Genesis still has q=0, so enforcing the per-step
            # jump limit would reject every valid bootstrap IK solution and silently keep
            # the zero pose far from the cube.
            full_arm_q = self._ik_arm_qpos(full_ee, init_arm_q=full_init, enforce_jump_limit=False)
            # Scatter solved arm qpos into the reset rows for every bootstrap env.
            boot_local = torch.nonzero(bootstrap_mask, as_tuple=False).squeeze(-1)
            global_boot = envs_idx[bootstrap_mask]
            q_boot = full_arm_q[global_boot]
            for j, dof in enumerate(self.arm_dof_idx):
                default_qpos[boot_local, dof] = q_boot[:, j]

            if held_mask.any():
                # Place- and carry-phase bootstraps both start already holding the cube:
                # initialize the entire mimic linkage at its cube-contact state and keep
                # commanding close on the drive joint so PD force maintains the grasp.
                for dof in self.all_gripper_dof_idx:
                    default_qpos[held_mask, dof] = self.held_gripper_initial_drive
                global_held = envs_idx[held_mask]
                self.commanded_gap[global_held] = self.gripper_close_gap_m
                held_gap = self.drive_to_gap_m_t(
                    torch.full(
                        (len(global_held),),
                        self.held_gripper_initial_drive,
                        device=self.device,
                        dtype=gs.tc_float,
                    )
                )
                self.gripper_contact_hold.prime(global_held, held_gap)
                self.ever_grasped[global_held] = True
                self.ever_lifted[global_held] = True
                # Place bootstrap already starts held above the target: latch the
                # near-carry gate so lower+release shaping is available immediately.
                if place_mask.any():
                    self.ever_carried_near[envs_idx[place_mask]] = True

            # All bootstrap phases are excluded from learned-from-home metrics.
            self.is_bootstrap_episode[envs_idx[bootstrap_mask]] = True

        self.robot.set_qpos(default_qpos, envs_idx=envs_idx)

        if bootstrap_mask.any():
            # The G2 mimic joints move the link-frame midpoint laterally when the drive
            # joint is set. A z-only tool offset therefore leaves the physical fingers
            # off-center. Measure the actual post-set_qpos finger center for each reset
            # row, translate link6 by the exact residual, and solve once more. This also
            # handles the different open/closed offsets of grasp vs carry/place resets.
            global_boot = envs_idx[bootstrap_mask]
            desired_finger = logic_desired_grasp_pos_base(
                obj_base[bootstrap_mask],
                self.grasp_center_offset_z,
            )
            full_ee = self.world_to_base(self.ik_link.get_pos()).detach().clone()
            actual_finger = self.finger_center_base().detach()
            full_ee[global_boot] += desired_finger - actual_finger[global_boot]
            full_init = self.robot.get_dofs_position(self.arm_dof_idx).detach().clone()
            corrected_arm_q = self._ik_arm_qpos(full_ee, init_arm_q=full_init, enforce_jump_limit=False)
            corrected_boot = corrected_arm_q[global_boot]
            boot_local = torch.nonzero(bootstrap_mask, as_tuple=False).squeeze(-1)
            for j, dof in enumerate(self.arm_dof_idx):
                default_qpos[boot_local, dof] = corrected_boot[:, j]
            self.robot.set_qpos(default_qpos, envs_idx=envs_idx)

        obj_world = self.base_to_world(obj_base)
        target_world = self.base_to_world(target_base)
        obj_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=gs.tc_float).expand(n, 4)
        self.obj.set_pos(obj_world, envs_idx=envs_idx, zero_velocity=True)
        self.obj.set_quat(obj_quat, envs_idx=envs_idx, zero_velocity=True)
        self.initial_obj_pos[envs_idx] = obj_base
        self.target_pos[envs_idx] = target_base
        self.episode_layout_obj_pos[envs_idx] = layout_obj_base
        self.episode_layout_target_pos[envs_idx] = layout_target_base
        self.scenario_id[envs_idx] = scenario_ids
        self.target_marker.set_pos(target_world, envs_idx=envs_idx)
        self.prev_obj_z[envs_idx] = obj_base[:, 2]
        self.prev_target_xy_dist_m[envs_idx] = torch.norm(
            obj_base[:, :2] - target_base[:, :2],
            dim=-1,
        )
        self.prev_obj_speed_m_s[envs_idx] = 0.0
        # Seed the approach potential at the post-reset pose (no spurious first-step delta).
        self.prev_grasp_dist[envs_idx] = self.grasp_dist()[envs_idx]
        self.prev_gripper_gap[envs_idx] = self.gripper_gap_m()[envs_idx]
        # Re-latch the Cartesian setpoint onto the freshly teleported arm so the first
        # action of the episode is measured from the actual pose, not the previous one.
        self.ee_setpoint_base[envs_idx] = self.world_to_base(self.ik_link.get_pos())[envs_idx]
        current_finger = self.finger_center_base()[envs_idx]
        self.prev_finger_center_base[envs_idx] = current_finger
        self.ee_velocity_base[envs_idx] = 0.0

        if self.csv_log_path is not None:
            self._write_csv_log()
        self._initial_reset_done = True

    def _current_grasp_phase_reset_frac(self) -> float:
        """Linearly annealed grasp-phase bootstrap fraction (init -> final over steps)."""
        if self.grasp_phase_anneal_steps <= 0:
            return self.grasp_phase_reset_frac_init
        progress = self.total_env_steps / float(self.grasp_phase_anneal_steps)
        return annealed_frac(self.grasp_phase_reset_frac_init, self.grasp_phase_reset_frac_final, progress)

    def _sample_object_and_target_base(
        self,
        n: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._scenario_objects is not None:
            bank_size = int(self._scenario_objects.shape[0])
            available = max(0, min(n, bank_size - self._scenario_cursor))
            obj = self.fixed_obj_pos.unsqueeze(0).expand(n, 3).clone()
            target = self.fixed_target_pos.unsqueeze(0).expand(n, 3).clone()
            scenario_ids = torch.full((n,), -1, device=self.device, dtype=torch.int64)
            if available:
                indices = torch.arange(
                    self._scenario_cursor,
                    self._scenario_cursor + available,
                    device=self.device,
                )
                obj[:available] = self._scenario_objects[indices]
                target[:available] = self._scenario_targets[indices]
                scenario_ids[:available] = self._scenario_ids[indices]
                self._scenario_cursor += available
            return obj, target, scenario_ids
        # fixed_demo_layout locks the trajectory pick-place fixed object/target.
        stage = 0 if self.fixed_demo_layout else self.curriculum_stage
        obj, target = sample_object_and_target(
            stage,
            n,
            device=self.device,
            dtype=gs.tc_float,
            fixed_obj=self.fixed_obj_pos,
            fixed_target=self.fixed_target_pos,
            obj_spawn_lower=self.obj_spawn_lower,
            obj_spawn_upper=self.obj_spawn_upper,
            target_spawn_lower=self.target_spawn_lower,
            target_spawn_upper=self.target_spawn_upper,
            edge_fraction=self.curriculum_edge_fraction,
            randomize_target=self.randomize_target,
        )
        scenario_ids = torch.full((n,), -1, device=self.device, dtype=torch.int64)
        return obj, target, scenario_ids

    def _record_episode_outcomes(self, envs_idx: torch.Tensor) -> None:
        self.grasp_history_idx, self.grasp_history_count = self._append_history(
            self.grasp_success_history,
            self.grasp_history_idx,
            self.grasp_history_count,
            self.ever_grasped[envs_idx].float(),
        )
        self.lift_history_idx, self.lift_history_count = self._append_history(
            self.lift_success_history,
            self.lift_history_idx,
            self.lift_history_count,
            self.ever_lifted[envs_idx].float(),
        )

        # Keep honest metrics on the device: selecting the from-home rows in one tensor
        # operation avoids a GPU synchronization for every completed environment.
        current_stage_mask = self.episode_curriculum_stage[envs_idx] == self.curriculum_stage
        learned_mask = (~self.is_bootstrap_episode[envs_idx]) & current_stage_mask
        self.learned_grasp_history_idx, self.learned_grasp_history_count = self._append_history(
            self.learned_grasp_success_history,
            self.learned_grasp_history_idx,
            self.learned_grasp_history_count,
            self.ever_grasped[envs_idx][learned_mask].float(),
        )
        self.learned_lift_history_idx, self.learned_lift_history_count = self._append_history(
            self.learned_lift_success_history,
            self.learned_lift_history_idx,
            self.learned_lift_history_count,
            self.ever_lifted[envs_idx][learned_mask].float(),
        )
        self.learned_place_history_idx, self.learned_place_history_count = self._append_history(
            self.learned_place_success_history,
            self.learned_place_history_idx,
            self.learned_place_history_count,
            self.episode_place_success[envs_idx][learned_mask].float(),
        )
        self.learned_success_history_idx, self.learned_success_history_count = self._append_history(
            self.learned_success_history,
            self.learned_success_history_idx,
            self.learned_success_history_count,
            self.episode_success[envs_idx][learned_mask].float(),
        )
        self.place_history_idx, self.place_history_count = self._append_history(
            self.place_success_history,
            self.place_history_idx,
            self.place_history_count,
            self.episode_place_success[envs_idx].float(),
        )
        self.success_history_idx, self.success_history_count = self._append_history(
            self.success_history,
            self.success_history_idx,
            self.success_history_count,
            self.episode_success[envs_idx].float(),
        )
        bootstrap = self.is_bootstrap_episode[envs_idx]
        noisy = self.episode_action_noise_enabled[envs_idx]
        cohort_masks = {
            "learned_clean": (~bootstrap) & (~noisy) & current_stage_mask,
            "learned_noisy": (~bootstrap) & noisy & current_stage_mask,
            "bootstrap_clean": bootstrap & (~noisy) & current_stage_mask,
            "bootstrap_noisy": bootstrap & noisy & current_stage_mask,
        }
        for label, mask in cohort_masks.items():
            index = self.cohort_history_indices[label]
            count = self.cohort_history_counts[label]
            grasp_index, grasp_count = self._append_history(
                self.cohort_grasp_histories[label],
                index,
                count,
                self.ever_grasped[envs_idx][mask].float(),
            )
            success_index, success_count = self._append_history(
                self.cohort_success_histories[label],
                index,
                count,
                self.episode_success[envs_idx][mask].float(),
            )
            if (grasp_index, grasp_count) != (success_index, success_count):
                raise RuntimeError("cohort metric histories lost alignment")
            self.cohort_history_indices[label] = grasp_index
            self.cohort_history_counts[label] = grasp_count

    @staticmethod
    def _append_history(
        history: torch.Tensor,
        index: int,
        count: int,
        values: torch.Tensor,
    ) -> tuple[int, int]:
        """Append a device tensor to a fixed-size circular history without scalar syncs."""
        capacity = len(history)
        total = int(values.numel())
        if total == 0:
            return index, count

        if total > capacity:
            values = values[-capacity:]
            start = (index + total - capacity) % capacity
        else:
            start = index
        positions = (torch.arange(values.numel(), device=history.device) + start) % capacity
        history[positions] = values.to(device=history.device, dtype=history.dtype)
        return (index + total) % capacity, min(count + total, capacity)

    def _maybe_update_curriculum(self) -> None:
        if self.fixed_demo_layout:
            return
        stage_steps = self.total_env_steps - self.curriculum_stage_enter_step
        # A stage cannot consume the previous stage's success window and advance again
        # during another reset group in the same control step. Recipes may require a
        # longer dwell so PPO has time to adapt before the envelope expands again.
        if stage_steps <= 0 or stage_steps < self.curriculum_min_stage_steps:
            return
        grasp_rate = self._history_rate(
            self.learned_grasp_success_history,
            self.learned_grasp_history_count,
        )
        place_rate = self._history_rate(
            self.learned_success_history,
            self.learned_success_history_count,
        )
        new_stage = next_curriculum_stage(
            self.curriculum_stage,
            grasp_rate,
            place_rate,
            self.learned_grasp_history_count,
            self.learned_success_history_count,
            min_count=self.curriculum_min_count,
            stage0_grasp_rate=self.curriculum_stage0_grasp_rate,
            stage1_grasp_rate=self.curriculum_stage1_grasp_rate,
            stage2_place_rate=self.curriculum_stage2_place_rate,
            stage3_place_rate=self.curriculum_stage3_place_rate,
            max_stage=self.curriculum_max_stage,
        )
        if new_stage == self.curriculum_stage:
            return
        if self.curriculum_stage == 0 and new_stage == 1:
            print(f"[Curriculum] Stage 0 -> 1 (grasp_rate={grasp_rate:.2f}): narrow random object spawn")
        elif self.curriculum_stage == 1 and new_stage == 2:
            print(f"[Curriculum] Stage 1 -> 2 (grasp_rate={grasp_rate:.2f}): quarter object envelope")
        elif self.curriculum_stage == 2 and new_stage == 3:
            print(f"[Curriculum] Stage 2 -> 3 (place_rate={place_rate:.2f}): half object envelope")
        elif self.curriculum_stage == 3 and new_stage == 4:
            print(f"[Curriculum] Stage 3 -> 4 (place_rate={place_rate:.2f}): full object envelope")
        self.curriculum_stage = new_stage
        self.curriculum_stage_enter_step = self.total_env_steps
        self._reset_learned_histories()

    def _reset_learned_histories(self) -> None:
        """Start each curriculum stage with an empty, correctly indexed window."""

        for history in (
            self.learned_grasp_success_history,
            self.learned_lift_success_history,
            self.learned_place_success_history,
            self.learned_success_history,
        ):
            history.zero_()
        self.learned_grasp_history_idx = 0
        self.learned_grasp_history_count = 0
        self.learned_lift_history_idx = 0
        self.learned_lift_history_count = 0
        self.learned_place_history_idx = 0
        self.learned_place_history_count = 0
        self.learned_success_history_idx = 0
        self.learned_success_history_count = 0
        for label in self._metric_cohorts:
            self.cohort_grasp_histories[label].zero_()
            self.cohort_success_histories[label].zero_()
            self.cohort_history_indices[label] = 0
            self.cohort_history_counts[label] = 0

    def _history_rate(self, history: torch.Tensor, count: int) -> float:
        if count <= 0:
            return 0.0
        return float(history[:count].mean().item())

    def _write_episode_extras(self, envs_idx: torch.Tensor) -> None:
        steps = self.episode_length_buf[envs_idx].float().clamp(min=1.0)
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self.extras["episode"]["curriculum_stage"] = self.curriculum_stage
        self.extras["episode"]["grasp_success_rate"] = self._history_rate(
            self.grasp_success_history,
            self.grasp_history_count,
        )
        self.extras["episode"]["lift_success_rate"] = self._history_rate(
            self.lift_success_history,
            self.lift_history_count,
        )
        self.extras["episode"]["learned_grasp_success_rate"] = self._history_rate(
            self.learned_grasp_success_history,
            self.learned_grasp_history_count,
        )
        self.extras["episode"]["learned_lift_success_rate"] = self._history_rate(
            self.learned_lift_success_history,
            self.learned_lift_history_count,
        )
        self.extras["episode"]["learned_place_success_rate"] = self._history_rate(
            self.learned_place_success_history,
            self.learned_place_history_count,
        )
        self.extras["episode"]["learned_success_rate"] = self._history_rate(
            self.learned_success_history,
            self.learned_success_history_count,
        )
        self.extras["episode"]["place_success_rate"] = self._history_rate(
            self.place_success_history,
            self.place_history_count,
        )
        self.extras["episode"]["success_rate"] = self._history_rate(
            self.success_history,
            self.success_history_count,
        )
        self.extras["episode"]["batch_step"] = int(self.total_env_steps)
        steps_per_iteration = max(
            1,
            int(self.env_cfg.get("training_steps_per_iteration", 128)),
        )
        self.extras["episode"]["training_iteration"] = int(self.total_env_steps // steps_per_iteration)
        self.extras["episode"]["train_action_noise_std"] = float(self._current_action_noise_std())
        for label in self._metric_cohorts:
            count = self.cohort_history_counts[label]
            self.extras["episode"][f"{label}_episode_count"] = count
            self.extras["episode"][f"{label}_grasp_success_rate"] = self._history_rate(
                self.cohort_grasp_histories[label],
                count,
            )
            self.extras["episode"][f"{label}_success_rate"] = self._history_rate(
                self.cohort_success_histories[label],
                count,
            )
        self.extras["episode"]["action_saturation_fraction"] = torch.mean(
            self.episode_action_sat_sum[envs_idx] / steps
        ).item()
        self.extras["episode"]["action_near_bound_fraction"] = torch.mean(
            self.episode_action_near_bound_sum[envs_idx] / steps
        ).item()
        self.extras["episode"]["action_clip_fraction"] = torch.mean(
            self.episode_action_clip_sum[envs_idx] / steps
        ).item()
        self.extras["episode"]["delta_saturation_fraction"] = torch.mean(
            self.episode_delta_sat_sum[envs_idx] / steps
        ).item()
        self.extras["episode"]["gripper_bound_fraction"] = torch.mean(
            self.episode_gripper_bound_sum[envs_idx] / steps
        ).item()
        self.extras["episode"]["ik_failure_fraction"] = torch.mean(self.episode_ik_failure_sum[envs_idx] / steps).item()
        self.extras["episode"]["ik_jump_reject_fraction"] = torch.mean(
            self.episode_ik_jump_reject_sum[envs_idx] / steps
        ).item()
        self.extras["episode"]["quality_pass_rate"] = torch.mean(self.quality_ok[envs_idx].float()).item()
        self.extras["episode"]["release_valid_rate"] = torch.mean(self.release_valid[envs_idx].float()).item()
        self.extras["episode"]["release_violation_rate"] = torch.mean(self.release_violation[envs_idx].float()).item()
        self.extras["episode"]["hard_landing_rate"] = torch.mean(self.hard_landing_violation[envs_idx].float()).item()
        self.extras["episode"]["mean_pre_lift_drag_m"] = torch.mean(self.max_pre_lift_xy_m[envs_idx]).item()
        self.extras["episode"]["max_pre_lift_drag_m"] = torch.max(self.max_pre_lift_xy_m[envs_idx]).item()
        self.extras["episode"]["mean_post_release_drift_m"] = torch.mean(self.post_release_drift_m[envs_idx]).item()
        self.extras["episode"]["max_post_release_drift_m"] = torch.max(self.post_release_drift_m[envs_idx]).item()
        self.extras["episode"]["release_clearance_achieved_rate"] = torch.mean(
            self.release_clearance_achieved[envs_idx].float()
        ).item()
        self.extras["episode"]["post_release_recontact_rate"] = torch.mean(
            self.post_release_recontact[envs_idx].float()
        ).item()
        lifted_rows = envs_idx[self.ever_grasped[envs_idx]]
        self.extras["episode"]["mean_grasp_offset_xy_m"] = (
            torch.mean(self.grasp_offset_xy_m[lifted_rows]).item() if len(lifted_rows) > 0 else 0.0
        )

    def _write_csv_log(self) -> None:
        ep = self.extras.get("episode", {})
        if not ep:
            return
        row = {
            "batch_step": ep.get("batch_step", self.total_env_steps),
            "training_iteration": ep.get("training_iteration", 0),
            "train_action_noise_std": ep.get("train_action_noise_std", 0.0),
            "curriculum_stage": ep.get("curriculum_stage", self.curriculum_stage),
            "grasp_success_rate": ep.get("grasp_success_rate", 0.0),
            "lift_success_rate": ep.get("lift_success_rate", 0.0),
            "learned_grasp_success_rate": ep.get("learned_grasp_success_rate", 0.0),
            "learned_lift_success_rate": ep.get("learned_lift_success_rate", 0.0),
            "learned_place_success_rate": ep.get("learned_place_success_rate", 0.0),
            "learned_success_rate": ep.get("learned_success_rate", 0.0),
            "place_success_rate": ep.get("place_success_rate", 0.0),
            "success_rate": ep.get("success_rate", 0.0),
            "action_saturation_fraction": ep.get("action_saturation_fraction", 0.0),
            "action_near_bound_fraction": ep.get("action_near_bound_fraction", 0.0),
            "action_clip_fraction": ep.get("action_clip_fraction", 0.0),
            "delta_saturation_fraction": ep.get("delta_saturation_fraction", 0.0),
            "gripper_bound_fraction": ep.get("gripper_bound_fraction", 0.0),
            "ik_failure_fraction": ep.get("ik_failure_fraction", 0.0),
            "ik_jump_reject_fraction": ep.get("ik_jump_reject_fraction", 0.0),
            "quality_pass_rate": ep.get("quality_pass_rate", 0.0),
            "release_valid_rate": ep.get("release_valid_rate", 0.0),
            "release_violation_rate": ep.get("release_violation_rate", 0.0),
            "hard_landing_rate": ep.get("hard_landing_rate", 0.0),
            "mean_pre_lift_drag_m": ep.get("mean_pre_lift_drag_m", 0.0),
            "max_pre_lift_drag_m": ep.get("max_pre_lift_drag_m", 0.0),
            "mean_post_release_drift_m": ep.get("mean_post_release_drift_m", 0.0),
            "max_post_release_drift_m": ep.get("max_post_release_drift_m", 0.0),
            "release_clearance_achieved_rate": ep.get(
                "release_clearance_achieved_rate",
                0.0,
            ),
            "post_release_recontact_rate": ep.get("post_release_recontact_rate", 0.0),
            "mean_grasp_offset_xy_m": ep.get("mean_grasp_offset_xy_m", 0.0),
        }
        for label in self._metric_cohorts:
            row[f"{label}_episode_count"] = ep.get(f"{label}_episode_count", 0)
            row[f"{label}_grasp_success_rate"] = ep.get(
                f"{label}_grasp_success_rate",
                0.0,
            )
            row[f"{label}_success_rate"] = ep.get(f"{label}_success_rate", 0.0)
        for key in self.reward_scales.keys():
            row["rew_" + key] = ep.get("rew_" + key, 0.0)

        if self._csv_writer is None:
            write_header = not os.path.exists(self.csv_log_path)
            self._csv_file = open(self.csv_log_path, "a", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(row.keys()))
            if write_header:
                self._csv_writer.writeheader()

        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def write_metrics_snapshot(self) -> None:
        self._write_csv_log()

    def training_state_dict(self) -> dict:
        """Small non-physics state required for a faithful iteration resume."""

        return {
            "schema_version": 1,
            "total_env_steps": int(self.total_env_steps),
            "curriculum_stage": int(self.curriculum_stage),
            "curriculum_stage_enter_step": int(getattr(self, "curriculum_stage_enter_step", 0)),
            "action_noise_schedule": {
                "start_std": float(self.train_action_noise_std),
                "end_std": float(self.train_action_noise_std_end),
                "anneal_steps": int(self.noise_anneal_steps),
                "clean_episode_frac": float(self.train_action_noise_clean_episode_frac),
            },
        }

    def load_training_state_dict(self, state: dict) -> None:
        """Restore schedule progress and reject a mislabeled cross-recipe resume."""

        if not isinstance(state, dict) or state.get("schema_version") != 1:
            raise ValueError("checkpoint has no supported environment training state")
        expected_schedule = self.training_state_dict()["action_noise_schedule"]
        if state.get("action_noise_schedule") != expected_schedule:
            raise ValueError(
                "resume action-noise schedule differs from the checkpoint; use an "
                "actor/critic warm start for a new training branch"
            )
        total_env_steps = int(state.get("total_env_steps", -1))
        curriculum_stage = int(state.get("curriculum_stage", -1))
        curriculum_stage_enter_step = int(state.get("curriculum_stage_enter_step", total_env_steps))
        if total_env_steps < 0:
            raise ValueError("checkpoint total_env_steps must be non-negative")
        if not 0 <= curriculum_stage <= self.curriculum_max_stage:
            raise ValueError("checkpoint curriculum_stage is outside the configured range")
        if not 0 <= curriculum_stage_enter_step <= total_env_steps:
            raise ValueError("checkpoint curriculum stage entry step is invalid")
        self.total_env_steps = total_env_steps
        self.curriculum_stage = curriculum_stage
        self.curriculum_stage_enter_step = curriculum_stage_enter_step

    def hold_home_step(self) -> None:
        """Pin the arm at default_ee_pos (gripper open) for scene preview — absolute, not Δ=0."""
        home_ee = torch.tensor(
            self.env_cfg["default_ee_pos"],
            device=self.device,
            dtype=gs.tc_float,
        ).expand(self.num_envs, 3)
        current_q = self.robot.get_dofs_position(self.arm_dof_idx)
        target_q = self._ik_arm_qpos(home_ee, init_arm_q=current_q)
        open_drive = torch.full(
            (self.num_envs, 1),
            float(self.default_gripper_drive.item()),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.robot.control_dofs_position(target_q, self.arm_dof_idx)
        self.robot.control_dofs_position(open_drive, self.gripper_dof_idx)
        self.scene.step()

    def _current_action_noise_std(self) -> float:
        """Annealed execution-noise std over batch steps (fixed when no curriculum set)."""
        return scheduled_action_noise_std(
            self.train_action_noise_std,
            self.train_action_noise_std_end,
            self.noise_anneal_steps,
            self.total_env_steps,
        )

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(
                f"Action shape {tuple(actions.shape)} does not match ({self.num_envs}, {self.num_actions})"
            )

        self.episode_length_buf += 1
        raw_actions = actions
        if not torch.isfinite(raw_actions).all():
            raise RuntimeError("policy emitted a non-finite action")
        if self.strict_action_bounds and torch.any(raw_actions.abs() > self.action_clip + 1e-6):
            observed = float(raw_actions.abs().amax().item())
            raise RuntimeError(
                f"policy action exceeded the bounded contract: max_abs={observed:.8f}, limit={self.action_clip:.8f}"
            )
        clipped_actions = torch.clamp(raw_actions, -self.action_clip, self.action_clip)
        policy_actions = clipped_actions
        # Execution noise (domain rand): keep PPO log-probs on the clean policy action,
        # but match evaluate.py's --action-noise-std robustness trial at train time.
        # The std anneals from train_action_noise_std -> train_action_noise_std_end over
        # noise_anneal_steps when a curriculum is configured (gradual robustness, no
        # catastrophic forgetting of the deterministic braking skill).
        noise_std = self._current_action_noise_std()
        if noise_std > 0.0:
            if self.noise_std_uniform_sample:
                # Magnitude varies per step up to the current ceiling.
                noise_std = float(noise_std) * float(torch.rand((), device=self.device))
            execution_noise = noise_std * torch.randn_like(clipped_actions)
            execution_noise = execution_noise * self.episode_action_noise_enabled.unsqueeze(-1)
            clipped_actions = torch.clamp(
                clipped_actions + execution_noise,
                -self.action_clip,
                self.action_clip,
            )
        execution_noise = clipped_actions - policy_actions
        self.total_env_steps += 1
        current_q = self.robot.get_dofs_position(self.arm_dof_idx)
        measured_ee_base = self.world_to_base(self.ik_link.get_pos())
        if self.ee_command_integration == "commanded":
            integration_base = leashed_ee_setpoint(
                self.ee_setpoint_base,
                measured_ee_base,
                self.ee_setpoint_leash_m,
            )
        else:
            integration_base = measured_ee_base
        target_ee_base, cart_delta_unclipped = arm_target_ee_pos(
            integration_base,
            clipped_actions[:, :3],
            self.action_scale,
            self.max_cartesian_delta_m,
            self.action_response_exponent,
        )
        self.ee_setpoint_base = target_ee_base.detach()
        target_q = self._ik_arm_qpos(target_ee_base, init_arm_q=current_q)

        # Integrate the COMMANDED gap (not the measured gap): driving to a command that can
        # reach close_gap lets the gripper apply full PD force to squeeze a rigid cube.
        # Referencing the measured gap would cap the target at ~measured-4mm and stall on contact.
        self.commanded_gap = gripper_target_gap(
            self.commanded_gap,
            clipped_actions[:, 3],
            self.gripper_delta_m,
            self.gripper_min_command_gap_m,
            self.gripper_open_gap_m,
        )
        target_gap = self.commanded_gap
        measured_gap = self.gripper_gap_m()
        left_force, right_force = self.finger_contact_forces_n()
        release_intent = (clipped_actions[:, 3] > self.release_action_threshold) & (
            self.commanded_gap > self.gripper_min_command_gap_m + self.release_command_margin_m
        )
        closing = (self.commanded_gap < self.gripper_open_gap_m - 1e-9) & (~release_intent)
        applied_gap = self.gripper_contact_hold.update(
            requested_gap_m=self.commanded_gap,
            measured_gap_m=measured_gap,
            left_force_n=left_force,
            right_force_n=right_force,
            closing=closing,
            release=release_intent,
        )
        target_drive = self.gap_m_to_drive_t(applied_gap)

        self._accumulate_action_stats(raw_actions, cart_delta_unclipped, target_gap)
        self.previous_action.copy_(clipped_actions.detach())
        self.robot.control_dofs_position(target_q, self.arm_dof_idx)
        self.robot.control_dofs_position(target_drive.unsqueeze(-1), self.gripper_dof_idx)
        try:
            self.scene.step()
        except Exception as exc:
            raise RuntimeError(f"{self.physics_profile} failed during batch step {self.total_env_steps}") from exc
        self._assert_finite_simulation_state()
        current_finger_center = self.finger_center_base()
        self.ee_velocity_base = (current_finger_center - self.prev_finger_center_base) / self.ctrl_dt
        self.prev_finger_center_base = current_finger_center.detach().clone()

        # Snapshot all one-shot quality latches before the current transition.
        self.prev_ever_grasped = self.ever_grasped.clone()
        self.prev_release_started = self.release_started.clone()
        self.prev_near_table_entered = self.near_table_entered.clone()
        self.prev_hard_landing_violation = self.hard_landing_violation.clone()
        self.prev_max_pre_lift_xy_m = self.max_pre_lift_xy_m.clone()
        self._update_task_state()
        self._update_post_release_clearance_state()

        timeout_buf = self.episode_length_buf > self.max_episode_length
        done_buf = timeout_buf | self.episode_success
        self.reset_buf = done_buf
        self.extras["time_outs"] = timeout_buf.float()

        reward = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        reward_terms: dict[str, torch.Tensor] = {}
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew
            reward_terms[name] = rew.detach().clone()

        self.prev_obj_z = self.obj_pos_base()[:, 2].detach()
        self.prev_grasp_dist = self.grasp_dist().detach()
        self.prev_gripper_gap = self.gripper_gap_m().detach()
        self.prev_target_xy_dist_m = torch.norm(
            self.obj_pos_base()[:, :2] - self.target_pos[:, :2],
            dim=-1,
        ).detach()
        self.prev_obj_speed_m_s = torch.norm(self.obj.get_vel(), dim=-1).detach()
        self.prev_post_release_clearance_m = self.post_release_clearance_m.detach().clone()

        self.extras["episode_grasp_success"] = self.ever_grasped.clone()
        self.extras["episode_lift_success"] = self.ever_lifted.clone()
        self.extras["episode_place_success"] = self.episode_place_success.clone()
        self.extras["episode_success"] = self.episode_success.clone()
        self.extras["episode_scenario_id"] = self.scenario_id.clone()
        self.extras["episode_initial_obj_pos"] = self.initial_obj_pos.clone()
        self.extras["episode_target_pos"] = self.target_pos.clone()
        episode_steps = self.episode_length_buf.float().clamp(min=1.0)
        self.extras["episode_action_clip_fraction"] = self.episode_action_clip_sum / episode_steps
        self.extras["episode_action_near_bound_fraction"] = self.episode_action_near_bound_sum / episode_steps
        self.extras["episode_ik_failure_fraction"] = self.episode_ik_failure_sum / episode_steps
        self.extras["episode_ik_jump_reject_fraction"] = self.episode_ik_jump_reject_sum / episode_steps
        self.extras["episode_final_obj_pos"] = self.obj_pos_base().clone()
        self.extras["episode_final_obj_vel"] = self.obj.get_vel().clone()
        self.extras["episode_final_gripper_gap"] = self.gripper_gap_m().clone()
        self.extras["episode_max_pre_lift_xy_m"] = self.max_pre_lift_xy_m.clone()
        self.extras["episode_release_started"] = self.release_started.clone()
        self.extras["episode_release_valid"] = self.release_valid.clone()
        self.extras["episode_release_violation"] = self.release_violation.clone()
        self.extras["episode_release_xy_dist_m"] = self.release_xy_dist_m.clone()
        self.extras["episode_release_height_error_m"] = self.release_height_error_m.clone()
        self.extras["episode_release_speed_m_s"] = self.release_speed_m_s.clone()
        self.extras["episode_max_landing_xy_speed_m_s"] = self.max_landing_xy_speed_m_s.clone()
        self.extras["episode_max_landing_down_speed_m_s"] = self.max_landing_down_speed_m_s.clone()
        self.extras["episode_hard_landing_violation"] = self.hard_landing_violation.clone()
        self.extras["episode_post_release_drift_m"] = self.post_release_drift_m.clone()
        self.extras["episode_post_release_clearance_m"] = self.post_release_clearance_m.clone()
        self.extras["episode_post_release_recontact"] = self.post_release_recontact.clone()
        self.extras["episode_quality_ok"] = self.quality_ok.clone()
        self.extras["reward_terms"] = reward_terms
        self.extras["step_snapshot"] = self._build_step_snapshot(
            clipped_actions,
            policy_actions,
            execution_noise,
            reward,
            done_buf,
            reward_terms,
        )
        self.extras["event_frame_rgb"] = None
        if self.capture_cam is not None and self.env_cfg.get("capture_event_frames", False):
            snapshot = self.extras["step_snapshot"]
            event_now = any(
                bool(snapshot[name][0].item())
                for name in (
                    "first_push_event",
                    "first_lift_event",
                    "release_event",
                    "table_contact_event",
                    "final_stable_event",
                )
            )
            if event_now:
                rgb, *_ = self.capture_cam.render(rgb=True)
                frame = np.asarray(rgb)
                self.extras["event_frame_rgb"] = frame[0].copy() if frame.ndim == 4 else frame.copy()

        done_idx = done_buf.nonzero(as_tuple=True)[0]
        if len(done_idx) > 0:
            self.reset_idx(done_idx)

        obs = self.get_observations()
        return obs, reward, done_buf, self.extras

    def _build_step_snapshot(
        self,
        actions: torch.Tensor,
        policy_actions: torch.Tensor,
        action_noise: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        reward_terms: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Freeze post-transition data before terminal environments are reset."""

        obj_base = self.obj_pos_base()
        return {
            "ee_base": self.finger_center_base().detach().clone(),
            "ik_ee_base": self.world_to_base(self.ik_link.get_pos()).detach().clone(),
            "ee_setpoint_base": self.ee_setpoint_base.detach().clone(),
            "obj_base": obj_base.detach().clone(),
            "obj_vel": self.obj.get_vel().detach().clone(),
            "ee_velocity_base": self.ee_velocity_base.detach().clone(),
            "target_pos": self.target_pos.detach().clone(),
            "initial_obj_pos": self.initial_obj_pos.detach().clone(),
            "scenario_id": self.scenario_id.detach().clone(),
            "grasp_pos": self.desired_grasp_pos_base().detach().clone(),
            "gap_m": self.gripper_gap_m().detach().clone(),
            "commanded_gap_m": self.commanded_gap.detach().clone(),
            "applied_gripper_gap_m": self.gripper_contact_hold.applied_gap_m.detach().clone(),
            "gripper_force_hold_latched": self.gripper_contact_hold.latched.detach().clone(),
            "actions": actions.detach().clone(),
            "policy_actions": policy_actions.detach().clone(),
            "action_noise": action_noise.detach().clone(),
            "left_contact_force_n": self.left_contact_force_n.detach().clone(),
            "right_contact_force_n": self.right_contact_force_n.detach().clone(),
            "holding": self.holding.detach().clone(),
            "carry": self.carry.detach().clone(),
            "ever_grasped": self.ever_grasped.detach().clone(),
            "ever_carried_near": self.ever_carried_near.detach().clone(),
            "release_started": self.release_started.detach().clone(),
            "release_valid": self.release_valid.detach().clone(),
            "release_violation": self.release_violation.detach().clone(),
            "near_table_entered": self.near_table_entered.detach().clone(),
            "hard_landing_violation": self.hard_landing_violation.detach().clone(),
            "hard_landing_event": (self.hard_landing_violation & (~self.prev_hard_landing_violation)).detach().clone(),
            "quality_ok": self.quality_ok.detach().clone(),
            "max_pre_lift_xy_m": self.max_pre_lift_xy_m.detach().clone(),
            "release_xy_dist_m": self.release_xy_dist_m.detach().clone(),
            "release_height_error_m": self.release_height_error_m.detach().clone(),
            "release_speed_m_s": self.release_speed_m_s.detach().clone(),
            "max_landing_xy_speed_m_s": self.max_landing_xy_speed_m_s.detach().clone(),
            "max_landing_down_speed_m_s": self.max_landing_down_speed_m_s.detach().clone(),
            "post_release_drift_m": self.post_release_drift_m.detach().clone(),
            "post_release_clearance_m": self.post_release_clearance_m.detach().clone(),
            "release_clearance_achieved": self.release_clearance_achieved.detach().clone(),
            "post_release_recontact_event": self.post_release_recontact_event.detach().clone(),
            "post_release_recontact": self.post_release_recontact.detach().clone(),
            "first_push_event": ((self.prev_max_pre_lift_xy_m <= 0.001) & (self.max_pre_lift_xy_m > 0.001))
            .detach()
            .clone(),
            "first_lift_event": (self.carry & (~self.prev_ever_grasped)).detach().clone(),
            "release_event": (self.release_started & (~self.prev_release_started)).detach().clone(),
            "table_contact_event": (self.near_table_entered & (~self.prev_near_table_entered)).detach().clone(),
            "final_stable_event": (self.episode_success & done).detach().clone(),
            "reward": reward.detach().clone(),
            "done": done.detach().clone(),
            "reward_terms": reward_terms,
        }

    def _ik_arm_qpos(
        self,
        target_ee_base: torch.Tensor,
        *,
        init_arm_q: torch.Tensor,
        enforce_jump_limit: bool = True,
    ) -> torch.Tensor:
        """Solve batched link6 IK with fixed gripper-down orientation; fall back on failure/jump."""
        target_ee_world = self.base_to_world(target_ee_base)
        x, y, z, w = GRIPPER_DOWN_QUAT_XYZW
        down_quat = torch.tensor([[w, x, y, z]], device=self.device, dtype=gs.tc_float).expand(self.num_envs, 4)
        try:
            full_q = self.robot.get_qpos().detach().clone()
            full_q[:, self.arm_dof_idx] = init_arm_q
            sol = self.robot.inverse_kinematics(
                link=self.ik_link,
                pos=target_ee_world,
                quat=down_quat,
                dofs_idx_local=self.arm_dof_idx,
                init_qpos=full_q,
                damping=self._ik_damping,
            )
            target_q = sol[:, self.arm_dof_idx]
        except Exception:
            if hasattr(self, "episode_ik_failure_sum"):
                self.episode_ik_failure_sum += 1.0
            return init_arm_q

        if not torch.isfinite(target_q).all():
            bad = ~torch.isfinite(target_q).all(dim=-1)
            self.episode_ik_failure_sum += bad.float()
            target_q = torch.where(bad.unsqueeze(-1), init_arm_q, target_q)

        if enforce_jump_limit:
            jump = (target_q - init_arm_q).abs().amax(dim=-1)
            too_far = jump > self.max_ik_jump_rad
            if too_far.any():
                self.episode_ik_jump_reject_sum += too_far.float()
                target_q = torch.where(too_far.unsqueeze(-1), init_arm_q, target_q)
        return target_q

    def _accumulate_action_stats(
        self,
        raw_actions: torch.Tensor,
        cart_delta_unclipped: torch.Tensor,
        target_gap: torch.Tensor,
    ) -> None:
        eps = 1e-6
        action_sat = (raw_actions.abs() >= self.action_clip - eps).float().mean(dim=-1)
        action_near_bound = (raw_actions.abs() >= 0.95 * self.action_clip).float().mean(dim=-1)
        action_clip = (raw_actions.abs() > self.action_clip + eps).float().mean(dim=-1)
        delta_sat = (cart_delta_unclipped.abs() >= self.max_cartesian_delta_m - eps).float().mean(dim=-1)
        gripper_bound = (
            (target_gap <= self.gripper_min_command_gap_m + eps) | (target_gap >= self.gripper_open_gap_m - eps)
        ).float()
        self.episode_action_sat_sum += action_sat
        self.episode_action_near_bound_sum += action_near_bound
        self.episode_action_clip_sum += action_clip
        self.episode_delta_sat_sum += delta_sat
        self.episode_gripper_bound_sum += gripper_bound

    def _update_task_state(self) -> None:
        obj_base = self.obj_pos_base()
        # Detection point = the grasp point between the pads: finger-center lowered by the
        # grasp offset. finger-center (mount-frame midpoint) sits ~grasp_center_offset_z
        # ABOVE the cube even in a perfect grasp, so |finger_center - obj| was borderline
        # (~0.065 vs the 0.07 threshold) and dropped below "holding" during the lift. The
        # lowered grasp point coincides with the cube (|.-obj|~0), making holding robust.
        grasp_point_offset = torch.zeros(3, device=self.device, dtype=gs.tc_float)
        grasp_point_offset[2] = self.grasp_center_offset_z
        ee_base = self.finger_center_base() - grasp_point_offset
        gap = self.gripper_gap_m()
        obj_vel = self.obj.get_vel()
        state = update_task_state(
            obj_base,
            ee_base,
            gap,
            obj_vel,
            self.target_pos,
            ever_grasped=self.ever_grasped,
            ever_lifted=self.ever_lifted,
            ever_carried_near=self.ever_carried_near,
            episode_place_success=self.episode_place_success,
            episode_success=self.episode_success,
            place_stable_steps=self.place_stable_steps,
            obj_rest_z_base=self.obj_rest_z_base,
            object_width=self.obj_size[1],
            place_success_dist_m=self.place_success_dist_m,
            success_hold_steps=self.success_hold_steps,
            success_table_z_tolerance_m=self.success_table_z_tolerance_m,
            success_max_obj_speed_m_s=self.success_max_obj_speed_m_s,
            carry_near_dist_m=self.carry_near_dist_m,
            initial_obj_pos=self.initial_obj_pos,
            commanded_gap=self.commanded_gap,
            action_gripper=self.previous_action[:, 3],
            release_started=self.release_started,
            release_valid=self.release_valid,
            release_violation=self.release_violation,
            near_table_entered=self.near_table_entered,
            hard_landing_violation=self.hard_landing_violation,
            max_pre_lift_xy_m=self.max_pre_lift_xy_m,
            max_landing_xy_speed_m_s=self.max_landing_xy_speed_m_s,
            max_landing_down_speed_m_s=self.max_landing_down_speed_m_s,
            release_xy_dist_m=self.release_xy_dist_m,
            release_height_error_m=self.release_height_error_m,
            release_speed_m_s=self.release_speed_m_s,
            release_obj_xy=self.release_obj_xy,
            post_release_drift_m=self.post_release_drift_m,
            grasp_offset_xy_m=self.grasp_offset_xy_m,
            release_success_dist_m=self.release_success_dist_m,
            release_height_tolerance_m=self.release_height_tolerance_m,
            release_max_obj_speed_m_s=self.release_max_obj_speed_m_s,
            pre_lift_max_drag_m=self.pre_lift_max_drag_m,
            post_release_max_drift_m=self.post_release_max_drift_m,
            landing_near_table_height_m=self.landing_near_table_height_m,
            landing_max_xy_speed_m_s=self.landing_max_xy_speed_m_s,
            landing_max_down_speed_m_s=self.landing_max_down_speed_m_s,
            release_action_threshold=self.release_action_threshold,
            release_command_margin_m=self.release_command_margin_m,
            gripper_close_gap_m=self.gripper_min_command_gap_m,
            bilateral_contact=self.bilateral_contact(),
        )
        self.holding = state.holding
        self.carry = state.carry
        self.grasped = state.grasped
        self.ever_lifted = state.ever_lifted
        self.ever_grasped = state.ever_grasped
        self.ever_carried_near = state.ever_carried_near
        self.place_stable_steps = state.place_stable_steps
        self.episode_place_success = state.episode_place_success
        self.episode_success = state.episode_success
        self.release_started = state.release_started
        self.release_valid = state.release_valid
        self.release_violation = state.release_violation
        self.near_table_entered = state.near_table_entered
        self.hard_landing_violation = state.hard_landing_violation
        self.max_pre_lift_xy_m = state.max_pre_lift_xy_m
        self.max_landing_xy_speed_m_s = state.max_landing_xy_speed_m_s
        self.max_landing_down_speed_m_s = state.max_landing_down_speed_m_s
        self.release_xy_dist_m = state.release_xy_dist_m
        self.release_height_error_m = state.release_height_error_m
        self.release_speed_m_s = state.release_speed_m_s
        self.release_obj_xy = state.release_obj_xy
        self.post_release_drift_m = state.post_release_drift_m
        self.grasp_offset_xy_m = state.grasp_offset_xy_m
        self.quality_ok = state.quality_ok

    def _update_post_release_clearance_state(self) -> None:
        """Track true pad clearance and the first later object contact."""

        grasp_point = self.finger_center_base().clone()
        grasp_point[:, 2] -= self.grasp_center_offset_z
        obj_base = self.obj_pos_base()
        self.post_release_clearance_m = torch.clamp(
            grasp_point[:, 2] - obj_base[:, 2],
            min=0.0,
            max=self.post_release_clearance_max_m,
        )
        left_force, right_force = self.finger_contact_forces_n()
        self.left_contact_force_n = left_force.detach().clone()
        self.right_contact_force_n = right_force.detach().clone()
        left_contact = left_force > self.contact_force_threshold_n
        right_contact = right_force > self.contact_force_threshold_n
        any_contact = left_contact | right_contact
        gap_clear = self.gripper_gap_m() >= (self.obj_size[1] + self.post_release_clearance_gap_margin_m)
        cleared_now = self.release_valid & gap_clear & (~any_contact)
        self.post_release_recontact_event = (
            self.release_clearance_achieved & any_contact & (~self.post_release_recontact)
        )
        self.post_release_recontact |= self.post_release_recontact_event
        self.release_clearance_achieved |= cleared_now

    def drive_to_gap_m_t(self, drive: torch.Tensor) -> torch.Tensor:
        return drive_to_gap_m(drive, self.gripper_open_gap_m)

    def gap_m_to_drive_t(self, gap_m: torch.Tensor) -> torch.Tensor:
        return gap_m_to_drive(gap_m, self.gripper_min_command_gap_m, self.gripper_open_gap_m)

    def gripper_gap_m(self) -> torch.Tensor:
        drive = self.robot.get_dofs_position(self.gripper_dof_idx).squeeze(-1)
        return self.drive_to_gap_m_t(drive)

    def finger_center_world(self) -> torch.Tensor:
        return (self.left_finger_link.get_pos() + self.right_finger_link.get_pos()) / 2.0

    def finger_span_m(self) -> torch.Tensor:
        """Distance between the two finger link frames (true pad separation proxy).

        Unlike the nominal gap, this comes from the solved link poses, so it reflects
        where the pads physically are once they stall against the cube.
        """
        return torch.norm(
            self.left_finger_link.get_pos() - self.right_finger_link.get_pos(),
            dim=-1,
        )

    def finger_contact_forces_n(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Object-only contact-force magnitude carried by each fingertip link."""

        contacts = self.obj.get_contacts(with_entity=self.robot, exclude_self_contact=True)
        left, right = object_finger_contact_forces_n(
            contacts,
            left_link_idx=int(self.left_finger_link.idx),
            right_link_idx=int(self.right_finger_link.idx),
        )
        return (
            left.to(device=self.device, dtype=gs.tc_float),
            right.to(device=self.device, dtype=gs.tc_float),
        )

    def _assert_finite_simulation_state(self) -> None:
        """Stop at the first corrupted environment instead of propagating it."""

        left_force, right_force = self.finger_contact_forces_n()
        fields = {
            "robot_qpos": self.robot.get_dofs_position(),
            "robot_qvel": self.robot.get_dofs_velocity(),
            "object_pos": self.obj.get_pos(),
            "object_vel": self.obj.get_vel(),
            "left_object_contact_force": left_force,
            "right_object_contact_force": right_force,
        }
        for name, value in fields.items():
            finite = torch.isfinite(value)
            if bool(finite.all().item()):
                continue
            per_env = finite.reshape(self.num_envs, -1).all(dim=1)
            first_bad = int(torch.nonzero(~per_env, as_tuple=False)[0, 0].item())
            raise RuntimeError(
                f"{self.physics_profile} produced NaN or Inf in {name} "
                f"at batch step {self.total_env_steps}, env {first_bad}"
            )

    def bilateral_contact(self) -> torch.Tensor:
        """True where both pads carry object-contact load."""
        left, right = self.finger_contact_forces_n()
        return torch.minimum(left, right) > self.contact_force_threshold_n

    def finger_center_base(self) -> torch.Tensor:
        return self.world_to_base(self.finger_center_world())

    def obj_pos_base(self) -> torch.Tensor:
        return self.world_to_base(self.obj.get_pos())

    def desired_grasp_pos_base(self) -> torch.Tensor:
        return logic_desired_grasp_pos_base(self.obj_pos_base(), self.grasp_center_offset_z)

    def grasp_dist(self) -> torch.Tensor:
        """Euclidean distance from finger center to the desired grasp pose (base frame)."""
        return torch.norm(self.finger_center_base() - self.desired_grasp_pos_base(), dim=-1)

    def get_observations(self) -> TensorDict:
        joint_pos = self.robot.get_dofs_position(self.arm_dof_idx)
        joint_vel = self.robot.get_dofs_velocity(self.arm_dof_idx)
        ee_base = self.finger_center_base()
        gripper_gap = self.gripper_gap_m()
        obj_base = self.obj_pos_base()
        layout_offsets = None
        if self.include_normalized_layout_offsets:
            layout_offsets = normalized_pick_place_layout_offsets(
                self.episode_layout_obj_pos,
                self.episode_layout_target_pos,
                self.fixed_obj_pos,
                self.fixed_target_pos,
                self.obj_spawn_lower,
                self.obj_spawn_upper,
                self.target_spawn_lower,
                self.target_spawn_upper,
            )
        scripted_action_hint = None
        if self.include_scripted_action_hint:
            if self._scripted_action_guide is None:
                raise RuntimeError("scripted action hint is enabled without a guide")
            if self._scripted_action_hint_step != self.total_env_steps:
                with torch.no_grad():
                    self._scripted_action_hint.copy_(self._scripted_action_guide())
                self._scripted_action_hint_step = self.total_env_steps
            scripted_action_hint = self._scripted_action_hint
        quality_features = None
        if self.include_quality_observations:
            quality_features = normalized_pick_place_quality_features(
                self.obj.get_vel(),
                self.holding,
                self.ever_carried_near,
                self.release_started,
                velocity_scale_m_s=self.quality_velocity_obs_scale_m_s,
            )
        contact_features = None
        if self.include_contact_observations:
            left_force, right_force = self.finger_contact_forces_n()
            contact_features = normalized_pick_place_contact_features(
                self.finger_span_m(),
                left_force,
                right_force,
                object_width_m=self.obj_size[1],
                contact_force_scale_n=self.contact_force_scale_n,
                contact_force_threshold_n=self.contact_force_threshold_n,
            )
        setpoint_residual = None
        if self.include_ee_setpoint_residual:
            setpoint_residual = normalized_ee_setpoint_residual(
                self.ee_setpoint_base,
                self.world_to_base(self.ik_link.get_pos()),
                self.ee_setpoint_leash_m,
            )
        obs = pick_place_observation(
            joint_pos,
            joint_vel,
            ee_base,
            gripper_gap,
            obj_base,
            self.target_pos,
            self.grasped,
            self.ever_grasped,
            self.commanded_gap if self.include_commanded_gap else None,
            self.previous_action if self.include_previous_action else None,
            normalized_layout_offsets=layout_offsets,
            quality_features=quality_features,
            contact_features=contact_features,
            normalized_setpoint_residual=setpoint_residual,
            scripted_action_hint=scripted_action_hint,
        )
        if obs.shape[-1] != self.num_obs:
            raise RuntimeError(f"Observation shape {obs.shape[-1]} does not match num_obs={self.num_obs}")
        groups = {"policy": obs}
        if self.privileged_critic_obs:
            privileged = torch.cat([self.obj.get_vel(), self.obj.get_ang()], dim=-1)
            groups["privileged"] = privileged
            self.extras["observations"] = {"policy": obs, "privileged": privileged}
        else:
            self.extras["observations"] = {"policy": obs}
        return TensorDict(groups, batch_size=[self.num_envs])

    def get_privileged_observations(self) -> None:
        return None

    def _reward_reach(self) -> torch.Tensor:
        return reward_reach(self.finger_center_base(), self.desired_grasp_pos_base(), self.carry)

    def _reward_approach_potential(self) -> torch.Tensor:
        return reward_approach_potential(self.prev_grasp_dist, self.grasp_dist(), self.carry)

    def _reward_align(self) -> torch.Tensor:
        return reward_align(self.finger_center_base(), self.desired_grasp_pos_base(), self.carry)

    def _reward_keypoints(self) -> torch.Tensor:
        return reward_keypoints(self.finger_center_base(), self.desired_grasp_pos_base(), self.carry)

    def _reward_close_gripper(self) -> torch.Tensor:
        return reward_close_gripper(
            self.finger_center_base(),
            self.desired_grasp_pos_base(),
            self.gripper_gap_m(),
            self.gripper_open_gap_m,
            self.carry,
            self.gripper_close_gap_m,
        )

    def _reward_grasp_ready_closing(self) -> torch.Tensor:
        return reward_grasp_ready_closing(
            self.previous_action[:, 3],
            self.grasp_dist(),
            self.gripper_gap_m(),
            self.commanded_gap,
            self.holding,
            self.ever_grasped,
            open_gap_m=self.gripper_open_gap_m,
            contact_gap_m=self.obj_size[1] + 0.02,
        )

    def _reward_grasp_settle_action(self) -> torch.Tensor:
        return reward_grasp_settle_action(
            self.previous_action[:, :3],
            self.grasp_dist(),
            self.gripper_gap_m(),
            self.commanded_gap,
            self.ever_grasped,
            open_gap_m=self.gripper_open_gap_m,
            contact_gap_m=self.obj_size[1] + 0.001,
        )

    def _reward_grasp_gap_progress(self) -> torch.Tensor:
        return reward_grasp_gap_progress(
            self.prev_gripper_gap,
            self.gripper_gap_m(),
            self.ever_grasped,
            contact_gap_m=self.obj_size[1] + 0.001,
        )

    def _reward_grasp_lift_action(self) -> torch.Tensor:
        return reward_grasp_lift_action(
            self.previous_action[:, 2],
            self.gripper_gap_m(),
            self.holding,
            self.ever_grasped,
            self.max_pre_lift_xy_m,
            contact_gap_m=self.obj_size[1] + 0.001,
            max_drag_m=self.pre_lift_max_drag_m,
        )

    def _reward_grasp_lift_progress(self) -> torch.Tensor:
        return reward_grasp_lift_progress(
            self.prev_obj_z,
            self.obj_pos_base()[:, 2],
            self.gripper_gap_m(),
            self.holding,
            self.prev_ever_grasped,
            self.max_pre_lift_xy_m,
            contact_gap_m=self.obj_size[1] + 0.001,
            max_drag_m=self.pre_lift_max_drag_m,
        )

    def _reward_lift(self) -> torch.Tensor:
        obj = self.obj_pos_base()
        return reward_lift(
            obj[:, 2],
            self.obj_rest_z_base,
            self.lift_height_m,
            self.carry,
            obj,
            self.target_pos,
            self.place_shaping_dist_m,
        )

    def _reward_grasp(self) -> torch.Tensor:
        return reward_grasp(
            self.carry,
            self.obj_pos_base(),
            self.target_pos,
            self.place_shaping_dist_m,
        )

    def _reward_grasp_bonus(self) -> torch.Tensor:
        return reward_grasp_bonus(self.carry, self.prev_ever_grasped)

    def _pre_release_shaping(self, reward: torch.Tensor) -> torch.Tensor:
        if not self.quality_pre_release_shaping_only:
            return reward
        active = self.holding & (~self.release_started)
        return reward * active.float()

    def _reward_place_xy(self) -> torch.Tensor:
        return self._pre_release_shaping(
            reward_place_xy(
                self.obj_pos_base(),
                self.target_pos,
                self.carry,
                self.ever_grasped,
                self.place_shaping_dist_m,
                ever_carried_near=self.ever_carried_near,
            )
        )

    def _reward_transport_progress(self) -> torch.Tensor:
        current_xy_dist = torch.norm(
            self.obj_pos_base()[:, :2] - self.target_pos[:, :2],
            dim=-1,
        )
        return reward_transport_progress(
            self.prev_target_xy_dist_m,
            current_xy_dist,
            self.ever_grasped,
            self.holding,
            self.prev_release_started,
            normalization_m=0.10,
        )

    def _reward_place_z(self) -> torch.Tensor:
        return self._pre_release_shaping(
            reward_place_z(
                self.obj_pos_base(),
                self.target_pos,
                self.ever_grasped,
                self.obj_rest_z_base,
                self.place_shaping_dist_m,
                near_factor=self.place_lower_near_factor,
                ever_carried_near=self.ever_carried_near,
            )
        )

    def _reward_lower(self) -> torch.Tensor:
        obj = self.obj_pos_base()
        return reward_lower(
            obj[:, 2],
            self.prev_obj_z,
            obj,
            self.target_pos,
            self.carry,
            self.place_shaping_dist_m,
            near_factor=self.place_lower_near_factor,
        )

    def _reward_descent_progress(self) -> torch.Tensor:
        return reward_descent_progress(
            self.prev_obj_z,
            self.obj_pos_base()[:, 2],
            self.ever_carried_near,
            self.holding,
            self.prev_release_started,
            normalization_m=0.04,
        )

    def _reward_holding_table(self) -> torch.Tensor:
        return self._pre_release_shaping(
            reward_holding_table(
                self.holding,
                self.carry,
                self.obj_pos_base(),
                self.target_pos,
                self.ever_grasped,
                self.obj_rest_z_base,
                self.place_shaping_dist_m,
                near_factor=self.place_lower_near_factor,
                ever_carried_near=self.ever_carried_near,
            )
        )

    def _reward_drop_far(self) -> torch.Tensor:
        return reward_drop_far(
            self.obj_pos_base(),
            self.target_pos,
            self.carry,
            self.ever_grasped,
            self.place_shaping_dist_m,
            obj_rest_z_base=self.obj_rest_z_base,
        )

    def _reward_release(self) -> torch.Tensor:
        return reward_release(
            self.obj_pos_base(),
            self.target_pos,
            self.gripper_gap_m(),
            self.gripper_open_gap_m,
            self.ever_grasped,
            self.obj_rest_z_base,
            self.release_success_dist_m,
            release_height_m=self.release_height_m,
            ever_carried_near=self.ever_carried_near,
            obj_vel=self.obj.get_vel(),
            release_speed_k=self.release_speed_k,
        )

    def _reward_throw_release(self) -> torch.Tensor:
        return reward_throw_release(
            self.obj.get_vel(),
            self.gripper_gap_m(),
            self.ever_carried_near,
            self.obj_size[1],
            throw_speed_m_s=self.throw_speed_m_s,
        )

    def _reward_push_before_grasp(self) -> torch.Tensor:
        grasp_point_offset = torch.zeros(3, device=self.device, dtype=gs.tc_float)
        grasp_point_offset[2] = self.grasp_center_offset_z
        ee_base = self.finger_center_base() - grasp_point_offset
        return reward_push_before_grasp(
            self.obj_pos_base(),
            self.initial_obj_pos,
            ee_base,
            self.gripper_gap_m(),
            self.ever_grasped,
            self.obj_size[1],
            push_dist_m=self.push_before_grasp_dist_m,
        )

    def _reward_push_after_release(self) -> torch.Tensor:
        # Detection point matches holding/carry: finger-center lowered by grasp offset.
        grasp_point_offset = torch.zeros(3, device=self.device, dtype=gs.tc_float)
        grasp_point_offset[2] = self.grasp_center_offset_z
        ee_base = self.finger_center_base() - grasp_point_offset
        return reward_push_after_release(
            ee_base,
            self.obj_pos_base(),
            self.gripper_gap_m(),
            self.ever_carried_near,
            self.carry,
            self.obj_rest_z_base,
            self.obj_size[1],
        )

    def _reward_pre_lift_xy_progress(self) -> torch.Tensor:
        return reward_pre_lift_xy_progress(
            self.prev_max_pre_lift_xy_m,
            self.max_pre_lift_xy_m,
            normalization_m=0.01,
        )

    def _reward_clean_lift_bonus(self) -> torch.Tensor:
        return reward_clean_lift_bonus(
            self.carry,
            self.prev_ever_grasped,
            self.max_pre_lift_xy_m,
            max_drag_m=self.pre_lift_max_drag_m,
        )

    def _reward_clean_lift_quality(self) -> torch.Tensor:
        return reward_clean_lift_quality(
            self.carry,
            self.prev_ever_grasped,
            self.max_pre_lift_xy_m,
            distance_scale_m=self.clean_lift_reward_scale_m,
        )

    def _reward_grasp_centering(self) -> torch.Tensor:
        return reward_grasp_centering(
            self.carry,
            self.prev_ever_grasped,
            self.grasp_offset_xy_m,
            distance_scale_m=self.grasp_centering_reward_scale_m,
        )

    def _reward_valid_release(self) -> torch.Tensor:
        return reward_valid_release(
            self.release_started,
            self.prev_release_started,
            self.release_valid,
        )

    def _reward_invalid_release(self) -> torch.Tensor:
        return reward_invalid_release(
            self.release_started,
            self.prev_release_started,
            self.release_violation,
        )

    def _reward_release_quality(self) -> torch.Tensor:
        return reward_release_quality(
            self.release_started,
            self.prev_release_started,
            self.release_xy_dist_m,
            self.release_height_error_m,
            self.release_speed_m_s,
            xy_scale_m=self.release_quality_xy_scale_m,
            height_scale_m=self.release_quality_height_scale_m,
            speed_scale_m_s=self.release_quality_speed_scale_m_s,
        )

    def _reward_release_readiness_progress(self) -> torch.Tensor:
        obj = self.obj_pos_base()
        obj_vel = self.obj.get_vel()
        return reward_release_readiness_progress(
            self.prev_target_xy_dist_m,
            torch.abs(self.prev_obj_z - self.obj_rest_z_base),
            self.prev_obj_speed_m_s,
            torch.norm(obj[:, :2] - self.target_pos[:, :2], dim=-1),
            torch.abs(obj[:, 2] - self.obj_rest_z_base),
            torch.norm(obj_vel, dim=-1),
            self.ever_carried_near,
            self.prev_release_started,
            xy_scale_m=self.release_quality_xy_scale_m,
            height_scale_m=self.release_quality_height_scale_m,
            speed_scale_m_s=self.release_quality_speed_scale_m_s,
        )

    def _reward_premature_opening(self) -> torch.Tensor:
        obj = self.obj_pos_base()
        obj_vel = self.obj.get_vel()
        release_ready = (
            (torch.norm(obj[:, :2] - self.target_pos[:, :2], dim=-1) <= self.release_success_dist_m)
            & (torch.abs(obj[:, 2] - self.obj_rest_z_base) <= self.release_height_tolerance_m)
            & (torch.norm(obj_vel, dim=-1) <= self.release_max_obj_speed_m_s)
            & self.ever_carried_near
        )
        return reward_premature_opening(
            self.previous_action[:, 3],
            self.ever_grasped,
            self.prev_release_started,
            release_ready,
        )

    def _reward_ready_opening(self) -> torch.Tensor:
        obj = self.obj_pos_base()
        obj_vel = self.obj.get_vel()
        release_ready = (
            (torch.norm(obj[:, :2] - self.target_pos[:, :2], dim=-1) <= self.release_success_dist_m)
            & (torch.abs(obj[:, 2] - self.obj_rest_z_base) <= self.release_height_tolerance_m)
            & (torch.norm(obj_vel, dim=-1) <= self.release_max_obj_speed_m_s)
            & self.ever_carried_near
        )
        return reward_ready_opening(
            self.previous_action[:, 3],
            release_ready,
            self.prev_release_started,
        )

    def _reward_release_clearance_opening(self) -> torch.Tensor:
        return reward_release_clearance_opening(
            self.previous_action[:, 3],
            self.commanded_gap,
            self.release_valid,
            target_commanded_gap_m=self.obj_size[1] + 0.025,
        )

    def _reward_setdown_action(self) -> torch.Tensor:
        return reward_setdown_action(
            self.previous_action[:, 2],
            torch.abs(self.obj_pos_base()[:, 2] - self.obj_rest_z_base),
            self.ever_carried_near,
            self.holding,
            self.prev_release_started,
            release_height_tolerance_m=self.release_height_tolerance_m,
        )

    def _reward_landing_quality(self) -> torch.Tensor:
        return reward_landing_quality(
            self.near_table_entered,
            self.prev_near_table_entered,
            self.obj.get_vel(),
            xy_scale_m_s=self.landing_max_xy_speed_m_s,
            down_scale_m_s=self.landing_max_down_speed_m_s,
        )

    def _reward_hard_landing(self) -> torch.Tensor:
        return reward_hard_landing(
            self.hard_landing_violation,
            self.prev_hard_landing_violation,
        )

    def _reward_precision_progress(self) -> torch.Tensor:
        current_xy_dist = torch.norm(
            self.obj_pos_base()[:, :2] - self.target_pos[:, :2],
            dim=-1,
        )
        return self._pre_release_shaping(
            reward_precision_progress(
                self.prev_target_xy_dist_m,
                current_xy_dist,
                self.ever_carried_near,
                normalization_m=0.01,
            )
        )

    def _reward_near_target_speed(self) -> torch.Tensor:
        return reward_near_target_speed(
            self.obj_pos_base(),
            self.target_pos,
            self.obj.get_vel(),
            near_dist_m=self.place_shaping_dist_m,
            normalization_m_s=self.landing_max_xy_speed_m_s,
            max_height_error_m=self.near_target_speed_height_m,
        )

    def _reward_near_table_xy_speed_margin(self) -> torch.Tensor:
        return reward_near_table_xy_speed_margin(
            self.obj_pos_base(),
            self.obj.get_vel(),
            self.ever_carried_near,
            obj_rest_z_base=self.obj_rest_z_base,
            near_table_height_m=self.landing_speed_margin_height_m,
            speed_limit_m_s=self.landing_max_xy_speed_m_s,
            safety_zone_frac=self.landing_xy_speed_safety_zone_frac,
        )

    def _reward_near_table_down_speed_margin(self) -> torch.Tensor:
        return reward_near_table_down_speed_margin(
            self.obj_pos_base(),
            self.obj.get_vel(),
            self.ever_carried_near,
            obj_rest_z_base=self.obj_rest_z_base,
            near_table_height_m=self.landing_speed_margin_height_m,
            speed_limit_m_s=self.landing_max_down_speed_m_s,
            safety_zone_frac=self.landing_down_speed_safety_zone_frac,
        )

    def _reward_near_table_xy_action(self) -> torch.Tensor:
        return reward_near_table_xy_action(
            self.previous_action[:, :2],
            torch.abs(self.obj_pos_base()[:, 2] - self.obj_rest_z_base),
            self.ever_carried_near,
            max_height_error_m=self.near_table_xy_action_height_m,
        )

    def _reward_post_release_contact(self) -> torch.Tensor:
        grasp_point_offset = torch.zeros(3, device=self.device, dtype=gs.tc_float)
        grasp_point_offset[2] = self.grasp_center_offset_z
        ee_base = self.finger_center_base() - grasp_point_offset
        return reward_post_release_contact(
            ee_base,
            self.obj_pos_base(),
            self.release_started,
            self.carry,
            self.obj_rest_z_base,
        )

    def _reward_post_release_clearance_progress(self) -> torch.Tensor:
        return reward_post_release_clearance_progress(
            self.prev_post_release_clearance_m,
            self.post_release_clearance_m,
            self.release_valid,
            max_clearance_m=self.post_release_clearance_max_m,
        )

    def _reward_post_release_recontact(self) -> torch.Tensor:
        return reward_post_release_recontact(self.post_release_recontact_event)

    def _reward_success(self) -> torch.Tensor:
        return reward_success(self.episode_success)

    def _reward_action_penalty(self) -> torch.Tensor:
        return reward_action_penalty(self.robot.get_dofs_velocity(self.arm_dof_idx))

    def _reward_table_collision(self) -> torch.Tensor:
        if not self.collision_monitor_links:
            return torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        link_zs = torch.stack([link.get_pos()[:, 2] for link in self.collision_monitor_links], dim=-1)
        return reward_table_collision(link_zs, float(self.base_pos_world[2].item()), margin=0.04)

    def _reward_workspace_violation(self) -> torch.Tensor:
        return reward_workspace_violation(
            self.finger_center_base(),
            self.obj_pos_base(),
            self.workspace_lower,
            self.workspace_upper,
        )
