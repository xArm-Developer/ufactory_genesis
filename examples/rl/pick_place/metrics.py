"""Vectorized geometric and contact criteria for gentle tabletop placement."""

from __future__ import annotations

import math
from typing import Any

import torch

from ufactory.training.pick_place_config import (
    SPAWN_X,
    SPAWN_Y,
    SPAWN_XY_LOWER as SPAWN_XY_LOWER,
    SPAWN_XY_UPPER as SPAWN_XY_UPPER,
    TARGET_X,
    TARGET_Y,
    CUBE_SIZE,
    CUBE_REST_Z,
    SAFE_LIFT_Z,
)

TARGET_TOL_M = 0.020
DRAG_LIMIT_M = 0.010
TILT_LIMIT_DEG = 15.0
SUCCESS_HOLD_STEPS = 10
CONTACT_CONFIRM_STEPS = 3
CONTACT_FORCE_N = 0.2
RELEASE_FORCE_N = 0.05
SUPPORT_FORCE_N = 0.05
RELEASE_MAX_HEIGHT_M = 0.003
PRESS_LIMIT_M = 0.001
PLACE_CLEARANCE_M = 0.002


def cube_clearance(pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Lowest cube point above the z=0 tabletop, using a WXYZ quaternion."""
    quat = torch.nn.functional.normalize(quat, dim=-1)
    w, x, y, z = quat.unbind(-1)
    row_z = torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1)
    return pos[:, 2] - (CUBE_SIZE / 2) * row_z.abs().sum(-1)


def grasp_distance(pad_center: torch.Tensor, obj_pos: torch.Tensor) -> torch.Tensor:
    """Distance to the cube's graspable vertical band, with 3 mm edge margin.

    G2 pad height changes with opening. A band provides approach guidance
    without driving open fingertips into the table to match cube-center height.
    """
    relative = pad_center - obj_pos
    relative = torch.cat((relative[:, :2], (relative[:, 2].abs() - CUBE_SIZE * 0.4).clamp(min=0)[:, None]), -1)
    return torch.linalg.vector_norm(relative, dim=-1)


class PickPlaceMetrics:
    """Track release and touchdown until both fingers have cleared for 3 samples.

    Touchdown includes the preceding EE sample. Contact loss never resets that
    reference; one remaining finger and short contact gaps keep the same event.
    Release height includes the preceding object sample to reject sampled drops.
    """

    def __init__(
        self,
        n: int,
        device: Any,
        *,
        ctrl_dt: float = 0.02,
        target_tol: float = TARGET_TOL_M,
        drag_limit: float = DRAG_LIMIT_M,
        tilt_limit: float = TILT_LIMIT_DEG,
        hold_steps: int = SUCCESS_HOLD_STEPS,
        place_clearance: float = PLACE_CLEARANCE_M,
        release_max_height: float = RELEASE_MAX_HEIGHT_M,
        press_limit: float = PRESS_LIMIT_M,
    ) -> None:
        self.n, self.device = int(n), device
        self.target_tol, self.drag_limit, self.tilt_limit = target_tol, drag_limit, tilt_limit
        self.place_clearance = float(place_clearance)
        self.release_max_height = float(release_max_height)
        self.press_limit = float(press_limit)
        self.required_hold_steps = max(int(hold_steps), math.ceil(0.2 / ctrl_dt))
        self.spawn_xy = torch.zeros(n, 2, device=device)
        self.target_xy = torch.tensor([[TARGET_X, TARGET_Y]], device=device).expand(n, 2)
        self.finger_contact = torch.zeros(n, 2, device=device, dtype=torch.bool)
        self._float_names = (
            "max_drag",
            "max_tilt_deg",
            "max_held_z",
            "target_error",
            "clearance",
            "max_press",
            "touchdown_ee_z",
            "min_ee_z",
            "release_height",
            "release_candidate_height",
            "max_table_force",
            "max_place_table_force",
            "max_held_table_force",
            "max_robot_table_force",
            "prev_ee_z",
            "prev_clearance",
        )
        self._bool_names = (
            "ever_carried",
            "ever_lifted_clear",
            "touchdown_seen",
            "release_confirmed",
            "high_release",
            "prev_any_contact",
        )
        for name in self._float_names:
            setattr(self, name, torch.zeros(n, device=device))
        for name in self._bool_names:
            setattr(self, name, torch.zeros(n, device=device, dtype=torch.bool))
        for name in ("contact_steps", "hold_steps", "loss_steps", "clear_steps"):
            setattr(self, name, torch.zeros(n, device=device, dtype=torch.long))
        self.reset(torch.arange(n, device=device))

    def reset(self, idx: Any, xy: torch.Tensor | None = None, ee_z: torch.Tensor | None = None) -> None:
        self.spawn_xy[idx] = self.spawn_xy.new_tensor((SPAWN_X, SPAWN_Y)) if xy is None else xy
        for name in (*self._float_names, *self._bool_names, "contact_steps", "hold_steps", "loss_steps", "clear_steps"):
            getattr(self, name)[idx] = 0
        self.finger_contact[idx] = False
        self.max_held_z[idx] = CUBE_REST_Z
        self.prev_ee_z[idx] = 0.3 if ee_z is None else ee_z
        self.target_error[idx] = torch.linalg.vector_norm(self.spawn_xy[idx] - self.target_xy[idx], dim=-1)

    def update(
        self,
        pos: torch.Tensor,
        quat: torch.Tensor,
        forces: torch.Tensor,
        obj_vel: torch.Tensor,
        gap: torch.Tensor,
        *,
        ee_z: torch.Tensor,
        table_force: torch.Tensor,
        robot_table_force: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self.clearance = cube_clearance(pos, quat)
        # Retain solver penetration for diagnostics; release height is an
        # above-table distance, so micrometre penetration counts as zero.
        height = self.clearance.clamp(min=0)
        up_z = (1 - 2 * (quat[:, 1].square() + quat[:, 2].square())).clamp(-1, 1)
        tilt = torch.rad2deg(torch.acos(up_z))
        on_table = self.clearance < 0.015
        before_lift = (~self.ever_carried) & on_table
        drag = torch.linalg.vector_norm(pos[:, :2] - self.spawn_xy, dim=-1)
        self.max_drag = torch.where(before_lift, torch.maximum(self.max_drag, drag), self.max_drag)
        self.max_tilt_deg = torch.where(before_lift, torch.maximum(self.max_tilt_deg, tilt), self.max_tilt_deg)
        quality_ok = (self.max_drag <= self.drag_limit) & (self.max_tilt_deg <= self.tilt_limit)

        self.finger_contact = torch.where(
            forces > CONTACT_FORCE_N, True, torch.where(forces < RELEASE_FORCE_N, False, self.finger_contact)
        )
        touching = self.finger_contact.all(-1)
        any_contact = (forces >= RELEASE_FORCE_N).any(-1)
        # A later regrasp is a new contact interval. Retain episode maxima,
        # but measure its touchdown and separation again, including one finger.
        recontact = self.release_confirmed & any_contact
        self.release_confirmed &= ~recontact
        self.touchdown_seen &= ~recontact
        self.contact_steps = torch.where(touching, self.contact_steps + 1, 0)
        stable_contact = self.contact_steps >= CONTACT_CONFIRM_STEPS
        carried = stable_contact & (height > 0.02)
        first_carry = carried & (~self.ever_carried) & quality_ok
        self.ever_carried |= carried
        self.max_held_z = torch.where(
            stable_contact, torch.maximum(self.max_held_z, height + CUBE_REST_Z), self.max_held_z
        )
        self.loss_steps = torch.where(self.ever_carried & (~touching), self.loss_steps + 1, 0)
        lifted = carried & (height >= SAFE_LIFT_Z - CUBE_REST_Z) & quality_ok
        first_lift_clear = lifted & (~self.ever_lifted_clear)
        self.ever_lifted_clear |= lifted

        supported = (table_force > SUPPORT_FORCE_N) & (self.clearance <= self.place_clearance)
        first_touchdown = (
            self.ever_lifted_clear
            & supported
            & (~self.touchdown_seen)
            & (~self.release_confirmed)
            & (any_contact | self.prev_any_contact)
        )
        self.touchdown_ee_z = torch.where(first_touchdown, torch.maximum(self.prev_ee_z, ee_z), self.touchdown_ee_z)
        self.min_ee_z = torch.where(first_touchdown, ee_z, self.min_ee_z)
        self.touchdown_seen |= first_touchdown
        tracking = self.touchdown_seen & (~self.release_confirmed) & (any_contact | self.prev_any_contact)
        self.min_ee_z = torch.where(tracking, torch.minimum(self.min_ee_z, ee_z), self.min_ee_z)
        old_press = self.max_press.clone()
        self.max_press = torch.where(
            tracking, torch.maximum(self.max_press, (self.touchdown_ee_z - self.min_ee_z).clamp(min=0)), self.max_press
        )

        both_clear = (forces < RELEASE_FORCE_N).all(-1)
        first_clear = self.ever_carried & both_clear & (self.clear_steps == 0) & (~self.release_confirmed)
        self.release_candidate_height = torch.where(
            first_clear, torch.maximum(self.prev_clearance.clamp(min=0), height), self.release_candidate_height
        )
        self.clear_steps = torch.where(both_clear & self.ever_carried, self.clear_steps + 1, 0)
        first_release = (self.clear_steps >= CONTACT_CONFIRM_STEPS) & (~self.release_confirmed)
        self.release_height = torch.where(
            first_release, torch.maximum(self.release_height, self.release_candidate_height), self.release_height
        )
        self.high_release |= first_release & (self.release_height > self.release_max_height + 1e-7)
        self.release_confirmed |= first_release
        valid_release = (
            self.release_confirmed & (~self.high_release) & (self.release_height <= self.release_max_height + 1e-7)
        )
        released = both_clear & (gap >= 0.035) & valid_release

        self.max_table_force = torch.maximum(self.max_table_force, table_force)
        self.max_place_table_force = torch.maximum(
            self.max_place_table_force, torch.where(self.ever_carried, table_force, 0)
        )
        self.max_held_table_force = torch.maximum(
            self.max_held_table_force, torch.where(any_contact | self.prev_any_contact, table_force, 0)
        )
        self.max_robot_table_force = torch.maximum(self.max_robot_table_force, robot_table_force)
        self.target_error = torch.linalg.vector_norm(pos[:, :2] - self.target_xy, dim=-1)
        in_target_xy = self.target_error <= self.target_tol
        at_target_table = self.ever_lifted_clear & in_target_xy & supported & (tilt <= self.tilt_limit)
        press_failed = self.max_press > self.press_limit + 1e-7
        collision_failed = self.max_robot_table_force > CONTACT_FORCE_N
        settled = (
            at_target_table
            & released
            & quality_ok
            & (~press_failed)
            & (~collision_failed)
            & (torch.linalg.vector_norm(obj_vel, dim=-1) < 0.03)
        )
        self.hold_steps = torch.where(settled, self.hold_steps + 1, 0)
        self.prev_ee_z = ee_z.clone()
        self.prev_clearance = self.clearance.clone()
        self.prev_any_contact = any_contact.clone()
        return {
            "touching": touching,
            "any_contact": any_contact,
            "stable_contact": stable_contact,
            "quality_ok": quality_ok,
            "first_carry": first_carry,
            "first_lift_clear": first_lift_clear,
            "first_release": first_release,
            "first_touchdown": first_touchdown,
            "recontact": recontact,
            "in_target_xy": in_target_xy,
            "at_target_table": at_target_table,
            "released": released,
            "supported": supported,
            "settled": settled,
            "success": self.hold_steps >= self.required_hold_steps,
            "drag_failed": self.max_drag > self.drag_limit,
            "tilt_failed": self.max_tilt_deg > self.tilt_limit,
            "dropped": self.ever_carried & on_table & (~in_target_xy) & (self.loss_steps >= 15),
            "press_failed": press_failed,
            "high_release": self.high_release,
            "collision_failed": collision_failed,
            "press_increment": self.max_press - old_press,
            "up_z": up_z,
        }
