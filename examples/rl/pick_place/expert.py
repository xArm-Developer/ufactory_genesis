"""Independent scripted policy for physics checks; never used for PPO training."""

from __future__ import annotations

from typing import Any

import torch

from ufactory.simulation.g2_adapter import pad_centers

from .metrics import cube_clearance


class PickPlaceExpert:
    """Stateful baseline using measured pads and cube height to release at 2 mm."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.stage = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.steps = torch.zeros_like(self.stage)
        self.episode_id = torch.full_like(self.stage, -1)

    def __call__(self, _obs: Any) -> torch.Tensor:
        env = self.env
        reset = self.episode_id != env.episode_id
        self.stage[reset] = 0
        self.steps[reset] = 0
        self.episode_id = env.episode_id.clone()
        ee, obj = env.ee_link.get_pos(), env.obj.get_pos()
        ee_vel = env.ee_link.get_vel()
        left, right = pad_centers(env.g2_model, env.left_finger_link, env.right_finger_link)
        center = (left + right) / 2
        height = cube_clearance(obj, env.obj.get_quat()).clamp(min=0)
        goal = ee + obj - center
        # Open fingers have their pad centers above the closed-grasp center;
        # leave fingertip clearance across the 300-500 mm reach envelope.
        goal[:, 2] += 0.016 + torch.where(self.stage == 0, 0.05, 0)
        aligned = (torch.linalg.vector_norm(goal - env.ee_setpoint, dim=-1) < 0.002) & (
            torch.linalg.vector_norm(ee_vel, dim=-1) < 0.02
        )
        target_error = torch.linalg.vector_norm(obj[:, :2] - env.target_xy, dim=-1)
        advance = (
            ((self.stage <= 1) & aligned & (self.steps > 5))
            | ((self.stage == 2) & (env.metrics.contact_steps >= 10) & (self.steps >= 20))
            | ((self.stage == 3) & (obj[:, 2] >= 0.090))
            | ((self.stage == 4) & (target_error < 0.005) & (torch.linalg.vector_norm(ee_vel[:, :2], dim=-1) < 0.02))
            | ((self.stage == 5) & (height <= 0.0025))
        )
        self.stage = torch.where(advance, self.stage + 1, self.stage)
        self.steps = torch.where(advance, 0, self.steps + 1)
        actions = torch.zeros(env.num_envs, 4, device=env.device)
        approach = (self.stage <= 1)[:, None]
        actions[:, :3] = torch.where(approach, (goal - env.ee_setpoint) / env.action_scale, 0)
        actions[:, 3] = torch.where(self.stage == 2, -1.0, 0)
        actions[:, 2] = torch.where(self.stage == 3, 0.7, actions[:, 2])
        actions[:, :2] = torch.where(
            (self.stage >= 4)[:, None],
            ((ee[:, :2] + env.target_xy - obj[:, :2] - env.ee_setpoint[:, :2]) / env.action_scale).clamp(-0.9, 0.9),
            actions[:, :2],
        )
        # Slow final approach while every control channel remains available.
        speed = torch.where(height > 0.015, 0.12, 0.015)
        descent = -torch.minimum((height - 0.0015).clamp(min=0), speed * env.ctrl_dt) / env.action_scale
        actions[:, 2] = torch.where(self.stage == 5, descent, actions[:, 2])
        actions[:, 3] = torch.where(self.stage >= 6, 1.0, actions[:, 3])
        return actions.clamp(-1, 1)
