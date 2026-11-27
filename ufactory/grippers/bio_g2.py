"""Bio Gripper G2 controller and command-space conversions for Genesis robot entities.

Provides a reusable :class:`BioGripperG2` class that discovers Bio Gripper G2
joints on a Genesis robot entity and exposes calibrated force/rate-limited servo
control, PD setup, and open/close demo helpers.  The same class works for any
supported UFACTORY arm (xArm5 / xArm6 / xArm7 / UF850) because joint names are
identical across all combo URDFs.
"""

from __future__ import annotations

import numpy as np

RIGHT_JOINT = "bio_gripper_g2_right_finger_joint"
LEFT_JOINT = "bio_gripper_g2_left_finger_joint"

BIO_GRIPPER_G2_CLOSED_GAP_MM = 71.0
BIO_GRIPPER_G2_OPEN_GAP_MM = 150.0
STROKE = (BIO_GRIPPER_G2_OPEN_GAP_MM - BIO_GRIPPER_G2_CLOSED_GAP_MM) / 2000.0  # 0.0395 m
BIO_GRIPPER_G2_JOINT_VELOCITY_M_S = 1.0 / 3.0

# Joint-space constants (prismatic, meters).
#
# The right-finger prismatic dof drives the gripper; the left finger mimics it (-1).
# Joint zero is the CLOSED pose (real 71 mm two-finger gap) and the dof opens the jaws
# symmetrically up to STROKE (real 150 mm gap).  Per finger the travel is
# (150 - 71) / 2 = 39.5 mm, so the full finger stroke is 0.0395 m.
CLOSE_POS = 0.0  # Right-finger joint value when gripper is fully closed (71 mm gap).
OPEN_POS = STROKE  # Right-finger joint value when gripper is fully open (150 mm gap).

# Physical two-finger gap (meters) at each stroke extreme — the real Bio Gripper G2
# mechanical range.  Gap scales linearly: GAP = CLOSED_GAP + 2 * right_joint_value.
CLOSED_GAP = BIO_GRIPPER_G2_CLOSED_GAP_MM / 1000.0
OPEN_GAP = BIO_GRIPPER_G2_OPEN_GAP_MM / 1000.0

BIO_G2_JOINT_NAMES = (
    RIGHT_JOINT,
    LEFT_JOINT,
)
BIO_GRIPPER_G2_CLOSED_GAP_M = CLOSED_GAP
BIO_GRIPPER_G2_OPEN_GAP_M = OPEN_GAP
BIO_GRIPPER_G2_SIM_CLOSED_DRIVE = CLOSE_POS
BIO_GRIPPER_G2_SIM_OPEN_DRIVE = OPEN_POS

_DEFAULT_HOLD_STEPS = 200
BIO_GRIPPER_G2_DEMO_HOLD_S = 0.5
BIO_GRIPPER_G2_ARRIVAL_TOL_MM = 1.0

# TCP (gripping-ridge centre at bio_gripper_g2_tcp_link) expressed in the xArm flange frame (link6), metres.
BIO_GRIPPER_G2_TCP_IN_FLANGE_M: tuple[float, float, float] = (0.15764, 0.0, 0.05752)

BIO_G2_BLOCKED_MODES = frozenset(("sdk-sim", "real"))
BIO_G2_CONTROLLER_VALIDATED = frozenset({("pick_place", "xarm6")})


def bio_g2_controller_mode_error(task: str, robot: str, gripper: str, mode: str) -> str | None:
    """CLI error for Bio Gripper G2 in a controller mode not yet hardware-validated, else ``None``."""
    if gripper != "bio_gripper_g2" or mode not in BIO_G2_BLOCKED_MODES:
        return None
    if (task, robot) in BIO_G2_CONTROLLER_VALIDATED:
        return None
    validated = ", ".join(f"{t} on {r}" for t, r in sorted(BIO_G2_CONTROLLER_VALIDATED))
    return (
        f"--gripper bio_gripper_g2 currently requires --mode sim or dry-run for {task} on {robot}: "
        f"Bio Gripper G2 on a real controller has only been validated on hardware for {validated}"
    )


def clamp_bio_gripper_g2_gap_m(gap_m: float) -> float:
    """Clamp a two-finger gap in metres to the Bio Gripper G2 physical range [0.071, 0.150]."""
    return max(BIO_GRIPPER_G2_CLOSED_GAP_M, min(BIO_GRIPPER_G2_OPEN_GAP_M, float(gap_m)))


def bio_gripper_g2_gap_m_to_sim_drive(gap_m: float) -> float:
    """Map a clamped two-finger gap in metres to the right-finger prismatic position in metres."""
    return (clamp_bio_gripper_g2_gap_m(gap_m) - BIO_GRIPPER_G2_CLOSED_GAP_M) / 2.0


def bio_gripper_g2_sim_drive_to_gap_m(drive: float) -> float:
    """Map the right-finger prismatic joint position in metres to a clamped gap in metres."""
    clipped = max(BIO_GRIPPER_G2_SIM_CLOSED_DRIVE, min(BIO_GRIPPER_G2_SIM_OPEN_DRIVE, float(drive)))
    return clamp_bio_gripper_g2_gap_m(BIO_GRIPPER_G2_CLOSED_GAP_M + 2.0 * clipped)


def bio_gripper_g2_gap_m_to_sdk_pos_mm(gap_m: float) -> float:
    """Map a physical gap in metres to ``set_bio_gripper_g2_position`` millimetres."""
    return clamp_bio_gripper_g2_gap_m(gap_m) * 1000.0


class BioGripperG2:
    """Reusable Bio Gripper G2 controller bound to a Genesis robot entity.

    Class-level constants (also available as module-level names)::

        BioGripperG2.RIGHT_JOINT   # "bio_gripper_g2_right_finger_joint"
        BioGripperG2.LEFT_JOINT    # "bio_gripper_g2_left_finger_joint"
        BioGripperG2.STROKE        # 0.0395 m (per-finger travel)
        BioGripperG2.CLOSE_POS     # 0.0     (closed, 71 mm two-finger gap)
        BioGripperG2.OPEN_POS      # 0.0395  (open, 150 mm two-finger gap)
        BioGripperG2.CLOSED_GAP    # 0.071 m
        BioGripperG2.OPEN_GAP      # 0.150 m
    """

    RIGHT_JOINT = RIGHT_JOINT
    LEFT_JOINT = LEFT_JOINT
    STROKE = STROKE
    OPEN_POS = OPEN_POS
    CLOSE_POS = CLOSE_POS
    CLOSED_GAP = CLOSED_GAP
    OPEN_GAP = OPEN_GAP

    def __init__(
        self,
        robot,
        *,
        demo_dt: float = 0.02,
        demo_hold_s: float = BIO_GRIPPER_G2_DEMO_HOLD_S,
        arrival_tol_mm: float = BIO_GRIPPER_G2_ARRIVAL_TOL_MM,
    ):
        """Discover Bio Gripper G2 joints on *robot*."""
        self._robot = robot
        joint_map = {j.name.split("/")[-1]: j for j in getattr(robot, "joints", ())}
        self._right_joint = joint_map.get(RIGHT_JOINT)
        self._left_joint = joint_map.get(LEFT_JOINT)
        self._controller = None
        self.demo_dt = float(demo_dt)
        self.demo_hold_s = float(demo_hold_s)
        self.arrival_tol_mm = float(arrival_tol_mm)
        self.demo_hold_steps_required = max(1, int(round(self.demo_hold_s / self.demo_dt)))
        self.demo_phase: str = "closed"
        self.demo_hold_steps: int = 0
        self.completed_cycles: int = 0
        self._pending_observation: bool = False

    # -- discovery ----------------------------------------------------------

    @property
    def found(self) -> bool:
        """``True`` if the robot entity has Bio Gripper G2 joints."""
        return self._right_joint is not None

    @property
    def drive_dof_idx(self) -> list[int]:
        """DOF index of the drive joint (right finger)."""
        return [] if self._right_joint is None else [self._right_joint.dofs_idx_local[0]]

    @property
    def all_dof_idx(self) -> list[int]:
        """DOF indices of both finger joints (drive + mimic)."""
        return [joint.dofs_idx_local[0] for joint in (self._right_joint, self._left_joint) if joint is not None]

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def demo_target(step: int, hold_steps: int = _DEFAULT_HOLD_STEPS) -> float:
        """Alternating open/close target value for demo loops.

        Returns:
            ``CLOSE_POS`` during odd phases, ``OPEN_POS`` during even phases.
        """
        phase = (step // hold_steps) % 2
        return CLOSE_POS if phase else OPEN_POS

    def step_demo_target(self) -> tuple[float, str]:
        """Return ``(target_drive, phase_label)`` for the arrival-confirmed 0.5 s hold demo cycle."""
        if self._pending_observation:
            self.on_step_completed(self._robot)
        self._pending_observation = True
        target = CLOSE_POS if self.demo_phase == "closed" else OPEN_POS
        return target, self.demo_phase

    def on_step_completed(self, _robot=None) -> None:
        if not self._pending_observation:
            return
        self._pending_observation = False
        if self._is_at_demo_target():
            self.demo_hold_steps += 1
            if self.demo_hold_steps >= self.demo_hold_steps_required:
                self.demo_hold_steps = 0
                if self.demo_phase == "open":
                    self.completed_cycles += 1
                    self.demo_phase = "closed"
                else:
                    self.demo_phase = "open"
        else:
            self.demo_hold_steps = 0

    def _is_at_demo_target(self) -> bool:
        settings = getattr(
            self._robot,
            "_ufactory_bio_g2_demo_settings",
            (BIO_GRIPPER_G2_CLOSED_GAP_MM, 50.0, 50.0, None),
        )
        closed_gap_mm = float(settings[0])
        speed_percent = float(settings[1])
        speed_mm_s = settings[3] if len(settings) >= 4 else None
        if speed_mm_s is None and speed_percent <= 0.0:
            return False
        target_gap_mm = closed_gap_mm if self.demo_phase == "closed" else BIO_GRIPPER_G2_OPEN_GAP_MM
        controller = self._controller or getattr(self._robot, "_ufactory_bio_g2_controller", None)
        if controller is None:
            return True
        if hasattr(controller, "telemetry"):
            telemetry = controller.telemetry()
            ref_gap_mm = float(telemetry.get("reference_gap_mm", target_gap_mm))
            if abs(ref_gap_mm - target_gap_mm) > 1e-4:
                return False
        return abs(self.gap_mm() - target_gap_mm) <= self.arrival_tol_mm

    # -- PD & calibrated servo control --------------------------------------

    def setup_pd(self) -> None:
        """Configure calibrated Bio Gripper G2 mimic constraints and servo controller."""
        if not self.found:
            return
        from ufactory.simulation.bio_g2_adapter import (
            BioG2GenesisController,
            configure_bio_g2_mimic_constraints,
        )

        configure_bio_g2_mimic_constraints(self._robot)
        self._controller = BioG2GenesisController(self._robot, initial_position_mm=BIO_GRIPPER_G2_OPEN_GAP_MM)
        self._robot._ufactory_bio_g2_controller = self._controller
        self._robot._ufactory_bio_g2_demo_cycle = self
        self.set_pose(OPEN_POS)

    def set_pose(self, right_value: float) -> None:
        """Teleport gripper joints to *right_value* (initializes state & controller)."""
        active = self.all_dof_idx or self.drive_dof_idx
        if not active:
            return
        controller = self._controller or getattr(self._robot, "_ufactory_bio_g2_controller", None)
        if controller is not None:
            gap_mm = 1000.0 * bio_gripper_g2_sim_drive_to_gap_m(right_value)
            controller.reset(gap_mm)
            controller.command(gap_mm)
            controller.apply(0.02)
            return
        target = self._targets(active, right_value)
        self._robot.set_dofs_position(target, active)
        if hasattr(self._robot, "control_dofs_position"):
            self._robot.control_dofs_position(target, active)

    def control_pose(self, right_value: float, dt: float = 0.02) -> None:
        """Servo-control gripper toward *right_value* without teleporting DOFs."""
        active = self.all_dof_idx or self.drive_dof_idx
        if not active:
            return
        controller = self._controller or getattr(self._robot, "_ufactory_bio_g2_controller", None)
        if controller is not None:
            settings = getattr(
                self._robot,
                "_ufactory_bio_g2_demo_settings",
                (BIO_GRIPPER_G2_CLOSED_GAP_MM, 50.0, 50.0, None),
            )
            if len(settings) == 4:
                position_mm, speed_percent, force_percent, speed_mm_s = settings
            else:
                position_mm, speed_percent, force_percent = settings
                speed_mm_s = None
            target_mm = float(position_mm) if float(right_value) < OPEN_POS else BIO_GRIPPER_G2_OPEN_GAP_MM
            if speed_mm_s is not None:
                controller.command(target_mm, speed_mm_s=float(speed_mm_s), force_percent=float(force_percent))
            else:
                controller.command(target_mm, speed_percent=float(speed_percent), force_percent=float(force_percent))
            controller.apply(dt)
            return
        target = self._targets(active, float(right_value))
        self._robot.control_dofs_position(target, active)

    def gap_mm(self) -> float:
        """Current two-finger gap in millimetres."""
        controller = self._controller or getattr(self._robot, "_ufactory_bio_g2_controller", None)
        if controller is not None:
            return float(controller.gap_mm())
        return BIO_GRIPPER_G2_OPEN_GAP_MM

    # -- internal -----------------------------------------------------------

    def _targets(self, active_dof_idx: list[int], right_value: float) -> np.ndarray:
        """Build a position target array for *active_dof_idx*."""
        values = {}
        if self._right_joint is not None:
            values[self._right_joint.dofs_idx_local[0]] = right_value
        if self._left_joint is not None:
            values[self._left_joint.dofs_idx_local[0]] = -right_value
        return np.array([values.get(dof_i, 0.0) for dof_i in active_dof_idx], dtype=np.float64)
