"""xArm SDK implementation of the pure ArmTransport port."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from ufactory.hardware.ft_sensor import XArmFTSensor
from ufactory.hardware.xarm import (
    MODE_POSITION,
    MODE_SERVO,
    prepare_arm_for_motion,
    prime_servo_angle_j,
    wait_for_servo_motion_ready,
)
from ufactory.kinematics.orientation import GRIPPER_DOWN_RPY_RAD, wrap_rpy_diff_rad
from ufactory.kinematics.tcp_offset import pose_flange_to_tcp
from ufactory.safety.interfaces import ArmState


class XArmTransport:
    """Thin SDK adapter; it never clears errors, warnings, or resumes itself.

    Program Cartesian targets are flange (``link6``) poses. The controller
    interprets ``set_servo_cartesian``/``set_position``/``get_position`` in its
    TCP frame, so every Cartesian command is converted with the controller
    ``tcp_offset`` (``T_tcp = T_flange @ T_offset``) before it is sent.
    """

    def __init__(
        self,
        arm: Any,
        *,
        robot_key: str,
        serial_number: str,
        cartesian_orientation_rpy_rad: tuple[float, float, float] = GRIPPER_DOWN_RPY_RAD,
        speed: float = 0.0,
        acceleration: float = 0.0,
        tcp_offset: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        self.arm = arm
        self.robot_key = robot_key
        self.serial_number = serial_number.strip()
        self.cartesian_orientation_rpy_rad = tuple(map(float, cartesian_orientation_rpy_rad))
        self.speed = float(speed)
        self.acceleration = float(acceleration)
        offset = np.zeros(6, dtype=np.float64) if tcp_offset is None else np.asarray(tcp_offset, dtype=np.float64)
        offset = offset.reshape(-1)
        if offset.shape != (6,) or not np.all(np.isfinite(offset)):
            raise ValueError("tcp_offset must contain 6 finite values [x_mm, y_mm, z_mm, roll, pitch, yaw]")
        self.tcp_offset = offset

    def flange_to_command_pose(
        self,
        xyz_m: np.ndarray,
        rpy_rad: Sequence[float] | None = None,
    ) -> list[float]:
        """Convert a flange pose (xyz in m) to the controller TCP pose ``[mm, mm, mm, rad, rad, rad]``."""

        xyz = np.asarray(xyz_m, dtype=np.float64).reshape(-1)
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            raise ValueError("flange position must contain 3 finite values")
        rpy = self.cartesian_orientation_rpy_rad if rpy_rad is None else tuple(map(float, rpy_rad))
        flange = [*(xyz * 1000.0).tolist(), *rpy]
        return pose_flange_to_tcp(flange, self.tcp_offset).tolist()

    def read_state(self) -> ArmState:
        code, angles = self.arm.get_servo_angle(is_radian=True)
        axis = int(getattr(self.arm, "axis", 0) or 0)
        if code != 0:
            q = np.full(max(axis, 1), np.nan)
        else:
            # xArm SDK often returns a padded 7-vector even for 6-axis arms.
            q = np.asarray(angles, dtype=np.float64).reshape(-1)
            if axis > 0:
                q = q[:axis]
        state_code = int(getattr(self.arm, "state", -1))
        error_code = int(getattr(self.arm, "error_code", -1))
        ready = bool(getattr(self.arm, "connected", False)) and error_code == 0 and state_code not in (3, 4)
        return ArmState(
            q_rad=q,
            monotonic_ns=time.monotonic_ns(),
            ready=ready,
            error_code=error_code,
            state_code=state_code,
            serial_number=self.serial_number,
            robot_key=self.robot_key,
        )

    def send_joint_target(self, q_rad: np.ndarray) -> int:
        q = np.asarray(q_rad, dtype=np.float64).reshape(-1)
        return int(
            self.arm.set_servo_angle_j(
                angles=q.tolist(),
                speed=self.speed,
                mvacc=self.acceleration,
                mvtime=0,
                is_radian=True,
            )
        )

    def send_cartesian_target(self, pose: np.ndarray) -> int:
        target = np.asarray(pose, dtype=np.float64).reshape(-1)
        if target.size == 3:
            command = self.flange_to_command_pose(target)
        elif target.size == 6:
            command = self.flange_to_command_pose(target[:3], target[3:])
        else:
            raise ValueError("Cartesian target must contain xyz or xyz+rpy")
        return int(
            self.arm.set_servo_cartesian(
                mvpose=command,
                speed=self.speed,
                mvacc=self.acceleration,
                mvtime=0,
                is_radian=True,
            )
        )

    def pause(self) -> int:
        return int(self.arm.set_state(3))

    def stop(self) -> int:
        return int(self.arm.set_state(4))

    def set_allow_approx_motion(self, on_off: bool) -> int:
        """Enable or disable controller singularity approximate motion (mirrors xArm SDK)."""

        if not isinstance(on_off, bool):
            raise TypeError("on_off must be a bool")
        return int(self.arm.set_allow_approx_motion(on_off))

    def get_allow_approx_motion(self) -> tuple[int, int]:
        """Return ``(sdk_code, enabled_int)`` from the controller."""

        code, enabled = self.arm.get_allow_approx_motion()
        return int(code), int(enabled)

    @property
    def is_approx_motion(self) -> bool:
        """Whether controller approximate motion is currently enabled."""

        code, enabled = self.get_allow_approx_motion()
        if code != 0:
            raise RuntimeError(f"get_allow_approx_motion failed with SDK code {code}")
        return bool(enabled)

    def disconnect(self) -> None:
        self.arm.disconnect()

    def ft_sensor(self) -> XArmFTSensor:
        """Return a safe F/T sensor controller bound to this transport's arm and robot key."""

        return XArmFTSensor(self.arm, robot_key=self.robot_key)

    def leave_software_stop(self) -> None:
        """Exit controller STOP when there is no active error (Studio Enable equivalent).

        Does not call clean_error/clean_warn. A previous software STOP (state=4,
        error_code=0) is cleared with set_state(0) only.
        """

        state = self.read_state()
        if state.error_code != 0:
            raise RuntimeError("active controller error must be resolved manually before leaving STOP")
        if int(getattr(self.arm, "warn_code", 0)) != 0:
            raise RuntimeError("active controller warning must be inspected manually before leaving STOP")
        if int(state.state_code) != 4:
            return
        code = self.arm.motion_enable(enable=True)
        if int(code) != 0:
            raise RuntimeError(f"motion_enable failed with SDK code {code} while leaving STOP")
        code = self.arm.set_state(0)
        if int(code) != 0:
            raise RuntimeError(f"set_state(0) failed with SDK code {code} while leaving STOP")
        state = self.read_state()
        if int(state.state_code) == 4:
            raise RuntimeError("controller remained in STOP after set_state(0); inspect UFACTORY Studio")

    def preposition_joints(
        self,
        q_rad: np.ndarray,
        *,
        speed_rad_s: float = 0.35,
        mvacc_rad_s2: float = 2.0,
        tolerance_rad: float = 0.02,
    ) -> float:
        """Move to a known joint start with MODE_POSITION ``set_servo_angle`` (not servo stream).

        Returns the max absolute joint error after the blocking move.
        """

        target = np.asarray(q_rad, dtype=np.float64).reshape(-1)
        if target.size == 0 or not np.all(np.isfinite(target)):
            raise ValueError("preposition joint target must be finite")
        self.leave_software_stop()
        state = self.read_state()
        if state.error_code != 0:
            raise RuntimeError("active controller error must be resolved manually before preposition")
        if int(getattr(self.arm, "warn_code", 0)) != 0:
            raise RuntimeError("active controller warning must be inspected manually before preposition")
        prepare_arm_for_motion(self.arm, mode=MODE_POSITION)
        code = self.arm.set_servo_angle(
            angle=target.tolist(),
            speed=float(speed_rad_s),
            mvacc=float(mvacc_rad_s2),
            wait=True,
            is_radian=True,
        )
        if int(code) != 0:
            self.stop()
            raise RuntimeError(f"preposition set_servo_angle failed with SDK code {code}; controller was stopped")
        reported = self.read_state().q_rad
        if reported.size < target.size or not np.all(np.isfinite(reported[: target.size])):
            self.stop()
            raise RuntimeError("preposition joint feedback was invalid; controller was stopped")
        err = float(np.max(np.abs(reported[: target.size] - target)))
        if err > float(tolerance_rad):
            self.stop()
            raise RuntimeError(
                f"preposition joint error {err:.4f} rad exceeds {float(tolerance_rad):.4f} rad; controller was stopped"
            )
        return err

    def preposition_cartesian(
        self,
        xyz_m: np.ndarray,
        *,
        rpy_rad: tuple[float, float, float] | None = None,
        speed_mm_s: float = 100.0,
        mvacc_mm_s2: float = 500.0,
        tolerance_mm: float = 2.0,
        orient_tolerance_rad: float = 0.05,
    ) -> tuple[float, float]:
        """Move to a known Cartesian start with MODE_POSITION ``set_position``.

        ``xyz_m``/``rpy_rad`` describe the flange pose; the command and the
        feedback comparison use the controller TCP pose.
        Returns ``(xyz_err_mm, rpy_err_rad)`` after the blocking move.
        """

        target = np.asarray(xyz_m, dtype=np.float64).reshape(-1)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("preposition Cartesian target must be finite xyz")
        pose = self.flange_to_command_pose(target, rpy_rad)
        self.leave_software_stop()
        state = self.read_state()
        if state.error_code != 0:
            raise RuntimeError("active controller error must be resolved manually before preposition")
        if int(getattr(self.arm, "warn_code", 0)) != 0:
            raise RuntimeError("active controller warning must be inspected manually before preposition")
        prepare_arm_for_motion(self.arm, mode=MODE_POSITION)
        code = self.arm.set_position(
            *pose,
            speed=float(speed_mm_s),
            mvacc=float(mvacc_mm_s2),
            wait=True,
            is_radian=True,
        )
        if int(code) != 0:
            self.stop()
            raise RuntimeError(f"preposition set_position failed with SDK code {code}; controller was stopped")
        get_code, reported = self.arm.get_position(is_radian=True)
        if int(get_code) != 0:
            self.stop()
            raise RuntimeError(f"preposition get_position failed with SDK code {get_code}; controller was stopped")
        reported_arr = np.asarray(reported, dtype=np.float64).reshape(-1)
        if reported_arr.size < 6 or not np.all(np.isfinite(reported_arr[:6])):
            self.stop()
            raise RuntimeError("preposition Cartesian feedback was invalid; controller was stopped")
        xyz_err = float(np.linalg.norm(reported_arr[:3] - np.asarray(pose[:3], dtype=np.float64)))
        rpy_err = float(np.linalg.norm(wrap_rpy_diff_rad(reported_arr[3:6] - np.asarray(pose[3:6], dtype=np.float64))))
        if xyz_err > float(tolerance_mm):
            self.stop()
            raise RuntimeError(
                f"preposition Cartesian xyz error {xyz_err:.2f} mm exceeds {float(tolerance_mm):.2f} mm; "
                "controller was stopped"
            )
        if rpy_err > float(orient_tolerance_rad):
            self.stop()
            raise RuntimeError(
                f"preposition Cartesian rpy error {rpy_err:.4f} rad exceeds {float(orient_tolerance_rad):.4f} rad; "
                "controller was stopped"
            )
        return xyz_err, rpy_err

    def authorize_motion(self, *, mode: int = 1) -> None:
        """Enable motion only when the caller has already confirmed real mode."""

        self.leave_software_stop()
        state = self.read_state()
        if state.error_code != 0:
            raise RuntimeError("active controller error must be resolved manually before motion")
        if int(getattr(self.arm, "warn_code", 0)) != 0:
            raise RuntimeError("active controller warning must be inspected manually before motion")
        operations = (
            ("motion_enable", lambda: self.arm.motion_enable(enable=True)),
            ("set_mode", lambda: self.arm.set_mode(mode)),
            ("set_state", lambda: self.arm.set_state(0)),
        )
        for label, operation in operations:
            code = operation()
            if int(code) != 0:
                self.stop()
                raise RuntimeError(f"{label} failed with SDK code {code}; controller was stopped")
        if int(mode) == MODE_SERVO:
            # Official servoj entry requires the report stream to show mode=1
            # before set_servo_angle_j; otherwise SDK warns mode: 1 (0).
            wait_for_servo_motion_ready(self.arm)

    def prime_joint_stream(self, q_rad: np.ndarray) -> None:
        """Prime MODE_SERVO with the first joint target before timed streaming."""

        if int(getattr(self.arm, "mode", -1)) != MODE_SERVO:
            self.authorize_motion(mode=MODE_SERVO)
        target = np.asarray(q_rad, dtype=np.float64).reshape(-1)
        prime_servo_angle_j(
            self.arm,
            target.tolist(),
            speed=self.speed,
            mvacc=self.acceleration,
        )
        wait_for_servo_motion_ready(self.arm)
