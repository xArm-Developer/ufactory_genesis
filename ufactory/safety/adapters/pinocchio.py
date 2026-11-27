"""Pinocchio 4 + Coal implementation of kinematics and collision ports."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

import numpy as np

from ufactory.safety.interfaces import CollisionResult


@dataclass(frozen=True)
class EnvironmentObstacle:
    """A static axis-aligned box in the robot base frame."""

    name: str
    size_m: tuple[float, float, float]
    center_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        values = (*self.size_m, *self.center_m)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("obstacle dimensions and position must be finite")
        if any(value <= 0.0 for value in self.size_m):
            raise ValueError("obstacle dimensions must be positive")


class _PinocchioModel:
    def __init__(
        self,
        urdf_path: str | Path,
        *,
        joint_names: tuple[str, ...],
        ee_link: str,
        passive_joint_positions: dict[str, float] | None = None,
    ) -> None:
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise RuntimeError("Pinocchio 4 and Coal are required for safety preflight") from exc
        self.pin = pin
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(self.urdf_path)
        try:
            self.model = pin.buildModelFromUrdf(str(self.urdf_path), mimic=True)
            self._mimic_fallback = False
        except ValueError as exc:
            if "mimic" not in str(exc).lower():
                raise
            self.model = pin.buildModelFromUrdf(str(self.urdf_path))
            self._mimic_fallback = True
        self.data = self.model.createData()
        self.joint_names = joint_names
        self.ee_link = ee_link
        try:
            self.ee_frame_id = int(self.model.getFrameId(ee_link))
        except Exception as exc:
            raise ValueError(f"URDF does not contain end-effector frame {ee_link!r}") from exc
        if self.ee_frame_id >= len(self.model.frames):
            raise ValueError(f"URDF does not contain end-effector frame {ee_link!r}")
        self.q_indices: list[int] = []
        self.v_indices: list[int] = []
        for name in joint_names:
            joint_id = int(self.model.getJointId(name))
            if joint_id == 0:
                raise ValueError(f"URDF is missing configured joint {name!r}")
            joint = self.model.joints[joint_id]
            if joint.nq != 1:
                raise ValueError(f"configured arm joint {name!r} must have nq=1")
            self.q_indices.append(int(joint.idx_q))
            self.v_indices.append(int(joint.idx_v))
        self.neutral = np.asarray(pin.neutral(self.model), dtype=np.float64)
        self.passive_q_indices: dict[str, int] = {}
        for name, value in (passive_joint_positions or {}).items():
            joint_id = int(self.model.getJointId(name))
            if joint_id == 0:
                raise ValueError(f"URDF is missing passive joint {name!r}")
            joint = self.model.joints[joint_id]
            if joint.nq == 1:
                index = int(joint.idx_q)
                self.passive_q_indices[name] = index
                self.neutral[index] = float(value)
        self._mimic_map: tuple[tuple[int, int, float, float], ...] = (
            self._parse_mimic_map() if self._mimic_fallback else ()
        )
        if self._mimic_fallback:
            self._apply_mimic_map(self.neutral)

    def _parse_mimic_map(self) -> tuple[tuple[int, int, float, float], ...]:
        root = ET.parse(self.urdf_path).getroot()
        mappings: list[tuple[int, int, float, float]] = []
        for joint_elem in root.findall("joint"):
            mimic_elem = joint_elem.find("mimic")
            if mimic_elem is None:
                continue
            child_name = joint_elem.get("name") or ""
            parent_name = mimic_elem.get("joint") or ""
            if not child_name or not parent_name:
                continue
            child_id = int(self.model.getJointId(child_name))
            parent_id = int(self.model.getJointId(parent_name))
            if child_id == 0 or parent_id == 0:
                continue
            child_joint = self.model.joints[child_id]
            parent_joint = self.model.joints[parent_id]
            if child_joint.nq != 1 or parent_joint.nq != 1:
                continue
            multiplier = float(mimic_elem.get("multiplier", "1"))
            offset = float(mimic_elem.get("offset", "0"))
            mappings.append((int(child_joint.idx_q), int(parent_joint.idx_q), multiplier, offset))
        return tuple(mappings)

    def _apply_mimic_map(self, q: np.ndarray) -> None:
        for child_idx, parent_idx, multiplier, offset in self._mimic_map:
            q[child_idx] = multiplier * q[parent_idx] + offset

    def full_q(self, q_rad: np.ndarray, *, gripper_drive: float | None = None) -> np.ndarray:
        values = np.asarray(q_rad, dtype=np.float64).reshape(-1)
        if values.shape != (len(self.q_indices),) or not np.all(np.isfinite(values)):
            raise ValueError(f"joint vector must contain {len(self.q_indices)} finite values")
        q = self.neutral.copy()
        q[self.q_indices] = values
        if gripper_drive is not None:
            if len(self.passive_q_indices) != 1:
                raise ValueError("gripper_drive requires exactly one configured passive drive joint")
            q[next(iter(self.passive_q_indices.values()))] = float(gripper_drive)
        if self._mimic_fallback:
            self._apply_mimic_map(q)
        return q

    def xyz(self, q_rad: np.ndarray) -> np.ndarray:
        q = self.full_q(q_rad)
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        return np.asarray(self.data.oMf[self.ee_frame_id].translation, dtype=np.float64).copy()

    def pose(self, q_rad: np.ndarray) -> np.ndarray:
        q = self.full_q(q_rad)
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        placement = self.data.oMf[self.ee_frame_id]
        # Pinocchio Quaternion.coeffs() is [x, y, z, w].
        quaternion = np.asarray(self.pin.Quaternion(placement.rotation).coeffs(), dtype=np.float64).copy()
        quaternion /= np.linalg.norm(quaternion)
        return np.concatenate((np.asarray(placement.translation, dtype=np.float64), quaternion))


class PinocchioKinematicsBackend(_PinocchioModel):
    """Calibrated URDF FK and deterministic numerical xyz/full-pose IK."""

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        joint_names: tuple[str, ...],
        ee_link: str,
        max_iterations: int = 120,
        tolerance_m: float = 1e-7,
        orientation_tolerance_rad: float = 1e-5,
        damping: float = 1e-6,
        max_step_rad: float = 0.05,
        passive_joint_positions: dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            urdf_path,
            joint_names=joint_names,
            ee_link=ee_link,
            passive_joint_positions=passive_joint_positions,
        )
        self.max_iterations = int(max_iterations)
        self.tolerance_m = float(tolerance_m)
        self.orientation_tolerance_rad = float(orientation_tolerance_rad)
        self.damping = float(damping)
        self.max_step_rad = float(max_step_rad)
        if (
            self.max_iterations < 1
            or self.tolerance_m <= 0.0
            or self.orientation_tolerance_rad <= 0.0
            or self.damping <= 0.0
            or self.max_step_rad <= 0.0
        ):
            raise ValueError("IK iterations, tolerances, damping, and max step must be positive")

    def forward(self, q_rad: np.ndarray) -> np.ndarray:
        return self.pose(q_rad)

    def inverse(self, pose: np.ndarray, seed_q_rad: np.ndarray) -> np.ndarray:
        target = np.asarray(pose, dtype=np.float64).reshape(-1)
        if target.size not in (3, 7) or not np.all(np.isfinite(target)):
            raise ValueError("IK pose must contain finite xyz or xyz+quaternion_xyzw")
        q = np.asarray(seed_q_rad, dtype=np.float64).reshape(-1).copy()
        if q.shape != (len(self.joint_names),) or not np.all(np.isfinite(q)):
            raise ValueError("IK seed has invalid dimensions or values")
        if target.size == 7:
            return self._inverse_pose(target, q)
        return self._inverse_xyz(target, q)

    def _inverse_xyz(self, target: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Preserve the established position-only IK behavior for legacy callers."""

        epsilon = 1e-6
        damping = 1e-8
        for _ in range(self.max_iterations):
            xyz = self.xyz(q)
            error = target[:3] - xyz
            if float(np.linalg.norm(error)) <= self.tolerance_m:
                return q
            jacobian = np.empty((3, len(q)), dtype=np.float64)
            for index in range(len(q)):
                perturbed = q.copy()
                perturbed[index] += epsilon
                jacobian[:, index] = (self.xyz(perturbed) - xyz) / epsilon
            normal = jacobian @ jacobian.T + damping * np.eye(3)
            step = jacobian.T @ np.linalg.solve(normal, error)
            max_step = float(np.max(np.abs(step), initial=0.0))
            if max_step > 0.05:
                step *= 0.05 / max_step
            q += step
        residual = float(np.linalg.norm(target[:3] - self.xyz(q)))
        raise ValueError(f"IK did not converge after {self.max_iterations} iterations (residual={residual:.6g} m)")

    def _inverse_pose(self, target: np.ndarray, arm_q: np.ndarray) -> np.ndarray:
        """Solve a full link pose with a bounded damped least-squares SE(3) iteration."""

        quaternion = target[3:7].copy()
        quaternion_norm = float(np.linalg.norm(quaternion))
        if not math.isfinite(quaternion_norm) or quaternion_norm < 1e-12:
            raise ValueError("IK quaternion must have non-zero finite norm")
        quaternion /= quaternion_norm
        rotation = self.pin.Quaternion(
            float(quaternion[3]),
            float(quaternion[0]),
            float(quaternion[1]),
            float(quaternion[2]),
        ).matrix()
        desired = self.pin.SE3(rotation, target[:3])
        q = self.full_q(arm_q)
        position_error = float("inf")
        orientation_error = float("inf")
        for _ in range(self.max_iterations):
            self.pin.forwardKinematics(self.model, self.data, q)
            self.pin.updateFramePlacements(self.model, self.data)
            current = self.data.oMf[self.ee_frame_id]
            current_to_desired = current.actInv(desired)
            position_error = float(np.linalg.norm(current.translation - desired.translation))
            orientation_error = float(np.linalg.norm(self.pin.log3(current.rotation.T @ desired.rotation)))
            if position_error <= self.tolerance_m and orientation_error <= self.orientation_tolerance_rad:
                return np.asarray(q[self.q_indices], dtype=np.float64).copy()
            error = np.asarray(self.pin.log6(current_to_desired).vector, dtype=np.float64)
            jacobian = self.pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                self.ee_frame_id,
                self.pin.ReferenceFrame.LOCAL,
            )
            jacobian = -self.pin.Jlog6(current_to_desired.inverse()) @ jacobian
            arm_jacobian = np.asarray(jacobian[:, self.v_indices], dtype=np.float64)
            # Keep the specified 1e-6 nominal damping.  Close to a spherical-
            # wrist singularity, increase only the normal-matrix regularizer
            # (capped at 1e-4) so equivalent wrist splits do not alternate
            # between adjacent shadow samples.
            smallest_singular = float(np.min(np.linalg.svd(arm_jacobian, compute_uv=False)))
            singular_scale = min(10.0, max(1.0, 1e-2 / max(smallest_singular, 1e-12)))
            regularizer = self.damping * singular_scale**2
            normal = arm_jacobian @ arm_jacobian.T + regularizer * np.eye(6)
            step = -arm_jacobian.T @ np.linalg.solve(normal, error)
            max_step = float(np.max(np.abs(step), initial=0.0))
            if max_step > self.max_step_rad:
                step *= self.max_step_rad / max_step
            velocity = np.zeros(self.model.nv, dtype=np.float64)
            velocity[self.v_indices] = step
            q = self.pin.integrate(self.model, q, velocity)
        raise ValueError(
            f"full-pose IK did not converge after {self.max_iterations} iterations "
            f"(position_residual={position_error:.6g} m, orientation_residual={orientation_error:.6g} rad)"
        )


def urdf_subtree_links(urdf_root: ET.Element, root_link: str) -> frozenset[str]:
    """``root_link`` and every URDF link below it."""
    children: dict[str, list[str]] = {}
    for joint in urdf_root.findall("joint"):
        parent, child = joint.find("parent"), joint.find("child")
        if parent is not None and child is not None:
            children.setdefault(parent.get("link", ""), []).append(child.get("link", ""))
    links: set[str] = set()
    stack = [root_link]
    while stack:
        link = stack.pop()
        if link not in links:
            links.add(link)
            stack.extend(children.get(link, ()))
    return frozenset(links)


class PinocchioCollisionBackend(_PinocchioModel):
    """Full-sample self/environment collision and minimum-distance checks.

    ``tool_root_link`` (the arm flange link) skips every pair whose two links both lie in the URDF subtree rooted at
    it: the flange link, camera stand, cameras, and end effector. That subtree is fixed to the flange, so the relative
    poses inside it depend only on the gripper drive, never on the arm trajectory; whether those parts clear each
    other over the full stroke is an asset property, verified by ``tests/robots/test_tool_internal_clearance.py``.
    Pairs between the subtree and the arm or the environment remain checked.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        joint_names: tuple[str, ...],
        ee_link: str,
        obstacles: Iterable[EnvironmentObstacle] = (),
        passive_joint_positions: dict[str, float] | None = None,
        tool_root_link: str | None = None,
    ) -> None:
        super().__init__(
            urdf_path,
            joint_names=joint_names,
            ee_link=ee_link,
            passive_joint_positions=passive_joint_positions,
        )
        try:
            import coal
        except ImportError as exc:
            raise RuntimeError("Coal/HPP-FCL is required for safety preflight") from exc
        self.coal = coal
        pin = self.pin
        self.geometry_model = pin.buildGeomFromUrdf(
            self.model,
            str(self.urdf_path),
            pin.GeometryType.COLLISION,
            package_dirs=[str(self.urdf_path.parent)],
        )
        self.geometry_model.addAllCollisionPairs()
        urdf_root = ET.parse(self.urdf_path).getroot()
        adjacent_links = {
            frozenset((parent.get("link", ""), child.get("link", "")))
            for joint in urdf_root.findall("joint")
            if (parent := joint.find("parent")) is not None and (child := joint.find("child")) is not None
        }
        self.tool_links: frozenset[str] = frozenset()
        if tool_root_link is not None:
            if urdf_root.find(f"./link[@name='{tool_root_link}']") is None:
                raise ValueError(f"tool root link {tool_root_link!r} missing from URDF")
            self.tool_links = urdf_subtree_links(urdf_root, tool_root_link)
        # Parent-child geometry intersects at the joint by construction.
        remove_pairs: list[tuple[int, int]] = []
        for pair in self.geometry_model.collisionPairs:
            first = self.geometry_model.geometryObjects[pair.first]
            second = self.geometry_model.geometryObjects[pair.second]
            joint_a, joint_b = int(first.parentJoint), int(second.parentJoint)
            link_a = str(self.model.frames[int(first.parentFrame)].name)
            link_b = str(self.model.frames[int(second.parentFrame)].name)
            adjacent = (
                joint_a == joint_b
                or int(self.model.parents[joint_a]) == joint_b
                or int(self.model.parents[joint_b]) == joint_a
                or frozenset((link_a, link_b)) in adjacent_links
            )
            if adjacent or (link_a in self.tool_links and link_b in self.tool_links):
                remove_pairs.append((int(pair.first), int(pair.second)))
        for first, second in remove_pairs:
            self.geometry_model.removeCollisionPair(pin.CollisionPair(first, second))
        self.environment_indices: set[int] = set()
        for obstacle in obstacles:
            placement = pin.SE3(np.eye(3), np.asarray(obstacle.center_m, dtype=np.float64))
            geometry = pin.GeometryObject(
                obstacle.name,
                0,
                0,
                placement,
                coal.Box(*obstacle.size_m),
            )
            obstacle_index = int(self.geometry_model.addGeometryObject(geometry))
            self.environment_indices.add(obstacle_index)
            for robot_index in range(obstacle_index):
                if robot_index not in self.environment_indices:
                    self.geometry_model.addCollisionPair(pin.CollisionPair(robot_index, obstacle_index))
        self.geometry_data = self.geometry_model.createData()

    @property
    def collision_pair_count(self) -> int:
        return len(self.geometry_model.collisionPairs)

    def _link_name(self, geometry_index: int) -> str:
        geometry = self.geometry_model.geometryObjects[geometry_index]
        if geometry_index in self.environment_indices:
            return str(geometry.name)
        parent_frame = int(geometry.parentFrame)
        if 0 <= parent_frame < len(self.model.frames):
            return str(self.model.frames[parent_frame].name)
        return str(geometry.name)

    def check(self, q_rad: np.ndarray, *, stage: str, gripper_drive: float | None = None) -> CollisionResult:
        results = self.check_all(q_rad, stage=stage, gripper_drive=gripper_drive)
        if not results:
            return CollisionResult(False, float(np.finfo(np.float64).max))
        return min(results, key=lambda result: result.min_distance_m)

    def check_all(
        self, q_rad: np.ndarray, *, stage: str, gripper_drive: float | None = None
    ) -> tuple[CollisionResult, ...]:
        del stage  # stage is interpreted by safety preflight, not this adapter.
        q = self.full_q(q_rad, gripper_drive=gripper_drive)
        self.pin.computeCollisions(
            self.model,
            self.data,
            self.geometry_model,
            self.geometry_data,
            q,
            False,
        )
        self.pin.computeDistances(self.geometry_model, self.geometry_data)
        return tuple(
            CollisionResult(
                colliding=bool(self.geometry_data.collisionResults[index].isCollision()),
                min_distance_m=float(self.geometry_data.distanceResults[index].min_distance),
                link_a=self._link_name(pair.first),
                link_b=self._link_name(pair.second),
                environment=pair.first in self.environment_indices or pair.second in self.environment_indices,
                geometry_a=str(self.geometry_model.geometryObjects[pair.first].name),
                geometry_b=str(self.geometry_model.geometryObjects[pair.second].name),
            )
            for index, pair in enumerate(self.geometry_model.collisionPairs)
        )

    def check_all_within_margin(
        self,
        q_rad: np.ndarray,
        *,
        stage: str,
        margin_m: float,
        gripper_drive: float | None = None,
    ) -> tuple[CollisionResult, ...]:
        """Return only geometry pairs at or inside ``margin_m``.

        Coal can test a positive security margin as part of its collision
        traversal.  That is substantially cheaper than computing an exact
        distance for every geometry pair.  A zero-margin batch query preserves
        exact collision classification, and exact distance is then computed
        for every candidate so callers retain the established
        :class:`CollisionResult` semantics.

        The shared request objects are restored before returning because
        ``check_all`` must remain an exact, zero-margin query for consumers
        that need the complete distance set.
        """

        del stage  # stage is interpreted by safety preflight, not this adapter.
        margin = float(margin_m)
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError("collision margin must be finite and non-negative")

        q = self.full_q(q_rad, gripper_drive=gripper_drive)
        requests = self.geometry_data.collisionRequests
        previous_margins = [float(request.security_margin) for request in requests]
        try:
            for request in requests:
                request.security_margin = margin
            self.pin.computeCollisions(
                self.model,
                self.data,
                self.geometry_model,
                self.geometry_data,
                q,
                False,
            )
            candidate_indices = [
                index for index, result in enumerate(self.geometry_data.collisionResults) if bool(result.isCollision())
            ]

            # Restore a zero-margin collision query before materializing the
            # candidates.  Pinocchio's batch collision semantics distinguish
            # exact touching from penetration slightly differently from a
            # standalone ``coal.collide`` call, so this second inexpensive
            # batch preserves the established ``colliding`` classification.
            for request in requests:
                request.security_margin = 0.0
            self.pin.computeCollisions(
                self.model,
                self.data,
                self.geometry_model,
                self.geometry_data,
                q,
                False,
            )
            exact_collision_flags = {
                index: bool(self.geometry_data.collisionResults[index].isCollision()) for index in candidate_indices
            }

            distance_request = self.coal.DistanceRequest()
            results: list[CollisionResult] = []
            for index in candidate_indices:
                pair = self.geometry_model.collisionPairs[index]
                first = self.geometry_model.geometryObjects[pair.first]
                second = self.geometry_model.geometryObjects[pair.second]
                first_placement = self.geometry_data.oMg[pair.first]
                second_placement = self.geometry_data.oMg[pair.second]

                distance_result = self.coal.DistanceResult()
                min_distance = float(
                    self.coal.distance(
                        first.geometry,
                        first_placement,
                        second.geometry,
                        second_placement,
                        distance_request,
                        distance_result,
                    )
                )
                if not math.isfinite(min_distance):
                    raise RuntimeError("Coal returned a non-finite candidate distance")
                results.append(
                    CollisionResult(
                        colliding=exact_collision_flags[index],
                        min_distance_m=min_distance,
                        link_a=self._link_name(pair.first),
                        link_b=self._link_name(pair.second),
                        environment=pair.first in self.environment_indices or pair.second in self.environment_indices,
                        geometry_a=str(first.name),
                        geometry_b=str(second.name),
                    )
                )
            return tuple(results)
        finally:
            for request, previous_margin in zip(requests, previous_margins, strict=True):
                request.security_margin = previous_margin


class StageAwareObjectCollisionBackend:
    """Relocate the pick-place cube obstacle after the object has been moved.

    Preflight keeps a single static ``object`` box. After place, the real cube is
    at the target, but a spawn-only model falsely flags the return-to-home path
    (especially the legacy y+ place layout). Stages listed in
    ``relocated_stages`` use ``place_center_m``; all others use ``spawn_center_m``.
    """

    def __init__(
        self,
        backend: PinocchioCollisionBackend,
        *,
        object_name: str = "object",
        spawn_center_m: tuple[float, float, float],
        place_center_m: tuple[float, float, float],
        relocated_stages: tuple[str, ...] = (
            "place-descend",
            "place-settle",
            "release",
            "retreat",
            "return-home",
        ),
    ) -> None:
        self._backend = backend
        self._object_name = str(object_name)
        self._spawn = np.asarray(spawn_center_m, dtype=np.float64).reshape(3)
        self._place = np.asarray(place_center_m, dtype=np.float64).reshape(3)
        self._relocated_stages = frozenset(str(stage) for stage in relocated_stages)
        self._object_index = self._find_object_index()
        self._current = "spawn"
        self._set_center(self._spawn)

    @property
    def collision_pair_count(self) -> int:
        return self._backend.collision_pair_count

    def _find_object_index(self) -> int:
        for index in self._backend.environment_indices:
            if str(self._backend.geometry_model.geometryObjects[index].name) == self._object_name:
                return int(index)
        raise ValueError(f"environment obstacle {self._object_name!r} not found")

    def _set_center(self, center: np.ndarray) -> None:
        pin = self._backend.pin
        geometry = self._backend.geometry_model.geometryObjects[self._object_index]
        geometry.placement = pin.SE3(np.eye(3), np.asarray(center, dtype=np.float64))

    def _apply_stage(self, stage: str) -> None:
        key = "place" if stage in self._relocated_stages else "spawn"
        if key == self._current:
            return
        self._set_center(self._place if key == "place" else self._spawn)
        self._current = key

    def check(self, q_rad: np.ndarray, *, stage: str, gripper_drive: float | None = None) -> CollisionResult:
        results = self.check_all(q_rad, stage=stage, gripper_drive=gripper_drive)
        if not results:
            return CollisionResult(False, float(np.finfo(np.float64).max))
        return min(results, key=lambda result: result.min_distance_m)

    def check_all(
        self, q_rad: np.ndarray, *, stage: str, gripper_drive: float | None = None
    ) -> tuple[CollisionResult, ...]:
        self._apply_stage(stage)
        return self._backend.check_all(q_rad, stage=stage, gripper_drive=gripper_drive)

    def check_all_within_margin(
        self,
        q_rad: np.ndarray,
        *,
        stage: str,
        margin_m: float,
        gripper_drive: float | None = None,
    ) -> tuple[CollisionResult, ...]:
        self._apply_stage(stage)
        return self._backend.check_all_within_margin(
            q_rad,
            stage=stage,
            margin_m=margin_m,
            gripper_drive=gripper_drive,
        )
