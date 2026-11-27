"""Minimal robot catalog and source-asset access API."""

from ufactory.config.assets import AssetLayoutError, RepositoryAssetStore
from ufactory.control import ArmMotionController
from ufactory.robots.registry import (
    PROJECT_ROOT,
    ROBOT_PROFILES,
    RobotModelSpec,
    get_robot_profile,
    robot_cli_choices,
)
from ufactory.robots.assembly import compose_robot_urdf, supported_end_effectors
from ufactory.robots.paths import robot_assets, robot_urdf, robot_visual_glb_urdf

__all__ = [
    "PROJECT_ROOT",
    "ROBOT_PROFILES",
    "RobotModelSpec",
    "ArmMotionController",
    "AssetLayoutError",
    "compose_robot_urdf",
    "RepositoryAssetStore",
    "get_robot_profile",
    "robot_assets",
    "robot_cli_choices",
    "robot_urdf",
    "robot_visual_glb_urdf",
    "supported_end_effectors",
]
