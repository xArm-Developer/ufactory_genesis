"""Robot profile resolution tests."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from ufactory.robots.paths import (
    lite6_gripper_movable_visual_urdf,
    robot_urdf,
    robot_visual_glb_urdf,
    vacuum_gripper_urdf,
    xarm5_urdf,
    xarm6_urdf,
    xarm7_urdf,
)
from ufactory.robots.registry import (
    PROJECT_ROOT,
    get_profile_key_for_robot_name,
    get_robot_profile,
    robot_cli_choices,
    robot_name_for_controller,
)


@pytest.mark.parametrize(
    ("name", "expected_key"),
    [
        ("xarm5", "xarm5_1305"),
        ("xarm6", "xarm6_1305"),
        ("xarm7", "xarm7_1305"),
        ("xarm5_1305", "xarm5_1305"),
        ("xarm6_1305", "xarm6_1305"),
        ("xarm7_1305", "xarm7_1305"),
        ("lite6", "lite6"),
        ("uf850", "uf850"),
    ],
)
def test_get_profile_key_for_robot_name(name: str, expected_key: str) -> None:
    assert get_profile_key_for_robot_name(name) == expected_key


def test_get_robot_profile_accepts_short_names() -> None:
    profile = get_robot_profile("xarm6")
    assert profile.key == "xarm6_1305"
    assert profile.default_urdf == "xarm6_1305.urdf"


@pytest.mark.parametrize(
    ("axis", "device_type", "expected"),
    [
        (5, 5, "xarm5"),
        (6, 6, "xarm6"),
        (7, 7, "xarm7"),
        (6, 9, "lite6"),
        (6, 12, "uf850"),
        (6, 99, None),
        (7, 9, None),
    ],
)
def test_robot_name_for_controller(axis: int, device_type: int, expected: str | None) -> None:
    # (axis, device_type) pairs read from the five supported controllers (firmware v2.8.x).
    assert robot_name_for_controller(axis, device_type) == expected


def test_xarm_urdf_defaults_point_to_1305() -> None:
    for path_fn, suffix in (
        (xarm5_urdf, "xarm5_1305.urdf"),
        (xarm6_urdf, "xarm6_1305.urdf"),
        (xarm7_urdf, "xarm7_1305.urdf"),
    ):
        path = Path(path_fn())
        assert path.name == suffix
        assert path.is_file()


def test_robot_urdf_short_name() -> None:
    assert Path(robot_urdf("xarm6")).name == "xarm6_1305.urdf"


def test_robot_cli_choices_includes_aliases() -> None:
    choices = robot_cli_choices()
    assert "xarm6" in choices
    assert "xarm6_1305" in choices


def test_lite6_movable_gripper_assembly_matches_standalone_part() -> None:
    combo_path = Path(robot_visual_glb_urdf("lite6", with_lite6_gripper=True, movable=True))
    standalone_path = Path(lite6_gripper_movable_visual_urdf())

    assert combo_path.name.startswith("lite6__lite6_gripper__glb.")
    finger_mesh_dir = PROJECT_ROOT / "assets" / "urdf" / "end_effectors" / "lite6_gripper" / "meshes" / "collision"

    for path in (combo_path, standalone_path):
        root = ET.parse(path).getroot()
        joints = {joint.get("name"): joint for joint in root.findall("joint")}
        assert joints["finger_joint1"].find("origin").get("xyz") == "0 0 0.0543"
        assert joints["finger_joint2"].find("origin").get("xyz") == "0 0 0.0543"
        finger_visual_meshes = {
            link.get("name"): (path.parent / link.find("./visual/geometry/mesh").get("filename")).resolve()
            for link in root.findall("link")
            if link.get("name") in {"uflite_finger1", "uflite_finger2"}
        }
        assert finger_visual_meshes == {
            "uflite_finger1": finger_mesh_dir / "finger1.stl",
            "uflite_finger2": finger_mesh_dir / "finger2.stl",
        }
        expected_boxes = {
            "uflite_finger1": [
                ("0 0.00775 0.002", "0.02 0.0155 0.0095"),
                ("0 0.013375 0.016875", "0.02 0.00425 0.01975"),
            ],
            "uflite_finger2": [
                ("0 -0.00775 0.002", "0.02 0.0155 0.0095"),
                ("0 -0.013375 0.016875", "0.02 0.00425 0.01975"),
            ],
        }
        for link_name, boxes in expected_boxes.items():
            link = root.find(f".//link[@name='{link_name}']")
            assert link is not None
            visual_origin = link.find("./visual/origin")
            assert visual_origin is not None
            assert visual_origin.get("xyz") == "0 0 0"
            collisions = link.findall("collision")
            assert len(collisions) == 2
            actual = []
            for collision in collisions:
                assert collision.find("./geometry/mesh") is None
                box = collision.find("./geometry/box")
                assert box is not None
                actual.append((collision.find("origin").get("xyz"), box.get("size")))
            assert actual == boxes


def test_vacuum_gripper_urdf_resolution() -> None:
    standalone = Path(vacuum_gripper_urdf())
    assert standalone.name == "vacuum_gripper.urdf"
    assert standalone.is_file()

    for robot, key in {"xarm5": "xarm5_1305", "xarm6": "xarm6_1305", "xarm7": "xarm7_1305", "uf850": "uf850"}.items():
        combo = Path(robot_visual_glb_urdf(robot, with_vacuum_gripper=True))
        assert combo.name.startswith(f"{key}__vacuum_gripper__glb.")
        assert combo.is_file()
        mount = ET.parse(combo).getroot().find("./joint[@name='vacuum_gripper_fix']")
        assert mount.find("parent").get("link") == get_robot_profile(robot).ee_link

    with pytest.raises(ValueError, match="does not support Vacuum Gripper"):
        robot_visual_glb_urdf("lite6", with_vacuum_gripper=True)

    with pytest.raises(ValueError, match="movable=True requires"):
        robot_visual_glb_urdf("xarm6", with_vacuum_gripper=True, movable=True)

    with pytest.raises(ValueError, match="mutually exclusive"):
        robot_visual_glb_urdf("xarm6", with_gripper_g2=True, with_vacuum_gripper=True)
