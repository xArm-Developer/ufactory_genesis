"""Lightweight checks for public robot asset consistency."""

from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))
from conftest import PROJECT_ROOT


def _public_files() -> set[Path]:
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=PROJECT_ROOT).split(b"\0")[:-1]
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=PROJECT_ROOT,
    ).split(b"\0")[:-1]
    return {PROJECT_ROOT / p.decode() for p in (*tracked, *untracked) if (PROJECT_ROOT / p.decode()).exists()}


def test_public_urdf_mesh_references_exist() -> None:
    urdfs = sorted(p for p in _public_files() if p.suffix == ".urdf" or p.name.endswith(".glb.urdf"))
    missing: list[str] = []
    for urdf in urdfs:
        root = ET.parse(urdf).getroot()
        for mesh in root.findall(".//mesh"):
            filename = mesh.get("filename")
            if not filename or filename.startswith("package://"):
                continue
            target = (urdf.parent / filename).resolve()
            if not target.exists():
                missing.append(f"{urdf.relative_to(PROJECT_ROOT)} -> {filename}")

    assert missing == []


def test_standard_g2_urdfs_share_joint_limits_and_mimic_linkage() -> None:
    from ufactory.grippers.g2 import (
        G2_JOINT_NAMES,
        G2_LINKAGE_GEOMETRY,
        GRIPPER_G2_SIM_OPEN_DRIVE,
        GRIPPER_G2_SIM_CLOSE_DRIVE,
        GRIPPER_G2_JOINT_VELOCITY_RAD_S,
    )

    models = []
    for path in sorted((PROJECT_ROOT / "assets/urdf").rglob("*.urdf")):
        root = ET.parse(path).getroot()
        if root.find("./joint[@name='drive_joint']") is not None:
            models.append((path, root))
    assert len(models) == 11
    for path, root in models:
        for name in G2_JOINT_NAMES:
            joint = root.find(f"./joint[@name='{name}']")
            assert joint is not None, (path, name)
            limit = joint.find("limit")
            assert limit is not None, (path, name)
            assert float(limit.attrib["lower"]) == GRIPPER_G2_SIM_OPEN_DRIVE, path
            assert float(limit.attrib["upper"]) == GRIPPER_G2_SIM_CLOSE_DRIVE, path
            assert float(limit.attrib["velocity"]) == GRIPPER_G2_JOINT_VELOCITY_RAD_S, path
            if name != "drive_joint":
                mimic = joint.find("mimic")
                assert mimic is not None, (path, name)
                assert mimic.get("joint") == "drive_joint", (path, name)
                assert float(mimic.get("multiplier", "1")) == 1, (path, name)
                assert float(mimic.get("offset", "0")) == 0, (path, name)
    assert (
        GRIPPER_G2_JOINT_VELOCITY_RAD_S * G2_LINKAGE_GEOMETRY.gap_jacobian_m_per_rad(GRIPPER_G2_SIM_OPEN_DRIVE) >= 0.225
    )


def test_standard_g2_urdf_joint_origins_and_mesh_alignment() -> None:
    """Standard Gripper G2 URDFs and meshes must match the SolidWorks STEP AG1200 calibration."""
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("DracoPy")
    pytest.importorskip("pygltflib")
    from ufactory.grippers.g2 import gripper_g2_gap_m_to_sim_drive

    expected_joint_origins = {
        "drive_joint": (0.0, 0.035, 0.0606),
        "left_finger_joint": (0.0, 0.035465, 0.042039),
        "left_inner_knuckle_joint": (0.0, 0.02, 0.075602),
        "right_outer_knuckle_joint": (0.0, -0.035, 0.0606),
        "right_finger_joint": (0.0, -0.035465, 0.042039),
        "right_inner_knuckle_joint": (0.0, -0.02, 0.075602),
        "joint_tcp": (0.0, 0.0, 0.1735),
    }
    g2_links = {
        "gripper_g2_visual",
        "xarm_gripper_base_link",
        "left_outer_knuckle",
        "left_finger",
        "left_inner_knuckle",
        "right_outer_knuckle",
        "right_finger",
        "right_inner_knuckle",
    }

    models = []
    for path in sorted((PROJECT_ROOT / "assets/urdf").rglob("*.urdf")):
        root = ET.parse(path).getroot()
        if root.find("./joint[@name='drive_joint']") is not None:
            models.append((path, root))
    assert len(models) == 11

    for path, root in models:
        for jname, expected_xyz in expected_joint_origins.items():
            joint = root.find(f"./joint[@name='{jname}']")
            assert joint is not None, (path, jname)
            origin = joint.find("origin")
            assert origin is not None, (path, jname)
            assert _rounded_xyz(origin.get("xyz")) == expected_xyz, (path, jname)
            assert _rounded_xyz(origin.get("rpy")) == (0.0, 0.0, 0.0), (path, jname)
        for link in root.findall("link"):
            lname = link.get("name") or ""
            if lname not in g2_links:
                continue
            for tag in ("visual", "collision"):
                for node in link.findall(tag):
                    orig = node.find("origin")
                    if orig is not None:
                        assert _rounded_xyz(orig.get("xyz")) == (0.0, 0.0, 0.0), (path, lname, tag)
                        assert _rounded_xyz(orig.get("rpy")) == (0.0, 0.0, 0.0), (path, lname, tag)

    pkg = PROJECT_ROOT / "assets" / "urdf" / "gripper_g2" / "meshes"
    base_stl = _mesh_vertices(pkg / "collision" / "base_link.stl", trimesh)
    base_glb = _mesh_vertices(pkg / "visual" / "visual_glb" / "base.glb", trimesh)
    static_glb = _mesh_vertices(pkg / "visual" / "gripper_g2_static.glb", trimesh)
    lf_glb = _mesh_vertices(pkg / "visual" / "visual_glb" / "left_finger.glb", trimesh)
    rf_glb = _mesh_vertices(pkg / "visual" / "visual_glb" / "right_finger.glb", trimesh)

    assert base_stl[:, 2].min() * 1000.0 == pytest.approx(-5.0, abs=0.01)
    assert base_stl[:, 2].max() * 1000.0 == pytest.approx(104.5, abs=0.01)
    assert base_glb[:, 2].min() * 1000.0 == pytest.approx(-5.0, abs=0.01)
    assert base_glb[:, 2].max() * 1000.0 == pytest.approx(104.5, abs=0.01)
    assert static_glb[:, 2].min() * 1000.0 == pytest.approx(-5.0, abs=0.01)
    assert static_glb[:, 2].max() * 1000.0 == pytest.approx(176.323, abs=0.01)

    for gap_mm, expected_axial_mm in (
        (0.0, 175.86527),
        (18.0, 176.60609),
        (29.1558585, 176.32219),
        (84.0, 165.60379),
    ):
        q = gripper_g2_gap_m_to_sim_drive(gap_mm / 1000.0)
        c, s = float(np.cos(q)), float(np.sin(q))
        y_fj = 0.035465 * c - 0.042039 * s
        z_fj = 0.035465 * s + 0.042039 * c
        wl = lf_glb + np.array([0.0, 0.035 + y_fj, 0.0606 + z_fj])
        wr = rf_glb + np.array([0.0, -0.035 - y_fj, 0.0606 + z_fj])
        actual_gap_mm = float((wl[:, 1].min() - wr[:, 1].max()) * 1000.0)
        actual_axial_mm = float(max(wl[:, 2].max(), wr[:, 2].max()) * 1000.0)
        assert actual_gap_mm == pytest.approx(gap_mm, abs=0.01)
        assert actual_axial_mm == pytest.approx(expected_axial_mm, abs=0.01)


def test_intermediate_assets_and_dev_scripts_are_not_public_files() -> None:
    forbidden_markers = (
        "visual_glb_raw",
        "visual_glb_src",
        "relocalize_metrics.json",
        "scripts/relocalize_",
        "scripts/diagnose_",
        "scripts/vendor_robot_assets.py",
        "bio_gripper_g2/meshes/visual/link",
        "bio_gripper_g2_visual_link",
        "gripper_g2_static_link",
    )
    forbidden_script_shapes = (
        ("scripts/generate_", "_combo_urdf.py"),
        ("scripts/verify_", "_assets.py"),
    )
    public = [str(p.relative_to(PROJECT_ROOT)) for p in _public_files()]
    leaked = [
        p
        for p in public
        if any(marker in p for marker in forbidden_markers)
        or any(p.startswith(prefix) and p.endswith(suffix) for prefix, suffix in forbidden_script_shapes)
    ]
    assert leaked == []


_REGISTERED_ARM_URDFS: dict[str, tuple[int, str, tuple[str, ...]]] = {
    "xarm5": (
        5,
        "xarm5_1305",
        (
            "xarm5_1305.urdf",
            "xarm5_1305_visual.glb.urdf",
            "xarm5_1305_g2_visual.urdf",
            "xarm5_1305_g2_movable_visual.urdf",
            "xarm5_1305_bio_gripper_g2_visual.glb.urdf",
            "xarm5_1305_bio_gripper_g2_movable_visual.glb.urdf",
            "xarm5_1305_vacuum_gripper_visual.glb.urdf",
        ),
    ),
    "xarm6": (
        6,
        "xarm6_1305",
        (
            "xarm6_1305.urdf",
            "xarm6_1305_visual.glb.urdf",
            "xarm6_1305_g2_visual.urdf",
            "xarm6_1305_g2_movable_visual.urdf",
            "xarm6_1305_bio_gripper_g2_visual.glb.urdf",
            "xarm6_1305_bio_gripper_g2_movable_visual.glb.urdf",
            "xarm6_1305_vacuum_gripper_visual.glb.urdf",
            "xarm6_with_gripper.urdf",
        ),
    ),
    "xarm7": (
        7,
        "xarm7_1305",
        (
            "xarm7_1305.urdf",
            "xarm7_1305_visual.glb.urdf",
            "xarm7_1305_g2_visual.urdf",
            "xarm7_1305_g2_movable_visual.urdf",
            "xarm7_1305_bio_gripper_g2_visual.glb.urdf",
            "xarm7_1305_bio_gripper_g2_movable_visual.glb.urdf",
            "xarm7_1305_vacuum_gripper_visual.glb.urdf",
        ),
    ),
    "lite6": (
        6,
        "lite6",
        (
            "lite6.urdf",
            "lite6_visual.glb.urdf",
            "lite6_gripper_visual.glb.urdf",
            "lite6_gripper_movable_visual.glb.urdf",
            "lite6_gripper_reversed_movable_visual.glb.urdf",
            "lite6_vacuum_gripper_visual.glb.urdf",
            "lite6_with_gripper.urdf",
            "lite6_with_vacuum_gripper.urdf",
        ),
    ),
    "uf850": (
        6,
        "uf850",
        (
            "uf850.urdf",
            "uf850_visual.glb.urdf",
            "uf850_g2_visual.urdf",
            "uf850_g2_movable_visual.urdf",
            "uf850_bio_gripper_g2_visual.glb.urdf",
            "uf850_bio_gripper_g2_movable_visual.glb.urdf",
            "uf850_vacuum_gripper_visual.glb.urdf",
        ),
    ),
}

_BIO_GRIPPER_G2_LINKS = (
    "bio_gripper_g2_base_link",
    "bio_gripper_g2_left_finger",
    "bio_gripper_g2_right_finger",
)

# Calibrated Bio Gripper G2 convex collision STLs are already expressed in each link frame (origin 0 0 0).
_BIO_GRIPPER_G2_COLLISION_ORIGINS = {
    "bio_gripper_g2_base_link": (0.0, 0.0, 0.0),
    "bio_gripper_g2_left_finger": (0.0, 0.0, 0.0),
    "bio_gripper_g2_right_finger": (0.0, 0.0, 0.0),
}

_BIO_GRIPPER_G2_COLLISION_ORIGIN_URDFS = (
    PROJECT_ROOT / "assets" / "urdf" / "bio_gripper_g2" / "bio_gripper_g2.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "bio_gripper_g2" / "bio_gripper_g2_movable_visual.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "xarm5" / "xarm5_1305_bio_gripper_g2_visual.glb.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "xarm5" / "xarm5_1305_bio_gripper_g2_movable_visual.glb.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "xarm6" / "xarm6_1305_bio_gripper_g2_visual.glb.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "xarm6" / "xarm6_1305_bio_gripper_g2_movable_visual.glb.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "xarm7" / "xarm7_1305_bio_gripper_g2_visual.glb.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "xarm7" / "xarm7_1305_bio_gripper_g2_movable_visual.glb.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "uf850" / "uf850_bio_gripper_g2_visual.glb.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "uf850" / "uf850_bio_gripper_g2_movable_visual.glb.urdf",
)

_BIO_GRIPPER_G2_MOVABLE_COLLISION_ALIGNMENT_URDFS = (
    PROJECT_ROOT / "assets" / "urdf" / "bio_gripper_g2" / "bio_gripper_g2_movable_visual.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "xarm5" / "xarm5_1305_bio_gripper_g2_movable_visual.glb.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "xarm6" / "xarm6_1305_bio_gripper_g2_movable_visual.glb.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "xarm7" / "xarm7_1305_bio_gripper_g2_movable_visual.glb.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "uf850" / "uf850_bio_gripper_g2_movable_visual.glb.urdf",
)


def _arm_link_names(dof: int) -> set[str]:
    return {"link_base", *(f"link{i}" for i in range(1, dof + 1))}


def _xyz_tuple(value: str | None) -> tuple[float, float, float]:
    parts = tuple(float(part) for part in (value or "0 0 0").split())
    if len(parts) != 3:
        raise ValueError(f"expected xyz/rpy triplet, got {value!r}")
    return parts


def _rounded_xyz(value: str | None) -> tuple[float, float, float]:
    return tuple(round(part, 6) for part in _xyz_tuple(value))


def _mesh_vertices(mesh_path: Path, trimesh_module) -> np.ndarray:
    """Load mesh vertices, including Draco-compressed GLB visuals."""
    if mesh_path.suffix.lower() == ".glb":
        try:
            from pygltflib import GLTF2
            import DracoPy
        except ImportError:
            DracoPy = None  # type: ignore[assignment]
            GLTF2 = None  # type: ignore[assignment]
        if GLTF2 is not None and DracoPy is not None:
            glb = GLTF2().load(str(mesh_path))
            if glb.extensionsUsed and "KHR_draco_mesh_compression" in glb.extensionsUsed:
                blob = glb.binary_blob() or b""
                chunks: list[np.ndarray] = []
                for mesh in glb.meshes or []:
                    for prim in mesh.primitives or []:
                        extensions = prim.extensions or {}
                        draco = extensions.get("KHR_draco_mesh_compression")
                        if not draco:
                            continue
                        view = glb.bufferViews[draco["bufferView"]]
                        offset = view.byteOffset or 0
                        decoded = DracoPy.decode(blob[offset : offset + view.byteLength])
                        chunks.append(np.asarray(decoded.points, dtype=np.float64))
                if chunks:
                    return np.vstack(chunks)

    scene = trimesh_module.load(mesh_path, force="scene", process=False)
    vertices: list[np.ndarray] = []
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph.get(node_name)
        geometry = scene.geometry[geometry_name]
        local = np.asarray(geometry.vertices, dtype=np.float64)
        vertices.append(local @ transform[:3, :3].T + transform[:3, 3])
    if vertices:
        return np.vstack(vertices)
    mesh = trimesh_module.load(mesh_path, process=False)
    return np.asarray(mesh.vertices, dtype=np.float64)


def _collision_mesh_filenames(urdf: Path) -> list[str]:
    root = ET.parse(urdf).getroot()
    filenames: list[str] = []
    for collision in root.findall(".//collision"):
        mesh = collision.find(".//mesh")
        if mesh is not None and mesh.get("filename"):
            filenames.append(mesh.get("filename") or "")
    return filenames


def _mesh_filenames(urdf: Path) -> list[str]:
    root = ET.parse(urdf).getroot()
    return [mesh.get("filename") or "" for mesh in root.findall(".//mesh") if mesh.get("filename")]


def _collision_meshes_by_link(urdf: Path) -> dict[str, list[str]]:
    root = ET.parse(urdf).getroot()
    out: dict[str, list[str]] = {}
    for link in root.findall("link"):
        link_name = link.get("name") or ""
        for collision in link.findall("collision"):
            mesh = collision.find(".//mesh")
            if mesh is not None and mesh.get("filename"):
                out.setdefault(link_name, []).append(mesh.get("filename") or "")
    return out


def _collision_links(urdf: Path) -> set[str]:
    root = ET.parse(urdf).getroot()
    return {
        link.get("name") or ""
        for link in root.findall("link")
        if link.get("name") and link.find("collision") is not None
    }


def _collision_boxes_by_link(urdf: Path) -> dict[str, tuple[tuple[str, str], ...]]:
    """Map link name -> ((origin_xyz, box_size), ...) for box collision geoms."""
    root = ET.parse(urdf).getroot()
    out: dict[str, tuple[tuple[str, str], ...]] = {}
    for link in root.findall("link"):
        link_name = link.get("name") or ""
        boxes: list[tuple[str, str]] = []
        for collision in link.findall("collision"):
            box = collision.find("./geometry/box")
            if box is None:
                continue
            origin = collision.find("origin")
            boxes.append((origin.get("xyz") if origin is not None else "0 0 0", box.get("size") or ""))
        if boxes:
            out[link_name] = tuple(boxes)
    return out


def _collision_origin_pose_by_link(
    urdf: Path,
) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
    root = ET.parse(urdf).getroot()
    out: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {}
    for link in root.findall("link"):
        link_name = link.get("name") or ""
        if link_name not in _BIO_GRIPPER_G2_LINKS:
            continue
        origin = link.find("./collision/origin")
        out[link_name] = (
            _rounded_xyz(origin.get("xyz") if origin is not None else None),
            _rounded_xyz(origin.get("rpy") if origin is not None else None),
        )
    return out


def _resolved_collision_meshes_by_link(urdf: Path) -> dict[str, tuple[Path, ...]]:
    out: dict[str, tuple[Path, ...]] = {}
    for link, filenames in _collision_meshes_by_link(urdf).items():
        out[link] = tuple((urdf.parent / filename).resolve() for filename in filenames)
    return out


def _non_fixed_joint_children(urdf: Path) -> set[str]:
    root = ET.parse(urdf).getroot()
    children: set[str] = set()
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed":
            continue
        child = joint.find("child")
        if child is not None and child.get("link"):
            children.add(child.get("link") or "")
    return children


def _non_fixed_joint_names(urdf: Path) -> set[str]:
    root = ET.parse(urdf).getroot()
    return {joint.get("name") or "" for joint in root.findall("joint") if joint.get("type") != "fixed"}


_EXPECTED_ARM_COLLISION_COUNTS: dict[str, dict[str, int]] = {
    "xarm5": {
        "link_base": 2,
        "link1": 2,
        "link2": 5,
        "link3": 7,
        "link4": 2,
        "link5": 1,
    },
    "xarm6": {
        "link_base": 2,
        "link1": 2,
        "link2": 5,
        "link3": 5,
        "link4": 5,
        "link5": 2,
        "link6": 1,
    },
    "xarm7": {
        "link_base": 2,
        "link1": 2,
        "link2": 5,
        "link3": 4,
        "link4": 5,
        "link5": 5,
        "link6": 2,
        "link7": 1,
    },
    "lite6": {
        "link_base": 3,
        "link1": 2,
        "link2": 4,
        "link3": 2,
        "link4": 4,
        "link5": 1,
        "link6": 1,
    },
    "uf850": {
        "link_base": 3,
        "link1": 4,
        "link2": 7,
        "link3": 5,
        "link4": 5,
        "link5": 2,
        "link6": 1,
    },
}


def test_registered_arm_collision_meshes_use_visual_stl() -> None:
    """Registered arm links use selective convex-hull collision STLs; accessories keep their own collision geoms."""
    bad: list[str] = []
    for robot_dir, (dof, mesh_prefix, basenames) in _REGISTERED_ARM_URDFS.items():
        urdf_dir = PROJECT_ROOT / "assets" / "urdf" / robot_dir
        arm_links = _arm_link_names(dof)
        expected_counts = _EXPECTED_ARM_COLLISION_COUNTS[robot_dir]
        for basename in basenames:
            urdf = urdf_dir / basename
            if not urdf.exists():
                bad.append(f"missing registered URDF: {urdf.relative_to(PROJECT_ROOT)}")
                continue
            root = ET.parse(urdf).getroot()
            seen_arm_links: set[str] = set()
            all_col_basenames: list[str] = []
            for link in root.findall("link"):
                link_name = link.get("name") or ""
                arm_filenames: list[str] = []
                for collision in link.findall("collision"):
                    mesh = collision.find(".//mesh")
                    if mesh is None:
                        continue
                    filename = mesh.get("filename") or ""
                    all_col_basenames.append(Path(filename).name)
                    if filename.endswith(".obj"):
                        bad.append(f"{basename}: collision still uses OBJ {filename}")
                    elif "/collision/" in filename and not filename.endswith(".stl"):
                        bad.append(f"{basename}: collision mesh not STL {filename}")
                    if link_name in arm_links:
                        arm_filenames.append(filename)
                if link_name not in arm_links:
                    continue
                if arm_filenames:
                    seen_arm_links.add(link_name)
                expected = [
                    f"meshes/{mesh_prefix}/collision/{robot_dir}_{link_name}_col_{idx:02d}.stl"
                    for idx in range(expected_counts[link_name])
                ]
                if arm_filenames != expected:
                    bad.append(f"{basename}: {link_name} collision should be {expected}, got {arm_filenames}")
            duplicates = sorted({name for name in all_col_basenames if all_col_basenames.count(name) > 1})
            if duplicates:
                bad.append(f"{basename}: duplicate collision mesh basename(s) {duplicates}")
            missing_arm = sorted(arm_links - seen_arm_links)
            if missing_arm:
                bad.append(f"{basename}: missing arm collision links {missing_arm}")
    assert bad == []


def test_bio_gripper_g2_collision_mesh_basenames_are_unique_in_combos() -> None:
    """MuJoCo keys URDF mesh assets by basename, so Bio G2 combo basenames must not collide."""
    bad: list[str] = []
    combo_urdfs = sorted((PROJECT_ROOT / "assets" / "urdf").glob("*/*bio_gripper_g2*.urdf"))
    for urdf in combo_urdfs:
        names = [Path(filename).name for filename in _collision_mesh_filenames(urdf)]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            bad.append(f"{urdf.relative_to(PROJECT_ROOT)}: duplicate collision mesh basename(s) {duplicates}")
    assert bad == []


def test_bio_gripper_g2_urdfs_do_not_reference_legacy_short_stl_names() -> None:
    legacy_refs = {
        "meshes/visual/link_base.stl",
        "meshes/visual/left_finger.stl",
        "meshes/visual/right_finger.stl",
        "../bio_gripper_g2/meshes/visual/link_base.stl",
        "../bio_gripper_g2/meshes/visual/left_finger.stl",
        "../bio_gripper_g2/meshes/visual/right_finger.stl",
    }
    bad: list[str] = []
    bio_urdfs = sorted((PROJECT_ROOT / "assets" / "urdf").glob("*/*bio_gripper_g2*.urdf"))
    for urdf in bio_urdfs:
        for filename in _mesh_filenames(urdf):
            if filename in legacy_refs:
                bad.append(f"{urdf.relative_to(PROJECT_ROOT)}: legacy Bio G2 STL reference {filename}")
    assert bad == []


def test_bio_gripper_g2_collision_origins_align_stl_to_glb_reference() -> None:
    bad: list[str] = []
    for urdf in _BIO_GRIPPER_G2_COLLISION_ORIGIN_URDFS:
        actual = _collision_origin_pose_by_link(urdf)
        for link_name, expected_xyz in _BIO_GRIPPER_G2_COLLISION_ORIGINS.items():
            pose = actual.get(link_name)
            if pose is None:
                bad.append(f"{urdf.relative_to(PROJECT_ROOT)}: missing collision origin for {link_name}")
                continue
            actual_xyz, actual_rpy = pose
            if actual_xyz != expected_xyz:
                bad.append(
                    f"{urdf.relative_to(PROJECT_ROOT)}: {link_name} collision origin xyz "
                    f"should be {expected_xyz}, got {actual_xyz}"
                )
            if actual_rpy != (0.0, 0.0, 0.0):
                bad.append(
                    f"{urdf.relative_to(PROJECT_ROOT)}: {link_name} collision origin rpy "
                    f"should be (0.0, 0.0, 0.0), got {actual_rpy}"
                )
    assert bad == []


def test_bio_gripper_g2_movable_collision_bbox_centers_match_visual_glb() -> None:
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("DracoPy")
    pytest.importorskip("pygltflib")
    bad: list[str] = []
    for urdf in _BIO_GRIPPER_G2_MOVABLE_COLLISION_ALIGNMENT_URDFS:
        root = ET.parse(urdf).getroot()
        masses = []
        for link in root.findall("link"):
            link_name = link.get("name") or ""
            if link_name not in _BIO_GRIPPER_G2_LINKS:
                continue
            mass_node = link.find("inertial/mass")
            if mass_node is not None:
                masses.append(float(mass_node.get("value")))
            visual_mesh = link.find("./visual/geometry/mesh")
            visual_origin = link.find("./visual/origin")
            collisions = link.findall("collision")
            if visual_mesh is None or not collisions:
                bad.append(f"{urdf.relative_to(PROJECT_ROOT)}: {link_name} missing visual/collision mesh")
                continue
            visual_vertices = _mesh_vertices((urdf.parent / (visual_mesh.get("filename") or "")).resolve(), trimesh)
            visual_vertices += np.asarray(_xyz_tuple(visual_origin.get("xyz") if visual_origin is not None else None))
            collision_chunks = []
            for coll in collisions:
                c_mesh = coll.find("./geometry/mesh")
                c_origin = coll.find("origin")
                if c_mesh is None:
                    continue
                c_verts = _mesh_vertices((urdf.parent / (c_mesh.get("filename") or "")).resolve(), trimesh)
                c_verts += np.asarray(_xyz_tuple(c_origin.get("xyz") if c_origin is not None else None))
                collision_chunks.append(c_verts)
            collision_vertices = np.vstack(collision_chunks)
            if link_name in ("bio_gripper_g2_left_finger", "bio_gripper_g2_right_finger"):
                # Verify that the inner gripping plane of the fingertip (X > 0.031 m in finger frame)
                # matches between visual GLB and convex collision mesh at +/-35.5 mm (71 mm closed gap).
                vis_tip = visual_vertices[visual_vertices[:, 0] > 0.031]
                col_tip = collision_vertices[collision_vertices[:, 0] > 0.031]
                vis_inner_y = float(vis_tip[:, 1].max() if "left" in link_name else vis_tip[:, 1].min())
                col_inner_y = float(col_tip[:, 1].max() if "left" in link_name else col_tip[:, 1].min())
                if abs(vis_inner_y - col_inner_y) * 1000.0 > 0.01:
                    bad.append(
                        f"{urdf.relative_to(PROJECT_ROOT)}: {link_name} inner gripping plane mismatch "
                        f"vis={vis_inner_y:.6f} col={col_inner_y:.6f}"
                    )
        if abs(sum(masses) - 0.79) > 1e-6:
            bad.append(f"{urdf.relative_to(PROJECT_ROOT)}: Bio Gripper G2 total mass {sum(masses)} != 0.79 kg")
    assert bad == []


def test_bio_gripper_g2_visual_glb_budget() -> None:
    """High-poly + native Draco should keep Bio G2 visual GLBs under 1 MiB."""
    package_dir = PROJECT_ROOT / "assets" / "urdf" / "bio_gripper_g2"
    visual_dir = package_dir / "meshes" / "visual"
    glbs = sorted(
        path
        for path in visual_dir.rglob("*.glb")
        if "visual_glb_src" not in path.parts and "visual_glb_raw" not in path.parts
    )
    assert glbs, "expected Bio G2 visual GLBs"
    glb_total = sum(path.stat().st_size for path in glbs)
    # Same tris as pre-slim hipoly; native draco_encoder ~0.65 MB vs DracoPy ~5.5 MB.
    # Collision STL stay at original density; only GLB size is the optimization target.
    assert glb_total < 1_000_000, f"Bio G2 visual GLBs total {glb_total} bytes; expected < 1 MB"
    names = {path.name for path in glbs}
    assert "bio_gripper_g2_visual.glb" in names
    assert "bio_gripper_g2_base.glb" in names
    assert "bio_gripper_g2_left_finger.glb" in names
    assert "bio_gripper_g2_right_finger.glb" in names
    assert not any(name.startswith("bio_gripper_g2_visual_link") for name in names)


def test_gripper_g2_visual_glb_budget() -> None:
    """Canonical Gripper G2 visual GLBs (native Draco) must stay under 1 MiB."""
    package_dir = PROJECT_ROOT / "assets" / "urdf" / "gripper_g2"
    visual_dir = package_dir / "meshes" / "visual"
    glbs = sorted(
        path
        for path in visual_dir.rglob("*.glb")
        if "visual_glb_src" not in path.parts and "visual_glb_raw" not in path.parts
    )
    assert glbs, "expected Gripper G2 visual GLBs"
    glb_total = sum(path.stat().st_size for path in glbs)
    assert glb_total < 1_000_000, f"Gripper G2 visual GLBs total {glb_total} bytes; expected < 1 MB"
    names = {path.name for path in glbs}
    assert "gripper_g2_static.glb" in names
    assert "base.glb" in names
    assert "left_finger.glb" in names
    assert "right_finger.glb" in names
    assert not any(name.startswith("gripper_g2_static_link") for name in names)
    assert not any(part in {"link5", "link6", "link7"} for path in glbs for part in path.parts)


def test_lite6_gripper_visual_glb_budget() -> None:
    """Lite6 gripper visual GLBs (native Draco) must stay under 1 MiB."""
    package_dir = PROJECT_ROOT / "assets" / "urdf" / "lite6_gripper"
    visual_dir = package_dir / "meshes" / "visual"
    glbs = sorted(visual_dir.rglob("*.glb"))
    assert glbs, "expected Lite6 gripper visual GLBs"
    glb_total = sum(path.stat().st_size for path in glbs)
    assert glb_total < 1_000_000, f"Lite6 gripper visual GLBs total {glb_total} bytes; expected < 1 MB"
    names = {path.name for path in glbs}
    assert "lite6_gripper_static_link6.glb" in names
    assert "shell.glb" in names
    assert names == {"lite6_gripper_static_link6.glb", "shell.glb"}


def test_lite6_vacuum_gripper_visual_glb_budget() -> None:
    """Lite6 vacuum gripper visual GLB (native Draco) must stay under 1 MiB."""
    package_dir = PROJECT_ROOT / "assets" / "urdf" / "lite6_vacuum_gripper"
    visual_dir = package_dir / "meshes" / "visual"
    glbs = sorted(visual_dir.rglob("*.glb"))
    assert glbs, "expected Lite6 vacuum gripper visual GLBs"
    glb_total = sum(path.stat().st_size for path in glbs)
    assert glb_total < 1_000_000, f"Lite6 vacuum visual GLBs total {glb_total} bytes; expected < 1 MB"
    names = {path.name for path in glbs}
    assert "lite6_vacuum_gripper_visual_link6.glb" in names


def test_vacuum_gripper_visual_glb_budget() -> None:
    """xArm/UF850 vacuum gripper visual GLB (native Draco) must stay under 1 MiB."""
    package_dir = PROJECT_ROOT / "assets" / "urdf" / "vacuum_gripper"
    visual_dir = package_dir / "meshes" / "visual"
    glbs = sorted(visual_dir.rglob("*.glb"))
    assert glbs, "expected xArm/UF850 vacuum gripper visual GLBs"
    glb_total = sum(path.stat().st_size for path in glbs)
    assert glb_total < 1_000_000, f"Vacuum gripper visual GLBs total {glb_total} bytes; expected < 1 MB"
    names = {path.name for path in glbs}
    assert names == {"vacuum_gripper.glb"}


@pytest.mark.parametrize(
    ("arm", "visual_glb_dir", "budget", "expected_names"),
    [
        (
            "xarm5",
            "assets/urdf/xarm5/meshes/xarm5_1305/visual_glb",
            1_500_000,
            {"link_base.glb", "link1.glb", "link2.glb", "link3.glb", "link4.glb", "link5.glb"},
        ),
        (
            "xarm6",
            "assets/urdf/xarm6/meshes/xarm6_1305/visual_glb",
            1_500_000,
            {
                "link_base.glb",
                "link1.glb",
                "link2.glb",
                "link3.glb",
                "link4.glb",
                "link5.glb",
                "link6.glb",
            },
        ),
        (
            "xarm7",
            "assets/urdf/xarm7/meshes/xarm7_1305/visual_glb",
            1_000_000,
            {
                "link_base.glb",
                "link1.glb",
                "link2.glb",
                "link3.glb",
                "link4.glb",
                "link5.glb",
                "link6.glb",
                "link7.glb",
            },
        ),
        (
            "uf850",
            "assets/urdf/uf850/meshes/uf850/visual_glb",
            1_000_000,
            {
                "link_base.glb",
                "link1.glb",
                "link2.glb",
                "link3.glb",
                "link4.glb",
                "link5.glb",
                "link6.glb",
            },
        ),
        (
            "lite6",
            "assets/urdf/lite6/meshes/lite6/visual_glb",
            1_000_000,
            {
                "link_base.glb",
                "link1.glb",
                "link2.glb",
                "link3.glb",
                "link4.glb",
                "link5.glb",
                "link6.glb",
            },
        ),
    ],
)
def test_arm_body_visual_glb_budget(
    arm: str,
    visual_glb_dir: str,
    budget: int,
    expected_names: set[str],
) -> None:
    """Arm-body visual GLBs (native Draco, STL untouched) stay within per-arm budgets."""
    visual_dir = PROJECT_ROOT / visual_glb_dir
    glbs = sorted(visual_dir.glob("*.glb"))
    assert glbs, f"expected {arm} visual GLBs under {visual_glb_dir}"
    glb_total = sum(path.stat().st_size for path in glbs)
    assert glb_total < budget, f"{arm} visual GLBs total {glb_total} bytes; expected < {budget}"
    names = {path.name for path in glbs}
    assert names == expected_names, f"{arm} GLB names {names} != {expected_names}"


def test_accessory_collision_meshes_are_stl_files() -> None:
    urdfs = (
        PROJECT_ROOT / "assets" / "urdf" / "gripper_g2" / "gripper_g2.urdf",
        PROJECT_ROOT / "assets" / "urdf" / "gripper_g2" / "gripper_g2_movable_visual.urdf",
        PROJECT_ROOT / "assets" / "urdf" / "bio_gripper_g2" / "bio_gripper_g2.urdf",
        PROJECT_ROOT / "assets" / "urdf" / "bio_gripper_g2" / "bio_gripper_g2_movable_visual.urdf",
        PROJECT_ROOT / "assets" / "urdf" / "lite6_gripper" / "lite6_gripper.urdf",
        PROJECT_ROOT / "assets" / "urdf" / "lite6_gripper" / "lite6_gripper_movable_visual.urdf",
        PROJECT_ROOT / "assets" / "urdf" / "lite6_vacuum_gripper" / "lite6_vacuum_gripper.urdf",
        PROJECT_ROOT / "assets" / "urdf" / "lite6_vacuum_gripper" / "lite6_vacuum_gripper_collision.urdf",
        PROJECT_ROOT / "assets" / "urdf" / "vacuum_gripper" / "vacuum_gripper.urdf",
        PROJECT_ROOT / "assets" / "urdf" / "vacuum_gripper" / "vacuum_gripper_stl.urdf",
        PROJECT_ROOT / "assets" / "urdf" / "xarm5" / "xarm5_1305_vacuum_gripper_visual.glb.urdf",
        PROJECT_ROOT / "assets" / "urdf" / "xarm6" / "xarm6_1305_g2_movable_visual.urdf",
        PROJECT_ROOT / "assets" / "urdf" / "xarm6" / "xarm6_1305_bio_gripper_g2_movable_visual.glb.urdf",
        PROJECT_ROOT / "assets" / "urdf" / "xarm6" / "xarm6_1305_vacuum_gripper_visual.glb.urdf",
        PROJECT_ROOT / "assets" / "urdf" / "xarm7" / "xarm7_1305_vacuum_gripper_visual.glb.urdf",
        PROJECT_ROOT / "assets" / "urdf" / "uf850" / "uf850_vacuum_gripper_visual.glb.urdf",
        PROJECT_ROOT / "assets" / "urdf" / "lite6" / "lite6_gripper_reversed_movable_visual.glb.urdf",
        PROJECT_ROOT / "assets" / "urdf" / "lite6" / "lite6_vacuum_gripper_visual.glb.urdf",
    )
    bad: list[str] = []
    for urdf in urdfs:
        for filename in _collision_mesh_filenames(urdf):
            if not filename.endswith(".stl"):
                bad.append(f"{urdf.relative_to(PROJECT_ROOT)}: collision mesh not STL {filename}")
                continue
            target = (urdf.parent / filename).resolve()
            if not target.exists():
                bad.append(f"{urdf.relative_to(PROJECT_ROOT)}: missing collision mesh {filename}")
    assert bad == []


def test_standalone_movable_accessory_collision_links_are_movable() -> None:
    expected = {
        PROJECT_ROOT / "assets" / "urdf" / "gripper_g2" / "gripper_g2_movable_visual.urdf": {
            "left_outer_knuckle",
            "left_finger",
            "left_inner_knuckle",
            "right_outer_knuckle",
            "right_finger",
            "right_inner_knuckle",
        },
        PROJECT_ROOT / "assets" / "urdf" / "bio_gripper_g2" / "bio_gripper_g2_movable_visual.urdf": {
            "bio_gripper_g2_left_finger",
            "bio_gripper_g2_right_finger",
        },
        PROJECT_ROOT / "assets" / "urdf" / "lite6_gripper" / "lite6_gripper_movable_visual.urdf": {
            "uflite_finger1",
            "uflite_finger2",
        },
    }
    bad: list[str] = []
    for urdf, moving_links in expected.items():
        collision_mesh_by_link = _collision_meshes_by_link(urdf)
        collision_links = _collision_links(urdf)
        non_fixed_children = _non_fixed_joint_children(urdf)
        for link in sorted(moving_links):
            if link not in collision_links:
                bad.append(f"{urdf.relative_to(PROJECT_ROOT)}: {link} has no collision geometry")
            # Mesh-based accessories still require a collision mesh; Lite6 fingers use boxes.
            if urdf.name != "lite6_gripper_movable_visual.urdf" and link not in collision_mesh_by_link:
                bad.append(f"{urdf.relative_to(PROJECT_ROOT)}: {link} has no collision mesh")
            if link not in non_fixed_children:
                bad.append(f"{urdf.relative_to(PROJECT_ROOT)}: {link} is not child of a movable joint")
    assert bad == []


def test_lite6_vacuum_standalone_collision_is_static_only() -> None:
    urdfs = (
        PROJECT_ROOT / "assets" / "urdf" / "lite6_vacuum_gripper" / "lite6_vacuum_gripper.urdf",
        PROJECT_ROOT / "assets" / "urdf" / "lite6_vacuum_gripper" / "lite6_vacuum_gripper_collision.urdf",
    )
    bad: list[str] = []
    for urdf in urdfs:
        movable_joints = sorted(_non_fixed_joint_names(urdf))
        if movable_joints:
            bad.append(f"{urdf.relative_to(PROJECT_ROOT)}: unexpected movable joints {movable_joints}")
    assert bad == []


def test_arm_movable_accessory_collision_links_match_standalone_models() -> None:
    cases = (
        (
            PROJECT_ROOT / "assets" / "urdf" / "gripper_g2" / "gripper_g2_movable_visual.urdf",
            {
                "left_outer_knuckle",
                "left_finger",
                "left_inner_knuckle",
                "right_outer_knuckle",
                "right_finger",
                "right_inner_knuckle",
            },
            (
                PROJECT_ROOT / "assets" / "urdf" / "xarm5" / "xarm5_1305_g2_movable_visual.urdf",
                PROJECT_ROOT / "assets" / "urdf" / "xarm6" / "xarm6_1305_g2_movable_visual.urdf",
                PROJECT_ROOT / "assets" / "urdf" / "xarm7" / "xarm7_1305_g2_movable_visual.urdf",
                PROJECT_ROOT / "assets" / "urdf" / "uf850" / "uf850_g2_movable_visual.urdf",
            ),
        ),
        (
            PROJECT_ROOT / "assets" / "urdf" / "bio_gripper_g2" / "bio_gripper_g2_movable_visual.urdf",
            {
                "bio_gripper_g2_left_finger",
                "bio_gripper_g2_right_finger",
            },
            (
                PROJECT_ROOT / "assets" / "urdf" / "xarm5" / "xarm5_1305_bio_gripper_g2_movable_visual.glb.urdf",
                PROJECT_ROOT / "assets" / "urdf" / "xarm6" / "xarm6_1305_bio_gripper_g2_movable_visual.glb.urdf",
                PROJECT_ROOT / "assets" / "urdf" / "xarm7" / "xarm7_1305_bio_gripper_g2_movable_visual.glb.urdf",
                PROJECT_ROOT / "assets" / "urdf" / "uf850" / "uf850_bio_gripper_g2_movable_visual.glb.urdf",
            ),
        ),
        (
            PROJECT_ROOT / "assets" / "urdf" / "lite6_gripper" / "lite6_gripper_movable_visual.urdf",
            {
                "uflite_finger1",
                "uflite_finger2",
            },
            (
                PROJECT_ROOT / "assets" / "urdf" / "lite6" / "lite6_gripper_movable_visual.glb.urdf",
                PROJECT_ROOT / "assets" / "urdf" / "lite6" / "lite6_gripper_reversed_movable_visual.glb.urdf",
            ),
        ),
    )
    bad: list[str] = []
    for standalone, moving_links, combos in cases:
        if standalone.name == "lite6_gripper_movable_visual.urdf":
            standalone_boxes = _collision_boxes_by_link(standalone)
            expected = {link: standalone_boxes.get(link, ()) for link in moving_links}
            for combo in combos:
                combo_boxes = _collision_boxes_by_link(combo)
                for link, expected_boxes in expected.items():
                    actual_boxes = combo_boxes.get(link, ())
                    if actual_boxes != expected_boxes:
                        bad.append(
                            f"{combo.relative_to(PROJECT_ROOT)}: {link} collision boxes {actual_boxes} "
                            f"do not match standalone {expected_boxes}"
                        )
            continue
        standalone_meshes = _resolved_collision_meshes_by_link(standalone)
        expected = {link: standalone_meshes.get(link, ()) for link in moving_links}
        for combo in combos:
            combo_meshes = _resolved_collision_meshes_by_link(combo)
            for link, expected_meshes in expected.items():
                actual_meshes = combo_meshes.get(link, ())
                if actual_meshes != expected_meshes:
                    bad.append(
                        f"{combo.relative_to(PROJECT_ROOT)}: {link} collision {actual_meshes} "
                        f"does not match standalone {expected_meshes}"
                    )
    assert bad == []


_LITE6_GRIPPER_REVERSIBLE_URDFS = (
    PROJECT_ROOT / "assets" / "urdf" / "lite6_gripper" / "lite6_gripper_movable_visual.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "lite6" / "lite6_gripper_movable_visual.glb.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "lite6" / "lite6_gripper_reversed_movable_visual.glb.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "lite6" / "lite6_with_gripper.urdf",
)

_LITE6_FINGER_BOXES = {
    "uflite_finger1": (
        ("0 0.00775 0.002", "0.02 0.0155 0.0095"),
        ("0 0.013375 0.016875", "0.02 0.00425 0.01975"),
    ),
    "uflite_finger2": (
        ("0 -0.00775 0.002", "0.02 0.0155 0.0095"),
        ("0 -0.013375 0.016875", "0.02 0.00425 0.01975"),
    ),
}


def _lite6_gripper_visual_origin_by_link(urdf: Path) -> dict[str, np.ndarray]:
    root = ET.parse(urdf).getroot()
    out: dict[str, np.ndarray] = {}
    for link_name in ("uflite_finger1", "uflite_finger2"):
        link = root.find(f".//link[@name='{link_name}']")
        if link is None:
            continue
        origin = link.find("./visual/origin")
        out[link_name] = np.asarray(_xyz_tuple(origin.get("xyz") if origin is not None else None), dtype=np.float64)
    return out


def _lite6_gripper_visual_mesh_by_link(urdf: Path) -> dict[str, str]:
    root = ET.parse(urdf).getroot()
    out: dict[str, str] = {}
    for link_name in ("uflite_finger1", "uflite_finger2"):
        link = root.find(f".//link[@name='{link_name}']")
        mesh = link.find("./visual/geometry/mesh") if link is not None else None
        if mesh is not None:
            out[link_name] = mesh.get("filename") or ""
    return out


def _lite6_gripper_root_inner_gap_mm(
    finger1_vertices: np.ndarray,
    finger2_vertices: np.ndarray,
    *,
    finger1_origin_xyz: tuple[float, float, float],
    finger2_origin_xyz: tuple[float, float, float],
    finger1_axis_xyz: tuple[float, float, float],
    finger2_axis_xyz: tuple[float, float, float],
    finger1_visual_origin_xyz: tuple[float, float, float],
    finger2_visual_origin_xyz: tuple[float, float, float],
    q_m: float,
) -> float:
    origin1 = np.asarray(finger1_origin_xyz, dtype=np.float64) + np.asarray(finger1_axis_xyz, dtype=np.float64) * q_m
    origin2 = np.asarray(finger2_origin_xyz, dtype=np.float64) + np.asarray(finger2_axis_xyz, dtype=np.float64) * q_m
    vis1 = np.asarray(finger1_visual_origin_xyz, dtype=np.float64)
    vis2 = np.asarray(finger2_visual_origin_xyz, dtype=np.float64)
    w1 = finger1_vertices + origin1 + vis1
    w2 = finger2_vertices + origin2 + vis2
    z_root = finger1_origin_xyz[2]
    root1 = w1[(w1[:, 2] >= z_root) & (w1[:, 2] <= z_root + 0.008)]
    root2 = w2[(w2[:, 2] >= z_root) & (w2[:, 2] <= z_root + 0.008)]
    return float((root1[:, 1].min() - root2[:, 1].max()) * 1000.0)


def _lite6_gripper_finger_center_separation_mm(
    finger1_vertices: np.ndarray,
    finger2_vertices: np.ndarray,
    *,
    finger1_origin_xyz: tuple[float, float, float],
    finger2_origin_xyz: tuple[float, float, float],
    finger1_axis_xyz: tuple[float, float, float],
    finger2_axis_xyz: tuple[float, float, float],
    finger1_visual_origin_xyz: tuple[float, float, float],
    finger2_visual_origin_xyz: tuple[float, float, float],
    q_m: float,
) -> float:
    origin1 = np.asarray(finger1_origin_xyz, dtype=np.float64) + np.asarray(finger1_axis_xyz, dtype=np.float64) * q_m
    origin2 = np.asarray(finger2_origin_xyz, dtype=np.float64) + np.asarray(finger2_axis_xyz, dtype=np.float64) * q_m
    vis1 = np.asarray(finger1_visual_origin_xyz, dtype=np.float64)
    vis2 = np.asarray(finger2_visual_origin_xyz, dtype=np.float64)
    w1 = finger1_vertices + origin1 + vis1
    w2 = finger2_vertices + origin2 + vis2
    center1 = (w1.min(axis=0) + w1.max(axis=0)) / 2.0
    center2 = (w2.min(axis=0) + w2.max(axis=0)) / 2.0
    return float(abs(center2[1] - center1[1]) * 1000.0)


def test_lite6_gripper_finger_collision_gap_at_keyframes() -> None:
    trimesh = pytest.importorskip("trimesh")
    mesh_dir = PROJECT_ROOT / "assets" / "urdf" / "lite6_gripper" / "meshes" / "collision"
    finger1_vertices = _mesh_vertices(mesh_dir / "finger1.stl", trimesh)
    finger2_vertices = _mesh_vertices(mesh_dir / "finger2.stl", trimesh)

    for urdf in _LITE6_GRIPPER_REVERSIBLE_URDFS:
        root = ET.parse(urdf).getroot()
        joints = {joint.get("name"): joint for joint in root.findall("joint")}
        j1 = joints["finger_joint1"]
        j2 = joints["finger_joint2"]
        origin1 = _xyz_tuple(j1.find("origin").get("xyz"))
        origin2 = _xyz_tuple(j2.find("origin").get("xyz"))
        axis1 = _xyz_tuple(j1.find("axis").get("xyz"))
        axis2 = _xyz_tuple(j2.find("axis").get("xyz"))
        visual_origins = _lite6_gripper_visual_origin_by_link(urdf)
        vis1 = tuple(visual_origins.get("uflite_finger1", np.zeros(3)).tolist())
        vis2 = tuple(visual_origins.get("uflite_finger2", np.zeros(3)).tolist())
        visual_meshes = _lite6_gripper_visual_mesh_by_link(urdf)
        boxes = _collision_boxes_by_link(urdf)
        assert vis1 == (0.0, 0.0, 0.0), (
            f"{urdf.relative_to(PROJECT_ROOT)}: finger1 visual origin should align with the mesh frame, got {vis1}"
        )
        assert vis2 == (0.0, 0.0, 0.0), (
            f"{urdf.relative_to(PROJECT_ROOT)}: finger2 visual origin should align with the mesh frame, got {vis2}"
        )
        assert visual_meshes == {
            "uflite_finger1": "../lite6_gripper/meshes/collision/finger1.stl",
            "uflite_finger2": "../lite6_gripper/meshes/collision/finger2.stl",
        }
        assert boxes.get("uflite_finger1") == _LITE6_FINGER_BOXES["uflite_finger1"], (
            f"{urdf.relative_to(PROJECT_ROOT)}: finger1 dual-box collision mismatch"
        )
        assert boxes.get("uflite_finger2") == _LITE6_FINGER_BOXES["uflite_finger2"], (
            f"{urdf.relative_to(PROJECT_ROOT)}: finger2 dual-box collision mismatch"
        )

        closed_sep = _lite6_gripper_finger_center_separation_mm(
            finger1_vertices,
            finger2_vertices,
            finger1_origin_xyz=origin1,
            finger2_origin_xyz=origin2,
            finger1_axis_xyz=axis1,
            finger2_axis_xyz=axis2,
            finger1_visual_origin_xyz=vis1,
            finger2_visual_origin_xyz=vis2,
            q_m=0.0,
        )
        open_sep = _lite6_gripper_finger_center_separation_mm(
            finger1_vertices,
            finger2_vertices,
            finger1_origin_xyz=origin1,
            finger2_origin_xyz=origin2,
            finger1_axis_xyz=axis1,
            finger2_axis_xyz=axis2,
            finger1_visual_origin_xyz=vis1,
            finger2_visual_origin_xyz=vis2,
            q_m=0.0089,
        )

        closed_gap = _lite6_gripper_root_inner_gap_mm(
            finger1_vertices,
            finger2_vertices,
            finger1_origin_xyz=origin1,
            finger2_origin_xyz=origin2,
            finger1_axis_xyz=axis1,
            finger2_axis_xyz=axis2,
            finger1_visual_origin_xyz=vis1,
            finger2_visual_origin_xyz=vis2,
            q_m=0.0,
        )
        mid_gap = _lite6_gripper_root_inner_gap_mm(
            finger1_vertices,
            finger2_vertices,
            finger1_origin_xyz=origin1,
            finger2_origin_xyz=origin2,
            finger1_axis_xyz=axis1,
            finger2_axis_xyz=axis2,
            finger1_visual_origin_xyz=vis1,
            finger2_visual_origin_xyz=vis2,
            q_m=0.00445,
        )
        open_gap = _lite6_gripper_root_inner_gap_mm(
            finger1_vertices,
            finger2_vertices,
            finger1_origin_xyz=origin1,
            finger2_origin_xyz=origin2,
            finger1_axis_xyz=axis1,
            finger2_axis_xyz=axis2,
            finger1_visual_origin_xyz=vis1,
            finger2_visual_origin_xyz=vis2,
            q_m=0.0089,
        )

        assert 14.0 <= closed_sep <= 17.0, (
            f"{urdf.relative_to(PROJECT_ROOT)}: closed finger center separation {closed_sep:.2f} mm"
        )
        assert 32.0 <= open_sep <= 35.0, (
            f"{urdf.relative_to(PROJECT_ROOT)}: open finger center separation {open_sep:.2f} mm"
        )
        assert open_sep > closed_sep, (
            f"{urdf.relative_to(PROJECT_ROOT)}: opening should increase collision finger separation"
        )
        assert closed_gap == pytest.approx(0.0, abs=0.5), (
            f"{urdf.relative_to(PROJECT_ROOT)}: closed root-region inner gap should overlap at the finger root"
        )
        assert mid_gap == pytest.approx(8.9, abs=0.5), (
            f"{urdf.relative_to(PROJECT_ROOT)}: mid-stroke root-region inner gap should track prismatic travel"
        )
        assert open_gap == pytest.approx(17.8, abs=0.5), (
            f"{urdf.relative_to(PROJECT_ROOT)}: open root-region inner gap should track prismatic travel"
        )


def test_lite6_vacuum_collision_uses_visual_stl() -> None:
    expected_by_urdf = {
        PROJECT_ROOT / "assets" / "urdf" / "lite6_vacuum_gripper" / "lite6_vacuum_gripper.urdf": {
            "meshes/visual/vacuum_gripper_lite.stl",
        },
        PROJECT_ROOT / "assets" / "urdf" / "lite6_vacuum_gripper" / "lite6_vacuum_gripper_collision.urdf": {
            "meshes/visual/vacuum_gripper_lite.stl",
        },
        PROJECT_ROOT / "assets" / "urdf" / "lite6" / "lite6_with_vacuum_gripper.urdf": {
            "../lite6_vacuum_gripper/meshes/visual/vacuum_gripper_lite.stl",
        },
        PROJECT_ROOT / "assets" / "urdf" / "lite6" / "lite6_vacuum_gripper_visual.glb.urdf": {
            "../lite6_vacuum_gripper/meshes/visual/vacuum_gripper_lite.stl",
        },
    }
    bad: list[str] = []
    for urdf, expected in expected_by_urdf.items():
        actual = {filename for filename in _collision_mesh_filenames(urdf) if "vacuum_gripper_lite.stl" in filename}
        if actual != expected:
            bad.append(f"{urdf.relative_to(PROJECT_ROOT)}: expected {sorted(expected)}, got {sorted(actual)}")
    assert bad == []


def test_no_collision_obj_meshes_in_public_assets() -> None:
    obj_files = sorted(
        p.relative_to(PROJECT_ROOT) for p in _public_files() if "collision" in str(p) and p.suffix == ".obj"
    )
    assert obj_files == []


def test_vacuum_gripper_static_urdfs_and_parameters() -> None:
    expected_inertia = {
        "ixx": 0.00047106,
        "ixy": 3.9292e-7,
        "ixz": 2.6537e-6,
        "iyy": 0.00033072,
        "iyz": -1.0975e-5,
        "izz": 0.00025642,
    }
    cases = (
        (PROJECT_ROOT / "assets/urdf/vacuum_gripper/vacuum_gripper.urdf", "vacuum_mount", 0),
        (PROJECT_ROOT / "assets/urdf/vacuum_gripper/vacuum_gripper_stl.urdf", "vacuum_mount", 0),
        (PROJECT_ROOT / "assets/urdf/xarm5/xarm5_1305_vacuum_gripper_visual.glb.urdf", "link5", 5),
        (PROJECT_ROOT / "assets/urdf/xarm6/xarm6_1305_vacuum_gripper_visual.glb.urdf", "link6", 6),
        (PROJECT_ROOT / "assets/urdf/xarm7/xarm7_1305_vacuum_gripper_visual.glb.urdf", "link7", 7),
        (PROJECT_ROOT / "assets/urdf/uf850/uf850_vacuum_gripper_visual.glb.urdf", "link6", 6),
    )
    for urdf, expected_parent, expected_dof in cases:
        root = ET.parse(urdf).getroot()
        assert len(_non_fixed_joint_names(urdf)) == expected_dof

        tool_link = root.find("./link[@name='xarm_vacuum_gripper_link']")
        assert tool_link is not None, urdf
        mass = float(tool_link.find("./inertial/mass").attrib["value"])
        assert mass == pytest.approx(0.61), urdf
        com = _rounded_xyz(tool_link.find("./inertial/origin").attrib["xyz"])
        assert com == (0.0, 0.0, 0.053), urdf
        inertia = {k: float(v) for k, v in tool_link.find("./inertial/inertia").attrib.items()}
        for key, expected_val in expected_inertia.items():
            assert inertia[key] == pytest.approx(expected_val), (urdf, key)

        collisions = tool_link.findall("collision")
        assert len(collisions) == 33, urdf
        hull_names = [Path(c.find("./geometry/mesh").attrib["filename"]).name for c in collisions]
        assert len(set(hull_names)) == 33, urdf
        assert hull_names[0] == "hull_000_flange.stl"
        assert hull_names[-1] == "hull_032_cup_05.stl"

        fix_joint = root.find("./joint[@name='vacuum_gripper_fix']")
        assert fix_joint is not None and fix_joint.attrib["type"] == "fixed", urdf
        assert fix_joint.find("parent").attrib["link"] == expected_parent
        assert fix_joint.find("child").attrib["link"] == "xarm_vacuum_gripper_link"

        tcp_joint = root.find("./joint[@name='vacuum_joint_tcp']")
        assert tcp_joint is not None and tcp_joint.attrib["type"] == "fixed", urdf
        assert tcp_joint.find("parent").attrib["link"] == "xarm_vacuum_gripper_link"
        assert tcp_joint.find("child").attrib["link"] == "vacuum_tcp"
        assert _rounded_xyz(tcp_joint.find("origin").attrib["xyz"]) == (0.0, 0.0, 0.126), urdf
