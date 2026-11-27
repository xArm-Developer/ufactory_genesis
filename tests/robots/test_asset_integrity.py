"""Lightweight checks for public robot asset consistency."""

from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from ufactory.grippers.g2 import G2_LINKAGE_GEOMETRY
from ufactory.robots.assembly import accessories, compose_robot_urdf, end_effectors

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_URDF_ROOT = PROJECT_ROOT / "assets" / "urdf"


def _assembly_urdfs(
    end_effector: str | None = None,
    *,
    robot: str | None = None,
    modes: tuple[str, ...] | None = None,
) -> list[Path]:
    """Every supported composed URDF variant (robot × end effector × visual mode), optionally filtered."""
    out: list[Path] = []
    for name, ee in end_effectors().items():
        if end_effector is not None and name != end_effector:
            continue
        variants = ("glb", "stl")
        for key in ee.robots:
            if robot is not None and key != robot:
                continue
            for visual in variants:
                if modes is None or visual in modes:
                    out.append(Path(compose_robot_urdf(key, name, visual=visual)))
    assert out, (end_effector, robot, modes)
    return out


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _public_files() -> set[Path]:
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=PROJECT_ROOT).split(b"\0")[:-1]
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=PROJECT_ROOT,
    ).split(b"\0")[:-1]
    return {PROJECT_ROOT / p.decode() for p in (*tracked, *untracked) if (PROJECT_ROOT / p.decode()).exists()}


def test_public_urdf_mesh_references_exist() -> None:
    urdfs = sorted(p for p in _public_files() if p.suffix == ".urdf" or p.name.endswith(".glb.urdf"))
    urdfs += _assembly_urdfs()
    urdfs += [
        Path(compose_robot_urdf(robot, ee, accessory=accessory, visual=visual))
        for robot in ("xarm6_1305", "uf850")
        for ee in (None, "gripper_g2", "vacuum_gripper", "robotiq_2f_85", "robotiq_2f_140")
        for accessory in accessories()
        for visual in ("glb", "stl")
    ]
    urdfs += [_URDF_ROOT / "mounts" / "robotiq_mounting" / "robotiq_mounting.urdf"]
    urdfs += [
        _URDF_ROOT / f"mounts/{plate}/{plate}.urdf"
        for plate in ("xarm_mounting_plate", "uf850_mounting_plate", "lite6_mounting_plate")
    ]
    urdfs += [
        Path(compose_robot_urdf(robot, None, mounting_plate=True, visual=visual))
        for robot in ("xarm5_1305", "xarm6_1305", "xarm7_1305", "uf850", "lite6")
        for visual in ("glb", "stl")
    ]
    missing: list[str] = []
    for urdf in urdfs:
        root = ET.parse(urdf).getroot()
        for mesh in root.findall(".//mesh"):
            filename = mesh.get("filename")
            if not filename or filename.startswith("package://"):
                continue
            target = (urdf.parent / filename).resolve()
            if not target.exists():
                missing.append(f"{_rel(urdf)} -> {filename}")

    assert missing == []


def _g2_models() -> list[tuple[Path, ET.Element]]:
    """The Gripper G2 part file plus every composed G2 assembly (4 arms × glb/stl)."""
    parts = []
    for path in sorted(_URDF_ROOT.rglob("*.urdf")):
        root = ET.parse(path).getroot()
        if root.find("./joint[@name='drive_joint']") is not None:
            parts.append((path, root))
    assert [path.name for path, _ in parts] == ["gripper_g2_movable_visual.urdf"]
    composed = [(path, ET.parse(path).getroot()) for path in _assembly_urdfs("gripper_g2")]
    assert len(composed) == 8
    return parts + composed


def test_standard_g2_urdfs_share_joint_limits_and_mimic_linkage() -> None:
    from ufactory.grippers.g2 import (
        G2_JOINT_NAMES,
        G2_LINKAGE_GEOMETRY,
        GRIPPER_G2_SIM_OPEN_DRIVE,
        GRIPPER_G2_SIM_CLOSE_DRIVE,
        GRIPPER_G2_JOINT_VELOCITY_RAD_S,
    )

    models = _g2_models()
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
        "xarm_gripper_base_link",
        "left_outer_knuckle",
        "left_finger",
        "left_inner_knuckle",
        "right_outer_knuckle",
        "right_finger",
        "right_inner_knuckle",
    }

    models = _g2_models()

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

    pkg = PROJECT_ROOT / "assets" / "urdf" / "end_effectors" / "gripper_g2" / "meshes"
    base_stl = _mesh_vertices(pkg / "visual" / "base_link.stl", trimesh)
    base_glb = _mesh_vertices(pkg / "visual" / "visual_glb" / "base.glb", trimesh)
    lf_glb = _mesh_vertices(pkg / "visual" / "visual_glb" / "left_finger.glb", trimesh)
    rf_glb = _mesh_vertices(pkg / "visual" / "visual_glb" / "right_finger.glb", trimesh)

    assert base_stl[:, 2].min() * 1000.0 == pytest.approx(-5.0, abs=0.01)
    assert base_stl[:, 2].max() * 1000.0 == pytest.approx(104.5, abs=0.01)
    assert base_glb[:, 2].min() * 1000.0 == pytest.approx(-5.0, abs=0.01)
    assert base_glb[:, 2].max() * 1000.0 == pytest.approx(104.5, abs=0.01)

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


_G2_PART = _URDF_ROOT / "end_effectors" / "gripper_g2" / "gripper_g2_movable_visual.urdf"
_G2_BASE_GEOMETRY = ("visual/visual_glb/base.glb", "visual/base_link.stl", "collision hulls")


def _triangles(mesh_path: Path) -> np.ndarray:
    """Triangle corners (m) of a binary STL or a Draco GLB without node transforms, in the file frame."""
    if mesh_path.suffix.lower() == ".glb":
        DracoPy = pytest.importorskip("DracoPy")
        pygltflib = pytest.importorskip("pygltflib")
        glb = pygltflib.GLTF2().load(str(mesh_path))
        blob = glb.binary_blob()
        out = []
        for mesh in glb.meshes:
            for prim in mesh.primitives:
                view = glb.bufferViews[prim.extensions["KHR_draco_mesh_compression"]["bufferView"]]
                decoded = DracoPy.decode(blob[view.byteOffset : view.byteOffset + view.byteLength])
                points = np.asarray(decoded.points, dtype=np.float64)
                out.append(points[np.asarray(decoded.faces, dtype=np.int64).reshape(-1, 3)])
        return np.concatenate(out)
    trimesh = pytest.importorskip("trimesh")
    return np.asarray(trimesh.load(mesh_path, force="mesh", process=False).triangles, dtype=np.float64)


def _g2_base_triangles(geometry: str) -> np.ndarray:
    if geometry == "collision hulls":
        names = _collision_meshes_by_link(_G2_PART)["xarm_gripper_base_link"]
        return np.concatenate([_triangles((_G2_PART.parent / name).resolve()) for name in names])
    return _triangles(_URDF_ROOT / "end_effectors" / "gripper_g2" / "meshes" / geometry)


def _section_mm(triangles_mm: np.ndarray, z_mm: float) -> np.ndarray:
    """Points (mm) where triangle edges cross the plane z = z_mm."""
    points = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        a, b = triangles_mm[:, i], triangles_mm[:, j]
        da, db = a[:, 2] - z_mm, b[:, 2] - z_mm
        cross = (da * db <= 0.0) & (np.abs(db - da) > 1e-12)
        t = (-da[cross] / (db[cross] - da[cross]))[:, None]
        points.append(a[cross, :2] + t * (b[cross, :2] - a[cross, :2]))
    return np.concatenate(points)


@pytest.mark.parametrize("geometry", _G2_BASE_GEOMETRY)
def test_gripper_g2_base_below_the_flange_face_fits_the_openings(geometry: str) -> None:
    """Below the flange face the G2 base is only the centring boss, the tool-connector block and the locating pin at
    (0, -25) mm, which fits the flange's 6 mm pin hole. Same check as ufactory_mujoco 0.1.3
    (test_plan_a_flange_section_partition_and_pin_bound) across a 3 mm camera bracket: pin r <= 3.00 mm (6.10 mm
    hole), the rest within r < 16.20 mm (34.5 mm bore) or the Pogo-Pin cutout.

    The export's pin was a coarse prism out to r = 3.32 mm, some face vertices sat 4.8-11.5 um below z = 0, and the
    single visual collision mesh became one hull spanning boss, block and pin.
    """
    tri = _g2_base_triangles(geometry) * 1000.0
    for z in np.linspace(-2.95, -0.01, 60):
        pts = _section_mm(tri, z)
        pin_r = np.hypot(pts[:, 0], pts[:, 1] + 25.0)
        pin = pin_r <= 3.5
        rest = pts[~pin]
        bore = np.hypot(rest[:, 0], rest[:, 1]) < 16.20
        pogo = (np.abs(rest[:, 0]) < 7.60) & (rest[:, 1] > 20.40) & (rest[:, 1] < 25.60)
        assert pin.any() and pin_r[pin].max() <= 3.00, f"pin r = {pin_r[pin].max():.3f} mm at z = {z:.2f} mm"
        assert (bore | pogo).all(), f"{(~(bore | pogo)).sum()} section points outside the openings at z = {z:.2f} mm"


def test_gripper_g2_base_collides_as_ten_convex_hulls() -> None:
    """The base collides as ufactory_mujoco's 10 convex hulls (353.2 cm3 together). Its visual mesh, 420 open shells,
    used to be the collision mesh, and Genesis wrapped it in a single hull of 638.7 cm3."""
    trimesh = pytest.importorskip("trimesh")
    names = _collision_meshes_by_link(_G2_PART)["xarm_gripper_base_link"]
    assert [Path(name).name for name in names] == [f"base_link_col_{i:02d}.stl" for i in range(10)]
    hulls = [trimesh.load((_G2_PART.parent / name).resolve(), force="mesh") for name in names]
    for name, mesh in zip(names, hulls):
        assert mesh.is_watertight and 1.0 - mesh.volume / mesh.convex_hull.volume <= 1e-5, name
    assert sum(mesh.volume for mesh in hulls) * 1e6 == pytest.approx(353.2, abs=0.1)
    visual = _triangles(_URDF_ROOT / "end_effectors" / "gripper_g2" / "meshes" / "visual" / "base_link.stl").reshape(
        -1, 3
    )
    np.testing.assert_allclose(np.min([m.bounds[0] for m in hulls], axis=0), visual.min(axis=0), atol=1e-5)
    np.testing.assert_allclose(np.max([m.bounds[1] for m in hulls], axis=0), visual.max(axis=0), atol=1e-5)


def _glb_material_triangles(path: Path) -> list[tuple[object, np.ndarray]]:
    """(pbrMetallicRoughness, triangle corners in mm) per Draco primitive of a GLB without node transforms."""
    DracoPy = pytest.importorskip("DracoPy")
    pygltflib = pytest.importorskip("pygltflib")
    glb = pygltflib.GLTF2().load(str(path))
    blob = glb.binary_blob()
    out = []
    for mesh in glb.meshes:
        for prim in mesh.primitives:
            view = glb.bufferViews[prim.extensions["KHR_draco_mesh_compression"]["bufferView"]]
            decoded = DracoPy.decode(blob[view.byteOffset : view.byteOffset + view.byteLength])
            points = np.asarray(decoded.points, dtype=np.float64) * 1000.0
            triangles = points[np.asarray(decoded.faces, dtype=np.int64).reshape(-1, 3)]
            out.append((glb.materials[prim.material].pbrMetallicRoughness, triangles))
    return out


def test_gripper_g2_base_flange_uses_the_arm_flange_material() -> None:
    """The G2 base GLB shows the flange, centring boss and locating pin in the xArm / UF850 flange material, the +Y
    connector cover in the fingers' black, and the bearing cover between the fingers (plus its four M3x8 screws) in
    matte black anodized aluminium; only the outer housing stays white."""
    parts = _glb_material_triangles(_URDF_ROOT / "end_effectors/gripper_g2/meshes/visual/visual_glb/base.glb")
    finger = _glb_material_triangles(_URDF_ROOT / "end_effectors/gripper_g2/meshes/visual/visual_glb/left_finger.glb")[
        0
    ][0]
    assert len(parts) == 4
    [(flange_pbr, flange)] = [(m, t) for m, t in parts if m.metallicFactor > 0.5 and m.baseColorFactor[0] > 0.5]
    [(bearing_pbr, bearing)] = [(m, t) for m, t in parts if m.metallicFactor > 0.5 and m.baseColorFactor[0] < 0.25]
    [(cover_pbr, cover)] = [(m, t) for m, t in parts if m.baseColorFactor == finger.baseColorFactor]
    [(housing_pbr, housing)] = [(m, t) for m, t in parts if m.baseColorFactor == [1.0, 1.0, 1.0, 1.0]]
    assert (cover_pbr.metallicFactor, cover_pbr.roughnessFactor) == (finger.metallicFactor, finger.roughnessFactor)
    assert housing_pbr.metallicFactor == 0.0
    assert bearing_pbr.metallicFactor == pytest.approx(0.8, abs=1e-6)
    assert bearing_pbr.roughnessFactor == pytest.approx(0.62, abs=0.01)
    np.testing.assert_allclose(bearing_pbr.baseColorFactor[:3], [0.12, 0.12, 0.125], atol=1e-4)
    assert bearing[..., 0].min() > -14.6 and bearing[..., 0].max() < 14.4
    assert bearing[..., 1].min() > -15.1 and bearing[..., 1].max() < 15.1
    assert bearing[..., 2].min() == pytest.approx(92.94, abs=0.05)
    assert bearing[..., 2].max() == pytest.approx(101.94, abs=0.05)

    for arm_glb in (
        "robots/xarm6/meshes/xarm6_1305/visual_glb/link6.glb",
        "robots/uf850/meshes/uf850/visual_glb/link6.glb",
    ):
        (arm_pbr,) = [m for m, _ in _glb_material_triangles(_URDF_ROOT / arm_glb) if m.metallicFactor > 0.5]
        np.testing.assert_allclose(flange_pbr.baseColorFactor, arm_pbr.baseColorFactor, atol=1e-6)
        assert flange_pbr.metallicFactor == pytest.approx(arm_pbr.metallicFactor, abs=1e-6)
        assert flange_pbr.roughnessFactor == pytest.approx(arm_pbr.roughnessFactor, abs=0.01)

    # Flange (boss bottom z = -5, top z = 8.3, Ø75 ring), cover (+Y sector, x -15.5..15.5) and housing (from z = 8.18).
    assert flange[..., 2].min() == pytest.approx(-5.0, abs=0.01) and flange[..., 2].max() < 8.35
    assert np.hypot(flange[..., 0], flange[..., 1]).max() < 37.95
    assert cover.reshape(-1, 3).min(0) == pytest.approx([-15.50, 17.15, 0.0], abs=0.02)
    assert cover.reshape(-1, 3).max(0) == pytest.approx([15.50, 37.50, 8.10], abs=0.02)
    assert housing[..., 2].min() > 8.17
    in_bearing_cavity = (
        (housing[..., 2].max(axis=1) > 97.9)
        & (np.abs(housing[..., 0]).max(axis=1) < 12.0)
        & (np.abs(housing[..., 1]).max(axis=1) < 14.0)
    )
    assert not in_bearing_cavity.any()
    # The ring's outer face is metal all round except the cover's sector (azimuth 65-115 deg), which the cover fills.
    # Whole degrees reached by points sampled over the outer-face triangles:
    grid = np.array([(i, j) for i in range(31) for j in range(31 - i)]) / 30.0
    weights = np.c_[grid, 1.0 - grid.sum(axis=1)]

    def outer_degrees(triangles: np.ndarray) -> set[int]:
        centre = triangles.mean(axis=1)
        side = (np.hypot(centre[:, 0], centre[:, 1]) > 37.0) & (centre[:, 2] > 0.5) & (centre[:, 2] < 7.5)
        points = np.einsum("sk,nkd->nsd", weights, triangles[side]).reshape(-1, 3)
        return set((np.degrees(np.arctan2(points[:, 1], points[:, 0])) // 1 % 360).astype(int).tolist())

    metal, black = outer_degrees(flange), outer_degrees(cover)
    assert metal | black == set(range(360))
    assert black <= set(range(64, 116)) and not metal & set(range(66, 114))


def test_gripper_g2_finger_pads_use_white_translucent_silicone() -> None:
    """The Gripper G2 left/right finger GLBs split the inner grip pad (320 triangles) into a white translucent
    silicone primitive while keeping the rest of the finger in the original dark material as primitive 0."""
    pygltflib = pytest.importorskip("pygltflib")
    cover_black = [
        m.baseColorFactor
        for m, _ in _glb_material_triangles(_URDF_ROOT / "end_effectors/gripper_g2/meshes/visual/visual_glb/base.glb")
        if m.metallicFactor == 0.0 and m.baseColorFactor != [1.0, 1.0, 1.0, 1.0]
    ][0]

    for side in ("left", "right"):
        glb_path = _URDF_ROOT / f"end_effectors/gripper_g2/meshes/visual/visual_glb/{side}_finger.glb"
        glb = pygltflib.GLTF2().load(str(glb_path))
        assert glb.materials[glb.meshes[0].primitives[0].material].alphaMode in (None, "OPAQUE")
        assert glb.materials[glb.meshes[1].primitives[0].material].alphaMode == "BLEND"

        parts = _glb_material_triangles(glb_path)
        assert len(parts) == 2
        (body_pbr, _), (pad_pbr, pad) = parts
        assert body_pbr.baseColorFactor == cover_black
        assert body_pbr.metallicFactor == 0.0
        np.testing.assert_allclose(pad_pbr.baseColorFactor, [0.92, 0.92, 0.90, 0.75], atol=1e-4)
        assert pad_pbr.metallicFactor == 0.0
        assert pad_pbr.roughnessFactor == pytest.approx(0.30, abs=0.01)
        assert len(pad) == 320

        pts = pad.reshape(-1, 3)
        want_y_min, want_y_max = (-26.0, -23.0) if side == "left" else (23.0, 26.0)
        assert pts.min(0) == pytest.approx([-14.75, want_y_min, 22.26], abs=0.05)
        assert pts.max(0) == pytest.approx([14.75, want_y_max, 59.75], abs=0.05)

        inner_y_m = float(pts[:, 1].min() if side == "left" else -pts[:, 1].max()) * 0.001
        assert inner_y_m == pytest.approx(G2_LINKAGE_GEOMETRY.pad_inner_y_m, abs=5e-5)


@pytest.mark.parametrize("geometry", ("visual/visual_glb/base.glb", "visual/base_link.stl"))
def test_gripper_g2_locating_pin_is_round_and_slotted(geometry: str) -> None:
    """The locating pin at (0, -25) mm is a round dome-tipped cylinder (r = 2.98 mm, tip z = -4.50 mm)
    with a 1.00 mm wide x 2.00 mm deep slot at the STEP yaw (-0.12283004 rad)."""
    tri_mm = _triangles(_URDF_ROOT / "end_effectors" / "gripper_g2" / "meshes" / geometry) * 1000.0
    pin_axis = np.array([0.0, -25.0])
    tri_r = np.hypot(tri_mm[:, :, 0] - pin_axis[0], tri_mm[:, :, 1] - pin_axis[1])
    pin_tri = tri_mm[(tri_r.max(axis=1) <= 3.1) & (tri_mm[:, :, 2].min(axis=1) < -0.05)]
    assert len(pin_tri) > 200, geometry
    pts = pin_tri.reshape(-1, 3)
    assert float(pts[:, 2].min()) == pytest.approx(-4.50, abs=0.02)

    # Cylindrical body z in [-2.40, -0.10] mm: round (r = 2.98 mm) and smoothly tessellated (max gap < 15 deg).
    cyl_sec = _section_mm(pin_tri, -1.25)
    cyl_d = cyl_sec - pin_axis
    cyl_r = np.hypot(cyl_d[:, 0], cyl_d[:, 1])
    np.testing.assert_allclose(cyl_r, 2.98, atol=0.02)
    angles = np.sort(np.degrees(np.arctan2(cyl_d[:, 1], cyl_d[:, 0])) % 360.0)
    diffs = np.diff(np.r_[angles, angles[0] + 360.0])
    assert float(diffs.max()) < 15.0

    # Dome section z in [-4.50, -2.50] mm: stays within r <= 3.00 mm and opens a clear 1.00 mm slot.
    yaw = -0.12283004
    c, s = np.cos(-yaw), np.sin(-yaw)
    rot2 = np.array([[c, -s], [s, c]])
    for z in np.linspace(-4.30, -2.65, 6):
        sec = _section_mm(pin_tri, z)
        sec_r = np.hypot(sec[:, 0] - pin_axis[0], sec[:, 1] - pin_axis[1])
        assert sec.any() and float(sec_r.max()) <= 3.00
        uv = (sec - pin_axis) @ rot2.T
        assert not (np.abs(uv[:, 0]) < 0.45).any(), f"slot blocked at z = {z:.2f} mm"
        assert (uv[:, 0] > 0.55).any() and (uv[:, 0] < -0.55).any(), f"asymmetric dome lobes at z = {z:.2f} mm"

    # Slot floor at z = -2.50 mm spans across |u| < 0.40 mm.
    floor_pts = pts[np.abs(pts[:, 2] + 2.50) < 0.02]
    floor_uv = (floor_pts[:, :2] - pin_axis) @ rot2.T
    assert (np.abs(floor_uv[:, 0]) < 0.40).any()


_G2_FLANGE_CIRCLES = (
    ("boss", (0.0, 0.0), 15.75, (-4.4, 1.4), 1, -1.5, None),
    ("flange outer", (0.0, 0.0), 37.50, (0.4, 7.9), 1, 4.5, (117.0, 423.0)),
    ("boss pocket", (0.0, 0.0), 15.75, (0.35, 1.95), -1, 1.15, None),
    ("pin hole", (0.0, -25.0), 3.00, (0.35, 1.95), -1, 1.15, None),
    ("cover outer", (0.0, 0.0), 37.50, (0.1, 8.0), 1, 4.5, (67.0, 113.0)),
    ("cover fillet +x", (5.5, 27.15), 10.00, (0.1, 2.8), 1, 1.45, (272.0, 358.0)),
    ("cover fillet -x", (-5.5, 27.15), 10.00, (0.1, 2.8), 1, 1.45, (182.0, 268.0)),
)


@pytest.mark.parametrize("geometry", ("visual/visual_glb/base.glb", "visual/base_link.stl"))
def test_gripper_g2_flange_and_boss_circles_are_round(geometry: str) -> None:
    """The centring boss, mounting flange, pin hole, and +Y connector cover have smooth analytic circular surfaces."""
    tri_mm = _triangles(_URDF_ROOT / "end_effectors" / "gripper_g2" / "meshes" / geometry) * 1000.0
    fn = np.cross(tri_mm[:, 1] - tri_mm[:, 0], tri_mm[:, 2] - tri_mm[:, 0])
    fn_norm = np.linalg.norm(fn, axis=1)
    valid = fn_norm > 1e-10
    assert int((~valid).sum()) <= 2
    tri_mm = tri_mm[valid]
    fn = fn[valid] / fn_norm[valid, None]
    centers = tri_mm.mean(axis=1)

    for name, xy, radius, (z_lo, z_hi), side, z_sec, arc in _G2_FLANGE_CIRCLES:
        axis = np.asarray(xy)
        d = centers[:, :2] - axis
        r = np.hypot(d[:, 0], d[:, 1])
        u = d / np.maximum(r[:, None], 1e-12)
        ang = np.degrees(np.arctan2(u[:, 1], u[:, 0])) % 360.0
        sel = (
            (tri_mm[:, :, 2].min(axis=1) >= z_lo - 0.5)
            & (tri_mm[:, :, 2].max(axis=1) <= z_hi + 0.5)
            & (np.abs(r - radius) < 0.8)
            & (side * (fn[:, :2] * u).sum(axis=1) > 0.85)
            & (np.abs(fn[:, 2]) < 0.1)
        )
        if arc is not None:
            a0, a1 = arc
            sel &= ((ang >= a0) | (ang <= a1 - 360.0)) if a1 > 360.0 else ((ang >= a0) & (ang <= a1))
        assert sel.sum() >= 60, f"{geometry} {name}: too few triangles ({sel.sum()})"
        sec = _section_mm(tri_mm[sel], z_sec)
        sec_d = sec - axis
        assert float(np.abs(np.hypot(sec_d[:, 0], sec_d[:, 1]) - radius).max()) < 0.02, f"{geometry} {name}"
        sa = np.sort(np.degrees(np.arctan2(sec_d[:, 1], sec_d[:, 0])) % 360.0)
        diffs = (
            np.diff(np.sort((sa - arc[0]) % 360.0))
            if arc is not None and arc[1] > 360.0
            else (np.diff(sa) if arc is not None else np.diff(np.r_[sa, sa[0] + 360.0]))
        )
        assert float(diffs.max()) < 6.0, f"{geometry} {name} gap {diffs.max():.2f} deg"

    if geometry.endswith(".glb"):
        DracoPy = pytest.importorskip("DracoPy")
        pygltflib = pytest.importorskip("pygltflib")
        glb = pygltflib.GLTF2().load(str(_URDF_ROOT / "end_effectors" / "gripper_g2" / "meshes" / geometry))
        blob = glb.binary_blob()
        pts_chunks, nrm_chunks = [], []
        for mesh in glb.meshes:
            for prim in mesh.primitives:
                view = glb.bufferViews[prim.extensions["KHR_draco_mesh_compression"]["bufferView"]]
                decoded = DracoPy.decode(blob[view.byteOffset : view.byteOffset + view.byteLength])
                pts_chunks.append(np.asarray(decoded.points, dtype=np.float64) * 1000.0)
                nrm_chunks.append(np.asarray(decoded.normals, dtype=np.float64))
        pts = np.vstack(pts_chunks)
        nrms = np.vstack(nrm_chunks)
        for name, xy, radius, (z_lo, z_hi), side, _, arc in _G2_FLANGE_CIRCLES:
            axis = np.asarray(xy)
            d = pts[:, :2] - axis
            r = np.hypot(d[:, 0], d[:, 1])
            u = d / np.maximum(r[:, None], 1e-12)
            ang = np.degrees(np.arctan2(u[:, 1], u[:, 0])) % 360.0
            sel = (
                (pts[:, 2] >= z_lo - 0.15)
                & (pts[:, 2] <= z_hi + 0.15)
                & (np.abs(r - radius) < 0.05)
                & (side * (nrms[:, :2] * u).sum(axis=1) > 0.9)
                & (np.abs(nrms[:, 2]) < 0.1)
            )
            if arc is not None:
                a0, a1 = arc
                sel &= ((ang >= a0) | (ang <= a1 - 360.0)) if a1 > 360.0 else ((ang >= a0) & (ang <= a1))
            assert sel.sum() >= 60, f"GLB normals {name}: too few vertices ({sel.sum()})"
            expected = side * np.c_[u[sel], np.zeros(sel.sum())]
            err_deg = np.degrees(np.arccos(np.clip((nrms[sel] * expected).sum(axis=1), -1.0, 1.0)))
            assert float(err_deg.max()) < 1.0, f"GLB normals {name}: {err_deg.max():.2f} deg"


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
        "bio_gripper_g2_visual.glb",
        "gripper_g2_static.glb",
        "gripper_g2_static_link",
        "lite6_gripper_static_link6.glb",
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


_REGISTERED_ARMS: dict[str, tuple[int, str, str]] = {
    "xarm5": (5, "xarm5_1305", "xarm5_1305"),
    "xarm6": (6, "xarm6_1305", "xarm6_1305"),
    "xarm7": (7, "xarm7_1305", "xarm7_1305"),
    "lite6": (6, "lite6", "lite6"),
    "uf850": (6, "uf850", "uf850"),
}


def _registered_arm_urdfs(robot_dir: str) -> list[Path]:
    """Tracked arm URDFs (dynamics + GLB visual) and every composed assembly of that arm."""
    _, mesh_prefix, key = _REGISTERED_ARMS[robot_dir]
    urdf_dir = _URDF_ROOT / "robots" / robot_dir
    tracked = [urdf_dir / f"{mesh_prefix}.urdf", urdf_dir / f"{mesh_prefix}_visual.glb.urdf"]
    assert sorted(p.name for p in urdf_dir.glob("*.urdf")) == sorted(p.name for p in tracked), robot_dir
    return tracked + _assembly_urdfs(robot=key)


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

_BIO_GRIPPER_G2_PARTS = (
    _URDF_ROOT / "end_effectors" / "bio_gripper_g2" / "bio_gripper_g2.urdf",
    _URDF_ROOT / "end_effectors" / "bio_gripper_g2" / "bio_gripper_g2_movable_visual.urdf",
)


def _bio_gripper_g2_collision_origin_urdfs() -> list[Path]:
    return [*_BIO_GRIPPER_G2_PARTS, *_assembly_urdfs("bio_gripper_g2")]


def _bio_gripper_g2_movable_collision_alignment_urdfs() -> list[Path]:
    return [_BIO_GRIPPER_G2_PARTS[1], *_assembly_urdfs("bio_gripper_g2", modes=("glb",))]


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


# Arm links built from another arm's parts, as (source arm, source link) in URDF element order. Hardware facts:
# xArm5 is xArm6 with the J4 module replaced by a fixed aluminium key at J4 = 0, and xArm5/6/7 share the production
# base (version A) and the flange link.
_ARM_PART_SOURCES: dict[str, dict[str, tuple[tuple[str, str], ...]]] = {
    "xarm5": {
        "link_base": (("xarm6", "link_base"),),
        "link1": (("xarm6", "link1"),),
        "link2": (("xarm6", "link2"),),
        "link3": (("xarm6", "link3"), ("xarm6", "link4")),
        "link4": (("xarm6", "link5"),),
        "link5": (("xarm6", "link6"),),
    },
    "xarm7": {
        "link_base": (("xarm6", "link_base"),),
        "link7": (("xarm6", "link6"),),
    },
}


def _part_sources(robot_dir: str, link_name: str) -> tuple[tuple[str, str], ...]:
    return _ARM_PART_SOURCES.get(robot_dir, {}).get(link_name, ((robot_dir, link_name),))


# Convex hull count of each arm's own link parts; shared parts are counted at their source.
_EXPECTED_ARM_COLLISION_COUNTS: dict[str, dict[str, int]] = {
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
        "link1": 2,
        "link2": 5,
        "link3": 4,
        "link4": 5,
        "link5": 5,
        "link6": 2,
    },
    "lite6": {
        "link_base": 3,
        "link1": 2,
        "link2": 4,
        "link3": 2,
        "link4": 4,
        "link5": 1,
        "link6": 3,
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
    for robot_dir, (dof, _mesh_prefix, _key) in _REGISTERED_ARMS.items():
        arm_links = _arm_link_names(dof)
        for urdf in _registered_arm_urdfs(robot_dir):
            basename = _rel(urdf)
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
                        arm_filenames.append(str((urdf.parent / filename).resolve()))
                if link_name not in arm_links:
                    continue
                if arm_filenames:
                    seen_arm_links.add(link_name)
                expected = [
                    str(
                        _URDF_ROOT
                        / "robots"
                        / src_dir
                        / f"meshes/{_REGISTERED_ARMS[src_dir][1]}/collision/{src_dir}_{src_link}_col_{idx:02d}.stl"
                    )
                    for src_dir, src_link in _part_sources(robot_dir, link_name)
                    for idx in range(_EXPECTED_ARM_COLLISION_COUNTS[src_dir][src_link])
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


_XARM_URDF_VARIANTS = ("{}_1305.urdf", "{}_1305_visual.glb.urdf")


def _rpy_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    (cr, cp, cy), (sr, sp, sy) = np.cos(rpy), np.sin(rpy)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _origin_matrix(elem: ET.Element) -> np.ndarray:
    origin = elem.find("origin")
    out = np.eye(4)
    out[:3, :3] = _rpy_matrix(_xyz_tuple(origin.get("rpy") if origin is not None else None))
    out[:3, 3] = _xyz_tuple(origin.get("xyz") if origin is not None else None)
    return out


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    out = np.eye(4)
    out[:3, :3] = np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)
    return out


def _link_poses(urdf: Path, q: dict[str, float]) -> dict[str, np.ndarray]:
    """World pose of every link for joint positions ``q``; joints not in ``q`` stay at 0."""
    root = ET.parse(urdf).getroot()
    joints = {joint.find("child").get("link"): joint for joint in root.findall("joint")}
    poses: dict[str, np.ndarray] = {}

    def pose(link: str) -> np.ndarray:
        if link not in poses:
            joint = joints.get(link)
            if joint is None:
                poses[link] = np.eye(4)
            else:
                out = pose(joint.find("parent").get("link")) @ _origin_matrix(joint)
                if joint.get("type") != "fixed":
                    axis = np.asarray(_xyz_tuple(joint.find("axis").get("xyz")))
                    out = out @ _axis_angle_matrix(axis / np.linalg.norm(axis), q.get(joint.get("name") or "", 0.0))
                poses[link] = out
        return poses[link]

    for link in root.findall("link"):
        pose(link.get("name") or "")
    return poses


def _placed_meshes(urdf: Path, link_name: str, tag: str) -> list[tuple[np.ndarray, Path]]:
    """(origin in the link frame, resolved mesh file) of each ``tag`` element of ``link_name``."""
    link = ET.parse(urdf).getroot().find(f"link[@name='{link_name}']")
    assert link is not None, (urdf, link_name)
    out = []
    for elem in link.findall(tag):
        mesh = elem.find("geometry/mesh")
        assert mesh is not None, (urdf, link_name, tag)
        out.append((_origin_matrix(elem), (urdf.parent / (mesh.get("filename") or "")).resolve()))
    return out


@pytest.mark.parametrize("variant", _XARM_URDF_VARIANTS)
def test_xarm5_is_xarm6_with_j4_locked(variant: str) -> None:
    """xArm5 link frames and placed meshes equal xArm6's at J4 = 0 (xArm5 J4/J5 are xArm6 J5/J6)."""
    xarm5 = _URDF_ROOT / "robots" / "xarm5" / variant.format("xarm5")
    xarm6 = _URDF_ROOT / "robots" / "xarm6" / variant.format("xarm6")
    frames = {link: _part_sources("xarm5", link)[0][1] for link in _arm_link_names(5)}
    for q in np.random.default_rng(0).uniform(-np.pi, np.pi, size=(8, 5)):
        poses5 = _link_poses(xarm5, {f"joint{i}": float(v) for i, v in zip((1, 2, 3, 4, 5), q)})
        poses6 = _link_poses(xarm6, {f"joint{i}": float(v) for i, v in zip((1, 2, 3, 5, 6), q)})
        for link, frame in frames.items():
            np.testing.assert_allclose(poses5[link], poses6[frame], atol=1e-9, err_msg=link)
            for tag in ("visual", "collision"):
                got = [(poses5[link] @ origin, path) for origin, path in _placed_meshes(xarm5, link, tag)]
                want = [
                    (poses6[src] @ origin, path)
                    for _, src in _part_sources("xarm5", link)
                    for origin, path in _placed_meshes(xarm6, src, tag)
                ]
                assert [path for _, path in got] == [path for _, path in want], (link, tag)
                for (a, _), (b, _) in zip(got, want):
                    np.testing.assert_allclose(a, b, atol=1e-9, err_msg=f"{link} {tag}")


@pytest.mark.parametrize("variant", _XARM_URDF_VARIANTS)
def test_shared_arm_links_reuse_the_source_link_meshes(variant: str) -> None:
    """A shared link with a single source carries exactly the source link's visuals and collisions."""
    for robot_dir, links in _ARM_PART_SOURCES.items():
        urdf = _URDF_ROOT / "robots" / robot_dir / variant.format(robot_dir)
        for link, sources in links.items():
            if len(sources) != 1:
                continue
            ((src_dir, src_link),) = sources
            source = _URDF_ROOT / "robots" / src_dir / variant.format(src_dir)
            for tag in ("visual", "collision"):
                got, want = _placed_meshes(urdf, link, tag), _placed_meshes(source, src_link, tag)
                assert [path for _, path in got] == [path for _, path in want], (robot_dir, link, tag)
                for (a, _), (b, _) in zip(got, want):
                    np.testing.assert_array_equal(a, b, err_msg=f"{robot_dir} {link} {tag}")


def test_xarm_bases_use_the_production_base() -> None:
    """Only base version A ships (flat top at z = 150.35 mm); prototype version B rises to 152.55 mm."""
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("DracoPy")
    pytest.importorskip("pygltflib")
    meshes = _URDF_ROOT / "robots" / "xarm6" / "meshes" / "xarm6_1305"
    stl, glb = meshes / "visual" / "link_base.stl", meshes / "visual_glb" / "link_base.glb"
    for robot_dir in ("xarm5", "xarm6", "xarm7"):
        for variant, visual in zip(_XARM_URDF_VARIANTS, (stl, glb)):
            urdf = _URDF_ROOT / "robots" / robot_dir / variant.format(robot_dir)
            assert [path for _, path in _placed_meshes(urdf, "link_base", "visual")] == [visual], _rel(urdf)
    hulls = [
        path
        for _, path in _placed_meshes(_URDF_ROOT / "robots" / "xarm6" / "xarm6_1305.urdf", "link_base", "collision")
    ]
    stl_top = _mesh_vertices(stl, trimesh)[:, 2].max()
    assert stl_top == pytest.approx(0.15035, abs=5e-5)
    assert _mesh_vertices(glb, trimesh)[:, 2].max() == pytest.approx(stl_top, abs=1e-4)
    assert max(_mesh_vertices(path, trimesh)[:, 2].max() for path in hulls) == pytest.approx(stl_top, abs=1e-4)


def test_xarm_bases_share_unified_inertial_parameters() -> None:
    """xArm5/6/7 share the SolidWorks Xarm+_P01 link_base inertials (standard URDF negative off-diagonals)."""
    expected_mass = 2.788411
    expected_com = (-0.002587, -0.000269, 0.103569)
    expected_rpy = (0.0, 0.0, 0.0)
    expected_inertia = {
        "ixx": 0.008738440348,
        "ixy": -1.7243198e-05,
        "ixz": -0.000169824247,
        "iyy": 0.008801398397,
        "iyz": -4.797104e-06,
        "izz": 0.003241739845,
    }
    expected_principal = np.array([3236492.910, 8739003.273, 8806082.405], dtype=np.float64) * 1e-9

    serialized: list[str] = []
    for robot_dir in ("xarm5", "xarm6", "xarm7"):
        for variant in _XARM_URDF_VARIANTS:
            urdf = _URDF_ROOT / "robots" / robot_dir / variant.format(robot_dir)
            inertial = ET.parse(urdf).getroot().find("./link[@name='link_base']/inertial")
            assert inertial is not None, _rel(urdf)

            mass = float(inertial.find("mass").attrib["value"])
            assert mass == pytest.approx(expected_mass, rel=0.0, abs=1e-12), _rel(urdf)

            origin = inertial.find("origin")
            xyz = tuple(float(v) for v in origin.attrib["xyz"].split())
            rpy = tuple(float(v) for v in origin.attrib.get("rpy", "0 0 0").split())
            assert xyz == pytest.approx(expected_com, rel=0.0, abs=1e-12), _rel(urdf)
            assert rpy == pytest.approx(expected_rpy, rel=0.0, abs=1e-12), _rel(urdf)

            inertia = {k: float(v) for k, v in inertial.find("inertia").attrib.items()}
            for key, expected_val in expected_inertia.items():
                assert inertia[key] == pytest.approx(expected_val, rel=0.0, abs=1e-12), (_rel(urdf), key)

            serialized.append(ET.tostring(inertial, encoding="unicode").strip())

    assert len(set(serialized)) == 1

    tensor = np.array(
        [
            [expected_inertia["ixx"], expected_inertia["ixy"], expected_inertia["ixz"]],
            [expected_inertia["ixy"], expected_inertia["iyy"], expected_inertia["iyz"]],
            [expected_inertia["ixz"], expected_inertia["iyz"], expected_inertia["izz"]],
        ],
        dtype=np.float64,
    )
    eig = np.linalg.eigvalsh(tensor)
    assert eig.min() > 0.0
    assert eig[0] + eig[1] > eig[2]
    np.testing.assert_allclose(eig, expected_principal, rtol=0.0, atol=1e-11)


def test_uf850_base_uses_solidworks_inertial_parameters() -> None:
    """UF850 STL and GLB URDFs share the SolidWorks Arm850_P01 link_base inertials (URDF negative off-diagonals)."""
    expected_mass = 5.869631
    expected_com = (-0.000762, -4.4e-05, 0.11761)
    expected_rpy = (0.0, 0.0, 0.0)
    expected_inertia = {
        "ixx": 0.032711870245,
        "ixy": 8.0733473e-05,
        "ixz": -0.000343329636,
        "iyy": 0.03295126767,
        "iyz": -1.0328638e-05,
        "izz": 0.013458380527,
    }
    expected_principal = np.array([13452256.142, 32692678.345, 32976583.954], dtype=np.float64) * 1e-9

    serialized: list[str] = []
    for filename in ("uf850.urdf", "uf850_visual.glb.urdf"):
        urdf = _URDF_ROOT / "robots" / "uf850" / filename
        inertial = ET.parse(urdf).getroot().find("./link[@name='link_base']/inertial")
        assert inertial is not None, _rel(urdf)

        mass = float(inertial.find("mass").attrib["value"])
        assert mass == pytest.approx(expected_mass, rel=0.0, abs=1e-12), _rel(urdf)

        origin = inertial.find("origin")
        xyz = tuple(float(v) for v in origin.attrib["xyz"].split())
        rpy = tuple(float(v) for v in origin.attrib.get("rpy", "0 0 0").split())
        assert xyz == pytest.approx(expected_com, rel=0.0, abs=1e-12), _rel(urdf)
        assert rpy == pytest.approx(expected_rpy, rel=0.0, abs=1e-12), _rel(urdf)

        inertia = {k: float(v) for k, v in inertial.find("inertia").attrib.items()}
        for key, expected_val in expected_inertia.items():
            assert inertia[key] == pytest.approx(expected_val, rel=0.0, abs=1e-12), (_rel(urdf), key)

        serialized.append(ET.tostring(inertial, encoding="unicode").strip())

    assert len(set(serialized)) == 1

    tensor = np.array(
        [
            [expected_inertia["ixx"], expected_inertia["ixy"], expected_inertia["ixz"]],
            [expected_inertia["ixy"], expected_inertia["iyy"], expected_inertia["iyz"]],
            [expected_inertia["ixz"], expected_inertia["iyz"], expected_inertia["izz"]],
        ],
        dtype=np.float64,
    )
    eig = np.linalg.eigvalsh(tensor)
    assert eig.min() > 0.0
    assert eig[0] + eig[1] > eig[2]
    np.testing.assert_allclose(eig, expected_principal, rtol=0.0, atol=1e-11)


def test_xarm_flange_visual_is_centred_on_the_flange_frame() -> None:
    """xArm5/6/7 share one flange link; its GLB stays centred on the flange axis like the STL.

    The former xArm5 copy of this GLB sat 0.9 mm off in the flange plane.
    """
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("DracoPy")
    pytest.importorskip("pygltflib")
    meshes = _URDF_ROOT / "robots" / "xarm6" / "meshes" / "xarm6_1305"
    stl, glb = meshes / "visual" / "link6.stl", meshes / "visual_glb" / "link6.glb"
    for robot_dir, flange in (("xarm5", "link5"), ("xarm6", "link6"), ("xarm7", "link7")):
        for variant, visual in zip(_XARM_URDF_VARIANTS, (stl, glb)):
            urdf = _URDF_ROOT / "robots" / robot_dir / variant.format(robot_dir)
            assert [path for _, path in _placed_meshes(urdf, flange, "visual")] == [visual], _rel(urdf)

    def bbox_centre(path: Path) -> np.ndarray:
        vertices = _mesh_vertices(path, trimesh)
        return (vertices.min(axis=0) + vertices.max(axis=0)) / 2

    assert np.allclose(bbox_centre(glb)[:2], bbox_centre(stl)[:2], atol=2e-4)


def test_lite6_base_glb_is_aligned_with_stl_at_z0() -> None:
    """Lite 6 link_base.glb bottom mounting plane sits at z = 0 matching link_base.stl."""
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("DracoPy")
    pytest.importorskip("pygltflib")
    meshes = _URDF_ROOT / "robots" / "lite6" / "meshes" / "lite6"
    stl = _mesh_vertices(meshes / "visual" / "link_base.stl", trimesh)
    glb = _mesh_vertices(meshes / "visual_glb" / "link_base.glb", trimesh)
    assert float(glb[:, 2].min()) == pytest.approx(float(stl[:, 2].min()), abs=2e-5)
    assert float(glb[:, 2].max()) == pytest.approx(float(stl[:, 2].max()), abs=5e-4)


def test_arm_and_bio_g2_collision_meshes_are_convex_hulls() -> None:
    """Genesis keeps its pure-convex path, and the Bio G2 collision audit checks these meshes as convex shapes."""
    trimesh = pytest.importorskip("trimesh")
    urdfs = [
        _URDF_ROOT / "robots" / robot_dir / f"{prefix}.urdf" for robot_dir, (_, prefix, _) in _REGISTERED_ARMS.items()
    ]
    meshes = sorted(
        {
            (urdf.parent / name).resolve()
            for urdf in (*urdfs, *_BIO_GRIPPER_G2_PARTS)
            for name in _collision_mesh_filenames(urdf)
        }
    )
    bad = []
    for path in meshes:
        mesh = trimesh.load(path, force="mesh")
        if not mesh.is_watertight or 1.0 - mesh.volume / mesh.convex_hull.volume > 1e-5:
            bad.append(_rel(path))
    assert meshes and bad == []


def test_bio_gripper_g2_collision_mesh_basenames_are_unique_in_combos() -> None:
    """MuJoCo keys URDF mesh assets by basename, so Bio G2 combo basenames must not collide."""
    bad: list[str] = []
    for urdf in _assembly_urdfs("bio_gripper_g2"):
        names = [Path(filename).name for filename in _collision_mesh_filenames(urdf)]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            bad.append(f"{_rel(urdf)}: duplicate collision mesh basename(s) {duplicates}")
    assert bad == []


def test_bio_gripper_g2_urdfs_do_not_reference_legacy_short_stl_names() -> None:
    legacy = {
        (_URDF_ROOT / "end_effectors" / "bio_gripper_g2" / "meshes" / "visual" / name).resolve()
        for name in ("link_base.stl", "left_finger.stl", "right_finger.stl")
    }
    bad: list[str] = []
    for urdf in _bio_gripper_g2_collision_origin_urdfs():
        for filename in _mesh_filenames(urdf):
            if (urdf.parent / filename).resolve() in legacy:
                bad.append(f"{_rel(urdf)}: legacy Bio G2 STL reference {filename}")
    assert bad == []


def test_bio_gripper_g2_collision_origins_align_stl_to_glb_reference() -> None:
    bad: list[str] = []
    for urdf in _bio_gripper_g2_collision_origin_urdfs():
        actual = _collision_origin_pose_by_link(urdf)
        for link_name, expected_xyz in _BIO_GRIPPER_G2_COLLISION_ORIGINS.items():
            pose = actual.get(link_name)
            if pose is None:
                bad.append(f"{_rel(urdf)}: missing collision origin for {link_name}")
                continue
            actual_xyz, actual_rpy = pose
            if actual_xyz != expected_xyz:
                bad.append(f"{_rel(urdf)}: {link_name} collision origin xyz should be {expected_xyz}, got {actual_xyz}")
            if actual_rpy != (0.0, 0.0, 0.0):
                bad.append(
                    f"{_rel(urdf)}: {link_name} collision origin rpy should be (0.0, 0.0, 0.0), got {actual_rpy}"
                )
    assert bad == []


def test_bio_gripper_g2_movable_collision_bbox_centers_match_visual_glb() -> None:
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("DracoPy")
    pytest.importorskip("pygltflib")
    bad: list[str] = []
    for urdf in _bio_gripper_g2_movable_collision_alignment_urdfs():
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
                bad.append(f"{_rel(urdf)}: {link_name} missing visual/collision mesh")
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
                        f"{_rel(urdf)}: {link_name} inner gripping plane mismatch "
                        f"vis={vis_inner_y:.6f} col={col_inner_y:.6f}"
                    )
        if abs(sum(masses) - 0.79) > 1e-6:
            bad.append(f"{_rel(urdf)}: Bio Gripper G2 total mass {sum(masses)} != 0.79 kg")
    assert bad == []


def test_bio_gripper_g2_visual_glb_budget() -> None:
    """High-poly + native Draco should keep Bio G2 visual GLBs under 1 MiB."""
    package_dir = PROJECT_ROOT / "assets" / "urdf" / "end_effectors" / "bio_gripper_g2"
    visual_dir = package_dir / "meshes" / "visual"
    glbs = sorted(
        path
        for path in visual_dir.rglob("*.glb")
        if "visual_glb_src" not in path.parts and "visual_glb_raw" not in path.parts
    )
    assert glbs, "expected Bio G2 visual GLBs"
    glb_total = sum(path.stat().st_size for path in glbs)
    assert glb_total < 1_000_000, f"Bio G2 visual GLBs total {glb_total} bytes; expected < 1 MB"
    names = {path.name for path in glbs}
    assert names == {
        "bio_gripper_g2_base.glb",
        "bio_gripper_g2_left_finger.glb",
        "bio_gripper_g2_right_finger.glb",
    }


_BIO_G2_GLBS = ["visual_glb/bio_gripper_g2_base.glb"]
_PCD = 25.0 * np.sqrt(0.5)
_LUG = {"+y": (15.01, 43.07), "-y": (15.04, -43.02)}  # lug arc / lug hole / screw axis
_PIN = (0.044, -24.990)  # stepped pin hole at -90 deg
# Adapter-plate surfaces of revolution in the adapter frame (mm): centre; radius r0 and slope of r = r0 + slope * z;
# z band; +1 if the solid is on the axis side (convex), -1 if not (hole, concave fillet); azimuth ranges (deg, 0..360).
_ADAPTER_CIRCLES = {
    "rim": ((0.0, 0.0), 37.59, 0.0, (1.6, 3.4), 1, ((0, 42), (90, 270), (318, 360))),  # between the lugs
    "bore": ((0.0, 0.0), 15.97, 0.0, (1.6, 3.4), -1, None),
    "M6 hole 45": ((_PCD, _PCD), 3.13, 0.0, (0.2, 0.9), -1, None),
    "M6 hole 135": ((-_PCD, _PCD), 3.13, 0.0, (0.2, 0.9), -1, None),
    "M6 hole -135": ((-_PCD, -_PCD), 3.13, 0.0, (0.2, 0.9), -1, None),
    "M6 hole -45": ((_PCD, -_PCD), 3.13, 0.0, (0.2, 0.9), -1, None),
    "lug arc +y": (_LUG["+y"], 8.02, 0.0, (1.6, 3.4), 1, ((10, 170),)),
    "lug arc -y": (_LUG["-y"], 8.02, 0.0, (1.6, 3.4), 1, ((190, 350),)),
    "lug arc fillet +y": ((2.28, 42.27), 4.74, 0.0, (1.1, 3.9), -1, ((277, 353),)),
    "lug arc fillet -y": ((2.31, -42.26), 4.74, 0.0, (1.1, 3.9), -1, ((7, 83),)),
    "lug flank fillet +y": ((36.26, 35.69), 13.27, 0.0, (1.1, 3.9), -1, ((185, 219),)),
    "lug flank fillet -y": ((36.30, -35.65), 13.27, 0.0, (1.1, 3.9), -1, ((141, 175),)),
    "lug countersink +y": (_LUG["+y"], 7.01, -1.0, (0.4, 3.4), -1, None),
    "lug countersink -y": (_LUG["-y"], 7.01, -1.0, (0.4, 3.4), -1, None),
    "lug screw head +y": (_LUG["+y"], 5.68, 0.0, (0.15, 0.6), 1, None),
    "lug screw head -y": (_LUG["-y"], 5.68, 0.0, (0.15, 0.6), 1, None),
    # Remeshed: vertices only on the end circles. The counterbore's band leaves out its lower circle, which it shares
    # with the entry chamfer.
    "pin counterbore": (_PIN, 2.54, 0.0, (0.25, 3.1), -1, None),
    "pin hole": (_PIN, 1.04, 0.0, (2.9, 5.1), -1, None),
}
# Hole rims where the contact face (z = 0) and the plate top (z = 5) end: centre, radius at z = 0, radius at z = 5.
_ADAPTER_RIMS = {
    "bore": ((0.031, 0.026), 16.972, 16.972),
    "M6 hole 45": ((17.688, 17.695), 3.129, 7.025),
    "M6 hole 135": ((-17.657, 17.678), 3.129, 7.025),
    "M6 hole -135": ((-17.647, -17.670), 3.129, 7.025),
    "M6 hole -45": ((17.700, -17.656), 3.129, 7.025),
    "pin hole": (_PIN, 2.739, 1.040),
}


def _bio_g2_adapter_metal(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Metallic primitive of a Bio G2 visual GLB in the adapter frame (README): contact face z = 0, +Z to the housing.
    Returns points (mm), normals and triangles."""
    DracoPy = pytest.importorskip("DracoPy")
    pygltflib = pytest.importorskip("pygltflib")
    from scipy.spatial.transform import Rotation

    glb = pygltflib.GLTF2().load(str(_URDF_ROOT / "end_effectors" / "bio_gripper_g2" / "meshes" / "visual" / name))
    blob = glb.binary_blob()
    metal = [i for i, m in enumerate(glb.materials) if m.pbrMetallicRoughness.metallicFactor > 0.5]
    prim = next(p for m in glb.meshes for p in m.primitives if p.material in metal)
    view = glb.bufferViews[prim.extensions["KHR_draco_mesh_compression"]["bufferView"]]
    mesh = DracoPy.decode(blob[view.byteOffset : view.byteOffset + view.byteLength])
    rotation = Rotation.from_euler("xyz", np.radians([0.031, 0.254, 0.360])).as_matrix()
    points = (np.asarray(mesh.points, np.float64) * 1000 - [-22.636, -0.956, -2.516]) @ rotation
    return points, np.asarray(mesh.normals, np.float64) @ rotation, np.asarray(mesh.faces, np.int64)


@pytest.mark.parametrize("feature", list(_ADAPTER_CIRCLES))
@pytest.mark.parametrize("name", _BIO_G2_GLBS)
def test_bio_gripper_g2_adapter_circles_are_round(name: str, feature: str) -> None:
    """The adapter plate's curved faces are surfaces of revolution with analytic normals: rim, centre bore, the four
    countersunk M6 holes, on each lug the arc, its two fillets, the countersunk hole and the screw head in it, and the
    stepped pin hole.

    The export had them as twisted facet strips, up to 1.1 mm (rim, bore), 0.95 mm (pin hole), 0.8 mm (lug holes) and
    0.5 mm (M6 holes) off, with flat normals up to 25-85 deg off.
    """
    points, normals, _ = _bio_g2_adapter_metal(name)
    centre, nominal, slope, (z_low, z_high), side, arcs = _ADAPTER_CIRCLES[feature]
    z = points[:, 2]

    def frame(c):
        d = points[:, :2] - c
        r = np.linalg.norm(d, axis=1)
        return d / r[:, None], r - slope * z  # radial unit, equivalent radius at z = 0

    u, equivalent = frame(np.asarray(centre))
    # Normals point out of the solid: away from the axis when the solid is on its side, towards it otherwise. The
    # threshold leaves out flat faces at the same height, whose quantized normals tilt by up to ~0.01.
    facing = side * (normals[:, :2] * u).sum(1) > 0.3
    surface = (z > z_low) & (z < z_high) & (np.abs(equivalent - nominal) < 0.6) & facing
    if arcs is not None:
        azimuth = np.degrees(np.arctan2(u[:, 1], u[:, 0])) % 360
        surface &= np.any([(azimuth >= lo) & (azimuth <= hi) for lo, hi in arcs], axis=0)
    xy = np.asarray(centre) + equivalent[surface, None] * u[surface]
    fx, fy, k = np.linalg.lstsq(np.c_[2 * xy, np.ones(len(xy))], (xy**2).sum(1), rcond=None)[0]
    fitted = np.sqrt(k + fx**2 + fy**2)
    residual = np.hypot(xy[:, 0] - fx, xy[:, 1] - fy) - fitted
    u, _ = frame(np.array([fx, fy]))
    expected = side * np.c_[u, np.full(len(u), -slope)] / np.hypot(1.0, slope)
    normal_error = np.degrees(np.arccos(np.clip((normals * expected).sum(1), -1.0, 1.0)))[surface]
    assert surface.sum() > 30
    assert np.abs(residual).max() < 0.03, f"surface deviates {np.abs(residual).max():.3f} mm from its circle"
    assert normal_error.max() < 1.0, f"normals up to {normal_error.max():.1f} deg off the analytic normal"
    assert fitted == pytest.approx(nominal, abs=0.05)  # free fit: short fillet arcs leave ~0.04 mm of radius play


@pytest.mark.parametrize("name", _BIO_G2_GLBS)
def test_bio_gripper_g2_adapter_has_through_holes_not_screws(name: str) -> None:
    """The real adapter has open M6 holes at 135/-135 deg (PCD 50).

    The export carried the two mounting screws, their tips standing 5 mm proud of the contact face.
    """
    points, _, _ = _bio_g2_adapter_metal(name)
    assert points[:, 2].min() > -0.1, f"metal {-points[:, 2].min():.2f} mm proud of the contact face"
    for azimuth in (135.0, -135.0):
        axis = 25.0 * np.array([np.cos(np.radians(azimuth)), np.sin(np.radians(azimuth))])
        assert not (np.hypot(*(points[:, :2] - axis).T) < 2.5).any(), f"geometry inside the {azimuth:+.0f} deg hole"


@pytest.mark.parametrize("name", _BIO_G2_GLBS)
def test_bio_gripper_g2_adapter_faces_stop_at_hole_rims(name: str) -> None:
    """The contact face and the plate top end at each hole's round rim, so the holes look round from both sides.

    The export's flat faces ended at the old polygonal rims and filled T-junctions there with slivers. After the rims
    were rounded, a sliver face reached 0.16 mm into the +45 deg M6 hole over 37 deg of its rim (a straight edge on
    the round hole), and T-junction vertices and long rim edges up to 0.09 mm into others. The limit is about two
    quantization steps.
    """
    points, normals, faces = _bio_g2_adapter_metal(name)
    normal_z, tri = normals[faces][:, :, 2], points[faces]
    deepest = {}
    for hole, (centre, rim_bottom, rim_top) in _ADAPTER_RIMS.items():
        c = np.asarray(centre)
        for z_face, sign, rim in ((0.0, -1.0, rim_bottom), (5.0, 1.0, rim_top)):
            flat = (sign * normal_z > 0.99).all(axis=1) & (np.abs(tri[:, :, 2] - z_face) < 0.03).all(axis=1)
            t = tri[flat][:, :, :2]
            t = t[np.linalg.norm(t - c, axis=2).min(axis=1) < rim + 1.0]
            # Distance from the hole's axis to each triangle (0 when the triangle covers the axis).
            dist = np.full(len(t), np.inf)
            for i in range(3):
                a, d = t[:, i], t[:, (i + 1) % 3] - t[:, i]
                s = np.clip(((c - a) * d).sum(1) / np.maximum((d * d).sum(1), 1e-30), 0.0, 1.0)
                dist = np.minimum(dist, np.linalg.norm(a + s[:, None] * d - c, axis=1))
            e, w = [t[:, (i + 1) % 3] - t[:, i] for i in range(3)], [c - t[:, i] for i in range(3)]
            side = [np.sign(e[i][:, 0] * w[i][:, 1] - e[i][:, 1] * w[i][:, 0]) for i in range(3)]
            dist[(side[0] == side[1]) & (side[1] == side[2])] = 0.0
            deepest[f"{hole} z{z_face:.0f}"] = float(rim - dist.min())
    worst = max(deepest, key=deepest.get)
    assert deepest[worst] < 0.025, f"flat faces reach {deepest[worst]:.3f} mm into the {worst} rim"


def test_gripper_g2_visual_glb_budget() -> None:
    """Canonical Gripper G2 visual GLBs (native Draco) must stay under 1 MiB."""
    package_dir = PROJECT_ROOT / "assets" / "urdf" / "end_effectors" / "gripper_g2"
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
    assert names == {
        "base.glb",
        "left_finger.glb",
        "left_inner_knuckle.glb",
        "left_outer_knuckle.glb",
        "right_finger.glb",
        "right_inner_knuckle.glb",
        "right_outer_knuckle.glb",
    }
    assert not any("static" in name for name in names)
    assert not any(part in {"link5", "link6", "link7"} for path in glbs for part in path.parts)


def test_lite6_gripper_visual_glb_budget() -> None:
    """Lite6 gripper visual GLBs (native Draco) must stay under 1 MiB."""
    package_dir = PROJECT_ROOT / "assets" / "urdf" / "end_effectors" / "lite6_gripper"
    visual_dir = package_dir / "meshes" / "visual"
    glbs = sorted(visual_dir.rglob("*.glb"))
    assert glbs, "expected Lite6 gripper visual GLBs"
    glb_total = sum(path.stat().st_size for path in glbs)
    assert glb_total < 1_000_000, f"Lite6 gripper visual GLBs total {glb_total} bytes; expected < 1 MB"
    names = {path.name for path in glbs}
    assert names == {"shell.glb"}
    assert "lite6_gripper_static_link6.glb" not in names


def test_lite6_vacuum_gripper_visual_glb_budget() -> None:
    """Lite6 vacuum gripper visual GLB (native Draco) must stay under 1 MiB."""
    package_dir = PROJECT_ROOT / "assets" / "urdf" / "end_effectors" / "lite6_vacuum_gripper"
    visual_dir = package_dir / "meshes" / "visual"
    glbs = sorted(visual_dir.rglob("*.glb"))
    assert glbs, "expected Lite6 vacuum gripper visual GLBs"
    glb_total = sum(path.stat().st_size for path in glbs)
    assert glb_total < 1_000_000, f"Lite6 vacuum visual GLBs total {glb_total} bytes; expected < 1 MB"
    names = {path.name for path in glbs}
    assert "lite6_vacuum_gripper_visual_link6.glb" in names


def test_vacuum_gripper_visual_glb_budget() -> None:
    """xArm/UF850 vacuum gripper visual GLB (native Draco) must stay under 1 MiB."""
    package_dir = PROJECT_ROOT / "assets" / "urdf" / "end_effectors" / "vacuum_gripper"
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
            "xarm6",
            "assets/urdf/robots/xarm6/meshes/xarm6_1305/visual_glb",
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
            "assets/urdf/robots/xarm7/meshes/xarm7_1305/visual_glb",
            1_000_000,
            {"link1.glb", "link2.glb", "link3.glb", "link4.glb", "link5.glb", "link6.glb"},
        ),
        (
            "uf850",
            "assets/urdf/robots/uf850/meshes/uf850/visual_glb",
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
            "assets/urdf/robots/lite6/meshes/lite6/visual_glb",
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
        _URDF_ROOT / "end_effectors" / "gripper_g2" / "gripper_g2_movable_visual.urdf",
        *_BIO_GRIPPER_G2_PARTS,
        _URDF_ROOT / "end_effectors" / "lite6_gripper" / "lite6_gripper_movable_visual.urdf",
        _URDF_ROOT / "end_effectors" / "lite6_vacuum_gripper" / "lite6_vacuum_gripper.urdf",
        _URDF_ROOT / "end_effectors" / "vacuum_gripper" / "vacuum_gripper.urdf",
        _URDF_ROOT / "mounts" / "xarm_mounting_plate" / "xarm_mounting_plate.urdf",
        _URDF_ROOT / "mounts" / "uf850_mounting_plate" / "uf850_mounting_plate.urdf",
        _URDF_ROOT / "mounts" / "lite6_mounting_plate" / "lite6_mounting_plate.urdf",
        *_assembly_urdfs(),
    )
    bad: list[str] = []
    for urdf in urdfs:
        for filename in _collision_mesh_filenames(urdf):
            if not filename.endswith(".stl"):
                bad.append(f"{_rel(urdf)}: collision mesh not STL {filename}")
                continue
            target = (urdf.parent / filename).resolve()
            if not target.exists():
                bad.append(f"{_rel(urdf)}: missing collision mesh {filename}")
    assert bad == []


def test_standalone_movable_accessory_collision_links_are_movable() -> None:
    expected = {
        PROJECT_ROOT / "assets" / "urdf" / "end_effectors" / "gripper_g2" / "gripper_g2_movable_visual.urdf": {
            "left_outer_knuckle",
            "left_finger",
            "left_inner_knuckle",
            "right_outer_knuckle",
            "right_finger",
            "right_inner_knuckle",
        },
        PROJECT_ROOT / "assets" / "urdf" / "end_effectors" / "bio_gripper_g2" / "bio_gripper_g2_movable_visual.urdf": {
            "bio_gripper_g2_left_finger",
            "bio_gripper_g2_right_finger",
        },
        PROJECT_ROOT / "assets" / "urdf" / "end_effectors" / "lite6_gripper" / "lite6_gripper_movable_visual.urdf": {
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
                bad.append(f"{_rel(urdf)}: {link} has no collision geometry")
            # Mesh-based accessories still require a collision mesh; Lite6 fingers use boxes.
            if urdf.name != "lite6_gripper_movable_visual.urdf" and link not in collision_mesh_by_link:
                bad.append(f"{_rel(urdf)}: {link} has no collision mesh")
            if link not in non_fixed_children:
                bad.append(f"{_rel(urdf)}: {link} is not child of a movable joint")
    assert bad == []


def test_lite6_vacuum_standalone_collision_is_static_only() -> None:
    urdfs = (_URDF_ROOT / "end_effectors" / "lite6_vacuum_gripper" / "lite6_vacuum_gripper.urdf",)
    bad: list[str] = []
    for urdf in urdfs:
        movable_joints = sorted(_non_fixed_joint_names(urdf))
        if movable_joints:
            bad.append(f"{_rel(urdf)}: unexpected movable joints {movable_joints}")
    assert bad == []


def test_arm_movable_accessory_collision_links_match_standalone_models() -> None:
    cases = (
        (
            PROJECT_ROOT / "assets" / "urdf" / "end_effectors" / "gripper_g2" / "gripper_g2_movable_visual.urdf",
            {
                "left_outer_knuckle",
                "left_finger",
                "left_inner_knuckle",
                "right_outer_knuckle",
                "right_finger",
                "right_inner_knuckle",
            },
            tuple(_assembly_urdfs("gripper_g2")),
        ),
        (
            PROJECT_ROOT
            / "assets"
            / "urdf"
            / "end_effectors"
            / "bio_gripper_g2"
            / "bio_gripper_g2_movable_visual.urdf",
            {
                "bio_gripper_g2_left_finger",
                "bio_gripper_g2_right_finger",
            },
            tuple(_assembly_urdfs("bio_gripper_g2")),
        ),
        (
            PROJECT_ROOT / "assets" / "urdf" / "end_effectors" / "lite6_gripper" / "lite6_gripper_movable_visual.urdf",
            {
                "uflite_finger1",
                "uflite_finger2",
            },
            tuple(_assembly_urdfs("lite6_gripper")),
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
                            f"{_rel(combo)}: {link} collision boxes {actual_boxes} "
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
                        f"{_rel(combo)}: {link} collision {actual_meshes} does not match standalone {expected_meshes}"
                    )
    assert bad == []


def _lite6_gripper_finger_urdfs() -> list[Path]:
    """Lite6 gripper part and composed assemblies that keep per-finger visuals (GLB and STL arm visuals)."""
    return [
        _URDF_ROOT / "end_effectors" / "lite6_gripper" / "lite6_gripper_movable_visual.urdf",
        *_assembly_urdfs("lite6_gripper", modes=("glb", "stl")),
    ]


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
            out[link_name] = str((urdf.parent / (mesh.get("filename") or "")).resolve())
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
    mesh_dir = PROJECT_ROOT / "assets" / "urdf" / "end_effectors" / "lite6_gripper" / "meshes" / "collision"
    finger1_vertices = _mesh_vertices(mesh_dir / "finger1.stl", trimesh)
    finger2_vertices = _mesh_vertices(mesh_dir / "finger2.stl", trimesh)

    for urdf in _lite6_gripper_finger_urdfs():
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
            f"{_rel(urdf)}: finger1 visual origin should align with the mesh frame, got {vis1}"
        )
        assert vis2 == (0.0, 0.0, 0.0), (
            f"{_rel(urdf)}: finger2 visual origin should align with the mesh frame, got {vis2}"
        )
        assert visual_meshes == {
            "uflite_finger1": str(mesh_dir / "finger1.stl"),
            "uflite_finger2": str(mesh_dir / "finger2.stl"),
        }
        assert boxes.get("uflite_finger1") == _LITE6_FINGER_BOXES["uflite_finger1"], (
            f"{_rel(urdf)}: finger1 dual-box collision mismatch"
        )
        assert boxes.get("uflite_finger2") == _LITE6_FINGER_BOXES["uflite_finger2"], (
            f"{_rel(urdf)}: finger2 dual-box collision mismatch"
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

        assert 14.0 <= closed_sep <= 17.0, f"{_rel(urdf)}: closed finger center separation {closed_sep:.2f} mm"
        assert 32.0 <= open_sep <= 35.0, f"{_rel(urdf)}: open finger center separation {open_sep:.2f} mm"
        assert open_sep > closed_sep, f"{_rel(urdf)}: opening should increase collision finger separation"
        assert closed_gap == pytest.approx(0.0, abs=0.5), (
            f"{_rel(urdf)}: closed root-region inner gap should overlap at the finger root"
        )
        assert mid_gap == pytest.approx(8.9, abs=0.5), (
            f"{_rel(urdf)}: mid-stroke root-region inner gap should track prismatic travel"
        )
        assert open_gap == pytest.approx(17.8, abs=0.5), (
            f"{_rel(urdf)}: open root-region inner gap should track prismatic travel"
        )


def test_lite6_vacuum_collision_uses_visual_stl() -> None:
    expected = {
        (
            _URDF_ROOT / "end_effectors" / "lite6_vacuum_gripper" / "meshes" / "visual" / "vacuum_gripper_lite.stl"
        ).resolve()
    }
    urdfs = (
        _URDF_ROOT / "end_effectors" / "lite6_vacuum_gripper" / "lite6_vacuum_gripper.urdf",
        *_assembly_urdfs("lite6_vacuum_gripper"),
    )
    bad: list[str] = []
    for urdf in urdfs:
        actual = {
            (urdf.parent / filename).resolve()
            for filename in _collision_mesh_filenames(urdf)
            if "vacuum_gripper_lite.stl" in filename
        }
        if actual != expected:
            bad.append(f"{_rel(urdf)}: expected {sorted(expected)}, got {sorted(actual)}")
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
    from ufactory.robots.registry import get_robot_profile

    cases = [(_URDF_ROOT / "end_effectors" / "vacuum_gripper" / "vacuum_gripper.urdf", "vacuum_mount", 0)]
    for key in end_effectors()["vacuum_gripper"].robots:
        profile = get_robot_profile(key)
        cases += [(urdf, profile.ee_link, profile.dof) for urdf in _assembly_urdfs("vacuum_gripper", robot=key)]
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


@pytest.mark.parametrize("variant", ("robotiq_2f_85", "robotiq_2f_140"))
def test_robotiq_urdfs_share_joint_limits_mimic_and_glb_budget(variant: str) -> None:
    from ufactory.grippers.robotiq import get_robotiq_spec

    spec = get_robotiq_spec(variant)
    glb_dir = _URDF_ROOT / "end_effectors" / variant / "meshes" / "visual" / "visual_glb"
    glb_total = sum(p.stat().st_size for p in glb_dir.glob("*.glb"))
    assert 0 < glb_total < 1_000_000, f"{variant} GLB total {glb_total} bytes exceeds 1 MB"

    part = _URDF_ROOT / "end_effectors" / variant / f"{variant}_movable_visual.urdf"
    models = [part, *_assembly_urdfs(variant)]
    assert len(models) == 9
    for path in models:
        root = ET.parse(path).getroot()
        for name in spec.joint_names:
            joint = root.find(f"./joint[@name='{name}']")
            assert joint is not None, (path, name)
            limit = joint.find("limit")
            assert limit is not None, (path, name)
            assert float(limit.attrib["lower"]) == pytest.approx(spec.linkage.drive_lower_rad, abs=1e-9), path
            assert float(limit.attrib["upper"]) == pytest.approx(spec.linkage.drive_upper_rad, abs=1e-9), path
            if name != spec.drive_joint_name:
                mimic = joint.find("mimic")
                assert mimic is not None, (path, name)
                assert mimic.get("joint") == spec.drive_joint_name, (path, name)
                assert float(mimic.get("multiplier", "1")) == 1.0, (path, name)
                assert float(mimic.get("offset", "0")) == 0.0, (path, name)


def test_lite6_link6_three_hull_enclosure_and_wrist_clearance() -> None:
    """Lite6 link6's three convex hulls enclose link6.stl tightly without hiding near-limit link4/link6 contact."""
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("pinocchio")
    pytest.importorskip("coal")
    from ufactory.safety.adapters.pinocchio import PinocchioCollisionBackend

    meshes_dir = _URDF_ROOT / "robots" / "lite6" / "meshes" / "lite6"
    visual = trimesh.load(meshes_dir / "visual" / "link6.stl", force="mesh")
    hulls = [
        trimesh.load(meshes_dir / "collision" / f"lite6_link6_col_{idx:02d}.stl", force="mesh") for idx in range(3)
    ]

    for hull in hulls:
        _, dists, _ = trimesh.proximity.closest_point(visual, hull.vertices)
        assert float(np.max(dists)) < 5e-5

    pts = np.asarray(visual.vertices, dtype=np.float64)
    min_outside = np.full(len(pts), np.inf, dtype=np.float64)
    for hull in hulls:
        normals = np.asarray(hull.face_normals, dtype=np.float64)
        v0 = np.asarray(hull.vertices[hull.faces[:, 0]], dtype=np.float64)
        outside = np.max(pts @ normals.T - np.sum(v0 * normals, axis=1)[None, :], axis=1)
        min_outside = np.minimum(min_outside, outside)
    assert float(np.max(min_outside)) < 1e-6

    backend = PinocchioCollisionBackend(
        _URDF_ROOT / "robots" / "lite6" / "lite6.urdf",
        joint_names=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
        ee_link="link6",
        tool_root_link="link6",
    )

    def link4_link6_min_mm(q5: float, q6: float) -> float:
        q = np.array([0.0, 0.0, 0.0, 0.0, q5, q6], dtype=np.float64)
        return min(
            res.min_distance_m * 1000.0
            for res in backend.check_all(q, stage="start")
            if {res.link_a, res.link_b} == {"link4", "link6"}
        )

    # Normal working range stays comfortably above the 5 mm preflight margin across all q6.
    normal_min_mm = min(
        link4_link6_min_mm(q5, q6)
        for q5 in (-1.8, -1.0, 0.0, 0.2742, 1.0, 1.8)
        for q6 in np.linspace(-np.pi, np.pi, 13)
    )
    assert normal_min_mm > 5.8

    # Near the J5 mechanical limit, real link4/link6 contact is still detected.
    limit_min_mm = min(link4_link6_min_mm(2.1642, q6) for q6 in np.linspace(-np.pi, np.pi, 13))
    assert limit_min_mm < 0.5
