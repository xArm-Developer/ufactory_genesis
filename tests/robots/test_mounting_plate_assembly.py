"""Mounting plate asset, assembly, kinematics, safety, and Genesis simulation tests."""

from __future__ import annotations

import itertools
import json
import math
import os
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from ufactory.robots.assembly import (
    AssemblyError,
    AssemblyRef,
    assembly_ref,
    compose_robot_urdf,
    compose_standalone_mounting_plate_urdf,
    get_mounting_plate,
    mounting_plate_for,
    mounting_plates,
    parse_assembly_ref,
    resolve_robot_urdf,
    validate_robot_urdf,
)
from ufactory.robots.paths import (
    lite6_mounting_plate_urdf,
    robot_urdf,
    robot_visual_glb_urdf,
    uf850_mounting_plate_urdf,
    xarm_mounting_plate_urdf,
)
from ufactory.robots.registry import ROBOT_PROFILES, get_robot_profile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
URDF_ROOT = PROJECT_ROOT / "assets" / "urdf"

PLATES: dict[str, dict[str, object]] = {
    "xarm_mounting_plate": {
        "robots": ("xarm5_1305", "xarm6_1305", "xarm7_1305"),
        "thickness_mm": 8.0,
        "extent_mm": (136.0, 185.0),
        "x_bounds_mm": (-63.0, 73.0),
        "y_bounds_mm": (-92.5, 92.5),
        "collision_count": 17,
        "mass_kg": 0.3554330342,
        "mount_xyz": (0.0, 0.0, 0.0),
        "solid_xy_mm": (0.0, 75.0),
    },
    "uf850_mounting_plate": {
        "robots": ("uf850",),
        "thickness_mm": 10.0,
        "extent_mm": (190.0, 260.0),
        "x_bounds_mm": (-95.0, 95.0),
        "y_bounds_mm": (-130.0, 130.0),
        "collision_count": 1,
        "mass_kg": 1.056,
        "mount_xyz": (0.0, 0.0, 0.0),
        "solid_xy_mm": (0.0, 105.0),
    },
    "lite6_mounting_plate": {
        "robots": ("lite6",),
        "thickness_mm": 5.0,
        "extent_mm": (130.0, 190.0),
        "x_bounds_mm": (-65.0, 65.0),
        "y_bounds_mm": (-95.0, 95.0),
        "collision_count": 1,
        "mass_kg": 0.2431144384,
        "mount_xyz": (-0.003, 0.0, 0.0),
        "solid_xy_mm": (0.0, 70.0),
    },
}

XARM_ARMS = ("xarm5_1305", "xarm6_1305", "xarm7_1305", "uf850")
ARMS = (*XARM_ARMS, "lite6")
ARM_END_EFFECTORS: dict[str, tuple[str | None, ...]] = {
    **{
        arm: (None, "gripper_g2", "bio_gripper_g2", "robotiq_2f_85", "robotiq_2f_140", "vacuum_gripper")
        for arm in XARM_ARMS
    },
    "lite6": (None, "lite6_gripper", "lite6_vacuum_gripper"),
}
STANDALONE_LOADERS = {
    "xarm_mounting_plate": xarm_mounting_plate_urdf,
    "uf850_mounting_plate": uf850_mounting_plate_urdf,
    "lite6_mounting_plate": lite6_mounting_plate_urdf,
}


# --- catalog and reference parsing --------------------------------------------------------------------------------


def test_catalog_partitions_all_five_robots() -> None:
    plates = mounting_plates()
    assert tuple(plates) == ("xarm_mounting_plate", "uf850_mounting_plate", "lite6_mounting_plate")
    covered: list[str] = []
    for name, cfg in PLATES.items():
        plate = plates[name]
        assert plate.robots == cfg["robots"]
        assert plate.thickness_m == pytest.approx(float(cfg["thickness_mm"]) * 0.001)
        assert plate.mount[0] == pytest.approx(cfg["mount_xyz"])
        assert plate.mount[1] == (0.0, 0.0, 0.0)
        assert get_mounting_plate(name) == plate
        for arm in plate.robots:
            assert mounting_plate_for(arm) == plate
            profile = get_robot_profile(arm)
            assert profile.supports_mounting_plate
            assert profile.mounting_plate == name
            covered.append(arm)
    assert sorted(covered) == sorted(ROBOT_PROFILES)
    with pytest.raises(AssemblyError, match="unknown mounting plate"):
        get_mounting_plate("missing_plate")
    with pytest.raises(KeyError):
        mounting_plate_for("unknown_arm")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("assembly:bare+mounting_plate", AssemblyRef(None, "glb", None, True)),
        ("assembly:bare+mounting_plate:stl", AssemblyRef(None, "stl", None, True)),
        ("assembly:gripper_g2+mounting_plate", AssemblyRef("gripper_g2", "glb", None, True)),
        ("assembly:gripper_g2+d435+mounting_plate", AssemblyRef("gripper_g2", "glb", "d435", True)),
        ("assembly:bare+camera_stand+mounting_plate:stl", AssemblyRef(None, "stl", "camera_stand", True)),
    ],
)
def test_parse_mounting_plate_refs_round_trip(value: str, expected: AssemblyRef) -> None:
    assert parse_assembly_ref(value) == expected
    assert str(expected) == value


def test_parse_mounting_plate_refs_canonicalizes_and_rejects_malformed() -> None:
    parsed = parse_assembly_ref("assembly:gripper_g2+mounting_plate+d435:stl")
    assert parsed == AssemblyRef("gripper_g2", "stl", "d435", True)
    assert str(parsed) == "assembly:gripper_g2+d435+mounting_plate:stl"

    for bad in (
        "assembly:mounting_plate",
        "assembly:+mounting_plate",
        "assembly:bare+mounting_plate+mounting_plate",
        "assembly:gripper_g2+d435+d405+mounting_plate",
        "assembly:bare+mounting_plate:obj",
    ):
        with pytest.raises(AssemblyError):
            parse_assembly_ref(bad)

    assert assembly_ref(None, mounting_plate=True) == "assembly:bare+mounting_plate"


def test_path_helpers_and_asset_store_support_mounting_plates(tmp_path: Path) -> None:
    assert robot_urdf("xarm6", "assembly:bare+mounting_plate") == compose_robot_urdf("xarm6", None, mounting_plate=True)
    assert robot_urdf("lite6", "assembly:lite6_gripper+d435+mounting_plate:stl") == compose_robot_urdf(
        "lite6", "lite6_gripper", accessory="d435", mounting_plate=True, visual="stl"
    )
    assert robot_visual_glb_urdf("uf850", mounting_plate=True) == compose_robot_urdf("uf850", None, mounting_plate=True)
    assert robot_visual_glb_urdf(
        "xarm6", with_gripper_g2=True, accessory="d435", mounting_plate=True
    ) == compose_robot_urdf("xarm6", "gripper_g2", accessory="d435", mounting_plate=True)

    from ufactory.config import RepositoryAssetStore

    store = RepositoryAssetStore(PROJECT_ROOT)
    validate_robot_urdf(
        "xarm6_1305", "assembly:gripper_g2+mounting_plate", assets_dir="assets/urdf/robots/xarm6", store=store
    )
    resolved = resolve_robot_urdf("lite6", "assembly:bare+mounting_plate", store=store)
    assert resolved == Path(compose_robot_urdf("lite6", None, mounting_plate=True))


# --- deterministic composition ------------------------------------------------------------------------------------


def test_every_mounting_plate_combination_composes_deterministically(tmp_path: Path) -> None:
    count = 0
    for arm in ARMS:
        plate = mounting_plate_for(arm)
        combos: list[tuple[str | None, str | None]] = [(ee, None) for ee in ARM_END_EFFECTORS[arm]]
        default_ee = "lite6_gripper" if arm == "lite6" else "gripper_g2"
        combos += [(None, "camera_stand"), (default_ee, "d435")]
        for (ee, acc), visual in itertools.product(combos, ("glb", "stl")):
            path = Path(compose_robot_urdf(arm, ee, accessory=acc, mounting_plate=True, visual=visual))
            assert path == Path(compose_robot_urdf(arm, ee, accessory=acc, mounting_plate=True, visual=visual))
            root = ET.parse(path).getroot()
            names = [e.get("name") for e in root if e.tag in ("link", "joint")]
            assert len(names) == len(set(names))
            assert sum(e.tag == "link" for e in root) == sum(e.tag == "joint" for e in root) + 1
            fix = root.find(f"./joint[@name='{plate.mount_joint}']")
            assert fix is not None and fix.get("type") == "fixed"
            assert fix.find("parent").get("link") == "link_base"
            assert fix.find("child").get("link") == plate.base_link
            for mesh in root.iter("mesh"):
                fn = mesh.get("filename") or ""
                assert not os.path.isabs(fn)
                assert (path.parent / fn).resolve().is_file(), fn
            count += 1
    assert count == 74

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        elsewhere = Path(compose_robot_urdf("xarm6", "gripper_g2", accessory="d435", mounting_plate=True))
    finally:
        os.chdir(cwd)
    assert (
        elsewhere.read_bytes()
        == Path(compose_robot_urdf("xarm6", "gripper_g2", accessory="d435", mounting_plate=True)).read_bytes()
    )


def test_stl_mode_and_standalone_mounting_plates(tmp_path: Path) -> None:
    root = ET.parse(compose_robot_urdf("xarm6", None, mounting_plate=True, visual="stl")).getroot()
    plate_link = root.find("./link[@name='xarm_mounting_plate']")
    assert plate_link is not None
    visuals = plate_link.findall("visual")
    assert len(visuals) == 17
    assert all((v.find("material").get("name") == "PlateAluminum") for v in visuals)  # type: ignore[union-attr]
    assert all(
        (v.find("geometry/mesh").get("filename") or "").endswith(f"xarm_mounting_plate_col_{i:02d}.stl")
        for i, v in enumerate(visuals)
    )

    for name, loader in STANDALONE_LOADERS.items():
        path = Path(loader())
        assert path.is_file()
        assert path == Path(compose_standalone_mounting_plate_urdf(name, visual="glb"))
        st_path = Path(compose_standalone_mounting_plate_urdf(name, visual="stl", out_dir=tmp_path))
        st_root = ET.parse(st_path).getroot()
        fix = st_root.find("./joint[@name='mounting_plate_fix']")
        assert fix is not None and fix.find("parent").get("link") == "world"
        assert fix.find("child").get("link") == name
        assert fix.find("origin").get("xyz") == "0 0 0"


# --- geometry, openings, holes, and inertials ---------------------------------------------------------------------


def _glb_json(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    assert raw[:4] == b"glTF"
    (length,) = struct.unpack("<I", raw[12:16])
    return json.loads(raw[20 : 20 + length])


def _point_on_top_surface(mesh, xy_mm: tuple[float, float]) -> bool:
    verts_mm = np.asarray(mesh.vertices, dtype=np.float64) * 1000.0
    tris = verts_mm[np.asarray(mesh.faces, dtype=np.int64)]
    top = tris[np.all(np.abs(tris[:, :, 2]) < 0.02, axis=1), :, :2]
    a, b, c = top[:, 0], top[:, 1], top[:, 2]
    v0, v1, v2 = c - a, b - a, np.asarray(xy_mm, dtype=np.float64) - a
    denom = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    valid = np.abs(denom) > 1e-12
    u = np.divide(v2[:, 0] * v1[:, 1] - v2[:, 1] * v1[:, 0], denom, out=np.full_like(denom, -1.0), where=valid)
    v = np.divide(v0[:, 0] * v2[:, 1] - v0[:, 1] * v2[:, 0], denom, out=np.full_like(denom, -1.0), where=valid)
    return bool(np.any(valid & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1.0 + 1e-6)))


def _expected_openings_xy_mm(plate: str) -> list[tuple[float, float]]:
    if plate == "xarm_mounting_plate":
        pts = [(0.0, 0.0)]
        pts += [(55.0 * math.cos(math.radians(a)), 55.0 * math.sin(math.radians(a))) for a in range(0, 360, 30)]
        pts += [(x, y) for x in (-37.5, 37.5) for y in (-83.0, 83.0)]
        pts += [(53.601, -58.945), (53.601, 58.945), (-51.253, -58.863), (-51.253, 58.743)]
        return pts
    if plate == "uf850_mounting_plate":
        pts = [(0.0, 0.0)]
        pts += [(x, y) for x in (-60.104076, 60.104076) for y in (-60.104076, 60.104076)]
        pts += [(x, y) for x in (-70.0, 70.0) for y in (-115.0, 115.0)]
        pts += [(x, y) for x in (-80.0, 80.0) for y in (-85.0, 85.0)]
        return pts
    return (
        [(0.0, 0.0)]
        + [(x, y) for x in (-57.0, 57.0) for y in (-57.0, 57.0)]
        + [(x, y) for x in (-50.0, 50.0) for y in (-80.0, 80.0)]
    )


def _is_convex(mesh, tol_m: float = 1e-6) -> bool:
    """Every vertex lies on or behind every face plane (numpy only; trimesh ``is_convex`` needs scipy)."""
    vertices, faces, normals = (np.asarray(a) for a in (mesh.vertices, mesh.faces, mesh.face_normals))
    offsets = np.einsum("ij,ij->i", normals, vertices[faces[:, 0]])
    return bool((vertices @ normals.T - offsets).max() <= tol_m)


def test_mounting_plate_urdf_inertials() -> None:
    """Validate plate URDF masses and inertia tensors without optional mesh dependencies."""
    uf850_expected_inertia = {
        "ixx": 0.007494367693,
        "ixy": 2.827268798e-09,
        "ixz": 8.907087421e-11,
        "iyy": 0.003785981885,
        "iyz": -2.155216145e-11,
        "izz": 0.0112627498,
    }
    for plate, cfg in PLATES.items():
        root = ET.parse(URDF_ROOT / "mounts" / plate / f"{plate}.urdf").getroot()
        inertial = root.find(f"./link[@name='{plate}']/inertial")
        assert inertial is not None
        mass = float(inertial.find("mass").attrib["value"])  # type: ignore[union-attr]
        assert mass == pytest.approx(float(cfg["mass_kg"]), rel=0.0, abs=1e-10)
        xyz = tuple(float(v) for v in inertial.find("origin").attrib["xyz"].split())  # type: ignore[union-attr]
        i = {k: float(v) for k, v in inertial.find("inertia").attrib.items()}  # type: ignore[union-attr]
        tensor = np.array(
            [[i["ixx"], i["ixy"], i["ixz"]], [i["ixy"], i["iyy"], i["iyz"]], [i["ixz"], i["iyz"], i["izz"]]]
        )
        eig = np.linalg.eigvalsh(tensor)
        assert eig.min() > 0.0 and eig[0] + eig[1] > eig[2]
        if plate == "uf850_mounting_plate":
            assert mass == pytest.approx(1.056, rel=0.0, abs=1e-12)
            assert xyz == pytest.approx((-5.112679085e-08, -1.296273053e-07, -0.005000003168), rel=0.0, abs=1e-12)
            for key, expected_val in uf850_expected_inertia.items():
                assert i[key] == pytest.approx(expected_val, rel=0.0, abs=1e-12), key


def test_mounting_plate_materials_match_arm_flanges() -> None:
    """Validate plate GLB PBR and URDF materials match the arm flange metal without optional mesh deps."""
    flange_glbs = (
        "robots/xarm6/meshes/xarm6_1305/visual_glb/link6.glb",
        "robots/uf850/meshes/uf850/visual_glb/link6.glb",
        "robots/lite6/meshes/lite6/visual_glb/link6.glb",
    )
    for plate in PLATES:
        plate_dir = URDF_ROOT / "mounts" / plate
        meta = _glb_json(plate_dir / "meshes" / "visual" / f"{plate}.glb")
        assert meta["extensionsRequired"] == ["KHR_draco_mesh_compression"]
        mats = meta["materials"]
        assert isinstance(mats, list) and len(mats) == 1
        assert mats[0]["name"] == "plate_anodized_aluminum"
        pbr = mats[0]["pbrMetallicRoughness"]
        np.testing.assert_allclose(
            pbr["baseColorFactor"],
            [0.9529411764705883, 0.9529411764705883, 0.9490196078431372, 1.0],
            atol=1e-5,
        )
        assert pbr["metallicFactor"] == pytest.approx(0.8, abs=1e-5)
        assert pbr["roughnessFactor"] == pytest.approx(0.640130877494812, abs=1e-5)

        for arm_glb in flange_glbs:
            arm_meta = _glb_json(URDF_ROOT / arm_glb)
            arm_metals = [
                m["pbrMetallicRoughness"]
                for m in arm_meta["materials"]  # type: ignore[index]
                if m.get("pbrMetallicRoughness", {}).get("metallicFactor", 0.0) > 0.5
            ]
            assert len(arm_metals) == 1, arm_glb
            np.testing.assert_allclose(pbr["baseColorFactor"], arm_metals[0]["baseColorFactor"], atol=1e-5)
            assert pbr["metallicFactor"] == pytest.approx(arm_metals[0]["metallicFactor"], abs=1e-5)
            assert pbr["roughnessFactor"] == pytest.approx(arm_metals[0]["roughnessFactor"], abs=0.01)

        root = ET.parse(plate_dir / f"{plate}.urdf").getroot()
        color = root.find("./material[@name='PlateAluminum']/color")
        assert color is not None
        rgba = [float(v) for v in color.attrib["rgba"].split()]
        np.testing.assert_allclose(rgba, [0.952941, 0.952941, 0.94902, 1.0], atol=1e-5)


@pytest.mark.parametrize("plate", tuple(PLATES))
def test_plate_assets_geometry_openings_and_inertials(plate: str) -> None:
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("DracoPy")
    pytest.importorskip("pygltflib")

    cfg = PLATES[plate]
    plate_dir = URDF_ROOT / "mounts" / plate
    glb = plate_dir / "meshes" / "visual" / f"{plate}.glb"
    assert glb.stat().st_size < 1_000_000
    meta = _glb_json(glb)
    assert meta["extensionsRequired"] == ["KHR_draco_mesh_compression"]
    materials = [m["name"] for m in meta["materials"]]  # type: ignore[attr-defined,index]
    assert materials == ["plate_anodized_aluminum"]

    mesh = trimesh.load(glb, force="mesh")
    bounds_mm = np.asarray(mesh.bounds) * 1000.0
    np.testing.assert_allclose(mesh.extents * 1000.0, (*cfg["extent_mm"], cfg["thickness_mm"]), atol=0.05)  # type: ignore[misc]
    assert bounds_mm[0, 0] == pytest.approx(cfg["x_bounds_mm"][0], abs=0.05)  # type: ignore[index]
    assert bounds_mm[1, 0] == pytest.approx(cfg["x_bounds_mm"][1], abs=0.05)  # type: ignore[index]
    assert bounds_mm[0, 1] == pytest.approx(cfg["y_bounds_mm"][0], abs=0.05)  # type: ignore[index]
    assert bounds_mm[1, 1] == pytest.approx(cfg["y_bounds_mm"][1], abs=0.05)  # type: ignore[index]
    assert bounds_mm[1, 2] == pytest.approx(0.0, abs=0.02)
    assert bounds_mm[0, 2] == pytest.approx(-float(cfg["thickness_mm"]), abs=0.02)
    assert bounds_mm[1, 1] - bounds_mm[0, 1] > bounds_mm[1, 0] - bounds_mm[0, 0]

    for xy in _expected_openings_xy_mm(plate):
        assert not _point_on_top_surface(mesh, xy), (plate, xy)
    assert _point_on_top_surface(mesh, cfg["solid_xy_mm"])  # type: ignore[arg-type]

    cols = sorted((plate_dir / "meshes" / "collision").glob("*.stl"))
    assert len(cols) == cfg["collision_count"]
    assert [c.name for c in cols] == [f"{plate}_col_{i:02d}.stl" for i in range(int(cfg["collision_count"]))]  # type: ignore[arg-type]
    col_meshes = [trimesh.load(c, force="mesh") for c in cols]
    for cm in col_meshes:
        assert cm.is_watertight and _is_convex(cm)
    col_pts_mm = np.concatenate([np.asarray(cm.vertices) * 1000.0 for cm in col_meshes], axis=0)
    assert float(col_pts_mm[:, 2].max()) == pytest.approx(0.0, abs=0.02)
    assert float(col_pts_mm[:, 2].min()) == pytest.approx(-float(cfg["thickness_mm"]), abs=0.02)
    np.testing.assert_allclose(np.ptp(col_pts_mm, axis=0), (*cfg["extent_mm"], cfg["thickness_mm"]), atol=0.05)  # type: ignore[misc]

    root = ET.parse(plate_dir / f"{plate}.urdf").getroot()
    inertial = root.find(f"./link[@name='{plate}']/inertial")
    assert inertial is not None
    mass = float(inertial.find("mass").attrib["value"])  # type: ignore[union-attr]
    assert mass == pytest.approx(float(cfg["mass_kg"]), abs=1e-4)
    i = {k: float(v) for k, v in inertial.find("inertia").attrib.items()}  # type: ignore[union-attr]
    tensor = np.array([[i["ixx"], i["ixy"], i["ixz"]], [i["ixy"], i["iyy"], i["iyz"]], [i["ixz"], i["iyz"], i["izz"]]])
    eig = np.linalg.eigvalsh(tensor)
    assert eig.min() > 0.0 and eig[0] + eig[1] > eig[2]
    assert (plate_dir / "README.md").is_file()


def _base_mesh_vertices_mm(arm: str, *, glb: bool) -> np.ndarray:
    trimesh = pytest.importorskip("trimesh")
    profile = get_robot_profile(arm)
    urdf = profile.assets_dir / (profile.visual_glb_urdf if glb else profile.default_urdf)
    root = ET.parse(urdf).getroot()
    fn = root.find("./link[@name='link_base']/visual/geometry/mesh").attrib["filename"]  # type: ignore[union-attr]
    mesh = trimesh.load((urdf.parent / fn).resolve(), force="mesh")
    return np.asarray(mesh.vertices, dtype=np.float64) * 1000.0


def _nominal_holes_mm(arm: str) -> tuple[np.ndarray, np.ndarray, float]:
    if arm.startswith("xarm"):
        centers = np.array(
            [
                [55.0, 0.0],
                [27.5, 55.0 * math.sin(math.radians(60.0))],
                [-27.5, 55.0 * math.sin(math.radians(60.0))],
                [27.5, -55.0 * math.sin(math.radians(60.0))],
                [-27.5, -55.0 * math.sin(math.radians(60.0))],
            ]
        )
        return centers, np.zeros(2), 2.75
    if arm == "uf850":
        return (
            np.array([[x, y] for x in (-60.104076, 60.104076) for y in (-60.104076, 60.104076)]),
            np.zeros(2),
            4.25,
        )
    return np.array([[x, y] for x in (-57.0, 57.0) for y in (-57.0, 57.0)]), np.array([-3.0, 0.0]), 2.6


def _fit_circle_center(points: np.ndarray) -> np.ndarray:
    """Algebraic (Kasa) least-squares circle fit: centre of x² + y² + D·x + E·y + F = 0."""
    a = np.column_stack([points, np.ones(len(points))])
    (d, e, _), *_ = np.linalg.lstsq(a, (points**2).sum(axis=1), rcond=None)
    return np.array([d, e]) / 2.0


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("glb", [False, True])
def test_robot_mounting_holes_and_bottom_plane_align_to_plate(arm: str, glb: bool) -> None:
    """Both link_base.stl and link_base.glb mounting holes align to the plate holes at z = 0."""
    plate = mounting_plate_for(arm)
    plate_xyz_mm = np.asarray(plate.mount[0], dtype=np.float64) * 1000.0
    vertices = _base_mesh_vertices_mm(arm, glb=glb)
    z_min = float(vertices[:, 2].min())
    assert z_min == pytest.approx(0.0, abs=0.02)

    bottom = vertices[np.abs(vertices[:, 2] - z_min) < 0.05, :2]
    centers, shift, radius = _nominal_holes_mm(arm)
    tol_mm = 0.5 if (glb and arm.startswith("xarm")) else 0.03
    for nominal in centers:
        guess = nominal + shift
        candidates = bottom[np.abs(np.linalg.norm(bottom - guess, axis=1) - radius) < 1.2]
        assert len(candidates) >= 8
        in_plate_mm = _fit_circle_center(candidates) - plate_xyz_mm[:2]
        np.testing.assert_allclose(in_plate_mm, nominal, atol=tol_mm)


# --- kinematics, dynamics, and safety collision checks ------------------------------------------------------------


@pytest.mark.parametrize("arm", ARMS)
def test_kinematics_dynamics_and_safety_collision_with_mounting_plate(arm: str) -> None:
    pin = pytest.importorskip("pinocchio")
    coal = pytest.importorskip("coal")
    from ufactory.safety.adapters.pinocchio import PinocchioCollisionBackend

    profile = get_robot_profile(arm)
    plate = mounting_plate_for(arm)
    bare_urdf = str(profile.assets_dir / profile.default_urdf)
    plated_urdf = compose_robot_urdf(arm, None, mounting_plate=True, visual="stl")

    bm = pin.buildModelFromUrdf(bare_urdf, mimic=True)
    bd = bm.createData()
    pm = pin.buildModelFromUrdf(plated_urdf, mimic=True)
    pd = pm.createData()
    assert (pm.nq, pm.nv) == (bm.nq, bm.nv)
    assert sum(i.mass for i in pm.inertias) - sum(i.mass for i in bm.inertias) == pytest.approx(
        float(PLATES[plate.name]["mass_kg"]), abs=1e-4
    )

    rng = np.random.default_rng(17)
    q = np.clip(rng.uniform(-1.0, 1.0, bm.nq), bm.lowerPositionLimit, bm.upperPositionLimit)
    pin.framesForwardKinematics(bm, bd, q)
    pin.framesForwardKinematics(pm, pd, q)
    for frame_name in ("link_base", profile.ee_link):
        b_pose = bd.oMf[bm.getFrameId(frame_name)]
        p_pose = pd.oMf[pm.getFrameId(frame_name)]
        np.testing.assert_allclose(p_pose.translation, b_pose.translation, atol=1e-12)
        np.testing.assert_allclose(p_pose.rotation, b_pose.rotation, atol=1e-12)
    np.testing.assert_allclose(pin.crba(pm, pd, q), pin.crba(bm, bd, q), atol=1e-12)
    np.testing.assert_allclose(
        pin.computeGeneralizedGravity(pm, pd, q), pin.computeGeneralizedGravity(bm, bd, q), atol=1e-12
    )

    geom = pin.buildGeomFromUrdf(
        pm, plated_urdf, pin.GeometryType.COLLISION, package_dirs=[str(Path(plated_urdf).parent)]
    )
    gdata = geom.createData()
    pin.updateGeometryPlacements(pm, pd, geom, gdata, pin.neutral(pm))
    owners = [pm.frames[g.parentFrame].name for g in geom.geometryObjects]
    plate_idx = [i for i, n in enumerate(owners) if n == plate.base_link]
    base_idx = [i for i, n in enumerate(owners) if n == "link_base"]
    assert plate_idx and base_idx
    req = coal.DistanceRequest()
    nearest = min(
        float(
            coal.distance(
                geom.geometryObjects[i].geometry,
                gdata.oMg[i],
                geom.geometryObjects[j].geometry,
                gdata.oMg[j],
                req,
                coal.DistanceResult(),
            )
        )
        for i in plate_idx
        for j in base_idx
    )
    assert -1e-4 <= nearest <= 1e-4

    joint_names = tuple(f"joint{i}" for i in range(1, profile.dof + 1))
    backend = PinocchioCollisionBackend(plated_urdf, joint_names=joint_names, ee_link=profile.ee_link)
    pair_links = {
        frozenset((backend._link_name(p.first), backend._link_name(p.second)))
        for p in backend.geometry_model.collisionPairs
    }
    assert frozenset(("link_base", plate.base_link)) not in pair_links
    assert frozenset(("link1", plate.base_link)) not in pair_links
    assert frozenset(("link2", plate.base_link)) in pair_links

    for q_test in (np.zeros(profile.dof), q * 0.25):
        res = backend.check(q_test, stage="unit")
        assert not res.colliding and res.min_distance_m > 0.005


def test_full_stack_camera_and_gripper_safety_with_mounting_plate() -> None:
    pytest.importorskip("pinocchio")
    pytest.importorskip("coal")
    from ufactory.config import load_runtime_config
    from ufactory.safety.adapters.pinocchio import PinocchioCollisionBackend

    x_cfg = load_runtime_config("xarm6")
    assert x_cfg.gripper is not None
    xarm_urdf = compose_robot_urdf("xarm6", "gripper_g2", accessory="d435", mounting_plate=True)
    x_backend = PinocchioCollisionBackend(
        xarm_urdf,
        joint_names=x_cfg.robot.joint_names,
        ee_link=x_cfg.robot.ee_link,
        passive_joint_positions={x_cfg.gripper.drive_joint: x_cfg.gripper.open_drive},
        tool_root_link=x_cfg.robot.ee_link,
    )
    q_ready = np.array([0.0, -0.63088336, -0.837867545, 0.0, 1.468750905, 0.0])
    x_res = x_backend.check(q_ready, stage="ready")
    assert not x_res.colliding and x_res.min_distance_m > 0.001

    l_cfg = load_runtime_config("lite6")
    assert l_cfg.gripper is not None
    lite_urdf = compose_robot_urdf("lite6", "lite6_gripper", accessory="d435", mounting_plate=True)
    l_backend = PinocchioCollisionBackend(
        lite_urdf,
        joint_names=l_cfg.robot.joint_names,
        ee_link=l_cfg.robot.ee_link,
        passive_joint_positions={l_cfg.gripper.drive_joint: l_cfg.gripper.open_drive},
        tool_root_link=l_cfg.robot.ee_link,
    )
    l_res = l_backend.check(np.array([0.0, -0.6, 0.0, 0.0, 0.6, 0.0]), stage="ready")
    assert not l_res.colliding and l_res.min_distance_m > 0.005


# --- Genesis GPU simulation & drop-sphere contact tests -----------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("arm", ARMS)
def test_genesis_builds_arm_with_mounting_plate(arm: str) -> None:
    gs = pytest.importorskip("genesis")
    pin = pytest.importorskip("pinocchio")
    from ufactory.config import load_runtime_config
    from ufactory.simulation import GenesisRuntimeManager, make_rigid_options

    plate = mounting_plate_for(arm)
    config = load_runtime_config(arm)
    ee = "lite6_gripper" if arm == "lite6" else ("gripper_g2" if arm == "xarm6_1305" else None)
    acc = "d435" if arm == "xarm6_1305" else None
    urdf = compose_robot_urdf(arm, ee, accessory=acc, mounting_plate=True)

    with GenesisRuntimeManager(config.simulation):
        scene = gs.Scene(
            show_viewer=False,
            sim_options=gs.options.SimOptions(dt=0.01, substeps=2),
            rigid_options=make_rigid_options(gs),
        )
        scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", pos=(0.0, 0.0, -plate.thickness_m), fixed=True))
        robot = scene.add_entity(gs.morphs.URDF(file=urdf, fixed=True))
        scene.build()
        pin_model = pin.buildModelFromUrdf(urdf, mimic=True)
        gs_mass = sum(float(link.get_mass()) for link in robot.links)
        assert gs_mass == pytest.approx(sum(i.mass for i in pin_model.inertias), abs=1e-4)
        for _ in range(10):
            scene.step()
        qpos = robot.get_dofs_position().cpu().numpy()
        assert np.all(np.isfinite(qpos))


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("plate", "xy_mm"),
    [
        ("xarm_mounting_plate", (0.0, 75.0)),
        ("xarm_mounting_plate", (0.0, 0.0)),
        ("xarm_mounting_plate", (55.0, 0.0)),
        ("xarm_mounting_plate", (67.0, 0.0)),
        ("xarm_mounting_plate", (63.0, 0.0)),
        ("uf850_mounting_plate", (0.0, 105.0)),
        ("uf850_mounting_plate", (0.0, 0.0)),
        ("uf850_mounting_plate", (60.104076, 60.104076)),
        ("lite6_mounting_plate", (0.0, 70.0)),
        ("lite6_mounting_plate", (0.0, 0.0)),
        ("lite6_mounting_plate", (57.0, 57.0)),
    ],
)
def test_genesis_drop_sphere_lands_on_plate_and_filled_holes(plate: str, xy_mm: tuple[float, float]) -> None:
    gs = pytest.importorskip("genesis")
    from ufactory.simulation import GenesisRuntimeManager, make_rigid_options
    from ufactory.config import load_runtime_config

    config = load_runtime_config("xarm6")
    radius = 0.004 if xy_mm in ((0.0, 0.0), (0.0, 75.0), (0.0, 105.0), (0.0, 70.0)) else 0.002
    x_m, y_m = xy_mm[0] * 0.001, xy_mm[1] * 0.001
    plate_urdf = compose_standalone_mounting_plate_urdf(plate, visual="glb")

    with GenesisRuntimeManager(config.simulation):
        scene = gs.Scene(
            show_viewer=False,
            sim_options=gs.options.SimOptions(dt=0.001, substeps=4),
            rigid_options=make_rigid_options(gs),
        )
        scene.add_entity(gs.morphs.URDF(file=plate_urdf, fixed=True))
        ball = scene.add_entity(
            gs.morphs.Sphere(radius=radius, pos=(x_m, y_m, radius + 0.015)),
        )
        scene.build()
        for _ in range(300):
            scene.step()
        pos = ball.get_pos().cpu().numpy().reshape(-1)
        vel = ball.get_vel().cpu().numpy().reshape(-1)
        assert np.all(np.isfinite(pos)) and np.all(np.isfinite(vel))
        assert float(pos[2]) == pytest.approx(radius, abs=0.0005)
        assert abs(float(vel[2])) < 0.002
        max_speed = 0.05 if plate == "xarm_mounting_plate" and xy_mm[0] >= 63.0 else 0.01
        assert float(np.linalg.norm(vel)) < max_speed
