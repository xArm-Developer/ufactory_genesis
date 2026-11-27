"""Robotiq flange adapter (robotiq_mounting) asset, assembly, clearance, and Genesis tests."""

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

from ufactory.robots import assembly
from ufactory.robots.assembly import (
    AssemblyError,
    adapters,
    compose_robot_urdf,
    compose_standalone_adapter_urdf,
    end_effectors,
    get_adapter,
    supported_accessories,
)
from ufactory.robots.paths import robot_visual_glb_urdf, robotiq_mounting_urdf
from ufactory.robots.registry import get_robot_profile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
URDF_ROOT = PROJECT_ROOT / "assets" / "urdf"
ADAPTER_DIR = URDF_ROOT / "mounts" / "robotiq_mounting"

XARM_ARMS = ("xarm5_1305", "xarm6_1305", "xarm7_1305", "uf850")
ROBOTIQ_EES = ("robotiq_2f_85", "robotiq_2f_140")
CAMERAS = ("d405", "d435", "gemini_305", "zed_x_nano")
ADAPTER_LIFT_M = 0.011
STAND_LIFT_M = 0.003
ADAPTER_MASS_KG = 0.118
ROBOTIQ_BASE_TCP_Z_M = {"robotiq_2f_85": 0.162253, "robotiq_2f_140": 0.234466}
ROBOTIQ_TOOL_TIP_Z_M = {"robotiq_2f_85": 0.1093, "robotiq_2f_140": 0.1403}


# --- catalog and schema -------------------------------------------------------------------------------------------


def test_catalog_declares_robotiq_mounting_for_xarm_and_uf850_only() -> None:
    all_adapters = adapters()
    assert tuple(all_adapters) == ("robotiq_mounting",)
    ad = get_adapter("robotiq_mounting")
    assert ad == all_adapters["robotiq_mounting"]
    assert ad.flange == "xarm"
    assert ad.robots == XARM_ARMS
    assert "lite6" not in ad.robots
    assert ad.tool_lift_m == pytest.approx(ADAPTER_LIFT_M)
    assert ad.mount == assembly.IDENTITY
    assert ad.base_link == "robotiq_mounting"
    assert ad.mount_joint == "robotiq_mounting_fix"

    for name, ee in end_effectors().items():
        if name in ROBOTIQ_EES:
            assert ee.adapter == "robotiq_mounting"
            assert ee.robots == XARM_ARMS
        else:
            assert ee.adapter is None

    for arm in XARM_ARMS:
        for ee in ROBOTIQ_EES:
            assert supported_accessories(arm, ee) == ("camera_stand", *CAMERAS)
    for ee in ROBOTIQ_EES:
        assert supported_accessories("lite6", ee) == ()
        with pytest.raises(AssemblyError, match="not supported on lite6"):
            compose_robot_urdf("lite6", ee)
        with pytest.raises(AssemblyError, match="not supported on lite6"):
            compose_robot_urdf("lite6", ee, accessory="d435")

    with pytest.raises(AssemblyError, match="unknown adapter"):
        get_adapter("missing_adapter")


def _write_adapter_schema(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "assembly.yaml"
    path.write_text(
        f"schema_version: 5\nflanges:\n  xarm:\n    robots: [xarm6]\n  lite6:\n    robots: [lite6]\n{body}",
        encoding="utf-8",
    )
    return path


def test_adapter_schema_validation_rejects_bad_configs(tmp_path: Path) -> None:
    unknown_adapter = (
        "end_effectors:\n"
        "  rq:\n    source: rq.urdf\n    base_link: b\n    mount_joint: j\n    flange: xarm\n    adapter: nope\n"
    )
    with pytest.raises(AssemblyError, match="unknown adapter"):
        assembly._load_catalog(_write_adapter_schema(tmp_path, unknown_adapter), 1)

    wrong_flange = (
        "adapters:\n"
        "  ad:\n    source: ad.urdf\n    base_link: ad\n    mount_joint: aj\n    flange: xarm\n    tool_lift_m: 0.011\n"
        "end_effectors:\n"
        "  l6:\n    source: l6.urdf\n    base_link: b\n    mount_joint: j\n    flange: lite6\n    adapter: ad\n"
    )
    with pytest.raises(AssemblyError, match="is on flange 'xarm', not 'lite6'"):
        assembly._load_catalog(_write_adapter_schema(tmp_path, wrong_flange), 2)

    neg_lift = (
        "adapters:\n"
        "  ad:\n    source: ad.urdf\n    base_link: ad\n    mount_joint: aj\n    flange: xarm\n    tool_lift_m: -0.001\n"
    )
    with pytest.raises(AssemblyError, match="tool_lift_m must be non-negative"):
        assembly._load_catalog(_write_adapter_schema(tmp_path, neg_lift), 3)

    clash = (
        "adapters:\n"
        "  dup:\n    source: ad.urdf\n    base_link: ad\n    mount_joint: aj\n    flange: xarm\n"
        "end_effectors:\n"
        "  dup:\n    source: e.urdf\n    base_link: e\n    mount_joint: ej\n    flange: xarm\n"
    )
    with pytest.raises(AssemblyError, match="part names must be unique"):
        assembly._load_catalog(_write_adapter_schema(tmp_path, clash), 4)


# --- asset integrity ----------------------------------------------------------------------------------------------


def _glb_json(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    assert raw[:4] == b"glTF"
    (length,) = struct.unpack("<I", raw[12:16])
    data: dict[str, object] = json.loads(raw[20 : 20 + length])
    return data


def _is_convex_halfspace(vertices: np.ndarray, faces: np.ndarray, face_normals: np.ndarray, tol: float = 1e-6) -> bool:
    origins = vertices[faces[:, 0]]
    offsets = np.einsum("ij,ij->i", face_normals, origins)
    max_violation = float(np.max(vertices @ face_normals.T - offsets[None, :]))
    return max_violation <= tol


def test_adapter_assets_budget_material_hulls_and_inertial() -> None:
    glb = ADAPTER_DIR / "meshes" / "visual" / "robotiq_mounting.glb"
    stl = ADAPTER_DIR / "meshes" / "visual" / "robotiq_mounting.stl"
    urdf = ADAPTER_DIR / "robotiq_mounting.urdf"
    assert glb.is_file() and stl.is_file() and urdf.is_file()
    assert (ADAPTER_DIR / "README.md").is_file()
    assert 0 < glb.stat().st_size < 1_000_000

    meta = _glb_json(glb)
    assert meta["extensionsRequired"] == ["KHR_draco_mesh_compression"]
    mats = meta["materials"]
    assert isinstance(mats, list) and len(mats) == 1
    assert mats[0]["name"] == "flange_metal"
    pbr = mats[0]["pbrMetallicRoughness"]
    np.testing.assert_allclose(
        pbr["baseColorFactor"],
        [0.9529411764705883, 0.9529411764705883, 0.9490196078431372, 1.0],
        atol=1e-5,
    )
    assert pbr["metallicFactor"] == pytest.approx(0.8, abs=1e-5)
    assert pbr["roughnessFactor"] == pytest.approx(0.640130877494812, abs=1e-5)
    for arm_glb in (
        "robots/xarm6/meshes/xarm6_1305/visual_glb/link6.glb",
        "robots/uf850/meshes/uf850/visual_glb/link6.glb",
    ):
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

    root = ET.parse(urdf).getroot()
    inertial = root.find("./link[@name='robotiq_mounting']/inertial")
    assert inertial is not None
    mass = float(inertial.find("mass").attrib["value"])  # type: ignore[union-attr]
    assert mass == pytest.approx(ADAPTER_MASS_KG, abs=1e-6)
    com = np.array([float(v) for v in inertial.find("origin").attrib["xyz"].split()])  # type: ignore[union-attr]
    np.testing.assert_allclose(com * 1000.0, [0.0, -0.799645, 4.455899], atol=1e-3)
    i = {k: float(v) for k, v in inertial.find("inertia").attrib.items()}  # type: ignore[union-attr]
    tensor = np.array([[i["ixx"], i["ixy"], i["ixz"]], [i["ixy"], i["iyy"], i["iyz"]], [i["ixz"], i["iyz"], i["izz"]]])
    eig = np.linalg.eigvalsh(tensor)
    assert eig.min() > 0.0 and eig[0] + eig[1] > eig[2]
    expected_inertia = {
        "ixx": 4.427070717e-05,
        "ixy": 0.0,
        "ixz": 0.0,
        "iyy": 4.997953153e-05,
        "iyz": 7.208106518e-08,
        "izz": 9.018403314e-05,
    }
    for key, expected_val in expected_inertia.items():
        assert i[key] == pytest.approx(expected_val, rel=0.0, abs=1e-12), key

    cols = [m.get("filename") for m in root.findall("./link[@name='robotiq_mounting']/collision/geometry/mesh")]
    assert cols == ["meshes/collision/robotiq_mounting_col_00.stl", "meshes/collision/robotiq_mounting_col_01.stl"]

    trimesh = pytest.importorskip("trimesh")
    vis_mesh = trimesh.load(stl, force="mesh")
    assert vis_mesh.is_watertight
    np.testing.assert_allclose(vis_mesh.bounds[0] * 1000.0, [-37.5, -37.5, -5.6], atol=0.05)
    np.testing.assert_allclose(vis_mesh.bounds[1] * 1000.0, [37.5, 37.5, 13.9], atol=0.05)

    body = trimesh.load(ADAPTER_DIR / cols[0], force="mesh")
    boss = trimesh.load(ADAPTER_DIR / cols[1], force="mesh")
    for hull in (body, boss):
        assert hull.is_watertight
        assert _is_convex_halfspace(
            np.asarray(hull.vertices, dtype=np.float64),
            np.asarray(hull.faces, dtype=np.int64),
            np.asarray(hull.face_normals, dtype=np.float64),
        )
    np.testing.assert_allclose(body.bounds[:, 2] * 1000.0, [0.0, 13.9], atol=0.05)
    np.testing.assert_allclose(boss.bounds[:, 2] * 1000.0, [-5.6, 0.0], atol=0.05)
    assert float(np.hypot(boss.vertices[:, 0], boss.vertices[:, 1]).max() * 1000.0) <= 15.77


@pytest.mark.parametrize("ee", ROBOTIQ_EES)
def test_robotiq_base_collar_is_smooth_coated_metal(ee: str) -> None:
    """Robotiq 2F-85 and 2F-140 base visual GLBs have a wrinkle-free analytic Ø75 mm outer collar
    with uniform satin coated dark metal PBR and exact radial vertex normals (< 1 deg error)."""
    pygltflib = pytest.importorskip("pygltflib")
    DracoPy = pytest.importorskip("DracoPy")

    glb_path = URDF_ROOT / "end_effectors" / ee / "meshes" / "visual" / "visual_glb" / "base.glb"
    glb = pygltflib.GLTF2().load(str(glb_path))
    blob = glb.binary_blob()
    assert len(glb.meshes) == 2

    housing_prim = glb.meshes[0].primitives[0]
    pbr = glb.materials[housing_prim.material].pbrMetallicRoughness
    assert pbr is not None
    np.testing.assert_allclose(pbr.baseColorFactor, [0.11, 0.11, 0.12, 1.0], atol=1e-4)
    assert pbr.metallicFactor == pytest.approx(0.45, abs=1e-4)
    assert pbr.roughnessFactor == pytest.approx(0.48, abs=1e-4)

    bv = glb.bufferViews[housing_prim.extensions["KHR_draco_mesh_compression"]["bufferView"]]
    dec = DracoPy.decode(blob[bv.byteOffset : bv.byteOffset + bv.byteLength])
    pts_mm = np.asarray(dec.points, dtype=np.float64) * 1000.0
    nrm = np.asarray(dec.normals, dtype=np.float64)
    fac = np.asarray(dec.faces, dtype=np.int64)

    v0, v1, v2 = pts_mm[fac[:, 0]], pts_mm[fac[:, 1]], pts_mm[fac[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * np.linalg.norm(fn, axis=1)
    assert int((area < 1e-4).sum()) == 0, f"{ee}: degenerate triangles remain in base housing"

    r = np.hypot(pts_mm[:, 0], pts_mm[:, 1])
    z = pts_mm[:, 2]
    outer_v = (z >= 4.0) & (z <= 28.0) & (r >= 37.3) & (np.abs(pts_mm[:, 0]) > 0.65)
    assert int(outer_v.sum()) >= 1000
    np.testing.assert_allclose(r[outer_v], 37.50, atol=0.05)

    vrad = np.column_stack([pts_mm[outer_v, 0], pts_mm[outer_v, 1], np.zeros(int(outer_v.sum()))])
    vrad /= np.linalg.norm(vrad, axis=1, keepdims=True)
    vn_u = nrm[outer_v] / np.linalg.norm(nrm[outer_v], axis=1, keepdims=True)
    vang_deg = np.degrees(np.arccos(np.clip(np.sum(vn_u * vrad, axis=1), -1.0, 1.0)))
    assert float(vang_deg.max()) < 1.0, f"{ee}: max collar vertex normal error {float(vang_deg.max()):.3f} deg"

    fc = (v0 + v1 + v2) / 3.0
    fr = np.hypot(fc[:, 0], fc[:, 1])
    wall_f = (fc[:, 2] >= 4.0) & (fc[:, 2] <= 28.0) & (fr >= 37.2) & (np.abs(fc[:, 0]) > 0.65)
    fn_u = fn[wall_f] / np.linalg.norm(fn[wall_f], axis=1, keepdims=True)
    frad = np.column_stack([fc[wall_f, 0], fc[wall_f, 1], np.zeros(int(wall_f.sum()))])
    frad /= np.linalg.norm(frad, axis=1, keepdims=True)
    fang_deg = np.degrees(np.arccos(np.clip(np.sum(fn_u * frad, axis=1), -1.0, 1.0)))
    assert float(fang_deg.max()) < 3.0, f"{ee}: max collar face normal error {float(fang_deg.max()):.3f} deg"

    hw_prim = glb.meshes[1].primitives[0]
    hw_bv = glb.bufferViews[hw_prim.extensions["KHR_draco_mesh_compression"]["bufferView"]]
    hw_dec = DracoPy.decode(blob[hw_bv.byteOffset : hw_bv.byteOffset + hw_bv.byteLength])
    hw_pts_mm = np.asarray(hw_dec.points, dtype=np.float64) * 1000.0
    hw_fc = hw_pts_mm[np.asarray(hw_dec.faces, dtype=np.int64)].mean(axis=1)
    hw_collar_txt = (hw_fc[:, 2] >= 3.5) & (hw_fc[:, 2] <= 22.0) & (np.hypot(hw_fc[:, 0], hw_fc[:, 1]) >= 36.8)
    assert int(hw_collar_txt.sum()) == 0, f"{ee}: collar nameplate text triangles still present"


# --- composition & standalone -------------------------------------------------------------------------------------


def test_robotiq_assemblies_insert_adapter_and_lift_gripper(tmp_path: Path) -> None:
    for arm, ee, visual in itertools.product(XARM_ARMS, ROBOTIQ_EES, ("glb", "stl")):
        path = Path(compose_robot_urdf(arm, ee, visual=visual))
        root = ET.parse(path).getroot()
        flange = get_robot_profile(arm).ee_link

        ad_joint = root.find("./joint[@name='robotiq_mounting_fix']")
        assert ad_joint is not None and ad_joint.get("type") == "fixed"
        assert ad_joint.find("parent").get("link") == flange  # type: ignore[union-attr]
        assert ad_joint.find("child").get("link") == "robotiq_mounting"  # type: ignore[union-attr]
        np.testing.assert_allclose([float(v) for v in ad_joint.find("origin").get("xyz").split()], [0, 0, 0])  # type: ignore[union-attr]

        ee_joint = root.find(f"./joint[@name='{ee}_fix']")
        assert ee_joint is not None and ee_joint.get("type") == "fixed"
        assert ee_joint.find("parent").get("link") == flange  # type: ignore[union-attr]
        assert ee_joint.find("child").get("link") == f"{ee}_base_link"  # type: ignore[union-attr]
        np.testing.assert_allclose(
            [float(v) for v in ee_joint.find("origin").get("xyz").split()],  # type: ignore[union-attr]
            [0.0, 0.0, ADAPTER_LIFT_M],
            atol=1e-12,
        )

        ad_link = root.find("./link[@name='robotiq_mounting']")
        assert ad_link is not None
        vis_mesh = ad_link.find("visual/geometry/mesh")
        assert vis_mesh is not None
        if visual == "glb":
            assert (vis_mesh.get("filename") or "").endswith("robotiq_mounting/meshes/visual/robotiq_mounting.glb")
        else:
            assert (vis_mesh.get("filename") or "").endswith("robotiq_mounting/meshes/visual/robotiq_mounting.stl")
            assert ad_link.find("visual/material").get("name") == "AdapterAluminum"  # type: ignore[union-attr]

    # Stacking with camera_stand + camera + mounting_plate shifts adapter by +3 mm and Robotiq by +14 mm.
    stacked = Path(compose_robot_urdf("xarm6", "robotiq_2f_140", accessory="d435", mounting_plate=True))
    assert stacked == Path(
        robot_visual_glb_urdf("xarm6", with_robotiq_2f_140=True, accessory="d435", mounting_plate=True)
    )
    sroot = ET.parse(stacked).getroot()
    ad_xyz = [float(v) for v in sroot.find("./joint[@name='robotiq_mounting_fix']/origin").get("xyz").split()]  # type: ignore[union-attr]
    rq_xyz = [float(v) for v in sroot.find("./joint[@name='robotiq_2f_140_fix']/origin").get("xyz").split()]  # type: ignore[union-attr]
    np.testing.assert_allclose(ad_xyz, [0.0, 0.0, STAND_LIFT_M], atol=1e-12)
    np.testing.assert_allclose(rq_xyz, [0.0, 0.0, STAND_LIFT_M + ADAPTER_LIFT_M], atol=1e-12)

    standalone = Path(robotiq_mounting_urdf())
    assert standalone.is_file()
    assert standalone == Path(compose_standalone_adapter_urdf("robotiq_mounting", visual="glb"))
    custom = Path(compose_standalone_adapter_urdf("robotiq_mounting", visual="stl", out_dir=tmp_path))
    assert custom.parent == tmp_path
    for mesh in ET.parse(custom).getroot().iter("mesh"):
        fn = mesh.get("filename") or ""
        assert not os.path.isabs(fn) and (custom.parent / fn).resolve().is_file()


# --- kinematics & Coal clearances ---------------------------------------------------------------------------------


def test_adapter_frames_and_robotiq_tcp_offsets() -> None:
    pin = pytest.importorskip("pinocchio")

    model = pin.buildModelFromUrdf(str(ADAPTER_DIR / "robotiq_mounting.urdf"))
    data = model.createData()
    pin.framesForwardKinematics(model, data, pin.neutral(model))

    def frame_mm(name: str) -> np.ndarray:
        fid = model.getFrameId(name)
        assert fid < len(model.frames), name
        return np.asarray(data.oMf[fid].translation, dtype=np.float64) * 1000.0

    half = 25.0 * math.sqrt(0.5)
    expected_mm = {
        "robotiq_mounting_flange_mount": (0.0, 0.0, 0.0),
        "robotiq_mounting_tool_mount": (0.0, 0.0, 11.0),
        "robotiq_mounting_xarm_pin": (0.0, -25.0, 0.0),
        "robotiq_mounting_robotiq_pin": (0.0, -30.5, 11.0),
        "robotiq_mounting_xarm_hole_0": (half, half, 0.0),
        "robotiq_mounting_xarm_hole_1": (-half, half, 0.0),
        "robotiq_mounting_xarm_hole_2": (-half, -half, 0.0),
        "robotiq_mounting_xarm_hole_3": (half, -half, 0.0),
        "robotiq_mounting_robotiq_hole_0": (16.0, 27.0, 11.0),
        "robotiq_mounting_robotiq_hole_1": (-16.0, 27.0, 11.0),
        "robotiq_mounting_robotiq_hole_2": (-16.0, -27.0, 11.0),
        "robotiq_mounting_robotiq_hole_3": (16.0, -27.0, 11.0),
    }
    for name, xyz in expected_mm.items():
        np.testing.assert_allclose(frame_mm(name), xyz, atol=1e-6, err_msg=name)

    for arm in ("xarm6_1305", "uf850"):
        bare_urdf = get_robot_profile(arm).assets_dir / get_robot_profile(arm).visual_glb_urdf
        bm = pin.buildModelFromUrdf(str(bare_urdf))
        bare_mass = sum(i.mass for i in bm.inertias)
        flange = get_robot_profile(arm).ee_link

        for ee in ROBOTIQ_EES:
            rq_part = pin.buildModelFromUrdf(
                str(URDF_ROOT / "end_effectors" / ee / f"{ee}_movable_visual.urdf"), mimic=True
            )
            rq_mass = sum(i.mass for i in rq_part.inertias)

            m = pin.buildModelFromUrdf(compose_robot_urdf(arm, ee), mimic=True)
            d = m.createData()
            pin.framesForwardKinematics(m, d, pin.neutral(m))
            oMf_flange = d.oMf[m.getFrameId(flange)]
            oMf_tcp = d.oMf[m.getFrameId(f"{ee}_tcp_link")]
            rel = (oMf_flange.inverse() * oMf_tcp).translation
            np.testing.assert_allclose(rel, [0.0, 0.0, ROBOTIQ_BASE_TCP_Z_M[ee] + ADAPTER_LIFT_M], atol=1e-9)
            assert sum(i.mass for i in m.inertias) == pytest.approx(bare_mass + rq_mass + ADAPTER_MASS_KG, abs=1e-6)


@pytest.mark.parametrize("ee", ROBOTIQ_EES)
@pytest.mark.parametrize("camera", CAMERAS)
def test_robotiq_with_camera_stand_clearances(ee: str, camera: str) -> None:
    pin = pytest.importorskip("pinocchio")
    coal = pytest.importorskip("coal")

    urdf = compose_robot_urdf("xarm6_1305", ee, accessory=camera)
    model = pin.buildModelFromUrdf(urdf, mimic=True)
    data = model.createData()
    geom = pin.buildGeomFromUrdf(model, urdf, pin.GeometryType.COLLISION, package_dirs=[str(Path(urdf).parent)])
    gdata = geom.createData()
    owner = [model.frames[g.parentFrame].name for g in geom.geometryObjects]
    req = coal.DistanceRequest()

    def min_dist(q: np.ndarray, a_links: set[str], b_links: set[str]) -> float:
        pin.updateGeometryPlacements(model, data, geom, gdata, q)
        a_idx = [idx for idx, n in enumerate(owner) if n in a_links]
        b_idx = [idx for idx, n in enumerate(owner) if n in b_links]
        assert a_idx and b_idx, (a_links, b_links)
        best = float("inf")
        for i in a_idx:
            for j in b_idx:
                res = coal.DistanceResult()
                best = min(
                    best,
                    float(
                        coal.distance(
                            geom.geometryObjects[i].geometry,
                            gdata.oMg[i],
                            geom.geometryObjects[j].geometry,
                            gdata.oMg[j],
                            req,
                            res,
                        )
                    ),
                )
        return best

    q = pin.neutral(model)
    body = {f"{camera}_body"}
    assert min_dist(q, {"camera_stand"}, {"robotiq_mounting"}) > -1e-6
    assert min_dist(q, body, {"robotiq_mounting"}) > 0.0085
    assert min_dist(q, body, {f"{ee}_base_link"}) > 0.0080

    drive_j = model.joints[model.getJointId(f"{ee}_drive_joint")]
    lo = float(model.lowerPositionLimit[drive_j.idx_q])
    hi = float(model.upperPositionLimit[drive_j.idx_q])
    moving = {
        f"{ee}_{side}_{part}" for side in ("left", "right") for part in ("outer_knuckle", "finger", "inner_knuckle")
    }
    for val in (lo, 0.5 * (lo + hi), hi):
        q_s = q.copy()
        q_s[drive_j.idx_q] = val
        assert min_dist(q_s, body, moving) > 0.045
        assert min_dist(q_s, {"robotiq_mounting"}, moving) > 0.030


# --- Genesis GPU --------------------------------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("arm", "ee", "accessory", "mounting_plate"),
    [
        ("xarm6_1305", "robotiq_2f_85", None, False),
        ("xarm6_1305", "robotiq_2f_140", "d435", False),
        ("uf850", "robotiq_2f_85", None, True),
    ],
)
def test_genesis_builds_robotiq_with_adapter_mass_and_offset(
    arm: str, ee: str, accessory: str | None, mounting_plate: bool
) -> None:
    gs = pytest.importorskip("genesis")
    pin = pytest.importorskip("pinocchio")
    from ufactory.config import load_runtime_config
    from ufactory.grippers.robotiq import get_robotiq_spec
    from ufactory.simulation import (
        GenesisRuntimeManager,
        configure_robotiq_drive,
        configure_robotiq_mimic_constraints,
        make_rigid_options,
    )

    spec = get_robotiq_spec(ee)
    cfg = load_runtime_config(arm, gripper=ee)
    urdf = compose_robot_urdf(arm, ee, accessory=accessory, mounting_plate=mounting_plate)
    pin_model = pin.buildModelFromUrdf(urdf, mimic=True)
    flange_name = get_robot_profile(arm).ee_link

    with GenesisRuntimeManager(cfg.simulation):
        scene = gs.Scene(
            show_viewer=False,
            sim_options=gs.options.SimOptions(dt=0.02, substeps=32),
            rigid_options=make_rigid_options(gs, constraint_timeconst=0.00125),
        )
        scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -0.02)))
        robot = scene.add_entity(gs.morphs.URDF(file=urdf, fixed=True))
        scene.build()
        configure_robotiq_mimic_constraints(robot, ee)
        drive_idx, all_idx = configure_robotiq_drive(robot, ee)

        gs_mass = sum(float(link.get_mass()) for link in robot.links)
        assert gs_mass == pytest.approx(sum(i.mass for i in pin_model.inertias), abs=1e-4)

        # Verify flange-to-finger Z offset at exact open pose (before free-fall gravity swings uncontrolled DOFs).
        robot.set_dofs_position([spec.linkage.drive_lower_rad] * len(all_idx), all_idx, zero_velocity=True)
        flange_pos = robot.get_link(flange_name).get_pos().cpu().numpy().reshape(-1)
        lf_pos = robot.get_link(f"{ee}_left_finger").get_pos().cpu().numpy().reshape(-1)
        rf_pos = robot.get_link(f"{ee}_right_finger").get_pos().cpu().numpy().reshape(-1)
        measured_z = float(np.linalg.norm(0.5 * (lf_pos + rf_pos) - flange_pos))
        expected_z = ROBOTIQ_TOOL_TIP_Z_M[ee] + (STAND_LIFT_M if accessory else 0.0)
        assert measured_z == pytest.approx(expected_z, abs=0.001)

        # Verify the gripper strokes cleanly under PD control with the adapter mounted.
        arm_idx = [robot.get_joint(f"joint{i}").dofs_idx_local[0] for i in range(1, 7)]
        q_hold = [0.0, -0.5, -0.5, 0.0, 1.0, 0.0]
        robot.set_dofs_kp([1500.0, 1500.0, 1000.0, 1000.0, 500.0, 500.0], arm_idx)
        robot.set_dofs_kv([150.0, 150.0, 100.0, 100.0, 50.0, 50.0], arm_idx)
        robot.set_dofs_position(q_hold, arm_idx, zero_velocity=True)
        robot.control_dofs_position(q_hold, arm_idx)
        robot.control_dofs_position([spec.linkage.drive_lower_rad], drive_idx)
        for _ in range(10):
            scene.step()
        q_open = float(robot.get_dofs_position(drive_idx).cpu().numpy().reshape(-1)[0])
        robot.control_dofs_position([spec.linkage.drive_upper_rad], drive_idx)
        for _ in range(40):
            scene.step()
        q_closed = float(robot.get_dofs_position(drive_idx).cpu().numpy().reshape(-1)[0])
        assert q_closed - q_open > 0.35
        assert len(drive_idx) == 1
