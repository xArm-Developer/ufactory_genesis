"""Camera stand + depth camera assembly contract tests (assets/urdf/assembly.yaml spacers/cameras)."""

from __future__ import annotations

import itertools
import json
import os
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from ufactory.robots.assembly import (
    AssemblyError,
    AssemblyRef,
    accessories,
    assembly_ref,
    cameras,
    compose_robot_urdf,
    parse_assembly_ref,
    resolve_accessory,
    spacers,
    supported_accessories,
)
from ufactory.robots.paths import robot_urdf, robot_visual_glb_urdf
from ufactory.robots.registry import PROJECT_ROOT, get_robot_profile

pin = pytest.importorskip("pinocchio")

URDF_ROOT = PROJECT_ROOT / "assets" / "urdf"
CAMERAS = ("d405", "d435", "gemini_305", "zed_x_nano")
ACCESSORIES = ("camera_stand", *CAMERAS)
XARM_ARMS = ("xarm5_1305", "xarm6_1305", "xarm7_1305", "uf850")
ARMS = (*XARM_ARMS, "lite6")
ARM_END_EFFECTORS: dict[str, tuple[str | None, ...]] = {
    **{arm: (None, "gripper_g2", "vacuum_gripper", "robotiq_2f_85", "robotiq_2f_140") for arm in XARM_ARMS},
    "lite6": (None, "lite6_gripper", "lite6_vacuum_gripper"),
}
STAND_MASS = 0.048
CAMERA_MASS = {"d405": 0.058, "d435": 0.075, "gemini_305": 0.068, "zed_x_nano": 0.075}
# camera_stand hole frames matched index-by-index with each camera's <camera>_hole_<i> frames.
STAND_HOLES = {
    "d405": ("stand_hole_20_0", "stand_hole_20_1"),
    "d435": ("stand_hole_45_0", "stand_hole_45_1"),
    "gemini_305": ("stand_hole_20_0", "stand_hole_20_1"),
    "zed_x_nano": tuple(f"stand_hole_20x20_{i}" for i in range(4)),
}
# Gap between the stand face and the camera's rear face (the D435 ROS mesh pokes 0.0547 mm past its nominal face).
MOUNT_GAP_M = {"d405": 0.0, "d435": 0.000055, "gemini_305": 0.0, "zed_x_nano": 0.0}
G2_FINGERS = {
    "left_outer_knuckle",
    "left_finger",
    "left_inner_knuckle",
    "right_outer_knuckle",
    "right_finger",
    "right_inner_knuckle",
}
LITE6_FINGERS = {"uflite_finger1", "uflite_finger2"}


def _model(urdf: str | Path) -> tuple[pin.Model, pin.Data]:
    model = pin.buildModelFromUrdf(str(urdf), mimic=True)
    return model, model.createData()


def _placements(urdf: str | Path, q: np.ndarray | None = None) -> tuple[pin.Model, pin.Data]:
    model, data = _model(urdf)
    pin.framesForwardKinematics(model, data, pin.neutral(model) if q is None else q)
    return model, data


def _frame(model: pin.Model, data: pin.Data, name: str) -> pin.SE3:
    fid = model.getFrameId(name)
    assert fid < len(model.frames), name
    return data.oMf[fid]


# --- catalog and references -------------------------------------------------------------------------------------


def test_catalog_lists_the_stand_and_four_cameras() -> None:
    assert accessories() == ACCESSORIES
    assert tuple(spacers()) == ("camera_stand",)
    assert tuple(cameras()) == CAMERAS
    stand = spacers()["camera_stand"]
    assert stand.flanges == ("xarm", "lite6")
    assert stand.robots == ARMS
    assert stand.tool_lift_m == pytest.approx(0.003)
    assert stand.end_effectors == (
        "gripper_g2",
        "vacuum_gripper",
        "robotiq_2f_85",
        "robotiq_2f_140",
        "lite6_gripper",
        "lite6_vacuum_gripper",
    )
    for name, cam in cameras().items():
        assert cam.spacer == "camera_stand"
        assert cam.optical_frame == f"{name}_optical_frame"
        assert resolve_accessory(name) == (stand, cam)
    assert resolve_accessory("camera_stand") == (stand, None)
    with pytest.raises(AssemblyError, match="unknown accessory"):
        resolve_accessory("d455")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("assembly:gripper_g2+d435", AssemblyRef("gripper_g2", "glb", "d435")),
        ("assembly:bare+camera_stand:stl", AssemblyRef(None, "stl", "camera_stand")),
        ("assembly:vacuum_gripper+gemini_305:stl", AssemblyRef("vacuum_gripper", "stl", "gemini_305")),
    ],
)
def test_parse_accessory_refs_round_trip(value: str, expected: AssemblyRef) -> None:
    assert parse_assembly_ref(value) == expected
    assert str(expected) == value


@pytest.mark.parametrize(
    "value",
    [
        "assembly:bare",
        "assembly:+d435",
        "assembly:gripper_g2+",
        "assembly:a+b+c",
        "assembly:gripper_g2+zed_x_nano:glb:static",
    ],
)
def test_parse_accessory_refs_rejects_malformed(value: str) -> None:
    with pytest.raises(AssemblyError, match="malformed"):
        parse_assembly_ref(value)


def test_assembly_ref_needs_a_part() -> None:
    assert assembly_ref(None, accessory="d405") == "assembly:bare+d405"
    with pytest.raises(AssemblyError, match="end effector or an accessory"):
        assembly_ref(None)


def test_compatibility_matrix() -> None:
    for arm in ARMS:
        assert supported_accessories(arm) == ACCESSORIES
        assert get_robot_profile(arm).supports_camera_stand
        for ee in ARM_END_EFFECTORS[arm]:
            assert supported_accessories(arm, ee) == ACCESSORIES
    for arm in XARM_ARMS:
        for ee in ("bio_gripper_g2", "lite6_gripper", "lite6_vacuum_gripper"):
            assert supported_accessories(arm, ee) == ()
    for ee in ("gripper_g2", "bio_gripper_g2", "vacuum_gripper", "robotiq_2f_85", "robotiq_2f_140"):
        assert supported_accessories("lite6", ee) == ()
    cases = [
        (("lite6", "gripper_g2"), {"accessory": "d435"}, "not supported on lite6"),
        (("xarm6", "lite6_gripper"), {"accessory": "d435"}, "not supported on xarm6_1305"),
        (("xarm6", "bio_gripper_g2"), {"accessory": "d405"}, "does not fit with bio_gripper_g2"),
        (("xarm6", None), {"accessory": "d455"}, "unknown accessory"),
        (("xarm6", None), {}, "end effector or an accessory"),
    ]
    for args, kwargs, message in cases:
        with pytest.raises(AssemblyError, match=message):
            compose_robot_urdf(*args, **kwargs)


# --- composition --------------------------------------------------------------------------------------------------


def _variants() -> tuple[str, ...]:
    return ("glb", "stl")


def test_every_combination_composes_deterministically_with_relative_existing_meshes(tmp_path: Path) -> None:
    count = 0
    for arm in ARMS:
        for ee, acc in itertools.product(ARM_END_EFFECTORS[arm], ACCESSORIES):
            for visual in _variants():
                path = Path(compose_robot_urdf(arm, ee, accessory=acc, visual=visual))
                again = Path(compose_robot_urdf(arm, ee, accessory=acc, visual=visual))
                assert path == again
                label = f"{ee or 'bare'}__{acc}"
                assert path.name.startswith(f"{arm}__{label}__{visual}.")
                root = ET.parse(path).getroot()
                names = [e.get("name") for e in root if e.tag in ("link", "joint")]
                assert len(names) == len(set(names))
                assert sum(e.tag == "link" for e in root) == sum(e.tag == "joint" for e in root) + 1
                for mesh in root.iter("mesh"):
                    filename = mesh.get("filename") or ""
                    assert not os.path.isabs(filename), filename
                    assert (path.parent / filename).resolve().is_file(), filename
                count += 1
    assert count == (4 * 5 + 3) * 5 * 2

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        elsewhere = Path(compose_robot_urdf("xarm6", "gripper_g2", accessory="zed_x_nano"))
    finally:
        os.chdir(cwd)
    assert (
        elsewhere.read_bytes() == Path(compose_robot_urdf("xarm6", "gripper_g2", accessory="zed_x_nano")).read_bytes()
    )


def test_path_helpers_accept_accessories() -> None:
    assert robot_urdf("xarm6", "assembly:gripper_g2+d435") == compose_robot_urdf(
        "xarm6", "gripper_g2", accessory="d435"
    )
    assert robot_urdf("uf850", "assembly:bare+camera_stand:stl") == compose_robot_urdf(
        "uf850", None, accessory="camera_stand", visual="stl"
    )
    assert robot_urdf("lite6", "assembly:lite6_gripper+d435") == compose_robot_urdf(
        "lite6", "lite6_gripper", accessory="d435"
    )
    assert robot_visual_glb_urdf("xarm7", accessory="d405") == compose_robot_urdf("xarm7", None, accessory="d405")
    assert robot_visual_glb_urdf("lite6", accessory="d405") == compose_robot_urdf("lite6", None, accessory="d405")
    assert robot_visual_glb_urdf("xarm6", with_gripper_g2=True, accessory="zed_x_nano") == compose_robot_urdf(
        "xarm6", "gripper_g2", accessory="zed_x_nano"
    )
    assert robot_visual_glb_urdf("lite6", with_lite6_gripper=True, accessory="zed_x_nano") == compose_robot_urdf(
        "lite6", "lite6_gripper", accessory="zed_x_nano"
    )
    assert robot_visual_glb_urdf(
        "lite6", with_lite6_vacuum_gripper=True, accessory="camera_stand"
    ) == compose_robot_urdf("lite6", "lite6_vacuum_gripper", accessory="camera_stand")
    with pytest.raises(ValueError, match="does not fit"):
        robot_visual_glb_urdf("xarm6", with_bio_gripper_g2=True, accessory="d405")
    with pytest.raises(ValueError, match="does not support"):
        robot_visual_glb_urdf("lite6", with_gripper_g2=True, accessory="d405")


def test_stl_mode_uses_stand_stl_and_camera_boxes() -> None:
    root = ET.parse(compose_robot_urdf("xarm6", "gripper_g2", accessory="d435", visual="stl")).getroot()
    stand = root.find("./link[@name='camera_stand']")
    cam = root.find("./link[@name='d435_body']")
    assert stand is not None and cam is not None
    stand_meshes = [m.get("filename") or "" for m in stand.findall("visual/geometry/mesh")]
    assert len(stand_meshes) == 1 and stand_meshes[0].endswith("camera_stand/meshes/visual/camera_stand.stl")
    assert [v.find("material").get("name") for v in stand.findall("visual")] == ["StandAluminum"]  # type: ignore[union-attr]
    assert cam.find("visual/geometry/box") is not None and cam.find("visual/geometry/mesh") is None
    assert {v.find("material").get("name") for v in cam.findall("visual")} == {"d435_housing"}  # type: ignore[union-attr]
    defined = {m.get("name") for m in root.findall("material")}
    assert {"StandAluminum", "d435_housing"} <= defined


def _tcp(model: pin.Model, data: pin.Data, arm: str, ee: str | None) -> np.ndarray:
    name = {
        "gripper_g2": "link_tcp",
        "vacuum_gripper": "vacuum_tcp",
        "robotiq_2f_85": "robotiq_2f_85_tcp_link",
        "robotiq_2f_140": "robotiq_2f_140_tcp_link",
        "lite6_gripper": "link_tcp",
        "lite6_vacuum_gripper": "link_tcp",
    }.get(ee or "", "")
    if not name:
        name = "link_eef" if model.existFrame("link_eef") else get_robot_profile(arm).ee_link
    return np.asarray(_frame(model, data, name).translation, dtype=np.float64)


@pytest.mark.parametrize("arm", ARMS)
def test_tool_moves_3_mm_and_mass_adds_up(arm: str) -> None:
    flange = get_robot_profile(arm).ee_link
    rng = np.random.default_rng(39)
    for ee in ARM_END_EFFECTORS[arm]:
        base_urdf = (
            compose_robot_urdf(arm, ee)
            if ee
            else get_robot_profile(arm).assets_dir / get_robot_profile(arm).visual_glb_urdf
        )
        bm, bd = _model(base_urdf)
        q = np.clip(rng.uniform(-1.0, 1.0, bm.nq), bm.lowerPositionLimit, bm.upperPositionLimit)
        pin.framesForwardKinematics(bm, bd, q)
        base_tcp = _tcp(bm, bd, arm, ee)
        flange_z = _frame(bm, bd, flange).rotation[:, 2]
        moves = ee is not None or bm.existFrame("link_eef")
        for acc in ACCESSORIES:
            m, d = _model(compose_robot_urdf(arm, ee, accessory=acc))
            assert m.nq == bm.nq
            pin.framesForwardKinematics(m, d, q)
            np.testing.assert_allclose(_tcp(m, d, arm, ee) - base_tcp, flange_z * (0.003 if moves else 0.0), atol=1e-12)
            added = sum(i.mass for i in m.inertias) - sum(i.mass for i in bm.inertias)
            assert added == pytest.approx(STAND_MASS + CAMERA_MASS.get(acc, 0.0), abs=1e-12)


@pytest.mark.parametrize("arm", ["xarm6", "lite6"])
@pytest.mark.parametrize("camera", CAMERAS)
def test_camera_sits_on_the_stand_holes_and_looks_along_flange_z(arm: str, camera: str) -> None:
    model, data = _placements(compose_robot_urdf(arm, None, accessory=camera))
    flange = _frame(model, data, "link6")
    stand_mount = _frame(model, data, "stand_camera_mount")
    cam_mount = _frame(model, data, f"{camera}_mount")
    n_stand = stand_mount.rotation[:, 2]
    assert float(n_stand @ cam_mount.rotation[:, 2]) == pytest.approx(-1.0, abs=1e-9)
    gap = float((cam_mount.translation - stand_mount.translation) @ n_stand)
    assert gap == pytest.approx(MOUNT_GAP_M[camera], abs=1e-9)
    for i, stand_hole in enumerate(STAND_HOLES[camera]):
        offset = _frame(model, data, f"{camera}_hole_{i}").translation - _frame(model, data, stand_hole).translation
        assert float(np.linalg.norm(offset - n_stand * MOUNT_GAP_M[camera])) < 6e-5, (camera, i)
    assert not model.existFrame(f"{camera}_hole_{len(STAND_HOLES[camera])}")

    optical = flange.inverse() * _frame(model, data, f"{camera}_optical_frame")
    np.testing.assert_allclose(optical.rotation[:, 2], [0.0, 0.0, 1.0], atol=1e-9)  # looks along flange +Z
    np.testing.assert_allclose(optical.rotation[:, 1], [-1.0, 0.0, 0.0], atol=1e-9)  # image top towards flange +X
    assert optical.translation[0] == pytest.approx(0.0675, abs=1e-4)
    assert optical.translation[1] == pytest.approx(0.0, abs=1e-5)


def test_stand_frames_match_the_drawing() -> None:
    model, data = _placements(URDF_ROOT / "mounts" / "camera_stand" / "camera_stand.urdf")
    expected_mm = {
        "stand_camera_mount": (67.5, 3.0, 0.0),
        "stand_hole_20_0": (67.5, 3.0, -10.0),
        "stand_hole_20_1": (67.5, 3.0, 10.0),
        "stand_hole_45_0": (67.5, 3.0, -22.5),
        "stand_hole_45_1": (67.5, 3.0, 22.5),
        "stand_hole_20x20_0": (57.5, 3.0, -10.0),
        "stand_hole_20x20_1": (57.5, 3.0, 10.0),
        "stand_hole_20x20_2": (77.5, 3.0, -10.0),
        "stand_hole_20x20_3": (77.5, 3.0, 10.0),
    }
    for name, xyz in expected_mm.items():
        np.testing.assert_allclose(_frame(model, data, name).translation * 1000.0, xyz, atol=1e-9, err_msg=name)


# --- collision clearances -----------------------------------------------------------------------------------------


def _geometry(urdf: str) -> tuple[pin.Model, pin.Data, pin.GeometryModel, pin.GeometryData]:
    model, data = _model(urdf)
    geom = pin.buildGeomFromUrdf(model, urdf, pin.GeometryType.COLLISION, package_dirs=[str(Path(urdf).parent)])
    return model, data, geom, geom.createData()


def _min_distance(
    model: pin.Model,
    data: pin.Data,
    geom: pin.GeometryModel,
    gdata: pin.GeometryData,
    q: np.ndarray,
    links_a: set[str],
    links_b: set[str],
) -> float:
    import coal

    pin.updateGeometryPlacements(model, data, geom, gdata, q)
    owner = [model.frames[g.parentFrame].name for g in geom.geometryObjects]
    a = [i for i, n in enumerate(owner) if n in links_a]
    b = [i for i, n in enumerate(owner) if n in links_b]
    assert a and b, (links_a, links_b)
    req = coal.DistanceRequest()
    best = float("inf")
    for i in a:
        for j in b:
            res = coal.DistanceResult()
            gi, gj = geom.geometryObjects[i], geom.geometryObjects[j]
            best = min(best, float(coal.distance(gi.geometry, gdata.oMg[i], gj.geometry, gdata.oMg[j], req, res)))
    return best


@pytest.mark.parametrize("camera", CAMERAS)
def test_g2_clearances_and_zed_connector_in_the_stand_opening(camera: str) -> None:
    urdf = compose_robot_urdf("xarm6", "gripper_g2", accessory=camera)
    model, data, geom, gdata = _geometry(urdf)
    body = {f"{camera}_body"}
    drive = model.joints[model.getJointId("drive_joint")].idx_q
    q = pin.neutral(model)
    # Stand vs camera: they touch at the rear face (ZED: its connector passes the opening without overlap).
    assert _min_distance(model, data, geom, gdata, q, {"camera_stand"}, body) > -1e-6
    assert _min_distance(model, data, geom, gdata, q, {"camera_stand"}, {"xarm_gripper_base_link"}) > 0.0
    assert _min_distance(model, data, geom, gdata, q, body, {"xarm_gripper_base_link"}) > 0.008
    assert _min_distance(model, data, geom, gdata, q, body, {"link6"}) > 0.010
    for angle in (0.0572, 0.46, 0.865):  # open, mid, closed
        q[drive] = angle
        assert _min_distance(model, data, geom, gdata, q, body, G2_FINGERS) > 0.040


@pytest.mark.parametrize("camera", CAMERAS)
def test_lite6_clearances_with_camera_stand_and_tools(camera: str) -> None:
    body = {f"{camera}_body"}
    urdf = compose_robot_urdf("lite6", "lite6_gripper", accessory=camera)
    model, data, geom, gdata = _geometry(urdf)
    stroke = model.joints[model.getJointId("finger_joint1")].idx_q
    q = pin.neutral(model)
    assert _min_distance(model, data, geom, gdata, q, {"camera_stand"}, body) > -1e-6
    assert _min_distance(model, data, geom, gdata, q, {"camera_stand"}, {"link6"}) > -1e-6
    assert _min_distance(model, data, geom, gdata, q, {"camera_stand"}, {"uflite_gripper_link"}) > -1e-6
    assert _min_distance(model, data, geom, gdata, q, body, {"uflite_gripper_link"}) > 0.007
    assert _min_distance(model, data, geom, gdata, q, body, {"link6"}) > 0.008
    for pos in (0.0, 0.00445, 0.0089):  # open, mid, closed
        q[stroke] = pos
        assert _min_distance(model, data, geom, gdata, q, body, LITE6_FINGERS) > 0.045

    vac_urdf = compose_robot_urdf("lite6", "lite6_vacuum_gripper", accessory=camera)
    vm, vd, vg, vgd = _geometry(vac_urdf)
    vq = pin.neutral(vm)
    assert _min_distance(vm, vd, vg, vgd, vq, {"camera_stand"}, {"uflite_vacuum_gripper_link"}) > -1e-6
    assert _min_distance(vm, vd, vg, vgd, vq, body, {"uflite_vacuum_gripper_link"}) > 0.007


# --- assets -------------------------------------------------------------------------------------------------------


def _glb_json(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    assert raw[:4] == b"glTF"
    (length,) = struct.unpack("<I", raw[12:16])
    data: dict[str, object] = json.loads(raw[20 : 20 + length])
    return data


@pytest.mark.parametrize("part", ACCESSORIES)
def test_part_assets_budget_inertia_and_license(part: str) -> None:
    part_dir = URDF_ROOT / ("mounts" if part == "camera_stand" else "cameras") / part
    glb = part_dir / "meshes" / "visual" / f"{part}.glb"
    assert glb.stat().st_size < 1_000_000
    meta = _glb_json(glb)
    assert meta["extensionsRequired"] == ["KHR_draco_mesh_compression"]
    materials = [m["name"] for m in meta["materials"]]  # type: ignore[attr-defined,index]
    assert materials == (
        ["stand_anodized_aluminum"] if part == "camera_stand" else [f"{part}_housing", f"{part}_glass"]
    )
    if part in ("camera_stand", "d405", "d435", "gemini_305"):
        housing_pbr = meta["materials"][0]["pbrMetallicRoughness"]  # type: ignore[index]
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
            np.testing.assert_allclose(housing_pbr["baseColorFactor"], arm_metals[0]["baseColorFactor"], atol=1e-5)
            assert housing_pbr["metallicFactor"] == pytest.approx(arm_metals[0]["metallicFactor"], abs=1e-5)
            assert housing_pbr["roughnessFactor"] == pytest.approx(arm_metals[0]["roughnessFactor"], abs=0.01)
    elif part == "zed_x_nano":
        housing_pbr = meta["materials"][0]["pbrMetallicRoughness"]  # type: ignore[index]
        np.testing.assert_allclose(housing_pbr["baseColorFactor"], [0.16, 0.17, 0.19, 1.0], atol=1e-4)
    if part != "camera_stand":
        glass_pbr = meta["materials"][1]["pbrMetallicRoughness"]  # type: ignore[index]
        np.testing.assert_allclose(glass_pbr["baseColorFactor"], [0.025, 0.03, 0.035, 1.0], atol=1e-4)

    root = ET.parse(part_dir / f"{part}.urdf").getroot()
    base = "camera_stand" if part == "camera_stand" else f"{part}_body"
    inertial = root.find(f"./link[@name='{base}']/inertial")
    assert inertial is not None
    mass = float(inertial.find("mass").get("value"))  # type: ignore[union-attr,arg-type]
    assert mass == pytest.approx(STAND_MASS if part == "camera_stand" else CAMERA_MASS[part])
    i = {k: float(v) for k, v in inertial.find("inertia").attrib.items()}  # type: ignore[union-attr]
    tensor = np.array([[i["ixx"], i["ixy"], i["ixz"]], [i["ixy"], i["iyy"], i["iyz"]], [i["ixz"], i["iyz"], i["izz"]]])
    eig = np.linalg.eigvalsh(tensor)
    assert eig.min() > 0.0 and eig[0] + eig[1] > eig[2]
    assert (part_dir / "README.md").is_file()
    assert (part_dir / "LICENSE").is_file() == (part in ("d405", "d435"))


def test_stand_collision_is_45_prisms() -> None:
    stand_dir = URDF_ROOT / "mounts" / "camera_stand"
    root = ET.parse(stand_dir / "camera_stand.urdf").getroot()
    meshes = [m.get("filename") for m in root.findall("./link[@name='camera_stand']/collision/geometry/mesh")]
    assert meshes == [f"meshes/collision/piece_{i:03d}.stl" for i in range(45)]
    files = sorted(p.name for p in (stand_dir / "meshes" / "collision").iterdir())
    assert files == [f"piece_{i:03d}.stl" for i in range(45)]


# --- Genesis ------------------------------------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("camera", ["d435", "zed_x_nano"])
def test_genesis_builds_and_g2_strokes_with_camera(camera: str) -> None:
    gs = pytest.importorskip("genesis")
    from ufactory.config import load_runtime_config
    from ufactory.simulation import GenesisRuntimeManager, make_rigid_options

    config = load_runtime_config("xarm6")
    with GenesisRuntimeManager(config.simulation):
        scene = gs.Scene(
            show_viewer=False,
            sim_options=gs.options.SimOptions(dt=0.005, substeps=2),
            rigid_options=make_rigid_options(gs),
        )
        scene.add_entity(gs.morphs.Plane())
        robot = scene.add_entity(
            gs.morphs.URDF(file=compose_robot_urdf("xarm6", "gripper_g2", accessory=camera), fixed=True)
        )
        scene.build()
        # Fixed links merge into link6, so the moving links are those of the plain G2 assembly.
        assert [link.name for link in robot.links] == [
            "world",
            *(f"link{i}" for i in range(1, 7)),
            *(f"{side}_{part}" for side in ("left", "right") for part in ("outer_knuckle", "finger", "inner_knuckle")),
        ]
        pin_model, _ = _model(compose_robot_urdf("xarm6", "gripper_g2", accessory=camera))
        gs_mass = sum(float(link.get_mass()) for link in robot.links)
        assert gs_mass == pytest.approx(sum(i.mass for i in pin_model.inertias), abs=1e-4)
        arm = [robot.get_joint(f"joint{i}").dofs_idx_local[0] for i in range(1, 7)]
        joints = (
            "drive_joint",
            "left_finger_joint",
            "left_inner_knuckle_joint",
            "right_outer_knuckle_joint",
            "right_finger_joint",
            "right_inner_knuckle_joint",
        )
        grip = [robot.get_joint(j).dofs_idx_local[0] for j in joints]
        q_ready = [0.0, -0.63088336, -0.837867545, 0.0, 1.468750905, 0.0]
        robot.set_dofs_kp([1500.0, 1500.0, 1000.0, 1000.0, 500.0, 500.0], arm)
        robot.set_dofs_kv([150.0, 150.0, 100.0, 100.0, 50.0, 50.0], arm)
        robot.set_dofs_kp([150.0] * len(grip), grip)
        robot.set_dofs_kv([15.0] * len(grip), grip)
        robot.set_dofs_position(q_ready, arm, zero_velocity=True)
        robot.control_dofs_position(q_ready, arm)
        robot.control_dofs_position([0.0572] * len(grip), grip)
        for _ in range(30):
            scene.step()
        q_open = float(robot.get_dofs_position([grip[0]]).cpu().numpy().reshape(-1)[0])
        robot.control_dofs_position([0.865] * len(grip), grip)
        for _ in range(180):
            scene.step()
        q_closed = float(robot.get_dofs_position([grip[0]]).cpu().numpy().reshape(-1)[0])
        q_arm = robot.get_dofs_position(arm).cpu().numpy().reshape(-1)
        assert np.all(np.isfinite(q_arm))
        np.testing.assert_allclose(q_arm, q_ready, atol=0.02)
        assert q_closed - q_open > 0.55
