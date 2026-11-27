"""Runtime URDF assembly (arm + end effector) contract tests."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import yaml

from ufactory.config import RepositoryAssetStore, load_runtime_config
from ufactory.kinematics.tcp_offset import rpy_matrix
from ufactory.robots import assembly
from ufactory.robots.assembly import (
    AssemblyError,
    AssemblyRef,
    assembly_ref,
    compose_robot_urdf,
    end_effectors,
    get_end_effector,
    parse_assembly_ref,
    resolve_robot_urdf,
    supported_end_effectors,
    validate_robot_urdf,
)
from ufactory.robots.paths import robot_urdf, robot_visual_glb_urdf
from ufactory.robots.registry import PROJECT_ROOT, ROBOT_PROFILES, get_robot_profile

_VARIANTS = ("glb", "stl")


def _supported() -> list[tuple[str, str, str]]:
    out = []
    for name, ee in end_effectors().items():
        for key in ee.robots:
            for visual in _VARIANTS:
                out.append((key, name, visual))
    return out


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("assembly:gripper_g2", AssemblyRef("gripper_g2", "glb")),
        ("assembly:lite6_gripper:stl", AssemblyRef("lite6_gripper", "stl")),
        ("assembly:bio_gripper_g2", AssemblyRef("bio_gripper_g2", "glb")),
        ("xarm6_1305.urdf", None),
    ],
)
def test_parse_assembly_ref(value: str, expected: AssemblyRef | None) -> None:
    assert parse_assembly_ref(value) == expected
    if expected is not None:
        assert str(expected) == value


@pytest.mark.parametrize(
    "value",
    ["assembly:", "assembly:g2:obj", "assembly:gripper_g2:glb:static", "assembly:g2:glb:moving", "assembly:a:b:c:d"],
)
def test_parse_assembly_ref_rejects_malformed(value: str) -> None:
    with pytest.raises(AssemblyError):
        parse_assembly_ref(value)


def test_assembly_ref_omits_defaults() -> None:
    assert assembly_ref("gripper_g2") == "assembly:gripper_g2"
    assert assembly_ref("gripper_g2", visual="stl") == "assembly:gripper_g2:stl"
    assert parse_assembly_ref("assembly:bio_gripper_g2:glb") == AssemblyRef("bio_gripper_g2", "glb")


def test_compatibility_matrix() -> None:
    assert supported_end_effectors("xarm6") == (
        "gripper_g2",
        "bio_gripper_g2",
        "vacuum_gripper",
        "robotiq_2f_85",
        "robotiq_2f_140",
    )
    assert supported_end_effectors("lite6") == ("lite6_gripper", "lite6_vacuum_gripper")
    assert get_robot_profile("uf850").supports_bio_gripper_g2
    assert get_robot_profile("xarm6").supports_robotiq_2f_85
    assert get_robot_profile("xarm6").supports_robotiq_2f_140
    assert not get_robot_profile("lite6").supports_gripper_g2
    assert not get_robot_profile("lite6").supports_robotiq_2f_85
    assert not get_robot_profile("lite6").supports_robotiq_2f_140
    with pytest.raises(AssemblyError, match="not supported on lite6"):
        compose_robot_urdf("lite6", "gripper_g2")
    with pytest.raises(AssemblyError, match="not supported on lite6"):
        compose_robot_urdf("lite6", "robotiq_2f_85")
    with pytest.raises(AssemblyError, match="not supported on lite6"):
        compose_robot_urdf("lite6", "robotiq_2f_140")
    with pytest.raises(AssemblyError, match="unknown end effector"):
        compose_robot_urdf("xarm6", "camera")


def test_composed_urdf_is_deterministic_and_location_independent(tmp_path: Path) -> None:
    first = Path(compose_robot_urdf("xarm6", "gripper_g2"))
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        second = Path(compose_robot_urdf("xarm6_1305", "gripper_g2"))
    finally:
        os.chdir(cwd)
    assert first == second
    assert first.parent == PROJECT_ROOT / assembly.COMPOSED_DIR
    data = first.read_bytes()
    assert data == assembly.serialize_assembly(assembly.build_assembly_tree("xarm6", "gripper_g2"))
    meshes = [m.get("filename") or "" for m in ET.fromstring(data).iter("mesh")]
    assert meshes and not any(os.path.isabs(name) for name in meshes)
    assert all(name.startswith("../../../../assets/urdf/") for name in meshes)


def test_custom_output_directory_keeps_meshes_resolvable(tmp_path: Path) -> None:
    path = Path(compose_robot_urdf("lite6", "lite6_gripper", visual="stl", out_dir=tmp_path))
    assert path.parent == tmp_path
    for mesh in ET.parse(path).getroot().iter("mesh"):
        assert (path.parent / (mesh.get("filename") or "")).resolve().is_file()


def _physics(path: Path) -> dict[tuple[str, str | None], str]:
    out = {}
    for e in ET.parse(path).getroot():
        if e.tag not in ("link", "joint"):
            continue
        e = ET.fromstring(ET.tostring(e))
        for v in e.findall("visual"):
            e.remove(v)
        for mesh in e.iter("mesh"):
            mesh.set("filename", str((path.parent / (mesh.get("filename") or "")).resolve()))
        out[(e.tag, e.get("name"))] = ET.tostring(e, encoding="unicode")
    return out


@pytest.mark.parametrize(("robot", "end_effector"), sorted({(r, ee) for r, ee, _ in _supported()}))
def test_visual_modes_do_not_change_physics(robot: str, end_effector: str) -> None:
    reference = _physics(Path(compose_robot_urdf(robot, end_effector)))
    stl = _physics(Path(compose_robot_urdf(robot, end_effector, visual="stl")))
    # Arm dynamics URDFs differ from GLB visual URDFs only in visuals, except Lite6's fixed-base link_base inertia.
    assert {k for k in reference if reference[k] != stl.get(k)} <= (
        {("link", "link_base")} if robot == "lite6" else set()
    )
    assert reference.keys() == stl.keys()


def test_stl_visual_override_keeps_the_stl_material(tmp_path: Path) -> None:
    """The G2 base collides as 10 hulls, so visual "stl" shows its full mesh (stl_visuals) in its stl_materials colour;
    the fingers still show their collision meshes."""
    path = Path(compose_robot_urdf("xarm6", "gripper_g2", visual="stl", out_dir=tmp_path))
    root = ET.parse(path).getroot()
    meshes = PROJECT_ROOT.resolve() / "assets/urdf/end_effectors/gripper_g2/meshes"

    def shown(visual: ET.Element) -> tuple[Path, str]:
        mesh = (path.parent / visual.find("geometry/mesh").get("filename")).resolve()
        return mesh, visual.find("material").get("name")

    base = root.find("./link[@name='xarm_gripper_base_link']")
    assert [shown(v) for v in base.findall("visual")] == [(meshes / "visual/base_link.stl", "White")]
    assert len(base.findall("collision")) == 10
    assert shown(root.find("./link[@name='left_finger']/visual")) == (meshes / "collision/left_finger.stl", "Silver")
    vacuum = ET.parse(compose_robot_urdf("xarm6", "vacuum_gripper", visual="stl", out_dir=tmp_path)).getroot()
    assert vacuum.find("./link[@name='xarm_vacuum_gripper_link']/visual/material") is None


@pytest.mark.parametrize(("robot", "end_effector", "visual"), _supported())
def test_every_supported_assembly_mounts_on_the_flange(robot: str, end_effector: str, visual: str) -> None:
    ee = end_effectors()[end_effector]
    root = ET.parse(compose_robot_urdf(robot, end_effector, visual=visual)).getroot()
    mount = root.find(f"./joint[@name='{ee.mount_joint}']")
    assert mount is not None and mount.get("type") == "fixed"
    assert mount.find("parent").get("link") == get_robot_profile(robot).ee_link
    assert mount.find("child").get("link") == ee.base_link
    names = [e.get("name") for e in root if e.tag in ("link", "joint")]
    assert len(names) == len(set(names))
    assert root.find("./link[@name='world']") is not None
    assert len([e for e in root if e.tag == "link"]) == len([e for e in root if e.tag == "joint"]) + 1


def test_flanges_partition_registry_and_define_compatibility() -> None:
    data = yaml.safe_load((PROJECT_ROOT / assembly.ASSEMBLY_YAML).read_text(encoding="utf-8"))
    flanges = {name: tuple(spec["robots"]) for name, spec in data["flanges"].items()}
    assert flanges == {"xarm": ("xarm5_1305", "xarm6_1305", "xarm7_1305", "uf850"), "lite6": ("lite6",)}
    assert sorted(r for robots in flanges.values() for r in robots) == sorted(ROBOT_PROFILES)
    for ee in end_effectors().values():
        assert ee.robots == flanges[ee.flange]
    identity = {n for n, ee in end_effectors().items() if ee.mount == assembly.IDENTITY}
    assert identity == {
        "gripper_g2",
        "vacuum_gripper",
        "robotiq_2f_85",
        "robotiq_2f_140",
        "lite6_gripper",
        "lite6_vacuum_gripper",
    }


def _write_schema(tmp_path: Path, flanges: str, flange: str, *, extra: str = "") -> Path:
    path = tmp_path / "assembly.yaml"
    path.write_text(
        f"schema_version: 4\nflanges:\n{flanges}end_effectors:\n  p:\n    source: s.urdf\n    base_link: b\n"
        f"    mount_joint: j\n    flange: {flange}\n{extra}",
        encoding="utf-8",
    )
    return path


def test_flange_schema_rejects_shared_robots_and_unknown_flanges(tmp_path: Path) -> None:
    shared = "  a:\n    robots: [xarm6]\n  b:\n    robots: [xarm6_1305]\n"
    with pytest.raises(AssemblyError, match="listed under flanges"):
        assembly._load_catalog(_write_schema(tmp_path, shared, "a"), 0)
    with pytest.raises(AssemblyError, match="unknown flange"):
        assembly._load_catalog(_write_schema(tmp_path, "  a:\n    robots: [lite6]\n", "c"), 1)


def test_mounting_plate_schema_rejects_shared_robots_invalid_thickness_and_clashes(tmp_path: Path) -> None:
    flanges = "  a:\n    robots: [xarm6, lite6]\n"
    shared_plate = (
        "mounting_plates:\n"
        "  p1:\n    source: p1.urdf\n    base_link: p1\n    mount_joint: j1\n"
        "    robots: [xarm6]\n    thickness_m: 0.008\n"
        "  p2:\n    source: p2.urdf\n    base_link: p2\n    mount_joint: j2\n"
        "    robots: [xarm6_1305]\n    thickness_m: 0.008\n"
    )
    with pytest.raises(AssemblyError, match="listed under mounting_plates"):
        assembly._load_catalog(_write_schema(tmp_path, flanges, "a", extra=shared_plate), 0)

    zero_thick = (
        "mounting_plates:\n"
        "  p1:\n    source: p1.urdf\n    base_link: p1\n    mount_joint: j1\n"
        "    robots: [lite6]\n    thickness_m: 0.0\n"
    )
    with pytest.raises(AssemblyError, match="thickness_m must be positive"):
        assembly._load_catalog(_write_schema(tmp_path, flanges, "a", extra=zero_thick), 1)

    name_clash = (
        "mounting_plates:\n"
        "  mounting_plate:\n    source: p1.urdf\n    base_link: p1\n    mount_joint: j1\n"
        "    robots: [lite6]\n    thickness_m: 0.005\n"
    )
    with pytest.raises(AssemblyError, match="part names must be unique"):
        assembly._load_catalog(_write_schema(tmp_path, flanges, "a", extra=name_clash), 2)


@pytest.mark.parametrize("end_effector", sorted(end_effectors()))
def test_mount_is_shared_by_every_robot_on_a_flange(end_effector: str) -> None:
    ee = end_effectors()[end_effector]
    origins = set()
    for robot in ee.robots:
        root = ET.parse(compose_robot_urdf(robot, end_effector))
        origin = root.getroot().find(f"./joint[@name='{ee.mount_joint}']/origin")
        origins.add((origin.get("xyz"), origin.get("rpy")))
    assert len(origins) == 1


def _pose(xyz: object, rpy: object) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rpy_matrix(*[float(v) for v in rpy])
    out[:3, 3] = [float(v) for v in xyz]
    return out


def _origin(elem: ET.Element | None) -> np.ndarray:
    o = elem.find("origin") if elem is not None else None
    return _pose(*(o.get(k, "0 0 0").split() for k in ("xyz", "rpy"))) if o is not None else np.eye(4)


def _flange_vertices(mesh_path: Path, flange_T_mesh: np.ndarray) -> np.ndarray:
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("DracoPy")
    vertices = np.asarray(trimesh.load(mesh_path, force="mesh").vertices)
    return vertices @ flange_T_mesh[:3, :3].T + flange_T_mesh[:3, 3]


def test_bio_gripper_g2_adapter_sits_on_the_xarm_flange() -> None:
    """Adapter face flush with the flange face; its M6 through-holes over the 135/225 deg holes (PCD 50 mm)."""
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("DracoPy")
    xyz, rpy = get_end_effector("bio_gripper_g2").mount
    # Flat faces cannot tilt; a tilted mount also breaks TCP-down IK on xArm5 (5-DOF) and xArm6 (wrist).
    assert rpy[:2] == (0.0, 0.0)
    mount = _pose(xyz, rpy)
    base = PROJECT_ROOT / "assets/urdf/end_effectors/bio_gripper_g2/meshes/visual/visual_glb/bio_gripper_g2_base.glb"
    mesh = trimesh.load(base, force="mesh")
    normals = mesh.face_normals @ mount[:3, :3].T
    centres = mesh.triangles_center @ mount[:3, :3].T + mount[:3, 3]
    face = (normals[:, 2] < -0.99) & (np.abs(centres[:, 2]) < 1e-3)
    assert mesh.area_faces[face].sum() > 3e-3
    assert np.average(centres[face, 2], weights=mesh.area_faces[face]) == pytest.approx(0.0, abs=2e-5)
    # Straight part of the two countersunk through-holes (R 3.13 mm, 0-1.1 mm into the plate).
    wall = _flange_vertices(base, mount)
    wall = wall[(wall[:, 2] > 0.0003) & (wall[:, 2] < 0.0008)]
    for sign in (1, -1):
        expected = np.array([-0.025, sign * 0.025]) * np.sqrt(0.5)
        xy = wall[np.linalg.norm(wall[:, :2] - expected, axis=1) < 0.004, :2]
        assert len(xy) > 50
        centre = np.linalg.lstsq(np.c_[2 * xy, np.ones(len(xy))], (xy**2).sum(1), rcond=None)[0][:2]
        np.testing.assert_allclose(centre, expected, atol=1e-4)


def test_bio_gripper_g2_fingertips_match_cad_reach() -> None:
    """CAD: flange axis to fingertip 157.5 mm along the finger direction (+X of the flange)."""
    path = Path(compose_robot_urdf("xarm6", "bio_gripper_g2"))
    root = ET.parse(path).getroot()
    flange_T_base = _origin(root.find("./joint[@name='bio_gripper_g2_attach']"))
    tips = []
    for side in ("left", "right"):
        joint = root.find(f"./joint[@name='bio_gripper_g2_{side}_finger_joint']")
        visual = root.find(f"./link[@name='bio_gripper_g2_{side}_finger']/visual")
        mesh = (path.parent / visual.find("geometry/mesh").get("filename")).resolve()
        tips.append(_flange_vertices(mesh, flange_T_base @ _origin(joint) @ _origin(visual))[:, 0].max())
    # Per-finger GLB placement differs by ~0.6 mm; the old per-robot mounts gave 161-163 mm.
    assert np.mean(tips) == pytest.approx(0.1575, abs=1e-3)
    assert abs(tips[0] - tips[1]) < 1e-3


def test_runtime_configs_resolve_assembled_models() -> None:
    store = RepositoryAssetStore.discover()
    expected = {
        "xarm5_1305": "assembly:gripper_g2",
        "xarm6_1305": "assembly:gripper_g2",
        "xarm7_1305": "assembly:gripper_g2",
        "uf850": "assembly:gripper_g2",
        "lite6": "assembly:lite6_gripper:stl",
    }
    for key, ref in expected.items():
        config = load_runtime_config(key)
        assert config.robot.urdf == ref
        path = resolve_robot_urdf(key, config.robot.urdf, assets_dir=config.robot.assets_dir, store=store)
        assert path == Path(robot_urdf(key, config.robot.urdf))
        assert path.is_file()
    reach = load_runtime_config("xarm6", task="reach")
    assert reach.robot.urdf == "xarm6_1305.urdf"
    assert resolve_robot_urdf("xarm6", reach.robot.urdf, assets_dir=reach.robot.assets_dir, store=store) == Path(
        robot_urdf("xarm6")
    )


def test_validate_robot_urdf_never_composes(monkeypatch: pytest.MonkeyPatch) -> None:
    store = RepositoryAssetStore.discover()

    def _fail(*args: object, **kwargs: object) -> str:
        raise AssertionError("validation must not compose")

    monkeypatch.setattr(assembly, "compose_robot_urdf", _fail)
    validate_robot_urdf("xarm6", "assembly:gripper_g2", assets_dir="assets/urdf/robots/xarm6", store=store)
    validate_robot_urdf("xarm6", "xarm6_1305.urdf", assets_dir="assets/urdf/robots/xarm6", store=store)
    with pytest.raises(AssemblyError):
        validate_robot_urdf("lite6", "assembly:gripper_g2", assets_dir="assets/urdf/robots/lite6", store=store)


def test_visual_glb_wrapper_maps_flags_to_assemblies() -> None:
    assert Path(robot_visual_glb_urdf("xarm6")).name == "xarm6_1305_visual.glb.urdf"
    cases = {
        "gripper_g2": (
            dict(with_gripper_g2=True, movable=True),
            dict(with_gripper_g2=True),
        ),
        "bio_gripper_g2": (
            dict(with_bio_gripper_g2=True, movable=True),
            dict(with_bio_gripper_g2=True),
        ),
        "robotiq_2f_85": (
            dict(with_robotiq_2f_85=True, movable=True),
            dict(with_robotiq_2f_85=True),
        ),
        "robotiq_2f_140": (
            dict(with_robotiq_2f_140=True, movable=True),
            dict(with_robotiq_2f_140=True),
        ),
        "vacuum_gripper": (dict(with_vacuum_gripper=True),),
    }
    for end_effector, flag_sets in cases.items():
        expected = compose_robot_urdf("xarm6", end_effector)
        for flags in flag_sets:
            assert robot_visual_glb_urdf("xarm6", **flags) == expected


def test_compose_standalone_end_effector_urdf(tmp_path: Path) -> None:
    from ufactory.robots.paths import lite6_vacuum_gripper_visual_glb_urdf

    out = Path(lite6_vacuum_gripper_visual_glb_urdf())
    assert out.is_file()
    assert out == Path(assembly.compose_standalone_end_effector_urdf("lite6_vacuum_gripper", visual="glb"))

    composed = Path(
        assembly.compose_standalone_end_effector_urdf("lite6_vacuum_gripper", visual="glb", out_dir=tmp_path)
    )
    assert composed.parent == tmp_path
    root = ET.parse(composed).getroot()
    assert [link.get("name") for link in root.findall("link")][:2] == ["world", "uflite_vacuum_gripper_link"]
    fix = root.find("./joint[@name='vacuum_gripper_fix']")
    assert fix is not None
    assert fix.find("parent").get("link") == "world"
    assert fix.find("child").get("link") == "uflite_vacuum_gripper_link"
    vis = root.find("./link[@name='uflite_vacuum_gripper_link']/visual")
    assert vis is not None
    assert vis.find("material") is None
    mesh_attr = vis.find("geometry/mesh").get("filename")
    assert mesh_attr is not None and not os.path.isabs(mesh_attr)
    assert Path(mesh_attr).name == "lite6_vacuum_gripper_visual_link6.glb"
    assert (composed.parent / mesh_attr).is_file()

    vac = Path(assembly.compose_standalone_end_effector_urdf("vacuum_gripper", out_dir=tmp_path))
    vac_root = ET.parse(vac).getroot()
    vac_fix = vac_root.find("./joint[@name='vacuum_gripper_fix']")
    assert vac_fix is not None and vac_fix.find("parent").get("link") == "world"
    assert vac_fix.find("child").get("link") == "xarm_vacuum_gripper_link"
    for mesh in vac_root.findall(".//mesh"):
        fn = mesh.get("filename")
        assert fn is not None and (vac.parent / fn).is_file()

    with pytest.raises(AssemblyError):
        assembly.compose_standalone_end_effector_urdf("lite6_vacuum_gripper", visual="obj", out_dir=tmp_path)
