"""AI1500 six-axis F/T sensor asset, assembly, Pinocchio FK/collision, and Genesis smoke tests."""

from __future__ import annotations

import itertools
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from ufactory.robots import assembly
from ufactory.robots.assembly import (
    AssemblyError,
    AssemblyRef,
    compose_robot_urdf,
    compose_standalone_ft_sensor_urdf,
    ft_sensors,
    get_ft_sensor,
    parse_assembly_ref,
    resolve_robot_urdf,
    supported_ft_sensors,
    supports_ft_sensor,
    validate_robot_urdf,
)
from ufactory.robots.paths import ft_sensor_ai1500_urdf, robot_urdf, robot_visual_glb_urdf
from ufactory.robots.registry import PROJECT_ROOT, get_robot_profile
from ufactory.config import RepositoryAssetStore

URDF_ROOT = PROJECT_ROOT / "assets" / "urdf"
SENSOR_DIR = URDF_ROOT / "accessories" / "ft_sensor_ai1500"
XARM_ARMS = ("xarm5_1305", "xarm6_1305", "xarm7_1305", "uf850")
SENSOR_LIFT_M = 0.0586
STAND_LIFT_M = 0.003
ADAPTER_LIFT_M = 0.011
SENSOR_MASS_KG = 0.6
SENSOR_COM_M = np.array([0.0, 0.0, 0.022], dtype=np.float64)
SENSOR_INERTIA_DIAG = (0.000428064, 0.000428064, 0.000451632)


def test_catalog_registers_ft_sensor_ai1500_for_xarm_and_uf850_only() -> None:
    sensors = ft_sensors()
    assert tuple(sensors) == ("ft_sensor_ai1500",)
    ft = get_ft_sensor("ft_sensor_ai1500")
    assert ft == sensors["ft_sensor_ai1500"]
    assert ft.flange == "xarm"
    assert ft.robots == XARM_ARMS
    assert ft.output_link == "ft_sensor_ai1500_output"
    assert ft.tool_lift_m == pytest.approx(SENSOR_LIFT_M)

    for arm in XARM_ARMS:
        assert supported_ft_sensors(arm) == ("ft_sensor_ai1500",)
        assert supports_ft_sensor(arm, "ft_sensor_ai1500")
        assert get_robot_profile(arm).supports_ft_sensor_ai1500

    assert supported_ft_sensors("lite6") == ()
    assert not supports_ft_sensor("lite6", "ft_sensor_ai1500")
    assert not get_robot_profile("lite6").supports_ft_sensor_ai1500

    with pytest.raises(AssemblyError, match="unknown F/T sensor"):
        get_ft_sensor("ft_sensor_missing")


def test_ft_sensor_schema_validation_rejects_invalid_configs(tmp_path: Path) -> None:
    def write_schema(body: str) -> Path:
        p = tmp_path / "assembly.yaml"
        p.write_text(f"schema_version: 6\nflanges:\n  xarm:\n    robots: [xarm6]\n{body}", encoding="utf-8")
        return p

    bad_prefix = "ft_sensors:\n  ai1500:\n    source: s.urdf\n    base_link: b\n    output_link: o\n    mount_joint: j\n    flange: xarm\n"
    with pytest.raises(AssemblyError, match="must start with 'ft_sensor_'"):
        assembly._load_catalog(write_schema(bad_prefix), 1)

    bad_lift = (
        "ft_sensors:\n"
        "  ft_sensor_x:\n    source: s.urdf\n    base_link: b\n    output_link: o\n"
        "    mount_joint: j\n    flange: xarm\n    tool_lift_m: -0.01\n"
    )
    with pytest.raises(AssemblyError, match="tool_lift_m must be non-negative"):
        assembly._load_catalog(write_schema(bad_lift), 2)

    missing_out = (
        "ft_sensors:\n"
        "  ft_sensor_x:\n    source: s.urdf\n    base_link: b\n    output_link: ''\n"
        "    mount_joint: j\n    flange: xarm\n"
    )
    with pytest.raises(AssemblyError, match="output_link is required"):
        assembly._load_catalog(write_schema(missing_out), 3)


def test_ft_sensor_ai1500_assets_and_urdf_integrity() -> None:
    trimesh = pytest.importorskip("trimesh")
    pygltflib = pytest.importorskip("pygltflib")
    DracoPy = pytest.importorskip("DracoPy")

    urdf_path = SENSOR_DIR / "ft_sensor_ai1500.urdf"
    glb_path = SENSOR_DIR / "meshes" / "visual" / "ft_sensor_ai1500.glb"
    stl_path = SENSOR_DIR / "meshes" / "visual" / "ft_sensor_ai1500.stl"
    col_path = SENSOR_DIR / "meshes" / "collision" / "ft_sensor_ai1500_col_00.stl"

    for path in (urdf_path, glb_path, stl_path, col_path):
        assert path.is_file(), path

    assert glb_path.stat().st_size < 1_000_000
    assert glb_path.read_bytes()[:4] == b"glTF"
    glb = pygltflib.GLTF2().load(str(glb_path))
    assert "KHR_draco_mesh_compression" in (glb.extensionsUsed or [])
    assert len(glb.materials) == 5
    blob = glb.binary_blob() or b""
    decoded_chunks = []
    for mesh in glb.meshes:
        for prim in mesh.primitives:
            view = glb.bufferViews[prim.extensions["KHR_draco_mesh_compression"]["bufferView"]]
            dec = DracoPy.decode(blob[view.byteOffset : view.byteOffset + view.byteLength])
            decoded_chunks.append(np.asarray(dec.points, dtype=np.float64))
    glb_pts = np.vstack(decoded_chunks)
    assert float(glb_pts[:, 2].min() * 1000.0) == pytest.approx(-5.0, abs=0.05)
    assert float(glb_pts[:, 2].max() * 1000.0) == pytest.approx(58.6, abs=0.05)

    col_mesh = trimesh.load(col_path, force="mesh")
    assert col_mesh.is_watertight and col_mesh.is_convex
    assert len(col_mesh.vertices) == 48
    assert len(col_mesh.faces) == 92

    root = ET.parse(urdf_path).getroot()
    inertial = root.find("./link[@name='ft_sensor_ai1500']/inertial")
    assert inertial is not None
    assert float(inertial.find("mass").attrib["value"]) == pytest.approx(SENSOR_MASS_KG)
    com = np.array([float(x) for x in inertial.find("origin").attrib["xyz"].split()])
    np.testing.assert_allclose(com, SENSOR_COM_M, atol=1e-9)
    inert = inertial.find("inertia").attrib
    assert float(inert["ixx"]) == pytest.approx(SENSOR_INERTIA_DIAG[0], abs=1e-12)
    assert float(inert["iyy"]) == pytest.approx(SENSOR_INERTIA_DIAG[1], abs=1e-12)
    assert float(inert["izz"]) == pytest.approx(SENSOR_INERTIA_DIAG[2], abs=1e-12)

    for frame_name in ("ft_sensor_ai1500_output", "ft_sensor_ai1500_measurement"):
        j = root.find(f"./joint[@name='{frame_name}_joint']")
        assert j is not None
        xyz = np.array([float(x) for x in j.find("origin").attrib["xyz"].split()])
        np.testing.assert_allclose(xyz, [0.0, 0.0, SENSOR_LIFT_M], atol=1e-9)

    standalone = Path(ft_sensor_ai1500_urdf())
    assert standalone == Path(compose_standalone_ft_sensor_urdf("ft_sensor_ai1500", visual="glb"))
    assert standalone.is_file()


def test_assembly_ref_round_trip_and_validation() -> None:
    cases = [
        ("assembly:bare+ft_sensor_ai1500", AssemblyRef(None, "glb", None, False, "ft_sensor_ai1500")),
        ("assembly:gripper_g2+ft_sensor_ai1500", AssemblyRef("gripper_g2", "glb", None, False, "ft_sensor_ai1500")),
        (
            "assembly:gripper_g2+ft_sensor_ai1500+d435+mounting_plate:stl",
            AssemblyRef("gripper_g2", "stl", "d435", True, "ft_sensor_ai1500"),
        ),
        (
            "assembly:bare+ft_sensor_ai1500+camera_stand+mounting_plate",
            AssemblyRef(None, "glb", "camera_stand", True, "ft_sensor_ai1500"),
        ),
    ]
    for text, expected in cases:
        parsed = parse_assembly_ref(text)
        assert parsed == expected
        assert str(parsed) == text

    for bad in (
        "assembly:ft_sensor_ai1500",
        "assembly:bare+ft_sensor_ai1500+ft_sensor_other",
        "assembly:bare+d435+camera_stand+ft_sensor_ai1500",
    ):
        with pytest.raises(AssemblyError):
            parse_assembly_ref(bad)

    with pytest.raises(AssemblyError, match="not supported on lite6"):
        compose_robot_urdf("lite6", None, ft_sensor="ft_sensor_ai1500")
    with pytest.raises(ValueError, match="does not support F/T sensor"):
        robot_visual_glb_urdf("lite6", ft_sensor="ft_sensor_ai1500")
    with pytest.raises(AssemblyError, match="camera_stand does not fit with bio_gripper_g2"):
        compose_robot_urdf("xarm6_1305", "bio_gripper_g2", accessory="camera_stand", ft_sensor="ft_sensor_ai1500")

    # Resolving via paths & assembly store works without error.
    store = RepositoryAssetStore.discover(PROJECT_ROOT)
    validate_robot_urdf(
        "xarm6_1305",
        "assembly:gripper_g2+ft_sensor_ai1500+d435",
        assets_dir=URDF_ROOT / "robots" / "xarm6",
        store=store,
    )
    resolved = resolve_robot_urdf("xarm6_1305", "assembly:bare+ft_sensor_ai1500")
    assert resolved.is_file()
    assert Path(robot_urdf("xarm6_1305", "assembly:gripper_g2+ft_sensor_ai1500")).is_file()


def test_backward_compatibility_without_ft_sensor() -> None:
    baseline = Path(compose_robot_urdf("xarm6_1305", "gripper_g2", accessory="d435", mounting_plate=True))
    assert "ft_sensor" not in baseline.name
    root = ET.parse(baseline).getroot()
    assert root.find("./link[@name='ft_sensor_ai1500']") is None


def test_pinocchio_cumulative_lift_mass_and_internal_clearance() -> None:
    pin = pytest.importorskip("pinocchio")
    pytest.importorskip("coal")
    from ufactory.safety.adapters.pinocchio import PinocchioCollisionBackend, urdf_subtree_links

    def _build_pin_model(urdf_path: str):
        try:
            return pin.buildModelFromUrdf(urdf_path, mimic=True)
        except ValueError as exc:
            if "mimic" not in str(exc).lower():
                raise
            return pin.buildModelFromUrdf(urdf_path)

    def link_in_flange_z(urdf_path: str, flange_link: str, target_link: str) -> tuple[float, float]:
        model = _build_pin_model(urdf_path)
        data = model.createData()
        pin.framesForwardKinematics(model, data, pin.neutral(model))
        flange_id = model.getFrameId(flange_link)
        target_id = model.getFrameId(target_link)
        rel = data.oMf[flange_id].actInv(data.oMf[target_id])
        total_mass = sum(float(inertia.mass) for inertia in model.inertias)
        return float(rel.translation[2]), total_mass

    for robot_key in XARM_ARMS:
        profile = get_robot_profile(robot_key)
        bare_urdf = str(profile.assets_dir / profile.default_urdf)
        bare_model = pin.buildModelFromUrdf(bare_urdf, mimic=True)
        m_bare = sum(float(inertia.mass) for inertia in bare_model.inertias)

        # 1. Bare + FT sensor -> output frame at +58.6 mm in flange +Z, mass +0.6 kg.
        ft_bare = compose_robot_urdf(robot_key, None, ft_sensor="ft_sensor_ai1500", visual="stl")
        z_out, m_ft_bare = link_in_flange_z(ft_bare, profile.ee_link, "ft_sensor_ai1500_output")
        assert z_out == pytest.approx(SENSOR_LIFT_M, abs=1e-6)
        assert m_ft_bare - m_bare == pytest.approx(SENSOR_MASS_KG, abs=1e-6)

        # 2. If the arm defines link_eef (e.g. xarm6_1305), link_eef lifts +58.6 mm / +61.6 mm.
        if bare_model.existFrame("link_eef"):
            z_bare, _ = link_in_flange_z(bare_urdf, profile.ee_link, "link_eef")
            z_ft_eef, _ = link_in_flange_z(ft_bare, profile.ee_link, "link_eef")
            assert z_ft_eef - z_bare == pytest.approx(SENSOR_LIFT_M, abs=1e-6)

            ft_cam_bare = compose_robot_urdf(
                robot_key, None, accessory="camera_stand", ft_sensor="ft_sensor_ai1500", visual="stl"
            )
            z_ft_cam_eef, _ = link_in_flange_z(ft_cam_bare, profile.ee_link, "link_eef")
            assert z_ft_cam_eef - z_bare == pytest.approx(SENSOR_LIFT_M + STAND_LIFT_M, abs=1e-6)

    # 3. Downstream end effectors stack onto ft_sensor_ai1500_output cleanly.
    combos: list[tuple[str, str | None, str, float, float | None]] = [
        ("gripper_g2", None, "xarm_gripper_base_link", SENSOR_LIFT_M, SENSOR_LIFT_M),
        ("gripper_g2", "d435", "xarm_gripper_base_link", SENSOR_LIFT_M + STAND_LIFT_M, SENSOR_LIFT_M + STAND_LIFT_M),
        ("bio_gripper_g2", None, "bio_gripper_g2_base_link", SENSOR_LIFT_M, None),
        (
            "vacuum_gripper",
            "d405",
            "xarm_vacuum_gripper_link",
            SENSOR_LIFT_M + STAND_LIFT_M,
            SENSOR_LIFT_M + STAND_LIFT_M,
        ),
        ("robotiq_2f_85", None, "robotiq_2f_85_base_link", SENSOR_LIFT_M, SENSOR_LIFT_M + ADAPTER_LIFT_M),
        (
            "robotiq_2f_140",
            "zed_x_nano",
            "robotiq_2f_140_base_link",
            SENSOR_LIFT_M + STAND_LIFT_M,
            SENSOR_LIFT_M + STAND_LIFT_M + ADAPTER_LIFT_M,
        ),
    ]
    for ee, acc, link_name, expected_delta, expected_abs_z in combos:
        base_urdf = compose_robot_urdf("xarm6_1305", ee, visual="stl")
        ft_urdf = compose_robot_urdf("xarm6_1305", ee, accessory=acc, ft_sensor="ft_sensor_ai1500", visual="stl")
        z_no_ft, _ = link_in_flange_z(base_urdf, "link6", link_name)
        z_with_ft, _ = link_in_flange_z(ft_urdf, "link6", link_name)
        assert z_with_ft - z_no_ft == pytest.approx(expected_delta, abs=1e-6)
        if expected_abs_z is not None:
            assert z_with_ft == pytest.approx(expected_abs_z, abs=1e-6)

    # 4. Flange-subtree internal clearance sweep with ft_sensor_ai1500 (+ optional d435).
    for ee, acc in itertools.product(
        ("gripper_g2", "vacuum_gripper", "robotiq_2f_85", "robotiq_2f_140"), (None, "d435")
    ):
        urdf = Path(compose_robot_urdf("xarm6_1305", ee, accessory=acc, ft_sensor="ft_sensor_ai1500", visual="stl"))
        root_xml = ET.parse(urdf).getroot()
        arm_joints = tuple(f"joint{i}" for i in range(1, 7))
        drives = [
            j
            for j in root_xml.findall("joint")
            if j.get("type") in {"revolute", "prismatic"}
            and j.find("mimic") is None
            and j.get("name") not in arm_joints
        ]
        passive = {}
        values: list[float | None] = [None]
        if drives:
            lim = drives[0].find("limit")
            assert lim is not None
            lo, hi = float(lim.attrib["lower"]), float(lim.attrib["upper"])
            passive = {str(drives[0].attrib["name"]): lo}
            values = list(np.linspace(lo, hi, 11))
        backend = PinocchioCollisionBackend(
            urdf, joint_names=arm_joints, ee_link="link6", passive_joint_positions=passive
        )
        tool = urdf_subtree_links(root_xml, "link6")
        for pair in list(backend.geometry_model.collisionPairs):
            names = {backend._link_name(pair.first), backend._link_name(pair.second)}
            if not names <= tool:
                backend.geometry_model.removeCollisionPair(pin.CollisionPair(int(pair.first), int(pair.second)))
        backend.geometry_data = backend.geometry_model.createData()
        q_arm = np.zeros(6)
        for val in values:
            for res in backend.check_all(q_arm, stage="sweep", gripper_drive=val):
                if res.colliding:
                    # Only expected four-bar pin / closed-finger pairs inside the gripper; never ft_sensor_ai1500.
                    assert "ft_sensor_ai1500" not in (res.link_a, res.link_b)


@pytest.mark.gpu
def test_genesis_loads_ft_sensor_standalone_and_assembled() -> None:
    gs = pytest.importorskip("genesis")
    from ufactory.config import load_runtime_config
    from ufactory.simulation import GenesisRuntimeManager, make_rigid_options
    from ufactory.visualization.glb import glb_pbr_surfaces, glb_view_surface

    config = load_runtime_config("xarm6_1305")
    with GenesisRuntimeManager(config.simulation):
        scene = gs.Scene(
            show_viewer=False,
            sim_options=gs.options.SimOptions(dt=0.02, substeps=8),
            rigid_options=make_rigid_options(gs),
        )
        with glb_pbr_surfaces():
            sensor_ent = scene.add_entity(
                gs.morphs.URDF(file=ft_sensor_ai1500_urdf(), fixed=True, pos=(0.5, 0.0, 0.1)),
                surface=glb_view_surface(),
            )
            arm_urdf = robot_visual_glb_urdf(
                "xarm6_1305",
                with_gripper_g2=True,
                accessory="d435",
                ft_sensor="ft_sensor_ai1500",
            )
            arm_ent = scene.add_entity(
                gs.morphs.URDF(file=arm_urdf, fixed=True),
                surface=glb_view_surface(),
            )
        scene.build()
        for _ in range(5):
            scene.step()
        assert sensor_ent.n_links >= 1
        assert arm_ent.n_links >= 7
