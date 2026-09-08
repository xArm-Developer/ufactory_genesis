"""Version and private-hook contract tests for the Genesis 1.4.0 baseline."""

from __future__ import annotations

from types import SimpleNamespace
import warnings

import pytest

import ufactory.simulation.compat as compat


@pytest.fixture(autouse=True)
def reset_unvalidated_warning(monkeypatch):
    monkeypatch.setattr(compat, "_WARNED_UNVALIDATED", False)


def test_version_below_minimum_is_rejected(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _name: "1.3.3")

    with pytest.raises(compat.GenesisCompatibilityError, match=r"Genesis>=1\.4\.0 is required"):
        compat.require_genesis_version()


def test_validated_version_is_accepted_without_warning(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _name: "1.4.0")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert compat.require_genesis_version() == compat.VALIDATED_GENESIS_VERSION
    assert caught == []


def test_older_version_is_rejected(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _name: "1.3.3")

    with pytest.raises(compat.GenesisCompatibilityError, match=r"Genesis>=1\.4\.0 is required"):
        compat.require_genesis_version()


def test_newer_version_warns_only_once(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _name: "1.4.1")

    with pytest.warns(RuntimeWarning, match="only 1.4.0 is the project's reference baseline") as caught:
        compat.require_genesis_version()
        compat.require_genesis_version()
    assert len(caught) == 1


def test_newer_version_with_complete_capabilities_passes_and_warns(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _name: "1.4.1")
    gs = pytest.importorskip("genesis")

    with pytest.warns(RuntimeWarning, match="only 1.4.0 is the project's reference baseline") as caught:
        assert compat.require_genesis_capabilities(gs, pbr=True, deferred_viewer=True) is gs
    assert len(caught) == 1


def test_invalid_version_is_rejected(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _name: "not-a-version")

    with pytest.raises(compat.GenesisCompatibilityError, match="Cannot parse"):
        compat.require_genesis_version()


def _parse_mesh_glb(path, group_by_material, scale, is_mesh_zup, surface):
    return path, group_by_material, scale, is_mesh_zup, surface


def _surface_uvs_to_trimesh_visual(surface, uvs=None, n_verts=None):
    return surface, uvs, n_verts


class _CompatibleMesh:
    @classmethod
    def from_trimesh(
        cls,
        mesh,
        scale=None,
        convexify=False,
        decimate=False,
        decimate_face_num=500,
        decimate_aggressiveness=2,
        metadata=None,
        surface=None,
        is_mesh_zup=True,
    ):
        return (
            cls,
            mesh,
            scale,
            convexify,
            decimate,
            decimate_face_num,
            decimate_aggressiveness,
            metadata,
            surface,
            is_mesh_zup,
        )


def test_pbr_hook_contract_accepts_required_signatures(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _name: "1.4.0")
    gs = SimpleNamespace(Mesh=_CompatibleMesh)
    gltf = SimpleNamespace(parse_mesh_glb=_parse_mesh_glb)
    mesh = SimpleNamespace(surface_uvs_to_trimesh_visual=_surface_uvs_to_trimesh_visual)

    compat.require_pbr_hooks(gs, gltf, mesh)


def test_pbr_hook_contract_rejects_changed_signature_before_patch(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _name: "1.4.1")
    gs = SimpleNamespace(Mesh=_CompatibleMesh)
    gltf = SimpleNamespace(parse_mesh_glb=lambda path: path)
    mesh = SimpleNamespace(surface_uvs_to_trimesh_visual=_surface_uvs_to_trimesh_visual)

    with pytest.warns(RuntimeWarning):
        with pytest.raises(compat.GenesisCompatibilityError, match="missing parameters"):
            compat.require_pbr_hooks(gs, gltf, mesh)


def test_viewer_contract_rejects_missing_private_registry(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _name: "1.4.0")

    with pytest.raises(compat.GenesisCompatibilityError, match="_scene_registry"):
        compat.load_deferred_viewer_api(SimpleNamespace())


def test_ik_scratch_is_noop_on_140(monkeypatch):
    """Genesis 1.4.0 allocates IK scratch lazily, so ensure_ik_scratch is a no-op."""
    monkeypatch.setattr(compat.metadata, "version", lambda _name: "1.4.0")
    robot = SimpleNamespace(_IK_qpos_orig=None, n_qs=6, _solver=SimpleNamespace())

    compat.ensure_ik_scratch(robot)
    assert robot._IK_qpos_orig is None


def test_installed_genesis_matches_runtime_and_hook_contracts():
    gs = pytest.importorskip("genesis")
    gltf_utils = pytest.importorskip("genesis.utils.gltf")
    mesh_utils = pytest.importorskip("genesis.utils.mesh")
    try:
        version = compat.require_genesis_version()
    except compat.GenesisCompatibilityError as exc:
        pytest.skip(f"genesis-world metadata unavailable: {exc}")
    if str(version) != "1.4.0":
        pytest.skip("the full installed-contract assertion targets the reference 1.4.0 baseline")

    assert compat.require_genesis_runtime(gs) is gs
    compat.require_pbr_hooks(gs, gltf_utils, mesh_utils)
    viewer = compat.load_deferred_viewer_api(gs)
    assert viewer.default_aspect_ratio > 0.0
    assert viewer.default_height_ratio > 0.0
