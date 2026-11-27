"""Offline SBS 96-well microplate model and builder for Bio Gripper G2 pick-and-place."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from ufactory.robots.registry import PROJECT_ROOT
from ufactory.simulation.bio_g2_model import sha256

ASSET_DIR = PROJECT_ROOT / "assets" / "urdf" / "objects" / "microplate"
PLATE_JSON_PATH = ASSET_DIR / "SBS_96_well_plate.json"
PLATE_URDF = ASSET_DIR / "sbs_96_well.urdf"
PLATE_MASS_KG = 0.100


def plate_definition() -> dict:
    return json.loads(PLATE_JSON_PATH.read_text(encoding="utf-8"))


def plate_size_m() -> tuple[float, float, float]:
    outer = plate_definition()["outer_plate_dimensions"]
    return tuple(float(outer[key]) for key in ("length", "width", "height"))


def _box(extents, center):
    import trimesh

    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def _deck_cell(pitch: float, radius: float, top: float, thickness: float):
    """Closed square annulus, triangulated without optional polygon dependencies."""
    import trimesh

    angles = np.linspace(0, 2 * np.pi, 33)[:-1]
    directions = np.c_[np.cos(angles), np.sin(angles)]
    square = directions * ((pitch / 2) / np.abs(directions).max(axis=1))[:, None]
    rings = [np.c_[xy, np.full(len(angles), z)] for z in (top, top - thickness) for xy in (square, radius * directions)]
    vertices = np.vstack(rings)
    n = len(angles)
    faces = []
    for first, second in ((0, n), (n, 3 * n), (3 * n, 2 * n), (2 * n, 0)):
        for idx in range(n):
            nxt = (idx + 1) % n
            faces += [
                (first + idx, first + nxt, second + nxt),
                (first + idx, second + nxt, second + idx),
            ]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.fix_normals()
    return mesh


def build_microplate(output_dir: str | Path | None = None, *, export_stl: bool = False) -> dict:
    """Return or generate local GLB/STL/URDF from the ANSI/SLAS 96-well dimensions."""
    target_dir = ASSET_DIR if output_dir is None else Path(output_dir)
    definition = plate_definition()
    length, width, height = size = plate_size_m()
    wells = definition["well_dimensions"]
    rows, columns, pitch = wells["rows"], wells["columns"], wells["pitch"]
    radius, depth = wells["diameter"] / 2, wells["depth"]
    wall, bottom, skirt_height, skirt_wall = 0.001, 0.001, 0.0061, 0.00127
    inertia = PLATE_MASS_KG / 12 * np.array([width**2 + height**2, length**2 + height**2, length**2 + width**2])
    urdf_path = target_dir / "sbs_96_well.urdf"
    need_generate = output_dir is not None or not all(
        (target_dir / name).exists() for name in ("sbs_96_well.glb", "sbs_96_well_convex.stl", "sbs_96_well.urdf")
    )
    if need_generate:
        import trimesh

        top = height / 2
        parts = []
        cell = _deck_cell(pitch, radius, top, wall)
        bowl_z = top - depth + radius
        angles = np.linspace(0, np.pi / 2, 13)
        inner = np.c_[radius * np.cos(angles), bowl_z - radius * np.sin(angles)]
        outer_radius = radius + bottom
        outer = np.c_[outer_radius * np.cos(angles[::-1]), bowl_z - outer_radius * np.sin(angles[::-1])]
        profile = np.vstack(([[radius, top - wall]], inner, outer, [[outer_radius, top - wall], [radius, top - wall]]))
        profile[np.abs(profile[:, 0]) < 1e-12, 0] = 0
        well = trimesh.creation.revolve(profile, sections=32)
        reference = definition["alignment_reference"]
        for row in range(rows):
            for column in range(columns):
                center = [
                    -length / 2 + reference["offset_from_left_edge"] + column * pitch,
                    width / 2 - reference["offset_from_top_edge"] - row * pitch,
                    0,
                ]
                for prototype in (cell, well):
                    part = prototype.copy()
                    part.apply_translation(center)
                    parts.append(part)
        grid_x, grid_y = columns * pitch, rows * pitch
        for sign in (-1, 1):
            parts += [
                _box(((length - grid_x) / 2, width, wall), (sign * (length + grid_x) / 4, 0, top - wall / 2)),
                _box((grid_x, (width - grid_y) / 2, wall), (0, sign * (width + grid_y) / 4, top - wall / 2)),
            ]
        for thickness, z_low, z_high in (
            (wall, -height / 2, height / 2),
            (skirt_wall, -height / 2, -height / 2 + skirt_height),
        ):
            for sign in (-1, 1):
                parts += [
                    _box(
                        (length, thickness, z_high - z_low), (0, sign * (width - thickness) / 2, (z_low + z_high) / 2)
                    ),
                    _box(
                        (thickness, width - 2 * thickness, z_high - z_low),
                        (sign * (length - thickness) / 2, 0, (z_low + z_high) / 2),
                    ),
                ]
        visual = trimesh.util.concatenate(parts)
        visual.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                name="SBS 96 well polypropylene",
                baseColorFactor=[125, 181, 212, 255],
                metallicFactor=0.0,
                roughnessFactor=0.45,
                doubleSided=True,
            )
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        visual.export(target_dir / "sbs_96_well.glb")
        if export_stl:
            visual.export(target_dir / "sbs_96_well.stl")
        collision = trimesh.creation.box(extents=size)
        collision.export(target_dir / "sbs_96_well_convex.stl")
        robot = ET.Element("robot", name="sbs_96_well")
        link = ET.SubElement(robot, "link", name="microplate")
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=str(PLATE_MASS_KG))
        ET.SubElement(
            inertial,
            "inertia",
            ixx=str(inertia[0]),
            iyy=str(inertia[1]),
            izz=str(inertia[2]),
            ixy="0",
            ixz="0",
            iyz="0",
        )
        for tag, filename in (("visual", "sbs_96_well.glb"), ("collision", "sbs_96_well_convex.stl")):
            node = ET.SubElement(link, tag)
            ET.SubElement(ET.SubElement(node, "geometry"), "mesh", filename=filename)
        ET.indent(robot)
        ET.ElementTree(robot).write(urdf_path, encoding="utf-8", xml_declaration=True)
    tracked_names = ["sbs_96_well.glb", "sbs_96_well_convex.stl", urdf_path.name]
    if export_stl and (target_dir / "sbs_96_well.stl").exists():
        tracked_names.insert(1, "sbs_96_well.stl")
    return {
        "source": {
            "license": "BSD-3-Clause",
            "definition_sha256": sha256(PLATE_JSON_PATH),
        },
        "size_m": size,
        "rows": rows,
        "columns": columns,
        "pitch_m": pitch,
        "well_diameter_m": 2 * radius,
        "well_depth_m": depth,
        "assumptions": {
            "wall_m": wall,
            "bottom_m": bottom,
            "skirt_height_m": skirt_height,
            "skirt_wall_m": skirt_wall,
            "well_bottom": "hemisphere (U)",
            "corners": "square; no vendor-specific chamfers",
            "mesh": "overlapping closed shells",
        },
        "mass_kg": PLATE_MASS_KG,
        "inertia_diagonal_kg_m2": inertia.tolist(),
        "sha256": {name: sha256(target_dir / name) for name in tracked_names},
    }
