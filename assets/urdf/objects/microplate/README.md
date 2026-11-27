# SBS 96-Well Microplate

Standard ANSI/SLAS 96-well microplate (`127.76 × 85.48 × 14.35 mm`, `0.100 kg`) used by the Bio Gripper G2 pick-and-place simulation (`ufactory.simulation.bio_g2_microplate`).

## Files

- `SBS_96_well_plate.json`: parametric labware definition from upstream [ramonaoptics/labware](https://github.com/ramonaoptics/labware/tree/b2de87e717584b2e27f436ee03c17b827085e87b) (`b2de87e`).
- `sbs_96_well.glb`: visual mesh.
- `sbs_96_well_convex.stl`: watertight convex collision box (`127.76 × 85.48 × 14.35 mm`).
- `sbs_96_well.urdf`: Genesis URDF with the GLB visual, the convex collision and a solid-box inertia for `0.100 kg` (`PLATE_MASS_KG`).
- `LICENSE.txt` / `manifest.json`: upstream BSD-3-Clause license and source checksums.

`build_microplate()` in `ufactory/simulation/bio_g2_microplate.py` generates the GLB, STL and URDF from the JSON when any of them is missing.
