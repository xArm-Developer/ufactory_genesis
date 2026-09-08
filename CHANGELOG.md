# Changelog

All notable changes to genesis-ufactory will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **BREAKING:** Raised the minimum and validated Genesis World baseline from **1.3.3 to 1.4.0**. The compatibility gate and all version pins now target 1.4.0.
- **BREAKING:** Adapted to Genesis 1.4.0 API changes:
  - `RigidEntity.forward_kinematics()` was removed upstream. Kinematics queries now use `set_qpos()` followed by `get_links_pos()` / `get_links_quat()`.
  - `RigidEntity.set_links_inertial_mass()` was renamed to `set_links_mass()`. All call sites updated.
  - IK scratch fields (`_IK_qpos_orig` etc.) are now allocated lazily inside the solver; `ensure_ik_scratch()` is retained as a no-op for API compatibility.
- The compatibility check in `require_genesis_runtime()` no longer validates `forward_kinematics`; it validates `get_links_pos`, `get_links_quat`, and `set_links_mass` instead.

### Notes

- **GPU simulation and RL policy revalidation were not performed.** The bundled `model_299_g2stable.pt` policy was trained under the `g2_stable_v1_3_3` contact profile on Genesis 1.3.3. It must be revalidated or retrained before its v0.2.13 evaluation results can be trusted on Genesis 1.4.0. Physics behavior may differ due to upstream solver changes.
- The following fast checks pass without Genesis/GPU: `ruff check`, `mypy` (domain subset), and `pytest` excluding `gpu`/`slow`/`integration`/`display`/`hardware` markers. Genesis-dependent compatibility tests require a simulation-capable environment.

## [0.2.13] — 2026-08-25

### Added

- Bundled the fixed-layout seed-7 `model_299_g2stable.pt` (SHA-256 `f2142a63…f783f2`), trained end to end under the `g2_stable_v1_3_3` profile, with a sanitized Genesis 1.3.3 configuration, regenerated SHA-256 manifest, and machine-readable release summary. Public evaluation uses this read-only bundle by default.
- Added fixed-layout behavior-cloning gate and unattended campaign helpers. Release evaluation now applies all nine seed/batch combinations to one selected checkpoint before running the fixed-bank and action-noise gates.

### Changed

- Raised the minimum, reference, and locked Genesis World baseline to **1.3.3** and locked its Quadrants runtime to **1.3.0**. Genesis 1.3.2 and older now fail closed; later versions still run the private-hook capability checks and emit the existing unvalidated-version warning.
- Added the `g2_stable_v1_3_3` physical-contact profile: Newton/100, pyramidal friction, convex contact resolution, GJK collision, and 32 rigid substeps. The 32-substep setting is a conservative project safety margin, not a universal Genesis 1.3.3 requirement; the primary NaN fix is the compliant five-equality G2 mimic tuple `(0.02, 1.0, 0.9, 0.95, 0.001, 0.5, 2.0)`, which also passed the 128-environment perturbed-contact check at 8 substeps.
- Extended RL checkpoint compatibility with the named physics profile. Checkpoints saved under earlier contact physics are rejected by the current evaluator and must be retrained.
- Raised the pick-place release-intent hysteresis `release_command_margin_m` from **0.5 mm to 18 mm** in the RL recipe. Under the integrated gripper command (`gripper_delta_mm: 4`) and sigma-0.02 evaluation noise, the old margin latched phantom releases at the pickup in effectively every noisy episode; a genuine release ramps the commanded gap to 45-84 mm, so detection now only lags about 0.3 s while the pads are still on the cube. Evaluation accepts `--release-command-margin-m` to score checkpoints saved with the previous margin.
- Made `simulation.substeps` the only rigid-step-count setting and raised its default to **32**. Removed the pick-place task's `substeps` and packaging task's `simulation_substeps`; strict configuration loading rejects both legacy fields instead of migrating them.
- Restored the packaging placement acceptance distance to **25 mm** for the 32-substep physics baseline.
- Reset the RL provenance chain to new Genesis 1.3.3 expert demonstrations, behavior cloning, and independent 300-iteration PPO runs for seeds 1, 7, and 17. Older-physics policies are forbidden, and the selected nonzero checkpoint must pass every fixed-layout and action-noise gate.
- Extended new RL run provenance with the Quadrants version and hashes for the shared simulation, artifact, model, transfer, and imported training-entry sources.

### Fixed

- Replaced open-loop G2 over-closing with object-only bilateral pad-force confirmation and a bounded, anti-windup physical hold controller targeting `10 +/- 2 N`; release keeps priority and every physics tick now fails closed on NaN or Inf.
- The scripted expert passed 128/128 uniform-random and 128/128 edge-start episodes with zero numerical, action, IK, or post-release-recontact faults. The xArm6 trajectory example placed within 3.734 mm and the one-cycle packaging example within 2.2 mm.
- Centralized the Gripper G2 mimic-equality solver parameters across trajectory, packaging, and RL scenes. G2 contact scenes now reject rigid substeps longer than **0.625 ms** before scene construction, preventing the Genesis 1.3.3 constraint-force NaN failure mode.
- Removed the gradual G2 packaging preload-relax phase at the 32-substep baseline. The one-tick full-open target now establishes finger clearance before gravity starts the drop while preserving the restored 25 mm placement gate.
- Pinned the development Ruff formatter to the lock-validated 0.15.21 release so the Torch/Genesis-free CPU CI cannot silently resolve a new formatter contract (0.16.1 began reformatting fenced Python in previously unchanged Markdown files).
- Excluded the new Torch/RSL-RL transfer test module from collection in the dev-only CPU CI environment, matching the existing handling for other simulation-stack training tests.
- Excluded Torch/RSL-RL-only model, transfer, and ignored legacy-migration modules from the dev-only coverage denominator; their dedicated tests continue to run in the maintainer's simulation/RL environment.

### Removed

- Retired the Genesis 1.3.1 fixed and hierarchical random-start checkpoint bundles, their configs, and their machine-readable summaries. The fixed bundle is replaced by the accepted 1.3.3 artifact; the public random-start example, recipe, documentation, and release contract are removed from v0.2.13 and deferred to a later version.

### Security

- Locked `gitpython` to 3.1.59 (fixes GHSA-9rj7-rf2p-w77r / GHSA-4gmw-gg2m-w46p / GHSA-wvpp-8hx9-p66j / GHSA-jm78-9fvv-mhgr / CVE-2026-76217 on 3.1.57) and `pip` to 26.2.1 (fixes PYSEC-2026-3721 on 26.1.2) so `project-check release` lock audit passes.

## [0.2.12] — 2026-08-03

### Added

- Added a separate xArm6 + Gripper G2 random-object-start, fixed-target pick-place workflow under `examples/rl/pick_place/random_start`, including a canonical recipe, deterministic uniform/boundary scenario banks, a validated checkpoint bundle, and bilingual train/evaluate guidance.
- Added object-only scenario generation and validation. Boundary banks cycle all four object corners, all published positions are unique and runtime-bounded, and target coordinates remain fixed.
- Added explicit `standard` and `robustness` aggregate evaluation gates. The robustness gate requires at least 99% full and quality success, P99 final error ≤10 mm, pre-lift drag ≤5 mm, post-release drift ≤3 mm, and zero action/IK faults.
- Added a central rigid-physics builder and runtime configuration for solver, friction cone, and contact-resolution selection.
- Added a fail-closed 44→54 guided-policy contract: the original 44-value actor is preserved bit-for-bit on the canonical layout, while six immutable layout offsets select four explicitly recorded scripted-guide actions on randomized object starts. The bundled random-start artifact is documented as a hierarchical simulator policy, not as learned PPO generalization, and every inherited actor tensor is frozen and mutation-guarded.
- Added protected 44→50 appended-column and layout-residual transfer paths for continued BC, DAgger, PPO, and curriculum research. These learned variants did not meet the v0.2.12 release gates and are not the bundled random-start solution.
- Added optional phase-balanced behavior-cloning weights and phase-count diagnostics so slow set-down demonstrations cannot silently dominate the supervised dataset.

### Changed

- Raised the minimum, reference, and locked Genesis World baseline to **1.3.1**. Project scenes now default to Newton constraints with the elliptic friction cone and Signorini contact resolution, 100 solver iterations, no no-slip post-pass, and a 5 ms constraint time constant.
- Checkpoint runtime contracts now include every physics field plus layout/target-randomization semantics. Full resume remains exact; actor or actor/critic transfer may explicitly change only `fixed_demo_layout`.
- The fixed-layout release checkpoint keeps its original weights while its published configuration, manifest, scenario binding, and evaluation evidence are rebaselined against Genesis 1.3.1 physics.
- The unchanged fixed-layout checkpoint passes all nine deterministic combinations (219/219 episodes), the independent fixed 64-episode gate, and 512/512 full/quality episodes with action-noise standard deviation 0.02. The noise diagnostic still records 120/512 post-release recontacts; that event is disclosed separately and is not a robustness-gate predicate.

### Fixed

- Curriculum advancement now enforces a per-stage dwell, blocks multiple transitions in one control step, clears circular histories and indices on entry, and excludes late episodes that began in the previous stage, preventing stale successes from skipping or contaminating envelopes.
- Scripted-expert evaluation now mounts and validates the requested scenario bank instead of reporting repeated implicit resets, derives the default episode count from that bank, and pins curriculum state before the constructor's first reset.
- Robustness summaries now require the quality-pass rate as well as full success, and `--require-target` returns a failing process status when the selected immutable gate is missed.
- The scripted expert compensates the repeatable G2 coupled-finger lateral bias, releases stored squeeze energy gradually, and adds near-table measured-velocity braking. The selected policy passed 512/512 final uniform and 256/256 final boundary full/quality episodes with zero action or IK faults and zero post-release recontacts.

## [0.2.11] — 2026-08-03

### Added

- Published the initial Linux/NVIDIA `examples/rl/pick_place` workflow for xArm6 + Gripper G2 in a fixed `+Y` layout: expert reachability evaluation, behavior cloning, PPO training, checkpoint fine-tuning, trace diagnostics, one canonical recipe, and a fixed 512-episode scenario bank.
- Bundled the 2.7 MB `model_199.pt` policy with a sanitized full configuration, regenerated SHA-256 manifest, and machine-readable evaluation summary. Public evaluation locates this bundle by default and explicit checkpoints resolve configuration from their own directory.
- Promoted fixed scenario generation/loading, the immutable `contact_v1` acceptance profile, run provenance, runtime-config body validation, and safe validated RSL checkpoint loading through `ufactory.training`.
- Added bilingual RL guides covering installation, GPU memory boundaries, viewer/headless/formal evaluation, training and fine-tuning, output locations, metric definitions, and scope limitations.

### Performance

- Viewer evaluation defaults to Genesis dynamic arrays, avoiding the former roughly 39-second static-array compilation wait; headless training and evaluation retain the saved performance-mode contract unless explicitly overridden.
- The release checkpoint passes all nine deterministic seed/batch combinations and the fixed 64-episode gate (64/64).

### Changed

- Raised both the minimum and reference Genesis World version to **1.3.0** and locked it at 1.3.0. Raised the RSL-RL minimum to **5.3.0** and locked the training reference at 5.4.2.
- Training and behavior-cloning outputs now default to ignored `outputs/rl/pick_place/` directories. The public evaluator is read-only with respect to the bundled checkpoint directory.
- The root `/rl/` ignore protects the private experiment archive; public RL code is tracked only under `examples/rl/`. Historical recipes, migration/action-head experiments, large logs, and server orchestration remain private.

### Fixed

- Public RL modules now run as `python -m examples.rl.pick_place.<entry>` without `sys.path` mutation or dependence on the private root `/rl/` tree.
- Checkpoint configuration and file hashes are verified before PyTorch deserialization; public entries use `weights_only=True` and do not call RSL-RL's unsafe default loader.
- Documented the fixed action-noise result honestly: standard deviation 0.02 yields **442/512 (86.3%)** with 512/512 grasp/lift. The 99% robustness goal remains unmet, so v0.2.11 makes no random-layout or high-robustness claim.

### Security

- Locked `gitpython` to 3.1.57 (fixes GHSA-94p4-4cq8-9g67 on 3.1.54) so `project-check release` lock audit passes.

## [0.2.10] — 2026-07-22

### Changed

- Raised the simulation baseline to Genesis World **1.2.3** (`genesis-world>=1.2.3`, lock pin, runtime `MIN`/`VALIDATED` versions, README badges, and hook-contract tests).
- Raised the optional `[rl]` extra to **`rsl-rl-lib>=5.0.0`**. Training recipe fixtures moved to the rsl-rl 5.x `actor` / `critic` / `obs_groups` schema; the previous 2.x `policy` / `ActorCritic` shape is not compatible.

### Fixed

- Release evidence summary fixture examples updated to the 1.2.3 Genesis baseline string.

### Security

- Locked `gitpython` to 3.1.54 (fixes GHSA-2f96-g7mh-g2hx / GHSA-v396-v7q4-x2qj / GHSA-956x-8gvw-wg5v on 3.1.50) and refreshed transitive pins so `project-check release` lock audit and installed-match pass.

## [0.2.9] — 2026-07-21

### Added

- Windows / CPU-only simulation path for pick-place and packaging: `--backend cpu|gpu`, packaging sim uses `GenesisRuntimeManager`, and bilingual install notes for CPU torch + unsupported GPUs.
- `AssetStore` protocol with `RepositoryAssetStore` (default) and `PackageAssetStore` extension point for a future wheel layout; discovery remains fail-closed on a source checkout. PyPI upload is not part of this release.

### Fixed

- Public CI mypy `--strict` failures in `ufactory/safety/gate.py` (joint-bound annotations and orientation-step index typing).
- `test_frac_down_on_grasp_forgetting` no longer depends on a local RL recipe's `place_phase_reset_frac` value.
- Packaging simulation no longer opens a deferred viewer or hangs in `_hold_final_view` unless `--visual` is set (headless CPU/Windows runs can exit after cycles).
- Public CI pytest collection skips Torch/Genesis-dependent modules (and the local-only RL next-params helper) when those stacks are not installed.

### Security

- `project-check release` continues to ignore pillow advisories on the validated Genesis stack pin `11.3.0` (adds CVE-2026-54058 / 59197 / 59198 / 59200 / 59204; fix line is 12.3.0).

## [0.2.8] — 2026-07-17

### Added

- Public GitHub Actions workflow (`.github/workflows/ci.yml`) runs the CPU `fast` check without installing Genesis/Torch.
- Root `NOTICE` for upstream xArm / UFACTORY URDF–mesh attribution; `ROADMAP.md` for the 0.2 preview → 0.3 official-org plan.
- `scripts/export_release_evidence_summary.py` builds a sanitized Release attachment from local `project-check` JSON reports.
- `ufactory.kinematics.tcp_offset` helpers read the controller TCP offset and convert flange ↔ tool-frame poses for FK/IK validation.

### Changed

- End-effector visual GLBs for Bio Gripper G2, Gripper G2, Lite6 Gripper, and Lite6 Vacuum Gripper are recompressed with native `draco_encoder` (same triangle counts) so each package stays under 1 MB of GLB; collision/visual STLs are unchanged. Gripper G2 also canonicalizes link5/6/7 visuals onto link6 meshes with URDF visual-origin offsets. Asset-integrity budget tests lock the GLB totals; the Lite6 accessory collision demo now imports `ufactory.visualization.lite6_gripper_viewer` instead of a removed examples shim.
- Robot-arm body visual GLBs for xArm5, xArm6, xArm7, UF850, and Lite6 are recompressed with native `draco_encoder` (no STL changes; only zero-area faces dropped where Draco requires it) so per-arm `visual_glb` totals land around 0.7–1.0 MB (from ~19–29 MB uncompressed). Budget tests and `dev/assets/restore_arm_body_native_draco.py` lock the pipeline.
- Local quality checks now use `project-check` as the single entry (`fast` / `sim` / `sdk-sim` / `hardware` / `release` / optional `deep`). Default `sim` runs `gpu and not slow` (xArm6 representative matrices). `deep` collects only `gpu or integration or display or slow` (not the full unmarked suite); packaging deep matrix is five robots × one `servo_j` cycle plus one xArm6 cartesian sample; five-robot CLI smoke belongs to parallel `sdk-sim`, not `deep`. Release reuses same-commit fast reports, can carry evidence across docs-only commits, reads the version from `pyproject.toml`, and runs sdk-sim/hardware inventory commands per robot in parallel.
- Documentation honesty pass: README/CONTRIBUTING replace “validated / hardware-validated” with reference baseline and maintainer-verified wording; Project status states 0.2.x Alpha on a personal GitHub account with a planned 0.3.x move to a UFACTORY organization repository; verification-boundary table clarifies public CI vs local sim vs hardware.
- Package metadata authors/maintainers and Citation use the maintainer name; LICENSE copyright aligned; asset READMEs add Source/License pointers to `NOTICE`.
- Fast-tier Genesis imports in compatibility/PBR tests use `importorskip` so CPU CI can collect and skip without a sim stack.
- `ufactory.quality.evidence_summary` powers `scripts/export_release_evidence_summary.py` for sanitized Release attachments.
- `scripts/README.md` splits user helpers from maintainer tools; root README/CONTRIBUTING point at that index. Local `rl_*` experiment scripts remain uncatalogued and gitignored.
- Fast-tier demotion: `test_default_pick_place_program_preflight_passes_servo_j` is marked `gpu` (Genesis compile); `test_servo_cartesian_packaging_preflight_passes_for_all_robots` is marked `slow` (Pinocchio multi-robot, not in-process GPU).
- Test module filenames drop historical `_v025` / `_v026` / `_v027` suffixes (no compatibility aliases); CONTRIBUTING examples updated.
- Integration example smoke is consolidated into `tests/robots/test_representative_example_smoke.py` (xArm6 representative + strong pick-place assertions); duplicate weak `verify_robot` / exit-only pick-place paths removed. Ad-hoc real FK/IK/dynamics stay in `tests/hardware/test_xarm6_real.py`.
- `project-check release` audits the frozen `uv export` set with `pip-audit --no-deps --disable-pip` (no temp-venv wheel downloads), runs that audit in parallel with `snapshot_fast`, and adds `lock-installed-match` (installed version drift fails; lock-only missing extras are reported but do not fail).

### Removed

- Reach RL task end-to-end: local `examples/rl/reach/`, `ufactory.training.logic.reach`, `ufactory.training.actions`, `build_reach_task_configs`, `assets/configs/runtime/tasks/reach.yaml`, and related fixtures/tests. Pick-place keeps its `reward_reach` approach term.
- `scripts/migrate_legacy_checkpoint.py` (one-shot trusted `cfgs.pkl` migration); no longer part of the public helper surface.
- `tests/robots/test_xarm5_collision_mesh_equivalence.py` (historical OBJ↔STL Hausdorff lock from the 0.2.0 mesh migration); public trees still forbid collision OBJ via asset-integrity tests.

### Fixed

- FK/IK verification scripts now read the controller `tcp_offset` and compare Genesis flange FK against the SDK tool pose (and convert IK targets back to flange) so non-zero TCP offsets no longer fail validation spuriously.
- Representative example smoke runs `dynamics-sim-check` / `dynamics-hardware-check --dry-run` in subprocesses so consecutive cases no longer hit `Genesis already initialized` in one pytest process.

## [0.2.7] — 2026-07-15

### Added

- Task-oriented public examples under `visualization/`, `kinematics/`, `pick_place/`, and `packaging/`, with strict runtime overlays beside their entry points. Local RL training lives under gitignored `/rl/` (not under `examples/`) until the fixed-layout goal succeeds.
- `ufactory.manipulation.packaging` as the reusable packaging geometry, planning, scene, simulation, and natural-drop diagnostics package.
- Shared `ufactory.visualization` viewer implementations, package-level RL configuration builders in `ufactory.training`, overwrite-safe training directories, and complete checkpoint/config inventories.
- Architecture tests that reject production imports from `examples`, public `sys.path` mutation, legacy bootstrap paths, and Pinocchio/Coal leakage into the simulation/RL extras.
- Runtime compatibility checks for the Genesis version, GLB PBR hooks, deferred Viewer internals, and FK/IK scratch allocation. Versions newer than the validated 1.2.2 baseline emit a one-time warning and fail early when required hooks have changed.
- YAML-driven packaging profiles for xArm5, xArm6, xArm7, UF850, and Lite6, including strict box/table/target/gripper validation, robot-specific gripper geometry, effective scene hashing, and a Lite6 task overlay.
- Full real-packaging capability for Lite6 with its binary open/close gripper adapter; xArm6 + Gripper G2 remains enabled.

### Changed

- **Breaking:** public examples are organized by task rather than robot; the old per-robot wrappers and root underscore modules are removed without compatibility shims. See the English and Chinese example indexes for migration commands.
- **Breaking:** the shared trajectory task identity, CLI, and example paths are renamed from `grasp_place` to `pick_place` (`ufactory-pick-place`, `task.name: pick_place`, `examples/pick_place/`) with no compatibility aliases. Packaging remains `packaging_showcase` / `examples/packaging/`. Public RL example paths are deferred; library APIs stay under `ufactory.training`.
- Pick-place RL accepts only xArm6 + Gripper G2 and binds `env.runtime_config_sha256`; recipes contain only RL environment/reward/PPO/run settings and physical values resolve from `ResolvedRuntimeConfig`.
- Packaging release and post-release motion remains entirely Genesis contact-driven. The isolated drop report records impact, post-impact velocity, rebound peak, and settle time without imposing a minimum visible bounce.
- Pinocchio 4/Coal remain optional public backends: absent from `.[sim,rl]`, present in `.[real]` for safety preflight and `.[dynamics]` for reference dynamics.
- The simulation extra now requires `genesis-world>=1.2.2` and `packaging>=24`; the reproducible lock remains on the validated Genesis World 1.2.2 and PyTorch 2.10.0 stack.
- Existing manipulation physics settings remain unchanged while inheriting Genesis 1.2.2 contact-pruning, non-convex collision, thin-shell, and no-slip solver fixes.
- Packaging scene construction and physical execution are robot-generic and now exposed through the task-oriented packaging example and unchanged console command.
- Packaging CLI, report names, calibrated URDF selection, SDK simulation, and isolated real-time mirrors now preserve the selected robot key. xArm5, xArm7, and UF850 real packaging fails before controller connection while their simulation and SDK-simulation paths remain available.
- `box_floor_top_z_m` is consumed as the actual floor top, placement uses `fixed_target_position_m`, and grasp/release link heights derive from configured gripper geometry instead of G2-only constants.
- Packaging showcase `simulation_substeps` is restored from 32 to 8 to cut per-step physics cost during the motion phase (about 4× less work per control step on the measured packaging scene). The GPU three-cycle place-success threshold is softened to 150 mm to match that tradeoff.
- Lite6 gripper finger collision uses dual convex boxes (inner pad + outer boss) that preserve the L-shaped concavity. Trajectory and packaging scenes no longer disable Genesis `convexify`/`decimate`/`watertighten` for Lite6 whole-robot URDFs. Safety exemptions for opposing-finger pairs are bound to both the nominal and SN-calibrated Lite6 URDF hashes; the Lite6 packaging box center is moved to Y=0.280 so calibrated preflight clears the near wall.

### Removed

- The disabled `ufactory.deploy` package, xArm6 reach deployment entry point, online policy/session/action adapters, legacy `SafetyGuard`, random-action execution, and their tests.
- Duplicate reach/grasp RL defaults from `TaskProfile`, the legacy `_grasp_place_traj.py`, all `_bootstrap.py` files, and per-robot public example wrappers.

### Fixed

- Removed the obsolete exact-1.2.1 runtime version check that rejected the validated Genesis 1.2.2 environment before scene initialization.
- Lite6 packaging no longer fails the two prior inverse-kinematics points or reports the gripper-stage collision set; its collision exemptions are limited to the two verified settle stages and bound to the current URDF hash.
- `project-check release` snapshot now uses a detached git worktree so asset tests that call `git ls-files`/`git show` still work, and `pip-audit --strict` audits the frozen `uv export` dependency set while skipping the editable local install and known advisories on the validated torch/pillow pins.

### Security

- The public package no longer ships online real-policy execution code. Real robot motion continues to require predictive safety checks, exact calibration binding, and explicit confirmation.

## [0.2.5] — 2026-07-14

### Added

- Versioned immutable YAML configuration with strict validation, deterministic source precedence, canonical hashes, `--print-config`, and source-tree asset integrity checks.
- `SafetyGate`, structured violations/reports, hash-bound `ApprovedProgram`, calibrated Pinocchio 4/Coal full-sample collision prediction, and injected kinematics/collision/clock/transport ports.
- Unified `ufactory-grasp-place` entry point, monotonic fail-closed real executor, same-run SDK simulation evidence for `servo_cartesian`, strict calibration manifests, and capability-driven G2/Lite6 adapters.
- Safe checkpoint YAML/manifests, scoped Genesis 1.2.1 PBR integration, `project-check` local quality reports, and a reproducible `uv.lock`.

### Changed

- **Breaking:** v0.2.5 supports cloned source repositories installed with `pip install -e .` only; wheel/sdist installation is unsupported.
- **Breaking:** configuration, trajectory and safety APIs are replaced by `ResolvedRuntimeConfig`, `preflight_program()`, `ApprovedProgram`, `execute_sim()` and `execute_real()`; no compatibility shim is promised.
- Maintainer technical notes live under the local gitignored `dev/docs/` workspace and are not published with the repository; breaking API changes are recorded in this CHANGELOG.
- Whole-program motion statistics now use signed velocity, vector acceleration, wrapped orientation differences and static endpoint velocities. Genesis initialization has a single runtime owner.
- Dynamics reports write schema v4 while readers retain v1-v3 support. Genesis and Torch are pinned to the validated 1.2.1 / 2.10.0 environment.
- Grasp-place trajectory, training/evaluation, and packaging examples now resolve one runtime-configured 30 mm, 17 g reference block; its table-resting base-frame center is consistently 15 mm in Genesis and predictive collision scenes.
- Pinocchio Cartesian shadow IK now enforces the shared gripper-down full pose with nominal `1e-6` damping and bounded near-singularity regularization. Offline Cartesian simulation uses a simulation-only approval while hardware approval still requires same-run, hash-bound SDK evidence no older than five minutes.

### Fixed

- Fixed Lite6 grasp-place preflight failures caused by a 5 mm object-height mismatch and non-existent collision-exemption stage names. Failure output now groups the complete violation set instead of silently showing only the first 20 samples.
- Simulation final metrics now hold the last approved target for two physics ticks before observation, eliminating end-of-tick tracking-lag false failures without changing the grasp-place command stream.

### Security

- Online real policy deployment and random-action real modes are hard-disabled. Runtime pickle loading is removed; checkpoints use `weights_only=True` and integrity manifests.
- Real execution never automatically clears errors, resets, resumes or catches up missed servo deadlines. Real mode requires exact serial/calibration binding and `--confirm-real`.

## [0.2.4] — 2026-07-09

### Changed

- Moved maintainer-only diagnose and legacy dynamics/demo scripts from `examples/` into the local gitignored workspace `dev/diagnostics/` (`manipulation/`, `dynamics/`); public docs and CI now point at `dynamics-sim-check` / `dynamics-hardware-check` instead of the old example wrappers.
- Gripper G2 grasp-place examples now insert a 0.5 s closed-gap `grip-settle` segment before lift so xArm5/6/7 and UF850 wait for the grasp to settle before raising the arm.
- Lite6 grasp-place `grip` and `release` segments now use 0.5 s durations, making the Lite6 gripper open/close four times faster while keeping the 0.18 s `place-settle`.
- Real gripper execution treats same-gap gripper hold segments as paced no-ops, so `grip-settle` and `place-settle` do not resend SDK gripper commands.
- Grasp-place MoveL source timing is now controlled by `--speed-mm-s` / `--mvacc-mm-s2` for sim, dry-run, `servo_cartesian`, and `servo_j`; defaults are 150 mm/s and 800 mm/s².
- Host-side IK `servo_j` now preserves the source Cartesian tick count and duration instead of silently retiming, so unsafe custom speeds fail through the real executor's finite-difference safety checks.
- UF850 host-side IK uses higher damping during `servo_j` compilation to keep a calibrated unit's wrist solution continuous near singular poses.
- Root README (EN/ZH): simplified GLB preview and trajectory grasp-place sections; added English Showcase section; aligned dynamics CLI examples.

## [0.2.3] — 2026-07-09

### Added

- Host-side IK `servo_j` path for shared grasp-place trajectory examples: MoveL samples are compiled with Genesis IK into explicit `set_servo_angle_j` joint streams.
- `--kinematics-yaml`, `--kinematics-yaml-dir`, and `--force-kinematics` options on grasp-place real paths; `--ip` still auto-resolves the SN-derived kinematics suffix.
- `compile_cartesian_program_to_joint_stream()` trajectory API and explicit `q_samples` MoveJ segments.

### Changed

- Grasp-place `--executor` values renamed to underscores: `servo_j` and `servo_cartesian` (hyphen forms removed).
- Grasp-place `--executor servo_j` is now supported for dry-run, SDK simulation validation, real streaming, mirror mode, and optional physical gripper commands.
- Host-side IK joint streams are plateau-collapsed and LSPB-retimed after Genesis IK so 100 Hz `servo_j` finite-difference acceleration stays within servo limits.
- Real-path mirror/carry tracking now handles compiled `movej` arm segments by label, preserving existing grasp/release behavior.
- Shared grasp-place cube physics unified across robots: 30 mm painted wood block at 17 g with friction 1.0; silicone fingertip pads at friction 1.2; contact stiffness left at Genesis rigid defaults (no per-object sol_params).
- Lite6 default sim hold bias raised from 0.8 mm to 2.0 mm so contact friction still carries the unified 17 g cube after lowering material μ.

### Fixed

- `servo_j` SDK feedback comparison now respects the program DOF instead of assuming six joints.

## [0.2.2] — 2026-07-08

### Added

- Contact-friction grasp diagnostics for Gripper G2 (`g2_contact_grasp_diagnose.py`), Lite6 reversed gripper (`lite6_contact_grasp_diagnose.py`), and standalone Lite6 gripper+cube collision (`lite6_gripper_cube_diagnose.py`).
- Lite6 `place-settle` segment: a 0.18 s closed-gripper hold before release so the block opens at table height.
- `gap_lspb_samples` support for holding an identical gripper gap over a requested duration (used by `place-settle`).

### Changed

- Grasp-place simulation now defaults to rigid-body contact and friction (`sim_grasp_weld=False`); distance weld, geometric snap, block freeze, and forced block motion are removed from the default path.
- Gripper G2 default grasp gap is now 22 mm for contact preload on the 30 mm cube.
- Lite6 grasp-place uses raw STL finger collision (no Genesis convex proxy), contact-latched close after bilateral finger contact, and a 0.8 mm default sim hold bias.
- Lite6 finger visual/collision origins are reset to zero so the flat pad grips the cube instead of the finger root/limit area.

### Fixed

- Lite6 grasping on the finger limit area instead of the fingertip pad.
- Lite6 premature top-surface contact from processed convex collision proxies.
- Lite6 block sliding during lift/transit and pressing into the table during place.
- Lite6 place bounce and mid-air release after removing `place-standoff`.
- Gripper G2 over-tight preload causing visible carry jitter; default gap tuned to 22 mm.
- Lite6 headless real dry-run now includes `place-settle` because `dry_heights()` exposes gripper metadata.

## [0.2.1] — 2026-07-08

### Added

- Robot-aware LSPB trajectory planning APIs for joint, Cartesian, and mixed waypoint programs.
- Shared five-robot grasp-place trajectory examples for xArm5, xArm6, xArm7, UF850, and Lite6, with Genesis sim, dry-run, real `servo_cartesian`, and kinematic mirror paths.
- Lite6 gripper command conversion and tests for its 20-38 mm physical opening range.

### Changed

- Grasp-place scenes now use 30 mm default cubes, calibrated grasp gaps, and shared trajectory scene defaults across supported robots.
- Registered arm links for xArm5/6/7 1305, Lite6, and UF850 now use `visual/*.stl` meshes for collision to better match the physical curved link surfaces.
- Collision visualization tooling can now load arm+accessory combo URDFs with raw STL collision meshes, including Gripper G2, Bio Gripper G2, Lite6 Gripper, and Lite6 Vacuum.
- Gripper G2 and Lite6 Gripper keep their packaged STL paths because they match upstream xarm_ros2 visual STL assets; Bio Gripper G2 already uses visual STL collision, and Lite6 Vacuum now uses upstream visual STL collision.
- Asset integrity tests now assert registered arm collision references point to visual STL, validate accessory STL collision references, and still reject OBJ collision meshes.

### Fixed

- Grasp-place viewers now start from the planned home/pregrasp pose instead of briefly showing zero-position arms with grippers inside the table.
- Lite6 grasp-place now keeps finger collision/visual gaps aligned with the 30 mm cube and freezes release/place transitions to avoid pushing the cube into the table.
- Simulation final place-error reporting now handles mixed CPU/CUDA tensor devices in test doubles and Genesis contexts.

## [0.2.0] — 2026-07-03

### Added

- **Five-robot dynamics validation**: xArm5, xArm6, xArm7, Lite6, and UF850 hardware-verified against Genesis PD hold (20 calibration poses each).
- **xArm5/6/7 collision meshes**: vendor simplified collision hulls converted from OBJ to STL (geometry-equivalent; aligns with firmware STL collision format).
- **Tests reorganized by module**: `tests/robots/`, `tests/dynamics/`, `tests/hardware/`, `tests/trajectory/`, `tests/deploy/`, `tests/manipulation/`.
- **`test_public_api_layout.py`**: asserts canonical subpackage imports and that legacy root modules raise `ModuleNotFoundError`.
- **`test_xarm5_collision_mesh_equivalence.py`**: git OBJ vs working-tree STL Hausdorff/volume equivalence.
- **`test_xarm5_pose4_stl_collision.py`**: documents link3↔link5 Genesis self-contact and `pd_tracking_saturation` check bypass.
- **`assets/urdf/xarm{5,6,7}/README.md`**: collision/visual mesh layout notes.

### Changed

- **Breaking package layout**: public Python modules are now grouped by function domain: `ufactory.robots`, `ufactory.kinematics`, `ufactory.dynamics`, `ufactory.hardware`, `ufactory.grippers`, `ufactory.trajectory`, `ufactory.manipulation`, `ufactory.visualization`, and `ufactory.deploy`.
- The root `ufactory` namespace now exposes only core robot convenience APIs: profile lookup, generic URDF/asset paths, and runtime profile lookup.
- Five-robot arm dynamics URDFs now reference `collision/*.stl` instead of `collision/*.obj`.
- `observe-hold-current` console entry point targets `ufactory.hardware.observe`; dynamics CLI entry points remain under `ufactory.dynamics.cli`.
- Dynamics report schema v3: explicit per-joint torque units (`*_nm`), `status_reason`, `worst_joint`, `worst_abs_err_nm`; reports archived under `reports/dyn_ver_<SN>/`.

### Removed

- Legacy root modules were removed with no compatibility wrappers: `ufactory.paths`, `ufactory.robot_registry`, `ufactory.robot_params`, `ufactory.kinematics_validation`, `ufactory.real_robot_session`, `ufactory.xarm_control`, `ufactory.gripper_g2`, `ufactory.bio_gripper_g2`, `ufactory.glb_visual`, `ufactory.dynamics_validation`, `ufactory.dynamics_static_analysis`, and `ufactory.dynamics_verify`.
- xArm5/6/7 `collision/*.obj` meshes (replaced by equivalent STL).

### Known Issues

- **xArm5 pose 4 (Genesis)**: link3↔link5 self-contact persists in kinematic and PD-hold modes (mesh geometry overlap in simulation). Real robot and SDK collision checks pass; `pd_tracking_saturation` check allows hardware validation when Pinocchio gravity at the target is nominal.

## [0.1.6] — 2026-07-02

### Added

- **Asset integrity tests**: public URDF mesh references must resolve, dynamics URDF collision meshes must use `collision/*.stl` (not visual or OBJ), and raw/source GLB intermediates must stay out of the public file set.
- Optional dependency extras: `real` for xArm SDK, `rl` for rsl-rl training/evaluation, and `showcase` for the packaging showcase scipy dependency.

### Changed

- Public source tree slimmed for GitHub release: final GLB visual assets remain bundled, while raw/source GLBs and relocalize metrics are no longer part of the public surface.
- Default `requirements.txt` now contains only baseline simulation/viewer dependencies; advanced real-robot and RL dependencies are opt-in extras.
- xArm6 1305 URDF collision entries now reference `meshes/xarm6_1305/collision/*.obj` instead of reusing visual STL meshes.
- Standalone Gripper G2 and Bio Gripper G2 URDFs now point to repository-local mesh paths so all public URDF mesh references resolve.

### Removed

- Public asset-regeneration scripts (`vendor_*`, `relocalize_*`, `generate_*_combo_urdf.py`, `verify_*_assets.py`) from `scripts/`; generated final URDF/GLB assets remain available.
- `ufactory.dynamics_pose_selection`, `scripts/select_dynamics_calib_poses.py`, and their tests; default validation poses now live in `assets/configs/dynamics_validation_pose.yaml` and are loaded through `ufactory.dynamics.poses_config`.

## [0.1.5] — 2026-06-29

### Added

- **Runtime profiles**: typed runtime profiles (joint PD, torque limits, dynamics poses, task capabilities)
- **Dynamics validation**: Genesis PD hold vs real-robot static torque validation; CLIs `dynamics-sim-check`, `dynamics-hardware-check`, `dynamics-sim-collision-check`, `dynamics-report-compare`
- **Real-robot helpers**: real-robot connection, rad/rad/s motion, hold sampling
- **`ufactory.dynamics_pose_selection`** and **`scripts/select_dynamics_calib_poses.py`**: EE y hemisphere stratified calibration pose selection
- **`ufactory.kinematics.validation`**: shared FK/IK verification CLI for generic robot examples
- Generic examples: `packaging_showcase.py`, `fk_verify_robot.py`, `ik_verify_robot.py`, `arm_reach_env.py`, `grasp_place_env.py`; expanded test suite

### Changed

- **xArm6 dynamics poses**: `home` + 20 `calib_*` points (10 y+ / 10 y−) from absolute-accuracy calibration file, replacing legacy `config_*` presets
- **Real-robot motion units**: unified **rad / rad/s**; default speed ≈0.698 rad/s (40°/s equivalent)
- **Default move strategy**: `direct` (axis-sequential caused self-collision on several calibration poses)
- Multi-robot examples slimmed to profile-driven thin wrappers

### Removed

- Public **`docs/`** directory (content moved to local `dev/` workspace)
- Two misnamed xarm6 cached calibration URDFs under `assets/urdf/xarm6/`

### Fixed

- **`speed=40` with `is_radian=True`** was interpreted as 40 rad/s and clamped to π rad/s (~180°/s); default now correctly uses `math.radians(40.0)`

## [0.1.3] — 2026-06-25

### Changed

- **Minimum Genesis World version** raised to 1.2.0 (`ViewerOptions.max_FPS` → `refresh_rate`)

### Fixed

- **Bio Gripper G2 on xArm7 (link7)**: reject mirrored pin-hole solution that sank the static GLB into the flange in Genesis preview
- **Regenerated** link5/6/7 Bio G2 visual GLBs and `relocalize_metrics.json`; updated uf850 movable attach origin

## [0.1.2] — 2026-06-22

### Added

- **`BioGripperG2` controller module** for reusable open/close control across all supported arms
- **Per-robot `bio_gripper_g2_attach` origins** computed during relocalize and written into combo URDFs
- **`robot_cli_choices()`** and short-name aliases: `xarm5` / `xarm6` / `xarm7` resolve to `*_1305` profiles
- **`tests/test_robot_registry.py`** for profile resolution and default URDF paths

### Changed

- **Bio Gripper G2 movable mount** uses canonical EE pin-hole alignment (fixes UF850 flange inversion and finger +X orientation)
- **Regenerated** Bio Gripper G2 combo URDFs, per-link GLBs, and `relocalize_metrics.json`
- **Default xArm URDF** paths now point to `*_1305.urdf` (`xarm6_urdf()` and friends); CLI/docs recommend short names `xarm6` etc.

### Fixed

- UF850 Bio Gripper G2 movable attach (`ring_gap_mm=0`, fingers toward base +X)
- `verify_bio_gripper_g2_assets.py` world-frame finger-direction checks for movable combos

## [0.1.1] — 2026-06-18

### Changed

- **Rename `bio_gripper` → `bio_gripper_g2`** across all assets, URDFs, examples, and scripts
- **Fix Bio Gripper G2 flange orientation** when mounted on robot arm (link5/link6/link7)

### Fixed

- `xarm6_1305_visual_glb_urdf()` now accepts `with_bio_gripper_g2` parameter (consistent with xarm5/xarm7/uf850)
- README `--gripper-demo` flag table now includes Bio Gripper G2

### Removed

- Diagnostic and keyframe-capture scripts moved from `scripts/` and `examples/xarm6/` to `dev/diagnostics/` (gitignored)

## [0.1.0] — 2026-06-18

### Added

- **Robot profiles:** xArm 5/6/7 (1305 variant), UF850, Lite6 with `RobotModelSpec` registry
- **End-effector support:** Gripper G2, Bio Gripper G2 (xArm/UF850), Lite6 Gripper, Lite6 Vacuum
- **GLB visual rendering** with PBR material preservation (metallic/roughness) via Genesis monkey-patching
- **Unified GLB viewer** (`examples/view_robot_glb.py`) supporting all robots and accessories
- **xArm6 reference verification:** FK/IK comparison with real robot, dynamics validation
- **Kinematic calibration:** Per-unit URDF patching from firmware YAML (SN-based eligibility)
- **RL environments:** Reach and grasp-place tasks for xArm6 (rsl-rl-lib)
- **Showcase scene:** Physical pick-place demo (xArm6 + Gripper G2 + cardboard box)
- **Multi-robot smoke tests** (headless, no hardware required)
