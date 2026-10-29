# Changelog

All notable changes to genesis-ufactory will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.6] — 2026-10-29

### Changed

- Updated `examples/rl/pick_place` to sample initial cube positions uniformly at random in `x ∈ [300, 500] mm`, `y ∈ [-200, 200] mm` (`SPAWN_XY_LOWER = (0.300, -0.200)`, `SPAWN_XY_UPPER = (0.500, 0.200)`) with a fixed target at `(300, 300) mm`, added `spawn_xy_lower` / `spawn_xy_upper` to the pick-place runtime environment contract, computed horizontal drag and transport potential shaping relative to each episode's actual spawn `init_obj_pos`, updated `PickPlaceExpert` and `evaluate.py`'s CPU-deterministic `make_spawns()` generator across the full workspace envelope, and refreshed the bilingual guides (`examples/rl/pick_place/README.md`, `README_zh.md`).
- Decoupled bare-arm `reach` runtime configurations and checkpoint contract validation from gripper parameters: `load_runtime_config(..., task="reach")` sets `gripper = None`, and `_REACH_RUNTIME_ENV_CONTRACT_KEYS` validates the 21 Reach-specific environment and solver parameters without coupling to the global `runtime_config_sha256`.
- Upgraded transitive `msgpack` from `1.2.1` to `1.2.4` in `uv.lock` (resolving `CVE-2026-114613`) and added two new `pillow 11.3.0` advisory aliases (`PYSEC-2026-4327`, `PYSEC-2026-4332`) to `PIP_AUDIT_IGNORED_VULNS`.

### Fixed

- Recalibrated Gripper G2 meshes, URDF kinematics, and runtime TCP offsets against the official SolidWorks STEP assembly (`Gripper G2 AG1200 20250710.STEP`):
  - Rebuilt all 8 Draco-compressed visual GLBs (`501.0 KB` total across 7 movable links, `386.8 KB` static assembly) and 7 collision STLs under `assets/urdf/gripper_g2/meshes/` directly in their canonical flange/joint reference frames, eliminating legacy G1 housing mix-ups and centroid-fit visual offsets (`0.00000 mm` axial error and `<= 0.00044 mm` opening error across all 85 poses `0–84 mm`).
  - Updated all 11 standard Gripper G2 URDFs (`drive_joint` and `right_outer_knuckle_joint` Z `0.059098` → `0.0606 m`, `left/right_inner_knuckle_joint` Z `0.074098` → `0.075602 m`, `joint_tcp` Z `0.172` → `0.1735 m`) and zeroed residual `<visual><origin>` offsets on `xarm5` and `xarm7` G2 combo URDFs.
  - Updated `gripper.tool_tip_offset_z_m` from `0.1011` to `0.1026` across `assets/configs/runtime/robots/{xarm5_1305,xarm6_1305,xarm7_1305,uf850}.yaml`.

## [0.3.5] — 2026-10-27

### Added

- Extended `examples/rl/reach` across all five bare-arm models (`xarm5`, `xarm6`, `xarm7`, `uf850`, and `lite6` via `REACH_RL_ROBOTS`), with per-robot runtime overlays (`assets/configs/runtime/tasks/robots/*_reach.yaml`), dynamic `3 * dof + 6` observation and `dof` action dimensions in `ReachEnv` (`XArm6ReachEnv` kept as a compatibility alias), `--robot` CLI selection, automatic robot inference from saved checkpoint `config.yaml`, and subprocess peak GPU memory sampling (`process_memory_mib`) in `run_experiments.py`.
- Published pretrained Reach checkpoints (`model_299.pt` with SHA-256 manifests and repo-relative `robot.urdf_path`) for all five robots to Hugging Face (`UFACTORY-Developer/ufactory_genesis_rl`) and documented one-command `hf download` instructions and RTX 4060 Ti (8 GB) benchmark tables (`num_envs = 4096`, peak VRAM `1.82–2.60 GB`, training time, and dual held-out evaluation pass rates).

### Changed

- Unified the default initial joint configuration (`arm.default_qpos_rad`) across all five Reach robot configs to the analytical IK solution for flange TCP `(0.30, 0.00, 0.30) m`, `RPY = (180°, 0°, 0°)`.
- `examples/rl/reach` realtime viewer now auto-detects screen resolution via `try_get_display_size()` when `ViewerOptions.res` is unset instead of hardcoding `1280x720`.
- Upgraded six transitive dependencies in `uv.lock` to resolve 15 vulnerability advisories: `certifi` (`2026.7.22` → `2026.10.23`), `filelock` (`3.29.7` → `4.2.4`), `gitpython` (`3.1.59` → `3.2.1`), `setuptools` (`83.0.0` → `84.2.1`), `urllib3` (`2.7.0` → `2.8.0`), and `werkzeug` (`3.1.8` → `3.1.9`).
- Updated `PIP_AUDIT_IGNORED_VULNS` in `ufactory.quality.project_check`: added 36 `pillow 11.3.0` advisories whose upstream fix requires `pillow>=13.0.0` (blocked by `genesis-world 1.4.5` → `moviepy 2.2.1` requiring `pillow<12.0`; `ufactory` does not import `PIL` or parse untrusted images/PDFs) and removed three stale ignore IDs (`PYSEC-2026-196`, `PYSEC-2026-3447`, `PYSEC-2026-2689`) already resolved by locked versions.

## [0.3.4] — 2026-10-26

### Changed

- Set the Genesis World reference baseline to **1.4.5** (minimum stays **1.4.2**; 1.4.2 to 1.4.4 emit one unvalidated-baseline warning). The lock file pins Genesis 1.4.5, trimesh 5.2.0, and zstandard 0.25.0.
- Capped Genesis World at **1.4.5**: the `[sim]` extra requires `genesis-world>=1.4.2,<=1.4.5`, and `require_genesis_version` rejects newer releases (including post and dev builds) with `GenesisCompatibilityError` instead of warning.
- The `[sim]` extra keeps `trimesh>=4.8.2`, matching the minimum supported Genesis 1.4.2; Genesis 1.4.5 itself requires `trimesh>=5.0.0`.
- The `[rl]` extra now requires `rsl-rl-lib>=5.4.2,<6` and declares `tensordict>=0.7.0`, which `examples/rl` imports directly.
- Bio Gripper G2 microplate simulation now uses the shared `require_genesis_runtime` check instead of requiring exactly `genesis-world==1.4.4`.
- Documented why `torch==2.10.0` is pinned (validated baseline; last PyPI default CUDA 12.x build) and how to install another build of the same version (`+cu126`, `+cu130`, `+cpu`) before `.[sim]`.
- `project-check` lock audit also ignores `PYSEC-2025-194`, the alias of the already ignored `CVE-2025-3000` (torch 2.10.0, `torch.jit.script`, not used).
- GitHub Actions CI (`project-check fast`) now runs across a Python `3.12` and `3.13` matrix.
- Rechecked three Genesis 1.4.4 behavior notes under 1.4.5 and updated them to `1.4.4/1.4.5`: Reach default-pose hold error (max 4.5 mrad), packaging showcase one-tick G2 release (`test_packaging_physics_grasp_and_drop_regression` passes), and xArm 5 pose `[90, -90, -60, 160, -90]°` (no false self-collision, J4 not saturated).

### Fixed

- `packaging` moved from the `[sim]` extra to the core dependencies. `ufactory.simulation.compat` imports it at module level, so `dynamics-hardware-check`, `dynamics-sim-check`, and `ufactory.visualization` failed with `ModuleNotFoundError` in a clean environment installed with only `.[real]` or no extras.
- Worked around a Genesis 1.4.5 SupportField regression (commit `375fd7eb`): for directions with azimuth just above -π the cell index underflows to -1 and reads the previous geom's support table or the wrong chart, producing false penetrations (the Bio G2 microplate was launched off the nest during reset). `require_genesis_runtime` replaces `_func_direction_cell` with a copy that restores the lower-bound wrap, only when the installed source matches the known 1.4.5 implementation; 1.4.4 and older are untouched.
- Worked around a Genesis `Mesh.decimate` ordering bug (`patch_mesh_decimate` in `ufactory.simulation.compat`): `Mesh.decimate` checked `len(faces) > decimate_face_num` before calling `process(validate=True)`, so when degenerate-face removal dropped a mesh from just above 500 faces to below 500 (e.g. `vacuum_gripper` `hull_012_cup_01.stl` going from 504 to 499 faces with SciPy 1.18.1), `fast_simplification.simplify` raised `ValueError: target_count must be less than the number of faces`.
- `tests/simulation/test_support_field_wrap_gpu.py` now writes its subprocess probe script to a temporary `.py` file instead of passing `-c`, because Python 3.12 does not populate `linecache` for `<string>` and `@qd.kernel` requires `inspect.getsourcelines`.

### Removed

- Removed `requirements.txt`; `pyproject.toml` extras (`pip install -e ".[sim]"`) are the only dependency declaration.

## [0.3.3] — 2026-10-23

### Added

- Added calibrated Bio Gripper G2 dynamics (`0.79 kg` total mass and link inertias, 5 watertight convex collision meshes, virtual-work drive conversion `F_drive = 2 * F_finger`, and a ramp-limited conditional-integral controller with `41.67–333.33 mm/s` per-finger speed and `40 N` drive limit), the SBS 96-well microplate asset (`assets/urdf/microplate/`, BSD-3-Clause attribution in `NOTICE`), and multi-robot microplate pick-and-place simulation (`xarm5`, `xarm6`, `xarm7`, `uf850`) supporting both `servo_cartesian` (default) and `servo_j` with a shared flange home pose and 0.4 s quintic/SLERP transitions.
- Added static Vacuum Gripper assets for the xArm series (`xarm5`, `xarm6`, `xarm7`) and `uf850` under `assets/urdf/vacuum_gripper/` (`0.61 kg` mass, CoM `[0, 0, 0.053] m`, TCP `[0, 0, 0.126] m`, Draco visual GLB, and 33 region-split watertight convex collision hulls), combo URDFs (`*_vacuum_gripper_visual.glb.urdf`), `vacuum_gripper_urdf()`, `robot_visual_glb_urdf(..., with_vacuum_gripper=True)`, and `view_robot.py --vacuum-gripper`.
- Added dedicated bilingual task guides under `examples/pick_place/` (`README.md`, `README_zh.md`) and `examples/packaging/` (`README.md`, `README_zh.md`).

### Changed

- Updated Lite6 Gripper pneumatic binary open/close dynamics and runtime configuration (`arm.kv: [300.0, 300.0, 200.0, 4.0, 8.0, 4.0]`, gripper `kp: 700.0, kv: 20.0, force_lower_n: -3.0, force_upper_n: 3.0`), driving only the active `finger_joint1` via `configure_lite6_gripper_drive` while `finger_joint2` follows passively, and setting `LITE6_GRIPPER_CONTACT_HOLD_BIAS_GAP_M = 0.018 m`.
- Updated gripper open/close viewer demos (`view_robot.py --gripper-demo`, `view_gripper_g2.py`, `view_bio_gripper_g2.py`, `view_lite6_gripper.py`) to close immediately on start and hold for 0.5 s after reaching each endpoint before toggling, and removed the hardcoded `1280x720` viewer resolution in `bio_g2_replay.py`.
- Standardized all Chinese documentation filenames to `README_zh.md` (`README_zh.md`, `examples/rl/**/README_zh.md`) and streamlined the top-level `examples/README*.md` guides.

### Fixed

- Fixed 7 kinematics and trajectory replay bugs: `matrix_to_rpy` gimbal-lock sign at `+π/2`, `preposition_cartesian` RPY `±π` wrap-around via `wrap_rpy_diff_rad`, `solve_link6_ik` world-frame naming and 2.0 mm FK residual check, `SafetyGate._build_timeline` `movel` sample-index offset after failed segments, `retime_ik_joint_stream` unconstrained `v_max <= 0` handling and `1_000_000` sample cap, `run_ik_validation` solver signature and multi-DOF `arm_dof_idx` slicing, and `mirror_executor` / `sim_executor` / `g2_replay` chained `init_qpos` seeding.
- Fixed 1D force-control tensor shapes in `configure_lite6_gripper_drive` and `set_lite6_gripper_pose` for single-environment viewers, and enabled headless open/close demo cycling for non-calibrated grippers in `robot_viewer.py`.
- Fixed `RealRobotSession.configure_for_dynamics` (`set_gravity_direction` state reset before `set_tcp_load`, plus `wait_for_simulation_robot` polling before `motion_enable` in `dynamics-sim-collision-check`), added a controller `tcp_offset` `z > 0` precheck in `dynamics-hardware-check` that aborts before any hardware configuration or motion when an attached tool would collide with the mounting surface, and added flange-to-TCP `tcp_offset` conversion in `XArmTransport` / `sdk_sim_feedback` for `sdk-sim` and real `servo_cartesian` execution.

## [0.3.2] — 2026-10-21

### Changed

- Share RL metrics, checkpoint manifests and cache setup in `ufactory.training`; reuse Reach state, action and episode accounting in its realtime environment while retaining independent success holds.
- Limit pick-place configuration to active v3 settings, centralize fixed scene constants, and reject unsupported overrides. Remove unused runtime YAML fields (`object_spawn_*`, `target_spawn_*`, `table_*`, `place_height_offset_m`, `arm.home_qpos_rad`) and legacy checkpoint compatibility shims (`allow_physics_transfer`, `allowed_runtime_env_mismatches`, `_PICK_PLACE_V3_UNUSED_KEYS`).
- Packaging and pick-place share manipulation backend construction, paced start holds, initial MoveJ priming, real G2 gripper sender setup, and execution summary formatting in `ufactory.manipulation.commands`.
- Consolidate G2 linkage geometry into `ufactory.grippers.g2` and inline Torch gap/drive conversions into `ufactory.simulation.g2_controller`; remove `scipy` from `ufactory.grippers.lite6` and drop the `[project.optional-dependencies].showcase` extra.
- Remove unused legacy modules (`ufactory/grippers/g2_torch.py`, `ufactory/visualization/bio_gripper_g2_viewer.py`, `ufactory/dynamics/poses_config.py`, `ufactory/dynamics/validation_service.py`, `ufactory/trajectory/real_executor.py`, `ufactory/manipulation/frames.py`, `ufactory/manipulation/packaging/safety.py`) and one-off scripts (`scripts/generate_showcase_textures.py`, `scripts/observe_hold_current.py`), and merge 9 redundant test modules into their canonical suites.
- GitHub Actions uses `project-check fast` as the single quality definition. Refresh bilingual guides, URDF package READMEs, and the asset manifest.
- Simplified the Gripper G2 simulation modules. `ufactory/simulation/g2.py` was merged into `g2_adapter.py`, and the task-level `g2_confirmation.py` and `g2_replay.py` moved to `ufactory/trajectory/`. Package-level exports from `ufactory.simulation` are unchanged.
- Pad contact forces now come from a single `object_finger_contact_forces_n`. Pass `left_normal`/`right_normal` to get the pad-normal clamping force used by the grasp gates; without them it returns the net force magnitude used by RL observations. Both values are unchanged.
- `G2Controller` is now the single Torch implementation of the calibrated servo (formerly `G2BatchController`); states and outputs are `(num_envs,)` tensors and `num_envs` is keyword-only. Single scenes run it on the CPU in float64 with `num_envs=1`; `force_lookup()` is a method returning the calibration lookup of one environment. A recorded 521-step trajectory from the former scalar controller is kept as a regression reference (float64 within 1e-10).
- Unified anti-slip rigid contact configuration across deterministic G2 pick-and-place, packaging, and RL training/inference using `DEFAULT_IMPRATIO = 100.0` in `ufactory/simulation/physics.py` and `G2_NOSLIP_ITERATIONS = 0` in `ufactory/config/models.py`, eliminating pyramidal friction creep without the 3x throughput penalty of post-solve no-slip iterations.
- Refactored `examples/rl/pick_place` (`env_version=3`) to pure end-to-end PPO with 70 observations and 4 continuous Cartesian and G2 opening increments (`cartesian_and_g2_gap_increment`), strict tabletop pressing (`press_mm <= 1.0 mm`), low-release (`0–3 mm` clearance), and collision-free hold acceptance criteria, plus `visualize.py` and `run_seeds.py` entry points.
- Training provenance in `examples/rl/pick_place` hashes `g2_adapter.py`, `contacts.py`, `metrics.py`, `runtime.py`, and `pick_place_config.py`; a test checks that every listed source exists.

### Removed

- The standalone Gripper G2 cube grasp and 33-case acceptance suite are no longer a public example. Calibrated G2 dynamics remain in `ufactory/simulation`, and visualization, pick-place, packaging, and RL examples are unchanged.
- Remove unused historical pick-place logic, custom policy/transfer, scenario-bank, acceptance-profile and Hugging Face modules with their dedicated tests; drop Hugging Face and Robotics Toolbox dependencies and synchronize the lock file.
- Remove the unused trajectory compiler/backend stubs, package-embedded asset backend, and 40 unreferenced assets (35 arm collision STL files, two Lite6 finger GLBs, and three textures; 3.80 MB). Current URDF variants and scalar/Torch geometry remain supported.
- Module paths `ufactory.simulation.g2`, `ufactory.simulation.g2_batch`, `ufactory.simulation.g2_confirmation`, `ufactory.simulation.g2_replay`, `ufactory.simulation.g2_scene` and `ufactory.simulation.g2_acceptance`, the class `G2BatchController`, and the helper `g2_adapter.normal_contact_forces`.
- Legacy `examples/rl/pick_place` bundled `g2_stable_v1_3_3` weights (`pretrained/`), scenario banks (`scenarios/`), and behavior-cloning/campaign helpers (`bc_gate.py`, `pretrain_bc.py`, `overnight.py`, `make_scenario_bank.py`, `monitor_utils.py`, `trace_utils.py`).

### Fixed

- Packaging GPU regressions accept natural high-release drops and rebound by checking that the final object collision bounds lie inside the box (1 mm contact tolerance). Remove the unused 25 mm center-distance setting; retain grasp, transport, release and return-home checks.
- Replaced single-hull arm `<collision>` meshes across all 33 registered URDF variants for `xarm5`, `xarm6`, `xarm7`, `lite6`, and `uf850` with 111 selective structure-aware watertight convex-hull STL meshes (`meshes/<prefix>/collision/<robot>_link*_col_*.stl`, `19 / 22 / 26 / 17 / 27` hulls per arm, `<= 128` faces per hull). This eliminates single-hull bridging across curved links (reducing outward P95 envelope inflation from `27–53 mm` to `2.3–7.2 mm`), resolves spurious `xarm5` pose-4 (`link3↔link5`), `xarm6` J5 `175°` (`link4↔link6`), and `lite6` `qpos0` (`link_base↔link6`) self-collisions and torque saturation, and preserves Genesis's purely convex `SupportField` + MPR fast path without significant GPU or RL overhead.

## [0.3.1] — 2026-10-15

### Added

- Added the `g2_calibrated_v1` Gripper G2 model: rated motor torque 0.22 N·m, peak 0.5 N·m, and holding force from the official current/speed curve. Commands are a 0–84 mm pad gap, speed 0–100%, and force 1–100%. `examples/gripper_g2` provides the free-cube grasp and the 33-case acceptance suite.
- Added a simulation-only bare-arm xArm 6 reach example under `examples/rl/reach`: a fixed 5 mm success threshold, joint-position increments, random end-effector targets, PPO training, checkpoint selection and evaluation. No trained weights are bundled, and there is no real-robot policy executor.

### Changed

- Updated joint torque limits for xArm5, xArm6, xArm7, UF850, and Lite6 so each runtime `effort_limits_nm` matches the default URDF. Lite6 joints 4–6 are 1.364505, 3.274812, and 1.364505 N·m. Those joints use PD gains `kp` 40/80/40 and `kv` 8/16/8; URDF damping and friction are 0 on joints 4 and 6 and 0.1 on joint 5.
- Pick-and-place, packaging, the dynamic viewers, and pick-and-place RL now drive `drive_joint` with the calibrated controller (`kp=40`, `kv=5`, `ki=40`, 0.01 kg·m² added inertia). Grasp and release continue on the next control cycle after two confirming frames; a timeout stops the following arm motion. All 11 standard G2 URDFs encode the nominal 0–84 mm stroke directly; six linkage joints use limits `0.057210812728026743–0.86509464599328212 rad` and velocity `2.608067454724 rad/s`. Scalar, Torch, adapter, preflight and mirror conversions use actual joint geometry. The `derive_urdf()` / `derive_g2_urdf` interfaces were removed.
- Bundled pick-and-place weights stay on `g2_stable_v1_3_3`. Evaluating them on the calibrated model requires `--allow-physics-transfer`.
- Set the Genesis World reference baseline to **1.4.4** while keeping the minimum at **1.4.2** (`genesis-world>=1.4.2`). Genesis 1.4.2 and 1.4.3 still run and emit one unvalidated-baseline warning. Genesis 1.4.1 and older fail closed. Private hook signatures used by forward kinematics, inverse kinematics, GLB PBR, and the deferred viewer are unchanged. The lock file pins Genesis 1.4.4 and Quadrants 1.3.3 (Genesis 1.4.2 still brings Quadrants 1.3.0). Contact-solver internals changed in 1.4.4, so simulation checks were re-run on 1.4.4; contact numbers are not claimed identical to 1.4.2.

### Fixed

- Importing the reach viewer no longer selects the GLX platform. That selection stays on the viewer command, so headless simulation checks can release the offscreen context.
- G2 pick-place and packaging commands now return failure when transport acceptance fails, even if timing and the final pose pass.
- Reach now accumulates position commands within URDF limits, maps the full action range linearly, and uses 24 observations with command tracking error. PPO uses a bounded Beta distribution and a one-time success bonus. Evaluation runs each fixed target once through success or timeout and can enforce success-rate and distance thresholds. The `visualize` entry provides real-time GLB inference with an independent 1-second hold after each successful reach; `evaluate --vis` prints its replacement command. Reach configurations and checkpoints with other success thresholds are rejected.
- A stationary joint seed whose forward pose sat a few micrometres off the Cartesian home is re-solved onto that home before preflight. The xArm7 `servo_cartesian` pick-and-place stream stays within the configured 0.8 m/s² Cartesian acceleration limit.
- Updated the asset manifest hash for `assets/configs/runtime/robots/lite6.yaml` so it matches the committed file.
- Replaced Lite6's link-wide, stage-specific collision exemptions with one finger-root geometry-pair exclusion shared by Genesis and Coal. Finger pads and external collisions retain the 5 mm safety margin. The filtering API supports Genesis 1.4.2 and later.
- Deterministic G2 runtime configurations now default to `simulation.noslip_iterations = 5` (`G2_NOSLIP_ITERATIONS`), eliminating pyramidal friction creep during grasp transport with the existing 5 mm slip and pick-and-place 10 mm placement gates. The CPU transport regression requires slip ≤1 mm. G2 tests now occupy seven files with a 22-case GPU suite (9 pick-place, 4 faults, 5 packaging, 2 RL, 2 demos). Batched `pick_place` RL keeps `simulation.noslip_iterations = 0` via `load_pick_place_rl_runtime_config` to preserve training throughput.

## [0.3.0] — 2026-09-24

### Changed

- Moved the official repository to [xArm-Developer/ufactory_genesis](https://github.com/xArm-Developer/ufactory_genesis). README, contributing guide, roadmap, security policy, and package URLs now point there. The personal-account preview wording is retired.
- Documented the v0.3.0 entry points without a compatibility promise. 0.x may rename or remove them before 1.0.0. Genesis World stays validated at 1.4.2 only.
- Packaging repetition is `--cycles` only. `--loop` and `--no-loop` are removed.

## [0.2.15] — 2026-09-24

### Changed

- Raised the minimum and reference Genesis World baseline to **1.4.2** (`genesis-world>=1.4.2`). Genesis 1.4.1 and older now fail closed with `GenesisCompatibilityError`. Private hook signatures used by forward kinematics, inverse kinematics, GLB PBR, and the deferred viewer are unchanged on 1.4.2. `RigidOptions.use_hibernation` stays off by default, and scenes still pass `iterations=100`, so the Gripper G2 contact profile is unchanged.

## [0.2.14] — 2026-09-20

### Added

- Added `ufactory.simulation.forward_kinematics(robot, qpos, envs_idx=None)` helper function wrapping `solver.forward_kinematics_query` to provide unified forward kinematics evaluation across all robot models.

### Changed

- Raised the minimum and reference Genesis World baseline to **1.4.1** (`genesis-world>=1.4.1`). Genesis versions below 1.4.1 now fail closed with `GenesisCompatibilityError`.
- Migrated forward kinematics call sites across verification, visualization, and validation modules to the unified `forward_kinematics` helper, adapting to Genesis 1.4.1's deprecation of `RigidEntity.forward_kinematics`.
- Derived rigid solver substeps dynamically via `SimOptions` instead of passing `dt` to `RigidOptions`, preventing substep count conflicts under Genesis 1.4.1 solver validation.
- Updated rigid link mass modification to use `set_links_mass` with backward-compatible fallback across packaging and RL environments.
- Ensured PyTorch default device resets cleanly upon Genesis runtime exit and after test fixtures.

### Removed

- Removed obsolete `ensure_ik_scratch` helper and associated private scratch memory allocation hooks, as kinematics scratch storage is now managed natively inside Genesis solvers.

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
