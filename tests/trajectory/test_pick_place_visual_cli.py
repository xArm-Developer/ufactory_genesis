"""CLI argument checks for ufactory-pick-place --visual."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import numpy as np

import pytest

from ufactory.cli import pick_place
from ufactory.safety import PreflightCheck, SafetyViolation, ViolationType
from ufactory.visualization.viewer import start_deferred_viewer


def test_visual_compatibility_fails_before_hardware_connection(monkeypatch):
    connected = False

    def _reject_capabilities(**_kwargs):
        raise RuntimeError("Genesis hooks changed")

    def _unexpected_connect(_ip):
        nonlocal connected
        connected = True
        pytest.fail("hardware connection must not be attempted")

    monkeypatch.setattr(pick_place, "require_genesis_capabilities", _reject_capabilities)
    monkeypatch.setattr(pick_place, "_connect", _unexpected_connect)

    with pytest.raises(RuntimeError, match="Genesis hooks changed"):
        pick_place.main(
            [
                "--robot",
                "xarm6",
                "--mode",
                "real",
                "--executor",
                "servo_j",
                "--visual",
                "--ip",
                "192.0.2.1",
                "--calibration",
                "/tmp/not-read-before-compatibility.yaml",
                "--confirm-real",
            ]
        )
    assert connected is False


@pytest.mark.parametrize(
    ("cli_name", "argv"),
    [
        ("pick_place", ["--robot", "xarm6", "--mode", "dry-run", "--executor", "servo_j"]),
        (
            "pick_place",
            ["--robot", "xarm6", "--gripper", "bio_gripper_g2", "--mode", "dry-run", "--executor", "servo_j"],
        ),
        ("packaging", ["--robot", "xarm6", "--mode", "dry-run", "--executor", "servo_j"]),
    ],
)
def test_headless_servo_j_skips_genesis_for_pick_place_and_packaging(monkeypatch, tmp_path, cli_name, argv):
    pytest.importorskip("pinocchio")
    from ufactory.cli import packaging
    import ufactory.simulation as simulation_pkg

    def _forbid_genesis(*_args, **_kwargs):
        raise AssertionError("Genesis must not be checked or initialized for headless servo_j")

    monkeypatch.setattr(pick_place, "require_genesis_capabilities", _forbid_genesis)
    monkeypatch.setattr(pick_place, "GenesisRuntimeManager", _forbid_genesis)
    monkeypatch.setattr(packaging, "require_genesis_capabilities", _forbid_genesis)
    monkeypatch.setattr(simulation_pkg, "GenesisRuntimeManager", _forbid_genesis)

    report = tmp_path / f"{cli_name}_servo_j.json"
    module = pick_place if cli_name == "pick_place" else packaging
    rc = module.main([*argv, "--report", str(report)])
    assert rc == 0
    assert report.is_file()


def test_visual_rejected_for_dry_run(capsys):
    with pytest.raises(SystemExit) as excinfo:
        pick_place.main(
            [
                "--robot",
                "xarm6",
                "--mode",
                "dry-run",
                "--executor",
                "servo_j",
                "--visual",
            ]
        )
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "--visual is only supported with --mode sim or --mode real" in err


def test_visual_rejected_for_sdk_sim(capsys):
    with pytest.raises(SystemExit) as excinfo:
        pick_place.main(
            [
                "--robot",
                "xarm6",
                "--mode",
                "sdk-sim",
                "--executor",
                "servo_j",
                "--visual",
                "--ip",
                "127.0.0.1",
                "--calibration",
                "/tmp/fake.yaml",
            ]
        )
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "--visual is only supported with --mode sim or --mode real" in err


def test_visual_start_hold_must_be_non_negative(capsys):
    with pytest.raises(SystemExit) as excinfo:
        pick_place.main(
            [
                "--robot",
                "xarm6",
                "--mode",
                "real",
                "--executor",
                "servo_j",
                "--visual",
                "--visual-start-hold-s",
                "-0.1",
            ]
        )
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "--visual-start-hold-s must be non-negative" in err


def test_preflight_failures_are_grouped_without_losing_counts(capsys):
    violations = (
        SafetyViolation(
            ViolationType.CLEARANCE,
            "descend",
            "unsafe",
            sample_index=4,
            link="finger/object",
            actual=0.004,
            limit=0.005,
        ),
        SafetyViolation(
            ViolationType.CLEARANCE,
            "descend",
            "unsafe",
            sample_index=5,
            link="finger/object",
            actual=0.003,
            limit=0.005,
        ),
        SafetyViolation(ViolationType.ACCELERATION, "program", "unsafe", actual=13.0, limit=12.0),
    )
    pick_place._print_preflight_violation_summary(SimpleNamespace(violations=violations), None)
    err = capsys.readouterr().err
    assert "total=3 groups=2" in err
    assert "finger/object count=2 samples=4-5 min_distance_m=0.003 threshold_m=0.005" in err
    assert "rerun with --report PATH" in err


def test_preparation_progress_logs_include_samples_margin_and_timings(capsys):
    segment = SimpleNamespace(
        kind="movej",
        q_start=(0.0,) * 6,
        samples=lambda _rate: (None, 2),
    )
    program = SimpleNamespace(rate=50.0, segments=(segment,))
    config = SimpleNamespace(safety=SimpleNamespace(min_collision_distance_m=0.005))
    preflight = SimpleNamespace(
        passed=True,
        checks=(
            PreflightCheck("collision", True, 3, 625.0),
            PreflightCheck("total", True, 3, 675.0),
        ),
    )

    pick_place._print_ik_compile_complete(program, elapsed_s=7.25)
    pick_place._print_preflight_start(program, config)
    pick_place._print_preflight_complete(preflight)

    assert capsys.readouterr().out.splitlines() == [
        "[ik-compile] complete samples=3 elapsed_s=7.25",
        "[preflight] checking samples=3 collision_margin_mm=5.0...",
        "[preflight] complete status=PASS collision_s=0.625 total_s=0.675",
    ]


def test_kinematic_mirror_disables_existing_viewer_scene_pacer():
    viewer = SimpleNamespace(realtime_factor=1.0)
    scene = SimpleNamespace(visualizer=SimpleNamespace(viewer=viewer))

    start_deferred_viewer(scene, kinematic_mirror=True)

    assert viewer.realtime_factor is None


def test_open_sim_viewer_warms_then_start_deferred_viewer(monkeypatch):
    calls: list[str] = []
    scene = SimpleNamespace(step=lambda: calls.append("step"))

    monkeypatch.setattr(
        "ufactory.visualization.start_deferred_viewer",
        lambda target, **kwargs: calls.append(f"start_deferred_viewer:{target is scene}:{kwargs}"),
    )

    pick_place._open_sim_viewer(SimpleNamespace(scene=scene), warmup_steps=3)

    assert calls[:3] == ["step", "step", "step"]
    assert calls[3].startswith("start_deferred_viewer:True:")


@pytest.mark.parametrize(
    ("robot", "transport_passed", "expected_rc"),
    [("lite6", False, 0), ("xarm6", True, 0), ("xarm6", False, 1)],
)
def test_sim_visual_defers_viewer_and_checks_acceptance(monkeypatch, robot, transport_passed, expected_rc):
    calls: list[str] = []
    scene = SimpleNamespace(step=lambda: calls.append("step"))
    ctx = SimpleNamespace(scene=scene, robot_urdf_path="/tmp/urdf", home_qpos=np.zeros(1), arm_dof_idx=[0])

    monkeypatch.setattr(pick_place, "require_genesis_capabilities", lambda **_kwargs: calls.append("capabilities"))
    monkeypatch.setattr(
        pick_place,
        "load_runtime_config",
        lambda *_args, **_kwargs: SimpleNamespace(
            robot=SimpleNamespace(key=robot),
            gripper=SimpleNamespace(adapter="lite6" if robot == "lite6" else "g2"),
            motion=SimpleNamespace(rate_hz=50.0),
            simulation=SimpleNamespace(
                show_viewer=False, backend="gpu", precision="32", seed=1, g2=SimpleNamespace(speed_percent=50)
            ),
            task=SimpleNamespace(parameters={}, allowed_contacts={}),
        ),
    )
    monkeypatch.setattr(
        pick_place,
        "_print_summary",
        lambda _config: None,
    )
    monkeypatch.setattr(pick_place, "_model_and_hashes", lambda *_args, **_kwargs: (Path("/tmp/urdf"), "u", "c"))
    monkeypatch.setattr(pick_place, "sha256_file", lambda *_args: "u")
    monkeypatch.setattr(pick_place, "_backends", lambda *_args, **_kwargs: (object(), object()))
    monkeypatch.setattr(pick_place, "_build_program", lambda *_args, **_kwargs: SimpleNamespace(segments=()))
    monkeypatch.setattr("ufactory.trajectory.g2_confirmation.prepare_g2_program", lambda program, _settings: program)
    monkeypatch.setattr(pick_place, "_json_sha256", lambda _value: "scene")
    monkeypatch.setattr(
        pick_place,
        "create_safety_gate",
        lambda *_args, **_kwargs: SimpleNamespace(
            preflight=lambda _program, executor: SimpleNamespace(
                passed=True,
                violations=(),
                program_sha256="p",
                to_dict=lambda: {},
            ),
            approve_simulation=lambda program, preflight: program,
        ),
    )

    class _Runtime:
        def __init__(self, _config):
            pass

        def __enter__(self):
            calls.append("genesis_enter")
            return self

        def __exit__(self, *_exc):
            calls.append("genesis_exit")
            return False

    monkeypatch.setattr(pick_place, "GenesisRuntimeManager", _Runtime)

    def _build_ik_scene(_config, *, calibration=None, show_viewer=False, **_kwargs):
        calls.append(f"build_scene:show_viewer={show_viewer}")
        return ctx

    monkeypatch.setattr(pick_place, "_build_ik_scene", _build_ik_scene)
    monkeypatch.setattr(
        pick_place,
        "_open_sim_viewer",
        lambda target, **_kwargs: calls.append(f"open_sim_viewer:{target is ctx}"),
    )
    monkeypatch.setattr(
        pick_place,
        "execute_sim",
        lambda _approved, runner: (
            calls.append("execute_sim")
            or SimpleNamespace(
                place_error_mm=1.0,
                home_drift_mm=1.0,
                metric_settle_ticks=1,
                completed=True,
                timing_passed=True,
                transport_passed=transport_passed,
                failure_reason=None,
            )
        ),
    )
    monkeypatch.setattr(
        pick_place,
        "_hold_viewer",
        lambda target, **_kwargs: calls.append(f"hold_viewer:{target is ctx}"),
    )
    monkeypatch.setattr(
        "ufactory.trajectory.sim_executor.replay_sim",
        lambda *_args, **_kwargs: calls.append("replay_sim"),
    )
    monkeypatch.setattr(pick_place, "_write_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        pick_place,
        "_print_preflight_start",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        pick_place,
        "_print_preflight_complete",
        lambda *_args, **_kwargs: None,
    )

    rc = pick_place.main(
        [
            "--robot",
            robot,
            "--mode",
            "sim",
            "--executor",
            "servo_cartesian",
            "--visual",
        ]
    )

    assert rc == expected_rc
    assert "build_scene:show_viewer=False" in calls
    assert calls.index("open_sim_viewer:True") < calls.index("execute_sim")
    assert calls.index("execute_sim") < calls.index("hold_viewer:True")


def test_hold_viewer_runs_until_window_closes_and_paces_mirror(monkeypatch, capsys):
    alive_checks = 0
    step_calls = 0
    sleeps: list[float] = []

    class Viewer:
        def is_alive(self) -> bool:
            nonlocal alive_checks
            alive_checks += 1
            return alive_checks <= 3

    def on_step() -> None:
        nonlocal step_calls
        step_calls += 1

    scene = SimpleNamespace(visualizer=SimpleNamespace(viewer=Viewer()))
    monkeypatch.setattr(pick_place.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(pick_place.time, "sleep", lambda seconds: sleeps.append(seconds))

    pick_place._hold_viewer(SimpleNamespace(scene=scene), on_step=on_step)

    assert step_calls == 3
    assert sleeps == pytest.approx([1.0 / 30.0] * 3)
    assert "Viewer open. Press Ctrl+C to exit..." in capsys.readouterr().out


def test_mirror_visual_update_drops_frame_when_render_lock_is_busy():
    pytest.importorskip("torch")
    pytest.importorskip("genesis")
    from ufactory.trajectory.mirror_executor import update_scene_visualizer

    class BusyLock:
        def acquire(self, *, blocking: bool) -> bool:
            assert blocking is False
            return False

    context_updates: list[bool] = []
    pyrender_viewer = SimpleNamespace(render_lock=BusyLock(), update_on_sim_step=lambda: None)
    viewer = SimpleNamespace(
        _pyrender_viewer=pyrender_viewer,
        context=SimpleNamespace(update=lambda **kwargs: context_updates.append(kwargs["force_render"])),
    )
    scene = SimpleNamespace(visualizer=SimpleNamespace(viewer=viewer))

    assert update_scene_visualizer(scene) is False
    assert context_updates == []


def test_mirror_visual_update_supports_genesis_1_4_7_rigid_geoms_t():
    pytest.importorskip("torch")
    pytest.importorskip("genesis")
    from ufactory.trajectory.mirror_executor import update_scene_visualizer
    import numpy as np

    class FreeLock:
        def acquire(self, *, blocking: bool) -> bool:
            assert blocking is False
            return True

        def release(self) -> None:
            pass

    geom_0 = SimpleNamespace(uid="g0", idx=0)
    geom_1 = SimpleNamespace(uid="g1", idx=1)
    solver = SimpleNamespace()  # No _vgeoms_render_T (Genesis 1.4.7+).
    entity_a = SimpleNamespace(
        _morph=SimpleNamespace(enable_custom_vverts=False),
        _solver=solver,
        surface=SimpleNamespace(vis_mode="visual"),
        vgeoms=[geom_0],
        geoms=[],
    )
    entity_b = SimpleNamespace(
        _morph=SimpleNamespace(enable_custom_vverts=False),
        _solver=solver,
        surface=SimpleNamespace(vis_mode="visual"),
        vgeoms=[geom_1],
        geoms=[],
    )
    t0 = np.eye(4, dtype=np.float32)
    t0[0, 3] = 0.25
    t1 = np.eye(4, dtype=np.float32)
    t1[1, 3] = -0.15
    # Genesis 1.4.7 rigid_geoms_T layout: (n_rendered_envs, n_geoms, 4, 4).
    geoms_t = np.stack([t0, t1], axis=0)[None, ...]
    rigid_calls: list[tuple[object, bool]] = []

    def rigid_geoms_t(s, is_visual: bool):
        rigid_calls.append((s, is_visual))
        return geoms_t

    node_0 = SimpleNamespace(mesh=SimpleNamespace(_bounds="old", primitives=[SimpleNamespace(poses=None)]))
    node_1 = SimpleNamespace(mesh=SimpleNamespace(_bounds="old", primitives=[SimpleNamespace(poses=None)]))
    uploaded: list[tuple[object, str, np.ndarray]] = []
    context = SimpleNamespace(
        visualizer=SimpleNamespace(update_visual_states=lambda **kwargs: None),
        _scene=SimpleNamespace(_bounds="old"),
        rigid_nodes={"g0": node_0, "g1": node_1},
        rendered_envs_idx=[0],
        rigid_geoms_T=rigid_geoms_t,
        _get_geom_active_envs_idx=lambda geom, rendered: list(rendered),
        jit=SimpleNamespace(update_buffer=lambda node, name, arr: uploaded.append((node, name, arr))),
        update=lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not fall back to full update")),
    )
    viewer = SimpleNamespace(
        _pyrender_viewer=SimpleNamespace(render_lock=FreeLock(), update_on_sim_step=lambda: None),
        context=context,
        is_alive=lambda: True,
    )
    scene = SimpleNamespace(visualizer=SimpleNamespace(viewer=viewer))

    assert update_scene_visualizer(scene, entities=(entity_a, entity_b)) is True
    assert rigid_calls == [(solver, True)]
    assert len(uploaded) == 2
    np.testing.assert_allclose(node_0.mesh.primitives[0].poses[0], t0)
    np.testing.assert_allclose(node_1.mesh.primitives[0].poses[0], t1)
    np.testing.assert_allclose(uploaded[0][2][0], t0.T)
    np.testing.assert_allclose(uploaded[1][2][0], t1.T)


def test_mirror_visual_update_falls_back_to_context_update_for_genesis_1_5():
    pytest.importorskip("torch")
    pytest.importorskip("genesis")
    from ufactory.trajectory.mirror_executor import update_scene_visualizer

    class FreeLock:
        def acquire(self, *, blocking: bool) -> bool:
            return True

        def release(self) -> None:
            pass

    fallback_calls: list[bool] = []
    # Genesis 1.5+ removes context.rigid_nodes and visualizer.update_visual_states.
    context = SimpleNamespace(
        visualizer=SimpleNamespace(),
        update=lambda **kwargs: fallback_calls.append(kwargs["force_render"]),
    )
    viewer = SimpleNamespace(
        _pyrender_viewer=SimpleNamespace(render_lock=FreeLock(), update_on_sim_step=lambda: None),
        context=context,
        is_alive=lambda: True,
    )
    scene = SimpleNamespace(visualizer=SimpleNamespace(viewer=viewer))

    assert update_scene_visualizer(scene, entities=(SimpleNamespace(),)) is True
    assert fallback_calls == [True]
