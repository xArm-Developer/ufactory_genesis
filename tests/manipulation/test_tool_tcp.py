"""Unit tests for unified ToolTcpSpec transforms and real-controller TCP consistency checks."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ufactory.config import load_runtime_config
from ufactory.manipulation.tcp import (
    DEFAULT_REAL_TCP_POS_TOL_MM,
    DEFAULT_REAL_TCP_ROT_TOL_DEG,
    ToolTcpSpec,
    check_controller_tcp_offset,
    resolve_tool_tcp_spec,
    tcp_offset_mm_deg_to_mm_rad,
    tcp_offset_mm_rad_to_mm_deg,
)
from ufactory.robots.runtime import LITE6_VACUUM_GRIPPER_PARAMS, VACUUM_GRIPPER_PARAMS


@pytest.mark.parametrize(
    ("robot", "gripper", "expected_grasp_flange_xyz"),
    [
        ("xarm6", None, (0.30, 0.00, 0.1886)),
        ("xarm5", None, (0.30, 0.00, 0.1886)),
        ("xarm7", None, (0.30, 0.00, 0.1886)),
        ("uf850", None, (0.30, 0.00, 0.1886)),
        ("lite6", None, (0.20, 0.00, 0.0963)),
        ("xarm6", "robotiq_2f_85", (0.30, 0.00, 0.1838)),
        ("xarm6", "robotiq_2f_140", (0.30, 0.00, 0.2561)),
        ("xarm6", "bio_gripper_g2", (0.38 - 0.15764, -0.12, 0.1595)),
    ],
)
def test_resolve_tool_tcp_spec_matches_expected_grasp_flange_waypoints(
    robot: str,
    gripper: str | None,
    expected_grasp_flange_xyz: tuple[float, float, float],
) -> None:
    cfg = load_runtime_config(robot, gripper=gripper)
    spec = resolve_tool_tcp_spec(cfg)
    obj = cfg.task.parameters["fixed_object_position_m"]
    flange = spec.tcp_to_flange_xyz([float(obj[0]), float(obj[1]), spec.grasp_tcp_z_m])
    np.testing.assert_allclose(flange, expected_grasp_flange_xyz, atol=1e-9)
    roundtrip = spec.flange_to_tcp_xyz(flange)
    np.testing.assert_allclose(roundtrip, [float(obj[0]), float(obj[1]), spec.grasp_tcp_z_m], atol=1e-9)


def test_tool_tcp_spec_zero_dof_and_deg_rad_conversion() -> None:
    vac = resolve_tool_tcp_spec(VACUUM_GRIPPER_PARAMS)
    assert vac.tcp_offset_mm_deg == (0.0, 0.0, 126.0, 0.0, 0.0, 0.0)
    assert vac.tool_tip_link == "vacuum_tcp"
    assert vac.tcp_offset_flange_m == pytest.approx((0.0, 0.0, 0.126))

    lite_vac = resolve_tool_tcp_spec(LITE6_VACUUM_GRIPPER_PARAMS)
    assert lite_vac.tcp_offset_mm_deg == (0.0, 0.0, 61.0, 0.0, 0.0, 0.0)
    assert lite_vac.tool_tip_link == "link_tcp"

    bare = resolve_tool_tcp_spec(None)
    assert bare == ToolTcpSpec()
    np.testing.assert_allclose(bare.tcp_to_flange_xyz([0.3, 0.1, 0.2]), [0.3, 0.1, 0.2], atol=1e-12)

    mm_deg = (157.64, 0.0, 59.5, 10.0, -20.0, 45.0)
    mm_rad = tcp_offset_mm_deg_to_mm_rad(mm_deg)
    np.testing.assert_allclose(mm_rad[:3], [157.64, 0.0, 59.5])
    np.testing.assert_allclose(mm_rad[3:], [math.radians(10.0), math.radians(-20.0), math.radians(45.0)])
    np.testing.assert_allclose(tcp_offset_mm_rad_to_mm_deg(mm_rad), mm_deg, atol=1e-9)


def test_check_controller_tcp_offset_tolerates_minor_diff_and_fails_closed_on_real(capsys) -> None:
    bio_yaml = (157.64, 0.0, 59.5, 0.0, 0.0, 0.0)
    bio_ctrl = (159.5, 0.0, 59.5, 0.0, 0.0, 0.0)
    pos_mm, rot_deg = check_controller_tcp_offset(bio_ctrl, bio_yaml, mode="real")
    assert pos_mm == pytest.approx(1.86, abs=1e-6)
    assert rot_deg == pytest.approx(0.0, abs=1e-6)
    assert "tcp_offset check PASS" in capsys.readouterr().out

    # Major position mismatch: YAML bare flange vs controller G2 (172 mm).
    g2_ctrl = (0.0, 0.0, 172.0, 0.0, 0.0, 0.0)
    bare_yaml = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(RuntimeError, match="real motion refused"):
        check_controller_tcp_offset(g2_ctrl, bare_yaml, mode="real")

    # sdk-sim warns instead of raising.
    pos_mm, rot_deg = check_controller_tcp_offset(g2_ctrl, bare_yaml, mode="sdk-sim")
    assert pos_mm == pytest.approx(172.0, abs=1e-6)
    assert "WARNING" in capsys.readouterr().out

    # Rotation mismatch > 2 deg fails closed in real mode.
    rot_ctrl = (157.64, 0.0, 59.5, 0.0, 0.0, math.radians(DEFAULT_REAL_TCP_ROT_TOL_DEG + 1.0))
    with pytest.raises(RuntimeError, match="real motion refused"):
        check_controller_tcp_offset(rot_ctrl, bio_yaml, mode="real")

    # Position mismatch just above 10 mm fails closed in real mode.
    edge_ctrl = (157.64, 0.0, 59.5 + DEFAULT_REAL_TCP_POS_TOL_MM + 0.1, 0.0, 0.0, 0.0)
    with pytest.raises(RuntimeError, match="real motion refused"):
        check_controller_tcp_offset(edge_ctrl, bio_yaml, mode="real")
