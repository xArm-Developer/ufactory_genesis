"""View any supported robot GLB visual model in Genesis."""

from __future__ import annotations

import argparse

from ufactory.robots.assembly import accessories, ft_sensors, mounting_plate_for
from ufactory.robots.paths import robot_visual_glb_urdf
from ufactory.robots.registry import get_robot_profile, robot_cli_choices
from ufactory.visualization.robot_viewer import run_glb_diagnose, run_glb_viewer


def main() -> None:
    parser = argparse.ArgumentParser(description="View robot GLB visual model")
    parser.add_argument("--robot", required=True, choices=robot_cli_choices())
    parser.add_argument(
        "--bio-gripper-g2",
        action="store_true",
        help="Load Bio Gripper G2 static visual combo URDF",
    )
    parser.add_argument(
        "--lite6-gripper",
        action="store_true",
        help="Load Lite6 parallel gripper visual combo URDF (lite6 only)",
    )
    parser.add_argument(
        "--lite6-vacuum-gripper",
        action="store_true",
        help="Load Lite6 vacuum gripper static visual combo URDF (lite6 only)",
    )
    parser.add_argument(
        "--gripper-g2",
        action="store_true",
        help="Load Gripper G2 visual combo URDF",
    )
    parser.add_argument(
        "--robotiq-2f-85",
        action="store_true",
        help="Load Robotiq 2F-85 visual combo URDF",
    )
    parser.add_argument(
        "--robotiq-2f-140",
        action="store_true",
        help="Load Robotiq 2F-140 visual combo URDF",
    )
    parser.add_argument(
        "--vacuum-gripper",
        action="store_true",
        help="Load xArm/UF850 vacuum gripper static visual combo URDF",
    )
    parser.add_argument(
        "--camera",
        choices=accessories(),
        default=None,
        help="Mount the camera stand (camera_stand) or a camera on it between the flange and the end effector "
        "(all robots; bare arm or any end effector except --bio-gripper-g2)",
    )
    parser.add_argument(
        "--ft-sensor",
        nargs="?",
        const="ft_sensor_ai1500",
        default=None,
        choices=tuple(ft_sensors()),
        help="Mount a six-axis F/T sensor on the tool flange (default: ft_sensor_ai1500; xArm5/6/7 and UF850)",
    )
    parser.add_argument(
        "--mounting-plate",
        action="store_true",
        help="Mount the robot's official aluminium mounting plate below link_base",
    )
    parser.add_argument(
        "--movable",
        action="store_true",
        help="Kept for compatibility (grippers always use per-link movable GLBs)",
    )
    parser.add_argument(
        "--gripper-demo",
        action="store_true",
        help="Cycle gripper open/close with a 0.5 s hold at each endpoint (requires a gripper flag)",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--position-mm", type=float, default=None)
    parser.add_argument("--speed-percent", type=float, default=50)
    parser.add_argument("--speed-mm-s", type=float, default=None)
    parser.add_argument("--force-percent", type=float, default=50)
    parser.add_argument("--diagnose", action="store_true", help="Headless STL/GLB link-pose diagnostic")
    parser.add_argument(
        "--pd",
        action="store_true",
        help="Joint motion demo (smooth ~0.873 rad/s waypoints; visual only, not stiff PD)",
    )
    parser.add_argument(
        "--show-tcp",
        action="store_true",
        help="Show red DH TCP debug marker on EE flange (default: hidden)",
    )
    parser.add_argument(
        "--show-base-axes",
        action="store_true",
        help="Show temporary robot-base RGB axes at origin (X=red, Y=green, Z=blue)",
    )
    parser.add_argument(
        "--backend",
        choices=("cpu", "gpu"),
        default="gpu",
        help="Genesis backend (use cpu without a supported GPU)",
    )
    args = parser.parse_args()

    profile = get_robot_profile(args.robot)
    accessory_flags = (
        args.bio_gripper_g2,
        args.gripper_g2,
        args.robotiq_2f_85,
        args.robotiq_2f_140,
        args.vacuum_gripper,
        args.lite6_gripper,
        args.lite6_vacuum_gripper,
    )
    if sum(accessory_flags) > 1:
        parser.error("Only one end-effector flag may be set at a time")
    if args.movable and not (
        args.gripper_g2 or args.robotiq_2f_85 or args.robotiq_2f_140 or args.lite6_gripper or args.bio_gripper_g2
    ):
        parser.error(
            "--movable requires --gripper-g2, --robotiq-2f-85, --robotiq-2f-140, --lite6-gripper, or --bio-gripper-g2"
        )
    if args.gripper_demo and not (
        args.gripper_g2 or args.robotiq_2f_85 or args.robotiq_2f_140 or args.lite6_gripper or args.bio_gripper_g2
    ):
        parser.error(
            "--gripper-demo requires --gripper-g2, --robotiq-2f-85, --robotiq-2f-140, --lite6-gripper, or --bio-gripper-g2"
        )
    if args.bio_gripper_g2 and not profile.supports_bio_gripper_g2:
        parser.error(f"{args.robot} does not support Bio Gripper G2")
    if args.gripper_g2 and not profile.supports_gripper_g2:
        parser.error(f"{args.robot} does not support Gripper G2")
    if args.robotiq_2f_85 and not profile.supports_robotiq_2f_85:
        parser.error(f"{args.robot} does not support Robotiq 2F-85")
    if args.robotiq_2f_140 and not profile.supports_robotiq_2f_140:
        parser.error(f"{args.robot} does not support Robotiq 2F-140")
    if args.vacuum_gripper and not profile.supports_vacuum_gripper:
        parser.error(f"{args.robot} does not support Vacuum Gripper")
    if args.lite6_gripper and not profile.supports_lite6_gripper:
        parser.error(f"{args.robot} does not support Lite6 Gripper")
    if args.lite6_vacuum_gripper and not profile.supports_lite6_vacuum_gripper:
        parser.error(f"{args.robot} does not support Lite6 Vacuum Gripper")
    end_effector = next(
        (
            name
            for flag, name in zip(
                accessory_flags,
                (
                    "bio_gripper_g2",
                    "gripper_g2",
                    "robotiq_2f_85",
                    "robotiq_2f_140",
                    "vacuum_gripper",
                    "lite6_gripper",
                    "lite6_vacuum_gripper",
                ),
                strict=True,
            )
            if flag
        ),
        None,
    )
    if args.ft_sensor and not profile.supports_ft_sensor(args.ft_sensor):
        parser.error(f"--ft-sensor {args.ft_sensor} is not supported on {args.robot}")
    if args.camera and not profile.supports_accessory(args.camera, end_effector):
        parser.error(f"--camera {args.camera} does not fit {args.robot} with {end_effector or 'no end effector'}")

    if args.diagnose:
        run_glb_diagnose(
            profile,
            with_gripper_g2=args.gripper_g2,
            with_robotiq_2f_85=args.robotiq_2f_85,
            with_robotiq_2f_140=args.robotiq_2f_140,
            backend=args.backend,
        )
        return

    urdf_path = robot_visual_glb_urdf(
        args.robot,
        with_bio_gripper_g2=args.bio_gripper_g2,
        with_gripper_g2=args.gripper_g2,
        with_robotiq_2f_85=args.robotiq_2f_85,
        with_robotiq_2f_140=args.robotiq_2f_140,
        with_vacuum_gripper=args.vacuum_gripper,
        with_lite6_gripper=args.lite6_gripper,
        with_lite6_vacuum_gripper=args.lite6_vacuum_gripper,
        movable=args.movable,
        accessory=args.camera,
        mounting_plate=args.mounting_plate,
        ft_sensor=args.ft_sensor,
    )
    floor_z = -mounting_plate_for(profile.key).thickness_m if args.mounting_plate else 0.0
    print(f"Loading: {urdf_path}")
    run_glb_viewer(
        profile,
        urdf_path,
        headless=args.headless,
        pd_demo=args.pd,
        gripper_demo=args.gripper_demo,
        position_mm=args.position_mm,
        speed_percent=args.speed_percent,
        speed_mm_s=args.speed_mm_s,
        force_percent=args.force_percent,
        show_tcp=args.show_tcp,
        show_base_axes=args.show_base_axes,
        backend=args.backend,
        floor_z=floor_z,
    )


if __name__ == "__main__":
    main()
