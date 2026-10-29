"""Dependency-free contract for the end-to-end pick-place policy."""

PICK_PLACE_ENV_VERSION = 3
PICK_PLACE_NUM_OBS = 70
PICK_PLACE_NUM_ACTIONS = 4
PICK_PLACE_CONTROL = "cartesian_and_g2_gap_increment"
PICK_PLACE_ACTION_CONTRACT = "cartesian_and_g2_gap_delta_v3"

# Fixed scene in robot-base coordinates, shared by configuration and metrics.
SPAWN_X, SPAWN_Y = 0.300, 0.000
SPAWN_XY_LOWER = (0.300, -0.200)
SPAWN_XY_UPPER = (0.500, 0.200)
TARGET_X, TARGET_Y = 0.300, 0.300
CUBE_SIZE = 0.030
CUBE_REST_Z = CUBE_SIZE / 2
SAFE_LIFT_Z = 0.080
HOME_EE_POS = (0.30, 0.0, 0.30)
TABLE_SIZE = (0.60, 0.90, 0.04)
TABLE_POS = (0.35, 0.15, -0.02)
TARGET_PAD_SIZE = (0.05, 0.05, 0.001)
TARGET_PAD_POS = (TARGET_X, TARGET_Y, 0.0005)
PICK_PLACE_ROBOT = "xarm6"
PICK_PLACE_URDF = "xarm6_with_gripper.urdf"
