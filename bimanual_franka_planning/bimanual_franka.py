"""The bimanual Franka FR3 robot's bundled description.

This module collects every concrete value that describes the one
robot this project ships: joint groupings, the home pose, the URDF
chains TRAC-IK and Pinocchio operate on, the VAMP planning subgroups,
and the top-level ``RobotConfig`` instance.

The dataclass *types* themselves live in
:mod:`bimanual_franka_planning.types.robot` — this file holds *values*
of those types.

Frame design (deployment / calibration)
---------------------------------------

The ``world`` link is co-located with ``fr3_left_link0`` — the LEFT
arm's base is the calibration anchor.  In the field you place the
left arm wherever you want the world frame to be; no extrinsic to
measure on that side.

The right arm's base is connected to ``world`` through a 3-DOF planar
chain (``relative_base_x`` → ``relative_base_y`` → ``relative_base_yaw``)
so the only extrinsic you ever need to set is the *relative* pose of
the right arm w.r.t. the left arm.  Pin those three joints via
``base_config`` to plan single-arm motion at a known cell layout, or
leave them active to let the planner search over the relative pose
(useful for cell-layout optimisation or whole-system motion).

Joint order in the 17-DOF state vector (matches cricket's URDF tree
traversal in ``ext/ompl_vamp/robot/bimanual_fr3.hh``):

    [0:3]    relative_base   (x, y, yaw)
    [3:10]   right arm       (fr3_right_joint1..7)
    [10:17]  left arm        (fr3_left_joint1..7)
"""

from __future__ import annotations

import os

import numpy as np

from bimanual_franka_planning.types.robot import (
    CameraConfig,
    ChainConfig,
    RobotConfig,
)

_PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
_RESOURCES_DIR = os.path.join(_PKG_ROOT, "resources", "robot", "bimanual_fr3")

# Atomic joint groups — index slices into the full 17-DOF configuration array.
# Order must match the cricket-generated FK header's joint_names list.
JOINT_GROUPS = {
    "relative_base": slice(0, 3),
    "right_arm": slice(3, 10),
    "left_arm": slice(10, 17),
}

CHAIN_CONFIGS: dict[str, ChainConfig] = {
    # Single-arm chains: TRAC-IK / Pinocchio operate on 7 DOFs from each
    # arm's link0 to the wrist flange.  The relative-base joints are NOT
    # part of any IK chain — they are deployment parameters, frozen for
    # every plan.  IK targets are expressed in each arm's own base frame.
    "left_arm": ChainConfig(
        base_link="fr3_left_link0",
        ee_link="fr3_left_link8",
        num_joints=7,
        urdf_path=os.path.join(_RESOURCES_DIR, "bimanual_fr3.urdf"),
    ),
    "right_arm": ChainConfig(
        base_link="fr3_right_link0",
        ee_link="fr3_right_link8",
        num_joints=7,
        urdf_path=os.path.join(_RESOURCES_DIR, "bimanual_fr3.urdf"),
    ),
}

VIZ_URDF_PATH = os.path.join(_RESOURCES_DIR, "bimanual_fr3_viz.urdf")

# Per-joint TOTG limits.  FR3 datasheet velocity limits are aggressive
# (joint 5/7 reach 5.26 rad/s); keep some headroom.  Relative-base
# joints get conservative limits since they're typically pinned, not
# planned over.  Order matches the 17-DOF state vector exactly.
_FR3_VELOCITY = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
_FR3_ACCEL = np.full(7, 6.0)
MAX_VELOCITY = np.concatenate(
    [
        np.array([0.5, 0.5, 1.0]),  # relative_base x, y, yaw
        _FR3_VELOCITY,  # right arm
        _FR3_VELOCITY,  # left arm
    ]
)
MAX_ACCELERATION = np.concatenate(
    [
        np.array([1.0, 1.0, 2.0]),  # relative_base
        _FR3_ACCEL,  # right arm
        _FR3_ACCEL,  # left arm
    ]
)

bimanual_fr3_robot_config = RobotConfig(
    urdf_path=os.path.join(_RESOURCES_DIR, "bimanual_fr3.urdf"),
    joint_names=[
        # [0:3]  relative base (planar)
        "relative_base_x",
        "relative_base_y",
        "relative_base_yaw",
        # [3:10] right arm
        "fr3_right_joint1",
        "fr3_right_joint2",
        "fr3_right_joint3",
        "fr3_right_joint4",
        "fr3_right_joint5",
        "fr3_right_joint6",
        "fr3_right_joint7",
        # [10:17] left arm
        "fr3_left_joint1",
        "fr3_left_joint2",
        "fr3_left_joint3",
        "fr3_left_joint4",
        "fr3_left_joint5",
        "fr3_left_joint6",
        "fr3_left_joint7",
    ],
    camera=CameraConfig(
        link_name="world_camera",
        width=640,
        height=480,
        fov=60.0,
        near=0.1,
        far=10.0,
    ),
    max_velocity=MAX_VELOCITY,
    max_acceleration=MAX_ACCELERATION,
)

# VAMP subgroup robot names for planning.  The relative-base joints
# (relative_base_x/y/yaw) are deployment parameters, not planning
# variables — they are always pinned by ``base_config`` to the cell
# layout you measured at calibration time.  These three subgroups
# cover every meaningful planning task on a bimanual arm cell.
_LEFT_ARM_JOINTS = [f"fr3_left_joint{i}" for i in range(1, 8)]
_RIGHT_ARM_JOINTS = [f"fr3_right_joint{i}" for i in range(1, 8)]
PLANNING_SUBGROUPS = {
    "bimanual_fr3_left_arm": {"dof": 7, "joints": _LEFT_ARM_JOINTS},
    "bimanual_fr3_right_arm": {"dof": 7, "joints": _RIGHT_ARM_JOINTS},
    "bimanual_fr3_dual_arm": {
        "dof": 14,
        "joints": _LEFT_ARM_JOINTS + _RIGHT_ARM_JOINTS,
    },
}

# Franka "ready" stance per arm: shoulder back, elbow folded, wrist
# pointing forward — the canonical neutral pose from franka_description.
_READY = np.array([0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398])

# Default cell layout: right arm 0.8 m to the -y side of the left arm,
# both facing forward.  Override via ``base_config`` to deploy at any
# other relative pose.
_RELATIVE_BASE_HOME = np.array([0.0, -0.8, 0.0])

HOME_JOINTS = np.concatenate(
    [
        _RELATIVE_BASE_HOME,  # relative_base x, y, yaw
        _READY,  # right arm joints 1-7
        _READY,  # left arm joints 1-7
    ]
)
