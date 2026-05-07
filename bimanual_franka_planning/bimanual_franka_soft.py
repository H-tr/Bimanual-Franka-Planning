"""Bimanual soft-gripper Franka FR3 description.

Same cell as :mod:`bimanual_franka` (left arm + relative 3-DOF
planar base + right arm) with the standard parallel-jaw fingers
replaced by a pair of soft-rubber fingers (``soft_gripper_finger.stl``).
Arm kinematics and the relative-base layout are unchanged — only
the per-arm gripper geometry differs, which is enough to require a
dedicated cricket FK header (``ext/ompl_vamp/robot/bimanual_fr3_soft.hh``).

Joint order in the 21-DOF state vector (matches the cricket-
generated FK header in ``ext/ompl_vamp/robot/bimanual_fr3_soft.hh``):

    [0:3]    relative_base   (x, y, yaw)
    [3:10]   right arm       (fr3_right_joint1..7)
    [10:12]  right gripper   (fr3_right_finger_joint1..2, prismatic)
    [12:19]  left arm        (fr3_left_joint1..7)
    [19:21]  left gripper    (fr3_left_finger_joint1..2, prismatic)

Both finger joints per gripper are exposed as independent DOFs.
By convention, keep finger_joint1 = finger_joint2 within each arm —
the URDF ``<mimic>`` enforces this in any consumer that honors it.

The URDFs and meshes live in
``resources/robot/bimanual_fr3_soft/`` and are produced by
``tools/build_soft_fr3_urdfs.py``.
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
_RESOURCES_DIR = os.path.join(_PKG_ROOT, "resources", "robot", "bimanual_fr3_soft")

# Atomic joint groups — identical layout to bimanual_franka.
JOINT_GROUPS = {
    "relative_base": slice(0, 3),
    "right_arm": slice(3, 10),
    "right_gripper": slice(10, 12),
    "left_arm": slice(12, 19),
    "left_gripper": slice(19, 21),
}

CHAIN_CONFIGS: dict[str, ChainConfig] = {
    "left_arm_soft": ChainConfig(
        base_link="fr3_left_link0",
        ee_link="fr3_left_link8",
        num_joints=7,
        urdf_path=os.path.join(_RESOURCES_DIR, "bimanual_fr3_soft.urdf"),
    ),
    "right_arm_soft": ChainConfig(
        base_link="fr3_right_link0",
        ee_link="fr3_right_link8",
        num_joints=7,
        urdf_path=os.path.join(_RESOURCES_DIR, "bimanual_fr3_soft.urdf"),
    ),
}

VIZ_URDF_PATH = os.path.join(_RESOURCES_DIR, "bimanual_fr3_soft_viz.urdf")

# Same FR3 datasheet limits as bimanual_franka, plus 0.2 m/s gripper
# finger limits (matching the URDF ``<limit velocity>``).
_FR3_VELOCITY = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
_FR3_ACCEL = np.full(7, 6.0)
_GRIPPER_VELOCITY = np.array([0.2, 0.2])
_GRIPPER_ACCEL = np.array([1.0, 1.0])
MAX_VELOCITY = np.concatenate(
    [
        np.array([0.5, 0.5, 1.0]),  # relative_base x, y, yaw
        _FR3_VELOCITY,  # right arm
        _GRIPPER_VELOCITY,  # right gripper
        _FR3_VELOCITY,  # left arm
        _GRIPPER_VELOCITY,  # left gripper
    ]
)
MAX_ACCELERATION = np.concatenate(
    [
        np.array([1.0, 1.0, 2.0]),  # relative_base
        _FR3_ACCEL,  # right arm
        _GRIPPER_ACCEL,  # right gripper
        _FR3_ACCEL,  # left arm
        _GRIPPER_ACCEL,  # left gripper
    ]
)

bimanual_fr3_soft_robot_config = RobotConfig(
    urdf_path=os.path.join(_RESOURCES_DIR, "bimanual_fr3_soft.urdf"),
    joint_names=[
        "relative_base_x",
        "relative_base_y",
        "relative_base_yaw",
        "fr3_right_joint1",
        "fr3_right_joint2",
        "fr3_right_joint3",
        "fr3_right_joint4",
        "fr3_right_joint5",
        "fr3_right_joint6",
        "fr3_right_joint7",
        "fr3_right_finger_joint1",
        "fr3_right_finger_joint2",
        "fr3_left_joint1",
        "fr3_left_joint2",
        "fr3_left_joint3",
        "fr3_left_joint4",
        "fr3_left_joint5",
        "fr3_left_joint6",
        "fr3_left_joint7",
        "fr3_left_finger_joint1",
        "fr3_left_finger_joint2",
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

# Per-arm subgroup definitions (mirroring bimanual_franka but with
# a "_soft" suffix so they don't collide with the original cell).
_LEFT_ARM_JOINTS = [f"fr3_left_joint{i}" for i in range(1, 8)]
_RIGHT_ARM_JOINTS = [f"fr3_right_joint{i}" for i in range(1, 8)]
_LEFT_GRIPPER_JOINTS = ["fr3_left_finger_joint1", "fr3_left_finger_joint2"]
_RIGHT_GRIPPER_JOINTS = ["fr3_right_finger_joint1", "fr3_right_finger_joint2"]
PLANNING_SUBGROUPS = {
    "bimanual_fr3_soft_left_arm": {"dof": 7, "joints": _LEFT_ARM_JOINTS},
    "bimanual_fr3_soft_right_arm": {"dof": 7, "joints": _RIGHT_ARM_JOINTS},
    "bimanual_fr3_soft_left_gripper": {"dof": 2, "joints": _LEFT_GRIPPER_JOINTS},
    "bimanual_fr3_soft_right_gripper": {"dof": 2, "joints": _RIGHT_GRIPPER_JOINTS},
    "bimanual_fr3_soft_dual_arm": {
        "dof": 14,
        "joints": _LEFT_ARM_JOINTS + _RIGHT_ARM_JOINTS,
    },
    "bimanual_fr3_soft_dual_gripper": {
        "dof": 4,
        "joints": _LEFT_GRIPPER_JOINTS + _RIGHT_GRIPPER_JOINTS,
    },
}

# Franka "ready" stance per arm.
_READY = np.array([0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398])

# Default cell layout: right arm 0.8 m to the -y side of the left arm.
_RELATIVE_BASE_HOME = np.array([0.0, -0.8, 0.0])

# Gripper home: fully closed (q=0), both finger joints equal.
_GRIPPER_HOME = np.array([0.0, 0.0])

HOME_JOINTS = np.concatenate(
    [
        _RELATIVE_BASE_HOME,
        _READY,
        _GRIPPER_HOME,  # right gripper
        _READY,
        _GRIPPER_HOME,  # left gripper
    ]
)
