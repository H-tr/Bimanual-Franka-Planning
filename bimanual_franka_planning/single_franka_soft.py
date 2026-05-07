"""Single-arm soft-gripper Franka FR3 description.

A standalone FR3 with a custom **soft-rubber** gripper (two long
flexure-style fingers carved from ``soft_gripper_finger.stl``) in
place of the stock Franka parallel-jaw fingers.  Joint structure and
arm kinematics are identical to :mod:`single_franka` — only the
end-effector geometry changes, which is enough to require a
dedicated cricket-generated FK header
(``ext/ompl_vamp/robot/single_fr3_soft.hh``) because each finger
contributes a different number / placement of collision spheres.

State vector layout (9 DOF, identical to ``single_fr3``):

    [0:7]    fr3 arm        (fr3_joint1..7)
    [7]      gripper finger1   (fr3_finger_joint1, prismatic 0..0.04 m)
    [8]      gripper finger2   (fr3_finger_joint2, prismatic 0..0.04 m)

The URDFs and meshes live in
``resources/robot/single_fr3_soft/`` and are produced by
``tools/build_soft_fr3_urdfs.py`` from the existing single-arm FR3
URDFs.  Re-run that tool whenever you replace ``soft_gripper_finger.stl``
or change the finger mounting geometry.
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
_RESOURCES_DIR = os.path.join(_PKG_ROOT, "resources", "robot", "single_fr3_soft")

# Atomic joint groups — index slices into the full 9-DOF configuration.
JOINT_GROUPS = {
    "arm": slice(0, 7),
    "gripper": slice(7, 9),
}

CHAIN_CONFIGS: dict[str, ChainConfig] = {
    "single_fr3_soft": ChainConfig(
        base_link="fr3_link0",
        ee_link="fr3_link8",
        num_joints=7,
        urdf_path=os.path.join(_RESOURCES_DIR, "single_fr3_soft.urdf"),
    ),
}

VIZ_URDF_PATH = os.path.join(_RESOURCES_DIR, "single_fr3_soft.urdf")

# Per-joint TOTG limits — FR3 datasheet numbers for the arm, plus
# 0.2 m/s velocity and 1.0 m/s² acceleration for each gripper finger
# (matching the URDF ``<limit velocity>``).
_ARM_VELOCITY = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
_ARM_ACCEL = np.full(7, 6.0)
_GRIPPER_VELOCITY = np.array([0.2, 0.2])
_GRIPPER_ACCEL = np.array([1.0, 1.0])
MAX_VELOCITY = np.concatenate([_ARM_VELOCITY, _GRIPPER_VELOCITY])
MAX_ACCELERATION = np.concatenate([_ARM_ACCEL, _GRIPPER_ACCEL])

single_fr3_soft_robot_config = RobotConfig(
    urdf_path=os.path.join(_RESOURCES_DIR, "single_fr3_soft.urdf"),
    joint_names=[
        "fr3_joint1",
        "fr3_joint2",
        "fr3_joint3",
        "fr3_joint4",
        "fr3_joint5",
        "fr3_joint6",
        "fr3_joint7",
        "fr3_finger_joint1",
        "fr3_finger_joint2",
    ],
    camera=CameraConfig(
        link_name="fr3_link0",
        width=640,
        height=480,
        fov=60.0,
        near=0.1,
        far=10.0,
    ),
    max_velocity=MAX_VELOCITY,
    max_acceleration=MAX_ACCELERATION,
)

# Subgroups for partial planning (mirrors single_fr3).
_ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
_GRIPPER_JOINTS = ["fr3_finger_joint1", "fr3_finger_joint2"]
PLANNING_SUBGROUPS = {
    "single_fr3_soft_arm": {"dof": 7, "joints": _ARM_JOINTS},
    "single_fr3_soft_gripper": {"dof": 2, "joints": _GRIPPER_JOINTS},
}

# Franka "ready" stance — same neutral pose as single_fr3.  Gripper
# home is fully closed (q=0).
HOME_JOINTS = np.array(
    [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398, 0.0, 0.0]
)
