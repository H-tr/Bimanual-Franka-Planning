"""Single-arm soft-gripper Franka FR3 description.

A standalone FR3 with a custom **soft-rubber** gripper (two long
flexure-style fingers carved from ``soft_gripper_finger.stl``) in
place of the stock Franka parallel-jaw fingers.  Joint structure and
arm kinematics are identical to :mod:`single_franka` — only the
end-effector geometry changes, which is enough to require a
dedicated cricket-generated FK header
(``ext/ompl_vamp/robot/single_fr3_soft.hh``) because each finger
contributes a different number / placement of collision spheres.

State vector layout (7 DOF, identical to ``single_fr3``):

    [0:7]    fr3 arm    (fr3_joint1..7)

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

# Atomic joint group — index slice into the full 7-DOF configuration.
JOINT_GROUPS = {
    "arm": slice(0, 7),
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

# Per-joint TOTG limits — same FR3 datasheet numbers as single_fr3
# (the arm itself is unchanged; the gripper is fixed).
MAX_VELOCITY = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
MAX_ACCELERATION = np.full(7, 6.0)

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

# A single arm has no meaningful sub-group.
PLANNING_SUBGROUPS: dict[str, dict] = {}

# Franka "ready" stance — same neutral pose as single_fr3.
HOME_JOINTS = np.array([0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398])
