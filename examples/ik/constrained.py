"""Constrained IK example — Pink solver with collision avoidance.

Shows the two main features of the Pink backend on a single arm:
    1. Singularity-robust IK via Levenberg-Marquardt damping
    2. Collision avoidance (self-collision + pointcloud obstacles)

All solvers are created through the same ``create_ik_solver`` factory.

Usage:
    python examples/ik/constrained.py
"""

import os

import numpy as np

from bimanual_franka_planning.bimanual_franka import (
    CHAIN_CONFIGS,
    HOME_JOINTS,
    JOINT_GROUPS,
)
from bimanual_franka_planning.kinematics import create_ik_solver
from bimanual_franka_planning.kinematics.collision_model import (
    add_pointcloud_obstacles,
    build_collision_model,
)
from bimanual_franka_planning.kinematics.pink_ik_solver import PinkIKSolver
from bimanual_franka_planning.types import PinkIKConfig, SE3Pose

G = JOINT_GROUPS
HOME_LEFT_ARM = HOME_JOINTS[G["left_arm"]]


def basic_ik():
    """1. Basic Pink IK — same factory, same solve() interface as TRAC-IK."""
    print("=" * 60)
    print("1. Basic Pink IK (singularity-robust)")
    print("=" * 60)

    solver = create_ik_solver("left_arm", backend="pink")
    print(
        f"Chain: {solver.base_frame} -> {solver.ee_frame} "
        f"({solver.num_joints} joints)"
    )

    home_pose = solver.fk(HOME_LEFT_ARM)
    target = SE3Pose(
        position=home_pose.position + np.array([0.05, 0.0, -0.05]),
        rotation=home_pose.rotation,
    )

    # solve() returns IKResult — same interface as TracIKSolver
    result = solver.solve(target, seed=HOME_LEFT_ARM)
    print(f"Status: {result.status.value}")
    print(f"  position error:    {result.position_error:.6f} m")
    print(f"  orientation error: {result.orientation_error:.6f} rad")
    if result.success:
        print(f"  solution: {np.round(result.joint_positions, 4)}")
    print()


def collision_avoidance():
    """2. Collision avoidance — self-collision + pointcloud obstacles."""
    print("=" * 60)
    print("2. Collision avoidance (self + obstacles)")
    print("=" * 60)

    urdf_path = CHAIN_CONFIGS["left_arm"].urdf_path
    urdf_dir = os.path.dirname(urdf_path)
    srdf_path = os.path.join(urdf_dir, "bimanual_fr3.srdf")
    collision_ctx = build_collision_model(urdf_path, srdf_path=srdf_path)
    print(f"Self-collision pairs: {len(collision_ctx.collision_model.collisionPairs)}")

    # Obstacle cluster placed near (but not on) the home configuration
    obstacle_points = np.array(
        [
            [0.45, 0.30, 0.55],
            [0.47, 0.32, 0.57],
            [0.43, 0.28, 0.53],
        ]
    )
    n_obs = add_pointcloud_obstacles(collision_ctx, obstacle_points, radius=0.02)
    print(
        f"Added {n_obs} obstacle spheres "
        f"(total pairs: {len(collision_ctx.collision_model.collisionPairs)})"
    )

    config = PinkIKConfig(
        lm_damping=1e-3,
        self_collision=True,
        collision_pairs=5,
        collision_d_min=0.005,
        solver="proxqp",
        max_iterations=300,
    )

    solver = create_ik_solver("left_arm", backend="pink", config=config)
    assert isinstance(solver, PinkIKSolver)
    solver.set_collision_context(collision_ctx)

    home_pose = solver.fk(HOME_LEFT_ARM)
    target = SE3Pose(
        position=home_pose.position + np.array([0.05, 0.0, -0.05]),
        rotation=home_pose.rotation,
    )

    result = solver.solve_constrained(target, seed=HOME_LEFT_ARM)
    print(f"Status: {result.status.value}  ({result.iterations} iterations)")
    print(f"  position error:    {result.position_error:.6f} m")
    print(f"  orientation error: {result.orientation_error:.6f} rad")
    if result.success and result.joint_positions is not None:
        achieved = solver.fk(result.joint_positions)
        print(f"  achieved position: {np.round(achieved.position, 4)}")
    print()


def main():
    basic_ik()
    collision_avoidance()


if __name__ == "__main__":
    main()
