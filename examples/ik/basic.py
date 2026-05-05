"""Minimal IK example — no visualization, no PyBullet.

Available IK chains (single-arm only):
    "left_arm"     7 DOF   fr3_left_link0  → fr3_left_link8
    "right_arm"    7 DOF   fr3_right_link0 → fr3_right_link8

The relative-base joints (x, y, yaw between the two arm bases) are
deployment parameters — they configure the cell layout, but they are
not part of any IK chain or planning subgroup.

JOINT_GROUPS (indices into the full 17-DOF state vector):
    relative_base [0:3]    deployment-only — frozen for every plan
    right_arm     [3:10]
    left_arm      [10:17]
"""

import numpy as np

from bimanual_franka_planning.bimanual_franka import HOME_JOINTS, JOINT_GROUPS
from bimanual_franka_planning.kinematics import create_ik_solver
from bimanual_franka_planning.types import IKConfig, SE3Pose, SolveType

G = JOINT_GROUPS
HOME_LEFT_ARM = HOME_JOINTS[G["left_arm"]]
HOME_RIGHT_ARM = HOME_JOINTS[G["right_arm"]]


def main():
    # --- IKConfig (all fields shown with defaults) ---
    config = IKConfig(
        timeout=0.2,  # seconds per TRAC-IK attempt
        epsilon=1e-5,  # convergence tolerance
        solve_type=SolveType.SPEED,  # SPEED | DISTANCE | MANIP1 | MANIP2
        max_attempts=10,  # random restart attempts
        position_tolerance=1e-4,  # post-solve check (meters)
        orientation_tolerance=1e-4,  # post-solve check (radians)
    )

    # --- Create solver ---
    solver = create_ik_solver("left_arm", config=config)
    print(
        f"Chain: {solver.base_frame} -> {solver.ee_frame} ({solver.num_joints} joints)"
    )

    # Forward kinematics: get current end-effector pose at the home configuration
    home_pose = solver.fk(HOME_LEFT_ARM)
    print(f"Home EE position: {home_pose.position}")

    # Define a target pose: small offset, keep same orientation
    target = SE3Pose(
        position=home_pose.position + np.array([0.05, 0.0, -0.05]),
        rotation=home_pose.rotation,
    )

    # Solve IK (seed is optional; if None, uses random within joint limits)
    result = solver.solve(target, seed=HOME_LEFT_ARM)
    print(f"IK status: {result.status.value}")
    print(f"  position error:    {result.position_error:.6f} m")
    print(f"  orientation error: {result.orientation_error:.6f} rad")

    if result.success:
        print(f"  solution: {np.round(result.joint_positions, 4)}")

        # Verify with FK
        achieved = solver.fk(result.joint_positions)
        print(f"  achieved position: {np.round(achieved.position, 4)}")

    else:
        print("  IK failed to find a valid solution.")


if __name__ == "__main__":
    main()
