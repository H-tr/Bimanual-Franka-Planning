"""Visualize bimanual planning with open vs. closed grippers.

Boots PyBullet, plans three trajectories that demonstrate the gripper
joint as a planning DOF, and plays them back so you can SEE the
fingers open and close along the planned path.

The relative-base joints (``relative_base_x/y/yaw``) — i.e. the
right-arm-relative-to-left cell layout — are deployment parameters,
not planning variables.  They MUST stay at HOME for every waypoint.
We achieve that by using **subgroup planners**:

  * ``bimanual_fr3_dual_arm``     — 14 DOF, base + grippers frozen
  * ``bimanual_fr3_dual_gripper`` —  4 DOF, base + arms     frozen

Each subgroup planner takes a ``base_config`` that pins the inactive
joints to a chosen full-DOF stance — the C++ collision checker injects
those values around the active subset on every query, and
``embed_path`` re-injects them when re-expanding the path back to the
21-DOF configuration the visualizer renders.

Phase 1 — arm motion with CLOSED grippers
    Plan over both arms only.  Base and grippers are frozen at HOME
    (grippers q=0).

Phase 2 — arm motion with OPEN grippers
    Plan over both arms only.  Base frozen at HOME; grippers frozen
    at q≈0.04 (fully open).

Phase 3 — grasp cycle (open → reach → close)
    Three back-to-back subgroup plans:
      a. Dual-gripper plan: closed → open at HOME arm pose.
      b. Dual-arm plan:    HOME arms → sampled goal, grippers held open.
      c. Dual-gripper plan: open → closed at the goal arm pose.

Controls in each animation window:
    SPACE = play/pause   ←/→ = step   'n' = advance to the next demo

Usage:
    pixi run python examples/planning/gripper_motion_vis.py
    pixi run python examples/planning/gripper_motion_vis.py --time_limit 4
"""

from __future__ import annotations

import numpy as np
from fire import Fire

from bimanual_franka_planning.bimanual_franka import (
    HOME_JOINTS,
    JOINT_GROUPS,
    bimanual_fr3_robot_config,
)
from bimanual_franka_planning.envs.pybullet_env import PyBulletEnv
from bimanual_franka_planning.planning import create_planner
from bimanual_franka_planning.types import PlannerConfig

# OMPL's bounds check is strict at the upper limit; back off by 1e-4
# because the FK header rounds 0.04 to single precision (~0.0399999991).
GRIP_OPEN = 0.04 - 1e-4
GRIP_CLOSED = 0.0


def _hold_grippers(full_config: np.ndarray, value: float) -> np.ndarray:
    """Return a copy of a full 21-DOF config with both grippers pinned."""
    out = full_config.copy()
    out[JOINT_GROUPS["left_gripper"]] = value
    out[JOINT_GROUPS["right_gripper"]] = value
    return out


def _plan_arms_with_grippers_held(
    planner_name: str, time_limit: float, gripper_value: float, seed: int
) -> tuple[np.ndarray, str]:
    """Plan a dual-arm motion with the base AND both grippers frozen.

    Uses ``bimanual_fr3_dual_arm`` (14 DOF).  ``base_config`` carries
    the desired gripper value, so the frozen-joint injection keeps
    the grippers at ``gripper_value`` for every collision query and
    every embedded waypoint.
    """
    np.random.seed(seed)
    base = _hold_grippers(HOME_JOINTS, gripper_value)
    p = create_planner(
        "bimanual_fr3_dual_arm",
        config=PlannerConfig(planner_name=planner_name, time_limit=time_limit),
        base_config=base,
    )
    start = p.extract_config(base)
    goal = p.sample_valid()
    res = p.plan(start, goal)
    if not res.success or res.path is None:
        return np.empty((0, HOME_JOINTS.size)), f"failed: {res.status.value}"
    return p.embed_path(res.path), (
        f"{len(res.path)} waypoints in {res.planning_time_ns/1e6:.0f} ms"
    )


def _plan_grippers_only(
    planner_name: str,
    time_limit: float,
    base: np.ndarray,
    start_value: float,
    goal_value: float,
) -> np.ndarray:
    """Plan a dual-gripper motion with the base AND both arms frozen.

    Uses ``bimanual_fr3_dual_gripper`` (4 DOF).  ``base`` provides the
    full 21-DOF stance to freeze around — typically HOME, optionally
    with the arms at a goal pose for the closing segment of a grasp.
    """
    p = create_planner(
        "bimanual_fr3_dual_gripper",
        config=PlannerConfig(planner_name=planner_name, time_limit=time_limit),
        base_config=base,
    )
    start = np.full(p.num_dof, start_value)
    goal = np.full(p.num_dof, goal_value)
    res = p.plan(start, goal)
    if not res.success or res.path is None:
        raise RuntimeError(f"gripper plan failed: {res.status.value}")
    return p.embed_path(res.path)


def _plan_grasp_cycle(
    planner_name: str, time_limit: float, seed: int
) -> tuple[np.ndarray, str]:
    """Open grippers → arm reach → close grippers, with the base pinned."""
    np.random.seed(seed)

    # Segment 1: open grippers at HOME arm pose (base frozen by subgroup).
    seg1 = _plan_grippers_only(
        planner_name, time_limit, HOME_JOINTS, GRIP_CLOSED, GRIP_OPEN
    )

    # Segment 2: arm motion with grippers held open (base frozen).
    base_open = _hold_grippers(HOME_JOINTS, GRIP_OPEN)
    arms = create_planner(
        "bimanual_fr3_dual_arm",
        config=PlannerConfig(planner_name=planner_name, time_limit=time_limit),
        base_config=base_open,
    )
    arm_start = arms.extract_config(base_open)
    arm_goal = arms.sample_valid()
    res = arms.plan(arm_start, arm_goal)
    if not res.success or res.path is None:
        raise RuntimeError(f"arm reach failed: {res.status.value}")
    seg2 = arms.embed_path(res.path)

    # Segment 3: close grippers at the goal arm pose (base + arms frozen).
    base_at_goal = seg2[-1].copy()  # arms at goal, grippers still open
    seg3 = _plan_grippers_only(
        planner_name, time_limit, base_at_goal, GRIP_OPEN, GRIP_CLOSED
    )

    # Concatenate, dropping the duplicate seam waypoints.
    path = np.vstack([seg1, seg2[1:], seg3[1:]])
    info = (
        f"{len(path)} waypoints "
        f"(open {len(seg1)} + reach {len(seg2)-1} + close {len(seg3)-1})"
    )
    return path, info


def _assert_base_pinned(path: np.ndarray) -> None:
    """Sanity check: every waypoint must keep the relative_base at HOME."""
    base_slc = JOINT_GROUPS["relative_base"]
    drift = np.max(np.abs(path[:, base_slc] - HOME_JOINTS[base_slc]))
    assert drift < 1e-9, f"relative_base drifted by {drift:.3e}; subgroup misuse"


def main(
    planner_name: str = "rrtc", time_limit: float = 2.0, fps: float = 60.0
) -> None:
    env = PyBulletEnv(bimanual_fr3_robot_config, visualize=True)

    print("\n=== Phase 1 / 3: arm motion with CLOSED grippers ===")
    path1, info1 = _plan_arms_with_grippers_held(
        planner_name, time_limit, GRIP_CLOSED, seed=0
    )
    print(f"  {info1}")
    if len(path1):
        _assert_base_pinned(path1)
        env.animate_path(path1, fps=fps, next_key="n")

    print("\n=== Phase 2 / 3: arm motion with OPEN grippers ===")
    path2, info2 = _plan_arms_with_grippers_held(
        planner_name, time_limit, GRIP_OPEN, seed=1
    )
    print(f"  {info2}")
    if len(path2):
        _assert_base_pinned(path2)
        env.animate_path(path2, fps=fps, next_key="n")

    print("\n=== Phase 3 / 3: open → reach → close grasp cycle ===")
    path3, info3 = _plan_grasp_cycle(planner_name, time_limit, seed=2)
    print(f"  {info3}")
    if len(path3):
        _assert_base_pinned(path3)
        env.animate_path(path3, fps=fps)


if __name__ == "__main__":
    Fire(main)
