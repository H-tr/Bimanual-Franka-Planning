"""Gripper motion planning — driving the parallel-jaw fingers as DOFs.

Each FR3 gripper now contributes two prismatic finger joints to the
configuration vector (single_fr3: 7 + 2 = 9 DOF; bimanual_fr3: 17 + 4 =
21 DOF).  ``finger_joint2`` is a URDF mimic of ``finger_joint1`` — the
planner exposes both, but by convention you should keep them equal so
the gripper opens symmetrically.

This example walks through three idiomatic ways to drive the gripper
through configuration space:

  1. Whole-body plan: arm stays put, gripper closes → opens.
  2. Combined plan: arm reaches a new pose AND gripper opens at the
     same time.
  3. Gripper-only subgroup planner: plan exclusively over the 4-DOF
     bimanual gripper space with the rest of the body frozen.

Usage:
    pixi run python examples/planning/gripper_motion.py
    pixi run python examples/planning/gripper_motion.py --robot single_fr3
"""

from __future__ import annotations

import numpy as np
from fire import Fire

from bimanual_franka_planning.planning import create_planner
from bimanual_franka_planning.types import PlannerConfig

# Travel range for finger_joint1 / finger_joint2 in the URDF.  We back
# off by 1e-4 because OMPL's bounds check is strict at the upper limit
# while the FK header rounds 0.04 to single precision (~0.0399999991).
GRIPPER_OPEN = 0.04 - 1e-4
GRIPPER_CLOSED = 0.0


def _set_gripper(q: np.ndarray, slc: slice, value: float) -> np.ndarray:
    """Set both finger joints in ``q`` to ``value`` (mimic convention)."""
    out = q.copy()
    out[slc] = value
    return out


def demo_single_fr3(time_limit: float) -> None:
    from bimanual_franka_planning.single_franka import HOME_JOINTS, JOINT_GROUPS

    p = create_planner(
        "single_fr3",
        config=PlannerConfig(planner_name="rrtc", time_limit=time_limit),
    )
    print(f"\n[single_fr3] {p.num_dof}-DOF; gripper slice = {JOINT_GROUPS['gripper']}")

    # 1. Gripper-only motion: arm fixed at HOME, fingers go closed → open.
    closed = HOME_JOINTS.copy()  # HOME has gripper = 0 (closed)
    open_ = _set_gripper(closed, JOINT_GROUPS["gripper"], GRIPPER_OPEN)
    res = p.plan(closed, open_)
    print(
        f"  close → open: success={res.success}  waypoints={(len(res.path) if res.path is not None else 0)}"
    )
    if res.success:
        d_arm = np.max(np.abs(res.path[-1, :7] - res.path[0, :7]))
        d_grip = np.max(np.abs(res.path[-1, 7:9] - res.path[0, 7:9]))
        print(f"    Δarm = {d_arm:.4f} rad,  Δgripper = {d_grip:.4f} m")

    # 2. Combined arm + gripper move (sample a random arm goal, open grip).
    np.random.seed(0)
    goal = p.sample_valid()
    goal[JOINT_GROUPS["gripper"]] = GRIPPER_OPEN  # override sampled grip
    res = p.plan(closed, goal)
    print(
        f"  arm-pose change + open grip: success={res.success}  "
        f"waypoints={(len(res.path) if res.path is not None else 0)}"
    )


def demo_bimanual_fr3(time_limit: float) -> None:
    from bimanual_franka_planning.bimanual_franka import (
        HOME_JOINTS,
        JOINT_GROUPS,
    )

    p = create_planner(
        "bimanual_fr3",
        config=PlannerConfig(planner_name="rrtc", time_limit=time_limit),
    )
    print(
        f"\n[bimanual_fr3] {p.num_dof}-DOF; "
        f"left_gripper={JOINT_GROUPS['left_gripper']}, "
        f"right_gripper={JOINT_GROUPS['right_gripper']}"
    )

    # 1. Both grippers closed → both open, arms fixed at HOME.
    start = HOME_JOINTS.copy()
    goal = start.copy()
    goal[JOINT_GROUPS["left_gripper"]] = GRIPPER_OPEN
    goal[JOINT_GROUPS["right_gripper"]] = GRIPPER_OPEN
    res = p.plan(start, goal)
    print(
        f"  both grippers close → open: success={res.success}  "
        f"waypoints={(len(res.path) if res.path is not None else 0)}"
    )

    # 2. Asymmetric: left closed, right open.
    goal2 = start.copy()
    goal2[JOINT_GROUPS["right_gripper"]] = GRIPPER_OPEN  # right opens
    # left stays closed at 0
    res = p.plan(start, goal2)
    print(
        f"  right opens, left stays closed: success={res.success}  "
        f"waypoints={(len(res.path) if res.path is not None else 0)}"
    )

    # 3. Subgroup planner — operate ONLY over the 4-DOF dual-gripper space.
    #    The arms and base are frozen at the supplied base_config (HOME).
    sub = create_planner(
        "bimanual_fr3_dual_gripper",
        base_config=HOME_JOINTS,
        config=PlannerConfig(planner_name="rrtc", time_limit=time_limit),
    )
    print(f"  dual-gripper subgroup: {sub.num_dof}-DOF, " f"joints={sub.joint_names}")
    sub_start = np.zeros(4)  # both grippers closed
    sub_goal = np.full(4, GRIPPER_OPEN)
    res = sub.plan(sub_start, sub_goal)
    print(
        f"    subgroup close → open: success={res.success}  "
        f"waypoints={(len(res.path) if res.path is not None else 0)}"
    )


def main(robot: str = "both", time_limit: float = 2.0) -> None:
    if robot in ("single_fr3", "both"):
        demo_single_fr3(time_limit)
    if robot in ("bimanual_fr3", "both"):
        demo_bimanual_fr3(time_limit)


if __name__ == "__main__":
    Fire(main)
