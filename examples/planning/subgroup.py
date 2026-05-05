"""Sweep every kinematic subgroup at three different cell layouts.

Demonstrates that the same subgroup name (e.g. ``bimanual_fr3_left_arm``)
can be planned around any 17-DOF base configuration the caller passes
in — the relative-base offset between the two arms is part of that
base config and gets pinned by ``base_config`` on every call.

The three layouts below are *example data* (the cell calibration you
might measure at deployment time) — not part of the planning API.

    pixi run python examples/planning/subgroup.py
"""

import numpy as np
from fire import Fire

from bimanual_franka_planning.bimanual_franka import (
    HOME_JOINTS,
    bimanual_fr3_robot_config,
)
from bimanual_franka_planning.envs.pybullet_env import PyBulletEnv
from bimanual_franka_planning.planning import create_planner
from bimanual_franka_planning.types import PlannerConfig

# Three cell layouts: (relative_base_x, relative_base_y, relative_base_yaw).
# These pin the right-arm base at deploy time; replace this dict with any
# (x, y, yaw) measured at calibration to plan around an arbitrary cell.
LAYOUTS = {
    "wide": {
        "relative_base_x": 0.0,
        "relative_base_y": -1.0,
        "relative_base_yaw": 0.0,
    },
    "default": {
        "relative_base_x": 0.0,
        "relative_base_y": -0.8,
        "relative_base_yaw": 0.0,
    },
    "narrow_angled": {
        "relative_base_x": 0.0,
        "relative_base_y": -0.6,
        "relative_base_yaw": -0.5,
    },
}

SUBGROUPS = [
    "bimanual_fr3_left_arm",
    "bimanual_fr3_right_arm",
    "bimanual_fr3_dual_arm",
]


def base_with_layout(layout: dict[str, float]) -> np.ndarray:
    base = HOME_JOINTS.copy()
    for joint_name, value in layout.items():
        base[bimanual_fr3_robot_config.joint_names.index(joint_name)] = value
    return base


def plan_and_show(
    env, robot_name: str, base: np.ndarray, config: PlannerConfig, label: str
) -> bool:
    """Plan one subgroup against *base* and animate it interactively.

    Returns ``True`` if the user pressed ``n`` to advance to the next
    demo, ``False`` if the user closed the GUI window (in which case the
    caller should stop iterating).
    """
    planner = create_planner(robot_name, config=config, base_config=base)
    start = planner.extract_config(base)
    goal = planner.sample_valid()

    result = planner.plan(start, goal)
    n_wp = result.path.shape[0] if result.path is not None else 0
    print(f"  [{label}] {result.status.value} — {n_wp} waypoints")

    if result.success and result.path is not None:
        return env.animate_path(planner.embed_path(result.path), next_key="n")
    env.wait_key("n", f"[{label}] no path — press 'n' for next")
    return env.sim.client.isConnected()


def main(planner_name: str = "bitstar", time_limit: float = 0.5):
    """Run the subgroup × layout sweep with the chosen OMPL planner.

    Available planner names (pick one and pass as ``--planner_name``):

        RRT family ........... rrtc / rrtconnect, rrt, rrtstar,
                               informed_rrtstar, rrtsharp, rrtxstatic,
                               strrtstar, lbtrrt, trrt, bitrrt
        Informed trees ....... bitstar, abitstar, aitstar, eitstar, blitstar
        FMT .................. fmt, bfmt
        KPIECE ............... kpiece, bkpiece, lbkpiece
        PRM family ........... prm, prmstar, lazyprm, lazyprmstar,
                               spars, spars2
        Exploration-based .... est, biest, sbl, stride, pdst

    Single-query feasibility planners (``rrtc``, ``rrt``, ``kpiece``,
    ``est``, …) terminate as soon as they find any valid path.
    Asymptotically optimal anytime planners (``bitstar``, ``aitstar``,
    ``rrtstar``, …) keep refining the path until ``time_limit`` expires.
    """
    env = PyBulletEnv(bimanual_fr3_robot_config, visualize=True)
    config = PlannerConfig(planner_name=planner_name, time_limit=time_limit)

    for layout_name, layout in LAYOUTS.items():
        base = base_with_layout(layout)
        for robot_name in SUBGROUPS:
            cont = plan_and_show(
                env, robot_name, base, config, f"{robot_name} @ {layout_name}"
            )
            if not cont:
                return

    env.wait_for_close()


if __name__ == "__main__":
    Fire(main)
