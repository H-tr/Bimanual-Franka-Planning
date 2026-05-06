"""Single-arm FR3 motion planning — same factory, different robot name.

The standalone single-arm FR3 plugs into the same ``create_planner``
API the bimanual cell uses; the only thing that changes is the robot
name (and, by extension, the dimensionality of the configurations
the caller deals in).  Under the hood ``create_planner("single_fr3")``
dispatches to the dedicated ``SingleFr3OmplVampPlanner`` C++ class
generated from ``single_fr3_spherized.urdf``.

Usage:

    pixi run python examples/planning/single_arm_motion.py
    pixi run python examples/planning/single_arm_motion.py --planner_name bitstar --time_limit 3
"""

import numpy as np
from fire import Fire

from bimanual_franka_planning.planning import create_planner
from bimanual_franka_planning.single_franka import HOME_JOINTS
from bimanual_franka_planning.types import PlannerConfig


def main(
    planner_name: str = "rrtc",
    time_limit: float = 2.0,
    seed: int = 0,
) -> None:
    np.random.seed(seed)

    planner = create_planner(
        "single_fr3",
        config=PlannerConfig(planner_name=planner_name, time_limit=time_limit),
    )
    print(
        f"single-arm FR3 planner: {planner.num_dof}-DOF "
        f"({type(planner._planner).__name__})"
    )

    start = HOME_JOINTS.copy()
    assert planner.validate(start), "HOME joints must be collision-free"

    goal = planner.sample_valid()

    result = planner.plan(start, goal)
    n_wp = result.path.shape[0] if result.path is not None else 0
    print(
        f"  {result.status.value}: {n_wp} waypoints in "
        f"{result.planning_time_ns / 1e6:.0f} ms, cost {result.path_cost:.2f}"
    )


if __name__ == "__main__":
    Fire(main)
