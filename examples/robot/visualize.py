"""Interactive path visualizer for bimanual_fr3_planner.plan_to_joints.

Cycles through every supported subgroup, plans from HOME to a
hand-crafted goal, and animates the result in PyBullet.

Usage:
    pixi run -e dev python examples/robot/visualize.py
    pixi run -e dev python examples/robot/visualize.py --group bimanual_fr3_left_arm
"""

import sys
import time
from pathlib import Path

import numpy as np
from fire import Fire

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot.bimanual_fr3_planner import (  # noqa: E402
    SUPPORTED_GROUPS,
    BimanualFr3Planner,
)

from bimanual_franka_planning.bimanual_franka import (  # noqa: E402
    HOME_JOINTS,
    VIZ_URDF_PATH,
    bimanual_fr3_robot_config,
)
from bimanual_franka_planning.envs.pybullet_env import PyBulletEnv  # noqa: E402

# ---------------------------------------------------------------------------
# Default goals — one hand-crafted 17-DOF target per supported group.
#
# Layout: [relative_base(3), right_arm(7), left_arm(7)]
#
# Only the joints owned by the group differ from HOME; every frozen
# joint keeps its HOME value so plan_to_joints never returns "not same".
# ---------------------------------------------------------------------------

# Goal segments (active joints moved to natural reach poses).
_GOAL_LEFT_ARM = np.array([0.6, -0.50, 0.30, -1.80, -0.20, 1.40, 0.50])
_GOAL_RIGHT_ARM = np.array([-0.6, -0.50, 0.30, -1.80, 0.20, 1.40, -0.50])


def _goal_for(group: str) -> np.ndarray:
    goal = HOME_JOINTS.copy()
    if group == "bimanual_fr3_left_arm":
        goal[10:17] = _GOAL_LEFT_ARM
    elif group == "bimanual_fr3_right_arm":
        goal[3:10] = _GOAL_RIGHT_ARM
    elif group == "bimanual_fr3_dual_arm":
        goal[3:10] = _GOAL_RIGHT_ARM
        goal[10:17] = _GOAL_LEFT_ARM
    else:
        raise ValueError(f"Unknown group: {group!r}")
    return goal


# Ordered for a natural demo progression: each arm individually, then both.
_DEFAULT_GROUP_ORDER = [
    "bimanual_fr3_left_arm",
    "bimanual_fr3_right_arm",
    "bimanual_fr3_dual_arm",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _banner(group: str, result, t_ms: float) -> None:
    if result is None:
        status = "NO PATH (timeout)"
    elif isinstance(result, str):
        status = f"SKIPPED ({result})"
    else:
        status = f"OK  {result.shape[0]} waypoints"
    print(f"  [{group}]  {status}  ({t_ms:.0f} ms)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    group: str | None = None,
    planner: str = "bitstar",
    time_limit: float = 2.0,
    fps: float = 50.0,
) -> None:
    """Visualize planned paths for one or all supported planning groups.

    Args:
        group:      Name of a single group to visualize.  Omit to cycle
                    through all supported groups in sequence.
        planner:    OMPL planner name (rrtc, rrtstar, bitstar, …).
        time_limit: Planning timeout in seconds per group.
        fps:        Playback frame rate while auto-playing.
    """
    if group is not None:
        if group not in SUPPORTED_GROUPS:
            print(
                f"Unknown group {group!r}.\n"
                f"Supported: {', '.join(sorted(SUPPORTED_GROUPS))}"
            )
            sys.exit(1)
        groups = [group]
    else:
        groups = [g for g in _DEFAULT_GROUP_ORDER if g in SUPPORTED_GROUPS]

    env = PyBulletEnv(
        bimanual_fr3_robot_config, visualize=True, viz_urdf_path=VIZ_URDF_PATH
    )

    print(
        f"\n── bimanual_fr3 path visualizer ──\n"
        f"  planner={planner}  time_limit={time_limit}s\n"
        f"  {len(groups)} group(s): {', '.join(groups)}\n"
    )

    ap = BimanualFr3Planner(planner_name=planner, time_limit=time_limit)
    start = HOME_JOINTS.copy()
    env.set_configuration(start)

    for g in groups:
        goal = _goal_for(g)

        t0 = time.perf_counter()
        result = ap.plan_to_joints(g, start, goal)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        _banner(g, result, elapsed_ms)

        if isinstance(result, np.ndarray):
            _, path = ap.time_parameterize(result)
            cont = env.animate_path(path, fps=fps, next_key="n")
        else:
            env.set_configuration(start)
            msg = (
                "  (no path found — press 'n' for next group, close to quit)"
                if result is None
                else "  (skipped: frozen joints differ — press 'n' for next)"
            )
            env.wait_key("n", msg)
            cont = True

        if not cont:
            break

    env.wait_for_close()


if __name__ == "__main__":
    Fire(main)
