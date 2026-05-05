"""Trajectory visualisation for joint-space paths.

Plans a path with BimanualFr3Planner, time-parameterizes it with
``time_parameterize`` (from robot.bimanual_fr3_planner), and plots position,
velocity, and acceleration profiles for selected joints.

Usage::

    pixi run python robot/trajectory.py
    pixi run python robot/trajectory.py --group bimanual_fr3_dual_arm
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot.bimanual_fr3_planner import BimanualFr3Planner  # noqa: E402
from robot.visualize import _DEFAULT_GOALS  # noqa: E402

from bimanual_franka_planning.bimanual_franka import bimanual_fr3_robot_config  # noqa: E402

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_trajectory(
    path: np.ndarray,
    times_after: np.ndarray,
    traj_after: np.ndarray,
    joint_names: list[str],
    active_indices: list[int],
    v_max: float = 0.8,
    a_max: float = 0.8,
    dt_before: float = 0.02,
    title: str = "",
) -> None:
    """Plot position, velocity, and acceleration before/after time parameterization.

    Only the *active* joints (those that actually move) are shown.
    Three rows per joint column: position · velocity · acceleration.

    Args:
        path:           ``(N, DOF)`` raw waypoints.
        times_after:    ``(M,)`` timestamps of the parameterized trajectory.
        traj_after:     ``(M, DOF)`` parameterized trajectory.
        joint_names:    Full list of 24 joint names.
        active_indices: Indices of the joints active in this group.
        v_max:          Velocity limit — dashed line on velocity row.
        a_max:          Acceleration limit — dashed line on acceleration row.
        dt_before:      Assumed uniform spacing for the raw waypoints.
        title:          Figure suptitle.
    """
    n_active = len(active_indices)
    cols = min(n_active, 7)
    stride = max(1, n_active // cols)
    plot_idx = active_indices[::stride][:cols]

    times_before = np.arange(len(path)) * dt_before
    dt_out = float(times_after[1] - times_after[0])
    vel = np.gradient(traj_after, dt_out, axis=0)
    acc = np.gradient(vel, dt_out, axis=0)

    fig, axes = plt.subplots(
        3,
        len(plot_idx),
        figsize=(2.8 * len(plot_idx), 8),
        sharey="row",
    )
    if len(plot_idx) == 1:
        axes = axes.reshape(3, 1)

    fig.suptitle(title or "Time parameterization", fontsize=11, fontweight="bold")

    for col, j in enumerate(plot_idx):
        short = joint_names[j].replace("Joint_", "")

        # ── position ──────────────────────────────────────────────────
        ax = axes[0, col]
        ax.plot(
            times_before,
            path[:, j],
            "o",
            ms=4,
            color="steelblue",
            zorder=3,
            label="waypoints\n(Δt=0.02 s assumed)",
        )
        ax.plot(
            times_after,
            traj_after[:, j],
            lw=1.5,
            color="tomato",
            label="parameterized",
        )
        ax.set_title(short, fontsize=8)
        if col == 0:
            ax.set_ylabel("pos (rad)", fontsize=8)
            ax.legend(fontsize=6, loc="best")
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

        # ── velocity ──────────────────────────────────────────────────
        ax = axes[1, col]
        ax.plot(times_after, vel[:, j], lw=1.3, color="darkorange")
        ax.axhline(v_max, color="gray", ls="--", lw=0.8)
        ax.axhline(-v_max, color="gray", ls="--", lw=0.8, label=f"±{v_max} rad/s")
        if col == 0:
            ax.set_ylabel("vel (rad/s)", fontsize=8)
            ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

        # ── acceleration ───────────────────────────────────────────────
        ax = axes[2, col]
        ax.plot(times_after, acc[:, j], lw=1.3, color="mediumseagreen")
        ax.axhline(a_max, color="gray", ls="--", lw=0.8)
        ax.axhline(-a_max, color="gray", ls="--", lw=0.8, label=f"±{a_max} rad/s²")
        ax.set_xlabel("time (s)", fontsize=7)
        if col == 0:
            ax.set_ylabel("acc (rad/s²)", fontsize=8)
            ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Demo entry point
# ---------------------------------------------------------------------------


def main(
    group: str = "bimanual_fr3_dual_arm",
    planner: str = "bitstar",
    time_limit: float = 2.0,
    v_max: float = 0.5,
    a_max: float = 0.6,
    dt: float = 0.02,
    vel_scale: float = 1.0,
    a_scale: float = 1.0,
) -> None:
    from robot.bimanual_fr3_planner import _active_indices

    from bimanual_franka_planning.bimanual_franka import HOME_JOINTS

    ap = BimanualFr3Planner(planner_name=planner, time_limit=time_limit)
    start = HOME_JOINTS.copy()
    goal = _DEFAULT_GOALS.get(group)
    if goal is None:
        print(f"No default goal for {group!r}. Available: {list(_DEFAULT_GOALS)}")
        sys.exit(1)

    print(f"Planning {group} …")
    start_time = time.time()
    path = ap.plan_to_joints(group, start, goal)
    end_time = time.time()
    print(f"Planning time: {end_time - start_time} seconds")
    if not isinstance(path, np.ndarray):
        print(f"Planning failed: {path}")
        sys.exit(1)

    print(f"  {len(path)} waypoints  →  time-parameterizing …")
    start_time = time.time()
    times, traj = ap.time_parameterize(
        path, v_max=v_max, a_max=a_max, dt=dt, vel_scale=vel_scale, a_scale=a_scale
    )
    end_time = time.time()
    print(f"Time-parameterizing time: {end_time - start_time} seconds")
    print(f"  {len(traj)} trajectory points  (total {times[-1]:.2f} s)")

    from bimanual_franka_planning.bimanual_franka import JOINT_GROUPS

    active = _active_indices(group).tolist()

    # For dual_arm show only legs + torso + left arm; skip right arm.
    if group == "bimanual_fr3_dual_arm":
        right_arm = set(
            range(JOINT_GROUPS["right_arm"].start, JOINT_GROUPS["right_arm"].stop)
        )
        active = [i for i in active if i not in right_arm]

    plot_trajectory(
        path,
        times,
        traj,
        joint_names=list(bimanual_fr3_robot_config.joint_names),
        active_indices=active,
        v_max=v_max,
        a_max=a_max,
        dt_before=dt,
        title=f"{group}  |  {len(path)} waypoints → {len(traj)} traj pts @ dt={dt} s",
    )


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
