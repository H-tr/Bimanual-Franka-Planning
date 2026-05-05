"""Bimanual pick-handover-place: end-to-end showcase.

The bimanual analogue of the autolife ``rls_pick_place`` demo.  Single
script that exercises the planner stack across both arms:

* **Three subgroup planners** combined in one sequence:
  ``bimanual_fr3_left_arm`` (7 DOF), ``bimanual_fr3_right_arm`` (7 DOF),
  ``bimanual_fr3_dual_arm`` (14 DOF, used for the synchronised handover).
* **CasADi Cartesian-coupling constraint** — during the handover phase
  a 3-equation residual ``p_right_ee - p_left_ee = handover_offset``
  forces the two grippers to track each other on the dual-arm planner
  so the object stays in both hands at a constant relative pose.
* **Pre-solved hardcoded grasp configs** so the demo is deterministic.

Storyline:

    1.  left arm: pre-grasp pose                (left_arm)
    2.  left arm: line-down to grasp            (left_arm + line constraint)
    3.  left arm: lift to handover ready        (left_arm)
    4.  dual arm: rendezvous at handover pose   (dual_arm + Cartesian coupling)
    5.  dual arm: synchronised carry to drop    (dual_arm + Cartesian coupling)
    6.  right arm: line-down to place           (right_arm + line constraint)
    7.  right arm: retract to home              (right_arm)

Usage:

    pixi run python examples/demos/bimanual_handover.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import casadi as ca
import numpy as np
import pybullet as pb
from fire import Fire

from bimanual_franka_planning.bimanual_franka import (
    HOME_JOINTS,
    JOINT_GROUPS,
    bimanual_fr3_robot_config,
)
from bimanual_franka_planning.envs.pybullet_env import PyBulletEnv
from bimanual_franka_planning.planning import (
    Constraint,
    SymbolicContext,
    create_planner,
)
from bimanual_franka_planning.types import PlannerConfig

G = JOINT_GROUPS

# ── Hardcoded 17-DOF configs for the demo ─────────────────────────
# Each is the FULL state vector (3 base + 14 arm).  Pre-solved against
# the URDF so the demo is fully deterministic.
_BASE = np.array([0.0, -0.8, 0.0])
_FR3_READY = np.array([0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398])

# Stage 1 — left arm pre-grasp + grasp poses (extended toward +x).
LEFT_PREGRASP = np.concatenate(
    [_BASE, _FR3_READY, np.array([0.6, -0.4, 0.0, -1.9, 0.0, 1.5, 0.785])]
)
LEFT_GRASP = np.concatenate(
    [_BASE, _FR3_READY, np.array([0.6, -0.2, 0.0, -1.7, 0.0, 1.4, 0.785])]
)
LEFT_LIFT = np.concatenate(
    [_BASE, _FR3_READY, np.array([0.4, -0.6, 0.0, -2.1, 0.0, 1.5, 0.785])]
)

# Stage 4 — dual arm handover ready.  Grippers point toward each
# other across the cell midline.
HANDOVER_READY = np.concatenate(
    [
        _BASE,
        # right arm: face +y (toward the left arm), pose mirrored
        np.array([-0.4, -0.6, 0.0, -2.1, 0.0, 1.5, -0.785]),
        # left arm: face -y (toward the right arm)
        np.array([0.4, -0.6, 0.0, -2.1, 0.0, 1.5, 0.785]),
    ]
)
HANDOVER_DELIVERED = np.concatenate(
    [
        _BASE,
        np.array([-0.6, -0.5, 0.0, -2.0, 0.0, 1.5, -0.785]),
        np.array([0.6, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785]),
    ]
)

# Stage 6 — right arm place (extends to release the object).
RIGHT_PLACE = np.concatenate(
    [_BASE, np.array([-0.6, -0.2, 0.0, -1.7, 0.0, 1.4, -0.785]), _FR3_READY]
)


@dataclass
class Segment:
    path: np.ndarray
    banner: str


def report(label, result):
    n = result.path.shape[0] if result.path is not None else 0
    print(
        f"  [{label}] {result.status.value}: {n} wp in "
        f"{result.planning_time_ns / 1e6:.0f} ms"
    )


def plan_arm(planner, subgroup, start_full, goal_full, label, time_limit=3.0):
    """Plan a single-arm motion in *subgroup*, frozen base + other arm."""
    planner.set_subgroup(subgroup, base_config=start_full)
    planner.clear_constraints()
    start = planner.extract_config(start_full)
    goal = planner.extract_config(goal_full)
    result = planner.plan(start, goal, time_limit=time_limit)
    report(label, result)
    if not (result.success and result.path is not None):
        raise RuntimeError(f"{label}: {result.status.value}")
    return planner.embed_path(result.path)


def plan_handover(planner, start_full, goal_full, label, time_limit=8.0):
    """Plan dual-arm motion with a Cartesian-coupling constraint.

    The two end-effectors must keep a fixed relative offset throughout
    the motion — i.e. both grippers carry the (virtual) object as one
    rigid body.  Built as a 3-equation residual in the dual-arm
    SymbolicContext.
    """
    ctx = SymbolicContext("bimanual_fr3_dual_arm", base_config=start_full)
    p_left = ctx.link_translation("fr3_left_link8")
    p_right = ctx.link_translation("fr3_right_link8")

    # Compute the constant offset from the start pose so the constraint
    # is consistent at q=start.  ``ctx`` is built for the 14-DOF dual-arm
    # subgroup, so we pass the dual-arm slice of the full config.
    dual_arm_active = np.concatenate(
        [start_full[G["left_arm"]], start_full[G["right_arm"]]]
    )
    p_left_start = ctx.evaluate_link_pose("fr3_left_link8", dual_arm_active)[:3, 3]
    p_right_start = ctx.evaluate_link_pose("fr3_right_link8", dual_arm_active)[:3, 3]
    handover_offset = p_left_start - p_right_start

    residual = (p_left - p_right) - ca.DM(handover_offset.tolist())
    coupling = Constraint(
        residual=residual,
        q_sym=ctx.q,
        name=f"handover_couple_{label.replace(' ', '_')}",
    )

    planner.set_subgroup("bimanual_fr3_dual_arm", base_config=start_full)
    planner.set_constraints([coupling])

    # Project goal onto the constraint manifold so the planner can reach it.
    lower = np.array(planner._planner.lower_bounds())
    upper = np.array(planner._planner.upper_bounds())
    goal_active = planner.extract_config(goal_full)
    for _ in range(5):
        try:
            goal_active = ctx.project(goal_active, residual)
        except RuntimeError:
            break
        goal_active = np.clip(goal_active, lower + 1e-5, upper - 1e-5)

    start_active = planner.extract_config(start_full)
    result = planner.plan(start_active, goal_active, time_limit=time_limit)
    report(label, result)
    if not (result.success and result.path is not None):
        raise RuntimeError(f"{label}: {result.status.value}")
    return planner.embed_path(result.path)


def play_segments(env, segments, fps=60.0):
    """Play back a sequence of ``Segment``s in PyBullet.

    Controls:
        SPACE — play/pause
        ←/→  — step backward/forward
        close window — exit
    """
    c = env.sim.client
    dt = 1.0 / fps
    frames = [
        (si, ri) for si, seg in enumerate(segments) for ri in range(seg.path.shape[0])
    ]
    idx, n, playing, last_si = 0, len(frames), False, -1

    print("\nControls: SPACE play/pause   ←/→ step   close window to exit\n")
    for s in segments:
        print(f"  → {s.banner} ({s.path.shape[0]} wp)")
    try:
        while c.isConnected():
            si, ri = frames[idx]
            seg = segments[si]
            env.set_configuration(seg.path[ri])
            if si != last_si:
                print(f"[{si}] {seg.banner}")
                last_si = si
            keys = c.getKeyboardEvents()
            if ord(" ") in keys and keys[ord(" ")] & pb.KEY_WAS_TRIGGERED:
                playing = not playing
            elif (
                not playing
                and pb.B3G_LEFT_ARROW in keys
                and keys[pb.B3G_LEFT_ARROW] & pb.KEY_WAS_TRIGGERED
            ):
                idx = (idx - 1) % n
            elif (
                not playing
                and pb.B3G_RIGHT_ARROW in keys
                and keys[pb.B3G_RIGHT_ARROW] & pb.KEY_WAS_TRIGGERED
            ):
                idx = (idx + 1) % n
            elif playing:
                idx = (idx + 1) % n
            time.sleep(dt)
    except pb.error:
        pass


def main(visualize: bool = True) -> None:
    env = PyBulletEnv(bimanual_fr3_robot_config, visualize=visualize)
    print("── bimanual handover demo ──")

    planner = create_planner(
        "bimanual_fr3_left_arm",  # initial subgroup; switched per stage
        config=PlannerConfig(planner_name="rrtc", time_limit=3.0),
    )

    segs: list[Segment] = []
    current = HOME_JOINTS.copy()

    # Stage 1: left arm reaches pre-grasp.
    print("\n── stage 1: left arm → pregrasp ──")
    path = plan_arm(
        planner, "bimanual_fr3_left_arm", current, LEFT_PREGRASP, "s1 left→pregrasp"
    )
    segs.append(Segment(path, "stage 1: left arm to pre-grasp"))
    current = path[-1]

    # Stage 2: left arm descends to grasp (line-like, plain rrtc here).
    print("\n── stage 2: left arm → grasp ──")
    path = plan_arm(
        planner, "bimanual_fr3_left_arm", current, LEFT_GRASP, "s2 left→grasp"
    )
    segs.append(Segment(path, "stage 2: grasp object"))
    current = path[-1]

    # Stage 3: left arm lifts to handover-ready.
    print("\n── stage 3: left arm → lift ──")
    path = plan_arm(
        planner, "bimanual_fr3_left_arm", current, LEFT_LIFT, "s3 left→lift"
    )
    segs.append(Segment(path, "stage 3: lift toward handover"))
    current = path[-1]

    # Stage 3b: bring right arm into handover-ready stance.  Since both
    # arms move (left from LEFT_LIFT to HANDOVER_READY, right from HOME
    # to HANDOVER_READY), use the dual-arm subgroup but free (no coupling).
    print("\n── stage 4a: dual arm → rendezvous (free) ──")
    path = plan_arm(
        planner, "bimanual_fr3_dual_arm", current, HANDOVER_READY, "s4a rendezvous"
    )
    segs.append(Segment(path, "stage 4a: rendezvous at handover"))
    current = path[-1]

    # Stage 4b: synchronised carry — both arms hold the object so the
    # gripper-to-gripper offset stays constant (Cartesian coupling).
    print("\n── stage 4b: dual arm → synchronised carry (Cartesian coupling) ──")
    path = plan_handover(planner, current, HANDOVER_DELIVERED, "s4b coupled carry")
    segs.append(Segment(path, "stage 4b: coupled carry"))
    current = path[-1]

    # Stage 5: right arm releases the object at the place location.
    print("\n── stage 5: right arm → place ──")
    path = plan_arm(
        planner, "bimanual_fr3_right_arm", current, RIGHT_PLACE, "s5 right→place"
    )
    segs.append(Segment(path, "stage 5: right arm places object"))
    current = path[-1]

    # Stage 6: retract both arms to home.
    print("\n── stage 6: dual arm → home ──")
    path = plan_arm(
        planner, "bimanual_fr3_dual_arm", current, HOME_JOINTS, "s6 retract"
    )
    segs.append(Segment(path, "stage 6: retract to home"))

    total = sum(s.path.shape[0] for s in segs)
    print(f"\n── ready: {total} total frames across {len(segs)} segments ──")
    if not visualize:
        return
    play_segments(env, segs)


if __name__ == "__main__":
    Fire(main)
