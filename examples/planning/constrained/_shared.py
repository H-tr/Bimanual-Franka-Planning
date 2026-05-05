"""Shared scaffolding for the constrained-planning gallery.

Every demo in this folder boots the same PyBullet env and left-arm
SymbolicContext, searches for a random manifold-feasible goal via
seed-and-project, and then plans + animates.  Those three pieces
are factored out here so each demo script can focus on the one
interesting line: the CasADi residual that defines the manifold.

Visualisation primitives (``draw_plane``, ``draw_rod``,
``draw_sphere``, ``draw_frame``) live on :class:`PyBulletEnv` itself
— each demo calls ``env.draw_*`` exactly like it already calls
``env.animate_path``.
"""

from __future__ import annotations

import numpy as np

from bimanual_franka_planning.bimanual_franka import (
    HOME_JOINTS,
    bimanual_fr3_robot_config,
)
from bimanual_franka_planning.envs.pybullet_env import PyBulletEnv
from bimanual_franka_planning.planning import SymbolicContext

SUBGROUP = "bimanual_fr3_left_arm"
# EE_LINK is the wrist flange (fr3_left_link8) that the constraint
# residuals use both for orientation locks and translation targets.
# The Franka gripper's tool-center-point sits ~0.103 m past flange
# along +z (see fr3_hand_tcp in the URDF), so for translation-only
# constraints we expose ``ee_translation`` as the flange position
# (no offset) — callers add the gripper depth in their own frame.
EE_LINK = "fr3_left_link8"


def ee_translation(ctx: "SymbolicContext"):
    """CasADi expression for the EE flange position."""
    return ctx.link_translation(EE_LINK)


def ee_position(ctx: "SymbolicContext", q: np.ndarray) -> np.ndarray:
    """Numeric evaluation of the EE flange position at joint config *q*."""
    return np.asarray(ctx.evaluate_link_pose(EE_LINK, q))[:3, 3]


def setup():
    """Boot a PyBullet env, a SymbolicContext, and the HOME start config."""
    env = PyBulletEnv(bimanual_fr3_robot_config, visualize=True)
    ctx = SymbolicContext(SUBGROUP)
    start = HOME_JOINTS[ctx.active_indices].copy()
    return env, ctx, start


def find_goal(ctx, residual, start, planner, score, n: int = 400, seed: int = 0):
    """Return a valid manifold-feasible goal that maximises ``score(xyz)``.

    Samples random joint perturbations of *start*, projects each
    onto the manifold via ``ctx.project``, keeps the valid config
    whose gripper position maximises ``score``.  Valid means inside
    the joint bounds AND collision-free under *planner*.
    """
    lower = np.array(planner._planner.lower_bounds())
    upper = np.array(planner._planner.upper_bounds())
    rng = np.random.default_rng(seed)
    best_q, best_s = None, -np.inf
    for _ in range(n):
        seed_q = np.clip(
            start + rng.uniform(-0.7, 0.7, start.shape[0]),
            lower + 0.02,
            upper - 0.02,
        )
        try:
            g = ctx.project(seed_q, residual)
        except RuntimeError:
            continue
        if np.any(g < lower) or np.any(g > upper):
            continue
        if not planner.validate(g):
            continue
        s = float(score(ee_position(ctx, g)))
        if s > best_s:
            best_q, best_s = g, s
    if best_q is None:
        raise RuntimeError("no reachable manifold goal found")
    return best_q


def run_demo(env, planner, start, goal, title: str) -> None:
    """Plan from start to goal, print a one-line summary, animate the path.

    The animation is interactive — ``env.animate_path`` blocks on the
    PyBullet window with space/arrow-key controls and only returns when
    the user closes the viewer.
    """
    print(f"── {title} ──")
    result = planner.plan(start, goal)
    n = result.path.shape[0] if result.path is not None else 0
    print(
        f"  {result.status.value}: {n} waypoints in "
        f"{result.planning_time_ns / 1e6:.0f} ms, cost {result.path_cost:.2f}"
    )
    if result.success and result.path is not None:
        env.animate_path(planner.embed_path(result.path), fps=60)
    else:
        env.wait_for_close()
