"""Runtime joint-limit overrides on the OMPL/VAMP planner.

Covers ``MotionPlanner.set_joint_limits`` / ``clear_joint_limits``: the
override is applied in active-DOF space, persists across subgroup
switches, and rejects ill-formed inputs.  Each test instantiates its
own planner so mutations don't leak via the session-scoped fixtures
in ``conftest.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("bimanual_franka_planning._ompl_vamp")


def _make_planner(robot_name: str):
    from bimanual_franka_planning.planning import create_planner
    from bimanual_franka_planning.types import PlannerConfig

    return create_planner(
        robot_name,
        config=PlannerConfig(planner_name="rrtc", time_limit=2.0),
    )


def _full_dim(planner) -> int:
    return len(planner._robot.joint_names)


def _default_full_dof_bounds(robot_name: str):
    """Default full-DOF bounds by querying a fresh full-body planner."""
    full_body = _make_planner(robot_name)
    lo = np.asarray(full_body._planner.lower_bounds(), dtype=np.float64)
    hi = np.asarray(full_body._planner.upper_bounds(), dtype=np.float64)
    return lo, hi


def test_set_joint_limits_updates_active_bounds():
    """After set_joint_limits, the active subgroup's lower/upper bounds
    equal the active subset of the supplied full-DOF arrays."""
    planner = _make_planner("bimanual_fr3_left_arm")
    full_lo, full_hi = _default_full_dof_bounds("bimanual_fr3")
    # Shrink every joint to 80% of its range, centred on the midpoint.
    mid = 0.5 * (full_lo + full_hi)
    half = 0.4 * (full_hi - full_lo)
    new_full_lo = mid - half
    new_full_hi = mid + half

    planner.set_joint_limits(new_full_lo, new_full_hi)

    active = planner._subgroup_indices
    assert active is not None
    got_lo = np.asarray(planner._planner.lower_bounds())
    got_hi = np.asarray(planner._planner.upper_bounds())
    np.testing.assert_allclose(got_lo, new_full_lo[active], atol=1e-12)
    np.testing.assert_allclose(got_hi, new_full_hi[active], atol=1e-12)


def test_clear_joint_limits_restores_defaults():
    planner = _make_planner("bimanual_fr3_left_arm")
    active = planner._subgroup_indices
    default_lo = np.asarray(planner._planner.lower_bounds())
    default_hi = np.asarray(planner._planner.upper_bounds())

    full_lo, full_hi = _default_full_dof_bounds("bimanual_fr3")
    mid = 0.5 * (full_lo + full_hi)
    planner.set_joint_limits(mid - 0.01, mid + 0.01)
    # Sanity-check: the override is actually applied first.
    tight = np.asarray(planner._planner.upper_bounds())
    assert np.all(tight < default_hi[: len(tight)] - 1e-6) or active is None

    planner.clear_joint_limits()
    got_lo = np.asarray(planner._planner.lower_bounds())
    got_hi = np.asarray(planner._planner.upper_bounds())
    np.testing.assert_allclose(got_lo, default_lo, atol=1e-12)
    np.testing.assert_allclose(got_hi, default_hi, atol=1e-12)


def test_set_joint_limits_persists_across_subgroup_switch():
    """Custom limits must survive a set_subgroup() call so callers can
    configure once at startup and switch subgroups freely."""
    planner = _make_planner("bimanual_fr3_left_arm")
    full_lo, full_hi = _default_full_dof_bounds("bimanual_fr3")
    mid = 0.5 * (full_lo + full_hi)
    half = 0.3 * (full_hi - full_lo)
    new_full_lo = mid - half
    new_full_hi = mid + half

    planner.set_joint_limits(new_full_lo, new_full_hi)

    # Switch to the right arm, then back.  Inactive frozen joints are
    # not consulted for sampling, so we don't worry about their
    # bounds; only the active subset matters.
    planner.set_subgroup("bimanual_fr3_right_arm")
    right_active = planner._subgroup_indices
    np.testing.assert_allclose(
        np.asarray(planner._planner.lower_bounds()),
        new_full_lo[right_active],
        atol=1e-12,
    )

    planner.set_subgroup("bimanual_fr3_left_arm")
    left_active = planner._subgroup_indices
    np.testing.assert_allclose(
        np.asarray(planner._planner.upper_bounds()),
        new_full_hi[left_active],
        atol=1e-12,
    )


def test_plan_path_stays_inside_custom_limits(home_joints):
    """Every waypoint on a planned path lies inside the active custom bounds.

    ``validate`` itself is a collision-only predicate in this planner —
    OMPL's state-space bounds are enforced by the sampler / motion
    validator, not by the user validity checker — so the meaningful
    end-to-end test is that ``plan`` returns a path entirely within
    the configured envelope.
    """
    planner = _make_planner("bimanual_fr3_left_arm")
    full_lo, full_hi = _default_full_dof_bounds("bimanual_fr3")
    q_home_active = planner.extract_config(home_joints)

    # Tight box around HOME: ±0.3 rad on every active joint — wide
    # enough for a non-trivial plan but well inside the URDF defaults.
    active = planner._subgroup_indices
    eps = 0.3
    new_full_lo = full_lo.copy()
    new_full_hi = full_hi.copy()
    new_full_lo[active] = q_home_active - eps
    new_full_hi[active] = q_home_active + eps
    planner.set_joint_limits(new_full_lo, new_full_hi)

    goal = q_home_active + 0.5 * eps  # well inside the new box
    result = planner.plan(q_home_active, goal)
    assert result.success, f"plan failed: {result.status}"

    active_lo = np.asarray(planner._planner.lower_bounds())
    active_hi = np.asarray(planner._planner.upper_bounds())
    assert np.all(result.path >= active_lo - 1e-9)
    assert np.all(result.path <= active_hi + 1e-9)


def test_set_joint_limits_rejects_wrong_length():
    planner = _make_planner("bimanual_fr3_left_arm")
    full_dim = _full_dim(planner)
    bad = np.zeros(full_dim - 1)
    with pytest.raises(ValueError, match="full-DOF"):
        planner.set_joint_limits(bad, bad)


def test_set_joint_limits_rejects_inverted_bounds():
    """Passing upper < lower must fail (C++ raises std::invalid_argument)."""
    planner = _make_planner("bimanual_fr3_left_arm")
    full_dim = _full_dim(planner)
    lo = np.ones(full_dim)
    hi = np.zeros(full_dim)
    with pytest.raises(Exception):  # nanobind may map to ValueError or RuntimeError
        planner.set_joint_limits(lo, hi)


def test_sample_valid_respects_custom_limits():
    """A sample drawn from sample_valid lies inside the active custom bounds."""
    planner = _make_planner("bimanual_fr3_left_arm")
    full_lo, full_hi = _default_full_dof_bounds("bimanual_fr3")
    mid = 0.5 * (full_lo + full_hi)
    half = 0.2 * (full_hi - full_lo)
    planner.set_joint_limits(mid - half, mid + half)

    active_lo = np.asarray(planner._planner.lower_bounds())
    active_hi = np.asarray(planner._planner.upper_bounds())
    np.random.seed(0)
    for _ in range(8):
        q = planner.sample_valid()
        assert np.all(q >= active_lo - 1e-9)
        assert np.all(q <= active_hi + 1e-9)
