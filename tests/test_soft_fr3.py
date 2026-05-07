"""Soft-gripper FR3 — registry, factories, and motion-planning smoke tests.

Mirrors :mod:`tests.test_single_fr3` but for the
``single_fr3_soft`` and ``bimanual_fr3_soft`` robots that swap the
Franka parallel-jaw fingers for a pair of soft-rubber fingers.  Only
the end-effector geometry differs from the existing FR3 robots, so
each soft variant gets its own cricket FK header and dedicated
``OmplVampPlanner`` C++ class.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("bimanual_franka_planning._ompl_vamp")


# ── Registry / factory dispatch ───────────────────────────────────────


def test_soft_robots_registered():
    from bimanual_franka_planning._robots import ROBOT_REGISTRY

    assert "single_fr3_soft" in ROBOT_REGISTRY
    assert "bimanual_fr3_soft" in ROBOT_REGISTRY


def test_soft_robots_in_available():
    from bimanual_franka_planning.planning import available_robots

    names = available_robots()
    assert "single_fr3_soft" in names
    assert "bimanual_fr3_soft" in names
    assert "bimanual_fr3_soft_left_arm" in names
    assert "bimanual_fr3_soft_right_arm" in names
    assert "bimanual_fr3_soft_dual_arm" in names


def test_single_soft_uses_dedicated_cpp_class():
    """The single-arm soft planner must dispatch to its own
    ``SingleFr3SoftOmplVampPlanner``, not the rigid-finger
    ``SingleFr3OmplVampPlanner`` — the FK header carries different
    sphere geometry."""
    from bimanual_franka_planning._ompl_vamp import (
        SingleFr3OmplVampPlanner,
        SingleFr3SoftOmplVampPlanner,
    )
    from bimanual_franka_planning.planning import create_planner

    p = create_planner("single_fr3_soft")
    assert isinstance(p._planner, SingleFr3SoftOmplVampPlanner)
    assert not isinstance(p._planner, SingleFr3OmplVampPlanner)


def test_bimanual_soft_uses_dedicated_cpp_class():
    from bimanual_franka_planning._ompl_vamp import (
        BimanualFr3SoftOmplVampPlanner,
        OmplVampPlanner,
    )
    from bimanual_franka_planning.planning import create_planner

    p = create_planner("bimanual_fr3_soft")
    assert isinstance(p._planner, BimanualFr3SoftOmplVampPlanner)
    assert not isinstance(p._planner, OmplVampPlanner)


# ── Single-arm soft motion planning ───────────────────────────────────


@pytest.fixture(scope="module")
def single_soft_planner():
    from bimanual_franka_planning.planning import create_planner
    from bimanual_franka_planning.types import PlannerConfig

    return create_planner(
        "single_fr3_soft",
        config=PlannerConfig(planner_name="rrtc", time_limit=2.0),
    )


@pytest.fixture(scope="module")
def single_soft_home():
    from bimanual_franka_planning.single_franka_soft import HOME_JOINTS

    return HOME_JOINTS.copy()


def test_single_soft_planner_dim(single_soft_planner):
    assert single_soft_planner.num_dof == 7
    assert single_soft_planner.joint_names == [f"fr3_joint{i}" for i in range(1, 8)]


def test_single_soft_home_is_valid(single_soft_planner, single_soft_home):
    assert single_soft_planner.validate(single_soft_home)


def test_single_soft_trivial_plan(single_soft_planner, single_soft_home):
    result = single_soft_planner.plan(single_soft_home, single_soft_home)
    assert result.success
    assert result.path is not None and result.path.shape[1] == 7


def test_single_soft_validate_batch(single_soft_planner, single_soft_home):
    np.random.seed(0)
    lo = np.asarray(single_soft_planner._planner.lower_bounds())
    hi = np.asarray(single_soft_planner._planner.upper_bounds())
    samples = np.random.uniform(lo, hi, size=(20, 7))
    samples[0] = single_soft_home
    expected = np.array(
        [single_soft_planner.validate(s) for s in samples], dtype=bool
    )
    got = single_soft_planner.validate_batch(samples)
    np.testing.assert_array_equal(got, expected)


# ── Bimanual soft motion planning ─────────────────────────────────────


@pytest.fixture(scope="module")
def bimanual_soft_planner():
    from bimanual_franka_planning.planning import create_planner
    from bimanual_franka_planning.types import PlannerConfig

    return create_planner(
        "bimanual_fr3_soft",
        config=PlannerConfig(planner_name="rrtc", time_limit=2.0),
    )


@pytest.fixture(scope="module")
def bimanual_soft_home():
    from bimanual_franka_planning.bimanual_franka_soft import HOME_JOINTS

    return HOME_JOINTS.copy()


def test_bimanual_soft_planner_dim(bimanual_soft_planner):
    assert bimanual_soft_planner.num_dof == 17


def test_bimanual_soft_home_is_valid(bimanual_soft_planner, bimanual_soft_home):
    assert bimanual_soft_planner.validate(bimanual_soft_home)


def test_bimanual_soft_trivial_plan(bimanual_soft_planner, bimanual_soft_home):
    result = bimanual_soft_planner.plan(bimanual_soft_home, bimanual_soft_home)
    assert result.success


def test_bimanual_soft_subgroup_planner():
    """Subgroup planning on bimanual_fr3_soft should freeze the inactive
    joints and operate on a 7-DOF reduced state."""
    from bimanual_franka_planning.planning import create_planner

    p = create_planner("bimanual_fr3_soft_left_arm")
    assert p.num_dof == 7
    assert p.is_subgroup


# ── FK / IK consistency cross-check ───────────────────────────────────


def test_single_soft_pinocchio_matches_planner_bounds():
    """Cricket-derived joint bounds in the FK header must match
    Pinocchio reading the source URDF — guards against drift between
    ``single_fr3_soft_spherized.urdf`` and the regenerated header."""
    pytest.importorskip("pinocchio")
    import pinocchio as pin

    from bimanual_franka_planning.planning import create_planner
    from bimanual_franka_planning.single_franka_soft import (
        CHAIN_CONFIGS,
        single_fr3_soft_robot_config,
    )

    p = create_planner("single_fr3_soft")
    assert p.joint_names == single_fr3_soft_robot_config.joint_names

    model = pin.buildModelFromUrdf(CHAIN_CONFIGS["single_fr3_soft"].urdf_path)
    pin_lo = np.asarray(model.lowerPositionLimit, dtype=float)
    pin_hi = np.asarray(model.upperPositionLimit, dtype=float)
    p_lo = np.asarray(p._planner.lower_bounds(), dtype=float)
    p_hi = np.asarray(p._planner.upper_bounds(), dtype=float)
    np.testing.assert_allclose(p_lo, pin_lo[:7], atol=1e-4)
    np.testing.assert_allclose(p_hi, pin_hi[:7], atol=1e-4)


def test_single_soft_finger_geometry():
    """Sanity-check that the soft fingers don't overlap across the
    centerline of the hand — a regression test for the URDF
    finger-mounting transform."""
    pytest.importorskip("pinocchio")
    import pinocchio as pin

    from bimanual_franka_planning.single_franka_soft import (
        CHAIN_CONFIGS,
    )

    m = pin.buildModelFromUrdf(CHAIN_CONFIGS["single_fr3_soft"].urdf_path)
    data = m.createData()
    q = np.clip(pin.neutral(m), m.lowerPositionLimit, m.upperPositionLimit)
    pin.forwardKinematics(m, data, q)
    pin.updateFramePlacements(m, data)

    hand = data.oMf[m.getFrameId("fr3_hand")]
    left = data.oMf[m.getFrameId("fr3_leftfinger")]
    right = data.oMf[m.getFrameId("fr3_rightfinger")]

    left_in_hand = hand.actInv(left).translation
    right_in_hand = hand.actInv(right).translation

    # The two finger frames should sit symmetrically across the hand's
    # y-axis, with the inner faces at least 40 mm apart (i.e. the
    # 50 mm preset gap minus a bit of tolerance).
    assert left_in_hand[1] > 0 > right_in_hand[1]
    assert abs(left_in_hand[1] - right_in_hand[1]) >= 0.04
    # And both at the same z offset on the hand.
    np.testing.assert_allclose(left_in_hand[2], right_in_hand[2])
