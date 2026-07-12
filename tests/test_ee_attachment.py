"""End-effector sphere attachment tests.

Verifies that ``MotionPlanner.attach_ee_spheres`` flips a previously-valid
configuration into collision when the attached spheres overlap the rest
of the robot body, that ``detach_ee()`` undoes it, and that the side
resolution / input validation surface clean errors.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("bimanual_franka_planning._ompl_vamp")

from bimanual_franka_planning.planning import create_planner  # noqa: E402
from bimanual_franka_planning.types import PlannerConfig  # noqa: E402

# A large sphere centred well behind the EE link (down the -z of EE frame)
# is guaranteed to overlap the wrist / hand links for any reasonable home
# pose — the cleanest "must collide" probe for the body-vs-attachment
# check that doesn't depend on the cell layout.
_HUGE_BACKWARD = np.array([[0.0, 0.0, -0.3, 0.35]], dtype=np.float32)
# A tiny sphere half a metre out in front of the EE link is far enough
# from every body link to be irrelevant for any of the home-anchored
# configs the tests use.
_TINY_FAR = np.array([[0.0, 0.0, 0.5, 0.01]], dtype=np.float32)


def _bimanual_planner():
    return create_planner(
        "bimanual_fr3",
        config=PlannerConfig(planner_name="rrtc", time_limit=1.0),
    )


def _single_planner():
    return create_planner(
        "single_fr3",
        config=PlannerConfig(planner_name="rrtc", time_limit=1.0),
    )


def test_num_end_effectors():
    bp = _bimanual_planner()
    sp = _single_planner()
    assert bp.num_end_effectors == 2
    assert sp.num_end_effectors == 1
    assert not bp.has_attachment
    assert not sp.has_attachment


def test_tiny_attachment_keeps_home_valid_single(home_joints):
    from bimanual_franka_planning.single_franka import HOME_JOINTS as SINGLE_HOME

    p = _single_planner()
    home = SINGLE_HOME.copy()
    assert p.validate(home)
    p.attach_ee_spheres("ee", _TINY_FAR)
    assert p.has_attachment
    assert p.validate(home), "small far attachment should not invalidate home"


def test_huge_backward_attachment_invalidates_home_single():
    from bimanual_franka_planning.single_franka import HOME_JOINTS as SINGLE_HOME

    p = _single_planner()
    home = SINGLE_HOME.copy()
    assert p.validate(home)
    p.attach_ee_spheres("ee", _HUGE_BACKWARD)
    assert not p.validate(
        home
    ), "huge attachment behind the EE should hit the robot body at home"


def test_detach_restores_validity_single():
    from bimanual_franka_planning.single_franka import HOME_JOINTS as SINGLE_HOME

    p = _single_planner()
    home = SINGLE_HOME.copy()
    p.attach_ee_spheres("ee", _HUGE_BACKWARD)
    assert not p.validate(home)
    assert p.detach_ee()
    assert not p.has_attachment
    assert p.validate(home), "detaching should restore the original validity"
    # Detaching twice is a no-op return.
    assert not p.detach_ee()


def test_bimanual_left_and_right_attach_independently(home_joints):
    p = _bimanual_planner()
    assert p.validate(home_joints)
    p.attach_ee_spheres("left", _HUGE_BACKWARD)
    assert not p.validate(
        home_joints
    ), "left huge attachment should hit the left arm at home"
    p.detach_ee()
    assert p.validate(home_joints)
    p.attach_ee_spheres("right", _HUGE_BACKWARD)
    assert not p.validate(
        home_joints
    ), "right huge attachment should hit the right arm at home"


def test_attach_replaces_prior_attachment(home_joints):
    p = _bimanual_planner()
    p.attach_ee_spheres("left", _HUGE_BACKWARD)
    assert not p.validate(home_joints)
    # Replace with a benign attachment — validity should recover.
    p.attach_ee_spheres("left", _TINY_FAR)
    assert p.has_attachment
    assert p.validate(home_joints)


def test_attach_via_integer_index(home_joints):
    p = _bimanual_planner()
    p.attach_ee_spheres(1, _HUGE_BACKWARD)  # 1 == right
    assert not p.validate(home_joints)


def test_subgroup_planner_sees_attachment(home_joints):
    """A subgroup planner (left arm) should also flip validity when the
    left EE carries a huge attachment."""
    planner = create_planner(
        "bimanual_fr3_left_arm",
        config=PlannerConfig(planner_name="rrtc", time_limit=1.0),
        base_config=home_joints,
    )
    start = planner.extract_config(home_joints)
    assert planner.validate(start)
    planner.attach_ee_spheres("left", _HUGE_BACKWARD)
    assert not planner.validate(start)


def test_attach_with_transform(home_joints):
    """A transform that pushes the attachment far ahead of the EE should
    yield the same validity as supplying the same offset directly."""
    p = _bimanual_planner()
    tf = np.eye(4, dtype=np.float32)
    tf[2, 3] = 0.5  # translate sphere +0.5 m in EE frame
    p.attach_ee_spheres("left", np.array([[0.0, 0.0, 0.0, 0.01]], np.float32), tf)
    assert p.validate(home_joints)


def test_unknown_side_raises():
    p = _bimanual_planner()
    with pytest.raises(ValueError, match="Unknown end-effector side"):
        p.attach_ee_spheres("middle", _TINY_FAR)


def test_out_of_range_index_raises():
    p = _bimanual_planner()
    with pytest.raises(ValueError, match="out of range"):
        p.attach_ee_spheres(7, _TINY_FAR)


def test_bad_sphere_shape_raises():
    p = _bimanual_planner()
    with pytest.raises(ValueError, match=r"shape \(N, 4\)"):
        p.attach_ee_spheres("left", np.zeros((3, 3), dtype=np.float32))


def test_non_positive_radius_raises():
    p = _bimanual_planner()
    with pytest.raises(ValueError, match="radii"):
        p.attach_ee_spheres("left", np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32))


def test_bad_transform_shape_raises():
    p = _bimanual_planner()
    with pytest.raises(ValueError, match=r"shape \(4, 4\)"):
        p.attach_ee_spheres("left", _TINY_FAR, transform=np.eye(3, dtype=np.float32))
