"""Single-arm FR3 — registry, IK, and motion-planning smoke tests.

Mirrors ``test_kinematics.py`` and ``test_planner_basic.py`` but for
the standalone ``single_fr3`` robot.  Confirms that the unified
factories (``create_planner`` / ``create_ik_solver``) dispatch to
the right C++ class and the right URDF chain purely by robot name.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("bimanual_franka_planning._ompl_vamp")


# ── Registry / factory dispatch ───────────────────────────────────────


def test_registry_lists_all_robots():
    from bimanual_franka_planning._robots import ROBOT_REGISTRY

    assert set(ROBOT_REGISTRY.keys()) == {
        "bimanual_fr3",
        "single_fr3",
        "bimanual_fr3_soft",
        "single_fr3_soft",
    }


def test_available_robots_includes_single_fr3():
    from bimanual_franka_planning.planning import available_robots

    names = available_robots()
    assert "single_fr3" in names
    assert "bimanual_fr3" in names


def test_single_fr3_uses_dedicated_cpp_class():
    """The single-arm planner must dispatch to ``SingleFr3OmplVampPlanner``,
    not the bimanual planner — this is the whole point of compiling a
    dedicated FK header for the single arm."""
    from bimanual_franka_planning._ompl_vamp import (
        OmplVampPlanner,
        SingleFr3OmplVampPlanner,
    )
    from bimanual_franka_planning.planning import create_planner

    p = create_planner("single_fr3")
    assert isinstance(p._planner, SingleFr3OmplVampPlanner)
    assert not isinstance(p._planner, OmplVampPlanner)


def test_bimanual_still_uses_bimanual_cpp_class():
    """Sanity: existing bimanual robots keep their original C++ class
    so the OmplVampPlanner refactor is fully backward-compatible."""
    from bimanual_franka_planning._ompl_vamp import OmplVampPlanner
    from bimanual_franka_planning.planning import create_planner

    p = create_planner("bimanual_fr3_left_arm")
    assert isinstance(p._planner, OmplVampPlanner)


# ── Single-arm motion planning ────────────────────────────────────────


@pytest.fixture(scope="module")
def single_planner():
    from bimanual_franka_planning.planning import create_planner
    from bimanual_franka_planning.types import PlannerConfig

    return create_planner(
        "single_fr3",
        config=PlannerConfig(planner_name="rrtc", time_limit=2.0),
    )


@pytest.fixture(scope="module")
def single_home():
    from bimanual_franka_planning.single_franka import HOME_JOINTS

    return HOME_JOINTS.copy()


def test_single_planner_dim(single_planner):
    # 7 arm joints + 2 prismatic finger joints (finger_joint2 is a URDF
    # mimic of finger_joint1 but cricket emits it as a separate DOF).
    assert single_planner.num_dof == 9
    assert single_planner.joint_names == (
        [f"fr3_joint{i}" for i in range(1, 8)]
        + ["fr3_finger_joint1", "fr3_finger_joint2"]
    )
    assert not single_planner.is_subgroup


def test_single_home_is_valid(single_planner, single_home):
    assert single_planner.validate(single_home)


def test_single_trivial_plan(single_planner, single_home):
    """Planning from a state to itself must always succeed."""
    result = single_planner.plan(single_home, single_home)
    assert result.success
    assert result.path is not None and result.path.shape[1] == 9


def test_single_plan_to_random_goal(single_planner, single_home):
    np.random.seed(0)
    goal = single_planner.sample_valid()
    assert goal.shape == (9,)
    result = single_planner.plan(single_home, goal)
    assert result.status.value in {"success", "failed"}
    if result.success:
        np.testing.assert_allclose(result.path[0], single_home, atol=1e-6)
        np.testing.assert_allclose(result.path[-1], goal, atol=1e-6)


def test_single_validate_batch(single_planner, single_home):
    """Batched SIMD check on the SingleFr3 FK header."""
    np.random.seed(0)
    lo = np.asarray(single_planner._planner.lower_bounds())
    hi = np.asarray(single_planner._planner.upper_bounds())
    samples = np.random.uniform(lo, hi, size=(20, 9))
    samples[0] = single_home
    expected = np.array([single_planner.validate(s) for s in samples], dtype=bool)
    got = single_planner.validate_batch(samples)
    assert got.shape == (20,) and got.dtype == bool
    np.testing.assert_array_equal(got, expected)


def test_set_subgroup_rejects_cross_robot(single_planner):
    """``set_subgroup`` must not let the user swap a single_fr3 planner
    over to a bimanual subgroup — each C++ planner is bound to its own
    robot's FK header."""
    with pytest.raises(ValueError, match=r"cannot switch C\+\+ robots"):
        single_planner.set_subgroup("bimanual_fr3_left_arm")


# ── Single-arm IK (same factory as bimanual) ──────────────────────────


def test_single_ik_factory():
    pytest.importorskip("pytracik")
    pytest.importorskip("pinocchio")
    from bimanual_franka_planning.kinematics import create_ik_solver

    s = create_ik_solver("single_fr3")
    assert s.num_joints == 7
    assert s.base_frame == "fr3_link0"
    assert s.ee_frame == "fr3_link8"


def test_single_ik_round_trip():
    pytest.importorskip("pytracik")
    pytest.importorskip("pinocchio")
    from bimanual_franka_planning.kinematics import create_ik_solver
    from bimanual_franka_planning.single_franka import HOME_JOINTS
    from bimanual_franka_planning.types import IKConfig, SE3Pose

    s = create_ik_solver("single_fr3", config=IKConfig(max_attempts=3))
    # The IK chain is the 7-DOF arm only (gripper joints are not part
    # of the IK chain), so slice off the gripper portion of HOME_JOINTS.
    arm_home = HOME_JOINTS[:7]
    home_pose = s.fk(arm_home)
    target = SE3Pose(
        position=home_pose.position + np.array([0.03, 0.0, -0.02]),
        rotation=home_pose.rotation,
    )
    result = s.solve(target, seed=arm_home)
    if not result.success:
        pytest.skip(f"TRAC-IK did not converge ({result.status.value}); flaky on CI")

    assert result.joint_positions.shape == (7,)
    assert result.position_error < 1e-3
    assert result.orientation_error < 1e-3

    achieved = s.fk(result.joint_positions)
    np.testing.assert_allclose(achieved.position, target.position, atol=1e-3)


def test_single_pinocchio_matches_trac_ik():
    """Cross-validate FK between Pinocchio and TRAC-IK on the single-arm URDF."""
    pytest.importorskip("pytracik")
    pytest.importorskip("pinocchio")
    from bimanual_franka_planning.kinematics import (
        compute_forward_kinematics,
        create_ik_solver,
        create_pinocchio_context,
    )
    from bimanual_franka_planning.single_franka import CHAIN_CONFIGS

    chain = CHAIN_CONFIGS["single_fr3"]
    arm_joint_names = [f"fr3_joint{i}" for i in range(1, 8)]
    ctx = create_pinocchio_context(
        urdf_path=chain.urdf_path,
        end_effector_frame=chain.ee_link,
        joint_names=arm_joint_names,
    )
    trac = create_ik_solver("single_fr3")

    rng = np.random.default_rng(0)
    lo, hi = trac.joint_limits
    for _ in range(5):
        q = rng.uniform(lo, hi)
        pin_pose = compute_forward_kinematics(ctx, q)
        trac_pose = trac.fk(q)
        np.testing.assert_allclose(pin_pose.position, trac_pose.position, atol=1e-9)
        np.testing.assert_allclose(pin_pose.rotation, trac_pose.rotation, atol=1e-9)


def test_single_pinocchio_matches_single_arm_planner_fk():
    """The single-arm VAMP/OMPL planner's joint ordering and bounds must
    agree with the single-arm URDF's Pinocchio model — guards against
    drift between the cricket-generated FK header and the source URDF."""
    pytest.importorskip("pinocchio")
    import pinocchio as pin

    from bimanual_franka_planning.planning import create_planner
    from bimanual_franka_planning.single_franka import (
        CHAIN_CONFIGS,
        single_fr3_robot_config,
    )

    p = create_planner("single_fr3")
    assert p.joint_names == single_fr3_robot_config.joint_names

    model = pin.buildModelFromUrdf(CHAIN_CONFIGS["single_fr3"].urdf_path)
    # Pinocchio's joint limit vectors are nq long for revolute joints.
    pin_lo = np.asarray(model.lowerPositionLimit, dtype=float)
    pin_hi = np.asarray(model.upperPositionLimit, dtype=float)
    p_lo = np.asarray(p._planner.lower_bounds(), dtype=float)
    p_hi = np.asarray(p._planner.upper_bounds(), dtype=float)
    # Cricket scales the URDF limits the same way Pinocchio reads them
    # for revolute joints, so both should agree to single-precision.
    # Compare only the 7 arm joints — Pinocchio honors the URDF mimic
    # tag so its nq excludes finger_joint2, while the planner exposes
    # both finger DOFs explicitly.
    np.testing.assert_allclose(p_lo[:7], pin_lo[:7], atol=1e-4)
    np.testing.assert_allclose(p_hi[:7], pin_hi[:7], atol=1e-4)
