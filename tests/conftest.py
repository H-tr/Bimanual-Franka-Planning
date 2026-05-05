"""Shared pytest fixtures.

The native ``_ompl_vamp`` extension and the URDFs it depends on are
built by the project's CMake / scikit-build pipeline.  Tests that need
the extension import it lazily through ``planner_factory`` so missing
artefacts surface as a clean *skip* rather than a collection error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def home_joints() -> np.ndarray:
    from bimanual_franka_planning.bimanual_franka import HOME_JOINTS

    return HOME_JOINTS.copy()


@pytest.fixture(scope="session")
def left_arm_planner():
    """Left-arm planner with a fast RRT-Connect config — shared across tests."""
    pytest.importorskip("bimanual_franka_planning._ompl_vamp")
    from bimanual_franka_planning.planning import create_planner
    from bimanual_franka_planning.types import PlannerConfig

    return create_planner(
        "bimanual_fr3_left_arm",
        config=PlannerConfig(planner_name="rrtc", time_limit=2.0),
    )


@pytest.fixture(scope="session")
def left_arm_start(left_arm_planner, home_joints) -> np.ndarray:
    return left_arm_planner.extract_config(home_joints)


@pytest.fixture(scope="session")
def table_pointcloud(repo_root) -> np.ndarray:
    """Bundled table.ply rotated and shifted in front of the robot.

    Mirrors ``examples/motion_planning_example.py::load_table`` so the
    motion-planning tests exercise the same geometry the demo uses.
    """
    trimesh = pytest.importorskip("trimesh")
    import bimanual_franka_planning

    pkg_root = Path(bimanual_franka_planning.__file__).parent
    pcd = trimesh.load(str(pkg_root / "resources" / "envs" / "pcd" / "table.ply"))
    pts = np.asarray(pcd.vertices, dtype=np.float32)
    pts = pts - pts.mean(axis=0)
    rot = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    pts = pts @ rot.T
    # Shift the table forward of the left-arm base; height chosen so the
    # FR3 ``ready`` HOME pose (flange ~0.31 m forward, ~0.59 m up) is
    # collision-free but the table still occupies the natural reach
    # zone the planner has to weave around.
    pts[:, 0] += 0.85
    pts[:, 2] += 0.20
    return pts
