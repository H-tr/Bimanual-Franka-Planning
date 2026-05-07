"""Robot registry — aggregates every bundled description module.

Each robot ships as a parallel description module (``bimanual_franka``,
``single_franka``, …) following the same shape: ``RobotConfig``,
``CHAIN_CONFIGS``, ``HOME_JOINTS``, ``PLANNING_SUBGROUPS``.  This
module pulls them together behind a single dispatch table so
:func:`create_planner` and :func:`create_ik_solver` can resolve a
robot purely by its string name without any module-specific
branching.

Adding a new robot is a two-step change:

  1. Drop a description module next to ``bimanual_franka.py`` and
     ``single_franka.py`` with the same shape.
  2. Append an entry to :data:`ROBOT_REGISTRY` below.

Subgroup planners (``bimanual_fr3_left_arm`` / ``bimanual_fr3_right_arm``
/ ``bimanual_fr3_dual_arm``) are auto-derived from each robot's
``PLANNING_SUBGROUPS`` dict; a robot that doesn't expose subgroups
(like the single-arm FR3) just leaves the dict empty.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from bimanual_franka_planning.types import ChainConfig, RobotConfig


@dataclass(frozen=True)
class RobotEntry:
    """Per-robot metadata used by the planner and IK factories."""

    name: str
    """Public string name — what callers pass to ``create_planner``."""

    description: ModuleType
    """The description module (``bimanual_franka``, ``single_franka``, …).

    Provides ``HOME_JOINTS``, ``CHAIN_CONFIGS``, ``PLANNING_SUBGROUPS``
    and a top-level ``RobotConfig`` instance.
    """

    robot_config: "RobotConfig"
    """The top-level :class:`RobotConfig` for this robot."""

    cpp_planner_cls_name: str
    """Name of the matching ``OmplVampPlanner<Robot>`` class exported
    from ``bimanual_franka_planning._ompl_vamp``."""

    @property
    def home_joints(self) -> "np.ndarray":
        return self.description.HOME_JOINTS

    @property
    def planning_subgroups(self) -> dict:
        return self.description.PLANNING_SUBGROUPS

    @property
    def chain_configs(self) -> dict[str, "ChainConfig"]:
        return self.description.CHAIN_CONFIGS

    @property
    def joint_names(self) -> list[str]:
        return self.robot_config.joint_names


def _build_registry() -> dict[str, RobotEntry]:
    from bimanual_franka_planning import (
        bimanual_franka,
        bimanual_franka_soft,
        single_franka,
        single_franka_soft,
    )

    entries = [
        RobotEntry(
            name="bimanual_fr3",
            description=bimanual_franka,
            robot_config=bimanual_franka.bimanual_fr3_robot_config,
            cpp_planner_cls_name="OmplVampPlanner",
        ),
        RobotEntry(
            name="single_fr3",
            description=single_franka,
            robot_config=single_franka.single_fr3_robot_config,
            cpp_planner_cls_name="SingleFr3OmplVampPlanner",
        ),
        RobotEntry(
            name="bimanual_fr3_soft",
            description=bimanual_franka_soft,
            robot_config=bimanual_franka_soft.bimanual_fr3_soft_robot_config,
            cpp_planner_cls_name="BimanualFr3SoftOmplVampPlanner",
        ),
        RobotEntry(
            name="single_fr3_soft",
            description=single_franka_soft,
            robot_config=single_franka_soft.single_fr3_soft_robot_config,
            cpp_planner_cls_name="SingleFr3SoftOmplVampPlanner",
        ),
    ]
    return {e.name: e for e in entries}


ROBOT_REGISTRY: dict[str, RobotEntry] = _build_registry()


def get_robot(name: str) -> RobotEntry:
    """Look up a robot by name, raising a helpful error if unknown."""
    if name in ROBOT_REGISTRY:
        return ROBOT_REGISTRY[name]
    # A subgroup name resolves to its parent robot (e.g.
    # ``bimanual_fr3_left_arm`` → ``bimanual_fr3``).
    parent = _resolve_parent_robot(name)
    if parent is not None:
        return parent
    available = ", ".join(sorted(_all_planner_names()))
    raise ValueError(f"Unknown robot '{name}'. Available: {available}")


def _resolve_parent_robot(planner_name: str) -> RobotEntry | None:
    """Find the ``RobotEntry`` whose ``PLANNING_SUBGROUPS`` contains ``planner_name``."""
    for entry in ROBOT_REGISTRY.values():
        if planner_name in entry.planning_subgroups:
            return entry
    return None


def _all_planner_names() -> list[str]:
    """Every name accepted by ``create_planner`` — robots + their subgroups."""
    names = set(ROBOT_REGISTRY.keys())
    for entry in ROBOT_REGISTRY.values():
        names.update(entry.planning_subgroups.keys())
    return sorted(names)


def all_chain_configs() -> dict[str, "ChainConfig"]:
    """Merge every robot's ``CHAIN_CONFIGS`` into one dict.

    Used by the IK factory so callers can ask for a chain by name
    without knowing which description module owns it.
    """
    merged: dict[str, "ChainConfig"] = {}
    for entry in ROBOT_REGISTRY.values():
        for k, v in entry.chain_configs.items():
            if k in merged and merged[k] != v:
                raise RuntimeError(
                    f"Chain name '{k}' is registered by multiple robots with "
                    f"different configs — rename it in one of the description "
                    f"modules to disambiguate."
                )
            merged[k] = v
    return merged
