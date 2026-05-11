"""OMPL + VAMP motion planner.

Uses OMPL for planning algorithms and VAMP for SIMD-accelerated
collision checking.  The entire planning pipeline runs in C++ via
the ``_ompl_vamp`` extension — Python only crosses the boundary
once per ``plan()`` call.

Supports subgroup planning: each subgroup operates on a reduced
state space while frozen joints are expanded to the full 17-DOF
config before collision checks.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from bimanual_franka_planning.types import PlannerConfig, PlanningResult, PlanningStatus


@runtime_checkable
class MotionPlannerBase(Protocol):
    """Protocol for motion planner backends."""

    @property
    def robot_name(self) -> str:
        ...

    @property
    def num_dof(self) -> int:
        ...

    def plan(self, start: np.ndarray, goal: np.ndarray) -> PlanningResult:
        ...

    def validate(self, configuration: np.ndarray) -> bool:
        ...


class MotionPlanner:
    """Motion planner using OMPL + VAMP C++ backend.

    All internals are private.  The public API accepts and returns
    only numpy arrays.

    For subgroup planners the helper methods :meth:`extract_config`,
    :meth:`embed_config`, and :meth:`embed_path` convert between the
    reduced DOF space used by the planner and the full 17-DOF state
    configuration.
    """

    def __init__(
        self,
        robot_name: str,
        config: PlannerConfig | None = None,
        pointcloud: np.ndarray | None = None,
        base_config: np.ndarray | None = None,
        constraints: list | None = None,
        costs: list | None = None,
    ) -> None:
        import bimanual_franka_planning._ompl_vamp as _ext
        from bimanual_franka_planning._robots import get_robot

        if config is None:
            config = PlannerConfig()

        self._config = config
        self._robot_name = robot_name

        # Resolve which robot this planner is for.  ``robot_name`` may
        # be either a top-level robot key (``"bimanual_fr3"``,
        # ``"single_fr3"``) or a subgroup of one (``"bimanual_fr3_left_arm"``).
        robot = get_robot(robot_name)
        self._robot = robot
        cpp_planner_cls = getattr(_ext, robot.cpp_planner_cls_name)

        # Frozen full-body joint values for any joint not controlled by
        # this planner.  Defaults to the robot's HOME, but the caller
        # can pass any full-DOF array — e.g. the live config from the
        # env — so the inactive joints are pinned wherever they
        # currently are.
        home = robot.home_joints
        if base_config is None:
            base_config = home
        self._base_config = np.asarray(base_config, dtype=np.float64).copy()
        if self._base_config.shape != home.shape:
            raise ValueError(
                f"base_config must have shape {home.shape} for robot "
                f"'{robot.name}', got {self._base_config.shape}"
            )

        full_names = robot.joint_names
        sg = robot.planning_subgroups.get(robot_name)

        if sg is None:
            # Full-body planner
            self._planner = cpp_planner_cls()
            self._joint_names = list(full_names)
            self._subgroup_indices = None
        else:
            # Subgroup planner — frozen joints come from the supplied
            # base_config; the C++ checker injects them around the
            # active subset before every collision query.
            sg_joint_names = sg["joints"]
            active_indices = [full_names.index(j) for j in sg_joint_names]
            self._planner = cpp_planner_cls(active_indices, self._base_config.tolist())
            self._joint_names = list(sg_joint_names)
            self._subgroup_indices = np.array(active_indices)

        self._ndof = self._planner.dimension()

        if pointcloud is not None:
            r_min, r_max = self._planner.min_max_radii()
            self._planner.add_pointcloud(
                np.asarray(pointcloud, dtype=np.float32).tolist(),
                r_min,
                r_max,
                config.point_radius,
            )

        if constraints:
            self._push_constraints(constraints)

        if costs:
            self._push_costs(costs)

    # ── Properties ────────────────────────────────────────────────────

    @property
    def robot_name(self) -> str:
        return self._robot_name

    @property
    def num_dof(self) -> int:
        return self._ndof

    @property
    def joint_names(self) -> list[str]:
        """Joint names controlled by this planner, in DOF order."""
        return self._joint_names

    @property
    def is_subgroup(self) -> bool:
        """True if this planner controls a subset of the full body."""
        return self._subgroup_indices is not None

    @property
    def subgroup_indices(self) -> np.ndarray | None:
        """Indices of this planner's joints in the full 17-DOF config."""
        return self._subgroup_indices

    @property
    def base_config(self) -> np.ndarray:
        """The 17-DOF stance frozen for joints outside this subgroup."""
        return self._base_config.copy()

    # ── Constraint integration ────────────────────────────────────────

    def _push_constraints(self, constraints) -> None:
        """Push compiled CasADi constraints to the C++ planner."""
        from bimanual_franka_planning.planning.constraints import Constraint

        for c in constraints:
            if not isinstance(c, Constraint):
                raise TypeError(
                    f"constraints must be Constraint instances from "
                    f"bimanual_franka_planning.planning.constraints; "
                    f"got {type(c).__name__}"
                )
            if c.ambient_dim != self._ndof:
                raise ValueError(
                    f"Constraint ambient_dim ({c.ambient_dim}) does not match "
                    f"planner active dimension ({self._ndof}).  Build the "
                    f"Constraint with a SymbolicContext for the same subgroup."
                )
            self._planner.add_compiled_constraint(
                str(c.so_path),
                c.symbol_name,
                c.ambient_dim,
                c.co_dim,
            )

    def clear_constraints(self) -> None:
        """Remove all constraints from the planner."""
        self._planner.clear_constraints()

    def set_constraints(self, constraints: list) -> None:
        """Replace all constraints: clear existing, then push new ones."""
        self.clear_constraints()
        self._push_constraints(constraints)

    # ── Cost integration ──────────────────────────────────────────────

    def _push_costs(self, costs) -> None:
        """Push compiled CasADi costs to the C++ planner."""
        from bimanual_franka_planning.planning.costs import Cost

        for c in costs:
            if not isinstance(c, Cost):
                raise TypeError(
                    f"costs must be Cost instances from "
                    f"bimanual_franka_planning.planning.costs; "
                    f"got {type(c).__name__}"
                )
            if c.ambient_dim != self._ndof:
                raise ValueError(
                    f"Cost ambient_dim ({c.ambient_dim}) does not match "
                    f"planner active dimension ({self._ndof}).  Build the "
                    f"Cost with a SymbolicContext for the same subgroup."
                )
            self._planner.add_compiled_cost(
                str(c.so_path),
                c.symbol_name,
                c.ambient_dim,
                float(c.weight),
            )

    def clear_costs(self) -> None:
        """Remove all costs from the planner (falls back to path length)."""
        self._planner.clear_costs()

    def set_costs(self, costs: list) -> None:
        """Replace all costs: clear existing, then push new ones."""
        self.clear_costs()
        self._push_costs(costs)

    # ── Pointcloud environment ────────────────────────────────────────

    def add_pointcloud(self, pointcloud: np.ndarray) -> None:
        """Set the scene pointcloud, replacing any previously-registered cloud.

        Args:
            pointcloud: ``(N, 3)`` array of obstacle positions in world
                frame.  Uses ``config.point_radius`` as the per-point
                inflation radius.
        """
        r_min, r_max = self._planner.min_max_radii()
        self._planner.add_pointcloud(
            np.asarray(pointcloud, dtype=np.float32).tolist(),
            r_min,
            r_max,
            self._config.point_radius,
        )

    def remove_pointcloud(self) -> bool:
        """Drop the currently-registered pointcloud.

        Returns ``False`` if there was no cloud to remove.
        """
        return self._planner.remove_pointcloud()

    @property
    def has_pointcloud(self) -> bool:
        """True if a pointcloud is currently registered."""
        return self._planner.has_pointcloud()

    # ── End-effector sphere attachment ───────────────────────────────

    def attach_ee_spheres(
        self,
        side: "str | int",
        spheres: np.ndarray,
        transform: np.ndarray | None = None,
    ) -> None:
        """Attach a set of spheres to an end-effector.

        The spheres move with the gripper and are collision-checked
        against the world *and* against the rest of the robot body for
        every state and motion edge VAMP evaluates.  Each planner
        instance holds at most one attachment — calling this method
        replaces any prior attachment.

        Args:
            side: Which end-effector to attach to.  Accepts a string key
                from the robot's ``ee_index`` map (``"left"``, ``"right"``,
                or ``"ee"`` for single-arm robots) or the integer index
                directly.
            spheres: ``(N, 4)`` array of ``[x, y, z, radius]`` rows
                expressed in the EE link frame (after applying
                ``transform``).
            transform: Optional ``(4, 4)`` isometry applied to the sphere
                positions inside the EE frame.  Defaults to the identity.
        """
        ee_index = self._resolve_ee_index(side)

        spheres_arr = np.asarray(spheres, dtype=np.float32)
        if spheres_arr.ndim != 2 or spheres_arr.shape[1] != 4:
            raise ValueError(
                f"spheres must have shape (N, 4) — got {spheres_arr.shape}"
            )
        if spheres_arr.size and float(np.min(spheres_arr[:, 3])) <= 0.0:
            raise ValueError("sphere radii must be strictly positive")

        if transform is None:
            tf = np.eye(4, dtype=np.float32)
        else:
            tf = np.asarray(transform, dtype=np.float32)
            if tf.shape != (4, 4):
                raise ValueError(f"transform must have shape (4, 4) — got {tf.shape}")

        self._planner.attach_ee_spheres(
            ee_index,
            tf.reshape(-1).tolist(),
            spheres_arr.tolist(),
        )

    def detach_ee(self) -> bool:
        """Drop the current EE attachment, if any.

        Returns ``False`` if no attachment was registered.
        """
        return self._planner.detach_ee()

    @property
    def has_attachment(self) -> bool:
        """True if an EE attachment is currently registered."""
        return self._planner.has_attachment()

    @property
    def num_end_effectors(self) -> int:
        """Number of end-effectors this robot exposes for attachment."""
        return self._planner.num_end_effectors()

    def _resolve_ee_index(self, side: "str | int") -> int:
        if isinstance(side, str):
            mapping = self._robot.ee_index
            if side not in mapping:
                valid = ", ".join(repr(k) for k in mapping)
                raise ValueError(
                    f"Unknown end-effector side {side!r} for robot "
                    f"'{self._robot.name}'.  Valid sides: {valid}."
                )
            ee_index = int(mapping[side])
        else:
            ee_index = int(side)
        n = self.num_end_effectors
        if ee_index < 0 or ee_index >= n:
            raise ValueError(
                f"ee_index {ee_index} out of range for robot "
                f"'{self._robot.name}' (n_end_effectors = {n})."
            )
        return ee_index

    # ── Point cloud filtering ────────────────────────────────────────

    def filter_pointcloud(
        self,
        pointcloud: np.ndarray,
        min_dist: float,
        max_range: float,
        origin: np.ndarray | list[float],
        workspace_min: np.ndarray | list[float],
        workspace_max: np.ndarray | list[float],
        cull: bool = True,
    ) -> np.ndarray:
        """Spatially downsample a point cloud via Morton-curve sorting.

        Keeps one representative point per ``min_dist`` neighbourhood and
        discards points farther than ``max_range`` from ``origin`` or
        outside the ``[workspace_min, workspace_max]`` bounding box.

        Args:
            pointcloud: ``(N, 3)`` array of 3-D points.
            min_dist: Minimum distance between two retained points.
            max_range: Maximum distance from ``origin`` to keep a point.
            origin: ``(3,)`` reference position for range culling.
            workspace_min: ``(3,)`` lower corner of the workspace AABB.
            workspace_max: ``(3,)`` upper corner of the workspace AABB.
            cull: If ``True`` (default), apply range and AABB culling.

        Returns:
            ``(M, 3)`` filtered point cloud with ``M <= N``.
        """
        pts = np.asarray(pointcloud, dtype=np.float32).tolist()
        origin = [float(x) for x in origin]
        workspace_min = [float(x) for x in workspace_min]
        workspace_max = [float(x) for x in workspace_max]
        filtered = self._planner.filter_pointcloud(
            pts,
            float(min_dist),
            float(max_range),
            origin,
            workspace_min,
            workspace_max,
            cull,
        )
        return np.asarray(filtered, dtype=np.float32)

    def filter_self_from_pointcloud(
        self,
        pointcloud: np.ndarray,
        point_radius: float,
        config: np.ndarray,
    ) -> np.ndarray:
        """Remove points that collide with the robot body or environment.

        Computes forward kinematics at ``config``, then drops every
        point whose inflated sphere (radius ``point_radius``) overlaps
        any robot collision sphere or any registered obstacle.

        Args:
            pointcloud: ``(N, 3)`` array of 3-D points.
            point_radius: Inflation radius for each point.
            config: Active-DOF configuration (same space as ``plan``).

        Returns:
            ``(M, 3)`` filtered point cloud with ``M <= N``.
        """
        pts = np.asarray(pointcloud, dtype=np.float32).tolist()
        config = np.asarray(config, dtype=np.float64)
        if len(config) != self._ndof:
            raise ValueError(f"config has {len(config)} DOF, expected {self._ndof}")
        filtered = self._planner.filter_self_from_pointcloud(
            pts,
            float(point_radius),
            config.tolist(),
        )
        return np.asarray(filtered, dtype=np.float32)

    # ── Subgroup switching ───────────────────────────────────────────

    def set_subgroup(
        self,
        robot_name: str,
        base_config: np.ndarray | None = None,
    ) -> None:
        """Switch active joints without rebuilding the collision environment.

        Clears all constraints.  The pointcloud is preserved.  ``robot_name``
        must resolve to the same underlying C++ robot — switching from
        a bimanual subgroup to a single-arm robot (or vice versa)
        requires building a new ``MotionPlanner`` since each robot
        has its own dedicated VAMP FK.

        Args:
            robot_name: Subgroup name from this robot's
                ``PLANNING_SUBGROUPS`` (e.g. ``"bimanual_fr3_left_arm"``),
                or the robot's own name (``"bimanual_fr3"``,
                ``"single_fr3"``) for the full state.
            base_config: Full-DOF frozen config for inactive joints.
                Defaults to the previously stored base config.
        """
        from bimanual_franka_planning._robots import get_robot

        robot = get_robot(robot_name)
        if robot.name != self._robot.name:
            raise ValueError(
                f"set_subgroup cannot switch C++ robots — this planner is "
                f"bound to '{self._robot.name}', got '{robot_name}' which "
                f"resolves to '{robot.name}'.  Build a new MotionPlanner "
                f"for the other robot."
            )

        if base_config is not None:
            self._base_config = np.asarray(base_config, dtype=np.float64).copy()
        self._robot_name = robot_name

        full_names = robot.joint_names
        sg = robot.planning_subgroups.get(robot_name)

        if sg is None:
            self._planner.set_full_body()
            self._joint_names = list(full_names)
            self._subgroup_indices = None
        else:
            sg_joint_names = sg["joints"]
            active_indices = [full_names.index(j) for j in sg_joint_names]
            self._planner.set_subgroup(active_indices, self._base_config.tolist())
            self._joint_names = list(sg_joint_names)
            self._subgroup_indices = np.array(active_indices)

        self._ndof = self._planner.dimension()

    # ── Subgroup helpers ──────────────────────────────────────────────

    def extract_config(self, full_config: np.ndarray) -> np.ndarray:
        """Extract this planner's joints from a full 17-DOF configuration."""
        full_config = np.asarray(full_config, dtype=np.float64)
        if self._subgroup_indices is None:
            return full_config.copy()
        return full_config[self._subgroup_indices].copy()

    def embed_config(
        self,
        config: np.ndarray,
        base_config: np.ndarray | None = None,
    ) -> np.ndarray:
        """Embed a subgroup config into a full 17-DOF configuration.

        ``base_config`` defaults to the planner's stored base — the same
        17-DOF values the C++ collision checker injects for inactive
        joints — so the embedded config matches what was validated.
        """
        config = np.asarray(config, dtype=np.float64)
        if self._subgroup_indices is None:
            return config.copy()

        if base_config is None:
            base_config = self._base_config
        full = np.array(base_config, dtype=np.float64)
        full[self._subgroup_indices] = config
        return full

    def embed_path(
        self,
        path: np.ndarray,
        base_config: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert a subgroup path ``(N, sub_dof)`` to ``(N, 24)``.

        ``base_config`` defaults to the planner's stored base — the same
        17-DOF values the C++ collision checker injects for inactive
        joints — so the embedded path matches what was validated.
        """
        path = np.asarray(path, dtype=np.float64)
        if self._subgroup_indices is None:
            return path.copy()

        if base_config is None:
            base_config = self._base_config
        n = path.shape[0]
        full_path = np.tile(np.array(base_config, dtype=np.float64), (n, 1))
        full_path[:, self._subgroup_indices] = path
        return full_path

    # ── Planning ──────────────────────────────────────────────────────

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        time_limit: float | None = None,
    ) -> PlanningResult:
        """Plan a collision-free path from start to goal.

        Args:
            start: Start configuration (active DOF).
            goal: Goal configuration (active DOF).
            time_limit: Optional per-call override for the solver time
                limit.  Defaults to ``self._config.time_limit``.
        """
        start = np.asarray(start, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)

        if len(start) != self._ndof:
            raise ValueError(f"start has {len(start)} DOF, expected {self._ndof}")
        if len(goal) != self._ndof:
            raise ValueError(f"goal has {len(goal)} DOF, expected {self._ndof}")

        if not self._planner.validate(start.tolist()):
            return PlanningResult(
                status=PlanningStatus.INVALID_START,
                path=None,
                planning_time_ns=0,
                iterations=0,
                path_cost=float("inf"),
            )
        if not self._planner.validate(goal.tolist()):
            return PlanningResult(
                status=PlanningStatus.INVALID_GOAL,
                path=None,
                planning_time_ns=0,
                iterations=0,
                path_cost=float("inf"),
            )

        if time_limit is None:
            time_limit = self._config.time_limit

        result = self._planner.plan(
            start.tolist(),
            goal.tolist(),
            self._config.planner_name,
            time_limit,
            self._config.simplify,
            self._config.interpolate,
            self._config.interpolate_count,
            self._config.resolution,
        )

        if not result.solved:
            return PlanningResult(
                status=PlanningStatus.FAILED,
                path=None,
                planning_time_ns=result.planning_time_ns,
                iterations=0,
                path_cost=float("inf"),
            )

        path_np = np.array(result.path, dtype=np.float64)

        return PlanningResult(
            status=PlanningStatus.SUCCESS,
            path=path_np,
            planning_time_ns=result.planning_time_ns,
            iterations=0,
            path_cost=result.path_cost,
        )

    def simplify_path(self, path: np.ndarray, time_limit: float = 1.0) -> np.ndarray:
        """Run OMPL's shortcut-based path simplifier on ``path``.

        Same pipeline ``plan(..., simplify=True)`` uses internally
        (``reduceVertices`` + ``collapseCloseVertices`` + ``shortcutPath``
        + B-spline smoothing), but detached so you can apply it to any
        path you already have — e.g. replay an old plan with a
        different collision environment.

        Shortcuts only consult the motion validator.  Custom soft
        costs (:class:`Cost`) are ignored; for cost-driven plans, run
        :meth:`plan` with ``simplify=False`` and leave the path
        untouched unless you've explicitly decided shortcut shaping
        is acceptable.

        Args:
            path: ``(N, ndof)`` array of waypoints in the planner's
                active DOF space.
            time_limit: Wall-clock budget for the simplifier, seconds.

        Returns:
            ``(M, ndof)`` simplified waypoint array with ``M <= N``.
        """
        path = np.asarray(path, dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != self._ndof:
            raise ValueError(
                f"path must have shape (N, {self._ndof}), got {path.shape}"
            )
        simp = self._planner.simplify_path(path.tolist(), float(time_limit))
        return np.array(simp, dtype=np.float64)

    def interpolate_path(
        self,
        path: np.ndarray,
        count: int = 0,
        resolution: float = 64.0,
    ) -> np.ndarray:
        """Densify ``path`` with uniform waypoints along every edge.

        Three modes (pick one; the other must be zero):

            * ``count > 0``        — exactly that many total waypoints
              distributed proportionally to edge length.
            * ``resolution > 0.0`` — ``ceil(edge_length * resolution)``
              waypoints per edge (uniform density in state-space
              distance — the default).
            * both ``0``           — OMPL's default longest-valid-segment
              fraction.

        Uses ``StateSpace::interpolate`` internally, so the inserted
        states stay on the constraint manifold for projected state
        spaces as well as flat ones.  No collision check is performed
        — the densification only lifts points along the existing
        piecewise-linear edges.

        Args:
            path: ``(N, ndof)`` waypoint array.
            count: Exact total waypoint count if ``> 0``.
            resolution: Waypoints per unit state-space distance if
                ``> 0.0``.

        Returns:
            ``(M, ndof)`` densified waypoint array with ``M >= N``.
        """
        path = np.asarray(path, dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != self._ndof:
            raise ValueError(
                f"path must have shape (N, {self._ndof}), got {path.shape}"
            )
        dense = self._planner.interpolate_path(
            path.tolist(), int(count), float(resolution)
        )
        return np.array(dense, dtype=np.float64)

    def validate(self, configuration: np.ndarray) -> bool:
        """Check if a configuration is collision-free."""
        configuration = np.asarray(configuration, dtype=np.float64)
        return self._planner.validate(configuration.tolist())

    def validate_batch(self, configurations: np.ndarray) -> np.ndarray:
        """Batched collision check — one SIMD block per ``rake`` configs.

        Packs ``rake`` distinct configurations into a single VAMP
        ``ConfigurationBlock<rake>`` and runs one ``fkcc<rake>`` call
        per block, so ``N`` queries cost ``ceil(N / rake)`` SIMD
        sweeps in the common case.  When a packed block fails we fall
        back to per-lane single-state checks for that block only, so
        the returned array is always exactly one bool per input.

        Args:
            configurations: ``(N, ndof)`` array of active-DOF
                configurations.

        Returns:
            ``(N,)`` boolean array; ``True`` at index ``i`` iff
            ``configurations[i]`` is collision-free.
        """
        configurations = np.asarray(configurations, dtype=np.float64)
        if configurations.ndim != 2 or configurations.shape[1] != self._ndof:
            raise ValueError(
                f"configurations must have shape (N, {self._ndof}), "
                f"got {configurations.shape}"
            )
        valid = self._planner.validate_batch(configurations.tolist())
        return np.asarray(valid, dtype=bool)

    def sample_valid(self) -> np.ndarray:
        """Sample a random collision-free configuration."""
        lo = np.array(self._planner.lower_bounds())
        hi = np.array(self._planner.upper_bounds())
        while True:
            config = np.random.uniform(lo, hi)
            if self._planner.validate(config.tolist()):
                return config


def available_robots() -> list[str]:
    """Return all available robot names for planning.

    Includes top-level robots (``"bimanual_fr3"``, ``"single_fr3"``)
    and every named subgroup each one exposes.
    """
    from bimanual_franka_planning._robots import (
        ROBOT_REGISTRY,
        _all_planner_names,
    )

    # Sort top-level robots first, then subgroups, for stability.
    top_level = sorted(ROBOT_REGISTRY.keys())
    subgroups = sorted(set(_all_planner_names()) - set(top_level))
    return top_level + subgroups


def create_planner(
    robot_name: str = "bimanual_fr3",
    config: PlannerConfig | None = None,
    pointcloud: np.ndarray | None = None,
    base_config: np.ndarray | None = None,
    constraints: list | None = None,
    costs: list | None = None,
) -> MotionPlanner:
    """Create a motion planner for any robot or subgroup.

    Args:
        robot_name: Robot or subgroup name. Use :func:`available_robots`
            to list all names.
        config: Planner configuration (uses defaults if None).
        pointcloud: ``(N, 3)`` obstacle point cloud (optional).
        base_config: 17-DOF values to inject for joints not controlled
            by this planner (i.e. the frozen joints of a subgroup).
            Defaults to ``HOME_JOINTS``.  Supply any 17-DOF array — for
            example the live configuration read from your env — to pin
            the rest of the body wherever it currently is.  Ignored for
            the full-body ``"bimanual_fr3"`` planner.
        constraints: Optional list of
            :class:`~bimanual_franka_planning.planning.constraints.Constraint`
            instances (CasADi-backed).  When non-empty, the planner
            switches to ``ProjectedStateSpace`` and projects every state
            onto the constraint manifold.  Both ``start`` and ``goal``
            passed to ``plan(...)`` must already lie on the manifold.
        costs: Optional list of
            :class:`~bimanual_franka_planning.planning.costs.Cost` instances
            (CasADi-backed).  Soft per-state terms summed with their
            weights and trapezoidally integrated along every motion —
            the asymptotically-optimal planners (``rrtstar``,
            ``bitstar``, ``aitstar``, …) minimise this objective.
            Without any costs the planner uses OMPL's default
            path-length objective.

    Returns:
        A :class:`MotionPlanner` instance.
    """
    return MotionPlanner(
        robot_name, config, pointcloud, base_config, constraints, costs
    )
