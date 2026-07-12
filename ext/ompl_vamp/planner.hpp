/**
 * Main planner class — OMPL frontend with VAMP collision backend.
 *
 * Templated on the VAMP ``Robot`` type so the same code drives every
 * registered description (``BimanualFr3``, ``SingleFr3``, …).  Each
 * concrete instantiation is bound separately into the Python module.
 *
 * Two construction modes:
 *
 *  - ``OmplVampPlanner()`` — full body, ``Robot::dimension`` DOF.
 *  - ``OmplVampPlanner(active_indices, frozen_config)`` — subgroup
 *    planning over the listed joint indices, with the rest of the
 *    body pinned to ``frozen_config`` for every collision check.
 *
 * The planner exposes a uniform Python-friendly API:
 * ``add_pointcloud`` / ``add_sphere`` / ``clear_environment`` build
 * the obstacle environment, ``plan(start, goal, ...)`` runs OMPL,
 * ``validate(...)``, ``dimension()``, ``lower_bounds()``,
 * ``upper_bounds()`` and ``min_max_radii()`` round out the surface.
 */

#pragma once

#include <ompl/base/ConstrainedSpaceInformation.h>
#include <ompl/base/Constraint.h>
#include <ompl/base/OptimizationObjective.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/spaces/constraint/ConstrainedStateSpace.h>
#include <ompl/base/spaces/constraint/ProjectedStateSpace.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/geometric/PathSimplifier.h>
#include <ompl/geometric/SimpleSetup.h>

#include "compiled_constraint.hpp"
#include "compiled_cost.hpp"
#include "validity.hpp"
// OMPL — informed trees
#include <ompl/geometric/planners/informedtrees/ABITstar.h>
#include <ompl/geometric/planners/informedtrees/AITstar.h>
#include <ompl/geometric/planners/informedtrees/BITstar.h>
#include <ompl/geometric/planners/informedtrees/EITstar.h>
#include <ompl/geometric/planners/lazyinformedtrees/BLITstar.h>
// OMPL — FMT
#include <ompl/geometric/planners/fmt/BFMT.h>
#include <ompl/geometric/planners/fmt/FMT.h>
// OMPL — KPIECE
#include <ompl/geometric/planners/kpiece/BKPIECE1.h>
#include <ompl/geometric/planners/kpiece/KPIECE1.h>
#include <ompl/geometric/planners/kpiece/LBKPIECE1.h>
// OMPL — PRM
#include <ompl/geometric/planners/prm/LazyPRM.h>
#include <ompl/geometric/planners/prm/LazyPRMstar.h>
#include <ompl/geometric/planners/prm/PRM.h>
#include <ompl/geometric/planners/prm/PRMstar.h>
#include <ompl/geometric/planners/prm/SPARS.h>
#include <ompl/geometric/planners/prm/SPARStwo.h>
// OMPL — RRT family
#include <ompl/geometric/planners/rrt/BiTRRT.h>
#include <ompl/geometric/planners/rrt/InformedRRTstar.h>
#include <ompl/geometric/planners/rrt/LBTRRT.h>
#include <ompl/geometric/planners/rrt/RRT.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/planners/rrt/RRTXstatic.h>
#include <ompl/geometric/planners/rrt/RRTsharp.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/geometric/planners/rrt/STRRTstar.h>
#include <ompl/geometric/planners/rrt/TRRT.h>
// OMPL — exploration-based
#include <ompl/geometric/planners/est/BiEST.h>
#include <ompl/geometric/planners/est/EST.h>
#include <ompl/geometric/planners/pdst/PDST.h>
#include <ompl/geometric/planners/sbl/SBL.h>
#include <ompl/geometric/planners/stride/STRIDE.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vamp/collision/attachments.hh>
#include <vamp/collision/filter.hh>
#include <vamp/collision/shapes.hh>
#include <vamp/collision/sphere_sphere.hh>
#include <vamp/collision/validity.hh>
#include <vector>

namespace bimanual_franka {

namespace og = ompl::geometric;

struct PlanResult {
  bool solved;
  std::vector<std::vector<double>> path;
  int64_t planning_time_ns;
  double path_cost;
};

template <class Robot>
class OmplVampPlanner {
  // Type aliases that strip the ``typename`` noise off every dependent
  // reference to ``Robot::``.  Used freely below.
  using Configuration = typename Robot::Configuration;
  using ConfigurationArray = typename Robot::ConfigurationArray;
  template <std::size_t Rake>
  using ConfigurationBlock = typename Robot::template ConfigurationBlock<Rake>;
  template <std::size_t Rake>
  using SpheresT = typename Robot::template Spheres<Rake>;

 public:
  /// Full-body constructor (``Robot::dimension`` DOF).
  OmplVampPlanner() : active_dim_(Robot::dimension), is_subgroup_(false) {
    Configuration lo, hi;
    // Use num_scalars_rounded so the FloatVector pointer-constructor
    // can read a SIMD-aligned full block; reading past dimension on a
    // size-dimension array is UB and produces NaNs in the padding.
    alignas(Configuration::S::Alignment)
        std::array<float, Configuration::num_scalars_rounded>
            zeros{}, ones{};
    std::fill(ones.begin(), ones.begin() + Robot::dimension, 1.0f);
    lo = Configuration(zeros.data());
    hi = Configuration(ones.data());
    Robot::scale_configuration(lo);
    Robot::scale_configuration(hi);

    auto lo_arr = lo.to_array();
    auto hi_arr = hi.to_array();

    auto space = std::make_shared<ob::RealVectorStateSpace>(Robot::dimension);
    ob::RealVectorBounds bounds(Robot::dimension);
    for (std::size_t i = 0; i < Robot::dimension; ++i) {
      bounds.setLow(i, std::min(lo_arr[i], hi_arr[i]));
      bounds.setHigh(i, std::max(lo_arr[i], hi_arr[i]));
    }
    space->setBounds(bounds);
    space_ = space;
  }

  /// Subgroup constructor (reduced DOF).
  OmplVampPlanner(std::vector<int> active_indices,
                  std::vector<double> frozen_config)
      : active_dim_(active_indices.size()),
        is_subgroup_(true),
        active_indices_(std::move(active_indices)) {
    frozen_config_.resize(frozen_config.size());
    for (std::size_t i = 0; i < frozen_config.size(); ++i)
      frozen_config_[i] = static_cast<float>(frozen_config[i]);

    Configuration lo, hi;
    // Use num_scalars_rounded so the FloatVector pointer-constructor
    // can read a SIMD-aligned full block; reading past dimension on a
    // size-dimension array is UB and produces NaNs in the padding.
    alignas(Configuration::S::Alignment)
        std::array<float, Configuration::num_scalars_rounded>
            zeros{}, ones{};
    std::fill(ones.begin(), ones.begin() + Robot::dimension, 1.0f);
    lo = Configuration(zeros.data());
    hi = Configuration(ones.data());
    Robot::scale_configuration(lo);
    Robot::scale_configuration(hi);
    auto lo_arr = lo.to_array();
    auto hi_arr = hi.to_array();

    auto space = std::make_shared<ob::RealVectorStateSpace>(active_dim_);
    ob::RealVectorBounds bounds(active_dim_);
    for (std::size_t i = 0; i < active_indices_.size(); ++i) {
      auto idx = active_indices_[i];
      bounds.setLow(i, std::min(lo_arr[idx], hi_arr[idx]));
      bounds.setHigh(i, std::max(lo_arr[idx], hi_arr[idx]));
    }
    space->setBounds(bounds);
    space_ = space;
  }

  /// Set the scene pointcloud.  The planner holds at most one cloud;
  /// calling this replaces any previously-registered cloud.
  void add_pointcloud(const std::vector<std::array<float, 3>> &points,
                      float r_min, float r_max, float point_radius) {
    std::vector<vamp::collision::Point> pts;
    pts.reserve(points.size());
    for (const auto &p : points) pts.push_back({p[0], p[1], p[2]});
    float_env_.pointclouds.clear();
    float_env_.pointclouds.emplace_back(pts, r_min, r_max, point_radius);
    sync_env();
  }

  /// Drop the currently-registered pointcloud.  Returns ``false`` if
  /// there was no cloud to remove.
  bool remove_pointcloud() {
    if (float_env_.pointclouds.empty()) return false;
    float_env_.pointclouds.clear();
    sync_env();
    return true;
  }

  bool has_pointcloud() const { return !float_env_.pointclouds.empty(); }

  void add_sphere(const std::array<float, 3> &center, float radius) {
    float_env_.spheres.emplace_back(vamp::collision::Sphere<float>{
        center[0], center[1], center[2], radius});
    float_env_.sort();
    sync_env();
  }

  void clear_environment() {
    float_env_ = FloatEnv{};
    env_ = VampEnv{};
  }

  // ── End-effector attachment API ───────────────────────────────────
  //
  // Attach a set of spheres to one of the robot's end-effectors so they
  // move with the gripper and are collision-checked at every state and
  // motion edge.  Each planner instance holds at most one attachment —
  // calling this method replaces any existing attachment.
  //
  // The ``relative_tf`` is a 4x4 row-major isometry expressed in the EE
  // link frame; the supplied spheres are interpreted in EE frame (after
  // applying ``relative_tf``).  Sphere format: x, y, z, radius.
  void attach_ee_spheres(
      std::size_t ee_index, const std::array<float, 16> &relative_tf_row_major,
      const std::vector<std::array<float, 4>> &spheres_xyzr) {
    if (ee_index >= Robot::n_end_effectors) {
      throw std::invalid_argument(
          "attach_ee_spheres: ee_index " + std::to_string(ee_index) +
          " is out of range for robot '" + std::string(Robot::name) +
          "' (n_end_effectors = " + std::to_string(Robot::n_end_effectors) +
          ").");
    }
    Eigen::Matrix4f m;
    for (std::size_t r = 0; r < 4; ++r)
      for (std::size_t c = 0; c < 4; ++c)
        m(r, c) = relative_tf_row_major[r * 4 + c];
    Eigen::Transform<float, 3, Eigen::Isometry> tf;
    tf.matrix() = m;

    vamp::collision::Attachment<float> att(tf);
    att.ee_index = ee_index;
    att.spheres.reserve(spheres_xyzr.size());
    for (const auto &s : spheres_xyzr) {
      att.spheres.emplace_back(
          vamp::collision::Sphere<float>{s[0], s[1], s[2], s[3]});
    }
    float_env_.attachments = std::move(att);
    sync_env();
  }

  bool detach_ee() {
    if (!float_env_.attachments.has_value()) return false;
    float_env_.attachments.reset();
    sync_env();
    return true;
  }

  bool has_attachment() const { return float_env_.attachments.has_value(); }

  std::size_t num_end_effectors() const { return Robot::n_end_effectors; }

  // ── Constraint API ────────────────────────────────────────────────
  //
  // Constraints are accumulated by repeated add_*() calls and consumed
  // by the next plan() call.  Use clear_constraints() to reset.

  void add_compiled_constraint(const std::string &so_path,
                               const std::string &symbol_name,
                               unsigned int ambient_dim, unsigned int co_dim,
                               unsigned int param_dim = 0,
                               const std::vector<double> &params = {}) {
    if (static_cast<int>(ambient_dim) != active_dim_) {
      throw std::invalid_argument(
          "add_compiled_constraint: ambient_dim (" +
          std::to_string(ambient_dim) +
          ") does not match planner active dimension (" +
          std::to_string(active_dim_) + ")");
    }
    constraints_.push_back(std::make_shared<CompiledConstraint>(
        ambient_dim, co_dim, so_path, symbol_name, param_dim, params));
  }

  void clear_constraints() { constraints_.clear(); }

  std::size_t num_constraints() const { return constraints_.size(); }

  // ── Cost API ──────────────────────────────────────────────────────
  //
  // Costs are soft per-state terms integrated along every motion by
  // OMPL's StateCostIntegralObjective.  They do not constrain the
  // feasible set — collision checking still does that — but they
  // shape the solution returned by asymptotically-optimal planners
  // such as RRT*, BIT*, AIT*, …  Without a user-supplied cost the
  // planner falls back to OMPL's default path-length objective.
  //
  // Multiple costs are summed via MultiOptimizationObjective with
  // the weights supplied at add time.

  void add_compiled_cost(const std::string &so_path,
                         const std::string &symbol_name,
                         unsigned int ambient_dim, double weight) {
    if (static_cast<int>(ambient_dim) != active_dim_) {
      throw std::invalid_argument(
          "add_compiled_cost: ambient_dim (" + std::to_string(ambient_dim) +
          ") does not match planner active dimension (" +
          std::to_string(active_dim_) + ")");
    }
    if (weight < 0.0) {
      throw std::invalid_argument("add_compiled_cost: weight must be >= 0");
    }
    cost_libs_.push_back(
        std::make_shared<CostLibrary>(ambient_dim, so_path, symbol_name));
    cost_weights_.push_back(weight);
  }

  void clear_costs() {
    cost_libs_.clear();
    cost_weights_.clear();
  }

  std::size_t num_costs() const { return cost_libs_.size(); }

  auto plan(std::vector<double> start, std::vector<double> goal,
            const std::string &planner_name, double time_limit, bool simplify,
            bool interpolate, int interpolate_count, double resolution)
      -> PlanResult {
    if (interpolate_count > 0 && resolution > 0.0) {
      throw std::invalid_argument(
          "plan: pass at most one of interpolate_count (>0) or resolution "
          "(>0), not both.");
    }
    if (resolution < 0.0) {
      throw std::invalid_argument(
          "plan: resolution must be >= 0 (0 disables).");
    }
    const bool constrained = !constraints_.empty();
    if (constrained) {
      reject_incompatible_planner(planner_name);
      // Both endpoints must already lie on the constraint manifold —
      // we don't run a manifold IK on them.  This catches the common
      // foot-gun where the user computes a target pose from FK on a
      // different configuration than the one they're starting from.
      check_constraint_satisfaction(start, "start");
      check_constraint_satisfaction(goal, "goal");
    }

    // Pick the right state space + space information for this plan.
    auto [si, active_space] = make_space_information_(constrained);

    og::SimpleSetup ss(si);
    ss.setPlanner(create_planner(si, planner_name));

    // Wire soft costs (if any) as the optimisation objective.  RRT*,
    // BIT*, AIT* and friends will drive their rewiring against this
    // objective; for planners that don't consume it the setting is
    // harmless.
    if (!cost_libs_.empty()) {
      ss.setOptimizationObjective(build_objective(si));
    }

    ob::ScopedState<> ompl_start(active_space);
    ob::ScopedState<> ompl_goal(active_space);
    for (int i = 0; i < active_dim_; ++i) {
      ompl_start[i] = start[i];
      ompl_goal[i] = goal[i];
    }
    ss.setStartAndGoalStates(ompl_start, ompl_goal);

    auto t0 = std::chrono::steady_clock::now();
    auto status = ss.solve(time_limit);
    auto t1 = std::chrono::steady_clock::now();
    auto elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    PlanResult result;
    result.planning_time_ns = elapsed_ns;
    result.solved = static_cast<bool>(status);

    if (result.solved) {
      if (simplify) ss.simplifySolution();

      auto &path = ss.getSolutionPath();
      // Interpolate after simplify so the returned path has enough
      // waypoints to animate smoothly.  Three modes:
      //   - interpolate_count > 0 : fixed total waypoint count,
      //       distributed by OMPL across edges by relative length.
      //   - resolution > 0        : waypoints per unit state-space
      //       distance — each edge of length d is split into
      //       ceil(d * resolution) equal segments.
      //   - both 0                : OMPL default (longest valid
      //       segment fraction of the state space).
      if (interpolate) {
        if (interpolate_count > 0) {
          path.interpolate(static_cast<unsigned int>(interpolate_count));
        } else if (resolution > 0.0) {
          densify_by_resolution(path, active_space, resolution);
        } else {
          path.interpolate();
        }
      }
      result.path_cost = path.length();

      for (std::size_t i = 0; i < path.getStateCount(); ++i) {
        const auto *rv = extract_real_state(path.getState(i));
        std::vector<double> config(active_dim_);
        for (int j = 0; j < active_dim_; ++j) config[j] = rv->values[j];
        result.path.push_back(std::move(config));
      }
    } else {
      result.path_cost = std::numeric_limits<double>::infinity();
    }

    return result;
  }

  // Standalone path simplification — same pipeline OMPL's
  // ``SimpleSetup::simplifySolution`` runs (reduceVertices,
  // collapseCloseVertices, shortcutPath, optional B-spline smoothing)
  // against the current collision environment, but detached from
  // ``plan(...)``.  The input and output are plain waypoint lists in
  // the planner's active DOF space.
  //
  // Shortcuts only consult the motion validator — custom soft costs
  // are ignored.  For constrained/cost planners, prefer
  // ``plan(simplify=False)`` and leave the path untouched, or call
  // this only when you've explicitly decided shortcut shaping is
  // acceptable.
  auto simplify_path(const std::vector<std::vector<double>> &path,
                     double time_limit) -> std::vector<std::vector<double>> {
    if (path.size() < 2) return path;
    const bool constrained = !constraints_.empty();
    auto [si, active_space] = make_space_information_(constrained);
    og::PathGeometric geo = waypoints_to_path_(path, si, active_space);
    og::PathSimplifier simp(si);
    simp.simplify(geo, time_limit);
    return path_to_waypoints_(geo);
  }

  // Standalone path densification — same three modes as
  // ``plan(..., interpolate=True, ...)``:
  //   * ``count > 0``        → exactly that many total waypoints.
  //   * ``resolution > 0.0`` → ``ceil(edge_length * resolution)`` waypoints
  //                            per edge (uniform density in state-space
  //                            distance).
  //   * both 0               → OMPL's default longest-valid-segment
  //                            fraction.
  // No collision check — densification only calls
  // ``StateSpace::interpolate`` on the existing edges, so the resulting
  // waypoints lie on the same (piecewise-linear) path the caller
  // already had and stay on the constraint manifold for projected
  // spaces.
  auto interpolate_path(const std::vector<std::vector<double>> &path, int count,
                        double resolution) -> std::vector<std::vector<double>> {
    if (count > 0 && resolution > 0.0) {
      throw std::invalid_argument(
          "interpolate_path: pass at most one of count (>0) or resolution "
          "(>0), not both.");
    }
    if (resolution < 0.0) {
      throw std::invalid_argument(
          "interpolate_path: resolution must be >= 0 (0 disables).");
    }
    if (path.size() < 2) return path;
    const bool constrained = !constraints_.empty();
    auto [si, active_space] = make_space_information_(constrained);
    og::PathGeometric geo = waypoints_to_path_(path, si, active_space);
    if (count > 0) {
      geo.interpolate(static_cast<unsigned int>(count));
    } else if (resolution > 0.0) {
      densify_by_resolution(geo, active_space, resolution);
    } else {
      geo.interpolate();
    }
    return path_to_waypoints_(geo);
  }

  auto validate(std::vector<double> config) -> bool {
    if (static_cast<int>(config.size()) != active_dim_) {
      throw std::invalid_argument(std::string("validate: config length ") +
                                  std::to_string(config.size()) +
                                  " does not match active DOF " +
                                  std::to_string(active_dim_) + ".");
    }
    auto q = build_full_config_(config);
    return vamp::planning::validate_motion<Robot, kRake, 1>(q, q, env_);
  }

  // Batched collision check — packs up to ``kRake`` distinct
  // configurations directly into a VAMP ``ConfigurationBlock<kRake>``
  // so a single ``Robot::fkcc<kRake>`` call sphere-FKs and
  // collision-checks them against the SIMD environment in one sweep.
  // Same SIMD primitive the motion-edge validator uses for interpolated
  // samples — we just feed it independent configs per lane.
  //
  // If a packed block fails, we fall back to per-lane single-state
  // checks so the caller still gets one bool per input config.  In the
  // common case (most configs valid) this is O(N/kRake) SIMD calls;
  // worst case (every block fails) degrades to the baseline N single
  // checks.
  //
  // Subgroup planners expand each reduced-DOF config to the full 24-DOF
  // body via the stored frozen pose before packing, mirroring
  // ``validate(...)``.
  auto validate_batch(const std::vector<std::vector<double>> &configs)
      -> std::vector<bool> {
    const std::size_t n = configs.size();
    std::vector<bool> result(n, false);
    if (n == 0) return result;

    for (std::size_t i = 0; i < n; ++i) {
      if (static_cast<int>(configs[i].size()) != active_dim_) {
        throw std::invalid_argument(
            std::string("validate_batch: config[") + std::to_string(i) +
            "] length " + std::to_string(configs[i].size()) +
            " does not match active DOF " + std::to_string(active_dim_) + ".");
      }
    }

    // ``ConfigurationBlock<kRake>::pack`` expects row-major layout
    // where row ``d`` (joint dimension) holds kRake scalars — the
    // lane values for that joint across the rake.  So
    //     buf[d * kRake + lane] = full_config(configs[lane])[d].
    alignas(vamp::FloatVectorAlignment)
        std::array<float, Robot::dimension * kRake>
            blk_buf{};

    auto write_lane = [&](std::size_t lane, const std::vector<double> &cfg) {
      if (is_subgroup_) {
        for (std::size_t d = 0; d < Robot::dimension; ++d)
          blk_buf[d * kRake + lane] = frozen_config_[d];
        for (std::size_t k = 0; k < active_indices_.size(); ++k)
          blk_buf[active_indices_[k] * kRake + lane] =
              static_cast<float>(cfg[k]);
      } else {
        for (std::size_t d = 0; d < Robot::dimension; ++d)
          blk_buf[d * kRake + lane] = static_cast<float>(cfg[d]);
      }
    };

    for (std::size_t i = 0; i < n; i += kRake) {
      const std::size_t chunk = std::min(kRake, n - i);
      for (std::size_t lane = 0; lane < chunk; ++lane)
        write_lane(lane, configs[i + lane]);
      // Pad the tail lanes by repeating the first real config in this
      // block.  Padding with a real (tested) config preserves
      // correctness: if the packed block passes, every real lane is
      // valid; if it fails we only run the per-lane fallback over the
      // real lanes, so the padded lanes never appear in the output.
      for (std::size_t lane = chunk; lane < kRake; ++lane)
        write_lane(lane, configs[i]);

      // Constructor forwards to VectorInterface::pack(), which is
      // itself protected — this is the public entry-point.  The
      // buffer is aligned to FloatVectorAlignment and its length
      // (Robot::dimension * kRake) is a multiple of VectorWidth, so
      // the aligned load path is safe.
      typename Robot::template ConfigurationBlock<kRake> block(blk_buf.data());

      const bool block_valid =
          env_.attachments ? Robot::template fkcc_attach<kRake>(env_, block)
                           : Robot::template fkcc<kRake>(env_, block);

      if (block_valid) {
        for (std::size_t lane = 0; lane < chunk; ++lane)
          result[i + lane] = true;
      } else {
        for (std::size_t lane = 0; lane < chunk; ++lane) {
          auto q = build_full_config_(configs[i + lane]);
          result[i + lane] =
              vamp::planning::validate_motion<Robot, kRake, 1>(q, q, env_);
        }
      }
    }

    return result;
  }

  // ── Point cloud filtering ───────────────────────────────────────

  /// Spatial down-sampling via Morton-curve sorting.
  auto filter_pointcloud(const std::vector<std::array<float, 3>> &points,
                         float min_dist, float max_range,
                         const std::array<float, 3> &origin,
                         const std::array<float, 3> &workspace_min,
                         const std::array<float, 3> &workspace_max, bool cull)
      -> std::vector<std::array<float, 3>> {
    vamp::collision::Point o{origin[0], origin[1], origin[2]};
    vamp::collision::Point ws_min{workspace_min[0], workspace_min[1],
                                  workspace_min[2]};
    vamp::collision::Point ws_max{workspace_max[0], workspace_max[1],
                                  workspace_max[2]};

    std::vector<vamp::collision::Point> pc;
    pc.reserve(points.size());
    for (const auto &p : points) pc.push_back({p[0], p[1], p[2]});

    auto filtered = vamp::collision::filter_pointcloud(pc, min_dist, max_range,
                                                       o, ws_min, ws_max, cull);

    std::vector<std::array<float, 3>> out;
    out.reserve(filtered.size());
    for (const auto &p : filtered) out.push_back({p[0], p[1], p[2]});
    return out;
  }

  /// Remove points that collide with the robot body or the environment.
  auto filter_self_from_pointcloud(
      const std::vector<std::array<float, 3>> &points, float point_radius,
      const std::vector<double> &config) -> std::vector<std::array<float, 3>> {
    if (static_cast<int>(config.size()) != active_dim_) {
      throw std::invalid_argument(
          std::string("filter_self_from_pointcloud: config length ") +
          std::to_string(config.size()) + " does not match active DOF " +
          std::to_string(active_dim_) + ".");
    }

    // FK at the given configuration
    auto full_arr = build_full_config_(config).to_array();
    typename Robot::template ConfigurationBlock<1> block;
    for (std::size_t i = 0; i < Robot::dimension; ++i) block[i] = full_arr[i];

    typename Robot::template Spheres<1> spheres;
    Robot::template sphere_fk<1>(block, spheres);

    std::vector<std::array<float, 3>> out;
    out.reserve(points.size());

    for (const auto &pt : points) {
      const float x = pt[0], y = pt[1], z = pt[2], r = point_radius;
      bool valid = true;
      for (std::size_t i = 0; i < Robot::n_spheres; ++i) {
        if (vamp::collision::sphere_sphere_sql2(
                spheres.x[{i, 0}], spheres.y[{i, 0}], spheres.z[{i, 0}],
                spheres.r[{i, 0}], x, y, z, r) < 0 ||
            vamp::sphere_environment_in_collision(env_, x, y, z, r)) {
          valid = false;
          break;
        }
      }
      if (valid) out.push_back(pt);
    }
    return out;
  }

  auto dimension() const -> int { return active_dim_; }

  auto lower_bounds() const -> std::vector<double> {
    auto bounds = space_->as<ob::RealVectorStateSpace>()->getBounds();
    std::vector<double> lo(active_dim_);
    for (int i = 0; i < active_dim_; ++i) lo[i] = bounds.low[i];
    return lo;
  }

  auto upper_bounds() const -> std::vector<double> {
    auto bounds = space_->as<ob::RealVectorStateSpace>()->getBounds();
    std::vector<double> hi(active_dim_);
    for (int i = 0; i < active_dim_; ++i) hi[i] = bounds.high[i];
    return hi;
  }

  auto min_max_radii() const -> std::pair<float, float> {
    return {Robot::min_radius, Robot::max_radius};
  }

  /// Switch to a different subgroup without rebuilding the environment.
  void set_subgroup(std::vector<int> active_indices,
                    std::vector<double> frozen_config) {
    active_dim_ = static_cast<int>(active_indices.size());
    is_subgroup_ = true;
    active_indices_ = std::move(active_indices);
    frozen_config_.resize(frozen_config.size());
    for (std::size_t i = 0; i < frozen_config.size(); ++i)
      frozen_config_[i] = static_cast<float>(frozen_config[i]);
    constraints_.clear();
    clear_costs();
    rebuild_space_();
  }

  /// Switch back to full-body mode without rebuilding the environment.
  void set_full_body() {
    active_dim_ = Robot::dimension;
    is_subgroup_ = false;
    active_indices_.clear();
    frozen_config_.clear();
    constraints_.clear();
    clear_costs();
    rebuild_space_();
  }

  /// Override per-joint position bounds for the planner's state space.
  ///
  /// ``lower`` and ``upper`` must each have length ``Robot::dimension``
  /// — full-DOF arrays, independent of any active subgroup.  In
  /// subgroup mode the active subset is selected automatically by
  /// ``rebuild_space_()``.  The custom limits persist across
  /// ``set_subgroup()`` / ``set_full_body()`` calls; use
  /// ``clear_joint_limits()`` to revert to the robot's compile-time
  /// defaults derived from ``Robot::scale_configuration``.
  ///
  /// Typical use: pass the *real controller's* joint limits so any
  /// path the planner returns lies inside what the controller will
  /// actually execute.  The compile-time defaults track the URDF
  /// (often the manufacturer's hardware envelope), which is wider
  /// than what a deployed controller will accept.
  void set_joint_limits(std::vector<double> lower, std::vector<double> upper) {
    if (lower.size() != Robot::dimension || upper.size() != Robot::dimension) {
      throw std::invalid_argument(
          std::string("set_joint_limits: lower/upper must each have length ") +
          std::to_string(Robot::dimension) +
          ", got lower=" + std::to_string(lower.size()) +
          " upper=" + std::to_string(upper.size()) + ".");
    }
    for (std::size_t i = 0; i < Robot::dimension; ++i) {
      if (!(lower[i] <= upper[i])) {
        throw std::invalid_argument(
            std::string("set_joint_limits: lower[") + std::to_string(i) +
            "]=" + std::to_string(lower[i]) + " > upper[" + std::to_string(i) +
            "]=" + std::to_string(upper[i]) + ".");
      }
    }
    custom_lower_ = std::move(lower);
    custom_upper_ = std::move(upper);
    rebuild_space_();
  }

  /// Clear any custom limits set by :meth:`set_joint_limits`,
  /// reverting to the robot's compile-time bounds derived from
  /// ``Robot::scale_configuration``.
  void clear_joint_limits() {
    custom_lower_.clear();
    custom_upper_.clear();
    rebuild_space_();
  }

 private:
  int active_dim_;
  bool is_subgroup_;
  std::vector<int> active_indices_;
  std::vector<float> frozen_config_;
  ob::StateSpacePtr space_;
  FloatEnv float_env_;
  VampEnv env_;
  std::vector<ob::ConstraintPtr> constraints_;
  std::vector<std::shared_ptr<CostLibrary>> cost_libs_;
  std::vector<double> cost_weights_;
  // Custom per-joint position bounds (full-DOF, both empty unless
  // ``set_joint_limits`` has been called).  When populated, override
  // the ``Robot::scale_configuration`` defaults in ``rebuild_space_``.
  std::vector<double> custom_lower_;
  std::vector<double> custom_upper_;

  void sync_env() { env_ = VampEnv(float_env_); }

  // Expand an active-DOF config into a full 24-DOF VAMP Configuration,
  // injecting the frozen pose for joints outside ``active_indices_``
  // when running as a subgroup planner.
  auto build_full_config_(const std::vector<double> &config) const
      -> Configuration {
    alignas(Configuration::S::Alignment)
        std::array<float, Configuration::num_scalars_rounded>
            buf;
    // Zero the SIMD padding explicitly — see comment in
    // validity.hpp::ompl_to_vamp.
    std::fill(buf.begin(), buf.end(), 0.0f);
    if (is_subgroup_) {
      std::copy(frozen_config_.begin(), frozen_config_.end(), buf.begin());
      for (std::size_t i = 0; i < active_indices_.size(); ++i)
        buf[active_indices_[i]] = static_cast<float>(config[i]);
    } else {
      for (std::size_t i = 0; i < Robot::dimension; ++i)
        buf[i] = static_cast<float>(config[i]);
    }
    return Configuration(buf.data());
  }

  // Build the OMPL OptimizationObjective for the active cost set.
  // One CostLibrary backs each CompiledCost adapter — the adapter
  // is re-created per plan() call so it binds to the correct
  // (flat or constrained) SpaceInformation, but the dlopen'd
  // library is reused.
  std::shared_ptr<ob::OptimizationObjective> build_objective(
      const ob::SpaceInformationPtr &si) const {
    if (cost_libs_.size() == 1) {
      return std::make_shared<CompiledCost>(si, cost_libs_[0],
                                            cost_weights_[0]);
    }
    auto multi = std::make_shared<ob::MultiOptimizationObjective>(si);
    for (std::size_t i = 0; i < cost_libs_.size(); ++i) {
      // MultiOptimizationObjective applies its own weight on top of
      // whatever OptimizationObjective::stateCost() returns, so we
      // bake the per-cost weight into the adapter (weight 1.0 here)
      // to keep a single source of truth for the scaling factor.
      multi->addObjective(
          std::make_shared<CompiledCost>(si, cost_libs_[i], cost_weights_[i]),
          1.0);
    }
    return multi;
  }

  void rebuild_space_() {
    Configuration lo, hi;
    // Use num_scalars_rounded so the FloatVector pointer-constructor
    // can read a SIMD-aligned full block; reading past dimension on a
    // size-dimension array is UB and produces NaNs in the padding.
    alignas(Configuration::S::Alignment)
        std::array<float, Configuration::num_scalars_rounded>
            zeros{}, ones{};
    std::fill(ones.begin(), ones.begin() + Robot::dimension, 1.0f);
    lo = Configuration(zeros.data());
    hi = Configuration(ones.data());
    Robot::scale_configuration(lo);
    Robot::scale_configuration(hi);
    auto lo_arr = lo.to_array();
    auto hi_arr = hi.to_array();

    // When custom limits are set, override the scale_configuration-
    // derived defaults.  Both vectors are either empty (defaults) or
    // length Robot::dimension (validated by set_joint_limits).
    const bool has_custom = !custom_lower_.empty() && !custom_upper_.empty();

    auto bound_for_dim = [&](std::size_t full_i) -> std::pair<double, double> {
      if (has_custom) {
        return {custom_lower_[full_i], custom_upper_[full_i]};
      }
      return {std::min<double>(lo_arr[full_i], hi_arr[full_i]),
              std::max<double>(lo_arr[full_i], hi_arr[full_i])};
    };

    auto space = std::make_shared<ob::RealVectorStateSpace>(active_dim_);
    ob::RealVectorBounds bounds(active_dim_);
    if (is_subgroup_) {
      for (std::size_t i = 0; i < active_indices_.size(); ++i) {
        auto [lo_v, hi_v] = bound_for_dim(active_indices_[i]);
        bounds.setLow(i, lo_v);
        bounds.setHigh(i, hi_v);
      }
    } else {
      for (int i = 0; i < active_dim_; ++i) {
        auto [lo_v, hi_v] = bound_for_dim(static_cast<std::size_t>(i));
        bounds.setLow(i, lo_v);
        bounds.setHigh(i, hi_v);
      }
    }
    space->setBounds(bounds);
    space_ = space;
  }

  // Check that *active_q* satisfies every constraint within the
  // OMPL constraint tolerance.  Throws std::invalid_argument with a
  // descriptive message naming which constraint and how badly it was
  // violated, so the user gets a clear error rather than a hang.
  void check_constraint_satisfaction(const std::vector<double> &active_q,
                                     const char *which) const {
    Eigen::VectorXd q(active_dim_);
    for (int i = 0; i < active_dim_; ++i) q[i] = active_q[i];
    for (std::size_t i = 0; i < constraints_.size(); ++i) {
      const auto &c = constraints_[i];
      Eigen::VectorXd r(c->getCoDimension());
      c->function(q, r);
      const double residual = r.norm();
      if (residual > c->getTolerance()) {
        throw std::invalid_argument(
            std::string("Constraint #") + std::to_string(i) +
            " is violated at " + which + " (residual " +
            std::to_string(residual) + " > tolerance " +
            std::to_string(c->getTolerance()) +
            ").  Both start and goal must already lie on the constraint "
            "manifold — compute target poses from FK on the start config "
            "you intend to plan from.");
      }
    }
  }

  // Build a ``SpaceInformation`` configured with our validity /
  // motion checkers — extracted from ``plan(...)`` so the standalone
  // simplify/interpolate paths share exactly the same configuration.
  auto make_space_information_(bool constrained)
      -> std::pair<ob::SpaceInformationPtr, ob::StateSpacePtr> {
    ob::StateSpacePtr active_space = space_;
    ob::SpaceInformationPtr si;
    if (constrained) {
      auto intersection = std::make_shared<ob::ConstraintIntersection>(
          static_cast<unsigned int>(active_dim_), constraints_);
      auto css =
          std::make_shared<ob::ProjectedStateSpace>(space_, intersection);
      auto csi = std::make_shared<ob::ConstrainedSpaceInformation>(css);
      css->setup();
      si = csi;
      active_space = css;
    } else {
      si = std::make_shared<ob::SpaceInformation>(space_);
    }

    if (is_subgroup_) {
      si->setStateValidityChecker(
          std::make_shared<SubgroupValidityChecker<Robot>>(
              si, env_, active_indices_, frozen_config_));
      if (!constrained) {
        si->setMotionValidator(std::make_shared<SubgroupMotionValidator<Robot>>(
            si, env_, active_indices_, frozen_config_));
      }
    } else {
      si->setStateValidityChecker(
          std::make_shared<FullBodyValidityChecker<Robot>>(si, env_));
      if (!constrained) {
        si->setMotionValidator(
            std::make_shared<FullBodyMotionValidator<Robot>>(si, env_));
      }
    }
    si->setup();
    return {si, active_space};
  }

  // Lift a flat waypoint list into an OMPL ``PathGeometric``.  States
  // are allocated from ``active_space`` so the caller can hand it to
  // ``PathSimplifier``, ``PathGeometric::interpolate``, or
  // ``densify_by_resolution`` interchangeably.
  auto waypoints_to_path_(const std::vector<std::vector<double>> &waypoints,
                          const ob::SpaceInformationPtr &si,
                          const ob::StateSpacePtr &active_space)
      -> og::PathGeometric {
    og::PathGeometric path(si);
    for (const auto &w : waypoints) {
      if (static_cast<int>(w.size()) != active_dim_) {
        throw std::invalid_argument(
            std::string("Waypoint dimension ") + std::to_string(w.size()) +
            " does not match active DOF " + std::to_string(active_dim_) + ".");
      }
      auto *s = active_space->allocState();
      auto *rv = extract_real_state(s);
      for (int j = 0; j < active_dim_; ++j) rv->values[j] = w[j];
      path.append(s);
      active_space->freeState(s);
    }
    return path;
  }

  // Flatten a PathGeometric's states back into the waypoint list the
  // Python side expects.
  auto path_to_waypoints_(const og::PathGeometric &path)
      -> std::vector<std::vector<double>> {
    std::vector<std::vector<double>> out;
    out.reserve(path.getStateCount());
    for (std::size_t i = 0; i < path.getStateCount(); ++i) {
      const auto *rv = extract_real_state(path.getState(i));
      std::vector<double> config(active_dim_);
      for (int j = 0; j < active_dim_; ++j) config[j] = rv->values[j];
      out.push_back(std::move(config));
    }
    return out;
  }

  // Densify ``path`` so each edge of state-space length ``d`` is
  // split into ``ceil(d * resolution)`` equal segments.  ``resolution``
  // is waypoints per unit of state-space distance: higher values give
  // denser paths.  Uses ``StateSpace::interpolate`` so the inserted
  // states remain valid under projected/constrained spaces as well
  // as flat ones.
  static void densify_by_resolution(og::PathGeometric &path,
                                    const ob::StateSpacePtr &stsp,
                                    double resolution) {
    auto &states = path.getStates();
    if (states.size() < 2) return;
    std::vector<ob::State *> snap;
    snap.reserve(states.size());
    for (auto *s : states) {
      auto *c = stsp->allocState();
      stsp->copyState(c, s);
      snap.push_back(c);
    }
    for (auto *s : states) stsp->freeState(s);
    states.clear();
    states.push_back(snap.front());
    for (std::size_t i = 1; i < snap.size(); ++i) {
      double d = stsp->distance(snap[i - 1], snap[i]);
      int n = std::max(1, static_cast<int>(std::ceil(d * resolution)));
      for (int k = 1; k < n; ++k) {
        auto *tmp = stsp->allocState();
        stsp->interpolate(snap[i - 1], snap[i], static_cast<double>(k) / n,
                          tmp);
        states.push_back(tmp);
      }
      states.push_back(snap[i]);
    }
  }

  // ProjectedStateSpace only supports single-tree planners — batch
  // / informed-tree variants don't go through manifold projection.
  static void reject_incompatible_planner(const std::string &name) {
    static const std::vector<std::string> bad = {
        "bitstar",  "abitstar",   "aitstar",  "eitstar",
        "blitstar", "fmt",        "bfmt",     "informed_rrtstar",
        "rrtsharp", "rrtxstatic", "strrtstar"};
    for (const auto &b : bad) {
      if (name == b) {
        throw std::invalid_argument(
            "Planner '" + name +
            "' is incompatible with constrained planning.  Use one of: "
            "rrtc, rrt, rrtstar, prm, prmstar, kpiece, bkpiece, lbkpiece, "
            "est, biest, sbl, stride.");
      }
    }
  }

  static auto create_planner(const ob::SpaceInformationPtr &si,
                             const std::string &name) -> ob::PlannerPtr {
    // RRT family
    if (name == "rrtc" || name == "rrtconnect")
      return std::make_shared<og::RRTConnect>(si);
    if (name == "rrt") return std::make_shared<og::RRT>(si);
    if (name == "rrtstar") return std::make_shared<og::RRTstar>(si);
    if (name == "informed_rrtstar")
      return std::make_shared<og::InformedRRTstar>(si);
    if (name == "rrtsharp") return std::make_shared<og::RRTsharp>(si);
    if (name == "rrtxstatic") return std::make_shared<og::RRTXstatic>(si);
    if (name == "strrtstar") return std::make_shared<og::STRRTstar>(si);
    if (name == "lbtrrt") return std::make_shared<og::LBTRRT>(si);
    if (name == "trrt") return std::make_shared<og::TRRT>(si);
    if (name == "bitrrt") return std::make_shared<og::BiTRRT>(si);
    // Informed trees (asymptotically optimal)
    if (name == "bitstar") return std::make_shared<og::BITstar>(si);
    if (name == "abitstar") return std::make_shared<og::ABITstar>(si);
    if (name == "aitstar") return std::make_shared<og::AITstar>(si);
    if (name == "eitstar") return std::make_shared<og::EITstar>(si);
    if (name == "blitstar") return std::make_shared<og::BLITstar>(si);
    // FMT
    if (name == "fmt") return std::make_shared<og::FMT>(si);
    if (name == "bfmt") return std::make_shared<og::BFMT>(si);
    // KPIECE
    if (name == "kpiece") return std::make_shared<og::KPIECE1>(si);
    if (name == "bkpiece") return std::make_shared<og::BKPIECE1>(si);
    if (name == "lbkpiece") return std::make_shared<og::LBKPIECE1>(si);
    // PRM family
    if (name == "prm") return std::make_shared<og::PRM>(si);
    if (name == "prmstar") return std::make_shared<og::PRMstar>(si);
    if (name == "lazyprm") return std::make_shared<og::LazyPRM>(si);
    if (name == "lazyprmstar") return std::make_shared<og::LazyPRMstar>(si);
    if (name == "spars") return std::make_shared<og::SPARS>(si);
    if (name == "spars2") return std::make_shared<og::SPARStwo>(si);
    // Exploration-based
    if (name == "est") return std::make_shared<og::EST>(si);
    if (name == "biest") return std::make_shared<og::BiEST>(si);
    if (name == "sbl") return std::make_shared<og::SBL>(si);
    if (name == "stride") return std::make_shared<og::STRIDE>(si);
    if (name == "pdst") return std::make_shared<og::PDST>(si);
    throw std::invalid_argument("Unknown planner: " + name);
  }
};

}  // namespace bimanual_franka
