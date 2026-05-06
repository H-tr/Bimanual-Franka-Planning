/**
 * OMPL + VAMP Python extension — nanobind bindings.
 *
 * The actual planner, validity checkers, constraint primitives, and
 * pinocchio robot loader live in self-contained internal headers
 * under this directory.  This file is intentionally kept thin: it
 * only imports those headers and exposes the C++ API to Python via
 * nanobind.  If you find yourself adding more than a few lines of
 * non-binding code here, that's a sign it belongs in one of the
 * internal headers instead.
 *
 * One ``OmplVampPlanner<Robot>`` template is instantiated per
 * registered robot description; each instantiation is bound under
 * its own Python class name (``OmplVampPlanner`` for the bimanual
 * cell, ``SingleFr3OmplVampPlanner`` for the standalone single arm,
 * …).  Adding a new robot is a two-line change here: include its
 * generated FK header and call ``bind_planner<NewRobot>(m, "...")``.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "planner.hpp"
#include "robot/bimanual_fr3.hh"
#include "robot/single_fr3.hh"

namespace nb = nanobind;
using bimanual_franka::OmplVampPlanner;
using bimanual_franka::PlanResult;

namespace {

template <class Robot>
void bind_planner(nb::module_ &m, const char *class_name) {
  using Planner = OmplVampPlanner<Robot>;
  nb::class_<Planner>(m, class_name)
      .def(nb::init<>(), "Create a full-body planner.")
      .def(nb::init<std::vector<int>, std::vector<double>>(),
           "Create a subgroup planner.", nb::arg("active_indices"),
           nb::arg("frozen_config"))
      .def("add_pointcloud", &Planner::add_pointcloud, nb::arg("points"),
           nb::arg("r_min"), nb::arg("r_max"), nb::arg("point_radius"))
      .def("remove_pointcloud", &Planner::remove_pointcloud)
      .def("has_pointcloud", &Planner::has_pointcloud)
      .def("add_sphere", &Planner::add_sphere, nb::arg("center"),
           nb::arg("radius"))
      .def("clear_environment", &Planner::clear_environment)
      .def("add_compiled_constraint", &Planner::add_compiled_constraint,
           nb::arg("so_path"), nb::arg("symbol_name"), nb::arg("ambient_dim"),
           nb::arg("co_dim"))
      .def("clear_constraints", &Planner::clear_constraints)
      .def("num_constraints", &Planner::num_constraints)
      .def("add_compiled_cost", &Planner::add_compiled_cost, nb::arg("so_path"),
           nb::arg("symbol_name"), nb::arg("ambient_dim"),
           nb::arg("weight") = 1.0)
      .def("clear_costs", &Planner::clear_costs)
      .def("num_costs", &Planner::num_costs)
      .def("plan", &Planner::plan, nb::arg("start"), nb::arg("goal"),
           nb::arg("planner_name") = "rrtc", nb::arg("time_limit") = 10.0,
           nb::arg("simplify") = true, nb::arg("interpolate") = true,
           nb::arg("interpolate_count") = 0, nb::arg("resolution") = 64.0)
      .def("simplify_path", &Planner::simplify_path, nb::arg("path"),
           nb::arg("time_limit") = 1.0)
      .def("interpolate_path", &Planner::interpolate_path, nb::arg("path"),
           nb::arg("count") = 0, nb::arg("resolution") = 64.0)
      .def("validate", &Planner::validate, nb::arg("config"))
      .def("validate_batch", &Planner::validate_batch, nb::arg("configs"))
      .def("dimension", &Planner::dimension)
      .def("lower_bounds", &Planner::lower_bounds)
      .def("upper_bounds", &Planner::upper_bounds)
      .def("min_max_radii", &Planner::min_max_radii)
      .def("filter_pointcloud", &Planner::filter_pointcloud, nb::arg("points"),
           nb::arg("min_dist"), nb::arg("max_range"), nb::arg("origin"),
           nb::arg("workspace_min"), nb::arg("workspace_max"),
           nb::arg("cull") = true)
      .def("filter_self_from_pointcloud", &Planner::filter_self_from_pointcloud,
           nb::arg("points"), nb::arg("point_radius"), nb::arg("config"))
      .def("set_subgroup", &Planner::set_subgroup, nb::arg("active_indices"),
           nb::arg("frozen_config"))
      .def("set_full_body", &Planner::set_full_body);
}

}  // namespace

NB_MODULE(_ompl_vamp, m) {
  m.doc() =
      "OMPL + VAMP C++ planning extension. One planner class is bound per "
      "registered robot description (BimanualFr3, SingleFr3, ...).";

  nb::class_<PlanResult>(m, "PlanResult")
      .def_ro("solved", &PlanResult::solved)
      .def_ro("path", &PlanResult::path)
      .def_ro("planning_time_ns", &PlanResult::planning_time_ns)
      .def_ro("path_cost", &PlanResult::path_cost);

  // Bimanual cell — keeps the historical ``OmplVampPlanner`` name so
  // existing Python callers continue to work unchanged.
  bind_planner<vamp::robots::BimanualFr3>(m, "OmplVampPlanner");
  // Standalone single-arm FR3.
  bind_planner<vamp::robots::SingleFr3>(m, "SingleFr3OmplVampPlanner");
}
