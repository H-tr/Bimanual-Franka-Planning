# Bimanual-Franka-Planning

<div align="center">

[![Python](https://img.shields.io/badge/python-3.11%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green?style=for-the-badge)](https://opensource.org/licenses/BSD-3-Clause)

<sub>↑ 5× sped-up preview — record yours after running the demo</sub>

</div>

A planning library for a bimanual Franka FR3 cell. Provides inverse kinematics (TRAC-IK and Pink), motion planning (OMPL + [VAMP](https://github.com/KavrakiLab/vamp)), and collision-aware planning through a unified Python interface.

The two FR3 arms share a common ``world`` frame anchored at the LEFT arm's base; the relative pose of the RIGHT arm (a 3-DOF planar offset: ``relative_base_x``, ``relative_base_y``, ``relative_base_yaw``) is a **deployment parameter** measured at calibration time and pinned into every plan via ``base_config`` — never a planning DOF.

State vector layout (17 DOF total):

```
[0:3]    relative_base   (x, y, yaw)        deployment-only — pinned for every plan
[3:10]   right arm       (fr3_right_joint1..7)
[10:17]  left arm        (fr3_left_joint1..7)
```

## Features

- **Inverse Kinematics** — TRAC-IK (unconstrained) and Pink (QP-based constrained) solvers with optional self-collision avoidance, on each arm independently
- **Motion Planning** — OMPL + VAMP planner with SIMD collision checking, path validation, and subgroup planning over the left arm, right arm, or both arms (14-DOF dual-arm)
- **Cartesian Coupling** — CasADi-backed `Constraint`s let you couple the two end-effectors (e.g. handover, rigid-object manipulation) and project to the manifold automatically
- **Time Parameterization** — Time-optimal trajectory generation (TOTG) converts planned paths into executable trajectories with per-joint velocity/acceleration limits
- **Collision Geometry** — Spherized URDF representations for efficient collision detection, pointcloud obstacle support

## Quick Start

**Platform**: Linux, Python 3.11+ (see `pixi.toml`).

For inference — running the planners and IK solvers — just pip install:

```bash
git clone --recursive https://github.com/H-tr/Bimanual-Franka-Planning.git
cd Bimanual-Franka-Planning
pip install -e .
```

For development — rebuilding URDFs, regenerating FK headers, running the C++ toolchain end-to-end — use the setup script, which also installs pixi and the conda-forge deps (pinocchio, orocos-kdl, eigen, boost, ...):

```bash
bash scripts/setup.sh
```

## Usage

```bash
# Inverse kinematics
pixi run python examples/ik/basic.py
pixi run -e dev python examples/ik/basic_vis.py           # PyBullet visualization
pixi run -e dev python examples/ik/constrained_vis.py     # Pink QP with optional self-collision

# Motion planning
pixi run python examples/planning/motion.py
pixi run python examples/planning/subgroup.py
pixi run -e dev python examples/planning/constrained/plane.py
pixi run -e dev python examples/planning/cost/orientation_lock.py

# Time parameterization
pixi run python examples/planning/time_parameterization.py

# End-to-end bimanual handover demo
pixi run -e dev python examples/demos/bimanual_handover.py

# Tests
pixi run -e dev test
```

## Project Structure

```
bimanual_franka_planning/  # Core Python package
  kinematics/              # TRAC-IK + Pink IK, FK, collision checking
  planning/                # OMPL+VAMP motion planning, cost + constrained planners
  envs/                    # Simulation environments (PyBullet)
  types/                   # Shared dataclasses (Pose, JointState, ...)
  trajectory/              # TOTG time-optimal trajectory parameterization
  utils/                   # Rotation conversions, video recorder
  resources/               # Packaged URDFs, meshes, scene pointclouds
third_party/
  cricket/                 # FK code generator
  foam/                    # Collision geometry processing
  vamp/                    # SIMD-accelerated collision checking (header-only)
  ompl/                    # The Open Motion Planning Library (built from source)
ext/
  ompl_vamp/               # OMPL + VAMP C++ extension (nanobind)
  time_parameterization/   # TOTG vendored from MoveIt 2 (nanobind)
  trac_ik/                 # KDL + NLopt IK solver (pybind11)
examples/
  ik/                      # TRAC-IK + Pink examples
  planning/                # Motion, subgroup, cost, constrained planning demos
  demos/                   # End-to-end scenarios (bimanual_handover, ...)
  robot/                   # High-level helpers (BimanualFr3Planner, visualizer)
tests/                     # Pytest suite (CI)
scripts/
  setup.sh                 # One-shot: pixi + submodules + cricket/foam build
  build_robot.sh           # Re-compose the bimanual URDF from cricket FR3
  generate_fk.sh           # Regenerate ext/ompl_vamp/robot/bimanual_fr3.hh
  ...
tools/
  build_bimanual_urdf.py   # Stamps two cricket FR3s into one bimanual cell
assets/
  bimanual_fr3_description/  # Source URDF + meshes (input to scripts/build_robot.sh)
```

## Acknowledgements

This project builds on several outstanding open-source libraries:

- **[VAMP](https://github.com/KavrakiLab/vamp)** — SIMD-accelerated motion planning and collision checking (Kavrakilab, Rice University).
- **[OMPL](https://ompl.kavrakilab.org/)** — The Open Motion Planning Library (Kavrakilab, Rice University).
- **[MoveIt 2](https://github.com/moveit/moveit2)** — The vendored TOTG (Time-Optimal Trajectory Generation) implementation in `ext/time_parameterization/` is adapted from MoveIt 2's `trajectory_processing` module, originally by Tobias Kunz and Mike Stilman (Georgia Tech). See `ext/time_parameterization/LICENSE.TOTG` for the full BSD license.
- **[TRAC-IK](https://traclabs.com/projects/trac-ik/)** — Inverse kinematics solver (TRACLabs).
- **[franka_description](https://github.com/frankaemika/franka_description)** — The single-arm FR3 URDF and meshes used as the building block for the bimanual cell, vendored via the [cricket](https://github.com/H-tr/cricket) submodule.
- **[Autolife-Planning](https://github.com/AdaCompNUS/Autolife-Planning)** — This library mirrors the structure, types, and build pipeline of the Autolife-Planning project at NUS AdaComp; switching between the two libraries should require only a robot description swap.
