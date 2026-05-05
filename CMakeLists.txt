cmake_minimum_required(VERSION 3.20)
project(bimanual_franka_planning_ext LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(CheckCXXCompilerFlag)

# CMP0167 NEW requires BoostConfig.cmake (Boost >= 1.70). manylinux_2_28
# (AlmaLinux 8) ships Boost 1.66, which only provides the legacy
# FindBoost.cmake interface. Force the legacy path so the same CMakeLists
# builds against both old (EL8) and modern (conda) Boost.
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif()
set(Boost_NO_BOOST_CMAKE ON)

# ── Dependencies ──────────────────────────────────────────────────────

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization filesystem system)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(orocos_kdl QUIET)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND AND NOT orocos_kdl_FOUND)
    pkg_check_modules(orocos_kdl orocos-kdl)
endif()
find_package(NLopt QUIET)

# ── Architecture flags ───────────────────────────────────────────────
# x86-64-v3 is the Haswell (2013+) micro-arch level: guarantees AVX2,
# BMI1/BMI2, FMA — everything VAMP/SIMDxorshift rely on. Using a fixed
# level instead of -march=native so CI-built wheels run on any modern
# x86_64 CPU, not only the builder. Override for local perf builds:
#   cmake -DBIMANUAL_FR3_ARCH_FLAGS=-march=native ...
set(BIMANUAL_FR3_ARCH_FLAGS "-march=x86-64-v3"
    CACHE STRING "Target CPU arch flags for native extensions")

# Older compilers may not recognize newer micro-architecture levels such as
# x86-64-v3. If that happens, keep the required SIMD feature set by falling
# back to explicit AVX2/FMA/BMI2 flags.
set(_bimanual_fr3_arch_flags "${BIMANUAL_FR3_ARCH_FLAGS}")
check_cxx_compiler_flag("${_bimanual_fr3_arch_flags}" BIMANUAL_FR3_ARCH_FLAGS_SUPPORTED)
if(NOT BIMANUAL_FR3_ARCH_FLAGS_SUPPORTED)
    message(WARNING "Compiler does not support BIMANUAL_FR3_ARCH_FLAGS='${BIMANUAL_FR3_ARCH_FLAGS}'. Falling back to '-march=x86-64 -mavx2 -mfma -mbmi2'.")
    set(_bimanual_fr3_arch_flags "-march=x86-64 -mavx2 -mfma -mbmi2")
    check_cxx_compiler_flag("${_bimanual_fr3_arch_flags}" BIMANUAL_FR3_ARCH_FALLBACK_SUPPORTED)
    if(NOT BIMANUAL_FR3_ARCH_FALLBACK_SUPPORTED)
        message(FATAL_ERROR "Compiler does not support fallback architecture flags '${_bimanual_fr3_arch_flags}'. Set -DBIMANUAL_FR3_ARCH_FLAGS to a supported value.")
    endif()
endif()

separate_arguments(BIMANUAL_FR3_ARCH_FLAGS_LIST NATIVE_COMMAND "${_bimanual_fr3_arch_flags}")

# ── VAMP dependencies (header-only, fetched via CPM) ─────────────────

include(third_party/vamp/cmake/FetchInitCPM.cmake)
include(third_party/vamp/cmake/Dependencies.cmake)

if(TARGET simdxorshift)
    target_compile_options(simdxorshift PRIVATE ${BIMANUAL_FR3_ARCH_FLAGS_LIST})
endif()

# VAMP headers (our fork with BimanualFr3 robot — header-only)
set(VAMP_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/vamp/src/impl")

# ── OMPL (from source, no Python bindings) ────────────────────────────

set(OMPL_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(OMPL_BUILD_DEMOS OFF CACHE BOOL "" FORCE)
set(OMPL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(OMPL_BUILD_VAMP OFF CACHE BOOL "" FORCE)
set(OMPL_REGISTRATION OFF CACHE BOOL "" FORCE)
set(OMPL_BUILD_SHARED OFF CACHE BOOL "" FORCE)

# Apply our carried patches against vendored OMPL before its CMake runs.
# Each patch is idempotent: we read the target file and skip when the patch
# is already present, so re-running cmake or running ``git submodule
# update`` (which resets the OMPL tree) both work.
set(_OMPL_PATCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches")
set(_OMPL_SUBMODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ompl")

# ompl-nanobind-guard.patch: OMPL's OMPLDependencies.cmake unconditionally
# add_subdirectory()s its vendored external/nanobind whenever the path
# exists, ignoring OMPL_BUILD_PYTHON_BINDINGS.  That nanobind 2.x then
# rejects Python < 3.9, breaking cp38 wheel builds.  The patch adds the
# missing OMPL_BUILD_PYTHON_BINDINGS gate.  Delete the patch (and this
# block) once OMPL accepts the fix upstream.
set(_patch "${_OMPL_PATCH_DIR}/ompl-nanobind-guard.patch")
set(_target "${_OMPL_SUBMODULE_DIR}/CMakeModules/OMPLDependencies.cmake")
if(EXISTS "${_target}")
    file(READ "${_target}" _ompl_deps_content)
    if(NOT _ompl_deps_content MATCHES "OMPL_BUILD_PYTHON_BINDINGS AND OMPL_HAVE_PYTHON")
        find_program(PATCH_EXECUTABLE patch REQUIRED)
        execute_process(
            COMMAND "${PATCH_EXECUTABLE}" -p1 -i "${_patch}"
            WORKING_DIRECTORY "${_OMPL_SUBMODULE_DIR}"
            RESULT_VARIABLE _patch_rc
            OUTPUT_VARIABLE _patch_stdout
            ERROR_VARIABLE _patch_stderr
        )
        if(NOT _patch_rc EQUAL 0)
            message(FATAL_ERROR
                "Failed to apply ${_patch} (rc=${_patch_rc}):\n"
                "stdout: ${_patch_stdout}\n"
                "stderr: ${_patch_stderr}")
        endif()
        message(STATUS "Applied carried patch: ompl-nanobind-guard.patch")
    endif()
endif()

add_subdirectory(third_party/ompl EXCLUDE_FROM_ALL)

# ── OMPL + VAMP extension ────────────────────────────────────────────

nanobind_add_module(_ompl_vamp
    ext/ompl_vamp/ompl_vamp_ext.cpp
)

target_include_directories(_ompl_vamp PRIVATE
    ${VAMP_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ompl/src
    # Project-local robot FK/sphere header lives at
    # ext/ompl_vamp/robot/bimanual_fr3.hh; this path lets validity.hpp include
    # it via "robot/bimanual_fr3.hh" without pulling anything from third_party.
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/ompl_vamp
)

target_link_libraries(_ompl_vamp PRIVATE
    ompl
    Eigen3::Eigen
    nigh
    pdqsort
    Boost::serialization
    Boost::filesystem
    Boost::system
    ${CMAKE_DL_LIBS}   # for dlopen/dlsym in CompiledConstraint
)

target_compile_options(_ompl_vamp PRIVATE
    ${BIMANUAL_FR3_ARCH_FLAGS_LIST}
)

install(TARGETS _ompl_vamp LIBRARY DESTINATION bimanual_franka_planning)

# ── Time parameterization extension (TOTG, Eigen-only) ───────────────

nanobind_add_module(_time_parameterization
    ext/time_parameterization/totg.cpp
    ext/time_parameterization/time_parameterization_ext.cpp
)

target_include_directories(_time_parameterization PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/time_parameterization
)

target_link_libraries(_time_parameterization PRIVATE
    Eigen3::Eigen
)

target_compile_options(_time_parameterization PRIVATE
    ${BIMANUAL_FR3_ARCH_FLAGS_LIST}
)

install(TARGETS _time_parameterization LIBRARY DESTINATION bimanual_franka_planning)

# ── TRAC-IK extension (pybind11) ─────────────────────────────────────

if(orocos_kdl_FOUND AND NLopt_FOUND)
    file(GLOB_RECURSE TRAC_IK_SOURCES "ext/trac_ik/*.cpp")

    pybind11_add_module(pytracik ${TRAC_IK_SOURCES})

    target_include_directories(pytracik PRIVATE
        ext/trac_ik
        ext/trac_ik/urdf
        ${orocos_kdl_INCLUDE_DIRS}
    )

    target_link_libraries(pytracik PRIVATE
        ${orocos_kdl_LIBRARIES}
        NLopt::nlopt
    )

    install(TARGETS pytracik LIBRARY DESTINATION .)
else()
    message(STATUS "Skipping pytracik: orocos-kdl or NLopt not found")
endif()
