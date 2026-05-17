/**
 * CompiledConstraint — an ompl::base::Constraint subclass that
 * dispatches function() and jacobian() to a CasADi-generated C
 * function loaded at runtime via dlopen.
 *
 * The Python side (bimanual_franka_planning.planning.constraints.Constraint)
 * is responsible for codegen, compilation, and caching.  This class
 * just consumes the resulting .so file.
 *
 * Two input shapes are supported, selected by ``param_dim``:
 *   * ``param_dim == 0``: ``f(q) -> (residual, jac_wrt_q)`` — the
 *     original single-input form; constants are baked into the
 *     residual at codegen time.
 *   * ``param_dim  > 0``: ``f(q, p) -> (residual, jac_wrt_q)`` — the
 *     residual closes over a runtime parameter vector ``p`` whose
 *     length equals ``param_dim``.  Use ``setParameters`` to update
 *     the stored ``p`` between plans without touching the .so cache.
 */

#pragma once

#include <dlfcn.h>
#include <ompl/base/Constraint.h>

#include <Eigen/Core>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace bimanual_franka {

namespace ob = ompl::base;

// CasADi C ABI — casadi_int is long long on 64-bit Linux.
using casadi_fn = int (*)(const double** arg, double** res, long long* iw,
                          double* w, int mem);
using casadi_work_fn = int (*)(long long* sz_arg, long long* sz_res,
                               long long* sz_iw, long long* sz_w);

class CompiledConstraint : public ob::Constraint {
 public:
  CompiledConstraint(unsigned int ambient_dim, unsigned int co_dim,
                     const std::string& so_path, const std::string& symbol_name,
                     unsigned int param_dim = 0,
                     const std::vector<double>& params = {})
      : ob::Constraint(ambient_dim, co_dim),
        so_path_(so_path),
        symbol_name_(symbol_name),
        param_dim_(param_dim) {
    handle_ = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle_)
      throw std::runtime_error("CompiledConstraint: dlopen failed: " +
                               std::string(dlerror()));

    fn_ = reinterpret_cast<casadi_fn>(dlsym(handle_, symbol_name.c_str()));
    if (!fn_) {
      dlclose(handle_);
      throw std::runtime_error("CompiledConstraint: dlsym('" + symbol_name +
                               "') failed: " + std::string(dlerror()));
    }
    std::string work_name = symbol_name + "_work";
    work_fn_ =
        reinterpret_cast<casadi_work_fn>(dlsym(handle_, work_name.c_str()));
    if (!work_fn_) {
      dlclose(handle_);
      throw std::runtime_error("CompiledConstraint: dlsym('" + work_name +
                               "') failed: " + std::string(dlerror()));
    }

    long long sz_arg = 0, sz_res = 0, sz_iw = 0, sz_w = 0;
    work_fn_(&sz_arg, &sz_res, &sz_iw, &sz_w);
    iw_.resize(static_cast<std::size_t>(sz_iw));
    w_.resize(static_cast<std::size_t>(sz_w));

    const long long expected_arg = (param_dim_ > 0) ? 2 : 1;
    if (sz_arg != expected_arg || sz_res != 2) {
      dlclose(handle_);
      throw std::runtime_error(
          "CompiledConstraint: generated function has wrong signature "
          "(expected " +
          std::to_string(expected_arg) + " input(s), 2 outputs; got " +
          std::to_string(sz_arg) + " in, " + std::to_string(sz_res) + " out)");
    }

    if (param_dim_ > 0) {
      params_.resize(param_dim_);
      if (!params.empty()) setParameters(params);
    }
  }

  ~CompiledConstraint() override {
    if (handle_) dlclose(handle_);
  }

  CompiledConstraint(const CompiledConstraint&) = delete;
  CompiledConstraint& operator=(const CompiledConstraint&) = delete;

  void setParameters(const std::vector<double>& p) {
    if (p.size() != static_cast<std::size_t>(param_dim_))
      throw std::invalid_argument(
          "CompiledConstraint::setParameters: expected " +
          std::to_string(param_dim_) + " values, got " +
          std::to_string(p.size()));
    for (unsigned int i = 0; i < param_dim_; ++i)
      params_[static_cast<Eigen::Index>(i)] = p[i];
  }

  unsigned int parameterDim() const { return param_dim_; }

  void function(const Eigen::Ref<const Eigen::VectorXd>& q,
                Eigen::Ref<Eigen::VectorXd> out) const override {
    jac_scratch_.resize(out.size() * static_cast<Eigen::Index>(n_));
    const double* arg[2] = {q.data(),
                            param_dim_ > 0 ? params_.data() : nullptr};
    double* res[2] = {out.data(), jac_scratch_.data()};
    fn_(arg, res, iw_.data(), w_.data(), 0);
  }

  void jacobian(const Eigen::Ref<const Eigen::VectorXd>& q,
                Eigen::Ref<Eigen::MatrixXd> out) const override {
    res_scratch_.resize(getCoDimension());
    const double* arg[2] = {q.data(),
                            param_dim_ > 0 ? params_.data() : nullptr};
    double* res[2] = {res_scratch_.data(), out.data()};
    fn_(arg, res, iw_.data(), w_.data(), 0);
  }

 private:
  std::string so_path_;
  std::string symbol_name_;
  unsigned int param_dim_ = 0;
  void* handle_ = nullptr;
  casadi_fn fn_ = nullptr;
  casadi_work_fn work_fn_ = nullptr;

  // Scratch buffers; mutable because function()/jacobian() are const
  // and OMPL's Newton projection is single-threaded per constraint.
  mutable std::vector<long long> iw_;
  mutable std::vector<double> w_;
  mutable Eigen::VectorXd jac_scratch_;
  mutable Eigen::VectorXd res_scratch_;
  Eigen::VectorXd params_;
};

}  // namespace bimanual_franka
