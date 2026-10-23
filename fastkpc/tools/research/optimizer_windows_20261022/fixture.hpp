#pragma once
#include "cuda/mgcv_multi_penalty_gcv_capacity.hpp"
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace window_diag {
inline void need(bool b, const std::string& message) {
  if (!b) throw std::runtime_error(message);
}
struct IO {
  std::istream* in = nullptr;
  std::ostream* out = nullptr;
  template<class T> void scalar(T& x) {
    if (out) out->write(reinterpret_cast<const char*>(&x), sizeof(x));
    else in->read(reinterpret_cast<char*>(&x), sizeof(x));
    need(out ? bool(*out) : bool(*in), "fixture I/O failed");
  }
  void value(int& x) { scalar(x); }
  void value(double& x) { scalar(x); }
  void value(bool& x) { scalar(x); }
  void value(std::string& x) {
    std::uint64_t n=x.size(); scalar(n); need(n<10000000, "fixture string bound");
    if (out) out->write(x.data(),n);
    else {x.resize(n); in->read(x.data(),n);}
    need(out ? bool(*out) : bool(*in), "fixture string I/O");
  }
  template<class T> void value(std::vector<T>& x) {
    std::uint64_t n=x.size(); scalar(n); need(n<10000000, "fixture vector bound");
    if (in) x.resize(n);
    for (auto& v:x) value(v);
  }
};
struct Geometry {
  int n=0,q=0,penalties=0,capacity=0;
  std::vector<double> X,qr,tau,R,initial;
  std::vector<int> pivot,ranks;
  std::vector<std::vector<double>> roots,matrices;
  void io(IO& io) {
    io.value(n);io.value(q);io.value(penalties);io.value(capacity);
    io.value(X);io.value(qr);io.value(tau);io.value(R);io.value(initial);
    io.value(pivot);io.value(ranks);io.value(roots);io.value(matrices);
  }
};
struct Snapshot {
  std::vector<std::vector<double>> floating;
  std::vector<std::vector<int>> integral;
  void io(IO& io) {io.value(floating);io.value(integral);}
  bool same(const Snapshot& b) const {
    if (integral!=b.integral || floating.size()!=b.floating.size()) return false;
    for (std::size_t i=0;i<floating.size();++i)
      if (floating[i].size()!=b.floating[i].size() ||
          std::memcmp(floating[i].data(),b.floating[i].data(),floating[i].size()*8)) return false;
    return true;
  }
};
inline Snapshot snapshot(const fastkpc::MultiPenaltyGcvCudaOptimization& a,
    const std::shared_ptr<fastkpc::MultiPenaltyGcvCapacityResidualBatch>& r) {
  using namespace fastkpc;
  const auto& d=a.diagnostics;
  need(d.exact_primary_only && d.cuda_exact_primary_target_count==a.target_count &&
    !d.cpu_objective_count && !d.cpu_optimizer_count && !d.cpu_multi_penalty_solve_count &&
    !d.fallback_count && !d.cuda_error_count && !d.cuda_stability_replay_target_count &&
    !d.cuda_terminal_boundary_confirmation_count,"GPU exact-primary authority failed");
  const auto info=multi_penalty_gcv_capacity_residual_info(r);
  const auto shadow=materialize_multi_penalty_gcv_capacity_residual_shadow(r);
  for (int i=0;i<a.target_count;++i)
    need(info.solver_info.at(i)==0 && info.optimizer_status.at(i)==0 &&
      info.decomposition_route.at(i)==kMultiPenaltyGcvDecompositionStableSvd,
      "optimizer residual route/status failed");
  Snapshot s;
  s.floating={a.initial_log_sp,a.selected_log_sp,a.rss,a.edf,a.score,a.condition,
    a.gradient,a.hessian,a.coefficients,a.rms_gradient,a.hessian_eigenvalues,
    shadow.coefficients,shadow.fitted,shadow.residuals};
  s.integral={a.aggregate_penalty_rank,a.numerical_rank,a.solver_info,
    a.optimizer_iterations,a.score_calls,a.objective_calls,a.step_halving_count,
    a.newton_trial_count,a.steepest_descent_trial_count,a.boundary_probe_count,
    a.boundary_accepted_count,a.boundary_status,a.fully_converged,
    a.hessian_positive_definite,a.step_failed,a.optimizer_status,
    info.solver_info,info.decomposition_route,info.numerical_rank};
  return s;
}
struct Request {
  Geometry geometry;
  std::vector<double> Y;
  std::vector<std::string> keys;
  fastkpc::MultiPenaltyGcvCudaOptimizerControl control;
  Snapshot expected;
  void io(IO& io) {
    geometry.io(io);io.value(Y);io.value(keys);
    io.value(control.convergence_tolerance);io.value(control.max_step_halving);
    io.value(control.max_iterations);io.value(control.max_newton_step);
    io.value(control.boundary_probe_step);io.value(control.max_boundary_probes);
    io.value(control.rank_tolerance);io.value(control.decomposition_trace_capacity_per_target);
    io.value(control.force_stability_replay_all);io.value(control.exact_primary_only);
    io.value(control.force_final_selected_evaluation);expected.io(io);
  }
};
struct Fixture {
  int level=0,window=0;
  std::vector<Request> requests;
  void io(IO& io) {
    std::string schema="optimizer-window-fixture-v1";io.value(schema);
    need(schema=="optimizer-window-fixture-v1","fixture schema");
    io.value(level);io.value(window);
    int n=requests.size();io.value(n);need(n==64,"fixture must hold exactly 64 setups");
    if (io.in) requests.resize(n);
    for (auto& r:requests) r.io(io);
  }
  void save(const std::string& path) {
    std::ofstream out(path,std::ios::binary);IO t{nullptr,&out};io(t);
  }
  static Fixture read(const std::string& path) {
    Fixture f;std::ifstream in(path,std::ios::binary);IO t{&in,nullptr};f.io(t);
    need(in.peek()==EOF,"fixture trailing bytes");return f;
  }
};
}
