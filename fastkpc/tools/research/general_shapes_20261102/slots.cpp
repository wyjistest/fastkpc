// Diagnostic-only adapter: fresh exact GPU optimizer versus reusable slots.
#include "cuda/mgcv_multi_penalty_gcv_capacity.hpp"
#include "cuda/mgcv_multi_penalty_gcv_postback.hpp"
#include <Rcpp.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>

using namespace fastkpc;

struct OriginalBackend {
  std::string old;
  bool present;
  OriginalBackend() {
    const char* value = std::getenv("FASTKPC_DCC_GAMMA_VECTOR_POSTBACK");
    present = value != nullptr; old = value ? value : "";
    setenv("FASTKPC_DCC_GAMMA_VECTOR_POSTBACK", "0", 1);
  }
  ~OriginalBackend() {
    if (present) setenv("FASTKPC_DCC_GAMMA_VECTOR_POSTBACK", old.c_str(), 1);
    else unsetenv("FASTKPC_DCC_GAMMA_VECTOR_POSTBACK");
  }
};

template<class T> bool bits(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() &&
    (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}
template<class F> void rejects(F f) {
  bool rejected = false;
  try { f(); } catch (const std::exception&) { rejected = true; }
  if (!rejected) Rcpp::stop("expected a pre-submit rejection");
}
template<class G> G packet(Rcpp::List input, int capacity) {
  Rcpp::NumericMatrix X = input["X"];
  Rcpp::List geometry = input["geometry"];
  G g; g.n = X.nrow(); g.target_capacity = capacity;
  g.X.assign(X.begin(), X.end());
  g.qr = Rcpp::as<std::vector<double>>(geometry["magic_qr_packed"]);
  g.tau = Rcpp::as<std::vector<double>>(geometry["magic_tau"]);
  g.r = Rcpp::as<std::vector<double>>(geometry["magic_r"]);
  g.pivot = Rcpp::as<std::vector<int>>(geometry["magic_pivot"]);
  for (auto& p : g.pivot) --p;
  g.ranks = Rcpp::as<std::vector<int>>(input["ranks"]);
  g.initial_log_sp = Rcpp::as<std::vector<double>>(geometry["initial_log_sp"]);
  Rcpp::List roots = geometry["penalty_roots"], matrices = geometry["penalty_matrices"];
  int rank = 0;
  for (int j = 0; j < g.penalty_count; ++j) {
    g.offsets.push_back(rank); rank += g.ranks[j];
    Rcpp::NumericMatrix root = roots[j], matrix = matrices[j];
    g.roots.insert(g.roots.end(), root.begin(), root.end());
    g.matrices.insert(g.matrices.end(), matrix.begin(), matrix.end());
  }
  return g;
}

template<class G, class Create, class Bind, class Drain>
SEXP qualify(Rcpp::List inputs, Rcpp::NumericMatrix Y, Create create, Bind bind, Drain drain) {
  const int count = Y.ncol();
  std::vector<G> packets;
  for (SEXP input : inputs) packets.push_back(packet<G>(Rcpp::List(input), count));
  const auto largest = std::max_element(packets.begin(), packets.end(),
    [](const G& a, const G& b) { return a.roots.size() < b.roots.size(); });
  const auto counts_before = multi_penalty_gcv_postback_execution_counts();
  const bool vector_backend = multi_penalty_gcv_vector_postback_requested() &&
    (largest->q == 37 || largest->q == 46);
  auto slot = create(*largest, 0);
  std::vector<std::string> keys;
  for (int j = 0; j < count; ++j) {
    std::ostringstream key; key << std::setfill('0') << std::setw(64) << std::hex << (j + 1);
    keys.push_back(key.str());
  }
  MultiPenaltyGcvCudaOptimizerControl control;
  control.exact_primary_only = true; control.force_final_selected_evaluation = true;
  int compared = 0, negative = 0;
  std::shared_ptr<MultiPenaltyGcvCapacityResidualBatch> a, b;
  try {
    for (int round = 0; round < 2; ++round) for (const G& g : packets) {
      bind(slot, g);
      std::vector<std::vector<double>> roots, matrices;
      for (int j = 0; j < g.penalty_count; ++j) {
        roots.emplace_back(g.roots.begin() + g.q * g.offsets[j],
          g.roots.begin() + g.q * (g.offsets[j] + g.ranks[j]));
        matrices.emplace_back(g.matrices.begin() + g.q * g.q * j,
          g.matrices.begin() + g.q * g.q * (j + 1));
      }
      auto fresh = [&] {
        OriginalBackend original;
        return create_multi_penalty_gcv_capacity_prepared(g.X.data(), g.qr.data(),
          g.tau.data(), g.r.data(), g.pivot.data(), roots, matrices, g.ranks,
          g.initial_log_sp.data(), g.n, g.q, g.penalty_count, count, 0);
      }();
      auto reference = multi_penalty_gcv_capacity_optimize_batch_with_residual(
        fresh, Y.begin(), g.n, count, keys, control);
      a = reference.residual;
      auto candidate = multi_penalty_gcv_capacity_optimize_batch_with_residual(
        slot, Y.begin(), g.n, count, keys, control);
      b = candidate.residual;
      rejects([&] { bind(slot, g); }); ++negative;
      auto x = materialize_multi_penalty_gcv_capacity_residual_shadow(a);
      auto y = materialize_multi_penalty_gcv_capacity_residual_shadow(b);
      auto ix = multi_penalty_gcv_capacity_residual_info(a);
      auto iy = multi_penalty_gcv_capacity_residual_info(b);
      if (!bits(reference.optimization.selected_log_sp, candidate.optimization.selected_log_sp) ||
          !bits(x.coefficients, y.coefficients) || !bits(x.fitted, y.fitted) || !bits(x.residuals, y.residuals) ||
          ix.solver_info != iy.solver_info || ix.decomposition_route != iy.decomposition_route ||
          ix.numerical_rank != iy.numerical_rank || ix.optimizer_status != iy.optimizer_status)
        Rcpp::stop("fresh/rebound exact GPU numerical state differs");
      compared += count;
      release_multi_penalty_gcv_capacity_residual(a); a.reset();
      release_multi_penalty_gcv_capacity_residual(b); b.reset();
      const auto before = multi_penalty_gcv_capacity_prepared_info(slot);
      auto wrong = g; wrong.target_capacity = count == 2 ? 3 : 2;
      rejects([&] { bind(slot, wrong); }); ++negative;
      if (multi_penalty_gcv_capacity_prepared_info(slot).generation != before.generation)
        Rcpp::stop("rejected bind changed generation");
    }
    drain(slot); slot.reset();
  } catch (...) {
    try { drain(slot); } catch (...) {}
    if (a) release_multi_penalty_gcv_capacity_residual(a);
    if (b) release_multi_penalty_gcv_capacity_residual(b);
    throw;
  }
  const auto counts_after = multi_penalty_gcv_postback_execution_counts();
  for (int backend = 0; backend < 2; ++backend) {
    const auto targets = std::uint64_t(compared) *
      ((backend == 0 ? 1 : 0) + (backend == int(vector_backend) ? 1 : 0));
    if (counts_after.optimizer_targets[backend] - counts_before.optimizer_targets[backend] != targets ||
        counts_after.created[backend] - counts_before.created[backend] !=
        counts_after.destroyed[backend] - counts_before.destroyed[backend])
      Rcpp::stop("backend identity or lifecycle differs");
  }
  return Rcpp::List::create(Rcpp::Named("targets_compared") = compared,
    Rcpp::Named("negative_cases") = negative, Rcpp::Named("vector_backend") = vector_backend,
    Rcpp::Named("pass") = true);
}

extern "C" SEXP C_general_shape_slots(SEXP level_s, SEXP inputs_s, SEXP Y_s) {
  BEGIN_RCPP
  Rcpp::List inputs(inputs_s); Rcpp::NumericMatrix Y(Y_s);
  switch (Rcpp::as<int>(level_s)) {
    case 3: return qualify<MultiPenaltyGcvL3Geometry>(inputs, Y, create_multi_penalty_gcv_capacity_l3_slot,
      rebind_multi_penalty_gcv_capacity_l3_slot, drain_multi_penalty_gcv_capacity_l3_slot);
    case 4: return qualify<MultiPenaltyGcvL4Geometry>(inputs, Y, create_multi_penalty_gcv_capacity_l4_slot,
      rebind_multi_penalty_gcv_capacity_l4_slot, drain_multi_penalty_gcv_capacity_l4_slot);
    case 5: return qualify<MultiPenaltyGcvL5Geometry>(inputs, Y, create_multi_penalty_gcv_capacity_l5_slot,
      rebind_multi_penalty_gcv_capacity_l5_slot, drain_multi_penalty_gcv_capacity_l5_slot);
    case 6: return qualify<MultiPenaltyGcvL6Geometry>(inputs, Y, create_multi_penalty_gcv_capacity_l6_slot,
      rebind_multi_penalty_gcv_capacity_l6_slot, drain_multi_penalty_gcv_capacity_l6_slot);
  }
  Rcpp::stop("invalid test level");
  END_RCPP
}
