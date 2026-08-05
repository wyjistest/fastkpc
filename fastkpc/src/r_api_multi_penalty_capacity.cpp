#include "cuda/mgcv_multi_penalty_gcv_capacity.hpp"

#include <Rcpp.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace {

template <typename Vector>
bool all_finite_capacity_input(const Vector& values) {
  return std::all_of(values.begin(), values.end(), [](double value) {
    return std::isfinite(value);
  });
}

const char* capacity_bucket_name(
    fastkpc::MultiPenaltyGcvCapacityBucket bucket) {
  switch (bucket) {
    case fastkpc::MultiPenaltyGcvCapacityBucket::Small64:
      return "q64-p7";
    case fastkpc::MultiPenaltyGcvCapacityBucket::Base80:
      return "q80-p8";
    case fastkpc::MultiPenaltyGcvCapacityBucket::Extended192:
      return "q192-p21";
    case fastkpc::MultiPenaltyGcvCapacityBucket::Extended384:
      return "q384-p42";
    case fastkpc::MultiPenaltyGcvCapacityBucket::Extended559:
      return "q559-p62";
  }
  return "unknown";
}

}  // namespace

extern "C" SEXP C_full_cuda_ci_multi_penalty_capacity_qualify(
    SEXP Xs,
    SEXP magic_qr_packed_s,
    SEXP magic_tau_s,
    SEXP magic_r_s,
    SEXP magic_pivot_s,
    SEXP penalty_roots_s,
    SEXP penalty_matrices_s,
    SEXP penalty_ranks_s,
    SEXP initial_log_sp_s,
    SEXP Ys) {
  BEGIN_RCPP
  if (!Rf_isReal(Xs) || !Rf_isMatrix(Xs) ||
      !Rf_isReal(magic_qr_packed_s) || !Rf_isMatrix(magic_qr_packed_s) ||
      !Rf_isReal(magic_tau_s) ||
      !Rf_isReal(magic_r_s) || !Rf_isMatrix(magic_r_s) ||
      !Rf_isInteger(magic_pivot_s) ||
      !Rf_isNewList(penalty_roots_s) ||
      !Rf_isNewList(penalty_matrices_s) ||
      !Rf_isInteger(penalty_ranks_s) ||
      !Rf_isReal(initial_log_sp_s) ||
      !Rf_isReal(Ys) || !Rf_isMatrix(Ys)) {
    Rcpp::stop("capacity qualification inputs have invalid storage types");
  }

  Rcpp::NumericMatrix X(Xs);
  Rcpp::NumericMatrix qr_packed(magic_qr_packed_s);
  Rcpp::NumericVector tau(magic_tau_s);
  Rcpp::NumericMatrix magic_r(magic_r_s);
  Rcpp::IntegerVector pivot(magic_pivot_s);
  Rcpp::List penalty_roots(penalty_roots_s);
  Rcpp::List penalty_matrices(penalty_matrices_s);
  Rcpp::IntegerVector penalty_ranks(penalty_ranks_s);
  Rcpp::NumericVector initial_log_sp(initial_log_sp_s);
  Rcpp::NumericMatrix Y(Ys);
  const int n = X.nrow();
  const int q = X.ncol();
  const int penalty_count = penalty_roots.size();
  const int target_count = Y.ncol();
  if (n <= q || q <= 0 || qr_packed.nrow() != n ||
      qr_packed.ncol() != q || tau.size() != q ||
      magic_r.nrow() != q || magic_r.ncol() != q || pivot.size() != q ||
      penalty_count <= 1 || penalty_matrices.size() != penalty_count ||
      penalty_ranks.size() != penalty_count ||
      initial_log_sp.size() != penalty_count || Y.nrow() != n ||
      target_count <= 1 || !all_finite_capacity_input(X) ||
      !all_finite_capacity_input(qr_packed) ||
      !all_finite_capacity_input(tau) ||
      !all_finite_capacity_input(magic_r) ||
      !all_finite_capacity_input(initial_log_sp) ||
      !all_finite_capacity_input(Y)) {
    Rcpp::stop("capacity qualification dimensions are inconsistent");
  }

  std::vector<int> pivot_zero_based(static_cast<std::size_t>(q));
  std::vector<unsigned char> pivot_seen(static_cast<std::size_t>(q), 0U);
  for (int index = 0; index < q; ++index) {
    const int value = pivot[index] - 1;
    if (value < 0 || value >= q ||
        pivot_seen[static_cast<std::size_t>(value)] != 0U) {
      Rcpp::stop("capacity qualification QR pivot is invalid");
    }
    pivot_seen[static_cast<std::size_t>(value)] = 1U;
    pivot_zero_based[static_cast<std::size_t>(index)] = value;
  }

  std::vector<std::vector<double>> roots;
  std::vector<std::vector<double>> matrices;
  std::vector<int> ranks;
  roots.reserve(static_cast<std::size_t>(penalty_count));
  matrices.reserve(static_cast<std::size_t>(penalty_count));
  ranks.reserve(static_cast<std::size_t>(penalty_count));
  for (int penalty = 0; penalty < penalty_count; ++penalty) {
    SEXP root_s = penalty_roots[penalty];
    SEXP matrix_s = penalty_matrices[penalty];
    if (!Rf_isReal(root_s) || !Rf_isMatrix(root_s) ||
        !Rf_isReal(matrix_s) || !Rf_isMatrix(matrix_s)) {
      Rcpp::stop("capacity qualification penalties must be matrices");
    }
    Rcpp::NumericMatrix root(root_s);
    Rcpp::NumericMatrix matrix(matrix_s);
    const int rank = penalty_ranks[penalty];
    if (root.nrow() != q || root.ncol() != rank || rank <= 0 ||
        matrix.nrow() != q || matrix.ncol() != q ||
        !all_finite_capacity_input(root) ||
        !all_finite_capacity_input(matrix)) {
      Rcpp::stop("capacity qualification penalty geometry is invalid");
    }
    roots.emplace_back(root.begin(), root.end());
    matrices.emplace_back(matrix.begin(), matrix.end());
    ranks.push_back(rank);
  }

  const fastkpc::MultiPenaltyGcvCapacityBucket bucket =
    fastkpc::multi_penalty_gcv_capacity_bucket(q, penalty_count);
  std::shared_ptr<fastkpc::MultiPenaltyGcvCapacityPrepared> prepared =
    fastkpc::create_multi_penalty_gcv_capacity_prepared(
      X.begin(), qr_packed.begin(), tau.begin(), magic_r.begin(),
      pivot_zero_based.data(), roots, matrices, ranks,
      initial_log_sp.begin(), n, q, penalty_count, target_count, 0);
  std::vector<std::string> target_keys;
  target_keys.reserve(static_cast<std::size_t>(target_count));
  for (int target = 0; target < target_count; ++target) {
    target_keys.push_back(
      "capacity-qualification-target-" + std::to_string(target + 1));
  }
  fastkpc::MultiPenaltyGcvCudaOptimizerControl control;
  const fastkpc::MultiPenaltyGcvCudaOptimization result =
    fastkpc::multi_penalty_gcv_capacity_optimize_batch(
      prepared, Y.begin(), n, target_count, target_keys, control);
  prepared.reset();

  Rcpp::NumericMatrix selected_log_sp(penalty_count, target_count);
  std::copy(
    result.selected_log_sp.begin(), result.selected_log_sp.end(),
    selected_log_sp.begin());
  Rcpp::CharacterMatrix boundary_status(penalty_count, target_count);
  for (int target = 0; target < target_count; ++target) {
    for (int penalty = 0; penalty < penalty_count; ++penalty) {
      const int value = result.boundary_status[
        penalty + penalty_count * target];
      boundary_status(penalty, target) = value == 0 ? "finite-interior" :
        (value == 1 ? "positive-boundary" :
          (value == 2 ? "negative-boundary" :
            (value == 3 ? "finite-after-boundary-probe" : "unresolved")));
    }
  }
  return Rcpp::List::create(
    Rcpp::Named("schema_version") =
      "full-cuda-ci-multi-penalty-capacity-qualification-v1",
    Rcpp::Named("capacity_bucket") = capacity_bucket_name(bucket),
    Rcpp::Named("max_concurrent_setups") =
      fastkpc::multi_penalty_gcv_capacity_max_concurrent_setups(
        q, penalty_count),
    Rcpp::Named("coefficient_dim") = result.coefficient_dim,
    Rcpp::Named("penalty_count") = result.penalty_count,
    Rcpp::Named("target_count") = result.target_count,
    Rcpp::Named("selected_log_sp") = selected_log_sp,
    Rcpp::Named("condition") = Rcpp::wrap(result.condition),
    Rcpp::Named("numerical_rank") = Rcpp::wrap(result.numerical_rank),
    Rcpp::Named("optimizer_status") = Rcpp::wrap(result.optimizer_status),
    Rcpp::Named("optimizer_iterations") =
      Rcpp::wrap(result.optimizer_iterations),
    Rcpp::Named("score_calls") = Rcpp::wrap(result.score_calls),
    Rcpp::Named("objective_calls") = Rcpp::wrap(result.objective_calls),
    Rcpp::Named("step_halving_count") =
      Rcpp::wrap(result.step_halving_count),
    Rcpp::Named("boundary_probe_count") =
      Rcpp::wrap(result.boundary_probe_count),
    Rcpp::Named("boundary_accepted_count") =
      Rcpp::wrap(result.boundary_accepted_count),
    Rcpp::Named("boundary_status") = boundary_status,
    Rcpp::Named("diagnostics") = Rcpp::List::create(
      Rcpp::Named("cpu_objective_count") =
        result.diagnostics.cpu_objective_count,
      Rcpp::Named("cpu_optimizer_count") =
        result.diagnostics.cpu_optimizer_count,
      Rcpp::Named("cpu_multi_penalty_solve_count") =
        result.diagnostics.cpu_multi_penalty_solve_count,
      Rcpp::Named("fallback_count") = result.diagnostics.fallback_count,
      Rcpp::Named("cuda_error_count") =
        result.diagnostics.cuda_error_count,
      Rcpp::Named("true_batched_kernel") =
        result.diagnostics.true_batched_kernel,
      Rcpp::Named("independent_target_states") =
        result.diagnostics.independent_target_states));
  END_RCPP
}
