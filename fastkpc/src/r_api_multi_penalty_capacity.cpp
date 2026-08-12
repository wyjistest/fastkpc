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
    SEXP Ys,
    SEXP fixed_log_sp_s) {
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
      !Rf_isReal(Ys) || !Rf_isMatrix(Ys) ||
      (!Rf_isNull(fixed_log_sp_s) &&
       (!Rf_isReal(fixed_log_sp_s) || !Rf_isMatrix(fixed_log_sp_s)))) {
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
  fastkpc::MultiPenaltyGcvCapacityBatchResult optimized =
    fastkpc::multi_penalty_gcv_capacity_optimize_batch_with_residual(
      prepared, Y.begin(), n, target_count, target_keys, control);
  const fastkpc::MultiPenaltyGcvCudaOptimization& result =
    optimized.optimization;
  const fastkpc::MultiPenaltyGcvCudaResidualInfo optimizer_info =
    fastkpc::multi_penalty_gcv_capacity_residual_info(optimized.residual);
  const fastkpc::MultiPenaltyGcvCudaResidualShadow optimizer_shadow =
    fastkpc::materialize_multi_penalty_gcv_capacity_residual_shadow(
      optimized.residual);
  fastkpc::release_multi_penalty_gcv_capacity_residual(optimized.residual);
  optimized.residual.reset();
  const fastkpc::MultiPenaltyGcvCudaPreparedInfo prepared_after_optimizer =
    fastkpc::multi_penalty_gcv_capacity_prepared_info(prepared);
  std::vector<double> evaluated_log_sp = result.selected_log_sp;
  if (!Rf_isNull(fixed_log_sp_s)) {
    Rcpp::NumericMatrix fixed_log_sp(fixed_log_sp_s);
    if (fixed_log_sp.nrow() != penalty_count ||
        fixed_log_sp.ncol() != target_count ||
        !all_finite_capacity_input(fixed_log_sp)) {
      Rcpp::stop(
        "capacity qualification fixed log-SP dimensions are inconsistent");
    }
    evaluated_log_sp.assign(fixed_log_sp.begin(), fixed_log_sp.end());
  }

  std::shared_ptr<fastkpc::MultiPenaltyGcvCapacityResidualBatch>
    fixed_residual = fastkpc::multi_penalty_gcv_capacity_evaluate_fixed_batch(
      prepared, Y.begin(), evaluated_log_sp.data(), n, target_count,
      target_keys, control.rank_tolerance);
  const fastkpc::MultiPenaltyGcvCudaResidualInfo fixed_info =
    fastkpc::multi_penalty_gcv_capacity_residual_info(fixed_residual);
  const fastkpc::MultiPenaltyGcvCudaResidualShadow fixed_shadow =
    fastkpc::materialize_multi_penalty_gcv_capacity_residual_shadow(
      fixed_residual);
  fastkpc::release_multi_penalty_gcv_capacity_residual(fixed_residual);
  fixed_residual.reset();
  const fastkpc::MultiPenaltyGcvCudaPreparedInfo prepared_after_fixed =
    fastkpc::multi_penalty_gcv_capacity_prepared_info(prepared);
  prepared.reset();

  Rcpp::NumericMatrix selected_log_sp(penalty_count, target_count);
  std::copy(
    result.selected_log_sp.begin(), result.selected_log_sp.end(),
    selected_log_sp.begin());
  Rcpp::NumericMatrix fixed_log_sp(penalty_count, target_count);
  std::copy(evaluated_log_sp.begin(), evaluated_log_sp.end(),
            fixed_log_sp.begin());
  Rcpp::CharacterMatrix boundary_status(penalty_count, target_count);
  Rcpp::NumericMatrix selected_gradient(penalty_count, target_count);
  Rcpp::NumericVector selected_hessian = Rcpp::wrap(result.hessian);
  selected_hessian.attr("dim") = Rcpp::IntegerVector::create(
    penalty_count, penalty_count, target_count);
  Rcpp::NumericMatrix selected_hessian_eigenvalues(
    penalty_count, target_count);
  std::copy(result.gradient.begin(), result.gradient.end(),
            selected_gradient.begin());
  std::copy(result.hessian_eigenvalues.begin(),
            result.hessian_eigenvalues.end(),
            selected_hessian_eigenvalues.begin());
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
  Rcpp::NumericMatrix fixed_coefficients(q, target_count);
  Rcpp::NumericMatrix fixed_fitted(n, target_count);
  Rcpp::NumericMatrix fixed_residuals(n, target_count);
  Rcpp::NumericMatrix optimizer_coefficients(q, target_count);
  Rcpp::NumericMatrix optimizer_fitted(n, target_count);
  Rcpp::NumericMatrix optimizer_residuals(n, target_count);
  std::copy(optimizer_shadow.coefficients.begin(),
            optimizer_shadow.coefficients.end(),
            optimizer_coefficients.begin());
  std::copy(optimizer_shadow.fitted.begin(), optimizer_shadow.fitted.end(),
            optimizer_fitted.begin());
  std::copy(optimizer_shadow.residuals.begin(),
            optimizer_shadow.residuals.end(), optimizer_residuals.begin());
  std::copy(fixed_shadow.coefficients.begin(), fixed_shadow.coefficients.end(),
            fixed_coefficients.begin());
  std::copy(fixed_shadow.fitted.begin(), fixed_shadow.fitted.end(),
            fixed_fitted.begin());
  std::copy(fixed_shadow.residuals.begin(), fixed_shadow.residuals.end(),
            fixed_residuals.begin());
  return Rcpp::List::create(
    Rcpp::Named("schema_version") =
      "full-cuda-ci-multi-penalty-capacity-qualification-v2",
    Rcpp::Named("capacity_bucket") = capacity_bucket_name(bucket),
    Rcpp::Named("max_concurrent_setups") =
      fastkpc::multi_penalty_gcv_capacity_max_concurrent_setups(
        q, penalty_count),
    Rcpp::Named("coefficient_dim") = result.coefficient_dim,
    Rcpp::Named("penalty_count") = result.penalty_count,
    Rcpp::Named("target_count") = result.target_count,
    Rcpp::Named("selected_log_sp") = selected_log_sp,
    Rcpp::Named("fixed_log_sp") = fixed_log_sp,
    Rcpp::Named("optimizer_solver_info") =
      Rcpp::wrap(optimizer_info.solver_info),
    Rcpp::Named("optimizer_decomposition_route") =
      Rcpp::wrap(optimizer_info.decomposition_route),
    Rcpp::Named("optimizer_numerical_rank") =
      Rcpp::wrap(optimizer_info.numerical_rank),
    Rcpp::Named("optimizer_fixed_sp_evaluation") =
      optimizer_info.fixed_sp_evaluation,
    Rcpp::Named("optimizer_device_resident_before_release") =
      optimizer_info.device_resident,
    Rcpp::Named("optimizer_residual_slot_released") =
      !prepared_after_optimizer.residual_slot_leased,
    Rcpp::Named("optimizer_coefficients") = optimizer_coefficients,
    Rcpp::Named("optimizer_fitted") = optimizer_fitted,
    Rcpp::Named("optimizer_residuals") = optimizer_residuals,
    Rcpp::Named("selected_rss") = Rcpp::wrap(result.rss),
    Rcpp::Named("selected_edf") = Rcpp::wrap(result.edf),
    Rcpp::Named("selected_score") = Rcpp::wrap(result.score),
    Rcpp::Named("selected_gradient") = selected_gradient,
    Rcpp::Named("selected_hessian") = selected_hessian,
    Rcpp::Named("selected_rms_gradient") = Rcpp::wrap(result.rms_gradient),
    Rcpp::Named("selected_hessian_eigenvalues") =
      selected_hessian_eigenvalues,
    Rcpp::Named("condition") = Rcpp::wrap(result.condition),
    Rcpp::Named("numerical_rank") = Rcpp::wrap(result.numerical_rank),
    Rcpp::Named("optimizer_status") = Rcpp::wrap(result.optimizer_status),
    Rcpp::Named("solver_info") = Rcpp::wrap(result.solver_info),
    Rcpp::Named("fully_converged") = Rcpp::wrap(result.fully_converged),
    Rcpp::Named("hessian_positive_definite") =
      Rcpp::wrap(result.hessian_positive_definite),
    Rcpp::Named("step_failed") = Rcpp::wrap(result.step_failed),
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
    Rcpp::Named("fixed_solver_info") =
      Rcpp::wrap(fixed_info.solver_info),
    Rcpp::Named("fixed_decomposition_route") =
      Rcpp::wrap(fixed_info.decomposition_route),
    Rcpp::Named("fixed_numerical_rank") =
      Rcpp::wrap(fixed_info.numerical_rank),
    Rcpp::Named("fixed_sp_evaluation") = fixed_info.fixed_sp_evaluation,
    Rcpp::Named("fixed_device_resident_before_release") =
      fixed_info.device_resident,
    Rcpp::Named("fixed_residual_slot_released") =
      !prepared_after_fixed.residual_slot_leased,
    Rcpp::Named("fixed_coefficients") = fixed_coefficients,
    Rcpp::Named("fixed_fitted") = fixed_fitted,
    Rcpp::Named("fixed_residuals") = fixed_residuals,
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
      Rcpp::Named("cuda_guarded_qr_evaluation_count") =
        result.diagnostics.cuda_guarded_qr_evaluation_count,
      Rcpp::Named("cuda_stable_svd_evaluation_count") =
        result.diagnostics.cuda_stable_svd_evaluation_count,
      Rcpp::Named("cuda_exact_primary_target_count") =
        result.diagnostics.cuda_exact_primary_target_count,
      Rcpp::Named("cuda_stability_replay_target_count") =
        result.diagnostics.cuda_stability_replay_target_count,
      Rcpp::Named("cuda_stability_replay_screened_count") =
        result.diagnostics.cuda_stability_replay_screened_count,
      Rcpp::Named("cuda_stability_replay_selected_count") =
        result.diagnostics.cuda_stability_replay_selected_count,
      Rcpp::Named("cuda_stability_replay_error_count") =
        result.diagnostics.cuda_stability_replay_error_count,
      Rcpp::Named("cuda_terminal_boundary_confirmation_count") =
        result.diagnostics.cuda_terminal_boundary_confirmation_count,
      Rcpp::Named(
        "cuda_stability_replay_guarded_qr_trajectory_reason_count") =
          result.diagnostics
            .cuda_stability_replay_guarded_qr_trajectory_reason_count,
      Rcpp::Named(
        "cuda_stability_replay_discarded_guarded_qr_evaluation_count") =
          result.diagnostics
            .cuda_stability_replay_discarded_guarded_qr_evaluation_count,
      Rcpp::Named(
        "cuda_stability_replay_discarded_stable_svd_evaluation_count") =
          result.diagnostics
            .cuda_stability_replay_discarded_stable_svd_evaluation_count,
      Rcpp::Named("true_batched_kernel") =
        result.diagnostics.true_batched_kernel,
      Rcpp::Named("independent_target_states") =
        result.diagnostics.independent_target_states,
      Rcpp::Named("exact_primary_only") =
        result.diagnostics.exact_primary_only));
  END_RCPP
}
