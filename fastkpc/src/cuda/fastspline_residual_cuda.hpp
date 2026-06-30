#ifndef FASTKPC_FASTSPLINE_RESIDUAL_CUDA_HPP
#define FASTKPC_FASTSPLINE_RESIDUAL_CUDA_HPP

#include "../fastspline_basis.hpp"
#include "../fastspline_solver.hpp"

#include <Rcpp.h>
#include <string>
#include <vector>

struct FastSplineCudaDiagnostics {
  bool cuda_used;
  bool fallback_used;
  std::string reason;
  int batch_group_id;
  int batch_position;
  bool true_batched;
  std::string cholesky_backend;
};

struct FastSplineCudaFit {
  FastSplineFit fit;
  FastSplineCudaDiagnostics diagnostics;
};

struct FastSplineCudaResidualOnlyFit {
  std::vector<double> residuals;
  FastSplineCudaDiagnostics diagnostics;
};

struct FastSplineCudaBatchDiagnostics {
  int requested_fits;
  int groups;
  int true_batched_groups;
  int true_batched_fits;
  int single_fit_calls;
  int cpu_fallback_fits;
  int unique_designs;
  int duplicate_design_fits;
  int max_fits_per_design;
  int max_group_size;
  int min_group_size;
  std::string cholesky_backend;
  std::string batch_mode;
  double grouping_sec;
  double host_pack_sec;
  double alloc_sec;
  double h2d_sec;
  double xtx_xty_sec;
  double pointer_setup_sec;
  double active_copy_sec;
  double build_system_sec;
  double factor_solve_sec;
  double factor_cholesky_sec;
  double factor_rhs_solve_sec;
  double factor_inverse_solve_sec;
  double residual_summary_sec;
  double d2h_sec;
  double d2h_residuals_sec;
  double d2h_metadata_sec;
  double d2h_info_sec;
  int d2h_copy_count;
  double d2h_bytes;
  double d2h_residual_bytes;
  double d2h_metadata_bytes;
  int d2h_metadata_coalesced_count;
  double d2h_metadata_coalesced_bytes;
  double host_select_sec;
  double free_sec;
  double true_batch_total_sec;
  int factorization_count;
  int rhs_solve_count;
  int inverse_solve_count;
  int rhs_solve_api_calls;
  int rhs_target_solves;
  int rhs_custom_solve_count;
  int rhs_cublas_solve_count;
  int rhs_solve_fallback_count;
  double rhs_custom_solve_sec;
  double rhs_cublas_solve_sec;
  int winning_factor_reuse_count;
  int factor_cache_hits;
  int factor_cache_misses;
  int factor_cache_entries;
  double factor_cache_bytes;
  int lambda_candidates;
  int workspace_reuse_count;
  int workspace_grow_count;
  int solver_handle_create_count;
  int per_request_design_x_values;
  int duplicate_design_x_values_avoided;
  int algebraic_rss_count;
  int candidate_residual_materialize_count;
  int winning_residual_materialize_count;
  int algebraic_rss_clamp_count;
  int residual_only_batch_count;
  int residual_full_fit_batch_count;
  int residual_only_fit_count;
  int residual_full_fit_materialize_count;
  int residual_fitted_values_avoided;
  double residual_result_materialize_sec;
  double residual_fitted_materialize_sec;
  double residual_batch_top_level_wall_sec;
  double residual_batch_top_level_unaccounted_sec;
  std::vector<int> group_id;
  std::vector<int> group_n;
  std::vector<int> group_design_cols;
  std::vector<int> group_fit_count;
  std::vector<int> group_true_batched;
  std::vector<int> group_single_fit_calls;
  std::vector<int> group_cpu_fallback_fits;
  std::vector<int> group_unique_designs;
  std::vector<int> group_duplicate_design_fits;
  std::vector<int> group_max_fits_per_design;
  std::vector<std::string> group_cholesky_backend;
  std::vector<std::string> group_status;
  std::vector<std::string> group_reason;
};

struct FastSplineCudaBatchResult {
  std::vector<FastSplineCudaFit> fits;
  FastSplineCudaBatchDiagnostics diagnostics;
};

struct FastSplineCudaResidualOnlyBatchResult {
  std::vector<FastSplineCudaResidualOnlyFit> fits;
  FastSplineCudaBatchDiagnostics diagnostics;
};

struct FastSplineCudaWorkspace;

FastSplineCudaWorkspace* create_fastspline_cuda_workspace();

void destroy_fastspline_cuda_workspace(FastSplineCudaWorkspace* workspace);

FastSplineCudaFit fit_fastspline_residuals_cuda(
  const Rcpp::NumericMatrix& data,
  int target,
  const std::vector<int>& conditioning_set,
  const FastSplineParams& params,
  bool fallback);

FastSplineCudaBatchResult fit_fastspline_residuals_cuda_batch_result(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& targets,
  const std::vector<std::vector<int> >& conditioning_sets,
  const FastSplineParams& params,
  bool fallback);

FastSplineCudaBatchResult fit_fastspline_residuals_cuda_batch_result(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& targets,
  const std::vector<std::vector<int> >& conditioning_sets,
  const FastSplineParams& params,
  bool fallback,
  FastSplineCudaWorkspace* workspace);

FastSplineCudaResidualOnlyBatchResult
fit_fastspline_residuals_cuda_batch_residuals_only(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& targets,
  const std::vector<std::vector<int> >& conditioning_sets,
  const FastSplineParams& params,
  bool fallback,
  FastSplineCudaWorkspace* workspace);

std::vector<FastSplineCudaFit> fit_fastspline_residuals_cuda_batch(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& targets,
  const std::vector<std::vector<int> >& conditioning_sets,
  const FastSplineParams& params,
  bool fallback);

#endif
