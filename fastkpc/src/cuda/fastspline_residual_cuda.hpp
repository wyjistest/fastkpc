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
  double residual_summary_sec;
  double d2h_sec;
  double host_select_sec;
  double free_sec;
  double true_batch_total_sec;
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

std::vector<FastSplineCudaFit> fit_fastspline_residuals_cuda_batch(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& targets,
  const std::vector<std::vector<int> >& conditioning_sets,
  const FastSplineParams& params,
  bool fallback);

#endif
