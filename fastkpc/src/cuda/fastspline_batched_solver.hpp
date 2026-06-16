#ifndef FASTKPC_FASTSPLINE_BATCHED_SOLVER_HPP
#define FASTKPC_FASTSPLINE_BATCHED_SOLVER_HPP

#include "fastspline_residual_cuda.hpp"

#include <Rcpp.h>
#include <string>
#include <vector>

struct FastSplineBatchRequest {
  int original_index;
  int target;
  std::vector<int> conditioning_set;
  FastSplineDesign design;
};

struct FastSplineBatchGroup {
  int group_id;
  int n;
  int design_cols;
  std::vector<FastSplineBatchRequest> requests;
};

std::vector<FastSplineBatchGroup> make_fastspline_batch_groups(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& targets,
  const std::vector<std::vector<int> >& conditioning_sets,
  const FastSplineParams& params);

FastSplineCudaBatchResult fit_fastspline_residuals_cuda_true_batch(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& targets,
  const std::vector<std::vector<int> >& conditioning_sets,
  const FastSplineParams& params,
  bool fallback);

#endif
