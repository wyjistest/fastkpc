#ifndef FASTKPC_FASTSPLINE_SOLVER_HPP
#define FASTKPC_FASTSPLINE_SOLVER_HPP

#include "fastspline_basis.hpp"

#include <Rcpp.h>
#include <vector>

struct FastSplineFit {
  std::vector<double> residuals;
  std::vector<double> fitted;
  double selected_lambda;
  double gcv;
  double rss;
  double edf;
  int design_cols;
  int ridge_attempts;
};

FastSplineFit fit_fastspline_residuals(const Rcpp::NumericMatrix& data,
                                       int target,
                                       const std::vector<int>& conditioning_set,
                                       const FastSplineParams& params);

std::vector<double> lambda_grid(const FastSplineParams& params);

#endif
