#ifndef FASTKPC_MGCV_EXTRACT_FIXED_SP_CUDA_HPP
#define FASTKPC_MGCV_EXTRACT_FIXED_SP_CUDA_HPP

#include <string>
#include <vector>

struct MgcvExtractGpuFixedSpResult {
  std::vector<double> theta;
  std::vector<double> coefficients;
  std::vector<double> fitted;
  std::vector<double> residuals;
  double rss;
  int n;
  int coefficient_dim;
  int null_dim;
  std::string cholesky_backend;
};

MgcvExtractGpuFixedSpResult mgcv_extract_fixed_sp_solve_cuda(
  const double* X,
  int n,
  int coefficient_dim,
  const double* y,
  const double* Z,
  const double* XtX_null,
  const double* penalty_null,
  const double* Xty_null,
  int null_dim);

#endif
