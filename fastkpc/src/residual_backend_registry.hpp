#ifndef FASTKPC_RESIDUAL_BACKEND_REGISTRY_HPP
#define FASTKPC_RESIDUAL_BACKEND_REGISTRY_HPP

#include "fastspline_basis.hpp"

#include <Rcpp.h>
#include <string>
#include <vector>

enum class ResidualBackendKind {
  Linear,
  FastSpline
};

struct ResidualBackendConfig {
  ResidualBackendKind kind;
  std::string name;
  std::string params;
  FastSplineParams fastspline;
};

std::vector<std::string> list_residual_backend_names();

ResidualBackendConfig make_residual_backend_config(
  const std::string& name,
  const FastSplineParams& fastspline_params);

std::vector<double> compute_residuals_with_backend(
  const Rcpp::NumericMatrix& data,
  int target,
  const std::vector<int>& conditioning_set,
  const ResidualBackendConfig& config);

#endif
