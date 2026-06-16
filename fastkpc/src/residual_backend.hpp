#ifndef FASTKPC_RESIDUAL_BACKEND_HPP
#define FASTKPC_RESIDUAL_BACKEND_HPP

#include <Rcpp.h>
#include <string>
#include <vector>

struct ResidualBackendDescriptor {
  std::string name;
  std::string params;
};

ResidualBackendDescriptor linear_residual_backend_descriptor();

std::vector<double> compute_linear_residuals(
  const Rcpp::NumericMatrix& data,
  int target,
  const std::vector<int>& conditioning_set);

#endif
