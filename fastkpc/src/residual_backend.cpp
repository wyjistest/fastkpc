#include "residual_backend.hpp"

#include "dcov_exact_cpu.hpp"

ResidualBackendDescriptor linear_residual_backend_descriptor() {
  ResidualBackendDescriptor descriptor;
  descriptor.name = "linear";
  descriptor.params = "intercept=true;ridge=1e-8";
  return descriptor;
}

std::vector<double> compute_linear_residuals(
  const Rcpp::NumericMatrix& data,
  int target,
  const std::vector<int>& conditioning_set) {
  return residualize_lm(data, target, conditioning_set);
}
