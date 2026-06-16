#include "residual_backend_registry.hpp"

#include "fastspline_solver.hpp"
#include "residual_backend.hpp"

#include <stdexcept>

std::vector<std::string> list_residual_backend_names() {
  std::vector<std::string> out;
  out.push_back("linear");
  out.push_back("fastSpline");
  return out;
}

ResidualBackendConfig make_residual_backend_config(
  const std::string& name,
  const FastSplineParams& fastspline_params) {
  ResidualBackendConfig config;
  config.fastspline = fastspline_params;
  if (name == "linear") {
    const ResidualBackendDescriptor descriptor = linear_residual_backend_descriptor();
    config.kind = ResidualBackendKind::Linear;
    config.name = descriptor.name;
    config.params = descriptor.params;
    return config;
  }
  if (name == "fastSpline") {
    config.kind = ResidualBackendKind::FastSpline;
    config.name = "fastSpline";
    config.params = serialize_fastspline_params(fastspline_params);
    return config;
  }
  throw std::runtime_error("Unknown residual backend: " + name);
}

std::vector<double> compute_residuals_with_backend(
  const Rcpp::NumericMatrix& data,
  int target,
  const std::vector<int>& conditioning_set,
  const ResidualBackendConfig& config) {
  if (config.kind == ResidualBackendKind::Linear) {
    return compute_linear_residuals(data, target, conditioning_set);
  }
  if (config.kind == ResidualBackendKind::FastSpline) {
    return fit_fastspline_residuals(data, target, conditioning_set,
                                    config.fastspline).residuals;
  }
  throw std::runtime_error("Unknown residual backend: " + config.name);
}
