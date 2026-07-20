#include "mgcv_extract_fixed_sp_cuda.hpp"
#include "mgcv_fixed_sp_runtime.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kCompatibilityTolerance = 1e-12;
constexpr const char* kCompatibilityTargetKey =
  "transient-fixed-sp-compatibility-target-v1";

void require_finite(const double* values, std::size_t count,
                    const char* name) {
  for (std::size_t index = 0; index < count; ++index) {
    if (!std::isfinite(values[index])) {
      throw std::runtime_error(
        std::string(name) + " contains missing or infinite values");
    }
  }
}

bool compatibility_close(double actual, double expected) {
  if (!std::isfinite(actual) || !std::isfinite(expected)) return false;
  const double scale = std::max(std::abs(actual), std::abs(expected));
  return std::abs(actual - expected) <=
    kCompatibilityTolerance + kCompatibilityTolerance * scale;
}

void require_compatibility_match(const std::vector<double>& actual,
                                 const double* expected,
                                 const char* message) {
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (!compatibility_close(actual[index], expected[index])) {
      throw std::runtime_error(message);
    }
  }
}

std::vector<double> build_nullspace_design(const double* X,
                                           const double* Z,
                                           int n,
                                           int coefficient_dim,
                                           int null_dim) {
  std::vector<double> result(
    static_cast<std::size_t>(n) * static_cast<std::size_t>(null_dim), 0.0);
  for (int column = 0; column < null_dim; ++column) {
    for (int inner = 0; inner < coefficient_dim; ++inner) {
      const double z = Z[static_cast<std::size_t>(inner) +
        static_cast<std::size_t>(coefficient_dim) * column];
      for (int row = 0; row < n; ++row) {
        result[static_cast<std::size_t>(row) +
               static_cast<std::size_t>(n) * column] +=
          X[static_cast<std::size_t>(row) +
            static_cast<std::size_t>(n) * inner] * z;
      }
    }
  }
  return result;
}

std::vector<double> build_gram(const std::vector<double>& X_null,
                               int n,
                               int null_dim) {
  std::vector<double> result(
    static_cast<std::size_t>(null_dim) *
      static_cast<std::size_t>(null_dim), 0.0);
  for (int column = 0; column < null_dim; ++column) {
    for (int row = 0; row < null_dim; ++row) {
      double value = 0.0;
      for (int observation = 0; observation < n; ++observation) {
        value += X_null[static_cast<std::size_t>(observation) +
                        static_cast<std::size_t>(n) * row] *
          X_null[static_cast<std::size_t>(observation) +
                 static_cast<std::size_t>(n) * column];
      }
      result[static_cast<std::size_t>(row) +
             static_cast<std::size_t>(null_dim) * column] = value;
    }
  }
  return result;
}

std::vector<double> build_rhs(const std::vector<double>& X_null,
                              const double* y,
                              int n,
                              int null_dim) {
  std::vector<double> result(static_cast<std::size_t>(null_dim), 0.0);
  for (int column = 0; column < null_dim; ++column) {
    for (int row = 0; row < n; ++row) {
      result[static_cast<std::size_t>(column)] +=
        X_null[static_cast<std::size_t>(row) +
               static_cast<std::size_t>(n) * column] * y[row];
    }
  }
  return result;
}

void cleanup_compatibility_runtime_noexcept(
    std::shared_ptr<fastkpc::DeviceResidualBatch>* token,
    std::shared_ptr<fastkpc::PreparedSGpuHandle>* handle,
    std::shared_ptr<fastkpc::CudaRuntimeContext>* context) noexcept {
  if (token != nullptr && *token) {
    try {
      fastkpc::release_device_residual(*token);
    } catch (...) {
    }
    try {
      fastkpc::free_device_residual(token);
    } catch (...) {
    }
  }
  if (handle != nullptr && *handle) {
    try {
      fastkpc::free_prepared_s_gpu(handle);
    } catch (...) {
    }
  }
  if (context != nullptr && *context) {
    try {
      fastkpc::free_fixed_sp_runtime(context);
    } catch (...) {
    }
  }
}

}  // namespace

MgcvExtractGpuFixedSpResult mgcv_extract_fixed_sp_solve_cuda(
    const double* X,
    int n,
    int coefficient_dim,
    const double* y,
    const double* Z,
    const double* XtX_null,
    const double* penalty_null,
    const double* Xty_null,
    int null_dim) {
  if (n <= 0) throw std::runtime_error("n must be positive");
  if (coefficient_dim <= 0) {
    throw std::runtime_error("coefficient_dim must be positive");
  }
  if (null_dim <= 0 || null_dim > coefficient_dim) {
    throw std::runtime_error("null_dim must be positive and no larger than coefficient_dim");
  }
  if (X == nullptr || y == nullptr || Z == nullptr || XtX_null == nullptr ||
      penalty_null == nullptr || Xty_null == nullptr) {
    throw std::runtime_error("fixed-sp compatibility input pointer is null");
  }

  const std::size_t x_count = static_cast<std::size_t>(n) *
    static_cast<std::size_t>(coefficient_dim);
  const std::size_t z_count = static_cast<std::size_t>(coefficient_dim) *
    static_cast<std::size_t>(null_dim);
  const std::size_t square_count = static_cast<std::size_t>(null_dim) *
    static_cast<std::size_t>(null_dim);
  require_finite(X, x_count, "X");
  require_finite(y, static_cast<std::size_t>(n), "y");
  require_finite(Z, z_count, "Z");
  require_finite(XtX_null, square_count, "XtX_null");
  require_finite(penalty_null, square_count, "penalty_null");
  require_finite(Xty_null, static_cast<std::size_t>(null_dim), "Xty_null");

  const std::vector<double> X_null = build_nullspace_design(
    X, Z, n, coefficient_dim, null_dim);
  const std::vector<double> recomputed_gram =
    build_gram(X_null, n, null_dim);
  const std::vector<double> recomputed_rhs =
    build_rhs(X_null, y, n, null_dim);
  require_finite(X_null.data(), X_null.size(), "recomputed X_null");
  require_finite(recomputed_gram.data(), recomputed_gram.size(),
                 "recomputed XtX_null");
  require_finite(recomputed_rhs.data(), recomputed_rhs.size(),
                 "recomputed Xty_null");
  require_compatibility_match(
    recomputed_gram, XtX_null,
    "XtX_null does not match independently recomputed crossprod(X_null)");
  require_compatibility_match(
    recomputed_rhs, Xty_null,
    "Xty_null does not match independently recomputed crossprod(X_null, y)");

  std::shared_ptr<fastkpc::CudaRuntimeContext> context;
  std::shared_ptr<fastkpc::PreparedSGpuHandle> handle;
  std::shared_ptr<fastkpc::DeviceResidualBatch> token;
  try {
    int current_device = -1;
    const cudaError_t device_status = cudaGetDevice(&current_device);
    if (device_status != cudaSuccess) {
      throw std::runtime_error(
        std::string("get current CUDA device: ") +
        cudaGetErrorString(device_status));
    }
    context = fastkpc::create_fixed_sp_runtime(current_device);
    fastkpc::FixedSpCapacities capacities;
    capacities.n = n;
    capacities.null_dim = null_dim;
    capacities.target_count = 1;
    capacities.penalty_count = 1;
    capacities.augmented_rows = n + null_dim;
    fastkpc::reserve_fixed_sp_runtime(context, capacities);

    fastkpc::TransientFixedSpCompatibilityHostView setup;
    setup.n = n;
    setup.null_dim = null_dim;
    setup.X_null = X_null.data();
    setup.gram = XtX_null;
    setup.projected_penalty = penalty_null;
    handle =
      fastkpc::create_transient_fixed_sp_compatibility_prepared_s_gpu(
        context, setup);

    const double sp = 1.0;
    fastkpc::FixedSpBatchHostView batch;
    batch.Y = y;
    batch.SP = &sp;
    batch.n = n;
    batch.null_dim = null_dim;
    batch.penalty_count = 1;
    batch.target_count = 1;
    batch.output_mask = fastkpc::FixedSpOutputCoefficients |
      fastkpc::FixedSpOutputFitted |
      fastkpc::FixedSpOutputResiduals |
      fastkpc::FixedSpOutputRss |
      fastkpc::FixedSpOutputRhs;
    batch.planned_routes = {fastkpc::FixedSpRoute::AugmentedSvd};
    batch.target_keys = {kCompatibilityTargetKey};
    token = fastkpc::solve_fixed_sp_batch(handle, batch);

    const fastkpc::DeviceResidualInfo info =
      fastkpc::device_residual_info(token);
    const fastkpc::FixedSpShadowResult shadow =
      fastkpc::materialize_fixed_sp_shadow(token, batch.output_mask);
    if (info.target_count != 1 || info.executed_routes.size() != 1U ||
        info.solver_statuses.size() != 1U ||
        shadow.successful_targets.size() != 1U ||
        shadow.successful_targets[0] == 0U ||
        shadow.coefficients.size() != static_cast<std::size_t>(null_dim) ||
        shadow.fitted.size() != static_cast<std::size_t>(n) ||
        shadow.residuals.size() != static_cast<std::size_t>(n) ||
        shadow.rss.size() != 1U ||
        shadow.cuda_nullspace_rhs.size() !=
          static_cast<std::size_t>(null_dim)) {
      throw std::runtime_error(
        "stable fixed-sp compatibility result is malformed");
    }
    require_compatibility_match(
      shadow.cuda_nullspace_rhs, recomputed_rhs.data(),
      "CUDA-built fixed-sp RHS does not match compatibility oracle");

    MgcvExtractGpuFixedSpResult result;
    result.theta = shadow.coefficients;
    result.coefficients.assign(
      static_cast<std::size_t>(coefficient_dim), 0.0);
    for (int column = 0; column < null_dim; ++column) {
      for (int row = 0; row < coefficient_dim; ++row) {
        result.coefficients[static_cast<std::size_t>(row)] +=
          Z[static_cast<std::size_t>(row) +
            static_cast<std::size_t>(coefficient_dim) * column] *
          result.theta[static_cast<std::size_t>(column)];
      }
    }
    require_finite(
      result.coefficients.data(), result.coefficients.size(),
      "reconstructed coefficients");
    result.fitted = shadow.fitted;
    result.residuals = shadow.residuals;
    result.rss = shadow.rss[0];
    result.n = n;
    result.coefficient_dim = coefficient_dim;
    result.null_dim = null_dim;
    result.cholesky_backend = "stable-runtime-augmented-svd";
    result.runtime_version = "full-cuda-ci-fixed-sp-runtime-v1";
    result.compatibility_transient_context = true;
    result.planned_route = fastkpc::fixed_sp_route_name(
      info.planned_routes[0]);
    result.executed_route = fastkpc::fixed_sp_route_name(
      info.executed_routes[0]);
    result.solver_status = fastkpc::fixed_sp_status_name(
      info.solver_statuses[0]);
    result.cpu_fallback_count = info.cpu_fallback_count;
    result.rhs_device_build_count = info.rhs_device_build_count;
    result.rhs_authority = info.rhs_authority;
    result.full_cuda_data_plane = info.full_cuda_data_plane;

    fastkpc::release_device_residual(token);
    fastkpc::free_device_residual(&token);
    fastkpc::free_prepared_s_gpu(&handle);
    fastkpc::free_fixed_sp_runtime(&context);
    return result;
  } catch (...) {
    const std::exception_ptr error = std::current_exception();
    cleanup_compatibility_runtime_noexcept(&token, &handle, &context);
    std::rethrow_exception(error);
  }
}
