#include "regrvonps_device.hpp"

#include "ci_method.hpp"
#include "cuda/dcov_batch_cuda.hpp"
#include "cuda/fastspline_residual_cuda.hpp"
#include "cuda/hsic_batch_cuda.hpp"
#include "dcov_batch_types.hpp"
#include "hsic_batch_types.hpp"
#include "residual_backend_registry.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

void validate_node(int node, int p) {
  if (node < 0 || node >= p) {
    throw std::runtime_error("node index out of range");
  }
}

int resolve_orientation_batch_size(int requested, int total) {
  if (total <= 0) return 0;
  if (requested <= 0) return total;
  return std::max(1, std::min(requested, total));
}

void fill_repeated_residual_and_columns(const Rcpp::NumericMatrix& data,
                                        const std::vector<double>& residuals,
                                        const std::vector<int>& nodes,
                                        int start,
                                        int count,
                                        std::vector<double>* xmat,
                                        std::vector<double>* ymat) {
  const int n = data.nrow();
  xmat->assign(static_cast<std::size_t>(n) * count, 0.0);
  ymat->assign(static_cast<std::size_t>(n) * count, 0.0);
  for (int k = 0; k < count; ++k) {
    const int node = nodes[start + k];
    for (int row = 0; row < n; ++row) {
      const std::size_t offset = static_cast<std::size_t>(k) * n + row;
      (*xmat)[offset] = residuals[row];
      (*ymat)[offset] = data(row, node);
    }
  }
}

RegrVonPsResult make_empty_result(const Rcpp::NumericMatrix& data,
                                  const std::vector<int>& pdag,
                                  int p,
                                  int V,
                                  const std::vector<int>& S,
                                  ResidualCache* residual_cache) {
  if (data.ncol() != p) {
    throw std::runtime_error("data and pdag dimensions differ");
  }
  validate_node(V, p);
  for (int node : S) validate_node(node, p);

  RegrVonPsResult result;
  result.reject_count = 0;
  result.parents = parents_of(pdag, p, V);
  result.conditioning_set = sorted_unique_union(S, result.parents);
  result.used_cuda = true;
  result.used_cpu_fallback = false;
  result.dcov_batches = 0;
  result.dcov_pairs = 0;
  result.dcc_gamma_tests = 0;
  result.hsic_gamma_tests = 0;
  result.hsic_perm_tests = 0;
  result.hsic_permutation_replicates = 0;
  result.hsic_gamma_cuda_tests = 0;
  result.hsic_perm_cuda_tests = 0;
  result.hsic_cuda_batches = 0;
  result.hsic_cuda_pairs = 0;
  result.hsic_cuda_fallback_tests = 0;
  result.residual_fits = 1;
  result.cuda_residual_fits = 0;
  result.cpu_fallback_fits = 0;
  result.ci_backend = "cuda-dcov";
  result.ci_backend_reason = "";
  ResidualCacheStats before = residual_cache->stats();
  result.cache_requests_before = before.requests;
  result.cache_requests_after = before.requests;
  result.cache_hits_before = before.hits;
  result.cache_hits_after = before.hits;
  return result;
}

std::vector<double> extract_column(const Rcpp::NumericMatrix& data, int col) {
  if (col < 0 || col >= data.ncol()) {
    throw std::runtime_error("data column index out of range");
  }
  std::vector<double> out(data.nrow());
  for (int row = 0; row < data.nrow(); ++row) out[row] = data(row, col);
  return out;
}

HsicBatchOptions make_hsic_batch_options(const HsicOptions& hsic_options,
                                         CiMethodKind kind) {
  HsicBatchOptions options = default_hsic_batch_options();
  if (std::isfinite(hsic_options.sig) && hsic_options.sig > 0.0) {
    options.sig = hsic_options.sig;
  }
  options.permutation_replicates = hsic_options.replicates;
  options.include_observed = hsic_options.include_observed;
  options.has_seed = hsic_options.has_seed;
  options.seed = hsic_options.seed;
  options.return_replicates = false;
  options.max_n = hsic_options.cuda_max_n;
  options.max_batch_pairs = hsic_options.cuda_max_batch_pairs;
  if (kind == CiMethodKind::HsicGamma) {
    options.permutation_replicates = 0;
  }
  return options;
}

RegrVonPsResult make_native_fallback(
    const Rcpp::NumericMatrix& data,
    const std::vector<int>& pdag,
    int p,
    int V,
    const std::vector<int>& S,
    const OrientationOptions& options,
    ResidualCache* residual_cache,
    const std::string& reason) {
  RegrVonPsResult fallback =
    regrvonps_native(data, pdag, p, V, S, options, residual_cache);
  fallback.used_cpu_fallback = true;
  fallback.ci_backend = "native-cpu";
  fallback.ci_backend_reason = reason;
  fallback.hsic_cuda_fallback_tests =
    fallback.hsic_gamma_tests + fallback.hsic_perm_tests;
  return fallback;
}

RegrVonPsResult regrvonps_hsic_cuda(
    const Rcpp::NumericMatrix& data,
    const std::vector<int>& pdag,
    int p,
    int V,
    const std::vector<int>& S,
    const OrientationOptions& options,
    ResidualCache* residual_cache,
    CiMethodKind kind) {
  if (kind == CiMethodKind::HsicPermutation &&
      !options.hsic_options.has_seed) {
    return make_native_fallback(
      data, pdag, p, V, S, options, residual_cache,
      "CUDA HSIC permutation requires explicit seed in this stage");
  }

  std::string reason;
  if (!hsic_cuda_available(&reason)) {
    if (!options.hsic_options.cuda_memory_fallback) {
      throw std::runtime_error(
        reason.empty() ? "CUDA HSIC backend is unavailable" : reason);
    }
    return make_native_fallback(
      data, pdag, p, V, S, options, residual_cache,
      reason.empty() ? "CUDA HSIC backend is unavailable" : reason);
  }

  try {
    RegrVonPsResult result =
      make_empty_result(data, pdag, p, V, S, residual_cache);
    result.ci_backend = "cuda-hsic";
    result.ci_backend_reason = "";

    std::vector<double> residuals;
    if (options.orientation_residual_device == "cuda" &&
        options.residual_backend_name == "fastSpline") {
      const FastSplineCudaFit fit =
        fit_fastspline_residuals_cuda(data, V, result.conditioning_set,
                                      options.fastspline_params,
                                      options.cuda_residual_fallback);
      residuals = fit.fit.residuals;
      result.used_cpu_fallback = fit.diagnostics.fallback_used;
      if (fit.diagnostics.fallback_used) {
        result.cpu_fallback_fits = 1;
      } else {
        result.cuda_residual_fits = 1;
      }
    } else {
      residuals = residual_cache->get(data, V, result.conditioning_set);
      result.used_cuda = true;
      result.residual_fits = 1;
      result.cuda_residual_fits = 0;
      result.cpu_fallback_fits = 0;
    }

    if (!S.empty()) {
      HsicBatchOptions hsic_options =
        make_hsic_batch_options(options.hsic_options, kind);
      int actual_batch_size =
        resolve_orientation_batch_size(options.orientation_batch_size,
                                       static_cast<int>(S.size()));
      if (hsic_options.max_batch_pairs > 0) {
        actual_batch_size = std::min(actual_batch_size,
                                     hsic_options.max_batch_pairs);
      }
      actual_batch_size = std::max(1, actual_batch_size);

      std::vector<double> xmat;
      std::vector<double> ymat;
      for (int start = 0; start < static_cast<int>(S.size());
           start += actual_batch_size) {
        const int count = std::min(actual_batch_size,
                                   static_cast<int>(S.size()) - start);
        fill_repeated_residual_and_columns(data, residuals, S, start, count,
                                           &xmat, &ymat);
        const HsicBatchResult batch =
          kind == CiMethodKind::HsicGamma ?
            hsic_gamma_batch_cuda(xmat.data(), ymat.data(), data.nrow(),
                                  count, hsic_options) :
            hsic_permutation_batch_cuda(xmat.data(), ymat.data(), data.nrow(),
                                        count, hsic_options);
        if (batch.diagnostics.backend != "cuda-hsic") {
          throw std::runtime_error(
            "CUDA HSIC orientation batch did not report cuda-hsic backend");
        }
        for (int k = 0; k < count; ++k) {
          const double p_value = batch.p_values[k];
          result.p_values.push_back(p_value);
          if (p_value < options.alpha) ++result.reject_count;
        }
        ++result.hsic_cuda_batches;
        result.hsic_cuda_pairs += count;
        if (kind == CiMethodKind::HsicGamma) {
          result.hsic_gamma_tests += count;
          result.hsic_gamma_cuda_tests += count;
        } else {
          result.hsic_perm_tests += count;
          result.hsic_perm_cuda_tests += count;
          result.hsic_permutation_replicates +=
            count * batch.diagnostics.permutation_replicates;
        }
      }
    }

    const ResidualCacheStats after = residual_cache->stats();
    result.cache_requests_after = after.requests;
    result.cache_hits_after = after.hits;
    return result;
  } catch (const std::exception& ex) {
    if (!options.cuda_residual_fallback ||
        !options.hsic_options.cuda_memory_fallback) {
      throw;
    }
    return make_native_fallback(data, pdag, p, V, S, options,
                                residual_cache, ex.what());
  }
}

}  // namespace

RegrVonPsResult regrvonps_device(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& pdag,
  int p,
  int V,
  const std::vector<int>& S,
  const OrientationOptions& options,
  ResidualCache* residual_cache) {
  if (residual_cache == NULL) {
    throw std::runtime_error("residual cache is required");
  }
  const CiMethodKind ci_method = parse_ci_method_kind(options.ci_method);
  if (ci_method != CiMethodKind::DccGamma) {
    return regrvonps_hsic_cuda(data, pdag, p, V, S, options,
                               residual_cache, ci_method);
  }
  if (options.orientation_residual_device != "cuda" ||
      options.residual_backend_name != "fastSpline") {
    return regrvonps_native(data, pdag, p, V, S, options, residual_cache);
  }

  try {
    RegrVonPsResult result =
      make_empty_result(data, pdag, p, V, S, residual_cache);

    const FastSplineCudaFit fit =
      fit_fastspline_residuals_cuda(data, V, result.conditioning_set,
                                    options.fastspline_params,
                                    options.cuda_residual_fallback);
    result.used_cpu_fallback = fit.diagnostics.fallback_used;
    if (fit.diagnostics.fallback_used) {
      result.cpu_fallback_fits = 1;
    } else {
      result.cuda_residual_fits = 1;
    }

    if (!S.empty()) {
      DcovBatchOptions dcov_options;
      dcov_options.index = options.index;
      dcov_options.legacy_index = options.legacy_index;
      const int actual_batch_size =
        resolve_orientation_batch_size(options.orientation_batch_size,
                                       static_cast<int>(S.size()));
      std::vector<double> xmat;
      std::vector<double> ymat;
      for (int start = 0; start < static_cast<int>(S.size());
           start += actual_batch_size) {
        const int count = std::min(actual_batch_size,
                                   static_cast<int>(S.size()) - start);
        fill_repeated_residual_and_columns(data, fit.fit.residuals, S, start,
                                           count, &xmat, &ymat);
        const DcovBatchResult batch =
          dcov_batch_cuda(xmat.data(), ymat.data(), data.nrow(), count,
                          dcov_options);
        for (int k = 0; k < count; ++k) {
          const double p_value = batch.p_values[k];
          result.p_values.push_back(p_value);
          if (p_value < options.alpha) ++result.reject_count;
        }
        ++result.dcov_batches;
        result.dcov_pairs += count;
      }
    }

    result.cache_requests_after = result.cache_requests_before;
    result.cache_hits_after = result.cache_hits_before;
    return result;
  } catch (...) {
    if (!options.cuda_residual_fallback) throw;
    RegrVonPsResult fallback =
      regrvonps_native(data, pdag, p, V, S, options, residual_cache);
    fallback.used_cpu_fallback = true;
    fallback.cpu_fallback_fits = 1;
    return fallback;
  }
}
