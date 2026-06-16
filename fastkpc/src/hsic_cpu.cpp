#include "hsic_cpu.hpp"

#include <Rcpp.h>
#include <Rmath.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace {

int mat_idx(int row, int col, int n) {
  return row * n + col;
}

bool finite_vector(const std::vector<double>& values) {
  for (double value : values) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

void validate_hsic_vectors(const std::vector<double>& x,
                           const std::vector<double>& y,
                           const HsicOptions& options) {
  if (x.size() != y.size()) {
    throw std::runtime_error("Sample sizes must agree");
  }
  if (x.size() < 4) {
    throw std::runtime_error("HSIC requires at least 4 observations");
  }
  if (!finite_vector(x) || !finite_vector(y)) {
    throw std::runtime_error("Data contains missing or infinite values");
  }
  if (!std::isfinite(options.sig) || options.sig <= 0.0) {
    throw std::runtime_error("HSIC sig must be positive and finite");
  }
}

std::vector<double> rbf_kernel(const std::vector<double>& values, double sig) {
  const int n = static_cast<int>(values.size());
  const double sigma = 1.0 / sig;
  std::vector<double> out(static_cast<std::size_t>(n) * n, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      const double diff = values[i] - values[j];
      out[mat_idx(i, j, n)] = std::exp(-sigma * diff * diff);
    }
  }
  return out;
}

std::vector<double> center_kernel(const std::vector<double>& kernel, int n) {
  std::vector<double> row_mean(n, 0.0);
  double grand = 0.0;
  for (int i = 0; i < n; ++i) {
    double row_sum = 0.0;
    for (int j = 0; j < n; ++j) {
      row_sum += kernel[mat_idx(i, j, n)];
    }
    row_mean[i] = row_sum / static_cast<double>(n);
    grand += row_sum;
  }
  grand /= static_cast<double>(n) * n;

  std::vector<double> centered(static_cast<std::size_t>(n) * n, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      centered[mat_idx(i, j, n)] =
        kernel[mat_idx(i, j, n)] - row_mean[i] - row_mean[j] + grand;
    }
  }
  return centered;
}

double off_diagonal_mean(const std::vector<double>& kernel, int n) {
  double total = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (i != j) total += kernel[mat_idx(i, j, n)];
    }
  }
  return total / (static_cast<double>(n) * (n - 1.0));
}

double sum_squares(const std::vector<double>& values) {
  double total = 0.0;
  for (double value : values) total += value * value;
  return total;
}

double hsic_statistic_from_centered(const std::vector<double>& Kc,
                                    const std::vector<double>& Lc,
                                    int n) {
  double total = 0.0;
  for (std::size_t i = 0; i < Kc.size(); ++i) total += Kc[i] * Lc[i];
  return total / (static_cast<double>(n) * n);
}

double hsic_statistic_permuted(const std::vector<double>& Kc,
                               const std::vector<double>& Lc,
                               const std::vector<int>& perm,
                               int n) {
  double total = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      total += Kc[mat_idx(i, j, n)] * Lc[mat_idx(perm[i], perm[j], n)];
    }
  }
  return total / (static_cast<double>(n) * n);
}

HsicResult base_result(const char* method, int n, const HsicOptions& options) {
  HsicResult result;
  result.statistic = std::numeric_limits<double>::quiet_NaN();
  result.p_value = std::numeric_limits<double>::quiet_NaN();
  result.mean = std::numeric_limits<double>::quiet_NaN();
  result.variance = std::numeric_limits<double>::quiet_NaN();
  result.shape = std::numeric_limits<double>::quiet_NaN();
  result.scale = std::numeric_limits<double>::quiet_NaN();
  result.n = n;
  result.replicates = 0;
  result.used_seed = options.has_seed;
  result.seed = options.has_seed ? options.seed : 0U;
  result.method = method;
  result.reason = "";
  return result;
}

void set_invalid_gamma_fallback(HsicResult* result, const char* reason) {
  result->p_value = 1.0;
  result->shape = std::numeric_limits<double>::quiet_NaN();
  result->scale = std::numeric_limits<double>::quiet_NaN();
  result->reason = reason;
}

std::vector<int> seeded_permutation(int n, std::mt19937* rng) {
  std::vector<int> perm(n);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), *rng);
  return perm;
}

std::vector<int> r_rng_permutation(int n) {
  std::vector<int> remaining(n);
  std::iota(remaining.begin(), remaining.end(), 0);
  std::vector<int> perm;
  perm.reserve(n);
  while (!remaining.empty()) {
    const int picked = static_cast<int>(std::floor(R::unif_rand() * remaining.size()));
    const int bounded = std::min(picked, static_cast<int>(remaining.size()) - 1);
    perm.push_back(remaining[bounded]);
    remaining.erase(remaining.begin() + bounded);
  }
  return perm;
}

}  // namespace

HsicOptions default_hsic_options() {
  HsicOptions options;
  options.sig = 1.0;
  options.replicates = 100;
  options.include_observed = true;
  options.has_seed = false;
  options.seed = 0U;
  options.return_replicates = true;
  options.cuda_max_n = 2048;
  options.cuda_max_batch_pairs = 64;
  options.cuda_memory_fallback = true;
  return options;
}

HsicResult hsic_gamma_cpu(const std::vector<double>& x,
                          const std::vector<double>& y,
                          const HsicOptions& options) {
  validate_hsic_vectors(x, y, options);
  const int n = static_cast<int>(x.size());
  HsicResult result = base_result("hsic.gamma", n, options);

  const std::vector<double> K = rbf_kernel(x, options.sig);
  const std::vector<double> L = rbf_kernel(y, options.sig);
  const std::vector<double> Kc = center_kernel(K, n);
  const std::vector<double> Lc = center_kernel(L, n);

  result.statistic = hsic_statistic_from_centered(Kc, Lc, n);
  const double mux = off_diagonal_mean(K, n);
  const double muy = off_diagonal_mean(L, n);
  result.mean = (1.0 + mux * muy - mux - muy) / static_cast<double>(n);
  result.variance =
    (2.0 * (n - 4.0) * (n - 5.0) /
     (static_cast<double>(n) * (n - 1.0) * (n - 2.0) * (n - 3.0))) *
    sum_squares(Kc) * sum_squares(Lc) /
    std::pow(static_cast<double>(n), 4.0);

  if (!std::isfinite(result.statistic) || result.statistic < 0.0) {
    set_invalid_gamma_fallback(&result, "HSIC statistic is invalid");
    return result;
  }
  if (!std::isfinite(result.mean) || !std::isfinite(result.variance) ||
      result.mean <= 0.0 || result.variance <= 0.0) {
    set_invalid_gamma_fallback(&result, "HSIC gamma approximation variance is invalid");
    return result;
  }

  result.shape = result.mean * result.mean / result.variance;
  result.scale = result.variance / result.mean;
  if (!std::isfinite(result.shape) || !std::isfinite(result.scale) ||
      result.shape <= 0.0 || result.scale <= 0.0) {
    set_invalid_gamma_fallback(&result, "HSIC gamma approximation parameters are invalid");
    return result;
  }
  result.p_value = R::pgamma(result.statistic, result.shape, result.scale,
                            false, false);
  if (!std::isfinite(result.p_value)) result.p_value = 1.0;
  if (result.p_value < 0.0) result.p_value = 0.0;
  if (result.p_value > 1.0) result.p_value = 1.0;
  return result;
}

HsicResult hsic_permutation_cpu(const std::vector<double>& x,
                                const std::vector<double>& y,
                                const HsicOptions& options) {
  validate_hsic_vectors(x, y, options);
  if (options.replicates < 0) {
    throw std::runtime_error("HSIC permutation replicates must be non-negative");
  }
  const int n = static_cast<int>(x.size());
  HsicResult result = base_result("hsic.perm", n, options);

  const std::vector<double> Kc = center_kernel(rbf_kernel(x, options.sig), n);
  const std::vector<double> Lc = center_kernel(rbf_kernel(y, options.sig), n);
  result.statistic = hsic_statistic_from_centered(Kc, Lc, n);
  result.replicates = options.replicates;
  if (options.return_replicates) {
    result.replicate_statistics.reserve(options.replicates);
  }

  std::mt19937 rng(options.seed);
  int exceedances = options.include_observed ? 1 : 0;
  for (int i = 0; i < options.replicates; ++i) {
    const std::vector<int> perm =
      options.has_seed ? seeded_permutation(n, &rng) : r_rng_permutation(n);
    const double statistic = hsic_statistic_permuted(Kc, Lc, perm, n);
    if (options.return_replicates) result.replicate_statistics.push_back(statistic);
    if (statistic >= result.statistic) ++exceedances;
  }

  const int denominator = options.replicates + (options.include_observed ? 1 : 0);
  result.p_value = denominator > 0 ?
    static_cast<double>(exceedances) / static_cast<double>(denominator) : 1.0;

  if (!result.replicate_statistics.empty()) {
    double total = 0.0;
    for (double value : result.replicate_statistics) total += value;
    result.mean = total / static_cast<double>(result.replicate_statistics.size());
    if (result.replicate_statistics.size() > 1) {
      double ss = 0.0;
      for (double value : result.replicate_statistics) {
        const double diff = value - result.mean;
        ss += diff * diff;
      }
      result.variance =
        ss / static_cast<double>(result.replicate_statistics.size() - 1);
    } else {
      result.variance = 0.0;
    }
  }
  return result;
}
