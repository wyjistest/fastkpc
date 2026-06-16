#include "dcov_exact_cpu.hpp"

#include <Rmath.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

double point_distance(const Rcpp::NumericMatrix& mat, int i, int j, double index,
                      bool legacy_index) {
  const int d = mat.ncol();
  double dist = 0.0;
  if (d == 1) {
    dist = std::abs(mat(i, 0) - mat(j, 0));
  } else {
    double ss = 0.0;
    for (int k = 0; k < d; ++k) {
      const double diff = mat(i, k) - mat(j, k);
      ss += diff * diff;
    }
    dist = std::sqrt(ss);
  }
  if (!legacy_index && index != 1.0) dist = std::pow(dist, index);
  return dist;
}

std::vector<double> pairwise_distance_vector(const std::vector<double>& values,
                                             double index,
                                             bool legacy_index) {
  const int n = static_cast<int>(values.size());
  std::vector<double> out(static_cast<std::size_t>(n) * n, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double dist = std::abs(values[i] - values[j]);
      if (!legacy_index && index != 1.0) dist = std::pow(dist, index);
      out[static_cast<std::size_t>(i) * n + j] = dist;
    }
  }
  return out;
}

std::vector<double> center_distance(const std::vector<double>& dist, int n) {
  std::vector<double> row_mean(n, 0.0);
  double grand = 0.0;
  for (int i = 0; i < n; ++i) {
    double row_sum = 0.0;
    for (int j = 0; j < n; ++j) {
      row_sum += dist[static_cast<std::size_t>(i) * n + j];
    }
    row_mean[i] = row_sum / n;
    grand += row_sum;
  }
  grand /= static_cast<double>(n) * n;

  std::vector<double> centered(static_cast<std::size_t>(n) * n, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      centered[static_cast<std::size_t>(i) * n + j] =
        dist[static_cast<std::size_t>(i) * n + j] - row_mean[i] - row_mean[j] + grand;
    }
  }
  return centered;
}

double matrix_mean(const std::vector<double>& values) {
  double total = 0.0;
  for (double value : values) total += value;
  return total / static_cast<double>(values.size());
}

bool finite_vector(const std::vector<double>& values) {
  for (double value : values) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

std::vector<double> solve_linear_system(std::vector<double> A,
                                        std::vector<double> b,
                                        int n) {
  for (int k = 0; k < n; ++k) {
    int pivot = k;
    double pivot_abs = std::abs(A[static_cast<std::size_t>(k) * n + k]);
    for (int i = k + 1; i < n; ++i) {
      const double candidate = std::abs(A[static_cast<std::size_t>(i) * n + k]);
      if (candidate > pivot_abs) {
        pivot = i;
        pivot_abs = candidate;
      }
    }
    if (pivot_abs < 1e-12) {
      A[static_cast<std::size_t>(k) * n + k] += 1e-8;
      pivot_abs = std::abs(A[static_cast<std::size_t>(k) * n + k]);
      pivot = k;
    }
    if (pivot_abs < 1e-14) {
      throw std::runtime_error("linear residualization system is singular");
    }
    if (pivot != k) {
      for (int j = k; j < n; ++j) {
        std::swap(A[static_cast<std::size_t>(k) * n + j],
                  A[static_cast<std::size_t>(pivot) * n + j]);
      }
      std::swap(b[k], b[pivot]);
    }
    for (int i = k + 1; i < n; ++i) {
      const double factor = A[static_cast<std::size_t>(i) * n + k] /
        A[static_cast<std::size_t>(k) * n + k];
      A[static_cast<std::size_t>(i) * n + k] = 0.0;
      for (int j = k + 1; j < n; ++j) {
        A[static_cast<std::size_t>(i) * n + j] -=
          factor * A[static_cast<std::size_t>(k) * n + j];
      }
      b[i] -= factor * b[k];
    }
  }

  std::vector<double> x(n, 0.0);
  for (int i = n - 1; i >= 0; --i) {
    double rhs = b[i];
    for (int j = i + 1; j < n; ++j) {
      rhs -= A[static_cast<std::size_t>(i) * n + j] * x[j];
    }
    x[i] = rhs / A[static_cast<std::size_t>(i) * n + i];
  }
  return x;
}

}  // namespace

double dcov_exact_pvalue(const std::vector<double>& x,
                         const std::vector<double>& y,
                         double index,
                         bool legacy_index) {
  const int n = static_cast<int>(x.size());
  if (n != static_cast<int>(y.size())) throw std::runtime_error("Sample sizes must agree");
  if (n <= 5) throw std::runtime_error("gamma approximation requires n > 5");
  if (!finite_vector(x) || !finite_vector(y)) {
    throw std::runtime_error("Data contains missing or infinite values");
  }
  if (index < 0.0 || index > 2.0) index = 1.0;

  const std::vector<double> K = pairwise_distance_vector(x, index, legacy_index);
  const std::vector<double> L = pairwise_distance_vector(y, index, legacy_index);
  const std::vector<double> A = center_distance(K, n);
  const std::vector<double> B = center_distance(L, n);

  double Sab = 0.0;
  double Saa = 0.0;
  double Sbb = 0.0;
  for (std::size_t i = 0; i < A.size(); ++i) {
    Sab += A[i] * B[i];
    Saa += A[i] * A[i];
    Sbb += B[i] * B[i];
  }

  const double nV2 = Sab / n;
  const double nV2Mean = matrix_mean(K) * matrix_mean(L);
  const double nV2Variance = 2.0 * (n - 4.0) * (n - 5.0) /
    n / (n - 1.0) / (n - 2.0) / (n - 3.0) * Saa * Sbb / (n * n);

  const double alpha = nV2Mean * nV2Mean / nV2Variance;
  const double beta = nV2Variance / nV2Mean;
  const double p = R::pgamma(nV2, alpha, beta, false, false);
  return p;
}

std::vector<double> residualize_lm(const Rcpp::NumericMatrix& data,
                                   int target,
                                   const std::vector<int>& conditioning_set) {
  const int n = data.nrow();
  const int q = static_cast<int>(conditioning_set.size()) + 1;
  std::vector<double> xtx(static_cast<std::size_t>(q) * q, 0.0);
  std::vector<double> xty(q, 0.0);

  for (int row = 0; row < n; ++row) {
    std::vector<double> design(q, 1.0);
    for (int col = 1; col < q; ++col) {
      design[col] = data(row, conditioning_set[col - 1]);
    }
    const double response = data(row, target);
    for (int a = 0; a < q; ++a) {
      xty[a] += design[a] * response;
      for (int b = 0; b < q; ++b) {
        xtx[static_cast<std::size_t>(a) * q + b] += design[a] * design[b];
      }
    }
  }

  const std::vector<double> beta = solve_linear_system(xtx, xty, q);
  std::vector<double> residuals(n, 0.0);
  for (int row = 0; row < n; ++row) {
    double fitted = beta[0];
    for (int col = 1; col < q; ++col) {
      fitted += beta[col] * data(row, conditioning_set[col - 1]);
    }
    residuals[row] = data(row, target) - fitted;
  }
  return residuals;
}
