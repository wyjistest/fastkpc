#include <RcppArmadillo.h>

#include "fastspline_solver.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

arma::mat design_matrix_to_arma(const FastSplineDesign& design) {
  arma::mat X(design.n, design.p);
  for (int row = 0; row < design.n; ++row) {
    for (int col = 0; col < design.p; ++col) {
      X(row, col) = design.X[static_cast<std::size_t>(row) * design.p + col];
    }
  }
  return X;
}

arma::mat penalty_matrix_to_arma(const FastSplineDesign& design) {
  arma::mat P(design.p, design.p);
  for (int row = 0; row < design.p; ++row) {
    for (int col = 0; col < design.p; ++col) {
      P(row, col) = design.P[static_cast<std::size_t>(row) * design.p + col];
    }
  }
  return P;
}

arma::vec response_vector(const Rcpp::NumericMatrix& data, int target) {
  if (target < 0 || target >= data.ncol()) throw std::runtime_error("target column out of range");
  arma::vec y(data.nrow());
  for (int row = 0; row < data.nrow(); ++row) y(row) = data(row, target);
  return y;
}

arma::mat ridge_matrix(int p, double ridge) {
  arma::mat R(p, p, arma::fill::zeros);
  for (int i = 1; i < p; ++i) R(i, i) = ridge;
  return R;
}

std::vector<double> arma_vec_to_std(const arma::vec& values) {
  std::vector<double> out(values.n_elem);
  for (arma::uword i = 0; i < values.n_elem; ++i) out[i] = values(i);
  return out;
}

bool finite_vec(const arma::vec& values) {
  for (arma::uword i = 0; i < values.n_elem; ++i) {
    if (!std::isfinite(values(i))) return false;
  }
  return true;
}

}  // namespace

std::vector<double> lambda_grid(const FastSplineParams& params) {
  const int count = std::max(1, params.lambda_count);
  std::vector<double> out(count);
  if (count == 1) {
    out[0] = params.lambda_min;
    return out;
  }
  const double log_min = std::log(params.lambda_min);
  const double log_max = std::log(params.lambda_max);
  for (int i = 0; i < count; ++i) {
    const double t = static_cast<double>(i) / (count - 1);
    out[i] = std::exp(log_min * (1.0 - t) + log_max * t);
  }
  return out;
}

FastSplineFit fit_fastspline_residuals(const Rcpp::NumericMatrix& data,
                                       int target,
                                       const std::vector<int>& conditioning_set,
                                       const FastSplineParams& params) {
  const FastSplineDesign design = make_fastspline_design(data, conditioning_set, params);
  const arma::mat X = design_matrix_to_arma(design);
  const arma::mat P = penalty_matrix_to_arma(design);
  const arma::vec y = response_vector(data, target);
  const arma::mat XtX = X.t() * X;
  const arma::vec Xty = X.t() * y;
  const std::vector<double> lambdas = lambda_grid(params);

  bool found = false;
  double best_gcv = std::numeric_limits<double>::infinity();
  double best_lambda = std::numeric_limits<double>::quiet_NaN();
  double best_rss = std::numeric_limits<double>::quiet_NaN();
  double best_edf = std::numeric_limits<double>::quiet_NaN();
  int best_ridge_attempts = 0;
  arma::vec best_beta;
  arma::vec best_fitted;
  arma::vec best_residuals;

  double ridge = params.ridge;
  for (int ridge_attempt = 0; ridge <= 1e-4 * (1.0 + 1e-12); ++ridge_attempt) {
    const arma::mat R = ridge_matrix(design.p, ridge);
    for (double lambda : lambdas) {
      const arma::mat A = XtX + lambda * P + R;
      arma::vec beta;
      bool ok = arma::solve(beta, A, Xty, arma::solve_opts::likely_sympd);
      if (!ok || !finite_vec(beta)) {
        ok = arma::solve(beta, A, Xty);
      }
      if (!ok || !finite_vec(beta)) continue;

      arma::mat A_inv;
      ok = arma::inv_sympd(A_inv, A);
      if (!ok) ok = arma::inv(A_inv, A);
      if (!ok || !A_inv.is_finite()) continue;

      const arma::vec fitted = X * beta;
      const arma::vec residuals = y - fitted;
      if (!finite_vec(fitted) || !finite_vec(residuals)) continue;
      const double rss = arma::dot(residuals, residuals);
      const double edf = arma::trace(XtX * A_inv);
      const double denom = static_cast<double>(design.n) - edf;
      if (!std::isfinite(rss) || !std::isfinite(edf) || denom <= 1e-8) continue;
      const double gcv = static_cast<double>(design.n) * rss / (denom * denom);
      if (!std::isfinite(gcv)) continue;

      if (!found || gcv < best_gcv ||
          (std::abs(gcv - best_gcv) <= 1e-14 && lambda < best_lambda)) {
        found = true;
        best_gcv = gcv;
        best_lambda = lambda;
        best_rss = rss;
        best_edf = edf;
        best_ridge_attempts = ridge_attempt;
        best_beta = beta;
        best_fitted = fitted;
        best_residuals = residuals;
      }
    }
    if (found) break;
    ridge *= 100.0;
    if (ridge <= 0.0) ridge = 1e-8;
  }

  if (!found) throw std::runtime_error("fastSpline solve failed");

  FastSplineFit fit;
  fit.residuals = arma_vec_to_std(best_residuals);
  fit.fitted = arma_vec_to_std(best_fitted);
  fit.selected_lambda = best_lambda;
  fit.gcv = best_gcv;
  fit.rss = best_rss;
  fit.edf = best_edf;
  fit.design_cols = design.p;
  fit.ridge_attempts = best_ridge_attempts;
  return fit;
}
