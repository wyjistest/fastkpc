#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp17)]]

#include "dcov_exact_cpu.hpp"
#include "fastspline_basis.hpp"
#include "fastspline_solver.hpp"
#include "hsic_cpu.hpp"
#include "orientation_matrix.hpp"
#include "orientation_rules.hpp"
#include "regrvonps_native.hpp"
#include "residual_backend.hpp"
#include "residual_backend_registry.hpp"
#include "residual_cache.hpp"
#include "skeleton_engine.hpp"
#include "wanpdag_engine.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <utility>

namespace {

std::vector<double> numeric_vector_to_std(Rcpp::NumericVector values) {
  std::vector<double> out(values.size());
  for (int i = 0; i < values.size(); ++i) out[i] = values[i];
  return out;
}

bool finite_numeric_matrix(Rcpp::NumericMatrix values) {
  for (int col = 0; col < values.ncol(); ++col) {
    for (int row = 0; row < values.nrow(); ++row) {
      if (!std::isfinite(values(row, col))) return false;
    }
  }
  return true;
}

std::string row_key(Rcpp::NumericMatrix values, int row) {
  std::ostringstream out;
  out.precision(17);
  for (int col = 0; col < values.ncol(); ++col) {
    if (col > 0) out << "|";
    out << values(row, col);
  }
  return out.str();
}

Rcpp::NumericMatrix unique_rows_in_order(Rcpp::NumericMatrix values) {
  std::vector<int> keep;
  std::vector<std::string> seen;
  for (int row = 0; row < values.nrow(); ++row) {
    const std::string key = row_key(values, row);
    if (std::find(seen.begin(), seen.end(), key) != seen.end()) continue;
    seen.push_back(key);
    keep.push_back(row);
  }
  Rcpp::NumericMatrix out(keep.size(), values.ncol());
  for (int i = 0; i < static_cast<int>(keep.size()); ++i) {
    for (int col = 0; col < values.ncol(); ++col) {
      out(i, col) = values(keep[static_cast<std::size_t>(i)], col);
    }
  }
  return out;
}

Rcpp::NumericMatrix evenly_spaced_knots(Rcpp::NumericMatrix unique_rows,
                                        int requested_k) {
  const int unique_n = unique_rows.nrow();
  const int d = unique_rows.ncol();
  const int knot_count = std::max(1, std::min(requested_k, unique_n));
  Rcpp::NumericMatrix knots(knot_count, d);
  if (knot_count == 1) {
    for (int col = 0; col < d; ++col) knots(0, col) = unique_rows(0, col);
    return knots;
  }
  for (int i = 0; i < knot_count; ++i) {
    const double pos = static_cast<double>(i) *
      static_cast<double>(unique_n - 1) / static_cast<double>(knot_count - 1);
    const int idx = static_cast<int>(std::floor(pos + 0.5));
    for (int col = 0; col < d; ++col) knots(i, col) = unique_rows(idx, col);
  }
  return knots;
}

arma::mat rcpp_matrix_to_arma(Rcpp::NumericMatrix values) {
  arma::mat out(values.nrow(), values.ncol());
  for (int row = 0; row < values.nrow(); ++row) {
    for (int col = 0; col < values.ncol(); ++col) out(row, col) = values(row, col);
  }
  return out;
}

Rcpp::NumericMatrix arma_matrix_to_rcpp(const arma::mat& values) {
  Rcpp::NumericMatrix out(values.n_rows, values.n_cols);
  for (arma::uword row = 0; row < values.n_rows; ++row) {
    for (arma::uword col = 0; col < values.n_cols; ++col) {
      out(row, col) = values(row, col);
    }
  }
  return out;
}

Rcpp::NumericMatrix constraint_null_space_cpp(Rcpp::NumericMatrix C,
                                              int p,
                                              double tol) {
  if (C.nrow() == 0) {
    Rcpp::NumericMatrix identity(p, p);
    for (int i = 0; i < p; ++i) identity(i, i) = 1.0;
    return identity;
  }
  arma::mat Ct = rcpp_matrix_to_arma(C).t();
  arma::mat Q;
  arma::mat R;
  arma::qr_econ(Q, R, Ct);
  int rank = 0;
  const int diag_count = std::min(R.n_rows, R.n_cols);
  for (int i = 0; i < diag_count; ++i) {
    if (std::abs(R(i, i)) > tol) ++rank;
  }
  arma::mat Qfull;
  arma::mat Rfull;
  arma::qr(Qfull, Rfull, Ct);
  if (rank >= p) Rcpp::stop("constraint matrix leaves no free space");
  return arma_matrix_to_rcpp(Qfull.cols(rank, p - 1));
}

double tprs_radial_value(double radius, int dimension) {
  if (radius <= 0.0) return 0.0;
  if (dimension == 1) return radius * radius * radius;
  return radius * radius * std::log(radius);
}

Rcpp::NumericMatrix kpc_tprs_polynomial_null_space(Rcpp::NumericMatrix S) {
  const int n = S.nrow();
  const int d = S.ncol();
  Rcpp::NumericMatrix out(n, d + 1);
  for (int row = 0; row < n; ++row) {
    out(row, 0) = 1.0;
    for (int col = 0; col < d; ++col) out(row, col + 1) = S(row, col);
  }
  return out;
}

Rcpp::NumericMatrix center_columns(Rcpp::NumericMatrix values,
                                   Rcpp::NumericVector* shift_out) {
  Rcpp::NumericMatrix out(values.nrow(), values.ncol());
  Rcpp::NumericVector shift(values.ncol());
  for (int col = 0; col < values.ncol(); ++col) {
    double total = 0.0;
    for (int row = 0; row < values.nrow(); ++row) total += values(row, col);
    shift[col] = total / static_cast<double>(values.nrow());
    for (int row = 0; row < values.nrow(); ++row) {
      out(row, col) = values(row, col) - shift[col];
    }
  }
  if (shift_out != nullptr) *shift_out = shift;
  return out;
}

Rcpp::NumericMatrix kpc_tprs_radial_basis(Rcpp::NumericMatrix S,
                                          Rcpp::NumericMatrix knots) {
  const int n = S.nrow();
  const int d = S.ncol();
  const int k = knots.nrow();
  Rcpp::NumericMatrix out(n, k);
  for (int row = 0; row < n; ++row) {
    for (int knot = 0; knot < k; ++knot) {
      double dist_sq = 0.0;
      for (int col = 0; col < d; ++col) {
        const double diff = S(row, col) - knots(knot, col);
        dist_sq += diff * diff;
      }
      out(row, knot) = tprs_radial_value(std::sqrt(dist_sq), d);
    }
  }
  return out;
}

Rcpp::NumericMatrix cbind_numeric_matrices(Rcpp::NumericMatrix left,
                                           Rcpp::NumericMatrix right) {
  if (left.nrow() != right.nrow()) {
    Rcpp::stop("cannot combine matrices with different row counts");
  }
  Rcpp::NumericMatrix out(left.nrow(), left.ncol() + right.ncol());
  for (int row = 0; row < out.nrow(); ++row) {
    for (int col = 0; col < left.ncol(); ++col) out(row, col) = left(row, col);
    for (int col = 0; col < right.ncol(); ++col) {
      out(row, left.ncol() + col) = right(row, col);
    }
  }
  return out;
}

Rcpp::NumericMatrix kpc_tprs_penalty_matrix(Rcpp::NumericMatrix knots,
                                            int null_space_rank) {
  const int d = knots.ncol();
  const int k = knots.nrow();
  const int p = null_space_rank + k;
  Rcpp::NumericMatrix out(p, p);
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < k; ++j) {
      double dist_sq = 0.0;
      for (int col = 0; col < d; ++col) {
        const double diff = knots(i, col) - knots(j, col);
        dist_sq += diff * diff;
      }
      out(null_space_rank + i, null_space_rank + j) =
        tprs_radial_value(std::sqrt(dist_sq), d);
    }
  }
  return out;
}

Rcpp::NumericMatrix kpc_tprs_centering_constraint(Rcpp::NumericMatrix X) {
  Rcpp::NumericMatrix out(1, X.ncol());
  for (int col = 0; col < X.ncol(); ++col) {
    double total = 0.0;
    for (int row = 0; row < X.nrow(); ++row) total += X(row, col);
    out(0, col) = total / static_cast<double>(X.nrow());
  }
  return out;
}

Rcpp::NumericMatrix kpc_tprs_column_sum_constraint(Rcpp::NumericMatrix X) {
  Rcpp::NumericMatrix out(1, X.ncol());
  for (int col = 0; col < X.ncol(); ++col) {
    double total = 0.0;
    for (int row = 0; row < X.nrow(); ++row) total += X(row, col);
    out(0, col) = total;
  }
  return out;
}

Rcpp::NumericVector arma_vector_to_rcpp(const arma::vec& values) {
  Rcpp::NumericVector out(values.n_elem);
  for (arma::uword i = 0; i < values.n_elem; ++i) out[i] = values(i);
  return out;
}

Rcpp::NumericVector arma_row_rms(const arma::mat& values) {
  Rcpp::NumericVector out(values.n_cols);
  for (arma::uword col = 0; col < values.n_cols; ++col) {
    out[col] = std::sqrt(arma::dot(values.col(col), values.col(col)) /
                         static_cast<double>(values.n_rows));
  }
  return out;
}

int arma_rank(const arma::mat& values, double tol) {
  arma::vec singular_values;
  arma::svd(singular_values, values);
  int rank = 0;
  for (arma::uword i = 0; i < singular_values.n_elem; ++i) {
    if (singular_values(i) > tol) ++rank;
  }
  return rank;
}

double arma_frobenius_norm(const arma::mat& values) {
  return std::sqrt(arma::accu(arma::square(values)));
}

double tprs_eta_const_cpp(int m, int d) {
  const double pi = std::asin(1.0) * 2.0;
  const double ghalf = std::sqrt(pi);
  double f = 1.0;
  const int d2 = d / 2;
  const int m2 = 2 * m;
  if (m2 <= d) Rcpp::stop("thin plate spline requires 2m > d");
  if (d % 2 == 0) {
    f = ((m + 1 + d2) % 2) ? -1.0 : 1.0;
    for (int i = 0; i < m2 - 1; ++i) f /= 2.0;
    for (int i = 0; i < d2; ++i) f /= pi;
    for (int i = 2; i < m; ++i) f /= static_cast<double>(i);
    for (int i = 2; i <= m - d2; ++i) f /= static_cast<double>(i);
  } else {
    f = ghalf;
    const int k = m - (d - 1) / 2;
    for (int i = 0; i < k; ++i) f /= -0.5 - static_cast<double>(i);
    for (int i = 0; i < m; ++i) f /= 4.0;
    for (int i = 0; i < d2; ++i) f /= pi;
    f /= ghalf;
    for (int i = 2; i < m; ++i) f /= static_cast<double>(i);
  }
  return f;
}

double tprs_eta_from_squared_distance(double dist_sq, int m, int d) {
  if (dist_sq <= 0.0) return 0.0;
  double f = tprs_eta_const_cpp(m, d);
  const int d2 = d / 2;
  if (d % 2 == 0) {
    f *= std::log(dist_sq) * 0.5;
    for (int i = 0; i < m - d2; ++i) f *= dist_sq;
  } else {
    for (int i = 0; i < m - d2 - 1; ++i) f *= dist_sq;
    f *= std::sqrt(dist_sq);
  }
  return f;
}

struct UniqueRows1D {
  Rcpp::NumericMatrix unique_rows;
  std::vector<int> map_to_unique;
};

UniqueRows1D sorted_unique_rows_1d(Rcpp::NumericMatrix shifted) {
  std::vector<std::pair<double, int> > rows;
  rows.reserve(static_cast<std::size_t>(shifted.nrow()));
  for (int row = 0; row < shifted.nrow(); ++row) {
    rows.push_back(std::make_pair(shifted(row, 0), row));
  }
  std::stable_sort(rows.begin(), rows.end(),
                   [](const std::pair<double, int>& a,
                      const std::pair<double, int>& b) {
                     if (a.first == b.first) return a.second < b.second;
                     return a.first < b.first;
                   });

  std::vector<double> unique_values;
  std::vector<int> map_to_unique(static_cast<std::size_t>(shifted.nrow()));
  int current = -1;
  double last = 0.0;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const double value = rows[i].first;
    if (i == 0 || value != last) {
      unique_values.push_back(value);
      last = value;
      ++current;
    }
    map_to_unique[static_cast<std::size_t>(rows[i].second)] = current;
  }

  Rcpp::NumericMatrix unique_rows(unique_values.size(), 1);
  for (std::size_t row = 0; row < unique_values.size(); ++row) {
    unique_rows(static_cast<int>(row), 0) = unique_values[row];
  }
  return UniqueRows1D{unique_rows, map_to_unique};
}

Rcpp::IntegerVector std_int_vector_to_rcpp(const std::vector<int>& values) {
  Rcpp::IntegerVector out(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) out[static_cast<int>(i)] = values[i];
  return out;
}

Rcpp::List kpc_tprs_residual_cpp_setup_1d(
    Rcpp::NumericMatrix S,
    Rcpp::NumericMatrix shifted,
    Rcpp::NumericVector shift,
    const UniqueRows1D& unique_info,
    int basis_rank,
    int null_space_rank,
    int k_def,
    double tol) {
  const int n = S.nrow();
  const int unique_n = unique_info.unique_rows.nrow();
  if (unique_n > 2000) {
    Rcpp::stop("kpcTprsResidualCPP requires unique conditioning locations <= 2000");
  }
  if (unique_n < basis_rank) {
    Rcpp::stop("A term has fewer unique conditioning locations than the basis dimension");
  }

  const int penalized_rank = basis_rank - null_space_rank;
  arma::vec x(unique_n);
  for (int row = 0; row < unique_n; ++row) x(row) = unique_info.unique_rows(row, 0);

  arma::mat E(unique_n, unique_n, arma::fill::zeros);
  for (int row = 0; row < unique_n; ++row) {
    for (int col = 0; col < row; ++col) {
      const double diff = x(row) - x(col);
      const double value = tprs_eta_from_squared_distance(diff * diff, 2, 1);
      E(row, col) = value;
      E(col, row) = value;
    }
  }

  arma::mat T(unique_n, null_space_rank, arma::fill::ones);
  T.col(1) = x;

  arma::vec eigval;
  arma::mat eigvec;
  if (!arma::eig_sym(eigval, eigvec, E)) {
    Rcpp::stop("1D TPRS eigen decomposition failed");
  }
  std::vector<int> order(static_cast<std::size_t>(eigval.n_elem));
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    const double aa = std::abs(eigval(static_cast<arma::uword>(a)));
    const double bb = std::abs(eigval(static_cast<arma::uword>(b)));
    if (aa == bb) return eigval(static_cast<arma::uword>(a)) >
      eigval(static_cast<arma::uword>(b));
    return aa > bb;
  });

  arma::mat U(unique_n, basis_rank);
  arma::vec selected_eigenvalues(basis_rank);
  for (int col = 0; col < basis_rank; ++col) {
    const int idx = order[static_cast<std::size_t>(col)];
    U.col(col) = eigvec.col(static_cast<arma::uword>(idx));
    selected_eigenvalues(col) = eigval(static_cast<arma::uword>(idx));
  }
  double truncation_eigengap = R_PosInf;
  if (static_cast<int>(order.size()) > basis_rank) {
    const double kept = std::abs(selected_eigenvalues(basis_rank - 1));
    const double next = std::abs(eigval(static_cast<arma::uword>(
      order[static_cast<std::size_t>(basis_rank)])));
    truncation_eigengap = kept - next;
  }

  const arma::mat TU = T.t() * U;
  const int rank_T = arma_rank(T, tol);
  const int rank_TU = arma_rank(TU, tol);
  if (rank_TU != null_space_rank) {
    Rcpp::stop("1D TPRS T'U constraint rank does not match null-space rank");
  }

  arma::mat svd_u;
  arma::vec svd_s;
  arma::mat svd_v;
  if (!arma::svd(svd_u, svd_s, svd_v, TU)) {
    Rcpp::stop("1D TPRS T'U SVD failed");
  }
  arma::mat Z_tps = svd_v.cols(rank_TU, basis_rank - 1);
  const arma::mat X_pen_unique = U * arma::diagmat(selected_eigenvalues) * Z_tps;

  arma::mat X_unique(unique_n, basis_rank, arma::fill::zeros);
  X_unique.cols(0, penalized_rank - 1) = X_pen_unique;
  X_unique.cols(penalized_rank, basis_rank - 1) = T;

  arma::mat X(n, basis_rank, arma::fill::zeros);
  for (int row = 0; row < n; ++row) {
    X.row(row) = X_unique.row(
      static_cast<arma::uword>(unique_info.map_to_unique[static_cast<std::size_t>(row)]));
  }

  arma::mat penalty(basis_rank, basis_rank, arma::fill::zeros);
  penalty.submat(0, 0, penalized_rank - 1, penalized_rank - 1) =
    Z_tps.t() * arma::diagmat(selected_eigenvalues) * Z_tps;

  arma::mat UZ(unique_n + null_space_rank, basis_rank, arma::fill::zeros);
  UZ.submat(0, 0, unique_n - 1, penalized_rank - 1) = U * Z_tps;
  UZ.submat(unique_n, penalized_rank,
            unique_n + null_space_rank - 1, basis_rank - 1) =
    arma::eye(null_space_rank, null_space_rank);
  arma::mat UZ_unscaled = UZ;

  const Rcpp::NumericVector pre_rms_column_norms = arma_row_rms(X);
  for (int col = 0; col < basis_rank; ++col) {
    const double w = pre_rms_column_norms[col];
    if (!std::isfinite(w) || w <= tol) {
      Rcpp::stop("1D TPRS RMS scaling encountered a zero column");
    }
    X.col(col) /= w;
    UZ.col(col) /= w;
    penalty.row(col) /= w;
    penalty.col(col) /= w;
  }
  const Rcpp::NumericVector post_rms_column_norms = arma_row_rms(X);

  Rcpp::NumericMatrix X_rcpp = arma_matrix_to_rcpp(X);
  Rcpp::NumericMatrix penalty_rcpp = arma_matrix_to_rcpp(penalty);
  Rcpp::NumericMatrix constraint = kpc_tprs_column_sum_constraint(X_rcpp);
  Rcpp::NumericMatrix Z_ident = constraint_null_space_cpp(
    constraint, basis_rank, tol);
  arma::mat Z_ident_arma = rcpp_matrix_to_arma(Z_ident);
  Rcpp::NumericMatrix X_absorbed = arma_matrix_to_rcpp(X * Z_ident_arma);
  Rcpp::NumericMatrix penalty_absorbed =
    arma_matrix_to_rcpp(Z_ident_arma.t() * penalty * Z_ident_arma);

  arma::mat radial_full(n, penalized_rank, arma::fill::zeros);
  arma::mat polynomial_full(n, null_space_rank, arma::fill::zeros);
  radial_full = X.cols(0, penalized_rank - 1);
  polynomial_full = X.cols(penalized_rank, basis_rank - 1);

  const double z_orthogonality_error = arma_frobenius_norm(
    Z_tps.t() * Z_tps - arma::eye(penalized_rank, penalized_rank));
  const double tps_constraint_error = arma_frobenius_norm(TU * Z_tps);

  return Rcpp::List::create(
    Rcpp::Named("backend_family") = "kpcTprsResidualCPP",
    Rcpp::Named("schema_version") = "setup-shadow-v1",
    Rcpp::Named("X") = X_rcpp,
    Rcpp::Named("penalty") = penalty_rcpp,
    Rcpp::Named("constraint") = constraint,
    Rcpp::Named("raw") = Rcpp::List::create(
      Rcpp::Named("shift") = shift,
      Rcpp::Named("shifted_covariates") = shifted,
      Rcpp::Named("unique_locations") = unique_info.unique_rows,
      Rcpp::Named("unique_row_index") = std_int_vector_to_rcpp(unique_info.map_to_unique),
      Rcpp::Named("radial_kernel_block") = arma_matrix_to_rcpp(E),
      Rcpp::Named("radial") = arma_matrix_to_rcpp(radial_full),
      Rcpp::Named("polynomial") = arma_matrix_to_rcpp(polynomial_full),
      Rcpp::Named("penalty") = penalty_rcpp,
      Rcpp::Named("constraint") = constraint,
      Rcpp::Named("UZ") = arma_matrix_to_rcpp(UZ),
      Rcpp::Named("UZ_unscaled") = arma_matrix_to_rcpp(UZ_unscaled),
      Rcpp::Named("eigenvectors") = arma_matrix_to_rcpp(U),
      Rcpp::Named("selected_eigenvalues") = arma_vector_to_rcpp(selected_eigenvalues),
      Rcpp::Named("tps_side_constraint") = arma_matrix_to_rcpp(TU),
      Rcpp::Named("tps_null_space") = arma_matrix_to_rcpp(Z_tps)
    ),
    Rcpp::Named("absorbed") = Rcpp::List::create(
      Rcpp::Named("Z") = Z_ident,
      Rcpp::Named("X") = X_absorbed,
      Rcpp::Named("penalty") = penalty_absorbed,
      Rcpp::Named("effective_rank") = X_absorbed.ncol(),
      Rcpp::Named("null_space_rank") = null_space_rank
    ),
    Rcpp::Named("knots") = unique_info.unique_rows,
    Rcpp::Named("unique_rows") = unique_info.unique_rows,
    Rcpp::Named("basis_rank") = basis_rank,
    Rcpp::Named("null_space_rank") = null_space_rank,
    Rcpp::Named("penalized_rank") = penalized_rank,
    Rcpp::Named("effective_rank") = X_absorbed.ncol(),
    Rcpp::Named("k_def") = k_def,
    Rcpp::Named("k") = basis_rank,
    Rcpp::Named("radial_basis") = "eta_1d_m2",
    Rcpp::Named("polynomial_basis") = "1 + s1",
    Rcpp::Named("smooth_geometry") = "joint-isotropic",
    Rcpp::Named("tol") = tol,
    Rcpp::Named("diagnostics") = Rcpp::List::create(
      Rcpp::Named("selected_eigenvalues") = arma_vector_to_rcpp(selected_eigenvalues),
      Rcpp::Named("truncation_eigengap") = truncation_eigengap,
      Rcpp::Named("rank_T") = rank_T,
      Rcpp::Named("rank_TU") = rank_TU,
      Rcpp::Named("Z_orthogonality_error") = z_orthogonality_error,
      Rcpp::Named("TPS_constraint_error") = tps_constraint_error,
      Rcpp::Named("pre_rms_column_norms") = pre_rms_column_norms,
      Rcpp::Named("post_rms_column_norms") = post_rms_column_norms
    )
  );
}

Rcpp::LogicalMatrix adjacency_to_matrix(const std::vector<int>& adjacency, int p) {
  Rcpp::LogicalMatrix out(p, p);
  for (int i = 0; i < p; ++i) {
    for (int j = 0; j < p; ++j) {
      out(i, j) = adjacency[static_cast<std::size_t>(i) * p + j] != 0;
    }
  }
  return out;
}

Rcpp::NumericMatrix pmax_to_matrix(const std::vector<double>& pmax, int p) {
  Rcpp::NumericMatrix out(p, p);
  for (int i = 0; i < p; ++i) {
    for (int j = 0; j < p; ++j) {
      out(i, j) = pmax[static_cast<std::size_t>(i) * p + j];
    }
  }
  return out;
}

Rcpp::List sepsets_to_list(const std::vector<std::vector<std::vector<int> > >& sepsets) {
  const int p = static_cast<int>(sepsets.size());
  Rcpp::List out(p);
  for (int i = 0; i < p; ++i) {
    Rcpp::List row(p);
    for (int j = 0; j < p; ++j) {
      Rcpp::IntegerVector value(sepsets[i][j].size());
      for (int k = 0; k < value.size(); ++k) value[k] = sepsets[i][j][k] + 1;
      row[j] = value;
    }
    out[i] = row;
  }
  return out;
}

Rcpp::List level_log_to_list(const std::vector<std::vector<LevelDeletion> >& logs) {
  Rcpp::List out(logs.size());
  for (int level = 0; level < static_cast<int>(logs.size()); ++level) {
    Rcpp::List entries(logs[level].size());
    for (int i = 0; i < static_cast<int>(logs[level].size()); ++i) {
      const LevelDeletion& entry = logs[level][i];
      Rcpp::IntegerVector cond(entry.conditioning_set.size());
      for (int k = 0; k < cond.size(); ++k) cond[k] = entry.conditioning_set[k] + 1;
      entries[i] = Rcpp::List::create(
        Rcpp::Named("x") = entry.x + 1,
        Rcpp::Named("y") = entry.y + 1,
        Rcpp::Named("S") = cond,
        Rcpp::Named("p.value") = entry.p_value
      );
    }
    out[level] = entries;
  }
  return out;
}

Rcpp::List residual_cache_stats_to_list(const SkeletonResult& result) {
  return Rcpp::List::create(
    Rcpp::Named("enabled") = result.residual_cache_enabled,
    Rcpp::Named("requests") = result.residual_cache_requests,
    Rcpp::Named("hits") = result.residual_cache_hits,
    Rcpp::Named("misses") = result.residual_cache_misses,
    Rcpp::Named("computations") = result.residual_cache_computations,
    Rcpp::Named("stored_vectors") = result.residual_cache_stored_vectors,
    Rcpp::Named("stored_values") = result.residual_cache_stored_values,
    Rcpp::Named("backend_name") = result.residual_backend
  );
}

Rcpp::List hsic_result_to_list(const HsicResult& result) {
  return Rcpp::List::create(
    Rcpp::Named("method") = result.method,
    Rcpp::Named("statistic") = result.statistic,
    Rcpp::Named("estimate") = result.statistic,
    Rcpp::Named("estimates") = Rcpp::NumericVector::create(
      Rcpp::Named("HSIC") = result.statistic,
      Rcpp::Named("HSIC mean") = result.mean,
      Rcpp::Named("HSIC variance") = result.variance
    ),
    Rcpp::Named("p.value") = result.p_value,
    Rcpp::Named("replicates") = Rcpp::NumericVector(
      result.replicate_statistics.begin(),
      result.replicate_statistics.end()
    ),
    Rcpp::Named("diagnostics") = Rcpp::List::create(
      Rcpp::Named("hsic") = result.statistic,
      Rcpp::Named("mean") = result.mean,
      Rcpp::Named("variance") = result.variance,
      Rcpp::Named("shape") = result.shape,
      Rcpp::Named("scale") = result.scale,
      Rcpp::Named("n") = result.n,
      Rcpp::Named("replicates") = result.replicates,
      Rcpp::Named("used_seed") = result.used_seed,
      Rcpp::Named("seed") = result.used_seed ? static_cast<int>(result.seed) : NA_INTEGER,
      Rcpp::Named("reason") = result.reason
    )
  );
}

Rcpp::List skeleton_result_to_list(const SkeletonResult& result, int p,
                                   const char* backend) {
  return Rcpp::List::create(
    Rcpp::Named("adjacency") = adjacency_to_matrix(result.adjacency, p),
    Rcpp::Named("sepsets") = sepsets_to_list(result.sepsets),
    Rcpp::Named("pMax") = pmax_to_matrix(result.pmax, p),
    Rcpp::Named("n.edgetests") = Rcpp::IntegerVector(result.n_edge_tests.begin(),
                                                     result.n_edge_tests.end()),
    Rcpp::Named("per.level.log") = level_log_to_list(result.per_level_log),
    Rcpp::Named("backend") = backend,
    Rcpp::Named("residual_backend") = result.residual_backend,
    Rcpp::Named("residual_backend_params") = result.residual_backend_params,
    Rcpp::Named("residual_cache") = residual_cache_stats_to_list(result),
    Rcpp::Named("ci_method") = result.ci_method.empty() ? "dcc.gamma" : result.ci_method,
    Rcpp::Named("ci_backend") = result.ci_backend.empty() ? "native-cpu" : result.ci_backend,
    Rcpp::Named("ci_backend_reason") = result.ci_backend_reason,
    Rcpp::Named("ci_diagnostics") = Rcpp::List::create(
      Rcpp::Named("ci_dcc_gamma_tests") = result.ci_dcc_gamma_tests,
      Rcpp::Named("ci_hsic_gamma_tests") = result.ci_hsic_gamma_tests,
      Rcpp::Named("ci_hsic_perm_tests") = result.ci_hsic_perm_tests,
      Rcpp::Named("ci_hsic_permutation_replicates") =
        result.ci_hsic_permutation_replicates,
      Rcpp::Named("ci_hsic_gamma_cuda_tests") =
        result.ci_hsic_gamma_cuda_tests,
      Rcpp::Named("ci_hsic_perm_cuda_tests") =
        result.ci_hsic_perm_cuda_tests,
      Rcpp::Named("ci_hsic_cuda_batches") = result.ci_hsic_cuda_batches,
      Rcpp::Named("ci_hsic_cuda_pairs") = result.ci_hsic_cuda_pairs,
      Rcpp::Named("ci_hsic_cuda_fallback_tests") =
        result.ci_hsic_cuda_fallback_tests,
      Rcpp::Named("ci_hsic_cuda_memory_bytes") =
        static_cast<double>(result.ci_hsic_cuda_memory_bytes),
      Rcpp::Named("ci_hsic_cuda_max_n") = result.ci_hsic_cuda_max_n,
      Rcpp::Named("ci_hsic_cuda_max_batch_pairs") =
        result.ci_hsic_cuda_max_batch_pairs
    )
  );
}

Rcpp::List residual_stats_to_list(const ResidualCacheStats& stats) {
  return Rcpp::List::create(
    Rcpp::Named("enabled") = stats.enabled,
    Rcpp::Named("requests") = stats.requests,
    Rcpp::Named("hits") = stats.hits,
    Rcpp::Named("misses") = stats.misses,
    Rcpp::Named("computations") = stats.computations,
    Rcpp::Named("stored_vectors") = stats.stored_vectors,
    Rcpp::Named("stored_values") = stats.stored_values,
    Rcpp::Named("backend_name") = stats.backend_name
  );
}

Rcpp::IntegerMatrix pdag_to_matrix(const std::vector<int>& pdag, int p) {
  Rcpp::IntegerMatrix out(p, p);
  for (int row = 0; row < p; ++row) {
    for (int col = 0; col < p; ++col) {
      out(row, col) = pdag[static_cast<std::size_t>(row) * p + col];
    }
  }
  return out;
}

Rcpp::List orientation_events_to_list(const std::vector<OrientationEvent>& events) {
  Rcpp::List out(events.size());
  for (int i = 0; i < static_cast<int>(events.size()); ++i) {
    const OrientationEvent& event = events[i];
    Rcpp::IntegerVector S(event.S.size());
    for (int j = 0; j < S.size(); ++j) S[j] = event.S[j] + 1;
    out[i] = Rcpp::List::create(
      Rcpp::Named("phase") = event.phase,
      Rcpp::Named("rule") = event.rule,
      Rcpp::Named("x") = event.x < 0 ? NA_INTEGER : event.x + 1,
      Rcpp::Named("y") = event.y < 0 ? NA_INTEGER : event.y + 1,
      Rcpp::Named("z") = event.z < 0 ? NA_INTEGER : event.z + 1,
      Rcpp::Named("S") = S,
      Rcpp::Named("p.value") = event.p_value,
      Rcpp::Named("accepted") = event.accepted,
      Rcpp::Named("message") = event.message
    );
  }
  return out;
}

Rcpp::List orientation_result_to_list(const OrientationResult& result) {
  return Rcpp::List::create(
    Rcpp::Named("pdag") = pdag_to_matrix(result.pdag, result.p),
    Rcpp::Named("events") = orientation_events_to_list(result.events),
    Rcpp::Named("counts") = Rcpp::List::create(
      Rcpp::Named("collider") = result.collider_orientations,
      Rcpp::Named("rule1") = result.rule1_orientations,
      Rcpp::Named("rule2") = result.rule2_orientations,
      Rcpp::Named("rule3") = result.rule3_orientations,
      Rcpp::Named("generalized") = result.generalized_orientations,
      Rcpp::Named("regrvonps_calls") = result.regrvonps_calls
    ),
    Rcpp::Named("residual_backend") = result.residual_backend,
    Rcpp::Named("residual_backend_params") = result.residual_backend_params,
    Rcpp::Named("residual_device") = result.residual_device,
    Rcpp::Named("residual_device_requested") = result.residual_device_requested,
    Rcpp::Named("residual_device_reason") = result.residual_device_reason,
    Rcpp::Named("orientation_batch_size_requested") =
      result.orientation_batch_size_requested,
    Rcpp::Named("orientation_batch_size_used") =
      result.orientation_batch_size_used,
    Rcpp::Named("residual_cache") = Rcpp::List::create(
      Rcpp::Named("requests") = result.residual_cache_requests,
      Rcpp::Named("hits") = result.residual_cache_hits,
      Rcpp::Named("computations") = result.residual_cache_computations
    ),
    Rcpp::Named("diagnostics") = Rcpp::List::create(
      Rcpp::Named("orientation_residual_device") = result.residual_device,
      Rcpp::Named("orientation_residual_device_requested") =
        result.residual_device_requested,
      Rcpp::Named("orientation_residual_device_reason") =
        result.residual_device_reason,
      Rcpp::Named("orientation_batch_size_requested") =
        result.orientation_batch_size_requested,
      Rcpp::Named("orientation_batch_size_used") =
        result.orientation_batch_size_used,
      Rcpp::Named("regrvonps_calls") = result.regrvonps_calls,
      Rcpp::Named("regrvonps_cuda_calls") = result.regrvonps_cuda_calls,
      Rcpp::Named("regrvonps_cpu_calls") = result.regrvonps_cpu_calls,
      Rcpp::Named("orientation_dcov_batches") =
        result.orientation_dcov_batches,
      Rcpp::Named("orientation_dcov_pairs") = result.orientation_dcov_pairs,
      Rcpp::Named("regrvonps_dcc_gamma_tests") =
        result.regrvonps_dcc_gamma_tests,
      Rcpp::Named("regrvonps_hsic_gamma_tests") =
        result.regrvonps_hsic_gamma_tests,
      Rcpp::Named("regrvonps_hsic_perm_tests") =
        result.regrvonps_hsic_perm_tests,
      Rcpp::Named("regrvonps_hsic_permutation_replicates") =
        result.regrvonps_hsic_permutation_replicates,
      Rcpp::Named("regrvonps_hsic_gamma_cuda_tests") =
        result.regrvonps_hsic_gamma_cuda_tests,
      Rcpp::Named("regrvonps_hsic_perm_cuda_tests") =
        result.regrvonps_hsic_perm_cuda_tests,
      Rcpp::Named("regrvonps_hsic_cuda_batches") =
        result.regrvonps_hsic_cuda_batches,
      Rcpp::Named("regrvonps_hsic_cuda_pairs") =
        result.regrvonps_hsic_cuda_pairs,
      Rcpp::Named("regrvonps_hsic_cuda_fallback_tests") =
        result.regrvonps_hsic_cuda_fallback_tests,
      Rcpp::Named("orientation_residual_fits") =
        result.orientation_residual_fits,
      Rcpp::Named("orientation_cuda_residual_fits") =
        result.orientation_cuda_residual_fits,
      Rcpp::Named("orientation_cpu_fallback_fits") =
        result.orientation_cpu_fallback_fits,
      Rcpp::Named("orientation_cache_requests") =
        result.residual_cache_requests,
      Rcpp::Named("orientation_cache_hits") = result.residual_cache_hits,
      Rcpp::Named("orientation_cache_computations") =
        result.residual_cache_computations
    ),
    Rcpp::Named("ci_method") = result.ci_method.empty() ? "dcc.gamma" : result.ci_method,
    Rcpp::Named("ci_backend") =
      result.ci_backend.empty() ? "native-cpu" : result.ci_backend,
    Rcpp::Named("ci_backend_reason") = result.ci_backend_reason,
    Rcpp::Named("ci_diagnostics") = Rcpp::List::create(
      Rcpp::Named("regrvonps_dcc_gamma_tests") =
        result.regrvonps_dcc_gamma_tests,
      Rcpp::Named("regrvonps_hsic_gamma_tests") =
        result.regrvonps_hsic_gamma_tests,
      Rcpp::Named("regrvonps_hsic_perm_tests") =
        result.regrvonps_hsic_perm_tests,
      Rcpp::Named("regrvonps_hsic_permutation_replicates") =
        result.regrvonps_hsic_permutation_replicates,
      Rcpp::Named("regrvonps_hsic_gamma_cuda_tests") =
        result.regrvonps_hsic_gamma_cuda_tests,
      Rcpp::Named("regrvonps_hsic_perm_cuda_tests") =
        result.regrvonps_hsic_perm_cuda_tests,
      Rcpp::Named("regrvonps_hsic_cuda_batches") =
        result.regrvonps_hsic_cuda_batches,
      Rcpp::Named("regrvonps_hsic_cuda_pairs") =
        result.regrvonps_hsic_cuda_pairs,
      Rcpp::Named("regrvonps_hsic_cuda_fallback_tests") =
        result.regrvonps_hsic_cuda_fallback_tests
    )
  );
}

FastSplineParams parse_fastspline_params(Rcpp::List values);
HsicOptions parse_hsic_options(Rcpp::List hsic_params,
                               Rcpp::List permutation_params);

SkeletonResult skeleton_result_from_R(Rcpp::LogicalMatrix adjacency,
                                      Rcpp::List sepsets) {
  const int p = adjacency.nrow();
  if (adjacency.ncol() != p) {
    Rcpp::stop("adjacency must be a square matrix");
  }
  if (sepsets.size() != p) {
    Rcpp::stop("sepsets dimension mismatch");
  }

  SkeletonResult result;
  result.adjacency.assign(static_cast<std::size_t>(p) * p, 0);
  for (int row = 0; row < p; ++row) {
    for (int col = 0; col < p; ++col) {
      const int edge = adjacency(row, col);
      if (edge == NA_LOGICAL) {
        Rcpp::stop("adjacency contains NA");
      }
      result.adjacency[static_cast<std::size_t>(row) * p + col] =
        edge == TRUE ? 1 : 0;
    }
  }

  result.sepsets.assign(p, std::vector<std::vector<int> >(p));
  for (int row = 0; row < p; ++row) {
    Rcpp::List sepset_row = sepsets[row];
    if (sepset_row.size() != p) {
      Rcpp::stop("sepsets dimension mismatch");
    }
    for (int col = 0; col < p; ++col) {
      Rcpp::IntegerVector value = sepset_row[col];
      for (int i = 0; i < value.size(); ++i) {
        if (Rcpp::IntegerVector::is_na(value[i])) continue;
        const int node = value[i] - 1;
        if (node < 0 || node >= p) {
          Rcpp::stop("sepset index out of range");
        }
        result.sepsets[row][col].push_back(node);
      }
      std::sort(result.sepsets[row][col].begin(), result.sepsets[row][col].end());
      result.sepsets[row][col].erase(
        std::unique(result.sepsets[row][col].begin(),
                    result.sepsets[row][col].end()),
        result.sepsets[row][col].end());
    }
  }

  result.pmax.assign(static_cast<std::size_t>(p) * p, 0.0);
  result.residual_cache_enabled = false;
  result.residual_cache_requests = 0;
  result.residual_cache_hits = 0;
  result.residual_cache_misses = 0;
  result.residual_cache_computations = 0;
  result.residual_cache_stored_vectors = 0;
  result.residual_cache_stored_values = 0;
  result.residual_backend = "";
  result.residual_backend_params = "";
  return result;
}

OrientationOptions make_orientation_options(double alpha,
                                            double index,
                                            bool legacy_index,
                                            bool residual_cache,
                                            const std::string& residual_backend,
                                            Rcpp::List fastspline_params,
                                            bool orient_collider,
                                            bool solve_confl,
                                            Rcpp::LogicalVector rules,
                                            const std::string& ci_method = "dcc.gamma",
                                            Rcpp::List hsic_params = Rcpp::List::create(),
                                            Rcpp::List permutation_params = Rcpp::List::create(),
                                            bool ci_diagnostics = true) {
  if (rules.size() != 3) {
    Rcpp::stop("rules must have length 3");
  }
  const FastSplineParams parsed_params = parse_fastspline_params(fastspline_params);
  make_residual_backend_config(residual_backend, parsed_params);
  OrientationOptions options = default_orientation_options();
  options.alpha = alpha;
  options.index = index;
  options.legacy_index = legacy_index;
  options.residual_cache_enabled = residual_cache;
  options.residual_backend_name = residual_backend;
  options.fastspline_params = parsed_params;
  options.orient_collider = orient_collider;
  options.solve_confl = solve_confl;
  if (rules[0] == NA_LOGICAL || rules[1] == NA_LOGICAL ||
      rules[2] == NA_LOGICAL) {
    Rcpp::stop("rules must not contain NA");
  }
  options.rule1 = rules[0] == TRUE;
  options.rule2 = rules[1] == TRUE;
  options.rule3 = rules[2] == TRUE;
  options.ci_method = ci_method;
  options.hsic_options = parse_hsic_options(hsic_params, permutation_params);
  options.ci_diagnostics_enabled = ci_diagnostics;
  return options;
}

bool finite_vector_values(const std::vector<double>& values) {
  for (double value : values) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

bool penalty_symmetric(const std::vector<double>& P, int p) {
  for (int i = 0; i < p; ++i) {
    for (int j = 0; j < p; ++j) {
      const double a = P[static_cast<std::size_t>(i) * p + j];
      const double b = P[static_cast<std::size_t>(j) * p + i];
      if (std::abs(a - b) > 1e-10) return false;
    }
  }
  return true;
}

Rcpp::List basis_design_stats(const FastSplineDesign& design,
                              bool check_row_sums) {
  bool row_sums_close = true;
  if (check_row_sums && design.p > 1) {
    for (int row = 0; row < design.n; ++row) {
      double total = 0.0;
      for (int col = 1; col < design.p; ++col) {
        total += design.X[static_cast<std::size_t>(row) * design.p + col];
      }
      if (std::abs(total - 1.0) > 1e-8) {
        row_sums_close = false;
        break;
      }
    }
  }
  return Rcpp::List::create(
    Rcpp::Named("nrow") = design.n,
    Rcpp::Named("ncol") = design.p,
    Rcpp::Named("row_sums_close_to_one") = row_sums_close,
    Rcpp::Named("finite") = finite_vector_values(design.X) && finite_vector_values(design.P),
    Rcpp::Named("penalty_dim") = design.p,
    Rcpp::Named("penalty_symmetric") = penalty_symmetric(design.P, design.p)
  );
}

bool non_intercept_cols_all_zero_or_constant(const FastSplineDesign& design) {
  for (int col = 1; col < design.p; ++col) {
    double first = design.X[col];
    bool all_zero = std::abs(first) < 1e-14;
    bool all_constant = true;
    for (int row = 0; row < design.n; ++row) {
      const double value = design.X[static_cast<std::size_t>(row) * design.p + col];
      if (std::abs(value) >= 1e-14) all_zero = false;
      if (std::abs(value - first) > 1e-14) all_constant = false;
    }
    if (!all_zero && !all_constant) return false;
  }
  return true;
}

Rcpp::NumericMatrix cbind_columns(const std::vector<double>& a) {
  Rcpp::NumericMatrix out(a.size(), 1);
  for (int i = 0; i < static_cast<int>(a.size()); ++i) out(i, 0) = a[i];
  return out;
}

Rcpp::NumericMatrix cbind_columns(const std::vector<double>& a,
                                  const std::vector<double>& b) {
  Rcpp::NumericMatrix out(a.size(), 2);
  for (int i = 0; i < static_cast<int>(a.size()); ++i) {
    out(i, 0) = a[i];
    out(i, 1) = b[i];
  }
  return out;
}

Rcpp::NumericMatrix cbind_columns(const std::vector<double>& a,
                                  const std::vector<double>& b,
                                  const std::vector<double>& c) {
  Rcpp::NumericMatrix out(a.size(), 3);
  for (int i = 0; i < static_cast<int>(a.size()); ++i) {
    out(i, 0) = a[i];
    out(i, 1) = b[i];
    out(i, 2) = c[i];
  }
  return out;
}

double rss_from_residuals(const std::vector<double>& residuals) {
  double rss = 0.0;
  for (double value : residuals) rss += value * value;
  return rss;
}

double mean_from_residuals(const std::vector<double>& residuals) {
  double total = 0.0;
  for (double value : residuals) total += value;
  return total / static_cast<double>(residuals.size());
}

Rcpp::List fit_case_to_list(const FastSplineFit& fit,
                            const std::vector<double>& linear_residuals) {
  return Rcpp::List::create(
    Rcpp::Named("fastspline_rss") = fit.rss,
    Rcpp::Named("linear_rss") = rss_from_residuals(linear_residuals),
    Rcpp::Named("residual_mean") = mean_from_residuals(fit.residuals),
    Rcpp::Named("selected_lambda") = fit.selected_lambda,
    Rcpp::Named("edf") = fit.edf,
    Rcpp::Named("design_cols") = fit.design_cols
  );
}

Rcpp::List spectral_score_batch_impl(
    Rcpp::NumericMatrix eigenvectors,
    Rcpp::NumericMatrix inv_chol,
    Rcpp::NumericVector eigenvalues,
    Rcpp::NumericMatrix y,
    Rcpp::NumericMatrix Xty_null,
    Rcpp::NumericVector sp_grid,
    double tol) {
  const int n = y.nrow();
  const int q = y.ncol();
  const int rank = eigenvalues.size();
  const int grid_size = sp_grid.size();
  if (inv_chol.nrow() != rank || inv_chol.ncol() != rank ||
      eigenvectors.nrow() != rank || eigenvectors.ncol() != rank ||
      Xty_null.nrow() != rank || Xty_null.ncol() != q) {
    Rcpp::stop("spectral score batch dimension mismatch");
  }

  Rcpp::NumericMatrix z(rank, q);
  Rcpp::NumericVector y_sq(q);
  for (int target = 0; target < q; ++target) {
    double y_total = 0.0;
    for (int row = 0; row < n; ++row) {
      const double value = y(row, target);
      y_total += value * value;
    }
    y_sq[target] = y_total;

    std::vector<double> tmp(rank, 0.0);
    for (int col = 0; col < rank; ++col) {
      double value = 0.0;
      for (int row = 0; row < rank; ++row) {
        value += inv_chol(row, col) * Xty_null(row, target);
      }
      tmp[static_cast<std::size_t>(col)] = value;
    }
    for (int component = 0; component < rank; ++component) {
      double value = 0.0;
      for (int row = 0; row < rank; ++row) {
        value += eigenvectors(row, component) * tmp[static_cast<std::size_t>(row)];
      }
      z(component, target) = value;
    }
  }

  Rcpp::NumericMatrix rss(q, grid_size);
  Rcpp::NumericMatrix gcv(q, grid_size);
  Rcpp::NumericVector edf(grid_size);
  for (int grid = 0; grid < grid_size; ++grid) {
    const double sp = sp_grid[grid];
    double edf_value = 0.0;
    std::vector<double> h(rank);
    for (int component = 0; component < rank; ++component) {
      const double shrinkage =
        1.0 / (1.0 + sp * static_cast<double>(eigenvalues[component]));
      h[static_cast<std::size_t>(component)] = shrinkage;
      edf_value += shrinkage;
    }
    edf[grid] = edf_value;
    const double denom = static_cast<double>(n) - edf_value;
    for (int target = 0; target < q; ++target) {
      double linear = 0.0;
      double quadratic = 0.0;
      for (int component = 0; component < rank; ++component) {
        const double z_value = z(component, target);
        const double z_sq = z_value * z_value;
        const double h_value = h[static_cast<std::size_t>(component)];
        linear += h_value * z_sq;
        quadratic += h_value * h_value * z_sq;
      }
      double rss_value = y_sq[target] - 2.0 * linear + quadratic;
      if (rss_value < 0.0) rss_value = 0.0;
      rss(target, grid) = rss_value;
      if (std::isfinite(edf_value) && denom > tol) {
        gcv(target, grid) = static_cast<double>(n) * rss_value / (denom * denom);
      } else {
        gcv(target, grid) = R_PosInf;
      }
    }
  }

  return Rcpp::List::create(
    Rcpp::Named("rss") = rss,
    Rcpp::Named("edf") = edf,
    Rcpp::Named("gcv") = gcv,
    Rcpp::Named("z") = z,
    Rcpp::Named("y_sq") = y_sq
  );
}

double get_named_double(Rcpp::List values, const char* name, double fallback) {
  if (!values.containsElementNamed(name)) return fallback;
  return Rcpp::as<double>(values[name]);
}

int get_named_int(Rcpp::List values, const char* name, int fallback) {
  if (!values.containsElementNamed(name)) return fallback;
  return Rcpp::as<int>(values[name]);
}

std::string get_named_string(Rcpp::List values, const char* name,
                             const std::string& fallback) {
  if (!values.containsElementNamed(name)) return fallback;
  return Rcpp::as<std::string>(values[name]);
}

bool get_named_bool(Rcpp::List values, const char* name, bool fallback) {
  if (!values.containsElementNamed(name)) return fallback;
  return Rcpp::as<bool>(values[name]);
}

FastSplineParams parse_fastspline_params(Rcpp::List values) {
  FastSplineParams params = default_fastspline_params();
  params.degree = get_named_int(values, "degree", params.degree);
  params.knots = get_named_int(values, "knots", params.knots);
  params.lambda_min = get_named_double(values, "lambda_min", params.lambda_min);
  params.lambda_max = get_named_double(values, "lambda_max", params.lambda_max);
  params.lambda_count = get_named_int(values, "lambda_count", params.lambda_count);
  params.ridge = get_named_double(values, "ridge", params.ridge);
  params.mode = get_named_string(values, "mode", params.mode);
  return params;
}

HsicOptions parse_hsic_options(Rcpp::List hsic_params,
                               Rcpp::List permutation_params) {
  HsicOptions options = default_hsic_options();
  options.sig = get_named_double(hsic_params, "sig", options.sig);
  options.cuda_max_n = get_named_int(hsic_params, "cuda_max_n",
                                     options.cuda_max_n);
  options.cuda_max_batch_pairs =
    get_named_int(hsic_params, "cuda_max_batch_pairs",
                  options.cuda_max_batch_pairs);
  options.cuda_memory_fallback =
    get_named_bool(hsic_params, "cuda_memory_fallback",
                   options.cuda_memory_fallback);
  options.replicates = get_named_int(permutation_params, "replicates",
                                     options.replicates);
  options.include_observed = get_named_bool(permutation_params,
                                            "include_observed",
                                            options.include_observed);
  if (permutation_params.containsElementNamed("seed")) {
    SEXP seed = permutation_params["seed"];
    if (!Rf_isNull(seed)) {
      options.has_seed = true;
      options.seed = static_cast<unsigned int>(Rcpp::as<int>(seed));
    }
  }
  return options;
}

void apply_ci_options(SkeletonOptions* options,
                      const std::string& ci_method,
                      Rcpp::List hsic_params,
                      Rcpp::List permutation_params,
                      bool ci_diagnostics) {
  options->ci_method = ci_method.empty() ? "dcc.gamma" : ci_method;
  options->hsic_options = parse_hsic_options(hsic_params, permutation_params);
  options->ci_diagnostics_enabled = ci_diagnostics;
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List mgcv_extract_gpu_spectral_score_batch_export(
    Rcpp::NumericMatrix eigenvectors,
    Rcpp::NumericMatrix inv_chol,
    Rcpp::NumericVector eigenvalues,
    Rcpp::NumericMatrix y,
    Rcpp::NumericMatrix Xty_null,
    Rcpp::NumericVector sp_grid,
    double tol) {
  return spectral_score_batch_impl(eigenvectors, inv_chol, eigenvalues, y,
                                   Xty_null, sp_grid, tol);
}

// [[Rcpp::export]]
Rcpp::List kpc_tprs_residual_cpp_setup_export(Rcpp::NumericMatrix S,
                                              int k = 0,
                                              double tol = 1.490116e-8) {
  if (S.nrow() <= 0) {
    Rcpp::stop("S must have at least one row");
  }
  if (S.ncol() < 1 || S.ncol() > 2) {
    Rcpp::stop("kpcTprsResidualCPP supports |S| = 1 or 2");
  }
  if (!finite_numeric_matrix(S)) {
    Rcpp::stop("S must be finite numeric");
  }
  if (!std::isfinite(tol) || tol <= 0.0) {
    Rcpp::stop("tol must be positive and finite");
  }

  Rcpp::NumericVector shift;
  Rcpp::NumericMatrix shifted = center_columns(S, &shift);
  Rcpp::NumericMatrix unique_rows = unique_rows_in_order(shifted);
  if (unique_rows.nrow() > 2000) {
    Rcpp::stop("kpcTprsResidualCPP requires unique conditioning locations <= 2000");
  }

  const int null_space_rank = S.ncol() + 1;
  const int k_def = S.ncol() == 1 ? 8 : 27;
  const int basis_rank = k > 0 ? k : null_space_rank + k_def;
  if (basis_rank < null_space_rank + 1) {
    Rcpp::stop("basis dimension must exceed null-space rank");
  }
  if (S.ncol() == 1) {
    return kpc_tprs_residual_cpp_setup_1d(
      S, shifted, shift, sorted_unique_rows_1d(shifted),
      basis_rank, null_space_rank, k_def, tol);
  }
  const int penalized_rank = basis_rank - null_space_rank;
  Rcpp::NumericMatrix knots = evenly_spaced_knots(unique_rows, penalized_rank);
  Rcpp::NumericMatrix polynomial = kpc_tprs_polynomial_null_space(shifted);
  Rcpp::NumericMatrix radial = kpc_tprs_radial_basis(shifted, knots);
  Rcpp::NumericMatrix X = cbind_numeric_matrices(polynomial, radial);
  Rcpp::NumericMatrix penalty = kpc_tprs_penalty_matrix(knots, polynomial.ncol());
  Rcpp::NumericMatrix constraint = kpc_tprs_centering_constraint(X);
  Rcpp::NumericMatrix Z = constraint_null_space_cpp(constraint, X.ncol(), tol);
  arma::mat Xa = rcpp_matrix_to_arma(X);
  arma::mat Pa = rcpp_matrix_to_arma(penalty);
  arma::mat Za = rcpp_matrix_to_arma(Z);
  Rcpp::NumericMatrix X_absorbed = arma_matrix_to_rcpp(Xa * Za);
  Rcpp::NumericMatrix penalty_absorbed = arma_matrix_to_rcpp(Za.t() * Pa * Za);
  Rcpp::NumericMatrix radial_kernel = kpc_tprs_radial_basis(knots, knots);

  return Rcpp::List::create(
    Rcpp::Named("backend_family") = "kpcTprsResidualCPP",
    Rcpp::Named("schema_version") = "setup-shadow-v1",
    Rcpp::Named("X") = X,
    Rcpp::Named("penalty") = penalty,
    Rcpp::Named("constraint") = constraint,
    Rcpp::Named("raw") = Rcpp::List::create(
      Rcpp::Named("shift") = shift,
      Rcpp::Named("shifted_covariates") = shifted,
      Rcpp::Named("unique_locations") = unique_rows,
      Rcpp::Named("radial_kernel_block") = radial_kernel,
      Rcpp::Named("radial") = radial,
      Rcpp::Named("polynomial") = polynomial,
      Rcpp::Named("penalty") = penalty,
      Rcpp::Named("constraint") = constraint
    ),
    Rcpp::Named("absorbed") = Rcpp::List::create(
      Rcpp::Named("Z") = Z,
      Rcpp::Named("X") = X_absorbed,
      Rcpp::Named("penalty") = penalty_absorbed,
      Rcpp::Named("effective_rank") = X_absorbed.ncol(),
      Rcpp::Named("null_space_rank") = null_space_rank
    ),
    Rcpp::Named("knots") = knots,
    Rcpp::Named("unique_rows") = unique_rows,
    Rcpp::Named("basis_rank") = basis_rank,
    Rcpp::Named("null_space_rank") = null_space_rank,
    Rcpp::Named("penalized_rank") = penalized_rank,
    Rcpp::Named("effective_rank") = X_absorbed.ncol(),
    Rcpp::Named("k_def") = k_def,
    Rcpp::Named("k") = knots.nrow(),
    Rcpp::Named("radial_basis") = S.ncol() == 1 ? "r^3" : "r^2 log(r)",
    Rcpp::Named("polynomial_basis") =
      S.ncol() == 1 ? "1 + s1" : "1 + s1 + s2",
    Rcpp::Named("smooth_geometry") = "joint-isotropic",
    Rcpp::Named("tol") = tol
  );
}

// [[Rcpp::export]]
double fast_dcov_exact_cpp_export(Rcpp::NumericVector x,
                                  Rcpp::NumericVector y,
                                  double index = 1.0,
                                  bool legacy_index = true) {
  return dcov_exact_pvalue(numeric_vector_to_std(x), numeric_vector_to_std(y),
                           index, legacy_index);
}

// [[Rcpp::export]]
Rcpp::List fast_hsic_gamma_cpp_export(Rcpp::NumericVector x,
                                      Rcpp::NumericVector y,
                                      double sig = 1.0) {
  HsicOptions options = default_hsic_options();
  options.sig = sig;
  return hsic_result_to_list(
    hsic_gamma_cpu(numeric_vector_to_std(x), numeric_vector_to_std(y), options)
  );
}

// [[Rcpp::export]]
Rcpp::List fast_hsic_perm_cpp_export(Rcpp::NumericVector x,
                                     Rcpp::NumericVector y,
                                     double sig = 1.0,
                                     int replicates = 100,
                                     Rcpp::Nullable<int> seed = R_NilValue,
                                     bool include_observed = true) {
  HsicOptions options = default_hsic_options();
  options.sig = sig;
  options.replicates = replicates;
  options.include_observed = include_observed;
  options.return_replicates = true;
  options.has_seed = seed.isNotNull();
  if (options.has_seed) {
    const int parsed_seed = Rcpp::as<int>(seed);
    options.seed = static_cast<unsigned int>(parsed_seed);
  }
  Rcpp::RNGScope rng_scope;
  return hsic_result_to_list(
    hsic_permutation_cpu(numeric_vector_to_std(x), numeric_vector_to_std(y),
                         options)
  );
}

// [[Rcpp::export]]
Rcpp::List fast_skeleton_cpp_export(Rcpp::NumericMatrix data,
                                    double alpha,
                                    int max_conditioning_size,
                                    double index = 1.0,
                                    bool legacy_index = true) {
  SkeletonOptions options;
  options.alpha = alpha;
  options.max_conditioning_size = max_conditioning_size;
  options.na_delete = true;
  options.stable = true;
  options.index = index;
  options.legacy_index = legacy_index;
  options.residual_cache_enabled = false;
  options.residual_backend_name = "linear";
  options.fastspline_params = default_fastspline_params();
  apply_ci_options(&options, "dcc.gamma", Rcpp::List::create(),
                   Rcpp::List::create(), true);

  const SkeletonResult result = run_skeleton_exact(data, options);
  const int p = data.ncol();
  return skeleton_result_to_list(result, p, "cpu");
}

// [[Rcpp::export]]
Rcpp::List fast_skeleton_cpp_cached_export(Rcpp::NumericMatrix data,
                                           double alpha,
                                           int max_conditioning_size,
                                           double index = 1.0,
                                           bool legacy_index = true,
                                           bool residual_cache = true) {
  SkeletonOptions options;
  options.alpha = alpha;
  options.max_conditioning_size = max_conditioning_size;
  options.na_delete = true;
  options.stable = true;
  options.index = index;
  options.legacy_index = legacy_index;
  options.residual_cache_enabled = residual_cache;
  options.residual_backend_name = "linear";
  options.fastspline_params = default_fastspline_params();
  apply_ci_options(&options, "dcc.gamma", Rcpp::List::create(),
                   Rcpp::List::create(), true);

  const SkeletonResult result = run_skeleton_exact(data, options);
  return skeleton_result_to_list(result, data.ncol(), "cpu");
}

// [[Rcpp::export]]
Rcpp::List fast_skeleton_cpp_backend_export(Rcpp::NumericMatrix data,
                                            double alpha,
                                            int max_conditioning_size,
                                            double index,
                                            bool legacy_index,
                                            bool residual_cache,
                                            std::string residual_backend,
                                            Rcpp::List fastspline_params,
                                            std::string ci_method,
                                            Rcpp::List hsic_params,
                                            Rcpp::List permutation_params,
                                            bool ci_diagnostics) {
  const FastSplineParams parsed_params = parse_fastspline_params(fastspline_params);
  make_residual_backend_config(residual_backend, parsed_params);

  SkeletonOptions options;
  options.alpha = alpha;
  options.max_conditioning_size = max_conditioning_size;
  options.na_delete = true;
  options.stable = true;
  options.index = index;
  options.legacy_index = legacy_index;
  options.residual_cache_enabled = residual_cache;
  options.residual_backend_name = residual_backend;
  options.fastspline_params = parsed_params;
  apply_ci_options(&options, ci_method, hsic_params, permutation_params,
                   ci_diagnostics);

  const SkeletonResult result = run_skeleton_exact(data, options);
  return skeleton_result_to_list(result, data.ncol(), "cpu");
}

// [[Rcpp::export]]
Rcpp::List fast_orient_wanpdag_cpp_export(Rcpp::NumericMatrix data,
                                          Rcpp::LogicalMatrix adjacency,
                                          Rcpp::List sepsets,
                                          double alpha,
                                          double index,
                                          bool legacy_index,
                                          bool residual_cache,
                                          std::string residual_backend,
                                          Rcpp::List fastspline_params,
                                          bool orient_collider,
                                          bool solve_confl,
                                          Rcpp::LogicalVector rules,
                                          std::string ci_method,
                                          Rcpp::List hsic_params,
                                          Rcpp::List permutation_params,
                                          bool ci_diagnostics) {
  SkeletonResult skeleton = skeleton_result_from_R(adjacency, sepsets);
  const OrientationOptions orientation_options = make_orientation_options(
    alpha, index, legacy_index, residual_cache, residual_backend,
    fastspline_params, orient_collider, solve_confl, rules, ci_method,
    hsic_params, permutation_params, ci_diagnostics);
  const OrientationResult orientation =
    orient_wanpdag_native(data, skeleton, orientation_options);
  return orientation_result_to_list(orientation);
}

// [[Rcpp::export]]
Rcpp::List fast_kpc_wanpdag_cpp_export(Rcpp::NumericMatrix data,
                                       double alpha,
                                       int max_conditioning_size,
                                       double index,
                                       bool legacy_index,
                                       bool residual_cache,
                                       std::string residual_backend,
                                       Rcpp::List fastspline_params,
                                       bool orient_collider,
                                       bool solve_confl,
                                       Rcpp::LogicalVector rules,
                                       std::string ci_method,
                                       Rcpp::List hsic_params,
                                       Rcpp::List permutation_params,
                                       bool ci_diagnostics) {
  const FastSplineParams parsed_params = parse_fastspline_params(fastspline_params);
  make_residual_backend_config(residual_backend, parsed_params);

  SkeletonOptions skeleton_options;
  skeleton_options.alpha = alpha;
  skeleton_options.max_conditioning_size = max_conditioning_size;
  skeleton_options.na_delete = true;
  skeleton_options.stable = true;
  skeleton_options.index = index;
  skeleton_options.legacy_index = legacy_index;
  skeleton_options.residual_cache_enabled = residual_cache;
  skeleton_options.residual_backend_name = residual_backend;
  skeleton_options.fastspline_params = parsed_params;
  apply_ci_options(&skeleton_options, ci_method, hsic_params,
                   permutation_params, ci_diagnostics);

  const SkeletonResult skeleton = run_skeleton_exact(data, skeleton_options);
  const OrientationOptions orientation_options = make_orientation_options(
    alpha, index, legacy_index, residual_cache, residual_backend,
    fastspline_params, orient_collider, solve_confl, rules, ci_method,
    hsic_params, permutation_params, ci_diagnostics);
  const OrientationResult orientation =
    orient_wanpdag_native(data, skeleton, orientation_options);
  return Rcpp::List::create(
    Rcpp::Named("skeleton") = skeleton_result_to_list(skeleton, data.ncol(), "cpu"),
    Rcpp::Named("orientation") = orientation_result_to_list(orientation)
  );
}

// [[Rcpp::export]]
Rcpp::List fast_residual_cache_selftest_export(Rcpp::NumericMatrix data) {
  std::vector<int> cond_a;
  cond_a.push_back(2);
  cond_a.push_back(1);
  std::vector<int> cond_b;
  cond_b.push_back(1);
  cond_b.push_back(2);

  const ResidualBackendDescriptor descriptor = linear_residual_backend_descriptor();
  const ResidualCacheKey key_a = make_residual_cache_key(
    0, cond_a, data.nrow(), data.ncol(), descriptor.name, descriptor.params);
  const ResidualCacheKey key_b = make_residual_cache_key(
    0, cond_b, data.nrow(), data.ncol(), descriptor.name, descriptor.params);
  const ResidualCacheKey key_target = make_residual_cache_key(
    1, cond_b, data.nrow(), data.ncol(), descriptor.name, descriptor.params);
  const ResidualCacheKey key_params = make_residual_cache_key(
    0, cond_b, data.nrow(), data.ncol(), descriptor.name, "different=params");

  ResidualCache enabled(linear_residual_cache_options(true));
  const std::vector<double>& cached_first = enabled.get(data, 0, cond_a);
  std::vector<double> cached_copy = cached_first;
  const std::vector<double>& cached_second = enabled.get(data, 0, cond_b);
  const ResidualCacheStats enabled_stats = enabled.stats();

  ResidualCache disabled(linear_residual_cache_options(false));
  disabled.get(data, 0, cond_a);
  disabled.get(data, 0, cond_b);
  const ResidualCacheStats disabled_stats = disabled.stats();

  const std::vector<double> direct = compute_linear_residuals(data, 0, cond_b);
  double max_abs_diff = 0.0;
  for (int i = 0; i < static_cast<int>(direct.size()); ++i) {
    max_abs_diff = std::max(max_abs_diff, std::abs(cached_second[i] - direct[i]));
    max_abs_diff = std::max(max_abs_diff, std::abs(cached_copy[i] - direct[i]));
  }

  return Rcpp::List::create(
    Rcpp::Named("key_order_invariant") = !(key_a < key_b) && !(key_b < key_a),
    Rcpp::Named("target_distinct") = (key_a < key_target) || (key_target < key_a),
    Rcpp::Named("params_distinct") = (key_a < key_params) || (key_params < key_a),
    Rcpp::Named("enabled_stats") = residual_stats_to_list(enabled_stats),
    Rcpp::Named("disabled_stats") = residual_stats_to_list(disabled_stats),
    Rcpp::Named("max_abs_residual_diff") = max_abs_diff
  );
}

// [[Rcpp::export]]
Rcpp::CharacterVector list_residual_backends_export() {
  const std::vector<std::string> names = list_residual_backend_names();
  return Rcpp::CharacterVector(names.begin(), names.end());
}

// [[Rcpp::export]]
Rcpp::List fast_residual_backend_selftest_export() {
  const int n = 110;
  Rcpp::NumericMatrix data(n, 3);
  for (int i = 0; i < n; ++i) {
    const double z = -2.5 + 5.0 * static_cast<double>(i) / (n - 1);
    data(i, 0) = std::sin(z) + 0.03 * std::cos(11.0 * z);
    data(i, 1) = z;
    data(i, 2) = std::cos(0.13 * i);
  }

  std::vector<int> cond;
  cond.push_back(1);

  const FastSplineParams defaults = default_fastspline_params();
  const ResidualBackendConfig linear =
    make_residual_backend_config("linear", defaults);
  FastSplineParams alt_params = defaults;
  alt_params.knots = defaults.knots + 2;
  const ResidualBackendConfig fastspline =
    make_residual_backend_config("fastSpline", defaults);
  const ResidualBackendConfig fastspline_alt =
    make_residual_backend_config("fastSpline", alt_params);

  const std::vector<double> direct_linear = compute_linear_residuals(data, 0, cond);
  const std::vector<double> registry_linear =
    compute_residuals_with_backend(data, 0, cond, linear);
  const std::vector<double> registry_fastspline =
    compute_residuals_with_backend(data, 0, cond, fastspline);

  double max_linear_diff = 0.0;
  double max_backend_diff = 0.0;
  for (int i = 0; i < n; ++i) {
    max_linear_diff = std::max(max_linear_diff,
                               std::abs(direct_linear[i] - registry_linear[i]));
    max_backend_diff = std::max(max_backend_diff,
                                std::abs(registry_fastspline[i] - registry_linear[i]));
  }

  const ResidualCacheKey key_linear = make_residual_cache_key(
    0, cond, data.nrow(), data.ncol(), linear.name, linear.params);
  const ResidualCacheKey key_fastspline = make_residual_cache_key(
    0, cond, data.nrow(), data.ncol(), fastspline.name, fastspline.params);
  const ResidualCacheKey key_fastspline_alt = make_residual_cache_key(
    0, cond, data.nrow(), data.ncol(), fastspline_alt.name, fastspline_alt.params);

  ResidualCache cache(backend_residual_cache_options("fastSpline", defaults, true));
  cache.get(data, 0, cond);
  cache.get(data, 0, cond);
  const ResidualCacheStats stats = cache.stats();

  return Rcpp::List::create(
    Rcpp::Named("linear_matches_direct") = max_linear_diff < 1e-12,
    Rcpp::Named("fastspline_differs_from_linear") = max_backend_diff > 1e-4,
    Rcpp::Named("key_separates_backend") =
      (key_linear < key_fastspline) || (key_fastspline < key_linear),
    Rcpp::Named("key_separates_fastspline_params") =
      (key_fastspline < key_fastspline_alt) || (key_fastspline_alt < key_fastspline),
    Rcpp::Named("fastspline_cache_stats") = residual_stats_to_list(stats)
  );
}

// [[Rcpp::export]]
void fast_residual_backend_unknown_selftest_export() {
  make_residual_backend_config("not-a-backend", default_fastspline_params());
}

// [[Rcpp::export]]
Rcpp::List fastspline_basis_selftest_export(Rcpp::NumericMatrix data) {
  const FastSplineParams params = default_fastspline_params();
  std::vector<int> one_cols;
  one_cols.push_back(0);
  std::vector<int> two_cols;
  two_cols.push_back(0);
  two_cols.push_back(1);
  std::vector<int> additive_cols;
  additive_cols.push_back(0);
  additive_cols.push_back(1);
  additive_cols.push_back(2);
  std::vector<int> constant_cols;
  constant_cols.push_back(2);

  const FastSplineDesign one_d = make_fastspline_design(data, one_cols, params);
  const FastSplineDesign two_d = make_fastspline_design(data, two_cols, params);
  const FastSplineDesign additive = make_fastspline_design(data, additive_cols, params);
  const FastSplineDesign constant = make_fastspline_design(data, constant_cols, params);

  return Rcpp::List::create(
    Rcpp::Named("one_d") = basis_design_stats(one_d, true),
    Rcpp::Named("two_d") = basis_design_stats(two_d, false),
    Rcpp::Named("additive") = basis_design_stats(additive, false),
    Rcpp::Named("constant") = Rcpp::List::create(
      Rcpp::Named("finite") = finite_vector_values(constant.X) &&
        finite_vector_values(constant.P),
      Rcpp::Named("non_intercept_cols_all_zero_or_constant") =
        non_intercept_cols_all_zero_or_constant(constant)
    )
  );
}

// [[Rcpp::export]]
Rcpp::List fastspline_solver_selftest_export() {
  const FastSplineParams params = default_fastspline_params();
  const int n = 140;

  Rcpp::NumericMatrix one_data(n, 2);
  for (int i = 0; i < n; ++i) {
    const double t = -3.0 + 6.0 * static_cast<double>(i) / (n - 1);
    one_data(i, 0) = std::sin(t) + 0.04 * std::cos(17.0 * t);
    one_data(i, 1) = t;
  }
  std::vector<int> one_cond;
  one_cond.push_back(1);
  const FastSplineFit one_fit = fit_fastspline_residuals(one_data, 0, one_cond, params);
  const std::vector<double> one_linear = compute_linear_residuals(one_data, 0, one_cond);

  Rcpp::NumericMatrix two_data(n, 3);
  for (int i = 0; i < n; ++i) {
    const double t1 = -2.5 + 5.0 * static_cast<double>(i) / (n - 1);
    const double t2 = std::cos(0.17 * i);
    two_data(i, 0) = std::sin(t1) + std::cos(t2) + 0.04 * std::sin(13.0 * t1);
    two_data(i, 1) = t1;
    two_data(i, 2) = t2;
  }
  std::vector<int> two_cond;
  two_cond.push_back(1);
  two_cond.push_back(2);
  const FastSplineFit two_fit = fit_fastspline_residuals(two_data, 0, two_cond, params);
  const std::vector<double> two_linear = compute_linear_residuals(two_data, 0, two_cond);

  Rcpp::NumericMatrix three_data(n, 4);
  for (int i = 0; i < n; ++i) {
    const double t1 = -2.0 + 4.0 * static_cast<double>(i) / (n - 1);
    const double t2 = std::sin(0.11 * i);
    const double t3 = -1.0 + 2.0 * ((i * 37) % n) / static_cast<double>(n - 1);
    three_data(i, 0) = std::sin(t1) + std::cos(t2) + 0.5 * t3 * t3 +
      0.03 * std::cos(19.0 * t2);
    three_data(i, 1) = t1;
    three_data(i, 2) = t2;
    three_data(i, 3) = t3;
  }
  std::vector<int> three_cond;
  three_cond.push_back(1);
  three_cond.push_back(2);
  three_cond.push_back(3);
  const FastSplineFit three_fit = fit_fastspline_residuals(three_data, 0, three_cond, params);
  const std::vector<double> three_linear = compute_linear_residuals(three_data, 0, three_cond);

  Rcpp::NumericMatrix constant_data(n, 2);
  for (int i = 0; i < n; ++i) {
    const double t = static_cast<double>(i) / (n - 1);
    constant_data(i, 0) = std::sin(t);
    constant_data(i, 1) = 3.0;
  }
  const FastSplineFit constant_fit =
    fit_fastspline_residuals(constant_data, 0, one_cond, params);

  return Rcpp::List::create(
    Rcpp::Named("one_d") = fit_case_to_list(one_fit, one_linear),
    Rcpp::Named("two_d") = fit_case_to_list(two_fit, two_linear),
    Rcpp::Named("three_d") = fit_case_to_list(three_fit, three_linear),
    Rcpp::Named("constant") = Rcpp::List::create(
      Rcpp::Named("finite_residuals") = finite_vector_values(constant_fit.residuals),
      Rcpp::Named("residual_mean") = mean_from_residuals(constant_fit.residuals),
      Rcpp::Named("selected_lambda") = constant_fit.selected_lambda
    )
  );
}

// [[Rcpp::export]]
Rcpp::List fastspline_residual_export(Rcpp::NumericVector y,
                                      Rcpp::NumericMatrix S,
                                      Rcpp::List fastspline_params) {
  if (y.size() != S.nrow()) {
    Rcpp::stop("y and S must have the same number of rows");
  }
  Rcpp::NumericMatrix data(y.size(), S.ncol() + 1);
  for (int row = 0; row < y.size(); ++row) {
    data(row, 0) = y[row];
    for (int col = 0; col < S.ncol(); ++col) data(row, col + 1) = S(row, col);
  }
  std::vector<int> cond;
  for (int col = 0; col < S.ncol(); ++col) cond.push_back(col + 1);
  const FastSplineParams params = parse_fastspline_params(fastspline_params);
  const FastSplineFit fit = fit_fastspline_residuals(data, 0, cond, params);
  return Rcpp::List::create(
    Rcpp::Named("residual") = Rcpp::NumericVector(fit.residuals.begin(), fit.residuals.end()),
    Rcpp::Named("fitted") = Rcpp::NumericVector(fit.fitted.begin(), fit.fitted.end()),
    Rcpp::Named("selected_lambda") = fit.selected_lambda,
    Rcpp::Named("gcv") = fit.gcv,
    Rcpp::Named("rss") = fit.rss,
    Rcpp::Named("edf") = fit.edf,
    Rcpp::Named("design_cols") = fit.design_cols,
    Rcpp::Named("ridge_attempts") = fit.ridge_attempts,
    Rcpp::Named("params") = serialize_fastspline_params(params)
  );
}

// [[Rcpp::export]]
Rcpp::List orientation_matrix_selftest_export() {
  bool empty_has_no_edges = false;
  bool undirected_roundtrip = false;
  bool directed_roundtrip = false;
  bool conflict_roundtrip = false;
  bool edge_predicates_correct = false;
  bool from_skeleton_symmetric = false;
  bool diff_counts_correct = false;
  bool invalid_indices_rejected = false;

  const int p = 4;
  std::vector<int> pdag(static_cast<std::size_t>(p) * p, FASTKPC_EDGE_NONE);
  empty_has_no_edges = !has_any_edge(pdag, p, 0, 1);

  set_undirected_edge(&pdag, p, 0, 1);
  undirected_roundtrip = has_undirected_edge(pdag, p, 0, 1) &&
    has_any_edge(pdag, p, 0, 1);

  set_directed_edge(&pdag, p, 1, 2);
  directed_roundtrip = has_directed_edge(pdag, p, 1, 2) &&
    !has_directed_edge(pdag, p, 2, 1);

  set_conflict_edge(&pdag, p, 2, 3);
  conflict_roundtrip = has_conflict_edge(pdag, p, 2, 3);

  edge_predicates_correct =
    has_undirected_edge(pdag, p, 0, 1) &&
    has_directed_edge(pdag, p, 1, 2) &&
    has_conflict_edge(pdag, p, 2, 3) &&
    !has_undirected_edge(pdag, p, 1, 2) &&
    !has_conflict_edge(pdag, p, 0, 1);

  std::vector<int> adjacency(static_cast<std::size_t>(p) * p, 0);
  adjacency[1] = 1;
  adjacency[p] = 1;
  adjacency[2 * p + 3] = 1;
  adjacency[3 * p + 2] = 1;
  const std::vector<int> from_skeleton = pdag_from_skeleton_adjacency(adjacency, p);
  from_skeleton_symmetric =
    has_undirected_edge(from_skeleton, p, 0, 1) &&
    has_undirected_edge(from_skeleton, p, 2, 3) &&
    !has_any_edge(from_skeleton, p, 0, 2);

  int directed_count = 0;
  int undirected_count = 0;
  int conflict_count = 0;
  for (int a = 0; a < p - 1; ++a) {
    for (int b = a + 1; b < p; ++b) {
      if (has_conflict_edge(pdag, p, a, b)) ++conflict_count;
      else if (has_undirected_edge(pdag, p, a, b)) ++undirected_count;
      else if (has_directed_edge(pdag, p, a, b) ||
               has_directed_edge(pdag, p, b, a)) ++directed_count;
    }
  }
  diff_counts_correct = directed_count == 1 && undirected_count == 1 &&
    conflict_count == 1;

  try {
    pdag_get(pdag, p, -1, 0);
  } catch (const std::exception& e) {
    invalid_indices_rejected =
      std::string(e.what()).find("pdag index out of range") != std::string::npos;
  }

  return Rcpp::List::create(
    Rcpp::Named("empty_has_no_edges") = empty_has_no_edges,
    Rcpp::Named("undirected_roundtrip") = undirected_roundtrip,
    Rcpp::Named("directed_roundtrip") = directed_roundtrip,
    Rcpp::Named("conflict_roundtrip") = conflict_roundtrip,
    Rcpp::Named("edge_predicates_correct") = edge_predicates_correct,
    Rcpp::Named("from_skeleton_symmetric") = from_skeleton_symmetric,
    Rcpp::Named("diff_counts_correct") = diff_counts_correct,
    Rcpp::Named("invalid_indices_rejected") = invalid_indices_rejected
  );
}

// [[Rcpp::export]]
Rcpp::List orientation_rules_selftest_export() {
  const int p = 5;
  std::vector<std::vector<std::vector<int> > > empty_sepsets(
    p, std::vector<std::vector<int> >(p));
  std::vector<OrientationEvent> events;
  std::vector<int> unf_vect;

  std::vector<int> collider(static_cast<std::size_t>(p) * p, FASTKPC_EDGE_NONE);
  set_undirected_edge(&collider, p, 0, 1);
  set_undirected_edge(&collider, p, 1, 2);
  const int collider_count =
    orient_colliders(&collider, p, empty_sepsets, false, unf_vect, &events);
  const bool collider_orients_unshielded_triple =
    collider_count > 0 &&
    has_directed_edge(collider, p, 0, 1) &&
    has_directed_edge(collider, p, 2, 1);

  std::vector<std::vector<std::vector<int> > > blocked_sepsets(
    p, std::vector<std::vector<int> >(p));
  blocked_sepsets[0][2].push_back(1);
  std::vector<int> blocked(static_cast<std::size_t>(p) * p, FASTKPC_EDGE_NONE);
  set_undirected_edge(&blocked, p, 0, 1);
  set_undirected_edge(&blocked, p, 1, 2);
  orient_colliders(&blocked, p, blocked_sepsets, false, unf_vect, &events);
  const bool collider_respects_sepset =
    has_undirected_edge(blocked, p, 0, 1) &&
    has_undirected_edge(blocked, p, 1, 2);

  std::vector<int> conflict(static_cast<std::size_t>(p) * p, FASTKPC_EDGE_NONE);
  set_directed_edge(&conflict, p, 1, 0);
  set_undirected_edge(&conflict, p, 1, 2);
  orient_colliders(&conflict, p, empty_sepsets, true, unf_vect, &events);
  const bool conflict_collider_marks_bidirected =
    has_conflict_edge(conflict, p, 0, 1) &&
    has_directed_edge(conflict, p, 2, 1);

  std::vector<int> immor(static_cast<std::size_t>(p) * p, FASTKPC_EDGE_NONE);
  set_directed_edge(&immor, p, 0, 3);
  set_undirected_edge(&immor, p, 0, 1);
  set_undirected_edge(&immor, p, 0, 2);
  set_undirected_edge(&immor, p, 1, 2);
  std::vector<int> clique_S;
  clique_S.push_back(1);
  clique_S.push_back(2);
  const bool check_immor_accepts_clique_S =
    check_immor(immor, p, 3, clique_S);
  set_no_edge(&immor, p, 1, 2);
  const bool check_immor_rejects_nonclique_S =
    !check_immor(immor, p, 3, clique_S);
  set_undirected_edge(&immor, p, 1, 2);
  set_no_edge(&immor, p, 0, 2);
  const bool check_immor_rejects_unconnected_parent =
    !check_immor(immor, p, 3, clique_S);

  std::vector<int> rule1_graph(static_cast<std::size_t>(p) * p,
                               FASTKPC_EDGE_NONE);
  set_directed_edge(&rule1_graph, p, 0, 1);
  set_undirected_edge(&rule1_graph, p, 1, 2);
  apply_rule1(&rule1_graph, p, false, unf_vect, &events);
  const bool rule1_orients_chain_tail =
    has_directed_edge(rule1_graph, p, 1, 2);

  std::vector<int> rule2_graph(static_cast<std::size_t>(p) * p,
                               FASTKPC_EDGE_NONE);
  set_undirected_edge(&rule2_graph, p, 0, 1);
  set_directed_edge(&rule2_graph, p, 0, 2);
  set_directed_edge(&rule2_graph, p, 2, 1);
  apply_rule2(&rule2_graph, p, false, &events);
  const bool rule2_orients_directed_chain =
    has_directed_edge(rule2_graph, p, 0, 1);

  std::vector<int> rule3_graph(static_cast<std::size_t>(p) * p,
                               FASTKPC_EDGE_NONE);
  set_undirected_edge(&rule3_graph, p, 0, 1);
  set_undirected_edge(&rule3_graph, p, 0, 2);
  set_undirected_edge(&rule3_graph, p, 0, 3);
  set_directed_edge(&rule3_graph, p, 2, 1);
  set_directed_edge(&rule3_graph, p, 3, 1);
  apply_rule3(&rule3_graph, p, false, unf_vect, &events);
  const bool rule3_orients_double_parent_pattern =
    has_directed_edge(rule3_graph, p, 0, 1);

  std::vector<int> fixed_point(static_cast<std::size_t>(p) * p,
                               FASTKPC_EDGE_NONE);
  set_directed_edge(&fixed_point, p, 0, 1);
  set_undirected_edge(&fixed_point, p, 1, 2);
  set_undirected_edge(&fixed_point, p, 2, 3);
  OrientationOptions options;
  options.alpha = 0.2;
  options.verbose = false;
  options.solve_confl = false;
  options.orient_collider = true;
  options.rule1 = true;
  options.rule2 = true;
  options.rule3 = true;
  options.residual_cache_enabled = true;
  options.residual_backend_name = "linear";
  options.fastspline_params = default_fastspline_params();
  options.index = 1.0;
  options.legacy_index = true;
  const RuleApplicationCounts counts =
    apply_rules_until_converged(&fixed_point, p, options, unf_vect, &events);
  const bool fixed_point_converges =
    counts.rule1 >= 2 &&
    has_directed_edge(fixed_point, p, 1, 2) &&
    has_directed_edge(fixed_point, p, 2, 3);

  std::vector<int> disabled(static_cast<std::size_t>(p) * p,
                            FASTKPC_EDGE_NONE);
  set_directed_edge(&disabled, p, 0, 1);
  set_undirected_edge(&disabled, p, 1, 2);
  const std::vector<int> disabled_before = disabled;
  options.rule1 = false;
  options.rule2 = false;
  options.rule3 = false;
  apply_rules_until_converged(&disabled, p, options, unf_vect, &events);
  const bool rules_disabled_no_change = disabled == disabled_before;

  return Rcpp::List::create(
    Rcpp::Named("collider_orients_unshielded_triple") =
      collider_orients_unshielded_triple,
    Rcpp::Named("collider_respects_sepset") = collider_respects_sepset,
    Rcpp::Named("conflict_collider_marks_bidirected") =
      conflict_collider_marks_bidirected,
    Rcpp::Named("check_immor_accepts_clique_S") =
      check_immor_accepts_clique_S,
    Rcpp::Named("check_immor_rejects_nonclique_S") =
      check_immor_rejects_nonclique_S,
    Rcpp::Named("check_immor_rejects_unconnected_parent") =
      check_immor_rejects_unconnected_parent,
    Rcpp::Named("rule1_orients_chain_tail") = rule1_orients_chain_tail,
    Rcpp::Named("rule2_orients_directed_chain") =
      rule2_orients_directed_chain,
    Rcpp::Named("rule3_orients_double_parent_pattern") =
      rule3_orients_double_parent_pattern,
    Rcpp::Named("fixed_point_converges") = fixed_point_converges,
    Rcpp::Named("rules_disabled_no_change") = rules_disabled_no_change
  );
}

// [[Rcpp::export]]
Rcpp::List regrvonps_native_selftest_export() {
  const int n = 160;
  const int p = 4;

  OrientationOptions options;
  options.alpha = 0.2;
  options.verbose = false;
  options.solve_confl = false;
  options.orient_collider = true;
  options.rule1 = true;
  options.rule2 = true;
  options.rule3 = true;
  options.residual_cache_enabled = true;
  options.residual_backend_name = "linear";
  options.fastspline_params = default_fastspline_params();
  options.index = 1.0;
  options.legacy_index = true;

  Rcpp::NumericMatrix dependent_data(n, p);
  for (int i = 0; i < n; ++i) {
    const double z = -3.0 + 6.0 * static_cast<double>(i) / (n - 1);
    const double w = std::sin(1.7 * z) + 0.03 * std::cos(11.0 * z);
    dependent_data(i, 0) = w + 0.35 * z;
    dependent_data(i, 1) = z;
    dependent_data(i, 2) = w;
    dependent_data(i, 3) = z * z;
  }

  std::vector<int> pdag(static_cast<std::size_t>(p) * p, FASTKPC_EDGE_NONE);
  set_directed_edge(&pdag, p, 2, 0);
  set_undirected_edge(&pdag, p, 0, 1);

  ResidualCache linear_cache(backend_residual_cache_options(
    "linear", default_fastspline_params(), true));
  std::vector<int> S;
  S.push_back(1);
  const RegrVonPsResult dependent_linear =
    regrvonps_native(dependent_data, pdag, p, 0, S, options, &linear_cache);
  const bool dependent_linear_rejects = dependent_linear.reject_count > 0;

  const bool parents_in_conditioning =
    std::find(dependent_linear.parents.begin(), dependent_linear.parents.end(), 2) !=
      dependent_linear.parents.end() &&
    std::find(dependent_linear.conditioning_set.begin(),
              dependent_linear.conditioning_set.end(), 1) !=
      dependent_linear.conditioning_set.end() &&
    std::find(dependent_linear.conditioning_set.begin(),
              dependent_linear.conditioning_set.end(), 2) !=
      dependent_linear.conditioning_set.end();

  const RegrVonPsResult repeated =
    regrvonps_native(dependent_data, pdag, p, 0, S, options, &linear_cache);
  const bool cache_hits_repeated =
    repeated.cache_hits_after > repeated.cache_hits_before;
  const bool pvalue_count_correct =
    static_cast<int>(dependent_linear.p_values.size()) == static_cast<int>(S.size());

  Rcpp::NumericMatrix smooth_data(n, p);
  for (int i = 0; i < n; ++i) {
    const double z = -3.141592653589793 + 6.283185307179586 *
      static_cast<double>(i) / (n - 1);
    const double noise = 0.03 * std::sin(37.0 * z);
    smooth_data(i, 0) = std::sin(z) + noise;
    smooth_data(i, 1) = z;
    smooth_data(i, 2) = std::cos(0.5 * z);
    smooth_data(i, 3) = z * z;
  }
  std::vector<int> smooth_pdag(static_cast<std::size_t>(p) * p,
                               FASTKPC_EDGE_NONE);
  set_undirected_edge(&smooth_pdag, p, 0, 1);
  options.residual_backend_name = "fastSpline";
  ResidualCache spline_cache(backend_residual_cache_options(
    "fastSpline", default_fastspline_params(), true));
  const RegrVonPsResult smooth_fastspline =
    regrvonps_native(smooth_data, smooth_pdag, p, 0, S, options, &spline_cache);
  const bool smooth_fastspline_accepts =
    smooth_fastspline.reject_count == 0 &&
    !smooth_fastspline.p_values.empty() &&
    smooth_fastspline.p_values[0] >= options.alpha;

  bool unknown_backend_error = false;
  try {
    ResidualCache bad_cache(backend_residual_cache_options(
      "unknownBackend", default_fastspline_params(), true));
    regrvonps_native(smooth_data, smooth_pdag, p, 0, S, options, &bad_cache);
  } catch (const std::exception& e) {
    unknown_backend_error =
      std::string(e.what()).find("Unknown residual backend") != std::string::npos;
  }

  std::vector<int> empty_S;
  const RegrVonPsResult empty =
    regrvonps_native(smooth_data, smooth_pdag, p, 0, empty_S, options,
                     &spline_cache);
  const bool empty_S_safe =
    empty.reject_count == 0 && empty.p_values.empty();

  return Rcpp::List::create(
    Rcpp::Named("dependent_linear_rejects") = dependent_linear_rejects,
    Rcpp::Named("smooth_fastspline_accepts") = smooth_fastspline_accepts,
    Rcpp::Named("parents_in_conditioning") = parents_in_conditioning,
    Rcpp::Named("cache_hits_repeated") = cache_hits_repeated,
    Rcpp::Named("pvalue_count_correct") = pvalue_count_correct,
    Rcpp::Named("unknown_backend_error") = unknown_backend_error,
    Rcpp::Named("empty_S_safe") = empty_S_safe
  );
}

// [[Rcpp::export]]
Rcpp::List wanpdag_engine_core_selftest_export() {
  const int n = 160;
  const int p = 4;
  Rcpp::NumericMatrix data(n, p);
  for (int i = 0; i < n; ++i) {
    const double z = -3.0 + 6.0 * static_cast<double>(i) / (n - 1);
    const double noise = 0.02 * std::sin(17.0 * z);
    data(i, 0) = z + noise;
    data(i, 1) = z;
    data(i, 2) = std::sin(z) + noise;
    data(i, 3) = std::cos(z);
  }

  SkeletonResult empty_skeleton;
  empty_skeleton.adjacency.assign(static_cast<std::size_t>(p) * p, 0);
  empty_skeleton.sepsets.assign(p, std::vector<std::vector<int> >(p));
  empty_skeleton.pmax.assign(static_cast<std::size_t>(p) * p, 0.0);
  empty_skeleton.residual_cache_enabled = true;
  empty_skeleton.residual_backend = "linear";
  empty_skeleton.residual_backend_params = "";

  OrientationOptions options = default_orientation_options();
  options.residual_backend_name = "linear";
  options.alpha = 0.2;
  const OrientationResult empty_result =
    orient_wanpdag_native(data, empty_skeleton, options);
  bool empty_pdag_all_zero = true;
  for (int value : empty_result.pdag) {
    if (value != FASTKPC_EDGE_NONE) empty_pdag_all_zero = false;
  }
  const bool empty_skeleton_returns_empty_pdag =
    empty_pdag_all_zero &&
    empty_result.collider_orientations == 0 &&
    empty_result.generalized_orientations == 0;

  SkeletonResult collider_skeleton = empty_skeleton;
  collider_skeleton.adjacency.assign(static_cast<std::size_t>(p) * p, 0);
  collider_skeleton.adjacency[0 * p + 1] = 1;
  collider_skeleton.adjacency[1 * p + 0] = 1;
  collider_skeleton.adjacency[1 * p + 2] = 1;
  collider_skeleton.adjacency[2 * p + 1] = 1;
  collider_skeleton.sepsets.assign(p, std::vector<std::vector<int> >(p));
  OrientationOptions collider_options = options;
  collider_options.orient_collider = true;
  collider_options.rule1 = true;
  collider_options.rule2 = true;
  collider_options.rule3 = true;
  const OrientationResult collider_result =
    orient_wanpdag_native(data, collider_skeleton, collider_options);
  const bool collider_stage_count_correct =
    collider_result.collider_orientations > 0 &&
    has_directed_edge(collider_result.pdag, p, 0, 1) &&
    has_directed_edge(collider_result.pdag, p, 2, 1);
  const bool solve_confl_false_no_bidirected =
    !has_conflict_edge(collider_result.pdag, p, 0, 1) &&
    !has_conflict_edge(collider_result.pdag, p, 1, 2);

  SkeletonResult generalized_skeleton = empty_skeleton;
  generalized_skeleton.adjacency.assign(static_cast<std::size_t>(p) * p, 0);
  generalized_skeleton.adjacency[0 * p + 1] = 1;
  generalized_skeleton.adjacency[1 * p + 0] = 1;
  generalized_skeleton.sepsets.assign(p, std::vector<std::vector<int> >(p));
  Rcpp::NumericMatrix generalized_data(n, p);
  for (int i = 0; i < n; ++i) {
    const double z = -3.141592653589793 + 6.283185307179586 *
      static_cast<double>(i) / (n - 1);
    generalized_data(i, 0) = z;
    generalized_data(i, 1) = std::sin(z) + 0.03 * std::cos(19.0 * z);
    generalized_data(i, 2) = std::cos(0.5 * z);
    generalized_data(i, 3) = z * z;
  }
  OrientationOptions generalized_options = options;
  generalized_options.alpha = 0.05;
  generalized_options.residual_backend_name = "fastSpline";
  generalized_options.orient_collider = false;
  generalized_options.rule1 = true;
  generalized_options.rule2 = true;
  generalized_options.rule3 = true;
  const OrientationResult generalized_result =
    orient_wanpdag_native(generalized_data, generalized_skeleton,
                          generalized_options);
  const bool generalized_stage_orients_expected_edge =
    generalized_result.generalized_orientations > 0 &&
    has_directed_edge(generalized_result.pdag, p, 0, 1);

  SkeletonResult rule_skeleton = empty_skeleton;
  rule_skeleton.adjacency.assign(static_cast<std::size_t>(p) * p, 0);
  rule_skeleton.adjacency[0 * p + 1] = 1;
  rule_skeleton.adjacency[1 * p + 0] = 1;
  rule_skeleton.adjacency[1 * p + 2] = 1;
  rule_skeleton.adjacency[2 * p + 1] = 1;
  rule_skeleton.adjacency[2 * p + 3] = 1;
  rule_skeleton.adjacency[3 * p + 2] = 1;
  rule_skeleton.sepsets.assign(p, std::vector<std::vector<int> >(p));

  Rcpp::NumericMatrix rule_data(n, p);
  for (int i = 0; i < n; ++i) {
    const double z0 = -3.141592653589793 + 6.283185307179586 *
      static_cast<double>(i) / (n - 1);
    const double z2 = -2.0 + 4.0 * ((i * 37) % n) / static_cast<double>(n - 1);
    rule_data(i, 0) = z0;
    rule_data(i, 1) = std::sin(z0) + 0.03 * std::cos(19.0 * z0);
    rule_data(i, 2) = z2;
    rule_data(i, 3) = z2 + 0.03 * std::sin(11.0 * z2);
  }

  OrientationOptions rule_options = options;
  rule_options.alpha = 0.05;
  rule_options.residual_backend_name = "fastSpline";
  rule_options.orient_collider = false;
  rule_options.rule1 = true;
  rule_options.rule2 = true;
  rule_options.rule3 = true;
  const OrientationResult rule_result =
    orient_wanpdag_native(rule_data, rule_skeleton, rule_options);
  const bool rules_stage_count_correct =
    rule_result.rule1_orientations > 0 &&
    has_directed_edge(rule_result.pdag, p, 2, 3);

  Rcpp::List generalized_list = orientation_result_to_list(generalized_result);
  Rcpp::List events = generalized_list["events"];
  bool event_log_has_one_based_indices_in_R = events.size() == 0;
  for (int i = 0; i < events.size(); ++i) {
    Rcpp::List event = events[i];
    if (!Rcpp::IntegerVector::is_na(Rcpp::as<int>(event["y"]))) {
      event_log_has_one_based_indices_in_R =
        Rcpp::as<int>(event["y"]) >= 1 && Rcpp::as<int>(event["y"]) <= p;
      break;
    }
  }

  const bool residual_backend_params_recorded =
    generalized_result.residual_backend == "fastSpline" &&
    generalized_result.residual_backend_params.find("degree=3") !=
      std::string::npos;
  const bool cache_stats_recorded =
    generalized_result.residual_cache_requests >=
      generalized_result.residual_cache_computations &&
    generalized_result.residual_cache_computations > 0;

  OrientationOptions disabled_options = generalized_options;
  disabled_options.rule1 = false;
  disabled_options.rule2 = false;
  disabled_options.rule3 = false;
  const OrientationResult disabled_result =
    orient_wanpdag_native(data, generalized_skeleton, disabled_options);
  const bool rule_flags_disable_rules =
    disabled_result.rule1_orientations == 0 &&
    disabled_result.rule2_orientations == 0 &&
    disabled_result.rule3_orientations == 0;

  return Rcpp::List::create(
    Rcpp::Named("empty_skeleton_returns_empty_pdag") =
      empty_skeleton_returns_empty_pdag,
    Rcpp::Named("collider_stage_count_correct") =
      collider_stage_count_correct,
    Rcpp::Named("rules_stage_count_correct") = rules_stage_count_correct,
    Rcpp::Named("generalized_stage_orients_expected_edge") =
      generalized_stage_orients_expected_edge,
    Rcpp::Named("event_log_has_one_based_indices_in_R") =
      event_log_has_one_based_indices_in_R,
    Rcpp::Named("residual_backend_params_recorded") =
      residual_backend_params_recorded,
    Rcpp::Named("cache_stats_recorded") = cache_stats_recorded,
    Rcpp::Named("solve_confl_false_no_bidirected") =
      solve_confl_false_no_bidirected,
    Rcpp::Named("rule_flags_disable_rules") = rule_flags_disable_rules
  );
}
