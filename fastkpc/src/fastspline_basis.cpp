#include "fastspline_basis.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {

double elapsed_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(
    std::chrono::steady_clock::now() - start).count();
}

double nonnegative_gap(double total, double accounted) {
  return std::max(0.0, total - accounted);
}

std::chrono::steady_clock::time_point design_timing_start(
    const FastSplineDesignBuildDiagnostics* diagnostics) {
  return diagnostics == nullptr ?
    std::chrono::steady_clock::time_point() :
    std::chrono::steady_clock::now();
}

void add_elapsed(double* out,
                 const FastSplineDesignBuildDiagnostics* diagnostics,
                 std::chrono::steady_clock::time_point start) {
  if (diagnostics == nullptr) return;
  *out += elapsed_since(start);
}

std::size_t ridx(int row, int col, int ncol) {
  return static_cast<std::size_t>(row) * ncol + col;
}

bool finite_values(const std::vector<double>& values) {
  for (double value : values) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

std::vector<double> column_values(const Rcpp::NumericMatrix& data, int col) {
  if (col < 0 || col >= data.ncol()) throw std::runtime_error("conditioning column out of range");
  std::vector<double> values(data.nrow());
  for (int i = 0; i < data.nrow(); ++i) values[i] = data(i, col);
  return values;
}

double interpolate_sorted_quantile(const std::vector<double>& sorted, double prob) {
  if (sorted.empty()) return 0.0;
  if (prob <= 0.0) return sorted.front();
  if (prob >= 1.0) return sorted.back();
  const double pos = prob * static_cast<double>(sorted.size() - 1);
  const int lo = static_cast<int>(std::floor(pos));
  const int hi = static_cast<int>(std::ceil(pos));
  const double frac = pos - lo;
  return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

bool near_constant(const std::vector<double>& x) {
  if (x.empty()) return true;
  double lo = x[0];
  double hi = x[0];
  for (double value : x) {
    lo = std::min(lo, value);
    hi = std::max(hi, value);
  }
  return std::abs(hi - lo) <= 100.0 * std::numeric_limits<double>::epsilon() *
    std::max(1.0, std::max(std::abs(lo), std::abs(hi)));
}

std::vector<double> identity_penalty(int n_basis) {
  std::vector<double> out(static_cast<std::size_t>(n_basis) * n_basis, 0.0);
  for (int i = 0; i < n_basis; ++i) out[ridx(i, i, n_basis)] = 1.0;
  return out;
}

void add_penalty_block(std::vector<double>* P,
                       int p,
                       int offset,
                       const std::vector<double>& block,
                       int block_dim,
                       double multiplier) {
  for (int i = 0; i < block_dim; ++i) {
    for (int j = 0; j < block_dim; ++j) {
      (*P)[ridx(offset + i, offset + j, p)] += multiplier * block[ridx(i, j, block_dim)];
    }
  }
}

void copy_basis_into_design(const std::vector<double>& basis,
                            int n,
                            int n_basis,
                            int dest_offset,
                            int p,
                            std::vector<double>* X) {
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n_basis; ++col) {
      (*X)[ridx(row, dest_offset + col, p)] = basis[ridx(row, col, n_basis)];
    }
  }
}

FastSplineDesign one_dimensional_design(const Rcpp::NumericMatrix& data,
                                        int col,
                                        const FastSplineParams& params,
                                        FastSplineDesignBuildDiagnostics* diagnostics) {
  std::chrono::steady_clock::time_point stage =
    design_timing_start(diagnostics);
  const std::vector<double> x = column_values(data, col);
  if (diagnostics != nullptr) {
    add_elapsed(&diagnostics->column_extract_sec, diagnostics, stage);
    diagnostics->condition_cols += 1;
  }
  int n_basis = 0;
  stage = design_timing_start(diagnostics);
  const std::vector<double> basis = cubic_bspline_basis(x, params, &n_basis);
  if (diagnostics != nullptr) {
    add_elapsed(&diagnostics->basis_sec, diagnostics, stage);
    diagnostics->basis_values += static_cast<int>(basis.size());
  }
  const int n = data.nrow();
  const int p = 1 + n_basis;

  FastSplineDesign design;
  design.n = n;
  design.p = p;
  stage = design_timing_start(diagnostics);
  design.X.assign(static_cast<std::size_t>(n) * p, 0.0);
  design.P.assign(static_cast<std::size_t>(p) * p, 0.0);
  if (diagnostics != nullptr) {
    add_elapsed(&diagnostics->alloc_sec, diagnostics, stage);
  }
  stage = design_timing_start(diagnostics);
  for (int row = 0; row < n; ++row) design.X[ridx(row, 0, p)] = 1.0;
  copy_basis_into_design(basis, n, n_basis, 1, p, &design.X);
  if (diagnostics != nullptr) {
    add_elapsed(&diagnostics->x_pack_sec, diagnostics, stage);
  }
  stage = design_timing_start(diagnostics);
  const std::vector<double> penalty = second_difference_penalty(n_basis);
  if (diagnostics != nullptr) {
    add_elapsed(&diagnostics->penalty_sec, diagnostics, stage);
    diagnostics->penalty_values += static_cast<int>(penalty.size());
  }
  stage = design_timing_start(diagnostics);
  add_penalty_block(&design.P, p, 1, penalty, n_basis, 1.0);
  if (diagnostics != nullptr) {
    add_elapsed(&diagnostics->p_pack_sec, diagnostics, stage);
  }
  return design;
}

FastSplineDesign tensor_design(const Rcpp::NumericMatrix& data,
                               const std::vector<int>& conditioning_set,
                               const FastSplineParams& params,
                               FastSplineDesignBuildDiagnostics* diagnostics) {
  std::chrono::steady_clock::time_point stage =
    design_timing_start(diagnostics);
  const std::vector<double> x1 = column_values(data, conditioning_set[0]);
  const std::vector<double> x2 = column_values(data, conditioning_set[1]);
  if (diagnostics != nullptr) {
    add_elapsed(&diagnostics->column_extract_sec, diagnostics, stage);
    diagnostics->condition_cols += 2;
  }
  int b1_cols = 0;
  int b2_cols = 0;
  stage = design_timing_start(diagnostics);
  const std::vector<double> b1 = cubic_bspline_basis(x1, params, &b1_cols);
  const std::vector<double> b2 = cubic_bspline_basis(x2, params, &b2_cols);
  if (diagnostics != nullptr) {
    add_elapsed(&diagnostics->basis_sec, diagnostics, stage);
    diagnostics->basis_values +=
      static_cast<int>(b1.size() + b2.size());
  }
  const int n = data.nrow();
  const int tensor_cols = b1_cols * b2_cols;
  const int p = 1 + tensor_cols;

  FastSplineDesign design;
  design.n = n;
  design.p = p;
  stage = design_timing_start(diagnostics);
  design.X.assign(static_cast<std::size_t>(n) * p, 0.0);
  design.P.assign(static_cast<std::size_t>(p) * p, 0.0);
  if (diagnostics != nullptr) {
    add_elapsed(&diagnostics->alloc_sec, diagnostics, stage);
  }
  stage = design_timing_start(diagnostics);
  for (int row = 0; row < n; ++row) {
    design.X[ridx(row, 0, p)] = 1.0;
    int dest = 1;
    for (int a = 0; a < b1_cols; ++a) {
      for (int b = 0; b < b2_cols; ++b) {
        design.X[ridx(row, dest, p)] =
          b1[ridx(row, a, b1_cols)] * b2[ridx(row, b, b2_cols)];
        ++dest;
      }
    }
  }
  if (diagnostics != nullptr) {
    add_elapsed(&diagnostics->x_pack_sec, diagnostics, stage);
  }

  stage = design_timing_start(diagnostics);
  const std::vector<double> p1 = second_difference_penalty(b1_cols);
  const std::vector<double> p2 = second_difference_penalty(b2_cols);
  if (diagnostics != nullptr) {
    add_elapsed(&diagnostics->penalty_sec, diagnostics, stage);
    diagnostics->penalty_values +=
      static_cast<int>(p1.size() + p2.size());
  }
  stage = design_timing_start(diagnostics);
  for (int a1 = 0; a1 < b1_cols; ++a1) {
    for (int a2 = 0; a2 < b1_cols; ++a2) {
      const double value = p1[ridx(a1, a2, b1_cols)];
      for (int b = 0; b < b2_cols; ++b) {
        const int row = 1 + a1 * b2_cols + b;
        const int col = 1 + a2 * b2_cols + b;
        design.P[ridx(row, col, p)] += value;
      }
    }
  }
  for (int b1_idx = 0; b1_idx < b2_cols; ++b1_idx) {
    for (int b2_idx = 0; b2_idx < b2_cols; ++b2_idx) {
      const double value = p2[ridx(b1_idx, b2_idx, b2_cols)];
      for (int a = 0; a < b1_cols; ++a) {
        const int row = 1 + a * b2_cols + b1_idx;
        const int col = 1 + a * b2_cols + b2_idx;
        design.P[ridx(row, col, p)] += value;
      }
    }
  }
  if (diagnostics != nullptr) {
    add_elapsed(&diagnostics->p_pack_sec, diagnostics, stage);
  }
  return design;
}

FastSplineDesign additive_design(const Rcpp::NumericMatrix& data,
                                 const std::vector<int>& conditioning_set,
                                 const FastSplineParams& params,
                                 FastSplineDesignBuildDiagnostics* diagnostics) {
  const int n = data.nrow();
  std::vector<std::vector<double> > bases;
  std::vector<int> basis_cols;
  int total_basis_cols = 0;
  for (int col : conditioning_set) {
    int n_basis = 0;
    std::chrono::steady_clock::time_point stage =
      design_timing_start(diagnostics);
    std::vector<double> values = column_values(data, col);
    if (diagnostics != nullptr) {
      add_elapsed(&diagnostics->column_extract_sec, diagnostics, stage);
      diagnostics->condition_cols += 1;
    }
    stage = design_timing_start(diagnostics);
    std::vector<double> basis = cubic_bspline_basis(values, params, &n_basis);
    if (diagnostics != nullptr) {
      add_elapsed(&diagnostics->basis_sec, diagnostics, stage);
      diagnostics->basis_values += static_cast<int>(basis.size());
    }
    bases.push_back(basis);
    basis_cols.push_back(n_basis);
    total_basis_cols += n_basis;
  }
  const int p = 1 + total_basis_cols;

  FastSplineDesign design;
  design.n = n;
  design.p = p;
  std::chrono::steady_clock::time_point stage =
    design_timing_start(diagnostics);
  design.X.assign(static_cast<std::size_t>(n) * p, 0.0);
  design.P.assign(static_cast<std::size_t>(p) * p, 0.0);
  if (diagnostics != nullptr) {
    add_elapsed(&diagnostics->alloc_sec, diagnostics, stage);
  }
  stage = design_timing_start(diagnostics);
  for (int row = 0; row < n; ++row) design.X[ridx(row, 0, p)] = 1.0;
  if (diagnostics != nullptr) {
    add_elapsed(&diagnostics->x_pack_sec, diagnostics, stage);
  }

  int offset = 1;
  for (int block = 0; block < static_cast<int>(bases.size()); ++block) {
    stage = design_timing_start(diagnostics);
    copy_basis_into_design(bases[block], n, basis_cols[block], offset, p, &design.X);
    if (diagnostics != nullptr) {
      add_elapsed(&diagnostics->x_pack_sec, diagnostics, stage);
    }
    stage = design_timing_start(diagnostics);
    const std::vector<double> penalty =
      second_difference_penalty(basis_cols[block]);
    if (diagnostics != nullptr) {
      add_elapsed(&diagnostics->penalty_sec, diagnostics, stage);
      diagnostics->penalty_values += static_cast<int>(penalty.size());
    }
    stage = design_timing_start(diagnostics);
    add_penalty_block(&design.P, p, offset,
                      penalty, basis_cols[block], 1.0);
    if (diagnostics != nullptr) {
      add_elapsed(&diagnostics->p_pack_sec, diagnostics, stage);
    }
    offset += basis_cols[block];
  }
  return design;
}

}  // namespace

FastSplineParams default_fastspline_params() {
  FastSplineParams params;
  params.degree = 3;
  params.knots = 10;
  params.lambda_min = 1e-4;
  params.lambda_max = 1e4;
  params.lambda_count = 25;
  params.ridge = 1e-8;
  params.mode = "auto";
  return params;
}

std::string serialize_fastspline_params(const FastSplineParams& params) {
  std::ostringstream out;
  out << std::setprecision(17)
      << "degree=" << params.degree
      << ";knots=" << params.knots
      << ";lambda_grid=" << params.lambda_min << ":" << params.lambda_max
      << ":" << params.lambda_count
      << ";ridge=" << params.ridge
      << ";mode=" << params.mode;
  return out.str();
}

FastSplineDesignBuildDiagnostics make_empty_fastspline_design_build_diagnostics() {
  FastSplineDesignBuildDiagnostics out;
  out.total_sec = 0.0;
  out.basis_sec = 0.0;
  out.penalty_sec = 0.0;
  out.x_pack_sec = 0.0;
  out.p_pack_sec = 0.0;
  out.alloc_sec = 0.0;
  out.column_extract_sec = 0.0;
  out.unaccounted_sec = 0.0;
  out.build_count = 0;
  out.x_values = 0;
  out.p_values = 0;
  out.basis_values = 0;
  out.penalty_values = 0;
  out.condition_cols = 0;
  return out;
}

std::vector<double> quantile_knots(const std::vector<double>& x, int knots) {
  const int count = std::max(2, knots);
  std::vector<double> sorted = x;
  std::sort(sorted.begin(), sorted.end());
  std::vector<double> out(count);
  if (near_constant(sorted)) {
    std::fill(out.begin(), out.end(), sorted.empty() ? 0.0 : sorted.front());
    return out;
  }
  for (int i = 0; i < count; ++i) {
    out[i] = interpolate_sorted_quantile(sorted, static_cast<double>(i) / (count - 1));
  }
  return out;
}

std::vector<double> cubic_bspline_basis(const std::vector<double>& x,
                                        const FastSplineParams& params,
                                        int* n_basis) {
  const int n = static_cast<int>(x.size());
  const int basis_cols = std::max(6, params.knots);
  if (n_basis != nullptr) *n_basis = basis_cols;
  std::vector<double> out(static_cast<std::size_t>(n) * basis_cols, 0.0);
  if (n == 0) return out;

  if (near_constant(x)) {
    for (int row = 0; row < n; ++row) out[ridx(row, 0, basis_cols)] = 1.0;
    return out;
  }

  const std::vector<double> centers = quantile_knots(x, basis_cols);
  double min_gap = std::numeric_limits<double>::infinity();
  for (int i = 1; i < static_cast<int>(centers.size()); ++i) {
    const double gap = centers[i] - centers[i - 1];
    if (gap > 0.0) min_gap = std::min(min_gap, gap);
  }
  if (!std::isfinite(min_gap) || min_gap <= 0.0) min_gap = 1.0;
  const double width = 2.5 * min_gap;

  for (int row = 0; row < n; ++row) {
    double total = 0.0;
    for (int col = 0; col < basis_cols; ++col) {
      const double scaled = std::abs(x[row] - centers[col]) / width;
      const double value = scaled < 1.0 ? std::pow(1.0 - scaled, 3.0) : 0.0;
      out[ridx(row, col, basis_cols)] = value;
      total += value;
    }
    if (total <= 0.0 || !std::isfinite(total)) {
      int nearest = 0;
      double best = std::abs(x[row] - centers[0]);
      for (int col = 1; col < basis_cols; ++col) {
        const double candidate = std::abs(x[row] - centers[col]);
        if (candidate < best) {
          nearest = col;
          best = candidate;
        }
      }
      for (int col = 0; col < basis_cols; ++col) out[ridx(row, col, basis_cols)] = 0.0;
      out[ridx(row, nearest, basis_cols)] = 1.0;
    } else {
      for (int col = 0; col < basis_cols; ++col) out[ridx(row, col, basis_cols)] /= total;
    }
  }
  return out;
}

std::vector<double> second_difference_penalty(int n_basis) {
  if (n_basis <= 0) return std::vector<double>();
  if (n_basis < 3) return identity_penalty(n_basis);
  std::vector<double> out(static_cast<std::size_t>(n_basis) * n_basis, 0.0);
  for (int row = 0; row < n_basis - 2; ++row) {
    const int cols[3] = {row, row + 1, row + 2};
    const double vals[3] = {1.0, -2.0, 1.0};
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        out[ridx(cols[a], cols[b], n_basis)] += vals[a] * vals[b];
      }
    }
  }
  return out;
}

FastSplineDesign make_fastspline_design(const Rcpp::NumericMatrix& data,
                                        const std::vector<int>& conditioning_set,
                                        const FastSplineParams& params,
                                        FastSplineDesignBuildDiagnostics* diagnostics) {
  const std::chrono::steady_clock::time_point total_start =
    design_timing_start(diagnostics);
  if (diagnostics != nullptr) diagnostics->build_count += 1;
  if (!finite_values(std::vector<double>(data.begin(), data.end()))) {
    throw std::runtime_error("fastSpline design data contains non-finite values");
  }
  FastSplineDesign design;
  if (conditioning_set.empty()) {
    design.n = data.nrow();
    design.p = 1;
    std::chrono::steady_clock::time_point stage =
      design_timing_start(diagnostics);
    design.X.assign(static_cast<std::size_t>(design.n), 1.0);
    design.P.assign(1, 0.0);
    if (diagnostics != nullptr) {
      add_elapsed(&diagnostics->alloc_sec, diagnostics, stage);
    }
  } else if (conditioning_set.size() == 1) {
    design = one_dimensional_design(data, conditioning_set[0], params,
                                    diagnostics);
  } else if (conditioning_set.size() == 2) {
    design = tensor_design(data, conditioning_set, params, diagnostics);
  } else {
    design = additive_design(data, conditioning_set, params, diagnostics);
  }
  if (diagnostics != nullptr) {
    diagnostics->x_values += static_cast<int>(design.X.size());
    diagnostics->p_values += static_cast<int>(design.P.size());
    const double total = elapsed_since(total_start);
    diagnostics->total_sec += total;
    const double accounted =
      diagnostics->basis_sec +
      diagnostics->penalty_sec +
      diagnostics->x_pack_sec +
      diagnostics->p_pack_sec +
      diagnostics->alloc_sec +
      diagnostics->column_extract_sec;
    diagnostics->unaccounted_sec += nonnegative_gap(total, accounted);
  }
  return design;
}
