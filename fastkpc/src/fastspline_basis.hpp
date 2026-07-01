#ifndef FASTKPC_FASTSPLINE_BASIS_HPP
#define FASTKPC_FASTSPLINE_BASIS_HPP

#include <Rcpp.h>
#include <map>
#include <string>
#include <vector>

struct FastSplineParams {
  int degree;
  int knots;
  double lambda_min;
  double lambda_max;
  int lambda_count;
  double ridge;
  std::string mode;
};

struct FastSplineDesign {
  std::vector<double> X;
  std::vector<double> P;
  int n;
  int p;
};

struct FastSplineDesignBuildDiagnostics {
  double total_sec;
  double basis_sec;
  double penalty_sec;
  double x_pack_sec;
  double p_pack_sec;
  double alloc_sec;
  double column_extract_sec;
  double unaccounted_sec;
  int build_count;
  int x_values;
  int p_values;
  int basis_values;
  int penalty_values;
  int condition_cols;
  int basis_cache_hit_count;
  int basis_cache_miss_count;
  int basis_cache_insert_count;
  int basis_cache_entries;
  double basis_cache_hit_sec;
  double basis_cache_miss_build_sec;
  double basis_build_total_sec;
  double basis_build_alloc_sec;
  double basis_build_near_constant_sec;
  double basis_build_knots_sec;
  double basis_build_knots_copy_sec;
  double basis_build_knots_sort_sec;
  double basis_build_knots_center_sec;
  double basis_build_min_gap_sec;
  double basis_build_width_sec;
  double basis_build_eval_sec;
  double basis_build_eval_fill_sec;
  double basis_build_normalize_sec;
  double basis_build_normalize_scale_sec;
  double basis_build_fallback_sec;
  double basis_build_return_sec;
  double basis_build_unaccounted_sec;
  int basis_build_count;
  int basis_build_rows;
  int basis_build_cols;
  int basis_build_values;
  int basis_build_near_constant_count;
  int basis_build_fallback_row_count;
};

struct FastSplineBasisBlock {
  int n;
  int n_basis;
  std::vector<double> values;
};

struct FastSplineBasisCache {
  std::map<std::string, FastSplineBasisBlock> entries;
};

FastSplineParams default_fastspline_params();
std::string serialize_fastspline_params(const FastSplineParams& params);

std::vector<double> quantile_knots(const std::vector<double>& x, int knots);
std::vector<double> cubic_bspline_basis(const std::vector<double>& x,
                                        const FastSplineParams& params,
                                        int* n_basis,
                                        FastSplineDesignBuildDiagnostics* diagnostics = nullptr);
std::vector<double> second_difference_penalty(int n_basis);

FastSplineDesignBuildDiagnostics make_empty_fastspline_design_build_diagnostics();

FastSplineDesign make_fastspline_design(
    const Rcpp::NumericMatrix& data,
    const std::vector<int>& conditioning_set,
    const FastSplineParams& params,
    FastSplineDesignBuildDiagnostics* diagnostics = nullptr,
    FastSplineBasisCache* basis_cache = nullptr);

#endif
