#include "skeleton_engine.hpp"

#include "ci_method.hpp"
#include "dcov_exact_cpu.hpp"
#include "residual_cache.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

namespace {

int idx(int row, int col, int p) {
  return row * p + col;
}

std::vector<int> neighbors_from_snapshot(const std::vector<int>& adjacency,
                                         int p,
                                         int vertex,
                                         int excluded) {
  std::vector<int> out;
  for (int i = 0; i < p; ++i) {
    if (i != excluded && adjacency[idx(i, vertex, p)] != 0) out.push_back(i);
  }
  return out;
}

void enumerate_combinations(const std::vector<int>& values,
                            int choose,
                            const std::function<bool(const std::vector<int>&)>& visitor) {
  if (choose == 0) {
    std::vector<int> empty;
    visitor(empty);
    return;
  }
  if (static_cast<int>(values.size()) < choose) return;

  std::vector<int> current;
  std::function<bool(int, int)> rec = [&](int start, int remaining) {
    if (remaining == 0) return visitor(current);
    for (int i = start; i <= static_cast<int>(values.size()) - remaining; ++i) {
      current.push_back(values[i]);
      const bool keep_going = rec(i + 1, remaining - 1);
      current.pop_back();
      if (!keep_going) return false;
    }
    return true;
  };
  rec(0, choose);
}

std::vector<double> column_as_vector(const Rcpp::NumericMatrix& data, int col) {
  std::vector<double> out(data.nrow());
  for (int i = 0; i < data.nrow(); ++i) out[i] = data(i, col);
  return out;
}

CiEvaluation ci_pvalue_exact(const Rcpp::NumericMatrix& data,
                             int x,
                             int y,
                             const std::vector<int>& conditioning_set,
                             CiMethodKind ci_method,
                             double index,
                             bool legacy_index,
                             const HsicOptions& hsic_options,
                             ResidualCache* residual_cache) {
  if (conditioning_set.empty()) {
    return evaluate_ci_vectors(column_as_vector(data, x), column_as_vector(data, y),
                               ci_method, index, legacy_index, hsic_options);
  }
  const std::vector<double> rx = residual_cache->get(data, x, conditioning_set);
  const std::vector<double> ry = residual_cache->get(data, y, conditioning_set);
  return evaluate_ci_vectors(rx, ry, ci_method, index, legacy_index,
                             hsic_options);
}

void record_ci_diagnostic(CiMethodKind kind,
                          const HsicOptions& hsic_options,
                          SkeletonResult* result) {
  if (kind == CiMethodKind::DccGamma) {
    ++result->ci_dcc_gamma_tests;
  } else if (kind == CiMethodKind::HsicGamma) {
    ++result->ci_hsic_gamma_tests;
  } else {
    ++result->ci_hsic_perm_tests;
    result->ci_hsic_permutation_replicates += hsic_options.replicates;
  }
}

}  // namespace

SkeletonResult run_skeleton_exact(const Rcpp::NumericMatrix& data,
                                  const SkeletonOptions& options) {
  const int p = data.ncol();
  const std::string residual_backend_name =
    options.residual_backend_name.empty() ? "linear" : options.residual_backend_name;
  const CiMethodKind ci_method = parse_ci_method_kind(options.ci_method);
  HsicOptions hsic_options = options.hsic_options;
  if (hsic_options.sig <= 0.0 || !std::isfinite(hsic_options.sig)) {
    hsic_options = default_hsic_options();
  }
  ResidualCache residual_cache(backend_residual_cache_options(
    residual_backend_name, options.fastspline_params, options.residual_cache_enabled));
  SkeletonResult result;
  result.adjacency.assign(static_cast<std::size_t>(p) * p, 1);
  result.pmax.assign(static_cast<std::size_t>(p) * p, -std::numeric_limits<double>::infinity());
  result.sepsets.resize(p, std::vector<std::vector<int> >(p));
  result.ci_method = ci_method_name(ci_method);
  result.ci_backend = "native-cpu";
  result.ci_backend_reason = "";
  result.ci_dcc_gamma_tests = 0;
  result.ci_hsic_gamma_tests = 0;
  result.ci_hsic_perm_tests = 0;
  result.ci_hsic_permutation_replicates = 0;
  result.ci_hsic_gamma_cuda_tests = 0;
  result.ci_hsic_perm_cuda_tests = 0;
  result.ci_hsic_cuda_batches = 0;
  result.ci_hsic_cuda_pairs = 0;
  result.ci_hsic_cuda_fallback_tests = 0;
  result.ci_hsic_cuda_memory_bytes = 0;
  result.ci_hsic_cuda_max_n = 0;
  result.ci_hsic_cuda_max_batch_pairs = 0;

  for (int i = 0; i < p; ++i) {
    result.adjacency[idx(i, i, p)] = 0;
    result.pmax[idx(i, i, p)] = 1.0;
  }

  const int max_order = std::max(0, options.max_conditioning_size);
  for (int ord = 0; ord <= max_order; ++ord) {
    const std::vector<int> snapshot = result.adjacency;
    std::vector<int> delete_edges(static_cast<std::size_t>(p) * p, 0);
    int level_tests = 0;
    std::vector<LevelDeletion> level_log;

    for (int x = 0; x < p - 1; ++x) {
      for (int y = x + 1; y < p; ++y) {
        if (snapshot[idx(x, y, p)] == 0) continue;
        bool edge_done = false;

        const std::vector<int> nx = neighbors_from_snapshot(snapshot, p, x, y);
        enumerate_combinations(nx, ord, [&](const std::vector<int>& cond) {
          ++level_tests;
          const CiEvaluation ci = ci_pvalue_exact(data, x, y, cond, ci_method,
                                                  options.index,
                                                  options.legacy_index,
                                                  hsic_options,
                                                  &residual_cache);
          record_ci_diagnostic(ci.kind, hsic_options, &result);
          double pval = ci.p_value;
          if (!std::isfinite(pval)) pval = options.na_delete ? 1.0 : 0.0;
          const double current = result.pmax[idx(x, y, p)];
          if (pval > current) {
            result.pmax[idx(x, y, p)] = pval;
            result.pmax[idx(y, x, p)] = pval;
          }
          if (pval >= options.alpha) {
            delete_edges[idx(x, y, p)] = 1;
            delete_edges[idx(y, x, p)] = 1;
            result.sepsets[x][y] = cond;
            result.sepsets[y][x] = cond;
            level_log.push_back(LevelDeletion{x, y, cond, pval});
            edge_done = true;
            return false;
          }
          return true;
        });

        if (edge_done) continue;

        const std::vector<int> ny = neighbors_from_snapshot(snapshot, p, y, x);
        enumerate_combinations(ny, ord, [&](const std::vector<int>& cond) {
          ++level_tests;
          const CiEvaluation ci = ci_pvalue_exact(data, y, x, cond, ci_method,
                                                  options.index,
                                                  options.legacy_index,
                                                  hsic_options,
                                                  &residual_cache);
          record_ci_diagnostic(ci.kind, hsic_options, &result);
          double pval = ci.p_value;
          if (!std::isfinite(pval)) pval = options.na_delete ? 1.0 : 0.0;
          const double current = result.pmax[idx(x, y, p)];
          if (pval > current) {
            result.pmax[idx(x, y, p)] = pval;
            result.pmax[idx(y, x, p)] = pval;
          }
          if (pval >= options.alpha) {
            delete_edges[idx(x, y, p)] = 1;
            delete_edges[idx(y, x, p)] = 1;
            result.sepsets[x][y] = cond;
            result.sepsets[y][x] = cond;
            level_log.push_back(LevelDeletion{x, y, cond, pval});
            edge_done = true;
            return false;
          }
          return true;
        });
      }
    }

    for (int i = 0; i < p * p; ++i) {
      if (delete_edges[i] != 0) result.adjacency[i] = 0;
    }
    result.n_edge_tests.push_back(level_tests);
    result.per_level_log.push_back(level_log);
  }
  const ResidualCacheStats stats = residual_cache.stats();
  result.residual_cache_enabled = stats.enabled;
  result.residual_cache_requests = stats.requests;
  result.residual_cache_hits = stats.hits;
  result.residual_cache_misses = stats.misses;
  result.residual_cache_computations = stats.computations;
  result.residual_cache_stored_vectors = stats.stored_vectors;
  result.residual_cache_stored_values = stats.stored_values;
  result.residual_backend = stats.backend_name;
  result.residual_backend_params =
    make_residual_backend_config(residual_backend_name, options.fastspline_params).params;
  return result;
}
