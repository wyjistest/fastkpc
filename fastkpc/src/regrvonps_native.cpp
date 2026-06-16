#include "regrvonps_native.hpp"

#include "ci_method.hpp"
#include "dcov_exact_cpu.hpp"
#include "orientation_matrix.hpp"

#include <algorithm>
#include <stdexcept>

namespace {

std::vector<double> extract_column(const Rcpp::NumericMatrix& data, int col) {
  if (col < 0 || col >= data.ncol()) {
    throw std::runtime_error("data column index out of range");
  }
  std::vector<double> out(data.nrow());
  for (int row = 0; row < data.nrow(); ++row) out[row] = data(row, col);
  return out;
}

void validate_node(int node, int p) {
  if (node < 0 || node >= p) {
    throw std::runtime_error("node index out of range");
  }
}

}  // namespace

std::vector<int> parents_of(const std::vector<int>& pdag, int p, int V) {
  validate_node(V, p);
  std::vector<int> parents;
  for (int node = 0; node < p; ++node) {
    if (node != V && has_directed_edge(pdag, p, node, V)) {
      parents.push_back(node);
    }
  }
  return parents;
}

std::vector<int> sorted_unique_union(const std::vector<int>& a,
                                     const std::vector<int>& b) {
  std::vector<int> out = a;
  out.insert(out.end(), b.begin(), b.end());
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

RegrVonPsResult regrvonps_native(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& pdag,
  int p,
  int V,
  const std::vector<int>& S,
  const OrientationOptions& options,
  ResidualCache* residual_cache) {
  if (data.ncol() != p) {
    throw std::runtime_error("data and pdag dimensions differ");
  }
  if (residual_cache == NULL) {
    throw std::runtime_error("residual cache is required");
  }
  validate_node(V, p);
  for (int node : S) validate_node(node, p);

  RegrVonPsResult result;
  result.reject_count = 0;
  result.parents = parents_of(pdag, p, V);
  result.conditioning_set = sorted_unique_union(S, result.parents);
  result.used_cuda = false;
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
  result.residual_fits = 0;
  result.cuda_residual_fits = 0;
  result.cpu_fallback_fits = 0;
  result.ci_backend = "native-cpu";
  result.ci_backend_reason = "";

  ResidualCacheStats before = residual_cache->stats();
  result.cache_requests_before = before.requests;
  result.cache_hits_before = before.hits;

  const std::vector<double>& residuals =
    residual_cache->get(data, V, result.conditioning_set);
  result.residual_fits = 1;
  const CiMethodKind ci_method = parse_ci_method_kind(options.ci_method);

  if (!S.empty()) {
    for (int node : S) {
      const std::vector<double> other = extract_column(data, node);
      const CiEvaluation ci = evaluate_ci_vectors(
        residuals, other, ci_method, options.index, options.legacy_index,
        options.hsic_options);
      const double p_value = ci.p_value;
      result.p_values.push_back(p_value);
      if (p_value < options.alpha) ++result.reject_count;
      if (ci.kind == CiMethodKind::DccGamma) {
        ++result.dcc_gamma_tests;
      } else if (ci.kind == CiMethodKind::HsicGamma) {
        ++result.hsic_gamma_tests;
      } else {
        ++result.hsic_perm_tests;
        result.hsic_permutation_replicates += options.hsic_options.replicates;
      }
    }
    if (ci_method == CiMethodKind::DccGamma) {
      result.dcov_batches = static_cast<int>(S.size());
      result.dcov_pairs = static_cast<int>(S.size());
    }
  }

  ResidualCacheStats after = residual_cache->stats();
  result.cache_requests_after = after.requests;
  result.cache_hits_after = after.hits;
  return result;
}
