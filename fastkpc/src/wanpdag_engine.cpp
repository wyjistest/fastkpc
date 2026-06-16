#include "wanpdag_engine.hpp"

#include "ci_method.hpp"
#include "orientation_matrix.hpp"
#include "orientation_rules.hpp"
#include "residual_cache.hpp"

#include <Rmath.h>
#include <algorithm>
#include <functional>
#include <stdexcept>

namespace {

bool pdag_has_no_edges(const std::vector<int>& pdag) {
  for (int value : pdag) {
    if (value != FASTKPC_EDGE_NONE) return false;
  }
  return true;
}

std::vector<std::vector<int> > undirected_neighborhoods(
    const std::vector<int>& pdag,
    int p) {
  std::vector<std::vector<int> > out(p);
  for (int a = 0; a < p; ++a) {
    for (int b = 0; b < p; ++b) {
      if (a != b && has_undirected_edge(pdag, p, a, b)) {
        out[a].push_back(b);
      }
    }
  }
  return out;
}

int max_neighborhood_size(const std::vector<std::vector<int> >& neighborhoods) {
  int out = 0;
  for (std::size_t i = 0; i < neighborhoods.size(); ++i) {
    out = std::max(out, static_cast<int>(neighborhoods[i].size()));
  }
  return out;
}

std::vector<std::vector<int> > combinations_of_size(
    const std::vector<int>& values,
    int k) {
  std::vector<std::vector<int> > out;
  if (k <= 0 || k > static_cast<int>(values.size())) return out;
  std::vector<int> current;
  const int n = static_cast<int>(values.size());
  std::function<void(int, int)> rec = [&](int start, int remaining) {
    if (remaining == 0) {
      out.push_back(current);
      return;
    }
    for (int i = start; i <= n - remaining; ++i) {
      current.push_back(values[i]);
      rec(i + 1, remaining - 1);
      current.pop_back();
    }
  };
  rec(0, k);
  return out;
}

bool vector_contains(const std::vector<int>& values, int node) {
  return std::find(values.begin(), values.end(), node) != values.end();
}

OrientationEvent make_generalized_event(const std::string& rule,
                                        int x,
                                        int y,
                                        const std::vector<int>& S,
                                        double p_value,
                                        bool accepted,
                                        const std::string& message) {
  OrientationEvent event;
  event.phase = "generalized";
  event.rule = rule;
  event.x = x;
  event.y = y;
  event.z = -1;
  event.S = S;
  event.p_value = p_value;
  event.accepted = accepted;
  event.message = message;
  return event;
}

double first_or_nan(const std::vector<double>& values) {
  if (values.empty()) return R_NaN;
  return values[0];
}

void apply_generalized_orientation(std::vector<int>* pdag,
                                   int p,
                                   int V,
                                   const std::vector<int>& S,
                                   const std::vector<int>& neighborhood) {
  for (int node : S) set_directed_edge(pdag, p, node, V);
  for (int node : neighborhood) {
    if (!vector_contains(S, node) && has_undirected_edge(*pdag, p, V, node)) {
      set_directed_edge(pdag, p, V, node);
    }
  }
}

}  // namespace

OrientationOptions default_orientation_options() {
  OrientationOptions options;
  options.alpha = 0.2;
  options.verbose = false;
  options.solve_confl = false;
  options.orient_collider = true;
  options.rule1 = true;
  options.rule2 = true;
  options.rule3 = true;
  options.residual_cache_enabled = true;
  options.residual_backend_name = "fastSpline";
  options.orientation_residual_device_requested = "cpu";
  options.orientation_residual_device = "cpu";
  options.orientation_residual_device_reason = "";
  options.orientation_batch_size = 0;
  options.orientation_diagnostics_enabled = true;
  options.cuda_residual_fallback = true;
  options.fastspline_params = default_fastspline_params();
  options.index = 1.0;
  options.legacy_index = true;
  options.ci_method = "dcc.gamma";
  options.hsic_options = default_hsic_options();
  options.ci_diagnostics_enabled = true;
  return options;
}

namespace {

RegrVonPsResult evaluate_regrvonps(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& pdag,
  int p,
  int V,
  const std::vector<int>& S,
  const OrientationOptions& options,
  ResidualCache* residual_cache,
  RegrVonPsEvaluator evaluator) {
  const CiMethodKind ci_method = parse_ci_method_kind(options.ci_method);
  if (evaluator != NULL &&
      (options.orientation_residual_device == "cuda" ||
       ci_method == CiMethodKind::HsicGamma ||
       ci_method == CiMethodKind::HsicPermutation)) {
    return evaluator(data, pdag, p, V, S, options, residual_cache);
  }
  return regrvonps_native(data, pdag, p, V, S, options, residual_cache);
}

void add_regrvonps_diagnostics(OrientationResult* result,
                               const RegrVonPsResult& value) {
  if (value.used_cuda) {
    ++result->regrvonps_cuda_calls;
  } else {
    ++result->regrvonps_cpu_calls;
  }
  result->orientation_dcov_batches += value.dcov_batches;
  result->orientation_dcov_pairs += value.dcov_pairs;
  result->regrvonps_dcc_gamma_tests += value.dcc_gamma_tests;
  result->regrvonps_hsic_gamma_tests += value.hsic_gamma_tests;
  result->regrvonps_hsic_perm_tests += value.hsic_perm_tests;
  result->regrvonps_hsic_permutation_replicates +=
    value.hsic_permutation_replicates;
  result->regrvonps_hsic_gamma_cuda_tests += value.hsic_gamma_cuda_tests;
  result->regrvonps_hsic_perm_cuda_tests += value.hsic_perm_cuda_tests;
  result->regrvonps_hsic_cuda_batches += value.hsic_cuda_batches;
  result->regrvonps_hsic_cuda_pairs += value.hsic_cuda_pairs;
  result->regrvonps_hsic_cuda_fallback_tests +=
    value.hsic_cuda_fallback_tests;
  if (value.ci_backend == "cuda-hsic" &&
      (value.hsic_gamma_cuda_tests + value.hsic_perm_cuda_tests) > 0) {
    result->ci_backend = "cuda-hsic";
    result->ci_backend_reason = "";
  } else if (value.ci_backend == "native-cpu" &&
             !value.ci_backend_reason.empty() &&
             result->ci_backend != "cuda-hsic") {
    result->ci_backend = "native-cpu";
    result->ci_backend_reason = value.ci_backend_reason;
  }
  result->orientation_residual_fits += value.residual_fits;
  result->orientation_cuda_residual_fits += value.cuda_residual_fits;
  result->orientation_cpu_fallback_fits += value.cpu_fallback_fits;
}

}  // namespace

OrientationResult orient_wanpdag_native(
  const Rcpp::NumericMatrix& data,
  const SkeletonResult& skeleton,
  const OrientationOptions& options,
  RegrVonPsEvaluator evaluator) {
  const int p = data.ncol();
  if (p <= 0) {
    throw std::runtime_error("data must have at least one column");
  }
  if (static_cast<int>(skeleton.adjacency.size()) != p * p) {
    throw std::runtime_error("skeleton adjacency dimension mismatch");
  }

  OrientationResult result;
  result.p = p;
  result.pdag = pdag_from_skeleton_adjacency(skeleton.adjacency, p);
  result.collider_orientations = 0;
  result.rule1_orientations = 0;
  result.rule2_orientations = 0;
  result.rule3_orientations = 0;
  result.generalized_orientations = 0;
  result.regrvonps_calls = 0;
  result.regrvonps_cuda_calls = 0;
  result.regrvonps_cpu_calls = 0;
  result.orientation_dcov_batches = 0;
  result.orientation_dcov_pairs = 0;
  result.regrvonps_dcc_gamma_tests = 0;
  result.regrvonps_hsic_gamma_tests = 0;
  result.regrvonps_hsic_perm_tests = 0;
  result.regrvonps_hsic_permutation_replicates = 0;
  result.regrvonps_hsic_gamma_cuda_tests = 0;
  result.regrvonps_hsic_perm_cuda_tests = 0;
  result.regrvonps_hsic_cuda_batches = 0;
  result.regrvonps_hsic_cuda_pairs = 0;
  result.regrvonps_hsic_cuda_fallback_tests = 0;
  result.orientation_residual_fits = 0;
  result.orientation_cuda_residual_fits = 0;
  result.orientation_cpu_fallback_fits = 0;
  result.residual_cache_requests = 0;
  result.residual_cache_hits = 0;
  result.residual_cache_computations = 0;
  result.residual_backend = options.residual_backend_name;
  result.residual_backend_params = make_residual_backend_config(
    options.residual_backend_name, options.fastspline_params).params;
  result.residual_device = options.orientation_residual_device;
  result.residual_device_requested = options.orientation_residual_device_requested;
  result.residual_device_reason = options.orientation_residual_device_reason;
  result.orientation_batch_size_requested = options.orientation_batch_size;
  result.orientation_batch_size_used = options.orientation_batch_size <= 0 ?
    0 : options.orientation_batch_size;
  result.ci_method = options.ci_method.empty() ? "dcc.gamma" : options.ci_method;
  result.ci_backend = "native-cpu";
  result.ci_backend_reason = "";

  if (pdag_has_no_edges(result.pdag)) return result;

  std::vector<int> unf_vect;
  if (options.orient_collider) {
    result.collider_orientations =
      orient_colliders(&result.pdag, p, skeleton.sepsets, options.solve_confl,
                       unf_vect, &result.events);
  }

  ResidualCache residual_cache(backend_residual_cache_options(
    options.residual_backend_name, options.fastspline_params,
    options.residual_cache_enabled));

  std::vector<std::vector<int> > neighborhoods =
    undirected_neighborhoods(result.pdag, p);
  std::vector<int> nbhd_updated(p, 0);
  int s = 1;

  while (max_neighborhood_size(neighborhoods) >= s) {
    for (int V = 0; V < p; ++V) {
      int s2 = s;
      const int size_V = static_cast<int>(neighborhoods[V].size());
      if (!(size_V == s2 || (size_V < s2 && nbhd_updated[V] != 0))) {
        continue;
      }
      s2 = size_V;
      while (s2 > 0) {
        const std::vector<std::vector<int> > subsets =
          combinations_of_size(neighborhoods[V], s2);
        bool accepted_for_V = false;
        for (std::size_t subset_index = 0; subset_index < subsets.size();
             ++subset_index) {
          const std::vector<int>& S = subsets[subset_index];
          if (!check_immor(result.pdag, p, V, S)) {
            result.events.push_back(make_generalized_event(
              "checkImmor", -1, V, S, R_NaN, false,
              "rejected by checkImmor"));
            continue;
          }

          const RegrVonPsResult pval1 =
            evaluate_regrvonps(data, result.pdag, p, V, S, options,
                               &residual_cache, evaluator);
          ++result.regrvonps_calls;
          add_regrvonps_diagnostics(&result, pval1);
          if (pval1.reject_count > 0) {
            result.events.push_back(make_generalized_event(
              "regrVonPS", -1, V, S, first_or_nan(pval1.p_values), false,
              "target residual remains dependent"));
            continue;
          }

          bool to_update = true;
          for (int W : S) {
            int s3 = static_cast<int>(neighborhoods[W].size());
            bool keep_searching = true;
            while (keep_searching && s3 > 0) {
              const std::vector<std::vector<int> > w_subsets =
                combinations_of_size(neighborhoods[W], s3);
              for (std::size_t w_subset_index = 0;
                   w_subset_index < w_subsets.size(); ++w_subset_index) {
                const std::vector<int>& S2 = w_subsets[w_subset_index];
                if (!check_immor(result.pdag, p, W, S2)) continue;
                if (!vector_contains(S2, V)) continue;
                const RegrVonPsResult pval2 =
                  evaluate_regrvonps(data, result.pdag, p, W, S2, options,
                                     &residual_cache, evaluator);
                ++result.regrvonps_calls;
                add_regrvonps_diagnostics(&result, pval2);
                if (pval2.reject_count == 0) {
                  to_update = false;
                  keep_searching = false;
                  result.events.push_back(make_generalized_event(
                    "reverseRegrVonPS", W, V, S2, first_or_nan(pval2.p_values),
                    false, "reverse residualization accepted"));
                  break;
                }
              }
              --s3;
            }
            if (!to_update) break;
          }

          if (to_update) {
            const std::vector<int> neighborhood_before = neighborhoods[V];
            apply_generalized_orientation(&result.pdag, p, V, S,
                                          neighborhood_before);
            ++result.generalized_orientations;
            result.events.push_back(make_generalized_event(
              "accept", -1, V, S, first_or_nan(pval1.p_values), true,
              "accepted generalized orientation"));

            const RuleApplicationCounts counts =
              apply_rules_until_converged(&result.pdag, p, options, unf_vect,
                                          &result.events);
            result.rule1_orientations += counts.rule1;
            result.rule2_orientations += counts.rule2;
            result.rule3_orientations += counts.rule3;
            neighborhoods = undirected_neighborhoods(result.pdag, p);
            nbhd_updated[V] = 1;
            accepted_for_V = true;
            s2 = 0;
            break;
          }
        }
        if (accepted_for_V) break;
        --s2;
      }
    }
    ++s;
  }

  const ResidualCacheStats stats = residual_cache.stats();
  if (result.residual_device == "cuda") {
    result.residual_cache_requests = result.orientation_residual_fits;
    result.residual_cache_hits = 0;
    result.residual_cache_computations = result.orientation_residual_fits;
  } else {
    result.residual_cache_requests = stats.requests;
    result.residual_cache_hits = stats.hits;
    result.residual_cache_computations = stats.computations;
  }
  return result;
}
