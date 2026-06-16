#ifndef FASTKPC_REGRVONPS_NATIVE_HPP
#define FASTKPC_REGRVONPS_NATIVE_HPP

#include "orientation_types.hpp"
#include "residual_cache.hpp"

#include <Rcpp.h>
#include <vector>

struct RegrVonPsResult {
  int reject_count;
  std::vector<double> p_values;
  std::vector<int> parents;
  std::vector<int> conditioning_set;
  bool used_cuda;
  bool used_cpu_fallback;
  int dcov_batches;
  int dcov_pairs;
  int dcc_gamma_tests;
  int hsic_gamma_tests;
  int hsic_perm_tests;
  int hsic_permutation_replicates;
  int hsic_gamma_cuda_tests;
  int hsic_perm_cuda_tests;
  int hsic_cuda_batches;
  int hsic_cuda_pairs;
  int hsic_cuda_fallback_tests;
  int residual_fits;
  int cuda_residual_fits;
  int cpu_fallback_fits;
  int cache_requests_before;
  int cache_requests_after;
  int cache_hits_before;
  int cache_hits_after;
  std::string ci_backend;
  std::string ci_backend_reason;
};

typedef RegrVonPsResult (*RegrVonPsEvaluator)(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& pdag,
  int p,
  int V,
  const std::vector<int>& S,
  const OrientationOptions& options,
  ResidualCache* residual_cache);

std::vector<int> parents_of(const std::vector<int>& pdag, int p, int V);

std::vector<int> sorted_unique_union(const std::vector<int>& a,
                                     const std::vector<int>& b);

RegrVonPsResult regrvonps_native(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& pdag,
  int p,
  int V,
  const std::vector<int>& S,
  const OrientationOptions& options,
  ResidualCache* residual_cache);

#endif
