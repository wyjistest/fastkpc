#ifndef FASTKPC_TYPES_HPP
#define FASTKPC_TYPES_HPP

#include <cstddef>
#include <string>
#include <vector>

#include "hsic_cpu.hpp"
#include "fastspline_basis.hpp"
#include "skeleton_task_scheduler.hpp"

struct CiTask {
  int x;
  int y;
  std::vector<int> conditioning_set;
};

struct CiResult {
  int x;
  int y;
  std::vector<int> conditioning_set;
  double p_value;
};

struct SkeletonOptions {
  double alpha;
  int max_conditioning_size;
  bool na_delete;
  bool stable;
  double index;
  bool legacy_index;
  bool residual_cache_enabled;
  std::string residual_backend_name;
  std::string residual_device_requested;
  bool cuda_residual_fallback;
  std::string scheduler_requested;
  int residual_batch_size;
  bool scheduler_diagnostics_enabled;
  FastSplineParams fastspline_params;
  std::string ci_method;
  HsicOptions hsic_options;
  bool ci_diagnostics_enabled;
};

struct LevelDeletion {
  int x;
  int y;
  std::vector<int> conditioning_set;
  double p_value;
};

struct SkeletonResult {
  std::vector<int> adjacency;
  std::vector<std::vector<std::vector<int> > > sepsets;
  std::vector<double> pmax;
  std::vector<int> n_edge_tests;
  std::vector<std::vector<LevelDeletion> > per_level_log;
  bool residual_cache_enabled;
  int residual_cache_requests;
  int residual_cache_hits;
  int residual_cache_misses;
  int residual_cache_computations;
  int residual_cache_stored_vectors;
  int residual_cache_stored_values;
  std::string residual_backend;
  std::string residual_backend_params;
  std::string residual_device;
  std::string residual_device_requested;
  std::string residual_device_reason;
  std::string scheduler;
  std::string scheduler_requested;
  SchedulerDiagnostics scheduler_diagnostics;
  std::string ci_method;
  std::string ci_backend;
  std::string ci_backend_reason;
  int ci_dcc_gamma_tests;
  int ci_hsic_gamma_tests;
  int ci_hsic_perm_tests;
  int ci_hsic_permutation_replicates;
  int ci_hsic_gamma_cuda_tests;
  int ci_hsic_perm_cuda_tests;
  int ci_hsic_cuda_batches;
  int ci_hsic_cuda_pairs;
  int ci_hsic_cuda_fallback_tests;
  std::size_t ci_hsic_cuda_memory_bytes;
  int ci_hsic_cuda_max_n;
  int ci_hsic_cuda_max_batch_pairs;
};

#endif
