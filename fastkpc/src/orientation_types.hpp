#ifndef FASTKPC_ORIENTATION_TYPES_HPP
#define FASTKPC_ORIENTATION_TYPES_HPP

#include "fastspline_basis.hpp"
#include "hsic_cpu.hpp"

#include <string>
#include <vector>

static const int FASTKPC_EDGE_NONE = 0;
static const int FASTKPC_EDGE_PRESENT = 1;
static const int FASTKPC_EDGE_CONFLICT = 2;

struct OrientationEvent {
  std::string phase;
  std::string rule;
  int x;
  int y;
  int z;
  std::vector<int> S;
  double p_value;
  bool accepted;
  std::string message;
};

struct OrientationOptions {
  double alpha;
  bool verbose;
  bool solve_confl;
  bool orient_collider;
  bool rule1;
  bool rule2;
  bool rule3;
  bool residual_cache_enabled;
  std::string residual_backend_name;
  std::string orientation_residual_device_requested;
  std::string orientation_residual_device;
  std::string orientation_residual_device_reason;
  int orientation_batch_size;
  bool orientation_diagnostics_enabled;
  bool cuda_residual_fallback;
  FastSplineParams fastspline_params;
  double index;
  bool legacy_index;
  std::string ci_method;
  HsicOptions hsic_options;
  bool ci_diagnostics_enabled;
};

struct OrientationResult {
  std::vector<int> pdag;
  int p;
  std::vector<OrientationEvent> events;
  int collider_orientations;
  int rule1_orientations;
  int rule2_orientations;
  int rule3_orientations;
  int generalized_orientations;
  int regrvonps_calls;
  int regrvonps_cuda_calls;
  int regrvonps_cpu_calls;
  int orientation_dcov_batches;
  int orientation_dcov_pairs;
  int regrvonps_dcc_gamma_tests;
  int regrvonps_hsic_gamma_tests;
  int regrvonps_hsic_perm_tests;
  int regrvonps_hsic_permutation_replicates;
  int regrvonps_hsic_gamma_cuda_tests;
  int regrvonps_hsic_perm_cuda_tests;
  int regrvonps_hsic_cuda_batches;
  int regrvonps_hsic_cuda_pairs;
  int regrvonps_hsic_cuda_fallback_tests;
  int orientation_residual_fits;
  int orientation_cuda_residual_fits;
  int orientation_cpu_fallback_fits;
  int residual_cache_requests;
  int residual_cache_hits;
  int residual_cache_computations;
  std::string residual_backend;
  std::string residual_backend_params;
  std::string residual_device;
  std::string residual_device_requested;
  std::string residual_device_reason;
  int orientation_batch_size_requested;
  int orientation_batch_size_used;
  std::string ci_method;
  std::string ci_backend;
  std::string ci_backend_reason;
};

#endif
