#ifndef FASTKPC_MGCV_FIXED_SP_RUNTIME_TYPES_HPP
#define FASTKPC_MGCV_FIXED_SP_RUNTIME_TYPES_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fastkpc {

enum class FixedSpRoute : int {
  CholeskyBatched = 0,
  AugmentedQr = 1,
  AugmentedSvd = 2
};

enum class FixedSpStatus : int {
  OkCholeskyBatched = 0,
  OkCholeskySingle = 1,
  OkAugmentedQr = 2,
  OkAugmentedSvd = 3,
  ErrNonfiniteInput = 10,
  ErrSpShapeOrOrder = 11,
  ErrRouteMetadata = 12,
  ErrStablePathNotImplemented = 13,
  ErrQrFailed = 14,
  ErrSvdFailed = 15,
  ErrNonfiniteOutput = 16,
  ErrInternalCuda = 17
};

struct FixedSpCapacities {
  int n = 0;
  int null_dim = 0;
  int target_count = 0;
  int penalty_count = 0;
  int augmented_rows = 0;
};

struct FixedSpRuntimeInfo {
  int device_id = -1;
  std::int64_t creator_pid = -1;
  std::uint64_t generation = 0;
  int runtime_context_create_count = 0;
  int stream_create_count = 0;
  int cublas_handle_create_count = 0;
  int cusolver_handle_create_count = 0;
  int workspace_reserve_count = 0;
  int workspace_grow_count = 0;
  int cuda_device_synchronize_count = 0;
  int cholesky_factor_checkpoint_record_count = 0;
  int cholesky_factor_checkpoint_wait_count = 0;
  int cholesky_solve_checkpoint_record_count = 0;
  int cholesky_solve_checkpoint_wait_count = 0;
  std::size_t workspace_bytes = 0;
  std::size_t cublas_workspace_bytes = 0;
  std::size_t cublas_workspace_alignment = 0;
  int cuda_toolkit_version = 0;
  int cuda_driver_version = 0;
  int compute_capability_major = 0;
  int compute_capability_minor = 0;
  int sm_count = 0;
  bool cusolver_deterministic_mode_enabled = false;
  bool cublas_pedantic_math_enabled = false;
  bool cublas_atomics_not_allowed = false;
  bool cublas_user_workspace_installed = false;
  bool freed = false;
};

struct PreparedSHostView {
  std::string dataset_sha256;
  std::string prepared_s_key_sha256;
  std::string same_s_group_id;
  std::string semantic_fingerprint;
  std::string representation_fingerprint;
  int n = 0;
  int coefficient_dim = 0;
  int null_dim = 0;
  int penalty_count = 0;
  const double* X = nullptr;
  const double* Z = nullptr;
  const double* gram = nullptr;
  std::vector<const double*> penalty_blocks;
  std::vector<int> penalty_dimensions;
  std::vector<int> penalty_offsets_zero_based;
  std::vector<int> penalty_ranks;
  std::vector<int> penalty_sp_indices_zero_based;
};

struct PreparedSInfo {
  std::string prepared_s_key_sha256;
  int n = 0;
  int coefficient_dim = 0;
  int null_dim = 0;
  int penalty_count = 0;
  int setup_h2d_upload_count = 0;
  std::size_t setup_h2d_bytes = 0;
  std::uint64_t generation = 0;
  bool output_slot_leased = false;
};

struct PreparedSStaticShadow {
  int n = 0;
  int null_dim = 0;
  int penalty_count = 0;
  std::vector<double> X_null;
  std::vector<double> gram;
  std::vector<double> projected_penalties;
};

const char* fixed_sp_status_name(FixedSpStatus status);
const char* fixed_sp_route_name(FixedSpRoute route);

}  // namespace fastkpc

#endif
