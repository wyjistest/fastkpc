#ifndef FASTKPC_MGCV_FIXED_SP_STABLE_CUH
#define FASTKPC_MGCV_FIXED_SP_STABLE_CUH

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cstddef>

namespace fastkpc {

struct FixedSpStableWorkspace {
  double* B = nullptr;
  double* c = nullptr;
  double* tau = nullptr;
  double* eigen_work = nullptr;
  double* qr_work = nullptr;
  double* svd_work = nullptr;
  double* singular_values = nullptr;
  double* U = nullptr;
  double* V = nullptr;
  double* scaled_projection = nullptr;
  double* diagonal = nullptr;
  double* aggregate_penalty_factor = nullptr;
  double* aggregate_factor_work = nullptr;
  int* info = nullptr;
  int eigen_lwork = 0;
  int qr_lwork = 0;
  int ormqr_lwork = 0;
  int svd_lwork = 0;
  int max_rows = 0;
  int max_q = 0;
};

struct FixedSpRootValidationRecord {
  int solver_info = 0;
  int finite = 0;
  int ascending = 0;
  int psd = 0;
  int rank = 0;
};

struct AugmentedSystemView {
  double* B = nullptr;
  double* c = nullptr;
  int leading_dimension = 0;
  int rows = 0;
  int cols = 0;
  int target_index = -1;
};

std::size_t fixed_sp_stable_probe_double_count(int max_rows, int max_q);
FixedSpStableWorkspace fixed_sp_stable_probe_view(
  double* storage, int max_rows, int max_q);
void query_fixed_sp_stable_workspace(
  cusolverDnHandle_t solver,
  gesvdjInfo_t svd_params,
  FixedSpStableWorkspace* workspace);
std::size_t fixed_sp_stable_workspace_double_count(
  int max_rows,
  int max_q,
  int eigen_lwork,
  int qr_lwork,
  int ormqr_lwork,
  int svd_lwork);
FixedSpStableWorkspace fixed_sp_stable_workspace_view(
  double* storage,
  int* info,
  int max_rows,
  int max_q,
  int eigen_lwork,
  int qr_lwork,
  int ormqr_lwork,
  int svd_lwork);
void launch_fixed_sp_root_build(
  const double* eigenvectors,
  const double* eigenvalues,
  const int* solver_info,
  int q,
  double* roots,
  int root_leading_dimension,
  int root_row_offset,
  int root_row_capacity,
  int expected_rank,
  double epsilon,
  FixedSpRootValidationRecord* validation,
  cudaStream_t stream);
AugmentedSystemView build_fixed_sp_augmented_system(
  const double* X_null,
  const double* penalty_roots,
  const int* penalty_root_offsets,
  const int* penalty_root_ranks,
  int total_penalty_root_rows,
  int penalty_count,
  const double* H_root,
  int H_root_rank,
  const double* Y,
  const double* SP,
  const double* host_SP,
  int n,
  int q,
  int target_index,
  FixedSpStableWorkspace* workspace,
  cudaStream_t stream);
void launch_fixed_sp_qr_rank_status(
  const double* R,
  const double* theta,
  int rows,
  int q,
  int target_index,
  double epsilon,
  int* qr_rank,
  int* reroute,
  int* finite_status,
  cudaStream_t stream);
void launch_fixed_sp_svd_rank_scale(
  const double* singular_values,
  double* scaled_projection,
  const int* svd_info,
  int rows,
  int q,
  int target_index,
  double epsilon,
  int* effective_rank,
  double* sigma_max,
  double* smallest_retained_sigma,
  cudaStream_t stream);
void launch_fixed_sp_aggregate_diagnostics_init(
  int target_capacity,
  int q,
  int* aggregate_root_rank,
  int* aggregate_factor_call_count,
  int* aggregate_b_build_count,
  int* aggregate_pivots,
  double* aggregate_dstop,
  cudaStream_t stream);
void launch_fixed_sp_aggregate_factor(
  const double* projected_penalties,
  const double* projected_H,
  const double* SP,
  int penalty_count,
  int q,
  double* aggregate_penalty_factor,
  double* aggregate_factor_work,
  int* aggregate_root_rank,
  int* aggregate_factor_call_count,
  int* aggregate_pivots,
  double* aggregate_dstop,
  cudaStream_t stream);
AugmentedSystemView build_fixed_sp_aggregate_augmented_system(
  const double* X_null,
  const double* aggregate_penalty_factor,
  const int* aggregate_root_rank,
  const int* aggregate_pivots,
  const double* Y,
  int n,
  int q,
  int target_index,
  int* aggregate_b_build_count,
  FixedSpStableWorkspace* workspace,
  cudaStream_t stream);

}  // namespace fastkpc

#endif
