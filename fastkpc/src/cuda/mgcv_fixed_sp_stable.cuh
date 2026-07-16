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
  int* info = nullptr;
  int eigen_lwork = 0;
  int qr_lwork = 0;
  int ormqr_lwork = 0;
  int svd_lwork = 0;
  int max_rows = 0;
  int max_q = 0;
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

}  // namespace fastkpc

#endif
