#ifndef FASTKPC_LEGACY_DCOV_SPECTRA_MATVEC_CUDA_HPP
#define FASTKPC_LEGACY_DCOV_SPECTRA_MATVEC_CUDA_HPP

#include <vector>
#include <cstddef>

namespace fastkpc {

struct LegacyDcovSpectraMatvecCudaHandle;

struct LegacyDcovSpectraMatvecCudaResult {
  std::vector<double> values;
  int n = 0;
  int rhs_count = 0;
  int kernel_launch_count = 0;
  int device_matrix_reuse_count = 0;
  int device_workspace_reuse_count = 0;
  int workspace_realloc_count = 0;
  std::size_t matrix_bytes = 0;
  std::size_t workspace_bytes = 0;
  std::size_t d2h_bytes = 0;
  double matrix_h2d_ms = 0.0;
  double workspace_alloc_ms = 0.0;
  double alloc_ms = 0.0;
  double h2d_ms = 0.0;
  double kernel_ms = 0.0;
  double d2h_ms = 0.0;
  double free_ms = 0.0;
  double total_ms = 0.0;
};

LegacyDcovSpectraMatvecCudaResult legacy_dcov_spectra_matvec_cuda(
    const double* matrix,
    const double* rhs,
    int n,
    int rhs_count);

LegacyDcovSpectraMatvecCudaHandle*
legacy_dcov_spectra_matvec_cuda_handle_create(
    const double* matrix,
    int n);

void legacy_dcov_spectra_matvec_cuda_handle_destroy(
    LegacyDcovSpectraMatvecCudaHandle* handle);

int legacy_dcov_spectra_matvec_cuda_handle_n(
    const LegacyDcovSpectraMatvecCudaHandle* handle);

std::size_t legacy_dcov_spectra_matvec_cuda_handle_matrix_bytes(
    const LegacyDcovSpectraMatvecCudaHandle* handle);

double legacy_dcov_spectra_matvec_cuda_handle_matrix_h2d_ms(
    const LegacyDcovSpectraMatvecCudaHandle* handle);

LegacyDcovSpectraMatvecCudaResult
legacy_dcov_spectra_matvec_cuda_handle_apply(
    LegacyDcovSpectraMatvecCudaHandle* handle,
    const double* rhs,
    int rhs_count);

LegacyDcovSpectraMatvecCudaResult
legacy_dcov_spectra_matvec_cuda_handle_project(
    LegacyDcovSpectraMatvecCudaHandle* handle,
    const double* basis,
    int basis_count);

}  // namespace fastkpc

#endif
