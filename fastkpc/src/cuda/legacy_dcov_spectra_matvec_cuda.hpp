#ifndef FASTKPC_LEGACY_DCOV_SPECTRA_MATVEC_CUDA_HPP
#define FASTKPC_LEGACY_DCOV_SPECTRA_MATVEC_CUDA_HPP

#include <vector>

namespace fastkpc {

struct LegacyDcovSpectraMatvecCudaResult {
  std::vector<double> values;
  int n = 0;
  int rhs_count = 0;
  int kernel_launch_count = 0;
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

}  // namespace fastkpc

#endif
