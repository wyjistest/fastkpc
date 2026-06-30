#ifndef FASTKPC_DCOV_BATCH_CUDA_HPP
#define FASTKPC_DCOV_BATCH_CUDA_HPP

#include "../dcov_batch_types.hpp"

struct DcovCudaWorkspace;

DcovCudaWorkspace* create_dcov_cuda_workspace();

void destroy_dcov_cuda_workspace(DcovCudaWorkspace* workspace);

DcovBatchResult dcov_batch_cuda(const double* x,
                                const double* y,
                                int n,
                                int batch,
                                const DcovBatchOptions& options);

DcovBatchResult dcov_batch_cuda(const double* x,
                                const double* y,
                                int n,
                                int batch,
                                const DcovBatchOptions& options,
                                DcovCudaWorkspace* workspace);

DcovBatchResult dcov_batch_cuda_pvalues_into(const double* x,
                                             const double* y,
                                             int n,
                                             int batch,
                                             const DcovBatchOptions& options,
                                             DcovCudaWorkspace* workspace,
                                             double* out_pvalues);

#endif
