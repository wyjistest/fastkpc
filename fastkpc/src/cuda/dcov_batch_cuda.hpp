#ifndef FASTKPC_DCOV_BATCH_CUDA_HPP
#define FASTKPC_DCOV_BATCH_CUDA_HPP

#include "../dcov_batch_types.hpp"

DcovBatchResult dcov_batch_cuda(const double* x,
                                const double* y,
                                int n,
                                int batch,
                                const DcovBatchOptions& options);

#endif
