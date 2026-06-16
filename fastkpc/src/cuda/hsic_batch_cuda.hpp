#ifndef FASTKPC_CUDA_HSIC_BATCH_CUDA_HPP
#define FASTKPC_CUDA_HSIC_BATCH_CUDA_HPP

#include "../hsic_batch_types.hpp"

#include <string>

HsicBatchResult hsic_gamma_batch_cuda(const double* x,
                                      const double* y,
                                      int n,
                                      int pairs,
                                      const HsicBatchOptions& options);

HsicBatchResult hsic_permutation_batch_cuda(const double* x,
                                            const double* y,
                                            int n,
                                            int pairs,
                                            const HsicBatchOptions& options);

bool hsic_cuda_available(std::string* reason);

#endif
