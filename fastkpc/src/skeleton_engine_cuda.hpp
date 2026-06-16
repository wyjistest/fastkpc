#ifndef FASTKPC_SKELETON_ENGINE_CUDA_HPP
#define FASTKPC_SKELETON_ENGINE_CUDA_HPP

#include "fastkpc_types.hpp"

#include <Rcpp.h>

SkeletonResult run_skeleton_cuda_batch(const Rcpp::NumericMatrix& data,
                                       const SkeletonOptions& options,
                                       int batch_size);

#endif
