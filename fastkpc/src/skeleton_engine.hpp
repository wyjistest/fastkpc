#ifndef FASTKPC_SKELETON_ENGINE_HPP
#define FASTKPC_SKELETON_ENGINE_HPP

#include "fastkpc_types.hpp"

#include <Rcpp.h>

SkeletonResult run_skeleton_exact(const Rcpp::NumericMatrix& data,
                                  const SkeletonOptions& options);

#endif
