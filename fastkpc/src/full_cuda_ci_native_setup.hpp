#ifndef FASTKPC_FULL_CUDA_CI_NATIVE_SETUP_HPP
#define FASTKPC_FULL_CUDA_CI_NATIVE_SETUP_HPP

#include <Rcpp.h>

namespace fastkpc {

Rcpp::List full_cuda_ci_native_setup(const Rcpp::NumericMatrix& conditioning);

Rcpp::List full_cuda_ci_native_geometry_prepare(
    const Rcpp::NumericMatrix& X,
    const Rcpp::List& penalty_blocks,
    const Rcpp::IntegerVector& penalty_offsets,
    const Rcpp::IntegerVector& penalty_ranks);

}  // namespace fastkpc

#endif
