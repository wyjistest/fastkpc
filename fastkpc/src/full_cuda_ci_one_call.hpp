#ifndef FASTKPC_FULL_CUDA_CI_ONE_CALL_HPP
#define FASTKPC_FULL_CUDA_CI_ONE_CALL_HPP

#include <Rcpp.h>

#include <string>

namespace fastkpc {

Rcpp::List full_cuda_ci_one_call_skeleton(
    const Rcpp::NumericMatrix& data,
    double alpha,
    int max_conditioning_size,
    double index,
    int num_col,
    const std::string& trace_level,
    bool compatible_cuda_strict);

}  // namespace fastkpc

#endif
