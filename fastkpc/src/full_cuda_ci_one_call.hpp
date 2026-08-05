#ifndef FASTKPC_FULL_CUDA_CI_ONE_CALL_HPP
#define FASTKPC_FULL_CUDA_CI_ONE_CALL_HPP

#include <Rcpp.h>

#include <string>

namespace fastkpc {

struct FullCudaCiOneCallMethodOptions {
  std::string ci_method = "dcc.gamma";
  double hsic_sig = 1.0;
  int permutation_replicates = 100;
  bool permutation_include_observed = true;
  bool permutation_has_seed = false;
  unsigned int permutation_seed = 0U;
};

Rcpp::List full_cuda_ci_one_call_skeleton(
    const Rcpp::NumericMatrix& data,
    double alpha,
    int max_conditioning_size,
    double index,
    int num_col,
    const std::string& trace_level,
    bool compatible_cuda_strict);

Rcpp::List full_cuda_ci_one_call_skeleton_method(
    const Rcpp::NumericMatrix& data,
    double alpha,
    int max_conditioning_size,
    double index,
    int num_col,
    const std::string& trace_level,
    bool compatible_cuda_strict,
    const FullCudaCiOneCallMethodOptions& method_options);

Rcpp::List full_cuda_ci_one_call_cache_control(
    const std::string& action,
    int capacity);

Rcpp::List full_cuda_ci_one_call_cache_state(
    const Rcpp::NumericMatrix& data);

}  // namespace fastkpc

#endif
