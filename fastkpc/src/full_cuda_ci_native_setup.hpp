#ifndef FASTKPC_FULL_CUDA_CI_NATIVE_SETUP_HPP
#define FASTKPC_FULL_CUDA_CI_NATIVE_SETUP_HPP

#include <Rcpp.h>

#include <memory>
#include <vector>

namespace fastkpc {

class NativeUnivariateSmooth;

struct NativeSetupProfile {
  double input_validation_ms = 0.0;
  double smooth_build_ms = 0.0;
  double block_assembly_ms = 0.0;
  double gram_ms = 0.0;
  double fingerprint_ms = 0.0;
  double setup_packaging_ms = 0.0;
  double geometry_input_ms = 0.0;
  double geometry_qr_ms = 0.0;
  double geometry_penalty_ms = 0.0;
  double geometry_initial_sp_ms = 0.0;
  double geometry_packaging_ms = 0.0;
};

Rcpp::List full_cuda_ci_native_setup(
    const Rcpp::NumericMatrix& conditioning,
    NativeSetupProfile* profile = nullptr);

std::shared_ptr<const NativeUnivariateSmooth>
full_cuda_ci_native_univariate_smooth(
    const Rcpp::NumericMatrix& data,
    int source_column,
    NativeSetupProfile* profile = nullptr);

Rcpp::List full_cuda_ci_native_additive_setup(
    const Rcpp::NumericMatrix& conditioning,
    const std::vector<int>& source_columns,
    const std::vector<std::shared_ptr<const NativeUnivariateSmooth>>& smooths,
    NativeSetupProfile* profile = nullptr);

Rcpp::List full_cuda_ci_native_geometry_prepare(
    const Rcpp::NumericMatrix& X,
    const Rcpp::List& penalty_blocks,
    const Rcpp::IntegerVector& penalty_offsets,
    const Rcpp::IntegerVector& penalty_ranks,
    NativeSetupProfile* profile = nullptr);

}  // namespace fastkpc

#endif
