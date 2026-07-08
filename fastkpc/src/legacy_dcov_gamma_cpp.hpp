#ifndef FASTKPC_LEGACY_DCOV_GAMMA_CPP_HPP
#define FASTKPC_LEGACY_DCOV_GAMMA_CPP_HPP

#include <Rcpp.h>

namespace fastkpc {

struct LegacyDcovLowrankTimings {
  double eig_ms = 0.0;
  double select_ms = 0.0;
  double center_ms = 0.0;
  int full_eig_count = 0;
  int spectra_count = 0;
  int spectra_converged_count = 0;
  int spectra_failed_count = 0;
  int spectra_fallback_full_eig_count = 0;
  int spectra_iterations = 0;
  int spectra_nconv = 0;
  int spectra_ncv = 0;
  double spectra_tol = 0.0;
};

enum class LegacyDcovLowrankMode {
  FullEig,
  Spectra
};

struct LegacyDcovGammaCppResult {
  double p_value = NA_REAL;
  double nV2 = NA_REAL;
  double mean = NA_REAL;
  double variance = NA_REAL;
  double statistic = NA_REAL;
  double estimate = NA_REAL;
  int n = 0;
  int num_col = 0;
  double index = 1.0;
  LegacyDcovLowrankMode lowrank_mode = LegacyDcovLowrankMode::FullEig;
  double input_ms = 0.0;
  double distance_ms = 0.0;
  double lowrank_ms = 0.0;
  double lowrank_unaccounted_ms = 0.0;
  double statistic_ms = 0.0;
  double moment_ms = 0.0;
  double pgamma_ms = 0.0;
  double accounted_ms = 0.0;
  double total_ms = 0.0;
  LegacyDcovLowrankTimings lowrank_timings;
};

LegacyDcovLowrankMode legacy_dcov_lowrank_mode_from_env();
const char* legacy_dcov_lowrank_mode_name(LegacyDcovLowrankMode mode);

LegacyDcovGammaCppResult legacy_dcov_gamma_cpp_compute(
    Rcpp::NumericVector x,
    Rcpp::NumericVector y,
    int numCol,
    double index = 1.0);

Rcpp::List legacy_dcov_gamma_cpp_result_to_list(
    const LegacyDcovGammaCppResult& result);

}  // namespace fastkpc

#endif  // FASTKPC_LEGACY_DCOV_GAMMA_CPP_HPP
