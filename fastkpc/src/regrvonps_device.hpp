#ifndef FASTKPC_REGRVONPS_DEVICE_HPP
#define FASTKPC_REGRVONPS_DEVICE_HPP

#include "regrvonps_native.hpp"

RegrVonPsResult regrvonps_device(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& pdag,
  int p,
  int V,
  const std::vector<int>& S,
  const OrientationOptions& options,
  ResidualCache* residual_cache);

#endif
