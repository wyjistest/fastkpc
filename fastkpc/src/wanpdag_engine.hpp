#ifndef FASTKPC_WANPDAG_ENGINE_HPP
#define FASTKPC_WANPDAG_ENGINE_HPP

#include "fastkpc_types.hpp"
#include "orientation_types.hpp"
#include "regrvonps_native.hpp"

#include <Rcpp.h>

OrientationOptions default_orientation_options();

OrientationResult orient_wanpdag_native(
  const Rcpp::NumericMatrix& data,
  const SkeletonResult& skeleton,
  const OrientationOptions& options,
  RegrVonPsEvaluator evaluator = NULL);

#endif
