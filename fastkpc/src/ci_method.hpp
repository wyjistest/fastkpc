#ifndef FASTKPC_CI_METHOD_HPP
#define FASTKPC_CI_METHOD_HPP

#include "hsic_cpu.hpp"

#include <string>
#include <vector>

enum class CiMethodKind {
  DccGamma,
  HsicGamma,
  HsicPermutation
};

struct CiEvaluation {
  CiMethodKind kind;
  std::string method;
  double p_value;
  double statistic;
  double hsic_mean;
  double hsic_variance;
  int permutation_replicates;
};

CiMethodKind parse_ci_method_kind(const std::string& method);
std::string ci_method_name(CiMethodKind kind);

CiEvaluation evaluate_ci_vectors(const std::vector<double>& x,
                                 const std::vector<double>& y,
                                 CiMethodKind kind,
                                 double index,
                                 bool legacy_index,
                                 const HsicOptions& hsic_options);

#endif
