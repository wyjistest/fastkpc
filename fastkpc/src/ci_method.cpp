#include "ci_method.hpp"

#include "dcov_exact_cpu.hpp"

#include <stdexcept>

CiMethodKind parse_ci_method_kind(const std::string& method) {
  if (method.empty() || method == "dcc.gamma") return CiMethodKind::DccGamma;
  if (method == "hsic.gamma") return CiMethodKind::HsicGamma;
  if (method == "hsic.perm") return CiMethodKind::HsicPermutation;
  throw std::runtime_error("Unknown ci_method: " + method);
}

std::string ci_method_name(CiMethodKind kind) {
  switch (kind) {
    case CiMethodKind::DccGamma:
      return "dcc.gamma";
    case CiMethodKind::HsicGamma:
      return "hsic.gamma";
    case CiMethodKind::HsicPermutation:
      return "hsic.perm";
  }
  return "dcc.gamma";
}

CiEvaluation evaluate_ci_vectors(const std::vector<double>& x,
                                 const std::vector<double>& y,
                                 CiMethodKind kind,
                                 double index,
                                 bool legacy_index,
                                 const HsicOptions& hsic_options) {
  CiEvaluation out;
  out.kind = kind;
  out.method = ci_method_name(kind);
  out.p_value = 1.0;
  out.statistic = 0.0;
  out.hsic_mean = 0.0;
  out.hsic_variance = 0.0;
  out.permutation_replicates = 0;

  if (kind == CiMethodKind::DccGamma) {
    out.p_value = dcov_exact_pvalue(x, y, index, legacy_index);
    return out;
  }

  if (kind == CiMethodKind::HsicGamma) {
    const HsicResult result = hsic_gamma_cpu(x, y, hsic_options);
    out.p_value = result.p_value;
    out.statistic = result.statistic;
    out.hsic_mean = result.mean;
    out.hsic_variance = result.variance;
    return out;
  }

  const HsicResult result = hsic_permutation_cpu(x, y, hsic_options);
  out.p_value = result.p_value;
  out.statistic = result.statistic;
  out.hsic_mean = result.mean;
  out.hsic_variance = result.variance;
  out.permutation_replicates = result.replicates;
  return out;
}
