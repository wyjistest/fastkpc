#ifndef FASTKPC_DCOV_EXACT_CPU_HPP
#define FASTKPC_DCOV_EXACT_CPU_HPP

#include <Rcpp.h>
#include <string>
#include <vector>

struct DcovPermutationResult {
  double statistic;
  double p_value;
  int n;
  int replicates;
  bool used_seed;
  unsigned int seed;
  std::string method;
  std::vector<double> replicate_statistics;
};

double dcov_exact_pvalue(const std::vector<double>& x,
                         const std::vector<double>& y,
                         double index,
                         bool legacy_index);

DcovPermutationResult dcov_permutation_cpu(
    const std::vector<double>& x,
    const std::vector<double>& y,
    double index,
    bool legacy_index,
    int replicates,
    bool include_observed,
    bool has_seed,
    unsigned int seed,
    bool return_replicates);

std::vector<double> residualize_lm(const Rcpp::NumericMatrix& data,
                                   int target,
                                   const std::vector<int>& conditioning_set);

#endif
