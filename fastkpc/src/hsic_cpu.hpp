#ifndef FASTKPC_HSIC_CPU_HPP
#define FASTKPC_HSIC_CPU_HPP

#include <string>
#include <vector>

struct HsicOptions {
  double sig;
  int replicates;
  bool include_observed;
  bool has_seed;
  unsigned int seed;
  bool return_replicates;
  int cuda_max_n;
  int cuda_max_batch_pairs;
  bool cuda_memory_fallback;
};

struct HsicResult {
  double statistic;
  double p_value;
  double mean;
  double variance;
  double shape;
  double scale;
  int n;
  int replicates;
  bool used_seed;
  unsigned int seed;
  std::string method;
  std::string reason;
  std::vector<double> replicate_statistics;
};

HsicOptions default_hsic_options();

HsicResult hsic_gamma_cpu(const std::vector<double>& x,
                          const std::vector<double>& y,
                          const HsicOptions& options);

HsicResult hsic_permutation_cpu(const std::vector<double>& x,
                                const std::vector<double>& y,
                                const HsicOptions& options);

#endif
