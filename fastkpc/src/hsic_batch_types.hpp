#ifndef FASTKPC_HSIC_BATCH_TYPES_HPP
#define FASTKPC_HSIC_BATCH_TYPES_HPP

#include <cstddef>
#include <string>
#include <vector>

struct HsicBatchOptions {
  double sig;
  int permutation_replicates;
  bool include_observed;
  bool has_seed;
  unsigned int seed;
  bool return_replicates;
  int max_n;
  int max_batch_pairs;
};

struct HsicBatchDiagnostics {
  std::string backend;
  std::string reason;
  int n;
  int pairs;
  int batches;
  int permutation_replicates;
  bool used_seed;
  unsigned int seed;
  std::size_t bytes_allocated;
  int cuda_blocks;
  int cuda_threads;
};

struct HsicBatchResult {
  std::vector<double> statistics;
  std::vector<double> p_values;
  std::vector<double> means;
  std::vector<double> variances;
  std::vector<double> shapes;
  std::vector<double> scales;
  std::vector<double> permutation_replicates;
  HsicBatchDiagnostics diagnostics;
};

HsicBatchOptions default_hsic_batch_options();

#endif
