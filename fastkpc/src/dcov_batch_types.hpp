#ifndef FASTKPC_DCOV_BATCH_TYPES_HPP
#define FASTKPC_DCOV_BATCH_TYPES_HPP

#include <vector>

struct DcovBatchOptions {
  double index;
  bool legacy_index;
};

struct DcovBatchResult {
  std::vector<double> p_values;
  std::vector<double> nV2;
  std::vector<double> means;
  std::vector<double> variances;
  std::vector<double> raw_scalars;
};

#endif
