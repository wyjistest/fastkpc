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
  double alloc_sec = 0.0;
  double h2d_sec = 0.0;
  double memset_sec = 0.0;
  double rowsum_sec = 0.0;
  double totals_d2h_sec = 0.0;
  double reduce_sec = 0.0;
  double scalars_d2h_sec = 0.0;
  double host_scalar_sec = 0.0;
  double free_sec = 0.0;
  double total_sec = 0.0;
  int chunks = 0;
  int max_chunk_batch = 0;
  int workspace_reuse_count = 0;
  int workspace_grow_count = 0;
  int raw_aggregate_fused_count = 0;
  int row_product_reduce_count = 0;
};

#endif
