#ifndef FASTKPC_DCOV_BATCH_TYPES_HPP
#define FASTKPC_DCOV_BATCH_TYPES_HPP

#include <cstddef>
#include <vector>

struct DcovBatchOptions {
  double index = 1.0;
  bool legacy_index = true;
  int permutation_replicates = 0;
  bool include_observed = true;
  bool has_seed = false;
  unsigned int seed = 0U;
  bool return_replicates = false;
  int max_n = 2048;
  int max_batch_pairs = 64;
};

struct DcovBatchResult {
  std::vector<double> p_values;
  std::vector<double> nV2;
  std::vector<double> means;
  std::vector<double> variances;
  std::vector<double> raw_scalars;
  std::vector<double> permutation_statistics;
  int permutation_replicates = 0;
  bool permutation_used_seed = false;
  unsigned int permutation_seed = 0U;
  std::size_t permutation_bytes = 0;
  double permutation_h2d_sec = 0.0;
  double permutation_reduce_sec = 0.0;
  double permutation_d2h_sec = 0.0;
  double permutation_host_sec = 0.0;
  double alloc_sec = 0.0;
  double h2d_sec = 0.0;
  double memset_sec = 0.0;
  double rowsum_sec = 0.0;
  double totals_d2h_sec = 0.0;
  double reduce_sec = 0.0;
  double scalars_d2h_sec = 0.0;
  double host_scalar_sec = 0.0;
  double result_materialize_sec = 0.0;
  double free_sec = 0.0;
  double total_sec = 0.0;
  double top_level_wall_sec = 0.0;
  double grid_limit_query_sec = 0.0;
  double chunk_dispatch_sec = 0.0;
  double top_level_unaccounted_sec = 0.0;
  int chunks = 0;
  int max_chunk_batch = 0;
  int workspace_reuse_count = 0;
  int workspace_grow_count = 0;
  int raw_aggregate_fused_count = 0;
  int rowsum_kernel_launch_count = 0;
  int rowsum_chunk_count = 0;
  double rowsum_total_blocks = 0.0;
  double rowsum_pair_count = 0.0;
  int rowsum_abs_fast_count = 0;
  int rowsum_pow_generic_count = 0;
  double rowsum_abs_pair_count = 0.0;
  double rowsum_generic_pair_count = 0.0;
  int rowsum_threads = 0;
  int rowsum_n_max = 0;
  int rowsum_batch_total = 0;
  int rowsum_max_chunk_batch = 0;
  double rowsum_max_chunk_sec = 0.0;
  int rowsum_max_chunk_n = 0;
  int row_product_reduce_count = 0;
  int pvalue_only_count = 0;
  int full_result_materialize_count = 0;
  int grid_limit_query_count = 0;
  int grid_limit_cache_hit_count = 0;
  int grid_limit_process_cache_hit_count = 0;
};

#endif
