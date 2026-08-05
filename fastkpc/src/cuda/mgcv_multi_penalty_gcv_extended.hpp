#ifndef FASTKPC_MGCV_MULTI_PENALTY_GCV_EXTENDED_HPP
#define FASTKPC_MGCV_MULTI_PENALTY_GCV_EXTENDED_HPP

#include "mgcv_multi_penalty_gcv.hpp"

#include <memory>
#include <vector>

namespace fastkpc {

#define FASTKPC_DECLARE_MULTI_PENALTY_EXTENDED_BACKEND(SUFFIX)             \
  class MultiPenaltyGcvCudaPrepared##SUFFIX;                              \
  class MultiPenaltyGcvCudaResidualBatch##SUFFIX;                         \
  struct MultiPenaltyGcvCudaBatchResult##SUFFIX {                         \
    MultiPenaltyGcvCudaOptimization optimization;                         \
    std::shared_ptr<MultiPenaltyGcvCudaResidualBatch##SUFFIX> residual;    \
  };                                                                      \
  struct MultiPenaltyGcvCudaMultiRequest##SUFFIX {                        \
    std::shared_ptr<MultiPenaltyGcvCudaPrepared##SUFFIX> prepared;         \
    std::vector<double> Y;                                                \
    int n = 0;                                                            \
    int target_count = 0;                                                 \
    std::vector<std::string> target_keys;                                 \
    MultiPenaltyGcvCudaOptimizerControl control;                          \
  };                                                                      \
  struct MultiPenaltyGcvCudaMultiResult##SUFFIX {                         \
    std::string schema_version;                                           \
    std::vector<MultiPenaltyGcvCudaBatchResult##SUFFIX> setups;           \
    MultiPenaltyGcvCudaMultiDiagnostics diagnostics;                      \
  };                                                                      \
  std::shared_ptr<MultiPenaltyGcvCudaPrepared##SUFFIX>                    \
  create_multi_penalty_gcv_cuda_prepared_##SUFFIX(                        \
      const double* X, const double* magic_qr_packed,                     \
      const double* magic_tau, const double* magic_r,                     \
      const int* magic_pivot_zero_based,                                  \
      const std::vector<std::vector<double>>& penalty_roots,              \
      const std::vector<std::vector<double>>& penalty_matrices,           \
      const std::vector<int>& penalty_ranks, const double* initial_log_sp,\
      int n, int coefficient_dim, int penalty_count,                      \
      int target_capacity, int device_id);                                \
  MultiPenaltyGcvCudaPreparedInfo                                         \
  multi_penalty_gcv_cuda_prepared_info_##SUFFIX(                          \
      const std::shared_ptr<MultiPenaltyGcvCudaPrepared##SUFFIX>&);        \
  MultiPenaltyGcvCudaBatchResult##SUFFIX                                  \
  multi_penalty_gcv_cuda_optimize_batch_##SUFFIX(                         \
      const std::shared_ptr<MultiPenaltyGcvCudaPrepared##SUFFIX>&,         \
      const double* Y, int n, int target_count,                           \
      const std::vector<std::string>& target_keys,                        \
      const MultiPenaltyGcvCudaOptimizerControl& control);                \
  MultiPenaltyGcvCudaMultiResult##SUFFIX                                  \
  multi_penalty_gcv_cuda_optimize_multi_##SUFFIX(                         \
      std::vector<MultiPenaltyGcvCudaMultiRequest##SUFFIX> requests,       \
      int requested_concurrency);                                        \
  void release_multi_penalty_gcv_cuda_residual_##SUFFIX(                  \
      const std::shared_ptr<MultiPenaltyGcvCudaResidualBatch##SUFFIX>&)

FASTKPC_DECLARE_MULTI_PENALTY_EXTENDED_BACKEND(small64);
FASTKPC_DECLARE_MULTI_PENALTY_EXTENDED_BACKEND(ext192);
FASTKPC_DECLARE_MULTI_PENALTY_EXTENDED_BACKEND(ext384);
FASTKPC_DECLARE_MULTI_PENALTY_EXTENDED_BACKEND(ext559);

#undef FASTKPC_DECLARE_MULTI_PENALTY_EXTENDED_BACKEND

}  // namespace fastkpc

#endif
