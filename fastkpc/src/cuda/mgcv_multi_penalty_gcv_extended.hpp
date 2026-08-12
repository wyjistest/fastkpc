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
  std::shared_ptr<MultiPenaltyGcvCudaResidualBatch##SUFFIX>                \
  multi_penalty_gcv_cuda_evaluate_fixed_batch_##SUFFIX(                   \
      const std::shared_ptr<MultiPenaltyGcvCudaPrepared##SUFFIX>&,         \
      const double* Y, const double* log_sp, int n, int target_count,      \
      const std::vector<std::string>& target_keys, double rank_tolerance); \
  std::shared_ptr<MultiPenaltyGcvCudaResidualBatch##SUFFIX>                \
  multi_penalty_gcv_cuda_finalize_selected_coefficients_##SUFFIX(         \
      const std::shared_ptr<MultiPenaltyGcvCudaPrepared##SUFFIX>&,         \
      const double* Y,                                                     \
      const MultiPenaltyGcvCudaCoefficientConsumerView& coefficients,     \
      int n, int target_count,                                             \
      const std::vector<std::string>& target_keys);                        \
  MultiPenaltyGcvCudaMultiResult##SUFFIX                                  \
  multi_penalty_gcv_cuda_optimize_multi_##SUFFIX(                         \
      std::vector<MultiPenaltyGcvCudaMultiRequest##SUFFIX> requests,       \
      int requested_concurrency);                                        \
  MultiPenaltyGcvCudaResidualInfo                                         \
  multi_penalty_gcv_cuda_residual_info_##SUFFIX(                          \
      const std::shared_ptr<MultiPenaltyGcvCudaResidualBatch##SUFFIX>&);   \
  MultiPenaltyGcvCudaResidualConsumerView                                 \
  acquire_multi_penalty_gcv_cuda_residual_consumer_view_##SUFFIX(         \
      const std::shared_ptr<MultiPenaltyGcvCudaResidualBatch##SUFFIX>&);   \
  MultiPenaltyGcvCudaResidualShadow                                       \
  materialize_multi_penalty_gcv_cuda_residual_shadow_##SUFFIX(            \
      const std::shared_ptr<MultiPenaltyGcvCudaResidualBatch##SUFFIX>&);   \
  void release_multi_penalty_gcv_cuda_residual_##SUFFIX(                  \
      const std::shared_ptr<MultiPenaltyGcvCudaResidualBatch##SUFFIX>&)

FASTKPC_DECLARE_MULTI_PENALTY_EXTENDED_BACKEND(small64);
FASTKPC_DECLARE_MULTI_PENALTY_EXTENDED_BACKEND(ext192);
FASTKPC_DECLARE_MULTI_PENALTY_EXTENDED_BACKEND(ext384);
FASTKPC_DECLARE_MULTI_PENALTY_EXTENDED_BACKEND(ext559);

#undef FASTKPC_DECLARE_MULTI_PENALTY_EXTENDED_BACKEND

MultiPenaltyGcvCudaResponseTranscript
multi_penalty_gcv_response_transcript_cuda_for_test_ext192(
    const double* Y,
    const double* magic_qr_packed,
    const double* magic_tau,
    int n,
    int coefficient_dim,
    int target_count);

MultiPenaltyGcvCudaObjectiveTranscript
multi_penalty_gcv_objective_transcript_cuda_for_test_ext192(
    const double* magic_r,
    const int* magic_pivot_zero_based,
    const std::vector<std::vector<double>>& penalty_roots,
    const std::vector<std::vector<double>>& penalty_matrices,
    const std::vector<int>& penalty_ranks,
    const double* y0,
    double squared_norm,
    const double* log_sp,
    int n,
    int coefficient_dim,
    int penalty_count,
    double rank_tolerance,
    bool force_stable_svd);

MultiPenaltyGcvCudaSvdTranscript
multi_penalty_gcv_svd_transcript_cuda_for_test_ext192(
    const double* augmented,
    int rows,
    int columns,
    bool force_stable_svd);

MultiPenaltyGcvCudaAugmentedTranscript
multi_penalty_gcv_augmented_transcript_cuda_for_test_ext192(
    const double* magic_r,
    const std::vector<std::vector<double>>& penalty_matrices,
    const double* log_sp,
    int columns,
    int penalty_count);

}  // namespace fastkpc

#endif
