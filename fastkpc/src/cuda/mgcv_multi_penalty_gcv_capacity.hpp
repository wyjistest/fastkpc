#ifndef FASTKPC_MGCV_MULTI_PENALTY_GCV_CAPACITY_HPP
#define FASTKPC_MGCV_MULTI_PENALTY_GCV_CAPACITY_HPP

#include "mgcv_multi_penalty_gcv.hpp"

#include <memory>
#include <string>
#include <vector>

namespace fastkpc {

enum class MultiPenaltyGcvCapacityBucket {
  Small64,
  Base80,
  Extended192,
  Extended384,
  Extended559
};

class MultiPenaltyGcvCapacityPrepared;

struct MultiPenaltyGcvCapacityRequest {
  std::shared_ptr<MultiPenaltyGcvCapacityPrepared> prepared;
  std::vector<double> Y;
  int n = 0;
  int target_count = 0;
  std::vector<std::string> target_keys;
  MultiPenaltyGcvCudaOptimizerControl control;
};

struct MultiPenaltyGcvCapacityMultiResult {
  std::vector<MultiPenaltyGcvCudaOptimization> setups;
  MultiPenaltyGcvCudaMultiDiagnostics diagnostics;
};

MultiPenaltyGcvCapacityBucket multi_penalty_gcv_capacity_bucket(
    int coefficient_dim, int penalty_count);

int multi_penalty_gcv_capacity_max_concurrent_setups(
    int coefficient_dim, int penalty_count);

std::shared_ptr<MultiPenaltyGcvCapacityPrepared>
create_multi_penalty_gcv_capacity_prepared(
    const double* X,
    const double* magic_qr_packed,
    const double* magic_tau,
    const double* magic_r,
    const int* magic_pivot_zero_based,
    const std::vector<std::vector<double>>& penalty_roots,
    const std::vector<std::vector<double>>& penalty_matrices,
    const std::vector<int>& penalty_ranks,
    const double* initial_log_sp,
    int n,
    int coefficient_dim,
    int penalty_count,
    int target_capacity,
    int device_id);

MultiPenaltyGcvCudaPreparedInfo multi_penalty_gcv_capacity_prepared_info(
    const std::shared_ptr<MultiPenaltyGcvCapacityPrepared>& prepared);

MultiPenaltyGcvCudaOptimization multi_penalty_gcv_capacity_optimize_batch(
    const std::shared_ptr<MultiPenaltyGcvCapacityPrepared>& prepared,
    const double* Y,
    int n,
    int target_count,
    const std::vector<std::string>& target_keys,
    const MultiPenaltyGcvCudaOptimizerControl& control);

MultiPenaltyGcvCapacityMultiResult multi_penalty_gcv_capacity_optimize_multi(
    std::vector<MultiPenaltyGcvCapacityRequest> requests,
    int requested_concurrency);

}  // namespace fastkpc

#endif
