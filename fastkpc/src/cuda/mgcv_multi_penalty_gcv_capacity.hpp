#ifndef FASTKPC_MGCV_MULTI_PENALTY_GCV_CAPACITY_HPP
#define FASTKPC_MGCV_MULTI_PENALTY_GCV_CAPACITY_HPP

#include "mgcv_multi_penalty_gcv.hpp"
#include "mgcv_multi_penalty_gcv_l5.hpp"

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
class MultiPenaltyGcvCapacityResidualBatch;

// Default-off L5 runtime experiment. Each slot has capacity eight; calls still
// use the original target counts. A live residual lease forbids rebinding.
std::shared_ptr<MultiPenaltyGcvCapacityPrepared>
create_multi_penalty_gcv_capacity_l5_slot(
    const MultiPenaltyGcvL5Geometry& geometry, int device_id);
void rebind_multi_penalty_gcv_capacity_l5_slot(
    const std::shared_ptr<MultiPenaltyGcvCapacityPrepared>& prepared,
    const MultiPenaltyGcvL5Geometry& geometry);
void drain_multi_penalty_gcv_capacity_l5_slot(
    const std::shared_ptr<MultiPenaltyGcvCapacityPrepared>& prepared);

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
  // Populated only when retain_residuals is requested. Each token owns the
  // selected-fit output slot for the corresponding setup until explicitly
  // released by the caller.
  std::vector<std::shared_ptr<MultiPenaltyGcvCapacityResidualBatch>> residuals;
  MultiPenaltyGcvCudaMultiDiagnostics diagnostics;
};

struct MultiPenaltyGcvCapacityBatchResult {
  MultiPenaltyGcvCudaOptimization optimization;
  std::shared_ptr<MultiPenaltyGcvCapacityResidualBatch> residual;
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

// Qualification/ownership-preserving entry point.  Production callers use
// multi_penalty_gcv_capacity_optimize_batch(), which releases the selected-fit
// residual token before returning.  This variant exposes the same token so a
// diagnostic can compare the exact-primary final state with a subsequent
// exact fixed-SP evaluation without changing either numerical path.
MultiPenaltyGcvCapacityBatchResult
multi_penalty_gcv_capacity_optimize_batch_with_residual(
    const std::shared_ptr<MultiPenaltyGcvCapacityPrepared>& prepared,
    const double* Y,
    int n,
    int target_count,
    const std::vector<std::string>& target_keys,
    const MultiPenaltyGcvCudaOptimizerControl& control);

std::shared_ptr<MultiPenaltyGcvCapacityResidualBatch>
multi_penalty_gcv_capacity_evaluate_fixed_batch(
    const std::shared_ptr<MultiPenaltyGcvCapacityPrepared>& prepared,
    const double* Y,
    const double* log_sp,
    int n,
    int target_count,
    const std::vector<std::string>& target_keys,
    double rank_tolerance);

std::shared_ptr<MultiPenaltyGcvCapacityResidualBatch>
multi_penalty_gcv_capacity_finalize_selected_coefficients(
    const std::shared_ptr<MultiPenaltyGcvCapacityPrepared>& prepared,
    const double* Y,
    const MultiPenaltyGcvCudaCoefficientConsumerView& coefficients,
    int n,
    int target_count,
    const std::vector<std::string>& target_keys);

MultiPenaltyGcvCudaResidualInfo multi_penalty_gcv_capacity_residual_info(
    const std::shared_ptr<MultiPenaltyGcvCapacityResidualBatch>& residual);

MultiPenaltyGcvCudaResidualConsumerView
acquire_multi_penalty_gcv_capacity_residual_consumer_view(
    const std::shared_ptr<MultiPenaltyGcvCapacityResidualBatch>& residual);

MultiPenaltyGcvCudaResidualShadow
materialize_multi_penalty_gcv_capacity_residual_shadow(
    const std::shared_ptr<MultiPenaltyGcvCapacityResidualBatch>& residual);

void release_multi_penalty_gcv_capacity_residual(
    const std::shared_ptr<MultiPenaltyGcvCapacityResidualBatch>& residual);

MultiPenaltyGcvCapacityMultiResult multi_penalty_gcv_capacity_optimize_multi(
    std::vector<MultiPenaltyGcvCapacityRequest> requests,
    int requested_concurrency,
    bool retain_residuals = false);

}  // namespace fastkpc

#endif
