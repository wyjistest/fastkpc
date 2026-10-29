#ifndef FASTKPC_MGCV_MULTI_PENALTY_GCV_POSTBACK_HPP
#define FASTKPC_MGCV_MULTI_PENALTY_GCV_POSTBACK_HPP

#include "mgcv_multi_penalty_gcv_extended.hpp"
#include "mgcv_multi_penalty_gcv_l4.hpp"
#include "mgcv_multi_penalty_gcv_l5.hpp"
#include <cstdint>

namespace fastkpc {

// Strict 0/1 flag, unset means OFF. Read only when a prepared object is created;
// its backend identity remains immutable throughout rebind and residual leases.
bool multi_penalty_gcv_vector_postback_requested();

struct MultiPenaltyGcvPostbackExecutionCounts {
  // Index 0: original backend; index 1: independent vector-postback backend.
  std::uint64_t created[2] = {}, destroyed[2] = {};
  std::uint64_t optimizer_batches[2] = {}, optimizer_targets[2] = {};
  std::uint64_t q37_targets[2] = {}, q46_targets[2] = {}, other_targets[2] = {};
  std::uint64_t fixed_targets[2] = {}, coefficient_finalization_targets[2] = {};
};
MultiPenaltyGcvPostbackExecutionCounts multi_penalty_gcv_postback_execution_counts();

void rebind_multi_penalty_gcv_cuda_prepared_postback_l4(
    const std::shared_ptr<MultiPenaltyGcvCudaPreparedpostback64>&,
    const MultiPenaltyGcvL4Geometry&);
void drain_multi_penalty_gcv_cuda_prepared_postback_l4(
    const std::shared_ptr<MultiPenaltyGcvCudaPreparedpostback64>&);
void rebind_multi_penalty_gcv_cuda_prepared_postback_l5(
    const std::shared_ptr<MultiPenaltyGcvCudaPreparedpostback64>&,
    const MultiPenaltyGcvL5Geometry&);
void drain_multi_penalty_gcv_cuda_prepared_postback_l5(
    const std::shared_ptr<MultiPenaltyGcvCudaPreparedpostback64>&);

}  // namespace fastkpc
#endif
