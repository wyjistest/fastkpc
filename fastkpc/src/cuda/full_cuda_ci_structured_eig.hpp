#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include "full_cuda_ci_structured_launch.hpp"

namespace fastkpc {

// Internal, opt-in component producer. Residual solver route/status are separate.
constexpr double kStructuredComponentGapThreshold = 1e-8;
enum StructuredComponentFallback { StructuredAccepted = 0, StructuredInfo = 1,
  StructuredRotationOverflow = 2, StructuredBoundaryGap = 3 };
struct StructuredComponentStatus {
  int info, distinct, rotations, deflated, reason, reserved;
  double kth, next, gap;
};
struct StructuredComponentView {
  const double* values;
  const double* vectors;
  const StructuredComponentStatus* status;
};
struct StructuredComponentPlan;
using StructuredComponentAllocate = void* (*)(void*, std::size_t);

// Host control-plane counters; no device reads or numerical authority.
// Index 0 is original, index 1 is fixed-VT work pair.
struct StructuredComponentExecutionCounts {
  std::uint64_t created[2]{}, destroyed[2]{}, submissions[2]{}, targets[2]{};
};
StructuredComponentExecutionCounts structured_component_execution_counts() noexcept;
StructuredDbdsqrMode structured_component_plan_mode(const StructuredComponentPlan*);

// The caller owns every CUDA allocation, the stream and its completion events.
// Destruction releases host metadata only; join the stream before freeing buffers.
StructuredComponentPlan* create_structured_component_plan(
    int n, int k, int capacity, int rotation_capacity, cudaStream_t stream,
    StructuredComponentAllocate allocate, void* owner);
void destroy_structured_component_plan(StructuredComponentPlan*) noexcept;
StructuredComponentView submit_structured_components(
    StructuredComponentPlan*, const double* residuals, int count);

}  // namespace fastkpc
