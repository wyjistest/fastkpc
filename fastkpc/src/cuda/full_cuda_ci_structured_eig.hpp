#pragma once

#include <cuda_runtime.h>
#include <cstddef>

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

// The caller owns every CUDA allocation, the stream and its completion events.
// Destruction releases host metadata only; join the stream before freeing buffers.
StructuredComponentPlan* create_structured_component_plan(
    int n, int k, int capacity, int rotation_capacity, cudaStream_t stream,
    StructuredComponentAllocate allocate, void* owner);
void destroy_structured_component_plan(StructuredComponentPlan*) noexcept;
StructuredComponentView submit_structured_components(
    StructuredComponentPlan*, const double* residuals, int count);

}  // namespace fastkpc
