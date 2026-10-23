#pragma once
#include <cuda_runtime.h>

// Research ABI. Solver identity is separate from the residual's SVD route.
// gap = (abs(lambda[k-1])-abs(lambda[k])) / max(abs(...), DBL_MIN).
constexpr double kStructuredAdapterGapThreshold = 1e-8;
enum StructuredFallbackReason { kNoFallback=0, kStructuredInfo=1,
  kRotationOverflow=2, kSmallBoundaryGap=3, kDenseReference=4 };
struct StructuredStatus {
  int info, distinct, rotations, deflated, reason, reserved;
  double kth, next, gap;
};
struct StructuredDeviceView {
  const double* values;
  const double* vectors;
  const StructuredStatus* status;
  cudaEvent_t complete;
};
extern "C" void* adapter_structured_create(int n,int k,int capacity,int rotations,char* error);
extern "C" int adapter_structured_submit(void*,const double* device_residual,int count,
    cudaEvent_t ready,int fault_index,int fault,StructuredDeviceView*,char* error);
extern "C" void adapter_structured_destroy(void*);
extern "C" size_t adapter_structured_bytes(void*);
