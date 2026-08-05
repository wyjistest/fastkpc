#include "mgcv_multi_penalty_gcv.hpp"

#define FASTKPC_LAPACK_SMALL_MAX_COLUMNS \
  FASTKPC_MULTI_PENALTY_MAX_COEFFICIENT_DIM
#define FASTKPC_LAPACK_SMALL_NAMESPACE lapack312_multi
#include "lapack_312_small_dgesdd.cuh"
#undef FASTKPC_LAPACK_SMALL_NAMESPACE
#include "third_party/glibc_2_35_exp_fma.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fastkpc {
namespace {

constexpr int kBlockSize = 64;
constexpr int kMaximumRows =
  2 * kMultiPenaltyGcvMaximumCoefficientDim;
constexpr double kLapackEpsilon =
  1.1102230246251565404236316680908203125e-16;
constexpr double kDenominatorFloor = 1e-8;
constexpr int kStabilityReplayLongTrajectoryMinimumIterations = 101;
constexpr int kStabilityReplayBoundaryMinimumIterations = 25;
constexpr int kStabilityReplayBoundaryMaximumIterations = 100;
constexpr int kStabilityReplayBoundaryMinimumAcceptedProbes = 2;
constexpr int kStabilityReplayBoundaryStepHalvingNumerator = 9;
constexpr int kStabilityReplayBoundaryStepHalvingDenominator = 4;
constexpr int kStabilityReplayDenseRiskMaximumStepHalvingNumerator = 49;
constexpr int kStabilityReplayDenseRiskMaximumStepHalvingDenominator = 20;
constexpr double kStabilityReplayDenseRiskConditionOverride = 8388608.0;
constexpr int kStabilityReplayLongRiskStepHalvingMultiplier = 4;
constexpr int kStabilityReplayHighConditionMinimumIterations = 16;
constexpr int kStabilityReplayHighConditionStepHalvingNumerator = 3;
constexpr int kStabilityReplayHighConditionStepHalvingDenominator = 4;
constexpr double kStabilityReplayHighConditionThreshold = 16777216.0;
constexpr int kDirectNewtonReplayMaximumIterations = 20;
constexpr int kDirectNewtonReplayMinimumAcceptedProbes = 1;
constexpr double kDirectNewtonReplayMaximumSmallestEigenvalue = 1e-12;
constexpr double kDirectNewtonReplayMinimumHessianCondition = 1e9;
constexpr double kDirectNewtonHalvingExtrapolationFraction = 0.75;
constexpr double kDirectNewtonMaximumExtrapolation = 2e-5;
constexpr double kStabilityReplayLogSpSpread = 3e-8;
constexpr double kStabilityReplayDenseBoundaryMaximumSpread = 5e-8;
constexpr double kStabilityReplayMaximumInwardShift = 1e-6;
constexpr double kStabilityReplayExtrapolationFraction = 0.25;
constexpr double kStabilityReplayDenseBoundaryExtrapolationFraction = 8.0;
constexpr double kStabilityReplayMaximumExtrapolation = 4e-6;
constexpr double kAmbiguousStepRelativeTolerance =
  128.0 * kLapackEpsilon;
constexpr double kAmbiguousStepConservativeDescentFraction = 0.5;
constexpr double kAmbiguousStepConservativeMinimumHessianCondition = 5.0;
constexpr double kAmbiguousStepDirectDeltaDisagreementFraction = 0.25;
constexpr double kRejectedBoundaryReplayDeltaRatio = 2.0;
constexpr double kTerminalBoundaryTieCondition = 8388608.0;
constexpr double kTerminalBoundaryOverrideCondition = 33554432.0;
constexpr double kTerminalBoundaryTieRelativeTolerance =
  1e-10;

enum DeviceDecompositionTraceFlag : unsigned int {
  kDecompositionTraceInitial = 1U << 0,
  kDecompositionTraceNewtonTrial = 1U << 1,
  kDecompositionTraceSteepestTrial = 1U << 2,
  kDecompositionTraceStepHalving = 1U << 3,
  kDecompositionTraceBoundaryProbe = 1U << 4,
  kDecompositionTraceTerminalConfirmation = 1U << 5,
  kDecompositionTraceStabilityReplay = 1U << 6,
  kDecompositionTraceSelectedFit = 1U << 7,
  kDecompositionTraceCompleteCapability = 1U << 8,
  kDecompositionTraceForceStableSvd = 1U << 9
};

constexpr std::array<unsigned int,
                     kMultiPenaltyGcvDecompositionTraceStageCount>
  kDecompositionTraceStageFlags = {{
    kDecompositionTraceInitial,
    kDecompositionTraceNewtonTrial,
    kDecompositionTraceSteepestTrial,
    kDecompositionTraceStepHalving,
    kDecompositionTraceBoundaryProbe,
    kDecompositionTraceTerminalConfirmation,
    kDecompositionTraceStabilityReplay,
    kDecompositionTraceSelectedFit
  }};

void check_cuda(cudaError_t status, const char* stage) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string(stage) + ": " + cudaGetErrorString(status));
  }
}

void check_cublas(cublasStatus_t status, const char* stage) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
      std::string(stage) + ": cuBLAS status " +
      std::to_string(static_cast<int>(status)));
  }
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() {
    if (owns_data_ && data_ != nullptr) cudaFree(data_);
  }

  void allocate(std::size_t count,
                MultiPenaltyGcvCudaDiagnostics* diagnostics) {
    if (count == 0) return;
    check_cuda(cudaMalloc(
      reinterpret_cast<void**>(&data_), count * sizeof(T)),
      "allocate multi-penalty GCV buffer");
    count_ = count;
    owns_data_ = true;
    diagnostics->device_allocation_count += 1;
  }

  std::size_t bind_from_arena(
      unsigned char* arena,
      std::size_t offset,
      std::size_t count) {
    if (arena == nullptr || data_ != nullptr || count == 0) {
      throw std::runtime_error("multi-penalty CUDA arena binding is invalid");
    }
    constexpr std::size_t alignment = alignof(T);
    const std::size_t aligned =
      (offset + alignment - 1) & ~(alignment - 1);
    data_ = reinterpret_cast<T*>(arena + aligned);
    count_ = count;
    owns_data_ = false;
    return aligned + count * sizeof(T);
  }

  T* get() { return data_; }
  const T* get() const { return data_; }
  std::size_t count() const { return count_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
  bool owns_data_ = false;
};

template <typename T>
std::size_t arena_advance(std::size_t offset, std::size_t count) {
  constexpr std::size_t alignment = alignof(T);
  const std::size_t aligned =
    (offset + alignment - 1) & ~(alignment - 1);
  return aligned + count * sizeof(T);
}

class Stream {
 public:
  Stream() {
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "create multi-penalty GCV stream");
  }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  ~Stream() {
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }
  cudaStream_t get() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

class Event {
 public:
  Event() {
    check_cuda(cudaEventCreateWithFlags(
      &event_, cudaEventDisableTiming),
      "create multi-penalty GCV completion event");
  }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  ~Event() {
    if (event_ != nullptr) cudaEventDestroy(event_);
  }
  cudaEvent_t get() const { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

class TimingEvent {
 public:
  TimingEvent() {
    check_cuda(cudaEventCreate(&event_),
               "create multi-penalty GCV timing event");
  }
  TimingEvent(const TimingEvent&) = delete;
  TimingEvent& operator=(const TimingEvent&) = delete;
  ~TimingEvent() {
    if (event_ != nullptr) cudaEventDestroy(event_);
  }
  cudaEvent_t get() const { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

class CublasHandle {
 public:
  explicit CublasHandle(cudaStream_t stream) {
    check_cublas(cublasCreate(&handle_),
                 "create multi-penalty GCV cuBLAS handle");
    check_cublas(cublasSetStream(handle_, stream),
                 "bind multi-penalty GCV cuBLAS stream");
  }
  CublasHandle(const CublasHandle&) = delete;
  CublasHandle& operator=(const CublasHandle&) = delete;
  ~CublasHandle() {
    if (handle_ != nullptr) cublasDestroy(handle_);
  }
  cublasHandle_t get() const { return handle_; }

 private:
  cublasHandle_t handle_ = nullptr;
};

template <typename T>
void upload_async(DeviceBuffer<T>* destination,
                  const T* source,
                  std::size_t count,
                  cudaStream_t stream,
                  MultiPenaltyGcvCudaDiagnostics* diagnostics) {
  if (count == 0) return;
  check_cuda(cudaMemcpyAsync(
    destination->get(), source, count * sizeof(T),
    cudaMemcpyHostToDevice, stream),
    "upload multi-penalty GCV input");
  diagnostics->h2d_copy_count += 1;
}

__device__ double rounded_multiply_add(double accumulator,
                                       double left,
                                       double right) {
  return __dadd_rn(accumulator, __dmul_rn(left, right));
}

__global__ void target_squared_norm_kernel(
    const double* Y, double* squared_norm, int n, int target_count) {
  const int target = blockIdx.x * blockDim.x + threadIdx.x;
  if (target >= target_count) return;
  double value = 0.0;
  const double* y = Y + static_cast<std::size_t>(n) * target;
  for (int row = 0; row < n; ++row) {
    value = rounded_multiply_add(value, y[row], y[row]);
  }
  squared_norm[target] = value;
}

__global__ void magic_qt_y_kernel(
    const double* qr_packed,
    const double* tau,
    const double* Y,
    double* work,
    double* y0,
    int n,
    int q,
    int target_count) {
  const int target = blockIdx.x * blockDim.x + threadIdx.x;
  if (target >= target_count) return;
  double* target_work = work + static_cast<std::size_t>(n) * target;
  const double* target_y = Y + static_cast<std::size_t>(n) * target;
  for (int row = 0; row < n; ++row) target_work[row] = target_y[row];
  for (int reflector = 0; reflector < q; ++reflector) {
    double dot = target_work[reflector];
    const double* vector =
      qr_packed + static_cast<std::size_t>(n) * reflector;
    for (int row = reflector + 1; row < n; ++row) {
      dot = rounded_multiply_add(dot, vector[row], target_work[row]);
    }
    const double scale = __dmul_rn(-tau[reflector], dot);
    target_work[reflector] = __dadd_rn(target_work[reflector], scale);
    for (int row = reflector + 1; row < n; ++row) {
      target_work[row] = rounded_multiply_add(
        target_work[row], vector[row], scale);
    }
  }
  for (int row = 0; row < q; ++row) {
    y0[row + q * target] = target_work[row];
  }
}

__device__ void swap_double(double* left, double* right) {
  const double saved = *left;
  *left = *right;
  *right = saved;
}

__device__ int dpstf2_upper_block(
    double* matrix, int* pivot, double* work, int q) {
  for (int index = threadIdx.x; index < q; index += blockDim.x) {
    pivot[index] = index + 1;
    work[index] = 0.0;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int pivot_index = 0;
    double diagonal = matrix[0];
    for (int index = 1; index < q; ++index) {
      const double candidate = matrix[index + q * index];
      if (candidate > diagonal) {
        pivot_index = index;
        diagonal = candidate;
      }
    }
    pivot[q] = diagonal <= 0.0 || isnan(diagonal) ? 0 : -1;
    pivot[q + 1] = pivot_index;
    work[2 * q] = __dmul_rn(
      __dmul_rn(static_cast<double>(q), kLapackEpsilon), diagonal);
    work[2 * q + 1] = diagonal;
  }
  __syncthreads();
  if (pivot[q] == 0) return 0;
  for (int column = 0; column < q; ++column) {
    for (int index = column + threadIdx.x; index < q;
         index += blockDim.x) {
      if (column > 0) {
        const double previous = matrix[(column - 1) + q * index];
        work[index] = rounded_multiply_add(
          work[index], previous, previous);
      }
      work[q + index] = __dadd_rn(
        matrix[index + q * index], -work[index]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int pivot_index = pivot[q + 1];
      double diagonal = work[2 * q + 1];
      if (column > 0) {
        pivot_index = column;
        diagonal = work[q + column];
        for (int index = column + 1; index < q; ++index) {
          const double candidate = work[q + index];
          if (candidate > diagonal) {
            pivot_index = index;
            diagonal = candidate;
          }
        }
        if (diagonal <= work[2 * q] || isnan(diagonal)) {
          matrix[column + q * column] = diagonal;
          pivot[q] = column;
        }
      }
      if (pivot[q] < 0) {
        if (column != pivot_index) {
          matrix[pivot_index + q * pivot_index] =
            matrix[column + q * column];
          for (int row = 0; row < column; ++row) {
            swap_double(matrix + row + q * column,
                        matrix + row + q * pivot_index);
          }
          for (int trailing = pivot_index + 1; trailing < q; ++trailing) {
            swap_double(matrix + column + q * trailing,
                        matrix + pivot_index + q * trailing);
          }
          for (int index = column + 1; index < pivot_index; ++index) {
            swap_double(matrix + column + q * index,
                        matrix + index + q * pivot_index);
          }
          swap_double(work + column, work + pivot_index);
          const int saved_pivot = pivot[pivot_index];
          pivot[pivot_index] = pivot[column];
          pivot[column] = saved_pivot;
        }
        diagonal = sqrt(diagonal);
        matrix[column + q * column] = diagonal;
        work[2 * q + 1] = diagonal;
        pivot[q + 1] = pivot_index;
      }
    }
    __syncthreads();
    if (pivot[q] >= 0) return pivot[q];
    const double diagonal = work[2 * q + 1];
    if (column + 1 < q) {
      for (int trailing = column + 1 + threadIdx.x; trailing < q;
           trailing += blockDim.x) {
        double dot = 0.0;
        for (int row = 0; row < column; ++row) {
          dot = rounded_multiply_add(
            dot, matrix[row + q * trailing],
            matrix[row + q * column]);
        }
        matrix[column + q * trailing] = __ddiv_rn(
          __dadd_rn(matrix[column + q * trailing], -dot), diagonal);
      }
    }
    __syncthreads();
  }
  return q;
}

struct DeviceTargetEvaluation {
  double rss;
  double edf;
  double score;
  double condition;
  double gradient[kMultiPenaltyGcvMaximumPenaltyCount];
  double hessian[
    kMultiPenaltyGcvMaximumPenaltyCount *
    kMultiPenaltyGcvMaximumPenaltyCount];
  double coefficients[kMultiPenaltyGcvMaximumCoefficientDim];
  int aggregate_penalty_rank;
  int numerical_rank;
  int solver_info;
  int decomposition_route;
};

enum DeviceDecompositionRoute : int {
  kDecompositionUnknown = 0,
  kDecompositionGuardedQr = 1,
  kDecompositionStableSvd = 2
};

struct DeviceEvaluationWorkspace {
  lapack312_multi::Workspace decomposition;
  double metric[
    kMultiPenaltyGcvMaximumPenaltyCount *
    kMultiPenaltyGcvMaximumCoefficientDim *
    kMultiPenaltyGcvMaximumCoefficientDim];
  double influence[
    kMultiPenaltyGcvMaximumPenaltyCount *
    kMultiPenaltyGcvMaximumCoefficientDim *
    kMultiPenaltyGcvMaximumCoefficientDim];
  double metric_y[
    kMultiPenaltyGcvMaximumPenaltyCount *
    kMultiPenaltyGcvMaximumCoefficientDim];
  double influence_y[
    kMultiPenaltyGcvMaximumPenaltyCount *
    kMultiPenaltyGcvMaximumCoefficientDim];
  double y_influence[
    kMultiPenaltyGcvMaximumPenaltyCount *
    kMultiPenaltyGcvMaximumCoefficientDim];
  double derivative_norm[kMultiPenaltyGcvMaximumPenaltyCount];
  double derivative_delta[kMultiPenaltyGcvMaximumPenaltyCount];
  double second_derivative_norm[
    kMultiPenaltyGcvMaximumPenaltyCount *
    kMultiPenaltyGcvMaximumPenaltyCount];
  double second_derivative_delta[
    kMultiPenaltyGcvMaximumPenaltyCount *
    kMultiPenaltyGcvMaximumPenaltyCount];
};

enum DeviceBoundaryStatus : int {
  kBoundaryFiniteInterior = 0,
  kBoundaryPositive = 1,
  kBoundaryNegative = 2,
  kBoundaryFiniteAfterProbe = 3
};

struct DevicePhaseTiming {
  unsigned long long penalty_factor_augmentation_cycles;
  unsigned long long qr_svd_cycles;
  unsigned long long score_construction_cycles;
  unsigned long long derivative_hessian_cycles;
  // Entries 4-7 partition the QR/reduction total in entry 0.
  unsigned long long decomposition_stage_cycles[8];
  int complete_evaluation_count;
  int score_only_evaluation_count;
  int guarded_qr_evaluation_count;
  int stable_svd_evaluation_count;
};

struct DeviceDecompositionTraceRecord {
  unsigned long long
    log_sp_bits[kMultiPenaltyGcvMaximumPenaltyCount];
  int target;
  int iteration;
  unsigned int flags;
  int route;
};

struct DeviceOptimizerState {
  DeviceTargetEvaluation current;
  DeviceTargetEvaluation trial;
  double log_sp[kMultiPenaltyGcvMaximumPenaltyCount];
  double trial_log_sp[kMultiPenaltyGcvMaximumPenaltyCount];
  double gradient[kMultiPenaltyGcvMaximumPenaltyCount];
  double newton_step[kMultiPenaltyGcvMaximumPenaltyCount];
  double steepest_step[kMultiPenaltyGcvMaximumPenaltyCount];
  double working_step[kMultiPenaltyGcvMaximumPenaltyCount];
  double hessian_eigenvalues[kMultiPenaltyGcvMaximumPenaltyCount];
  double hessian_eigenvectors[
    kMultiPenaltyGcvMaximumPenaltyCount *
    kMultiPenaltyGcvMaximumPenaltyCount];
  double hessian_work[
    kMultiPenaltyGcvMaximumPenaltyCount *
    kMultiPenaltyGcvMaximumPenaltyCount];
  double minimum_score;
  double score_reduction;
  double rms_gradient;
  int optimizer_iterations;
  int score_calls;
  int objective_calls;
  int step_halving_count;
  int newton_trial_count;
  int steepest_descent_trial_count;
  int boundary_probe_count;
  int boundary_accepted_count;
  int boundary_status[kMultiPenaltyGcvMaximumPenaltyCount];
  int fully_converged;
  int hessian_positive_definite;
  int step_failed;
  int optimizer_status;
  int penalty_count;
  int use_steepest_descent;
  int converged;
  int trying;
  int tries;
  int selected_evaluation_reuse_count;
  int ambiguous_step_descent_count;
  int ambiguous_step_non_descent_count;
  int stability_replay_attempted;
  int stability_replay_screened;
  int stability_replay_selected;
  int stability_replay_error;
  int stability_replay_long_trajectory_reason;
  int stability_replay_dense_boundary_reason;
  int stability_replay_high_condition_reason;
  int stability_replay_ambiguous_step_reason;
  int stability_replay_rejected_boundary_reason;
  int stability_replay_direct_newton_reason;
  int stability_replay_dense_score_guard_rejected;
  int stability_replay_extrapolation_applied;
  int terminal_boundary_confirmation_count;
  int terminal_boundary_confirmation_accepted_count;
  int terminal_boundary_confirmation_rejected_count;
  int terminal_boundary_confirmation_strong_delta_accepted_count;
  int terminal_boundary_confirmation_identity_tie_accepted_count;
  int terminal_boundary_confirmation_delta_identity_accepted_count;
  int terminal_boundary_confirmation_pending;
  double stability_replay_log_sp_spread;
  double stability_replay_max_extrapolation;
  double terminal_boundary_confirmation_max_identity_disagreement;
  double terminal_boundary_confirmation_max_identity_ratio;
  double terminal_boundary_confirmation_max_delta_disagreement;
  double terminal_boundary_confirmation_max_delta_ratio;
  double terminal_boundary_primary_current_score;
  double terminal_boundary_primary_trial_score;
  double terminal_boundary_stable_current_score;
  double terminal_boundary_stable_current_direct_score;
  double terminal_boundary_stable_trial_direct_score;
  DevicePhaseTiming phase_timing;
  DevicePhaseTiming discarded_phase_timing;
  DevicePhaseTiming terminal_boundary_confirmation_phase_timing;
};

struct HostDecompositionTraceKey {
  std::array<std::uint64_t, kMultiPenaltyGcvMaximumPenaltyCount>
    log_sp_bits{};
  unsigned int capability_flags = 0;

  bool operator<(const HostDecompositionTraceKey& other) const {
    if (log_sp_bits != other.log_sp_bits) {
      return log_sp_bits < other.log_sp_bits;
    }
    return capability_flags < other.capability_flags;
  }
};

struct HostDecompositionTraceGroup {
  std::uint64_t request_count = 0;
  int route = kDecompositionUnknown;
};

using HostDecompositionTraceGroups =
  std::map<HostDecompositionTraceKey, HostDecompositionTraceGroup>;

HostDecompositionTraceKey decomposition_trace_key(
    const DeviceDecompositionTraceRecord& record) {
  HostDecompositionTraceKey key;
  for (int penalty = 0;
       penalty < kMultiPenaltyGcvMaximumPenaltyCount; ++penalty) {
    key.log_sp_bits[static_cast<std::size_t>(penalty)] =
      static_cast<std::uint64_t>(record.log_sp_bits[penalty]);
  }
  key.capability_flags = record.flags &
    (kDecompositionTraceCompleteCapability |
     kDecompositionTraceForceStableSvd);
  return key;
}

void add_decomposition_trace_group(
    HostDecompositionTraceGroups* groups,
    const HostDecompositionTraceKey& key,
    int route,
    std::uint64_t* route_mismatch_count) {
  auto inserted = groups->emplace(
    key, HostDecompositionTraceGroup{1U, route});
  if (inserted.second) return;
  HostDecompositionTraceGroup& group = inserted.first->second;
  group.request_count += 1U;
  if (group.route != route) {
    *route_mismatch_count += 1U;
  }
}

std::uint64_t decomposition_trace_group_requests(
    const HostDecompositionTraceGroups& groups) {
  std::uint64_t count = 0;
  for (const auto& entry : groups) count += entry.second.request_count;
  return count;
}

void materialize_decomposition_trace_diagnostics(
    const std::vector<DeviceDecompositionTraceRecord>& records,
    std::uint64_t request_count,
    std::uint64_t overflow_count,
    int capacity_per_target,
    MultiPenaltyGcvCudaDiagnostics* diagnostics) {
  diagnostics->cuda_decomposition_trace_enabled = true;
  diagnostics->cuda_decomposition_trace_capacity_per_target =
    capacity_per_target;
  diagnostics->cuda_decomposition_request_count = request_count;
  diagnostics->cuda_decomposition_stored_count = records.size();
  diagnostics->cuda_decomposition_trace_overflow_count = overflow_count;

  HostDecompositionTraceGroups all_groups;
  std::array<HostDecompositionTraceGroups,
             kMultiPenaltyGcvDecompositionTraceStageCount> stage_groups;
  std::array<HostDecompositionTraceGroups,
             kMultiPenaltyGcvDecompositionTraceRouteCount> route_groups;
  using IterationKey = std::pair<bool, int>;
  std::map<IterationKey, HostDecompositionTraceGroups> iteration_groups;
  std::uint64_t ignored_route_mismatch_count = 0;
  for (const DeviceDecompositionTraceRecord& record : records) {
    const HostDecompositionTraceKey key = decomposition_trace_key(record);
    add_decomposition_trace_group(
      &all_groups, key, record.route,
      &diagnostics->cuda_decomposition_route_mismatch_count);
    for (int stage = 0;
         stage < kMultiPenaltyGcvDecompositionTraceStageCount; ++stage) {
      if ((record.flags & kDecompositionTraceStageFlags[
            static_cast<std::size_t>(stage)]) != 0U) {
        add_decomposition_trace_group(
          &stage_groups[static_cast<std::size_t>(stage)], key,
          record.route,
          &ignored_route_mismatch_count);
      }
    }
    const int route = record.route >= kDecompositionUnknown &&
        record.route <= kDecompositionStableSvd ?
      record.route : kDecompositionUnknown;
    add_decomposition_trace_group(
      &route_groups[static_cast<std::size_t>(route)], key, route,
      &ignored_route_mismatch_count);
    if (record.iteration >= 0) {
      const bool replay =
        (record.flags & kDecompositionTraceStabilityReplay) != 0U;
      add_decomposition_trace_group(
        &iteration_groups[IterationKey{replay, record.iteration}], key,
        record.route,
        &ignored_route_mismatch_count);
    }
  }

  diagnostics->cuda_decomposition_unique_key_count = all_groups.size();
  diagnostics->cuda_decomposition_reuse_count =
    records.size() - all_groups.size();
  std::size_t maximum_group_size = 0;
  for (const auto& entry : all_groups) {
    maximum_group_size = std::max(
      maximum_group_size,
      static_cast<std::size_t>(entry.second.request_count));
  }
  diagnostics->cuda_decomposition_reuse_group_size_histogram.assign(
    maximum_group_size + 1U, 0U);
  for (const auto& entry : all_groups) {
    diagnostics->cuda_decomposition_reuse_group_size_histogram[
      static_cast<std::size_t>(entry.second.request_count)] += 1U;
  }
  for (int stage = 0;
       stage < kMultiPenaltyGcvDecompositionTraceStageCount; ++stage) {
    const HostDecompositionTraceGroups& groups =
      stage_groups[static_cast<std::size_t>(stage)];
    const std::uint64_t requests =
      decomposition_trace_group_requests(groups);
    diagnostics->cuda_decomposition_stage_request_count[
      static_cast<std::size_t>(stage)] = requests;
    diagnostics->cuda_decomposition_stage_unique_key_count[
      static_cast<std::size_t>(stage)] = groups.size();
    diagnostics->cuda_decomposition_stage_reuse_count[
      static_cast<std::size_t>(stage)] = requests - groups.size();
  }
  for (int route = 0;
       route < kMultiPenaltyGcvDecompositionTraceRouteCount; ++route) {
    const HostDecompositionTraceGroups& groups =
      route_groups[static_cast<std::size_t>(route)];
    const std::uint64_t requests =
      decomposition_trace_group_requests(groups);
    diagnostics->cuda_decomposition_route_request_count[
      static_cast<std::size_t>(route)] = requests;
    diagnostics->cuda_decomposition_route_unique_key_count[
      static_cast<std::size_t>(route)] = groups.size();
    diagnostics->cuda_decomposition_route_reuse_count[
      static_cast<std::size_t>(route)] = requests - groups.size();
  }
  diagnostics->cuda_decomposition_iteration_reuse.reserve(
    iteration_groups.size());
  for (const auto& entry : iteration_groups) {
    const std::uint64_t requests =
      decomposition_trace_group_requests(entry.second);
    diagnostics->cuda_decomposition_iteration_reuse.push_back(
      MultiPenaltyGcvCudaDecompositionIterationReuse{
        entry.first.second,
        entry.first.first,
        requests,
        entry.second.size(),
        requests - entry.second.size()
      });
  }
}

__device__ bool requires_high_condition_stability_replay(
    const DeviceOptimizerState& state) {
  return state.optimizer_status == 0 &&
    state.boundary_accepted_count > 0 &&
    state.optimizer_iterations >=
      kStabilityReplayHighConditionMinimumIterations &&
    state.current.condition >= kStabilityReplayHighConditionThreshold &&
    state.step_halving_count *
      kStabilityReplayHighConditionStepHalvingDenominator >=
        state.optimizer_iterations *
          kStabilityReplayHighConditionStepHalvingNumerator;
}

__device__ bool requires_long_trajectory_stability_replay(
    const DeviceOptimizerState& state) {
  return state.optimizer_status == 0 &&
    state.optimizer_iterations >=
      kStabilityReplayLongTrajectoryMinimumIterations;
}

__device__ bool requires_dense_boundary_stability_replay(
    const DeviceOptimizerState& state) {
  return state.optimizer_status == 0 &&
    state.optimizer_iterations >= kStabilityReplayBoundaryMinimumIterations &&
    state.optimizer_iterations <= kStabilityReplayBoundaryMaximumIterations &&
    state.boundary_accepted_count >=
      kStabilityReplayBoundaryMinimumAcceptedProbes &&
    state.step_halving_count *
      kStabilityReplayBoundaryStepHalvingDenominator >=
        state.optimizer_iterations *
          kStabilityReplayBoundaryStepHalvingNumerator;
}

__device__ bool requires_ambiguous_step_stability_replay(
    const DeviceOptimizerState& state) {
  return state.optimizer_status == 0 &&
    (state.ambiguous_step_descent_count > 0 ||
     state.ambiguous_step_non_descent_count > 0);
}

__device__ bool requires_rejected_boundary_stability_replay(
    const DeviceOptimizerState& state) {
  return state.optimizer_status == 0 &&
    state.terminal_boundary_confirmation_rejected_count > 0 &&
    state.terminal_boundary_confirmation_max_delta_ratio >
      kRejectedBoundaryReplayDeltaRatio;
}

__device__ bool requires_direct_newton_stability_replay(
    const DeviceOptimizerState& state) {
  const double smallest = fabs(state.hessian_eigenvalues[0]);
  double largest = smallest;
  for (int penalty = 1;
       penalty < state.penalty_count; ++penalty) {
    largest = fmax(largest, fabs(state.hessian_eigenvalues[penalty]));
  }
  return state.optimizer_status == 0 &&
    !requires_high_condition_stability_replay(state) &&
    state.optimizer_iterations <= kDirectNewtonReplayMaximumIterations &&
    state.step_halving_count <= state.optimizer_iterations &&
    state.boundary_accepted_count >=
      kDirectNewtonReplayMinimumAcceptedProbes &&
    state.hessian_positive_definite != 0 && smallest > 0.0 &&
    smallest <= kDirectNewtonReplayMaximumSmallestEigenvalue &&
    largest >= __dmul_rn(
      kDirectNewtonReplayMinimumHessianCondition, smallest);
}

__device__ bool is_stability_replay_candidate(
    const DeviceOptimizerState& state) {
  if (state.optimizer_status != 0) return false;
  if (requires_long_trajectory_stability_replay(state)) {
    return true;
  }
  return requires_dense_boundary_stability_replay(state) ||
    requires_high_condition_stability_replay(state) ||
    requires_ambiguous_step_stability_replay(state) ||
    requires_rejected_boundary_stability_replay(state) ||
    requires_direct_newton_stability_replay(state);
}

__device__ bool requires_full_stability_replay(
    const DeviceOptimizerState& state) {
  if (state.optimizer_status != 0) return false;
  if (requires_ambiguous_step_stability_replay(state) ||
      requires_rejected_boundary_stability_replay(state) ||
      requires_direct_newton_stability_replay(state) ||
      requires_high_condition_stability_replay(state)) {
    return true;
  }
  if (requires_long_trajectory_stability_replay(state)) {
    return state.step_halving_count >=
      kStabilityReplayLongRiskStepHalvingMultiplier *
        state.optimizer_iterations;
  }
  return requires_dense_boundary_stability_replay(state) &&
    (state.step_halving_count *
         kStabilityReplayDenseRiskMaximumStepHalvingDenominator <=
       state.optimizer_iterations *
         kStabilityReplayDenseRiskMaximumStepHalvingNumerator ||
     state.current.condition >=
       kStabilityReplayDenseRiskConditionOverride);
}

__device__ bool symmetric_jacobi_eigen(
    const double* input,
    int dimension,
    double* matrix,
    double* values,
    double* vectors) {
  const int square = dimension * dimension;
  for (int index = 0; index < square; ++index) {
    matrix[index] = input[index];
    vectors[index] = 0.0;
  }
  for (int index = 0; index < dimension; ++index) {
    vectors[index + dimension * index] = 1.0;
  }

  bool converged = false;
  for (int sweep = 0; sweep < 128; ++sweep) {
    double maximum_off_diagonal = 0.0;
    double maximum_diagonal = 0.0;
    for (int column = 0; column < dimension; ++column) {
      maximum_diagonal = fmax(
        maximum_diagonal, fabs(matrix[column + dimension * column]));
      for (int row = 0; row < column; ++row) {
        maximum_off_diagonal = fmax(
          maximum_off_diagonal,
          fabs(matrix[row + dimension * column]));
      }
    }
    if (maximum_off_diagonal <= __dmul_rn(
          8.0 * kLapackEpsilon, fmax(1.0, maximum_diagonal))) {
      converged = true;
      break;
    }

    for (int p = 0; p < dimension - 1; ++p) {
      for (int q = p + 1; q < dimension; ++q) {
        const double apq = matrix[p + dimension * q];
        if (apq == 0.0) continue;
        const double app = matrix[p + dimension * p];
        const double aqq = matrix[q + dimension * q];
        const double tau = __ddiv_rn(
          __dadd_rn(aqq, -app), __dmul_rn(2.0, apq));
        const double root = hypot(1.0, tau);
        const double t = copysign(
          __ddiv_rn(1.0, __dadd_rn(fabs(tau), root)), tau);
        const double cosine = __ddiv_rn(1.0, sqrt(
          __dadd_rn(1.0, __dmul_rn(t, t))));
        const double sine = __dmul_rn(t, cosine);

        for (int index = 0; index < dimension; ++index) {
          if (index == p || index == q) continue;
          const double aip = matrix[index + dimension * p];
          const double aiq = matrix[index + dimension * q];
          const double new_aip = __dadd_rn(
            __dmul_rn(cosine, aip), -__dmul_rn(sine, aiq));
          const double new_aiq = __dadd_rn(
            __dmul_rn(sine, aip), __dmul_rn(cosine, aiq));
          matrix[index + dimension * p] = new_aip;
          matrix[p + dimension * index] = new_aip;
          matrix[index + dimension * q] = new_aiq;
          matrix[q + dimension * index] = new_aiq;
        }
        matrix[p + dimension * p] =
          __dadd_rn(app, -__dmul_rn(t, apq));
        matrix[q + dimension * q] =
          __dadd_rn(aqq, __dmul_rn(t, apq));
        matrix[p + dimension * q] = 0.0;
        matrix[q + dimension * p] = 0.0;
        for (int row = 0; row < dimension; ++row) {
          const double vip = vectors[row + dimension * p];
          const double viq = vectors[row + dimension * q];
          vectors[row + dimension * p] = __dadd_rn(
            __dmul_rn(cosine, vip), -__dmul_rn(sine, viq));
          vectors[row + dimension * q] = __dadd_rn(
            __dmul_rn(sine, vip), __dmul_rn(cosine, viq));
        }
      }
    }
  }
  for (int index = 0; index < dimension; ++index) {
    values[index] = matrix[index + dimension * index];
    if (!isfinite(values[index])) return false;
  }
  for (int left = 0; left < dimension - 1; ++left) {
    int selected = left;
    for (int right = left + 1; right < dimension; ++right) {
      if (values[right] < values[selected]) selected = right;
    }
    if (selected == left) continue;
    swap_double(values + left, values + selected);
    for (int row = 0; row < dimension; ++row) {
      swap_double(vectors + row + dimension * left,
                  vectors + row + dimension * selected);
    }
  }
  return converged;
}

__device__ bool solve_newton_system_direct(
    const double* hessian,
    const double* gradient,
    int dimension,
    double* matrix,
    double* solution) {
  for (int column = 0; column < dimension; ++column) {
    for (int row = 0; row < dimension; ++row) {
      matrix[row + dimension * column] =
        hessian[row + dimension * column];
    }
    solution[column] = -gradient[column];
  }
  for (int column = 0; column < dimension; ++column) {
    int pivot = column;
    double maximum = fabs(matrix[column + dimension * column]);
    for (int row = column + 1; row < dimension; ++row) {
      const double candidate = fabs(matrix[row + dimension * column]);
      if (candidate > maximum) {
        pivot = row;
        maximum = candidate;
      }
    }
    if (!(maximum > 0.0) || !isfinite(maximum)) return false;
    if (pivot != column) {
      for (int trailing = 0; trailing < dimension; ++trailing) {
        swap_double(
          matrix + column + dimension * trailing,
          matrix + pivot + dimension * trailing);
      }
      swap_double(solution + column, solution + pivot);
    }
    const double diagonal = matrix[column + dimension * column];
    for (int row = column + 1; row < dimension; ++row) {
      const double factor = __ddiv_rn(
        matrix[row + dimension * column], diagonal);
      matrix[row + dimension * column] = 0.0;
      for (int trailing = column + 1; trailing < dimension; ++trailing) {
        matrix[row + dimension * trailing] = __dadd_rn(
          matrix[row + dimension * trailing],
          -__dmul_rn(
            factor, matrix[column + dimension * trailing]));
      }
      solution[row] = __dadd_rn(
        solution[row], -__dmul_rn(factor, solution[column]));
    }
  }
  for (int row = dimension - 1; row >= 0; --row) {
    double value = solution[row];
    for (int column = row + 1; column < dimension; ++column) {
      value = __dadd_rn(
        value,
        -__dmul_rn(matrix[row + dimension * column], solution[column]));
    }
    const double diagonal = matrix[row + dimension * row];
    if (diagonal == 0.0 || !isfinite(diagonal)) return false;
    solution[row] = __ddiv_rn(value, diagonal);
    if (!isfinite(solution[row])) return false;
  }
  return true;
}

__device__ void evaluate_multi_penalty_target_device(
    const double* magic_r,
    const int* magic_pivot,
    const double* penalty_roots,
    const int* root_offsets,
    const int* penalty_ranks,
    const double* penalty_matrices,
    const double* target_y0,
    double target_squared_norm,
    const double* target_log_sp,
    DeviceEvaluationWorkspace* workspace,
    DeviceTargetEvaluation* result,
    int n,
    int q,
    int penalty_count,
    bool complete_evaluation,
    DevicePhaseTiming* phase_timing,
    double rank_tolerance,
    bool force_stable_svd = false,
    DeviceDecompositionTraceRecord* decomposition_trace = nullptr,
    unsigned int* decomposition_trace_counters = nullptr,
    unsigned int decomposition_trace_capacity = 0U,
    int trace_target = -1,
    int trace_iteration = -1,
    unsigned int trace_flags = 0U) {
  lapack312_multi::Workspace* decomposition = &workspace->decomposition;
  const int square = q * q;
  double* aggregate = decomposition->bidiagonal_vt;
  int* pivot = decomposition->iwork;
  double* factor_work = decomposition->work;
  unsigned long long phase_started = 0;
  if (threadIdx.x == 0 && phase_timing != nullptr) {
    phase_started = clock64();
  }
  if (threadIdx.x == 0) {
    result->rss = CUDART_NAN;
    result->edf = CUDART_NAN;
    result->score = CUDART_INF;
    result->condition = CUDART_NAN;
    result->aggregate_penalty_rank = 0;
    result->numerical_rank = 0;
    result->solver_info = 0;
    result->decomposition_route = kDecompositionUnknown;
    for (int index = 0;
         index < kMultiPenaltyGcvMaximumPenaltyCount; ++index) {
      result->gradient[index] = CUDART_NAN;
    }
    for (int index = 0;
         index < kMultiPenaltyGcvMaximumPenaltyCount *
           kMultiPenaltyGcvMaximumPenaltyCount; ++index) {
      result->hessian[index] = CUDART_NAN;
    }
    for (int index = 0;
         index < kMultiPenaltyGcvMaximumCoefficientDim; ++index) {
      result->coefficients[index] = CUDART_NAN;
    }
  }
  for (int element = threadIdx.x; element < square;
       element += blockDim.x) {
    double value = 0.0;
    for (int penalty = 0; penalty < penalty_count; ++penalty) {
      const double multiplier = glibc235::exp_fma_rn(
        target_log_sp[penalty]);
      value = __dadd_rn(
        value,
        __dmul_rn(multiplier,
          penalty_matrices[element + square * penalty]));
    }
    aggregate[element] = value;
  }
  __syncthreads();
  const int factor_rank = dpstf2_upper_block(
    aggregate, pivot, factor_work, q);
  if (threadIdx.x == 0) {
    result->aggregate_penalty_rank = factor_rank;
    decomposition->iwork[2 * q] = factor_rank;
  }
  __syncthreads();
  const int root_rank = decomposition->iwork[2 * q];
  if (root_rank <= 0 || root_rank + q > kMaximumRows) {
    if (threadIdx.x == 0) result->solver_info = -1;
    return;
  }
  const int rows = q + root_rank;
  for (int output = threadIdx.x; output < rows * q;
       output += blockDim.x) {
    const int row = output % rows;
    const int column = output / rows;
    double value = 0.0;
    if (row < q) {
      value = magic_r[row + q * column];
    } else {
      const int root_row = row - q;
      int factor_column = 0;
      while (factor_column < q && pivot[factor_column] != column + 1) {
        ++factor_column;
      }
      if (factor_column < q && root_row <= factor_column) {
        value = aggregate[root_row + q * factor_column];
      }
    }
    decomposition->a[output] = value;
  }
  __syncthreads();
  if (threadIdx.x == 0 && phase_timing != nullptr) {
    phase_timing->penalty_factor_augmentation_cycles +=
      clock64() - phase_started;
    phase_started = clock64();
  }
  int decomposition_trace_slot = -1;
  if (threadIdx.x == 0 && decomposition_trace != nullptr &&
      decomposition_trace_counters != nullptr &&
      decomposition_trace_capacity > 0U) {
    const unsigned int slot = atomicAdd(decomposition_trace_counters, 1U);
    if (slot < decomposition_trace_capacity) {
      decomposition_trace_slot = static_cast<int>(slot);
      DeviceDecompositionTraceRecord& record = decomposition_trace[slot];
      for (int penalty = 0;
           penalty < kMultiPenaltyGcvMaximumPenaltyCount; ++penalty) {
        record.log_sp_bits[penalty] = penalty < penalty_count ?
          static_cast<unsigned long long>(
            __double_as_longlong(target_log_sp[penalty])) : 0ULL;
      }
      record.target = trace_target;
      record.iteration = trace_iteration;
      record.flags = trace_flags |
        (complete_evaluation ?
          kDecompositionTraceCompleteCapability : 0U) |
        (force_stable_svd ? kDecompositionTraceForceStableSvd : 0U);
      record.route = kDecompositionUnknown;
    } else {
      atomicAdd(decomposition_trace_counters + 1, 1U);
    }
  }
  if (threadIdx.x == 0 && phase_timing != nullptr) {
    phase_started = clock64();
  }
  const int solver_info =
    lapack312_multi::small_dgesdd_left(
      rows, q, decomposition, complete_evaluation,
      phase_timing == nullptr ? nullptr :
        phase_timing->decomposition_stage_cycles,
      true, force_stable_svd);
  if (threadIdx.x == 0) {
    result->solver_info = solver_info;
    result->decomposition_route = decomposition->qr_basis_used != 0 ?
      kDecompositionGuardedQr : kDecompositionStableSvd;
    if (decomposition_trace_slot >= 0) {
      decomposition_trace[decomposition_trace_slot].route =
        result->decomposition_route;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && phase_timing != nullptr) {
    phase_timing->qr_svd_cycles += clock64() - phase_started;
  }
  if (solver_info != 0) return;

  if (threadIdx.x == 0 && phase_timing != nullptr) {
    phase_started = clock64();
  }
  if (threadIdx.x == 0) {
    int rank = q;
    const double threshold =
      __dmul_rn(decomposition->bidiagonal[0], rank_tolerance);
    while (rank > 0 && decomposition->bidiagonal[rank - 1] < threshold) {
      --rank;
    }
    result->numerical_rank = rank;
    decomposition->iwork[2 * q + 1] = rank;
  }
  __syncthreads();
  const int rank = decomposition->iwork[2 * q + 1];
  if (rank <= 0) {
    if (threadIdx.x == 0) result->solver_info = -2;
    return;
  }
  double* projected = decomposition->qr_tau;
  for (int component = threadIdx.x; component < rank;
       component += blockDim.x) {
    double value = 0.0;
    for (int row = 0; row < q; ++row) {
      value = rounded_multiply_add(
        value, decomposition->left_u[row + rows * component],
        target_y0[row]);
    }
    projected[component] = value;
  }
  __syncthreads();
  double* fitted_coordinates = decomposition->tau_q;
  for (int row = threadIdx.x; row < q; row += blockDim.x) {
    double value = 0.0;
    for (int component = 0; component < rank; ++component) {
      value = rounded_multiply_add(
        value, decomposition->left_u[row + rows * component],
        projected[component]);
    }
    fitted_coordinates[row] = value;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    double y_ay = 0.0;
    for (int component = 0; component < rank; ++component) {
      y_ay = rounded_multiply_add(
        y_ay, projected[component], projected[component]);
    }
    double y_aay = 0.0;
    for (int row = 0; row < q; ++row) {
      y_aay = rounded_multiply_add(
        y_aay, fitted_coordinates[row], fitted_coordinates[row]);
    }
    double edf = 0.0;
    for (int column = 0; column < rank; ++column) {
      for (int row = 0; row < q; ++row) {
        const double value = decomposition->left_u[row + rows * column];
        edf = rounded_multiply_add(edf, value, value);
      }
    }
    double rss = __dadd_rn(
      __dadd_rn(target_squared_norm, __dmul_rn(-2.0, y_ay)), y_aay);
    if (rss < 0.0 && isfinite(rss)) rss = 0.0;
    const double delta = static_cast<double>(n) - edf;
    if (!isfinite(rss) || !isfinite(edf) || delta <= kDenominatorFloor) {
      result->solver_info = -3;
    } else {
      const double score = __ddiv_rn(
        __dmul_rn(static_cast<double>(n), rss), __dmul_rn(delta, delta));
      result->rss = rss;
      result->edf = edf;
      result->score = score;
      result->condition = decomposition->qr_basis_used != 0 ?
        decomposition->qr_condition_estimate :
        __ddiv_rn(
          decomposition->bidiagonal[0],
          decomposition->bidiagonal[rank - 1]);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && phase_timing != nullptr) {
    phase_timing->score_construction_cycles += clock64() - phase_started;
    if (complete_evaluation) {
      ++phase_timing->complete_evaluation_count;
    } else {
      ++phase_timing->score_only_evaluation_count;
    }
    if (result->decomposition_route == kDecompositionGuardedQr) {
      ++phase_timing->guarded_qr_evaluation_count;
    } else {
      ++phase_timing->stable_svd_evaluation_count;
    }
  }
  if (result->solver_info != 0 || !complete_evaluation) return;
  if (threadIdx.x == 0 && phase_timing != nullptr) {
    phase_started = clock64();
  }
  const double rss = result->rss;
  const double delta = static_cast<double>(n) - result->edf;

  double* u1_u1 = decomposition->a;
  for (int element = threadIdx.x; element < rank * rank;
       element += blockDim.x) {
    const int row = element % rank;
    const int column = element / rank;
    if (row >= column) {
      double value = 0.0;
      for (int coordinate = 0; coordinate < q; ++coordinate) {
        value = rounded_multiply_add(
          value,
          decomposition->left_u[coordinate + rows * row],
          decomposition->left_u[coordinate + rows * column]);
      }
      u1_u1[row + rank * column] = value;
      u1_u1[column + rank * row] = value;
    }
  }
  __syncthreads();
  double* transformed = decomposition->r;
  double* intermediate = decomposition->bidiagonal_u;
  for (int penalty = 0; penalty < penalty_count; ++penalty) {
    const int penalty_rank = penalty_ranks[penalty];
    const int root_offset = root_offsets[penalty];
    for (int output = threadIdx.x; output < rank * penalty_rank;
         output += blockDim.x) {
      const int component = output % rank;
      const int root_column = output / rank;
      double value = 0.0;
      for (int coordinate = 0; coordinate < q; ++coordinate) {
        value = rounded_multiply_add(
          value,
          decomposition->bidiagonal_vt[component + q * coordinate],
          penalty_roots[
            coordinate + q * (root_offset + root_column)]);
      }
      transformed[component + rank * root_column] = __ddiv_rn(
        value, decomposition->bidiagonal[component]);
    }
    __syncthreads();
    for (int output = threadIdx.x; output < penalty_rank * rank;
         output += blockDim.x) {
      const int root_column = output % penalty_rank;
      const int column = output / penalty_rank;
      double value = 0.0;
      for (int component = 0; component < rank; ++component) {
        value = rounded_multiply_add(
          value,
          transformed[component + rank * root_column],
          u1_u1[component + rank * column]);
      }
      intermediate[root_column + penalty_rank * column] = value;
    }
    __syncthreads();
    double* metric = workspace->metric +
      static_cast<std::size_t>(penalty) * square;
    double* influence = workspace->influence +
      static_cast<std::size_t>(penalty) * square;
    for (int element = threadIdx.x; element < rank * rank;
         element += blockDim.x) {
      const int row = element % rank;
      const int column = element / rank;
      if (row >= column) {
        double value = 0.0;
        for (int root_column = 0; root_column < penalty_rank; ++root_column) {
          value = rounded_multiply_add(
            value,
            transformed[row + rank * root_column],
            transformed[column + rank * root_column]);
        }
        metric[row + rank * column] = value;
        metric[column + rank * row] = value;
      }
    }
    __syncthreads();
    for (int element = threadIdx.x; element < rank * rank;
         element += blockDim.x) {
      const int row = element % rank;
      const int column = element / rank;
      double value = 0.0;
      for (int root_column = 0; root_column < penalty_rank; ++root_column) {
        value = rounded_multiply_add(
          value,
          transformed[row + rank * root_column],
          intermediate[root_column + penalty_rank * column]);
      }
      influence[row + rank * column] = value;
    }
    __syncthreads();
    for (int output = threadIdx.x; output < rank;
         output += blockDim.x) {
      double metric_value = 0.0;
      double influence_value = 0.0;
      double y_influence_value = 0.0;
      for (int component = 0; component < rank; ++component) {
        metric_value = rounded_multiply_add(
          metric_value, projected[component],
          metric[component + rank * output]);
        y_influence_value = rounded_multiply_add(
          y_influence_value, projected[component],
          influence[component + rank * output]);
        influence_value = rounded_multiply_add(
          influence_value, projected[component],
          influence[output + rank * component]);
      }
      workspace->metric_y[penalty * q + output] = metric_value;
      workspace->influence_y[penalty * q + output] = influence_value;
      workspace->y_influence[penalty * q + output] = y_influence_value;
    }
    __syncthreads();
  }

  double* dnorm = workspace->derivative_norm;
  double* ddelta = workspace->derivative_delta;
  double* d2norm = workspace->second_derivative_norm;
  double* d2delta = workspace->second_derivative_delta;
  for (int i = threadIdx.x; i < penalty_count; i += blockDim.x) {
    const double lambda_i = glibc235::exp_fma_rn(
      target_log_sp[i]);
    const double* influence_i = workspace->influence +
      static_cast<std::size_t>(i) * square;
    double trace = 0.0;
    for (int component = 0; component < rank; ++component) {
      trace = __dadd_rn(trace, influence_i[component + rank * component]);
    }
    ddelta[i] = __dmul_rn(lambda_i, trace);
    double norm_derivative = 0.0;
    for (int component = 0; component < rank; ++component) {
      norm_derivative = rounded_multiply_add(
        norm_derivative, projected[component],
        __dadd_rn(
          workspace->metric_y[i * q + component],
          -workspace->influence_y[i * q + component]));
    }
    dnorm[i] = __dmul_rn(
      __dmul_rn(2.0, lambda_i), norm_derivative);
  }
  __syncthreads();
  for (int pair = threadIdx.x; pair < penalty_count * penalty_count;
       pair += blockDim.x) {
    const int i = pair % penalty_count;
    const int j = pair / penalty_count;
    if (i >= j) {
      const double* influence_i = workspace->influence +
        static_cast<std::size_t>(i) * square;
      const double* metric_j = workspace->metric +
        static_cast<std::size_t>(j) * square;
      double product_sum = 0.0;
      for (int element = 0; element < rank * rank; ++element) {
        product_sum = rounded_multiply_add(
          product_sum, metric_j[element], influence_i[element]);
      }
      double value = __dmul_rn(
        -2.0,
        glibc235::exp_fma_rn(__dadd_rn(
          target_log_sp[i], target_log_sp[j])));
      value = __dmul_rn(value, product_sum);
      if (i == j) value = __dadd_rn(value, ddelta[i]);
      d2delta[i + penalty_count * j] = value;
      d2delta[j + penalty_count * i] = value;
      double norm_value = 0.0;
      for (int component = 0; component < rank; ++component) {
        const double my_i = workspace->metric_y[i * q + component];
        const double my_j = workspace->metric_y[j * q + component];
        const double ky_i = workspace->influence_y[i * q + component];
        const double ky_j = workspace->influence_y[j * q + component];
        double term = __dadd_rn(
          __dmul_rn(my_i, ky_j), __dmul_rn(my_j, ky_i));
        term = __dadd_rn(term, -__dmul_rn(__dmul_rn(2.0, my_i), my_j));
        term = __dadd_rn(
          term,
          __dmul_rn(workspace->y_influence[i * q + component], my_j));
        norm_value = __dadd_rn(norm_value, term);
      }
      double norm_second_value = __dmul_rn(
        2.0,
        glibc235::exp_fma_rn(__dadd_rn(
          target_log_sp[i], target_log_sp[j])));
      norm_second_value = __dmul_rn(norm_second_value, norm_value);
      if (i == j) {
        norm_second_value = __dadd_rn(norm_second_value, dnorm[i]);
      }
      d2norm[i + penalty_count * j] = norm_second_value;
      d2norm[j + penalty_count * i] = norm_second_value;
    }
  }
  __syncthreads();
  const double score_scale = __ddiv_rn(
    static_cast<double>(n), __dmul_rn(delta, delta));
  const double delta_scale = __ddiv_rn(
    __dmul_rn(__dmul_rn(2.0, score_scale), rss), delta);
  const double cross_scale = __ddiv_rn(
    __dmul_rn(-2.0, score_scale), delta);
  const double curvature_scale = __ddiv_rn(
    __dmul_rn(3.0, delta_scale), delta);
  for (int i = threadIdx.x; i < penalty_count; i += blockDim.x) {
    result->gradient[i] = __dadd_rn(
      __dmul_rn(score_scale, dnorm[i]),
      -__dmul_rn(delta_scale, ddelta[i]));
  }
  for (int pair = threadIdx.x; pair < penalty_count * penalty_count;
       pair += blockDim.x) {
    const int i = pair % penalty_count;
    const int j = pair / penalty_count;
    if (i >= j) {
      double value = __dmul_rn(
        cross_scale,
        __dadd_rn(__dmul_rn(ddelta[j], dnorm[i]),
                  __dmul_rn(ddelta[i], dnorm[j])));
      value = __dadd_rn(
        value,
        __dmul_rn(score_scale, d2norm[i + penalty_count * j]));
      value = __dadd_rn(
        value,
        __dmul_rn(curvature_scale,
                  __dmul_rn(ddelta[i], ddelta[j])));
      value = __dadd_rn(
        value,
        -__dmul_rn(delta_scale, d2delta[i + penalty_count * j]));
      result->hessian[i + penalty_count * j] = value;
      result->hessian[j + penalty_count * i] = value;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && phase_timing != nullptr) {
    phase_timing->derivative_hessian_cycles += clock64() - phase_started;
  }

  for (int component = threadIdx.x; component < rank;
       component += blockDim.x) {
    fitted_coordinates[component] = __ddiv_rn(
      projected[component], decomposition->bidiagonal[component]);
  }
  __syncthreads();
  for (int row = threadIdx.x; row < q; row += blockDim.x) {
    double value = 0.0;
    for (int component = 0; component < rank; ++component) {
      value = rounded_multiply_add(
        value,
        decomposition->bidiagonal_vt[component + q * row],
        fitted_coordinates[component]);
    }
    result->coefficients[magic_pivot[row]] = value;
  }
}

__global__ void evaluate_multi_penalty_targets_kernel(
    const double* magic_r,
    const int* magic_pivot,
    const double* penalty_roots,
    const int* root_offsets,
    const int* penalty_ranks,
    const double* penalty_matrices,
    const double* y0,
    const double* squared_norm,
    const double* log_sp,
    DeviceEvaluationWorkspace* workspaces,
    DeviceTargetEvaluation* results,
    int n,
    int q,
    int penalty_count,
    int target_count,
    double rank_tolerance) {
  const int target = blockIdx.x;
  if (target >= target_count) return;
  evaluate_multi_penalty_target_device(
    magic_r, magic_pivot, penalty_roots, root_offsets, penalty_ranks,
    penalty_matrices,
    y0 + static_cast<std::size_t>(q) * target,
    squared_norm[target],
    log_sp + static_cast<std::size_t>(penalty_count) * target,
    workspaces + target, results + target, n, q, penalty_count, true,
    nullptr, rank_tolerance);
}

struct DeviceGroupedPrototypeParity {
  unsigned long long solver_route_mismatch_count;
  unsigned long long solver_info_mismatch_count;
  unsigned long long aggregate_rank_mismatch_count;
  unsigned long long augmented_rows_mismatch_count;
  unsigned long long r_bitwise_mismatch_count;
  unsigned long long explicit_q_bitwise_mismatch_count;
  unsigned long long left_basis_bitwise_mismatch_count;
  unsigned long long singular_value_bitwise_mismatch_count;
  unsigned long long right_basis_bitwise_mismatch_count;
  unsigned long long qr_condition_estimate_bitwise_mismatch_count;
};

__device__ bool grouped_prototype_double_bits_equal(double left,
                                                     double right) {
  return __double_as_longlong(left) == __double_as_longlong(right);
}

__device__ void initialize_grouped_prototype_result(
    DeviceTargetEvaluation* result) {
  if (threadIdx.x != 0) return;
  result->rss = CUDART_NAN;
  result->edf = CUDART_NAN;
  result->score = CUDART_INF;
  result->condition = CUDART_NAN;
  result->aggregate_penalty_rank = 0;
  result->numerical_rank = 0;
  result->solver_info = 0;
  result->decomposition_route = kDecompositionUnknown;
  for (int index = 0;
       index < kMultiPenaltyGcvMaximumPenaltyCount; ++index) {
    result->gradient[index] = CUDART_NAN;
  }
  for (int index = 0;
       index < kMultiPenaltyGcvMaximumPenaltyCount *
         kMultiPenaltyGcvMaximumPenaltyCount; ++index) {
    result->hessian[index] = CUDART_NAN;
  }
  for (int index = 0;
       index < kMultiPenaltyGcvMaximumCoefficientDim; ++index) {
    result->coefficients[index] = CUDART_NAN;
  }
}

__global__ void prepare_grouped_prototype_targets_kernel(
    const double* magic_r,
    const double* penalty_matrices,
    const double* log_sp,
    DeviceEvaluationWorkspace* workspaces,
    DeviceTargetEvaluation* results,
    int* augmented_rows,
    int q,
    int penalty_count,
    int target_count) {
  const int target = blockIdx.x;
  if (target >= target_count) return;
  DeviceEvaluationWorkspace* workspace = workspaces + target;
  DeviceTargetEvaluation* result = results + target;
  lapack312_multi::Workspace* decomposition = &workspace->decomposition;
  const double* target_log_sp =
    log_sp + static_cast<std::size_t>(penalty_count) * target;
  const int square = q * q;
  double* aggregate = decomposition->bidiagonal_vt;
  int* pivot = decomposition->iwork;
  double* factor_work = decomposition->work;

  initialize_grouped_prototype_result(result);
  if (threadIdx.x == 0) augmented_rows[target] = 0;
  for (int element = threadIdx.x; element < square;
       element += blockDim.x) {
    double value = 0.0;
    for (int penalty = 0; penalty < penalty_count; ++penalty) {
      const double multiplier = glibc235::exp_fma_rn(
        target_log_sp[penalty]);
      value = __dadd_rn(
        value,
        __dmul_rn(multiplier,
          penalty_matrices[element + square * penalty]));
    }
    aggregate[element] = value;
  }
  __syncthreads();
  const int factor_rank = dpstf2_upper_block(
    aggregate, pivot, factor_work, q);
  if (threadIdx.x == 0) {
    result->aggregate_penalty_rank = factor_rank;
    decomposition->iwork[2 * q] = factor_rank;
  }
  __syncthreads();
  const int root_rank = decomposition->iwork[2 * q];
  if (root_rank <= 0 || root_rank + q > kMaximumRows) {
    if (threadIdx.x == 0) result->solver_info = -1;
    return;
  }
  const int rows = q + root_rank;
  if (threadIdx.x == 0) augmented_rows[target] = rows;
  for (int output = threadIdx.x; output < rows * q;
       output += blockDim.x) {
    const int row = output % rows;
    const int column = output / rows;
    double value = 0.0;
    if (row < q) {
      value = magic_r[row + q * column];
    } else {
      const int root_row = row - q;
      int factor_column = 0;
      while (factor_column < q && pivot[factor_column] != column + 1) {
        ++factor_column;
      }
      if (factor_column < q && root_row <= factor_column) {
        value = aggregate[root_row + q * factor_column];
      }
    }
    decomposition->a[output] = value;
  }
}

__global__ void baseline_grouped_prototype_decomposition_kernel(
    DeviceEvaluationWorkspace* workspaces,
    DeviceTargetEvaluation* results,
    const int* augmented_rows,
    const int* force_stable_svd,
    int q,
    int target_count) {
  const int target = blockIdx.x;
  if (target >= target_count || augmented_rows[target] <= 0) return;
  lapack312_multi::Workspace* decomposition =
    &workspaces[target].decomposition;
  const int solver_info = lapack312_multi::small_dgesdd_left(
    augmented_rows[target], q, decomposition, true, nullptr, true,
    force_stable_svd[target] != 0);
  if (threadIdx.x == 0) {
    results[target].solver_info = solver_info;
    results[target].decomposition_route =
      decomposition->qr_basis_used != 0 ?
        kDecompositionGuardedQr : kDecompositionStableSvd;
  }
}

__global__ void grouped_prototype_guarded_qr_kernel(
    DeviceEvaluationWorkspace* workspaces,
    DeviceTargetEvaluation* results,
    const int* augmented_rows,
    const int* force_stable_svd,
    int* failure_targets,
    unsigned int* failure_count,
    int q,
    int target_count) {
  const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int target = global_thread >> 5;
  const int lane = threadIdx.x & 31;
  if (target >= target_count || augmented_rows[target] <= 0) return;
  lapack312_multi::Workspace* decomposition =
    &workspaces[target].decomposition;
  const int solver_info = lapack312_multi::grouped_guarded_qr_warp(
    augmented_rows[target], q, decomposition, true, true,
    force_stable_svd[target] != 0);
  if (lane == 0) {
    results[target].solver_info = solver_info;
    if (solver_info == 0 && decomposition->qr_basis_used != 0) {
      results[target].decomposition_route = kDecompositionGuardedQr;
    } else {
      results[target].decomposition_route = kDecompositionStableSvd;
      if (solver_info == 0) {
        const unsigned int slot = atomicAdd(failure_count, 1U);
        failure_targets[slot] = target;
      }
    }
  }
}

__global__ void grouped_prototype_stable_svd_kernel(
    DeviceEvaluationWorkspace* workspaces,
    DeviceTargetEvaluation* results,
    const int* augmented_rows,
    const int* failure_targets,
    const unsigned int* failure_count,
    int q) {
  const unsigned int queue_index = blockIdx.x;
  if (queue_index >= *failure_count) return;
  const int target = failure_targets[queue_index];
  const int solver_info =
    lapack312_multi::grouped_stable_svd_two_warp_continuation(
      augmented_rows[target], q, &workspaces[target].decomposition, true);
  if (threadIdx.x == 0) {
    results[target].solver_info = solver_info;
    results[target].decomposition_route = kDecompositionStableSvd;
  }
}

__global__ void compare_grouped_prototype_decompositions_kernel(
    const DeviceEvaluationWorkspace* baseline_workspaces,
    const DeviceEvaluationWorkspace* grouped_workspaces,
    const DeviceTargetEvaluation* baseline_results,
    const DeviceTargetEvaluation* grouped_results,
    const int* baseline_rows,
    const int* grouped_rows,
    DeviceGroupedPrototypeParity* parity,
    int q,
    int target_count) {
  const int target = blockIdx.x;
  if (target >= target_count) return;
  const lapack312_multi::Workspace& baseline =
    baseline_workspaces[target].decomposition;
  const lapack312_multi::Workspace& grouped =
    grouped_workspaces[target].decomposition;
  if (threadIdx.x == 0) {
    if (baseline_results[target].decomposition_route !=
        grouped_results[target].decomposition_route) {
      atomicAdd(&parity->solver_route_mismatch_count, 1ULL);
    }
    if (baseline_results[target].solver_info !=
        grouped_results[target].solver_info) {
      atomicAdd(&parity->solver_info_mismatch_count, 1ULL);
    }
    if (baseline_results[target].aggregate_penalty_rank !=
        grouped_results[target].aggregate_penalty_rank) {
      atomicAdd(&parity->aggregate_rank_mismatch_count, 1ULL);
    }
    if (baseline_rows[target] != grouped_rows[target]) {
      atomicAdd(&parity->augmented_rows_mismatch_count, 1ULL);
    }
    if (!grouped_prototype_double_bits_equal(
          baseline.qr_condition_estimate,
          grouped.qr_condition_estimate)) {
      atomicAdd(
        &parity->qr_condition_estimate_bitwise_mismatch_count, 1ULL);
    }
  }
  const int rows = baseline_rows[target];
  if (rows <= 0 || rows != grouped_rows[target]) return;
  for (int element = threadIdx.x; element < q * q;
       element += blockDim.x) {
    if (!grouped_prototype_double_bits_equal(
          baseline.r[element], grouped.r[element])) {
      atomicAdd(&parity->r_bitwise_mismatch_count, 1ULL);
    }
    if (!grouped_prototype_double_bits_equal(
          baseline.bidiagonal_vt[element],
          grouped.bidiagonal_vt[element])) {
      atomicAdd(&parity->right_basis_bitwise_mismatch_count, 1ULL);
    }
  }
  for (int element = threadIdx.x; element < rows * q;
       element += blockDim.x) {
    if (!grouped_prototype_double_bits_equal(
          baseline.a[element], grouped.a[element])) {
      atomicAdd(&parity->explicit_q_bitwise_mismatch_count, 1ULL);
    }
    if (!grouped_prototype_double_bits_equal(
          baseline.left_u[element], grouped.left_u[element])) {
      atomicAdd(&parity->left_basis_bitwise_mismatch_count, 1ULL);
    }
  }
  for (int element = threadIdx.x; element < q;
       element += blockDim.x) {
    if (!grouped_prototype_double_bits_equal(
          baseline.bidiagonal[element], grouped.bidiagonal[element])) {
      atomicAdd(&parity->singular_value_bitwise_mismatch_count, 1ULL);
    }
  }
}

// Mirrors the certified post-decomposition sequence in
// evaluate_multi_penalty_target_device(). Keeping this prototype helper
// separate avoids changing the production optimizer's monolithic state
// machine before the grouped execution shape clears its stop/go gate.
__device__ void complete_grouped_prototype_target_device(
    const int* magic_pivot,
    const double* penalty_roots,
    const int* root_offsets,
    const int* penalty_ranks,
    const double* target_y0,
    double target_squared_norm,
    const double* target_log_sp,
    DeviceEvaluationWorkspace* workspace,
    DeviceTargetEvaluation* result,
    int n,
    int q,
    int penalty_count,
    double rank_tolerance) {
  if (result->solver_info != 0) return;
  lapack312_multi::Workspace* decomposition = &workspace->decomposition;
  const int rows = q + result->aggregate_penalty_rank;
  const int square = q * q;
  if (threadIdx.x == 0) {
    int rank = q;
    const double threshold =
      __dmul_rn(decomposition->bidiagonal[0], rank_tolerance);
    while (rank > 0 && decomposition->bidiagonal[rank - 1] < threshold) {
      --rank;
    }
    result->numerical_rank = rank;
    decomposition->iwork[2 * q + 1] = rank;
  }
  __syncthreads();
  const int rank = decomposition->iwork[2 * q + 1];
  if (rank <= 0) {
    if (threadIdx.x == 0) result->solver_info = -2;
    return;
  }
  double* projected = decomposition->qr_tau;
  for (int component = threadIdx.x; component < rank;
       component += blockDim.x) {
    double value = 0.0;
    for (int row = 0; row < q; ++row) {
      value = rounded_multiply_add(
        value, decomposition->left_u[row + rows * component],
        target_y0[row]);
    }
    projected[component] = value;
  }
  __syncthreads();
  double* fitted_coordinates = decomposition->tau_q;
  for (int row = threadIdx.x; row < q; row += blockDim.x) {
    double value = 0.0;
    for (int component = 0; component < rank; ++component) {
      value = rounded_multiply_add(
        value, decomposition->left_u[row + rows * component],
        projected[component]);
    }
    fitted_coordinates[row] = value;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    double y_ay = 0.0;
    for (int component = 0; component < rank; ++component) {
      y_ay = rounded_multiply_add(
        y_ay, projected[component], projected[component]);
    }
    double y_aay = 0.0;
    for (int row = 0; row < q; ++row) {
      y_aay = rounded_multiply_add(
        y_aay, fitted_coordinates[row], fitted_coordinates[row]);
    }
    double edf = 0.0;
    for (int column = 0; column < rank; ++column) {
      for (int row = 0; row < q; ++row) {
        const double value = decomposition->left_u[row + rows * column];
        edf = rounded_multiply_add(edf, value, value);
      }
    }
    double rss = __dadd_rn(
      __dadd_rn(target_squared_norm, __dmul_rn(-2.0, y_ay)), y_aay);
    if (rss < 0.0 && isfinite(rss)) rss = 0.0;
    const double delta = static_cast<double>(n) - edf;
    if (!isfinite(rss) || !isfinite(edf) || delta <= kDenominatorFloor) {
      result->solver_info = -3;
    } else {
      const double score = __ddiv_rn(
        __dmul_rn(static_cast<double>(n), rss),
        __dmul_rn(delta, delta));
      result->rss = rss;
      result->edf = edf;
      result->score = score;
      result->condition = decomposition->qr_basis_used != 0 ?
        decomposition->qr_condition_estimate :
        __ddiv_rn(
          decomposition->bidiagonal[0],
          decomposition->bidiagonal[rank - 1]);
    }
  }
  __syncthreads();
  if (result->solver_info != 0) return;
  const double rss = result->rss;
  const double delta = static_cast<double>(n) - result->edf;

  double* u1_u1 = decomposition->a;
  for (int element = threadIdx.x; element < rank * rank;
       element += blockDim.x) {
    const int row = element % rank;
    const int column = element / rank;
    if (row >= column) {
      double value = 0.0;
      for (int coordinate = 0; coordinate < q; ++coordinate) {
        value = rounded_multiply_add(
          value,
          decomposition->left_u[coordinate + rows * row],
          decomposition->left_u[coordinate + rows * column]);
      }
      u1_u1[row + rank * column] = value;
      u1_u1[column + rank * row] = value;
    }
  }
  __syncthreads();
  double* transformed = decomposition->r;
  double* intermediate = decomposition->bidiagonal_u;
  for (int penalty = 0; penalty < penalty_count; ++penalty) {
    const int penalty_rank = penalty_ranks[penalty];
    const int root_offset = root_offsets[penalty];
    for (int output = threadIdx.x; output < rank * penalty_rank;
         output += blockDim.x) {
      const int component = output % rank;
      const int root_column = output / rank;
      double value = 0.0;
      for (int coordinate = 0; coordinate < q; ++coordinate) {
        value = rounded_multiply_add(
          value,
          decomposition->bidiagonal_vt[component + q * coordinate],
          penalty_roots[
            coordinate + q * (root_offset + root_column)]);
      }
      transformed[component + rank * root_column] = __ddiv_rn(
        value, decomposition->bidiagonal[component]);
    }
    __syncthreads();
    for (int output = threadIdx.x; output < penalty_rank * rank;
         output += blockDim.x) {
      const int root_column = output % penalty_rank;
      const int column = output / penalty_rank;
      double value = 0.0;
      for (int component = 0; component < rank; ++component) {
        value = rounded_multiply_add(
          value,
          transformed[component + rank * root_column],
          u1_u1[component + rank * column]);
      }
      intermediate[root_column + penalty_rank * column] = value;
    }
    __syncthreads();
    double* metric = workspace->metric +
      static_cast<std::size_t>(penalty) * square;
    double* influence = workspace->influence +
      static_cast<std::size_t>(penalty) * square;
    for (int element = threadIdx.x; element < rank * rank;
         element += blockDim.x) {
      const int row = element % rank;
      const int column = element / rank;
      if (row >= column) {
        double value = 0.0;
        for (int root_column = 0; root_column < penalty_rank;
             ++root_column) {
          value = rounded_multiply_add(
            value,
            transformed[row + rank * root_column],
            transformed[column + rank * root_column]);
        }
        metric[row + rank * column] = value;
        metric[column + rank * row] = value;
      }
    }
    __syncthreads();
    for (int element = threadIdx.x; element < rank * rank;
         element += blockDim.x) {
      const int row = element % rank;
      const int column = element / rank;
      double value = 0.0;
      for (int root_column = 0; root_column < penalty_rank;
           ++root_column) {
        value = rounded_multiply_add(
          value,
          transformed[row + rank * root_column],
          intermediate[root_column + penalty_rank * column]);
      }
      influence[row + rank * column] = value;
    }
    __syncthreads();
    for (int output = threadIdx.x; output < rank;
         output += blockDim.x) {
      double metric_value = 0.0;
      double influence_value = 0.0;
      double y_influence_value = 0.0;
      for (int component = 0; component < rank; ++component) {
        metric_value = rounded_multiply_add(
          metric_value, projected[component],
          metric[component + rank * output]);
        y_influence_value = rounded_multiply_add(
          y_influence_value, projected[component],
          influence[component + rank * output]);
        influence_value = rounded_multiply_add(
          influence_value, projected[component],
          influence[output + rank * component]);
      }
      workspace->metric_y[penalty * q + output] = metric_value;
      workspace->influence_y[penalty * q + output] = influence_value;
      workspace->y_influence[penalty * q + output] = y_influence_value;
    }
    __syncthreads();
  }

  double* dnorm = workspace->derivative_norm;
  double* ddelta = workspace->derivative_delta;
  double* d2norm = workspace->second_derivative_norm;
  double* d2delta = workspace->second_derivative_delta;
  for (int i = threadIdx.x; i < penalty_count; i += blockDim.x) {
    const double lambda_i = glibc235::exp_fma_rn(target_log_sp[i]);
    const double* influence_i = workspace->influence +
      static_cast<std::size_t>(i) * square;
    double trace = 0.0;
    for (int component = 0; component < rank; ++component) {
      trace = __dadd_rn(trace, influence_i[component + rank * component]);
    }
    ddelta[i] = __dmul_rn(lambda_i, trace);
    double norm_derivative = 0.0;
    for (int component = 0; component < rank; ++component) {
      norm_derivative = rounded_multiply_add(
        norm_derivative, projected[component],
        __dadd_rn(
          workspace->metric_y[i * q + component],
          -workspace->influence_y[i * q + component]));
    }
    dnorm[i] = __dmul_rn(
      __dmul_rn(2.0, lambda_i), norm_derivative);
  }
  __syncthreads();
  for (int pair = threadIdx.x; pair < penalty_count * penalty_count;
       pair += blockDim.x) {
    const int i = pair % penalty_count;
    const int j = pair / penalty_count;
    if (i >= j) {
      const double* influence_i = workspace->influence +
        static_cast<std::size_t>(i) * square;
      const double* metric_j = workspace->metric +
        static_cast<std::size_t>(j) * square;
      double product_sum = 0.0;
      for (int element = 0; element < rank * rank; ++element) {
        product_sum = rounded_multiply_add(
          product_sum, metric_j[element], influence_i[element]);
      }
      double value = __dmul_rn(
        -2.0,
        glibc235::exp_fma_rn(__dadd_rn(
          target_log_sp[i], target_log_sp[j])));
      value = __dmul_rn(value, product_sum);
      if (i == j) value = __dadd_rn(value, ddelta[i]);
      d2delta[i + penalty_count * j] = value;
      d2delta[j + penalty_count * i] = value;
      double norm_value = 0.0;
      for (int component = 0; component < rank; ++component) {
        const double my_i = workspace->metric_y[i * q + component];
        const double my_j = workspace->metric_y[j * q + component];
        const double ky_i = workspace->influence_y[i * q + component];
        const double ky_j = workspace->influence_y[j * q + component];
        double term = __dadd_rn(
          __dmul_rn(my_i, ky_j), __dmul_rn(my_j, ky_i));
        term = __dadd_rn(
          term, -__dmul_rn(__dmul_rn(2.0, my_i), my_j));
        term = __dadd_rn(
          term,
          __dmul_rn(
            workspace->y_influence[i * q + component], my_j));
        norm_value = __dadd_rn(norm_value, term);
      }
      double norm_second_value = __dmul_rn(
        2.0,
        glibc235::exp_fma_rn(__dadd_rn(
          target_log_sp[i], target_log_sp[j])));
      norm_second_value = __dmul_rn(norm_second_value, norm_value);
      if (i == j) {
        norm_second_value = __dadd_rn(norm_second_value, dnorm[i]);
      }
      d2norm[i + penalty_count * j] = norm_second_value;
      d2norm[j + penalty_count * i] = norm_second_value;
    }
  }
  __syncthreads();
  const double score_scale = __ddiv_rn(
    static_cast<double>(n), __dmul_rn(delta, delta));
  const double delta_scale = __ddiv_rn(
    __dmul_rn(__dmul_rn(2.0, score_scale), rss), delta);
  const double cross_scale = __ddiv_rn(
    __dmul_rn(-2.0, score_scale), delta);
  const double curvature_scale = __ddiv_rn(
    __dmul_rn(3.0, delta_scale), delta);
  for (int i = threadIdx.x; i < penalty_count; i += blockDim.x) {
    result->gradient[i] = __dadd_rn(
      __dmul_rn(score_scale, dnorm[i]),
      -__dmul_rn(delta_scale, ddelta[i]));
  }
  for (int pair = threadIdx.x; pair < penalty_count * penalty_count;
       pair += blockDim.x) {
    const int i = pair % penalty_count;
    const int j = pair / penalty_count;
    if (i >= j) {
      double value = __dmul_rn(
        cross_scale,
        __dadd_rn(__dmul_rn(ddelta[j], dnorm[i]),
                  __dmul_rn(ddelta[i], dnorm[j])));
      value = __dadd_rn(
        value,
        __dmul_rn(score_scale, d2norm[i + penalty_count * j]));
      value = __dadd_rn(
        value,
        __dmul_rn(curvature_scale,
                  __dmul_rn(ddelta[i], ddelta[j])));
      value = __dadd_rn(
        value,
        -__dmul_rn(delta_scale, d2delta[i + penalty_count * j]));
      result->hessian[i + penalty_count * j] = value;
      result->hessian[j + penalty_count * i] = value;
    }
  }
  __syncthreads();

  for (int component = threadIdx.x; component < rank;
       component += blockDim.x) {
    fitted_coordinates[component] = __ddiv_rn(
      projected[component], decomposition->bidiagonal[component]);
  }
  __syncthreads();
  for (int row = threadIdx.x; row < q; row += blockDim.x) {
    double value = 0.0;
    for (int component = 0; component < rank; ++component) {
      value = rounded_multiply_add(
        value,
        decomposition->bidiagonal_vt[component + q * row],
        fitted_coordinates[component]);
    }
    result->coefficients[magic_pivot[row]] = value;
  }
}

__global__ void complete_grouped_prototype_targets_kernel(
    const int* magic_pivot,
    const double* penalty_roots,
    const int* root_offsets,
    const int* penalty_ranks,
    const double* y0,
    const double* squared_norm,
    const double* log_sp,
    DeviceEvaluationWorkspace* workspaces,
    DeviceTargetEvaluation* results,
    int n,
    int q,
    int penalty_count,
    int target_count,
    double rank_tolerance) {
  const int target = blockIdx.x;
  if (target >= target_count) return;
  complete_grouped_prototype_target_device(
    magic_pivot, penalty_roots, root_offsets, penalty_ranks,
    y0 + static_cast<std::size_t>(q) * target,
    squared_norm[target],
    log_sp + static_cast<std::size_t>(penalty_count) * target,
    workspaces + target, results + target, n, q, penalty_count,
    rank_tolerance);
}

__device__ double direct_qr_residual_score(
    const double* magic_r,
    const int* magic_pivot,
    const double* target_y0,
    double target_squared_norm,
    const DeviceTargetEvaluation& evaluation,
    int n,
    int q) {
  double y0_squared_norm = 0.0;
  double residual_squared_norm = 0.0;
  for (int row = 0; row < q; ++row) {
    y0_squared_norm = rounded_multiply_add(
      y0_squared_norm, target_y0[row], target_y0[row]);
    double fitted = 0.0;
    for (int column = row; column < q; ++column) {
      fitted = rounded_multiply_add(
        fitted, magic_r[row + q * column],
        evaluation.coefficients[magic_pivot[column]]);
    }
    const double residual = __dadd_rn(target_y0[row], -fitted);
    residual_squared_norm = rounded_multiply_add(
      residual_squared_norm, residual, residual);
  }
  const double rss = __dadd_rn(
    __dadd_rn(target_squared_norm, -y0_squared_norm),
    residual_squared_norm);
  const double delta = __dadd_rn(
    static_cast<double>(n), -evaluation.edf);
  return __ddiv_rn(
    __dmul_rn(static_cast<double>(n), rss),
    __dmul_rn(delta, delta));
}

__global__ void optimize_multi_penalty_targets_kernel(
    const double* magic_r,
    const int* magic_pivot,
    const double* penalty_roots,
    const int* root_offsets,
    const int* penalty_ranks,
    const double* penalty_matrices,
    const double* y0,
    const double* squared_norm,
    const double* initial_log_sp,
    DeviceEvaluationWorkspace* workspaces,
    DeviceOptimizerState* states,
    const DeviceOptimizerState* primary_states,
    int n,
    int q,
    int penalty_count,
    int target_count,
    double convergence_tolerance,
    double gradient_tolerance_factor,
    int max_step_halving,
    int max_iterations,
    double max_newton_step,
    double boundary_probe_step,
    int max_boundary_probes,
    double rank_tolerance,
    bool stability_replay,
    bool force_stable_svd,
    DeviceDecompositionTraceRecord* decomposition_trace,
    unsigned int* decomposition_trace_counters,
    unsigned int decomposition_trace_capacity) {
  const int target = blockIdx.x;
  if (target >= target_count) return;
  if (stability_replay &&
      (primary_states == nullptr ||
       !is_stability_replay_candidate(primary_states[target]))) {
    return;
  }
  DeviceEvaluationWorkspace* workspace = workspaces + target;
  DeviceOptimizerState* state = states + target;
  const bool long_trajectory_replay = stability_replay &&
    primary_states != nullptr &&
    requires_long_trajectory_stability_replay(primary_states[target]);
  const bool dense_boundary_replay = stability_replay &&
    primary_states != nullptr &&
    requires_dense_boundary_stability_replay(primary_states[target]);
  const bool high_condition_replay = stability_replay &&
    primary_states != nullptr &&
    requires_high_condition_stability_replay(primary_states[target]);
  const bool ambiguous_step_replay = stability_replay &&
    primary_states != nullptr &&
    requires_ambiguous_step_stability_replay(primary_states[target]);
  const bool rejected_boundary_replay = stability_replay &&
    primary_states != nullptr &&
    requires_rejected_boundary_stability_replay(primary_states[target]);
  const bool direct_newton_replay = stability_replay &&
    primary_states != nullptr &&
    requires_direct_newton_stability_replay(primary_states[target]);
  const bool full_stability_replay = stability_replay &&
    primary_states != nullptr &&
    requires_full_stability_replay(primary_states[target]);
  const bool target_force_stable_svd =
    force_stable_svd && !direct_newton_replay;
  const unsigned int stability_trace_flag = stability_replay ?
    kDecompositionTraceStabilityReplay : 0U;
  const double* target_y0 = y0 + static_cast<std::size_t>(q) * target;
  const double target_squared_norm = squared_norm[target];

  if (threadIdx.x == 0) {
    state->minimum_score = CUDART_INF;
    state->score_reduction = 1e10;
    state->rms_gradient = 0.0;
    state->optimizer_iterations = 0;
    state->score_calls = 0;
    state->objective_calls = 0;
    state->step_halving_count = 0;
    state->newton_trial_count = 0;
    state->steepest_descent_trial_count = 0;
    state->boundary_probe_count = 0;
    state->boundary_accepted_count = 0;
    state->fully_converged = 0;
    state->hessian_positive_definite = 0;
    state->step_failed = 0;
    state->optimizer_status = 0;
    state->penalty_count = penalty_count;
    state->use_steepest_descent = 0;
    state->converged = 0;
    state->trying = 0;
    state->tries = 0;
    state->selected_evaluation_reuse_count = 0;
    state->ambiguous_step_descent_count = 0;
    state->ambiguous_step_non_descent_count = 0;
    state->stability_replay_attempted = stability_replay ? 1 : 0;
    state->stability_replay_screened =
      stability_replay && !full_stability_replay ? 1 : 0;
    state->stability_replay_selected = 0;
    state->stability_replay_error = 0;
    state->stability_replay_long_trajectory_reason =
      long_trajectory_replay ? 1 : 0;
    state->stability_replay_dense_boundary_reason =
      dense_boundary_replay ? 1 : 0;
    state->stability_replay_high_condition_reason =
      high_condition_replay ? 1 : 0;
    state->stability_replay_ambiguous_step_reason =
      ambiguous_step_replay ? 1 : 0;
    state->stability_replay_rejected_boundary_reason =
      rejected_boundary_replay ? 1 : 0;
    state->stability_replay_direct_newton_reason =
      direct_newton_replay ? 1 : 0;
    state->stability_replay_dense_score_guard_rejected = 0;
    state->stability_replay_extrapolation_applied = 0;
    state->terminal_boundary_confirmation_count = 0;
    state->terminal_boundary_confirmation_accepted_count = 0;
    state->terminal_boundary_confirmation_rejected_count = 0;
    state->terminal_boundary_confirmation_strong_delta_accepted_count = 0;
    state->terminal_boundary_confirmation_identity_tie_accepted_count = 0;
    state->terminal_boundary_confirmation_delta_identity_accepted_count = 0;
    state->terminal_boundary_confirmation_pending = 0;
    state->stability_replay_log_sp_spread = 0.0;
    state->stability_replay_max_extrapolation = 0.0;
    state->terminal_boundary_confirmation_max_identity_disagreement = 0.0;
    state->terminal_boundary_confirmation_max_identity_ratio = 0.0;
    state->terminal_boundary_confirmation_max_delta_disagreement = 0.0;
    state->terminal_boundary_confirmation_max_delta_ratio = 0.0;
    state->terminal_boundary_primary_current_score = CUDART_NAN;
    state->terminal_boundary_primary_trial_score = CUDART_NAN;
    state->terminal_boundary_stable_current_score = CUDART_NAN;
    state->terminal_boundary_stable_current_direct_score = CUDART_NAN;
    state->terminal_boundary_stable_trial_direct_score = CUDART_NAN;
    state->phase_timing.penalty_factor_augmentation_cycles = 0;
    state->phase_timing.qr_svd_cycles = 0;
    state->phase_timing.score_construction_cycles = 0;
    state->phase_timing.derivative_hessian_cycles = 0;
    for (int stage = 0; stage < 8; ++stage) {
      state->phase_timing.decomposition_stage_cycles[stage] = 0;
    }
    state->phase_timing.complete_evaluation_count = 0;
    state->phase_timing.score_only_evaluation_count = 0;
    state->phase_timing.guarded_qr_evaluation_count = 0;
    state->phase_timing.stable_svd_evaluation_count = 0;
    state->discarded_phase_timing.penalty_factor_augmentation_cycles = 0;
    state->discarded_phase_timing.qr_svd_cycles = 0;
    state->discarded_phase_timing.score_construction_cycles = 0;
    state->discarded_phase_timing.derivative_hessian_cycles = 0;
    for (int stage = 0; stage < 8; ++stage) {
      state->discarded_phase_timing.decomposition_stage_cycles[stage] = 0;
    }
    state->discarded_phase_timing.complete_evaluation_count = 0;
    state->discarded_phase_timing.score_only_evaluation_count = 0;
    state->discarded_phase_timing.guarded_qr_evaluation_count = 0;
    state->discarded_phase_timing.stable_svd_evaluation_count = 0;
    state->terminal_boundary_confirmation_phase_timing
      .penalty_factor_augmentation_cycles = 0;
    state->terminal_boundary_confirmation_phase_timing.qr_svd_cycles = 0;
    state->terminal_boundary_confirmation_phase_timing
      .score_construction_cycles = 0;
    state->terminal_boundary_confirmation_phase_timing
      .derivative_hessian_cycles = 0;
    for (int stage = 0; stage < 8; ++stage) {
      state->terminal_boundary_confirmation_phase_timing
        .decomposition_stage_cycles[stage] = 0;
    }
    state->terminal_boundary_confirmation_phase_timing
      .complete_evaluation_count = 0;
    state->terminal_boundary_confirmation_phase_timing
      .score_only_evaluation_count = 0;
    state->terminal_boundary_confirmation_phase_timing
      .guarded_qr_evaluation_count = 0;
    state->terminal_boundary_confirmation_phase_timing
      .stable_svd_evaluation_count = 0;
    for (int penalty = 0; penalty < penalty_count; ++penalty) {
      state->log_sp[penalty] = initial_log_sp[penalty];
      state->trial_log_sp[penalty] = initial_log_sp[penalty];
      state->gradient[penalty] = 0.0;
      state->newton_step[penalty] = 0.0;
      state->steepest_step[penalty] = 0.0;
      state->working_step[penalty] = 0.0;
      state->hessian_eigenvalues[penalty] = 0.0;
      state->boundary_status[penalty] = -1;
    }
  }
  __syncthreads();
  if (stability_replay && !full_stability_replay) return;
  evaluate_multi_penalty_target_device(
    magic_r, magic_pivot, penalty_roots, root_offsets, penalty_ranks,
    penalty_matrices, target_y0, target_squared_norm, state->log_sp,
    workspace, &state->current, n, q, penalty_count, true,
    &state->phase_timing, rank_tolerance, target_force_stable_svd,
    decomposition_trace, decomposition_trace_counters,
    decomposition_trace_capacity, target, 0,
    kDecompositionTraceInitial | stability_trace_flag);
  __syncthreads();
  if (threadIdx.x == 0) {
    state->score_calls = 1;
    state->objective_calls = 1;
    if (state->current.solver_info != 0 ||
        !isfinite(state->current.score)) {
      state->optimizer_status = 1;
    } else {
      state->minimum_score = state->current.score;
    }
  }
  __syncthreads();
  if (state->optimizer_status != 0) return;

  for (int iteration = 1; iteration <= max_iterations; ++iteration) {
    if (threadIdx.x == 0) {
      state->optimizer_iterations = iteration;
      state->tries = 0;
      state->trying = iteration > 1 ? 1 : 0;
      if (iteration > 1) {
        const double* source = state->use_steepest_descent != 0 ?
          state->steepest_step : state->newton_step;
        for (int penalty = 0; penalty < penalty_count; ++penalty) {
          state->working_step[penalty] = source[penalty];
        }
      }
    }
    __syncthreads();

    if (iteration > 1) {
      for (int trial_index = 1; trial_index <= max_step_halving;
           ++trial_index) {
        if (threadIdx.x == 0 && state->trying != 0) {
          state->tries = trial_index;
          if (trial_index == 4 && state->use_steepest_descent == 0) {
            state->use_steepest_descent = 1;
            for (int penalty = 0; penalty < penalty_count; ++penalty) {
              state->working_step[penalty] =
                state->steepest_step[penalty];
            }
          }
          if (state->use_steepest_descent == 0) {
            ++state->newton_trial_count;
          } else {
            ++state->steepest_descent_trial_count;
          }
          for (int penalty = 0; penalty < penalty_count; ++penalty) {
            state->trial_log_sp[penalty] = __dadd_rn(
              state->log_sp[penalty], state->working_step[penalty]);
          }
        }
        __syncthreads();
        if (state->trying == 0) break;
        evaluate_multi_penalty_target_device(
          magic_r, magic_pivot, penalty_roots, root_offsets, penalty_ranks,
          penalty_matrices, target_y0, target_squared_norm,
          state->trial_log_sp, workspace, &state->trial, n, q,
          penalty_count, true, &state->phase_timing, rank_tolerance,
          target_force_stable_svd, decomposition_trace,
          decomposition_trace_counters, decomposition_trace_capacity,
          target, iteration,
          (state->use_steepest_descent == 0 ?
            kDecompositionTraceNewtonTrial :
            kDecompositionTraceSteepestTrial) |
            (trial_index > 1 ? kDecompositionTraceStepHalving : 0U) |
            stability_trace_flag);
        __syncthreads();
        if (threadIdx.x == 0) {
          ++state->score_calls;
          ++state->objective_calls;
          if (state->trial.solver_info != 0 ||
              !isfinite(state->trial.score)) {
            state->optimizer_status = 2;
            state->trying = 0;
          } else {
            double directional_derivative = 0.0;
            double maximum_step = 0.0;
            double maximum_log_sp = 0.0;
            for (int penalty = 0; penalty < penalty_count; ++penalty) {
              directional_derivative = rounded_multiply_add(
                directional_derivative, state->gradient[penalty],
                state->working_step[penalty]);
              maximum_step = fmax(
                maximum_step, fabs(state->working_step[penalty]));
              maximum_log_sp = fmax(
                maximum_log_sp, fabs(state->log_sp[penalty]));
            }
            const double score_delta = __dadd_rn(
              state->trial.score, -state->minimum_score);
            const double comparison_slack = __dmul_rn(
              16.0 * kLapackEpsilon,
              __dadd_rn(1.0, fabs(state->minimum_score)));
            const double convergence_step = __dmul_rn(
              convergence_tolerance, __dadd_rn(1.0, maximum_log_sp));
            const double ambiguity_tolerance = __dmul_rn(
              kAmbiguousStepRelativeTolerance,
              __dadd_rn(1.0, fabs(state->minimum_score)));
            const bool ambiguous_first_newton_step =
              state->use_steepest_descent == 0 &&
              trial_index == 1 && directional_derivative < 0.0 &&
              maximum_step > convergence_step &&
              fabs(score_delta) <= ambiguity_tolerance;
            double smallest_hessian_eigenvalue = fabs(
              state->hessian_eigenvalues[0]);
            double largest_hessian_eigenvalue =
              smallest_hessian_eigenvalue;
            for (int penalty = 1; penalty < penalty_count; ++penalty) {
              const double value = fabs(
                state->hessian_eigenvalues[penalty]);
              smallest_hessian_eigenvalue = fmin(
                smallest_hessian_eigenvalue, value);
              largest_hessian_eigenvalue = fmax(
                largest_hessian_eigenvalue, value);
            }
            const bool conservative_hessian_condition =
              smallest_hessian_eigenvalue > 0.0 &&
              largest_hessian_eigenvalue >= __dmul_rn(
                kAmbiguousStepConservativeMinimumHessianCondition,
                smallest_hessian_eigenvalue);
            bool material_direct_delta_disagreement = false;
            if (!stability_replay && ambiguous_first_newton_step &&
                score_delta < 0.0 && conservative_hessian_condition) {
              const double current_direct_score = direct_qr_residual_score(
                magic_r, magic_pivot, target_y0, target_squared_norm,
                state->current, n, q);
              const double trial_direct_score = direct_qr_residual_score(
                magic_r, magic_pivot, target_y0, target_squared_norm,
                state->trial, n, q);
              const double direct_score_delta = __dadd_rn(
                trial_direct_score, -current_direct_score);
              material_direct_delta_disagreement =
                fabs(__dadd_rn(score_delta, -direct_score_delta)) >=
                  __dmul_rn(
                    kAmbiguousStepDirectDeltaDisagreementFraction,
                    ambiguity_tolerance);
            }
            if (!stability_replay && ambiguous_first_newton_step) {
              if (score_delta < -__dmul_rn(
                    kAmbiguousStepConservativeDescentFraction,
                    ambiguity_tolerance) &&
                  conservative_hessian_condition &&
                  material_direct_delta_disagreement) {
                ++state->ambiguous_step_descent_count;
              } else if (score_delta >= 0.0) {
                ++state->ambiguous_step_non_descent_count;
              }
            }
            const bool conservative_replay_halving =
              stability_replay && ambiguous_first_newton_step &&
              primary_states[target].ambiguous_step_descent_count > 0;
            // Resolve only near-convergence Newton comparisons whose score
            // ordering is below the binary64 LAPACK/CUDA arithmetic floor.
            const bool roundoff_limited_newton_descent =
              state->use_steepest_descent == 0 &&
              directional_derivative < 0.0 &&
              maximum_step <= convergence_step &&
              score_delta <= comparison_slack;
            const bool accepted =
              !conservative_replay_halving &&
              (state->trial.score < state->minimum_score ||
               roundoff_limited_newton_descent);
            if (accepted) {
              state->score_reduction = __dadd_rn(
                state->minimum_score, -state->trial.score);
              state->minimum_score = state->trial.score;
              for (int penalty = 0; penalty < penalty_count; ++penalty) {
                state->log_sp[penalty] = state->trial_log_sp[penalty];
              }
              state->current = state->trial;
              state->trying = 0;
            } else {
              for (int penalty = 0; penalty < penalty_count; ++penalty) {
                state->working_step[penalty] = __dmul_rn(
                  0.5, state->working_step[penalty]);
              }
              ++state->step_halving_count;
            }
          }
          if (trial_index == max_step_halving - 1 &&
              state->trying != 0) {
            for (int penalty = 0; penalty < penalty_count; ++penalty) {
              state->working_step[penalty] = 0.0;
            }
          }
          if (trial_index == max_step_halving) state->trying = 0;
        }
        __syncthreads();
        if (state->optimizer_status != 0 || state->trying == 0) break;
      }
    }
    __syncthreads();
    if (state->optimizer_status != 0) break;

    if (threadIdx.x == 0) {
      if (iteration > 3) {
        double squared_gradient_norm = 0.0;
        for (int penalty = 0; penalty < penalty_count; ++penalty) {
          squared_gradient_norm = rounded_multiply_add(
            squared_gradient_norm, state->gradient[penalty],
            state->gradient[penalty]);
        }
        const double gradient_norm = sqrt(squared_gradient_norm);
        state->converged =
          state->score_reduction <= __dmul_rn(
            convergence_tolerance,
            __dadd_rn(1.0, state->minimum_score)) &&
          gradient_norm <= __dmul_rn(
            gradient_tolerance_factor,
            __dadd_rn(1.0, fabs(state->minimum_score)));
        if (state->tries == max_step_halving) state->converged = 1;
        if (state->converged != 0) {
          state->rms_gradient = __ddiv_rn(
            gradient_norm, sqrt(static_cast<double>(penalty_count)));
          if (state->tries == max_step_halving) state->step_failed = 1;
        }
      }

      for (int penalty = 0; penalty < penalty_count; ++penalty) {
        state->gradient[penalty] = state->current.gradient[penalty];
      }
      const bool eigen_converged = symmetric_jacobi_eigen(
        state->current.hessian, penalty_count, state->hessian_work,
        state->hessian_eigenvalues, state->hessian_eigenvectors);
      if (!eigen_converged) {
        state->optimizer_status = 3;
      } else {
        state->use_steepest_descent = 0;
        for (int penalty = 0; penalty < penalty_count; ++penalty) {
          if (state->hessian_eigenvalues[penalty] < 0.0) {
            state->use_steepest_descent = 1;
          }
        }
        if (state->use_steepest_descent == 0) {
          double spectral_step[kMultiPenaltyGcvMaximumPenaltyCount];
          double projected[kMultiPenaltyGcvMaximumPenaltyCount];
          for (int column = 0; column < penalty_count; ++column) {
            double value = 0.0;
            for (int row = 0; row < penalty_count; ++row) {
              value = rounded_multiply_add(
                value,
                state->hessian_eigenvectors[row + penalty_count * column],
                state->gradient[row]);
            }
            projected[column] = __ddiv_rn(
              value, state->hessian_eigenvalues[column]);
          }
          for (int row = 0; row < penalty_count; ++row) {
            double value = 0.0;
            for (int column = 0; column < penalty_count; ++column) {
              value = rounded_multiply_add(
                value,
                state->hessian_eigenvectors[row + penalty_count * column],
                projected[column]);
            }
            spectral_step[row] = -value;
          }
          bool direct_solved = false;
          if (direct_newton_replay) {
            direct_solved = solve_newton_system_direct(
              state->current.hessian, state->gradient, penalty_count,
              state->hessian_work, state->newton_step);
          }
          if (!direct_newton_replay || !direct_solved) {
            for (int row = 0; row < penalty_count; ++row) {
              state->newton_step[row] = spectral_step[row];
            }
          }
          double maximum_component = fabs(state->newton_step[0]);
          for (int penalty = 1; penalty < penalty_count; ++penalty) {
            maximum_component = fmax(
              maximum_component, fabs(state->newton_step[penalty]));
          }
          if (maximum_component > max_newton_step) {
            const double scale = __ddiv_rn(
              max_newton_step, maximum_component);
            for (int penalty = 0; penalty < penalty_count; ++penalty) {
              state->newton_step[penalty] = __dmul_rn(
                state->newton_step[penalty], scale);
            }
          }
        }
        double maximum_gradient = fabs(state->gradient[0]);
        for (int penalty = 1; penalty < penalty_count; ++penalty) {
          maximum_gradient = fmax(
            maximum_gradient, fabs(state->gradient[penalty]));
        }
        for (int penalty = 0; penalty < penalty_count; ++penalty) {
          state->steepest_step[penalty] = maximum_gradient == 0.0 ? 0.0 :
            __ddiv_rn(-state->gradient[penalty], maximum_gradient);
        }
      }
    }
    __syncthreads();
    if (state->optimizer_status != 0 || state->converged != 0) break;
  }
  __syncthreads();
  if (threadIdx.x == 0 && state->optimizer_status == 0 &&
      state->converged == 0) {
    state->optimizer_status = 4;
  }
  __syncthreads();
  if (state->optimizer_status != 0) return;

  if (threadIdx.x == 0) {
    state->fully_converged = state->step_failed == 0 ? 1 : 0;
    state->hessian_positive_definite =
      state->use_steepest_descent == 0 ? 1 : 0;
  }
  __syncthreads();

  for (int coordinate = 0; coordinate < penalty_count; ++coordinate) {
    if (threadIdx.x == 0) {
      state->tries = 0;
      state->trying = 1;
      const double direction = state->gradient[coordinate] < 0.0 ?
        1.0 : -1.0;
      for (int penalty = 0; penalty < penalty_count; ++penalty) {
        state->working_step[penalty] = 0.0;
      }
      state->working_step[coordinate] = __dmul_rn(
        direction, boundary_probe_step);
    }
    __syncthreads();
    for (int probe = 1; probe <= max_boundary_probes; ++probe) {
      if (threadIdx.x == 0 && state->trying != 0) {
        for (int penalty = 0; penalty < penalty_count; ++penalty) {
          state->trial_log_sp[penalty] = __dadd_rn(
            state->log_sp[penalty], state->working_step[penalty]);
        }
      }
      __syncthreads();
      if (state->trying == 0) break;
      evaluate_multi_penalty_target_device(
        magic_r, magic_pivot, penalty_roots, root_offsets, penalty_ranks,
        penalty_matrices, target_y0, target_squared_norm,
        state->trial_log_sp, workspace, &state->trial, n, q,
        penalty_count, false, &state->phase_timing, rank_tolerance,
        target_force_stable_svd, decomposition_trace,
        decomposition_trace_counters, decomposition_trace_capacity,
        target, -1,
        kDecompositionTraceBoundaryProbe | stability_trace_flag);
      __syncthreads();
      if (threadIdx.x == 0) {
        state->terminal_boundary_confirmation_pending = 0;
        ++state->objective_calls;
        ++state->boundary_probe_count;
        if (state->trial.solver_info != 0 ||
            !isfinite(state->trial.score)) {
          state->optimizer_status = 5;
          state->trying = 0;
        } else {
          state->terminal_boundary_primary_current_score =
            state->minimum_score;
          state->terminal_boundary_primary_trial_score = state->trial.score;
          const double score_delta = __dadd_rn(
            state->terminal_boundary_primary_trial_score,
            -state->terminal_boundary_primary_current_score);
          const double tie_tolerance = __dmul_rn(
            kTerminalBoundaryTieRelativeTolerance,
            __dadd_rn(1.0, fabs(state->minimum_score)));
          state->terminal_boundary_confirmation_pending =
            !target_force_stable_svd &&
            probe == max_boundary_probes &&
            state->trial.condition >= kTerminalBoundaryTieCondition &&
            (state->trial.condition >= kTerminalBoundaryOverrideCondition ||
             score_delta >= 0.0) &&
            fabs(score_delta) <= tie_tolerance ? 1 : 0;
        }
      }
      __syncthreads();
      if (state->optimizer_status != 0) return;
      if (state->terminal_boundary_confirmation_pending != 0) {
        evaluate_multi_penalty_target_device(
          magic_r, magic_pivot, penalty_roots, root_offsets, penalty_ranks,
          penalty_matrices, target_y0, target_squared_norm, state->log_sp,
          workspace, &state->trial, n, q, penalty_count, true,
          &state->terminal_boundary_confirmation_phase_timing,
          rank_tolerance, true, decomposition_trace,
          decomposition_trace_counters, decomposition_trace_capacity,
          target, -1,
          kDecompositionTraceTerminalConfirmation |
            stability_trace_flag);
        __syncthreads();
        if (threadIdx.x == 0) {
          if (state->trial.solver_info != 0 ||
              !isfinite(state->trial.score)) {
            state->optimizer_status = 5;
          } else {
            state->terminal_boundary_stable_current_score =
              state->trial.score;
            state->terminal_boundary_stable_current_direct_score =
              direct_qr_residual_score(
                magic_r, magic_pivot, target_y0, target_squared_norm,
                state->trial, n, q);
          }
        }
        __syncthreads();
        if (state->optimizer_status != 0) return;
        evaluate_multi_penalty_target_device(
          magic_r, magic_pivot, penalty_roots, root_offsets, penalty_ranks,
          penalty_matrices, target_y0, target_squared_norm,
          state->trial_log_sp, workspace, &state->trial, n, q,
          penalty_count, true,
          &state->terminal_boundary_confirmation_phase_timing,
          rank_tolerance, true, decomposition_trace,
          decomposition_trace_counters, decomposition_trace_capacity,
          target, -1,
          kDecompositionTraceTerminalConfirmation |
            stability_trace_flag);
        __syncthreads();
        if (threadIdx.x == 0 &&
            (state->trial.solver_info != 0 ||
             !isfinite(state->trial.score))) {
          state->optimizer_status = 5;
        } else if (threadIdx.x == 0) {
          state->terminal_boundary_stable_trial_direct_score =
            direct_qr_residual_score(
              magic_r, magic_pivot, target_y0, target_squared_norm,
              state->trial, n, q);
        }
        __syncthreads();
        if (state->optimizer_status != 0) return;
      }
      if (threadIdx.x == 0) {
        const double primary_score_delta = __dadd_rn(
          state->terminal_boundary_primary_trial_score,
          -state->terminal_boundary_primary_current_score);
        bool accepted = primary_score_delta < 0.0;
        if (state->terminal_boundary_confirmation_pending != 0) {
          const double stable_score_delta = __dadd_rn(
            state->trial.score,
            -state->terminal_boundary_stable_current_score);
          const double score_identity_disagreement = fmax(
            fabs(__dadd_rn(
              state->terminal_boundary_stable_current_direct_score,
              -state->terminal_boundary_stable_current_score)),
            fabs(__dadd_rn(
              state->terminal_boundary_stable_trial_direct_score,
              -state->trial.score)));
          const double identity_tolerance = __dmul_rn(
            __dmul_rn(512.0 * kLapackEpsilon, static_cast<double>(q)),
            __dadd_rn(
              1.0, fabs(state->terminal_boundary_stable_current_score)));
          const double stable_direct_score_delta = __dadd_rn(
            state->terminal_boundary_stable_trial_direct_score,
            -state->terminal_boundary_stable_current_direct_score);
          const double delta_identity_disagreement = fabs(__dadd_rn(
            stable_score_delta, -stable_direct_score_delta));
          const bool identity_verified_tie_descent =
            stable_score_delta < 0.0 &&
            score_identity_disagreement <= identity_tolerance;
          const bool strong_delta_descent =
            !identity_verified_tie_descent &&
            stable_score_delta < -identity_tolerance &&
            stable_direct_score_delta < stable_score_delta;
          const bool delta_identity_verified_descent =
            !identity_verified_tie_descent && !strong_delta_descent &&
            stable_direct_score_delta < 0.0 &&
            delta_identity_disagreement <= identity_tolerance;
          accepted = strong_delta_descent ||
            identity_verified_tie_descent ||
            delta_identity_verified_descent;
          ++state->terminal_boundary_confirmation_count;
          state->terminal_boundary_confirmation_accepted_count +=
            accepted ? 1 : 0;
          state->terminal_boundary_confirmation_rejected_count +=
            accepted ? 0 : 1;
          state
            ->terminal_boundary_confirmation_strong_delta_accepted_count +=
              strong_delta_descent ? 1 : 0;
          state
            ->terminal_boundary_confirmation_identity_tie_accepted_count +=
              identity_verified_tie_descent ? 1 : 0;
          state
            ->terminal_boundary_confirmation_delta_identity_accepted_count +=
              delta_identity_verified_descent ? 1 : 0;
          state->terminal_boundary_confirmation_max_identity_disagreement =
            fmax(
              state
                ->terminal_boundary_confirmation_max_identity_disagreement,
              score_identity_disagreement);
          state->terminal_boundary_confirmation_max_identity_ratio = fmax(
            state->terminal_boundary_confirmation_max_identity_ratio,
            __ddiv_rn(score_identity_disagreement, identity_tolerance));
          state->terminal_boundary_confirmation_max_delta_disagreement =
            fmax(
              state
                ->terminal_boundary_confirmation_max_delta_disagreement,
              delta_identity_disagreement);
          state->terminal_boundary_confirmation_max_delta_ratio = fmax(
            state->terminal_boundary_confirmation_max_delta_ratio,
            __ddiv_rn(delta_identity_disagreement, identity_tolerance));
        }
        if (state->optimizer_status == 0) {
          if (accepted) {
            for (int penalty = 0; penalty < penalty_count; ++penalty) {
              state->log_sp[penalty] = state->trial_log_sp[penalty];
            }
            state->minimum_score =
              state->terminal_boundary_primary_trial_score;
            ++state->tries;
            ++state->boundary_accepted_count;
          } else {
            state->trying = 0;
          }
        }
      }
      __syncthreads();
      if (state->optimizer_status != 0 || state->trying == 0) break;
    }
    if (threadIdx.x == 0 && state->optimizer_status == 0) {
      if (state->tries == 0) {
        state->boundary_status[coordinate] = kBoundaryFiniteInterior;
      } else if (state->tries == max_boundary_probes) {
        state->boundary_status[coordinate] =
          state->working_step[coordinate] > 0.0 ?
            kBoundaryPositive : kBoundaryNegative;
      } else {
        state->boundary_status[coordinate] = kBoundaryFiniteAfterProbe;
      }
    }
    __syncthreads();
    if (state->optimizer_status != 0) return;
  }

  if (stability_replay && threadIdx.x == 0) {
    double maximum_spread = 0.0;
    for (int penalty = 0; penalty < penalty_count; ++penalty) {
      maximum_spread = fmax(
        maximum_spread,
        fabs(__dadd_rn(
          state->log_sp[penalty],
          -primary_states[target].log_sp[penalty])));
    }
    state->stability_replay_log_sp_spread = maximum_spread;
    if (maximum_spread > kStabilityReplayLogSpSpread) {
      state->stability_replay_selected = 1;
      double extrapolation_fraction = 0.0;
      double maximum_extrapolation =
        kStabilityReplayMaximumExtrapolation;
      if (direct_newton_replay &&
          primary_states[target].step_halving_count > 0) {
        extrapolation_fraction =
          kDirectNewtonHalvingExtrapolationFraction;
        maximum_extrapolation = kDirectNewtonMaximumExtrapolation;
      } else if (
          requires_high_condition_stability_replay(primary_states[target])) {
        extrapolation_fraction = kStabilityReplayExtrapolationFraction;
      } else if (
          requires_dense_boundary_stability_replay(primary_states[target]) &&
          maximum_spread <= kStabilityReplayDenseBoundaryMaximumSpread) {
        extrapolation_fraction =
          kStabilityReplayDenseBoundaryExtrapolationFraction;
      }
      if (extrapolation_fraction > 0.0) {
        for (int penalty = 0; penalty < penalty_count; ++penalty) {
          const double path_delta = __dadd_rn(
            state->log_sp[penalty],
            -primary_states[target].log_sp[penalty]);
          const double extrapolation = copysign(
            fmin(
              maximum_extrapolation,
              fabs(__dmul_rn(
                extrapolation_fraction, path_delta))),
            path_delta);
          state->log_sp[penalty] = __dadd_rn(
            state->log_sp[penalty], extrapolation);
          state->stability_replay_max_extrapolation = fmax(
            state->stability_replay_max_extrapolation,
            fabs(extrapolation));
        }
        state->stability_replay_extrapolation_applied = 1;
      }
      if (primary_states[target].optimizer_iterations >=
          kStabilityReplayLongTrajectoryMinimumIterations) {
        const double inward_shift = fmin(
          kStabilityReplayMaximumInwardShift,
          __dmul_rn(2.0, maximum_spread));
        for (int penalty = 0; penalty < penalty_count; ++penalty) {
          if (state->boundary_status[penalty] == kBoundaryPositive) {
            state->log_sp[penalty] = __dadd_rn(
              state->log_sp[penalty], -inward_shift);
          } else if (state->boundary_status[penalty] == kBoundaryNegative) {
            state->log_sp[penalty] = __dadd_rn(
              state->log_sp[penalty], inward_shift);
          }
        }
      }
    }
  }
  __syncthreads();

  if (state->boundary_accepted_count == 0) {
    if (threadIdx.x == 0) {
      ++state->objective_calls;
      state->selected_evaluation_reuse_count = 1;
    }
    return;
  }

  evaluate_multi_penalty_target_device(
    magic_r, magic_pivot, penalty_roots, root_offsets, penalty_ranks,
    penalty_matrices, target_y0, target_squared_norm, state->log_sp,
    workspace, &state->trial, n, q, penalty_count, true,
    &state->phase_timing, rank_tolerance, target_force_stable_svd,
    decomposition_trace, decomposition_trace_counters,
    decomposition_trace_capacity, target, -1,
    kDecompositionTraceSelectedFit | stability_trace_flag);
  __syncthreads();
  if (threadIdx.x == 0) {
    ++state->objective_calls;
    if (state->trial.solver_info != 0 || !isfinite(state->trial.score)) {
      state->optimizer_status = 6;
    } else {
      state->current = state->trial;
      if (stability_replay && state->stability_replay_selected != 0 &&
          requires_dense_boundary_stability_replay(
            primary_states[target]) &&
          state->current.score >= primary_states[target].current.score) {
        state->stability_replay_selected = 0;
        state->stability_replay_dense_score_guard_rejected = 1;
      }
    }
  }
}

__global__ void merge_stability_replay_states_kernel(
    DeviceOptimizerState* primary_states,
    const DeviceOptimizerState* replay_states,
    int target_count) {
  const int target = blockIdx.x * blockDim.x + threadIdx.x;
  if (target >= target_count ||
      !is_stability_replay_candidate(primary_states[target])) {
    return;
  }
  DeviceOptimizerState& primary = primary_states[target];
  const DeviceOptimizerState& replay = replay_states[target];
  const int primary_confirmation_count =
    primary.terminal_boundary_confirmation_count;
  const int primary_confirmation_accepted_count =
    primary.terminal_boundary_confirmation_accepted_count;
  const int primary_confirmation_rejected_count =
    primary.terminal_boundary_confirmation_rejected_count;
  const int primary_confirmation_strong_delta_accepted_count =
    primary.terminal_boundary_confirmation_strong_delta_accepted_count;
  const int primary_confirmation_identity_tie_accepted_count =
    primary.terminal_boundary_confirmation_identity_tie_accepted_count;
  const int primary_confirmation_delta_identity_accepted_count =
    primary.terminal_boundary_confirmation_delta_identity_accepted_count;
  const double primary_confirmation_max_identity_disagreement =
    primary.terminal_boundary_confirmation_max_identity_disagreement;
  const double primary_confirmation_max_identity_ratio =
    primary.terminal_boundary_confirmation_max_identity_ratio;
  const double primary_confirmation_max_delta_disagreement =
    primary.terminal_boundary_confirmation_max_delta_disagreement;
  const double primary_confirmation_max_delta_ratio =
    primary.terminal_boundary_confirmation_max_delta_ratio;
  const DevicePhaseTiming primary_confirmation_timing =
    primary.terminal_boundary_confirmation_phase_timing;
  const int replay_screened =
    requires_full_stability_replay(primary) ? 0 : 1;
  const bool selected = replay.stability_replay_selected != 0 &&
    replay.optimizer_status == 0;
  const DevicePhaseTiming discarded = selected ?
    primary.phase_timing : replay.phase_timing;
  const double spread = replay.stability_replay_log_sp_spread;
  const int replay_error = replay.optimizer_status != 0 ? 1 : 0;
  if (selected) {
    primary = replay;
    primary.terminal_boundary_confirmation_count =
      primary_confirmation_count;
    primary.terminal_boundary_confirmation_accepted_count =
      primary_confirmation_accepted_count;
    primary.terminal_boundary_confirmation_rejected_count =
      primary_confirmation_rejected_count;
    primary.terminal_boundary_confirmation_strong_delta_accepted_count =
      primary_confirmation_strong_delta_accepted_count;
    primary.terminal_boundary_confirmation_identity_tie_accepted_count =
      primary_confirmation_identity_tie_accepted_count;
    primary.terminal_boundary_confirmation_delta_identity_accepted_count =
      primary_confirmation_delta_identity_accepted_count;
    primary.terminal_boundary_confirmation_max_identity_disagreement =
      primary_confirmation_max_identity_disagreement;
    primary.terminal_boundary_confirmation_max_identity_ratio =
      primary_confirmation_max_identity_ratio;
    primary.terminal_boundary_confirmation_max_delta_disagreement =
      primary_confirmation_max_delta_disagreement;
    primary.terminal_boundary_confirmation_max_delta_ratio =
      primary_confirmation_max_delta_ratio;
    primary.terminal_boundary_confirmation_phase_timing =
      primary_confirmation_timing;
  }
  primary.stability_replay_attempted = 1;
  primary.stability_replay_screened = replay_screened;
  primary.stability_replay_selected = selected ? 1 : 0;
  primary.stability_replay_error = replay_error;
  primary.stability_replay_long_trajectory_reason =
    replay.stability_replay_long_trajectory_reason;
  primary.stability_replay_dense_boundary_reason =
    replay.stability_replay_dense_boundary_reason;
  primary.stability_replay_high_condition_reason =
    replay.stability_replay_high_condition_reason;
  primary.stability_replay_ambiguous_step_reason =
    replay.stability_replay_ambiguous_step_reason;
  primary.stability_replay_rejected_boundary_reason =
    replay.stability_replay_rejected_boundary_reason;
  primary.stability_replay_direct_newton_reason =
    replay.stability_replay_direct_newton_reason;
  primary.stability_replay_dense_score_guard_rejected =
    replay.stability_replay_dense_score_guard_rejected;
  primary.stability_replay_log_sp_spread = spread;
  primary.discarded_phase_timing = discarded;
}

__global__ void gather_multi_penalty_coefficients_kernel(
    const DeviceOptimizerState* states,
    double* coefficients,
    int q,
    int target_count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = q * target_count;
  if (index >= count) return;
  const int target = index / q;
  const int row = index - target * q;
  coefficients[index] = states[target].optimizer_status == 0 ?
    states[target].current.coefficients[row] : CUDART_NAN;
}

__global__ void form_multi_penalty_residuals_kernel(
    const double* Y,
    const double* fitted,
    double* residuals,
    int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  residuals[index] = __dadd_rn(Y[index], -fitted[index]);
}

}  // namespace

class MultiPenaltyGcvCudaPrepared {
 public:
  MultiPenaltyGcvCudaPrepared() : cublas(stream.get()) {}
  ~MultiPenaltyGcvCudaPrepared() {
    if (device_id >= 0) cudaSetDevice(device_id);
  }

  int device_id = -1;
  int n = 0;
  int q = 0;
  int penalty_count = 0;
  int target_capacity = 0;
  int total_root_rank = 0;
  std::vector<double> initial_log_sp;
  Stream stream;
  CublasHandle cublas;
  Event completion_event;
  DeviceBuffer<unsigned char> d_arena;
  DeviceBuffer<double> d_X;
  DeviceBuffer<double> d_qr_packed;
  DeviceBuffer<double> d_tau;
  DeviceBuffer<double> d_r;
  DeviceBuffer<int> d_pivot;
  DeviceBuffer<double> d_roots;
  DeviceBuffer<int> d_root_offsets;
  DeviceBuffer<int> d_ranks;
  DeviceBuffer<double> d_matrices;
  DeviceBuffer<double> d_initial_log_sp;
  DeviceBuffer<double> d_Y;
  DeviceBuffer<double> d_qt_work;
  DeviceBuffer<double> d_y0;
  DeviceBuffer<double> d_squared_norm;
  DeviceBuffer<DeviceEvaluationWorkspace> d_workspaces;
  DeviceBuffer<DeviceOptimizerState> d_states;
  DeviceBuffer<DeviceOptimizerState> d_replay_states;
  DeviceBuffer<DeviceDecompositionTraceRecord> d_decomposition_trace;
  DeviceBuffer<unsigned int> d_decomposition_trace_counters;
  DeviceBuffer<double> d_coefficients;
  DeviceBuffer<double> d_fitted;
  DeviceBuffer<double> d_residuals;
  std::vector<DeviceOptimizerState> host_states;
  int decomposition_trace_capacity_per_target = 0;
  MultiPenaltyGcvCudaPreparedInfo info;
  std::uint64_t generation = 0;
  bool residual_slot_leased = false;
  std::mutex mutex;
};

class MultiPenaltyGcvCudaResidualBatch {
 public:
  ~MultiPenaltyGcvCudaResidualBatch() {
    if (!released && owner) {
      std::lock_guard<std::mutex> lock(owner->mutex);
      if (owner->generation == generation) {
        owner->residual_slot_leased = false;
      }
      released = true;
    }
  }

  std::shared_ptr<MultiPenaltyGcvCudaPrepared> owner;
  int n = 0;
  int q = 0;
  int target_count = 0;
  int device_id = -1;
  std::vector<std::string> target_keys;
  std::vector<int> optimizer_status;
  std::uint64_t generation = 0;
  bool released = false;
};

namespace {

MultiPenaltyGcvCudaOptimization materialize_optimizer_output(
    const DeviceOptimizerState* states,
    int n,
    int q,
    int penalty_count,
    int target_count,
    const std::vector<double>& initial_log_sp,
    MultiPenaltyGcvCudaDiagnostics diagnostics) {
  MultiPenaltyGcvCudaOptimization output;
  output.schema_version =
    "full-cuda-ci-multi-penalty-gcv-cuda-optimization-v1";
  output.rank_path =
    "cuda-dpstf2-guarded-qr-lapack-3.12-dgesdd";
  output.optimizer_path =
    "cuda-independent-target-newton-steepest-boundary-v1";
  output.n = n;
  output.coefficient_dim = q;
  output.penalty_count = penalty_count;
  output.target_count = target_count;
  output.initial_log_sp = initial_log_sp;
  const std::size_t targets = static_cast<std::size_t>(target_count);
  const std::size_t penalty_targets =
    static_cast<std::size_t>(penalty_count) * targets;
  output.selected_log_sp.resize(penalty_targets);
  output.rss.resize(targets);
  output.edf.resize(targets);
  output.score.resize(targets);
  output.condition.resize(targets);
  output.gradient.resize(penalty_targets);
  output.hessian.resize(
    static_cast<std::size_t>(penalty_count) * penalty_count * targets);
  output.coefficients.resize(static_cast<std::size_t>(q) * targets);
  output.rms_gradient.resize(targets);
  output.hessian_eigenvalues.resize(penalty_targets);
  output.aggregate_penalty_rank.resize(targets);
  output.numerical_rank.resize(targets);
  output.solver_info.resize(targets);
  output.optimizer_iterations.resize(targets);
  output.score_calls.resize(targets);
  output.objective_calls.resize(targets);
  output.step_halving_count.resize(targets);
  output.newton_trial_count.resize(targets);
  output.steepest_descent_trial_count.resize(targets);
  output.boundary_probe_count.resize(targets);
  output.boundary_accepted_count.resize(targets);
  output.boundary_status.resize(penalty_targets);
  output.fully_converged.resize(targets);
  output.hessian_positive_definite.resize(targets);
  output.step_failed.resize(targets);
  output.optimizer_status.resize(targets);
  for (int target = 0; target < target_count; ++target) {
    const DeviceOptimizerState& state = states[target];
    const DeviceTargetEvaluation& value = state.current;
    output.rss[target] = value.rss;
    output.edf[target] = value.edf;
    output.score[target] = value.score;
    output.condition[target] = value.condition;
    output.rms_gradient[target] = state.rms_gradient;
    output.aggregate_penalty_rank[target] = value.aggregate_penalty_rank;
    output.numerical_rank[target] = value.numerical_rank;
    output.solver_info[target] = value.solver_info;
    output.optimizer_iterations[target] = state.optimizer_iterations;
    output.score_calls[target] = state.score_calls;
    output.objective_calls[target] = state.objective_calls;
    output.step_halving_count[target] = state.step_halving_count;
    output.newton_trial_count[target] = state.newton_trial_count;
    output.steepest_descent_trial_count[target] =
      state.steepest_descent_trial_count;
    output.boundary_probe_count[target] = state.boundary_probe_count;
    output.boundary_accepted_count[target] = state.boundary_accepted_count;
    output.fully_converged[target] = state.fully_converged;
    output.hessian_positive_definite[target] =
      state.hessian_positive_definite;
    output.step_failed[target] = state.step_failed;
    output.optimizer_status[target] = state.optimizer_status;
    diagnostics.cuda_optimizer_objective_count += state.objective_calls;
    diagnostics.cuda_penalty_factor_augmentation_cycles +=
      state.phase_timing.penalty_factor_augmentation_cycles;
    diagnostics.cuda_qr_svd_cycles += state.phase_timing.qr_svd_cycles;
    diagnostics.cuda_qr_bidiagonal_reduction_cycles +=
      state.phase_timing.decomposition_stage_cycles[0];
    diagnostics.cuda_qr_factorization_cycles +=
      state.phase_timing.decomposition_stage_cycles[4];
    diagnostics.cuda_q_generation_cycles +=
      state.phase_timing.decomposition_stage_cycles[5];
    diagnostics.cuda_qr_guard_cycles +=
      state.phase_timing.decomposition_stage_cycles[6];
    diagnostics.cuda_stable_bidiagonal_reduction_cycles +=
      state.phase_timing.decomposition_stage_cycles[7];
    diagnostics.cuda_bidiagonal_svd_cycles +=
      state.phase_timing.decomposition_stage_cycles[1];
    diagnostics.cuda_svd_vector_postback_cycles +=
      state.phase_timing.decomposition_stage_cycles[2];
    diagnostics.cuda_left_vector_product_cycles +=
      state.phase_timing.decomposition_stage_cycles[3];
    diagnostics.cuda_score_construction_cycles +=
      state.phase_timing.score_construction_cycles;
    diagnostics.cuda_derivative_hessian_cycles +=
      state.phase_timing.derivative_hessian_cycles;
    diagnostics.cuda_complete_evaluation_count +=
      state.phase_timing.complete_evaluation_count;
    diagnostics.cuda_score_only_evaluation_count +=
      state.phase_timing.score_only_evaluation_count;
    diagnostics.cuda_guarded_qr_evaluation_count +=
      state.phase_timing.guarded_qr_evaluation_count;
    diagnostics.cuda_stable_svd_evaluation_count +=
      state.phase_timing.stable_svd_evaluation_count;
    diagnostics.cuda_selected_evaluation_reuse_count +=
      state.selected_evaluation_reuse_count;
    diagnostics.cuda_terminal_boundary_confirmation_count +=
      state.terminal_boundary_confirmation_count;
    diagnostics.cuda_terminal_boundary_confirmation_accepted_count +=
      state.terminal_boundary_confirmation_accepted_count;
    diagnostics.cuda_terminal_boundary_confirmation_rejected_count +=
      state.terminal_boundary_confirmation_rejected_count;
    diagnostics
      .cuda_terminal_boundary_confirmation_strong_delta_accepted_count +=
        state.terminal_boundary_confirmation_strong_delta_accepted_count;
    diagnostics
      .cuda_terminal_boundary_confirmation_identity_tie_accepted_count +=
        state.terminal_boundary_confirmation_identity_tie_accepted_count;
    diagnostics
      .cuda_terminal_boundary_confirmation_delta_identity_accepted_count +=
        state.terminal_boundary_confirmation_delta_identity_accepted_count;
    diagnostics
      .cuda_terminal_boundary_confirmation_complete_evaluation_count +=
        state.terminal_boundary_confirmation_phase_timing
          .complete_evaluation_count;
    diagnostics
      .cuda_terminal_boundary_confirmation_stable_svd_evaluation_count +=
        state.terminal_boundary_confirmation_phase_timing
          .stable_svd_evaluation_count;
    diagnostics.cuda_terminal_boundary_confirmation_cycles +=
      state.terminal_boundary_confirmation_phase_timing
        .penalty_factor_augmentation_cycles +
      state.terminal_boundary_confirmation_phase_timing.qr_svd_cycles +
      state.terminal_boundary_confirmation_phase_timing
        .score_construction_cycles +
      state.terminal_boundary_confirmation_phase_timing
        .derivative_hessian_cycles;
    diagnostics
      .cuda_terminal_boundary_confirmation_max_identity_disagreement =
        std::max(
          diagnostics
            .cuda_terminal_boundary_confirmation_max_identity_disagreement,
          state.terminal_boundary_confirmation_max_identity_disagreement);
    diagnostics.cuda_terminal_boundary_confirmation_max_identity_ratio =
      std::max(
        diagnostics.cuda_terminal_boundary_confirmation_max_identity_ratio,
        state.terminal_boundary_confirmation_max_identity_ratio);
    diagnostics
      .cuda_terminal_boundary_confirmation_max_delta_disagreement =
        std::max(
          diagnostics
            .cuda_terminal_boundary_confirmation_max_delta_disagreement,
          state.terminal_boundary_confirmation_max_delta_disagreement);
    diagnostics.cuda_terminal_boundary_confirmation_max_delta_ratio =
      std::max(
        diagnostics.cuda_terminal_boundary_confirmation_max_delta_ratio,
        state.terminal_boundary_confirmation_max_delta_ratio);
    if (state.stability_replay_attempted != 0) {
      ++diagnostics.cuda_stability_replay_target_count;
      const int discarded_replay_evaluations =
        state.discarded_phase_timing.complete_evaluation_count +
          state.discarded_phase_timing.score_only_evaluation_count;
      diagnostics.cuda_stability_replay_screened_count +=
        state.stability_replay_selected == 0 &&
          state.stability_replay_error == 0 &&
          discarded_replay_evaluations == 0 ? 1 : 0;
      diagnostics.cuda_stability_replay_selected_count +=
        state.stability_replay_selected != 0 ? 1 : 0;
      diagnostics.cuda_stability_replay_error_count +=
        state.stability_replay_error != 0 ? 1 : 0;
      diagnostics.cuda_stability_replay_long_trajectory_reason_count +=
        state.stability_replay_long_trajectory_reason != 0 ? 1 : 0;
      diagnostics.cuda_stability_replay_dense_boundary_reason_count +=
        state.stability_replay_dense_boundary_reason != 0 ? 1 : 0;
      diagnostics.cuda_stability_replay_high_condition_reason_count +=
        state.stability_replay_high_condition_reason != 0 ? 1 : 0;
      diagnostics.cuda_stability_replay_ambiguous_step_reason_count +=
        state.stability_replay_ambiguous_step_reason != 0 ? 1 : 0;
      diagnostics.cuda_stability_replay_rejected_boundary_reason_count +=
        state.stability_replay_rejected_boundary_reason != 0 ? 1 : 0;
      diagnostics.cuda_stability_replay_direct_newton_reason_count +=
        state.stability_replay_direct_newton_reason != 0 ? 1 : 0;
      diagnostics.cuda_stability_replay_dense_score_guard_rejected_count +=
        state.stability_replay_dense_score_guard_rejected != 0 ? 1 : 0;
      diagnostics.cuda_stability_replay_extrapolation_target_count +=
        state.stability_replay_extrapolation_applied != 0 ? 1 : 0;
      diagnostics.cuda_stability_replay_max_log_sp_spread = std::max(
        diagnostics.cuda_stability_replay_max_log_sp_spread,
        state.stability_replay_log_sp_spread);
      diagnostics.cuda_stability_replay_max_extrapolation = std::max(
        diagnostics.cuda_stability_replay_max_extrapolation,
        state.stability_replay_max_extrapolation);
      diagnostics.cuda_stability_replay_discarded_complete_evaluation_count +=
        state.discarded_phase_timing.complete_evaluation_count;
      diagnostics.cuda_stability_replay_discarded_score_only_evaluation_count +=
        state.discarded_phase_timing.score_only_evaluation_count;
      diagnostics.cuda_stability_replay_discarded_guarded_qr_evaluation_count +=
        state.discarded_phase_timing.guarded_qr_evaluation_count;
      diagnostics.cuda_stability_replay_discarded_stable_svd_evaluation_count +=
        state.discarded_phase_timing.stable_svd_evaluation_count;
      diagnostics.cuda_stability_replay_discarded_cycles +=
        state.discarded_phase_timing.penalty_factor_augmentation_cycles +
        state.discarded_phase_timing.qr_svd_cycles +
        state.discarded_phase_timing.score_construction_cycles +
        state.discarded_phase_timing.derivative_hessian_cycles;
    }
    diagnostics.cuda_hessian_eigensolver_count +=
      state.optimizer_iterations;
    if (state.optimizer_status == 0) {
      diagnostics.cuda_selected_fit_count += 1;
    } else {
      diagnostics.cuda_error_count += 1;
    }
    if (value.solver_info > 0) diagnostics.svd_nonconverged_count += 1;
    if (value.solver_info == -1) {
      diagnostics.aggregate_rank_failure_count += 1;
    }
    for (int penalty = 0; penalty < penalty_count; ++penalty) {
      const std::size_t penalty_target =
        static_cast<std::size_t>(penalty) +
        static_cast<std::size_t>(penalty_count) * target;
      output.selected_log_sp[penalty_target] = state.log_sp[penalty];
      output.gradient[penalty_target] = value.gradient[penalty];
      output.hessian_eigenvalues[penalty_target] =
        state.hessian_eigenvalues[penalty];
      output.boundary_status[penalty_target] =
        state.boundary_status[penalty];
      for (int other = 0; other < penalty_count; ++other) {
        output.hessian[penalty + penalty_count *
          (other + penalty_count * target)] =
            value.hessian[penalty + penalty_count * other];
      }
    }
    for (int row = 0; row < q; ++row) {
      output.coefficients[row + q * target] = value.coefficients[row];
    }
  }
  diagnostics.cuda_objective_target_count =
    diagnostics.cuda_optimizer_objective_count;
  output.diagnostics = std::move(diagnostics);
  return output;
}

void require_live_multi_penalty_residual(
    const std::shared_ptr<MultiPenaltyGcvCudaResidualBatch>& residual) {
  if (!residual || !residual->owner || residual->released ||
      residual->owner->generation != residual->generation ||
      !residual->owner->residual_slot_leased) {
    throw std::runtime_error(
      "multi-penalty CUDA residual token is stale or released");
  }
}

}  // namespace

MultiPenaltyGcvCudaEvaluation multi_penalty_gcv_evaluate_cuda(
    const double* X,
    const double* Y,
    const double* magic_qr_packed,
    const double* magic_tau,
    const double* magic_r,
    const int* magic_pivot_zero_based,
    const std::vector<std::vector<double>>& penalty_roots,
    const std::vector<std::vector<double>>& penalty_matrices,
    const std::vector<int>& penalty_ranks,
    const double* log_sp,
    int n,
    int coefficient_dim,
    int penalty_count,
    int target_count,
    double rank_tolerance) {
  const auto started = std::chrono::steady_clock::now();
  if (X == nullptr || Y == nullptr || magic_qr_packed == nullptr ||
      magic_tau == nullptr || magic_r == nullptr ||
      magic_pivot_zero_based == nullptr || log_sp == nullptr ||
      n <= coefficient_dim || coefficient_dim <= 0 ||
      coefficient_dim > kMultiPenaltyGcvMaximumCoefficientDim ||
      penalty_count <= 1 ||
      penalty_count > kMultiPenaltyGcvMaximumPenaltyCount ||
      target_count <= 1 || !std::isfinite(rank_tolerance) ||
      rank_tolerance <= 0.0 || rank_tolerance >= 1.0 ||
      static_cast<int>(penalty_roots.size()) != penalty_count ||
      static_cast<int>(penalty_matrices.size()) != penalty_count ||
      static_cast<int>(penalty_ranks.size()) != penalty_count) {
    throw std::runtime_error("multi-penalty CUDA inputs are invalid");
  }
  const int q = coefficient_dim;
  const std::size_t square = static_cast<std::size_t>(q) * q;
  int total_root_rank = 0;
  std::vector<int> root_offsets(static_cast<std::size_t>(penalty_count));
  for (int penalty = 0; penalty < penalty_count; ++penalty) {
    const int rank = penalty_ranks[static_cast<std::size_t>(penalty)];
    if (rank <= 0 || rank > q ||
        penalty_roots[static_cast<std::size_t>(penalty)].size() !=
          static_cast<std::size_t>(q) * rank ||
        penalty_matrices[static_cast<std::size_t>(penalty)].size() !=
          square) {
      throw std::runtime_error(
        "multi-penalty CUDA penalty geometry is invalid");
    }
    root_offsets[static_cast<std::size_t>(penalty)] = total_root_rank;
    total_root_rank += rank;
  }
  if (q + total_root_rank > kMaximumRows) {
    throw std::runtime_error(
      "multi-penalty CUDA augmented dimensions exceed the envelope");
  }
  std::vector<double> roots(
    static_cast<std::size_t>(q) * total_root_rank);
  std::vector<double> matrices(square * penalty_count);
  for (int penalty = 0; penalty < penalty_count; ++penalty) {
    const std::vector<double>& root =
      penalty_roots[static_cast<std::size_t>(penalty)];
    std::copy(root.begin(), root.end(), roots.begin() +
      static_cast<std::size_t>(q) *
        root_offsets[static_cast<std::size_t>(penalty)]);
    const std::vector<double>& matrix =
      penalty_matrices[static_cast<std::size_t>(penalty)];
    std::copy(matrix.begin(), matrix.end(),
              matrices.begin() + square * penalty);
  }

  MultiPenaltyGcvCudaEvaluation output;
  output.schema_version =
    "full-cuda-ci-multi-penalty-gcv-cuda-evaluation-v1";
  output.rank_path =
    "cuda-dpstf2-guarded-qr-lapack-3.12-dgesdd";
  output.n = n;
  output.coefficient_dim = q;
  output.penalty_count = penalty_count;
  output.target_count = target_count;
  MultiPenaltyGcvCudaDiagnostics& diagnostics = output.diagnostics;
  diagnostics.schema_version =
    "full-cuda-ci-multi-penalty-gcv-cuda-diagnostics-v1";
  diagnostics.execution_strategy =
    "one-setup-one-block-per-target-true-batch";
  diagnostics.prepared_setup_upload_count = 1;
  diagnostics.target_batch_upload_count = 1;
  diagnostics.cuda_qt_y_kernel_launch_count = 1;
  diagnostics.cuda_objective_kernel_launch_count = 1;
  diagnostics.cuda_objective_target_count = target_count;
  diagnostics.target_specific_log_sp = true;
  diagnostics.true_batched_kernel = target_count > 1;
  diagnostics.normal_equations_used = false;
  check_cuda(cudaGetDevice(&diagnostics.device_id),
             "query multi-penalty GCV device");
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, diagnostics.device_id),
             "query multi-penalty GCV device properties");
  diagnostics.gpu_name = properties.name;

  Stream stream;
  DeviceBuffer<double> d_Y;
  DeviceBuffer<double> d_qr_packed;
  DeviceBuffer<double> d_tau;
  DeviceBuffer<double> d_r;
  DeviceBuffer<int> d_pivot;
  DeviceBuffer<double> d_roots;
  DeviceBuffer<int> d_root_offsets;
  DeviceBuffer<int> d_ranks;
  DeviceBuffer<double> d_matrices;
  DeviceBuffer<double> d_log_sp;
  DeviceBuffer<double> d_qt_work;
  DeviceBuffer<double> d_y0;
  DeviceBuffer<double> d_squared_norm;
  DeviceBuffer<DeviceEvaluationWorkspace> d_workspaces;
  DeviceBuffer<DeviceTargetEvaluation> d_results;
  const std::size_t y_count = static_cast<std::size_t>(n) * target_count;
  d_Y.allocate(y_count, &diagnostics);
  d_qr_packed.allocate(static_cast<std::size_t>(n) * q, &diagnostics);
  d_tau.allocate(q, &diagnostics);
  d_r.allocate(square, &diagnostics);
  d_pivot.allocate(q, &diagnostics);
  d_roots.allocate(roots.size(), &diagnostics);
  d_root_offsets.allocate(penalty_count, &diagnostics);
  d_ranks.allocate(penalty_count, &diagnostics);
  d_matrices.allocate(matrices.size(), &diagnostics);
  d_log_sp.allocate(
    static_cast<std::size_t>(penalty_count) * target_count, &diagnostics);
  d_qt_work.allocate(y_count, &diagnostics);
  d_y0.allocate(static_cast<std::size_t>(q) * target_count, &diagnostics);
  d_squared_norm.allocate(target_count, &diagnostics);
  d_workspaces.allocate(target_count, &diagnostics);
  d_results.allocate(target_count, &diagnostics);
  upload_async(&d_Y, Y, y_count, stream.get(), &diagnostics);
  upload_async(&d_qr_packed, magic_qr_packed,
               static_cast<std::size_t>(n) * q, stream.get(), &diagnostics);
  upload_async(&d_tau, magic_tau, q, stream.get(), &diagnostics);
  upload_async(&d_r, magic_r, square, stream.get(), &diagnostics);
  upload_async(&d_pivot, magic_pivot_zero_based, q,
               stream.get(), &diagnostics);
  upload_async(&d_roots, roots.data(), roots.size(),
               stream.get(), &diagnostics);
  upload_async(&d_root_offsets, root_offsets.data(), penalty_count,
               stream.get(), &diagnostics);
  upload_async(&d_ranks, penalty_ranks.data(), penalty_count,
               stream.get(), &diagnostics);
  upload_async(&d_matrices, matrices.data(), matrices.size(),
               stream.get(), &diagnostics);
  upload_async(&d_log_sp, log_sp,
               static_cast<std::size_t>(penalty_count) * target_count,
               stream.get(), &diagnostics);

  const int target_blocks = (target_count + kBlockSize - 1) / kBlockSize;
  target_squared_norm_kernel<<<target_blocks, kBlockSize, 0, stream.get()>>>(
    d_Y.get(), d_squared_norm.get(), n, target_count);
  check_cuda(cudaGetLastError(),
             "launch multi-penalty target norm kernel");
  magic_qt_y_kernel<<<target_blocks, kBlockSize, 0, stream.get()>>>(
    d_qr_packed.get(), d_tau.get(), d_Y.get(), d_qt_work.get(),
    d_y0.get(), n, q, target_count);
  check_cuda(cudaGetLastError(),
             "launch multi-penalty QR projection kernel");
  evaluate_multi_penalty_targets_kernel<<<
    target_count, kBlockSize, 0, stream.get()>>>(
      d_r.get(), d_pivot.get(), d_roots.get(), d_root_offsets.get(),
      d_ranks.get(), d_matrices.get(), d_y0.get(), d_squared_norm.get(),
      d_log_sp.get(), d_workspaces.get(), d_results.get(), n, q,
      penalty_count, target_count, rank_tolerance);
  check_cuda(cudaGetLastError(),
             "launch multi-penalty objective batch kernel");
  std::vector<DeviceTargetEvaluation> host_results(
    static_cast<std::size_t>(target_count));
  check_cuda(cudaMemcpyAsync(
    host_results.data(), d_results.get(),
    host_results.size() * sizeof(DeviceTargetEvaluation),
    cudaMemcpyDeviceToHost, stream.get()),
    "download multi-penalty objective results");
  diagnostics.d2h_copy_count = 1;
  check_cuda(cudaStreamSynchronize(stream.get()),
             "synchronize multi-penalty objective batch");

  output.rss.resize(target_count);
  output.edf.resize(target_count);
  output.score.resize(target_count);
  output.condition.resize(target_count);
  output.aggregate_penalty_rank.resize(target_count);
  output.numerical_rank.resize(target_count);
  output.solver_info.resize(target_count);
  output.gradient.resize(
    static_cast<std::size_t>(penalty_count) * target_count);
  output.hessian.resize(
    static_cast<std::size_t>(penalty_count) * penalty_count * target_count);
  output.coefficients.resize(static_cast<std::size_t>(q) * target_count);
  for (int target = 0; target < target_count; ++target) {
    const DeviceTargetEvaluation& value =
      host_results[static_cast<std::size_t>(target)];
    output.rss[static_cast<std::size_t>(target)] = value.rss;
    output.edf[static_cast<std::size_t>(target)] = value.edf;
    output.score[static_cast<std::size_t>(target)] = value.score;
    output.condition[static_cast<std::size_t>(target)] = value.condition;
    output.aggregate_penalty_rank[static_cast<std::size_t>(target)] =
      value.aggregate_penalty_rank;
    output.numerical_rank[static_cast<std::size_t>(target)] =
      value.numerical_rank;
    output.solver_info[static_cast<std::size_t>(target)] = value.solver_info;
    if (value.solver_info != 0) diagnostics.cuda_error_count += 1;
    if (value.solver_info > 0) diagnostics.svd_nonconverged_count += 1;
    if (value.solver_info == -1) {
      diagnostics.aggregate_rank_failure_count += 1;
    }
    if (value.decomposition_route == kDecompositionGuardedQr) {
      diagnostics.cuda_guarded_qr_evaluation_count += 1;
    } else if (value.decomposition_route == kDecompositionStableSvd) {
      diagnostics.cuda_stable_svd_evaluation_count += 1;
    }
    for (int penalty = 0; penalty < penalty_count; ++penalty) {
      output.gradient[penalty + penalty_count * target] =
        value.gradient[penalty];
      for (int other = 0; other < penalty_count; ++other) {
        output.hessian[penalty + penalty_count *
          (other + penalty_count * target)] =
            value.hessian[penalty + penalty_count * other];
      }
    }
    for (int row = 0; row < q; ++row) {
      output.coefficients[row + q * target] = value.coefficients[row];
    }
  }
  diagnostics.total_host_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - started).count();
  return output;
}

namespace {

double grouped_prototype_median(std::vector<double> values) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2U;
  if ((values.size() & 1U) != 0U) return values[middle];
  return 0.5 * (values[middle - 1U] + values[middle]);
}

bool grouped_prototype_host_double_bits_equal(double left, double right) {
  std::uint64_t left_bits = 0;
  std::uint64_t right_bits = 0;
  std::memcpy(&left_bits, &left, sizeof(left_bits));
  std::memcpy(&right_bits, &right, sizeof(right_bits));
  return left_bits == right_bits;
}

int grouped_prototype_nonfinite_class(double value) {
  if (std::isfinite(value)) return 0;
  if (std::isnan(value)) return 1;
  return std::signbit(value) ? 2 : 3;
}

}  // namespace

MultiPenaltyGcvCudaGroupedPrototypeResult
multi_penalty_gcv_grouped_evaluate_prototype_cuda(
    const double* X,
    const double* Y,
    const double* magic_qr_packed,
    const double* magic_tau,
    const double* magic_r,
    const int* magic_pivot_zero_based,
    const std::vector<std::vector<double>>& penalty_roots,
    const std::vector<std::vector<double>>& penalty_matrices,
    const std::vector<int>& penalty_ranks,
    const double* log_sp,
    const int* force_stable_svd,
    int n,
    int coefficient_dim,
    int penalty_count,
    int target_count,
    double rank_tolerance,
    int grouped_warps_per_block,
    int timing_repetitions) {
  const auto started = std::chrono::steady_clock::now();
  if (X == nullptr || Y == nullptr || magic_qr_packed == nullptr ||
      magic_tau == nullptr || magic_r == nullptr ||
      magic_pivot_zero_based == nullptr || log_sp == nullptr ||
      force_stable_svd == nullptr || n <= coefficient_dim ||
      coefficient_dim <= FASTKPC_LAPACK_SMALL_SMLSIZ ||
      coefficient_dim > kMultiPenaltyGcvMaximumCoefficientDim ||
      penalty_count <= 1 ||
      penalty_count > kMultiPenaltyGcvMaximumPenaltyCount ||
      target_count <= 1 || target_count > 512 ||
      !std::isfinite(rank_tolerance) || rank_tolerance <= 0.0 ||
      rank_tolerance >= 1.0 ||
      static_cast<int>(penalty_roots.size()) != penalty_count ||
      static_cast<int>(penalty_matrices.size()) != penalty_count ||
      static_cast<int>(penalty_ranks.size()) != penalty_count ||
      (grouped_warps_per_block != 2 &&
       grouped_warps_per_block != 4 &&
       grouped_warps_per_block != 8) ||
      timing_repetitions <= 0 || timing_repetitions > 50) {
    throw std::runtime_error(
      "grouped multi-penalty CUDA prototype inputs are invalid");
  }
  const int q = coefficient_dim;
  const std::size_t square = static_cast<std::size_t>(q) * q;
  int total_root_rank = 0;
  std::vector<int> root_offsets(static_cast<std::size_t>(penalty_count));
  for (int penalty = 0; penalty < penalty_count; ++penalty) {
    const int rank = penalty_ranks[static_cast<std::size_t>(penalty)];
    if (rank <= 0 || rank > q ||
        penalty_roots[static_cast<std::size_t>(penalty)].size() !=
          static_cast<std::size_t>(q) * rank ||
        penalty_matrices[static_cast<std::size_t>(penalty)].size() !=
          square) {
      throw std::runtime_error(
        "grouped multi-penalty CUDA prototype geometry is invalid");
    }
    root_offsets[static_cast<std::size_t>(penalty)] = total_root_rank;
    total_root_rank += rank;
  }
  if (q + total_root_rank > kMaximumRows) {
    throw std::runtime_error(
      "grouped multi-penalty CUDA prototype exceeds the row envelope");
  }
  std::vector<double> roots(
    static_cast<std::size_t>(q) * total_root_rank);
  std::vector<double> matrices(square * penalty_count);
  for (int penalty = 0; penalty < penalty_count; ++penalty) {
    const std::vector<double>& root =
      penalty_roots[static_cast<std::size_t>(penalty)];
    std::copy(root.begin(), root.end(), roots.begin() +
      static_cast<std::size_t>(q) *
        root_offsets[static_cast<std::size_t>(penalty)]);
    const std::vector<double>& matrix =
      penalty_matrices[static_cast<std::size_t>(penalty)];
    std::copy(matrix.begin(), matrix.end(),
              matrices.begin() + square * penalty);
  }

  MultiPenaltyGcvCudaGroupedPrototypeResult output;
  MultiPenaltyGcvCudaEvaluation& evaluation = output.evaluation;
  evaluation.schema_version =
    "full-cuda-ci-multi-penalty-grouped-evaluation-prototype-v1";
  evaluation.rank_path =
    "development-only-grouped-guarded-qr-stable-svd-queue-v1";
  evaluation.n = n;
  evaluation.coefficient_dim = q;
  evaluation.penalty_count = penalty_count;
  evaluation.target_count = target_count;
  MultiPenaltyGcvCudaDiagnostics& evaluation_diagnostics =
    evaluation.diagnostics;
  evaluation_diagnostics.schema_version =
    "full-cuda-ci-multi-penalty-grouped-prototype-diagnostics-v1";
  evaluation_diagnostics.execution_strategy =
    "development-only-staged-grouped-guarded-qr";
  evaluation_diagnostics.prepared_setup_upload_count = 1;
  evaluation_diagnostics.target_batch_upload_count = 1;
  evaluation_diagnostics.cuda_qt_y_kernel_launch_count = 1;
  evaluation_diagnostics.cuda_objective_kernel_launch_count = 3;
  evaluation_diagnostics.cuda_objective_target_count = target_count;
  evaluation_diagnostics.target_specific_log_sp = true;
  evaluation_diagnostics.true_batched_kernel = target_count > 1;
  evaluation_diagnostics.normal_equations_used = false;
  check_cuda(cudaGetDevice(&evaluation_diagnostics.device_id),
             "query grouped prototype CUDA device");
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(
               &properties, evaluation_diagnostics.device_id),
             "query grouped prototype CUDA properties");
  evaluation_diagnostics.gpu_name = properties.name;

  Stream stream;
  DeviceBuffer<double> d_Y;
  DeviceBuffer<double> d_qr_packed;
  DeviceBuffer<double> d_tau;
  DeviceBuffer<double> d_r;
  DeviceBuffer<int> d_pivot;
  DeviceBuffer<double> d_roots;
  DeviceBuffer<int> d_root_offsets;
  DeviceBuffer<int> d_ranks;
  DeviceBuffer<double> d_matrices;
  DeviceBuffer<double> d_log_sp;
  DeviceBuffer<int> d_force_stable_svd;
  DeviceBuffer<double> d_qt_work;
  DeviceBuffer<double> d_y0;
  DeviceBuffer<double> d_squared_norm;
  DeviceBuffer<DeviceEvaluationWorkspace> d_baseline_workspaces;
  DeviceBuffer<DeviceEvaluationWorkspace> d_grouped_workspaces;
  DeviceBuffer<DeviceTargetEvaluation> d_baseline_results;
  DeviceBuffer<DeviceTargetEvaluation> d_grouped_results;
  DeviceBuffer<int> d_baseline_rows;
  DeviceBuffer<int> d_grouped_rows;
  DeviceBuffer<int> d_failure_targets;
  DeviceBuffer<unsigned int> d_failure_count;
  DeviceBuffer<DeviceGroupedPrototypeParity> d_parity;
  const std::size_t y_count = static_cast<std::size_t>(n) * target_count;
  d_Y.allocate(y_count, &evaluation_diagnostics);
  d_qr_packed.allocate(
    static_cast<std::size_t>(n) * q, &evaluation_diagnostics);
  d_tau.allocate(q, &evaluation_diagnostics);
  d_r.allocate(square, &evaluation_diagnostics);
  d_pivot.allocate(q, &evaluation_diagnostics);
  d_roots.allocate(roots.size(), &evaluation_diagnostics);
  d_root_offsets.allocate(penalty_count, &evaluation_diagnostics);
  d_ranks.allocate(penalty_count, &evaluation_diagnostics);
  d_matrices.allocate(matrices.size(), &evaluation_diagnostics);
  d_log_sp.allocate(
    static_cast<std::size_t>(penalty_count) * target_count,
    &evaluation_diagnostics);
  d_force_stable_svd.allocate(target_count, &evaluation_diagnostics);
  d_qt_work.allocate(y_count, &evaluation_diagnostics);
  d_y0.allocate(
    static_cast<std::size_t>(q) * target_count, &evaluation_diagnostics);
  d_squared_norm.allocate(target_count, &evaluation_diagnostics);
  d_baseline_workspaces.allocate(target_count, &evaluation_diagnostics);
  d_grouped_workspaces.allocate(target_count, &evaluation_diagnostics);
  d_baseline_results.allocate(target_count, &evaluation_diagnostics);
  d_grouped_results.allocate(target_count, &evaluation_diagnostics);
  d_baseline_rows.allocate(target_count, &evaluation_diagnostics);
  d_grouped_rows.allocate(target_count, &evaluation_diagnostics);
  d_failure_targets.allocate(target_count, &evaluation_diagnostics);
  d_failure_count.allocate(1, &evaluation_diagnostics);
  d_parity.allocate(1, &evaluation_diagnostics);
  upload_async(&d_Y, Y, y_count, stream.get(), &evaluation_diagnostics);
  upload_async(&d_qr_packed, magic_qr_packed,
               static_cast<std::size_t>(n) * q, stream.get(),
               &evaluation_diagnostics);
  upload_async(&d_tau, magic_tau, q, stream.get(),
               &evaluation_diagnostics);
  upload_async(&d_r, magic_r, square, stream.get(),
               &evaluation_diagnostics);
  upload_async(&d_pivot, magic_pivot_zero_based, q, stream.get(),
               &evaluation_diagnostics);
  upload_async(&d_roots, roots.data(), roots.size(), stream.get(),
               &evaluation_diagnostics);
  upload_async(&d_root_offsets, root_offsets.data(), penalty_count,
               stream.get(), &evaluation_diagnostics);
  upload_async(&d_ranks, penalty_ranks.data(), penalty_count,
               stream.get(), &evaluation_diagnostics);
  upload_async(&d_matrices, matrices.data(), matrices.size(), stream.get(),
               &evaluation_diagnostics);
  upload_async(&d_log_sp, log_sp,
               static_cast<std::size_t>(penalty_count) * target_count,
               stream.get(), &evaluation_diagnostics);
  upload_async(&d_force_stable_svd, force_stable_svd, target_count,
               stream.get(), &evaluation_diagnostics);

  const int target_blocks = (target_count + kBlockSize - 1) / kBlockSize;
  target_squared_norm_kernel<<<target_blocks, kBlockSize, 0, stream.get()>>>(
    d_Y.get(), d_squared_norm.get(), n, target_count);
  check_cuda(cudaGetLastError(),
             "launch grouped prototype target norm kernel");
  magic_qt_y_kernel<<<target_blocks, kBlockSize, 0, stream.get()>>>(
    d_qr_packed.get(), d_tau.get(), d_Y.get(), d_qt_work.get(),
    d_y0.get(), n, q, target_count);
  check_cuda(cudaGetLastError(),
             "launch grouped prototype QR projection kernel");

  const int grouped_threads = grouped_warps_per_block * 32;
  const int grouped_blocks =
    (target_count + grouped_warps_per_block - 1) /
      grouped_warps_per_block;
  auto prepare = [&](DeviceEvaluationWorkspace* workspaces,
                     DeviceTargetEvaluation* results,
                     int* rows) {
    prepare_grouped_prototype_targets_kernel<<<
      target_count, kBlockSize, 0, stream.get()>>>(
        d_r.get(), d_matrices.get(), d_log_sp.get(), workspaces, results,
        rows, q, penalty_count, target_count);
    check_cuda(cudaGetLastError(),
               "launch grouped prototype preparation kernel");
  };
  auto launch_baseline = [&]() {
    baseline_grouped_prototype_decomposition_kernel<<<
      target_count, kBlockSize, 0, stream.get()>>>(
        d_baseline_workspaces.get(), d_baseline_results.get(),
        d_baseline_rows.get(), d_force_stable_svd.get(), q, target_count);
    check_cuda(cudaGetLastError(),
               "launch grouped prototype baseline decomposition");
  };
  auto launch_grouped = [&]() {
    check_cuda(cudaMemsetAsync(
      d_failure_count.get(), 0, sizeof(unsigned int), stream.get()),
      "reset grouped prototype failure queue");
    grouped_prototype_guarded_qr_kernel<<<
      grouped_blocks, grouped_threads, 0, stream.get()>>>(
        d_grouped_workspaces.get(), d_grouped_results.get(),
        d_grouped_rows.get(), d_force_stable_svd.get(),
        d_failure_targets.get(), d_failure_count.get(), q, target_count);
    check_cuda(cudaGetLastError(),
               "launch grouped prototype guarded QR");
    grouped_prototype_stable_svd_kernel<<<
      target_count, kBlockSize, 0, stream.get()>>>(
        d_grouped_workspaces.get(), d_grouped_results.get(),
        d_grouped_rows.get(), d_failure_targets.get(),
        d_failure_count.get(), q);
    check_cuda(cudaGetLastError(),
               "launch grouped prototype stable SVD queue");
  };

  prepare(d_baseline_workspaces.get(), d_baseline_results.get(),
          d_baseline_rows.get());
  launch_baseline();
  prepare(d_grouped_workspaces.get(), d_grouped_results.get(),
          d_grouped_rows.get());
  launch_grouped();
  check_cuda(cudaStreamSynchronize(stream.get()),
             "warm grouped prototype decomposition paths");

  TimingEvent timing_start;
  TimingEvent timing_stop;
  output.diagnostics.baseline_qr_ms.reserve(timing_repetitions);
  output.diagnostics.grouped_qr_ms.reserve(timing_repetitions);
  for (int repetition = 0; repetition < timing_repetitions; ++repetition) {
    prepare(d_baseline_workspaces.get(), d_baseline_results.get(),
            d_baseline_rows.get());
    check_cuda(cudaEventRecord(timing_start.get(), stream.get()),
               "record grouped prototype baseline start");
    launch_baseline();
    check_cuda(cudaEventRecord(timing_stop.get(), stream.get()),
               "record grouped prototype baseline stop");
    check_cuda(cudaEventSynchronize(timing_stop.get()),
               "synchronize grouped prototype baseline timing");
    float elapsed_ms = 0.0F;
    check_cuda(cudaEventElapsedTime(
      &elapsed_ms, timing_start.get(), timing_stop.get()),
      "measure grouped prototype baseline QR");
    output.diagnostics.baseline_qr_ms.push_back(
      static_cast<double>(elapsed_ms));

    prepare(d_grouped_workspaces.get(), d_grouped_results.get(),
            d_grouped_rows.get());
    check_cuda(cudaMemsetAsync(
      d_failure_count.get(), 0, sizeof(unsigned int), stream.get()),
      "reset grouped prototype timed failure queue");
    check_cuda(cudaEventRecord(timing_start.get(), stream.get()),
               "record grouped prototype grouped start");
    grouped_prototype_guarded_qr_kernel<<<
      grouped_blocks, grouped_threads, 0, stream.get()>>>(
        d_grouped_workspaces.get(), d_grouped_results.get(),
        d_grouped_rows.get(), d_force_stable_svd.get(),
        d_failure_targets.get(), d_failure_count.get(), q, target_count);
    check_cuda(cudaGetLastError(),
               "launch timed grouped prototype guarded QR");
    grouped_prototype_stable_svd_kernel<<<
      target_count, kBlockSize, 0, stream.get()>>>(
        d_grouped_workspaces.get(), d_grouped_results.get(),
        d_grouped_rows.get(), d_failure_targets.get(),
        d_failure_count.get(), q);
    check_cuda(cudaGetLastError(),
               "launch timed grouped prototype stable SVD queue");
    check_cuda(cudaEventRecord(timing_stop.get(), stream.get()),
               "record grouped prototype grouped stop");
    check_cuda(cudaEventSynchronize(timing_stop.get()),
               "synchronize grouped prototype grouped timing");
    elapsed_ms = 0.0F;
    check_cuda(cudaEventElapsedTime(
      &elapsed_ms, timing_start.get(), timing_stop.get()),
      "measure grouped prototype grouped QR");
    output.diagnostics.grouped_qr_ms.push_back(
      static_cast<double>(elapsed_ms));
  }

  prepare(d_baseline_workspaces.get(), d_baseline_results.get(),
          d_baseline_rows.get());
  launch_baseline();
  prepare(d_grouped_workspaces.get(), d_grouped_results.get(),
          d_grouped_rows.get());
  launch_grouped();
  check_cuda(cudaMemsetAsync(
    d_parity.get(), 0, sizeof(DeviceGroupedPrototypeParity), stream.get()),
    "reset grouped prototype parity counters");
  compare_grouped_prototype_decompositions_kernel<<<
    target_count, kBlockSize, 0, stream.get()>>>(
      d_baseline_workspaces.get(), d_grouped_workspaces.get(),
      d_baseline_results.get(), d_grouped_results.get(),
      d_baseline_rows.get(), d_grouped_rows.get(), d_parity.get(), q,
      target_count);
  check_cuda(cudaGetLastError(),
             "launch grouped prototype decomposition comparison");
  complete_grouped_prototype_targets_kernel<<<
    target_count, kBlockSize, 0, stream.get()>>>(
      d_pivot.get(), d_roots.get(), d_root_offsets.get(), d_ranks.get(),
      d_y0.get(), d_squared_norm.get(), d_log_sp.get(),
      d_baseline_workspaces.get(), d_baseline_results.get(), n, q,
      penalty_count, target_count, rank_tolerance);
  check_cuda(cudaGetLastError(),
             "launch grouped prototype baseline completion");
  complete_grouped_prototype_targets_kernel<<<
    target_count, kBlockSize, 0, stream.get()>>>(
      d_pivot.get(), d_roots.get(), d_root_offsets.get(), d_ranks.get(),
      d_y0.get(), d_squared_norm.get(), d_log_sp.get(),
      d_grouped_workspaces.get(), d_grouped_results.get(), n, q,
      penalty_count, target_count, rank_tolerance);
  check_cuda(cudaGetLastError(),
             "launch grouped prototype grouped completion");

  std::vector<DeviceTargetEvaluation> baseline_results(
    static_cast<std::size_t>(target_count));
  std::vector<DeviceTargetEvaluation> grouped_results(
    static_cast<std::size_t>(target_count));
  DeviceGroupedPrototypeParity parity{};
  unsigned int failure_count = 0;
  check_cuda(cudaMemcpyAsync(
    baseline_results.data(), d_baseline_results.get(),
    baseline_results.size() * sizeof(DeviceTargetEvaluation),
    cudaMemcpyDeviceToHost, stream.get()),
    "download grouped prototype baseline results");
  check_cuda(cudaMemcpyAsync(
    grouped_results.data(), d_grouped_results.get(),
    grouped_results.size() * sizeof(DeviceTargetEvaluation),
    cudaMemcpyDeviceToHost, stream.get()),
    "download grouped prototype grouped results");
  check_cuda(cudaMemcpyAsync(
    &parity, d_parity.get(), sizeof(parity), cudaMemcpyDeviceToHost,
    stream.get()),
    "download grouped prototype parity counters");
  check_cuda(cudaMemcpyAsync(
    &failure_count, d_failure_count.get(), sizeof(failure_count),
    cudaMemcpyDeviceToHost, stream.get()),
    "download grouped prototype failure count");
  evaluation_diagnostics.d2h_copy_count = 4;
  check_cuda(cudaStreamSynchronize(stream.get()),
             "synchronize grouped prototype results");

  evaluation.rss.resize(target_count);
  evaluation.edf.resize(target_count);
  evaluation.score.resize(target_count);
  evaluation.condition.resize(target_count);
  evaluation.aggregate_penalty_rank.resize(target_count);
  evaluation.numerical_rank.resize(target_count);
  evaluation.solver_info.resize(target_count);
  evaluation.gradient.resize(
    static_cast<std::size_t>(penalty_count) * target_count);
  evaluation.hessian.resize(
    static_cast<std::size_t>(penalty_count) * penalty_count * target_count);
  evaluation.coefficients.resize(static_cast<std::size_t>(q) * target_count);

  MultiPenaltyGcvCudaGroupedPrototypeDiagnostics& diagnostics =
    output.diagnostics;
  diagnostics.schema_version =
    "full-cuda-ci-multi-penalty-grouped-prototype-diagnostics-v1";
  diagnostics.execution_strategy =
    "development-only-warp-grouped-qr-stable-svd-queue";
  diagnostics.device_id = evaluation_diagnostics.device_id;
  diagnostics.gpu_name = evaluation_diagnostics.gpu_name;
  diagnostics.grouped_warps_per_block = grouped_warps_per_block;
  diagnostics.timing_repetitions = timing_repetitions;
  diagnostics.target_count = target_count;
  diagnostics.grouped_failure_queue_count =
    static_cast<int>(failure_count);
  diagnostics.solver_route_mismatch_count =
    parity.solver_route_mismatch_count;
  diagnostics.solver_info_mismatch_count =
    parity.solver_info_mismatch_count;
  diagnostics.aggregate_rank_mismatch_count =
    parity.aggregate_rank_mismatch_count;
  diagnostics.augmented_rows_mismatch_count =
    parity.augmented_rows_mismatch_count;
  diagnostics.r_bitwise_mismatch_count = parity.r_bitwise_mismatch_count;
  diagnostics.explicit_q_bitwise_mismatch_count =
    parity.explicit_q_bitwise_mismatch_count;
  diagnostics.left_basis_bitwise_mismatch_count =
    parity.left_basis_bitwise_mismatch_count;
  diagnostics.singular_value_bitwise_mismatch_count =
    parity.singular_value_bitwise_mismatch_count;
  diagnostics.right_basis_bitwise_mismatch_count =
    parity.right_basis_bitwise_mismatch_count;
  diagnostics.qr_condition_estimate_bitwise_mismatch_count =
    parity.qr_condition_estimate_bitwise_mismatch_count;

  auto compare_scalar = [&](double baseline, double grouped,
                            std::uint64_t* mismatch_count) {
    if (!grouped_prototype_host_double_bits_equal(baseline, grouped)) {
      *mismatch_count += 1U;
    }
    if (grouped_prototype_nonfinite_class(baseline) !=
        grouped_prototype_nonfinite_class(grouped)) {
      diagnostics.nonfinite_status_mismatch_count += 1U;
    }
  };
  for (int target = 0; target < target_count; ++target) {
    const DeviceTargetEvaluation& baseline =
      baseline_results[static_cast<std::size_t>(target)];
    const DeviceTargetEvaluation& grouped =
      grouped_results[static_cast<std::size_t>(target)];
    if (baseline.decomposition_route == kDecompositionGuardedQr) {
      diagnostics.baseline_guarded_qr_count += 1;
    } else if (baseline.decomposition_route == kDecompositionStableSvd) {
      diagnostics.baseline_stable_svd_count += 1;
    }
    if (grouped.decomposition_route == kDecompositionGuardedQr) {
      diagnostics.grouped_guarded_qr_count += 1;
    } else if (grouped.decomposition_route == kDecompositionStableSvd) {
      diagnostics.grouped_stable_svd_count += 1;
    }
    if (baseline.numerical_rank != grouped.numerical_rank) {
      diagnostics.numerical_rank_mismatch_count += 1U;
    }
    if (baseline.solver_info != grouped.solver_info) {
      diagnostics.solver_info_mismatch_count += 1U;
    }
    compare_scalar(
      baseline.rss, grouped.rss, &diagnostics.rss_bitwise_mismatch_count);
    compare_scalar(
      baseline.edf, grouped.edf, &diagnostics.edf_bitwise_mismatch_count);
    compare_scalar(
      baseline.score, grouped.score,
      &diagnostics.score_bitwise_mismatch_count);
    compare_scalar(
      baseline.condition, grouped.condition,
      &diagnostics.condition_bitwise_mismatch_count);
    evaluation.rss[static_cast<std::size_t>(target)] = grouped.rss;
    evaluation.edf[static_cast<std::size_t>(target)] = grouped.edf;
    evaluation.score[static_cast<std::size_t>(target)] = grouped.score;
    evaluation.condition[static_cast<std::size_t>(target)] =
      grouped.condition;
    evaluation.aggregate_penalty_rank[static_cast<std::size_t>(target)] =
      grouped.aggregate_penalty_rank;
    evaluation.numerical_rank[static_cast<std::size_t>(target)] =
      grouped.numerical_rank;
    evaluation.solver_info[static_cast<std::size_t>(target)] =
      grouped.solver_info;
    if (grouped.solver_info != 0) evaluation_diagnostics.cuda_error_count += 1;
    if (grouped.solver_info > 0) {
      evaluation_diagnostics.svd_nonconverged_count += 1;
    }
    if (grouped.solver_info == -1) {
      evaluation_diagnostics.aggregate_rank_failure_count += 1;
    }
    for (int penalty = 0; penalty < penalty_count; ++penalty) {
      compare_scalar(
        baseline.gradient[penalty], grouped.gradient[penalty],
        &diagnostics.gradient_bitwise_mismatch_count);
      evaluation.gradient[penalty + penalty_count * target] =
        grouped.gradient[penalty];
      for (int other = 0; other < penalty_count; ++other) {
        const int index = penalty + penalty_count * other;
        compare_scalar(
          baseline.hessian[index], grouped.hessian[index],
          &diagnostics.hessian_bitwise_mismatch_count);
        evaluation.hessian[penalty + penalty_count *
          (other + penalty_count * target)] = grouped.hessian[index];
      }
    }
    for (int row = 0; row < q; ++row) {
      compare_scalar(
        baseline.coefficients[row], grouped.coefficients[row],
        &diagnostics.coefficient_bitwise_mismatch_count);
      evaluation.coefficients[row + q * target] = grouped.coefficients[row];
    }
  }
  evaluation_diagnostics.cuda_guarded_qr_evaluation_count =
    diagnostics.grouped_guarded_qr_count;
  evaluation_diagnostics.cuda_stable_svd_evaluation_count =
    diagnostics.grouped_stable_svd_count;
  diagnostics.baseline_qr_median_ms = grouped_prototype_median(
    diagnostics.baseline_qr_ms);
  diagnostics.grouped_qr_median_ms = grouped_prototype_median(
    diagnostics.grouped_qr_ms);
  diagnostics.qr_throughput_speedup =
    diagnostics.grouped_qr_median_ms > 0.0 ?
      diagnostics.baseline_qr_median_ms /
        diagnostics.grouped_qr_median_ms : 0.0;
  diagnostics.exact_parity =
    diagnostics.solver_route_mismatch_count == 0U &&
    diagnostics.solver_info_mismatch_count == 0U &&
    diagnostics.aggregate_rank_mismatch_count == 0U &&
    diagnostics.numerical_rank_mismatch_count == 0U &&
    diagnostics.augmented_rows_mismatch_count == 0U &&
    diagnostics.r_bitwise_mismatch_count == 0U &&
    diagnostics.explicit_q_bitwise_mismatch_count == 0U &&
    diagnostics.left_basis_bitwise_mismatch_count == 0U &&
    diagnostics.singular_value_bitwise_mismatch_count == 0U &&
    diagnostics.right_basis_bitwise_mismatch_count == 0U &&
    diagnostics.qr_condition_estimate_bitwise_mismatch_count == 0U &&
    diagnostics.condition_bitwise_mismatch_count == 0U &&
    diagnostics.rss_bitwise_mismatch_count == 0U &&
    diagnostics.edf_bitwise_mismatch_count == 0U &&
    diagnostics.score_bitwise_mismatch_count == 0U &&
    diagnostics.gradient_bitwise_mismatch_count == 0U &&
    diagnostics.hessian_bitwise_mismatch_count == 0U &&
    diagnostics.coefficient_bitwise_mismatch_count == 0U &&
    diagnostics.nonfinite_status_mismatch_count == 0U;
  evaluation_diagnostics.total_host_ms =
    std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
  return output;
}

MultiPenaltyGcvCudaOptimization multi_penalty_gcv_optimize_cuda(
    const double* X,
    const double* Y,
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
    int target_count,
    const MultiPenaltyGcvCudaOptimizerControl& control) {
  const auto started = std::chrono::steady_clock::now();
  if (X == nullptr || Y == nullptr || magic_qr_packed == nullptr ||
      magic_tau == nullptr || magic_r == nullptr ||
      magic_pivot_zero_based == nullptr || initial_log_sp == nullptr ||
      n <= coefficient_dim || coefficient_dim <= 0 ||
      coefficient_dim > kMultiPenaltyGcvMaximumCoefficientDim ||
      penalty_count <= 1 ||
      penalty_count > kMultiPenaltyGcvMaximumPenaltyCount ||
      target_count <= 1 ||
      static_cast<int>(penalty_roots.size()) != penalty_count ||
      static_cast<int>(penalty_matrices.size()) != penalty_count ||
      static_cast<int>(penalty_ranks.size()) != penalty_count ||
      !std::isfinite(control.convergence_tolerance) ||
      control.convergence_tolerance <= 0.0 ||
      control.convergence_tolerance >= 1.0 ||
      control.max_step_halving < 4 || control.max_iterations <= 3 ||
      !std::isfinite(control.max_newton_step) ||
      control.max_newton_step <= 0.0 ||
      !std::isfinite(control.boundary_probe_step) ||
      control.boundary_probe_step <= 0.0 ||
      control.max_boundary_probes <= 0 ||
      !std::isfinite(control.rank_tolerance) ||
      control.rank_tolerance <= 0.0 || control.rank_tolerance >= 1.0 ||
      control.decomposition_trace_capacity_per_target < 0 ||
      control.decomposition_trace_capacity_per_target > 4096) {
    throw std::runtime_error(
      "multi-penalty CUDA optimizer inputs are invalid");
  }
  const int q = coefficient_dim;
  const std::size_t square = static_cast<std::size_t>(q) * q;
  int total_root_rank = 0;
  std::vector<int> root_offsets(static_cast<std::size_t>(penalty_count));
  for (int penalty = 0; penalty < penalty_count; ++penalty) {
    const int rank = penalty_ranks[static_cast<std::size_t>(penalty)];
    if (rank <= 0 || rank > q ||
        penalty_roots[static_cast<std::size_t>(penalty)].size() !=
          static_cast<std::size_t>(q) * rank ||
        penalty_matrices[static_cast<std::size_t>(penalty)].size() !=
          square || !std::isfinite(initial_log_sp[penalty])) {
      throw std::runtime_error(
        "multi-penalty CUDA optimizer geometry is invalid");
    }
    root_offsets[static_cast<std::size_t>(penalty)] = total_root_rank;
    total_root_rank += rank;
  }
  if (q + total_root_rank > kMaximumRows) {
    throw std::runtime_error(
      "multi-penalty CUDA optimizer dimensions exceed the envelope");
  }
  std::vector<double> roots(
    static_cast<std::size_t>(q) * total_root_rank);
  std::vector<double> matrices(square * penalty_count);
  for (int penalty = 0; penalty < penalty_count; ++penalty) {
    const std::vector<double>& root =
      penalty_roots[static_cast<std::size_t>(penalty)];
    std::copy(root.begin(), root.end(), roots.begin() +
      static_cast<std::size_t>(q) *
        root_offsets[static_cast<std::size_t>(penalty)]);
    const std::vector<double>& matrix =
      penalty_matrices[static_cast<std::size_t>(penalty)];
    std::copy(matrix.begin(), matrix.end(),
              matrices.begin() + square * penalty);
  }

  MultiPenaltyGcvCudaOptimization output;
  output.schema_version =
    "full-cuda-ci-multi-penalty-gcv-cuda-optimization-v1";
  output.rank_path =
    "cuda-dpstf2-guarded-qr-lapack-3.12-dgesdd";
  output.optimizer_path =
    "cuda-independent-target-newton-steepest-boundary-v1";
  output.n = n;
  output.coefficient_dim = q;
  output.penalty_count = penalty_count;
  output.target_count = target_count;
  output.initial_log_sp.assign(
    initial_log_sp, initial_log_sp + penalty_count);
  MultiPenaltyGcvCudaDiagnostics& diagnostics = output.diagnostics;
  diagnostics.schema_version =
    "full-cuda-ci-multi-penalty-gcv-cuda-diagnostics-v1";
  diagnostics.execution_strategy =
    "one-setup-one-block-per-target-independent-optimizer";
  diagnostics.prepared_setup_upload_count = 1;
  diagnostics.target_batch_upload_count = 1;
  diagnostics.cuda_qt_y_kernel_launch_count = 1;
  diagnostics.cuda_optimizer_kernel_launch_count = 1;
  diagnostics.cuda_optimizer_target_count = target_count;
  diagnostics.independent_target_states = true;
  diagnostics.target_specific_log_sp = true;
  diagnostics.true_batched_kernel = target_count > 1;
  diagnostics.normal_equations_used = false;
  check_cuda(cudaGetDevice(&diagnostics.device_id),
             "query multi-penalty optimizer device");
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, diagnostics.device_id),
             "query multi-penalty optimizer device properties");
  diagnostics.gpu_name = properties.name;

  Stream stream;
  DeviceBuffer<double> d_Y;
  DeviceBuffer<double> d_qr_packed;
  DeviceBuffer<double> d_tau;
  DeviceBuffer<double> d_r;
  DeviceBuffer<int> d_pivot;
  DeviceBuffer<double> d_roots;
  DeviceBuffer<int> d_root_offsets;
  DeviceBuffer<int> d_ranks;
  DeviceBuffer<double> d_matrices;
  DeviceBuffer<double> d_initial_log_sp;
  DeviceBuffer<double> d_qt_work;
  DeviceBuffer<double> d_y0;
  DeviceBuffer<double> d_squared_norm;
  DeviceBuffer<DeviceEvaluationWorkspace> d_workspaces;
  DeviceBuffer<DeviceOptimizerState> d_states;
  DeviceBuffer<DeviceOptimizerState> d_replay_states;
  const std::size_t y_count = static_cast<std::size_t>(n) * target_count;
  d_Y.allocate(y_count, &diagnostics);
  d_qr_packed.allocate(static_cast<std::size_t>(n) * q, &diagnostics);
  d_tau.allocate(q, &diagnostics);
  d_r.allocate(square, &diagnostics);
  d_pivot.allocate(q, &diagnostics);
  d_roots.allocate(roots.size(), &diagnostics);
  d_root_offsets.allocate(penalty_count, &diagnostics);
  d_ranks.allocate(penalty_count, &diagnostics);
  d_matrices.allocate(matrices.size(), &diagnostics);
  d_initial_log_sp.allocate(penalty_count, &diagnostics);
  d_qt_work.allocate(y_count, &diagnostics);
  d_y0.allocate(static_cast<std::size_t>(q) * target_count, &diagnostics);
  d_squared_norm.allocate(target_count, &diagnostics);
  d_workspaces.allocate(target_count, &diagnostics);
  d_states.allocate(target_count, &diagnostics);
  d_replay_states.allocate(target_count, &diagnostics);
  upload_async(&d_Y, Y, y_count, stream.get(), &diagnostics);
  upload_async(&d_qr_packed, magic_qr_packed,
               static_cast<std::size_t>(n) * q, stream.get(), &diagnostics);
  upload_async(&d_tau, magic_tau, q, stream.get(), &diagnostics);
  upload_async(&d_r, magic_r, square, stream.get(), &diagnostics);
  upload_async(&d_pivot, magic_pivot_zero_based, q,
               stream.get(), &diagnostics);
  upload_async(&d_roots, roots.data(), roots.size(),
               stream.get(), &diagnostics);
  upload_async(&d_root_offsets, root_offsets.data(), penalty_count,
               stream.get(), &diagnostics);
  upload_async(&d_ranks, penalty_ranks.data(), penalty_count,
               stream.get(), &diagnostics);
  upload_async(&d_matrices, matrices.data(), matrices.size(),
               stream.get(), &diagnostics);
  upload_async(&d_initial_log_sp, initial_log_sp, penalty_count,
               stream.get(), &diagnostics);

  const int target_blocks = (target_count + kBlockSize - 1) / kBlockSize;
  target_squared_norm_kernel<<<target_blocks, kBlockSize, 0, stream.get()>>>(
    d_Y.get(), d_squared_norm.get(), n, target_count);
  check_cuda(cudaGetLastError(),
             "launch multi-penalty optimizer norm kernel");
  magic_qt_y_kernel<<<target_blocks, kBlockSize, 0, stream.get()>>>(
    d_qr_packed.get(), d_tau.get(), d_Y.get(), d_qt_work.get(),
    d_y0.get(), n, q, target_count);
  check_cuda(cudaGetLastError(),
             "launch multi-penalty optimizer QR projection kernel");
  const double gradient_tolerance_factor = std::pow(
    control.convergence_tolerance, 1.0 / 3.0);
  optimize_multi_penalty_targets_kernel<<<
    target_count, kBlockSize, 0, stream.get()>>>(
      d_r.get(), d_pivot.get(), d_roots.get(), d_root_offsets.get(),
      d_ranks.get(), d_matrices.get(), d_y0.get(), d_squared_norm.get(),
      d_initial_log_sp.get(), d_workspaces.get(), d_states.get(), nullptr,
      n, q, penalty_count, target_count, control.convergence_tolerance,
      gradient_tolerance_factor, control.max_step_halving,
      control.max_iterations, control.max_newton_step,
      control.boundary_probe_step, control.max_boundary_probes,
      control.rank_tolerance, false, false, nullptr, nullptr, 0U);
  check_cuda(cudaGetLastError(),
             "launch multi-penalty optimizer batch kernel");
  optimize_multi_penalty_targets_kernel<<<
    target_count, kBlockSize, 0, stream.get()>>>(
      d_r.get(), d_pivot.get(), d_roots.get(), d_root_offsets.get(),
      d_ranks.get(), d_matrices.get(), d_y0.get(), d_squared_norm.get(),
      d_initial_log_sp.get(), d_workspaces.get(), d_replay_states.get(),
      d_states.get(), n, q, penalty_count, target_count,
      control.convergence_tolerance, gradient_tolerance_factor,
      control.max_step_halving, control.max_iterations,
      control.max_newton_step, control.boundary_probe_step,
      control.max_boundary_probes, control.rank_tolerance, true, true,
      nullptr, nullptr, 0U);
  check_cuda(cudaGetLastError(),
             "launch multi-penalty stability replay kernel");
  diagnostics.cuda_stability_replay_kernel_launch_count = 1;
  merge_stability_replay_states_kernel<<<
    target_blocks, kBlockSize, 0, stream.get()>>>(
      d_states.get(), d_replay_states.get(), target_count);
  check_cuda(cudaGetLastError(),
             "launch multi-penalty stability replay merge kernel");
  diagnostics.cuda_stability_merge_kernel_launch_count = 1;
  std::vector<DeviceOptimizerState> host_states(
    static_cast<std::size_t>(target_count));
  check_cuda(cudaMemcpyAsync(
    host_states.data(), d_states.get(),
    host_states.size() * sizeof(DeviceOptimizerState),
    cudaMemcpyDeviceToHost, stream.get()),
    "download multi-penalty optimizer results");
  diagnostics.d2h_copy_count = 1;
  check_cuda(cudaStreamSynchronize(stream.get()),
             "synchronize multi-penalty optimizer batch");

  const std::size_t target_size = static_cast<std::size_t>(target_count);
  const std::size_t penalty_target_size =
    static_cast<std::size_t>(penalty_count) * target_count;
  output.selected_log_sp.resize(penalty_target_size);
  output.rss.resize(target_size);
  output.edf.resize(target_size);
  output.score.resize(target_size);
  output.condition.resize(target_size);
  output.gradient.resize(penalty_target_size);
  output.hessian.resize(
    static_cast<std::size_t>(penalty_count) * penalty_count * target_count);
  output.coefficients.resize(static_cast<std::size_t>(q) * target_count);
  output.rms_gradient.resize(target_size);
  output.hessian_eigenvalues.resize(penalty_target_size);
  output.aggregate_penalty_rank.resize(target_size);
  output.numerical_rank.resize(target_size);
  output.solver_info.resize(target_size);
  output.optimizer_iterations.resize(target_size);
  output.score_calls.resize(target_size);
  output.objective_calls.resize(target_size);
  output.step_halving_count.resize(target_size);
  output.newton_trial_count.resize(target_size);
  output.steepest_descent_trial_count.resize(target_size);
  output.boundary_probe_count.resize(target_size);
  output.boundary_accepted_count.resize(target_size);
  output.boundary_status.resize(penalty_target_size);
  output.fully_converged.resize(target_size);
  output.hessian_positive_definite.resize(target_size);
  output.step_failed.resize(target_size);
  output.optimizer_status.resize(target_size);
  for (int target = 0; target < target_count; ++target) {
    const DeviceOptimizerState& state =
      host_states[static_cast<std::size_t>(target)];
    const DeviceTargetEvaluation& value = state.current;
    output.rss[target] = value.rss;
    output.edf[target] = value.edf;
    output.score[target] = value.score;
    output.condition[target] = value.condition;
    output.rms_gradient[target] = state.rms_gradient;
    output.aggregate_penalty_rank[target] = value.aggregate_penalty_rank;
    output.numerical_rank[target] = value.numerical_rank;
    output.solver_info[target] = value.solver_info;
    output.optimizer_iterations[target] = state.optimizer_iterations;
    output.score_calls[target] = state.score_calls;
    output.objective_calls[target] = state.objective_calls;
    output.step_halving_count[target] = state.step_halving_count;
    output.newton_trial_count[target] = state.newton_trial_count;
    output.steepest_descent_trial_count[target] =
      state.steepest_descent_trial_count;
    output.boundary_probe_count[target] = state.boundary_probe_count;
    output.boundary_accepted_count[target] = state.boundary_accepted_count;
    output.fully_converged[target] = state.fully_converged;
    output.hessian_positive_definite[target] =
      state.hessian_positive_definite;
    output.step_failed[target] = state.step_failed;
    output.optimizer_status[target] = state.optimizer_status;
    diagnostics.cuda_optimizer_objective_count += state.objective_calls;
    diagnostics.cuda_penalty_factor_augmentation_cycles +=
      state.phase_timing.penalty_factor_augmentation_cycles;
    diagnostics.cuda_qr_svd_cycles += state.phase_timing.qr_svd_cycles;
    diagnostics.cuda_qr_bidiagonal_reduction_cycles +=
      state.phase_timing.decomposition_stage_cycles[0];
    diagnostics.cuda_qr_factorization_cycles +=
      state.phase_timing.decomposition_stage_cycles[4];
    diagnostics.cuda_q_generation_cycles +=
      state.phase_timing.decomposition_stage_cycles[5];
    diagnostics.cuda_qr_guard_cycles +=
      state.phase_timing.decomposition_stage_cycles[6];
    diagnostics.cuda_stable_bidiagonal_reduction_cycles +=
      state.phase_timing.decomposition_stage_cycles[7];
    diagnostics.cuda_bidiagonal_svd_cycles +=
      state.phase_timing.decomposition_stage_cycles[1];
    diagnostics.cuda_svd_vector_postback_cycles +=
      state.phase_timing.decomposition_stage_cycles[2];
    diagnostics.cuda_left_vector_product_cycles +=
      state.phase_timing.decomposition_stage_cycles[3];
    diagnostics.cuda_score_construction_cycles +=
      state.phase_timing.score_construction_cycles;
    diagnostics.cuda_derivative_hessian_cycles +=
      state.phase_timing.derivative_hessian_cycles;
    diagnostics.cuda_complete_evaluation_count +=
      state.phase_timing.complete_evaluation_count;
    diagnostics.cuda_score_only_evaluation_count +=
      state.phase_timing.score_only_evaluation_count;
    diagnostics.cuda_guarded_qr_evaluation_count +=
      state.phase_timing.guarded_qr_evaluation_count;
    diagnostics.cuda_stable_svd_evaluation_count +=
      state.phase_timing.stable_svd_evaluation_count;
    diagnostics.cuda_selected_evaluation_reuse_count +=
      state.selected_evaluation_reuse_count;
    diagnostics.cuda_terminal_boundary_confirmation_count +=
      state.terminal_boundary_confirmation_count;
    diagnostics.cuda_terminal_boundary_confirmation_accepted_count +=
      state.terminal_boundary_confirmation_accepted_count;
    diagnostics.cuda_terminal_boundary_confirmation_rejected_count +=
      state.terminal_boundary_confirmation_rejected_count;
    diagnostics
      .cuda_terminal_boundary_confirmation_strong_delta_accepted_count +=
        state.terminal_boundary_confirmation_strong_delta_accepted_count;
    diagnostics
      .cuda_terminal_boundary_confirmation_identity_tie_accepted_count +=
        state.terminal_boundary_confirmation_identity_tie_accepted_count;
    diagnostics
      .cuda_terminal_boundary_confirmation_delta_identity_accepted_count +=
        state.terminal_boundary_confirmation_delta_identity_accepted_count;
    diagnostics
      .cuda_terminal_boundary_confirmation_complete_evaluation_count +=
        state.terminal_boundary_confirmation_phase_timing
          .complete_evaluation_count;
    diagnostics
      .cuda_terminal_boundary_confirmation_stable_svd_evaluation_count +=
        state.terminal_boundary_confirmation_phase_timing
          .stable_svd_evaluation_count;
    diagnostics.cuda_terminal_boundary_confirmation_cycles +=
      state.terminal_boundary_confirmation_phase_timing
        .penalty_factor_augmentation_cycles +
      state.terminal_boundary_confirmation_phase_timing.qr_svd_cycles +
      state.terminal_boundary_confirmation_phase_timing
        .score_construction_cycles +
      state.terminal_boundary_confirmation_phase_timing
        .derivative_hessian_cycles;
    diagnostics
      .cuda_terminal_boundary_confirmation_max_identity_disagreement =
        std::max(
          diagnostics
            .cuda_terminal_boundary_confirmation_max_identity_disagreement,
          state.terminal_boundary_confirmation_max_identity_disagreement);
    diagnostics.cuda_terminal_boundary_confirmation_max_identity_ratio =
      std::max(
        diagnostics.cuda_terminal_boundary_confirmation_max_identity_ratio,
        state.terminal_boundary_confirmation_max_identity_ratio);
    diagnostics
      .cuda_terminal_boundary_confirmation_max_delta_disagreement =
        std::max(
          diagnostics
            .cuda_terminal_boundary_confirmation_max_delta_disagreement,
          state.terminal_boundary_confirmation_max_delta_disagreement);
    diagnostics.cuda_terminal_boundary_confirmation_max_delta_ratio =
      std::max(
        diagnostics.cuda_terminal_boundary_confirmation_max_delta_ratio,
        state.terminal_boundary_confirmation_max_delta_ratio);
    if (state.stability_replay_attempted != 0) {
      ++diagnostics.cuda_stability_replay_target_count;
      const int discarded_replay_evaluations =
        state.discarded_phase_timing.complete_evaluation_count +
          state.discarded_phase_timing.score_only_evaluation_count;
      diagnostics.cuda_stability_replay_screened_count +=
        state.stability_replay_selected == 0 &&
          state.stability_replay_error == 0 &&
          discarded_replay_evaluations == 0 ? 1 : 0;
      diagnostics.cuda_stability_replay_selected_count +=
        state.stability_replay_selected != 0 ? 1 : 0;
      diagnostics.cuda_stability_replay_error_count +=
        state.stability_replay_error != 0 ? 1 : 0;
      diagnostics.cuda_stability_replay_long_trajectory_reason_count +=
        state.stability_replay_long_trajectory_reason != 0 ? 1 : 0;
      diagnostics.cuda_stability_replay_dense_boundary_reason_count +=
        state.stability_replay_dense_boundary_reason != 0 ? 1 : 0;
      diagnostics.cuda_stability_replay_high_condition_reason_count +=
        state.stability_replay_high_condition_reason != 0 ? 1 : 0;
      diagnostics.cuda_stability_replay_ambiguous_step_reason_count +=
        state.stability_replay_ambiguous_step_reason != 0 ? 1 : 0;
      diagnostics.cuda_stability_replay_rejected_boundary_reason_count +=
        state.stability_replay_rejected_boundary_reason != 0 ? 1 : 0;
      diagnostics.cuda_stability_replay_direct_newton_reason_count +=
        state.stability_replay_direct_newton_reason != 0 ? 1 : 0;
      diagnostics.cuda_stability_replay_dense_score_guard_rejected_count +=
        state.stability_replay_dense_score_guard_rejected != 0 ? 1 : 0;
      diagnostics.cuda_stability_replay_extrapolation_target_count +=
        state.stability_replay_extrapolation_applied != 0 ? 1 : 0;
      diagnostics.cuda_stability_replay_max_log_sp_spread = std::max(
        diagnostics.cuda_stability_replay_max_log_sp_spread,
        state.stability_replay_log_sp_spread);
      diagnostics.cuda_stability_replay_max_extrapolation = std::max(
        diagnostics.cuda_stability_replay_max_extrapolation,
        state.stability_replay_max_extrapolation);
      diagnostics.cuda_stability_replay_discarded_complete_evaluation_count +=
        state.discarded_phase_timing.complete_evaluation_count;
      diagnostics.cuda_stability_replay_discarded_score_only_evaluation_count +=
        state.discarded_phase_timing.score_only_evaluation_count;
      diagnostics.cuda_stability_replay_discarded_guarded_qr_evaluation_count +=
        state.discarded_phase_timing.guarded_qr_evaluation_count;
      diagnostics.cuda_stability_replay_discarded_stable_svd_evaluation_count +=
        state.discarded_phase_timing.stable_svd_evaluation_count;
      diagnostics.cuda_stability_replay_discarded_cycles +=
        state.discarded_phase_timing.penalty_factor_augmentation_cycles +
        state.discarded_phase_timing.qr_svd_cycles +
        state.discarded_phase_timing.score_construction_cycles +
        state.discarded_phase_timing.derivative_hessian_cycles;
    }
    diagnostics.cuda_hessian_eigensolver_count +=
      state.optimizer_iterations;
    if (state.optimizer_status == 0) {
      diagnostics.cuda_selected_fit_count += 1;
    } else {
      diagnostics.cuda_error_count += 1;
    }
    if (value.solver_info > 0) diagnostics.svd_nonconverged_count += 1;
    if (value.solver_info == -1) {
      diagnostics.aggregate_rank_failure_count += 1;
    }
    for (int penalty = 0; penalty < penalty_count; ++penalty) {
      const std::size_t penalty_target =
        static_cast<std::size_t>(penalty) +
        static_cast<std::size_t>(penalty_count) * target;
      output.selected_log_sp[penalty_target] = state.log_sp[penalty];
      output.gradient[penalty_target] = value.gradient[penalty];
      output.hessian_eigenvalues[penalty_target] =
        state.hessian_eigenvalues[penalty];
      output.boundary_status[penalty_target] =
        state.boundary_status[penalty];
      for (int other = 0; other < penalty_count; ++other) {
        output.hessian[penalty + penalty_count *
          (other + penalty_count * target)] =
            value.hessian[penalty + penalty_count * other];
      }
    }
    for (int row = 0; row < q; ++row) {
      output.coefficients[row + q * target] = value.coefficients[row];
    }
  }
  diagnostics.cuda_objective_target_count =
    diagnostics.cuda_optimizer_objective_count;
  diagnostics.total_host_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - started).count();
  return output;
}

std::shared_ptr<MultiPenaltyGcvCudaPrepared>
create_multi_penalty_gcv_cuda_prepared(
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
    int device_id) {
  if (X == nullptr || magic_qr_packed == nullptr || magic_tau == nullptr ||
      magic_r == nullptr || magic_pivot_zero_based == nullptr ||
      initial_log_sp == nullptr || n <= coefficient_dim ||
      coefficient_dim <= 0 ||
      coefficient_dim > kMultiPenaltyGcvMaximumCoefficientDim ||
      penalty_count <= 1 ||
      penalty_count > kMultiPenaltyGcvMaximumPenaltyCount ||
      target_capacity <= 1 || device_id < 0 ||
      static_cast<int>(penalty_roots.size()) != penalty_count ||
      static_cast<int>(penalty_matrices.size()) != penalty_count ||
      static_cast<int>(penalty_ranks.size()) != penalty_count) {
    throw std::runtime_error(
      "persistent multi-penalty CUDA setup inputs are invalid");
  }
  const int q = coefficient_dim;
  const std::size_t square = static_cast<std::size_t>(q) * q;
  int total_root_rank = 0;
  std::vector<int> root_offsets(static_cast<std::size_t>(penalty_count));
  for (int penalty = 0; penalty < penalty_count; ++penalty) {
    const int rank = penalty_ranks[static_cast<std::size_t>(penalty)];
    if (rank <= 0 || rank > q ||
        penalty_roots[static_cast<std::size_t>(penalty)].size() !=
          static_cast<std::size_t>(q) * rank ||
        penalty_matrices[static_cast<std::size_t>(penalty)].size() !=
          square || !std::isfinite(initial_log_sp[penalty])) {
      throw std::runtime_error(
        "persistent multi-penalty CUDA setup geometry is invalid");
    }
    root_offsets[static_cast<std::size_t>(penalty)] = total_root_rank;
    total_root_rank += rank;
  }
  if (q + total_root_rank > kMaximumRows) {
    throw std::runtime_error(
      "persistent multi-penalty CUDA setup exceeds the envelope");
  }
  std::vector<double> roots(
    static_cast<std::size_t>(q) * total_root_rank);
  std::vector<double> matrices(square * penalty_count);
  for (int penalty = 0; penalty < penalty_count; ++penalty) {
    const std::vector<double>& root =
      penalty_roots[static_cast<std::size_t>(penalty)];
    std::copy(root.begin(), root.end(), roots.begin() +
      static_cast<std::size_t>(q) *
        root_offsets[static_cast<std::size_t>(penalty)]);
    const std::vector<double>& matrix =
      penalty_matrices[static_cast<std::size_t>(penalty)];
    std::copy(matrix.begin(), matrix.end(),
              matrices.begin() + square * penalty);
  }

  int previous_device = -1;
  check_cuda(cudaGetDevice(&previous_device),
             "query device before persistent multi-penalty setup");
  check_cuda(cudaSetDevice(device_id),
             "select device for persistent multi-penalty setup");
  try {
    std::shared_ptr<MultiPenaltyGcvCudaPrepared> prepared =
      std::make_shared<MultiPenaltyGcvCudaPrepared>();
    prepared->device_id = device_id;
    prepared->n = n;
    prepared->q = q;
    prepared->penalty_count = penalty_count;
    prepared->target_capacity = target_capacity;
    prepared->total_root_rank = total_root_rank;
    prepared->initial_log_sp.assign(
      initial_log_sp, initial_log_sp + penalty_count);
    prepared->host_states.resize(static_cast<std::size_t>(target_capacity));
    MultiPenaltyGcvCudaDiagnostics allocation_diagnostics;
    const std::size_t y_capacity =
      static_cast<std::size_t>(n) * target_capacity;
    const std::size_t q_targets =
      static_cast<std::size_t>(q) * target_capacity;
    std::size_t arena_bytes = 0;
    arena_bytes = arena_advance<double>(arena_bytes,
      static_cast<std::size_t>(n) * q);
    arena_bytes = arena_advance<double>(arena_bytes,
      static_cast<std::size_t>(n) * q);
    arena_bytes = arena_advance<double>(arena_bytes, q);
    arena_bytes = arena_advance<double>(arena_bytes, square);
    arena_bytes = arena_advance<int>(arena_bytes, q);
    arena_bytes = arena_advance<double>(arena_bytes, roots.size());
    arena_bytes = arena_advance<int>(arena_bytes, penalty_count);
    arena_bytes = arena_advance<int>(arena_bytes, penalty_count);
    arena_bytes = arena_advance<double>(arena_bytes, matrices.size());
    arena_bytes = arena_advance<double>(arena_bytes, penalty_count);
    arena_bytes = arena_advance<double>(arena_bytes, y_capacity);
    arena_bytes = arena_advance<double>(arena_bytes, y_capacity);
    arena_bytes = arena_advance<double>(arena_bytes, q_targets);
    arena_bytes = arena_advance<double>(arena_bytes, target_capacity);
    arena_bytes = arena_advance<DeviceEvaluationWorkspace>(
      arena_bytes, target_capacity);
    arena_bytes = arena_advance<DeviceOptimizerState>(
      arena_bytes, target_capacity);
    arena_bytes = arena_advance<DeviceOptimizerState>(
      arena_bytes, target_capacity);
    arena_bytes = arena_advance<double>(arena_bytes, q_targets);
    arena_bytes = arena_advance<double>(arena_bytes, y_capacity);
    arena_bytes = arena_advance<double>(arena_bytes, y_capacity);
    prepared->d_arena.allocate(arena_bytes, &allocation_diagnostics);
    unsigned char* arena = prepared->d_arena.get();
    std::size_t arena_offset = 0;
    arena_offset = prepared->d_X.bind_from_arena(
      arena, arena_offset, static_cast<std::size_t>(n) * q);
    arena_offset = prepared->d_qr_packed.bind_from_arena(
      arena, arena_offset, static_cast<std::size_t>(n) * q);
    arena_offset = prepared->d_tau.bind_from_arena(
      arena, arena_offset, q);
    arena_offset = prepared->d_r.bind_from_arena(
      arena, arena_offset, square);
    arena_offset = prepared->d_pivot.bind_from_arena(
      arena, arena_offset, q);
    arena_offset = prepared->d_roots.bind_from_arena(
      arena, arena_offset, roots.size());
    arena_offset = prepared->d_root_offsets.bind_from_arena(
      arena, arena_offset, penalty_count);
    arena_offset = prepared->d_ranks.bind_from_arena(
      arena, arena_offset, penalty_count);
    arena_offset = prepared->d_matrices.bind_from_arena(
      arena, arena_offset, matrices.size());
    arena_offset = prepared->d_initial_log_sp.bind_from_arena(
      arena, arena_offset, penalty_count);
    arena_offset = prepared->d_Y.bind_from_arena(
      arena, arena_offset, y_capacity);
    arena_offset = prepared->d_qt_work.bind_from_arena(
      arena, arena_offset, y_capacity);
    arena_offset = prepared->d_y0.bind_from_arena(
      arena, arena_offset, q_targets);
    arena_offset = prepared->d_squared_norm.bind_from_arena(
      arena, arena_offset, target_capacity);
    arena_offset = prepared->d_workspaces.bind_from_arena(
      arena, arena_offset, target_capacity);
    arena_offset = prepared->d_states.bind_from_arena(
      arena, arena_offset, target_capacity);
    arena_offset = prepared->d_replay_states.bind_from_arena(
      arena, arena_offset, target_capacity);
    arena_offset = prepared->d_coefficients.bind_from_arena(
      arena, arena_offset, q_targets);
    arena_offset = prepared->d_fitted.bind_from_arena(
      arena, arena_offset, y_capacity);
    arena_offset = prepared->d_residuals.bind_from_arena(
      arena, arena_offset, y_capacity);
    if (arena_offset != arena_bytes) {
      throw std::runtime_error(
        "persistent multi-penalty CUDA arena layout is inconsistent");
    }
    MultiPenaltyGcvCudaDiagnostics upload_diagnostics;
    upload_async(&prepared->d_X, X, static_cast<std::size_t>(n) * q,
                 prepared->stream.get(), &upload_diagnostics);
    upload_async(&prepared->d_qr_packed, magic_qr_packed,
                 static_cast<std::size_t>(n) * q,
                 prepared->stream.get(), &upload_diagnostics);
    upload_async(&prepared->d_tau, magic_tau, q,
                 prepared->stream.get(), &upload_diagnostics);
    upload_async(&prepared->d_r, magic_r, square,
                 prepared->stream.get(), &upload_diagnostics);
    upload_async(&prepared->d_pivot, magic_pivot_zero_based, q,
                 prepared->stream.get(), &upload_diagnostics);
    upload_async(&prepared->d_roots, roots.data(), roots.size(),
                 prepared->stream.get(), &upload_diagnostics);
    upload_async(&prepared->d_root_offsets, root_offsets.data(),
                 penalty_count, prepared->stream.get(),
                 &upload_diagnostics);
    upload_async(&prepared->d_ranks, penalty_ranks.data(), penalty_count,
                 prepared->stream.get(), &upload_diagnostics);
    upload_async(&prepared->d_matrices, matrices.data(), matrices.size(),
                 prepared->stream.get(), &upload_diagnostics);
    upload_async(&prepared->d_initial_log_sp, initial_log_sp,
                 penalty_count, prepared->stream.get(),
                 &upload_diagnostics);
    check_cuda(cudaStreamSynchronize(prepared->stream.get()),
               "synchronize persistent multi-penalty setup");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device_id),
               "query persistent multi-penalty device properties");
    prepared->info.device_id = device_id;
    prepared->info.gpu_name = properties.name;
    prepared->info.n = n;
    prepared->info.coefficient_dim = q;
    prepared->info.penalty_count = penalty_count;
    prepared->info.target_capacity = target_capacity;
    prepared->info.setup_upload_count = 1;
    prepared->info.device_allocation_count =
      allocation_diagnostics.device_allocation_count;
    prepared->info.setup_h2d_bytes =
      (static_cast<std::size_t>(2) * n * q + square + roots.size() +
       matrices.size() + static_cast<std::size_t>(2) * penalty_count + q) *
        sizeof(double) +
      (static_cast<std::size_t>(q) +
       static_cast<std::size_t>(2) * penalty_count) * sizeof(int);
    check_cuda(cudaSetDevice(previous_device),
               "restore device after persistent multi-penalty setup");
    return prepared;
  } catch (...) {
    cudaSetDevice(previous_device);
    throw;
  }
}

MultiPenaltyGcvCudaPreparedInfo multi_penalty_gcv_cuda_prepared_info(
    const std::shared_ptr<MultiPenaltyGcvCudaPrepared>& prepared) {
  if (!prepared) {
    throw std::runtime_error(
      "persistent multi-penalty CUDA setup has been freed");
  }
  std::lock_guard<std::mutex> lock(prepared->mutex);
  MultiPenaltyGcvCudaPreparedInfo info = prepared->info;
  info.residual_slot_leased = prepared->residual_slot_leased;
  info.generation = prepared->generation;
  return info;
}

MultiPenaltyGcvCudaBatchResult multi_penalty_gcv_cuda_optimize_batch(
    const std::shared_ptr<MultiPenaltyGcvCudaPrepared>& prepared,
    const double* Y,
    int n,
    int target_count,
    const std::vector<std::string>& target_keys,
    const MultiPenaltyGcvCudaOptimizerControl& control) {
  const auto started = std::chrono::steady_clock::now();
  if (!prepared || Y == nullptr) {
    throw std::runtime_error(
      "persistent multi-penalty CUDA batch inputs are invalid");
  }
  std::lock_guard<std::mutex> lock(prepared->mutex);
  if (prepared->residual_slot_leased) {
    throw std::runtime_error("ERR_MULTI_PENALTY_OUTPUT_SLOT_BUSY");
  }
  if (n != prepared->n || target_count <= 1 ||
      target_count > prepared->target_capacity ||
      target_keys.size() != static_cast<std::size_t>(target_count) ||
      !std::isfinite(control.convergence_tolerance) ||
      control.convergence_tolerance <= 0.0 ||
      control.convergence_tolerance >= 1.0 ||
      control.max_step_halving < 4 || control.max_iterations <= 3 ||
      !std::isfinite(control.max_newton_step) ||
      control.max_newton_step <= 0.0 ||
      !std::isfinite(control.boundary_probe_step) ||
      control.boundary_probe_step <= 0.0 ||
      control.max_boundary_probes <= 0 ||
      !std::isfinite(control.rank_tolerance) ||
      control.rank_tolerance <= 0.0 || control.rank_tolerance >= 1.0) {
    throw std::runtime_error(
      "persistent multi-penalty CUDA batch dimensions are invalid");
  }
  int previous_device = -1;
  check_cuda(cudaGetDevice(&previous_device),
             "query device before persistent multi-penalty batch");
  check_cuda(cudaSetDevice(prepared->device_id),
             "select device for persistent multi-penalty batch");
  try {
    const int q = prepared->q;
    const int penalty_count = prepared->penalty_count;
    const std::size_t y_count =
      static_cast<std::size_t>(n) * target_count;
    MultiPenaltyGcvCudaDiagnostics diagnostics;
    diagnostics.schema_version =
      "full-cuda-ci-multi-penalty-gcv-cuda-diagnostics-v1";
    diagnostics.execution_strategy =
      "persistent-one-setup-one-block-per-target-independent-optimizer";
    diagnostics.device_id = prepared->device_id;
    diagnostics.gpu_name = prepared->info.gpu_name;
    diagnostics.prepared_setup_upload_count =
      prepared->info.setup_upload_count;
    diagnostics.target_batch_upload_count = 1;
    diagnostics.cuda_qt_y_kernel_launch_count = 1;
    diagnostics.cuda_optimizer_kernel_launch_count = 1;
    diagnostics.cuda_optimizer_target_count = target_count;
    diagnostics.independent_target_states = true;
    diagnostics.target_specific_log_sp = true;
    diagnostics.true_batched_kernel = true;
    diagnostics.normal_equations_used = false;
    DeviceDecompositionTraceRecord* decomposition_trace = nullptr;
    unsigned int* decomposition_trace_counters = nullptr;
    unsigned int decomposition_trace_capacity = 0U;
    // Zero capacity preserves the production allocation and synchronization path.
    if (control.decomposition_trace_capacity_per_target > 0) {
      const int requested_capacity =
        control.decomposition_trace_capacity_per_target;
      if (prepared->d_decomposition_trace.count() == 0U) {
        prepared->d_decomposition_trace.allocate(
          static_cast<std::size_t>(prepared->target_capacity) *
            requested_capacity,
          &diagnostics);
        prepared->d_decomposition_trace_counters.allocate(2U, &diagnostics);
        prepared->decomposition_trace_capacity_per_target =
          requested_capacity;
      } else if (prepared->decomposition_trace_capacity_per_target !=
                   requested_capacity) {
        throw std::runtime_error(
          "persistent multi-penalty decomposition trace capacity changed");
      }
      decomposition_trace = prepared->d_decomposition_trace.get();
      decomposition_trace_counters =
        prepared->d_decomposition_trace_counters.get();
      decomposition_trace_capacity = static_cast<unsigned int>(
        static_cast<std::size_t>(target_count) * requested_capacity);
      check_cuda(cudaMemsetAsync(
        decomposition_trace_counters, 0, 2U * sizeof(unsigned int),
        prepared->stream.get()),
        "reset persistent multi-penalty decomposition trace counters");
      diagnostics.cuda_decomposition_trace_enabled = true;
      diagnostics.cuda_decomposition_trace_capacity_per_target =
        requested_capacity;
    }
    check_cuda(cudaMemcpyAsync(
      prepared->d_Y.get(), Y, y_count * sizeof(double),
      cudaMemcpyHostToDevice, prepared->stream.get()),
      "upload persistent multi-penalty target batch");
    diagnostics.h2d_copy_count = 1;
    const int target_blocks =
      (target_count + kBlockSize - 1) / kBlockSize;
    target_squared_norm_kernel<<<
      target_blocks, kBlockSize, 0, prepared->stream.get()>>>(
        prepared->d_Y.get(), prepared->d_squared_norm.get(),
        n, target_count);
    check_cuda(cudaGetLastError(),
               "launch persistent multi-penalty norm kernel");
    magic_qt_y_kernel<<<
      target_blocks, kBlockSize, 0, prepared->stream.get()>>>(
        prepared->d_qr_packed.get(), prepared->d_tau.get(),
        prepared->d_Y.get(), prepared->d_qt_work.get(),
        prepared->d_y0.get(), n, q, target_count);
    check_cuda(cudaGetLastError(),
               "launch persistent multi-penalty QR projection kernel");
    const double gradient_tolerance_factor = std::pow(
      control.convergence_tolerance, 1.0 / 3.0);
    optimize_multi_penalty_targets_kernel<<<
      target_count, kBlockSize, 0, prepared->stream.get()>>>(
        prepared->d_r.get(), prepared->d_pivot.get(),
        prepared->d_roots.get(), prepared->d_root_offsets.get(),
        prepared->d_ranks.get(), prepared->d_matrices.get(),
        prepared->d_y0.get(), prepared->d_squared_norm.get(),
        prepared->d_initial_log_sp.get(), prepared->d_workspaces.get(),
        prepared->d_states.get(), nullptr, n, q, penalty_count,
        target_count, control.convergence_tolerance,
        gradient_tolerance_factor,
        control.max_step_halving, control.max_iterations,
        control.max_newton_step, control.boundary_probe_step,
        control.max_boundary_probes, control.rank_tolerance, false, false,
        decomposition_trace, decomposition_trace_counters,
        decomposition_trace_capacity);
    check_cuda(cudaGetLastError(),
               "launch persistent multi-penalty optimizer kernel");
    optimize_multi_penalty_targets_kernel<<<
      target_count, kBlockSize, 0, prepared->stream.get()>>>(
        prepared->d_r.get(), prepared->d_pivot.get(),
        prepared->d_roots.get(), prepared->d_root_offsets.get(),
        prepared->d_ranks.get(), prepared->d_matrices.get(),
        prepared->d_y0.get(), prepared->d_squared_norm.get(),
        prepared->d_initial_log_sp.get(), prepared->d_workspaces.get(),
        prepared->d_replay_states.get(), prepared->d_states.get(),
        n, q, penalty_count, target_count, control.convergence_tolerance,
        gradient_tolerance_factor, control.max_step_halving,
        control.max_iterations, control.max_newton_step,
        control.boundary_probe_step, control.max_boundary_probes,
        control.rank_tolerance, true, true, decomposition_trace,
        decomposition_trace_counters, decomposition_trace_capacity);
    check_cuda(cudaGetLastError(),
               "launch persistent multi-penalty stability replay kernel");
    diagnostics.cuda_stability_replay_kernel_launch_count = 1;
    merge_stability_replay_states_kernel<<<
      target_blocks, kBlockSize, 0, prepared->stream.get()>>>(
        prepared->d_states.get(), prepared->d_replay_states.get(),
        target_count);
    check_cuda(cudaGetLastError(),
               "launch persistent multi-penalty stability merge kernel");
    diagnostics.cuda_stability_merge_kernel_launch_count = 1;
    const int coefficient_count = q * target_count;
    const int coefficient_blocks =
      (coefficient_count + 255) / 256;
    gather_multi_penalty_coefficients_kernel<<<
      coefficient_blocks, 256, 0, prepared->stream.get()>>>(
        prepared->d_states.get(), prepared->d_coefficients.get(),
        q, target_count);
    check_cuda(cudaGetLastError(),
               "launch persistent multi-penalty coefficient gather");
    const double alpha = 1.0;
    const double beta = 0.0;
    check_cublas(cublasDgemm(
      prepared->cublas.get(), CUBLAS_OP_N, CUBLAS_OP_N,
      n, target_count, q, &alpha, prepared->d_X.get(), n,
      prepared->d_coefficients.get(), q, &beta,
      prepared->d_fitted.get(), n),
      "form persistent multi-penalty selected fits");
    const int residual_blocks =
      (static_cast<int>(y_count) + 255) / 256;
    form_multi_penalty_residuals_kernel<<<
      residual_blocks, 256, 0, prepared->stream.get()>>>(
        prepared->d_Y.get(), prepared->d_fitted.get(),
        prepared->d_residuals.get(), static_cast<int>(y_count));
    check_cuda(cudaGetLastError(),
               "launch persistent multi-penalty residual kernel");
    check_cuda(cudaEventRecord(
      prepared->completion_event.get(), prepared->stream.get()),
      "record persistent multi-penalty residual completion");
    check_cuda(cudaMemcpyAsync(
      prepared->host_states.data(), prepared->d_states.get(),
      static_cast<std::size_t>(target_count) *
        sizeof(DeviceOptimizerState),
      cudaMemcpyDeviceToHost, prepared->stream.get()),
      "download persistent multi-penalty optimizer status");
    diagnostics.d2h_copy_count = 1;
    std::array<unsigned int, 2> host_trace_counters{{0U, 0U}};
    if (decomposition_trace != nullptr) {
      check_cuda(cudaMemcpyAsync(
        host_trace_counters.data(), decomposition_trace_counters,
        2U * sizeof(unsigned int), cudaMemcpyDeviceToHost,
        prepared->stream.get()),
        "download persistent multi-penalty decomposition trace counters");
      diagnostics.d2h_copy_count += 1;
    }
    check_cuda(cudaStreamSynchronize(prepared->stream.get()),
               "synchronize persistent multi-penalty batch");
    if (decomposition_trace != nullptr) {
      const std::uint64_t trace_requests = host_trace_counters[0];
      const std::uint64_t trace_overflow = host_trace_counters[1];
      const std::size_t trace_stored = static_cast<std::size_t>(std::min(
        trace_requests,
        static_cast<std::uint64_t>(decomposition_trace_capacity)));
      if (trace_requests != trace_stored + trace_overflow) {
        throw std::runtime_error(
          "persistent multi-penalty decomposition trace accounting failed");
      }
      std::vector<DeviceDecompositionTraceRecord> host_trace(trace_stored);
      if (!host_trace.empty()) {
        check_cuda(cudaMemcpyAsync(
          host_trace.data(), decomposition_trace,
          host_trace.size() * sizeof(DeviceDecompositionTraceRecord),
          cudaMemcpyDeviceToHost, prepared->stream.get()),
          "download persistent multi-penalty decomposition trace");
        diagnostics.d2h_copy_count += 1;
        check_cuda(cudaStreamSynchronize(prepared->stream.get()),
                   "synchronize persistent decomposition trace download");
      }
      materialize_decomposition_trace_diagnostics(
        host_trace, trace_requests, trace_overflow,
        control.decomposition_trace_capacity_per_target, &diagnostics);
    }
    MultiPenaltyGcvCudaBatchResult result;
    result.optimization = materialize_optimizer_output(
      prepared->host_states.data(), n, q, penalty_count, target_count,
      prepared->initial_log_sp, std::move(diagnostics));
    result.optimization.diagnostics.total_host_ms =
      std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    prepared->generation += 1;
    prepared->residual_slot_leased = true;
    prepared->info.solve_count += 1;
    prepared->info.cublas_gemm_count += 1;
    prepared->info.residual_kernel_count += 1;
    prepared->info.generation = prepared->generation;
    prepared->info.residual_slot_leased = true;
    result.residual =
      std::make_shared<MultiPenaltyGcvCudaResidualBatch>();
    result.residual->owner = prepared;
    result.residual->n = n;
    result.residual->q = q;
    result.residual->target_count = target_count;
    result.residual->device_id = prepared->device_id;
    result.residual->target_keys = target_keys;
    result.residual->optimizer_status =
      result.optimization.optimizer_status;
    result.residual->generation = prepared->generation;
    check_cuda(cudaSetDevice(previous_device),
               "restore device after persistent multi-penalty batch");
    return result;
  } catch (...) {
    cudaSetDevice(previous_device);
    throw;
  }
}

MultiPenaltyGcvCudaMultiResult multi_penalty_gcv_cuda_optimize_multi(
    std::vector<MultiPenaltyGcvCudaMultiRequest> requests,
    int requested_concurrency) {
  const auto started = std::chrono::steady_clock::now();
  if (requests.empty() || requested_concurrency <= 0 ||
      requested_concurrency > kMultiPenaltyGcvMaximumConcurrentSetups) {
    throw std::runtime_error(
      "multi-setup multi-penalty CUDA concurrency is invalid");
  }
  int target_count = 0;
  for (const MultiPenaltyGcvCudaMultiRequest& request : requests) {
    if (!request.prepared || request.n <= 0 || request.target_count <= 1 ||
        request.Y.size() != static_cast<std::size_t>(request.n) *
          request.target_count ||
        request.target_keys.size() !=
          static_cast<std::size_t>(request.target_count)) {
      throw std::runtime_error(
        "multi-setup multi-penalty CUDA request is malformed");
    }
    target_count += request.target_count;
  }

  MultiPenaltyGcvCudaMultiResult output;
  output.schema_version =
    "full-cuda-ci-multi-penalty-gcv-cuda-multi-setup-v1";
  output.setups.resize(requests.size());
  MultiPenaltyGcvCudaMultiDiagnostics& diagnostics = output.diagnostics;
  diagnostics.schema_version =
    "full-cuda-ci-multi-penalty-gcv-cuda-multi-diagnostics-v1";
  diagnostics.execution_strategy =
    "bounded-independent-prepared-streams-v1";
  diagnostics.setup_count = static_cast<int>(requests.size());
  diagnostics.target_count = target_count;
  diagnostics.requested_concurrency = requested_concurrency;
  diagnostics.worker_count = std::min(
    requested_concurrency, static_cast<int>(requests.size()));
  diagnostics.setup_stream_count = static_cast<int>(requests.size());

  std::atomic<std::size_t> next{0};
  std::atomic<int> ready{0};
  std::atomic<int> active{0};
  std::atomic<int> maximum_active{0};
  std::atomic<bool> start{false};
  std::atomic<bool> cancelled{false};
  std::mutex error_mutex;
  std::exception_ptr first_error;
  auto worker = [&]() {
    ready.fetch_add(1, std::memory_order_release);
    while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
    while (!cancelled.load(std::memory_order_acquire)) {
      const std::size_t index = next.fetch_add(1, std::memory_order_relaxed);
      if (index >= requests.size()) break;
      const int in_flight = active.fetch_add(1, std::memory_order_acq_rel) + 1;
      int observed = maximum_active.load(std::memory_order_relaxed);
      while (in_flight > observed && !maximum_active.compare_exchange_weak(
               observed, in_flight, std::memory_order_release,
               std::memory_order_relaxed)) {
      }
      try {
        MultiPenaltyGcvCudaMultiRequest& request = requests[index];
        output.setups[index] = multi_penalty_gcv_cuda_optimize_batch(
          request.prepared, request.Y.data(), request.n,
          request.target_count, request.target_keys, request.control);
      } catch (...) {
        {
          std::lock_guard<std::mutex> lock(error_mutex);
          if (!first_error) first_error = std::current_exception();
        }
        cancelled.store(true, std::memory_order_release);
      }
      active.fetch_sub(1, std::memory_order_acq_rel);
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>(diagnostics.worker_count));
  try {
    for (int index = 0; index < diagnostics.worker_count; ++index) {
      workers.emplace_back(worker);
    }
  } catch (...) {
    cancelled.store(true, std::memory_order_release);
    start.store(true, std::memory_order_release);
    for (std::thread& thread : workers) {
      if (thread.joinable()) thread.join();
    }
    throw;
  }
  while (ready.load(std::memory_order_acquire) < diagnostics.worker_count) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (std::thread& thread : workers) thread.join();
  if (first_error) std::rethrow_exception(first_error);

  diagnostics.max_host_calls_in_flight =
    maximum_active.load(std::memory_order_acquire);
  for (const MultiPenaltyGcvCudaBatchResult& setup : output.setups) {
    diagnostics.summed_setup_host_ms +=
      setup.optimization.diagnostics.total_host_ms;
  }
  diagnostics.wall_host_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - started).count();
  diagnostics.host_overlap_factor = diagnostics.wall_host_ms > 0.0 ?
    diagnostics.summed_setup_host_ms / diagnostics.wall_host_ms : 0.0;
  return output;
}

MultiPenaltyGcvCudaResidualInfo multi_penalty_gcv_cuda_residual_info(
    const std::shared_ptr<MultiPenaltyGcvCudaResidualBatch>& residual) {
  if (!residual || !residual->owner) {
    throw std::runtime_error(
      "multi-penalty CUDA residual token has been freed");
  }
  std::lock_guard<std::mutex> lock(residual->owner->mutex);
  MultiPenaltyGcvCudaResidualInfo info;
  info.n = residual->n;
  info.coefficient_dim = residual->q;
  info.target_count = residual->target_count;
  info.device_id = residual->device_id;
  info.target_keys = residual->target_keys;
  info.optimizer_status = residual->optimizer_status;
  info.released = residual->released;
  info.generation = residual->generation;
  info.device_resident = !residual->released &&
    residual->owner->generation == residual->generation &&
    residual->owner->residual_slot_leased;
  return info;
}

MultiPenaltyGcvCudaResidualConsumerView
acquire_multi_penalty_gcv_cuda_residual_consumer_view(
    const std::shared_ptr<MultiPenaltyGcvCudaResidualBatch>& residual) {
  if (!residual || !residual->owner) {
    throw std::runtime_error(
      "multi-penalty CUDA residual token has been freed");
  }
  std::lock_guard<std::mutex> lock(residual->owner->mutex);
  require_live_multi_penalty_residual(residual);
  if (!std::all_of(
        residual->optimizer_status.begin(), residual->optimizer_status.end(),
        [](int value) { return value == 0; })) {
    throw std::runtime_error(
      "multi-penalty CUDA residual consumer rejects optimizer failure");
  }
  MultiPenaltyGcvCudaResidualConsumerView view;
  view.residuals = residual->owner->d_residuals.get();
  view.n = residual->n;
  view.target_count = residual->target_count;
  view.device_id = residual->device_id;
  view.producer_stream = residual->owner->stream.get();
  view.producer_completion_event =
    residual->owner->completion_event.get();
  view.target_keys = residual->target_keys;
  return view;
}

MultiPenaltyGcvCudaResidualShadow
materialize_multi_penalty_gcv_cuda_residual_shadow(
    const std::shared_ptr<MultiPenaltyGcvCudaResidualBatch>& residual) {
  if (!residual || !residual->owner) {
    throw std::runtime_error(
      "multi-penalty CUDA residual token has been freed");
  }
  std::lock_guard<std::mutex> lock(residual->owner->mutex);
  require_live_multi_penalty_residual(residual);
  int previous_device = -1;
  check_cuda(cudaGetDevice(&previous_device),
             "query device before multi-penalty residual shadow");
  check_cuda(cudaSetDevice(residual->device_id),
             "select device for multi-penalty residual shadow");
  try {
    check_cuda(cudaEventSynchronize(
      residual->owner->completion_event.get()),
      "wait for multi-penalty residual shadow");
    MultiPenaltyGcvCudaResidualShadow shadow;
    shadow.n = residual->n;
    shadow.coefficient_dim = residual->q;
    shadow.target_count = residual->target_count;
    const std::size_t coefficient_count =
      static_cast<std::size_t>(residual->q) * residual->target_count;
    const std::size_t observation_count =
      static_cast<std::size_t>(residual->n) * residual->target_count;
    shadow.coefficients.resize(coefficient_count);
    shadow.fitted.resize(observation_count);
    shadow.residuals.resize(observation_count);
    check_cuda(cudaMemcpy(
      shadow.coefficients.data(), residual->owner->d_coefficients.get(),
      coefficient_count * sizeof(double), cudaMemcpyDeviceToHost),
      "download multi-penalty coefficient shadow");
    check_cuda(cudaMemcpy(
      shadow.fitted.data(), residual->owner->d_fitted.get(),
      observation_count * sizeof(double), cudaMemcpyDeviceToHost),
      "download multi-penalty fitted shadow");
    check_cuda(cudaMemcpy(
      shadow.residuals.data(), residual->owner->d_residuals.get(),
      observation_count * sizeof(double), cudaMemcpyDeviceToHost),
      "download multi-penalty residual shadow");
    residual->owner->info.residual_shadow_d2h_count += 3;
    residual->owner->info.residual_shadow_d2h_bytes +=
      (coefficient_count + 2 * observation_count) * sizeof(double);
    check_cuda(cudaSetDevice(previous_device),
               "restore device after multi-penalty residual shadow");
    return shadow;
  } catch (...) {
    cudaSetDevice(previous_device);
    throw;
  }
}

void release_multi_penalty_gcv_cuda_residual(
    const std::shared_ptr<MultiPenaltyGcvCudaResidualBatch>& residual) {
  if (!residual || !residual->owner) return;
  std::lock_guard<std::mutex> lock(residual->owner->mutex);
  if (residual->released) return;
  if (residual->owner->generation == residual->generation) {
    residual->owner->residual_slot_leased = false;
    residual->owner->info.residual_slot_leased = false;
  }
  residual->released = true;
}

}  // namespace fastkpc
