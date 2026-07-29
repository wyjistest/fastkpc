#include "mgcv_single_penalty_gcv.hpp"
#include "mgcv_fixed_sp_runtime_types.hpp"
#include "lapack_312_small_dgesdd.cuh"
#include "third_party/glibc_2_35_exp_fma.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fastkpc {
namespace {

constexpr int kReductionBlock = 128;
constexpr int kOptimizerBlock = 64;
constexpr int kExactWarp = 32;
constexpr int kMagicEigensolverBatch = 64;
constexpr double kDenominatorFloor = 1e-8;
constexpr double kConvergenceTolerance = 1e-7;
constexpr int kMaxStepHalving = 15;
constexpr int kMaxIterations = 400;
constexpr double kMaxNewtonStep = 5.0;
constexpr double kBoundaryProbeStep = 2.0;
constexpr int kMaxBoundaryProbes = 5;
constexpr double kFlatObjectiveResolutionMultiplier = 8.0;
constexpr double kDoubleEpsilon = 2.22044604925031308085e-16;
constexpr double kLapackEpsilon = 1.11022302462515654042e-16;
constexpr double kMagicRankTolerance = 1.490116119384765625e-8;
constexpr int kMaximumMrootCoefficientDim = 64;
constexpr int kExactEndpointCounterCount = 10;
constexpr double kEndpointResidualRiskLimit = 1e-8;
constexpr double kEndpointResolutionMultiplier = 24.0;

void check_cuda(cudaError_t status, const char* stage) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string(stage) + ": " + cudaGetErrorString(status));
  }
}

void check_cublas(cublasStatus_t status, const char* stage) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
      std::string(stage) + " failed with cuBLAS status " +
      std::to_string(static_cast<int>(status)));
  }
}

thread_local cudaStream_t g_device_allocation_stream = nullptr;

class DeviceAllocationStreamScope {
 public:
  explicit DeviceAllocationStreamScope(cudaStream_t stream)
      : previous_(g_device_allocation_stream) {
    g_device_allocation_stream = stream;
  }
  DeviceAllocationStreamScope(const DeviceAllocationStreamScope&) = delete;
  DeviceAllocationStreamScope& operator=(
    const DeviceAllocationStreamScope&) = delete;
  ~DeviceAllocationStreamScope() {
    g_device_allocation_stream = previous_;
  }

 private:
  cudaStream_t previous_ = nullptr;
};

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() : allocation_stream_(g_device_allocation_stream) {}
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() {
    if (data_ == nullptr) return;
    if (allocation_stream_ != nullptr) {
      cudaFreeAsync(data_, allocation_stream_);
    } else {
      cudaFree(data_);
    }
  }

  void allocate(std::size_t count,
                SinglePenaltyGcvCudaDiagnostics* diagnostics) {
    if (count == 0) return;
    if (allocation_stream_ != nullptr) {
      check_cuda(cudaMallocAsync(
                   reinterpret_cast<void**>(&data_), count * sizeof(T),
                   allocation_stream_),
                 "stream-allocate single-penalty GCV buffer");
      diagnostics->stream_ordered_allocation_count += 1;
    } else {
      check_cuda(cudaMalloc(
                   reinterpret_cast<void**>(&data_), count * sizeof(T)),
                 "allocate single-penalty GCV buffer");
      diagnostics->synchronous_allocation_count += 1;
    }
    count_ = count;
    diagnostics->device_allocation_count += 1;
    diagnostics->device_allocation_bytes += count * sizeof(T);
  }

  T* get() { return data_; }
  const T* get() const { return data_; }
  std::size_t count() const { return count_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
  cudaStream_t allocation_stream_ = nullptr;
};

class Stream {
 public:
  Stream() {
    check_cuda(cudaStreamCreateWithFlags(&value_, cudaStreamNonBlocking),
               "create single-penalty GCV stream");
  }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  ~Stream() {
    if (value_ != nullptr) cudaStreamDestroy(value_);
  }
  cudaStream_t get() const { return value_; }

 private:
  cudaStream_t value_ = nullptr;
};

class Event {
 public:
  Event() {
    check_cuda(cudaEventCreate(&value_),
               "create single-penalty GCV timing event");
  }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  ~Event() {
    if (value_ != nullptr) cudaEventDestroy(value_);
  }
  cudaEvent_t get() const { return value_; }

 private:
  cudaEvent_t value_ = nullptr;
};

class BlasHandle {
 public:
  explicit BlasHandle(cudaStream_t stream) {
    check_cublas(cublasCreate(&value_),
                 "create single-penalty GCV cuBLAS handle");
    check_cublas(cublasSetStream(value_, stream),
                 "bind single-penalty GCV cuBLAS stream");
    check_cublas(cublasSetMathMode(value_, kFixedSpCublasMathModeValue),
                 "set single-penalty GCV pedantic math");
    check_cublas(cublasSetAtomicsMode(
                   value_, kFixedSpCublasAtomicsModeValue),
                 "disable single-penalty GCV cuBLAS atomics");
  }
  BlasHandle(const BlasHandle&) = delete;
  BlasHandle& operator=(const BlasHandle&) = delete;
  ~BlasHandle() {
    if (value_ != nullptr) cublasDestroy(value_);
  }
  cublasHandle_t get() const { return value_; }

 private:
  cublasHandle_t value_ = nullptr;
};

float elapsed_ms(const Event& begin, const Event& end) {
  float value = 0.0f;
  check_cuda(cudaEventElapsedTime(&value, begin.get(), end.get()),
             "measure single-penalty GCV stage");
  return value;
}

__device__ double stable_hat_value(double log_sp, double eigenvalue) {
  if (eigenvalue == 0.0) return 1.0;
  const double coordinate = log_sp + log(eigenvalue);
  if (coordinate >= 0.0) {
    const double value = glibc235::exp_fma_rn(-coordinate);
    return value / (1.0 + value);
  }
  const double value = glibc235::exp_fma_rn(coordinate);
  return 1.0 / (1.0 + value);
}

struct DeviceObjective {
  double rss;
  double edf;
  double score;
  double gradient;
  double hessian;
  double residual_sensitivity;
  int valid;
};

__device__ double rounded_multiply_add(double accumulator,
                                       double left,
                                       double right) {
  return __dadd_rn(accumulator, __dmul_rn(left, right));
}

__device__ DeviceObjective evaluate_objective(
    double log_sp,
    const double* eigenvalues,
    const double* spectral_projection,
    double y_squared_norm,
    int p,
    int target,
    int n) {
  double linear = 0.0;
  double quadratic = 0.0;
  double edf = 0.0;
  double rss_gradient = 0.0;
  double rss_hessian = 0.0;
  double df_gradient = 0.0;
  double df_hessian = 0.0;
  double residual_derivative_squared = 0.0;
  for (int component = 0; component < p; ++component) {
    const double h = stable_hat_value(log_sp, eigenvalues[component]);
    const double z = spectral_projection[component + p * target];
    const double z_squared = z * z;
    const double one_minus_h = 1.0 - h;
    const double weight = h * one_minus_h;
    linear += h * z_squared;
    quadratic += h * h * z_squared;
    edf += h;
    rss_gradient += 2.0 * h * one_minus_h * one_minus_h * z_squared;
    rss_hessian += 2.0 * h * one_minus_h * one_minus_h *
      (3.0 * h - 1.0) * z_squared;
    df_gradient += weight;
    df_hessian -= weight * (1.0 - 2.0 * h);
    const double residual_derivative = weight * z;
    residual_derivative_squared +=
      residual_derivative * residual_derivative;
  }
  double rss = y_squared_norm - 2.0 * linear + quadratic;
  if (rss < 0.0 && isfinite(rss)) rss = 0.0;
  const double residual_df = static_cast<double>(n) - edf;
  DeviceObjective result;
  result.rss = rss;
  result.edf = edf;
  result.score = CUDART_INF;
  result.gradient = CUDART_NAN;
  result.hessian = CUDART_NAN;
  result.residual_sensitivity = CUDART_NAN;
  result.valid = 0;
  if (!isfinite(rss) || !isfinite(edf) || rss <= 0.0 ||
      residual_df <= kDenominatorFloor) {
    return result;
  }
  const double score = static_cast<double>(n) * rss /
    (residual_df * residual_df);
  const double log_gradient = rss_gradient / rss -
    2.0 * df_gradient / residual_df;
  const double log_hessian = rss_hessian / rss -
    (rss_gradient / rss) * (rss_gradient / rss) -
    2.0 * (df_hessian / residual_df -
           (df_gradient / residual_df) *
             (df_gradient / residual_df));
  result.score = score;
  result.gradient = score * log_gradient;
  result.hessian = score *
    (log_hessian + log_gradient * log_gradient);
  result.residual_sensitivity =
    sqrt(residual_derivative_squared) / sqrt(rss);
  result.valid = isfinite(result.score) && isfinite(result.gradient) &&
    isfinite(result.hessian) && isfinite(result.residual_sensitivity);
  return result;
}

__global__ void magic_target_squared_norm_kernel(
    const double* Y, double* y_squared_norm, int n, int target_count) {
  const int target = blockIdx.x * blockDim.x + threadIdx.x;
  if (target >= target_count) return;
  double value = 0.0;
  for (int row = 0; row < n; ++row) {
    const double y = Y[row + n * target];
    value = rounded_multiply_add(value, y, y);
  }
  y_squared_norm[target] = value;
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

  // DORM2R, SIDE='L', TRANS='T': apply H(1), ..., H(q).
  for (int reflector = 0; reflector < q; ++reflector) {
    double dot = target_work[reflector];
    const double* vector = qr_packed + static_cast<std::size_t>(n) * reflector;
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
  const double value = *left;
  *left = *right;
  *right = value;
}

__device__ int dpstf2_upper(double* matrix, int* pivot, double* work, int q) {
  for (int index = 0; index < q; ++index) {
    pivot[index] = index + 1;
    work[index] = 0.0;
  }

  int pivot_index = 0;
  double diagonal = matrix[0];
  for (int index = 1; index < q; ++index) {
    const double candidate = matrix[index + q * index];
    if (candidate > diagonal) {
      pivot_index = index;
      diagonal = candidate;
    }
  }
  if (diagonal <= 0.0 || isnan(diagonal)) return 0;
  const double stopping_value = __dmul_rn(
    __dmul_rn(static_cast<double>(q), kLapackEpsilon), diagonal);

  for (int column = 0; column < q; ++column) {
    for (int index = column; index < q; ++index) {
      if (column > 0) {
        const double previous = matrix[(column - 1) + q * index];
        work[index] = rounded_multiply_add(
          work[index], previous, previous);
      }
      work[q + index] = __dadd_rn(
        matrix[index + q * index], -work[index]);
    }

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
      if (diagonal <= stopping_value || isnan(diagonal)) {
        matrix[column + q * column] = diagonal;
        return column;
      }
    }

    if (column != pivot_index) {
      matrix[pivot_index + q * pivot_index] =
        matrix[column + q * column];
      for (int row = 0; row < column; ++row) {
        swap_double(
          matrix + row + q * column,
          matrix + row + q * pivot_index);
      }
      for (int trailing = pivot_index + 1; trailing < q; ++trailing) {
        swap_double(
          matrix + column + q * trailing,
          matrix + pivot_index + q * trailing);
      }
      for (int index = column + 1; index < pivot_index; ++index) {
        swap_double(
          matrix + column + q * index,
          matrix + index + q * pivot_index);
      }
      swap_double(work + column, work + pivot_index);
      const int saved_pivot = pivot[pivot_index];
      pivot[pivot_index] = pivot[column];
      pivot[column] = saved_pivot;
    }

    diagonal = sqrt(diagonal);
    matrix[column + q * column] = diagonal;
    if (column + 1 < q) {
      for (int trailing = column + 1; trailing < q; ++trailing) {
        double dot = 0.0;
        for (int row = 0; row < column; ++row) {
          dot = rounded_multiply_add(
            dot,
            matrix[row + q * trailing],
            matrix[row + q * column]);
        }
        const double value = __dadd_rn(
          matrix[column + q * trailing], -dot);
        matrix[column + q * trailing] =
          __dmul_rn(value, 1.0 / diagonal);
      }
    }
  }
  return q;
}

__global__ void single_penalty_mroot_kernel(
    const double* penalty_matrix,
    const double* log_sp,
    double* factor,
    double* roots,
    double* work,
    int* ranks,
    int* pivots,
    int q,
    int requested_rank,
    int candidate_count) {
  const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
  if (candidate >= candidate_count) return;
  const int square = q * q;
  double* candidate_factor = factor +
    static_cast<std::size_t>(candidate) * square;
  double* candidate_work = work +
    static_cast<std::size_t>(candidate) * 2 * q;
  int* candidate_pivot = pivots +
    static_cast<std::size_t>(candidate) * q;
  double* candidate_root = roots +
    static_cast<std::size_t>(candidate) * q * requested_rank;
  const double sp = glibc235::exp_fma_rn(log_sp[candidate]);
  for (int index = 0; index < square; ++index) {
    candidate_factor[index] = __dmul_rn(sp, penalty_matrix[index]);
  }

  const int rank = dpstf2_upper(
    candidate_factor, candidate_pivot, candidate_work, q);
  ranks[candidate] = rank;
  for (int root_column = 0; root_column < requested_rank; ++root_column) {
    for (int original_column = 0; original_column < q; ++original_column) {
      double value = CUDART_NAN;
      if (rank >= requested_rank) {
        value = 0.0;
        int factor_column = 0;
        while (factor_column < q &&
               candidate_pivot[factor_column] != original_column + 1) {
          ++factor_column;
        }
        if (factor_column < q && root_column <= factor_column) {
          value = candidate_factor[root_column + q * factor_column];
        }
      }
      candidate_root[original_column + q * root_column] = value;
    }
  }
}

struct ExactEndpointObjective {
  double rss;
  double edf;
  double score;
  double gradient;
  double hessian;
  double residual_sensitivity;
  int valid;
  int solver_info;
};

struct SpectralEndpointRefinementState {
  int active;
};

__device__ ExactEndpointObjective evaluate_exact_endpoint_objective(
    double log_sp,
    const double* magic_r,
    const double* penalty_matrix,
    const double* penalty_root,
    const double* y0,
    double y_squared_norm,
    int q,
    int penalty_rank,
    int n,
    lapack312::Workspace* workspace) {
  ExactEndpointObjective result;
  result.rss = CUDART_NAN;
  result.edf = CUDART_NAN;
  result.score = CUDART_INF;
  result.gradient = CUDART_NAN;
  result.hessian = CUDART_NAN;
  result.residual_sensitivity = 0.0;
  result.valid = 0;
  result.solver_info = 0;
  const int lane_count = lapack312::cooperative_lane_count();
  const int lane = lapack312::cooperative_lane_index(lane_count);
  const int square = q * q;
  double* factor = workspace->bidiagonal_vt;
  const double sp = glibc235::exp_fma_rn(log_sp);
  const int rows = q + penalty_rank;
  int root_rank = 0;
  if (threadIdx.x == 0) {
    for (int index = 0; index < square; ++index) {
      factor[index] = __dmul_rn(sp, penalty_matrix[index]);
    }
    root_rank = dpstf2_upper(
      factor, workspace->iwork, workspace->work, q);
    workspace->iwork[lapack312::kMaximumColumns] = root_rank;
  }
  __syncthreads();
  root_rank = workspace->iwork[lapack312::kMaximumColumns];
  if (root_rank != penalty_rank) {
    result.solver_info = -1;
    return result;
  }

  if (threadIdx.x == 0) {
    for (int column = 0; column < q; ++column) {
      int factor_column = 0;
      while (factor_column < q &&
             workspace->iwork[factor_column] != column + 1) {
        ++factor_column;
      }
      for (int row = 0; row < rows; ++row) {
        double value = 0.0;
        if (row < q) {
          value = magic_r[row + q * column];
        } else {
          const int root_column = row - q;
          if (factor_column < q && root_column <= factor_column) {
            value = factor[root_column + q * factor_column];
          }
        }
        workspace->a[row + rows * column] = value;
      }
    }
  }
  __syncthreads();

  result.solver_info = lapack312::small_dgesdd_left(rows, q, workspace);
  if (result.solver_info != 0) return result;
  if (blockDim.x > 32 && threadIdx.x >= 32) return result;
  int rank = q;
  const double threshold = __dmul_rn(
    workspace->bidiagonal[0], kMagicRankTolerance);
  while (rank > 0 && workspace->bidiagonal[rank - 1] < threshold) --rank;
  if (rank <= 0) {
    result.solver_info = -2;
    return result;
  }

  double* projected = workspace->qr_tau;
  for (int component = lane; component < rank; component += lane_count) {
    double value = 0.0;
    for (int row = 0; row < q; ++row) {
      value = rounded_multiply_add(
        value, workspace->left_u[row + rows * component], y0[row]);
    }
    projected[component] = value;
  }
  lapack312::cooperative_sync(lane_count);
  double* fitted_coordinates = workspace->work;
  for (int row = lane; row < q; row += lane_count) {
    double fitted_coordinate = 0.0;
    for (int component = 0; component < rank; ++component) {
      fitted_coordinate = rounded_multiply_add(
        fitted_coordinate,
        workspace->left_u[row + rows * component], projected[component]);
    }
    fitted_coordinates[row] = fitted_coordinate;
  }
  lapack312::cooperative_sync(lane_count);
  double y_ay = 0.0;
  double y_aay = 0.0;
  double trace_a = 0.0;
  if (lane == 0) {
    for (int component = 0; component < rank; ++component) {
      const double value = projected[component];
      y_ay = rounded_multiply_add(y_ay, value, value);
    }
    for (int row = 0; row < q; ++row) {
      const double value = fitted_coordinates[row];
      y_aay = rounded_multiply_add(y_aay, value, value);
    }
    for (int component = 0; component < rank; ++component) {
      for (int row = 0; row < q; ++row) {
        const double value = workspace->left_u[row + rows * component];
        trace_a = rounded_multiply_add(trace_a, value, value);
      }
    }
  }
  if (lane_count == 32) {
    y_ay = __shfl_sync(0xffffffffu, y_ay, 0);
    y_aay = __shfl_sync(0xffffffffu, y_aay, 0);
    trace_a = __shfl_sync(0xffffffffu, trace_a, 0);
  }
  double rss = __dadd_rn(
    __dadd_rn(y_squared_norm, __dmul_rn(-2.0, y_ay)), y_aay);
  if (rss < 0.0 && isfinite(rss)) rss = 0.0;
  const double residual_df = static_cast<double>(n) - trace_a;
  result.rss = rss;
  result.edf = trace_a;
  result.valid = isfinite(rss) && isfinite(trace_a) &&
    residual_df > kDenominatorFloor;
  if (result.valid) {
    result.score = __dmul_rn(static_cast<double>(n), rss) /
      __dmul_rn(residual_df, residual_df);
    result.valid = isfinite(result.score);
  }
  if (!result.valid) return result;

  // Replay magic_gH for m=1 using the same column-major operation order as
  // mgcv 1.9-1. All matrix scratch lives in the explicit device workspace.
  double* u1_u1 = workspace->a;
  double* penalty_metric = workspace->a + square;
  double* scaled_vt_root = workspace->r;
  double* root_times_u1_u1 = workspace->bidiagonal_u;
  double* penalty_hat = workspace->bidiagonal_vt;
  double* metric_y = workspace->bidiagonal_e;
  double* y_penalty_hat = workspace->tau_q;
  double* penalty_hat_y = workspace->tau_p;

  for (int column = 0; column < rank; ++column) {
    for (int row = column + lane; row < rank; row += lane_count) {
      double value = 0.0;
      for (int coordinate = 0; coordinate < q; ++coordinate) {
        value = rounded_multiply_add(
          value,
          workspace->left_u[coordinate + rows * row],
          workspace->left_u[coordinate + rows * column]);
      }
      u1_u1[row + rank * column] = value;
      u1_u1[column + rank * row] = value;
    }
  }
  lapack312::cooperative_sync(lane_count);

  for (int output = lane; output < rank * penalty_rank;
       output += lane_count) {
    const int component = output % rank;
    const int root_column = output / rank;
    double value = 0.0;
    for (int coordinate = 0; coordinate < q; ++coordinate) {
      value = rounded_multiply_add(
        value,
        workspace->bidiagonal_vt[component + q * coordinate],
        penalty_root[coordinate + q * root_column]);
    }
    scaled_vt_root[output] =
      __ddiv_rn(value, workspace->bidiagonal[component]);
  }
  lapack312::cooperative_sync(lane_count);

  for (int output = lane; output < penalty_rank * rank;
       output += lane_count) {
    const int root_column = output % penalty_rank;
    const int column = output / penalty_rank;
    double value = 0.0;
    for (int component = 0; component < rank; ++component) {
      value = rounded_multiply_add(
        value,
        scaled_vt_root[component + rank * root_column],
        u1_u1[component + rank * column]);
    }
    root_times_u1_u1[output] = value;
  }
  lapack312::cooperative_sync(lane_count);

  for (int output = lane; output < rank * rank; output += lane_count) {
    const int row = output % rank;
    const int column = output / rank;
    double value = 0.0;
    for (int root_column = 0; root_column < penalty_rank; ++root_column) {
      value = rounded_multiply_add(
        value,
        scaled_vt_root[row + rank * root_column],
        root_times_u1_u1[root_column + penalty_rank * column]);
    }
    penalty_hat[output] = value;
  }
  lapack312::cooperative_sync(lane_count);

  for (int output = lane; output < rank * rank; output += lane_count) {
    const int row = output % rank;
    const int column = output / rank;
    double value = 0.0;
    for (int root_column = 0; root_column < penalty_rank; ++root_column) {
      value = rounded_multiply_add(
        value,
        scaled_vt_root[row + rank * root_column],
        scaled_vt_root[column + rank * root_column]);
    }
    penalty_metric[output] = value;
  }
  lapack312::cooperative_sync(lane_count);

  for (int component = lane; component < rank; component += lane_count) {
    double my = 0.0;
    double yk = 0.0;
    double ky = 0.0;
    for (int inner = 0; inner < rank; ++inner) {
      my = rounded_multiply_add(
        my, projected[inner], penalty_metric[inner + rank * component]);
      yk = rounded_multiply_add(
        yk, projected[inner], penalty_hat[inner + rank * component]);
      ky = rounded_multiply_add(
        ky, projected[inner], penalty_hat[component + rank * inner]);
    }
    metric_y[component] = my;
    y_penalty_hat[component] = yk;
    penalty_hat_y[component] = ky;
  }
  lapack312::cooperative_sync(lane_count);

  double trace_penalty_hat = 0.0;
  for (int component = 0; component < rank; ++component) {
    trace_penalty_hat = __dadd_rn(
      trace_penalty_hat, penalty_hat[component + rank * component]);
  }
  const double delta_gradient = __dmul_rn(trace_penalty_hat, sp);

  double metric_hat_inner_product = 0.0;
  for (int index = 0; index < rank * rank; ++index) {
    metric_hat_inner_product = rounded_multiply_add(
      metric_hat_inner_product, penalty_metric[index], penalty_hat[index]);
  }
  const double exp_twice_log_sp =
    glibc235::exp_fma_rn(__dadd_rn(log_sp, log_sp));
  double delta_hessian = __dmul_rn(-2.0, exp_twice_log_sp);
  delta_hessian = __dmul_rn(delta_hessian, metric_hat_inner_product);
  delta_hessian = __dadd_rn(delta_hessian, delta_gradient);

  double norm_gradient_inner = 0.0;
  for (int component = 0; component < rank; ++component) {
    norm_gradient_inner = rounded_multiply_add(
      norm_gradient_inner, projected[component],
      __dadd_rn(metric_y[component], -penalty_hat_y[component]));
  }
  double norm_gradient = __dmul_rn(2.0, sp);
  norm_gradient = __dmul_rn(norm_gradient, norm_gradient_inner);

  double norm_hessian_inner = 0.0;
  for (int component = 0; component < rank; ++component) {
    const double my = metric_y[component];
    const double ky = penalty_hat_y[component];
    double term = __dadd_rn(__dmul_rn(my, ky), __dmul_rn(my, ky));
    term = __dadd_rn(
      term, -__dmul_rn(__dmul_rn(2.0, my), my));
    term = __dadd_rn(
      term, __dmul_rn(y_penalty_hat[component], my));
    norm_hessian_inner = __dadd_rn(norm_hessian_inner, term);
  }
  double norm_hessian = __dmul_rn(norm_hessian_inner, 2.0);
  norm_hessian = __dmul_rn(norm_hessian, exp_twice_log_sp);
  norm_hessian = __dadd_rn(norm_hessian, norm_gradient);

  const double score_scale = __ddiv_rn(
    static_cast<double>(n), __dmul_rn(residual_df, residual_df));
  double score_delta_scale = __dmul_rn(score_scale, 2.0);
  score_delta_scale = __dmul_rn(score_delta_scale, rss);
  score_delta_scale = __ddiv_rn(score_delta_scale, residual_df);
  const double mixed_scale = __ddiv_rn(
    __dmul_rn(-2.0, score_scale), residual_df);
  const double delta_square_scale = __ddiv_rn(
    __dmul_rn(3.0, score_delta_scale), residual_df);

  result.gradient = __dadd_rn(
    __dmul_rn(score_scale, norm_gradient),
    -__dmul_rn(score_delta_scale, delta_gradient));
  double hessian = __dmul_rn(
    mixed_scale,
    __dadd_rn(
      __dmul_rn(delta_gradient, norm_gradient),
      __dmul_rn(delta_gradient, norm_gradient)));
  hessian = __dadd_rn(
    hessian, __dmul_rn(score_scale, norm_hessian));
  hessian = __dadd_rn(
    hessian,
    __dmul_rn(
      __dmul_rn(delta_square_scale, delta_gradient), delta_gradient));
  result.hessian = __dadd_rn(
    hessian, -__dmul_rn(score_delta_scale, delta_hessian));
  result.valid = isfinite(result.gradient) && isfinite(result.hessian);
  return result;
}

__device__ DeviceObjective exact_endpoint_as_device_objective(
    const ExactEndpointObjective& exact) {
  DeviceObjective result;
  result.rss = exact.rss;
  result.edf = exact.edf;
  result.score = exact.score;
  result.gradient = exact.gradient;
  result.hessian = exact.hessian;
  result.residual_sensitivity = exact.residual_sensitivity;
  result.valid = exact.valid;
  return result;
}

__device__ ExactEndpointObjective broadcast_exact_endpoint_objective(
    const ExactEndpointObjective& local,
    ExactEndpointObjective* shared) {
  if (threadIdx.x == 0) *shared = local;
  __syncthreads();
  return *shared;
}

__global__ void target_squared_norm_kernel(
    const double* Y, double* y_squared_norm, int n, int target_count) {
  const int target = blockIdx.x;
  if (target >= target_count) return;
  __shared__ double scratch[kReductionBlock];
  double value = 0.0;
  for (int row = threadIdx.x; row < n; row += blockDim.x) {
    const double y = Y[row + n * target];
    value += y * y;
  }
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) y_squared_norm[target] = scratch[0];
}

__global__ void factor_exact_grid_augmented_matrices_kernel(
    const double* magic_r,
    const double* roots,
    const int* root_ranks,
    lapack312::Workspace* workspaces,
    double* singular_vectors,
    double* singular_values,
    int* solver_info,
  int q,
  int penalty_rank,
  int candidate_count) {
  const int candidate = blockIdx.x;
  if (candidate >= candidate_count) return;
  const int rows = q + penalty_rank;
  const int matrix_size = rows * q;
  lapack312::Workspace* workspace = workspaces + candidate;
  const double* candidate_root = roots +
    static_cast<std::size_t>(candidate) * q * penalty_rank;
  for (int index = threadIdx.x; index < matrix_size; index += blockDim.x) {
    const int row = index % rows;
    const int column = index / rows;
    double value = 0.0;
    if (row < q) {
      value = magic_r[row + q * column];
    } else if (root_ranks[candidate] == penalty_rank) {
      value = candidate_root[column + q * (row - q)];
    }
    workspace->a[index] = value;
  }
  __syncthreads();

  const int info = lapack312::small_dgesdd_left(rows, q, workspace);
  if (threadIdx.x == 0) solver_info[candidate] = info;
  if (info != 0) return;
  double* candidate_vectors = singular_vectors +
    static_cast<std::size_t>(candidate) * matrix_size;
  double* candidate_values = singular_values +
    static_cast<std::size_t>(candidate) * q;
  for (int component = threadIdx.x; component < q;
       component += blockDim.x) {
    candidate_values[component] = workspace->bidiagonal[component];
  }
  for (int index = threadIdx.x; index < matrix_size; index += blockDim.x) {
    candidate_vectors[index] = workspace->left_u[index];
  }
}

__global__ void evaluate_exact_grid_kernel(
    const double* singular_vectors,
    const double* singular_values,
    const int* solver_info,
    const int* root_ranks,
    const double* y0,
    const double* y_squared_norm,
    SinglePenaltyGcvGridCell* grid,
    int q,
    int penalty_rank,
    int n,
    int target_count,
    int candidate_count) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = target_count * candidate_count;
  if (cell >= total) return;
  const int target = cell % target_count;
  const int candidate = cell / target_count;
  SinglePenaltyGcvGridCell result;
  result.rss = CUDART_NAN;
  result.edf = CUDART_NAN;
  result.score = CUDART_INF;
  result.valid = 0;
  if (solver_info[candidate] != 0 ||
      root_ranks[candidate] != penalty_rank) {
    grid[cell] = result;
    return;
  }

  const int rows = q + penalty_rank;
  const double* u = singular_vectors +
    static_cast<std::size_t>(candidate) * rows * q;
  const double* d = singular_values +
    static_cast<std::size_t>(candidate) * q;
  int rank = q;
  const double threshold = __dmul_rn(d[0], kMagicRankTolerance);
  while (rank > 0 && d[rank - 1] < threshold) --rank;
  if (rank <= 0) {
    grid[cell] = result;
    return;
  }

  double y_ay = 0.0;
  double y1[kMaximumMrootCoefficientDim];
  for (int component = 0; component < rank; ++component) {
    double value = 0.0;
    for (int row = 0; row < q; ++row) {
      value = rounded_multiply_add(
        value, u[row + rows * component], y0[row + q * target]);
    }
    y1[component] = value;
    y_ay = rounded_multiply_add(y_ay, value, value);
  }

  double y_aay = 0.0;
  for (int row = 0; row < q; ++row) {
    double fitted_coordinate = 0.0;
    for (int component = 0; component < rank; ++component) {
      fitted_coordinate = rounded_multiply_add(
        fitted_coordinate, u[row + rows * component], y1[component]);
    }
    y_aay = rounded_multiply_add(
      y_aay, fitted_coordinate, fitted_coordinate);
  }

  double trace_a = 0.0;
  for (int component = 0; component < rank; ++component) {
    for (int row = 0; row < q; ++row) {
      const double value = u[row + rows * component];
      trace_a = rounded_multiply_add(trace_a, value, value);
    }
  }
  double rss = __dadd_rn(
    __dadd_rn(y_squared_norm[target], __dmul_rn(-2.0, y_ay)), y_aay);
  if (rss < 0.0 && isfinite(rss)) rss = 0.0;
  const double residual_df = static_cast<double>(n) - trace_a;
  result.rss = rss;
  result.edf = trace_a;
  result.valid = isfinite(rss) && isfinite(trace_a) &&
    residual_df > kDenominatorFloor;
  if (result.valid) {
    result.score = __dmul_rn(static_cast<double>(n), rss) /
      __dmul_rn(residual_df, residual_df);
    result.valid = isfinite(result.score);
  }
  grid[cell] = result;
}

__device__ void append_transcript(
    SinglePenaltyGcvTranscriptEntry* transcript,
    int* transcript_count,
    int* transcript_overflow,
    int target,
    int stage,
    int iteration,
    int evaluation,
    double current_log_sp,
    double proposed_step,
    double trial_log_sp,
    const DeviceObjective& objective,
    int accepted,
    int step_source) {
  if (transcript == nullptr) return;
  int count = transcript_count[target];
  if (count >= kSinglePenaltyGcvTranscriptCapacity) {
    transcript_overflow[target] = 1;
    return;
  }
  SinglePenaltyGcvTranscriptEntry entry;
  entry.current_log_sp = current_log_sp;
  entry.proposed_step = proposed_step;
  entry.trial_log_sp = trial_log_sp;
  entry.objective = objective.score;
  entry.gradient = objective.gradient;
  entry.hessian = objective.hessian;
  entry.stage = stage;
  entry.iteration = iteration;
  entry.evaluation = evaluation;
  entry.accepted = accepted;
  entry.step_source = step_source;
  transcript[target * kSinglePenaltyGcvTranscriptCapacity + count] = entry;
  transcript_count[target] = count + 1;
}

struct DeviceMagicOptimizerState {
  DeviceObjective current;
  DeviceObjective unresolved_trial;
  double log_sp;
  double unresolved_trial_log_sp;
  double minimum_score;
  double score_reduction;
  double prior_gradient;
  double newton_step;
  double steepest_step;
  double proposed_step;
  double reported_gradient;
  double pre_boundary_log_sp;
  double probe_direction;
  double iteration_origin_score;
  int iteration;
  int tries;
  int score_calls;
  int actual_calls;
  int use_steepest_descent;
  int step_source;
  int try_step;
  int converged;
  int step_failed;
  int iteration_flat;
  int iteration_accepted;
  int exact_iteration_replayed;
  int current_exact;
  int unresolved_pending;
  int unresolved_from_approximate;
  int unresolved_newton_seen;
  int hessian_positive;
  int boundary_active;
  int boundary_exact_pending;
  int boundary_probe_count;
  int boundary_accepted_count;
  int selective_risk;
  int selective_deferred;
  int selective_deferred_improved;
  int selective_steepest;
  int invalid;
};

__global__ void mark_spectral_exact_replay_kernel(
    const SpectralEndpointRefinementState* endpoint_states,
    const SinglePenaltyGcvOptimizerResult* targets,
    int* replay_flags,
    int* replay_reason_counts,
    int target_count) {
  const int target = blockIdx.x * blockDim.x + threadIdx.x;
  if (target >= target_count) return;
  const SinglePenaltyGcvOptimizerResult& result = targets[target];
  const bool endpoint_risk = endpoint_states[target].active != 0;
  const bool convergence_risk = result.fully_converged == 0 ||
    result.termination != static_cast<int>(
      SinglePenaltyGcvTermination::ScoreAndGradient);
  const bool boundary_risk = result.boundary_accepted_count > 0;
  const bool spectral_precision_floor_risk =
    !isfinite(result.reported_rms_gradient) ||
    result.reported_rms_gradient <= 0.5 * kConvergenceTolerance;
  const bool numerical_risk = !isfinite(result.sp) ||
    !isfinite(result.score) || !isfinite(result.edf) ||
    !isfinite(result.rss) || !isfinite(result.gradient) ||
    !isfinite(result.hessian) || result.hessian_positive_definite == 0 ||
    spectral_precision_floor_risk;
  const bool replay = endpoint_risk || convergence_risk ||
    boundary_risk || numerical_risk;
  replay_flags[target] = replay ? 1 : 0;
  if (!replay) return;
  if (endpoint_risk) atomicAdd(replay_reason_counts, 1);
  if (convergence_risk) atomicAdd(replay_reason_counts + 1, 1);
  if (boundary_risk) atomicAdd(replay_reason_counts + 2, 1);
  if (numerical_risk) atomicAdd(replay_reason_counts + 3, 1);
}

__device__ void replay_magic_exact_optimizer_target(
    DeviceMagicOptimizerState* output_state,
    const double* magic_r,
    const double* penalty_matrix,
    const double* penalty_root,
    const double* target_y0,
    double target_y_squared_norm,
    lapack312::Workspace* workspace,
    ExactEndpointObjective* shared_objective,
    SinglePenaltyGcvTranscriptEntry* transcript,
    int* transcript_count,
    int* transcript_overflow,
    int transcript_target,
    int* exact_counts,
    double initial_log_sp,
    int run_boundary,
    int q,
    int penalty_rank,
    int n) {
  const bool leader = threadIdx.x == 0;
  if (leader && transcript_count != nullptr) {
    transcript_count[transcript_target] = 0;
  }
  if (leader && transcript_overflow != nullptr) {
    transcript_overflow[transcript_target] = 0;
  }

  DeviceMagicOptimizerState state{};
  state.log_sp = initial_log_sp;
  state.score_reduction = 1e10;
  state.prior_gradient = CUDART_NAN;
  state.reported_gradient = CUDART_NAN;
  state.score_calls = 1;
  state.actual_calls = 1;
  state.current_exact = 1;
  state.selective_risk = 1;

  if (leader) atomicAdd(exact_counts + 4, 1);
  const ExactEndpointObjective initial = broadcast_exact_endpoint_objective(
    evaluate_exact_endpoint_objective(
      initial_log_sp, magic_r, penalty_matrix, penalty_root, target_y0,
      target_y_squared_norm, q, penalty_rank, n, workspace),
    shared_objective);
  if (!initial.valid) {
    if (leader) atomicAdd(exact_counts + 5, 1);
    state.invalid = 1;
    if (leader) *output_state = state;
    return;
  }
  state.current.rss = initial.rss;
  state.current.edf = initial.edf;
  state.current.score = initial.score;
  state.current.gradient = initial.gradient;
  state.current.hessian = initial.hessian;
  state.current.residual_sensitivity = initial.residual_sensitivity;
  state.current.valid = 1;
  state.minimum_score = initial.score;

  while (!state.converged && state.iteration < kMaxIterations) {
    state.iteration += 1;
    state.tries = 0;
    state.iteration_flat = 0;
    state.iteration_accepted = 0;
    state.iteration_origin_score = state.minimum_score;
    double proposed_step = state.use_steepest_descent ?
      state.steepest_step : state.newton_step;
    int step_source = static_cast<int>(state.use_steepest_descent ?
      SinglePenaltyGcvStepSource::SteepestDescent :
      SinglePenaltyGcvStepSource::Newton);

    if (state.iteration > 1) {
      if (leader) atomicAdd(exact_counts, 1);
      bool use_steepest_descent = state.use_steepest_descent != 0;
      for (int exact_try = 1; exact_try <= kMaxStepHalving; ++exact_try) {
        state.tries = exact_try;
        if (exact_try == 4 && !use_steepest_descent) {
          use_steepest_descent = true;
          proposed_step = state.steepest_step;
          step_source = static_cast<int>(
            SinglePenaltyGcvStepSource::SteepestDescentAfterNewtonRejection);
        }
        const double trial_log_sp = state.log_sp + proposed_step;
        const ExactEndpointObjective evaluated =
          broadcast_exact_endpoint_objective(
            evaluate_exact_endpoint_objective(
              trial_log_sp, magic_r, penalty_matrix, penalty_root,
              target_y0, target_y_squared_norm, q, penalty_rank, n,
              workspace),
            shared_objective);
        if (leader) atomicAdd(exact_counts + 1, 1);
        state.score_calls += 1;
        state.actual_calls += 1;
        if (!evaluated.valid) {
          if (leader) atomicAdd(exact_counts + 2, 1);
          state.invalid = 1;
          if (leader) *output_state = state;
          return;
        }

        const DeviceObjective trial =
          exact_endpoint_as_device_objective(evaluated);
        const bool accepted = trial.score < state.minimum_score;
        if (leader) {
          append_transcript(
            transcript, transcript_count, transcript_overflow,
            transcript_target,
            static_cast<int>(SinglePenaltyGcvTranscriptStage::Refinement),
            state.iteration, exact_try, state.log_sp, proposed_step,
            trial_log_sp, trial, accepted ? 1 : 0, step_source);
        }
        if (accepted) {
          if (leader) atomicAdd(exact_counts + 3, 1);
          state.score_reduction = state.minimum_score - trial.score;
          state.minimum_score = trial.score;
          state.log_sp = trial_log_sp;
          state.current = trial;
          state.iteration_accepted = 1;
          break;
        }
        proposed_step /= 2.0;
        if (exact_try == kMaxStepHalving - 1) proposed_step = 0.0;
      }
      state.use_steepest_descent = use_steepest_descent ? 1 : 0;
    }

    bool converged = false;
    if (state.iteration > 3) {
      converged = true;
      if (state.score_reduction >
          kConvergenceTolerance * (1.0 + state.minimum_score)) {
        converged = false;
      }
      const double gradient_norm = fabs(state.prior_gradient);
      if (gradient_norm > pow(kConvergenceTolerance, 1.0 / 3.0) *
                            (1.0 + fabs(state.minimum_score))) {
        converged = false;
      }
      if (state.tries == kMaxStepHalving) converged = true;
      if (state.iteration_flat) converged = true;
      if (converged) {
        state.reported_gradient = gradient_norm;
        if (state.tries == kMaxStepHalving) state.step_failed = 1;
      }
    }
    state.prior_gradient = state.current.gradient;
    state.use_steepest_descent = state.current.hessian < 0.0 ? 1 : 0;
    if (!state.use_steepest_descent) {
      state.newton_step = -state.current.gradient / state.current.hessian;
      if (fabs(state.newton_step) > kMaxNewtonStep) {
        state.newton_step = copysign(kMaxNewtonStep, state.newton_step);
      }
    }
    state.steepest_step = state.current.gradient == 0.0 ?
      0.0 : -state.current.gradient / fabs(state.current.gradient);
    if (leader) {
      append_transcript(
        transcript, transcript_count, transcript_overflow,
        transcript_target,
        static_cast<int>(SinglePenaltyGcvTranscriptStage::IterationState),
        state.iteration, 0, state.log_sp,
        state.use_steepest_descent ? state.steepest_step : state.newton_step,
        state.log_sp, state.current, 1,
        static_cast<int>(state.use_steepest_descent ?
          SinglePenaltyGcvStepSource::SteepestDescent :
          SinglePenaltyGcvStepSource::Newton));
    }
    state.converged = converged ? 1 : 0;
  }
  if (run_boundary != 0 && state.converged && !state.invalid) {
    state.hessian_positive = state.use_steepest_descent ? 0 : 1;
    state.pre_boundary_log_sp = state.log_sp;
    state.probe_direction = state.current.gradient < 0.0 ? 1.0 : -1.0;
    for (int probe = 1; probe <= kMaxBoundaryProbes; ++probe) {
      const double proposed_step =
        state.probe_direction * kBoundaryProbeStep;
      const double trial_log_sp = state.log_sp + proposed_step;
      if (leader) {
        atomicAdd(exact_counts, 1);
        atomicAdd(exact_counts + 1, 1);
      }
      const ExactEndpointObjective evaluated =
        broadcast_exact_endpoint_objective(
          evaluate_exact_endpoint_objective(
            trial_log_sp, magic_r, penalty_matrix, penalty_root,
            target_y0, target_y_squared_norm, q, penalty_rank, n,
            workspace),
          shared_objective);
      state.actual_calls += 1;
      state.boundary_probe_count += 1;
      const DeviceObjective trial =
        exact_endpoint_as_device_objective(evaluated);
      if (!evaluated.valid) {
        if (leader) atomicAdd(exact_counts + 2, 1);
        state.invalid = 1;
      }
      const bool accepted = evaluated.valid &&
        trial.score < state.minimum_score;
      if (leader) {
        append_transcript(
          transcript, transcript_count, transcript_overflow,
          transcript_target,
          static_cast<int>(SinglePenaltyGcvTranscriptStage::BoundaryProbe),
          state.iteration, probe, state.log_sp, proposed_step, trial_log_sp,
          trial, accepted ? 1 : 0,
          static_cast<int>(SinglePenaltyGcvStepSource::InfinityProbe));
      }
      if (!accepted) break;
      if (leader) atomicAdd(exact_counts + 3, 1);
      state.log_sp = trial_log_sp;
      state.current = trial;
      state.current_exact = 1;
      state.minimum_score = trial.score;
      state.boundary_accepted_count += 1;
    }
  }
  if (leader) *output_state = state;
}

__global__ void replay_magic_exact_optimizer_kernel(
    DeviceMagicOptimizerState* states,
    const int* replay_flags,
    const double* magic_r,
    const double* penalty_matrix,
    const double* penalty_root,
    const double* y0,
    const double* y_squared_norm,
    lapack312::Workspace* workspaces,
    SinglePenaltyGcvTranscriptEntry* transcript,
    int* transcript_count,
    int* transcript_overflow,
    int* exact_counts,
    double initial_log_sp,
    int run_boundary,
    int q,
    int penalty_rank,
    int n,
    int target_count) {
  const int target = blockIdx.x;
  if (target >= target_count || replay_flags[target] == 0) return;
  __shared__ ExactEndpointObjective shared_objective;
  replay_magic_exact_optimizer_target(
    states + target, magic_r, penalty_matrix, penalty_root,
    y0 + static_cast<std::size_t>(q) * target, y_squared_norm[target],
    workspaces + target, &shared_objective, transcript, transcript_count,
    transcript_overflow, target, exact_counts, initial_log_sp, run_boundary,
    q, penalty_rank, n);
}

struct FusedExactReplayTarget {
  std::size_t qr_offset = 0;
  std::size_t tau_offset = 0;
  std::size_t magic_r_offset = 0;
  std::size_t penalty_matrix_offset = 0;
  std::size_t penalty_root_offset = 0;
  std::size_t y_offset = 0;
  std::size_t y0_offset = 0;
  double initial_log_sp = 0.0;
  int setup_index = 0;
  int target_index = 0;
  int n = 0;
  int q = 0;
  int penalty_rank = 0;
};

__global__ void project_fused_exact_replay_targets_kernel(
    const FusedExactReplayTarget* metadata,
    const double* qr_packed,
    const double* tau,
    const double* responses,
    double* projection_work,
    double* y0,
    double* y_squared_norm,
    int target_count) {
  const int target = blockIdx.x * blockDim.x + threadIdx.x;
  if (target >= target_count) return;
  const FusedExactReplayTarget& item = metadata[target];
  const double* target_y = responses + item.y_offset;
  double* target_work = projection_work + item.y_offset;
  const double* target_qr = qr_packed + item.qr_offset;
  const double* target_tau = tau + item.tau_offset;

  double squared_norm = 0.0;
  for (int row = 0; row < item.n; ++row) {
    const double value = target_y[row];
    squared_norm = rounded_multiply_add(squared_norm, value, value);
    target_work[row] = value;
  }
  y_squared_norm[target] = squared_norm;

  for (int reflector = 0; reflector < item.q; ++reflector) {
    double dot = target_work[reflector];
    const double* vector =
      target_qr + static_cast<std::size_t>(item.n) * reflector;
    for (int row = reflector + 1; row < item.n; ++row) {
      dot = rounded_multiply_add(dot, vector[row], target_work[row]);
    }
    const double scale = __dmul_rn(-target_tau[reflector], dot);
    target_work[reflector] = __dadd_rn(target_work[reflector], scale);
    for (int row = reflector + 1; row < item.n; ++row) {
      target_work[row] = rounded_multiply_add(
        target_work[row], vector[row], scale);
    }
  }
  for (int row = 0; row < item.q; ++row) {
    y0[item.y0_offset + row] = target_work[row];
  }
}

__global__ void replay_fused_exact_optimizer_kernel(
    const FusedExactReplayTarget* metadata,
    const double* magic_r,
    const double* penalty_matrix,
    const double* penalty_root,
    const double* y0,
    const double* y_squared_norm,
    lapack312::Workspace* workspaces,
    DeviceMagicOptimizerState* states,
    int* setup_exact_counts,
    int first_target,
    int batch_target_count) {
  const int target = first_target + blockIdx.x;
  if (blockIdx.x >= batch_target_count) return;
  const FusedExactReplayTarget& item = metadata[target];
  __shared__ ExactEndpointObjective shared_objective;
  replay_magic_exact_optimizer_target(
    states + target, magic_r + item.magic_r_offset,
    penalty_matrix + item.penalty_matrix_offset,
    penalty_root + item.penalty_root_offset, y0 + item.y0_offset,
    y_squared_norm[target], workspaces + target, &shared_objective, nullptr,
    nullptr, nullptr, target,
    setup_exact_counts +
      static_cast<std::size_t>(item.setup_index) * kExactEndpointCounterCount,
    item.initial_log_sp, 1, item.q, item.penalty_rank, item.n);
}

__global__ void finalize_magic_optimizer_kernel(
    const DeviceMagicOptimizerState* states,
    const int* output_flags,
    const int* transcript_overflow,
    SinglePenaltyGcvOptimizerResult* output,
    int target_count) {
  const int target = blockIdx.x * blockDim.x + threadIdx.x;
  if (target >= target_count ||
      (output_flags != nullptr && output_flags[target] == 0)) return;
  const DeviceMagicOptimizerState& state = states[target];
  SinglePenaltyGcvOptimizerResult result;
  if (state.invalid || !state.converged) {
    result.termination = static_cast<int>(state.invalid ?
      SinglePenaltyGcvTermination::NonfiniteObjective :
      SinglePenaltyGcvTermination::IterationLimit);
    result.iteration_count = state.iteration;
    result.score_call_count = state.score_calls;
    result.actual_objective_call_count = state.actual_calls;
    output[target] = result;
    return;
  }
  result.sp = glibc235::exp_fma_rn(state.log_sp);
  result.log_sp = state.log_sp;
  result.rss = state.current.rss;
  result.edf = state.current.edf;
  result.score = state.current.score;
  result.gradient = state.current.gradient;
  result.hessian = state.current.hessian;
  result.reported_rms_gradient = state.reported_gradient;
  result.pre_boundary_log_sp = state.pre_boundary_log_sp;
  result.iteration_count = state.iteration;
  result.score_call_count = state.score_calls;
  result.actual_objective_call_count = state.actual_calls;
  result.fully_converged =
    (state.step_failed || state.unresolved_newton_seen) ? 0 : 1;
  result.hessian_positive_definite = state.hessian_positive;
  result.boundary_probe_count = state.boundary_probe_count;
  result.boundary_accepted_count = state.boundary_accepted_count;
  result.termination = static_cast<int>(
    state.unresolved_newton_seen ?
      SinglePenaltyGcvTermination::FlatObjective :
      (state.step_failed ?
        SinglePenaltyGcvTermination::StepHalvingExhausted :
        SinglePenaltyGcvTermination::ScoreAndGradient));
  if (transcript_overflow != nullptr && transcript_overflow[target] != 0) {
    result.termination = static_cast<int>(
      SinglePenaltyGcvTermination::TranscriptOverflow);
    result.fully_converged = 0;
  }
  output[target] = result;
}

__global__ void optimize_targets_kernel(
    const double* eigenvalues,
    const double* spectral_projection,
    const double* y_squared_norm,
    SpectralEndpointRefinementState* endpoint_states,
    SinglePenaltyGcvOptimizerResult* output,
    SinglePenaltyGcvTranscriptEntry* transcript,
    int* transcript_count,
    int* transcript_overflow,
    double initial_sp,
    int p,
    int penalty_rank,
    int n,
    int target_count) {
  const int target = blockIdx.x * blockDim.x + threadIdx.x;
  if (target >= target_count) return;
  if (transcript_count != nullptr) transcript_count[target] = 0;
  if (transcript_overflow != nullptr) transcript_overflow[target] = 0;

  SinglePenaltyGcvOptimizerResult result;
  double log_sp = log(initial_sp);
  DeviceObjective current = evaluate_objective(
    log_sp, eigenvalues, spectral_projection, y_squared_norm[target],
    p, target, n);
  if (!current.valid) {
    result.termination = static_cast<int>(
      SinglePenaltyGcvTermination::NonfiniteObjective);
    output[target] = result;
    return;
  }
  double minimum_score = current.score;
  double score_reduction = 1e10;
  bool use_steepest_descent = false;
  bool step_failed = false;
  double prior_gradient = CUDART_NAN;
  double newton_step = 0.0;
  double steepest_step = 0.0;
  double reported_gradient = CUDART_NAN;
  int iteration = 0;
  int score_calls = 1;
  int actual_calls = 1;
  bool converged = false;
  bool flat_objective = false;

  while (!converged && iteration < kMaxIterations) {
    ++iteration;
    bool try_step = iteration > 1;
    int tries = 0;
    double proposed_step = use_steepest_descent ?
      steepest_step : newton_step;
    int step_source = static_cast<int>(use_steepest_descent ?
      SinglePenaltyGcvStepSource::SteepestDescent :
      SinglePenaltyGcvStepSource::Newton);
    bool iteration_flat = false;
    while (try_step) {
      ++tries;
      if (tries == 4 && !use_steepest_descent) {
        use_steepest_descent = true;
        proposed_step = steepest_step;
        step_source = static_cast<int>(
          SinglePenaltyGcvStepSource::
            SteepestDescentAfterNewtonRejection);
      }
      double trial_log_sp = log_sp + proposed_step;
      DeviceObjective trial = evaluate_objective(
        trial_log_sp, eigenvalues, spectral_projection,
        y_squared_norm[target], p, target, n);
      ++score_calls;
      ++actual_calls;
      const double predicted_change =
        current.gradient * proposed_step +
        0.5 * current.hessian * proposed_step * proposed_step;
      const double resolution = kFlatObjectiveResolutionMultiplier *
        kDoubleEpsilon * (1.0 + fabs(minimum_score));
      const bool unresolved_newton_step =
        step_source == static_cast<int>(
          SinglePenaltyGcvStepSource::Newton) &&
        isfinite(predicted_change) && predicted_change <= 0.0 &&
        fabs(predicted_change) <= resolution;
      bool accepted = trial.valid && trial.score < minimum_score;
      const double endpoint_resolution = kEndpointResolutionMultiplier *
        kDoubleEpsilon * (1.0 + fabs(minimum_score));
      const bool endpoint_risk =
        step_source == static_cast<int>(
          SinglePenaltyGcvStepSource::Newton) &&
        isfinite(predicted_change) && predicted_change <= 0.0 &&
        fabs(predicted_change) <= endpoint_resolution &&
        isfinite(current.residual_sensitivity) &&
        fabs(proposed_step) * current.residual_sensitivity >
          kEndpointResidualRiskLimit;
      if (endpoint_risk) {
        SpectralEndpointRefinementState state{};
        state.active = 1;
        endpoint_states[target] = state;
      }
      append_transcript(
        transcript, transcript_count, transcript_overflow, target,
        static_cast<int>(SinglePenaltyGcvTranscriptStage::Refinement),
        iteration, tries, log_sp, proposed_step, trial_log_sp, trial,
        accepted ? 1 : 0, step_source);
      if (accepted) {
        try_step = false;
        score_reduction = minimum_score - trial.score;
        minimum_score = trial.score;
        log_sp = trial_log_sp;
        current = trial;
      } else {
        if (unresolved_newton_step) {
          try_step = false;
          iteration_flat = true;
          flat_objective = true;
        } else {
          proposed_step /= 2.0;
        }
      }
      if (tries == kMaxStepHalving - 1 && try_step) proposed_step = 0.0;
      if (tries == kMaxStepHalving) try_step = false;
    }

    if (iteration > 3) {
      converged = true;
      if (score_reduction >
          kConvergenceTolerance * (1.0 + minimum_score)) {
        converged = false;
      }
      const double gradient_norm = fabs(prior_gradient);
      if (gradient_norm > pow(kConvergenceTolerance, 1.0 / 3.0) *
                            (1.0 + fabs(minimum_score))) {
        converged = false;
      }
      if (tries == kMaxStepHalving) converged = true;
      if (iteration_flat) converged = true;
      if (converged) {
        reported_gradient = gradient_norm;
        if (tries == kMaxStepHalving) step_failed = true;
      }
    }

    prior_gradient = current.gradient;
    use_steepest_descent = current.hessian < 0.0;
    if (!use_steepest_descent) {
      newton_step = -current.gradient / current.hessian;
      if (fabs(newton_step) > kMaxNewtonStep) {
        newton_step = copysign(kMaxNewtonStep, newton_step);
      }
    }
    steepest_step = current.gradient == 0.0 ?
      0.0 : -current.gradient / fabs(current.gradient);
    append_transcript(
      transcript, transcript_count, transcript_overflow, target,
      static_cast<int>(SinglePenaltyGcvTranscriptStage::IterationState),
      iteration, 0, log_sp,
      use_steepest_descent ? steepest_step : newton_step,
      log_sp, current, 1,
      static_cast<int>(use_steepest_descent ?
        SinglePenaltyGcvStepSource::SteepestDescent :
        SinglePenaltyGcvStepSource::Newton));
  }

  if (!converged) {
    result.termination = static_cast<int>(
      SinglePenaltyGcvTermination::IterationLimit);
    result.iteration_count = iteration;
    result.score_call_count = score_calls;
    result.actual_objective_call_count = actual_calls;
    output[target] = result;
    return;
  }

  const int hessian_positive = use_steepest_descent ? 0 : 1;
  const double pre_boundary_log_sp = log_sp;
  const double probe_direction = current.gradient < 0.0 ? 1.0 : -1.0;
  int boundary_probe_count = 0;
  int boundary_accepted_count = 0;
  for (int probe = 1; probe <= kMaxBoundaryProbes; ++probe) {
    ++boundary_probe_count;
    const double trial_log_sp = log_sp +
      probe_direction * kBoundaryProbeStep;
    const DeviceObjective trial = evaluate_objective(
      trial_log_sp, eigenvalues, spectral_projection,
      y_squared_norm[target], p, target, n);
    ++actual_calls;
    const bool accepted = trial.valid && trial.score < minimum_score;
    append_transcript(
      transcript, transcript_count, transcript_overflow, target,
      static_cast<int>(SinglePenaltyGcvTranscriptStage::BoundaryProbe),
      iteration, probe, log_sp, probe_direction * kBoundaryProbeStep,
      trial_log_sp, trial, accepted ? 1 : 0,
      static_cast<int>(SinglePenaltyGcvStepSource::InfinityProbe));
    if (!accepted) break;
    log_sp = trial_log_sp;
    current = trial;
    minimum_score = trial.score;
    ++boundary_accepted_count;
  }

  result.sp = glibc235::exp_fma_rn(log_sp);
  result.log_sp = log_sp;
  result.rss = current.rss;
  result.edf = current.edf;
  result.score = current.score;
  result.gradient = current.gradient;
  result.hessian = current.hessian;
  result.reported_rms_gradient = reported_gradient;
  result.pre_boundary_log_sp = pre_boundary_log_sp;
  result.iteration_count = iteration;
  result.score_call_count = score_calls;
  result.actual_objective_call_count = actual_calls;
  result.fully_converged = (step_failed || flat_objective) ? 0 : 1;
  result.hessian_positive_definite = hessian_positive;
  result.boundary_probe_count = boundary_probe_count;
  result.boundary_accepted_count = boundary_accepted_count;
  result.termination = static_cast<int>(step_failed ?
    SinglePenaltyGcvTermination::StepHalvingExhausted :
    (flat_objective ? SinglePenaltyGcvTermination::FlatObjective :
      SinglePenaltyGcvTermination::ScoreAndGradient));
  if (!isfinite(result.sp) || !isfinite(result.score) ||
      !isfinite(result.edf) || !isfinite(result.rss)) {
    result.termination = static_cast<int>(
      SinglePenaltyGcvTermination::NonfiniteObjective);
    result.fully_converged = 0;
  }
  if (transcript_overflow != nullptr && transcript_overflow[target] != 0) {
    result.termination = static_cast<int>(
      SinglePenaltyGcvTermination::TranscriptOverflow);
    result.fully_converged = 0;
  }
  output[target] = result;
}

template <typename T>
void upload_async(DeviceBuffer<T>* destination,
                  const T* source,
                  std::size_t count,
                  cudaStream_t stream,
                  SinglePenaltyGcvCudaDiagnostics* diagnostics) {
  if (count == 0) return;
  check_cuda(cudaMemcpyAsync(
               destination->get(), source, count * sizeof(T),
               cudaMemcpyHostToDevice, stream),
             "upload single-penalty GCV input");
  diagnostics->h2d_copy_count += 1;
  diagnostics->h2d_bytes += count * sizeof(T);
}

template <typename T>
void download_async(T* destination,
                    const DeviceBuffer<T>& source,
                    std::size_t count,
                    cudaStream_t stream) {
  if (count == 0) return;
  check_cuda(cudaMemcpyAsync(
               destination, source.get(), count * sizeof(T),
               cudaMemcpyDeviceToHost, stream),
             "download single-penalty GCV output");
}

}  // namespace

const char* single_penalty_gcv_termination_name(int value) {
  switch (static_cast<SinglePenaltyGcvTermination>(value)) {
    case SinglePenaltyGcvTermination::ScoreAndGradient:
      return "score_and_gradient";
    case SinglePenaltyGcvTermination::StepHalvingExhausted:
      return "step_halving_exhausted";
    case SinglePenaltyGcvTermination::FlatObjective:
      return "flat_objective";
    case SinglePenaltyGcvTermination::IterationLimit:
      return "iteration_limit";
    case SinglePenaltyGcvTermination::NonfiniteObjective:
      return "nonfinite_objective";
    case SinglePenaltyGcvTermination::TranscriptOverflow:
      return "transcript_overflow";
  }
  return "unknown";
}

const char* single_penalty_gcv_transcript_stage_name(int value) {
  switch (static_cast<SinglePenaltyGcvTranscriptStage>(value)) {
    case SinglePenaltyGcvTranscriptStage::Refinement:
      return "refinement";
    case SinglePenaltyGcvTranscriptStage::IterationState:
      return "iteration_state";
    case SinglePenaltyGcvTranscriptStage::BoundaryProbe:
      return "boundary_probe";
  }
  return "unknown";
}

const char* single_penalty_gcv_step_source_name(int value) {
  switch (static_cast<SinglePenaltyGcvStepSource>(value)) {
    case SinglePenaltyGcvStepSource::Newton: return "newton";
    case SinglePenaltyGcvStepSource::SteepestDescent:
      return "steepest_descent";
    case SinglePenaltyGcvStepSource::SteepestDescentAfterNewtonRejection:
      return "steepest_descent_after_newton_rejection";
    case SinglePenaltyGcvStepSource::InfinityProbe:
      return "mgcv_infinity_probe";
  }
  return "unknown";
}

SinglePenaltyMrootCudaResult single_penalty_mroot_cuda(
    const double* penalty_matrix,
    int coefficient_dim,
    int penalty_rank,
    const std::vector<double>& log_sp) {
  if (penalty_matrix == nullptr) {
    throw std::runtime_error("single-penalty mroot matrix is null");
  }
  if (coefficient_dim <= 0 ||
      coefficient_dim > kMaximumMrootCoefficientDim ||
      penalty_rank <= 0 || penalty_rank > coefficient_dim) {
    throw std::runtime_error("single-penalty mroot dimensions are invalid");
  }
  if (log_sp.empty()) {
    throw std::runtime_error("single-penalty mroot candidates are empty");
  }
  for (double value : log_sp) {
    if (!std::isfinite(value) || !std::isfinite(std::exp(value)) ||
        std::exp(value) <= 0.0) {
      throw std::runtime_error(
        "single-penalty mroot log smoothing parameter is invalid");
    }
  }

  SinglePenaltyMrootCudaResult result;
  result.coefficient_dim = coefficient_dim;
  result.requested_rank = penalty_rank;
  result.candidate_count = static_cast<int>(log_sp.size());
  const std::size_t square =
    static_cast<std::size_t>(coefficient_dim) * coefficient_dim;
  const std::size_t matrix_count = square * log_sp.size();
  const std::size_t root_count =
    static_cast<std::size_t>(coefficient_dim) * penalty_rank * log_sp.size();
  const std::size_t work_count =
    static_cast<std::size_t>(2 * coefficient_dim) * log_sp.size();
  const std::size_t pivot_count =
    static_cast<std::size_t>(coefficient_dim) * log_sp.size();

  SinglePenaltyGcvCudaDiagnostics allocation_diagnostics;
  Stream stream;
  DeviceAllocationStreamScope allocation_scope(stream.get());
  DeviceBuffer<double> d_penalty_matrix;
  DeviceBuffer<double> d_log_sp;
  DeviceBuffer<double> d_factor;
  DeviceBuffer<double> d_roots;
  DeviceBuffer<double> d_work;
  DeviceBuffer<int> d_ranks;
  DeviceBuffer<int> d_pivots;
  d_penalty_matrix.allocate(square, &allocation_diagnostics);
  d_log_sp.allocate(log_sp.size(), &allocation_diagnostics);
  d_factor.allocate(matrix_count, &allocation_diagnostics);
  d_roots.allocate(root_count, &allocation_diagnostics);
  d_work.allocate(work_count, &allocation_diagnostics);
  d_ranks.allocate(log_sp.size(), &allocation_diagnostics);
  d_pivots.allocate(pivot_count, &allocation_diagnostics);
  upload_async(
    &d_penalty_matrix, penalty_matrix, square, stream.get(),
    &allocation_diagnostics);
  upload_async(
    &d_log_sp, log_sp.data(), log_sp.size(), stream.get(),
    &allocation_diagnostics);

  constexpr int block_size = 64;
  const int block_count =
    (result.candidate_count + block_size - 1) / block_size;
  single_penalty_mroot_kernel<<<block_count, block_size, 0, stream.get()>>>(
    d_penalty_matrix.get(), d_log_sp.get(), d_factor.get(), d_roots.get(),
    d_work.get(), d_ranks.get(), d_pivots.get(), coefficient_dim,
    penalty_rank, result.candidate_count);
  check_cuda(cudaGetLastError(), "launch single-penalty dynamic mroot");

  result.roots.resize(root_count);
  result.ranks.resize(log_sp.size());
  result.pivots.resize(pivot_count);
  download_async(result.roots.data(), d_roots, root_count, stream.get());
  download_async(result.ranks.data(), d_ranks, log_sp.size(), stream.get());
  download_async(result.pivots.data(), d_pivots, pivot_count, stream.get());
  check_cuda(cudaStreamSynchronize(stream.get()),
             "synchronize single-penalty dynamic mroot");
  return result;
}

SinglePenaltyGcvCudaResult single_penalty_gcv_cuda(
    const double* X,
    const double* Y,
    const double* rhs_transform,
    const double* eigenvalues,
    const double* magic_qr_packed,
    const double* magic_tau,
    const double* magic_r,
    const double* magic_penalty_root,
    const double* magic_penalty_matrix,
    const int* target_ids,
    int n,
    int coefficient_dim,
    int target_count,
    int penalty_rank,
    double initial_sp,
    const std::vector<double>& sp_grid,
    bool materialize_grid,
    bool keep_transcript,
    bool defer_exact_replay) {
  if (X == nullptr || Y == nullptr || rhs_transform == nullptr ||
      eigenvalues == nullptr || magic_qr_packed == nullptr ||
      magic_tau == nullptr || magic_r == nullptr ||
      magic_penalty_root == nullptr || magic_penalty_matrix == nullptr ||
      target_ids == nullptr) {
    throw std::runtime_error("single-penalty GCV input pointer is null");
  }
  if (n <= coefficient_dim || coefficient_dim <= 0 ||
      coefficient_dim > lapack312::kMaximumColumns || target_count <= 0) {
    throw std::runtime_error("single-penalty GCV dimensions are invalid");
  }
  if (target_count > kMagicEigensolverBatch) {
    throw std::runtime_error(
      "single-penalty GCV target batch exceeds the fixed eigensolver shape");
  }
  if (penalty_rank <= 0 || penalty_rank > coefficient_dim) {
    throw std::runtime_error("single-penalty GCV penalty rank is invalid");
  }
  if (!std::isfinite(initial_sp) || initial_sp <= 0.0) {
    throw std::runtime_error("single-penalty GCV initial sp is invalid");
  }
  if (materialize_grid && sp_grid.empty()) {
    throw std::runtime_error(
      "single-penalty GCV materialized grid must be non-empty");
  }
  if (defer_exact_replay && (materialize_grid || keep_transcript)) {
    throw std::runtime_error(
      "deferred exact replay requires compact targets without transcripts");
  }
  for (double value : sp_grid) {
    if (!std::isfinite(value) || value <= 0.0) {
      throw std::runtime_error("single-penalty GCV grid contains invalid sp");
    }
  }
  std::vector<double> grid_log_sp(sp_grid.size());
  std::transform(
    sp_grid.begin(), sp_grid.end(), grid_log_sp.begin(),
    [](double value) { return std::log(value); });
  std::vector<int> seen_target_ids(
    static_cast<std::size_t>(kMagicEigensolverBatch), 0);
  for (int target = 0; target < target_count; ++target) {
    const int slot = target_ids[target] - 1;
    if (slot < 0 || slot >= kMagicEigensolverBatch ||
        seen_target_ids[static_cast<std::size_t>(slot)] != 0) {
      throw std::runtime_error(
        "single-penalty GCV target ids do not define unique solver slots");
    }
    seen_target_ids[static_cast<std::size_t>(slot)] = 1;
  }

  const auto host_begin = std::chrono::steady_clock::now();
  SinglePenaltyGcvCudaResult result;
  result.schema_version = kSinglePenaltyGcvCudaSchemaVersion;
  result.n = n;
  result.coefficient_dim = coefficient_dim;
  result.target_count = target_count;
  result.candidate_count = static_cast<int>(sp_grid.size());
  SinglePenaltyGcvCudaDiagnostics& diagnostics = result.diagnostics;
  diagnostics.schema_version = kSinglePenaltyGcvCudaSchemaVersion;
  diagnostics.sp_selection_backend_executed = "cuda";
  diagnostics.gcv_score_backend_executed = "cuda";
  diagnostics.optimizer_backend_executed =
    "cuda-spectral-risk-gated-exact-replay";
  diagnostics.exact_replay_backend_executed =
    "cuda-dpstf2-lapack-3.12-dgesdd";
  diagnostics.n = n;
  diagnostics.coefficient_dim = coefficient_dim;
  diagnostics.target_count = target_count;
  diagnostics.candidate_count = result.candidate_count;
  diagnostics.penalty_rank = penalty_rank;
  diagnostics.penalty_nullity = coefficient_dim - penalty_rank;
  diagnostics.cuda_gcv_batches = 1;
  diagnostics.cuda_gcv_targets = target_count;
  diagnostics.score_matrix_materialized = materialize_grid;
  diagnostics.transcript_materialized = keep_transcript;

  check_cuda(cudaGetDevice(&diagnostics.device_id),
             "query single-penalty GCV CUDA device");
  cudaDeviceProp properties;
  check_cuda(cudaGetDeviceProperties(&properties, diagnostics.device_id),
             "query single-penalty GCV GPU properties");
  diagnostics.gpu_name = properties.name;

  Stream stream;
  DeviceAllocationStreamScope allocation_scope(stream.get());
  BlasHandle blas(stream.get());
  DeviceBuffer<unsigned char> blas_workspace;
  blas_workspace.allocate(kFixedSpCublasWorkspaceBytes, &diagnostics);
  check_cublas(cublasSetWorkspace(
                 blas.get(), blas_workspace.get(),
                 kFixedSpCublasWorkspaceBytes),
               "install single-penalty GCV cuBLAS workspace");
  diagnostics.cublas_pedantic_math = true;
  diagnostics.cublas_atomics_not_allowed = true;
  diagnostics.cublas_user_workspace_installed = true;

  const std::size_t x_count = static_cast<std::size_t>(n) * coefficient_dim;
  const std::size_t y_count = static_cast<std::size_t>(n) * target_count;
  const std::size_t transform_count =
    static_cast<std::size_t>(coefficient_dim) * coefficient_dim;
  const std::size_t projected_count =
    static_cast<std::size_t>(coefficient_dim) * target_count;
  const std::size_t square_count =
    static_cast<std::size_t>(coefficient_dim) * coefficient_dim;
  const std::size_t magic_root_count =
    static_cast<std::size_t>(coefficient_dim) * penalty_rank;
  const std::size_t grid_count =
    static_cast<std::size_t>(target_count) * sp_grid.size();
  const int exact_augmented_rows = coefficient_dim + penalty_rank;
  const int exact_endpoint_threads =
    coefficient_dim > FASTKPC_LAPACK_SMALL_SMLSIZ ? 64 : kExactWarp;
  const std::size_t exact_augmented_count =
    static_cast<std::size_t>(exact_augmented_rows) * coefficient_dim *
      sp_grid.size();
  const std::size_t exact_candidate_root_count =
    magic_root_count * sp_grid.size();
  const std::size_t exact_mroot_work_count =
    static_cast<std::size_t>(2 * coefficient_dim) * sp_grid.size();
  const std::size_t transcript_capacity = keep_transcript ?
    static_cast<std::size_t>(target_count) *
      kSinglePenaltyGcvTranscriptCapacity : 0;
  const std::size_t exact_workspace_count = defer_exact_replay ? 0 :
    static_cast<std::size_t>(target_count);

  DeviceBuffer<double> d_X;
  DeviceBuffer<double> d_Y;
  DeviceBuffer<double> d_transform;
  DeviceBuffer<double> d_eigenvalues;
  DeviceBuffer<double> d_projected_rhs;
  DeviceBuffer<double> d_spectral_projection;
  DeviceBuffer<double> d_y_squared_norm;
  DeviceBuffer<double> d_magic_qr_packed;
  DeviceBuffer<double> d_magic_tau;
  DeviceBuffer<double> d_magic_r;
  DeviceBuffer<double> d_magic_penalty_root;
  DeviceBuffer<double> d_magic_penalty_matrix;
  DeviceBuffer<double> d_magic_y0;
  DeviceBuffer<double> d_magic_qt_y_work;
  DeviceBuffer<double> d_magic_y_squared_norm;
  DeviceBuffer<int> d_magic_replay_flags;
  DeviceBuffer<DeviceMagicOptimizerState> d_magic_states;
  DeviceBuffer<SpectralEndpointRefinementState> d_spectral_endpoint_states;
  DeviceBuffer<lapack312::Workspace> d_exact_endpoint_workspaces;
  DeviceBuffer<int> d_exact_endpoint_counts;
  DeviceBuffer<double> d_exact_grid_log_sp;
  DeviceBuffer<double> d_exact_factor;
  DeviceBuffer<double> d_exact_roots;
  DeviceBuffer<double> d_exact_mroot_work;
  DeviceBuffer<int> d_exact_root_ranks;
  DeviceBuffer<int> d_exact_pivots;
  DeviceBuffer<lapack312::Workspace> d_exact_grid_workspaces;
  DeviceBuffer<double> d_exact_singular_vectors;
  DeviceBuffer<double> d_exact_singular_values;
  DeviceBuffer<int> d_exact_svd_info;
  DeviceBuffer<SinglePenaltyGcvGridCell> d_grid;
  DeviceBuffer<SinglePenaltyGcvOptimizerResult> d_targets;
  DeviceBuffer<SinglePenaltyGcvTranscriptEntry> d_transcript;
  DeviceBuffer<int> d_transcript_counts;
  DeviceBuffer<int> d_transcript_overflow;
  d_X.allocate(x_count, &diagnostics);
  d_Y.allocate(y_count, &diagnostics);
  d_transform.allocate(transform_count, &diagnostics);
  d_eigenvalues.allocate(coefficient_dim, &diagnostics);
  d_projected_rhs.allocate(projected_count, &diagnostics);
  d_spectral_projection.allocate(projected_count, &diagnostics);
  d_y_squared_norm.allocate(target_count, &diagnostics);
  d_magic_qr_packed.allocate(x_count, &diagnostics);
  d_magic_tau.allocate(coefficient_dim, &diagnostics);
  d_magic_r.allocate(square_count, &diagnostics);
  d_magic_penalty_root.allocate(magic_root_count, &diagnostics);
  d_magic_penalty_matrix.allocate(square_count, &diagnostics);
  d_magic_y0.allocate(projected_count, &diagnostics);
  d_magic_qt_y_work.allocate(y_count, &diagnostics);
  d_magic_y_squared_norm.allocate(target_count, &diagnostics);
  d_magic_replay_flags.allocate(target_count, &diagnostics);
  d_magic_states.allocate(
    defer_exact_replay ? 0 : static_cast<std::size_t>(target_count),
    &diagnostics);
  d_spectral_endpoint_states.allocate(target_count, &diagnostics);
  d_exact_endpoint_workspaces.allocate(exact_workspace_count, &diagnostics);
  d_exact_endpoint_counts.allocate(kExactEndpointCounterCount, &diagnostics);
  if (materialize_grid) {
    d_exact_grid_log_sp.allocate(sp_grid.size(), &diagnostics);
    d_exact_factor.allocate(square_count * sp_grid.size(), &diagnostics);
    d_exact_roots.allocate(exact_candidate_root_count, &diagnostics);
    d_exact_mroot_work.allocate(exact_mroot_work_count, &diagnostics);
    d_exact_root_ranks.allocate(sp_grid.size(), &diagnostics);
    d_exact_pivots.allocate(
      static_cast<std::size_t>(coefficient_dim) * sp_grid.size(),
      &diagnostics);
    d_exact_grid_workspaces.allocate(sp_grid.size(), &diagnostics);
    d_exact_singular_vectors.allocate(exact_augmented_count, &diagnostics);
    d_exact_singular_values.allocate(
      static_cast<std::size_t>(coefficient_dim) * sp_grid.size(),
      &diagnostics);
    d_exact_svd_info.allocate(sp_grid.size(), &diagnostics);
    d_grid.allocate(grid_count, &diagnostics);
  }
  d_targets.allocate(target_count, &diagnostics);
  if (keep_transcript) {
    d_transcript.allocate(transcript_capacity, &diagnostics);
    d_transcript_counts.allocate(target_count, &diagnostics);
    d_transcript_overflow.allocate(target_count, &diagnostics);
  }

  Event stage_start;
  Event upload_done;
  Event projection_done;
  Event grid_done;
  Event optimizer_done;
  Event d2h_done;
  check_cuda(cudaEventRecord(stage_start.get(), stream.get()),
             "record single-penalty GCV start");
  upload_async(&d_X, X, x_count, stream.get(), &diagnostics);
  upload_async(&d_Y, Y, y_count, stream.get(), &diagnostics);
  upload_async(&d_transform, rhs_transform, transform_count,
               stream.get(), &diagnostics);
  upload_async(&d_eigenvalues, eigenvalues, coefficient_dim,
               stream.get(), &diagnostics);
  upload_async(
    &d_magic_qr_packed, magic_qr_packed, x_count, stream.get(),
    &diagnostics);
  upload_async(
    &d_magic_tau, magic_tau, coefficient_dim, stream.get(), &diagnostics);
  upload_async(&d_magic_r, magic_r, square_count,
               stream.get(), &diagnostics);
  upload_async(&d_magic_penalty_root, magic_penalty_root, magic_root_count,
               stream.get(), &diagnostics);
  upload_async(&d_magic_penalty_matrix, magic_penalty_matrix, square_count,
               stream.get(), &diagnostics);
  if (materialize_grid) {
    upload_async(&d_exact_grid_log_sp, grid_log_sp.data(), grid_log_sp.size(),
                 stream.get(), &diagnostics);
  }
  check_cuda(cudaEventRecord(upload_done.get(), stream.get()),
             "record single-penalty GCV upload");

  const double one = 1.0;
  const double zero = 0.0;
  check_cublas(cublasDgemm(
                 blas.get(), CUBLAS_OP_T, CUBLAS_OP_N,
                 coefficient_dim, target_count, n,
                 &one, d_X.get(), n, d_Y.get(), n,
                 &zero, d_projected_rhs.get(), coefficient_dim),
               "project single-penalty target RHS");
  check_cublas(cublasDgemm(
                 blas.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                 coefficient_dim, target_count, coefficient_dim,
                 &one, d_transform.get(), coefficient_dim,
                 d_projected_rhs.get(), coefficient_dim,
                 &zero, d_spectral_projection.get(), coefficient_dim),
               "transform single-penalty target RHS");
  diagnostics.projection_gemm_count = 2;
  target_squared_norm_kernel<<<target_count, kReductionBlock, 0,
                               stream.get()>>>(
    d_Y.get(), d_y_squared_norm.get(), n, target_count);
  check_cuda(cudaGetLastError(),
             "launch single-penalty target norm kernel");
  const int optimizer_blocks =
    (target_count + kOptimizerBlock - 1) / kOptimizerBlock;
  magic_target_squared_norm_kernel<<<optimizer_blocks, kOptimizerBlock, 0,
                                     stream.get()>>>(
    d_Y.get(), d_magic_y_squared_norm.get(), n, target_count);
  check_cuda(cudaGetLastError(),
             "launch mgcv-compatible target norm kernel");
  magic_qt_y_kernel<<<optimizer_blocks, kOptimizerBlock, 0, stream.get()>>>(
    d_magic_qr_packed.get(), d_magic_tau.get(), d_Y.get(),
    d_magic_qt_y_work.get(), d_magic_y0.get(), n, coefficient_dim,
    target_count);
  check_cuda(cudaGetLastError(),
             "launch mgcv-compatible packed QR response projection");
  diagnostics.mgcv_qt_y_kernel_launch_count = 1;
  diagnostics.target_rhs_projected_on_cuda = true;
  check_cuda(cudaEventRecord(projection_done.get(), stream.get()),
             "record single-penalty GCV projection");

  if (materialize_grid) {
    constexpr int exact_block_size = 64;
    const int exact_candidate_count = static_cast<int>(sp_grid.size());
    const int exact_candidate_blocks =
      (exact_candidate_count + exact_block_size - 1) / exact_block_size;
    single_penalty_mroot_kernel<<<
      exact_candidate_blocks, exact_block_size, 0, stream.get()>>>(
        d_magic_penalty_matrix.get(), d_exact_grid_log_sp.get(),
        d_exact_factor.get(), d_exact_roots.get(),
        d_exact_mroot_work.get(), d_exact_root_ranks.get(),
        d_exact_pivots.get(), coefficient_dim, penalty_rank,
        exact_candidate_count);
    check_cuda(cudaGetLastError(),
               "launch exact-grid dynamic penalty mroot");
    diagnostics.exact_mroot_kernel_launch_count = 1;

    factor_exact_grid_augmented_matrices_kernel<<<
      exact_candidate_count, exact_endpoint_threads, 0, stream.get()>>>(
        d_magic_r.get(), d_exact_roots.get(), d_exact_root_ranks.get(),
        d_exact_grid_workspaces.get(), d_exact_singular_vectors.get(),
        d_exact_singular_values.get(), d_exact_svd_info.get(),
        coefficient_dim, penalty_rank, exact_candidate_count);
    check_cuda(cudaGetLastError(),
               "factor exact-grid augmented matrices with LAPACK 3.12 DGESDD");
    diagnostics.exact_svd_call_count = exact_candidate_count;

    const int exact_grid_blocks = static_cast<int>(
      (grid_count + exact_block_size - 1) / exact_block_size);
    evaluate_exact_grid_kernel<<<
      exact_grid_blocks, exact_block_size, 0, stream.get()>>>(
        d_exact_singular_vectors.get(), d_exact_singular_values.get(),
        d_exact_svd_info.get(), d_exact_root_ranks.get(), d_magic_y0.get(),
        d_magic_y_squared_norm.get(), d_grid.get(), coefficient_dim,
        penalty_rank, n, target_count, exact_candidate_count);
    check_cuda(cudaGetLastError(),
               "launch exact-grid objective kernel");
    diagnostics.grid_kernel_launch_count = 1;
    diagnostics.exact_objective_kernel_launch_count = 1;
  }
  check_cuda(cudaEventRecord(grid_done.get(), stream.get()),
             "record single-penalty GCV grid");

  {
    if (keep_transcript) {
      check_cuda(cudaMemsetAsync(
        d_transcript_counts.get(), 0,
        static_cast<std::size_t>(target_count) * sizeof(int), stream.get()),
        "zero single-penalty GCV transcript counts");
      check_cuda(cudaMemsetAsync(
        d_transcript_overflow.get(), 0,
        static_cast<std::size_t>(target_count) * sizeof(int), stream.get()),
        "zero single-penalty GCV transcript overflow flags");
    }
    check_cuda(cudaMemsetAsync(
      d_exact_endpoint_counts.get(), 0,
      kExactEndpointCounterCount * sizeof(int), stream.get()),
      "zero LAPACK 3.12 endpoint scorer counters");
    check_cuda(cudaMemsetAsync(
      d_spectral_endpoint_states.get(), 0,
      static_cast<std::size_t>(target_count) *
        sizeof(SpectralEndpointRefinementState), stream.get()),
      "zero spectral endpoint refinement states");
    optimize_targets_kernel<<<
      optimizer_blocks, kOptimizerBlock, 0, stream.get()>>>(
        d_eigenvalues.get(), d_spectral_projection.get(),
        d_y_squared_norm.get(), d_spectral_endpoint_states.get(),
        d_targets.get(),
        keep_transcript ? d_transcript.get() : nullptr,
        keep_transcript ? d_transcript_counts.get() : nullptr,
        keep_transcript ? d_transcript_overflow.get() : nullptr,
        initial_sp, coefficient_dim, penalty_rank, n, target_count);
    check_cuda(cudaGetLastError(),
               "launch single-kernel spectral GCV optimizer");
    mark_spectral_exact_replay_kernel<<<
      optimizer_blocks, kOptimizerBlock, 0, stream.get()>>>(
        d_spectral_endpoint_states.get(), d_targets.get(),
        d_magic_replay_flags.get(), d_exact_endpoint_counts.get() + 6,
        target_count);
    check_cuda(cudaGetLastError(),
               "mark spectral targets for full exact replay");
    if (!defer_exact_replay) {
      check_cuda(cudaMemsetAsync(
        d_magic_states.get(), 0,
        static_cast<std::size_t>(target_count) *
          sizeof(DeviceMagicOptimizerState), stream.get()),
        "zero spectral exact-replay states");
      replay_magic_exact_optimizer_kernel<<<
        target_count, exact_endpoint_threads, 0, stream.get()>>>(
          d_magic_states.get(), d_magic_replay_flags.get(),
          d_magic_r.get(), d_magic_penalty_matrix.get(),
          d_magic_penalty_root.get(), d_magic_y0.get(),
          d_magic_y_squared_norm.get(), d_exact_endpoint_workspaces.get(),
          keep_transcript ? d_transcript.get() : nullptr,
          keep_transcript ? d_transcript_counts.get() : nullptr,
          keep_transcript ? d_transcript_overflow.get() : nullptr,
          d_exact_endpoint_counts.get(), log(initial_sp), 1,
          coefficient_dim, penalty_rank, n, target_count);
      check_cuda(cudaGetLastError(),
                 "fully replay exceptional spectral targets exactly");
      finalize_magic_optimizer_kernel<<<
        optimizer_blocks, kOptimizerBlock, 0, stream.get()>>>(
          d_magic_states.get(), d_magic_replay_flags.get(),
          keep_transcript ? d_transcript_overflow.get() : nullptr,
          d_targets.get(), target_count);
      check_cuda(cudaGetLastError(),
                 "finalize exact spectral target replays");
      diagnostics.exact_endpoint_kernel_launch_count += 1;
      diagnostics.optimizer_kernel_launch_count = 4;
    } else {
      diagnostics.optimizer_kernel_launch_count = 2;
    }
  }
  diagnostics.target_selection_on_cuda = true;
  check_cuda(cudaEventRecord(optimizer_done.get(), stream.get()),
             "record single-penalty GCV optimizer");

  result.targets.resize(target_count);
  download_async(result.targets.data(), d_targets, target_count, stream.get());
  if (defer_exact_replay) {
    result.deferred_exact_replay_flags.resize(target_count);
    download_async(
      result.deferred_exact_replay_flags.data(), d_magic_replay_flags,
      target_count, stream.get());
  }
  int exact_endpoint_counts[kExactEndpointCounterCount] = {};
  check_cuda(cudaMemcpyAsync(
    exact_endpoint_counts, d_exact_endpoint_counts.get(),
    sizeof(exact_endpoint_counts), cudaMemcpyDeviceToHost, stream.get()),
    "download LAPACK 3.12 endpoint scorer counters");
  diagnostics.compact_d2h_count = defer_exact_replay ? 3 : 2;
  diagnostics.compact_d2h_bytes =
    static_cast<std::size_t>(target_count) *
      sizeof(SinglePenaltyGcvOptimizerResult) + sizeof(exact_endpoint_counts) +
    (defer_exact_replay ?
      static_cast<std::size_t>(target_count) * sizeof(int) : 0);
  std::vector<int> exact_svd_info;
  if (materialize_grid) {
    result.grid.resize(grid_count);
    exact_svd_info.resize(sp_grid.size());
    download_async(result.grid.data(), d_grid, grid_count, stream.get());
    download_async(
      exact_svd_info.data(), d_exact_svd_info, sp_grid.size(), stream.get());
    diagnostics.grid_d2h_count = 1;
    diagnostics.grid_d2h_bytes =
      grid_count * sizeof(SinglePenaltyGcvGridCell);
  }
  if (keep_transcript) {
    result.transcript.resize(transcript_capacity);
    result.transcript_counts.resize(target_count);
    result.transcript_overflow.resize(target_count);
    download_async(result.transcript.data(), d_transcript,
                   transcript_capacity, stream.get());
    download_async(result.transcript_counts.data(), d_transcript_counts,
                   target_count, stream.get());
    download_async(result.transcript_overflow.data(), d_transcript_overflow,
                   target_count, stream.get());
    diagnostics.transcript_d2h_count = 3;
    diagnostics.transcript_d2h_bytes =
      transcript_capacity * sizeof(SinglePenaltyGcvTranscriptEntry) +
      static_cast<std::size_t>(target_count) * 2 * sizeof(int);
  }
  check_cuda(cudaEventRecord(d2h_done.get(), stream.get()),
             "record single-penalty GCV download");
  check_cuda(cudaEventSynchronize(d2h_done.get()),
             "synchronize single-penalty GCV result");

  diagnostics.exact_endpoint_comparison_count = exact_endpoint_counts[0];
  diagnostics.exact_endpoint_svd_call_count = exact_endpoint_counts[1];
  diagnostics.exact_endpoint_failure_count = exact_endpoint_counts[2];
  diagnostics.exact_endpoint_trial_accepted_count = exact_endpoint_counts[3];
  diagnostics.exact_derivative_refresh_count = exact_endpoint_counts[4];
  diagnostics.exact_derivative_svd_call_count = exact_endpoint_counts[4];
  diagnostics.exact_derivative_failure_count = exact_endpoint_counts[5];
  diagnostics.selective_replay_target_count = exact_endpoint_counts[4];
  diagnostics.selective_replay_deferred_count = exact_endpoint_counts[6];
  diagnostics.selective_replay_steepest_count = exact_endpoint_counts[7];
  diagnostics.selective_replay_residual_risk_count = exact_endpoint_counts[8];
  diagnostics.selective_replay_other_count = exact_endpoint_counts[9];
  diagnostics.spectral_optimizer_target_count = target_count;
  diagnostics.exact_replay_target_count = defer_exact_replay ?
    static_cast<int>(std::count(
      result.deferred_exact_replay_flags.begin(),
      result.deferred_exact_replay_flags.end(), 1)) :
    exact_endpoint_counts[4];
  diagnostics.spectral_only_target_count =
    target_count - diagnostics.exact_replay_target_count;
  diagnostics.exact_replay_endpoint_risk_count = exact_endpoint_counts[6];
  diagnostics.exact_replay_convergence_risk_count = exact_endpoint_counts[7];
  diagnostics.exact_replay_boundary_risk_count = exact_endpoint_counts[8];
  diagnostics.exact_replay_numerical_risk_count = exact_endpoint_counts[9];
  diagnostics.optimizer_target_coverage_complete =
    diagnostics.spectral_only_target_count >= 0 &&
    diagnostics.spectral_only_target_count +
      diagnostics.exact_replay_target_count == target_count;

  diagnostics.upload_cuda_ms = elapsed_ms(stage_start, upload_done);
  diagnostics.projection_cuda_ms = elapsed_ms(upload_done, projection_done);
  diagnostics.grid_cuda_ms = elapsed_ms(projection_done, grid_done);
  diagnostics.optimizer_cuda_ms = elapsed_ms(grid_done, optimizer_done);
  diagnostics.d2h_cuda_ms = elapsed_ms(optimizer_done, d2h_done);
  diagnostics.exact_svd_nonconverged_count = static_cast<int>(std::count_if(
    exact_svd_info.begin(), exact_svd_info.end(),
    [](int value) { return value != 0; }));
  for (const SinglePenaltyGcvOptimizerResult& target : result.targets) {
    diagnostics.cuda_gcv_iterations += target.iteration_count;
    if (!target.fully_converged) diagnostics.cuda_gcv_nonconverged += 1;
    if (target.boundary_accepted_count > 0) {
      diagnostics.cuda_gcv_boundary_targets += 1;
    }
  }
  diagnostics.total_host_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - host_begin).count();
  return result;
}

namespace {

struct FusedExactReplayExecution {
  int target_count = 0;
  int kernel_launch_count = 0;
  std::size_t device_bytes = 0;
  double cuda_ms = 0.0;
  double host_ms = 0.0;
};

struct FusedExactReplaySetupOffsets {
  std::size_t qr_offset = 0;
  std::size_t tau_offset = 0;
  std::size_t magic_r_offset = 0;
  std::size_t penalty_matrix_offset = 0;
  std::size_t penalty_root_offset = 0;
  bool present = false;
};

template <typename T>
std::size_t append_fused_values(
    std::vector<T>* destination,
    const std::vector<T>& source) {
  const std::size_t offset = destination->size();
  destination->insert(destination->end(), source.begin(), source.end());
  return offset;
}

void recompute_optimizer_diagnostics(SinglePenaltyGcvCudaResult* setup) {
  SinglePenaltyGcvCudaDiagnostics& diagnostics = setup->diagnostics;
  diagnostics.cuda_gcv_iterations = 0;
  diagnostics.cuda_gcv_nonconverged = 0;
  diagnostics.cuda_gcv_boundary_targets = 0;
  for (const SinglePenaltyGcvOptimizerResult& target : setup->targets) {
    diagnostics.cuda_gcv_iterations += target.iteration_count;
    if (!target.fully_converged) diagnostics.cuda_gcv_nonconverged += 1;
    if (target.boundary_accepted_count > 0) {
      diagnostics.cuda_gcv_boundary_targets += 1;
    }
  }
}

FusedExactReplayExecution execute_fused_exact_replay(
    const std::vector<SinglePenaltyGcvCudaOwnedInput>& inputs,
    std::vector<SinglePenaltyGcvCudaResult>* setups) {
  const auto host_begin = std::chrono::steady_clock::now();
  if (setups == nullptr || setups->size() != inputs.size()) {
    throw std::runtime_error("fused exact replay setup geometry is invalid");
  }

  std::vector<FusedExactReplaySetupOffsets> setup_offsets(inputs.size());
  std::vector<double> qr_packed;
  std::vector<double> tau;
  std::vector<double> magic_r;
  std::vector<double> penalty_matrix;
  std::vector<double> penalty_root;
  std::vector<double> responses;
  std::vector<FusedExactReplayTarget> metadata;
  std::vector<int> output_setup;
  std::vector<int> output_target;
  std::size_t y0_count = 0;

  for (std::size_t setup_index = 0; setup_index < inputs.size();
       ++setup_index) {
    const SinglePenaltyGcvCudaOwnedInput& input = inputs[setup_index];
    const SinglePenaltyGcvCudaResult& setup = (*setups)[setup_index];
    if (setup.deferred_exact_replay_flags.size() !=
        static_cast<std::size_t>(input.target_count)) {
      throw std::runtime_error(
        "fused exact replay flags do not match the setup target count");
    }
    const bool has_risk = std::find(
      setup.deferred_exact_replay_flags.begin(),
      setup.deferred_exact_replay_flags.end(), 1) !=
      setup.deferred_exact_replay_flags.end();
    if (!has_risk) continue;
    const std::size_t q = static_cast<std::size_t>(input.coefficient_dim);
    const std::size_t n = static_cast<std::size_t>(input.n);
    const std::size_t rank = static_cast<std::size_t>(input.penalty_rank);
    if (input.magic_qr_packed.size() != n * q ||
        input.magic_tau.size() != q || input.magic_r.size() != q * q ||
        input.magic_penalty_matrix.size() != q * q ||
        input.magic_penalty_root.size() != q * rank ||
        input.Y.size() != n * static_cast<std::size_t>(input.target_count)) {
      throw std::runtime_error(
        "fused exact replay source arrays are malformed");
    }
    FusedExactReplaySetupOffsets& offsets = setup_offsets[setup_index];
    offsets.qr_offset = append_fused_values(&qr_packed, input.magic_qr_packed);
    offsets.tau_offset = append_fused_values(&tau, input.magic_tau);
    offsets.magic_r_offset = append_fused_values(&magic_r, input.magic_r);
    offsets.penalty_matrix_offset =
      append_fused_values(&penalty_matrix, input.magic_penalty_matrix);
    offsets.penalty_root_offset =
      append_fused_values(&penalty_root, input.magic_penalty_root);
    offsets.present = true;
  }

  int small_target_count = 0;
  for (int large_shape = 0; large_shape <= 1; ++large_shape) {
    for (std::size_t setup_index = 0; setup_index < inputs.size();
         ++setup_index) {
      const SinglePenaltyGcvCudaOwnedInput& input = inputs[setup_index];
      const SinglePenaltyGcvCudaResult& setup = (*setups)[setup_index];
      const bool is_large =
        input.coefficient_dim > FASTKPC_LAPACK_SMALL_SMLSIZ;
      if (is_large != (large_shape != 0)) continue;
      const FusedExactReplaySetupOffsets& offsets =
        setup_offsets[setup_index];
      if (!offsets.present) continue;
      for (int target = 0; target < input.target_count; ++target) {
        const int flag = setup.deferred_exact_replay_flags[
          static_cast<std::size_t>(target)];
        if (flag != 0 && flag != 1) {
          throw std::runtime_error(
            "fused exact replay flag is not binary");
        }
        if (flag == 0) continue;
        FusedExactReplayTarget item;
        item.qr_offset = offsets.qr_offset;
        item.tau_offset = offsets.tau_offset;
        item.magic_r_offset = offsets.magic_r_offset;
        item.penalty_matrix_offset = offsets.penalty_matrix_offset;
        item.penalty_root_offset = offsets.penalty_root_offset;
        item.y_offset = responses.size();
        item.y0_offset = y0_count;
        item.initial_log_sp = std::log(input.initial_sp);
        item.setup_index = static_cast<int>(setup_index);
        item.target_index = target;
        item.n = input.n;
        item.q = input.coefficient_dim;
        item.penalty_rank = input.penalty_rank;
        const double* source_y = input.Y.data() +
          static_cast<std::size_t>(input.n) * target;
        responses.insert(
          responses.end(), source_y, source_y + input.n);
        y0_count += static_cast<std::size_t>(input.coefficient_dim);
        metadata.push_back(item);
        output_setup.push_back(static_cast<int>(setup_index));
        output_target.push_back(target);
      }
    }
    if (large_shape == 0) {
      small_target_count = static_cast<int>(metadata.size());
    }
  }

  FusedExactReplayExecution execution;
  execution.target_count = static_cast<int>(metadata.size());
  if (metadata.empty()) {
    for (SinglePenaltyGcvCudaResult& setup : *setups) {
      setup.deferred_exact_replay_flags.clear();
    }
    execution.host_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - host_begin).count();
    return execution;
  }

  SinglePenaltyGcvCudaDiagnostics allocation_diagnostics;
  Stream stream;
  DeviceAllocationStreamScope allocation_scope(stream.get());
  DeviceBuffer<FusedExactReplayTarget> d_metadata;
  DeviceBuffer<double> d_qr_packed;
  DeviceBuffer<double> d_tau;
  DeviceBuffer<double> d_magic_r;
  DeviceBuffer<double> d_penalty_matrix;
  DeviceBuffer<double> d_penalty_root;
  DeviceBuffer<double> d_responses;
  DeviceBuffer<double> d_projection_work;
  DeviceBuffer<double> d_y0;
  DeviceBuffer<double> d_y_squared_norm;
  DeviceBuffer<lapack312::Workspace> d_workspaces;
  DeviceBuffer<DeviceMagicOptimizerState> d_states;
  DeviceBuffer<SinglePenaltyGcvOptimizerResult> d_targets;
  DeviceBuffer<int> d_setup_exact_counts;
  d_metadata.allocate(metadata.size(), &allocation_diagnostics);
  d_qr_packed.allocate(qr_packed.size(), &allocation_diagnostics);
  d_tau.allocate(tau.size(), &allocation_diagnostics);
  d_magic_r.allocate(magic_r.size(), &allocation_diagnostics);
  d_penalty_matrix.allocate(penalty_matrix.size(), &allocation_diagnostics);
  d_penalty_root.allocate(penalty_root.size(), &allocation_diagnostics);
  d_responses.allocate(responses.size(), &allocation_diagnostics);
  d_projection_work.allocate(responses.size(), &allocation_diagnostics);
  d_y0.allocate(y0_count, &allocation_diagnostics);
  d_y_squared_norm.allocate(metadata.size(), &allocation_diagnostics);
  d_workspaces.allocate(metadata.size(), &allocation_diagnostics);
  d_states.allocate(metadata.size(), &allocation_diagnostics);
  d_targets.allocate(metadata.size(), &allocation_diagnostics);
  d_setup_exact_counts.allocate(
    inputs.size() * kExactEndpointCounterCount, &allocation_diagnostics);
  upload_async(
    &d_metadata, metadata.data(), metadata.size(), stream.get(),
    &allocation_diagnostics);
  upload_async(
    &d_qr_packed, qr_packed.data(), qr_packed.size(), stream.get(),
    &allocation_diagnostics);
  upload_async(
    &d_tau, tau.data(), tau.size(), stream.get(), &allocation_diagnostics);
  upload_async(
    &d_magic_r, magic_r.data(), magic_r.size(), stream.get(),
    &allocation_diagnostics);
  upload_async(
    &d_penalty_matrix, penalty_matrix.data(), penalty_matrix.size(),
    stream.get(), &allocation_diagnostics);
  upload_async(
    &d_penalty_root, penalty_root.data(), penalty_root.size(), stream.get(),
    &allocation_diagnostics);
  upload_async(
    &d_responses, responses.data(), responses.size(), stream.get(),
    &allocation_diagnostics);
  check_cuda(cudaMemsetAsync(
    d_setup_exact_counts.get(), 0,
    inputs.size() * kExactEndpointCounterCount * sizeof(int), stream.get()),
    "zero fused exact replay counters");

  Event fused_start;
  Event fused_done;
  check_cuda(cudaEventRecord(fused_start.get(), stream.get()),
             "record fused exact replay start");
  const int projection_blocks =
    (execution.target_count + kOptimizerBlock - 1) / kOptimizerBlock;
  project_fused_exact_replay_targets_kernel<<<
    projection_blocks, kOptimizerBlock, 0, stream.get()>>>(
      d_metadata.get(), d_qr_packed.get(), d_tau.get(), d_responses.get(),
      d_projection_work.get(), d_y0.get(), d_y_squared_norm.get(),
      execution.target_count);
  check_cuda(cudaGetLastError(), "project fused exact replay responses");

  if (small_target_count > 0) {
    replay_fused_exact_optimizer_kernel<<<
      small_target_count, kExactWarp, 0, stream.get()>>>(
        d_metadata.get(), d_magic_r.get(), d_penalty_matrix.get(),
        d_penalty_root.get(), d_y0.get(), d_y_squared_norm.get(),
        d_workspaces.get(), d_states.get(), d_setup_exact_counts.get(), 0,
        small_target_count);
    check_cuda(cudaGetLastError(),
               "replay fused small exact optimizer targets");
    execution.kernel_launch_count += 1;
  }
  const int large_target_count = execution.target_count - small_target_count;
  if (large_target_count > 0) {
    replay_fused_exact_optimizer_kernel<<<
      large_target_count, 64, 0, stream.get()>>>(
        d_metadata.get(), d_magic_r.get(), d_penalty_matrix.get(),
        d_penalty_root.get(), d_y0.get(), d_y_squared_norm.get(),
        d_workspaces.get(), d_states.get(), d_setup_exact_counts.get(),
        small_target_count, large_target_count);
    check_cuda(cudaGetLastError(),
               "replay fused large exact optimizer targets");
    execution.kernel_launch_count += 1;
  }
  const int finalize_blocks =
    (execution.target_count + kOptimizerBlock - 1) / kOptimizerBlock;
  finalize_magic_optimizer_kernel<<<
    finalize_blocks, kOptimizerBlock, 0, stream.get()>>>(
      d_states.get(), nullptr, nullptr, d_targets.get(),
      execution.target_count);
  check_cuda(cudaGetLastError(), "finalize fused exact replay targets");

  std::vector<SinglePenaltyGcvOptimizerResult> exact_targets(metadata.size());
  std::vector<int> exact_counts(
    inputs.size() * kExactEndpointCounterCount);
  download_async(
    exact_targets.data(), d_targets, exact_targets.size(), stream.get());
  download_async(
    exact_counts.data(), d_setup_exact_counts, exact_counts.size(),
    stream.get());
  check_cuda(cudaEventRecord(fused_done.get(), stream.get()),
             "record fused exact replay completion");
  check_cuda(cudaEventSynchronize(fused_done.get()),
             "synchronize fused exact replay completion");
  execution.cuda_ms = elapsed_ms(fused_start, fused_done);
  execution.device_bytes = allocation_diagnostics.device_allocation_bytes;

  for (std::size_t index = 0; index < exact_targets.size(); ++index) {
    (*setups)[static_cast<std::size_t>(output_setup[index])]
      .targets[static_cast<std::size_t>(output_target[index])] =
      exact_targets[index];
  }
  for (std::size_t setup_index = 0; setup_index < setups->size();
       ++setup_index) {
    SinglePenaltyGcvCudaResult& setup = (*setups)[setup_index];
    SinglePenaltyGcvCudaDiagnostics& diagnostics = setup.diagnostics;
    const int* counts = exact_counts.data() +
      setup_index * kExactEndpointCounterCount;
    if (counts[4] != diagnostics.exact_replay_target_count) {
      throw std::runtime_error(
        "fused exact replay target coverage does not match risk flags");
    }
    diagnostics.exact_endpoint_comparison_count = counts[0];
    diagnostics.exact_endpoint_svd_call_count = counts[1];
    diagnostics.exact_endpoint_failure_count = counts[2];
    diagnostics.exact_endpoint_trial_accepted_count = counts[3];
    diagnostics.exact_derivative_refresh_count = counts[4];
    diagnostics.exact_derivative_svd_call_count = counts[4];
    diagnostics.exact_derivative_failure_count = counts[5];
    diagnostics.selective_replay_target_count = counts[4];
    diagnostics.spectral_only_target_count =
      diagnostics.target_count - counts[4];
    diagnostics.optimizer_target_coverage_complete =
      diagnostics.spectral_only_target_count >= 0 &&
      diagnostics.spectral_only_target_count + counts[4] ==
        diagnostics.target_count;
    const double share = execution.target_count > 0 ?
      static_cast<double>(counts[4]) / execution.target_count : 0.0;
    diagnostics.optimizer_cuda_ms += share * execution.cuda_ms;
    recompute_optimizer_diagnostics(&setup);
    setup.deferred_exact_replay_flags.clear();
  }
  execution.host_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - host_begin).count();
  for (SinglePenaltyGcvCudaResult& setup : *setups) {
    const double share = execution.target_count > 0 ?
      static_cast<double>(setup.diagnostics.exact_replay_target_count) /
        execution.target_count : 0.0;
    setup.diagnostics.total_host_ms += share * execution.host_ms;
  }
  return execution;
}

}  // namespace

SinglePenaltyGcvCudaMultiResult single_penalty_gcv_cuda_multi(
    const std::vector<SinglePenaltyGcvCudaOwnedInput>& inputs,
    int requested_concurrency) {
  if (inputs.empty()) {
    throw std::runtime_error(
      "multi-setup single-penalty GCV input must be non-empty");
  }
  if (requested_concurrency <= 0 ||
      requested_concurrency > kSinglePenaltyGcvMaximumConcurrentSetups) {
    throw std::runtime_error(
      "multi-setup single-penalty GCV concurrency must be in [1, 16]");
  }

  SinglePenaltyGcvCudaMultiResult result;
  result.schema_version = kSinglePenaltyGcvCudaMultiSchemaVersion;
  result.setups.resize(inputs.size());
  SinglePenaltyGcvCudaMultiDiagnostics& diagnostics = result.diagnostics;
  diagnostics.schema_version = kSinglePenaltyGcvCudaMultiSchemaVersion;
  diagnostics.setup_count = static_cast<int>(inputs.size());
  diagnostics.requested_concurrency = requested_concurrency;
  diagnostics.worker_count = std::min(
    requested_concurrency, static_cast<int>(inputs.size()));
  const bool use_fused_exact_replay =
    std::all_of(inputs.begin(), inputs.end(), [](const auto& input) {
      return !input.materialize_grid && !input.keep_transcript;
    });
  diagnostics.execution_strategy = use_fused_exact_replay ?
    "cuda-cross-setup-fused-exact-replay" :
    "cuda-independent-setup-streams";
  diagnostics.fused_exact_replay_executed = use_fused_exact_replay;
  check_cuda(cudaGetDevice(&diagnostics.device_id),
             "query multi-setup single-penalty GCV CUDA device");

  const auto wall_begin = std::chrono::steady_clock::now();
  std::atomic<int> next_setup{0};
  std::atomic<int> active_calls{0};
  std::atomic<int> max_active_calls{0};
  std::atomic<int> device_bind_count{0};
  std::atomic<bool> stop{false};
  std::vector<std::exception_ptr> setup_errors(inputs.size());
  std::vector<std::exception_ptr> worker_errors(
    static_cast<std::size_t>(diagnostics.worker_count));

  auto update_max_active = [&](int active) {
    int observed = max_active_calls.load(std::memory_order_relaxed);
    while (observed < active &&
           !max_active_calls.compare_exchange_weak(
             observed, active, std::memory_order_relaxed)) {
    }
  };
  auto worker = [&](int worker_id) {
    try {
      check_cuda(cudaSetDevice(diagnostics.device_id),
                 "bind multi-setup single-penalty GCV worker device");
      device_bind_count.fetch_add(1, std::memory_order_relaxed);
    } catch (...) {
      worker_errors[static_cast<std::size_t>(worker_id)] =
        std::current_exception();
      stop.store(true, std::memory_order_relaxed);
      return;
    }

    for (;;) {
      if (stop.load(std::memory_order_relaxed)) return;
      const int setup_index =
        next_setup.fetch_add(1, std::memory_order_relaxed);
      if (setup_index >= static_cast<int>(inputs.size())) return;
      const SinglePenaltyGcvCudaOwnedInput& input =
        inputs[static_cast<std::size_t>(setup_index)];
      const int active =
        active_calls.fetch_add(1, std::memory_order_relaxed) + 1;
      update_max_active(active);
      try {
        result.setups[static_cast<std::size_t>(setup_index)] =
          single_penalty_gcv_cuda(
            input.X.data(), input.Y.data(), input.rhs_transform.data(),
            input.eigenvalues.data(), input.magic_qr_packed.data(),
            input.magic_tau.data(), input.magic_r.data(),
            input.magic_penalty_root.data(),
            input.magic_penalty_matrix.data(), input.target_ids.data(),
            input.n, input.coefficient_dim, input.target_count,
            input.penalty_rank, input.initial_sp, input.sp_grid,
            input.materialize_grid, input.keep_transcript,
            use_fused_exact_replay);
      } catch (...) {
        setup_errors[static_cast<std::size_t>(setup_index)] =
          std::current_exception();
        stop.store(true, std::memory_order_relaxed);
      }
      active_calls.fetch_sub(1, std::memory_order_relaxed);
      if (stop.load(std::memory_order_relaxed)) return;
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>(diagnostics.worker_count));
  for (int worker_id = 0; worker_id < diagnostics.worker_count; ++worker_id) {
    workers.emplace_back(worker, worker_id);
  }
  for (std::thread& thread : workers) thread.join();

  for (const std::exception_ptr& error : worker_errors) {
    if (error) std::rethrow_exception(error);
  }
  for (const std::exception_ptr& error : setup_errors) {
    if (error) std::rethrow_exception(error);
  }

  diagnostics.spectral_setup_wall_ms =
    std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - wall_begin).count();
  if (use_fused_exact_replay) {
    const FusedExactReplayExecution fused =
      execute_fused_exact_replay(inputs, &result.setups);
    diagnostics.fused_exact_replay_target_count = fused.target_count;
    diagnostics.fused_exact_replay_kernel_launch_count =
      fused.kernel_launch_count;
    diagnostics.fused_exact_replay_device_bytes = fused.device_bytes;
    diagnostics.fused_exact_replay_cuda_ms = fused.cuda_ms;
    diagnostics.fused_exact_replay_host_ms = fused.host_ms;
  }

  diagnostics.worker_device_bind_count =
    device_bind_count.load(std::memory_order_relaxed);
  diagnostics.max_host_calls_in_flight =
    max_active_calls.load(std::memory_order_relaxed);
  for (const SinglePenaltyGcvCudaResult& setup : result.setups) {
    diagnostics.setup_host_ms_sum += setup.diagnostics.total_host_ms;
  }
  diagnostics.wall_host_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - wall_begin).count();
  if (diagnostics.wall_host_ms > 0.0) {
    diagnostics.host_overlap_factor =
      diagnostics.setup_host_ms_sum / diagnostics.wall_host_ms;
  }
  return result;
}

}  // namespace fastkpc
