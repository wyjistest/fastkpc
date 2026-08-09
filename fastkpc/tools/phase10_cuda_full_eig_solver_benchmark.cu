#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* stage) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string(stage) + ": " + cudaGetErrorString(status));
  }
}

void check_solver(cusolverStatus_t status, const char* stage) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error(
      std::string(stage) + ": cuSOLVER status " +
      std::to_string(static_cast<int>(status)));
  }
}

class Stream {
 public:
  Stream() {
    check_cuda(cudaStreamCreateWithFlags(&value_, cudaStreamNonBlocking),
               "create stream");
  }
  ~Stream() {
    if (value_ != nullptr) cudaStreamDestroy(value_);
  }
  cudaStream_t get() const noexcept { return value_; }

 private:
  cudaStream_t value_ = nullptr;
};

class Event {
 public:
  Event() { check_cuda(cudaEventCreate(&value_), "create event"); }
  ~Event() {
    if (value_ != nullptr) cudaEventDestroy(value_);
  }
  cudaEvent_t get() const noexcept { return value_; }

 private:
  cudaEvent_t value_ = nullptr;
};

class Solver {
 public:
  explicit Solver(cudaStream_t stream) {
    check_solver(cusolverDnCreate(&value_), "create solver");
    check_solver(cusolverDnSetStream(value_, stream), "set solver stream");
    check_solver(cusolverDnSetDeterministicMode(
      value_, CUSOLVER_DETERMINISTIC_RESULTS),
      "set deterministic solver mode");
  }
  ~Solver() {
    if (value_ != nullptr) cusolverDnDestroy(value_);
  }
  cusolverDnHandle_t get() const noexcept { return value_; }

 private:
  cusolverDnHandle_t value_ = nullptr;
};

class JacobiParameters {
 public:
  JacobiParameters() {
    check_solver(cusolverDnCreateSyevjInfo(&value_),
                 "create Jacobi parameters");
    check_solver(cusolverDnXsyevjSetTolerance(value_, 1.0e-12),
                 "set Jacobi tolerance");
    check_solver(cusolverDnXsyevjSetMaxSweeps(value_, 100),
                 "set Jacobi maximum sweeps");
    check_solver(cusolverDnXsyevjSetSortEig(value_, 1),
                 "enable Jacobi eigenvalue sorting");
  }
  ~JacobiParameters() {
    if (value_ != nullptr) cusolverDnDestroySyevjInfo(value_);
  }
  syevjInfo_t get() const noexcept { return value_; }

 private:
  syevjInfo_t value_ = nullptr;
};

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count_ == 0U) throw std::runtime_error("zero-size device buffer");
    check_cuda(cudaMalloc(&value_, count_ * sizeof(T)), "allocate buffer");
  }
  ~DeviceBuffer() {
    if (value_ != nullptr) cudaFree(value_);
  }
  T* get() const noexcept { return static_cast<T*>(value_); }
  std::size_t bytes() const noexcept { return count_ * sizeof(T); }

 private:
  void* value_ = nullptr;
  std::size_t count_ = 0U;
};

float elapsed(cudaEvent_t start, cudaEvent_t stop) {
  check_cuda(cudaEventSynchronize(stop), "wait benchmark stop");
  float milliseconds = 0.0F;
  check_cuda(cudaEventElapsedTime(&milliseconds, start, stop),
             "read benchmark elapsed time");
  return milliseconds;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const int n = argc > 1 ? std::atoi(argv[1]) : 351;
    const int repeats = argc > 2 ? std::atoi(argv[2]) : 10;
    if (n < 2 || repeats < 1) {
      throw std::runtime_error("invalid benchmark arguments");
    }
    check_cuda(cudaSetDevice(0), "select benchmark device");
    Stream stream;
    Solver solver(stream.get());
    JacobiParameters jacobi;
    const std::size_t cells = static_cast<std::size_t>(n) * n;
    std::vector<double> host_source(cells);
    std::vector<double> residual(static_cast<std::size_t>(n));
    for (int row = 0; row < n; ++row) {
      residual[static_cast<std::size_t>(row)] =
        std::sin(0.017 * (row + 1)) +
        std::cos(0.0031 * (row + 1) * (row + 1));
    }
    for (int column = 0; column < n; ++column) {
      for (int row = 0; row < n; ++row) {
        host_source[static_cast<std::size_t>(column) * n + row] =
          std::abs(residual[static_cast<std::size_t>(row)] -
                   residual[static_cast<std::size_t>(column)]);
      }
    }

    DeviceBuffer<double> source(cells);
    DeviceBuffer<double> divide_matrix(cells);
    DeviceBuffer<double> jacobi_matrix(cells);
    DeviceBuffer<double> divide_values(static_cast<std::size_t>(n));
    DeviceBuffer<double> jacobi_values(static_cast<std::size_t>(n));
    DeviceBuffer<int> divide_info(1U);
    DeviceBuffer<int> jacobi_info(1U);
    check_cuda(cudaMemcpyAsync(
      source.get(), host_source.data(), source.bytes(),
      cudaMemcpyHostToDevice, stream.get()), "upload source matrix");

    int divide_lwork = 0;
    check_solver(cusolverDnDsyevd_bufferSize(
      solver.get(), CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
      n, divide_matrix.get(), n, divide_values.get(), &divide_lwork),
      "query divide-and-conquer workspace");
    int jacobi_lwork = 0;
    check_solver(cusolverDnDsyevj_bufferSize(
      solver.get(), CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
      n, jacobi_matrix.get(), n, jacobi_values.get(), &jacobi_lwork,
      jacobi.get()), "query Jacobi workspace");
    DeviceBuffer<double> divide_work(
      static_cast<std::size_t>(divide_lwork));
    DeviceBuffer<double> jacobi_work(
      static_cast<std::size_t>(jacobi_lwork));

    auto run_divide = [&]() {
      check_cuda(cudaMemcpyAsync(
        divide_matrix.get(), source.get(), source.bytes(),
        cudaMemcpyDeviceToDevice, stream.get()), "restore divide matrix");
      check_solver(cusolverDnDsyevd(
        solver.get(), CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        n, divide_matrix.get(), n, divide_values.get(), divide_work.get(),
        divide_lwork, divide_info.get()), "run divide-and-conquer eig");
    };
    auto run_jacobi = [&]() {
      check_cuda(cudaMemcpyAsync(
        jacobi_matrix.get(), source.get(), source.bytes(),
        cudaMemcpyDeviceToDevice, stream.get()), "restore Jacobi matrix");
      check_solver(cusolverDnDsyevj(
        solver.get(), CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        n, jacobi_matrix.get(), n, jacobi_values.get(), jacobi_work.get(),
        jacobi_lwork, jacobi_info.get(), jacobi.get()),
        "run Jacobi eig");
    };

    for (int warmup = 0; warmup < 2; ++warmup) run_divide();
    check_cuda(cudaStreamSynchronize(stream.get()), "wait divide warmup");
    Event divide_start;
    Event divide_stop;
    check_cuda(cudaEventRecord(divide_start.get(), stream.get()),
               "record divide start");
    for (int repeat = 0; repeat < repeats; ++repeat) run_divide();
    check_cuda(cudaEventRecord(divide_stop.get(), stream.get()),
               "record divide stop");
    const float divide_ms = elapsed(divide_start.get(), divide_stop.get());

    for (int warmup = 0; warmup < 2; ++warmup) run_jacobi();
    check_cuda(cudaStreamSynchronize(stream.get()), "wait Jacobi warmup");
    Event jacobi_start;
    Event jacobi_stop;
    check_cuda(cudaEventRecord(jacobi_start.get(), stream.get()),
               "record Jacobi start");
    for (int repeat = 0; repeat < repeats; ++repeat) run_jacobi();
    check_cuda(cudaEventRecord(jacobi_stop.get(), stream.get()),
               "record Jacobi stop");
    const float jacobi_ms = elapsed(jacobi_start.get(), jacobi_stop.get());

    // Leave one result from each solver resident for the numerical comparison.
    run_divide();
    run_jacobi();
    check_cuda(cudaStreamSynchronize(stream.get()), "wait comparison solves");
    std::vector<double> host_divide_matrix(cells);
    std::vector<double> host_jacobi_matrix(cells);
    std::vector<double> host_divide_values(static_cast<std::size_t>(n));
    std::vector<double> host_jacobi_values(static_cast<std::size_t>(n));
    int host_divide_info = -1;
    int host_jacobi_info = -1;
    check_cuda(cudaMemcpy(
      host_divide_matrix.data(), divide_matrix.get(), divide_matrix.bytes(),
      cudaMemcpyDeviceToHost), "download divide eigenvectors");
    check_cuda(cudaMemcpy(
      host_jacobi_matrix.data(), jacobi_matrix.get(), jacobi_matrix.bytes(),
      cudaMemcpyDeviceToHost), "download Jacobi eigenvectors");
    check_cuda(cudaMemcpy(
      host_divide_values.data(), divide_values.get(), divide_values.bytes(),
      cudaMemcpyDeviceToHost), "download divide eigenvalues");
    check_cuda(cudaMemcpy(
      host_jacobi_values.data(), jacobi_values.get(), jacobi_values.bytes(),
      cudaMemcpyDeviceToHost), "download Jacobi eigenvalues");
    check_cuda(cudaMemcpy(
      &host_divide_info, divide_info.get(), sizeof(int), cudaMemcpyDeviceToHost),
      "download divide info");
    check_cuda(cudaMemcpy(
      &host_jacobi_info, jacobi_info.get(), sizeof(int), cudaMemcpyDeviceToHost),
      "download Jacobi info");

    double maximum_eigenvalue_error = 0.0;
    double maximum_sign_aligned_vector_error = 0.0;
    for (int column = 0; column < n; ++column) {
      maximum_eigenvalue_error = std::max(
        maximum_eigenvalue_error,
        std::abs(host_divide_values[static_cast<std::size_t>(column)] -
                 host_jacobi_values[static_cast<std::size_t>(column)]));
      double dot = 0.0;
      for (int row = 0; row < n; ++row) {
        const std::size_t index = static_cast<std::size_t>(column) * n + row;
        dot += host_divide_matrix[index] * host_jacobi_matrix[index];
      }
      const double sign = dot < 0.0 ? -1.0 : 1.0;
      for (int row = 0; row < n; ++row) {
        const std::size_t index = static_cast<std::size_t>(column) * n + row;
        maximum_sign_aligned_vector_error = std::max(
          maximum_sign_aligned_vector_error,
          std::abs(host_divide_matrix[index] -
                   sign * host_jacobi_matrix[index]));
      }
    }

    const double divide_per_solve = divide_ms / repeats;
    const double jacobi_per_solve = jacobi_ms / repeats;
    std::printf(
      "n=%d repeats=%d divide_info=%d jacobi_info=%d "
      "divide_lwork=%d jacobi_lwork=%d "
      "divide_per_solve_ms=%.9g jacobi_per_solve_ms=%.9g "
      "jacobi_speedup=%.9g maximum_eigenvalue_error=%.17g "
      "maximum_sign_aligned_vector_error=%.17g\n",
      n, repeats, host_divide_info, host_jacobi_info,
      divide_lwork, jacobi_lwork, divide_per_solve, jacobi_per_solve,
      divide_per_solve / jacobi_per_solve, maximum_eigenvalue_error,
      maximum_sign_aligned_vector_error);
    return host_divide_info == 0 && host_jacobi_info == 0 ?
      EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return EXIT_FAILURE;
  }
}
