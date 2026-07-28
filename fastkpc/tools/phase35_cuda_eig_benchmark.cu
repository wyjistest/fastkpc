#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
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

class Event {
 public:
  Event() { check_cuda(cudaEventCreate(&value_), "create event"); }
  ~Event() { if (value_ != nullptr) cudaEventDestroy(value_); }
  cudaEvent_t get() const { return value_; }
 private:
  cudaEvent_t value_ = nullptr;
};

class Stream {
 public:
  Stream() {
    check_cuda(cudaStreamCreateWithFlags(&value_, cudaStreamNonBlocking),
               "create stream");
  }
  ~Stream() { if (value_ != nullptr) cudaStreamDestroy(value_); }
  cudaStream_t get() const { return value_; }
 private:
  cudaStream_t value_ = nullptr;
};

class Solver {
 public:
  explicit Solver(cudaStream_t stream) {
    check_solver(cusolverDnCreate(&value_), "create solver");
    check_solver(cusolverDnSetStream(value_, stream), "set solver stream");
    check_solver(cusolverDnSetDeterministicMode(
      value_, CUSOLVER_DETERMINISTIC_RESULTS), "set deterministic mode");
  }
  ~Solver() { if (value_ != nullptr) cusolverDnDestroy(value_); }
  cusolverDnHandle_t get() const { return value_; }
 private:
  cusolverDnHandle_t value_ = nullptr;
};

float elapsed(cudaEvent_t start, cudaEvent_t stop) {
  check_cuda(cudaEventSynchronize(stop), "wait for benchmark");
  float milliseconds = 0.0F;
  check_cuda(cudaEventElapsedTime(&milliseconds, start, stop),
             "read benchmark time");
  return milliseconds;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const int n = argc > 1 ? std::atoi(argv[1]) : 351;
    const int repeats = argc > 2 ? std::atoi(argv[2]) : 100;
    const int selected = argc > 3 ? std::atoi(argv[3]) : 35;
    if (n < 2 || repeats < 1 || selected < 1 || 2 * selected >= n) {
      throw std::runtime_error("invalid benchmark dimensions");
    }
    check_cuda(cudaSetDevice(0), "select device");
    Stream stream;
    Solver solver(stream.get());
    const std::size_t cells = static_cast<std::size_t>(n) * n;
    const std::size_t matrix_bytes = cells * sizeof(double);
    std::vector<double> values(static_cast<std::size_t>(n));
    std::vector<double> matrix(cells);
    for (int index = 0; index < n; ++index) {
      values[static_cast<std::size_t>(index)] =
        std::sin(0.37 * index) + std::cos(0.013 * index * index);
    }
    for (int column = 0; column < n; ++column) {
      for (int row = 0; row < n; ++row) {
        matrix[static_cast<std::size_t>(column) * n + row] = std::abs(
          values[static_cast<std::size_t>(row)] -
          values[static_cast<std::size_t>(column)]);
      }
    }

    double* source = nullptr;
    double* work_matrix = nullptr;
    double* eigenvalues = nullptr;
    int* info = nullptr;
    check_cuda(cudaMalloc(&source, matrix_bytes), "allocate source");
    check_cuda(cudaMalloc(&work_matrix, matrix_bytes), "allocate matrix");
    check_cuda(cudaMalloc(&eigenvalues,
                          static_cast<std::size_t>(n) * sizeof(double)),
               "allocate eigenvalues");
    check_cuda(cudaMalloc(&info, sizeof(int)), "allocate info");
    check_cuda(cudaMemcpyAsync(source, matrix.data(), matrix_bytes,
                               cudaMemcpyHostToDevice, stream.get()),
               "upload source");

    int full_lwork = 0;
    check_solver(cusolverDnDsyevd_bufferSize(
      solver.get(), CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
      n, work_matrix, n, eigenvalues, &full_lwork),
      "query full eig workspace");
    int partial_lwork = 0;
    int measured = 0;
    check_solver(cusolverDnDsyevdx_bufferSize(
      solver.get(), CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_I,
      CUBLAS_FILL_MODE_LOWER, n, work_matrix, n, 0.0, 0.0,
      1, selected, &measured, eigenvalues, &partial_lwork),
      "query partial eig workspace");
    const int lwork = std::max(full_lwork, partial_lwork);
    double* work = nullptr;
    check_cuda(cudaMalloc(&work,
                          static_cast<std::size_t>(lwork) * sizeof(double)),
               "allocate eig workspace");

    auto copy_source = [&]() {
      check_cuda(cudaMemcpyAsync(work_matrix, source, matrix_bytes,
                                 cudaMemcpyDeviceToDevice, stream.get()),
                 "copy source matrix");
    };
    auto full_solve = [&]() {
      copy_source();
      check_solver(cusolverDnDsyevd(
        solver.get(), CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        n, work_matrix, n, eigenvalues, work, lwork, info),
        "run full eig");
    };
    auto partial_solve = [&](int lower, int upper) {
      copy_source();
      check_solver(cusolverDnDsyevdx(
        solver.get(), CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_I,
        CUBLAS_FILL_MODE_LOWER, n, work_matrix, n, 0.0, 0.0,
        lower, upper, &measured, eigenvalues, work, lwork, info),
        "run partial eig");
    };

    for (int warmup = 0; warmup < 3; ++warmup) full_solve();
    check_cuda(cudaStreamSynchronize(stream.get()), "wait full warmup");
    Event full_start;
    Event full_stop;
    check_cuda(cudaEventRecord(full_start.get(), stream.get()),
               "record full start");
    for (int repeat = 0; repeat < repeats; ++repeat) full_solve();
    check_cuda(cudaEventRecord(full_stop.get(), stream.get()),
               "record full stop");
    const float full_ms = elapsed(full_start.get(), full_stop.get());

    for (int warmup = 0; warmup < 3; ++warmup) {
      partial_solve(1, selected);
      partial_solve(n - selected + 1, n);
    }
    check_cuda(cudaStreamSynchronize(stream.get()), "wait partial warmup");
    Event partial_start;
    Event partial_stop;
    check_cuda(cudaEventRecord(partial_start.get(), stream.get()),
               "record partial start");
    for (int repeat = 0; repeat < repeats; ++repeat) {
      partial_solve(1, selected);
      partial_solve(n - selected + 1, n);
    }
    check_cuda(cudaEventRecord(partial_stop.get(), stream.get()),
               "record partial stop");
    const float partial_ms = elapsed(partial_start.get(), partial_stop.get());

    int host_info = -1;
    check_cuda(cudaMemcpy(&host_info, info, sizeof(int),
                          cudaMemcpyDeviceToHost), "download solver info");
    const double full_per_component = full_ms / repeats;
    const double partial_per_component = partial_ms / repeats;
    std::printf(
      "n=%d repeats=%d selected=%d info=%d "
      "full_lwork=%d partial_lwork=%d "
      "full_total_ms=%.9g full_per_component_ms=%.9g "
      "two_sided_partial_total_ms=%.9g "
      "two_sided_partial_per_component_ms=%.9g "
      "full_110617_bound_ms=%.9g partial_110617_bound_ms=%.9g\n",
      n, repeats, selected, host_info, full_lwork, partial_lwork,
      static_cast<double>(full_ms), full_per_component,
      static_cast<double>(partial_ms), partial_per_component,
      full_per_component * 110617.0,
      partial_per_component * 110617.0);

    cudaFree(work);
    cudaFree(info);
    cudaFree(eigenvalues);
    cudaFree(work_matrix);
    cudaFree(source);
    return host_info == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return EXIT_FAILURE;
  }
}
