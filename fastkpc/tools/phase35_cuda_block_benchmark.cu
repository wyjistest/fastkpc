#include <cublas_v2.h>
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

void cuda_ok(cudaError_t status, const char* stage) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string(stage) + ": " + cudaGetErrorString(status));
  }
}

void blas_ok(cublasStatus_t status, const char* stage) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
      std::string(stage) + ": cuBLAS status " +
      std::to_string(static_cast<int>(status)));
  }
}

void solver_ok(cusolverStatus_t status, const char* stage) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error(
      std::string(stage) + ": cuSOLVER status " +
      std::to_string(static_cast<int>(status)));
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const int n = argc > 1 ? std::atoi(argv[1]) : 351;
    const int block = argc > 2 ? std::atoi(argv[2]) : 62;
    const int batch = argc > 3 ? std::atoi(argv[3]) : 47;
    const int iterations = argc > 4 ? std::atoi(argv[4]) : 12;
    const int repeats = argc > 5 ? std::atoi(argv[5]) : 20;
    if (n < block || block < 2 || batch < 1 || iterations < 1 ||
        repeats < 1) {
      throw std::runtime_error("invalid block benchmark dimensions");
    }
    cuda_ok(cudaSetDevice(0), "select device");
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
    cusolverDnHandle_t solver = nullptr;
    cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            "create stream");
    blas_ok(cublasCreate(&handle), "create cuBLAS");
    solver_ok(cusolverDnCreate(&solver), "create cuSOLVER");
    blas_ok(cublasSetStream(handle, stream), "set cuBLAS stream");
    solver_ok(cusolverDnSetStream(solver, stream), "set cuSOLVER stream");
    blas_ok(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH),
            "set pedantic math");
    blas_ok(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED),
            "disable atomics");

    const std::size_t d_stride = static_cast<std::size_t>(n) * n;
    const std::size_t q_stride = static_cast<std::size_t>(n) * block;
    const std::size_t g_stride = static_cast<std::size_t>(block) * block;
    std::vector<double> host_d(d_stride * batch);
    std::vector<double> host_q(q_stride * batch);
    for (int component = 0; component < batch; ++component) {
      for (int column = 0; column < n; ++column) {
        const double x_column = std::sin(
          0.013 * (column + 1) * (component + 1));
        for (int row = 0; row < n; ++row) {
          const double x_row = std::sin(
            0.013 * (row + 1) * (component + 1));
          host_d[static_cast<std::size_t>(component) * d_stride +
                 static_cast<std::size_t>(column) * n + row] =
            std::abs(x_row - x_column);
        }
      }
      for (int column = 0; column < block; ++column) {
        for (int row = 0; row < n; ++row) {
          host_q[static_cast<std::size_t>(component) * q_stride +
                 static_cast<std::size_t>(column) * n + row] =
            std::sin(0.754877666 * (row + 1) +
                     0.569840296 * (column + 1)) +
            std::cos(0.017453293 * (row + 1) * (column + 1));
        }
      }
    }

    double* d = nullptr;
    double* q = nullptr;
    double* z = nullptr;
    double* gram = nullptr;
    int* info = nullptr;
    double** z_pointers = nullptr;
    double** gram_pointers = nullptr;
    cuda_ok(cudaMalloc(&d, host_d.size() * sizeof(double)), "allocate D");
    cuda_ok(cudaMalloc(&q, host_q.size() * sizeof(double)), "allocate Q");
    cuda_ok(cudaMalloc(&z, host_q.size() * sizeof(double)), "allocate Z");
    cuda_ok(cudaMalloc(&gram, g_stride * batch * sizeof(double)),
            "allocate Gram");
    cuda_ok(cudaMalloc(&info, static_cast<std::size_t>(batch) * sizeof(int)),
            "allocate info");
    cuda_ok(cudaMalloc(&z_pointers,
                       static_cast<std::size_t>(batch) * sizeof(double*)),
            "allocate Z pointers");
    cuda_ok(cudaMalloc(&gram_pointers,
                       static_cast<std::size_t>(batch) * sizeof(double*)),
            "allocate Gram pointers");
    cuda_ok(cudaMemcpyAsync(d, host_d.data(), host_d.size() * sizeof(double),
                            cudaMemcpyHostToDevice, stream), "upload D");
    cuda_ok(cudaMemcpyAsync(q, host_q.data(), host_q.size() * sizeof(double),
                            cudaMemcpyHostToDevice, stream), "upload Q");
    std::vector<double*> host_z_pointers(batch);
    std::vector<double*> host_gram_pointers(batch);
    for (int component = 0; component < batch; ++component) {
      host_z_pointers[component] = z +
        static_cast<std::size_t>(component) * q_stride;
      host_gram_pointers[component] = gram +
        static_cast<std::size_t>(component) * g_stride;
    }
    cuda_ok(cudaMemcpyAsync(
      z_pointers, host_z_pointers.data(),
      static_cast<std::size_t>(batch) * sizeof(double*),
      cudaMemcpyHostToDevice, stream), "upload Z pointers");
    cuda_ok(cudaMemcpyAsync(
      gram_pointers, host_gram_pointers.data(),
      static_cast<std::size_t>(batch) * sizeof(double*),
      cudaMemcpyHostToDevice, stream), "upload Gram pointers");
    const double one = 1.0;
    const double zero = 0.0;

    auto sequence = [&]() {
      for (int iteration = 0; iteration < iterations; ++iteration) {
        blas_ok(cublasDgemmStridedBatched(
          handle, CUBLAS_OP_N, CUBLAS_OP_N, n, block, n,
          &one, d, n, static_cast<long long>(d_stride),
          q, n, static_cast<long long>(q_stride),
          &zero, z, n, static_cast<long long>(q_stride), batch),
          "multiply DQ");
        blas_ok(cublasDgemmStridedBatched(
          handle, CUBLAS_OP_T, CUBLAS_OP_N, block, block, n,
          &one, z, n, static_cast<long long>(q_stride),
          z, n, static_cast<long long>(q_stride),
          &zero, gram, block, static_cast<long long>(g_stride), batch),
          "form Gram");
        solver_ok(cusolverDnDpotrfBatched(
          solver, CUBLAS_FILL_MODE_LOWER, block, gram_pointers, block,
          info, batch), "factor Gram");
        blas_ok(cublasDtrsmBatched(
          handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
          CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, n, block, &one,
          const_cast<const double**>(gram_pointers), block,
          z_pointers, n, batch), "orthonormalize");
        std::swap(q, z);
        for (int component = 0; component < batch; ++component) {
          host_z_pointers[component] = z +
            static_cast<std::size_t>(component) * q_stride;
        }
        cuda_ok(cudaMemcpyAsync(
          z_pointers, host_z_pointers.data(),
          static_cast<std::size_t>(batch) * sizeof(double*),
          cudaMemcpyHostToDevice, stream), "refresh Z pointers");
      }
    };

    sequence();
    cuda_ok(cudaStreamSynchronize(stream), "warmup");
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cuda_ok(cudaEventCreate(&start), "create start");
    cuda_ok(cudaEventCreate(&stop), "create stop");
    cuda_ok(cudaEventRecord(start, stream), "record start");
    for (int repeat = 0; repeat < repeats; ++repeat) sequence();
    cuda_ok(cudaEventRecord(stop, stream), "record stop");
    cuda_ok(cudaEventSynchronize(stop), "wait benchmark");
    float total_ms = 0.0F;
    cuda_ok(cudaEventElapsedTime(&total_ms, start, stop), "read time");
    const double per_sequence = total_ms / repeats;
    const double per_component = per_sequence / batch;
    std::printf(
      "n=%d block=%d batch=%d iterations=%d repeats=%d "
      "sequence_ms=%.9g per_component_ms=%.9g "
      "bound_110617_ms=%.9g\n",
      n, block, batch, iterations, repeats, per_sequence, per_component,
      per_component * 110617.0);

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(gram_pointers);
    cudaFree(z_pointers);
    cudaFree(info);
    cudaFree(gram);
    cudaFree(z);
    cudaFree(q);
    cudaFree(d);
    cusolverDnDestroy(solver);
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return EXIT_FAILURE;
  }
}
