// Replays captured coefficients only: no optimizer, decomposition, or CPU fit.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
void cuda_ok(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void blas_ok(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS failed");
}
struct Device {
  double* p = nullptr;
  explicit Device(std::size_t n) { cuda_ok(cudaMalloc(&p, n * sizeof(double))); }
  ~Device() { if (p) cudaFree(p); }
};
struct Context {
  cudaStream_t stream = nullptr;
  cublasHandle_t blas = nullptr;
  Context(int math, int atomics) {
    cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    blas_ok(cublasCreate(&blas));
    blas_ok(cublasSetStream(blas, stream));
    blas_ok(cublasSetMathMode(blas, cublasMath_t(math)));
    blas_ok(cublasSetAtomicsMode(blas, cublasAtomicsMode_t(atomics)));
  }
  ~Context() { if (blas) cublasDestroy(blas); if (stream) cudaStreamDestroy(stream); }
};
__global__ void residual_kernel(const double* Y, const double* fitted,
                                double* residual, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) residual[i] = __dadd_rn(Y[i], -fitted[i]);
}
}
extern "C" int replay_frozen(int n, int q, int targets, int math, int atomics,
    const double* X, const double* coefficients, const double* Y,
    double* fitted, double* residual) {
  try {
    Context context(math, atomics);
    Device dx(std::size_t(n)*q), dc(std::size_t(q)*targets),
           dy(std::size_t(n)*targets), df(std::size_t(n)*targets),
           dr(std::size_t(n)*targets);
    const std::size_t bytes = std::size_t(n)*targets*sizeof(double);
    cuda_ok(cudaMemcpyAsync(dx.p, X, std::size_t(n)*q*sizeof(double),
                            cudaMemcpyHostToDevice, context.stream));
    cuda_ok(cudaMemcpyAsync(dc.p, coefficients, std::size_t(q)*targets*sizeof(double),
                            cudaMemcpyHostToDevice, context.stream));
    cuda_ok(cudaMemcpyAsync(dy.p, Y, bytes, cudaMemcpyHostToDevice, context.stream));
    const double alpha = 1.0, beta = 0.0;
    blas_ok(cublasDgemm(context.blas, CUBLAS_OP_N, CUBLAS_OP_N,
      n, targets, q, &alpha, dx.p, n, dc.p, q, &beta, df.p, n));
    residual_kernel<<<(n*targets+255)/256,256,0,context.stream>>>(
      dy.p, df.p, dr.p, n*targets);
    cuda_ok(cudaGetLastError());
    cuda_ok(cudaMemcpyAsync(fitted, df.p, bytes, cudaMemcpyDeviceToHost, context.stream));
    cuda_ok(cudaMemcpyAsync(residual, dr.p, bytes, cudaMemcpyDeviceToHost, context.stream));
    cuda_ok(cudaStreamSynchronize(context.stream));
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "frozen replay failed: %s\n", error.what());
    return 1;
  }
}
