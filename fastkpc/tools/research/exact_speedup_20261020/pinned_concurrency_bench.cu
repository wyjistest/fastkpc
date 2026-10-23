// Mimics the optimizer window: T host threads, each owns a stream and runs
// many "setups": H2D upload (small), one small kernel (~few CTAs, ~tens of ms),
// D2H download, then the next setup.  Compares pageable vs pinned host buffers.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA %s: %s\n", #x, cudaGetErrorString(e)); exit(1);} } while (0)

__global__ void busy_kernel(double* out, const double* in, long iters) {
  double v = in[blockIdx.x] + threadIdx.x;
  for (long i = 0; i < iters; ++i) v = fma(v, 1.0000001, 1e-9);   // dependent FP64 chain
  if (threadIdx.x == 0) out[blockIdx.x] = v;
}

int main(int argc, char** argv) {
  const int threads = argc > 1 ? atoi(argv[1]) : 47;
  const int setups_per_thread = argc > 2 ? atoi(argv[2]) : 6;
  const long iters = argc > 3 ? atol(argv[3]) : 200000;
  const int blocks = 8, block_threads = 64;
  const size_t up_bytes = 351 * 11 * sizeof(double), down_bytes = 64 * 1024;
  CK(cudaFree(0));
  for (int pinned = 0; pinned <= 1; ++pinned) {
    std::atomic<long long> kernel_ns{0};
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) {
      pool.emplace_back([&, t]() {
        cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        double *din, *dout; CK(cudaMalloc(&din, up_bytes)); CK(cudaMalloc(&dout, down_bytes));
        char *hup, *hdown;
        std::vector<char> pu(up_bytes, 1), pd(down_bytes);
        if (pinned) { CK(cudaMallocHost(&hup, up_bytes)); CK(cudaMallocHost(&hdown, down_bytes)); }
        else { hup = pu.data(); hdown = pd.data(); }
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        for (int k = 0; k < setups_per_thread; ++k) {
          CK(cudaMemcpyAsync(din, hup, up_bytes, cudaMemcpyHostToDevice, s));
          CK(cudaEventRecord(a, s));
          busy_kernel<<<blocks, block_threads, 0, s>>>(dout, din, iters);
          CK(cudaEventRecord(b, s));
          CK(cudaMemcpyAsync(hdown, dout, down_bytes, cudaMemcpyDeviceToHost, s));
          CK(cudaStreamSynchronize(s));
          float ms = 0; CK(cudaEventElapsedTime(&ms, a, b)); kernel_ns += (long long)(ms * 1e6);
        }
        if (pinned) { cudaFreeHost(hup); cudaFreeHost(hdown); }
        cudaFree(din); cudaFree(dout); cudaStreamDestroy(s);
      });
    }
    for (auto& th : pool) th.join();
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double ksum = kernel_ns.load() / 1e9;
    printf("%-8s host threads %d, setups %d: wall %.2f s, summed kernel time %.2f s, "
           "average concurrent kernels %.1f (resident CTAs ~%.0f)\n",
           pinned ? "pinned" : "pageable", threads, threads * setups_per_thread, wall, ksum,
           ksum / wall, blocks * ksum / wall);
  }
  return 0;
}
