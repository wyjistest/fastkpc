// Rolling refill: 32 streams in flight (1 s kernel + dependent tiny kernel each,
// except stream 5 which runs 10 ms). After stream 5 drains, submit on a new
// stream (fresh) or on stream 5 again (reused, like a worker-owned stream).
// Is the new work placed behind a busy 1 s queue? Then a 33rd concurrent
// submission is made with all 32 queues busy (expected to share).
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#define CK(x) do { cudaError_t err_ = (x); if (err_ != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); std::exit(1); } } while (0)
__global__ void spin(unsigned long long ns) {
  unsigned long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < ns);
}
int main() {
  CK(cudaSetDevice(0)); CK(cudaFree(nullptr));
  std::vector<cudaStream_t> st(40);
  for (auto& s : st) CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  cudaEvent_t e0, a, b; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  for (std::string mode : {"fresh", "reused", "fresh", "reused"}) {
    CK(cudaEventRecord(e0, st[0]));
    for (int k = 0; k < 32; ++k) {
      spin<<<1, 32, 0, st[k]>>>(k == 5 ? 10000000ULL : 1000000000ULL); spin<<<1, 32, 0, st[k]>>>(1000ULL);
    }
    CK(cudaStreamSynchronize(st[5]));
    cudaStream_t refill = mode == "fresh" ? st[32] : st[5];
    spin<<<1, 32, 0, refill>>>(1000000ULL); spin<<<1, 32, 0, refill>>>(1000ULL); CK(cudaEventRecord(a, refill));
    spin<<<1, 32, 0, st[33]>>>(1000000ULL); CK(cudaEventRecord(b, st[33]));
    CK(cudaDeviceSynchronize());
    float ma, mb; CK(cudaEventElapsedTime(&ma, e0, a)); CK(cudaEventElapsedTime(&mb, e0, b));
    std::printf("refill on %-6s stream: done at %7.1f ms | extra 33rd stream: done at %7.1f ms\n", mode.c_str(), ma, mb);
  }
  return 0;
}
