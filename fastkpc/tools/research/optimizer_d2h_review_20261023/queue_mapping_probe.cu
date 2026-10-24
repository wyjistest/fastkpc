// Which streams share stream j's hardware queue? Streams are created consecutively;
// optionally each is first used in reverse order. A 200 ms kernel goes on stream j,
// followed by a dependent tiny kernel on the same stream (the queue must wait for
// it), then a 1 ms kernel on every other stream; streams whose short kernel finishes
// after ~200 ms share j's queue.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } } while (0)
__global__ void spin(unsigned long long ns) {
  unsigned long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < ns);
}
int main(int argc, char** argv) {
  const int N = std::atoi(argv[1]);
  const std::string first_use = argv[2];  // "forward" | "reverse" | "none"
  CK(cudaSetDevice(0)); CK(cudaFree(nullptr));
  std::vector<cudaStream_t> st(N);
  for (auto& s : st) CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  if (first_use != "none")
    for (int k = 0; k < N; ++k) spin<<<1, 32, 0, st[first_use == "reverse" ? N - 1 - k : k]>>>(1000);
  CK(cudaDeviceSynchronize());
  for (int j : {0, 1, 5}) {
    cudaEvent_t e0; std::vector<cudaEvent_t> done(N);
    CK(cudaEventCreate(&e0)); for (auto& x : done) CK(cudaEventCreate(&x));
    CK(cudaEventRecord(e0, st[j])); spin<<<1, 32, 0, st[j]>>>(200000000ULL); spin<<<1, 32, 0, st[j]>>>(1000ULL);
    for (int k = 0; k < N; ++k) if (k != j) { spin<<<1, 32, 0, st[k]>>>(1000000ULL); CK(cudaEventRecord(done[k], st[k])); }
    CK(cudaDeviceSynchronize());
    std::printf("N=%d first_use=%s long on %d -> blocked:", N, first_use.c_str(), j);
    for (int k = 0; k < N; ++k) if (k != j) { float ms; CK(cudaEventElapsedTime(&ms, e0, done[k])); if (ms > 100) std::printf(" %d", k); }
    std::printf("\n");
  }
  return 0;
}
