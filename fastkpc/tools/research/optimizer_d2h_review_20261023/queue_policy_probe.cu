// (A) Assignment policy: stream 0 runs a 1 s kernel + dependent tiny kernel; streams
//     1..31 run and finish short kernels (their queues go idle); then streams 32..63
//     submit short kernels. Blocked ones were placed on stream 0's busy queue.
// (B) What blocks a shared queue: stream 0 runs a 1 s kernel followed by X, then
//     streams 1..63 submit short kernels (stream 32 is stream 0's queue partner).
//     X = none | event (cudaEventRecord) | blockevent (blocking-sync event record)
//         | kernel (dependent kernel) | d2h_pinned | sync (host cudaStreamSynchronize
//         issued from another thread, no command).
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
#define CK(x) do { cudaError_t err_ = (x); if (err_ != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); std::exit(1); } } while (0)
__global__ void spin(unsigned long long ns) {
  unsigned long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < ns);
}
const int N = 64;
std::vector<cudaStream_t> st(N);
void report(const char* label, cudaEvent_t e0, std::vector<cudaEvent_t>& done, int from) {
  CK(cudaDeviceSynchronize());
  std::printf("%-40s blocked:", label);
  for (int k = from; k < N; ++k) { float ms; CK(cudaEventElapsedTime(&ms, e0, done[k])); if (ms > 500) std::printf(" %d", k); }
  std::printf("\n");
}
int main() {
  CK(cudaSetDevice(0)); CK(cudaFree(nullptr));
  for (auto& s : st) CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  std::vector<cudaEvent_t> done(N); for (auto& x : done) CK(cudaEventCreate(&x));
  cudaEvent_t e0, ev, bev; CK(cudaEventCreate(&e0));
  CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&bev, cudaEventDisableTiming | cudaEventBlockingSync));
  double* pinned; double* dev; CK(cudaMallocHost(&pinned, 4096)); CK(cudaMalloc(&dev, 4096));
  // (A)
  for (int rep = 0; rep < 2; ++rep) {
    CK(cudaEventRecord(e0, st[0])); spin<<<1, 32, 0, st[0]>>>(1000000000ULL); spin<<<1, 32, 0, st[0]>>>(1000ULL);
    for (int k = 1; k < 32; ++k) spin<<<1, 32, 0, st[k]>>>(1000000ULL);
    for (int k = 1; k < 32; ++k) CK(cudaStreamSynchronize(st[k]));
    for (int k = 32; k < N; ++k) { spin<<<1, 32, 0, st[k]>>>(1000000ULL); CK(cudaEventRecord(done[k], st[k])); }
    report("(A) queues 1..31 idle, submit 32..63", e0, done, 32);
  }
  // (B)
  for (std::string x : {"none", "event", "blockevent", "kernel", "d2h_pinned", "sync"}) {
    CK(cudaEventRecord(e0, st[0])); spin<<<1, 32, 0, st[0]>>>(1000000000ULL);
    std::thread waiter;
    if (x == "event") CK(cudaEventRecord(ev, st[0]));
    if (x == "blockevent") CK(cudaEventRecord(bev, st[0]));
    if (x == "kernel") spin<<<1, 32, 0, st[0]>>>(1000ULL);
    if (x == "d2h_pinned") CK(cudaMemcpyAsync(pinned, dev, 4096, cudaMemcpyDeviceToHost, st[0]));
    if (x == "sync") waiter = std::thread([] { CK(cudaStreamSynchronize(st[0])); });
    for (int k = 1; k < N; ++k) { spin<<<1, 32, 0, st[k]>>>(1000000ULL); CK(cudaEventRecord(done[k], st[k])); }
    if (waiter.joinable()) waiter.join();
    report(("(B) 1 s kernel then " + x).c_str(), e0, done, 1);
  }
  return 0;
}
