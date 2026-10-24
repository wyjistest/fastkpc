// Does a device->pageable cudaMemcpyAsync queued behind a long kernel on its own
// stream block other threads' pageable D2H on other streams? Thread 0 runs a long
// kernel and enters its D2H first; threads 1..T-1 run short kernels.
//   P: launch; pageable D2H; stream sync        (production pattern)
//   S: launch; stream sync; pageable D2H; sync  (wait first, then copy)
//   E: launch; blocking-sync event wait; pageable D2H; sync
//   H: launch; pinned D2H; stream sync
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } } while (0)

__global__ void spin(unsigned long long ns, double* out) {
  unsigned long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < ns);
  out[threadIdx.x] = double(t - t0);
}

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

int main(int argc, char** argv) {
  const int T = argc > 1 ? std::atoi(argv[1]) : 8;
  const double long_ms = argc > 2 ? std::atof(argv[2]) : 1000, short_ms = argc > 3 ? std::atof(argv[3]) : 50;
  const int reps = argc > 4 ? std::atoi(argv[4]) : 3;
  const size_t n = 768;  // 6 KiB, same order as one setup's optimizer state download
  CK(cudaSetDevice(0)); CK(cudaFree(nullptr));
  std::vector<cudaStream_t> st(T); std::vector<cudaEvent_t> ev(T);
  std::vector<double*> dev(T), pinned(T); std::vector<std::vector<double>> pageable(T, std::vector<double>(n));
  for (int i = 0; i < T; ++i) {
    CK(cudaStreamCreateWithFlags(&st[i], cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&ev[i], cudaEventBlockingSync | cudaEventDisableTiming));
    CK(cudaMalloc(&dev[i], n * sizeof(double))); CK(cudaMallocHost(&pinned[i], n * sizeof(double)));
  }
  std::printf("threads=%d long_ms=%.0f short_ms=%.0f (thread 0 long, enters D2H first; others start 20 ms later)\n",
              T, long_ms, short_ms);
  for (char v : std::string("PSEH")) {
    for (int r = -1; r < reps; ++r) {  // r = -1 is warmup
      std::vector<double> ret(T); std::atomic<int> ready{0}; std::atomic<bool> go{false};
      Clock::time_point t0;
      auto body = [&](int i) {
        ready.fetch_add(1); while (!go.load()) {}
        if (i) std::this_thread::sleep_for(std::chrono::milliseconds(20));
        spin<<<1, 64, 0, st[i]>>>((unsigned long long)((i ? short_ms : long_ms) * 1e6), dev[i]);
        CK(cudaGetLastError());
        double* dst = v == 'H' ? pinned[i] : pageable[i].data();
        if (v == 'S') CK(cudaStreamSynchronize(st[i]));
        if (v == 'E') { CK(cudaEventRecord(ev[i], st[i])); CK(cudaEventSynchronize(ev[i])); }
        CK(cudaMemcpyAsync(dst, dev[i], n * sizeof(double), cudaMemcpyDeviceToHost, st[i]));
        CK(cudaStreamSynchronize(st[i]));
        ret[i] = ms_since(t0);
      };
      std::vector<std::thread> th;
      for (int i = 0; i < T; ++i) th.emplace_back(body, i);
      while (ready.load() < T) {}
      t0 = Clock::now(); go = true;
      for (auto& t : th) t.join();
      if (r < 0) continue;
      std::vector<double> s(ret.begin() + 1, ret.end()); std::sort(s.begin(), s.end());
      std::printf("%c rep%d  long thread %.1f ms | short threads median %.1f max %.1f ms\n",
                  v, r, ret[0], s[s.size() / 2], s.back());
    }
  }
  return 0;
}
