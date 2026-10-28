// Does a launch that grows the context's local-memory reservation hold back kernels
// launched after it on other streams until the kernels then running have drained?
// And does pre-sizing cudaLimitStackSize at start-up remove that serialization?
//
// ./lmem_growth_probe 0|1 [scenario 0-5]   (1 = cudaDeviceSetLimit(cudaLimitStackSize, 8192) first)
// Each kernel records its own %globaltimer start/end; times are relative to kernel 0.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

__device__ __forceinline__ unsigned long long now_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__global__ void zero_stack(double* sink, int id) {
  if (threadIdx.x == 0 && blockIdx.x == 0) sink[id] += 1.0;
}

template <int BYTES>
__global__ void spin_with_stack(unsigned long long* stamps, double* sink,
                                unsigned long long duration_ns, int id) {
  volatile double frame[BYTES / sizeof(double)];
  const unsigned long long start = now_ns();
  for (int i = 0; i < BYTES / (int)sizeof(double); i += 16) frame[i] = id + i;
  double acc = 0;
  int k = 0;
  while (now_ns() - start < duration_ns) {
    acc += frame[(k * 16) % (BYTES / (int)sizeof(double))];
    ++k;
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    stamps[2 * id] = start;
    stamps[2 * id + 1] = now_ns();
    sink[id] = acc;
  }
}

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  std::printf("CUDA %s at %d\n", cudaGetErrorString(e), __LINE__); std::exit(1); } } while (0)

int main(int argc, char** argv) {
  const bool presize = argc > 1 && argv[1][0] == '1';
  CHECK(cudaSetDevice(0));
  CHECK(cudaFree(nullptr));
  if (presize) CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));
  size_t limit = 0, free_before = 0, total = 0;
  CHECK(cudaDeviceGetLimit(&limit, cudaLimitStackSize));
  CHECK(cudaMemGetInfo(&free_before, &total));
  unsigned long long* stamps;
  double* sink;
  CHECK(cudaMalloc(&stamps, 64 * sizeof(unsigned long long)));
  CHECK(cudaMalloc(&sink, 64 * sizeof(double)));
  const int n = 6;
  std::vector<cudaStream_t> s(n);
  for (auto& x : s) CHECK(cudaStreamCreateWithFlags(&x, cudaStreamNonBlocking));
  const unsigned long long ms = 1000000ULL;
  const int scenario = argc > 2 ? std::atoi(argv[2]) : 0;
  if (scenario == 0) {
    // 0: 4 KiB stack, 400 ms (first growth). 1-2: 4 KiB, 50 ms, launched right after.
    // 3: 6 KiB, 50 ms (second growth while 0 still runs). 4-5: 4 KiB, 50 ms, after 3.
    spin_with_stack<4096><<<2, 64, 0, s[0]>>>(stamps, sink, 400 * ms, 0);
    spin_with_stack<4096><<<2, 64, 0, s[1]>>>(stamps, sink, 50 * ms, 1);
    spin_with_stack<4096><<<2, 64, 0, s[2]>>>(stamps, sink, 50 * ms, 2);
    spin_with_stack<6144><<<2, 64, 0, s[3]>>>(stamps, sink, 50 * ms, 3);
    spin_with_stack<4096><<<2, 64, 0, s[4]>>>(stamps, sink, 50 * ms, 4);
    spin_with_stack<4096><<<2, 64, 0, s[5]>>>(stamps, sink, 50 * ms, 5);
  } else {
    // 1/2/3: context already ran a 4 KiB kernel and is idle; then a 6 KiB kernel (growth)
    // runs 400 ms on stream 0 and five more 6 KiB kernels follow at once. Before each
    // follower: 1 nothing, 2 a 1 KiB-stack kernel, 3 a zero-stack kernel (like the
    // production norm/QR kernels). 4: like 3 but the warm-up already used 6 KiB, so
    // there is no growth (the L4/L5 slot steady state). 5: like 3 after preloading the
    // zero-stack function with cudaFuncGetAttributes.
    if (scenario == 5) {
      // 5: like 3, but the zero-stack function is preloaded before any concurrency.
      cudaFuncAttributes attributes;
      CHECK(cudaFuncGetAttributes(&attributes, zero_stack));
    }
    if (scenario == 4) spin_with_stack<6144><<<2, 64, 0, s[0]>>>(stamps, sink, 20 * ms, 10);
    else spin_with_stack<4096><<<2, 64, 0, s[0]>>>(stamps, sink, 20 * ms, 10);
    CHECK(cudaDeviceSynchronize());
    spin_with_stack<6144><<<2, 64, 0, s[0]>>>(stamps, sink, 400 * ms, 0);
    for (int i = 1; i < n; ++i) {
      if (scenario == 2) spin_with_stack<1024><<<2, 64, 0, s[i]>>>(stamps, sink, 1 * ms, 10 + i);
      if (scenario == 3 || scenario == 4 || scenario == 5) zero_stack<<<2, 64, 0, s[i]>>>(sink, 20 + i);
      spin_with_stack<6144><<<2, 64, 0, s[i]>>>(stamps, sink, 50 * ms, i);
    }
  }
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  size_t free_after = 0;
  CHECK(cudaMemGetInfo(&free_after, &total));
  unsigned long long h[2 * n];
  CHECK(cudaMemcpy(h, stamps, sizeof(h), cudaMemcpyDeviceToHost));
  std::printf("presize=%d stack_limit=%zu free_drop_during_launches=%.1f MiB\n",
              presize ? 1 : 0, limit, (double(free_before) - double(free_after)) / 1048576.0);
  for (int i = 0; i < n; ++i) {
    std::printf("kernel %d stack %s: start %+8.2f ms end %+8.2f ms\n", i,
                (scenario == 0 ? (i == 3 ? "6K" : "4K") : "6K"), (double(h[2 * i]) - double(h[0])) / 1e6,
                (double(h[2 * i + 1]) - double(h[0])) / 1e6);
  }
  return 0;
}
