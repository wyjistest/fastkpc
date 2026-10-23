// Microbenchmark: same scalar Householder QR + explicit Q (dgeqr2 + dorg2r
// order) on many small FP64 matrices, three execution organizations:
//   A: one 64-thread CTA per target, thread 0 does all work (serial f2c style)
//   C: one 64-thread CTA per target, warp 0 cooperates over columns
//      (current production style for dlarf-type updates)
//   B: one lane per target, 32 targets per warp, interleaved layout
// All three use the identical per-element operation order, so results must be
// bitwise identical.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)

__device__ __host__ inline double init_value(int t, int i, int j) {
  unsigned int h = (unsigned int)t * 2654435761u ^ (unsigned int)(i * 40503 + j * 9973 + 17);
  h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
  return ((double)(h & 0xFFFFFF) / 16777216.0) - 0.5 + (i == j ? 2.0 : 0.0);
}

// Accessor abstraction: element (i,j) of target t.
struct AoS {
  double* base; int m, n;
  __device__ double& a(int t, int i, int j) const { return base[(size_t)t * m * n + i + (size_t)m * j]; }
  __device__ double& q(int t, int i, int j, double* qb) const { return qb[(size_t)t * m * n + i + (size_t)m * j]; }
};
struct Interleaved {
  double* base; int m, n; int T;
  __device__ double& a(int t, int i, int j) const { return base[((size_t)i + (size_t)m * j) * T + t]; }
  __device__ double& q(int t, int i, int j, double* qb) const { return qb[((size_t)i + (size_t)m * j) * T + t]; }
};

// Serial scalar dgeqr2 + dorg2r, element order identical in every variant.
template <class L>
__device__ void qr_serial(const L& L_, double* qb, double* tau, int t) {
  const int m = L_.m, n = L_.n;
  for (int j = 0; j < n; ++j) {
    double alpha = L_.a(t, j, j);
    double ss = 0.0;
    for (int i = j + 1; i < m; ++i) { double v = L_.a(t, i, j); ss = fma(v, v, ss); }
    double beta = -copysign(sqrt(fma(alpha, alpha, ss)), alpha);
    double tj = (beta - alpha) / beta;
    double scal = 1.0 / (alpha - beta);
    for (int i = j + 1; i < m; ++i) L_.a(t, i, j) *= scal;
    L_.a(t, j, j) = beta;
    tau[j] = tj;
    for (int l = j + 1; l < n; ++l) {
      double w = L_.a(t, j, l);
      for (int i = j + 1; i < m; ++i) w = fma(L_.a(t, i, j), L_.a(t, i, l), w);
      w *= tj;
      L_.a(t, j, l) -= w;
      for (int i = j + 1; i < m; ++i) L_.a(t, i, l) = fma(-w, L_.a(t, i, j), L_.a(t, i, l));
    }
  }
  // dorg2r: Q = H(0)...H(n-1) applied to I (backward accumulation)
  for (int j = n - 1; j >= 0; --j) {
    for (int l = j + 1; l < n; ++l) {
      double w = L_.q(t, j, l, qb);
      for (int i = j + 1; i < m; ++i) w = fma(L_.a(t, i, j), L_.q(t, i, l, qb), w);
      w *= tau[j];
      L_.q(t, j, l, qb) = -w + L_.q(t, j, l, qb);
      for (int i = j + 1; i < m; ++i) L_.q(t, i, l, qb) = fma(-w, L_.a(t, i, j), L_.q(t, i, l, qb));
    }
    L_.q(t, j, j, qb) = 1.0 - tau[j];
    for (int i = j + 1; i < m; ++i) L_.q(t, i, j, qb) = -tau[j] * L_.a(t, i, j);
    for (int i = 0; i < j; ++i) L_.q(t, i, j, qb) = 0.0;
  }
}

template <class L>
__global__ void init_kernel(L L_, double* qb, int T) {
  size_t total = (size_t)T * L_.m * L_.n;
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < total;
       idx += (size_t)gridDim.x * blockDim.x) {
    int t = (int)(idx % T); size_t e = idx / T; int i = (int)(e % L_.m); int j = (int)(e / L_.m);
    L_.a(t, i, j) = init_value(t, i, j);
    L_.q(t, i, j, qb) = 0.0;
  }
}

// A: serial thread 0 per CTA (64 threads, 63 idle)
__global__ void __launch_bounds__(64) qr_cta_serial(AoS L_, double* qb, double* taub) {
  int t = blockIdx.x;
  if (threadIdx.x == 0) qr_serial(L_, qb, taub + (size_t)t * L_.n, t);
  __syncthreads();
}

// C: warp-cooperative over columns, same element order per column
__global__ void __launch_bounds__(64) qr_cta_coop(AoS L_, double* qb, double* taub) {
  int t = blockIdx.x; const int m = L_.m, n = L_.n;
  double* tau = taub + (size_t)t * n;
  __shared__ double s_tau, s_scal, s_beta;
  int lane = threadIdx.x;
  for (int j = 0; j < n; ++j) {
    if (lane == 0) {
      double alpha = L_.a(t, j, j); double ss = 0.0;
      for (int i = j + 1; i < m; ++i) { double v = L_.a(t, i, j); ss = fma(v, v, ss); }
      double beta = -copysign(sqrt(fma(alpha, alpha, ss)), alpha);
      s_tau = (beta - alpha) / beta; s_scal = 1.0 / (alpha - beta); s_beta = beta;
      tau[j] = s_tau;
    }
    __syncthreads();
    for (int i = j + 1 + lane; i < m; i += blockDim.x) L_.a(t, i, j) *= s_scal;
    __syncthreads();
    if (lane == 0) L_.a(t, j, j) = s_beta;
    double tj = s_tau;
    for (int l = j + 1 + lane; l < n; l += blockDim.x) {
      double w = L_.a(t, j, l);
      for (int i = j + 1; i < m; ++i) w = fma(L_.a(t, i, j), L_.a(t, i, l), w);
      w *= tj;
      L_.a(t, j, l) -= w;
      for (int i = j + 1; i < m; ++i) L_.a(t, i, l) = fma(-w, L_.a(t, i, j), L_.a(t, i, l));
    }
    __syncthreads();
  }
  for (int j = n - 1; j >= 0; --j) {
    double tj = tau[j];
    for (int l = j + 1 + lane; l < n; l += blockDim.x) {
      double w = L_.q(t, j, l, qb);
      for (int i = j + 1; i < m; ++i) w = fma(L_.a(t, i, j), L_.q(t, i, l, qb), w);
      w *= tj;
      L_.q(t, j, l, qb) = -w + L_.q(t, j, l, qb);
      for (int i = j + 1; i < m; ++i) L_.q(t, i, l, qb) = fma(-w, L_.a(t, i, j), L_.q(t, i, l, qb));
    }
    __syncthreads();
    if (lane == 0) {
      L_.q(t, j, j, qb) = 1.0 - tj;
      for (int i = j + 1; i < m; ++i) L_.q(t, i, j, qb) = -tj * L_.a(t, i, j);
      for (int i = 0; i < j; ++i) L_.q(t, i, j, qb) = 0.0;
    }
    __syncthreads();
  }
}

// B: one lane per target, interleaved layout
__global__ void __launch_bounds__(128) qr_lane_per_target(Interleaved L_, double* qb, double* taub, int T) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  // tau kept interleaved too
  double tau_local[80];
  qr_serial(L_, qb, tau_local, t);
  for (int j = 0; j < L_.n; ++j) taub[(size_t)j * T + t] = tau_local[j];
}

static double run_variant(int variant, int m, int n, int T, std::vector<double>* outR, std::vector<double>* outQ) {
  size_t elems = (size_t)T * m * n;
  double *a, *q, *tau;
  CK(cudaMalloc(&a, elems * 8)); CK(cudaMalloc(&q, elems * 8)); CK(cudaMalloc(&tau, (size_t)T * n * 8));
  AoS aos{a, m, n}; Interleaved il{a, m, n, T};
  if (variant == 2) init_kernel<<<4096, 256>>>(il, q, T); else init_kernel<<<4096, 256>>>(aos, q, T);
  CK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  if (variant == 0) qr_cta_serial<<<T, 64>>>(aos, q, tau);
  else if (variant == 1) qr_cta_coop<<<T, 64>>>(aos, q, tau);
  else qr_lane_per_target<<<(T + 127) / 128, 128>>>(il, q, tau, T);
  cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
  float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
  // gather a sample of targets in canonical order for bitwise comparison
  std::vector<double> ha(elems), hq(elems);
  CK(cudaMemcpy(ha.data(), a, elems * 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hq.data(), q, elems * 8, cudaMemcpyDeviceToHost));
  int sample = T < 4096 ? T : 4096;
  outR->assign((size_t)sample * m * n, 0); outQ->assign((size_t)sample * m * n, 0);
  for (int t = 0; t < sample; ++t)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i) {
        size_t src = variant == 2 ? ((size_t)i + (size_t)m * j) * T + t : (size_t)t * m * n + i + (size_t)m * j;
        size_t dst = (size_t)t * m * n + i + (size_t)m * j;
        (*outR)[dst] = ha[src]; (*outQ)[dst] = hq[src];
      }
  cudaFree(a); cudaFree(q); cudaFree(tau);
  return ms;
}

int main(int argc, char** argv) {
  int shapes[][2] = {{52, 28}, {88, 37}, {120, 64}};
  int T = argc > 1 ? atoi(argv[1]) : 32768;
  const char* names[] = {"A cta-serial(thread0)", "C cta-warp-coop     ", "B lane-per-target   "};
  for (auto& s : shapes) {
    int m = s[0], n = s[1];
    double flops = 2.0 * (2.0 * m * n * n - 2.0 * n * n * n / 3.0); // QR + Q gen approx
    std::vector<double> R0, Q0, R, Q;
    printf("shape %dx%d, targets %d\n", m, n, T);
    double base = 0;
    for (int v = 0; v < 3; ++v) {
      run_variant(v, m, n, T, &R, &Q);              // warm-up
      double ms = run_variant(v, m, n, T, &R, &Q);
      if (v == 0) { R0 = R; Q0 = Q; base = ms; }
      bool same = (memcmp(R.data(), R0.data(), R.size() * 8) == 0) && (memcmp(Q.data(), Q0.data(), Q.size() * 8) == 0);
      printf("  %s %9.3f ms  %7.1f GFLOP/s  speedup vs A %6.2fx  bitwise==A %s\n", names[v], ms,
             flops * T / (ms * 1e6), base / ms, same ? "yes" : "NO");
    }
  }
  return 0;
}
