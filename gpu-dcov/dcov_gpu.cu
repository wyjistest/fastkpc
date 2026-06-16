// dcov_gpu.cu — GPU implementation of the distance-covariance gamma test statistics
// used by kpcalg::dcov.gamma (R/dcovgamma.R).
//
// Mathematical identity exploited (full-rank equivalence, no eigendecomposition):
//   nV^2          = sum(A o B) / n
//   nV^2 mean     = mean(K) * mean(L)
//   nV^2 variance = 2(n-4)(n-5)/(n(n-1)(n-2)(n-3)) * sum(A o A) * sum(B o B) / n^2
// where K, L are the pairwise distance matrices of x and y, and A = HKH,
// B = HLH their double-centered versions:
//   a_ij = k_ij - rowmean_i(K) - rowmean_j(K) + grandmean(K)
//
// Nothing n x n is ever materialized: distances are recomputed on the fly
// (O(n) memory), all accumulation is FP64.
//
// Exposed to R via .Call:
//   C_dcov_gpu_stats(x, y, index) -> c(sum(AoB), sum(AoA), sum(BoB), sum(K), sum(L))
//   C_dcov_gpu_warmup()           -> initializes the CUDA context (one-time cost)

#include <R.h>
#include <Rinternals.h>
#include <cuda_runtime.h>

#define BLOCK 256

// distance between observations i and j of a column-major n x d matrix,
// raised to `index` (kpcalg documents K_ij = ||x_i - x_j||^index)
__device__ __forceinline__ double pair_dist(const double *v, int n, int d,
                                            int i, int j, double index) {
  double dist;
  if (d == 1) {
    dist = fabs(v[i] - v[j]);
  } else {
    double s = 0.0;
    for (int k = 0; k < d; ++k) {
      double diff = v[i + (size_t)k * n] - v[j + (size_t)k * n];
      s += diff * diff;
    }
    dist = sqrt(s);
  }
  if (index != 1.0) dist = pow(dist, index);
  return dist;
}

// one block per row: rowsum[i] = sum_j dist(i,j); total += rowsum[i]
__global__ void rowsum_kernel(const double *v, int n, int d, double index,
                              double *rowsum, double *total) {
  __shared__ double sdata[BLOCK];
  int i = blockIdx.x;
  double acc = 0.0;
  for (int j = threadIdx.x; j < n; j += blockDim.x)
    acc += pair_dist(v, n, d, i, j, index);
  sdata[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    rowsum[i] = sdata[0];
    atomicAdd(total, sdata[0]);
  }
}

// fused double-centering + three reductions over all (i,j) pairs
__global__ void fused_center_reduce(const double *x, const double *y, int n,
                                    int dx, int dy, double index,
                                    const double *rsK, const double *rsL,
                                    const double *totK, const double *totL,
                                    double *out /* Sab, Saa, Sbb */) {
  __shared__ double sab_s[BLOCK], saa_s[BLOCK], sbb_s[BLOCK];
  const double inv_n = 1.0 / n;
  const double gK = *totK * inv_n * inv_n;
  const double gL = *totL * inv_n * inv_n;
  double sab = 0.0, saa = 0.0, sbb = 0.0;
  for (int i = blockIdx.y; i < n; i += gridDim.y) {
    const double cKi = gK - rsK[i] * inv_n;  // grand mean minus row mean, hoisted
    const double cLi = gL - rsL[i] * inv_n;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += gridDim.x * blockDim.x) {
      double a = pair_dist(x, n, dx, i, j, index) - rsK[j] * inv_n + cKi;
      double b = pair_dist(y, n, dy, i, j, index) - rsL[j] * inv_n + cLi;
      sab += a * b;
      saa += a * a;
      sbb += b * b;
    }
  }
  sab_s[threadIdx.x] = sab;
  saa_s[threadIdx.x] = saa;
  sbb_s[threadIdx.x] = sbb;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sab_s[threadIdx.x] += sab_s[threadIdx.x + s];
      saa_s[threadIdx.x] += saa_s[threadIdx.x + s];
      sbb_s[threadIdx.x] += sbb_s[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(&out[0], sab_s[0]);
    atomicAdd(&out[1], saa_s[0]);
    atomicAdd(&out[2], sbb_s[0]);
  }
}

extern "C" {

SEXP C_dcov_gpu_warmup(void) {
  cudaError_t err = cudaFree(0);
  if (err != cudaSuccess)
    Rf_error("CUDA init failed: %s", cudaGetErrorString(err));
  return R_NilValue;
}

SEXP C_dcov_gpu_stats(SEXP xs, SEXP ys, SEXP indexs) {
  if (!Rf_isReal(xs) || !Rf_isReal(ys))
    Rf_error("x and y must be numeric (double)");
  int n, dx, dy;
  if (Rf_isMatrix(xs)) { n = Rf_nrows(xs); dx = Rf_ncols(xs); }
  else                 { n = Rf_length(xs); dx = 1; }
  int ny;
  if (Rf_isMatrix(ys)) { ny = Rf_nrows(ys); dy = Rf_ncols(ys); }
  else                 { ny = Rf_length(ys); dy = 1; }
  if (n != ny) Rf_error("Sample sizes must agree (%d vs %d)", n, ny);
  if (n < 2) Rf_error("Need at least 2 observations");
  double index = Rf_asReal(indexs);

  double *d_x = NULL, *d_y = NULL, *d_rsK = NULL, *d_rsL = NULL, *d_sc = NULL;
  cudaError_t err = cudaSuccess;
  const char *stage = "";

#define CK(call, what)                          \
  do {                                          \
    if (err == cudaSuccess) {                   \
      stage = what;                             \
      err = (call);                             \
    }                                           \
  } while (0)

  CK(cudaMalloc(&d_x, sizeof(double) * n * dx), "alloc x");
  CK(cudaMalloc(&d_y, sizeof(double) * n * dy), "alloc y");
  CK(cudaMalloc(&d_rsK, sizeof(double) * n), "alloc rowsums K");
  CK(cudaMalloc(&d_rsL, sizeof(double) * n), "alloc rowsums L");
  // d_sc: [0]=totK [1]=totL [2]=Sab [3]=Saa [4]=Sbb
  CK(cudaMalloc(&d_sc, sizeof(double) * 5), "alloc scalars");
  CK(cudaMemcpy(d_x, REAL(xs), sizeof(double) * n * dx,
                cudaMemcpyHostToDevice), "copy x");
  CK(cudaMemcpy(d_y, REAL(ys), sizeof(double) * n * dy,
                cudaMemcpyHostToDevice), "copy y");
  CK(cudaMemset(d_sc, 0, sizeof(double) * 5), "zero scalars");

  if (err == cudaSuccess) {
    stage = "rowsum kernels";
    rowsum_kernel<<<n, BLOCK>>>(d_x, n, dx, index, d_rsK, &d_sc[0]);
    rowsum_kernel<<<n, BLOCK>>>(d_y, n, dy, index, d_rsL, &d_sc[1]);
    int gy = n < 1024 ? n : 1024;
    int gx = 2048 / gy; if (gx < 1) gx = 1;
    dim3 grid(gx, gy);
    fused_center_reduce<<<grid, BLOCK>>>(d_x, d_y, n, dx, dy, index, d_rsK,
                                         d_rsL, &d_sc[0], &d_sc[1], &d_sc[2]);
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
  }

  double h_sc[5] = {0, 0, 0, 0, 0};
  CK(cudaMemcpy(h_sc, d_sc, sizeof(double) * 5, cudaMemcpyDeviceToHost),
     "copy results");
#undef CK

  cudaFree(d_x); cudaFree(d_y); cudaFree(d_rsK); cudaFree(d_rsL); cudaFree(d_sc);
  if (err != cudaSuccess)
    Rf_error("CUDA error (%s): %s", stage, cudaGetErrorString(err));

  SEXP out = PROTECT(Rf_allocVector(REALSXP, 5));
  REAL(out)[0] = h_sc[2];  // sum(A o B)
  REAL(out)[1] = h_sc[3];  // sum(A o A)
  REAL(out)[2] = h_sc[4];  // sum(B o B)
  REAL(out)[3] = h_sc[0];  // sum(K)
  REAL(out)[4] = h_sc[1];  // sum(L)
  UNPROTECT(1);
  return out;
}

}  // extern "C"
