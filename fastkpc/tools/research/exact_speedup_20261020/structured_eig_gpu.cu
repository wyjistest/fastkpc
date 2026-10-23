// Isolated research implementation; never selected by the production runtime.
// Build after generate_recorded_dbdsqr.py (put its output on the include path).
#include <cuda_runtime.h>
#include <math_constants.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "../../../src/cuda/lapack_312_small_dgesdd.cuh"

namespace fastkpc { namespace lapack312 {
struct Rotation { int i, j; double c, s; };
struct RotationLog {
  Rotation* data;
  double* base;
  int count, capacity, overflow;
};
__device__ void record_rotation(RotationLog* log, int i, int j, double c, double s) {
  if (log->count >= log->capacity) { log->overflow = 1; return; }
  log->data[log->count++] = {i, j, c, s};
}
__device__ int record_drot_(RotationLog* log, int* n, double* x, int* ix,
                            double* y, int* iy, double* c, double* s) {
  if (*n != 1) { log->overflow = 2; return 0; }
  record_rotation(log, int(x - log->base), int(y - log->base), *c, *s);
  const double a = *x, b = *y;
  *x = *c * a + *s * b;
  *y = *c * b - *s * a;
  return 0;
}
__device__ int record_dlasr_(RotationLog* log, char* side, char* pivot, char* direct,
    int* m, int* n, double* c, double* s, double* a, int* lda, int, int, int) {
  if (*side != 'L' || *pivot != 'V' || *n != 1) {
    log->overflow = 3; return 0;
  }
  for (int t = 0; t < *m - 1; ++t) {
    const int j = *direct == 'F' ? t : *m - 2 - t;
    if (c[j] == 1.0 && s[j] == 0.0) continue;
    int one = 1;
    record_drot_(log, &one, a+j, lda, a+j+1, lda, c+j, s+j);
  }
  return 0;
}
__device__ int record_dscal_(RotationLog* log, int* n, double* c, double* x, int*) {
  if (*n != 1) { log->overflow = 4; return 0; }
  record_rotation(log, int(x - log->base), -1, *c, 0);
  *x *= *c;
  return 0;
}
__device__ int record_dswap_(RotationLog* log, int* n, double* x, int*, double* y, int*) {
  if (*n != 1) { log->overflow = 5; return 0; }
  record_rotation(log, int(x - log->base), int(y - log->base), 0, 2);
  const double v = *x; *x = *y; *y = v;
  return 0;
}
#define TRUE_ 1
#define FALSE_ 0
#define abs(x) ((x) >= 0 ? (x) : -(x))
#define min(a,b) ((a) <= (b) ? (a) : (b))
#define max(a,b) ((a) >= (b) ? (a) : (b))
#include "recorded_dbdsqr.inc"
#undef TRUE_
#undef FALSE_
#undef abs
#undef min
#undef max
}}

namespace {
using fastkpc::lapack312::Rotation;
using fastkpc::lapack312::RotationLog;
constexpr int max_n = 1024;
constexpr int max_k = 128;
constexpr int vectors_per_component = 20;
struct Meta {
  int info, m, live, candidates, rotations, deflated;
  double alpha, scale;
};
struct Candidate { double value, delta; int kind, index; };
struct Workspace {
  double *x, *v, *cnt, *t, *ld, *le, *d, *e, *q, *a, *b, *work, *rows;
  int *order, *group, *live, *selected;
};
__device__ Workspace workspace(double* d, int* i, int n, int batch) {
  d += size_t(batch) * vectors_per_component * n;
  i += size_t(batch) * 4 * n;
  return {d, d+n, d+2*n, d+3*n, d+4*n, d+5*n, d+6*n, d+7*n,
          d+8*n, d+9*n, d+10*n, d+11*n, d+15*n, i, i+n, i+2*n, i+3*n};
}

__global__ void prepare(const double* input, int n, double* data, int* ints, Meta* meta) {
  const int component = blockIdx.x;
  auto w = workspace(data, ints, n, component);
  __shared__ double values[max_n];
  __shared__ int ids[max_n];
  int width = 1; while (width < n) width *= 2;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    values[i] = i < n ? input[size_t(component)*n+i] : CUDART_INF;
    ids[i] = i;
  }
  __syncthreads();
  for (int size = 2; size <= width; size *= 2) {
    for (int stride = size/2; stride; stride /= 2) {
      for (int i = threadIdx.x; i < width; i += blockDim.x) {
        int j = i ^ stride;
        if (j > i) {
          bool greater = values[i] > values[j] || (values[i] == values[j] && ids[i] > ids[j]);
          if (greater == ((i & size) == 0)) {
            double v = values[i]; values[i] = values[j]; values[j] = v;
            int id = ids[i]; ids[i] = ids[j]; ids[j] = id;
          }
        }
      }
      __syncthreads();
    }
  }
  if (threadIdx.x != 0) return;
  Meta z = {};
  for (int i = 0; i < n; ++i) {
    if (!isfinite(values[i])) z.info = 1;
    w.x[i] = values[i]; w.order[i] = ids[i];
    if (i == 0 || values[i] != values[i-1]) {
      w.v[z.m] = values[i]; w.cnt[z.m] = 0; ++z.m;
    }
    w.group[i] = z.m-1; w.cnt[z.m-1] += 1;
  }
  if (z.m < 2 || z.info) { z.info = z.info ? z.info : 2; meta[component] = z; return; }
  double cumulative = 0;
  for (int j = 0; j < z.m; ++j) { cumulative += w.cnt[j]; w.t[j] = cumulative; }
  double left = 0, right = 0;
  for (int i = 0; i < n; ++i) {
    if (i) left += i * (values[i]-values[i-1]);
    w.rows[i] = left;
  }
  for (int i = n-1; i >= 0; --i) {
    if (i < n-1) right += (n-i-1) * (values[i+1]-values[i]);
    w.rows[i] += right; z.alpha += w.rows[i];
  }
  z.alpha /= n;
  const double sqrt_n = sqrt(double(n));
  double sum_s = 0;
  for (int i = 0; i < n; ++i) sum_s += w.rows[i]/sqrt_n;
  const double mean_s = sum_s/n;
  cumulative = 0;
  int row = 0;
  for (int j = 0; j < z.m-1; ++j) {
    while (row < int(w.t[j])) { cumulative += w.rows[row]/sqrt_n-mean_s; ++row; }
    w.q[j] = cumulative;
    w.ld[j] = sqrt(w.t[j+1] / (w.t[j]*w.cnt[j+1]));
    w.le[j] = j < z.m-2 ? -sqrt(w.t[j]/(w.t[j+1]*w.cnt[j+1])) : 0;
    w.d[j] = w.ld[j]/sqrt(w.v[j+1]-w.v[j]);
    w.e[j] = j < z.m-2 ? w.le[j]/sqrt(w.v[j+2]-w.v[j+1]) : 0;
  }
  for (int j = 0; j < z.m-1; ++j) {
    w.q[j] = w.ld[j]*w.q[j] + (j < z.m-2 ? w.le[j]*w.q[j+1] : 0);
    if (!isfinite(w.q[j]) || !isfinite(w.d[j]) || !(w.d[j]>0)) z.info = 3;
  }
  meta[component] = z;
}

__global__ void bidiagonal(int n, double* data, int* ints, Meta* meta,
                          Rotation* rotations, int capacity) {
  const int component = blockIdx.x;
  Meta& z = meta[component];
  if (z.info) return;
  auto w = workspace(data, ints, n, component);
  // QR repeatedly revisits its short diagonals; keep those and P^T q on chip.
  // Only the append-only rotation log needs global memory.
  extern __shared__ double local[];
  double* d = local;
  double* e = d+n;
  double* q = e+n;
  double* work = q+n;
  for (int j=threadIdx.x;j<z.m-1;j+=blockDim.x) {
    d[j]=w.d[j]; e[j]=w.e[j]; q[j]=w.q[j];
  }
  __syncthreads();
  if (threadIdx.x != 0) return;
  RotationLog log{rotations+size_t(component)*capacity, q, 0, capacity, 0};
  int r = z.m-1, one = 1, zero = 0, info = 0;
  double dummy[4] = {};
  fastkpc::lapack312::recorded_dbdsqr_(const_cast<char*>("L"), &r, &one, &zero, &zero,
      d, e, q, &r, dummy+2, &one, dummy+2, &one, work, &info, 1, &log);
  z.rotations = log.count;
  if (info || log.overflow) { z.info = info ? 10+info : 20+log.overflow; return; }
  double norm_b = 0;
  z.scale = fabs(z.alpha);
  for (int j = 0; j < r; ++j) {
    const int original = r-1-j;
    w.a[j] = (-2.0/d[original])/d[original];
    w.b[j] = q[original];
    norm_b = hypot(norm_b, w.b[j]);
    z.scale = fmax(z.scale, fabs(w.a[j]));
    if (!isfinite(w.a[j]) || !isfinite(w.b[j])) z.info = 30;
  }
  z.scale = fmax(z.scale, norm_b);
  for (int j = 0; j < r; ++j) {
    if (fabs(w.b[j]) <= 8*DBL_EPSILON*z.scale) ++z.deflated;
    else w.live[z.live++] = j;
  }
  for (int j = 1; j < z.live; ++j) {
    if (!(w.a[w.live[j]] > w.a[w.live[j-1]])) z.info = 31;
  }
}

__device__ double secular_value(const Workspace& w, const Meta& z, int origin, double delta) {
  double sum = 0;
  for (int j = 0; j < z.live; ++j) {
    int index = w.live[j];
    sum += (w.b[index]/((w.a[index]-w.a[origin])-delta))*w.b[index];
  }
  return z.alpha - (w.a[origin]+delta) - sum;
}
__device__ Candidate secular_root(const Workspace& w, const Meta& z, int low, int high) {
  double norm = 0;
  for (int j = 0; j < z.live; ++j) norm = hypot(norm, w.b[w.live[j]]);
  double lo, hi;
  int origin;
  if (low < 0) {
    origin = w.live[high]; lo = -2*(fabs(w.a[origin])+norm+fabs(z.alpha)+1); hi = 0;
  } else if (high < 0) {
    origin = w.live[low]; lo = 0; hi = 2*(fabs(w.a[origin])+norm+fabs(z.alpha)+1);
  } else {
    const double half = 0.5*(w.a[w.live[high]]-w.a[w.live[low]]);
    if (secular_value(w,z,w.live[low],half) > 0) {
      origin = w.live[high]; lo = -half; hi = 0;
    } else { origin = w.live[low]; lo = 0; hi = half; }
  }
  for (int iteration = 0; iteration < 400; ++iteration) {
    const double mid = 0.5*(lo+hi);
    if (mid <= lo || mid >= hi) break;
    if (secular_value(w,z,origin,mid) > 0) lo = mid; else hi = mid;
  }
  const double delta = 0.5*(lo+hi);
  return {w.a[origin]+delta, delta, 0, origin};
}
__global__ void roots(int n, int k, double* data, int* ints, Meta* meta, Candidate* candidates) {
  const int component = blockIdx.x;
  Meta& z = meta[component];
  if (z.info) return;
  auto w = workspace(data,ints,n,component);
  Candidate* out = candidates+size_t(component)*(2*k+4);
  const int root_count = z.live ? 2+min(k,z.live-1) : 1;
  for (int j = threadIdx.x; j < root_count; j += blockDim.x) {
    if (!z.live) out[j] = {z.alpha,0,2,-1};
    else if (j == 0) out[j] = secular_root(w,z,z.live-1,-1);
    else if (j == 1) out[j] = secular_root(w,z,-1,0);
    else out[j] = secular_root(w,z,j-2,j-1);
  }
  __syncthreads();
  if (threadIdx.x != 0) return;
  int count = root_count;
  for (int j = 0, inserted = 0; j < z.m-1 && inserted <= k; ++j) {
    if (fabs(w.b[j]) <= 8*DBL_EPSILON*z.scale) {
      out[count++] = {w.a[j],0,1,j}; ++inserted;
    }
  }
  for (int j = 0; j < count; ++j) {
    if (!isfinite(out[j].value) || (out[j].kind == 0 && out[j].delta == 0)) z.info = 40;
    w.selected[j] = j;
  }
  for (int j = 1; j < count; ++j) {
    const int item = w.selected[j]; int i = j;
    while (i && (fabs(out[item].value)>fabs(out[w.selected[i-1]].value) ||
        (fabs(out[item].value)==fabs(out[w.selected[i-1]].value) &&
         out[item].value>out[w.selected[i-1]].value))) {
      w.selected[i] = w.selected[i-1]; --i;
    }
    w.selected[i] = item;
  }
  z.candidates = count;
}

__global__ void recover(int n, int k, double* data, int* ints, Meta* meta,
    const Rotation* rotations, int capacity, const Candidate* candidates,
    int tile, double* evals, double* vectors, double* means) {
  const int component = blockIdx.x;
  Meta& z = meta[component];
  if (z.info) return;
  auto w = workspace(data,ints,n,component);
  const int r = z.m-1;
  extern __shared__ double tile_vectors[];
  if (threadIdx.x == 0 && blockIdx.y == 0) means[component] = z.alpha/n;
  const int col = blockIdx.y*tile+threadIdx.x;
  if (threadIdx.x < tile && col < k) {
    double* y = tile_vectors+size_t(threadIdx.x)*n;
    double* out = vectors+(size_t(component)*k+col)*n;
    double eigenvalue = 0;
    for (int j = 0; j < n; ++j) { y[j] = 0; out[j] = 0; }
    if (z.m == 2) {
      // Exact |lambda| tie: positive eigenvalue has the higher ascending index.
      if (col < 2) {
        eigenvalue = (col ? -1.0 : 1.0)*(w.v[1]-w.v[0])*sqrt(w.cnt[0]*w.cnt[1]);
        for (int row = 0; row < n; ++row) {
          int g = w.group[row];
          out[w.order[row]] = (g && col ? -1.0 : 1.0)/sqrt(2*w.cnt[g]);
        }
      }
    } else if (col < min(z.m,k) && col < z.candidates) {
      const Candidate c = candidates[size_t(component)*(2*k+4)+w.selected[col]];
      eigenvalue = c.value;
      double y0 = c.kind == 1 ? 0 : 1;
      if (c.kind == 0) {
        for (int j = 0; j < z.live; ++j) {
          int p = w.live[j]; y[r-1-p] = w.b[p]/((w.a[c.index]-w.a[p])+c.delta);
        }
      } else if (c.kind == 1) y[r-1-c.index] = 1;
      double norm = fabs(y0);
      for (int j = 0; j < r; ++j) norm = hypot(norm,y[j]);
      y0 /= norm;
      for (int j = 0; j < r; ++j) y[j] /= norm;
      const Rotation* log = rotations+size_t(component)*capacity;
      for (int j = z.rotations-1; j >= 0; --j) {
        const Rotation v = log[j];
        if (v.j < 0) y[v.i] *= v.c;
        else if (v.s == 2) { const double tmp = y[v.i]; y[v.i] = y[v.j]; y[v.j] = tmp; }
        else {
          const double a = y[v.i], b = y[v.j];
          y[v.i] = v.c*a-v.s*b; y[v.j] = v.s*a+v.c*b;
        }
      }
      for (int j = r-1; j >= 0; --j) y[j] = w.ld[j]*y[j]+(j ? w.le[j-1]*y[j-1] : 0);
      double running = 0, mean = 0;
      y[r] = 0;
      for (int j = r-1; j >= 0; --j) { running += y[j]; y[j] = running; }
      for (int j = 0; j < z.m; ++j) mean += w.cnt[j]*y[j];
      mean /= n;
      norm = 0;
      for (int row = 0; row < n; ++row) {
        const double v = y[w.group[row]]-mean+y0/sqrt(double(n));
        out[w.order[row]] = v; norm = hypot(norm,v);
      }
      for (int row = 0; row < n; ++row) out[row] /= norm;
    }
    evals[size_t(component)*k+col] = eigenvalue;
    for (int row = 0; row < n; ++row) if (!isfinite(out[row])) atomicCAS(&z.info,0,50);
  }
}

void check(cudaError_t code) {
  if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
struct Context {
  int n, k, count, capacity, tile;
  cudaStream_t stream = nullptr;
  cudaEvent_t events[6] = {};
  std::vector<void*> allocations;
  size_t bytes = 0;
  double *input, *data, *evals, *vectors, *means;
  int* ints;
  Meta* meta;
  Rotation* rotations;
  Candidate* candidates;
  template<class T> T* allocate(size_t length) {
    void* ptr; check(cudaMalloc(&ptr,length*sizeof(T)));
    allocations.push_back(ptr); bytes += length*sizeof(T); return static_cast<T*>(ptr);
  }
  ~Context() {
    if (stream) cudaStreamSynchronize(stream);
    for (void* p : allocations) cudaFree(p);
    for (auto e : events) if (e) cudaEventDestroy(e);
    if (stream) cudaStreamDestroy(stream);
  }
};
void error_text(char* error, const char* text) { if (error) std::snprintf(error,512,"%s",text); }
}

extern "C" void* structured_eig_create(int n, int k, int count, int rotation_capacity, char* error) {
  Context* c = nullptr;
  try {
    if (n<6 || n>max_n || k<1 || k>max_k || k>=n || count<1 || count>64 ||
        2*k+4>n || rotation_capacity<1) throw std::runtime_error("unsupported prototype capacity");
    c = new Context; c->n=n; c->k=k; c->count=count; c->capacity=rotation_capacity;
    int device=0, shared_limit=0; check(cudaGetDevice(&device));
    check(cudaDeviceGetAttribute(&shared_limit,cudaDevAttrMaxSharedMemoryPerBlockOptin,device));
    c->tile=std::min(k,std::min(64,shared_limit/int(n*sizeof(double))));
    if (c->tile<1) throw std::runtime_error("insufficient shared memory");
    check(cudaFuncSetAttribute(recover,cudaFuncAttributeMaxDynamicSharedMemorySize,c->tile*n*sizeof(double)));
    check(cudaFuncSetAttribute(bidiagonal,cudaFuncAttributeMaxDynamicSharedMemorySize,7*n*sizeof(double)));
    check(cudaStreamCreateWithFlags(&c->stream,cudaStreamNonBlocking));
    for (auto& e : c->events) check(cudaEventCreate(&e));
    size_t rows=size_t(n)*count, cols=size_t(k)*count;
    c->input=c->allocate<double>(rows); c->data=c->allocate<double>(rows*vectors_per_component);
    c->ints=c->allocate<int>(rows*4);
    c->evals=c->allocate<double>(cols); c->vectors=c->allocate<double>(rows*k);
    c->means=c->allocate<double>(count); c->meta=c->allocate<Meta>(count);
    c->candidates=c->allocate<Candidate>(size_t(count)*(2*k+4));
    c->rotations=c->allocate<Rotation>(size_t(count)*rotation_capacity);
    return c;
  } catch (const std::exception& e) { error_text(error,e.what()); delete c; return nullptr; }
}
extern "C" int structured_eig_run(void* handle, const double* input, double* evals,
    double* vectors, double* means, double* statistics, int* diagnostics, char* error) {
  auto* c = static_cast<Context*>(handle);
  try {
    if (!c) throw std::runtime_error("null prototype handle");
    const size_t rows=size_t(c->n)*c->count;
    check(cudaMemcpyAsync(c->input,input,rows*sizeof(double),cudaMemcpyHostToDevice,c->stream));
    check(cudaEventRecord(c->events[0],c->stream));
    prepare<<<c->count,128,0,c->stream>>>(c->input,c->n,c->data,c->ints,c->meta);
    check(cudaEventRecord(c->events[1],c->stream));
    bidiagonal<<<c->count,32,7*c->n*sizeof(double),c->stream>>>(c->n,c->data,c->ints,c->meta,c->rotations,c->capacity);
    check(cudaEventRecord(c->events[2],c->stream));
    roots<<<c->count,128,0,c->stream>>>(c->n,c->k,c->data,c->ints,c->meta,c->candidates);
    check(cudaEventRecord(c->events[3],c->stream));
    recover<<<dim3(c->count,(c->k+c->tile-1)/c->tile),64,c->tile*c->n*sizeof(double),c->stream>>>(
        c->n,c->k,c->data,c->ints,c->meta,c->rotations,
        c->capacity,c->candidates,c->tile,c->evals,c->vectors,c->means);
    check(cudaEventRecord(c->events[4],c->stream)); check(cudaGetLastError());
    std::vector<Meta> status(c->count);
    check(cudaMemcpyAsync(status.data(),c->meta,status.size()*sizeof(Meta),cudaMemcpyDeviceToHost,c->stream));
    check(cudaStreamSynchronize(c->stream));
    bool okay=true;
    for (int j=0;j<c->count;++j) {
      diagnostics[4*j]=status[j].info; diagnostics[4*j+1]=status[j].m;
      diagnostics[4*j+2]=status[j].rotations; diagnostics[4*j+3]=status[j].deflated;
      okay=okay && !status[j].info;
    }
    for (int j=0;j<4;++j) { float ms; check(cudaEventElapsedTime(&ms,c->events[j],c->events[j+1])); statistics[j]=ms; }
    statistics[4]=double(c->bytes);
    if (!okay) throw std::runtime_error("structured eig failed closed; inspect per-component diagnostics");
    // Qualification-only shadows. Production integration is deliberately absent.
    check(cudaMemcpyAsync(evals,c->evals,size_t(c->count)*c->k*sizeof(double),cudaMemcpyDeviceToHost,c->stream));
    check(cudaMemcpyAsync(vectors,c->vectors,rows*c->k*sizeof(double),cudaMemcpyDeviceToHost,c->stream));
    check(cudaMemcpyAsync(means,c->means,c->count*sizeof(double),cudaMemcpyDeviceToHost,c->stream));
    check(cudaStreamSynchronize(c->stream));
    return 0;
  } catch (const std::exception& e) { error_text(error,e.what()); return 1; }
}
extern "C" void structured_eig_destroy(void* handle) { delete static_cast<Context*>(handle); }
