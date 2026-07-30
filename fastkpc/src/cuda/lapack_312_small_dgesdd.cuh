#ifndef FASTKPC_LAPACK_312_SMALL_DGESDD_CUH
#define FASTKPC_LAPACK_312_SMALL_DGESDD_CUH

#include <cfloat>
#include <cuda_runtime.h>

namespace fastkpc {
#ifndef FASTKPC_LAPACK_SMALL_NAMESPACE
#define FASTKPC_LAPACK_SMALL_NAMESPACE lapack312
#define FASTKPC_LAPACK_SMALL_NAMESPACE_DEFAULTED
#endif
namespace FASTKPC_LAPACK_SMALL_NAMESPACE {

#ifndef FASTKPC_LAPACK_SMALL_SMLSIZ
#define FASTKPC_LAPACK_SMALL_SMLSIZ 25
#endif

using integer = int;
using logical = int;
using doublereal = double;
using ftnlen = int;

#define TRUE_ 1
#define FALSE_ 0
#define abs(x) ((x) >= 0 ? (x) : -(x))
#define dabs(x) static_cast<doublereal>(abs(x))
#define min(a, b) ((a) <= (b) ? (a) : (b))
#define max(a, b) ((a) >= (b) ? (a) : (b))

__device__ doublereal d_sign(doublereal* a, doublereal* b) {
  return copysign(abs(*a), *b);
}

__device__ doublereal pow_di(doublereal* value, integer* power) {
  integer exponent = *power;
  doublereal base = *value;
  doublereal result = 1.0;
  if (exponent < 0) {
    exponent = -exponent;
    base = 1.0 / base;
  }
  while (exponent != 0) {
    if ((exponent & 1) != 0) result *= base;
    exponent >>= 1;
    if (exponent != 0) base *= base;
  }
  return result;
}

__device__ integer pow_ii(integer* value, integer* power) {
  integer exponent = *power;
  integer base = *value;
  integer result = 1;
  while (exponent > 0) {
    if ((exponent & 1) != 0) result *= base;
    exponent >>= 1;
    if (exponent != 0) base *= base;
  }
  return result;
}

__device__ doublereal pow_dd(doublereal* value, doublereal* power) {
  return pow(*value, *power);
}

__device__ logical lsame_(char* left, char* right, ftnlen, ftnlen) {
  char a = *left;
  char b = *right;
  if (a >= 'a' && a <= 'z') a -= 'a' - 'A';
  if (b >= 'a' && b <= 'z') b -= 'a' - 'A';
  return a == b;
}

__device__ logical disnan_(doublereal* value) {
  return isnan(*value);
}

__device__ int xerbla_(char*, integer*, ftnlen) { return 0; }

__device__ integer ilaenv_(integer* ispec, char*, char*, integer*, integer*,
                           integer*, integer*, ftnlen, ftnlen) {
  if (*ispec == 9) return FASTKPC_LAPACK_SMALL_SMLSIZ;
  return 1;
}

__device__ doublereal dlamch_(char* cmach, ftnlen) {
  char value = *cmach;
  if (value >= 'a' && value <= 'z') value -= 'a' - 'A';
  switch (value) {
    case 'E': return 0x1p-53;
    case 'S': return DBL_MIN;
    case 'B': return 2.0;
    case 'P': return DBL_EPSILON;
    case 'N': return 53.0;
    case 'R': return 1.0;
    case 'M': return -1021.0;
    case 'U': return DBL_MIN;
    case 'L': return 1024.0;
    case 'O': return DBL_MAX;
  }
  return 0.0;
}

__device__ int cooperative_lane_count() {
  return blockDim.x >= 32 && __activemask() == 0xffffffffu ? 32 : 1;
}

__device__ int cooperative_lane_index(int lane_count) {
  return lane_count == 32 ? static_cast<int>(threadIdx.x) & 31 : 0;
}

__device__ void cooperative_sync(int lane_count) {
  if (lane_count == 32) __syncwarp();
}

__device__ int dcopy_(integer* n, doublereal* x, integer* incx,
                      doublereal* y, integer* incy) {
  const int lane_count = cooperative_lane_count();
  const int lane = cooperative_lane_index(lane_count);
  const integer initial_x = *incx >= 0 ? 0 : (1 - *n) * *incx;
  const integer initial_y = *incy >= 0 ? 0 : (1 - *n) * *incy;
  for (integer i = lane; i < *n; i += lane_count) {
    y[initial_y + i * *incy] = x[initial_x + i * *incx];
  }
  cooperative_sync(lane_count);
  return 0;
}

__device__ int dswap_(integer* n, doublereal* x, integer* incx,
                      doublereal* y, integer* incy) {
  const int lane_count = cooperative_lane_count();
  const int lane = cooperative_lane_index(lane_count);
  const integer initial_x = *incx >= 0 ? 0 : (1 - *n) * *incx;
  const integer initial_y = *incy >= 0 ? 0 : (1 - *n) * *incy;
  for (integer i = lane; i < *n; i += lane_count) {
    const integer ix = initial_x + i * *incx;
    const integer iy = initial_y + i * *incy;
    const doublereal saved = x[ix];
    x[ix] = y[iy];
    y[iy] = saved;
  }
  cooperative_sync(lane_count);
  return 0;
}

__device__ int dscal_(integer* n, doublereal* alpha, doublereal* x,
                      integer* incx) {
  if (*incx <= 0) return 0;
  const int lane_count = cooperative_lane_count();
  const int lane = cooperative_lane_index(lane_count);
  for (integer i = lane; i < *n; i += lane_count) {
    x[i * *incx] *= *alpha;
  }
  cooperative_sync(lane_count);
  return 0;
}

__device__ int drot_(integer* n, doublereal* x, integer* incx,
                     doublereal* y, integer* incy, doublereal* c,
                     doublereal* s) {
  const int lane_count = cooperative_lane_count();
  const int lane = cooperative_lane_index(lane_count);
  const integer initial_x = *incx >= 0 ? 0 : (1 - *n) * *incx;
  const integer initial_y = *incy >= 0 ? 0 : (1 - *n) * *incy;
  for (integer i = lane; i < *n; i += lane_count) {
    const integer ix = initial_x + i * *incx;
    const integer iy = initial_y + i * *incy;
    const doublereal old_x = x[ix];
    const doublereal old_y = y[iy];
    x[ix] = *c * old_x + *s * old_y;
    y[iy] = *c * old_y - *s * old_x;
  }
  cooperative_sync(lane_count);
  return 0;
}

__device__ doublereal dnrm2_(integer* n, doublereal* x, integer* incx) {
  if (*n <= 0) return 0.0;
  constexpr doublereal tsml = 0x1p-511;
  constexpr doublereal tbig = 0x1p486;
  constexpr doublereal ssml = 0x1p537;
  constexpr doublereal sbig = 0x1p-538;
  bool notbig = true;
  doublereal asml = 0.0;
  doublereal amed = 0.0;
  doublereal abig = 0.0;
  integer ix = *incx < 0 ? (1 - *n) * *incx : 0;
  for (integer i = 0; i < *n; ++i, ix += *incx) {
    const doublereal ax = abs(x[ix]);
    if (ax > tbig) {
      const doublereal scaled = ax * sbig;
      abig += scaled * scaled;
      notbig = false;
    } else if (ax < tsml) {
      if (notbig) {
        const doublereal scaled = ax * ssml;
        asml += scaled * scaled;
      }
    } else {
      amed += ax * ax;
    }
  }
  doublereal scale = 1.0;
  doublereal sumsq = 0.0;
  if (abig > 0.0) {
    if (amed > 0.0 || isinf(amed) || isnan(amed)) {
      abig += (amed * sbig) * sbig;
    }
    scale = 1.0 / sbig;
    sumsq = abig;
  } else if (asml > 0.0) {
    if (amed > 0.0 || isinf(amed) || isnan(amed)) {
      amed = sqrt(amed);
      asml = sqrt(asml) / ssml;
      const doublereal ymin = asml > amed ? amed : asml;
      const doublereal ymax = asml > amed ? asml : amed;
      const doublereal ratio = ymin / ymax;
      sumsq = ymax * ymax * (1.0 + ratio * ratio);
    } else {
      scale = 1.0 / ssml;
      sumsq = asml;
    }
  } else {
    sumsq = amed;
  }
  return scale * sqrt(sumsq);
}

__device__ int dlassq_(integer* n, doublereal* x, integer* incx,
                       doublereal* scale, doublereal* sumsq) {
  if (isnan(*scale) || isnan(*sumsq)) return 0;
  if (*sumsq == 0.0) *scale = 1.0;
  if (*scale == 0.0) {
    *scale = 1.0;
    *sumsq = 0.0;
  }
  if (*n <= 0) return 0;
  constexpr doublereal tsml = 0x1p-511;
  constexpr doublereal tbig = 0x1p486;
  constexpr doublereal ssml = 0x1p537;
  constexpr doublereal sbig = 0x1p-538;
  bool notbig = true;
  doublereal asml = 0.0;
  doublereal amed = 0.0;
  doublereal abig = 0.0;
  integer ix = *incx < 0 ? (1 - *n) * *incx : 0;
  for (integer i = 0; i < *n; ++i, ix += *incx) {
    const doublereal ax = abs(x[ix]);
    if (ax > tbig) {
      const doublereal scaled = ax * sbig;
      abig += scaled * scaled;
      notbig = false;
    } else if (ax < tsml) {
      if (notbig) {
        const doublereal scaled = ax * ssml;
        asml += scaled * scaled;
      }
    } else {
      amed += ax * ax;
    }
  }
  if (*sumsq > 0.0) {
    const doublereal ax = *scale * sqrt(*sumsq);
    if (ax > tbig) {
      if (*scale > 1.0) {
        *scale *= sbig;
        abig += *scale * (*scale * *sumsq);
      } else {
        abig += *scale * (*scale * (sbig * (sbig * *sumsq)));
      }
    } else if (ax < tsml) {
      if (notbig) {
        if (*scale < 1.0) {
          *scale *= ssml;
          asml += *scale * (*scale * *sumsq);
        } else {
          asml += *scale * (*scale * (ssml * (ssml * *sumsq)));
        }
      }
    } else {
      amed += *scale * (*scale * *sumsq);
    }
  }
  if (abig > 0.0) {
    if (amed > 0.0 || isnan(amed)) abig += (amed * sbig) * sbig;
    *scale = 1.0 / sbig;
    *sumsq = abig;
  } else if (asml > 0.0) {
    if (amed > 0.0 || isnan(amed)) {
      amed = sqrt(amed);
      asml = sqrt(asml) / ssml;
      const doublereal ymin = asml > amed ? amed : asml;
      const doublereal ymax = asml > amed ? asml : amed;
      const doublereal ratio = ymin / ymax;
      *scale = 1.0;
      *sumsq = ymax * ymax * (1.0 + ratio * ratio);
    } else {
      *scale = 1.0 / ssml;
      *sumsq = asml;
    }
  } else {
    *scale = 1.0;
    *sumsq = amed;
  }
  return 0;
}

__device__ int dgemv_(char* trans, integer* m, integer* n,
                      doublereal* alpha, doublereal* a, integer* lda,
                      doublereal* x, integer* incx, doublereal* beta,
                      doublereal* y, integer* incy, ftnlen) {
  const bool no_trans = lsame_(trans, const_cast<char*>("N"), 1, 1);
  const integer len_y = no_trans ? *m : *n;
  const int lane_count = cooperative_lane_count();
  const int lane = cooperative_lane_index(lane_count);
  const integer initial_y = *incy >= 0 ? 0 : (1 - len_y) * *incy;
  for (integer i = lane; i < len_y; i += lane_count) {
    const integer iy = initial_y + i * *incy;
    y[iy] = *beta == 0.0 ? 0.0 : *beta * y[iy];
  }
  cooperative_sync(lane_count);
  if (*alpha == 0.0) return 0;
  if (no_trans) {
    const integer initial_x = *incx >= 0 ? 0 : (1 - *n) * *incx;
    for (integer i = lane; i < *m; i += lane_count) {
      const integer iy = initial_y + i * *incy;
      doublereal value = y[iy];
      for (integer j = 0; j < *n; ++j) {
        const doublereal x_value = x[initial_x + j * *incx];
        if (x_value != 0.0) {
          const doublereal temp = *alpha * x_value;
          value += temp * a[i + *lda * j];
        }
      }
      y[iy] = value;
    }
  } else {
    const integer initial_x = *incx >= 0 ? 0 : (1 - *m) * *incx;
    for (integer j = lane; j < *n; j += lane_count) {
      doublereal temp = 0.0;
      for (integer i = 0; i < *m; ++i) {
        temp += a[i + *lda * j] * x[initial_x + i * *incx];
      }
      y[initial_y + j * *incy] += *alpha * temp;
    }
  }
  cooperative_sync(lane_count);
  return 0;
}

__device__ int dger_(integer* m, integer* n, doublereal* alpha,
                     doublereal* x, integer* incx, doublereal* y,
                     integer* incy, doublereal* a, integer* lda) {
  const int lane_count = cooperative_lane_count();
  const int lane = cooperative_lane_index(lane_count);
  const integer initial_x = *incx >= 0 ? 0 : (1 - *m) * *incx;
  const integer initial_y = *incy >= 0 ? 0 : (1 - *n) * *incy;
  for (integer j = lane; j < *n; j += lane_count) {
    const integer jy = initial_y + j * *incy;
    if (y[jy] == 0.0) continue;
    const doublereal temp = *alpha * y[jy];
    for (integer i = 0; i < *m; ++i) {
      a[i + *lda * j] += x[initial_x + i * *incx] * temp;
    }
  }
  cooperative_sync(lane_count);
  return 0;
}

__device__ int dgemm_(char* transa, char* transb, integer* m, integer* n,
                      integer* k, doublereal* alpha, doublereal* a,
                      integer* lda, doublereal* b, integer* ldb,
                      doublereal* beta, doublereal* c, integer* ldc,
                      ftnlen, ftnlen) {
  const bool nota = lsame_(transa, const_cast<char*>("N"), 1, 1);
  const bool notb = lsame_(transb, const_cast<char*>("N"), 1, 1);
  const int lane_count = cooperative_lane_count();
  const int lane = cooperative_lane_index(lane_count);
  const integer output_count = *m * *n;
  for (integer output = lane; output < output_count;
       output += lane_count) {
    const integer i = output % *m;
    const integer j = output / *m;
    doublereal value = 0.0;
    for (integer inner = 0; inner < *k; ++inner) {
      const doublereal left = nota ? a[i + *lda * inner]
                                   : a[inner + *lda * i];
      const doublereal right = notb ? b[inner + *ldb * j]
                                    : b[j + *ldb * inner];
      value += left * right;
    }
    c[i + *ldc * j] = *alpha * value +
      (*beta == 0.0 ? 0.0 : *beta * c[i + *ldc * j]);
  }
  cooperative_sync(lane_count);
  return 0;
}

__device__ int dlartg_(doublereal* f, doublereal* g, doublereal* c,
                       doublereal* s, doublereal* r) {
  constexpr doublereal safmin = 0x1p-1023;
  constexpr doublereal safmax = 0x1p1023;
  const doublereal rtmin = sqrt(safmin);
  const doublereal rtmax = sqrt(safmax / 2.0);
  const doublereal f1 = abs(*f);
  const doublereal g1 = abs(*g);
  if (*g == 0.0) {
    *c = 1.0;
    *s = 0.0;
    *r = *f;
  } else if (*f == 0.0) {
    *c = 0.0;
    *s = copysign(1.0, *g);
    *r = g1;
  } else if (f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax) {
    const doublereal d = sqrt(*f * *f + *g * *g);
    *c = f1 / d;
    *r = copysign(d, *f);
    *s = *g / *r;
  } else {
    const doublereal u = min(safmax, max(safmin, max(f1, g1)));
    const doublereal fs = *f / u;
    const doublereal gs = *g / u;
    const doublereal d = sqrt(fs * fs + gs * gs);
    *c = abs(fs) / d;
    *r = copysign(d, *f);
    *s = gs / *r;
    *r *= u;
  }
  return 0;
}

__device__ int dlasq1_(integer*, doublereal*, doublereal*, doublereal*,
                       integer* info) {
  *info = 1;
  return 0;
}

__device__ int dlasda_(integer*, integer*, integer*, integer*, doublereal*,
                       doublereal*, doublereal*, integer*, doublereal*,
                       integer*, doublereal*, doublereal*, doublereal*,
                       doublereal*, integer*, integer*, integer*, integer*,
                       doublereal*, doublereal*, doublereal*, doublereal*,
                       integer*, integer*) {
  return 0;
}

#include "third_party/lapack_3_12_small_dgesdd_device.inc"

#undef max
#undef min
#undef dabs
#undef abs
#undef FALSE_
#undef TRUE_

#ifndef FASTKPC_LAPACK_SMALL_MAX_COLUMNS
#define FASTKPC_LAPACK_SMALL_MAX_COLUMNS 32
#endif

constexpr integer kMaximumColumns = FASTKPC_LAPACK_SMALL_MAX_COLUMNS;
constexpr integer kMaximumRows = 2 * kMaximumColumns;
constexpr integer kDbdsdcWorkSize =
  3 * kMaximumColumns * kMaximumColumns + 4 * kMaximumColumns;
constexpr doublereal kGuardedQrMinimumDiagonalRatio = 1e-6;
constexpr doublereal kGuardedQrConditionUpperBound = 1e7;

struct Workspace {
  doublereal a[kMaximumRows * kMaximumColumns];
  doublereal left_u[kMaximumRows * kMaximumColumns];
  doublereal qr_tau[kMaximumColumns];
  doublereal bidiagonal[kMaximumColumns];
  doublereal bidiagonal_e[kMaximumColumns];
  doublereal tau_q[kMaximumColumns];
  doublereal tau_p[kMaximumColumns];
  doublereal work[kDbdsdcWorkSize];
  integer iwork[8 * kMaximumColumns];
  doublereal r[kMaximumColumns * kMaximumColumns];
  doublereal bidiagonal_u[kMaximumColumns * kMaximumColumns];
  doublereal bidiagonal_vt[kMaximumColumns * kMaximumColumns];
  integer qr_basis_used;
  integer qr_guard_accepted;
  doublereal qr_condition_estimate;
};

__device__ int small_dbdsdc_upper_i_two_warp(
    integer n, Workspace* workspace) {
  constexpr integer kStatusBase = 8 * kMaximumColumns - 4;
  const integer warp = static_cast<integer>(threadIdx.x) >> 5;
  const integer lane = static_cast<integer>(threadIdx.x) & 31;
  doublereal* d = workspace->bidiagonal;
  doublereal* e = workspace->bidiagonal_e;
  doublereal* u = workspace->bidiagonal_u;
  doublereal* vt = workspace->bidiagonal_vt;
  doublereal* work = workspace->work;
  integer* iwork = workspace->iwork;
  const integer nm1 = n - 1;
  const integer smlsiz = FASTKPC_LAPACK_SMALL_SMLSIZ;

  if (threadIdx.x == 0) {
    iwork[kStatusBase] = 0;
    iwork[kStatusBase + 1] = 0;
    iwork[kStatusBase + 2] = 0;
    iwork[kStatusBase + 3] = 0;
  }
  __syncthreads();

  if (warp == 0) {
    integer level_count = 0;
    integer node_count = 0;
    dlasdt_(&n, &level_count, &node_count, iwork, iwork + n,
            iwork + 2 * n, const_cast<integer*>(&smlsiz));
    const doublereal norm = dlanst_(
      const_cast<char*>("M"), &n, d, e, 1);
    bool use_original = node_count != 1 || level_count != 1 ||
      !isfinite(norm) || norm == 0.0;
    if (!use_original) {
      const doublereal eps = dlamch_(
        const_cast<char*>("Epsilon"), 7) * 0.9;
      for (integer index = 0; index < nm1; ++index) {
        if (abs(e[index] / norm) <= eps * 2.0) {
          use_original = true;
          break;
        }
      }
    }
    if (lane == 0) {
      workspace->left_u[0] = norm;
      iwork[kStatusBase] = use_original ? 1 : 0;
    }
  }
  __syncthreads();

  if (iwork[kStatusBase] != 0) {
    if (warp == 0) {
      integer info = 0;
      char upper = 'U';
      char vectors = 'I';
      doublereal compact_dummy = 0.0;
      integer integer_dummy = 0;
      dbdsdc_(&upper, &vectors, &n, d, e, u, &n, vt, &n,
              &compact_dummy, &integer_dummy, work, iwork, &info, 1, 1);
      if (lane == 0) iwork[kStatusBase + 1] = info;
    }
    __syncthreads();
    return iwork[kStatusBase + 1];
  }

  if (warp == 0) {
    integer info = 0;
    integer zero_i = 0;
    integer one_i = 1;
    doublereal zero = 0.0;
    doublereal one = 1.0;
    doublereal norm = workspace->left_u[0];
    dlaset_(const_cast<char*>("A"), &n, &n, &zero, &one, u, &n, 1);
    dlaset_(const_cast<char*>("A"), &n, &n, &zero, &one, vt, &n, 1);
    dlascl_(const_cast<char*>("G"), &zero_i, &zero_i, &norm, &one,
            &n, &one_i, d, &n, &info, 1);
    dlascl_(const_cast<char*>("G"), &zero_i, &zero_i, &norm, &one,
            const_cast<integer*>(&nm1), &one_i, e,
            const_cast<integer*>(&nm1), &info, 1);
    const doublereal eps = dlamch_(
      const_cast<char*>("Epsilon"), 7) * 0.9;
    for (integer index = 0; index < n; ++index) {
      if (abs(d[index]) < eps) d[index] = copysign(eps, d[index]);
    }
    if (lane == 0) iwork[kStatusBase + 1] = info;
  }
  __syncthreads();
  if (iwork[kStatusBase + 1] != 0) return iwork[kStatusBase + 1];

  const integer center = iwork[0];
  integer nl = iwork[n];
  integer nr = iwork[2 * n];
  const integer left_start = center - nl;
  const integer right_start = center + 1;
  integer local_info = 0;
  integer ncc = 0;
  if (warp == 0) {
    integer sqre = 1;
    integer nlp1 = nl + 1;
    dlasdq_(const_cast<char*>("U"), &sqre, &nl, &nlp1, &nl, &ncc,
            d + left_start - 1, e + left_start - 1,
            vt + (left_start - 1) + n * (left_start - 1), &n,
            u + (left_start - 1) + n * (left_start - 1), &n,
            u + (left_start - 1) + n * (left_start - 1), &n,
            work, &local_info, 1);
    for (integer index = lane; index < nl; index += 32) {
      iwork[3 * n + left_start - 1 + index] = index + 1;
    }
    if (lane == 0) iwork[kStatusBase + 2] = local_info;
  } else {
    integer sqre = 0;
    integer nrp1 = nr;
    dlasdq_(const_cast<char*>("U"), &sqre, &nr, &nrp1, &nr, &ncc,
            d + right_start - 1, e + right_start - 1,
            vt + (right_start - 1) + n * (right_start - 1), &n,
            u + (right_start - 1) + n * (right_start - 1), &n,
            u + (right_start - 1) + n * (right_start - 1), &n,
            work + 4 * n, &local_info, 1);
    for (integer index = lane; index < nr; index += 32) {
      iwork[3 * n + center + index] = index + 1;
    }
    if (lane == 0) iwork[kStatusBase + 3] = local_info;
  }
  __syncthreads();
  if (iwork[kStatusBase + 2] != 0) return iwork[kStatusBase + 2];
  if (iwork[kStatusBase + 3] != 0) return iwork[kStatusBase + 3];

  if (warp == 0) {
    integer sqre = 0;
    doublereal alpha = d[center - 1];
    doublereal beta = e[center - 1];
    dlasd1_(&nl, &nr, &sqre, d + left_start - 1, &alpha, &beta,
            u + (left_start - 1) + n * (left_start - 1), &n,
            vt + (left_start - 1) + n * (left_start - 1), &n,
            iwork + 3 * n + left_start - 1, iwork + 4 * n,
            work, &local_info);
    if (lane == 0) iwork[0] = local_info;
  }
  __syncthreads();
  if (iwork[0] != 0) return iwork[0];

  if (warp == 0) {
    integer info = 0;
    integer zero_i = 0;
    integer one_i = 1;
    doublereal one = 1.0;
    doublereal norm = workspace->left_u[0];
    dlascl_(const_cast<char*>("G"), &zero_i, &zero_i, &one, &norm,
            &n, &one_i, d, &n, &info, 1);
    for (integer ii = 2; ii <= n; ++ii) {
      const integer current = ii - 1;
      integer maximum = current;
      doublereal value = d[current - 1];
      for (integer j = ii; j <= n; ++j) {
        if (d[j - 1] > value) {
          maximum = j;
          value = d[j - 1];
        }
      }
      if (maximum != current) {
        d[maximum - 1] = d[current - 1];
        d[current - 1] = value;
        dswap_(&n, u + n * (current - 1), &one_i,
               u + n * (maximum - 1), &one_i);
        dswap_(&n, vt + current - 1, &n, vt + maximum - 1, &n);
      }
    }
    if (lane == 0) iwork[0] = info;
  }
  __syncthreads();
  return iwork[0];
}

__device__ int small_dgesdd_left(
    integer m, integer n, Workspace* workspace,
    bool compute_right_vectors = true,
    unsigned long long* stage_cycles = nullptr,
    bool allow_qr_complete = false,
    bool force_stable_svd = false) {
  if (blockDim.x == 64 && n > FASTKPC_LAPACK_SMALL_SMLSIZ) {
    const integer warp = static_cast<integer>(threadIdx.x) >> 5;
    const integer lane = static_cast<integer>(threadIdx.x) & 31;
    integer info = 0;
    unsigned long long stage_started = 0;
    if (threadIdx.x == 0 && stage_cycles != nullptr) {
      stage_started = clock64();
    }
    if (warp == 0) {
      if (lane == 0) {
        workspace->qr_basis_used = 0;
        workspace->qr_guard_accepted = 0;
      }
      cooperative_sync(32);
      dgeqr2_(&m, &n, workspace->a, &m, workspace->qr_tau,
              workspace->work, &info);
      if (info == 0) {
        for (integer index = lane; index < n * n; index += 32) {
          const integer row = index % n;
          const integer column = index / n;
          workspace->r[index] =
            row <= column ? workspace->a[row + m * column] : 0.0;
        }
        cooperative_sync(32);
        dorg2r_(&m, &n, &n, workspace->a, &m, workspace->qr_tau,
                workspace->work, &info);
      }
      if (lane == 0) {
        doublereal maximum_diagonal = 0.0;
        doublereal minimum_diagonal = DBL_MAX;
        for (integer index = 0; index < n; ++index) {
          const doublereal diagonal = abs(workspace->r[index + n * index]);
          maximum_diagonal = max(maximum_diagonal, diagonal);
          minimum_diagonal = min(minimum_diagonal, diagonal);
        }
        workspace->qr_guard_accepted =
          !force_stable_svd &&
            (!compute_right_vectors || allow_qr_complete) &&
            isfinite(maximum_diagonal) &&
            isfinite(minimum_diagonal) && maximum_diagonal > 0.0 &&
            minimum_diagonal > maximum_diagonal *
              kGuardedQrMinimumDiagonalRatio ? 1 : 0;
      }
      cooperative_sync(32);
      if (info == 0 && workspace->qr_guard_accepted != 0) {
        // In the QR basis, R^-T replaces D^-1 V^T.  The Frobenius
        // condition product is an upper bound on the spectral condition.
        for (integer column = lane; column < n; column += 32) {
          for (integer row = 0; row < column; ++row) {
            workspace->bidiagonal_vt[row + n * column] = 0.0;
          }
          for (integer row = column; row < n; ++row) {
            doublereal value = row == column ? 1.0 : 0.0;
            for (integer inner = column; inner < row; ++inner) {
              value -= workspace->r[inner + n * row] *
                workspace->bidiagonal_vt[inner + n * column];
            }
            workspace->bidiagonal_vt[row + n * column] =
              value / workspace->r[row + n * row];
          }
        }
        cooperative_sync(32);
        doublereal r_squared_norm = 0.0;
        doublereal inverse_squared_norm = 0.0;
        for (integer index = lane; index < n * n; index += 32) {
          r_squared_norm += workspace->r[index] * workspace->r[index];
          inverse_squared_norm += workspace->bidiagonal_vt[index] *
            workspace->bidiagonal_vt[index];
        }
        for (integer offset = 16; offset > 0; offset /= 2) {
          r_squared_norm += __shfl_down_sync(
            0xffffffffu, r_squared_norm, offset);
          inverse_squared_norm += __shfl_down_sync(
            0xffffffffu, inverse_squared_norm, offset);
        }
        if (lane == 0) {
          const doublereal condition_upper_bound =
            sqrt(r_squared_norm) * sqrt(inverse_squared_norm);
          workspace->qr_guard_accepted =
            isfinite(condition_upper_bound) &&
              condition_upper_bound < kGuardedQrConditionUpperBound ? 1 : 0;
          workspace->qr_condition_estimate = condition_upper_bound;
        }
        cooperative_sync(32);
      }
      if (info == 0 && workspace->qr_guard_accepted != 0) {
        for (integer index = lane; index < m * n; index += 32) {
          workspace->left_u[index] = workspace->a[index];
        }
        for (integer index = lane; index < n; index += 32) {
          workspace->bidiagonal[index] = 1.0;
        }
        cooperative_sync(32);
      } else if (info == 0) {
        dgebd2_(&n, &n, workspace->r, &n, workspace->bidiagonal,
                workspace->bidiagonal_e, workspace->tau_q,
                workspace->tau_p, workspace->work, &info);
      }
      if (lane == 0) workspace->iwork[0] = info;
    }
    __syncthreads();
    if (threadIdx.x == 0 && stage_cycles != nullptr) {
      stage_cycles[0] += clock64() - stage_started;
      stage_started = clock64();
    }
    if (workspace->iwork[0] != 0) return workspace->iwork[0];
    if (workspace->qr_guard_accepted != 0) {
      if (threadIdx.x == 0) workspace->qr_basis_used = 1;
      return 0;
    }
    if (threadIdx.x == 0) workspace->qr_basis_used = 0;

    info = small_dbdsdc_upper_i_two_warp(n, workspace);
    if (threadIdx.x == 0 && stage_cycles != nullptr) {
      stage_cycles[1] += clock64() - stage_started;
      stage_started = clock64();
    }
    if (info != 0) return info;

    if (warp == 0) {
      char left = 'L';
      char no_transpose = 'N';
      dorm2r_(&left, &no_transpose, &n, &n, &n, workspace->r, &n,
              workspace->tau_q, workspace->bidiagonal_u, &n,
              workspace->work, &info, 1, 1);
      if (info == 0 && compute_right_vectors) {
        char right = 'R';
        for (integer reflector = n - 2; reflector >= 0; --reflector) {
          integer reflector_order = n - reflector - 1;
          doublereal* vector =
            workspace->r + reflector + n * (reflector + 1);
          const doublereal saved = *vector;
          *vector = 1.0;
          dlarf_(&right, &n, &reflector_order, vector, &n,
                 workspace->tau_p + reflector,
                 workspace->bidiagonal_vt + n * (reflector + 1), &n,
                 workspace->work, 1);
          *vector = saved;
        }
      }
      if (lane == 0) workspace->iwork[0] = info;
    }
    __syncthreads();
    if (threadIdx.x == 0 && stage_cycles != nullptr) {
      stage_cycles[2] += clock64() - stage_started;
      stage_started = clock64();
    }
    if (workspace->iwork[0] != 0) return workspace->iwork[0];

    for (integer output = static_cast<integer>(threadIdx.x);
         output < m * n; output += static_cast<integer>(blockDim.x)) {
      const integer row = output % m;
      const integer column = output / m;
      doublereal value = 0.0;
      for (integer inner = 0; inner < n; ++inner) {
        value += workspace->a[row + m * inner] *
                 workspace->bidiagonal_u[inner + n * column];
      }
      workspace->left_u[output] = value;
    }
    __syncthreads();
    if (threadIdx.x == 0 && stage_cycles != nullptr) {
      stage_cycles[3] += clock64() - stage_started;
    }
    return 0;
  }

  integer info = 0;
  if (threadIdx.x == 0) workspace->qr_basis_used = 0;
  unsigned long long stage_started = 0;
  if (threadIdx.x == 0 && stage_cycles != nullptr) {
    stage_started = clock64();
  }
  dgeqr2_(&m, &n, workspace->a, &m, workspace->qr_tau,
          workspace->work, &info);
  if (info != 0) return info;
  const int lane_count = cooperative_lane_count();
  const int lane = cooperative_lane_index(lane_count);
  for (integer index = lane; index < n * n; index += lane_count) {
    const integer row = index % n;
    const integer column = index / n;
    workspace->r[index] =
      row <= column ? workspace->a[row + m * column] : 0.0;
  }
  cooperative_sync(lane_count);
  dorg2r_(&m, &n, &n, workspace->a, &m, workspace->qr_tau,
          workspace->work, &info);
  if (info != 0) return info;
  dgebd2_(&n, &n, workspace->r, &n, workspace->bidiagonal,
          workspace->bidiagonal_e, workspace->tau_q, workspace->tau_p,
          workspace->work, &info);
  if (threadIdx.x == 0 && stage_cycles != nullptr) {
    stage_cycles[0] += clock64() - stage_started;
    stage_started = clock64();
  }
  if (info != 0) return info;
  char upper = 'U';
  char vectors = 'I';
  doublereal compact_dummy = 0.0;
  integer integer_dummy = 0;
  dbdsdc_(&upper, &vectors, &n, workspace->bidiagonal,
          workspace->bidiagonal_e, workspace->bidiagonal_u, &n,
          workspace->bidiagonal_vt, &n, &compact_dummy, &integer_dummy,
          workspace->work, workspace->iwork, &info, 1, 1);
  if (threadIdx.x == 0 && stage_cycles != nullptr) {
    stage_cycles[1] += clock64() - stage_started;
    stage_started = clock64();
  }
  if (info != 0) return info;
  char left = 'L';
  char no_transpose = 'N';
  dorm2r_(&left, &no_transpose, &n, &n, &n, workspace->r, &n,
          workspace->tau_q, workspace->bidiagonal_u, &n,
          workspace->work, &info, 1, 1);
  if (info != 0) return info;

  // DORMBR('P','R','T') for the square upper-bidiagonal reduction.  In
  // this case DORMBR dispatches to the unblocked DORML2 path on columns
  // 2:n, applying G(n-1), ..., G(1) from the right.
  if (compute_right_vectors) {
    char right = 'R';
    for (integer reflector = n - 2; reflector >= 0; --reflector) {
      integer reflector_order = n - reflector - 1;
      doublereal* vector = workspace->r + reflector + n * (reflector + 1);
      const doublereal saved = *vector;
      *vector = 1.0;
      dlarf_(&right, &n, &reflector_order, vector, &n,
             workspace->tau_p + reflector,
             workspace->bidiagonal_vt + n * (reflector + 1), &n,
             workspace->work, 1);
      *vector = saved;
    }
  }
  if (threadIdx.x == 0 && stage_cycles != nullptr) {
    stage_cycles[2] += clock64() - stage_started;
    stage_started = clock64();
  }
  for (integer output = lane; output < m * n; output += lane_count) {
    const integer row = output % m;
    const integer column = output / m;
    doublereal value = 0.0;
    for (integer inner = 0; inner < n; ++inner) {
      value += workspace->a[row + m * inner] *
               workspace->bidiagonal_u[inner + n * column];
    }
    workspace->left_u[output] = value;
  }
  cooperative_sync(lane_count);
  if (threadIdx.x == 0 && stage_cycles != nullptr) {
    stage_cycles[3] += clock64() - stage_started;
  }
  return 0;
}

}  // namespace FASTKPC_LAPACK_SMALL_NAMESPACE
}  // namespace fastkpc

#ifdef FASTKPC_LAPACK_SMALL_NAMESPACE_DEFAULTED
#undef FASTKPC_LAPACK_SMALL_NAMESPACE_DEFAULTED
#undef FASTKPC_LAPACK_SMALL_NAMESPACE
#endif

#endif
