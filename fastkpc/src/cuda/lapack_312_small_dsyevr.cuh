#ifndef FASTKPC_LAPACK_312_SMALL_DSYEVR_CUH
#define FASTKPC_LAPACK_312_SMALL_DSYEVR_CUH

#include "lapack_312_small_dgesdd.cuh"

namespace fastkpc {
namespace lapack312_multi {
namespace symmetric_eigen {

using integer = lapack312_multi::integer;
using logical = lapack312_multi::logical;
using doublereal = lapack312_multi::doublereal;
using ftnlen = lapack312_multi::ftnlen;
using lapack312_multi::dcopy_;
using lapack312_multi::disnan_;
using lapack312_multi::dlamch_;
using lapack312_multi::dlanst_;
using lapack312_multi::dlapy2_;
using lapack312_multi::dlarfg_;
using lapack312_multi::dlascl_;
using lapack312_multi::dlaset_;
using lapack312_multi::dlassq_;
using lapack312_multi::dnrm2_;
using lapack312_multi::dorm2r_;
using lapack312_multi::dscal_;
using lapack312_multi::dswap_;
using lapack312_multi::ilaenv_;
using lapack312_multi::lsame_;
using lapack312_multi::xerbla_;
using lapack312_multi::d_sign;

__device__ int dormtr_(char*, char*, char*, integer*, integer*,
                       doublereal*, integer*, doublereal*, doublereal*,
                       integer*, doublereal*, integer*, integer*,
                       ftnlen, ftnlen, ftnlen);
__device__ int dsytrd_(char*, integer*, doublereal*, integer*, doublereal*,
                       doublereal*, doublereal*, doublereal*, integer*,
                       integer*, ftnlen);

#define TRUE_ 1
#define FALSE_ 0
#include "third_party/lapack_3_12_small_dsyevr_device.inc"
#undef FALSE_
#undef TRUE_

__device__ int dormtr_(char* side, char* uplo, char* trans, integer* m,
                       integer* n, doublereal* a, integer* lda,
                       doublereal* tau, doublereal* c, integer* ldc,
                       doublereal* work, integer* lwork, integer* info,
                       ftnlen side_len, ftnlen uplo_len,
                       ftnlen trans_len) {
  if (*lwork == -1) {
    work[0] = static_cast<doublereal>(*n > 1 ? *n : 1);
    *info = 0;
    return 0;
  }
  char left = 'L';
  char lower = 'L';
  if (!lsame_(side, &left, side_len, 1) ||
      !lsame_(uplo, &lower, uplo_len, 1)) {
    *info = -100;
    return 0;
  }
  integer reduced_rows = *m - 1;
  integer reflectors = *m - 1;
  return dorm2r_(side, trans, &reduced_rows, n, &reflectors, a + 1, lda,
                 tau, c + 1, ldc, work, info, side_len, trans_len);
}

__device__ int dsytrd_(char* uplo, integer* n, doublereal* a, integer* lda,
                       doublereal* d, doublereal* e, doublereal* tau,
                       doublereal* work, integer* lwork, integer* info,
                       ftnlen uplo_len) {
  if (*lwork == -1) {
    work[0] = static_cast<doublereal>(*n > 1 ? *n : 1);
    *info = 0;
    return 0;
  }
  char lower = 'L';
  if (!lsame_(uplo, &lower, uplo_len, 1)) {
    *info = -100;
    return 0;
  }
  return dsytd2_(uplo, n, a, lda, d, e, tau, info, uplo_len);
}

constexpr int kMaximumDimension =
  FASTKPC_MULTI_PENALTY_MAX_PENALTY_COUNT;

struct Workspace {
  doublereal matrix[kMaximumDimension * kMaximumDimension];
  doublereal values[kMaximumDimension];
  doublereal vectors[kMaximumDimension * kMaximumDimension];
  doublereal work[26 * kMaximumDimension];
  integer iwork[10 * kMaximumDimension];
  integer support[2 * kMaximumDimension];
};

__device__ bool solve_lower(
    const doublereal* input, integer dimension, Workspace* workspace) {
  if (input == nullptr || workspace == nullptr || dimension <= 0 ||
      dimension > kMaximumDimension) {
    return false;
  }
  for (integer index = 0; index < dimension * dimension; ++index) {
    workspace->matrix[index] = input[index];
  }
  char job = 'V';
  char range = 'A';
  char uplo = 'L';
  doublereal lower_value = 0.0;
  doublereal upper_value = 0.0;
  integer lower_index = 0;
  integer upper_index = 0;
  doublereal absolute_tolerance = 0.0;
  integer eigenvalue_count = 0;
  integer leading_dimension = dimension;
  integer workspace_size = 26 * dimension;
  integer integer_workspace_size = 10 * dimension;
  integer info = 0;
  dsyevr_(&job, &range, &uplo, &dimension, workspace->matrix,
          &leading_dimension, &lower_value, &upper_value, &lower_index,
          &upper_index, &absolute_tolerance, &eigenvalue_count,
          workspace->values, workspace->vectors, &leading_dimension,
          workspace->support, workspace->work, &workspace_size,
          workspace->iwork, &integer_workspace_size, &info, 1, 1, 1);
  if (info != 0 || eigenvalue_count != dimension) return false;
  for (integer index = 0; index < dimension; ++index) {
    if (!isfinite(workspace->values[index])) return false;
  }
  for (integer index = 0; index < dimension * dimension; ++index) {
    if (!isfinite(workspace->vectors[index])) return false;
  }
  return true;
}

}  // namespace symmetric_eigen
}  // namespace lapack312_multi
}  // namespace fastkpc

#endif
