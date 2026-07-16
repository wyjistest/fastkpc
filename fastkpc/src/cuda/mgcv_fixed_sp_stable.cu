#include "mgcv_fixed_sp_stable.cuh"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace fastkpc {
namespace {

std::size_t positive_size(int value, const char* name) {
  if (value <= 0) {
    throw std::runtime_error(std::string(name) + " must be positive");
  }
  return static_cast<std::size_t>(value);
}

std::size_t checked_add(std::size_t left,
                        std::size_t right,
                        const char* name) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::runtime_error(std::string(name) + " size overflow");
  }
  return left + right;
}

std::size_t checked_multiply(std::size_t left,
                             std::size_t right,
                             const char* name) {
  if (left != 0U &&
      right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::runtime_error(std::string(name) + " size overflow");
  }
  return left * right;
}

std::size_t nonnegative_size(int value, const char* name) {
  if (value < 0) {
    throw std::runtime_error(std::string(name) + " is invalid");
  }
  return static_cast<std::size_t>(value);
}

void check_cusolver(cusolverStatus_t status, const char* stage) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(stage) + ": cuSOLVER status " +
                             std::to_string(static_cast<int>(status)));
  }
}

double* take(double** cursor, std::size_t count) {
  double* result = *cursor;
  *cursor += count;
  return result;
}

std::size_t data_double_count(int max_rows, int max_q) {
  const std::size_t rows = positive_size(max_rows, "stable maximum rows");
  const std::size_t q = positive_size(max_q, "stable maximum q");
  const std::size_t rows_q = checked_multiply(
    rows, q, "stable augmented matrix");
  const std::size_t q_squared = checked_multiply(
    q, q, "stable square matrix");
  std::size_t count = checked_multiply(
    rows_q, 2U, "stable B and U storage");
  count = checked_add(count, rows, "stable vector storage");
  count = checked_add(
    count, checked_multiply(q, 3U, "stable q-vector storage"),
    "stable vector storage");
  return checked_add(
    count, checked_multiply(q_squared, 2U, "stable square storage"),
    "stable data storage");
}

void bind_data_views(FixedSpStableWorkspace* workspace, double** cursor) {
  const std::size_t rows = static_cast<std::size_t>(workspace->max_rows);
  const std::size_t q = static_cast<std::size_t>(workspace->max_q);
  const std::size_t rows_q = rows * q;
  const std::size_t q_squared = q * q;
  workspace->B = take(cursor, rows_q);
  workspace->c = take(cursor, rows);
  workspace->tau = take(cursor, q);
  workspace->singular_values = take(cursor, q);
  workspace->U = take(cursor, rows_q);
  workspace->V = take(cursor, q_squared);
  workspace->scaled_projection = take(cursor, q_squared);
  workspace->diagonal = take(cursor, q);
}

}  // namespace

std::size_t fixed_sp_stable_probe_double_count(int max_rows, int max_q) {
  return data_double_count(std::max(max_rows, max_q), max_q);
}

FixedSpStableWorkspace fixed_sp_stable_probe_view(
    double* storage, int max_rows, int max_q) {
  if (storage == nullptr) {
    throw std::runtime_error("stable reserve probe storage is null");
  }
  const int probe_rows = std::max(max_rows, max_q);
  data_double_count(probe_rows, max_q);
  FixedSpStableWorkspace workspace;
  workspace.max_rows = probe_rows;
  workspace.max_q = max_q;
  double* cursor = storage;
  bind_data_views(&workspace, &cursor);
  workspace.max_rows = max_rows;
  return workspace;
}

void query_fixed_sp_stable_workspace(
    cusolverDnHandle_t solver,
    gesvdjInfo_t svd_params,
    FixedSpStableWorkspace* workspace) {
  if (solver == nullptr || svd_params == nullptr || workspace == nullptr ||
      workspace->B == nullptr || workspace->c == nullptr ||
      workspace->tau == nullptr || workspace->singular_values == nullptr ||
      workspace->U == nullptr || workspace->V == nullptr ||
      workspace->scaled_projection == nullptr ||
      workspace->diagonal == nullptr) {
    throw std::runtime_error("stable workspace query inputs are unavailable");
  }
  positive_size(workspace->max_rows, "stable maximum rows");
  positive_size(workspace->max_q, "stable maximum q");
  const int probe_rows = std::max(workspace->max_rows, workspace->max_q);

  check_cusolver(cusolverDnDsyevd_bufferSize(
    solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
    workspace->max_q, workspace->scaled_projection, workspace->max_q,
    workspace->diagonal, &workspace->eigen_lwork
  ), "query Phase 3C eigensolver workspace");
  check_cusolver(cusolverDnDgeqrf_bufferSize(
    solver, probe_rows, workspace->max_q,
    workspace->B, probe_rows, &workspace->qr_lwork
  ), "query Phase 3C QR workspace");
  check_cusolver(cusolverDnDormqr_bufferSize(
    solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
    probe_rows, 1, workspace->max_q,
    workspace->B, probe_rows, workspace->tau,
    workspace->c, probe_rows, &workspace->ormqr_lwork
  ), "query Phase 3C ormqr workspace");
  check_cusolver(cusolverDnDgesvdj_bufferSize(
    solver, CUSOLVER_EIG_MODE_VECTOR, 1,
    probe_rows, workspace->max_q,
    workspace->B, probe_rows, workspace->singular_values,
    workspace->U, probe_rows,
    workspace->V, workspace->max_q,
    &workspace->svd_lwork, svd_params
  ), "query Phase 3C SVD workspace");

  if (workspace->eigen_lwork <= 0 || workspace->qr_lwork <= 0 ||
      workspace->ormqr_lwork <= 0 || workspace->svd_lwork <= 0) {
    throw std::runtime_error("Phase 3C workspace query returned invalid size");
  }
}

std::size_t fixed_sp_stable_workspace_double_count(
    int max_rows,
    int max_q,
    int eigen_lwork,
    int qr_lwork,
    int ormqr_lwork,
    int svd_lwork) {
  std::size_t count = data_double_count(max_rows, max_q);
  count = checked_add(
    count, nonnegative_size(eigen_lwork, "stable eigen lwork"),
    "stable workspace");
  count = checked_add(
    count,
    nonnegative_size(std::max(qr_lwork, ormqr_lwork), "stable QR lwork"),
    "stable workspace");
  return checked_add(
    count, nonnegative_size(svd_lwork, "stable SVD lwork"),
    "stable workspace");
}

FixedSpStableWorkspace fixed_sp_stable_workspace_view(
    double* storage,
    int* info,
    int max_rows,
    int max_q,
    int eigen_lwork,
    int qr_lwork,
    int ormqr_lwork,
    int svd_lwork) {
  if (storage == nullptr || info == nullptr) {
    throw std::runtime_error("stable persistent workspace storage is null");
  }
  fixed_sp_stable_workspace_double_count(
    max_rows, max_q, eigen_lwork, qr_lwork, ormqr_lwork, svd_lwork);

  FixedSpStableWorkspace workspace;
  workspace.info = info;
  workspace.eigen_lwork = eigen_lwork;
  workspace.qr_lwork = qr_lwork;
  workspace.ormqr_lwork = ormqr_lwork;
  workspace.svd_lwork = svd_lwork;
  workspace.max_rows = max_rows;
  workspace.max_q = max_q;

  const std::size_t q = static_cast<std::size_t>(max_q);
  const std::size_t rows = static_cast<std::size_t>(max_rows);
  double* cursor = storage;
  workspace.B = take(&cursor, rows * q);
  workspace.c = take(&cursor, rows);
  workspace.tau = take(&cursor, q);
  workspace.eigen_work = take(
    &cursor, static_cast<std::size_t>(eigen_lwork));
  workspace.qr_work = take(
    &cursor, static_cast<std::size_t>(std::max(qr_lwork, ormqr_lwork)));
  workspace.svd_work = take(
    &cursor, static_cast<std::size_t>(svd_lwork));
  workspace.singular_values = take(&cursor, q);
  workspace.U = take(&cursor, rows * q);
  workspace.V = take(&cursor, q * q);
  workspace.scaled_projection = take(&cursor, q * q);
  workspace.diagonal = take(&cursor, q);
  return workspace;
}

}  // namespace fastkpc
