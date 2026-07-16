#include "mgcv_fixed_sp_stable.cuh"

#include <algorithm>
#include <cmath>
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

void check_cuda(cudaError_t status, const char* stage) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(stage) + ": " +
                             cudaGetErrorString(status));
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

__global__ void build_fixed_sp_root_kernel(
    const double* eigenvectors,
    const double* eigenvalues,
    const int* solver_info,
    int q,
    double* roots,
    int root_leading_dimension,
    int root_row_offset,
    int root_row_capacity,
    int expected_rank,
    double epsilon,
    FixedSpRootValidationRecord* validation) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  FixedSpRootValidationRecord record;
  record.solver_info = *solver_info;
  if (record.solver_info != 0) {
    *validation = record;
    return;
  }

  const std::size_t q_size = static_cast<std::size_t>(q);
  const std::size_t eigenvector_count = q_size * q_size;
  const std::size_t root_leading_dimension_size =
    static_cast<std::size_t>(root_leading_dimension);
  const std::size_t root_row_offset_size =
    static_cast<std::size_t>(root_row_offset);
  bool finite = true;
  bool ascending = true;
  double maximum_absolute_eigenvalue = 0.0;
  for (std::size_t index = 0; index < q_size; ++index) {
    const double value = eigenvalues[index];
    finite = finite && isfinite(value);
    if (index > 0U && value < eigenvalues[index - 1U]) ascending = false;
    maximum_absolute_eigenvalue =
      fmax(maximum_absolute_eigenvalue, fabs(value));
  }
  for (std::size_t index = 0; index < eigenvector_count; ++index) {
    finite = finite && isfinite(eigenvectors[index]);
  }

  const double tolerance =
    static_cast<double>(q) * maximum_absolute_eigenvalue * epsilon;
  finite = finite && isfinite(tolerance);
  bool psd = finite;
  int retained_rank = 0;
  if (finite) {
    for (std::size_t index = 0; index < q_size; ++index) {
      if (eigenvalues[index] < -tolerance) psd = false;
      if (eigenvalues[index] > tolerance) retained_rank += 1;
    }
  }

  record.finite = finite ? 1 : 0;
  record.ascending = ascending ? 1 : 0;
  record.psd = psd ? 1 : 0;
  record.rank = retained_rank;
  *validation = record;

  const bool rank_matches =
    expected_rank < 0 || retained_rank == expected_rank;
  if (!finite || !ascending || !psd || !rank_matches ||
      retained_rank > root_row_capacity) {
    return;
  }

  std::size_t root_row = 0U;
  for (std::size_t eigen_index = 0; eigen_index < q_size; ++eigen_index) {
    const double eigenvalue = eigenvalues[eigen_index];
    if (eigenvalue <= tolerance) continue;

    const std::size_t eigenvector_column_offset = q_size * eigen_index;
    std::size_t pivot = 0U;
    double pivot_absolute = -1.0;
    for (std::size_t component = 0; component < q_size; ++component) {
      const double candidate = fabs(
        eigenvectors[component + eigenvector_column_offset]);
      if (candidate > pivot_absolute) {
        pivot = component;
        pivot_absolute = candidate;
      }
    }
    const double sign =
      eigenvectors[pivot + eigenvector_column_offset] < 0.0 ? -1.0 : 1.0;
    const double scale = sign * sqrt(eigenvalue);
    for (std::size_t column = 0; column < q_size; ++column) {
      const std::size_t root_index = root_row_offset_size + root_row +
        root_leading_dimension_size * column;
      roots[root_index] =
        scale * eigenvectors[column + eigenvector_column_offset];
    }
    root_row += 1U;
  }
}

__global__ void copy_fixed_sp_augmented_design_kernel(
    const double* X_null,
    int n,
    int q,
    double* B,
    int leading_dimension) {
  const std::size_t flattened_index =
    static_cast<std::size_t>(blockIdx.x) *
      static_cast<std::size_t>(blockDim.x) +
    static_cast<std::size_t>(threadIdx.x);
  const std::size_t n_size = static_cast<std::size_t>(n);
  const std::size_t q_size = static_cast<std::size_t>(q);
  const std::size_t element_count = n_size * q_size;
  if (flattened_index >= element_count) return;

  const std::size_t row = flattened_index % n_size;
  const std::size_t column = flattened_index / n_size;
  const std::size_t source_index = row + n_size * column;
  const std::size_t destination_index = row +
    static_cast<std::size_t>(leading_dimension) * column;
  B[destination_index] = X_null[source_index];
}

__global__ void build_fixed_sp_augmented_response_kernel(
    const double* Y,
    int n,
    double* c,
    int rows) {
  const std::size_t row =
    static_cast<std::size_t>(blockIdx.x) *
      static_cast<std::size_t>(blockDim.x) +
    static_cast<std::size_t>(threadIdx.x);
  const std::size_t rows_size = static_cast<std::size_t>(rows);
  if (row >= rows_size) return;
  c[row] = row < static_cast<std::size_t>(n) ? Y[row] : 0.0;
}

__global__ void scale_fixed_sp_augmented_root_kernel(
    const double* penalty_roots,
    int root_leading_dimension,
    int root_row_offset,
    int rank,
    int q,
    const double* SP,
    int sp_index,
    double* B,
    int B_leading_dimension,
    int B_row_offset) {
  const std::size_t flattened_index =
    static_cast<std::size_t>(blockIdx.x) *
      static_cast<std::size_t>(blockDim.x) +
    static_cast<std::size_t>(threadIdx.x);
  const std::size_t rank_size = static_cast<std::size_t>(rank);
  const std::size_t q_size = static_cast<std::size_t>(q);
  const std::size_t element_count = rank_size * q_size;
  if (flattened_index >= element_count) return;

  const std::size_t row = flattened_index % rank_size;
  const std::size_t column = flattened_index / rank_size;
  const std::size_t source_index =
    static_cast<std::size_t>(root_row_offset) + row +
    static_cast<std::size_t>(root_leading_dimension) * column;
  const std::size_t destination_index =
    static_cast<std::size_t>(B_row_offset) + row +
    static_cast<std::size_t>(B_leading_dimension) * column;
  const double scale = sqrt(SP[static_cast<std::size_t>(sp_index)]);
  B[destination_index] = scale * penalty_roots[source_index];
}

__global__ void copy_fixed_sp_augmented_H_root_kernel(
    const double* H_root,
    int source_leading_dimension,
    int rank,
    int q,
    double* B,
    int B_leading_dimension,
    int B_row_offset) {
  const std::size_t flattened_index =
    static_cast<std::size_t>(blockIdx.x) *
      static_cast<std::size_t>(blockDim.x) +
    static_cast<std::size_t>(threadIdx.x);
  const std::size_t rank_size = static_cast<std::size_t>(rank);
  const std::size_t q_size = static_cast<std::size_t>(q);
  const std::size_t element_count = rank_size * q_size;
  if (flattened_index >= element_count) return;

  const std::size_t row = flattened_index % rank_size;
  const std::size_t column = flattened_index / rank_size;
  const std::size_t source_index = row +
    static_cast<std::size_t>(source_leading_dimension) * column;
  const std::size_t destination_index =
    static_cast<std::size_t>(B_row_offset) + row +
    static_cast<std::size_t>(B_leading_dimension) * column;
  B[destination_index] = H_root[source_index];
}

__global__ void fixed_sp_qr_rank_status_kernel(
    const double* R,
    const double* theta,
    int rows,
    int q,
    int target_index,
    double epsilon,
    int* qr_rank,
    int* reroute,
    int* finite_status) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) return;

  const std::size_t rows_size = static_cast<std::size_t>(rows);
  const std::size_t q_size = static_cast<std::size_t>(q);
  bool diagonal_finite = true;
  bool finite = true;
  double max_diag = 0.0;
  for (std::size_t index = 0U; index < q_size; ++index) {
    const double diagonal = R[index + rows_size * index];
    diagonal_finite = diagonal_finite && isfinite(diagonal);
    finite = finite && isfinite(diagonal) && isfinite(theta[index]);
    max_diag = fmax(max_diag, fabs(diagonal));
  }
  const double rank_tol =
    static_cast<double>(rows > q ? rows : q) * max_diag * epsilon;
  diagonal_finite = diagonal_finite && isfinite(rank_tol);
  finite = finite && isfinite(rank_tol);

  int rank = 0;
  if (diagonal_finite) {
    for (std::size_t index = 0U; index < q_size; ++index) {
      if (fabs(R[index + rows_size * index]) > rank_tol) rank += 1;
    }
  }
  const std::size_t target = static_cast<std::size_t>(target_index);
  qr_rank[target] = rank;
  reroute[target] = rank < q ? 1 : 0;
  finite_status[target] = finite ? 0 : 1;
}

unsigned int fixed_sp_augmented_block_count(
    std::size_t element_count,
    const char* name) {
  constexpr std::size_t threads = 256U;
  const std::size_t blocks = (element_count + threads - 1U) / threads;
  if (blocks == 0U ||
      blocks > static_cast<std::size_t>(
        std::numeric_limits<unsigned int>::max())) {
    throw std::runtime_error(std::string(name) + " launch size overflow");
  }
  return static_cast<unsigned int>(blocks);
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

void launch_fixed_sp_root_build(
    const double* eigenvectors,
    const double* eigenvalues,
    const int* solver_info,
    int q,
    double* roots,
    int root_leading_dimension,
    int root_row_offset,
    int root_row_capacity,
    int expected_rank,
    double epsilon,
    FixedSpRootValidationRecord* validation,
    cudaStream_t stream) {
  if (eigenvectors == nullptr || eigenvalues == nullptr ||
      solver_info == nullptr || validation == nullptr || stream == nullptr ||
      q <= 0 || root_leading_dimension < 0 || root_row_offset < 0 ||
      root_row_capacity < 0 ||
      root_row_offset > root_leading_dimension - root_row_capacity ||
      (root_row_capacity > 0 && roots == nullptr) ||
      !std::isfinite(epsilon) || epsilon <= 0.0) {
    throw std::runtime_error("fixed-sp root build inputs are invalid");
  }
  build_fixed_sp_root_kernel<<<1, 1, 0, stream>>>(
    eigenvectors, eigenvalues, solver_info, q, roots,
    root_leading_dimension, root_row_offset, root_row_capacity,
    expected_rank, epsilon, validation);
  check_cuda(cudaGetLastError(), "launch fixed-sp root build kernel");
}

AugmentedSystemView build_fixed_sp_augmented_system(
    const double* X_null,
    const double* penalty_roots,
    const int* penalty_root_offsets,
    const int* penalty_root_ranks,
    int total_penalty_root_rows,
    int penalty_count,
    const double* H_root,
    int H_root_rank,
    const double* Y,
    const double* SP,
    const double* host_SP,
    int n,
    int q,
    int target_index,
    FixedSpStableWorkspace* workspace,
    cudaStream_t stream) {
  const std::size_t n_size = positive_size(n, "augmented row count");
  const std::size_t q_size = positive_size(q, "augmented column count");
  const std::size_t penalty_count_size =
    positive_size(penalty_count, "augmented penalty count");
  const std::size_t smooth_rows = nonnegative_size(
    total_penalty_root_rows, "augmented smooth root rows");
  const std::size_t H_rows =
    nonnegative_size(H_root_rank, "augmented H root rows");
  const std::size_t root_rows = checked_add(
    smooth_rows, H_rows, "augmented root rows");
  const std::size_t rows_size =
    checked_add(n_size, root_rows, "augmented rows");
  if (rows_size >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("augmented row count overflow");
  }
  const int rows = static_cast<int>(rows_size);

  if (X_null == nullptr || Y == nullptr || SP == nullptr ||
      host_SP == nullptr || penalty_root_offsets == nullptr ||
      penalty_root_ranks == nullptr || workspace == nullptr ||
      workspace->B == nullptr || workspace->c == nullptr ||
      stream == nullptr || target_index < 0 || H_root_rank > q ||
      rows > workspace->max_rows || q > workspace->max_q ||
      (smooth_rows > 0U && penalty_roots == nullptr) ||
      (H_rows > 0U && H_root == nullptr)) {
    throw std::runtime_error("fixed-sp augmented build inputs are invalid");
  }

  std::size_t expected_root_offset = 0U;
  for (std::size_t penalty = 0U; penalty < penalty_count_size; ++penalty) {
    const int offset = penalty_root_offsets[penalty];
    const int rank = penalty_root_ranks[penalty];
    if (offset < 0 || rank < 0 ||
        static_cast<std::size_t>(offset) != expected_root_offset ||
        static_cast<std::size_t>(rank) >
          smooth_rows - expected_root_offset) {
      throw std::runtime_error(
        "fixed-sp augmented root metadata is invalid");
    }
    expected_root_offset = checked_add(
      expected_root_offset, static_cast<std::size_t>(rank),
      "augmented root metadata");
    if (!std::isfinite(host_SP[penalty]) || host_SP[penalty] < 0.0) {
      throw std::runtime_error(
        "augmented SP must contain only finite non-negative values");
    }
  }
  if (expected_root_offset != smooth_rows) {
    throw std::runtime_error("fixed-sp augmented root metadata is invalid");
  }

  constexpr unsigned int threads = 256U;
  const std::size_t design_count = checked_multiply(
    n_size, q_size, "augmented design");
  copy_fixed_sp_augmented_design_kernel<<<
    fixed_sp_augmented_block_count(design_count, "augmented design"),
    threads, 0, stream
  >>>(X_null, n, q, workspace->B, rows);
  check_cuda(cudaGetLastError(),
             "launch fixed-sp augmented design copy");

  build_fixed_sp_augmented_response_kernel<<<
    fixed_sp_augmented_block_count(rows_size, "augmented response"),
    threads, 0, stream
  >>>(Y, n, workspace->c, rows);
  check_cuda(cudaGetLastError(),
             "launch fixed-sp augmented response build");

  for (std::size_t penalty = 0U; penalty < penalty_count_size; ++penalty) {
    const int rank = penalty_root_ranks[penalty];
    if (rank == 0) continue;
    const std::size_t block_count = checked_multiply(
      static_cast<std::size_t>(rank), q_size,
      "augmented smooth root block");
    const int destination_row = static_cast<int>(
      n_size + static_cast<std::size_t>(penalty_root_offsets[penalty]));
    scale_fixed_sp_augmented_root_kernel<<<
      fixed_sp_augmented_block_count(
        block_count, "augmented smooth root block"),
      threads, 0, stream
    >>>(
      penalty_roots, total_penalty_root_rows,
      penalty_root_offsets[penalty], rank, q, SP,
      static_cast<int>(penalty), workspace->B, rows, destination_row);
    check_cuda(cudaGetLastError(),
               "launch fixed-sp augmented smooth root scale");
  }

  if (H_root_rank > 0) {
    const std::size_t H_count = checked_multiply(
      H_rows, q_size, "augmented H root block");
    const int destination_row =
      static_cast<int>(n_size + smooth_rows);
    copy_fixed_sp_augmented_H_root_kernel<<<
      fixed_sp_augmented_block_count(H_count, "augmented H root block"),
      threads, 0, stream
    >>>(H_root, q, H_root_rank, q, workspace->B, rows, destination_row);
    check_cuda(cudaGetLastError(),
               "launch fixed-sp augmented H root copy");
  }

  AugmentedSystemView view;
  view.B = workspace->B;
  view.c = workspace->c;
  view.leading_dimension = rows;
  view.rows = rows;
  view.cols = q;
  view.target_index = target_index;
  return view;
}

void launch_fixed_sp_qr_rank_status(
    const double* R,
    const double* theta,
    int rows,
    int q,
    int target_index,
    double epsilon,
    int* qr_rank,
    int* reroute,
    int* finite_status,
    cudaStream_t stream) {
  if (R == nullptr || theta == nullptr || rows <= 0 || q <= 0 || rows < q ||
      target_index < 0 || !std::isfinite(epsilon) || epsilon <= 0.0 ||
      qr_rank == nullptr || reroute == nullptr || finite_status == nullptr ||
      stream == nullptr) {
    throw std::runtime_error("fixed-sp QR rank status inputs are invalid");
  }
  fixed_sp_qr_rank_status_kernel<<<1U, 1U, 0, stream>>>(
    R, theta, rows, q, target_index, epsilon,
    qr_rank, reroute, finite_status);
  check_cuda(cudaGetLastError(), "launch fixed-sp QR rank status kernel");
}

}  // namespace fastkpc
