#include "mgcv_multi_penalty_cpp.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fastkpc {
namespace {

using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::ColMajor>;
using Vector = Eigen::VectorXd;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

Matrix symmetric_matrix(const Matrix& value, const char* label) {
  require(value.rows() == value.cols(), "penalty matrix must be square");
  require(value.allFinite(), "penalty matrix must be finite");
  const double scale = std::max(1.0, value.cwiseAbs().maxCoeff());
  const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() *
    static_cast<double>(std::max<Eigen::Index>(1, value.rows())) * scale;
  if ((value - value.transpose()).cwiseAbs().maxCoeff() > tolerance) {
    throw std::runtime_error(std::string(label) + " must be symmetric");
  }
  return 0.5 * (value + value.transpose());
}

std::string condition_bucket(int rank, int dimension, double condition) {
  if (rank < dimension || !std::isfinite(condition)) {
    return "rank-deficient-or-unauthenticated-condition";
  }
  if (condition < 1e8) return "finite-full-rank-lt-1e8";
  if (condition < 1e12) return "finite-full-rank-1e8-to-lt-1e12";
  return "finite-full-rank-ge-1e12";
}

Matrix constraint_null_space(const double* constraint,
                             int constraint_rows,
                             int coefficient_dim,
                             double rank_tolerance,
                             int* constraint_rank) {
  *constraint_rank = 0;
  if (constraint_rows == 0) {
    return Matrix::Identity(coefficient_dim, coefficient_dim);
  }
  Eigen::Map<const Matrix> C(constraint, constraint_rows, coefficient_dim);
  require(C.allFinite(), "constraint matrix must be finite");
  Eigen::JacobiSVD<Matrix> decomposition(C, Eigen::ComputeFullV);
  const Vector singular = decomposition.singularValues();
  const double threshold = singular.size() == 0 ? 0.0 :
    singular[0] * rank_tolerance;
  int rank = 0;
  for (Eigen::Index index = 0; index < singular.size(); ++index) {
    if (singular[index] >= threshold && singular[index] > 0.0) ++rank;
  }
  require(rank < coefficient_dim,
          "constraint matrix leaves no free coefficient space");
  *constraint_rank = rank;
  return decomposition.matrixV().rightCols(coefficient_dim - rank);
}

void validate_penalty_rank(const Matrix& penalty, int expected_rank) {
  Eigen::SelfAdjointEigenSolver<Matrix> decomposition(penalty);
  require(decomposition.info() == Eigen::Success,
          "penalty eigendecomposition failed");
  const Vector values = decomposition.eigenvalues();
  const Matrix vectors = decomposition.eigenvectors();
  const double spectral_scale =
    std::max(1.0, values.cwiseAbs().maxCoeff());
  const double negative_tolerance = 64.0 *
    std::numeric_limits<double>::epsilon() *
    static_cast<double>(std::max<Eigen::Index>(1, penalty.rows())) *
    spectral_scale;
  require(values[0] >= -negative_tolerance,
          "penalty matrix is not positive semidefinite");
  const double positive_tolerance =
    std::numeric_limits<double>::epsilon() *
    static_cast<double>(std::max<Eigen::Index>(1, penalty.rows())) *
    spectral_scale;
  int available_rank = 0;
  for (Eigen::Index index = 0; index < values.size(); ++index) {
    if (values[index] > positive_tolerance) ++available_rank;
  }
  require(available_rank >= std::min<int>(expected_rank, penalty.rows()),
          "authenticated penalty rank exceeds the stable penalty rank");
}

Matrix pivoted_cholesky_root(
    const Matrix& penalty,
    int* root_rank,
    int requested_rank = 0,
    double tolerance = -1.0) {
  require(root_rank != nullptr, "penalty root rank pointer is missing");
  Matrix factor = penalty;
  const La_INT dimension = static_cast<La_INT>(penalty.rows());
  const La_INT leading_dimension = dimension;
  std::vector<La_INT> pivot(static_cast<std::size_t>(dimension));
  std::vector<double> work(static_cast<std::size_t>(2 * dimension));
  La_INT rank = 0;
  La_INT info = 0;
  const char uplo = 'U';
  F77_CALL(dpstrf)(
    &uplo, &dimension, factor.data(), &leading_dimension, pivot.data(),
    &rank, &tolerance, work.data(), &info FCONE);
  require(info >= 0 && rank > 0 && rank <= dimension,
          "aggregate penalty pivoted Cholesky failed");
  require(requested_rank >= 0 &&
            (requested_rank == 0 || requested_rank <= rank),
          "requested penalty root rank is unavailable");
  if (requested_rank > 0) rank = static_cast<La_INT>(requested_rank);
  Matrix root = Matrix::Zero(rank, dimension);
  for (La_INT column = 0; column < dimension; ++column) {
    const La_INT original_column = pivot[static_cast<std::size_t>(column)] - 1;
    require(original_column >= 0 && original_column < dimension,
            "aggregate penalty pivot is invalid");
    const La_INT copied_rows = std::min<La_INT>(rank, column + 1);
    for (La_INT row = 0; row < copied_rows; ++row) {
      root(row, original_column) = factor(row, column);
    }
  }
  require(root.allFinite(), "aggregate penalty root is non-finite");
  *root_rank = static_cast<int>(rank);
  return root;
}

Matrix blas_multiply(
    const Matrix& left,
    bool transpose_left,
    const Matrix& right,
    bool transpose_right) {
  const La_INT rows = static_cast<La_INT>(
    transpose_left ? left.cols() : left.rows());
  const La_INT common_left = static_cast<La_INT>(
    transpose_left ? left.rows() : left.cols());
  const La_INT common_right = static_cast<La_INT>(
    transpose_right ? right.cols() : right.rows());
  const La_INT columns = static_cast<La_INT>(
    transpose_right ? right.rows() : right.cols());
  require(rows > 0 && columns > 0 && common_left > 0 &&
            common_left == common_right,
          "BLAS matrix product dimensions are invalid");
  const char left_operation = transpose_left ? 'T' : 'N';
  const char right_operation = transpose_right ? 'T' : 'N';
  const La_INT leading_left = static_cast<La_INT>(left.rows());
  const La_INT leading_right = static_cast<La_INT>(right.rows());
  const La_INT leading_result = rows;
  const double alpha = 1.0;
  const double beta = 0.0;
  Matrix result(rows, columns);
  F77_CALL(dgemm)(
    &left_operation, &right_operation, &rows, &columns, &common_left,
    &alpha, left.data(), &leading_left, right.data(), &leading_right,
    &beta, result.data(), &leading_result FCONE FCONE);
  require(result.allFinite(), "BLAS matrix product is non-finite");
  return result;
}

Matrix blas_symmetric_xxt(const Matrix& input) {
  require(input.rows() > 0 && input.cols() > 0,
          "BLAS XXt input dimensions are invalid");
  const La_INT dimension = static_cast<La_INT>(input.rows());
  const La_INT common = static_cast<La_INT>(input.cols());
  const La_INT leading_input = dimension;
  const La_INT leading_result = dimension;
  const char uplo = 'L';
  const char transpose = 'N';
  const double alpha = 1.0;
  const double beta = 0.0;
  Matrix result = Matrix::Zero(dimension, dimension);
  F77_CALL(dsyrk)(
    &uplo, &transpose, &dimension, &common, &alpha, input.data(),
    &leading_input, &beta, result.data(), &leading_result FCONE FCONE);
  for (La_INT column = 0; column < dimension; ++column) {
    for (La_INT row = 0; row < column; ++row) {
      result(row, column) = result(column, row);
    }
  }
  require(result.allFinite(), "BLAS XXt product is non-finite");
  return result;
}

Matrix blas_symmetric_xtx(const Matrix& input) {
  require(input.rows() > 0 && input.cols() > 0,
          "BLAS XtX input dimensions are invalid");
  const La_INT dimension = static_cast<La_INT>(input.cols());
  const La_INT common = static_cast<La_INT>(input.rows());
  const La_INT leading_input = common;
  const La_INT leading_result = dimension;
  const char uplo = 'L';
  const char transpose = 'T';
  const double alpha = 1.0;
  const double beta = 0.0;
  Matrix result = Matrix::Zero(dimension, dimension);
  F77_CALL(dsyrk)(
    &uplo, &transpose, &dimension, &common, &alpha, input.data(),
    &leading_input, &beta, result.data(), &leading_result FCONE FCONE);
  for (La_INT column = 0; column < dimension; ++column) {
    for (La_INT row = 0; row < column; ++row) {
      result(row, column) = result(column, row);
    }
  }
  require(result.allFinite(), "BLAS XtX product is non-finite");
  return result;
}

std::vector<double> copy_vector(const Vector& value) {
  return std::vector<double>(value.data(), value.data() + value.size());
}

double r_corrected_mean(const std::vector<double>& values) {
  require(!values.empty(), "mean input is empty");
  long double mean = 0.0L;
  for (double value : values) mean += value;
  mean /= static_cast<long double>(values.size());
  long double correction = 0.0L;
  for (double value : values) {
    correction += static_cast<long double>(value) - mean;
  }
  mean += correction / static_cast<long double>(values.size());
  return static_cast<double>(mean);
}

Vector initial_log_sp(
    const double* X_data,
    int n,
    int coefficient_dim,
    const std::vector<std::vector<double>>& penalty_blocks,
    const std::vector<int>& penalty_dimensions,
    const std::vector<int>& penalty_offsets) {
  const std::size_t penalty_count = penalty_blocks.size();
  Eigen::Map<const Matrix> X(X_data, n, coefficient_dim);
  Vector design_diagonal(coefficient_dim);
  for (int column = 0; column < coefficient_dim; ++column) {
    long double value = 0.0L;
    for (int row = 0; row < n; ++row) {
      value += X(row, column) * X(row, column);
    }
    design_diagonal[column] = static_cast<double>(value);
  }
  Vector scaled_penalty_diagonal = Vector::Zero(coefficient_dim);
  std::vector<unsigned char> penalized(
    static_cast<std::size_t>(coefficient_dim), 0);
  Vector initial(penalty_count);
  const double threshold_scale = std::pow(
    std::numeric_limits<double>::epsilon(), 0.8);
  for (std::size_t index = 0; index < penalty_count; ++index) {
    const int dimension = penalty_dimensions[index];
    const int offset = penalty_offsets[index];
    Eigen::Map<const Matrix> block(
      penalty_blocks[index].data(), dimension, dimension);
    const Matrix absolute = block.cwiseAbs();
    const double threshold = threshold_scale * absolute.maxCoeff();
    std::vector<double> selected_design;
    std::vector<double> selected_penalty;
    selected_design.reserve(static_cast<std::size_t>(dimension));
    selected_penalty.reserve(static_cast<std::size_t>(dimension));
    int selected = 0;
    for (int row = 0; row < dimension; ++row) {
      const double row_mean = absolute.row(row).mean();
      const double column_mean = absolute.col(row).mean();
      const double diagonal = absolute(row, row);
      if (row_mean > threshold && column_mean > threshold &&
          diagonal > threshold) {
        selected_design.push_back(design_diagonal[offset + row]);
        selected_penalty.push_back(block(row, row));
        penalized[static_cast<std::size_t>(offset + row)] = 1;
        ++selected;
      }
    }
    require(selected > 0,
            "multi-penalty initial smoothing scale is invalid");
    const double design_mean = r_corrected_mean(selected_design);
    const double penalty_mean = r_corrected_mean(selected_penalty);
    require(penalty_mean > 0.0,
            "multi-penalty initial smoothing scale is invalid");
    initial[static_cast<Eigen::Index>(index)] =
      design_mean / penalty_mean;
    for (int row = 0; row < dimension; ++row) {
      scaled_penalty_diagonal[offset + row] +=
        initial[static_cast<Eigen::Index>(index)] * block(row, row);
    }
  }
  auto mean_influence = [&]() {
    std::vector<double> influence;
    influence.reserve(static_cast<std::size_t>(coefficient_dim));
    for (int column = 0; column < coefficient_dim; ++column) {
      if (penalized[static_cast<std::size_t>(column)] != 0 &&
          design_diagonal[column] > 0.0 &&
          scaled_penalty_diagonal[column] > 0.0) {
        influence.push_back(
          design_diagonal[column] /
            (design_diagonal[column] + scaled_penalty_diagonal[column]));
      }
    }
    require(!influence.empty(),
            "multi-penalty initial smoothing support is empty");
    return r_corrected_mean(influence);
  };
  int scaling_iterations = 0;
  while (mean_influence() > 0.4) {
    initial *= 10.0;
    scaled_penalty_diagonal *= 10.0;
    require(++scaling_iterations <= 400,
            "multi-penalty initial smoothing scaling did not converge");
  }
  while (mean_influence() < 0.4) {
    initial /= 10.0;
    scaled_penalty_diagonal /= 10.0;
    require(++scaling_iterations <= 400,
            "multi-penalty initial smoothing scaling did not converge");
  }
  require(initial.allFinite() && (initial.array() > 0.0).all(),
          "multi-penalty initial smoothing vector is invalid");
  Vector log_initial(initial.size());
  for (Eigen::Index index = 0; index < initial.size(); ++index) {
    log_initial[index] = std::log(initial[index]);
  }
  return log_initial;
}

struct PreparedMultiPenaltyProblem {
  int n = 0;
  int coefficient_dim = 0;
  int free_dim = 0;
  int penalty_count = 0;
  int constraint_rank = 0;
  double rank_tolerance = 0.0;
  Matrix X;
  Matrix Z_pivoted;
  Matrix qr_packed;
  Vector qr_tau;
  Matrix design_r;
  std::vector<int> qr_pivot;
  std::vector<Matrix> penalty_roots;
  std::vector<Matrix> penalties;
  Matrix fixed_penalty;
  bool has_fixed_penalty = false;
  Vector initial_log_sp;
};

struct PreparedMultiPenaltyResponse {
  Vector y;
  Vector y0;
  double squared_norm = 0.0;
};

struct MultiPenaltyCoreEvaluation {
  Vector log_sp;
  int augmented_penalty_rank = 0;
  int numerical_rank = 0;
  double rss = 0.0;
  double edf = 0.0;
  double score = 0.0;
  double condition = 0.0;
  Matrix U1;
  Matrix V;
  Vector d;
  Vector y1;
};

struct ThinLapackSvd {
  Matrix U;
  Matrix V;
  Vector singular;
};

struct SymmetricLapackEigen {
  Matrix vectors;
  Vector values;
};

La_INT lapack_workspace_size(double query, const char* message) {
  require(std::isfinite(query) && query >= 1.0, message);
  double rounded = std::floor(query);
  if (query - rounded > 0.5) rounded += 1.0;
  require(rounded <= static_cast<double>(std::numeric_limits<La_INT>::max()),
          message);
  return static_cast<La_INT>(rounded);
}

SymmetricLapackEigen symmetric_lapack_eigen(const Matrix& input) {
  require(input.rows() == input.cols() && input.rows() > 0 &&
            input.allFinite(),
          "symmetric LAPACK eigen input is invalid");
  const La_INT dimension = static_cast<La_INT>(input.rows());
  const La_INT leading_dimension = dimension;
  const char job = 'V';
  const char range = 'A';
  const char uplo = 'L';
  const double lower_value = 0.0;
  const double upper_value = 0.0;
  const La_INT lower_index = 0;
  const La_INT upper_index = 0;
  const double absolute_tolerance = 0.0;
  La_INT eigenvalue_count = 0;
  Vector values(dimension);
  Matrix vectors(dimension, dimension);
  std::vector<La_INT> support(static_cast<std::size_t>(2 * dimension));
  double workspace_query = 0.0;
  La_INT integer_workspace_query = 0;
  La_INT workspace_size = -1;
  La_INT integer_workspace_size = -1;
  La_INT info = 0;
  Matrix factor = input;
  F77_CALL(dsyevr)(
    &job, &range, &uplo, &dimension, factor.data(), &leading_dimension,
    &lower_value, &upper_value, &lower_index, &upper_index,
    &absolute_tolerance, &eigenvalue_count, values.data(), vectors.data(),
    &leading_dimension, support.data(), &workspace_query, &workspace_size,
    &integer_workspace_query, &integer_workspace_size, &info
    FCONE FCONE FCONE);
  require(info == 0 && integer_workspace_query > 0,
          "symmetric LAPACK eigen workspace query failed");
  workspace_size = lapack_workspace_size(
    workspace_query, "symmetric LAPACK eigen workspace query failed");
  integer_workspace_size = integer_workspace_query;
  std::vector<double> workspace(
    static_cast<std::size_t>(workspace_size));
  std::vector<La_INT> integer_workspace(
    static_cast<std::size_t>(integer_workspace_size));
  factor = input;
  F77_CALL(dsyevr)(
    &job, &range, &uplo, &dimension, factor.data(), &leading_dimension,
    &lower_value, &upper_value, &lower_index, &upper_index,
    &absolute_tolerance, &eigenvalue_count, values.data(), vectors.data(),
    &leading_dimension, support.data(), workspace.data(), &workspace_size,
    integer_workspace.data(), &integer_workspace_size, &info
    FCONE FCONE FCONE);
  require(info == 0 && eigenvalue_count == dimension &&
            values.allFinite() && vectors.allFinite(),
          "symmetric LAPACK eigen decomposition failed");
  SymmetricLapackEigen result;
  result.vectors = std::move(vectors);
  result.values = std::move(values);
  return result;
}

struct PivotedLapackQr {
  Matrix packed;
  Vector tau;
  Matrix R;
  std::vector<int> pivot;
};

PivotedLapackQr pivoted_lapack_qr(const Matrix& input) {
  require(input.rows() >= input.cols() && input.cols() > 0,
          "pivoted LAPACK QR input dimensions are invalid");
  const La_INT rows = static_cast<La_INT>(input.rows());
  const La_INT columns = static_cast<La_INT>(input.cols());
  Matrix packed = input;
  Vector tau(columns);
  std::vector<La_INT> pivot(static_cast<std::size_t>(columns), 0);
  double workspace_query = 0.0;
  La_INT workspace_size = -1;
  La_INT info = 0;
  F77_CALL(dgeqp3)(
    &rows, &columns, packed.data(), &rows, pivot.data(), tau.data(),
    &workspace_query, &workspace_size, &info);
  require(info == 0, "pivoted LAPACK QR workspace query failed");
  workspace_size = lapack_workspace_size(
    workspace_query, "pivoted LAPACK QR workspace query failed");
  std::vector<double> workspace(
    static_cast<std::size_t>(workspace_size));
  packed = input;
  std::fill(pivot.begin(), pivot.end(), 0);
  F77_CALL(dgeqp3)(
    &rows, &columns, packed.data(), &rows, pivot.data(), tau.data(),
    workspace.data(), &workspace_size, &info);
  require(info == 0 && packed.allFinite() && tau.allFinite(),
          "pivoted LAPACK QR failed");

  PivotedLapackQr result;
  result.packed = std::move(packed);
  result.tau = std::move(tau);
  result.R = Matrix::Zero(columns, columns);
  result.pivot.resize(static_cast<std::size_t>(columns));
  std::vector<unsigned char> seen(static_cast<std::size_t>(columns), 0);
  for (La_INT column = 0; column < columns; ++column) {
    const La_INT original = pivot[static_cast<std::size_t>(column)] - 1;
    require(original >= 0 && original < columns &&
              seen[static_cast<std::size_t>(original)] == 0,
            "pivoted LAPACK QR returned an invalid permutation");
    seen[static_cast<std::size_t>(original)] = 1;
    result.pivot[static_cast<std::size_t>(column)] =
      static_cast<int>(original);
    for (La_INT row = 0; row <= column; ++row) {
      result.R(row, column) = result.packed(row, column);
    }
  }
  return result;
}

Vector apply_q_transpose(
    const Matrix& qr_packed,
    const Vector& qr_tau,
    const Vector& value) {
  require(qr_packed.rows() == value.size() &&
            qr_packed.cols() == qr_tau.size(),
          "pivoted LAPACK QR response dimensions are invalid");
  const La_INT rows = static_cast<La_INT>(qr_packed.rows());
  const La_INT columns = 1;
  const La_INT reflectors = static_cast<La_INT>(qr_packed.cols());
  const La_INT leading_qr = rows;
  const La_INT leading_value = rows;
  const char side = 'L';
  const char transpose = 'T';
  Vector transformed = value;
  double workspace_query = 0.0;
  La_INT workspace_size = -1;
  La_INT info = 0;
  F77_CALL(dormqr)(
    &side, &transpose, &rows, &columns, &reflectors,
    const_cast<double*>(qr_packed.data()), &leading_qr,
    const_cast<double*>(qr_tau.data()), transformed.data(), &leading_value,
    &workspace_query, &workspace_size, &info FCONE FCONE);
  require(info == 0, "pivoted LAPACK QR response workspace query failed");
  workspace_size = lapack_workspace_size(
    workspace_query, "pivoted LAPACK QR response workspace query failed");
  std::vector<double> workspace(
    static_cast<std::size_t>(workspace_size));
  transformed = value;
  F77_CALL(dormqr)(
    &side, &transpose, &rows, &columns, &reflectors,
    const_cast<double*>(qr_packed.data()), &leading_qr,
    const_cast<double*>(qr_tau.data()), transformed.data(), &leading_value,
    workspace.data(), &workspace_size, &info FCONE FCONE);
  require(info == 0 && transformed.allFinite(),
          "pivoted LAPACK QR response transform failed");
  return transformed.head(reflectors);
}

ThinLapackSvd thin_lapack_svd(const Matrix& input) {
  require(input.rows() >= input.cols() && input.cols() > 0,
          "thin LAPACK SVD input dimensions are invalid");
  const La_INT rows = static_cast<La_INT>(input.rows());
  const La_INT columns = static_cast<La_INT>(input.cols());
  const La_INT leading_input = rows;
  const La_INT leading_u = rows;
  const La_INT leading_vt = columns;
  Matrix factor = input;
  Matrix Vt(columns, columns);
  Vector singular(columns);
  std::vector<La_INT> integer_work(
    static_cast<std::size_t>(8 * columns));
  double workspace_query = 0.0;
  La_INT workspace_size = -1;
  La_INT info = 0;
  double unused_u = 0.0;
  const char job = 'O';
  F77_CALL(dgesdd)(
    &job, &rows, &columns, factor.data(), &leading_input,
    singular.data(), &unused_u, &leading_u, Vt.data(), &leading_vt,
    &workspace_query, &workspace_size, integer_work.data(), &info FCONE);
  require(info == 0, "thin LAPACK SVD workspace query failed");
  workspace_size = lapack_workspace_size(
    workspace_query, "thin LAPACK SVD workspace query failed");
  std::vector<double> workspace(
    static_cast<std::size_t>(workspace_size));
  factor = input;
  F77_CALL(dgesdd)(
    &job, &rows, &columns, factor.data(), &leading_input,
    singular.data(), &unused_u, &leading_u, Vt.data(), &leading_vt,
    workspace.data(), &workspace_size, integer_work.data(), &info FCONE);
  require(info == 0 && singular.allFinite() && factor.allFinite() &&
            Vt.allFinite(),
          "thin LAPACK SVD failed");
  ThinLapackSvd result;
  result.U = std::move(factor);
  result.V = Vt.transpose();
  result.singular = std::move(singular);
  return result;
}

PreparedMultiPenaltyProblem prepare_multi_penalty_problem(
    const double* X_data,
    int n,
    int coefficient_dim,
    const std::vector<std::vector<double>>& penalty_blocks,
    const std::vector<int>& penalty_dimensions,
    const std::vector<int>& penalty_offsets,
    const std::vector<int>& penalty_ranks,
    const double* H_data,
    bool has_H,
    const double* constraint,
    int constraint_rows,
    double rank_tolerance) {
  require(X_data != nullptr, "model matrix must be present");
  require(n > coefficient_dim && coefficient_dim > 0,
          "multi-penalty evaluation requires n > p > 0");
  require(std::isfinite(rank_tolerance) && rank_tolerance > 0.0 &&
            rank_tolerance < 1.0,
          "rank tolerance must be finite and in (0, 1)");
  const std::size_t penalty_count = penalty_blocks.size();
  require(penalty_count > 1,
          "multi-penalty evaluation requires at least two penalties");
  require(penalty_dimensions.size() == penalty_count &&
            penalty_offsets.size() == penalty_count &&
            penalty_ranks.size() == penalty_count,
          "multi-penalty metadata lengths must match");
  require(constraint_rows >= 0 &&
            (constraint_rows == 0 || constraint != nullptr),
          "constraint matrix pointer is missing");

  Eigen::Map<const Matrix> X(X_data, n, coefficient_dim);
  require(X.allFinite(), "model matrix must be finite");

  PreparedMultiPenaltyProblem problem;
  problem.n = n;
  problem.coefficient_dim = coefficient_dim;
  problem.penalty_count = static_cast<int>(penalty_count);
  problem.rank_tolerance = rank_tolerance;
  problem.X = X;
  const Matrix Z = constraint_null_space(
    constraint, constraint_rows, coefficient_dim, rank_tolerance,
    &problem.constraint_rank);
  problem.free_dim = static_cast<int>(Z.cols());
  const Matrix X_free = problem.constraint_rank == 0 ?
    problem.X : blas_multiply(problem.X, false, Z, false);
  const PivotedLapackQr qr = pivoted_lapack_qr(X_free);
  problem.qr_packed = qr.packed;
  problem.qr_tau = qr.tau;
  problem.design_r = qr.R;
  problem.qr_pivot = qr.pivot;
  problem.Z_pivoted.resize(coefficient_dim, problem.free_dim);
  for (int index = 0; index < problem.free_dim; ++index) {
    problem.Z_pivoted.col(index) =
      Z.col(problem.qr_pivot[static_cast<std::size_t>(index)]);
  }
  problem.penalty_roots.reserve(penalty_count);
  problem.penalties.reserve(penalty_count);
  for (std::size_t index = 0; index < penalty_count; ++index) {
    const int dimension = penalty_dimensions[index];
    const int offset = penalty_offsets[index];
    const int expected_rank = penalty_ranks[index];
    require(dimension > 0 && offset >= 0 &&
              offset + dimension <= coefficient_dim &&
              expected_rank > 0 && expected_rank <= dimension &&
              penalty_blocks[index].size() ==
                static_cast<std::size_t>(dimension) * dimension,
            "penalty block metadata is invalid");
    Eigen::Map<const Matrix> block(
      penalty_blocks[index].data(), dimension, dimension);
    const Matrix raw_block = block;
    Matrix full = Matrix::Zero(coefficient_dim, coefficient_dim);
    full.block(offset, offset, dimension, dimension) =
      symmetric_matrix(block, "penalty block");
    const Matrix free_penalty = symmetric_matrix(
      Z.transpose() * full * Z, "projected penalty");
    validate_penalty_rank(
      free_penalty, std::min(expected_rank, problem.free_dim));
    int root_rank = 0;
    const Matrix block_root = pivoted_cholesky_root(
      raw_block, &root_rank, expected_rank, 0.0);
    require(root_rank == expected_rank,
            "penalty root rank differs from authenticated rank");
    Matrix full_root = Matrix::Zero(coefficient_dim, root_rank);
    full_root.block(offset, 0, dimension, root_rank) =
      block_root.transpose();
    const Matrix free_root = problem.constraint_rank == 0 ?
      full_root : blas_multiply(Z, true, full_root, false);
    Matrix pivoted_root(problem.free_dim, root_rank);
    for (int row = 0; row < problem.free_dim; ++row) {
      pivoted_root.row(row) = free_root.row(
        problem.qr_pivot[static_cast<std::size_t>(row)]);
    }
    problem.penalties.push_back(blas_symmetric_xxt(pivoted_root));
    problem.penalty_roots.push_back(std::move(pivoted_root));
  }
  if (has_H) {
    require(H_data != nullptr, "fixed penalty pointer is missing");
    Eigen::Map<const Matrix> H(H_data, coefficient_dim, coefficient_dim);
    const Matrix free_fixed_penalty = symmetric_matrix(
      Z.transpose() * symmetric_matrix(H, "fixed penalty") * Z,
      "projected fixed penalty");
    problem.fixed_penalty.resize(problem.free_dim, problem.free_dim);
    for (int column = 0; column < problem.free_dim; ++column) {
      const int original_column =
        problem.qr_pivot[static_cast<std::size_t>(column)];
      for (int row = 0; row < problem.free_dim; ++row) {
        const int original_row =
          problem.qr_pivot[static_cast<std::size_t>(row)];
        problem.fixed_penalty(row, column) =
          free_fixed_penalty(original_row, original_column);
      }
    }
    problem.has_fixed_penalty = true;
  }
  problem.initial_log_sp = initial_log_sp(
    X_data, n, coefficient_dim, penalty_blocks, penalty_dimensions,
    penalty_offsets);
  return problem;
}

PreparedMultiPenaltyResponse prepare_multi_penalty_response(
    const PreparedMultiPenaltyProblem& problem,
    const double* y_data,
    int n) {
  require(y_data != nullptr, "response must be present");
  Eigen::Map<const Vector> y(y_data, n);
  require(y.allFinite(), "response must be finite");
  PreparedMultiPenaltyResponse response;
  response.y = y;
  response.y0 = apply_q_transpose(
    problem.qr_packed, problem.qr_tau, response.y);
  response.squared_norm = 0.0;
  for (int index = 0; index < n; ++index) {
    response.squared_norm += response.y[index] * response.y[index];
  }
  return response;
}

Matrix multi_penalty_augmented_system(
    const PreparedMultiPenaltyProblem& problem,
    const Vector& log_sp,
    int* augmented_penalty_rank) {
  require(augmented_penalty_rank != nullptr,
          "augmented penalty rank pointer is missing");
  require(log_sp.size() == problem.penalty_count && log_sp.allFinite(),
          "smoothing parameter vector is invalid");
  Matrix aggregate_penalty = Matrix::Zero(
    problem.free_dim, problem.free_dim);
  for (int index = 0; index < problem.penalty_count; ++index) {
    const double multiplier = std::exp(log_sp[index]);
    require(std::isfinite(multiplier) && multiplier > 0.0,
            "smoothing parameter must be positive and finite");
    const Matrix& penalty =
      problem.penalties[static_cast<std::size_t>(index)];
    for (Eigen::Index element = 0;
         element < aggregate_penalty.size(); ++element) {
      aggregate_penalty.data()[element] +=
        penalty.data()[element] * multiplier;
    }
  }
  if (problem.has_fixed_penalty) {
    for (Eigen::Index element = 0;
         element < aggregate_penalty.size(); ++element) {
      aggregate_penalty.data()[element] +=
        problem.fixed_penalty.data()[element];
    }
  }
  require(aggregate_penalty.allFinite(),
          "aggregate penalty is non-finite");
  const Matrix aggregate_root = pivoted_cholesky_root(
    aggregate_penalty, augmented_penalty_rank);
  Matrix augmented(
    problem.free_dim + *augmented_penalty_rank, problem.free_dim);
  augmented.topRows(problem.free_dim) = problem.design_r;
  augmented.bottomRows(*augmented_penalty_rank) = aggregate_root;
  require(augmented.allFinite(),
          "augmented multi-penalty system is malformed");
  return augmented;
}

MultiPenaltyCoreEvaluation evaluate_multi_penalty_core(
    const PreparedMultiPenaltyProblem& problem,
    const PreparedMultiPenaltyResponse& response,
    const Vector& log_sp) {
  int augmented_penalty_rank = 0;
  const Matrix augmented = multi_penalty_augmented_system(
    problem, log_sp, &augmented_penalty_rank);
  const ThinLapackSvd decomposition = thin_lapack_svd(augmented);
  const Vector& singular = decomposition.singular;
  require(singular.size() == problem.free_dim && singular[0] > 0.0,
          "augmented multi-penalty singular values are invalid");
  const double singular_threshold = singular[0] * problem.rank_tolerance;
  int numerical_rank = 0;
  for (Eigen::Index index = 0; index < singular.size(); ++index) {
    if (singular[index] >= singular_threshold) ++numerical_rank;
  }
  require(numerical_rank > 0,
          "augmented multi-penalty numerical rank is zero");

  MultiPenaltyCoreEvaluation core;
  core.log_sp = log_sp;
  core.augmented_penalty_rank = augmented_penalty_rank;
  core.numerical_rank = numerical_rank;
  core.U1 = decomposition.U.topRows(problem.free_dim).leftCols(
    numerical_rank);
  core.V = decomposition.V.leftCols(numerical_rank);
  core.d = singular.head(numerical_rank);
  core.y1.resize(numerical_rank);
  double yAy = 0.0;
  for (int column = 0; column < numerical_rank; ++column) {
    double value = 0.0;
    for (int row = 0; row < problem.free_dim; ++row) {
      value += core.U1(row, column) * response.y0[row];
    }
    core.y1[column] = value;
    yAy += value * value;
  }
  double yAAy = 0.0;
  for (int row = 0; row < problem.free_dim; ++row) {
    double value = 0.0;
    for (int column = 0; column < numerical_rank; ++column) {
      value += core.U1(row, column) * core.y1[column];
    }
    yAAy += value * value;
  }
  core.rss = response.squared_norm - 2.0 * yAy + yAAy;
  if (core.rss < 0.0) core.rss = 0.0;
  core.edf = 0.0;
  for (Eigen::Index index = 0; index < core.U1.size(); ++index) {
    core.edf += core.U1.data()[index] * core.U1.data()[index];
  }
  const double delta = static_cast<double>(problem.n) - core.edf;
  require(std::isfinite(core.rss) && core.rss >= 0.0 &&
            std::isfinite(core.edf) && delta > 1e-8,
          "multi-penalty GCV objective is invalid");
  core.score = static_cast<double>(problem.n) * core.rss /
    (delta * delta);
  core.condition = core.d[0] / core.d[numerical_rank - 1];
  require(std::isfinite(core.score),
          "multi-penalty evaluation produced non-finite output");
  return core;
}

MultiPenaltyGcvEvaluation materialize_multi_penalty_evaluation(
    const PreparedMultiPenaltyProblem& problem,
    const PreparedMultiPenaltyResponse& response,
    const MultiPenaltyCoreEvaluation& core,
    bool include_derivatives_and_fit) {
  MultiPenaltyGcvEvaluation result;
  result.n = problem.n;
  result.coefficient_dim = problem.coefficient_dim;
  result.free_dim = problem.free_dim;
  result.penalty_count = problem.penalty_count;
  result.constraint_rank = problem.constraint_rank;
  result.augmented_penalty_rank = core.augmented_penalty_rank;
  result.numerical_rank = core.numerical_rank;
  result.rss = core.rss;
  result.edf = core.edf;
  result.score = core.score;
  result.condition = core.condition;
  result.condition_bucket = condition_bucket(
    core.numerical_rank, problem.free_dim, core.condition);
  result.log_sp = copy_vector(core.log_sp);
  if (!include_derivatives_and_fit) return result;

  const Vector scaled_response =
    (core.y1.array() / core.d.array()).matrix();
  Vector theta(problem.free_dim);
  for (int row = 0; row < problem.free_dim; ++row) {
    double value = 0.0;
    for (int column = 0; column < core.numerical_rank; ++column) {
      value += core.V(row, column) * scaled_response[column];
    }
    theta[row] = value;
  }
  const Vector coefficients = problem.Z_pivoted * theta;
  const Vector fitted = problem.X * coefficients;
  const Vector residuals = response.y - fitted;

  const Matrix U1U1 = blas_symmetric_xtx(core.U1);
  std::vector<Matrix> M;
  std::vector<Matrix> K;
  std::vector<Vector> My;
  std::vector<Vector> Ky;
  std::vector<Vector> yK;
  M.reserve(static_cast<std::size_t>(problem.penalty_count));
  K.reserve(static_cast<std::size_t>(problem.penalty_count));
  My.reserve(static_cast<std::size_t>(problem.penalty_count));
  Ky.reserve(static_cast<std::size_t>(problem.penalty_count));
  yK.reserve(static_cast<std::size_t>(problem.penalty_count));
  for (const Matrix& root : problem.penalty_roots) {
    Matrix transformed = blas_multiply(core.V, true, root, false);
    for (Eigen::Index column = 0; column < transformed.cols(); ++column) {
      for (int row = 0; row < core.numerical_rank; ++row) {
        transformed(row, column) /= core.d[row];
      }
    }
    const Matrix intermediate = blas_multiply(
      transformed, true, U1U1, false);
    K.push_back(blas_multiply(
      transformed, false, intermediate, false));
    M.push_back(blas_symmetric_xxt(transformed));

    Vector metric_y(core.numerical_rank);
    Vector influence_y(core.numerical_rank);
    Vector y_influence(core.numerical_rank);
    for (int output = 0; output < core.numerical_rank; ++output) {
      double metric_value = 0.0;
      double y_influence_value = 0.0;
      double influence_value = 0.0;
      for (int component = 0;
           component < core.numerical_rank; ++component) {
        metric_value += core.y1[component] * M.back()(component, output);
        y_influence_value +=
          core.y1[component] * K.back()(component, output);
        influence_value +=
          core.y1[component] * K.back()(output, component);
      }
      metric_y[output] = metric_value;
      y_influence[output] = y_influence_value;
      influence_y[output] = influence_value;
    }
    My.push_back(std::move(metric_y));
    Ky.push_back(std::move(influence_y));
    yK.push_back(std::move(y_influence));
  }
  Vector dnorm(problem.penalty_count);
  Vector ddelta(problem.penalty_count);
  Matrix d2norm = Matrix::Zero(
    problem.penalty_count, problem.penalty_count);
  Matrix d2delta = Matrix::Zero(
    problem.penalty_count, problem.penalty_count);
  for (int i = 0; i < problem.penalty_count; ++i) {
    const double lambda_i = std::exp(core.log_sp[i]);
    double trace = 0.0;
    for (int component = 0;
         component < core.numerical_rank; ++component) {
      trace += K[static_cast<std::size_t>(i)](component, component);
    }
    ddelta[i] = lambda_i * trace;
    for (int j = 0; j <= i; ++j) {
      double product_sum = 0.0;
      const Matrix& metric_j = M[static_cast<std::size_t>(j)];
      const Matrix& influence_i = K[static_cast<std::size_t>(i)];
      for (Eigen::Index element = 0; element < metric_j.size(); ++element) {
        product_sum += metric_j.data()[element] * influence_i.data()[element];
      }
      const double delta_value = -2.0 *
        std::exp(core.log_sp[i] + core.log_sp[j]) * product_sum;
      d2delta(i, j) = delta_value;
      d2delta(j, i) = delta_value;
    }
    d2delta(i, i) += ddelta[i];

    double norm_derivative = 0.0;
    for (int component = 0;
         component < core.numerical_rank; ++component) {
      norm_derivative += core.y1[component] *
        (My[static_cast<std::size_t>(i)][component] -
         Ky[static_cast<std::size_t>(i)][component]);
    }
    dnorm[i] = 2.0 * lambda_i * norm_derivative;
    for (int j = 0; j <= i; ++j) {
      double norm_value = 0.0;
      for (int component = 0; component < core.numerical_rank; ++component) {
        norm_value +=
          My[static_cast<std::size_t>(i)][component] *
            Ky[static_cast<std::size_t>(j)][component] +
          My[static_cast<std::size_t>(j)][component] *
            Ky[static_cast<std::size_t>(i)][component] -
          2.0 * My[static_cast<std::size_t>(i)][component] *
            My[static_cast<std::size_t>(j)][component] +
          yK[static_cast<std::size_t>(i)][component] *
            My[static_cast<std::size_t>(j)][component];
      }
      norm_value *= 2.0 * std::exp(core.log_sp[i] + core.log_sp[j]);
      d2norm(i, j) = norm_value;
      d2norm(j, i) = norm_value;
    }
    d2norm(i, i) += dnorm[i];
  }
  const double delta = static_cast<double>(problem.n) - core.edf;
  const double score_scale = static_cast<double>(problem.n) /
    (delta * delta);
  const double delta_scale = 2.0 * score_scale * core.rss / delta;
  const double cross_scale = -2.0 * score_scale / delta;
  const double curvature_scale = 3.0 * delta_scale / delta;
  const Vector gradient = score_scale * dnorm - delta_scale * ddelta;
  Matrix hessian(problem.penalty_count, problem.penalty_count);
  for (int i = 0; i < problem.penalty_count; ++i) {
    for (int j = 0; j <= i; ++j) {
      const double value =
        cross_scale * (ddelta[j] * dnorm[i] + ddelta[i] * dnorm[j]) +
        score_scale * d2norm(i, j) +
        curvature_scale * ddelta[i] * ddelta[j] -
        delta_scale * d2delta(i, j);
      hessian(i, j) = value;
      hessian(j, i) = value;
    }
  }
  require(gradient.allFinite() && hessian.allFinite() &&
            coefficients.allFinite() && residuals.allFinite(),
          "multi-penalty evaluation produced non-finite output");
  result.gradient = copy_vector(gradient);
  result.hessian.assign(hessian.data(), hessian.data() + hessian.size());
  result.coefficients = copy_vector(coefficients);
  result.fitted = copy_vector(fitted);
  result.residuals = copy_vector(residuals);
  return result;
}

void refine_multi_penalty_selected_fit(
    const PreparedMultiPenaltyProblem& problem,
    const PreparedMultiPenaltyResponse& response,
    const Vector& log_sp,
    MultiPenaltyGcvEvaluation* evaluation) {
  require(evaluation != nullptr,
          "selected fit refinement output is missing");
  int augmented_penalty_rank = 0;
  const Matrix augmented = multi_penalty_augmented_system(
    problem, log_sp, &augmented_penalty_rank);
  const ThinLapackSvd decomposition = thin_lapack_svd(augmented);
  require(decomposition.singular.size() == problem.free_dim &&
            decomposition.singular[0] > 0.0,
          "selected fit refinement singular values are invalid");
  const double threshold = decomposition.singular[0] *
    problem.rank_tolerance;
  int numerical_rank = 0;
  for (Eigen::Index index = 0;
       index < decomposition.singular.size(); ++index) {
    if (decomposition.singular[index] >= threshold) ++numerical_rank;
  }
  require(augmented_penalty_rank == evaluation->augmented_penalty_rank &&
            numerical_rank == evaluation->numerical_rank &&
            numerical_rank > 0,
          "selected fit refinement rank drifted");
  const Matrix U1 = decomposition.U.topRows(problem.free_dim).leftCols(
    numerical_rank);
  const Vector d = decomposition.singular.head(numerical_rank);
  Vector y1(numerical_rank);
  for (int column = 0; column < numerical_rank; ++column) {
    double value = 0.0;
    for (int row = 0; row < problem.free_dim; ++row) {
      value += U1(row, column) * response.y0[row];
    }
    y1[column] = value;
  }
  const Vector scaled_response = (y1.array() / d.array()).matrix();
  Vector theta(problem.free_dim);
  for (int row = 0; row < problem.free_dim; ++row) {
    double value = 0.0;
    for (int column = 0; column < numerical_rank; ++column) {
      value += decomposition.V(row, column) * scaled_response[column];
    }
    theta[row] = value;
  }
  const Vector coefficients = problem.Z_pivoted * theta;
  const Vector fitted = problem.X * coefficients;
  const Vector residuals = response.y - fitted;
  require(coefficients.allFinite() && fitted.allFinite() &&
            residuals.allFinite(),
          "selected fit refinement produced non-finite output");
  evaluation->coefficients = copy_vector(coefficients);
  evaluation->fitted = copy_vector(fitted);
  evaluation->residuals = copy_vector(residuals);
}

void append_transcript(
    MultiPenaltyGcvOptimization* result,
    const MultiPenaltyOptimizerControl& control,
    const std::string& stage,
    int iteration,
    int evaluation,
    int coordinate,
    const Vector& current_log_sp,
    const Vector& proposed_step,
    const Vector& trial_log_sp,
    const MultiPenaltyGcvEvaluation& value,
    const Vector& hessian_eigenvalues,
    bool accepted,
    const std::string& step_source) {
  if (!control.keep_transcript) return;
  MultiPenaltyOptimizerTranscriptEntry entry;
  entry.stage = stage;
  entry.iteration = iteration;
  entry.evaluation = evaluation;
  entry.coordinate = coordinate;
  entry.current_log_sp = copy_vector(current_log_sp);
  entry.proposed_step = copy_vector(proposed_step);
  entry.trial_log_sp = copy_vector(trial_log_sp);
  entry.score = value.score;
  entry.gradient = value.gradient;
  entry.hessian = value.hessian;
  entry.hessian_eigenvalues = copy_vector(hessian_eigenvalues);
  entry.accepted = accepted;
  entry.step_source = step_source;
  entry.numerical_rank = value.numerical_rank;
  entry.condition = value.condition;
  result->transcript.push_back(std::move(entry));
}

}  // namespace

MultiPenaltyGcvEvaluation multi_penalty_gcv_evaluate_cpp(
    const double* X_data,
    const double* y_data,
    int n,
    int coefficient_dim,
    const std::vector<std::vector<double>>& penalty_blocks,
    const std::vector<int>& penalty_dimensions,
    const std::vector<int>& penalty_offsets,
    const std::vector<int>& penalty_ranks,
    const std::vector<double>& log_sp,
    const double* H_data,
    bool has_H,
    const double* constraint,
    int constraint_rows,
    double rank_tolerance) {
  require(log_sp.size() == penalty_blocks.size(),
          "multi-penalty smoothing parameter length is invalid");
  const PreparedMultiPenaltyProblem problem = prepare_multi_penalty_problem(
    X_data, n, coefficient_dim, penalty_blocks, penalty_dimensions,
    penalty_offsets, penalty_ranks, H_data, has_H, constraint,
    constraint_rows, rank_tolerance);
  const PreparedMultiPenaltyResponse response =
    prepare_multi_penalty_response(problem, y_data, n);
  const Eigen::Map<const Vector> log_sp_vector(
    log_sp.data(), static_cast<Eigen::Index>(log_sp.size()));
  const MultiPenaltyCoreEvaluation core = evaluate_multi_penalty_core(
    problem, response, log_sp_vector);
  return materialize_multi_penalty_evaluation(
    problem, response, core, true);
}

MultiPenaltyGcvOptimization multi_penalty_gcv_optimize_cpp(
    const double* X,
    const double* y,
    int n,
    int coefficient_dim,
    const std::vector<std::vector<double>>& penalty_blocks,
    const std::vector<int>& penalty_dimensions,
    const std::vector<int>& penalty_offsets,
    const std::vector<int>& penalty_ranks,
    const double* H,
    bool has_H,
    const double* constraint,
    int constraint_rows,
    const MultiPenaltyOptimizerControl& control) {
  require(std::isfinite(control.convergence_tolerance) &&
            control.convergence_tolerance > 0.0 &&
            control.convergence_tolerance < 1.0,
          "optimizer convergence tolerance must be in (0, 1)");
  require(control.max_step_halving >= 4 &&
            control.max_iterations > 3 &&
            std::isfinite(control.max_newton_step) &&
            control.max_newton_step > 0.0 &&
            std::isfinite(control.boundary_probe_step) &&
            control.boundary_probe_step > 0.0 &&
            control.max_boundary_probes > 0,
          "multi-penalty optimizer control is invalid");
  const PreparedMultiPenaltyProblem problem = prepare_multi_penalty_problem(
    X, n, coefficient_dim, penalty_blocks, penalty_dimensions,
    penalty_offsets, penalty_ranks, H, has_H, constraint, constraint_rows,
    control.rank_tolerance);
  const PreparedMultiPenaltyResponse response =
    prepare_multi_penalty_response(problem, y, n);
  const int penalty_count = problem.penalty_count;
  Vector log_sp = problem.initial_log_sp;
  auto evaluate_core = [&](const Vector& value) {
    return evaluate_multi_penalty_core(problem, response, value);
  };
  auto evaluate_full = [&](const Vector& value) {
    const MultiPenaltyCoreEvaluation core = evaluate_core(value);
    return materialize_multi_penalty_evaluation(
      problem, response, core, true);
  };

  MultiPenaltyGcvOptimization result;
  result.initial_log_sp = copy_vector(log_sp);
  MultiPenaltyGcvEvaluation current = evaluate_full(log_sp);
  result.score_calls = 1;
  result.objective_calls = 1;
  double minimum_score = current.score;
  double score_reduction = 1e10;
  Vector gradient = Vector::Zero(penalty_count);
  Vector newton_step = Vector::Zero(penalty_count);
  Vector steepest_step = Vector::Zero(penalty_count);
  Vector eigenvalues = Vector::Zero(penalty_count);
  bool use_steepest_descent = false;
  bool converged = false;

  for (int iteration = 1; iteration <= control.max_iterations; ++iteration) {
    result.optimizer_iterations = iteration;
    int tries = 0;
    if (iteration > 1) {
      Vector step = use_steepest_descent ? steepest_step : newton_step;
      std::string step_source = use_steepest_descent ?
        "steepest_descent" : "newton";
      bool trying = true;
      while (trying) {
        ++tries;
        if (tries == 4 && !use_steepest_descent) {
          use_steepest_descent = true;
          step = steepest_step;
          step_source = "steepest_descent_after_newton_rejection";
        }
        if (step_source == "newton") {
          ++result.newton_trial_count;
        } else {
          ++result.steepest_descent_trial_count;
        }
        const Vector trial_log_sp = log_sp + step;
        const MultiPenaltyCoreEvaluation trial_core =
          evaluate_core(trial_log_sp);
        ++result.score_calls;
        ++result.objective_calls;
        const bool accepted = trial_core.score < minimum_score;
        MultiPenaltyGcvEvaluation trial =
          materialize_multi_penalty_evaluation(
            problem, response, trial_core,
            accepted || control.keep_transcript);
        append_transcript(
          &result, control, "step_trial", iteration, tries, -1, log_sp,
          step, trial_log_sp, trial, eigenvalues, accepted, step_source);
        if (accepted) {
          trying = false;
          score_reduction = minimum_score - trial.score;
          minimum_score = trial.score;
          log_sp = trial_log_sp;
          current = std::move(trial);
        } else {
          step /= 2.0;
          ++result.step_halving_count;
        }
        if (tries == control.max_step_halving - 1 && trying) {
          step.setZero();
        }
        if (tries == control.max_step_halving) trying = false;
      }
    }

    if (iteration > 3) {
      double squared_gradient_norm = 0.0;
      for (int index = 0; index < penalty_count; ++index) {
        squared_gradient_norm += gradient[index] * gradient[index];
      }
      const double gradient_norm = std::sqrt(squared_gradient_norm);
      converged =
        score_reduction <= control.convergence_tolerance *
          (1.0 + minimum_score) &&
        gradient_norm <= std::pow(control.convergence_tolerance, 1.0 / 3.0) *
          (1.0 + std::abs(minimum_score));
      if (tries == control.max_step_halving) converged = true;
      if (converged) {
        result.rms_gradient = gradient_norm /
          std::sqrt(static_cast<double>(penalty_count));
        if (tries == control.max_step_halving) {
          result.step_failed = true;
        }
      }
    }

    gradient = Eigen::Map<const Vector>(
      current.gradient.data(), penalty_count);
    Eigen::Map<const Matrix> current_hessian(
      current.hessian.data(), penalty_count, penalty_count);
    const SymmetricLapackEigen decomposition =
      symmetric_lapack_eigen(current_hessian);
    eigenvalues = decomposition.values;
    use_steepest_descent = (eigenvalues.array() < 0.0).any();
    if (!use_steepest_descent) {
      Vector projected(penalty_count);
      for (int column = 0; column < penalty_count; ++column) {
        double value = 0.0;
        for (int row = 0; row < penalty_count; ++row) {
          value += decomposition.vectors(row, column) * gradient[row];
        }
        projected[column] = value / eigenvalues[column];
      }
      for (int row = 0; row < penalty_count; ++row) {
        double value = 0.0;
        for (int column = 0; column < penalty_count; ++column) {
          value += decomposition.vectors(row, column) * projected[column];
        }
        newton_step[row] = -value;
      }
      double maximum_component = std::abs(newton_step[0]);
      for (int index = 1; index < penalty_count; ++index) {
        maximum_component = std::max(
          maximum_component, std::abs(newton_step[index]));
      }
      if (maximum_component > control.max_newton_step) {
        newton_step *= control.max_newton_step / maximum_component;
      }
    }
    double maximum_gradient = std::abs(gradient[0]);
    for (int index = 1; index < penalty_count; ++index) {
      maximum_gradient = std::max(
        maximum_gradient, std::abs(gradient[index]));
    }
    if (maximum_gradient == 0.0) {
      steepest_step.setZero();
    } else {
      steepest_step = -gradient / maximum_gradient;
    }
    const Vector& proposed_step = use_steepest_descent ?
      steepest_step : newton_step;
    append_transcript(
      &result, control, "iteration_state", iteration, 0, -1, log_sp,
      proposed_step, log_sp, current, eigenvalues, true,
      use_steepest_descent ? "steepest_descent" : "newton");
    if (converged) break;
  }
  require(converged,
          "multi-penalty optimizer exceeded its iteration limit");
  result.fully_converged = !result.step_failed;
  result.hessian_positive_definite = !use_steepest_descent;
  result.convergence_code = result.step_failed ?
    "step_halving_exhausted" : "fully_converged";

  result.boundary_status.reserve(static_cast<std::size_t>(penalty_count));
  for (int coordinate = 0; coordinate < penalty_count; ++coordinate) {
    const double direction = gradient[coordinate] < 0.0 ? 1.0 : -1.0;
    int accepted_count = 0;
    for (int probe = 1; probe <= control.max_boundary_probes; ++probe) {
      Vector step = Vector::Zero(penalty_count);
      step[coordinate] = direction * control.boundary_probe_step;
      const Vector trial_log_sp = log_sp + step;
      const MultiPenaltyCoreEvaluation trial_core =
        evaluate_core(trial_log_sp);
      ++result.objective_calls;
      ++result.boundary_probe_count;
      const bool accepted = trial_core.score < minimum_score;
      MultiPenaltyGcvEvaluation trial =
        materialize_multi_penalty_evaluation(
          problem, response, trial_core,
          accepted || control.keep_transcript);
      append_transcript(
        &result, control, "boundary_probe", result.optimizer_iterations,
        probe, coordinate, log_sp, step, trial_log_sp, trial, eigenvalues,
        accepted, "mgcv_infinity_probe");
      if (!accepted) break;
      log_sp = trial_log_sp;
      current = std::move(trial);
      minimum_score = current.score;
      ++accepted_count;
      ++result.boundary_accepted_count;
    }
    if (accepted_count == 0) {
      result.boundary_status.push_back("finite-interior");
    } else if (accepted_count == control.max_boundary_probes) {
      result.boundary_status.push_back(
        direction > 0.0 ? "positive-boundary" : "negative-boundary");
    } else {
      result.boundary_status.push_back("finite-after-boundary-probe");
    }
  }
  refine_multi_penalty_selected_fit(
    problem, response, log_sp, &current);
  result.selected = std::move(current);
  ++result.objective_calls;
  return result;
}

}  // namespace fastkpc
