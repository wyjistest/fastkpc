#include "mgcv_multi_penalty_cpp.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
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

Matrix pivoted_cholesky_root(const Matrix& penalty, int* root_rank) {
  require(root_rank != nullptr, "penalty root rank pointer is missing");
  Matrix factor = penalty;
  const La_INT dimension = static_cast<La_INT>(penalty.rows());
  const La_INT leading_dimension = dimension;
  std::vector<La_INT> pivot(static_cast<std::size_t>(dimension));
  std::vector<double> work(static_cast<std::size_t>(2 * dimension));
  La_INT rank = 0;
  La_INT info = 0;
  double tolerance = -1.0;
  const char uplo = 'U';
  F77_CALL(dpstrf)(
    &uplo, &dimension, factor.data(), &leading_dimension, pivot.data(),
    &rank, &tolerance, work.data(), &info FCONE);
  require(info >= 0 && rank > 0 && rank <= dimension,
          "aggregate penalty pivoted Cholesky failed");
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

std::vector<double> copy_vector(const Vector& value) {
  return std::vector<double>(value.data(), value.data() + value.size());
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
  Vector design_diagonal = X.colwise().squaredNorm().transpose();
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
    double design_sum = 0.0;
    double penalty_sum = 0.0;
    int selected = 0;
    for (int row = 0; row < dimension; ++row) {
      const double row_mean = absolute.row(row).mean();
      const double column_mean = absolute.col(row).mean();
      const double diagonal = absolute(row, row);
      if (row_mean > threshold && column_mean > threshold &&
          diagonal > threshold) {
        design_sum += design_diagonal[offset + row];
        penalty_sum += block(row, row);
        penalized[static_cast<std::size_t>(offset + row)] = 1;
        ++selected;
      }
    }
    require(selected > 0 && penalty_sum > 0.0,
            "multi-penalty initial smoothing scale is invalid");
    initial[static_cast<Eigen::Index>(index)] = design_sum / penalty_sum;
    for (int row = 0; row < dimension; ++row) {
      scaled_penalty_diagonal[offset + row] +=
        initial[static_cast<Eigen::Index>(index)] * block(row, row);
    }
  }
  auto mean_influence = [&]() {
    double total = 0.0;
    int count = 0;
    for (int column = 0; column < coefficient_dim; ++column) {
      if (penalized[static_cast<std::size_t>(column)] != 0 &&
          design_diagonal[column] > 0.0 &&
          scaled_penalty_diagonal[column] > 0.0) {
        total += design_diagonal[column] /
          (design_diagonal[column] + scaled_penalty_diagonal[column]);
        ++count;
      }
    }
    require(count > 0, "multi-penalty initial smoothing support is empty");
    return total / static_cast<double>(count);
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
  return initial.array().log().matrix();
}

struct PreparedMultiPenaltyProblem {
  int n = 0;
  int coefficient_dim = 0;
  int free_dim = 0;
  int penalty_count = 0;
  int constraint_rank = 0;
  double rank_tolerance = 0.0;
  Matrix X;
  Matrix Z;
  Matrix X_free;
  std::vector<Matrix> penalties;
  Matrix fixed_penalty;
  bool has_fixed_penalty = false;
  Vector initial_log_sp;
};

struct PreparedMultiPenaltyResponse {
  Vector y;
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

ThinLapackSvd thin_lapack_svd(const Matrix& input) {
  require(input.rows() >= input.cols() && input.cols() > 0,
          "thin LAPACK SVD input dimensions are invalid");
  const La_INT rows = static_cast<La_INT>(input.rows());
  const La_INT columns = static_cast<La_INT>(input.cols());
  const La_INT leading_input = rows;
  const La_INT leading_u = rows;
  const La_INT leading_vt = columns;
  Matrix factor = input;
  Matrix U(rows, columns);
  Matrix Vt(columns, columns);
  Vector singular(columns);
  std::vector<La_INT> integer_work(
    static_cast<std::size_t>(8 * columns));
  double workspace_query = 0.0;
  La_INT workspace_size = -1;
  La_INT info = 0;
  const char job = 'S';
  F77_CALL(dgesdd)(
    &job, &rows, &columns, factor.data(), &leading_input,
    singular.data(), U.data(), &leading_u, Vt.data(), &leading_vt,
    &workspace_query, &workspace_size, integer_work.data(), &info FCONE);
  require(info == 0 && std::isfinite(workspace_query) &&
            workspace_query >= 1.0,
          "thin LAPACK SVD workspace query failed");
  workspace_size = static_cast<La_INT>(std::ceil(workspace_query));
  std::vector<double> workspace(
    static_cast<std::size_t>(workspace_size));
  factor = input;
  F77_CALL(dgesdd)(
    &job, &rows, &columns, factor.data(), &leading_input,
    singular.data(), U.data(), &leading_u, Vt.data(), &leading_vt,
    workspace.data(), &workspace_size, integer_work.data(), &info FCONE);
  require(info == 0 && singular.allFinite() && U.allFinite() &&
            Vt.allFinite(),
          "thin LAPACK SVD failed");
  ThinLapackSvd result;
  result.U = std::move(U);
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
  problem.Z = constraint_null_space(
    constraint, constraint_rows, coefficient_dim, rank_tolerance,
    &problem.constraint_rank);
  problem.free_dim = static_cast<int>(problem.Z.cols());
  problem.X_free = problem.X * problem.Z;
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
    Matrix full = Matrix::Zero(coefficient_dim, coefficient_dim);
    full.block(offset, offset, dimension, dimension) =
      symmetric_matrix(block, "penalty block");
    Matrix free_penalty = symmetric_matrix(
      problem.Z.transpose() * full * problem.Z, "projected penalty");
    validate_penalty_rank(
      free_penalty, std::min(expected_rank, problem.free_dim));
    problem.penalties.push_back(std::move(free_penalty));
  }
  if (has_H) {
    require(H_data != nullptr, "fixed penalty pointer is missing");
    Eigen::Map<const Matrix> H(H_data, coefficient_dim, coefficient_dim);
    problem.fixed_penalty = symmetric_matrix(
      problem.Z.transpose() * symmetric_matrix(H, "fixed penalty") *
        problem.Z,
      "projected fixed penalty");
    problem.has_fixed_penalty = true;
  }
  problem.initial_log_sp = initial_log_sp(
    X_data, n, coefficient_dim, penalty_blocks, penalty_dimensions,
    penalty_offsets);
  return problem;
}

PreparedMultiPenaltyResponse prepare_multi_penalty_response(
    const double* y_data,
    int n) {
  require(y_data != nullptr, "response must be present");
  Eigen::Map<const Vector> y(y_data, n);
  require(y.allFinite(), "response must be finite");
  PreparedMultiPenaltyResponse response;
  response.y = y;
  response.squared_norm = response.y.squaredNorm();
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
    aggregate_penalty +=
      multiplier * problem.penalties[static_cast<std::size_t>(index)];
  }
  if (problem.has_fixed_penalty) {
    aggregate_penalty += problem.fixed_penalty;
  }
  aggregate_penalty = symmetric_matrix(
    aggregate_penalty, "aggregate penalty");
  const Matrix aggregate_root = pivoted_cholesky_root(
    aggregate_penalty, augmented_penalty_rank);
  Matrix augmented(
    problem.n + *augmented_penalty_rank, problem.free_dim);
  augmented.topRows(problem.n) = problem.X_free;
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
  Eigen::JacobiSVD<Matrix> decomposition(
    augmented, Eigen::ComputeThinU | Eigen::ComputeThinV);
  require(decomposition.info() == Eigen::Success,
          "augmented multi-penalty SVD failed");
  const Vector singular = decomposition.singularValues();
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
  core.U1 = decomposition.matrixU().topRows(problem.n).leftCols(
    numerical_rank);
  core.V = decomposition.matrixV().leftCols(numerical_rank);
  core.d = singular.head(numerical_rank);
  core.y1 = core.U1.transpose() * response.y;
  const Vector Ay = core.U1 * core.y1;
  core.rss = response.squared_norm - 2.0 * core.y1.squaredNorm() +
    Ay.squaredNorm();
  if (core.rss < 0.0 && core.rss >= -64.0 *
      std::numeric_limits<double>::epsilon() *
      std::max(1.0, response.squared_norm)) {
    core.rss = 0.0;
  }
  core.edf = core.U1.squaredNorm();
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

  const Vector theta = core.V *
    (core.y1.array() / core.d.array()).matrix();
  const Vector coefficients = problem.Z * theta;
  const Vector fitted = problem.X * coefficients;
  const Vector residuals = response.y - fitted;

  const Matrix U1U1 = core.U1.transpose() * core.U1;
  const Matrix inverse_d = core.d.cwiseInverse().asDiagonal();
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
  for (const Matrix& penalty : problem.penalties) {
    Matrix metric = inverse_d * core.V.transpose() * penalty * core.V *
      inverse_d;
    Matrix influence = metric * U1U1;
    M.push_back(std::move(metric));
    K.push_back(std::move(influence));
    My.push_back(M.back() * core.y1);
    Ky.push_back(K.back() * core.y1);
    yK.push_back(K.back().transpose() * core.y1);
  }
  Vector dnorm(problem.penalty_count);
  Vector ddelta(problem.penalty_count);
  Matrix d2norm = Matrix::Zero(
    problem.penalty_count, problem.penalty_count);
  Matrix d2delta = Matrix::Zero(
    problem.penalty_count, problem.penalty_count);
  for (int i = 0; i < problem.penalty_count; ++i) {
    const double lambda_i = std::exp(core.log_sp[i]);
    ddelta[i] = lambda_i * K[static_cast<std::size_t>(i)].trace();
    dnorm[i] = 2.0 * lambda_i * core.y1.dot(
      My[static_cast<std::size_t>(i)] - Ky[static_cast<std::size_t>(i)]);
    for (int j = 0; j <= i; ++j) {
      const double lambda_j = std::exp(core.log_sp[j]);
      double delta_value = -2.0 * lambda_i * lambda_j *
        (M[static_cast<std::size_t>(j)].array() *
         K[static_cast<std::size_t>(i)].array()).sum();
      if (i == j) delta_value += ddelta[i];
      d2delta(i, j) = delta_value;
      d2delta(j, i) = delta_value;
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
      norm_value *= 2.0 * lambda_i * lambda_j;
      if (i == j) norm_value += dnorm[i];
      d2norm(i, j) = norm_value;
      d2norm(j, i) = norm_value;
    }
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
    for (int j = 0; j < problem.penalty_count; ++j) {
      hessian(i, j) =
        cross_scale * (ddelta[j] * dnorm[i] + ddelta[i] * dnorm[j]) +
        score_scale * d2norm(i, j) +
        curvature_scale * ddelta[i] * ddelta[j] -
        delta_scale * d2delta(i, j);
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
  const Matrix U1 = decomposition.U.topRows(problem.n).leftCols(
    numerical_rank);
  const Vector d = decomposition.singular.head(numerical_rank);
  const Vector y1 = U1.transpose() * response.y;
  const Vector theta = decomposition.V.leftCols(numerical_rank) *
    (y1.array() / d.array()).matrix();
  const Vector coefficients = problem.Z * theta;
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
  require(X_data != nullptr && y_data != nullptr,
          "model matrix and response must be present");
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
            penalty_ranks.size() == penalty_count &&
            log_sp.size() == penalty_count,
          "multi-penalty metadata lengths must match");
  Eigen::Map<const Matrix> X(X_data, n, coefficient_dim);
  Eigen::Map<const Vector> y(y_data, n);
  require(X.allFinite() && y.allFinite(),
          "model matrix and response must be finite");

  int constraint_rank = 0;
  const Matrix Z = constraint_null_space(
    constraint, constraint_rows, coefficient_dim, rank_tolerance,
    &constraint_rank);
  const int free_dim = static_cast<int>(Z.cols());
  const Matrix X_free = X * Z;
  std::vector<Matrix> penalties;
  penalties.reserve(penalty_count);
  Matrix aggregate_penalty = Matrix::Zero(free_dim, free_dim);
  for (std::size_t index = 0; index < penalty_count; ++index) {
    const int dimension = penalty_dimensions[index];
    const int offset = penalty_offsets[index];
    const int expected_rank = penalty_ranks[index];
    require(dimension > 0 && offset >= 0 &&
              offset + dimension <= coefficient_dim &&
              expected_rank > 0 && expected_rank <= dimension &&
              penalty_blocks[index].size() ==
                static_cast<std::size_t>(dimension) * dimension &&
              std::isfinite(log_sp[index]),
            "penalty block metadata is invalid");
    Eigen::Map<const Matrix> block(
      penalty_blocks[index].data(), dimension, dimension);
    Matrix full = Matrix::Zero(coefficient_dim, coefficient_dim);
    full.block(offset, offset, dimension, dimension) =
      symmetric_matrix(block, "penalty block");
    Matrix free_penalty = symmetric_matrix(
      Z.transpose() * full * Z, "projected penalty");
    validate_penalty_rank(
      free_penalty, std::min(expected_rank, free_dim));
    penalties.push_back(free_penalty);
    const double multiplier = std::exp(log_sp[index]);
    require(std::isfinite(multiplier) && multiplier > 0.0,
            "smoothing parameter must be positive and finite");
    aggregate_penalty += multiplier * free_penalty;
  }
  if (has_H) {
    require(H_data != nullptr, "fixed penalty pointer is missing");
    Eigen::Map<const Matrix> H(H_data, coefficient_dim, coefficient_dim);
    aggregate_penalty += symmetric_matrix(
      Z.transpose() * symmetric_matrix(H, "fixed penalty") * Z,
      "projected fixed penalty");
  }
  aggregate_penalty = symmetric_matrix(
    aggregate_penalty, "aggregate penalty");
  int augmented_penalty_rank = 0;
  const Matrix aggregate_root = pivoted_cholesky_root(
    aggregate_penalty, &augmented_penalty_rank);

  Matrix augmented(n + augmented_penalty_rank, free_dim);
  augmented.topRows(n) = X_free;
  augmented.bottomRows(augmented_penalty_rank) = aggregate_root;
  require(augmented.allFinite(),
          "augmented multi-penalty system is malformed");
  Eigen::JacobiSVD<Matrix> decomposition(
    augmented, Eigen::ComputeThinU | Eigen::ComputeThinV);
  require(decomposition.info() == Eigen::Success,
          "augmented multi-penalty SVD failed");
  const Vector singular = decomposition.singularValues();
  require(singular.size() == free_dim && singular[0] > 0.0,
          "augmented multi-penalty singular values are invalid");
  const double singular_threshold = singular[0] * rank_tolerance;
  int numerical_rank = 0;
  for (Eigen::Index index = 0; index < singular.size(); ++index) {
    if (singular[index] >= singular_threshold) ++numerical_rank;
  }
  require(numerical_rank > 0,
          "augmented multi-penalty numerical rank is zero");
  const Matrix U1 = decomposition.matrixU().topRows(n).leftCols(numerical_rank);
  const Matrix V = decomposition.matrixV().leftCols(numerical_rank);
  const Vector d = singular.head(numerical_rank);
  const Vector y1 = U1.transpose() * y;
  const Vector Ay = U1 * y1;
  double rss = y.squaredNorm() - 2.0 * y1.squaredNorm() + Ay.squaredNorm();
  if (rss < 0.0 && rss >= -64.0 * std::numeric_limits<double>::epsilon() *
      std::max(1.0, y.squaredNorm())) {
    rss = 0.0;
  }
  const double edf = U1.squaredNorm();
  const double delta = static_cast<double>(n) - edf;
  require(std::isfinite(rss) && rss >= 0.0 && std::isfinite(edf) &&
            delta > 1e-8,
          "multi-penalty GCV objective is invalid");
  const double score = static_cast<double>(n) * rss / (delta * delta);
  const Vector theta = V * (y1.array() / d.array()).matrix();
  const Vector coefficients = Z * theta;
  const Vector fitted = X * coefficients;
  const Vector residuals = y - fitted;

  const Matrix U1U1 = U1.transpose() * U1;
  const Matrix inverse_d = d.cwiseInverse().asDiagonal();
  std::vector<Matrix> M;
  std::vector<Matrix> K;
  std::vector<Vector> My;
  std::vector<Vector> Ky;
  std::vector<Vector> yK;
  M.reserve(penalty_count);
  K.reserve(penalty_count);
  My.reserve(penalty_count);
  Ky.reserve(penalty_count);
  yK.reserve(penalty_count);
  for (const Matrix& penalty : penalties) {
    Matrix metric = inverse_d * V.transpose() * penalty * V * inverse_d;
    Matrix influence = metric * U1U1;
    M.push_back(std::move(metric));
    K.push_back(std::move(influence));
    My.push_back(M.back() * y1);
    Ky.push_back(K.back() * y1);
    yK.push_back(K.back().transpose() * y1);
  }
  Vector dnorm(penalty_count);
  Vector ddelta(penalty_count);
  Matrix d2norm = Matrix::Zero(penalty_count, penalty_count);
  Matrix d2delta = Matrix::Zero(penalty_count, penalty_count);
  for (std::size_t i = 0; i < penalty_count; ++i) {
    const double lambda_i = std::exp(log_sp[i]);
    ddelta[static_cast<Eigen::Index>(i)] =
      lambda_i * K[i].trace();
    dnorm[static_cast<Eigen::Index>(i)] =
      2.0 * lambda_i * y1.dot(My[i] - Ky[i]);
    for (std::size_t j = 0; j <= i; ++j) {
      const double lambda_j = std::exp(log_sp[j]);
      double delta_value = -2.0 * lambda_i * lambda_j *
        (M[j].array() * K[i].array()).sum();
      if (i == j) delta_value += ddelta[static_cast<Eigen::Index>(i)];
      d2delta(static_cast<Eigen::Index>(i),
              static_cast<Eigen::Index>(j)) = delta_value;
      d2delta(static_cast<Eigen::Index>(j),
              static_cast<Eigen::Index>(i)) = delta_value;
      double norm_value = 0.0;
      for (int component = 0; component < numerical_rank; ++component) {
        norm_value +=
          My[i][component] * Ky[j][component] +
          My[j][component] * Ky[i][component] -
          2.0 * My[i][component] * My[j][component] +
          yK[i][component] * My[j][component];
      }
      norm_value *= 2.0 * lambda_i * lambda_j;
      if (i == j) norm_value += dnorm[static_cast<Eigen::Index>(i)];
      d2norm(static_cast<Eigen::Index>(i),
             static_cast<Eigen::Index>(j)) = norm_value;
      d2norm(static_cast<Eigen::Index>(j),
             static_cast<Eigen::Index>(i)) = norm_value;
    }
  }
  const double score_scale = static_cast<double>(n) / (delta * delta);
  const double delta_scale = 2.0 * score_scale * rss / delta;
  const double cross_scale = -2.0 * score_scale / delta;
  const double curvature_scale = 3.0 * delta_scale / delta;
  const Vector gradient = score_scale * dnorm - delta_scale * ddelta;
  Matrix hessian(penalty_count, penalty_count);
  for (std::size_t i = 0; i < penalty_count; ++i) {
    for (std::size_t j = 0; j < penalty_count; ++j) {
      hessian(static_cast<Eigen::Index>(i),
              static_cast<Eigen::Index>(j)) =
        cross_scale *
          (ddelta[static_cast<Eigen::Index>(j)] *
             dnorm[static_cast<Eigen::Index>(i)] +
           ddelta[static_cast<Eigen::Index>(i)] *
             dnorm[static_cast<Eigen::Index>(j)]) +
        score_scale * d2norm(static_cast<Eigen::Index>(i),
                             static_cast<Eigen::Index>(j)) +
        curvature_scale *
          ddelta[static_cast<Eigen::Index>(i)] *
          ddelta[static_cast<Eigen::Index>(j)] -
        delta_scale * d2delta(static_cast<Eigen::Index>(i),
                              static_cast<Eigen::Index>(j));
    }
  }
  require(std::isfinite(score) && gradient.allFinite() &&
            hessian.allFinite() && coefficients.allFinite() &&
            residuals.allFinite(),
          "multi-penalty evaluation produced non-finite output");

  MultiPenaltyGcvEvaluation result;
  result.n = n;
  result.coefficient_dim = coefficient_dim;
  result.free_dim = free_dim;
  result.penalty_count = static_cast<int>(penalty_count);
  result.constraint_rank = constraint_rank;
  result.augmented_penalty_rank = augmented_penalty_rank;
  result.numerical_rank = numerical_rank;
  result.rss = rss;
  result.edf = edf;
  result.score = score;
  result.condition = d[0] / d[numerical_rank - 1];
  result.condition_bucket = condition_bucket(
    numerical_rank, free_dim, result.condition);
  result.log_sp = log_sp;
  result.gradient = copy_vector(gradient);
  result.hessian.assign(
    hessian.data(), hessian.data() + hessian.size());
  result.coefficients = copy_vector(coefficients);
  result.fitted = copy_vector(fitted);
  result.residuals = copy_vector(residuals);
  return result;
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
    prepare_multi_penalty_response(y, n);
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
      const double gradient_norm = gradient.norm();
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
    Eigen::SelfAdjointEigenSolver<Matrix> decomposition(
      0.5 * (current_hessian + current_hessian.transpose()));
    require(decomposition.info() == Eigen::Success &&
              decomposition.eigenvalues().allFinite() &&
              decomposition.eigenvectors().allFinite(),
            "multi-penalty optimizer Hessian eigendecomposition failed");
    eigenvalues = decomposition.eigenvalues();
    use_steepest_descent = (eigenvalues.array() < 0.0).any();
    if (!use_steepest_descent) {
      const Vector projected = decomposition.eigenvectors().transpose() *
        gradient;
      if ((eigenvalues.array() == 0.0).any()) {
        use_steepest_descent = true;
      } else {
        newton_step = -decomposition.eigenvectors() *
          (projected.array() / eigenvalues.array()).matrix();
        const double maximum_component = newton_step.cwiseAbs().maxCoeff();
        if (maximum_component > control.max_newton_step) {
          newton_step *= control.max_newton_step / maximum_component;
        }
      }
    }
    const double maximum_gradient = gradient.cwiseAbs().maxCoeff();
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
