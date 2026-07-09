#include <RcppArmadillo.h>

#include "legacy_dcov_gamma_cpp.hpp"

#include <Eigen/Core>
#include <Spectra/MatOp/DenseSymMatProd.h>
#include <Spectra/SymEigsSolver.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace fastkpc {
namespace {

double legacy_dcov_elapsed_ms_since(
    std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - start).count();
}

int legacy_dcov_batch_threads_from_env(int batch) {
  if (batch <= 1) return 1;
  const char* raw = std::getenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH_THREADS");
  if (raw == nullptr) return 1;
  char* end = nullptr;
  const long parsed = std::strtol(raw, &end, 10);
  if (end == raw || parsed <= 1) return 1;
  return std::max(1, std::min(batch, static_cast<int>(parsed)));
}

bool legacy_dcov_spectra_matvec_diag_enabled() {
  const char* raw =
    std::getenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_SPECTRA_MATVEC_DIAG");
  if (raw == nullptr) return false;
  const std::string value(raw);
  return !value.empty() && value != "0" && value != "false" &&
         value != "FALSE" && value != "off" && value != "OFF" &&
         value != "no" && value != "NO";
}

void legacy_dcov_fill_distance_matrix(const double* values,
                                      int n,
                                      arma::mat& out) {
  out.set_size(n, n);
  out.diag().zeros();
  for (int col = 0; col < n; ++col) {
    const double vc = values[col];
    for (int row = col + 1; row < n; ++row) {
      const double dist = std::abs(values[row] - vc);
      out(row, col) = dist;
      out(col, row) = dist;
    }
  }
}

arma::mat legacy_dcov_distance_matrix(const double* values, int n) {
  arma::mat out;
  legacy_dcov_fill_distance_matrix(values, n, out);
  return out;
}

arma::uvec legacy_dcov_top_abs_eigen_indices(const arma::vec& values,
                                             int num_col) {
  std::vector<int> order(values.n_elem);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&values](int lhs, int rhs) {
                     const double la = std::abs(values(lhs));
                     const double ra = std::abs(values(rhs));
                     if (la == ra) return lhs > rhs;
                     return la > ra;
                   });
  arma::uvec idx(num_col);
  for (int i = 0; i < num_col; ++i) idx(i) = order[static_cast<std::size_t>(i)];
  return idx;
}

struct LegacyDcovLowrank {
  arma::vec values;
  arma::mat centered_vectors;
};

struct LegacyDcovLowrankEigWorkspace {
  arma::vec full_eigenvalues;
  arma::mat full_eigenvectors;
  Eigen::VectorXd spectra_eigenvalues;
  Eigen::MatrixXd spectra_eigenvectors;
  int reuse_count = 0;
};

class LegacyDcovSpectraCountingMatProd {
 public:
  using MatrixMap = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic,
                                                   Eigen::Dynamic,
                                                   Eigen::ColMajor>>;

  LegacyDcovSpectraCountingMatProd(
      const MatrixMap& matrix,
      LegacyDcovLowrankTimings* timings)
      : matrix_(matrix), timings_(timings) {}

  int rows() const { return static_cast<int>(matrix_.rows()); }
  int cols() const { return static_cast<int>(matrix_.cols()); }

  void perform_op(const double* x_in, double* y_out) const {
    const auto matvec_start = std::chrono::steady_clock::now();
    Eigen::Map<const Eigen::VectorXd> x(x_in, matrix_.cols());
    Eigen::Map<Eigen::VectorXd> y(y_out, matrix_.rows());
    y.noalias() = matrix_ * x;
    if (timings_ != nullptr) {
      timings_->spectra_matvec_count += 1;
      timings_->spectra_matvec_ms +=
        legacy_dcov_elapsed_ms_since(matvec_start);
    }
  }

 private:
  const MatrixMap& matrix_;
  LegacyDcovLowrankTimings* timings_;
};

void legacy_dcov_lowrank_full_eig(
    const arma::mat& distance,
    int num_col,
    LegacyDcovLowrank& output,
    LegacyDcovLowrankEigWorkspace* eig_workspace = nullptr,
    LegacyDcovLowrankTimings* timings = nullptr) {
  const auto eig_start = std::chrono::steady_clock::now();
  arma::vec local_eigenvalues;
  arma::mat local_eigenvectors;
  arma::vec& eigenvalues = eig_workspace == nullptr ?
    local_eigenvalues : eig_workspace->full_eigenvalues;
  arma::mat& eigenvectors = eig_workspace == nullptr ?
    local_eigenvectors : eig_workspace->full_eigenvectors;
  if (!arma::eig_sym(eigenvalues, eigenvectors, distance)) {
    Rcpp::stop("legacy dCov gamma eigen decomposition failed");
  }
  if (eig_workspace != nullptr) eig_workspace->reuse_count += 1;
  const double eig_ms = legacy_dcov_elapsed_ms_since(eig_start);

  const auto select_start = std::chrono::steady_clock::now();
  const arma::uvec idx = legacy_dcov_top_abs_eigen_indices(eigenvalues, num_col);
  output.centered_vectors.set_size(distance.n_rows, num_col);
  output.values.set_size(num_col);
  for (int col = 0; col < num_col; ++col) {
    const arma::uword selected = idx(col);
    output.values(col) = eigenvalues(selected);
    output.centered_vectors.col(col) = eigenvectors.col(selected);
  }
  const double select_ms = legacy_dcov_elapsed_ms_since(select_start);

  const auto center_start = std::chrono::steady_clock::now();
  output.centered_vectors.each_row() -= arma::mean(output.centered_vectors, 0);
  const double center_ms = legacy_dcov_elapsed_ms_since(center_start);

  if (timings != nullptr) {
    timings->eig_ms += eig_ms;
    timings->select_ms += select_ms;
    timings->center_ms += center_ms;
    timings->full_eig_count += 1;
  }
}

void legacy_dcov_lowrank_spectra(
    const arma::mat& distance,
    int num_col,
    LegacyDcovLowrank& output,
    LegacyDcovLowrankEigWorkspace* eig_workspace = nullptr,
    LegacyDcovLowrankTimings* timings = nullptr) {
  const int n = static_cast<int>(distance.n_rows);
  const int ncv = std::min(n, std::max(2 * num_col + 1, 20));
  constexpr double kTol = 1e-10;
  constexpr int kMaxIterations = 1000;

  if (timings != nullptr) {
    timings->spectra_count += 1;
    timings->spectra_ncv = std::max(timings->spectra_ncv, ncv);
    timings->spectra_tol = std::max(timings->spectra_tol, kTol);
  }

  const auto eig_start = std::chrono::steady_clock::now();
  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::ColMajor>> mapped(
    distance.memptr(), distance.n_rows, distance.n_cols);
  Eigen::VectorXd local_eigenvalues;
  Eigen::MatrixXd local_eigenvectors;
  Eigen::VectorXd& eigenvalues = eig_workspace == nullptr ?
    local_eigenvalues : eig_workspace->spectra_eigenvalues;
  Eigen::MatrixXd& eigenvectors = eig_workspace == nullptr ?
    local_eigenvectors : eig_workspace->spectra_eigenvectors;
  std::chrono::steady_clock::time_point select_start;

  if (legacy_dcov_spectra_matvec_diag_enabled()) {
    LegacyDcovSpectraCountingMatProd op(mapped, timings);
    Spectra::SymEigsSolver<double, Spectra::LARGEST_MAGN,
                           LegacyDcovSpectraCountingMatProd> eigs(
      &op, num_col, ncv);
    eigs.init();
    const int nconv = static_cast<int>(eigs.compute(
      kMaxIterations, kTol, Spectra::LARGEST_MAGN));
    const int iterations = static_cast<int>(eigs.num_iterations());
    const bool ok = eigs.info() == Spectra::SUCCESSFUL && nconv >= num_col;
    const double spectra_eig_ms = legacy_dcov_elapsed_ms_since(eig_start);
    if (timings != nullptr) {
      timings->eig_ms += spectra_eig_ms;
      timings->spectra_iterations += iterations;
      timings->spectra_nconv += nconv;
      if (ok) {
        timings->spectra_converged_count += 1;
      } else {
        timings->spectra_failed_count += 1;
        timings->spectra_fallback_full_eig_count += 1;
      }
    }
    if (!ok) {
      legacy_dcov_lowrank_full_eig(
        distance, num_col, output, eig_workspace, timings);
      return;
    }
    select_start = std::chrono::steady_clock::now();
    eigenvalues = eigs.eigenvalues();
    eigenvectors = eigs.eigenvectors();
  } else {
    Spectra::DenseSymMatProd<double> op(mapped);
    Spectra::SymEigsSolver<double, Spectra::LARGEST_MAGN,
                           Spectra::DenseSymMatProd<double>> eigs(
      &op, num_col, ncv);
    eigs.init();
    const int nconv = static_cast<int>(eigs.compute(
      kMaxIterations, kTol, Spectra::LARGEST_MAGN));
    const int iterations = static_cast<int>(eigs.num_iterations());
    const bool ok = eigs.info() == Spectra::SUCCESSFUL && nconv >= num_col;
    const double spectra_eig_ms = legacy_dcov_elapsed_ms_since(eig_start);
    if (timings != nullptr) {
      timings->eig_ms += spectra_eig_ms;
      timings->spectra_iterations += iterations;
      timings->spectra_nconv += nconv;
      if (ok) {
        timings->spectra_converged_count += 1;
      } else {
        timings->spectra_failed_count += 1;
        timings->spectra_fallback_full_eig_count += 1;
      }
    }
    if (!ok) {
      legacy_dcov_lowrank_full_eig(
        distance, num_col, output, eig_workspace, timings);
      return;
    }
    select_start = std::chrono::steady_clock::now();
    eigenvalues = eigs.eigenvalues();
    eigenvectors = eigs.eigenvectors();
  }

  if (eig_workspace != nullptr) eig_workspace->reuse_count += 1;
  output.centered_vectors.set_size(distance.n_rows, num_col);
  output.values.set_size(num_col);
  for (int col = 0; col < num_col; ++col) {
    output.values(col) = eigenvalues(col);
    for (int row = 0; row < n; ++row) {
      output.centered_vectors(row, col) = eigenvectors(row, col);
    }
  }
  const double select_ms = legacy_dcov_elapsed_ms_since(select_start);

  const auto center_start = std::chrono::steady_clock::now();
  output.centered_vectors.each_row() -= arma::mean(output.centered_vectors, 0);
  const double center_ms = legacy_dcov_elapsed_ms_since(center_start);

  if (timings != nullptr) {
    timings->select_ms += select_ms;
    timings->center_ms += center_ms;
  }
}

void legacy_dcov_lowrank(const arma::mat& distance,
                         int num_col,
                         LegacyDcovLowrankMode mode,
                         LegacyDcovLowrank& output,
                         LegacyDcovLowrankEigWorkspace* eig_workspace = nullptr,
                         LegacyDcovLowrankTimings* timings = nullptr) {
  if (mode == LegacyDcovLowrankMode::Spectra) {
    legacy_dcov_lowrank_spectra(
      distance, num_col, output, eig_workspace, timings);
    return;
  }
  legacy_dcov_lowrank_full_eig(
    distance, num_col, output, eig_workspace, timings);
}

double legacy_dcov_weighted_cross_sum(const arma::mat& left,
                                      const arma::vec& left_values,
                                      const arma::mat& right,
                                      const arma::vec& right_values,
                                      arma::mat* cross_workspace = nullptr,
                                      int* reuse_count = nullptr) {
  arma::mat local_cross;
  arma::mat& cross =
    cross_workspace == nullptr ? local_cross : *cross_workspace;
  cross = left.t() * right;
  if (reuse_count != nullptr) *reuse_count += 1;
  double total = 0.0;
  for (arma::uword col = 0; col < cross.n_cols; ++col) {
    for (arma::uword row = 0; row < cross.n_rows; ++row) {
      total += left_values(row) * right_values(col) *
        cross(row, col) * cross(row, col);
    }
  }
  return total;
}

}  // namespace

LegacyDcovLowrankMode legacy_dcov_lowrank_mode_from_env() {
  const char* raw = std::getenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK");
  if (raw == nullptr) return LegacyDcovLowrankMode::FullEig;
  const std::string mode(raw);
  if (mode == "spectra" || mode == "selected" || mode == "selected_eigs") {
    return LegacyDcovLowrankMode::Spectra;
  }
  return LegacyDcovLowrankMode::FullEig;
}

const char* legacy_dcov_lowrank_mode_name(LegacyDcovLowrankMode mode) {
  return mode == LegacyDcovLowrankMode::Spectra ? "spectra" : "full_eig";
}

struct LegacyDcovGammaCppWorkspace {
  arma::mat x_distance;
  arma::mat y_distance;
  arma::mat statistic_moment_cross;
  LegacyDcovLowrank x_lowrank;
  LegacyDcovLowrank y_lowrank;
  LegacyDcovLowrankEigWorkspace lowrank_eig_workspace;
  int n = 0;
  int distance_workspace_reuse_count = 0;
  int statistic_moment_workspace_reuse_count = 0;
  int lowrank_output_workspace_reuse_count = 0;
};

LegacyDcovGammaCppResult legacy_dcov_gamma_cpp_compute(
    Rcpp::NumericVector x,
    Rcpp::NumericVector y,
    int numCol,
    double index) {
  if (y.size() != x.size()) Rcpp::stop("Sample sizes must agree");
  return legacy_dcov_gamma_cpp_compute_workspace(
    x.begin(), y.begin(), x.size(), numCol, index, nullptr);
}

LegacyDcovGammaCppResult legacy_dcov_gamma_cpp_compute_workspace(
    const double* x,
    const double* y,
    int n,
    int numCol,
    double index,
    LegacyDcovGammaCppWorkspace* workspace) {
  const auto total_start = std::chrono::steady_clock::now();
  const auto input_start = std::chrono::steady_clock::now();
  if (x == nullptr || y == nullptr) Rcpp::stop("Sample sizes must agree");
  if (n <= 5) Rcpp::stop("legacy dCov gamma oracle requires n > 5");
  if (numCol <= 0 || numCol >= n) {
    Rcpp::stop("numCol must be positive and less than sample size");
  }
  if (index < 0.0 || index > 2.0) index = 1.0;
  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(x[i]) || !std::isfinite(y[i])) {
      Rcpp::stop("Data contains missing or infinite values");
    }
  }
  const double input_ms = legacy_dcov_elapsed_ms_since(input_start);

  const auto distance_start = std::chrono::steady_clock::now();
  arma::mat local_matx;
  arma::mat local_maty;
  arma::mat& matx = workspace == nullptr ? local_matx : workspace->x_distance;
  arma::mat& maty = workspace == nullptr ? local_maty : workspace->y_distance;
  legacy_dcov_fill_distance_matrix(x, n, matx);
  legacy_dcov_fill_distance_matrix(y, n, maty);
  if (workspace != nullptr) {
    workspace->n = n;
    workspace->distance_workspace_reuse_count += 2;
  }
  const double distance_ms = legacy_dcov_elapsed_ms_since(distance_start);

  const auto lowrank_start = std::chrono::steady_clock::now();
  const LegacyDcovLowrankMode lowrank_mode =
    legacy_dcov_lowrank_mode_from_env();
  LegacyDcovLowrankTimings lowrank_timings;
  LegacyDcovLowrank local_x_lowrank;
  LegacyDcovLowrank local_y_lowrank;
  LegacyDcovLowrank& x_lowrank =
    workspace == nullptr ? local_x_lowrank : workspace->x_lowrank;
  LegacyDcovLowrank& y_lowrank =
    workspace == nullptr ? local_y_lowrank : workspace->y_lowrank;
  LegacyDcovLowrankEigWorkspace* lowrank_eig_workspace = workspace == nullptr ?
    nullptr : &workspace->lowrank_eig_workspace;
  legacy_dcov_lowrank(
    matx, numCol, lowrank_mode, x_lowrank, lowrank_eig_workspace,
    &lowrank_timings);
  legacy_dcov_lowrank(
    maty, numCol, lowrank_mode, y_lowrank, lowrank_eig_workspace,
    &lowrank_timings);
  if (workspace != nullptr) workspace->lowrank_output_workspace_reuse_count += 2;
  const double lowrank_ms = legacy_dcov_elapsed_ms_since(lowrank_start);
  const double lowrank_accounted_ms = lowrank_timings.eig_ms +
    lowrank_timings.select_ms + lowrank_timings.center_ms;
  const double lowrank_unaccounted_ms =
    std::max(0.0, lowrank_ms - lowrank_accounted_ms);

  const auto statistic_start = std::chrono::steady_clock::now();
  arma::mat* cross_workspace =
    workspace == nullptr ? nullptr : &workspace->statistic_moment_cross;
  int* statistic_moment_workspace_reuse_count = workspace == nullptr ?
    nullptr : &workspace->statistic_moment_workspace_reuse_count;
  const double nV2 = legacy_dcov_weighted_cross_sum(
    x_lowrank.centered_vectors, x_lowrank.values,
    y_lowrank.centered_vectors, y_lowrank.values,
    cross_workspace, statistic_moment_workspace_reuse_count) /
    static_cast<double>(n);
  const double statistic_ms = legacy_dcov_elapsed_ms_since(statistic_start);

  const auto moment_start = std::chrono::steady_clock::now();
  const double n_double = static_cast<double>(n);
  const double nV2Mean = (arma::accu(matx) / (n_double * n_double)) *
    (arma::accu(maty) / (n_double * n_double));
  const double x_moment = legacy_dcov_weighted_cross_sum(
    x_lowrank.centered_vectors, x_lowrank.values,
    x_lowrank.centered_vectors, x_lowrank.values,
    cross_workspace, statistic_moment_workspace_reuse_count);
  const double y_moment = legacy_dcov_weighted_cross_sum(
    y_lowrank.centered_vectors, y_lowrank.values,
    y_lowrank.centered_vectors, y_lowrank.values,
    cross_workspace, statistic_moment_workspace_reuse_count);
  const double variance_factor =
    2.0 * (n_double - 4.0) * (n_double - 5.0) /
    n_double / (n_double - 1.0) / (n_double - 2.0) / (n_double - 3.0);
  const double nV2Variance =
    variance_factor * x_moment * y_moment /
    std::pow(n_double, 4.0) * std::pow(n_double, 2.0);
  const double alpha = (nV2Mean * nV2Mean) / nV2Variance;
  const double beta = nV2Variance / nV2Mean;
  const double moment_ms = legacy_dcov_elapsed_ms_since(moment_start);

  const auto pgamma_start = std::chrono::steady_clock::now();
  const double p_value = 1.0 - R::pgamma(nV2, alpha, beta, true, false);
  const double statistic = nV2;
  const double dCov = std::sqrt(nV2 / n_double);
  const double pgamma_ms = legacy_dcov_elapsed_ms_since(pgamma_start);
  const double total_ms = legacy_dcov_elapsed_ms_since(total_start);
  const double accounted_ms = input_ms + distance_ms + lowrank_ms +
    statistic_ms + moment_ms + pgamma_ms;

  LegacyDcovGammaCppResult result;
  result.p_value = p_value;
  result.nV2 = nV2;
  result.mean = nV2Mean;
  result.variance = nV2Variance;
  result.statistic = statistic;
  result.estimate = dCov;
  result.n = n;
  result.num_col = numCol;
  result.index = index;
  result.lowrank_mode = lowrank_mode;
  result.input_ms = input_ms;
  result.distance_ms = distance_ms;
  result.lowrank_ms = lowrank_ms;
  result.lowrank_unaccounted_ms = lowrank_unaccounted_ms;
  result.statistic_ms = statistic_ms;
  result.moment_ms = moment_ms;
  result.pgamma_ms = pgamma_ms;
  result.accounted_ms = accounted_ms;
  result.total_ms = total_ms;
  result.lowrank_timings = lowrank_timings;
  return result;
}

Rcpp::List legacy_dcov_gamma_cpp_result_to_list(
    const LegacyDcovGammaCppResult& result) {
  return Rcpp::List::create(
    Rcpp::Named("p.value") = result.p_value,
    Rcpp::Named("nV2") = result.nV2,
    Rcpp::Named("mean") = result.mean,
    Rcpp::Named("variance") = result.variance,
    Rcpp::Named("statistic") = result.statistic,
    Rcpp::Named("estimate") = result.estimate,
    Rcpp::Named("estimates") = Rcpp::NumericVector::create(
      Rcpp::Named("nV^2") = result.nV2,
      Rcpp::Named("nV^2 mean") = result.mean,
      Rcpp::Named("nV^2 variance") = result.variance),
    Rcpp::Named("diagnostics") = Rcpp::List::create(
      Rcpp::Named("n") = result.n,
      Rcpp::Named("numCol") = result.num_col,
      Rcpp::Named("index") = result.index,
      Rcpp::Named("lowrank_mode") =
        std::string(legacy_dcov_lowrank_mode_name(result.lowrank_mode)),
      Rcpp::Named("input_ms") = result.input_ms,
      Rcpp::Named("distance_ms") = result.distance_ms,
      Rcpp::Named("lowrank_ms") = result.lowrank_ms,
      Rcpp::Named("lowrank_eig_ms") = result.lowrank_timings.eig_ms,
      Rcpp::Named("lowrank_select_ms") = result.lowrank_timings.select_ms,
      Rcpp::Named("lowrank_center_ms") = result.lowrank_timings.center_ms,
      Rcpp::Named("lowrank_unaccounted_ms") =
        result.lowrank_unaccounted_ms,
      Rcpp::Named("lowrank_full_eig_count") =
        result.lowrank_timings.full_eig_count,
      Rcpp::Named("lowrank_spectra_count") =
        result.lowrank_timings.spectra_count,
      Rcpp::Named("lowrank_spectra_converged_count") =
        result.lowrank_timings.spectra_converged_count,
      Rcpp::Named("lowrank_spectra_failed_count") =
        result.lowrank_timings.spectra_failed_count,
      Rcpp::Named("lowrank_spectra_fallback_full_eig_count") =
        result.lowrank_timings.spectra_fallback_full_eig_count,
      Rcpp::Named("lowrank_spectra_iterations") =
        result.lowrank_timings.spectra_iterations,
      Rcpp::Named("lowrank_spectra_nconv") =
        result.lowrank_timings.spectra_nconv,
      Rcpp::Named("lowrank_spectra_ncv") =
        result.lowrank_timings.spectra_ncv,
      Rcpp::Named("lowrank_spectra_tol") =
        result.lowrank_timings.spectra_tol,
      Rcpp::Named("lowrank_spectra_matvec_count") =
        result.lowrank_timings.spectra_matvec_count,
      Rcpp::Named("lowrank_spectra_matvec_ms") =
        result.lowrank_timings.spectra_matvec_ms,
      Rcpp::Named("statistic_ms") = result.statistic_ms,
      Rcpp::Named("moment_ms") = result.moment_ms,
      Rcpp::Named("pgamma_ms") = result.pgamma_ms,
      Rcpp::Named("accounted_ms") = result.accounted_ms,
      Rcpp::Named("unaccounted_ms") =
        std::max(0.0, result.total_ms - result.accounted_ms),
      Rcpp::Named("total_ms") = result.total_ms
    )
  );
}

Rcpp::List legacy_dcov_gamma_cpp_compute_batch_ptrs(
    const std::vector<const double*>& x_columns,
    const std::vector<const double*>& y_columns,
    int n,
    int numCol,
    double index) {
  const auto total_start = std::chrono::steady_clock::now();
  if (n <= 0) {
    Rcpp::stop("n must be positive");
  }
  if (x_columns.size() != y_columns.size()) {
    Rcpp::stop("x and y column pointer batches must have matching lengths");
  }
  const int batch = static_cast<int>(x_columns.size());
  for (int col = 0; col < batch; ++col) {
    if (x_columns[static_cast<std::size_t>(col)] == nullptr ||
        y_columns[static_cast<std::size_t>(col)] == nullptr) {
      Rcpp::stop("x and y column pointer batches must not contain nulls");
    }
  }
  Rcpp::NumericVector p_values(batch);
  Rcpp::NumericVector nV2_values(batch);
  Rcpp::NumericVector mean_values(batch);
  Rcpp::NumericVector variance_values(batch);
  Rcpp::NumericVector estimate_values(batch);
  double scalar_total_ms = 0.0;
  double input_ms = 0.0;
  double distance_ms = 0.0;
  double lowrank_ms = 0.0;
  double lowrank_eig_ms = 0.0;
  double lowrank_select_ms = 0.0;
  double lowrank_center_ms = 0.0;
  double lowrank_unaccounted_ms = 0.0;
  double statistic_ms = 0.0;
  double moment_ms = 0.0;
  double pgamma_ms = 0.0;
  LegacyDcovLowrankTimings lowrank_timings;
  LegacyDcovLowrankMode lowrank_mode = legacy_dcov_lowrank_mode_from_env();
  const int batch_threads = legacy_dcov_batch_threads_from_env(batch);
  const bool batch_parallel_enabled = batch_threads > 1;
  std::vector<LegacyDcovGammaCppResult> results(batch);
  std::vector<LegacyDcovGammaCppWorkspace> workspaces(
    static_cast<std::size_t>(batch_threads));

  if (batch_parallel_enabled) {
    std::mutex error_mutex;
    std::string first_error;
    auto worker = [&](int thread_id) {
      for (int col = thread_id; col < batch; col += batch_threads) {
        {
          std::lock_guard<std::mutex> lock(error_mutex);
          if (!first_error.empty()) return;
        }
        try {
          const double* x_col =
            x_columns[static_cast<std::size_t>(col)];
          const double* y_col =
            y_columns[static_cast<std::size_t>(col)];
          results[static_cast<std::size_t>(col)] =
            legacy_dcov_gamma_cpp_compute_workspace(
              x_col, y_col, n, numCol, index,
              &workspaces[static_cast<std::size_t>(thread_id)]);
        } catch (const std::exception& error) {
          std::lock_guard<std::mutex> lock(error_mutex);
          if (first_error.empty()) first_error = error.what();
          return;
        } catch (...) {
          std::lock_guard<std::mutex> lock(error_mutex);
          if (first_error.empty()) {
            first_error = "legacy dCov gamma threaded batch failed";
          }
          return;
        }
      }
    };

    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(batch_threads));
    for (int thread_id = 0; thread_id < batch_threads; ++thread_id) {
      threads.emplace_back(worker, thread_id);
    }
    for (std::thread& thread : threads) thread.join();
    if (!first_error.empty()) Rcpp::stop(first_error);
  } else {
    for (int col = 0; col < batch; ++col) {
      const double* x_col = x_columns[static_cast<std::size_t>(col)];
      const double* y_col = y_columns[static_cast<std::size_t>(col)];
      results[static_cast<std::size_t>(col)] =
        legacy_dcov_gamma_cpp_compute_workspace(
          x_col, y_col, n, numCol, index, &workspaces[0]);
    }
  }

  int distance_workspace_reuse_count = 0;
  int statistic_moment_workspace_reuse_count = 0;
  int lowrank_output_workspace_reuse_count = 0;
  int lowrank_eig_workspace_reuse_count = 0;
  for (const LegacyDcovGammaCppWorkspace& workspace : workspaces) {
    distance_workspace_reuse_count += workspace.distance_workspace_reuse_count;
    statistic_moment_workspace_reuse_count +=
      workspace.statistic_moment_workspace_reuse_count;
    lowrank_output_workspace_reuse_count +=
      workspace.lowrank_output_workspace_reuse_count;
    lowrank_eig_workspace_reuse_count +=
      workspace.lowrank_eig_workspace.reuse_count;
  }

  for (int col = 0; col < batch; ++col) {
    const LegacyDcovGammaCppResult& result =
      results[static_cast<std::size_t>(col)];
    p_values[col] = result.p_value;
    nV2_values[col] = result.nV2;
    mean_values[col] = result.mean;
    variance_values[col] = result.variance;
    estimate_values[col] = result.estimate;
    scalar_total_ms += result.total_ms;
    input_ms += result.input_ms;
    distance_ms += result.distance_ms;
    lowrank_ms += result.lowrank_ms;
    lowrank_eig_ms += result.lowrank_timings.eig_ms;
    lowrank_select_ms += result.lowrank_timings.select_ms;
    lowrank_center_ms += result.lowrank_timings.center_ms;
    lowrank_unaccounted_ms += result.lowrank_unaccounted_ms;
    statistic_ms += result.statistic_ms;
    moment_ms += result.moment_ms;
    pgamma_ms += result.pgamma_ms;
    lowrank_timings.full_eig_count +=
      result.lowrank_timings.full_eig_count;
    lowrank_timings.spectra_count += result.lowrank_timings.spectra_count;
    lowrank_timings.spectra_converged_count +=
      result.lowrank_timings.spectra_converged_count;
    lowrank_timings.spectra_failed_count +=
      result.lowrank_timings.spectra_failed_count;
    lowrank_timings.spectra_fallback_full_eig_count +=
      result.lowrank_timings.spectra_fallback_full_eig_count;
    lowrank_timings.spectra_iterations +=
      result.lowrank_timings.spectra_iterations;
    lowrank_timings.spectra_nconv += result.lowrank_timings.spectra_nconv;
    lowrank_timings.spectra_ncv = std::max(
      lowrank_timings.spectra_ncv, result.lowrank_timings.spectra_ncv);
    lowrank_timings.spectra_tol = std::max(
      lowrank_timings.spectra_tol, result.lowrank_timings.spectra_tol);
    lowrank_timings.spectra_matvec_count +=
      result.lowrank_timings.spectra_matvec_count;
    lowrank_timings.spectra_matvec_ms +=
      result.lowrank_timings.spectra_matvec_ms;
    lowrank_mode = result.lowrank_mode;
  }
  const double total_ms = legacy_dcov_elapsed_ms_since(total_start);
  const double accounted_ms = input_ms + distance_ms + lowrank_ms +
    statistic_ms + moment_ms + pgamma_ms;
  const double wrapper_overhead_ms = std::max(0.0, total_ms - scalar_total_ms);
  const double batch_overhead_ms = std::max(0.0, total_ms - accounted_ms);
  return Rcpp::List::create(
    Rcpp::Named("p.value") = p_values,
    Rcpp::Named("nV2") = nV2_values,
    Rcpp::Named("mean") = mean_values,
    Rcpp::Named("variance") = variance_values,
    Rcpp::Named("statistic") = nV2_values,
    Rcpp::Named("estimate") = estimate_values,
    Rcpp::Named("diagnostics") = Rcpp::List::create(
      Rcpp::Named("n") = n,
      Rcpp::Named("batch_count") = batch,
      Rcpp::Named("numCol") = numCol,
      Rcpp::Named("index") = index,
      Rcpp::Named("lowrank_mode") =
        std::string(legacy_dcov_lowrank_mode_name(lowrank_mode)),
      Rcpp::Named("input_ms") = input_ms,
      Rcpp::Named("distance_ms") = distance_ms,
      Rcpp::Named("lowrank_ms") = lowrank_ms,
      Rcpp::Named("lowrank_eig_ms") = lowrank_eig_ms,
      Rcpp::Named("lowrank_select_ms") = lowrank_select_ms,
      Rcpp::Named("lowrank_center_ms") = lowrank_center_ms,
      Rcpp::Named("lowrank_unaccounted_ms") = lowrank_unaccounted_ms,
      Rcpp::Named("lowrank_full_eig_count") =
        lowrank_timings.full_eig_count,
      Rcpp::Named("lowrank_spectra_count") =
        lowrank_timings.spectra_count,
      Rcpp::Named("lowrank_spectra_converged_count") =
        lowrank_timings.spectra_converged_count,
      Rcpp::Named("lowrank_spectra_failed_count") =
        lowrank_timings.spectra_failed_count,
      Rcpp::Named("lowrank_spectra_fallback_full_eig_count") =
        lowrank_timings.spectra_fallback_full_eig_count,
      Rcpp::Named("lowrank_spectra_iterations") =
        lowrank_timings.spectra_iterations,
      Rcpp::Named("lowrank_spectra_nconv") =
        lowrank_timings.spectra_nconv,
      Rcpp::Named("lowrank_spectra_ncv") =
        lowrank_timings.spectra_ncv,
      Rcpp::Named("lowrank_spectra_tol") =
        lowrank_timings.spectra_tol,
      Rcpp::Named("lowrank_spectra_matvec_count") =
        lowrank_timings.spectra_matvec_count,
      Rcpp::Named("lowrank_spectra_matvec_ms") =
        lowrank_timings.spectra_matvec_ms,
      Rcpp::Named("statistic_ms") = statistic_ms,
      Rcpp::Named("moment_ms") = moment_ms,
      Rcpp::Named("pgamma_ms") = pgamma_ms,
      Rcpp::Named("accounted_ms") = accounted_ms,
      Rcpp::Named("scalar_total_ms") = scalar_total_ms,
      Rcpp::Named("wrapper_overhead_ms") = wrapper_overhead_ms,
      Rcpp::Named("batch_overhead_ms") = batch_overhead_ms,
      Rcpp::Named("workspace_reuse_enabled") = true,
      Rcpp::Named("distance_workspace_reuse_count") =
        distance_workspace_reuse_count,
      Rcpp::Named("statistic_moment_workspace_reuse_count") =
        statistic_moment_workspace_reuse_count,
      Rcpp::Named("lowrank_output_workspace_reuse_count") =
        lowrank_output_workspace_reuse_count,
      Rcpp::Named("lowrank_eig_workspace_reuse_count") =
        lowrank_eig_workspace_reuse_count,
      Rcpp::Named("column_copy_count") = 0,
      Rcpp::Named("batch_parallel_enabled") = batch_parallel_enabled,
      Rcpp::Named("batch_parallel_threads") = batch_threads,
      Rcpp::Named("unaccounted_ms") = std::max(0.0, total_ms - accounted_ms),
      Rcpp::Named("total_ms") = total_ms
    )
  );
}

Rcpp::List legacy_dcov_gamma_cpp_compute_batch(Rcpp::NumericMatrix x,
                                               Rcpp::NumericMatrix y,
                                               int numCol,
                                               double index) {
  if (x.nrow() != y.nrow() || x.ncol() != y.ncol()) {
    Rcpp::stop("x and y must have matching dimensions");
  }
  const int n = x.nrow();
  const int batch = x.ncol();
  std::vector<const double*> x_columns(static_cast<std::size_t>(batch));
  std::vector<const double*> y_columns(static_cast<std::size_t>(batch));
  for (int col = 0; col < batch; ++col) {
    x_columns[static_cast<std::size_t>(col)] =
      x.begin() + static_cast<std::ptrdiff_t>(col) * n;
    y_columns[static_cast<std::size_t>(col)] =
      y.begin() + static_cast<std::ptrdiff_t>(col) * n;
  }
  return legacy_dcov_gamma_cpp_compute_batch_ptrs(
    x_columns, y_columns, n, numCol, index);
}

}  // namespace fastkpc
