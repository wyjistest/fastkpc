#include "full_cuda_ci_one_call.hpp"

#include "full_cuda_ci_contract.hpp"
#include "full_cuda_ci_native_setup.hpp"
#include "skeleton_task_scheduler.hpp"
#include "cuda/full_cuda_ci_vertical.hpp"
#include "cuda/mgcv_fixed_sp_runtime.hpp"
#include "cuda/mgcv_multi_penalty_gcv.hpp"
#include "cuda/mgcv_single_penalty_gcv.hpp"

#include <RcppEigen.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fastkpc {
namespace {

using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::ColMajor>;

constexpr double kQualifiedAlpha = 0.1;
constexpr double kGuardLower = 0.05;
constexpr double kGuardUpper = 0.15;
constexpr int kPreparedCacheCapacity = 64;
constexpr int kReferenceComponentCapacity = 47;
constexpr double kCholeskyConditionMax = 1e8;
constexpr double kSvdConditionMin = 1e12;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

double elapsed_ms(
    const std::chrono::steady_clock::time_point& start) {
  return std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - start).count();
}

bool finite_matrix(const Rcpp::NumericMatrix& values) {
  return std::all_of(values.begin(), values.end(), [](double value) {
    return std::isfinite(value);
  });
}

bool is_lower_sha256(const std::string& value) {
  return value.size() == 64U &&
    std::all_of(value.begin(), value.end(), [](unsigned char character) {
      return (character >= '0' && character <= '9') ||
        (character >= 'a' && character <= 'f');
    });
}

std::string conditioning_key(const std::vector<int>& conditioning_set) {
  std::ostringstream output;
  for (std::size_t index = 0; index < conditioning_set.size(); ++index) {
    if (index != 0U) output << "|";
    output << conditioning_set[index] + 1;
  }
  return output.str();
}

std::string dataset_key(const Rcpp::NumericMatrix& data) {
  std::ostringstream payload;
  payload << "schema=full-cuda-ci-dataset-key-v1\n"
          << "n=" << data.nrow() << "\n"
          << "p=" << data.ncol() << "\n"
          << "type=binary64-column-major\n"
          << "values=" << std::hex << std::setfill('0');
  for (double value : data) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    payload << std::setw(16) << bits;
  }
  payload << "\n";
  return full_cuda_ci_sha256_utf8(payload.str());
}

std::string prepared_key(const std::string& dataset,
                         const std::vector<int>& conditioning_set) {
  std::ostringstream payload;
  payload << "schema=full-cuda-ci-prepared-s-key-v1\n"
          << "dataset=" << dataset << "\n"
          << "S=" << conditioning_key(conditioning_set) << "\n"
          << "formula_route="
          << (conditioning_set.size() <= 2U ? "full-smooth" :
              "additive-smooth") << "\n"
          << "semantic=mgcv-1.9-1-tprs-native-setup-v1\n";
  return full_cuda_ci_sha256_utf8(payload.str());
}

std::string target_key(const std::string& prepared, int target) {
  return full_cuda_ci_sha256_utf8(
    "schema=full-cuda-ci-target-key-v1\nprepared=" + prepared +
    "\ntarget=" + std::to_string(target + 1) + "\n");
}

std::string residual_key(const std::string& target,
                         const double* smoothing_parameters,
                         int penalty_count) {
  std::ostringstream payload;
  payload << "schema=full-cuda-ci-residual-key-v1\n"
          << "target=" << target << "\n"
          << "solver=full-cuda-ci-fixed-sp-runtime-v1\n"
          << "selected_sp_bits=" << std::hex << std::setfill('0');
  for (int penalty = 0; penalty < penalty_count; ++penalty) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, smoothing_parameters + penalty, sizeof(bits));
    payload << std::setw(16) << bits;
  }
  payload << "\n";
  return full_cuda_ci_sha256_utf8(payload.str());
}

Rcpp::LogicalMatrix adjacency_matrix(const std::vector<int>& adjacency,
                                     int p) {
  Rcpp::LogicalMatrix output(p, p);
  for (int row = 0; row < p; ++row) {
    for (int column = 0; column < p; ++column) {
      output(row, column) =
        adjacency[static_cast<std::size_t>(row) * p + column] != 0;
    }
  }
  return output;
}

Rcpp::NumericMatrix pmax_matrix(const std::vector<double>& pmax, int p) {
  Rcpp::NumericMatrix output(p, p);
  for (int row = 0; row < p; ++row) {
    for (int column = 0; column < p; ++column) {
      output(row, column) =
        pmax[static_cast<std::size_t>(row) * p + column];
    }
  }
  return output;
}

Rcpp::List sepset_list(
    const std::vector<std::vector<std::vector<int>>>& sepsets) {
  const int p = static_cast<int>(sepsets.size());
  Rcpp::List output(p);
  for (int row = 0; row < p; ++row) {
    Rcpp::List values(p);
    for (int column = 0; column < p; ++column) {
      Rcpp::IntegerVector entry(sepsets[row][column].size());
      for (int index = 0; index < entry.size(); ++index) {
        entry[index] = sepsets[row][column][index] + 1;
      }
      values[column] = entry;
    }
    output[row] = values;
  }
  return output;
}

std::vector<double> copy_numeric_matrix(SEXP value,
                                        int expected_rows,
                                        int expected_columns,
                                        const std::string& label) {
  require(Rf_isReal(value) && Rf_isMatrix(value),
          label + " must be a numeric matrix");
  Rcpp::NumericMatrix matrix(value);
  require(matrix.nrow() == expected_rows &&
            matrix.ncol() == expected_columns && finite_matrix(matrix),
          label + " dimensions or values are invalid");
  return std::vector<double>(matrix.begin(), matrix.end());
}

std::vector<double> copy_numeric_vector(SEXP value,
                                        int expected_size,
                                        const std::string& label) {
  require(Rf_isReal(value), label + " must be numeric");
  Rcpp::NumericVector vector(value);
  require(vector.size() == expected_size &&
            std::all_of(vector.begin(), vector.end(), [](double item) {
              return std::isfinite(item);
            }),
          label + " length or values are invalid");
  return std::vector<double>(vector.begin(), vector.end());
}

struct SinglePenaltyGeometry {
  int penalty_rank = 0;
  double initial_sp = 0.0;
  std::vector<double> rhs_transform;
  std::vector<double> eigenvalues;
  std::vector<double> magic_qr_packed;
  std::vector<double> magic_tau;
  std::vector<double> magic_r;
  std::vector<double> magic_penalty_root;
  std::vector<double> magic_penalty_matrix;
  Matrix full_penalty;
  Matrix gram;
};

struct NativeSetupContext {
  std::string prepared_key;
  std::vector<int> conditioning_set;
  Rcpp::List setup;
  Rcpp::List geometry;
  int n = 0;
  int coefficient_dim = 0;
  int penalty_count = 0;
  std::shared_ptr<PreparedSGpuHandle> fixed_sp;
  std::shared_ptr<MultiPenaltyGcvCudaPrepared> multi_penalty;
  std::unique_ptr<SinglePenaltyGeometry> single_penalty;
};

struct OneCallDiagnostics {
  int native_setup_count = 0;
  int native_setup_cache_request_count = 0;
  int native_setup_cache_hit_count = 0;
  int native_setup_cache_miss_count = 0;
  int native_setup_cache_eviction_count = 0;
  int cuda_single_penalty_target_count = 0;
  int cuda_multi_penalty_target_count = 0;
  int cuda_residual_batch_count = 0;
  int logical_residual_request_count = 0;
  int physical_residual_fit_count = 0;
  int cuda_dcov_component_count = 0;
  int cuda_dcov_pair_count = 0;
  int cuda_gamma_pvalue_count = 0;
  int guarded_pair_count = 0;
  int component_cache_request_count = 0;
  int component_cache_hit_count = 0;
  int component_cache_miss_count = 0;
  int component_cache_eviction_count = 0;
  int host_synchronization_count = 0;
  int cpu_dcov_component_count = 0;
  int cpu_dcov_eigen_or_lowrank_count = 0;
  int cpu_dcov_pair_stat_count = 0;
  int cpu_gamma_pvalue_count = 0;
  int cpu_spectra_count = 0;
  std::size_t matrix_h2d_bytes = 0;
  std::size_t residual_d2h_bytes = 0;
  std::size_t component_d2h_bytes = 0;
  std::size_t compact_result_d2h_bytes = 0;
  double native_setup_ms = 0.0;
  double cuda_optimizer_host_ms = 0.0;
  double cuda_dcov_host_ms = 0.0;
};

std::vector<const double*> penalty_block_pointers(
    const Rcpp::List& penalty_blocks) {
  std::vector<const double*> pointers;
  pointers.reserve(static_cast<std::size_t>(penalty_blocks.size()));
  for (int index = 0; index < penalty_blocks.size(); ++index) {
    Rcpp::NumericMatrix block(penalty_blocks[index]);
    pointers.push_back(block.begin());
  }
  return pointers;
}

std::vector<int> penalty_block_dimensions(
    const Rcpp::List& penalty_blocks) {
  std::vector<int> dimensions;
  dimensions.reserve(static_cast<std::size_t>(penalty_blocks.size()));
  for (int index = 0; index < penalty_blocks.size(); ++index) {
    Rcpp::NumericMatrix block(penalty_blocks[index]);
    require(block.nrow() == block.ncol() && finite_matrix(block),
            "native setup penalty block is malformed");
    dimensions.push_back(block.nrow());
  }
  return dimensions;
}

std::shared_ptr<PreparedSGpuHandle> create_fixed_sp_handle(
    const std::shared_ptr<CudaRuntimeContext>& runtime,
    const std::string& dataset,
    const std::shared_ptr<NativeSetupContext>& context) {
  Rcpp::NumericMatrix X = context->setup["X"];
  Rcpp::NumericMatrix gram = context->setup["gram_matrix"];
  Rcpp::List blocks = context->setup["penalty_blocks"];
  Rcpp::IntegerVector offsets = context->setup["penalty_offsets"];
  Rcpp::IntegerVector ranks = context->setup["penalty_ranks"];
  require(X.nrow() == context->n && X.ncol() == context->coefficient_dim &&
            gram.nrow() == context->coefficient_dim &&
            gram.ncol() == context->coefficient_dim &&
            offsets.size() == context->penalty_count &&
            ranks.size() == context->penalty_count,
          "native setup fixed-sp geometry is malformed");

  const std::vector<const double*> pointers = penalty_block_pointers(blocks);
  const std::vector<int> dimensions = penalty_block_dimensions(blocks);
  std::vector<int> offsets_zero_based(offsets.size());
  std::vector<int> rank_values(ranks.size());
  std::vector<int> sp_indices(ranks.size());
  for (int index = 0; index < offsets.size(); ++index) {
    offsets_zero_based[static_cast<std::size_t>(index)] = offsets[index] - 1;
    rank_values[static_cast<std::size_t>(index)] = ranks[index];
    sp_indices[static_cast<std::size_t>(index)] = index;
  }

  PreparedSHostView view;
  view.dataset_sha256 = dataset;
  view.prepared_s_key_sha256 = context->prepared_key;
  view.same_s_group_id = full_cuda_ci_sha256_utf8(
    "full-cuda-ci-one-call-group-v1\n" + context->prepared_key + "\n");
  view.semantic_fingerprint = Rcpp::as<std::string>(
    context->setup["semantic_fingerprint"]);
  view.representation_fingerprint = full_cuda_ci_sha256_utf8(
    "full-cuda-ci-one-call-representation-v1\n" +
    context->prepared_key + "\n");
  view.n = context->n;
  view.coefficient_dim = context->coefficient_dim;
  view.null_dim = context->coefficient_dim;
  view.penalty_count = context->penalty_count;
  view.X = X.begin();
  view.Z = nullptr;
  view.gram = gram.begin();
  view.H = nullptr;
  view.penalty_blocks = pointers;
  view.penalty_dimensions = dimensions;
  view.penalty_offsets_zero_based = offsets_zero_based;
  view.penalty_ranks = rank_values;
  view.penalty_sp_indices_zero_based = sp_indices;
  return create_prepared_s_gpu(runtime, view);
}

SinglePenaltyGeometry prepare_single_penalty_geometry(
    const std::shared_ptr<NativeSetupContext>& context) {
  Rcpp::NumericMatrix X = context->setup["X"];
  Rcpp::NumericMatrix gram_input = context->setup["gram_matrix"];
  Rcpp::List blocks = context->setup["penalty_blocks"];
  Rcpp::IntegerVector offsets = context->setup["penalty_offsets"];
  Rcpp::IntegerVector ranks = context->setup["penalty_ranks"];
  require(blocks.size() == 1 && offsets.size() == 1 && ranks.size() == 1,
          "single-penalty geometry requires one penalty block");
  Rcpp::NumericMatrix block = blocks[0];
  const int q = context->coefficient_dim;
  const int offset = offsets[0] - 1;
  const int rank = ranks[0];
  require(offset >= 0 && offset + block.nrow() <= q &&
            rank > 0 && rank <= q,
          "single-penalty block placement is invalid");

  Eigen::Map<const Matrix> gram_map(gram_input.begin(), q, q);
  Matrix gram = gram_map;
  Matrix penalty = Matrix::Zero(q, q);
  Eigen::Map<const Matrix> block_map(block.begin(), block.nrow(), block.ncol());
  penalty.block(offset, offset, block.nrow(), block.ncol()) = block_map;
  Eigen::LLT<Matrix> decomposition(gram);
  require(decomposition.info() == Eigen::Success,
          "single-penalty Gram Cholesky failed closed");
  const Matrix upper = decomposition.matrixU();
  const Matrix inverse = upper.template triangularView<Eigen::Upper>().solve(
    Matrix::Identity(q, q));
  Matrix whitened = inverse.transpose() * penalty * inverse;
  whitened = 0.5 * (whitened + whitened.transpose()).eval();
  Eigen::SelfAdjointEigenSolver<Matrix> eigen(whitened);
  require(eigen.info() == Eigen::Success,
          "single-penalty spectral decomposition failed closed");

  Matrix vectors(q, q);
  std::vector<double> values(static_cast<std::size_t>(q), 0.0);
  for (int column = 0; column < q; ++column) {
    const int source = q - column - 1;
    values[static_cast<std::size_t>(column)] =
      std::max(0.0, eigen.eigenvalues()[source]);
    vectors.col(column) = eigen.eigenvectors().col(source);
  }
  const int nullity = q - rank;
  for (int index = q - nullity; index < q; ++index) {
    if (index >= 0) values[static_cast<std::size_t>(index)] = 0.0;
  }
  require(static_cast<int>(std::count_if(
            values.begin(), values.end(), [](double value) {
              return value > 0.0;
            })) == rank,
          "single-penalty spectral rank disagrees with native setup");
  const Matrix rhs_transform = vectors.transpose() * inverse.transpose();

  SinglePenaltyGeometry result;
  result.penalty_rank = rank;
  result.rhs_transform.assign(rhs_transform.data(),
                              rhs_transform.data() + rhs_transform.size());
  result.eigenvalues = std::move(values);
  result.full_penalty = std::move(penalty);
  result.gram = std::move(gram);
  result.magic_qr_packed = copy_numeric_matrix(
    context->geometry["magic_qr_packed"], context->n, q,
    "single-penalty native QR");
  result.magic_tau = copy_numeric_vector(
    context->geometry["magic_tau"], q, "single-penalty native tau");
  result.magic_r = copy_numeric_matrix(
    context->geometry["magic_r"], q, q, "single-penalty native R");
  Rcpp::List roots = context->geometry["penalty_roots"];
  Rcpp::List matrices = context->geometry["penalty_matrices"];
  result.magic_penalty_root = copy_numeric_matrix(
    roots[0], q, rank, "single-penalty native root");
  result.magic_penalty_matrix = copy_numeric_matrix(
    matrices[0], q, q, "single-penalty native matrix");
  Rcpp::NumericVector initial = context->geometry["initial_sp"];
  require(initial.size() == 1 && std::isfinite(initial[0]) && initial[0] > 0,
          "single-penalty initial sp is invalid");
  result.initial_sp = initial[0];
  return result;
}

std::vector<std::vector<double>> numeric_matrix_list(
    const Rcpp::List& values,
    int expected_rows,
    const std::string& label) {
  std::vector<std::vector<double>> output;
  output.reserve(static_cast<std::size_t>(values.size()));
  for (int index = 0; index < values.size(); ++index) {
    require(Rf_isReal(values[index]) && Rf_isMatrix(values[index]),
            label + " entry must be a numeric matrix");
    Rcpp::NumericMatrix matrix(values[index]);
    require(matrix.nrow() == expected_rows && finite_matrix(matrix),
            label + " entry is malformed");
    output.emplace_back(matrix.begin(), matrix.end());
  }
  return output;
}

std::shared_ptr<MultiPenaltyGcvCudaPrepared> create_multi_penalty_handle(
    const std::shared_ptr<NativeSetupContext>& context,
    int target_capacity) {
  Rcpp::NumericMatrix X = context->setup["X"];
  Rcpp::NumericMatrix qr = context->geometry["magic_qr_packed"];
  Rcpp::NumericVector tau = context->geometry["magic_tau"];
  Rcpp::NumericMatrix R = context->geometry["magic_r"];
  Rcpp::IntegerVector pivot = context->geometry["magic_pivot"];
  Rcpp::List roots_input = context->geometry["penalty_roots"];
  Rcpp::List matrices_input = context->geometry["penalty_matrices"];
  Rcpp::IntegerVector ranks_input = context->setup["penalty_ranks"];
  Rcpp::NumericVector initial_log_sp = context->geometry["initial_log_sp"];
  const int q = context->coefficient_dim;
  require(qr.nrow() == context->n && qr.ncol() == q &&
            tau.size() == q && R.nrow() == q && R.ncol() == q &&
            pivot.size() == q && ranks_input.size() == context->penalty_count &&
            initial_log_sp.size() == context->penalty_count,
          "multi-penalty native geometry is malformed");
  std::vector<int> pivot_zero(static_cast<std::size_t>(q));
  for (int index = 0; index < q; ++index) pivot_zero[index] = pivot[index] - 1;
  std::vector<int> ranks(ranks_input.begin(), ranks_input.end());
  const std::vector<std::vector<double>> roots = numeric_matrix_list(
    roots_input, q, "multi-penalty roots");
  const std::vector<std::vector<double>> matrices = numeric_matrix_list(
    matrices_input, q, "multi-penalty matrices");
  return create_multi_penalty_gcv_cuda_prepared(
    X.begin(), qr.begin(), tau.begin(), R.begin(), pivot_zero.data(),
    roots, matrices, ranks, initial_log_sp.begin(), context->n, q,
    context->penalty_count, target_capacity, 0);
}

std::shared_ptr<NativeSetupContext> build_native_context(
    const Rcpp::NumericMatrix& data,
    const std::string& dataset,
    const std::vector<int>& conditioning_set,
    const std::shared_ptr<CudaRuntimeContext>& runtime,
    OneCallDiagnostics* diagnostics) {
  const auto started = std::chrono::steady_clock::now();
  Rcpp::NumericMatrix conditioning(data.nrow(), conditioning_set.size());
  for (int column = 0; column < conditioning.ncol(); ++column) {
    const int source = conditioning_set[static_cast<std::size_t>(column)];
    std::copy(data.begin() + static_cast<std::ptrdiff_t>(source) * data.nrow(),
              data.begin() + static_cast<std::ptrdiff_t>(source + 1) * data.nrow(),
              conditioning.begin() +
                static_cast<std::ptrdiff_t>(column) * data.nrow());
  }
  auto context = std::make_shared<NativeSetupContext>();
  context->prepared_key = prepared_key(dataset, conditioning_set);
  context->conditioning_set = conditioning_set;
  context->setup = full_cuda_ci_native_setup(conditioning);
  context->n = data.nrow();
  Rcpp::NumericMatrix X = context->setup["X"];
  Rcpp::List blocks = context->setup["penalty_blocks"];
  Rcpp::IntegerVector offsets = context->setup["penalty_offsets"];
  Rcpp::IntegerVector ranks = context->setup["penalty_ranks"];
  context->coefficient_dim = X.ncol();
  context->penalty_count = blocks.size();
  context->geometry = full_cuda_ci_native_geometry_prepare(
    X, blocks, offsets, ranks);
  context->fixed_sp = create_fixed_sp_handle(runtime, dataset, context);
  if (context->penalty_count == 1) {
    context->single_penalty.reset(
      new SinglePenaltyGeometry(prepare_single_penalty_geometry(context)));
  } else {
    context->multi_penalty = create_multi_penalty_handle(context, 64);
  }
  diagnostics->native_setup_count += 1;
  diagnostics->native_setup_ms += elapsed_ms(started);
  return context;
}

std::shared_ptr<NativeSetupContext> build_direct_context(
    const Rcpp::NumericMatrix& data,
    const std::string& dataset,
    const std::shared_ptr<CudaRuntimeContext>& runtime) {
  auto context = std::make_shared<NativeSetupContext>();
  context->prepared_key = full_cuda_ci_sha256_utf8(
    "schema=full-cuda-ci-direct-prepared-key-v1\ndataset=" + dataset +
    "\n");
  context->n = data.nrow();
  context->coefficient_dim = 1;
  context->penalty_count = 1;
  Rcpp::NumericMatrix X(data.nrow(), 1);
  std::fill(X.begin(), X.end(), 1.0);
  Rcpp::NumericMatrix gram(1, 1);
  gram(0, 0) = static_cast<double>(data.nrow());
  Rcpp::NumericMatrix block(1, 1);
  block(0, 0) = 1.0;
  Rcpp::List blocks = Rcpp::List::create(block);
  context->setup = Rcpp::List::create(
    Rcpp::Named("semantic_fingerprint") = full_cuda_ci_sha256_utf8(
      "full-cuda-ci-direct-shift-invariant-v1\n" + dataset + "\n"),
    Rcpp::Named("X") = X,
    Rcpp::Named("gram_matrix") = gram,
    Rcpp::Named("penalty_blocks") = blocks,
    Rcpp::Named("penalty_offsets") = Rcpp::IntegerVector::create(1),
    Rcpp::Named("penalty_ranks") = Rcpp::IntegerVector::create(1));
  context->fixed_sp = create_fixed_sp_handle(runtime, dataset, context);
  return context;
}

void close_context(const std::shared_ptr<NativeSetupContext>& context) {
  if (!context) return;
  context->multi_penalty.reset();
  if (context->fixed_sp) {
    try {
      free_prepared_s_gpu(&context->fixed_sp);
    } catch (...) {
      context->fixed_sp.reset();
    }
  }
}

class NativeSetupCache {
 public:
  NativeSetupCache(const Rcpp::NumericMatrix& data,
                   std::string dataset,
                   std::shared_ptr<CudaRuntimeContext> runtime,
                   OneCallDiagnostics* diagnostics)
      : data_(data),
        dataset_(std::move(dataset)),
        runtime_(std::move(runtime)),
        diagnostics_(diagnostics) {}

  std::shared_ptr<NativeSetupContext> get(
      const std::vector<int>& conditioning_set) {
    const std::string key = prepared_key(dataset_, conditioning_set);
    diagnostics_->native_setup_cache_request_count += 1;
    auto found = entries_.find(key);
    if (found != entries_.end()) {
      diagnostics_->native_setup_cache_hit_count += 1;
      order_.erase(found->second.order);
      order_.push_front(key);
      found->second.order = order_.begin();
      return found->second.context;
    }
    diagnostics_->native_setup_cache_miss_count += 1;
    if (entries_.size() >= kPreparedCacheCapacity) {
      const std::string evicted = order_.back();
      order_.pop_back();
      auto victim = entries_.find(evicted);
      close_context(victim->second.context);
      entries_.erase(victim);
      diagnostics_->native_setup_cache_eviction_count += 1;
    }
    std::shared_ptr<NativeSetupContext> context = build_native_context(
      data_, dataset_, conditioning_set, runtime_, diagnostics_);
    order_.push_front(key);
    entries_.emplace(key, Entry{context, order_.begin()});
    return context;
  }

  void clear() {
    for (auto& entry : entries_) close_context(entry.second.context);
    entries_.clear();
    order_.clear();
  }

  ~NativeSetupCache() {
    clear();
  }

 private:
  struct Entry {
    std::shared_ptr<NativeSetupContext> context;
    std::list<std::string>::iterator order;
  };

  Rcpp::NumericMatrix data_;
  std::string dataset_;
  std::shared_ptr<CudaRuntimeContext> runtime_;
  OneCallDiagnostics* diagnostics_;
  std::list<std::string> order_;
  std::unordered_map<std::string, Entry> entries_;
};

FixedSpRoute route_from_condition(double condition, bool full_rank) {
  if (!full_rank || !std::isfinite(condition) ||
      condition >= kSvdConditionMin) {
    return FixedSpRoute::AugmentedSvd;
  }
  if (condition >= kCholeskyConditionMax) {
    return FixedSpRoute::AugmentedQr;
  }
  return FixedSpRoute::CholeskyBatched;
}

double single_penalty_condition(const SinglePenaltyGeometry& geometry,
                                double sp) {
  Matrix system = geometry.gram + sp * geometry.full_penalty;
  Eigen::SelfAdjointEigenSolver<Matrix> eigen(system,
                                              Eigen::EigenvaluesOnly);
  require(eigen.info() == Eigen::Success,
          "single-penalty selected system condition failed closed");
  const double maximum = eigen.eigenvalues().cwiseAbs().maxCoeff();
  const double minimum = eigen.eigenvalues().cwiseAbs().minCoeff();
  if (!std::isfinite(maximum) || !std::isfinite(minimum) || minimum <= 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  return maximum / minimum;
}

struct GroupResult {
  std::vector<double> p_values;
};

void accumulate_exact_diagnostics(
    const FullCudaCiExactBatchDiagnostics& value,
    OneCallDiagnostics* diagnostics) {
  diagnostics->cuda_residual_batch_count += 1;
  diagnostics->cuda_dcov_component_count += value.component_build_count;
  diagnostics->cuda_dcov_pair_count += value.pair_evaluation_count;
  diagnostics->cuda_gamma_pvalue_count += value.pair_evaluation_count;
  diagnostics->component_cache_request_count +=
    value.component_cache_lookup_count;
  diagnostics->component_cache_hit_count += value.component_cache_hit_count;
  diagnostics->component_cache_miss_count += value.component_cache_miss_count;
  diagnostics->component_cache_eviction_count +=
    value.component_cache_eviction_count;
  diagnostics->host_synchronization_count += value.explicit_host_wait_count;
  diagnostics->cpu_dcov_component_count += value.cpu_dcov_component_count;
  diagnostics->cpu_dcov_pair_stat_count +=
    value.cpu_dcov_pair_statistic_count;
  diagnostics->cpu_gamma_pvalue_count += value.cpu_gamma_p_value_count;
  diagnostics->residual_d2h_bytes += value.residual_d2h_bytes;
  diagnostics->component_d2h_bytes += value.component_d2h_bytes;
  diagnostics->compact_result_d2h_bytes += value.compact_result_d2h_bytes;
  diagnostics->cuda_dcov_host_ms += value.dcov_host_boundary_ms;
  require(value.bounded_allocation && value.leak_free_teardown &&
            value.component_capacity_respected &&
            value.residual_d2h_bytes == 0U &&
            value.component_d2h_bytes == 0U &&
            value.cpu_dcov_component_count == 0 &&
            value.cpu_dcov_pair_statistic_count == 0 &&
            value.cpu_gamma_p_value_count == 0,
          "one-call exact CUDA dCov authority gate failed closed");
}

void accumulate_refinement_diagnostics(
    const FullCudaCiLegacyEigBatchDiagnostics& value,
    OneCallDiagnostics* diagnostics) {
  diagnostics->cuda_residual_batch_count += 1;
  diagnostics->cuda_dcov_component_count += value.component_build_count;
  diagnostics->cuda_dcov_pair_count += value.cuda_pair_count;
  diagnostics->cuda_gamma_pvalue_count += value.cuda_gamma_count;
  diagnostics->component_cache_request_count += 2 * value.pair_count;
  diagnostics->component_cache_miss_count += value.referenced_component_count;
  diagnostics->component_cache_hit_count +=
    2 * value.pair_count - value.referenced_component_count;
  diagnostics->host_synchronization_count += value.explicit_host_wait_count;
  diagnostics->cpu_dcov_component_count += value.cpu_dcov_component_count;
  diagnostics->cpu_dcov_eigen_or_lowrank_count += value.cpu_dcov_eigen_count;
  diagnostics->cpu_dcov_pair_stat_count +=
    value.cpu_dcov_pair_statistic_count;
  diagnostics->cpu_gamma_pvalue_count += value.cpu_gamma_p_value_count;
  diagnostics->residual_d2h_bytes += value.residual_d2h_bytes;
  diagnostics->component_d2h_bytes += value.component_d2h_bytes;
  diagnostics->compact_result_d2h_bytes += value.compact_result_d2h_bytes;
  diagnostics->cuda_dcov_host_ms += value.dcov_host_boundary_ms;
  require(value.solver_failure_count == 0 && value.bounded_allocation &&
            value.leak_free_teardown && value.component_capacity_respected &&
            value.residual_d2h_bytes == 0U &&
            value.component_d2h_bytes == 0U &&
            value.cpu_dcov_component_count == 0 &&
            value.cpu_dcov_eigen_count == 0 &&
            value.cpu_dcov_pair_statistic_count == 0 &&
            value.cpu_gamma_p_value_count == 0,
          "one-call guarded CUDA legacy dCov authority gate failed closed");
}

GroupResult execute_group(
    const Rcpp::NumericMatrix& data,
    const std::shared_ptr<NativeSetupContext>& context,
    const LayerPlan& plan,
    const std::vector<int>& task_indices,
    const std::vector<std::uint64_t>& logical_sequence_ids,
    int num_col,
    bool direct,
    OneCallDiagnostics* diagnostics) {
  require(!task_indices.empty() &&
            task_indices.size() == logical_sequence_ids.size(),
          "one-call CUDA group request is empty or misaligned");
  std::set<int> target_set;
  for (int task_index : task_indices) {
    const LayerCiTask& task = plan.tasks[static_cast<std::size_t>(task_index)];
    target_set.insert(task.orientation_x);
    target_set.insert(task.orientation_y);
  }
  std::vector<int> targets(target_set.begin(), target_set.end());
  const int target_count = static_cast<int>(targets.size());
  require(target_count >= 2 && target_count <= 64,
          "one-call CUDA same-S target count is outside the envelope");
  std::unordered_map<int, int> target_position;
  for (int index = 0; index < target_count; ++index) {
    target_position[targets[static_cast<std::size_t>(index)]] = index;
  }
  std::vector<double> Y(
    static_cast<std::size_t>(data.nrow()) * target_count);
  for (int column = 0; column < target_count; ++column) {
    const int source = targets[static_cast<std::size_t>(column)];
    std::copy(data.begin() + static_cast<std::ptrdiff_t>(source) * data.nrow(),
              data.begin() + static_cast<std::ptrdiff_t>(source + 1) * data.nrow(),
              Y.begin() + static_cast<std::ptrdiff_t>(column) * data.nrow());
  }

  std::vector<double> selected_sp(
    static_cast<std::size_t>(context->penalty_count) * target_count, 0.0);
  std::vector<FixedSpRoute> planned_routes(
    static_cast<std::size_t>(target_count), FixedSpRoute::CholeskyBatched);
  std::vector<std::string> base_target_keys;
  base_target_keys.reserve(static_cast<std::size_t>(target_count));
  for (int target : targets) {
    base_target_keys.push_back(target_key(context->prepared_key, target));
  }

  if (!direct) {
    const auto optimize_start = std::chrono::steady_clock::now();
    if (context->penalty_count == 1) {
      require(context->single_penalty != nullptr,
              "single-penalty CUDA geometry is missing");
      const SinglePenaltyGeometry& geometry = *context->single_penalty;
      Rcpp::NumericMatrix X = context->setup["X"];
      std::vector<int> target_ids(target_count);
      for (int index = 0; index < target_count; ++index) {
        target_ids[static_cast<std::size_t>(index)] =
          targets[static_cast<std::size_t>(index)] + 1;
      }
      const SinglePenaltyGcvCudaResult optimization =
        single_penalty_gcv_cuda(
          X.begin(), Y.data(), geometry.rhs_transform.data(),
          geometry.eigenvalues.data(), geometry.magic_qr_packed.data(),
          geometry.magic_tau.data(), geometry.magic_r.data(),
          geometry.magic_penalty_root.data(),
          geometry.magic_penalty_matrix.data(), target_ids.data(),
          context->n, context->coefficient_dim, target_count,
          geometry.penalty_rank, geometry.initial_sp, {}, false, false);
      require(optimization.target_count == target_count &&
                optimization.diagnostics.legacy_mgcv_target_calls == 0 &&
                optimization.diagnostics.cpu_score_count == 0 &&
                optimization.diagnostics.cpu_optimizer_count == 0 &&
                optimization.diagnostics.fallback_count == 0 &&
                optimization.diagnostics.optimizer_target_coverage_complete,
              "single-penalty live CUDA optimizer authority gate failed");
      for (int target = 0; target < target_count; ++target) {
        const SinglePenaltyGcvOptimizerResult& result =
          optimization.targets[static_cast<std::size_t>(target)];
        const bool accepted =
          result.termination == static_cast<int>(
            SinglePenaltyGcvTermination::ScoreAndGradient) ||
          result.termination == static_cast<int>(
            SinglePenaltyGcvTermination::StepHalvingExhausted) ||
          result.termination == static_cast<int>(
            SinglePenaltyGcvTermination::FlatObjective);
        require(accepted && std::isfinite(result.sp) && result.sp > 0.0,
                "single-penalty live CUDA optimizer failed closed");
        selected_sp[static_cast<std::size_t>(target)] = result.sp;
        planned_routes[static_cast<std::size_t>(target)] =
          route_from_condition(
            single_penalty_condition(geometry, result.sp), true);
      }
      diagnostics->cuda_single_penalty_target_count += target_count;
    } else {
      require(context->multi_penalty != nullptr,
              "multi-penalty CUDA prepared handle is missing");
      MultiPenaltyGcvCudaOptimizerControl control;
      MultiPenaltyGcvCudaBatchResult optimization =
        multi_penalty_gcv_cuda_optimize_batch(
          context->multi_penalty, Y.data(), context->n, target_count,
          base_target_keys, control);
      try {
        const MultiPenaltyGcvCudaOptimization& result =
          optimization.optimization;
        require(result.target_count == target_count &&
                  result.penalty_count == context->penalty_count &&
                  result.selected_log_sp.size() == selected_sp.size() &&
                  result.diagnostics.cpu_objective_count == 0 &&
                  result.diagnostics.cpu_optimizer_count == 0 &&
                  result.diagnostics.cpu_multi_penalty_solve_count == 0 &&
                  result.diagnostics.fallback_count == 0 &&
                  result.diagnostics.cuda_error_count == 0 &&
                  result.diagnostics.true_batched_kernel &&
                  result.diagnostics.independent_target_states,
                "multi-penalty live CUDA optimizer authority gate failed");
        for (std::size_t index = 0; index < selected_sp.size(); ++index) {
          selected_sp[index] = std::exp(result.selected_log_sp[index]);
          require(std::isfinite(selected_sp[index]) && selected_sp[index] > 0,
                  "multi-penalty selected sp is non-finite");
        }
        for (int target = 0; target < target_count; ++target) {
          require(result.optimizer_status[static_cast<std::size_t>(target)] == 0,
                  "multi-penalty CUDA optimizer returned an error status");
          planned_routes[static_cast<std::size_t>(target)] =
            route_from_condition(
              result.condition[static_cast<std::size_t>(target)],
              result.numerical_rank[static_cast<std::size_t>(target)] ==
                context->coefficient_dim);
        }
        release_multi_penalty_gcv_cuda_residual(optimization.residual);
        optimization.residual.reset();
      } catch (...) {
        if (optimization.residual) {
          try {
            release_multi_penalty_gcv_cuda_residual(optimization.residual);
          } catch (...) {
          }
        }
        throw;
      }
      diagnostics->cuda_multi_penalty_target_count += target_count;
    }
    diagnostics->cuda_optimizer_host_ms += elapsed_ms(optimize_start);
  }

  std::vector<std::string> residual_keys;
  residual_keys.reserve(static_cast<std::size_t>(target_count));
  for (int target = 0; target < target_count; ++target) {
    residual_keys.push_back(residual_key(
      base_target_keys[static_cast<std::size_t>(target)],
      selected_sp.data() +
        static_cast<std::size_t>(context->penalty_count) * target,
      context->penalty_count));
  }
  diagnostics->logical_residual_request_count += target_count;
  diagnostics->physical_residual_fit_count += target_count;
  diagnostics->matrix_h2d_bytes +=
    (Y.size() + selected_sp.size()) * sizeof(double);

  FixedSpBatchHostView batch;
  batch.Y = Y.data();
  batch.SP = selected_sp.data();
  batch.n = context->n;
  batch.null_dim = context->coefficient_dim;
  batch.penalty_count = context->penalty_count;
  batch.target_count = target_count;
  batch.output_mask = FixedSpOutputResiduals;
  batch.planned_routes = planned_routes;
  batch.target_keys = residual_keys;

  FullCudaCiExactBatchRequest exact_request;
  exact_request.expected_prepared_s_key_sha256 = context->prepared_key;
  exact_request.component_capacity = target_count;
  exact_request.pairs.reserve(task_indices.size());
  for (std::size_t index = 0; index < task_indices.size(); ++index) {
    const LayerCiTask& task =
      plan.tasks[static_cast<std::size_t>(task_indices[index])];
    FullCudaCiExactBatchPairRequest pair;
    pair.logical_sequence_id = logical_sequence_ids[index];
    pair.left_target_index = target_position.at(task.orientation_x);
    pair.right_target_index = target_position.at(task.orientation_y);
    pair.alpha = kQualifiedAlpha;
    exact_request.pairs.push_back(pair);
  }
  exact_request.request_identity_sha256 =
    full_cuda_ci_exact_batch_request_identity(exact_request, residual_keys);
  FullCudaCiExactBatchResult exact = run_full_cuda_ci_phase35_exact_batch(
    context->fixed_sp, batch, exact_request);
  accumulate_exact_diagnostics(exact.diagnostics, diagnostics);
  require(exact.records.size() == task_indices.size(),
          "exact CUDA dCov result count changed");

  GroupResult output;
  output.p_values.resize(task_indices.size());
  std::vector<std::size_t> guarded;
  for (std::size_t index = 0; index < exact.records.size(); ++index) {
    exact.records[index].optimizer_status = direct ?
      "DIRECT_SHIFT_INVARIANT" : "LIVE_CUDA_GCV_ACCEPTED";
    const double p_value = exact.records[index].p_value;
    require(std::isfinite(p_value) && p_value >= 0.0 && p_value <= 1.0,
            "exact CUDA dCov returned a non-finite p-value");
    output.p_values[index] = p_value;
    if (p_value >= kGuardLower && p_value <= kGuardUpper) {
      guarded.push_back(index);
    }
  }

  if (!guarded.empty()) {
    FullCudaCiLegacyEigBatchRequest refinement_request;
    refinement_request.expected_prepared_s_key_sha256 =
      context->prepared_key;
    refinement_request.component_capacity = target_count;
    refinement_request.num_col = num_col;
    refinement_request.pairs.reserve(guarded.size());
    for (std::size_t index : guarded) {
      refinement_request.pairs.push_back(exact_request.pairs[index]);
    }
    refinement_request.request_identity_sha256 =
      full_cuda_ci_legacy_eig_batch_request_identity(
        refinement_request, residual_keys);
    FullCudaCiLegacyEigBatchResult refinement =
      run_full_cuda_ci_phase35_legacy_eig_batch(
        context->fixed_sp, batch, refinement_request);
    accumulate_refinement_diagnostics(refinement.diagnostics, diagnostics);
    require(refinement.records.size() == guarded.size(),
            "guarded CUDA dCov result count changed");
    for (std::size_t index = 0; index < guarded.size(); ++index) {
      const double p_value = refinement.records[index].p_value;
      require(std::isfinite(p_value) && p_value >= 0.0 && p_value <= 1.0,
              "guarded CUDA dCov returned a non-finite p-value");
      output.p_values[guarded[index]] = p_value;
    }
    diagnostics->guarded_pair_count += static_cast<int>(guarded.size());
  }
  return output;
}

}  // namespace

Rcpp::List full_cuda_ci_one_call_skeleton(
    const Rcpp::NumericMatrix& data,
    double alpha,
    int max_conditioning_size,
    double index,
    int num_col,
    const std::string& trace_level,
    bool compatible_cuda_strict) {
  const auto run_started = std::chrono::steady_clock::now();
  const int n = data.nrow();
  const int p = data.ncol();
  require(compatible_cuda_strict,
          "compatible.cuda one-call route requires strict mode");
  require(n > 5, "compatible.cuda one-call route requires n > 5");
  require(p >= 2 && p <= 64,
          "compatible.cuda one-call route requires 2 <= p <= 64");
  require(std::isfinite(alpha) &&
            std::abs(alpha - kQualifiedAlpha) <= 1e-15,
          "compatible.cuda one-call alpha is outside the qualified value 0.1");
  require(index == 1.0,
          "compatible.cuda one-call index is outside the qualified value 1");
  require(num_col == 35 && num_col < n,
          "compatible.cuda one-call numCol is outside the qualified value 35");
  require(max_conditioning_size >= 0 && max_conditioning_size <= 7,
          "compatible.cuda one-call conditioning size is outside [0, 7]");
  require(trace_level == "summary" || trace_level == "logical" ||
            trace_level == "full" || trace_level == "none",
          "compatible.cuda one-call trace level is invalid");
  require(finite_matrix(data),
          "compatible.cuda one-call data must contain finite doubles");
  const bool collect_trace = trace_level == "logical" ||
    trace_level == "full";

  const std::string dataset = dataset_key(data);
  std::shared_ptr<CudaRuntimeContext> runtime = create_fixed_sp_runtime(0);
  FixedSpCapacities capacities;
  capacities.n = n;
  capacities.null_dim = 64;
  capacities.target_count = 64;
  capacities.penalty_count = 7;
  capacities.augmented_rows = n + 64;
  reserve_fixed_sp_runtime(runtime, capacities);

  OneCallDiagnostics diagnostics;
  NativeSetupCache setup_cache(data, dataset, runtime, &diagnostics);
  std::shared_ptr<NativeSetupContext> direct = build_direct_context(
    data, dataset, runtime);

  std::vector<int> adjacency(static_cast<std::size_t>(p) * p, 1);
  std::vector<double> pmax(
    static_cast<std::size_t>(p) * p,
    -std::numeric_limits<double>::infinity());
  std::vector<std::vector<std::vector<int>>> sepsets(
    p, std::vector<std::vector<int>>(p));
  for (int vertex = 0; vertex < p; ++vertex) {
    adjacency[static_cast<std::size_t>(vertex) * p + vertex] = 0;
    pmax[static_cast<std::size_t>(vertex) * p + vertex] = 1.0;
  }

  std::vector<int> n_edgetests;
  std::vector<int> level_values;
  std::vector<int> level_tasks_planned;
  std::vector<int> level_tests_replayed;
  std::vector<int> level_ignored;
  std::vector<int> level_deletions;
  std::vector<int> level_rounds;
  std::vector<double> level_elapsed_ms;

  std::vector<int> trace_id, trace_level_value, trace_task_index,
    trace_edge_x, trace_edge_y, trace_x, trace_y,
    trace_conditioning_size;
  std::vector<std::string> trace_s_key;
  std::vector<double> trace_p_used;
  std::vector<int> trace_deleted;

  std::uint64_t next_request_sequence_id = 1U;
  int next_logical_id = 1;
  int total_tasks_planned = 0;
  int total_tests_replayed = 0;
  int total_ignored = 0;
  int total_deletions = 0;

  try {
    for (int level = 0; level <= max_conditioning_size; ++level) {
      const auto level_started = std::chrono::steady_clock::now();
      const LayerPlan plan = make_layer_plan(adjacency, p, level);
      const int task_count = static_cast<int>(plan.tasks.size());
      std::vector<unsigned char> replayed(
        static_cast<std::size_t>(task_count), 0U);
      std::vector<unsigned char> edge_done(
        static_cast<std::size_t>(p) * p, 0U);
      std::vector<unsigned char> delete_edges(
        static_cast<std::size_t>(p) * p, 0U);
      std::vector<unsigned char> consumed(
        static_cast<std::size_t>(task_count), 0U);
      std::vector<double> task_p_values(
        static_cast<std::size_t>(task_count), NA_REAL);
      int remaining = task_count;
      int tests_replayed = 0;
      int ignored = 0;
      int deletions = 0;
      int rounds = 0;

      while (remaining > 0) {
        std::vector<int> selected;
        std::vector<unsigned char> pending_edges(
          static_cast<std::size_t>(p) * p, 0U);
        for (int task_index = 0; task_index < task_count; ++task_index) {
          if (replayed[static_cast<std::size_t>(task_index)] != 0U) continue;
          const LayerCiTask& task =
            plan.tasks[static_cast<std::size_t>(task_index)];
          const std::size_t edge = static_cast<std::size_t>(task.edge_key);
          const bool no_longer_reachable = edge_done[edge] != 0U ||
            adjacency[static_cast<std::size_t>(task.edge_x) * p +
                      task.edge_y] == 0;
          if (no_longer_reachable) {
            replayed[static_cast<std::size_t>(task_index)] = 1U;
            --remaining;
            ++ignored;
            continue;
          }
          if (pending_edges[edge] != 0U) continue;
          pending_edges[edge] = 1U;
          selected.push_back(task_index);
        }
        require(!selected.empty() || remaining == 0,
                "compatible.cuda round scheduler made no progress");
        if (selected.empty()) break;
        ++rounds;

        std::vector<std::uint64_t> selected_ids(selected.size());
        for (std::size_t index_value = 0;
             index_value < selected.size(); ++index_value) {
          selected_ids[index_value] = next_request_sequence_id++;
        }
        std::map<std::string, std::vector<std::size_t>> grouped_positions;
        for (std::size_t position = 0; position < selected.size(); ++position) {
          const LayerCiTask& task =
            plan.tasks[static_cast<std::size_t>(selected[position])];
          grouped_positions[conditioning_key(task.conditioning_set)]
            .push_back(position);
        }
        std::vector<double> selected_pvalues(selected.size(), NA_REAL);
        for (const auto& group : grouped_positions) {
          std::vector<int> group_tasks;
          std::vector<std::uint64_t> group_ids;
          group_tasks.reserve(group.second.size());
          group_ids.reserve(group.second.size());
          for (std::size_t position : group.second) {
            group_tasks.push_back(selected[position]);
            group_ids.push_back(selected_ids[position]);
          }
          const LayerCiTask& representative =
            plan.tasks[static_cast<std::size_t>(group_tasks.front())];
          const bool is_direct = representative.conditioning_set.empty();
          std::shared_ptr<NativeSetupContext> context = is_direct ?
            direct : setup_cache.get(representative.conditioning_set);
          const GroupResult result = execute_group(
            data, context, plan, group_tasks, group_ids, num_col,
            is_direct, &diagnostics);
          require(result.p_values.size() == group.second.size(),
                  "compatible.cuda group result alignment changed");
          for (std::size_t index_value = 0;
               index_value < group.second.size(); ++index_value) {
            selected_pvalues[group.second[index_value]] =
              result.p_values[index_value];
          }
        }

        for (std::size_t position = 0; position < selected.size(); ++position) {
          const int task_index = selected[position];
          const LayerCiTask& task =
            plan.tasks[static_cast<std::size_t>(task_index)];
          const double p_value = selected_pvalues[position];
          require(std::isfinite(p_value),
                  "compatible.cuda replay received a non-finite p-value");
          replayed[static_cast<std::size_t>(task_index)] = 1U;
          consumed[static_cast<std::size_t>(task_index)] = 1U;
          task_p_values[static_cast<std::size_t>(task_index)] = p_value;
          --remaining;
          const std::size_t forward =
            static_cast<std::size_t>(task.edge_x) * p + task.edge_y;
          const std::size_t reverse =
            static_cast<std::size_t>(task.edge_y) * p + task.edge_x;
          const bool deleted = p_value >= alpha;
          if (deleted) {
            delete_edges[forward] = 1U;
            delete_edges[reverse] = 1U;
            edge_done[static_cast<std::size_t>(task.edge_key)] = 1U;
          }
        }
      }

      // Physical CUDA rounds preserve per-edge reachability. Replay their
      // compact results in the canonical layer-plan order used by the oracle.
      for (int task_index = 0; task_index < task_count; ++task_index) {
        if (consumed[static_cast<std::size_t>(task_index)] == 0U) continue;
        const LayerCiTask& task =
          plan.tasks[static_cast<std::size_t>(task_index)];
        const double p_value =
          task_p_values[static_cast<std::size_t>(task_index)];
        require(std::isfinite(p_value),
                "compatible.cuda canonical replay received a non-finite p-value");
        ++tests_replayed;
        const std::size_t forward =
          static_cast<std::size_t>(task.edge_x) * p + task.edge_y;
        const std::size_t reverse =
          static_cast<std::size_t>(task.edge_y) * p + task.edge_x;
        if (p_value > pmax[forward]) {
          pmax[forward] = p_value;
          pmax[reverse] = p_value;
        }
        const bool deleted = p_value >= alpha;
        if (deleted) {
          ++deletions;
          sepsets[task.edge_x][task.edge_y] = task.conditioning_set;
          sepsets[task.edge_y][task.edge_x] = task.conditioning_set;
        }
        if (collect_trace) {
          trace_id.push_back(next_logical_id++);
          trace_level_value.push_back(level);
          trace_task_index.push_back(task_index + 1);
          trace_edge_x.push_back(task.edge_x + 1);
          trace_edge_y.push_back(task.edge_y + 1);
          trace_x.push_back(task.orientation_x + 1);
          trace_y.push_back(task.orientation_y + 1);
          trace_s_key.push_back(conditioning_key(task.conditioning_set));
          trace_conditioning_size.push_back(
            static_cast<int>(task.conditioning_set.size()));
          trace_p_used.push_back(p_value);
          trace_deleted.push_back(deleted ? 1 : 0);
        }
      }

      for (std::size_t index_value = 0;
           index_value < adjacency.size(); ++index_value) {
        if (delete_edges[index_value] != 0U) adjacency[index_value] = 0;
      }
      total_tasks_planned += task_count;
      total_tests_replayed += tests_replayed;
      total_ignored += ignored;
      total_deletions += deletions;
      n_edgetests.push_back(tests_replayed);
      level_values.push_back(level);
      level_tasks_planned.push_back(task_count);
      level_tests_replayed.push_back(tests_replayed);
      level_ignored.push_back(ignored);
      level_deletions.push_back(deletions);
      level_rounds.push_back(rounds);
      level_elapsed_ms.push_back(elapsed_ms(level_started));
      if (level > 0 && deletions == 0) break;
    }
  } catch (...) {
    close_context(direct);
    setup_cache.clear();
    free_fixed_sp_runtime(&runtime);
    throw;
  }

  close_context(direct);
  setup_cache.clear();
  const FixedSpRuntimeInfo runtime_info = fixed_sp_runtime_info(runtime);
  free_fixed_sp_runtime(&runtime);

  Rcpp::LogicalVector deleted_trace(trace_deleted.size());
  for (std::size_t index_value = 0;
       index_value < trace_deleted.size(); ++index_value) {
    deleted_trace[index_value] = trace_deleted[index_value] != 0;
  }
  Rcpp::DataFrame task_rows = Rcpp::DataFrame::create(
    Rcpp::Named("canonical_test_order_id") = Rcpp::wrap(trace_id),
    Rcpp::Named("level") = Rcpp::wrap(trace_level_value),
    Rcpp::Named("task_index") = Rcpp::wrap(trace_task_index),
    Rcpp::Named("edge_x") = Rcpp::wrap(trace_edge_x),
    Rcpp::Named("edge_y") = Rcpp::wrap(trace_edge_y),
    Rcpp::Named("x") = Rcpp::wrap(trace_x),
    Rcpp::Named("y") = Rcpp::wrap(trace_y),
    Rcpp::Named("S_key") = Rcpp::wrap(trace_s_key),
    Rcpp::Named("conditioning_size") =
      Rcpp::wrap(trace_conditioning_size),
    Rcpp::Named("p_used") = Rcpp::wrap(trace_p_used),
    Rcpp::Named("native_edge_deleted") = deleted_trace,
    Rcpp::Named("native_edge_ignored") =
      Rcpp::LogicalVector(trace_deleted.size(), false),
    Rcpp::Named("stringsAsFactors") = false);
  Rcpp::DataFrame level_rows = Rcpp::DataFrame::create(
    Rcpp::Named("level") = Rcpp::wrap(level_values),
    Rcpp::Named("tasks_planned") = Rcpp::wrap(level_tasks_planned),
    Rcpp::Named("tests_replayed") = Rcpp::wrap(level_tests_replayed),
    Rcpp::Named("tasks_ignored_after_delete") = Rcpp::wrap(level_ignored),
    Rcpp::Named("deletions") = Rcpp::wrap(level_deletions),
    Rcpp::Named("rounds") = Rcpp::wrap(level_rounds),
    Rcpp::Named("elapsed_ms") = Rcpp::wrap(level_elapsed_ms),
    Rcpp::Named("stringsAsFactors") = false);

  const int physical_tests_evaluated = diagnostics.cuda_dcov_pair_count;
  const bool authority_clean =
    diagnostics.cpu_dcov_component_count == 0 &&
    diagnostics.cpu_dcov_eigen_or_lowrank_count == 0 &&
    diagnostics.cpu_dcov_pair_stat_count == 0 &&
    diagnostics.cpu_gamma_pvalue_count == 0 &&
    diagnostics.cpu_spectra_count == 0 &&
    diagnostics.residual_d2h_bytes == 0U &&
    diagnostics.component_d2h_bytes == 0U;
  require(authority_clean,
          "compatible.cuda one-call authority counters failed closed");

  return Rcpp::List::create(
    Rcpp::Named("adjacency") = adjacency_matrix(adjacency, p),
    Rcpp::Named("sepsets") = sepset_list(sepsets),
    Rcpp::Named("pMax") = pmax_matrix(pmax, p),
    Rcpp::Named("n.edgetests") = Rcpp::wrap(n_edgetests),
    Rcpp::Named("tasks") = task_rows,
    Rcpp::Named("levels") = level_rows,
    Rcpp::Named("summary") = Rcpp::List::create(
      Rcpp::Named("run_status") = "ok",
      Rcpp::Named("entrypoint") =
        "compatible-cuda-full-skeleton-native-v1",
      Rcpp::Named("compatible_cuda_route") = "compatible.cuda",
      Rcpp::Named("compatible_cuda_strict") = true,
      Rcpp::Named("native_call_count") = 1,
      Rcpp::Named("dataset_key") = dataset,
      Rcpp::Named("p") = p,
      Rcpp::Named("n") = n,
      Rcpp::Named("alpha") = alpha,
      Rcpp::Named("index") = index,
      Rcpp::Named("numCol") = num_col,
      Rcpp::Named("levels") = static_cast<int>(level_values.size()),
      Rcpp::Named("tasks_planned") = total_tasks_planned,
      Rcpp::Named("logical_tests_consumed") = total_tests_replayed,
      Rcpp::Named("physical_tests_evaluated") = physical_tests_evaluated,
      Rcpp::Named("speculative_tests_ignored") = 0,
      Rcpp::Named("tasks_ignored_after_delete") = total_ignored,
      Rcpp::Named("deletions") = total_deletions,
      Rcpp::Named("r_callback_count") = 0,
      Rcpp::Named("legacy_mgcv_fit_count") = 0,
      Rcpp::Named("legacy_mgcv_setup_count") = 0,
      Rcpp::Named("cpu_residual_solve_count") = 0,
      Rcpp::Named("cpu_dcov_component_count") =
        diagnostics.cpu_dcov_component_count,
      Rcpp::Named("cpu_dcov_eigen_or_lowrank_count") =
        diagnostics.cpu_dcov_eigen_or_lowrank_count,
      Rcpp::Named("cpu_dcov_pair_stat_count") =
        diagnostics.cpu_dcov_pair_stat_count,
      Rcpp::Named("cpu_gamma_pvalue_count") =
        diagnostics.cpu_gamma_pvalue_count,
      Rcpp::Named("cpu_spectra_count") = diagnostics.cpu_spectra_count,
      Rcpp::Named("cuda_single_penalty_target_count") =
        diagnostics.cuda_single_penalty_target_count,
      Rcpp::Named("cuda_multi_penalty_target_count") =
        diagnostics.cuda_multi_penalty_target_count,
      Rcpp::Named("cuda_residual_batch_count") =
        diagnostics.cuda_residual_batch_count,
      Rcpp::Named("logical_residual_requests") =
        diagnostics.logical_residual_request_count,
      Rcpp::Named("physical_residual_fits") =
        diagnostics.physical_residual_fit_count,
      Rcpp::Named("cuda_dcov_component_count") =
        diagnostics.cuda_dcov_component_count,
      Rcpp::Named("cuda_dcov_pair_count") = diagnostics.cuda_dcov_pair_count,
      Rcpp::Named("cuda_gamma_pvalue_count") =
        diagnostics.cuda_gamma_pvalue_count,
      Rcpp::Named("guarded_pair_count") = diagnostics.guarded_pair_count,
      Rcpp::Named("native_setup_count") = diagnostics.native_setup_count,
      Rcpp::Named("native_setup_cache_capacity") = kPreparedCacheCapacity,
      Rcpp::Named("native_setup_cache_request_count") =
        diagnostics.native_setup_cache_request_count,
      Rcpp::Named("native_setup_cache_hit_count") =
        diagnostics.native_setup_cache_hit_count,
      Rcpp::Named("native_setup_cache_miss_count") =
        diagnostics.native_setup_cache_miss_count,
      Rcpp::Named("native_setup_cache_eviction_count") =
        diagnostics.native_setup_cache_eviction_count,
      Rcpp::Named("component_cache_capacity") =
        kReferenceComponentCapacity,
      Rcpp::Named("component_cache_request_count") =
        diagnostics.component_cache_request_count,
      Rcpp::Named("component_cache_hit_count") =
        diagnostics.component_cache_hit_count,
      Rcpp::Named("component_cache_miss_count") =
        diagnostics.component_cache_miss_count,
      Rcpp::Named("component_cache_eviction_count") =
        diagnostics.component_cache_eviction_count,
      Rcpp::Named("host_synchronization_count") =
        diagnostics.host_synchronization_count,
      Rcpp::Named("matrix_h2d_bytes") =
        static_cast<double>(diagnostics.matrix_h2d_bytes),
      Rcpp::Named("residual_h2d_bytes") = 0,
      Rcpp::Named("residual_d2h_bytes") =
        static_cast<double>(diagnostics.residual_d2h_bytes),
      Rcpp::Named("component_d2h_bytes") =
        static_cast<double>(diagnostics.component_d2h_bytes),
      Rcpp::Named("compact_result_d2h_bytes") =
        static_cast<double>(diagnostics.compact_result_d2h_bytes),
      Rcpp::Named("unknown_fallback_count") = 0,
      Rcpp::Named("approximate_backend_count") = 0,
      Rcpp::Named("native_setup_ms") = diagnostics.native_setup_ms,
      Rcpp::Named("cuda_optimizer_host_ms") =
        diagnostics.cuda_optimizer_host_ms,
      Rcpp::Named("cuda_dcov_host_ms") = diagnostics.cuda_dcov_host_ms,
      Rcpp::Named("runtime_workspace_bytes") =
        static_cast<double>(runtime_info.workspace_bytes),
      Rcpp::Named("elapsed_sec") = elapsed_ms(run_started) / 1000.0,
      Rcpp::Named("authority_gate_pass") = authority_clean));
}

}  // namespace fastkpc
