#include "full_cuda_ci_one_call.hpp"

#include "full_cuda_ci_contract.hpp"
#include "full_cuda_ci_native_setup.hpp"
#include "skeleton_task_scheduler.hpp"
#include "cuda/full_cuda_ci_method_batch.hpp"
#include "cuda/full_cuda_ci_vertical.hpp"
#include "cuda/mgcv_fixed_sp_runtime.hpp"
#include "cuda/mgcv_multi_penalty_gcv_capacity.hpp"
#include "cuda/mgcv_single_penalty_gcv.hpp"

#include <RcppEigen.h>
#include <R_ext/Random.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <future>
#include <iomanip>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fastkpc {
namespace {

using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::ColMajor>;

constexpr double kQualifiedAlpha = 0.1;
constexpr double kGuardLower = 0.05;
constexpr double kGuardUpper = 0.15;
constexpr int kSinglePenaltyPreparedCacheCapacity = 2048;
constexpr int kMultiPenaltyPreparedCacheCapacity = 8192;
constexpr int kPreparedCacheCapacity =
  kSinglePenaltyPreparedCacheCapacity + kMultiPenaltyPreparedCacheCapacity;
constexpr int kHostPreparedCacheCapacity = 16384;
constexpr std::size_t kSinglePenaltyLevelPrefillGroupWindow = 64U;
constexpr int kReferenceComponentCapacity = 47;
constexpr int kDefaultCompactResultCacheCapacity = 262144;
constexpr int kMaximumCompactResultCacheCapacity = 1048576;
constexpr int kDefaultTargetStateCacheCapacity = 131072;
constexpr int kMaximumTargetStateCacheCapacity = 524288;
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

int decomposition_trace_capacity_from_environment() {
  const char* raw = std::getenv(
    "FASTKPC_PHASE10_DECOMPOSITION_TRACE_CAPACITY");
  if (raw == nullptr || raw[0] == '\0' || std::string(raw) == "0") return 0;
  std::size_t parsed = 0;
  int capacity = 0;
  try {
    capacity = std::stoi(raw, &parsed);
  } catch (...) {
    throw std::runtime_error(
      "Phase 10 decomposition trace capacity is not an integer");
  }
  require(parsed == std::string(raw).size() &&
            capacity >= 64 && capacity <= 4096,
          "Phase 10 decomposition trace capacity is outside [64, 4096]");
  return capacity;
}

bool fixed_sp_root_cache_policy_enabled() {
  const char* raw = std::getenv("FASTKPC_PHASE10_FIXED_SP_ROOT_CACHE");
  if (raw == nullptr || std::string(raw).empty() ||
      std::string(raw) == "1") {
    return true;
  }
  if (std::string(raw) == "0") return false;
  throw std::runtime_error(
    "FASTKPC_PHASE10_FIXED_SP_ROOT_CACHE must be 0 or 1");
}

std::string strict_hsic_gamma_residual_route_policy(
    const std::string& ci_method) {
  const char* raw = std::getenv(
    "FASTKPC_STRICT_HSIC_GAMMA_RESIDUAL_ROUTE");
  if (raw == nullptr || raw[0] == '\0') {
    return ci_method == "hsic.gamma" ? "qr-through-2" : "stable-svd";
  }
  const std::string value(raw);
  if (value == "stable-svd" || value == "qr-through-2") {
    return value;
  }
  throw std::runtime_error(
    "FASTKPC_STRICT_HSIC_GAMMA_RESIDUAL_ROUTE must be stable-svd or "
    "qr-through-2");
}

std::string strict_permutation_residual_route_policy(
    const std::string& ci_method) {
  const char* raw = std::getenv(
    "FASTKPC_STRICT_PERMUTATION_RESIDUAL_ROUTE");
  if (raw == nullptr || raw[0] == '\0') {
    return (ci_method == "dcc.perm" || ci_method == "hsic.perm") ?
      "qr-through-2" : "stable-svd";
  }
  const std::string value(raw);
  if (value != "stable-svd" && value != "qr-through-2") {
    throw std::runtime_error(
      "FASTKPC_STRICT_PERMUTATION_RESIDUAL_ROUTE must be stable-svd or "
      "qr-through-2");
  }
  if (ci_method != "dcc.perm" && ci_method != "hsic.perm" &&
      value != "stable-svd") {
    throw std::runtime_error(
      "strict permutation residual route override requires a permutation "
      "CI method");
  }
  return value;
}

bool strict_hsic_perm_inline_r_index_policy_enabled(
    const std::string& ci_method) {
  const char* raw = std::getenv(
    "FASTKPC_STRICT_HSIC_PERM_INLINE_R_UNIF_INDEX");
  if (raw == nullptr || raw[0] == '\0') {
    return ci_method == "hsic.perm";
  }
  if (std::string(raw) == "0") {
    return false;
  }
  if (std::string(raw) != "1") {
    throw std::runtime_error(
      "FASTKPC_STRICT_HSIC_PERM_INLINE_R_UNIF_INDEX must be 0 or 1");
  }
  if (ci_method != "hsic.perm") {
    throw std::runtime_error(
      "inline R uniform-index generation requires hsic.perm");
  }
  return true;
}

bool setup_optimizer_pipeline_policy_enabled() {
  const char* raw = std::getenv(
    "FASTKPC_PHASE10_SETUP_OPTIMIZER_PIPELINE");
  if (raw == nullptr || raw[0] == '\0' || std::string(raw) == "0") {
    return false;
  }
  if (std::string(raw) == "1") return true;
  throw std::runtime_error(
    "FASTKPC_PHASE10_SETUP_OPTIMIZER_PIPELINE must be 0 or 1");
}

int setup_optimizer_pipeline_producer_delay_us() {
  const char* raw = std::getenv(
    "FASTKPC_PHASE10_SETUP_OPTIMIZER_PRODUCER_DELAY_US");
  if (raw == nullptr || raw[0] == '\0') return 0;
  std::size_t parsed = 0;
  int delay = 0;
  try {
    delay = std::stoi(raw, &parsed);
  } catch (...) {
    throw std::runtime_error(
      "setup/optimizer producer delay is not an integer");
  }
  require(parsed == std::string(raw).size() &&
            delay >= 0 && delay <= 100000,
          "setup/optimizer producer delay is outside [0,100000] us");
  return delay;
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

std::string compact_result_key(const std::string& dataset,
                               int num_col,
                               int level,
                               int task_index,
                               const LayerCiTask& task,
                               const FullCudaCiOneCallMethodOptions& method) {
  std::ostringstream output;
  output << "schema=full-cuda-ci-compact-result-key-v2\n"
         << "backend=compatible-cuda-full-skeleton-native-v1\n"
         << "dataset=" << dataset << "\n"
         << "alpha=0.1\nindex=1\n"
         << "numCol=" << num_col << "\n"
         << "ci_method=" << method.ci_method << "\n"
         << "hsic_sig_bits=" << std::hex << std::setfill('0');
  std::uint64_t sig_bits = 0;
  std::memcpy(&sig_bits, &method.hsic_sig, sizeof(sig_bits));
  output << std::setw(16) << sig_bits << std::dec << "\n"
         << "permutation_replicates=" << method.permutation_replicates << "\n"
         << "permutation_include_observed="
         << (method.permutation_include_observed ? 1 : 0) << "\n"
         << "permutation_has_seed="
         << (method.permutation_has_seed ? 1 : 0) << "\n"
         << "permutation_seed=" << method.permutation_seed << "\n"
         << "level=" << level << "\n"
         << "task_index=" << task_index + 1 << "\n"
         << "edge=" << task.edge_x + 1 << "|" << task.edge_y + 1 << "\n"
         << "orientation=" << task.orientation_x + 1 << "|"
         << task.orientation_y + 1 << "\n"
         << "S=" << conditioning_key(task.conditioning_set) << "\n";
  return output.str();
}

bool is_permutation_method(const std::string& method) {
  return method == "dcc.perm" || method == "hsic.perm";
}

unsigned int task_permutation_seed(
    const FullCudaCiOneCallMethodOptions& method,
    const LayerCiTask& task) {
  constexpr std::uint64_t modulus = 2147483647ULL;
  std::ostringstream payload;
  payload << "schema=full-cuda-ci-task-permutation-seed-v1\n"
          << "method=" << method.ci_method << "\n"
          << "level=" << task.level << "\n"
          << "edge=" << task.edge_x + 1 << "|" << task.edge_y + 1 << "\n"
          << "orientation=" << task.orientation_x + 1 << "|"
          << task.orientation_y + 1 << "\n"
          << "S=" << conditioning_key(task.conditioning_set) << "\n";
  const std::string digest = full_cuda_ci_sha256_utf8(payload.str());
  require(digest.size() == 64U,
          "permutation task identity digest has invalid length");
  std::uint64_t prefix = 0U;
  for (std::size_t index = 0; index < 8U; ++index) {
    const char value = digest[index];
    const unsigned int nibble = value >= '0' && value <= '9' ?
      static_cast<unsigned int>(value - '0') :
      static_cast<unsigned int>(value - 'a' + 10);
    require(nibble < 16U,
            "permutation task identity digest is not lowercase hex");
    prefix = (prefix << 4U) | nibble;
  }
  const std::uint64_t seed =
    (static_cast<std::uint64_t>(method.permutation_seed) +
     prefix % modulus) % modulus;
  return static_cast<unsigned int>(seed);
}

bool pcalg_task_less(const LayerCiTask& left, const LayerCiTask& right) {
  if (left.orientation_x != right.orientation_x) {
    return left.orientation_x < right.orientation_x;
  }
  if (left.orientation_y != right.orientation_y) {
    return left.orientation_y < right.orientation_y;
  }
  return std::lexicographical_compare(
    left.conditioning_set.begin(), left.conditioning_set.end(),
    right.conditioning_set.begin(), right.conditioning_set.end());
}

}  // namespace

struct PermutationTableBuilder::Impl {
  std::vector<int> values;
  FullCudaCiSha256Builder digest;
  std::size_t committed = 0U;
  std::exception_ptr deferred_digest_error;
  bool active = false;
};

PermutationTableBuilder::PermutationTableBuilder() : impl_(new Impl()) {}
PermutationTableBuilder::~PermutationTableBuilder() = default;
PermutationTableBuilder::PermutationTableBuilder(
    PermutationTableBuilder&&) noexcept = default;
PermutationTableBuilder& PermutationTableBuilder::operator=(
    PermutationTableBuilder&&) noexcept = default;

void PermutationTableBuilder::reset(std::size_t size) {
  require(impl_ != nullptr, "permutation table builder has been moved");
  impl_->values.resize(size);
  impl_->committed = 0U;
  impl_->deferred_digest_error = std::exception_ptr();
  impl_->active = true;
  try {
    impl_->digest.reset();
  } catch (...) {
    impl_->deferred_digest_error = std::current_exception();
  }
}

std::size_t PermutationTableBuilder::capacity() const noexcept {
  return impl_ ? impl_->values.capacity() : 0U;
}

std::size_t PermutationTableBuilder::size() const noexcept {
  return impl_ ? impl_->values.size() : 0U;
}

void PermutationTableBuilder::append_row(
    const int* values,
    std::size_t size) {
  int* output = begin_row(size);
  std::copy_n(values, size, output);
  commit_row(output, size);
}

int* PermutationTableBuilder::begin_row(std::size_t size) {
  require(impl_ != nullptr && impl_->active &&
            impl_->committed <= impl_->values.size() &&
            size <= impl_->values.size() - impl_->committed,
          "permutation table builder row exceeds capacity");
  return impl_->values.data() + impl_->committed;
}

void PermutationTableBuilder::commit_row(
    const int* values,
    std::size_t size) {
  require(impl_ != nullptr && impl_->active &&
            impl_->committed <= impl_->values.size() &&
            values == impl_->values.data() + impl_->committed &&
            size <= impl_->values.size() - impl_->committed,
          "permutation table builder row identity changed");
  if (!impl_->deferred_digest_error) {
    try {
      impl_->digest.update(values, size * sizeof(int));
    } catch (...) {
      impl_->deferred_digest_error = std::current_exception();
    }
  }
  impl_->committed += size;
}

SealedPermutationTableHandle PermutationTableBuilder::seal() {
  require(impl_ != nullptr && impl_->active &&
            impl_->committed == impl_->values.size() &&
            !impl_->values.empty(),
          "permutation table builder is incomplete");
  impl_->active = false;
  if (impl_->deferred_digest_error) {
    std::rethrow_exception(impl_->deferred_digest_error);
  }
  SealedPermutationTableHandle handle;
  handle.values_ = std::move(impl_->values);
  handle.sha256_ = impl_->digest.finish();
  handle.sealed_ = true;
  return handle;
}

void PermutationTableBuilder::reclaim(
    SealedPermutationTableHandle&& handle) {
  require(impl_ != nullptr && !impl_->active,
          "permutation table builder is still active");
  require(handle.sealed_, "cannot reclaim an unsealed permutation table");
  impl_->values = std::move(handle.values_);
  impl_->committed = 0U;
  handle.sha256_.fill(0U);
  handle.sealed_ = false;
}

namespace {

struct LegacyRPermutationWorkspace {
  PermutationTableBuilder table;
  std::vector<int> scratch;
  std::vector<unsigned int> rejection_masks;
  int table_growth_count = 0;
  int scratch_growth_count = 0;
  std::size_t peak_table_values = 0U;
  bool inline_r_index_requested = false;
  bool inline_r_index_active = false;
  std::uint64_t inline_r_index_count = 0U;
  std::uint64_t inline_r_draw_count = 0U;
};

void make_legacy_r_permutation_table(
    const FullCudaCiOneCallMethodOptions& method,
    int n,
    std::size_t pair_count,
    LegacyRPermutationWorkspace* workspace) {
  require(is_permutation_method(method.ci_method) && n >= 2 &&
            method.permutation_replicates >= 1 && pair_count >= 1U &&
            workspace != nullptr,
          "legacy R permutation table request is malformed");
  const std::size_t replicate_count =
    static_cast<std::size_t>(method.permutation_replicates);
  require(pair_count <= std::numeric_limits<std::size_t>::max() /
            replicate_count &&
            pair_count * replicate_count <=
              std::numeric_limits<std::size_t>::max() /
                static_cast<std::size_t>(n),
          "legacy R permutation table size overflow");
  const std::size_t table_values =
    pair_count * replicate_count * static_cast<std::size_t>(n);
  if (workspace->table.capacity() < table_values) {
    workspace->table_growth_count += 1;
  }
  workspace->table.reset(table_values);
  workspace->peak_table_values = std::max(
    workspace->peak_table_values, table_values);
  if (workspace->scratch.capacity() < static_cast<std::size_t>(n)) {
    workspace->scratch_growth_count += 1;
  }
  workspace->scratch.resize(static_cast<std::size_t>(n));
  std::uint64_t inline_index_count = 0U;
  std::uint64_t inline_draw_count = 0U;

  for (std::size_t pair = 0; pair < pair_count; ++pair) {
    if (method.ci_method == "dcc.perm") {
      // energy::dCOVtest keeps one permutation vector per CI call and
      // mutates it across replicates using runif(0, remaining).
      std::vector<int>& permutation = workspace->scratch;
      std::iota(permutation.begin(), permutation.end(), 0);
      for (int replicate = 0; replicate < method.permutation_replicates;
           ++replicate) {
        int remaining = n;
        for (int index = 0; index < n - 1; ++index) {
          int picked = static_cast<int>(std::floor(
            R::unif_rand() * static_cast<double>(remaining)));
          if (picked < 0) picked = 0;
          if (picked >= remaining) picked = remaining - 1;
          --remaining;
          std::swap(permutation[static_cast<std::size_t>(picked)],
                    permutation[static_cast<std::size_t>(remaining)]);
        }
        workspace->table.append_row(
          permutation.data(), static_cast<std::size_t>(n));
      }
      continue;
    }

    // stats::sample.int(n, n, replace = FALSE) uses R_unif_index with a
    // shrinking pool under R's configured sample.kind contract.
    for (int replicate = 0; replicate < method.permutation_replicates;
         ++replicate) {
      std::vector<int>& pool = workspace->scratch;
      std::iota(pool.begin(), pool.end(), 0);
      int remaining = n;
      int* output = workspace->table.begin_row(
        static_cast<std::size_t>(n));
      for (int index = 0; index < n; ++index) {
        int picked = 0;
        if (workspace->inline_r_index_active) {
          const unsigned int mask = workspace->rejection_masks[
            static_cast<std::size_t>(remaining)];
          do {
            const int bits = static_cast<int>(std::floor(
              R::unif_rand() * 65536.0));
            inline_draw_count += 1U;
            picked = bits & static_cast<int>(mask);
          } while (picked >= remaining);
          inline_index_count += 1U;
        } else {
          picked = static_cast<int>(R_unif_index(
            static_cast<double>(remaining)));
        }
        if (picked < 0) picked = 0;
        if (picked >= remaining) picked = remaining - 1;
        output[index] = pool[static_cast<std::size_t>(picked)];
        --remaining;
        pool[static_cast<std::size_t>(picked)] =
          pool[static_cast<std::size_t>(remaining)];
      }
      workspace->table.commit_row(output, static_cast<std::size_t>(n));
    }
  }
  workspace->inline_r_index_count += inline_index_count;
  workspace->inline_r_draw_count += inline_draw_count;
}

void make_method_permutation_table(
    const FullCudaCiOneCallMethodOptions& method,
    int n,
    const LayerPlan& plan,
    const std::vector<int>& task_indices,
    LegacyRPermutationWorkspace* workspace) {
  if (!is_permutation_method(method.ci_method)) {
    return;
  }
  (void)plan;
  make_legacy_r_permutation_table(
    method, n, task_indices.size(), workspace);
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
  std::string dataset_key;
  std::string prepared_key;
  std::vector<int> conditioning_set;
  Rcpp::List setup;
  Rcpp::List geometry;
  int n = 0;
  int coefficient_dim = 0;
  int penalty_count = 0;
  int multi_penalty_target_capacity = 0;
  std::shared_ptr<PreparedSGpuHandle> fixed_sp;
  std::shared_ptr<MultiPenaltyGcvCapacityPrepared> multi_penalty;
  std::unique_ptr<SinglePenaltyGeometry> single_penalty;
};

struct DecompositionReuseAggregate {
  std::uint64_t batch_count = 0;
  std::uint64_t request_count = 0;
  std::uint64_t stored_count = 0;
  std::uint64_t overflow_count = 0;
  std::uint64_t unique_key_count = 0;
  std::uint64_t reuse_count = 0;
  std::uint64_t route_mismatch_count = 0;
  std::array<std::uint64_t,
             kMultiPenaltyGcvDecompositionTraceStageCount>
    stage_request_count{};
  std::array<std::uint64_t,
             kMultiPenaltyGcvDecompositionTraceStageCount>
    stage_unique_key_count{};
  std::array<std::uint64_t,
             kMultiPenaltyGcvDecompositionTraceStageCount>
    stage_reuse_count{};
  std::array<std::uint64_t,
             kMultiPenaltyGcvDecompositionTraceRouteCount>
    route_request_count{};
  std::array<std::uint64_t,
             kMultiPenaltyGcvDecompositionTraceRouteCount>
    route_unique_key_count{};
  std::array<std::uint64_t,
             kMultiPenaltyGcvDecompositionTraceRouteCount>
    route_reuse_count{};
  std::vector<std::uint64_t> group_size_histogram;
};

struct DecompositionIterationReuseRow {
  std::string prepared_key;
  int coefficient_dim = 0;
  int penalty_count = 0;
  int iteration = 0;
  bool stability_replay = false;
  std::uint64_t request_count = 0;
  std::uint64_t unique_key_count = 0;
  std::uint64_t reuse_count = 0;
};

struct PrefillBatchDiagnostic {
  int level = 0;
  std::string penalty_class;
  int window_id = 0;
  int conditioning_group_count = 0;
  int optimizer_setup_count = 0;
  int target_optimization_count = 0;
  int singleton_skipped_request_count = 0;
  int singleton_skipped_target_count = 0;
  double optimizer_host_ms = 0.0;
  double batch_wall_ms = 0.0;
  std::vector<std::string> optimized_target_keys;
};

struct OneCallDiagnostics {
  std::unordered_set<std::string> physical_prepared_keys;
  std::unordered_set<std::string> consumed_prepared_keys;
  std::unordered_set<std::string> unique_target_keys;
  std::unordered_set<std::string> unique_residual_keys;
  std::unordered_set<std::string> exact_residual_cohort_signatures;
  std::unordered_set<std::string> prefill_target_keys;
  std::vector<PrefillBatchDiagnostic> prefill_batches;
  int native_setup_count = 0;
  int native_setup_cache_request_count = 0;
  int native_setup_cache_hit_count = 0;
  int native_setup_cache_miss_count = 0;
  int native_setup_cache_eviction_count = 0;
  int native_setup_device_cache_hit_count = 0;
  int native_setup_host_cache_hit_count = 0;
  int native_setup_host_cache_eviction_count = 0;
  int native_setup_host_cache_peak_entries = 0;
  int native_setup_device_rehydrate_count = 0;
  int native_setup_univariate_primitive_request_count = 0;
  int native_setup_univariate_primitive_hit_count = 0;
  int native_setup_univariate_primitive_build_count = 0;
  int native_setup_univariate_primitive_cache_capacity = 0;
  int native_setup_univariate_primitive_cache_peak_entries = 0;
  int cuda_single_penalty_target_count = 0;
  int cuda_multi_penalty_target_count = 0;
  int cuda_single_penalty_optimizer_setup_count = 0;
  int cuda_single_penalty_optimizer_call_count = 0;
  int cuda_multi_penalty_optimizer_setup_count = 0;
  int cuda_multi_penalty_optimizer_call_count = 0;
  int cuda_multi_penalty_prepared_build_count = 0;
  int cuda_multi_penalty_prepared_release_count = 0;
  int cuda_multi_penalty_prepared_target_capacity_sum = 0;
  int cuda_multi_penalty_prepared_target_capacity_peak = 0;
  int cuda_multi_penalty_decomposition_trace_capacity_per_target = 0;
  DecompositionReuseAggregate cuda_multi_penalty_decomposition_reuse;
  std::map<std::pair<int, int>, DecompositionReuseAggregate>
    cuda_multi_penalty_decomposition_reuse_by_shape;
  std::vector<DecompositionIterationReuseRow>
    cuda_multi_penalty_decomposition_iteration_reuse;
  std::uint64_t cuda_multi_penalty_optimizer_iteration_sum = 0;
  int cuda_multi_penalty_optimizer_iteration_max = 0;
  std::uint64_t cuda_multi_penalty_score_call_sum = 0;
  std::uint64_t cuda_multi_penalty_objective_call_sum = 0;
  std::uint64_t cuda_multi_penalty_step_halving_sum = 0;
  std::uint64_t cuda_multi_penalty_newton_trial_sum = 0;
  std::uint64_t cuda_multi_penalty_steepest_descent_trial_sum = 0;
  std::uint64_t cuda_multi_penalty_boundary_probe_sum = 0;
  std::uint64_t cuda_multi_penalty_complete_evaluation_count = 0;
  std::uint64_t cuda_multi_penalty_score_only_evaluation_count = 0;
  std::uint64_t cuda_multi_penalty_guarded_qr_evaluation_count = 0;
  std::uint64_t cuda_multi_penalty_stable_svd_evaluation_count = 0;
  std::uint64_t cuda_multi_penalty_selected_evaluation_reuse_count = 0;
  std::uint64_t cuda_multi_penalty_stability_replay_target_count = 0;
  std::uint64_t cuda_multi_penalty_stability_replay_selected_count = 0;
  std::uint64_t cuda_multi_penalty_terminal_confirmation_count = 0;
  std::uint64_t cuda_multi_penalty_hessian_eigensolver_count = 0;
  std::uint64_t cuda_multi_penalty_penalty_factor_cycles = 0;
  std::uint64_t cuda_multi_penalty_qr_svd_cycles = 0;
  std::uint64_t cuda_multi_penalty_qr_bidiagonal_reduction_cycles = 0;
  std::uint64_t cuda_multi_penalty_qr_factorization_cycles = 0;
  std::uint64_t cuda_multi_penalty_q_generation_cycles = 0;
  std::uint64_t cuda_multi_penalty_qr_guard_cycles = 0;
  std::uint64_t cuda_multi_penalty_stable_bidiagonal_reduction_cycles = 0;
  std::uint64_t cuda_multi_penalty_bidiagonal_svd_cycles = 0;
  std::uint64_t cuda_multi_penalty_svd_vector_postback_cycles = 0;
  std::uint64_t cuda_multi_penalty_left_vector_product_cycles = 0;
  std::uint64_t cuda_multi_penalty_score_construction_cycles = 0;
  std::uint64_t cuda_multi_penalty_derivative_hessian_cycles = 0;
  std::uint64_t cuda_multi_penalty_stability_replay_discarded_cycles = 0;
  std::uint64_t cuda_multi_penalty_terminal_confirmation_cycles = 0;
  int cuda_optimizer_kernel_launch_count = 0;
  int cuda_optimizer_host_boundary_count = 0;
  int frontier_optimizer_boundary_count = 0;
  int frontier_live_target_optimization_count = 0;
  int frontier_physical_target_optimization_count = 0;
  int singleton_padding_batch_count = 0;
  int singleton_padding_target_count = 0;
  int cuda_residual_batch_count = 0;
  int cuda_exact_screen_residual_batch_count = 0;
  int cuda_exact_screen_residual_target_count = 0;
  int cuda_exact_screen_component_count = 0;
  int cuda_exact_screen_pair_count = 0;
  int cuda_guard_refinement_residual_batch_count = 0;
  int cuda_guard_refinement_residual_target_count = 0;
  int cuda_guard_refinement_component_count = 0;
  int cuda_guard_refinement_pair_count = 0;
  int exact_residual_all_miss_batch_count = 0;
  int exact_residual_all_miss_target_count = 0;
  int exact_residual_mixed_batch_count = 0;
  int exact_residual_mixed_target_count = 0;
  int exact_residual_all_hit_new_cohort_batch_count = 0;
  int exact_residual_all_hit_new_cohort_target_count = 0;
  int exact_residual_all_hit_repeated_cohort_batch_count = 0;
  int exact_residual_all_hit_repeated_cohort_target_count = 0;
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
  int method_component_cache_request_count = 0;
  int method_component_cache_lookup_count = 0;
  int method_component_cache_hit_count = 0;
  int method_component_cache_miss_count = 0;
  int method_component_cache_insert_count = 0;
  int method_component_cache_eviction_count = 0;
  std::size_t method_component_cache_capacity_entries = 0U;
  std::size_t method_component_cache_device_bytes = 0U;
  std::size_t method_component_cache_gather_d2d_bytes = 0U;
  std::size_t method_component_cache_store_d2d_bytes = 0U;
  int method_residual_cache_lookup_count = 0;
  int method_residual_cache_hit_count = 0;
  int method_residual_cache_insert_count = 0;
  int method_residual_cache_eviction_count = 0;
  int method_residual_cache_all_hit_batch_count = 0;
  int method_residual_cache_bypassed_target_count = 0;
  std::size_t method_residual_cache_capacity_entries = 0U;
  std::size_t method_residual_cache_device_bytes = 0U;
  std::size_t method_residual_cache_gather_d2d_bytes = 0U;
  int method_execution_context_call_count = 0;
  int method_execution_context_reuse_count = 0;
  int method_execution_context_buffer_growth_count = 0;
  std::size_t method_execution_context_peak_device_bytes = 0U;
  int method_permutation_table_build_count = 0;
  int method_permutation_table_growth_count = 0;
  int method_permutation_scratch_growth_count = 0;
  std::size_t method_permutation_table_value_count = 0U;
  std::size_t method_permutation_peak_table_values = 0U;
  double method_permutation_table_host_ms = 0.0;
  double method_request_identity_build_host_ms = 0.0;
  double method_request_identity_validation_host_ms = 0.0;
  int method_permutation_payload_validation_scan_count = 0;
  std::size_t method_permutation_payload_validation_scan_bytes = 0U;
  int method_permutation_attestation_count = 0;
  std::string strict_hsic_gamma_residual_route = "stable-svd";
  std::string strict_permutation_residual_route = "stable-svd";
  bool method_permutation_inline_r_index_requested = false;
  bool method_permutation_inline_r_index_active = false;
  std::uint64_t method_permutation_inline_r_index_count = 0U;
  std::uint64_t method_permutation_inline_r_draw_count = 0U;
  int host_synchronization_count = 0;
  int result_cache_capacity = 0;
  int result_cache_warm_start_entries = 0;
  int result_cache_dataset_warm_start_entries = 0;
  std::uint64_t result_cache_warm_start_insert_count = 0;
  int result_cache_request_count = 0;
  int result_cache_hit_count = 0;
  int result_cache_preexisting_hit_count = 0;
  int result_cache_miss_count = 0;
  int result_cache_insert_count = 0;
  int result_cache_eviction_count = 0;
  int target_cache_capacity = 0;
  int target_cache_warm_start_entries = 0;
  int target_cache_dataset_warm_start_entries = 0;
  std::uint64_t target_cache_warm_start_insert_count = 0;
  int target_cache_request_count = 0;
  int target_cache_hit_count = 0;
  int target_cache_preexisting_hit_count = 0;
  int target_cache_miss_count = 0;
  int target_cache_insert_count = 0;
  int target_cache_eviction_count = 0;
  std::uint64_t dataset_cache_epoch_at_start = 0;
  int cpu_dcov_component_count = 0;
  int cpu_dcov_eigen_or_lowrank_count = 0;
  int cpu_dcov_pair_stat_count = 0;
  int cpu_gamma_pvalue_count = 0;
  int cpu_spectra_count = 0;
  std::size_t matrix_h2d_bytes = 0;
  std::size_t residual_d2h_bytes = 0;
  std::size_t component_d2h_bytes = 0;
  std::size_t compact_result_d2h_bytes = 0;
  bool fixed_sp_root_cache_enabled = false;
  bool setup_optimizer_pipeline_enabled = false;
  int setup_optimizer_pipeline_window_count = 0;
  int setup_optimizer_pipeline_peak_pending_count = 0;
  int setup_optimizer_pipeline_producer_delay_us = 0;
  int setup_optimizer_pipeline_producer_delay_count = 0;
  double setup_optimizer_pipeline_prepare_ms = 0.0;
  double setup_optimizer_pipeline_device_prepare_ms = 0.0;
  double setup_optimizer_pipeline_wait_ms = 0.0;
  double setup_optimizer_pipeline_overlap_ms = 0.0;
  double setup_optimizer_pipeline_level_wall_ms = 0.0;
  double setup_optimizer_pipeline_producer_delay_ms = 0.0;
  int fixed_sp_root_cache_runtime_count = 0;
  std::size_t fixed_sp_root_cache_capacity_entries_per_runtime = 0;
  std::size_t fixed_sp_root_cache_capacity_entries_total = 0;
  std::size_t fixed_sp_root_cache_capacity_bytes_total = 0;
  std::uint64_t fixed_sp_root_cache_lookup_count = 0;
  std::uint64_t fixed_sp_root_cache_hit_count = 0;
  std::uint64_t fixed_sp_root_cache_miss_count = 0;
  std::uint64_t fixed_sp_root_cache_insert_count = 0;
  std::uint64_t fixed_sp_root_cache_bypass_count = 0;
  std::uint64_t fixed_sp_root_cache_identity_rejection_count = 0;
  std::size_t fixed_sp_root_cache_entries = 0;
  std::size_t fixed_sp_root_cache_peak_entries = 0;
  std::size_t fixed_sp_root_cache_device_bytes = 0;
  std::size_t fixed_sp_root_cache_peak_device_bytes = 0;
  std::size_t fixed_sp_root_cache_hit_d2d_bytes = 0;
  std::size_t fixed_sp_root_cache_insert_d2d_bytes = 0;
  double native_setup_ms = 0.0;
  double native_setup_conditioning_copy_ms = 0.0;
  double native_setup_input_validation_ms = 0.0;
  double native_setup_smooth_build_ms = 0.0;
  double native_setup_block_assembly_ms = 0.0;
  double native_setup_gram_ms = 0.0;
  double native_setup_fingerprint_ms = 0.0;
  double native_setup_result_packaging_ms = 0.0;
  double native_setup_geometry_input_ms = 0.0;
  double native_setup_geometry_qr_ms = 0.0;
  double native_setup_geometry_penalty_ms = 0.0;
  double native_setup_geometry_initial_sp_ms = 0.0;
  double native_setup_geometry_packaging_ms = 0.0;
  double native_setup_fixed_sp_h2d_ms = 0.0;
  double native_setup_single_penalty_geometry_ms = 0.0;
  double native_setup_context_overhead_ms = 0.0;
  double native_setup_device_rehydrate_ms = 0.0;
  double cuda_optimizer_host_ms = 0.0;
  double cuda_single_penalty_optimizer_host_ms = 0.0;
  double cuda_multi_penalty_optimizer_host_ms = 0.0;
  double cuda_multi_penalty_optimizer_summed_setup_host_ms = 0.0;
  double cuda_multi_penalty_optimizer_max_setup_host_ms = 0.0;
  double cuda_multi_penalty_prepared_build_ms = 0.0;
  double cuda_single_penalty_optimizer_cuda_ms = 0.0;
  double frontier_optimizer_host_ms = 0.0;
  double singleton_padding_batch_host_ms = 0.0;
  double cuda_residual_solve_host_ms = 0.0;
  double cuda_single_penalty_residual_solve_host_ms = 0.0;
  double cuda_multi_penalty_residual_solve_host_ms = 0.0;
  double cuda_exact_screen_residual_solve_host_ms = 0.0;
  double cuda_guard_refinement_residual_solve_host_ms = 0.0;
  double cuda_exact_screen_component_build_ms = 0.0;
  double exact_residual_all_miss_solve_host_ms = 0.0;
  double exact_residual_all_miss_component_build_ms = 0.0;
  double exact_residual_mixed_solve_host_ms = 0.0;
  double exact_residual_mixed_component_build_ms = 0.0;
  double exact_residual_all_hit_new_cohort_solve_host_ms = 0.0;
  double exact_residual_all_hit_new_cohort_component_build_ms = 0.0;
  double exact_residual_all_hit_repeated_cohort_solve_host_ms = 0.0;
  double exact_residual_all_hit_repeated_cohort_component_build_ms = 0.0;
  double cuda_dcov_metadata_h2d_ms = 0.0;
  double cuda_dcov_component_build_ms = 0.0;
  double cuda_dcov_pair_gamma_ms = 0.0;
  double cuda_dcov_compact_d2h_ms = 0.0;
  double cuda_dcov_teardown_host_ms = 0.0;
  double cuda_dcov_host_ms = 0.0;
};

void delay_setup_optimizer_pipeline_producer(
    OneCallDiagnostics* diagnostics) {
  require(diagnostics != nullptr,
          "setup/optimizer producer delay diagnostics are missing");
  if (diagnostics->setup_optimizer_pipeline_producer_delay_us == 0) return;
  const auto started = std::chrono::steady_clock::now();
  std::this_thread::sleep_for(std::chrono::microseconds(
    diagnostics->setup_optimizer_pipeline_producer_delay_us));
  diagnostics->setup_optimizer_pipeline_producer_delay_count += 1;
  diagnostics->setup_optimizer_pipeline_producer_delay_ms +=
    elapsed_ms(started);
}

class CompactResultCache {
 public:
  struct Snapshot {
    int capacity = 0;
    int entries = 0;
    int dataset_entries = 0;
    std::uint64_t total_requests = 0;
    std::uint64_t total_hits = 0;
    std::uint64_t total_misses = 0;
    std::uint64_t total_inserts = 0;
    std::uint64_t total_evictions = 0;
    std::uint64_t generation = 0;
  };

  bool lookup(const std::string& key,
              const std::string& dataset,
              double* p_value,
              OneCallDiagnostics* diagnostics) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++total_requests_;
    ++diagnostics->result_cache_request_count;
    auto found = entries_.find(key);
    if (found == entries_.end()) {
      ++total_misses_;
      ++diagnostics->result_cache_miss_count;
      return false;
    }
    require(found->second.dataset == dataset,
            "compact result cache DatasetKey identity changed");
    ++total_hits_;
    ++diagnostics->result_cache_hit_count;
    if (found->second.insertion_ordinal <=
        diagnostics->result_cache_warm_start_insert_count) {
      ++diagnostics->result_cache_preexisting_hit_count;
    }
    order_.erase(found->second.order);
    order_.push_front(key);
    found->second.order = order_.begin();
    *p_value = found->second.p_value;
    return true;
  }

  void insert(const std::string& key,
              const std::string& dataset,
              double p_value,
              OneCallDiagnostics* diagnostics) {
    require(std::isfinite(p_value) && p_value >= 0.0 && p_value <= 1.0,
            "compact result cache refuses a non-finite p-value");
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = entries_.find(key);
    if (found != entries_.end()) {
      require(found->second.dataset == dataset,
              "compact result cache DatasetKey identity changed");
      found->second.p_value = p_value;
      order_.erase(found->second.order);
      order_.push_front(key);
      found->second.order = order_.begin();
      return;
    }
    while (entries_.size() >= static_cast<std::size_t>(capacity_)) {
      const std::string victim_key = order_.back();
      order_.pop_back();
      entries_.erase(victim_key);
      ++total_evictions_;
      ++diagnostics->result_cache_eviction_count;
    }
    order_.push_front(key);
    ++total_inserts_;
    entries_.emplace(
      key, Entry{dataset, p_value, total_inserts_, order_.begin()});
    ++diagnostics->result_cache_insert_count;
  }

  Snapshot snapshot(const std::string& dataset = std::string()) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return snapshot_locked(dataset);
  }

  Snapshot configure(int capacity) {
    require(capacity >= 1 && capacity <= kMaximumCompactResultCacheCapacity,
            "compact result cache capacity is outside [1, 1048576]");
    std::lock_guard<std::mutex> lock(mutex_);
    capacity_ = capacity;
    while (entries_.size() > static_cast<std::size_t>(capacity_)) {
      const std::string victim_key = order_.back();
      order_.pop_back();
      entries_.erase(victim_key);
      ++total_evictions_;
    }
    ++generation_;
    return snapshot_locked(std::string());
  }

  Snapshot reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    order_.clear();
    total_requests_ = 0;
    total_hits_ = 0;
    total_misses_ = 0;
    total_inserts_ = 0;
    total_evictions_ = 0;
    ++generation_;
    return snapshot_locked(std::string());
  }

 private:
  struct Entry {
    std::string dataset;
    double p_value = 0.0;
    std::uint64_t insertion_ordinal = 0;
    std::list<std::string>::iterator order;
  };

  Snapshot snapshot_locked(const std::string& dataset) const {
    Snapshot value;
    value.capacity = capacity_;
    value.entries = static_cast<int>(entries_.size());
    if (!dataset.empty()) {
      value.dataset_entries = static_cast<int>(std::count_if(
        entries_.begin(), entries_.end(), [&dataset](const auto& entry) {
          return entry.second.dataset == dataset;
        }));
    }
    value.total_requests = total_requests_;
    value.total_hits = total_hits_;
    value.total_misses = total_misses_;
    value.total_inserts = total_inserts_;
    value.total_evictions = total_evictions_;
    value.generation = generation_;
    return value;
  }

  mutable std::mutex mutex_;
  int capacity_ = kDefaultCompactResultCacheCapacity;
  std::list<std::string> order_;
  std::unordered_map<std::string, Entry> entries_;
  std::uint64_t total_requests_ = 0;
  std::uint64_t total_hits_ = 0;
  std::uint64_t total_misses_ = 0;
  std::uint64_t total_inserts_ = 0;
  std::uint64_t total_evictions_ = 0;
  std::uint64_t generation_ = 1;
};

CompactResultCache& compact_result_cache() {
  static CompactResultCache cache;
  return cache;
}

struct CachedTargetState {
  std::vector<double> selected_sp;
  FixedSpRoute planned_route = FixedSpRoute::CholeskyBatched;
};

class TargetStateCache {
 public:
  struct Snapshot {
    int capacity = 0;
    int entries = 0;
    int dataset_entries = 0;
    std::uint64_t total_requests = 0;
    std::uint64_t total_hits = 0;
    std::uint64_t total_misses = 0;
    std::uint64_t total_inserts = 0;
    std::uint64_t total_evictions = 0;
    std::uint64_t generation = 0;
  };

  bool lookup(const std::string& key,
              const std::string& dataset,
              int penalty_count,
              CachedTargetState* state,
              OneCallDiagnostics* diagnostics) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++total_requests_;
    ++diagnostics->target_cache_request_count;
    auto found = entries_.find(key);
    if (found == entries_.end()) {
      ++total_misses_;
      ++diagnostics->target_cache_miss_count;
      return false;
    }
    require(found->second.dataset == dataset,
            "target optimizer cache DatasetKey identity changed");
    require(static_cast<int>(found->second.state.selected_sp.size()) ==
              penalty_count,
            "target optimizer cache penalty shape changed");
    ++total_hits_;
    ++diagnostics->target_cache_hit_count;
    if (found->second.insertion_ordinal <=
        diagnostics->target_cache_warm_start_insert_count) {
      ++diagnostics->target_cache_preexisting_hit_count;
    }
    order_.erase(found->second.order);
    order_.push_front(key);
    found->second.order = order_.begin();
    *state = found->second.state;
    return true;
  }

  void insert(const std::string& key,
              const std::string& dataset,
              const CachedTargetState& state,
              OneCallDiagnostics* diagnostics) {
    require(!state.selected_sp.empty() &&
              std::all_of(state.selected_sp.begin(), state.selected_sp.end(),
                          [](double value) {
                            return std::isfinite(value) && value > 0.0;
                          }),
            "target optimizer cache refuses invalid smoothing parameters");
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = entries_.find(key);
    if (found != entries_.end()) {
      require(found->second.dataset == dataset,
              "target optimizer cache DatasetKey identity changed");
      found->second.state = state;
      order_.erase(found->second.order);
      order_.push_front(key);
      found->second.order = order_.begin();
      return;
    }
    while (entries_.size() >= static_cast<std::size_t>(capacity_)) {
      const std::string victim_key = order_.back();
      order_.pop_back();
      entries_.erase(victim_key);
      ++total_evictions_;
      ++diagnostics->target_cache_eviction_count;
    }
    order_.push_front(key);
    ++total_inserts_;
    entries_.emplace(
      key, Entry{dataset, state, total_inserts_, order_.begin()});
    ++diagnostics->target_cache_insert_count;
  }

  Snapshot snapshot(const std::string& dataset = std::string()) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return snapshot_locked(dataset);
  }

  Snapshot configure(int capacity) {
    require(capacity >= 1 && capacity <= kMaximumTargetStateCacheCapacity,
            "target optimizer cache capacity is outside [1, 524288]");
    std::lock_guard<std::mutex> lock(mutex_);
    capacity_ = capacity;
    while (entries_.size() > static_cast<std::size_t>(capacity_)) {
      const std::string victim_key = order_.back();
      order_.pop_back();
      entries_.erase(victim_key);
      ++total_evictions_;
    }
    ++generation_;
    return snapshot_locked(std::string());
  }

  Snapshot reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    order_.clear();
    total_requests_ = 0;
    total_hits_ = 0;
    total_misses_ = 0;
    total_inserts_ = 0;
    total_evictions_ = 0;
    ++generation_;
    return snapshot_locked(std::string());
  }

 private:
  struct Entry {
    std::string dataset;
    CachedTargetState state;
    std::uint64_t insertion_ordinal = 0;
    std::list<std::string>::iterator order;
  };

  Snapshot snapshot_locked(const std::string& dataset) const {
    Snapshot value;
    value.capacity = capacity_;
    value.entries = static_cast<int>(entries_.size());
    if (!dataset.empty()) {
      value.dataset_entries = static_cast<int>(std::count_if(
        entries_.begin(), entries_.end(), [&dataset](const auto& entry) {
          return entry.second.dataset == dataset;
        }));
    }
    value.total_requests = total_requests_;
    value.total_hits = total_hits_;
    value.total_misses = total_misses_;
    value.total_inserts = total_inserts_;
    value.total_evictions = total_evictions_;
    value.generation = generation_;
    return value;
  }

  mutable std::mutex mutex_;
  int capacity_ = kDefaultTargetStateCacheCapacity;
  std::list<std::string> order_;
  std::unordered_map<std::string, Entry> entries_;
  std::uint64_t total_requests_ = 0;
  std::uint64_t total_hits_ = 0;
  std::uint64_t total_misses_ = 0;
  std::uint64_t total_inserts_ = 0;
  std::uint64_t total_evictions_ = 0;
  std::uint64_t generation_ = 1;
};

TargetStateCache& target_state_cache() {
  static TargetStateCache cache;
  return cache;
}

std::atomic<std::uint64_t>& dataset_cache_epoch() {
  static std::atomic<std::uint64_t> epoch{1U};
  return epoch;
}

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

std::string fixed_sp_root_cache_key(
    int coefficient_dim,
    int offset_zero_based,
    int rank,
    const Rcpp::NumericMatrix& block) {
  std::ostringstream header;
  header << "schema=full-cuda-ci-fixed-sp-root-cache-key-v1\n"
         << "q=" << coefficient_dim << "\n"
         << "offset=" << offset_zero_based << "\n"
         << "rank=" << rank << "\n"
         << "dimension=" << block.nrow() << "\n"
         << "payload=";
  std::string payload = header.str();
  payload.append(
    reinterpret_cast<const char*>(block.begin()),
    static_cast<std::size_t>(block.size()) * sizeof(double));
  return full_cuda_ci_sha256_utf8(payload);
}

std::shared_ptr<PreparedSGpuHandle> create_fixed_sp_handle(
    const std::shared_ptr<CudaRuntimeContext>& runtime,
    const std::string& dataset,
    const std::shared_ptr<NativeSetupContext>& context,
    bool enable_root_cache) {
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
  if (enable_root_cache) {
    view.penalty_root_cache_keys.reserve(
      static_cast<std::size_t>(context->penalty_count));
    for (int penalty = 0; penalty < context->penalty_count; ++penalty) {
      view.penalty_root_cache_keys.push_back(fixed_sp_root_cache_key(
        context->coefficient_dim,
        offsets_zero_based[static_cast<std::size_t>(penalty)],
        rank_values[static_cast<std::size_t>(penalty)],
        Rcpp::NumericMatrix(blocks[penalty])));
    }
  }
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

std::shared_ptr<MultiPenaltyGcvCapacityPrepared> create_multi_penalty_handle(
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
  return create_multi_penalty_gcv_capacity_prepared(
    X.begin(), qr.begin(), tau.begin(), R.begin(), pivot_zero.data(),
    roots, matrices, ranks, initial_log_sp.begin(), context->n, q,
    context->penalty_count, target_capacity, 0);
}

void release_multi_penalty_handle(
    const std::shared_ptr<NativeSetupContext>& context,
    OneCallDiagnostics* diagnostics) {
  require(context != nullptr && diagnostics != nullptr,
          "multi-penalty prepared release state is invalid");
  if (!context->multi_penalty) {
    require(context->multi_penalty_target_capacity == 0,
            "multi-penalty prepared capacity survived release");
    return;
  }
  const MultiPenaltyGcvCudaPreparedInfo info =
    multi_penalty_gcv_capacity_prepared_info(context->multi_penalty);
  require(!info.residual_slot_leased,
          "multi-penalty prepared release observed a live residual");
  context->multi_penalty.reset();
  context->multi_penalty_target_capacity = 0;
  diagnostics->cuda_multi_penalty_prepared_release_count += 1;
}

void ensure_multi_penalty_handle(
    const std::shared_ptr<NativeSetupContext>& context,
    int target_capacity,
    OneCallDiagnostics* diagnostics) {
  require(context != nullptr && diagnostics != nullptr &&
            context->penalty_count > 1 && target_capacity >= 2 &&
            target_capacity <= 64,
          "multi-penalty prepared demand is invalid");
  if (context->multi_penalty &&
      context->multi_penalty_target_capacity >= target_capacity) {
    return;
  }
  if (context->multi_penalty) {
    release_multi_penalty_handle(context, diagnostics);
  }
  const auto started = std::chrono::steady_clock::now();
  context->multi_penalty = create_multi_penalty_handle(
    context, target_capacity);
  context->multi_penalty_target_capacity = target_capacity;
  diagnostics->cuda_multi_penalty_prepared_build_count += 1;
  diagnostics->cuda_multi_penalty_prepared_target_capacity_sum +=
    target_capacity;
  diagnostics->cuda_multi_penalty_prepared_target_capacity_peak = std::max(
    diagnostics->cuda_multi_penalty_prepared_target_capacity_peak,
    target_capacity);
  diagnostics->cuda_multi_penalty_prepared_build_ms += elapsed_ms(started);
}

class FixedSpRuntimePool {
 public:
  explicit FixedSpRuntimePool(int n) : n_(n) {
    runtime_for(64, 7);
  }

  std::shared_ptr<CudaRuntimeContext> base_runtime() {
    return runtime_for(64, 7);
  }

  std::shared_ptr<CudaRuntimeContext> runtime_for(
      int coefficient_dim, int penalty_count) {
    int q_capacity = 64;
    int penalty_capacity = 7;
    if (coefficient_dim > 64 || penalty_count > 7) {
      switch (multi_penalty_gcv_capacity_bucket(
                coefficient_dim, penalty_count)) {
        case MultiPenaltyGcvCapacityBucket::Small64:
          break;
        case MultiPenaltyGcvCapacityBucket::Base80:
          q_capacity = 80;
          penalty_capacity = 8;
          break;
        case MultiPenaltyGcvCapacityBucket::Extended192:
          q_capacity = 192;
          penalty_capacity = 21;
          break;
        case MultiPenaltyGcvCapacityBucket::Extended384:
          q_capacity = 384;
          penalty_capacity = 42;
          break;
        case MultiPenaltyGcvCapacityBucket::Extended559:
          q_capacity = 559;
          penalty_capacity = 62;
          break;
      }
    }
    auto found = runtimes_.find(q_capacity);
    if (found != runtimes_.end()) return found->second;

    std::shared_ptr<CudaRuntimeContext> runtime = create_fixed_sp_runtime(0);
    FixedSpCapacities capacities;
    capacities.n = n_;
    capacities.null_dim = q_capacity;
    capacities.target_count = 64;
    capacities.penalty_count = penalty_capacity;
    capacities.augmented_rows = n_ + q_capacity;
    reserve_fixed_sp_runtime(runtime, capacities);
    runtimes_.emplace(q_capacity, runtime);
    return runtime;
  }

  std::size_t close(OneCallDiagnostics* one_call_diagnostics = nullptr) {
    if (closed_) return workspace_bytes_;
    for (auto& entry : runtimes_) {
      if (!entry.second) continue;
      const FixedSpRuntimeInfo info = fixed_sp_runtime_info(entry.second);
      workspace_bytes_ += info.workspace_bytes;
      if (one_call_diagnostics != nullptr) {
        one_call_diagnostics->fixed_sp_root_cache_runtime_count += 1;
        one_call_diagnostics
          ->fixed_sp_root_cache_capacity_entries_per_runtime = std::max(
            one_call_diagnostics
              ->fixed_sp_root_cache_capacity_entries_per_runtime,
            info.penalty_root_cache_capacity_entries);
        one_call_diagnostics->fixed_sp_root_cache_capacity_entries_total +=
          info.penalty_root_cache_capacity_entries;
        one_call_diagnostics->fixed_sp_root_cache_capacity_bytes_total +=
          info.penalty_root_cache_capacity_bytes;
        one_call_diagnostics->fixed_sp_root_cache_lookup_count +=
          info.penalty_root_cache_lookup_count;
        one_call_diagnostics->fixed_sp_root_cache_hit_count +=
          info.penalty_root_cache_hit_count;
        one_call_diagnostics->fixed_sp_root_cache_miss_count +=
          info.penalty_root_cache_miss_count;
        one_call_diagnostics->fixed_sp_root_cache_insert_count +=
          info.penalty_root_cache_insert_count;
        one_call_diagnostics->fixed_sp_root_cache_bypass_count +=
          info.penalty_root_cache_bypass_count;
        one_call_diagnostics->fixed_sp_root_cache_identity_rejection_count +=
          info.penalty_root_cache_identity_rejection_count;
        one_call_diagnostics->fixed_sp_root_cache_entries +=
          info.penalty_root_cache_entries;
        one_call_diagnostics->fixed_sp_root_cache_peak_entries +=
          info.penalty_root_cache_peak_entries;
        one_call_diagnostics->fixed_sp_root_cache_device_bytes +=
          info.penalty_root_cache_device_bytes;
        one_call_diagnostics->fixed_sp_root_cache_peak_device_bytes +=
          info.penalty_root_cache_peak_device_bytes;
        one_call_diagnostics->fixed_sp_root_cache_hit_d2d_bytes +=
          info.penalty_root_cache_hit_d2d_bytes;
        one_call_diagnostics->fixed_sp_root_cache_insert_d2d_bytes +=
          info.penalty_root_cache_insert_d2d_bytes;
      }
      free_fixed_sp_runtime(&entry.second);
    }
    runtimes_.clear();
    closed_ = true;
    return workspace_bytes_;
  }

  ~FixedSpRuntimePool() {
    if (closed_) return;
    try {
      close();
    } catch (...) {
    }
  }

 private:
  int n_ = 0;
  std::map<int, std::shared_ptr<CudaRuntimeContext>> runtimes_;
  std::size_t workspace_bytes_ = 0U;
  bool closed_ = false;
};

std::shared_ptr<NativeSetupContext> build_native_context(
    const Rcpp::NumericMatrix& data,
    const std::string& dataset,
    const std::vector<int>& conditioning_set,
    FixedSpRuntimePool* runtime_pool,
    OneCallDiagnostics* diagnostics,
    bool enable_root_cache,
    bool defer_device_handles,
    std::vector<std::shared_ptr<const NativeUnivariateSmooth>>*
      univariate_smooths) {
  const auto started = std::chrono::steady_clock::now();
  require(runtime_pool != nullptr && univariate_smooths != nullptr &&
            univariate_smooths->size() ==
              static_cast<std::size_t>(data.ncol()),
          "native univariate smooth cache is malformed");
  auto stage_started = std::chrono::steady_clock::now();
  Rcpp::NumericMatrix conditioning(data.nrow(), conditioning_set.size());
  for (int column = 0; column < conditioning.ncol(); ++column) {
    const int source = conditioning_set[static_cast<std::size_t>(column)];
    require(source >= 0 && source < data.ncol(),
            "native conditioning source is out of range");
    std::copy(data.begin() + static_cast<std::ptrdiff_t>(source) * data.nrow(),
              data.begin() + static_cast<std::ptrdiff_t>(source + 1) * data.nrow(),
              conditioning.begin() +
                static_cast<std::ptrdiff_t>(column) * data.nrow());
  }
  const double conditioning_copy_ms = elapsed_ms(stage_started);
  auto context = std::make_shared<NativeSetupContext>();
  context->dataset_key = dataset;
  context->prepared_key = prepared_key(dataset, conditioning_set);
  context->conditioning_set = conditioning_set;
  NativeSetupProfile setup_profile;
  if (conditioning_set.size() > 2U) {
    std::vector<std::shared_ptr<const NativeUnivariateSmooth>> smooths;
    smooths.reserve(conditioning_set.size());
    for (int source : conditioning_set) {
      diagnostics->native_setup_univariate_primitive_request_count += 1;
      std::shared_ptr<const NativeUnivariateSmooth>& cached =
        (*univariate_smooths)[static_cast<std::size_t>(source)];
      if (cached) {
        diagnostics->native_setup_univariate_primitive_hit_count += 1;
      } else {
        cached = full_cuda_ci_native_univariate_smooth(
          data, source, &setup_profile);
        diagnostics->native_setup_univariate_primitive_build_count += 1;
        diagnostics->native_setup_univariate_primitive_cache_peak_entries =
          std::max(
            diagnostics->native_setup_univariate_primitive_cache_peak_entries,
            diagnostics->native_setup_univariate_primitive_build_count);
      }
      smooths.push_back(cached);
    }
    context->setup = full_cuda_ci_native_additive_setup(
      conditioning, conditioning_set, smooths, &setup_profile);
  } else {
    context->setup = full_cuda_ci_native_setup(conditioning, &setup_profile);
  }
  context->n = data.nrow();
  Rcpp::NumericMatrix X = context->setup["X"];
  Rcpp::List blocks = context->setup["penalty_blocks"];
  Rcpp::IntegerVector offsets = context->setup["penalty_offsets"];
  Rcpp::IntegerVector ranks = context->setup["penalty_ranks"];
  context->coefficient_dim = X.ncol();
  context->penalty_count = blocks.size();
  stage_started = std::chrono::steady_clock::now();
  context->geometry = full_cuda_ci_native_geometry_prepare(
    X, blocks, offsets, ranks, &setup_profile);
  const double geometry_ms = elapsed_ms(stage_started);
  double fixed_sp_h2d_ms = 0.0;
  if (!defer_device_handles) {
    stage_started = std::chrono::steady_clock::now();
    context->fixed_sp = create_fixed_sp_handle(
      runtime_pool->runtime_for(
        context->coefficient_dim, context->penalty_count),
      dataset, context, enable_root_cache);
    fixed_sp_h2d_ms = elapsed_ms(stage_started);
  }
  double single_penalty_geometry_ms = 0.0;
  if (context->penalty_count == 1) {
    stage_started = std::chrono::steady_clock::now();
    context->single_penalty.reset(
      new SinglePenaltyGeometry(prepare_single_penalty_geometry(context)));
    single_penalty_geometry_ms = elapsed_ms(stage_started);
  }
  const double total_ms = elapsed_ms(started);
  const double setup_profile_ms =
    setup_profile.input_validation_ms + setup_profile.smooth_build_ms +
    setup_profile.block_assembly_ms + setup_profile.gram_ms +
    setup_profile.fingerprint_ms + setup_profile.setup_packaging_ms;
  const double accounted_ms = conditioning_copy_ms + setup_profile_ms +
    geometry_ms + fixed_sp_h2d_ms + single_penalty_geometry_ms;
  diagnostics->native_setup_count += 1;
  diagnostics->native_setup_ms += total_ms;
  diagnostics->native_setup_conditioning_copy_ms += conditioning_copy_ms;
  diagnostics->native_setup_input_validation_ms +=
    setup_profile.input_validation_ms;
  diagnostics->native_setup_smooth_build_ms += setup_profile.smooth_build_ms;
  diagnostics->native_setup_block_assembly_ms +=
    setup_profile.block_assembly_ms;
  diagnostics->native_setup_gram_ms += setup_profile.gram_ms;
  diagnostics->native_setup_fingerprint_ms += setup_profile.fingerprint_ms;
  diagnostics->native_setup_result_packaging_ms +=
    setup_profile.setup_packaging_ms;
  diagnostics->native_setup_geometry_input_ms +=
    setup_profile.geometry_input_ms;
  diagnostics->native_setup_geometry_qr_ms += setup_profile.geometry_qr_ms;
  diagnostics->native_setup_geometry_penalty_ms +=
    setup_profile.geometry_penalty_ms;
  diagnostics->native_setup_geometry_initial_sp_ms +=
    setup_profile.geometry_initial_sp_ms;
  diagnostics->native_setup_geometry_packaging_ms +=
    setup_profile.geometry_packaging_ms;
  diagnostics->native_setup_fixed_sp_h2d_ms += fixed_sp_h2d_ms;
  diagnostics->native_setup_single_penalty_geometry_ms +=
    single_penalty_geometry_ms;
  diagnostics->native_setup_context_overhead_ms +=
    std::max(0.0, total_ms - accounted_ms);
  return context;
}

void materialize_new_native_context_device(
    const std::shared_ptr<NativeSetupContext>& context,
    FixedSpRuntimePool* runtime_pool,
    OneCallDiagnostics* diagnostics,
    bool enable_root_cache) {
  require(context && runtime_pool != nullptr && diagnostics != nullptr &&
            context->penalty_count >= 1 && !context->fixed_sp &&
            !context->multi_penalty,
          "deferred native setup device state is invalid");
  const auto started = std::chrono::steady_clock::now();
  context->fixed_sp = create_fixed_sp_handle(
    runtime_pool->runtime_for(
      context->coefficient_dim, context->penalty_count),
    context->dataset_key, context, enable_root_cache);
  const double elapsed = elapsed_ms(started);
  diagnostics->native_setup_ms += elapsed;
  diagnostics->native_setup_fixed_sp_h2d_ms += elapsed;
}

std::shared_ptr<NativeSetupContext> build_direct_context(
    const Rcpp::NumericMatrix& data,
    const std::string& dataset,
    const std::shared_ptr<CudaRuntimeContext>& runtime) {
  auto context = std::make_shared<NativeSetupContext>();
  context->dataset_key = dataset;
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
  context->fixed_sp = create_fixed_sp_handle(
    runtime, dataset, context, false);
  return context;
}

void close_context(const std::shared_ptr<NativeSetupContext>& context) {
  if (!context) return;
  context->multi_penalty.reset();
  context->multi_penalty_target_capacity = 0;
  if (context->fixed_sp) {
    try {
      free_prepared_s_gpu(&context->fixed_sp);
    } catch (...) {
      context->fixed_sp.reset();
    }
  }
}

void rehydrate_native_context(
    const std::shared_ptr<NativeSetupContext>& context,
    FixedSpRuntimePool* runtime_pool,
    OneCallDiagnostics* diagnostics,
    bool enable_root_cache) {
  const auto started = std::chrono::steady_clock::now();
  require(context && runtime_pool != nullptr &&
            !context->fixed_sp && !context->multi_penalty,
          "native setup device rehydrate state is invalid");
  try {
    context->fixed_sp = create_fixed_sp_handle(
      runtime_pool->runtime_for(
        context->coefficient_dim, context->penalty_count),
      context->dataset_key, context, enable_root_cache);
  } catch (...) {
    close_context(context);
    throw;
  }
  diagnostics->native_setup_device_rehydrate_count += 1;
  diagnostics->native_setup_device_rehydrate_ms += elapsed_ms(started);
}

class NativeSetupCache {
 public:
  NativeSetupCache(const Rcpp::NumericMatrix& data,
                   std::string dataset,
                   FixedSpRuntimePool* runtime_pool,
                   OneCallDiagnostics* diagnostics,
                   bool enable_root_cache,
                   bool defer_device_handles)
      : data_(data),
        dataset_(std::move(dataset)),
        runtime_pool_(runtime_pool),
        diagnostics_(diagnostics),
        enable_root_cache_(enable_root_cache),
        defer_device_handles_(defer_device_handles),
        univariate_smooths_(static_cast<std::size_t>(data.ncol())) {
    require(runtime_pool_ != nullptr && diagnostics_ != nullptr,
            "native setup diagnostics are missing");
    diagnostics_->native_setup_univariate_primitive_cache_capacity =
      data.ncol();
  }

  std::shared_ptr<NativeSetupContext> get(
      const std::vector<int>& conditioning_set) {
    const std::string key = prepared_key(dataset_, conditioning_set);
    diagnostics_->physical_prepared_keys.insert(key);
    diagnostics_->native_setup_cache_request_count += 1;
    auto found = entries_.find(key);
    if (found != entries_.end()) {
      diagnostics_->native_setup_cache_hit_count += 1;
      diagnostics_->native_setup_device_cache_hit_count += 1;
      std::list<std::string>& order = found->second.single_penalty ?
        single_penalty_order_ : multi_penalty_order_;
      order.erase(found->second.order);
      order.push_front(key);
      found->second.order = order.begin();
      touch_host(key);
      return found->second.context;
    }

    auto host = host_entries_.find(key);
    if (host != host_entries_.end()) {
      diagnostics_->native_setup_cache_hit_count += 1;
      diagnostics_->native_setup_host_cache_hit_count += 1;
      std::shared_ptr<NativeSetupContext> context = host->second.context;
      const bool single_penalty = context->penalty_count == 1;
      std::list<std::string>& order = single_penalty ?
        single_penalty_order_ : multi_penalty_order_;
      const std::size_t capacity = static_cast<std::size_t>(single_penalty ?
        kSinglePenaltyPreparedCacheCapacity : kMultiPenaltyPreparedCacheCapacity);
      if (order.size() >= capacity) evict_one(&order);
      rehydrate_native_context(
        context, runtime_pool_, diagnostics_, enable_root_cache_);
      order.push_front(key);
      entries_.emplace(
        key, Entry{context, single_penalty, order.begin()});
      live_conditioning_keys_.insert(conditioning_key(conditioning_set));
      touch_host(key);
      return context;
    }

    diagnostics_->native_setup_cache_miss_count += 1;
    if (host_entries_.size() >=
        static_cast<std::size_t>(kHostPreparedCacheCapacity)) {
      evict_host_one();
    }
    std::shared_ptr<NativeSetupContext> context = build_native_context(
      data_, dataset_, conditioning_set, runtime_pool_, diagnostics_,
      enable_root_cache_, defer_device_handles_,
      &univariate_smooths_);
    const bool single_penalty = context->penalty_count == 1;
    std::list<std::string>& order = single_penalty ?
      single_penalty_order_ : multi_penalty_order_;
    const std::size_t capacity = static_cast<std::size_t>(single_penalty ?
      kSinglePenaltyPreparedCacheCapacity : kMultiPenaltyPreparedCacheCapacity);
    if (order.size() >= capacity) evict_one(&order);
    order.push_front(key);
    entries_.emplace(
      key, Entry{context, single_penalty, order.begin()});
    host_order_.push_front(key);
    host_entries_.emplace(key, HostEntry{context, host_order_.begin()});
    diagnostics_->native_setup_host_cache_peak_entries = std::max(
      diagnostics_->native_setup_host_cache_peak_entries,
      static_cast<int>(host_entries_.size()));
    live_conditioning_keys_.insert(conditioning_key(conditioning_set));
    return context;
  }

  void materialize_device(
      const std::shared_ptr<NativeSetupContext>& context) {
    require(context != nullptr,
            "deferred native setup context is missing");
    if (context->fixed_sp) return;
    materialize_new_native_context_device(
      context, runtime_pool_, diagnostics_, enable_root_cache_);
  }

  bool contains_conditioning_key(const std::string& key) const {
    return live_conditioning_keys_.find(key) !=
      live_conditioning_keys_.end();
  }

  void clear() {
    for (auto& entry : entries_) close_context(entry.second.context);
    entries_.clear();
    single_penalty_order_.clear();
    multi_penalty_order_.clear();
    live_conditioning_keys_.clear();
    host_entries_.clear();
    host_order_.clear();
    univariate_smooths_.clear();
  }

  ~NativeSetupCache() {
    clear();
  }

 private:
  struct Entry {
    std::shared_ptr<NativeSetupContext> context;
    bool single_penalty = false;
    std::list<std::string>::iterator order;
  };

  struct HostEntry {
    std::shared_ptr<NativeSetupContext> context;
    std::list<std::string>::iterator order;
  };

  void touch_host(const std::string& key) {
    auto found = host_entries_.find(key);
    require(found != host_entries_.end(),
            "native host setup cache key is missing");
    host_order_.erase(found->second.order);
    host_order_.push_front(key);
    found->second.order = host_order_.begin();
  }

  void evict_host_one() {
    auto candidate = host_order_.end();
    while (candidate != host_order_.begin()) {
      --candidate;
      if (entries_.find(*candidate) == entries_.end()) {
        const std::string key = *candidate;
        host_order_.erase(candidate);
        host_entries_.erase(key);
        diagnostics_->native_setup_host_cache_eviction_count += 1;
        return;
      }
    }
    require(false, "native host setup cache has no dormant eviction candidate");
  }

  void evict_one(std::list<std::string>* order) {
    require(order != nullptr && !order->empty(),
            "native setup cache eviction order is empty");
    const std::string evicted = order->back();
    order->pop_back();
    auto victim = entries_.find(evicted);
    require(victim != entries_.end(),
            "native setup cache eviction key is missing");
    live_conditioning_keys_.erase(
      conditioning_key(victim->second.context->conditioning_set));
    close_context(victim->second.context);
    entries_.erase(victim);
    diagnostics_->native_setup_cache_eviction_count += 1;
  }

  Rcpp::NumericMatrix data_;
  std::string dataset_;
  FixedSpRuntimePool* runtime_pool_ = nullptr;
  OneCallDiagnostics* diagnostics_;
  bool enable_root_cache_ = false;
  bool defer_device_handles_ = false;
  std::list<std::string> single_penalty_order_;
  std::list<std::string> multi_penalty_order_;
  std::unordered_map<std::string, Entry> entries_;
  std::unordered_set<std::string> live_conditioning_keys_;
  std::vector<std::shared_ptr<const NativeUnivariateSmooth>>
    univariate_smooths_;
  std::list<std::string> host_order_;
  std::unordered_map<std::string, HostEntry> host_entries_;
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
  std::vector<double> statistics;
  std::vector<double> means;
  std::vector<double> variances;
};

enum class ExactResidualOpportunityClass {
  AllMiss,
  Mixed,
  AllHitNewCohort,
  AllHitRepeatedCohort
};

void accumulate_exact_residual_opportunity(
    ExactResidualOpportunityClass category,
    int target_count,
    const FullCudaCiExactBatchDiagnostics& value,
    OneCallDiagnostics* diagnostics) {
  switch (category) {
    case ExactResidualOpportunityClass::AllMiss:
      diagnostics->exact_residual_all_miss_batch_count += 1;
      diagnostics->exact_residual_all_miss_target_count += target_count;
      diagnostics->exact_residual_all_miss_solve_host_ms +=
        value.residual_solve_host_ms;
      diagnostics->exact_residual_all_miss_component_build_ms +=
        value.component_build_cuda_ms;
      return;
    case ExactResidualOpportunityClass::Mixed:
      diagnostics->exact_residual_mixed_batch_count += 1;
      diagnostics->exact_residual_mixed_target_count += target_count;
      diagnostics->exact_residual_mixed_solve_host_ms +=
        value.residual_solve_host_ms;
      diagnostics->exact_residual_mixed_component_build_ms +=
        value.component_build_cuda_ms;
      return;
    case ExactResidualOpportunityClass::AllHitNewCohort:
      diagnostics->exact_residual_all_hit_new_cohort_batch_count += 1;
      diagnostics->exact_residual_all_hit_new_cohort_target_count +=
        target_count;
      diagnostics->exact_residual_all_hit_new_cohort_solve_host_ms +=
        value.residual_solve_host_ms;
      diagnostics->exact_residual_all_hit_new_cohort_component_build_ms +=
        value.component_build_cuda_ms;
      return;
    case ExactResidualOpportunityClass::AllHitRepeatedCohort:
      diagnostics->exact_residual_all_hit_repeated_cohort_batch_count += 1;
      diagnostics->exact_residual_all_hit_repeated_cohort_target_count +=
        target_count;
      diagnostics->exact_residual_all_hit_repeated_cohort_solve_host_ms +=
        value.residual_solve_host_ms;
      diagnostics->exact_residual_all_hit_repeated_cohort_component_build_ms +=
        value.component_build_cuda_ms;
      return;
  }
  throw std::runtime_error("unknown exact residual opportunity class");
}

void accumulate_exact_diagnostics(
    const FullCudaCiExactBatchDiagnostics& value,
    OneCallDiagnostics* diagnostics) {
  diagnostics->cuda_residual_batch_count += 1;
  diagnostics->cuda_exact_screen_residual_batch_count += 1;
  diagnostics->cuda_exact_screen_residual_target_count += value.target_count;
  diagnostics->cuda_exact_screen_component_count +=
    value.component_build_count;
  diagnostics->cuda_exact_screen_pair_count += value.pair_evaluation_count;
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
  diagnostics->cuda_residual_solve_host_ms += value.residual_solve_host_ms;
  diagnostics->cuda_exact_screen_residual_solve_host_ms +=
    value.residual_solve_host_ms;
  diagnostics->cuda_dcov_metadata_h2d_ms += value.metadata_h2d_cuda_ms;
  diagnostics->cuda_dcov_component_build_ms += value.component_build_cuda_ms;
  diagnostics->cuda_exact_screen_component_build_ms +=
    value.component_build_cuda_ms;
  diagnostics->cuda_dcov_pair_gamma_ms += value.pair_evaluation_cuda_ms;
  diagnostics->cuda_dcov_compact_d2h_ms += value.compact_d2h_cuda_ms;
  diagnostics->cuda_dcov_teardown_host_ms += value.teardown_host_ms;
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
  diagnostics->cuda_guard_refinement_residual_batch_count += 1;
  diagnostics->cuda_guard_refinement_residual_target_count +=
    value.target_count;
  diagnostics->cuda_guard_refinement_component_count +=
    value.component_build_count;
  diagnostics->cuda_guard_refinement_pair_count += value.pair_count;
  diagnostics->cuda_dcov_component_count += value.component_build_count;
  diagnostics->cuda_dcov_pair_count += value.cuda_pair_count;
  diagnostics->cuda_gamma_pvalue_count += value.cuda_gamma_count;
  diagnostics->component_cache_request_count += 2 * value.pair_count;
  diagnostics->component_cache_miss_count += value.component_build_count;
  diagnostics->component_cache_hit_count +=
    2 * value.pair_count - value.component_build_count;
  diagnostics->host_synchronization_count += value.explicit_host_wait_count;
  diagnostics->cpu_dcov_component_count += value.cpu_dcov_component_count;
  diagnostics->cpu_dcov_eigen_or_lowrank_count += value.cpu_dcov_eigen_count;
  diagnostics->cpu_dcov_pair_stat_count +=
    value.cpu_dcov_pair_statistic_count;
  diagnostics->cpu_gamma_pvalue_count += value.cpu_gamma_p_value_count;
  diagnostics->residual_d2h_bytes += value.residual_d2h_bytes;
  diagnostics->component_d2h_bytes += value.component_d2h_bytes;
  diagnostics->compact_result_d2h_bytes += value.compact_result_d2h_bytes;
  diagnostics->cuda_residual_solve_host_ms += value.residual_solve_host_ms;
  diagnostics->cuda_guard_refinement_residual_solve_host_ms +=
    value.residual_solve_host_ms;
  diagnostics->cuda_dcov_metadata_h2d_ms += value.metadata_h2d_cuda_ms;
  diagnostics->cuda_dcov_component_build_ms += value.component_build_cuda_ms;
  diagnostics->cuda_dcov_pair_gamma_ms += value.pair_evaluation_cuda_ms;
  diagnostics->cuda_dcov_compact_d2h_ms += value.compact_d2h_cuda_ms;
  diagnostics->cuda_dcov_teardown_host_ms += value.teardown_host_ms;
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

void accumulate_method_diagnostics(
    const FullCudaCiMethodBatchDiagnostics& value,
    OneCallDiagnostics* diagnostics) {
  const int solved_target_count = value.residual_cache_all_hit_batch_count > 0 ?
    0 : value.target_count;
  const int solved_batch_count = solved_target_count > 0 ? 1 : 0;
  diagnostics->cuda_residual_batch_count += solved_batch_count;
  diagnostics->cuda_exact_screen_residual_batch_count += solved_batch_count;
  diagnostics->cuda_exact_screen_residual_target_count += solved_target_count;
  diagnostics->cuda_exact_screen_component_count +=
    value.component_build_count;
  diagnostics->cuda_exact_screen_pair_count += value.pair_evaluation_count;
  diagnostics->cuda_dcov_component_count += value.component_build_count;
  diagnostics->cuda_dcov_pair_count += value.pair_evaluation_count;
  if (value.ci_method == "hsic.gamma") {
    diagnostics->cuda_gamma_pvalue_count += value.pair_evaluation_count;
  }
  diagnostics->component_cache_request_count += 2 * value.pair_count;
  diagnostics->component_cache_miss_count += value.component_build_count;
  diagnostics->component_cache_hit_count +=
    2 * value.pair_count - value.component_build_count;
  diagnostics->component_cache_eviction_count +=
    value.component_cache_persistent_eviction_count;
  diagnostics->method_component_cache_request_count +=
    value.component_cache_persistent_request_count;
  diagnostics->method_component_cache_lookup_count +=
    value.component_cache_persistent_lookup_count;
  diagnostics->method_component_cache_hit_count +=
    value.component_cache_persistent_hit_count;
  diagnostics->method_component_cache_miss_count +=
    value.component_cache_persistent_miss_count;
  diagnostics->method_component_cache_insert_count +=
    value.component_cache_persistent_insert_count;
  diagnostics->method_component_cache_eviction_count +=
    value.component_cache_persistent_eviction_count;
  diagnostics->method_component_cache_capacity_entries = std::max(
    diagnostics->method_component_cache_capacity_entries,
    value.component_cache_persistent_capacity_entries);
  diagnostics->method_component_cache_device_bytes = std::max(
    diagnostics->method_component_cache_device_bytes,
    value.component_cache_persistent_device_bytes);
  diagnostics->method_component_cache_gather_d2d_bytes +=
    value.component_cache_persistent_gather_d2d_bytes;
  diagnostics->method_component_cache_store_d2d_bytes +=
    value.component_cache_persistent_store_d2d_bytes;
  diagnostics->method_residual_cache_lookup_count +=
    value.residual_cache_lookup_count;
  diagnostics->method_residual_cache_hit_count +=
    value.residual_cache_hit_count;
  diagnostics->method_residual_cache_insert_count +=
    value.residual_cache_insert_count;
  diagnostics->method_residual_cache_eviction_count +=
    value.residual_cache_eviction_count;
  diagnostics->method_residual_cache_all_hit_batch_count +=
    value.residual_cache_all_hit_batch_count;
  diagnostics->method_residual_cache_bypassed_target_count +=
    value.residual_cache_bypassed_target_count;
  diagnostics->method_residual_cache_capacity_entries = std::max(
    diagnostics->method_residual_cache_capacity_entries,
    value.residual_cache_capacity_entries);
  diagnostics->method_residual_cache_device_bytes = std::max(
    diagnostics->method_residual_cache_device_bytes,
    value.residual_cache_device_bytes);
  diagnostics->method_residual_cache_gather_d2d_bytes +=
    value.residual_cache_gather_d2d_bytes;
  diagnostics->method_execution_context_call_count +=
    value.execution_context_call_count;
  diagnostics->method_execution_context_reuse_count +=
    value.execution_context_reuse_count;
  diagnostics->method_execution_context_buffer_growth_count +=
    value.execution_context_buffer_growth_count;
  diagnostics->method_execution_context_peak_device_bytes = std::max(
    diagnostics->method_execution_context_peak_device_bytes,
    value.execution_context_device_bytes);
  diagnostics->residual_d2h_bytes += value.residual_d2h_bytes;
  diagnostics->component_d2h_bytes += value.component_d2h_bytes;
  diagnostics->compact_result_d2h_bytes += value.compact_result_d2h_bytes;
  diagnostics->cuda_residual_solve_host_ms += value.residual_solve_host_ms;
  diagnostics->cuda_exact_screen_residual_solve_host_ms +=
    value.residual_solve_host_ms;
  diagnostics->cuda_dcov_component_build_ms += value.component_build_cuda_ms;
  diagnostics->cuda_exact_screen_component_build_ms +=
    value.component_build_cuda_ms;
  diagnostics->cuda_dcov_pair_gamma_ms += value.pair_evaluation_cuda_ms;
  diagnostics->cuda_dcov_compact_d2h_ms += value.compact_d2h_cuda_ms;
  diagnostics->cuda_dcov_host_ms += value.total_host_ms;
  diagnostics->method_request_identity_validation_host_ms +=
    value.request_identity_validation_host_ms;
  diagnostics->method_permutation_payload_validation_scan_count +=
    value.permutation_payload_validation_scan_count;
  diagnostics->method_permutation_payload_validation_scan_bytes +=
    value.permutation_payload_validation_scan_bytes;
  diagnostics->method_permutation_attestation_count +=
    value.permutation_attestation_authenticated ? 1 : 0;
  require(value.request_identity_authenticated &&
            value.permutation_attestation_authenticated &&
            value.prepared_identity_authenticated &&
            value.target_identity_authenticated &&
            value.residuals_device_resident &&
            value.compact_result_only_d2h &&
            value.caller_device_restored &&
            value.residual_d2h_bytes == 0U &&
            value.component_d2h_bytes == 0U,
          "one-call strict CUDA CI method authority gate failed closed");
}

void accumulate_single_penalty_optimizer_diagnostics(
    const SinglePenaltyGcvCudaDiagnostics& value,
    OneCallDiagnostics* diagnostics) {
  diagnostics->cuda_single_penalty_optimizer_setup_count += 1;
  diagnostics->cuda_single_penalty_optimizer_cuda_ms +=
    value.optimizer_cuda_ms;
  diagnostics->cuda_optimizer_kernel_launch_count +=
    value.mgcv_qt_y_kernel_launch_count + value.grid_kernel_launch_count +
    value.exact_mroot_kernel_launch_count +
    value.exact_objective_kernel_launch_count +
    value.exact_endpoint_kernel_launch_count +
    value.optimizer_kernel_launch_count +
      value.augmented_objective_kernel_launch_count;
}

void merge_decomposition_reuse_aggregate(
    const MultiPenaltyGcvCudaDiagnostics& source,
    DecompositionReuseAggregate* destination) {
  destination->batch_count += 1U;
  destination->request_count += source.cuda_decomposition_request_count;
  destination->stored_count += source.cuda_decomposition_stored_count;
  destination->overflow_count +=
    source.cuda_decomposition_trace_overflow_count;
  destination->unique_key_count +=
    source.cuda_decomposition_unique_key_count;
  destination->reuse_count += source.cuda_decomposition_reuse_count;
  destination->route_mismatch_count +=
    source.cuda_decomposition_route_mismatch_count;
  for (int stage = 0;
       stage < kMultiPenaltyGcvDecompositionTraceStageCount; ++stage) {
    const std::size_t index = static_cast<std::size_t>(stage);
    destination->stage_request_count[index] +=
      source.cuda_decomposition_stage_request_count[index];
    destination->stage_unique_key_count[index] +=
      source.cuda_decomposition_stage_unique_key_count[index];
    destination->stage_reuse_count[index] +=
      source.cuda_decomposition_stage_reuse_count[index];
  }
  for (int route = 0;
       route < kMultiPenaltyGcvDecompositionTraceRouteCount; ++route) {
    const std::size_t index = static_cast<std::size_t>(route);
    destination->route_request_count[index] +=
      source.cuda_decomposition_route_request_count[index];
    destination->route_unique_key_count[index] +=
      source.cuda_decomposition_route_unique_key_count[index];
    destination->route_reuse_count[index] +=
      source.cuda_decomposition_route_reuse_count[index];
  }
  if (destination->group_size_histogram.size() <
      source.cuda_decomposition_reuse_group_size_histogram.size()) {
    destination->group_size_histogram.resize(
      source.cuda_decomposition_reuse_group_size_histogram.size(), 0U);
  }
  for (std::size_t size = 0;
       size < source.cuda_decomposition_reuse_group_size_histogram.size();
       ++size) {
    destination->group_size_histogram[size] +=
      source.cuda_decomposition_reuse_group_size_histogram[size];
  }
}

void accumulate_decomposition_reuse_diagnostics(
    const MultiPenaltyGcvCudaOptimization& value,
    const std::string& prepared_key,
    OneCallDiagnostics* diagnostics) {
  const MultiPenaltyGcvCudaDiagnostics& source = value.diagnostics;
  const bool expected =
    diagnostics->cuda_multi_penalty_decomposition_trace_capacity_per_target >
      0;
  require(source.cuda_decomposition_trace_enabled == expected,
          "multi-penalty decomposition trace activation drifted");
  if (!expected) {
    require(source.cuda_decomposition_request_count == 0U &&
              source.cuda_decomposition_stored_count == 0U &&
              source.cuda_decomposition_trace_overflow_count == 0U,
            "disabled multi-penalty decomposition trace recorded work");
    return;
  }
  require(source.cuda_decomposition_trace_capacity_per_target ==
            diagnostics
              ->cuda_multi_penalty_decomposition_trace_capacity_per_target &&
            source.cuda_decomposition_request_count > 0U &&
            source.cuda_decomposition_stored_count <=
              source.cuda_decomposition_request_count &&
            source.cuda_decomposition_unique_key_count <=
              source.cuda_decomposition_stored_count &&
            source.cuda_decomposition_reuse_count ==
              source.cuda_decomposition_stored_count -
                source.cuda_decomposition_unique_key_count,
          "multi-penalty decomposition trace accounting is malformed");
  merge_decomposition_reuse_aggregate(
    source, &diagnostics->cuda_multi_penalty_decomposition_reuse);
  merge_decomposition_reuse_aggregate(
    source,
    &diagnostics->cuda_multi_penalty_decomposition_reuse_by_shape[
      std::make_pair(value.coefficient_dim, value.penalty_count)]);
  for (const MultiPenaltyGcvCudaDecompositionIterationReuse& row :
       source.cuda_decomposition_iteration_reuse) {
    diagnostics->cuda_multi_penalty_decomposition_iteration_reuse.push_back(
      DecompositionIterationReuseRow{
        prepared_key,
        value.coefficient_dim,
        value.penalty_count,
        row.iteration,
        row.stability_replay,
        row.request_count,
        row.unique_key_count,
        row.reuse_count
      });
  }
}

void accumulate_multi_penalty_optimizer_diagnostics(
    const MultiPenaltyGcvCudaOptimization& value,
    const std::string& prepared_key,
    OneCallDiagnostics* diagnostics) {
  require(diagnostics != nullptr && value.target_count > 0,
          "multi-penalty optimizer diagnostics are malformed");
  const MultiPenaltyGcvCudaDiagnostics& source = value.diagnostics;
  accumulate_decomposition_reuse_diagnostics(
    value, prepared_key, diagnostics);
  diagnostics->cuda_multi_penalty_optimizer_summed_setup_host_ms +=
    source.total_host_ms;
  diagnostics->cuda_multi_penalty_optimizer_max_setup_host_ms = std::max(
    diagnostics->cuda_multi_penalty_optimizer_max_setup_host_ms,
    source.total_host_ms);
  diagnostics->cuda_multi_penalty_complete_evaluation_count +=
    source.cuda_complete_evaluation_count;
  diagnostics->cuda_multi_penalty_score_only_evaluation_count +=
    source.cuda_score_only_evaluation_count;
  diagnostics->cuda_multi_penalty_guarded_qr_evaluation_count +=
    source.cuda_guarded_qr_evaluation_count;
  diagnostics->cuda_multi_penalty_stable_svd_evaluation_count +=
    source.cuda_stable_svd_evaluation_count;
  diagnostics->cuda_multi_penalty_selected_evaluation_reuse_count +=
    source.cuda_selected_evaluation_reuse_count;
  diagnostics->cuda_multi_penalty_stability_replay_target_count +=
    source.cuda_stability_replay_target_count;
  diagnostics->cuda_multi_penalty_stability_replay_selected_count +=
    source.cuda_stability_replay_selected_count;
  diagnostics->cuda_multi_penalty_terminal_confirmation_count +=
    source.cuda_terminal_boundary_confirmation_count;
  diagnostics->cuda_multi_penalty_hessian_eigensolver_count +=
    source.cuda_hessian_eigensolver_count;
  diagnostics->cuda_multi_penalty_penalty_factor_cycles +=
    source.cuda_penalty_factor_augmentation_cycles;
  diagnostics->cuda_multi_penalty_qr_svd_cycles +=
    source.cuda_qr_svd_cycles;
  diagnostics->cuda_multi_penalty_qr_bidiagonal_reduction_cycles +=
    source.cuda_qr_bidiagonal_reduction_cycles;
  diagnostics->cuda_multi_penalty_qr_factorization_cycles +=
    source.cuda_qr_factorization_cycles;
  diagnostics->cuda_multi_penalty_q_generation_cycles +=
    source.cuda_q_generation_cycles;
  diagnostics->cuda_multi_penalty_qr_guard_cycles +=
    source.cuda_qr_guard_cycles;
  diagnostics->cuda_multi_penalty_stable_bidiagonal_reduction_cycles +=
    source.cuda_stable_bidiagonal_reduction_cycles;
  diagnostics->cuda_multi_penalty_bidiagonal_svd_cycles +=
    source.cuda_bidiagonal_svd_cycles;
  diagnostics->cuda_multi_penalty_svd_vector_postback_cycles +=
    source.cuda_svd_vector_postback_cycles;
  diagnostics->cuda_multi_penalty_left_vector_product_cycles +=
    source.cuda_left_vector_product_cycles;
  diagnostics->cuda_multi_penalty_score_construction_cycles +=
    source.cuda_score_construction_cycles;
  diagnostics->cuda_multi_penalty_derivative_hessian_cycles +=
    source.cuda_derivative_hessian_cycles;
  diagnostics->cuda_multi_penalty_stability_replay_discarded_cycles +=
    source.cuda_stability_replay_discarded_cycles;
  diagnostics->cuda_multi_penalty_terminal_confirmation_cycles +=
    source.cuda_terminal_boundary_confirmation_cycles;

  require(
    value.optimizer_iterations.size() ==
      static_cast<std::size_t>(value.target_count) &&
    value.score_calls.size() == static_cast<std::size_t>(value.target_count) &&
    value.objective_calls.size() ==
      static_cast<std::size_t>(value.target_count) &&
    value.step_halving_count.size() ==
      static_cast<std::size_t>(value.target_count) &&
    value.newton_trial_count.size() ==
      static_cast<std::size_t>(value.target_count) &&
    value.steepest_descent_trial_count.size() ==
      static_cast<std::size_t>(value.target_count) &&
    value.boundary_probe_count.size() ==
      static_cast<std::size_t>(value.target_count),
    "multi-penalty optimizer transcript diagnostics are malformed");
  for (int target = 0; target < value.target_count; ++target) {
    const std::size_t index = static_cast<std::size_t>(target);
    diagnostics->cuda_multi_penalty_optimizer_iteration_sum +=
      value.optimizer_iterations[index];
    diagnostics->cuda_multi_penalty_optimizer_iteration_max = std::max(
      diagnostics->cuda_multi_penalty_optimizer_iteration_max,
      value.optimizer_iterations[index]);
    diagnostics->cuda_multi_penalty_score_call_sum += value.score_calls[index];
    diagnostics->cuda_multi_penalty_objective_call_sum +=
      value.objective_calls[index];
    diagnostics->cuda_multi_penalty_step_halving_sum +=
      value.step_halving_count[index];
    diagnostics->cuda_multi_penalty_newton_trial_sum +=
      value.newton_trial_count[index];
    diagnostics->cuda_multi_penalty_steepest_descent_trial_sum +=
      value.steepest_descent_trial_count[index];
    diagnostics->cuda_multi_penalty_boundary_probe_sum +=
      value.boundary_probe_count[index];
  }
}

const std::array<const char*,
                 kMultiPenaltyGcvDecompositionTraceStageCount>&
decomposition_trace_stage_names() {
  static const std::array<const char*,
                          kMultiPenaltyGcvDecompositionTraceStageCount>
    names = {{
      "initial", "newton_trial", "steepest_descent_trial",
      "step_halving", "boundary_probe", "terminal_confirmation",
      "stability_replay", "selected_fit"
    }};
  return names;
}

const std::array<const char*,
                 kMultiPenaltyGcvDecompositionTraceRouteCount>&
decomposition_trace_route_names() {
  static const std::array<const char*,
                          kMultiPenaltyGcvDecompositionTraceRouteCount>
    names = {{"unknown", "guarded_qr", "stable_svd"}};
  return names;
}

double decomposition_group_size_quantile(
    const std::vector<std::uint64_t>& histogram,
    double probability) {
  std::uint64_t group_count = 0;
  for (std::uint64_t count : histogram) group_count += count;
  if (group_count == 0U) return 0.0;
  const std::uint64_t rank = static_cast<std::uint64_t>(std::ceil(
    probability * static_cast<double>(group_count)));
  std::uint64_t cumulative = 0;
  for (std::size_t size = 0; size < histogram.size(); ++size) {
    cumulative += histogram[size];
    if (cumulative >= std::max<std::uint64_t>(1U, rank)) {
      return static_cast<double>(size);
    }
  }
  return static_cast<double>(histogram.empty() ? 0U : histogram.size() - 1U);
}

Rcpp::DataFrame decomposition_reuse_stage_frame(
    const DecompositionReuseAggregate& value) {
  const auto& names = decomposition_trace_stage_names();
  Rcpp::CharacterVector stage(names.size());
  Rcpp::NumericVector requests(names.size());
  Rcpp::NumericVector unique_keys(names.size());
  Rcpp::NumericVector reuse(names.size());
  Rcpp::NumericVector ratio(names.size());
  for (std::size_t index = 0; index < names.size(); ++index) {
    stage[index] = names[index];
    requests[index] = static_cast<double>(value.stage_request_count[index]);
    unique_keys[index] =
      static_cast<double>(value.stage_unique_key_count[index]);
    reuse[index] = static_cast<double>(value.stage_reuse_count[index]);
    ratio[index] = value.stage_request_count[index] > 0U ?
      static_cast<double>(value.stage_reuse_count[index]) /
        static_cast<double>(value.stage_request_count[index]) : 0.0;
  }
  return Rcpp::DataFrame::create(
    Rcpp::Named("stage") = stage,
    Rcpp::Named("request_count") = requests,
    Rcpp::Named("unique_key_count") = unique_keys,
    Rcpp::Named("reuse_count") = reuse,
    Rcpp::Named("reuse_ratio") = ratio,
    Rcpp::Named("stringsAsFactors") = false);
}

Rcpp::DataFrame decomposition_reuse_route_frame(
    const DecompositionReuseAggregate& value) {
  const auto& names = decomposition_trace_route_names();
  Rcpp::CharacterVector route(names.size());
  Rcpp::NumericVector requests(names.size());
  Rcpp::NumericVector unique_keys(names.size());
  Rcpp::NumericVector reuse(names.size());
  Rcpp::NumericVector ratio(names.size());
  for (std::size_t index = 0; index < names.size(); ++index) {
    route[index] = names[index];
    requests[index] = static_cast<double>(value.route_request_count[index]);
    unique_keys[index] =
      static_cast<double>(value.route_unique_key_count[index]);
    reuse[index] = static_cast<double>(value.route_reuse_count[index]);
    ratio[index] = value.route_request_count[index] > 0U ?
      static_cast<double>(value.route_reuse_count[index]) /
        static_cast<double>(value.route_request_count[index]) : 0.0;
  }
  return Rcpp::DataFrame::create(
    Rcpp::Named("route") = route,
    Rcpp::Named("request_count") = requests,
    Rcpp::Named("unique_key_count") = unique_keys,
    Rcpp::Named("reuse_count") = reuse,
    Rcpp::Named("reuse_ratio") = ratio,
    Rcpp::Named("stringsAsFactors") = false);
}

Rcpp::DataFrame decomposition_reuse_shape_frame(
    const std::map<std::pair<int, int>, DecompositionReuseAggregate>& values) {
  const int count = static_cast<int>(values.size());
  Rcpp::IntegerVector coefficient_dim(count);
  Rcpp::IntegerVector penalty_count(count);
  Rcpp::NumericVector batches(count);
  Rcpp::NumericVector requests(count);
  Rcpp::NumericVector unique_keys(count);
  Rcpp::NumericVector reuse(count);
  Rcpp::NumericVector ratio(count);
  int index = 0;
  for (const auto& entry : values) {
    coefficient_dim[index] = entry.first.first;
    penalty_count[index] = entry.first.second;
    batches[index] = static_cast<double>(entry.second.batch_count);
    requests[index] = static_cast<double>(entry.second.request_count);
    unique_keys[index] = static_cast<double>(entry.second.unique_key_count);
    reuse[index] = static_cast<double>(entry.second.reuse_count);
    ratio[index] = entry.second.request_count > 0U ?
      static_cast<double>(entry.second.reuse_count) /
        static_cast<double>(entry.second.request_count) : 0.0;
    ++index;
  }
  return Rcpp::DataFrame::create(
    Rcpp::Named("coefficient_dim") = coefficient_dim,
    Rcpp::Named("penalty_count") = penalty_count,
    Rcpp::Named("batch_count") = batches,
    Rcpp::Named("request_count") = requests,
    Rcpp::Named("unique_key_count") = unique_keys,
    Rcpp::Named("reuse_count") = reuse,
    Rcpp::Named("reuse_ratio") = ratio,
    Rcpp::Named("stringsAsFactors") = false);
}

Rcpp::DataFrame decomposition_iteration_reuse_frame(
    std::vector<DecompositionIterationReuseRow> values) {
  std::sort(values.begin(), values.end(), [](const auto& left,
                                              const auto& right) {
    return std::tie(left.prepared_key, left.stability_replay, left.iteration) <
      std::tie(right.prepared_key, right.stability_replay, right.iteration);
  });
  const int count = static_cast<int>(values.size());
  Rcpp::CharacterVector prepared_key(count);
  Rcpp::IntegerVector coefficient_dim(count);
  Rcpp::IntegerVector penalty_count(count);
  Rcpp::IntegerVector iteration(count);
  Rcpp::LogicalVector stability_replay(count);
  Rcpp::NumericVector requests(count);
  Rcpp::NumericVector unique_keys(count);
  Rcpp::NumericVector reuse(count);
  for (int index = 0; index < count; ++index) {
    const DecompositionIterationReuseRow& row =
      values[static_cast<std::size_t>(index)];
    prepared_key[index] = row.prepared_key;
    coefficient_dim[index] = row.coefficient_dim;
    penalty_count[index] = row.penalty_count;
    iteration[index] = row.iteration;
    stability_replay[index] = row.stability_replay;
    requests[index] = static_cast<double>(row.request_count);
    unique_keys[index] = static_cast<double>(row.unique_key_count);
    reuse[index] = static_cast<double>(row.reuse_count);
  }
  return Rcpp::DataFrame::create(
    Rcpp::Named("prepared_s_key") = prepared_key,
    Rcpp::Named("coefficient_dim") = coefficient_dim,
    Rcpp::Named("penalty_count") = penalty_count,
    Rcpp::Named("iteration") = iteration,
    Rcpp::Named("stability_replay") = stability_replay,
    Rcpp::Named("request_count") = requests,
    Rcpp::Named("unique_key_count") = unique_keys,
    Rcpp::Named("reuse_count") = reuse,
    Rcpp::Named("stringsAsFactors") = false);
}

struct PrefillAttributionSummary {
  int conditioning_group_count = 0;
  int window_count = 0;
  int optimizer_boundary_count = 0;
  int optimizer_setup_count = 0;
  int target_optimization_count = 0;
  int single_penalty_target_count = 0;
  int multi_penalty_target_count = 0;
  int unique_target_key_count = 0;
  int consumed_unique_target_key_count = 0;
  int unconsumed_unique_target_key_count = 0;
  int singleton_skipped_request_count = 0;
  int singleton_skipped_target_count = 0;
  double optimizer_host_ms = 0.0;
  double batch_wall_ms = 0.0;
};

PrefillAttributionSummary summarize_prefill_attribution(
    const OneCallDiagnostics& diagnostics) {
  PrefillAttributionSummary value;
  value.window_count = static_cast<int>(diagnostics.prefill_batches.size());
  for (const PrefillBatchDiagnostic& batch : diagnostics.prefill_batches) {
    require(batch.level >= 1 &&
              (batch.penalty_class == "single" ||
               batch.penalty_class == "multi") &&
              batch.window_id > 0 && batch.conditioning_group_count > 0 &&
              batch.optimizer_setup_count >= 0 &&
              batch.target_optimization_count >= 0 &&
              batch.singleton_skipped_request_count >= 0 &&
              batch.singleton_skipped_target_count >= 0 &&
              batch.optimizer_host_ms >= 0.0 && batch.batch_wall_ms >= 0.0 &&
              batch.target_optimization_count ==
                static_cast<int>(batch.optimized_target_keys.size()),
            "prefill batch attribution is malformed");
    value.conditioning_group_count += batch.conditioning_group_count;
    value.optimizer_boundary_count += batch.optimizer_setup_count > 0 ? 1 : 0;
    value.optimizer_setup_count += batch.optimizer_setup_count;
    value.target_optimization_count += batch.target_optimization_count;
    if (batch.penalty_class == "single") {
      value.single_penalty_target_count += batch.target_optimization_count;
    } else {
      value.multi_penalty_target_count += batch.target_optimization_count;
    }
    value.singleton_skipped_request_count +=
      batch.singleton_skipped_request_count;
    value.singleton_skipped_target_count +=
      batch.singleton_skipped_target_count;
    value.optimizer_host_ms += batch.optimizer_host_ms;
    value.batch_wall_ms += batch.batch_wall_ms;
    for (const std::string& key : batch.optimized_target_keys) {
      require(diagnostics.prefill_target_keys.count(key) == 1U,
              "prefill batch target key was not published globally");
    }
  }
  value.unique_target_key_count = static_cast<int>(
    diagnostics.prefill_target_keys.size());
  for (const std::string& key : diagnostics.prefill_target_keys) {
    if (diagnostics.unique_target_keys.count(key) == 1U) {
      value.consumed_unique_target_key_count += 1;
    }
  }
  value.unconsumed_unique_target_key_count =
    value.unique_target_key_count - value.consumed_unique_target_key_count;
  require(value.target_optimization_count >= value.unique_target_key_count &&
            value.unique_target_key_count >=
              value.consumed_unique_target_key_count &&
            value.singleton_skipped_request_count ==
              value.singleton_skipped_target_count,
          "prefill aggregate attribution is malformed");
  return value;
}

Rcpp::DataFrame prefill_batch_diagnostics_frame(
    const std::vector<PrefillBatchDiagnostic>& batches,
    const std::unordered_set<std::string>& consumed_target_keys) {
  const int count = static_cast<int>(batches.size());
  Rcpp::IntegerVector level(count), window_id(count), conditioning_groups(count),
    optimizer_setups(count), target_optimizations(count), unique_targets(count),
    consumed_targets(count), unconsumed_targets(count),
    singleton_skipped_requests(count), singleton_skipped_targets(count);
  Rcpp::CharacterVector penalty_class(count);
  Rcpp::NumericVector optimizer_host_ms(count), batch_wall_ms(count);
  for (int index = 0; index < count; ++index) {
    const PrefillBatchDiagnostic& batch =
      batches[static_cast<std::size_t>(index)];
    const std::unordered_set<std::string> unique(
      batch.optimized_target_keys.begin(), batch.optimized_target_keys.end());
    require(unique.size() == batch.optimized_target_keys.size(),
            "prefill batch optimized a duplicate TargetKey");
    int consumed = 0;
    for (const std::string& key : unique) {
      consumed += consumed_target_keys.count(key) == 1U ? 1 : 0;
    }
    level[index] = batch.level;
    penalty_class[index] = batch.penalty_class;
    window_id[index] = batch.window_id;
    conditioning_groups[index] = batch.conditioning_group_count;
    optimizer_setups[index] = batch.optimizer_setup_count;
    target_optimizations[index] = batch.target_optimization_count;
    unique_targets[index] = static_cast<int>(unique.size());
    consumed_targets[index] = consumed;
    unconsumed_targets[index] = static_cast<int>(unique.size()) - consumed;
    singleton_skipped_requests[index] =
      batch.singleton_skipped_request_count;
    singleton_skipped_targets[index] = batch.singleton_skipped_target_count;
    optimizer_host_ms[index] = batch.optimizer_host_ms;
    batch_wall_ms[index] = batch.batch_wall_ms;
  }
  return Rcpp::DataFrame::create(
    Rcpp::Named("level") = level,
    Rcpp::Named("penalty_class") = penalty_class,
    Rcpp::Named("window_id") = window_id,
    Rcpp::Named("conditioning_group_count") = conditioning_groups,
    Rcpp::Named("optimizer_setup_count") = optimizer_setups,
    Rcpp::Named("target_optimization_count") = target_optimizations,
    Rcpp::Named("unique_target_key_count") = unique_targets,
    Rcpp::Named("consumed_unique_target_key_count") = consumed_targets,
    Rcpp::Named("unconsumed_unique_target_key_count") = unconsumed_targets,
    Rcpp::Named("singleton_skipped_request_count") =
      singleton_skipped_requests,
    Rcpp::Named("singleton_skipped_target_count") = singleton_skipped_targets,
    Rcpp::Named("optimizer_host_ms") = optimizer_host_ms,
    Rcpp::Named("batch_wall_ms") = batch_wall_ms,
    Rcpp::Named("stringsAsFactors") = false);
}

struct SinglePenaltyPrefillGroup {
  std::shared_ptr<NativeSetupContext> context;
  std::vector<int> task_indices;
};

struct SinglePenaltyPrefillRequest {
  std::shared_ptr<NativeSetupContext> context;
  std::vector<int> targets;
  std::vector<std::string> base_target_keys;
  std::vector<std::string> optimizer_state_keys;
};

struct SinglePenaltyPrefillWindowWork {
  std::chrono::steady_clock::time_point started;
  PrefillBatchDiagnostic batch;
  std::vector<std::shared_ptr<NativeSetupContext>> contexts;
  std::vector<SinglePenaltyPrefillRequest> requests;
  std::vector<SinglePenaltyGcvCudaOwnedInput> inputs;
  double preparation_ms = 0.0;
};

SinglePenaltyPrefillWindowWork prepare_single_penalty_prefill_window(
    const Rcpp::NumericMatrix& data,
    const LayerPlan& plan,
    const std::vector<SinglePenaltyPrefillGroup>& groups,
    PrefillBatchDiagnostic batch,
    OneCallDiagnostics* diagnostics) {
  SinglePenaltyPrefillWindowWork work;
  work.started = std::chrono::steady_clock::now();
  work.batch = std::move(batch);
  require(diagnostics != nullptr &&
            work.batch.level == plan.level &&
            work.batch.penalty_class == "single" &&
            work.batch.conditioning_group_count ==
              static_cast<int>(groups.size()),
          "single-penalty prefill diagnostics are malformed");
  work.contexts.reserve(groups.size());
  work.requests.reserve(groups.size());
  work.inputs.reserve(groups.size());

  for (const SinglePenaltyPrefillGroup& group : groups) {
    const std::shared_ptr<NativeSetupContext>& context = group.context;
    if (!context || context->penalty_count != 1 ||
        group.task_indices.empty()) {
      continue;
    }
    work.contexts.push_back(context);
    require(context->single_penalty != nullptr,
            "single-penalty prefill geometry is missing");
    std::set<int> target_set;
    for (int task_index : group.task_indices) {
      const LayerCiTask& task =
        plan.tasks[static_cast<std::size_t>(task_index)];
      target_set.insert(task.orientation_x);
      target_set.insert(task.orientation_y);
    }

    SinglePenaltyPrefillRequest request;
    request.context = context;
    for (int target : target_set) {
      const std::string base_key = target_key(context->prepared_key, target);
      const std::string optimizer_key =
        "schema=full-cuda-ci-target-optimizer-state-v1\ntarget=" +
        base_key + "\n";
      CachedTargetState cached;
      if (!target_state_cache().lookup(
            optimizer_key, context->dataset_key, 1, &cached,
            diagnostics)) {
        request.targets.push_back(target);
        request.base_target_keys.push_back(base_key);
        request.optimizer_state_keys.push_back(optimizer_key);
      }
    }
    if (request.targets.empty()) continue;

    const SinglePenaltyGeometry& geometry = *context->single_penalty;
    Rcpp::NumericMatrix X = context->setup["X"];
    SinglePenaltyGcvCudaOwnedInput input;
    input.X.assign(X.begin(), X.end());
    input.Y.resize(
      static_cast<std::size_t>(context->n) * request.targets.size());
    input.target_ids.resize(request.targets.size());
    for (std::size_t index = 0; index < request.targets.size(); ++index) {
      const int target = request.targets[index];
      std::copy(
        data.begin() + static_cast<std::ptrdiff_t>(target) * context->n,
        data.begin() + static_cast<std::ptrdiff_t>(target + 1) * context->n,
        input.Y.begin() + static_cast<std::ptrdiff_t>(index) * context->n);
      input.target_ids[index] = target + 1;
    }
    input.rhs_transform = geometry.rhs_transform;
    input.eigenvalues = geometry.eigenvalues;
    input.magic_qr_packed = geometry.magic_qr_packed;
    input.magic_tau = geometry.magic_tau;
    input.magic_r = geometry.magic_r;
    input.magic_penalty_root = geometry.magic_penalty_root;
    input.magic_penalty_matrix = geometry.magic_penalty_matrix;
    input.n = context->n;
    input.coefficient_dim = context->coefficient_dim;
    input.target_count = static_cast<int>(request.targets.size());
    input.penalty_rank = geometry.penalty_rank;
    input.initial_sp = geometry.initial_sp;
    input.materialize_grid = false;
    input.keep_transcript = false;
    work.inputs.push_back(std::move(input));
    work.requests.push_back(std::move(request));
  }
  work.preparation_ms = elapsed_ms(work.started);
  return work;
}

void materialize_single_penalty_prefill_window(
    SinglePenaltyPrefillWindowWork* work,
    NativeSetupCache* setup_cache,
    OneCallDiagnostics* diagnostics) {
  require(work != nullptr && setup_cache != nullptr &&
            diagnostics != nullptr,
          "deferred single-penalty prefill state is malformed");
  const auto started = std::chrono::steady_clock::now();
  for (const std::shared_ptr<NativeSetupContext>& context : work->contexts) {
    setup_cache->materialize_device(context);
  }
  const double device_prepare_ms = elapsed_ms(started);
  work->preparation_ms += device_prepare_ms;
  diagnostics->setup_optimizer_pipeline_device_prepare_ms +=
    device_prepare_ms;
}

SinglePenaltyGcvCudaMultiResult optimize_single_penalty_prefill_window(
    std::vector<SinglePenaltyGcvCudaOwnedInput> inputs) {
  require(!inputs.empty(), "single-penalty prefill optimizer work is empty");
  return single_penalty_gcv_cuda_multi(
    inputs, std::min(
      kSinglePenaltyGcvMaximumConcurrentSetups,
      static_cast<int>(inputs.size())));
}

void complete_single_penalty_prefill_window(
    SinglePenaltyPrefillWindowWork* work,
    SinglePenaltyGcvCudaMultiResult optimization,
    OneCallDiagnostics* diagnostics) {
  require(work != nullptr && diagnostics != nullptr &&
            !work->requests.empty(),
          "single-penalty prefill completion state is malformed");
  const auto completion_started = std::chrono::steady_clock::now();
  require(optimization.setups.size() == work->requests.size() &&
            optimization.diagnostics.setup_count ==
              static_cast<int>(work->requests.size()) &&
            optimization.diagnostics.max_host_calls_in_flight > 0,
          "single-penalty cross-setup CUDA optimizer coverage changed");

  int optimized_targets = 0;
  for (std::size_t setup_index = 0; setup_index < work->requests.size();
       ++setup_index) {
    const SinglePenaltyPrefillRequest& request =
      work->requests[setup_index];
    const SinglePenaltyGcvCudaResult& setup =
      optimization.setups[setup_index];
    require(setup.target_count == static_cast<int>(request.targets.size()) &&
              setup.diagnostics.legacy_mgcv_target_calls == 0 &&
              setup.diagnostics.cpu_score_count == 0 &&
              setup.diagnostics.cpu_optimizer_count == 0 &&
              setup.diagnostics.fallback_count == 0 &&
              setup.diagnostics.optimizer_target_coverage_complete,
            "single-penalty cross-setup CUDA optimizer authority gate failed");
    accumulate_single_penalty_optimizer_diagnostics(
      setup.diagnostics, diagnostics);
    const SinglePenaltyGeometry& geometry = *request.context->single_penalty;
    for (std::size_t target_index = 0;
         target_index < request.targets.size(); ++target_index) {
      const SinglePenaltyGcvOptimizerResult& result =
        setup.targets[target_index];
      const bool accepted =
        result.termination == static_cast<int>(
          SinglePenaltyGcvTermination::ScoreAndGradient) ||
        result.termination == static_cast<int>(
          SinglePenaltyGcvTermination::StepHalvingExhausted) ||
        result.termination == static_cast<int>(
          SinglePenaltyGcvTermination::FlatObjective);
      require(accepted && std::isfinite(result.sp) && result.sp > 0.0,
              "single-penalty cross-setup CUDA optimizer failed closed");
      CachedTargetState state;
      state.selected_sp.push_back(result.sp);
      state.planned_route = route_from_condition(
        single_penalty_condition(geometry, result.sp), true);
      target_state_cache().insert(
        request.optimizer_state_keys[target_index],
        request.context->dataset_key, state, diagnostics);
      const std::string& base_key =
        request.base_target_keys[target_index];
      work->batch.optimized_target_keys.push_back(base_key);
      diagnostics->prefill_target_keys.insert(base_key);
    }
    optimized_targets += static_cast<int>(request.targets.size());
  }
  diagnostics->cuda_single_penalty_target_count += optimized_targets;
  diagnostics->cuda_single_penalty_optimizer_call_count += 1;
  diagnostics->cuda_optimizer_host_boundary_count += 1;
  diagnostics->cuda_single_penalty_optimizer_host_ms +=
    optimization.diagnostics.wall_host_ms;
  const double logical_boundary_ms =
    work->preparation_ms + optimization.diagnostics.wall_host_ms +
      elapsed_ms(completion_started);
  diagnostics->cuda_optimizer_host_ms += logical_boundary_ms;
  work->batch.optimizer_setup_count =
    static_cast<int>(work->requests.size());
  work->batch.target_optimization_count = optimized_targets;
  work->batch.optimizer_host_ms = optimization.diagnostics.wall_host_ms;
  work->batch.batch_wall_ms = logical_boundary_ms;
}

void prefill_single_penalty_target_states(
    const Rcpp::NumericMatrix& data,
    const LayerPlan& plan,
    const std::vector<SinglePenaltyPrefillGroup>& groups,
    PrefillBatchDiagnostic* batch,
    OneCallDiagnostics* diagnostics) {
  require(batch != nullptr,
          "single-penalty prefill output batch is missing");
  SinglePenaltyPrefillWindowWork work = prepare_single_penalty_prefill_window(
    data, plan, groups, std::move(*batch), diagnostics);
  if (work.inputs.empty()) {
    work.batch.batch_wall_ms = work.preparation_ms;
    *batch = std::move(work.batch);
    return;
  }
  SinglePenaltyGcvCudaMultiResult optimization =
    optimize_single_penalty_prefill_window(std::move(work.inputs));
  complete_single_penalty_prefill_window(
    &work, std::move(optimization), diagnostics);
  *batch = std::move(work.batch);
}

struct PendingSinglePenaltyPrefillWindow {
  SinglePenaltyPrefillWindowWork work;
  std::future<SinglePenaltyGcvCudaMultiResult> optimization;
};

void prefill_single_penalty_level_target_states(
    const Rcpp::NumericMatrix& data,
    const LayerPlan& plan,
    NativeSetupCache* setup_cache,
    OneCallDiagnostics* diagnostics) {
  require(setup_cache != nullptr,
          "single-penalty level prefill setup cache is missing");
  std::map<std::string, std::vector<int>> tasks_by_conditioning;
  for (std::size_t task_index = 0; task_index < plan.tasks.size();
       ++task_index) {
    const LayerCiTask& task = plan.tasks[task_index];
    if (task.conditioning_set.empty() ||
        task.conditioning_set.size() > 2U) {
      continue;
    }
    tasks_by_conditioning[conditioning_key(task.conditioning_set)].push_back(
      static_cast<int>(task_index));
  }

  std::vector<SinglePenaltyPrefillGroup> window;
  window.reserve(kSinglePenaltyLevelPrefillGroupWindow);
  int window_id = 0;
  const auto pipeline_started = std::chrono::steady_clock::now();
  std::unique_ptr<PendingSinglePenaltyPrefillWindow> pending;
  const auto finish_pending = [&]() {
    if (!pending) return;
    const auto wait_started = std::chrono::steady_clock::now();
    SinglePenaltyGcvCudaMultiResult optimization =
      pending->optimization.get();
    const double wait_ms = elapsed_ms(wait_started);
    diagnostics->setup_optimizer_pipeline_wait_ms += wait_ms;
    diagnostics->setup_optimizer_pipeline_overlap_ms += std::max(
      0.0, optimization.diagnostics.wall_host_ms - wait_ms);
    complete_single_penalty_prefill_window(
      &pending->work, std::move(optimization), diagnostics);
    diagnostics->prefill_batches.push_back(
      std::move(pending->work.batch));
    pending.reset();
  };
  const auto run_window = [&]() {
    require(!window.empty(),
            "single-penalty level prefill window is empty");
    PrefillBatchDiagnostic batch;
    batch.level = plan.level;
    batch.penalty_class = "single";
    batch.window_id = ++window_id;
    batch.conditioning_group_count = static_cast<int>(window.size());
    if (!diagnostics->setup_optimizer_pipeline_enabled) {
      diagnostics->prefill_batches.push_back(std::move(batch));
      prefill_single_penalty_target_states(
        data, plan, window, &diagnostics->prefill_batches.back(), diagnostics);
      window.clear();
      return;
    }
    SinglePenaltyPrefillWindowWork work =
      prepare_single_penalty_prefill_window(
        data, plan, window, std::move(batch), diagnostics);
    diagnostics->setup_optimizer_pipeline_prepare_ms += work.preparation_ms;
    finish_pending();
    materialize_single_penalty_prefill_window(
      &work, setup_cache, diagnostics);
    if (work.inputs.empty()) {
      work.batch.batch_wall_ms = work.preparation_ms;
      diagnostics->prefill_batches.push_back(std::move(work.batch));
      window.clear();
      return;
    }
    std::vector<SinglePenaltyGcvCudaOwnedInput> inputs =
      std::move(work.inputs);
    std::future<SinglePenaltyGcvCudaMultiResult> future = std::async(
      std::launch::async,
      [inputs = std::move(inputs)]() mutable {
        return optimize_single_penalty_prefill_window(std::move(inputs));
      });
    pending.reset(new PendingSinglePenaltyPrefillWindow{
      std::move(work), std::move(future)});
    diagnostics->setup_optimizer_pipeline_window_count += 1;
    diagnostics->setup_optimizer_pipeline_peak_pending_count = 1;
    delay_setup_optimizer_pipeline_producer(diagnostics);
    window.clear();
  };
  for (const auto& entry : tasks_by_conditioning) {
    require(!entry.second.empty(),
            "single-penalty level prefill group is empty");
    const LayerCiTask& representative =
      plan.tasks[static_cast<std::size_t>(entry.second.front())];
    std::shared_ptr<NativeSetupContext> context =
      setup_cache->get(representative.conditioning_set);
    require(context->penalty_count == 1,
            "single-penalty level prefill shape changed");
    window.push_back(SinglePenaltyPrefillGroup{context, entry.second});
    if (window.size() == kSinglePenaltyLevelPrefillGroupWindow) {
      run_window();
    }
  }
  if (!window.empty()) {
    run_window();
  }
  finish_pending();
  if (diagnostics->setup_optimizer_pipeline_enabled) {
    diagnostics->setup_optimizer_pipeline_level_wall_ms +=
      elapsed_ms(pipeline_started);
  }
}

struct MultiPenaltyPrefillRequest {
  std::shared_ptr<NativeSetupContext> context;
  std::vector<int> targets;
  std::vector<std::string> base_target_keys;
  std::vector<std::string> optimizer_state_keys;
};

struct MultiPenaltyPrefillWindowWork {
  std::chrono::steady_clock::time_point started;
  PrefillBatchDiagnostic batch;
  std::vector<std::shared_ptr<NativeSetupContext>> contexts;
  std::vector<MultiPenaltyPrefillRequest> metadata;
  std::vector<MultiPenaltyGcvCapacityRequest> requests;
  int maximum_concurrency = 0;
  double preparation_ms = 0.0;
};

MultiPenaltyPrefillWindowWork prepare_multi_penalty_prefill_window(
    const Rcpp::NumericMatrix& data,
    const LayerPlan& plan,
    const std::vector<SinglePenaltyPrefillGroup>& groups,
    PrefillBatchDiagnostic batch,
    OneCallDiagnostics* diagnostics,
    bool defer_device_handles = false) {
  MultiPenaltyPrefillWindowWork work;
  work.started = std::chrono::steady_clock::now();
  work.batch = std::move(batch);
  require(diagnostics != nullptr &&
            work.batch.level == plan.level &&
            work.batch.penalty_class == "multi" &&
            work.batch.conditioning_group_count ==
              static_cast<int>(groups.size()),
          "multi-penalty prefill diagnostics are malformed");
  work.contexts.reserve(groups.size());
  work.metadata.reserve(groups.size());
  work.requests.reserve(groups.size());

  for (const SinglePenaltyPrefillGroup& group : groups) {
    const std::shared_ptr<NativeSetupContext>& context = group.context;
    if (!context || context->penalty_count <= 1 ||
        group.task_indices.empty()) {
      continue;
    }
    work.contexts.push_back(context);
    std::set<int> target_set;
    for (int task_index : group.task_indices) {
      const LayerCiTask& task =
        plan.tasks[static_cast<std::size_t>(task_index)];
      target_set.insert(task.orientation_x);
      target_set.insert(task.orientation_y);
    }

    MultiPenaltyPrefillRequest request;
    request.context = context;
    MultiPenaltyGcvCapacityRequest input;
    std::vector<std::string> base_target_keys;
    for (int target : target_set) {
      const std::string base_key = target_key(context->prepared_key, target);
      const std::string optimizer_key =
        "schema=full-cuda-ci-target-optimizer-state-v1\ntarget=" +
        base_key + "\n";
      CachedTargetState cached;
      if (!target_state_cache().lookup(
            optimizer_key, context->dataset_key, context->penalty_count,
            &cached, diagnostics)) {
        request.targets.push_back(target);
        request.optimizer_state_keys.push_back(optimizer_key);
        base_target_keys.push_back(base_key);
      }
    }
    if (request.targets.size() < 2U) {
      if (request.targets.size() == 1U) {
        work.batch.singleton_skipped_request_count += 1;
        work.batch.singleton_skipped_target_count += 1;
      }
      continue;
    }

    if (!defer_device_handles) {
      ensure_multi_penalty_handle(
        context, static_cast<int>(request.targets.size()), diagnostics);
    }
    request.base_target_keys = base_target_keys;
    input.prepared = defer_device_handles ? nullptr : context->multi_penalty;
    input.n = context->n;
    input.target_count = static_cast<int>(request.targets.size());
    input.target_keys = std::move(base_target_keys);
    input.control.decomposition_trace_capacity_per_target =
      diagnostics
        ->cuda_multi_penalty_decomposition_trace_capacity_per_target;
    input.Y.resize(
      static_cast<std::size_t>(context->n) * request.targets.size());
    for (std::size_t index = 0; index < request.targets.size(); ++index) {
      const int target = request.targets[index];
      std::copy(
        data.begin() + static_cast<std::ptrdiff_t>(target) * context->n,
        data.begin() + static_cast<std::ptrdiff_t>(target + 1) * context->n,
        input.Y.begin() + static_cast<std::ptrdiff_t>(index) * context->n);
    }
    work.metadata.push_back(std::move(request));
    work.requests.push_back(std::move(input));
  }

  if (!work.requests.empty()) {
    work.maximum_concurrency = kMultiPenaltyGcvMaximumConcurrentSetups;
    for (const MultiPenaltyPrefillRequest& request : work.metadata) {
      work.maximum_concurrency = std::min(
        work.maximum_concurrency,
        multi_penalty_gcv_capacity_max_concurrent_setups(
          request.context->coefficient_dim,
          request.context->penalty_count));
    }
    work.maximum_concurrency = std::min(
      work.maximum_concurrency, static_cast<int>(work.metadata.size()));
  }
  work.preparation_ms = elapsed_ms(work.started);
  return work;
}

void materialize_multi_penalty_prefill_window(
    MultiPenaltyPrefillWindowWork* work,
    NativeSetupCache* setup_cache,
    OneCallDiagnostics* diagnostics) {
  require(work != nullptr && setup_cache != nullptr &&
            diagnostics != nullptr &&
            work->metadata.size() == work->requests.size(),
          "deferred multi-penalty prefill state is malformed");
  const auto started = std::chrono::steady_clock::now();
  for (const std::shared_ptr<NativeSetupContext>& context : work->contexts) {
    setup_cache->materialize_device(context);
  }
  for (std::size_t index = 0; index < work->metadata.size(); ++index) {
    MultiPenaltyPrefillRequest& metadata = work->metadata[index];
    MultiPenaltyGcvCapacityRequest& request = work->requests[index];
    require(metadata.context != nullptr && !request.prepared,
            "deferred multi-penalty prepared state changed");
    ensure_multi_penalty_handle(
      metadata.context, static_cast<int>(metadata.targets.size()),
      diagnostics);
    request.prepared = metadata.context->multi_penalty;
  }
  const double device_prepare_ms = elapsed_ms(started);
  work->preparation_ms += device_prepare_ms;
  diagnostics->setup_optimizer_pipeline_device_prepare_ms +=
    device_prepare_ms;
}

MultiPenaltyGcvCapacityMultiResult optimize_multi_penalty_prefill_window(
    std::vector<MultiPenaltyGcvCapacityRequest> requests,
    int maximum_concurrency) {
  require(!requests.empty() && maximum_concurrency > 0,
          "multi-penalty prefill optimizer work is empty");
  return multi_penalty_gcv_capacity_optimize_multi(
    std::move(requests), maximum_concurrency);
}

void release_multi_penalty_prefill_window_handles(
    const std::vector<MultiPenaltyPrefillRequest>& metadata,
    OneCallDiagnostics* diagnostics) {
  for (const MultiPenaltyPrefillRequest& request : metadata) {
    if (request.context && request.context->multi_penalty) {
      try {
        release_multi_penalty_handle(request.context, diagnostics);
      } catch (...) {
      }
    }
  }
}

void complete_multi_penalty_prefill_window(
    MultiPenaltyPrefillWindowWork* work,
    MultiPenaltyGcvCapacityMultiResult optimization,
    OneCallDiagnostics* diagnostics) {
  require(work != nullptr && diagnostics != nullptr &&
            !work->metadata.empty(),
          "multi-penalty prefill completion state is malformed");
  const auto completion_started = std::chrono::steady_clock::now();
  try {
    require(optimization.setups.size() == work->metadata.size() &&
              optimization.diagnostics.setup_count ==
                static_cast<int>(work->metadata.size()) &&
              optimization.diagnostics.max_host_calls_in_flight > 0,
            "multi-penalty cross-setup CUDA optimizer coverage changed");
    for (std::size_t setup_index = 0; setup_index < work->metadata.size();
         ++setup_index) {
      const MultiPenaltyPrefillRequest& request =
        work->metadata[setup_index];
      const MultiPenaltyGcvCudaOptimization& result =
        optimization.setups[setup_index];
      const int target_count = static_cast<int>(request.targets.size());
      const int penalty_count = request.context->penalty_count;
      diagnostics->cuda_multi_penalty_optimizer_setup_count += 1;
      diagnostics->cuda_optimizer_kernel_launch_count +=
        result.diagnostics.cuda_qt_y_kernel_launch_count +
        result.diagnostics.cuda_optimizer_kernel_launch_count +
        result.diagnostics.cuda_stability_replay_kernel_launch_count +
        result.diagnostics.cuda_stability_merge_kernel_launch_count;
      require(result.target_count == target_count &&
                result.penalty_count == penalty_count &&
                result.selected_log_sp.size() ==
                  static_cast<std::size_t>(penalty_count) * target_count &&
                result.diagnostics.cpu_objective_count == 0 &&
                result.diagnostics.cpu_optimizer_count == 0 &&
                result.diagnostics.cpu_multi_penalty_solve_count == 0 &&
                result.diagnostics.fallback_count == 0 &&
                result.diagnostics.cuda_error_count == 0 &&
                result.diagnostics.true_batched_kernel &&
                result.diagnostics.independent_target_states,
              "multi-penalty cross-setup CUDA optimizer authority gate failed");
      accumulate_multi_penalty_optimizer_diagnostics(
        result, request.context->prepared_key, diagnostics);
      for (int target_index = 0; target_index < target_count; ++target_index) {
        require(result.optimizer_status[
                  static_cast<std::size_t>(target_index)] == 0,
                "multi-penalty cross-setup optimizer returned an error");
        CachedTargetState state;
        state.selected_sp.resize(static_cast<std::size_t>(penalty_count));
        for (int penalty = 0; penalty < penalty_count; ++penalty) {
          const double sp = std::exp(result.selected_log_sp[
            static_cast<std::size_t>(penalty_count) * target_index + penalty]);
          require(std::isfinite(sp) && sp > 0.0,
                  "multi-penalty cross-setup selected sp is non-finite");
          state.selected_sp[static_cast<std::size_t>(penalty)] = sp;
        }
        state.planned_route = route_from_condition(
          result.condition[static_cast<std::size_t>(target_index)],
          result.numerical_rank[static_cast<std::size_t>(target_index)] ==
            request.context->coefficient_dim);
        target_state_cache().insert(
          request.optimizer_state_keys[static_cast<std::size_t>(target_index)],
          request.context->dataset_key, state, diagnostics);
        const std::string& base_key = request.base_target_keys[
          static_cast<std::size_t>(target_index)];
        work->batch.optimized_target_keys.push_back(base_key);
        diagnostics->prefill_target_keys.insert(base_key);
      }
      diagnostics->cuda_multi_penalty_target_count += target_count;
      release_multi_penalty_handle(request.context, diagnostics);
    }
  } catch (...) {
    release_multi_penalty_prefill_window_handles(work->metadata, diagnostics);
    throw;
  }
  const double logical_boundary_ms =
    work->preparation_ms + optimization.diagnostics.wall_host_ms +
      elapsed_ms(completion_started);
  diagnostics->cuda_multi_penalty_optimizer_call_count += 1;
  diagnostics->cuda_optimizer_host_boundary_count += 1;
  diagnostics->cuda_multi_penalty_optimizer_host_ms +=
    optimization.diagnostics.wall_host_ms;
  diagnostics->cuda_optimizer_host_ms += logical_boundary_ms;
  work->batch.optimizer_setup_count = static_cast<int>(work->metadata.size());
  work->batch.target_optimization_count = static_cast<int>(
    work->batch.optimized_target_keys.size());
  work->batch.optimizer_host_ms = optimization.diagnostics.wall_host_ms;
  work->batch.batch_wall_ms = logical_boundary_ms;
}

void prefill_multi_penalty_target_states(
    const Rcpp::NumericMatrix& data,
    const LayerPlan& plan,
    const std::vector<SinglePenaltyPrefillGroup>& groups,
    PrefillBatchDiagnostic* batch,
    OneCallDiagnostics* diagnostics) {
  require(batch != nullptr,
          "multi-penalty prefill output batch is missing");
  MultiPenaltyPrefillWindowWork work = prepare_multi_penalty_prefill_window(
    data, plan, groups, std::move(*batch), diagnostics);
  if (work.requests.empty()) {
    work.batch.batch_wall_ms = work.preparation_ms;
    *batch = std::move(work.batch);
    return;
  }
  MultiPenaltyGcvCapacityMultiResult optimization;
  try {
    optimization = optimize_multi_penalty_prefill_window(
      std::move(work.requests), work.maximum_concurrency);
  } catch (...) {
    release_multi_penalty_prefill_window_handles(work.metadata, diagnostics);
    throw;
  }
  complete_multi_penalty_prefill_window(
    &work, std::move(optimization), diagnostics);
  *batch = std::move(work.batch);
}

struct PendingMultiPenaltyPrefillWindow {
  MultiPenaltyPrefillWindowWork work;
  std::future<MultiPenaltyGcvCapacityMultiResult> optimization;
};

void prefill_multi_penalty_level_target_states(
    const Rcpp::NumericMatrix& data,
    const LayerPlan& plan,
    NativeSetupCache* setup_cache,
    OneCallDiagnostics* diagnostics) {
  require(setup_cache != nullptr,
          "multi-penalty level prefill setup cache is missing");
  std::map<std::string, std::vector<int>> tasks_by_conditioning;
  for (std::size_t task_index = 0; task_index < plan.tasks.size();
       ++task_index) {
    const LayerCiTask& task = plan.tasks[task_index];
    if (task.conditioning_set.size() <= 2U) continue;
    tasks_by_conditioning[conditioning_key(task.conditioning_set)].push_back(
      static_cast<int>(task_index));
  }

  std::vector<SinglePenaltyPrefillGroup> window;
  window.reserve(
    static_cast<std::size_t>(kMultiPenaltyGcvMaximumConcurrentSetups));
  std::size_t window_capacity =
    static_cast<std::size_t>(kMultiPenaltyGcvMaximumConcurrentSetups);
  int window_id = 0;
  const auto pipeline_started = std::chrono::steady_clock::now();
  std::unique_ptr<PendingMultiPenaltyPrefillWindow> pending;
  const auto finish_pending = [&]() {
    if (!pending) return;
    const auto wait_started = std::chrono::steady_clock::now();
    MultiPenaltyGcvCapacityMultiResult optimization;
    try {
      optimization = pending->optimization.get();
    } catch (...) {
      release_multi_penalty_prefill_window_handles(
        pending->work.metadata, diagnostics);
      pending.reset();
      throw;
    }
    const double wait_ms = elapsed_ms(wait_started);
    diagnostics->setup_optimizer_pipeline_wait_ms += wait_ms;
    diagnostics->setup_optimizer_pipeline_overlap_ms += std::max(
      0.0, optimization.diagnostics.wall_host_ms - wait_ms);
    complete_multi_penalty_prefill_window(
      &pending->work, std::move(optimization), diagnostics);
    diagnostics->prefill_batches.push_back(
      std::move(pending->work.batch));
    pending.reset();
  };
  const auto run_window = [&]() {
    require(!window.empty(),
            "multi-penalty level prefill window is empty");
    PrefillBatchDiagnostic batch;
    batch.level = plan.level;
    batch.penalty_class = "multi";
    batch.window_id = ++window_id;
    batch.conditioning_group_count = static_cast<int>(window.size());
    if (!diagnostics->setup_optimizer_pipeline_enabled) {
      diagnostics->prefill_batches.push_back(std::move(batch));
      prefill_multi_penalty_target_states(
        data, plan, window, &diagnostics->prefill_batches.back(), diagnostics);
      window.clear();
      return;
    }
    MultiPenaltyPrefillWindowWork work =
      prepare_multi_penalty_prefill_window(
        data, plan, window, std::move(batch), diagnostics, true);
    diagnostics->setup_optimizer_pipeline_prepare_ms += work.preparation_ms;
    finish_pending();
    materialize_multi_penalty_prefill_window(
      &work, setup_cache, diagnostics);
    if (work.requests.empty()) {
      work.batch.batch_wall_ms = work.preparation_ms;
      diagnostics->prefill_batches.push_back(std::move(work.batch));
      window.clear();
      return;
    }
    const int maximum_concurrency = work.maximum_concurrency;
    std::vector<MultiPenaltyGcvCapacityRequest> requests =
      std::move(work.requests);
    std::future<MultiPenaltyGcvCapacityMultiResult> future = std::async(
      std::launch::async,
      [requests = std::move(requests), maximum_concurrency]() mutable {
        return optimize_multi_penalty_prefill_window(
          std::move(requests), maximum_concurrency);
      });
    pending.reset(new PendingMultiPenaltyPrefillWindow{
      std::move(work), std::move(future)});
    diagnostics->setup_optimizer_pipeline_window_count += 1;
    diagnostics->setup_optimizer_pipeline_peak_pending_count = 1;
    delay_setup_optimizer_pipeline_producer(diagnostics);
    window.clear();
  };
  for (const auto& entry : tasks_by_conditioning) {
    require(!entry.second.empty(),
            "multi-penalty level prefill group is empty");
    const LayerCiTask& representative =
      plan.tasks[static_cast<std::size_t>(entry.second.front())];
    std::shared_ptr<NativeSetupContext> context =
      setup_cache->get(representative.conditioning_set);
    require(context->penalty_count > 1,
            "multi-penalty level prefill shape changed");
    const std::size_t context_window_capacity = static_cast<std::size_t>(
      multi_penalty_gcv_capacity_max_concurrent_setups(
        context->coefficient_dim, context->penalty_count));
    if (!window.empty() && context_window_capacity != window_capacity) {
      run_window();
    }
    window_capacity = context_window_capacity;
    window.push_back(SinglePenaltyPrefillGroup{context, entry.second});
    if (window.size() == window_capacity) {
      run_window();
    }
  }
  if (!window.empty()) {
    run_window();
  }
  finish_pending();
  if (diagnostics->setup_optimizer_pipeline_enabled) {
    diagnostics->setup_optimizer_pipeline_level_wall_ms +=
      elapsed_ms(pipeline_started);
  }
}

GroupResult execute_group(
    const Rcpp::NumericMatrix& data,
    const std::shared_ptr<NativeSetupContext>& context,
    const LayerPlan& plan,
    const std::vector<int>& task_indices,
    const std::vector<std::uint64_t>& logical_sequence_ids,
    int num_col,
    bool direct,
    const FullCudaCiOneCallMethodOptions& method_options,
    const std::shared_ptr<FullCudaCiMethodResidualCache>& method_residual_cache,
    const std::shared_ptr<FullCudaCiMethodExecutionContext>&
      method_execution_context,
    LegacyRPermutationWorkspace* permutation_workspace,
    OneCallDiagnostics* diagnostics) {
  require(!task_indices.empty() &&
            task_indices.size() == logical_sequence_ids.size(),
          "one-call CUDA group request is empty or misaligned");
  if (!direct) {
    diagnostics->consumed_prepared_keys.insert(context->prepared_key);
  }
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
    if (!direct) {
      diagnostics->unique_target_keys.insert(base_target_keys.back());
    }
  }

  if (!direct) {
    std::vector<std::string> optimizer_state_keys(target_count);
    std::vector<int> missing_positions;
    for (int position = 0; position < target_count; ++position) {
      optimizer_state_keys[static_cast<std::size_t>(position)] =
        "schema=full-cuda-ci-target-optimizer-state-v1\ntarget=" +
        base_target_keys[static_cast<std::size_t>(position)] + "\n";
      CachedTargetState cached;
      if (target_state_cache().lookup(
            optimizer_state_keys[static_cast<std::size_t>(position)],
            context->dataset_key, context->penalty_count, &cached,
            diagnostics)) {
        for (int penalty = 0; penalty < context->penalty_count; ++penalty) {
          selected_sp[
            static_cast<std::size_t>(context->penalty_count) * position +
            penalty] = cached.selected_sp[static_cast<std::size_t>(penalty)];
        }
        planned_routes[static_cast<std::size_t>(position)] =
          cached.planned_route;
      } else {
        missing_positions.push_back(position);
      }
    }

    if (!missing_positions.empty()) {
      const auto optimize_start = std::chrono::steady_clock::now();
      std::vector<int> optimization_positions = missing_positions;
      bool singleton_padding = false;
      if (context->penalty_count > 1 &&
          optimization_positions.size() == 1U) {
        for (int position = 0; position < target_count; ++position) {
          if (position != optimization_positions.front()) {
            optimization_positions.push_back(position);
            singleton_padding = true;
            break;
          }
        }
      }
      const int optimization_count =
        static_cast<int>(optimization_positions.size());
      require(optimization_count ==
                static_cast<int>(missing_positions.size()) +
                  (singleton_padding ? 1 : 0),
              "live optimizer singleton padding accounting changed");
      double optimizer_host_ms = 0.0;
      std::vector<double> optimization_Y(
        static_cast<std::size_t>(context->n) * optimization_count);
      std::vector<std::string> optimization_base_keys;
      optimization_base_keys.reserve(
        static_cast<std::size_t>(optimization_count));
      for (int local = 0; local < optimization_count; ++local) {
        const int position =
          optimization_positions[static_cast<std::size_t>(local)];
        std::copy(
          Y.begin() + static_cast<std::ptrdiff_t>(position) * context->n,
          Y.begin() + static_cast<std::ptrdiff_t>(position + 1) * context->n,
          optimization_Y.begin() +
            static_cast<std::ptrdiff_t>(local) * context->n);
        optimization_base_keys.push_back(
          base_target_keys[static_cast<std::size_t>(position)]);
      }

    if (context->penalty_count == 1) {
      require(context->single_penalty != nullptr,
              "single-penalty CUDA geometry is missing");
      const SinglePenaltyGeometry& geometry = *context->single_penalty;
      Rcpp::NumericMatrix X = context->setup["X"];
      std::vector<int> target_ids(optimization_count);
      for (int index = 0; index < optimization_count; ++index) {
        const int position =
          optimization_positions[static_cast<std::size_t>(index)];
        target_ids[static_cast<std::size_t>(index)] =
          targets[static_cast<std::size_t>(position)] + 1;
      }
      const SinglePenaltyGcvCudaResult optimization =
        single_penalty_gcv_cuda(
          X.begin(), optimization_Y.data(), geometry.rhs_transform.data(),
          geometry.eigenvalues.data(), geometry.magic_qr_packed.data(),
          geometry.magic_tau.data(), geometry.magic_r.data(),
          geometry.magic_penalty_root.data(),
          geometry.magic_penalty_matrix.data(), target_ids.data(),
          context->n, context->coefficient_dim, optimization_count,
          geometry.penalty_rank, geometry.initial_sp, {}, false, false);
      accumulate_single_penalty_optimizer_diagnostics(
        optimization.diagnostics, diagnostics);
      diagnostics->cuda_single_penalty_optimizer_call_count += 1;
      diagnostics->cuda_optimizer_host_boundary_count += 1;
      diagnostics->cuda_single_penalty_optimizer_host_ms +=
        optimization.diagnostics.total_host_ms;
      optimizer_host_ms = optimization.diagnostics.total_host_ms;
      require(optimization.target_count == optimization_count &&
                optimization.diagnostics.legacy_mgcv_target_calls == 0 &&
                optimization.diagnostics.cpu_score_count == 0 &&
                optimization.diagnostics.cpu_optimizer_count == 0 &&
                optimization.diagnostics.fallback_count == 0 &&
                optimization.diagnostics.optimizer_target_coverage_complete,
              "single-penalty live CUDA optimizer authority gate failed");
      for (int local = 0; local < optimization_count; ++local) {
        const int position =
          optimization_positions[static_cast<std::size_t>(local)];
        const SinglePenaltyGcvOptimizerResult& result =
          optimization.targets[static_cast<std::size_t>(local)];
        const bool accepted =
          result.termination == static_cast<int>(
            SinglePenaltyGcvTermination::ScoreAndGradient) ||
          result.termination == static_cast<int>(
            SinglePenaltyGcvTermination::StepHalvingExhausted) ||
          result.termination == static_cast<int>(
            SinglePenaltyGcvTermination::FlatObjective);
        require(accepted && std::isfinite(result.sp) && result.sp > 0.0,
                "single-penalty live CUDA optimizer failed closed");
        selected_sp[static_cast<std::size_t>(position)] = result.sp;
        planned_routes[static_cast<std::size_t>(position)] =
          route_from_condition(
            single_penalty_condition(geometry, result.sp), true);
      }
      diagnostics->cuda_single_penalty_target_count += optimization_count;
    } else {
      ensure_multi_penalty_handle(
        context, optimization_count, diagnostics);
      MultiPenaltyGcvCudaOptimizerControl control;
      control.decomposition_trace_capacity_per_target =
        diagnostics
          ->cuda_multi_penalty_decomposition_trace_capacity_per_target;
      MultiPenaltyGcvCudaOptimization optimization;
      try {
        optimization = multi_penalty_gcv_capacity_optimize_batch(
          context->multi_penalty, optimization_Y.data(), context->n,
          optimization_count, optimization_base_keys, control);
        const MultiPenaltyGcvCudaOptimization& result =
          optimization;
        diagnostics->cuda_multi_penalty_optimizer_setup_count += 1;
        diagnostics->cuda_multi_penalty_optimizer_call_count += 1;
        diagnostics->cuda_optimizer_host_boundary_count += 1;
        diagnostics->cuda_multi_penalty_optimizer_host_ms +=
          result.diagnostics.total_host_ms;
        optimizer_host_ms = result.diagnostics.total_host_ms;
        diagnostics->cuda_optimizer_kernel_launch_count +=
          result.diagnostics.cuda_qt_y_kernel_launch_count +
          result.diagnostics.cuda_optimizer_kernel_launch_count +
          result.diagnostics.cuda_stability_replay_kernel_launch_count +
          result.diagnostics.cuda_stability_merge_kernel_launch_count;
        require(result.target_count == optimization_count &&
                  result.penalty_count == context->penalty_count &&
                  result.selected_log_sp.size() ==
                    static_cast<std::size_t>(context->penalty_count) *
                      optimization_count &&
                  result.diagnostics.cpu_objective_count == 0 &&
                  result.diagnostics.cpu_optimizer_count == 0 &&
                  result.diagnostics.cpu_multi_penalty_solve_count == 0 &&
                  result.diagnostics.fallback_count == 0 &&
                  result.diagnostics.cuda_error_count == 0 &&
                  result.diagnostics.true_batched_kernel &&
                  result.diagnostics.independent_target_states,
                "multi-penalty live CUDA optimizer authority gate failed");
        accumulate_multi_penalty_optimizer_diagnostics(
          result, context->prepared_key, diagnostics);
        for (int local = 0; local < optimization_count; ++local) {
          const int position =
            optimization_positions[static_cast<std::size_t>(local)];
          for (int penalty = 0; penalty < context->penalty_count; ++penalty) {
            const double sp = std::exp(result.selected_log_sp[
              static_cast<std::size_t>(context->penalty_count) * local +
              penalty]);
            require(std::isfinite(sp) && sp > 0,
                    "multi-penalty selected sp is non-finite");
            selected_sp[
              static_cast<std::size_t>(context->penalty_count) * position +
              penalty] = sp;
          }
          require(result.optimizer_status[static_cast<std::size_t>(local)] == 0,
                  "multi-penalty CUDA optimizer returned an error status");
          planned_routes[static_cast<std::size_t>(position)] =
            route_from_condition(
              result.condition[static_cast<std::size_t>(local)],
              result.numerical_rank[static_cast<std::size_t>(local)] ==
                context->coefficient_dim);
        }
        release_multi_penalty_handle(context, diagnostics);
      } catch (...) {
        if (context->multi_penalty) {
          try {
            release_multi_penalty_handle(context, diagnostics);
          } catch (...) {
          }
        }
        throw;
      }
      diagnostics->cuda_multi_penalty_target_count += optimization_count;
    }

      diagnostics->frontier_optimizer_boundary_count += 1;
      diagnostics->frontier_live_target_optimization_count +=
        static_cast<int>(missing_positions.size());
      diagnostics->frontier_physical_target_optimization_count +=
        optimization_count;
      diagnostics->frontier_optimizer_host_ms += optimizer_host_ms;
      if (singleton_padding) {
        diagnostics->singleton_padding_batch_count += 1;
        diagnostics->singleton_padding_target_count += 1;
        diagnostics->singleton_padding_batch_host_ms += optimizer_host_ms;
      }

      for (int position : optimization_positions) {
        CachedTargetState state;
        state.selected_sp.resize(
          static_cast<std::size_t>(context->penalty_count));
        for (int penalty = 0; penalty < context->penalty_count; ++penalty) {
          state.selected_sp[static_cast<std::size_t>(penalty)] = selected_sp[
            static_cast<std::size_t>(context->penalty_count) * position +
            penalty];
        }
        state.planned_route =
          planned_routes[static_cast<std::size_t>(position)];
        target_state_cache().insert(
          optimizer_state_keys[static_cast<std::size_t>(position)],
          context->dataset_key, state, diagnostics);
      }
      diagnostics->cuda_optimizer_host_ms += elapsed_ms(optimize_start);
    }
  }

  // Strict permutation methods retain stable SVD residual authority. HSIC
  // gamma admits augmented QR only through two penalties; larger additive
  // systems retain SVD because their QR residuals exceed the 1e-10 contract.
  if (!direct && method_options.ci_method != "dcc.gamma") {
    FixedSpRoute forced_route = FixedSpRoute::AugmentedSvd;
    if (method_options.ci_method == "hsic.gamma" &&
        diagnostics->strict_hsic_gamma_residual_route == "qr-through-2") {
      forced_route = context->penalty_count <= 2 ?
        FixedSpRoute::AugmentedQr : FixedSpRoute::AugmentedSvd;
    } else if (is_permutation_method(method_options.ci_method) &&
               diagnostics->strict_permutation_residual_route ==
                 "qr-through-2") {
      forced_route = context->penalty_count <= 2 ?
        FixedSpRoute::AugmentedQr : FixedSpRoute::AugmentedSvd;
    }
    std::fill(planned_routes.begin(), planned_routes.end(), forced_route);
  }

  std::vector<std::string> residual_keys;
  residual_keys.reserve(static_cast<std::size_t>(target_count));
  int new_residual_key_count = 0;
  for (int target = 0; target < target_count; ++target) {
    residual_keys.push_back(residual_key(
      base_target_keys[static_cast<std::size_t>(target)],
      selected_sp.data() +
        static_cast<std::size_t>(context->penalty_count) * target,
      context->penalty_count));
    if (diagnostics->unique_residual_keys.insert(
          residual_keys.back()).second) {
      new_residual_key_count += 1;
    }
  }
  std::vector<std::string> cohort_entries;
  cohort_entries.reserve(static_cast<std::size_t>(target_count));
  for (int target = 0; target < target_count; ++target) {
    cohort_entries.push_back(
      residual_keys[static_cast<std::size_t>(target)] + "|" +
      fixed_sp_route_name(planned_routes[static_cast<std::size_t>(target)]));
  }
  std::sort(cohort_entries.begin(), cohort_entries.end());
  std::ostringstream cohort_payload;
  cohort_payload <<
    "schema=full-cuda-ci-phase10-exact-residual-cohort-v1\n" <<
    "target_count=" << target_count << "\n";
  for (const std::string& entry : cohort_entries) {
    cohort_payload << "target=" << entry << "\n";
  }
  const std::string cohort_signature = full_cuda_ci_sha256_utf8(
    cohort_payload.str());
  const bool repeated_cohort =
    !diagnostics->exact_residual_cohort_signatures.insert(
      cohort_signature).second;
  ExactResidualOpportunityClass opportunity_class =
    ExactResidualOpportunityClass::Mixed;
  if (new_residual_key_count == target_count) {
    opportunity_class = ExactResidualOpportunityClass::AllMiss;
  } else if (new_residual_key_count == 0) {
    opportunity_class = repeated_cohort ?
      ExactResidualOpportunityClass::AllHitRepeatedCohort :
      ExactResidualOpportunityClass::AllHitNewCohort;
  }
  diagnostics->logical_residual_request_count += target_count;
  diagnostics->physical_residual_fit_count += target_count;
  const std::size_t fixed_sp_batch_h2d_bytes =
    (Y.size() + selected_sp.size()) * sizeof(double);
  diagnostics->matrix_h2d_bytes += fixed_sp_batch_h2d_bytes;

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

  if (method_options.ci_method != "dcc.gamma") {
    FullCudaCiMethodBatchRequest method_request;
    method_request.expected_prepared_s_key_sha256 = context->prepared_key;
    method_request.ci_method = method_options.ci_method;
    method_request.alpha = kQualifiedAlpha;
    method_request.index = 1.0;
    method_request.num_col = num_col;
    method_request.hsic_sig = method_options.hsic_sig;
    method_request.permutation_replicates =
      is_permutation_method(method_options.ci_method) ?
        method_options.permutation_replicates : 0;
    method_request.permutation_include_observed =
      method_options.permutation_include_observed;
    method_request.pairs.reserve(task_indices.size());
    for (std::size_t index = 0; index < task_indices.size(); ++index) {
      const LayerCiTask& task =
        plan.tasks[static_cast<std::size_t>(task_indices[index])];
      FullCudaCiMethodPairRequest pair;
      pair.logical_sequence_id = logical_sequence_ids[index];
      pair.left_target_index = target_position.at(task.orientation_x);
      pair.right_target_index = target_position.at(task.orientation_y);
      method_request.pairs.push_back(pair);
    }
    const auto permutation_started = std::chrono::steady_clock::now();
    const int table_growth_before = permutation_workspace == nullptr ? 0 :
      permutation_workspace->table_growth_count;
    const int scratch_growth_before = permutation_workspace == nullptr ? 0 :
      permutation_workspace->scratch_growth_count;
    make_method_permutation_table(
      method_options, context->n, plan, task_indices,
      permutation_workspace);
    if (is_permutation_method(method_options.ci_method)) {
      diagnostics->method_permutation_table_build_count += 1;
      diagnostics->method_permutation_table_value_count +=
        permutation_workspace->table.size();
      diagnostics->method_permutation_table_host_ms +=
        elapsed_ms(permutation_started);
      diagnostics->method_permutation_table_growth_count +=
        permutation_workspace->table_growth_count - table_growth_before;
      diagnostics->method_permutation_scratch_growth_count +=
        permutation_workspace->scratch_growth_count - scratch_growth_before;
      diagnostics->method_permutation_peak_table_values = std::max(
        diagnostics->method_permutation_peak_table_values,
        permutation_workspace->peak_table_values);
      diagnostics->method_permutation_inline_r_index_count =
        permutation_workspace->inline_r_index_count;
      diagnostics->method_permutation_inline_r_draw_count =
        permutation_workspace->inline_r_draw_count;
    }
    const auto identity_started = std::chrono::steady_clock::now();
    if (is_permutation_method(method_options.ci_method)) {
      method_request.permutation_table = permutation_workspace->table.seal();
    }
    method_request.request_identity_sha256 =
      full_cuda_ci_method_batch_request_identity(
        method_request, residual_keys, context->n);
    diagnostics->method_request_identity_build_host_ms +=
      elapsed_ms(identity_started);
    FullCudaCiMethodBatchResult method_result =
      run_full_cuda_ci_method_batch(
        context->fixed_sp, batch, method_request, method_residual_cache,
        method_execution_context);
    if (is_permutation_method(method_options.ci_method)) {
      permutation_workspace->table.reclaim(
        std::move(method_request.permutation_table));
    }
    accumulate_method_diagnostics(method_result.diagnostics, diagnostics);
    if (method_result.diagnostics.residual_cache_all_hit_batch_count > 0) {
      require(method_result.diagnostics.residual_solve_host_ms == 0.0,
              "strict CI cached residual batch unexpectedly ran a solve");
      require(
        method_result.diagnostics.residual_cache_bypassed_target_count ==
          target_count &&
        diagnostics->matrix_h2d_bytes >= fixed_sp_batch_h2d_bytes,
        "strict CI cached residual batch accounting changed");
      diagnostics->physical_residual_fit_count -= target_count;
      diagnostics->matrix_h2d_bytes -= fixed_sp_batch_h2d_bytes;
    } else {
      require(
        method_result.diagnostics.residual_cache_bypassed_target_count == 0,
        "strict CI residual cache reported an unexplained bypass");
    }
    if (context->penalty_count == 1) {
      diagnostics->cuda_single_penalty_residual_solve_host_ms +=
        method_result.diagnostics.residual_solve_host_ms;
    } else {
      diagnostics->cuda_multi_penalty_residual_solve_host_ms +=
        method_result.diagnostics.residual_solve_host_ms;
    }
    require(method_result.records.size() == task_indices.size(),
            "strict CUDA CI method result count changed");
    GroupResult method_output;
    method_output.p_values.reserve(method_result.records.size());
    method_output.statistics.reserve(method_result.records.size());
    method_output.means.reserve(method_result.records.size());
    method_output.variances.reserve(method_result.records.size());
    for (const FullCudaCiMethodCompactRecord& record : method_result.records) {
      require(record.status == 0 && std::isfinite(record.p_value) &&
                record.p_value >= 0.0 && record.p_value <= 1.0,
              "strict CUDA CI method returned an invalid result");
      method_output.p_values.push_back(record.p_value);
      method_output.statistics.push_back(record.statistic);
      method_output.means.push_back(record.mean);
      method_output.variances.push_back(record.variance);
    }
    return method_output;
  }

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
  accumulate_exact_residual_opportunity(
    opportunity_class, target_count, exact.diagnostics, diagnostics);
  if (context->penalty_count == 1) {
    diagnostics->cuda_single_penalty_residual_solve_host_ms +=
      exact.diagnostics.residual_solve_host_ms;
  } else {
    diagnostics->cuda_multi_penalty_residual_solve_host_ms +=
      exact.diagnostics.residual_solve_host_ms;
  }
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
    if (context->penalty_count == 1) {
      diagnostics->cuda_single_penalty_residual_solve_host_ms +=
        refinement.diagnostics.residual_solve_host_ms;
    } else {
      diagnostics->cuda_multi_penalty_residual_solve_host_ms +=
        refinement.diagnostics.residual_solve_host_ms;
    }
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

Rcpp::List full_cuda_ci_one_call_skeleton_method(
    const Rcpp::NumericMatrix& data,
    double alpha,
    int max_conditioning_size,
    double index,
    int num_col,
    const std::string& trace_level,
    bool compatible_cuda_strict,
    const FullCudaCiOneCallMethodOptions& method_options) {
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
  require(max_conditioning_size >= 0 && max_conditioning_size <= p - 2,
          "compatible.cuda one-call conditioning size is outside [0, p-2]");
  require(trace_level == "summary" || trace_level == "logical" ||
            trace_level == "full" || trace_level == "none",
          "compatible.cuda one-call trace level is invalid");
  require(finite_matrix(data),
          "compatible.cuda one-call data must contain finite doubles");
  require(method_options.ci_method == "dcc.gamma" ||
            method_options.ci_method == "dcc.perm" ||
            method_options.ci_method == "hsic.gamma" ||
            method_options.ci_method == "hsic.perm",
          "compatible.cuda one-call CI method is unsupported");
  require(std::isfinite(method_options.hsic_sig) &&
            method_options.hsic_sig > 0.0,
          "compatible.cuda one-call HSIC sig must be positive and finite");
  if (is_permutation_method(method_options.ci_method)) {
    require(method_options.permutation_has_seed,
            "compatible.cuda one-call permutation method requires a seed");
    require(method_options.permutation_replicates >= 1 &&
              method_options.permutation_replicates <= 10000,
            "compatible.cuda one-call permutation replicates are outside [1,10000]");
    require(method_options.permutation_include_observed,
            "compatible.cuda one-call strict permutation requires observed inclusion");
  }
  const bool collect_trace = trace_level == "logical" ||
    trace_level == "full";
  const bool legacy_r_permutation =
    is_permutation_method(method_options.ci_method);
  std::unique_ptr<Rcpp::RNGScope> rng_scope;
  if (legacy_r_permutation) {
    rng_scope.reset(new Rcpp::RNGScope());
  }

  const std::string dataset = dataset_key(data);
  FixedSpRuntimePool runtime_pool(n);

  OneCallDiagnostics diagnostics;
  diagnostics.strict_hsic_gamma_residual_route =
    strict_hsic_gamma_residual_route_policy(method_options.ci_method);
  diagnostics.strict_permutation_residual_route =
    strict_permutation_residual_route_policy(method_options.ci_method);
  require(method_options.ci_method == "hsic.gamma" ||
            diagnostics.strict_hsic_gamma_residual_route == "stable-svd",
          "HSIC gamma residual route override requires hsic.gamma");
  diagnostics.setup_optimizer_pipeline_enabled =
    setup_optimizer_pipeline_policy_enabled();
  diagnostics.setup_optimizer_pipeline_producer_delay_us =
    setup_optimizer_pipeline_producer_delay_us();
  require(diagnostics.setup_optimizer_pipeline_enabled ||
            diagnostics.setup_optimizer_pipeline_producer_delay_us == 0,
          "setup/optimizer producer delay requires the pipeline");
  const bool enable_fixed_sp_root_cache =
    fixed_sp_root_cache_policy_enabled();
  diagnostics.fixed_sp_root_cache_enabled = enable_fixed_sp_root_cache;
  diagnostics.cuda_multi_penalty_decomposition_trace_capacity_per_target =
    decomposition_trace_capacity_from_environment();
  const CompactResultCache::Snapshot result_cache_start =
    compact_result_cache().snapshot(dataset);
  diagnostics.result_cache_capacity = result_cache_start.capacity;
  diagnostics.result_cache_warm_start_entries = result_cache_start.entries;
  diagnostics.result_cache_dataset_warm_start_entries =
    result_cache_start.dataset_entries;
  diagnostics.result_cache_warm_start_insert_count =
    result_cache_start.total_inserts;
  const TargetStateCache::Snapshot target_cache_start =
    target_state_cache().snapshot(dataset);
  diagnostics.target_cache_capacity = target_cache_start.capacity;
  diagnostics.target_cache_warm_start_entries = target_cache_start.entries;
  diagnostics.target_cache_dataset_warm_start_entries =
    target_cache_start.dataset_entries;
  diagnostics.target_cache_warm_start_insert_count =
    target_cache_start.total_inserts;
  const bool fresh_dataset_semantic_caches =
    result_cache_start.dataset_entries == 0U &&
    target_cache_start.dataset_entries == 0U;
  diagnostics.dataset_cache_epoch_at_start =
    dataset_cache_epoch().load(std::memory_order_acquire);
  NativeSetupCache setup_cache(
    data, dataset, &runtime_pool, &diagnostics,
    enable_fixed_sp_root_cache,
    diagnostics.setup_optimizer_pipeline_enabled);
  std::shared_ptr<NativeSetupContext> direct = build_direct_context(
    data, dataset, runtime_pool.base_runtime());
  constexpr std::size_t kStrictMethodResidualCacheBudget =
    384U * 1024U * 1024U;
  std::shared_ptr<FullCudaCiMethodResidualCache> method_residual_cache;
  std::shared_ptr<FullCudaCiMethodExecutionContext> method_execution_context;
  if (method_options.ci_method != "dcc.gamma") {
    method_residual_cache = create_full_cuda_ci_method_residual_cache(
      n, kStrictMethodResidualCacheBudget);
    method_execution_context = create_full_cuda_ci_method_execution_context(
      n, method_options.ci_method, num_col,
      is_permutation_method(method_options.ci_method) ?
        method_options.permutation_replicates : 0);
  }
  LegacyRPermutationWorkspace permutation_workspace;
  permutation_workspace.inline_r_index_requested =
    strict_hsic_perm_inline_r_index_policy_enabled(
      method_options.ci_method);
  permutation_workspace.inline_r_index_active =
    permutation_workspace.inline_r_index_requested &&
      R_sample_kind() == REJECTION && n <= 32768;
  if (permutation_workspace.inline_r_index_active) {
    permutation_workspace.rejection_masks.resize(
      static_cast<std::size_t>(n) + 1U, 0U);
    for (int remaining = 1; remaining <= n; ++remaining) {
      unsigned int power = 1U;
      while (power < static_cast<unsigned int>(remaining)) power <<= 1U;
      permutation_workspace.rejection_masks[
        static_cast<std::size_t>(remaining)] = power - 1U;
    }
  }
  diagnostics.method_permutation_inline_r_index_requested =
    permutation_workspace.inline_r_index_requested;
  diagnostics.method_permutation_inline_r_index_active =
    permutation_workspace.inline_r_index_active;

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
  std::vector<double> trace_p_used, trace_permutation_seed,
    trace_method_statistic, trace_method_mean, trace_method_variance;
  std::vector<int> trace_deleted;

  std::uint64_t next_request_sequence_id = 1U;
  int next_logical_id = 1;
  int total_tasks_planned = 0;
  int total_tests_replayed = 0;
  int total_ignored = 0;
  int total_deletions = 0;
  int total_frontier_batches = 0;

  try {
    for (int level = 0; level <= max_conditioning_size; ++level) {
      const auto level_started = std::chrono::steady_clock::now();
      const LayerPlan plan = make_layer_plan(adjacency, p, level);
      const int task_count = static_cast<int>(plan.tasks.size());
      std::vector<unsigned char> delete_edges(
        static_cast<std::size_t>(p) * p, 0U);
      std::vector<unsigned char> consumed(
        static_cast<std::size_t>(task_count), 0U);
      std::vector<double> task_p_values(
        static_cast<std::size_t>(task_count), NA_REAL);
      std::vector<double> task_method_statistics(
        static_cast<std::size_t>(task_count), NA_REAL);
      std::vector<double> task_method_means(
        static_cast<std::size_t>(task_count), NA_REAL);
      std::vector<double> task_method_variances(
        static_cast<std::size_t>(task_count), NA_REAL);
      int remaining = task_count;
      int tests_replayed = 0;
      int ignored = 0;
      int deletions = 0;
      int rounds = 0;
      bool pcalg_done = true;

      std::vector<std::vector<int>> edge_tasks(
        static_cast<std::size_t>(p) * p);
      for (int task_index = 0; task_index < task_count; ++task_index) {
        const LayerCiTask& task =
          plan.tasks[static_cast<std::size_t>(task_index)];
        edge_tasks[static_cast<std::size_t>(task.edge_key)].push_back(
          task_index);
      }
      if (fresh_dataset_semantic_caches && level >= 1 && level <= 2) {
        prefill_single_penalty_level_target_states(
          data, plan, &setup_cache, &diagnostics);
      } else if (fresh_dataset_semantic_caches && level >= 3) {
        prefill_multi_penalty_level_target_states(
          data, plan, &setup_cache, &diagnostics);
      }
      std::vector<std::size_t> edge_positions(
        static_cast<std::size_t>(p) * p, 0U);
      std::map<std::string, std::vector<int>> ready_by_conditioning;
      int active_edges = 0;
      for (std::size_t edge = 0; edge < edge_tasks.size(); ++edge) {
        if (edge_tasks[edge].empty()) continue;
        const int task_index = edge_tasks[edge].front();
        const LayerCiTask& task =
          plan.tasks[static_cast<std::size_t>(task_index)];
        ready_by_conditioning[conditioning_key(task.conditioning_set)]
          .push_back(static_cast<int>(edge));
        ++active_edges;
      }

      while (active_edges > 0) {
        require(!ready_by_conditioning.empty(),
                "compatible.cuda frontier scheduler made no progress");
        struct FrontierWork {
          std::vector<int> selected;
          std::vector<std::uint64_t> selected_ids;
          std::vector<std::string> selected_cache_keys;
          std::vector<double> selected_pvalues;
          std::vector<std::size_t> missing_positions;
          std::vector<int> group_tasks;
          std::vector<std::uint64_t> group_ids;
          std::shared_ptr<NativeSetupContext> context;
          bool direct = false;
        };
        std::vector<FrontierWork> wave;
        const int wave_capacity = 1;
        wave.reserve(static_cast<std::size_t>(wave_capacity));

        for (int wave_index = 0; wave_index < wave_capacity; ++wave_index) {
          std::vector<int> selected_edges;
          if (legacy_r_permutation) {
            auto selected_bucket = ready_by_conditioning.end();
            std::size_t selected_position = 0U;
            int selected_task = -1;
            for (auto candidate = ready_by_conditioning.begin();
                 candidate != ready_by_conditioning.end(); ++candidate) {
              for (std::size_t position = 0;
                   position < candidate->second.size(); ++position) {
                const std::size_t edge = static_cast<std::size_t>(
                  candidate->second[position]);
                require(edge < edge_tasks.size() &&
                          edge_positions[edge] < edge_tasks[edge].size(),
                        "legacy R permutation frontier is malformed");
                const int task_index =
                  edge_tasks[edge][edge_positions[edge]];
                if (selected_task < 0 || pcalg_task_less(
                      plan.tasks[static_cast<std::size_t>(task_index)],
                      plan.tasks[static_cast<std::size_t>(selected_task)])) {
                  selected_bucket = candidate;
                  selected_position = position;
                  selected_task = task_index;
                }
              }
            }
            require(selected_bucket != ready_by_conditioning.end() &&
                      selected_task >= 0,
                    "legacy R permutation scheduler selected no task");
            selected_edges.push_back(
              selected_bucket->second[selected_position]);
            selected_bucket->second.erase(
              selected_bucket->second.begin() +
                static_cast<std::ptrdiff_t>(selected_position));
            if (selected_bucket->second.empty()) {
              ready_by_conditioning.erase(selected_bucket);
            }
          } else {
            auto global_largest = ready_by_conditioning.begin();
            auto cached_largest = ready_by_conditioning.end();
            for (auto candidate = ready_by_conditioning.begin();
                 candidate != ready_by_conditioning.end(); ++candidate) {
              if (candidate->second.size() > global_largest->second.size()) {
                global_largest = candidate;
              }
              if (!candidate->first.empty() &&
                  setup_cache.contains_conditioning_key(candidate->first) &&
                  (cached_largest == ready_by_conditioning.end() ||
                   candidate->second.size() > cached_largest->second.size())) {
                cached_largest = candidate;
              }
            }
            auto selected_bucket = global_largest;
            if (cached_largest != ready_by_conditioning.end() &&
                cached_largest->second.size() * 4U >=
                  global_largest->second.size()) {
              selected_bucket = cached_largest;
            }
            selected_edges = std::move(selected_bucket->second);
            ready_by_conditioning.erase(selected_bucket);
          }

          FrontierWork work;
          work.selected.reserve(selected_edges.size());
          for (int edge_value : selected_edges) {
            const std::size_t edge = static_cast<std::size_t>(edge_value);
            require(edge < edge_tasks.size() &&
                      edge_positions[edge] < edge_tasks[edge].size(),
                    "compatible.cuda frontier edge position is invalid");
            work.selected.push_back(
              edge_tasks[edge][edge_positions[edge]]);
          }
          require(!work.selected.empty(),
                  "compatible.cuda frontier scheduler selected an empty batch");
          ++rounds;

          const LayerCiTask& representative = plan.tasks[
            static_cast<std::size_t>(work.selected.front())];
          const std::string selected_conditioning_key =
            conditioning_key(representative.conditioning_set);
          for (int task_index : work.selected) {
            require(conditioning_key(plan.tasks[
                      static_cast<std::size_t>(task_index)].conditioning_set) ==
                      selected_conditioning_key,
                    "compatible.cuda frontier batch mixed conditioning sets");
          }

          work.selected_ids.resize(work.selected.size());
          work.selected_cache_keys.resize(work.selected.size());
          work.selected_pvalues.assign(work.selected.size(), NA_REAL);
          for (std::size_t position = 0; position < work.selected.size();
               ++position) {
            work.selected_ids[position] = next_request_sequence_id++;
            const int task_index = work.selected[position];
            const LayerCiTask& task =
              plan.tasks[static_cast<std::size_t>(task_index)];
            work.selected_cache_keys[position] = compact_result_key(
              dataset, num_col, level, task_index, task, method_options);
            double cached_p_value = NA_REAL;
            if (!legacy_r_permutation && compact_result_cache().lookup(
                  work.selected_cache_keys[position], dataset,
                  &cached_p_value, &diagnostics)) {
              require(std::isfinite(cached_p_value) &&
                        cached_p_value >= 0.0 && cached_p_value <= 1.0,
                      "compact result cache returned an invalid p-value");
              work.selected_pvalues[position] = cached_p_value;
            } else {
              work.missing_positions.push_back(position);
              work.group_tasks.push_back(task_index);
              work.group_ids.push_back(work.selected_ids[position]);
            }
          }
          if (!work.group_tasks.empty()) {
            work.direct = representative.conditioning_set.empty();
            work.context = work.direct ? direct :
              setup_cache.get(representative.conditioning_set);
            if (!work.direct) setup_cache.materialize_device(work.context);
          }
          wave.push_back(std::move(work));
        }

        for (FrontierWork& work : wave) {
          if (!work.group_tasks.empty()) {
            const GroupResult result = execute_group(
              data, work.context, plan, work.group_tasks, work.group_ids,
              num_col, work.direct, method_options, method_residual_cache,
              method_execution_context, &permutation_workspace, &diagnostics);
            require(result.p_values.size() == work.missing_positions.size(),
                    "compatible.cuda group result alignment changed");
            const bool method_records =
              result.statistics.size() == result.p_values.size() &&
              result.means.size() == result.p_values.size() &&
              result.variances.size() == result.p_values.size();
            for (std::size_t index = 0;
                 index < work.missing_positions.size(); ++index) {
              const std::size_t selected_position =
                work.missing_positions[index];
              work.selected_pvalues[selected_position] =
                result.p_values[index];
              if (method_records) {
                const int task_index = work.group_tasks[index];
                task_method_statistics[
                  static_cast<std::size_t>(task_index)] =
                    result.statistics[index];
                task_method_means[static_cast<std::size_t>(task_index)] =
                  result.means[index];
                task_method_variances[
                  static_cast<std::size_t>(task_index)] =
                    result.variances[index];
              }
              if (!legacy_r_permutation) {
                compact_result_cache().insert(
                  work.selected_cache_keys[selected_position], dataset,
                  result.p_values[index], &diagnostics);
              }
            }
          }

          for (std::size_t position = 0; position < work.selected.size();
               ++position) {
            const int task_index = work.selected[position];
            const LayerCiTask& task =
              plan.tasks[static_cast<std::size_t>(task_index)];
            const double p_value = work.selected_pvalues[position];
            require(std::isfinite(p_value),
                    "compatible.cuda replay received a non-finite p-value");
            consumed[static_cast<std::size_t>(task_index)] = 1U;
            task_p_values[static_cast<std::size_t>(task_index)] = p_value;
            if (task.opens_next_level) pcalg_done = false;
            --remaining;
            const std::size_t forward =
              static_cast<std::size_t>(task.edge_x) * p + task.edge_y;
            const std::size_t reverse =
              static_cast<std::size_t>(task.edge_y) * p + task.edge_x;
            const bool deleted = p_value >= alpha;
            if (deleted) {
              delete_edges[forward] = 1U;
              delete_edges[reverse] = 1U;
            }

            const std::size_t edge = static_cast<std::size_t>(task.edge_key);
            require(edge < edge_tasks.size() &&
                      edge_positions[edge] < edge_tasks[edge].size() &&
                      edge_tasks[edge][edge_positions[edge]] == task_index,
                    "compatible.cuda frontier advancement changed task order");
            ++edge_positions[edge];
            if (deleted) {
              const int trailing = static_cast<int>(
                edge_tasks[edge].size() - edge_positions[edge]);
              ignored += trailing;
              remaining -= trailing;
              edge_positions[edge] = edge_tasks[edge].size();
              --active_edges;
            } else if (edge_positions[edge] == edge_tasks[edge].size()) {
              --active_edges;
            } else {
              const int next_task_index =
                edge_tasks[edge][edge_positions[edge]];
              const LayerCiTask& next_task =
                plan.tasks[static_cast<std::size_t>(next_task_index)];
              ready_by_conditioning[
                conditioning_key(next_task.conditioning_set)]
                  .push_back(static_cast<int>(edge));
            }
          }
        }
      }
      require(remaining == 0 && ready_by_conditioning.empty(),
              "compatible.cuda frontier scheduler left reachable tasks");

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
          if (method_options.ci_method == "hsic.gamma") {
            trace_method_statistic.push_back(task_method_statistics[
              static_cast<std::size_t>(task_index)]);
            trace_method_mean.push_back(task_method_means[
              static_cast<std::size_t>(task_index)]);
            trace_method_variance.push_back(task_method_variances[
              static_cast<std::size_t>(task_index)]);
          }
          if (is_permutation_method(method_options.ci_method)) {
            trace_permutation_seed.push_back(static_cast<double>(
              task_permutation_seed(method_options, task)));
          }
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
      total_frontier_batches += rounds;
      n_edgetests.push_back(tests_replayed);
      level_values.push_back(level);
      level_tasks_planned.push_back(task_count);
      level_tests_replayed.push_back(tests_replayed);
      level_ignored.push_back(ignored);
      level_deletions.push_back(deletions);
      level_rounds.push_back(rounds);
      level_elapsed_ms.push_back(elapsed_ms(level_started));
      if (pcalg_done) break;
    }
  } catch (...) {
    close_context(direct);
    setup_cache.clear();
    runtime_pool.close();
    throw;
  }

  close_context(direct);
  setup_cache.clear();
  const std::size_t runtime_workspace_bytes = runtime_pool.close(&diagnostics);

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
  if (is_permutation_method(method_options.ci_method)) {
    require(trace_permutation_seed.size() == trace_deleted.size(),
            "permutation trace seed count changed");
    task_rows["permutation_seed"] = Rcpp::wrap(trace_permutation_seed);
    task_rows.attr("class") = "data.frame";
    task_rows.attr("row.names") = Rcpp::IntegerVector::create(
      NA_INTEGER, -static_cast<int>(trace_deleted.size()));
  } else if (method_options.ci_method == "hsic.gamma") {
    require(trace_method_statistic.size() == trace_deleted.size() &&
              trace_method_mean.size() == trace_deleted.size() &&
              trace_method_variance.size() == trace_deleted.size(),
            "HSIC gamma compact trace count changed");
    task_rows["method_statistic"] = Rcpp::wrap(trace_method_statistic);
    task_rows["method_mean"] = Rcpp::wrap(trace_method_mean);
    task_rows["method_variance"] = Rcpp::wrap(trace_method_variance);
    task_rows.attr("class") = "data.frame";
    task_rows.attr("row.names") = Rcpp::IntegerVector::create(
      NA_INTEGER, -static_cast<int>(trace_deleted.size()));
  }
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
  const int unique_prepared_s_key_count = static_cast<int>(
    diagnostics.consumed_prepared_keys.size());
  const int physical_prepared_s_key_count = static_cast<int>(
    diagnostics.physical_prepared_keys.size());
  const int unique_target_key_count = static_cast<int>(
    diagnostics.unique_target_keys.size());
  const int unique_residual_key_count = static_cast<int>(
    diagnostics.unique_residual_keys.size());
  const int physical_target_optimization_count =
    diagnostics.cuda_single_penalty_target_count +
      diagnostics.cuda_multi_penalty_target_count;
  const PrefillAttributionSummary prefill =
    summarize_prefill_attribution(diagnostics);
  const Rcpp::DataFrame prefill_batches = prefill_batch_diagnostics_frame(
    diagnostics.prefill_batches, diagnostics.unique_target_keys);
  const bool method_residual_cache_enabled =
    static_cast<bool>(method_residual_cache);
  const bool cacheable_strict_method =
    method_options.ci_method != "dcc.gamma";
  const bool method_residual_cache_accounted = !cacheable_strict_method ||
    ((!method_residual_cache_enabled &&
       diagnostics.method_residual_cache_lookup_count == 0 &&
       diagnostics.method_residual_cache_hit_count == 0 &&
       diagnostics.method_residual_cache_insert_count == 0 &&
       diagnostics.method_residual_cache_all_hit_batch_count == 0 &&
       diagnostics.method_residual_cache_bypassed_target_count == 0 &&
       diagnostics.physical_residual_fit_count ==
         diagnostics.logical_residual_request_count) ||
     (method_residual_cache_enabled &&
       diagnostics.method_residual_cache_lookup_count ==
         diagnostics.logical_residual_request_count &&
       diagnostics.method_residual_cache_lookup_count ==
         diagnostics.method_residual_cache_hit_count +
           diagnostics.method_residual_cache_insert_count &&
       diagnostics.method_residual_cache_bypassed_target_count >=
         diagnostics.method_residual_cache_all_hit_batch_count &&
       diagnostics.physical_residual_fit_count ==
         diagnostics.logical_residual_request_count -
           diagnostics.method_residual_cache_bypassed_target_count &&
       diagnostics.cuda_exact_screen_residual_target_count ==
         diagnostics.physical_residual_fit_count &&
       diagnostics.cuda_exact_screen_residual_batch_count +
           diagnostics.method_residual_cache_all_hit_batch_count ==
         total_frontier_batches &&
       diagnostics.method_residual_cache_gather_d2d_bytes ==
         static_cast<std::size_t>(
           diagnostics.method_residual_cache_bypassed_target_count) *
           static_cast<std::size_t>(n) * sizeof(double)));
  const bool fixed_sp_root_cache_accounted =
    diagnostics.fixed_sp_root_cache_runtime_count >= 1 &&
    diagnostics.fixed_sp_root_cache_capacity_entries_per_runtime > 0U &&
    diagnostics.fixed_sp_root_cache_capacity_entries_total ==
      diagnostics.fixed_sp_root_cache_capacity_entries_per_runtime *
        static_cast<std::size_t>(
          diagnostics.fixed_sp_root_cache_runtime_count) &&
    diagnostics.fixed_sp_root_cache_identity_rejection_count == 0U &&
    diagnostics.fixed_sp_root_cache_lookup_count ==
      diagnostics.fixed_sp_root_cache_hit_count +
        diagnostics.fixed_sp_root_cache_miss_count &&
    diagnostics.fixed_sp_root_cache_miss_count ==
      diagnostics.fixed_sp_root_cache_insert_count +
        diagnostics.fixed_sp_root_cache_bypass_count &&
    diagnostics.fixed_sp_root_cache_entries ==
      diagnostics.fixed_sp_root_cache_insert_count &&
    diagnostics.fixed_sp_root_cache_peak_entries ==
      diagnostics.fixed_sp_root_cache_entries &&
    diagnostics.fixed_sp_root_cache_entries <=
      diagnostics.fixed_sp_root_cache_capacity_entries_total &&
    diagnostics.fixed_sp_root_cache_device_bytes ==
      diagnostics.fixed_sp_root_cache_peak_device_bytes &&
    diagnostics.fixed_sp_root_cache_device_bytes <=
      diagnostics.fixed_sp_root_cache_capacity_bytes_total &&
    diagnostics.fixed_sp_root_cache_insert_d2d_bytes ==
      diagnostics.fixed_sp_root_cache_device_bytes &&
    (diagnostics.fixed_sp_root_cache_enabled ||
      (diagnostics.fixed_sp_root_cache_lookup_count == 0U &&
       diagnostics.fixed_sp_root_cache_entries == 0U &&
       diagnostics.fixed_sp_root_cache_device_bytes == 0U &&
       diagnostics.fixed_sp_root_cache_hit_d2d_bytes == 0U));
  const bool method_execution_context_accounted =
    (!cacheable_strict_method &&
      diagnostics.method_execution_context_call_count == 0 &&
      diagnostics.method_execution_context_reuse_count == 0 &&
      diagnostics.method_execution_context_buffer_growth_count == 0 &&
      diagnostics.method_execution_context_peak_device_bytes == 0U) ||
    (cacheable_strict_method &&
      diagnostics.method_execution_context_call_count ==
        total_frontier_batches &&
      diagnostics.method_execution_context_reuse_count ==
        std::max(0, diagnostics.method_execution_context_call_count - 1) &&
      diagnostics.method_execution_context_buffer_growth_count > 0 &&
      diagnostics.method_execution_context_peak_device_bytes > 0U);
  const bool persistent_component_cache_expected =
    method_options.ci_method == "hsic.perm";
  const std::size_t persistent_component_bytes =
    static_cast<std::size_t>(n) *
      static_cast<std::size_t>(std::min(num_col, n)) * sizeof(double) +
      sizeof(int);
  const bool method_component_cache_accounted =
    (!persistent_component_cache_expected &&
      diagnostics.method_component_cache_request_count == 0 &&
      diagnostics.method_component_cache_lookup_count == 0 &&
      diagnostics.method_component_cache_hit_count == 0 &&
      diagnostics.method_component_cache_miss_count == 0 &&
      diagnostics.method_component_cache_insert_count == 0 &&
      diagnostics.method_component_cache_eviction_count == 0 &&
      diagnostics.method_component_cache_capacity_entries == 0U &&
      diagnostics.method_component_cache_device_bytes == 0U &&
      diagnostics.method_component_cache_gather_d2d_bytes == 0U &&
      diagnostics.method_component_cache_store_d2d_bytes == 0U) ||
    (persistent_component_cache_expected &&
      diagnostics.method_component_cache_capacity_entries >= 2U &&
      diagnostics.method_component_cache_request_count ==
        diagnostics.method_component_cache_hit_count +
          diagnostics.method_component_cache_miss_count &&
      diagnostics.method_component_cache_miss_count ==
        diagnostics.cuda_exact_screen_component_count &&
      diagnostics.method_component_cache_hit_count <=
        diagnostics.method_component_cache_lookup_count &&
      diagnostics.method_component_cache_lookup_count <=
        diagnostics.method_component_cache_request_count &&
      diagnostics.method_component_cache_insert_count <=
        diagnostics.method_component_cache_miss_count &&
      diagnostics.method_component_cache_eviction_count <=
        diagnostics.method_component_cache_insert_count &&
      diagnostics.method_component_cache_device_bytes ==
        diagnostics.method_component_cache_capacity_entries *
          persistent_component_bytes &&
      diagnostics.method_component_cache_gather_d2d_bytes ==
        static_cast<std::size_t>(
          diagnostics.method_component_cache_hit_count) *
          persistent_component_bytes &&
      diagnostics.method_component_cache_store_d2d_bytes ==
        static_cast<std::size_t>(
          diagnostics.method_component_cache_insert_count) *
          persistent_component_bytes);
  const bool method_permutation_inline_r_index_accounted =
    (!diagnostics.method_permutation_inline_r_index_active &&
      diagnostics.method_permutation_inline_r_index_count == 0U &&
      diagnostics.method_permutation_inline_r_draw_count == 0U) ||
    (diagnostics.method_permutation_inline_r_index_active &&
      method_options.ci_method == "hsic.perm" &&
      diagnostics.method_permutation_inline_r_index_count ==
        diagnostics.method_permutation_table_value_count &&
      diagnostics.method_permutation_inline_r_draw_count >=
        diagnostics.method_permutation_inline_r_index_count);
  require(diagnostics.native_setup_count >= physical_prepared_s_key_count &&
            physical_prepared_s_key_count >= unique_prepared_s_key_count &&
            diagnostics.physical_residual_fit_count >=
              unique_residual_key_count &&
            diagnostics.native_setup_univariate_primitive_request_count ==
              diagnostics.native_setup_univariate_primitive_hit_count +
                diagnostics.native_setup_univariate_primitive_build_count &&
            diagnostics.native_setup_univariate_primitive_build_count ==
              diagnostics.native_setup_univariate_primitive_cache_peak_entries &&
            diagnostics.native_setup_univariate_primitive_build_count <=
              diagnostics.native_setup_univariate_primitive_cache_capacity &&
            diagnostics.cuda_multi_penalty_prepared_build_count ==
              diagnostics.cuda_multi_penalty_prepared_release_count &&
            diagnostics.cuda_multi_penalty_prepared_target_capacity_peak <=
              64 &&
            prefill.target_optimization_count ==
              prefill.single_penalty_target_count +
                prefill.multi_penalty_target_count &&
            physical_target_optimization_count ==
              prefill.target_optimization_count +
                diagnostics.frontier_physical_target_optimization_count &&
            diagnostics.frontier_physical_target_optimization_count ==
              diagnostics.frontier_live_target_optimization_count +
                diagnostics.singleton_padding_target_count &&
            diagnostics.singleton_padding_batch_count ==
              diagnostics.singleton_padding_target_count &&
            diagnostics.cuda_optimizer_host_boundary_count ==
              prefill.optimizer_boundary_count +
                diagnostics.frontier_optimizer_boundary_count &&
            method_residual_cache_accounted &&
            method_execution_context_accounted &&
            method_component_cache_accounted &&
            method_permutation_inline_r_index_accounted &&
            fixed_sp_root_cache_accounted,
          "compatible.cuda one-call physical work accounting changed");
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

  const DecompositionReuseAggregate& decomposition_reuse =
    diagnostics.cuda_multi_penalty_decomposition_reuse;
  const bool decomposition_trace_enabled =
    diagnostics.cuda_multi_penalty_decomposition_trace_capacity_per_target >
      0;
  require(
    (!decomposition_trace_enabled && decomposition_reuse.batch_count == 0U &&
      decomposition_reuse.request_count == 0U &&
      decomposition_reuse.stored_count == 0U) ||
    (decomposition_trace_enabled &&
      ((diagnostics.cuda_multi_penalty_target_count == 0 &&
        decomposition_reuse.batch_count == 0U &&
        decomposition_reuse.request_count == 0U) ||
       (diagnostics.cuda_multi_penalty_target_count > 0 &&
        decomposition_reuse.batch_count > 0U &&
        decomposition_reuse.request_count ==
          decomposition_reuse.stored_count +
            decomposition_reuse.overflow_count &&
        decomposition_reuse.unique_key_count <=
          decomposition_reuse.stored_count &&
        decomposition_reuse.reuse_count ==
          decomposition_reuse.stored_count -
            decomposition_reuse.unique_key_count))),
    "compatible.cuda decomposition trace summary is malformed");
  const double decomposition_reuse_ratio =
    decomposition_reuse.stored_count > 0U ?
      static_cast<double>(decomposition_reuse.reuse_count) /
        static_cast<double>(decomposition_reuse.stored_count) : 0.0;
  const double decomposition_group_size_p50 =
    decomposition_group_size_quantile(
      decomposition_reuse.group_size_histogram, 0.50);
  const double decomposition_group_size_p95 =
    decomposition_group_size_quantile(
      decomposition_reuse.group_size_histogram, 0.95);
  const double decomposition_group_size_max =
    decomposition_reuse.group_size_histogram.empty() ? 0.0 :
      static_cast<double>(
        decomposition_reuse.group_size_histogram.size() - 1U);
  const Rcpp::DataFrame decomposition_reuse_by_stage =
    decomposition_reuse_stage_frame(decomposition_reuse);
  const Rcpp::DataFrame decomposition_reuse_by_route =
    decomposition_reuse_route_frame(decomposition_reuse);
  const Rcpp::DataFrame decomposition_reuse_by_shape =
    decomposition_reuse_shape_frame(
      diagnostics.cuda_multi_penalty_decomposition_reuse_by_shape);
  const Rcpp::DataFrame decomposition_reuse_by_iteration =
    decomposition_iteration_reuse_frame(
      diagnostics.cuda_multi_penalty_decomposition_iteration_reuse);

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
      Rcpp::Named("ci_method") = method_options.ci_method,
      Rcpp::Named("ci_backend") = method_options.ci_method == "dcc.gamma" ?
        "guarded-exact-screen-legacy-full-eig-cuda" :
        "strict-device-residual-ci-cuda-v1",
      Rcpp::Named("hsic_sig") = method_options.hsic_sig,
      Rcpp::Named("permutation_replicates") =
        is_permutation_method(method_options.ci_method) ?
          method_options.permutation_replicates : 0,
      Rcpp::Named("permutation_include_observed") =
        method_options.permutation_include_observed,
      Rcpp::Named("permutation_has_seed") =
        method_options.permutation_has_seed,
      Rcpp::Named("permutation_seed") =
        method_options.permutation_has_seed ?
          static_cast<double>(method_options.permutation_seed) : NA_REAL,
      Rcpp::Named("scheduler") = "cuda-level-target-prefill-host-v5",
      Rcpp::Named("frontier_batch_count") = total_frontier_batches,
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
      Rcpp::Named("physical_target_optimization_count") =
        physical_target_optimization_count,
      Rcpp::Named("excess_target_optimization_count") =
        std::max(0, physical_target_optimization_count - unique_target_key_count),
      Rcpp::Named("reused_target_state_count") =
        std::max(0, unique_target_key_count - physical_target_optimization_count),
      Rcpp::Named("prefill_conditioning_group_count") =
        prefill.conditioning_group_count,
      Rcpp::Named("prefill_window_count") = prefill.window_count,
      Rcpp::Named("prefill_optimizer_boundary_count") =
        prefill.optimizer_boundary_count,
      Rcpp::Named("prefill_optimizer_setup_count") =
        prefill.optimizer_setup_count,
      Rcpp::Named("prefill_target_optimization_count") =
        prefill.target_optimization_count,
      Rcpp::Named("prefill_single_penalty_target_count") =
        prefill.single_penalty_target_count,
      Rcpp::Named("prefill_multi_penalty_target_count") =
        prefill.multi_penalty_target_count,
      Rcpp::Named("prefill_unique_target_key_count") =
        prefill.unique_target_key_count,
      Rcpp::Named("prefill_consumed_unique_target_key_count") =
        prefill.consumed_unique_target_key_count,
      Rcpp::Named("prefill_unconsumed_unique_target_key_count") =
        prefill.unconsumed_unique_target_key_count,
      Rcpp::Named("prefill_singleton_skipped_request_count") =
        prefill.singleton_skipped_request_count,
      Rcpp::Named("prefill_singleton_skipped_target_count") =
        prefill.singleton_skipped_target_count,
      Rcpp::Named("frontier_optimizer_boundary_count") =
        diagnostics.frontier_optimizer_boundary_count,
      Rcpp::Named("frontier_live_target_optimization_count") =
        diagnostics.frontier_live_target_optimization_count,
      Rcpp::Named("frontier_physical_target_optimization_count") =
        diagnostics.frontier_physical_target_optimization_count,
      Rcpp::Named("singleton_padding_batch_count") =
        diagnostics.singleton_padding_batch_count,
      Rcpp::Named("singleton_padding_target_count") =
        diagnostics.singleton_padding_target_count,
      Rcpp::Named("prefill_batches") = prefill_batches,
      Rcpp::Named("cuda_single_penalty_optimizer_setup_count") =
        diagnostics.cuda_single_penalty_optimizer_setup_count,
      Rcpp::Named("cuda_single_penalty_optimizer_call_count") =
        diagnostics.cuda_single_penalty_optimizer_call_count,
      Rcpp::Named("cuda_multi_penalty_optimizer_setup_count") =
        diagnostics.cuda_multi_penalty_optimizer_setup_count,
      Rcpp::Named("cuda_multi_penalty_optimizer_call_count") =
        diagnostics.cuda_multi_penalty_optimizer_call_count,
      Rcpp::Named("cuda_multi_penalty_prepared_build_count") =
        diagnostics.cuda_multi_penalty_prepared_build_count,
      Rcpp::Named("cuda_multi_penalty_prepared_release_count") =
        diagnostics.cuda_multi_penalty_prepared_release_count,
      Rcpp::Named("cuda_multi_penalty_prepared_target_capacity_sum") =
        diagnostics.cuda_multi_penalty_prepared_target_capacity_sum,
      Rcpp::Named("cuda_multi_penalty_prepared_target_capacity_peak") =
        diagnostics.cuda_multi_penalty_prepared_target_capacity_peak,
      Rcpp::Named("cuda_multi_penalty_decomposition_trace_enabled") =
        decomposition_trace_enabled,
      Rcpp::Named(
        "cuda_multi_penalty_decomposition_trace_capacity_per_target") =
          diagnostics
            .cuda_multi_penalty_decomposition_trace_capacity_per_target,
      Rcpp::Named("cuda_multi_penalty_decomposition_trace_batch_count") =
        static_cast<double>(decomposition_reuse.batch_count),
      Rcpp::Named("cuda_multi_penalty_decomposition_request_count") =
        static_cast<double>(decomposition_reuse.request_count),
      Rcpp::Named("cuda_multi_penalty_decomposition_stored_count") =
        static_cast<double>(decomposition_reuse.stored_count),
      Rcpp::Named(
        "cuda_multi_penalty_decomposition_trace_overflow_count") =
          static_cast<double>(decomposition_reuse.overflow_count),
      Rcpp::Named("cuda_multi_penalty_decomposition_unique_key_count") =
        static_cast<double>(decomposition_reuse.unique_key_count),
      Rcpp::Named("cuda_multi_penalty_decomposition_reuse_count") =
        static_cast<double>(decomposition_reuse.reuse_count),
      Rcpp::Named("cuda_multi_penalty_decomposition_reuse_ratio") =
        decomposition_reuse_ratio,
      Rcpp::Named(
        "cuda_multi_penalty_decomposition_route_mismatch_count") =
          static_cast<double>(decomposition_reuse.route_mismatch_count),
      Rcpp::Named("cuda_multi_penalty_decomposition_group_size_p50") =
        decomposition_group_size_p50,
      Rcpp::Named("cuda_multi_penalty_decomposition_group_size_p95") =
        decomposition_group_size_p95,
      Rcpp::Named("cuda_multi_penalty_decomposition_group_size_max") =
        decomposition_group_size_max,
      Rcpp::Named("cuda_multi_penalty_decomposition_reuse_by_stage") =
        decomposition_reuse_by_stage,
      Rcpp::Named("cuda_multi_penalty_decomposition_reuse_by_route") =
        decomposition_reuse_by_route,
      Rcpp::Named("cuda_multi_penalty_decomposition_reuse_by_shape") =
        decomposition_reuse_by_shape,
      Rcpp::Named("cuda_multi_penalty_decomposition_reuse_by_iteration") =
        decomposition_reuse_by_iteration,
      Rcpp::Named("cuda_multi_penalty_optimizer_iteration_sum") =
        static_cast<double>(
          diagnostics.cuda_multi_penalty_optimizer_iteration_sum),
      Rcpp::Named("cuda_multi_penalty_optimizer_iteration_max") =
        diagnostics.cuda_multi_penalty_optimizer_iteration_max,
      Rcpp::Named("cuda_multi_penalty_score_call_sum") =
        static_cast<double>(diagnostics.cuda_multi_penalty_score_call_sum),
      Rcpp::Named("cuda_multi_penalty_objective_call_sum") =
        static_cast<double>(diagnostics.cuda_multi_penalty_objective_call_sum),
      Rcpp::Named("cuda_multi_penalty_step_halving_sum") =
        static_cast<double>(diagnostics.cuda_multi_penalty_step_halving_sum),
      Rcpp::Named("cuda_multi_penalty_newton_trial_sum") =
        static_cast<double>(diagnostics.cuda_multi_penalty_newton_trial_sum),
      Rcpp::Named("cuda_multi_penalty_steepest_descent_trial_sum") =
        static_cast<double>(
          diagnostics.cuda_multi_penalty_steepest_descent_trial_sum),
      Rcpp::Named("cuda_multi_penalty_boundary_probe_sum") =
        static_cast<double>(diagnostics.cuda_multi_penalty_boundary_probe_sum),
      Rcpp::Named("cuda_multi_penalty_complete_evaluation_count") =
        static_cast<double>(
          diagnostics.cuda_multi_penalty_complete_evaluation_count),
      Rcpp::Named("cuda_multi_penalty_score_only_evaluation_count") =
        static_cast<double>(
          diagnostics.cuda_multi_penalty_score_only_evaluation_count),
      Rcpp::Named("cuda_multi_penalty_guarded_qr_evaluation_count") =
        static_cast<double>(
          diagnostics.cuda_multi_penalty_guarded_qr_evaluation_count),
      Rcpp::Named("cuda_multi_penalty_stable_svd_evaluation_count") =
        static_cast<double>(
          diagnostics.cuda_multi_penalty_stable_svd_evaluation_count),
      Rcpp::Named("cuda_multi_penalty_selected_evaluation_reuse_count") =
        static_cast<double>(
          diagnostics.cuda_multi_penalty_selected_evaluation_reuse_count),
      Rcpp::Named("cuda_multi_penalty_stability_replay_target_count") =
        static_cast<double>(
          diagnostics.cuda_multi_penalty_stability_replay_target_count),
      Rcpp::Named("cuda_multi_penalty_stability_replay_selected_count") =
        static_cast<double>(
          diagnostics.cuda_multi_penalty_stability_replay_selected_count),
      Rcpp::Named("cuda_multi_penalty_terminal_confirmation_count") =
        static_cast<double>(
          diagnostics.cuda_multi_penalty_terminal_confirmation_count),
      Rcpp::Named("cuda_multi_penalty_hessian_eigensolver_count") =
        static_cast<double>(
          diagnostics.cuda_multi_penalty_hessian_eigensolver_count),
      Rcpp::Named("cuda_multi_penalty_penalty_factor_cycles") =
        static_cast<double>(diagnostics.cuda_multi_penalty_penalty_factor_cycles),
      Rcpp::Named("cuda_multi_penalty_qr_svd_cycles") =
        static_cast<double>(diagnostics.cuda_multi_penalty_qr_svd_cycles),
      Rcpp::Named("cuda_multi_penalty_qr_bidiagonal_reduction_cycles") =
        static_cast<double>(
          diagnostics.cuda_multi_penalty_qr_bidiagonal_reduction_cycles),
      Rcpp::Named("cuda_multi_penalty_qr_factorization_cycles") =
        static_cast<double>(
          diagnostics.cuda_multi_penalty_qr_factorization_cycles),
      Rcpp::Named("cuda_multi_penalty_q_generation_cycles") =
        static_cast<double>(
          diagnostics.cuda_multi_penalty_q_generation_cycles),
      Rcpp::Named("cuda_multi_penalty_qr_guard_cycles") =
        static_cast<double>(diagnostics.cuda_multi_penalty_qr_guard_cycles),
      Rcpp::Named("cuda_multi_penalty_stable_bidiagonal_reduction_cycles") =
        static_cast<double>(
          diagnostics
            .cuda_multi_penalty_stable_bidiagonal_reduction_cycles),
      Rcpp::Named("cuda_multi_penalty_bidiagonal_svd_cycles") =
        static_cast<double>(diagnostics.cuda_multi_penalty_bidiagonal_svd_cycles),
      Rcpp::Named("cuda_multi_penalty_svd_vector_postback_cycles") =
        static_cast<double>(
          diagnostics.cuda_multi_penalty_svd_vector_postback_cycles),
      Rcpp::Named("cuda_multi_penalty_left_vector_product_cycles") =
        static_cast<double>(
          diagnostics.cuda_multi_penalty_left_vector_product_cycles),
      Rcpp::Named("cuda_multi_penalty_score_construction_cycles") =
        static_cast<double>(
          diagnostics.cuda_multi_penalty_score_construction_cycles),
      Rcpp::Named("cuda_multi_penalty_derivative_hessian_cycles") =
        static_cast<double>(
          diagnostics.cuda_multi_penalty_derivative_hessian_cycles),
      Rcpp::Named("cuda_multi_penalty_stability_replay_discarded_cycles") =
        static_cast<double>(
          diagnostics.cuda_multi_penalty_stability_replay_discarded_cycles),
      Rcpp::Named("cuda_multi_penalty_terminal_confirmation_cycles") =
        static_cast<double>(
          diagnostics.cuda_multi_penalty_terminal_confirmation_cycles),
      Rcpp::Named("cuda_optimizer_kernel_launch_count") =
        diagnostics.cuda_optimizer_kernel_launch_count,
      Rcpp::Named("cuda_optimizer_host_boundary_count") =
        diagnostics.cuda_optimizer_host_boundary_count,
      Rcpp::Named("cuda_residual_batch_count") =
        diagnostics.cuda_residual_batch_count,
      Rcpp::Named("cuda_exact_screen_residual_batch_count") =
        diagnostics.cuda_exact_screen_residual_batch_count,
      Rcpp::Named("cuda_exact_screen_residual_target_count") =
        diagnostics.cuda_exact_screen_residual_target_count,
      Rcpp::Named("cuda_exact_screen_component_count") =
        diagnostics.cuda_exact_screen_component_count,
      Rcpp::Named("cuda_exact_screen_pair_count") =
        diagnostics.cuda_exact_screen_pair_count,
      Rcpp::Named("cuda_guard_refinement_residual_batch_count") =
        diagnostics.cuda_guard_refinement_residual_batch_count,
      Rcpp::Named("cuda_guard_refinement_residual_target_count") =
        diagnostics.cuda_guard_refinement_residual_target_count,
      Rcpp::Named("cuda_guard_refinement_component_count") =
        diagnostics.cuda_guard_refinement_component_count,
      Rcpp::Named("cuda_guard_refinement_pair_count") =
        diagnostics.cuda_guard_refinement_pair_count,
      Rcpp::Named("exact_residual_all_miss_batch_count") =
        diagnostics.exact_residual_all_miss_batch_count,
      Rcpp::Named("exact_residual_all_miss_target_count") =
        diagnostics.exact_residual_all_miss_target_count,
      Rcpp::Named("exact_residual_mixed_batch_count") =
        diagnostics.exact_residual_mixed_batch_count,
      Rcpp::Named("exact_residual_mixed_target_count") =
        diagnostics.exact_residual_mixed_target_count,
      Rcpp::Named("exact_residual_all_hit_new_cohort_batch_count") =
        diagnostics.exact_residual_all_hit_new_cohort_batch_count,
      Rcpp::Named("exact_residual_all_hit_new_cohort_target_count") =
        diagnostics.exact_residual_all_hit_new_cohort_target_count,
      Rcpp::Named("exact_residual_all_hit_repeated_cohort_batch_count") =
        diagnostics.exact_residual_all_hit_repeated_cohort_batch_count,
      Rcpp::Named("exact_residual_all_hit_repeated_cohort_target_count") =
        diagnostics.exact_residual_all_hit_repeated_cohort_target_count,
      Rcpp::Named("logical_residual_requests") =
        diagnostics.logical_residual_request_count,
      Rcpp::Named("physical_residual_fits") =
        diagnostics.physical_residual_fit_count,
      Rcpp::Named("excess_residual_fit_count") =
        diagnostics.physical_residual_fit_count - unique_residual_key_count,
      Rcpp::Named("cuda_dcov_component_count") =
        diagnostics.cuda_dcov_component_count,
      Rcpp::Named("cuda_dcov_pair_count") = diagnostics.cuda_dcov_pair_count,
      Rcpp::Named("cuda_gamma_pvalue_count") =
        diagnostics.cuda_gamma_pvalue_count,
      Rcpp::Named("guarded_pair_count") = diagnostics.guarded_pair_count,
      Rcpp::Named("unique_prepared_s_key_count") =
        unique_prepared_s_key_count,
      Rcpp::Named("physical_prepared_s_key_count") =
        physical_prepared_s_key_count,
      Rcpp::Named("speculative_prepared_s_build_count") =
        physical_prepared_s_key_count - unique_prepared_s_key_count,
      Rcpp::Named("unique_target_key_count") = unique_target_key_count,
      Rcpp::Named("unique_residual_key_count") = unique_residual_key_count,
      Rcpp::Named("native_setup_count") = diagnostics.native_setup_count,
      Rcpp::Named("excess_native_setup_build_count") =
        diagnostics.native_setup_count - physical_prepared_s_key_count,
      Rcpp::Named("native_setup_cache_capacity") = kPreparedCacheCapacity,
      Rcpp::Named("native_setup_host_cache_capacity") =
        kHostPreparedCacheCapacity,
      Rcpp::Named("native_setup_single_penalty_cache_capacity") =
        kSinglePenaltyPreparedCacheCapacity,
      Rcpp::Named("native_setup_multi_penalty_cache_capacity") =
        kMultiPenaltyPreparedCacheCapacity,
      Rcpp::Named("native_setup_cache_warm_start_entries") = 0,
      Rcpp::Named("native_setup_cache_request_count") =
        diagnostics.native_setup_cache_request_count,
      Rcpp::Named("native_setup_cache_hit_count") =
        diagnostics.native_setup_cache_hit_count,
      Rcpp::Named("native_setup_device_cache_hit_count") =
        diagnostics.native_setup_device_cache_hit_count,
      Rcpp::Named("native_setup_host_cache_hit_count") =
        diagnostics.native_setup_host_cache_hit_count,
      Rcpp::Named("native_setup_cache_miss_count") =
        diagnostics.native_setup_cache_miss_count,
      Rcpp::Named("native_setup_cache_eviction_count") =
        diagnostics.native_setup_cache_eviction_count,
      Rcpp::Named("native_setup_host_cache_eviction_count") =
        diagnostics.native_setup_host_cache_eviction_count,
      Rcpp::Named("native_setup_host_cache_peak_entries") =
        diagnostics.native_setup_host_cache_peak_entries,
      Rcpp::Named("native_setup_device_rehydrate_count") =
        diagnostics.native_setup_device_rehydrate_count,
      Rcpp::Named("native_setup_univariate_primitive_request_count") =
        diagnostics.native_setup_univariate_primitive_request_count,
      Rcpp::Named("native_setup_univariate_primitive_hit_count") =
        diagnostics.native_setup_univariate_primitive_hit_count,
      Rcpp::Named("native_setup_univariate_primitive_build_count") =
        diagnostics.native_setup_univariate_primitive_build_count,
      Rcpp::Named("native_setup_univariate_primitive_cache_capacity") =
        diagnostics.native_setup_univariate_primitive_cache_capacity,
      Rcpp::Named("native_setup_univariate_primitive_cache_peak_entries") =
        diagnostics.native_setup_univariate_primitive_cache_peak_entries,
      Rcpp::Named("component_cache_capacity") =
        kReferenceComponentCapacity,
      Rcpp::Named("component_cache_warm_start_entries") = 0,
      Rcpp::Named("residual_cache_warm_start_entries") = 0,
      Rcpp::Named("method_residual_cache_capacity_entries") =
        static_cast<double>(
          diagnostics.method_residual_cache_capacity_entries),
      Rcpp::Named("method_residual_cache_device_bytes") =
        static_cast<double>(diagnostics.method_residual_cache_device_bytes),
      Rcpp::Named("method_residual_cache_lookup_count") =
        diagnostics.method_residual_cache_lookup_count,
      Rcpp::Named("method_residual_cache_hit_count") =
        diagnostics.method_residual_cache_hit_count,
      Rcpp::Named("method_residual_cache_insert_count") =
        diagnostics.method_residual_cache_insert_count,
      Rcpp::Named("method_residual_cache_eviction_count") =
        diagnostics.method_residual_cache_eviction_count,
      Rcpp::Named("method_residual_cache_all_hit_batch_count") =
        diagnostics.method_residual_cache_all_hit_batch_count,
      Rcpp::Named("method_residual_cache_bypassed_target_count") =
        diagnostics.method_residual_cache_bypassed_target_count,
      Rcpp::Named("method_residual_cache_gather_d2d_bytes") =
        static_cast<double>(
          diagnostics.method_residual_cache_gather_d2d_bytes),
      Rcpp::Named("method_execution_context_call_count") =
        diagnostics.method_execution_context_call_count,
      Rcpp::Named("method_execution_context_reuse_count") =
        diagnostics.method_execution_context_reuse_count,
      Rcpp::Named("method_execution_context_buffer_growth_count") =
        diagnostics.method_execution_context_buffer_growth_count,
      Rcpp::Named("method_execution_context_peak_device_bytes") =
        static_cast<double>(
          diagnostics.method_execution_context_peak_device_bytes),
      Rcpp::Named("method_permutation_table_build_count") =
        diagnostics.method_permutation_table_build_count,
      Rcpp::Named("method_permutation_table_growth_count") =
        diagnostics.method_permutation_table_growth_count,
      Rcpp::Named("method_permutation_scratch_growth_count") =
        diagnostics.method_permutation_scratch_growth_count,
      Rcpp::Named("method_permutation_table_value_count") =
        static_cast<double>(
          diagnostics.method_permutation_table_value_count),
      Rcpp::Named("method_permutation_peak_table_values") =
        static_cast<double>(
          diagnostics.method_permutation_peak_table_values),
      Rcpp::Named("method_permutation_table_host_ms") =
        diagnostics.method_permutation_table_host_ms,
      Rcpp::Named("method_request_identity_build_host_ms") =
        diagnostics.method_request_identity_build_host_ms,
      Rcpp::Named("method_request_identity_validation_host_ms") =
        diagnostics.method_request_identity_validation_host_ms,
      Rcpp::Named("method_permutation_payload_validation_scan_count") =
        diagnostics.method_permutation_payload_validation_scan_count,
      Rcpp::Named("method_permutation_payload_validation_scan_bytes") =
        static_cast<double>(
          diagnostics.method_permutation_payload_validation_scan_bytes),
      Rcpp::Named("method_permutation_attestation_count") =
        diagnostics.method_permutation_attestation_count,
      Rcpp::Named("sha256_backend") = full_cuda_ci_sha256_backend(),
      Rcpp::Named("strict_hsic_gamma_residual_route") =
        diagnostics.strict_hsic_gamma_residual_route,
      Rcpp::Named("strict_permutation_residual_route") =
        diagnostics.strict_permutation_residual_route,
      Rcpp::Named("method_permutation_inline_r_index_requested") =
        diagnostics.method_permutation_inline_r_index_requested,
      Rcpp::Named("method_permutation_inline_r_index_active") =
        diagnostics.method_permutation_inline_r_index_active,
      Rcpp::Named("method_permutation_inline_r_index_count") =
        static_cast<double>(
          diagnostics.method_permutation_inline_r_index_count),
      Rcpp::Named("method_permutation_inline_r_draw_count") =
        static_cast<double>(diagnostics.method_permutation_inline_r_draw_count),
      Rcpp::Named("setup_optimizer_pipeline_enabled") =
        diagnostics.setup_optimizer_pipeline_enabled,
      Rcpp::Named("setup_optimizer_pipeline_window_count") =
        diagnostics.setup_optimizer_pipeline_window_count,
      Rcpp::Named("setup_optimizer_pipeline_peak_pending_count") =
        diagnostics.setup_optimizer_pipeline_peak_pending_count,
      Rcpp::Named("setup_optimizer_pipeline_producer_delay_us") =
        diagnostics.setup_optimizer_pipeline_producer_delay_us,
      Rcpp::Named("setup_optimizer_pipeline_producer_delay_count") =
        diagnostics.setup_optimizer_pipeline_producer_delay_count,
      Rcpp::Named("setup_optimizer_pipeline_producer_delay_ms") =
        diagnostics.setup_optimizer_pipeline_producer_delay_ms,
      Rcpp::Named("setup_optimizer_pipeline_prepare_ms") =
        diagnostics.setup_optimizer_pipeline_prepare_ms,
      Rcpp::Named("setup_optimizer_pipeline_device_prepare_ms") =
        diagnostics.setup_optimizer_pipeline_device_prepare_ms,
      Rcpp::Named("setup_optimizer_pipeline_wait_ms") =
        diagnostics.setup_optimizer_pipeline_wait_ms,
      Rcpp::Named("setup_optimizer_pipeline_overlap_ms") =
        diagnostics.setup_optimizer_pipeline_overlap_ms,
      Rcpp::Named("setup_optimizer_pipeline_level_wall_ms") =
        diagnostics.setup_optimizer_pipeline_level_wall_ms,
      Rcpp::Named("fixed_sp_root_cache_enabled") =
        diagnostics.fixed_sp_root_cache_enabled,
      Rcpp::Named("fixed_sp_root_cache_runtime_count") =
        diagnostics.fixed_sp_root_cache_runtime_count,
      Rcpp::Named("fixed_sp_root_cache_capacity_entries_per_runtime") =
        static_cast<double>(
          diagnostics.fixed_sp_root_cache_capacity_entries_per_runtime),
      Rcpp::Named("fixed_sp_root_cache_capacity_entries_total") =
        static_cast<double>(
          diagnostics.fixed_sp_root_cache_capacity_entries_total),
      Rcpp::Named("fixed_sp_root_cache_capacity_bytes_total") =
        static_cast<double>(
          diagnostics.fixed_sp_root_cache_capacity_bytes_total),
      Rcpp::Named("fixed_sp_root_cache_lookup_count") =
        static_cast<double>(diagnostics.fixed_sp_root_cache_lookup_count),
      Rcpp::Named("fixed_sp_root_cache_hit_count") =
        static_cast<double>(diagnostics.fixed_sp_root_cache_hit_count),
      Rcpp::Named("fixed_sp_root_cache_miss_count") =
        static_cast<double>(diagnostics.fixed_sp_root_cache_miss_count),
      Rcpp::Named("fixed_sp_root_cache_insert_count") =
        static_cast<double>(diagnostics.fixed_sp_root_cache_insert_count),
      Rcpp::Named("fixed_sp_root_cache_bypass_count") =
        static_cast<double>(diagnostics.fixed_sp_root_cache_bypass_count),
      Rcpp::Named("fixed_sp_root_cache_identity_rejection_count") =
        static_cast<double>(
          diagnostics.fixed_sp_root_cache_identity_rejection_count),
      Rcpp::Named("fixed_sp_root_cache_entries") =
        static_cast<double>(diagnostics.fixed_sp_root_cache_entries),
      Rcpp::Named("fixed_sp_root_cache_peak_entries") =
        static_cast<double>(diagnostics.fixed_sp_root_cache_peak_entries),
      Rcpp::Named("fixed_sp_root_cache_device_bytes") =
        static_cast<double>(diagnostics.fixed_sp_root_cache_device_bytes),
      Rcpp::Named("fixed_sp_root_cache_peak_device_bytes") =
        static_cast<double>(
          diagnostics.fixed_sp_root_cache_peak_device_bytes),
      Rcpp::Named("fixed_sp_root_cache_hit_d2d_bytes") =
        static_cast<double>(diagnostics.fixed_sp_root_cache_hit_d2d_bytes),
      Rcpp::Named("fixed_sp_root_cache_insert_d2d_bytes") =
        static_cast<double>(
          diagnostics.fixed_sp_root_cache_insert_d2d_bytes),
      Rcpp::Named("component_cache_request_count") =
        diagnostics.component_cache_request_count,
      Rcpp::Named("component_cache_hit_count") =
        diagnostics.component_cache_hit_count,
      Rcpp::Named("component_cache_miss_count") =
        diagnostics.component_cache_miss_count,
      Rcpp::Named("component_cache_eviction_count") =
        diagnostics.component_cache_eviction_count,
      Rcpp::Named("method_component_cache_lookup_count") =
        diagnostics.method_component_cache_lookup_count,
      Rcpp::Named("method_component_cache_request_count") =
        diagnostics.method_component_cache_request_count,
      Rcpp::Named("method_component_cache_hit_count") =
        diagnostics.method_component_cache_hit_count,
      Rcpp::Named("method_component_cache_miss_count") =
        diagnostics.method_component_cache_miss_count,
      Rcpp::Named("method_component_cache_insert_count") =
        diagnostics.method_component_cache_insert_count,
      Rcpp::Named("method_component_cache_eviction_count") =
        diagnostics.method_component_cache_eviction_count,
      Rcpp::Named("method_component_cache_capacity_entries") =
        static_cast<double>(
          diagnostics.method_component_cache_capacity_entries),
      Rcpp::Named("method_component_cache_device_bytes") =
        static_cast<double>(diagnostics.method_component_cache_device_bytes),
      Rcpp::Named("method_component_cache_gather_d2d_bytes") =
        static_cast<double>(
          diagnostics.method_component_cache_gather_d2d_bytes),
      Rcpp::Named("method_component_cache_store_d2d_bytes") =
        static_cast<double>(
          diagnostics.method_component_cache_store_d2d_bytes),
      Rcpp::Named("result_cache_capacity") =
        diagnostics.result_cache_capacity,
      Rcpp::Named("result_cache_warm_start_entries") =
        diagnostics.result_cache_warm_start_entries,
      Rcpp::Named("result_cache_dataset_warm_start_entries") =
        diagnostics.result_cache_dataset_warm_start_entries,
      Rcpp::Named("result_cache_request_count") =
        diagnostics.result_cache_request_count,
      Rcpp::Named("result_cache_hit_count") =
        diagnostics.result_cache_hit_count,
      Rcpp::Named("result_cache_preexisting_hit_count") =
        diagnostics.result_cache_preexisting_hit_count,
      Rcpp::Named("result_cache_miss_count") =
        diagnostics.result_cache_miss_count,
      Rcpp::Named("result_cache_insert_count") =
        diagnostics.result_cache_insert_count,
      Rcpp::Named("result_cache_eviction_count") =
        diagnostics.result_cache_eviction_count,
      Rcpp::Named("target_cache_capacity") =
        diagnostics.target_cache_capacity,
      Rcpp::Named("target_cache_warm_start_entries") =
        diagnostics.target_cache_warm_start_entries,
      Rcpp::Named("target_cache_dataset_warm_start_entries") =
        diagnostics.target_cache_dataset_warm_start_entries,
      Rcpp::Named("target_cache_request_count") =
        diagnostics.target_cache_request_count,
      Rcpp::Named("target_cache_hit_count") =
        diagnostics.target_cache_hit_count,
      Rcpp::Named("target_cache_preexisting_hit_count") =
        diagnostics.target_cache_preexisting_hit_count,
      Rcpp::Named("target_cache_miss_count") =
        diagnostics.target_cache_miss_count,
      Rcpp::Named("target_cache_insert_count") =
        diagnostics.target_cache_insert_count,
      Rcpp::Named("target_cache_eviction_count") =
        diagnostics.target_cache_eviction_count,
      Rcpp::Named("dataset_cache_epoch_at_start") =
        static_cast<double>(diagnostics.dataset_cache_epoch_at_start),
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
      Rcpp::Named("native_setup_conditioning_copy_ms") =
        diagnostics.native_setup_conditioning_copy_ms,
      Rcpp::Named("native_setup_input_validation_ms") =
        diagnostics.native_setup_input_validation_ms,
      Rcpp::Named("native_setup_smooth_build_ms") =
        diagnostics.native_setup_smooth_build_ms,
      Rcpp::Named("native_setup_block_assembly_ms") =
        diagnostics.native_setup_block_assembly_ms,
      Rcpp::Named("native_setup_gram_ms") = diagnostics.native_setup_gram_ms,
      Rcpp::Named("native_setup_fingerprint_ms") =
        diagnostics.native_setup_fingerprint_ms,
      Rcpp::Named("native_setup_result_packaging_ms") =
        diagnostics.native_setup_result_packaging_ms,
      Rcpp::Named("native_setup_geometry_input_ms") =
        diagnostics.native_setup_geometry_input_ms,
      Rcpp::Named("native_setup_geometry_qr_ms") =
        diagnostics.native_setup_geometry_qr_ms,
      Rcpp::Named("native_setup_geometry_penalty_ms") =
        diagnostics.native_setup_geometry_penalty_ms,
      Rcpp::Named("native_setup_geometry_initial_sp_ms") =
        diagnostics.native_setup_geometry_initial_sp_ms,
      Rcpp::Named("native_setup_geometry_packaging_ms") =
        diagnostics.native_setup_geometry_packaging_ms,
      Rcpp::Named("native_setup_fixed_sp_h2d_ms") =
        diagnostics.native_setup_fixed_sp_h2d_ms,
      Rcpp::Named("native_setup_single_penalty_geometry_ms") =
        diagnostics.native_setup_single_penalty_geometry_ms,
      Rcpp::Named("native_setup_context_overhead_ms") =
        diagnostics.native_setup_context_overhead_ms,
      Rcpp::Named("native_setup_device_rehydrate_ms") =
        diagnostics.native_setup_device_rehydrate_ms,
      Rcpp::Named("cuda_optimizer_host_ms") =
        diagnostics.cuda_optimizer_host_ms,
      Rcpp::Named("cuda_single_penalty_optimizer_host_ms") =
        diagnostics.cuda_single_penalty_optimizer_host_ms,
      Rcpp::Named("cuda_multi_penalty_optimizer_host_ms") =
        diagnostics.cuda_multi_penalty_optimizer_host_ms,
      Rcpp::Named("cuda_multi_penalty_optimizer_summed_setup_host_ms") =
        diagnostics.cuda_multi_penalty_optimizer_summed_setup_host_ms,
      Rcpp::Named("cuda_multi_penalty_optimizer_max_setup_host_ms") =
        diagnostics.cuda_multi_penalty_optimizer_max_setup_host_ms,
      Rcpp::Named("cuda_multi_penalty_prepared_build_ms") =
        diagnostics.cuda_multi_penalty_prepared_build_ms,
      Rcpp::Named("cuda_single_penalty_optimizer_cuda_ms") =
        diagnostics.cuda_single_penalty_optimizer_cuda_ms,
      Rcpp::Named("prefill_optimizer_host_ms") =
        prefill.optimizer_host_ms,
      Rcpp::Named("prefill_batch_wall_ms") = prefill.batch_wall_ms,
      Rcpp::Named("frontier_optimizer_host_ms") =
        diagnostics.frontier_optimizer_host_ms,
      Rcpp::Named("singleton_padding_batch_host_ms") =
        diagnostics.singleton_padding_batch_host_ms,
      Rcpp::Named("cuda_residual_solve_host_ms") =
        diagnostics.cuda_residual_solve_host_ms,
      Rcpp::Named("cuda_single_penalty_residual_solve_host_ms") =
        diagnostics.cuda_single_penalty_residual_solve_host_ms,
      Rcpp::Named("cuda_multi_penalty_residual_solve_host_ms") =
        diagnostics.cuda_multi_penalty_residual_solve_host_ms,
      Rcpp::Named("cuda_exact_screen_residual_solve_host_ms") =
        diagnostics.cuda_exact_screen_residual_solve_host_ms,
      Rcpp::Named("cuda_guard_refinement_residual_solve_host_ms") =
        diagnostics.cuda_guard_refinement_residual_solve_host_ms,
      Rcpp::Named("cuda_exact_screen_component_build_ms") =
        diagnostics.cuda_exact_screen_component_build_ms,
      Rcpp::Named("exact_residual_all_miss_solve_host_ms") =
        diagnostics.exact_residual_all_miss_solve_host_ms,
      Rcpp::Named("exact_residual_all_miss_component_build_ms") =
        diagnostics.exact_residual_all_miss_component_build_ms,
      Rcpp::Named("exact_residual_mixed_solve_host_ms") =
        diagnostics.exact_residual_mixed_solve_host_ms,
      Rcpp::Named("exact_residual_mixed_component_build_ms") =
        diagnostics.exact_residual_mixed_component_build_ms,
      Rcpp::Named("exact_residual_all_hit_new_cohort_solve_host_ms") =
        diagnostics.exact_residual_all_hit_new_cohort_solve_host_ms,
      Rcpp::Named("exact_residual_all_hit_new_cohort_component_build_ms") =
        diagnostics.exact_residual_all_hit_new_cohort_component_build_ms,
      Rcpp::Named("exact_residual_all_hit_repeated_cohort_solve_host_ms") =
        diagnostics.exact_residual_all_hit_repeated_cohort_solve_host_ms,
      Rcpp::Named(
        "exact_residual_all_hit_repeated_cohort_component_build_ms") =
          diagnostics
            .exact_residual_all_hit_repeated_cohort_component_build_ms,
      Rcpp::Named("cuda_dcov_metadata_h2d_ms") =
        diagnostics.cuda_dcov_metadata_h2d_ms,
      Rcpp::Named("cuda_dcov_component_build_ms") =
        diagnostics.cuda_dcov_component_build_ms,
      Rcpp::Named("cuda_dcov_pair_gamma_ms") =
        diagnostics.cuda_dcov_pair_gamma_ms,
      Rcpp::Named("cuda_dcov_compact_d2h_ms") =
        diagnostics.cuda_dcov_compact_d2h_ms,
      Rcpp::Named("cuda_dcov_teardown_host_ms") =
        diagnostics.cuda_dcov_teardown_host_ms,
      Rcpp::Named("cuda_dcov_host_ms") = diagnostics.cuda_dcov_host_ms,
      Rcpp::Named("runtime_workspace_bytes") =
        static_cast<double>(runtime_workspace_bytes),
      Rcpp::Named("elapsed_sec") = elapsed_ms(run_started) / 1000.0,
      Rcpp::Named("authority_gate_pass") = authority_clean));
}

Rcpp::List full_cuda_ci_one_call_skeleton(
    const Rcpp::NumericMatrix& data,
    double alpha,
    int max_conditioning_size,
    double index,
    int num_col,
    const std::string& trace_level,
    bool compatible_cuda_strict) {
  FullCudaCiOneCallMethodOptions method_options;
  return full_cuda_ci_one_call_skeleton_method(
    data, alpha, max_conditioning_size, index, num_col, trace_level,
    compatible_cuda_strict, method_options);
}

Rcpp::List full_cuda_ci_one_call_cache_control(
    const std::string& action,
    int capacity) {
  CompactResultCache::Snapshot value;
  TargetStateCache::Snapshot target_value;
  if (action == "info") {
    require(capacity == -1,
            "compact result cache info does not accept a capacity");
    value = compact_result_cache().snapshot();
    target_value = target_state_cache().snapshot();
  } else if (action == "reset") {
    require(capacity == -1,
            "compact result cache reset does not accept a capacity");
    value = compact_result_cache().reset();
    target_value = target_state_cache().reset();
    dataset_cache_epoch().fetch_add(1U, std::memory_order_acq_rel);
  } else if (action == "configure") {
    value = compact_result_cache().configure(capacity);
    target_value = target_state_cache().snapshot();
  } else if (action == "configure_target") {
    target_value = target_state_cache().configure(capacity);
    value = compact_result_cache().snapshot();
  } else {
    throw std::runtime_error("unknown compact result cache control action");
  }
  return Rcpp::List::create(
    Rcpp::Named("schema_version") =
      "full-cuda-ci-compact-result-cache-control-v1",
    Rcpp::Named("action") = action,
    Rcpp::Named("capacity") = value.capacity,
    Rcpp::Named("entries") = value.entries,
    Rcpp::Named("total_requests") =
      static_cast<double>(value.total_requests),
    Rcpp::Named("total_hits") = static_cast<double>(value.total_hits),
    Rcpp::Named("total_misses") = static_cast<double>(value.total_misses),
    Rcpp::Named("total_inserts") = static_cast<double>(value.total_inserts),
    Rcpp::Named("total_evictions") =
      static_cast<double>(value.total_evictions),
    Rcpp::Named("generation") = static_cast<double>(value.generation),
    Rcpp::Named("target_capacity") = target_value.capacity,
    Rcpp::Named("target_entries") = target_value.entries,
    Rcpp::Named("target_total_requests") =
      static_cast<double>(target_value.total_requests),
    Rcpp::Named("target_total_hits") =
      static_cast<double>(target_value.total_hits),
    Rcpp::Named("target_total_misses") =
      static_cast<double>(target_value.total_misses),
    Rcpp::Named("target_total_inserts") =
      static_cast<double>(target_value.total_inserts),
    Rcpp::Named("target_total_evictions") =
      static_cast<double>(target_value.total_evictions),
    Rcpp::Named("target_generation") =
      static_cast<double>(target_value.generation),
    Rcpp::Named("cache_epoch") = static_cast<double>(
      dataset_cache_epoch().load(std::memory_order_acquire)));
}

Rcpp::List full_cuda_ci_one_call_cache_state(
    const Rcpp::NumericMatrix& data) {
  require(data.nrow() > 0 && data.ncol() > 0 && finite_matrix(data),
          "dataset cache state requires a finite non-empty numeric matrix");
  const std::string dataset = dataset_key(data);
  const CompactResultCache::Snapshot result =
    compact_result_cache().snapshot(dataset);
  const TargetStateCache::Snapshot target =
    target_state_cache().snapshot(dataset);
  return Rcpp::List::create(
    Rcpp::Named("schema_version") =
      "full-cuda-ci-dataset-cache-state-v2",
    Rcpp::Named("dataset_key") = dataset,
    Rcpp::Named("cache_epoch") = static_cast<double>(
      dataset_cache_epoch().load(std::memory_order_acquire)),
    Rcpp::Named("result_cache_capacity") = result.capacity,
    Rcpp::Named("result_cache_entries") = result.entries,
    Rcpp::Named("result_cache_dataset_entries") = result.dataset_entries,
    Rcpp::Named("result_cache_total_requests") =
      static_cast<double>(result.total_requests),
    Rcpp::Named("result_cache_total_hits") =
      static_cast<double>(result.total_hits),
    Rcpp::Named("result_cache_total_misses") =
      static_cast<double>(result.total_misses),
    Rcpp::Named("result_cache_total_inserts") =
      static_cast<double>(result.total_inserts),
    Rcpp::Named("result_cache_total_evictions") =
      static_cast<double>(result.total_evictions),
    Rcpp::Named("result_cache_generation") =
      static_cast<double>(result.generation),
    Rcpp::Named("target_cache_capacity") = target.capacity,
    Rcpp::Named("target_cache_entries") = target.entries,
    Rcpp::Named("target_cache_dataset_entries") = target.dataset_entries,
    Rcpp::Named("target_cache_total_requests") =
      static_cast<double>(target.total_requests),
    Rcpp::Named("target_cache_total_hits") =
      static_cast<double>(target.total_hits),
    Rcpp::Named("target_cache_total_misses") =
      static_cast<double>(target.total_misses),
    Rcpp::Named("target_cache_total_inserts") =
      static_cast<double>(target.total_inserts),
    Rcpp::Named("target_cache_total_evictions") =
      static_cast<double>(target.total_evictions),
    Rcpp::Named("target_cache_generation") =
      static_cast<double>(target.generation));
}

}  // namespace fastkpc
