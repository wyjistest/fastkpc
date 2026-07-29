#include "dcov_batch_types.hpp"
#include "dcov_exact_cpu.hpp"
#include "fastspline_basis.hpp"
#include "full_cuda_ci_contract.hpp"
#include "full_cuda_ci_semantic_abi.hpp"
#include "hsic_cpu.hpp"
#include "legacy_dcov_gamma_cpp.hpp"
#include "mgcv_multi_penalty_cpp.hpp"
#include "orientation_types.hpp"
#include "regrvonps_device.hpp"
#include "residual_backend_registry.hpp"
#include "skeleton_engine_cuda.hpp"
#include "skeleton_task_scheduler.hpp"
#include "wanpdag_engine.hpp"
#include "cuda/cuda_status.hpp"
#include "cuda/dcov_batch_cuda.hpp"
#include "cuda/fastspline_residual_cuda.hpp"
#include "cuda/full_cuda_ci_vertical.hpp"
#include "cuda/hsic_batch_cuda.hpp"
#include "cuda/legacy_dcov_spectra_matvec_cuda.hpp"
#include "cuda/mgcv_extract_fixed_sp_cuda.hpp"
#include "cuda/mgcv_fixed_sp_runtime.hpp"
#include "cuda/mgcv_multi_penalty_gcv.hpp"
#include "cuda/mgcv_single_penalty_gcv.hpp"

#include <Rcpp.h>
#include <R_ext/Rdynload.h>
#include <Eigen/Core>
#include <Spectra/MatOp/DenseSymMatProd.h>
#include <Spectra/SymEigsSolver.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#ifdef _WIN32
#include <process.h>
#else
#include <dlfcn.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#endif

namespace {

std::int64_t fixed_sp_current_pid() {
  return static_cast<std::int64_t>(getpid());
}

template <typename T>
struct FixedSpExternalHolder {
  std::shared_ptr<T> value;
  std::int64_t creator_pid = -1;
  bool owner_counted = false;
};

SEXP fixed_sp_cuda_runtime_tag() {
  static SEXP tag = Rf_install("fastkpc_fixed_sp_cuda_runtime");
  return tag;
}

using FixedSpRuntimeHolder =
  FixedSpExternalHolder<fastkpc::CudaRuntimeContext>;

enum class FixedSpOwnerKind {
  Runtime,
  Prepared,
  Residual
};

std::atomic<int> g_fixed_sp_runtime_extptr_owners{0};
std::atomic<int> g_fixed_sp_prepared_extptr_owners{0};
std::atomic<int> g_fixed_sp_residual_extptr_owners{0};

std::atomic<int>& fixed_sp_owner_counter(FixedSpOwnerKind kind) {
  switch (kind) {
  case FixedSpOwnerKind::Runtime:
    return g_fixed_sp_runtime_extptr_owners;
  case FixedSpOwnerKind::Prepared:
    return g_fixed_sp_prepared_extptr_owners;
  case FixedSpOwnerKind::Residual:
    return g_fixed_sp_residual_extptr_owners;
  }
  return g_fixed_sp_residual_extptr_owners;
}

template <typename Holder>
void fixed_sp_owner_acquire(Holder* holder, FixedSpOwnerKind kind) {
  if (holder == nullptr || holder->owner_counted) return;
  fixed_sp_owner_counter(kind).fetch_add(1, std::memory_order_relaxed);
  holder->owner_counted = true;
}

template <typename Holder>
void fixed_sp_owner_release(Holder* holder, FixedSpOwnerKind kind) {
  if (holder == nullptr || !holder->owner_counted) return;
  holder->owner_counted = false;
  std::atomic<int>& counter = fixed_sp_owner_counter(kind);
  const int previous = counter.fetch_sub(1, std::memory_order_relaxed);
  if (previous <= 0) {
    counter.store(0, std::memory_order_relaxed);
  }
}

struct NativeSymbolImageIdentity {
  std::string path;
  std::string device_major_hex;
  std::string device_minor_hex;
  std::string inode;
};

std::string fixed_sp_hex_device_component(unsigned long long value) {
  std::ostringstream stream;
  stream << std::hex << std::nouppercase << value;
  return stream.str();
}

#ifndef _WIN32
NativeSymbolImageIdentity native_symbol_image_identity(void* symbol,
                                                       const char* label) {
  Dl_info image_info;
  if (dladdr(symbol, &image_info) == 0 ||
      image_info.dli_fname == nullptr ||
      image_info.dli_fname[0] == '\0') {
    Rcpp::stop(
      std::string("fixed-sp CUDA native symbol image is unavailable: ") +
      label);
  }
  char* resolved_path = realpath(image_info.dli_fname, nullptr);
  if (resolved_path == nullptr) {
    Rcpp::stop(
      std::string("fixed-sp CUDA native symbol image path is unresolved: ") +
      label);
  }
  const std::string path(resolved_path);
  std::free(resolved_path);

  struct stat image_stat;
  if (stat(path.c_str(), &image_stat) != 0) {
    Rcpp::stop(
      std::string("fixed-sp CUDA native symbol image stat failed: ") +
      label);
  }

  return NativeSymbolImageIdentity{
    path,
    fixed_sp_hex_device_component(
      static_cast<unsigned long long>(major(image_stat.st_dev))),
    fixed_sp_hex_device_component(
      static_cast<unsigned long long>(minor(image_stat.st_dev))),
    std::to_string(static_cast<unsigned long long>(image_stat.st_ino))
  };
}
#else
NativeSymbolImageIdentity native_symbol_image_identity(void*, const char*) {
  Rcpp::stop("fixed-sp CUDA native symbol image identity requires POSIX dladdr");
}
#endif

FixedSpRuntimeHolder* fixed_sp_cuda_runtime_holder(SEXP ptr,
                                                   bool require_live) {
  if (TYPEOF(ptr) != EXTPTRSXP ||
      R_ExternalPtrTag(ptr) != fixed_sp_cuda_runtime_tag()) {
    Rcpp::stop(
      "wrong fixed-sp external pointer tag: fixed-sp CUDA runtime must be "
      "a tagged external pointer");
  }
  auto* holder =
    static_cast<FixedSpRuntimeHolder*>(R_ExternalPtrAddr(ptr));
  if (holder == nullptr || (require_live && !holder->value)) {
    Rcpp::stop("fixed-sp CUDA runtime has been freed");
  }
  return holder;
}

void fixed_sp_cuda_runtime_finalizer(SEXP ptr) {
  auto* holder =
    static_cast<FixedSpRuntimeHolder*>(R_ExternalPtrAddr(ptr));
  if (holder == nullptr) return;
  if (holder->creator_pid == fixed_sp_current_pid()) {
    try {
      fastkpc::free_fixed_sp_runtime(&holder->value);
    } catch (...) {
    }
  }
  fixed_sp_owner_release(holder, FixedSpOwnerKind::Runtime);
  delete holder;
  R_ClearExternalPtr(ptr);
}

SEXP fixed_sp_cuda_prepared_tag() {
  static SEXP tag = Rf_install("fastkpc_fixed_sp_cuda_prepared");
  return tag;
}

using FixedSpPreparedHolder =
  FixedSpExternalHolder<fastkpc::PreparedSGpuHandle>;

FixedSpPreparedHolder* fixed_sp_cuda_prepared_holder(SEXP ptr,
                                                     bool require_live) {
  if (TYPEOF(ptr) != EXTPTRSXP ||
      R_ExternalPtrTag(ptr) != fixed_sp_cuda_prepared_tag()) {
    Rcpp::stop(
      "wrong fixed-sp external pointer tag: fixed-sp CUDA prepared handle "
      "must be a tagged external pointer");
  }
  auto* holder =
    static_cast<FixedSpPreparedHolder*>(R_ExternalPtrAddr(ptr));
  if (holder == nullptr || (require_live && !holder->value)) {
    Rcpp::stop("fixed-sp CUDA prepared handle has been freed");
  }
  return holder;
}

void fixed_sp_cuda_prepared_finalizer(SEXP ptr) {
  auto* holder =
    static_cast<FixedSpPreparedHolder*>(R_ExternalPtrAddr(ptr));
  if (holder == nullptr) return;
  if (holder->creator_pid == fixed_sp_current_pid()) {
    try {
      fastkpc::free_prepared_s_gpu(&holder->value);
    } catch (...) {
    }
  }
  fixed_sp_owner_release(holder, FixedSpOwnerKind::Prepared);
  delete holder;
  R_ClearExternalPtr(ptr);
}

SEXP fixed_sp_cuda_residual_tag() {
  static SEXP tag = Rf_install("fastkpc_fixed_sp_cuda_residual");
  return tag;
}

using FixedSpResidualHolder =
  FixedSpExternalHolder<fastkpc::DeviceResidualBatch>;

FixedSpResidualHolder* fixed_sp_cuda_residual_holder(SEXP ptr,
                                                     bool require_live) {
  if (TYPEOF(ptr) != EXTPTRSXP ||
      R_ExternalPtrTag(ptr) != fixed_sp_cuda_residual_tag()) {
    Rcpp::stop(
      "wrong fixed-sp external pointer tag: fixed-sp CUDA residual token "
      "must be a tagged external pointer");
  }
  auto* holder =
    static_cast<FixedSpResidualHolder*>(R_ExternalPtrAddr(ptr));
  if (holder == nullptr) {
    if (require_live) Rcpp::stop("fixed-sp CUDA residual token has been freed");
    return nullptr;
  }
  if (require_live && !holder->value) {
    Rcpp::stop("fixed-sp CUDA residual token has been freed");
  }
  return holder;
}

void fixed_sp_cuda_residual_finalizer(SEXP ptr) {
  auto* holder =
    static_cast<FixedSpResidualHolder*>(R_ExternalPtrAddr(ptr));
  if (holder == nullptr) return;
  if (holder->creator_pid == fixed_sp_current_pid()) {
    try {
      fastkpc::free_device_residual(&holder->value);
    } catch (...) {
    }
  }
  fixed_sp_owner_release(holder, FixedSpOwnerKind::Residual);
  delete holder;
  R_ClearExternalPtr(ptr);
}

SEXP multi_penalty_cuda_prepared_tag() {
  static SEXP tag = Rf_install("fastkpc_multi_penalty_cuda_prepared");
  return tag;
}

using MultiPenaltyCudaPreparedHolder =
  FixedSpExternalHolder<fastkpc::MultiPenaltyGcvCudaPrepared>;

MultiPenaltyCudaPreparedHolder* multi_penalty_cuda_prepared_holder(
    SEXP ptr, bool require_live) {
  if (TYPEOF(ptr) != EXTPTRSXP ||
      R_ExternalPtrTag(ptr) != multi_penalty_cuda_prepared_tag()) {
    Rcpp::stop(
      "multi-penalty CUDA prepared handle has the wrong external pointer tag");
  }
  auto* holder = static_cast<MultiPenaltyCudaPreparedHolder*>(
    R_ExternalPtrAddr(ptr));
  if (holder == nullptr || (require_live && !holder->value)) {
    Rcpp::stop("multi-penalty CUDA prepared handle has been freed");
  }
  return holder;
}

void multi_penalty_cuda_prepared_finalizer(SEXP ptr) {
  auto* holder = static_cast<MultiPenaltyCudaPreparedHolder*>(
    R_ExternalPtrAddr(ptr));
  if (holder == nullptr) return;
  if (holder->creator_pid == fixed_sp_current_pid()) holder->value.reset();
  delete holder;
  R_ClearExternalPtr(ptr);
}

SEXP multi_penalty_cuda_residual_tag() {
  static SEXP tag = Rf_install("fastkpc_multi_penalty_cuda_residual");
  return tag;
}

using MultiPenaltyCudaResidualHolder =
  FixedSpExternalHolder<fastkpc::MultiPenaltyGcvCudaResidualBatch>;

MultiPenaltyCudaResidualHolder* multi_penalty_cuda_residual_holder(
    SEXP ptr, bool require_live) {
  if (TYPEOF(ptr) != EXTPTRSXP ||
      R_ExternalPtrTag(ptr) != multi_penalty_cuda_residual_tag()) {
    Rcpp::stop(
      "multi-penalty CUDA residual has the wrong external pointer tag");
  }
  auto* holder = static_cast<MultiPenaltyCudaResidualHolder*>(
    R_ExternalPtrAddr(ptr));
  if (holder == nullptr || (require_live && !holder->value)) {
    Rcpp::stop("multi-penalty CUDA residual token has been freed");
  }
  return holder;
}

void multi_penalty_cuda_residual_finalizer(SEXP ptr) {
  auto* holder = static_cast<MultiPenaltyCudaResidualHolder*>(
    R_ExternalPtrAddr(ptr));
  if (holder == nullptr) return;
  if (holder->creator_pid == fixed_sp_current_pid() && holder->value) {
    try {
      fastkpc::release_multi_penalty_gcv_cuda_residual(holder->value);
    } catch (...) {
    }
    holder->value.reset();
  }
  delete holder;
  R_ClearExternalPtr(ptr);
}

int scalar_integer(SEXP value, const char* name) {
  if (TYPEOF(value) != INTSXP || XLENGTH(value) != 1 ||
      INTEGER(value)[0] == NA_INTEGER || Rf_isObject(value) ||
      ATTRIB(value) != R_NilValue) {
    Rcpp::stop(std::string(name) + " must be a scalar integer");
  }
  return INTEGER(value)[0];
}

bool all_finite(Rcpp::NumericMatrix values) {
  for (double value : values) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

bool all_finite_vector(Rcpp::NumericVector values) {
  for (double value : values) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

bool has_only_attributes(SEXP value,
                         const std::vector<SEXP>& allowed_tags) {
  std::vector<SEXP> seen;
  for (SEXP node = ATTRIB(value); node != R_NilValue; node = CDR(node)) {
    const SEXP tag = TAG(node);
    if (std::find(allowed_tags.begin(), allowed_tags.end(), tag) ==
          allowed_tags.end() ||
        std::find(seen.begin(), seen.end(), tag) != seen.end()) {
      return false;
    }
    seen.push_back(tag);
  }
  return true;
}

bool is_bare_scalar_string(SEXP value) {
  return TYPEOF(value) == STRSXP && XLENGTH(value) == 1 &&
    !Rf_isObject(value) && ATTRIB(value) == R_NilValue &&
    STRING_ELT(value, 0) != NA_STRING;
}

std::string bare_scalar_string(SEXP value, const char* name) {
  if (!is_bare_scalar_string(value)) {
    Rcpp::stop(std::string(name) + " must be a bare character scalar");
  }
  return Rcpp::as<std::string>(value);
}

bool is_sha256_string(SEXP value) {
  if (!is_bare_scalar_string(value)) return false;
  const std::string text = Rcpp::as<std::string>(value);
  if (text.size() != 64U) return false;
  return std::all_of(text.begin(), text.end(), [](unsigned char character) {
    return (character >= '0' && character <= '9') ||
      (character >= 'a' && character <= 'f');
  });
}

void require_sha256_string(SEXP value, const char* name) {
  if (!is_sha256_string(value)) {
    Rcpp::stop(std::string(name) + " must be a lowercase SHA-256 string");
  }
}

int positive_scalar_integer(SEXP value, const char* name) {
  const int result = scalar_integer(value, name);
  if (result <= 0) {
    Rcpp::stop(std::string(name) + " must be positive");
  }
  return result;
}

void require_finite_double_matrix(SEXP value,
                                  int expected_rows,
                                  int expected_columns,
                                  const char* name) {
  if (TYPEOF(value) != REALSXP || !Rf_isMatrix(value) ||
      Rf_isObject(value) ||
      !has_only_attributes(value, {R_DimSymbol, R_DimNamesSymbol})) {
    Rcpp::stop(std::string(name) + " must be a finite double matrix");
  }
  SEXP dimensions = Rf_getAttrib(value, R_DimSymbol);
  if (TYPEOF(dimensions) != INTSXP || XLENGTH(dimensions) != 2 ||
      INTEGER(dimensions)[0] != expected_rows ||
      INTEGER(dimensions)[1] != expected_columns) {
    Rcpp::stop(std::string(name) + " shape mismatch");
  }
  const R_xlen_t count = XLENGTH(value);
  const double* values = REAL(value);
  for (R_xlen_t index = 0; index < count; ++index) {
    if (!std::isfinite(values[index])) {
      Rcpp::stop(std::string(name) + " must be finite");
    }
  }
}

void require_bare_double_matrix(SEXP value,
                                int expected_rows,
                                int expected_columns,
                                const char* name) {
  if (TYPEOF(value) != REALSXP || !Rf_isMatrix(value) ||
      Rf_isObject(value) ||
      !has_only_attributes(value, {R_DimSymbol})) {
    Rcpp::stop(std::string(name) + " must be a bare double matrix");
  }
  SEXP dimensions = Rf_getAttrib(value, R_DimSymbol);
  if (Rf_isObject(dimensions) || ATTRIB(dimensions) != R_NilValue) {
    Rcpp::stop(std::string(name) + " must be a bare double matrix");
  }
  if (TYPEOF(dimensions) != INTSXP || XLENGTH(dimensions) != 2 ||
      INTEGER(dimensions)[0] != expected_rows ||
      INTEGER(dimensions)[1] != expected_columns) {
    Rcpp::stop(std::string(name) + " shape mismatch");
  }
}

void require_bare_integer_vector(SEXP value,
                                 int expected_length,
                                 const char* name) {
  if (TYPEOF(value) != INTSXP || XLENGTH(value) != expected_length ||
      Rf_isObject(value) || ATTRIB(value) != R_NilValue) {
    Rcpp::stop(std::string(name) + " must be a bare integer vector");
  }
  for (int index = 0; index < expected_length; ++index) {
    if (INTEGER(value)[index] == NA_INTEGER) {
      Rcpp::stop(std::string(name) + " must not contain NA");
    }
  }
}

std::vector<std::string> bare_character_vector(
    SEXP value,
    int expected_length,
    const char* name) {
  if (TYPEOF(value) != STRSXP || XLENGTH(value) != expected_length ||
      Rf_isObject(value) || ATTRIB(value) != R_NilValue) {
    Rcpp::stop(std::string(name) + " must be a bare character vector");
  }
  std::vector<std::string> result;
  result.reserve(static_cast<std::size_t>(expected_length));
  for (int index = 0; index < expected_length; ++index) {
    if (STRING_ELT(value, index) == NA_STRING) {
      Rcpp::stop(std::string(name) + " must not contain NA");
    }
    result.emplace_back(CHAR(STRING_ELT(value, index)));
  }
  return result;
}

fastkpc::FixedSpRoute fixed_sp_route_from_string(
    const std::string& route) {
  if (route == "CHOLESKY_BATCHED") {
    return fastkpc::FixedSpRoute::CholeskyBatched;
  }
  if (route == "AUGMENTED_QR") {
    return fastkpc::FixedSpRoute::AugmentedQr;
  }
  if (route == "AUGMENTED_SVD") {
    return fastkpc::FixedSpRoute::AugmentedSvd;
  }
  Rcpp::stop("planned_route contains an unknown fixed-sp route");
  return fastkpc::FixedSpRoute::Unset;
}

std::uint32_t fixed_sp_output_mask(SEXP outputs_s) {
  if (TYPEOF(outputs_s) != STRSXP || XLENGTH(outputs_s) < 1 ||
      Rf_isObject(outputs_s) || ATTRIB(outputs_s) != R_NilValue) {
    Rcpp::stop("outputs must be a non-empty bare character vector");
  }
  std::uint32_t mask = 0;
  for (R_xlen_t index = 0; index < XLENGTH(outputs_s); ++index) {
    if (STRING_ELT(outputs_s, index) == NA_STRING) {
      Rcpp::stop("outputs must not contain NA");
    }
    const std::string output = CHAR(STRING_ELT(outputs_s, index));
    std::uint32_t bit = 0;
    if (output == "coefficients") {
      bit = fastkpc::FixedSpOutputCoefficients;
    } else if (output == "fitted") {
      bit = fastkpc::FixedSpOutputFitted;
    } else if (output == "residuals") {
      bit = fastkpc::FixedSpOutputResiduals;
    } else if (output == "rss") {
      bit = fastkpc::FixedSpOutputRss;
    } else if (output == "rhs") {
      bit = fastkpc::FixedSpOutputRhs;
    } else {
      Rcpp::stop("outputs contains an unknown fixed-sp output");
    }
    if ((mask & bit) != 0) {
      Rcpp::stop("outputs must not contain duplicates");
    }
    mask |= bit;
  }
  return mask;
}

void require_prepared_dto_fields(SEXP dto_s) {
  static const char* expected[] = {
    "schema_version", "dataset_sha256", "prepared_s_key_sha256",
    "same_S_group_id", "phase1_setup_fingerprint", "provider_fingerprint",
    "semantic_fingerprint", "representation_fingerprint",
    "prepared_s_setup_schema_version", "native_dto_schema_version",
    "data_p", "n", "coefficient_dim", "null_dim", "penalty_count", "X",
    "constraint_mode", "constraint_nullspace", "gram_matrix",
    "nullspace_gram_matrix", "penalty_blocks",
    "penalty_offsets_zero_based", "penalty_ranks",
    "penalty_sp_indices_zero_based", "penalty_sp_labels", "H",
    "weights_policy", "offset_policy"
  };
  constexpr int expected_count =
    static_cast<int>(sizeof(expected) / sizeof(expected[0]));
  if (TYPEOF(dto_s) != VECSXP || XLENGTH(dto_s) != expected_count ||
      Rf_isObject(dto_s) ||
      !has_only_attributes(dto_s, {R_NamesSymbol})) {
    Rcpp::stop("prepared DTO must have exact fields");
  }
  SEXP names = Rf_getAttrib(dto_s, R_NamesSymbol);
  if (TYPEOF(names) != STRSXP || XLENGTH(names) != expected_count ||
      Rf_isObject(names) || ATTRIB(names) != R_NilValue) {
    Rcpp::stop("prepared DTO must have exact fields");
  }
  for (int index = 0; index < expected_count; ++index) {
    if (STRING_ELT(names, index) == NA_STRING ||
        std::string(CHAR(STRING_ELT(names, index))) != expected[index]) {
      Rcpp::stop("prepared DTO must have exact fields");
    }
  }
}

void require_full_cuda_ci_vertical_request_fields(SEXP request_s) {
  static const char* expected[] = {
    "schema_version", "expected_prepared_s_key_sha256",
    "request_identity_sha256", "logical_sequence_id",
    "left_target_ordinal", "right_target_ordinal", "alpha",
    "exercise_eviction"
  };
  constexpr int expected_count =
    static_cast<int>(sizeof(expected) / sizeof(expected[0]));
  if (TYPEOF(request_s) != VECSXP || XLENGTH(request_s) != expected_count ||
      Rf_isObject(request_s) ||
      !has_only_attributes(request_s, {R_NamesSymbol})) {
    Rcpp::stop("vertical request must have exact fields");
  }
  SEXP names = Rf_getAttrib(request_s, R_NamesSymbol);
  if (TYPEOF(names) != STRSXP || XLENGTH(names) != expected_count ||
      Rf_isObject(names) || ATTRIB(names) != R_NilValue) {
    Rcpp::stop("vertical request must have exact fields");
  }
  for (int index = 0; index < expected_count; ++index) {
    if (STRING_ELT(names, index) == NA_STRING ||
        std::string(CHAR(STRING_ELT(names, index))) != expected[index]) {
      Rcpp::stop("vertical request must have exact fields");
    }
  }
}

fastkpc::FullCudaCiVerticalRequest parse_full_cuda_ci_vertical_request(
    SEXP request_s) {
  require_full_cuda_ci_vertical_request_fields(request_s);
  Rcpp::List request(request_s);
  const std::string schema = bare_scalar_string(
    request[0], "vertical request schema_version");
  if (schema != fastkpc::kFullCudaCiVerticalRequestSchemaVersion) {
    Rcpp::stop("vertical request schema_version mismatch");
  }
  require_sha256_string(
    request[1], "vertical expected_prepared_s_key_sha256");
  require_sha256_string(request[2], "vertical request_identity_sha256");

  SEXP logical_s = request[3];
  if (TYPEOF(logical_s) != REALSXP || XLENGTH(logical_s) != 1 ||
      Rf_isObject(logical_s) || ATTRIB(logical_s) != R_NilValue ||
      !std::isfinite(REAL(logical_s)[0]) || REAL(logical_s)[0] < 1.0 ||
      REAL(logical_s)[0] > 9007199254740991.0 ||
      std::floor(REAL(logical_s)[0]) != REAL(logical_s)[0]) {
    Rcpp::stop(
      "vertical logical_sequence_id must be a positive safe-53-bit double integer");
  }
  const int left_ordinal = positive_scalar_integer(
    request[4], "vertical left_target_ordinal");
  const int right_ordinal = positive_scalar_integer(
    request[5], "vertical right_target_ordinal");

  SEXP alpha_s = request[6];
  if (TYPEOF(alpha_s) != REALSXP || XLENGTH(alpha_s) != 1 ||
      Rf_isObject(alpha_s) || ATTRIB(alpha_s) != R_NilValue ||
      !std::isfinite(REAL(alpha_s)[0])) {
    Rcpp::stop("vertical alpha must be a bare finite double scalar");
  }
  SEXP eviction_s = request[7];
  if (TYPEOF(eviction_s) != LGLSXP || XLENGTH(eviction_s) != 1 ||
      Rf_isObject(eviction_s) || ATTRIB(eviction_s) != R_NilValue ||
      LOGICAL(eviction_s)[0] == NA_LOGICAL) {
    Rcpp::stop("vertical exercise_eviction must be a bare logical scalar");
  }

  fastkpc::FullCudaCiVerticalRequest result;
  result.expected_prepared_s_key_sha256 =
    Rcpp::as<std::string>(request[1]);
  result.request_identity_sha256 = Rcpp::as<std::string>(request[2]);
  result.logical_sequence_id =
    static_cast<std::uint64_t>(REAL(logical_s)[0]);
  result.left_target_index = left_ordinal - 1;
  result.right_target_index = right_ordinal - 1;
  result.alpha = REAL(alpha_s)[0];
  result.exercise_eviction = LOGICAL(eviction_s)[0] == TRUE;
  return result;
}

void require_full_cuda_ci_exact_batch_request_fields(SEXP request_s) {
  static const char* expected[] = {
    "schema_version", "expected_prepared_s_key_sha256",
    "request_identity_sha256", "logical_sequence_ids",
    "left_target_ordinals", "right_target_ordinals", "alpha",
    "component_capacity"
  };
  constexpr int expected_count =
    static_cast<int>(sizeof(expected) / sizeof(expected[0]));
  if (TYPEOF(request_s) != VECSXP || XLENGTH(request_s) != expected_count ||
      Rf_isObject(request_s) ||
      !has_only_attributes(request_s, {R_NamesSymbol})) {
    Rcpp::stop("exact batch request must have exact fields");
  }
  SEXP names = Rf_getAttrib(request_s, R_NamesSymbol);
  if (TYPEOF(names) != STRSXP || XLENGTH(names) != expected_count ||
      Rf_isObject(names) || ATTRIB(names) != R_NilValue) {
    Rcpp::stop("exact batch request must have exact fields");
  }
  for (int index = 0; index < expected_count; ++index) {
    if (STRING_ELT(names, index) == NA_STRING ||
        std::string(CHAR(STRING_ELT(names, index))) != expected[index]) {
      Rcpp::stop("exact batch request must have exact fields");
    }
  }
}

fastkpc::FullCudaCiExactBatchRequest
parse_full_cuda_ci_exact_batch_request(SEXP request_s) {
  require_full_cuda_ci_exact_batch_request_fields(request_s);
  Rcpp::List request(request_s);
  const std::string schema = bare_scalar_string(
    request[0], "exact batch request schema_version");
  if (schema != fastkpc::kFullCudaCiExactBatchRequestSchemaVersion) {
    Rcpp::stop("exact batch request schema_version mismatch");
  }
  require_sha256_string(
    request[1], "exact batch expected_prepared_s_key_sha256");
  require_sha256_string(request[2], "exact batch request_identity_sha256");

  SEXP logical_s = request[3];
  SEXP left_s = request[4];
  SEXP right_s = request[5];
  if (TYPEOF(logical_s) != REALSXP || XLENGTH(logical_s) < 1 ||
      XLENGTH(logical_s) > std::numeric_limits<int>::max() ||
      Rf_isObject(logical_s) || ATTRIB(logical_s) != R_NilValue ||
      TYPEOF(left_s) != INTSXP || XLENGTH(left_s) != XLENGTH(logical_s) ||
      Rf_isObject(left_s) || ATTRIB(left_s) != R_NilValue ||
      TYPEOF(right_s) != INTSXP ||
      XLENGTH(right_s) != XLENGTH(logical_s) || Rf_isObject(right_s) ||
      ATTRIB(right_s) != R_NilValue) {
    Rcpp::stop("exact batch pair vectors are malformed");
  }

  SEXP alpha_s = request[6];
  if (TYPEOF(alpha_s) != REALSXP || XLENGTH(alpha_s) != 1 ||
      Rf_isObject(alpha_s) || ATTRIB(alpha_s) != R_NilValue ||
      !std::isfinite(REAL(alpha_s)[0])) {
    Rcpp::stop("exact batch alpha must be a bare finite double scalar");
  }
  const int component_capacity = positive_scalar_integer(
    request[7], "exact batch component_capacity");

  fastkpc::FullCudaCiExactBatchRequest result;
  result.expected_prepared_s_key_sha256 =
    Rcpp::as<std::string>(request[1]);
  result.request_identity_sha256 = Rcpp::as<std::string>(request[2]);
  result.component_capacity = component_capacity;
  const int pair_count = static_cast<int>(XLENGTH(logical_s));
  result.pairs.reserve(static_cast<std::size_t>(pair_count));
  for (int index = 0; index < pair_count; ++index) {
    const double logical_id = REAL(logical_s)[index];
    const int left_ordinal = INTEGER(left_s)[index];
    const int right_ordinal = INTEGER(right_s)[index];
    if (!std::isfinite(logical_id) || logical_id < 1.0 ||
        logical_id > 9007199254740991.0 ||
        std::floor(logical_id) != logical_id ||
        left_ordinal == NA_INTEGER || right_ordinal == NA_INTEGER ||
        left_ordinal < 1 || right_ordinal < 1) {
      Rcpp::stop("exact batch pair vectors are malformed");
    }
    fastkpc::FullCudaCiExactBatchPairRequest pair;
    pair.logical_sequence_id = static_cast<std::uint64_t>(logical_id);
    pair.left_target_index = left_ordinal - 1;
    pair.right_target_index = right_ordinal - 1;
    pair.alpha = REAL(alpha_s)[0];
    result.pairs.push_back(pair);
  }
  return result;
}

void require_full_cuda_ci_legacy_eig_batch_request_fields(SEXP request_s) {
  static const char* expected[] = {
    "schema_version", "expected_prepared_s_key_sha256",
    "request_identity_sha256", "logical_sequence_ids",
    "left_target_ordinals", "right_target_ordinals", "alpha",
    "component_capacity", "num_col"
  };
  constexpr int expected_count =
    static_cast<int>(sizeof(expected) / sizeof(expected[0]));
  if (TYPEOF(request_s) != VECSXP || XLENGTH(request_s) != expected_count ||
      Rf_isObject(request_s) ||
      !has_only_attributes(request_s, {R_NamesSymbol})) {
    Rcpp::stop("legacy eig batch request must have exact fields");
  }
  SEXP names = Rf_getAttrib(request_s, R_NamesSymbol);
  if (TYPEOF(names) != STRSXP || XLENGTH(names) != expected_count ||
      Rf_isObject(names) || ATTRIB(names) != R_NilValue) {
    Rcpp::stop("legacy eig batch request must have exact fields");
  }
  for (int index = 0; index < expected_count; ++index) {
    if (STRING_ELT(names, index) == NA_STRING ||
        std::string(CHAR(STRING_ELT(names, index))) != expected[index]) {
      Rcpp::stop("legacy eig batch request must have exact fields");
    }
  }
}

fastkpc::FullCudaCiLegacyEigBatchRequest
parse_full_cuda_ci_legacy_eig_batch_request(SEXP request_s) {
  require_full_cuda_ci_legacy_eig_batch_request_fields(request_s);
  Rcpp::List request(request_s);
  const std::string schema = bare_scalar_string(
    request[0], "legacy eig batch request schema_version");
  if (schema != fastkpc::kFullCudaCiLegacyEigBatchRequestSchemaVersion) {
    Rcpp::stop("legacy eig batch request schema_version mismatch");
  }
  require_sha256_string(
    request[1], "legacy eig expected_prepared_s_key_sha256");
  require_sha256_string(
    request[2], "legacy eig request_identity_sha256");
  SEXP logical_s = request[3];
  SEXP left_s = request[4];
  SEXP right_s = request[5];
  if (TYPEOF(logical_s) != REALSXP || XLENGTH(logical_s) < 1 ||
      XLENGTH(logical_s) > std::numeric_limits<int>::max() ||
      Rf_isObject(logical_s) || ATTRIB(logical_s) != R_NilValue ||
      TYPEOF(left_s) != INTSXP || XLENGTH(left_s) != XLENGTH(logical_s) ||
      Rf_isObject(left_s) || ATTRIB(left_s) != R_NilValue ||
      TYPEOF(right_s) != INTSXP ||
      XLENGTH(right_s) != XLENGTH(logical_s) || Rf_isObject(right_s) ||
      ATTRIB(right_s) != R_NilValue) {
    Rcpp::stop("legacy eig batch pair vectors are malformed");
  }
  SEXP alpha_s = request[6];
  if (TYPEOF(alpha_s) != REALSXP || XLENGTH(alpha_s) != 1 ||
      Rf_isObject(alpha_s) || ATTRIB(alpha_s) != R_NilValue ||
      !std::isfinite(REAL(alpha_s)[0])) {
    Rcpp::stop("legacy eig alpha must be a bare finite double scalar");
  }
  const int component_capacity = positive_scalar_integer(
    request[7], "legacy eig component_capacity");
  const int num_col = positive_scalar_integer(
    request[8], "legacy eig num_col");

  fastkpc::FullCudaCiLegacyEigBatchRequest result;
  result.expected_prepared_s_key_sha256 =
    Rcpp::as<std::string>(request[1]);
  result.request_identity_sha256 = Rcpp::as<std::string>(request[2]);
  result.component_capacity = component_capacity;
  result.num_col = num_col;
  const int pair_count = static_cast<int>(XLENGTH(logical_s));
  result.pairs.reserve(static_cast<std::size_t>(pair_count));
  for (int index = 0; index < pair_count; ++index) {
    const double logical_id = REAL(logical_s)[index];
    const int left_ordinal = INTEGER(left_s)[index];
    const int right_ordinal = INTEGER(right_s)[index];
    if (!std::isfinite(logical_id) || logical_id < 1.0 ||
        logical_id > 9007199254740991.0 ||
        std::floor(logical_id) != logical_id ||
        left_ordinal == NA_INTEGER || right_ordinal == NA_INTEGER ||
        left_ordinal < 1 || right_ordinal < 1) {
      Rcpp::stop("legacy eig batch pair vectors are malformed");
    }
    fastkpc::FullCudaCiExactBatchPairRequest pair;
    pair.logical_sequence_id = static_cast<std::uint64_t>(logical_id);
    pair.left_target_index = left_ordinal - 1;
    pair.right_target_index = right_ordinal - 1;
    pair.alpha = REAL(alpha_s)[0];
    result.pairs.push_back(pair);
  }
  return result;
}

void legacy_dcov_spectra_matvec_handle_finalizer(SEXP ext) {
  auto* handle =
    static_cast<fastkpc::LegacyDcovSpectraMatvecCudaHandle*>(
      R_ExternalPtrAddr(ext));
  if (handle != nullptr) {
    fastkpc::legacy_dcov_spectra_matvec_cuda_handle_destroy(handle);
    R_ClearExternalPtr(ext);
  }
}

fastkpc::LegacyDcovSpectraMatvecCudaHandle*
legacy_dcov_spectra_matvec_handle_from_externalptr(SEXP ptr) {
  if (TYPEOF(ptr) != EXTPTRSXP) {
    Rcpp::stop("CUDA matvec handle must be an external pointer");
  }
  auto* handle =
    static_cast<fastkpc::LegacyDcovSpectraMatvecCudaHandle*>(
      R_ExternalPtrAddr(ptr));
  if (handle == nullptr) {
    Rcpp::stop("CUDA matvec handle has been freed");
  }
  return handle;
}

struct LegacyDcovSpectraCudaOperatorDiagnostics {
  int spectra_matvec_count = 0;
  int kernel_launch_count = 0;
  int device_matrix_reuse_count = 0;
  int device_workspace_reuse_count = 0;
  int workspace_realloc_count = 0;
  std::size_t matrix_bytes = 0;
  std::size_t workspace_bytes = 0;
  double spectra_matvec_ms = 0.0;
  double matrix_h2d_ms_during_compute = 0.0;
  double workspace_alloc_ms = 0.0;
  double h2d_ms = 0.0;
  double kernel_ms = 0.0;
  double d2h_ms = 0.0;
  double total_ms = 0.0;
};

double elapsed_ms_since(std::chrono::steady_clock::time_point start);

class LegacyDcovSpectraCudaMatProd {
 public:
  LegacyDcovSpectraCudaMatProd(
      fastkpc::LegacyDcovSpectraMatvecCudaHandle* handle,
      int n,
      LegacyDcovSpectraCudaOperatorDiagnostics* diagnostics)
      : handle_(handle), n_(n), diagnostics_(diagnostics) {}

  int rows() const { return n_; }
  int cols() const { return n_; }

  void perform_op(const double* x_in, double* y_out) const {
    const auto start = std::chrono::steady_clock::now();
    const fastkpc::LegacyDcovSpectraMatvecCudaResult result =
      fastkpc::legacy_dcov_spectra_matvec_cuda_handle_apply(
        handle_, x_in, 1);
    std::copy(result.values.begin(), result.values.end(), y_out);
    if (diagnostics_ != nullptr) {
      diagnostics_->spectra_matvec_count += 1;
      diagnostics_->kernel_launch_count += result.kernel_launch_count;
      diagnostics_->device_matrix_reuse_count +=
        result.device_matrix_reuse_count;
      diagnostics_->device_workspace_reuse_count +=
        result.device_workspace_reuse_count;
      diagnostics_->workspace_realloc_count += result.workspace_realloc_count;
      diagnostics_->matrix_bytes = result.matrix_bytes;
      diagnostics_->workspace_bytes =
        std::max(diagnostics_->workspace_bytes, result.workspace_bytes);
      diagnostics_->matrix_h2d_ms_during_compute += result.matrix_h2d_ms;
      diagnostics_->workspace_alloc_ms += result.workspace_alloc_ms;
      diagnostics_->h2d_ms += result.h2d_ms;
      diagnostics_->kernel_ms += result.kernel_ms;
      diagnostics_->d2h_ms += result.d2h_ms;
      diagnostics_->total_ms += result.total_ms;
      diagnostics_->spectra_matvec_ms += std::chrono::duration<double,
        std::milli>(std::chrono::steady_clock::now() - start).count();
    }
  }

 private:
  fastkpc::LegacyDcovSpectraMatvecCudaHandle* handle_;
  int n_;
  LegacyDcovSpectraCudaOperatorDiagnostics* diagnostics_;
};

struct LegacyDcovSpectraLowrankShadowRun {
  Eigen::VectorXd eigenvalues;
  Eigen::MatrixXd centered_vectors;
  LegacyDcovSpectraCudaOperatorDiagnostics cuda_diagnostics;
  bool converged = false;
  int nconv = 0;
  int iterations = 0;
  int info = -1;
  double eig_ms = 0.0;
  double matrix_h2d_ms = 0.0;
  double matrix_bytes = 0.0;
};

struct LegacyDcovCudaLowrankComponentRun {
  Eigen::VectorXd eigenvalues;
  Eigen::MatrixXd centered_vectors;
  LegacyDcovSpectraCudaOperatorDiagnostics cuda_diagnostics;
  bool converged = false;
  int nconv = 0;
  int iterations = 0;
  int info = -1;
  double eig_ms = 0.0;
  double matrix_h2d_ms = 0.0;
  double matrix_bytes = 0.0;
  double distance_sum = 0.0;
  double moment = 0.0;
  double distance_ms = 0.0;
  double lowrank_ms = 0.0;
  double moment_ms = 0.0;
  double total_ms = 0.0;
};

struct LegacyDcovCudaLowrankGammaRun {
  double p_value = NA_REAL;
  double nV2 = NA_REAL;
  double mean = NA_REAL;
  double variance = NA_REAL;
  double statistic = NA_REAL;
  double estimate = NA_REAL;
  double x_moment = NA_REAL;
  double y_moment = NA_REAL;
  bool converged_x = false;
  bool converged_y = false;
  int nconv_x = 0;
  int nconv_y = 0;
  int iterations_x = 0;
  int iterations_y = 0;
  int info_x = -1;
  int info_y = -1;
  double eig_ms = 0.0;
  int spectra_matvec_count = 0;
  double spectra_matvec_ms = 0.0;
  int kernel_launch_count = 0;
  int device_matrix_reuse_count = 0;
  int device_workspace_reuse_count = 0;
  int workspace_realloc_count = 0;
  double matrix_bytes = 0.0;
  double workspace_bytes = 0.0;
  double matrix_h2d_ms = 0.0;
  double matrix_h2d_ms_during_compute = 0.0;
  double workspace_alloc_ms = 0.0;
  double h2d_ms = 0.0;
  double kernel_ms = 0.0;
  double d2h_ms = 0.0;
  double total_ms = 0.0;
};

Eigen::MatrixXd legacy_dcov_distance_matrix_eigen(const double* values,
                                                  int n) {
  Eigen::MatrixXd out(n, n);
  out.diagonal().setZero();
  for (int col = 0; col < n; ++col) {
    const double value_col = values[col];
    for (int row = col + 1; row < n; ++row) {
      const double dist = std::abs(values[row] - value_col);
      out(row, col) = dist;
      out(col, row) = dist;
    }
  }
  return out;
}

void legacy_dcov_center_lowrank_vectors(Eigen::MatrixXd& vectors) {
  const Eigen::RowVectorXd means = vectors.colwise().mean();
  vectors.rowwise() -= means;
}

LegacyDcovSpectraLowrankShadowRun legacy_dcov_cpu_spectra_lowrank_shadow(
    const Eigen::MatrixXd& distance,
    int num_col,
    int ncv,
    double tol,
    int maxitr) {
  LegacyDcovSpectraLowrankShadowRun run;
  const auto eig_start = std::chrono::steady_clock::now();
  Spectra::DenseSymMatProd<double> op(distance);
  Spectra::SymEigsSolver<double, Spectra::LARGEST_MAGN,
                         Spectra::DenseSymMatProd<double>> eigs(
    &op, num_col, ncv);
  eigs.init();
  run.nconv = static_cast<int>(
    eigs.compute(maxitr, tol, Spectra::LARGEST_MAGN));
  run.iterations = static_cast<int>(eigs.num_iterations());
  run.info = static_cast<int>(eigs.info());
  run.converged = eigs.info() == Spectra::SUCCESSFUL &&
    run.nconv >= num_col;
  run.eig_ms = elapsed_ms_since(eig_start);
  if (run.converged) {
    run.eigenvalues = eigs.eigenvalues();
    run.centered_vectors = eigs.eigenvectors();
    legacy_dcov_center_lowrank_vectors(run.centered_vectors);
  }
  return run;
}

LegacyDcovSpectraLowrankShadowRun legacy_dcov_cuda_spectra_lowrank_shadow(
    const Eigen::MatrixXd& distance,
    int num_col,
    int ncv,
    double tol,
    int maxitr) {
  LegacyDcovSpectraLowrankShadowRun run;
  const int n = static_cast<int>(distance.rows());
  fastkpc::LegacyDcovSpectraMatvecCudaHandle* handle =
    fastkpc::legacy_dcov_spectra_matvec_cuda_handle_create(
      distance.data(), n);
  run.matrix_h2d_ms =
    fastkpc::legacy_dcov_spectra_matvec_cuda_handle_matrix_h2d_ms(handle);
  run.matrix_bytes = static_cast<double>(
    fastkpc::legacy_dcov_spectra_matvec_cuda_handle_matrix_bytes(handle));

  try {
    run.cuda_diagnostics.matrix_bytes =
      static_cast<std::size_t>(run.matrix_bytes);
    LegacyDcovSpectraCudaMatProd op(handle, n, &run.cuda_diagnostics);
    const auto eig_start = std::chrono::steady_clock::now();
    Spectra::SymEigsSolver<double, Spectra::LARGEST_MAGN,
                           LegacyDcovSpectraCudaMatProd> eigs(
      &op, num_col, ncv);
    eigs.init();
    run.nconv = static_cast<int>(
      eigs.compute(maxitr, tol, Spectra::LARGEST_MAGN));
    run.iterations = static_cast<int>(eigs.num_iterations());
    run.info = static_cast<int>(eigs.info());
    run.converged = eigs.info() == Spectra::SUCCESSFUL &&
      run.nconv >= num_col;
    run.eig_ms = elapsed_ms_since(eig_start);
    if (run.converged) {
      run.eigenvalues = eigs.eigenvalues();
      run.centered_vectors = eigs.eigenvectors();
      legacy_dcov_center_lowrank_vectors(run.centered_vectors);
    }
    fastkpc::legacy_dcov_spectra_matvec_cuda_handle_destroy(handle);
    handle = nullptr;
  } catch (...) {
    if (handle != nullptr) {
      fastkpc::legacy_dcov_spectra_matvec_cuda_handle_destroy(handle);
    }
    throw;
  }
  return run;
}

double legacy_dcov_weighted_cross_sum_eigen(
    const Eigen::MatrixXd& left,
    const Eigen::VectorXd& left_values,
    const Eigen::MatrixXd& right,
    const Eigen::VectorXd& right_values) {
  const Eigen::MatrixXd cross = left.transpose() * right;
  double total = 0.0;
  for (int col = 0; col < cross.cols(); ++col) {
    for (int row = 0; row < cross.rows(); ++row) {
      const double value = cross(row, col);
      total += left_values(row) * right_values(col) * value * value;
    }
  }
  return total;
}

LegacyDcovCudaLowrankComponentRun legacy_dcov_cuda_lowrank_component_compute(
    const double* values,
    int n,
    int num_col,
    int ncv,
    double tol,
    int maxitr) {
  const auto total_start = std::chrono::steady_clock::now();
  LegacyDcovCudaLowrankComponentRun out;
  const auto distance_start = std::chrono::steady_clock::now();
  const Eigen::MatrixXd distance =
    legacy_dcov_distance_matrix_eigen(values, n);
  out.distance_ms = elapsed_ms_since(distance_start);
  out.distance_sum = distance.sum();
  const auto lowrank_start = std::chrono::steady_clock::now();
  const LegacyDcovSpectraLowrankShadowRun lowrank =
    legacy_dcov_cuda_spectra_lowrank_shadow(
      distance, num_col, ncv, tol, maxitr);
  out.lowrank_ms = elapsed_ms_since(lowrank_start);
  out.converged = lowrank.converged;
  out.nconv = lowrank.nconv;
  out.iterations = lowrank.iterations;
  out.info = lowrank.info;
  out.eig_ms = lowrank.eig_ms;
  out.matrix_h2d_ms = lowrank.matrix_h2d_ms;
  out.matrix_bytes = lowrank.matrix_bytes;
  out.cuda_diagnostics = lowrank.cuda_diagnostics;
  if (!lowrank.converged) {
    out.total_ms = elapsed_ms_since(total_start);
    return out;
  }
  out.eigenvalues = lowrank.eigenvalues;
  out.centered_vectors = lowrank.centered_vectors;
  const auto moment_start = std::chrono::steady_clock::now();
  out.moment = legacy_dcov_weighted_cross_sum_eigen(
    out.centered_vectors, out.eigenvalues,
    out.centered_vectors, out.eigenvalues);
  out.moment_ms = elapsed_ms_since(moment_start);
  out.total_ms = elapsed_ms_since(total_start);
  return out;
}

LegacyDcovCudaLowrankGammaRun legacy_dcov_cuda_lowrank_gamma_from_components(
    const LegacyDcovCudaLowrankComponentRun& x_component,
    const LegacyDcovCudaLowrankComponentRun& y_component,
    int n,
    double index,
    bool include_component_diagnostics) {
  LegacyDcovCudaLowrankGammaRun out;
  if (!x_component.converged || !y_component.converged) {
    Rcpp::stop("CUDA Spectra lowrank did not converge");
  }
  const double n_double = static_cast<double>(n);
  out.nV2 = legacy_dcov_weighted_cross_sum_eigen(
    x_component.centered_vectors, x_component.eigenvalues,
    y_component.centered_vectors, y_component.eigenvalues) / n_double;
  out.mean =
    (x_component.distance_sum / (n_double * n_double)) *
    (y_component.distance_sum / (n_double * n_double));
  out.x_moment = x_component.moment;
  out.y_moment = y_component.moment;
  const double variance_factor =
    2.0 * (n_double - 4.0) * (n_double - 5.0) /
    n_double / (n_double - 1.0) / (n_double - 2.0) / (n_double - 3.0);
  out.variance =
    variance_factor * out.x_moment * out.y_moment /
    std::pow(n_double, 4.0) * std::pow(n_double, 2.0);
  const double alpha = (out.mean * out.mean) / out.variance;
  const double beta = out.variance / out.mean;
  out.p_value = 1.0 - R::pgamma(out.nV2, alpha, beta, true, false);
  out.statistic = out.nV2;
  out.estimate = std::sqrt(out.nV2 / n_double);
  out.converged_x = x_component.converged;
  out.converged_y = y_component.converged;
  out.nconv_x = x_component.nconv;
  out.nconv_y = y_component.nconv;
  out.iterations_x = x_component.iterations;
  out.iterations_y = y_component.iterations;
  out.info_x = x_component.info;
  out.info_y = y_component.info;
  out.eig_ms = include_component_diagnostics
    ? x_component.eig_ms + y_component.eig_ms
    : 0.0;

  const LegacyDcovSpectraCudaOperatorDiagnostics& dx =
    x_component.cuda_diagnostics;
  const LegacyDcovSpectraCudaOperatorDiagnostics& dy =
    y_component.cuda_diagnostics;
  if (include_component_diagnostics) {
    out.spectra_matvec_count =
      dx.spectra_matvec_count + dy.spectra_matvec_count;
    out.spectra_matvec_ms = dx.spectra_matvec_ms + dy.spectra_matvec_ms;
    out.kernel_launch_count =
      dx.kernel_launch_count + dy.kernel_launch_count;
    out.device_matrix_reuse_count =
      dx.device_matrix_reuse_count + dy.device_matrix_reuse_count;
    out.device_workspace_reuse_count =
      dx.device_workspace_reuse_count + dy.device_workspace_reuse_count;
    out.workspace_realloc_count =
      dx.workspace_realloc_count + dy.workspace_realloc_count;
    out.matrix_bytes = x_component.matrix_bytes + y_component.matrix_bytes;
    out.workspace_bytes = static_cast<double>(
      std::max(dx.workspace_bytes, dy.workspace_bytes));
    out.matrix_h2d_ms = x_component.matrix_h2d_ms + y_component.matrix_h2d_ms;
    out.matrix_h2d_ms_during_compute =
      dx.matrix_h2d_ms_during_compute + dy.matrix_h2d_ms_during_compute;
    out.workspace_alloc_ms = dx.workspace_alloc_ms + dy.workspace_alloc_ms;
    out.h2d_ms = dx.h2d_ms + dy.h2d_ms;
    out.kernel_ms = dx.kernel_ms + dy.kernel_ms;
    out.d2h_ms = dx.d2h_ms + dy.d2h_ms;
    out.total_ms = x_component.total_ms + y_component.total_ms;
  }
  (void)index;
  return out;
}

LegacyDcovCudaLowrankGammaRun legacy_dcov_cuda_lowrank_gamma_compute(
    const double* x,
    const double* y,
    int n,
    int num_col,
    double index,
    int ncv,
    double tol,
    int maxitr) {
  const auto total_start = std::chrono::steady_clock::now();
  const LegacyDcovCudaLowrankComponentRun x_component =
    legacy_dcov_cuda_lowrank_component_compute(
      x, n, num_col, ncv, tol, maxitr);
  const LegacyDcovCudaLowrankComponentRun y_component =
    legacy_dcov_cuda_lowrank_component_compute(
      y, n, num_col, ncv, tol, maxitr);
  LegacyDcovCudaLowrankGammaRun out =
    legacy_dcov_cuda_lowrank_gamma_from_components(
      x_component, y_component, n, index, true);
  out.total_ms = elapsed_ms_since(total_start);
  return out;
}

Rcpp::List legacy_dcov_cuda_lowrank_gamma_run_to_list(
    const LegacyDcovCudaLowrankGammaRun& run,
    int n,
    int num_col,
    double index,
    int ncv,
    double tol,
    int maxitr) {
  return Rcpp::List::create(
    Rcpp::Named("backend") =
      "cuda-dense-sym-matvec-spectra-lowrank-gamma",
    Rcpp::Named("p.value") = run.p_value,
    Rcpp::Named("nV2") = run.nV2,
    Rcpp::Named("mean") = run.mean,
    Rcpp::Named("variance") = run.variance,
    Rcpp::Named("statistic") = run.statistic,
    Rcpp::Named("estimate") = run.estimate,
    Rcpp::Named("estimates") = Rcpp::NumericVector::create(
      Rcpp::Named("nV^2") = run.nV2,
      Rcpp::Named("nV^2 mean") = run.mean,
      Rcpp::Named("nV^2 variance") = run.variance),
    Rcpp::Named("n") = n,
    Rcpp::Named("numCol") = num_col,
    Rcpp::Named("index") = index,
    Rcpp::Named("ncv") = ncv,
    Rcpp::Named("tol") = tol,
    Rcpp::Named("maxitr") = maxitr,
    Rcpp::Named("converged_x") = run.converged_x,
    Rcpp::Named("converged_y") = run.converged_y,
    Rcpp::Named("nconv_x") = run.nconv_x,
    Rcpp::Named("nconv_y") = run.nconv_y,
    Rcpp::Named("iterations_x") = run.iterations_x,
    Rcpp::Named("iterations_y") = run.iterations_y,
    Rcpp::Named("info_x") = run.info_x,
    Rcpp::Named("info_y") = run.info_y,
    Rcpp::Named("x_moment") = run.x_moment,
    Rcpp::Named("y_moment") = run.y_moment,
    Rcpp::Named("eig_ms") = run.eig_ms,
    Rcpp::Named("spectra_matvec_count") = run.spectra_matvec_count,
    Rcpp::Named("spectra_matvec_ms") = run.spectra_matvec_ms,
    Rcpp::Named("kernel_launch_count") = run.kernel_launch_count,
    Rcpp::Named("device_matrix_reuse_count") =
      run.device_matrix_reuse_count,
    Rcpp::Named("device_workspace_reuse_count") =
      run.device_workspace_reuse_count,
    Rcpp::Named("workspace_realloc_count") = run.workspace_realloc_count,
    Rcpp::Named("matrix_bytes") = run.matrix_bytes,
    Rcpp::Named("workspace_bytes") = run.workspace_bytes,
    Rcpp::Named("matrix_h2d_ms") = run.matrix_h2d_ms,
    Rcpp::Named("matrix_h2d_ms_during_compute") =
      run.matrix_h2d_ms_during_compute,
    Rcpp::Named("workspace_alloc_ms") = run.workspace_alloc_ms,
    Rcpp::Named("h2d_ms") = run.h2d_ms,
    Rcpp::Named("kernel_ms") = run.kernel_ms,
    Rcpp::Named("d2h_ms") = run.d2h_ms,
    Rcpp::Named("total_ms") = run.total_ms
  );
}

double legacy_dcov_max_abs_vector_diff(const Eigen::VectorXd& left,
                                       const Eigen::VectorXd& right) {
  if (left.size() != right.size()) {
    return std::numeric_limits<double>::infinity();
  }
  double out = 0.0;
  for (int i = 0; i < left.size(); ++i) {
    out = std::max(out, std::abs(left[i] - right[i]));
  }
  return out;
}

double legacy_dcov_min_centered_abs_corr(const Eigen::MatrixXd& left,
                                         const Eigen::MatrixXd& right) {
  if (left.rows() != right.rows() || left.cols() != right.cols()) {
    return 0.0;
  }
  double out = 1.0;
  for (int col = 0; col < left.cols(); ++col) {
    const double left_norm = left.col(col).norm();
    const double right_norm = right.col(col).norm();
    if (left_norm == 0.0 || right_norm == 0.0) return 0.0;
    const double corr =
      std::abs(left.col(col).dot(right.col(col)) / (left_norm * right_norm));
    out = std::min(out, corr);
  }
  return out;
}

enum class NativeLegacyDcovBatchMode {
  None,
  Level,
  Canonical,
  Round
};

NativeLegacyDcovBatchMode native_legacy_dcov_batch_mode_from_env() {
  const char* raw = std::getenv("FASTKPC_NATIVE_LEGACY_DCOV_BATCH");
  if (raw == nullptr) return NativeLegacyDcovBatchMode::None;
  std::string value(raw);
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (value == "level") return NativeLegacyDcovBatchMode::Level;
  if (value == "canonical") return NativeLegacyDcovBatchMode::Canonical;
  if (value == "round") return NativeLegacyDcovBatchMode::Round;
  return NativeLegacyDcovBatchMode::None;
}

const char* native_legacy_dcov_batch_mode_name(
    NativeLegacyDcovBatchMode mode) {
  switch (mode) {
    case NativeLegacyDcovBatchMode::Level:
      return "level";
    case NativeLegacyDcovBatchMode::Canonical:
      return "canonical";
    case NativeLegacyDcovBatchMode::Round:
      return "round";
    case NativeLegacyDcovBatchMode::None:
    default:
      return "none";
  }
}

bool native_legacy_dcov_cuda_lowrank_requested() {
  const char* raw = std::getenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK");
  if (raw == nullptr) return false;
  std::string value(raw);
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  return value == "cuda_spectra";
}

int native_legacy_dcov_cuda_lowrank_ncv(int n, int num_col) {
  return std::min(n, std::max(2 * num_col + 1, 20));
}

struct NativeLegacyDcovCudaLowrankBackendMetrics {
  bool enabled = false;
  bool component_cache_enabled = false;
  std::string component_cache_scope = "none";
  int component_cache_level_max_entries = 0;
  int count = 0;
  double ms = 0.0;
  int error_count = 0;
  int fallback_count = 0;
  int converged_count = 0;
  int component_cache_lookup_count = 0;
  int component_cache_hit_count = 0;
  int component_cache_miss_count = 0;
  int component_cache_entry_count = 0;
  int component_cache_cross_batch_hit_count = 0;
  int component_cache_eviction_count = 0;
  int component_cache_level_entry_count_max = 0;
  int component_batch_substrate_count = 0;
  int component_batch_substrate_pair_count = 0;
  double component_distance_ms = 0.0;
  double component_lowrank_ms = 0.0;
  double component_moment_ms = 0.0;
  double component_unaccounted_ms = 0.0;
  double component_eig_ms = 0.0;
  double combine_ms = 0.0;
  int spectra_matvec_count = 0;
  double spectra_matvec_ms = 0.0;
  int kernel_launch_count = 0;
  int device_matrix_reuse_count = 0;
  int device_workspace_reuse_count = 0;
  int workspace_realloc_count = 0;
  double matrix_bytes = 0.0;
  double workspace_bytes = 0.0;
  double matrix_h2d_ms = 0.0;
  double matrix_h2d_ms_during_compute = 0.0;
  double matrix_h2d_ms_during_compute_max = 0.0;
  double workspace_alloc_ms = 0.0;
  double h2d_ms = 0.0;
  double kernel_ms = 0.0;
  double d2h_ms = 0.0;
};

double elapsed_ms_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - start).count();
}

long long wall_epoch_ms_now() {
  return static_cast<long long>(
    std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count());
}

int native_progress_pid() {
#ifdef _WIN32
  return static_cast<int>(_getpid());
#else
  return static_cast<int>(getpid());
#endif
}

std::string native_legacy_progress_csv_path_from_env() {
  const char* raw = std::getenv("FASTKPC_NATIVE_LEGACY_PROGRESS_CSV");
  if (raw == nullptr) return std::string();
  return std::string(raw);
}

std::string native_cuda_lowrank_component_cache_progress_csv_path_from_env() {
  const char* raw = std::getenv(
    "FASTKPC_NATIVE_CUDA_LOWRANK_COMPONENT_CACHE_PROGRESS_CSV");
  if (raw == nullptr) return std::string();
  return std::string(raw);
}

int native_legacy_cuda_lowrank_progress_interval() {
  const char* raw = std::getenv("FASTKPC_NATIVE_CUDA_LOWRANK_PROGRESS_INTERVAL");
  if (raw == nullptr) return 256;
  const int value = std::atoi(raw);
  return value > 0 ? value : 256;
}

int native_legacy_cuda_lowrank_batch_threads(int batch_size) {
  const char* raw = std::getenv("FASTKPC_NATIVE_CUDA_LOWRANK_BATCH_THREADS");
  if (raw == nullptr) return 1;
  const int requested = std::atoi(raw);
  if (requested <= 1 || batch_size <= 1) return 1;
  const unsigned int hardware = std::thread::hardware_concurrency();
  const int hardware_limit = hardware > 0
    ? static_cast<int>(hardware)
    : requested;
  return std::max(1, std::min(batch_size, std::min(requested, hardware_limit)));
}

bool native_legacy_cuda_lowrank_component_cache_enabled() {
  const char* raw =
    std::getenv("FASTKPC_NATIVE_CUDA_LOWRANK_COMPONENT_CACHE");
  if (raw == nullptr) return false;
  std::string value(raw);
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  return value == "1" || value == "true" || value == "yes" ||
    value == "on";
}

std::string native_legacy_cuda_lowrank_component_cache_scope() {
  const char* raw =
    std::getenv("FASTKPC_NATIVE_CUDA_LOWRANK_COMPONENT_CACHE_SCOPE");
  if (raw == nullptr) return "batch";
  std::string value(raw);
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  return value == "level" ? "level" : "batch";
}

int native_legacy_cuda_lowrank_component_cache_max_entries() {
  const char* raw =
    std::getenv("FASTKPC_NATIVE_CUDA_LOWRANK_COMPONENT_CACHE_MAX_ENTRIES");
  if (raw == nullptr) return 128;
  const int value = std::atoi(raw);
  return value > 0 ? value : 128;
}

struct NativeLegacyDcovCudaLowrankLevelComponentCacheEntry {
  std::shared_ptr<LegacyDcovCudaLowrankComponentRun> component;
  std::list<std::uintptr_t>::iterator lru_position;
};

class NativeLegacyDcovCudaLowrankLevelComponentCache {
 public:
  explicit NativeLegacyDcovCudaLowrankLevelComponentCache(int max_entries)
      : max_entries_(max_entries > 0 ? max_entries : 128) {}

  std::shared_ptr<LegacyDcovCudaLowrankComponentRun> find(
      std::uintptr_t key) {
    const auto found = entries_.find(key);
    if (found == entries_.end()) return std::shared_ptr<
      LegacyDcovCudaLowrankComponentRun>();
    lru_.erase(found->second.lru_position);
    lru_.push_front(key);
    found->second.lru_position = lru_.begin();
    return found->second.component;
  }

  int insert(
      std::uintptr_t key,
      std::shared_ptr<LegacyDcovCudaLowrankComponentRun> component) {
    int eviction_count = 0;
    const auto found = entries_.find(key);
    if (found != entries_.end()) {
      lru_.erase(found->second.lru_position);
      lru_.push_front(key);
      found->second.lru_position = lru_.begin();
      found->second.component = component;
    } else {
      lru_.push_front(key);
      NativeLegacyDcovCudaLowrankLevelComponentCacheEntry entry;
      entry.component = component;
      entry.lru_position = lru_.begin();
      entries_[key] = entry;
    }
    while (static_cast<int>(entries_.size()) > max_entries_) {
      const std::uintptr_t evict_key = lru_.back();
      lru_.pop_back();
      entries_.erase(evict_key);
      ++eviction_count;
    }
    max_entry_count_ = std::max(
      max_entry_count_,
      static_cast<int>(entries_.size()));
    return eviction_count;
  }

  int max_entry_count() const { return max_entry_count_; }

 private:
  int max_entries_;
  std::list<std::uintptr_t> lru_;
  std::unordered_map<
    std::uintptr_t,
    NativeLegacyDcovCudaLowrankLevelComponentCacheEntry> entries_;
  int max_entry_count_ = 0;
};

enum class LegacyDcovCudaLowrankComponentBatchKeyMode {
  Value,
  Pointer
};

struct LegacyDcovCudaLowrankComponentBatchOptions {
  int n = 0;
  int num_col = 0;
  double index = 1.0;
  int ncv = 0;
  double tol = 1e-10;
  int maxitr = 1000;
  int batch_threads = 1;
  LegacyDcovCudaLowrankComponentBatchKeyMode key_mode =
    LegacyDcovCudaLowrankComponentBatchKeyMode::Value;
  NativeLegacyDcovCudaLowrankLevelComponentCache* level_component_cache =
    nullptr;
};

struct LegacyDcovCudaLowrankComponentBatchRun {
  std::vector<double> p_values;
  std::vector<double> nV2;
  std::vector<double> means;
  std::vector<double> variances;
  std::vector<double> statistics;
  std::vector<double> estimates;
  std::vector<double> x_moments;
  std::vector<double> y_moments;
  int converged_count = 0;
  int component_cache_lookup_count = 0;
  int component_cache_hit_count = 0;
  int component_cache_miss_count = 0;
  int component_cache_entry_count = 0;
  int component_cache_cross_batch_hit_count = 0;
  int component_cache_eviction_count = 0;
  int component_cache_level_entry_count_max = 0;
  int component_count = 0;
  int component_iterations = 0;
  int component_nconv = 0;
  double component_total_ms = 0.0;
  double component_distance_ms = 0.0;
  double component_lowrank_ms = 0.0;
  double component_eig_ms = 0.0;
  double component_moment_ms = 0.0;
  double component_unaccounted_ms = 0.0;
  double combine_ms = 0.0;
  int spectra_matvec_count = 0;
  double spectra_matvec_ms = 0.0;
  int kernel_launch_count = 0;
  int device_matrix_reuse_count = 0;
  int device_workspace_reuse_count = 0;
  int workspace_realloc_count = 0;
  double matrix_bytes = 0.0;
  double workspace_bytes = 0.0;
  double matrix_h2d_ms = 0.0;
  double matrix_h2d_ms_during_compute = 0.0;
  double matrix_h2d_ms_during_compute_max = 0.0;
  double workspace_alloc_ms = 0.0;
  double h2d_ms = 0.0;
  double kernel_ms = 0.0;
  double d2h_ms = 0.0;
  double pair_total_ms = 0.0;
  double total_ms = 0.0;
};

void accumulate_cuda_lowrank_component_diagnostics(
    LegacyDcovCudaLowrankComponentBatchRun* batch,
    const LegacyDcovCudaLowrankComponentRun& component) {
  const LegacyDcovSpectraCudaOperatorDiagnostics& diagnostics =
    component.cuda_diagnostics;
  batch->component_total_ms += component.total_ms;
  batch->component_distance_ms += component.distance_ms;
  batch->component_lowrank_ms += component.lowrank_ms;
  batch->component_eig_ms += component.eig_ms;
  batch->component_moment_ms += component.moment_ms;
  const double accounted_ms =
    component.distance_ms + component.lowrank_ms + component.moment_ms;
  if (component.total_ms > accounted_ms) {
    batch->component_unaccounted_ms += component.total_ms - accounted_ms;
  }
  batch->component_iterations += component.iterations;
  batch->component_nconv += component.nconv;
  batch->spectra_matvec_count += diagnostics.spectra_matvec_count;
  batch->spectra_matvec_ms += diagnostics.spectra_matvec_ms;
  batch->kernel_launch_count += diagnostics.kernel_launch_count;
  batch->device_matrix_reuse_count += diagnostics.device_matrix_reuse_count;
  batch->device_workspace_reuse_count += diagnostics.device_workspace_reuse_count;
  batch->workspace_realloc_count += diagnostics.workspace_realloc_count;
  batch->matrix_bytes += component.matrix_bytes;
  batch->workspace_bytes = std::max(
    batch->workspace_bytes,
    static_cast<double>(diagnostics.workspace_bytes));
  batch->matrix_h2d_ms += component.matrix_h2d_ms;
  batch->matrix_h2d_ms_during_compute +=
    diagnostics.matrix_h2d_ms_during_compute;
  batch->matrix_h2d_ms_during_compute_max = std::max(
    batch->matrix_h2d_ms_during_compute_max,
    diagnostics.matrix_h2d_ms_during_compute);
  batch->workspace_alloc_ms += diagnostics.workspace_alloc_ms;
  batch->h2d_ms += diagnostics.h2d_ms;
  batch->kernel_ms += diagnostics.kernel_ms;
  batch->d2h_ms += diagnostics.d2h_ms;
}

LegacyDcovCudaLowrankComponentBatchRun
legacy_dcov_cuda_lowrank_gamma_component_batch(
    const std::vector<const double*>& x_columns,
    const std::vector<const double*>& y_columns,
    const LegacyDcovCudaLowrankComponentBatchOptions& options) {
  if (x_columns.size() != y_columns.size()) {
    Rcpp::stop("x and y lowrank component batches must have identical size");
  }
  const auto total_start = std::chrono::steady_clock::now();
  const int batch_size = static_cast<int>(x_columns.size());
  LegacyDcovCudaLowrankComponentBatchRun out;
  out.p_values.assign(static_cast<std::size_t>(batch_size), NA_REAL);
  out.nV2.assign(static_cast<std::size_t>(batch_size), NA_REAL);
  out.means.assign(static_cast<std::size_t>(batch_size), NA_REAL);
  out.variances.assign(static_cast<std::size_t>(batch_size), NA_REAL);
  out.statistics.assign(static_cast<std::size_t>(batch_size), NA_REAL);
  out.estimates.assign(static_cast<std::size_t>(batch_size), NA_REAL);
  out.x_moments.assign(static_cast<std::size_t>(batch_size), NA_REAL);
  out.y_moments.assign(static_cast<std::size_t>(batch_size), NA_REAL);
  if (batch_size == 0) return out;

  std::unordered_map<std::string, int> component_by_value;
  std::map<std::uintptr_t, int> component_by_pointer;
  std::vector<std::shared_ptr<LegacyDcovCudaLowrankComponentRun> > components;
  std::vector<unsigned char> component_from_level_cache;
  std::vector<const double*> component_miss_columns;
  std::vector<int> component_miss_indices;
  std::vector<int> x_component_index(static_cast<std::size_t>(batch_size));
  std::vector<int> y_component_index(static_cast<std::size_t>(batch_size));
  components.reserve(static_cast<std::size_t>(2 * batch_size));

  auto intern_component = [&](const double* ptr) {
    out.component_cache_lookup_count += 1;
    if (options.key_mode ==
        LegacyDcovCudaLowrankComponentBatchKeyMode::Pointer) {
      const std::uintptr_t key = reinterpret_cast<std::uintptr_t>(ptr);
      const auto found = component_by_pointer.find(key);
      if (found != component_by_pointer.end()) {
        const int index = found->second;
        if (component_from_level_cache[static_cast<std::size_t>(index)] != 0) {
          out.component_cache_cross_batch_hit_count += 1;
        }
        out.component_cache_hit_count += 1;
        return index;
      }
      const int component_index = static_cast<int>(components.size());
      component_by_pointer[key] = component_index;
      std::shared_ptr<LegacyDcovCudaLowrankComponentRun> cached_component;
      if (options.level_component_cache != nullptr) {
        cached_component = options.level_component_cache->find(key);
      }
      if (cached_component) {
        components.push_back(cached_component);
        component_from_level_cache.push_back(1);
        out.component_cache_cross_batch_hit_count += 1;
        out.component_cache_hit_count += 1;
      } else {
        components.push_back(std::shared_ptr<
          LegacyDcovCudaLowrankComponentRun>());
        component_from_level_cache.push_back(0);
        component_miss_indices.push_back(component_index);
        component_miss_columns.push_back(ptr);
        out.component_cache_miss_count += 1;
        out.component_cache_entry_count += 1;
      }
      return component_index;
    }

    const std::string key(
      reinterpret_cast<const char*>(ptr),
      sizeof(double) * static_cast<std::size_t>(options.n));
    const auto found = component_by_value.find(key);
    if (found != component_by_value.end()) {
      out.component_cache_hit_count += 1;
      return found->second;
    }
    const int component_index = static_cast<int>(components.size());
    component_by_value.emplace(key, component_index);
    components.push_back(std::shared_ptr<LegacyDcovCudaLowrankComponentRun>());
    component_from_level_cache.push_back(0);
    component_miss_indices.push_back(component_index);
    component_miss_columns.push_back(ptr);
    out.component_cache_miss_count += 1;
    out.component_cache_entry_count += 1;
    return component_index;
  };

  for (int i = 0; i < batch_size; ++i) {
    x_component_index[static_cast<std::size_t>(i)] =
      intern_component(x_columns[static_cast<std::size_t>(i)]);
    y_component_index[static_cast<std::size_t>(i)] =
      intern_component(y_columns[static_cast<std::size_t>(i)]);
  }

  const int component_miss_count =
    static_cast<int>(component_miss_columns.size());
  std::vector<LegacyDcovCudaLowrankComponentRun> computed_components(
    static_cast<std::size_t>(component_miss_count));
  const int batch_threads =
    std::max(1, std::min(options.batch_threads, component_miss_count));
  if (batch_threads > 1) {
    std::atomic<int> next_component(0);
    std::mutex error_mutex;
    std::string first_error;
    auto worker = [&]() {
      for (;;) {
        const int component_index = next_component.fetch_add(1);
        if (component_index >= component_miss_count) return;
        {
          std::lock_guard<std::mutex> lock(error_mutex);
          if (!first_error.empty()) return;
        }
        try {
          computed_components[static_cast<std::size_t>(component_index)] =
            legacy_dcov_cuda_lowrank_component_compute(
              component_miss_columns[static_cast<std::size_t>(component_index)],
              options.n,
              options.num_col,
              options.ncv,
              options.tol,
              options.maxitr);
        } catch (const std::exception& error) {
          std::lock_guard<std::mutex> lock(error_mutex);
          if (first_error.empty()) first_error = error.what();
          return;
        } catch (...) {
          std::lock_guard<std::mutex> lock(error_mutex);
          if (first_error.empty()) {
            first_error = "CUDA lowrank cached component failed";
          }
          return;
        }
      }
    };
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(batch_threads));
    for (int thread_id = 0; thread_id < batch_threads; ++thread_id) {
      threads.emplace_back(worker);
    }
    for (std::thread& thread : threads) thread.join();
    if (!first_error.empty()) Rcpp::stop(first_error);
  } else {
    for (int component_index = 0;
         component_index < component_miss_count;
         ++component_index) {
      computed_components[static_cast<std::size_t>(component_index)] =
        legacy_dcov_cuda_lowrank_component_compute(
          component_miss_columns[static_cast<std::size_t>(component_index)],
          options.n,
          options.num_col,
          options.ncv,
          options.tol,
          options.maxitr);
    }
  }

  for (int component_miss_index = 0;
       component_miss_index < component_miss_count;
       ++component_miss_index) {
    const LegacyDcovCudaLowrankComponentRun& component =
      computed_components[static_cast<std::size_t>(component_miss_index)];
    if (!component.converged) {
      Rcpp::stop("CUDA Spectra lowrank did not converge");
    }
    accumulate_cuda_lowrank_component_diagnostics(&out, component);
    std::shared_ptr<LegacyDcovCudaLowrankComponentRun> component_ptr =
      std::make_shared<LegacyDcovCudaLowrankComponentRun>(component);
    const int batch_component_index =
      component_miss_indices[static_cast<std::size_t>(component_miss_index)];
    components[static_cast<std::size_t>(batch_component_index)] =
      component_ptr;
    if (options.key_mode ==
          LegacyDcovCudaLowrankComponentBatchKeyMode::Pointer &&
        options.level_component_cache != nullptr) {
      const std::uintptr_t key = reinterpret_cast<std::uintptr_t>(
        component_miss_columns[static_cast<std::size_t>(component_miss_index)]);
      out.component_cache_eviction_count +=
        options.level_component_cache->insert(key, component_ptr);
      out.component_cache_level_entry_count_max = std::max(
        out.component_cache_level_entry_count_max,
        options.level_component_cache->max_entry_count());
    }
  }

  for (const std::shared_ptr<LegacyDcovCudaLowrankComponentRun>& component :
       components) {
    if (!component || !component->converged) {
      Rcpp::stop("CUDA Spectra lowrank did not converge");
    }
  }

  const std::chrono::steady_clock::time_point combine_start =
    std::chrono::steady_clock::now();
  for (int i = 0; i < batch_size; ++i) {
    const LegacyDcovCudaLowrankGammaRun run =
      legacy_dcov_cuda_lowrank_gamma_from_components(
        *components[static_cast<std::size_t>(
          x_component_index[static_cast<std::size_t>(i)])],
        *components[static_cast<std::size_t>(
          y_component_index[static_cast<std::size_t>(i)])],
        options.n,
        options.index,
        false);
    out.p_values[static_cast<std::size_t>(i)] = run.p_value;
    out.nV2[static_cast<std::size_t>(i)] = run.nV2;
    out.means[static_cast<std::size_t>(i)] = run.mean;
    out.variances[static_cast<std::size_t>(i)] = run.variance;
    out.statistics[static_cast<std::size_t>(i)] = run.statistic;
    out.estimates[static_cast<std::size_t>(i)] = run.estimate;
    out.x_moments[static_cast<std::size_t>(i)] = run.x_moment;
    out.y_moments[static_cast<std::size_t>(i)] = run.y_moment;
    out.converged_count += run.converged_x ? 1 : 0;
    out.converged_count += run.converged_y ? 1 : 0;
  }
  out.combine_ms = elapsed_ms_since(combine_start);
  out.component_count = static_cast<int>(components.size());
  out.pair_total_ms = out.component_total_ms + out.combine_ms;
  out.total_ms = elapsed_ms_since(total_start);
  return out;
}

void append_native_legacy_progress(
    const std::string& path,
    const std::string& event,
    int level,
    int task_count,
    int residual_request_count,
    int tests_replayed,
    int ignored,
    int deletions,
    double provider_call_ms,
    double provider_copy_ms,
    double dcov_materialize_ms,
    double dcov_call_ms,
    double elapsed_ms) {
  if (path.empty()) return;
  std::ifstream probe(path.c_str());
  const bool write_header = !probe.good();
  probe.close();

  std::ofstream out(path.c_str(), std::ios::out | std::ios::app);
  if (!out.good()) return;
  if (write_header) {
    out << "timestamp_ms,pid,event,level,task_count,"
        << "residual_request_count,tests_replayed,ignored,deletions,"
        << "provider_call_ms,provider_copy_ms,dcov_materialize_ms,"
        << "dcov_call_ms,elapsed_ms\n";
  }
  out << wall_epoch_ms_now() << ","
      << native_progress_pid() << ","
      << event << ","
      << level << ","
      << task_count << ","
      << residual_request_count << ","
      << tests_replayed << ","
      << ignored << ","
      << deletions << ","
      << std::setprecision(17) << provider_call_ms << ","
      << provider_copy_ms << ","
      << dcov_materialize_ms << ","
      << dcov_call_ms << ","
      << elapsed_ms << "\n";
}

void append_native_cuda_lowrank_component_cache_progress(
    const std::string& path,
    const std::string& event,
    int level,
    int batch_size,
    const std::string& scope,
    int level_max_entries,
    int component_lookup_count,
    int component_hit_count,
    int component_miss_count,
    int component_entry_count,
    int component_cross_batch_hit_count,
    int component_eviction_count,
    int component_level_entry_count_max,
    int component_count,
    double component_total_ms,
    double component_distance_ms,
    double component_lowrank_ms,
    double component_moment_ms,
    double component_unaccounted_ms,
    double component_eig_ms,
    double combine_ms,
    int spectra_matvec_count,
    double spectra_matvec_ms,
    int kernel_launch_count,
    int device_matrix_reuse_count,
    int device_workspace_reuse_count,
    int workspace_realloc_count,
    double matrix_bytes,
    double workspace_bytes,
    double matrix_h2d_ms,
    double matrix_h2d_ms_during_compute,
    double matrix_h2d_ms_during_compute_max,
    double workspace_alloc_ms,
    double h2d_ms,
    double kernel_ms,
    double d2h_ms,
    double elapsed_ms) {
  if (path.empty()) return;
  std::ifstream probe(path.c_str());
  const bool write_header = !probe.good();
  probe.close();

  std::ofstream out(path.c_str(), std::ios::out | std::ios::app);
  if (!out.good()) return;
  if (write_header) {
    out << "timestamp_ms,pid,event,level,batch_size,"
        << "component_cache_scope,component_cache_level_max_entries,"
        << "component_lookup_count,component_hit_count,"
        << "component_miss_count,component_entry_count,"
        << "component_cross_batch_hit_count,component_eviction_count,"
        << "component_level_entry_count_max,component_count,"
        << "component_total_ms,component_distance_ms,"
        << "component_lowrank_ms,component_moment_ms,"
        << "component_unaccounted_ms,component_eig_ms,combine_ms,"
        << "spectra_matvec_count,spectra_matvec_ms,"
        << "kernel_launch_count,device_matrix_reuse_count,"
        << "device_workspace_reuse_count,workspace_realloc_count,"
        << "matrix_bytes,workspace_bytes,matrix_h2d_ms,"
        << "matrix_h2d_ms_during_compute,"
        << "matrix_h2d_ms_during_compute_max,workspace_alloc_ms,"
        << "h2d_ms,kernel_ms,d2h_ms,"
        << "elapsed_ms\n";
  }
  out << wall_epoch_ms_now() << ","
      << native_progress_pid() << ","
      << event << ","
      << level << ","
      << batch_size << ","
      << scope << ","
      << level_max_entries << ","
      << component_lookup_count << ","
      << component_hit_count << ","
      << component_miss_count << ","
      << component_entry_count << ","
      << component_cross_batch_hit_count << ","
      << component_eviction_count << ","
      << component_level_entry_count_max << ","
      << component_count << ","
      << std::setprecision(17) << component_total_ms << ","
      << component_distance_ms << ","
      << component_lowrank_ms << ","
      << component_moment_ms << ","
      << component_unaccounted_ms << ","
      << component_eig_ms << ","
      << combine_ms << ","
      << spectra_matvec_count << ","
      << spectra_matvec_ms << ","
      << kernel_launch_count << ","
      << device_matrix_reuse_count << ","
      << device_workspace_reuse_count << ","
      << workspace_realloc_count << ","
      << matrix_bytes << ","
      << workspace_bytes << ","
      << matrix_h2d_ms << ","
      << matrix_h2d_ms_during_compute << ","
      << matrix_h2d_ms_during_compute_max << ","
      << workspace_alloc_ms << ","
      << h2d_ms << ","
      << kernel_ms << ","
      << d2h_ms << ","
      << elapsed_ms << "\n";
}

double list_numeric_value(const Rcpp::List& values, const char* name) {
  if (!values.containsElementNamed(name)) return 0.0;
  return Rcpp::as<double>(values[name]);
}

int list_integer_value(const Rcpp::List& values, const char* name) {
  if (!values.containsElementNamed(name)) return 0;
  return Rcpp::as<int>(values[name]);
}

bool list_logical_value(const Rcpp::List& values, const char* name) {
  if (!values.containsElementNamed(name)) return false;
  return Rcpp::as<bool>(values[name]);
}

std::string list_string_value(const Rcpp::List& values,
                              const char* name,
                              const std::string& fallback) {
  if (!values.containsElementNamed(name)) return fallback;
  SEXP value = values[name];
  if (Rf_isNull(value)) return fallback;
  return Rcpp::as<std::string>(value);
}

Rcpp::NumericMatrix residual_provider_response_matrix(
    SEXP response,
    std::string* response_mode,
    std::string* response_backend) {
  if (Rf_isMatrix(response)) {
    *response_mode = "matrix";
    *response_backend = "matrix-provider";
    return Rcpp::NumericMatrix(response);
  }
  if (TYPEOF(response) == VECSXP) {
    Rcpp::List values(response);
    if (!values.containsElementNamed("residuals")) {
      Rcpp::stop("residual provider list response must contain residuals");
    }
    SEXP residuals = values["residuals"];
    if (!Rf_isMatrix(residuals)) {
      Rcpp::stop("residual provider list residuals must be a matrix");
    }
    *response_mode = "list";
    *response_backend = list_string_value(values, "backend", "list-provider");
    return Rcpp::NumericMatrix(residuals);
  }
  Rcpp::stop("residual provider must return a matrix or list");
}

void update_provider_response_label(std::string* aggregate,
                                    const std::string& value) {
  if (aggregate->empty()) {
    *aggregate = value;
  } else if (*aggregate != value) {
    *aggregate = "mixed";
  }
}

Rcpp::LogicalMatrix adjacency_to_matrix(const std::vector<int>& adjacency, int p) {
  Rcpp::LogicalMatrix out(p, p);
  for (int i = 0; i < p; ++i) {
    for (int j = 0; j < p; ++j) {
      out(i, j) = adjacency[static_cast<std::size_t>(i) * p + j] != 0;
    }
  }
  return out;
}

Rcpp::NumericMatrix pmax_to_matrix(const std::vector<double>& pmax, int p) {
  Rcpp::NumericMatrix out(p, p);
  for (int i = 0; i < p; ++i) {
    for (int j = 0; j < p; ++j) {
      out(i, j) = pmax[static_cast<std::size_t>(i) * p + j];
    }
  }
  return out;
}

Rcpp::List sepsets_to_list(const std::vector<std::vector<std::vector<int> > >& sepsets) {
  const int p = static_cast<int>(sepsets.size());
  Rcpp::List out(p);
  for (int i = 0; i < p; ++i) {
    Rcpp::List row(p);
    for (int j = 0; j < p; ++j) {
      Rcpp::IntegerVector value(sepsets[i][j].size());
      for (int k = 0; k < value.size(); ++k) value[k] = sepsets[i][j][k] + 1;
      row[j] = value;
    }
    out[i] = row;
  }
  return out;
}

Rcpp::List level_log_to_list(const std::vector<std::vector<LevelDeletion> >& logs) {
  Rcpp::List out(logs.size());
  for (int level = 0; level < static_cast<int>(logs.size()); ++level) {
    Rcpp::List entries(logs[level].size());
    for (int i = 0; i < static_cast<int>(logs[level].size()); ++i) {
      const LevelDeletion& entry = logs[level][i];
      Rcpp::IntegerVector cond(entry.conditioning_set.size());
      for (int k = 0; k < cond.size(); ++k) cond[k] = entry.conditioning_set[k] + 1;
      entries[i] = Rcpp::List::create(
        Rcpp::Named("x") = entry.x + 1,
        Rcpp::Named("y") = entry.y + 1,
        Rcpp::Named("S") = cond,
        Rcpp::Named("p.value") = entry.p_value
      );
    }
    out[level] = entries;
  }
  return out;
}

Rcpp::List residual_cache_stats_to_list(const SkeletonResult& result) {
  return Rcpp::List::create(
    Rcpp::Named("enabled") = result.residual_cache_enabled,
    Rcpp::Named("requests") = result.residual_cache_requests,
    Rcpp::Named("hits") = result.residual_cache_hits,
    Rcpp::Named("misses") = result.residual_cache_misses,
    Rcpp::Named("computations") = result.residual_cache_computations,
    Rcpp::Named("stored_vectors") = result.residual_cache_stored_vectors,
    Rcpp::Named("stored_values") = result.residual_cache_stored_values,
    Rcpp::Named("backend_name") = result.residual_backend,
    Rcpp::Named("residual_device") = result.residual_device
  );
}

Rcpp::List hsic_batch_result_to_list(const HsicBatchResult& result,
                                     int pair,
                                     double sig) {
  Rcpp::NumericVector replicates;
  if (!result.permutation_replicates.empty()) {
    const int reps = result.diagnostics.permutation_replicates;
    replicates = Rcpp::NumericVector(reps);
    const std::size_t base = static_cast<std::size_t>(pair) * reps;
    for (int i = 0; i < reps; ++i) replicates[i] = result.permutation_replicates[base + i];
  }

  return Rcpp::List::create(
    Rcpp::Named("method") =
      result.diagnostics.permutation_replicates > 0 ? "hsic.perm" : "hsic.gamma",
    Rcpp::Named("backend") = result.diagnostics.backend,
    Rcpp::Named("statistic") = result.statistics[pair],
    Rcpp::Named("estimate") = result.statistics[pair],
    Rcpp::Named("estimates") = Rcpp::NumericVector::create(
      Rcpp::Named("HSIC") = result.statistics[pair],
      Rcpp::Named("HSIC mean") = result.means[pair],
      Rcpp::Named("HSIC variance") = result.variances[pair]
    ),
    Rcpp::Named("p.value") = result.p_values[pair],
    Rcpp::Named("replicates") = replicates,
    Rcpp::Named("diagnostics") = Rcpp::List::create(
      Rcpp::Named("n") = result.diagnostics.n,
      Rcpp::Named("pairs") = result.diagnostics.pairs,
      Rcpp::Named("backend") = result.diagnostics.backend,
      Rcpp::Named("kernel") = "rbf",
      Rcpp::Named("sig") = sig,
      Rcpp::Named("bytes_allocated") =
        static_cast<double>(result.diagnostics.bytes_allocated),
      Rcpp::Named("cuda_blocks") = result.diagnostics.cuda_blocks,
      Rcpp::Named("cuda_threads") = result.diagnostics.cuda_threads,
      Rcpp::Named("replicates") = result.diagnostics.permutation_replicates,
      Rcpp::Named("used_seed") = result.diagnostics.used_seed,
      Rcpp::Named("seed") = result.diagnostics.used_seed ?
        static_cast<int>(result.diagnostics.seed) : NA_INTEGER,
      Rcpp::Named("reason") = result.diagnostics.reason,
      Rcpp::Named("shape") = result.shapes[pair],
      Rcpp::Named("scale") = result.scales[pair]
    )
  );
}

Rcpp::DataFrame scheduler_levels_to_data_frame(
  const std::vector<LayerDiagnosticsLevel>& levels) {
  const int n = static_cast<int>(levels.size());
  Rcpp::IntegerVector level(n), tasks_planned(n), tasks_evaluated(n);
  Rcpp::IntegerVector tests_replayed(n), tasks_ignored_after_delete(n);
  Rcpp::IntegerVector deletions(n), unconditional_tasks(n), conditional_tasks(n);
  Rcpp::IntegerVector unique_residual_requests(n), dcov_batches(n);
  Rcpp::IntegerVector residual_batches(n);
  Rcpp::NumericVector plan_elapsed_sec(n), residual_prefetch_elapsed_sec(n);
  Rcpp::NumericVector ci_eval_elapsed_sec(n), replay_elapsed_sec(n);
  Rcpp::NumericVector total_elapsed_sec(n);
  for (int i = 0; i < n; ++i) {
    level[i] = levels[i].level;
    tasks_planned[i] = levels[i].tasks_planned;
    tasks_evaluated[i] = levels[i].tasks_evaluated;
    tests_replayed[i] = levels[i].tests_replayed;
    tasks_ignored_after_delete[i] = levels[i].tasks_ignored_after_delete;
    deletions[i] = levels[i].deletions;
    unconditional_tasks[i] = levels[i].unconditional_tasks;
    conditional_tasks[i] = levels[i].conditional_tasks;
    unique_residual_requests[i] = levels[i].unique_residual_requests;
    dcov_batches[i] = levels[i].dcov_batches;
    residual_batches[i] = levels[i].residual_batches;
    plan_elapsed_sec[i] = levels[i].plan_elapsed_sec;
    residual_prefetch_elapsed_sec[i] = levels[i].residual_prefetch_elapsed_sec;
    ci_eval_elapsed_sec[i] = levels[i].ci_eval_elapsed_sec;
    replay_elapsed_sec[i] = levels[i].replay_elapsed_sec;
    total_elapsed_sec[i] = levels[i].total_elapsed_sec;
  }
  return Rcpp::DataFrame::create(
    Rcpp::Named("level") = level,
    Rcpp::Named("tasks_planned") = tasks_planned,
    Rcpp::Named("tasks_evaluated") = tasks_evaluated,
    Rcpp::Named("tests_replayed") = tests_replayed,
    Rcpp::Named("tasks_ignored_after_delete") = tasks_ignored_after_delete,
    Rcpp::Named("deletions") = deletions,
    Rcpp::Named("unconditional_tasks") = unconditional_tasks,
    Rcpp::Named("conditional_tasks") = conditional_tasks,
    Rcpp::Named("unique_residual_requests") = unique_residual_requests,
    Rcpp::Named("dcov_batches") = dcov_batches,
    Rcpp::Named("residual_batches") = residual_batches,
    Rcpp::Named("plan_elapsed_sec") = plan_elapsed_sec,
    Rcpp::Named("residual_prefetch_elapsed_sec") =
      residual_prefetch_elapsed_sec,
    Rcpp::Named("ci_eval_elapsed_sec") = ci_eval_elapsed_sec,
    Rcpp::Named("replay_elapsed_sec") = replay_elapsed_sec,
    Rcpp::Named("total_elapsed_sec") = total_elapsed_sec,
    Rcpp::Named("stringsAsFactors") = false
  );
}

Rcpp::DataFrame scheduler_batches_to_data_frame(
  const std::vector<SchedulerBatchDiagnostic>& batches) {
  const int n = static_cast<int>(batches.size());
  Rcpp::IntegerVector level(n), batch_id(n), start_task_id(n), task_count(n), rows(n);
  Rcpp::IntegerVector groups(n), true_batched_groups(n), true_batched_fits(n);
  Rcpp::IntegerVector single_fit_calls(n), cpu_fallback_fits(n);
  Rcpp::IntegerVector unique_designs(n), duplicate_design_fits(n);
  Rcpp::IntegerVector max_fits_per_design(n);
  Rcpp::IntegerVector max_group_size(n), min_group_size(n);
  Rcpp::IntegerVector max_design_cols(n), min_design_cols(n);
  Rcpp::CharacterVector kind(n), status(n);
  for (int i = 0; i < n; ++i) {
    level[i] = batches[i].level;
    batch_id[i] = batches[i].batch_id;
    kind[i] = batches[i].kind;
    start_task_id[i] = batches[i].start_task_id;
    task_count[i] = batches[i].task_count;
    rows[i] = batches[i].n;
    status[i] = batches[i].status;
    groups[i] = batches[i].groups;
    true_batched_groups[i] = batches[i].true_batched_groups;
    true_batched_fits[i] = batches[i].true_batched_fits;
    single_fit_calls[i] = batches[i].single_fit_calls;
    cpu_fallback_fits[i] = batches[i].cpu_fallback_fits;
    unique_designs[i] = batches[i].unique_designs;
    duplicate_design_fits[i] = batches[i].duplicate_design_fits;
    max_fits_per_design[i] = batches[i].max_fits_per_design;
    max_group_size[i] = batches[i].max_group_size;
    min_group_size[i] = batches[i].min_group_size;
    max_design_cols[i] = batches[i].max_design_cols;
    min_design_cols[i] = batches[i].min_design_cols;
  }
  return Rcpp::DataFrame::create(
    Rcpp::Named("level") = level,
    Rcpp::Named("batch_id") = batch_id,
    Rcpp::Named("kind") = kind,
    Rcpp::Named("start_task_id") = start_task_id,
    Rcpp::Named("task_count") = task_count,
    Rcpp::Named("n") = rows,
    Rcpp::Named("status") = status,
    Rcpp::Named("groups") = groups,
    Rcpp::Named("true_batched_groups") = true_batched_groups,
    Rcpp::Named("true_batched_fits") = true_batched_fits,
    Rcpp::Named("single_fit_calls") = single_fit_calls,
    Rcpp::Named("cpu_fallback_fits") = cpu_fallback_fits,
    Rcpp::Named("unique_designs") = unique_designs,
    Rcpp::Named("duplicate_design_fits") = duplicate_design_fits,
    Rcpp::Named("max_fits_per_design") = max_fits_per_design,
    Rcpp::Named("max_group_size") = max_group_size,
    Rcpp::Named("min_group_size") = min_group_size,
    Rcpp::Named("max_design_cols") = max_design_cols,
    Rcpp::Named("min_design_cols") = min_design_cols,
    Rcpp::Named("stringsAsFactors") = false
  );
}

Rcpp::DataFrame scheduler_residuals_to_data_frame(
  const std::vector<SchedulerResidualDiagnostic>& residuals) {
  const int n = static_cast<int>(residuals.size());
  Rcpp::IntegerVector level(n), request_id(n), target(n), conditioning_size(n);
  Rcpp::CharacterVector residual_backend(n), residual_device(n), reason(n);
  Rcpp::LogicalVector materialized(n), fallback_used(n);
  for (int i = 0; i < n; ++i) {
    level[i] = residuals[i].level;
    request_id[i] = residuals[i].request_id;
    target[i] = residuals[i].target + 1;
    conditioning_size[i] = residuals[i].conditioning_size;
    residual_backend[i] = residuals[i].residual_backend;
    residual_device[i] = residuals[i].residual_device;
    materialized[i] = residuals[i].materialized;
    fallback_used[i] = residuals[i].fallback_used;
    reason[i] = residuals[i].reason;
  }
  return Rcpp::DataFrame::create(
    Rcpp::Named("level") = level,
    Rcpp::Named("request_id") = request_id,
    Rcpp::Named("target") = target,
    Rcpp::Named("conditioning_size") = conditioning_size,
    Rcpp::Named("residual_backend") = residual_backend,
    Rcpp::Named("residual_device") = residual_device,
    Rcpp::Named("materialized") = materialized,
    Rcpp::Named("fallback_used") = fallback_used,
    Rcpp::Named("reason") = reason,
    Rcpp::Named("stringsAsFactors") = false
  );
}

Rcpp::List scheduler_diagnostics_to_list(const SchedulerDiagnostics& diagnostics) {
  return Rcpp::List::create(
    Rcpp::Named("summary") = Rcpp::List::create(
      Rcpp::Named("scheduler") = diagnostics.scheduler,
      Rcpp::Named("scheduler_requested") = diagnostics.scheduler_requested,
      Rcpp::Named("levels") = diagnostics.levels,
      Rcpp::Named("tasks_planned") = diagnostics.tasks_planned,
      Rcpp::Named("tasks_evaluated") = diagnostics.tasks_evaluated,
      Rcpp::Named("tests_replayed") = diagnostics.tests_replayed,
      Rcpp::Named("tasks_ignored_after_delete") =
        diagnostics.tasks_ignored_after_delete,
      Rcpp::Named("dcov_batches") = diagnostics.dcov_batches,
      Rcpp::Named("residual_requests") = diagnostics.residual_requests,
      Rcpp::Named("unique_residual_requests") =
        diagnostics.unique_residual_requests,
      Rcpp::Named("residual_batches") = diagnostics.residual_batches,
      Rcpp::Named("cuda_residual_batch_groups") =
        diagnostics.cuda_residual_batch_groups,
      Rcpp::Named("cuda_residual_true_batched_groups") =
        diagnostics.cuda_residual_true_batched_groups,
      Rcpp::Named("cuda_residual_true_batched_fits") =
        diagnostics.cuda_residual_true_batched_fits,
      Rcpp::Named("cuda_residual_single_fit_calls") =
        diagnostics.cuda_residual_single_fit_calls,
      Rcpp::Named("cuda_residual_cpu_fallback_fits") =
        diagnostics.cuda_residual_cpu_fallback_fits,
      Rcpp::Named("cuda_residual_unique_designs") =
        diagnostics.cuda_residual_unique_designs,
      Rcpp::Named("cuda_residual_duplicate_design_fits") =
        diagnostics.cuda_residual_duplicate_design_fits,
      Rcpp::Named("cuda_residual_max_fits_per_design") =
        diagnostics.cuda_residual_max_fits_per_design,
      Rcpp::Named("max_level_tasks") = diagnostics.max_level_tasks,
      Rcpp::Named("max_level_unique_residuals") =
        diagnostics.max_level_unique_residuals,
      Rcpp::Named("dcov_batch_size_requested") =
        diagnostics.dcov_batch_size_requested,
      Rcpp::Named("dcov_batch_size_used") = diagnostics.dcov_batch_size_used,
      Rcpp::Named("residual_batch_size_requested") =
        diagnostics.residual_batch_size_requested,
      Rcpp::Named("residual_batch_size_used") =
        diagnostics.residual_batch_size_used,
      Rcpp::Named("plan_elapsed_sec") = diagnostics.plan_elapsed_sec,
      Rcpp::Named("residual_prefetch_elapsed_sec") =
        diagnostics.residual_prefetch_elapsed_sec,
      Rcpp::Named("residual_request_collect_sec") =
        diagnostics.residual_request_collect_sec,
      Rcpp::Named("residual_prefetch_missing_scan_sec") =
        diagnostics.residual_prefetch_missing_scan_sec,
      Rcpp::Named("residual_prefetch_batch_input_sec") =
        diagnostics.residual_prefetch_batch_input_sec,
      Rcpp::Named("residual_batch_call_wall_sec") =
        diagnostics.residual_batch_call_wall_sec,
      Rcpp::Named("residual_diagnostic_merge_sec") =
        diagnostics.residual_diagnostic_merge_sec,
      Rcpp::Named("residual_prefetch_unaccounted_sec") =
        diagnostics.residual_prefetch_unaccounted_sec,
      Rcpp::Named("residual_grouping_sec") =
        diagnostics.residual_grouping_sec,
      Rcpp::Named("residual_grouping_condition_key_sec") =
        diagnostics.residual_grouping_condition_key_sec,
      Rcpp::Named("residual_grouping_group_key_sec") =
        diagnostics.residual_grouping_group_key_sec,
      Rcpp::Named("residual_grouping_design_build_sec") =
        diagnostics.residual_grouping_design_build_sec,
      Rcpp::Named("residual_grouping_map_insert_sec") =
        diagnostics.residual_grouping_map_insert_sec,
      Rcpp::Named("residual_grouping_design_cache_lookup_sec") =
        diagnostics.residual_grouping_design_cache_lookup_sec,
      Rcpp::Named("residual_grouping_design_cache_insert_sec") =
        diagnostics.residual_grouping_design_cache_insert_sec,
      Rcpp::Named("residual_grouping_group_lookup_sec") =
        diagnostics.residual_grouping_group_lookup_sec,
      Rcpp::Named("residual_grouping_group_insert_sec") =
        diagnostics.residual_grouping_group_insert_sec,
      Rcpp::Named("residual_grouping_group_design_lookup_sec") =
        diagnostics.residual_grouping_group_design_lookup_sec,
      Rcpp::Named("residual_grouping_group_design_copy_sec") =
        diagnostics.residual_grouping_group_design_copy_sec,
      Rcpp::Named("residual_grouping_group_design_index_insert_sec") =
        diagnostics.residual_grouping_group_design_index_insert_sec,
      Rcpp::Named("residual_grouping_request_insert_sec") =
        diagnostics.residual_grouping_request_insert_sec,
      Rcpp::Named("residual_grouping_unaccounted_sec") =
        diagnostics.residual_grouping_unaccounted_sec,
      Rcpp::Named("residual_grouping_group_count") =
        diagnostics.residual_grouping_group_count,
      Rcpp::Named("residual_grouping_design_count") =
        diagnostics.residual_grouping_design_count,
      Rcpp::Named("residual_grouping_condition_key_sort_count") =
        diagnostics.residual_grouping_condition_key_sort_count,
      Rcpp::Named("residual_grouping_string_key_count") =
        diagnostics.residual_grouping_string_key_count,
      Rcpp::Named("residual_structural_group_key_count") =
        diagnostics.residual_structural_group_key_count,
      Rcpp::Named("residual_structural_condition_key_count") =
        diagnostics.residual_structural_condition_key_count,
      Rcpp::Named("residual_string_group_key_count") =
        diagnostics.residual_string_group_key_count,
      Rcpp::Named("residual_string_condition_key_count") =
        diagnostics.residual_string_condition_key_count,
      Rcpp::Named("residual_grouping_group_design_copy_count") =
        diagnostics.residual_grouping_group_design_copy_count,
      Rcpp::Named("residual_grouping_group_design_x_values") =
        diagnostics.residual_grouping_group_design_x_values,
      Rcpp::Named("residual_grouping_group_design_p_values") =
        diagnostics.residual_grouping_group_design_p_values,
      Rcpp::Named("residual_grouping_request_insert_count") =
        diagnostics.residual_grouping_request_insert_count,
      Rcpp::Named("residual_design_cache_hit_count") =
        diagnostics.residual_design_cache_hit_count,
      Rcpp::Named("residual_design_cache_miss_count") =
        diagnostics.residual_design_cache_miss_count,
      Rcpp::Named("residual_design_cache_insert_count") =
        diagnostics.residual_design_cache_insert_count,
      Rcpp::Named("residual_design_cache_entries") =
        diagnostics.residual_design_cache_entries,
      Rcpp::Named("residual_design_build_total_sec") =
        diagnostics.residual_design_build_total_sec,
      Rcpp::Named("residual_design_build_basis_sec") =
        diagnostics.residual_design_build_basis_sec,
      Rcpp::Named("residual_design_build_penalty_sec") =
        diagnostics.residual_design_build_penalty_sec,
      Rcpp::Named("residual_design_build_x_pack_sec") =
        diagnostics.residual_design_build_x_pack_sec,
      Rcpp::Named("residual_design_build_p_pack_sec") =
        diagnostics.residual_design_build_p_pack_sec,
      Rcpp::Named("residual_design_build_alloc_sec") =
        diagnostics.residual_design_build_alloc_sec,
      Rcpp::Named("residual_design_build_column_extract_sec") =
        diagnostics.residual_design_build_column_extract_sec,
      Rcpp::Named("residual_design_build_finite_check_sec") =
        diagnostics.residual_design_build_finite_check_sec,
      Rcpp::Named("residual_design_build_unaccounted_sec") =
        diagnostics.residual_design_build_unaccounted_sec,
      Rcpp::Named("residual_design_build_count") =
        diagnostics.residual_design_build_count,
      Rcpp::Named("residual_design_build_x_values") =
        diagnostics.residual_design_build_x_values,
      Rcpp::Named("residual_design_build_p_values") =
        diagnostics.residual_design_build_p_values,
      Rcpp::Named("residual_design_build_basis_values") =
        diagnostics.residual_design_build_basis_values,
      Rcpp::Named("residual_design_build_penalty_values") =
        diagnostics.residual_design_build_penalty_values,
      Rcpp::Named("residual_design_build_condition_cols") =
        diagnostics.residual_design_build_condition_cols,
      Rcpp::Named("residual_design_build_finite_check_values") =
        diagnostics.residual_design_build_finite_check_values,
      Rcpp::Named("residual_design_build_intercept_count") =
        diagnostics.residual_design_build_intercept_count,
      Rcpp::Named("residual_design_build_one_dimensional_count") =
        diagnostics.residual_design_build_one_dimensional_count,
      Rcpp::Named("residual_design_build_additive_count") =
        diagnostics.residual_design_build_additive_count,
      Rcpp::Named("residual_design_build_tensor_count") =
        diagnostics.residual_design_build_tensor_count,
      Rcpp::Named("residual_design_build_one_dimensional_basis_sec") =
        diagnostics.residual_design_build_one_dimensional_basis_sec,
      Rcpp::Named("residual_design_build_one_dimensional_alloc_sec") =
        diagnostics.residual_design_build_one_dimensional_alloc_sec,
      Rcpp::Named("residual_design_build_one_dimensional_x_pack_sec") =
        diagnostics.residual_design_build_one_dimensional_x_pack_sec,
      Rcpp::Named("residual_design_build_one_dimensional_p_build_sec") =
        diagnostics.residual_design_build_one_dimensional_p_build_sec,
      Rcpp::Named("residual_design_build_one_dimensional_p_pack_sec") =
        diagnostics.residual_design_build_one_dimensional_p_pack_sec,
      Rcpp::Named("residual_design_build_one_dimensional_cols") =
        diagnostics.residual_design_build_one_dimensional_cols,
      Rcpp::Named("residual_design_build_one_dimensional_values") =
        diagnostics.residual_design_build_one_dimensional_values,
      Rcpp::Named("residual_design_build_additive_basis_sec") =
        diagnostics.residual_design_build_additive_basis_sec,
      Rcpp::Named("residual_design_build_additive_alloc_sec") =
        diagnostics.residual_design_build_additive_alloc_sec,
      Rcpp::Named("residual_design_build_additive_x_pack_sec") =
        diagnostics.residual_design_build_additive_x_pack_sec,
      Rcpp::Named("residual_design_build_additive_p_build_sec") =
        diagnostics.residual_design_build_additive_p_build_sec,
      Rcpp::Named("residual_design_build_additive_p_pack_sec") =
        diagnostics.residual_design_build_additive_p_pack_sec,
      Rcpp::Named("residual_design_build_additive_component_count") =
        diagnostics.residual_design_build_additive_component_count,
      Rcpp::Named("residual_design_build_additive_basis_cols") =
        diagnostics.residual_design_build_additive_basis_cols,
      Rcpp::Named("residual_design_build_additive_values") =
        diagnostics.residual_design_build_additive_values,
      Rcpp::Named("residual_design_build_tensor_basis_sec") =
        diagnostics.residual_design_build_tensor_basis_sec,
      Rcpp::Named("residual_design_build_tensor_alloc_sec") =
        diagnostics.residual_design_build_tensor_alloc_sec,
      Rcpp::Named("residual_design_build_tensor_x_pack_sec") =
        diagnostics.residual_design_build_tensor_x_pack_sec,
      Rcpp::Named("residual_design_build_tensor_product_sec") =
        diagnostics.residual_design_build_tensor_product_sec,
      Rcpp::Named("residual_design_build_tensor_p_build_sec") =
        diagnostics.residual_design_build_tensor_p_build_sec,
      Rcpp::Named("residual_design_build_tensor_p_pack_sec") =
        diagnostics.residual_design_build_tensor_p_pack_sec,
      Rcpp::Named("residual_design_build_tensor_cols") =
        diagnostics.residual_design_build_tensor_cols,
      Rcpp::Named("residual_design_build_tensor_values") =
        diagnostics.residual_design_build_tensor_values,
      Rcpp::Named("residual_basis_cache_hit_count") =
        diagnostics.residual_basis_cache_hit_count,
      Rcpp::Named("residual_basis_cache_miss_count") =
        diagnostics.residual_basis_cache_miss_count,
      Rcpp::Named("residual_basis_cache_insert_count") =
        diagnostics.residual_basis_cache_insert_count,
      Rcpp::Named("residual_basis_cache_entries") =
        diagnostics.residual_basis_cache_entries,
      Rcpp::Named("residual_basis_cache_hit_sec") =
        diagnostics.residual_basis_cache_hit_sec,
      Rcpp::Named("residual_basis_cache_miss_build_sec") =
        diagnostics.residual_basis_cache_miss_build_sec,
      Rcpp::Named("residual_basis_build_total_sec") =
        diagnostics.residual_basis_build_total_sec,
      Rcpp::Named("residual_basis_build_alloc_sec") =
        diagnostics.residual_basis_build_alloc_sec,
      Rcpp::Named("residual_basis_build_near_constant_sec") =
        diagnostics.residual_basis_build_near_constant_sec,
      Rcpp::Named("residual_basis_build_knots_sec") =
        diagnostics.residual_basis_build_knots_sec,
      Rcpp::Named("residual_basis_build_knots_copy_sec") =
        diagnostics.residual_basis_build_knots_copy_sec,
      Rcpp::Named("residual_basis_build_knots_sort_sec") =
        diagnostics.residual_basis_build_knots_sort_sec,
      Rcpp::Named("residual_basis_build_knots_center_sec") =
        diagnostics.residual_basis_build_knots_center_sec,
      Rcpp::Named("residual_basis_build_min_gap_sec") =
        diagnostics.residual_basis_build_min_gap_sec,
      Rcpp::Named("residual_basis_build_width_sec") =
        diagnostics.residual_basis_build_width_sec,
      Rcpp::Named("residual_basis_build_eval_sec") =
        diagnostics.residual_basis_build_eval_sec,
      Rcpp::Named("residual_basis_build_eval_fill_sec") =
        diagnostics.residual_basis_build_eval_fill_sec,
      Rcpp::Named("residual_basis_build_normalize_sec") =
        diagnostics.residual_basis_build_normalize_sec,
      Rcpp::Named("residual_basis_build_normalize_scale_sec") =
        diagnostics.residual_basis_build_normalize_scale_sec,
      Rcpp::Named("residual_basis_build_fallback_sec") =
        diagnostics.residual_basis_build_fallback_sec,
      Rcpp::Named("residual_basis_build_return_sec") =
        diagnostics.residual_basis_build_return_sec,
      Rcpp::Named("residual_basis_build_unaccounted_sec") =
        diagnostics.residual_basis_build_unaccounted_sec,
      Rcpp::Named("residual_basis_build_count") =
        diagnostics.residual_basis_build_count,
      Rcpp::Named("residual_basis_build_rows") =
        diagnostics.residual_basis_build_rows,
      Rcpp::Named("residual_basis_build_cols") =
        diagnostics.residual_basis_build_cols,
      Rcpp::Named("residual_basis_build_values") =
        diagnostics.residual_basis_build_values,
      Rcpp::Named("residual_basis_build_near_constant_count") =
        diagnostics.residual_basis_build_near_constant_count,
      Rcpp::Named("residual_basis_build_fallback_row_count") =
        diagnostics.residual_basis_build_fallback_row_count,
      Rcpp::Named("residual_host_pack_sec") =
        diagnostics.residual_host_pack_sec,
      Rcpp::Named("residual_alloc_sec") = diagnostics.residual_alloc_sec,
      Rcpp::Named("residual_h2d_sec") = diagnostics.residual_h2d_sec,
      Rcpp::Named("residual_h2d_design_sec") =
        diagnostics.residual_h2d_design_sec,
      Rcpp::Named("residual_h2d_penalty_sec") =
        diagnostics.residual_h2d_penalty_sec,
      Rcpp::Named("residual_h2d_y_sec") =
        diagnostics.residual_h2d_y_sec,
      Rcpp::Named("residual_h2d_index_sec") =
        diagnostics.residual_h2d_index_sec,
      Rcpp::Named("residual_h2d_lambda_sec") =
        diagnostics.residual_h2d_lambda_sec,
      Rcpp::Named("residual_h2d_active_sec") =
        diagnostics.residual_h2d_active_sec,
      Rcpp::Named("residual_h2d_copy_count") =
        diagnostics.residual_h2d_copy_count,
      Rcpp::Named("residual_h2d_bytes") = diagnostics.residual_h2d_bytes,
      Rcpp::Named("residual_h2d_design_bytes") =
        diagnostics.residual_h2d_design_bytes,
      Rcpp::Named("residual_h2d_y_bytes") =
        diagnostics.residual_h2d_y_bytes,
      Rcpp::Named("residual_h2d_metadata_bytes") =
        diagnostics.residual_h2d_metadata_bytes,
      Rcpp::Named("residual_h2d_metadata_coalesced_count") =
        diagnostics.residual_h2d_metadata_coalesced_count,
      Rcpp::Named("residual_h2d_metadata_coalesced_bytes") =
        diagnostics.residual_h2d_metadata_coalesced_bytes,
      Rcpp::Named("residual_h2d_selected_metadata_copy_count") =
        diagnostics.residual_h2d_selected_metadata_copy_count,
      Rcpp::Named("residual_xtx_xty_sec") =
        diagnostics.residual_xtx_xty_sec,
      Rcpp::Named("residual_pointer_setup_sec") =
        diagnostics.residual_pointer_setup_sec,
      Rcpp::Named("residual_active_copy_sec") =
        diagnostics.residual_active_copy_sec,
      Rcpp::Named("residual_build_system_sec") =
        diagnostics.residual_build_system_sec,
      Rcpp::Named("residual_factor_solve_sec") =
        diagnostics.residual_factor_solve_sec,
      Rcpp::Named("residual_factor_cholesky_sec") =
        diagnostics.residual_factor_cholesky_sec,
      Rcpp::Named("residual_factor_rhs_solve_sec") =
        diagnostics.residual_factor_rhs_solve_sec,
      Rcpp::Named("residual_factor_inverse_solve_sec") =
        diagnostics.residual_factor_inverse_solve_sec,
      Rcpp::Named("residual_summary_sec") =
        diagnostics.residual_summary_sec,
      Rcpp::Named("residual_d2h_sec") = diagnostics.residual_d2h_sec,
      Rcpp::Named("residual_d2h_residuals_sec") =
        diagnostics.residual_d2h_residuals_sec,
      Rcpp::Named("residual_d2h_metadata_sec") =
        diagnostics.residual_d2h_metadata_sec,
      Rcpp::Named("residual_d2h_info_sec") =
        diagnostics.residual_d2h_info_sec,
      Rcpp::Named("residual_d2h_copy_count") =
        diagnostics.residual_d2h_copy_count,
      Rcpp::Named("residual_d2h_bytes") = diagnostics.residual_d2h_bytes,
      Rcpp::Named("residual_d2h_residual_bytes") =
        diagnostics.residual_d2h_residual_bytes,
      Rcpp::Named("residual_d2h_metadata_bytes") =
        diagnostics.residual_d2h_metadata_bytes,
      Rcpp::Named("residual_d2h_metadata_coalesced_count") =
        diagnostics.residual_d2h_metadata_coalesced_count,
      Rcpp::Named("residual_d2h_metadata_coalesced_bytes") =
        diagnostics.residual_d2h_metadata_coalesced_bytes,
      Rcpp::Named("residual_host_select_sec") =
        diagnostics.residual_host_select_sec,
      Rcpp::Named("residual_free_sec") = diagnostics.residual_free_sec,
      Rcpp::Named("residual_true_batch_total_sec") =
        diagnostics.residual_true_batch_total_sec,
      Rcpp::Named("residual_factorization_count") =
        diagnostics.residual_factorization_count,
      Rcpp::Named("residual_rhs_solve_count") =
        diagnostics.residual_rhs_solve_count,
      Rcpp::Named("residual_inverse_solve_count") =
        diagnostics.residual_inverse_solve_count,
      Rcpp::Named("residual_rhs_solve_api_calls") =
        diagnostics.residual_rhs_solve_api_calls,
      Rcpp::Named("residual_rhs_target_solves") =
        diagnostics.residual_rhs_target_solves,
      Rcpp::Named("residual_rhs_custom_solve_count") =
        diagnostics.residual_rhs_custom_solve_count,
      Rcpp::Named("residual_rhs_cublas_solve_count") =
        diagnostics.residual_rhs_cublas_solve_count,
      Rcpp::Named("residual_rhs_solve_fallback_count") =
        diagnostics.residual_rhs_solve_fallback_count,
      Rcpp::Named("residual_rhs_custom_solve_sec") =
        diagnostics.residual_rhs_custom_solve_sec,
      Rcpp::Named("residual_rhs_cublas_solve_sec") =
        diagnostics.residual_rhs_cublas_solve_sec,
      Rcpp::Named("residual_candidate_rhs_fused_solve_count") =
        diagnostics.residual_candidate_rhs_fused_solve_count,
      Rcpp::Named("residual_candidate_rhs_materialized_solve_count") =
        diagnostics.residual_candidate_rhs_materialized_solve_count,
      Rcpp::Named("residual_selected_rhs_materialized_solve_count") =
        diagnostics.residual_selected_rhs_materialized_solve_count,
      Rcpp::Named("residual_candidate_beta_values_avoided") =
        diagnostics.residual_candidate_beta_values_avoided,
      Rcpp::Named("residual_summary_candidate_launch_count") =
        diagnostics.residual_summary_candidate_launch_count,
      Rcpp::Named("residual_summary_group_batched_launch_count") =
        diagnostics.residual_summary_group_batched_launch_count,
      Rcpp::Named("residual_summary_group_batched_candidate_count") =
        diagnostics.residual_summary_group_batched_candidate_count,
      Rcpp::Named("residual_edf_trace_shadow_sec") =
        diagnostics.residual_edf_trace_shadow_sec,
      Rcpp::Named("residual_edf_trace_shadow_count") =
        diagnostics.residual_edf_trace_shadow_count,
      Rcpp::Named("residual_edf_trace_mode_full_inverse_count") =
        diagnostics.residual_edf_trace_mode_full_inverse_count,
      Rcpp::Named("residual_edf_trace_mode_shadow_count") =
        diagnostics.residual_edf_trace_mode_shadow_count,
      Rcpp::Named("residual_edf_trace_winner_flip_count") =
        diagnostics.residual_edf_trace_winner_flip_count,
      Rcpp::Named("residual_edf_trace_max_abs_diff") =
        diagnostics.residual_edf_trace_max_abs_diff,
      Rcpp::Named("residual_edf_trace_max_rel_diff") =
        diagnostics.residual_edf_trace_max_rel_diff,
      Rcpp::Named("residual_edf_trace_cuda_sec") =
        diagnostics.residual_edf_trace_cuda_sec,
      Rcpp::Named("residual_edf_trace_cuda_count") =
        diagnostics.residual_edf_trace_cuda_count,
      Rcpp::Named("residual_edf_trace_cuda_candidate_count") =
        diagnostics.residual_edf_trace_cuda_candidate_count,
      Rcpp::Named("residual_edf_trace_full_inverse_skipped_count") =
        diagnostics.residual_edf_trace_full_inverse_skipped_count,
      Rcpp::Named("residual_edf_trace_cuda_fallback_count") =
        diagnostics.residual_edf_trace_cuda_fallback_count,
      Rcpp::Named("residual_edf_trace_cuda_values") =
        diagnostics.residual_edf_trace_cuda_values,
      Rcpp::Named("residual_edf_trace_cuda_kernel_launch_count") =
        diagnostics.residual_edf_trace_cuda_kernel_launch_count,
      Rcpp::Named("residual_edf_trace_cuda_system_count") =
        diagnostics.residual_edf_trace_cuda_system_count,
      Rcpp::Named("residual_edf_trace_cuda_trace_terms") =
        diagnostics.residual_edf_trace_cuda_trace_terms,
      Rcpp::Named("residual_edf_trace_cuda_p_max") =
        diagnostics.residual_edf_trace_cuda_p_max,
      Rcpp::Named("residual_edf_trace_cuda_p_weighted_sum") =
        diagnostics.residual_edf_trace_cuda_p_weighted_sum,
      Rcpp::Named("residual_candidate_inverse_values_avoided") =
        diagnostics.residual_candidate_inverse_values_avoided,
      Rcpp::Named("residual_winning_factor_reuse_count") =
        diagnostics.residual_winning_factor_reuse_count,
      Rcpp::Named("residual_factor_cache_hits") =
        diagnostics.residual_factor_cache_hits,
      Rcpp::Named("residual_factor_cache_misses") =
        diagnostics.residual_factor_cache_misses,
      Rcpp::Named("residual_factor_cache_entries") =
        diagnostics.residual_factor_cache_entries,
      Rcpp::Named("residual_factor_cache_bytes") =
        diagnostics.residual_factor_cache_bytes,
      Rcpp::Named("residual_lambda_candidates") =
        diagnostics.residual_lambda_candidates,
      Rcpp::Named("residual_workspace_reuse_count") =
        diagnostics.residual_workspace_reuse_count,
      Rcpp::Named("residual_workspace_grow_count") =
        diagnostics.residual_workspace_grow_count,
      Rcpp::Named("residual_workspace_slab_grow_count") =
        diagnostics.residual_workspace_slab_grow_count,
      Rcpp::Named("residual_workspace_slab_reuse_count") =
        diagnostics.residual_workspace_slab_reuse_count,
      Rcpp::Named("residual_workspace_slab_bytes") =
        diagnostics.residual_workspace_slab_bytes,
      Rcpp::Named("residual_workspace_legacy_alloc_count") =
        diagnostics.residual_workspace_legacy_alloc_count,
      Rcpp::Named("residual_solver_handle_create_count") =
        diagnostics.residual_solver_handle_create_count,
      Rcpp::Named("residual_per_request_design_x_values") =
        diagnostics.residual_per_request_design_x_values,
      Rcpp::Named("residual_duplicate_design_x_values_avoided") =
        diagnostics.residual_duplicate_design_x_values_avoided,
      Rcpp::Named("residual_cache_insert_sec") =
        diagnostics.residual_cache_insert_sec,
      Rcpp::Named("residual_cache_move_insert_count") =
        diagnostics.residual_cache_move_insert_count,
      Rcpp::Named("residual_cache_copy_insert_count") =
        diagnostics.residual_cache_copy_insert_count,
      Rcpp::Named("residual_algebraic_rss_count") =
        diagnostics.residual_algebraic_rss_count,
      Rcpp::Named("residual_candidate_residual_materialize_count") =
        diagnostics.residual_candidate_residual_materialize_count,
      Rcpp::Named("residual_winning_residual_materialize_count") =
        diagnostics.residual_winning_residual_materialize_count,
      Rcpp::Named("residual_algebraic_rss_clamp_count") =
        diagnostics.residual_algebraic_rss_clamp_count,
      Rcpp::Named("residual_only_batch_count") =
        diagnostics.residual_only_batch_count,
      Rcpp::Named("residual_full_fit_batch_count") =
        diagnostics.residual_full_fit_batch_count,
      Rcpp::Named("residual_only_fit_count") =
        diagnostics.residual_only_fit_count,
      Rcpp::Named("residual_full_fit_materialize_count") =
        diagnostics.residual_full_fit_materialize_count,
      Rcpp::Named("residual_fitted_values_avoided") =
        diagnostics.residual_fitted_values_avoided,
      Rcpp::Named("residual_result_materialize_sec") =
        diagnostics.residual_result_materialize_sec,
      Rcpp::Named("residual_fitted_materialize_sec") =
        diagnostics.residual_fitted_materialize_sec,
      Rcpp::Named("residual_batch_top_level_wall_sec") =
        diagnostics.residual_batch_top_level_wall_sec,
      Rcpp::Named("residual_batch_top_level_unaccounted_sec") =
        diagnostics.residual_batch_top_level_unaccounted_sec,
      Rcpp::Named("ci_eval_elapsed_sec") = diagnostics.ci_eval_elapsed_sec,
      Rcpp::Named("ci_host_pack_sec") = diagnostics.ci_host_pack_sec,
      Rcpp::Named("ci_dcov_call_wall_sec") =
        diagnostics.ci_dcov_call_wall_sec,
      Rcpp::Named("ci_pvalue_copy_sec") = diagnostics.ci_pvalue_copy_sec,
      Rcpp::Named("ci_diagnostic_append_sec") =
        diagnostics.ci_diagnostic_append_sec,
      Rcpp::Named("ci_eval_unaccounted_sec") =
        diagnostics.ci_eval_unaccounted_sec,
      Rcpp::Named("replay_elapsed_sec") = diagnostics.replay_elapsed_sec,
      Rcpp::Named("total_elapsed_sec") = diagnostics.total_elapsed_sec,
      Rcpp::Named("dcov_alloc_sec") = diagnostics.dcov_alloc_sec,
      Rcpp::Named("dcov_h2d_sec") = diagnostics.dcov_h2d_sec,
      Rcpp::Named("dcov_memset_sec") = diagnostics.dcov_memset_sec,
      Rcpp::Named("dcov_rowsum_sec") = diagnostics.dcov_rowsum_sec,
      Rcpp::Named("dcov_totals_d2h_sec") = diagnostics.dcov_totals_d2h_sec,
      Rcpp::Named("dcov_reduce_sec") = diagnostics.dcov_reduce_sec,
      Rcpp::Named("dcov_scalars_d2h_sec") = diagnostics.dcov_scalars_d2h_sec,
      Rcpp::Named("dcov_host_scalar_sec") = diagnostics.dcov_host_scalar_sec,
      Rcpp::Named("dcov_result_materialize_sec") =
        diagnostics.dcov_result_materialize_sec,
      Rcpp::Named("dcov_free_sec") = diagnostics.dcov_free_sec,
      Rcpp::Named("dcov_total_sec") = diagnostics.dcov_total_sec,
      Rcpp::Named("dcov_top_level_wall_sec") =
        diagnostics.dcov_top_level_wall_sec,
      Rcpp::Named("dcov_grid_limit_query_sec") =
        diagnostics.dcov_grid_limit_query_sec,
      Rcpp::Named("dcov_chunk_dispatch_sec") =
        diagnostics.dcov_chunk_dispatch_sec,
      Rcpp::Named("dcov_top_level_unaccounted_sec") =
        diagnostics.dcov_top_level_unaccounted_sec,
      Rcpp::Named("dcov_chunks") = diagnostics.dcov_chunks,
      Rcpp::Named("dcov_max_chunk_batch") = diagnostics.dcov_max_chunk_batch,
      Rcpp::Named("dcov_workspace_reuse_count") =
        diagnostics.dcov_workspace_reuse_count,
      Rcpp::Named("dcov_workspace_grow_count") =
        diagnostics.dcov_workspace_grow_count,
      Rcpp::Named("dcov_raw_aggregate_fused_count") =
        diagnostics.dcov_raw_aggregate_fused_count,
      Rcpp::Named("dcov_rowsum_kernel_launch_count") =
        diagnostics.dcov_rowsum_kernel_launch_count,
      Rcpp::Named("dcov_rowsum_chunk_count") =
        diagnostics.dcov_rowsum_chunk_count,
      Rcpp::Named("dcov_rowsum_total_blocks") =
        diagnostics.dcov_rowsum_total_blocks,
      Rcpp::Named("dcov_rowsum_pair_count") =
        diagnostics.dcov_rowsum_pair_count,
      Rcpp::Named("dcov_rowsum_abs_fast_count") =
        diagnostics.dcov_rowsum_abs_fast_count,
      Rcpp::Named("dcov_rowsum_pow_generic_count") =
        diagnostics.dcov_rowsum_pow_generic_count,
      Rcpp::Named("dcov_rowsum_abs_pair_count") =
        diagnostics.dcov_rowsum_abs_pair_count,
      Rcpp::Named("dcov_rowsum_generic_pair_count") =
        diagnostics.dcov_rowsum_generic_pair_count,
      Rcpp::Named("dcov_rowsum_threads") =
        diagnostics.dcov_rowsum_threads,
      Rcpp::Named("dcov_rowsum_n_max") =
        diagnostics.dcov_rowsum_n_max,
      Rcpp::Named("dcov_rowsum_batch_total") =
        diagnostics.dcov_rowsum_batch_total,
      Rcpp::Named("dcov_rowsum_max_chunk_batch") =
        diagnostics.dcov_rowsum_max_chunk_batch,
      Rcpp::Named("dcov_rowsum_max_chunk_sec") =
        diagnostics.dcov_rowsum_max_chunk_sec,
      Rcpp::Named("dcov_rowsum_max_chunk_n") =
        diagnostics.dcov_rowsum_max_chunk_n,
      Rcpp::Named("dcov_row_product_reduce_count") =
        diagnostics.dcov_row_product_reduce_count,
      Rcpp::Named("dcov_pvalue_only_count") =
        diagnostics.dcov_pvalue_only_count,
      Rcpp::Named("dcov_full_result_materialize_count") =
        diagnostics.dcov_full_result_materialize_count,
      Rcpp::Named("dcov_grid_limit_query_count") =
        diagnostics.dcov_grid_limit_query_count,
      Rcpp::Named("dcov_grid_limit_cache_hit_count") =
        diagnostics.dcov_grid_limit_cache_hit_count,
      Rcpp::Named("dcov_grid_limit_process_cache_hit_count") =
        diagnostics.dcov_grid_limit_process_cache_hit_count
    ),
    Rcpp::Named("levels") =
      scheduler_levels_to_data_frame(diagnostics.per_level),
    Rcpp::Named("batches") =
      scheduler_batches_to_data_frame(diagnostics.batches),
    Rcpp::Named("residuals") =
      scheduler_residuals_to_data_frame(diagnostics.residuals)
  );
}

Rcpp::List skeleton_result_to_list(const SkeletonResult& result, int p) {
  return Rcpp::List::create(
    Rcpp::Named("adjacency") = adjacency_to_matrix(result.adjacency, p),
    Rcpp::Named("sepsets") = sepsets_to_list(result.sepsets),
    Rcpp::Named("pMax") = pmax_to_matrix(result.pmax, p),
    Rcpp::Named("n.edgetests") = Rcpp::IntegerVector(result.n_edge_tests.begin(),
                                                     result.n_edge_tests.end()),
    Rcpp::Named("per.level.log") = level_log_to_list(result.per_level_log),
    Rcpp::Named("backend") = "cuda",
    Rcpp::Named("residual_backend") = result.residual_backend,
    Rcpp::Named("residual_backend_params") = result.residual_backend_params,
    Rcpp::Named("residual_device") = result.residual_device,
    Rcpp::Named("residual_device_requested") = result.residual_device_requested,
    Rcpp::Named("residual_device_reason") = result.residual_device_reason,
    Rcpp::Named("scheduler") = result.scheduler,
    Rcpp::Named("scheduler_requested") = result.scheduler_requested,
    Rcpp::Named("scheduler_diagnostics") =
      scheduler_diagnostics_to_list(result.scheduler_diagnostics),
    Rcpp::Named("residual_cache") = residual_cache_stats_to_list(result),
    Rcpp::Named("ci_method") = result.ci_method.empty() ? "dcc.gamma" : result.ci_method,
    Rcpp::Named("ci_backend") = result.ci_backend.empty() ? "cuda" : result.ci_backend,
    Rcpp::Named("ci_backend_reason") = result.ci_backend_reason,
    Rcpp::Named("ci_diagnostics") = Rcpp::List::create(
      Rcpp::Named("ci_dcc_gamma_tests") = result.ci_dcc_gamma_tests,
      Rcpp::Named("ci_hsic_gamma_tests") = result.ci_hsic_gamma_tests,
      Rcpp::Named("ci_hsic_perm_tests") = result.ci_hsic_perm_tests,
      Rcpp::Named("ci_hsic_permutation_replicates") =
        result.ci_hsic_permutation_replicates,
      Rcpp::Named("ci_hsic_gamma_cuda_tests") =
        result.ci_hsic_gamma_cuda_tests,
      Rcpp::Named("ci_hsic_perm_cuda_tests") =
        result.ci_hsic_perm_cuda_tests,
      Rcpp::Named("ci_hsic_cuda_batches") = result.ci_hsic_cuda_batches,
      Rcpp::Named("ci_hsic_cuda_pairs") = result.ci_hsic_cuda_pairs,
      Rcpp::Named("ci_hsic_cuda_fallback_tests") =
        result.ci_hsic_cuda_fallback_tests,
      Rcpp::Named("ci_hsic_cuda_memory_bytes") =
        static_cast<double>(result.ci_hsic_cuda_memory_bytes),
      Rcpp::Named("ci_hsic_cuda_max_n") = result.ci_hsic_cuda_max_n,
      Rcpp::Named("ci_hsic_cuda_max_batch_pairs") =
        result.ci_hsic_cuda_max_batch_pairs
    )
  );
}

std::string replay_s_key(const std::vector<int>& conditioning_set) {
  std::ostringstream out;
  for (std::size_t i = 0; i < conditioning_set.size(); ++i) {
    if (i != 0) out << "|";
    out << conditioning_set[i] + 1;
  }
  return out.str();
}

double native_ptable_p_for_task(const LayerCiTask& task, double alpha) {
  if (task.conditioning_set.empty()) return alpha / 5.0;
  if (task.edge_x == 0 && task.edge_y == 1 &&
      task.conditioning_set.size() == 1 && task.conditioning_set[0] == 2) {
    return alpha * 1.4;
  }
  if (task.edge_x == 0 && task.edge_y == 2 &&
      task.conditioning_set.size() == 1 && task.conditioning_set[0] == 1) {
    return alpha * 1.2;
  }
  if (task.edge_x == 1 && task.edge_y == 3 &&
      task.conditioning_set.size() == 1 &&
      (task.conditioning_set[0] == 0 || task.conditioning_set[0] == 2)) {
    return alpha * 1.1;
  }
  if (task.edge_x == 4 && task.edge_y == 5 &&
      task.conditioning_set.size() == 2) {
    std::vector<int> cond = task.conditioning_set;
    std::sort(cond.begin(), cond.end());
    if (cond[0] == 0 && cond[1] == 1) return alpha * 1.3;
  }
  return alpha / (3.0 + static_cast<double>(task.task_id + 1));
}

std::vector<double> numeric_matrix_column(Rcpp::NumericMatrix data, int col) {
  std::vector<double> out(data.nrow());
  for (int row = 0; row < data.nrow(); ++row) out[row] = data(row, col);
  return out;
}

std::string native_residual_key(int target,
                                const std::vector<int>& conditioning_set) {
  return std::to_string(target + 1) + "|" + replay_s_key(conditioning_set);
}

Rcpp::IntegerMatrix pdag_to_matrix(const std::vector<int>& pdag, int p) {
  Rcpp::IntegerMatrix out(p, p);
  for (int row = 0; row < p; ++row) {
    for (int col = 0; col < p; ++col) {
      out(row, col) = pdag[static_cast<std::size_t>(row) * p + col];
    }
  }
  return out;
}

Rcpp::List orientation_events_to_list(const std::vector<OrientationEvent>& events) {
  Rcpp::List out(events.size());
  for (int i = 0; i < static_cast<int>(events.size()); ++i) {
    const OrientationEvent& event = events[i];
    Rcpp::IntegerVector S(event.S.size());
    for (int j = 0; j < S.size(); ++j) S[j] = event.S[j] + 1;
    out[i] = Rcpp::List::create(
      Rcpp::Named("phase") = event.phase,
      Rcpp::Named("rule") = event.rule,
      Rcpp::Named("x") = event.x < 0 ? NA_INTEGER : event.x + 1,
      Rcpp::Named("y") = event.y < 0 ? NA_INTEGER : event.y + 1,
      Rcpp::Named("z") = event.z < 0 ? NA_INTEGER : event.z + 1,
      Rcpp::Named("S") = S,
      Rcpp::Named("p.value") = event.p_value,
      Rcpp::Named("accepted") = event.accepted,
      Rcpp::Named("message") = event.message
    );
  }
  return out;
}

Rcpp::List orientation_result_to_list(const OrientationResult& result) {
  return Rcpp::List::create(
    Rcpp::Named("pdag") = pdag_to_matrix(result.pdag, result.p),
    Rcpp::Named("events") = orientation_events_to_list(result.events),
    Rcpp::Named("counts") = Rcpp::List::create(
      Rcpp::Named("collider") = result.collider_orientations,
      Rcpp::Named("rule1") = result.rule1_orientations,
      Rcpp::Named("rule2") = result.rule2_orientations,
      Rcpp::Named("rule3") = result.rule3_orientations,
      Rcpp::Named("generalized") = result.generalized_orientations,
      Rcpp::Named("regrvonps_calls") = result.regrvonps_calls
    ),
    Rcpp::Named("residual_backend") = result.residual_backend,
    Rcpp::Named("residual_backend_params") = result.residual_backend_params,
    Rcpp::Named("residual_device") = result.residual_device,
    Rcpp::Named("residual_device_requested") = result.residual_device_requested,
    Rcpp::Named("residual_device_reason") = result.residual_device_reason,
    Rcpp::Named("orientation_batch_size_requested") =
      result.orientation_batch_size_requested,
    Rcpp::Named("orientation_batch_size_used") =
      result.orientation_batch_size_used,
    Rcpp::Named("residual_cache") = Rcpp::List::create(
      Rcpp::Named("requests") = result.residual_cache_requests,
      Rcpp::Named("hits") = result.residual_cache_hits,
      Rcpp::Named("computations") = result.residual_cache_computations
    ),
    Rcpp::Named("diagnostics") = Rcpp::List::create(
      Rcpp::Named("orientation_residual_device") = result.residual_device,
      Rcpp::Named("orientation_residual_device_requested") =
        result.residual_device_requested,
      Rcpp::Named("orientation_residual_device_reason") =
        result.residual_device_reason,
      Rcpp::Named("orientation_batch_size_requested") =
        result.orientation_batch_size_requested,
      Rcpp::Named("orientation_batch_size_used") =
        result.orientation_batch_size_used,
      Rcpp::Named("regrvonps_calls") = result.regrvonps_calls,
      Rcpp::Named("regrvonps_cuda_calls") = result.regrvonps_cuda_calls,
      Rcpp::Named("regrvonps_cpu_calls") = result.regrvonps_cpu_calls,
      Rcpp::Named("orientation_dcov_batches") =
        result.orientation_dcov_batches,
      Rcpp::Named("orientation_dcov_pairs") = result.orientation_dcov_pairs,
      Rcpp::Named("regrvonps_dcc_gamma_tests") =
        result.regrvonps_dcc_gamma_tests,
      Rcpp::Named("regrvonps_hsic_gamma_tests") =
        result.regrvonps_hsic_gamma_tests,
      Rcpp::Named("regrvonps_hsic_perm_tests") =
        result.regrvonps_hsic_perm_tests,
      Rcpp::Named("regrvonps_hsic_permutation_replicates") =
        result.regrvonps_hsic_permutation_replicates,
      Rcpp::Named("regrvonps_hsic_gamma_cuda_tests") =
        result.regrvonps_hsic_gamma_cuda_tests,
      Rcpp::Named("regrvonps_hsic_perm_cuda_tests") =
        result.regrvonps_hsic_perm_cuda_tests,
      Rcpp::Named("regrvonps_hsic_cuda_batches") =
        result.regrvonps_hsic_cuda_batches,
      Rcpp::Named("regrvonps_hsic_cuda_pairs") =
        result.regrvonps_hsic_cuda_pairs,
      Rcpp::Named("regrvonps_hsic_cuda_fallback_tests") =
        result.regrvonps_hsic_cuda_fallback_tests,
      Rcpp::Named("orientation_residual_fits") =
        result.orientation_residual_fits,
      Rcpp::Named("orientation_cuda_residual_fits") =
        result.orientation_cuda_residual_fits,
      Rcpp::Named("orientation_cpu_fallback_fits") =
        result.orientation_cpu_fallback_fits,
      Rcpp::Named("orientation_cache_requests") =
        result.residual_cache_requests,
      Rcpp::Named("orientation_cache_hits") = result.residual_cache_hits,
      Rcpp::Named("orientation_cache_computations") =
        result.residual_cache_computations
    ),
    Rcpp::Named("ci_method") = result.ci_method.empty() ? "dcc.gamma" : result.ci_method,
    Rcpp::Named("ci_backend") =
      result.ci_backend.empty() ? "native-cpu" : result.ci_backend,
    Rcpp::Named("ci_backend_reason") = result.ci_backend_reason,
    Rcpp::Named("ci_diagnostics") = Rcpp::List::create(
      Rcpp::Named("regrvonps_dcc_gamma_tests") =
        result.regrvonps_dcc_gamma_tests,
      Rcpp::Named("regrvonps_hsic_gamma_tests") =
        result.regrvonps_hsic_gamma_tests,
      Rcpp::Named("regrvonps_hsic_perm_tests") =
        result.regrvonps_hsic_perm_tests,
      Rcpp::Named("regrvonps_hsic_permutation_replicates") =
        result.regrvonps_hsic_permutation_replicates,
      Rcpp::Named("regrvonps_hsic_gamma_cuda_tests") =
        result.regrvonps_hsic_gamma_cuda_tests,
      Rcpp::Named("regrvonps_hsic_perm_cuda_tests") =
        result.regrvonps_hsic_perm_cuda_tests,
      Rcpp::Named("regrvonps_hsic_cuda_batches") =
        result.regrvonps_hsic_cuda_batches,
      Rcpp::Named("regrvonps_hsic_cuda_pairs") =
        result.regrvonps_hsic_cuda_pairs,
      Rcpp::Named("regrvonps_hsic_cuda_fallback_tests") =
        result.regrvonps_hsic_cuda_fallback_tests
    )
  );
}

Rcpp::List fastspline_cuda_fit_to_list(const FastSplineCudaFit& result) {
  const FastSplineFit& fit = result.fit;
  return Rcpp::List::create(
    Rcpp::Named("residuals") =
      Rcpp::NumericVector(fit.residuals.begin(), fit.residuals.end()),
    Rcpp::Named("fitted") =
      Rcpp::NumericVector(fit.fitted.begin(), fit.fitted.end()),
    Rcpp::Named("selected_lambda") = fit.selected_lambda,
    Rcpp::Named("gcv") = fit.gcv,
    Rcpp::Named("rss") = fit.rss,
    Rcpp::Named("edf") = fit.edf,
    Rcpp::Named("design_cols") = fit.design_cols,
    Rcpp::Named("ridge_attempts") = fit.ridge_attempts,
    Rcpp::Named("backend") = "cuda",
    Rcpp::Named("residual_backend") = "fastSpline",
    Rcpp::Named("residual_device") =
      result.diagnostics.fallback_used ? "cuda-fallback-cpu" : "cuda",
    Rcpp::Named("fallback_used") = result.diagnostics.fallback_used,
    Rcpp::Named("diagnostics") = Rcpp::List::create(
      Rcpp::Named("cuda_used") = result.diagnostics.cuda_used,
      Rcpp::Named("fallback_used") = result.diagnostics.fallback_used,
      Rcpp::Named("reason") = result.diagnostics.reason,
      Rcpp::Named("batch_group_id") = result.diagnostics.batch_group_id,
      Rcpp::Named("batch_position") = result.diagnostics.batch_position,
      Rcpp::Named("true_batched") = result.diagnostics.true_batched,
      Rcpp::Named("cholesky_backend") = result.diagnostics.cholesky_backend
    )
  );
}

Rcpp::DataFrame fastspline_batch_group_table_to_df(
  const FastSplineCudaBatchDiagnostics& diagnostics) {
  const int n = static_cast<int>(diagnostics.group_id.size());
  Rcpp::IntegerVector group_id(n), rows(n), design_cols(n), fit_count(n);
  Rcpp::IntegerVector single_fit_calls(n), cpu_fallback_fits(n);
  Rcpp::IntegerVector unique_designs(n), duplicate_design_fits(n);
  Rcpp::IntegerVector max_fits_per_design(n);
  Rcpp::LogicalVector true_batched(n);
  Rcpp::CharacterVector cholesky_backend(n), status(n), reason(n);
  for (int i = 0; i < n; ++i) {
    group_id[i] = diagnostics.group_id[i];
    rows[i] = diagnostics.group_n[i];
    design_cols[i] = diagnostics.group_design_cols[i];
    fit_count[i] = diagnostics.group_fit_count[i];
    true_batched[i] = diagnostics.group_true_batched[i] != 0;
    single_fit_calls[i] = diagnostics.group_single_fit_calls[i];
    cpu_fallback_fits[i] = diagnostics.group_cpu_fallback_fits[i];
    unique_designs[i] = diagnostics.group_unique_designs[i];
    duplicate_design_fits[i] = diagnostics.group_duplicate_design_fits[i];
    max_fits_per_design[i] = diagnostics.group_max_fits_per_design[i];
    cholesky_backend[i] = diagnostics.group_cholesky_backend[i];
    status[i] = diagnostics.group_status[i];
    reason[i] = diagnostics.group_reason[i];
  }
  return Rcpp::DataFrame::create(
    Rcpp::Named("group_id") = group_id,
    Rcpp::Named("n") = rows,
    Rcpp::Named("design_cols") = design_cols,
    Rcpp::Named("fit_count") = fit_count,
    Rcpp::Named("true_batched") = true_batched,
    Rcpp::Named("single_fit_calls") = single_fit_calls,
    Rcpp::Named("cpu_fallback_fits") = cpu_fallback_fits,
    Rcpp::Named("unique_designs") = unique_designs,
    Rcpp::Named("duplicate_design_fits") = duplicate_design_fits,
    Rcpp::Named("max_fits_per_design") = max_fits_per_design,
    Rcpp::Named("cholesky_backend") = cholesky_backend,
    Rcpp::Named("status") = status,
    Rcpp::Named("reason") = reason,
    Rcpp::Named("stringsAsFactors") = false
  );
}

Rcpp::List fastspline_batch_diagnostics_to_list(
  const FastSplineCudaBatchDiagnostics& diagnostics) {
  return Rcpp::List::create(
    Rcpp::Named("requested_fits") = diagnostics.requested_fits,
    Rcpp::Named("groups") = diagnostics.groups,
    Rcpp::Named("true_batched_groups") = diagnostics.true_batched_groups,
    Rcpp::Named("true_batched_fits") = diagnostics.true_batched_fits,
    Rcpp::Named("single_fit_calls") = diagnostics.single_fit_calls,
    Rcpp::Named("cpu_fallback_fits") = diagnostics.cpu_fallback_fits,
    Rcpp::Named("unique_designs") = diagnostics.unique_designs,
    Rcpp::Named("duplicate_design_fits") = diagnostics.duplicate_design_fits,
    Rcpp::Named("max_fits_per_design") = diagnostics.max_fits_per_design,
    Rcpp::Named("max_group_size") = diagnostics.max_group_size,
    Rcpp::Named("min_group_size") = diagnostics.min_group_size,
    Rcpp::Named("cholesky_backend") = diagnostics.cholesky_backend,
    Rcpp::Named("batch_mode") = diagnostics.batch_mode,
    Rcpp::Named("grouping_sec") = diagnostics.grouping_sec,
    Rcpp::Named("grouping_condition_key_sec") =
      diagnostics.grouping_condition_key_sec,
    Rcpp::Named("grouping_group_key_sec") =
      diagnostics.grouping_group_key_sec,
    Rcpp::Named("grouping_design_build_sec") =
      diagnostics.grouping_design_build_sec,
    Rcpp::Named("grouping_map_insert_sec") =
      diagnostics.grouping_map_insert_sec,
    Rcpp::Named("grouping_design_cache_lookup_sec") =
      diagnostics.grouping_design_cache_lookup_sec,
    Rcpp::Named("grouping_design_cache_insert_sec") =
      diagnostics.grouping_design_cache_insert_sec,
    Rcpp::Named("grouping_group_lookup_sec") =
      diagnostics.grouping_group_lookup_sec,
    Rcpp::Named("grouping_group_insert_sec") =
      diagnostics.grouping_group_insert_sec,
    Rcpp::Named("grouping_group_design_lookup_sec") =
      diagnostics.grouping_group_design_lookup_sec,
    Rcpp::Named("grouping_group_design_copy_sec") =
      diagnostics.grouping_group_design_copy_sec,
    Rcpp::Named("grouping_group_design_index_insert_sec") =
      diagnostics.grouping_group_design_index_insert_sec,
    Rcpp::Named("grouping_request_insert_sec") =
      diagnostics.grouping_request_insert_sec,
    Rcpp::Named("grouping_unaccounted_sec") =
      diagnostics.grouping_unaccounted_sec,
    Rcpp::Named("grouping_group_count") =
      diagnostics.grouping_group_count,
    Rcpp::Named("grouping_design_count") =
      diagnostics.grouping_design_count,
    Rcpp::Named("grouping_condition_key_sort_count") =
      diagnostics.grouping_condition_key_sort_count,
    Rcpp::Named("grouping_string_key_count") =
      diagnostics.grouping_string_key_count,
    Rcpp::Named("structural_group_key_count") =
      diagnostics.structural_group_key_count,
    Rcpp::Named("structural_condition_key_count") =
      diagnostics.structural_condition_key_count,
    Rcpp::Named("string_group_key_count") =
      diagnostics.string_group_key_count,
    Rcpp::Named("string_condition_key_count") =
      diagnostics.string_condition_key_count,
    Rcpp::Named("grouping_group_design_copy_count") =
      diagnostics.grouping_group_design_copy_count,
    Rcpp::Named("grouping_group_design_x_values") =
      diagnostics.grouping_group_design_x_values,
    Rcpp::Named("grouping_group_design_p_values") =
      diagnostics.grouping_group_design_p_values,
    Rcpp::Named("grouping_request_insert_count") =
      diagnostics.grouping_request_insert_count,
    Rcpp::Named("design_cache_hit_count") =
      diagnostics.design_cache_hit_count,
    Rcpp::Named("design_cache_miss_count") =
      diagnostics.design_cache_miss_count,
    Rcpp::Named("design_cache_insert_count") =
      diagnostics.design_cache_insert_count,
    Rcpp::Named("design_cache_entries") =
      diagnostics.design_cache_entries,
    Rcpp::Named("design_build_total_sec") =
      diagnostics.design_build_total_sec,
    Rcpp::Named("design_build_basis_sec") =
      diagnostics.design_build_basis_sec,
    Rcpp::Named("design_build_penalty_sec") =
      diagnostics.design_build_penalty_sec,
    Rcpp::Named("design_build_x_pack_sec") =
      diagnostics.design_build_x_pack_sec,
    Rcpp::Named("design_build_p_pack_sec") =
      diagnostics.design_build_p_pack_sec,
    Rcpp::Named("design_build_alloc_sec") =
      diagnostics.design_build_alloc_sec,
    Rcpp::Named("design_build_column_extract_sec") =
      diagnostics.design_build_column_extract_sec,
    Rcpp::Named("design_build_finite_check_sec") =
      diagnostics.design_build_finite_check_sec,
    Rcpp::Named("design_build_unaccounted_sec") =
      diagnostics.design_build_unaccounted_sec,
    Rcpp::Named("design_build_count") =
      diagnostics.design_build_count,
    Rcpp::Named("design_build_x_values") =
      diagnostics.design_build_x_values,
    Rcpp::Named("design_build_p_values") =
      diagnostics.design_build_p_values,
    Rcpp::Named("design_build_basis_values") =
      diagnostics.design_build_basis_values,
    Rcpp::Named("design_build_penalty_values") =
      diagnostics.design_build_penalty_values,
    Rcpp::Named("design_build_condition_cols") =
      diagnostics.design_build_condition_cols,
    Rcpp::Named("design_build_finite_check_values") =
      diagnostics.design_build_finite_check_values,
    Rcpp::Named("design_build_intercept_count") =
      diagnostics.design_build_intercept_count,
    Rcpp::Named("design_build_one_dimensional_count") =
      diagnostics.design_build_one_dimensional_count,
    Rcpp::Named("design_build_additive_count") =
      diagnostics.design_build_additive_count,
    Rcpp::Named("design_build_tensor_count") =
      diagnostics.design_build_tensor_count,
    Rcpp::Named("design_build_one_dimensional_basis_sec") =
      diagnostics.design_build_one_dimensional_basis_sec,
    Rcpp::Named("design_build_one_dimensional_alloc_sec") =
      diagnostics.design_build_one_dimensional_alloc_sec,
    Rcpp::Named("design_build_one_dimensional_x_pack_sec") =
      diagnostics.design_build_one_dimensional_x_pack_sec,
    Rcpp::Named("design_build_one_dimensional_p_build_sec") =
      diagnostics.design_build_one_dimensional_p_build_sec,
    Rcpp::Named("design_build_one_dimensional_p_pack_sec") =
      diagnostics.design_build_one_dimensional_p_pack_sec,
    Rcpp::Named("design_build_one_dimensional_cols") =
      diagnostics.design_build_one_dimensional_cols,
    Rcpp::Named("design_build_one_dimensional_values") =
      diagnostics.design_build_one_dimensional_values,
    Rcpp::Named("design_build_additive_basis_sec") =
      diagnostics.design_build_additive_basis_sec,
    Rcpp::Named("design_build_additive_alloc_sec") =
      diagnostics.design_build_additive_alloc_sec,
    Rcpp::Named("design_build_additive_x_pack_sec") =
      diagnostics.design_build_additive_x_pack_sec,
    Rcpp::Named("design_build_additive_p_build_sec") =
      diagnostics.design_build_additive_p_build_sec,
    Rcpp::Named("design_build_additive_p_pack_sec") =
      diagnostics.design_build_additive_p_pack_sec,
    Rcpp::Named("design_build_additive_component_count") =
      diagnostics.design_build_additive_component_count,
    Rcpp::Named("design_build_additive_basis_cols") =
      diagnostics.design_build_additive_basis_cols,
    Rcpp::Named("design_build_additive_values") =
      diagnostics.design_build_additive_values,
    Rcpp::Named("design_build_tensor_basis_sec") =
      diagnostics.design_build_tensor_basis_sec,
    Rcpp::Named("design_build_tensor_alloc_sec") =
      diagnostics.design_build_tensor_alloc_sec,
    Rcpp::Named("design_build_tensor_x_pack_sec") =
      diagnostics.design_build_tensor_x_pack_sec,
    Rcpp::Named("design_build_tensor_product_sec") =
      diagnostics.design_build_tensor_product_sec,
    Rcpp::Named("design_build_tensor_p_build_sec") =
      diagnostics.design_build_tensor_p_build_sec,
    Rcpp::Named("design_build_tensor_p_pack_sec") =
      diagnostics.design_build_tensor_p_pack_sec,
    Rcpp::Named("design_build_tensor_cols") =
      diagnostics.design_build_tensor_cols,
    Rcpp::Named("design_build_tensor_values") =
      diagnostics.design_build_tensor_values,
    Rcpp::Named("basis_cache_hit_count") =
      diagnostics.basis_cache_hit_count,
    Rcpp::Named("basis_cache_miss_count") =
      diagnostics.basis_cache_miss_count,
    Rcpp::Named("basis_cache_insert_count") =
      diagnostics.basis_cache_insert_count,
    Rcpp::Named("basis_cache_entries") =
      diagnostics.basis_cache_entries,
    Rcpp::Named("basis_cache_hit_sec") =
      diagnostics.basis_cache_hit_sec,
    Rcpp::Named("basis_cache_miss_build_sec") =
      diagnostics.basis_cache_miss_build_sec,
    Rcpp::Named("basis_build_total_sec") =
      diagnostics.basis_build_total_sec,
    Rcpp::Named("basis_build_alloc_sec") =
      diagnostics.basis_build_alloc_sec,
    Rcpp::Named("basis_build_near_constant_sec") =
      diagnostics.basis_build_near_constant_sec,
    Rcpp::Named("basis_build_knots_sec") =
      diagnostics.basis_build_knots_sec,
    Rcpp::Named("basis_build_knots_copy_sec") =
      diagnostics.basis_build_knots_copy_sec,
    Rcpp::Named("basis_build_knots_sort_sec") =
      diagnostics.basis_build_knots_sort_sec,
    Rcpp::Named("basis_build_knots_center_sec") =
      diagnostics.basis_build_knots_center_sec,
    Rcpp::Named("basis_build_min_gap_sec") =
      diagnostics.basis_build_min_gap_sec,
    Rcpp::Named("basis_build_width_sec") =
      diagnostics.basis_build_width_sec,
    Rcpp::Named("basis_build_eval_sec") =
      diagnostics.basis_build_eval_sec,
    Rcpp::Named("basis_build_eval_fill_sec") =
      diagnostics.basis_build_eval_fill_sec,
    Rcpp::Named("basis_build_normalize_sec") =
      diagnostics.basis_build_normalize_sec,
    Rcpp::Named("basis_build_normalize_scale_sec") =
      diagnostics.basis_build_normalize_scale_sec,
    Rcpp::Named("basis_build_fallback_sec") =
      diagnostics.basis_build_fallback_sec,
    Rcpp::Named("basis_build_return_sec") =
      diagnostics.basis_build_return_sec,
    Rcpp::Named("basis_build_unaccounted_sec") =
      diagnostics.basis_build_unaccounted_sec,
    Rcpp::Named("basis_build_count") =
      diagnostics.basis_build_count,
    Rcpp::Named("basis_build_rows") =
      diagnostics.basis_build_rows,
    Rcpp::Named("basis_build_cols") =
      diagnostics.basis_build_cols,
    Rcpp::Named("basis_build_values") =
      diagnostics.basis_build_values,
    Rcpp::Named("basis_build_near_constant_count") =
      diagnostics.basis_build_near_constant_count,
    Rcpp::Named("basis_build_fallback_row_count") =
      diagnostics.basis_build_fallback_row_count,
    Rcpp::Named("host_pack_sec") = diagnostics.host_pack_sec,
    Rcpp::Named("alloc_sec") = diagnostics.alloc_sec,
    Rcpp::Named("h2d_sec") = diagnostics.h2d_sec,
    Rcpp::Named("h2d_design_sec") = diagnostics.h2d_design_sec,
    Rcpp::Named("h2d_penalty_sec") = diagnostics.h2d_penalty_sec,
    Rcpp::Named("h2d_y_sec") = diagnostics.h2d_y_sec,
    Rcpp::Named("h2d_index_sec") = diagnostics.h2d_index_sec,
    Rcpp::Named("h2d_lambda_sec") = diagnostics.h2d_lambda_sec,
    Rcpp::Named("h2d_active_sec") = diagnostics.h2d_active_sec,
    Rcpp::Named("h2d_copy_count") = diagnostics.h2d_copy_count,
    Rcpp::Named("h2d_bytes") = diagnostics.h2d_bytes,
    Rcpp::Named("h2d_design_bytes") = diagnostics.h2d_design_bytes,
    Rcpp::Named("h2d_y_bytes") = diagnostics.h2d_y_bytes,
    Rcpp::Named("h2d_metadata_bytes") = diagnostics.h2d_metadata_bytes,
    Rcpp::Named("h2d_metadata_coalesced_count") =
      diagnostics.h2d_metadata_coalesced_count,
    Rcpp::Named("h2d_metadata_coalesced_bytes") =
      diagnostics.h2d_metadata_coalesced_bytes,
    Rcpp::Named("h2d_selected_metadata_copy_count") =
      diagnostics.h2d_selected_metadata_copy_count,
    Rcpp::Named("xtx_xty_sec") = diagnostics.xtx_xty_sec,
    Rcpp::Named("pointer_setup_sec") = diagnostics.pointer_setup_sec,
    Rcpp::Named("active_copy_sec") = diagnostics.active_copy_sec,
    Rcpp::Named("build_system_sec") = diagnostics.build_system_sec,
    Rcpp::Named("factor_solve_sec") = diagnostics.factor_solve_sec,
    Rcpp::Named("factor_cholesky_sec") = diagnostics.factor_cholesky_sec,
    Rcpp::Named("factor_rhs_solve_sec") = diagnostics.factor_rhs_solve_sec,
    Rcpp::Named("factor_inverse_solve_sec") =
      diagnostics.factor_inverse_solve_sec,
    Rcpp::Named("residual_summary_sec") = diagnostics.residual_summary_sec,
    Rcpp::Named("d2h_sec") = diagnostics.d2h_sec,
    Rcpp::Named("d2h_residuals_sec") = diagnostics.d2h_residuals_sec,
    Rcpp::Named("d2h_metadata_sec") = diagnostics.d2h_metadata_sec,
    Rcpp::Named("d2h_info_sec") = diagnostics.d2h_info_sec,
    Rcpp::Named("d2h_copy_count") = diagnostics.d2h_copy_count,
    Rcpp::Named("d2h_bytes") = diagnostics.d2h_bytes,
    Rcpp::Named("d2h_residual_bytes") = diagnostics.d2h_residual_bytes,
    Rcpp::Named("d2h_metadata_bytes") = diagnostics.d2h_metadata_bytes,
    Rcpp::Named("d2h_metadata_coalesced_count") =
      diagnostics.d2h_metadata_coalesced_count,
    Rcpp::Named("d2h_metadata_coalesced_bytes") =
      diagnostics.d2h_metadata_coalesced_bytes,
    Rcpp::Named("host_select_sec") = diagnostics.host_select_sec,
    Rcpp::Named("free_sec") = diagnostics.free_sec,
    Rcpp::Named("true_batch_total_sec") = diagnostics.true_batch_total_sec,
    Rcpp::Named("factorization_count") = diagnostics.factorization_count,
    Rcpp::Named("rhs_solve_count") = diagnostics.rhs_solve_count,
    Rcpp::Named("inverse_solve_count") = diagnostics.inverse_solve_count,
    Rcpp::Named("rhs_solve_api_calls") = diagnostics.rhs_solve_api_calls,
    Rcpp::Named("rhs_target_solves") = diagnostics.rhs_target_solves,
    Rcpp::Named("rhs_custom_solve_count") =
      diagnostics.rhs_custom_solve_count,
    Rcpp::Named("rhs_cublas_solve_count") =
      diagnostics.rhs_cublas_solve_count,
    Rcpp::Named("rhs_solve_fallback_count") =
      diagnostics.rhs_solve_fallback_count,
    Rcpp::Named("rhs_custom_solve_sec") =
      diagnostics.rhs_custom_solve_sec,
    Rcpp::Named("rhs_cublas_solve_sec") =
      diagnostics.rhs_cublas_solve_sec,
    Rcpp::Named("candidate_rhs_fused_solve_count") =
      diagnostics.candidate_rhs_fused_solve_count,
    Rcpp::Named("candidate_rhs_materialized_solve_count") =
      diagnostics.candidate_rhs_materialized_solve_count,
    Rcpp::Named("selected_rhs_materialized_solve_count") =
      diagnostics.selected_rhs_materialized_solve_count,
    Rcpp::Named("candidate_beta_values_avoided") =
      diagnostics.candidate_beta_values_avoided,
    Rcpp::Named("summary_candidate_launch_count") =
      diagnostics.summary_candidate_launch_count,
    Rcpp::Named("summary_group_batched_launch_count") =
      diagnostics.summary_group_batched_launch_count,
    Rcpp::Named("summary_group_batched_candidate_count") =
      diagnostics.summary_group_batched_candidate_count,
    Rcpp::Named("edf_trace_shadow_sec") =
      diagnostics.edf_trace_shadow_sec,
    Rcpp::Named("edf_trace_shadow_count") =
      diagnostics.edf_trace_shadow_count,
    Rcpp::Named("edf_trace_mode_full_inverse_count") =
      diagnostics.edf_trace_mode_full_inverse_count,
    Rcpp::Named("edf_trace_mode_shadow_count") =
      diagnostics.edf_trace_mode_shadow_count,
    Rcpp::Named("edf_trace_winner_flip_count") =
      diagnostics.edf_trace_winner_flip_count,
    Rcpp::Named("edf_trace_max_abs_diff") =
      diagnostics.edf_trace_max_abs_diff,
    Rcpp::Named("edf_trace_max_rel_diff") =
      diagnostics.edf_trace_max_rel_diff,
    Rcpp::Named("edf_trace_cuda_sec") =
      diagnostics.edf_trace_cuda_sec,
    Rcpp::Named("edf_trace_cuda_count") =
      diagnostics.edf_trace_cuda_count,
    Rcpp::Named("edf_trace_cuda_candidate_count") =
      diagnostics.edf_trace_cuda_candidate_count,
    Rcpp::Named("edf_trace_full_inverse_skipped_count") =
      diagnostics.edf_trace_full_inverse_skipped_count,
    Rcpp::Named("edf_trace_cuda_fallback_count") =
      diagnostics.edf_trace_cuda_fallback_count,
    Rcpp::Named("edf_trace_cuda_values") =
      diagnostics.edf_trace_cuda_values,
    Rcpp::Named("edf_trace_cuda_kernel_launch_count") =
      diagnostics.edf_trace_cuda_kernel_launch_count,
    Rcpp::Named("edf_trace_cuda_system_count") =
      diagnostics.edf_trace_cuda_system_count,
    Rcpp::Named("edf_trace_cuda_trace_terms") =
      diagnostics.edf_trace_cuda_trace_terms,
    Rcpp::Named("edf_trace_cuda_p_max") =
      diagnostics.edf_trace_cuda_p_max,
    Rcpp::Named("edf_trace_cuda_p_weighted_sum") =
      diagnostics.edf_trace_cuda_p_weighted_sum,
    Rcpp::Named("candidate_inverse_values_avoided") =
      diagnostics.candidate_inverse_values_avoided,
    Rcpp::Named("winning_factor_reuse_count") =
      diagnostics.winning_factor_reuse_count,
    Rcpp::Named("factor_cache_hits") = diagnostics.factor_cache_hits,
    Rcpp::Named("factor_cache_misses") = diagnostics.factor_cache_misses,
    Rcpp::Named("factor_cache_entries") = diagnostics.factor_cache_entries,
    Rcpp::Named("factor_cache_bytes") = diagnostics.factor_cache_bytes,
    Rcpp::Named("lambda_candidates") = diagnostics.lambda_candidates,
    Rcpp::Named("workspace_reuse_count") = diagnostics.workspace_reuse_count,
    Rcpp::Named("workspace_grow_count") = diagnostics.workspace_grow_count,
    Rcpp::Named("workspace_slab_grow_count") =
      diagnostics.workspace_slab_grow_count,
    Rcpp::Named("workspace_slab_reuse_count") =
      diagnostics.workspace_slab_reuse_count,
    Rcpp::Named("workspace_slab_bytes") = diagnostics.workspace_slab_bytes,
    Rcpp::Named("workspace_legacy_alloc_count") =
      diagnostics.workspace_legacy_alloc_count,
    Rcpp::Named("solver_handle_create_count") =
      diagnostics.solver_handle_create_count,
    Rcpp::Named("per_request_design_x_values") =
      diagnostics.per_request_design_x_values,
    Rcpp::Named("duplicate_design_x_values_avoided") =
      diagnostics.duplicate_design_x_values_avoided,
    Rcpp::Named("algebraic_rss_count") = diagnostics.algebraic_rss_count,
    Rcpp::Named("candidate_residual_materialize_count") =
      diagnostics.candidate_residual_materialize_count,
    Rcpp::Named("winning_residual_materialize_count") =
      diagnostics.winning_residual_materialize_count,
    Rcpp::Named("algebraic_rss_clamp_count") =
      diagnostics.algebraic_rss_clamp_count,
    Rcpp::Named("residual_only_batch_count") =
      diagnostics.residual_only_batch_count,
    Rcpp::Named("residual_full_fit_batch_count") =
      diagnostics.residual_full_fit_batch_count,
    Rcpp::Named("residual_only_fit_count") =
      diagnostics.residual_only_fit_count,
    Rcpp::Named("residual_full_fit_materialize_count") =
      diagnostics.residual_full_fit_materialize_count,
    Rcpp::Named("residual_fitted_values_avoided") =
      diagnostics.residual_fitted_values_avoided,
    Rcpp::Named("residual_result_materialize_sec") =
      diagnostics.residual_result_materialize_sec,
    Rcpp::Named("residual_fitted_materialize_sec") =
      diagnostics.residual_fitted_materialize_sec,
    Rcpp::Named("residual_batch_top_level_wall_sec") =
      diagnostics.residual_batch_top_level_wall_sec,
    Rcpp::Named("residual_batch_top_level_unaccounted_sec") =
      diagnostics.residual_batch_top_level_unaccounted_sec,
    Rcpp::Named("group_table") =
      fastspline_batch_group_table_to_df(diagnostics)
  );
}

double get_named_double(Rcpp::List values, const char* name, double fallback) {
  if (!values.containsElementNamed(name)) return fallback;
  return Rcpp::as<double>(values[name]);
}

int get_named_int(Rcpp::List values, const char* name, int fallback) {
  if (!values.containsElementNamed(name)) return fallback;
  return Rcpp::as<int>(values[name]);
}

std::string get_named_string(Rcpp::List values, const char* name,
                             const std::string& fallback) {
  if (!values.containsElementNamed(name)) return fallback;
  return Rcpp::as<std::string>(values[name]);
}

bool get_named_bool(Rcpp::List values, const char* name, bool fallback) {
  if (!values.containsElementNamed(name)) return fallback;
  return Rcpp::as<bool>(values[name]);
}

FastSplineParams parse_fastspline_params(Rcpp::List values) {
  FastSplineParams params = default_fastspline_params();
  params.degree = get_named_int(values, "degree", params.degree);
  params.knots = get_named_int(values, "knots", params.knots);
  params.lambda_min = get_named_double(values, "lambda_min", params.lambda_min);
  params.lambda_max = get_named_double(values, "lambda_max", params.lambda_max);
  params.lambda_count = get_named_int(values, "lambda_count", params.lambda_count);
  params.ridge = get_named_double(values, "ridge", params.ridge);
  params.mode = get_named_string(values, "mode", params.mode);
  return params;
}

HsicOptions parse_hsic_options(Rcpp::List hsic_params,
                               Rcpp::List permutation_params) {
  HsicOptions options = default_hsic_options();
  options.sig = get_named_double(hsic_params, "sig", options.sig);
  options.cuda_max_n = get_named_int(hsic_params, "cuda_max_n",
                                     options.cuda_max_n);
  options.cuda_max_batch_pairs =
    get_named_int(hsic_params, "cuda_max_batch_pairs",
                  options.cuda_max_batch_pairs);
  options.cuda_memory_fallback =
    get_named_bool(hsic_params, "cuda_memory_fallback",
                   options.cuda_memory_fallback);
  options.replicates = get_named_int(permutation_params, "replicates",
                                     options.replicates);
  options.include_observed = get_named_bool(permutation_params,
                                            "include_observed",
                                            options.include_observed);
  if (permutation_params.containsElementNamed("seed")) {
    SEXP seed = permutation_params["seed"];
    if (!Rf_isNull(seed)) {
      options.has_seed = true;
      options.seed = static_cast<unsigned int>(Rcpp::as<int>(seed));
    }
  }
  return options;
}

void apply_ci_options(SkeletonOptions* options,
                      const std::string& ci_method,
                      Rcpp::List hsic_params,
                      Rcpp::List permutation_params,
                      bool ci_diagnostics) {
  options->ci_method = ci_method.empty() ? "dcc.gamma" : ci_method;
  options->hsic_options = parse_hsic_options(hsic_params, permutation_params);
  options->ci_diagnostics_enabled = ci_diagnostics;
}

OrientationOptions make_orientation_options(double alpha,
                                            double index,
                                            bool legacy_index,
                                            bool residual_cache,
                                            const std::string& residual_backend,
                                            const std::string& orientation_device,
                                            int orientation_batch_size,
                                            bool orientation_diagnostics,
                                            bool cuda_residual_fallback,
                                            const FastSplineParams& fastspline_params,
                                            bool orient_collider,
                                            bool solve_confl,
                                            Rcpp::LogicalVector rules,
                                            const std::string& ci_method = "dcc.gamma",
                                            Rcpp::List hsic_params = Rcpp::List::create(),
                                            Rcpp::List permutation_params = Rcpp::List::create(),
                                            bool ci_diagnostics = true) {
  if (rules.size() != 3) {
    Rcpp::stop("rules must have length 3");
  }
  if (rules[0] == NA_LOGICAL || rules[1] == NA_LOGICAL ||
      rules[2] == NA_LOGICAL) {
    Rcpp::stop("rules must not contain NA");
  }
  make_residual_backend_config(residual_backend, fastspline_params);
  if (orientation_device != "auto" && orientation_device != "cpu" &&
      orientation_device != "cuda") {
    Rcpp::stop("Unknown orientation residual device: " + orientation_device);
  }
  OrientationOptions options = default_orientation_options();
  options.alpha = alpha;
  options.index = index;
  options.legacy_index = legacy_index;
  options.residual_cache_enabled = residual_cache;
  options.residual_backend_name = residual_backend;
  options.orientation_residual_device_requested = orientation_device;
  options.orientation_residual_device_reason = "";
  if (residual_backend == "linear") {
    options.orientation_residual_device = "cpu";
    if (orientation_device == "cuda") {
      options.orientation_residual_device_reason =
        "linear orientation residual CUDA device is not implemented";
    } else if (orientation_device == "auto") {
      options.orientation_residual_device_reason =
        "linear orientation residuals use CPU";
    }
  } else if (orientation_device == "cpu") {
    options.orientation_residual_device = "cpu";
  } else {
    options.orientation_residual_device = "cuda";
  }
  options.orientation_batch_size = orientation_batch_size;
  options.orientation_diagnostics_enabled = orientation_diagnostics;
  options.cuda_residual_fallback = cuda_residual_fallback;
  options.fastspline_params = fastspline_params;
  options.orient_collider = orient_collider;
  options.solve_confl = solve_confl;
  options.rule1 = rules[0] == TRUE;
  options.rule2 = rules[1] == TRUE;
  options.rule3 = rules[2] == TRUE;
  options.ci_method = ci_method;
  options.hsic_options = parse_hsic_options(hsic_params, permutation_params);
  options.ci_diagnostics_enabled = ci_diagnostics;
  return options;
}

}  // namespace

extern "C" SEXP C_full_cuda_ci_sha256_utf8(SEXP value_s) {
  BEGIN_RCPP
  Rcpp::CharacterVector value(value_s);
  if (value.size() != 1 || value[0] == NA_STRING) {
    Rcpp::stop("SHA-256 input must be one non-missing string");
  }
  return Rcpp::wrap(fastkpc::full_cuda_ci_sha256_utf8(
    Rcpp::as<std::string>(value)));
  END_RCPP
}

extern "C" SEXP C_full_cuda_ci_contract_identity(
    SEXP json_s,
    SEXP expected_name_s) {
  BEGIN_RCPP
  Rcpp::CharacterVector json(json_s);
  Rcpp::CharacterVector expected_name(expected_name_s);
  if (json.size() != 1 || json[0] == NA_STRING ||
      expected_name.size() != 1 || expected_name[0] == NA_STRING) {
    Rcpp::stop("contract JSON and expected name must be non-missing strings");
  }
  const fastkpc::FullCudaCiContractIdentity identity =
    fastkpc::full_cuda_ci_contract_identity(
      Rcpp::as<std::string>(json),
      Rcpp::as<std::string>(expected_name));
  return Rcpp::List::create(
    Rcpp::Named("contract_name") = identity.contract_name,
    Rcpp::Named("contract_schema_version") =
      identity.contract_schema_version,
    Rcpp::Named("semantic_major") = identity.semantic_major,
    Rcpp::Named("semantic_minor") = identity.semantic_minor,
    Rcpp::Named("semantic_patch") = identity.semantic_patch,
    Rcpp::Named("canonical_json") = identity.canonical_json,
    Rcpp::Named("sha256") = identity.sha256
  );
  END_RCPP
}

extern "C" SEXP C_full_cuda_ci_semantic_abi_info() {
  BEGIN_RCPP
  const fastkpc::FullCudaCiSemanticAbiInfo info =
    fastkpc::full_cuda_ci_semantic_abi_info();
  Rcpp::CharacterVector capability_status(info.capability_status.size());
  Rcpp::CharacterVector capability_status_names(info.capability_status.size());
  for (std::size_t index = 0; index < info.capability_status.size(); ++index) {
    capability_status[index] = info.capability_status[index].second;
    capability_status_names[index] = info.capability_status[index].first;
  }
  capability_status.attr("names") = capability_status_names;
  Rcpp::LogicalVector residency(info.device_residency_flags.size());
  Rcpp::CharacterVector residency_names(info.device_residency_flags.size());
  for (std::size_t index = 0; index < info.device_residency_flags.size();
       ++index) {
    residency[index] = info.device_residency_flags[index].second;
    residency_names[index] = info.device_residency_flags[index].first;
  }
  residency.attr("names") = residency_names;
  return Rcpp::List::create(
    Rcpp::Named("schema_version") = info.schema_version,
    Rcpp::Named("abi_major") = info.abi_major,
    Rcpp::Named("abi_minor") = info.abi_minor,
    Rcpp::Named("capabilities") = info.capabilities,
    Rcpp::Named("capability_status") = capability_status,
    Rcpp::Named("backend_semantic_version") =
      info.backend_semantic_version,
    Rcpp::Named("producer_contract_hash") = info.producer_contract_hash,
    Rcpp::Named("device_residency_flags") = residency,
    Rcpp::Named("semantic_objects") = info.semantic_objects,
    Rcpp::Named("compact_result_fields") = info.compact_result_fields
  );
  END_RCPP
}

namespace {

Rcpp::List full_cuda_ci_compact_record_to_list(
    const fastkpc::FullCudaCiCompactHostRecord& record) {
  return Rcpp::List::create(
    Rcpp::Named("logical_sequence_id") =
      static_cast<double>(record.logical_sequence_id),
    Rcpp::Named("p_value") = record.p_value,
    Rcpp::Named("status") = record.status,
    Rcpp::Named("solver_route") = record.solver_route,
    Rcpp::Named("optimizer_status") = record.optimizer_status,
    Rcpp::Named("dcov_status") = record.dcov_status,
    Rcpp::Named("diagnostic_flags") = record.diagnostic_flags
  );
}

Rcpp::List full_cuda_ci_numerical_diagnostics_to_list(
    const fastkpc::FullCudaCiNumericalDiagnostics& diagnostics) {
  return Rcpp::List::create(
    Rcpp::Named("statistic") = diagnostics.statistic,
    Rcpp::Named("mean") = diagnostics.mean,
    Rcpp::Named("variance") = diagnostics.variance,
    Rcpp::Named("gamma_shape") = diagnostics.gamma_shape,
    Rcpp::Named("gamma_scale") = diagnostics.gamma_scale,
    Rcpp::Named("gamma_iterations") = diagnostics.gamma_iterations
  );
}

Rcpp::List full_cuda_ci_vertical_diagnostics_to_list(
    const fastkpc::FullCudaCiVerticalDiagnostics& value) {
  Rcpp::List result;
  result.push_back(value.component_semantic_version,
                   "component_semantic_version");
  result.push_back(value.n, "n");
  result.push_back(value.target_count, "target_count");
  result.push_back(value.component_build_count, "component_build_count");
  result.push_back(value.component_cache_eviction_count,
                   "component_cache_eviction_count");
  result.push_back(value.pair_evaluation_count, "pair_evaluation_count");
  result.push_back(value.deterministic_replay_count,
                   "deterministic_replay_count");
  result.push_back(value.residual_d2h_count, "residual_d2h_count");
  result.push_back(static_cast<double>(value.residual_d2h_bytes),
                   "residual_d2h_bytes");
  result.push_back(value.component_d2h_count, "component_d2h_count");
  result.push_back(static_cast<double>(value.component_d2h_bytes),
                   "component_d2h_bytes");
  result.push_back(value.compact_result_d2h_count,
                   "compact_result_d2h_count");
  result.push_back(static_cast<double>(value.compact_result_d2h_bytes),
                   "compact_result_d2h_bytes");
  result.push_back(value.cpu_dcov_component_count,
                   "cpu_dcov_component_count");
  result.push_back(value.cpu_dcov_pair_statistic_count,
                   "cpu_dcov_pair_statistic_count");
  result.push_back(value.cpu_gamma_p_value_count,
                   "cpu_gamma_p_value_count");
  result.push_back(value.consumer_event_registration_count,
                   "consumer_event_registration_count");
  result.push_back(value.explicit_host_wait_count,
                   "explicit_host_wait_count");
  result.push_back(value.device_allocation_count,
                   "device_allocation_count");
  result.push_back(value.device_free_count, "device_free_count");
  result.push_back(static_cast<double>(value.device_allocation_bytes),
                   "device_allocation_bytes");
  result.push_back(static_cast<double>(value.peak_live_device_bytes),
                   "peak_live_device_bytes");
  result.push_back(static_cast<double>(value.component_bytes_per_target),
                   "component_bytes_per_target");
  result.push_back(static_cast<double>(value.peak_component_bytes),
                   "peak_component_bytes");
  result.push_back(static_cast<double>(value.live_device_allocations_before),
                   "live_device_allocations_before");
  result.push_back(static_cast<double>(value.live_device_allocations_after),
                   "live_device_allocations_after");
  result.push_back(static_cast<double>(value.live_device_bytes_before),
                   "live_device_bytes_before");
  result.push_back(static_cast<double>(value.live_device_bytes_after),
                   "live_device_bytes_after");
  result.push_back(value.residual_solve_host_ms,
                   "residual_solve_host_ms");
  result.push_back(value.first_component_build_cuda_ms,
                   "first_component_build_cuda_ms");
  result.push_back(value.first_pair_evaluation_cuda_ms,
                   "first_pair_evaluation_cuda_ms");
  result.push_back(value.first_compact_d2h_cuda_ms,
                   "first_compact_d2h_cuda_ms");
  result.push_back(value.replay_component_build_cuda_ms,
                   "replay_component_build_cuda_ms");
  result.push_back(value.replay_pair_evaluation_cuda_ms,
                   "replay_pair_evaluation_cuda_ms");
  result.push_back(value.replay_compact_d2h_cuda_ms,
                   "replay_compact_d2h_cuda_ms");
  result.push_back(value.teardown_host_ms, "teardown_host_ms");
  result.push_back(value.total_host_ms, "total_host_ms");
  result.push_back(value.request_identity_authenticated,
                   "request_identity_authenticated");
  result.push_back(value.prepared_identity_authenticated,
                   "prepared_identity_authenticated");
  result.push_back(value.target_identity_authenticated,
                   "target_identity_authenticated");
  result.push_back(value.residuals_device_resident,
                   "residuals_device_resident");
  result.push_back(value.components_device_resident,
                   "components_device_resident");
  result.push_back(value.compact_result_only_d2h,
                   "compact_result_only_d2h");
  result.push_back(value.eviction_result_bit_identical,
                   "eviction_result_bit_identical");
  result.push_back(value.deterministic_logical_replay,
                   "deterministic_logical_replay");
  result.push_back(value.bounded_allocation, "bounded_allocation");
  result.push_back(value.leak_free_teardown, "leak_free_teardown");
  result.push_back(value.caller_device_restored,
                   "caller_device_restored");
  return result;
}

Rcpp::DataFrame full_cuda_ci_exact_batch_records_to_frame(
    const std::vector<fastkpc::FullCudaCiCompactHostRecord>& records) {
  const int count = static_cast<int>(records.size());
  Rcpp::NumericVector logical_sequence_id(count);
  Rcpp::NumericVector p_value(count);
  Rcpp::CharacterVector status(count);
  Rcpp::CharacterVector solver_route(count);
  Rcpp::CharacterVector optimizer_status(count);
  Rcpp::CharacterVector dcov_status(count);
  Rcpp::IntegerVector diagnostic_flags(count);
  for (int index = 0; index < count; ++index) {
    const fastkpc::FullCudaCiCompactHostRecord& record =
      records[static_cast<std::size_t>(index)];
    logical_sequence_id[index] =
      static_cast<double>(record.logical_sequence_id);
    p_value[index] = record.p_value;
    status[index] = record.status;
    solver_route[index] = record.solver_route;
    optimizer_status[index] = record.optimizer_status;
    dcov_status[index] = record.dcov_status;
    diagnostic_flags[index] = record.diagnostic_flags;
  }
  return Rcpp::DataFrame::create(
    Rcpp::Named("logical_sequence_id") = logical_sequence_id,
    Rcpp::Named("p_value") = p_value,
    Rcpp::Named("status") = status,
    Rcpp::Named("solver_route") = solver_route,
    Rcpp::Named("optimizer_status") = optimizer_status,
    Rcpp::Named("dcov_status") = dcov_status,
    Rcpp::Named("diagnostic_flags") = diagnostic_flags,
    Rcpp::Named("stringsAsFactors") = false
  );
}

Rcpp::DataFrame full_cuda_ci_exact_batch_numerical_to_frame(
    const std::vector<fastkpc::FullCudaCiNumericalDiagnostics>& values) {
  const int count = static_cast<int>(values.size());
  Rcpp::NumericVector statistic(count);
  Rcpp::NumericVector mean(count);
  Rcpp::NumericVector variance(count);
  Rcpp::NumericVector gamma_shape(count);
  Rcpp::NumericVector gamma_scale(count);
  Rcpp::IntegerVector gamma_iterations(count);
  for (int index = 0; index < count; ++index) {
    const fastkpc::FullCudaCiNumericalDiagnostics& value =
      values[static_cast<std::size_t>(index)];
    statistic[index] = value.statistic;
    mean[index] = value.mean;
    variance[index] = value.variance;
    gamma_shape[index] = value.gamma_shape;
    gamma_scale[index] = value.gamma_scale;
    gamma_iterations[index] = value.gamma_iterations;
  }
  return Rcpp::DataFrame::create(
    Rcpp::Named("statistic") = statistic,
    Rcpp::Named("mean") = mean,
    Rcpp::Named("variance") = variance,
    Rcpp::Named("gamma_shape") = gamma_shape,
    Rcpp::Named("gamma_scale") = gamma_scale,
    Rcpp::Named("gamma_iterations") = gamma_iterations,
    Rcpp::Named("stringsAsFactors") = false
  );
}

Rcpp::List full_cuda_ci_exact_batch_diagnostics_to_list(
    const fastkpc::FullCudaCiExactBatchDiagnostics& value) {
  Rcpp::List result;
  result.push_back(value.component_semantic_version,
                   "component_semantic_version");
  result.push_back(value.n, "n");
  result.push_back(value.target_count, "target_count");
  result.push_back(value.pair_count, "pair_count");
  result.push_back(value.referenced_component_count,
                   "referenced_component_count");
  result.push_back(value.component_capacity, "component_capacity");
  result.push_back(value.component_cache_lookup_count,
                   "component_cache_lookup_count");
  result.push_back(value.component_cache_hit_count,
                   "component_cache_hit_count");
  result.push_back(value.component_cache_miss_count,
                   "component_cache_miss_count");
  result.push_back(value.component_build_count, "component_build_count");
  result.push_back(value.component_cache_eviction_count,
                   "component_cache_eviction_count");
  result.push_back(value.pair_evaluation_count, "pair_evaluation_count");
  result.push_back(value.residual_d2h_count, "residual_d2h_count");
  result.push_back(static_cast<double>(value.residual_d2h_bytes),
                   "residual_d2h_bytes");
  result.push_back(value.component_d2h_count, "component_d2h_count");
  result.push_back(static_cast<double>(value.component_d2h_bytes),
                   "component_d2h_bytes");
  result.push_back(value.compact_result_d2h_count,
                   "compact_result_d2h_count");
  result.push_back(static_cast<double>(value.compact_result_d2h_bytes),
                   "compact_result_d2h_bytes");
  result.push_back(value.metadata_h2d_count, "metadata_h2d_count");
  result.push_back(static_cast<double>(value.metadata_h2d_bytes),
                   "metadata_h2d_bytes");
  result.push_back(value.cpu_dcov_component_count,
                   "cpu_dcov_component_count");
  result.push_back(value.cpu_dcov_pair_statistic_count,
                   "cpu_dcov_pair_statistic_count");
  result.push_back(value.cpu_gamma_p_value_count,
                   "cpu_gamma_p_value_count");
  result.push_back(value.consumer_event_registration_count,
                   "consumer_event_registration_count");
  result.push_back(value.explicit_host_wait_count,
                   "explicit_host_wait_count");
  result.push_back(value.device_allocation_count,
                   "device_allocation_count");
  result.push_back(value.device_free_count, "device_free_count");
  result.push_back(static_cast<double>(value.device_allocation_bytes),
                   "device_allocation_bytes");
  result.push_back(static_cast<double>(value.peak_live_device_bytes),
                   "peak_live_device_bytes");
  result.push_back(static_cast<double>(value.component_bytes_per_target),
                   "component_bytes_per_target");
  result.push_back(static_cast<double>(value.peak_component_bytes),
                   "peak_component_bytes");
  result.push_back(static_cast<double>(value.live_device_allocations_before),
                   "live_device_allocations_before");
  result.push_back(static_cast<double>(value.live_device_allocations_after),
                   "live_device_allocations_after");
  result.push_back(static_cast<double>(value.live_device_bytes_before),
                   "live_device_bytes_before");
  result.push_back(static_cast<double>(value.live_device_bytes_after),
                   "live_device_bytes_after");
  result.push_back(value.residual_solve_host_ms,
                   "residual_solve_host_ms");
  result.push_back(value.metadata_h2d_cuda_ms,
                   "metadata_h2d_cuda_ms");
  result.push_back(value.component_build_cuda_ms,
                   "component_build_cuda_ms");
  result.push_back(value.pair_evaluation_cuda_ms,
                   "pair_evaluation_cuda_ms");
  result.push_back(value.compact_d2h_cuda_ms,
                   "compact_d2h_cuda_ms");
  result.push_back(value.dcov_host_boundary_ms,
                   "dcov_host_boundary_ms");
  result.push_back(value.teardown_host_ms, "teardown_host_ms");
  result.push_back(value.total_host_ms, "total_host_ms");
  result.push_back(value.request_identity_authenticated,
                   "request_identity_authenticated");
  result.push_back(value.prepared_identity_authenticated,
                   "prepared_identity_authenticated");
  result.push_back(value.target_identity_authenticated,
                   "target_identity_authenticated");
  result.push_back(value.residuals_device_resident,
                   "residuals_device_resident");
  result.push_back(value.components_device_resident,
                   "components_device_resident");
  result.push_back(value.compact_result_only_d2h,
                   "compact_result_only_d2h");
  result.push_back(value.deterministic_logical_order,
                   "deterministic_logical_order");
  result.push_back(value.component_capacity_respected,
                   "component_capacity_respected");
  result.push_back(value.bounded_allocation, "bounded_allocation");
  result.push_back(value.leak_free_teardown, "leak_free_teardown");
  result.push_back(value.caller_device_restored,
                   "caller_device_restored");
  return result;
}

Rcpp::List full_cuda_ci_legacy_eig_batch_diagnostics_to_list(
    const fastkpc::FullCudaCiLegacyEigBatchDiagnostics& value) {
  Rcpp::List result;
  result.push_back(value.component_semantic_version,
                   "component_semantic_version");
  result.push_back(value.n, "n");
  result.push_back(value.target_count, "target_count");
  result.push_back(value.pair_count, "pair_count");
  result.push_back(value.referenced_component_count,
                   "referenced_component_count");
  result.push_back(value.component_capacity, "component_capacity");
  result.push_back(value.num_col, "num_col");
  result.push_back(value.component_build_count, "component_build_count");
  result.push_back(value.pair_evaluation_count, "pair_evaluation_count");
  result.push_back(value.solver_failure_count, "solver_failure_count");
  result.push_back(value.residual_d2h_count, "residual_d2h_count");
  result.push_back(static_cast<double>(value.residual_d2h_bytes),
                   "residual_d2h_bytes");
  result.push_back(value.component_d2h_count, "component_d2h_count");
  result.push_back(static_cast<double>(value.component_d2h_bytes),
                   "component_d2h_bytes");
  result.push_back(value.compact_result_d2h_count,
                   "compact_result_d2h_count");
  result.push_back(static_cast<double>(value.compact_result_d2h_bytes),
                   "compact_result_d2h_bytes");
  result.push_back(value.compact_status_d2h_count,
                   "compact_status_d2h_count");
  result.push_back(static_cast<double>(value.compact_status_d2h_bytes),
                   "compact_status_d2h_bytes");
  result.push_back(value.metadata_h2d_count, "metadata_h2d_count");
  result.push_back(static_cast<double>(value.metadata_h2d_bytes),
                   "metadata_h2d_bytes");
  result.push_back(value.cpu_dcov_component_count,
                   "cpu_dcov_component_count");
  result.push_back(value.cpu_dcov_eigen_count, "cpu_dcov_eigen_count");
  result.push_back(value.cpu_dcov_pair_statistic_count,
                   "cpu_dcov_pair_statistic_count");
  result.push_back(value.cpu_gamma_p_value_count,
                   "cpu_gamma_p_value_count");
  result.push_back(value.cuda_full_eig_count, "cuda_full_eig_count");
  result.push_back(value.cuda_pair_count, "cuda_pair_count");
  result.push_back(value.cuda_gamma_count, "cuda_gamma_count");
  result.push_back(value.consumer_event_registration_count,
                   "consumer_event_registration_count");
  result.push_back(value.explicit_host_wait_count,
                   "explicit_host_wait_count");
  result.push_back(value.device_allocation_count,
                   "device_allocation_count");
  result.push_back(value.device_free_count, "device_free_count");
  result.push_back(static_cast<double>(value.device_allocation_bytes),
                   "device_allocation_bytes");
  result.push_back(static_cast<double>(value.peak_live_device_bytes),
                   "peak_live_device_bytes");
  result.push_back(static_cast<double>(value.persistent_component_bytes),
                   "persistent_component_bytes");
  result.push_back(static_cast<double>(value.eig_workspace_bytes),
                   "eig_workspace_bytes");
  result.push_back(static_cast<double>(value.pair_workspace_bytes),
                   "pair_workspace_bytes");
  result.push_back(value.residual_solve_host_ms, "residual_solve_host_ms");
  result.push_back(value.metadata_h2d_cuda_ms, "metadata_h2d_cuda_ms");
  result.push_back(value.distance_build_cuda_ms, "distance_build_cuda_ms");
  result.push_back(value.full_eig_cuda_ms, "full_eig_cuda_ms");
  result.push_back(value.component_finalize_cuda_ms,
                   "component_finalize_cuda_ms");
  result.push_back(value.component_build_cuda_ms,
                   "component_build_cuda_ms");
  result.push_back(value.pair_evaluation_cuda_ms,
                   "pair_evaluation_cuda_ms");
  result.push_back(value.compact_d2h_cuda_ms, "compact_d2h_cuda_ms");
  result.push_back(value.dcov_host_boundary_ms, "dcov_host_boundary_ms");
  result.push_back(value.teardown_host_ms, "teardown_host_ms");
  result.push_back(value.total_host_ms, "total_host_ms");
  result.push_back(value.request_identity_authenticated,
                   "request_identity_authenticated");
  result.push_back(value.prepared_identity_authenticated,
                   "prepared_identity_authenticated");
  result.push_back(value.target_identity_authenticated,
                   "target_identity_authenticated");
  result.push_back(value.residuals_device_resident,
                   "residuals_device_resident");
  result.push_back(value.components_device_resident,
                   "components_device_resident");
  result.push_back(value.compact_result_only_d2h,
                   "compact_result_only_d2h");
  result.push_back(value.deterministic_logical_order,
                   "deterministic_logical_order");
  result.push_back(value.component_capacity_respected,
                   "component_capacity_respected");
  result.push_back(value.bounded_allocation, "bounded_allocation");
  result.push_back(value.leak_free_teardown, "leak_free_teardown");
  result.push_back(value.caller_device_restored,
                   "caller_device_restored");
  return result;
}

Rcpp::List full_cuda_ci_vertical_resource_snapshot_to_list(
    const fastkpc::FullCudaCiVerticalResourceSnapshot& value) {
  return Rcpp::List::create(
    Rcpp::Named("live_device_allocations") =
      static_cast<double>(value.live_device_allocations),
    Rcpp::Named("live_device_bytes") =
      static_cast<double>(value.live_device_bytes),
    Rcpp::Named("live_streams") = static_cast<double>(value.live_streams),
    Rcpp::Named("live_events") = static_cast<double>(value.live_events),
    Rcpp::Named("total_device_allocations") =
      static_cast<double>(value.total_device_allocations),
    Rcpp::Named("total_device_frees") =
      static_cast<double>(value.total_device_frees),
    Rcpp::Named("total_stream_creates") =
      static_cast<double>(value.total_stream_creates),
    Rcpp::Named("total_stream_destroys") =
      static_cast<double>(value.total_stream_destroys),
    Rcpp::Named("total_event_creates") =
      static_cast<double>(value.total_event_creates),
    Rcpp::Named("total_event_destroys") =
      static_cast<double>(value.total_event_destroys)
  );
}

}  // namespace

extern "C" SEXP C_full_cuda_ci_phase35_vertical_resource_snapshot() {
  BEGIN_RCPP
  return full_cuda_ci_vertical_resource_snapshot_to_list(
    fastkpc::full_cuda_ci_vertical_resource_snapshot());
  END_RCPP
}

extern "C" SEXP C_full_cuda_ci_phase35_vertical(
    SEXP prepared_s,
    SEXP Y_s,
    SEXP SP_s,
    SEXP planned_route_s,
    SEXP target_keys_s,
    SEXP request_s) {
  BEGIN_RCPP
  FixedSpPreparedHolder* prepared_holder =
    fixed_sp_cuda_prepared_holder(prepared_s, true);
  const fastkpc::PreparedSInfo prepared_info =
    fastkpc::prepared_s_gpu_info(prepared_holder->value);

  if (TYPEOF(Y_s) != REALSXP || !Rf_isMatrix(Y_s) || Rf_isObject(Y_s) ||
      !has_only_attributes(Y_s, {R_DimSymbol, R_DimNamesSymbol})) {
    Rcpp::stop("vertical Y must be a finite double matrix");
  }
  SEXP Y_dimensions = Rf_getAttrib(Y_s, R_DimSymbol);
  if (TYPEOF(Y_dimensions) != INTSXP || XLENGTH(Y_dimensions) != 2 ||
      INTEGER(Y_dimensions)[0] != prepared_info.n ||
      INTEGER(Y_dimensions)[1] < 2) {
    Rcpp::stop("vertical Y shape mismatch");
  }
  const int target_count = INTEGER(Y_dimensions)[1];
  require_bare_double_matrix(Y_s, prepared_info.n, target_count,
                             "vertical Y");
  require_bare_double_matrix(SP_s, prepared_info.penalty_count,
                             target_count, "vertical SP");

  const std::vector<std::string> route_names = bare_character_vector(
    planned_route_s, target_count, "vertical planned_route");
  std::vector<fastkpc::FixedSpRoute> planned_routes;
  planned_routes.reserve(static_cast<std::size_t>(target_count));
  for (const std::string& route : route_names) {
    planned_routes.push_back(fixed_sp_route_from_string(route));
  }
  const std::vector<std::string> target_keys = bare_character_vector(
    target_keys_s, target_count, "vertical target_keys");
  for (int index = 0; index < target_count; ++index) {
    const std::string& key = target_keys[static_cast<std::size_t>(index)];
    const bool valid_sha = key.size() == 64U &&
      std::all_of(key.begin(), key.end(), [](unsigned char character) {
        return (character >= '0' && character <= '9') ||
          (character >= 'a' && character <= 'f');
      });
    if (!valid_sha) {
      Rcpp::stop(
        "vertical target_keys must contain lowercase SHA-256 strings");
    }
    if (std::find(target_keys.begin(), target_keys.begin() + index, key) !=
        target_keys.begin() + index) {
      Rcpp::stop("vertical target_keys must not contain duplicates");
    }
  }

  const fastkpc::FullCudaCiVerticalRequest request =
    parse_full_cuda_ci_vertical_request(request_s);
  fastkpc::FixedSpBatchHostView batch;
  batch.Y = REAL(Y_s);
  batch.SP = REAL(SP_s);
  batch.n = prepared_info.n;
  batch.null_dim = prepared_info.null_dim;
  batch.penalty_count = prepared_info.penalty_count;
  batch.target_count = target_count;
  batch.output_mask = fastkpc::FixedSpOutputResiduals;
  batch.planned_routes = std::move(planned_routes);
  batch.target_keys = target_keys;

  const fastkpc::FullCudaCiVerticalResult result =
    fastkpc::run_full_cuda_ci_phase35_vertical(
      prepared_holder->value, batch, request);
  return Rcpp::List::create(
    Rcpp::Named("schema_version") = result.schema_version,
    Rcpp::Named("request_identity_sha256") =
      result.request_identity_sha256,
    Rcpp::Named("prepared_s_key_sha256") =
      result.prepared_s_key_sha256,
    Rcpp::Named("target_keys") = Rcpp::wrap(result.target_keys),
    Rcpp::Named("first_result") =
      full_cuda_ci_compact_record_to_list(result.first_result),
    Rcpp::Named("replay_result") =
      full_cuda_ci_compact_record_to_list(result.replay_result),
    Rcpp::Named("first_numerical") =
      full_cuda_ci_numerical_diagnostics_to_list(result.first_numerical),
    Rcpp::Named("replay_numerical") =
      full_cuda_ci_numerical_diagnostics_to_list(result.replay_numerical),
    Rcpp::Named("diagnostics") =
      full_cuda_ci_vertical_diagnostics_to_list(result.diagnostics)
  );
  END_RCPP
}

extern "C" SEXP C_full_cuda_ci_phase35_exact_batch(
    SEXP prepared_s,
    SEXP Y_s,
    SEXP SP_s,
    SEXP planned_route_s,
    SEXP target_keys_s,
    SEXP request_s) {
  BEGIN_RCPP
  FixedSpPreparedHolder* prepared_holder =
    fixed_sp_cuda_prepared_holder(prepared_s, true);
  const fastkpc::PreparedSInfo prepared_info =
    fastkpc::prepared_s_gpu_info(prepared_holder->value);

  if (TYPEOF(Y_s) != REALSXP || !Rf_isMatrix(Y_s) || Rf_isObject(Y_s) ||
      !has_only_attributes(Y_s, {R_DimSymbol, R_DimNamesSymbol})) {
    Rcpp::stop("exact batch Y must be a finite double matrix");
  }
  SEXP Y_dimensions = Rf_getAttrib(Y_s, R_DimSymbol);
  if (TYPEOF(Y_dimensions) != INTSXP || XLENGTH(Y_dimensions) != 2 ||
      INTEGER(Y_dimensions)[0] != prepared_info.n ||
      INTEGER(Y_dimensions)[1] < 2) {
    Rcpp::stop("exact batch Y shape mismatch");
  }
  const int target_count = INTEGER(Y_dimensions)[1];
  require_bare_double_matrix(Y_s, prepared_info.n, target_count,
                             "exact batch Y");
  require_bare_double_matrix(SP_s, prepared_info.penalty_count,
                             target_count, "exact batch SP");

  const std::vector<std::string> route_names = bare_character_vector(
    planned_route_s, target_count, "exact batch planned_route");
  std::vector<fastkpc::FixedSpRoute> planned_routes;
  planned_routes.reserve(static_cast<std::size_t>(target_count));
  for (const std::string& route : route_names) {
    planned_routes.push_back(fixed_sp_route_from_string(route));
  }
  const std::vector<std::string> target_keys = bare_character_vector(
    target_keys_s, target_count, "exact batch target_keys");
  for (int index = 0; index < target_count; ++index) {
    const std::string& key = target_keys[static_cast<std::size_t>(index)];
    const bool valid_sha = key.size() == 64U &&
      std::all_of(key.begin(), key.end(), [](unsigned char character) {
        return (character >= '0' && character <= '9') ||
          (character >= 'a' && character <= 'f');
      });
    if (!valid_sha) {
      Rcpp::stop(
        "exact batch target_keys must contain lowercase SHA-256 strings");
    }
    if (std::find(target_keys.begin(), target_keys.begin() + index, key) !=
        target_keys.begin() + index) {
      Rcpp::stop("exact batch target_keys must not contain duplicates");
    }
  }

  const fastkpc::FullCudaCiExactBatchRequest request =
    parse_full_cuda_ci_exact_batch_request(request_s);
  fastkpc::FixedSpBatchHostView batch;
  batch.Y = REAL(Y_s);
  batch.SP = REAL(SP_s);
  batch.n = prepared_info.n;
  batch.null_dim = prepared_info.null_dim;
  batch.penalty_count = prepared_info.penalty_count;
  batch.target_count = target_count;
  batch.output_mask = fastkpc::FixedSpOutputResiduals;
  batch.planned_routes = std::move(planned_routes);
  batch.target_keys = target_keys;

  const fastkpc::FullCudaCiExactBatchResult result =
    fastkpc::run_full_cuda_ci_phase35_exact_batch(
      prepared_holder->value, batch, request);
  return Rcpp::List::create(
    Rcpp::Named("schema_version") = result.schema_version,
    Rcpp::Named("request_identity_sha256") =
      result.request_identity_sha256,
    Rcpp::Named("prepared_s_key_sha256") =
      result.prepared_s_key_sha256,
    Rcpp::Named("target_keys") = Rcpp::wrap(result.target_keys),
    Rcpp::Named("records") =
      full_cuda_ci_exact_batch_records_to_frame(result.records),
    Rcpp::Named("numerical") =
      full_cuda_ci_exact_batch_numerical_to_frame(result.numerical),
    Rcpp::Named("diagnostics") =
      full_cuda_ci_exact_batch_diagnostics_to_list(result.diagnostics)
  );
  END_RCPP
}

extern "C" SEXP C_full_cuda_ci_phase35_legacy_eig_batch(
    SEXP prepared_s,
    SEXP Y_s,
    SEXP SP_s,
    SEXP planned_route_s,
    SEXP target_keys_s,
    SEXP request_s) {
  BEGIN_RCPP
  FixedSpPreparedHolder* prepared_holder =
    fixed_sp_cuda_prepared_holder(prepared_s, true);
  const fastkpc::PreparedSInfo prepared_info =
    fastkpc::prepared_s_gpu_info(prepared_holder->value);

  if (TYPEOF(Y_s) != REALSXP || !Rf_isMatrix(Y_s) || Rf_isObject(Y_s) ||
      !has_only_attributes(Y_s, {R_DimSymbol, R_DimNamesSymbol})) {
    Rcpp::stop("legacy eig batch Y must be a finite double matrix");
  }
  SEXP Y_dimensions = Rf_getAttrib(Y_s, R_DimSymbol);
  if (TYPEOF(Y_dimensions) != INTSXP || XLENGTH(Y_dimensions) != 2 ||
      INTEGER(Y_dimensions)[0] != prepared_info.n ||
      INTEGER(Y_dimensions)[1] < 2) {
    Rcpp::stop("legacy eig batch Y shape mismatch");
  }
  const int target_count = INTEGER(Y_dimensions)[1];
  require_bare_double_matrix(Y_s, prepared_info.n, target_count,
                             "legacy eig batch Y");
  require_bare_double_matrix(SP_s, prepared_info.penalty_count,
                             target_count, "legacy eig batch SP");

  const std::vector<std::string> route_names = bare_character_vector(
    planned_route_s, target_count, "legacy eig batch planned_route");
  std::vector<fastkpc::FixedSpRoute> planned_routes;
  planned_routes.reserve(static_cast<std::size_t>(target_count));
  for (const std::string& route : route_names) {
    planned_routes.push_back(fixed_sp_route_from_string(route));
  }
  const std::vector<std::string> target_keys = bare_character_vector(
    target_keys_s, target_count, "legacy eig batch target_keys");
  for (int index = 0; index < target_count; ++index) {
    const std::string& key = target_keys[static_cast<std::size_t>(index)];
    const bool valid_sha = key.size() == 64U &&
      std::all_of(key.begin(), key.end(), [](unsigned char character) {
        return (character >= '0' && character <= '9') ||
          (character >= 'a' && character <= 'f');
      });
    if (!valid_sha) {
      Rcpp::stop(
        "legacy eig batch target_keys must contain lowercase SHA-256 strings");
    }
    if (std::find(target_keys.begin(), target_keys.begin() + index, key) !=
        target_keys.begin() + index) {
      Rcpp::stop("legacy eig batch target_keys must not contain duplicates");
    }
  }

  const fastkpc::FullCudaCiLegacyEigBatchRequest request =
    parse_full_cuda_ci_legacy_eig_batch_request(request_s);
  fastkpc::FixedSpBatchHostView batch;
  batch.Y = REAL(Y_s);
  batch.SP = REAL(SP_s);
  batch.n = prepared_info.n;
  batch.null_dim = prepared_info.null_dim;
  batch.penalty_count = prepared_info.penalty_count;
  batch.target_count = target_count;
  batch.output_mask = fastkpc::FixedSpOutputResiduals;
  batch.planned_routes = std::move(planned_routes);
  batch.target_keys = target_keys;

  const fastkpc::FullCudaCiLegacyEigBatchResult result =
    fastkpc::run_full_cuda_ci_phase35_legacy_eig_batch(
      prepared_holder->value, batch, request);
  return Rcpp::List::create(
    Rcpp::Named("schema_version") = result.schema_version,
    Rcpp::Named("request_identity_sha256") =
      result.request_identity_sha256,
    Rcpp::Named("prepared_s_key_sha256") =
      result.prepared_s_key_sha256,
    Rcpp::Named("target_keys") = Rcpp::wrap(result.target_keys),
    Rcpp::Named("records") =
      full_cuda_ci_exact_batch_records_to_frame(result.records),
    Rcpp::Named("numerical") =
      full_cuda_ci_exact_batch_numerical_to_frame(result.numerical),
    Rcpp::Named("diagnostics") =
      full_cuda_ci_legacy_eig_batch_diagnostics_to_list(result.diagnostics)
  );
  END_RCPP
}

extern "C" SEXP C_fastkpc_cuda_available() {
  BEGIN_RCPP
  std::string error;
  return Rcpp::wrap(fastkpc_cuda_available(&error));
  END_RCPP
}

extern "C" SEXP C_fastkpc_cuda_device_info() {
  BEGIN_RCPP
  const CudaDeviceInfo info = fastkpc_cuda_device_info();
  return Rcpp::List::create(
    Rcpp::Named("device_id") = info.device_id,
    Rcpp::Named("name") = info.name,
    Rcpp::Named("compute_capability") =
      std::to_string(info.major) + "." + std::to_string(info.minor),
    Rcpp::Named("total_global_mem") = info.total_global_mem
  );
  END_RCPP
}

extern "C" SEXP C_fastkpc_cuda_phase3_environment_identity(SEXP device_s) {
  BEGIN_RCPP
  const int device_id = Rcpp::as<int>(device_s);
  if (device_id < 0 || Rf_length(device_s) != 1) {
    Rcpp::stop("device_id must be one non-negative integer");
  }
  const CudaPhase3EnvironmentIdentity identity =
    fastkpc_cuda_phase3_environment_identity(device_id);
  return Rcpp::List::create(
    Rcpp::Named("schema_version") = identity.schema_version,
    Rcpp::Named("runtime_abi_schema_version") =
      identity.runtime_abi_schema_version,
    Rcpp::Named("configuration_schema_version") =
      identity.configuration_schema_version,
    Rcpp::Named("device_id") = identity.device_id,
    Rcpp::Named("cuda_toolkit_version") = identity.cuda_toolkit_version,
    Rcpp::Named("cuda_driver_version") = identity.cuda_driver_version,
    Rcpp::Named("gpu_name") = identity.gpu_name,
    Rcpp::Named("gpu_uuid") = identity.gpu_uuid,
    Rcpp::Named("compute_capability_major") =
      identity.compute_capability_major,
    Rcpp::Named("compute_capability_minor") =
      identity.compute_capability_minor,
    Rcpp::Named("sm_count") = identity.sm_count,
    Rcpp::Named("cusolver_deterministic_mode_required") =
      identity.cusolver_deterministic_mode_required,
    Rcpp::Named("cublas_math_mode_required") =
      identity.cublas_math_mode_required,
    Rcpp::Named("cublas_atomics_mode_required") =
      identity.cublas_atomics_mode_required,
    Rcpp::Named("cublas_user_workspace_required") =
      identity.cublas_user_workspace_required,
    Rcpp::Named("cublas_workspace_bytes_required") =
      static_cast<double>(identity.cublas_workspace_bytes_required),
    Rcpp::Named("cublas_workspace_min_alignment_required") =
      static_cast<double>(identity.cublas_workspace_min_alignment_required)
  );
  END_RCPP
}

extern "C" SEXP C_fastkpc_cuda_legacy_dcov_gamma_cpp_oracle(
    SEXP xs,
    SEXP ys,
    SEXP numCols,
    SEXP indexs) {
  BEGIN_RCPP
  return fastkpc::legacy_dcov_gamma_cpp_result_to_list(
    fastkpc::legacy_dcov_gamma_cpp_compute(
      Rcpp::NumericVector(xs),
      Rcpp::NumericVector(ys),
      Rcpp::as<int>(numCols),
      Rcpp::as<double>(indexs)));
  END_RCPP
}

extern "C" SEXP C_legacy_dcov_spectra_matvec_cuda(SEXP matrixs,
                                                   SEXP rhss) {
  BEGIN_RCPP
  if (!Rf_isReal(matrixs) || !Rf_isMatrix(matrixs)) {
    Rcpp::stop("matrix must be a numeric matrix");
  }
  if (!Rf_isReal(rhss) || !Rf_isMatrix(rhss)) {
    Rcpp::stop("rhs must be a numeric matrix");
  }
  Rcpp::NumericMatrix matrix(matrixs);
  Rcpp::NumericMatrix rhs(rhss);
  if (matrix.nrow() != matrix.ncol()) {
    Rcpp::stop("matrix must be square");
  }
  if (rhs.nrow() != matrix.nrow()) {
    Rcpp::stop("rhs row count must match matrix dimension");
  }
  if (rhs.ncol() < 1) {
    Rcpp::stop("rhs must have at least one column");
  }
  if (!all_finite(matrix) || !all_finite(rhs)) {
    Rcpp::stop("Data contains missing or infinite values");
  }

  const int n = matrix.nrow();
  const int rhs_count = rhs.ncol();
  const fastkpc::LegacyDcovSpectraMatvecCudaResult result =
    fastkpc::legacy_dcov_spectra_matvec_cuda(REAL(matrixs), REAL(rhss), n,
                                             rhs_count);
  Rcpp::NumericMatrix values(n, rhs_count);
  std::copy(result.values.begin(), result.values.end(), values.begin());
  return Rcpp::List::create(
    Rcpp::Named("values") = values,
    Rcpp::Named("backend") = "cuda-dense-sym-matvec",
    Rcpp::Named("n") = result.n,
    Rcpp::Named("rhs_count") = result.rhs_count,
    Rcpp::Named("kernel_launch_count") = result.kernel_launch_count,
    Rcpp::Named("alloc_ms") = result.alloc_ms,
    Rcpp::Named("h2d_ms") = result.h2d_ms,
    Rcpp::Named("kernel_ms") = result.kernel_ms,
    Rcpp::Named("d2h_ms") = result.d2h_ms,
    Rcpp::Named("free_ms") = result.free_ms,
    Rcpp::Named("total_ms") = result.total_ms
  );
  END_RCPP
}

extern "C" SEXP C_legacy_dcov_spectra_matvec_cuda_handle_create(
    SEXP matrixs) {
  BEGIN_RCPP
  if (!Rf_isReal(matrixs) || !Rf_isMatrix(matrixs)) {
    Rcpp::stop("matrix must be a numeric matrix");
  }
  Rcpp::NumericMatrix matrix(matrixs);
  if (matrix.nrow() != matrix.ncol()) {
    Rcpp::stop("matrix must be square");
  }
  if (!all_finite(matrix)) {
    Rcpp::stop("Data contains missing or infinite values");
  }

  fastkpc::LegacyDcovSpectraMatvecCudaHandle* handle =
    fastkpc::legacy_dcov_spectra_matvec_cuda_handle_create(
      REAL(matrixs), matrix.nrow());
  SEXP ext = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(ext, legacy_dcov_spectra_matvec_handle_finalizer,
                         TRUE);
  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("ptr") = ext,
    Rcpp::Named("backend") = "cuda-dense-sym-matvec-handle",
    Rcpp::Named("n") =
      fastkpc::legacy_dcov_spectra_matvec_cuda_handle_n(handle),
    Rcpp::Named("matrix_h2d_ms") =
      fastkpc::legacy_dcov_spectra_matvec_cuda_handle_matrix_h2d_ms(handle),
    Rcpp::Named("matrix_bytes") =
      static_cast<double>(
        fastkpc::legacy_dcov_spectra_matvec_cuda_handle_matrix_bytes(handle))
  );
  UNPROTECT(1);
  return result;
  END_RCPP
}

extern "C" SEXP C_fixed_sp_cuda_runtime_create(SEXP device_s) {
  BEGIN_RCPP
  const int device_id = scalar_integer(device_s, "device_id");
  if (device_id < 0) Rcpp::stop("device_id must be non-negative");

  FixedSpRuntimeHolder context =
    {fastkpc::create_fixed_sp_runtime(device_id), fixed_sp_current_pid()};
  auto* holder = new FixedSpRuntimeHolder(std::move(context));
  SEXP ext = PROTECT(R_MakeExternalPtr(
    holder, fixed_sp_cuda_runtime_tag(), R_NilValue));
  R_RegisterCFinalizerEx(ext, fixed_sp_cuda_runtime_finalizer, TRUE);
  fixed_sp_owner_acquire(holder, FixedSpOwnerKind::Runtime);
  UNPROTECT(1);
  return ext;
  END_RCPP
}

extern "C" SEXP C_fixed_sp_cuda_live_owner_snapshot() {
  BEGIN_RCPP
  const int runtime =
    g_fixed_sp_runtime_extptr_owners.load(std::memory_order_relaxed);
  const int prepared =
    g_fixed_sp_prepared_extptr_owners.load(std::memory_order_relaxed);
  const int residual =
    g_fixed_sp_residual_extptr_owners.load(std::memory_order_relaxed);
  return Rcpp::List::create(
    Rcpp::Named("runtime") = runtime,
    Rcpp::Named("prepared") = prepared,
    Rcpp::Named("residual") = residual,
    Rcpp::Named("total") = runtime + prepared + residual
  );
  END_RCPP
}

extern "C" SEXP C_fixed_sp_cuda_test_resource_snapshot() {
  BEGIN_RCPP
  const fastkpc::FixedSpResourceSnapshot snapshot =
    fastkpc::test_fixed_sp_cuda_resource_snapshot();
  Rcpp::List result;
  auto append = [&](const fastkpc::FixedSpResourceLifecycleSnapshot& value,
                    const std::string& resource,
                    const std::string& acquire,
                    const std::string& teardown) {
    result[resource + "_" + acquire + "_attempt_count"] =
      static_cast<double>(value.acquire_attempt_count);
    result[resource + "_" + acquire + "_success_count"] =
      static_cast<double>(value.acquire_success_count);
    result[resource + "_" + acquire + "_failure_count"] =
      static_cast<double>(value.acquire_failure_count);
    result[resource + "_" + teardown + "_attempt_count"] =
      static_cast<double>(value.teardown_attempt_count);
    result[resource + "_" + teardown + "_success_count"] =
      static_cast<double>(value.teardown_success_count);
    result[resource + "_" + teardown + "_failure_count"] =
      static_cast<double>(value.teardown_failure_count);
    result[resource + "_active_count"] =
      static_cast<double>(value.active_count);
    result[resource + "_ownership_indeterminate_count"] =
      static_cast<double>(value.ownership_indeterminate_count);
  };
  append(snapshot.cuda_device, "cuda_device", "allocate", "free");
  append(snapshot.cuda_host, "cuda_host", "allocate", "free");
  append(snapshot.stream, "stream", "create", "destroy");
  append(snapshot.event, "event", "create", "destroy");
  append(snapshot.cublas_handle, "cublas_handle", "create", "destroy");
  append(snapshot.cusolver_handle, "cusolver_handle", "create", "destroy");
  append(snapshot.gesvdj_info, "gesvdj_info", "create", "destroy");
  result["cleanup_error_count"] =
    static_cast<double>(snapshot.cleanup_error_count);
  return result;
  END_RCPP
}

extern "C" SEXP C_fixed_sp_cuda_test_device_count() {
  BEGIN_RCPP
  return Rf_ScalarInteger(fastkpc::test_fixed_sp_cuda_device_count());
  END_RCPP
}

extern "C" SEXP C_fixed_sp_cuda_test_set_device(SEXP device_s) {
  BEGIN_RCPP
  const int device_id = scalar_integer(device_s, "test device_id");
  fastkpc::test_fixed_sp_cuda_set_device(device_id);
  return R_NilValue;
  END_RCPP
}

extern "C" SEXP C_fixed_sp_cuda_test_get_device() {
  BEGIN_RCPP
  return Rf_ScalarInteger(fastkpc::test_fixed_sp_cuda_get_device());
  END_RCPP
}

extern "C" SEXP C_fixed_sp_cuda_test_inject_next_resource_acquire_failure(
    SEXP resource_s) {
  BEGIN_RCPP
  const std::string resource = bare_scalar_string(
    resource_s, "resource acquire failure injection target");
  fastkpc::test_inject_next_fixed_sp_cuda_resource_acquire_failure(resource);
  return R_NilValue;
  END_RCPP
}

extern "C" SEXP C_fixed_sp_cuda_test_inject_next_resource_teardown_failure(
    SEXP resource_s) {
  BEGIN_RCPP
  const std::string resource = bare_scalar_string(
    resource_s, "resource teardown failure injection target");
  fastkpc::test_inject_next_fixed_sp_cuda_resource_teardown_failure(resource);
  return R_NilValue;
  END_RCPP
}

extern "C" SEXP
C_fixed_sp_cuda_test_inject_next_resource_post_call_teardown_failure(
    SEXP resource_s) {
  BEGIN_RCPP
  const std::string resource = bare_scalar_string(
    resource_s, "resource post-call teardown failure injection target");
  fastkpc::test_inject_next_fixed_sp_cuda_resource_post_call_teardown_failure(
    resource);
  return R_NilValue;
  END_RCPP
}

extern "C" SEXP
C_fixed_sp_cuda_test_inject_next_prepared_static_shadow_body_failure() {
  BEGIN_RCPP
  fastkpc::test_inject_next_prepared_static_shadow_body_failure();
  return R_NilValue;
  END_RCPP
}

extern "C" SEXP
C_fixed_sp_cuda_test_inject_next_blocked_consumer_launch_failure() {
  BEGIN_RCPP
  fastkpc::test_inject_next_blocked_consumer_launch_failure();
  return R_NilValue;
  END_RCPP
}

extern "C" SEXP C_fixed_sp_cuda_test_force_next_potrf_info(SEXP info_s) {
  BEGIN_RCPP
  if (XLENGTH(info_s) > std::numeric_limits<int>::max()) {
    Rcpp::stop("forced potrf info is too long");
  }
  const int count = static_cast<int>(XLENGTH(info_s));
  require_bare_integer_vector(info_s, count, "forced potrf info");
  std::vector<int> info;
  if (count > 0) {
    const int* values = INTEGER(info_s);
    info.assign(values, values + count);
  }
  fastkpc::test_force_next_fixed_sp_cuda_potrf_info(info);
  return R_NilValue;
  END_RCPP
}

extern "C" SEXP C_fixed_sp_cuda_test_force_next_potrs_info(SEXP info_s) {
  BEGIN_RCPP
  fastkpc::test_force_next_fixed_sp_cuda_potrs_info(
    scalar_integer(info_s, "forced potrs info"));
  return R_NilValue;
  END_RCPP
}

extern "C" SEXP C_fixed_sp_cuda_test_inject_next_device_free_failure() {
  BEGIN_RCPP
  fastkpc::test_inject_next_fixed_sp_cuda_device_free_failure();
  return R_NilValue;
  END_RCPP
}

extern "C" SEXP C_fixed_sp_cuda_test_exercise_resource_teardown_failure(
    SEXP resource_s) {
  BEGIN_RCPP
  const std::string resource = bare_scalar_string(
    resource_s, "resource teardown failure test target");
  fastkpc::test_exercise_fixed_sp_cuda_resource_teardown_failure(resource);
  return R_NilValue;
  END_RCPP
}

extern "C" SEXP C_fixed_sp_cuda_runtime_reserve(
    SEXP runtime_s,
    SEXP n_s,
    SEXP q_s,
    SEXP targets_s,
    SEXP penalties_s,
    SEXP augmented_rows_s) {
  BEGIN_RCPP
  FixedSpRuntimeHolder* holder =
    fixed_sp_cuda_runtime_holder(runtime_s, true);
  fastkpc::FixedSpCapacities capacities;
  capacities.n = scalar_integer(n_s, "n");
  capacities.null_dim = scalar_integer(q_s, "null_dim");
  capacities.target_count = scalar_integer(targets_s, "target_count");
  capacities.penalty_count = scalar_integer(penalties_s, "penalty_count");
  capacities.augmented_rows =
    scalar_integer(augmented_rows_s, "augmented_rows");
  fastkpc::reserve_fixed_sp_runtime(holder->value, capacities);
  return R_NilValue;
  END_RCPP
}

extern "C" SEXP C_fixed_sp_cuda_runtime_info(SEXP runtime_s) {
  BEGIN_RCPP
  FixedSpRuntimeHolder* holder =
    fixed_sp_cuda_runtime_holder(runtime_s, true);
  fastkpc::FixedSpRuntimeInfo info =
    fastkpc::fixed_sp_runtime_info(holder->value);
  const NativeSymbolImageIdentity create_symbol =
    native_symbol_image_identity(
      reinterpret_cast<void*>(&C_fixed_sp_cuda_runtime_create),
      "C_fixed_sp_cuda_runtime_create");
  const NativeSymbolImageIdentity info_symbol =
    native_symbol_image_identity(
      reinterpret_cast<void*>(&C_fixed_sp_cuda_runtime_info),
      "C_fixed_sp_cuda_runtime_info");
  info.create_symbol_image_path = create_symbol.path;
  info.create_symbol_device_major_hex = create_symbol.device_major_hex;
  info.create_symbol_device_minor_hex = create_symbol.device_minor_hex;
  info.create_symbol_inode = create_symbol.inode;
  info.info_symbol_image_path = info_symbol.path;
  info.info_symbol_device_major_hex = info_symbol.device_major_hex;
  info.info_symbol_device_minor_hex = info_symbol.device_minor_hex;
  info.info_symbol_inode = info_symbol.inode;
  return Rcpp::List::create(
    Rcpp::Named("device_id") = info.device_id,
    Rcpp::Named("gpu_name") = info.gpu_name,
    Rcpp::Named("runtime_abi_schema_version") =
      info.runtime_abi_schema_version,
    Rcpp::Named("configuration_schema_version") =
      info.configuration_schema_version,
    Rcpp::Named("gpu_uuid") = info.gpu_uuid,
    Rcpp::Named("create_symbol_image_path") = info.create_symbol_image_path,
    Rcpp::Named("create_symbol_device_major_hex") =
      info.create_symbol_device_major_hex,
    Rcpp::Named("create_symbol_device_minor_hex") =
      info.create_symbol_device_minor_hex,
    Rcpp::Named("create_symbol_inode") = info.create_symbol_inode,
    Rcpp::Named("info_symbol_image_path") = info.info_symbol_image_path,
    Rcpp::Named("info_symbol_device_major_hex") =
      info.info_symbol_device_major_hex,
    Rcpp::Named("info_symbol_device_minor_hex") =
      info.info_symbol_device_minor_hex,
    Rcpp::Named("info_symbol_inode") = info.info_symbol_inode,
    Rcpp::Named("creator_pid") = static_cast<double>(info.creator_pid),
    Rcpp::Named("generation") = static_cast<double>(info.generation),
    Rcpp::Named("runtime_context_create_count") =
      info.runtime_context_create_count,
    Rcpp::Named("cuda_device_allocation_count") =
      info.cuda_device_allocation_count,
    Rcpp::Named("cuda_host_allocation_count") =
      info.cuda_host_allocation_count,
    Rcpp::Named("stream_create_count") = info.stream_create_count,
    Rcpp::Named("event_create_count") = info.event_create_count,
    Rcpp::Named("cublas_handle_create_count") =
      info.cublas_handle_create_count,
    Rcpp::Named("cusolver_handle_create_count") =
      info.cusolver_handle_create_count,
    Rcpp::Named("gesvdj_info_create_count") =
      info.gesvdj_info_create_count,
    Rcpp::Named("gesvdj_info_destroy_count") =
      info.gesvdj_info_destroy_count,
    Rcpp::Named("workspace_reserve_count") = info.workspace_reserve_count,
    Rcpp::Named("workspace_grow_count") = info.workspace_grow_count,
    Rcpp::Named("stable_workspace_grow_count") =
      info.stable_workspace_grow_count,
    Rcpp::Named("cuda_device_synchronize_count") =
      info.cuda_device_synchronize_count,
    Rcpp::Named("cholesky_factor_checkpoint_record_count") =
      info.cholesky_factor_checkpoint_record_count,
    Rcpp::Named("cholesky_factor_checkpoint_wait_count") =
      info.cholesky_factor_checkpoint_wait_count,
    Rcpp::Named("cholesky_solve_checkpoint_record_count") =
      info.cholesky_solve_checkpoint_record_count,
    Rcpp::Named("cholesky_solve_checkpoint_wait_count") =
      info.cholesky_solve_checkpoint_wait_count,
    Rcpp::Named("qr_checkpoint_record_count") =
      info.qr_checkpoint_record_count,
    Rcpp::Named("qr_checkpoint_wait_count") =
      info.qr_checkpoint_wait_count,
    Rcpp::Named("svd_checkpoint_record_count") =
      info.svd_checkpoint_record_count,
    Rcpp::Named("svd_checkpoint_wait_count") =
      info.svd_checkpoint_wait_count,
    Rcpp::Named("workspace_bytes") =
      static_cast<double>(info.workspace_bytes),
    Rcpp::Named("cublas_workspace_bytes") =
      static_cast<double>(info.cublas_workspace_bytes),
    Rcpp::Named("cublas_workspace_alignment") =
      static_cast<double>(info.cublas_workspace_alignment),
    Rcpp::Named("eigen_workspace_bytes") =
      static_cast<double>(info.eigen_workspace_bytes),
    Rcpp::Named("qr_workspace_bytes") =
      static_cast<double>(info.qr_workspace_bytes),
    Rcpp::Named("svd_workspace_bytes") =
      static_cast<double>(info.svd_workspace_bytes),
    Rcpp::Named("augmented_workspace_bytes") =
      static_cast<double>(info.augmented_workspace_bytes),
    Rcpp::Named("aggregate_factor_workspace_bytes") =
      static_cast<double>(info.aggregate_factor_workspace_bytes),
    Rcpp::Named("cuda_toolkit_version") = info.cuda_toolkit_version,
    Rcpp::Named("cuda_driver_version") = info.cuda_driver_version,
    Rcpp::Named("compute_capability_major") =
      info.compute_capability_major,
    Rcpp::Named("compute_capability_minor") =
      info.compute_capability_minor,
    Rcpp::Named("sm_count") = info.sm_count,
    Rcpp::Named("cusolver_deterministic_mode") =
      info.cusolver_deterministic_mode_enabled ? "enabled" : "disabled",
    Rcpp::Named("cublas_math_mode") =
      info.cublas_pedantic_math_enabled ? "pedantic" : "not_pedantic",
    Rcpp::Named("cublas_atomics_mode") =
      info.cublas_atomics_not_allowed ? "not_allowed" : "allowed",
    Rcpp::Named("cublas_user_workspace_installed") =
      info.cublas_user_workspace_installed,
    Rcpp::Named("freed") = info.freed
  );
  END_RCPP
}

extern "C" SEXP C_fixed_sp_cuda_runtime_free(SEXP runtime_s) {
  BEGIN_RCPP
  FixedSpRuntimeHolder* holder =
    fixed_sp_cuda_runtime_holder(runtime_s, false);
  fastkpc::free_fixed_sp_runtime(&holder->value);
  fixed_sp_owner_release(holder, FixedSpOwnerKind::Runtime);
  return R_NilValue;
  END_RCPP
}

extern "C" SEXP C_fixed_sp_cuda_prepared_create(SEXP runtime_s,
                                                 SEXP dto_s) {
  BEGIN_RCPP
  FixedSpRuntimeHolder* runtime_holder =
    fixed_sp_cuda_runtime_holder(runtime_s, true);
  require_prepared_dto_fields(dto_s);
  Rcpp::List dto(dto_s);

  const std::string schema_version =
    bare_scalar_string(dto[0], "schema_version");
  const std::string prepared_schema =
    bare_scalar_string(dto[8], "prepared_s_setup_schema_version");
  const std::string native_schema =
    bare_scalar_string(dto[9], "native_dto_schema_version");
  if (schema_version != "full-cuda-ci-prepared-s-native-dto-v1" ||
      native_schema != schema_version ||
      prepared_schema != "full-cuda-ci-prepared-s-setup-v1") {
    Rcpp::stop("prepared DTO schema lineage mismatch");
  }
  static const char* lineage_names[] = {
    "dataset_sha256", "prepared_s_key_sha256", "same_S_group_id",
    "phase1_setup_fingerprint", "provider_fingerprint",
    "semantic_fingerprint", "representation_fingerprint"
  };
  for (int index = 0; index < 7; ++index) {
    require_sha256_string(dto[index + 1], lineage_names[index]);
  }

  const int data_p = positive_scalar_integer(dto[10], "data_p");
  if (data_p != 48) {
    Rcpp::stop("canonical prepared DTO data_p must be 48");
  }
  const int n = positive_scalar_integer(dto[11], "n");
  const int p = positive_scalar_integer(dto[12], "coefficient_dim");
  const int q = positive_scalar_integer(dto[13], "null_dim");
  const int penalties = positive_scalar_integer(dto[14], "penalty_count");
  if (q > p) Rcpp::stop("null_dim must not exceed coefficient_dim");
  require_finite_double_matrix(dto[15], n, p, "X");

  const std::string constraint_mode =
    bare_scalar_string(dto[16], "constraint_mode");
  const bool identity_constraint = constraint_mode == "identity";
  if (!identity_constraint && constraint_mode != "explicit") {
    Rcpp::stop("constraint_mode must be identity or explicit");
  }
  require_finite_double_matrix(dto[18], p, p, "gram_matrix");
  const double* Z = nullptr;
  const double* gram = nullptr;
  if (identity_constraint) {
    if (q != p || dto[17] != R_NilValue || dto[19] != R_NilValue) {
      Rcpp::stop("identity constraint DTO shape mismatch");
    }
    gram = REAL(dto[18]);
  } else {
    if (q >= p || dto[17] == R_NilValue || dto[19] == R_NilValue) {
      Rcpp::stop("explicit constraint DTO shape mismatch");
    }
    require_finite_double_matrix(dto[17], p, q, "constraint_nullspace");
    require_finite_double_matrix(dto[19], q, q, "nullspace_gram_matrix");
    Z = REAL(dto[17]);
    gram = REAL(dto[19]);
  }

  SEXP penalty_blocks_s = dto[20];
  if (TYPEOF(penalty_blocks_s) != VECSXP ||
      XLENGTH(penalty_blocks_s) != penalties ||
      Rf_isObject(penalty_blocks_s) ||
      !has_only_attributes(penalty_blocks_s, {R_NamesSymbol})) {
    Rcpp::stop("penalty_blocks must be a canonical named list");
  }
  SEXP penalty_block_names =
    Rf_getAttrib(penalty_blocks_s, R_NamesSymbol);
  if (TYPEOF(penalty_block_names) != STRSXP ||
      XLENGTH(penalty_block_names) != penalties ||
      Rf_isObject(penalty_block_names) ||
      ATTRIB(penalty_block_names) != R_NilValue) {
    Rcpp::stop("penalty_blocks must be a canonical named list");
  }
  for (int index = 0; index < penalties; ++index) {
    const std::string expected_name =
      "penalty_" + std::to_string(index + 1);
    if (STRING_ELT(penalty_block_names, index) == NA_STRING ||
        std::string(CHAR(STRING_ELT(penalty_block_names, index))) !=
          expected_name) {
      Rcpp::stop("penalty_blocks must be a canonical named list");
    }
  }
  require_bare_integer_vector(
    dto[21], penalties, "penalty_offsets_zero_based");
  require_bare_integer_vector(dto[22], penalties, "penalty_ranks");
  require_bare_integer_vector(
    dto[23], penalties, "penalty_sp_indices_zero_based");
  SEXP penalty_labels_s = dto[24];
  if (TYPEOF(penalty_labels_s) != STRSXP ||
      XLENGTH(penalty_labels_s) != penalties ||
      Rf_isObject(penalty_labels_s) || ATTRIB(penalty_labels_s) != R_NilValue) {
    Rcpp::stop("penalty_sp_labels must be a bare character vector");
  }

  fastkpc::PreparedSHostView setup;
  setup.dataset_sha256 = Rcpp::as<std::string>(dto[1]);
  setup.prepared_s_key_sha256 = Rcpp::as<std::string>(dto[2]);
  setup.same_s_group_id = Rcpp::as<std::string>(dto[3]);
  setup.semantic_fingerprint = Rcpp::as<std::string>(dto[6]);
  setup.representation_fingerprint = Rcpp::as<std::string>(dto[7]);
  setup.n = n;
  setup.coefficient_dim = p;
  setup.null_dim = q;
  setup.penalty_count = penalties;
  setup.X = REAL(dto[15]);
  setup.Z = Z;
  setup.gram = gram;
  setup.penalty_blocks.reserve(static_cast<std::size_t>(penalties));
  setup.penalty_dimensions.reserve(static_cast<std::size_t>(penalties));
  setup.penalty_offsets_zero_based.reserve(
    static_cast<std::size_t>(penalties));
  setup.penalty_ranks.reserve(static_cast<std::size_t>(penalties));
  setup.penalty_sp_indices_zero_based.reserve(
    static_cast<std::size_t>(penalties));

  for (int index = 0; index < penalties; ++index) {
    SEXP block = VECTOR_ELT(penalty_blocks_s, index);
    if (TYPEOF(block) != REALSXP || !Rf_isMatrix(block) ||
        Rf_isObject(block) ||
        !has_only_attributes(block, {R_DimSymbol, R_DimNamesSymbol})) {
      Rcpp::stop("penalty block must be a finite double matrix");
    }
    SEXP dimensions = Rf_getAttrib(block, R_DimSymbol);
    if (TYPEOF(dimensions) != INTSXP || XLENGTH(dimensions) != 2 ||
        INTEGER(dimensions)[0] <= 0 ||
        INTEGER(dimensions)[0] != INTEGER(dimensions)[1]) {
      Rcpp::stop("penalty block shape mismatch");
    }
    const int dimension = INTEGER(dimensions)[0];
    require_finite_double_matrix(
      block, dimension, dimension, "penalty block");
    const int offset = INTEGER(dto[21])[index];
    if (offset < 0 || dimension > p || offset > p - dimension) {
      Rcpp::stop("penalty offset is out of range");
    }
    const int rank = INTEGER(dto[22])[index];
    if (rank < 0 || rank > dimension) {
      Rcpp::stop("penalty rank is out of range");
    }
    const int sp_index = INTEGER(dto[23])[index];
    if (sp_index < 0 || sp_index >= penalties) {
      Rcpp::stop("penalty SP index is out of range");
    }
    if (sp_index != index) {
      Rcpp::stop("penalty-to-SP mapping must be identity");
    }
    if (STRING_ELT(penalty_labels_s, index) == NA_STRING ||
        CHAR(STRING_ELT(penalty_labels_s, index))[0] == '\0') {
      Rcpp::stop("penalty SP labels must be non-empty");
    }
    setup.penalty_blocks.push_back(REAL(block));
    setup.penalty_dimensions.push_back(dimension);
    setup.penalty_offsets_zero_based.push_back(offset);
    setup.penalty_ranks.push_back(rank);
    setup.penalty_sp_indices_zero_based.push_back(sp_index);
  }

  const double* H = nullptr;
  if (dto[25] != R_NilValue) {
    if (identity_constraint) {
      Rcpp::stop("Phase 3C non-null H requires an explicit constraint");
    }
    require_finite_double_matrix(dto[25], p, p, "H");
    H = REAL(dto[25]);
  }
  setup.H = H;
  if (bare_scalar_string(dto[26], "weights_policy") != "none-or-unit") {
    Rcpp::stop("prepared DTO weights policy is unsupported");
  }
  if (bare_scalar_string(dto[27], "offset_policy") != "none-or-zero") {
    Rcpp::stop("prepared DTO offset policy is unsupported");
  }

  FixedSpPreparedHolder prepared =
    {fastkpc::create_prepared_s_gpu(runtime_holder->value, setup),
     fixed_sp_current_pid()};
  auto* holder = new FixedSpPreparedHolder(std::move(prepared));
  SEXP ext = PROTECT(R_MakeExternalPtr(
    holder, fixed_sp_cuda_prepared_tag(), R_NilValue));
  R_RegisterCFinalizerEx(ext, fixed_sp_cuda_prepared_finalizer, TRUE);
  fixed_sp_owner_acquire(holder, FixedSpOwnerKind::Prepared);
  UNPROTECT(1);
  return ext;
  END_RCPP
}

extern "C" SEXP C_fixed_sp_cuda_prepared_info(SEXP prepared_s) {
  BEGIN_RCPP
  FixedSpPreparedHolder* holder =
    fixed_sp_cuda_prepared_holder(prepared_s, true);
  const fastkpc::PreparedSInfo info =
    fastkpc::prepared_s_gpu_info(holder->value);
  return Rcpp::List::create(
    Rcpp::Named("prepared_s_key_sha256") = info.prepared_s_key_sha256,
    Rcpp::Named("n") = info.n,
    Rcpp::Named("coefficient_dim") = info.coefficient_dim,
    Rcpp::Named("null_dim") = info.null_dim,
    Rcpp::Named("penalty_count") = info.penalty_count,
    Rcpp::Named("setup_h2d_upload_count") = info.setup_h2d_upload_count,
    Rcpp::Named("setup_h2d_bytes") =
      static_cast<double>(info.setup_h2d_bytes),
    Rcpp::Named("penalty_root_build_count") =
      info.penalty_root_build_count,
    Rcpp::Named("penalty_root_rank_mismatch_count") =
      info.penalty_root_rank_mismatch_count,
    Rcpp::Named("penalty_root_bytes") =
      static_cast<double>(info.penalty_root_bytes),
    Rcpp::Named("penalty_root_build_ms") = info.penalty_root_build_ms,
    Rcpp::Named("penalty_root_matrix_count") =
      info.penalty_root_matrix_count,
    Rcpp::Named("penalty_root_row_count") = info.penalty_root_row_count,
    Rcpp::Named("H_root_matrix_count") = info.H_root_matrix_count,
    Rcpp::Named("H_root_rank") = info.H_root_rank,
    Rcpp::Named("setup_shadow_d2h_count") = info.setup_shadow_d2h_count,
    Rcpp::Named("setup_shadow_d2h_bytes") =
      static_cast<double>(info.setup_shadow_d2h_bytes),
    Rcpp::Named("augmented_test_shadow_d2h_count") =
      info.augmented_test_shadow_d2h_count,
    Rcpp::Named("augmented_test_shadow_d2h_bytes") =
      static_cast<double>(info.augmented_test_shadow_d2h_bytes),
    Rcpp::Named("projected_H_test_shadow_d2h_count") =
      info.projected_H_test_shadow_d2h_count,
    Rcpp::Named("projected_H_test_shadow_d2h_bytes") =
      static_cast<double>(info.projected_H_test_shadow_d2h_bytes),
    Rcpp::Named("coefficient_output_capacity") =
      static_cast<double>(info.coefficient_output_capacity),
    Rcpp::Named("generation") = static_cast<double>(info.generation),
    Rcpp::Named("output_slot_leased") = info.output_slot_leased,
    Rcpp::Named("output_slot_state") = info.output_slot_state,
    Rcpp::Named("output_slot_poison_reason") =
      info.output_slot_poison_reason
  );
  END_RCPP
}

extern "C" SEXP C_fixed_sp_cuda_prepared_free(SEXP prepared_s) {
  BEGIN_RCPP
  FixedSpPreparedHolder* holder =
    fixed_sp_cuda_prepared_holder(prepared_s, false);
  fastkpc::free_prepared_s_gpu(&holder->value);
  fixed_sp_owner_release(holder, FixedSpOwnerKind::Prepared);
  return R_NilValue;
  END_RCPP
}

extern "C" SEXP C_fixed_sp_cuda_test_prepared_static_shadow(
    SEXP prepared_s) {
  BEGIN_RCPP
  FixedSpPreparedHolder* holder =
    fixed_sp_cuda_prepared_holder(prepared_s, true);
  const fastkpc::PreparedSStaticShadow shadow =
    fastkpc::test_prepared_s_static_shadow(holder->value);
  Rcpp::NumericMatrix X_null(shadow.n, shadow.null_dim);
  std::copy(shadow.X_null.begin(), shadow.X_null.end(), X_null.begin());
  Rcpp::NumericMatrix gram(shadow.null_dim, shadow.null_dim);
  std::copy(shadow.gram.begin(), shadow.gram.end(), gram.begin());
  Rcpp::NumericVector projected(shadow.projected_penalties.size());
  std::copy(shadow.projected_penalties.begin(),
            shadow.projected_penalties.end(), projected.begin());
  projected.attr("dim") = Rcpp::IntegerVector::create(
    shadow.null_dim, shadow.null_dim, shadow.penalty_count);
  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("X_null") = X_null,
    Rcpp::Named("gram") = gram,
    Rcpp::Named("projected_penalties") = projected,
    Rcpp::Named("projected_H") = R_NilValue
  );
  if (shadow.has_H) {
    Rcpp::NumericMatrix projected_H(shadow.null_dim, shadow.null_dim);
    std::copy(shadow.projected_H.begin(), shadow.projected_H.end(),
              projected_H.begin());
    result["projected_H"] = projected_H;
  }
  return result;
  END_RCPP
}

extern "C" SEXP C_fixed_sp_cuda_prepared_materialize_roots_for_test(
    SEXP prepared_s) {
  BEGIN_RCPP
  FixedSpPreparedHolder* holder =
    fixed_sp_cuda_prepared_holder(prepared_s, true);
  const fastkpc::PreparedSRootsShadow shadow =
    fastkpc::test_prepared_s_roots_shadow(holder->value);

  Rcpp::List penalty_roots(shadow.penalty_root_ranks.size());
  for (std::size_t penalty = 0;
       penalty < shadow.penalty_root_ranks.size(); ++penalty) {
    const int rank = shadow.penalty_root_ranks[penalty];
    const int offset = shadow.penalty_root_offsets[penalty];
    Rcpp::NumericMatrix root(rank, shadow.null_dim);
    for (int column = 0; column < shadow.null_dim; ++column) {
      for (int row = 0; row < rank; ++row) {
        root(row, column) = shadow.penalty_roots[
          static_cast<std::size_t>(offset + row) +
          static_cast<std::size_t>(shadow.total_penalty_root_rows) * column];
      }
    }
    penalty_roots[static_cast<R_xlen_t>(penalty)] = root;
  }

  Rcpp::List result;
  result.push_back(penalty_roots, "penalty_roots");
  if (shadow.has_H) {
    Rcpp::NumericMatrix H_root(shadow.H_root_rank, shadow.null_dim);
    std::copy(shadow.H_root.begin(), shadow.H_root.end(), H_root.begin());
    result.push_back(H_root, "H_root");
  } else {
    result.push_back(R_NilValue, "H_root");
  }
  result.push_back(shadow.H_root_rank, "H_root_rank");
  return result;
  END_RCPP
}

extern "C" SEXP C_fixed_sp_cuda_build_augmented_for_test(
    SEXP prepared_s,
    SEXP Y_s,
    SEXP SP_s,
    SEXP target_index_s) {
  BEGIN_RCPP
  FixedSpPreparedHolder* holder =
    fixed_sp_cuda_prepared_holder(prepared_s, true);
  const fastkpc::PreparedSInfo info =
    fastkpc::prepared_s_gpu_info(holder->value);
  if (TYPEOF(Y_s) != REALSXP || Rf_isObject(Y_s) ||
      ATTRIB(Y_s) != R_NilValue) {
    Rcpp::stop("augmented Y must be a bare double vector");
  }
  if (TYPEOF(SP_s) != REALSXP || Rf_isObject(SP_s) ||
      ATTRIB(SP_s) != R_NilValue) {
    Rcpp::stop("augmented SP must be a bare double vector");
  }
  if (XLENGTH(Y_s) != static_cast<R_xlen_t>(info.n)) {
    Rcpp::stop("augmented Y shape mismatch");
  }
  if (XLENGTH(SP_s) != static_cast<R_xlen_t>(info.penalty_count)) {
    Rcpp::stop("augmented SP shape mismatch");
  }
  const int target_index =
    scalar_integer(target_index_s, "target_index");
  if (target_index <= 0) {
    Rcpp::stop("target_index must be positive");
  }

  const fastkpc::FixedSpAugmentedSystemShadow shadow =
    fastkpc::test_build_fixed_sp_augmented_shadow(
      holder->value, REAL(Y_s), static_cast<std::size_t>(XLENGTH(Y_s)),
      REAL(SP_s), static_cast<std::size_t>(XLENGTH(SP_s)),
      target_index - 1);
  Rcpp::NumericMatrix B(shadow.rows, shadow.cols);
  std::copy(shadow.B.begin(), shadow.B.end(), B.begin());
  Rcpp::NumericVector c(shadow.c.size());
  std::copy(shadow.c.begin(), shadow.c.end(), c.begin());
  return Rcpp::List::create(
    Rcpp::Named("B") = B,
    Rcpp::Named("c") = c,
    Rcpp::Named("leading_dimension") = shadow.leading_dimension,
    Rcpp::Named("rows") = shadow.rows,
    Rcpp::Named("cols") = shadow.cols,
    Rcpp::Named("target_index") = shadow.target_index + 1
  );
  END_RCPP
}

extern "C" SEXP C_fixed_sp_cuda_solve_batch(
    SEXP prepared_s,
    SEXP Y_s,
    SEXP SP_s,
    SEXP planned_route_s,
    SEXP target_keys_s,
    SEXP outputs_s) {
  BEGIN_RCPP
  FixedSpPreparedHolder* prepared_holder =
    fixed_sp_cuda_prepared_holder(prepared_s, true);
  const fastkpc::PreparedSInfo prepared_info =
    fastkpc::prepared_s_gpu_info(prepared_holder->value);

  if (TYPEOF(Y_s) != REALSXP || !Rf_isMatrix(Y_s) || Rf_isObject(Y_s) ||
      !has_only_attributes(Y_s, {R_DimSymbol, R_DimNamesSymbol})) {
    Rcpp::stop("Y must be a finite double matrix");
  }
  SEXP Y_dimensions = Rf_getAttrib(Y_s, R_DimSymbol);
  if (TYPEOF(Y_dimensions) != INTSXP || XLENGTH(Y_dimensions) != 2 ||
      INTEGER(Y_dimensions)[0] != prepared_info.n ||
      INTEGER(Y_dimensions)[1] <= 0) {
    Rcpp::stop("Y shape mismatch");
  }
  const int target_count = INTEGER(Y_dimensions)[1];
  require_bare_double_matrix(
    Y_s, prepared_info.n, target_count, "Y");
  require_bare_double_matrix(
    SP_s, prepared_info.penalty_count, target_count, "SP");

  const std::vector<std::string> route_names = bare_character_vector(
    planned_route_s, target_count, "planned_route");
  std::vector<fastkpc::FixedSpRoute> planned_routes;
  planned_routes.reserve(static_cast<std::size_t>(target_count));
  for (const std::string& route : route_names) {
    planned_routes.push_back(fixed_sp_route_from_string(route));
  }

  const std::vector<std::string> target_keys = bare_character_vector(
    target_keys_s, target_count, "target_keys");
  for (int index = 0; index < target_count; ++index) {
    const std::string& key = target_keys[static_cast<std::size_t>(index)];
    const bool valid_sha = key.size() == 64U &&
      std::all_of(key.begin(), key.end(), [](unsigned char character) {
        return (character >= '0' && character <= '9') ||
          (character >= 'a' && character <= 'f');
      });
    if (!valid_sha) {
      Rcpp::stop("target_keys must contain lowercase SHA-256 strings");
    }
    if (std::find(target_keys.begin(),
                  target_keys.begin() + index, key) !=
        target_keys.begin() + index) {
      Rcpp::stop("target_keys must not contain duplicates");
    }
  }

  fastkpc::FixedSpBatchHostView batch;
  batch.Y = REAL(Y_s);
  batch.SP = REAL(SP_s);
  batch.n = prepared_info.n;
  batch.null_dim = prepared_info.null_dim;
  batch.penalty_count = prepared_info.penalty_count;
  batch.target_count = target_count;
  batch.output_mask = fixed_sp_output_mask(outputs_s);
  batch.planned_routes = std::move(planned_routes);
  batch.target_keys = target_keys;

  SEXP ext = PROTECT(R_MakeExternalPtr(
    nullptr, fixed_sp_cuda_residual_tag(), R_NilValue));
  R_RegisterCFinalizerEx(ext, fixed_sp_cuda_residual_finalizer, TRUE);
  auto* holder = new FixedSpResidualHolder{
    std::shared_ptr<fastkpc::DeviceResidualBatch>(), fixed_sp_current_pid()
  };
  R_SetExternalPtrAddr(ext, holder);
  try {
    holder->value = fastkpc::solve_fixed_sp_batch(
      prepared_holder->value, batch);
  } catch (...) {
    delete holder;
    R_ClearExternalPtr(ext);
    UNPROTECT(1);
    throw;
  }
  fixed_sp_owner_acquire(holder, FixedSpOwnerKind::Residual);
  UNPROTECT(1);
  return ext;
  END_RCPP
}

extern "C" SEXP C_fixed_sp_cuda_residual_info(SEXP residual_s) {
  BEGIN_RCPP
  FixedSpResidualHolder* holder =
    fixed_sp_cuda_residual_holder(residual_s, true);
  const fastkpc::DeviceResidualInfo info =
    fastkpc::device_residual_info(holder->value);
  const std::size_t targets = static_cast<std::size_t>(info.target_count);
  const std::size_t aggregate_pivot_count = targets *
    static_cast<std::size_t>(info.null_dim);
  if (info.target_keys.size() != targets ||
      info.planned_routes.size() != targets ||
      info.executed_routes.size() != targets ||
      info.reroute_reasons.size() != targets ||
      info.solver_statuses.size() != targets ||
      info.qr_rank.size() != targets ||
      info.geqrf_info.size() != targets ||
      info.ormqr_info.size() != targets ||
      info.effective_rank.size() != targets ||
      info.sigma_max.size() != targets ||
      info.smallest_retained_sigma.size() != targets ||
      info.svd_info.size() != targets ||
      info.aggregate_penalty_root_rank.size() != targets ||
      info.aggregate_factor_call_count.size() != targets ||
      info.aggregate_b_build_count.size() != targets ||
      info.aggregate_penalty_root_pivot.size() != aggregate_pivot_count ||
      info.aggregate_dstop.size() != targets ||
      info.target_true_batched.size() != targets) {
    Rcpp::stop("fixed-sp residual diagnostics size mismatch");
  }

  Rcpp::CharacterVector planned_route(info.target_count);
  Rcpp::CharacterVector executed_route(info.target_count);
  Rcpp::CharacterVector reroute_reason(info.target_count);
  Rcpp::CharacterVector solver_status(info.target_count);
  Rcpp::LogicalVector target_true_batched(info.target_count);
  Rcpp::IntegerVector aggregate_penalty_root_rank(info.target_count);
  Rcpp::List aggregate_penalty_root_pivot(info.target_count);
  Rcpp::NumericVector aggregate_dstop(info.target_count);
  for (int index = 0; index < info.target_count; ++index) {
    const std::size_t offset = static_cast<std::size_t>(index);
    planned_route[index] =
      fastkpc::fixed_sp_route_name(info.planned_routes[offset]);
    if (info.executed_routes[offset] == fastkpc::FixedSpRoute::Unset) {
      executed_route[index] = NA_STRING;
    } else {
      executed_route[index] =
        fastkpc::fixed_sp_route_name(info.executed_routes[offset]);
    }
    reroute_reason[index] = info.reroute_reasons[offset];
    solver_status[index] =
      fastkpc::fixed_sp_status_name(info.solver_statuses[offset]);
    target_true_batched[index] = info.target_true_batched[offset];
    const bool executed_svd =
      info.executed_routes[offset] == fastkpc::FixedSpRoute::AugmentedSvd;
    if (!executed_svd) {
      aggregate_penalty_root_rank[index] = NA_INTEGER;
      aggregate_penalty_root_pivot[index] = Rcpp::IntegerVector(0);
      aggregate_dstop[index] = NA_REAL;
      continue;
    }
    aggregate_penalty_root_rank[index] =
      info.aggregate_penalty_root_rank[offset];
    aggregate_dstop[index] = info.aggregate_dstop[offset];
    Rcpp::IntegerVector pivot(info.null_dim);
    std::vector<bool> seen(static_cast<std::size_t>(info.null_dim), false);
    for (int column = 0; column < info.null_dim; ++column) {
      const int native_pivot = info.aggregate_penalty_root_pivot[
        offset * static_cast<std::size_t>(info.null_dim) +
        static_cast<std::size_t>(column)];
      if (native_pivot < 0 || native_pivot >= info.null_dim ||
          seen[static_cast<std::size_t>(native_pivot)]) {
        Rcpp::stop("fixed-sp aggregate pivot diagnostics are invalid");
      }
      seen[static_cast<std::size_t>(native_pivot)] = true;
      pivot[column] = native_pivot + 1;
    }
    aggregate_penalty_root_pivot[index] = pivot;
  }

  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("n") = info.n,
    Rcpp::Named("coefficient_dim") = info.coefficient_dim,
    Rcpp::Named("target_count") = info.target_count,
    Rcpp::Named("target_keys") = Rcpp::wrap(info.target_keys),
    Rcpp::Named("planned_route") = planned_route,
    Rcpp::Named("executed_route") = executed_route,
    Rcpp::Named("reroute_reason") = reroute_reason,
    Rcpp::Named("solver_status") = solver_status,
    Rcpp::Named("qr_rank") = Rcpp::wrap(info.qr_rank),
    Rcpp::Named("geqrf_info") = Rcpp::wrap(info.geqrf_info),
    Rcpp::Named("ormqr_info") = Rcpp::wrap(info.ormqr_info),
    Rcpp::Named("effective_rank") = Rcpp::wrap(info.effective_rank),
    Rcpp::Named("sigma_max") = Rcpp::wrap(info.sigma_max),
    Rcpp::Named("smallest_retained_sigma") =
      Rcpp::wrap(info.smallest_retained_sigma),
    Rcpp::Named("svd_info") = Rcpp::wrap(info.svd_info),
    Rcpp::Named("target_true_batched") = target_true_batched,
    Rcpp::Named("aggregate_penalty_root_rank") =
      aggregate_penalty_root_rank,
    Rcpp::Named("aggregate_penalty_root_pivot") =
      aggregate_penalty_root_pivot,
    Rcpp::Named("aggregate_factor_call_count") =
      Rcpp::wrap(info.aggregate_factor_call_count),
    Rcpp::Named("aggregate_b_build_count") =
      Rcpp::wrap(info.aggregate_b_build_count),
    Rcpp::Named("aggregate_dstop") = aggregate_dstop
  );
  result["native_batch_call"] = info.native_batch_call;
  result["batch_call_count"] = info.batch_call_count;
  result["true_batched_kernel"] = info.true_batched_kernel;
  result["true_batched_subgroup_count"] =
    info.true_batched_subgroup_count;
  result["true_batched_attempted_target_count"] =
    info.true_batched_attempted_target_count;
  result["true_batched_target_count"] = info.true_batched_target_count;
  result["cholesky_single_target_count"] =
    info.cholesky_single_target_count;
  result["potrf_batched_call_count"] = info.potrf_batched_call_count;
  result["potrs_batched_call_count"] = info.potrs_batched_call_count;
  result["target_batch_h2d_call_count"] =
    info.target_batch_h2d_call_count;
  result["target_h2d_copy_count"] = info.target_h2d_copy_count;
  result["target_h2d_bytes"] = static_cast<double>(info.target_h2d_bytes);
  result["coefficient_batch_finalize_call_count"] =
    info.coefficient_batch_finalize_call_count;
  result["fitted_batch_finalize_call_count"] =
    info.fitted_batch_finalize_call_count;
  result["residual_rss_batch_finalize_call_count"] =
    info.residual_rss_batch_finalize_call_count;
  result["per_target_output_finalize_call_count"] =
    info.per_target_output_finalize_call_count;
  result["batch_output_finalized_target_count"] =
    info.batch_output_finalized_target_count;
  result["canonical_output_order_exact"] =
    info.canonical_output_order_exact;
  result["stable_reroute_count"] = info.stable_reroute_count;
  result["planned_cholesky_target_count"] =
    info.planned_cholesky_target_count;
  result["planned_qr_target_count"] = info.planned_qr_target_count;
  result["planned_svd_target_count"] = info.planned_svd_target_count;
  result["executed_cholesky_target_count"] =
    info.executed_cholesky_target_count;
  result["executed_qr_target_count"] = info.executed_qr_target_count;
  result["executed_svd_target_count"] = info.executed_svd_target_count;
  result["cholesky_to_svd_count"] = info.cholesky_to_svd_count;
  result["qr_to_svd_count"] = info.qr_to_svd_count;
  result["aggregate_penalty_factor_count"] =
    info.aggregate_penalty_factor_count;
  result["aggregate_svd_b_build_count"] =
    info.aggregate_svd_b_build_count;
  result["aggregate_penalty_root_d2h_count"] =
    info.aggregate_penalty_root_d2h_count;
  result["aggregate_penalty_root_d2h_bytes"] =
    static_cast<double>(info.aggregate_penalty_root_d2h_bytes);
  result["output_slot_acquire_count"] = info.output_slot_acquire_count;
  result["output_slot_release_count"] = info.output_slot_release_count;
  result["output_slot_busy_count"] = info.output_slot_busy_count;
  result["stale_token_reject_count"] = info.stale_token_reject_count;
  result["invalid_output_init_count"] = info.invalid_output_init_count;
  result["nonfinite_output_count"] = info.nonfinite_output_count;
  result["cpu_fallback_count"] = info.cpu_fallback_count;
  result["unknown_fallback_count"] = info.unknown_fallback_count;
  result["resource_snapshot_captured"] = info.resource_snapshot_captured;
  result["resource_instrumentation_version"] =
    info.resource_instrumentation_version;
  result["resource_allocation_count_before_solve"] =
    info.resource_allocation_count_before_solve;
  result["resource_allocation_count_after_solve"] =
    info.resource_allocation_count_after_solve;
  result["resource_handle_create_count_before_solve"] =
    info.resource_handle_create_count_before_solve;
  result["resource_handle_create_count_after_solve"] =
    info.resource_handle_create_count_after_solve;
  result["cuda_device_allocation_count_during_solve"] =
    info.cuda_device_allocation_count_during_solve;
  result["cuda_host_allocation_count_during_solve"] =
    info.cuda_host_allocation_count_during_solve;
  result["stream_create_count_during_solve"] =
    info.stream_create_count_during_solve;
  result["event_create_count_during_solve"] =
    info.event_create_count_during_solve;
  result["cublas_handle_create_count_during_solve"] =
    info.cublas_handle_create_count_during_solve;
  result["cusolver_handle_create_count_during_solve"] =
    info.cusolver_handle_create_count_during_solve;
  result["per_target_allocation_count_after_warmup"] =
    info.per_target_allocation_count_after_warmup;
  result["per_target_handle_create_count"] =
    info.per_target_handle_create_count;
  result["implicit_residual_d2h_count"] =
    info.implicit_residual_d2h_count;
  result["rhs_device_build_count"] = info.rhs_device_build_count;
  result["rhs_authority"] = info.rhs_authority;
  result["full_cuda_data_plane"] = info.full_cuda_data_plane;
  result["shadow_materialize_call_count"] =
    info.shadow_materialize_call_count;
  result["shadow_materialize_target_count"] =
    info.shadow_materialize_target_count;
  result["shadow_d2h_bytes"] = static_cast<double>(info.shadow_d2h_bytes);
  result["owner_generation"] = static_cast<double>(info.owner_generation);
  result["slot_generation"] = static_cast<double>(info.slot_generation);
  return result;
  END_RCPP
}

extern "C" SEXP C_fixed_sp_cuda_materialize_shadow(
    SEXP residual_s,
    SEXP outputs_s) {
  BEGIN_RCPP
  FixedSpResidualHolder* holder =
    fixed_sp_cuda_residual_holder(residual_s, true);
  const std::uint32_t output_mask = fixed_sp_output_mask(outputs_s);
  const fastkpc::FixedSpShadowResult shadow =
    fastkpc::materialize_fixed_sp_shadow(holder->value, output_mask);
  const std::size_t targets = static_cast<std::size_t>(shadow.target_count);
  if (shadow.successful_targets.size() != targets) {
    Rcpp::stop("fixed-sp shadow success metadata size mismatch");
  }

  auto matrix_from_shadow = [&](const std::vector<double>& values,
                                int rows,
                                const char* name) {
    const std::size_t expected =
      static_cast<std::size_t>(rows) * targets;
    if (values.size() != expected) {
      Rcpp::stop(std::string(name) + " shadow size mismatch");
    }
    Rcpp::NumericMatrix matrix(rows, shadow.target_count);
    std::fill(matrix.begin(), matrix.end(), NA_REAL);
    for (std::size_t target = 0; target < targets; ++target) {
      if (shadow.successful_targets[target] == 0U) continue;
      const std::size_t offset = static_cast<std::size_t>(rows) * target;
      std::copy(values.begin() + offset,
                values.begin() + offset + static_cast<std::size_t>(rows),
                matrix.begin() + offset);
    }
    return matrix;
  };

  Rcpp::List result;
  if ((output_mask & fastkpc::FixedSpOutputCoefficients) != 0U) {
    result.push_back(matrix_from_shadow(
      shadow.coefficients, shadow.coefficient_dim, "coefficient"
    ), "coefficients");
  }
  if ((output_mask & fastkpc::FixedSpOutputFitted) != 0U) {
    result.push_back(matrix_from_shadow(
      shadow.fitted, shadow.n, "fitted"
    ), "fitted");
  }
  if ((output_mask & fastkpc::FixedSpOutputResiduals) != 0U) {
    result.push_back(matrix_from_shadow(
      shadow.residuals, shadow.n, "residual"
    ), "residuals");
  }
  if ((output_mask & fastkpc::FixedSpOutputRss) != 0U) {
    if (shadow.rss.size() != targets) {
      Rcpp::stop("RSS shadow size mismatch");
    }
    Rcpp::NumericVector rss(shadow.target_count, NA_REAL);
    for (std::size_t target = 0; target < targets; ++target) {
      if (shadow.successful_targets[target] != 0U) {
        rss[static_cast<R_xlen_t>(target)] = shadow.rss[target];
      }
    }
    result.push_back(rss, "rss");
  }
  if ((output_mask & fastkpc::FixedSpOutputRhs) != 0U) {
    result.push_back(matrix_from_shadow(
      shadow.cuda_nullspace_rhs, shadow.null_dim, "RHS"
    ), "cuda_nullspace_rhs");
  }
  return result;
  END_RCPP
}

extern "C" SEXP C_fixed_sp_cuda_residual_release(SEXP residual_s) {
  BEGIN_RCPP
  FixedSpResidualHolder* holder =
    fixed_sp_cuda_residual_holder(residual_s, true);
  fastkpc::release_device_residual(holder->value);
  return R_NilValue;
  END_RCPP
}

extern "C" SEXP C_fixed_sp_cuda_residual_free(SEXP residual_s) {
  BEGIN_RCPP
  FixedSpResidualHolder* holder =
    fixed_sp_cuda_residual_holder(residual_s, false);
  if (holder == nullptr) return R_NilValue;
  fastkpc::free_device_residual(&holder->value);
  fixed_sp_owner_release(holder, FixedSpOwnerKind::Residual);
  delete holder;
  R_ClearExternalPtr(residual_s);
  return R_NilValue;
  END_RCPP
}

extern "C" SEXP C_fixed_sp_cuda_test_coefficient_shadow(SEXP residual_s) {
  BEGIN_RCPP
  FixedSpResidualHolder* holder =
    fixed_sp_cuda_residual_holder(residual_s, true);
  const fastkpc::DeviceCoefficientShadow shadow =
    fastkpc::test_device_residual_coefficient_shadow(holder->value);
  Rcpp::NumericMatrix coefficients(
    shadow.coefficient_dim, shadow.target_count);
  std::copy(shadow.coefficients.begin(), shadow.coefficients.end(),
            coefficients.begin());
  return coefficients;
  END_RCPP
}

extern "C" SEXP
C_fixed_sp_cuda_test_inject_consumer_registration_failure(SEXP residual_s) {
  BEGIN_RCPP
  FixedSpResidualHolder* holder =
    fixed_sp_cuda_residual_holder(residual_s, true);
  fastkpc::test_inject_device_residual_consumer_registration_failure(
    holder->value);
  return R_NilValue;
  END_RCPP
}

extern "C" SEXP C_fixed_sp_cuda_test_register_blocked_consumer(
    SEXP residual_s) {
  BEGIN_RCPP
  FixedSpResidualHolder* holder =
    fixed_sp_cuda_residual_holder(residual_s, true);
  fastkpc::test_register_blocked_device_residual_consumer(holder->value);
  return R_NilValue;
  END_RCPP
}

extern "C" SEXP C_fixed_sp_cuda_test_complete_consumer(SEXP residual_s) {
  BEGIN_RCPP
  FixedSpResidualHolder* holder =
    fixed_sp_cuda_residual_holder(residual_s, true);
  fastkpc::test_complete_blocked_device_residual_consumer(holder->value);
  return R_NilValue;
  END_RCPP
}

extern "C" SEXP C_legacy_dcov_spectra_matvec_cuda_handle_apply(
    SEXP handles,
    SEXP rhss) {
  BEGIN_RCPP
  if (!Rf_isReal(rhss) || !Rf_isMatrix(rhss)) {
    Rcpp::stop("rhs must be a numeric matrix");
  }
  fastkpc::LegacyDcovSpectraMatvecCudaHandle* handle =
    legacy_dcov_spectra_matvec_handle_from_externalptr(handles);
  const int n = fastkpc::legacy_dcov_spectra_matvec_cuda_handle_n(handle);
  Rcpp::NumericMatrix rhs(rhss);
  if (rhs.nrow() != n) {
    Rcpp::stop("rhs row count must match matrix dimension");
  }
  if (rhs.ncol() < 1) {
    Rcpp::stop("rhs must have at least one column");
  }
  if (!all_finite(rhs)) {
    Rcpp::stop("Data contains missing or infinite values");
  }

  const int rhs_count = rhs.ncol();
  const fastkpc::LegacyDcovSpectraMatvecCudaResult result =
    fastkpc::legacy_dcov_spectra_matvec_cuda_handle_apply(
      handle, REAL(rhss), rhs_count);
  Rcpp::NumericMatrix values(n, rhs_count);
  std::copy(result.values.begin(), result.values.end(), values.begin());
  return Rcpp::List::create(
    Rcpp::Named("values") = values,
    Rcpp::Named("backend") = "cuda-dense-sym-matvec-handle",
    Rcpp::Named("n") = result.n,
    Rcpp::Named("rhs_count") = result.rhs_count,
    Rcpp::Named("kernel_launch_count") = result.kernel_launch_count,
    Rcpp::Named("device_matrix_reuse_count") =
      result.device_matrix_reuse_count,
    Rcpp::Named("device_workspace_reuse_count") =
      result.device_workspace_reuse_count,
    Rcpp::Named("workspace_realloc_count") =
      result.workspace_realloc_count,
    Rcpp::Named("matrix_bytes") = static_cast<double>(result.matrix_bytes),
    Rcpp::Named("workspace_bytes") =
      static_cast<double>(result.workspace_bytes),
    Rcpp::Named("matrix_h2d_ms") = result.matrix_h2d_ms,
    Rcpp::Named("workspace_alloc_ms") = result.workspace_alloc_ms,
    Rcpp::Named("alloc_ms") = result.alloc_ms,
    Rcpp::Named("h2d_ms") = result.h2d_ms,
    Rcpp::Named("kernel_ms") = result.kernel_ms,
    Rcpp::Named("d2h_ms") = result.d2h_ms,
    Rcpp::Named("free_ms") = result.free_ms,
    Rcpp::Named("total_ms") = result.total_ms
  );
  END_RCPP
}

extern "C" SEXP C_legacy_dcov_spectra_matvec_cuda_handle_project(
    SEXP handles,
    SEXP basiss) {
  BEGIN_RCPP
  if (!Rf_isReal(basiss) || !Rf_isMatrix(basiss)) {
    Rcpp::stop("basis must be a numeric matrix");
  }
  fastkpc::LegacyDcovSpectraMatvecCudaHandle* handle =
    legacy_dcov_spectra_matvec_handle_from_externalptr(handles);
  const int n = fastkpc::legacy_dcov_spectra_matvec_cuda_handle_n(handle);
  Rcpp::NumericMatrix basis(basiss);
  if (basis.nrow() != n) {
    Rcpp::stop("basis row count must match matrix dimension");
  }
  if (basis.ncol() < 1) {
    Rcpp::stop("basis must have at least one column");
  }
  if (!all_finite(basis)) {
    Rcpp::stop("Data contains missing or infinite values");
  }

  const int basis_count = basis.ncol();
  const fastkpc::LegacyDcovSpectraMatvecCudaResult result =
    fastkpc::legacy_dcov_spectra_matvec_cuda_handle_project(
      handle, REAL(basiss), basis_count);
  Rcpp::NumericMatrix values(basis_count, basis_count);
  std::copy(result.values.begin(), result.values.end(), values.begin());
  return Rcpp::List::create(
    Rcpp::Named("values") = values,
    Rcpp::Named("backend") = "cuda-dense-sym-matvec-handle-projection",
    Rcpp::Named("n") = result.n,
    Rcpp::Named("basis_count") = result.rhs_count,
    Rcpp::Named("kernel_launch_count") = result.kernel_launch_count,
    Rcpp::Named("device_matrix_reuse_count") =
      result.device_matrix_reuse_count,
    Rcpp::Named("device_workspace_reuse_count") =
      result.device_workspace_reuse_count,
    Rcpp::Named("workspace_realloc_count") =
      result.workspace_realloc_count,
    Rcpp::Named("matrix_bytes") = static_cast<double>(result.matrix_bytes),
    Rcpp::Named("workspace_bytes") =
      static_cast<double>(result.workspace_bytes),
    Rcpp::Named("d2h_bytes") = static_cast<double>(result.d2h_bytes),
    Rcpp::Named("matrix_h2d_ms") = result.matrix_h2d_ms,
    Rcpp::Named("workspace_alloc_ms") = result.workspace_alloc_ms,
    Rcpp::Named("alloc_ms") = result.alloc_ms,
    Rcpp::Named("h2d_ms") = result.h2d_ms,
    Rcpp::Named("kernel_ms") = result.kernel_ms,
    Rcpp::Named("d2h_ms") = result.d2h_ms,
    Rcpp::Named("free_ms") = result.free_ms,
    Rcpp::Named("total_ms") = result.total_ms
  );
  END_RCPP
}

extern "C" SEXP C_legacy_dcov_spectra_matvec_cuda_handle_apply_sequence(
    SEXP handles,
    SEXP rhss) {
  BEGIN_RCPP
  if (!Rf_isReal(rhss) || !Rf_isMatrix(rhss)) {
    Rcpp::stop("rhs must be a numeric matrix");
  }
  fastkpc::LegacyDcovSpectraMatvecCudaHandle* handle =
    legacy_dcov_spectra_matvec_handle_from_externalptr(handles);
  const int n = fastkpc::legacy_dcov_spectra_matvec_cuda_handle_n(handle);
  Rcpp::NumericMatrix rhs(rhss);
  if (rhs.nrow() != n) {
    Rcpp::stop("rhs row count must match matrix dimension");
  }
  if (rhs.ncol() < 1) {
    Rcpp::stop("rhs must have at least one column");
  }
  if (!all_finite(rhs)) {
    Rcpp::stop("Data contains missing or infinite values");
  }

  const auto total_start = std::chrono::steady_clock::now();
  const int rhs_count = rhs.ncol();
  Rcpp::NumericMatrix values(n, rhs_count);
  int kernel_launch_count = 0;
  int device_matrix_reuse_count = 0;
  int device_workspace_reuse_count = 0;
  int workspace_realloc_count = 0;
  std::size_t matrix_bytes = 0;
  std::size_t workspace_bytes = 0;
  double matrix_h2d_ms = 0.0;
  double workspace_alloc_ms = 0.0;
  double alloc_ms = 0.0;
  double h2d_ms = 0.0;
  double kernel_ms = 0.0;
  double d2h_ms = 0.0;
  double free_ms = 0.0;

  const double* rhs_data = REAL(rhss);
  for (int col = 0; col < rhs_count; ++col) {
    const double* rhs_col = rhs_data + static_cast<std::size_t>(col) * n;
    const fastkpc::LegacyDcovSpectraMatvecCudaResult result =
      fastkpc::legacy_dcov_spectra_matvec_cuda_handle_apply(
        handle, rhs_col, 1);
    std::copy(result.values.begin(), result.values.end(),
              values.begin() + static_cast<std::size_t>(col) * n);
    kernel_launch_count += result.kernel_launch_count;
    device_matrix_reuse_count += result.device_matrix_reuse_count;
    device_workspace_reuse_count += result.device_workspace_reuse_count;
    workspace_realloc_count += result.workspace_realloc_count;
    matrix_bytes = result.matrix_bytes;
    workspace_bytes = std::max(workspace_bytes, result.workspace_bytes);
    matrix_h2d_ms += result.matrix_h2d_ms;
    workspace_alloc_ms += result.workspace_alloc_ms;
    alloc_ms += result.alloc_ms;
    h2d_ms += result.h2d_ms;
    kernel_ms += result.kernel_ms;
    d2h_ms += result.d2h_ms;
    free_ms += result.free_ms;
  }

  return Rcpp::List::create(
    Rcpp::Named("values") = values,
    Rcpp::Named("backend") = "cuda-dense-sym-matvec-handle-sequence",
    Rcpp::Named("n") = n,
    Rcpp::Named("rhs_count") = rhs_count,
    Rcpp::Named("matvec_call_count") = rhs_count,
    Rcpp::Named("kernel_launch_count") = kernel_launch_count,
    Rcpp::Named("device_matrix_reuse_count") = device_matrix_reuse_count,
    Rcpp::Named("device_workspace_reuse_count") =
      device_workspace_reuse_count,
    Rcpp::Named("workspace_realloc_count") =
      workspace_realloc_count,
    Rcpp::Named("matrix_bytes") = static_cast<double>(matrix_bytes),
    Rcpp::Named("workspace_bytes") =
      static_cast<double>(workspace_bytes),
    Rcpp::Named("matrix_h2d_ms") = matrix_h2d_ms,
    Rcpp::Named("workspace_alloc_ms") = workspace_alloc_ms,
    Rcpp::Named("alloc_ms") = alloc_ms,
    Rcpp::Named("h2d_ms") = h2d_ms,
    Rcpp::Named("kernel_ms") = kernel_ms,
    Rcpp::Named("d2h_ms") = d2h_ms,
    Rcpp::Named("free_ms") = free_ms,
    Rcpp::Named("total_ms") = elapsed_ms_since(total_start)
  );
  END_RCPP
}

extern "C" SEXP C_legacy_dcov_spectra_matvec_cuda_operator_eigs(
    SEXP matrixs,
    SEXP nevs,
    SEXP ncvs,
    SEXP tols,
    SEXP maxitrs) {
  BEGIN_RCPP
  if (!Rf_isReal(matrixs) || !Rf_isMatrix(matrixs)) {
    Rcpp::stop("matrix must be a numeric matrix");
  }
  Rcpp::NumericMatrix matrix(matrixs);
  if (matrix.nrow() != matrix.ncol()) {
    Rcpp::stop("matrix must be square");
  }
  if (!all_finite(matrix)) {
    Rcpp::stop("Data contains missing or infinite values");
  }
  const int n = matrix.nrow();
  const int nev = Rcpp::as<int>(nevs);
  const int ncv = Rcpp::as<int>(ncvs);
  const double tol = Rcpp::as<double>(tols);
  const int maxitr = Rcpp::as<int>(maxitrs);
  if (nev < 1 || nev >= n) {
    Rcpp::stop("nev must be positive and smaller than matrix dimension");
  }
  if (ncv <= nev || ncv > n) {
    Rcpp::stop("ncv must be greater than nev and no larger than matrix dimension");
  }
  if (!std::isfinite(tol) || tol <= 0.0) {
    Rcpp::stop("tol must be a positive finite value");
  }
  if (maxitr <= 0) {
    Rcpp::stop("maxitr must be positive");
  }

  fastkpc::LegacyDcovSpectraMatvecCudaHandle* handle =
    fastkpc::legacy_dcov_spectra_matvec_cuda_handle_create(
      REAL(matrixs), n);
  const double matrix_h2d_ms =
    fastkpc::legacy_dcov_spectra_matvec_cuda_handle_matrix_h2d_ms(handle);
  const double matrix_bytes = static_cast<double>(
    fastkpc::legacy_dcov_spectra_matvec_cuda_handle_matrix_bytes(handle));

  try {
    LegacyDcovSpectraCudaOperatorDiagnostics diagnostics;
    diagnostics.matrix_bytes = static_cast<std::size_t>(matrix_bytes);
    LegacyDcovSpectraCudaMatProd op(handle, n, &diagnostics);
    const auto eig_start = std::chrono::steady_clock::now();
    Spectra::SymEigsSolver<double, Spectra::LARGEST_MAGN,
                           LegacyDcovSpectraCudaMatProd> eigs(
      &op, nev, ncv);
    eigs.init();
    const int nconv = static_cast<int>(eigs.compute(
      maxitr, tol, Spectra::LARGEST_MAGN));
    const int iterations = static_cast<int>(eigs.num_iterations());
    const bool converged =
      eigs.info() == Spectra::SUCCESSFUL && nconv >= nev;
    const double eig_ms = elapsed_ms_since(eig_start);

    const Eigen::VectorXd eigenvalues = eigs.eigenvalues();
    const Eigen::MatrixXd eigenvectors = eigs.eigenvectors();
    Rcpp::NumericVector values(eigenvalues.size());
    for (int i = 0; i < eigenvalues.size(); ++i) values[i] = eigenvalues[i];
    Rcpp::NumericMatrix vectors(eigenvectors.rows(), eigenvectors.cols());
    for (int col = 0; col < eigenvectors.cols(); ++col) {
      for (int row = 0; row < eigenvectors.rows(); ++row) {
        vectors(row, col) = eigenvectors(row, col);
      }
    }

    fastkpc::legacy_dcov_spectra_matvec_cuda_handle_destroy(handle);
    handle = nullptr;
    return Rcpp::List::create(
      Rcpp::Named("values") = values,
      Rcpp::Named("vectors") = vectors,
      Rcpp::Named("backend") = "cuda-dense-sym-matvec-spectra-operator",
      Rcpp::Named("n") = n,
      Rcpp::Named("nev") = nev,
      Rcpp::Named("ncv") = ncv,
      Rcpp::Named("tol") = tol,
      Rcpp::Named("maxitr") = maxitr,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("nconv") = nconv,
      Rcpp::Named("iterations") = iterations,
      Rcpp::Named("info") = static_cast<int>(eigs.info()),
      Rcpp::Named("eig_ms") = eig_ms,
      Rcpp::Named("spectra_matvec_count") =
        diagnostics.spectra_matvec_count,
      Rcpp::Named("spectra_matvec_ms") =
        diagnostics.spectra_matvec_ms,
      Rcpp::Named("kernel_launch_count") =
        diagnostics.kernel_launch_count,
      Rcpp::Named("device_matrix_reuse_count") =
        diagnostics.device_matrix_reuse_count,
      Rcpp::Named("device_workspace_reuse_count") =
        diagnostics.device_workspace_reuse_count,
      Rcpp::Named("workspace_realloc_count") =
        diagnostics.workspace_realloc_count,
      Rcpp::Named("matrix_bytes") = matrix_bytes,
      Rcpp::Named("workspace_bytes") =
        static_cast<double>(diagnostics.workspace_bytes),
      Rcpp::Named("matrix_h2d_ms") = matrix_h2d_ms,
      Rcpp::Named("matrix_h2d_ms_during_compute") =
        diagnostics.matrix_h2d_ms_during_compute,
      Rcpp::Named("workspace_alloc_ms") =
        diagnostics.workspace_alloc_ms,
      Rcpp::Named("h2d_ms") = diagnostics.h2d_ms,
      Rcpp::Named("kernel_ms") = diagnostics.kernel_ms,
      Rcpp::Named("d2h_ms") = diagnostics.d2h_ms,
      Rcpp::Named("total_ms") = diagnostics.total_ms
    );
  } catch (...) {
    if (handle != nullptr) {
      fastkpc::legacy_dcov_spectra_matvec_cuda_handle_destroy(handle);
    }
    throw;
  }
  END_RCPP
}

extern "C" SEXP C_legacy_dcov_spectra_matvec_cuda_lowrank_shadow(
    SEXP xs,
    SEXP ys,
    SEXP numCols,
    SEXP ncvs,
    SEXP tols,
    SEXP maxitrs) {
  BEGIN_RCPP
  if (!Rf_isReal(xs) || !Rf_isReal(ys)) {
    Rcpp::stop("x and y must be numeric vectors");
  }
  Rcpp::NumericVector x(xs);
  Rcpp::NumericVector y(ys);
  if (x.size() != y.size()) Rcpp::stop("Sample sizes must agree");
  const int n = x.size();
  if (n <= 5) {
    Rcpp::stop("legacy dCov gamma lowrank shadow requires n > 5");
  }
  if (!all_finite_vector(x) || !all_finite_vector(y)) {
    Rcpp::stop("Data contains missing or infinite values");
  }
  const int num_col = Rcpp::as<int>(numCols);
  const int ncv = Rcpp::as<int>(ncvs);
  const double tol = Rcpp::as<double>(tols);
  const int maxitr = Rcpp::as<int>(maxitrs);
  if (num_col <= 0 || num_col >= n) {
    Rcpp::stop("numCol must be positive and less than sample size");
  }
  if (ncv <= num_col || ncv > n) {
    Rcpp::stop("ncv must be greater than numCol and no larger than sample size");
  }
  if (!std::isfinite(tol) || tol <= 0.0) {
    Rcpp::stop("tol must be a positive finite value");
  }
  if (maxitr <= 0) Rcpp::stop("maxitr must be positive");

  const Eigen::MatrixXd x_distance =
    legacy_dcov_distance_matrix_eigen(REAL(xs), n);
  const Eigen::MatrixXd y_distance =
    legacy_dcov_distance_matrix_eigen(REAL(ys), n);

  const LegacyDcovSpectraLowrankShadowRun cpu_x =
    legacy_dcov_cpu_spectra_lowrank_shadow(
      x_distance, num_col, ncv, tol, maxitr);
  const LegacyDcovSpectraLowrankShadowRun cpu_y =
    legacy_dcov_cpu_spectra_lowrank_shadow(
      y_distance, num_col, ncv, tol, maxitr);
  const LegacyDcovSpectraLowrankShadowRun cuda_x =
    legacy_dcov_cuda_spectra_lowrank_shadow(
      x_distance, num_col, ncv, tol, maxitr);
  const LegacyDcovSpectraLowrankShadowRun cuda_y =
    legacy_dcov_cuda_spectra_lowrank_shadow(
      y_distance, num_col, ncv, tol, maxitr);

  double max_abs_eigenvalue_diff_x = NA_REAL;
  double max_abs_eigenvalue_diff_y = NA_REAL;
  double min_centered_abs_corr_x = NA_REAL;
  double min_centered_abs_corr_y = NA_REAL;
  double cpu_nV2 = NA_REAL;
  double cuda_nV2 = NA_REAL;
  double cpu_x_moment = NA_REAL;
  double cuda_x_moment = NA_REAL;
  double cpu_y_moment = NA_REAL;
  double cuda_y_moment = NA_REAL;
  double nV2_abs_diff = NA_REAL;
  double x_moment_abs_diff = NA_REAL;
  double y_moment_abs_diff = NA_REAL;
  double statistic_input_max_abs_diff = NA_REAL;

  const bool all_converged = cpu_x.converged && cpu_y.converged &&
    cuda_x.converged && cuda_y.converged;
  if (all_converged) {
    max_abs_eigenvalue_diff_x =
      legacy_dcov_max_abs_vector_diff(cpu_x.eigenvalues,
                                      cuda_x.eigenvalues);
    max_abs_eigenvalue_diff_y =
      legacy_dcov_max_abs_vector_diff(cpu_y.eigenvalues,
                                      cuda_y.eigenvalues);
    min_centered_abs_corr_x =
      legacy_dcov_min_centered_abs_corr(cpu_x.centered_vectors,
                                        cuda_x.centered_vectors);
    min_centered_abs_corr_y =
      legacy_dcov_min_centered_abs_corr(cpu_y.centered_vectors,
                                        cuda_y.centered_vectors);

    cpu_nV2 = legacy_dcov_weighted_cross_sum_eigen(
      cpu_x.centered_vectors, cpu_x.eigenvalues,
      cpu_y.centered_vectors, cpu_y.eigenvalues) /
      static_cast<double>(n);
    cuda_nV2 = legacy_dcov_weighted_cross_sum_eigen(
      cuda_x.centered_vectors, cuda_x.eigenvalues,
      cuda_y.centered_vectors, cuda_y.eigenvalues) /
      static_cast<double>(n);
    cpu_x_moment = legacy_dcov_weighted_cross_sum_eigen(
      cpu_x.centered_vectors, cpu_x.eigenvalues,
      cpu_x.centered_vectors, cpu_x.eigenvalues);
    cuda_x_moment = legacy_dcov_weighted_cross_sum_eigen(
      cuda_x.centered_vectors, cuda_x.eigenvalues,
      cuda_x.centered_vectors, cuda_x.eigenvalues);
    cpu_y_moment = legacy_dcov_weighted_cross_sum_eigen(
      cpu_y.centered_vectors, cpu_y.eigenvalues,
      cpu_y.centered_vectors, cpu_y.eigenvalues);
    cuda_y_moment = legacy_dcov_weighted_cross_sum_eigen(
      cuda_y.centered_vectors, cuda_y.eigenvalues,
      cuda_y.centered_vectors, cuda_y.eigenvalues);
    nV2_abs_diff = std::abs(cpu_nV2 - cuda_nV2);
    x_moment_abs_diff = std::abs(cpu_x_moment - cuda_x_moment);
    y_moment_abs_diff = std::abs(cpu_y_moment - cuda_y_moment);
    statistic_input_max_abs_diff = std::max(
      nV2_abs_diff, std::max(x_moment_abs_diff, y_moment_abs_diff));
  }

  const LegacyDcovSpectraCudaOperatorDiagnostics& dx =
    cuda_x.cuda_diagnostics;
  const LegacyDcovSpectraCudaOperatorDiagnostics& dy =
    cuda_y.cuda_diagnostics;

  return Rcpp::List::create(
    Rcpp::Named("backend") =
      "cuda-dense-sym-matvec-spectra-lowrank-shadow",
    Rcpp::Named("n") = n,
    Rcpp::Named("numCol") = num_col,
    Rcpp::Named("ncv") = ncv,
    Rcpp::Named("tol") = tol,
    Rcpp::Named("maxitr") = maxitr,
    Rcpp::Named("cpu_converged_x") = cpu_x.converged,
    Rcpp::Named("cpu_converged_y") = cpu_y.converged,
    Rcpp::Named("cuda_converged_x") = cuda_x.converged,
    Rcpp::Named("cuda_converged_y") = cuda_y.converged,
    Rcpp::Named("cpu_nconv_x") = cpu_x.nconv,
    Rcpp::Named("cpu_nconv_y") = cpu_y.nconv,
    Rcpp::Named("cuda_nconv_x") = cuda_x.nconv,
    Rcpp::Named("cuda_nconv_y") = cuda_y.nconv,
    Rcpp::Named("cpu_iterations_x") = cpu_x.iterations,
    Rcpp::Named("cpu_iterations_y") = cpu_y.iterations,
    Rcpp::Named("cuda_iterations_x") = cuda_x.iterations,
    Rcpp::Named("cuda_iterations_y") = cuda_y.iterations,
    Rcpp::Named("cpu_info_x") = cpu_x.info,
    Rcpp::Named("cpu_info_y") = cpu_y.info,
    Rcpp::Named("cuda_info_x") = cuda_x.info,
    Rcpp::Named("cuda_info_y") = cuda_y.info,
    Rcpp::Named("max_abs_eigenvalue_diff_x") =
      max_abs_eigenvalue_diff_x,
    Rcpp::Named("max_abs_eigenvalue_diff_y") =
      max_abs_eigenvalue_diff_y,
    Rcpp::Named("min_centered_abs_corr_x") = min_centered_abs_corr_x,
    Rcpp::Named("min_centered_abs_corr_y") = min_centered_abs_corr_y,
    Rcpp::Named("cpu_nV2") = cpu_nV2,
    Rcpp::Named("cuda_nV2") = cuda_nV2,
    Rcpp::Named("nV2_abs_diff") = nV2_abs_diff,
    Rcpp::Named("cpu_x_moment") = cpu_x_moment,
    Rcpp::Named("cuda_x_moment") = cuda_x_moment,
    Rcpp::Named("x_moment_abs_diff") = x_moment_abs_diff,
    Rcpp::Named("cpu_y_moment") = cpu_y_moment,
    Rcpp::Named("cuda_y_moment") = cuda_y_moment,
    Rcpp::Named("y_moment_abs_diff") = y_moment_abs_diff,
    Rcpp::Named("statistic_input_max_abs_diff") =
      statistic_input_max_abs_diff,
    Rcpp::Named("cpu_eig_ms") = cpu_x.eig_ms + cpu_y.eig_ms,
    Rcpp::Named("cuda_eig_ms") = cuda_x.eig_ms + cuda_y.eig_ms,
    Rcpp::Named("spectra_matvec_count") =
      dx.spectra_matvec_count + dy.spectra_matvec_count,
    Rcpp::Named("spectra_matvec_ms") =
      dx.spectra_matvec_ms + dy.spectra_matvec_ms,
    Rcpp::Named("kernel_launch_count") =
      dx.kernel_launch_count + dy.kernel_launch_count,
    Rcpp::Named("device_matrix_reuse_count") =
      dx.device_matrix_reuse_count + dy.device_matrix_reuse_count,
    Rcpp::Named("device_workspace_reuse_count") =
      dx.device_workspace_reuse_count + dy.device_workspace_reuse_count,
    Rcpp::Named("workspace_realloc_count") =
      dx.workspace_realloc_count + dy.workspace_realloc_count,
    Rcpp::Named("matrix_bytes") =
      cuda_x.matrix_bytes + cuda_y.matrix_bytes,
    Rcpp::Named("workspace_bytes") = static_cast<double>(
      std::max(dx.workspace_bytes, dy.workspace_bytes)),
    Rcpp::Named("matrix_h2d_ms") =
      cuda_x.matrix_h2d_ms + cuda_y.matrix_h2d_ms,
    Rcpp::Named("matrix_h2d_ms_during_compute") =
      dx.matrix_h2d_ms_during_compute +
      dy.matrix_h2d_ms_during_compute,
    Rcpp::Named("workspace_alloc_ms") =
      dx.workspace_alloc_ms + dy.workspace_alloc_ms,
    Rcpp::Named("h2d_ms") = dx.h2d_ms + dy.h2d_ms,
    Rcpp::Named("kernel_ms") = dx.kernel_ms + dy.kernel_ms,
    Rcpp::Named("d2h_ms") = dx.d2h_ms + dy.d2h_ms,
    Rcpp::Named("total_ms") = dx.total_ms + dy.total_ms
  );
  END_RCPP
}

extern "C" SEXP C_legacy_dcov_spectra_matvec_cuda_lowrank_gamma(
    SEXP xs,
    SEXP ys,
    SEXP numCols,
    SEXP indexs,
    SEXP ncvs,
    SEXP tols,
    SEXP maxitrs) {
  BEGIN_RCPP
  if (!Rf_isReal(xs) || !Rf_isReal(ys)) {
    Rcpp::stop("x and y must be numeric vectors");
  }
  Rcpp::NumericVector x(xs);
  Rcpp::NumericVector y(ys);
  if (x.size() != y.size()) Rcpp::stop("Sample sizes must agree");
  const int n = x.size();
  if (n <= 5) {
    Rcpp::stop("legacy dCov gamma lowrank CUDA requires n > 5");
  }
  if (!all_finite_vector(x) || !all_finite_vector(y)) {
    Rcpp::stop("Data contains missing or infinite values");
  }
  const int num_col = Rcpp::as<int>(numCols);
  double index = Rcpp::as<double>(indexs);
  const int ncv = Rcpp::as<int>(ncvs);
  const double tol = Rcpp::as<double>(tols);
  const int maxitr = Rcpp::as<int>(maxitrs);
  if (num_col <= 0 || num_col >= n) {
    Rcpp::stop("numCol must be positive and less than sample size");
  }
  if (index < 0.0 || index > 2.0) index = 1.0;
  if (ncv <= num_col || ncv > n) {
    Rcpp::stop("ncv must be greater than numCol and no larger than sample size");
  }
  if (!std::isfinite(tol) || tol <= 0.0) {
    Rcpp::stop("tol must be a positive finite value");
  }
  if (maxitr <= 0) Rcpp::stop("maxitr must be positive");

  const LegacyDcovCudaLowrankGammaRun run =
    legacy_dcov_cuda_lowrank_gamma_compute(
      REAL(xs), REAL(ys), n, num_col, index, ncv, tol, maxitr);
  return legacy_dcov_cuda_lowrank_gamma_run_to_list(
    run, n, num_col, index, ncv, tol, maxitr);
  END_RCPP
}

extern "C" SEXP C_legacy_dcov_spectra_matvec_cuda_lowrank_gamma_batch(
    SEXP xs,
    SEXP ys,
    SEXP numCols,
    SEXP indexs,
    SEXP ncvs,
    SEXP tols,
    SEXP maxitrs) {
  BEGIN_RCPP
  if (!Rf_isReal(xs) || !Rf_isReal(ys) ||
      !Rf_isMatrix(xs) || !Rf_isMatrix(ys)) {
    Rcpp::stop("x and y must be numeric matrices");
  }
  Rcpp::NumericMatrix x(xs);
  Rcpp::NumericMatrix y(ys);
  if (x.nrow() != y.nrow() || x.ncol() != y.ncol()) {
    Rcpp::stop("x and y must have identical dimensions");
  }
  const int n = x.nrow();
  const int batch = x.ncol();
  if (n <= 5) {
    Rcpp::stop("legacy dCov gamma lowrank CUDA requires n > 5");
  }
  if (batch < 1) {
    Rcpp::stop("batch must contain at least one pair");
  }
  if (!all_finite(x) || !all_finite(y)) {
    Rcpp::stop("Data contains missing or infinite values");
  }
  const int num_col = Rcpp::as<int>(numCols);
  double index = Rcpp::as<double>(indexs);
  const int ncv = Rcpp::as<int>(ncvs);
  const double tol = Rcpp::as<double>(tols);
  const int maxitr = Rcpp::as<int>(maxitrs);
  if (num_col <= 0 || num_col >= n) {
    Rcpp::stop("numCol must be positive and less than sample size");
  }
  if (index < 0.0 || index > 2.0) index = 1.0;
  if (ncv <= num_col || ncv > n) {
    Rcpp::stop("ncv must be greater than numCol and no larger than sample size");
  }
  if (!std::isfinite(tol) || tol <= 0.0) {
    Rcpp::stop("tol must be a positive finite value");
  }
  if (maxitr <= 0) Rcpp::stop("maxitr must be positive");

  const auto batch_start = std::chrono::steady_clock::now();
  Rcpp::NumericVector p_values(batch);
  Rcpp::NumericVector nV2(batch);
  Rcpp::NumericVector means(batch);
  Rcpp::NumericVector variances(batch);
  Rcpp::NumericVector statistics(batch);
  Rcpp::NumericVector estimates(batch);
  Rcpp::NumericVector x_moments(batch);
  Rcpp::NumericVector y_moments(batch);

  std::vector<const double*> x_columns(static_cast<std::size_t>(batch));
  std::vector<const double*> y_columns(static_cast<std::size_t>(batch));
  for (int col = 0; col < batch; ++col) {
    const std::size_t offset = static_cast<std::size_t>(n) * col;
    x_columns[static_cast<std::size_t>(col)] = REAL(xs) + offset;
    y_columns[static_cast<std::size_t>(col)] = REAL(ys) + offset;
  }
  LegacyDcovCudaLowrankComponentBatchOptions batch_options;
  batch_options.n = n;
  batch_options.num_col = num_col;
  batch_options.index = index;
  batch_options.ncv = ncv;
  batch_options.tol = tol;
  batch_options.maxitr = maxitr;
  batch_options.batch_threads = 1;
  batch_options.key_mode =
    LegacyDcovCudaLowrankComponentBatchKeyMode::Value;
  const LegacyDcovCudaLowrankComponentBatchRun batch_run =
    legacy_dcov_cuda_lowrank_gamma_component_batch(
      x_columns, y_columns, batch_options);
  for (int col = 0; col < batch; ++col) {
    p_values[col] = batch_run.p_values[static_cast<std::size_t>(col)];
    nV2[col] = batch_run.nV2[static_cast<std::size_t>(col)];
    means[col] = batch_run.means[static_cast<std::size_t>(col)];
    variances[col] = batch_run.variances[static_cast<std::size_t>(col)];
    statistics[col] = batch_run.statistics[static_cast<std::size_t>(col)];
    estimates[col] = batch_run.estimates[static_cast<std::size_t>(col)];
    x_moments[col] = batch_run.x_moments[static_cast<std::size_t>(col)];
    y_moments[col] = batch_run.y_moments[static_cast<std::size_t>(col)];
  }

  return Rcpp::List::create(
    Rcpp::Named("backend") =
      "cuda-dense-sym-matvec-spectra-lowrank-gamma-batch",
    Rcpp::Named("p.value") = p_values,
    Rcpp::Named("nV2") = nV2,
    Rcpp::Named("mean") = means,
    Rcpp::Named("variance") = variances,
    Rcpp::Named("statistic") = statistics,
    Rcpp::Named("estimate") = estimates,
    Rcpp::Named("x_moment") = x_moments,
    Rcpp::Named("y_moment") = y_moments,
    Rcpp::Named("diagnostics") = Rcpp::List::create(
      Rcpp::Named("n") = n,
      Rcpp::Named("numCol") = num_col,
      Rcpp::Named("index") = index,
      Rcpp::Named("ncv") = ncv,
      Rcpp::Named("tol") = tol,
      Rcpp::Named("maxitr") = maxitr,
      Rcpp::Named("batch_count") = batch,
      Rcpp::Named("converged_count") = batch_run.converged_count,
      Rcpp::Named("component_cache_enabled") = true,
      Rcpp::Named("component_cache_lookup_count") =
        batch_run.component_cache_lookup_count,
      Rcpp::Named("component_cache_hit_count") =
        batch_run.component_cache_hit_count,
      Rcpp::Named("component_cache_miss_count") =
        batch_run.component_cache_miss_count,
      Rcpp::Named("component_cache_entry_count") =
        batch_run.component_cache_entry_count,
      Rcpp::Named("component_count") =
        batch_run.component_count,
      Rcpp::Named("component_total_ms") = batch_run.component_total_ms,
      Rcpp::Named("component_distance_ms") =
        batch_run.component_distance_ms,
      Rcpp::Named("component_lowrank_ms") =
        batch_run.component_lowrank_ms,
      Rcpp::Named("component_eig_ms") = batch_run.component_eig_ms,
      Rcpp::Named("component_moment_ms") =
        batch_run.component_moment_ms,
      Rcpp::Named("component_unaccounted_ms") =
        batch_run.component_unaccounted_ms,
      Rcpp::Named("combine_ms") = batch_run.combine_ms,
      Rcpp::Named("spectra_matvec_count") = batch_run.spectra_matvec_count,
      Rcpp::Named("spectra_matvec_ms") = batch_run.spectra_matvec_ms,
      Rcpp::Named("kernel_launch_count") = batch_run.kernel_launch_count,
      Rcpp::Named("device_matrix_reuse_count") =
        batch_run.device_matrix_reuse_count,
      Rcpp::Named("device_workspace_reuse_count") =
        batch_run.device_workspace_reuse_count,
      Rcpp::Named("workspace_realloc_count") =
        batch_run.workspace_realloc_count,
      Rcpp::Named("matrix_bytes") = batch_run.matrix_bytes,
      Rcpp::Named("workspace_bytes") = batch_run.workspace_bytes,
      Rcpp::Named("matrix_h2d_ms") = batch_run.matrix_h2d_ms,
      Rcpp::Named("matrix_h2d_ms_during_compute") =
        batch_run.matrix_h2d_ms_during_compute,
      Rcpp::Named("workspace_alloc_ms") = batch_run.workspace_alloc_ms,
      Rcpp::Named("h2d_ms") = batch_run.h2d_ms,
      Rcpp::Named("kernel_ms") = batch_run.kernel_ms,
      Rcpp::Named("d2h_ms") = batch_run.d2h_ms,
      Rcpp::Named("eig_ms") = batch_run.component_eig_ms,
      Rcpp::Named("pair_total_ms") = batch_run.pair_total_ms,
      Rcpp::Named("total_ms") = elapsed_ms_since(batch_start)
    )
  );
  END_RCPP
}

extern "C" SEXP C_legacy_dcov_gamma_cpp_component_cache_batch(
    SEXP residuals_s,
    SEXP left_columns_s,
    SEXP right_columns_s,
    SEXP num_col_s,
    SEXP index_s) {
  BEGIN_RCPP
  if (!Rf_isReal(residuals_s) || !Rf_isMatrix(residuals_s) ||
      !Rf_isInteger(left_columns_s) || !Rf_isInteger(right_columns_s)) {
    Rcpp::stop("legacy dCov component-cache inputs are malformed");
  }
  return fastkpc::legacy_dcov_gamma_cpp_compute_component_cache_batch(
    Rcpp::NumericMatrix(residuals_s),
    Rcpp::IntegerVector(left_columns_s),
    Rcpp::IntegerVector(right_columns_s),
    Rcpp::as<int>(num_col_s), Rcpp::as<double>(index_s));
  END_RCPP
}

extern "C" SEXP C_legacy_dcov_spectra_matvec_cuda_handle_free(
    SEXP handles) {
  BEGIN_RCPP
  if (TYPEOF(handles) != EXTPTRSXP) {
    Rcpp::stop("CUDA matvec handle must be an external pointer");
  }
  auto* handle =
    static_cast<fastkpc::LegacyDcovSpectraMatvecCudaHandle*>(
      R_ExternalPtrAddr(handles));
  if (handle != nullptr) {
    fastkpc::legacy_dcov_spectra_matvec_cuda_handle_destroy(handle);
    R_ClearExternalPtr(handles);
    return Rcpp::wrap(true);
  }
  return Rcpp::wrap(false);
  END_RCPP
}

extern "C" SEXP C_fast_dcov_batch_cuda(SEXP xs, SEXP ys, SEXP indexs,
                                        SEXP legacy_indexs) {
  BEGIN_RCPP
  if (!Rf_isReal(xs) || !Rf_isReal(ys) || !Rf_isMatrix(xs) || !Rf_isMatrix(ys)) {
    Rcpp::stop("x and y must be numeric matrices");
  }
  Rcpp::NumericMatrix x(xs);
  Rcpp::NumericMatrix y(ys);
  if (x.nrow() != y.nrow() || x.ncol() != y.ncol()) {
    Rcpp::stop("x and y must have identical dimensions");
  }
  const int n = x.nrow();
  const int batch = x.ncol();
  if (n <= 5) Rcpp::stop("gamma approximation requires n > 5");
  if (!all_finite(x) || !all_finite(y)) {
    Rcpp::stop("Data contains missing or infinite values");
  }
  double index = Rf_asReal(indexs);
  if (index < 0.0 || index > 2.0) {
    Rf_warning("index must be in [0,2), using default index=1");
    index = 1.0;
  }
  DcovBatchOptions options;
  options.index = index;
  options.legacy_index = Rcpp::as<bool>(legacy_indexs);
  DcovBatchResult result = dcov_batch_cuda(REAL(xs), REAL(ys), n, batch, options);

  Rcpp::NumericMatrix raw(batch, 5);
  for (int k = 0; k < batch; ++k) {
    for (int j = 0; j < 5; ++j) {
      raw(k, j) = result.raw_scalars[static_cast<std::size_t>(k) * 5 + j];
    }
  }
  return Rcpp::List::create(
    Rcpp::Named("p.value") = result.p_values,
    Rcpp::Named("nV2") = result.nV2,
    Rcpp::Named("mean") = result.means,
    Rcpp::Named("variance") = result.variances,
    Rcpp::Named("raw") = raw,
    Rcpp::Named("diagnostics") = Rcpp::List::create(
      Rcpp::Named("alloc_sec") = result.alloc_sec,
      Rcpp::Named("h2d_sec") = result.h2d_sec,
      Rcpp::Named("memset_sec") = result.memset_sec,
      Rcpp::Named("rowsum_sec") = result.rowsum_sec,
      Rcpp::Named("totals_d2h_sec") = result.totals_d2h_sec,
      Rcpp::Named("reduce_sec") = result.reduce_sec,
      Rcpp::Named("scalars_d2h_sec") = result.scalars_d2h_sec,
      Rcpp::Named("host_scalar_sec") = result.host_scalar_sec,
      Rcpp::Named("free_sec") = result.free_sec,
      Rcpp::Named("total_sec") = result.total_sec,
      Rcpp::Named("chunks") = result.chunks,
      Rcpp::Named("max_chunk_batch") = result.max_chunk_batch,
      Rcpp::Named("workspace_reuse_count") = result.workspace_reuse_count,
      Rcpp::Named("workspace_grow_count") = result.workspace_grow_count,
      Rcpp::Named("raw_aggregate_fused_count") =
        result.raw_aggregate_fused_count,
      Rcpp::Named("rowsum_kernel_launch_count") =
        result.rowsum_kernel_launch_count,
      Rcpp::Named("rowsum_chunk_count") = result.rowsum_chunk_count,
      Rcpp::Named("rowsum_total_blocks") = result.rowsum_total_blocks,
      Rcpp::Named("rowsum_pair_count") = result.rowsum_pair_count,
      Rcpp::Named("rowsum_abs_fast_count") = result.rowsum_abs_fast_count,
      Rcpp::Named("rowsum_pow_generic_count") =
        result.rowsum_pow_generic_count,
      Rcpp::Named("rowsum_abs_pair_count") = result.rowsum_abs_pair_count,
      Rcpp::Named("rowsum_generic_pair_count") =
        result.rowsum_generic_pair_count,
      Rcpp::Named("rowsum_threads") = result.rowsum_threads,
      Rcpp::Named("rowsum_n_max") = result.rowsum_n_max,
      Rcpp::Named("rowsum_batch_total") = result.rowsum_batch_total,
      Rcpp::Named("rowsum_max_chunk_batch") =
        result.rowsum_max_chunk_batch,
      Rcpp::Named("rowsum_max_chunk_sec") = result.rowsum_max_chunk_sec,
      Rcpp::Named("rowsum_max_chunk_n") = result.rowsum_max_chunk_n,
      Rcpp::Named("row_product_reduce_count") =
        result.row_product_reduce_count,
      Rcpp::Named("pvalue_only_count") = result.pvalue_only_count,
      Rcpp::Named("full_result_materialize_count") =
        result.full_result_materialize_count,
      Rcpp::Named("result_materialize_sec") =
        result.result_materialize_sec,
      Rcpp::Named("top_level_wall_sec") = result.top_level_wall_sec,
      Rcpp::Named("grid_limit_query_sec") = result.grid_limit_query_sec,
      Rcpp::Named("grid_limit_query_count") =
        result.grid_limit_query_count,
      Rcpp::Named("grid_limit_cache_hit_count") =
        result.grid_limit_cache_hit_count,
      Rcpp::Named("grid_limit_process_cache_hit_count") =
        result.grid_limit_process_cache_hit_count,
      Rcpp::Named("chunk_dispatch_sec") = result.chunk_dispatch_sec,
      Rcpp::Named("top_level_unaccounted_sec") =
        result.top_level_unaccounted_sec
    )
  );
  END_RCPP
}

extern "C" SEXP C_fast_hsic_gamma_cuda(SEXP xs, SEXP ys, SEXP sigs) {
  BEGIN_RCPP
  if (!Rf_isReal(xs) || !Rf_isReal(ys)) {
    Rcpp::stop("x and y must be numeric");
  }
  Rcpp::NumericVector x(xs);
  Rcpp::NumericVector y(ys);
  if (x.size() != y.size()) Rcpp::stop("Sample sizes must agree");
  if (x.size() < 4) Rcpp::stop("HSIC requires at least 4 observations");
  if (!all_finite_vector(x) || !all_finite_vector(y)) {
    Rcpp::stop("Data contains missing or infinite values");
  }
  HsicBatchOptions options = default_hsic_batch_options();
  options.sig = Rf_asReal(sigs);
  const HsicBatchResult result =
    hsic_gamma_batch_cuda(REAL(xs), REAL(ys), x.size(), 1, options);
  return hsic_batch_result_to_list(result, 0, options.sig);
  END_RCPP
}

extern "C" SEXP C_fast_hsic_perm_cuda(SEXP xs, SEXP ys, SEXP sigs,
                                       SEXP replicatess, SEXP seeds,
                                       SEXP include_observeds) {
  BEGIN_RCPP
  if (!Rf_isReal(xs) || !Rf_isReal(ys)) {
    Rcpp::stop("x and y must be numeric");
  }
  Rcpp::NumericVector x(xs);
  Rcpp::NumericVector y(ys);
  if (x.size() != y.size()) Rcpp::stop("Sample sizes must agree");
  if (x.size() < 4) Rcpp::stop("HSIC requires at least 4 observations");
  if (!all_finite_vector(x) || !all_finite_vector(y)) {
    Rcpp::stop("Data contains missing or infinite values");
  }
  HsicBatchOptions options = default_hsic_batch_options();
  options.sig = Rf_asReal(sigs);
  options.permutation_replicates = Rf_asInteger(replicatess);
  options.has_seed = true;
  options.seed = static_cast<unsigned int>(Rf_asInteger(seeds));
  options.include_observed = Rcpp::as<bool>(include_observeds);
  options.return_replicates = true;
  const HsicBatchResult result =
    hsic_permutation_batch_cuda(REAL(xs), REAL(ys), x.size(), 1, options);
  return hsic_batch_result_to_list(result, 0, options.sig);
  END_RCPP
}

extern "C" SEXP C_fastspline_residual_cuda(SEXP ys,
                                            SEXP Ss,
                                            SEXP fastspline_paramss,
                                            SEXP fallbacks) {
  BEGIN_RCPP
  if (!Rf_isReal(ys)) Rcpp::stop("y must be numeric");
  if (!Rf_isReal(Ss) || !Rf_isMatrix(Ss)) Rcpp::stop("S must be a numeric matrix");
  Rcpp::NumericVector y(ys);
  Rcpp::NumericMatrix S(Ss);
  if (y.size() != S.nrow()) {
    Rcpp::stop("y and S must have the same number of rows");
  }
  for (int i = 0; i < y.size(); ++i) {
    if (!std::isfinite(y[i])) Rcpp::stop("y contains missing or infinite values");
  }
  if (!all_finite(S)) Rcpp::stop("S contains missing or infinite values");

  Rcpp::NumericMatrix data(y.size(), S.ncol() + 1);
  for (int row = 0; row < y.size(); ++row) {
    data(row, 0) = y[row];
    for (int col = 0; col < S.ncol(); ++col) data(row, col + 1) = S(row, col);
  }
  std::vector<int> cond;
  for (int col = 0; col < S.ncol(); ++col) cond.push_back(col + 1);

  const FastSplineParams params =
    parse_fastspline_params(Rcpp::as<Rcpp::List>(fastspline_paramss));
  const FastSplineCudaFit result =
    fit_fastspline_residuals_cuda(data, 0, cond, params,
                                  Rcpp::as<bool>(fallbacks));
  return fastspline_cuda_fit_to_list(result);
  END_RCPP
}

extern "C" SEXP C_fastspline_residual_batch_cuda(SEXP datas,
                                                  SEXP targetss,
                                                  SEXP conditioning_setss,
                                                  SEXP fastspline_paramss,
                                                  SEXP fallbacks) {
  BEGIN_RCPP
  if (!Rf_isReal(datas) || !Rf_isMatrix(datas)) {
    Rcpp::stop("data must be a numeric matrix");
  }
  Rcpp::NumericMatrix data(datas);
  if (!all_finite(data)) Rcpp::stop("data contains missing or infinite values");
  Rcpp::IntegerVector targets(targetss);
  Rcpp::List conditioning_sets(conditioning_setss);
  if (targets.size() != conditioning_sets.size()) {
    Rcpp::stop("targets and conditioning_sets length mismatch");
  }

  std::vector<int> cpp_targets;
  std::vector<std::vector<int> > cpp_conditioning_sets;
  cpp_targets.reserve(targets.size());
  cpp_conditioning_sets.reserve(targets.size());
  for (int i = 0; i < targets.size(); ++i) {
    if (Rcpp::IntegerVector::is_na(targets[i])) Rcpp::stop("targets contain NA");
    const int target = targets[i] - 1;
    if (target < 0 || target >= data.ncol()) Rcpp::stop("target index out of range");
    cpp_targets.push_back(target);

    Rcpp::IntegerVector cond = conditioning_sets[i];
    std::vector<int> cpp_cond;
    cpp_cond.reserve(cond.size());
    for (int j = 0; j < cond.size(); ++j) {
      if (Rcpp::IntegerVector::is_na(cond[j])) Rcpp::stop("conditioning set contains NA");
      const int value = cond[j] - 1;
      if (value < 0 || value >= data.ncol()) {
        Rcpp::stop("conditioning set index out of range");
      }
      cpp_cond.push_back(value);
    }
    cpp_conditioning_sets.push_back(cpp_cond);
  }

  const FastSplineParams params =
    parse_fastspline_params(Rcpp::as<Rcpp::List>(fastspline_paramss));
  const FastSplineCudaBatchResult batch_result =
    fit_fastspline_residuals_cuda_batch_result(data, cpp_targets,
                                               cpp_conditioning_sets, params,
                                               Rcpp::as<bool>(fallbacks));
  const std::vector<FastSplineCudaFit>& fits = batch_result.fits;

  const int n = data.nrow();
  const int batch = static_cast<int>(fits.size());
  Rcpp::NumericMatrix residuals(n, batch);
  Rcpp::NumericMatrix fitted(n, batch);
  Rcpp::NumericVector selected_lambda(batch);
  Rcpp::NumericVector gcv(batch);
  Rcpp::NumericVector rss(batch);
  Rcpp::NumericVector edf(batch);
  Rcpp::IntegerVector design_cols(batch);
  Rcpp::IntegerVector ridge_attempts(batch);
  Rcpp::CharacterVector residual_device(batch);
  Rcpp::LogicalVector fallback_used(batch);
  Rcpp::List diagnostics(batch);

  for (int k = 0; k < batch; ++k) {
    const FastSplineFit& fit = fits[k].fit;
    if (static_cast<int>(fit.residuals.size()) != n ||
        static_cast<int>(fit.fitted.size()) != n) {
      Rcpp::stop("CUDA residual batch result dimension mismatch");
    }
    for (int row = 0; row < n; ++row) {
      residuals(row, k) = fit.residuals[row];
      fitted(row, k) = fit.fitted[row];
    }
    selected_lambda[k] = fit.selected_lambda;
    gcv[k] = fit.gcv;
    rss[k] = fit.rss;
    edf[k] = fit.edf;
    design_cols[k] = fit.design_cols;
    ridge_attempts[k] = fit.ridge_attempts;
    residual_device[k] = fits[k].diagnostics.fallback_used ?
      "cuda-fallback-cpu" : "cuda";
    fallback_used[k] = fits[k].diagnostics.fallback_used;
    diagnostics[k] = Rcpp::List::create(
      Rcpp::Named("cuda_used") = fits[k].diagnostics.cuda_used,
      Rcpp::Named("fallback_used") = fits[k].diagnostics.fallback_used,
      Rcpp::Named("reason") = fits[k].diagnostics.reason,
      Rcpp::Named("batch_group_id") = fits[k].diagnostics.batch_group_id,
      Rcpp::Named("batch_position") = fits[k].diagnostics.batch_position,
      Rcpp::Named("true_batched") = fits[k].diagnostics.true_batched,
      Rcpp::Named("cholesky_backend") = fits[k].diagnostics.cholesky_backend
    );
  }

  return Rcpp::List::create(
    Rcpp::Named("residuals") = residuals,
    Rcpp::Named("fitted") = fitted,
    Rcpp::Named("selected_lambda") = selected_lambda,
    Rcpp::Named("gcv") = gcv,
    Rcpp::Named("rss") = rss,
    Rcpp::Named("edf") = edf,
    Rcpp::Named("design_cols") = design_cols,
    Rcpp::Named("ridge_attempts") = ridge_attempts,
    Rcpp::Named("backend") = "cuda",
    Rcpp::Named("residual_backend") = "fastSpline",
    Rcpp::Named("residual_device") = residual_device,
    Rcpp::Named("fallback_used") = fallback_used,
    Rcpp::Named("diagnostics") = diagnostics,
    Rcpp::Named("batch_diagnostics") =
      fastspline_batch_diagnostics_to_list(batch_result.diagnostics)
  );
  END_RCPP
}

extern "C" SEXP C_mgcv_extract_gpu_test_checked_augmented_rows(
    SEXP ns, SEXP null_dims) {
  BEGIN_RCPP
  return Rcpp::wrap(mgcv_extract_fixed_sp_checked_augmented_rows(
    Rcpp::as<int>(ns), Rcpp::as<int>(null_dims)));
  END_RCPP
}

extern "C" SEXP C_full_cuda_ci_multi_penalty_gcv_evaluate_cpp(
    SEXP Xs,
    SEXP ys,
    SEXP penalty_blocks_s,
    SEXP penalty_offsets_s,
    SEXP penalty_ranks_s,
    SEXP log_sp_s,
    SEXP Hs,
    SEXP constraint_s,
    SEXP rank_tolerance_s) {
  BEGIN_RCPP
  if (!Rf_isReal(Xs) || !Rf_isMatrix(Xs) || !Rf_isReal(ys) ||
      !Rf_isNewList(penalty_blocks_s) ||
      !Rf_isInteger(penalty_offsets_s) ||
      !Rf_isInteger(penalty_ranks_s) || !Rf_isReal(log_sp_s)) {
    Rcpp::stop("multi-penalty C++ inputs have invalid storage types");
  }
  Rcpp::NumericMatrix X(Xs);
  Rcpp::NumericVector y(ys);
  Rcpp::List penalty_blocks(penalty_blocks_s);
  Rcpp::IntegerVector penalty_offsets(penalty_offsets_s);
  Rcpp::IntegerVector penalty_ranks(penalty_ranks_s);
  Rcpp::NumericVector log_sp(log_sp_s);
  const int n = X.nrow();
  const int p = X.ncol();
  const int penalty_count = penalty_blocks.size();
  if (y.size() != n || penalty_count <= 1 ||
      penalty_offsets.size() != penalty_count ||
      penalty_ranks.size() != penalty_count ||
      log_sp.size() != penalty_count) {
    Rcpp::stop("multi-penalty C++ input dimensions are inconsistent");
  }
  std::vector<std::vector<double>> blocks;
  std::vector<int> dimensions;
  std::vector<int> offsets;
  std::vector<int> ranks;
  blocks.reserve(penalty_count);
  dimensions.reserve(penalty_count);
  offsets.reserve(penalty_count);
  ranks.reserve(penalty_count);
  for (int index = 0; index < penalty_count; ++index) {
    SEXP block_s = penalty_blocks[index];
    if (!Rf_isReal(block_s) || !Rf_isMatrix(block_s)) {
      Rcpp::stop("each multi-penalty block must be a numeric matrix");
    }
    Rcpp::NumericMatrix block(block_s);
    if (block.nrow() <= 0 || block.nrow() != block.ncol()) {
      Rcpp::stop("each multi-penalty block must be nonempty and square");
    }
    blocks.emplace_back(REAL(block_s), REAL(block_s) + XLENGTH(block_s));
    dimensions.push_back(block.nrow());
    offsets.push_back(penalty_offsets[index] - 1);
    ranks.push_back(penalty_ranks[index]);
  }
  const double* H = nullptr;
  bool has_H = false;
  if (!Rf_isNull(Hs)) {
    if (!Rf_isReal(Hs) || !Rf_isMatrix(Hs)) {
      Rcpp::stop("fixed penalty H must be NULL or a numeric matrix");
    }
    Rcpp::NumericMatrix H_matrix(Hs);
    if (H_matrix.nrow() != p || H_matrix.ncol() != p) {
      Rcpp::stop("fixed penalty H must have dimension p x p");
    }
    H = REAL(Hs);
    has_H = true;
  }
  const double* constraint = nullptr;
  int constraint_rows = 0;
  if (!Rf_isNull(constraint_s)) {
    if (!Rf_isReal(constraint_s) || !Rf_isMatrix(constraint_s)) {
      Rcpp::stop("constraint must be NULL or a numeric matrix");
    }
    Rcpp::NumericMatrix constraint_matrix(constraint_s);
    if (constraint_matrix.ncol() != p) {
      Rcpp::stop("constraint must have ncol equal to p");
    }
    constraint_rows = constraint_matrix.nrow();
    if (constraint_rows > 0) constraint = REAL(constraint_s);
  }
  const std::vector<double> log_sp_values(log_sp.begin(), log_sp.end());
  const fastkpc::MultiPenaltyGcvEvaluation result =
    fastkpc::multi_penalty_gcv_evaluate_cpp(
      REAL(Xs), REAL(ys), n, p, blocks, dimensions, offsets, ranks,
      log_sp_values, H, has_H, constraint, constraint_rows,
      Rcpp::as<double>(rank_tolerance_s));
  Rcpp::NumericMatrix hessian(result.penalty_count, result.penalty_count);
  std::copy(result.hessian.begin(), result.hessian.end(), hessian.begin());
  return Rcpp::List::create(
    Rcpp::Named("schema_version") =
      "full-cuda-ci-multi-penalty-cpp-evaluation-v1",
    Rcpp::Named("rank_path") =
      "pivoted-qr-augmented-lapack-dgesdd-svd",
    Rcpp::Named("constraint_aware") = true,
    Rcpp::Named("normal_equations_used") = false,
    Rcpp::Named("n") = result.n,
    Rcpp::Named("coefficient_dim") = result.coefficient_dim,
    Rcpp::Named("free_dim") = result.free_dim,
    Rcpp::Named("penalty_count") = result.penalty_count,
    Rcpp::Named("constraint_rank") = result.constraint_rank,
    Rcpp::Named("augmented_penalty_rank") =
      result.augmented_penalty_rank,
    Rcpp::Named("numerical_rank") = result.numerical_rank,
    Rcpp::Named("condition") = result.condition,
    Rcpp::Named("condition_bucket") = result.condition_bucket,
    Rcpp::Named("log_sp") = Rcpp::wrap(result.log_sp),
    Rcpp::Named("rss") = result.rss,
    Rcpp::Named("edf") = result.edf,
    Rcpp::Named("score") = result.score,
    Rcpp::Named("gradient") = Rcpp::wrap(result.gradient),
    Rcpp::Named("hessian") = hessian,
    Rcpp::Named("coefficients") = Rcpp::wrap(result.coefficients),
    Rcpp::Named("fitted") = Rcpp::wrap(result.fitted),
    Rcpp::Named("residuals") = Rcpp::wrap(result.residuals));
  END_RCPP
}

extern "C" SEXP C_full_cuda_ci_multi_penalty_gcv_optimize_cpp(
    SEXP Xs,
    SEXP ys,
    SEXP penalty_blocks_s,
    SEXP penalty_offsets_s,
    SEXP penalty_ranks_s,
    SEXP Hs,
    SEXP constraint_s,
    SEXP control_s) {
  BEGIN_RCPP
  if (!Rf_isReal(Xs) || !Rf_isMatrix(Xs) || !Rf_isReal(ys) ||
      !Rf_isNewList(penalty_blocks_s) ||
      !Rf_isInteger(penalty_offsets_s) ||
      !Rf_isInteger(penalty_ranks_s) || !Rf_isNewList(control_s)) {
    Rcpp::stop("multi-penalty optimizer inputs have invalid storage types");
  }
  Rcpp::NumericMatrix X(Xs);
  Rcpp::NumericVector y(ys);
  Rcpp::List penalty_blocks(penalty_blocks_s);
  Rcpp::IntegerVector penalty_offsets(penalty_offsets_s);
  Rcpp::IntegerVector penalty_ranks(penalty_ranks_s);
  Rcpp::List control_values(control_s);
  const int n = X.nrow();
  const int p = X.ncol();
  const int penalty_count = penalty_blocks.size();
  if (y.size() != n || penalty_count <= 1 ||
      penalty_offsets.size() != penalty_count ||
      penalty_ranks.size() != penalty_count) {
    Rcpp::stop("multi-penalty optimizer input dimensions are inconsistent");
  }
  std::vector<std::vector<double>> blocks;
  std::vector<int> dimensions;
  std::vector<int> offsets;
  std::vector<int> ranks;
  blocks.reserve(penalty_count);
  dimensions.reserve(penalty_count);
  offsets.reserve(penalty_count);
  ranks.reserve(penalty_count);
  for (int index = 0; index < penalty_count; ++index) {
    SEXP block_s = penalty_blocks[index];
    if (!Rf_isReal(block_s) || !Rf_isMatrix(block_s)) {
      Rcpp::stop("each multi-penalty block must be a numeric matrix");
    }
    Rcpp::NumericMatrix block(block_s);
    if (block.nrow() <= 0 || block.nrow() != block.ncol()) {
      Rcpp::stop("each multi-penalty block must be nonempty and square");
    }
    blocks.emplace_back(REAL(block_s), REAL(block_s) + XLENGTH(block_s));
    dimensions.push_back(block.nrow());
    offsets.push_back(penalty_offsets[index] - 1);
    ranks.push_back(penalty_ranks[index]);
  }
  const double* H = nullptr;
  bool has_H = false;
  if (!Rf_isNull(Hs)) {
    if (!Rf_isReal(Hs) || !Rf_isMatrix(Hs)) {
      Rcpp::stop("fixed penalty H must be NULL or a numeric matrix");
    }
    Rcpp::NumericMatrix H_matrix(Hs);
    if (H_matrix.nrow() != p || H_matrix.ncol() != p) {
      Rcpp::stop("fixed penalty H must have dimension p x p");
    }
    H = REAL(Hs);
    has_H = true;
  }
  const double* constraint = nullptr;
  int constraint_rows = 0;
  if (!Rf_isNull(constraint_s)) {
    if (!Rf_isReal(constraint_s) || !Rf_isMatrix(constraint_s)) {
      Rcpp::stop("constraint must be NULL or a numeric matrix");
    }
    Rcpp::NumericMatrix constraint_matrix(constraint_s);
    if (constraint_matrix.ncol() != p) {
      Rcpp::stop("constraint must have ncol equal to p");
    }
    constraint_rows = constraint_matrix.nrow();
    if (constraint_rows > 0) constraint = REAL(constraint_s);
  }
  auto numeric_control = [&](const char* name, double fallback) {
    return control_values.containsElementNamed(name) ?
      Rcpp::as<double>(control_values[name]) : fallback;
  };
  auto integer_control = [&](const char* name, int fallback) {
    return control_values.containsElementNamed(name) ?
      Rcpp::as<int>(control_values[name]) : fallback;
  };
  fastkpc::MultiPenaltyOptimizerControl control;
  control.convergence_tolerance = numeric_control(
    "convergence_tolerance", control.convergence_tolerance);
  control.max_step_halving = integer_control(
    "max_step_halving", control.max_step_halving);
  control.max_iterations = integer_control(
    "max_iterations", control.max_iterations);
  control.max_newton_step = numeric_control(
    "max_newton_step", control.max_newton_step);
  control.boundary_probe_step = numeric_control(
    "boundary_probe_step", control.boundary_probe_step);
  control.max_boundary_probes = integer_control(
    "max_boundary_probes", control.max_boundary_probes);
  control.rank_tolerance = numeric_control(
    "rank_tolerance", control.rank_tolerance);
  control.keep_transcript =
    control_values.containsElementNamed("keep_transcript") &&
    Rcpp::as<bool>(control_values["keep_transcript"]);

  const fastkpc::MultiPenaltyGcvOptimization result =
    fastkpc::multi_penalty_gcv_optimize_cpp(
      REAL(Xs), REAL(ys), n, p, blocks, dimensions, offsets, ranks,
      H, has_H, constraint, constraint_rows, control);
  const fastkpc::MultiPenaltyGcvEvaluation& selected = result.selected;
  Rcpp::NumericMatrix hessian(selected.penalty_count,
                              selected.penalty_count);
  std::copy(selected.hessian.begin(), selected.hessian.end(),
            hessian.begin());
  Rcpp::List transcript(result.transcript.size());
  for (std::size_t index = 0; index < result.transcript.size(); ++index) {
    const fastkpc::MultiPenaltyOptimizerTranscriptEntry& entry =
      result.transcript[index];
    Rcpp::NumericMatrix entry_hessian(selected.penalty_count,
                                      selected.penalty_count);
    std::copy(entry.hessian.begin(), entry.hessian.end(),
              entry_hessian.begin());
    transcript[index] = Rcpp::List::create(
      Rcpp::Named("stage") = entry.stage,
      Rcpp::Named("iteration") = entry.iteration,
      Rcpp::Named("evaluation") = entry.evaluation,
      Rcpp::Named("coordinate") = entry.coordinate < 0 ?
        NA_INTEGER : entry.coordinate + 1,
      Rcpp::Named("current_log_sp") = Rcpp::wrap(entry.current_log_sp),
      Rcpp::Named("proposed_step") = Rcpp::wrap(entry.proposed_step),
      Rcpp::Named("trial_log_sp") = Rcpp::wrap(entry.trial_log_sp),
      Rcpp::Named("objective") = entry.score,
      Rcpp::Named("gradient") = Rcpp::wrap(entry.gradient),
      Rcpp::Named("hessian") = entry_hessian,
      Rcpp::Named("hessian_eigenvalues") =
        Rcpp::wrap(entry.hessian_eigenvalues),
      Rcpp::Named("accepted") = entry.accepted,
      Rcpp::Named("step_source") = entry.step_source,
      Rcpp::Named("numerical_rank") = entry.numerical_rank,
      Rcpp::Named("condition") = entry.condition);
  }
  return Rcpp::List::create(
    Rcpp::Named("schema_version") =
      "full-cuda-ci-multi-penalty-cpp-optimization-v1",
    Rcpp::Named("rank_path") =
      "pivoted-qr-augmented-lapack-dgesdd-svd",
    Rcpp::Named("selected_fit_refinement_path") =
      "pivoted-qr-augmented-lapack-dgesdd-svd",
    Rcpp::Named("constraint_aware") = true,
    Rcpp::Named("normal_equations_used") = false,
    Rcpp::Named("response_independent_initialization") = true,
    Rcpp::Named("fallback_reason") = "NONE",
    Rcpp::Named("n") = selected.n,
    Rcpp::Named("coefficient_dim") = selected.coefficient_dim,
    Rcpp::Named("free_dim") = selected.free_dim,
    Rcpp::Named("penalty_count") = selected.penalty_count,
    Rcpp::Named("constraint_rank") = selected.constraint_rank,
    Rcpp::Named("augmented_penalty_rank") =
      selected.augmented_penalty_rank,
    Rcpp::Named("numerical_rank") = selected.numerical_rank,
    Rcpp::Named("condition") = selected.condition,
    Rcpp::Named("condition_bucket") = selected.condition_bucket,
    Rcpp::Named("initial_log_sp") = Rcpp::wrap(result.initial_log_sp),
    Rcpp::Named("selected_log_sp") = Rcpp::wrap(selected.log_sp),
    Rcpp::Named("rss") = selected.rss,
    Rcpp::Named("edf") = selected.edf,
    Rcpp::Named("score") = selected.score,
    Rcpp::Named("gradient") = Rcpp::wrap(selected.gradient),
    Rcpp::Named("hessian") = hessian,
    Rcpp::Named("coefficients") = Rcpp::wrap(selected.coefficients),
    Rcpp::Named("fitted") = Rcpp::wrap(selected.fitted),
    Rcpp::Named("residuals") = Rcpp::wrap(selected.residuals),
    Rcpp::Named("optimizer_iterations") = result.optimizer_iterations,
    Rcpp::Named("score_calls") = result.score_calls,
    Rcpp::Named("objective_calls") = result.objective_calls,
    Rcpp::Named("step_halving_count") = result.step_halving_count,
    Rcpp::Named("newton_trial_count") = result.newton_trial_count,
    Rcpp::Named("steepest_descent_trial_count") =
      result.steepest_descent_trial_count,
    Rcpp::Named("boundary_probe_count") = result.boundary_probe_count,
    Rcpp::Named("boundary_accepted_count") =
      result.boundary_accepted_count,
    Rcpp::Named("boundary_status") = Rcpp::wrap(result.boundary_status),
    Rcpp::Named("fully_converged") = result.fully_converged,
    Rcpp::Named("hessian_positive_definite") =
      result.hessian_positive_definite,
    Rcpp::Named("step_failed") = result.step_failed,
    Rcpp::Named("rms_gradient") = result.rms_gradient,
    Rcpp::Named("convergence_code") = result.convergence_code,
    Rcpp::Named("transcript") = transcript);
  END_RCPP
}

extern "C" SEXP C_full_cuda_ci_multi_penalty_gcv_evaluate_cuda(
    SEXP Xs,
    SEXP Ys,
    SEXP magic_qr_packed_s,
    SEXP magic_tau_s,
    SEXP magic_r_s,
    SEXP magic_pivot_s,
    SEXP penalty_roots_s,
    SEXP penalty_matrices_s,
    SEXP penalty_ranks_s,
    SEXP log_sp_s,
    SEXP rank_tolerance_s) {
  BEGIN_RCPP
  if (!Rf_isReal(Xs) || !Rf_isMatrix(Xs) ||
      !Rf_isReal(Ys) || !Rf_isMatrix(Ys) ||
      !Rf_isReal(magic_qr_packed_s) || !Rf_isMatrix(magic_qr_packed_s) ||
      !Rf_isReal(magic_tau_s) ||
      !Rf_isReal(magic_r_s) || !Rf_isMatrix(magic_r_s) ||
      !Rf_isInteger(magic_pivot_s) ||
      !Rf_isNewList(penalty_roots_s) ||
      !Rf_isNewList(penalty_matrices_s) ||
      !Rf_isInteger(penalty_ranks_s) ||
      !Rf_isReal(log_sp_s) || !Rf_isMatrix(log_sp_s)) {
    Rcpp::stop("multi-penalty CUDA inputs have invalid storage types");
  }
  Rcpp::NumericMatrix X(Xs);
  Rcpp::NumericMatrix Y(Ys);
  Rcpp::NumericMatrix qr_packed(magic_qr_packed_s);
  Rcpp::NumericVector tau(magic_tau_s);
  Rcpp::NumericMatrix magic_r(magic_r_s);
  Rcpp::IntegerVector pivot(magic_pivot_s);
  Rcpp::List penalty_roots(penalty_roots_s);
  Rcpp::List penalty_matrices(penalty_matrices_s);
  Rcpp::IntegerVector penalty_ranks(penalty_ranks_s);
  Rcpp::NumericMatrix log_sp(log_sp_s);
  const int n = X.nrow();
  const int q = X.ncol();
  const int target_count = Y.ncol();
  const int penalty_count = penalty_roots.size();
  if (Y.nrow() != n || target_count <= 1 ||
      qr_packed.nrow() != n || qr_packed.ncol() != q ||
      tau.size() != q || magic_r.nrow() != q || magic_r.ncol() != q ||
      pivot.size() != q || penalty_count <= 1 ||
      penalty_matrices.size() != penalty_count ||
      penalty_ranks.size() != penalty_count ||
      log_sp.nrow() != penalty_count || log_sp.ncol() != target_count) {
    Rcpp::stop("multi-penalty CUDA input dimensions are inconsistent");
  }
  std::vector<int> pivot_zero_based(static_cast<std::size_t>(q));
  std::vector<unsigned char> pivot_seen(static_cast<std::size_t>(q), 0);
  for (int index = 0; index < q; ++index) {
    const int value = pivot[index] - 1;
    if (value < 0 || value >= q ||
        pivot_seen[static_cast<std::size_t>(value)] != 0) {
      Rcpp::stop("multi-penalty CUDA QR pivot is invalid");
    }
    pivot_seen[static_cast<std::size_t>(value)] = 1;
    pivot_zero_based[static_cast<std::size_t>(index)] = value;
  }
  std::vector<std::vector<double>> roots;
  std::vector<std::vector<double>> matrices;
  std::vector<int> ranks;
  roots.reserve(static_cast<std::size_t>(penalty_count));
  matrices.reserve(static_cast<std::size_t>(penalty_count));
  ranks.reserve(static_cast<std::size_t>(penalty_count));
  for (int penalty = 0; penalty < penalty_count; ++penalty) {
    SEXP root_s = penalty_roots[penalty];
    SEXP matrix_s = penalty_matrices[penalty];
    if (!Rf_isReal(root_s) || !Rf_isMatrix(root_s) ||
        !Rf_isReal(matrix_s) || !Rf_isMatrix(matrix_s)) {
      Rcpp::stop("multi-penalty CUDA penalty inputs must be matrices");
    }
    Rcpp::NumericMatrix root(root_s);
    Rcpp::NumericMatrix matrix(matrix_s);
    const int rank = penalty_ranks[penalty];
    if (root.nrow() != q || root.ncol() != rank || rank <= 0 ||
        matrix.nrow() != q || matrix.ncol() != q) {
      Rcpp::stop("multi-penalty CUDA penalty dimensions are invalid");
    }
    roots.emplace_back(REAL(root_s), REAL(root_s) + XLENGTH(root_s));
    matrices.emplace_back(
      REAL(matrix_s), REAL(matrix_s) + XLENGTH(matrix_s));
    ranks.push_back(rank);
  }
  const fastkpc::MultiPenaltyGcvCudaEvaluation result =
    fastkpc::multi_penalty_gcv_evaluate_cuda(
      REAL(Xs), REAL(Ys), REAL(magic_qr_packed_s), REAL(magic_tau_s),
      REAL(magic_r_s), pivot_zero_based.data(), roots, matrices, ranks,
      REAL(log_sp_s), n, q, penalty_count, target_count,
      Rcpp::as<double>(rank_tolerance_s));
  Rcpp::NumericMatrix gradient(result.penalty_count, result.target_count);
  std::copy(result.gradient.begin(), result.gradient.end(), gradient.begin());
  Rcpp::NumericVector hessian(result.hessian.begin(), result.hessian.end());
  hessian.attr("dim") = Rcpp::IntegerVector::create(
    result.penalty_count, result.penalty_count, result.target_count);
  Rcpp::NumericMatrix coefficients(
    result.coefficient_dim, result.target_count);
  std::copy(
    result.coefficients.begin(), result.coefficients.end(),
    coefficients.begin());
  const fastkpc::MultiPenaltyGcvCudaDiagnostics& diagnostics =
    result.diagnostics;
  return Rcpp::List::create(
    Rcpp::Named("schema_version") = result.schema_version,
    Rcpp::Named("rank_path") = result.rank_path,
    Rcpp::Named("n") = result.n,
    Rcpp::Named("coefficient_dim") = result.coefficient_dim,
    Rcpp::Named("penalty_count") = result.penalty_count,
    Rcpp::Named("target_count") = result.target_count,
    Rcpp::Named("rss") = Rcpp::wrap(result.rss),
    Rcpp::Named("edf") = Rcpp::wrap(result.edf),
    Rcpp::Named("score") = Rcpp::wrap(result.score),
    Rcpp::Named("condition") = Rcpp::wrap(result.condition),
    Rcpp::Named("aggregate_penalty_rank") =
      Rcpp::wrap(result.aggregate_penalty_rank),
    Rcpp::Named("numerical_rank") = Rcpp::wrap(result.numerical_rank),
    Rcpp::Named("solver_info") = Rcpp::wrap(result.solver_info),
    Rcpp::Named("gradient") = gradient,
    Rcpp::Named("hessian") = hessian,
    Rcpp::Named("coefficients") = coefficients,
    Rcpp::Named("diagnostics") = Rcpp::List::create(
      Rcpp::Named("schema_version") = diagnostics.schema_version,
      Rcpp::Named("execution_strategy") = diagnostics.execution_strategy,
      Rcpp::Named("device_id") = diagnostics.device_id,
      Rcpp::Named("gpu_name") = diagnostics.gpu_name,
      Rcpp::Named("prepared_setup_upload_count") =
        diagnostics.prepared_setup_upload_count,
      Rcpp::Named("target_batch_upload_count") =
        diagnostics.target_batch_upload_count,
      Rcpp::Named("cuda_qt_y_kernel_launch_count") =
        diagnostics.cuda_qt_y_kernel_launch_count,
      Rcpp::Named("cuda_objective_kernel_launch_count") =
        diagnostics.cuda_objective_kernel_launch_count,
      Rcpp::Named("cuda_objective_target_count") =
        diagnostics.cuda_objective_target_count,
      Rcpp::Named("cuda_guarded_qr_evaluation_count") =
        diagnostics.cuda_guarded_qr_evaluation_count,
      Rcpp::Named("cuda_stable_svd_evaluation_count") =
        diagnostics.cuda_stable_svd_evaluation_count,
      Rcpp::Named("cpu_objective_count") = diagnostics.cpu_objective_count,
      Rcpp::Named("cpu_multi_penalty_solve_count") =
        diagnostics.cpu_multi_penalty_solve_count,
      Rcpp::Named("fallback_count") = diagnostics.fallback_count,
      Rcpp::Named("cuda_error_count") = diagnostics.cuda_error_count,
      Rcpp::Named("svd_nonconverged_count") =
        diagnostics.svd_nonconverged_count,
      Rcpp::Named("aggregate_rank_failure_count") =
        diagnostics.aggregate_rank_failure_count,
      Rcpp::Named("device_allocation_count") =
        diagnostics.device_allocation_count,
      Rcpp::Named("h2d_copy_count") = diagnostics.h2d_copy_count,
      Rcpp::Named("d2h_copy_count") = diagnostics.d2h_copy_count,
      Rcpp::Named("target_specific_log_sp") =
        diagnostics.target_specific_log_sp,
      Rcpp::Named("true_batched_kernel") = diagnostics.true_batched_kernel,
      Rcpp::Named("normal_equations_used") =
        diagnostics.normal_equations_used,
      Rcpp::Named("double_precision") = diagnostics.double_precision,
      Rcpp::Named("total_host_ms") = diagnostics.total_host_ms));
  END_RCPP
}

namespace {

Rcpp::List multi_penalty_cuda_optimization_to_list(
    const fastkpc::MultiPenaltyGcvCudaOptimization& result) {
  Rcpp::NumericMatrix selected_log_sp(
    result.penalty_count, result.target_count);
  std::copy(result.selected_log_sp.begin(), result.selected_log_sp.end(),
            selected_log_sp.begin());
  Rcpp::NumericMatrix gradient(result.penalty_count, result.target_count);
  std::copy(result.gradient.begin(), result.gradient.end(), gradient.begin());
  Rcpp::NumericVector hessian(result.hessian.begin(), result.hessian.end());
  hessian.attr("dim") = Rcpp::IntegerVector::create(
    result.penalty_count, result.penalty_count, result.target_count);
  Rcpp::NumericMatrix coefficients(
    result.coefficient_dim, result.target_count);
  std::copy(result.coefficients.begin(), result.coefficients.end(),
            coefficients.begin());
  Rcpp::NumericMatrix hessian_eigenvalues(
    result.penalty_count, result.target_count);
  std::copy(result.hessian_eigenvalues.begin(),
            result.hessian_eigenvalues.end(),
            hessian_eigenvalues.begin());
  Rcpp::CharacterMatrix boundary_status(
    result.penalty_count, result.target_count);
  Rcpp::LogicalVector fully_converged(result.target_count);
  Rcpp::LogicalVector hessian_positive_definite(result.target_count);
  Rcpp::LogicalVector step_failed(result.target_count);
  Rcpp::CharacterVector convergence_code(result.target_count);
  for (int target = 0; target < result.target_count; ++target) {
    fully_converged[target] = result.fully_converged[target] != 0;
    hessian_positive_definite[target] =
      result.hessian_positive_definite[target] != 0;
    step_failed[target] = result.step_failed[target] != 0;
    convergence_code[target] = result.optimizer_status[target] != 0 ?
      "cuda_optimizer_error" :
      (result.step_failed[target] != 0 ?
        "step_halving_exhausted" : "fully_converged");
    for (int penalty = 0; penalty < result.penalty_count; ++penalty) {
      const int value = result.boundary_status[
        penalty + result.penalty_count * target];
      boundary_status(penalty, target) = value == 0 ? "finite-interior" :
        (value == 1 ? "positive-boundary" :
          (value == 2 ? "negative-boundary" :
            (value == 3 ? "finite-after-boundary-probe" : "unresolved")));
    }
  }
  const fastkpc::MultiPenaltyGcvCudaDiagnostics& diagnostics =
    result.diagnostics;
  return Rcpp::List::create(
    Rcpp::Named("schema_version") = result.schema_version,
    Rcpp::Named("rank_path") = result.rank_path,
    Rcpp::Named("optimizer_path") = result.optimizer_path,
    Rcpp::Named("n") = result.n,
    Rcpp::Named("coefficient_dim") = result.coefficient_dim,
    Rcpp::Named("penalty_count") = result.penalty_count,
    Rcpp::Named("target_count") = result.target_count,
    Rcpp::Named("initial_log_sp") = Rcpp::wrap(result.initial_log_sp),
    Rcpp::Named("selected_log_sp") = selected_log_sp,
    Rcpp::Named("rss") = Rcpp::wrap(result.rss),
    Rcpp::Named("edf") = Rcpp::wrap(result.edf),
    Rcpp::Named("score") = Rcpp::wrap(result.score),
    Rcpp::Named("condition") = Rcpp::wrap(result.condition),
    Rcpp::Named("gradient") = gradient,
    Rcpp::Named("hessian") = hessian,
    Rcpp::Named("coefficients") = coefficients,
    Rcpp::Named("rms_gradient") = Rcpp::wrap(result.rms_gradient),
    Rcpp::Named("hessian_eigenvalues") = hessian_eigenvalues,
    Rcpp::Named("aggregate_penalty_rank") =
      Rcpp::wrap(result.aggregate_penalty_rank),
    Rcpp::Named("numerical_rank") = Rcpp::wrap(result.numerical_rank),
    Rcpp::Named("solver_info") = Rcpp::wrap(result.solver_info),
    Rcpp::Named("optimizer_iterations") =
      Rcpp::wrap(result.optimizer_iterations),
    Rcpp::Named("score_calls") = Rcpp::wrap(result.score_calls),
    Rcpp::Named("objective_calls") = Rcpp::wrap(result.objective_calls),
    Rcpp::Named("step_halving_count") =
      Rcpp::wrap(result.step_halving_count),
    Rcpp::Named("newton_trial_count") =
      Rcpp::wrap(result.newton_trial_count),
    Rcpp::Named("steepest_descent_trial_count") =
      Rcpp::wrap(result.steepest_descent_trial_count),
    Rcpp::Named("boundary_probe_count") =
      Rcpp::wrap(result.boundary_probe_count),
    Rcpp::Named("boundary_accepted_count") =
      Rcpp::wrap(result.boundary_accepted_count),
    Rcpp::Named("boundary_status") = boundary_status,
    Rcpp::Named("fully_converged") = fully_converged,
    Rcpp::Named("hessian_positive_definite") =
      hessian_positive_definite,
    Rcpp::Named("step_failed") = step_failed,
    Rcpp::Named("convergence_code") = convergence_code,
    Rcpp::Named("optimizer_status") = Rcpp::wrap(result.optimizer_status),
    Rcpp::Named("diagnostics") = Rcpp::List::create(
      Rcpp::Named("schema_version") = diagnostics.schema_version,
      Rcpp::Named("execution_strategy") = diagnostics.execution_strategy,
      Rcpp::Named("device_id") = diagnostics.device_id,
      Rcpp::Named("gpu_name") = diagnostics.gpu_name,
      Rcpp::Named("prepared_setup_upload_count") =
        diagnostics.prepared_setup_upload_count,
      Rcpp::Named("target_batch_upload_count") =
        diagnostics.target_batch_upload_count,
      Rcpp::Named("cuda_qt_y_kernel_launch_count") =
        diagnostics.cuda_qt_y_kernel_launch_count,
      Rcpp::Named("cuda_optimizer_kernel_launch_count") =
        diagnostics.cuda_optimizer_kernel_launch_count,
      Rcpp::Named("cuda_optimizer_target_count") =
        diagnostics.cuda_optimizer_target_count,
      Rcpp::Named("cuda_optimizer_objective_count") =
        diagnostics.cuda_optimizer_objective_count,
      Rcpp::Named("cuda_penalty_factor_augmentation_cycles") =
        static_cast<double>(
          diagnostics.cuda_penalty_factor_augmentation_cycles),
      Rcpp::Named("cuda_qr_svd_cycles") =
        static_cast<double>(diagnostics.cuda_qr_svd_cycles),
      Rcpp::Named("cuda_qr_bidiagonal_reduction_cycles") =
        static_cast<double>(
          diagnostics.cuda_qr_bidiagonal_reduction_cycles),
      Rcpp::Named("cuda_bidiagonal_svd_cycles") =
        static_cast<double>(diagnostics.cuda_bidiagonal_svd_cycles),
      Rcpp::Named("cuda_svd_vector_postback_cycles") =
        static_cast<double>(diagnostics.cuda_svd_vector_postback_cycles),
      Rcpp::Named("cuda_left_vector_product_cycles") =
        static_cast<double>(diagnostics.cuda_left_vector_product_cycles),
      Rcpp::Named("cuda_score_construction_cycles") =
        static_cast<double>(diagnostics.cuda_score_construction_cycles),
      Rcpp::Named("cuda_derivative_hessian_cycles") =
        static_cast<double>(diagnostics.cuda_derivative_hessian_cycles),
      Rcpp::Named("cuda_complete_evaluation_count") =
        diagnostics.cuda_complete_evaluation_count,
      Rcpp::Named("cuda_score_only_evaluation_count") =
        diagnostics.cuda_score_only_evaluation_count,
      Rcpp::Named("cuda_guarded_qr_evaluation_count") =
        diagnostics.cuda_guarded_qr_evaluation_count,
      Rcpp::Named("cuda_stable_svd_evaluation_count") =
        diagnostics.cuda_stable_svd_evaluation_count,
      Rcpp::Named("cuda_selected_evaluation_reuse_count") =
        diagnostics.cuda_selected_evaluation_reuse_count,
      Rcpp::Named("cuda_hessian_eigensolver_count") =
        diagnostics.cuda_hessian_eigensolver_count,
      Rcpp::Named("cuda_selected_fit_count") =
        diagnostics.cuda_selected_fit_count,
      Rcpp::Named("cuda_objective_target_count") =
        diagnostics.cuda_objective_target_count,
      Rcpp::Named("cpu_objective_count") = diagnostics.cpu_objective_count,
      Rcpp::Named("cpu_optimizer_count") = diagnostics.cpu_optimizer_count,
      Rcpp::Named("cpu_multi_penalty_solve_count") =
        diagnostics.cpu_multi_penalty_solve_count,
      Rcpp::Named("fallback_count") = diagnostics.fallback_count,
      Rcpp::Named("cuda_error_count") = diagnostics.cuda_error_count,
      Rcpp::Named("svd_nonconverged_count") =
        diagnostics.svd_nonconverged_count,
      Rcpp::Named("aggregate_rank_failure_count") =
        diagnostics.aggregate_rank_failure_count,
      Rcpp::Named("device_allocation_count") =
        diagnostics.device_allocation_count,
      Rcpp::Named("h2d_copy_count") = diagnostics.h2d_copy_count,
      Rcpp::Named("d2h_copy_count") = diagnostics.d2h_copy_count,
      Rcpp::Named("target_specific_log_sp") =
        diagnostics.target_specific_log_sp,
      Rcpp::Named("true_batched_kernel") = diagnostics.true_batched_kernel,
      Rcpp::Named("independent_target_states") =
        diagnostics.independent_target_states,
      Rcpp::Named("normal_equations_used") =
        diagnostics.normal_equations_used,
      Rcpp::Named("double_precision") = diagnostics.double_precision,
      Rcpp::Named("total_host_ms") = diagnostics.total_host_ms));
}

fastkpc::MultiPenaltyGcvCudaOptimizerControl
multi_penalty_cuda_optimizer_control(Rcpp::List values) {
  auto numeric = [&](const char* name, double fallback) {
    return values.containsElementNamed(name) ?
      Rcpp::as<double>(values[name]) : fallback;
  };
  auto integer = [&](const char* name, int fallback) {
    return values.containsElementNamed(name) ?
      Rcpp::as<int>(values[name]) : fallback;
  };
  fastkpc::MultiPenaltyGcvCudaOptimizerControl control;
  control.convergence_tolerance = numeric(
    "convergence_tolerance", control.convergence_tolerance);
  control.max_step_halving = integer(
    "max_step_halving", control.max_step_halving);
  control.max_iterations = integer("max_iterations", control.max_iterations);
  control.max_newton_step = numeric(
    "max_newton_step", control.max_newton_step);
  control.boundary_probe_step = numeric(
    "boundary_probe_step", control.boundary_probe_step);
  control.max_boundary_probes = integer(
    "max_boundary_probes", control.max_boundary_probes);
  control.rank_tolerance = numeric("rank_tolerance", control.rank_tolerance);
  return control;
}

Rcpp::List multi_penalty_cuda_batch_result_to_list(
    fastkpc::MultiPenaltyGcvCudaBatchResult result) {
  auto* residual_holder = new MultiPenaltyCudaResidualHolder{
    std::move(result.residual), fixed_sp_current_pid()
  };
  SEXP residual_ext = PROTECT(R_MakeExternalPtr(
    residual_holder, multi_penalty_cuda_residual_tag(), R_NilValue));
  R_RegisterCFinalizerEx(
    residual_ext, multi_penalty_cuda_residual_finalizer, TRUE);
  Rcpp::List output = Rcpp::List::create(
    Rcpp::Named("optimization") =
      multi_penalty_cuda_optimization_to_list(result.optimization),
    Rcpp::Named("residual") = residual_ext);
  UNPROTECT(1);
  return output;
}

}  // namespace

extern "C" SEXP C_full_cuda_ci_multi_penalty_gcv_optimize_cuda(
    SEXP Xs,
    SEXP Ys,
    SEXP magic_qr_packed_s,
    SEXP magic_tau_s,
    SEXP magic_r_s,
    SEXP magic_pivot_s,
    SEXP penalty_roots_s,
    SEXP penalty_matrices_s,
    SEXP penalty_ranks_s,
    SEXP initial_log_sp_s,
    SEXP control_s) {
  BEGIN_RCPP
  if (!Rf_isReal(Xs) || !Rf_isMatrix(Xs) ||
      !Rf_isReal(Ys) || !Rf_isMatrix(Ys) ||
      !Rf_isReal(magic_qr_packed_s) || !Rf_isMatrix(magic_qr_packed_s) ||
      !Rf_isReal(magic_tau_s) ||
      !Rf_isReal(magic_r_s) || !Rf_isMatrix(magic_r_s) ||
      !Rf_isInteger(magic_pivot_s) ||
      !Rf_isNewList(penalty_roots_s) ||
      !Rf_isNewList(penalty_matrices_s) ||
      !Rf_isInteger(penalty_ranks_s) ||
      !Rf_isReal(initial_log_sp_s) || !Rf_isNewList(control_s)) {
    Rcpp::stop(
      "multi-penalty CUDA optimizer inputs have invalid storage types");
  }
  Rcpp::NumericMatrix X(Xs);
  Rcpp::NumericMatrix Y(Ys);
  Rcpp::NumericMatrix qr_packed(magic_qr_packed_s);
  Rcpp::NumericVector tau(magic_tau_s);
  Rcpp::NumericMatrix magic_r(magic_r_s);
  Rcpp::IntegerVector pivot(magic_pivot_s);
  Rcpp::List penalty_roots(penalty_roots_s);
  Rcpp::List penalty_matrices(penalty_matrices_s);
  Rcpp::IntegerVector penalty_ranks(penalty_ranks_s);
  Rcpp::NumericVector initial_log_sp(initial_log_sp_s);
  Rcpp::List control_values(control_s);
  const int n = X.nrow();
  const int q = X.ncol();
  const int target_count = Y.ncol();
  const int penalty_count = penalty_roots.size();
  if (Y.nrow() != n || target_count <= 1 ||
      qr_packed.nrow() != n || qr_packed.ncol() != q ||
      tau.size() != q || magic_r.nrow() != q || magic_r.ncol() != q ||
      pivot.size() != q || penalty_count <= 1 ||
      penalty_matrices.size() != penalty_count ||
      penalty_ranks.size() != penalty_count ||
      initial_log_sp.size() != penalty_count) {
    Rcpp::stop(
      "multi-penalty CUDA optimizer input dimensions are inconsistent");
  }
  std::vector<int> pivot_zero_based(static_cast<std::size_t>(q));
  std::vector<unsigned char> pivot_seen(static_cast<std::size_t>(q), 0);
  for (int index = 0; index < q; ++index) {
    const int value = pivot[index] - 1;
    if (value < 0 || value >= q ||
        pivot_seen[static_cast<std::size_t>(value)] != 0) {
      Rcpp::stop("multi-penalty CUDA optimizer QR pivot is invalid");
    }
    pivot_seen[static_cast<std::size_t>(value)] = 1;
    pivot_zero_based[static_cast<std::size_t>(index)] = value;
  }
  std::vector<std::vector<double>> roots;
  std::vector<std::vector<double>> matrices;
  std::vector<int> ranks;
  roots.reserve(static_cast<std::size_t>(penalty_count));
  matrices.reserve(static_cast<std::size_t>(penalty_count));
  ranks.reserve(static_cast<std::size_t>(penalty_count));
  for (int penalty = 0; penalty < penalty_count; ++penalty) {
    SEXP root_s = penalty_roots[penalty];
    SEXP matrix_s = penalty_matrices[penalty];
    if (!Rf_isReal(root_s) || !Rf_isMatrix(root_s) ||
        !Rf_isReal(matrix_s) || !Rf_isMatrix(matrix_s)) {
      Rcpp::stop(
        "multi-penalty CUDA optimizer penalties must be matrices");
    }
    Rcpp::NumericMatrix root(root_s);
    Rcpp::NumericMatrix matrix(matrix_s);
    const int rank = penalty_ranks[penalty];
    if (root.nrow() != q || root.ncol() != rank || rank <= 0 ||
        matrix.nrow() != q || matrix.ncol() != q) {
      Rcpp::stop(
        "multi-penalty CUDA optimizer penalty dimensions are invalid");
    }
    roots.emplace_back(REAL(root_s), REAL(root_s) + XLENGTH(root_s));
    matrices.emplace_back(
      REAL(matrix_s), REAL(matrix_s) + XLENGTH(matrix_s));
    ranks.push_back(rank);
  }
  auto numeric_control = [&](const char* name, double fallback) {
    return control_values.containsElementNamed(name) ?
      Rcpp::as<double>(control_values[name]) : fallback;
  };
  auto integer_control = [&](const char* name, int fallback) {
    return control_values.containsElementNamed(name) ?
      Rcpp::as<int>(control_values[name]) : fallback;
  };
  fastkpc::MultiPenaltyGcvCudaOptimizerControl control;
  control.convergence_tolerance = numeric_control(
    "convergence_tolerance", control.convergence_tolerance);
  control.max_step_halving = integer_control(
    "max_step_halving", control.max_step_halving);
  control.max_iterations = integer_control(
    "max_iterations", control.max_iterations);
  control.max_newton_step = numeric_control(
    "max_newton_step", control.max_newton_step);
  control.boundary_probe_step = numeric_control(
    "boundary_probe_step", control.boundary_probe_step);
  control.max_boundary_probes = integer_control(
    "max_boundary_probes", control.max_boundary_probes);
  control.rank_tolerance = numeric_control(
    "rank_tolerance", control.rank_tolerance);

  const fastkpc::MultiPenaltyGcvCudaOptimization result =
    fastkpc::multi_penalty_gcv_optimize_cuda(
      REAL(Xs), REAL(Ys), REAL(magic_qr_packed_s), REAL(magic_tau_s),
      REAL(magic_r_s), pivot_zero_based.data(), roots, matrices, ranks,
      REAL(initial_log_sp_s), n, q, penalty_count, target_count, control);
  Rcpp::NumericMatrix selected_log_sp(
    result.penalty_count, result.target_count);
  std::copy(result.selected_log_sp.begin(), result.selected_log_sp.end(),
            selected_log_sp.begin());
  Rcpp::NumericMatrix gradient(result.penalty_count, result.target_count);
  std::copy(result.gradient.begin(), result.gradient.end(), gradient.begin());
  Rcpp::NumericVector hessian(result.hessian.begin(), result.hessian.end());
  hessian.attr("dim") = Rcpp::IntegerVector::create(
    result.penalty_count, result.penalty_count, result.target_count);
  Rcpp::NumericMatrix coefficients(
    result.coefficient_dim, result.target_count);
  std::copy(result.coefficients.begin(), result.coefficients.end(),
            coefficients.begin());
  Rcpp::NumericMatrix hessian_eigenvalues(
    result.penalty_count, result.target_count);
  std::copy(result.hessian_eigenvalues.begin(),
            result.hessian_eigenvalues.end(),
            hessian_eigenvalues.begin());
  Rcpp::CharacterMatrix boundary_status(
    result.penalty_count, result.target_count);
  Rcpp::LogicalVector fully_converged(result.target_count);
  Rcpp::LogicalVector hessian_positive_definite(result.target_count);
  Rcpp::LogicalVector step_failed(result.target_count);
  Rcpp::CharacterVector convergence_code(result.target_count);
  for (int target = 0; target < result.target_count; ++target) {
    fully_converged[target] = result.fully_converged[target] != 0;
    hessian_positive_definite[target] =
      result.hessian_positive_definite[target] != 0;
    step_failed[target] = result.step_failed[target] != 0;
    convergence_code[target] = result.optimizer_status[target] != 0 ?
      "cuda_optimizer_error" :
      (result.step_failed[target] != 0 ?
        "step_halving_exhausted" : "fully_converged");
    for (int penalty = 0; penalty < result.penalty_count; ++penalty) {
      const int value = result.boundary_status[
        penalty + result.penalty_count * target];
      boundary_status(penalty, target) = value == 0 ? "finite-interior" :
        (value == 1 ? "positive-boundary" :
          (value == 2 ? "negative-boundary" :
            (value == 3 ? "finite-after-boundary-probe" : "unresolved")));
    }
  }
  const fastkpc::MultiPenaltyGcvCudaDiagnostics& diagnostics =
    result.diagnostics;
  return Rcpp::List::create(
    Rcpp::Named("schema_version") = result.schema_version,
    Rcpp::Named("rank_path") = result.rank_path,
    Rcpp::Named("optimizer_path") = result.optimizer_path,
    Rcpp::Named("n") = result.n,
    Rcpp::Named("coefficient_dim") = result.coefficient_dim,
    Rcpp::Named("penalty_count") = result.penalty_count,
    Rcpp::Named("target_count") = result.target_count,
    Rcpp::Named("initial_log_sp") = Rcpp::wrap(result.initial_log_sp),
    Rcpp::Named("selected_log_sp") = selected_log_sp,
    Rcpp::Named("rss") = Rcpp::wrap(result.rss),
    Rcpp::Named("edf") = Rcpp::wrap(result.edf),
    Rcpp::Named("score") = Rcpp::wrap(result.score),
    Rcpp::Named("condition") = Rcpp::wrap(result.condition),
    Rcpp::Named("gradient") = gradient,
    Rcpp::Named("hessian") = hessian,
    Rcpp::Named("coefficients") = coefficients,
    Rcpp::Named("rms_gradient") = Rcpp::wrap(result.rms_gradient),
    Rcpp::Named("hessian_eigenvalues") = hessian_eigenvalues,
    Rcpp::Named("aggregate_penalty_rank") =
      Rcpp::wrap(result.aggregate_penalty_rank),
    Rcpp::Named("numerical_rank") = Rcpp::wrap(result.numerical_rank),
    Rcpp::Named("solver_info") = Rcpp::wrap(result.solver_info),
    Rcpp::Named("optimizer_iterations") =
      Rcpp::wrap(result.optimizer_iterations),
    Rcpp::Named("score_calls") = Rcpp::wrap(result.score_calls),
    Rcpp::Named("objective_calls") = Rcpp::wrap(result.objective_calls),
    Rcpp::Named("step_halving_count") =
      Rcpp::wrap(result.step_halving_count),
    Rcpp::Named("newton_trial_count") =
      Rcpp::wrap(result.newton_trial_count),
    Rcpp::Named("steepest_descent_trial_count") =
      Rcpp::wrap(result.steepest_descent_trial_count),
    Rcpp::Named("boundary_probe_count") =
      Rcpp::wrap(result.boundary_probe_count),
    Rcpp::Named("boundary_accepted_count") =
      Rcpp::wrap(result.boundary_accepted_count),
    Rcpp::Named("boundary_status") = boundary_status,
    Rcpp::Named("fully_converged") = fully_converged,
    Rcpp::Named("hessian_positive_definite") =
      hessian_positive_definite,
    Rcpp::Named("step_failed") = step_failed,
    Rcpp::Named("convergence_code") = convergence_code,
    Rcpp::Named("optimizer_status") = Rcpp::wrap(result.optimizer_status),
    Rcpp::Named("diagnostics") = Rcpp::List::create(
      Rcpp::Named("schema_version") = diagnostics.schema_version,
      Rcpp::Named("execution_strategy") = diagnostics.execution_strategy,
      Rcpp::Named("device_id") = diagnostics.device_id,
      Rcpp::Named("gpu_name") = diagnostics.gpu_name,
      Rcpp::Named("prepared_setup_upload_count") =
        diagnostics.prepared_setup_upload_count,
      Rcpp::Named("target_batch_upload_count") =
        diagnostics.target_batch_upload_count,
      Rcpp::Named("cuda_qt_y_kernel_launch_count") =
        diagnostics.cuda_qt_y_kernel_launch_count,
      Rcpp::Named("cuda_optimizer_kernel_launch_count") =
        diagnostics.cuda_optimizer_kernel_launch_count,
      Rcpp::Named("cuda_optimizer_target_count") =
        diagnostics.cuda_optimizer_target_count,
      Rcpp::Named("cuda_optimizer_objective_count") =
        diagnostics.cuda_optimizer_objective_count,
      Rcpp::Named("cuda_penalty_factor_augmentation_cycles") =
        static_cast<double>(
          diagnostics.cuda_penalty_factor_augmentation_cycles),
      Rcpp::Named("cuda_qr_svd_cycles") =
        static_cast<double>(diagnostics.cuda_qr_svd_cycles),
      Rcpp::Named("cuda_qr_bidiagonal_reduction_cycles") =
        static_cast<double>(
          diagnostics.cuda_qr_bidiagonal_reduction_cycles),
      Rcpp::Named("cuda_bidiagonal_svd_cycles") =
        static_cast<double>(diagnostics.cuda_bidiagonal_svd_cycles),
      Rcpp::Named("cuda_svd_vector_postback_cycles") =
        static_cast<double>(diagnostics.cuda_svd_vector_postback_cycles),
      Rcpp::Named("cuda_left_vector_product_cycles") =
        static_cast<double>(diagnostics.cuda_left_vector_product_cycles),
      Rcpp::Named("cuda_score_construction_cycles") =
        static_cast<double>(diagnostics.cuda_score_construction_cycles),
      Rcpp::Named("cuda_derivative_hessian_cycles") =
        static_cast<double>(diagnostics.cuda_derivative_hessian_cycles),
      Rcpp::Named("cuda_complete_evaluation_count") =
        diagnostics.cuda_complete_evaluation_count,
      Rcpp::Named("cuda_score_only_evaluation_count") =
        diagnostics.cuda_score_only_evaluation_count,
      Rcpp::Named("cuda_guarded_qr_evaluation_count") =
        diagnostics.cuda_guarded_qr_evaluation_count,
      Rcpp::Named("cuda_stable_svd_evaluation_count") =
        diagnostics.cuda_stable_svd_evaluation_count,
      Rcpp::Named("cuda_selected_evaluation_reuse_count") =
        diagnostics.cuda_selected_evaluation_reuse_count,
      Rcpp::Named("cuda_hessian_eigensolver_count") =
        diagnostics.cuda_hessian_eigensolver_count,
      Rcpp::Named("cuda_selected_fit_count") =
        diagnostics.cuda_selected_fit_count,
      Rcpp::Named("cuda_objective_target_count") =
        diagnostics.cuda_objective_target_count,
      Rcpp::Named("cpu_objective_count") = diagnostics.cpu_objective_count,
      Rcpp::Named("cpu_optimizer_count") = diagnostics.cpu_optimizer_count,
      Rcpp::Named("cpu_multi_penalty_solve_count") =
        diagnostics.cpu_multi_penalty_solve_count,
      Rcpp::Named("fallback_count") = diagnostics.fallback_count,
      Rcpp::Named("cuda_error_count") = diagnostics.cuda_error_count,
      Rcpp::Named("svd_nonconverged_count") =
        diagnostics.svd_nonconverged_count,
      Rcpp::Named("aggregate_rank_failure_count") =
        diagnostics.aggregate_rank_failure_count,
      Rcpp::Named("device_allocation_count") =
        diagnostics.device_allocation_count,
      Rcpp::Named("h2d_copy_count") = diagnostics.h2d_copy_count,
      Rcpp::Named("d2h_copy_count") = diagnostics.d2h_copy_count,
      Rcpp::Named("target_specific_log_sp") =
        diagnostics.target_specific_log_sp,
      Rcpp::Named("true_batched_kernel") = diagnostics.true_batched_kernel,
      Rcpp::Named("independent_target_states") =
        diagnostics.independent_target_states,
      Rcpp::Named("normal_equations_used") =
        diagnostics.normal_equations_used,
      Rcpp::Named("double_precision") = diagnostics.double_precision,
      Rcpp::Named("total_host_ms") = diagnostics.total_host_ms));
  END_RCPP
}

extern "C" SEXP C_full_cuda_ci_multi_penalty_gcv_prepared_create(
    SEXP Xs,
    SEXP magic_qr_packed_s,
    SEXP magic_tau_s,
    SEXP magic_r_s,
    SEXP magic_pivot_s,
    SEXP penalty_roots_s,
    SEXP penalty_matrices_s,
    SEXP penalty_ranks_s,
    SEXP initial_log_sp_s,
    SEXP target_capacity_s,
    SEXP device_id_s) {
  BEGIN_RCPP
  if (!Rf_isReal(Xs) || !Rf_isMatrix(Xs) ||
      !Rf_isReal(magic_qr_packed_s) || !Rf_isMatrix(magic_qr_packed_s) ||
      !Rf_isReal(magic_tau_s) ||
      !Rf_isReal(magic_r_s) || !Rf_isMatrix(magic_r_s) ||
      !Rf_isInteger(magic_pivot_s) ||
      !Rf_isNewList(penalty_roots_s) ||
      !Rf_isNewList(penalty_matrices_s) ||
      !Rf_isInteger(penalty_ranks_s) || !Rf_isReal(initial_log_sp_s)) {
    Rcpp::stop(
      "persistent multi-penalty CUDA setup has invalid storage types");
  }
  Rcpp::NumericMatrix X(Xs);
  Rcpp::NumericMatrix qr_packed(magic_qr_packed_s);
  Rcpp::NumericVector tau(magic_tau_s);
  Rcpp::NumericMatrix magic_r(magic_r_s);
  Rcpp::IntegerVector pivot(magic_pivot_s);
  Rcpp::List penalty_roots(penalty_roots_s);
  Rcpp::List penalty_matrices(penalty_matrices_s);
  Rcpp::IntegerVector penalty_ranks(penalty_ranks_s);
  Rcpp::NumericVector initial_log_sp(initial_log_sp_s);
  const int n = X.nrow();
  const int q = X.ncol();
  const int penalty_count = penalty_roots.size();
  const int target_capacity = Rcpp::as<int>(target_capacity_s);
  const int device_id = Rcpp::as<int>(device_id_s);
  if (n <= q || q <= 0 || qr_packed.nrow() != n ||
      qr_packed.ncol() != q || tau.size() != q ||
      magic_r.nrow() != q || magic_r.ncol() != q || pivot.size() != q ||
      penalty_count <= 1 || penalty_matrices.size() != penalty_count ||
      penalty_ranks.size() != penalty_count ||
      initial_log_sp.size() != penalty_count || target_capacity <= 1 ||
      !all_finite(X) || !all_finite(qr_packed) ||
      !all_finite(magic_r) || !all_finite_vector(tau) ||
      !all_finite_vector(initial_log_sp)) {
    Rcpp::stop(
      "persistent multi-penalty CUDA setup dimensions are inconsistent");
  }
  std::vector<int> pivot_zero_based(static_cast<std::size_t>(q));
  std::vector<unsigned char> pivot_seen(static_cast<std::size_t>(q), 0);
  for (int index = 0; index < q; ++index) {
    const int value = pivot[index] - 1;
    if (value < 0 || value >= q ||
        pivot_seen[static_cast<std::size_t>(value)] != 0) {
      Rcpp::stop("persistent multi-penalty CUDA QR pivot is invalid");
    }
    pivot_seen[static_cast<std::size_t>(value)] = 1;
    pivot_zero_based[static_cast<std::size_t>(index)] = value;
  }
  std::vector<std::vector<double>> roots;
  std::vector<std::vector<double>> matrices;
  std::vector<int> ranks;
  roots.reserve(static_cast<std::size_t>(penalty_count));
  matrices.reserve(static_cast<std::size_t>(penalty_count));
  ranks.reserve(static_cast<std::size_t>(penalty_count));
  for (int penalty = 0; penalty < penalty_count; ++penalty) {
    SEXP root_s = penalty_roots[penalty];
    SEXP matrix_s = penalty_matrices[penalty];
    if (!Rf_isReal(root_s) || !Rf_isMatrix(root_s) ||
        !Rf_isReal(matrix_s) || !Rf_isMatrix(matrix_s)) {
      Rcpp::stop(
        "persistent multi-penalty CUDA penalties must be matrices");
    }
    Rcpp::NumericMatrix root(root_s);
    Rcpp::NumericMatrix matrix(matrix_s);
    const int rank = penalty_ranks[penalty];
    if (root.nrow() != q || root.ncol() != rank || rank <= 0 ||
        matrix.nrow() != q || matrix.ncol() != q ||
        !all_finite(root) || !all_finite(matrix)) {
      Rcpp::stop(
        "persistent multi-penalty CUDA penalty geometry is invalid");
    }
    roots.emplace_back(REAL(root_s), REAL(root_s) + XLENGTH(root_s));
    matrices.emplace_back(
      REAL(matrix_s), REAL(matrix_s) + XLENGTH(matrix_s));
    ranks.push_back(rank);
  }
  auto* holder = new MultiPenaltyCudaPreparedHolder{
    std::shared_ptr<fastkpc::MultiPenaltyGcvCudaPrepared>(),
    fixed_sp_current_pid()
  };
  SEXP ext = PROTECT(R_MakeExternalPtr(
    holder, multi_penalty_cuda_prepared_tag(), R_NilValue));
  R_RegisterCFinalizerEx(
    ext, multi_penalty_cuda_prepared_finalizer, TRUE);
  try {
    holder->value = fastkpc::create_multi_penalty_gcv_cuda_prepared(
      REAL(Xs), REAL(magic_qr_packed_s), REAL(magic_tau_s),
      REAL(magic_r_s), pivot_zero_based.data(), roots, matrices, ranks,
      REAL(initial_log_sp_s), n, q, penalty_count, target_capacity,
      device_id);
  } catch (...) {
    delete holder;
    R_ClearExternalPtr(ext);
    UNPROTECT(1);
    throw;
  }
  UNPROTECT(1);
  return ext;
  END_RCPP
}

extern "C" SEXP C_full_cuda_ci_multi_penalty_gcv_prepared_info(
    SEXP prepared_s) {
  BEGIN_RCPP
  MultiPenaltyCudaPreparedHolder* holder =
    multi_penalty_cuda_prepared_holder(prepared_s, true);
  const fastkpc::MultiPenaltyGcvCudaPreparedInfo info =
    fastkpc::multi_penalty_gcv_cuda_prepared_info(holder->value);
  return Rcpp::List::create(
    Rcpp::Named("device_id") = info.device_id,
    Rcpp::Named("gpu_name") = info.gpu_name,
    Rcpp::Named("n") = info.n,
    Rcpp::Named("coefficient_dim") = info.coefficient_dim,
    Rcpp::Named("penalty_count") = info.penalty_count,
    Rcpp::Named("target_capacity") = info.target_capacity,
    Rcpp::Named("setup_upload_count") = info.setup_upload_count,
    Rcpp::Named("setup_h2d_bytes") =
      static_cast<double>(info.setup_h2d_bytes),
    Rcpp::Named("device_allocation_count") = info.device_allocation_count,
    Rcpp::Named("workspace_grow_count") = info.workspace_grow_count,
    Rcpp::Named("solve_count") = info.solve_count,
    Rcpp::Named("cublas_gemm_count") = info.cublas_gemm_count,
    Rcpp::Named("residual_kernel_count") = info.residual_kernel_count,
    Rcpp::Named("residual_shadow_d2h_count") =
      info.residual_shadow_d2h_count,
    Rcpp::Named("residual_shadow_d2h_bytes") =
      static_cast<double>(info.residual_shadow_d2h_bytes),
    Rcpp::Named("residual_slot_leased") = info.residual_slot_leased,
    Rcpp::Named("generation") = static_cast<double>(info.generation));
  END_RCPP
}

extern "C" SEXP C_full_cuda_ci_multi_penalty_gcv_prepared_free(
    SEXP prepared_s) {
  BEGIN_RCPP
  MultiPenaltyCudaPreparedHolder* holder =
    multi_penalty_cuda_prepared_holder(prepared_s, false);
  holder->value.reset();
  return R_NilValue;
  END_RCPP
}

extern "C" SEXP C_full_cuda_ci_multi_penalty_gcv_optimize_batch(
    SEXP prepared_s,
    SEXP Ys,
    SEXP target_keys_s,
    SEXP control_s) {
  BEGIN_RCPP
  MultiPenaltyCudaPreparedHolder* prepared_holder =
    multi_penalty_cuda_prepared_holder(prepared_s, true);
  if (!Rf_isReal(Ys) || !Rf_isMatrix(Ys) ||
      !Rf_isString(target_keys_s) || !Rf_isNewList(control_s)) {
    Rcpp::stop(
      "persistent multi-penalty CUDA batch has invalid storage types");
  }
  Rcpp::NumericMatrix Y(Ys);
  Rcpp::CharacterVector target_keys_r(target_keys_s);
  const fastkpc::MultiPenaltyGcvCudaPreparedInfo prepared_info =
    fastkpc::multi_penalty_gcv_cuda_prepared_info(prepared_holder->value);
  const int target_count = Y.ncol();
  if (Y.nrow() != prepared_info.n || target_count <= 1 ||
      target_keys_r.size() != target_count || !all_finite(Y)) {
    Rcpp::stop(
      "persistent multi-penalty CUDA batch dimensions are inconsistent");
  }
  std::vector<std::string> target_keys;
  target_keys.reserve(static_cast<std::size_t>(target_count));
  for (int target = 0; target < target_count; ++target) {
    if (target_keys_r[target] == NA_STRING) {
      Rcpp::stop("persistent multi-penalty target keys must not be missing");
    }
    const std::string key = Rcpp::as<std::string>(target_keys_r[target]);
    if (key.empty()) {
      Rcpp::stop("persistent multi-penalty target keys must be nonempty");
    }
    target_keys.push_back(key);
  }
  const fastkpc::MultiPenaltyGcvCudaOptimizerControl control =
    multi_penalty_cuda_optimizer_control(Rcpp::List(control_s));
  fastkpc::MultiPenaltyGcvCudaBatchResult result =
    fastkpc::multi_penalty_gcv_cuda_optimize_batch(
      prepared_holder->value, REAL(Ys), Y.nrow(), target_count,
      target_keys, control);
  return multi_penalty_cuda_batch_result_to_list(std::move(result));
  END_RCPP
}

extern "C" SEXP C_full_cuda_ci_multi_penalty_gcv_optimize_multi(
    SEXP requests_s,
    SEXP concurrency_s,
    SEXP control_s) {
  BEGIN_RCPP
  if (!Rf_isNewList(requests_s) || !Rf_isNewList(control_s)) {
    Rcpp::stop(
      "multi-setup multi-penalty CUDA inputs have invalid storage types");
  }
  Rcpp::List request_values(requests_s);
  const int concurrency = Rcpp::as<int>(concurrency_s);
  if (request_values.size() <= 0 || concurrency <= 0 ||
      concurrency > fastkpc::kMultiPenaltyGcvMaximumConcurrentSetups) {
    Rcpp::stop(
      "multi-setup multi-penalty CUDA concurrency must be in [1, 32]");
  }
  const fastkpc::MultiPenaltyGcvCudaOptimizerControl control =
    multi_penalty_cuda_optimizer_control(Rcpp::List(control_s));
  std::vector<fastkpc::MultiPenaltyGcvCudaMultiRequest> requests;
  requests.reserve(static_cast<std::size_t>(request_values.size()));
  std::vector<MultiPenaltyCudaPreparedHolder*> observed_holders;
  observed_holders.reserve(static_cast<std::size_t>(request_values.size()));
  for (R_xlen_t index = 0; index < request_values.size(); ++index) {
    SEXP request_s = request_values[index];
    if (!Rf_isNewList(request_s)) {
      Rcpp::stop("each multi-setup multi-penalty request must be a list");
    }
    Rcpp::List request(request_s);
    if (!request.containsElementNamed("handle") ||
        !request.containsElementNamed("Y") ||
        !request.containsElementNamed("target_keys")) {
      Rcpp::stop("multi-setup multi-penalty request fields are incomplete");
    }
    SEXP prepared_s = request["handle"];
    SEXP Ys = request["Y"];
    SEXP target_keys_s = request["target_keys"];
    MultiPenaltyCudaPreparedHolder* prepared_holder =
      multi_penalty_cuda_prepared_holder(prepared_s, true);
    if (std::find(
          observed_holders.begin(), observed_holders.end(),
          prepared_holder) != observed_holders.end()) {
      Rcpp::stop(
        "multi-setup multi-penalty requests must use distinct handles");
    }
    observed_holders.push_back(prepared_holder);
    if (!Rf_isReal(Ys) || !Rf_isMatrix(Ys) ||
        !Rf_isString(target_keys_s)) {
      Rcpp::stop(
        "multi-setup multi-penalty request payload types are invalid");
    }
    Rcpp::NumericMatrix Y(Ys);
    Rcpp::CharacterVector target_keys_r(target_keys_s);
    const fastkpc::MultiPenaltyGcvCudaPreparedInfo prepared_info =
      fastkpc::multi_penalty_gcv_cuda_prepared_info(prepared_holder->value);
    const int target_count = Y.ncol();
    if (prepared_info.residual_slot_leased || Y.nrow() != prepared_info.n ||
        target_count <= 1 || target_count > prepared_info.target_capacity ||
        target_keys_r.size() != target_count || !all_finite(Y)) {
      Rcpp::stop(
        "multi-setup multi-penalty request dimensions are inconsistent");
    }
    fastkpc::MultiPenaltyGcvCudaMultiRequest value;
    value.prepared = prepared_holder->value;
    value.Y.assign(REAL(Ys), REAL(Ys) + XLENGTH(Ys));
    value.n = Y.nrow();
    value.target_count = target_count;
    value.target_keys.reserve(static_cast<std::size_t>(target_count));
    for (int target = 0; target < target_count; ++target) {
      if (target_keys_r[target] == NA_STRING) {
        Rcpp::stop("multi-setup multi-penalty target keys must not be missing");
      }
      const std::string key = Rcpp::as<std::string>(target_keys_r[target]);
      if (key.empty()) {
        Rcpp::stop("multi-setup multi-penalty target keys must be nonempty");
      }
      value.target_keys.push_back(key);
    }
    value.control = control;
    requests.push_back(std::move(value));
  }

  fastkpc::MultiPenaltyGcvCudaMultiResult result =
    fastkpc::multi_penalty_gcv_cuda_optimize_multi(
      std::move(requests), concurrency);
  Rcpp::List setups(result.setups.size());
  for (std::size_t index = 0; index < result.setups.size(); ++index) {
    setups[index] = multi_penalty_cuda_batch_result_to_list(
      std::move(result.setups[index]));
  }
  const fastkpc::MultiPenaltyGcvCudaMultiDiagnostics& diagnostics =
    result.diagnostics;
  return Rcpp::List::create(
    Rcpp::Named("schema_version") = result.schema_version,
    Rcpp::Named("setups") = setups,
    Rcpp::Named("diagnostics") = Rcpp::List::create(
      Rcpp::Named("schema_version") = diagnostics.schema_version,
      Rcpp::Named("execution_strategy") = diagnostics.execution_strategy,
      Rcpp::Named("setup_count") = diagnostics.setup_count,
      Rcpp::Named("target_count") = diagnostics.target_count,
      Rcpp::Named("requested_concurrency") =
        diagnostics.requested_concurrency,
      Rcpp::Named("worker_count") = diagnostics.worker_count,
      Rcpp::Named("max_host_calls_in_flight") =
        diagnostics.max_host_calls_in_flight,
      Rcpp::Named("setup_stream_count") = diagnostics.setup_stream_count,
      Rcpp::Named("summed_setup_host_ms") =
        diagnostics.summed_setup_host_ms,
      Rcpp::Named("wall_host_ms") = diagnostics.wall_host_ms,
      Rcpp::Named("host_overlap_factor") = diagnostics.host_overlap_factor));
  END_RCPP
}

extern "C" SEXP C_full_cuda_ci_multi_penalty_gcv_residual_info(
    SEXP residual_s) {
  BEGIN_RCPP
  MultiPenaltyCudaResidualHolder* holder =
    multi_penalty_cuda_residual_holder(residual_s, true);
  const fastkpc::MultiPenaltyGcvCudaResidualInfo info =
    fastkpc::multi_penalty_gcv_cuda_residual_info(holder->value);
  return Rcpp::List::create(
    Rcpp::Named("n") = info.n,
    Rcpp::Named("coefficient_dim") = info.coefficient_dim,
    Rcpp::Named("target_count") = info.target_count,
    Rcpp::Named("device_id") = info.device_id,
    Rcpp::Named("target_keys") = Rcpp::wrap(info.target_keys),
    Rcpp::Named("optimizer_status") = Rcpp::wrap(info.optimizer_status),
    Rcpp::Named("device_resident") = info.device_resident,
    Rcpp::Named("released") = info.released,
    Rcpp::Named("generation") = static_cast<double>(info.generation));
  END_RCPP
}

extern "C" SEXP C_full_cuda_ci_multi_penalty_gcv_residual_shadow(
    SEXP residual_s) {
  BEGIN_RCPP
  MultiPenaltyCudaResidualHolder* holder =
    multi_penalty_cuda_residual_holder(residual_s, true);
  const fastkpc::MultiPenaltyGcvCudaResidualShadow shadow =
    fastkpc::materialize_multi_penalty_gcv_cuda_residual_shadow(
      holder->value);
  Rcpp::NumericMatrix coefficients(
    shadow.coefficient_dim, shadow.target_count);
  Rcpp::NumericMatrix fitted(shadow.n, shadow.target_count);
  Rcpp::NumericMatrix residuals(shadow.n, shadow.target_count);
  std::copy(shadow.coefficients.begin(), shadow.coefficients.end(),
            coefficients.begin());
  std::copy(shadow.fitted.begin(), shadow.fitted.end(), fitted.begin());
  std::copy(shadow.residuals.begin(), shadow.residuals.end(),
            residuals.begin());
  return Rcpp::List::create(
    Rcpp::Named("coefficients") = coefficients,
    Rcpp::Named("fitted") = fitted,
    Rcpp::Named("residuals") = residuals);
  END_RCPP
}

extern "C" SEXP C_full_cuda_ci_multi_penalty_gcv_residual_release(
    SEXP residual_s) {
  BEGIN_RCPP
  MultiPenaltyCudaResidualHolder* holder =
    multi_penalty_cuda_residual_holder(residual_s, true);
  fastkpc::release_multi_penalty_gcv_cuda_residual(holder->value);
  return R_NilValue;
  END_RCPP
}

extern "C" SEXP C_full_cuda_ci_multi_penalty_gcv_residual_free(
    SEXP residual_s) {
  BEGIN_RCPP
  MultiPenaltyCudaResidualHolder* holder =
    multi_penalty_cuda_residual_holder(residual_s, false);
  if (holder == nullptr) return R_NilValue;
  if (holder->value) {
    fastkpc::release_multi_penalty_gcv_cuda_residual(holder->value);
    holder->value.reset();
  }
  delete holder;
  R_ClearExternalPtr(residual_s);
  return R_NilValue;
  END_RCPP
}

extern "C" SEXP C_full_cuda_ci_single_penalty_mroot_cuda(
    SEXP penalty_matrix_s,
    SEXP penalty_rank_s,
    SEXP log_sp_s) {
  BEGIN_RCPP
  if (!Rf_isReal(penalty_matrix_s) || !Rf_isMatrix(penalty_matrix_s)) {
    Rcpp::stop("penalty_matrix must be a numeric matrix");
  }
  if (!Rf_isReal(log_sp_s)) {
    Rcpp::stop("log_sp must be numeric");
  }
  Rcpp::NumericMatrix penalty_matrix(penalty_matrix_s);
  Rcpp::NumericVector log_sp(log_sp_s);
  const int coefficient_dim = penalty_matrix.nrow();
  const int penalty_rank = Rcpp::as<int>(penalty_rank_s);
  if (coefficient_dim <= 0 || penalty_matrix.ncol() != coefficient_dim ||
      penalty_rank <= 0 || penalty_rank > coefficient_dim) {
    Rcpp::stop("dynamic mroot dimensions are invalid");
  }
  if (log_sp.size() <= 0 || !all_finite(penalty_matrix) ||
      !all_finite_vector(log_sp)) {
    Rcpp::stop("dynamic mroot inputs must be finite and non-empty");
  }
  for (int column = 0; column < coefficient_dim; ++column) {
    for (int row = 0; row < column; ++row) {
      if (penalty_matrix(row, column) != penalty_matrix(column, row)) {
        Rcpp::stop("penalty_matrix must be exactly symmetric");
      }
    }
  }
  const std::vector<double> candidates(log_sp.begin(), log_sp.end());
  const fastkpc::SinglePenaltyMrootCudaResult result =
    fastkpc::single_penalty_mroot_cuda(
      REAL(penalty_matrix_s), coefficient_dim, penalty_rank, candidates);
  Rcpp::NumericVector roots = Rcpp::wrap(result.roots);
  roots.attr("dim") = Rcpp::IntegerVector::create(
    coefficient_dim, penalty_rank, result.candidate_count);
  Rcpp::IntegerVector pivots = Rcpp::wrap(result.pivots);
  pivots.attr("dim") = Rcpp::IntegerVector::create(
    coefficient_dim, result.candidate_count);
  return Rcpp::List::create(
    Rcpp::Named("root") = roots,
    Rcpp::Named("rank") = Rcpp::wrap(result.ranks),
    Rcpp::Named("pivot") = pivots,
    Rcpp::Named("log_sp") = Rcpp::clone(log_sp));
  END_RCPP
}

extern "C" SEXP C_full_cuda_ci_single_penalty_gcv_cuda(
    SEXP Xs,
    SEXP Ys,
    SEXP rhs_transform_s,
    SEXP eigenvalues_s,
    SEXP magic_qr_packed_s,
    SEXP magic_tau_s,
    SEXP magic_r_s,
    SEXP magic_penalty_root_s,
    SEXP magic_penalty_matrix_s,
    SEXP target_ids_s,
    SEXP penalty_rank_s,
    SEXP initial_sp_s,
    SEXP sp_grid_s,
    SEXP materialize_grid_s,
    SEXP keep_transcript_s) {
  BEGIN_RCPP
  if (!Rf_isReal(Xs) || !Rf_isMatrix(Xs)) {
    Rcpp::stop("X must be a numeric matrix");
  }
  if (!Rf_isReal(Ys) || !Rf_isMatrix(Ys)) {
    Rcpp::stop("Y must be a numeric matrix");
  }
  if (!Rf_isReal(rhs_transform_s) || !Rf_isMatrix(rhs_transform_s)) {
    Rcpp::stop("rhs_transform must be a numeric matrix");
  }
  if (!Rf_isReal(magic_qr_packed_s) ||
      !Rf_isMatrix(magic_qr_packed_s) ||
      !Rf_isReal(magic_tau_s) ||
      !Rf_isReal(magic_r_s) || !Rf_isMatrix(magic_r_s) ||
      !Rf_isReal(magic_penalty_root_s) ||
      !Rf_isMatrix(magic_penalty_root_s) ||
      !Rf_isReal(magic_penalty_matrix_s) ||
      !Rf_isMatrix(magic_penalty_matrix_s)) {
    Rcpp::stop("mgcv-compatible QR/root inputs must be numeric matrices");
  }
  if (!Rf_isReal(eigenvalues_s)) {
    Rcpp::stop("eigenvalues must be numeric");
  }
  if (!Rf_isReal(sp_grid_s)) {
    Rcpp::stop("sp_grid must be numeric");
  }
  Rcpp::NumericMatrix X(Xs);
  Rcpp::NumericMatrix Y(Ys);
  Rcpp::NumericMatrix rhs_transform(rhs_transform_s);
  Rcpp::NumericMatrix magic_qr_packed(magic_qr_packed_s);
  Rcpp::NumericVector magic_tau(magic_tau_s);
  Rcpp::NumericMatrix magic_r(magic_r_s);
  Rcpp::NumericMatrix magic_penalty_root(magic_penalty_root_s);
  Rcpp::NumericMatrix magic_penalty_matrix(magic_penalty_matrix_s);
  Rcpp::IntegerVector target_ids(target_ids_s);
  Rcpp::NumericVector eigenvalues(eigenvalues_s);
  Rcpp::NumericVector sp_grid(sp_grid_s);
  const int n = X.nrow();
  const int p = X.ncol();
  const int target_count = Y.ncol();
  const int penalty_rank = Rcpp::as<int>(penalty_rank_s);
  const double initial_sp = Rcpp::as<double>(initial_sp_s);
  const bool materialize_grid = Rcpp::as<bool>(materialize_grid_s);
  const bool keep_transcript = Rcpp::as<bool>(keep_transcript_s);
  if (Y.nrow() != n || target_count <= 0) {
    Rcpp::stop("Y must have nrow(X) rows and at least one target");
  }
  if (rhs_transform.nrow() != p || rhs_transform.ncol() != p) {
    Rcpp::stop("rhs_transform dimensions must match ncol(X)");
  }
  if (magic_qr_packed.nrow() != n || magic_qr_packed.ncol() != p ||
      magic_tau.size() != p ||
      magic_r.nrow() != p || magic_r.ncol() != p ||
      magic_penalty_root.nrow() != p ||
      magic_penalty_root.ncol() != penalty_rank ||
      magic_penalty_matrix.nrow() != p || magic_penalty_matrix.ncol() != p) {
    Rcpp::stop("mgcv-compatible QR/root dimensions are invalid");
  }
  if (target_ids.size() != target_count) {
    Rcpp::stop("target_ids must match the GCV target count");
  }
  std::unordered_set<int> unique_target_ids;
  for (int value : target_ids) {
    if (value < 1 || value > 64 || !unique_target_ids.insert(value).second) {
      Rcpp::stop("target_ids must be unique integers in [1, 64]");
    }
  }
  if (eigenvalues.size() != p) {
    Rcpp::stop("length(eigenvalues) must equal ncol(X)");
  }
  if (!all_finite(X) || !all_finite(Y) ||
      !all_finite(rhs_transform) || !all_finite(magic_qr_packed) ||
      !all_finite_vector(magic_tau) ||
      !all_finite(magic_r) || !all_finite(magic_penalty_root) ||
      !all_finite(magic_penalty_matrix) ||
      !all_finite_vector(eigenvalues) ||
      !all_finite_vector(sp_grid)) {
    Rcpp::stop("single-penalty GCV inputs must be finite");
  }
  int positive_eigenvalues = 0;
  int zero_eigenvalues = 0;
  for (double value : eigenvalues) {
    if (value < 0.0) Rcpp::stop("eigenvalues must be non-negative");
    if (value == 0.0) {
      ++zero_eigenvalues;
    } else {
      ++positive_eigenvalues;
    }
  }
  if (positive_eigenvalues != penalty_rank ||
      zero_eigenvalues != p - penalty_rank) {
    Rcpp::stop(
      "rank-aware eigenvalue zeros disagree with penalty_rank");
  }
  std::vector<double> grid_values(sp_grid.begin(), sp_grid.end());
  const fastkpc::SinglePenaltyGcvCudaResult result =
    fastkpc::single_penalty_gcv_cuda(
      REAL(Xs), REAL(Ys), REAL(rhs_transform_s), REAL(eigenvalues_s),
      REAL(magic_qr_packed_s), REAL(magic_tau_s), REAL(magic_r_s),
      REAL(magic_penalty_root_s), REAL(magic_penalty_matrix_s),
      INTEGER(target_ids_s),
      n, p, target_count, penalty_rank, initial_sp, grid_values,
      materialize_grid, keep_transcript);

  Rcpp::NumericVector selected_sp(target_count);
  Rcpp::NumericVector selected_log_sp(target_count);
  Rcpp::NumericVector selected_score(target_count);
  Rcpp::NumericVector selected_edf(target_count);
  Rcpp::NumericVector selected_rss(target_count);
  Rcpp::NumericVector selected_gradient(target_count);
  Rcpp::NumericVector selected_hessian(target_count);
  Rcpp::NumericVector reported_rms_gradient(target_count);
  Rcpp::NumericVector pre_boundary_log_sp(target_count);
  Rcpp::IntegerVector iteration_count(target_count);
  Rcpp::IntegerVector score_call_count(target_count);
  Rcpp::IntegerVector actual_objective_call_count(target_count);
  Rcpp::LogicalVector fully_converged(target_count);
  Rcpp::LogicalVector hessian_positive_definite(target_count);
  Rcpp::IntegerVector boundary_probe_count(target_count);
  Rcpp::IntegerVector boundary_accepted_count(target_count);
  Rcpp::CharacterVector termination_reason(target_count);
  Rcpp::CharacterVector boundary_status(target_count);
  for (int target = 0; target < target_count; ++target) {
    const fastkpc::SinglePenaltyGcvOptimizerResult& value =
      result.targets[static_cast<std::size_t>(target)];
    selected_sp[target] = value.sp;
    selected_log_sp[target] = value.log_sp;
    selected_score[target] = value.score;
    selected_edf[target] = value.edf;
    selected_rss[target] = value.rss;
    selected_gradient[target] = value.gradient;
    selected_hessian[target] = value.hessian;
    reported_rms_gradient[target] = value.reported_rms_gradient;
    pre_boundary_log_sp[target] = value.pre_boundary_log_sp;
    iteration_count[target] = value.iteration_count;
    score_call_count[target] = value.score_call_count;
    actual_objective_call_count[target] =
      value.actual_objective_call_count;
    fully_converged[target] = value.fully_converged != 0;
    hessian_positive_definite[target] =
      value.hessian_positive_definite != 0;
    boundary_probe_count[target] = value.boundary_probe_count;
    boundary_accepted_count[target] = value.boundary_accepted_count;
    termination_reason[target] =
      fastkpc::single_penalty_gcv_termination_name(value.termination);
    boundary_status[target] = value.boundary_accepted_count > 0 ?
      "mgcv_infinity_probe_accepted" : "finite_refinement";
  }
  Rcpp::DataFrame targets = Rcpp::DataFrame::create(
    Rcpp::Named("target_position") = Rcpp::seq(1, target_count),
    Rcpp::Named("sp") = selected_sp,
    Rcpp::Named("log_sp") = selected_log_sp,
    Rcpp::Named("score") = selected_score,
    Rcpp::Named("edf") = selected_edf,
    Rcpp::Named("rss") = selected_rss,
    Rcpp::Named("gradient") = selected_gradient,
    Rcpp::Named("hessian") = selected_hessian,
    Rcpp::Named("reported_rms_gradient") = reported_rms_gradient,
    Rcpp::Named("pre_boundary_log_sp") = pre_boundary_log_sp,
    Rcpp::Named("iteration_count") = iteration_count,
    Rcpp::Named("score_call_count") = score_call_count,
    Rcpp::Named("actual_objective_call_count") =
      actual_objective_call_count,
    Rcpp::Named("fully_converged") = fully_converged,
    Rcpp::Named("hessian_positive_definite") =
      hessian_positive_definite,
    Rcpp::Named("boundary_probe_count") = boundary_probe_count,
    Rcpp::Named("boundary_accepted_count") = boundary_accepted_count,
    Rcpp::Named("termination_reason") = termination_reason,
    Rcpp::Named("boundary_status") = boundary_status,
    Rcpp::Named("stringsAsFactors") = false
  );

  Rcpp::List grid = R_NilValue;
  if (materialize_grid) {
    const int candidate_count = static_cast<int>(grid_values.size());
    Rcpp::NumericMatrix rss(target_count, candidate_count);
    Rcpp::NumericMatrix edf(target_count, candidate_count);
    Rcpp::NumericMatrix score(target_count, candidate_count);
    Rcpp::LogicalMatrix valid(target_count, candidate_count);
    for (int candidate = 0; candidate < candidate_count; ++candidate) {
      for (int target = 0; target < target_count; ++target) {
        const std::size_t index = static_cast<std::size_t>(target) +
          static_cast<std::size_t>(target_count) * candidate;
        const fastkpc::SinglePenaltyGcvGridCell& value = result.grid[index];
        rss(target, candidate) = value.rss;
        edf(target, candidate) = value.edf;
        score(target, candidate) = value.score;
        valid(target, candidate) = value.valid != 0;
      }
    }
    grid = Rcpp::List::create(
      Rcpp::Named("sp") = sp_grid,
      Rcpp::Named("rss") = rss,
      Rcpp::Named("edf") = edf,
      Rcpp::Named("score") = score,
      Rcpp::Named("valid") = valid
    );
  }

  Rcpp::List transcripts(target_count);
  if (keep_transcript) {
    for (int target = 0; target < target_count; ++target) {
      const int count = result.transcript_counts[
        static_cast<std::size_t>(target)];
      Rcpp::CharacterVector stage(count);
      Rcpp::IntegerVector iteration(count);
      Rcpp::IntegerVector evaluation(count);
      Rcpp::NumericVector current_log_sp(count);
      Rcpp::NumericVector proposed_step(count);
      Rcpp::NumericVector trial_log_sp(count);
      Rcpp::NumericVector objective(count);
      Rcpp::NumericVector gradient(count);
      Rcpp::NumericVector hessian(count);
      Rcpp::LogicalVector accepted(count);
      Rcpp::CharacterVector step_source(count);
      for (int row = 0; row < count; ++row) {
        const std::size_t index = static_cast<std::size_t>(target) *
          fastkpc::kSinglePenaltyGcvTranscriptCapacity + row;
        const fastkpc::SinglePenaltyGcvTranscriptEntry& value =
          result.transcript[index];
        stage[row] = fastkpc::single_penalty_gcv_transcript_stage_name(
          value.stage);
        iteration[row] = value.iteration;
        evaluation[row] = value.evaluation;
        current_log_sp[row] = value.current_log_sp;
        proposed_step[row] = value.proposed_step;
        trial_log_sp[row] = value.trial_log_sp;
        objective[row] = value.objective;
        gradient[row] = value.gradient;
        hessian[row] = value.hessian;
        accepted[row] = value.accepted != 0;
        step_source[row] = fastkpc::single_penalty_gcv_step_source_name(
          value.step_source);
      }
      transcripts[target] = Rcpp::DataFrame::create(
        Rcpp::Named("stage") = stage,
        Rcpp::Named("iteration") = iteration,
        Rcpp::Named("evaluation") = evaluation,
        Rcpp::Named("current_log_sp") = current_log_sp,
        Rcpp::Named("proposed_step") = proposed_step,
        Rcpp::Named("trial_log_sp") = trial_log_sp,
        Rcpp::Named("objective") = objective,
        Rcpp::Named("gradient") = gradient,
        Rcpp::Named("hessian") = hessian,
        Rcpp::Named("accepted") = accepted,
        Rcpp::Named("step_source") = step_source,
        Rcpp::Named("stringsAsFactors") = false
      );
    }
  }

  const fastkpc::SinglePenaltyGcvCudaDiagnostics& diagnostics =
    result.diagnostics;
  return Rcpp::List::create(
    Rcpp::Named("schema_version") = result.schema_version,
    Rcpp::Named("targets") = targets,
    Rcpp::Named("grid") = grid,
    Rcpp::Named("transcripts") = transcripts,
    Rcpp::Named("transcript_overflow") = result.transcript_overflow,
    Rcpp::Named("diagnostics") = Rcpp::List::create(
      Rcpp::Named("schema_version") = diagnostics.schema_version,
      Rcpp::Named("sp_selection_backend_executed") =
        diagnostics.sp_selection_backend_executed,
      Rcpp::Named("gcv_score_backend_executed") =
        diagnostics.gcv_score_backend_executed,
      Rcpp::Named("optimizer_backend_executed") =
        diagnostics.optimizer_backend_executed,
      Rcpp::Named("exact_replay_backend_executed") =
        diagnostics.exact_replay_backend_executed,
      Rcpp::Named("device_id") = diagnostics.device_id,
      Rcpp::Named("gpu_name") = diagnostics.gpu_name,
      Rcpp::Named("n") = diagnostics.n,
      Rcpp::Named("coefficient_dim") = diagnostics.coefficient_dim,
      Rcpp::Named("target_count") = diagnostics.target_count,
      Rcpp::Named("candidate_count") = diagnostics.candidate_count,
      Rcpp::Named("penalty_rank") = diagnostics.penalty_rank,
      Rcpp::Named("penalty_nullity") = diagnostics.penalty_nullity,
      Rcpp::Named("cuda_gcv_batches") = diagnostics.cuda_gcv_batches,
      Rcpp::Named("cuda_gcv_targets") = diagnostics.cuda_gcv_targets,
      Rcpp::Named("cuda_gcv_iterations") =
        diagnostics.cuda_gcv_iterations,
      Rcpp::Named("cuda_gcv_nonconverged") =
        diagnostics.cuda_gcv_nonconverged,
      Rcpp::Named("cuda_gcv_boundary_targets") =
        diagnostics.cuda_gcv_boundary_targets,
      Rcpp::Named("legacy_mgcv_target_calls") =
        diagnostics.legacy_mgcv_target_calls,
      Rcpp::Named("cpu_score_count") = diagnostics.cpu_score_count,
      Rcpp::Named("cpu_optimizer_count") = diagnostics.cpu_optimizer_count,
      Rcpp::Named("fallback_count") = diagnostics.fallback_count,
      Rcpp::Named("projection_gemm_count") =
        diagnostics.projection_gemm_count,
      Rcpp::Named("mgcv_qt_y_kernel_launch_count") =
        diagnostics.mgcv_qt_y_kernel_launch_count,
      Rcpp::Named("grid_kernel_launch_count") =
        diagnostics.grid_kernel_launch_count,
      Rcpp::Named("exact_mroot_kernel_launch_count") =
        diagnostics.exact_mroot_kernel_launch_count,
      Rcpp::Named("exact_svd_call_count") =
        diagnostics.exact_svd_call_count,
      Rcpp::Named("exact_svd_nonconverged_count") =
        diagnostics.exact_svd_nonconverged_count,
      Rcpp::Named("exact_objective_kernel_launch_count") =
        diagnostics.exact_objective_kernel_launch_count,
      Rcpp::Named("exact_endpoint_kernel_launch_count") =
        diagnostics.exact_endpoint_kernel_launch_count,
      Rcpp::Named("exact_endpoint_comparison_count") =
        diagnostics.exact_endpoint_comparison_count,
      Rcpp::Named("exact_endpoint_svd_call_count") =
        diagnostics.exact_endpoint_svd_call_count,
      Rcpp::Named("exact_endpoint_failure_count") =
        diagnostics.exact_endpoint_failure_count,
      Rcpp::Named("exact_endpoint_trial_accepted_count") =
        diagnostics.exact_endpoint_trial_accepted_count,
      Rcpp::Named("exact_derivative_refresh_count") =
        diagnostics.exact_derivative_refresh_count,
      Rcpp::Named("exact_derivative_svd_call_count") =
        diagnostics.exact_derivative_svd_call_count,
      Rcpp::Named("exact_derivative_failure_count") =
        diagnostics.exact_derivative_failure_count,
      Rcpp::Named("spectral_optimizer_target_count") =
        diagnostics.spectral_optimizer_target_count,
      Rcpp::Named("spectral_only_target_count") =
        diagnostics.spectral_only_target_count,
      Rcpp::Named("exact_replay_target_count") =
        diagnostics.exact_replay_target_count,
      Rcpp::Named("exact_replay_endpoint_risk_count") =
        diagnostics.exact_replay_endpoint_risk_count,
      Rcpp::Named("exact_replay_convergence_risk_count") =
        diagnostics.exact_replay_convergence_risk_count,
      Rcpp::Named("exact_replay_boundary_risk_count") =
        diagnostics.exact_replay_boundary_risk_count,
      Rcpp::Named("exact_replay_numerical_risk_count") =
        diagnostics.exact_replay_numerical_risk_count,
      Rcpp::Named("selective_replay_target_count") =
        diagnostics.selective_replay_target_count,
      Rcpp::Named("selective_replay_deferred_count") =
        diagnostics.selective_replay_deferred_count,
      Rcpp::Named("selective_replay_steepest_count") =
        diagnostics.selective_replay_steepest_count,
      Rcpp::Named("selective_replay_residual_risk_count") =
        diagnostics.selective_replay_residual_risk_count,
      Rcpp::Named("selective_replay_other_count") =
        diagnostics.selective_replay_other_count,
      Rcpp::Named("optimizer_kernel_launch_count") =
        diagnostics.optimizer_kernel_launch_count,
      Rcpp::Named("augmented_eigensolver_call_count") =
        diagnostics.augmented_eigensolver_call_count,
      Rcpp::Named("augmented_objective_kernel_launch_count") =
        diagnostics.augmented_objective_kernel_launch_count,
      Rcpp::Named("augmented_optimizer_control_sync_count") =
        diagnostics.augmented_optimizer_control_sync_count,
      Rcpp::Named("h2d_copy_count") = diagnostics.h2d_copy_count,
      Rcpp::Named("h2d_bytes") = static_cast<double>(diagnostics.h2d_bytes),
      Rcpp::Named("compact_d2h_count") = diagnostics.compact_d2h_count,
      Rcpp::Named("compact_d2h_bytes") =
        static_cast<double>(diagnostics.compact_d2h_bytes),
      Rcpp::Named("grid_d2h_count") = diagnostics.grid_d2h_count,
      Rcpp::Named("grid_d2h_bytes") =
        static_cast<double>(diagnostics.grid_d2h_bytes),
      Rcpp::Named("transcript_d2h_count") =
        diagnostics.transcript_d2h_count,
      Rcpp::Named("transcript_d2h_bytes") =
        static_cast<double>(diagnostics.transcript_d2h_bytes),
      Rcpp::Named("device_allocation_count") =
        diagnostics.device_allocation_count,
      Rcpp::Named("stream_ordered_allocation_count") =
        diagnostics.stream_ordered_allocation_count,
      Rcpp::Named("synchronous_allocation_count") =
        diagnostics.synchronous_allocation_count,
      Rcpp::Named("device_allocation_bytes") =
        static_cast<double>(diagnostics.device_allocation_bytes),
      Rcpp::Named("upload_cuda_ms") = diagnostics.upload_cuda_ms,
      Rcpp::Named("projection_cuda_ms") = diagnostics.projection_cuda_ms,
      Rcpp::Named("grid_cuda_ms") = diagnostics.grid_cuda_ms,
      Rcpp::Named("optimizer_cuda_ms") = diagnostics.optimizer_cuda_ms,
      Rcpp::Named("d2h_cuda_ms") = diagnostics.d2h_cuda_ms,
      Rcpp::Named("total_host_ms") = diagnostics.total_host_ms,
      Rcpp::Named("cublas_pedantic_math") =
        diagnostics.cublas_pedantic_math,
      Rcpp::Named("cublas_atomics_not_allowed") =
        diagnostics.cublas_atomics_not_allowed,
      Rcpp::Named("cublas_user_workspace_installed") =
        diagnostics.cublas_user_workspace_installed,
      Rcpp::Named("score_matrix_materialized") =
        diagnostics.score_matrix_materialized,
      Rcpp::Named("transcript_materialized") =
        diagnostics.transcript_materialized,
      Rcpp::Named("target_rhs_projected_on_cuda") =
        diagnostics.target_rhs_projected_on_cuda,
      Rcpp::Named("target_selection_on_cuda") =
        diagnostics.target_selection_on_cuda,
      Rcpp::Named("optimizer_target_coverage_complete") =
        diagnostics.optimizer_target_coverage_complete
    )
  );
  END_RCPP
}

namespace {

fastkpc::SinglePenaltyGcvCudaOwnedInput
single_penalty_gcv_owned_input_from_r(SEXP setup_s, int setup_index) {
  if (TYPEOF(setup_s) != VECSXP) {
    Rcpp::stop("multi-setup GCV entry %d must be a list", setup_index + 1);
  }
  Rcpp::List setup(setup_s);
  const std::vector<std::string> required = {
    "X", "Y", "rhs_transform", "eigenvalues", "magic_qr_packed",
    "magic_tau", "magic_r", "magic_penalty_root",
    "magic_penalty_matrix", "target_ids", "penalty_rank", "initial_sp",
    "sp_grid", "materialize_grid", "keep_transcript"
  };
  for (const std::string& field : required) {
    if (!setup.containsElementNamed(field.c_str())) {
      Rcpp::stop(
        "multi-setup GCV entry %d is missing field '%s'",
        setup_index + 1, field.c_str());
    }
  }

  SEXP X_s = setup["X"];
  SEXP Y_s = setup["Y"];
  SEXP rhs_transform_s = setup["rhs_transform"];
  SEXP eigenvalues_s = setup["eigenvalues"];
  SEXP magic_qr_packed_s = setup["magic_qr_packed"];
  SEXP magic_tau_s = setup["magic_tau"];
  SEXP magic_r_s = setup["magic_r"];
  SEXP magic_penalty_root_s = setup["magic_penalty_root"];
  SEXP magic_penalty_matrix_s = setup["magic_penalty_matrix"];
  SEXP target_ids_s = setup["target_ids"];
  SEXP sp_grid_s = setup["sp_grid"];
  if (!Rf_isReal(X_s) || !Rf_isMatrix(X_s) ||
      !Rf_isReal(Y_s) || !Rf_isMatrix(Y_s) ||
      !Rf_isReal(rhs_transform_s) || !Rf_isMatrix(rhs_transform_s) ||
      !Rf_isReal(eigenvalues_s) ||
      !Rf_isReal(magic_qr_packed_s) || !Rf_isMatrix(magic_qr_packed_s) ||
      !Rf_isReal(magic_tau_s) ||
      !Rf_isReal(magic_r_s) || !Rf_isMatrix(magic_r_s) ||
      !Rf_isReal(magic_penalty_root_s) ||
      !Rf_isMatrix(magic_penalty_root_s) ||
      !Rf_isReal(magic_penalty_matrix_s) ||
      !Rf_isMatrix(magic_penalty_matrix_s) ||
      TYPEOF(target_ids_s) != INTSXP || !Rf_isReal(sp_grid_s)) {
    Rcpp::stop(
      "multi-setup GCV entry %d has malformed numeric fields",
      setup_index + 1);
  }

  Rcpp::NumericMatrix X(X_s);
  Rcpp::NumericMatrix Y(Y_s);
  Rcpp::NumericMatrix rhs_transform(rhs_transform_s);
  Rcpp::NumericVector eigenvalues(eigenvalues_s);
  Rcpp::NumericMatrix magic_qr_packed(magic_qr_packed_s);
  Rcpp::NumericVector magic_tau(magic_tau_s);
  Rcpp::NumericMatrix magic_r(magic_r_s);
  Rcpp::NumericMatrix magic_penalty_root(magic_penalty_root_s);
  Rcpp::NumericMatrix magic_penalty_matrix(magic_penalty_matrix_s);
  Rcpp::IntegerVector target_ids(target_ids_s);
  Rcpp::NumericVector sp_grid(sp_grid_s);
  fastkpc::SinglePenaltyGcvCudaOwnedInput input;
  input.n = X.nrow();
  input.coefficient_dim = X.ncol();
  input.target_count = Y.ncol();
  input.penalty_rank = Rcpp::as<int>(setup["penalty_rank"]);
  input.initial_sp = Rcpp::as<double>(setup["initial_sp"]);
  input.materialize_grid = Rcpp::as<bool>(setup["materialize_grid"]);
  input.keep_transcript = Rcpp::as<bool>(setup["keep_transcript"]);
  const int p = input.coefficient_dim;
  if (input.n <= p || p <= 0 || input.target_count <= 0 ||
      Y.nrow() != input.n || rhs_transform.nrow() != p ||
      rhs_transform.ncol() != p || eigenvalues.size() != p ||
      magic_qr_packed.nrow() != input.n || magic_qr_packed.ncol() != p ||
      magic_tau.size() != p || magic_r.nrow() != p ||
      magic_r.ncol() != p || magic_penalty_root.nrow() != p ||
      magic_penalty_root.ncol() != input.penalty_rank ||
      magic_penalty_matrix.nrow() != p || magic_penalty_matrix.ncol() != p ||
      target_ids.size() != input.target_count || input.penalty_rank <= 0 ||
      input.penalty_rank > p || !std::isfinite(input.initial_sp) ||
      input.initial_sp <= 0.0 ||
      (input.materialize_grid && sp_grid.size() == 0)) {
    Rcpp::stop(
      "multi-setup GCV entry %d has invalid dimensions or scalars",
      setup_index + 1);
  }
  if (!all_finite(X) || !all_finite(Y) || !all_finite(rhs_transform) ||
      !all_finite_vector(eigenvalues) || !all_finite(magic_qr_packed) ||
      !all_finite_vector(magic_tau) || !all_finite(magic_r) ||
      !all_finite(magic_penalty_root) || !all_finite(magic_penalty_matrix) ||
      !all_finite_vector(sp_grid)) {
    Rcpp::stop(
      "multi-setup GCV entry %d contains non-finite values",
      setup_index + 1);
  }
  std::unordered_set<int> unique_target_ids;
  for (int value : target_ids) {
    if (value < 1 || value > 64 || !unique_target_ids.insert(value).second) {
      Rcpp::stop(
        "multi-setup GCV entry %d target_ids must be unique in [1, 64]",
        setup_index + 1);
    }
  }
  int positive_eigenvalues = 0;
  int zero_eigenvalues = 0;
  for (double value : eigenvalues) {
    if (value < 0.0) {
      Rcpp::stop(
        "multi-setup GCV entry %d has a negative eigenvalue",
        setup_index + 1);
    }
    if (value == 0.0) {
      ++zero_eigenvalues;
    } else {
      ++positive_eigenvalues;
    }
  }
  if (positive_eigenvalues != input.penalty_rank ||
      zero_eigenvalues != p - input.penalty_rank) {
    Rcpp::stop(
      "multi-setup GCV entry %d eigenvalues disagree with penalty_rank",
      setup_index + 1);
  }
  for (double value : sp_grid) {
    if (value <= 0.0) {
      Rcpp::stop(
        "multi-setup GCV entry %d has a non-positive sp_grid value",
        setup_index + 1);
    }
  }

  input.X.assign(X.begin(), X.end());
  input.Y.assign(Y.begin(), Y.end());
  input.rhs_transform.assign(rhs_transform.begin(), rhs_transform.end());
  input.eigenvalues.assign(eigenvalues.begin(), eigenvalues.end());
  input.magic_qr_packed.assign(
    magic_qr_packed.begin(), magic_qr_packed.end());
  input.magic_tau.assign(magic_tau.begin(), magic_tau.end());
  input.magic_r.assign(magic_r.begin(), magic_r.end());
  input.magic_penalty_root.assign(
    magic_penalty_root.begin(), magic_penalty_root.end());
  input.magic_penalty_matrix.assign(
    magic_penalty_matrix.begin(), magic_penalty_matrix.end());
  input.target_ids.assign(target_ids.begin(), target_ids.end());
  input.sp_grid.assign(sp_grid.begin(), sp_grid.end());
  return input;
}

Rcpp::List single_penalty_gcv_result_to_r(
    const fastkpc::SinglePenaltyGcvCudaResult& result,
    const std::vector<double>& sp_grid,
    bool materialize_grid,
    bool keep_transcript) {
  const int target_count = result.target_count;
  Rcpp::NumericVector selected_sp(target_count);
  Rcpp::NumericVector selected_log_sp(target_count);
  Rcpp::NumericVector selected_score(target_count);
  Rcpp::NumericVector selected_edf(target_count);
  Rcpp::NumericVector selected_rss(target_count);
  Rcpp::NumericVector selected_gradient(target_count);
  Rcpp::NumericVector selected_hessian(target_count);
  Rcpp::NumericVector reported_rms_gradient(target_count);
  Rcpp::NumericVector pre_boundary_log_sp(target_count);
  Rcpp::IntegerVector iteration_count(target_count);
  Rcpp::IntegerVector score_call_count(target_count);
  Rcpp::IntegerVector actual_objective_call_count(target_count);
  Rcpp::LogicalVector fully_converged(target_count);
  Rcpp::LogicalVector hessian_positive_definite(target_count);
  Rcpp::IntegerVector boundary_probe_count(target_count);
  Rcpp::IntegerVector boundary_accepted_count(target_count);
  Rcpp::CharacterVector termination_reason(target_count);
  Rcpp::CharacterVector boundary_status(target_count);
  for (int target = 0; target < target_count; ++target) {
    const fastkpc::SinglePenaltyGcvOptimizerResult& value =
      result.targets[static_cast<std::size_t>(target)];
    selected_sp[target] = value.sp;
    selected_log_sp[target] = value.log_sp;
    selected_score[target] = value.score;
    selected_edf[target] = value.edf;
    selected_rss[target] = value.rss;
    selected_gradient[target] = value.gradient;
    selected_hessian[target] = value.hessian;
    reported_rms_gradient[target] = value.reported_rms_gradient;
    pre_boundary_log_sp[target] = value.pre_boundary_log_sp;
    iteration_count[target] = value.iteration_count;
    score_call_count[target] = value.score_call_count;
    actual_objective_call_count[target] = value.actual_objective_call_count;
    fully_converged[target] = value.fully_converged != 0;
    hessian_positive_definite[target] =
      value.hessian_positive_definite != 0;
    boundary_probe_count[target] = value.boundary_probe_count;
    boundary_accepted_count[target] = value.boundary_accepted_count;
    termination_reason[target] =
      fastkpc::single_penalty_gcv_termination_name(value.termination);
    boundary_status[target] = value.boundary_accepted_count > 0 ?
      "mgcv_infinity_probe_accepted" : "finite_refinement";
  }
  Rcpp::DataFrame targets = Rcpp::DataFrame::create(
    Rcpp::Named("target_position") = Rcpp::seq(1, target_count),
    Rcpp::Named("sp") = selected_sp,
    Rcpp::Named("log_sp") = selected_log_sp,
    Rcpp::Named("score") = selected_score,
    Rcpp::Named("edf") = selected_edf,
    Rcpp::Named("rss") = selected_rss,
    Rcpp::Named("gradient") = selected_gradient,
    Rcpp::Named("hessian") = selected_hessian,
    Rcpp::Named("reported_rms_gradient") = reported_rms_gradient,
    Rcpp::Named("pre_boundary_log_sp") = pre_boundary_log_sp,
    Rcpp::Named("iteration_count") = iteration_count,
    Rcpp::Named("score_call_count") = score_call_count,
    Rcpp::Named("actual_objective_call_count") =
      actual_objective_call_count,
    Rcpp::Named("fully_converged") = fully_converged,
    Rcpp::Named("hessian_positive_definite") =
      hessian_positive_definite,
    Rcpp::Named("boundary_probe_count") = boundary_probe_count,
    Rcpp::Named("boundary_accepted_count") = boundary_accepted_count,
    Rcpp::Named("termination_reason") = termination_reason,
    Rcpp::Named("boundary_status") = boundary_status,
    Rcpp::Named("stringsAsFactors") = false);

  Rcpp::List grid = R_NilValue;
  if (materialize_grid) {
    const int candidate_count = static_cast<int>(sp_grid.size());
    Rcpp::NumericMatrix rss(target_count, candidate_count);
    Rcpp::NumericMatrix edf(target_count, candidate_count);
    Rcpp::NumericMatrix score(target_count, candidate_count);
    Rcpp::LogicalMatrix valid(target_count, candidate_count);
    for (int candidate = 0; candidate < candidate_count; ++candidate) {
      for (int target = 0; target < target_count; ++target) {
        const std::size_t index = static_cast<std::size_t>(target) +
          static_cast<std::size_t>(target_count) * candidate;
        const fastkpc::SinglePenaltyGcvGridCell& value = result.grid[index];
        rss(target, candidate) = value.rss;
        edf(target, candidate) = value.edf;
        score(target, candidate) = value.score;
        valid(target, candidate) = value.valid != 0;
      }
    }
    grid = Rcpp::List::create(
      Rcpp::Named("sp") = Rcpp::wrap(sp_grid),
      Rcpp::Named("rss") = rss,
      Rcpp::Named("edf") = edf,
      Rcpp::Named("score") = score,
      Rcpp::Named("valid") = valid);
  }

  Rcpp::List transcripts(target_count);
  if (keep_transcript) {
    for (int target = 0; target < target_count; ++target) {
      const int count =
        result.transcript_counts[static_cast<std::size_t>(target)];
      Rcpp::CharacterVector stage(count);
      Rcpp::IntegerVector iteration(count);
      Rcpp::IntegerVector evaluation(count);
      Rcpp::NumericVector current_log_sp(count);
      Rcpp::NumericVector proposed_step(count);
      Rcpp::NumericVector trial_log_sp(count);
      Rcpp::NumericVector objective(count);
      Rcpp::NumericVector gradient(count);
      Rcpp::NumericVector hessian(count);
      Rcpp::LogicalVector accepted(count);
      Rcpp::CharacterVector step_source(count);
      for (int row = 0; row < count; ++row) {
        const std::size_t index = static_cast<std::size_t>(target) *
          fastkpc::kSinglePenaltyGcvTranscriptCapacity + row;
        const fastkpc::SinglePenaltyGcvTranscriptEntry& value =
          result.transcript[index];
        stage[row] = fastkpc::single_penalty_gcv_transcript_stage_name(
          value.stage);
        iteration[row] = value.iteration;
        evaluation[row] = value.evaluation;
        current_log_sp[row] = value.current_log_sp;
        proposed_step[row] = value.proposed_step;
        trial_log_sp[row] = value.trial_log_sp;
        objective[row] = value.objective;
        gradient[row] = value.gradient;
        hessian[row] = value.hessian;
        accepted[row] = value.accepted != 0;
        step_source[row] = fastkpc::single_penalty_gcv_step_source_name(
          value.step_source);
      }
      transcripts[target] = Rcpp::DataFrame::create(
        Rcpp::Named("stage") = stage,
        Rcpp::Named("iteration") = iteration,
        Rcpp::Named("evaluation") = evaluation,
        Rcpp::Named("current_log_sp") = current_log_sp,
        Rcpp::Named("proposed_step") = proposed_step,
        Rcpp::Named("trial_log_sp") = trial_log_sp,
        Rcpp::Named("objective") = objective,
        Rcpp::Named("gradient") = gradient,
        Rcpp::Named("hessian") = hessian,
        Rcpp::Named("accepted") = accepted,
        Rcpp::Named("step_source") = step_source,
        Rcpp::Named("stringsAsFactors") = false);
    }
  }

  const fastkpc::SinglePenaltyGcvCudaDiagnostics& diagnostics =
    result.diagnostics;
  return Rcpp::List::create(
    Rcpp::Named("schema_version") = result.schema_version,
    Rcpp::Named("targets") = targets,
    Rcpp::Named("grid") = grid,
    Rcpp::Named("transcripts") = transcripts,
    Rcpp::Named("transcript_overflow") = result.transcript_overflow,
    Rcpp::Named("diagnostics") = Rcpp::List::create(
      Rcpp::Named("schema_version") = diagnostics.schema_version,
      Rcpp::Named("sp_selection_backend_executed") =
        diagnostics.sp_selection_backend_executed,
      Rcpp::Named("gcv_score_backend_executed") =
        diagnostics.gcv_score_backend_executed,
      Rcpp::Named("optimizer_backend_executed") =
        diagnostics.optimizer_backend_executed,
      Rcpp::Named("exact_replay_backend_executed") =
        diagnostics.exact_replay_backend_executed,
      Rcpp::Named("device_id") = diagnostics.device_id,
      Rcpp::Named("gpu_name") = diagnostics.gpu_name,
      Rcpp::Named("n") = diagnostics.n,
      Rcpp::Named("coefficient_dim") = diagnostics.coefficient_dim,
      Rcpp::Named("target_count") = diagnostics.target_count,
      Rcpp::Named("candidate_count") = diagnostics.candidate_count,
      Rcpp::Named("penalty_rank") = diagnostics.penalty_rank,
      Rcpp::Named("penalty_nullity") = diagnostics.penalty_nullity,
      Rcpp::Named("cuda_gcv_batches") = diagnostics.cuda_gcv_batches,
      Rcpp::Named("cuda_gcv_targets") = diagnostics.cuda_gcv_targets,
      Rcpp::Named("cuda_gcv_iterations") =
        diagnostics.cuda_gcv_iterations,
      Rcpp::Named("cuda_gcv_nonconverged") =
        diagnostics.cuda_gcv_nonconverged,
      Rcpp::Named("cuda_gcv_boundary_targets") =
        diagnostics.cuda_gcv_boundary_targets,
      Rcpp::Named("legacy_mgcv_target_calls") =
        diagnostics.legacy_mgcv_target_calls,
      Rcpp::Named("cpu_score_count") = diagnostics.cpu_score_count,
      Rcpp::Named("cpu_optimizer_count") = diagnostics.cpu_optimizer_count,
      Rcpp::Named("fallback_count") = diagnostics.fallback_count,
      Rcpp::Named("projection_gemm_count") =
        diagnostics.projection_gemm_count,
      Rcpp::Named("mgcv_qt_y_kernel_launch_count") =
        diagnostics.mgcv_qt_y_kernel_launch_count,
      Rcpp::Named("grid_kernel_launch_count") =
        diagnostics.grid_kernel_launch_count,
      Rcpp::Named("exact_mroot_kernel_launch_count") =
        diagnostics.exact_mroot_kernel_launch_count,
      Rcpp::Named("exact_svd_call_count") = diagnostics.exact_svd_call_count,
      Rcpp::Named("exact_svd_nonconverged_count") =
        diagnostics.exact_svd_nonconverged_count,
      Rcpp::Named("exact_objective_kernel_launch_count") =
        diagnostics.exact_objective_kernel_launch_count,
      Rcpp::Named("exact_endpoint_kernel_launch_count") =
        diagnostics.exact_endpoint_kernel_launch_count,
      Rcpp::Named("exact_endpoint_comparison_count") =
        diagnostics.exact_endpoint_comparison_count,
      Rcpp::Named("exact_endpoint_svd_call_count") =
        diagnostics.exact_endpoint_svd_call_count,
      Rcpp::Named("exact_endpoint_failure_count") =
        diagnostics.exact_endpoint_failure_count,
      Rcpp::Named("exact_endpoint_trial_accepted_count") =
        diagnostics.exact_endpoint_trial_accepted_count,
      Rcpp::Named("exact_derivative_refresh_count") =
        diagnostics.exact_derivative_refresh_count,
      Rcpp::Named("exact_derivative_svd_call_count") =
        diagnostics.exact_derivative_svd_call_count,
      Rcpp::Named("exact_derivative_failure_count") =
        diagnostics.exact_derivative_failure_count,
      Rcpp::Named("spectral_optimizer_target_count") =
        diagnostics.spectral_optimizer_target_count,
      Rcpp::Named("spectral_only_target_count") =
        diagnostics.spectral_only_target_count,
      Rcpp::Named("exact_replay_target_count") =
        diagnostics.exact_replay_target_count,
      Rcpp::Named("exact_replay_endpoint_risk_count") =
        diagnostics.exact_replay_endpoint_risk_count,
      Rcpp::Named("exact_replay_convergence_risk_count") =
        diagnostics.exact_replay_convergence_risk_count,
      Rcpp::Named("exact_replay_boundary_risk_count") =
        diagnostics.exact_replay_boundary_risk_count,
      Rcpp::Named("exact_replay_numerical_risk_count") =
        diagnostics.exact_replay_numerical_risk_count,
      Rcpp::Named("selective_replay_target_count") =
        diagnostics.selective_replay_target_count,
      Rcpp::Named("selective_replay_deferred_count") =
        diagnostics.selective_replay_deferred_count,
      Rcpp::Named("selective_replay_steepest_count") =
        diagnostics.selective_replay_steepest_count,
      Rcpp::Named("selective_replay_residual_risk_count") =
        diagnostics.selective_replay_residual_risk_count,
      Rcpp::Named("selective_replay_other_count") =
        diagnostics.selective_replay_other_count,
      Rcpp::Named("optimizer_kernel_launch_count") =
        diagnostics.optimizer_kernel_launch_count,
      Rcpp::Named("augmented_eigensolver_call_count") =
        diagnostics.augmented_eigensolver_call_count,
      Rcpp::Named("augmented_objective_kernel_launch_count") =
        diagnostics.augmented_objective_kernel_launch_count,
      Rcpp::Named("augmented_optimizer_control_sync_count") =
        diagnostics.augmented_optimizer_control_sync_count,
      Rcpp::Named("h2d_copy_count") = diagnostics.h2d_copy_count,
      Rcpp::Named("h2d_bytes") = static_cast<double>(diagnostics.h2d_bytes),
      Rcpp::Named("compact_d2h_count") = diagnostics.compact_d2h_count,
      Rcpp::Named("compact_d2h_bytes") =
        static_cast<double>(diagnostics.compact_d2h_bytes),
      Rcpp::Named("grid_d2h_count") = diagnostics.grid_d2h_count,
      Rcpp::Named("grid_d2h_bytes") =
        static_cast<double>(diagnostics.grid_d2h_bytes),
      Rcpp::Named("transcript_d2h_count") =
        diagnostics.transcript_d2h_count,
      Rcpp::Named("transcript_d2h_bytes") =
        static_cast<double>(diagnostics.transcript_d2h_bytes),
      Rcpp::Named("device_allocation_count") =
        diagnostics.device_allocation_count,
      Rcpp::Named("stream_ordered_allocation_count") =
        diagnostics.stream_ordered_allocation_count,
      Rcpp::Named("synchronous_allocation_count") =
        diagnostics.synchronous_allocation_count,
      Rcpp::Named("device_allocation_bytes") =
        static_cast<double>(diagnostics.device_allocation_bytes),
      Rcpp::Named("upload_cuda_ms") = diagnostics.upload_cuda_ms,
      Rcpp::Named("projection_cuda_ms") = diagnostics.projection_cuda_ms,
      Rcpp::Named("grid_cuda_ms") = diagnostics.grid_cuda_ms,
      Rcpp::Named("optimizer_cuda_ms") = diagnostics.optimizer_cuda_ms,
      Rcpp::Named("d2h_cuda_ms") = diagnostics.d2h_cuda_ms,
      Rcpp::Named("total_host_ms") = diagnostics.total_host_ms,
      Rcpp::Named("cublas_pedantic_math") =
        diagnostics.cublas_pedantic_math,
      Rcpp::Named("cublas_atomics_not_allowed") =
        diagnostics.cublas_atomics_not_allowed,
      Rcpp::Named("cublas_user_workspace_installed") =
        diagnostics.cublas_user_workspace_installed,
      Rcpp::Named("score_matrix_materialized") =
        diagnostics.score_matrix_materialized,
      Rcpp::Named("transcript_materialized") =
        diagnostics.transcript_materialized,
      Rcpp::Named("target_rhs_projected_on_cuda") =
        diagnostics.target_rhs_projected_on_cuda,
      Rcpp::Named("target_selection_on_cuda") =
        diagnostics.target_selection_on_cuda,
      Rcpp::Named("optimizer_target_coverage_complete") =
        diagnostics.optimizer_target_coverage_complete));
}

}  // namespace

extern "C" SEXP C_full_cuda_ci_single_penalty_gcv_multi_cuda(
    SEXP setups_s,
    SEXP concurrency_s) {
  BEGIN_RCPP
  if (TYPEOF(setups_s) != VECSXP || Rf_xlength(setups_s) == 0) {
    Rcpp::stop("multi-setup single-penalty GCV setups must be a non-empty list");
  }
  const int concurrency = Rcpp::as<int>(concurrency_s);
  if (concurrency <= 0 ||
      concurrency > fastkpc::kSinglePenaltyGcvMaximumConcurrentSetups) {
    Rcpp::stop("multi-setup single-penalty GCV concurrency must be in [1, 16]");
  }
  Rcpp::List setups(setups_s);
  std::vector<fastkpc::SinglePenaltyGcvCudaOwnedInput> inputs;
  inputs.reserve(static_cast<std::size_t>(setups.size()));
  for (int index = 0; index < setups.size(); ++index) {
    inputs.push_back(single_penalty_gcv_owned_input_from_r(setups[index], index));
  }

  const fastkpc::SinglePenaltyGcvCudaMultiResult result =
    fastkpc::single_penalty_gcv_cuda_multi(inputs, concurrency);
  Rcpp::List output_setups(result.setups.size());
  for (std::size_t index = 0; index < result.setups.size(); ++index) {
    const fastkpc::SinglePenaltyGcvCudaOwnedInput& input = inputs[index];
    output_setups[static_cast<R_xlen_t>(index)] =
      single_penalty_gcv_result_to_r(
        result.setups[index], input.sp_grid, input.materialize_grid,
        input.keep_transcript);
  }
  SEXP setup_names = Rf_getAttrib(setups_s, R_NamesSymbol);
  if (setup_names != R_NilValue) {
    output_setups.attr("names") = Rcpp::clone(Rcpp::CharacterVector(setup_names));
  }
  const fastkpc::SinglePenaltyGcvCudaMultiDiagnostics& diagnostics =
    result.diagnostics;
  return Rcpp::List::create(
    Rcpp::Named("schema_version") = result.schema_version,
    Rcpp::Named("setups") = output_setups,
    Rcpp::Named("diagnostics") = Rcpp::List::create(
      Rcpp::Named("schema_version") = diagnostics.schema_version,
      Rcpp::Named("execution_strategy") = diagnostics.execution_strategy,
      Rcpp::Named("device_id") = diagnostics.device_id,
      Rcpp::Named("setup_count") = diagnostics.setup_count,
      Rcpp::Named("requested_concurrency") =
        diagnostics.requested_concurrency,
      Rcpp::Named("worker_count") = diagnostics.worker_count,
      Rcpp::Named("worker_device_bind_count") =
        diagnostics.worker_device_bind_count,
      Rcpp::Named("max_host_calls_in_flight") =
        diagnostics.max_host_calls_in_flight,
      Rcpp::Named("fused_exact_replay_target_count") =
        diagnostics.fused_exact_replay_target_count,
      Rcpp::Named("fused_exact_replay_kernel_launch_count") =
        diagnostics.fused_exact_replay_kernel_launch_count,
      Rcpp::Named("fused_exact_replay_device_bytes") =
        static_cast<double>(diagnostics.fused_exact_replay_device_bytes),
      Rcpp::Named("setup_host_ms_sum") = diagnostics.setup_host_ms_sum,
      Rcpp::Named("spectral_setup_wall_ms") =
        diagnostics.spectral_setup_wall_ms,
      Rcpp::Named("fused_exact_replay_cuda_ms") =
        diagnostics.fused_exact_replay_cuda_ms,
      Rcpp::Named("fused_exact_replay_host_ms") =
        diagnostics.fused_exact_replay_host_ms,
      Rcpp::Named("wall_host_ms") = diagnostics.wall_host_ms,
      Rcpp::Named("host_overlap_factor") =
        diagnostics.host_overlap_factor,
      Rcpp::Named("fused_exact_replay_executed") =
        diagnostics.fused_exact_replay_executed));
  END_RCPP
}

extern "C" SEXP C_full_cuda_ci_single_penalty_gcv_fixed_sp_cuda(
    SEXP prepared_s,
    SEXP Xs,
    SEXP Ys,
    SEXP rhs_transform_s,
    SEXP eigenvalues_s,
    SEXP magic_qr_packed_s,
    SEXP magic_tau_s,
    SEXP magic_r_s,
    SEXP magic_penalty_root_s,
    SEXP magic_penalty_matrix_s,
    SEXP target_ids_s,
    SEXP penalty_rank_s,
    SEXP initial_sp_s,
    SEXP planned_route_s,
    SEXP target_keys_s,
    SEXP outputs_s) {
  BEGIN_RCPP
  FixedSpPreparedHolder* prepared_holder =
    fixed_sp_cuda_prepared_holder(prepared_s, true);
  const fastkpc::PreparedSInfo prepared_info =
    fastkpc::prepared_s_gpu_info(prepared_holder->value);
  if (prepared_info.penalty_count != 1) {
    Rcpp::stop("integrated Phase 4 solve requires one penalty");
  }
  if (!Rf_isReal(Xs) || !Rf_isMatrix(Xs) ||
      !Rf_isReal(Ys) || !Rf_isMatrix(Ys) ||
      !Rf_isReal(rhs_transform_s) || !Rf_isMatrix(rhs_transform_s) ||
      !Rf_isReal(eigenvalues_s) ||
      !Rf_isReal(magic_qr_packed_s) ||
      !Rf_isMatrix(magic_qr_packed_s) ||
      !Rf_isReal(magic_tau_s) ||
      !Rf_isReal(magic_r_s) || !Rf_isMatrix(magic_r_s) ||
      !Rf_isReal(magic_penalty_root_s) ||
      !Rf_isMatrix(magic_penalty_root_s) ||
      !Rf_isReal(magic_penalty_matrix_s) ||
      !Rf_isMatrix(magic_penalty_matrix_s)) {
    Rcpp::stop("integrated Phase 4 numeric inputs are malformed");
  }
  Rcpp::NumericMatrix X(Xs);
  Rcpp::NumericMatrix Y(Ys);
  Rcpp::NumericMatrix rhs_transform(rhs_transform_s);
  Rcpp::NumericVector eigenvalues(eigenvalues_s);
  Rcpp::NumericMatrix magic_qr_packed(magic_qr_packed_s);
  Rcpp::NumericVector magic_tau(magic_tau_s);
  Rcpp::NumericMatrix magic_r(magic_r_s);
  Rcpp::NumericMatrix magic_penalty_root(magic_penalty_root_s);
  Rcpp::NumericMatrix magic_penalty_matrix(magic_penalty_matrix_s);
  Rcpp::IntegerVector target_ids(target_ids_s);
  const int n = X.nrow();
  const int p = X.ncol();
  const int target_count = Y.ncol();
  const int penalty_rank = Rcpp::as<int>(penalty_rank_s);
  const double initial_sp = Rcpp::as<double>(initial_sp_s);
  if (n != prepared_info.n || p != prepared_info.coefficient_dim ||
      Y.nrow() != n || target_count <= 0) {
    Rcpp::stop("integrated Phase 4 dimensions disagree with prepared setup");
  }
  if (rhs_transform.nrow() != p || rhs_transform.ncol() != p ||
      eigenvalues.size() != p || magic_qr_packed.nrow() != n ||
      magic_qr_packed.ncol() != p || magic_tau.size() != p ||
      magic_r.nrow() != p ||
      magic_r.ncol() != p || magic_penalty_root.nrow() != p ||
      magic_penalty_root.ncol() != penalty_rank ||
      magic_penalty_matrix.nrow() != p || magic_penalty_matrix.ncol() != p) {
    Rcpp::stop("integrated Phase 4 spectral dimensions are invalid");
  }
  if (target_ids.size() != target_count) {
    Rcpp::stop("integrated Phase 4 target_ids length is invalid");
  }
  std::unordered_set<int> unique_gcv_target_ids;
  for (int value : target_ids) {
    if (value < 1 || value > 64 ||
        !unique_gcv_target_ids.insert(value).second) {
      Rcpp::stop(
        "integrated Phase 4 target_ids must be unique in [1, 64]");
    }
  }
  if (!all_finite(X) || !all_finite(Y) ||
      !all_finite(rhs_transform) || !all_finite(magic_qr_packed) ||
      !all_finite_vector(magic_tau) ||
      !all_finite(magic_r) || !all_finite(magic_penalty_root) ||
      !all_finite(magic_penalty_matrix) ||
      !all_finite_vector(eigenvalues) ||
      !std::isfinite(initial_sp) || initial_sp <= 0.0) {
    Rcpp::stop("integrated Phase 4 inputs must be finite");
  }
  int positive_eigenvalues = 0;
  int zero_eigenvalues = 0;
  for (double value : eigenvalues) {
    if (value < 0.0) Rcpp::stop("eigenvalues must be non-negative");
    if (value == 0.0) {
      ++zero_eigenvalues;
    } else {
      ++positive_eigenvalues;
    }
  }
  if (positive_eigenvalues != penalty_rank ||
      zero_eigenvalues != p - penalty_rank) {
    Rcpp::stop(
      "rank-aware eigenvalue zeros disagree with penalty_rank");
  }

  const std::vector<std::string> route_names = bare_character_vector(
    planned_route_s, target_count, "planned_route");
  std::vector<fastkpc::FixedSpRoute> planned_routes;
  planned_routes.reserve(static_cast<std::size_t>(target_count));
  for (const std::string& route : route_names) {
    planned_routes.push_back(fixed_sp_route_from_string(route));
  }
  const std::vector<std::string> target_keys = bare_character_vector(
    target_keys_s, target_count, "target_keys");
  std::unordered_set<std::string> unique_target_keys;
  unique_target_keys.reserve(static_cast<std::size_t>(target_count));
  for (const std::string& key : target_keys) {
    const bool valid_sha = key.size() == 64U &&
      std::all_of(key.begin(), key.end(), [](unsigned char character) {
        return (character >= '0' && character <= '9') ||
          (character >= 'a' && character <= 'f');
      });
    if (!valid_sha || !unique_target_keys.insert(key).second) {
      Rcpp::stop(
        "target_keys must be unique lowercase SHA-256 strings");
    }
  }

  const fastkpc::SinglePenaltyGcvCudaResult gcv =
    fastkpc::single_penalty_gcv_cuda(
      REAL(Xs), REAL(Ys), REAL(rhs_transform_s), REAL(eigenvalues_s),
      REAL(magic_qr_packed_s), REAL(magic_tau_s), REAL(magic_r_s),
      REAL(magic_penalty_root_s), REAL(magic_penalty_matrix_s),
      INTEGER(target_ids_s),
      n, p, target_count, penalty_rank, initial_sp,
      std::vector<double>(), false, false);
  std::vector<double> selected_sp(static_cast<std::size_t>(target_count));
  for (int target = 0; target < target_count; ++target) {
    const fastkpc::SinglePenaltyGcvOptimizerResult& value =
      gcv.targets[static_cast<std::size_t>(target)];
    const fastkpc::SinglePenaltyGcvTermination termination =
      static_cast<fastkpc::SinglePenaltyGcvTermination>(value.termination);
    const bool accepted_termination =
      termination == fastkpc::SinglePenaltyGcvTermination::ScoreAndGradient ||
      termination ==
        fastkpc::SinglePenaltyGcvTermination::StepHalvingExhausted ||
      termination == fastkpc::SinglePenaltyGcvTermination::FlatObjective;
    if (!accepted_termination || !std::isfinite(value.sp) || value.sp <= 0.0) {
      Rcpp::stop(
        "CUDA GCV did not produce an admissible smoothing parameter");
    }
    selected_sp[static_cast<std::size_t>(target)] = value.sp;
  }

  fastkpc::FixedSpBatchHostView batch;
  batch.Y = REAL(Ys);
  batch.SP = selected_sp.data();
  batch.n = prepared_info.n;
  batch.null_dim = prepared_info.null_dim;
  batch.penalty_count = prepared_info.penalty_count;
  batch.target_count = target_count;
  batch.output_mask = fixed_sp_output_mask(outputs_s);
  batch.planned_routes = std::move(planned_routes);
  batch.target_keys = target_keys;

  const auto solve_begin = std::chrono::steady_clock::now();
  std::shared_ptr<fastkpc::DeviceResidualBatch> residual =
    fastkpc::solve_fixed_sp_batch(prepared_holder->value, batch);
  const double solve_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - solve_begin).count();

  SEXP ext = PROTECT(R_MakeExternalPtr(
    nullptr, fixed_sp_cuda_residual_tag(), R_NilValue));
  R_RegisterCFinalizerEx(ext, fixed_sp_cuda_residual_finalizer, TRUE);
  auto* residual_holder = new FixedSpResidualHolder{
    std::move(residual), fixed_sp_current_pid()
  };
  R_SetExternalPtrAddr(ext, residual_holder);
  fixed_sp_owner_acquire(residual_holder, FixedSpOwnerKind::Residual);

  Rcpp::NumericVector sp(target_count);
  Rcpp::NumericVector log_sp(target_count);
  Rcpp::NumericVector score(target_count);
  Rcpp::NumericVector edf(target_count);
  Rcpp::NumericVector rss(target_count);
  Rcpp::IntegerVector iterations(target_count);
  Rcpp::LogicalVector fully_converged(target_count);
  Rcpp::CharacterVector termination_reason(target_count);
  Rcpp::CharacterVector boundary_status(target_count);
  for (int target = 0; target < target_count; ++target) {
    const fastkpc::SinglePenaltyGcvOptimizerResult& value =
      gcv.targets[static_cast<std::size_t>(target)];
    sp[target] = value.sp;
    log_sp[target] = value.log_sp;
    score[target] = value.score;
    edf[target] = value.edf;
    rss[target] = value.rss;
    iterations[target] = value.iteration_count;
    fully_converged[target] = value.fully_converged != 0;
    termination_reason[target] =
      fastkpc::single_penalty_gcv_termination_name(value.termination);
    boundary_status[target] = value.boundary_accepted_count > 0 ?
      "mgcv_infinity_probe_accepted" : "finite_refinement";
  }
  const fastkpc::SinglePenaltyGcvCudaDiagnostics& diagnostics =
    gcv.diagnostics;
  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("schema_version") =
      "full-cuda-ci-single-penalty-gcv-fixed-sp-native-v1",
    Rcpp::Named("residual_token") = ext,
    Rcpp::Named("targets") = Rcpp::DataFrame::create(
      Rcpp::Named("target_position") = Rcpp::seq(1, target_count),
      Rcpp::Named("sp") = sp,
      Rcpp::Named("log_sp") = log_sp,
      Rcpp::Named("score") = score,
      Rcpp::Named("edf") = edf,
      Rcpp::Named("rss") = rss,
      Rcpp::Named("iteration_count") = iterations,
      Rcpp::Named("fully_converged") = fully_converged,
      Rcpp::Named("termination_reason") = termination_reason,
      Rcpp::Named("boundary_status") = boundary_status,
      Rcpp::Named("stringsAsFactors") = false),
    Rcpp::Named("diagnostics") = Rcpp::List::create(
      Rcpp::Named("single_penalty_targets") = target_count,
      Rcpp::Named("cuda_gcv_targets") = diagnostics.cuda_gcv_targets,
      Rcpp::Named("cuda_gcv_batches") = diagnostics.cuda_gcv_batches,
      Rcpp::Named("cuda_gcv_iterations") =
        diagnostics.cuda_gcv_iterations,
      Rcpp::Named("cuda_gcv_nonconverged") =
        diagnostics.cuda_gcv_nonconverged,
      Rcpp::Named("cuda_gcv_boundary_targets") =
        diagnostics.cuda_gcv_boundary_targets,
      Rcpp::Named("cuda_gcv_score_ms") =
        diagnostics.projection_cuda_ms + diagnostics.grid_cuda_ms +
          diagnostics.optimizer_cuda_ms,
      Rcpp::Named("cuda_selected_sp_solve_ms") = solve_ms,
      Rcpp::Named("sp_selection_backend_executed") =
        diagnostics.sp_selection_backend_executed,
      Rcpp::Named("gcv_score_backend_executed") =
        diagnostics.gcv_score_backend_executed,
      Rcpp::Named("optimizer_backend_executed") =
        diagnostics.optimizer_backend_executed,
      Rcpp::Named("exact_replay_backend_executed") =
        diagnostics.exact_replay_backend_executed,
      Rcpp::Named("legacy_mgcv_target_calls") =
        diagnostics.legacy_mgcv_target_calls,
      Rcpp::Named("fallback_count") = diagnostics.fallback_count,
      Rcpp::Named("exact_endpoint_comparison_count") =
        diagnostics.exact_endpoint_comparison_count,
      Rcpp::Named("exact_endpoint_svd_call_count") =
        diagnostics.exact_endpoint_svd_call_count,
      Rcpp::Named("exact_endpoint_failure_count") =
        diagnostics.exact_endpoint_failure_count,
      Rcpp::Named("exact_endpoint_trial_accepted_count") =
        diagnostics.exact_endpoint_trial_accepted_count,
      Rcpp::Named("exact_derivative_refresh_count") =
        diagnostics.exact_derivative_refresh_count,
      Rcpp::Named("exact_derivative_svd_call_count") =
        diagnostics.exact_derivative_svd_call_count,
      Rcpp::Named("exact_derivative_failure_count") =
        diagnostics.exact_derivative_failure_count,
      Rcpp::Named("spectral_optimizer_target_count") =
        diagnostics.spectral_optimizer_target_count,
      Rcpp::Named("spectral_only_target_count") =
        diagnostics.spectral_only_target_count,
      Rcpp::Named("exact_replay_target_count") =
        diagnostics.exact_replay_target_count,
      Rcpp::Named("exact_replay_endpoint_risk_count") =
        diagnostics.exact_replay_endpoint_risk_count,
      Rcpp::Named("exact_replay_convergence_risk_count") =
        diagnostics.exact_replay_convergence_risk_count,
      Rcpp::Named("exact_replay_boundary_risk_count") =
        diagnostics.exact_replay_boundary_risk_count,
      Rcpp::Named("exact_replay_numerical_risk_count") =
        diagnostics.exact_replay_numerical_risk_count,
      Rcpp::Named("selective_replay_target_count") =
        diagnostics.selective_replay_target_count,
      Rcpp::Named("selective_replay_deferred_count") =
        diagnostics.selective_replay_deferred_count,
      Rcpp::Named("selective_replay_steepest_count") =
        diagnostics.selective_replay_steepest_count,
      Rcpp::Named("selective_replay_residual_risk_count") =
        diagnostics.selective_replay_residual_risk_count,
      Rcpp::Named("selective_replay_other_count") =
        diagnostics.selective_replay_other_count,
      Rcpp::Named("optimizer_target_coverage_complete") =
        diagnostics.optimizer_target_coverage_complete,
      Rcpp::Named("selected_sp_feed_path") =
        "cuda-compact-native-cpp-fixed-sp",
      Rcpp::Named("selected_sp_returned_to_r_before_solve") = false,
      Rcpp::Named("native_selection_and_solve_call_count") = 1,
      Rcpp::Named("fixed_sp_token_created") = true)
  );
  UNPROTECT(1);
  return result;
  END_RCPP
}

extern "C" SEXP C_mgcv_extract_gpu_solve_handle_fixed_sp(
    SEXP Xs,
    SEXP ys,
    SEXP Zs,
    SEXP XtX_nulls,
    SEXP penalty_nulls,
    SEXP Xty_nulls) {
  BEGIN_RCPP
  if (!Rf_isReal(Xs) || !Rf_isMatrix(Xs)) Rcpp::stop("X must be a numeric matrix");
  if (!Rf_isReal(ys)) Rcpp::stop("y must be numeric");
  if (!Rf_isReal(Zs) || !Rf_isMatrix(Zs)) Rcpp::stop("Z must be a numeric matrix");
  if (!Rf_isReal(XtX_nulls) || !Rf_isMatrix(XtX_nulls)) {
    Rcpp::stop("XtX_null must be a numeric matrix");
  }
  if (!Rf_isReal(penalty_nulls) || !Rf_isMatrix(penalty_nulls)) {
    Rcpp::stop("penalty_null must be a numeric matrix");
  }
  if (!Rf_isReal(Xty_nulls)) Rcpp::stop("Xty_null must be numeric");

  Rcpp::NumericMatrix X(Xs);
  Rcpp::NumericVector y(ys);
  Rcpp::NumericMatrix Z(Zs);
  Rcpp::NumericMatrix XtX_null(XtX_nulls);
  Rcpp::NumericMatrix penalty_null(penalty_nulls);
  Rcpp::NumericVector Xty_null(Xty_nulls);

  const int n = X.nrow();
  const int p = X.ncol();
  const int q = Z.ncol();
  if (y.size() != n) Rcpp::stop("length(y) must equal nrow(X)");
  if (Z.nrow() != p) Rcpp::stop("nrow(Z) must equal ncol(X)");
  if (XtX_null.nrow() != q || XtX_null.ncol() != q) {
    Rcpp::stop("XtX_null dimensions must match ncol(Z)");
  }
  if (penalty_null.nrow() != q || penalty_null.ncol() != q) {
    Rcpp::stop("penalty_null dimensions must match ncol(Z)");
  }
  if (Xty_null.size() != q) Rcpp::stop("length(Xty_null) must equal ncol(Z)");
  if (!all_finite(X)) Rcpp::stop("X contains missing or infinite values");
  if (!all_finite_vector(y)) Rcpp::stop("y contains missing or infinite values");
  if (!all_finite(Z)) Rcpp::stop("Z contains missing or infinite values");
  if (!all_finite(XtX_null)) {
    Rcpp::stop("XtX_null contains missing or infinite values");
  }
  if (!all_finite(penalty_null)) {
    Rcpp::stop("penalty_null contains missing or infinite values");
  }
  if (!all_finite_vector(Xty_null)) {
    Rcpp::stop("Xty_null contains missing or infinite values");
  }

  const MgcvExtractGpuFixedSpResult result =
    mgcv_extract_fixed_sp_solve_cuda(
      REAL(Xs), n, p, REAL(ys), REAL(Zs), REAL(XtX_nulls),
      REAL(penalty_nulls), REAL(Xty_nulls), q);

  return Rcpp::List::create(
    Rcpp::Named("theta") =
      Rcpp::NumericVector(result.theta.begin(), result.theta.end()),
    Rcpp::Named("coefficients") =
      Rcpp::NumericVector(result.coefficients.begin(),
                          result.coefficients.end()),
    Rcpp::Named("fitted") =
      Rcpp::NumericVector(result.fitted.begin(), result.fitted.end()),
    Rcpp::Named("residuals") =
      Rcpp::NumericVector(result.residuals.begin(), result.residuals.end()),
    Rcpp::Named("rss") = result.rss,
    Rcpp::Named("diagnostics") = Rcpp::List::create(
      Rcpp::Named("n") = result.n,
      Rcpp::Named("coefficient_dim") = result.coefficient_dim,
      Rcpp::Named("null_dim") = result.null_dim,
      Rcpp::Named("solve_stage") = "native-gpu-handle-linear-solve",
      Rcpp::Named("cholesky_backend") = result.cholesky_backend,
      Rcpp::Named("runtime_version") = result.runtime_version,
      Rcpp::Named("compatibility_transient_context") =
        result.compatibility_transient_context,
      Rcpp::Named("planned_route") = result.planned_route,
      Rcpp::Named("executed_route") = result.executed_route,
      Rcpp::Named("solver_status") = result.solver_status,
      Rcpp::Named("cpu_fallback_count") = result.cpu_fallback_count,
      Rcpp::Named("rhs_device_build_count") =
        result.rhs_device_build_count,
      Rcpp::Named("rhs_authority") = result.rhs_authority,
      Rcpp::Named("full_cuda_data_plane") = result.full_cuda_data_plane
    )
  );
  END_RCPP
}

extern "C" SEXP C_mgcv_extract_gpu_solve_same_setup_batch_fixed_sp(
    SEXP Xs,
    SEXP Ys,
    SEXP Zs,
    SEXP XtX_nulls,
    SEXP penalty_null_list_s,
    SEXP Xty_nulls) {
  BEGIN_RCPP
  if (!Rf_isReal(Xs) || !Rf_isMatrix(Xs)) Rcpp::stop("X must be a numeric matrix");
  if (!Rf_isReal(Ys) || !Rf_isMatrix(Ys)) Rcpp::stop("Y must be a numeric matrix");
  if (!Rf_isReal(Zs) || !Rf_isMatrix(Zs)) Rcpp::stop("Z must be a numeric matrix");
  if (!Rf_isReal(XtX_nulls) || !Rf_isMatrix(XtX_nulls)) {
    Rcpp::stop("XtX_null must be a numeric matrix");
  }
  if (!Rf_isNewList(penalty_null_list_s)) {
    Rcpp::stop("penalty_null_list must be a list of numeric matrices");
  }
  if (!Rf_isReal(Xty_nulls) || !Rf_isMatrix(Xty_nulls)) {
    Rcpp::stop("Xty_null must be a numeric matrix");
  }

  Rcpp::NumericMatrix X(Xs);
  Rcpp::NumericMatrix Y(Ys);
  Rcpp::NumericMatrix Z(Zs);
  Rcpp::NumericMatrix XtX_null(XtX_nulls);
  Rcpp::List penalty_list(penalty_null_list_s);
  Rcpp::NumericMatrix Xty_null(Xty_nulls);

  const int n = X.nrow();
  const int p = X.ncol();
  const int targets = Y.ncol();
  const int q = Z.ncol();
  if (Y.nrow() != n) Rcpp::stop("nrow(Y) must equal nrow(X)");
  if (Z.nrow() != p) Rcpp::stop("nrow(Z) must equal ncol(X)");
  if (XtX_null.nrow() != q || XtX_null.ncol() != q) {
    Rcpp::stop("XtX_null dimensions must match ncol(Z)");
  }
  if (penalty_list.size() != targets) {
    Rcpp::stop("penalty_null_list length must equal ncol(Y)");
  }
  if (Xty_null.nrow() != q || Xty_null.ncol() != targets) {
    Rcpp::stop("Xty_null dimensions must be ncol(Z) by ncol(Y)");
  }
  if (!all_finite(X)) Rcpp::stop("X contains missing or infinite values");
  if (!all_finite(Y)) Rcpp::stop("Y contains missing or infinite values");
  if (!all_finite(Z)) Rcpp::stop("Z contains missing or infinite values");
  if (!all_finite(XtX_null)) {
    Rcpp::stop("XtX_null contains missing or infinite values");
  }
  if (!all_finite(Xty_null)) {
    Rcpp::stop("Xty_null contains missing or infinite values");
  }

  Rcpp::NumericMatrix residuals(n, targets);
  Rcpp::NumericMatrix fitted(n, targets);
  Rcpp::NumericMatrix theta(q, targets);
  Rcpp::NumericMatrix coefficients(p, targets);
  Rcpp::NumericVector rss(targets);
  Rcpp::List diagnostics(targets);

  for (int target = 0; target < targets; ++target) {
    Rcpp::NumericMatrix penalty(penalty_list[target]);
    if (penalty.nrow() != q || penalty.ncol() != q) {
      Rcpp::stop("each penalty_null matrix dimensions must match ncol(Z)");
    }
    if (!all_finite(penalty)) {
      Rcpp::stop("penalty_null contains missing or infinite values");
    }
    const MgcvExtractGpuFixedSpResult result =
      mgcv_extract_fixed_sp_solve_cuda(
        REAL(Xs), n, p, &REAL(Ys)[static_cast<std::size_t>(target) * n],
        REAL(Zs), REAL(XtX_nulls), REAL(penalty),
        &REAL(Xty_nulls)[static_cast<std::size_t>(target) * q], q);
    for (int row = 0; row < n; ++row) {
      fitted(row, target) = result.fitted[row];
      residuals(row, target) = result.residuals[row];
    }
    for (int j = 0; j < q; ++j) theta(j, target) = result.theta[j];
    for (int j = 0; j < p; ++j) coefficients(j, target) = result.coefficients[j];
    rss[target] = result.rss;
    diagnostics[target] = Rcpp::List::create(
      Rcpp::Named("n") = result.n,
      Rcpp::Named("coefficient_dim") = result.coefficient_dim,
      Rcpp::Named("null_dim") = result.null_dim,
      Rcpp::Named("target_index") = target + 1,
      Rcpp::Named("solve_stage") = "native-gpu-same-setup-batch-linear-solve",
      Rcpp::Named("cholesky_backend") = result.cholesky_backend
    );
  }

  return Rcpp::List::create(
    Rcpp::Named("theta") = theta,
    Rcpp::Named("coefficients") = coefficients,
    Rcpp::Named("fitted") = fitted,
    Rcpp::Named("residuals") = residuals,
    Rcpp::Named("rss") = rss,
    Rcpp::Named("diagnostics") = diagnostics,
    Rcpp::Named("batch_diagnostics") = Rcpp::List::create(
      Rcpp::Named("n") = n,
      Rcpp::Named("targets") = targets,
      Rcpp::Named("coefficient_dim") = p,
      Rcpp::Named("null_dim") = q,
      Rcpp::Named("native_batch_call") = true,
      Rcpp::Named("setup_reused") = true,
      Rcpp::Named("true_batched_kernel") = false,
      Rcpp::Named("batch_stage") = "native-same-setup-repeated-cuda-solve"
    )
  );
  END_RCPP
}

extern "C" SEXP C_fast_skeleton_cuda(SEXP data, SEXP alphas, SEXP max_ords,
                                      SEXP indexs, SEXP legacy_indexs,
                                      SEXP batch_sizes) {
  BEGIN_RCPP
  Rcpp::NumericMatrix matrix(data);
  SkeletonOptions options;
  options.alpha = Rf_asReal(alphas);
  options.max_conditioning_size = Rf_asInteger(max_ords);
  options.na_delete = true;
  options.stable = true;
  options.index = Rf_asReal(indexs);
  options.legacy_index = Rcpp::as<bool>(legacy_indexs);
  options.residual_cache_enabled = false;
  options.residual_backend_name = "linear";
  options.residual_device_requested = "cpu";
  options.cuda_residual_fallback = true;
  options.scheduler_requested = "legacy";
  options.residual_batch_size = 0;
  options.scheduler_diagnostics_enabled = true;
  options.fastspline_params = default_fastspline_params();
  apply_ci_options(&options, "dcc.gamma", Rcpp::List::create(),
                   Rcpp::List::create(), true);
  const int batch_size = Rf_asInteger(batch_sizes);
  const SkeletonResult result = run_skeleton_cuda_batch(matrix, options, batch_size);
  return skeleton_result_to_list(result, matrix.ncol());
  END_RCPP
}

extern "C" SEXP C_fast_skeleton_cuda_cached(SEXP data, SEXP alphas,
                                             SEXP max_ords, SEXP indexs,
                                             SEXP legacy_indexs,
                                             SEXP batch_sizes,
                                             SEXP residual_caches) {
  BEGIN_RCPP
  Rcpp::NumericMatrix matrix(data);
  SkeletonOptions options;
  options.alpha = Rf_asReal(alphas);
  options.max_conditioning_size = Rf_asInteger(max_ords);
  options.na_delete = true;
  options.stable = true;
  options.index = Rf_asReal(indexs);
  options.legacy_index = Rcpp::as<bool>(legacy_indexs);
  options.residual_cache_enabled = Rcpp::as<bool>(residual_caches);
  options.residual_backend_name = "linear";
  options.residual_device_requested = "cpu";
  options.cuda_residual_fallback = true;
  options.scheduler_requested = "legacy";
  options.residual_batch_size = 0;
  options.scheduler_diagnostics_enabled = true;
  options.fastspline_params = default_fastspline_params();
  apply_ci_options(&options, "dcc.gamma", Rcpp::List::create(),
                   Rcpp::List::create(), true);
  const int batch_size = Rf_asInteger(batch_sizes);
  const SkeletonResult result = run_skeleton_cuda_batch(matrix, options, batch_size);
  return skeleton_result_to_list(result, matrix.ncol());
  END_RCPP
}

extern "C" SEXP C_fast_skeleton_cuda_backend(SEXP data, SEXP alphas,
                                             SEXP max_ords, SEXP indexs,
                                             SEXP legacy_indexs,
                                             SEXP batch_sizes,
                                             SEXP residual_caches,
                                             SEXP residual_backends,
                                             SEXP residual_devices,
                                             SEXP residual_batch_sizes,
                                             SEXP schedulers,
                                             SEXP scheduler_diagnosticss,
                                             SEXP fastspline_paramss,
                                             SEXP cuda_residual_fallbacks,
                                             SEXP ci_methods,
                                             SEXP hsic_paramss,
                                             SEXP permutation_paramss,
                                             SEXP ci_diagnosticss) {
  BEGIN_RCPP
  Rcpp::NumericMatrix matrix(data);
  const std::string residual_backend = Rcpp::as<std::string>(residual_backends);
  const FastSplineParams fastspline_params =
    parse_fastspline_params(Rcpp::as<Rcpp::List>(fastspline_paramss));
  make_residual_backend_config(residual_backend, fastspline_params);

  SkeletonOptions options;
  options.alpha = Rf_asReal(alphas);
  options.max_conditioning_size = Rf_asInteger(max_ords);
  options.na_delete = true;
  options.stable = true;
  options.index = Rf_asReal(indexs);
  options.legacy_index = Rcpp::as<bool>(legacy_indexs);
  options.residual_cache_enabled = Rcpp::as<bool>(residual_caches);
  options.residual_backend_name = residual_backend;
  options.residual_device_requested = Rcpp::as<std::string>(residual_devices);
  options.cuda_residual_fallback = Rcpp::as<bool>(cuda_residual_fallbacks);
  options.scheduler_requested = Rcpp::as<std::string>(schedulers);
  options.residual_batch_size = Rf_asInteger(residual_batch_sizes);
  options.scheduler_diagnostics_enabled = Rcpp::as<bool>(scheduler_diagnosticss);
  options.fastspline_params = fastspline_params;
  apply_ci_options(&options, Rcpp::as<std::string>(ci_methods),
                   Rcpp::as<Rcpp::List>(hsic_paramss),
                   Rcpp::as<Rcpp::List>(permutation_paramss),
                   Rcpp::as<bool>(ci_diagnosticss));
  const int batch_size = Rf_asInteger(batch_sizes);
  const SkeletonResult result = run_skeleton_cuda_batch(matrix, options, batch_size);
  return skeleton_result_to_list(result, matrix.ncol());
  END_RCPP
}

extern "C" SEXP C_fast_kpc_wanpdag_cuda(SEXP data, SEXP alphas,
                                         SEXP max_ords, SEXP indexs,
                                         SEXP legacy_indexs,
                                         SEXP batch_sizes,
                                         SEXP residual_caches,
                                         SEXP residual_backends,
                                         SEXP residual_devices,
                                         SEXP orientation_devices,
                                         SEXP residual_batch_sizes,
                                         SEXP orientation_batch_sizes,
                                         SEXP schedulers,
                                         SEXP scheduler_diagnosticss,
                                         SEXP orientation_diagnosticss,
                                         SEXP fastspline_paramss,
                                         SEXP cuda_residual_fallbacks,
                                         SEXP orient_colliders,
                                         SEXP solve_confls,
                                         SEXP ruless,
                                         SEXP ci_methods,
                                         SEXP hsic_paramss,
                                         SEXP permutation_paramss,
                                         SEXP ci_diagnosticss) {
  BEGIN_RCPP
  Rcpp::NumericMatrix matrix(data);
  const double alpha = Rf_asReal(alphas);
  const double index = Rf_asReal(indexs);
  const bool legacy_index = Rcpp::as<bool>(legacy_indexs);
  const bool residual_cache = Rcpp::as<bool>(residual_caches);
  const std::string residual_backend = Rcpp::as<std::string>(residual_backends);
  const std::string residual_device = Rcpp::as<std::string>(residual_devices);
  const std::string orientation_device =
    Rcpp::as<std::string>(orientation_devices);
  const FastSplineParams fastspline_params =
    parse_fastspline_params(Rcpp::as<Rcpp::List>(fastspline_paramss));
  make_residual_backend_config(residual_backend, fastspline_params);

  SkeletonOptions skeleton_options;
  skeleton_options.alpha = alpha;
  skeleton_options.max_conditioning_size = Rf_asInteger(max_ords);
  skeleton_options.na_delete = true;
  skeleton_options.stable = true;
  skeleton_options.index = index;
  skeleton_options.legacy_index = legacy_index;
  skeleton_options.residual_cache_enabled = residual_cache;
  skeleton_options.residual_backend_name = residual_backend;
  skeleton_options.residual_device_requested = residual_device;
  skeleton_options.cuda_residual_fallback = Rcpp::as<bool>(cuda_residual_fallbacks);
  skeleton_options.scheduler_requested = Rcpp::as<std::string>(schedulers);
  skeleton_options.residual_batch_size = Rf_asInteger(residual_batch_sizes);
  skeleton_options.scheduler_diagnostics_enabled =
    Rcpp::as<bool>(scheduler_diagnosticss);
  skeleton_options.fastspline_params = fastspline_params;
  apply_ci_options(&skeleton_options, Rcpp::as<std::string>(ci_methods),
                   Rcpp::as<Rcpp::List>(hsic_paramss),
                   Rcpp::as<Rcpp::List>(permutation_paramss),
                   Rcpp::as<bool>(ci_diagnosticss));

  const int batch_size = Rf_asInteger(batch_sizes);
  const SkeletonResult skeleton =
    run_skeleton_cuda_batch(matrix, skeleton_options, batch_size);
  const OrientationOptions orientation_options = make_orientation_options(
    alpha, index, legacy_index, residual_cache, residual_backend,
    orientation_device, Rf_asInteger(orientation_batch_sizes),
    Rcpp::as<bool>(orientation_diagnosticss),
    Rcpp::as<bool>(cuda_residual_fallbacks),
    fastspline_params, Rcpp::as<bool>(orient_colliders),
    Rcpp::as<bool>(solve_confls), Rcpp::as<Rcpp::LogicalVector>(ruless),
    Rcpp::as<std::string>(ci_methods), Rcpp::as<Rcpp::List>(hsic_paramss),
    Rcpp::as<Rcpp::List>(permutation_paramss),
    Rcpp::as<bool>(ci_diagnosticss));
  const OrientationResult orientation =
    orient_wanpdag_native(matrix, skeleton, orientation_options,
                          regrvonps_device);
  Rcpp::List orientation_list = orientation_result_to_list(orientation);
  return Rcpp::List::create(
    Rcpp::Named("skeleton") = skeleton_result_to_list(skeleton, matrix.ncol()),
    Rcpp::Named("orientation") = orientation_list
  );
  END_RCPP
}

extern "C" SEXP C_precision_replay_layer_native(SEXP adjacencys,
                                                SEXP pmaxs,
                                                SEXP edge_xs,
                                                SEXP edge_ys,
                                                SEXP xs,
                                                SEXP ys,
                                                SEXP conditioning_setss,
                                                SEXP p_values,
                                                SEXP alphas,
                                                SEXP trace_levels) {
  BEGIN_RCPP
  Rcpp::IntegerMatrix adjacency_in(adjacencys);
  Rcpp::NumericMatrix pmax_in(pmaxs);
  Rcpp::IntegerVector edge_x(edge_xs);
  Rcpp::IntegerVector edge_y(edge_ys);
  Rcpp::IntegerVector x(xs);
  Rcpp::IntegerVector y(ys);
  Rcpp::List conditioning_sets(conditioning_setss);
  Rcpp::NumericVector pvals(p_values);
  const double alpha = Rf_asReal(alphas);
  const std::string trace_level = Rcpp::as<std::string>(trace_levels);
  const bool full_trace = trace_level == "full";

  const int p = adjacency_in.nrow();
  if (adjacency_in.ncol() != p || pmax_in.nrow() != p || pmax_in.ncol() != p) {
    Rcpp::stop("adjacency and pmax must be square matrices with matching dimensions");
  }
  const int n_tasks = pvals.size();
  if (edge_x.size() != n_tasks || edge_y.size() != n_tasks ||
      x.size() != n_tasks || y.size() != n_tasks ||
      conditioning_sets.size() != n_tasks) {
    Rcpp::stop("task arrays must have the same length");
  }

  std::vector<int> adjacency(static_cast<std::size_t>(p) * p, 0);
  std::vector<double> pmax(static_cast<std::size_t>(p) * p, 0.0);
  std::vector<std::vector<std::vector<int> > > sepsets(
    p, std::vector<std::vector<int> >(p));
  for (int row = 0; row < p; ++row) {
    for (int col = 0; col < p; ++col) {
      adjacency[static_cast<std::size_t>(row) * p + col] =
        adjacency_in(row, col) != 0 ? 1 : 0;
      pmax[static_cast<std::size_t>(row) * p + col] = pmax_in(row, col);
    }
  }

  Rcpp::IntegerVector trace_id, trace_edge_x, trace_edge_y, trace_x, trace_y;
  Rcpp::CharacterVector trace_s_key;
  Rcpp::NumericVector trace_p;
  Rcpp::LogicalVector trace_deleted, trace_ignored;
  Rcpp::IntegerVector replayed_task_index, ignored_task_index, deleted_task_index;

  int tests_replayed = 0;
  int ignored_after_delete = 0;
  int deletions = 0;
  std::map<int, bool> edge_done;
  std::vector<int> delete_edges(static_cast<std::size_t>(p) * p, 0);
  std::vector<LevelDeletion> level_log;

  for (int i = 0; i < n_tasks; ++i) {
    const int ex = edge_x[i] - 1;
    const int ey = edge_y[i] - 1;
    const int ox = x[i] - 1;
    const int oy = y[i] - 1;
    if (ex < 0 || ex >= p || ey < 0 || ey >= p ||
        ox < 0 || ox >= p || oy < 0 || oy >= p) {
      Rcpp::stop("task vertex index out of range");
    }
    const int edge_key = ex < ey ? ex * p + ey : ey * p + ex;
    Rcpp::IntegerVector cond_in = conditioning_sets[i];
    std::vector<int> cond;
    cond.reserve(cond_in.size());
    for (int j = 0; j < cond_in.size(); ++j) {
      if (Rcpp::IntegerVector::is_na(cond_in[j])) {
        Rcpp::stop("conditioning set contains NA");
      }
      const int value = cond_in[j] - 1;
      if (value < 0 || value >= p) Rcpp::stop("conditioning set index out of range");
      cond.push_back(value);
    }

    const bool ignored = edge_done[edge_key] ||
      adjacency[static_cast<std::size_t>(ex) * p + ey] == 0;
    if (ignored) {
      ++ignored_after_delete;
      ignored_task_index.push_back(i + 1);
      if (full_trace) {
        trace_id.push_back(i + 1);
        trace_edge_x.push_back(ex + 1);
        trace_edge_y.push_back(ey + 1);
        trace_x.push_back(ox + 1);
        trace_y.push_back(oy + 1);
        trace_s_key.push_back(replay_s_key(cond));
        trace_p.push_back(NA_REAL);
        trace_deleted.push_back(false);
        trace_ignored.push_back(true);
      }
      continue;
    }

    ++tests_replayed;
    replayed_task_index.push_back(i + 1);
    double pval = pvals[i];
    if (!std::isfinite(pval)) pval = 1.0;
    const std::size_t pmax_idx = static_cast<std::size_t>(ex) * p + ey;
    const std::size_t pmax_rev = static_cast<std::size_t>(ey) * p + ex;
    if (pval > pmax[pmax_idx]) {
      pmax[pmax_idx] = pval;
      pmax[pmax_rev] = pval;
    }

    const bool deleted = pval >= alpha;
    if (deleted) {
      ++deletions;
      deleted_task_index.push_back(i + 1);
      delete_edges[static_cast<std::size_t>(ex) * p + ey] = 1;
      delete_edges[static_cast<std::size_t>(ey) * p + ex] = 1;
      sepsets[ex][ey] = cond;
      sepsets[ey][ex] = cond;
      level_log.push_back(LevelDeletion{ex, ey, cond, pval});
      edge_done[edge_key] = true;
    }

    if (full_trace) {
      trace_id.push_back(i + 1);
      trace_edge_x.push_back(ex + 1);
      trace_edge_y.push_back(ey + 1);
      trace_x.push_back(ox + 1);
      trace_y.push_back(oy + 1);
      trace_s_key.push_back(replay_s_key(cond));
      trace_p.push_back(pval);
      trace_deleted.push_back(deleted);
      trace_ignored.push_back(false);
    }
  }

  for (int i = 0; i < p * p; ++i) {
    if (delete_edges[i] != 0) adjacency[i] = 0;
  }

  Rcpp::DataFrame replay_rows = Rcpp::DataFrame::create(
    Rcpp::Named("task_index") = trace_id,
    Rcpp::Named("edge_x") = trace_edge_x,
    Rcpp::Named("edge_y") = trace_edge_y,
    Rcpp::Named("x") = trace_x,
    Rcpp::Named("y") = trace_y,
    Rcpp::Named("S_key") = trace_s_key,
    Rcpp::Named("p_used") = trace_p,
    Rcpp::Named("edge_deleted") = trace_deleted,
    Rcpp::Named("edge_already_deleted") = trace_ignored,
    Rcpp::Named("stringsAsFactors") = false
  );

  return Rcpp::List::create(
    Rcpp::Named("adjacency") = adjacency_to_matrix(adjacency, p),
    Rcpp::Named("sepsets") = sepsets_to_list(sepsets),
    Rcpp::Named("pMax") = pmax_to_matrix(pmax, p),
    Rcpp::Named("n.edgetests") = tests_replayed,
    Rcpp::Named("per.level.log") =
      level_log_to_list(std::vector<std::vector<LevelDeletion> >(1, level_log))[0],
    Rcpp::Named("summary") = Rcpp::List::create(
      Rcpp::Named("tasks_planned") = n_tasks,
      Rcpp::Named("tests_replayed") = tests_replayed,
      Rcpp::Named("tasks_ignored_after_delete") = ignored_after_delete,
      Rcpp::Named("deletions") = deletions
    ),
    Rcpp::Named("replayed_task_index") = replayed_task_index,
    Rcpp::Named("ignored_task_index") = ignored_task_index,
    Rcpp::Named("deleted_task_index") = deleted_task_index,
    Rcpp::Named("replay_rows") = replay_rows
  );
  END_RCPP
}

extern "C" SEXP C_precision_make_layer_plan_native(SEXP adjacencys,
                                                   SEXP levels) {
  BEGIN_RCPP
  Rcpp::IntegerMatrix adjacency_in(adjacencys);
  const int p = adjacency_in.nrow();
  const int level = Rf_asInteger(levels);
  if (p <= 0 || adjacency_in.ncol() != p) {
    Rcpp::stop("adjacency must be a square matrix");
  }
  if (level < 0) {
    Rcpp::stop("level must be non-negative");
  }

  std::vector<int> adjacency(static_cast<std::size_t>(p) * p, 0);
  for (int row = 0; row < p; ++row) {
    for (int col = 0; col < p; ++col) {
      adjacency[static_cast<std::size_t>(row) * p + col] =
        adjacency_in(row, col) != 0 ? 1 : 0;
    }
  }

  const LayerPlan plan = make_layer_plan(adjacency, p, level);
  Rcpp::List tasks(plan.tasks.size());
  for (int i = 0; i < static_cast<int>(plan.tasks.size()); ++i) {
    const LayerCiTask& task = plan.tasks[i];
    Rcpp::IntegerVector cond(task.conditioning_set.size());
    for (int j = 0; j < cond.size(); ++j) cond[j] = task.conditioning_set[j] + 1;
    tasks[i] = Rcpp::List::create(
      Rcpp::Named("task_id") = task.task_id + 1,
      Rcpp::Named("edge_x") = task.edge_x + 1,
      Rcpp::Named("edge_y") = task.edge_y + 1,
      Rcpp::Named("x") = task.orientation_x + 1,
      Rcpp::Named("y") = task.orientation_y + 1,
      Rcpp::Named("S") = cond,
      Rcpp::Named("S_key") = replay_s_key(task.conditioning_set),
      Rcpp::Named("conditioning_size") =
        static_cast<int>(task.conditioning_set.size()),
      Rcpp::Named("conditioning_target_side") =
        task.orientation_x == task.edge_x ? "x" : "y"
    );
  }

  return Rcpp::List::create(
    Rcpp::Named("tasks") = tasks,
    Rcpp::Named("summary") = Rcpp::List::create(
      Rcpp::Named("level") = plan.level,
      Rcpp::Named("p") = plan.p,
      Rcpp::Named("tasks_planned") = static_cast<int>(plan.tasks.size()),
      Rcpp::Named("unconditional_tasks") = plan.unconditional_tasks,
      Rcpp::Named("conditional_tasks") = plan.conditional_tasks
    )
  );
  END_RCPP
}

extern "C" SEXP C_precision_run_skeleton_ptable_native(SEXP ps,
                                                       SEXP alphas,
                                                       SEXP max_conditioning_sizes,
                                                       SEXP trace_levels) {
  BEGIN_RCPP
  const int p = Rf_asInteger(ps);
  const double alpha = Rf_asReal(alphas);
  const int max_conditioning_size = Rf_asInteger(max_conditioning_sizes);
  const std::string trace_level = Rcpp::as<std::string>(trace_levels);
  const bool full_trace = trace_level == "full";
  if (p < 3) {
    Rcpp::stop("p-table native skeleton requires p >= 3");
  }
  if (!std::isfinite(alpha) || alpha <= 0.0) {
    Rcpp::stop("alpha must be a positive finite value");
  }
  if (max_conditioning_size < 0) {
    Rcpp::stop("max_conditioning_size must be non-negative");
  }

  std::vector<int> adjacency(static_cast<std::size_t>(p) * p, 1);
  std::vector<double> pmax(static_cast<std::size_t>(p) * p,
                           -std::numeric_limits<double>::infinity());
  std::vector<std::vector<std::vector<int> > > sepsets(
    p, std::vector<std::vector<int> >(p));
  for (int i = 0; i < p; ++i) {
    adjacency[static_cast<std::size_t>(i) * p + i] = 0;
    pmax[static_cast<std::size_t>(i) * p + i] = 1.0;
  }

  Rcpp::IntegerVector n_edgetests;
  Rcpp::IntegerVector level_level, level_tasks_planned, level_tests_replayed,
    level_ignored, level_deletions;

  Rcpp::IntegerVector task_global_id, task_level, task_index, task_edge_x,
    task_edge_y, task_x, task_y, task_conditioning_size;
  Rcpp::CharacterVector task_s_key;
  Rcpp::NumericVector task_p_candidate;
  Rcpp::LogicalVector task_deleted, task_ignored;

  int global_task_id = 0;
  int total_tasks_planned = 0;
  int total_tests_replayed = 0;
  int total_ignored = 0;
  int total_deletions = 0;

  for (int level = 0; level <= max_conditioning_size; ++level) {
    const LayerPlan plan = make_layer_plan(adjacency, p, level);
    std::map<int, bool> edge_done;
    std::vector<int> delete_edges(static_cast<std::size_t>(p) * p, 0);
    int tests_replayed = 0;
    int ignored_after_delete = 0;
    int deletions = 0;

    for (int i = 0; i < static_cast<int>(plan.tasks.size()); ++i) {
      const LayerCiTask& task = plan.tasks[i];
      const int edge_key = task.edge_x < task.edge_y
        ? task.edge_x * p + task.edge_y
        : task.edge_y * p + task.edge_x;
      const bool ignored = edge_done[edge_key] ||
        adjacency[static_cast<std::size_t>(task.edge_x) * p + task.edge_y] == 0;
      const double p_candidate = native_ptable_p_for_task(task, alpha);
      bool deleted = false;

      if (ignored) {
        ++ignored_after_delete;
      } else {
        ++tests_replayed;
        double pval = p_candidate;
        if (!std::isfinite(pval)) pval = 1.0;
        const std::size_t pmax_idx =
          static_cast<std::size_t>(task.edge_x) * p + task.edge_y;
        const std::size_t pmax_rev =
          static_cast<std::size_t>(task.edge_y) * p + task.edge_x;
        if (pval > pmax[pmax_idx]) {
          pmax[pmax_idx] = pval;
          pmax[pmax_rev] = pval;
        }
        deleted = pval >= alpha;
        if (deleted) {
          ++deletions;
          delete_edges[static_cast<std::size_t>(task.edge_x) * p +
                       task.edge_y] = 1;
          delete_edges[static_cast<std::size_t>(task.edge_y) * p +
                       task.edge_x] = 1;
          sepsets[task.edge_x][task.edge_y] = task.conditioning_set;
          sepsets[task.edge_y][task.edge_x] = task.conditioning_set;
          edge_done[edge_key] = true;
        }
      }

      if (full_trace) {
        ++global_task_id;
        task_global_id.push_back(global_task_id);
        task_level.push_back(level);
        task_index.push_back(i + 1);
        task_edge_x.push_back(task.edge_x + 1);
        task_edge_y.push_back(task.edge_y + 1);
        task_x.push_back(task.orientation_x + 1);
        task_y.push_back(task.orientation_y + 1);
        task_s_key.push_back(replay_s_key(task.conditioning_set));
        task_conditioning_size.push_back(
          static_cast<int>(task.conditioning_set.size()));
        task_p_candidate.push_back(p_candidate);
        task_deleted.push_back(deleted);
        task_ignored.push_back(ignored);
      }
    }

    for (int i = 0; i < p * p; ++i) {
      if (delete_edges[i] != 0) adjacency[i] = 0;
    }

    total_tasks_planned += static_cast<int>(plan.tasks.size());
    total_tests_replayed += tests_replayed;
    total_ignored += ignored_after_delete;
    total_deletions += deletions;
    n_edgetests.push_back(tests_replayed);
    level_level.push_back(level);
    level_tasks_planned.push_back(static_cast<int>(plan.tasks.size()));
    level_tests_replayed.push_back(tests_replayed);
    level_ignored.push_back(ignored_after_delete);
    level_deletions.push_back(deletions);
  }

  Rcpp::DataFrame task_rows = Rcpp::DataFrame::create(
    Rcpp::Named("canonical_test_order_id") = task_global_id,
    Rcpp::Named("level") = task_level,
    Rcpp::Named("task_index") = task_index,
    Rcpp::Named("edge_x") = task_edge_x,
    Rcpp::Named("edge_y") = task_edge_y,
    Rcpp::Named("x") = task_x,
    Rcpp::Named("y") = task_y,
    Rcpp::Named("S_key") = task_s_key,
    Rcpp::Named("conditioning_size") = task_conditioning_size,
    Rcpp::Named("p_candidate") = task_p_candidate,
    Rcpp::Named("native_edge_deleted") = task_deleted,
    Rcpp::Named("native_edge_ignored") = task_ignored,
    Rcpp::Named("stringsAsFactors") = false
  );
  Rcpp::DataFrame level_rows = Rcpp::DataFrame::create(
    Rcpp::Named("level") = level_level,
    Rcpp::Named("tasks_planned") = level_tasks_planned,
    Rcpp::Named("tests_replayed") = level_tests_replayed,
    Rcpp::Named("tasks_ignored_after_delete") = level_ignored,
    Rcpp::Named("deletions") = level_deletions,
    Rcpp::Named("stringsAsFactors") = false
  );

  return Rcpp::List::create(
    Rcpp::Named("adjacency") = adjacency_to_matrix(adjacency, p),
    Rcpp::Named("sepsets") = sepsets_to_list(sepsets),
    Rcpp::Named("pMax") = pmax_to_matrix(pmax, p),
    Rcpp::Named("n.edgetests") = n_edgetests,
    Rcpp::Named("tasks") = task_rows,
    Rcpp::Named("levels") = level_rows,
    Rcpp::Named("summary") = Rcpp::List::create(
      Rcpp::Named("p") = p,
      Rcpp::Named("alpha") = alpha,
      Rcpp::Named("max_conditioning_size") = max_conditioning_size,
      Rcpp::Named("levels") = max_conditioning_size + 1,
      Rcpp::Named("tasks_planned") = total_tasks_planned,
      Rcpp::Named("tests_replayed") = total_tests_replayed,
      Rcpp::Named("tasks_ignored_after_delete") = total_ignored,
      Rcpp::Named("deletions") = total_deletions
    )
  );
  END_RCPP
}

extern "C" SEXP C_precision_run_skeleton_provider_native(SEXP ps,
                                                         SEXP alphas,
                                                         SEXP max_conditioning_sizes,
                                                         SEXP providers,
                                                         SEXP trace_levels) {
  BEGIN_RCPP
  const int p = Rf_asInteger(ps);
  const double alpha = Rf_asReal(alphas);
  const int max_conditioning_size = Rf_asInteger(max_conditioning_sizes);
  Rcpp::Function provider(providers);
  const std::string trace_level = Rcpp::as<std::string>(trace_levels);
  const bool full_trace = trace_level == "full";
  if (p < 2) {
    Rcpp::stop("provider native skeleton requires p >= 2");
  }
  if (!std::isfinite(alpha) || alpha <= 0.0) {
    Rcpp::stop("alpha must be a positive finite value");
  }
  if (max_conditioning_size < 0) {
    Rcpp::stop("max_conditioning_size must be non-negative");
  }

  std::vector<int> adjacency(static_cast<std::size_t>(p) * p, 1);
  std::vector<double> pmax(static_cast<std::size_t>(p) * p,
                           -std::numeric_limits<double>::infinity());
  std::vector<std::vector<std::vector<int> > > sepsets(
    p, std::vector<std::vector<int> >(p));
  for (int i = 0; i < p; ++i) {
    adjacency[static_cast<std::size_t>(i) * p + i] = 0;
    pmax[static_cast<std::size_t>(i) * p + i] = 1.0;
  }

  Rcpp::IntegerVector n_edgetests;
  Rcpp::IntegerVector level_level, level_tasks_planned, level_tests_replayed,
    level_ignored, level_deletions;

  Rcpp::IntegerVector task_global_id, task_level, task_index, task_edge_x,
    task_edge_y, task_x, task_y, task_conditioning_size;
  Rcpp::CharacterVector task_s_key;
  Rcpp::NumericVector task_p_used;
  Rcpp::LogicalVector task_deleted, task_ignored;

  int global_task_id = 0;
  int total_tasks_planned = 0;
  int total_tests_replayed = 0;
  int total_ignored = 0;
  int total_deletions = 0;

  for (int level = 0; level <= max_conditioning_size; ++level) {
    const LayerPlan plan = make_layer_plan(adjacency, p, level);
    const int task_count = static_cast<int>(plan.tasks.size());

    Rcpp::IntegerVector provider_task_index(task_count), provider_edge_x(task_count),
      provider_edge_y(task_count), provider_x(task_count), provider_y(task_count),
      provider_conditioning_size(task_count);
    Rcpp::CharacterVector provider_s_key(task_count);
    Rcpp::List provider_conditioning_sets(task_count);
    for (int i = 0; i < task_count; ++i) {
      const LayerCiTask& task = plan.tasks[i];
      Rcpp::IntegerVector cond(task.conditioning_set.size());
      for (int j = 0; j < cond.size(); ++j) cond[j] = task.conditioning_set[j] + 1;
      provider_task_index[i] = i + 1;
      provider_edge_x[i] = task.edge_x + 1;
      provider_edge_y[i] = task.edge_y + 1;
      provider_x[i] = task.orientation_x + 1;
      provider_y[i] = task.orientation_y + 1;
      provider_conditioning_sets[i] = cond;
      provider_s_key[i] = replay_s_key(task.conditioning_set);
      provider_conditioning_size[i] =
        static_cast<int>(task.conditioning_set.size());
    }
    provider_conditioning_sets.attr("class") = "AsIs";
    Rcpp::DataFrame provider_tasks = Rcpp::DataFrame::create(
      Rcpp::Named("task_index") = provider_task_index,
      Rcpp::Named("edge_x") = provider_edge_x,
      Rcpp::Named("edge_y") = provider_edge_y,
      Rcpp::Named("x") = provider_x,
      Rcpp::Named("y") = provider_y,
      Rcpp::Named("conditioning_sets") = provider_conditioning_sets,
      Rcpp::Named("S_key") = provider_s_key,
      Rcpp::Named("conditioning_size") = provider_conditioning_size,
      Rcpp::Named("stringsAsFactors") = false
    );
    Rcpp::NumericVector pvalues = provider(provider_tasks, level);
    if (pvalues.size() != task_count) {
      Rcpp::stop("provider returned p-value vector with wrong length");
    }

    std::map<int, bool> edge_done;
    std::vector<int> delete_edges(static_cast<std::size_t>(p) * p, 0);
    int tests_replayed = 0;
    int ignored_after_delete = 0;
    int deletions = 0;

    for (int i = 0; i < task_count; ++i) {
      const LayerCiTask& task = plan.tasks[i];
      const int edge_key = task.edge_x < task.edge_y
        ? task.edge_x * p + task.edge_y
        : task.edge_y * p + task.edge_x;
      const bool ignored = edge_done[edge_key] ||
        adjacency[static_cast<std::size_t>(task.edge_x) * p + task.edge_y] == 0;
      bool deleted = false;
      double pval = NA_REAL;

      if (ignored) {
        ++ignored_after_delete;
      } else {
        ++tests_replayed;
        pval = pvalues[i];
        if (!std::isfinite(pval)) pval = 1.0;
        const std::size_t pmax_idx =
          static_cast<std::size_t>(task.edge_x) * p + task.edge_y;
        const std::size_t pmax_rev =
          static_cast<std::size_t>(task.edge_y) * p + task.edge_x;
        if (pval > pmax[pmax_idx]) {
          pmax[pmax_idx] = pval;
          pmax[pmax_rev] = pval;
        }
        deleted = pval >= alpha;
        if (deleted) {
          ++deletions;
          delete_edges[static_cast<std::size_t>(task.edge_x) * p +
                       task.edge_y] = 1;
          delete_edges[static_cast<std::size_t>(task.edge_y) * p +
                       task.edge_x] = 1;
          sepsets[task.edge_x][task.edge_y] = task.conditioning_set;
          sepsets[task.edge_y][task.edge_x] = task.conditioning_set;
          edge_done[edge_key] = true;
        }
      }

      if (full_trace) {
        ++global_task_id;
        task_global_id.push_back(global_task_id);
        task_level.push_back(level);
        task_index.push_back(i + 1);
        task_edge_x.push_back(task.edge_x + 1);
        task_edge_y.push_back(task.edge_y + 1);
        task_x.push_back(task.orientation_x + 1);
        task_y.push_back(task.orientation_y + 1);
        task_s_key.push_back(replay_s_key(task.conditioning_set));
        task_conditioning_size.push_back(
          static_cast<int>(task.conditioning_set.size()));
        task_p_used.push_back(pval);
        task_deleted.push_back(deleted);
        task_ignored.push_back(ignored);
      }
    }

    for (int i = 0; i < p * p; ++i) {
      if (delete_edges[i] != 0) adjacency[i] = 0;
    }

    total_tasks_planned += task_count;
    total_tests_replayed += tests_replayed;
    total_ignored += ignored_after_delete;
    total_deletions += deletions;
    n_edgetests.push_back(tests_replayed);
    level_level.push_back(level);
    level_tasks_planned.push_back(task_count);
    level_tests_replayed.push_back(tests_replayed);
    level_ignored.push_back(ignored_after_delete);
    level_deletions.push_back(deletions);
  }

  Rcpp::DataFrame task_rows = Rcpp::DataFrame::create(
    Rcpp::Named("canonical_test_order_id") = task_global_id,
    Rcpp::Named("level") = task_level,
    Rcpp::Named("task_index") = task_index,
    Rcpp::Named("edge_x") = task_edge_x,
    Rcpp::Named("edge_y") = task_edge_y,
    Rcpp::Named("x") = task_x,
    Rcpp::Named("y") = task_y,
    Rcpp::Named("S_key") = task_s_key,
    Rcpp::Named("conditioning_size") = task_conditioning_size,
    Rcpp::Named("p_used") = task_p_used,
    Rcpp::Named("native_edge_deleted") = task_deleted,
    Rcpp::Named("native_edge_ignored") = task_ignored,
    Rcpp::Named("stringsAsFactors") = false
  );
  Rcpp::DataFrame level_rows = Rcpp::DataFrame::create(
    Rcpp::Named("level") = level_level,
    Rcpp::Named("tasks_planned") = level_tasks_planned,
    Rcpp::Named("tests_replayed") = level_tests_replayed,
    Rcpp::Named("tasks_ignored_after_delete") = level_ignored,
    Rcpp::Named("deletions") = level_deletions,
    Rcpp::Named("stringsAsFactors") = false
  );

  return Rcpp::List::create(
    Rcpp::Named("adjacency") = adjacency_to_matrix(adjacency, p),
    Rcpp::Named("sepsets") = sepsets_to_list(sepsets),
    Rcpp::Named("pMax") = pmax_to_matrix(pmax, p),
    Rcpp::Named("n.edgetests") = n_edgetests,
    Rcpp::Named("tasks") = task_rows,
    Rcpp::Named("levels") = level_rows,
    Rcpp::Named("summary") = Rcpp::List::create(
      Rcpp::Named("p") = p,
      Rcpp::Named("alpha") = alpha,
      Rcpp::Named("max_conditioning_size") = max_conditioning_size,
      Rcpp::Named("levels") = max_conditioning_size + 1,
      Rcpp::Named("tasks_planned") = total_tasks_planned,
      Rcpp::Named("tests_replayed") = total_tests_replayed,
      Rcpp::Named("tasks_ignored_after_delete") = total_ignored,
      Rcpp::Named("deletions") = total_deletions
    )
  );
  END_RCPP
}

extern "C" SEXP C_precision_run_skeleton_dcov0_native(SEXP datas,
                                                      SEXP alphas,
                                                      SEXP indexs,
                                                      SEXP legacy_indexs,
                                                      SEXP trace_levels) {
  BEGIN_RCPP
  Rcpp::NumericMatrix data(datas);
  const int n = data.nrow();
  const int p = data.ncol();
  const double alpha = Rf_asReal(alphas);
  const double index = Rf_asReal(indexs);
  const bool legacy_index = Rcpp::as<bool>(legacy_indexs);
  const std::string trace_level = Rcpp::as<std::string>(trace_levels);
  const bool full_trace = trace_level == "full";
  if (n <= 5) {
    Rcpp::stop("native dCov0 skeleton requires n > 5");
  }
  if (p < 2) {
    Rcpp::stop("native dCov0 skeleton requires at least two columns");
  }
  if (!std::isfinite(alpha) || alpha <= 0.0) {
    Rcpp::stop("alpha must be a positive finite value");
  }
  if (!all_finite(data)) {
    Rcpp::stop("Data contains missing or infinite values");
  }

  std::vector<int> adjacency(static_cast<std::size_t>(p) * p, 1);
  std::vector<double> pmax(static_cast<std::size_t>(p) * p,
                           -std::numeric_limits<double>::infinity());
  std::vector<std::vector<std::vector<int> > > sepsets(
    p, std::vector<std::vector<int> >(p));
  for (int i = 0; i < p; ++i) {
    adjacency[static_cast<std::size_t>(i) * p + i] = 0;
    pmax[static_cast<std::size_t>(i) * p + i] = 1.0;
  }

  const LayerPlan plan = make_layer_plan(adjacency, p, 0);
  const int task_count = static_cast<int>(plan.tasks.size());
  std::vector<double> pvalues(task_count, NA_REAL);
  for (int i = 0; i < task_count; ++i) {
    const LayerCiTask& task = plan.tasks[i];
    pvalues[i] = dcov_exact_pvalue(
      numeric_matrix_column(data, task.orientation_x),
      numeric_matrix_column(data, task.orientation_y),
      index,
      legacy_index);
  }

  Rcpp::IntegerVector task_global_id, task_level, task_index, task_edge_x,
    task_edge_y, task_x, task_y, task_conditioning_size;
  Rcpp::CharacterVector task_s_key;
  Rcpp::NumericVector task_p_used;
  Rcpp::LogicalVector task_deleted, task_ignored;

  std::map<int, bool> edge_done;
  std::vector<int> delete_edges(static_cast<std::size_t>(p) * p, 0);
  int tests_replayed = 0;
  int ignored_after_delete = 0;
  int deletions = 0;

  for (int i = 0; i < task_count; ++i) {
    const LayerCiTask& task = plan.tasks[i];
    const int edge_key = task.edge_x < task.edge_y
      ? task.edge_x * p + task.edge_y
      : task.edge_y * p + task.edge_x;
    const bool ignored = edge_done[edge_key] ||
      adjacency[static_cast<std::size_t>(task.edge_x) * p + task.edge_y] == 0;
    bool deleted = false;
    double pval = NA_REAL;

    if (ignored) {
      ++ignored_after_delete;
    } else {
      ++tests_replayed;
      pval = pvalues[i];
      if (!std::isfinite(pval)) pval = 1.0;
      const std::size_t pmax_idx =
        static_cast<std::size_t>(task.edge_x) * p + task.edge_y;
      const std::size_t pmax_rev =
        static_cast<std::size_t>(task.edge_y) * p + task.edge_x;
      if (pval > pmax[pmax_idx]) {
        pmax[pmax_idx] = pval;
        pmax[pmax_rev] = pval;
      }
      deleted = pval >= alpha;
      if (deleted) {
        ++deletions;
        delete_edges[static_cast<std::size_t>(task.edge_x) * p +
                     task.edge_y] = 1;
        delete_edges[static_cast<std::size_t>(task.edge_y) * p +
                     task.edge_x] = 1;
        sepsets[task.edge_x][task.edge_y] = task.conditioning_set;
        sepsets[task.edge_y][task.edge_x] = task.conditioning_set;
        edge_done[edge_key] = true;
      }
    }

    if (full_trace) {
      task_global_id.push_back(i + 1);
      task_level.push_back(0);
      task_index.push_back(i + 1);
      task_edge_x.push_back(task.edge_x + 1);
      task_edge_y.push_back(task.edge_y + 1);
      task_x.push_back(task.orientation_x + 1);
      task_y.push_back(task.orientation_y + 1);
      task_s_key.push_back(replay_s_key(task.conditioning_set));
      task_conditioning_size.push_back(
        static_cast<int>(task.conditioning_set.size()));
      task_p_used.push_back(pval);
      task_deleted.push_back(deleted);
      task_ignored.push_back(ignored);
    }
  }

  for (int i = 0; i < p * p; ++i) {
    if (delete_edges[i] != 0) adjacency[i] = 0;
  }

  Rcpp::DataFrame task_rows = Rcpp::DataFrame::create(
    Rcpp::Named("canonical_test_order_id") = task_global_id,
    Rcpp::Named("level") = task_level,
    Rcpp::Named("task_index") = task_index,
    Rcpp::Named("edge_x") = task_edge_x,
    Rcpp::Named("edge_y") = task_edge_y,
    Rcpp::Named("x") = task_x,
    Rcpp::Named("y") = task_y,
    Rcpp::Named("S_key") = task_s_key,
    Rcpp::Named("conditioning_size") = task_conditioning_size,
    Rcpp::Named("p_used") = task_p_used,
    Rcpp::Named("native_edge_deleted") = task_deleted,
    Rcpp::Named("native_edge_ignored") = task_ignored,
    Rcpp::Named("stringsAsFactors") = false
  );
  Rcpp::DataFrame level_rows = Rcpp::DataFrame::create(
    Rcpp::Named("level") = Rcpp::IntegerVector::create(0),
    Rcpp::Named("tasks_planned") = Rcpp::IntegerVector::create(task_count),
    Rcpp::Named("tests_replayed") = Rcpp::IntegerVector::create(tests_replayed),
    Rcpp::Named("tasks_ignored_after_delete") =
      Rcpp::IntegerVector::create(ignored_after_delete),
    Rcpp::Named("deletions") = Rcpp::IntegerVector::create(deletions),
    Rcpp::Named("stringsAsFactors") = false
  );

  return Rcpp::List::create(
    Rcpp::Named("adjacency") = adjacency_to_matrix(adjacency, p),
    Rcpp::Named("sepsets") = sepsets_to_list(sepsets),
    Rcpp::Named("pMax") = pmax_to_matrix(pmax, p),
    Rcpp::Named("n.edgetests") = Rcpp::IntegerVector::create(tests_replayed),
    Rcpp::Named("tasks") = task_rows,
    Rcpp::Named("levels") = level_rows,
    Rcpp::Named("summary") = Rcpp::List::create(
      Rcpp::Named("p") = p,
      Rcpp::Named("n") = n,
      Rcpp::Named("alpha") = alpha,
      Rcpp::Named("max_conditioning_size") = 0,
      Rcpp::Named("levels") = 1,
      Rcpp::Named("tasks_planned") = task_count,
      Rcpp::Named("tests_replayed") = tests_replayed,
      Rcpp::Named("tasks_ignored_after_delete") = ignored_after_delete,
      Rcpp::Named("deletions") = deletions,
      Rcpp::Named("dcov_native_count") = task_count,
      Rcpp::Named("ci_backend") = "native-exact-dcov"
    )
  );
  END_RCPP
}

extern "C" SEXP C_precision_run_skeleton_exact_ci_native(
    SEXP datas,
    SEXP alphas,
    SEXP max_conditioning_sizes,
    SEXP indexs,
    SEXP legacy_indexs,
    SEXP trace_levels) {
  BEGIN_RCPP
  Rcpp::NumericMatrix data(datas);
  const int n = data.nrow();
  const int p = data.ncol();
  const double alpha = Rf_asReal(alphas);
  const int max_conditioning_size = Rf_asInteger(max_conditioning_sizes);
  const double index = Rf_asReal(indexs);
  const bool legacy_index = Rcpp::as<bool>(legacy_indexs);
  const std::string trace_level = Rcpp::as<std::string>(trace_levels);
  const bool full_trace = trace_level == "full";
  if (n <= 5) {
    Rcpp::stop("native exact-CI skeleton requires n > 5");
  }
  if (p < 2) {
    Rcpp::stop("native exact-CI skeleton requires at least two columns");
  }
  if (!std::isfinite(alpha) || alpha <= 0.0) {
    Rcpp::stop("alpha must be a positive finite value");
  }
  if (max_conditioning_size < 0) {
    Rcpp::stop("max_conditioning_size must be non-negative");
  }
  if (!all_finite(data)) {
    Rcpp::stop("Data contains missing or infinite values");
  }

  std::vector<int> adjacency(static_cast<std::size_t>(p) * p, 1);
  std::vector<double> pmax(static_cast<std::size_t>(p) * p,
                           -std::numeric_limits<double>::infinity());
  std::vector<std::vector<std::vector<int> > > sepsets(
    p, std::vector<std::vector<int> >(p));
  for (int i = 0; i < p; ++i) {
    adjacency[static_cast<std::size_t>(i) * p + i] = 0;
    pmax[static_cast<std::size_t>(i) * p + i] = 1.0;
  }

  Rcpp::IntegerVector n_edgetests;
  Rcpp::IntegerVector level_level, level_tasks_planned, level_tests_replayed,
    level_ignored, level_deletions;
  Rcpp::IntegerVector task_global_id, task_level, task_index, task_edge_x,
    task_edge_y, task_x, task_y, task_conditioning_size;
  Rcpp::CharacterVector task_s_key;
  Rcpp::NumericVector task_p_used;
  Rcpp::LogicalVector task_deleted, task_ignored;

  int global_task_id = 0;
  int total_tasks_planned = 0;
  int total_tests_replayed = 0;
  int total_ignored = 0;
  int total_deletions = 0;
  int residual_native_count = 0;

  for (int level = 0; level <= max_conditioning_size; ++level) {
    const LayerPlan plan = make_layer_plan(adjacency, p, level);
    const int task_count = static_cast<int>(plan.tasks.size());
    std::vector<double> pvalues(task_count, NA_REAL);

    for (int i = 0; i < task_count; ++i) {
      const LayerCiTask& task = plan.tasks[i];
      std::vector<double> rx;
      std::vector<double> ry;
      if (task.conditioning_set.empty()) {
        rx = numeric_matrix_column(data, task.orientation_x);
        ry = numeric_matrix_column(data, task.orientation_y);
      } else {
        rx = residualize_lm(data, task.orientation_x, task.conditioning_set);
        ry = residualize_lm(data, task.orientation_y, task.conditioning_set);
        residual_native_count += 2;
      }
      pvalues[i] = dcov_exact_pvalue(rx, ry, index, legacy_index);
    }

    std::map<int, bool> edge_done;
    std::vector<int> delete_edges(static_cast<std::size_t>(p) * p, 0);
    int tests_replayed = 0;
    int ignored_after_delete = 0;
    int deletions = 0;

    for (int i = 0; i < task_count; ++i) {
      const LayerCiTask& task = plan.tasks[i];
      const int edge_key = task.edge_x < task.edge_y
        ? task.edge_x * p + task.edge_y
        : task.edge_y * p + task.edge_x;
      const bool ignored = edge_done[edge_key] ||
        adjacency[static_cast<std::size_t>(task.edge_x) * p + task.edge_y] == 0;
      bool deleted = false;
      double pval = NA_REAL;

      if (ignored) {
        ++ignored_after_delete;
      } else {
        ++tests_replayed;
        pval = pvalues[i];
        if (!std::isfinite(pval)) pval = 1.0;
        const std::size_t pmax_idx =
          static_cast<std::size_t>(task.edge_x) * p + task.edge_y;
        const std::size_t pmax_rev =
          static_cast<std::size_t>(task.edge_y) * p + task.edge_x;
        if (pval > pmax[pmax_idx]) {
          pmax[pmax_idx] = pval;
          pmax[pmax_rev] = pval;
        }
        deleted = pval >= alpha;
        if (deleted) {
          ++deletions;
          delete_edges[static_cast<std::size_t>(task.edge_x) * p +
                       task.edge_y] = 1;
          delete_edges[static_cast<std::size_t>(task.edge_y) * p +
                       task.edge_x] = 1;
          sepsets[task.edge_x][task.edge_y] = task.conditioning_set;
          sepsets[task.edge_y][task.edge_x] = task.conditioning_set;
          edge_done[edge_key] = true;
        }
      }

      if (full_trace) {
        ++global_task_id;
        task_global_id.push_back(global_task_id);
        task_level.push_back(level);
        task_index.push_back(i + 1);
        task_edge_x.push_back(task.edge_x + 1);
        task_edge_y.push_back(task.edge_y + 1);
        task_x.push_back(task.orientation_x + 1);
        task_y.push_back(task.orientation_y + 1);
        task_s_key.push_back(replay_s_key(task.conditioning_set));
        task_conditioning_size.push_back(
          static_cast<int>(task.conditioning_set.size()));
        task_p_used.push_back(pval);
        task_deleted.push_back(deleted);
        task_ignored.push_back(ignored);
      }
    }

    for (int i = 0; i < p * p; ++i) {
      if (delete_edges[i] != 0) adjacency[i] = 0;
    }

    total_tasks_planned += task_count;
    total_tests_replayed += tests_replayed;
    total_ignored += ignored_after_delete;
    total_deletions += deletions;
    n_edgetests.push_back(tests_replayed);
    level_level.push_back(level);
    level_tasks_planned.push_back(task_count);
    level_tests_replayed.push_back(tests_replayed);
    level_ignored.push_back(ignored_after_delete);
    level_deletions.push_back(deletions);
  }

  Rcpp::DataFrame task_rows = Rcpp::DataFrame::create(
    Rcpp::Named("canonical_test_order_id") = task_global_id,
    Rcpp::Named("level") = task_level,
    Rcpp::Named("task_index") = task_index,
    Rcpp::Named("edge_x") = task_edge_x,
    Rcpp::Named("edge_y") = task_edge_y,
    Rcpp::Named("x") = task_x,
    Rcpp::Named("y") = task_y,
    Rcpp::Named("S_key") = task_s_key,
    Rcpp::Named("conditioning_size") = task_conditioning_size,
    Rcpp::Named("p_used") = task_p_used,
    Rcpp::Named("native_edge_deleted") = task_deleted,
    Rcpp::Named("native_edge_ignored") = task_ignored,
    Rcpp::Named("stringsAsFactors") = false
  );
  Rcpp::DataFrame level_rows = Rcpp::DataFrame::create(
    Rcpp::Named("level") = level_level,
    Rcpp::Named("tasks_planned") = level_tasks_planned,
    Rcpp::Named("tests_replayed") = level_tests_replayed,
    Rcpp::Named("tasks_ignored_after_delete") = level_ignored,
    Rcpp::Named("deletions") = level_deletions,
    Rcpp::Named("stringsAsFactors") = false
  );

  return Rcpp::List::create(
    Rcpp::Named("adjacency") = adjacency_to_matrix(adjacency, p),
    Rcpp::Named("sepsets") = sepsets_to_list(sepsets),
    Rcpp::Named("pMax") = pmax_to_matrix(pmax, p),
    Rcpp::Named("n.edgetests") = n_edgetests,
    Rcpp::Named("tasks") = task_rows,
    Rcpp::Named("levels") = level_rows,
    Rcpp::Named("summary") = Rcpp::List::create(
      Rcpp::Named("p") = p,
      Rcpp::Named("n") = n,
      Rcpp::Named("alpha") = alpha,
      Rcpp::Named("max_conditioning_size") = max_conditioning_size,
      Rcpp::Named("levels") = max_conditioning_size + 1,
      Rcpp::Named("tasks_planned") = total_tasks_planned,
      Rcpp::Named("tests_replayed") = total_tests_replayed,
      Rcpp::Named("tasks_ignored_after_delete") = total_ignored,
      Rcpp::Named("deletions") = total_deletions,
      Rcpp::Named("ci_native_count") = total_tasks_planned,
      Rcpp::Named("residual_native_count") = residual_native_count,
      Rcpp::Named("ci_backend") = "native-exact-dcov",
      Rcpp::Named("residual_backend") = "native-linear-lm"
    )
  );
  END_RCPP
}

extern "C" SEXP C_precision_run_skeleton_residual_provider_native(
    SEXP datas,
    SEXP alphas,
    SEXP max_conditioning_sizes,
    SEXP residual_providers,
    SEXP indexs,
    SEXP legacy_indexs,
    SEXP trace_levels) {
  BEGIN_RCPP
  Rcpp::NumericMatrix data(datas);
  const int n = data.nrow();
  const int p = data.ncol();
  const double alpha = Rf_asReal(alphas);
  const int max_conditioning_size = Rf_asInteger(max_conditioning_sizes);
  Rcpp::Function residual_provider(residual_providers);
  const double index = Rf_asReal(indexs);
  const bool legacy_index = Rcpp::as<bool>(legacy_indexs);
  const std::string trace_level = Rcpp::as<std::string>(trace_levels);
  const bool full_trace = trace_level == "full";
  if (n <= 5) {
    Rcpp::stop("native residual-provider skeleton requires n > 5");
  }
  if (p < 2) {
    Rcpp::stop("native residual-provider skeleton requires at least two columns");
  }
  if (!std::isfinite(alpha) || alpha <= 0.0) {
    Rcpp::stop("alpha must be a positive finite value");
  }
  if (max_conditioning_size < 0) {
    Rcpp::stop("max_conditioning_size must be non-negative");
  }
  if (!all_finite(data)) {
    Rcpp::stop("Data contains missing or infinite values");
  }

  std::vector<int> adjacency(static_cast<std::size_t>(p) * p, 1);
  std::vector<double> pmax(static_cast<std::size_t>(p) * p,
                           -std::numeric_limits<double>::infinity());
  std::vector<std::vector<std::vector<int> > > sepsets(
    p, std::vector<std::vector<int> >(p));
  for (int i = 0; i < p; ++i) {
    adjacency[static_cast<std::size_t>(i) * p + i] = 0;
    pmax[static_cast<std::size_t>(i) * p + i] = 1.0;
  }

  Rcpp::IntegerVector n_edgetests;
  Rcpp::IntegerVector level_level, level_tasks_planned, level_tests_replayed,
    level_ignored, level_deletions;
  Rcpp::IntegerVector task_global_id, task_level, task_index, task_edge_x,
    task_edge_y, task_x, task_y, task_conditioning_size;
  Rcpp::CharacterVector task_s_key;
  Rcpp::NumericVector task_p_used;
  Rcpp::LogicalVector task_deleted, task_ignored;

  int global_task_id = 0;
  int total_tasks_planned = 0;
  int total_tests_replayed = 0;
  int total_ignored = 0;
  int total_deletions = 0;
  int residual_provider_level_count = 0;
  int residual_provider_request_count = 0;

  for (int level = 0; level <= max_conditioning_size; ++level) {
    const LayerPlan plan = make_layer_plan(adjacency, p, level);
    const int task_count = static_cast<int>(plan.tasks.size());

    std::vector<int> request_targets;
    std::vector<std::vector<int> > request_conditioning_sets;
    std::map<std::string, int> request_by_key;
    for (const LayerCiTask& task : plan.tasks) {
      if (task.conditioning_set.empty()) continue;
      const int targets[2] = {task.orientation_x, task.orientation_y};
      for (int k = 0; k < 2; ++k) {
        const std::string key =
          native_residual_key(targets[k], task.conditioning_set);
        if (request_by_key.find(key) != request_by_key.end()) continue;
        request_by_key[key] = static_cast<int>(request_targets.size());
        request_targets.push_back(targets[k]);
        request_conditioning_sets.push_back(task.conditioning_set);
      }
    }

    std::vector<std::vector<double> > residual_columns(request_targets.size());
    if (!request_targets.empty()) {
      const int request_count = static_cast<int>(request_targets.size());
      Rcpp::IntegerVector request_index(request_count), request_target(request_count),
        request_conditioning_size(request_count);
      Rcpp::CharacterVector request_s_key(request_count);
      Rcpp::List request_conditioning_list(request_count);
      for (int i = 0; i < request_count; ++i) {
        Rcpp::IntegerVector cond(request_conditioning_sets[i].size());
        for (int j = 0; j < cond.size(); ++j) {
          cond[j] = request_conditioning_sets[i][j] + 1;
        }
        request_index[i] = i + 1;
        request_target[i] = request_targets[i] + 1;
        request_conditioning_list[i] = cond;
        request_s_key[i] = replay_s_key(request_conditioning_sets[i]);
        request_conditioning_size[i] =
          static_cast<int>(request_conditioning_sets[i].size());
      }
      request_conditioning_list.attr("class") = "AsIs";
      Rcpp::DataFrame request_table = Rcpp::DataFrame::create(
        Rcpp::Named("request_index") = request_index,
        Rcpp::Named("target") = request_target,
        Rcpp::Named("conditioning_sets") = request_conditioning_list,
        Rcpp::Named("S_key") = request_s_key,
        Rcpp::Named("conditioning_size") = request_conditioning_size,
        Rcpp::Named("stringsAsFactors") = false
      );
      Rcpp::NumericMatrix residual_matrix =
        residual_provider(request_table, level);
      if (residual_matrix.nrow() != n || residual_matrix.ncol() != request_count) {
        Rcpp::stop("residual provider returned matrix with wrong dimensions");
      }
      for (int col = 0; col < request_count; ++col) {
        residual_columns[col].resize(n);
        for (int row = 0; row < n; ++row) {
          residual_columns[col][row] = residual_matrix(row, col);
        }
      }
      ++residual_provider_level_count;
      residual_provider_request_count += request_count;
    }

    std::vector<double> pvalues(task_count, NA_REAL);
    for (int i = 0; i < task_count; ++i) {
      const LayerCiTask& task = plan.tasks[i];
      std::vector<double> rx;
      std::vector<double> ry;
      if (task.conditioning_set.empty()) {
        rx = numeric_matrix_column(data, task.orientation_x);
        ry = numeric_matrix_column(data, task.orientation_y);
      } else {
        const std::string x_key =
          native_residual_key(task.orientation_x, task.conditioning_set);
        const std::string y_key =
          native_residual_key(task.orientation_y, task.conditioning_set);
        rx = residual_columns[request_by_key[x_key]];
        ry = residual_columns[request_by_key[y_key]];
      }
      pvalues[i] = dcov_exact_pvalue(rx, ry, index, legacy_index);
    }

    std::map<int, bool> edge_done;
    std::vector<int> delete_edges(static_cast<std::size_t>(p) * p, 0);
    int tests_replayed = 0;
    int ignored_after_delete = 0;
    int deletions = 0;

    for (int i = 0; i < task_count; ++i) {
      const LayerCiTask& task = plan.tasks[i];
      const int edge_key = task.edge_x < task.edge_y
        ? task.edge_x * p + task.edge_y
        : task.edge_y * p + task.edge_x;
      const bool ignored = edge_done[edge_key] ||
        adjacency[static_cast<std::size_t>(task.edge_x) * p + task.edge_y] == 0;
      bool deleted = false;
      double pval = NA_REAL;

      if (ignored) {
        ++ignored_after_delete;
      } else {
        ++tests_replayed;
        pval = pvalues[i];
        if (!std::isfinite(pval)) pval = 1.0;
        const std::size_t pmax_idx =
          static_cast<std::size_t>(task.edge_x) * p + task.edge_y;
        const std::size_t pmax_rev =
          static_cast<std::size_t>(task.edge_y) * p + task.edge_x;
        if (pval > pmax[pmax_idx]) {
          pmax[pmax_idx] = pval;
          pmax[pmax_rev] = pval;
        }
        deleted = pval >= alpha;
        if (deleted) {
          ++deletions;
          delete_edges[static_cast<std::size_t>(task.edge_x) * p +
                       task.edge_y] = 1;
          delete_edges[static_cast<std::size_t>(task.edge_y) * p +
                       task.edge_x] = 1;
          sepsets[task.edge_x][task.edge_y] = task.conditioning_set;
          sepsets[task.edge_y][task.edge_x] = task.conditioning_set;
          edge_done[edge_key] = true;
        }
      }

      if (full_trace) {
        ++global_task_id;
        task_global_id.push_back(global_task_id);
        task_level.push_back(level);
        task_index.push_back(i + 1);
        task_edge_x.push_back(task.edge_x + 1);
        task_edge_y.push_back(task.edge_y + 1);
        task_x.push_back(task.orientation_x + 1);
        task_y.push_back(task.orientation_y + 1);
        task_s_key.push_back(replay_s_key(task.conditioning_set));
        task_conditioning_size.push_back(
          static_cast<int>(task.conditioning_set.size()));
        task_p_used.push_back(pval);
        task_deleted.push_back(deleted);
        task_ignored.push_back(ignored);
      }
    }

    for (int i = 0; i < p * p; ++i) {
      if (delete_edges[i] != 0) adjacency[i] = 0;
    }

    total_tasks_planned += task_count;
    total_tests_replayed += tests_replayed;
    total_ignored += ignored_after_delete;
    total_deletions += deletions;
    n_edgetests.push_back(tests_replayed);
    level_level.push_back(level);
    level_tasks_planned.push_back(task_count);
    level_tests_replayed.push_back(tests_replayed);
    level_ignored.push_back(ignored_after_delete);
    level_deletions.push_back(deletions);
  }

  Rcpp::DataFrame task_rows = Rcpp::DataFrame::create(
    Rcpp::Named("canonical_test_order_id") = task_global_id,
    Rcpp::Named("level") = task_level,
    Rcpp::Named("task_index") = task_index,
    Rcpp::Named("edge_x") = task_edge_x,
    Rcpp::Named("edge_y") = task_edge_y,
    Rcpp::Named("x") = task_x,
    Rcpp::Named("y") = task_y,
    Rcpp::Named("S_key") = task_s_key,
    Rcpp::Named("conditioning_size") = task_conditioning_size,
    Rcpp::Named("p_used") = task_p_used,
    Rcpp::Named("native_edge_deleted") = task_deleted,
    Rcpp::Named("native_edge_ignored") = task_ignored,
    Rcpp::Named("stringsAsFactors") = false
  );
  Rcpp::DataFrame level_rows = Rcpp::DataFrame::create(
    Rcpp::Named("level") = level_level,
    Rcpp::Named("tasks_planned") = level_tasks_planned,
    Rcpp::Named("tests_replayed") = level_tests_replayed,
    Rcpp::Named("tasks_ignored_after_delete") = level_ignored,
    Rcpp::Named("deletions") = level_deletions,
    Rcpp::Named("stringsAsFactors") = false
  );

  return Rcpp::List::create(
    Rcpp::Named("adjacency") = adjacency_to_matrix(adjacency, p),
    Rcpp::Named("sepsets") = sepsets_to_list(sepsets),
    Rcpp::Named("pMax") = pmax_to_matrix(pmax, p),
    Rcpp::Named("n.edgetests") = n_edgetests,
    Rcpp::Named("tasks") = task_rows,
    Rcpp::Named("levels") = level_rows,
    Rcpp::Named("summary") = Rcpp::List::create(
      Rcpp::Named("p") = p,
      Rcpp::Named("n") = n,
      Rcpp::Named("alpha") = alpha,
      Rcpp::Named("max_conditioning_size") = max_conditioning_size,
      Rcpp::Named("levels") = max_conditioning_size + 1,
      Rcpp::Named("tasks_planned") = total_tasks_planned,
      Rcpp::Named("tests_replayed") = total_tests_replayed,
      Rcpp::Named("tasks_ignored_after_delete") = total_ignored,
      Rcpp::Named("deletions") = total_deletions,
      Rcpp::Named("ci_native_count") = total_tasks_planned,
      Rcpp::Named("residual_provider_level_count") =
        residual_provider_level_count,
      Rcpp::Named("residual_provider_request_count") =
        residual_provider_request_count,
      Rcpp::Named("ci_backend") = "native-exact-dcov",
      Rcpp::Named("residual_backend") = "provider-legacy-mgcv"
    )
  );
  END_RCPP
}

extern "C" SEXP C_precision_run_skeleton_residual_provider_legacy_dcov_native(
    SEXP datas,
    SEXP alphas,
    SEXP max_conditioning_sizes,
    SEXP residual_providers,
    SEXP indexs,
    SEXP num_cols,
    SEXP trace_levels) {
  BEGIN_RCPP
  Rcpp::NumericMatrix data(datas);
  const int n = data.nrow();
  const int p = data.ncol();
  const double alpha = Rf_asReal(alphas);
  const int max_conditioning_size = Rf_asInteger(max_conditioning_sizes);
  Rcpp::Function residual_provider(residual_providers);
  const double index = Rf_asReal(indexs);
  const int num_col = Rf_asInteger(num_cols);
  const std::string trace_level = Rcpp::as<std::string>(trace_levels);
  const bool full_trace = trace_level == "full";
  const bool logical_trace = trace_level == "logical";
  const bool collect_trace = full_trace || logical_trace;
  if (n <= 5) {
    Rcpp::stop("native residual-provider legacy dCov skeleton requires n > 5");
  }
  if (p < 2) {
    Rcpp::stop("native residual-provider legacy dCov skeleton requires at least two columns");
  }
  if (!std::isfinite(alpha) || alpha <= 0.0) {
    Rcpp::stop("alpha must be a positive finite value");
  }
  if (max_conditioning_size < 0) {
    Rcpp::stop("max_conditioning_size must be non-negative");
  }
  if (num_col <= 0 || num_col >= n) {
    Rcpp::stop("numCol must be positive and less than sample size");
  }
  if (!all_finite(data)) {
    Rcpp::stop("Data contains missing or infinite values");
  }

  std::vector<int> adjacency(static_cast<std::size_t>(p) * p, 1);
  std::vector<double> pmax(static_cast<std::size_t>(p) * p,
                           -std::numeric_limits<double>::infinity());
  std::vector<std::vector<std::vector<int> > > sepsets(
    p, std::vector<std::vector<int> >(p));
  for (int i = 0; i < p; ++i) {
    adjacency[static_cast<std::size_t>(i) * p + i] = 0;
    pmax[static_cast<std::size_t>(i) * p + i] = 1.0;
  }

  Rcpp::IntegerVector n_edgetests;
  Rcpp::IntegerVector level_level, level_tasks_planned, level_tests_replayed,
    level_ignored, level_deletions;
  Rcpp::IntegerVector level_residual_provider_request_count;
  Rcpp::NumericVector level_residual_provider_call_ms,
    level_residual_provider_matrix_copy_ms, level_residual_provider_total_ms,
    level_legacy_dcov_native_materialize_ms,
    level_legacy_dcov_native_call_ms;
  std::vector<int> task_global_id, task_level, task_index, task_edge_x,
    task_edge_y, task_x, task_y, task_conditioning_size;
  std::vector<std::string> task_s_key;
  std::vector<double> task_p_used;
  std::vector<int> task_deleted, task_ignored;

  int global_task_id = 0;
  int total_tasks_planned = 0;
  int total_tests_replayed = 0;
  int total_ignored = 0;
  int total_deletions = 0;
  int residual_provider_level_count = 0;
  int residual_provider_request_count = 0;
  int residual_provider_batch_count = 0;
  int residual_provider_batch_max_requests = 0;
  double residual_provider_batch_request_sum = 0.0;
  int residual_provider_matrix_cell_count = 0;
  double residual_provider_call_ms = 0.0;
  double residual_provider_matrix_copy_ms = 0.0;
  std::string residual_provider_response_mode;
  std::string residual_provider_response_backend;
  int legacy_dcov_native_count = 0;
  double legacy_dcov_native_ms = 0.0;
  double legacy_dcov_native_scalar_materialize_ms = 0.0;
  double legacy_dcov_native_scalar_call_ms = 0.0;
  fastkpc::LegacyDcovLowrankTimings legacy_lowrank_timings;
  fastkpc::LegacyDcovLowrankMode legacy_lowrank_mode =
    fastkpc::legacy_dcov_lowrank_mode_from_env();
  const NativeLegacyDcovBatchMode legacy_dcov_native_batch_mode =
    native_legacy_dcov_batch_mode_from_env();
  const bool legacy_dcov_native_batch_enabled =
    legacy_dcov_native_batch_mode != NativeLegacyDcovBatchMode::None;
  int legacy_dcov_native_batch_count = 0;
  int legacy_dcov_native_batch_pair_count = 0;
  bool legacy_dcov_native_batch_parallel_enabled = false;
  int legacy_dcov_native_batch_parallel_threads = 1;
  int legacy_dcov_native_batch_workspace_reuse_count = 0;
  int legacy_dcov_native_batch_distance_workspace_reuse_count = 0;
  int legacy_dcov_native_batch_statistic_moment_workspace_reuse_count = 0;
  int legacy_dcov_native_batch_lowrank_output_workspace_reuse_count = 0;
  int legacy_dcov_native_batch_lowrank_eig_workspace_reuse_count = 0;
  int legacy_dcov_native_batch_oracle_column_copy_count = 0;
  int legacy_dcov_native_batch_column_materialize_count = 0;
  bool legacy_dcov_native_batch_direct_input_enabled = false;
  double legacy_dcov_native_batch_ms = 0.0;
  double legacy_dcov_native_batch_materialize_ms = 0.0;
  double legacy_dcov_native_batch_call_ms = 0.0;
  double legacy_dcov_native_batch_input_ms = 0.0;
  double legacy_dcov_native_batch_distance_ms = 0.0;
  double legacy_dcov_native_batch_lowrank_ms = 0.0;
  double legacy_dcov_native_batch_lowrank_eig_ms = 0.0;
  double legacy_dcov_native_batch_lowrank_select_ms = 0.0;
  double legacy_dcov_native_batch_lowrank_center_ms = 0.0;
  double legacy_dcov_native_batch_lowrank_unaccounted_ms = 0.0;
  double legacy_dcov_native_batch_statistic_ms = 0.0;
  double legacy_dcov_native_batch_moment_ms = 0.0;
  double legacy_dcov_native_batch_pgamma_ms = 0.0;
  double legacy_dcov_native_batch_accounted_ms = 0.0;
  double legacy_dcov_native_batch_scalar_total_ms = 0.0;
  double legacy_dcov_native_batch_wrapper_overhead_ms = 0.0;
  double legacy_dcov_native_batch_overhead_ms = 0.0;
  NativeLegacyDcovCudaLowrankBackendMetrics
    legacy_dcov_native_cuda_lowrank_metrics;
  legacy_dcov_native_cuda_lowrank_metrics.enabled =
    native_legacy_dcov_cuda_lowrank_requested();
  legacy_dcov_native_cuda_lowrank_metrics.component_cache_enabled =
    legacy_dcov_native_cuda_lowrank_metrics.enabled &&
    native_legacy_cuda_lowrank_component_cache_enabled();
  if (legacy_dcov_native_cuda_lowrank_metrics.component_cache_enabled) {
    legacy_dcov_native_cuda_lowrank_metrics.component_cache_scope =
      native_legacy_cuda_lowrank_component_cache_scope();
    if (legacy_dcov_native_cuda_lowrank_metrics.component_cache_scope ==
        "level") {
      legacy_dcov_native_cuda_lowrank_metrics
        .component_cache_level_max_entries =
        native_legacy_cuda_lowrank_component_cache_max_entries();
    }
  }
  const int legacy_dcov_native_cuda_lowrank_ncv =
    native_legacy_dcov_cuda_lowrank_ncv(n, num_col);
  const double legacy_dcov_native_cuda_lowrank_tol = 1e-10;
  const int legacy_dcov_native_cuda_lowrank_maxitr = 1000;
  const int legacy_dcov_native_cuda_lowrank_progress_interval =
    native_legacy_cuda_lowrank_progress_interval();
  const std::string native_progress_csv_path =
    native_legacy_progress_csv_path_from_env();
  const std::string native_cuda_lowrank_component_cache_progress_csv_path =
    native_cuda_lowrank_component_cache_progress_csv_path_from_env();

  auto accumulate_native_cuda_lowrank_run =
    [&](const LegacyDcovCudaLowrankGammaRun& run) {
      legacy_dcov_native_cuda_lowrank_metrics.count += 1;
      legacy_dcov_native_cuda_lowrank_metrics.ms += run.total_ms;
      if (run.converged_x && run.converged_y) {
        legacy_dcov_native_cuda_lowrank_metrics.converged_count += 1;
      }
      legacy_dcov_native_cuda_lowrank_metrics.spectra_matvec_count +=
        run.spectra_matvec_count;
      legacy_dcov_native_cuda_lowrank_metrics.spectra_matvec_ms +=
        run.spectra_matvec_ms;
      legacy_dcov_native_cuda_lowrank_metrics.kernel_launch_count +=
        run.kernel_launch_count;
      legacy_dcov_native_cuda_lowrank_metrics.device_matrix_reuse_count +=
        run.device_matrix_reuse_count;
      legacy_dcov_native_cuda_lowrank_metrics.device_workspace_reuse_count +=
        run.device_workspace_reuse_count;
      legacy_dcov_native_cuda_lowrank_metrics.workspace_realloc_count +=
        run.workspace_realloc_count;
      legacy_dcov_native_cuda_lowrank_metrics.matrix_bytes +=
        run.matrix_bytes;
      legacy_dcov_native_cuda_lowrank_metrics.workspace_bytes = std::max(
        legacy_dcov_native_cuda_lowrank_metrics.workspace_bytes,
        run.workspace_bytes);
      legacy_dcov_native_cuda_lowrank_metrics.matrix_h2d_ms +=
        run.matrix_h2d_ms;
      legacy_dcov_native_cuda_lowrank_metrics.matrix_h2d_ms_during_compute +=
        run.matrix_h2d_ms_during_compute;
      const double previous_matrix_h2d_ms_during_compute_max =
        legacy_dcov_native_cuda_lowrank_metrics.matrix_h2d_ms_during_compute_max;
      legacy_dcov_native_cuda_lowrank_metrics.matrix_h2d_ms_during_compute_max =
        std::max(
          previous_matrix_h2d_ms_during_compute_max,
          run.matrix_h2d_ms_during_compute);
      legacy_dcov_native_cuda_lowrank_metrics.workspace_alloc_ms +=
        run.workspace_alloc_ms;
      legacy_dcov_native_cuda_lowrank_metrics.h2d_ms += run.h2d_ms;
      legacy_dcov_native_cuda_lowrank_metrics.kernel_ms += run.kernel_ms;
      legacy_dcov_native_cuda_lowrank_metrics.d2h_ms += run.d2h_ms;

      legacy_lowrank_timings.eig_ms += run.eig_ms;
      legacy_lowrank_timings.spectra_count += 2;
      if (run.converged_x) legacy_lowrank_timings.spectra_converged_count += 1;
      if (run.converged_y) legacy_lowrank_timings.spectra_converged_count += 1;
      if (!run.converged_x) legacy_lowrank_timings.spectra_failed_count += 1;
      if (!run.converged_y) legacy_lowrank_timings.spectra_failed_count += 1;
      legacy_lowrank_timings.spectra_iterations +=
        run.iterations_x + run.iterations_y;
      legacy_lowrank_timings.spectra_nconv += run.nconv_x + run.nconv_y;
      legacy_lowrank_timings.spectra_ncv = std::max(
        legacy_lowrank_timings.spectra_ncv,
        legacy_dcov_native_cuda_lowrank_ncv);
      legacy_lowrank_timings.spectra_tol = std::max(
        legacy_lowrank_timings.spectra_tol,
        legacy_dcov_native_cuda_lowrank_tol);
      legacy_lowrank_timings.spectra_matvec_count +=
        run.spectra_matvec_count;
      legacy_lowrank_timings.spectra_matvec_ms += run.spectra_matvec_ms;
      legacy_lowrank_mode = fastkpc::LegacyDcovLowrankMode::Spectra;
    };

  auto accumulate_native_cuda_lowrank_component =
    [&](const LegacyDcovCudaLowrankComponentRun& component) {
      legacy_dcov_native_cuda_lowrank_metrics.spectra_matvec_count +=
        component.cuda_diagnostics.spectra_matvec_count;
      legacy_dcov_native_cuda_lowrank_metrics.spectra_matvec_ms +=
        component.cuda_diagnostics.spectra_matvec_ms;
      legacy_dcov_native_cuda_lowrank_metrics.kernel_launch_count +=
        component.cuda_diagnostics.kernel_launch_count;
      legacy_dcov_native_cuda_lowrank_metrics.device_matrix_reuse_count +=
        component.cuda_diagnostics.device_matrix_reuse_count;
      legacy_dcov_native_cuda_lowrank_metrics.device_workspace_reuse_count +=
        component.cuda_diagnostics.device_workspace_reuse_count;
      legacy_dcov_native_cuda_lowrank_metrics.workspace_realloc_count +=
        component.cuda_diagnostics.workspace_realloc_count;
      legacy_dcov_native_cuda_lowrank_metrics.matrix_bytes +=
        component.matrix_bytes;
      legacy_dcov_native_cuda_lowrank_metrics.workspace_bytes = std::max(
        legacy_dcov_native_cuda_lowrank_metrics.workspace_bytes,
        static_cast<double>(component.cuda_diagnostics.workspace_bytes));
      legacy_dcov_native_cuda_lowrank_metrics.matrix_h2d_ms +=
        component.matrix_h2d_ms;
      legacy_dcov_native_cuda_lowrank_metrics.matrix_h2d_ms_during_compute +=
        component.cuda_diagnostics.matrix_h2d_ms_during_compute;
      const double previous_matrix_h2d_ms_during_compute_max =
        legacy_dcov_native_cuda_lowrank_metrics.matrix_h2d_ms_during_compute_max;
      legacy_dcov_native_cuda_lowrank_metrics.matrix_h2d_ms_during_compute_max =
        std::max(
          previous_matrix_h2d_ms_during_compute_max,
          component.cuda_diagnostics.matrix_h2d_ms_during_compute);
      legacy_dcov_native_cuda_lowrank_metrics.workspace_alloc_ms +=
        component.cuda_diagnostics.workspace_alloc_ms;
      legacy_dcov_native_cuda_lowrank_metrics.h2d_ms +=
        component.cuda_diagnostics.h2d_ms;
      legacy_dcov_native_cuda_lowrank_metrics.kernel_ms +=
        component.cuda_diagnostics.kernel_ms;
      legacy_dcov_native_cuda_lowrank_metrics.d2h_ms +=
        component.cuda_diagnostics.d2h_ms;

      legacy_lowrank_timings.eig_ms += component.eig_ms;
      legacy_lowrank_timings.spectra_count += 1;
      if (component.converged) {
        legacy_lowrank_timings.spectra_converged_count += 1;
      } else {
        legacy_lowrank_timings.spectra_failed_count += 1;
      }
      legacy_lowrank_timings.spectra_iterations += component.iterations;
      legacy_lowrank_timings.spectra_nconv += component.nconv;
      legacy_lowrank_timings.spectra_ncv = std::max(
        legacy_lowrank_timings.spectra_ncv,
        legacy_dcov_native_cuda_lowrank_ncv);
      legacy_lowrank_timings.spectra_tol = std::max(
        legacy_lowrank_timings.spectra_tol,
        legacy_dcov_native_cuda_lowrank_tol);
      legacy_lowrank_timings.spectra_matvec_count +=
        component.cuda_diagnostics.spectra_matvec_count;
      legacy_lowrank_timings.spectra_matvec_ms +=
        component.cuda_diagnostics.spectra_matvec_ms;
      legacy_lowrank_mode = fastkpc::LegacyDcovLowrankMode::Spectra;
    };

  for (int level = 0; level <= max_conditioning_size; ++level) {
    const std::chrono::steady_clock::time_point level_start =
      std::chrono::steady_clock::now();
    const LayerPlan plan = make_layer_plan(adjacency, p, level);
    const int task_count = static_cast<int>(plan.tasks.size());
    if (collect_trace && task_count > 0) {
      const std::size_t capacity = task_global_id.size() +
        static_cast<std::size_t>(task_count);
      task_global_id.reserve(capacity);
      task_level.reserve(capacity);
      task_index.reserve(capacity);
      task_edge_x.reserve(capacity);
      task_edge_y.reserve(capacity);
      task_x.reserve(capacity);
      task_y.reserve(capacity);
      task_s_key.reserve(capacity);
      task_conditioning_size.reserve(capacity);
      task_p_used.reserve(capacity);
      task_deleted.reserve(capacity);
      task_ignored.reserve(capacity);
    }
    int level_provider_request_count = 0;
    double level_provider_call_ms = 0.0;
    double level_provider_matrix_copy_ms = 0.0;
    double level_dcov_materialize_ms = 0.0;
    double level_dcov_call_ms = 0.0;

    std::vector<int> request_targets;
    std::vector<std::vector<int> > request_conditioning_sets;
    std::map<std::string, int> request_by_key;
    for (const LayerCiTask& task : plan.tasks) {
      if (task.conditioning_set.empty()) continue;
      const int targets[2] = {task.orientation_x, task.orientation_y};
      for (int k = 0; k < 2; ++k) {
        const std::string key =
          native_residual_key(targets[k], task.conditioning_set);
        if (request_by_key.find(key) != request_by_key.end()) continue;
        request_by_key[key] = static_cast<int>(request_targets.size());
        request_targets.push_back(targets[k]);
        request_conditioning_sets.push_back(task.conditioning_set);
      }
    }
    append_native_legacy_progress(
      native_progress_csv_path,
      "level_start",
      level,
      task_count,
      static_cast<int>(request_targets.size()),
      0,
      0,
      0,
      0.0,
      0.0,
      0.0,
      0.0,
      elapsed_ms_since(level_start));

    std::vector<std::vector<double> > residual_columns(request_targets.size());
    if (!request_targets.empty()) {
      const int request_count = static_cast<int>(request_targets.size());
      Rcpp::IntegerVector request_index(request_count), request_target(request_count),
        request_conditioning_size(request_count);
      Rcpp::CharacterVector request_s_key(request_count);
      Rcpp::List request_conditioning_list(request_count);
      for (int i = 0; i < request_count; ++i) {
        Rcpp::IntegerVector cond(request_conditioning_sets[i].size());
        for (int j = 0; j < cond.size(); ++j) {
          cond[j] = request_conditioning_sets[i][j] + 1;
        }
        request_index[i] = i + 1;
        request_target[i] = request_targets[i] + 1;
        request_conditioning_list[i] = cond;
        request_s_key[i] = replay_s_key(request_conditioning_sets[i]);
        request_conditioning_size[i] =
          static_cast<int>(request_conditioning_sets[i].size());
      }
      request_conditioning_list.attr("class") = "AsIs";
      Rcpp::DataFrame request_table = Rcpp::DataFrame::create(
        Rcpp::Named("request_index") = request_index,
        Rcpp::Named("target") = request_target,
        Rcpp::Named("conditioning_sets") = request_conditioning_list,
        Rcpp::Named("S_key") = request_s_key,
        Rcpp::Named("conditioning_size") = request_conditioning_size,
        Rcpp::Named("stringsAsFactors") = false
      );
      std::string provider_response_mode;
      std::string provider_response_backend;
      const std::chrono::steady_clock::time_point provider_call_start =
        std::chrono::steady_clock::now();
      SEXP provider_response = residual_provider(request_table, level);
      level_provider_call_ms = elapsed_ms_since(provider_call_start);
      residual_provider_call_ms += level_provider_call_ms;

      const std::chrono::steady_clock::time_point provider_copy_start =
        std::chrono::steady_clock::now();
      Rcpp::NumericMatrix residual_matrix =
        residual_provider_response_matrix(
          provider_response,
          &provider_response_mode,
          &provider_response_backend);
      if (residual_matrix.nrow() != n || residual_matrix.ncol() != request_count) {
        Rcpp::stop("residual provider returned matrix with wrong dimensions");
      }
      for (int col = 0; col < request_count; ++col) {
        residual_columns[col].resize(n);
        for (int row = 0; row < n; ++row) {
          residual_columns[col][row] = residual_matrix(row, col);
        }
      }
      level_provider_matrix_copy_ms = elapsed_ms_since(provider_copy_start);
      residual_provider_matrix_copy_ms += level_provider_matrix_copy_ms;
      ++residual_provider_level_count;
      residual_provider_request_count += request_count;
      level_provider_request_count = request_count;
      ++residual_provider_batch_count;
      residual_provider_batch_max_requests = std::max(
        residual_provider_batch_max_requests,
        request_count);
      residual_provider_batch_request_sum +=
        static_cast<double>(request_count);
      residual_provider_matrix_cell_count += n * request_count;
      update_provider_response_label(&residual_provider_response_mode,
                                     provider_response_mode);
      update_provider_response_label(&residual_provider_response_backend,
                                     provider_response_backend);
    }
    append_native_legacy_progress(
      native_progress_csv_path,
      "provider_complete",
      level,
      task_count,
      static_cast<int>(request_targets.size()),
      0,
      0,
      0,
      level_provider_call_ms,
      level_provider_matrix_copy_ms,
      0.0,
      0.0,
      elapsed_ms_since(level_start));

    std::map<int, bool> edge_done;
    std::vector<int> delete_edges(static_cast<std::size_t>(p) * p, 0);
    int tests_replayed = 0;
    int ignored_after_delete = 0;
    int deletions = 0;

    auto task_input_pointers = [&](const LayerCiTask& task) {
      const double* x_ptr = nullptr;
      const double* y_ptr = nullptr;
      if (task.conditioning_set.empty()) {
        x_ptr = data.begin() +
          static_cast<std::ptrdiff_t>(task.orientation_x) * n;
        y_ptr = data.begin() +
          static_cast<std::ptrdiff_t>(task.orientation_y) * n;
      } else {
        const std::string x_key =
          native_residual_key(task.orientation_x, task.conditioning_set);
        const std::string y_key =
          native_residual_key(task.orientation_y, task.conditioning_set);
        x_ptr = residual_columns[request_by_key[x_key]].data();
        y_ptr = residual_columns[request_by_key[y_key]].data();
      }
      return std::make_pair(x_ptr, y_ptr);
    };

    const bool cuda_lowrank_level_component_cache_enabled =
      legacy_dcov_native_cuda_lowrank_metrics.component_cache_enabled &&
      legacy_dcov_native_cuda_lowrank_metrics.component_cache_scope == "level";
    std::unique_ptr<NativeLegacyDcovCudaLowrankLevelComponentCache>
      cuda_lowrank_level_component_cache;
    if (cuda_lowrank_level_component_cache_enabled) {
      cuda_lowrank_level_component_cache.reset(
        new NativeLegacyDcovCudaLowrankLevelComponentCache(
          legacy_dcov_native_cuda_lowrank_metrics
            .component_cache_level_max_entries));
    }

    auto accumulate_batch_diag = [&](const Rcpp::List& batch_diag,
                                     int pair_count,
                                     double materialize_ms,
                                     double call_ms) {
      legacy_dcov_native_count += pair_count;
      legacy_dcov_native_ms +=
        list_numeric_value(batch_diag, "scalar_total_ms");
      legacy_dcov_native_batch_count += 1;
      legacy_dcov_native_batch_pair_count += pair_count;
      legacy_dcov_native_batch_ms +=
        list_numeric_value(batch_diag, "total_ms");
      legacy_dcov_native_batch_input_ms +=
        list_numeric_value(batch_diag, "input_ms");
      legacy_dcov_native_batch_distance_ms +=
        list_numeric_value(batch_diag, "distance_ms");
      legacy_dcov_native_batch_lowrank_ms +=
        list_numeric_value(batch_diag, "lowrank_ms");
      legacy_dcov_native_batch_lowrank_eig_ms +=
        list_numeric_value(batch_diag, "lowrank_eig_ms");
      legacy_dcov_native_batch_lowrank_select_ms +=
        list_numeric_value(batch_diag, "lowrank_select_ms");
      legacy_dcov_native_batch_lowrank_center_ms +=
        list_numeric_value(batch_diag, "lowrank_center_ms");
      legacy_dcov_native_batch_lowrank_unaccounted_ms +=
        list_numeric_value(batch_diag, "lowrank_unaccounted_ms");
      legacy_dcov_native_batch_statistic_ms +=
        list_numeric_value(batch_diag, "statistic_ms");
      legacy_dcov_native_batch_moment_ms +=
        list_numeric_value(batch_diag, "moment_ms");
      legacy_dcov_native_batch_pgamma_ms +=
        list_numeric_value(batch_diag, "pgamma_ms");
      legacy_dcov_native_batch_accounted_ms +=
        list_numeric_value(batch_diag, "accounted_ms");
      legacy_dcov_native_batch_scalar_total_ms +=
        list_numeric_value(batch_diag, "scalar_total_ms");
      legacy_dcov_native_batch_wrapper_overhead_ms +=
        list_numeric_value(batch_diag, "wrapper_overhead_ms");
      legacy_dcov_native_batch_overhead_ms +=
        list_numeric_value(batch_diag, "batch_overhead_ms");
      legacy_dcov_native_batch_materialize_ms += materialize_ms;
      legacy_dcov_native_batch_call_ms += call_ms;
      legacy_dcov_native_batch_parallel_enabled =
        legacy_dcov_native_batch_parallel_enabled ||
        list_logical_value(batch_diag, "batch_parallel_enabled");
      legacy_dcov_native_batch_parallel_threads = std::max(
        legacy_dcov_native_batch_parallel_threads,
        list_integer_value(batch_diag, "batch_parallel_threads"));
      legacy_dcov_native_batch_workspace_reuse_count +=
        list_logical_value(batch_diag, "workspace_reuse_enabled") ? 1 : 0;
      legacy_dcov_native_batch_distance_workspace_reuse_count +=
        list_integer_value(batch_diag, "distance_workspace_reuse_count");
      legacy_dcov_native_batch_statistic_moment_workspace_reuse_count +=
        list_integer_value(batch_diag, "statistic_moment_workspace_reuse_count");
      legacy_dcov_native_batch_lowrank_output_workspace_reuse_count +=
        list_integer_value(batch_diag, "lowrank_output_workspace_reuse_count");
      legacy_dcov_native_batch_lowrank_eig_workspace_reuse_count +=
        list_integer_value(batch_diag, "lowrank_eig_workspace_reuse_count");
      legacy_dcov_native_batch_oracle_column_copy_count +=
        list_integer_value(batch_diag, "column_copy_count");
      legacy_lowrank_timings.full_eig_count +=
        list_integer_value(batch_diag, "lowrank_full_eig_count");
      legacy_lowrank_timings.spectra_count +=
        list_integer_value(batch_diag, "lowrank_spectra_count");
      legacy_lowrank_timings.spectra_converged_count +=
        list_integer_value(batch_diag, "lowrank_spectra_converged_count");
      legacy_lowrank_timings.spectra_failed_count +=
        list_integer_value(batch_diag, "lowrank_spectra_failed_count");
      legacy_lowrank_timings.spectra_fallback_full_eig_count +=
        list_integer_value(batch_diag, "lowrank_spectra_fallback_full_eig_count");
      legacy_lowrank_timings.spectra_iterations +=
        list_integer_value(batch_diag, "lowrank_spectra_iterations");
      legacy_lowrank_timings.spectra_nconv +=
        list_integer_value(batch_diag, "lowrank_spectra_nconv");
      legacy_lowrank_timings.spectra_ncv = std::max(
        legacy_lowrank_timings.spectra_ncv,
        list_integer_value(batch_diag, "lowrank_spectra_ncv"));
      legacy_lowrank_timings.spectra_tol = std::max(
        legacy_lowrank_timings.spectra_tol,
        list_numeric_value(batch_diag, "lowrank_spectra_tol"));
      legacy_lowrank_timings.spectra_matvec_count +=
        list_integer_value(batch_diag, "lowrank_spectra_matvec_count");
      legacy_lowrank_timings.spectra_matvec_ms +=
        list_numeric_value(batch_diag, "lowrank_spectra_matvec_ms");
      const std::string batch_lowrank_mode =
        Rcpp::as<std::string>(batch_diag["lowrank_mode"]);
      legacy_lowrank_mode = batch_lowrank_mode == "spectra"
        ? fastkpc::LegacyDcovLowrankMode::Spectra
        : fastkpc::LegacyDcovLowrankMode::FullEig;
    };

    auto run_dcov_batch_for_tasks =
      [&](const std::vector<int>& task_indices) {
        const int batch_size = static_cast<int>(task_indices.size());
        std::vector<double> batch_pvalues(batch_size, NA_REAL);
        if (batch_size == 0) return batch_pvalues;

        const std::chrono::steady_clock::time_point materialize_start =
          std::chrono::steady_clock::now();
        std::vector<const double*> x_columns(
          static_cast<std::size_t>(batch_size));
        std::vector<const double*> y_columns(
          static_cast<std::size_t>(batch_size));
        for (int batch_col = 0; batch_col < batch_size; ++batch_col) {
          const std::pair<const double*, const double*> inputs =
            task_input_pointers(plan.tasks[task_indices[batch_col]]);
          x_columns[static_cast<std::size_t>(batch_col)] = inputs.first;
          y_columns[static_cast<std::size_t>(batch_col)] = inputs.second;
        }
        const double materialize_ms = elapsed_ms_since(materialize_start);
        level_dcov_materialize_ms += materialize_ms;
        legacy_dcov_native_batch_direct_input_enabled = true;

        const std::chrono::steady_clock::time_point batch_call_start =
          std::chrono::steady_clock::now();
        if (legacy_dcov_native_cuda_lowrank_metrics.enabled) {
          double pair_total_ms = 0.0;
          double eig_ms = 0.0;
          const int cuda_lowrank_batch_threads =
            native_legacy_cuda_lowrank_batch_threads(batch_size);
          const bool cuda_lowrank_batch_parallel_enabled =
            cuda_lowrank_batch_threads > 1;
          auto append_cuda_lowrank_progress =
            [&](const std::string& event,
                int completed_pairs,
                double batch_elapsed_ms) {
              append_native_legacy_progress(
                native_progress_csv_path,
                event,
                level,
                batch_size,
                level_provider_request_count,
                completed_pairs,
                ignored_after_delete,
                deletions,
                level_provider_call_ms,
                level_provider_matrix_copy_ms,
                level_dcov_materialize_ms,
                batch_elapsed_ms,
                elapsed_ms_since(level_start));
            };
          append_cuda_lowrank_progress(
            "dcov_cuda_lowrank_batch_start",
            legacy_dcov_native_count,
            0.0);
          try {
            if (legacy_dcov_native_cuda_lowrank_metrics
                  .component_cache_enabled) {
              LegacyDcovCudaLowrankComponentBatchOptions batch_options;
              batch_options.n = n;
              batch_options.num_col = num_col;
              batch_options.index = index;
              batch_options.ncv = legacy_dcov_native_cuda_lowrank_ncv;
              batch_options.tol = legacy_dcov_native_cuda_lowrank_tol;
              batch_options.maxitr = legacy_dcov_native_cuda_lowrank_maxitr;
              batch_options.batch_threads = cuda_lowrank_batch_threads;
              batch_options.key_mode =
                LegacyDcovCudaLowrankComponentBatchKeyMode::Pointer;
              batch_options.level_component_cache =
                (cuda_lowrank_level_component_cache_enabled &&
                 cuda_lowrank_level_component_cache)
                  ? cuda_lowrank_level_component_cache.get()
                  : nullptr;
              const LegacyDcovCudaLowrankComponentBatchRun batch_run =
                legacy_dcov_cuda_lowrank_gamma_component_batch(
                  x_columns, y_columns, batch_options);
              legacy_dcov_native_cuda_lowrank_metrics
                .component_batch_substrate_count += 1;
              legacy_dcov_native_cuda_lowrank_metrics
                .component_batch_substrate_pair_count += batch_size;
              legacy_dcov_native_cuda_lowrank_metrics
                .component_distance_ms += batch_run.component_distance_ms;
              legacy_dcov_native_cuda_lowrank_metrics
                .component_lowrank_ms += batch_run.component_lowrank_ms;
              legacy_dcov_native_cuda_lowrank_metrics
                .component_moment_ms += batch_run.component_moment_ms;
              legacy_dcov_native_cuda_lowrank_metrics
                .component_unaccounted_ms +=
                batch_run.component_unaccounted_ms;
              legacy_dcov_native_cuda_lowrank_metrics.component_eig_ms +=
                batch_run.component_eig_ms;
              legacy_dcov_native_cuda_lowrank_metrics.combine_ms +=
                batch_run.combine_ms;
              legacy_dcov_native_cuda_lowrank_metrics
                .component_cache_lookup_count +=
                batch_run.component_cache_lookup_count;
              legacy_dcov_native_cuda_lowrank_metrics
                .component_cache_hit_count +=
                batch_run.component_cache_hit_count;
              legacy_dcov_native_cuda_lowrank_metrics
                .component_cache_miss_count +=
                batch_run.component_cache_miss_count;
              legacy_dcov_native_cuda_lowrank_metrics
                .component_cache_entry_count +=
                batch_run.component_cache_entry_count;
              legacy_dcov_native_cuda_lowrank_metrics
                .component_cache_cross_batch_hit_count +=
                batch_run.component_cache_cross_batch_hit_count;
              legacy_dcov_native_cuda_lowrank_metrics
                .component_cache_eviction_count +=
                batch_run.component_cache_eviction_count;
              legacy_dcov_native_cuda_lowrank_metrics
                .component_cache_level_entry_count_max =
                std::max(
                  legacy_dcov_native_cuda_lowrank_metrics
                    .component_cache_level_entry_count_max,
                  batch_run.component_cache_level_entry_count_max);
              legacy_dcov_native_cuda_lowrank_metrics.spectra_matvec_count +=
                batch_run.spectra_matvec_count;
              legacy_dcov_native_cuda_lowrank_metrics.spectra_matvec_ms +=
                batch_run.spectra_matvec_ms;
              legacy_dcov_native_cuda_lowrank_metrics.kernel_launch_count +=
                batch_run.kernel_launch_count;
              legacy_dcov_native_cuda_lowrank_metrics
                .device_matrix_reuse_count +=
                batch_run.device_matrix_reuse_count;
              legacy_dcov_native_cuda_lowrank_metrics
                .device_workspace_reuse_count +=
                batch_run.device_workspace_reuse_count;
              legacy_dcov_native_cuda_lowrank_metrics.workspace_realloc_count +=
                batch_run.workspace_realloc_count;
              legacy_dcov_native_cuda_lowrank_metrics.matrix_bytes +=
                batch_run.matrix_bytes;
              legacy_dcov_native_cuda_lowrank_metrics.workspace_bytes =
                std::max(
                  legacy_dcov_native_cuda_lowrank_metrics.workspace_bytes,
                  batch_run.workspace_bytes);
              legacy_dcov_native_cuda_lowrank_metrics.matrix_h2d_ms +=
                batch_run.matrix_h2d_ms;
              legacy_dcov_native_cuda_lowrank_metrics
                .matrix_h2d_ms_during_compute +=
                batch_run.matrix_h2d_ms_during_compute;
              legacy_dcov_native_cuda_lowrank_metrics
                .matrix_h2d_ms_during_compute_max =
                std::max(
                  legacy_dcov_native_cuda_lowrank_metrics
                    .matrix_h2d_ms_during_compute_max,
                  batch_run.matrix_h2d_ms_during_compute_max);
              legacy_dcov_native_cuda_lowrank_metrics.workspace_alloc_ms +=
                batch_run.workspace_alloc_ms;
              legacy_dcov_native_cuda_lowrank_metrics.h2d_ms +=
                batch_run.h2d_ms;
              legacy_dcov_native_cuda_lowrank_metrics.kernel_ms +=
                batch_run.kernel_ms;
              legacy_dcov_native_cuda_lowrank_metrics.d2h_ms +=
                batch_run.d2h_ms;

              legacy_lowrank_timings.eig_ms += batch_run.component_eig_ms;
              legacy_lowrank_timings.spectra_count +=
                batch_run.component_cache_miss_count;
              legacy_lowrank_timings.spectra_converged_count +=
                batch_run.component_cache_miss_count;
              legacy_lowrank_timings.spectra_iterations +=
                batch_run.component_iterations;
              legacy_lowrank_timings.spectra_nconv +=
                batch_run.component_nconv;
              legacy_lowrank_timings.spectra_ncv = std::max(
                legacy_lowrank_timings.spectra_ncv,
                legacy_dcov_native_cuda_lowrank_ncv);
              legacy_lowrank_timings.spectra_tol = std::max(
                legacy_lowrank_timings.spectra_tol,
                legacy_dcov_native_cuda_lowrank_tol);
              legacy_lowrank_timings.spectra_matvec_count +=
                batch_run.spectra_matvec_count;
              legacy_lowrank_timings.spectra_matvec_ms +=
                batch_run.spectra_matvec_ms;
              legacy_lowrank_mode = fastkpc::LegacyDcovLowrankMode::Spectra;

              for (int i = 0; i < batch_size; ++i) {
                batch_pvalues[i] =
                  batch_run.p_values[static_cast<std::size_t>(i)];
                legacy_dcov_native_cuda_lowrank_metrics.count += 1;
                legacy_dcov_native_cuda_lowrank_metrics.converged_count += 1;
                const int completed_pairs = i + 1;
                if ((completed_pairs %
                     legacy_dcov_native_cuda_lowrank_progress_interval) == 0 ||
                    completed_pairs == batch_size) {
                  append_cuda_lowrank_progress(
                    "dcov_cuda_lowrank_pair_progress",
                    legacy_dcov_native_count + completed_pairs,
                    elapsed_ms_since(batch_call_start));
                }
              }
              pair_total_ms += batch_run.pair_total_ms;
              eig_ms += batch_run.component_eig_ms;
              legacy_dcov_native_cuda_lowrank_metrics.ms +=
                batch_run.pair_total_ms;
              append_native_cuda_lowrank_component_cache_progress(
                native_cuda_lowrank_component_cache_progress_csv_path,
                "component_cache_batch_complete",
                level,
                batch_size,
                legacy_dcov_native_cuda_lowrank_metrics.component_cache_scope,
                legacy_dcov_native_cuda_lowrank_metrics
                  .component_cache_level_max_entries,
                batch_run.component_cache_lookup_count,
                batch_run.component_cache_hit_count,
                batch_run.component_cache_miss_count,
                batch_run.component_cache_entry_count,
                batch_run.component_cache_cross_batch_hit_count,
                batch_run.component_cache_eviction_count,
                batch_run.component_cache_level_entry_count_max,
                batch_run.component_count,
                batch_run.component_total_ms,
                batch_run.component_distance_ms,
                batch_run.component_lowrank_ms,
                batch_run.component_moment_ms,
                batch_run.component_unaccounted_ms,
                batch_run.component_eig_ms,
                batch_run.combine_ms,
                batch_run.spectra_matvec_count,
                batch_run.spectra_matvec_ms,
                batch_run.kernel_launch_count,
                batch_run.device_matrix_reuse_count,
                batch_run.device_workspace_reuse_count,
                batch_run.workspace_realloc_count,
                batch_run.matrix_bytes,
                batch_run.workspace_bytes,
                batch_run.matrix_h2d_ms,
                batch_run.matrix_h2d_ms_during_compute,
                batch_run.matrix_h2d_ms_during_compute_max,
                batch_run.workspace_alloc_ms,
                batch_run.h2d_ms,
                batch_run.kernel_ms,
                batch_run.d2h_ms,
                elapsed_ms_since(level_start));
            } else {
              std::vector<LegacyDcovCudaLowrankGammaRun> runs(
                static_cast<std::size_t>(batch_size));
              if (cuda_lowrank_batch_parallel_enabled) {
                std::atomic<int> completed_in_batch(0);
                std::mutex error_mutex;
                std::mutex progress_mutex;
                std::string first_error;
                auto worker = [&](int thread_id) {
                  for (int i = thread_id; i < batch_size;
                       i += cuda_lowrank_batch_threads) {
                    {
                      std::lock_guard<std::mutex> lock(error_mutex);
                      if (!first_error.empty()) return;
                    }
                    try {
                      runs[static_cast<std::size_t>(i)] =
                        legacy_dcov_cuda_lowrank_gamma_compute(
                          x_columns[static_cast<std::size_t>(i)],
                          y_columns[static_cast<std::size_t>(i)],
                          n,
                          num_col,
                          index,
                          legacy_dcov_native_cuda_lowrank_ncv,
                          legacy_dcov_native_cuda_lowrank_tol,
                          legacy_dcov_native_cuda_lowrank_maxitr);
                      const int completed =
                        completed_in_batch.fetch_add(1) + 1;
                      if ((completed %
                           legacy_dcov_native_cuda_lowrank_progress_interval) == 0 ||
                          completed == batch_size) {
                        std::lock_guard<std::mutex> lock(progress_mutex);
                        append_cuda_lowrank_progress(
                          "dcov_cuda_lowrank_pair_progress",
                          legacy_dcov_native_count + completed,
                          elapsed_ms_since(batch_call_start));
                      }
                    } catch (const std::exception& error) {
                      std::lock_guard<std::mutex> lock(error_mutex);
                      if (first_error.empty()) first_error = error.what();
                      return;
                    } catch (...) {
                      std::lock_guard<std::mutex> lock(error_mutex);
                      if (first_error.empty()) {
                        first_error =
                          "native CUDA lowrank threaded batch failed";
                      }
                      return;
                    }
                  }
                };
                std::vector<std::thread> threads;
                threads.reserve(
                  static_cast<std::size_t>(cuda_lowrank_batch_threads));
                for (int thread_id = 0;
                     thread_id < cuda_lowrank_batch_threads;
                     ++thread_id) {
                  threads.emplace_back(worker, thread_id);
                }
                for (std::thread& thread : threads) thread.join();
                if (!first_error.empty()) Rcpp::stop(first_error);
              } else {
                for (int i = 0; i < batch_size; ++i) {
                  runs[static_cast<std::size_t>(i)] =
                    legacy_dcov_cuda_lowrank_gamma_compute(
                      x_columns[static_cast<std::size_t>(i)],
                      y_columns[static_cast<std::size_t>(i)],
                      n,
                      num_col,
                      index,
                      legacy_dcov_native_cuda_lowrank_ncv,
                      legacy_dcov_native_cuda_lowrank_tol,
                      legacy_dcov_native_cuda_lowrank_maxitr);
                  const int completed_pairs = legacy_dcov_native_count + i + 1;
                  if (((i + 1) %
                       legacy_dcov_native_cuda_lowrank_progress_interval) == 0 ||
                      i + 1 == batch_size) {
                    append_cuda_lowrank_progress(
                      "dcov_cuda_lowrank_pair_progress",
                      completed_pairs,
                      elapsed_ms_since(batch_call_start));
                  }
                }
              }
              for (int i = 0; i < batch_size; ++i) {
                const LegacyDcovCudaLowrankGammaRun& run =
                  runs[static_cast<std::size_t>(i)];
                batch_pvalues[i] = run.p_value;
                pair_total_ms += run.total_ms;
                eig_ms += run.eig_ms;
                accumulate_native_cuda_lowrank_run(run);
              }
            }
          } catch (...) {
            legacy_dcov_native_cuda_lowrank_metrics.error_count += 1;
            throw;
          }
          const double call_ms = elapsed_ms_since(batch_call_start);
          append_cuda_lowrank_progress(
            "dcov_cuda_lowrank_batch_complete",
            legacy_dcov_native_count + batch_size,
            call_ms);
          level_dcov_call_ms += call_ms;
          legacy_dcov_native_count += batch_size;
          legacy_dcov_native_ms += pair_total_ms;
          legacy_dcov_native_batch_count += 1;
          legacy_dcov_native_batch_pair_count += batch_size;
          legacy_dcov_native_batch_parallel_enabled =
            legacy_dcov_native_batch_parallel_enabled ||
            cuda_lowrank_batch_parallel_enabled;
          legacy_dcov_native_batch_parallel_threads = std::max(
            legacy_dcov_native_batch_parallel_threads,
            cuda_lowrank_batch_threads);
          legacy_dcov_native_batch_ms += call_ms;
          legacy_dcov_native_batch_lowrank_ms += eig_ms;
          legacy_dcov_native_batch_lowrank_eig_ms += eig_ms;
          legacy_dcov_native_batch_scalar_total_ms += pair_total_ms;
          legacy_dcov_native_batch_accounted_ms += eig_ms;
          legacy_dcov_native_batch_wrapper_overhead_ms +=
            std::max(0.0, call_ms - pair_total_ms);
          legacy_dcov_native_batch_overhead_ms +=
            std::max(0.0, call_ms - eig_ms);
          legacy_dcov_native_batch_materialize_ms += materialize_ms;
          legacy_dcov_native_batch_call_ms += call_ms;
        } else {
          Rcpp::List batch_result =
            fastkpc::legacy_dcov_gamma_cpp_compute_batch_ptrs(
              x_columns, y_columns, n, num_col, index);
          const double call_ms = elapsed_ms_since(batch_call_start);
          level_dcov_call_ms += call_ms;
          Rcpp::NumericVector batch_p_values = batch_result["p.value"];
          for (int i = 0; i < batch_size; ++i) {
            batch_pvalues[i] = batch_p_values[i];
          }
          Rcpp::List batch_diag = batch_result["diagnostics"];
          accumulate_batch_diag(batch_diag, batch_size, materialize_ms, call_ms);
        }
        return batch_pvalues;
      };

    auto replay_task = [&](int i, double pvalue) {
      const LayerCiTask& task = plan.tasks[i];
      const int edge_key = task.edge_x < task.edge_y
        ? task.edge_x * p + task.edge_y
        : task.edge_y * p + task.edge_x;
      const bool ignored = edge_done[edge_key] ||
        adjacency[static_cast<std::size_t>(task.edge_x) * p + task.edge_y] == 0;
      bool deleted = false;
      double pval = NA_REAL;

      if (ignored) {
        ++ignored_after_delete;
      } else {
        ++tests_replayed;
        pval = pvalue;
        if (!std::isfinite(pval)) pval = 1.0;
        const std::size_t pmax_idx =
          static_cast<std::size_t>(task.edge_x) * p + task.edge_y;
        const std::size_t pmax_rev =
          static_cast<std::size_t>(task.edge_y) * p + task.edge_x;
        if (pval > pmax[pmax_idx]) {
          pmax[pmax_idx] = pval;
          pmax[pmax_rev] = pval;
        }
        deleted = pval >= alpha;
        if (deleted) {
          ++deletions;
          delete_edges[static_cast<std::size_t>(task.edge_x) * p +
                       task.edge_y] = 1;
          delete_edges[static_cast<std::size_t>(task.edge_y) * p +
                       task.edge_x] = 1;
          sepsets[task.edge_x][task.edge_y] = task.conditioning_set;
          sepsets[task.edge_y][task.edge_x] = task.conditioning_set;
          edge_done[edge_key] = true;
        }
      }

      if (collect_trace && (!logical_trace || !ignored)) {
        ++global_task_id;
        task_global_id.push_back(global_task_id);
        task_level.push_back(level);
        task_index.push_back(i + 1);
        task_edge_x.push_back(task.edge_x + 1);
        task_edge_y.push_back(task.edge_y + 1);
        task_x.push_back(task.orientation_x + 1);
        task_y.push_back(task.orientation_y + 1);
        task_s_key.push_back(replay_s_key(task.conditioning_set));
        task_conditioning_size.push_back(
          static_cast<int>(task.conditioning_set.size()));
        task_p_used.push_back(pval);
        task_deleted.push_back(deleted ? 1 : 0);
        task_ignored.push_back(ignored ? 1 : 0);
      }
    };

    std::vector<double> pvalues(task_count, NA_REAL);
    append_native_legacy_progress(
      native_progress_csv_path,
      "dcov_start",
      level,
      task_count,
      static_cast<int>(request_targets.size()),
      0,
      0,
      0,
      level_provider_call_ms,
      level_provider_matrix_copy_ms,
      level_dcov_materialize_ms,
      level_dcov_call_ms,
      elapsed_ms_since(level_start));
    if (legacy_dcov_native_batch_mode == NativeLegacyDcovBatchMode::Level &&
        task_count > 0) {
      std::vector<int> task_indices(task_count);
      for (int i = 0; i < task_count; ++i) task_indices[i] = i;
      pvalues = run_dcov_batch_for_tasks(task_indices);
    } else if (legacy_dcov_native_batch_mode !=
                 NativeLegacyDcovBatchMode::Canonical &&
               legacy_dcov_native_batch_mode !=
                 NativeLegacyDcovBatchMode::Round) {
      for (int i = 0; i < task_count; ++i) {
        const std::chrono::steady_clock::time_point materialize_start =
          std::chrono::steady_clock::now();
        const LayerCiTask& task = plan.tasks[i];
        std::vector<double> rx;
        std::vector<double> ry;
        if (task.conditioning_set.empty()) {
          rx = numeric_matrix_column(data, task.orientation_x);
          ry = numeric_matrix_column(data, task.orientation_y);
        } else {
          const std::string x_key =
            native_residual_key(task.orientation_x, task.conditioning_set);
          const std::string y_key =
            native_residual_key(task.orientation_y, task.conditioning_set);
          rx = residual_columns[request_by_key[x_key]];
          ry = residual_columns[request_by_key[y_key]];
        }
        Rcpp::NumericVector x_vec(n);
        Rcpp::NumericVector y_vec(n);
        for (int row = 0; row < n; ++row) {
          x_vec[row] = rx[row];
          y_vec[row] = ry[row];
        }
        const double materialize_ms = elapsed_ms_since(materialize_start);
        level_dcov_materialize_ms += materialize_ms;
        legacy_dcov_native_scalar_materialize_ms += materialize_ms;
        const std::chrono::steady_clock::time_point scalar_call_start =
          std::chrono::steady_clock::now();
        if (legacy_dcov_native_cuda_lowrank_metrics.enabled) {
          try {
            const LegacyDcovCudaLowrankGammaRun run =
              legacy_dcov_cuda_lowrank_gamma_compute(
                REAL(x_vec),
                REAL(y_vec),
                n,
                num_col,
                index,
                legacy_dcov_native_cuda_lowrank_ncv,
                legacy_dcov_native_cuda_lowrank_tol,
                legacy_dcov_native_cuda_lowrank_maxitr);
            const double scalar_call_ms = elapsed_ms_since(scalar_call_start);
            level_dcov_call_ms += scalar_call_ms;
            legacy_dcov_native_scalar_call_ms += scalar_call_ms;
            pvalues[i] = run.p_value;
            ++legacy_dcov_native_count;
            legacy_dcov_native_ms += run.total_ms;
            accumulate_native_cuda_lowrank_run(run);
          } catch (...) {
            legacy_dcov_native_cuda_lowrank_metrics.error_count += 1;
            throw;
          }
        } else {
          const fastkpc::LegacyDcovGammaCppResult result =
            fastkpc::legacy_dcov_gamma_cpp_compute(x_vec, y_vec, num_col,
                                                   index);
          const double scalar_call_ms = elapsed_ms_since(scalar_call_start);
          level_dcov_call_ms += scalar_call_ms;
          legacy_dcov_native_scalar_call_ms += scalar_call_ms;
          pvalues[i] = result.p_value;
          ++legacy_dcov_native_count;
          legacy_dcov_native_ms += result.total_ms;
          legacy_lowrank_timings.full_eig_count +=
            result.lowrank_timings.full_eig_count;
          legacy_lowrank_timings.spectra_count +=
            result.lowrank_timings.spectra_count;
          legacy_lowrank_timings.spectra_converged_count +=
            result.lowrank_timings.spectra_converged_count;
          legacy_lowrank_timings.spectra_failed_count +=
            result.lowrank_timings.spectra_failed_count;
          legacy_lowrank_timings.spectra_fallback_full_eig_count +=
            result.lowrank_timings.spectra_fallback_full_eig_count;
          legacy_lowrank_timings.spectra_iterations +=
            result.lowrank_timings.spectra_iterations;
          legacy_lowrank_timings.spectra_nconv +=
            result.lowrank_timings.spectra_nconv;
          legacy_lowrank_timings.spectra_ncv = std::max(
            legacy_lowrank_timings.spectra_ncv,
            result.lowrank_timings.spectra_ncv);
          legacy_lowrank_timings.spectra_tol = std::max(
            legacy_lowrank_timings.spectra_tol,
            result.lowrank_timings.spectra_tol);
          legacy_lowrank_timings.spectra_matvec_count +=
            result.lowrank_timings.spectra_matvec_count;
          legacy_lowrank_timings.spectra_matvec_ms +=
            result.lowrank_timings.spectra_matvec_ms;
          legacy_lowrank_mode = result.lowrank_mode;
        }
      }
    }

    if (legacy_dcov_native_batch_mode ==
        NativeLegacyDcovBatchMode::Canonical) {
      std::vector<int> pending_task_indices;
      std::map<int, bool> pending_edge_keys;
      auto flush_pending = [&]() {
        if (pending_task_indices.empty()) return;
        const std::vector<double> pending_pvalues =
          run_dcov_batch_for_tasks(pending_task_indices);
        for (int i = 0; i < static_cast<int>(pending_task_indices.size()); ++i) {
          replay_task(pending_task_indices[i], pending_pvalues[i]);
        }
        pending_task_indices.clear();
        pending_edge_keys.clear();
      };

      for (int i = 0; i < task_count; ++i) {
        const LayerCiTask& task = plan.tasks[i];
        const int edge_key = task.edge_x < task.edge_y
          ? task.edge_x * p + task.edge_y
          : task.edge_y * p + task.edge_x;
        bool ignored = edge_done[edge_key] ||
          adjacency[static_cast<std::size_t>(task.edge_x) * p + task.edge_y] == 0;
        const bool conflicts_with_pending =
          pending_edge_keys.find(edge_key) != pending_edge_keys.end();
        if (!pending_task_indices.empty() &&
            (ignored || conflicts_with_pending)) {
          flush_pending();
          ignored = edge_done[edge_key] ||
            adjacency[static_cast<std::size_t>(task.edge_x) * p + task.edge_y] == 0;
        }
        if (ignored) {
          replay_task(i, NA_REAL);
        } else {
          pending_task_indices.push_back(i);
          pending_edge_keys[edge_key] = true;
        }
      }
      flush_pending();
    } else if (legacy_dcov_native_batch_mode ==
               NativeLegacyDcovBatchMode::Round) {
      std::vector<unsigned char> replayed(
        static_cast<std::size_t>(task_count), 0);
      int remaining_tasks = task_count;

      while (remaining_tasks > 0) {
        std::vector<int> pending_task_indices;
        std::map<int, bool> pending_edge_keys;
        bool made_progress = false;

        for (int i = 0; i < task_count; ++i) {
          if (replayed[static_cast<std::size_t>(i)] != 0) continue;
          const LayerCiTask& task = plan.tasks[i];
          const int edge_key = task.edge_x < task.edge_y
            ? task.edge_x * p + task.edge_y
            : task.edge_y * p + task.edge_x;
          const bool ignored = edge_done[edge_key] ||
            adjacency[static_cast<std::size_t>(task.edge_x) * p +
                      task.edge_y] == 0;
          if (ignored) {
            replay_task(i, NA_REAL);
            replayed[static_cast<std::size_t>(i)] = 1;
            --remaining_tasks;
            made_progress = true;
            continue;
          }
          if (pending_edge_keys.find(edge_key) != pending_edge_keys.end()) {
            continue;
          }
          pending_task_indices.push_back(i);
          pending_edge_keys[edge_key] = true;
        }

        if (!pending_task_indices.empty()) {
          const std::vector<double> pending_pvalues =
            run_dcov_batch_for_tasks(pending_task_indices);
          for (int i = 0; i < static_cast<int>(pending_task_indices.size()); ++i) {
            const int task_index = pending_task_indices[i];
            replay_task(task_index, pending_pvalues[i]);
            if (replayed[static_cast<std::size_t>(task_index)] == 0) {
              replayed[static_cast<std::size_t>(task_index)] = 1;
              --remaining_tasks;
            }
          }
          made_progress = true;
        }

        if (!made_progress) {
          Rcpp::stop("native round dCov batch made no replay progress");
        }
      }
    } else {
      for (int i = 0; i < task_count; ++i) {
        replay_task(i, pvalues[i]);
      }
    }

    for (int i = 0; i < p * p; ++i) {
      if (delete_edges[i] != 0) adjacency[i] = 0;
    }

    total_tasks_planned += task_count;
    total_tests_replayed += tests_replayed;
    total_ignored += ignored_after_delete;
    total_deletions += deletions;
    n_edgetests.push_back(tests_replayed);
    level_level.push_back(level);
    level_tasks_planned.push_back(task_count);
    level_tests_replayed.push_back(tests_replayed);
    level_ignored.push_back(ignored_after_delete);
    level_deletions.push_back(deletions);
    level_residual_provider_request_count.push_back(level_provider_request_count);
    level_residual_provider_call_ms.push_back(level_provider_call_ms);
    level_residual_provider_matrix_copy_ms.push_back(
      level_provider_matrix_copy_ms);
    level_residual_provider_total_ms.push_back(
      level_provider_call_ms + level_provider_matrix_copy_ms);
    level_legacy_dcov_native_materialize_ms.push_back(
      level_dcov_materialize_ms);
    level_legacy_dcov_native_call_ms.push_back(level_dcov_call_ms);
    append_native_legacy_progress(
      native_progress_csv_path,
      "level_complete",
      level,
      task_count,
      level_provider_request_count,
      tests_replayed,
      ignored_after_delete,
      deletions,
      level_provider_call_ms,
      level_provider_matrix_copy_ms,
      level_dcov_materialize_ms,
      level_dcov_call_ms,
      elapsed_ms_since(level_start));
    if (level > 0 && deletions == 0) break;
  }

  Rcpp::LogicalVector task_deleted_out(task_deleted.size());
  Rcpp::LogicalVector task_ignored_out(task_ignored.size());
  for (std::size_t i = 0; i < task_deleted.size(); ++i) {
    task_deleted_out[i] = task_deleted[i] != 0;
    task_ignored_out[i] = task_ignored[i] != 0;
  }
  Rcpp::DataFrame task_rows = Rcpp::DataFrame::create(
    Rcpp::Named("canonical_test_order_id") = Rcpp::wrap(task_global_id),
    Rcpp::Named("level") = Rcpp::wrap(task_level),
    Rcpp::Named("task_index") = Rcpp::wrap(task_index),
    Rcpp::Named("edge_x") = Rcpp::wrap(task_edge_x),
    Rcpp::Named("edge_y") = Rcpp::wrap(task_edge_y),
    Rcpp::Named("x") = Rcpp::wrap(task_x),
    Rcpp::Named("y") = Rcpp::wrap(task_y),
    Rcpp::Named("S_key") = Rcpp::wrap(task_s_key),
    Rcpp::Named("conditioning_size") =
      Rcpp::wrap(task_conditioning_size),
    Rcpp::Named("p_used") = Rcpp::wrap(task_p_used),
    Rcpp::Named("native_edge_deleted") = task_deleted_out,
    Rcpp::Named("native_edge_ignored") = task_ignored_out,
    Rcpp::Named("stringsAsFactors") = false
  );
  Rcpp::DataFrame level_rows = Rcpp::DataFrame::create(
    Rcpp::Named("level") = level_level,
    Rcpp::Named("tasks_planned") = level_tasks_planned,
    Rcpp::Named("tests_replayed") = level_tests_replayed,
    Rcpp::Named("tasks_ignored_after_delete") = level_ignored,
    Rcpp::Named("deletions") = level_deletions,
    Rcpp::Named("residual_provider_request_count") =
      level_residual_provider_request_count,
    Rcpp::Named("residual_provider_call_ms") =
      level_residual_provider_call_ms,
    Rcpp::Named("residual_provider_matrix_copy_ms") =
      level_residual_provider_matrix_copy_ms,
    Rcpp::Named("residual_provider_total_ms") =
      level_residual_provider_total_ms,
    Rcpp::Named("legacy_dcov_native_materialize_ms") =
      level_legacy_dcov_native_materialize_ms,
    Rcpp::Named("legacy_dcov_native_call_ms") =
      level_legacy_dcov_native_call_ms,
    Rcpp::Named("stringsAsFactors") = false
  );

  return Rcpp::List::create(
    Rcpp::Named("adjacency") = adjacency_to_matrix(adjacency, p),
    Rcpp::Named("sepsets") = sepsets_to_list(sepsets),
    Rcpp::Named("pMax") = pmax_to_matrix(pmax, p),
    Rcpp::Named("n.edgetests") = n_edgetests,
    Rcpp::Named("tasks") = task_rows,
    Rcpp::Named("levels") = level_rows,
    Rcpp::Named("summary") = Rcpp::List::create(
      Rcpp::Named("p") = p,
      Rcpp::Named("n") = n,
      Rcpp::Named("alpha") = alpha,
      Rcpp::Named("max_conditioning_size") = max_conditioning_size,
      Rcpp::Named("levels") = static_cast<int>(level_level.size()),
      Rcpp::Named("tasks_planned") = total_tasks_planned,
      Rcpp::Named("tests_replayed") = total_tests_replayed,
      Rcpp::Named("tasks_ignored_after_delete") = total_ignored,
      Rcpp::Named("deletions") = total_deletions,
      Rcpp::Named("ci_native_count") = total_tasks_planned,
      Rcpp::Named("legacy_dcov_native_count") = legacy_dcov_native_count,
      Rcpp::Named("legacy_dcov_native_ms") = legacy_dcov_native_ms,
      Rcpp::Named("legacy_dcov_native_scalar_materialize_ms") =
        legacy_dcov_native_scalar_materialize_ms,
      Rcpp::Named("legacy_dcov_native_scalar_call_ms") =
        legacy_dcov_native_scalar_call_ms,
      Rcpp::Named("legacy_dcov_native_numCol") = num_col,
      Rcpp::Named("legacy_dcov_native_lowrank_mode") =
        legacy_dcov_native_cuda_lowrank_metrics.enabled
          ? std::string("cuda_spectra")
          : std::string(fastkpc::legacy_dcov_lowrank_mode_name(
              legacy_lowrank_mode)),
      Rcpp::Named("legacy_dcov_native_lowrank_full_eig_count") =
        legacy_lowrank_timings.full_eig_count,
      Rcpp::Named("legacy_dcov_native_lowrank_spectra_count") =
        legacy_lowrank_timings.spectra_count,
      Rcpp::Named("legacy_dcov_native_lowrank_spectra_converged_count") =
        legacy_lowrank_timings.spectra_converged_count,
      Rcpp::Named("legacy_dcov_native_lowrank_spectra_failed_count") =
        legacy_lowrank_timings.spectra_failed_count,
      Rcpp::Named("legacy_dcov_native_lowrank_spectra_fallback_full_eig_count") =
        legacy_lowrank_timings.spectra_fallback_full_eig_count,
      Rcpp::Named("legacy_dcov_native_lowrank_spectra_iterations") =
        legacy_lowrank_timings.spectra_iterations,
      Rcpp::Named("legacy_dcov_native_lowrank_spectra_nconv") =
        legacy_lowrank_timings.spectra_nconv,
      Rcpp::Named("legacy_dcov_native_lowrank_spectra_ncv") =
        legacy_lowrank_timings.spectra_ncv,
      Rcpp::Named("legacy_dcov_native_lowrank_spectra_tol") =
        legacy_lowrank_timings.spectra_tol,
      Rcpp::Named("legacy_dcov_native_lowrank_spectra_matvec_count") =
        legacy_lowrank_timings.spectra_matvec_count,
      Rcpp::Named("legacy_dcov_native_lowrank_spectra_matvec_ms") =
        legacy_lowrank_timings.spectra_matvec_ms,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_backend_enabled") =
        legacy_dcov_native_cuda_lowrank_metrics.enabled,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_backend_count") =
        legacy_dcov_native_cuda_lowrank_metrics.count,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_backend_ms") =
        legacy_dcov_native_cuda_lowrank_metrics.ms,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_backend_error_count") =
        legacy_dcov_native_cuda_lowrank_metrics.error_count,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_backend_fallback_count") =
        legacy_dcov_native_cuda_lowrank_metrics.fallback_count,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_backend_converged_count") =
        legacy_dcov_native_cuda_lowrank_metrics.converged_count,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_component_cache_enabled") =
        legacy_dcov_native_cuda_lowrank_metrics.component_cache_enabled,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_component_cache_scope") =
        legacy_dcov_native_cuda_lowrank_metrics.component_cache_scope,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_component_cache_level_max_entries") =
        legacy_dcov_native_cuda_lowrank_metrics
          .component_cache_level_max_entries,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_component_cache_lookup_count") =
        legacy_dcov_native_cuda_lowrank_metrics.component_cache_lookup_count,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_component_cache_hit_count") =
        legacy_dcov_native_cuda_lowrank_metrics.component_cache_hit_count,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_component_cache_miss_count") =
        legacy_dcov_native_cuda_lowrank_metrics.component_cache_miss_count,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_component_cache_entry_count") =
        legacy_dcov_native_cuda_lowrank_metrics.component_cache_entry_count,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_component_cache_cross_batch_hit_count") =
        legacy_dcov_native_cuda_lowrank_metrics
          .component_cache_cross_batch_hit_count,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_component_cache_eviction_count") =
        legacy_dcov_native_cuda_lowrank_metrics.component_cache_eviction_count,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_component_cache_level_entry_count_max") =
        legacy_dcov_native_cuda_lowrank_metrics
          .component_cache_level_entry_count_max,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_component_batch_substrate_count") =
        legacy_dcov_native_cuda_lowrank_metrics
          .component_batch_substrate_count,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_component_batch_substrate_pair_count") =
        legacy_dcov_native_cuda_lowrank_metrics
          .component_batch_substrate_pair_count,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_backend_component_distance_ms") =
        legacy_dcov_native_cuda_lowrank_metrics.component_distance_ms,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_backend_component_lowrank_ms") =
        legacy_dcov_native_cuda_lowrank_metrics.component_lowrank_ms,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_backend_component_moment_ms") =
        legacy_dcov_native_cuda_lowrank_metrics.component_moment_ms,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_backend_component_unaccounted_ms") =
        legacy_dcov_native_cuda_lowrank_metrics.component_unaccounted_ms,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_backend_component_eig_ms") =
        legacy_dcov_native_cuda_lowrank_metrics.component_eig_ms,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_backend_combine_ms") =
        legacy_dcov_native_cuda_lowrank_metrics.combine_ms,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_backend_spectra_matvec_count") =
        legacy_dcov_native_cuda_lowrank_metrics.spectra_matvec_count,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_backend_spectra_matvec_ms") =
        legacy_dcov_native_cuda_lowrank_metrics.spectra_matvec_ms,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_backend_kernel_launch_count") =
        legacy_dcov_native_cuda_lowrank_metrics.kernel_launch_count,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_backend_device_matrix_reuse_count") =
        legacy_dcov_native_cuda_lowrank_metrics.device_matrix_reuse_count,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_backend_device_workspace_reuse_count") =
        legacy_dcov_native_cuda_lowrank_metrics.device_workspace_reuse_count,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_backend_workspace_realloc_count") =
        legacy_dcov_native_cuda_lowrank_metrics.workspace_realloc_count,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_backend_matrix_bytes") =
        legacy_dcov_native_cuda_lowrank_metrics.matrix_bytes,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_backend_workspace_bytes") =
        legacy_dcov_native_cuda_lowrank_metrics.workspace_bytes,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_backend_matrix_h2d_ms") =
        legacy_dcov_native_cuda_lowrank_metrics.matrix_h2d_ms,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_backend_matrix_h2d_ms_during_compute") =
        legacy_dcov_native_cuda_lowrank_metrics.matrix_h2d_ms_during_compute,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_backend_matrix_h2d_ms_during_compute_max") =
        legacy_dcov_native_cuda_lowrank_metrics.matrix_h2d_ms_during_compute_max,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_backend_workspace_alloc_ms") =
        legacy_dcov_native_cuda_lowrank_metrics.workspace_alloc_ms,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_backend_h2d_ms") =
        legacy_dcov_native_cuda_lowrank_metrics.h2d_ms,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_backend_kernel_ms") =
        legacy_dcov_native_cuda_lowrank_metrics.kernel_ms,
      Rcpp::Named("legacy_dcov_native_cuda_lowrank_backend_d2h_ms") =
        legacy_dcov_native_cuda_lowrank_metrics.d2h_ms,
      Rcpp::Named("legacy_dcov_native_batch_enabled") =
        legacy_dcov_native_batch_enabled,
      Rcpp::Named("legacy_dcov_native_batch_mode") =
        std::string(native_legacy_dcov_batch_mode_name(
          legacy_dcov_native_batch_mode)),
      Rcpp::Named("legacy_dcov_native_batch_count") =
        legacy_dcov_native_batch_count,
      Rcpp::Named("legacy_dcov_native_batch_pair_count") =
        legacy_dcov_native_batch_pair_count,
      Rcpp::Named("legacy_dcov_native_batch_parallel_enabled") =
        legacy_dcov_native_batch_parallel_enabled,
      Rcpp::Named("legacy_dcov_native_batch_parallel_threads") =
        legacy_dcov_native_batch_parallel_threads,
      Rcpp::Named("legacy_dcov_native_batch_ms") =
        legacy_dcov_native_batch_ms,
      Rcpp::Named("legacy_dcov_native_batch_materialize_ms") =
        legacy_dcov_native_batch_materialize_ms,
      Rcpp::Named("legacy_dcov_native_batch_call_ms") =
        legacy_dcov_native_batch_call_ms,
      Rcpp::Named("legacy_dcov_native_batch_input_ms") =
        legacy_dcov_native_batch_input_ms,
      Rcpp::Named("legacy_dcov_native_batch_distance_ms") =
        legacy_dcov_native_batch_distance_ms,
      Rcpp::Named("legacy_dcov_native_batch_lowrank_ms") =
        legacy_dcov_native_batch_lowrank_ms,
      Rcpp::Named("legacy_dcov_native_batch_lowrank_eig_ms") =
        legacy_dcov_native_batch_lowrank_eig_ms,
      Rcpp::Named("legacy_dcov_native_batch_lowrank_select_ms") =
        legacy_dcov_native_batch_lowrank_select_ms,
      Rcpp::Named("legacy_dcov_native_batch_lowrank_center_ms") =
        legacy_dcov_native_batch_lowrank_center_ms,
      Rcpp::Named("legacy_dcov_native_batch_lowrank_unaccounted_ms") =
        legacy_dcov_native_batch_lowrank_unaccounted_ms,
      Rcpp::Named("legacy_dcov_native_batch_statistic_ms") =
        legacy_dcov_native_batch_statistic_ms,
      Rcpp::Named("legacy_dcov_native_batch_moment_ms") =
        legacy_dcov_native_batch_moment_ms,
      Rcpp::Named("legacy_dcov_native_batch_pgamma_ms") =
        legacy_dcov_native_batch_pgamma_ms,
      Rcpp::Named("legacy_dcov_native_batch_accounted_ms") =
        legacy_dcov_native_batch_accounted_ms,
      Rcpp::Named("legacy_dcov_native_batch_scalar_total_ms") =
        legacy_dcov_native_batch_scalar_total_ms,
      Rcpp::Named("legacy_dcov_native_batch_wrapper_overhead_ms") =
        legacy_dcov_native_batch_wrapper_overhead_ms,
      Rcpp::Named("legacy_dcov_native_batch_overhead_ms") =
        legacy_dcov_native_batch_overhead_ms,
      Rcpp::Named("legacy_dcov_native_batch_workspace_reuse_count") =
        legacy_dcov_native_batch_workspace_reuse_count,
      Rcpp::Named("legacy_dcov_native_batch_distance_workspace_reuse_count") =
        legacy_dcov_native_batch_distance_workspace_reuse_count,
      Rcpp::Named("legacy_dcov_native_batch_statistic_moment_workspace_reuse_count") =
        legacy_dcov_native_batch_statistic_moment_workspace_reuse_count,
      Rcpp::Named("legacy_dcov_native_batch_lowrank_output_workspace_reuse_count") =
        legacy_dcov_native_batch_lowrank_output_workspace_reuse_count,
      Rcpp::Named("legacy_dcov_native_batch_lowrank_eig_workspace_reuse_count") =
        legacy_dcov_native_batch_lowrank_eig_workspace_reuse_count,
      Rcpp::Named("legacy_dcov_native_batch_oracle_column_copy_count") =
        legacy_dcov_native_batch_oracle_column_copy_count,
      Rcpp::Named("legacy_dcov_native_batch_direct_input_enabled") =
        legacy_dcov_native_batch_direct_input_enabled,
      Rcpp::Named("legacy_dcov_native_batch_column_materialize_count") =
        legacy_dcov_native_batch_column_materialize_count,
      Rcpp::Named("residual_provider_level_count") =
        residual_provider_level_count,
      Rcpp::Named("residual_provider_request_count") =
        residual_provider_request_count,
      Rcpp::Named("residual_provider_call_ms") =
        residual_provider_call_ms,
      Rcpp::Named("residual_provider_matrix_copy_ms") =
        residual_provider_matrix_copy_ms,
      Rcpp::Named("residual_provider_total_ms") =
        residual_provider_call_ms + residual_provider_matrix_copy_ms,
      Rcpp::Named("residual_provider_contract") =
        "level-residual-matrix-v1",
      Rcpp::Named("residual_provider_response_mode") =
        residual_provider_response_mode.empty()
          ? std::string("none")
          : residual_provider_response_mode,
      Rcpp::Named("residual_provider_response_backend") =
        residual_provider_response_backend.empty()
          ? std::string("none")
          : residual_provider_response_backend,
      Rcpp::Named("residual_provider_batch_count") =
        residual_provider_batch_count,
      Rcpp::Named("residual_provider_batch_max_requests") =
        residual_provider_batch_max_requests,
      Rcpp::Named("residual_provider_batch_mean_requests") =
        residual_provider_batch_count == 0
          ? 0.0
          : residual_provider_batch_request_sum /
            static_cast<double>(residual_provider_batch_count),
      Rcpp::Named("residual_provider_matrix_cell_count") =
        residual_provider_matrix_cell_count,
      Rcpp::Named("ci_backend") = "native-legacy-dcov.gamma",
      Rcpp::Named("residual_backend") = "provider-legacy-mgcv"
    )
  );
  END_RCPP
}

static const R_CallMethodDef call_methods[] = {
  {"C_full_cuda_ci_sha256_utf8", reinterpret_cast<DL_FUNC>(&C_full_cuda_ci_sha256_utf8), 1},
  {"C_full_cuda_ci_contract_identity", reinterpret_cast<DL_FUNC>(&C_full_cuda_ci_contract_identity), 2},
  {"C_full_cuda_ci_semantic_abi_info", reinterpret_cast<DL_FUNC>(&C_full_cuda_ci_semantic_abi_info), 0},
  {"C_full_cuda_ci_phase35_vertical_resource_snapshot", reinterpret_cast<DL_FUNC>(&C_full_cuda_ci_phase35_vertical_resource_snapshot), 0},
  {"C_full_cuda_ci_phase35_vertical", reinterpret_cast<DL_FUNC>(&C_full_cuda_ci_phase35_vertical), 6},
  {"C_full_cuda_ci_phase35_exact_batch", reinterpret_cast<DL_FUNC>(&C_full_cuda_ci_phase35_exact_batch), 6},
  {"C_full_cuda_ci_phase35_legacy_eig_batch", reinterpret_cast<DL_FUNC>(&C_full_cuda_ci_phase35_legacy_eig_batch), 6},
  {"C_fastkpc_cuda_available", reinterpret_cast<DL_FUNC>(&C_fastkpc_cuda_available), 0},
  {"C_fastkpc_cuda_device_info", reinterpret_cast<DL_FUNC>(&C_fastkpc_cuda_device_info), 0},
  {"C_fastkpc_cuda_phase3_environment_identity", reinterpret_cast<DL_FUNC>(&C_fastkpc_cuda_phase3_environment_identity), 1},
  {"C_fastkpc_cuda_legacy_dcov_gamma_cpp_oracle", reinterpret_cast<DL_FUNC>(&C_fastkpc_cuda_legacy_dcov_gamma_cpp_oracle), 4},
  {"C_legacy_dcov_spectra_matvec_cuda", reinterpret_cast<DL_FUNC>(&C_legacy_dcov_spectra_matvec_cuda), 2},
  {"C_legacy_dcov_spectra_matvec_cuda_handle_create", reinterpret_cast<DL_FUNC>(&C_legacy_dcov_spectra_matvec_cuda_handle_create), 1},
  {"C_legacy_dcov_spectra_matvec_cuda_handle_apply", reinterpret_cast<DL_FUNC>(&C_legacy_dcov_spectra_matvec_cuda_handle_apply), 2},
  {"C_legacy_dcov_spectra_matvec_cuda_handle_project", reinterpret_cast<DL_FUNC>(&C_legacy_dcov_spectra_matvec_cuda_handle_project), 2},
  {"C_legacy_dcov_spectra_matvec_cuda_handle_apply_sequence", reinterpret_cast<DL_FUNC>(&C_legacy_dcov_spectra_matvec_cuda_handle_apply_sequence), 2},
  {"C_legacy_dcov_spectra_matvec_cuda_operator_eigs", reinterpret_cast<DL_FUNC>(&C_legacy_dcov_spectra_matvec_cuda_operator_eigs), 5},
  {"C_legacy_dcov_spectra_matvec_cuda_lowrank_shadow", reinterpret_cast<DL_FUNC>(&C_legacy_dcov_spectra_matvec_cuda_lowrank_shadow), 6},
  {"C_legacy_dcov_spectra_matvec_cuda_lowrank_gamma", reinterpret_cast<DL_FUNC>(&C_legacy_dcov_spectra_matvec_cuda_lowrank_gamma), 7},
  {"C_legacy_dcov_spectra_matvec_cuda_lowrank_gamma_batch", reinterpret_cast<DL_FUNC>(&C_legacy_dcov_spectra_matvec_cuda_lowrank_gamma_batch), 7},
  {"C_legacy_dcov_gamma_cpp_component_cache_batch", reinterpret_cast<DL_FUNC>(&C_legacy_dcov_gamma_cpp_component_cache_batch), 5},
  {"C_legacy_dcov_spectra_matvec_cuda_handle_free", reinterpret_cast<DL_FUNC>(&C_legacy_dcov_spectra_matvec_cuda_handle_free), 1},
  {"C_fixed_sp_cuda_runtime_create", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_runtime_create), 1},
  {"C_fixed_sp_cuda_live_owner_snapshot", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_live_owner_snapshot), 0},
  {"C_fixed_sp_cuda_test_resource_snapshot", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_test_resource_snapshot), 0},
  {"C_fixed_sp_cuda_test_device_count", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_test_device_count), 0},
  {"C_fixed_sp_cuda_test_set_device", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_test_set_device), 1},
  {"C_fixed_sp_cuda_test_get_device", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_test_get_device), 0},
  {"C_fixed_sp_cuda_test_inject_next_resource_acquire_failure", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_test_inject_next_resource_acquire_failure), 1},
  {"C_fixed_sp_cuda_test_inject_next_resource_teardown_failure", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_test_inject_next_resource_teardown_failure), 1},
  {"C_fixed_sp_cuda_test_inject_next_resource_post_call_teardown_failure", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_test_inject_next_resource_post_call_teardown_failure), 1},
  {"C_fixed_sp_cuda_test_inject_next_prepared_static_shadow_body_failure", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_test_inject_next_prepared_static_shadow_body_failure), 0},
  {"C_fixed_sp_cuda_test_inject_next_blocked_consumer_launch_failure", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_test_inject_next_blocked_consumer_launch_failure), 0},
  {"C_fixed_sp_cuda_test_force_next_potrf_info", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_test_force_next_potrf_info), 1},
  {"C_fixed_sp_cuda_test_force_next_potrs_info", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_test_force_next_potrs_info), 1},
  {"C_fixed_sp_cuda_test_inject_next_device_free_failure", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_test_inject_next_device_free_failure), 0},
  {"C_fixed_sp_cuda_test_exercise_resource_teardown_failure", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_test_exercise_resource_teardown_failure), 1},
  {"C_fixed_sp_cuda_runtime_reserve", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_runtime_reserve), 6},
  {"C_fixed_sp_cuda_runtime_info", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_runtime_info), 1},
  {"C_fixed_sp_cuda_runtime_free", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_runtime_free), 1},
  {"C_fixed_sp_cuda_prepared_create", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_prepared_create), 2},
  {"C_fixed_sp_cuda_prepared_info", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_prepared_info), 1},
  {"C_fixed_sp_cuda_prepared_free", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_prepared_free), 1},
  {"C_fixed_sp_cuda_test_prepared_static_shadow", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_test_prepared_static_shadow), 1},
  {"C_fixed_sp_cuda_prepared_materialize_roots_for_test", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_prepared_materialize_roots_for_test), 1},
  {"C_fixed_sp_cuda_build_augmented_for_test", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_build_augmented_for_test), 4},
  {"C_fixed_sp_cuda_solve_batch", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_solve_batch), 6},
  {"C_fixed_sp_cuda_residual_info", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_residual_info), 1},
  {"C_fixed_sp_cuda_materialize_shadow", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_materialize_shadow), 2},
  {"C_fixed_sp_cuda_residual_release", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_residual_release), 1},
  {"C_fixed_sp_cuda_residual_free", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_residual_free), 1},
  {"C_fixed_sp_cuda_test_coefficient_shadow", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_test_coefficient_shadow), 1},
  {"C_fixed_sp_cuda_test_inject_consumer_registration_failure", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_test_inject_consumer_registration_failure), 1},
  {"C_fixed_sp_cuda_test_register_blocked_consumer", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_test_register_blocked_consumer), 1},
  {"C_fixed_sp_cuda_test_complete_consumer", reinterpret_cast<DL_FUNC>(&C_fixed_sp_cuda_test_complete_consumer), 1},
  {"C_fast_dcov_batch_cuda", reinterpret_cast<DL_FUNC>(&C_fast_dcov_batch_cuda), 4},
  {"C_fast_hsic_gamma_cuda", reinterpret_cast<DL_FUNC>(&C_fast_hsic_gamma_cuda), 3},
  {"C_fast_hsic_perm_cuda", reinterpret_cast<DL_FUNC>(&C_fast_hsic_perm_cuda), 6},
  {"C_fastspline_residual_cuda", reinterpret_cast<DL_FUNC>(&C_fastspline_residual_cuda), 4},
  {"C_fastspline_residual_batch_cuda", reinterpret_cast<DL_FUNC>(&C_fastspline_residual_batch_cuda), 5},
  {"C_mgcv_extract_gpu_test_checked_augmented_rows", reinterpret_cast<DL_FUNC>(&C_mgcv_extract_gpu_test_checked_augmented_rows), 2},
  {"C_full_cuda_ci_multi_penalty_gcv_evaluate_cpp", reinterpret_cast<DL_FUNC>(&C_full_cuda_ci_multi_penalty_gcv_evaluate_cpp), 9},
  {"C_full_cuda_ci_multi_penalty_gcv_optimize_cpp", reinterpret_cast<DL_FUNC>(&C_full_cuda_ci_multi_penalty_gcv_optimize_cpp), 8},
  {"C_full_cuda_ci_multi_penalty_gcv_evaluate_cuda", reinterpret_cast<DL_FUNC>(&C_full_cuda_ci_multi_penalty_gcv_evaluate_cuda), 11},
  {"C_full_cuda_ci_multi_penalty_gcv_optimize_cuda", reinterpret_cast<DL_FUNC>(&C_full_cuda_ci_multi_penalty_gcv_optimize_cuda), 11},
  {"C_full_cuda_ci_multi_penalty_gcv_prepared_create", reinterpret_cast<DL_FUNC>(&C_full_cuda_ci_multi_penalty_gcv_prepared_create), 11},
  {"C_full_cuda_ci_multi_penalty_gcv_prepared_info", reinterpret_cast<DL_FUNC>(&C_full_cuda_ci_multi_penalty_gcv_prepared_info), 1},
  {"C_full_cuda_ci_multi_penalty_gcv_prepared_free", reinterpret_cast<DL_FUNC>(&C_full_cuda_ci_multi_penalty_gcv_prepared_free), 1},
  {"C_full_cuda_ci_multi_penalty_gcv_optimize_batch", reinterpret_cast<DL_FUNC>(&C_full_cuda_ci_multi_penalty_gcv_optimize_batch), 4},
  {"C_full_cuda_ci_multi_penalty_gcv_optimize_multi", reinterpret_cast<DL_FUNC>(&C_full_cuda_ci_multi_penalty_gcv_optimize_multi), 3},
  {"C_full_cuda_ci_multi_penalty_gcv_residual_info", reinterpret_cast<DL_FUNC>(&C_full_cuda_ci_multi_penalty_gcv_residual_info), 1},
  {"C_full_cuda_ci_multi_penalty_gcv_residual_shadow", reinterpret_cast<DL_FUNC>(&C_full_cuda_ci_multi_penalty_gcv_residual_shadow), 1},
  {"C_full_cuda_ci_multi_penalty_gcv_residual_release", reinterpret_cast<DL_FUNC>(&C_full_cuda_ci_multi_penalty_gcv_residual_release), 1},
  {"C_full_cuda_ci_multi_penalty_gcv_residual_free", reinterpret_cast<DL_FUNC>(&C_full_cuda_ci_multi_penalty_gcv_residual_free), 1},
  {"C_full_cuda_ci_single_penalty_mroot_cuda", reinterpret_cast<DL_FUNC>(&C_full_cuda_ci_single_penalty_mroot_cuda), 3},
  {"C_full_cuda_ci_single_penalty_gcv_cuda", reinterpret_cast<DL_FUNC>(&C_full_cuda_ci_single_penalty_gcv_cuda), 15},
  {"C_full_cuda_ci_single_penalty_gcv_multi_cuda", reinterpret_cast<DL_FUNC>(&C_full_cuda_ci_single_penalty_gcv_multi_cuda), 2},
  {"C_full_cuda_ci_single_penalty_gcv_fixed_sp_cuda", reinterpret_cast<DL_FUNC>(&C_full_cuda_ci_single_penalty_gcv_fixed_sp_cuda), 16},
  {"C_mgcv_extract_gpu_solve_handle_fixed_sp", reinterpret_cast<DL_FUNC>(&C_mgcv_extract_gpu_solve_handle_fixed_sp), 6},
  {"C_mgcv_extract_gpu_solve_same_setup_batch_fixed_sp", reinterpret_cast<DL_FUNC>(&C_mgcv_extract_gpu_solve_same_setup_batch_fixed_sp), 6},
  {"C_fast_skeleton_cuda", reinterpret_cast<DL_FUNC>(&C_fast_skeleton_cuda), 6},
  {"C_fast_skeleton_cuda_cached", reinterpret_cast<DL_FUNC>(&C_fast_skeleton_cuda_cached), 7},
  {"C_fast_skeleton_cuda_backend", reinterpret_cast<DL_FUNC>(&C_fast_skeleton_cuda_backend), 18},
  {"C_fast_kpc_wanpdag_cuda", reinterpret_cast<DL_FUNC>(&C_fast_kpc_wanpdag_cuda), 24},
  {"C_precision_make_layer_plan_native", reinterpret_cast<DL_FUNC>(&C_precision_make_layer_plan_native), 2},
  {"C_precision_run_skeleton_ptable_native", reinterpret_cast<DL_FUNC>(&C_precision_run_skeleton_ptable_native), 4},
  {"C_precision_run_skeleton_provider_native", reinterpret_cast<DL_FUNC>(&C_precision_run_skeleton_provider_native), 5},
  {"C_precision_run_skeleton_dcov0_native", reinterpret_cast<DL_FUNC>(&C_precision_run_skeleton_dcov0_native), 5},
  {"C_precision_run_skeleton_exact_ci_native", reinterpret_cast<DL_FUNC>(&C_precision_run_skeleton_exact_ci_native), 6},
  {"C_precision_run_skeleton_residual_provider_native", reinterpret_cast<DL_FUNC>(&C_precision_run_skeleton_residual_provider_native), 7},
  {"C_precision_run_skeleton_residual_provider_legacy_dcov_native", reinterpret_cast<DL_FUNC>(&C_precision_run_skeleton_residual_provider_legacy_dcov_native), 7},
  {"C_precision_replay_layer_native", reinterpret_cast<DL_FUNC>(&C_precision_replay_layer_native), 10},
  {nullptr, nullptr, 0}
};

extern "C" void R_init_fastkpc_cuda(DllInfo* dll) {
  R_registerRoutines(dll, nullptr, call_methods, nullptr, nullptr);
  R_useDynamicSymbols(dll, FALSE);
}
