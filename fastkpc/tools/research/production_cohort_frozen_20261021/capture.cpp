// Diagnostic interposition only. The accepted native and every numerical call
// are preserved; explicit shadow copies make this unsuitable for timing.
#include "mgcv_multi_penalty_gcv_capacity.hpp"
#include <cublas_v2.h>
#include <dlfcn.h>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace {
void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
void cuda_ok(cudaError_t status) {
  require(status == cudaSuccess, cudaGetErrorString(status));
}
void* native_symbol(const char* name) {
  const char* path = std::getenv("FASTKPC_COHORT_CAPTURE_NATIVE");
  require(path != nullptr, "missing capture native path");
  void* library = dlopen(path, RTLD_NOW | RTLD_NOLOAD);
  require(library != nullptr, "accepted native is not loaded");
  void* symbol = dlsym(library, name);
  dlclose(library);
  require(symbol != nullptr, "native capture symbol missing");
  return symbol;
}
template<class T> T native(const char* name) {
  return reinterpret_cast<T>(native_symbol(name));
}
const char* acquire_name =
  "_ZN7fastkpc57acquire_multi_penalty_gcv_capacity_residual_consumer_viewERKSt10shared_ptrINS_36MultiPenaltyGcvCapacityResidualBatchEE";
using Acquire = decltype(&fastkpc::acquire_multi_penalty_gcv_capacity_residual_consumer_view);
std::vector<double> download(const double* values, std::size_t count,
                             cudaStream_t stream) {
  std::vector<double> result(count);
  cuda_ok(cudaMemcpyAsync(result.data(), values, count * sizeof(double),
                          cudaMemcpyDeviceToHost, stream));
  cuda_ok(cudaStreamSynchronize(stream));
  return result;
}
std::string prefix(const char* kind, unsigned id) {
  const char* directory = std::getenv("FASTKPC_COHORT_CAPTURE_DIR");
  require(directory != nullptr, "missing capture directory");
  std::ostringstream out;
  out << directory << '/' << kind << '-' << std::setw(6)
      << std::setfill('0') << id;
  return out.str();
}
void binary(const std::string& path,
            const std::vector<std::vector<double>>& arrays) {
  std::ofstream out(path + ".f64", std::ios::binary);
  for (const auto& values : arrays)
    out.write(reinterpret_cast<const char*>(values.data()),
              values.size() * sizeof(double));
  require(out.good(), "capture binary write failed");
}
template<class T> void integers(std::ostream& out, const std::vector<T>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    out << values[i];
  }
  out << ']';
}
void keys(std::ostream& out, const std::vector<std::string>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    require(values[i].size() == 64 &&
              values[i].find_first_not_of("0123456789abcdef") == std::string::npos,
            "invalid captured key");
    out << '"' << values[i] << '"';
  }
  out << ']';
}
void metadata(std::ostream& out, int n, int q, int count,
              const std::vector<std::string>& target_keys,
              const std::vector<int>& info, const std::vector<int>& route,
              const std::vector<int>& rank) {
  out << "\"n\":" << n << ",\"q\":" << q << ",\"targets\":" << count
      << ",\"keys\":";
  keys(out, target_keys);
  out << ",\"info\":"; integers(out, info);
  out << ",\"route\":"; integers(out, route);
  out << ",\"rank\":"; integers(out, rank);
}
struct Active {
  int n, q, count, calls = 0;
  int math = -1, atomics = -1, version = -1, pointer = -1;
  std::vector<double> X, coefficients, fitted;
};
thread_local Active* active = nullptr;
std::atomic<unsigned> final_count{0}, prefill_count{0};
}

extern "C" cublasStatus_t cublasDgemm_v2(
    cublasHandle_t handle, cublasOperation_t op_a, cublasOperation_t op_b,
    int m, int n, int k, const double* alpha, const double* A, int lda,
    const double* B, int ldb, const double* beta, double* C, int ldc) {
  using Function = decltype(&cublasDgemm_v2);
  static Function original = [] {
    void* library = dlopen("libcublas.so.12", RTLD_NOW | RTLD_NOLOAD);
    require(library != nullptr, "cuBLAS not loaded");
    auto result = reinterpret_cast<Function>(dlsym(library, "cublasDgemm_v2"));
    dlclose(library);
    require(result != nullptr, "cuBLAS GEMM missing");
    return result;
  }();
  const auto status = original(handle, op_a, op_b, m, n, k, alpha, A, lda,
                               B, ldb, beta, C, ldc);
  if (active && status == CUBLAS_STATUS_SUCCESS) {
    require(op_a == CUBLAS_OP_N && op_b == CUBLAS_OP_N &&
            m == active->n && n == active->count && k == active->q &&
            lda == m && ldb == k && ldc == m, "unexpected finalizer GEMM");
    cublasPointerMode_t pointer;
    cublasMath_t math;
    cublasAtomicsMode_t atomics;
    cudaStream_t stream;
    require(cublasGetPointerMode(handle, &pointer) == CUBLAS_STATUS_SUCCESS &&
            pointer == CUBLAS_POINTER_MODE_HOST && *alpha == 1.0 && *beta == 0.0,
            "unexpected finalizer GEMM scalars");
    require(cublasGetStream(handle, &stream) == CUBLAS_STATUS_SUCCESS &&
            cublasGetMathMode(handle, &math) == CUBLAS_STATUS_SUCCESS &&
            cublasGetAtomicsMode(handle, &atomics) == CUBLAS_STATUS_SUCCESS &&
            cublasGetVersion(handle, &active->version) == CUBLAS_STATUS_SUCCESS,
            "cannot inspect original cuBLAS configuration");
    active->math = int(math); active->atomics = int(atomics);
    active->pointer = int(pointer);
    active->X = download(A, std::size_t(m) * k, stream);
    active->coefficients = download(B, std::size_t(k) * n, stream);
    active->fitted = download(C, std::size_t(m) * n, stream);
    ++active->calls;
  }
  return status;
}

namespace fastkpc {
std::shared_ptr<MultiPenaltyGcvCapacityResidualBatch>
multi_penalty_gcv_capacity_finalize_selected_coefficients(
    const std::shared_ptr<MultiPenaltyGcvCapacityPrepared>& prepared,
    const double* Y, const MultiPenaltyGcvCudaCoefficientConsumerView& coefficients,
    int n, int target_count, const std::vector<std::string>& target_keys) {
  using Function = decltype(&multi_penalty_gcv_capacity_finalize_selected_coefficients);
  static Function original = native<Function>(
    "_ZN7fastkpc57multi_penalty_gcv_capacity_finalize_selected_coefficientsERKSt10shared_ptrINS_31MultiPenaltyGcvCapacityPreparedEEPKdRKNS_42MultiPenaltyGcvCudaCoefficientConsumerViewEiiRKSt6vectorINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaISG_EE");
  require(active == nullptr, "nested finalization capture");
  Active record{n, coefficients.coefficient_dim, target_count};
  active = &record;
  std::shared_ptr<MultiPenaltyGcvCapacityResidualBatch> result;
  try { result = original(prepared, Y, coefficients, n, target_count, target_keys); }
  catch (...) { active = nullptr; throw; }
  active = nullptr;
  require(record.calls == 1, "finalizer GEMM was not intercepted exactly once");
  const auto view = native<Acquire>(acquire_name)(result);
  cuda_ok(cudaEventSynchronize(view.producer_completion_event));
  auto residual = download(view.residuals, std::size_t(n) * target_count,
                            view.producer_stream);
  const auto name = prefix("finalize", final_count++);
  binary(name, {record.X, record.coefficients,
               std::vector<double>(Y, Y + std::size_t(n) * target_count),
               record.fitted, residual});
  std::ofstream out(name + ".json");
  out << '{';
  metadata(out, n, record.q, target_count, target_keys, coefficients.solver_info,
           coefficients.decomposition_route, coefficients.numerical_rank);
  out << ",\"math\":" << record.math << ",\"atomics\":" << record.atomics
      << ",\"pointer\":" << record.pointer << ",\"cublas_version\":"
      << record.version << ",\"layout\":\"X,coefficients,Y,fitted,residual\"}\n";
  require(out.good(), "finalizer metadata write failed");
  return result;
}

MultiPenaltyGcvCapacityMultiResult multi_penalty_gcv_capacity_optimize_multi(
    std::vector<MultiPenaltyGcvCapacityRequest> requests,
    int requested_concurrency, bool retain_residuals) {
  using Function = decltype(&multi_penalty_gcv_capacity_optimize_multi);
  static Function original = native<Function>(
    "_ZN7fastkpc41multi_penalty_gcv_capacity_optimize_multiESt6vectorINS_30MultiPenaltyGcvCapacityRequestESaIS1_EEib");
  // Keep requests alive until their actual arena-producer coefficients are
  // captured. This changes diagnostic lifetime/host work, never numerical work.
  auto result = original(requests, requested_concurrency, retain_residuals);
  if (!retain_residuals) return result;
  require(result.residuals.size() == requests.size(), "prefill capture coverage");
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto view = native<Acquire>(acquire_name)(result.residuals[i]);
    const auto& request = requests[i];
    const auto& optimizer = result.setups[i];
    require(!view.fixed_sp_evaluation && view.target_keys == request.target_keys,
            "prefill identity changed");
    cuda_ok(cudaEventSynchronize(view.producer_completion_event));
    auto coefficient = download(view.coefficients,
      std::size_t(view.coefficient_dim) * view.target_count, view.producer_stream);
    const auto name = prefix("prefill", prefill_count++);
    binary(name, {coefficient, request.Y, optimizer.selected_log_sp});
    std::ofstream out(name + ".json");
    out << '{';
    metadata(out, view.n, view.coefficient_dim, view.target_count,
             view.target_keys, view.solver_info, view.decomposition_route,
             view.numerical_rank);
    out << ",\"penalties\":" << optimizer.penalty_count
        << ",\"layout\":\"coefficients,Y,selected_log_sp\"}\n";
    require(out.good(), "prefill metadata write failed");
  }
  return result;
}
}
