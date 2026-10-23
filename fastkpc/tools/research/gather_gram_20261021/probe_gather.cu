// Isolated diagnostic for path (i) of the level component prefill plan:
// three cache fields produced in wide prefill batches, then gathered into the
// production execution-context buffers in the ORIGINAL cache-miss grouping and
// order, followed by the verbatim production Gram GEMM + self-moment reduction.
// build.py copies the exact production statements into the two includes.
// No optimizer, residual fit or CPU component producer is used.
#include "full_cuda_ci_vertical.cu"
#include <cstdio>
#include <numeric>

namespace fastkpc {
struct GatherProbe {
  VerticalCallAccounting accounting;
  std::shared_ptr<FullCudaCiLegacyEigComponentCache> cache;
  std::unique_ptr<TrackedDeviceBuffer> residuals;
  int n, k, capacity;
  GatherProbe(int rows, int columns, int width, int lanes, std::size_t budget)
      : n(rows), k(columns), capacity(width) {
    cache = create_full_cuda_ci_legacy_eig_component_cache(
      n, k, budget, true, capacity, capacity * capacity, lanes);
    residuals = std::make_unique<TrackedDeviceBuffer>(
      std::size_t(n) * capacity * sizeof(double), &accounting);
    for (int i = 0; i < capacity; ++i) {
      std::size_t slot;
      if (!cache->reserve_slot(&slot) || slot != std::size_t(i))
        throw std::runtime_error("diagnostic cache capacity changed");
    }
  }
  // Full production component stage (distance -> eig -> selection -> Gram ->
  // self moment -> cache D2D), identical to the payload gate probe.
  void run_full(const double* input, int count, double* output, int* lane_count) {
    if (!input || !output || count < 1 || count > capacity)
      throw std::runtime_error("invalid gather probe shape");
    auto* execution_context = cache->execution_context();
    const auto component_cache = cache;
    const int miss_count = count, component_count = count, num_col = k;
    const std::size_t n_size = n, k_size = k, cell_count = n_size * n_size;
    const std::size_t vector_stride = n_size * k_size;
    const std::size_t cross_stride = k_size * k_size;
    const cudaStream_t stream = execution_context->stream();
    FullCudaCiLegacyEigBatchDiagnostics diagnostics;
    FixedSpBatchHostView batch;
    batch.n = n;
    DeviceResidualConsumerView residual_view;
    residual_view.residuals = static_cast<double*>(residuals->get());
    std::vector<int> targets(count), component_to_miss(count);
    std::iota(targets.begin(), targets.end(), 0);
    std::iota(component_to_miss.begin(), component_to_miss.end(), 0);
    std::vector<unsigned char> component_cache_reservations(count, 1);
    std::vector<std::size_t> component_cache_slots(count);
    std::iota(component_cache_slots.begin(), component_cache_slots.end(), 0U);
    double* vectors = execution_context->vectors();
    double* selected_values = execution_context->selected_values();
    double* totals = execution_context->totals();
    double* self_moments = execution_context->self_moments();
    check_cuda(cudaMemcpyAsync(residuals->get(), input,
      n_size * count * sizeof(double), cudaMemcpyHostToDevice, stream),
      "upload frozen real residuals");
    check_cuda(cudaMemcpyAsync(execution_context->component_targets(),
      targets.data(), count * sizeof(int), cudaMemcpyHostToDevice, stream),
      "upload diagnostic component indices");
    const double one = 1.0, zero = 0.0;
#include "component_stage.inc"
    std::vector<int> info(count, -1);
    check_cuda(cudaMemcpyAsync(info.data(), execution_context->solver_info(),
      count * sizeof(int), cudaMemcpyDeviceToHost, stream), "read solver info");
    const std::size_t entry_values = vector_stride + k_size + 2;
    for (int i = 0; i < count; ++i)
      check_cuda(cudaMemcpyAsync(output + std::size_t(i) * entry_values,
        cache->vectors(i), entry_values * sizeof(double),
        cudaMemcpyDeviceToHost, stream), "diagnostic payload shadow");
    check_cuda(cudaEventRecord(execution_context->completion(), stream),
      "record gather probe completion");
    check_cuda(cudaEventSynchronize(execution_context->completion()),
      "wait for diagnostic payload");
    if (std::any_of(info.begin(), info.end(), [](int x) { return x != 0; }))
      throw std::runtime_error("gather probe eig failed closed");
    *lane_count = diagnostics.component_eig_active_lane_count;
  }
  // Gram/self-moment stage only: the supplied vectors and selected values are
  // placed exactly where the production selection kernel leaves them, then the
  // verbatim production Gram GEMM and reduction run with miss_count = count.
  void run_gram(const double* vectors_in, const double* selected_in, int count,
                double* self_moment_out) {
    if (!vectors_in || !selected_in || !self_moment_out || count < 1 ||
        count > capacity)
      throw std::runtime_error("invalid gather gram shape");
    auto* execution_context = cache->execution_context();
    const int miss_count = count, num_col = k;
    const std::size_t n_size = n, k_size = k;
    const std::size_t vector_stride = n_size * k_size;
    const std::size_t cross_stride = k_size * k_size;
    const cudaStream_t stream = execution_context->stream();
    FixedSpBatchHostView batch;
    batch.n = n;
    double* vectors = execution_context->vectors();
    double* selected_values = execution_context->selected_values();
    check_cuda(cudaMemcpyAsync(vectors, vectors_in,
      vector_stride * count * sizeof(double), cudaMemcpyHostToDevice, stream),
      "upload gathered vectors");
    check_cuda(cudaMemcpyAsync(selected_values, selected_in,
      k_size * count * sizeof(double), cudaMemcpyHostToDevice, stream),
      "upload gathered selected values");
    const double one = 1.0, zero = 0.0;
#include "gram_stage.inc"
    check_cuda(cudaMemcpyAsync(self_moment_out, execution_context->self_moments(),
      count * sizeof(double), cudaMemcpyDeviceToHost, stream),
      "read gathered self moments");
    check_cuda(cudaEventRecord(execution_context->completion(), stream),
      "record gather gram completion");
    check_cuda(cudaEventSynchronize(execution_context->completion()),
      "wait for gathered self moments");
  }
};
}

extern "C" void* gather_probe_create(int n, int k, int capacity, int lanes,
                                     unsigned long long budget) {
  try { return new fastkpc::GatherProbe(n, k, capacity, lanes, budget); }
  catch (const std::exception& e) {
    std::fprintf(stderr, "gather create: %s\n", e.what()); return nullptr;
  }
}
extern "C" int gather_probe_run_full(void* p, const double* input, int count,
                                     double* output, int* active_lanes) {
  try {
    static_cast<fastkpc::GatherProbe*>(p)->run_full(input, count, output, active_lanes);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "gather run_full: %s\n", e.what()); return 1;
  }
}
extern "C" int gather_probe_run_gram(void* p, const double* vectors,
                                     const double* selected, int count,
                                     double* self_moments) {
  try {
    static_cast<fastkpc::GatherProbe*>(p)->run_gram(vectors, selected, count, self_moments);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "gather run_gram: %s\n", e.what()); return 1;
  }
}
extern "C" void gather_probe_destroy(void* p) {
  delete static_cast<fastkpc::GatherProbe*>(p);
}
extern "C" int gather_probe_snapshot(long long* values) {
  const auto s = fastkpc::full_cuda_ci_vertical_resource_snapshot();
  values[0] = s.live_device_allocations;
  values[1] = s.live_device_bytes;
  values[2] = s.live_streams;
  values[3] = s.live_events;
  return 0;
}
