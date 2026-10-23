// Isolated diagnostic: use the production context, kernels and cache layout.
// build.py copies the exact production component-stage statements into the
// include below. No optimizer, residual fit or CPU component producer is used.
#include "full_cuda_ci_vertical.cu"
#include <cstdio>
#include <numeric>

namespace fastkpc {
struct PayloadProbe {
  VerticalCallAccounting accounting;
  std::shared_ptr<FullCudaCiLegacyEigComponentCache> cache;
  std::unique_ptr<TrackedDeviceBuffer> residuals;
  int n, k, capacity;
  PayloadProbe(int rows, int columns, int width, int lanes, std::size_t budget)
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
  void run(const double* input, int count, double* output, int* lane_count) {
    if (!input || !output || count < 1 || count > capacity)
      throw std::runtime_error("invalid payload probe shape");
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
      "record payload probe completion");
    check_cuda(cudaEventSynchronize(execution_context->completion()),
      "wait for diagnostic payload");
    if (std::any_of(info.begin(), info.end(), [](int x) { return x != 0; }))
      throw std::runtime_error("payload probe eig failed closed");
    *lane_count = diagnostics.component_eig_active_lane_count;
  }
};
}

extern "C" void* payload_probe_create(int n, int k, int capacity, int lanes,
                                      unsigned long long budget) {
  try { return new fastkpc::PayloadProbe(n, k, capacity, lanes, budget); }
  catch (const std::exception& e) {
    std::fprintf(stderr, "payload create: %s\n", e.what()); return nullptr;
  }
}
extern "C" int payload_probe_run(void* p, const double* input, int count,
                                 double* output, int* active_lanes) {
  try {
    static_cast<fastkpc::PayloadProbe*>(p)->run(input, count, output, active_lanes);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "payload run: %s\n", e.what()); return 1;
  }
}
extern "C" void payload_probe_destroy(void* p) {
  delete static_cast<fastkpc::PayloadProbe*>(p);
}
extern "C" int payload_probe_snapshot(long long* values) {
  const auto s = fastkpc::full_cuda_ci_vertical_resource_snapshot();
  values[0] = s.live_device_allocations;
  values[1] = s.live_device_bytes;
  values[2] = s.live_streams;
  values[3] = s.live_events;
  return 0;
}
