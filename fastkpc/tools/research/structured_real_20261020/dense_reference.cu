// Research-only adapter: reuse the retained benchmark's deterministic solver
// and lane submission, accepting the exact captured residual bytes.
#define main retained_benchmark_main
#include "../../phase10_cuda_full_eig_lane_benchmark.cu"
#undef main
#include <memory>

namespace {
__global__ void select_components(int n, int k, const double* all_values,
    const double* all_vectors, double* values, double* vectors) {
  const int c = blockIdx.x;
  __shared__ int chosen[128];
  if (threadIdx.x == 0) {
    for (int col = 0; col < k; ++col) {
      int best = -1;
      for (int j = 0; j < n; ++j) {
        bool used = false;
        for (int t = 0; t < col; ++t) used |= chosen[t] == j;
        if (!used && (best < 0 || fabs(all_values[size_t(c)*n+j]) >
              fabs(all_values[size_t(c)*n+best]) ||
            (fabs(all_values[size_t(c)*n+j]) == fabs(all_values[size_t(c)*n+best]) && j > best)))
          best = j;
      }
      chosen[col] = best;
      values[size_t(c)*k+col] = all_values[size_t(c)*n+best];
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < n*k; i += blockDim.x) {
    const int row = i % n, col = i/n;
    vectors[size_t(c)*n*k+i] = all_vectors[size_t(c)*n*n+size_t(chosen[col])*n+row];
  }
}
struct DenseContext {
  int n, k, count;
  DeviceBuffer<double> source, matrices, eigenvalues, values, vectors;
  DeviceBuffer<int> info;
  Stream control;
  Event ready;
  std::vector<std::unique_ptr<Lane>> owners;
  std::vector<Lane*> lanes;
  DenseContext(int n_, int k_, int count_, const double* inputs)
    : n(n_), k(k_), count(count_), source(size_t(n)*n*count),
      matrices(size_t(n)*n*count), eigenvalues(size_t(n)*count),
      values(size_t(k)*count), vectors(size_t(n)*k*count), info(count) {
    std::vector<double> host(size_t(n)*n*count);
    for (int c = 0; c < count; ++c)
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
          host[size_t(c)*n*n+size_t(j)*n+i] =
            std::abs(inputs[size_t(c)*n+i]-inputs[size_t(c)*n+j]);
    check_cuda(cudaMemcpy(source.get(), host.data(), source.bytes(), cudaMemcpyHostToDevice), "upload exact inputs");
    for (int l = 0; l < std::min(8,count); ++l) {
      owners.emplace_back(new Lane(n, matrices.get(), eigenvalues.get()));
      lanes.push_back(owners.back().get());
    }
  }
  size_t bytes() const {
    size_t result = source.bytes()+matrices.bytes()+eigenvalues.bytes()+
      values.bytes()+vectors.bytes()+info.bytes();
    for (const auto& lane : owners) result += lane->work->bytes();
    return result;
  }
};
void error_out(char* out, const char* message) { std::snprintf(out,512,"%s",message); }
}

extern "C" void* dense_create(int n, int k, int count, const double* inputs, char* error) {
  try {
    if (n<6 || k<1 || k>128 || k>=n || count<2) throw std::runtime_error("invalid dense dimensions");
    return new DenseContext(n,k,count,inputs);
  } catch (const std::exception& e) { error_out(error,e.what()); return nullptr; }
}
extern "C" int dense_run(void* handle, double* values, double* vectors, double* stats, char* error) {
  try {
    auto* c = static_cast<DenseContext*>(handle);
    const auto start = std::chrono::steady_clock::now();
    const Measurement m = run_lanes(c->n,0,c->count,size_t(c->n)*c->n,
      c->source,&c->matrices,&c->eigenvalues,&c->info,c->lanes,&c->control,&c->ready);
    std::vector<int> info(c->count);
    check_cuda(cudaMemcpy(info.data(),c->info.get(),c->info.bytes(),cudaMemcpyDeviceToHost),"dense info");
    for (int value : info) if (value) throw std::runtime_error("dense solver info nonzero");
    select_components<<<c->count,128,0,c->control.get()>>>(c->n,c->k,
      c->eigenvalues.get(),c->matrices.get(),c->values.get(),c->vectors.get());
    check_cuda(cudaGetLastError(),"select dense components");
    check_cuda(cudaMemcpyAsync(values,c->values.get(),c->values.bytes(),cudaMemcpyDeviceToHost,c->control.get()),"dense values shadow");
    check_cuda(cudaMemcpyAsync(vectors,c->vectors.get(),c->vectors.bytes(),cudaMemcpyDeviceToHost,c->control.get()),"dense vectors shadow");
    check_cuda(cudaStreamSynchronize(c->control.get()),"dense shadow completion");
    stats[0]=m.device_makespan_ms; stats[1]=m.host_submit_ms;
    stats[2]=host_elapsed_ms(start); stats[3]=double(c->bytes());
    return 0;
  } catch(const std::exception& e) { error_out(error,e.what()); return 1; }
}
extern "C" void dense_destroy(void* handle) { delete static_cast<DenseContext*>(handle); }

extern "C" void* reserve_memory(size_t bytes, char* error) {
  void* ptr = nullptr;
  try {
    check_cuda(cudaMalloc(&ptr,bytes),"reserve cache footprint");
    check_cuda(cudaMemset(ptr,0x5a,bytes),"touch cache footprint");
    check_cuda(cudaDeviceSynchronize(),"finish touching reservation");
    return ptr;
  } catch(const std::exception& e) {
    if(ptr) cudaFree(ptr);
    error_out(error,e.what()); return nullptr;
  }
}
extern "C" int release_memory(void* ptr, size_t bytes, char* error) {
  try {
    unsigned char ends[2];
    check_cuda(cudaMemcpy(ends,ptr,1,cudaMemcpyDeviceToHost),"check reserved head");
    check_cuda(cudaMemcpy(ends+1,static_cast<char*>(ptr)+bytes-1,1,cudaMemcpyDeviceToHost),"check reserved tail");
    check_cuda(cudaFree(ptr),"free cache reservation");
    if(ends[0]!=0x5a || ends[1]!=0x5a) throw std::runtime_error("cache reservation was overwritten");
    return 0;
  } catch(const std::exception& e) { error_out(error,e.what()); return 1; }
}
extern "C" int memory_info(size_t* free_bytes, size_t* total_bytes) {
  return int(cudaMemGetInfo(free_bytes,total_bytes));
}
