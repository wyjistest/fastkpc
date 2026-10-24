#define FASTKPC_MULTI_PENALTY_MAX_COEFFICIENT_DIM 64
#define FASTKPC_MULTI_PENALTY_MAX_PENALTY_COUNT 7
#define FASTKPC_MULTI_PENALTY_GCV_EXTENSION 1
#define lapack312_multi lapack312_multi_small64
#define MultiPenaltyGcvCudaPrepared MultiPenaltyGcvCudaPreparedsmall64
#define MultiPenaltyGcvCudaResidualBatch MultiPenaltyGcvCudaResidualBatchsmall64
#define MultiPenaltyGcvCudaBatchResult MultiPenaltyGcvCudaBatchResultsmall64
#define MultiPenaltyGcvCudaMultiRequest MultiPenaltyGcvCudaMultiRequestsmall64
#define MultiPenaltyGcvCudaMultiResult MultiPenaltyGcvCudaMultiResultsmall64
#define multi_penalty_gcv_evaluate_cuda multi_penalty_gcv_evaluate_cuda_small64
#define multi_penalty_gcv_grouped_evaluate_prototype_cuda \
  multi_penalty_gcv_grouped_evaluate_prototype_cuda_small64
#define multi_penalty_gcv_optimize_cuda multi_penalty_gcv_optimize_cuda_small64
#define create_multi_penalty_gcv_cuda_prepared \
  create_multi_penalty_gcv_cuda_prepared_small64
#define multi_penalty_gcv_cuda_prepared_info \
  multi_penalty_gcv_cuda_prepared_info_small64
#define multi_penalty_gcv_cuda_optimize_batch \
  multi_penalty_gcv_cuda_optimize_batch_small64
#define multi_penalty_gcv_cuda_evaluate_fixed_batch \
  multi_penalty_gcv_cuda_evaluate_fixed_batch_small64
#define multi_penalty_gcv_cuda_finalize_selected_coefficients \
  multi_penalty_gcv_cuda_finalize_selected_coefficients_small64
#define multi_penalty_gcv_cuda_optimize_multi \
  multi_penalty_gcv_cuda_optimize_multi_small64
#define multi_penalty_gcv_cuda_residual_info \
  multi_penalty_gcv_cuda_residual_info_small64
#define acquire_multi_penalty_gcv_cuda_residual_consumer_view \
  acquire_multi_penalty_gcv_cuda_residual_consumer_view_small64
#define materialize_multi_penalty_gcv_cuda_residual_shadow \
  materialize_multi_penalty_gcv_cuda_residual_shadow_small64
#define release_multi_penalty_gcv_cuda_residual \
  release_multi_penalty_gcv_cuda_residual_small64

#include "mgcv_multi_penalty_gcv.cu"

#include "mgcv_multi_penalty_gcv_l5.hpp"

namespace fastkpc {

void rebind_multi_penalty_gcv_cuda_prepared_l5(
    const std::shared_ptr<MultiPenaltyGcvCudaPreparedsmall64>& prepared,
    const MultiPenaltyGcvL5Geometry& g) {
  if (!prepared) throw std::runtime_error("L5 slot was released");
  validate_multi_penalty_gcv_l5_geometry(g);
  std::lock_guard<std::mutex> lock(prepared->mutex);
  auto& p = *prepared;
  if (p.residual_slot_leased) throw std::runtime_error("L5 slot output is leased");
  if (p.n != g.n || p.q != g.q || p.penalty_count != g.penalty_count ||
      p.target_capacity != 8 || g.roots.size() > p.d_roots.count()) {
    throw std::runtime_error("L5 slot shape or root capacity changed");
  }
  check_cuda(cudaSetDevice(p.device_id), "select L5 slot device");
  // The caller owns the packet through optimize/finalize and failure drain.
  // No allocation, library handle or stream is created in this operation.
  MultiPenaltyGcvCudaDiagnostics d;
  upload_async(&p.d_X, g.X.data(), g.X.size(), p.stream.get(), &d);
  upload_async(&p.d_qr_packed, g.qr.data(), g.qr.size(), p.stream.get(), &d);
  upload_async(&p.d_tau, g.tau.data(), g.tau.size(), p.stream.get(), &d);
  upload_async(&p.d_r, g.r.data(), g.r.size(), p.stream.get(), &d);
  upload_async(&p.d_pivot, g.pivot.data(), g.pivot.size(), p.stream.get(), &d);
  upload_async(&p.d_roots, g.roots.data(), g.roots.size(), p.stream.get(), &d);
  upload_async(&p.d_root_offsets, g.offsets.data(), g.offsets.size(), p.stream.get(), &d);
  upload_async(&p.d_ranks, g.ranks.data(), g.ranks.size(), p.stream.get(), &d);
  upload_async(&p.d_matrices, g.matrices.data(), g.matrices.size(), p.stream.get(), &d);
  upload_async(&p.d_initial_log_sp, g.initial_log_sp.data(), g.initial_log_sp.size(), p.stream.get(), &d);
  p.initial_log_sp = g.initial_log_sp;
  p.total_root_rank = g.offsets.back() + g.ranks.back();
  ++p.generation;
  p.info.generation = p.generation;
  ++p.info.setup_upload_count;
  p.info.setup_h2d_bytes += (g.X.size() + g.qr.size() + g.tau.size() +
    g.r.size() + g.roots.size() + g.matrices.size() + g.initial_log_sp.size()) * sizeof(double) +
    (g.pivot.size() + g.ranks.size() + g.offsets.size()) * sizeof(int);
}

void drain_multi_penalty_gcv_cuda_prepared_l5(
    const std::shared_ptr<MultiPenaltyGcvCudaPreparedsmall64>& prepared) {
  if (!prepared) throw std::runtime_error("L5 slot was released");
  std::lock_guard<std::mutex> lock(prepared->mutex);
  check_cuda(cudaSetDevice(prepared->device_id), "select L5 slot drain device");
  check_cuda(cudaStreamSynchronize(prepared->stream.get()), "drain L5 slot");
}

}  // namespace fastkpc
