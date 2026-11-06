#ifndef FASTKPC_MGCV_MULTI_PENALTY_GCV_L6_HPP
#define FASTKPC_MGCV_MULTI_PENALTY_GCV_L6_HPP

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace fastkpc {

// Owned pageable input. Keep it alive until the slot's stream has drained.
// The additive model fixes q and penalty_count; observation count and target
// capacity belong to the input, never to a development dataset.
struct MultiPenaltyGcvL6Geometry {
  int n = 0;
  int target_capacity = 0;
  int q = 55;
  int penalty_count = 6;
  std::vector<double> X, qr, tau, r, roots, matrices, initial_log_sp;
  std::vector<int> pivot, ranks, offsets;
};

inline void validate_multi_penalty_gcv_l6_geometry(
    const MultiPenaltyGcvL6Geometry& g) {
  bool valid = g.n > g.q && g.q == 55 && g.penalty_count == 6 &&
    g.target_capacity >= 2 && g.target_capacity <= 64 &&
    g.X.size() == static_cast<std::size_t>(g.n) * g.q && g.qr.size() == g.X.size() &&
    g.tau.size() == 55U && g.r.size() == 55U * 55U &&
    g.pivot.size() == 55U && g.ranks.size() == 6U &&
    g.offsets.size() == 6U && g.initial_log_sp.size() == 6U &&
    g.matrices.size() == 6U * 55U * 55U;
  int rank = 0;
  if (valid) {
    bool seen[55] = {};
    for (int p : g.pivot) {
      if (p < 0 || p >= 55 || seen[p]) { valid = false; break; }
      seen[p] = true;
    }
    for (int j = 0; j < 6; ++j) {
      valid = valid && g.ranks[j] > 0 && g.ranks[j] <= 55 &&
        g.offsets[j] == rank && std::isfinite(g.initial_log_sp[j]);
      rank += g.ranks[j];
    }
  }
  valid = valid && rank == 48 && 55 + rank <= 128 &&
    g.roots.size() == static_cast<std::size_t>(55 * rank);
  if (!valid) throw std::runtime_error("L6 slot geometry violates the additive model or buffer capacity");
}

// A failed upload requires draining and discarding the slot; callers must not
// reuse partially rebound geometry. A live output lease always forbids rebind.
class MultiPenaltyGcvCudaPreparedsmall64;
void rebind_multi_penalty_gcv_cuda_prepared_l6(
    const std::shared_ptr<MultiPenaltyGcvCudaPreparedsmall64>& prepared,
    const MultiPenaltyGcvL6Geometry& geometry);
void drain_multi_penalty_gcv_cuda_prepared_l6(
    const std::shared_ptr<MultiPenaltyGcvCudaPreparedsmall64>& prepared);

}  // namespace fastkpc
#endif

