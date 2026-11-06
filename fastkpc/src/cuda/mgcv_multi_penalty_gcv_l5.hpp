#ifndef FASTKPC_MGCV_MULTI_PENALTY_GCV_L5_HPP
#define FASTKPC_MGCV_MULTI_PENALTY_GCV_L5_HPP

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace fastkpc {

// Owned pageable input. Keep it alive until the slot's stream has drained.
// The additive model fixes q and penalty_count; observation count and target
// capacity belong to the input, never to a development dataset.
struct MultiPenaltyGcvL5Geometry {
  int n = 0;
  int target_capacity = 0;
  int q = 46;
  int penalty_count = 5;
  std::vector<double> X, qr, tau, r, roots, matrices, initial_log_sp;
  std::vector<int> pivot, ranks, offsets;
};

inline void validate_multi_penalty_gcv_l5_geometry(
    const MultiPenaltyGcvL5Geometry& g) {
  bool valid = g.n > g.q && g.q == 46 && g.penalty_count == 5 &&
    g.target_capacity >= 2 && g.target_capacity <= 64 &&
    g.X.size() == static_cast<std::size_t>(g.n) * g.q && g.qr.size() == g.X.size() &&
    g.tau.size() == 46U && g.r.size() == 46U * 46U &&
    g.pivot.size() == 46U && g.ranks.size() == 5U &&
    g.offsets.size() == 5U && g.initial_log_sp.size() == 5U &&
    g.matrices.size() == 5U * 46U * 46U;
  int rank = 0;
  if (valid) {
    bool seen[46] = {};
    for (int p : g.pivot) {
      if (p < 0 || p >= 46 || seen[p]) { valid = false; break; }
      seen[p] = true;
    }
    for (int j = 0; j < 5; ++j) {
      valid = valid && g.ranks[j] > 0 && g.ranks[j] <= 46 &&
        g.offsets[j] == rank && std::isfinite(g.initial_log_sp[j]);
      rank += g.ranks[j];
    }
  }
  valid = valid && rank > 0 && 46 + rank <= 128 &&
    g.roots.size() == static_cast<std::size_t>(46 * rank);
  if (!valid) throw std::runtime_error("L5 slot geometry violates the additive model or buffer capacity");
}

class MultiPenaltyGcvCudaPreparedsmall64;
void rebind_multi_penalty_gcv_cuda_prepared_l5(
    const std::shared_ptr<MultiPenaltyGcvCudaPreparedsmall64>& prepared,
    const MultiPenaltyGcvL5Geometry& geometry);
void drain_multi_penalty_gcv_cuda_prepared_l5(
    const std::shared_ptr<MultiPenaltyGcvCudaPreparedsmall64>& prepared);

}  // namespace fastkpc
#endif
