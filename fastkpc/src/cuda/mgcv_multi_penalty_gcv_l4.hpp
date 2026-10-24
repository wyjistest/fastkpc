#ifndef FASTKPC_MGCV_MULTI_PENALTY_GCV_L4_HPP
#define FASTKPC_MGCV_MULTI_PENALTY_GCV_L4_HPP

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace fastkpc {

// Owned pageable input. Keep it alive until the slot's stream has drained.
// This deliberately covers only the qualified 351 x 37, four-penalty shape.
struct MultiPenaltyGcvL4Geometry {
  int n = 351;
  int q = 37;
  int penalty_count = 4;
  std::vector<double> X, qr, tau, r, roots, matrices, initial_log_sp;
  std::vector<int> pivot, ranks, offsets;
};

inline void validate_multi_penalty_gcv_l4_geometry(
    const MultiPenaltyGcvL4Geometry& g) {
  bool valid = g.n == 351 && g.q == 37 && g.penalty_count == 4 &&
    g.X.size() == 351U * 37U && g.qr.size() == g.X.size() &&
    g.tau.size() == 37U && g.r.size() == 37U * 37U &&
    g.pivot.size() == 37U && g.ranks.size() == 4U &&
    g.offsets.size() == 4U && g.initial_log_sp.size() == 4U &&
    g.matrices.size() == 4U * 37U * 37U;
  int rank = 0;
  if (valid) {
    bool seen[37] = {};
    for (int p : g.pivot) {
      if (p < 0 || p >= 37 || seen[p]) { valid = false; break; }
      seen[p] = true;
    }
    for (int j = 0; j < 4; ++j) {
      valid = valid && g.ranks[j] > 0 && g.ranks[j] <= 37 &&
        g.offsets[j] == rank && std::isfinite(g.initial_log_sp[j]);
      rank += g.ranks[j];
    }
  }
  valid = valid && rank == 32 && 37 + rank <= 128 &&
    g.roots.size() == static_cast<std::size_t>(37 * rank);
  if (!valid) throw std::runtime_error("L4 slot geometry is outside the qualified envelope");
}

// A failed upload requires draining and discarding the slot; callers must not
// reuse partially rebound geometry. A live output lease always forbids rebind.
class MultiPenaltyGcvCudaPreparedsmall64;
void rebind_multi_penalty_gcv_cuda_prepared_l4(
    const std::shared_ptr<MultiPenaltyGcvCudaPreparedsmall64>& prepared,
    const MultiPenaltyGcvL4Geometry& geometry);
void drain_multi_penalty_gcv_cuda_prepared_l4(
    const std::shared_ptr<MultiPenaltyGcvCudaPreparedsmall64>& prepared);

}  // namespace fastkpc
#endif

