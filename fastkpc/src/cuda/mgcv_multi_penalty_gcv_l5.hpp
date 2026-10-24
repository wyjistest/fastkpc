#ifndef FASTKPC_MGCV_MULTI_PENALTY_GCV_L5_HPP
#define FASTKPC_MGCV_MULTI_PENALTY_GCV_L5_HPP

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace fastkpc {

// Owned pageable input. Keep it alive until the slot's stream has drained.
// This deliberately covers only the qualified 351 x 46, five-penalty shape.
struct MultiPenaltyGcvL5Geometry {
  int n = 351;
  int q = 46;
  int penalty_count = 5;
  std::vector<double> X, qr, tau, r, roots, matrices, initial_log_sp;
  std::vector<int> pivot, ranks, offsets;
};

inline void validate_multi_penalty_gcv_l5_geometry(
    const MultiPenaltyGcvL5Geometry& g) {
  bool valid = g.n == 351 && g.q == 46 && g.penalty_count == 5 &&
    g.X.size() == 351U * 46U && g.qr.size() == g.X.size() &&
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
  if (!valid) throw std::runtime_error("L5 slot geometry is outside the qualified envelope");
}

class MultiPenaltyGcvCudaPreparedsmall64;
void rebind_multi_penalty_gcv_cuda_prepared_l5(
    const std::shared_ptr<MultiPenaltyGcvCudaPreparedsmall64>& prepared,
    const MultiPenaltyGcvL5Geometry& geometry);
void drain_multi_penalty_gcv_cuda_prepared_l5(
    const std::shared_ptr<MultiPenaltyGcvCudaPreparedsmall64>& prepared);

}  // namespace fastkpc
#endif
