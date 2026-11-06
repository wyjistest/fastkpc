#include "cuda/mgcv_multi_penalty_gcv_l3.hpp"
#include "cuda/mgcv_multi_penalty_gcv_l4.hpp"
#include "cuda/mgcv_multi_penalty_gcv_l5.hpp"
#include "cuda/mgcv_multi_penalty_gcv_l6.hpp"
#include "cuda/full_cuda_ci_structured_launch.hpp"
#include <cassert>
#include <iostream>
#include <numeric>

template<class G, class Validate>
void check(int n, int targets, Validate validate) {
  G g;
  g.n = n; g.target_capacity = targets;
  g.X.resize(std::size_t(n) * g.q);
  g.qr = g.X; g.tau.resize(g.q); g.r.resize(g.q * g.q);
  g.pivot.resize(g.q); std::iota(g.pivot.begin(), g.pivot.end(), 0);
  g.ranks.assign(g.penalty_count, 8); g.offsets.resize(g.penalty_count);
  for (int j = 0; j < g.penalty_count; ++j) g.offsets[j] = 8 * j;
  g.initial_log_sp.assign(g.penalty_count, 0);
  g.roots.resize(g.q * 8 * g.penalty_count);
  g.matrices.resize(g.q * g.q * g.penalty_count);
  validate(g);
  auto reject = [&](const G& invalid) {
    bool failed = false;
    try { validate(invalid); } catch (const std::runtime_error&) { failed = true; }
    assert(failed);
  };
  auto bad = g; bad.n = g.q; reject(bad);
  bad = g; bad.target_capacity = 1; reject(bad);
  bad = g; bad.target_capacity = 65; reject(bad);
  bad = g; bad.X.pop_back(); reject(bad);
  bad = g; bad.pivot[0] = bad.pivot[1]; reject(bad);
  bad = g; bad.offsets.back() += 1; reject(bad);
}

int main() {
  using namespace fastkpc;
  for (int n : {76, 179, 243, 300, 351, 500, 768, 1025}) {
    for (int targets : {2, 5, 9, 19, 34, 58}) {
      check<MultiPenaltyGcvL3Geometry>(n, targets, validate_multi_penalty_gcv_l3_geometry);
      check<MultiPenaltyGcvL4Geometry>(n, targets, validate_multi_penalty_gcv_l4_geometry);
      check<MultiPenaltyGcvL5Geometry>(n, targets, validate_multi_penalty_gcv_l5_geometry);
      check<MultiPenaltyGcvL6Geometry>(n, targets, validate_multi_penalty_gcv_l6_geometry);
    }
  }
  assert(!structured_component_shape_supported(75, 35));
  assert(structured_component_shape_supported(76, 35));
  assert(structured_component_shape_supported(500, 35));
  assert(structured_component_shape_supported(1024, 35));
  assert(!structured_component_shape_supported(1025, 35));
  assert(!structured_component_shape_supported(500, 65));
  assert(structured_component_rotation_capacity(500) == 375000);
  assert(structured_component_rotation_capacity(1024) == 1572864);
  std::cout << "PASS 192 valid geometries, 1152 rejection cases, structured boundaries\n";
}
