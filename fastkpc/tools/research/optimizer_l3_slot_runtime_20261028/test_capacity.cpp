#include "cuda/mgcv_l3_optimizer_slot_capacity.hpp"
#include <cassert>
#include <iostream>
int main() {
  using namespace fastkpc::l3_optimizer_slot_capacity;
  const auto b = 8ULL * 1024 * MiB, c = b - creation_limit;
  assert(accepted(Phase::creation, b, c, c));
  assert(!accepted(Phase::creation, b, c-1, c-1));
  assert(accepted(Phase::optimizer_done, b, c, c-growth_limit));
  assert(!accepted(Phase::optimizer_done, b, c, c-growth_limit-1));
  assert(!accepted(Phase::optimizer_done, b, c-1, c-growth_limit));
  assert(accepted(Phase::stage_a, b, c, minimum_free));
  assert(increment(b, minimum_free) > total_limit);
  assert(!accepted(Phase::stage_a, b, c, minimum_free-1));
  assert(!accepted(Phase::creation, minimum_free, minimum_free-1, minimum_free-1));
  assert(!accepted(Phase::optimizer_done, minimum_free, minimum_free, minimum_free-1));
  assert(accepted(Phase::creation, b, b+1, b+1));
  assert(accepted(Phase::optimizer_done, b, b+1, b+2));
  assert(increment(b, b+1) == 0);
  // Recorded old failure would pass the new declaration, without subtraction.
  assert(accepted(Phase::creation, 5836177408ULL, 4617732096ULL, 4617732096ULL));
  assert(accepted(Phase::optimizer_done, 5836177408ULL, 4617732096ULL, 3411869696ULL));
  std::cout << "PASS 15 split-capacity boundary checks\n";
}
