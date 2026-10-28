#pragma once
#include <cstddef>
namespace l3_capacity {
constexpr std::size_t MiB = 1024ULL * 1024;
constexpr std::size_t creation_limit = 1536 * MiB;
constexpr std::size_t growth_limit = 1280 * MiB;
constexpr std::size_t total_limit = 2816 * MiB;
constexpr std::size_t minimum_free = 2048 * MiB;
enum class Phase { creation, optimizer_done, stage_a };
inline std::size_t increment(std::size_t before, std::size_t after) {
  return before > after ? before - after : 0;
}
inline bool accepted(Phase phase, std::size_t baseline, std::size_t created,
                     std::size_t free) {
  if (free < minimum_free) return false;
  if (phase == Phase::creation)
    return increment(baseline, free) <= creation_limit;
  if (phase == Phase::optimizer_done)
    return increment(baseline, created) <= creation_limit &&
           increment(created, free) <= growth_limit &&
           increment(baseline, free) <= total_limit;
  return true;  // Stage A has a headroom gate, not a raw-total ceiling.
}
}
