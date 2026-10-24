#pragma once
#include <cstddef>
namespace layer_audit {
struct Counts {
  std::size_t allocations=0,bytes=0,streams=0,events=0,blas=0;
  std::size_t peak_bytes=0,allocation_calls=0,forbidden_calls=0;
};
Counts counts();
void native_scope(bool enabled);
void memory_limit(std::size_t bytes);
}
