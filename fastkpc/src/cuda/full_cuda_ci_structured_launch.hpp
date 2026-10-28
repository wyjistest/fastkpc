#pragma once

#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace fastkpc {

// A plan captures this once. Never select entry and work storage independently.
enum class StructuredDbdsqrMode { Original = 0, FixedVtWorkPair = 1 };

inline StructuredDbdsqrMode parse_structured_dbdsqr_mode(const char* value) {
  if (!value || !*value || std::strcmp(value, "0") == 0)
    return StructuredDbdsqrMode::Original;
  if (std::strcmp(value, "1") == 0) return StructuredDbdsqrMode::FixedVtWorkPair;
  throw std::runtime_error("FASTKPC_DCC_GAMMA_STRUCTURED_WORK_PAIR must be 0 or 1");
}

inline std::size_t structured_dbdsqr_shared_bytes(StructuredDbdsqrMode mode, int n) {
  if (n < 6 || n > 1024) throw std::runtime_error("invalid structured DBDSQR dimension");
  switch (mode) {
    case StructuredDbdsqrMode::Original: return std::size_t(7)*n*sizeof(double);
    case StructuredDbdsqrMode::FixedVtWorkPair: return std::size_t(5)*n*sizeof(double);
  }
  throw std::runtime_error("invalid structured DBDSQR mode");
}

// Host-only rejection, before prepare or any other kernel is submitted. A
// device info code would instead enter the existing dense numerical fallback.
inline void validate_structured_dbdsqr_launch(
    StructuredDbdsqrMode plan_mode, StructuredDbdsqrMode entry_mode,
    int n, std::size_t shared_bytes, char uplo, int ncvt, int nru, int ncc) {
  const auto required = structured_dbdsqr_shared_bytes(plan_mode, n);
  if (plan_mode != entry_mode || shared_bytes != required ||
      uplo != 'L' || ncvt != 1 || nru != 0 || ncc != 0)
    throw std::runtime_error("structured DBDSQR entry/shared/configuration mismatch");
}

}  // namespace fastkpc
