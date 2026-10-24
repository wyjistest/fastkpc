#pragma once
#include "../optimizer_windows_20261022/fixture.hpp"
#include <memory>

// Research-only ABI. The DSO hides all included production entrypoints.
#define SLOT_API __attribute__((visibility("default")))
namespace slot_diag {
using window_diag::Request;
struct Packed {
  const Request* request=nullptr;
  std::vector<double> roots,matrices;
  std::vector<int> offsets;
  int total_rank=0;
};
struct Slot;
struct Token;
using Handle=std::shared_ptr<Slot>;
struct Result {
  fastkpc::MultiPenaltyGcvCudaOptimization optimization;
  fastkpc::MultiPenaltyGcvCudaResidualInfo info;
  std::shared_ptr<Token> token;
};
struct Times {double bind_ms=0,optimize_ms=0,detach_ms=0;std::size_t h2d_bytes=0,d2d_bytes=0;};
SLOT_API Packed pack(const Request&);
SLOT_API Handle create(const Packed&,int target_capacity);
SLOT_API void bind(const Handle&,const Packed&,std::uint64_t identity);
SLOT_API Result execute(const Handle&,const Request&);
SLOT_API void detach(const Handle&,Result&,double* coefficients,double* fitted,double* residuals);
SLOT_API void abandon(const Handle&,Result&);
SLOT_API bool token_live(const Result&);
SLOT_API bool token_rejected(const Result&);
SLOT_API fastkpc::MultiPenaltyGcvCudaPreparedInfo info(const Handle&);
SLOT_API std::uint64_t binding(const Handle&);
SLOT_API void drain(const Handle&);
}
