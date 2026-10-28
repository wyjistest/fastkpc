#pragma once
#include "../optimizer_windows_20261022/fixture.hpp"
#include <cuda_runtime.h>
#include <memory>

namespace fastkpc {class MultiPenaltyGcvCudaPreparedsmall64;}

#define SLOT_API __attribute__((visibility("default")))
namespace l6_slot {
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
struct Times {double bind_ms=0,optimize_ms=0,detach_ms=0;};
enum class Failure { None, AfterPartialUpload, AfterPartialDetach };
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
SLOT_API void arm_failure(const Handle&,Failure);
// Observer only: preserve the actual production prepared output without
// releasing its lease or changing its mathematical result.
SLOT_API void preserve_native_output(
  const std::shared_ptr<fastkpc::MultiPenaltyGcvCudaPreparedsmall64>&,
  int targets,double* coefficients,double* fitted,double* residuals,cudaEvent_t event);
}
