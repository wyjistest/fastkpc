#pragma once
#include "slot_api.hpp"
namespace l3_observe {
struct Destination {
  const window_diag::Request* request=nullptr;
  double* coefficients=nullptr;
  double* fitted=nullptr;
  double* residual=nullptr;
  cudaEvent_t event=nullptr;
  int visits=0;
  double native_ms=0,retain_ms=0;
};
void begin(std::vector<Destination>&);
void end(bool require_complete=true);
}
