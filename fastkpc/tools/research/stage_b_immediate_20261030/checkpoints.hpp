#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

constexpr int ip_checkpoint_capacity = 2048;
struct CheckpointHeader {
  int stage, ordinal, r, sweeps, ll, m, direction, iter, iterdivn, oldll, oldm;
  int info, rotations, overflow, coverage, reserved;
  unsigned long long tol, threshold, shift;
};
static_assert(offsetof(CheckpointHeader,tol)==16*sizeof(int),"checkpoint header layout");
__device__ unsigned long long ip_bits(double x) {
  return static_cast<unsigned long long>(__double_as_longlong(x));
}
struct CheckpointResult {
  int count, bad, first_event, first_stage;
  unsigned long long fields;
  unsigned long long stages[7];
};
struct CheckpointCursor {
  CheckpointHeader* headers = nullptr;
  unsigned long long* values = nullptr;
  CheckpointResult* result = nullptr;
  const CheckpointResult* reference = nullptr;
  int capacity = 0, stride = 0, compare = 0;
};

__device__ void ip_checkpoint_emit(CheckpointCursor& c, int stage, int r,
    int sweeps, int ll, int m, int direction, int iter, int iterdivn,
    int oldll, int oldm, int info, int rotations, int overflow, int coverage,
    double tol, double threshold, double shift,
    const double* d, const double* e, const double* q) {
  if (!c.result) return;
  auto& out = *c.result;
  const int event = out.count++;
  if (stage >= 1 && stage <= 7) ++out.stages[stage-1];
  auto fail = [&](int reason) {
    if (!out.bad) { out.first_event=event; out.first_stage=stage; }
    out.bad |= reason;
  };
  if (event >= c.capacity) { fail(1); return; }
  if (r < 0 || r > c.stride) { fail(2); return; }
  if (c.compare && (!c.reference || event >= c.reference->count)) {
    fail(4); return;
  }
  CheckpointHeader h{stage,event,r,sweeps,ll,m,direction,iter,iterdivn,oldll,oldm,
    info,rotations,overflow,coverage,0,ip_bits(tol),ip_bits(threshold),ip_bits(shift)};
  auto* saved = c.headers + event;
  if (!c.compare) *saved=h;
  else {
    const int* x=reinterpret_cast<const int*>(&h);
    const int* y=reinterpret_cast<const int*>(saved);
    for(int j=0;j<16;++j) if(x[j]!=y[j]) fail(8);
    if(h.tol!=saved->tol || h.threshold!=saved->threshold || h.shift!=saved->shift) fail(16);
  }
  out.fields += 19;
  auto* values=c.values+size_t(event)*3*c.stride;
  for(int j=0;j<r;++j) {
    const unsigned long long bits[3]={ip_bits(d[j]),ip_bits(e[j]),ip_bits(q[j])};
    for(int k=0;k<3;++k) {
      auto& expected=values[size_t(k)*c.stride+j];
      if(!c.compare) expected=bits[k];
      else if(expected!=bits[k]) fail(32);
    }
  }
  out.fields += size_t(3)*r;
}

struct EntryConfig { int r,ncvt,nru,ncc,ldvt,ldu,ldc,capacity,force; char uplo; };
