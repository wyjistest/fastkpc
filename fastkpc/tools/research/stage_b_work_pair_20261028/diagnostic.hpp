#pragma once
#include <cuda_runtime.h>
#include <cstddef>
struct WorkPairState { int info,count,overflow,coverage,sweeps,canary,r,reserved; };
struct WorkPairView {
  const double* deq;
  const WorkPairState* state;
  const void* rotations;
  const double* data;
  const int* ints;
  const void* meta;
  const void* candidates;
  const double* values;
  const double* vectors;
  const double* means;
  const void* status;
  int n,k,count,capacity;
  int meta_size,candidate_size,status_size,rotation_size;
};
using WpAllocate=void*(*)(void*,std::size_t);
using WpCreate=void*(*)(int,int,int,int,cudaStream_t,WpAllocate,void*);
using WpDestroy=void(*)(void*);
using WpSubmit=WorkPairView(*)(void*,const double*,int);
using WpDirect=void(*)(void*,int,int,int,const double*);
using WpGet=WorkPairView(*)(void*);
