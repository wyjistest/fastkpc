#pragma once
#include "cuda/full_cuda_ci_vertical.hpp"
#include "cuda/mgcv_multi_penalty_gcv.hpp"
#include <cuda_runtime.h>
#include <openssl/sha.h>
#include <dlfcn.h>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace stage_b_diag {
constexpr int n=351, k=35;
constexpr size_t fields=n*k+k+1, stride=fields+1;
inline void need(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }
inline void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
inline std::string env(const char* name) { auto p=std::getenv(name);need(p&&*p,name);return p; }
inline std::string hash(const void* p,size_t bytes) {
  unsigned char h[32];SHA256(static_cast<const unsigned char*>(p),bytes,h);
  std::ostringstream s;s<<std::hex<<std::setfill('0');for(auto v:h)s<<std::setw(2)<<int(v);return s.str();
}
inline double seconds(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
}
inline void mark(const char* name,int level) {
  using F=void(*)(const char*,int);
  static F f=[] {
    auto symbol=dlsym(RTLD_DEFAULT,"stage_b_trace_mark");
    if(!symbol) if(const char* path=std::getenv("CUDA_INJECTION64_PATH")) {
      void* h=dlopen(path,RTLD_NOW|RTLD_NOLOAD);
      if(h)symbol=dlsym(h,"stage_b_trace_mark");
    }
    if(std::getenv("STAGE_B_TRACE"))need(symbol,"injected trace marker unavailable");
    return reinterpret_cast<F>(symbol);
  }();
  if(f) f(name,level);
}
}
