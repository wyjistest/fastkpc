// Small activity/marker capability check before the long replay. No counters.
#include "common.hpp"
#include <iostream>
using namespace stage_b_diag;
int main() {
  try {
    check(cudaSetDevice(0));check(cudaFree(nullptr));
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    double *a=nullptr,*b=nullptr;check(cudaMalloc(&a,8192));check(cudaMalloc(&b,8192));
    std::vector<double> h(1024,1.0);mark("begin",2);
    check(cudaMemcpyAsync(a,h.data(),8192,cudaMemcpyHostToDevice,stream));
    check(cudaMemcpyAsync(b,a,8192,cudaMemcpyDeviceToDevice,stream));
    check(cudaMemsetAsync(a,0,8192,stream));
    check(cudaMemcpyAsync(h.data(),b,8192,cudaMemcpyDeviceToHost,stream));
    check(cudaStreamSynchronize(stream));mark("end",2);
    for(auto v:h)need(v==1.0,"copy check failed");
    check(cudaFree(a));check(cudaFree(b));check(cudaStreamDestroy(stream));
    std::cout<<"PASS activity marker/copies"<<std::endl;
  }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
