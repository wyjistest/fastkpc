#pragma once
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>
namespace fastkpc {class MultiPenaltyGcvCudaPreparedsmall64;}
namespace l3_probe {
struct Geometry {
  int n=351,q=28,penalty_count=3;
  std::vector<double> X,qr,tau,r,roots,matrices,initial_log_sp;
  std::vector<int> pivot,ranks,offsets;
};
inline void validate(const Geometry& g) {
  bool valid=g.n==351&&g.q==28&&g.penalty_count==3&&g.X.size()==351*28&&
    g.qr.size()==g.X.size()&&g.tau.size()==28&&g.r.size()==28*28&&g.pivot.size()==28&&
    g.ranks.size()==3&&g.offsets.size()==3&&g.initial_log_sp.size()==3&&g.matrices.size()==3*28*28;
  int rank=0;bool seen[28]={};
  if(valid){for(int p:g.pivot){if(p<0||p>=28||seen[p]){valid=false;break;}seen[p]=true;}
    for(int j=0;j<3;++j){valid=valid&&g.ranks[j]>0&&g.ranks[j]<=28&&g.offsets[j]==rank&&std::isfinite(g.initial_log_sp[j]);rank+=g.ranks[j];}}
  if(!valid||rank!=24||g.roots.size()!=28*24)throw std::runtime_error("L3 probe geometry outside envelope");
}
__attribute__((visibility("default"))) void rebind_cuda(
  const std::shared_ptr<fastkpc::MultiPenaltyGcvCudaPreparedsmall64>&,const Geometry&);
__attribute__((visibility("default"))) void drain_cuda(
  const std::shared_ptr<fastkpc::MultiPenaltyGcvCudaPreparedsmall64>&);
}
