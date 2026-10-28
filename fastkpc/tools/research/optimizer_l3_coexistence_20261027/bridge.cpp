#include "probe.hpp"
#include "cuda/mgcv_multi_penalty_gcv_extended.hpp"
#include <variant>
// Exact class definitions extracted from the bound production capacity source.
#include "capacity-layout.inc"
namespace l3_probe {
std::shared_ptr<fastkpc::MultiPenaltyGcvCapacityPrepared> create(const Geometry& g,int device){
  validate(g);std::vector<std::vector<double>> roots,matrices;
  for(int j=0;j<3;++j){roots.emplace_back(g.roots.begin()+28*g.offsets[j],g.roots.begin()+28*(g.offsets[j]+g.ranks[j]));
    matrices.emplace_back(g.matrices.begin()+28*28*j,g.matrices.begin()+28*28*(j+1));}
  return fastkpc::create_multi_penalty_gcv_capacity_prepared(g.X.data(),g.qr.data(),g.tau.data(),g.r.data(),g.pivot.data(),
    roots,matrices,g.ranks,g.initial_log_sp.data(),351,28,3,33,device);
}
void bind(const std::shared_ptr<fastkpc::MultiPenaltyGcvCapacityPrepared>& p,const Geometry& g){
  if(!p||p->bucket!=fastkpc::MultiPenaltyGcvCapacityBucket::Small64)throw std::runtime_error("L3 probe bucket");
  rebind_cuda(std::get<std::shared_ptr<fastkpc::MultiPenaltyGcvCudaPreparedsmall64>>(p->handle),g);
}
void drain(const std::shared_ptr<fastkpc::MultiPenaltyGcvCapacityPrepared>& p){
  if(!p||p->bucket!=fastkpc::MultiPenaltyGcvCapacityBucket::Small64)throw std::runtime_error("L3 probe drain bucket");
  drain_cuda(std::get<std::shared_ptr<fastkpc::MultiPenaltyGcvCudaPreparedsmall64>>(p->handle));
}
}
