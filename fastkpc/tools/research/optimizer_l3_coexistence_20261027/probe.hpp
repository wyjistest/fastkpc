#pragma once
#include "geometry.hpp"
#include "cuda/mgcv_multi_penalty_gcv_capacity.hpp"
namespace l3_probe {
using Prepared=std::shared_ptr<fastkpc::MultiPenaltyGcvCapacityPrepared>;
Prepared create(const Geometry&,int);
void bind(const Prepared&,const Geometry&);
void drain(const Prepared&);
void begin(bool slots);
void finish();
void checkpoint(const char* name);
void workers(bool active);
void geometry(const Geometry&,const fastkpc::MultiPenaltyGcvCapacityRequest&);
void optimizer(const fastkpc::MultiPenaltyGcvCudaOptimization&,
  const std::shared_ptr<fastkpc::MultiPenaltyGcvCapacityResidualBatch>&);
void stage_a(const std::vector<std::string>& base_keys,int offset,int count,
  const fastkpc::MultiPenaltyGcvCudaCoefficientConsumerView&,
  const std::shared_ptr<fastkpc::MultiPenaltyGcvCapacityResidualBatch>&);
}
