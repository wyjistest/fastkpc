// Compile the exact production implementation, with hidden host symbols.
// No numerical source, kernel launch shape or floating point operation changes.
#include "cuda/mgcv_multi_penalty_gcv_small64.cu"
#include "slot_api.hpp"

namespace l4_slot {
using namespace fastkpc;
using window_diag::need;
struct Slot {
  std::shared_ptr<MultiPenaltyGcvCudaPreparedsmall64> prepared;
  cudaEvent_t detached=nullptr;
  std::mutex mutex;
  bool bound=false,poisoned=false;
  Failure failure=Failure::None;
  std::uint64_t identity=0;
  std::vector<std::string> keys;
  ~Slot(){if(prepared){cudaSetDevice(prepared->device_id);cudaStreamSynchronize(prepared->stream.get());}
    if(detached)cudaEventDestroy(detached);}
};
struct Token {
  std::shared_ptr<MultiPenaltyGcvCudaResidualBatchsmall64> value;
  std::uint64_t binding=0;
};
Packed pack(const Request& r){
  const auto& g=r.geometry;Packed p;p.request=&r;
  need(g.n==351&&g.q==37&&g.penalties==4&&g.capacity>=2&&g.capacity<=18,"L4 shape required");
  need(g.X.size()==size_t(g.n)*g.q&&g.qr.size()==g.X.size()&&g.R.size()==size_t(g.q)*g.q&&
    g.tau.size()==size_t(g.q)&&g.pivot.size()==size_t(g.q)&&g.initial.size()==4&&
    g.ranks.size()==4&&g.roots.size()==4&&g.matrices.size()==4,"geometry dimensions");
  for(int j=0;j<4;++j){
    need(g.ranks[j]>0&&g.ranks[j]<=g.q&&g.roots[j].size()==size_t(g.q)*g.ranks[j]&&
      g.matrices[j].size()==size_t(g.q)*g.q&&std::isfinite(g.initial[j]),"penalty geometry");
    p.offsets.push_back(p.total_rank);p.total_rank+=g.ranks[j];
    p.roots.insert(p.roots.end(),g.roots[j].begin(),g.roots[j].end());
    p.matrices.insert(p.matrices.end(),g.matrices[j].begin(),g.matrices[j].end());
  }
  need(g.q+p.total_rank<=128,"small64 root envelope");return p;
}
Handle create(const Packed& packet,int capacity){
  need(packet.request!=nullptr,"missing geometry");const auto& g=packet.request->geometry;
  need(capacity>=g.capacity&&capacity<=18,"slot target capacity");
  auto s=std::make_shared<Slot>();s->prepared=create_multi_penalty_gcv_cuda_prepared_small64(
    g.X.data(),g.qr.data(),g.tau.data(),g.R.data(),g.pivot.data(),g.roots,g.matrices,
    g.ranks,g.initial.data(),g.n,g.q,g.penalties,capacity,0);
  check_cuda(cudaEventCreateWithFlags(&s->detached,cudaEventDisableTiming|cudaEventBlockingSync),"slot detach event");
  return s;
}
void bind(const Handle& s,const Packed& packet,std::uint64_t identity){
  need(s&&packet.request,"slot bind arguments");std::lock_guard<std::mutex> guard(s->mutex);
  need(!s->poisoned,"SLOT_POISONED");auto& p=*s->prepared;std::lock_guard<std::mutex> lock(p.mutex);
  need(!p.residual_slot_leased,"SLOT_OUTPUT_BUSY");const auto& r=*packet.request;const auto& g=r.geometry;
  need(g.n==p.n&&g.q==p.q&&g.penalties==p.penalty_count&&int(r.keys.size())==g.capacity&&
       g.capacity>=2&&g.capacity<=p.target_capacity&&r.Y.size()==size_t(g.n)*g.capacity&&
       g.X.size()==p.d_X.count()&&g.qr.size()==p.d_qr_packed.count()&&g.tau.size()==p.d_tau.count()&&
       g.R.size()==p.d_r.count()&&g.pivot.size()==p.d_pivot.count()&&g.ranks.size()==4&&g.initial.size()==4&&
       packet.roots.size()<=p.d_roots.count()&&packet.roots.size()==size_t(g.q)*packet.total_rank&&
       packet.total_rank==32&&packet.matrices.size()==p.d_matrices.count()&&packet.offsets.size()==4,
       "SLOT_SHAPE_REJECTED");
  check_cuda(cudaSetDevice(p.device_id),"slot device");
  // packet and its Request remain owned by the runner until all streams drain.
  // No temporary pageable payload is destroyed while an upload may reference it.
  MultiPenaltyGcvCudaDiagnostics d;
  try{
    upload_async(&p.d_X,g.X.data(),g.X.size(),p.stream.get(),&d);
    upload_async(&p.d_qr_packed,g.qr.data(),g.qr.size(),p.stream.get(),&d);
    upload_async(&p.d_tau,g.tau.data(),g.tau.size(),p.stream.get(),&d);
    if(s->failure==Failure::AfterPartialUpload)throw std::runtime_error("INJECTED_PARTIAL_UPLOAD");
    upload_async(&p.d_r,g.R.data(),g.R.size(),p.stream.get(),&d);
    upload_async(&p.d_pivot,g.pivot.data(),g.pivot.size(),p.stream.get(),&d);
    upload_async(&p.d_roots,packet.roots.data(),packet.roots.size(),p.stream.get(),&d);
    upload_async(&p.d_root_offsets,packet.offsets.data(),packet.offsets.size(),p.stream.get(),&d);
    upload_async(&p.d_ranks,g.ranks.data(),g.ranks.size(),p.stream.get(),&d);
    upload_async(&p.d_matrices,packet.matrices.data(),packet.matrices.size(),p.stream.get(),&d);
    upload_async(&p.d_initial_log_sp,g.initial.data(),g.initial.size(),p.stream.get(),&d);
    p.initial_log_sp=g.initial;p.total_root_rank=packet.total_rank;
    ++p.generation;p.info.generation=p.generation;++p.info.setup_upload_count;
    p.info.setup_h2d_bytes+=(g.X.size()+g.qr.size()+g.tau.size()+g.R.size()+packet.roots.size()+
      packet.matrices.size()+g.initial.size())*sizeof(double)+(g.pivot.size()+g.ranks.size()+packet.offsets.size())*sizeof(int);
    s->keys=r.keys;s->identity=identity;s->bound=true;
  }catch(...){s->poisoned=true;throw;}
}
Result execute(const Handle& s,const Request& r){
  need(bool(s),"missing slot");std::lock_guard<std::mutex> guard(s->mutex);
  need(s->bound&&!s->poisoned&&s->keys==r.keys,"SLOT_BINDING_REJECTED");
  auto b=multi_penalty_gcv_cuda_optimize_batch_small64(s->prepared,r.Y.data(),r.geometry.n,
    r.keys.size(),r.keys,r.control);
  Result result;result.optimization=std::move(b.optimization);
  result.info=multi_penalty_gcv_cuda_residual_info_small64(b.residual);
  result.token=std::make_shared<Token>();result.token->value=std::move(b.residual);
  result.token->binding=s->identity;s->bound=false;return result;
}
void detach(const Handle& s,Result& r,double* coefficients,double* fitted,double* residuals){
  need(s&&r.token&&coefficients&&fitted&&residuals,"detach arguments");
  std::lock_guard<std::mutex> guard(s->mutex);auto& p=*s->prepared;
  need(!s->poisoned&&r.token->binding==s->identity&&r.token->value->owner.get()==s->prepared.get(),"SLOT_TOKEN_OWNER");
  const auto view=acquire_multi_penalty_gcv_cuda_residual_consumer_view_small64(r.token->value);
  check_cuda(cudaSetDevice(p.device_id),"detach device");
  try{
    check_cuda(cudaMemcpyAsync(coefficients,view.coefficients,size_t(view.coefficient_dim)*view.target_count*sizeof(double),
      cudaMemcpyDeviceToDevice,p.stream.get()),"detach coefficients");
    if(s->failure==Failure::AfterPartialDetach)throw std::runtime_error("INJECTED_PARTIAL_DETACH");
    check_cuda(cudaMemcpyAsync(fitted,p.d_fitted.get(),size_t(view.n)*view.target_count*sizeof(double),
      cudaMemcpyDeviceToDevice,p.stream.get()),"detach fitted");
    check_cuda(cudaMemcpyAsync(residuals,view.residuals,size_t(view.n)*view.target_count*sizeof(double),
      cudaMemcpyDeviceToDevice,p.stream.get()),"detach residuals");
    check_cuda(cudaEventRecord(s->detached,p.stream.get()),"record detach");
    check_cuda(cudaEventSynchronize(s->detached),"complete detach before reuse");
    release_multi_penalty_gcv_cuda_residual_small64(r.token->value);
  }catch(...){s->poisoned=true;throw;}
}
void abandon(const Handle& s,Result& r){
  need(bool(s),"missing slot");std::lock_guard<std::mutex> guard(s->mutex);
  s->poisoned=true;s->bound=false;check_cuda(cudaSetDevice(s->prepared->device_id),"abandon device");
  check_cuda(cudaStreamSynchronize(s->prepared->stream.get()),"abandon drain");
  if(r.token)release_multi_penalty_gcv_cuda_residual_small64(r.token->value);
}
bool token_live(const Result& r){return r.token&&multi_penalty_gcv_cuda_residual_info_small64(r.token->value).device_resident;}
bool token_rejected(const Result& r){try{acquire_multi_penalty_gcv_cuda_residual_consumer_view_small64(r.token->value);return false;}
  catch(const std::runtime_error&){return true;}}
MultiPenaltyGcvCudaPreparedInfo info(const Handle& s){return multi_penalty_gcv_cuda_prepared_info_small64(s->prepared);}
std::uint64_t binding(const Handle& s){std::lock_guard<std::mutex> guard(s->mutex);return s->identity;}
void drain(const Handle& s){check_cuda(cudaSetDevice(s->prepared->device_id),"drain device");
  check_cuda(cudaStreamSynchronize(s->prepared->stream.get()),"slot drain");}
void arm_failure(const Handle& s,Failure failure){
  std::lock_guard<std::mutex> guard(s->mutex);s->failure=failure;
}
void preserve_native_output(const std::shared_ptr<MultiPenaltyGcvCudaPreparedsmall64>& p,
    int targets,double* coefficients,double* fitted,double* residuals,cudaEvent_t event){
  need(p&&coefficients&&fitted&&residuals&&event,"native retention arguments");
  std::lock_guard<std::mutex> lock(p->mutex);
  need(p->n==351&&p->q==37&&p->penalty_count==4&&targets>=2&&targets<=p->target_capacity&&
       p->residual_slot_leased,"native retention shape/lease");
  check_cuda(cudaSetDevice(p->device_id),"native retention device");
  check_cuda(cudaMemcpyAsync(coefficients,p->d_coefficients.get(),size_t(37)*targets*sizeof(double),
    cudaMemcpyDeviceToDevice,p->stream.get()),"native coefficients retention");
  check_cuda(cudaMemcpyAsync(fitted,p->d_fitted.get(),size_t(351)*targets*sizeof(double),
    cudaMemcpyDeviceToDevice,p->stream.get()),"native fitted retention");
  check_cuda(cudaMemcpyAsync(residuals,p->d_residuals.get(),size_t(351)*targets*sizeof(double),
    cudaMemcpyDeviceToDevice,p->stream.get()),"native residual retention");
  check_cuda(cudaEventRecord(event,p->stream.get()),"native retention event");
  check_cuda(cudaEventSynchronize(event),"native retention completion");
}
}
