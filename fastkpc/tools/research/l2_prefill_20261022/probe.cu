// Isolated GPU replay. Includes the unchanged production component lifecycle.
#include "full_cuda_ci_vertical.cu"
#include "mgcv_multi_penalty_gcv_capacity.hpp"
#include <cstdio>
#include <cstring>
#include <chrono>

namespace fastkpc {
namespace {
void require_probe(bool b,const char* s){if(!b)throw std::runtime_error(s);}
std::vector<std::string> decode(const char* keys,int n) {
  std::vector<std::string> out;for(int i=0;i<n;++i)out.emplace_back(keys+size_t(i)*64,64);return out;
}
struct Arena {
  VerticalCallAccounting accounting;
  TrackedDeviceBuffer residual;
  TrackedCudaStream stream;
  TrackedCudaEvent ready;
  int n,count;
  Arena(int rows,int targets,const double* input):residual(size_t(rows)*targets*8,&accounting),n(rows),count(targets) {
    if(input)check_cuda(cudaMemcpyAsync(residual.get(),input,size_t(n)*count*8,cudaMemcpyHostToDevice,stream.get()),"replay fixture upload");
    check_cuda(cudaEventRecord(ready.get(),stream.get()),"arena ready");
    check_cuda(cudaEventSynchronize(ready.get()),"arena upload complete");
  }
};
struct Fixed {
  std::shared_ptr<MultiPenaltyGcvCapacityPrepared> prepared;
  int n,q;
  Fixed(int rows,int cols,int rank,int capacity,const double* data,const int* pivot):n(rows),q(cols) {
    const double* X=data;const double* qr=X+size_t(n)*q;const double* tau=qr+size_t(n)*q;
    const double* R=tau+q;const double* root=R+size_t(q)*q;
    const double* matrix=root+size_t(q)*rank;const double* initial=matrix+size_t(q)*q;
    prepared=create_multi_penalty_gcv_capacity_prepared(X,qr,tau,R,pivot,
      {{root,root+size_t(q)*rank}},{{matrix,matrix+size_t(q)*q}},{rank},initial,n,q,1,capacity,0);
  }
  void evaluate(Arena* arena,int offset,int count,const double* Y,const double* sp,
      const char* keys,double* output,int* metadata,double* timing) {
    const auto started=std::chrono::steady_clock::now();
    auto r=multi_penalty_gcv_capacity_evaluate_fixed_batch(prepared,Y,sp,n,count,decode(keys,count),std::sqrt(std::numeric_limits<double>::epsilon()));
    timing[0]=host_elapsed_ms(started);
    try {
      const auto v=acquire_multi_penalty_gcv_capacity_residual_consumer_view(r);
      require_probe(v.fixed_sp_evaluation&&v.target_count==count,"fixed replay identity");
      for(int i=0;i<count;++i){require_probe(v.solver_info[i]==0&&v.decomposition_route[i]==kMultiPenaltyGcvDecompositionStableSvd&&v.numerical_rank[i]>0,"fixed replay authority");if(metadata){metadata[i]=v.solver_info[i];metadata[count+i]=v.decomposition_route[i];metadata[2*count+i]=v.numerical_rank[i];}}
      const auto detach=std::chrono::steady_clock::now();
      if(arena){require_probe(arena->n==n&&offset>=0&&offset+count<=arena->count,"arena range");
        check_cuda(cudaStreamWaitEvent(arena->stream.get(),v.producer_completion_event,0),"detach producer dependency");
        check_cuda(cudaMemcpyAsync(static_cast<double*>(arena->residual.get())+size_t(offset)*n,v.residuals,size_t(n)*count*8,cudaMemcpyDeviceToDevice,arena->stream.get()),"detach fixed residual");
        check_cuda(cudaEventRecord(arena->ready.get(),arena->stream.get()),"detached residual ready");
        check_cuda(cudaEventSynchronize(arena->ready.get()),"detach before workspace release");}
      timing[1]=host_elapsed_ms(detach);
      if(output){const auto shadow=materialize_multi_penalty_gcv_capacity_residual_shadow(r);
        std::memcpy(output,shadow.coefficients.data(),size_t(q)*count*8);
        std::memcpy(output+size_t(q)*count,shadow.fitted.data(),size_t(n)*count*8);
        std::memcpy(output+size_t(q+n)*count,shadow.residuals.data(),size_t(n)*count*8);}
      release_multi_penalty_gcv_capacity_residual(r);
    }catch(...){release_multi_penalty_gcv_capacity_residual(r);throw;}
  }
};
struct Pool {
  std::shared_ptr<FullCudaCiLegacyEigComponentCache> cache;
  std::vector<std::string> keys;
  int n=351,k=35,count;
  Pool(Arena* arena,const char* ids,const int* ranks,unsigned long long budget,int mode):keys(decode(ids,arena->count)),count(arena->count) {
    require_probe(arena->n==n,"component rows");
    cache=create_full_cuda_ci_legacy_eig_component_cache(n,k,budget,true,48,2304,8);
    cache->configure_structured(mode!=0);
    // Research calls the existing >=3 lifecycle with L2 fixtures. No runtime
    // shape/level gate is changed or claimed qualified by this probe.
    cache->begin_level(3,count);
    for(int off=0;off<count;off+=16){const int size=std::min(16,count-off);MultiPenaltyGcvCudaResidualConsumerView v;
      v.residuals=static_cast<const double*>(arena->residual.get())+size_t(off)*n;
      v.n=n;v.target_count=size;v.device_id=0;v.fixed_sp_evaluation=true;
      v.producer_stream=arena->stream.get();v.producer_completion_event=arena->ready.get();
      v.target_keys.assign(keys.begin()+off,keys.begin()+off+size);v.solver_info.assign(size,0);
      v.decomposition_route.assign(size,kMultiPenaltyGcvDecompositionStableSvd);
      v.numerical_rank.assign(ranks+off,ranks+off+size);cache->store_residual(v);}
  }
  void build(double* stats) {
    cache->build_level();const auto s=cache->info().level_prefill.back();
    const double a[]={s.prefill_host_ms,double(s.produced),double(s.structured_accepted),double(s.dense_fallback),
      double(s.fallback_info),double(s.fallback_rotation),double(s.fallback_gap),double(s.rotations_max),
      s.structured_minimum_gap,double(s.structured_pool_bytes),double(s.sampled_device_used_bytes),
      double(s.structured_peak_inflight),double(s.structured_waves),double(s.dense_fallback_batches),
      s.structured_create_host_ms,s.eig_cuda_ms};std::copy(a,a+16,stats);
  }
  void resolve(const int* ids,int size) {
    std::vector<std::string> group;for(int i=0;i<size;++i){require_probe(ids[i]>=0&&ids[i]<count,"Gram identity");group.push_back(keys[ids[i]]);}cache->resolve_gram(group);
  }
  void pairs(const int* left,const int* right,const unsigned long long* ids,int pair_count,void* output) {
    require_probe(pair_count>0&&pair_count<=2304,"pair replay capacity");
    auto* execution_context=cache->execution_context();auto stream=execution_context->stream();
    const int num_col=k;const size_t pair_size=pair_count,cross_stride=size_t(k)*k;
    FixedSpBatchHostView batch;batch.n=n;std::vector<std::uintptr_t> metadata(10*pair_size);
    for(int j=0;j<pair_count;++j){require_probe(left[j]>=0&&left[j]<count&&right[j]>=0&&right[j]<count,"pair identity");
      std::size_t a,b;FixedSpRoute route;FixedSpStatus status;
      require_probe(cache->lookup(keys[left[j]],&a,&route,&status)&&cache->lookup(keys[right[j]],&b,&route,&status)&&
        !cache->gram_pending(keys[left[j]])&&!cache->gram_pending(keys[right[j]]),"unresolved component");
      metadata[j]=ids[j];metadata[pair_size+j]=reinterpret_cast<std::uintptr_t>(cache->vectors(a));
      metadata[2*pair_size+j]=reinterpret_cast<std::uintptr_t>(cache->vectors(b));
      metadata[3*pair_size+j]=reinterpret_cast<std::uintptr_t>(execution_context->pair_cross()+j*cross_stride);
      metadata[4*pair_size+j]=reinterpret_cast<std::uintptr_t>(cache->selected_values(a));
      metadata[5*pair_size+j]=reinterpret_cast<std::uintptr_t>(cache->selected_values(b));
      metadata[6*pair_size+j]=reinterpret_cast<std::uintptr_t>(cache->total(a));
      metadata[7*pair_size+j]=reinterpret_cast<std::uintptr_t>(cache->total(b));
      metadata[8*pair_size+j]=reinterpret_cast<std::uintptr_t>(cache->self_moment(a));
      metadata[9*pair_size+j]=reinterpret_cast<std::uintptr_t>(cache->self_moment(b));}
    check_cuda(cudaMemcpyAsync(execution_context->pair_metadata(),metadata.data(),metadata.size()*sizeof(std::uintptr_t),cudaMemcpyHostToDevice,stream),"original pair metadata");
    const double one=1.0,zero=0.0;
#include "pair_stage.inc"
    check_cuda(cudaMemcpyAsync(output,execution_context->device_results(),pair_size*sizeof(DeviceCompactCiResult),cudaMemcpyDeviceToHost,stream),"compact pair result");
    check_cuda(cudaEventRecord(execution_context->completion(),stream),"pair ready");
    check_cuda(cudaEventSynchronize(execution_context->completion()),"compact join");
    const auto* r=static_cast<const DeviceCompactCiResult*>(output);for(int j=0;j<pair_count;++j)require_probe(r[j].status==kDcovStatusOk,"pair/gamma failed closed");
  }
};
}
}
static thread_local char error[1024];
template<class F> int protect(F f){try{f();error[0]=0;return 0;}catch(const std::exception& e){std::snprintf(error,sizeof(error),"%s",e.what());return 1;}}
extern "C" const char* l2_error(){return error;}
extern "C" void* l2_arena(int n,int count,const double* input){void* p=nullptr;protect([&]{p=new fastkpc::Arena(n,count,input);});return p;}
extern "C" void l2_arena_free(void* p){delete static_cast<fastkpc::Arena*>(p);}
extern "C" void* l2_fixed(int n,int q,int rank,int capacity,const double* data,const int* pivot,double* ms){void* p=nullptr;protect([&]{auto s=std::chrono::steady_clock::now();p=new fastkpc::Fixed(n,q,rank,capacity,data,pivot);*ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-s).count();});return p;}
extern "C" void l2_fixed_free(void* p){delete static_cast<fastkpc::Fixed*>(p);}
extern "C" int l2_evaluate(void* p,void* arena,int offset,int count,const double* Y,const double* sp,const char* keys,double* output,int* meta,double* timing){return protect([&]{static_cast<fastkpc::Fixed*>(p)->evaluate(static_cast<fastkpc::Arena*>(arena),offset,count,Y,sp,keys,output,meta,timing);});}
extern "C" void* l2_pool(void* arena,const char* keys,const int* ranks,unsigned long long budget,int mode){void* p=nullptr;protect([&]{p=new fastkpc::Pool(static_cast<fastkpc::Arena*>(arena),keys,ranks,budget,mode);});return p;}
extern "C" int l2_build(void* p,double* stats){return protect([&]{static_cast<fastkpc::Pool*>(p)->build(stats);});}
extern "C" int l2_resolve(void* p,const int* ids,int count){return protect([&]{static_cast<fastkpc::Pool*>(p)->resolve(ids,count);});}
extern "C" int l2_pairs(void* p,const int* left,const int* right,const unsigned long long* ids,int count,void* output){return protect([&]{static_cast<fastkpc::Pool*>(p)->pairs(left,right,ids,count,output);});}
extern "C" int l2_finish(void* p){return protect([&]{static_cast<fastkpc::Pool*>(p)->cache->finish_level();});}
extern "C" void l2_pool_free(void* p){delete static_cast<fastkpc::Pool*>(p);}
extern "C" int l2_snapshot(long long* out){return protect([&]{auto a=fastkpc::full_cuda_ci_vertical_resource_snapshot();out[0]=a.live_device_allocations;out[1]=a.live_device_bytes;out[2]=a.live_streams;out[3]=a.live_events;});}
