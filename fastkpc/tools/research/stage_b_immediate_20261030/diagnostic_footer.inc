namespace {
__global__ void ip_direct_kernel(int n,EntryConfig cfg,const double* input,
    Rotation* rotations,WorkPairState* state,double* output,CheckpointCursor checkpoint,int helper) {
  extern __shared__ double shared[];
  double* d=shared;double* e=d+n;double* q=e+n;
  double* work=IP_CANDIDATE ? nullptr : q+n;
  double* guard=shared+IP_FACTOR*n;
  if(threadIdx.x)return;
  for(int j=0;j<n;++j){d[j]=input[j];e[j]=input[n+j];q[j]=input[2*n+j];}
  for(int j=0;j<8;++j)guard[j]=0x1.2345p+9;
  RotationLog log{rotations,q,0,cfg.capacity,0};
  log.force_nonconvergence=cfg.force;log.checkpoint=checkpoint;
  int info=0;double dummy[4]={};
  if(helper) {
    double c=input[0],s=input[1];int one=1,two=2;
    if(helper==1) {
#if IP_CANDIDATE
      fastkpc::lapack312::ip_consume_sweep(&log,q,q+1,c,s);
#else
      fastkpc::lapack312::record_dlasr_(&log,const_cast<char*>("L"),const_cast<char*>("V"),
        const_cast<char*>("F"),&two,&one,&c,&s,q,&two,1,1,1);
#endif
    } else {
      fastkpc::lapack312::record_drot_(&log,&one,q,&one,q+1,&one,&c,&s);
    }
  } else {
    fastkpc::lapack312::recorded_dbdsqr_fixed_vt_(&cfg.uplo,&cfg.r,&cfg.ncvt,&cfg.nru,&cfg.ncc,
        d,e,q,&cfg.ldvt,dummy+2,&cfg.ldu,dummy+2,&cfg.ldc,work,&info,1,&log);
  }
  fastkpc::lapack312::ip_mark(&log,7,cfg.r<0?0:cfg.r,d,e,q,info);
  *state={info,log.count,log.overflow,int(log.coverage),log.sweeps,0,cfg.r,0};
  for(int j=0;j<8;++j)if(guard[j]!=0x1.2345p+9)state->canary=1;
  for(int j=0;j<cfg.r;++j){output[j]=d[j];output[n+j]=e[j];output[2*n+j]=q[j];}
}
__global__ void ip_corrupt_kernel(CheckpointHeader* h,unsigned long long* v,int numeric) {
  if(numeric) v[0]^=1ull; else h[0].ordinal^=1;
}
}
extern "C" void* wp_create(int n,int k,int count,int capacity,cudaStream_t stream,WpAllocate alloc,void* owner) {
  return fastkpc::create_structured_component_plan(n,k,count,capacity,stream,alloc,owner);
}
extern "C" void wp_destroy(void* p) {
  fastkpc::destroy_structured_component_plan(static_cast<fastkpc::StructuredComponentPlan*>(p));
}
extern "C" WorkPairView wp_get(void* ptr) {
  auto* p=static_cast<fastkpc::StructuredComponentPlan*>(ptr);
  return {p->deq,p->diag,p->rotations,p->data,p->ints,p->meta,p->candidates,p->evals,p->vectors,p->means,p->status,
    p->n,p->k,p->count,p->capacity,int(sizeof(Meta)),int(sizeof(Candidate)),
    int(sizeof(fastkpc::StructuredComponentStatus)),int(sizeof(Rotation)),p->cp_results};
}
extern "C" void ip_configure(void* ptr,CheckpointHeader* headers,unsigned long long* values,
    const CheckpointResult* reference,int capacity,int compare) {
  auto* p=static_cast<fastkpc::StructuredComponentPlan*>(ptr);
  if(!headers||!values||capacity<1||(compare&&!reference))throw std::runtime_error("invalid checkpoint configuration");
  p->cp_headers=headers;p->cp_values=values;p->cp_reference=reference;p->cp_capacity=capacity;p->cp_compare=compare;
}
extern "C" WorkPairView wp_submit(void* ptr,const double* input,int count) {
  auto* p=static_cast<fastkpc::StructuredComponentPlan*>(ptr);
  if(!p->cp_headers)throw std::runtime_error("checkpoint configuration missing");
  structured_check(cudaMemsetAsync(p->diag,0,p->count*sizeof(WorkPairState),p->stream));
  structured_check(cudaMemsetAsync(p->cp_results,0,p->count*sizeof(CheckpointResult),p->stream));
  fastkpc::submit_structured_components(p,input,count);
  return wp_get(ptr);
}
extern "C" void ip_entry(void* ptr,EntryConfig cfg,const double* input) {
  auto* p=static_cast<fastkpc::StructuredComponentPlan*>(ptr);
  if(!p->cp_headers||cfg.r>p->n||cfg.capacity>p->capacity)throw std::runtime_error("invalid direct diagnostic input");
  structured_check(cudaMemsetAsync(p->cp_results,0,p->count*sizeof(CheckpointResult),p->stream));
  CheckpointCursor cursor{p->cp_headers,p->cp_values,p->cp_results,p->cp_reference,p->cp_capacity,p->n,p->cp_compare};
  ip_direct_kernel<<<1,32,(IP_FACTOR*p->n+8)*sizeof(double),p->stream>>>(p->n,cfg,input,
    p->rotations,p->diag,p->deq,cursor,0);
  structured_check(cudaGetLastError());
}
extern "C" void wp_direct(void* ptr,int r,int capacity,int force,const double* input) {
  ip_entry(ptr,{r,1,0,0,std::max(1,r),1,1,capacity,force,'L'},input);
}
extern "C" void ip_rotation(void* ptr,int helper,int capacity,int force,const double* input) {
  auto* p=static_cast<fastkpc::StructuredComponentPlan*>(ptr);
  structured_check(cudaMemsetAsync(p->cp_results,0,p->count*sizeof(CheckpointResult),p->stream));
  CheckpointCursor cursor{p->cp_headers,p->cp_values,p->cp_results,p->cp_reference,p->cp_capacity,p->n,p->cp_compare};
  ip_direct_kernel<<<1,32,(IP_FACTOR*p->n+8)*sizeof(double),p->stream>>>(p->n,
    EntryConfig{2,1,0,0,2,1,1,capacity,force,'L'},input,p->rotations,p->diag,p->deq,cursor,helper);
  structured_check(cudaGetLastError());
}
extern "C" void ip_corrupt_trace(void* ptr,int numeric) {
  auto* p=static_cast<fastkpc::StructuredComponentPlan*>(ptr);
  ip_corrupt_kernel<<<1,1,0,p->stream>>>(p->cp_headers,p->cp_values,numeric);
  structured_check(cudaGetLastError());
}
