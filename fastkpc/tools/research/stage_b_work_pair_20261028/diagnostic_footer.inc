namespace {
__global__ void wp_direct_kernel(int n,int r,int capacity,int force,const double* input,
    Rotation* rotations,WorkPairState* state,double* output) {
  extern __shared__ double shared[];
  double* d=shared;double* e=d+n;double* q=e+n;double* work=q+n;
  if(threadIdx.x)return;
  for(int j=0;j<n;++j){d[j]=input[j];e[j]=input[n+j];q[j]=input[2*n+j];}
  for(int j=0;j<8;++j)work[WP_WORK_FACTOR*n+j]=0x1.2345p+9;
  RotationLog log{rotations,q,0,capacity,0};log.force_nonconvergence=force;
  int one=1,zero=0,info=0;double dummy[4]={};
  fastkpc::lapack312::recorded_dbdsqr_(const_cast<char*>("L"),&r,&one,&zero,&zero,
      d,e,q,&r,dummy+2,&one,dummy+2,&one,work,&info,1,&log);
  *state={info,log.count,log.overflow,int(log.coverage),log.sweeps,0,r,0};
  for(int j=0;j<8;++j)if(work[WP_WORK_FACTOR*n+j]!=0x1.2345p+9)state->canary=1;
  for(int j=0;j<r;++j){output[j]=d[j];output[n+j]=e[j];output[2*n+j]=q[j];}
}
}
extern "C" void* wp_create(int n,int k,int count,int capacity,cudaStream_t stream,WpAllocate alloc,void* owner) {
  return fastkpc::create_structured_component_plan(n,k,count,capacity,stream,alloc,owner);
}
extern "C" void wp_destroy(void* p) {fastkpc::destroy_structured_component_plan(static_cast<fastkpc::StructuredComponentPlan*>(p));}
extern "C" WorkPairView wp_get(void* ptr) {
  auto* p=static_cast<fastkpc::StructuredComponentPlan*>(ptr);
  return {p->deq,p->diag,p->rotations,p->data,p->ints,p->meta,p->candidates,p->evals,p->vectors,p->means,p->status,
    p->n,p->k,p->count,p->capacity,int(sizeof(Meta)),int(sizeof(Candidate)),int(sizeof(fastkpc::StructuredComponentStatus)),int(sizeof(Rotation))};
}
extern "C" WorkPairView wp_submit(void* ptr,const double* input,int count) {
  auto* p=static_cast<fastkpc::StructuredComponentPlan*>(ptr);
  structured_check(cudaMemsetAsync(p->diag,0,p->count*sizeof(WorkPairState),p->stream));
  fastkpc::submit_structured_components(p,input,count);
  return wp_get(ptr);
}
extern "C" void wp_direct(void* ptr,int r,int capacity,int force,const double* input) {
  auto* p=static_cast<fastkpc::StructuredComponentPlan*>(ptr);
  wp_direct_kernel<<<1,32,((3+WP_WORK_FACTOR)*p->n+8)*sizeof(double),p->stream>>>(
    p->n,r,capacity,force,input,p->rotations,p->diag,p->deq);
  structured_check(cudaGetLastError());
}
