// Diagnostic-only late-failure injection. No timing library includes this file.
namespace {
__global__ void rt_inject_nonfinite(double* data,int n) {
  data[4*n]=CUDART_NAN; // workspace.ld[0], component zero
}
}
extern "C" void rt_recover_fault(void* ptr) {
  auto* p=static_cast<fastkpc::StructuredComponentPlan*>(ptr);
  rt_inject_nonfinite<<<1,1,0,p->stream>>>(p->data,p->n);
  recover<<<dim3(1,(p->k+p->tile-1)/p->tile),64,p->tile*p->n*sizeof(double),p->stream>>>(
      p->n,p->k,p->data,p->ints,p->meta,p->rotations,p->capacity,p->candidates,
      p->tile,p->evals,p->vectors,p->means);
  check_structured_boundary<<<1,128,0,p->stream>>>(p->meta,p->evals,p->k-1,1,p->status);
  structured_check(cudaGetLastError());
}
