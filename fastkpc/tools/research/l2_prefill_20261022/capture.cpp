// Observe the actual L2 producer/consumer without changing native numerics.
// Explicit diagnostic shadows and serialization invalidate capture timings.
#include "full_cuda_ci_native_setup.hpp"
#include "skeleton_task_scheduler.hpp"
#include "cuda/full_cuda_ci_vertical.hpp"
#include "cuda/mgcv_multi_penalty_gcv_capacity.hpp"
#include "cuda/mgcv_single_penalty_gcv.hpp"
#include <cublas_v2.h>
#include <dlfcn.h>
#include <openssl/sha.h>
#include <atomic>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include "symbols.inc"

namespace {
std::atomic<int> current_level{-1};
std::mutex mutex;
unsigned prefills=0, solves=0, groups=0;
void need(bool b,const char* s) { if(!b) throw std::runtime_error(s); }
const char* env(const char* k) { auto p=std::getenv(k);need(p&&*p,k);return p; }
template<class F> F original(const char* s) {
  void* h=dlopen(env("FASTKPC_L2_NATIVE"),RTLD_NOW|RTLD_NOLOAD);need(h,"native not loaded");
  auto f=reinterpret_cast<F>(dlsym(h,s));dlclose(h);need(f,s);return f;
}
std::string path(const std::string& s) {return std::string(env("FASTKPC_L2_CAPTURE"))+"/"+s;}
std::string hash(const double* x,size_t n) {
  unsigned char h[32];SHA256(reinterpret_cast<const unsigned char*>(x),n*sizeof(double),h);
  std::ostringstream o;o<<std::hex<<std::setfill('0');for(auto b:h)o<<std::setw(2)<<int(b);return o.str();
}
template<class T> void array(std::ostream& o,const std::vector<T>& x) {
  o<<'[';for(size_t i=0;i<x.size();++i){if(i)o<<',';o<<x[i];}o<<']';
}
void strings(std::ostream& o,const std::vector<std::string>& x) {
  o<<'[';for(size_t i=0;i<x.size();++i){if(i)o<<',';o<<'"'<<x[i]<<'"';}o<<']';
}
void binary(const std::string& p,const std::vector<std::vector<double>>& arrays) {
  std::ofstream o(p,std::ios::binary);for(auto& a:arrays)o.write(reinterpret_cast<const char*>(a.data()),a.size()*8);need(o.good(),"capture write");
}
template<class R> std::vector<double> values(const R& x) {return {x.begin(),x.end()};}
std::vector<double> shadow(const double* p,size_t n) {
  std::vector<double> x(n);auto e=cudaMemcpy(x.data(),p,n*8,cudaMemcpyDeviceToHost);need(e==cudaSuccess,cudaGetErrorString(e));return x;
}
std::map<std::string,std::vector<int>> conditioning;
std::map<std::string,std::string> geometry_digest;
std::unordered_map<const void*,std::string> prepared_geometry;
const std::vector<double>& dataset() {
  static auto x=[] {std::vector<double> a(351*48);std::ifstream f(path("dataset.f64"),std::ios::binary);f.read(reinterpret_cast<char*>(a.data()),a.size()*8);need(f.gcount()==a.size()*8,"dataset capture input");return a;}();return x;
}
struct FitScope {int n,q,t,calls=0;const double* fitted=nullptr;};
thread_local FitScope* fit_scope=nullptr;
}

LayerPlan make_layer_plan(const std::vector<int>& a,int p,int l) {
  static auto f=original<decltype(&make_layer_plan)>(SYM_plan);
  current_level=l;return f(a,p,l);
}
extern "C" cublasStatus_t cublasDgemm_v2(cublasHandle_t h,cublasOperation_t ta,cublasOperation_t tb,
    int m,int n,int k,const double* alpha,const double* a,int lda,const double* b,int ldb,
    const double* beta,double* c,int ldc) {
  static auto f=[] {void* h=dlopen("libcublas.so.12",RTLD_NOW|RTLD_NOLOAD);need(h,"cuBLAS not loaded");auto f=reinterpret_cast<decltype(&cublasDgemm_v2)>(dlsym(h,"cublasDgemm_v2"));dlclose(h);need(f,"GEMM symbol");return f;}();
  auto s=f(h,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
  if(fit_scope&&s==CUBLAS_STATUS_SUCCESS){need(ta==CUBLAS_OP_N&&tb==CUBLAS_OP_N&&m==fit_scope->n&&n==fit_scope->t&&k==fit_scope->q,"fixed GEMM shape");fit_scope->fitted=c;++fit_scope->calls;}
  return s;
}
namespace fastkpc {
Rcpp::List full_cuda_ci_native_setup(const Rcpp::NumericMatrix& x,NativeSetupProfile* p) {
  static auto f=original<decltype(&full_cuda_ci_native_setup)>(SYM_setup);auto r=f(x,p);
  if(current_level!=2)return r;
  need(x.nrow()==351&&x.ncol()==2,"L2 setup shape");std::vector<int> s;
  for(int j=0;j<2;++j){int found=-1;for(int k=0;k<48;++k)if(std::memcmp(x.begin()+351*j,dataset().data()+351*k,351*8)==0){need(found<0,"ambiguous dataset column");found=k+1;}need(found>=0,"conditioning source missing");s.push_back(found);}
  Rcpp::NumericMatrix X=r["X"];std::lock_guard<std::mutex> lock(mutex);
  auto id=hash(X.begin(),X.size());auto it=conditioning.find(id);need(it==conditioning.end()||it->second==s,"geometry collision");conditioning[id]=s;return r;
}
Rcpp::List full_cuda_ci_native_geometry_prepare(const Rcpp::NumericMatrix& X,const Rcpp::List& b,
    const Rcpp::IntegerVector& o,const Rcpp::IntegerVector& ranks,NativeSetupProfile* p) {
  static auto f=original<decltype(&full_cuda_ci_native_geometry_prepare)>(SYM_geometry);auto g=f(X,b,o,ranks,p);
  if(current_level!=2)return g;
  need(b.size()==1,"L2 penalty count");auto id=hash(X.begin(),X.size());std::lock_guard<std::mutex> lock(mutex);
  if(geometry_digest.count(id))return g;
  Rcpp::NumericMatrix qr=g["magic_qr_packed"],R=g["magic_r"];
  Rcpp::NumericVector tau=g["magic_tau"],sp=g["initial_log_sp"];
  Rcpp::IntegerVector pivot=g["magic_pivot"];
  Rcpp::List roots=g["penalty_roots"],matrices=g["penalty_matrices"];
  Rcpp::NumericMatrix root=roots[0],matrix=matrices[0];
  binary(path("geometry-"+id+".f64"),{values(X),values(qr),values(tau),values(R),values(root),values(matrix),values(sp)});
  std::ofstream out(path("geometry-"+id+".json"));out<<"{\"id\":\""<<id<<"\",\"n\":"<<X.nrow()<<",\"q\":"<<X.ncol()<<",\"rank\":"<<ranks[0]<<",\"S\":";array(out,conditioning.at(id));out<<",\"pivot\":";std::vector<int> piv(pivot.begin(),pivot.end());for(auto& i:piv)--i;array(out,piv);out<<"}\n";need(out.good(),"geometry metadata");geometry_digest[id]=id;return g;
}
SinglePenaltyGcvCudaMultiResult single_penalty_gcv_cuda_multi(const std::vector<SinglePenaltyGcvCudaOwnedInput>& inputs,int concurrency) {
  static auto f=original<decltype(&single_penalty_gcv_cuda_multi)>(SYM_single);auto r=f(inputs,concurrency);
  if(current_level!=2)return r;need(r.setups.size()==inputs.size(),"single capture coverage");std::lock_guard<std::mutex> lock(mutex);
  for(size_t j=0;j<inputs.size();++j){const auto& a=inputs[j];const auto& result=r.setups[j];auto id=hash(a.X.data(),a.X.size());need(geometry_digest.count(id),"prefill geometry missing");std::vector<double> sp,log;for(auto& t:result.targets){sp.push_back(t.sp);log.push_back(t.log_sp);}std::string name="prefill-"+std::to_string(prefills++);
    binary(path(name+".f64"),{a.Y,sp,log});std::ofstream out(path(name+".json"));out<<"{\"geometry\":\""<<id<<"\",\"n\":"<<a.n<<",\"q\":"<<a.coefficient_dim<<",\"targets\":";array(out,a.target_ids);out<<"}\n";need(out.good(),"prefill metadata");}
  return r;
}
std::shared_ptr<MultiPenaltyGcvCapacityPrepared> create_multi_penalty_gcv_capacity_prepared(
    const double* X,const double* qr,const double* tau,const double* R,const int* pivot,
    const std::vector<std::vector<double>>& roots,const std::vector<std::vector<double>>& matrices,
    const std::vector<int>& ranks,const double* sp,int n,int q,int penalties,int capacity,int device) {
  static auto f=original<decltype(&create_multi_penalty_gcv_capacity_prepared)>(SYM_create);
  auto r=f(X,qr,tau,R,pivot,roots,matrices,ranks,sp,n,q,penalties,capacity,device);
  if(current_level==2){auto id=hash(X,size_t(n)*q);std::lock_guard<std::mutex> lock(mutex);need(geometry_digest.count(id)&&penalties==1,"fixed geometry identity");prepared_geometry[r.get()]=id;}
  return r;
}
std::shared_ptr<MultiPenaltyGcvCapacityResidualBatch> multi_penalty_gcv_capacity_evaluate_fixed_batch(
    const std::shared_ptr<MultiPenaltyGcvCapacityPrepared>& prepared,const double* Y,const double* log,
    int n,int count,const std::vector<std::string>& keys,double tolerance) {
  static auto f=original<decltype(&multi_penalty_gcv_capacity_evaluate_fixed_batch)>(SYM_fixed);
  if(current_level!=2)return f(prepared,Y,log,n,count,keys,tolerance);
  static auto pi=original<decltype(&multi_penalty_gcv_capacity_prepared_info)>(SYM_prepared_info);
  static auto acquire=original<decltype(&acquire_multi_penalty_gcv_capacity_residual_consumer_view)>(SYM_acquire);
  const int q=pi(prepared).coefficient_dim;FitScope scope{n,q,count};need(!fit_scope,"nested fixed capture");fit_scope=&scope;
  std::shared_ptr<MultiPenaltyGcvCapacityResidualBatch> r;
  try{r=f(prepared,Y,log,n,count,keys,tolerance);}catch(...){fit_scope=nullptr;throw;}fit_scope=nullptr;
  need(scope.calls==1&&scope.fitted,"fixed GEMM capture missing");auto view=acquire(r);
  auto e=cudaEventSynchronize(view.producer_completion_event);need(e==cudaSuccess,cudaGetErrorString(e));
  std::lock_guard<std::mutex> lock(mutex);std::string name="solve-"+std::to_string(solves++);
  binary(path(name+".f64"),{{Y,Y+size_t(n)*count},{log,log+count},shadow(view.coefficients,size_t(q)*count),shadow(scope.fitted,size_t(n)*count),shadow(view.residuals,size_t(n)*count)});
  std::ofstream out(path(name+".json"));out<<"{\"geometry\":\""<<prepared_geometry.at(prepared.get())<<"\",\"n\":"<<n<<",\"q\":"<<q<<",\"count\":"<<count<<",\"keys\":";strings(out,keys);out<<",\"info\":";array(out,view.solver_info);out<<",\"route\":";array(out,view.decomposition_route);out<<",\"rank\":";array(out,view.numerical_rank);out<<"}\n";need(out.good(),"fixed metadata");return r;
}
FullCudaCiLegacyEigBatchResult run_full_cuda_ci_phase35_legacy_eig_batch_impl(
    const std::shared_ptr<PreparedSGpuHandle>& prepared,const FixedSpBatchHostView& batch,
    const FullCudaCiLegacyEigBatchRequest& request,const MultiPenaltyGcvCudaResidualConsumerView* residual,
    const char* semantic,const std::shared_ptr<FullCudaCiLegacyEigComponentCache>& cache) {
  static auto f=original<decltype(&run_full_cuda_ci_phase35_legacy_eig_batch_impl)>(SYM_consumer);
  auto r=f(prepared,batch,request,residual,semantic,cache);if(current_level!=2)return r;
  std::lock_guard<std::mutex> lock(mutex);static std::ofstream out(path("groups.jsonl"));
  out<<std::setprecision(17)<<"{\"group\":"<<groups++<<",\"solve_count\":"<<solves<<",\"prepared\":\""<<request.expected_prepared_s_key_sha256<<"\",\"keys\":";strings(out,batch.target_keys);out<<",\"gram\":";strings(out,r.diagnostics.gram_component_keys);out<<",\"pairs\":[";
  for(size_t j=0;j<request.pairs.size();++j){if(j)out<<',';auto& p=request.pairs[j];auto& v=r.records.at(j);out<<'['<<p.logical_sequence_id<<','<<p.left_target_index<<','<<p.right_target_index<<','<<v.p_value<<']';}
  out<<"]}\n";out.flush();need(out.good(),"consumer capture");return r;
}
}
