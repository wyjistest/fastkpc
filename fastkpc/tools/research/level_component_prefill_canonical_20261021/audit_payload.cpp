// External observer for canonical off/on. All numerical work remains native CUDA.
// Hash every actually consumed multi-penalty residual and four-field payload.
// Preserve raw values for the baseline's >=17-wide first-miss targets only.
// Additional synchronization/D2H makes these diagnostic runs ineligible for timing.
#include "full_cuda_ci_vertical.hpp"
#include "mgcv_multi_penalty_gcv_capacity.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <openssl/sha.h>
#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace {
void require(bool ok,const char* why) { if(!ok) throw std::runtime_error(why); }
void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
const char* env(const char* name) { const char* p=std::getenv(name); require(p&&*p,"audit environment missing"); return p; }
bool compare_mode() { return std::string(env("FASTKPC_LSCP_CANONICAL_AUDIT_MODE"))=="compare"; }
template<class F> F symbol(const char* library,const char* name) {
  void* h=dlopen(library,RTLD_NOW|RTLD_NOLOAD); require(h,"audit library not loaded");
  auto f=reinterpret_cast<F>(dlsym(h,name)); dlclose(h); require(f,"audit symbol missing"); return f;
}
template<class F> F native(const char* name) { return symbol<F>(env("FASTKPC_LSCP_TEST_NATIVE"),name); }
std::string digest(const double* x,std::size_t count) {
  unsigned char bytes[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char*>(x),count*sizeof(double),bytes);
  std::ostringstream s; s<<std::hex<<std::setfill('0');
  for(auto b:bytes) s<<std::setw(2)<<static_cast<unsigned>(b);
  return s.str();
}
std::string digest(const std::vector<double>& x) { return digest(x.data(),x.size()); }
template<class T> std::vector<T> download(const void* p,std::size_t count) {
  std::vector<T> out(count); check(cudaMemcpy(out.data(),p,count*sizeof(T),cudaMemcpyDeviceToHost)); return out;
}
struct Residual {
  int q=0,width=0,info=0,route=0,rank=0;
  std::string y;
  std::vector<double> coefficient,fitted,residual;
};
struct Row {
  std::string key,request,prepared,y,coefficient,fitted,residual,fields[4];
  int q=0,width=0,info=0,route=0,rank=0;
};
std::unordered_map<std::string,Residual> residuals;
std::set<std::string> consumed;
const std::unordered_map<std::string,Row>& references() {
  static const auto result=[] {
    std::ifstream in(std::string(env("FASTKPC_LSCP_CANONICAL_REFERENCE"))+"/components.tsv");
    std::unordered_map<std::string,Row> result; std::string header; std::getline(in,header);
    Row r;
    while(in>>r.key>>r.request>>r.prepared>>r.q>>r.width>>r.info>>r.route>>r.rank>>r.y
            >>r.coefficient>>r.fitted>>r.residual>>r.fields[0]>>r.fields[1]>>r.fields[2]>>r.fields[3])
      require(result.emplace(r.key,r).second,"duplicate baseline component");
    require(in.eof()&&!result.empty(),"invalid baseline component manifest"); return result;
  }(); return result;
}
std::ofstream& component_log() {
  static std::ofstream out(std::string(env("FASTKPC_LSCP_CANONICAL_AUDIT_DIR"))+"/components.tsv");
  static bool first=true;
  if(first) { out<<"key\trequest\tprepared\tq\twidth\tinfo\troute\trank\ty\tcoefficient\tfitted\tresidual\tvectors\tselected_values\ttotal\tself_moment\n"; first=false; }
  return out;
}
std::ofstream& pair_log() {
  static std::ofstream out(std::string(env("FASTKPC_LSCP_CANONICAL_AUDIT_DIR"))+"/pairs.tsv");
  static bool first=true;
  if(first) { out<<"request\tprepared\tsequence\tleft\tright\tp_bits\n"; first=false; }
  return out;
}
std::ofstream& comparison_log() {
  static std::ofstream out(std::string(env("FASTKPC_LSCP_CANONICAL_AUDIT_DIR"))+"/comparison.tsv");
  static bool first=true;
  if(first) { out<<"key\tbaseline_width\tresidual_changed\tfitted_changed\tvectors_changed\tvalues_changed\ttotal_changed\tself_changed\tmax_residual_error\tmax_fitted_error\tmax_vectors_error\tmax_values_error\ttotal_error\tself_error\n"; first=false; }
  return out;
}
struct FinalScope { int n,q,count,calls=0; const double* fitted=nullptr; };
struct PairScope { int count,gemms=0; const std::uintptr_t* metadata=nullptr; };
thread_local FinalScope* final_scope=nullptr;
thread_local PairScope* pair_scope=nullptr;
double max_error(const std::vector<double>& a,const std::vector<double>& b) {
  require(a.size()==b.size(),"shadow shape changed"); double out=0;
  for(std::size_t i=0;i<a.size();++i) { require(std::isfinite(a[i])&&std::isfinite(b[i]),"nonfinite shadow"); out=std::max(out,std::abs(a[i]-b[i])); }
  return out;
}
void raw_write(const std::string& key,const Residual& r,const std::vector<double> fields[4]) {
  std::ofstream out(std::string(env("FASTKPC_LSCP_CANONICAL_AUDIT_DIR"))+"/raw/"+key+".f64",std::ios::binary);
  const std::vector<double>* parts[]={&r.coefficient,&r.fitted,&r.residual,&fields[0],&fields[1],&fields[2],&fields[3]};
  for(auto p:parts) out.write(reinterpret_cast<const char*>(p->data()),p->size()*sizeof(double));
  require(out.good(),"wide target raw write failed");
}
void audit(const Row& r,const Residual& values,const std::vector<double> fields[4]) {
  if(!compare_mode()) { if(r.width>=17) raw_write(r.key,values,fields); return; }
  const auto& expected=references().at(r.key);
  require(r.request==expected.request && r.prepared==expected.prepared && r.y==expected.y &&
    r.q==expected.q && r.info==expected.info && r.route==expected.route && r.rank==expected.rank &&
    r.coefficient==expected.coefficient,"component identity/coefficients/route/rank/info changed");
  bool changed[6]={r.residual!=expected.residual,r.fitted!=expected.fitted,
    r.fields[0]!=expected.fields[0],r.fields[1]!=expected.fields[1],
    r.fields[2]!=expected.fields[2],r.fields[3]!=expected.fields[3]};
  const bool any=std::any_of(changed,changed+6,[](bool x){return x;});
  require(!any || expected.width>=17,"unexpected change outside wide residual cohorts");
  require(!std::any_of(changed+2,changed+6,[](bool x){return x;}) || changed[0],
    "component changed despite identical residual");
  double errors[6]={};
  if(expected.width>=17) {
    std::ifstream in(std::string(env("FASTKPC_LSCP_CANONICAL_REFERENCE"))+"/raw/"+r.key+".f64",std::ios::binary);
    const std::vector<double>* actual[]={&values.coefficient,&values.fitted,&values.residual,&fields[0],&fields[1],&fields[2],&fields[3]};
    std::vector<double> reference[7];
    for(int i=0;i<7;++i) {
      reference[i].resize(actual[i]->size());
      in.read(reinterpret_cast<char*>(reference[i].data()),reference[i].size()*sizeof(double));
      require(in.gcount()==static_cast<std::streamsize>(reference[i].size()*sizeof(double)),"wide reference truncated");
    }
    require(digest(reference[0])==expected.coefficient && digest(reference[1])==expected.fitted &&
      digest(reference[2])==expected.residual,"wide residual reference hash changed");
    for(int i=0;i<4;++i) require(digest(reference[i+3])==expected.fields[i],"wide payload reference hash changed");
    require(std::memcmp(values.coefficient.data(),reference[0].data(),reference[0].size()*sizeof(double))==0,
      "wide coefficient bits changed");
    errors[0]=max_error(values.residual,reference[2]); errors[1]=max_error(values.fitted,reference[1]);
    for(int i=0;i<4;++i) errors[i+2]=max_error(fields[i],reference[i+3]);
    if(any) raw_write(r.key,values,fields);
  }
  auto& out=comparison_log(); out<<r.key<<'\t'<<expected.width;
  for(bool x:changed) out<<'\t'<<x;
  for(double x:errors) out<<'\t'<<std::setprecision(17)<<x;
  out<<'\n'; require(out.good(),"comparison log failed");
}
}

extern "C" cublasStatus_t cublasDgemm_v2(cublasHandle_t h,cublasOperation_t ta,cublasOperation_t tb,
    int m,int n,int k,const double* alpha,const double* a,int lda,const double* b,int ldb,
    const double* beta,double* c,int ldc) {
  static auto original=symbol<decltype(&cublasDgemm_v2)>("libcublas.so.12","cublasDgemm_v2");
  auto status=original(h,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
  if(final_scope && status==CUBLAS_STATUS_SUCCESS) {
    require(ta==CUBLAS_OP_N&&tb==CUBLAS_OP_N&&m==final_scope->n&&n==final_scope->count&&
      k==final_scope->q&&lda==m&&ldb==k&&ldc==m,"finalizer GEMM shape changed");
    final_scope->fitted=c; ++final_scope->calls;
  }
  return status;
}
extern "C" cublasStatus_t cublasDgemmBatched(cublasHandle_t h,cublasOperation_t ta,cublasOperation_t tb,
    int m,int n,int k,const double* alpha,const double* const a[],int lda,const double* const b[],int ldb,
    const double* beta,double* const c[],int ldc,int count) {
  static auto original=symbol<decltype(&cublasDgemmBatched)>("libcublas.so.12","cublasDgemmBatched");
  auto status=original(h,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,count);
  if(pair_scope && status==CUBLAS_STATUS_SUCCESS) {
    require(count==pair_scope->count&&m==35&&n==35&&k==351,"pair GEMM shape changed");
    pair_scope->metadata=reinterpret_cast<const std::uintptr_t*>(a)-count; ++pair_scope->gemms;
  }
  return status;
}
namespace fastkpc {
std::shared_ptr<MultiPenaltyGcvCapacityResidualBatch> multi_penalty_gcv_capacity_finalize_selected_coefficients(
    const std::shared_ptr<MultiPenaltyGcvCapacityPrepared>& prepared,const double* y,
    const MultiPenaltyGcvCudaCoefficientConsumerView& coefficients,int n,int count,
    const std::vector<std::string>& keys) {
  static auto original=native<decltype(&multi_penalty_gcv_capacity_finalize_selected_coefficients)>(
    "_ZN7fastkpc57multi_penalty_gcv_capacity_finalize_selected_coefficientsERKSt10shared_ptrINS_31MultiPenaltyGcvCapacityPreparedEEPKdRKNS_42MultiPenaltyGcvCudaCoefficientConsumerViewEiiRKSt6vectorINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaISG_EE");
  static auto acquire=native<decltype(&acquire_multi_penalty_gcv_capacity_residual_consumer_view)>(
    "_ZN7fastkpc57acquire_multi_penalty_gcv_capacity_residual_consumer_viewERKSt10shared_ptrINS_36MultiPenaltyGcvCapacityResidualBatchEE");
  FinalScope scope{n,coefficients.coefficient_dim,count}; require(!final_scope,"nested finalizer audit");
  final_scope=&scope; std::shared_ptr<MultiPenaltyGcvCapacityResidualBatch> result;
  try { result=original(prepared,y,coefficients,n,count,keys); } catch(...) {final_scope=nullptr;throw;}
  final_scope=nullptr;
  require(scope.calls==1&&scope.fitted&&n==351,"finalizer capture missing");
  const auto view=acquire(result); check(cudaEventSynchronize(view.producer_completion_event));
  for(int i=0;i<count;++i) if(!consumed.count(keys[i])) {
    Residual r; r.q=coefficients.coefficient_dim; r.width=count;
    r.info=coefficients.solver_info.at(i); r.route=coefficients.decomposition_route.at(i);
    r.rank=coefficients.numerical_rank.at(i); r.y=digest(y+std::size_t(i)*n,n);
    r.coefficient=download<double>(coefficients.coefficients+std::size_t(i)*r.q,r.q);
    r.fitted=download<double>(scope.fitted+std::size_t(i)*n,n);
    r.residual=download<double>(view.residuals+std::size_t(i)*n,n);
    residuals[keys[i]]=std::move(r);
  }
  return result;
}
FullCudaCiLegacyEigBatchResult run_full_cuda_ci_phase35_legacy_eig_batch_impl(
    const std::shared_ptr<PreparedSGpuHandle>& prepared,const FixedSpBatchHostView& batch,
    const FullCudaCiLegacyEigBatchRequest& request,const MultiPenaltyGcvCudaResidualConsumerView* residual,
    const char* semantic,const std::shared_ptr<FullCudaCiLegacyEigComponentCache>& cache) {
  static auto original=native<decltype(&run_full_cuda_ci_phase35_legacy_eig_batch_impl)>(
    "_ZN7fastkpc46run_full_cuda_ci_phase35_legacy_eig_batch_implERKSt10shared_ptrINS_18PreparedSGpuHandleEERKNS_20FixedSpBatchHostViewERKNS_31FullCudaCiLegacyEigBatchRequestEPKNS_39MultiPenaltyGcvCudaResidualConsumerViewEPKcRKS0_INS_33FullCudaCiLegacyEigComponentCacheEE");
  bool wanted=false;
  for(const auto& key:batch.target_keys) wanted=wanted||residuals.count(key)||consumed.count(key);
  if(!wanted) return original(prepared,batch,request,residual,semantic,cache);
  PairScope scope{static_cast<int>(request.pairs.size())}; require(!pair_scope,"nested pair audit");
  pair_scope=&scope; FullCudaCiLegacyEigBatchResult result;
  try {result=original(prepared,batch,request,residual,semantic,cache);} catch(...) {pair_scope=nullptr;throw;}
  pair_scope=nullptr; require(scope.gemms==1&&scope.metadata,"pair capture missing");
  const auto metadata=download<std::uintptr_t>(scope.metadata,10*scope.count);
  for(int j=0;j<scope.count;++j) {
    const auto& pair=request.pairs[j]; const int indices[]={pair.left_target_index,pair.right_target_index};
    auto& pairs=pair_log(); std::uint64_t bits=0; std::memcpy(&bits,&result.records.at(j).p_value,sizeof(bits));
    pairs<<request.request_identity_sha256<<'\t'<<request.expected_prepared_s_key_sha256<<'\t'
      <<pair.logical_sequence_id<<'\t'<<batch.target_keys.at(indices[0])<<'\t'<<batch.target_keys.at(indices[1])
      <<'\t'<<std::hex<<std::setw(16)<<std::setfill('0')<<bits<<std::dec<<'\n';
    for(int side=0;side<2;++side) {
      const auto& key=batch.target_keys.at(indices[side]); if(!consumed.insert(key).second) continue;
      const auto& r=residuals.at(key); std::vector<double> fields[4];
      const std::size_t parts[]={std::size_t(1+side),std::size_t(4+side),std::size_t(6+side),std::size_t(8+side)};
      const std::size_t lengths[]={351U*35U,35U,1U,1U}; Row row;
      row.key=key; row.request=request.request_identity_sha256; row.prepared=request.expected_prepared_s_key_sha256;
      row.q=r.q; row.width=r.width; row.info=r.info; row.route=r.route; row.rank=r.rank; row.y=r.y;
      row.coefficient=digest(r.coefficient); row.fitted=digest(r.fitted); row.residual=digest(r.residual);
      for(int f=0;f<4;++f) {fields[f]=download<double>(reinterpret_cast<const void*>(metadata[parts[f]*scope.count+j]),lengths[f]);row.fields[f]=digest(fields[f]);}
      auto& out=component_log(); out<<row.key<<'\t'<<row.request<<'\t'<<row.prepared<<'\t'<<row.q<<'\t'<<row.width
        <<'\t'<<row.info<<'\t'<<row.route<<'\t'<<row.rank<<'\t'<<row.y<<'\t'<<row.coefficient<<'\t'
        <<row.fitted<<'\t'<<row.residual;
      for(const auto& f:row.fields) out<<'\t'<<f;
      out<<'\n'; require(out.good()&&pairs.good(),"audit log failed");
      audit(row,r,fields); residuals.erase(key);
    }
  }
  return result;
}
}
