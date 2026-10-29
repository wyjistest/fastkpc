// Source-driven expression/ownership audit. No floating-point evaluation.
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

void require(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
enum Tag : unsigned char { Zero, One, MinusOne, Other };
struct Key {
  char op; uint64_t left, right;
  bool operator==(const Key& b) const { return op==b.op && left==b.left && right==b.right; }
};
struct KeyHash {
  size_t operator()(const Key& k) const {
    return (k.left*0x9e3779b97f4a7c15ULL) ^ (k.right*0x517cc1b727220a95ULL) ^ unsigned(k.op);
  }
};
struct Arena {
  std::unordered_map<Key,uint64_t,KeyHash> nodes;
  uint64_t next=4, operations=0, branches=0;
  uint64_t intern(char op,uint64_t a,uint64_t b) {
    const auto r=nodes.emplace(Key{op,a,b},next);
    if(r.second)++next;
    return r.first->second;
  }
};
Arena* arena=nullptr;
struct Symbol {
  uint64_t id=0, owners=0; Tag tag=Other;
  Symbol()=default;
  Symbol(double constant) { // Only source literals 0, 1, -1 are classified.
    require(constant==0 || constant==1 || constant==-1,"non-tag literal");
    tag=constant==0?Zero:constant==1?One:MinusOne;
    id=uint64_t(tag)+1;
  }
  static Symbol input(uint64_t label,Tag tag,uint64_t owners=0) {
    Symbol x;x.id=arena->intern('I',label,0);x.tag=tag;x.owners=owners;return x;
  }
};
Symbol operation(char op,const Symbol& a,const Symbol& b) {
  require(a.id && b.id,"undefined symbolic operand");++arena->operations;
  Symbol x;x.id=arena->intern(op,a.id,b.id);x.owners=a.owners|b.owners;
  if(op=='*') {
    if(a.tag==One)x.tag=b.tag;
    else if(b.tag==One)x.tag=a.tag;
    else if(a.tag==Zero || b.tag==Zero)x.tag=Zero;
    else if(a.tag==MinusOne && b.tag==MinusOne)x.tag=One;
  }
  return x;
}
Symbol operator+(const Symbol& a,const Symbol& b){return operation('+',a,b);}
Symbol operator-(const Symbol& a,const Symbol& b){return operation('-',a,b);}
Symbol operator*(const Symbol& a,const Symbol& b){return operation('*',a,b);}
Symbol& operator+=(Symbol& a,const Symbol& b){a=a+b;return a;}
Symbol& operator-=(Symbol& a,const Symbol& b){a=a-b;return a;}
Symbol& operator*=(Symbol& a,const Symbol& b){a=a*b;return a;}
bool operator==(const Symbol& a,const Symbol& b) {
  require(a.id && b.id,"undefined symbolic predicate");++arena->branches;
  require(a.tag!=Other || b.tag!=Other,"unexpected expression predicate");
  return a.tag==b.tag;
}
bool operator!=(const Symbol& a,const Symbol& b){return !(a==b);}

using integer=int;using logical=int;using ftnlen=int;using doublereal=Symbol;
using std::min;using std::max;
#define __device__
logical lsame_(char* a,char* b,ftnlen,ftnlen) { return std::toupper(*a)==std::toupper(*b); }
int xerbla_(char*,integer*,ftnlen){throw std::runtime_error("reference parameter rejection");}
#include "symbolic_vendor.inc"

struct Buffer {
  static constexpr int guard=128;
  std::vector<Symbol> data;
  int length;
  Buffer(int count,uint64_t label):data(count+2*guard),length(count) {
    for(int i=0;i<guard;++i){data[i]=Symbol::input(label+i,Other);data[guard+count+i]=Symbol::input(label+guard+i,Other);}
  }
  Symbol* ptr(){return data.data()+guard;}
  void guards_equal(const Buffer& old)const{
    for(int i=0;i<guard;++i)require(data[i].id==old.data[i].id && data[guard+length+i].id==old.data[guard+length+i].id,"guard changed");
  }
};
struct Shape {int id,q;char side;int m,n,k,ldv,ldc;};
Tag input_tag(uint64_t index,int pattern,bool diagonal=false) {
  if(pattern==0)return Other;
  if(pattern==1)return index%3==0?Zero:Other;
  if(diagonal)return index%2?One:Zero;
  return index%5==0?Zero:index%5==1?One:Other;
}
struct Totals {uint64_t cases=0,cells=0,operations=0,nodes=0,branches=0,owners=0;};
void one_case(const Shape& s,int pattern,Totals& totals) {
  Arena local;arena=&local;
  const bool left=s.side=='L';
  int m=s.m,n=s.n,k=s.k,ldv=s.ldv,ldc=s.ldc,ldt=65,ldwork=s.q;
  Buffer v(ldv*(left?k:n),1000000),t(ldt*k,2000000),c(ldc*n,3000000),w(ldwork*k,4000000);
  for(int j=0;j<(left?k:n);++j)for(int i=0;i<(left?m:k);++i){
    // Unit diagonal and the unused half of V1 are deliberately undefined.
    if(left?(i<k && i<=j):(j<k && i>=j))continue;
    uint64_t at=i+ldv*j;v.ptr()[at]=Symbol::input(10000+at,input_tag(at,pattern));
  }
  for(int j=0;j<k;++j)for(int i=0;i<=j;++i){uint64_t at=i+ldt*j;t.ptr()[at]=Symbol::input(20000+at,input_tag(at,pattern,i==j));}
  for(int j=0;j<n;++j)for(int i=0;i<m;++i){uint64_t at=i+ldc*j;c.ptr()[at]=Symbol::input(30000+at,Other,1ULL<<(left?j:i));}
  const auto old_v=v,old_t=t,old_c=c,old_w=w;
  auto sliced_c=c,sliced_w=w;
  char side=s.side,trans=left?'N':'T',direct='F',store=left?'C':'R';
  dlarfb_(&side,&trans,&direct,&store,&m,&n,&k,v.ptr(),&ldv,t.ptr(),&ldt,c.ptr(),&ldc,w.ptr(),&ldwork,1,1,1,1);
  // Deliberately reverse lane order; each lane still owns r=lane, lane+32.
  for(int lane=31;lane>=0;--lane)for(int r=lane;r<s.q;r+=32){
    int sm=left?m:1,sn=left?1:n;
    auto* column_or_row=sliced_c.ptr()+(left?r*ldc:r);
    dlarfb_(&side,&trans,&direct,&store,&sm,&sn,&k,v.ptr(),&ldv,t.ptr(),&ldt,column_or_row,&ldc,sliced_w.ptr()+r,&ldwork,1,1,1,1);
  }
  for(size_t i=0;i<c.data.size();++i)require(c.data[i].id==sliced_c.data[i].id,"C expression mismatch");
  for(size_t i=0;i<w.data.size();++i)require(w.data[i].id==sliced_w.data[i].id,"WORK expression mismatch");
  for(size_t i=0;i<v.data.size();++i)require(v.data[i].id==old_v.data[i].id,"V modified");
  for(size_t i=0;i<t.data.size();++i)require(t.data[i].id==old_t.data[i].id,"T modified");
  for(int j=0;j<n;++j)for(int i=0;i<m;++i){require(c.ptr()[i+ldc*j].owners==(1ULL<<(left?j:i)),"cross-vector C dependency");++totals.cells;}
  for(int j=0;j<k;++j)for(int r=0;r<s.q;++r){require(w.ptr()[r+ldwork*j].owners==(1ULL<<r),"cross-vector WORK dependency");++totals.cells;}
  c.guards_equal(old_c);w.guards_equal(old_w);sliced_c.guards_equal(old_c);sliced_w.guards_equal(old_w);
  totals.cases++;totals.operations+=local.operations;totals.nodes+=local.nodes.size();totals.branches+=local.branches;totals.owners+=s.q;
}
bool overwrite_counterexample() {
  Arena local;arena=&local;
  // R/U/N/nonunit: writing column zero first destroys an input to column one.
  std::array<Symbol,2> initial={Symbol::input(1,Other,1),Symbol::input(2,Other,1)};
  Symbol a00=Symbol::input(3,Other),a01=Symbol::input(4,Other),a11=Symbol::input(5,Other),alpha=1.;
  auto correct=initial,wrong=initial;
  correct[1]=(alpha*a11)*correct[1];correct[1]+=(alpha*a01)*correct[0];correct[0]=(alpha*a00)*correct[0];
  wrong[0]=(alpha*a00)*wrong[0];wrong[1]=(alpha*a11)*wrong[1];wrong[1]+=(alpha*a01)*wrong[0];
  return correct[1].id!=wrong[1].id;
}
int main(int argc,char** argv){try{
  require(argc==2,"symbolic SHAPES_TSV");std::ifstream input(argv[1]);require(bool(input),"shape input");
  std::string line;std::getline(input,line);Totals totals;uint64_t shapes=0;
  while(std::getline(input,line)){std::istringstream row(line);Shape s;row>>s.id>>s.q>>s.side>>s.m>>s.n>>s.k>>s.ldv>>s.ldc;require(bool(row),"shape row");
    for(int pattern=0;pattern<3;++pattern)one_case(s,pattern,totals);
    ++shapes;if(shapes%25==0)std::cerr<<"checked shapes "<<shapes<<'\n';
  }
  require(overwrite_counterexample(),"missing triangular overwrite counterexample");
  std::cout<<"{\"status\":\"PASS_SYMBOLIC_REFERENCE_DLARFB_VECTOR_SLICES\",\"shapes\":"<<shapes
    <<",\"cases\":"<<totals.cases<<",\"compared_cells\":"<<totals.cells<<",\"independent_vectors\":"<<totals.owners
    <<",\"expression_operations\":"<<totals.operations<<",\"interned_nodes\":"<<totals.nodes<<",\"predicate_checks\":"<<totals.branches
    <<",\"triangular_overwrite_counterexample\":true,\"gpu_execution\":false,\"floating_point_evaluation\":false}\n";
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
