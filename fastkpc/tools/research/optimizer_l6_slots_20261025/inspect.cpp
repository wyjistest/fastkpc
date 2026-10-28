#include "fixture.hpp"
#include <map>
#include <numeric>
#include <iostream>
int main(int argc,char** argv){try{
  window_diag::need(argc==2,"inspect RAW");
  for(int level:{6,8}){
    auto windows=late_diag::read_layer(argv[1],level);
    std::map<int,size_t> capacities,ranks;size_t targets=0,setups=0,iterations=0;
    for(const auto& w:windows)for(const auto& r:w.requests){
      ++setups;targets+=r.keys.size();++capacities[r.keys.size()];
      ++ranks[std::accumulate(r.geometry.ranks.begin(),r.geometry.ranks.end(),0)];
      window_diag::need(r.expected.floating.size()==14&&r.expected.integral.size()==19,"snapshot width");
      iterations+=std::accumulate(r.expected.integral[3].begin(),r.expected.integral[3].end(),size_t(0));
    }
    std::cout<<"L"<<level<<" windows="<<windows.size()<<" setups="<<setups<<" targets="<<targets<<" iterations="<<iterations<<" capacities=";
    for(auto [n,c]:capacities)std::cout<<n<<":"<<c<<",";
    std::cout<<" total_root_rank=";for(auto [n,c]:ranks)std::cout<<n<<":"<<c<<",";
    std::cout<<std::endl;
  }return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
