// Exercise status layout/correlation with first and most-iterated captured setup.
#include "../optimizer_l4_diagnostic_20261023/fixture.hpp"
#include "../optimizer_layer_refill_20261023/layer_fixture.hpp"
#include <iostream>
#include <algorithm>
int main(int argc,char** argv){try{
  using namespace fastkpc;using namespace window_diag;
  need(argc==3,"smoke L4_RAW L5_RAW");
  need(cudaSetDevice(0)==cudaSuccess,"select smoke device");
  auto run=[](auto windows){
    std::vector<const Request*> all;for(const auto& w:windows)for(const auto& r:w.requests)all.push_back(&r);
    auto cost=[](const Request* r){const auto& x=r->expected.integral.at(3);return *std::max_element(x.begin(),x.end());};
    auto* slow=*std::max_element(all.begin(),all.end(),[&](auto* a,auto* b){return cost(a)<cost(b);});
    for(const auto* r:{all.front(),slow}){const auto& g=r->geometry;
      auto p=create_multi_penalty_gcv_capacity_prepared(g.X.data(),g.qr.data(),g.tau.data(),g.R.data(),g.pivot.data(),
        g.roots,g.matrices,g.ranks,g.initial.data(),g.n,g.q,g.penalties,g.capacity,0);
      auto result=multi_penalty_gcv_capacity_optimize_batch_with_residual(p,r->Y.data(),g.n,r->keys.size(),r->keys,r->control);
      release_multi_penalty_gcv_capacity_residual(result.residual);result.residual.reset();p.reset();
      std::cout<<"PASS q="<<g.q<<" targets="<<r->keys.size()<<" max_iterations="<<cost(r)<<'\n';
    }
  };
  run(l4_diag::read_layer(argv[1]));run(layer_diag::read_layer(argv[2]));return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
