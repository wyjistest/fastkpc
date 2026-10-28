#pragma once
#include "../optimizer_windows_20261022/fixture.hpp"
#include <set>
namespace l3_diag {
using namespace window_diag;
struct Window {
  int level=3,window=0;
  std::vector<Request> requests;
  void io(IO& io) {
    std::string schema="optimizer-l3-window-v1";io.value(schema);
    need(schema=="optimizer-l3-window-v1","L3 fixture schema");
    io.value(level);io.value(window);
    need(level==3&&window>=1&&window<=64,"L3 window identity");
    int count=requests.size();io.value(count);
    need(count==(window==64?33:64),"L3 original window size");
    if(io.in)requests.resize(count);
    for(auto& r:requests)r.io(io);
  }
  void save(const std::string& path) {
    std::ifstream existing(path);need(!existing.good(),"preserve existing capture");
    std::ofstream out(path,std::ios::binary);IO io{nullptr,&out};this->io(io);
    need(bool(out),"capture write failed");
  }
  static Window read(const std::string& path) {
    Window w;std::ifstream in(path,std::ios::binary);IO io{&in,nullptr};w.io(io);
    need(in.peek()==EOF,"L3 fixture trailing bytes");return w;
  }
};
inline std::vector<Window> read_layer(const std::string& path) {
  std::vector<Window> windows;std::set<std::string> keys;size_t setups=0,targets=0;
  for(int i=1;i<=64;++i){
    auto w=Window::read(path+"/L3-W"+std::to_string(i)+".bin");
    need(w.window==i,"L3 original window order");
    for(const auto& r:w.requests){const auto& g=r.geometry;
      need(g.n==351&&g.q==28&&g.penalties==3&&g.capacity>=2&&g.capacity<=33&&
        g.capacity==int(r.keys.size())&&r.Y.size()==size_t(g.n)*r.keys.size(),"L3 shape");
      need(r.control.exact_primary_only&&r.control.force_final_selected_evaluation&&
        r.control.decomposition_trace_capacity_per_target==0,"L3 exact controls");
      for(const auto& key:r.keys)need(keys.insert(key).second,"duplicate L3 target");
      ++setups;targets+=r.keys.size();
    }
    windows.push_back(std::move(w));
  }
  need(setups==4065&&targets==54878,"L3 complete coverage");return windows;
}
}
