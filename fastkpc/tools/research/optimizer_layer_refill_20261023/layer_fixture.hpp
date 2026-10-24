#pragma once
#include "../optimizer_windows_20261022/fixture.hpp"
#include <set>

namespace layer_diag {
using namespace window_diag;
struct Window {
  int level=5,window=0;
  std::vector<Request> requests;
  void io(IO& io) {
    std::string schema="optimizer-layer-window-v1";io.value(schema);
    need(schema=="optimizer-layer-window-v1","layer fixture schema");
    io.value(level);io.value(window);
    need(level==5 && window>=1 && window<=15,"layer window identity");
    int count=requests.size();io.value(count);
    need(count==(window==15?59:64),"original L5 window size changed");
    if(io.in) requests.resize(count);
    for(auto& r:requests) r.io(io);
  }
  void save(const std::string& path) {
    std::ofstream out(path,std::ios::binary);IO io{nullptr,&out};this->io(io);
  }
  static Window read(const std::string& path) {
    Window w;std::ifstream in(path,std::ios::binary);IO io{&in,nullptr};w.io(io);
    need(in.peek()==EOF,"layer fixture trailing bytes");return w;
  }
};
inline std::vector<Window> read_layer(const std::string& path) {
  std::vector<Window> windows;std::set<std::string> keys;
  int setups=0,targets=0;
  for(int i=1;i<=15;++i) {
    auto w=Window::read(path+"/L5-W"+std::to_string(i)+".bin");
    need(w.window==i,"window order changed");
    for(const auto& r:w.requests) {
      const auto& g=r.geometry;
      need(g.n==351 && g.q==46 && g.penalties==5 &&
        g.capacity==int(r.keys.size()) && r.Y.size()==size_t(g.n)*r.keys.size(),
        "L5 shape changed");
      need(r.control.exact_primary_only && r.control.force_final_selected_evaluation &&
        r.control.decomposition_trace_capacity_per_target==0,"L5 control changed");
      for(const auto& key:r.keys) need(keys.insert(key).second,"duplicate target identity");
      ++setups;targets+=r.keys.size();
    }
    windows.push_back(std::move(w));
  }
  need(setups==955 && targets==6816,"L5 coverage changed");return windows;
}
}
