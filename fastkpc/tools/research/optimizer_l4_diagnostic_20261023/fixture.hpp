#pragma once
#include "../optimizer_windows_20261022/fixture.hpp"
#include <set>

namespace l4_diag {
using namespace window_diag;
struct Window {
  int level=4,window=0;
  std::vector<Request> requests;
  void io(IO& io) {
    std::string schema="optimizer-l4-window-v1";io.value(schema);
    need(schema=="optimizer-l4-window-v1","L4 fixture schema");
    io.value(level);io.value(window);
    need(level==4&&window>=1&&window<=34,"L4 window identity");
    int count=requests.size();io.value(count);
    need(count==(window==34?42:64),"original L4 window size changed");
    if(io.in)requests.resize(count);
    for(auto& r:requests)r.io(io);
  }
  void save(const std::string& path) {
    std::ofstream out(path,std::ios::binary);IO io{nullptr,&out};this->io(io);
  }
  static Window read(const std::string& path) {
    Window w;std::ifstream in(path,std::ios::binary);IO io{&in,nullptr};w.io(io);
    need(in.peek()==EOF,"L4 fixture trailing bytes");return w;
  }
};
inline std::vector<Window> read_layer(const std::string& path) {
  std::vector<Window> windows;std::set<std::string> keys;int setups=0,targets=0;
  for(int i=1;i<=34;++i) {
    auto w=Window::read(path+"/L4-W"+std::to_string(i)+".bin");
    need(w.window==i,"L4 window order changed");
    for(const auto& r:w.requests) {
      const auto& g=r.geometry;
      need(g.n==351&&g.q==37&&g.penalties==4&&g.capacity==int(r.keys.size())&&
        r.Y.size()==size_t(g.n)*r.keys.size(),"L4 shape changed");
      need(r.control.exact_primary_only&&r.control.force_final_selected_evaluation&&
        r.control.decomposition_trace_capacity_per_target==0,"L4 control changed");
      for(const auto& key:r.keys)need(keys.insert(key).second,"duplicate L4 target identity");
      ++setups;targets+=r.keys.size();
    }
    windows.push_back(std::move(w));
  }
  need(setups==2154&&targets==17767,"L4 coverage changed");return windows;
}
}
