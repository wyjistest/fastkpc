#pragma once
#include "../optimizer_windows_20261022/fixture.hpp"
#include <set>
namespace late_diag {
using namespace window_diag;
struct Window {
  int level=0,window=0;
  std::vector<Request> requests;
  void io(IO& io) {
    std::string schema="optimizer-late-window-v1";io.value(schema);
    need(schema=="optimizer-late-window-v1","late fixture schema");
    io.value(level);io.value(window);
    need((level==6&&window>=1&&window<=4)||(level==8&&window==1),"late window identity");
    int count=requests.size();io.value(count);
    need(count==(level==8?9:(window==4?53:64)),"late original window size");
    if(io.in)requests.resize(count);
    for(auto& r:requests)r.io(io);
  }
  void save(const std::string& path) {
    std::ofstream out(path,std::ios::binary);IO io{nullptr,&out};this->io(io);
  }
  static Window read(const std::string& path) {
    Window w;std::ifstream in(path,std::ios::binary);IO io{&in,nullptr};w.io(io);
    need(in.peek()==EOF,"late fixture trailing bytes");return w;
  }
};
inline std::vector<Window> read_layer(const std::string& path,int level) {
  need(level==6||level==8,"late layer");std::vector<Window> windows;
  std::set<std::string> keys;int setups=0;
  for(int i=1;i<=(level==6?4:1);++i){
    auto w=Window::read(path+"/L"+std::to_string(level)+"-W"+std::to_string(i)+".bin");
    need(w.level==level&&w.window==i,"late original window order");
    for(const auto& r:w.requests){const auto& g=r.geometry;
      need(g.n==351&&g.q==(level==6?55:73)&&g.penalties==level&&
           g.capacity==int(r.keys.size())&&r.Y.size()==size_t(g.n)*r.keys.size(),"late shape");
      need(r.control.exact_primary_only&&r.control.force_final_selected_evaluation&&
           r.control.decomposition_trace_capacity_per_target==0,"late exact control");
      for(const auto& key:r.keys)need(keys.insert(key).second,"duplicate late target");
      ++setups;
    }
    windows.push_back(std::move(w));
  }
  need(setups==(level==6?245:9),"late setup coverage");return windows;
}
}
