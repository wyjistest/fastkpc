// Test-only planner injection: exercise the real one-call empty level path.
#include "skeleton_task_scheduler.hpp"
#include <dlfcn.h>
#include <cstdlib>
#include <stdexcept>
LayerPlan make_layer_plan(const std::vector<int>& adjacency,int p,int level) {
  if(level==3) return LayerPlan{level,p,adjacency,{},0,0,0};
  using F=decltype(&make_layer_plan);
  static F original=[] {
    void* h=dlopen(std::getenv("FASTKPC_LSCP_TEST_NATIVE"),RTLD_NOW|RTLD_NOLOAD);
    if(!h) throw std::runtime_error("empty-layer test native missing");
    auto f=reinterpret_cast<F>(dlsym(h,"_Z15make_layer_planRKSt6vectorIiSaIiEEii"));
    dlclose(h);
    if(!f) throw std::runtime_error("empty-layer test planner symbol missing");
    return f;
  }();
  return original(adjacency,p,level);
}
