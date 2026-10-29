// Host-only projection of sealed fixtures; no numerical evaluation.
#include "../optimizer_l4_diagnostic_20261023/fixture.hpp"
#include "../optimizer_layer_refill_20261023/layer_fixture.hpp"
#include <iostream>
int main(int argc, char** argv) { try {
  using namespace window_diag;
  need(argc == 4, "fixture_index L4_RAW L5_RAW OUTPUT_PREFIX");
  std::ofstream out(std::string(argv[3])+".bin", std::ios::binary);
  std::ofstream keys(std::string(argv[3])+".tsv");
  need(bool(out)&&bool(keys), "open index");
  IO io{nullptr,&out};
  std::string schema="l45-optimizer-host-reference-v1"; io.value(schema);
  int total=3109;io.value(total);
  keys<<"level\trequest_index\ttarget_index\tkey\n";
  const auto emit=[&](auto windows,int level) {
    int index=0;
    for(auto& w:windows) for(auto& r:w.requests) {
      io.value(level);io.value(index);io.value(r.keys);
      r.expected.floating.resize(11);r.expected.integral.resize(16);
      r.expected.io(io);
      for(size_t t=0;t<r.keys.size();++t) {
        need(r.keys[t].find_first_of("\t\r\n")==std::string::npos,"key delimiter");
        keys<<level<<'\t'<<index<<'\t'<<t<<'\t'<<r.keys[t]<<'\n';
      }
      ++index;
    }
    std::cout<<"L"<<level<<" setups="<<index<<'\n';
  };
  emit(l4_diag::read_layer(argv[1]),4);emit(layer_diag::read_layer(argv[2]),5);
  need(bool(out)&&bool(keys),"index write");return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;} }
