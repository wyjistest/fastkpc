#pragma once
#include "../late_optimizer_20261025/fixture.hpp"
namespace l6_diag {
using namespace late_diag;
inline std::vector<Window> read_layer(const std::string& path) {
  return late_diag::read_layer(path,6);
}
}
