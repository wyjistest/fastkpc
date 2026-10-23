#pragma once
#include <string>
#include <vector>
namespace window_diag {
void observe_begin(const std::vector<std::string>& keys, bool enabled);
void observe_end(const std::string& path);
}
