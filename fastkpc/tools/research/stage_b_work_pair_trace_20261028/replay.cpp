// Keep the sealed replay as the real main, so its implicit success return is
// defined. The wrapper only authenticates existing host execution counters.
#include "common.hpp"
#include "cuda/full_cuda_ci_structured_eig.hpp"
#include <iostream>

// Falling off a renamed main would be undefined. Include its body as the real
// main and verify counters at scope exit, including exception/error paths.
namespace {
struct ModeAudit {
  ~ModeAudit() noexcept {
    const auto c = fastkpc::structured_component_execution_counts();
    const char* raw = std::getenv("FASTKPC_DCC_GAMMA_STRUCTURED_WORK_PAIR");
    const bool on = raw && std::string(raw) == "1";
    const int selected = on ? 1 : 0, other = 1-selected;
    const bool pass = raw && (std::string(raw)=="0" || std::string(raw)=="1") &&
      c.created[selected]==10 && c.destroyed[selected]==10 &&
      c.submissions[selected]==1638 && c.targets[selected]==104797 &&
      c.created[other]==0 && c.destroyed[other]==0 &&
      c.submissions[other]==0 && c.targets[other]==0;
    std::ofstream out(stage_b_diag::env("STAGE_B_OUTPUT")+"/mode.json");
    out << "{\"mode\":" << selected << ",\"pass\":" << (pass?"true":"false")
        << ",\"created\":" << c.created[selected] << ",\"destroyed\":" << c.destroyed[selected]
        << ",\"submissions\":" << c.submissions[selected] << ",\"targets\":" << c.targets[selected]
        << ",\"other_created\":" << c.created[other] << ",\"other_destroyed\":" << c.destroyed[other]
        << ",\"other_submissions\":" << c.submissions[other] << ",\"other_targets\":" << c.targets[other] << "}\n";
  }
};
ModeAudit audit;
}

#include "../stage_b_attribution_20261028/replay.cpp"
