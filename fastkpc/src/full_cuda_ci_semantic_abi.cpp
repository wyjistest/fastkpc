#include "full_cuda_ci_semantic_abi.hpp"

namespace fastkpc {

FullCudaCiSemanticAbiInfo full_cuda_ci_semantic_abi_info() {
  FullCudaCiSemanticAbiInfo info;
  info.schema_version = "full-cuda-ci-semantic-abi-info-v1";
  info.abi_major = 1;
  info.abi_minor = 0;
  info.capabilities = {
    "prepared-s-device-resident-v1",
    "fixed-sp-stable-batch-v1",
    "device-residual-consumer-event-v1"
  };
  info.capability_status = {
    {"prepared-s-device-resident-v1", "available"},
    {"target-optimizer-opaque-v1", "interface_only"},
    {"fixed-sp-stable-batch-v1", "available"},
    {"device-residual-consumer-event-v1", "available"},
    {"dcov-component-opaque-v1", "interface_only"},
    {"logical-ci-batch-v1", "interface_only"},
    {"compact-ci-result-v1", "interface_only"},
    {"deterministic-eviction-reconstruction-v1", "unavailable"}
  };
  info.backend_semantic_version = "full-cuda-ci-semantic-abi-v1";
  info.producer_contract_hash =
    "49bbfe6a3b6f2ad643f51500f1268bcf5a45a44f25e570ef833ad438a5ebeb7c";
  info.device_residency_flags = {
    {"prepared_s_device_resident", true},
    {"target_optimizer_device_resident", true},
    {"residual_device_resident", true},
    {"dcov_component_device_resident", true},
    {"logical_ci_batch_device_resident", true},
    {"compact_result_host_visible", true},
    {"production_residual_d2h_forbidden", true},
    {"production_component_d2h_forbidden", true}
  };
  info.semantic_objects = {
    "PreparedSGpuHandle",
    "TargetOptimizerStateHandle",
    "DeviceResidualHandle",
    "DcovComponentHandle",
    "LogicalCiBatchHandle",
    "CompactCiResult"
  };
  info.compact_result_fields = {
    "logical_sequence_id",
    "p_value",
    "status",
    "solver_route",
    "optimizer_status",
    "dcov_status",
    "diagnostic_flags"
  };
  return info;
}

}  // namespace fastkpc
