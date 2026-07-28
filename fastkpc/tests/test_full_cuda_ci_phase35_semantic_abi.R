source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) {
  if (!isTRUE(value)) fail(message)
}
assert_error <- function(expression, pattern, message) {
  error <- tryCatch({
    force(expression)
    NULL
  }, error = function(error) error)
  assert_true(inherits(error, "error"), message)
  assert_true(
    grepl(pattern, conditionMessage(error), fixed = TRUE),
    paste0(message, ": unexpected error: ", conditionMessage(error))
  )
}

contracts <- fastkpc_full_cuda_phase35_load_contract_set()
architecture <- contracts$architecture_contract_v1

load_fastkpc_cuda_native(rebuild = FALSE)
before <- .Call(
  "C_fixed_sp_cuda_test_resource_snapshot", PACKAGE = "fastkpc_cuda"
)
info <- fastkpc_full_cuda_phase35_semantic_abi_info(contracts = contracts)
after <- .Call(
  "C_fixed_sp_cuda_test_resource_snapshot", PACKAGE = "fastkpc_cuda"
)

assert_true(
  identical(before, after),
  "semantic ABI capability query must create no CUDA resources"
)
assert_true(
  identical(
    names(info),
    c(
      "schema_version", "abi_major", "abi_minor", "capabilities",
      "capability_status", "backend_semantic_version",
      "producer_contract_hash", "device_residency_flags",
      "semantic_objects", "compact_result_fields"
    )
  ),
  "semantic ABI query schema must be exact"
)
assert_true(
  identical(info$schema_version, "full-cuda-ci-semantic-abi-info-v1") &&
    identical(info$abi_major, 1L) &&
    identical(info$abi_minor, 0L) &&
    identical(info$backend_semantic_version,
              "full-cuda-ci-semantic-abi-v1") &&
    identical(info$producer_contract_hash, architecture$sha256),
  "native ABI query must bind the tracked architecture contract"
)

expected_vocabulary <- unlist(
  architecture$payload$capabilities, use.names = FALSE
)
assert_true(
  identical(names(info$capability_status), expected_vocabulary) &&
    all(unlist(info$capability_status, use.names = FALSE) %in%
        c("available", "interface_only", "unavailable")) &&
    identical(
      info$capabilities,
      names(info$capability_status)[
        unlist(info$capability_status, use.names = FALSE) == "available"
      ]
    ),
  "capability query must use the contract vocabulary and advertise only available capabilities"
)
assert_true(
  identical(
    info$capabilities,
    c(
      "prepared-s-device-resident-v1",
      "fixed-sp-stable-batch-v1",
      "device-residual-consumer-event-v1"
    )
  ) &&
    identical(
      info$capability_status[["target-optimizer-opaque-v1"]],
      "interface_only"
    ) &&
    identical(
      info$capability_status[["dcov-component-opaque-v1"]],
      "interface_only"
    ) &&
    identical(
      info$capability_status[["logical-ci-batch-v1"]],
      "interface_only"
    ) &&
    identical(
      info$capability_status[["compact-ci-result-v1"]],
      "interface_only"
    ) &&
    identical(
      info$capability_status[["deterministic-eviction-reconstruction-v1"]],
      "unavailable"
    ),
  "ABI query must not overstate unimplemented Phase 3.5 capabilities"
)

expected_objects <- vapply(
  architecture$payload$semantic_objects, `[[`, character(1L), "name"
)
assert_true(
  identical(info$semantic_objects, expected_objects) &&
    identical(
      info$compact_result_fields,
      unlist(architecture$payload$compact_result_fields, use.names = FALSE)
    ),
  "native ABI must expose every opaque semantic object and compact field"
)
assert_true(
  identical(
    names(info$device_residency_flags),
    c(
      "prepared_s_device_resident", "target_optimizer_device_resident",
      "residual_device_resident", "dcov_component_device_resident",
      "logical_ci_batch_device_resident", "compact_result_host_visible",
      "production_residual_d2h_forbidden",
      "production_component_d2h_forbidden"
    )
  ) && all(unlist(info$device_residency_flags, use.names = FALSE)),
  "ABI residency flags must expose production transfer prohibitions"
)

negotiated <- fastkpc_full_cuda_phase35_negotiate_semantic_abi(
  info,
  required_major = 1L,
  required_minor = 0L,
  required_capabilities = c(
    "prepared-s-device-resident-v1", "fixed-sp-stable-batch-v1"
  ),
  optional_capabilities = c(
    "dcov-component-opaque-v1", "future-optional-capability-v9"
  )
)
assert_true(
  isTRUE(negotiated$compatible) &&
    identical(negotiated$available_required_capabilities,
              c("prepared-s-device-resident-v1",
                "fixed-sp-stable-batch-v1")) &&
    identical(negotiated$unavailable_optional_capabilities,
              c("dcov-component-opaque-v1",
                "future-optional-capability-v9")),
  "compatible ABI negotiation must retain optional capability diagnostics"
)
assert_error(
  fastkpc_full_cuda_phase35_negotiate_semantic_abi(
    info, required_major = 2L, required_minor = 0L
  ),
  "semantic ABI major mismatch",
  "ABI major mismatch must fail closed"
)
assert_error(
  fastkpc_full_cuda_phase35_negotiate_semantic_abi(
    info, required_major = 1L, required_minor = 1L
  ),
  "semantic ABI minor capability mismatch",
  "provider minor below the consumer requirement must fail closed"
)
assert_error(
  fastkpc_full_cuda_phase35_negotiate_semantic_abi(
    info,
    required_capabilities = "dcov-component-opaque-v1"
  ),
  "required semantic capability is unavailable",
  "interface-only capability must not satisfy an execution requirement"
)
assert_error(
  fastkpc_full_cuda_phase35_negotiate_semantic_abi(
    info,
    required_capabilities = "unknown-required-capability-v1"
  ),
  "required semantic capability is unknown",
  "unknown required capability must fail closed"
)

header <- paste(
  readLines("fastkpc/src/full_cuda_ci_semantic_abi.hpp", warn = FALSE),
  collapse = "\n"
)
for (object in expected_objects) {
  if (identical(object, "CompactCiResult")) {
    pattern <- "class CompactCiResult;"
  } else {
    pattern <- paste0("class ", object, ";")
  }
  assert_true(
    grepl(pattern, header, fixed = TRUE),
    paste0(object, " must remain an opaque forward declaration")
  )
}
assert_true(
  !grepl("double*", header, fixed = TRUE) &&
    !grepl("void*", header, fixed = TRUE) &&
    !grepl("cudaStream_t", header, fixed = TRUE) &&
    !grepl("cudaEvent_t", header, fixed = TRUE),
  "public semantic ABI header must not freeze raw storage or CUDA layout"
)

cat("PASS full CUDA CI Phase 3.5 opaque semantic ABI negotiation\n")
