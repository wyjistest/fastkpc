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

expected_names <- c(
  "architecture_contract_v1",
  "numerical_contract_v1",
  "artifact_identity_contract_v1",
  "reference_machine_v1",
  "performance_budget_v1",
  "development_qualification_corpus_v1",
  "metamorphic_contract_v1",
  "promotion_holdout_manifest_v1"
)
assert_true(
  identical(fastkpc_full_cuda_phase35_contract_names(), expected_names),
  "Phase 3.5 tracked contract names and order must be exact"
)

contracts <- fastkpc_full_cuda_phase35_load_contract_set()
assert_true(
  identical(names(contracts), expected_names),
  "all tracked Phase 3.5 contracts must load"
)
for (name in expected_names) {
  contract <- contracts[[name]]
  assert_true(
    identical(contract$contract_name, name) &&
      identical(contract$contract_schema_version,
                "full-cuda-ci-tracked-contract-v1") &&
      identical(contract$semantic_version,
                list(major = 1L, minor = 0L, patch = 0L)) &&
      grepl("^[0-9a-f]{64}$", contract$sha256) &&
      identical(
        fastkpc_full_cuda_phase35_sha256_utf8(contract$canonical_json),
        contract$sha256
      ),
    paste0(name, " must expose an authenticated canonical snapshot")
  )
  reparsed <- fastkpc_full_cuda_phase35_parse_json(
    contract$canonical_json, paste0(name, " canonical snapshot")
  )
  assert_true(
    identical(
      fastkpc_full_cuda_phase35_canonical_json(reparsed),
      contract$canonical_json
    ),
    paste0(name, " canonical JSON must be idempotent")
  )
}

assert_true(
  identical(
    fastkpc_full_cuda_phase35_canonical_json(list(z = 1L, a = TRUE)),
    fastkpc_full_cuda_phase35_canonical_json(list(a = TRUE, z = 1L))
  ),
  "canonical JSON must sort object keys"
)
assert_true(
  identical(
    fastkpc_full_cuda_phase35_canonical_json(
      fastkpc_full_cuda_phase35_parse_json(
        '{"escaped":"\\u0061","array":[3,2,1]}', "escape fixture"
      )
    ),
    '{"array":[3,2,1],"escaped":"a"}'
  ),
  "canonical JSON must normalize equivalent string escapes"
)
assert_error(
  fastkpc_full_cuda_phase35_parse_json('{"a":1,"a":2}', "duplicate"),
  "duplicate object key",
  "duplicate JSON object keys must fail closed"
)
assert_error(
  fastkpc_full_cuda_phase35_parse_json('{"a":1.5}', "fractional"),
  "integer JSON numbers",
  "floating JSON numbers must use canonical decimal strings"
)
assert_error(
  fastkpc_full_cuda_phase35_parse_json('{"a":NaN}', "nonfinite"),
  "valid JSON",
  "non-standard non-finite JSON must fail closed"
)

architecture <- contracts$architecture_contract_v1$payload
assert_true(
  identical(
    vapply(architecture$semantic_objects, `[[`, character(1L), "name"),
    c(
      "PreparedSGpuHandle", "TargetOptimizerStateHandle",
      "DeviceResidualHandle", "DcovComponentHandle",
      "LogicalCiBatchHandle", "CompactCiResult"
    )
  ),
  "architecture contract must freeze every required semantic object"
)
assert_true(
  identical(architecture$abi$major, 1L) &&
    identical(architecture$abi$minor, 0L) &&
    identical(architecture$large_payload_policy$residual_d2h, "forbidden") &&
    identical(architecture$large_payload_policy$component_d2h, "forbidden"),
  "architecture ABI and production residency policy must be explicit"
)

numerical <- contracts$numerical_contract_v1$payload
assert_true(
  identical(numerical$decision_semantics$independent_when, "p_value >= alpha") &&
    identical(numerical$decision_semantics$allowed_flip_count, 0L) &&
    identical(numerical$precision$storage, "IEEE-754-binary64") &&
    identical(numerical$tolerances$residual$max_absolute, "1e-7") &&
    identical(numerical$tolerances$residual$relative_l2, "1e-8") &&
    identical(numerical$tolerances$p_value$absolute, "1e-10") &&
    identical(numerical$nonfinite_policy, "fail-closed-before-replay"),
  "numerical authority must pin Phase 3-compatible gates and decisions"
)
assert_true(
  identical(
    vapply(numerical$condition_buckets, `[[`, character(1L), "solver"),
    c("CHOLESKY_BATCHED", "AUGMENTED_QR", "AUGMENTED_SVD",
      "AUGMENTED_SVD")
  ),
  "condition buckets must retain the qualified Phase 3 route policy"
)

machine <- contracts$reference_machine_v1$payload
assert_true(
  identical(machine$gpu$model, "NVIDIA GeForce RTX 4090") &&
    identical(machine$gpu$device_id, 0L) &&
    identical(machine$gpu$compute_capability, "8.9") &&
    identical(machine$gpu$total_memory_bytes, 25757220864) &&
    identical(machine$timing$build_included, FALSE) &&
    identical(machine$timing$cold_and_warm_reported_separately, TRUE),
  "reference machine and timing boundaries must be frozen"
)

budget <- contracts$performance_budget_v1$payload
component_budget <- vapply(
  budget$component_budgets, `[[`, integer(1L), "warm_upper_bound_ms"
)
assert_true(
  identical(sum(component_budget), budget$feasibility$total_upper_bound_ms) &&
    identical(budget$feasibility$total_upper_bound_ms, 120000L) &&
    identical(
      sum(component_budget[c("dcov_component", "dcov_pair_and_gamma")]),
      budget$feasibility$dcov_total_upper_bound_ms
    ) &&
    identical(budget$feasibility$dcov_total_upper_bound_ms, 47000L) &&
    identical(budget$promotion$warm_median_upper_bound_ms, 120000L) &&
    identical(budget$promotion$correct_baseline_ratio_max, "0.80"),
  "performance budget must conserve the complete 120-second envelope"
)

development <- contracts$development_qualification_corpus_v1$payload
assert_true(
  identical(development$canonical_counts$target_count, 6143L) &&
    identical(development$canonical_counts$dcov_pair_count, 3808L) &&
    identical(development$canonical_counts$near_alpha_pair_count, 1478L) &&
    identical(
      development$source_identities$canonical_target_corpus_sha256,
      "b843630969f116da63f7fad095c54de2ff471540159ff97ca56c3871d6b2e1fa"
    ),
  "development corpus must bind the qualified Phase 3 risk corpus"
)
metamorphic <- contracts$metamorphic_contract_v1$payload
assert_true(
  identical(
    vapply(metamorphic$transformations, `[[`, character(1L), "name"),
    c(
      "conditioning-column-permutation", "basis-sign-flip",
      "equivalent-orthogonal-rotation", "batch-split-merge",
      "stream-count-variation", "cache-capacity-variation",
      "standalone-versus-batched"
    )
  ),
  "metamorphic corpus must cover every roadmap transformation"
)
holdout <- contracts$promotion_holdout_manifest_v1$payload
assert_true(
  identical(holdout$state, "SEALED_NOT_RELEASED") &&
    identical(holdout$ordinary_development_access, "forbidden") &&
    isFALSE(holdout$payload_present_in_repository) &&
    identical(holdout$implementation_change_after_open_policy,
              "retire-version-and-reseal-new-version"),
  "promotion holdout must remain sealed during ordinary development"
)

snapshots <- fastkpc_full_cuda_phase35_contract_snapshots(contracts)
assert_true(
  identical(names(snapshots), expected_names) &&
    all(vapply(snapshots, function(value) {
      identical(
        names(value),
        c("contract_name", "semantic_version", "sha256", "snapshot")
      ) && identical(value$contract_name, value$snapshot$contract_name)
    }, logical(1L))),
  "artifact snapshots must carry exact tracked contracts and hashes"
)
mutated_contracts <- contracts
mutated_contracts$architecture_contract_v1$document$payload$abi$minor <- 1L
assert_error(
  fastkpc_full_cuda_phase35_contract_snapshots(mutated_contracts),
  "tracked contract hash is invalid",
  "mutated in-memory contract snapshots must fail before publication"
)

producer <- fastkpc_full_cuda_phase35_producer_identity(
  producer_source_closure_sha256 = strrep("1", 64L),
  native_binary_sha256 = strrep("2", 64L),
  route_semantic_version = "phase35-contract-test-v1",
  dataset_or_corpus_sha256 = strrep("3", 64L),
  oracle_sha256 = strrep("4", 64L),
  backend_configuration_sha256 = strrep("5", 64L),
  build_recipe_sha256 = strrep("6", 64L),
  contracts = contracts
)
assert_true(
  !any(c("pid", "timestamp", "path", "inode", "session_id") %in%
       names(producer)) &&
    grepl("^[0-9a-f]{64}$", producer$identity_sha256),
  "producer semantic identity must exclude volatile receipt fields"
)
assert_error(
  fastkpc_full_cuda_phase35_producer_identity(
    producer_source_closure_sha256 = strrep("1", 64L),
    native_binary_sha256 = strrep("2", 64L),
    route_semantic_version = "phase35-contract-test-v1",
    dataset_or_corpus_sha256 = strrep("3", 64L),
    oracle_sha256 = strrep("4", 64L),
    backend_configuration_sha256 = strrep("5", 64L),
    build_recipe_sha256 = strrep("6", 64L),
    contracts = contracts,
    pid = 1L
  ),
  "unused argument",
  "producer identity must not accept volatile execution fields"
)

artifact <- fastkpc_full_cuda_phase35_identity_envelope(
  producer = producer,
  payload_manifest_sha256 = strrep("7", 64L)
)
semantic_hash <- artifact$artifact_semantic_sha256
producer_hash <- artifact$producer$identity_sha256
attestation <- fastkpc_full_cuda_phase35_validator_attestation(
  producer = producer,
  validator_source_closure_sha256 = strrep("8", 64L),
  validator_semantic_version = "phase35-validator-v1",
  validator_contracts = contracts,
  validation_timestamp_utc = "2026-07-28T00:00:00Z",
  environment_sha256 = strrep("9", 64L),
  validation_result = "PASS"
)
artifact_attested <- fastkpc_full_cuda_phase35_append_attestation(
  artifact, attestation
)
receipt <- fastkpc_full_cuda_phase35_execution_receipt(
  producer = producer,
  pid = 123L,
  session_id = "session-test",
  cuda_context_id = "context-test",
  artifact_path = "/tmp/artifact-test",
  artifact_inode = "42",
  staging_path = "/tmp/staging-test",
  recorded_at_utc = "2026-07-28T00:00:01Z"
)
artifact_receipted <- fastkpc_full_cuda_phase35_append_receipt(
  artifact_attested, receipt
)
assert_true(
  identical(artifact_receipted$producer$identity_sha256, producer_hash) &&
    identical(artifact_receipted$artifact_semantic_sha256, semantic_hash) &&
    length(artifact_receipted$attestations) == 1L &&
    length(artifact_receipted$execution_receipts) == 1L &&
    identical(
      artifact_receipted$attestations[[1L]]$attested_producer_sha256,
      producer_hash
    ) &&
    identical(
      artifact_receipted$execution_receipts[[1L]]$producer_sha256,
      producer_hash
    ),
  "attestations and receipts must remain append-only non-semantic evidence"
)
assert_true(
  isTRUE(fastkpc_full_cuda_phase35_validate_identity_envelope(
    artifact_receipted
  )),
  "three-layer identity envelope must validate"
)
forged_snapshot_producer <- producer
forged_snapshot_producer$contract_snapshots$architecture_contract_v1$
  snapshot$payload$abi$minor <- 1L
forged_snapshot <- forged_snapshot_producer$contract_snapshots$
  architecture_contract_v1$snapshot
forged_snapshot_producer$contract_snapshots$architecture_contract_v1$
  sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(forged_snapshot)
  )
forged_snapshot_producer$identity_sha256 <- NULL
forged_snapshot_producer <- .fastkpc_full_cuda_phase35_hash_fields(
  forged_snapshot_producer, "identity_sha256"
)
assert_error(
  .fastkpc_full_cuda_phase35_validate_producer(forged_snapshot_producer),
  "contract snapshot does not match tracked authority",
  "rehashed producer identity must not authorize a forged contract snapshot"
)
tampered <- artifact_receipted
tampered$producer$route_semantic_version <- "forged"
assert_error(
  fastkpc_full_cuda_phase35_validate_identity_envelope(tampered),
  "producer identity hash mismatch",
  "producer semantic mutation must fail validation"
)
forged_attestation <- attestation
forged_attestation$attested_producer_sha256 <- strrep("a", 64L)
forged_attestation$attestation_sha256 <- NULL
forged_attestation <- .fastkpc_full_cuda_phase35_hash_fields(
  forged_attestation, "attestation_sha256"
)
assert_error(
  fastkpc_full_cuda_phase35_append_attestation(
    artifact, forged_attestation
  ),
  "attested producer identity mismatch",
  "validator attestations must not impersonate another producer"
)

cat("PASS full CUDA CI Phase 3.5 tracked contracts and identity model\n")
