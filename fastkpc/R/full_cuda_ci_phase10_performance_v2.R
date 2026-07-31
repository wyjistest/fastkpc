fastkpc_full_cuda_phase10_performance_v2_require <- function(
    condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
  invisible(TRUE)
}

fastkpc_full_cuda_phase10_performance_budget_v2_path <- function(
    root = .fastkpc_full_cuda_phase35_contract_root()) {
  file.path(root, "performance_budget_v2.json")
}

fastkpc_full_cuda_phase10_validate_performance_budget_v2 <- function(payload) {
  required <- c(
    "reference_machine_contract", "supersedes_performance_claim_from",
    "legacy_v1_evidence", "boundaries", "cache_precondition", "promotion",
    "checkpoints", "measurement", "freeze"
  )
  .fastkpc_full_cuda_phase35_require_fields(
    payload, required, "Phase 10 performance budget v2"
  )
  boundaries <- payload$boundaries
  expected_boundaries <- c(
    "fresh_process_cold", "fresh_data_compute_warm", "replay_warm",
    "correct_baseline"
  )
  required_positive <- c(
    "physical_tests_evaluated", "physical_residual_fits",
    "native_setup_count", "cuda_single_penalty_target_count",
    "cuda_multi_penalty_target_count", "cuda_dcov_component_count",
    "cuda_dcov_pair_count", "cuda_gamma_pvalue_count"
  )
  required_summary <- c(
    "fresh_process_cold_median", "fresh_data_compute_warm_median",
    "replay_warm_median", "baseline_median",
    "compute_warm_to_baseline_ratio", "cold_to_baseline_ratio"
  )
  required_cache_fields <- c(
    "dataset_key", "cache_epoch_before_reset", "cache_epoch_after_reset",
    "result_cache_entries_before", "result_cache_entries_after_reset",
    "result_cache_dataset_entries_before",
    "result_cache_dataset_entries_after_reset",
    "target_cache_entries_before", "target_cache_entries_after_reset",
    "target_cache_dataset_entries_before",
    "target_cache_dataset_entries_after_reset",
    "preexisting_result_cache_hit_count",
    "preexisting_target_cache_hit_count"
  )
  cold <- boundaries$fresh_process_cold
  compute <- boundaries$fresh_data_compute_warm
  replay <- boundaries$replay_warm
  baseline <- boundaries$correct_baseline
  clean <- identical(payload$reference_machine_contract,
                     "reference_machine_v1") &&
    identical(payload$supersedes_performance_claim_from,
              "performance_budget_v1") &&
    identical(payload$legacy_v1_evidence$artifact,
              "promotion_351x48_v1") &&
    identical(payload$legacy_v1_evidence$historical_warm_classification,
              "replay-warm") &&
    identical(names(boundaries), expected_boundaries) &&
    identical(cold$repetitions, 5L) &&
    identical(cold$process_state, "new-R-process") &&
    identical(cold$cuda_context_state, "uninitialized") &&
    identical(cold$dataset_specific_cache_state, "empty") &&
    identical(cold$gate$correct_baseline_ratio_max, "1.00") &&
    identical(compute$repetitions, 5L) &&
    identical(compute$process_state, "new-R-process") &&
    identical(compute$prewarm_dataset_policy,
              "different-DatasetKey-and-noncanonical") &&
    isTRUE(compute$dataset_specific_cache_reset_required) &&
    identical(compute$preexisting_dataset_entry_count_max, 0L) &&
    isTRUE(compute$cache_epoch_advance_required) &&
    identical(
      unname(unlist(compute$required_positive_counters, use.names = FALSE)),
      required_positive
    ) &&
    identical(compute$gate$median_upper_bound_ms, 120000L) &&
    identical(compute$gate$correct_baseline_ratio_max, "0.80") &&
    identical(compute$gate$stretch_median_ms, 60000L) &&
    identical(replay$repetitions, 5L) &&
    identical(replay$dataset_key_state, "same-DatasetKey") &&
    identical(replay$promotion_gate, FALSE) && isTRUE(replay$report_only) &&
    identical(baseline$repetitions, 5L) &&
    isTRUE(baseline$same_machine_and_campaign_required) &&
    identical(
      unname(unlist(
        payload$cache_precondition$required_fields, use.names = FALSE
      )),
      required_cache_fields
    ) &&
    identical(payload$cache_precondition$reset_scope,
              "all-dataset-semantic-caches") &&
    identical(payload$cache_precondition$partial_reset, "forbidden") &&
    identical(payload$promotion$primary_performance_boundary,
              "fresh_data_compute_warm") &&
    isTRUE(payload$promotion$fresh_process_cold_gate_required) &&
    identical(payload$promotion$replay_warm_gate_allowed, FALSE) &&
    identical(payload$checkpoints$A_fresh_data_compute_upper_bound_ms,
              600000L) &&
    identical(payload$checkpoints$B_fresh_data_compute_upper_bound_ms,
              180000L) &&
    identical(payload$checkpoints$final_fresh_data_compute_upper_bound_ms,
              120000L) &&
    identical(
      unname(unlist(
        payload$measurement$required_summary_fields, use.names = FALSE
      )),
      required_summary
    ) &&
    identical(payload$measurement$generic_warm_label, "forbidden") &&
    isTRUE(payload$freeze$public_500x50_development_fixture_required) &&
    isTRUE(payload$freeze$fresh_data_compute_gate_required_before_holdout_open)
  fastkpc_full_cuda_phase10_performance_v2_require(
    clean, "Phase 10 performance budget v2 policy is invalid"
  )
  invisible(TRUE)
}

fastkpc_full_cuda_phase10_load_performance_budget_v2 <- function(
    root = .fastkpc_full_cuda_phase35_contract_root()) {
  contract <- fastkpc_full_cuda_phase35_contract_identity_from_path(
    fastkpc_full_cuda_phase10_performance_budget_v2_path(root),
    "performance_budget_v2", validate_known_contract = FALSE
  )
  fastkpc_full_cuda_phase10_performance_v2_require(
    identical(contract$semantic_version,
              list(major = 2L, minor = 0L, patch = 0L)),
    "Phase 10 performance budget v2 semantic version is invalid"
  )
  fastkpc_full_cuda_phase10_validate_performance_budget_v2(contract$payload)
  contract
}

fastkpc_full_cuda_phase10_contract_names_v2 <- function() {
  c(
    "architecture_contract_v1", "numerical_contract_v1",
    "artifact_identity_contract_v1", "reference_machine_v1",
    "performance_budget_v2", "development_qualification_corpus_v1",
    "metamorphic_contract_v1", "promotion_holdout_manifest_v1"
  )
}

fastkpc_full_cuda_phase10_load_contract_set_v2 <- function(
    root = .fastkpc_full_cuda_phase35_contract_root(),
    verify_source_artifacts = TRUE) {
  legacy <- fastkpc_full_cuda_phase35_load_contract_set(
    root = root, verify_source_artifacts = verify_source_artifacts
  )
  performance <- fastkpc_full_cuda_phase10_load_performance_budget_v2(root)
  contracts <- c(
    legacy[c(
      "architecture_contract_v1", "numerical_contract_v1",
      "artifact_identity_contract_v1", "reference_machine_v1"
    )],
    list(performance_budget_v2 = performance),
    legacy[c(
      "development_qualification_corpus_v1", "metamorphic_contract_v1",
      "promotion_holdout_manifest_v1"
    )]
  )
  fastkpc_full_cuda_phase10_performance_v2_require(
    identical(names(contracts), fastkpc_full_cuda_phase10_contract_names_v2()),
    "Phase 10 tracked v2 contract set is malformed"
  )
  contracts
}
