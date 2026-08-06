manifest_path <- paste0(
  "fastkpc/artifacts/strict_ci_methods_351x48_permutation_overlap_v1/",
  "manifest.json"
)

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
sha256_file <- function(path) unname(digest::digest(
  file = path, algo = "sha256", serialize = FALSE
))

assert_true(requireNamespace("jsonlite", quietly = TRUE) &&
              requireNamespace("digest", quietly = TRUE),
            "jsonlite and digest are required")
assert_true(file.exists(manifest_path),
            "strict permutation-overlap manifest is missing")
manifest <- jsonlite::read_json(manifest_path, simplifyVector = FALSE)
assert_true(
  identical(
    manifest$schema_version,
    "fastkpc-strict-ci-permutation-overlap-351x48-evidence-v1"
  ) && identical(manifest$phase10_status, "ACTIVE") &&
    identical(manifest$promotion_status, "NOT_PROMOTED") &&
    identical(manifest$sealed_holdout_status, "SEALED_NOT_RELEASED") &&
    !isTRUE(manifest$recommended_route_changed) &&
    identical(manifest$runtime_producer_commit,
              "6c01d71cd357cda5b9b78827c892a2b1105683a4"),
  "strict permutation-overlap status or producer changed"
)

expected <- list(
  "dcc.perm" = list(
    tests = 229675L, orientation = 11L,
    previous = 849.693, overlap = 839.726,
    ready_submit = 0L, ready_after = 229675L,
    classification = "conditional-small-improvement"
  ),
  "hsic.perm" = list(
    tests = 282113L, orientation = 2L,
    previous = 1042.358, overlap = 950.008,
    ready_submit = 68L, ready_after = 281890L,
    classification = "go-material-improvement"
  )
)
assert_true(setequal(names(manifest$methods), names(expected)),
            "strict permutation-overlap method set changed")
root <- dirname(manifest_path)
for (method in names(expected)) {
  value <- manifest$methods[[method]]
  performance <- value$performance
  overlap <- value$overlap
  identity <- value$identity_and_rng
  parity <- value$parity
  target <- expected[[method]]
  assert_true(
    abs(as.numeric(performance$previous_elapsed_sec) - target$previous) <
        1e-9 &&
      abs(as.numeric(performance$overlap_elapsed_sec) - target$overlap) <
        1e-9 &&
      as.numeric(performance$elapsed_saved_sec) > 0 &&
      as.numeric(performance$speedup) > 1 &&
      identical(performance$classification, target$classification) &&
      isTRUE(overlap$enabled) &&
      identical(as.integer(overlap$preparation_submit_before_rng_count),
                target$tests) &&
      identical(as.integer(overlap$preparation_ready_at_submit_count),
                target$ready_submit) &&
      identical(as.integer(overlap$preparation_ready_after_permutation_count),
                target$ready_after) &&
      identical(as.integer(overlap$deferred_preparation_error_count), 0L) &&
      identical(as.integer(overlap$hidden_stream_sync_count), 0L) &&
      identical(as.integer(overlap$hidden_device_sync_count), 0L) &&
      identical(as.integer(overlap$submit_completion_event_wait_count), 0L) &&
      identical(as.integer(overlap$intermediate_host_wait_count), 0L) &&
      identical(as.integer(overlap$final_host_wait_count), target$tests) &&
      identical(as.integer(overlap$in_flight_peak), 1L) &&
      as.numeric(overlap$measured_lower_bound_sec) >= 0 &&
      as.numeric(overlap$measured_lower_bound_sec) <=
        as.numeric(overlap$modeled_upper_bound_sec) &&
      identical(as.integer(identity$static_authentication_count),
                target$tests) &&
      identical(as.integer(identity$permutation_attestation_count),
                target$tests) &&
      identical(as.integer(identity$combined_authentication_count),
                target$tests) &&
      identical(as.integer(identity$exact_rng_receipt_count), target$tests) &&
      identical(as.integer(identity$trusted_payload_rescan_count), 0L) &&
      isTRUE(identity$rng_state_identical) &&
      identical(as.integer(parity$skeleton_ci_test_count), target$tests) &&
      identical(as.integer(parity$orientation_ci_test_count),
                target$orientation) &&
      as.numeric(parity$max_abs_p_diff) == 0 &&
      isTRUE(parity$p_values_bitwise_exact) &&
      identical(as.integer(parity$decision_flip_count), 0L) &&
      isTRUE(parity$logical_trace_identical) &&
      isTRUE(parity$adjacency_identical) &&
      isTRUE(parity$sepsets_identical) &&
      isTRUE(parity$pmax_bitwise_exact) &&
      isTRUE(parity$n_edgetests_identical) &&
      isTRUE(parity$orientation_trace_identical) &&
      isTRUE(parity$orientation_decisions_identical) &&
      isTRUE(parity$orientation_p_values_bitwise_exact) &&
      isTRUE(parity$final_pdag_identical) && isTRUE(parity$pass),
    paste(method, "strict permutation-overlap manifest gate failed")
  )

  for (payload in value$payloads) {
    assert_true(
      grepl("^[0-9a-f]{64}$", payload$sha256) &&
        as.numeric(payload$bytes) > 0,
      paste(method, "payload identity is invalid")
    )
    path <- file.path(root, payload$relative_path)
    if (file.exists(path)) {
      assert_true(
        identical(unname(as.numeric(file.info(path)$size)),
                  as.numeric(payload$bytes)) &&
          identical(sha256_file(path), payload$sha256),
        paste(method, "local payload differs from manifest")
      )
    }
  }
}

aggregate <- manifest$aggregate
assert_true(
  identical(as.integer(aggregate$method_count), 2L) &&
    identical(as.integer(aggregate$skeleton_ci_test_count), 511788L) &&
    identical(as.integer(aggregate$orientation_ci_test_count), 13L) &&
    abs(as.numeric(aggregate$previous_elapsed_sec) - 1892.051) < 1e-9 &&
    abs(as.numeric(aggregate$overlap_elapsed_sec) - 1789.734) < 1e-9 &&
    abs(as.numeric(aggregate$elapsed_saved_sec) - 102.317) < 1e-9 &&
    as.numeric(aggregate$speedup) > 1 &&
    isTRUE(aggregate$all_process_gates_pass),
  "strict permutation-overlap aggregate changed"
)
assert_true(
  identical(manifest$unchanged_method_reference$ci_method, "hsic.gamma") &&
    !isTRUE(manifest$unchanged_method_reference$rerun_in_this_evidence) &&
    abs(as.numeric(manifest$unchanged_method_reference$elapsed_sec) -
          593.814) < 1e-9 &&
    isTRUE(manifest$provenance$head_equals_origin_main) &&
    !isTRUE(manifest$provenance$runtime_sources_dirty_or_untracked) &&
    identical(
      manifest$provenance$native_library_sha256,
      "88eb8939759e6beb119126b06a2a110d2f7ca44ccda4333c8fef0109bc5fa59b"
    ) &&
    grepl("^[0-9a-f]{64}$", manifest$provenance$dataset_sha256),
  "strict permutation-overlap provenance changed"
)

provenance <- manifest$provenance
for (name in names(provenance$runtime_source_relative_paths)) {
  path <- provenance$runtime_source_relative_paths[[name]]
  assert_true(
    file.exists(path) &&
      identical(sha256_file(path), provenance$runtime_source_sha256[[name]]) &&
      identical(provenance$runtime_source_git_state[[name]], "clean"),
    paste(name, "runtime source closure changed")
  )
}
for (name in names(provenance$evidence_tool_relative_paths)) {
  path <- provenance$evidence_tool_relative_paths[[name]]
  assert_true(
    file.exists(path) &&
      identical(sha256_file(path), provenance$evidence_tool_sha256[[name]]),
    paste(name, "evidence tooling closure changed")
  )
}
if (file.exists(provenance$native_library_path)) {
  assert_true(
    identical(sha256_file(provenance$native_library_path),
              provenance$native_library_sha256),
    "local native library differs from overlap manifest"
  )
}
assert_true(
  file.exists(provenance$dataset_path) &&
    identical(sha256_file(provenance$dataset_path),
              provenance$dataset_sha256) &&
    identical(
      sha256_file(paste0(
        "fastkpc/artifacts/strict_ci_methods_351x48_optimized_v2/",
        "manifest.json"
      )),
      provenance$previous_manifest_sha256
    ),
  "dataset or prior-evidence identity changed"
)

cat("test_strict_ci_permutation_overlap_351x48_manifest.R: PASS\n")
