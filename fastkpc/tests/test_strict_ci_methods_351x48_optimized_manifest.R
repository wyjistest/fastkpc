manifest_path <- paste0(
  "fastkpc/artifacts/strict_ci_methods_351x48_optimized_v2/manifest.json"
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
            "optimized strict-method manifest is missing")
manifest <- jsonlite::read_json(manifest_path, simplifyVector = FALSE)
assert_true(
  identical(
    manifest$schema_version,
    "fastkpc-strict-ci-methods-351x48-optimized-evidence-v2"
  ) && identical(manifest$phase10_status, "ACTIVE") &&
    identical(manifest$promotion_status, "NOT_PROMOTED") &&
    identical(manifest$sealed_holdout_status, "SEALED_NOT_RELEASED") &&
    !isTRUE(manifest$recommended_route_changed),
  "optimized strict-method status changed"
)

expected <- list(
  "hsic.gamma" = list(
    tests = 322679L, orientation = 6L,
    old_fits = 308199L, new_fits = 194687L
  ),
  "dcc.perm" = list(
    tests = 229675L, orientation = 11L,
    old_fits = 459350L, new_fits = 189896L
  ),
  "hsic.perm" = list(
    tests = 282113L, orientation = 2L,
    old_fits = 564226L, new_fits = 189258L
  )
)
assert_true(setequal(names(manifest$methods), names(expected)),
            "optimized strict-method set changed")
for (method in names(expected)) {
  value <- manifest$methods[[method]]
  work <- value$physical_work
  parity <- value$parity
  optimization <- value$optimization
  assert_true(
    identical(as.integer(parity$skeleton_ci_test_count),
              expected[[method]]$tests) &&
      identical(as.integer(parity$orientation_ci_test_count),
                expected[[method]]$orientation) &&
      identical(as.integer(work$previous_physical_residual_fits),
                expected[[method]]$old_fits) &&
      identical(as.integer(work$optimized_physical_residual_fits),
                expected[[method]]$new_fits) &&
      as.numeric(value$performance$speedup) > 1 &&
      as.numeric(value$performance$elapsed_saved_sec) > 0 &&
      identical(optimization$residual_route, "qr-through-2") &&
      identical(optimization$sha256_backend, "openssl-sha256") &&
      isTRUE(optimization$persistent_execution_context) &&
      isTRUE(optimization$setup_optimizer_pipeline_enabled) &&
      as.numeric(optimization$setup_optimizer_pipeline_overlap_sec) > 0 &&
      as.integer(optimization$fixed_sp_root_cache_lookups) > 0L &&
      as.integer(optimization$fixed_sp_root_cache_hits) > 0L &&
      as.integer(optimization$fixed_sp_root_cache_hits) <
        as.integer(optimization$fixed_sp_root_cache_lookups) &&
      identical(
        as.integer(optimization$fixed_sp_root_cache_identity_rejections), 0L
      ) &&
      identical(
        as.integer(optimization$trusted_consumer_payload_rescan_count), 0L
      ) &&
      as.numeric(value$performance$request_identity_build_sec) >= 0 &&
      as.numeric(value$performance$request_identity_validation_sec) >= 0 &&
      as.numeric(value$performance$incremental_elapsed_saved_sec) > 0 &&
      as.numeric(value$performance$incremental_speedup) > 1 &&
      as.integer(work$optimized_physical_residual_fits) ==
        as.integer(work$logical_residual_requests) -
          as.integer(work$bypassed_target_count) &&
      identical(as.integer(work$cache_eviction_count), 0L) &&
      as.integer(work$execution_context_call_count) > 0L &&
      as.integer(work$execution_context_reuse_count) ==
        as.integer(work$execution_context_call_count) - 1L &&
      as.numeric(work$residual_d2h_bytes) == 0 &&
      as.numeric(work$component_d2h_bytes) == 0 &&
      isTRUE(parity$p_values_within_contract) &&
      identical(as.integer(parity$decision_flip_count), 0L) &&
      isTRUE(parity$logical_trace_identical) &&
      isTRUE(parity$adjacency_identical) &&
      isTRUE(parity$sepsets_identical) &&
      isTRUE(parity$pmax_within_tolerance) &&
      isTRUE(parity$n_edgetests_identical) &&
      isTRUE(parity$rng_state_identical) &&
      isTRUE(parity$orientation_trace_identical) &&
      isTRUE(parity$orientation_decisions_identical) &&
      isTRUE(parity$final_pdag_identical) &&
      isTRUE(parity$authority_clean) &&
      isTRUE(parity$cache_accounted) && isTRUE(parity$pass),
    paste(method, "optimized manifest gate failed")
  )
  if (method == "hsic.perm") {
    assert_true(
      isTRUE(optimization$hsic_permutation_inline_r_index) &&
        isTRUE(optimization$hsic_component_cache) &&
        as.numeric(work$permutation_inline_index_count) ==
          as.numeric(work$permutation_table_value_count) &&
        as.numeric(work$permutation_inline_draw_count) >=
          as.numeric(work$permutation_inline_index_count) &&
        as.integer(work$component_cache_hit_count) > 0L &&
        as.integer(work$component_cache_eviction_count) > 0L,
      "hsic.perm optimized cache/RNG evidence changed"
    )
  } else {
    assert_true(
      !isTRUE(optimization$hsic_permutation_inline_r_index) &&
        !isTRUE(optimization$hsic_component_cache) &&
        as.numeric(work$permutation_inline_index_count) == 0 &&
        as.integer(work$component_cache_hit_count) == 0L,
      paste(method, "unexpected HSIC permutation optimization evidence")
    )
  }
}
assert_true(
  identical(as.integer(manifest$aggregate$method_count), 3L) &&
    identical(as.integer(manifest$aggregate$skeleton_ci_test_count),
              834467L) &&
    identical(as.integer(manifest$aggregate$orientation_ci_test_count), 19L) &&
    as.numeric(manifest$aggregate$optimized_elapsed_sec) <
      as.numeric(manifest$aggregate$previous_elapsed_sec) &&
    abs(as.numeric(manifest$aggregate$optimized_elapsed_sec) - 2485.865) <
      1e-9 &&
    abs(as.numeric(manifest$aggregate$incremental_elapsed_saved_sec) -
          243.921) < 1e-9 &&
    as.numeric(manifest$aggregate$incremental_speedup) > 1 &&
    isTRUE(manifest$aggregate$all_process_gates_pass),
  "optimized aggregate gate failed"
)
assert_true(
  identical(
    manifest$optimization$request_authentication,
    "C++17-move-only-sealed-handle-incremental-OpenSSL-SHA256"
  ) &&
    identical(
      as.integer(manifest$optimization$trusted_consumer_payload_rescan_count),
      0L
    ) &&
    isTRUE(manifest$optimization$untrusted_boundary_full_payload_rehash_required) &&
    identical(manifest$optimization$permutation_gpu_preparation_overlap,
              "not-implemented") &&
    identical(manifest$provenance$sha256_backend, "openssl-sha256") &&
    nzchar(manifest$provenance$openssl_version),
  "optimized SHA-256 provenance changed"
)

source_paths <- unlist(
  manifest$provenance$source_file_relative_paths, use.names = TRUE
)
source_hashes <- unlist(
  manifest$provenance$source_file_sha256, use.names = TRUE
)
assert_true(identical(names(source_paths), names(source_hashes)) &&
              all(file.exists(source_paths)),
            "optimized source closure is incomplete")
actual_source_hashes <- vapply(source_paths, sha256_file, character(1L))
assert_true(identical(unname(actual_source_hashes), unname(source_hashes)),
            "optimized source closure hash changed")

native_path <- manifest$provenance$native_library_path
if (file.exists(native_path)) {
  assert_true(
    identical(sha256_file(native_path),
              manifest$provenance$native_library_sha256),
    "optimized native binary hash changed"
  )
}

manifest_dir <- dirname(normalizePath(manifest_path))
payloads <- unlist(lapply(manifest$methods, function(value) {
  value$payloads[c("candidate", "wanpdag_receipt")]
}), recursive = FALSE)
paths <- vapply(payloads, function(value) {
  file.path(manifest_dir, value$relative_path)
}, character(1L))
if (all(file.exists(paths))) {
  for (index in seq_along(paths)) {
    assert_true(
      identical(sha256_file(paths[[index]]), payloads[[index]]$sha256) &&
        identical(as.numeric(file.info(paths[[index]])$size),
                  as.numeric(payloads[[index]]$bytes)),
      paste("optimized payload changed:", payloads[[index]]$relative_path)
    )
  }
  cat("optimized strict-method payload hashes: PASS\n")
} else {
  cat("SKIP optimized payload hashes: local RDS payloads unavailable\n")
}

cat("test_strict_ci_methods_351x48_optimized_manifest.R: PASS\n")
