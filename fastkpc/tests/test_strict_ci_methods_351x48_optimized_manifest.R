manifest_path <- paste0(
  "fastkpc/artifacts/strict_ci_methods_351x48_optimized_v1/manifest.json"
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
    "fastkpc-strict-ci-methods-351x48-optimized-evidence-v1"
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
      as.integer(work$optimized_physical_residual_fits) ==
        as.integer(work$logical_residual_requests) -
          as.integer(work$bypassed_target_count) &&
      identical(as.integer(work$cache_eviction_count), 0L) &&
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
}
assert_true(
  identical(as.integer(manifest$aggregate$method_count), 3L) &&
    identical(as.integer(manifest$aggregate$skeleton_ci_test_count),
              834467L) &&
    identical(as.integer(manifest$aggregate$orientation_ci_test_count), 19L) &&
    as.numeric(manifest$aggregate$optimized_elapsed_sec) <
      as.numeric(manifest$aggregate$previous_elapsed_sec) &&
    isTRUE(manifest$aggregate$all_process_gates_pass),
  "optimized aggregate gate failed"
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
