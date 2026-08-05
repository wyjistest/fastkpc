manifest_path <- paste0(
  "fastkpc/artifacts/strict_ci_methods_351x48_v1/manifest.json"
)

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
sha256_file <- function(path) unname(digest::digest(
  file = path, algo = "sha256", serialize = FALSE
))

assert_true(requireNamespace("jsonlite", quietly = TRUE) &&
              requireNamespace("digest", quietly = TRUE),
            "jsonlite and digest are required for manifest validation")
assert_true(file.exists(manifest_path), "strict CI evidence manifest is missing")
manifest <- jsonlite::read_json(manifest_path, simplifyVector = FALSE)
assert_true(identical(
  manifest$schema_version,
  "fastkpc-strict-ci-methods-351x48-evidence-v1"
), "strict CI evidence schema changed")
assert_true(identical(manifest$promotion_status, "NOT_PROMOTED") &&
              identical(manifest$phase10_status, "ACTIVE") &&
              identical(manifest$sealed_holdout_status,
                        "SEALED_NOT_RELEASED") &&
              !isTRUE(manifest$recommended_route_changed),
            "strict CI evidence promotion state changed")
assert_true(identical(
  manifest$architecture$skeleton, "full-CUDA"
) && identical(
  manifest$architecture$orientation,
  "kpcalg CPU WAN-PDAG authority"
) && !isTRUE(manifest$architecture$full_cuda_wanpdag_claim),
"strict CI authority architecture changed")

expected <- list(
  "hsic.gamma" = c(skeleton = 322679L, orientation = 6L),
  "dcc.perm" = c(skeleton = 229675L, orientation = 11L),
  "hsic.perm" = c(skeleton = 282113L, orientation = 2L)
)
assert_true(setequal(names(manifest$methods), names(expected)),
            "strict CI method set changed")
for (method in names(expected)) {
  value <- manifest$methods[[method]]
  assert_true(
    identical(value$authority$skeleton_authority, "full_cuda") &&
      identical(value$authority$orientation_authority,
                "kpcalg_cpu_wanpdag") &&
      identical(value$authority$native_cuda_orientation_status,
                "experimental") &&
      identical(as.integer(value$parity$skeleton_ci_test_count),
                expected[[method]][["skeleton"]]) &&
      identical(as.integer(value$parity$orientation_ci_test_count),
                expected[[method]][["orientation"]]) &&
      identical(as.integer(value$parity$skeleton_SHD), 0L) &&
      isTRUE(value$parity$logical_trace_identical) &&
      isTRUE(value$parity$adjacency_identical) &&
      isTRUE(value$parity$sepsets_identical) &&
      isTRUE(value$parity$pmax_within_tolerance) &&
      isTRUE(value$parity$n_edgetests_identical) &&
      isTRUE(value$parity$orientation_trace_identical) &&
      isTRUE(value$parity$orientation_decisions_identical) &&
      isTRUE(value$parity$final_pdag_identical) &&
      isTRUE(value$parity$pass),
    paste(method, "manifest parity gate failed")
  )
}
assert_true(
  identical(as.integer(manifest$aggregate$method_count), 3L) &&
    identical(as.integer(manifest$aggregate$skeleton_ci_test_count),
              834467L) &&
    identical(as.integer(manifest$aggregate$orientation_ci_test_count), 19L) &&
    isTRUE(manifest$aggregate$all_SHD_zero) &&
    isTRUE(manifest$aggregate$all_process_gates_pass),
  "strict CI aggregate gate failed"
)

source_paths <- unlist(
  manifest$provenance$source_file_relative_paths, use.names = TRUE
)
source_hashes <- unlist(
  manifest$provenance$source_file_sha256, use.names = TRUE
)
assert_true(identical(names(source_paths), names(source_hashes)) &&
              all(file.exists(source_paths)),
            "strict CI source closure is incomplete")
actual_source_hashes <- vapply(source_paths, sha256_file, character(1L))
assert_true(identical(unname(actual_source_hashes), unname(source_hashes)),
            "strict CI source closure hash changed")

native_path <- manifest$provenance$native_library_path
if (file.exists(native_path)) {
  assert_true(identical(
    sha256_file(native_path), manifest$provenance$native_library_sha256
  ), "strict CI native binary hash changed")
}

manifest_dir <- dirname(normalizePath(manifest_path))
payload_entries <- unlist(lapply(manifest$methods, function(method) {
  method$payloads
}), recursive = FALSE)
payload_paths <- vapply(payload_entries, function(entry) {
  file.path(manifest_dir, entry$relative_path)
}, character(1L))
if (all(file.exists(payload_paths))) {
  for (index in seq_along(payload_entries)) {
    entry <- payload_entries[[index]]
    path <- payload_paths[[index]]
    assert_true(identical(sha256_file(path), entry$sha256) &&
                  identical(as.numeric(file.info(path)$size),
                            as.numeric(entry$bytes)),
                paste("strict CI payload changed:", entry$relative_path))
  }
  cat("strict CI payload hash validation: PASS\n")
} else {
  cat("SKIP strict CI payload hashes: persistent local payloads unavailable\n")
}

cat("test_strict_ci_methods_351x48_manifest.R: PASS\n")
