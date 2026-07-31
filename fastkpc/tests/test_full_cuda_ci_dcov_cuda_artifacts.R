source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_shadow.R")
source("fastkpc/R/full_cuda_ci_phase3_artifacts.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_phase35_bakeoff.R")
source("fastkpc/R/full_cuda_ci_single_penalty_gcv.R")
source("fastkpc/R/full_cuda_ci_phase4_artifacts.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cpp.R")
source("fastkpc/R/full_cuda_ci_phase5_artifacts.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cuda.R")
source("fastkpc/R/full_cuda_ci_phase6_artifacts.R")
source("fastkpc/R/full_cuda_ci_native_setup.R")
source("fastkpc/R/full_cuda_ci_phase7_artifacts.R")
source("fastkpc/R/full_cuda_ci_phase7_publication.R")
source("fastkpc/R/full_cuda_ci_phase8_dcov.R")
source("fastkpc/R/full_cuda_ci_phase8_publication.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_error <- function(expression, pattern, message) {
  error <- tryCatch(force(expression), error = identity)
  assert_true(
    inherits(error, "error") &&
      grepl(pattern, conditionMessage(error), fixed = TRUE),
    paste0(message, if (inherits(error, "error")) {
      paste0(": ", conditionMessage(error))
    } else ": no error")
  )
}

root <- Sys.getenv(
  "FASTKPC_PHASE8_ARTIFACT_ROOT",
  unset = fastkpc_full_cuda_phase8_artifact_root()
)
directories <- fastkpc_full_cuda_phase8_artifact_directories()
paths <- file.path(root, directories)
names(paths) <- names(directories)
assert_true(
  all(dir.exists(paths)), "Phase 8 artifacts must exist before validation"
)
validated <- lapply(names(paths), function(kind) {
  fastkpc_full_cuda_phase8_validate_artifact(
    paths[[kind]], expected_kind = kind, verify_current_sources = TRUE
  )
})
names(validated) <- names(paths)
assert_true(
  all(vapply(validated, function(value) isTRUE(value$summary$pass),
             logical(1L))) &&
    length(unique(vapply(
      validated,
      function(value) value$producer$native_binary_sha256,
      character(1L)
    ))) == 1L &&
    length(unique(vapply(
      validated,
      function(value) value$producer$producer_source_closure_sha256,
      character(1L)
    ))) == 1L &&
    isTRUE(validated$backend$summary$backend_authoritative) &&
    !isTRUE(validated$backend$summary$shadow_comparison),
  "Phase 8 artifact identities or authority claims differ"
)
required <- sort(fastkpc_full_cuda_phase8_required_files(), method = "radix")
assert_true(
  all(vapply(paths, function(path) {
    identical(
      sort(list.files(path, all.files = FALSE, no.. = TRUE),
           method = "radix"),
      required
    )
  }, logical(1L))),
  "Phase 8 standard artifact file set is incomplete"
)

copy_artifact <- function(kind) {
  destination <- tempfile(paste0("fastkpc-phase8-", kind, "-tamper-"))
  dir.create(destination)
  files <- list.files(paths[[kind]], full.names = TRUE)
  assert_true(
    all(file.copy(files, destination, overwrite = TRUE)),
    paste0("Phase 8 ", kind, " tamper fixture copy must succeed")
  )
  destination
}

run_tamper_tests <- function() {
  temporary_paths <- character()
  on.exit(
    unlink(temporary_paths, recursive = TRUE, force = TRUE),
    add = TRUE
  )

  numerical <- copy_artifact("component_oracle")
  temporary_paths <- c(temporary_paths, numerical)
  cases_path <- file.path(numerical, "case_results.csv")
  cases <- utils::read.csv(
    cases_path, stringsAsFactors = FALSE, check.names = FALSE
  )
  cases$final_p_value[[1L]] <- 1
  utils::write.csv(cases, cases_path, row.names = FALSE, na = "")
  assert_error(
    fastkpc_full_cuda_phase8_validate_artifact(
      numerical, expected_kind = "component_oracle"
    ),
    "Phase 8 payload file hash mismatch",
    "Phase 8 numerical payload tampering must fail closed"
  )

  graph <- copy_artifact("full_shadow")
  temporary_paths <- c(temporary_paths, graph)
  n_edgetests_path <- file.path(graph, "n_edgetests.csv")
  n_edgetests <- utils::read.csv(
    n_edgetests_path, stringsAsFactors = FALSE, check.names = FALSE
  )
  n_edgetests$candidate[[1L]] <- n_edgetests$candidate[[1L]] + 1L
  utils::write.csv(n_edgetests, n_edgetests_path,
                   row.names = FALSE, na = "")
  assert_error(
    fastkpc_full_cuda_phase8_validate_artifact(
      graph, expected_kind = "full_shadow"
    ),
    "Phase 8 payload file hash mismatch",
    "Phase 8 graph payload tampering must fail closed"
  )

  manifest_tampered <- copy_artifact("full_shadow")
  temporary_paths <- c(temporary_paths, manifest_tampered)
  manifest_path <- file.path(manifest_tampered, "manifest.json")
  manifest <- jsonlite::read_json(manifest_path, simplifyVector = FALSE)
  manifest$claim_scope <- "forged-phase10-promotion"
  fastkpc_full_cuda_write_json(manifest, manifest_path)
  assert_error(
    fastkpc_full_cuda_phase8_validate_artifact(
      manifest_tampered, expected_kind = "full_shadow"
    ),
    "Phase 8 artifact manifest schema mismatch",
    "Phase 8 manifest tampering must fail closed"
  )

  attestation_tampered <- copy_artifact("backend")
  temporary_paths <- c(temporary_paths, attestation_tampered)
  attestation_path <- file.path(
    attestation_tampered, "validator_attestations.json"
  )
  attestations <- jsonlite::read_json(
    attestation_path, simplifyVector = FALSE
  )
  attestations$attestations[[1L]]$validation_result <- "FAIL"
  fastkpc_full_cuda_write_json(attestations, attestation_path)
  assert_error(
    fastkpc_full_cuda_phase8_validate_artifact(
      attestation_tampered, expected_kind = "backend"
    ),
    "validator attestation hash mismatch",
    "Phase 8 attestation tampering must fail closed"
  )

  receipt_tampered <- copy_artifact("backend")
  temporary_paths <- c(temporary_paths, receipt_tampered)
  receipt_path <- file.path(receipt_tampered, "execution_receipts.json")
  receipts <- jsonlite::read_json(receipt_path, simplifyVector = FALSE)
  receipts$execution_receipts[[1L]]$artifact_path <- "/tmp/forged-artifact"
  fastkpc_full_cuda_write_json(receipts, receipt_path)
  assert_error(
    fastkpc_full_cuda_phase8_validate_artifact(
      receipt_tampered, expected_kind = "backend"
    ),
    "execution receipt hash mismatch",
    "Phase 8 receipt tampering must fail closed"
  )
}

run_tamper_tests()
cat(
  "PASS Phase 8 CUDA dCov artifacts; producers=",
  paste(vapply(
    validated, function(value) value$producer$identity_sha256,
    character(1L)
  ), collapse = ","), "\n", sep = ""
)
