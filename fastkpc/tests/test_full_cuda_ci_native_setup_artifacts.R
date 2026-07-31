source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_shadow.R")
source("fastkpc/R/full_cuda_ci_phase3_artifacts.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_single_penalty_gcv.R")
source("fastkpc/R/full_cuda_ci_phase4_artifacts.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cpp.R")
source("fastkpc/R/full_cuda_ci_phase5_artifacts.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cuda.R")
source("fastkpc/R/full_cuda_ci_phase6_artifacts.R")
source("fastkpc/R/full_cuda_ci_native_setup.R")
source("fastkpc/R/full_cuda_ci_phase7_artifacts.R")
source("fastkpc/R/full_cuda_ci_phase7_publication.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_error <- function(expression, pattern, message) {
  error <- tryCatch(force(expression), error = identity)
  assert_true(
    inherits(error, "error") &&
      grepl(pattern, conditionMessage(error), fixed = TRUE),
    paste0(message, if (inherits(error, "error")) {
      paste0(": ", conditionMessage(error))
    } else {
      ": no error"
    })
  )
}

catalog <- fastkpc_full_cuda_open_fixed_sp_catalog(
  file.path("fastkpc", "artifacts", "full_cuda_ci", "oracle_351x48_v1"),
  file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "workload_census_351x48_v1"
  ),
  file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1"
  ),
  file.path(
    "fastkpc", "artifacts", "kpc_tprs_real_zhu",
    "cancer_RD-causalDiscoveryInput.rds"
  ),
  require_full = TRUE
)
root <- Sys.getenv(
  "FASTKPC_PHASE7_ARTIFACT_ROOT",
  unset = file.path("fastkpc", "artifacts", "full_cuda_ci")
)
directories <- fastkpc_full_cuda_phase7_artifact_directory_names()
paths <- file.path(root, directories)
names(paths) <- names(directories)
assert_true(
  all(dir.exists(paths)), "Phase 7 artifacts must exist before validation"
)
validated <- lapply(names(paths), function(kind) {
  fastkpc_full_cuda_phase7_validate_artifact(
    paths[[kind]], expected_kind = kind, catalog = catalog,
    verify_current_sources = TRUE
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
    isTRUE(validated$backend$summary$native_setup_authoritative) &&
    !isTRUE(validated$backend$summary$shadow_comparison),
  "Phase 7 artifact identities or authority claims differ"
)
required_files <- sort(
  fastkpc_full_cuda_phase7_required_files(), method = "radix"
)
assert_true(
  all(vapply(paths, function(path) {
    identical(
      sort(list.files(path, all.files = FALSE, no.. = TRUE),
           method = "radix"),
      required_files
    )
  }, logical(1L))) &&
    all(vapply(validated, function(value) {
      summary <- value$summary
      !isTRUE(summary$timeout) &&
        summary$unknown_fallback_count == 0L &&
        summary$approximate_backend_count == 0L &&
        summary$backend_fallback_error_count == 0L &&
        is.finite(summary$elapsed_sec) && summary$elapsed_sec > 0
    }, logical(1L))),
  "Phase 7 standard artifact files or summary fields are incomplete"
)

copy_artifact <- function(kind) {
  destination <- tempfile(paste0("fastkpc-phase7-", kind, "-tamper-"))
  dir.create(destination)
  files <- list.files(paths[[kind]], full.names = TRUE)
  assert_true(
    all(file.copy(files, destination, overwrite = TRUE)),
    paste0("Phase 7 ", kind, " tamper fixture copy must succeed")
  )
  destination
}

run_tamper_tests <- function() {
  temporary_paths <- character()
  on.exit(
    unlink(temporary_paths, recursive = TRUE, force = TRUE),
    add = TRUE
  )

  oracle_tampered <- copy_artifact("oracle")
  temporary_paths <- c(temporary_paths, oracle_tampered)
  setup_path <- file.path(oracle_tampered, "setup_results.csv")
  setups <- utils::read.csv(
    setup_path, stringsAsFactors = FALSE, check.names = FALSE
  )
  setups$x_max_absolute_error[[1L]] <- 1
  utils::write.csv(setups, setup_path, row.names = FALSE, na = "")
  assert_error(
    fastkpc_full_cuda_phase7_validate_artifact(
      oracle_tampered, expected_kind = "oracle", catalog = catalog
    ),
    "Phase 7 payload file hash mismatch: setup_results.csv",
    "Phase 7 semantic payload tampering must fail closed"
  )

  standard_tampered <- copy_artifact("full_shadow")
  temporary_paths <- c(temporary_paths, standard_tampered)
  n_edgetests_path <- file.path(standard_tampered, "n_edgetests.csv")
  n_edgetests <- utils::read.csv(
    n_edgetests_path, stringsAsFactors = FALSE, check.names = FALSE
  )
  n_edgetests$candidate[[1L]] <- n_edgetests$candidate[[1L]] + 1L
  utils::write.csv(
    n_edgetests, n_edgetests_path, row.names = FALSE, na = ""
  )
  assert_error(
    fastkpc_full_cuda_phase7_validate_artifact(
      standard_tampered, expected_kind = "full_shadow", catalog = catalog
    ),
    "Phase 7 payload file hash mismatch: n_edgetests.csv",
    "Phase 7 standard graph payload tampering must fail closed"
  )

  shadow_tampered <- copy_artifact("full_shadow")
  temporary_paths <- c(temporary_paths, shadow_tampered)
  manifest_path <- file.path(shadow_tampered, "manifest.json")
  manifest <- jsonlite::read_json(manifest_path, simplifyVector = FALSE)
  manifest$claim_scope <- "forged-phase10-promotion"
  fastkpc_full_cuda_write_json(manifest, manifest_path)
  assert_error(
    fastkpc_full_cuda_phase7_validate_artifact(
      shadow_tampered, expected_kind = "full_shadow", catalog = catalog
    ),
    "Phase 7 artifact manifest schema mismatch",
    "Phase 7 manifest tampering must fail closed"
  )

  backend_tampered <- copy_artifact("backend")
  temporary_paths <- c(temporary_paths, backend_tampered)
  attestation_path <- file.path(
    backend_tampered, "validator_attestations.json"
  )
  attestations <- jsonlite::read_json(
    attestation_path, simplifyVector = FALSE
  )
  attestations$attestations[[1L]]$validation_result <- "FAIL"
  fastkpc_full_cuda_write_json(attestations, attestation_path)
  assert_error(
    fastkpc_full_cuda_phase7_validate_artifact(
      backend_tampered, expected_kind = "backend", catalog = catalog
    ),
    "validator attestation hash mismatch",
    "Phase 7 validator attestation tampering must fail closed"
  )

  unlink(backend_tampered, recursive = TRUE, force = TRUE)
  backend_tampered <- copy_artifact("backend")
  temporary_paths <- c(temporary_paths, backend_tampered)
  receipt_path <- file.path(backend_tampered, "execution_receipts.json")
  receipts <- jsonlite::read_json(receipt_path, simplifyVector = FALSE)
  receipts$execution_receipts[[1L]]$artifact_path <- "/tmp/forged-artifact"
  fastkpc_full_cuda_write_json(receipts, receipt_path)
  assert_error(
    fastkpc_full_cuda_phase7_validate_artifact(
      backend_tampered, expected_kind = "backend", catalog = catalog
    ),
    "execution receipt hash mismatch",
    "Phase 7 execution receipt tampering must fail closed"
  )
}

run_tamper_tests()

cat(
  "PASS Phase 7 native setup artifacts; producers=",
  paste(vapply(
    validated, function(value) value$producer$identity_sha256,
    character(1L)
  ), collapse = ","), "\n", sep = ""
)
