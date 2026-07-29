source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_shadow.R")
source("fastkpc/R/full_cuda_ci_phase3_artifacts.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cpp.R")
source("fastkpc/R/full_cuda_ci_phase5_artifacts.R")
source("fastkpc/R/full_cuda_ci_phase5_publication.R")

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

root <- Sys.getenv(
  "FASTKPC_PHASE5_ARTIFACT_ROOT",
  unset = file.path("fastkpc", "artifacts", "full_cuda_ci")
)
directories <- fastkpc_full_cuda_phase5_artifact_directory_names()
paths <- file.path(root, directories)
names(paths) <- names(directories)
assert_true(
  all(dir.exists(paths)), "Phase 5 artifacts must exist before validation"
)

validated <- lapply(names(paths), function(kind) {
  fastkpc_full_cuda_phase5_validate_artifact(
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
    ))) == 1L && length(unique(vapply(
      validated,
      function(value) value$producer$producer_source_closure_sha256,
      character(1L)
    ))) == 1L,
  "Phase 5 artifacts must share one source and native authority"
)

copy_artifact <- function(kind) {
  destination <- tempfile(paste0("fastkpc-phase5-", kind, "-tamper-"))
  dir.create(destination)
  files <- list.files(paths[[kind]], full.names = TRUE)
  assert_true(
    all(file.copy(files, destination, overwrite = TRUE)),
    paste0("Phase 5 ", kind, " tamper fixture copy must succeed")
  )
  destination
}

oracle_tampered <- copy_artifact("oracle")
on.exit(unlink(oracle_tampered, recursive = TRUE, force = TRUE), add = TRUE)
case_path <- file.path(oracle_tampered, "case_results.csv")
cases <- utils::read.csv(
  case_path, stringsAsFactors = FALSE, check.names = FALSE
)
cases$candidate_score[[1L]] <- cases$candidate_score[[1L]] + 1
utils::write.csv(cases, case_path, row.names = FALSE, na = "")
assert_error(
  fastkpc_full_cuda_phase5_validate_artifact(
    oracle_tampered, expected_kind = "oracle"
  ),
  "Phase 5 payload file hash mismatch: case_results.csv",
  "Phase 5 semantic payload tampering must fail closed"
)

shadow_tampered <- copy_artifact("full_shadow")
on.exit(unlink(shadow_tampered, recursive = TRUE, force = TRUE), add = TRUE)
manifest_path <- file.path(shadow_tampered, "manifest.json")
manifest <- jsonlite::read_json(manifest_path, simplifyVector = FALSE)
manifest$claim_scope <- "forged-phase10-promotion"
fastkpc_full_cuda_write_json(manifest, manifest_path)
assert_error(
  fastkpc_full_cuda_phase5_validate_artifact(
    shadow_tampered, expected_kind = "full_shadow"
  ),
  "Phase 5 artifact manifest schema mismatch",
  "Phase 5 manifest claim-scope tampering must fail closed"
)

backend_tampered <- copy_artifact("backend")
on.exit(unlink(backend_tampered, recursive = TRUE, force = TRUE), add = TRUE)
attestation_path <- file.path(
  backend_tampered, "validator_attestations.json"
)
attestations <- jsonlite::read_json(
  attestation_path, simplifyVector = FALSE
)
attestations$attestations[[1L]]$validation_result <- "FAIL"
fastkpc_full_cuda_write_json(attestations, attestation_path)
assert_error(
  fastkpc_full_cuda_phase5_validate_artifact(
    backend_tampered, expected_kind = "backend"
  ),
  "validator attestation hash mismatch",
  "Phase 5 validator attestation tampering must fail closed"
)

unlink(backend_tampered, recursive = TRUE, force = TRUE)
backend_tampered <- copy_artifact("backend")
receipt_path <- file.path(backend_tampered, "execution_receipts.json")
receipts <- jsonlite::read_json(receipt_path, simplifyVector = FALSE)
receipts$execution_receipts[[1L]]$artifact_path <- "/tmp/forged-artifact"
fastkpc_full_cuda_write_json(receipts, receipt_path)
assert_error(
  fastkpc_full_cuda_phase5_validate_artifact(
    backend_tampered, expected_kind = "backend"
  ),
  "execution receipt hash mismatch",
  "Phase 5 volatile receipt tampering must fail closed"
)

cat(
  "PASS Phase 5 multi-penalty C++ artifacts; producers=",
  paste(vapply(
    validated, function(value) value$producer$identity_sha256,
    character(1L)
  ), collapse = ","), "\n", sep = ""
)
