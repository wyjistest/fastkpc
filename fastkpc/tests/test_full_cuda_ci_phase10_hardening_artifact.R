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
source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase9_artifact.R")
source("fastkpc/R/full_cuda_ci_phase10_hardening.R")

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

artifact_dir <- Sys.getenv(
  "FASTKPC_PHASE10_HARDENING_ARTIFACT_DIR",
  unset = fastkpc_full_cuda_phase10_hardening_artifact_dir()
)
validated <- fastkpc_full_cuda_phase10_validate_hardening_artifact(
  artifact_dir, verify_current_sources = TRUE
)
assert_true(
  isTRUE(validated$summary$pass) &&
    isTRUE(validated$summary$hardening_gate) &&
    validated$summary$fail_closed_case_count >= 10L &&
    identical(validated$summary$stream_counts, "1|2|4") &&
    identical(validated$summary$max_conditioning_size_requested, "Inf") &&
    validated$summary$max_conditioning_size_resolved == 46L &&
    validated$summary$natural_stop_level == 8L &&
    validated$summary$logical_test_count == 240498L &&
    validated$summary$level8_test_count == 9L &&
    isTRUE(validated$summary$default_inf_full_regression_pass) &&
    isTRUE(validated$summary$default_inf_level8_qualification_pass) &&
    isTRUE(
      validated$summary$default_inf_extended_capacity_qualification_pass
    ) &&
    validated$summary$tracked_resource_leak_count == 0L,
  "Phase 10 hardening artifact summary is not promotable evidence"
)

level7_claim <- validated$summary
level7_claim$natural_stop_level <- 7L
assert_error(
  fastkpc_full_cuda_phase10_hardening_validate_summary(level7_claim),
  "Phase 10 hardening summary gate failed",
  "Phase 10 hardening must reject a max-7 natural-stop claim"
)

copy_artifact <- function(label) {
  destination <- tempfile(paste0("phase10-hardening-", label, "-"))
  dir.create(destination)
  files <- list.files(artifact_dir, full.names = TRUE)
  assert_true(
    all(file.copy(files, destination, overwrite = TRUE)),
    paste0("Phase 10 ", label, " tamper fixture copy failed")
  )
  destination
}

payload <- copy_artifact("payload")
on.exit(unlink(payload, recursive = TRUE, force = TRUE), add = TRUE)
cache_path <- file.path(payload, "cache.csv")
cache <- utils::read.csv(cache_path, stringsAsFactors = FALSE)
cache$pass[[1L]] <- FALSE
utils::write.csv(cache, cache_path, row.names = FALSE, na = "")
assert_error(
  fastkpc_full_cuda_phase10_validate_hardening_artifact(payload),
  "Phase 10 hardening artifact payload identity mismatch",
  "Phase 10 hardening payload tampering must fail closed"
)

claim <- copy_artifact("claim")
on.exit(unlink(claim, recursive = TRUE, force = TRUE), add = TRUE)
manifest_path <- file.path(claim, "manifest.json")
manifest <- jsonlite::read_json(manifest_path, simplifyVector = FALSE)
manifest$claim_scope <- "forged-phase10-promotion"
fastkpc_full_cuda_write_json(manifest, manifest_path)
assert_error(
  fastkpc_full_cuda_phase10_validate_hardening_artifact(claim),
  "Phase 10 hardening artifact manifest schema mismatch",
  "Phase 10 hardening claim-scope tampering must fail closed"
)

cat("PASS Phase 10 hardening artifact identity and tamper gates\n")
