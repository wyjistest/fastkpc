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
source("fastkpc/R/full_cuda_ci_phase10_campaign.R")
source("fastkpc/R/full_cuda_ci_phase10_holdout.R")

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

artifact_dir <- fastkpc_full_cuda_phase10_holdout_artifact_dir()
if (!file.exists(file.path(artifact_dir, "manifest.json"))) {
  cat("SKIP Phase 10 sealed holdout artifact: not released yet\n")
  quit(save = "no", status = 0L)
}
validated <- fastkpc_full_cuda_phase10_validate_holdout_artifact(
  artifact_dir, verify_current_sources = TRUE
)
assert_true(
  isTRUE(validated$summary$pass) &&
    validated$summary$maximum_SHD == 0L &&
    validated$summary$decision_flip_count == 0L &&
    isTRUE(validated$summary$coverage_gate) &&
    isTRUE(validated$summary$phase10_promotion_claim) &&
    isTRUE(validated$summary$recommended_compatible_cuda),
  "Phase 10 sealed holdout artifact is not promotion evidence"
)

destination <- tempfile(
  ".phase10-holdout-tamper-", tmpdir = dirname(artifact_dir)
)
dir.create(destination)
on.exit(unlink(destination, recursive = TRUE, force = TRUE), add = TRUE)
source_files <- list.files(artifact_dir, full.names = TRUE)
assert_true(
  all(file.link(
    source_files, file.path(destination, basename(source_files))
  )),
  "Phase 10 holdout tamper fixture link failed"
)
coverage_path <- file.path(destination, "coverage.csv")
unlink(coverage_path)
assert_true(
  file.copy(
    file.path(artifact_dir, "coverage.csv"), coverage_path,
    copy.mode = FALSE
  ),
  "Phase 10 holdout tamper fixture detach failed"
)
coverage <- utils::read.csv(coverage_path, stringsAsFactors = FALSE)
coverage$covered[[1L]] <- FALSE
utils::write.csv(coverage, coverage_path, row.names = FALSE, na = "")
assert_error(
  fastkpc_full_cuda_phase10_validate_holdout_artifact(destination),
  "Phase 10 holdout artifact payload identity mismatch",
  "Phase 10 holdout payload tampering must fail closed"
)

cat("PASS Phase 10 sealed holdout artifact identity and tamper gates\n")
