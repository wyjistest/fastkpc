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
  "FASTKPC_PHASE10_CAMPAIGN_ARTIFACT_DIR",
  unset = fastkpc_full_cuda_phase10_campaign_artifact_dir()
)
if (!file.exists(file.path(artifact_dir, "manifest.json"))) {
  cat("SKIP Phase 10 canonical campaign artifact: not published yet\n")
  quit(save = "no", status = 0L)
}
validated <- fastkpc_full_cuda_phase10_validate_campaign_artifact(
  artifact_dir,
  verify_current_sources = identical(
    Sys.getenv(
      "FASTKPC_PHASE10_CAMPAIGN_VERIFY_CURRENT_SOURCES", unset = "1"
    ),
    "1"
  )
)
assert_true(
  isTRUE(validated$summary$pass) &&
    validated$summary$SHD == 0L &&
    validated$summary$candidate_warm_repetitions == 5L &&
    validated$summary$warm_median_sec <= 120 &&
    validated$summary$candidate_to_baseline_ratio <= 0.80 &&
    !isTRUE(validated$summary$holdout_gate) &&
    !isTRUE(validated$summary$phase10_promotion_claim),
  "Phase 10 canonical campaign artifact summary is invalid"
)

linked_copy <- function(label, detached_file) {
  destination <- tempfile(
    paste0(".phase10-campaign-", label, "-"),
    tmpdir = dirname(normalizePath(artifact_dir, mustWork = TRUE))
  )
  dir.create(destination)
  source_files <- list.files(artifact_dir, full.names = TRUE)
  assert_true(
    all(file.link(
      source_files, file.path(destination, basename(source_files))
    )),
    paste0("Phase 10 ", label, " tamper fixture link failed")
  )
  source_path <- file.path(artifact_dir, detached_file)
  destination_path <- file.path(destination, detached_file)
  unlink(destination_path)
  assert_true(
    file.copy(source_path, destination_path, copy.mode = FALSE),
    paste0("Phase 10 ", label, " tamper fixture detach failed")
  )
  destination
}

payload <- linked_copy("payload", "raw_runs.csv")
on.exit(unlink(payload, recursive = TRUE, force = TRUE), add = TRUE)
raw_path <- file.path(payload, "raw_runs.csv")
raw <- utils::read.csv(raw_path, stringsAsFactors = FALSE)
raw$elapsed_sec[[1L]] <- raw$elapsed_sec[[1L]] + 1
utils::write.csv(raw, raw_path, row.names = FALSE, na = "")
assert_error(
  fastkpc_full_cuda_phase10_validate_campaign_artifact(payload),
  "Phase 10 campaign artifact payload identity mismatch",
  "Phase 10 campaign payload tampering must fail closed"
)

claim <- linked_copy("claim", "manifest.json")
on.exit(unlink(claim, recursive = TRUE, force = TRUE), add = TRUE)
manifest_path <- file.path(claim, "manifest.json")
manifest <- jsonlite::read_json(manifest_path, simplifyVector = FALSE)
manifest$claim_scope <- "forged-phase10-promotion"
fastkpc_full_cuda_write_json(manifest, manifest_path)
assert_error(
  fastkpc_full_cuda_phase10_validate_campaign_artifact(claim),
  "Phase 10 campaign artifact manifest schema mismatch",
  "Phase 10 campaign claim tampering must fail closed"
)

cat("PASS Phase 10 campaign artifact identity and tamper gates\n")
