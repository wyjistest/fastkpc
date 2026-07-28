source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")
source("fastkpc/R/full_cuda_ci_phase35_feasibility.R")

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

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP full CUDA CI Phase 3.5 feasibility artifact validation\n")
  quit(save = "no", status = 0L)
}

artifact_dir <- file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "phase35_feasibility_v1"
)
if (!dir.exists(artifact_dir)) {
  evidence_path <- "/tmp/fastkpc-phase35-prepublication-evidence.rds"
  assert_true(
    file.exists(evidence_path),
    "Phase 3.5 prepublication evidence is required to generate the artifact"
  )
  status <- system2(
    "Rscript",
    c(
      "fastkpc/tools/run_full_cuda_ci_phase35_feasibility.R",
      paste0("--evidence=", evidence_path), artifact_dir
    )
  )
  assert_true(status == 0L, "Phase 3.5 feasibility generation must succeed")
}

validated <- fastkpc_full_cuda_phase35_validate_feasibility_artifact(
  artifact_dir
)
validated_current <- fastkpc_full_cuda_phase35_validate_feasibility_artifact(
  artifact_dir, verify_current_sources = TRUE
)
assert_true(
  isTRUE(validated$summary$phase8_go) &&
    !isTRUE(validated$summary$production_backend_promoted) &&
    !isTRUE(validated$summary$phase10_promotion_claim) &&
    !isTRUE(validated$summary$full_graph_claim) &&
    !isTRUE(validated$summary$universal_p_value_parity_claim) &&
    validated$summary$full_conditional_final_flip_count == 0L &&
    validated$summary$conservative_dcov_upper_bound_ms <= 47000 &&
    validated$summary$full_campaign_used_or_reserved_ms <= 120000 &&
    identical(
      validated_current$producer$identity_sha256,
      validated$producer$identity_sha256
    ),
  "validated feasibility artifact must retain its narrow GO claim"
)

tampered_dir <- tempfile("fastkpc-phase35-feasibility-tamper-")
dir.create(tampered_dir)
on.exit(unlink(tampered_dir, recursive = TRUE, force = TRUE), add = TRUE)
files <- list.files(artifact_dir, full.names = TRUE)
assert_true(
  all(file.copy(files, tampered_dir, overwrite = TRUE)),
  "feasibility artifact tamper fixture copy must succeed"
)

candidate_path <- file.path(tampered_dir, "candidate_decisions.csv")
candidate <- read.csv(
  candidate_path, stringsAsFactors = FALSE, check.names = FALSE
)
candidate$phase8_go[candidate$phase8_go] <- FALSE
write.csv(candidate, candidate_path, row.names = FALSE, na = "")
assert_error(
  fastkpc_full_cuda_phase35_validate_feasibility_artifact(tampered_dir),
  "feasibility payload file hash mismatch: candidate_decisions.csv",
  "semantic payload tampering must fail before interpretation"
)

assert_true(
  file.copy(
    file.path(artifact_dir, "candidate_decisions.csv"), candidate_path,
    overwrite = TRUE
  ),
  "candidate payload restoration must succeed"
)
manifest_path <- file.path(tampered_dir, "manifest.json")
manifest <- jsonlite::read_json(manifest_path, simplifyVector = FALSE)
manifest$claim_scope <- "forged-production-promotion"
fastkpc_full_cuda_write_json(manifest, manifest_path)
assert_error(
  fastkpc_full_cuda_phase35_validate_feasibility_artifact(tampered_dir),
  "feasibility artifact manifest schema mismatch",
  "manifest claim-scope tampering must fail closed"
)

cat(
  "PASS full CUDA CI Phase 3.5 feasibility artifact; producer=",
  validated$producer$identity_sha256, "\n", sep = ""
)
