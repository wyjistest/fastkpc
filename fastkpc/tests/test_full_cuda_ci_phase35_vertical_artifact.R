source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")

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
  cat("SKIP full CUDA CI Phase 3.5D artifact validation\n")
  quit(save = "no", status = 0L)
}

artifact_dir <- file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "phase35_vertical_v1"
)
if (!dir.exists(artifact_dir)) {
  status <- system2(
    "Rscript",
    c("fastkpc/tools/run_full_cuda_ci_phase35_vertical.R", artifact_dir)
  )
  assert_true(status == 0L, "Phase 3.5D artifact generation must succeed")
}
validated <- fastkpc_full_cuda_phase35_validate_vertical_artifact(artifact_dir)
assert_true(
  identical(validated$summary$claim_scope,
            "phase3.5D-structural-only") &&
    isTRUE(validated$summary$structural_pass) &&
    !isTRUE(validated$summary$full_graph_claim) &&
    !isTRUE(validated$summary$promotion_authority),
  "validated artifact must retain its narrow non-promotion claim"
)

tampered_dir <- tempfile("fastkpc-phase35d-tamper-")
dir.create(tampered_dir)
on.exit(unlink(tampered_dir, recursive = TRUE, force = TRUE), add = TRUE)
files <- list.files(artifact_dir, full.names = TRUE)
assert_true(
  all(file.copy(files, tampered_dir, overwrite = TRUE)),
  "artifact tamper fixture copy must succeed"
)
case_path <- file.path(tampered_dir, "case_results.csv")
case_results <- read.csv(case_path, stringsAsFactors = FALSE,
                         check.names = FALSE)
case_results$candidate_p_value[[1L]] <- 0.5
write.csv(case_results, case_path, row.names = FALSE, na = "")
assert_error(
  fastkpc_full_cuda_phase35_validate_vertical_artifact(tampered_dir),
  "vertical payload file hash mismatch: case_results.csv",
  "semantic payload tampering must fail closed"
)

cat(
  "PASS full CUDA CI Phase 3.5D artifact validation; producer=",
  validated$producer$identity_sha256, "\n", sep = ""
)
