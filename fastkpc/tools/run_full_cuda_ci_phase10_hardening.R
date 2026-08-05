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

output_dir <- Sys.getenv(
  "FASTKPC_PHASE10_HARDENING_ARTIFACT_DIR",
  unset = fastkpc_full_cuda_phase10_hardening_artifact_dir()
)
parent <- dirname(output_dir)
dir.create(parent, recursive = TRUE, showWarnings = FALSE)
evidence_dir <- tempfile(".phase10-hardening-evidence-", tmpdir = parent)
dir.create(evidence_dir, recursive = TRUE)
on.exit(unlink(evidence_dir, recursive = TRUE, force = TRUE), add = TRUE)
hardening_path <- file.path(evidence_dir, "hardening.rds")
stream_path <- file.path(evidence_dir, "streams.rds")

base_env <- c(
  "CUDA_VISIBLE_DEVICES=0",
  "OPENBLAS_NUM_THREADS=1",
  "OMP_NUM_THREADS=1",
  "MKL_NUM_THREADS=1",
  "BLIS_NUM_THREADS=1",
  "VECLIB_MAXIMUM_THREADS=1"
)
test_rows <- list()
test_logs <- character()

run_test <- function(label, script, extra_env = character()) {
  command_text <- paste(
    c(base_env, extra_env, "Rscript", script), collapse = " "
  )
  started <- proc.time()[["elapsed"]]
  output <- suppressWarnings(system2(
    "Rscript", script,
    stdout = TRUE, stderr = TRUE,
    env = c(base_env, extra_env)
  ))
  elapsed <- proc.time()[["elapsed"]] - started
  status <- attr(output, "status", exact = TRUE)
  if (is.null(status)) status <- 0L
  passed <- identical(as.integer(status), 0L)
  test_rows[[length(test_rows) + 1L]] <<- data.frame(
    test = label,
    command = command_text,
    elapsed_sec = as.numeric(elapsed),
    pass = passed,
    stringsAsFactors = FALSE
  )
  test_logs <<- c(
    test_logs,
    paste0("===== ", label, " ====="),
    paste0("command: ", command_text),
    paste0("status: ", status, " elapsed_sec: ", elapsed),
    output,
    ""
  )
  cat(
    "Phase 10 hardening test ", label,
    ": status=", status, " elapsed_sec=", elapsed, "\n", sep = ""
  )
  if (!passed) {
    cat(paste(output, collapse = "\n"), "\n")
    stop("Phase 10 hardening test failed: ", label, call. = FALSE)
  }
  invisible(output)
}

run_test(
  "one-call-hardening",
  "fastkpc/tests/test_full_cuda_ci_phase10_hardening.R",
  paste0("FASTKPC_PHASE10_HARDENING_EVIDENCE_RDS=", hardening_path)
)
run_test(
  "stream-count-determinism",
  "fastkpc/tests/test_full_cuda_ci_phase10_stream_determinism.R",
  c(
    "FASTKPC_RUN_CUDA_TESTS=1",
    paste0("FASTKPC_PHASE10_STREAM_EVIDENCE_RDS=", stream_path)
  )
)
run_test(
  "one-call-cache",
  "fastkpc/tests/test_full_cuda_ci_one_call_cache.R"
)
run_test(
  "one-call-regression",
  "fastkpc/tests/test_full_cuda_ci_one_call.R"
)
run_test(
  "native-setup-authority",
  "fastkpc/tests/test_full_cuda_ci_native_setup_no_mgcv.R"
)
run_test(
  "dcov-authority",
  "fastkpc/tests/test_full_cuda_ci_dcov_cuda.R"
)
run_test(
  "semantic-abi",
  "fastkpc/tests/test_full_cuda_ci_phase35_semantic_abi.R"
)
run_test(
  "default-inf-level8-qualification",
  "fastkpc/tests/test_full_cuda_ci_default_inf_level8.R",
  "FASTKPC_RUN_CUDA_TESTS=1"
)
run_test(
  "default-inf-extended-capacity",
  "fastkpc/tests/test_full_cuda_ci_default_inf_level9_capacity.R",
  "FASTKPC_RUN_CUDA_TESTS=1"
)
run_test(
  "default-inf-full-regression",
  "fastkpc/tests/test_full_cuda_ci_default_inf_production.R",
  "FASTKPC_RUN_FULL_DEFAULT_INF_CUDA_TEST=1"
)

fastkpc_full_cuda_phase10_hardening_require(
  file.exists(hardening_path) && file.exists(stream_path),
  "Phase 10 hardening tests did not produce structured evidence"
)
test_results <- do.call(rbind, test_rows)
artifact <- fastkpc_full_cuda_phase10_publish_hardening(
  hardening_path = hardening_path,
  stream_path = stream_path,
  test_results = test_results,
  test_logs = test_logs,
  output_dir = output_dir
)
cat(
  "PASS Phase 10 hardening artifact: ", output_dir,
  " producer=", artifact$producer$identity_sha256,
  " fail_closed=", artifact$summary$fail_closed_case_count,
  " streams=", artifact$summary$stream_counts,
  "\n", sep = ""
)
