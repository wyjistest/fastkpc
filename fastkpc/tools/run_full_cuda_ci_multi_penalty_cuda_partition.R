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
source("fastkpc/R/full_cuda_ci_multi_penalty_cuda.R")
source("fastkpc/R/full_cuda_ci_phase6_artifacts.R")

read_integer <- function(name, default = NULL) {
  raw <- Sys.getenv(name, unset = "")
  if (!nzchar(raw)) {
    if (!is.null(default)) return(as.integer(default))
    return(NULL)
  }
  value <- suppressWarnings(as.integer(raw))
  if (length(value) != 1L || is.na(value)) {
    stop(name, " must be one integer", call. = FALSE)
  }
  value
}

partition_id <- read_integer("FASTKPC_PHASE6_PARTITION_ID")
partition_count <- read_integer("FASTKPC_PHASE6_PARTITION_COUNT")
max_setups <- read_integer("FASTKPC_PHASE6_MAX_SETUPS")
run_dcov <- read_integer("FASTKPC_PHASE6_RUN_DCOV", 1L) == 1L
concurrency <- read_integer("FASTKPC_PHASE6_CONCURRENCY", 32L)
output_path <- Sys.getenv("FASTKPC_PHASE6_PARTITION_OUTPUT", unset = "")
if (!nzchar(output_path)) {
  stop("FASTKPC_PHASE6_PARTITION_OUTPUT is required", call. = FALSE)
}
phase5_evidence_path <- Sys.getenv(
  "FASTKPC_PHASE6_PHASE5_EVIDENCE",
  unset = file.path(
    "fastkpc", "artifacts", "full_cuda_ci",
    "multi_penalty_cpp_full_shadow_v1", "source_evidence.rds"
  )
)
if (!file.exists(phase5_evidence_path)) {
  stop("FASTKPC_PHASE6_PHASE5_EVIDENCE is missing", call. = FALSE)
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
phase5_evidence <- readRDS(phase5_evidence_path)
evidence <- fastkpc_full_cuda_phase6_scan_partition(
  catalog,
  phase5_evidence = phase5_evidence,
  phase5_evidence_path = phase5_evidence_path,
  max_setups = max_setups,
  run_dcov = run_dcov,
  partition_id = partition_id,
  partition_count = partition_count,
  concurrency = concurrency,
  progress = TRUE
)
if (!isTRUE(evidence$summary$pass)) {
  print(evidence$summary)
  mismatched <- evidence$targets[
    evidence$targets$selected_log_sp_max_error > 1e-6 |
      evidence$targets$optimizer_iteration_mismatch |
      evidence$targets$score_call_mismatch |
      evidence$targets$objective_call_mismatch |
      evidence$targets$step_halving_mismatch |
      evidence$targets$boundary_probe_mismatch |
      evidence$targets$boundary_status_mismatch |
      evidence$targets$convergence_mismatch |
      evidence$targets$hessian_state_mismatch |
      evidence$targets$rank_mismatch,
    , drop = FALSE
  ]
  if (nrow(mismatched) > 0L) print(mismatched, row.names = FALSE)
  stop("Phase 6 partition gate failed", call. = FALSE)
}
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
temporary <- paste0(output_path, ".tmp-", Sys.getpid())
on.exit(unlink(temporary, force = TRUE), add = TRUE)
saveRDS(evidence, temporary, compress = "gzip", version = 3L)
if (!file.rename(temporary, output_path)) {
  stop("Phase 6 partition publication rename failed", call. = FALSE)
}
cat(
  "PASS Phase 6 partition", evidence$partition$partition_id, "/",
  evidence$partition$partition_count,
  " setups=", evidence$summary$setup_count,
  " targets=", evidence$summary$target_count,
  " logical=", evidence$summary$logical_test_count,
  " elapsed=", format(evidence$summary$elapsed_seconds, digits = 8L),
  " output=", output_path, "\n", sep = ""
)
