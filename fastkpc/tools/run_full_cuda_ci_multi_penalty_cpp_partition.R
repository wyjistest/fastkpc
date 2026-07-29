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

read_integer <- function(name, default = NULL) {
  raw <- Sys.getenv(name, unset = "")
  if (!nzchar(raw)) {
    if (!is.null(default)) return(as.integer(default))
    stop(name, " is required", call. = FALSE)
  }
  value <- suppressWarnings(as.integer(raw))
  if (length(value) != 1L || is.na(value)) {
    stop(name, " must be one integer", call. = FALSE)
  }
  value
}

partition_id <- read_integer("FASTKPC_PHASE5_PARTITION_ID")
partition_count <- read_integer("FASTKPC_PHASE5_PARTITION_COUNT")
run_dcov <- read_integer("FASTKPC_PHASE5_RUN_DCOV", 1L) == 1L
preserve_transcripts <-
  read_integer("FASTKPC_PHASE5_PRESERVE_TRANSCRIPTS", 1L) == 1L
output_path <- Sys.getenv("FASTKPC_PHASE5_PARTITION_OUTPUT", unset = "")
if (!nzchar(output_path)) {
  stop("FASTKPC_PHASE5_PARTITION_OUTPUT is required", call. = FALSE)
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
evidence <- fastkpc_full_cuda_phase5_scan_partition(
  catalog,
  run_dcov = run_dcov,
  partition_id = partition_id,
  partition_count = partition_count,
  preserve_transcripts = preserve_transcripts,
  progress = TRUE
)
if (!isTRUE(evidence$summary$pass)) {
  print(evidence$summary)
  stop("Phase 5 partition gate failed", call. = FALSE)
}
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
temporary <- paste0(output_path, ".tmp-", Sys.getpid())
on.exit(unlink(temporary, force = TRUE), add = TRUE)
saveRDS(evidence, temporary, compress = "gzip", version = 3L)
if (!file.rename(temporary, output_path)) {
  stop("Phase 5 partition publication rename failed", call. = FALSE)
}
cat(
  "PASS Phase 5 partition", partition_id, "/", partition_count,
  "setups=", evidence$summary$setup_count,
  "targets=", evidence$summary$target_count,
  "logical=", evidence$summary$logical_test_count,
  "elapsed=", format(evidence$summary$elapsed_seconds, digits = 8L),
  "output=", output_path, "\n"
)
