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
source("fastkpc/R/full_cuda_ci_phase4_backend.R")
source("fastkpc/R/full_cuda_ci_phase4_publication.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cpp.R")
source("fastkpc/R/full_cuda_ci_phase5_artifacts.R")

partition_dir <- Sys.getenv("FASTKPC_PHASE5_PARTITION_DIR", unset = "")
output_path <- Sys.getenv("FASTKPC_PHASE5_MERGED_OUTPUT", unset = "")
if (!nzchar(partition_dir) || !dir.exists(partition_dir)) {
  stop("FASTKPC_PHASE5_PARTITION_DIR must be an existing directory",
       call. = FALSE)
}
if (!nzchar(output_path)) {
  stop("FASTKPC_PHASE5_MERGED_OUTPUT is required", call. = FALSE)
}
paths <- list.files(
  partition_dir, pattern = "^partition_[0-9]+\\.rds$", full.names = TRUE
)
paths <- sort(paths, method = "radix")
if (length(paths) == 0L) {
  stop("Phase 5 partition directory is empty", call. = FALSE)
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
merged <- fastkpc_full_cuda_phase5_merge_partitions(catalog, paths)
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
temporary <- paste0(output_path, ".tmp-", Sys.getpid())
on.exit(unlink(temporary, force = TRUE), add = TRUE)
saveRDS(merged, temporary, compress = "gzip", version = 3L)
if (!file.rename(temporary, output_path)) {
  stop("Phase 5 merged evidence publication rename failed", call. = FALSE)
}
cat(
  "PASS Phase 5 merged shadow setups=", merged$summary$setup_count,
  " targets=", merged$summary$target_count,
  " logical=", merged$summary$logical_test_count,
  " SHD=", merged$mixed_graph$summary$SHD,
  " fallback=", merged$mixed_graph$summary$explicit_legacy_fallback_count,
  " output=", output_path, "\n", sep = ""
)
