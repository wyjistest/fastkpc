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
source("fastkpc/R/full_cuda_ci_multi_penalty_cpp.R")
source("fastkpc/R/full_cuda_ci_phase5_artifacts.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cuda.R")
source("fastkpc/R/full_cuda_ci_phase6_artifacts.R")
source("fastkpc/R/full_cuda_ci_native_setup.R")
source("fastkpc/R/full_cuda_ci_phase7_artifacts.R")

setup_corpus_path <- Sys.getenv(
  "FASTKPC_PHASE7_SETUP_CORPUS_MERGED",
  unset = file.path(
    "/tmp", "fastkpc-phase7-native-setup-corpus-v1",
    "source_evidence.rds"
  )
)
phase4_dir <- Sys.getenv(
  "FASTKPC_PHASE7_PHASE4_PARTITION_DIR",
  unset = "/tmp/fastkpc-phase7-native-phase4-partitions-v1"
)
phase4_count <- suppressWarnings(as.integer(Sys.getenv(
  "FASTKPC_PHASE7_PHASE4_PARTITION_COUNT", unset = "16"
)))
phase6_dir <- Sys.getenv(
  "FASTKPC_PHASE7_PHASE6_PARTITION_DIR",
  unset = "/tmp/fastkpc-phase7-native-phase6-partitions-v1"
)
phase5_evidence_path <- Sys.getenv(
  "FASTKPC_PHASE7_PHASE5_EVIDENCE",
  unset = file.path(
    "fastkpc", "artifacts", "full_cuda_ci",
    "multi_penalty_cpp_full_shadow_v1", "source_evidence.rds"
  )
)
output_path <- Sys.getenv(
  "FASTKPC_PHASE7_MERGED_EVIDENCE",
  unset = "/tmp/fastkpc-phase7-native-setup-full-evidence-v1.rds"
)
if (is.na(phase4_count) || phase4_count < 1L) {
  stop("FASTKPC_PHASE7_PHASE4_PARTITION_COUNT is malformed",
       call. = FALSE)
}
phase4_paths <- file.path(
  phase4_dir,
  sprintf(
    "partition-%03d-of-%03d.rds", 0:(phase4_count - 1L), phase4_count
  )
)
phase6_paths <- sort(list.files(
  phase6_dir, pattern = "^partition_[0-9]+\\.rds$", full.names = TRUE
), method = "radix")
if (!file.exists(setup_corpus_path) ||
    !all(file.exists(phase4_paths)) || length(phase6_paths) == 0L ||
    !file.exists(phase5_evidence_path)) {
  stop("Phase 7 merge inputs are incomplete", call. = FALSE)
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
evidence <- fastkpc_full_cuda_phase7_build_full_evidence(
  catalog = catalog,
  setup_corpus_path = setup_corpus_path,
  phase4_partition_paths = phase4_paths,
  phase6_partition_paths = phase6_paths,
  phase5_evidence_path = phase5_evidence_path
)
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
temporary <- paste0(output_path, ".tmp-", Sys.getpid())
on.exit(unlink(temporary, force = TRUE), add = TRUE)
saveRDS(evidence, temporary, compress = "gzip", version = 3L)
if (!file.rename(temporary, output_path)) {
  stop("Phase 7 merged evidence publication rename failed", call. = FALSE)
}
cat(
  "PASS Phase 7 native setup full evidence setups=",
  evidence$summary$setup_count,
  " targets=", evidence$summary$target_count,
  " logical=", evidence$summary$logical_test_count,
  " SHD=", evidence$summary$SHD,
  " output=", output_path, "\n", sep = ""
)
