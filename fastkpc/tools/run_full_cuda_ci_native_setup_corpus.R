source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_shadow.R")
source("fastkpc/R/full_cuda_ci_phase3_artifacts.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_native_setup.R")

output_dir <- Sys.getenv(
  "FASTKPC_PHASE7_SETUP_CORPUS_DIR",
  unset = "/tmp/fastkpc-phase7-native-setup-corpus-v1"
)
merged_path <- Sys.getenv(
  "FASTKPC_PHASE7_SETUP_CORPUS_MERGED",
  unset = file.path(output_dir, "source_evidence.rds")
)
rebuild_oracle <- identical(
  Sys.getenv("FASTKPC_PHASE7_REBUILD_ORACLE", unset = "1"), "1"
)
thread_variables <- c(
  "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"
)
thread_values <- Sys.getenv(thread_variables, unset = "")
if (any(thread_values != "1")) {
  stop(
    paste(thread_variables, collapse = ", "),
    " must all equal 1 for the Phase 7 oracle rebuild",
    call. = FALSE
  )
}
requested_text <- Sys.getenv("FASTKPC_PHASE7_SETUP_SHARD_IDS", unset = "")
requested <- if (nzchar(requested_text)) {
  values <- suppressWarnings(as.integer(strsplit(
    requested_text, ",", fixed = TRUE
  )[[1L]]))
  if (anyNA(values) || anyDuplicated(values) ||
      any(values < 0L | values > 63L)) {
    stop("FASTKPC_PHASE7_SETUP_SHARD_IDS is malformed", call. = FALSE)
  }
  sort(values)
} else {
  0:63
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
identity <- fastkpc_full_cuda_phase7_execution_identity(
  catalog, "native-setup-oracle"
)
identity_json <- fastkpc_full_cuda_phase35_canonical_json(identity)
preparation <- .fastkpc_full_cuda_prepared_s_prepare_shard_read(
  inputs = catalog$inputs,
  shard_count = catalog$catalog_contract$shard_count,
  expected_source_commit = catalog$phase2_manifest$source_commit
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

write_atomic <- function(value, path) {
  temporary <- paste0(path, ".tmp-", Sys.getpid())
  on.exit(unlink(temporary, force = TRUE), add = TRUE)
  saveRDS(value, temporary, compress = "gzip", version = 3L)
  if (!file.rename(temporary, path)) {
    stop("Phase 7 atomic publication rename failed", call. = FALSE)
  }
  invisible(path)
}

for (shard_id in requested) {
  path <- file.path(output_dir, sprintf("shard_%02d.rds", shard_id))
  reusable <- FALSE
  if (file.exists(path)) {
    prior <- tryCatch(readRDS(path), error = function(error) NULL)
    reusable <- is.list(prior) && identical(
      prior$schema_version,
      "full-cuda-ci-native-setup-shard-evidence-v1"
    ) && identical(prior$shard_id, as.integer(shard_id)) &&
      identical(prior$rebuild_oracle, rebuild_oracle) &&
      identical(
        fastkpc_full_cuda_phase35_canonical_json(prior$execution_identity),
        identity_json
      )
  }
  if (isTRUE(reusable)) {
    cat("REUSE Phase 7 native setup shard ", shard_id, ": ", path,
        "\n", sep = "")
    next
  }
  evidence <- fastkpc_full_cuda_phase7_scan_setup_shard(
    catalog = catalog, shard_id = shard_id,
    rebuild_oracle = rebuild_oracle, progress = TRUE,
    preparation = preparation, execution_identity = identity
  )
  write_atomic(evidence, path)
  cat(
    "PASS Phase 7 native setup shard ", shard_id,
    " setups=", nrow(evidence$rows),
    " elapsed=", format(evidence$elapsed_seconds, digits = 8L),
    " output=", path, "\n", sep = ""
  )
}

paths <- file.path(output_dir, sprintf("shard_%02d.rds", 0:63))
if (all(file.exists(paths))) {
  merged <- fastkpc_full_cuda_phase7_merge_setup_shards(catalog, paths)
  write_atomic(merged, merged_path)
  cat(
    "PASS Phase 7 native setup corpus setups=",
    merged$summary$setup_count,
    " unsupported=", merged$summary$unsupported_count,
    " output=", merged_path, "\n", sep = ""
  )
} else {
  cat(
    "PASS Phase 7 requested setup shards; merge pending ",
    sum(file.exists(paths)), "/64\n", sep = ""
  )
}
