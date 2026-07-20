started <- proc.time()[["elapsed"]]
timings <- list()

raw_env_value <- function(name, default) {
  value <- Sys.getenv(name, unset = NA_character_)
  if (is.na(value)) default else value
}

env_path <- function(name, default) {
  value <- raw_env_value(name, default)
  if (!is.character(value) || length(value) != 1L || is.na(value) ||
      !nzchar(value)) {
    stop(name, " must be one nonempty path", call. = FALSE)
  }
  value
}

env_device <- function(name, default) {
  value <- raw_env_value(name, as.character(default))
  if (!grepl("^(0|[1-9][0-9]*)$", value)) {
    stop(name, " must be one non-negative integer", call. = FALSE)
  }
  numeric_value <- suppressWarnings(as.numeric(value))
  if (!is.finite(numeric_value) || numeric_value > .Machine$integer.max) {
    stop(name, " must be one non-negative integer", call. = FALSE)
  }
  as.integer(numeric_value)
}

timed <- function(stage, expression) {
  stage_started <- proc.time()[["elapsed"]]
  value <- force(expression)
  timings[[length(timings) + 1L]] <<- data.frame(
    stage = stage,
    elapsed_seconds = as.double(
      proc.time()[["elapsed"]] - stage_started
    ),
    stringsAsFactors = FALSE
  )
  value
}

if (length(commandArgs(trailingOnly = TRUE)) != 0L) {
  stop("Phase 3C qualification runner accepts environment variables only",
       call. = FALSE)
}
if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  stop("FASTKPC_RUN_CUDA_TESTS=1 is required for qualification",
       call. = FALSE)
}

phase0_dir <- env_path(
  "FASTKPC_FULL_CUDA_PHASE3_PHASE0_DIR",
  "fastkpc/artifacts/full_cuda_ci/oracle_351x48_v1"
)
phase1_dir <- env_path(
  "FASTKPC_FULL_CUDA_PHASE3_PHASE1_DIR",
  "fastkpc/artifacts/full_cuda_ci/workload_census_351x48_v1"
)
phase2_dir <- env_path(
  "FASTKPC_FULL_CUDA_PHASE3_PHASE2_DIR",
  "fastkpc/artifacts/full_cuda_ci/prepared_s_contract_v1"
)
data_path <- env_path(
  "FASTKPC_FULL_CUDA_PHASE3_DATA",
  paste0(
    "fastkpc/artifacts/kpc_tprs_real_zhu/",
    "cancer_RD-causalDiscoveryInput.rds"
  )
)
output_dir <- env_path(
  "FASTKPC_FULL_CUDA_PHASE3_OUTPUT",
  "fastkpc/artifacts/full_cuda_ci/fixed_sp_cuda_qualification_v1"
)
device_id <- env_device("FASTKPC_FULL_CUDA_PHASE3_DEVICE", 0L)
if (file.exists(output_dir) || dir.exists(output_dir)) {
  stop("qualification output path already exists: ", output_dir,
       call. = FALSE)
}

execution_source_roots <- c(
  qualification_runner =
    "fastkpc/tools/run_full_cuda_ci_fixed_sp_qualification.R",
  workload_census = "fastkpc/R/full_cuda_ci_workload_census.R",
  prepared_s_contract = "fastkpc/R/full_cuda_ci_prepared_s_contract.R",
  fixed_sp_runtime = "fastkpc/R/full_cuda_ci_fixed_sp_runtime.R",
  cuda_native = "fastkpc/R/cuda_native.R"
)
if (!requireNamespace("digest", quietly = TRUE)) {
  stop("qualification execution provenance requires digest", call. = FALSE)
}
bootstrap_runtime_sha256 <- unname(digest::digest(
  file = "fastkpc/R/full_cuda_ci_fixed_sp_runtime.R",
  algo = "sha256", serialize = FALSE
))
base::source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
if (!identical(
  bootstrap_runtime_sha256,
  fastkpc_full_cuda_fixed_sp_sha256_file(
    "fastkpc/R/full_cuda_ci_fixed_sp_runtime.R"
  )
)) {
  stop("qualification runtime changed during provenance bootstrap",
       call. = FALSE)
}
execution_source_closure <-
  fastkpc_full_cuda_fixed_sp_discover_execution_source_closure(
    root_sources = execution_source_roots, project_root = "."
  )
preload_source_sha256 <- vapply(
  execution_source_closure$source_file_paths,
  fastkpc_full_cuda_fixed_sp_sha256_file,
  character(1L)
)
native_build_input_ids <- sort(c(
  "fastkpc/tools/build_cuda_native.sh",
  list.files(
    "fastkpc/src", pattern = "\\.(c|cc|cpp|cu|cuh|h|hpp)$",
    recursive = TRUE, full.names = TRUE
  )
), method = "radix")
native_build_input_paths <- setNames(vapply(
  native_build_input_ids,
  normalizePath, character(1L), winslash = "/", mustWork = TRUE
), native_build_input_ids)
preload_native_build_input_sha256 <- vapply(
  native_build_input_paths,
  fastkpc_full_cuda_fixed_sp_sha256_file,
  character(1L)
)
source <- function(file, ...) {
  fastkpc_full_cuda_fixed_sp_guarded_source(
    file = file, source_closure = execution_source_closure, ...
  )
}
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/cuda_native.R")

native_build_trace_path <- tempfile(
  "fastkpc-phase3c-native-build-", tmpdir = tempdir(), fileext = ".strace"
)
on.exit(unlink(native_build_trace_path, force = TRUE), add = TRUE)
cuda_initialization <- timed("cuda_initialize", {
  native_load <- load_fastkpc_cuda_native_qualified(
    trace_path = native_build_trace_path
  )
  native_build_dependencies <-
    fastkpc_full_cuda_fixed_sp_capture_native_build_dependencies(
      trace_path = native_load$trace_path,
      build_working_dir = ".",
      tracer_path = native_load$tracer_path,
      trace_invocation = native_load$trace_invocation
    )
  if (!fastkpc_cuda_available()) {
    stop("CUDA is unavailable for Phase 3C qualification", call. = FALSE)
  }
  normalized_native_path <- normalizePath(
    native_load$native_library_path, winslash = "/", mustWork = TRUE
  )
  fastkpc_full_cuda_fixed_sp_verify_loaded_native_library(
    normalized_native_path, native_load$native_library_sha256
  )
  list(
    device_info = fastkpc_cuda_device_info(),
    native_library_path = normalized_native_path,
    native_library_sha256 = native_load$native_library_sha256,
    native_build_dependencies = native_build_dependencies
  )
})
cuda_device_info <- cuda_initialization$device_info
execution_provenance <- timed(
  "capture_execution_provenance",
  fastkpc_full_cuda_fixed_sp_capture_execution_provenance(
    source_closure = execution_source_closure,
    expected_source_sha256 = preload_source_sha256,
    native_library_path = cuda_initialization$native_library_path,
    native_build_input_paths = native_build_input_paths,
    expected_native_build_input_sha256 =
      preload_native_build_input_sha256,
    native_build_dependencies =
      cuda_initialization$native_build_dependencies,
    expected_native_library_sha256 =
      cuda_initialization$native_library_sha256
  )
)

qualification_bundle <- timed(
  "qualification_numeric_lifecycle",
  fastkpc_run_full_cuda_fixed_sp_phase3c_qualification(
    phase2_dir = phase0_dir,
    census_dir = phase1_dir,
    prepared_dir = phase2_dir,
    data_path = data_path,
    device_id = device_id,
    return_residual_registry = TRUE
  )
)
if (!is.list(qualification_bundle) || !identical(
      names(qualification_bundle), c("result", "residual_registry")
    ) || !is.environment(qualification_bundle$residual_registry)) {
  stop("qualification residual-registry result is malformed",
       call. = FALSE)
}
qualification <- qualification_bundle$result
residual_registry <- qualification_bundle$residual_registry
dcov_evidence <- timed(
  "qualification_dcov_parity",
  fastkpc_full_cuda_fixed_sp_run_qualification_dcov_parity(
    census_dir = phase1_dir,
    prepared_dir = phase2_dir,
    data_path = data_path,
    phase0_dir = phase0_dir,
    residual_registry = residual_registry,
    target_records = qualification$target_records
  )
)
if (length(ls(residual_registry, all.names = TRUE)) != 0L) {
  stop("qualification residual registry was not cleared after dCov",
       call. = FALSE)
}
qualification_bundle$residual_registry <- NULL
residual_registry <- NULL
gc(verbose = FALSE)
execution_provenance <- timed(
  "verify_execution_provenance",
  fastkpc_full_cuda_fixed_sp_verify_execution_provenance(
    execution_provenance
  )
)

stage_timing <- do.call(rbind, timings)
rownames(stage_timing) <- NULL
final_runtime_index <- match("final", qualification$runtime_records$stage)
if (is.na(final_runtime_index)) {
  stop("qualification final runtime record is missing", call. = FALSE)
}
final_runtime <- qualification$runtime_records[
  final_runtime_index, , drop = FALSE
]
command_lines <- c(
  "FASTKPC_RUN_CUDA_TESTS=1",
  paste0("FASTKPC_FULL_CUDA_PHASE3_DEVICE=", device_id),
  paste0("FASTKPC_FULL_CUDA_PHASE3_PHASE0_DIR=", phase0_dir),
  paste0("FASTKPC_FULL_CUDA_PHASE3_PHASE1_DIR=", phase1_dir),
  paste0("FASTKPC_FULL_CUDA_PHASE3_PHASE2_DIR=", phase2_dir),
  paste0("FASTKPC_FULL_CUDA_PHASE3_DATA=", data_path),
  paste0("FASTKPC_FULL_CUDA_PHASE3_OUTPUT=", output_dir),
  "Rscript fastkpc/tools/run_full_cuda_ci_fixed_sp_qualification.R"
)
source_environment_lines <- unlist(lapply(
  names(execution_provenance$source_file_paths),
  function(source_id) {
    source_path <- execution_provenance$source_file_paths[[source_id]]
    c(
      paste0(
        "source.", source_id, ".path=", source_path
      ),
      paste0(
        "source.", source_id, ".sha256=",
        execution_provenance$source_file_sha256[[source_id]]
      ),
      paste0(
        "source.", source_id, ".git_state=",
        execution_provenance$source_file_git_state[[source_id]]
      )
    )
  }
), use.names = FALSE)
native_build_environment_lines <- unlist(lapply(
  names(execution_provenance$native_build_input_paths),
  function(input_id) c(
    paste0(
      "native_build_input.", input_id, ".path=",
      execution_provenance$native_build_input_paths[[input_id]]
    ),
    paste0(
      "native_build_input.", input_id, ".sha256=",
      execution_provenance$native_build_input_sha256[[input_id]]
    ),
    paste0(
      "native_build_input.", input_id, ".git_state=",
      execution_provenance$native_build_input_git_state[[input_id]]
    )
  )
), use.names = FALSE)
environment_lines <- c(
  paste0("provenance_mode=", execution_provenance$provenance_mode),
  paste0("head_base_commit=", execution_provenance$head_base_commit),
  paste0(
    "source_closure_schema_version=",
    execution_provenance$source_closure_schema_version
  ),
  paste0(
    "source_discovery_semantics=",
    execution_provenance$source_discovery_semantics
  ),
  paste0(
    "source_closure_count=", execution_provenance$source_closure_count
  ),
  paste0(
    "source_closure_sha256=", execution_provenance$source_closure_sha256
  ),
  paste0(
    "execution_snapshot_sha256=",
    execution_provenance$execution_snapshot_sha256
  ),
  paste0(
    "relevant_sources_dirty_or_untracked=",
    as.integer(execution_provenance$relevant_sources_dirty_or_untracked)
  ),
  source_environment_lines,
  native_build_environment_lines,
  paste0(
    "native_build_inputs_sha256=",
    execution_provenance$native_build_inputs_sha256
  ),
  paste0(
    "native_build_dependencies_schema_version=",
    execution_provenance$native_build_dependencies$schema_version
  ),
  paste0(
    "native_build_dependency_trace_semantics=",
    execution_provenance$native_build_dependencies$trace_semantics
  ),
  paste0(
    "native_build_dependency_trace_invocation=",
    execution_provenance$native_build_dependencies$trace_invocation
  ),
  "native_build_trace_path=native_build_trace.txt",
  paste0(
    "native_build_trace_sha256=",
    execution_provenance$native_build_dependencies$trace_sha256
  ),
  paste0(
    "native_build_tracer_path=",
    execution_provenance$native_build_dependencies$tracer_path
  ),
  paste0(
    "native_build_tracer_sha256=",
    execution_provenance$native_build_dependencies$tracer_sha256
  ),
  paste0(
    "native_build_dependency_count=",
    execution_provenance$native_build_dependencies$dependency_count
  ),
  paste0(
    "native_build_exclusion_count=",
    execution_provenance$native_build_dependencies$exclusion_count
  ),
  paste0(
    "native_build_dependencies_sha256=",
    execution_provenance$native_build_dependencies$aggregate_sha256
  ),
  paste0(
    "native_library_path=", execution_provenance$native_library_path
  ),
  paste0(
    "native_library_sha256=", execution_provenance$native_library_sha256
  ),
  paste0("device_id=", device_id),
  paste0("cuda_device_name=", cuda_device_info$name),
  paste0("cuda_compute_capability=", cuda_device_info$compute_capability),
  paste0(
    "cuda_toolkit_version=", final_runtime$cuda_toolkit_version[[1L]]
  ),
  paste0(
    "cuda_driver_version=", final_runtime$cuda_driver_version[[1L]]
  ),
  "qualification_dcov_backend=cpp",
  "qualification_dcov_low_rank_backend=spectra",
  "",
  capture.output(sessionInfo())
)
elapsed_seconds <- as.double(proc.time()[["elapsed"]] - started)
artifact <- fastkpc_full_cuda_write_fixed_sp_qualification_artifact(
  result = qualification,
  output_dir = output_dir,
  phase0_dir = phase0_dir,
  phase1_dir = phase1_dir,
  phase2_dir = phase2_dir,
  data_path = data_path,
  device_id = device_id,
  stage_timing = stage_timing,
  elapsed_seconds = elapsed_seconds,
  command_lines = command_lines,
  environment_lines = environment_lines,
  execution_provenance = execution_provenance,
  dcov_records = dcov_evidence$records
)

summary <- artifact$summary
cat("PASS Phase 3C fixed-sp qualification artifact\n")
cat(
  "counts setups/targets/roots/root_rows/H_roots=",
  summary$setup_count, "/", summary$target_count, "/",
  summary$penalty_root_matrix_count, "/",
  summary$penalty_root_row_count, "/", summary$H_root_matrix_count,
  "\n", sep = ""
)
cat(
  "planned/executed cholesky/qr/svd=",
  summary$planned_cholesky_count, "/", summary$planned_qr_count, "/",
  summary$planned_svd_count, " ", summary$executed_cholesky_count, "/",
  summary$executed_qr_count, "/", summary$executed_svd_count,
  "\n", sep = ""
)
cat(
  "route_status_hash=", summary$route_status_hash, "\n",
  "numeric_hash=", summary$numeric_hash, "\n",
  "dcov_rows_hash=", summary$qualification_dcov_rows_hash, "\n",
  "dcov rows/near_alpha=",
  summary$qualification_dcov_logical_test_count, "/",
  summary$qualification_dcov_near_alpha_count, "\n",
  "dcov max_abs/flips/errors/fallbacks=",
  format(
    summary$qualification_dcov_max_absolute_p_value_difference,
    digits = 17L
  ), "/", summary$qualification_dcov_decision_flip_count, "/",
  summary$qualification_dcov_backend_error_count, "/",
  summary$qualification_dcov_spectra_fallback_count, "\n",
  "elapsed_seconds=", format(summary$elapsed_seconds, digits = 17L),
  "\n", sep = ""
)
