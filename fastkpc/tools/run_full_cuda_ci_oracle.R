source("fastkpc/R/full_cuda_ci_gate.R")

args <- commandArgs(trailingOnly = TRUE)

arg_or_env <- function(index, env_name, default) {
  if (length(args) >= index && nzchar(args[[index]])) return(args[[index]])
  Sys.getenv(env_name, unset = default)
}

data_path <- arg_or_env(
  1L,
  "FASTKPC_FULL_CUDA_CI_DATA_PATH",
  file.path("fastkpc", "artifacts", "kpc_tprs_real_zhu",
            "cancer_RD-causalDiscoveryInput.rds")
)
oracle_result_path <- arg_or_env(
  2L,
  "FASTKPC_FULL_CUDA_CI_ORACLE_RESULT_PATH",
  file.path(
    "fastkpc", "artifacts", "legacy_mgcv_residual_cache_s_affinity_v1",
    "compatible_legacy_cpp_dcov_mgcv_cache_s_affinity_result.rds"
  )
)
candidate_result_path <- arg_or_env(
  3L,
  "FASTKPC_FULL_CUDA_CI_CANDIDATE_RESULT_PATH",
  file.path("fastkpc", "artifacts", "full_cuda_ci",
            "trace_source_351x48_v1", "result.rds")
)
output_root <- arg_or_env(
  4L,
  "FASTKPC_FULL_CUDA_CI_OUTPUT_ROOT",
  file.path("fastkpc", "artifacts", "full_cuda_ci")
)
logical_trace_result_path <- arg_or_env(
  5L,
  "FASTKPC_FULL_CUDA_CI_LOGICAL_TRACE_RESULT_PATH",
  file.path("fastkpc", "artifacts", "full_cuda_ci",
            "trace_source_351x48_v1", "result.rds")
)
oracle_dir <- file.path(output_root, "oracle_351x48_v1")
comparison_dir <- file.path(
  output_root,
  "current_correct_route_351x48_v1"
)
fastkpc_full_cuda_reset_output_dir(comparison_dir)

required_paths <- c(
  data_path,
  oracle_result_path,
  if (nzchar(logical_trace_result_path)) logical_trace_result_path
)
missing <- required_paths[!file.exists(required_paths)]
if (length(missing) > 0L) {
  stop("full CUDA CI oracle runner missing input: ",
       paste(missing, collapse = ", "), call. = FALSE)
}

data <- readRDS(data_path)
oracle_result <- readRDS(oracle_result_path)
oracle_skeleton <- fastkpc_full_cuda_extract_skeleton(
  oracle_result, role = "oracle"
)
if (!fastkpc_full_cuda_is_skeleton(oracle_skeleton)) {
  stop("oracle result does not contain a skeleton", call. = FALSE)
}

config <- oracle_result$config
alpha <- as.numeric(fastkpc_full_cuda_or(config$alpha, 0.1))
max_conditioning_size <- as.integer(fastkpc_full_cuda_or(
  config$max_conditioning_size,
  length(oracle_skeleton$n.edgetests) - 1L
))
index <- as.integer(fastkpc_full_cuda_or(config$index, 1L))
numCol <- as.integer(floor(nrow(as.matrix(data)) / 10L))
invisible(fastkpc_full_cuda_validate_canonical_fixture(
  data = data,
  skeleton = oracle_skeleton,
  alpha = alpha,
  index = index,
  numCol = numCol,
  max_conditioning_size = max_conditioning_size,
  source_result_path = oracle_result_path
))

invocation_environment <- c(
  FASTKPC_FULL_CUDA_CI_DATA_PATH = data_path,
  FASTKPC_FULL_CUDA_CI_ORACLE_RESULT_PATH = oracle_result_path,
  FASTKPC_FULL_CUDA_CI_CANDIDATE_RESULT_PATH = candidate_result_path,
  FASTKPC_FULL_CUDA_CI_OUTPUT_ROOT = output_root,
  FASTKPC_FULL_CUDA_CI_LOGICAL_TRACE_RESULT_PATH =
    logical_trace_result_path
)
invocation <- paste(
  paste0(names(invocation_environment), "=",
         vapply(invocation_environment, shQuote, character(1))),
  collapse = " "
)
invocation <- paste(invocation,
                    "Rscript fastkpc/tools/run_full_cuda_ci_oracle.R")
route_environment <- c(
  FASTKPC_LEGACY_DCOV_GAMMA_BACKEND = "cpp",
  FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = "spectra",
  FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE = "1",
  FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY = "s"
)

oracle <- fastkpc_write_full_cuda_ci_oracle(
  reference = oracle_skeleton,
  data = data,
  output_dir = oracle_dir,
  alpha = alpha,
  index = index,
  numCol = numCol,
  max_conditioning_size = max_conditioning_size,
  source_result_path = oracle_result_path,
  logical_trace_source = if (nzchar(logical_trace_result_path)) {
    logical_trace_result_path
  } else NULL,
  logical_trace_source_path = if (nzchar(logical_trace_result_path)) {
    logical_trace_result_path
  } else NA_character_,
  oracle_route_environment = route_environment,
  commands = invocation
)

comparison <- fastkpc_compare_full_cuda_ci_candidate(
  oracle = oracle,
  candidate = if (file.exists(candidate_result_path)) {
    candidate_result_path
  } else NULL,
  output_dir = comparison_dir,
  candidate_route = if (file.exists(candidate_result_path)) {
    "native-one-call-threaded-round-dcov"
  } else {
    "missing-candidate"
  },
  commands = invocation
)

summary <- comparison$summary
cat("full CUDA CI Phase 0 gate\n")
cat("oracle: ", oracle_dir, "\n", sep = "")
cat("comparison: ", comparison_dir, "\n", sep = "")
cat("edge_count: ", summary$edge_count_candidate, " / ",
    summary$edge_count_reference, "\n", sep = "")
cat("SHD: ", summary$SHD, "\n", sep = "")
cat("adjacency_identical: ", summary$adjacency_identical, "\n", sep = "")
cat("sepsets_identical: ", summary$sepsets_identical, "\n", sep = "")
cat("n_edgetests_identical: ", summary$n_edgetests_identical, "\n",
    sep = "")
cat("deletions_identical: ", summary$deletions_identical, "\n", sep = "")
cat("pass: ", summary$pass, "\n", sep = "")

if (!isTRUE(summary$pass)) quit(status = 1L)
