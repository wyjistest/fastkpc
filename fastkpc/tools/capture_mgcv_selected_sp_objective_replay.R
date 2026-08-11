source("fastkpc/R/legacy_runner.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/native.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cpp.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cuda.R")

args <- commandArgs(trailingOnly = TRUE)
source_path <- if (length(args) >= 1L) args[[1L]] else {
  "/tmp/fastkpc_causal_large_parity_20260810/residual_route_bakeoff.rds"
}
output_path <- if (length(args) >= 2L) args[[2L]] else {
  "/tmp/fastkpc_causal_large_parity_20260810/selected_sp_objective_replay.rds"
}
source_index <- if (length(args) >= 3L) as.integer(args[[3L]]) else 1L
target_index <- if (length(args) >= 4L) as.integer(args[[4L]]) else 1L

if (!file.exists(source_path)) stop("source receipt is missing", call. = FALSE)
if (file.exists(output_path)) stop("output already exists", call. = FALSE)
if (is.na(source_index) || source_index <= 0L ||
    is.na(target_index) || target_index <= 0L) {
  stop("source or target index is invalid", call. = FALSE)
}

data_paths <- c(
  `20230331145508` =
    "causalDiscovery/20230331145508/combined_data_shuffle-causalDiscoveryInput.RData",
  `20240102191119` =
    "causalDiscovery/20240102191119/GSE99254-482col-causalDiscoveryInput.RData",
  `20241215091815` =
    "causalDiscovery/20241215091815/logged-lung-exhausted-allData-causalDiscoveryInput.RData"
)

prepared_setup_from_native <- function(native_setup) {
  penalty_count <- length(native_setup$penalty_blocks)
  list(
    formula_class = "additive-smooth",
    X = unname(native_setup$X),
    gram_matrix = unname(native_setup$gram_matrix),
    penalty_blocks = setNames(
      lapply(native_setup$penalty_blocks, unname),
      paste0("penalty_", seq_len(penalty_count))
    ),
    penalty_offsets = as.integer(native_setup$penalty_offsets),
    penalty_ranks = as.integer(native_setup$penalty_ranks),
    mgcv_penalty_rank_metadata = as.integer(native_setup$penalty_ranks),
    constraint = matrix(
      numeric(), nrow = 0L, ncol = ncol(native_setup$X)
    ),
    H = NULL,
    weights = NULL
  )
}

source_receipt <- readRDS(source_path)
if (source_index > length(source_receipt$details)) {
  stop("source index exceeds receipt", call. = FALSE)
}
detail <- source_receipt$details[[source_index]]
if (target_index > length(detail$targets)) {
  stop("target index exceeds source target count", call. = FALSE)
}
dataset <- as.character(detail$request$dataset)
payload <- new.env(parent = emptyenv())
object_names <- load(data_paths[[dataset]], envir = payload)
if (length(object_names) != 1L) stop("data payload is malformed", call. = FALSE)
data <- as.matrix(payload[[object_names[[1L]]]])
storage.mode(data) <- "double"
Y <- unname(data[, detail$targets, drop = FALSE])
native_setup <- full_cuda_ci_native_setup_native(
  data[, detail$conditioning_set, drop = FALSE]
)
setup <- prepared_setup_from_native(native_setup)
prepared <- fastkpc_full_cuda_phase6_prepare(setup)

reference <- fastkpc_full_cuda_phase5_optimize_cpp(
  setup, Y[, target_index], keep_transcript = TRUE
)
transcript <- reference$transcript
point_log_sp <- do.call(cbind, lapply(transcript, function(entry) {
  if (identical(entry$stage, "iteration_state")) {
    as.double(entry$current_log_sp)
  } else {
    as.double(entry$trial_log_sp)
  }
}))
point_y <- matrix(
  rep(Y[, target_index], length(transcript)),
  nrow = nrow(Y), ncol = length(transcript)
)
load_fastkpc_cuda_native()
gpu <- fastkpc_full_cuda_phase6_evaluate_cuda(
  prepared, point_y, point_log_sp
)

receipt <- list(
  schema_version = "mgcv-selected-sp-objective-replay-v1",
  native_sha256 = unname(digest::digest(
    file = "fastkpc/build/fastkpc_cuda.so", algo = "sha256",
    serialize = FALSE
  )),
  source_path = normalizePath(source_path),
  source_index = source_index,
  target_index = target_index,
  reference = reference,
  cpp_transcript = transcript,
  point_log_sp = point_log_sp,
  gpu = gpu
)
saveRDS(receipt, output_path, compress = "xz", version = 3L)
cat(sprintf(
  "source=%d target=%d transcript=%d selected-log-sp-error=% .3e\n",
  source_index, target_index, length(transcript),
  max(abs(reference$selected_log_sp -
    log(detail$native_cpu_sp[, target_index])))
))
