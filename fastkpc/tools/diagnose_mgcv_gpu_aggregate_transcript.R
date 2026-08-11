source("fastkpc/R/legacy_runner.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/native.R")

args <- commandArgs(trailingOnly = TRUE)
input_path <- if (length(args) >= 1L) args[[1L]] else {
  "/tmp/fastkpc_causal_large_parity_20260810/residual_route_bakeoff.rds"
}
output_path <- if (length(args) >= 2L) args[[2L]] else {
  "/tmp/fastkpc_causal_large_parity_20260810/aggregate_transcript.rds"
}
requested_indices <- if (length(args) >= 3L) {
  as.integer(strsplit(args[[3L]], ",", fixed = TRUE)[[1L]])
} else {
  # The standalone transcript is instantiated for q <= 80. The q=82
  # witness is qualified through the ext192 evaluator path instead.
  c(seq_len(8L), 10L, 11L, 14L)
}

if (!file.exists(input_path)) stop("source receipt is missing", call. = FALSE)
if (file.exists(output_path)) stop("output already exists", call. = FALSE)
if (anyNA(requested_indices) || any(requested_indices <= 0L)) {
  stop("witness indices must be positive integers", call. = FALSE)
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
    X = unname(native_setup$X),
    penalty_blocks = setNames(
      lapply(native_setup$penalty_blocks, unname),
      paste0("penalty_", seq_len(penalty_count))
    ),
    penalty_offsets = as.integer(native_setup$penalty_offsets),
    penalty_ranks = as.integer(native_setup$penalty_ranks)
  )
}

cuda_geometry_from_setup <- function(setup) {
  full_cuda_ci_native_geometry_prepare_native(
    setup$X, setup$penalty_blocks, setup$penalty_offsets,
    setup$penalty_ranks
  )
}

assemble_penalty <- function(penalty_matrices, multipliers) {
  q <- nrow(penalty_matrices[[1L]])
  value <- matrix(0, q, q)
  for (penalty in seq_along(penalty_matrices)) {
    value[] <- value[] +
      as.double(multipliers[[penalty]]) * penalty_matrices[[penalty]][]
  }
  value
}

factor_penalty <- function(aggregate) {
  factor <- suppressWarnings(chol(aggregate, pivot = TRUE, tol = -1))
  rank <- as.integer(attr(factor, "rank"))
  pivot <- as.integer(attr(factor, "pivot"))
  root <- matrix(0, rank, ncol(aggregate))
  root[, pivot] <- factor[seq_len(rank), , drop = FALSE]
  list(factor = factor, rank = rank, pivot = pivot, root = root)
}

maximum_absolute_error <- function(left, right) {
  if (!identical(dim(left), dim(right)) && length(left) != length(right)) {
    return(Inf)
  }
  max(abs(as.double(left) - as.double(right)))
}

source_receipt <- readRDS(input_path)
if (any(requested_indices > length(source_receipt$details))) {
  stop("witness index exceeds source receipt", call. = FALSE)
}
load_fastkpc_cuda_native()
if (!fastkpc_cuda_available()) stop("CUDA is unavailable", call. = FALSE)

loaded_data <- new.env(parent = emptyenv())
rows <- list()
details <- list()
for (source_index in requested_indices) {
  source_detail <- source_receipt$details[[source_index]]
  dataset <- as.character(source_detail$request$dataset)
  if (!exists(dataset, loaded_data, inherits = FALSE)) {
    payload <- new.env(parent = emptyenv())
    object_names <- load(data_paths[[dataset]], envir = payload)
    if (length(object_names) != 1L) {
      stop("data payload must contain one object", call. = FALSE)
    }
    data <- as.matrix(payload[[object_names[[1L]]]])
    storage.mode(data) <- "double"
    assign(dataset, data, loaded_data)
  }
  data <- get(dataset, loaded_data, inherits = FALSE)
  native_setup <- full_cuda_ci_native_setup_native(
    data[, source_detail$conditioning_set, drop = FALSE]
  )
  geometry <- cuda_geometry_from_setup(
    prepared_setup_from_native(native_setup)
  )
  penalty_matrices <- geometry$penalty_matrices
  if (length(penalty_matrices) <= 1L) next

  target_details <- vector("list", ncol(source_detail$gpu_sp))
  for (target in seq_len(ncol(source_detail$gpu_sp))) {
    sp <- as.double(source_detail$gpu_sp[, target])
    log_sp <- log(sp)
    gpu <-
      full_cuda_ci_multi_penalty_gcv_augmented_transcript_for_test(
        geometry$magic_r, penalty_matrices, log_sp
      )
    aggregate_sp <- assemble_penalty(penalty_matrices, sp)
    aggregate_gpu_multiplier <- assemble_penalty(
      penalty_matrices, gpu$penalty_multipliers
    )
    factor_sp <- factor_penalty(aggregate_sp)
    factor_gpu_multiplier <- factor_penalty(aggregate_gpu_multiplier)
    q <- nrow(aggregate_sp)
    upper <- upper.tri(aggregate_sp, diag = TRUE)

    rows[[length(rows) + 1L]] <- data.frame(
      source_index = source_index,
      dataset = dataset,
      position = as.integer(source_detail$request$position),
      target = as.integer(source_detail$targets[[target]]),
      q = q,
      penalty_count = length(sp),
      max_multiplier_error = max(abs(gpu$penalty_multipliers - sp)),
      multiplier_bitwise_exact = identical(
        as.double(gpu$penalty_multipliers), sp
      ),
      max_aggregate_error_vs_sp = maximum_absolute_error(
        gpu$aggregate_penalty, aggregate_sp
      ),
      max_aggregate_error_vs_gpu_multiplier = maximum_absolute_error(
        gpu$aggregate_penalty, aggregate_gpu_multiplier
      ),
      aggregate_bitwise_exact_vs_sp = identical(
        as.double(gpu$aggregate_penalty), as.double(aggregate_sp)
      ),
      aggregate_bitwise_exact_vs_gpu_multiplier = identical(
        as.double(gpu$aggregate_penalty),
        as.double(aggregate_gpu_multiplier)
      ),
      rank_sp = factor_sp$rank,
      rank_gpu_multiplier = factor_gpu_multiplier$rank,
      rank_gpu = as.integer(gpu$aggregate_penalty_rank),
      pivot_exact_vs_sp = identical(
        as.integer(gpu$aggregate_pivot), factor_sp$pivot
      ),
      pivot_exact_vs_gpu_multiplier = identical(
        as.integer(gpu$aggregate_pivot), factor_gpu_multiplier$pivot
      ),
      max_factor_upper_error_vs_sp = max(abs(
        gpu$aggregate_factor[upper] - factor_sp$factor[upper]
      )),
      max_factor_upper_error_vs_gpu_multiplier = max(abs(
        gpu$aggregate_factor[upper] -
          factor_gpu_multiplier$factor[upper]
      )),
      max_root_error_vs_sp = maximum_absolute_error(
        gpu$aggregate_root, factor_sp$root
      ),
      max_root_error_vs_gpu_multiplier = maximum_absolute_error(
        gpu$aggregate_root, factor_gpu_multiplier$root
      ),
      stringsAsFactors = FALSE
    )
    target_details[[target]] <- list(
      sp = sp,
      log_sp = log_sp,
      gpu = gpu,
      aggregate_sp = aggregate_sp,
      aggregate_gpu_multiplier = aggregate_gpu_multiplier,
      factor_sp = factor_sp,
      factor_gpu_multiplier = factor_gpu_multiplier
    )
  }
  details[[as.character(source_index)]] <- target_details
}

rows <- do.call(rbind, rows)
receipt <- list(
  schema_version = "mgcv-gpu-aggregate-transcript-v1",
  native_sha256 = unname(digest::digest(
    file = "fastkpc/build/fastkpc_cuda.so", algo = "sha256",
    serialize = FALSE
  )),
  source_receipt = normalizePath(input_path),
  rows = rows,
  details = details,
  summary = list(
    target_count = nrow(rows),
    multiplier_bitwise_pass_count = sum(rows$multiplier_bitwise_exact),
    aggregate_vs_sp_bitwise_pass_count =
      sum(rows$aggregate_bitwise_exact_vs_sp),
    aggregate_vs_gpu_multiplier_bitwise_pass_count =
      sum(rows$aggregate_bitwise_exact_vs_gpu_multiplier),
    pivot_vs_sp_pass_count = sum(rows$pivot_exact_vs_sp),
    pivot_vs_gpu_multiplier_pass_count =
      sum(rows$pivot_exact_vs_gpu_multiplier),
    max_multiplier_error = max(rows$max_multiplier_error),
    max_aggregate_error_vs_sp = max(rows$max_aggregate_error_vs_sp),
    max_aggregate_error_vs_gpu_multiplier =
      max(rows$max_aggregate_error_vs_gpu_multiplier),
    max_factor_upper_error_vs_gpu_multiplier =
      max(rows$max_factor_upper_error_vs_gpu_multiplier),
    max_root_error_vs_gpu_multiplier =
      max(rows$max_root_error_vs_gpu_multiplier)
  )
)
saveRDS(receipt, output_path, compress = "xz", version = 3L)
print(rows, digits = 17, row.names = FALSE)
print(receipt$summary, digits = 17)
