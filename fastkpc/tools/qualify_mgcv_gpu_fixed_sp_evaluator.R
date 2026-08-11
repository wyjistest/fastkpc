source("fastkpc/R/legacy_runner.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/native.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cuda.R")

args <- commandArgs(trailingOnly = TRUE)
input_path <- if (length(args) >= 1L) args[[1L]] else {
  "/tmp/fastkpc_causal_large_parity_20260810/residual_route_bakeoff.rds"
}
output_path <- if (length(args) >= 2L) args[[2L]] else {
  "/tmp/fastkpc_causal_large_parity_20260810/fixed_sp_evaluator_v2.rds"
}
requested_indices <- if (length(args) >= 3L) {
  as.integer(strsplit(args[[3L]], ",", fixed = TRUE)[[1L]])
} else {
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

p_value <- function(residuals) {
  fastkpc_cuda_legacy_dcov_gamma_cpp_oracle(
    residuals[, 1L], residuals[, 2L], numCol = 35L, index = 1
  )$p.value
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
  Y <- unname(data[, source_detail$targets, drop = FALSE])
  native_setup <- full_cuda_ci_native_setup_native(
    data[, source_detail$conditioning_set, drop = FALSE]
  )
  prepared <- fastkpc_full_cuda_phase6_prepare(
    prepared_setup_from_native(native_setup)
  )
  q <- ncol(prepared$X)
  if (q > 80L) next
  SP <- unname(source_detail$gpu_sp)
  evaluation <- fastkpc_full_cuda_phase6_evaluate_cuda(
    prepared, Y, log(SP)
  )
  residuals <- Y - prepared$X %*% evaluation$coefficients
  reference <- unname(source_detail$residuals$native_gpu_sp)
  candidate_p <- p_value(residuals)
  reference_p <- p_value(reference)
  row <- data.frame(
    source_index = source_index,
    dataset = dataset,
    position = as.integer(source_detail$request$position),
    q = q,
    penalty_count = nrow(SP),
    max_residual_error = max(abs(residuals - reference)),
    residual_bitwise_exact = identical(
      as.double(residuals), as.double(reference)
    ),
    reference_p = reference_p,
    candidate_p = candidate_p,
    p_value_error = candidate_p - reference_p,
    contract_passed = abs(candidate_p - reference_p) <= 1e-10,
    cpu_objective_count = evaluation$diagnostics$cpu_objective_count,
    cpu_multi_penalty_solve_count =
      evaluation$diagnostics$cpu_multi_penalty_solve_count,
    fallback_count = evaluation$diagnostics$fallback_count,
    stringsAsFactors = FALSE
  )
  rows[[length(rows) + 1L]] <- row
  details[[as.character(source_index)]] <- list(
    evaluation = evaluation,
    residuals = residuals,
    reference_residuals = reference
  )
  cat(sprintf(
    "source=%d q=%d residual=% .3e p=% .3e pass=%s\n",
    source_index, q, row$max_residual_error, row$p_value_error,
    row$contract_passed
  ))
}

rows <- do.call(rbind, rows)
receipt <- list(
  schema_version = "mgcv-gpu-fixed-sp-evaluator-qualification-v2",
  native_sha256 = unname(digest::digest(
    file = "fastkpc/build/fastkpc_cuda.so", algo = "sha256",
    serialize = FALSE
  )),
  source_receipt = normalizePath(input_path),
  rows = rows,
  details = details,
  summary = list(
    case_count = nrow(rows),
    contract_pass_count = sum(rows$contract_passed),
    max_residual_error = max(rows$max_residual_error),
    max_abs_p_value_error = max(abs(rows$p_value_error)),
    cpu_authority_count = sum(
      rows$cpu_objective_count + rows$cpu_multi_penalty_solve_count
    ),
    fallback_count = sum(rows$fallback_count)
  )
)
saveRDS(receipt, output_path, compress = "xz", version = 3L)
print(rows, digits = 17, row.names = FALSE)
print(receipt$summary, digits = 17)
