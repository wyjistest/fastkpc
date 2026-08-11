source("fastkpc/R/cuda_native.R")
source("fastkpc/R/native.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cuda.R")

args <- commandArgs(trailingOnly = TRUE)
input_path <- if (length(args) >= 1L) args[[1L]] else {
  "/tmp/fastkpc_causal_large_parity_20260810/residual_route_bakeoff.rds"
}
output_path <- if (length(args) >= 2L) args[[2L]] else {
  "/tmp/fastkpc_gpu_persistent_fixed_sp_evaluator.rds"
}
requested_indices <- if (length(args) >= 3L) {
  as.integer(strsplit(args[[3L]], ",", fixed = TRUE)[[1L]])
} else {
  seq_len(14L)
}
if (!file.exists(input_path)) stop("source receipt is missing", call. = FALSE)
if (file.exists(output_path)) stop("output already exists", call. = FALSE)

data_paths <- c(
  `20230331145508` =
    "causalDiscovery/20230331145508/combined_data_shuffle-causalDiscoveryInput.RData",
  `20240102191119` =
    "causalDiscovery/20240102191119/GSE99254-482col-causalDiscoveryInput.RData",
  `20241215091815` =
    "causalDiscovery/20241215091815/logged-lung-exhausted-allData-causalDiscoveryInput.RData"
)

fixed_evaluator_prepared_from_native <- function(native_setup) {
  geometry <- full_cuda_ci_native_geometry_prepare_native(
    native_setup$X,
    native_setup$penalty_blocks,
    as.integer(native_setup$penalty_offsets),
    as.integer(native_setup$penalty_ranks)
  )
  list(
    schema_version = "full-cuda-ci-fixed-sp-evaluator-setup-v1",
    X = unname(native_setup$X),
    magic_qr_packed = unname(geometry$magic_qr_packed),
    magic_tau = unname(geometry$magic_tau),
    magic_r = unname(geometry$magic_r),
    magic_pivot = as.integer(geometry$magic_pivot),
    penalty_roots = lapply(geometry$penalty_roots, unname),
    penalty_matrices = lapply(geometry$penalty_matrices, unname),
    penalty_ranks = as.integer(native_setup$penalty_ranks),
    initial_log_sp = unname(geometry$initial_log_sp)
  )
}

p_value <- function(residuals) {
  fastkpc_cuda_legacy_dcov_gamma_cpp_oracle(
    residuals[, 1L], residuals[, 2L], numCol = 35L, index = 1
  )$p.value
}

load_fastkpc_cuda_native()
if (!fastkpc_cuda_available()) stop("CUDA is unavailable", call. = FALSE)
source_receipt <- readRDS(input_path)
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
  penalty_count <- length(native_setup$penalty_blocks)
  q <- ncol(native_setup$X)
  prepared <- fixed_evaluator_prepared_from_native(native_setup)
  Y <- unname(data[, source_detail$targets, drop = FALSE])
  log_sp <- log(unname(source_detail$gpu_sp))
  capacity_bucket <- if (q <= 64L && penalty_count <= 7L) {
    "q64-p7"
  } else if (q <= 80L && penalty_count <= 8L) {
    "q80-p8"
  } else {
    "q192-p21"
  }
  if (q <= 80L) {
    handle <- full_cuda_ci_multi_penalty_gcv_prepared_create_native(
      prepared, ncol(Y)
    )
    token <- NULL
    on.exit({
      if (!is.null(token)) {
        try(full_cuda_ci_multi_penalty_gcv_residual_release_native(token),
            silent = TRUE)
        try(full_cuda_ci_multi_penalty_gcv_residual_free_native(token),
            silent = TRUE)
      }
      if (!is.null(handle)) {
        try(full_cuda_ci_multi_penalty_gcv_prepared_free_native(handle),
            silent = TRUE)
      }
    }, add = TRUE)
    target_keys <- paste0("persistent-fixed-", source_index, "-",
                          seq_len(ncol(Y)))
    token <- full_cuda_ci_multi_penalty_gcv_evaluate_fixed_batch_native(
      handle, Y, log_sp, target_keys
    )
    info <- full_cuda_ci_multi_penalty_gcv_residual_info_native(token)
    shadow <- full_cuda_ci_multi_penalty_gcv_residual_shadow_native(token)
    full_cuda_ci_multi_penalty_gcv_residual_release_native(token)
    full_cuda_ci_multi_penalty_gcv_residual_free_native(token)
    token <- NULL
    prepared_after <-
      full_cuda_ci_multi_penalty_gcv_prepared_info_native(handle)
    full_cuda_ci_multi_penalty_gcv_prepared_free_native(handle)
    handle <- NULL
    fixed_log_sp_exact <- TRUE
  } else {
    capacity <- full_cuda_ci_multi_penalty_capacity_qualify_native(
      native_setup, prepared, Y, fixed_log_sp = log_sp
    )
    info <- list(
      solver_info = capacity$fixed_solver_info,
      decomposition_route = capacity$fixed_decomposition_route,
      numerical_rank = capacity$fixed_numerical_rank,
      fixed_sp_evaluation = capacity$fixed_sp_evaluation,
      device_resident = capacity$fixed_device_resident_before_release
    )
    shadow <- list(
      coefficients = capacity$fixed_coefficients,
      fitted = capacity$fixed_fitted,
      residuals = capacity$fixed_residuals
    )
    prepared_after <- list(
      residual_slot_leased = !capacity$fixed_residual_slot_released
    )
    capacity_bucket <- capacity$capacity_bucket
    fixed_log_sp_exact <- identical(capacity$fixed_log_sp, log_sp)
  }
  reference <- unname(source_detail$residuals$native_gpu_sp)
  candidate_p <- p_value(shadow$residuals)
  reference_p <- p_value(reference)
  row <- data.frame(
    source_index = source_index,
    dataset = dataset,
    position = as.integer(source_detail$request$position),
    q = q,
    penalty_count = penalty_count,
    capacity_bucket = capacity_bucket,
    solver_info_zero = all(info$solver_info == 0L),
    stable_svd_route = all(info$decomposition_route == 2L),
    fixed_sp_evaluation = isTRUE(info$fixed_sp_evaluation),
    fixed_log_sp_exact = fixed_log_sp_exact,
    max_residual_error = max(abs(shadow$residuals - reference)),
    residual_bitwise_exact = identical(
      as.double(shadow$residuals), as.double(reference)
    ),
    p_value_error = candidate_p - reference_p,
    contract_passed = abs(candidate_p - reference_p) <= 1e-10,
    stringsAsFactors = FALSE
  )
  rows[[length(rows) + 1L]] <- row
  details[[as.character(source_index)]] <- list(
    info = info,
    prepared_after = prepared_after,
    residuals = shadow$residuals,
    reference_residuals = reference
  )
  print(row, digits = 17, row.names = FALSE)
}

rows <- do.call(rbind, rows)
receipt <- list(
  schema_version = "mgcv-gpu-persistent-fixed-sp-evaluator-v2",
  native_sha256 = unname(digest::digest(
    file = "fastkpc/build/fastkpc_cuda.so", algo = "sha256",
    serialize = FALSE
  )),
  source_receipt = normalizePath(input_path),
  rows = rows,
  details = details,
  summary = list(
    case_count = nrow(rows),
    solver_pass_count = sum(rows$solver_info_zero),
    stable_svd_route_count = sum(rows$stable_svd_route),
    fixed_log_sp_exact_count = sum(rows$fixed_log_sp_exact),
    contract_pass_count = sum(rows$contract_passed),
    max_residual_error = max(rows$max_residual_error),
    max_abs_p_value_error = max(abs(rows$p_value_error)),
    all_slots_released = all(vapply(details, function(value) {
      !isTRUE(value$prepared_after$residual_slot_leased)
    }, logical(1L)))
  )
)
saveRDS(receipt, output_path, compress = "xz", version = 3L)
print(receipt$summary, digits = 17)
