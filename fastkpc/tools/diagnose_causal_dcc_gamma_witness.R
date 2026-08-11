source("fastkpc/R/legacy_runner.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/native.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cuda.R")
source("fastkpc/R/full_cuda_ci_single_penalty_gcv.R")
source("fastkpc/R/mgcv_extract_oracle.R")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) {
  stop(
    "usage: diagnose_causal_dcc_gamma_witness.R RECEIPT OUTPUT [TASK_ID]",
    call. = FALSE
  )
}
receipt_path <- args[[1L]]
output_path <- args[[2L]]
requested_task_id <- if (length(args) >= 3L) as.integer(args[[3L]]) else NA_integer_
if (!file.exists(receipt_path)) stop("receipt is missing", call. = FALSE)
if (file.exists(output_path)) stop("output already exists", call. = FALSE)

receipt <- readRDS(receipt_path)
tasks <- receipt$result$skeleton$tasks
if (is.na(requested_task_id)) {
  requested_task_id <- as.integer(
    receipt$max_difference_task$canonical_test_order_id[[1L]]
  )
}
task_index <- match(requested_task_id, tasks$canonical_test_order_id)
if (is.na(task_index)) stop("task is absent from receipt", call. = FALSE)
task <- tasks[task_index, , drop = FALSE]

payload <- new.env(parent = emptyenv())
object_names <- load(receipt$data_path, envir = payload)
if (length(object_names) != 1L) {
  stop("data payload must contain one object", call. = FALSE)
}
data <- as.matrix(payload[[object_names[[1L]]]])
storage.mode(data) <- "double"
conditioning_set <- as.integer(strsplit(task$S_key[[1L]], "|", fixed = TRUE)[[1L]])
group_rows <- tasks[
  tasks$level == task$level[[1L]] & tasks$S_key == task$S_key[[1L]],
  , drop = FALSE
]
group_targets <- sort(unique(c(group_rows$x, group_rows$y)))
witness_targets <- as.integer(c(task$x[[1L]], task$y[[1L]]))
if (!all(witness_targets %in% group_targets)) {
  stop("witness targets are absent from the S-group", call. = FALSE)
}

native_setup <- full_cuda_ci_native_setup_native(
  data[, conditioning_set, drop = FALSE]
)
geometry <- full_cuda_ci_native_geometry_prepare_native(
  native_setup$X, native_setup$penalty_blocks,
  native_setup$penalty_offsets, native_setup$penalty_ranks
)
penalty_count <- length(native_setup$penalty_blocks)
prepared_geometry <- list(
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
prepared_setup <- list(
  formula_class = "additive-smooth",
  X = unname(native_setup$X),
  gram_matrix = unname(native_setup$gram_matrix),
  penalty_blocks = native_setup$penalty_blocks,
  penalty_offsets = as.integer(native_setup$penalty_offsets),
  penalty_ranks = as.integer(native_setup$penalty_ranks),
  mgcv_penalty_rank_metadata = as.integer(native_setup$penalty_ranks),
  constraint = matrix(
    numeric(), nrow = 0L, ncol = ncol(native_setup$X)
  ),
  H = NULL,
  weights = NULL
)
run_single_gpu <- function(targets) {
  Y <- unname(data[, targets, drop = FALSE])
  optimization <- fastkpc_full_cuda_phase4_cuda_batch(
    prepared_setup, Y, target_ids = targets
  )
  log_sp <- matrix(optimization$targets$log_sp, nrow = 1L)
  handle <- full_cuda_ci_multi_penalty_gcv_prepared_create_native(
    prepared_geometry, length(targets)
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
  token <- full_cuda_ci_multi_penalty_gcv_evaluate_fixed_batch_native(
    handle, Y, log_sp, sprintf("%064x", targets)
  )
  info <- full_cuda_ci_multi_penalty_gcv_residual_info_native(token)
  shadow <- full_cuda_ci_multi_penalty_gcv_residual_shadow_native(token)
  full_cuda_ci_multi_penalty_gcv_residual_release_native(token)
  full_cuda_ci_multi_penalty_gcv_residual_free_native(token)
  token <- NULL
  full_cuda_ci_multi_penalty_gcv_prepared_free_native(handle)
  handle <- NULL
  list(
    selected_log_sp = log_sp,
    fixed_residuals = shadow$residuals,
    fixed_coefficients = shadow$coefficients,
    fixed_solver_info = info$solver_info,
    fixed_decomposition_route = info$decomposition_route,
    fixed_numerical_rank = info$numerical_rank,
    optimization = optimization,
    exact_info = info
  )
}
run_gpu <- function(targets) {
  if (penalty_count == 1L) {
    run_single_gpu(targets)
  } else {
    full_cuda_ci_multi_penalty_capacity_qualify_native(
      native_setup, geometry, data[, targets, drop = FALSE]
    )
  }
}
full_gpu <- run_gpu(group_targets)
pair_gpu <- run_gpu(witness_targets)
reverse_gpu <- run_gpu(rev(witness_targets))
full_positions <- match(witness_targets, group_targets)

p_value <- function(residuals) {
  fastkpc_cuda_legacy_dcov_gamma_cpp_oracle(
    residuals[, 1L], residuals[, 2L], numCol = 35L, index = 1
  )$p.value
}

S_data <- as.data.frame(data[, conditioning_set, drop = FALSE])
if (is.null(colnames(S_data))) {
  colnames(S_data) <- paste0("s", seq_len(ncol(S_data)))
}
rhs <- fastkpc_mgcv_regrxons_rhs(
  S_data = S_data,
  S = conditioning_set,
  formula_class = fastkpc_regrxons_formula_class(conditioning_set)
)
legacy_fits <- lapply(witness_targets, function(target) {
  local_data <- cbind(
    data.frame(.target = data[, target]),
    S_data
  )
  mgcv::gam(
    stats::as.formula(paste(".target ~", rhs)),
    data = local_data,
    family = stats::gaussian(),
    method = "GCV.Cp"
  )
})
legacy_log_sp <- do.call(cbind, lapply(legacy_fits, function(fit) log(fit$sp)))
legacy_residuals <- do.call(cbind, lapply(legacy_fits, `[[`, "residuals"))

full_log_sp <- full_gpu$selected_log_sp[, full_positions, drop = FALSE]
pair_log_sp <- pair_gpu$selected_log_sp
reverse_log_sp <- reverse_gpu$selected_log_sp[, 2:1, drop = FALSE]
cpu_fixed <- lapply(seq_along(witness_targets), function(index) {
  local_data <- cbind(
    data.frame(.target = data[, witness_targets[[index]]]),
    S_data
  )
  fastkpc_mgcv_gam_fixed_sp_reference(
    formula = stats::as.formula(paste(".target ~", rhs)),
    data = local_data,
    sp = exp(full_log_sp[, index]),
    method = "GCV.Cp",
    target = witness_targets[[index]],
    S = conditioning_set
  )
})
cpu_fixed_residuals <- do.call(cbind, lapply(cpu_fixed, `[[`, "residuals"))

full_residuals <- full_gpu$fixed_residuals[, full_positions, drop = FALSE]
pair_residuals <- pair_gpu$fixed_residuals
reverse_residuals <- reverse_gpu$fixed_residuals[, 2:1, drop = FALSE]
oracle_path <- file.path(
  "/tmp/fastkpc_causal_large_parity_20260810",
  paste0(receipt$dataset, "_dcc.gamma.rds")
)
oracle <- readRDS(oracle_path)
oracle_position <- match(requested_task_id, tasks$canonical_test_order_id)
cpu_oracle_p <- as.double(oracle$cpu_p_values[[oracle_position]])

summary <- list(
  task_id = requested_task_id,
  conditioning_set = conditioning_set,
  coefficient_dim = ncol(native_setup$X),
  penalty_count = penalty_count,
  group_target_count = length(group_targets),
  witness_targets = witness_targets,
  one_call_p = as.double(task$p_used[[1L]]),
  cpu_oracle_p = cpu_oracle_p,
  full_group_gpu_p = p_value(full_residuals),
  pair_gpu_p = p_value(pair_residuals),
  reverse_pair_gpu_p = p_value(reverse_residuals),
  cpu_fixed_at_gpu_sp_p = p_value(cpu_fixed_residuals),
  legacy_mgcv_p = p_value(legacy_residuals),
  max_full_pair_log_sp_error = max(abs(full_log_sp - pair_log_sp)),
  max_full_reverse_log_sp_error = max(abs(full_log_sp - reverse_log_sp)),
  max_gpu_legacy_log_sp_error = max(abs(full_log_sp - legacy_log_sp)),
  max_full_pair_residual_error = max(abs(full_residuals - pair_residuals)),
  max_full_reverse_residual_error = max(abs(full_residuals - reverse_residuals)),
  max_gpu_cpu_fixed_residual_error = max(abs(
    full_residuals - cpu_fixed_residuals
  )),
  max_gpu_legacy_residual_error = max(abs(full_residuals - legacy_residuals)),
  full_solver_info = full_gpu$fixed_solver_info[full_positions],
  full_route = full_gpu$fixed_decomposition_route[full_positions],
  full_rank = full_gpu$fixed_numerical_rank[full_positions]
)

output <- list(
  schema_version = "causal-dcc-gamma-witness-diagnostic-v1",
  source_receipt = normalizePath(receipt_path),
  native_sha256 = unname(digest::digest(
    file = "fastkpc/build/fastkpc_cuda.so",
    algo = "sha256", serialize = FALSE
  )),
  task = task,
  group_rows = group_rows,
  group_targets = group_targets,
  summary = summary,
  full_log_sp = full_log_sp,
  pair_log_sp = pair_log_sp,
  reverse_log_sp = reverse_log_sp,
  legacy_log_sp = legacy_log_sp,
  full_residuals = full_residuals,
  pair_residuals = pair_residuals,
  reverse_residuals = reverse_residuals,
  cpu_fixed_residuals = cpu_fixed_residuals,
  legacy_residuals = legacy_residuals,
  full_gpu = full_gpu,
  pair_gpu = pair_gpu,
  reverse_gpu = reverse_gpu,
  cpu_fixed = cpu_fixed
)
saveRDS(output, output_path, compress = "xz", version = 3L)
print(summary, digits = 17)
