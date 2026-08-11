source("fastkpc/R/fast_kpc.R")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) {
  stop("usage: run_causal_large_dcc_gamma_gpu_qualification.R DATASET OUTPUT",
       call. = FALSE)
}
dataset <- args[[1L]]
output_path <- args[[2L]]
if (file.exists(output_path)) stop("output already exists", call. = FALSE)

data_paths <- c(
  `20230331145508` =
    "causalDiscovery/20230331145508/combined_data_shuffle-causalDiscoveryInput.RData",
  `20240102191119` =
    "causalDiscovery/20240102191119/GSE99254-482col-causalDiscoveryInput.RData",
  `20241215091815` =
    "causalDiscovery/20241215091815/logged-lung-exhausted-allData-causalDiscoveryInput.RData"
)
if (!dataset %in% names(data_paths)) stop("unknown dataset", call. = FALSE)
data_path <- data_paths[[dataset]]
oracle_path <- file.path(
  "/tmp/fastkpc_causal_large_parity_20260810",
  paste0(dataset, "_dcc.gamma.rds")
)
if (!file.exists(data_path) || !file.exists(oracle_path)) {
  stop("dataset or CPU oracle is missing", call. = FALSE)
}

payload <- new.env(parent = emptyenv())
object_names <- load(data_path, envir = payload)
if (length(object_names) != 1L) {
  stop("data payload must contain one object", call. = FALSE)
}
data <- as.matrix(payload[[object_names[[1L]]]])
storage.mode(data) <- "double"
oracle <- readRDS(oracle_path)
old_candidate <- readRDS(oracle$candidate_path)

resource_snapshot <- function() {
  .Call("C_fixed_sp_cuda_test_resource_snapshot", PACKAGE = "fastkpc_cuda")
}
active_resource_counts <- function(value) {
  fields <- grep("_active_count$", names(value), value = TRUE)
  stats::setNames(
    as.numeric(unlist(value[fields], use.names = FALSE)), fields
  )
}
strip_p_value <- function(trace) {
  lapply(trace, function(value) {
    value$p.value <- NULL
    value
  })
}
replay_pmax <- function(tasks, p_values, node_count) {
  if (length(p_values) != nrow(tasks)) {
    stop("CPU p-value count differs from the canonical task trace",
         call. = FALSE)
  }
  pmax <- matrix(-Inf, node_count, node_count)
  diag(pmax) <- 1
  for (index in seq_len(nrow(tasks))) {
    edge_x <- as.integer(tasks$edge_x[[index]])
    edge_y <- as.integer(tasks$edge_y[[index]])
    p_value <- as.double(p_values[[index]])
    if (edge_x < 1L || edge_x > node_count ||
        edge_y < 1L || edge_y > node_count || !is.finite(p_value)) {
      stop("CPU pMax replay payload is malformed", call. = FALSE)
    }
    if (p_value > pmax[edge_x, edge_y]) {
      pmax[edge_x, edge_y] <- p_value
      pmax[edge_y, edge_x] <- p_value
    }
  }
  pmax
}
max_pmax_error <- function(candidate, reference) {
  if (!identical(dim(candidate), dim(reference))) return(Inf)
  candidate <- as.double(candidate)
  reference <- as.double(reference)
  finite <- is.finite(candidate) & is.finite(reference)
  special_exact <- identical(is.na(candidate), is.na(reference)) &&
    identical(is.infinite(candidate), is.infinite(reference)) &&
    identical(
      sign(candidate[is.infinite(candidate)]),
      sign(reference[is.infinite(reference)])
    )
  if (!special_exact) return(Inf)
  if (any(finite)) max(abs(candidate[finite] - reference[finite])) else 0
}

invisible(full_cuda_ci_one_call_cache_control_native("reset"))
resources_before <- resource_snapshot()
owners_before <- fixed_sp_cuda_live_owner_snapshot()
on.exit({
  try(full_cuda_ci_one_call_cache_control_native("reset"), silent = TRUE)
}, add = TRUE)

set.seed(707L)
rng_start <- .Random.seed
started <- proc.time()[["elapsed"]]
result <- fastkpc_compatible_cuda_wanpdag(
  data,
  alpha = 0.1,
  options = list(
    max_conditioning_size = Inf,
    index = 1,
    numCol = 35L,
    trace_level = "full",
    route = "full_cuda",
    compatible_cuda_strict = TRUE,
    ci_method = "dcc.gamma"
  )
)
elapsed_sec <- proc.time()[["elapsed"]] - started
rng_end <- .Random.seed

cache_after_run <- full_cuda_ci_one_call_cache_control_native("info")
invisible(full_cuda_ci_one_call_cache_control_native("reset"))
gc(verbose = FALSE)
resources_after <- resource_snapshot()
owners_after <- fixed_sp_cuda_live_owner_snapshot()

tasks <- result$skeleton$tasks
old_tasks <- old_candidate$result$skeleton$tasks
identity_fields <- c(
  "canonical_test_order_id", "level", "task_index", "edge_x", "edge_y",
  "x", "y", "S_key", "conditioning_size"
)
task_identity_exact <-
  nrow(tasks) == nrow(old_tasks) &&
  all(identity_fields %in% names(tasks)) &&
  all(identity_fields %in% names(old_tasks)) &&
  identical(tasks[identity_fields], old_tasks[identity_fields])
task_trace_fields <- setdiff(intersect(names(tasks), names(old_tasks)), "p_used")
logical_deletion_trace_exact <-
  nrow(tasks) == nrow(old_tasks) &&
  identical(tasks[task_trace_fields], old_tasks[task_trace_fields])
level_trace_fields <- setdiff(
  intersect(names(result$skeleton$levels),
            names(old_candidate$result$skeleton$levels)),
  "elapsed_ms"
)
level_trace_exact <- identical(
  result$skeleton$levels[level_trace_fields],
  old_candidate$result$skeleton$levels[level_trace_fields]
)
cpu_p <- as.double(oracle$cpu_p_values)
if (length(cpu_p) != nrow(tasks)) {
  stop("candidate task count differs from the CPU oracle", call. = FALSE)
}
differences <- as.double(tasks$p_used) - cpu_p
decision_flip <- (as.double(tasks$p_used) > 0.1) != (cpu_p > 0.1)
cpu_pmax <- replay_pmax(tasks, cpu_p, ncol(data))
pmax_cpu_reference_coverage <- task_identity_exact &&
  identical(dim(result$skeleton$pMax), dim(cpu_pmax))
max_abs_pmax_diff_from_cpu <- if (pmax_cpu_reference_coverage) {
  max_pmax_error(result$skeleton$pMax, cpu_pmax)
} else {
  Inf
}
pmax_within_contract <- is.finite(max_abs_pmax_diff_from_cpu) &&
  max_abs_pmax_diff_from_cpu <= 1e-10
summary <- result$skeleton$summary
orientation_trace <- result$orientation$ci_trace
old_orientation_trace <- old_candidate$result$orientation$ci_trace
orientation_p <- vapply(
  orientation_trace, function(value) as.double(value$p.value), numeric(1L)
)
old_orientation_p <- vapply(
  old_orientation_trace, function(value) as.double(value$p.value), numeric(1L)
)
orientation_ci_identity_exact <-
  length(orientation_trace) == length(old_orientation_trace) &&
  identical(strip_p_value(orientation_trace),
            strip_p_value(old_orientation_trace))
orientation_max_abs_p_diff <- if (length(orientation_p) == 0L) {
  0
} else if (length(orientation_p) == length(old_orientation_p)) {
  max(abs(orientation_p - old_orientation_p))
} else {
  Inf
}
active_before <- active_resource_counts(resources_before)
active_after <- active_resource_counts(resources_after)
receipt <- list(
  schema_version = "causal-large-dcc-gamma-gpu-qualification-v2",
  dataset = dataset,
  data_path = normalizePath(data_path),
  native_sha256 = unname(digest::digest(
    file = "fastkpc/build/fastkpc_cuda.so", algo = "sha256",
    serialize = FALSE
  )),
  elapsed_sec = elapsed_sec,
  task_count = nrow(tasks),
  task_identity_exact = task_identity_exact,
  logical_deletion_trace_exact = logical_deletion_trace_exact,
  level_trace_exact = level_trace_exact,
  max_abs_p_diff = max(abs(differences)),
  mismatch_count = sum(abs(differences) > 1e-10),
  decision_flip_count = sum(decision_flip),
  p_diff_quantiles = stats::quantile(
    abs(differences), c(0, 0.5, 0.9, 0.99, 0.999, 1)
  ),
  max_difference_task = cbind(
    tasks[which.max(abs(differences)), identity_fields, drop = FALSE],
    cpu_p = cpu_p[[which.max(abs(differences))]],
    gpu_p = tasks$p_used[[which.max(abs(differences))]],
    abs_diff = abs(differences[[which.max(abs(differences))]])
  ),
  adjacency_identical = identical(
    result$skeleton$adjacency,
    old_candidate$result$skeleton$adjacency
  ),
  sepsets_identical = identical(
    result$skeleton$sepsets,
    old_candidate$result$skeleton$sepsets
  ),
  pmax_identical_to_old_candidate = identical(
    result$skeleton$pMax,
    old_candidate$result$skeleton$pMax
  ),
  max_abs_pmax_diff_from_old_candidate = max(abs(
    result$skeleton$pMax - old_candidate$result$skeleton$pMax
  )),
  pmax_cpu_reference_coverage = pmax_cpu_reference_coverage,
  max_abs_pmax_diff_from_cpu = max_abs_pmax_diff_from_cpu,
  pmax_within_contract = pmax_within_contract,
  n_edgetests_identical = identical(
    result$skeleton$n.edgetests,
    old_candidate$result$skeleton$n.edgetests
  ),
  orientation_events_identical = identical(
    result$orientation$events,
    old_candidate$result$orientation$events
  ),
  orientation_counts_identical = identical(
    result$orientation$counts,
    old_candidate$result$orientation$counts
  ),
  orientation_ci_identity_exact = orientation_ci_identity_exact,
  orientation_max_abs_p_diff = orientation_max_abs_p_diff,
  final_pdag_identical = identical(
    result$orientation$pdag,
    old_candidate$result$orientation$pdag
  ),
  rng_start_exact = identical(
    rng_start, old_candidate$execution$rng_start
  ),
  rng_end_exact = identical(
    rng_end, old_candidate$execution$rng_end
  ),
  cpu_residual_solve_count = summary$cpu_residual_solve_count,
  cpu_dcov_component_count = summary$cpu_dcov_component_count,
  cpu_gamma_pvalue_count = summary$cpu_gamma_pvalue_count,
  cpu_spectra_count = summary$cpu_spectra_count,
  unknown_fallback_count = summary$unknown_fallback_count,
  approximate_backend_count = summary$approximate_backend_count,
  exact_fixed_evaluator_batch_count =
    summary$cuda_exact_fixed_evaluator_batch_count,
  exact_fixed_evaluator_target_count =
    summary$cuda_exact_fixed_evaluator_target_count,
  exact_fixed_evaluator_stable_svd_target_count =
    summary$cuda_exact_fixed_evaluator_stable_svd_target_count,
  exact_fixed_evaluator_host_ms =
    summary$cuda_exact_fixed_evaluator_host_ms,
  cache_after_run = cache_after_run,
  resources_before = resources_before,
  resources_after = resources_after,
  owners_before = owners_before,
  owners_after = owners_after,
  active_resources_before_zero = all(active_before == 0),
  active_resources_after_zero = all(active_after == 0),
  external_owners_before_zero = owners_before$total == 0L,
  external_owners_after_zero = owners_after$total == 0L,
  result = result
)
receipt$pass <- isTRUE(receipt$task_identity_exact) &&
  isTRUE(receipt$logical_deletion_trace_exact) &&
  isTRUE(receipt$level_trace_exact) &&
  receipt$mismatch_count == 0L && receipt$decision_flip_count == 0L &&
  isTRUE(receipt$adjacency_identical) &&
  isTRUE(receipt$sepsets_identical) &&
  isTRUE(receipt$pmax_cpu_reference_coverage) &&
  isTRUE(receipt$pmax_within_contract) &&
  isTRUE(receipt$n_edgetests_identical) &&
  isTRUE(receipt$orientation_events_identical) &&
  isTRUE(receipt$orientation_counts_identical) &&
  isTRUE(receipt$orientation_ci_identity_exact) &&
  receipt$orientation_max_abs_p_diff <= 1e-10 &&
  isTRUE(receipt$final_pdag_identical) &&
  isTRUE(receipt$rng_start_exact) && isTRUE(receipt$rng_end_exact) &&
  receipt$cpu_residual_solve_count == 0L &&
  receipt$cpu_dcov_component_count == 0L &&
  receipt$cpu_gamma_pvalue_count == 0L &&
  receipt$cpu_spectra_count == 0L &&
  receipt$unknown_fallback_count == 0L &&
  receipt$approximate_backend_count == 0L &&
  receipt$exact_fixed_evaluator_batch_count > 0L &&
  receipt$exact_fixed_evaluator_target_count > 0L &&
  receipt$exact_fixed_evaluator_stable_svd_target_count ==
    receipt$exact_fixed_evaluator_target_count &&
  isTRUE(receipt$active_resources_before_zero) &&
  isTRUE(receipt$active_resources_after_zero) &&
  isTRUE(receipt$external_owners_before_zero) &&
  isTRUE(receipt$external_owners_after_zero)
saveRDS(receipt, output_path, compress = "xz", version = 3L)
print(receipt[setdiff(names(receipt), "result")], digits = 17)
