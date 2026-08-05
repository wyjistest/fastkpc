source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/legacy_runner.R")

args <- commandArgs(trailingOnly = TRUE)
method <- if (length(args) >= 1L) args[[1L]] else ""
output_path <- if (length(args) >= 2L) args[[2L]] else ""

fail <- function(message) stop(message, call. = FALSE)
require_true <- function(value, message) if (!isTRUE(value)) fail(message)
elapsed_seconds <- function(started) {
  unname((proc.time() - started)[["elapsed"]])
}
normalize_sepsets <- function(value) {
  unname(lapply(unname(value), function(row) {
    unname(lapply(unname(row), function(entry) sort(as.integer(entry))))
  }))
}
atomic_save_rds <- function(value, path) {
  temporary <- tempfile(".strict-method-candidate-", tmpdir = dirname(path))
  on.exit(unlink(temporary, force = TRUE), add = TRUE)
  saveRDS(value, temporary, compress = "xz")
  require_true(file.rename(temporary, path),
               paste("failed to publish", basename(path)))
}

require_true(
  method %in% c("hsic.gamma", "dcc.perm", "hsic.perm"),
  "method must be hsic.gamma, dcc.perm, or hsic.perm"
)
require_true(nzchar(output_path) && !file.exists(output_path) &&
               dir.exists(dirname(output_path)),
             "candidate output path is invalid or already exists")
require_true(isTRUE(tryCatch(
  fastkpc_cuda_available(),
  error = function(error) FALSE
)), "CUDA is unavailable")

data_path <- paste0(
  "fastkpc/artifacts/kpc_tprs_real_zhu/",
  "cancer_RD-causalDiscoveryInput.rds"
)
evidence_root <- "fastkpc/artifacts/strict_ci_methods_351x48_v1/payload"
baseline_path <- file.path(evidence_root, method, "candidate.rds")
wanpdag_path <- file.path(evidence_root, method, "wanpdag_receipt.rds")
require_true(file.exists(data_path) && file.exists(baseline_path) &&
               file.exists(wanpdag_path),
             "persistent strict-method evidence is incomplete")

data <- as.matrix(readRDS(data_path))
storage.mode(data) <- "double"
require_true(identical(dim(data), c(351L, 48L)),
             "canonical strict-method data shape changed")
baseline_payload <- readRDS(baseline_path)
baseline <- if (!is.null(baseline_payload$result)) {
  baseline_payload$result
} else {
  baseline_payload
}
wanpdag <- readRDS(wanpdag_path)

invisible(full_cuda_ci_one_call_cache_control_native("reset"))
if (method %in% c("dcc.perm", "hsic.perm")) set.seed(707)
rng_initial <- if (exists(".Random.seed", envir = .GlobalEnv,
                          inherits = FALSE)) .Random.seed else NULL
started <- proc.time()
candidate <- precision_run_skeleton_full_cuda_native(
  data = data,
  alpha = 0.1,
  max_conditioning_size = Inf,
  index = 1,
  numCol = 35L,
  trace_level = "logical",
  compatible_cuda_strict = TRUE,
  ci_method = method,
  hsic_params = list(sig = 1),
  permutation_params = list(
    replicates = 100L,
    seed = 707L,
    include_observed = TRUE
  )
)
elapsed <- elapsed_seconds(started)
rng_final <- if (exists(".Random.seed", envir = .GlobalEnv,
                        inherits = FALSE)) .Random.seed else NULL

task_fields <- c(
  "level", "task_index", "edge_x", "edge_y", "x", "y", "S_key",
  "conditioning_size", "native_edge_deleted", "native_edge_ignored"
)
require_true(all(task_fields %in% names(candidate$tasks)) &&
               all(task_fields %in% names(baseline$tasks)),
             "logical task trace is incomplete")
structural_trace_identical <- identical(
  candidate$tasks[, task_fields, drop = FALSE],
  baseline$tasks[, task_fields, drop = FALSE]
)
p_differences <- abs(
  as.numeric(candidate$tasks$p_used) - as.numeric(baseline$tasks$p_used)
)
tolerance <- if (method == "hsic.gamma") 1e-10 else 0
max_abs_p_diff <- if (length(p_differences)) max(p_differences) else 0
decision_flip_count <- sum(
  (candidate$tasks$p_used >= 0.1) != (baseline$tasks$p_used >= 0.1)
)
adjacency_identical <- identical(
  unname(candidate$adjacency), unname(baseline$adjacency)
)
sepsets_identical <- identical(
  normalize_sepsets(candidate$sepsets),
  normalize_sepsets(baseline$sepsets)
)
pmax_difference <- max(abs(candidate$pMax - baseline$pMax), na.rm = TRUE)
n_edgetests_identical <- identical(
  as.integer(candidate$n.edgetests), as.integer(baseline$n.edgetests)
)

summary <- candidate$summary
cache_accounted <-
  summary$method_residual_cache_capacity_entries >= 2 &&
  summary$method_residual_cache_all_hit_batch_count > 0L &&
  summary$method_residual_cache_bypassed_target_count > 0L &&
  summary$method_residual_cache_lookup_count ==
    summary$logical_residual_requests &&
  summary$method_residual_cache_lookup_count ==
    summary$method_residual_cache_hit_count +
      summary$method_residual_cache_insert_count &&
  summary$physical_residual_fits ==
    summary$logical_residual_requests -
      summary$method_residual_cache_bypassed_target_count &&
  summary$cuda_exact_screen_residual_target_count ==
    summary$physical_residual_fits &&
  summary$cuda_exact_screen_residual_batch_count +
      summary$method_residual_cache_all_hit_batch_count ==
    summary$frontier_batch_count &&
  summary$method_residual_cache_gather_d2d_bytes ==
    summary$method_residual_cache_bypassed_target_count * nrow(data) * 8 &&
  summary$physical_residual_fits <
    baseline$summary$physical_residual_fits &&
  summary$unique_residual_key_count <= summary$physical_residual_fits

authority_clean <-
  summary$r_callback_count == 0L &&
  summary$legacy_mgcv_fit_count == 0L &&
  summary$legacy_mgcv_setup_count == 0L &&
  summary$cpu_residual_solve_count == 0L &&
  summary$cpu_dcov_component_count == 0L &&
  summary$cpu_dcov_eigen_or_lowrank_count == 0L &&
  summary$cpu_dcov_pair_stat_count == 0L &&
  summary$cpu_gamma_pvalue_count == 0L &&
  summary$residual_d2h_bytes == 0 &&
  summary$component_d2h_bytes == 0 &&
  summary$unknown_fallback_count == 0L &&
  summary$approximate_backend_count == 0L

rng_identical <- TRUE
if (method %in% c("dcc.perm", "hsic.perm")) {
  expected_rng <- wanpdag$kpcalg_authority_diagnostics$rng_start_state
  rng_identical <- identical(rng_final, expected_rng)
}

validation <- list(
  tolerance = tolerance,
  structural_trace_identical = structural_trace_identical,
  max_abs_p_diff = max_abs_p_diff,
  p_values_within_tolerance = all(p_differences <= tolerance),
  decision_flip_count = as.integer(decision_flip_count),
  adjacency_identical = adjacency_identical,
  sepsets_identical = sepsets_identical,
  max_abs_pmax_diff = pmax_difference,
  pmax_within_tolerance = pmax_difference <= tolerance,
  n_edgetests_identical = n_edgetests_identical,
  rng_state_identical = rng_identical,
  cache_accounted = cache_accounted,
  authority_clean = authority_clean
)
validation$pass <- all(vapply(validation[c(
  "structural_trace_identical", "p_values_within_tolerance",
  "adjacency_identical", "sepsets_identical", "pmax_within_tolerance",
  "n_edgetests_identical", "rng_state_identical", "cache_accounted",
  "authority_clean"
)], isTRUE, logical(1))) && decision_flip_count == 0L

payload <- list(
  elapsed = elapsed,
  result = candidate,
  rng_initial = rng_initial,
  rng_final = rng_final,
  baseline_path = normalizePath(baseline_path),
  validation = validation
)
atomic_save_rds(payload, output_path)
cat(sprintf(
  paste0(
    "strict method candidate: method=%s elapsed=%.3f tests=%d ",
    "physical_residual_fits=%d bypassed=%d max_abs_p_diff=%.17g ",
    "rng=%s pass=%s\n"
  ),
  method, elapsed, summary$logical_tests_consumed,
  summary$physical_residual_fits,
  summary$method_residual_cache_bypassed_target_count,
  max_abs_p_diff, rng_identical, validation$pass
))
require_true(validation$pass, "strict method candidate gate failed")
