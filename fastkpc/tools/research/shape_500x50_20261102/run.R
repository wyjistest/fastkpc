# Current-native input-shape diagnostic against the archived public CPU oracle.
source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_phase10_development_500x50.R")

args <- commandArgs(TRUE)
stopifnot(length(args) == 1L, !file.exists(args[[1L]]))
out <- args[[1L]]
dir.create(out, recursive = TRUE)
fixture_path <- fastkpc_full_cuda_phase10_development_500x50_path()
x <- fastkpc_full_cuda_phase10_load_development_500x50(fixture_path)
sha <- function(p) digest::digest(file = p, algo = "sha256", serialize = FALSE)
native <- "fastkpc/build/fastkpc_cuda.so"
native_sha <- sha(native)
stopifnot(native_sha == "b2b79fc6219841a7b9de193e312de27bbdb8bca77a9d3a9d4a626aa1e135b12b")
numeric_check <- function(a, b) {
  shape <- identical(dim(a), dim(b)) && length(a) == length(b)
  a <- as.double(a); b <- as.double(b)
  special <- shape && identical(is.na(a), is.na(b)) &&
    identical(is.infinite(a), is.infinite(b)) &&
    identical(a[is.infinite(a)], b[is.infinite(b)])
  keep <- if (shape) is.finite(a) & is.finite(b) else logical()
  error <- if (!special) Inf else if (any(keep)) max(abs(a[keep] - b[keep])) else 0
  list(pass = special && error <= 1e-10, max_abs_error = error,
       outside_tolerance = if (special) sum(abs(a[keep] - b[keep]) > 1e-10) else NA_integer_)
}
stopifnot(fastkpc_cuda_available())
invisible(full_cuda_ci_one_call_cache_control_native("reset"))
before <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
set.seed(707L); rng_start <- .Random.seed
cat("START public synthetic 500x50, strict dcc.gamma, numCol=35, default-Inf\n")
flush.console()
started <- proc.time()[["elapsed"]]
candidate <- fastkpc_compatible_cuda_skeleton(x$data, alpha = 0.1,
  options = list(route = "full_cuda", compatible_cuda_strict = TRUE,
    index = 1, numCol = 35L, trace_level = "logical", ci_method = "dcc.gamma"))
elapsed <- proc.time()[["elapsed"]] - started
rng_end <- .Random.seed
invisible(full_cuda_ci_one_call_cache_control_native("reset")); gc(verbose = FALSE)
after <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
owners <- fixed_sp_cuda_live_owner_snapshot()
comparison <- fastkpc_full_cuda_compare_candidate_skeleton(x$oracle, candidate)
logical_trace <- fastkpc_full_cuda_normalize_logical_trace(candidate)
ci <- numeric_check(logical_trace$p_value, x$oracle$logical_trace$p_value)
pmax <- numeric_check(candidate$pMax, x$oracle$reference$pMax)
s <- candidate$summary
authority_fields <- c("r_callback_count", "legacy_mgcv_fit_count", "legacy_mgcv_setup_count",
  "cpu_residual_solve_count", "cpu_dcov_component_count", "cpu_dcov_eigen_or_lowrank_count",
  "cpu_dcov_pair_stat_count", "cpu_gamma_pvalue_count", "cpu_spectra_count",
  "residual_d2h_bytes", "component_d2h_bytes", "unknown_fallback_count", "approximate_backend_count")
live <- c("live_device_allocations", "live_device_bytes", "live_streams", "live_events")
order_fields <- c("logical_sequence_id", "source_task_index", "level", "x", "y", "S_key")
gates <- list(graph_and_trace_contract = isTRUE(comparison$summary$pass),
  all_logical_ci_p_values = ci$pass, all_finite_pmax = pmax$pass,
  task_identity_order = identical(logical_trace[order_fields], x$oracle$logical_trace[order_fields]),
  decisions = identical(logical_trace$deletes_edge, x$oracle$logical_trace$deletes_edge),
  rng_unchanged = identical(rng_start, rng_end),
  dimensions = s$n == 500L && s$p == 50L,
  unbounded_default = identical(s$max_conditioning_size_requested, "Inf") &&
    s$max_conditioning_size_resolved == 48L && identical(as.integer(candidate$levels$level), 0:3),
  real_gpu_compute = s$physical_tests_evaluated > 0 && s$physical_residual_fits > 0 &&
    s$native_call_count == 1L,
  authority_zero = all(vapply(authority_fields, function(nm)
    !is.null(s[[nm]]) && identical(as.numeric(s[[nm]]), 0), logical(1L))),
  resources_zero = all(live %in% names(before)) && all(live %in% names(after)) &&
    all(unlist(before[live]) == 0) && all(unlist(after[live]) == 0) && owners$total == 0L,
  stable_svd = s$cuda_exact_fixed_evaluator_target_count == s$cuda_exact_fixed_evaluator_stable_svd_target_count,
  exact_primary = isTRUE(s$dcc_gamma_exact_primary_optimizer_requested) &&
    s$cuda_multi_penalty_target_count == s$cuda_multi_penalty_exact_primary_target_count &&
    s$cuda_multi_penalty_stability_replay_target_count == 0 && s$cuda_multi_penalty_terminal_confirmation_count == 0,
  cache_metadata = s$legacy_component_cache_route_mismatch_count == 0 && s$legacy_component_cache_status_mismatch_count == 0,
  fresh_caches = s$result_cache_warm_start_entries == 0 && s$target_cache_warm_start_entries == 0,
  native_unchanged = identical(sha(native), native_sha))
report <- list(evidence_class = "one fresh current-native GPU skeleton versus archived CPU oracle; public synthetic development fixture",
  fixture = x$fixture_id, fixture_sha256 = sha(fixture_path), data_sha256 = x$data_sha256,
  oracle_sha256 = x$oracle_sha256, native_sha256 = native_sha,
  n = 500L, p = 50L, ci_method = "dcc.gamma", alpha = 0.1, index = 1, numCol = 35L,
  max_conditioning_size_requested = "Inf", max_conditioning_size_resolved = 48L,
  natural_stop_level = 3L, logical_ci_count = nrow(logical_trace),
  physical_ci_count = s$physical_tests_evaluated, gpu_elapsed_sec = elapsed,
  ci = ci, pmax = pmax, graph = comparison$summary, authority_counters = s[authority_fields],
  resources_before = before, resources_after = after, owners_after = owners,
  environment = as.list(Sys.getenv()[grepl("^(FASTKPC_|CUDA_|OMP_|OPENBLAS_|MKL_)", names(Sys.getenv()))]),
  gates = gates, pass = all(vapply(gates, isTRUE, logical(1L))),
  limitations = "No fresh CPU rerun, final PDAG or performance campaign; canonical-only accelerations disabled.",
  session_info = capture.output(sessionInfo()))
saveRDS(list(result = candidate, comparison = comparison, rng_start = rng_start,
  rng_end = rng_end, report = report), file.path(out, "receipt.rds"), compress = FALSE)
jsonlite::write_json(report, file.path(out, "report.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
print(report[c("logical_ci_count", "physical_ci_count", "gpu_elapsed_sec", "ci", "pmax", "gates", "pass")])
stopifnot(isTRUE(report$pass))
