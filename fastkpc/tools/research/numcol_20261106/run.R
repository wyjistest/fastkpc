source("fastkpc/tools/research/numcol_20261106/common.R")
args <- commandArgs(TRUE)
stopifnot(length(args) == 6L)
case <- args[[1L]]; requested <- args[[2L]]; mode <- args[[3L]]
native <- args[[4L]]; out <- args[[5L]]; reference_path <- args[[6L]]
stopifnot(mode %in% c("cpu", "off", "on", "uncached"), !file.exists(out))
dir.create(out, recursive = TRUE)
.fastkpc_cuda_so <- local({ path <- native; function() path })
sha <- function(p) digest::digest(file = p, algo = "sha256", serialize = FALSE)
native_sha <- sha(native)
data <- rank_data(case)
k <- if (requested == "default") floor(nrow(data) / 10) else as.integer(requested)
input_sha <- digest::digest(data, algo = "sha256")
options <- list(route = if (mode == "cpu") "legacy" else "full_cuda",
  compatible_cuda_strict = TRUE, max_conditioning_size = Inf, index = 1,
  trace_level = "logical", ci_method = "dcc.gamma")
# Intentionally omit numCol for default cases: qualify the public default itself.
if (requested != "default") options$numCol <- k
if (mode == "cpu") {
  Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_BACKEND = "cpp",
    FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = "spectra",
    FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE = "1", FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY = "s",
    FASTKPC_NATIVE_LEGACY_MGCV_PROVIDER_CORES = "20",
    FASTKPC_NATIVE_LEGACY_DCOV_BATCH = "round", FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH_THREADS = "20")
  options$dcov_batch <- "round"; options$mgcv_residual_backend <- "r"
} else {
  rank_mode(mode)
  stopifnot(fastkpc_cuda_available())
  invisible(full_cuda_ci_one_call_cache_control_native("reset"))
}
set.seed(707L); rng_start <- .Random.seed
cat("START", case, mode, "numCol", k, "\n"); flush.console()
started <- proc.time()[["elapsed"]]
result <- fastkpc_compatible_cuda_skeleton(data, alpha = .1, options = options)
elapsed <- proc.time()[["elapsed"]] - started; rng_end <- .Random.seed
receipt <- list(result = result, input_sha256 = input_sha, native_sha256 = native_sha,
  elapsed_sec = elapsed, numCol = k, requested = requested,
  seed_before = rng_start, seed_after = rng_end)
saveRDS(receipt, file.path(out, "receipt.rds"), compress = FALSE)
cat("DONE skeleton", elapsed, "\n"); flush.console()
if (mode == "cpu") {
  jsonlite::write_json(list(pass = TRUE, n = nrow(data), p = ncol(data), numCol = k,
    native_sha256 = native_sha, input_sha256 = input_sha, ci_count = sum(result$n.edgetests),
    elapsed_sec = elapsed), file.path(out, "report.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
  quit(save = "no", status = 0L)
}
reference <- readRDS(reference_path)
stopifnot(identical(reference$input_sha256, input_sha), reference$numCol == k)
oracle <- fastkpc_full_cuda_phase10_development_oracle(reference$result)
comparison <- fastkpc_full_cuda_compare_candidate_skeleton(oracle, result)
trace <- fastkpc_full_cuda_normalize_logical_trace(result)
ci <- numeric_check(trace$p_value, oracle$logical_trace$p_value)
pmax <- numeric_check(result$pMax, oracle$reference$pMax)
s <- result$summary
resources <- rank_resources()
gates <- list(graph_trace = isTRUE(comparison$summary$pass), ci = ci$pass, pmax = pmax$pass,
  rng = identical(rng_start, rng_end), actual_numCol = s$numCol == k,
  authority = all(vapply(rank_authority, function(key) !is.null(s[[key]]) &&
    identical(as.numeric(s[[key]]), 0), logical(1L))), resources = TRUE,
  stable_svd = s$cuda_exact_fixed_evaluator_target_count == s$cuda_exact_fixed_evaluator_stable_svd_target_count,
  cache_metadata = s$legacy_component_cache_route_mismatch_count == 0 && s$legacy_component_cache_status_mismatch_count == 0,
  fresh_caches = s$result_cache_warm_start_entries == 0 && s$target_cache_warm_start_entries == 0,
  native_unchanged = identical(native_sha, sha(native)))
if (mode == "on") {
  structured <- nrow(data) <= 1024 && k <= 64 && k + 1 < nrow(data) && 2 * (k + 1) + 4 <= nrow(data)
  gates$shape_dispatch <- identical(isTRUE(s$structured_component_prefill_enabled), structured)
  # Prefill starts at conditioning level 2. A valid low-rank test may naturally
  # delete every edge earlier; that run cannot exercise a deeper producer.
  if (startsWith(case, "dense") && any(trace$level >= 2L))
    gates$producer_exercised <- s$level_component_produced > 0
}
orientation_checked <- mode == "on" && case %in% c("real179", "real243", "real300", "public500")
if (orientation_checked) {
  suff_stat <- function(data) list(data = data, ic.method = "dcc.gamma", index = 1, numCol = as.integer(k))
  set.seed(707L)
  actual <- fastkpc_orient_wanpdag_kpcalg_authority(result, data, numCol = k)
  orientation_rng <- .Random.seed
  set.seed(707L)
  expected <- orientation_reference(fastkpc_skeleton_as_pc_algo(oracle$reference, data), data)
  gates <- c(gates, compare_orientation(actual, expected))
  gates$orientation_rng <- identical(orientation_rng, expected$rng_end)
  saveRDS(list(candidate = actual, reference = expected), file.path(out, "orientation.rds"), compress = FALSE)
}
report <- list(case = case, mode = mode, n = nrow(data), p = ncol(data), numCol = k, requested = requested,
  native_sha256 = native_sha, reference_native_sha256 = reference$native_sha256,
  input_sha256 = input_sha, elapsed_sec = elapsed, ci_count = nrow(trace), ci = ci, pmax = pmax,
  comparison = comparison$summary, gates = gates, pass = all(vapply(gates, isTRUE, logical(1L))),
  orientation_checked = orientation_checked, resources = resources, maximum_level = max(trace$level),
  execution = s[grep("^(level_component|structured_component|cuda_structured|legacy_component_cache)", names(s))])
jsonlite::write_json(report, file.path(out, "report.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
print(gates); stopifnot(report$pass)
cat("PASS", case, mode, "numCol", k, "CI", nrow(trace), "error", ci$max_abs_error, "\n")
