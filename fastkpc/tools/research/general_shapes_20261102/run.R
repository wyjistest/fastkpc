source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_phase10_development_500x50.R")

args <- commandArgs(TRUE)
stopifnot(length(args) == 5L)
case <- args[[1L]]; mode <- args[[2L]]; native <- args[[3L]]
out <- args[[4L]]; reference_path <- args[[5L]]
stopifnot(mode %in% c("cpu", "off", "on"), !file.exists(out))
dir.create(out, recursive = TRUE)
.fastkpc_cuda_so <- local({ path <- native; function() path })
sha <- function(p) digest::digest(file = p, algo = "sha256", serialize = FALSE)
native_sha <- sha(native)

# Reuse pure comparison/orientation functions from the completed real-data check.
for (file in c("fastkpc/tests/test_compatible_cuda_wanpdag_authority.R",
               "fastkpc/tools/research/real_data_parity_20261102/run.R")) {
  for (e in parse(file)) if (is.call(e) && identical(e[[1L]], as.name("<-")) &&
    is.symbol(e[[2L]]) && as.character(e[[2L]]) %in% c("normalize_sepsets", "trace_frame",
      "numeric_check", "strict_p", "suff_stat", "orientation_reference", "compare_orientation")) eval(e)
}
real_paths <- c(
  real179 = "causalDiscovery/20221022181603/CD8.c11.Tex.PDCD1-causalDiscoveryInput.RData",
  real243 = "causalDiscovery/20220418113957/logged-liver-exhausted-allData-causalDiscoveryInput.RData",
  real300 = "causalDiscovery/20240315113222/CD8.c11.Tex.PDCD1-causalDiscoveryInput.RData")
real_ids <- c(real179 = "20221022181603", real243 = "20220418113957", real300 = "20240315113222")
oracle <- NULL
if (case %in% names(real_paths)) {
  payload <- new.env(parent = emptyenv()); loaded <- load(real_paths[[case]], envir = payload)
  stopifnot(length(loaded) == 1L)
  data <- as.matrix(payload[[loaded]])
  cpu <- readRDS(file.path("fastkpc/artifacts/full_cuda_ci/real_data_parity_20261102/attempt4",
                          real_ids[[case]], "cpu-schedule-replay.rds"))
  oracle <- fastkpc_full_cuda_phase10_development_oracle(cpu)
  # The independent CPU scheduler stores vertex indices, without the input's
  # gene labels. Attach labels from the same input before the name-aligning
  # comparator; never infer a permutation from candidate output.
  dimnames(oracle$reference$adjacency) <- list(colnames(data), colnames(data))
  dimnames(oracle$reference$pMax) <- list(colnames(data), colnames(data))
} else if (case == "public500") {
  fixture <- fastkpc_full_cuda_phase10_load_development_500x50()
  data <- fixture$data; oracle <- fixture$oracle
} else {
  settings <- list(dense500 = c(500L, 8L, 50008L), boundary75 = c(75L, 8L, 7508L),
    boundary76 = c(76L, 8L, 7608L), large1025 = c(1025L, 6L, 102506L),
    dense75 = c(75L, 6L, 7506L), dense76 = c(76L, 6L, 7606L),
    dense1025 = c(1025L, 4L, 102504L),
    blocks256 = c(256L, 12L, 25612L))
  stopifnot(case %in% names(settings))
  config <- settings[[case]]; n <- config[[1L]]; p <- config[[2L]]
  set.seed(config[[3L]])
  if (case %in% c("dense500", "dense75", "dense76", "dense1025")) {
    z <- rnorm(n)
    data <- matrix(z, n, p) + 0.2 * matrix(rnorm(n * p), n, p)
  } else {
    data <- matrix(rnorm(n * p), n, p)
    for (start in seq.int(1L, p, by = 3L)) {
      if (start + 1L <= p) data[, start + 1L] <- .8 * data[, start] + .6 * data[, start + 1L]
      if (start + 2L <= p) data[, start + 2L] <- .6 * data[, start + 1L] + .8 * data[, start + 2L]
    }
  }
  colnames(data) <- paste0("shape_v", seq_len(p))
}
storage.mode(data) <- "double"
input_sha <- digest::digest(data, algo = "sha256")
if (reference_path != "-") {
  reference <- readRDS(reference_path)
  stopifnot(identical(reference$input_sha256, input_sha))
  if (is.null(oracle)) oracle <- fastkpc_full_cuda_phase10_development_oracle(reference$result)
}
options <- list(route = if (mode == "cpu") "legacy" else "full_cuda",
  compatible_cuda_strict = TRUE, max_conditioning_size = Inf, index = 1,
  numCol = 35L, trace_level = "logical", ci_method = "dcc.gamma")
if (mode == "cpu") {
  Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_BACKEND = "cpp",
    FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = "spectra",
    FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE = "1", FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY = "s",
    FASTKPC_NATIVE_LEGACY_MGCV_PROVIDER_CORES = "20",
    FASTKPC_NATIVE_LEGACY_DCOV_BATCH = "round", FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH_THREADS = "20")
  options$dcov_batch <- "round"; options$mgcv_residual_backend <- "r"
} else {
  Sys.setenv(FASTKPC_DCC_GAMMA_EXACT_PRIMARY_OPTIMIZER = "1",
    FASTKPC_DCC_GAMMA_EXACT_PRIMARY_FIXED_REUSE = "1")
  flags <- c("FASTKPC_DCC_GAMMA_LEVEL_COMPONENT_PREFILL", "FASTKPC_DCC_GAMMA_STRUCTURED_COMPONENT_PREFILL",
    "FASTKPC_DCC_GAMMA_L2_COMPONENT_PREFILL", "FASTKPC_DCC_GAMMA_L3_OPTIMIZER_SLOTS",
    "FASTKPC_DCC_GAMMA_L4_OPTIMIZER_WORKERS32", "FASTKPC_DCC_GAMMA_L4_OPTIMIZER_SLOTS",
    "FASTKPC_DCC_GAMMA_L5_OPTIMIZER_SLOTS", "FASTKPC_DCC_GAMMA_L6_OPTIMIZER_SLOTS",
    "FASTKPC_DCC_GAMMA_VECTOR_POSTBACK", "FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H",
    "FASTKPC_DCC_GAMMA_STRUCTURED_WORK_PAIR")
  do.call(Sys.setenv, as.list(setNames(rep(if (mode == "on") "1" else "0", length(flags)), flags)))
  stopifnot(fastkpc_cuda_available())
  invisible(full_cuda_ci_one_call_cache_control_native("reset"))
}
before <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
set.seed(707L); rng_start <- .Random.seed
cat("START", case, mode, paste(dim(data), collapse = "x"), "\n"); flush.console()
started <- proc.time()[["elapsed"]]
result <- fastkpc_compatible_cuda_skeleton(data, alpha = 0.1, options = options)
elapsed <- proc.time()[["elapsed"]] - started; rng_end <- .Random.seed
after <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
owners <- fixed_sp_cuda_live_owner_snapshot()
receipt <- list(result = result, input_sha256 = input_sha, native_sha256 = native_sha,
  elapsed_sec = elapsed, seed_before = rng_start, seed_after = rng_end,
  resources_before = before, resources_after = after, owners_after = owners)
saveRDS(receipt, file.path(out, "receipt.rds"), compress = FALSE)
cat("SKELETON DONE", elapsed, "seconds\n"); flush.console()
if (mode == "cpu") {
  jsonlite::write_json(list(case = case, mode = mode, n = nrow(data), p = ncol(data),
    ci_count = sum(result$n.edgetests), elapsed_sec = elapsed, input_sha256 = input_sha,
    native_sha256 = native_sha, pass = TRUE), file.path(out, "report.json"),
    pretty = TRUE, auto_unbox = TRUE, digits = NA)
  quit(save = "no", status = 0L)
}
stopifnot(!is.null(oracle))
comparison <- fastkpc_full_cuda_compare_candidate_skeleton(oracle, result)
trace <- fastkpc_full_cuda_normalize_logical_trace(result)
ci <- numeric_check(trace$p_value, oracle$logical_trace$p_value)
pmax <- numeric_check(result$pMax, oracle$reference$pMax)
s <- result$summary
authority <- c("r_callback_count", "legacy_mgcv_fit_count", "legacy_mgcv_setup_count",
  "cpu_residual_solve_count", "cpu_dcov_component_count", "cpu_dcov_eigen_or_lowrank_count",
  "cpu_dcov_pair_stat_count", "cpu_gamma_pvalue_count", "cpu_spectra_count",
  "residual_d2h_bytes", "component_d2h_bytes", "unknown_fallback_count", "approximate_backend_count")
live <- c("live_device_allocations", "live_device_bytes", "live_streams", "live_events")
gates <- list(graph_trace = isTRUE(comparison$summary$pass), ci = ci$pass, pmax = pmax$pass,
  rng = identical(rng_start, rng_end),
  authority = all(vapply(authority, function(k) !is.null(s[[k]]) && identical(as.numeric(s[[k]]), 0), logical(1L))),
  resources = all(live %in% names(after)) && all(unlist(after[live]) == 0) && owners$total == 0,
  stable_svd = s$cuda_exact_fixed_evaluator_target_count == s$cuda_exact_fixed_evaluator_stable_svd_target_count,
  cache_metadata = s$legacy_component_cache_route_mismatch_count == 0 && s$legacy_component_cache_status_mismatch_count == 0,
  fresh_caches = s$result_cache_warm_start_entries == 0 && s$target_cache_warm_start_entries == 0,
  native_unchanged = identical(sha(native), native_sha))
if (mode == "on") {
  gates$shape_dispatch <- isTRUE(s$structured_component_prefill_requested) &&
    identical(isTRUE(s$structured_component_prefill_enabled), nrow(data) >= 76 && nrow(data) <= 1024)
  if (case %in% c("dense75", "dense76", "dense1025"))
    gates$boundary_producer_exercised <- s$level_component_produced > 0
}
# Orientation code is unchanged. Check it once per declared end-to-end input;
# repeated timings exercise the changed skeleton, with all numeric gates.
orientation_checked <- identical(Sys.getenv("FASTKPC_SHAPE_ORIENTATION"), "1")
if (orientation_checked) {
  set.seed(707L)
  candidate_orientation <- fastkpc_orient_wanpdag_kpcalg_authority(result, data,
    alpha = .1, index = 1, numCol = 35L, ci_method = "dcc.gamma")
  orientation_rng <- .Random.seed
  set.seed(707L)
  cpu_orientation <- orientation_reference(fastkpc_skeleton_as_pc_algo(oracle$reference, data), data)
  gates <- c(gates, compare_orientation(candidate_orientation, cpu_orientation))
  gates$orientation_rng <- identical(orientation_rng, cpu_orientation$rng_end)
  saveRDS(list(candidate = candidate_orientation, reference = cpu_orientation),
    file.path(out, "orientation.rds"), compress = FALSE)
}
report <- list(case = case, mode = mode, n = nrow(data), p = ncol(data),
  native_sha256 = native_sha, input_sha256 = input_sha, elapsed_sec = elapsed,
  ci_count = nrow(trace), levels = result$levels, ci = ci, pmax = pmax,
  graph = comparison$summary, gates = gates, pass = all(vapply(gates, isTRUE, logical(1L))),
  orientation_checked = orientation_checked,
  authority = s[authority], resources = after,
  execution = s[grep("^(l[3456]_optimizer_slot|level_component|structured_component|cuda_structured|cuda_multi_penalty.*postback)", names(s))],
  environment = as.list(Sys.getenv()[grepl("^(FASTKPC_|CUDA_|OMP_|OPENBLAS_|MKL_)", names(Sys.getenv()))]))
jsonlite::write_json(report, file.path(out, "report.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat("RESULT", case, mode, elapsed, "seconds; CI", nrow(trace), "error", ci$max_abs_error,
    "PASS", report$pass, "\n")
if (!report$pass) print(gates)
stopifnot(report$pass)
