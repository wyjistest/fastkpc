# Fresh real-data CPU/GPU correctness comparisons; production files stay intact.
source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
source("fastkpc/R/full_cuda_ci_gate.R")

args <- commandArgs(TRUE)
stopifnot(length(args) == 2L)
dataset <- args[[1L]]
out <- args[[2L]]
stopifnot(!file.exists(out))
dir.create(out, recursive = TRUE)
sha <- function(p) digest::digest(file = p, algo = "sha256", serialize = FALSE)
native <- "fastkpc/build/fastkpc_cuda.so"
native_sha <- sha(native)
stopifnot(native_sha == "b2b79fc6219841a7b9de193e312de27bbdb8bca77a9d3a9d4a626aa1e135b12b")

# Reuse only pure comparison helpers, never the synthetic test's execution.
for (e in parse("fastkpc/tests/test_compatible_cuda_wanpdag_authority.R")) {
  if (is.call(e) && identical(e[[1L]], as.name("<-")) &&
      is.symbol(e[[2L]]) && as.character(e[[2L]]) %in%
      c("normalize_sepsets", "trace_frame")) eval(e)
}
numeric_check <- function(a, b, tolerance = 1e-10) {
  shape <- identical(dim(a), dim(b)) && length(a) == length(b)
  a <- as.double(a); b <- as.double(b)
  special <- shape && identical(is.na(a), is.na(b)) &&
    identical(is.infinite(a), is.infinite(b)) &&
    identical(a[is.infinite(a)], b[is.infinite(b)])
  keep <- if (shape) is.finite(a) & is.finite(b) else logical()
  error <- if (!special) Inf else if (any(keep)) max(abs(a[keep] - b[keep])) else 0
  list(pass = special && error <= tolerance, max_abs_error = error,
       outside_tolerance = if (special) sum(abs(a[keep] - b[keep]) > tolerance) else NA_integer_)
}
key <- function(level, x, y, S) paste(level, x, y, S, sep = "\037")
strict_p <- function(p) {
  stopifnot(is.numeric(p), length(p) == 1L, is.finite(p), p >= 0, p <= 1)
  as.double(p)
}
suff_stat <- function(data) list(data = data, ic.method = "dcc.gamma", index = 1, numCol = 35L)

cpu_schedule_reference <- function(data, cpu_trace) {
  env <- fastkpc_legacy_env()
  cpu_keys <- key(cpu_trace$level, cpu_trace$x, cpu_trace$y, cpu_trace$S_key)
  speculative <- 0L
  provider <- function(planned, level) {
    if (nrow(planned) == 0L) return(numeric())
    positions <- match(key(level, planned$x, planned$y, planned$S_key), cpu_keys)
    values <- cpu_trace$p_value[positions]
    for (j in which(is.na(positions))) {
      speculative <<- speculative + 1L
      values[[j]] <- strict_p(env$kernelCItest(planned$x[[j]], planned$y[[j]],
        as.integer(planned$conditioning_sets[[j]]), suff_stat(data)))
    }
    values
  }
  result <- precision_run_skeleton_provider_native(p = ncol(data), alpha = 0.1,
    # This is replay of the independently completed, unbounded CPU search.
    # The older generic provider otherwise pads counts to the requested cap.
    max_conditioning_size = max(cpu_trace$level), provider = provider, trace_level = "full")
  list(result = result, speculative_cpu_checks = speculative)
}

orientation_reference <- function(skeleton, data) {
  env <- fastkpc_legacy_env()
  calls <- list()
  original <- env$regrVonPS
  env$regrVonPS <- function(G, V, S, suffStat, indepTest = env$kernelCItest, alpha = 0.2) {
    subset <- as.integer(S)
    parents <- which(G[, V] == 1 & G[V, ] == 0, arr.ind = TRUE)
    conditioning <- sort(unique(c(subset, as.integer(parents))))
    position <- 0L
    traced <- function(x, y, S = NULL, suffStat) {
      position <<- position + 1L
      p <- strict_p(indepTest(x = x, y = y, S = S, suffStat = suffStat))
      calls[[length(calls) + 1L]] <<- list(target = as.integer(V),
        other = subset[[position]], subset = subset,
        residual_conditioning_set = conditioning, p.value = p, rejected = p < alpha)
      p
    }
    original(G = G, V = V, S = subset, suffStat = suffStat, indepTest = traced, alpha = alpha)
  }
  pc <- env$udag2wanpdag(gInput = skeleton, suffStat = suff_stat(data),
    indepTest = env$kernelCItest, alpha = 0.1, verbose = FALSE,
    solve.confl = FALSE, orientCollider = TRUE, rules = c(TRUE, TRUE, TRUE))
  list(pdag = methods::as(pc@graph, "matrix"), trace = trace_frame(calls), rng_end = .Random.seed)
}

compare_orientation <- function(candidate, reference) {
  a <- trace_frame(candidate$ci_trace); b <- reference$trace
  identity <- setdiff(names(a), c("p_value", "rejected"))
  list(orientation_identity_order = identical(a[identity], b[identity]),
    orientation_p_values = numeric_check(a$p_value, b$p_value)$pass,
    orientation_decisions = identical(a$rejected, b$rejected),
    final_pdag = identical(dim(candidate$pdag), dim(reference$pdag)) &&
      identical(as.integer(candidate$pdag), as.integer(reference$pdag)))
}

if (dataset == "canonical_orientation") {
  data_path <- "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"
  data <- as.matrix(readRDS(data_path)); storage.mode(data) <- "double"
  receipt_path <- "fastkpc/artifacts/full_cuda_ci/real_data_parity_20261102/canonical/run/receipt.rds"
  receipt <- readRDS(receipt_path)
  stopifnot(all(unlist(receipt$gates)), identical(receipt$native_sha256, native_sha))
  oracle_dir <- "fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2"
  manifest <- jsonlite::read_json(file.path(oracle_dir, "manifest.json"), simplifyVector = TRUE)
  reference <- list(adjacency = readRDS(file.path(oracle_dir, "adjacency.rds")),
    pMax = readRDS(file.path(oracle_dir, "pmax.rds")),
    sepsets = readRDS(file.path(oracle_dir, "sepsets.rds")), n.edgetests = manifest$n_edgetests)
  set.seed(707L)
  started <- proc.time()[["elapsed"]]
  candidate <- fastkpc_orient_wanpdag_kpcalg_authority(receipt$result, data,
    alpha = 0.1, index = 1, numCol = 35L, ci_method = "dcc.gamma")
  gpu_rng <- .Random.seed
  candidate_seconds <- proc.time()[["elapsed"]] - started
  set.seed(707L)
  started <- proc.time()[["elapsed"]]
  cpu <- orientation_reference(fastkpc_skeleton_as_pc_algo(reference, data), data)
  cpu_seconds <- proc.time()[["elapsed"]] - started
  gates <- compare_orientation(candidate, cpu)
  gates$rng_endpoint <- identical(gpu_rng, cpu$rng_end)
  gates$sepsets_semantic <- identical(normalize_sepsets(receipt$result$sepsets),
                                    normalize_sepsets(reference$sepsets))
  saveRDS(list(candidate = candidate, cpu = cpu), file.path(out, "orientation.rds"), compress = FALSE)
  report <- list(dataset = "cancer_RD", n = nrow(data), p = ncol(data),
    evidence_class = "fresh CPU-authority orientation on new GPU skeleton versus archived CPU skeleton",
    native_sha256 = native_sha, input_sha256 = sha(data_path),
    skeleton_receipt_sha256 = sha(receipt_path),
    candidate_orientation_sec = candidate_seconds, reference_orientation_sec = cpu_seconds,
    orientation_ci_count = nrow(cpu$trace), gates = gates, pass = all(unlist(gates)))
} else {
  paths <- c(
    `20221022181603` = "causalDiscovery/20221022181603/CD8.c11.Tex.PDCD1-causalDiscoveryInput.RData",
    `20220418113957` = "causalDiscovery/20220418113957/logged-liver-exhausted-allData-causalDiscoveryInput.RData",
    `20240315113222` = "causalDiscovery/20240315113222/CD8.c11.Tex.PDCD1-causalDiscoveryInput.RData")
  stopifnot(dataset %in% names(paths))
  data_path <- paths[[dataset]]
  payload <- new.env(parent = emptyenv()); loaded <- load(data_path, envir = payload)
  stopifnot(length(loaded) == 1L)
  data <- as.matrix(payload[[loaded]]); storage.mode(data) <- "double"
  stopifnot(all(is.finite(data)), !is.null(colnames(data)))
  Sys.setenv(FASTKPC_DCC_GAMMA_EXACT_PRIMARY_OPTIMIZER = "1",
    FASTKPC_DCC_GAMMA_EXACT_PRIMARY_FIXED_REUSE = "1",
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES = "11811160064",
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS = "1",
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES = "8")
  stopifnot(fastkpc_cuda_available())
  invisible(full_cuda_ci_one_call_cache_control_native("reset"))
  resources_before <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
  set.seed(707L); rng_start <- .Random.seed
  cat("GPU START", dataset, paste(dim(data), collapse = "x"), "\n"); flush.console()
  started <- proc.time()[["elapsed"]]
  candidate <- fastkpc_compatible_cuda_wanpdag(data, alpha = 0.1,
    options = list(route = "full_cuda", compatible_cuda_strict = TRUE,
      max_conditioning_size = Inf, index = 1, numCol = 35L,
      trace_level = "logical", ci_method = "dcc.gamma"))
  gpu_seconds <- proc.time()[["elapsed"]] - started; gpu_rng <- .Random.seed
  invisible(full_cuda_ci_one_call_cache_control_native("reset")); gc(verbose = FALSE)
  resources_after <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
  owners_after <- fixed_sp_cuda_live_owner_snapshot()
  saveRDS(list(result = candidate, rng_start = rng_start, rng_end = gpu_rng,
    elapsed_sec = gpu_seconds, resources_before = resources_before,
    resources_after = resources_after, owners_after = owners_after),
    file.path(out, "gpu.rds"), compress = FALSE)
  cat("GPU DONE", gpu_seconds, "seconds; CPU START\n"); flush.console()

  env <- fastkpc_legacy_env(); calls <- list()
  traced <- function(x, y, S, suffStat) {
    p <- strict_p(env$kernelCItest(x, y, S, suffStat))
    calls[[length(calls) + 1L]] <<- data.frame(level = length(S), x = as.integer(x),
      y = as.integer(y), S_key = paste(sort(as.integer(S)), collapse = "|"),
      p_value = p, deleted = p >= 0.1, stringsAsFactors = FALSE)
    if (length(calls) %% 500L == 0L) { cat("CPU CI", length(calls), "\n"); flush.console() }
    p
  }
  set.seed(707L)
  started <- proc.time()[["elapsed"]]
  cpu_skeleton <- pcalg::skeleton(suffStat = suff_stat(data), indepTest = traced,
    alpha = 0.1, labels = colnames(data), m.max = Inf, method = "stable")
  cpu_trace <- do.call(rbind, calls); rownames(cpu_trace) <- NULL
  cpu_orientation <- orientation_reference(cpu_skeleton, data)
  cpu_rng <- .Random.seed; cpu_seconds <- proc.time()[["elapsed"]] - started
  saveRDS(list(skeleton = cpu_skeleton, trace = cpu_trace, orientation = cpu_orientation,
    elapsed_sec = cpu_seconds, rng_end = cpu_rng), file.path(out, "cpu.rds"), compress = FALSE)

  tasks <- candidate$skeleton$tasks
  active <- !tasks$native_edge_ignored
  gpu_keys <- key(tasks$level, tasks$x, tasks$y, tasks$S_key)
  cpu_keys <- key(cpu_trace$level, cpu_trace$x, cpu_trace$y, cpu_trace$S_key)
  # pcalg visits directed edges node-major; the established native contract
  # visits undirected edges and then both directions. Preserve that observed
  # difference, and independently replay CPU p-values in the CPU scheduler to
  # test the GPU's contractual task/deletion order without using GPU p-values.
  scheduled <- cpu_schedule_reference(data, cpu_trace)
  speculative_cpu_checks <- scheduled$speculative_cpu_checks
  cpu_schedule <- scheduled$result
  saveRDS(cpu_schedule, file.path(out, "cpu-schedule-replay.rds"), compress = FALSE)
  gpu_logical <- fastkpc_full_cuda_normalize_logical_trace(candidate$skeleton)
  cpu_logical <- fastkpc_full_cuda_normalize_logical_trace(cpu_schedule)
  order_fields <- c("logical_sequence_id", "source_task_index", "level", "x", "y", "S_key")
  matched <- match(gpu_keys, cpu_keys)
  cpu_all_p <- cpu_trace$p_value[matched]
  missing <- which(is.na(matched))
  # If a queued GPU evaluation was subsequently ignored, still check its p-value.
  for (i in missing) {
    S <- if (nzchar(tasks$S_key[[i]])) as.integer(strsplit(tasks$S_key[[i]], "|", fixed = TRUE)[[1L]]) else integer()
    cpu_all_p[[i]] <- strict_p(env$kernelCItest(tasks$x[[i]], tasks$y[[i]], S, suff_stat(data)))
  }
  ci <- numeric_check(tasks$p_used, cpu_all_p)
  pmax <- numeric_check(candidate$skeleton$pMax, cpu_skeleton@pMax)
  live <- c("live_device_allocations", "live_device_bytes", "live_streams", "live_events")
  authority_fields <- c("r_callback_count", "legacy_mgcv_fit_count", "legacy_mgcv_setup_count",
    "cpu_residual_solve_count", "cpu_dcov_component_count", "cpu_dcov_eigen_or_lowrank_count",
    "cpu_dcov_pair_stat_count", "cpu_gamma_pvalue_count", "cpu_spectra_count",
    "residual_d2h_bytes", "component_d2h_bytes", "unknown_fallback_count", "approximate_backend_count")
  s <- candidate$skeleton$summary
  gates <- list(all_ci_p_values = ci$pass, all_pmax = pmax$pass,
    task_identity_set = setequal(gpu_keys[active], cpu_keys),
    task_identity_order = identical(gpu_logical[order_fields], cpu_logical[order_fields]),
    task_identity_unique = !anyDuplicated(cpu_keys) && !anyDuplicated(gpu_keys),
    decision_trace = identical(gpu_logical$deletes_edge, cpu_logical$deletes_edge),
    cpu_schedule_p_values = numeric_check(gpu_logical$p_value, cpu_logical$p_value)$pass,
    cpu_schedule_matches_pcalg = identical(unname(cpu_schedule$adjacency != 0),
      unname(methods::as(cpu_skeleton@graph, "matrix") != 0)) &&
      identical(normalize_sepsets(cpu_schedule$sepsets), normalize_sepsets(cpu_skeleton@sepset)) &&
      identical(as.integer(cpu_schedule$n.edgetests), as.integer(cpu_skeleton@n.edgetests)) &&
      numeric_check(cpu_schedule$pMax, cpu_skeleton@pMax)$pass,
    zero_decision_flips = all((tasks$p_used >= 0.1) == (cpu_all_p >= 0.1)),
    skeleton_adjacency = identical(unname(candidate$skeleton$adjacency != 0),
                                  unname(methods::as(cpu_skeleton@graph, "matrix") != 0)),
    sepsets_semantic = identical(normalize_sepsets(candidate$skeleton$sepsets),
                                 normalize_sepsets(cpu_skeleton@sepset)),
    n_edgetests = identical(as.integer(candidate$skeleton$n.edgetests), as.integer(cpu_skeleton@n.edgetests)),
    rng_endpoint = identical(gpu_rng, cpu_rng),
    authority_zero = all(vapply(authority_fields, function(nm)
      !is.null(s[[nm]]) && identical(as.numeric(s[[nm]]), 0), logical(1L))),
    resources_zero = all(unlist(resources_before[live]) == 0) &&
      all(unlist(resources_after[live]) == 0) && owners_after$total == 0L,
    stable_svd = s$cuda_exact_fixed_evaluator_target_count == s$cuda_exact_fixed_evaluator_stable_svd_target_count,
    exact_primary = isTRUE(s$dcc_gamma_exact_primary_optimizer_requested) &&
      s$cuda_multi_penalty_target_count == s$cuda_multi_penalty_exact_primary_target_count &&
      s$cuda_multi_penalty_stability_replay_target_count == 0 && s$cuda_multi_penalty_terminal_confirmation_count == 0,
    cache_metadata = s$legacy_component_cache_route_mismatch_count == 0 && s$legacy_component_cache_status_mismatch_count == 0,
    fresh_caches = s$result_cache_warm_start_entries == 0 && s$target_cache_warm_start_entries == 0)
  gates <- c(gates, compare_orientation(candidate$orientation, cpu_orientation))
  saveRDS(list(tasks = tasks, cpu_p_values = cpu_all_p, cpu_trace = cpu_trace),
    file.path(out, "ci-comparison.rds"), compress = FALSE)
  report <- list(dataset = dataset, input_path = data_path, input_sha256 = sha(data_path),
    n = nrow(data), p = ncol(data), native_sha256 = native_sha,
    evidence_class = "fresh full real input, fresh GPU and pcalg/kpcalg CPU; correctness only",
    ci_method = "dcc.gamma", alpha = 0.1, numCol = 35L, max_conditioning_size = "Inf",
    configuration = "exact-primary, coefficient reuse, 11 GiB component cache, bypass, 8 eig lanes; canonical-only options disabled",
    logical_ci_count = nrow(tasks), cpu_ci_count = nrow(cpu_trace), extra_ignored_ci_checks = length(missing),
    raw_pcalg_call_order_exact = identical(gpu_keys[active], cpu_keys),
    task_order_reference = "existing CPU native scheduler replay of independently computed CPU p-values",
    speculative_cpu_checks = speculative_cpu_checks,
    ci = ci, pmax = pmax, decision_flip_count = sum((tasks$p_used >= .1) != (cpu_all_p >= .1)),
    skeleton_edges = sum(candidate$skeleton$adjacency != 0) / 2,
    orientation_ci_count = nrow(cpu_orientation$trace), gpu_elapsed_sec = gpu_seconds,
    cpu_elapsed_sec = cpu_seconds, authority_counters = s[authority_fields],
    resources_before = resources_before, resources_after = resources_after,
    gates = gates, pass = all(unlist(gates)))
}
report$native_unchanged <- identical(sha(native), native_sha)
report$pass <- report$pass && report$native_unchanged
report$session_info <- capture.output(sessionInfo())
jsonlite::write_json(report, file.path(out, "report.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
print(report[setdiff(names(report), c("session_info", "resources_before", "resources_after"))])
stopifnot(isTRUE(report$pass))
