# Public-input adaptation of the d7b0815 raw-receipt audit. Only the expected
# run count is generalized; numerical, graph, RNG and resource checks are retained.
# Never load or execute the native.
args <- commandArgs(TRUE)
stopifnot(length(args) == 4L, !file.exists(args[[4L]]),
  requireNamespace("pcalg", quietly = TRUE), !("fastkpc_cuda" %in% names(getLoadedDLLs())))
reference_root <- args[[1L]]; qualification_root <- args[[2L]]; timing_root <- args[[3L]]
source("fastkpc/R/full_cuda_ci_gate.R")
for (file in c("fastkpc/tests/test_compatible_cuda_wanpdag_authority.R",
               "fastkpc/tools/research/real_data_parity_20261102/run.R",
               "fastkpc/tools/research/numcol_20261106/common.R")) {
  for (expression in parse(file)) {
    if (is.call(expression) && identical(expression[[1L]], as.name("<-")) &&
        is.symbol(expression[[2L]]) && as.character(expression[[2L]]) %in%
        c("normalize_sepsets", "trace_frame", "numeric_check", "compare_orientation",
          "rank_authority", "rank_live")) eval(expression)
  }
}
read_json <- function(path) jsonlite::read_json(path, simplifyVector = FALSE)
key <- function(tasks) paste(tasks$level, tasks$x, tasks$y, tasks$S_key, sep = "\037")
near <- function(a, b) isTRUE(all.equal(as.double(a), as.double(b), tolerance = 1e-12))
zero <- function(x) is.numeric(x) && length(x) == 1L && is.finite(x) && x == 0
qualification <- read_json(file.path(qualification_root, "summary.json"))
timing <- read_json(file.path(timing_root, "summary.json"))
stopifnot(isTRUE(qualification$pass), isTRUE(timing$pass),
  setequal(names(qualification$reports), names(timing$reports)))
rows <- list(); ci_count <- 0L; orientation_count <- 0L; run_count <- 0L
for (case in names(qualification$reports)) {
  original <- readRDS(file.path(reference_root, case, "receipt.rds"))
  replay <- readRDS(file.path(qualification_root, paste0(case, "-replay"), "receipt.rds"))
  values <- replay$all_cpu_values
  stopifnot(!anyDuplicated(key(values)), identical(replay$input_sha256, original$input_sha256))
  original_positions <- match(key(original$trace), key(values))
  stopifnot(!anyNA(original_positions),
    identical(values$p_value[original_positions], original$trace$p_value))
  expected_trace <- fastkpc_full_cuda_normalize_logical_trace(replay$replay)
  order_fields <- c("logical_sequence_id", "source_task_index", "level", "x", "y", "S_key")
  specs <- list(
    off = list(root = qualification_root, stem = paste0(case, "-off"), mode = "off",
      phase = "correctness", report = qualification$reports[[case]]$off),
    on = list(root = qualification_root, stem = paste0(case, "-on"), mode = "on",
      phase = "correctness", report = qualification$reports[[case]]$on))
  for (label in c("A1", "C1", "C2", "A2", "A3", "C3")) {
    specs[[label]] <- list(root = timing_root, stem = paste0(case, "-", label),
      mode = if (startsWith(label, "A")) "off" else "on", phase = "timing",
      report = timing$reports[[case]]$runs[[label]])
  }
  runs <- list()
  for (label in names(specs)) {
    spec <- specs[[label]]; report <- spec$report
    receipt <- readRDS(file.path(spec$root, spec$stem, "receipt.rds"))
    result <- receipt$candidate$skeleton; s <- result$summary; tasks <- result$tasks
    positions <- match(key(tasks), key(values))
    stopifnot(!anyNA(positions), identical(report$mode, spec$mode), identical(report$phase, spec$phase),
      identical(receipt$input_sha256, original$input_sha256),
      identical(receipt$native_sha256, timing$native_sha256),
      identical(receipt$legacy_r_sha256, original$legacy_r_sha256),
      identical(receipt$seed_before, original$seed_before), identical(receipt$seed_after, original$seed_after),
      s$numCol == original$numCol, report$numCol == original$numCol,
      identical(receipt$cpu_p_values, values$p_value[positions]))
    ci <- numeric_check(tasks$p_used, values$p_value[positions])
    pmax <- numeric_check(result$pMax, original$skeleton@pMax)
    actual_trace <- fastkpc_full_cuda_normalize_logical_trace(result)
    orientation <- compare_orientation(receipt$candidate$orientation, original$orientation)
    stopifnot(ci$pass, pmax$pass, all(vapply(orientation, isTRUE, logical(1L))),
      identical(actual_trace[order_fields], expected_trace[order_fields]),
      identical(actual_trace$deletes_edge, expected_trace$deletes_edge),
      all((tasks$p_used >= .1) == (values$p_value[positions] >= .1)),
      setequal(key(tasks)[!tasks$native_edge_ignored], key(original$trace)),
      identical(unname(result$adjacency != 0), unname(methods::as(original$skeleton@graph, "matrix") != 0)),
      identical(normalize_sepsets(result$sepsets), normalize_sepsets(original$skeleton@sepset)),
      identical(as.integer(result$n.edgetests), as.integer(original$skeleton@n.edgetests)),
      all(vapply(s[rank_authority], zero, logical(1L))),
      all(vapply(receipt$resources_before[rank_live], zero, logical(1L))),
      all(vapply(receipt$resources_after[rank_live], zero, logical(1L))),
      s$cuda_exact_fixed_evaluator_target_count == s$cuda_exact_fixed_evaluator_stable_svd_target_count,
      isTRUE(s$dcc_gamma_exact_primary_optimizer_requested),
      s$cuda_multi_penalty_target_count == s$cuda_multi_penalty_exact_primary_target_count)
    zero_fields <- c("cuda_multi_penalty_stability_replay_target_count", "cuda_multi_penalty_terminal_confirmation_count",
      "legacy_component_cache_route_mismatch_count", "legacy_component_cache_status_mismatch_count",
      "result_cache_warm_start_entries", "target_cache_warm_start_entries",
      "fixed_sp_root_cache_identity_rejection_count", "exact_primary_fixed_reuse_fallback_target_count",
      "exact_primary_fixed_reuse_live_device_bytes_after")
    stopifnot(all(vapply(s[zero_fields], zero, logical(1L))),
      s$exact_primary_fixed_reuse_allocation_count == s$exact_primary_fixed_reuse_free_count)
    pools <- s[paste0("l", 3:6, "_optimizer_slots")]
    for (pool in pools) {
      stopifnot(!is.null(pool), pool$created == pool$destroyed, pool$created <= 32)
      if (pool$created > 0) stopifnot(pool$minimum_free_bytes >= 2 * 1024^3)
    }
    if (spec$mode == "on") {
      expected <- report$n >= 6 && report$n <= 1024 && report$numCol <= 64 &&
        report$numCol + 1 < report$n && 2 * (report$numCol + 1) + 4 <= report$n
      stopifnot(isTRUE(s$structured_component_prefill_requested),
        identical(isTRUE(s$structured_component_prefill_enabled), expected))
      if (case %in% c("dense500", "dense1025")) stopifnot(s$level_component_produced > 0)
      if (case == "dense500") stopifnot(max(actual_trace$level) >= 6L)
    }
    stages <- setNames(receipt$candidate$timings$elapsed_sec, receipt$candidate$timings$stage)
    stopifnot(isTRUE(report$pass), all(vapply(report$gates, isTRUE, logical(1L))),
      nrow(tasks) == report$ci_count, nrow(original$orientation$trace) == report$orientation_ci_count,
      near(ci$max_abs_error, report$ci$max_abs_error), near(pmax$max_abs_error, report$pmax$max_abs_error),
      near(receipt$outer_sec, report$outer_sec), near(stages[["skeleton"]], report$skeleton_sec),
      near(stages[["orientation"]], report$orientation_sec))
    runs[[label]] <- list(pass = TRUE, mode = spec$mode, phase = spec$phase,
      ci_count = nrow(tasks), orientation_ci_count = nrow(original$orientation$trace),
      ci_error = ci$max_abs_error, pmax_error = pmax$max_abs_error,
      outer_sec = receipt$outer_sec, skeleton_sec = unname(stages[["skeleton"]]),
      orientation_sec = unname(stages[["orientation"]]),
      minimum_pool_free_bytes = if (any(vapply(pools, function(p) p$created > 0, logical(1L))))
        min(vapply(Filter(function(p) p$created > 0, pools), function(p) p$minimum_free_bytes, numeric(1L))) else NULL)
    ci_count <- ci_count + nrow(tasks); orientation_count <- orientation_count + nrow(original$orientation$trace)
    run_count <- run_count + 1L
  }
  rows[[case]] <- runs
  cat("PASS raw receipt audit", case, length(runs), "runs\n")
}
stopifnot(run_count == 8L * length(qualification$reports),
  !("fastkpc_cuda" %in% names(getLoadedDLLs())))
jsonlite::write_json(list(pass = TRUE, run_count = run_count, ci_comparisons = ci_count,
  orientation_ci_comparisons = orientation_count, native_loaded = FALSE, cases = rows),
  args[[4L]], pretty = TRUE, auto_unbox = TRUE, digits = NA)
