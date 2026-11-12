# Runs in the clean checkout; this external harness does not alter its sources.
args <- commandArgs(TRUE)
stopifnot(length(args) == 5L, !file.exists(args[[3L]]))
mode <- args[[1L]]; input <- args[[2L]]; out <- args[[3L]]
reference_path <- args[[4L]]; phase <- args[[5L]]
stopifnot(mode %in% c("cpu", "replay", "off", "on"), phase %in% c("correctness", "timing"))
dir.create(out, recursive = TRUE)
source("fastkpc/tools/research/numcol_20261106/common.R")
for (e in parse("fastkpc/tools/research/real_data_parity_20261102/run.R"))
  if (is.call(e) && identical(e[[1L]], as.name("<-")) && is.symbol(e[[2L]]) &&
      as.character(e[[2L]]) == "key") eval(e)
sha <- function(path) digest::digest(file = path, algo = "sha256", serialize = FALSE)
legacy_sha <- sha("kpcalg/R/dcovgamma.R")
stopifnot(legacy_sha == "d0620dd9b06f006170b5e142b892c202619296797b2bf9ca842a7c2b5b3b715f")
data <- as.matrix(readRDS(input)); storage.mode(data) <- "double"
case <- sub("\\.rds$", "", basename(input)); k <- as.integer(floor(nrow(data)/10))
input_sha <- sha(input)
# Both public CPU and GPU entries really receive omitted numCol/m.max options.
suff_stat <- function(data) list(data = data, ic.method = "dcc.gamma", index = 1)
write_report <- function(report) {
  report$case <- case; report$n <- nrow(data); report$p <- ncol(data); report$numCol <- k
  report$input_sha256 <- input_sha; report$legacy_r_sha256 <- legacy_sha
  report$session_info <- capture.output(sessionInfo())
  jsonlite::write_json(report, file.path(out, "report.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
  print(report[setdiff(names(report), c("session_info", "execution", "levels"))])
  stopifnot(isTRUE(report$pass), identical(legacy_sha, sha("kpcalg/R/dcovgamma.R")))
}
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
cat("START", case, mode, phase, nrow(data), ncol(data), "default rank", k, "\n"); flush.console()

if (mode == "cpu") {
  env <- fastkpc_legacy_env(); calls <- list()
  traced <- function(x, y, S, suffStat) {
    value <- strict_p(env$kernelCItest(x, y, S, suffStat))
    calls[[length(calls) + 1L]] <<- data.frame(level = length(S), x = as.integer(x), y = as.integer(y),
      S_key = paste(sort(as.integer(S)), collapse = "|"), p_value = value, deleted = value >= .1)
    if (length(calls) %% 250L == 0L) { cat("CPU CI", length(calls), "\n"); flush.console() }
    value
  }
  set.seed(707L); seed_before <- .Random.seed
  started <- proc.time()[["elapsed"]]
  cpu <- pcalg::skeleton(suffStat = suff_stat(data), indepTest = traced, alpha = .1,
    labels = colnames(data), m.max = Inf, method = "stable")
  skeleton_elapsed <- proc.time()[["elapsed"]] - started
  orientation <- orientation_reference(cpu, data)
  outer_elapsed <- proc.time()[["elapsed"]] - started
  trace <- do.call(rbind, calls); rownames(trace) <- NULL
  stopifnot(!anyDuplicated(key(trace$level, trace$x, trace$y, trace$S_key)))
  saveRDS(list(skeleton = cpu, trace = trace, orientation = orientation,
    input_sha256 = input_sha, numCol = k, seed_before = seed_before, seed_after = .Random.seed,
    legacy_r_sha256 = legacy_sha), file.path(out, "receipt.rds"), compress = FALSE)
  write_report(list(pass = TRUE, ci_count = nrow(trace), orientation_ci_count = nrow(orientation$trace),
    max_level = max(trace$level), n_edgetests = cpu@n.edgetests,
    skeleton_sec = skeleton_elapsed, outer_sec = outer_elapsed,
    authority = "unmodified kpcalg R/mgcv/RSpectra and independent pcalg search; no native numerical provider"))
  quit(save = "no", status = 0L)
}

native <- normalizePath("fastkpc/build/fastkpc_cuda.so"); native_sha <- sha(native)
reference <- readRDS(reference_path)
stopifnot(identical(reference$input_sha256, input_sha), reference$numCol == k,
  identical(reference$legacy_r_sha256, legacy_sha), fastkpc_cuda_available())
stopifnot(identical(normalizePath(getLoadedDLLs()[["fastkpc_cuda"]][["path"]]), native))

if (mode == "replay") {
  # CPU values are computed before and independently of any GPU result.
  table <- reference$trace; env <- fastkpc_legacy_env(); extra <- 0L
  provider <- function(planned, level) {
    if (!nrow(planned)) return(numeric())
    indices <- match(key(level, planned$x, planned$y, planned$S_key),
      key(table$level, table$x, table$y, table$S_key))
    values <- table$p_value[indices]
    for (j in which(is.na(indices))) {
      values[[j]] <- strict_p(env$kernelCItest(planned$x[[j]], planned$y[[j]],
        as.integer(planned$conditioning_sets[[j]]), suff_stat(data)))
      table <<- rbind(table, data.frame(level = level, x = planned$x[[j]], y = planned$y[[j]],
        S_key = planned$S_key[[j]], p_value = values[[j]], deleted = values[[j]] >= .1))
      extra <<- extra + 1L
    }
    values
  }
  set.seed(707L)
  replay <- precision_run_skeleton_provider_native(p = ncol(data), alpha = .1,
    max_conditioning_size = max(reference$trace$level), provider = provider, trace_level = "full")
  cpu <- reference$skeleton
  gates <- list(adjacency = identical(unname(replay$adjacency != 0), unname(methods::as(cpu@graph, "matrix") != 0)),
    pmax = numeric_check(replay$pMax, cpu@pMax)$pass,
    sepsets = identical(normalize_sepsets(replay$sepsets), normalize_sepsets(cpu@sepset)),
    counts = identical(as.integer(replay$n.edgetests), as.integer(cpu@n.edgetests)))
  dimnames(replay$adjacency) <- dimnames(replay$pMax) <- list(colnames(data), colnames(data))
  reference$replay <- replay; reference$all_cpu_values <- table
  reference$replay_native_sha256 <- native_sha; reference$speculative_cpu_checks <- extra
  saveRDS(reference, file.path(out, "receipt.rds"), compress = FALSE)
  write_report(list(pass = all(vapply(gates, isTRUE, logical(1L))), gates = gates, ci_count = nrow(reference$trace),
    speculative_cpu_checks = extra, native_sha256 = native_sha,
    order_reference = "native host scheduling of independent R p-values; raw pcalg node-major order is distinct"))
  quit(save = "no", status = 0L)
}

rank_mode(mode)
if (phase == "timing") {
  # Warm only an unrelated input, then clear every dataset-specific cache.
  set.seed(19007L)
  warm <- matrix(rnorm(80L * 3L), 80L, 3L); colnames(warm) <- paste0("warm", 1:3)
  invisible(fastkpc_compatible_cuda_wanpdag(warm, .1,
    options = list(max_conditioning_size = 0L, trace_level = "logical")))
}
invisible(full_cuda_ci_one_call_cache_control_native("reset"))
before <- rank_resources()
set.seed(707L); seed_before <- .Random.seed
started <- proc.time()[["elapsed"]]
candidate <- fastkpc_compatible_cuda_wanpdag(data, alpha = .1,
  options = list(ci_method = "dcc.gamma", trace_level = "logical"))
outer_elapsed <- proc.time()[["elapsed"]] - started; seed_after <- .Random.seed
after <- rank_resources()
result <- candidate$skeleton; s <- result$summary; tasks <- result$tasks
oracle <- fastkpc_full_cuda_phase10_development_oracle(reference$replay)
comparison <- fastkpc_full_cuda_compare_candidate_skeleton(oracle, result)
gpu_keys <- key(tasks$level, tasks$x, tasks$y, tasks$S_key)
table <- reference$all_cpu_values
positions <- match(gpu_keys, key(table$level, table$x, table$y, table$S_key))
values <- table$p_value[positions]
# Speculatively evaluated GPU tasks are also checked, outside the timer.
for (j in which(is.na(positions))) {
  subset <- if (nzchar(tasks$S_key[[j]])) as.integer(strsplit(tasks$S_key[[j]], "|", fixed = TRUE)[[1L]]) else integer()
  values[[j]] <- strict_p(fastkpc_legacy_env()$kernelCItest(tasks$x[[j]], tasks$y[[j]], subset, suff_stat(data)))
}
ci <- numeric_check(tasks$p_used, values)
pmax <- numeric_check(result$pMax, reference$skeleton@pMax)
logical <- fastkpc_full_cuda_normalize_logical_trace(result)
cpu_logical <- fastkpc_full_cuda_normalize_logical_trace(reference$replay)
order_fields <- c("logical_sequence_id", "source_task_index", "level", "x", "y", "S_key")
gates <- c(list(all_ci = ci$pass, all_pmax = pmax$pass, graph_trace = isTRUE(comparison$summary$pass),
  task_identity_order = identical(logical[order_fields], cpu_logical[order_fields]),
  deletion_trace = identical(logical$deletes_edge, cpu_logical$deletes_edge),
  decision_flips_zero = all((tasks$p_used >= .1) == (values >= .1)),
  original_pcalg_task_set = setequal(gpu_keys[!tasks$native_edge_ignored],
    key(reference$trace$level, reference$trace$x, reference$trace$y, reference$trace$S_key)),
  adjacency = identical(unname(result$adjacency != 0), unname(methods::as(reference$skeleton@graph, "matrix") != 0)),
  sepsets = identical(normalize_sepsets(result$sepsets), normalize_sepsets(reference$skeleton@sepset)),
  counts = identical(as.integer(result$n.edgetests), as.integer(reference$skeleton@n.edgetests)),
  rng = identical(seed_after, reference$seed_after), default_numCol = s$numCol == k,
  authority = all(vapply(rank_authority, function(field) !is.null(s[[field]]) && s[[field]] == 0, logical(1L))),
  resources = TRUE, stable_svd = s$cuda_exact_fixed_evaluator_target_count == s$cuda_exact_fixed_evaluator_stable_svd_target_count,
  exact_primary = isTRUE(s$dcc_gamma_exact_primary_optimizer_requested) &&
    s$cuda_multi_penalty_target_count == s$cuda_multi_penalty_exact_primary_target_count &&
    s$cuda_multi_penalty_stability_replay_target_count == 0 && s$cuda_multi_penalty_terminal_confirmation_count == 0,
  cache_metadata = s$legacy_component_cache_route_mismatch_count == 0 && s$legacy_component_cache_status_mismatch_count == 0,
  fresh_caches = s$result_cache_warm_start_entries == 0 && s$target_cache_warm_start_entries == 0,
  native_unchanged = identical(sha(native), native_sha)), compare_orientation(candidate$orientation, reference$orientation))
if (mode == "on") {
  expected <- nrow(data) >= 6 && nrow(data) <= 1024 && k <= 64 && k+1 < nrow(data) && 2*(k+1)+4 <= nrow(data)
  gates$shape_dispatch <- isTRUE(s$structured_component_prefill_requested) &&
    identical(isTRUE(s$structured_component_prefill_enabled), expected)
  if (case %in% c("dense500", "dense1025")) gates$producer_exercised <- s$level_component_produced > 0
  if (case == "dense500") gates$deep_search <- max(logical$level) >= 6L
}
timing <- setNames(candidate$timings$elapsed_sec, candidate$timings$stage)
saveRDS(list(candidate = candidate, seed_before = seed_before, seed_after = seed_after,
  input_sha256 = input_sha, native_sha256 = native_sha, legacy_r_sha256 = legacy_sha,
  outer_sec = outer_elapsed, cpu_p_values = values, resources_before = before, resources_after = after),
  file.path(out, "receipt.rds"), compress = FALSE)
write_report(list(pass = all(vapply(gates, isTRUE, logical(1L))), gates = gates, mode = mode, phase = phase,
  native_sha256 = native_sha, ci_count = nrow(tasks), orientation_ci_count = nrow(reference$orientation$trace),
  ci = ci, pmax = pmax, max_level = max(logical$level), n_edgetests = result$n.edgetests,
  outer_sec = outer_elapsed, skeleton_sec = unname(timing[["skeleton"]]), orientation_sec = unname(timing[["orientation"]]),
  extra_cpu_checks = sum(is.na(positions)),
  raw_pcalg_call_order_exact = identical(gpu_keys[!tasks$native_edge_ignored],
    key(reference$trace$level, reference$trace$x, reference$trace$y, reference$trace$S_key)),
  levels = result$levels, execution = s[grep("^(l[3456]_optimizer_slot|level_component|structured_component|cuda_structured)", names(s))]))
cat("PASS", case, mode, phase, "CI", nrow(tasks), "outer", outer_elapsed, "\n")
