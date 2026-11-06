# One historical input remains a regression; it does not select the implementation.
args <- commandArgs(TRUE)
stopifnot(length(args) == 3L, !file.exists(args[[2L]]))
native <- normalizePath(args[[1L]]); out <- args[[2L]]; baseline <- args[[3L]]
dir.create(out)
source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
source("fastkpc/tools/research/exact_speedup_20261020/pinned_optimizer_gates.R")
.fastkpc_cuda_so <- local({ path <- native; function() path })
flags <- c("FASTKPC_DCC_GAMMA_EXACT_PRIMARY_OPTIMIZER", "FASTKPC_DCC_GAMMA_EXACT_PRIMARY_FIXED_REUSE",
  "FASTKPC_DCC_GAMMA_LEVEL_COMPONENT_PREFILL", "FASTKPC_DCC_GAMMA_STRUCTURED_COMPONENT_PREFILL",
  "FASTKPC_DCC_GAMMA_L2_COMPONENT_PREFILL", "FASTKPC_DCC_GAMMA_L3_OPTIMIZER_SLOTS",
  "FASTKPC_DCC_GAMMA_L4_OPTIMIZER_WORKERS32", "FASTKPC_DCC_GAMMA_L4_OPTIMIZER_SLOTS",
  "FASTKPC_DCC_GAMMA_L5_OPTIMIZER_SLOTS", "FASTKPC_DCC_GAMMA_L6_OPTIMIZER_SLOTS",
  "FASTKPC_DCC_GAMMA_VECTOR_POSTBACK", "FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H",
  "FASTKPC_DCC_GAMMA_STRUCTURED_WORK_PAIR", "FASTKPC_CUDA_DEVICE_PROPERTIES_CACHE")
do.call(Sys.setenv, as.list(setNames(rep("1", length(flags)), flags)))
Sys.setenv(FASTKPC_PHASE10_SETUP_OPTIMIZER_PIPELINE = "0")
stopifnot(fastkpc_cuda_available())
data <- as.matrix(readRDS("fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"))
storage.mode(data) <- "double"
reference <- readRDS(baseline)
invisible(full_cuda_ci_one_call_cache_control_native("configure", 262144L))
invisible(full_cuda_ci_one_call_cache_control_native("configure_target", 131072L))
invisible(full_cuda_ci_one_call_cache_control_native("reset"))
.Random.seed <- reference$seed_before
before <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
started <- proc.time()[["elapsed"]]
result <- fastkpc_compatible_cuda_skeleton(data, alpha = .1, options = list(
  route = "full_cuda", compatible_cuda_strict = TRUE, max_conditioning_size = Inf,
  index = 1, numCol = 35L, trace_level = "logical", ci_method = "dcc.gamma"))
elapsed <- proc.time()[["elapsed"]] - started
receipt <- list(workload = "canonical", result = result, elapsed_sec = elapsed,
  native_sha256 = digest::digest(file = native, algo = "sha256", serialize = FALSE),
  seed_before = reference$seed_before, seed_after = .Random.seed,
  resources_before = before, resources_after = fastkpc_full_cuda_phase35_vertical_resource_snapshot())
saveRDS(receipt, file.path(out, "pending.rds"), compress = FALSE)
receipt <- qualify_pinned_optimizer_receipt(receipt, baseline)
g <- receipt$gates; s <- result$summary
g$fresh_caches <- s$result_cache_warm_start_entries == 0 && s$target_cache_warm_start_entries == 0
g$stable_svd <- s$cuda_exact_fixed_evaluator_target_count == s$cuda_exact_fixed_evaluator_stable_svd_target_count
g$cache_metadata <- s$legacy_component_cache_route_mismatch_count == 0 && s$legacy_component_cache_status_mismatch_count == 0
g$arena_closed <- s$exact_primary_fixed_reuse_allocation_count == s$exact_primary_fixed_reuse_free_count &&
  s$exact_primary_fixed_reuse_live_device_bytes_after == 0 && s$exact_primary_fixed_reuse_fallback_target_count == 0
g$owners_closed <- fixed_sp_cuda_live_owner_snapshot()$total == 0
g$resources_zero <- all(unlist(receipt$resources_after[
  c("live_device_allocations", "live_device_bytes", "live_streams", "live_events")]) == 0)
g$exact_primary <- s$cuda_multi_penalty_exact_primary_target_count == s$cuda_multi_penalty_target_count &&
  s$cuda_multi_penalty_stability_replay_target_count == 0 && s$cuda_multi_penalty_terminal_confirmation_count == 0
for (level in 3:6) {
  pool <- s[[paste0("l", level, "_optimizer_slots")]]
  free_checkpoint <- pool$minimum_free_bytes
  g[[paste0("l", level, "_pool_lifecycle")]] <- isTRUE(pool$enabled) && pool$created > 0 &&
    pool$created <= 32 && pool$destroyed == pool$created && pool$targets > 0 &&
    pool$optimizer_binds == pool$setups && pool$stage_a_binds == pool$setups &&
    isTRUE(free_checkpoint >= 2 * 1024^3)
}
# Final graph authority is unchanged; compare its output and RNG with the
# archived, independently CPU-qualified skeleton, once outside skeleton timing.
for (file in c("fastkpc/tests/test_compatible_cuda_wanpdag_authority.R",
               "fastkpc/tools/research/real_data_parity_20261102/run.R"))
  for (e in parse(file))
    if (is.call(e) && identical(e[[1L]], as.name("<-")) && is.symbol(e[[2L]]) &&
        as.character(e[[2L]]) %in% c("trace_frame", "numeric_check", "compare_orientation")) eval(e)
set.seed(707L)
actual <- fastkpc_orient_wanpdag_kpcalg_authority(result, data, alpha = .1,
  index = 1, numCol = 35L, ci_method = "dcc.gamma")
rng <- .Random.seed
set.seed(707L)
expected <- fastkpc_orient_wanpdag_kpcalg_authority(reference$result, data, alpha = .1,
  index = 1, numCol = 35L, ci_method = "dcc.gamma")
stopifnot(is.matrix(actual$pdag), is.matrix(expected$pdag))
g <- c(g, compare_orientation(actual, list(trace = trace_frame(expected$ci_trace), pdag = expected$pdag)))
g$orientation_rng <- identical(rng, .Random.seed)
receipt$gates <- g
saveRDS(receipt, file.path(out, "receipt.rds"), compress = FALSE)
passed <- all(vapply(g, isTRUE, logical(1L)))
jsonlite::write_json(list(status = if (passed) "PASS" else "FAIL", gates = g,
  native_sha256 = receipt$native_sha256, baseline_native_sha256 = reference$native_sha256,
  elapsed_sec = elapsed, ci_count = nrow(result$tasks), oracle_errors = receipt$oracle_errors,
  pools = s[paste0("l", 3:6, "_optimizer_slots")],
  level_component_prefill = s$level_component_prefill,
  evidence_class = "single development regression, not a new repeated performance baseline"),
  file.path(out, "report.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
print(g); stopifnot(passed)
cat("PASS canonical regression", elapsed, "seconds\n")
