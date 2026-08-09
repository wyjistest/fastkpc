source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase10_compute_campaign.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!isTRUE(tryCatch(
  fastkpc_cuda_available(),
  error = function(error) FALSE
))) {
  cat("SKIP full CUDA CI legacy component cache: CUDA unavailable\n")
  quit(save = "no", status = 0L)
}

cache_environment <- "FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES"
bypass_environment <-
  "FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS"
lane_environment <- "FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES"
wave_environment <- "FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_WAVE_CAPACITY"
positive_miss_environment <-
  "FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_WAVE_POSITIVE_MISS_ONLY"
old_cache_budget <- Sys.getenv(cache_environment, unset = NA_character_)
old_bypass <- Sys.getenv(bypass_environment, unset = NA_character_)
old_lanes <- Sys.getenv(lane_environment, unset = NA_character_)
old_wave <- Sys.getenv(wave_environment, unset = NA_character_)
old_positive_miss <- Sys.getenv(
  positive_miss_environment, unset = NA_character_
)
on.exit({
  try(full_cuda_ci_one_call_cache_control_native("reset"), silent = TRUE)
  if (is.na(old_cache_budget)) {
    Sys.unsetenv(cache_environment)
  } else {
    do.call(Sys.setenv, setNames(list(old_cache_budget), cache_environment))
  }
  if (is.na(old_bypass)) {
    Sys.unsetenv(bypass_environment)
  } else {
    do.call(Sys.setenv, setNames(list(old_bypass), bypass_environment))
  }
  if (is.na(old_lanes)) {
    Sys.unsetenv(lane_environment)
  } else {
    do.call(Sys.setenv, setNames(list(old_lanes), lane_environment))
  }
  if (is.na(old_wave)) {
    Sys.unsetenv(wave_environment)
  } else {
    do.call(Sys.setenv, setNames(list(old_wave), wave_environment))
  }
  if (is.na(old_positive_miss)) {
    Sys.unsetenv(positive_miss_environment)
  } else {
    do.call(Sys.setenv, setNames(
      list(old_positive_miss), positive_miss_environment
    ))
  }
}, add = TRUE)

set.seed(6001)
n <- 90L
p <- 8L
data <- matrix(stats::rnorm(n * p), n, p)
for (column in 2:p) {
  parents <- which(stats::runif(column - 1L) < 0.65)
  if (length(parents) > 0L) {
    data[, column] <- data[, column] +
      data[, parents, drop = FALSE] %*%
        stats::runif(length(parents), 0.35, 0.9)
  }
}
colnames(data) <- paste0("v", seq_len(p))

live_fields <- c(
  "live_device_allocations", "live_device_bytes", "live_streams",
  "live_events"
)
run_candidate <- function(cache_bytes, residual_bypass = FALSE,
                          eig_lanes = 0L, wave_capacity = 1L,
                          positive_miss_only = FALSE) {
  full_cuda_ci_one_call_cache_control_native("reset")
  Sys.setenv(FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES =
               as.character(cache_bytes))
  Sys.setenv(FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS =
               if (isTRUE(residual_bypass)) "1" else "0")
  Sys.setenv(FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES =
               as.character(eig_lanes))
  Sys.setenv(FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_WAVE_CAPACITY =
               as.character(wave_capacity))
  Sys.setenv(FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_WAVE_POSITIVE_MISS_ONLY =
               if (isTRUE(positive_miss_only)) "1" else "0")
  resources_before <-
    fastkpc_full_cuda_phase35_vertical_resource_snapshot()
  started <- proc.time()[["elapsed"]]
  result <- precision_run_skeleton_full_cuda_native(
    data = data,
    alpha = 0.1,
    max_conditioning_size = 3L,
    index = 1,
    numCol = 35L,
    trace_level = "logical",
    compatible_cuda_strict = TRUE
  )
  elapsed <- proc.time()[["elapsed"]] - started
  resources_after <-
    fastkpc_full_cuda_phase35_vertical_resource_snapshot()
  assert_true(
    identical(resources_after[live_fields], resources_before[live_fields]),
    "legacy component cache leaked tracked CUDA resources"
  )
  list(result = result, elapsed = elapsed)
}

uncached_run <- run_candidate(0)
cached_run <- run_candidate(128 * 1024^2)
bypassed_run <- run_candidate(128 * 1024^2, residual_bypass = TRUE)
persistent_run <- run_candidate(
  128 * 1024^2, residual_bypass = TRUE, eig_lanes = 1L
)
multilane_run <- run_candidate(
  128 * 1024^2, residual_bypass = TRUE, eig_lanes = 8L
)
selective_wave_run <- run_candidate(
  128 * 1024^2, residual_bypass = TRUE, eig_lanes = 8L,
  wave_capacity = 2L, positive_miss_only = TRUE
)
uncached <- uncached_run$result
cached <- cached_run$result
bypassed <- bypassed_run$result
persistent <- persistent_run$result
multilane <- multilane_run$result
selective_wave <- selective_wave_run$result

canonicalize_tasks <- function(value) {
  value[order(value$level, value$task_index), , drop = FALSE]
}
normalize_sepsets <- function(value) {
  unname(lapply(unname(value), function(row) {
    unname(lapply(unname(row), function(entry) sort(as.integer(entry))))
  }))
}

uncached_tasks <- canonicalize_tasks(uncached$tasks)
cached_tasks <- canonicalize_tasks(cached$tasks)
bypassed_tasks <- canonicalize_tasks(bypassed$tasks)
persistent_tasks <- canonicalize_tasks(persistent$tasks)
multilane_tasks <- canonicalize_tasks(multilane$tasks)
selective_wave_tasks <- canonicalize_tasks(selective_wave$tasks)
assert_true(
  identical(cached_tasks$p_used, uncached_tasks$p_used) &&
    identical(cached$pMax, uncached$pMax) &&
    identical(cached$adjacency, uncached$adjacency) &&
    identical(normalize_sepsets(cached$sepsets),
              normalize_sepsets(uncached$sepsets)) &&
    identical(as.integer(cached$n.edgetests),
              as.integer(uncached$n.edgetests)),
  "legacy component cache changed a strict dcc.gamma result"
)
assert_true(
  identical(bypassed_tasks$p_used, uncached_tasks$p_used) &&
    identical(bypassed$pMax, uncached$pMax) &&
    identical(bypassed$adjacency, uncached$adjacency) &&
    identical(normalize_sepsets(bypassed$sepsets),
              normalize_sepsets(uncached$sepsets)) &&
    identical(as.integer(bypassed$n.edgetests),
              as.integer(uncached$n.edgetests)),
  "legacy component cache residual bypass changed a strict dcc.gamma result"
)
assert_true(
  identical(persistent_tasks$p_used, uncached_tasks$p_used) &&
    identical(multilane_tasks$p_used, uncached_tasks$p_used) &&
    identical(selective_wave_tasks$p_used, uncached_tasks$p_used) &&
    identical(persistent$pMax, uncached$pMax) &&
    identical(multilane$pMax, uncached$pMax) &&
    identical(selective_wave$pMax, uncached$pMax) &&
    identical(persistent$adjacency, uncached$adjacency) &&
    identical(multilane$adjacency, uncached$adjacency) &&
    identical(selective_wave$adjacency, uncached$adjacency) &&
    identical(normalize_sepsets(persistent$sepsets),
              normalize_sepsets(uncached$sepsets)) &&
    identical(normalize_sepsets(multilane$sepsets),
              normalize_sepsets(uncached$sepsets)) &&
    identical(normalize_sepsets(selective_wave$sepsets),
              normalize_sepsets(uncached$sepsets)) &&
    identical(as.integer(persistent$n.edgetests),
              as.integer(uncached$n.edgetests)) &&
    identical(as.integer(multilane$n.edgetests),
              as.integer(uncached$n.edgetests)) &&
    identical(as.integer(selective_wave$n.edgetests),
              as.integer(uncached$n.edgetests)),
  "persistent or multilane legacy eig context changed a strict result"
)

zero_authority_fields <- c(
  "cpu_residual_solve_count", "cpu_dcov_component_count",
  "cpu_dcov_eigen_or_lowrank_count", "cpu_dcov_pair_stat_count",
  "cpu_gamma_pvalue_count", "cpu_spectra_count", "residual_d2h_bytes",
  "component_d2h_bytes", "unknown_fallback_count",
  "approximate_backend_count"
)
assert_true(
  all(vapply(zero_authority_fields, function(field) {
    identical(as.numeric(cached$summary[[field]]), 0) &&
      identical(as.numeric(bypassed$summary[[field]]), 0) &&
      identical(as.numeric(persistent$summary[[field]]), 0) &&
      identical(as.numeric(multilane$summary[[field]]), 0) &&
      identical(as.numeric(selective_wave$summary[[field]]), 0)
  }, logical(1L))),
  "legacy component cache used forbidden CPU authority or numeric D2H"
)

uncached_summary <- uncached$summary
cached_summary <- cached$summary
bypassed_summary <- bypassed$summary
persistent_summary <- persistent$summary
multilane_summary <- multilane$summary
selective_wave_summary <- selective_wave$summary
uncached_cache_fields <- c(
  "legacy_component_cache_requested_budget_bytes",
  "legacy_component_cache_capacity_entries",
  "legacy_component_cache_device_bytes", "legacy_component_cache_entries",
  "legacy_component_cache_request_count", "legacy_component_cache_hit_count",
  "legacy_component_cache_miss_count",
  "legacy_component_cache_insert_count",
  "legacy_component_cache_bypass_count",
  "legacy_component_cache_route_mismatch_count",
  "legacy_component_cache_status_mismatch_count",
  "legacy_component_cache_store_d2d_bytes",
  "legacy_component_cache_create_host_ms",
  "legacy_component_cache_residual_bypass_requested",
  "legacy_component_cache_residual_bypass_batch_count",
  "legacy_component_cache_residual_bypass_target_count",
  "legacy_component_cache_batch_count",
  "legacy_component_cache_all_hit_batch_count",
  "legacy_component_cache_mixed_batch_count",
  "legacy_component_cache_all_miss_batch_count",
  "legacy_component_eig_lane_count_requested",
  "legacy_component_eig_lane_count",
  "legacy_component_eig_active_lane_peak",
  "legacy_component_eig_multilane_batch_count",
  "legacy_component_execution_context_reuse_count",
  "legacy_component_execution_context_device_bytes"
)
assert_true(
  all(vapply(uncached_cache_fields, function(field) {
    identical(as.numeric(uncached_summary[[field]]), 0)
  }, logical(1L))),
  "disabled legacy component cache changed the default execution ledger"
)

entry_bytes <- as.numeric((n * 35L + 35L + 2L) * 8L)
assert_true(
  cached_summary$legacy_component_cache_requested_budget_bytes ==
      128 * 1024^2 &&
    cached_summary$legacy_component_cache_capacity_entries ==
      floor((128 * 1024^2) / entry_bytes) &&
    cached_summary$legacy_component_cache_device_bytes ==
      cached_summary$legacy_component_cache_capacity_entries * entry_bytes &&
    cached_summary$legacy_component_cache_entries ==
      cached_summary$legacy_component_cache_insert_count &&
    cached_summary$legacy_component_cache_request_count ==
      cached_summary$legacy_component_cache_hit_count +
        cached_summary$legacy_component_cache_miss_count &&
    cached_summary$legacy_component_cache_miss_count ==
      cached_summary$legacy_component_cache_insert_count +
        cached_summary$legacy_component_cache_bypass_count &&
    cached_summary$legacy_component_cache_bypass_count == 0 &&
    cached_summary$legacy_component_cache_route_mismatch_count == 0 &&
    cached_summary$legacy_component_cache_status_mismatch_count == 0 &&
    cached_summary$legacy_component_cache_hit_count > 0 &&
    cached_summary$legacy_component_cache_request_count ==
      uncached_summary$cuda_dcov_component_count &&
    cached_summary$cuda_dcov_component_count ==
      cached_summary$legacy_component_cache_miss_count &&
    cached_summary$cuda_dcov_component_count <
      uncached_summary$cuda_dcov_component_count &&
    cached_summary$legacy_component_cache_store_d2d_bytes ==
      cached_summary$legacy_component_cache_insert_count * entry_bytes &&
    cached_summary$legacy_component_cache_batch_count ==
      cached_summary$legacy_component_cache_all_hit_batch_count +
        cached_summary$legacy_component_cache_mixed_batch_count +
        cached_summary$legacy_component_cache_all_miss_batch_count &&
    cached_summary$legacy_component_cache_all_hit_batch_count > 0 &&
    !isTRUE(cached_summary$legacy_component_cache_residual_bypass_requested) &&
    cached_summary$legacy_component_cache_residual_bypass_batch_count == 0 &&
    cached_summary$legacy_component_cache_residual_bypass_target_count == 0 &&
    cached_summary$legacy_component_cache_create_host_ms >= 0,
  "legacy component cache hit, capacity, or physical-work ledger is malformed"
)
assert_true(
  isTRUE(bypassed_summary$legacy_component_cache_residual_bypass_requested) &&
    bypassed_summary$legacy_component_cache_residual_bypass_batch_count ==
      bypassed_summary$legacy_component_cache_all_hit_batch_count &&
    bypassed_summary$legacy_component_cache_residual_bypass_batch_count > 0 &&
    bypassed_summary$legacy_component_cache_residual_bypass_target_count >=
      bypassed_summary$legacy_component_cache_residual_bypass_batch_count &&
    bypassed_summary$legacy_component_cache_route_mismatch_count == 0 &&
    bypassed_summary$legacy_component_cache_status_mismatch_count == 0 &&
    bypassed_summary$physical_residual_fits <
      cached_summary$physical_residual_fits &&
    bypassed_summary$cuda_residual_solve_host_ms <
      cached_summary$cuda_residual_solve_host_ms,
  "legacy component cache residual bypass accounting is malformed"
)
assert_true(
  persistent_summary$legacy_component_eig_lane_count_requested == 1 &&
    persistent_summary$legacy_component_eig_lane_count == 1 &&
    persistent_summary$legacy_component_eig_active_lane_peak == 1 &&
    persistent_summary$legacy_component_eig_multilane_batch_count == 0 &&
    persistent_summary$legacy_component_execution_context_reuse_count ==
      persistent_summary$legacy_component_cache_batch_count &&
    persistent_summary$legacy_component_execution_context_device_bytes > 0 &&
    multilane_summary$legacy_component_eig_lane_count_requested == 8 &&
    multilane_summary$legacy_component_eig_lane_count == 8 &&
    multilane_summary$legacy_component_eig_active_lane_peak > 1 &&
    multilane_summary$legacy_component_eig_active_lane_peak <= 8 &&
    multilane_summary$legacy_component_eig_multilane_batch_count > 0 &&
    multilane_summary$legacy_component_execution_context_reuse_count ==
      multilane_summary$legacy_component_cache_batch_count &&
    multilane_summary$legacy_component_execution_context_device_bytes >
      persistent_summary$legacy_component_execution_context_device_bytes,
  "persistent or multilane legacy eig context accounting is malformed"
)
selective_trace <- selective_wave$legacy_component_wave_group_trace
selective_pairs <- split(
  seq_len(nrow(selective_trace)),
  paste(selective_trace$level, selective_trace$wave_index, sep = "|")
)
selective_pairs <- selective_pairs[lengths(selective_pairs) == 2L]
assert_true(
  isTRUE(selective_wave_summary$legacy_component_wave_positive_miss_only_requested) &&
    selective_wave_summary$legacy_component_wave_capacity_requested == 2 &&
    selective_wave_summary$legacy_component_wave_prediction_count >=
      selective_wave_summary$legacy_component_wave_count +
        selective_wave_summary$legacy_component_wave_selective_pair_count &&
    selective_wave_summary$legacy_component_wave_prediction_incomplete_count ==
      0 &&
    selective_wave_summary$legacy_component_wave_selective_pair_count ==
      length(selective_pairs) &&
    all(vapply(selective_pairs, function(rows) {
      all(selective_trace$component_miss_count[rows] > 0L)
    }, logical(1L))) &&
    isTRUE(selective_wave_summary$authority_gate_pass),
  "positive-miss selective wave accounting is malformed"
)

cat(
  "PASS full CUDA CI legacy component cache; requests=",
  cached_summary$legacy_component_cache_request_count,
  "; hits=", cached_summary$legacy_component_cache_hit_count,
  "; misses=", cached_summary$legacy_component_cache_miss_count,
  "; eig_uncached=", uncached_summary$cuda_dcov_component_count,
  "; eig_cached=", cached_summary$cuda_dcov_component_count,
  "; uncached_sec=", format(uncached_run$elapsed, digits = 6L),
  "; cached_sec=", format(cached_run$elapsed, digits = 6L),
  "; bypassed_sec=", format(bypassed_run$elapsed, digits = 6L),
  "; persistent_sec=", format(persistent_run$elapsed, digits = 6L),
  "; multilane_sec=", format(multilane_run$elapsed, digits = 6L),
  "; selective_wave_sec=", format(selective_wave_run$elapsed, digits = 6L),
  "\n", sep = ""
)
