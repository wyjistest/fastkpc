source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

missing <- c("graph", "mgcv", "pcalg", "RSpectra")[
  !vapply(c("graph", "mgcv", "pcalg", "RSpectra"), requireNamespace,
          logical(1), quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP precision compatible legacy mgcv residual breakdown: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

set.seed(21037)
n <- 58L
z1 <- stats::runif(n, -2, 2)
z2 <- stats::rnorm(n)
data <- cbind(
  x1 = sin(z1) + stats::rnorm(n, sd = 0.1),
  x2 = cos(z1) + stats::rnorm(n, sd = 0.1),
  x3 = z1,
  x4 = z2 + 0.25 * z1,
  x5 = stats::rnorm(n)
)

result <- fast_kpc(
  data,
  alpha = 0.08,
  max_conditioning_size = 3,
  engine = "cuda",
  precision = "compatible",
  graph_stage = "skeleton",
  ci_method = "dcc.gamma",
  precision_trace_level = "summary",
  benchmark = TRUE
)

summary <- result$skeleton$scheduler_diagnostics$summary
required <- c(
  "legacy_mgcv_residual_request_count",
  "legacy_mgcv_cache_hit_count",
  "legacy_mgcv_cache_miss_count",
  "legacy_mgcv_unique_residual_key_count",
  "legacy_mgcv_duplicate_residual_key_count",
  "legacy_mgcv_unique_target_s_count",
  "legacy_mgcv_unique_s_count",
  "legacy_mgcv_same_s_group_count",
  "legacy_mgcv_same_s_total_targets",
  "legacy_mgcv_same_s_max_targets",
  "legacy_mgcv_same_s_mean_targets",
  "legacy_mgcv_same_s_reuse_opportunity_count",
  "legacy_mgcv_level_prefetch_request_count",
  "legacy_mgcv_level_prefetch_unique_key_count",
  "legacy_mgcv_level_prefetch_theoretical_hit_count",
  "legacy_mgcv_level_prefetch_current_hit_count",
  "legacy_mgcv_level_prefetch_fit_reduction_potential",
  "legacy_mgcv_level_prefetch_payload_bytes",
  "legacy_mgcv_level_prefetch_max_level_payload_bytes",
  "legacy_mgcv_level_prefetch_max_level_unique_keys",
  "legacy_mgcv_key_build_ms",
  "legacy_mgcv_cache_lookup_ms",
  "legacy_mgcv_formula_build_ms",
  "legacy_mgcv_data_subset_ms",
  "legacy_mgcv_fit_call_ms",
  "legacy_mgcv_residual_extract_ms",
  "legacy_mgcv_result_store_ms",
  "legacy_mgcv_unaccounted_ms",
  "legacy_mgcv_s_size_0_count",
  "legacy_mgcv_s_size_1_count",
  "legacy_mgcv_s_size_2_count",
  "legacy_mgcv_s_size_gt2_count"
)
missing_fields <- setdiff(required, names(summary))
assert_true(length(missing_fields) == 0L,
            paste("legacy mgcv residual summary missing", missing_fields[[1L]]))

assert_true(summary$legacy_mgcv_residual_request_count > 0L,
            "legacy mgcv residual request count should be recorded")
assert_true(identical(as.integer(summary$legacy_mgcv_residual_request_count),
                      as.integer(summary$legacy_mgcv_fit_count)),
            "legacy mgcv residual requests should match legacy mgcv fit count")
assert_true(identical(as.integer(summary$legacy_mgcv_cache_hit_count), 0L),
            "diagnostic-only route should not report mgcv residual cache hits")
assert_true(identical(as.integer(summary$legacy_mgcv_cache_miss_count),
                      as.integer(summary$legacy_mgcv_residual_request_count)),
            "diagnostic-only route should report all mgcv residual requests as misses")
assert_true(summary$legacy_mgcv_unique_residual_key_count > 0L,
            "legacy mgcv unique residual key count should be recorded")
assert_true(summary$legacy_mgcv_unique_residual_key_count <=
              summary$legacy_mgcv_residual_request_count,
            "legacy mgcv unique residual keys should not exceed requests")
assert_true(summary$legacy_mgcv_duplicate_residual_key_count ==
              summary$legacy_mgcv_residual_request_count -
                summary$legacy_mgcv_unique_residual_key_count,
            "legacy mgcv duplicate residual keys should be request minus unique")
assert_true(summary$legacy_mgcv_unique_target_s_count ==
              summary$legacy_mgcv_unique_residual_key_count,
            "legacy mgcv target|S count should match residual key count")
assert_true(summary$legacy_mgcv_unique_s_count > 0L,
            "legacy mgcv unique S count should be recorded")
assert_true(summary$legacy_mgcv_same_s_group_count ==
              summary$legacy_mgcv_unique_s_count,
            "legacy mgcv same-S groups should match unique S count")
assert_true(summary$legacy_mgcv_same_s_total_targets ==
              summary$legacy_mgcv_residual_request_count,
            "legacy mgcv same-S total targets should match residual requests")
assert_true(summary$legacy_mgcv_same_s_max_targets >=
              summary$legacy_mgcv_same_s_mean_targets,
            "legacy mgcv same-S max targets should dominate mean targets")
assert_true(summary$legacy_mgcv_level_prefetch_request_count ==
              summary$legacy_mgcv_residual_request_count,
            "legacy mgcv level prefetch requests should match residual requests")
assert_true(summary$legacy_mgcv_level_prefetch_unique_key_count ==
              summary$legacy_mgcv_unique_target_s_count,
            "legacy mgcv level prefetch unique keys should match unique target|S")
assert_true(summary$legacy_mgcv_level_prefetch_theoretical_hit_count ==
              summary$legacy_mgcv_level_prefetch_request_count -
                summary$legacy_mgcv_level_prefetch_unique_key_count,
            "legacy mgcv level prefetch theoretical hits should be requests minus unique keys")
assert_true(summary$legacy_mgcv_level_prefetch_current_hit_count ==
              summary$legacy_mgcv_cache_hit_count,
            "legacy mgcv level prefetch current hits should match cache hits")
assert_true(summary$legacy_mgcv_level_prefetch_fit_reduction_potential ==
              summary$legacy_mgcv_fit_count -
                summary$legacy_mgcv_level_prefetch_unique_key_count,
            "legacy mgcv level prefetch fit reduction should compare current fits to unique keys")
assert_true(summary$legacy_mgcv_level_prefetch_payload_bytes ==
              summary$legacy_mgcv_level_prefetch_unique_key_count * nrow(data) * 8,
            "legacy mgcv level prefetch payload bytes should count double residual payload")
assert_true(summary$legacy_mgcv_level_prefetch_max_level_payload_bytes >= 0,
            "legacy mgcv level prefetch max payload should be recorded")
assert_true(summary$legacy_mgcv_level_prefetch_max_level_unique_keys >= 0L,
            "legacy mgcv level prefetch max level unique keys should be recorded")
assert_true(summary$legacy_mgcv_fit_call_ms > 0,
            "legacy mgcv fit call time should be recorded")
assert_true(summary$legacy_mgcv_data_subset_ms >= 0,
            "legacy mgcv data subset time should be recorded")
assert_true(summary$legacy_mgcv_fit_call_ms <=
              summary$legacy_residual_total_ms + 1e-6,
            "legacy mgcv fit call time should fit within residual total time")
assert_true(summary$legacy_mgcv_s_size_1_count +
              summary$legacy_mgcv_s_size_2_count +
              summary$legacy_mgcv_s_size_gt2_count ==
              summary$legacy_mgcv_residual_request_count,
            "legacy mgcv residual size buckets should account for requests")

by_level <- result$skeleton$scheduler_diagnostics$legacy_runtime_by_level
required_by_level <- c(
  "mgcv_residual_request_count",
  "mgcv_unique_residual_key_count",
  "mgcv_unique_s_count",
  "mgcv_fit_call_ms",
  "mgcv_data_subset_ms",
  "mgcv_unaccounted_ms"
)
missing_by_level <- setdiff(required_by_level, names(by_level))
assert_true(length(missing_by_level) == 0L,
            paste("legacy mgcv by-level runtime missing",
                  missing_by_level[[1L]]))
assert_true(sum(by_level$mgcv_residual_request_count) ==
              summary$legacy_mgcv_residual_request_count,
            "legacy mgcv by-level requests should sum to summary requests")
assert_true(any(by_level$mgcv_unique_s_count > 0L),
            "legacy mgcv by-level unique S counts should be recorded")

prefetch_by_level <- result$skeleton$scheduler_diagnostics$
  legacy_mgcv_prefetch_potential_by_level
assert_true(is.data.frame(prefetch_by_level),
            "legacy mgcv prefetch potential by level should be a data frame")
required_prefetch_by_level <- c(
  "level",
  "request_count",
  "current_fit_count",
  "current_hit_count",
  "unique_target_s_count",
  "theoretical_fit_count",
  "theoretical_hit_count",
  "lost_duplicate_count",
  "fit_reduction_potential",
  "residual_payload_bytes",
  "unique_s_count",
  "task_count"
)
missing_prefetch_by_level <- setdiff(required_prefetch_by_level,
                                     names(prefetch_by_level))
assert_true(length(missing_prefetch_by_level) == 0L,
            paste("legacy mgcv prefetch potential by-level missing",
                  missing_prefetch_by_level[[1L]]))
assert_true(nrow(prefetch_by_level) == nrow(by_level),
            "legacy mgcv prefetch potential should align with runtime levels")
assert_true(sum(prefetch_by_level$request_count) ==
              summary$legacy_mgcv_level_prefetch_request_count,
            "legacy mgcv prefetch by-level requests should sum to summary")
assert_true(sum(prefetch_by_level$unique_target_s_count) ==
              summary$legacy_mgcv_level_prefetch_unique_key_count,
            "legacy mgcv prefetch by-level unique keys should sum to summary")
assert_true(all(prefetch_by_level$theoretical_fit_count ==
                  prefetch_by_level$unique_target_s_count),
            "legacy mgcv prefetch theoretical fits should equal unique target|S")
assert_true(all(prefetch_by_level$theoretical_hit_count ==
                  prefetch_by_level$request_count -
                    prefetch_by_level$unique_target_s_count),
            "legacy mgcv prefetch theoretical hits should be requests minus unique target|S")
assert_true(all(prefetch_by_level$fit_reduction_potential ==
                  pmax(0L, prefetch_by_level$current_fit_count -
                         prefetch_by_level$unique_target_s_count)),
            "legacy mgcv prefetch fit reduction should be current fits minus unique target|S")
assert_true(all(prefetch_by_level$residual_payload_bytes ==
                  prefetch_by_level$unique_target_s_count * nrow(data) * 8),
            "legacy mgcv prefetch by-level payload should count double residual payload")

cat("PASS precision compatible legacy mgcv residual breakdown\n")
