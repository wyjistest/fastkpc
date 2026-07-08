source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
`%||%` <- function(x, y) if (is.null(x)) y else x

missing <- c("graph", "mgcv", "pcalg", "RSpectra")[
  !vapply(c("graph", "mgcv", "pcalg", "RSpectra"), requireNamespace,
          logical(1), quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP precision compatible legacy mgcv residual level prefetch: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

old_cache <- Sys.getenv("FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE",
                        unset = NA_character_)
old_prefetch <- Sys.getenv("FASTKPC_LEGACY_MGCV_RESIDUAL_PREFETCH",
                           unset = NA_character_)
old_affinity <- Sys.getenv("FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY",
                           unset = NA_character_)
old_cores <- Sys.getenv("FASTKPC_LEGACY_PARALLEL_CORES",
                        unset = NA_character_)
restore_env <- function(name, value) {
  if (is.na(value)) {
    Sys.unsetenv(name)
  } else {
    do.call(Sys.setenv, stats::setNames(list(value), name))
  }
}
on.exit({
  restore_env("FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE", old_cache)
  restore_env("FASTKPC_LEGACY_MGCV_RESIDUAL_PREFETCH", old_prefetch)
  restore_env("FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY", old_affinity)
  restore_env("FASTKPC_LEGACY_PARALLEL_CORES", old_cores)
}, add = TRUE)

set.seed(73519)
n <- 58L
z1 <- stats::runif(n, -2, 2)
z2 <- stats::rnorm(n)
z3 <- stats::runif(n, -1, 1)
data <- cbind(
  x1 = sin(z1) + stats::rnorm(n, sd = 0.08),
  x2 = cos(z1) + stats::rnorm(n, sd = 0.08),
  x3 = z1 + stats::rnorm(n, sd = 0.04),
  x4 = z2 + 0.2 * z1,
  x5 = z3 + stats::rnorm(n, sd = 0.08),
  x6 = z1 * z3 + stats::rnorm(n, sd = 0.12),
  x7 = stats::rnorm(n)
)

run_compatible <- function() {
  fast_kpc(
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
}

Sys.setenv(FASTKPC_LEGACY_PARALLEL_CORES = "2")
Sys.setenv(FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE = "1")
Sys.setenv(FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY = "s")
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_PREFETCH")
baseline <- run_compatible()
baseline_summary <- baseline$skeleton$scheduler_diagnostics$summary

Sys.setenv(FASTKPC_LEGACY_MGCV_RESIDUAL_PREFETCH = "level")
prefetch <- run_compatible()
prefetch_summary <- prefetch$skeleton$scheduler_diagnostics$summary

required <- c(
  "legacy_mgcv_prefetch_enabled",
  "legacy_mgcv_prefetch_level_count",
  "legacy_mgcv_prefetch_key_count",
  "legacy_mgcv_prefetch_fit_count",
  "legacy_mgcv_prefetch_fit_ms",
  "legacy_mgcv_prefetch_collect_ms",
  "legacy_mgcv_prefetch_matrix_build_ms",
  "legacy_mgcv_prefetch_payload_bytes",
  "legacy_mgcv_prefetch_max_level_payload_bytes",
  "legacy_mgcv_prefetch_lookup_ms",
  "legacy_mgcv_prefetch_ci_phase_ms",
  "legacy_mgcv_prefetch_error_count"
)
missing_fields <- setdiff(required, names(prefetch_summary))
assert_true(length(missing_fields) == 0L,
            paste("legacy mgcv residual prefetch summary missing",
                  missing_fields[[1L]]))

assert_true(identical(prefetch$skeleton$adjacency,
                      baseline$skeleton$adjacency),
            "level mgcv residual prefetch should not change adjacency")
assert_true(identical(prefetch$skeleton$n.edgetests,
                      baseline$skeleton$n.edgetests),
            "level mgcv residual prefetch should not change n.edgetests")
assert_true(isTRUE(prefetch_summary$legacy_mgcv_prefetch_enabled),
            "level mgcv residual prefetch summary should report enabled")
assert_true(prefetch_summary$legacy_mgcv_prefetch_key_count > 0L,
            "level mgcv residual prefetch should compute residual keys")
assert_true(prefetch_summary$legacy_mgcv_prefetch_fit_count > 0L,
            "level mgcv residual prefetch should count residual fits")
assert_true(prefetch_summary$legacy_mgcv_prefetch_error_count == 0L,
            "level mgcv residual prefetch should not report errors")
assert_true(prefetch_summary$legacy_mgcv_prefetch_payload_bytes > 0,
            "level mgcv residual prefetch should report payload bytes")
assert_true(prefetch_summary$legacy_mgcv_prefetch_max_level_payload_bytes > 0,
            "level mgcv residual prefetch should report max level payload")
assert_true(prefetch_summary$legacy_mgcv_prefetch_lookup_ms >= 0,
            "level mgcv residual prefetch should report lookup time")
assert_true(prefetch_summary$legacy_mgcv_prefetch_ci_phase_ms >= 0,
            "level mgcv residual prefetch should report CI phase time")
assert_true(prefetch_summary$legacy_mgcv_residual_cache_hit_count ==
              prefetch_summary$legacy_mgcv_residual_request_count,
            "level mgcv residual prefetch should serve consumed requests from the level cache")
assert_true(prefetch_summary$legacy_mgcv_residual_cache_miss_count == 0L,
            "level mgcv residual prefetch should not miss consumed requests")
assert_true(prefetch_summary$legacy_mgcv_fit_avoided_count ==
              prefetch_summary$legacy_mgcv_residual_request_count,
            "level mgcv residual prefetch should avoid on-demand fits for consumed requests")
assert_true(prefetch_summary$legacy_mgcv_prefetch_fit_count ==
              prefetch_summary$legacy_mgcv_fit_count,
            "level mgcv residual prefetch fit count should match mgcv fit count")
assert_true(prefetch_summary$legacy_mgcv_prefetch_consumed_key_count ==
              prefetch_summary$legacy_mgcv_level_prefetch_unique_key_count,
            "level mgcv residual prefetch should consume the actual unique target|S keys")
assert_true(prefetch_summary$legacy_mgcv_prefetch_key_count >=
              prefetch_summary$legacy_mgcv_prefetch_consumed_key_count,
            "level mgcv residual prefetch key count should dominate consumed keys")
assert_true(prefetch_summary$legacy_mgcv_prefetch_unused_key_count ==
              prefetch_summary$legacy_mgcv_prefetch_key_count -
                prefetch_summary$legacy_mgcv_prefetch_consumed_key_count,
            "level mgcv residual prefetch should report unused over-prefetched keys")

by_level <- prefetch$skeleton$scheduler_diagnostics$
  legacy_mgcv_prefetch_by_level
assert_true(is.data.frame(by_level),
            "legacy mgcv prefetch by-level diagnostics should be a data frame")
required_by_level <- c(
  "level",
  "task_count",
  "residual_request_count",
  "unique_key_count",
  "fit_count",
  "payload_bytes",
  "prefetch_fit_ms",
  "prefetch_collect_ms",
  "matrix_build_ms",
  "ci_phase_ms",
  "elapsed_ms"
)
missing_by_level <- setdiff(required_by_level, names(by_level))
assert_true(length(missing_by_level) == 0L,
            paste("legacy mgcv prefetch by-level missing",
                  missing_by_level[[1L]]))
assert_true(sum(by_level$fit_count) ==
              prefetch_summary$legacy_mgcv_prefetch_fit_count,
            "legacy mgcv prefetch by-level fits should sum to summary")
assert_true(sum(by_level$unique_key_count) ==
              prefetch_summary$legacy_mgcv_prefetch_key_count,
            "legacy mgcv prefetch by-level keys should sum to summary")
assert_true(sum(by_level$payload_bytes) ==
              prefetch_summary$legacy_mgcv_prefetch_payload_bytes,
            "legacy mgcv prefetch by-level payload should sum to summary")

cat("PASS precision compatible legacy mgcv residual level prefetch\n")
