source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
`%||%` <- function(x, y) if (is.null(x)) y else x

missing <- c("graph", "mgcv", "pcalg", "RSpectra")[
  !vapply(c("graph", "mgcv", "pcalg", "RSpectra"), requireNamespace,
          logical(1), quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP precision compatible legacy mgcv residual cache: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

old_cache <- Sys.getenv("FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE",
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
  restore_env("FASTKPC_LEGACY_PARALLEL_CORES", old_cores)
}, add = TRUE)

set.seed(37219)
n <- 62L
z1 <- stats::runif(n, -2, 2)
z2 <- stats::rnorm(n)
z3 <- stats::runif(n, -1, 1)
data <- cbind(
  x1 = sin(z1) + stats::rnorm(n, sd = 0.08),
  x2 = cos(z1) + stats::rnorm(n, sd = 0.08),
  x3 = z1 + stats::rnorm(n, sd = 0.05),
  x4 = z2 + 0.2 * z1,
  x5 = z3 + stats::rnorm(n, sd = 0.08),
  x6 = stats::rnorm(n)
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

Sys.setenv(FASTKPC_LEGACY_PARALLEL_CORES = "1")
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE")
baseline <- run_compatible()
baseline_summary <- baseline$skeleton$scheduler_diagnostics$summary

assert_true(identical(as.integer(
  baseline_summary$legacy_mgcv_residual_cache_hit_count %||% 0L), 0L),
  "default compatible route should not report mgcv residual cache hits")
assert_true(identical(
  as.integer(baseline_summary$legacy_mgcv_fit_count),
  as.integer(baseline_summary$legacy_mgcv_residual_request_count)),
  "default compatible route should fit every mgcv residual request")

Sys.setenv(FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE = "1")
cached <- run_compatible()
cached_summary <- cached$skeleton$scheduler_diagnostics$summary

assert_true(identical(cached$skeleton$adjacency, baseline$skeleton$adjacency),
            "mgcv residual cache should not change skeleton adjacency")
assert_true(identical(cached$skeleton$n.edgetests,
                      baseline$skeleton$n.edgetests),
            "mgcv residual cache should not change n.edgetests")
assert_true(identical(as.integer(cached_summary$legacy_mgcv_residual_request_count),
                      as.integer(baseline_summary$legacy_mgcv_residual_request_count)),
            "mgcv residual cache should not change residual request count")
assert_true(cached_summary$legacy_mgcv_residual_cache_hit_count > 0L,
            "mgcv residual cache should hit repeated target|S requests")
assert_true(cached_summary$legacy_mgcv_fit_avoided_count ==
              cached_summary$legacy_mgcv_residual_cache_hit_count,
            "mgcv residual fit avoided count should match cache hits")
assert_true(cached_summary$legacy_mgcv_residual_cache_insert_count ==
              cached_summary$legacy_mgcv_residual_cache_miss_count,
            "mgcv residual cache inserts should match misses")
assert_true(cached_summary$legacy_mgcv_residual_cache_entries ==
              cached_summary$legacy_mgcv_residual_cache_insert_count,
            "serial mgcv residual cache entries should match inserts")
assert_true(cached_summary$legacy_mgcv_fit_count +
              cached_summary$legacy_mgcv_residual_cache_hit_count ==
              cached_summary$legacy_mgcv_residual_request_count,
            "mgcv residual fits plus cache hits should account for requests")
assert_true(cached_summary$legacy_mgcv_fit_count <
              baseline_summary$legacy_mgcv_fit_count,
            "mgcv residual cache should reduce mgcv fit count")
assert_true(cached_summary$legacy_mgcv_residual_cache_lookup_ms >= 0,
            "mgcv residual cache should report lookup time")
assert_true(cached_summary$legacy_mgcv_residual_cache_store_ms >= 0,
            "mgcv residual cache should report store time")

cat("PASS precision compatible legacy mgcv residual cache\n")
