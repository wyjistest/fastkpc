source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
`%||%` <- function(x, y) if (is.null(x)) y else x

missing <- c("graph", "mgcv", "pcalg", "RSpectra")[
  !vapply(c("graph", "mgcv", "pcalg", "RSpectra"), requireNamespace,
          logical(1), quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP precision compatible legacy mgcv residual affinity: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

old_cache <- Sys.getenv("FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE",
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
  restore_env("FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY", old_affinity)
  restore_env("FASTKPC_LEGACY_PARALLEL_CORES", old_cores)
}, add = TRUE)

set.seed(49211)
n <- 64L
z1 <- stats::runif(n, -2, 2)
z2 <- stats::rnorm(n)
z3 <- stats::runif(n, -1, 1)
data <- cbind(
  x1 = sin(z1) + stats::rnorm(n, sd = 0.08),
  x2 = cos(z1) + stats::rnorm(n, sd = 0.08),
  x3 = z1 + stats::rnorm(n, sd = 0.04),
  x4 = z2 + 0.25 * z1,
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
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY")
cached <- run_compatible()
cached_summary <- cached$skeleton$scheduler_diagnostics$summary

Sys.setenv(FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY = "s")
affinity <- run_compatible()
affinity_summary <- affinity$skeleton$scheduler_diagnostics$summary

required <- c(
  "legacy_mgcv_residual_affinity_enabled",
  "legacy_mgcv_residual_affinity_group_count",
  "legacy_mgcv_residual_affinity_task_count",
  "legacy_mgcv_residual_affinity_worker_count",
  "legacy_mgcv_residual_affinity_max_group_size",
  "legacy_mgcv_residual_affinity_mean_group_size",
  "legacy_mgcv_residual_affinity_load_imbalance",
  "legacy_mgcv_residual_affinity_split_group_count",
  "legacy_mgcv_residual_affinity_split_group_tasks",
  "legacy_mgcv_residual_cache_theoretical_hit_count",
  "legacy_mgcv_residual_cache_realized_hit_count",
  "legacy_mgcv_residual_cache_lost_duplicate_count",
  "legacy_mgcv_residual_cache_lost_cross_worker_count",
  "legacy_mgcv_residual_cache_lost_split_s_group_count",
  "legacy_mgcv_residual_cache_lost_cross_level_count",
  "legacy_mgcv_residual_cache_cross_worker_loss_estimate"
)
missing_fields <- setdiff(required, names(affinity_summary))
assert_true(length(missing_fields) == 0L,
            paste("legacy mgcv residual affinity summary missing",
                  missing_fields[[1L]]))

assert_true(identical(affinity$skeleton$adjacency, cached$skeleton$adjacency),
            "mgcv residual S-affinity should not change skeleton adjacency")
assert_true(identical(affinity$skeleton$n.edgetests,
                      cached$skeleton$n.edgetests),
            "mgcv residual S-affinity should not change n.edgetests")
assert_true(isTRUE(affinity_summary$legacy_mgcv_residual_affinity_enabled),
            "mgcv residual S-affinity summary should report enabled")
assert_true(affinity_summary$legacy_mgcv_residual_affinity_group_count > 0L,
            "mgcv residual S-affinity should report S groups")
assert_true(affinity_summary$legacy_mgcv_residual_affinity_task_count > 0L,
            "mgcv residual S-affinity should report scheduled tasks")
assert_true(affinity_summary$legacy_mgcv_residual_affinity_worker_count == 2L,
            "mgcv residual S-affinity should report worker count")
assert_true(affinity_summary$legacy_mgcv_residual_affinity_max_group_size >=
              affinity_summary$legacy_mgcv_residual_affinity_mean_group_size,
            "mgcv residual S-affinity max group size should dominate mean")
assert_true(affinity_summary$legacy_mgcv_residual_cache_hit_count > 0L,
            "mgcv residual S-affinity should retain cache hits")
assert_true(affinity_summary$legacy_mgcv_residual_cache_realized_hit_count ==
              affinity_summary$legacy_mgcv_residual_cache_hit_count,
            "mgcv residual S-affinity realized hits should match cache hits")
assert_true(affinity_summary$legacy_mgcv_residual_cache_theoretical_hit_count >=
              affinity_summary$legacy_mgcv_residual_cache_realized_hit_count,
            "mgcv residual S-affinity theoretical hits should dominate realized hits")
assert_true(affinity_summary$legacy_mgcv_residual_cache_cross_worker_loss_estimate >= 0L,
            "mgcv residual S-affinity should report nonnegative cross-worker loss")
assert_true(affinity_summary$legacy_mgcv_residual_cache_lost_duplicate_count ==
              affinity_summary$legacy_mgcv_residual_cache_theoretical_hit_count -
                affinity_summary$legacy_mgcv_residual_cache_realized_hit_count,
            "mgcv residual S-affinity lost duplicates should equal theoretical minus realized hits")
assert_true(affinity_summary$legacy_mgcv_residual_cache_lost_cross_worker_count >= 0L,
            "mgcv residual S-affinity should report nonnegative cross-worker lost duplicates")
assert_true(affinity_summary$legacy_mgcv_residual_cache_lost_split_s_group_count >= 0L,
            "mgcv residual S-affinity should report nonnegative split-group lost duplicates")
assert_true(affinity_summary$legacy_mgcv_residual_cache_lost_cross_level_count == 0L,
            "mgcv residual target|S keys should not have cross-level duplicate loss")
assert_true(affinity_summary$legacy_mgcv_residual_affinity_split_group_count >= 0L,
            "mgcv residual S-affinity should report split group count")
assert_true(affinity_summary$legacy_mgcv_residual_affinity_split_group_tasks >=
              affinity_summary$legacy_mgcv_residual_affinity_split_group_count,
            "mgcv residual S-affinity split group tasks should dominate split groups")
assert_true(affinity_summary$legacy_mgcv_fit_count +
              affinity_summary$legacy_mgcv_residual_cache_hit_count ==
              affinity_summary$legacy_mgcv_residual_request_count,
            "mgcv residual S-affinity fits plus hits should account for requests")

cat("PASS precision compatible legacy mgcv residual affinity\n")
