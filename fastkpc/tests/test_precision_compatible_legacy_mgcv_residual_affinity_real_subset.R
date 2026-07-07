source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_REAL_SUBSET_TESTS", unset = ""), "1")) {
  cat("SKIP precision compatible legacy mgcv residual affinity real subset: set FASTKPC_RUN_REAL_SUBSET_TESTS=1\n")
  quit(save = "no", status = 0)
}

missing <- c("graph", "mgcv", "pcalg", "RSpectra")[
  !vapply(c("graph", "mgcv", "pcalg", "RSpectra"), requireNamespace,
          logical(1), quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP precision compatible legacy mgcv residual affinity real subset: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

real_path <- "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"
if (!file.exists(real_path)) {
  cat("SKIP precision compatible legacy mgcv residual affinity real subset: real fixture unavailable\n")
  quit(save = "no", status = 0)
}

old_env <- Sys.getenv(c(
  "FASTKPC_LEGACY_PARALLEL_CORES",
  "FASTKPC_LEGACY_DCOV_GAMMA_BACKEND",
  "FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK",
  "FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE",
  "FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY"
), unset = NA_character_)
on.exit({
  for (name in names(old_env)) {
    if (is.na(old_env[[name]])) {
      Sys.unsetenv(name)
    } else {
      do.call(Sys.setenv, stats::setNames(list(old_env[[name]]), name))
    }
  }
}, add = TRUE)

data <- readRDS(real_path)
hot_cols <- c(1L, 2L, 3L, 4L, 5L, 6L, 9L, 12L,
              15L, 16L, 17L, 18L, 19L, 22L, 30L, 32L)
data <- data[, hot_cols, drop = FALSE]

run_subset <- function(affinity_mode = "") {
  Sys.setenv(
    FASTKPC_LEGACY_PARALLEL_CORES = "8",
    FASTKPC_LEGACY_DCOV_GAMMA_BACKEND = "cpp",
    FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = "spectra",
    FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE = "1"
  )
  if (nzchar(affinity_mode)) {
    Sys.setenv(FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY = affinity_mode)
  } else {
    Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY")
  }
  fast_kpc(
    data,
    alpha = 0.1,
    max_conditioning_size = ncol(data) - 2L,
    engine = "cuda",
    precision = "compatible",
    scheduler = "layer",
    graph_stage = "skeleton",
    ci_method = "dcc.gamma",
    precision_trace_level = "summary",
    benchmark = TRUE
  )
}

cached <- run_subset()
affinity <- run_subset("s")
owner <- run_subset("target_s")
summary <- affinity$skeleton$scheduler_diagnostics$summary
owner_summary <- owner$skeleton$scheduler_diagnostics$summary

assert_true(identical(affinity$skeleton$adjacency, cached$skeleton$adjacency),
            "real subset S-affinity should not change adjacency")
assert_true(identical(affinity$skeleton$n.edgetests,
                      cached$skeleton$n.edgetests),
            "real subset S-affinity should not change n.edgetests")
assert_true(identical(owner$skeleton$adjacency, cached$skeleton$adjacency),
            "real subset target|S owner scheduling should not change adjacency")
assert_true(identical(owner$skeleton$n.edgetests,
                      cached$skeleton$n.edgetests),
            "real subset target|S owner scheduling should not change n.edgetests")
assert_true(isTRUE(summary$legacy_mgcv_residual_affinity_enabled),
            "real subset S-affinity should report enabled")
assert_true(isTRUE(owner_summary$legacy_mgcv_residual_affinity_enabled),
            "real subset target|S owner scheduling should report affinity enabled")
assert_true(isTRUE(owner_summary$legacy_mgcv_residual_owner_enabled),
            "real subset target|S owner scheduling should report owner enabled")
assert_true(owner_summary$legacy_mgcv_residual_owner_key_count > 0L,
            "real subset target|S owner scheduling should report owner keys")
assert_true(owner_summary$legacy_mgcv_residual_owner_task_count > 0L,
            "real subset target|S owner scheduling should report owner tasks")
assert_true(owner_summary$legacy_mgcv_residual_owner_realized_hit_count ==
              owner_summary$legacy_mgcv_residual_cache_hit_count,
            "real subset target|S owner realized hits should match cache hits")
assert_true(owner_summary$legacy_mgcv_residual_owner_lost_duplicate_count ==
              owner_summary$legacy_mgcv_residual_cache_lost_duplicate_count,
            "real subset target|S owner lost duplicates should match cache lost duplicates")
assert_true(summary$legacy_mgcv_residual_request_count > 10000L,
            "real subset should exercise a nontrivial residual workload")
assert_true(owner_summary$legacy_mgcv_residual_request_count ==
              summary$legacy_mgcv_residual_request_count,
            "real subset target|S owner scheduling should preserve residual request count")
assert_true(owner_summary$legacy_mgcv_residual_cache_hit_count >=
              summary$legacy_mgcv_residual_cache_hit_count,
            "real subset target|S owner scheduling should not reduce cache hits")
assert_true(owner_summary$legacy_mgcv_fit_count <=
              summary$legacy_mgcv_fit_count,
            "real subset target|S owner scheduling should not increase mgcv fits")
assert_true(summary$legacy_mgcv_residual_cache_theoretical_hit_count >
              summary$legacy_mgcv_residual_cache_realized_hit_count,
            "real subset should expose remaining duplicate loss")
assert_true(owner_summary$legacy_mgcv_residual_cache_theoretical_hit_count >=
              owner_summary$legacy_mgcv_residual_cache_realized_hit_count,
            "real subset target|S owner theoretical hits should dominate realized hits")
assert_true(summary$legacy_mgcv_residual_cache_lost_duplicate_count > 0L,
            "real subset should report lost duplicate residuals")
assert_true(summary$legacy_mgcv_residual_cache_lost_split_s_group_count > 0L,
            "real subset should report split-S-group duplicate loss")
assert_true(summary$legacy_mgcv_residual_cache_lost_cross_level_count == 0L,
            "real subset target|S duplicates should not be cross-level")
assert_true(summary$legacy_mgcv_residual_affinity_split_group_count > 0L,
            "real subset should split at least one affinity group")
assert_true(summary$legacy_mgcv_fit_count +
              summary$legacy_mgcv_residual_cache_hit_count ==
              summary$legacy_mgcv_residual_request_count,
            "real subset fits plus cache hits should account for requests")
assert_true(owner_summary$legacy_mgcv_fit_count +
              owner_summary$legacy_mgcv_residual_cache_hit_count ==
              owner_summary$legacy_mgcv_residual_request_count,
            "real subset target|S owner fits plus cache hits should account for requests")

cat("PASS precision compatible legacy mgcv residual affinity real subset\n")
