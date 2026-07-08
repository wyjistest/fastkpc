source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
`%||%` <- function(x, y) if (is.null(x)) y else x

missing <- c("graph", "mgcv", "pcalg", "Rcpp", "RcppArmadillo",
             "RcppEigen", "RSpectra")[
  !vapply(c("graph", "mgcv", "pcalg", "Rcpp", "RcppArmadillo",
            "RcppEigen", "RSpectra"),
          requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP precision compatible legacy dCov C++ backend: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

old_backend <- Sys.getenv("FASTKPC_LEGACY_DCOV_GAMMA_BACKEND",
                          unset = NA_character_)
old_lowrank <- Sys.getenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK",
                          unset = NA_character_)
old_shadow <- Sys.getenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_SHADOW",
                         unset = NA_character_)
old_batch <- Sys.getenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH",
                        unset = NA_character_)
old_cache <- Sys.getenv("FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE",
                        unset = NA_character_)
old_affinity <- Sys.getenv("FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY",
                           unset = NA_character_)
old_mgcv_backend <- Sys.getenv("FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND",
                               unset = NA_character_)
old_same_s_setup <- Sys.getenv("FASTKPC_LEGACY_MGCV_RESIDUAL_SAME_S_SETUP",
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
  restore_env("FASTKPC_LEGACY_DCOV_GAMMA_BACKEND", old_backend)
  restore_env("FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK", old_lowrank)
  restore_env("FASTKPC_LEGACY_DCOV_GAMMA_CPP_SHADOW", old_shadow)
  restore_env("FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH", old_batch)
  restore_env("FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE", old_cache)
  restore_env("FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY", old_affinity)
  restore_env("FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND", old_mgcv_backend)
  restore_env("FASTKPC_LEGACY_MGCV_RESIDUAL_SAME_S_SETUP", old_same_s_setup)
  restore_env("FASTKPC_LEGACY_PARALLEL_CORES", old_cores)
}, add = TRUE)

set.seed(18127)
n <- 56L
z <- stats::runif(n, -2, 2)
z2 <- stats::rnorm(n)
data <- cbind(
  x1 = sin(z) + stats::rnorm(n, sd = 0.12),
  x2 = cos(z) + stats::rnorm(n, sd = 0.12),
  x3 = z,
  x4 = z2 + 0.2 * z,
  x5 = stats::rnorm(n)
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

Sys.unsetenv("FASTKPC_LEGACY_DCOV_GAMMA_BACKEND")
Sys.unsetenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK")
Sys.unsetenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_SHADOW")
Sys.unsetenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH")
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE")
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY")
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND")
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_SAME_S_SETUP")
baseline <- run_compatible()
baseline_summary <- baseline$skeleton$scheduler_diagnostics$summary

assert_true(identical(
  as.integer(baseline_summary$legacy_dcov_cpp_backend_count %||% 0L), 0L),
  "default compatible route should not run legacy dCov C++ backend")
assert_true(identical(
  as.integer(baseline_summary$legacy_dcov_r_backend_count %||% 0L),
  as.integer(baseline_summary$legacy_dcov_gamma_count)),
  "default compatible route should count R legacy dCov backend calls")

Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_BACKEND = "cpp")
Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = "spectra")
Sys.unsetenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_SHADOW")
Sys.unsetenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH")
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE")
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY")
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND")
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_SAME_S_SETUP")
cpp <- run_compatible()
cpp_summary <- cpp$skeleton$scheduler_diagnostics$summary

assert_true(identical(cpp$skeleton$adjacency, baseline$skeleton$adjacency),
            "legacy dCov C++ backend should match R legacy skeleton adjacency")
assert_true(identical(cpp$skeleton$n.edgetests,
                      baseline$skeleton$n.edgetests),
            "legacy dCov C++ backend should match R legacy n.edgetests")
assert_true(identical(cpp_summary$legacy_dcov_backend, "cpp"),
            "legacy dCov backend summary should report cpp authority")
assert_true(identical(as.integer(cpp_summary$legacy_dcov_cpp_backend_count),
                      as.integer(cpp_summary$legacy_dcov_gamma_count)),
            "legacy dCov C++ backend count should match dCov call count")
assert_true(identical(as.integer(cpp_summary$legacy_dcov_r_backend_count), 0L),
            "legacy dCov C++ backend should not run R backend as authority")
assert_true(identical(as.integer(cpp_summary$legacy_dcov_cpp_backend_error_count), 0L),
            "legacy dCov C++ backend should not report errors")
assert_true(identical(as.integer(cpp_summary$legacy_dcov_cpp_backend_fallback_count), 0L),
            "legacy dCov C++ backend should not fallback to R")
assert_true(cpp_summary$legacy_dcov_cpp_backend_ms > 0,
            "legacy dCov C++ backend should report elapsed time")
assert_true(cpp_summary$legacy_dcov_cpp_lowrank_spectra_count > 0,
            "legacy dCov C++ backend should use Spectra selected eigs")
assert_true(identical(
  as.integer(cpp_summary$legacy_dcov_cpp_lowrank_spectra_failed_count), 0L),
  "legacy dCov C++ backend should not report Spectra failures")
assert_true(identical(
  as.integer(cpp_summary$legacy_dcov_cpp_lowrank_spectra_fallback_full_eig_count), 0L),
  "legacy dCov C++ backend should not fallback to full eig")
assert_true(identical(as.integer(cpp_summary$legacy_dcov_cpp_shadow_count), 0L),
            "legacy dCov C++ backend should not run shadow unless requested")

batch_potential_fields <- c(
  "legacy_dcov_cpp_batch_potential_call_count",
  "legacy_dcov_cpp_batch_potential_group_count",
  "legacy_dcov_cpp_batch_potential_max_group_size",
  "legacy_dcov_cpp_batch_potential_mean_group_size",
  "legacy_dcov_cpp_batch_potential_reuse_opportunity_count",
  "legacy_dcov_cpp_batch_potential_reuse_ratio",
  "legacy_dcov_cpp_round_batch_potential_call_count",
  "legacy_dcov_cpp_round_batch_potential_round_count",
  "legacy_dcov_cpp_round_batch_potential_max_round_size",
  "legacy_dcov_cpp_round_batch_potential_mean_round_size",
  "legacy_dcov_cpp_round_batch_potential_reuse_opportunity_count",
  "legacy_dcov_cpp_round_batch_potential_reuse_ratio"
)
missing_batch_potential <- setdiff(batch_potential_fields, names(cpp_summary))
assert_true(length(missing_batch_potential) == 0L,
            paste("legacy dCov C++ backend summary missing batch potential",
                  missing_batch_potential[[1L]]))
assert_true(identical(
  as.integer(cpp_summary$legacy_dcov_cpp_batch_potential_call_count),
  as.integer(cpp_summary$legacy_dcov_cpp_backend_count)),
  "dCov batch potential call count should match C++ backend calls")
assert_true(cpp_summary$legacy_dcov_cpp_batch_potential_group_count > 0L,
            "dCov batch potential should report at least one shape group")
assert_true(cpp_summary$legacy_dcov_cpp_batch_potential_group_count <=
              cpp_summary$legacy_dcov_cpp_batch_potential_call_count,
            "dCov batch potential groups should not exceed calls")
assert_true(cpp_summary$legacy_dcov_cpp_batch_potential_max_group_size >=
              cpp_summary$legacy_dcov_cpp_batch_potential_mean_group_size,
            "dCov batch potential max group size should dominate mean")
assert_true(cpp_summary$legacy_dcov_cpp_batch_potential_mean_group_size >= 1,
            "dCov batch potential mean group size should be positive")
assert_true(identical(
  as.integer(cpp_summary$legacy_dcov_cpp_batch_potential_reuse_opportunity_count),
  as.integer(cpp_summary$legacy_dcov_cpp_batch_potential_call_count -
               cpp_summary$legacy_dcov_cpp_batch_potential_group_count)),
  "dCov batch potential reuse opportunity should be calls minus groups")
assert_true(cpp_summary$legacy_dcov_cpp_batch_potential_reuse_ratio >= 0 &&
              cpp_summary$legacy_dcov_cpp_batch_potential_reuse_ratio < 1,
            "dCov batch potential reuse ratio should be in [0, 1)")
assert_true(identical(
  as.integer(cpp_summary$legacy_dcov_cpp_round_batch_potential_call_count),
  as.integer(cpp_summary$legacy_dcov_cpp_backend_count)),
  "dCov round-batch potential call count should match C++ backend calls")
assert_true(cpp_summary$legacy_dcov_cpp_round_batch_potential_round_count > 0L,
            "dCov round-batch potential should report rounds")
assert_true(cpp_summary$legacy_dcov_cpp_round_batch_potential_round_count <=
              cpp_summary$legacy_dcov_cpp_round_batch_potential_call_count,
            "dCov round-batch potential rounds should not exceed calls")
assert_true(cpp_summary$legacy_dcov_cpp_round_batch_potential_max_round_size >=
              cpp_summary$legacy_dcov_cpp_round_batch_potential_mean_round_size,
            "dCov round-batch potential max round size should dominate mean")
assert_true(cpp_summary$legacy_dcov_cpp_round_batch_potential_mean_round_size >=
              1,
            "dCov round-batch potential mean round size should be positive")
assert_true(identical(
  as.integer(cpp_summary$legacy_dcov_cpp_round_batch_potential_reuse_opportunity_count),
  as.integer(cpp_summary$legacy_dcov_cpp_round_batch_potential_call_count -
               cpp_summary$legacy_dcov_cpp_round_batch_potential_round_count)),
  "dCov round-batch potential reuse should be calls minus rounds")
assert_true(cpp_summary$legacy_dcov_cpp_round_batch_potential_reuse_ratio >= 0 &&
              cpp_summary$legacy_dcov_cpp_round_batch_potential_reuse_ratio < 1,
            "dCov round-batch potential reuse ratio should be in [0, 1)")

Sys.setenv(
  FASTKPC_LEGACY_DCOV_GAMMA_BACKEND = "cpp",
  FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = "spectra",
  FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE = "1",
  FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY = "s",
  FASTKPC_LEGACY_PARALLEL_CORES = "2"
)
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND")
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_SAME_S_SETUP")
Sys.unsetenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH")
chunk_baseline <- run_compatible()
Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH = "chunk")
chunk_batch <- run_compatible()
chunk_baseline_summary <-
  chunk_baseline$skeleton$scheduler_diagnostics$summary
chunk_batch_summary <- chunk_batch$skeleton$scheduler_diagnostics$summary

assert_true(identical(chunk_batch$skeleton$adjacency,
                      chunk_baseline$skeleton$adjacency),
            "chunk-batched dCov C++ backend should preserve adjacency")
assert_true(identical(chunk_batch$skeleton$n.edgetests,
                      chunk_baseline$skeleton$n.edgetests),
            "chunk-batched dCov C++ backend should preserve n.edgetests")
batch_backend_fields <- c(
  "legacy_dcov_cpp_batch_backend_enabled",
  "legacy_dcov_cpp_batch_backend_count",
  "legacy_dcov_cpp_batch_backend_pair_count",
  "legacy_dcov_cpp_batch_backend_ms",
  "legacy_dcov_cpp_batch_backend_error_count",
  "legacy_dcov_cpp_batch_backend_fallback_count",
  "legacy_dcov_cpp_batch_backend_max_batch_size",
  "legacy_dcov_cpp_batch_backend_mean_batch_size"
)
missing_batch_backend <- setdiff(batch_backend_fields,
                                 names(chunk_batch_summary))
assert_true(length(missing_batch_backend) == 0L,
            paste("legacy dCov C++ backend summary missing chunk batch",
                  missing_batch_backend[[1L]]))
assert_true(isTRUE(chunk_batch_summary$legacy_dcov_cpp_batch_backend_enabled),
            "chunk-batched dCov C++ backend should report enabled")
assert_true(chunk_batch_summary$legacy_dcov_cpp_batch_backend_count > 0L,
            "chunk-batched dCov C++ backend should report batch calls")
assert_true(chunk_batch_summary$legacy_dcov_cpp_batch_backend_pair_count > 0L,
            "chunk-batched dCov C++ backend should report batched pairs")
assert_true(chunk_batch_summary$legacy_dcov_cpp_batch_backend_pair_count <=
              chunk_batch_summary$legacy_dcov_cpp_backend_count,
            "chunk-batched pair count should be covered by backend count")
assert_true(chunk_batch_summary$legacy_dcov_cpp_batch_backend_max_batch_size >=
              chunk_batch_summary$legacy_dcov_cpp_batch_backend_mean_batch_size,
            "chunk-batched max batch size should dominate mean")
assert_true(identical(
  as.integer(chunk_batch_summary$legacy_dcov_cpp_batch_backend_error_count), 0L),
  "chunk-batched dCov C++ backend should not report batch errors")
assert_true(identical(
  as.integer(chunk_batch_summary$legacy_dcov_cpp_batch_backend_fallback_count), 0L),
  "chunk-batched dCov C++ backend should not fallback")
assert_true(identical(
  as.integer(chunk_batch_summary$legacy_dcov_cpp_backend_count),
  as.integer(chunk_batch_summary$legacy_dcov_gamma_count)),
  "chunk-batched C++ backend count should still match dCov call count")
assert_true(!isTRUE(
  chunk_batch_summary$legacy_mgcv_cpp_same_s_setup_provider_chunk_enabled),
  "chunk-batched dCov should not enable mgcv same-S setup chunk provider")
assert_true(identical(
  as.integer(
    chunk_batch_summary$legacy_mgcv_cpp_same_s_setup_provider_chunk_count %||%
      0L
  ),
  0L),
  "chunk-batched dCov should not run mgcv same-S setup chunk provider")
assert_true(identical(
  as.integer(chunk_baseline_summary$legacy_dcov_cpp_batch_backend_count %||% 0L),
  0L),
  "chunk batching should be env-gated")

cat("PASS precision compatible legacy dCov C++ backend\n")
