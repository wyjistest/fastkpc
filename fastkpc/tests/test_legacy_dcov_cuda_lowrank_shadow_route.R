source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
`%||%` <- function(x, y) if (is.null(x)) y else x

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP legacy dCov CUDA lowrank shadow route: FASTKPC_RUN_CUDA_TESTS != 1\n")
  quit(save = "no", status = 0)
}

missing <- c("graph", "mgcv", "pcalg", "Rcpp", "RcppArmadillo",
             "RcppEigen", "RSpectra")[
  !vapply(c("graph", "mgcv", "pcalg", "Rcpp", "RcppArmadillo",
            "RcppEigen", "RSpectra"),
          requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP legacy dCov CUDA lowrank shadow route: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

build_fastkpc_cuda_native(rebuild = TRUE)
if (!fastkpc_cuda_available()) {
  cat("SKIP legacy dCov CUDA lowrank shadow route: CUDA unavailable\n")
  quit(save = "no", status = 0)
}

tracked_env <- c(
  "FASTKPC_LEGACY_DCOV_GAMMA_BACKEND",
  "FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK",
  "FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH",
  "FASTKPC_LEGACY_DCOV_GAMMA_CUDA_LOW_RANK_SHADOW",
  "FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE",
  "FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY",
  "FASTKPC_LEGACY_PARALLEL_CORES"
)
old_env <- stats::setNames(
  lapply(tracked_env, Sys.getenv, unset = NA_character_),
  tracked_env
)
restore_env <- function(name, value) {
  if (is.na(value)) {
    Sys.unsetenv(name)
  } else {
    do.call(Sys.setenv, stats::setNames(list(value), name))
  }
}
on.exit({
  for (name in tracked_env) restore_env(name, old_env[[name]])
}, add = TRUE)

set.seed(24601)
n <- 54L
z <- stats::runif(n, -2, 2)
z2 <- stats::rnorm(n)
data <- cbind(
  x1 = sin(z) + stats::rnorm(n, sd = 0.12),
  x2 = cos(z) + stats::rnorm(n, sd = 0.12),
  x3 = z,
  x4 = z2 + 0.15 * z,
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

Sys.setenv(
  FASTKPC_LEGACY_DCOV_GAMMA_BACKEND = "cpp",
  FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = "spectra",
  FASTKPC_LEGACY_PARALLEL_CORES = "1"
)
Sys.unsetenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH")
Sys.unsetenv("FASTKPC_LEGACY_DCOV_GAMMA_CUDA_LOW_RANK_SHADOW")
baseline <- run_compatible()
baseline_summary <- baseline$skeleton$scheduler_diagnostics$summary

assert_true(identical(
  as.integer(baseline_summary$legacy_dcov_cuda_lowrank_shadow_count %||% 0L),
  0L),
  "default C++/Spectra dCov route should not run CUDA lowrank shadow")

Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_CUDA_LOW_RANK_SHADOW = "1")
shadow <- run_compatible()
shadow_summary <- shadow$skeleton$scheduler_diagnostics$summary

assert_true(identical(shadow$skeleton$adjacency, baseline$skeleton$adjacency),
            "CUDA lowrank shadow should not change skeleton adjacency")
assert_true(identical(shadow$skeleton$n.edgetests,
                      baseline$skeleton$n.edgetests),
            "CUDA lowrank shadow should not change n.edgetests")
assert_true(identical(as.integer(shadow_summary$legacy_dcov_cpp_backend_count),
                      as.integer(shadow_summary$legacy_dcov_gamma_count)),
            "C++ backend should remain dCov authority with CUDA shadow enabled")
assert_true(identical(as.integer(shadow_summary$legacy_dcov_r_backend_count),
                      0L),
            "CUDA lowrank shadow should not force R dCov authority")
assert_true(isTRUE(shadow_summary$legacy_dcov_cuda_lowrank_shadow_enabled),
            "CUDA lowrank shadow summary should report enabled")
assert_true(identical(
  as.integer(shadow_summary$legacy_dcov_cuda_lowrank_shadow_count),
  as.integer(shadow_summary$legacy_dcov_cpp_backend_count)),
  "CUDA lowrank shadow should run for each scalar C++ dCov backend call")
assert_true(identical(
  as.integer(shadow_summary$legacy_dcov_cuda_lowrank_shadow_error_count), 0L),
  "CUDA lowrank shadow should not report errors")
assert_true(identical(
  as.integer(shadow_summary$legacy_dcov_cuda_lowrank_shadow_converged_count),
  as.integer(shadow_summary$legacy_dcov_cuda_lowrank_shadow_count)),
  "CUDA lowrank shadow should converge for all shadowed calls")
assert_true(
  shadow_summary$legacy_dcov_cuda_lowrank_shadow_max_eigenvalue_diff < 1e-7,
  "CUDA lowrank shadow eigenvalues should match CPU Spectra")
assert_true(
  shadow_summary$legacy_dcov_cuda_lowrank_shadow_min_centered_abs_corr >
    1 - 1e-7,
  "CUDA lowrank shadow centered vectors should match CPU Spectra")
assert_true(
  shadow_summary$legacy_dcov_cuda_lowrank_shadow_max_statistic_input_abs_diff <
    1e-7,
  "CUDA lowrank shadow statistic inputs should match CPU Spectra")
assert_true(identical(
  as.numeric(
    shadow_summary$legacy_dcov_cuda_lowrank_shadow_matrix_h2d_ms_during_compute_max),
  0),
  "CUDA lowrank shadow should not re-upload matrices during Spectra compute")
assert_true(
  as.integer(shadow_summary$legacy_dcov_cuda_lowrank_shadow_spectra_matvec_count) >
    0L,
  "CUDA lowrank shadow should report Spectra matvecs")
assert_true(shadow_summary$legacy_dcov_cuda_lowrank_shadow_ms > 0,
            "CUDA lowrank shadow should report elapsed time")

cat("PASS legacy dCov CUDA lowrank shadow route\n")
