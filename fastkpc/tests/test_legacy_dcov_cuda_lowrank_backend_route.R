source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
`%||%` <- function(x, y) if (is.null(x)) y else x

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP legacy dCov CUDA lowrank backend route: FASTKPC_RUN_CUDA_TESTS != 1\n")
  quit(save = "no", status = 0)
}

missing <- c("graph", "mgcv", "pcalg", "Rcpp", "RcppArmadillo",
             "RcppEigen", "RSpectra")[
  !vapply(c("graph", "mgcv", "pcalg", "Rcpp", "RcppArmadillo",
            "RcppEigen", "RSpectra"),
          requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP legacy dCov CUDA lowrank backend route: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

build_fastkpc_cuda_native(rebuild = FALSE)
if (!fastkpc_cuda_available()) {
  cat("SKIP legacy dCov CUDA lowrank backend route: CUDA unavailable\n")
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

set.seed(24680)
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
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE")
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY")
baseline <- run_compatible()
baseline_summary <- baseline$skeleton$scheduler_diagnostics$summary

assert_true(identical(
  as.integer(baseline_summary$legacy_dcov_cuda_lowrank_backend_count %||% 0L),
  0L),
  "CPU Spectra route should not run CUDA lowrank backend")

Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = "cuda_spectra")
candidate <- run_compatible()
candidate_summary <- candidate$skeleton$scheduler_diagnostics$summary

assert_true(identical(candidate$skeleton$adjacency,
                      baseline$skeleton$adjacency),
            "CUDA lowrank backend should match CPU Spectra skeleton adjacency")
assert_true(identical(candidate$skeleton$n.edgetests,
                      baseline$skeleton$n.edgetests),
            "CUDA lowrank backend should match CPU Spectra n.edgetests")
assert_true(identical(candidate_summary$legacy_dcov_backend, "cpp"),
            "CUDA lowrank backend should remain under cpp dCov authority")
assert_true(isTRUE(candidate_summary$legacy_dcov_cuda_lowrank_backend_enabled),
            "CUDA lowrank backend summary should report enabled")
assert_true(identical(
  as.integer(candidate_summary$legacy_dcov_cpp_backend_count),
  as.integer(candidate_summary$legacy_dcov_gamma_count)),
  "cpp backend count should still match dCov calls with CUDA lowrank authority")
assert_true(identical(
  as.integer(candidate_summary$legacy_dcov_cuda_lowrank_backend_count),
  as.integer(candidate_summary$legacy_dcov_gamma_count)),
  "CUDA lowrank backend count should match dCov calls")
assert_true(identical(
  as.integer(candidate_summary$legacy_dcov_r_backend_count), 0L),
  "CUDA lowrank backend should not use R dCov authority on success")
assert_true(identical(
  as.integer(candidate_summary$legacy_dcov_cuda_lowrank_backend_error_count), 0L),
  "CUDA lowrank backend should not report errors")
assert_true(identical(
  as.integer(candidate_summary$legacy_dcov_cuda_lowrank_backend_fallback_count), 0L),
  "CUDA lowrank backend should not fallback for the small route fixture")
assert_true(identical(
  as.integer(candidate_summary$legacy_dcov_cuda_lowrank_backend_converged_count),
  as.integer(candidate_summary$legacy_dcov_cuda_lowrank_backend_count)),
  "CUDA lowrank backend should converge for all authority calls")
assert_true(
  as.integer(
    candidate_summary$legacy_dcov_cuda_lowrank_backend_spectra_matvec_count
  ) > 0L,
  "CUDA lowrank backend should report Spectra matvecs")
assert_true(identical(
  as.numeric(
    candidate_summary$legacy_dcov_cuda_lowrank_backend_matrix_h2d_ms_during_compute_max),
  0),
  "CUDA lowrank backend should not re-upload matrices during Spectra compute")
assert_true(candidate_summary$legacy_dcov_cuda_lowrank_backend_ms > 0,
            "CUDA lowrank backend should report elapsed time")

cat("PASS legacy dCov CUDA lowrank backend route\n")
