source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
`%||%` <- function(x, y) if (is.null(x)) y else x

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP legacy dCov CUDA lowrank shadow real subset: FASTKPC_RUN_CUDA_TESTS != 1\n")
  quit(save = "no", status = 0)
}
if (!identical(Sys.getenv("FASTKPC_RUN_REAL_SUBSET_TESTS", unset = ""), "1")) {
  cat("SKIP legacy dCov CUDA lowrank shadow real subset: FASTKPC_RUN_REAL_SUBSET_TESTS != 1\n")
  quit(save = "no", status = 0)
}

missing <- c("graph", "mgcv", "pcalg", "Rcpp", "RcppArmadillo",
             "RcppEigen", "RSpectra")[
  !vapply(c("graph", "mgcv", "pcalg", "Rcpp", "RcppArmadillo",
            "RcppEigen", "RSpectra"),
          requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP legacy dCov CUDA lowrank shadow real subset: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

real_path <- "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"
if (!file.exists(real_path)) {
  cat("SKIP legacy dCov CUDA lowrank shadow real subset: real fixture unavailable\n")
  quit(save = "no", status = 0)
}

build_fastkpc_cuda_native(rebuild = TRUE)
if (!fastkpc_cuda_available()) {
  cat("SKIP legacy dCov CUDA lowrank shadow real subset: CUDA unavailable\n")
  quit(save = "no", status = 0)
}

tracked_env <- c(
  "FASTKPC_LEGACY_DCOV_GAMMA_BACKEND",
  "FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK",
  "FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH",
  "FASTKPC_LEGACY_DCOV_GAMMA_CUDA_LOW_RANK_SHADOW",
  "FASTKPC_LEGACY_DCOV_GAMMA_CUDA_LOW_RANK_SHADOW_MAX_CALLS",
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

real_data <- readRDS(real_path)
data <- as.matrix(real_data[, c(1L, 2L, 3L, 4L, 5L), drop = FALSE])
storage.mode(data) <- "double"

run_compatible <- function() {
  fast_kpc(
    data,
    alpha = 0.1,
    max_conditioning_size = 3L,
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
  FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE = "1",
  FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY = "s",
  FASTKPC_LEGACY_PARALLEL_CORES = "1"
)
Sys.unsetenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH")
Sys.unsetenv("FASTKPC_LEGACY_DCOV_GAMMA_CUDA_LOW_RANK_SHADOW")
Sys.unsetenv("FASTKPC_LEGACY_DCOV_GAMMA_CUDA_LOW_RANK_SHADOW_MAX_CALLS")
baseline <- run_compatible()
baseline_summary <- baseline$skeleton$scheduler_diagnostics$summary

assert_true(identical(
  as.integer(baseline_summary$legacy_dcov_cuda_lowrank_shadow_count %||% 0L),
  0L),
  "real subset default C++/Spectra route should not run CUDA lowrank shadow")

shadow_limit <- 5L
Sys.setenv(
  FASTKPC_LEGACY_DCOV_GAMMA_CUDA_LOW_RANK_SHADOW = "1",
  FASTKPC_LEGACY_DCOV_GAMMA_CUDA_LOW_RANK_SHADOW_MAX_CALLS =
    as.character(shadow_limit)
)
shadow <- run_compatible()
shadow_summary <- shadow$skeleton$scheduler_diagnostics$summary

assert_true(identical(shadow$skeleton$adjacency, baseline$skeleton$adjacency),
            "real subset CUDA lowrank shadow should not change skeleton adjacency")
assert_true(identical(shadow$skeleton$n.edgetests,
                      baseline$skeleton$n.edgetests),
            "real subset CUDA lowrank shadow should not change n.edgetests")
assert_true(identical(as.integer(shadow_summary$legacy_dcov_cpp_backend_count),
                      as.integer(shadow_summary$legacy_dcov_gamma_count)),
            "real subset C++ backend should remain dCov authority")
assert_true(identical(as.integer(shadow_summary$legacy_dcov_r_backend_count),
                      0L),
            "real subset CUDA lowrank shadow should not force R dCov authority")
assert_true(as.integer(shadow_summary$legacy_dcov_cpp_backend_count) >
              shadow_limit,
            "real subset should have more C++ dCov calls than the shadow cap")
assert_true(isTRUE(shadow_summary$legacy_dcov_cuda_lowrank_shadow_enabled),
            "real subset CUDA lowrank shadow summary should report enabled")
assert_true(identical(
  as.integer(shadow_summary$legacy_dcov_cuda_lowrank_shadow_max_calls),
  shadow_limit),
  "real subset CUDA lowrank shadow should report the configured cap")
assert_true(identical(
  as.integer(shadow_summary$legacy_dcov_cuda_lowrank_shadow_count),
  shadow_limit),
  "real subset CUDA lowrank shadow should stop at the configured cap")
assert_true(identical(
  as.integer(shadow_summary$legacy_dcov_cuda_lowrank_shadow_skipped_count),
  as.integer(shadow_summary$legacy_dcov_cpp_backend_count) - shadow_limit),
  "real subset CUDA lowrank shadow should report capped skipped calls")
assert_true(identical(
  as.integer(shadow_summary$legacy_dcov_cuda_lowrank_shadow_error_count), 0L),
  "real subset CUDA lowrank shadow should not report errors")
assert_true(identical(
  as.integer(shadow_summary$legacy_dcov_cuda_lowrank_shadow_converged_count),
  shadow_limit),
  "real subset CUDA lowrank shadow should converge for all shadowed calls")
assert_true(
  shadow_summary$legacy_dcov_cuda_lowrank_shadow_max_eigenvalue_diff < 1e-7,
  "real subset CUDA lowrank shadow eigenvalues should match CPU Spectra")
assert_true(
  shadow_summary$legacy_dcov_cuda_lowrank_shadow_min_centered_abs_corr >
    1 - 1e-7,
  "real subset CUDA lowrank shadow centered vectors should match CPU Spectra")
assert_true(
  shadow_summary$legacy_dcov_cuda_lowrank_shadow_max_statistic_input_abs_diff <
    1e-7,
  "real subset CUDA lowrank shadow statistic inputs should match CPU Spectra")
assert_true(identical(
  as.numeric(
    shadow_summary$legacy_dcov_cuda_lowrank_shadow_matrix_h2d_ms_during_compute_max),
  0),
  "real subset CUDA lowrank shadow should not re-upload matrices during Spectra compute")
assert_true(
  as.integer(shadow_summary$legacy_dcov_cuda_lowrank_shadow_spectra_matvec_count) >
    0L,
  "real subset CUDA lowrank shadow should report Spectra matvecs")

cat(sprintf(
  "PASS legacy dCov CUDA lowrank shadow real subset calls=%d shadow=%d skipped=%d\n",
  as.integer(shadow_summary$legacy_dcov_cpp_backend_count),
  as.integer(shadow_summary$legacy_dcov_cuda_lowrank_shadow_count),
  as.integer(shadow_summary$legacy_dcov_cuda_lowrank_shadow_skipped_count)
))
