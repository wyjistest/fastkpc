fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
`%||%` <- function(x, y) if (is.null(x)) y else x

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP legacy dCov CUDA lowrank backend real subset artifact: FASTKPC_RUN_CUDA_TESTS != 1\n")
  quit(save = "no", status = 0)
}
if (!identical(Sys.getenv("FASTKPC_RUN_REAL_SUBSET_TESTS", unset = ""), "1")) {
  cat("SKIP legacy dCov CUDA lowrank backend real subset artifact: FASTKPC_RUN_REAL_SUBSET_TESTS != 1\n")
  quit(save = "no", status = 0)
}

source("fastkpc/R/legacy_dcov_cuda_lowrank_backend_artifact.R")

missing <- c("graph", "mgcv", "pcalg", "Rcpp", "RcppArmadillo",
             "RcppEigen", "RSpectra")[
  !vapply(c("graph", "mgcv", "pcalg", "Rcpp", "RcppArmadillo",
            "RcppEigen", "RSpectra"),
          requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP legacy dCov CUDA lowrank backend real subset artifact: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

real_path <- "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"
if (!file.exists(real_path)) {
  cat("SKIP legacy dCov CUDA lowrank backend real subset artifact: real fixture unavailable\n")
  quit(save = "no", status = 0)
}

output_dir <- tempfile("legacy_dcov_cuda_lowrank_backend_real_subset_")
artifact <- fastkpc_run_legacy_dcov_cuda_lowrank_backend_real_subset_artifact(
  output_dir = output_dir,
  artifact_name = "legacy_dcov_cuda_lowrank_backend_real_subset_test",
  data_path = real_path,
  columns = c(1L, 2L, 3L, 4L, 5L),
  alpha = 0.1,
  max_conditioning_size = 3L,
  rebuild_cuda = FALSE
)

summary <- artifact$summary
paths <- artifact$paths

assert_true(file.exists(paths$summary_csv),
            "real subset CUDA lowrank backend artifact should write summary.csv")
assert_true(file.exists(paths$summary_md),
            "real subset CUDA lowrank backend artifact should write summary.md")
assert_true(file.exists(paths$result_rds),
            "real subset CUDA lowrank backend artifact should write result.rds")
assert_true(identical(summary$adjacency_identical[[1L]], TRUE),
            "real subset CUDA lowrank backend should match CPU Spectra adjacency")
assert_true(identical(summary$n_edgetests_exact[[1L]], TRUE),
            "real subset CUDA lowrank backend should match CPU Spectra n.edgetests")
assert_true(identical(as.integer(summary$shd[[1L]]), 0L),
            "real subset CUDA lowrank backend should have SHD 0 vs CPU Spectra")
assert_true(identical(
  as.integer(summary$candidate_legacy_dcov_cuda_lowrank_backend_count[[1L]]),
  as.integer(summary$candidate_legacy_dcov_gamma_count[[1L]])),
  "real subset CUDA lowrank backend should be authoritative for every dCov call")
assert_true(identical(
  as.integer(summary$candidate_legacy_dcov_cuda_lowrank_backend_error_count[[1L]]),
  0L),
  "real subset CUDA lowrank backend should not report errors")
assert_true(identical(
  as.integer(summary$candidate_legacy_dcov_cuda_lowrank_backend_fallback_count[[1L]]),
  0L),
  "real subset CUDA lowrank backend should not fallback on the real subset")
assert_true(identical(
  as.integer(summary$candidate_legacy_dcov_cuda_lowrank_backend_converged_count[[1L]]),
  as.integer(summary$candidate_legacy_dcov_cuda_lowrank_backend_count[[1L]])),
  "real subset CUDA lowrank backend should converge for every authority call")
assert_true(
  as.integer(
    summary$candidate_legacy_dcov_cuda_lowrank_backend_spectra_matvec_count[[1L]]
  ) > 0L,
  "real subset CUDA lowrank backend should report Spectra matvecs")
assert_true(identical(
  as.numeric(
    summary$candidate_legacy_dcov_cuda_lowrank_backend_matrix_h2d_ms_during_compute_max[[1L]]
  ),
  0),
  "real subset CUDA lowrank backend should not re-upload matrices during compute")

cat(sprintf(
  "PASS legacy dCov CUDA lowrank backend real subset artifact calls=%d shd=%d\n",
  as.integer(summary$candidate_legacy_dcov_gamma_count[[1L]]),
  as.integer(summary$shd[[1L]])
))
