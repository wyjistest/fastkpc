fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP legacy dCov CUDA lowrank backend hot12 artifact: FASTKPC_RUN_CUDA_TESTS != 1\n")
  quit(save = "no", status = 0)
}
if (!identical(Sys.getenv("FASTKPC_RUN_REAL_SUBSET_TESTS", unset = ""), "1")) {
  cat("SKIP legacy dCov CUDA lowrank backend hot12 artifact: FASTKPC_RUN_REAL_SUBSET_TESTS != 1\n")
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
  cat("SKIP legacy dCov CUDA lowrank backend hot12 artifact: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

real_path <- "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"
if (!file.exists(real_path)) {
  cat("SKIP legacy dCov CUDA lowrank backend hot12 artifact: real fixture unavailable\n")
  quit(save = "no", status = 0)
}

output_dir <- tempfile("legacy_dcov_cuda_lowrank_backend_hot12_")
artifact <- fastkpc_run_legacy_dcov_cuda_lowrank_backend_hot12_artifact(
  output_dir = output_dir,
  artifact_name = "legacy_dcov_cuda_lowrank_backend_hot12_test",
  data_path = real_path,
  alpha = 0.1,
  max_conditioning_size = 3L,
  rebuild_cuda = FALSE
)

summary <- artifact$summary
paths <- artifact$paths

assert_true(file.exists(paths$summary_csv),
            "hot12 CUDA lowrank backend artifact should write summary.csv")
assert_true(file.exists(paths$summary_md),
            "hot12 CUDA lowrank backend artifact should write summary.md")
assert_true(file.exists(paths$result_rds),
            "hot12 CUDA lowrank backend artifact should write result.rds")
assert_true(identical(summary$columns[[1L]],
                      "1,2,3,4,5,6,9,12,15,16,17,18"),
            "hot12 CUDA lowrank backend artifact should use canonical hot12 columns")
assert_true(identical(as.integer(summary$p[[1L]]), 12L),
            "hot12 CUDA lowrank backend artifact should run twelve variables")
assert_true(identical(summary$adjacency_identical[[1L]], TRUE),
            "hot12 CUDA lowrank backend should match CPU Spectra adjacency")
assert_true(identical(summary$n_edgetests_exact[[1L]], TRUE),
            "hot12 CUDA lowrank backend should match CPU Spectra n.edgetests")
assert_true(identical(as.integer(summary$shd[[1L]]), 0L),
            "hot12 CUDA lowrank backend should have SHD 0 vs CPU Spectra")
assert_true(identical(
  as.integer(summary$candidate_legacy_dcov_cuda_lowrank_backend_count[[1L]]),
  as.integer(summary$candidate_legacy_dcov_gamma_count[[1L]])),
  "hot12 CUDA lowrank backend should be authoritative for every dCov call")
assert_true(identical(
  as.integer(summary$candidate_legacy_dcov_cuda_lowrank_backend_error_count[[1L]]),
  0L),
  "hot12 CUDA lowrank backend should not report errors")
assert_true(identical(
  as.integer(summary$candidate_legacy_dcov_cuda_lowrank_backend_fallback_count[[1L]]),
  0L),
  "hot12 CUDA lowrank backend should not fallback on hot12")
assert_true(identical(
  as.integer(summary$candidate_legacy_dcov_cuda_lowrank_backend_converged_count[[1L]]),
  as.integer(summary$candidate_legacy_dcov_cuda_lowrank_backend_count[[1L]])),
  "hot12 CUDA lowrank backend should converge for every authority call")
assert_true(
  as.integer(
    summary$candidate_legacy_dcov_cuda_lowrank_backend_spectra_matvec_count[[1L]]
  ) > 0L,
  "hot12 CUDA lowrank backend should report Spectra matvecs")
assert_true(identical(
  as.numeric(
    summary$candidate_legacy_dcov_cuda_lowrank_backend_matrix_h2d_ms_during_compute_max[[1L]]
  ),
  0),
  "hot12 CUDA lowrank backend should not re-upload matrices during compute")

cat(sprintf(
  "PASS legacy dCov CUDA lowrank backend hot12 artifact calls=%d shd=%d\n",
  as.integer(summary$candidate_legacy_dcov_gamma_count[[1L]]),
  as.integer(summary$shd[[1L]])
))
