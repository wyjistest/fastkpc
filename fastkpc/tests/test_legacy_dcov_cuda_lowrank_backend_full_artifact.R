fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP legacy dCov CUDA lowrank backend full artifact: FASTKPC_RUN_CUDA_TESTS != 1\n")
  quit(save = "no", status = 0)
}
if (!identical(Sys.getenv("FASTKPC_RUN_REAL_SUBSET_TESTS", unset = ""), "1")) {
  cat("SKIP legacy dCov CUDA lowrank backend full artifact: FASTKPC_RUN_REAL_SUBSET_TESTS != 1\n")
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
  cat("SKIP legacy dCov CUDA lowrank backend full artifact: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

real_path <- "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"
reference_path <- paste0(
  "fastkpc/artifacts/legacy_mgcv_residual_cache_s_affinity_v1/",
  "compatible_legacy_cpp_dcov_mgcv_cache_s_affinity_result.rds"
)
if (!file.exists(real_path)) {
  cat("SKIP legacy dCov CUDA lowrank backend full artifact: real fixture unavailable\n")
  quit(save = "no", status = 0)
}
if (!file.exists(reference_path)) {
  cat("SKIP legacy dCov CUDA lowrank backend full artifact: reference unavailable\n")
  quit(save = "no", status = 0)
}

output_dir <- tempfile("legacy_dcov_cuda_lowrank_backend_full_timeout_")
artifact <- fastkpc_run_legacy_dcov_cuda_lowrank_backend_full_artifact(
  output_dir = output_dir,
  artifact_name = "legacy_dcov_cuda_lowrank_backend_full_timeout_test",
  data_path = real_path,
  reference_result_path = reference_path,
  candidate_timeout_sec = 0,
  rebuild_cuda = FALSE
)

summary <- artifact$summary
paths <- artifact$paths
progress <- utils::read.csv(paths$progress_csv, stringsAsFactors = FALSE)

assert_true(file.exists(paths$summary_csv),
            "full CUDA lowrank backend artifact should write summary.csv")
assert_true(file.exists(paths$summary_md),
            "full CUDA lowrank backend artifact should write summary.md")
assert_true(file.exists(paths$result_rds),
            "full CUDA lowrank backend artifact should write result.rds")
assert_true(file.exists(paths$progress_csv),
            "full CUDA lowrank backend artifact should write progress.csv")
assert_true(!is.null(paths$legacy_progress_csv) &&
              grepl("legacy_progress[.]csv$", paths$legacy_progress_csv),
            "full CUDA lowrank backend artifact should expose legacy progress path")
assert_true(identical(summary$run_status[[1L]], "timeout"),
            "full CUDA lowrank backend timeout should record run_status")
assert_true(isTRUE(summary$timeout[[1L]]),
            "full CUDA lowrank backend timeout should mark timeout")
assert_true(identical(as.numeric(summary$timeout_sec[[1L]]), 0),
            "full CUDA lowrank backend timeout should record timeout_sec")
assert_true(identical(as.integer(summary$n[[1L]]), 351L),
            "full CUDA lowrank backend artifact should use 351 rows")
assert_true(identical(as.integer(summary$p[[1L]]), 48L),
            "full CUDA lowrank backend artifact should use all 48 columns")
assert_true(identical(summary$columns[[1L]], "all"),
            "full CUDA lowrank backend artifact should record all columns")
assert_true(identical(as.integer(summary$expected_edge_count[[1L]]), 110L),
            "full CUDA lowrank backend artifact should record expected edge count")
assert_true(identical(summary$expected_n_edgetests[[1L]],
                      "2213,52659,125293,40694,13293,5422,835,80"),
            "full CUDA lowrank backend artifact should record expected n.edgetests")
assert_true(is.na(summary$shd[[1L]]),
            "full CUDA lowrank backend timeout should leave SHD unknown")
assert_true(is.null(artifact$candidate),
            "full CUDA lowrank backend timeout should not return a candidate result")
assert_true(any(progress$route == "candidate" &
                  progress$event == "timeout" &
                  progress$status == "timeout"),
            "full CUDA lowrank backend artifact should record candidate timeout")

cat("PASS legacy dCov CUDA lowrank backend full artifact timeout gate\n")
