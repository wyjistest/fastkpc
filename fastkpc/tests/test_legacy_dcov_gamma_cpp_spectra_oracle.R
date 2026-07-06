source("fastkpc/R/native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

missing <- c("Rcpp", "RcppArmadillo", "RcppEigen", "RSpectra")[
  !vapply(c("Rcpp", "RcppArmadillo", "RcppEigen", "RSpectra"),
          requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP legacy dCov gamma C++ Spectra oracle: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

old_mode <- Sys.getenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK",
                       unset = NA_character_)
on.exit({
  if (is.na(old_mode)) {
    Sys.unsetenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK")
  } else {
    Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = old_mode)
  }
}, add = TRUE)

fixture_path <- "fastkpc/tests/fixtures/legacy_dcov_gamma_oracle_v1.rds"
assert_true(file.exists(fixture_path),
            "legacy dCov gamma oracle fixture should exist")
cases <- readRDS(fixture_path)
assert_true(length(cases) >= 6L,
            "legacy dCov gamma oracle fixture should include representative cases")

Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = "spectra")
tol <- 1e-8
for (case in cases) {
  meta <- case$meta[1L, , drop = FALSE]
  residuals <- case$residuals
  cpp <- fastkpc_legacy_dcov_gamma_cpp_oracle(
    residuals$rx,
    residuals$ry,
    numCol = meta$numCol,
    index = meta$index
  )
  diag <- cpp$diagnostics
  label <- meta$case_id

  assert_true(abs(cpp$p.value - meta$p.value) <= tol,
              paste(label, "Spectra C++ oracle p.value should match legacy oracle"))
  assert_true(abs(cpp$nV2 - meta$nV2) <= tol,
              paste(label, "Spectra C++ oracle nV2 should match legacy oracle"))
  assert_true(abs(cpp$mean - meta$nV2Mean) <= tol,
              paste(label, "Spectra C++ oracle mean should match legacy oracle"))
  assert_true(abs(cpp$variance - meta$nV2Variance) <= tol,
              paste(label, "Spectra C++ oracle variance should match legacy oracle"))
  assert_true(identical(cpp$p.value >= meta$alpha, meta$delete_edge),
              paste(label, "Spectra C++ oracle decision should match legacy oracle"))
  assert_true(identical(diag$lowrank_mode, "spectra"),
              paste(label, "Spectra C++ oracle should report spectra lowrank mode"))
  assert_true(identical(as.integer(diag$lowrank_spectra_count), 2L),
              paste(label, "Spectra C++ oracle should run x/y selected eigs"))
  assert_true(identical(as.integer(diag$lowrank_spectra_failed_count), 0L),
              paste(label, "Spectra C++ oracle should not report selected eig failures"))
  assert_true(identical(as.integer(diag$lowrank_spectra_fallback_full_eig_count), 0L),
              paste(label, "Spectra C++ oracle should not fallback to full eig"))
  assert_true(as.integer(diag$lowrank_spectra_nconv) >= 2L * meta$numCol,
              paste(label, "Spectra C++ oracle should converge requested eigenpairs"))
}

cat("PASS legacy dCov gamma C++ Spectra oracle parity\n")
