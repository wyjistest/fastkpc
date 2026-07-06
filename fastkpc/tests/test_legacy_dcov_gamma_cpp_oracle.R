source("fastkpc/R/native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

missing <- c("Rcpp", "RcppArmadillo")[
  !vapply(c("Rcpp", "RcppArmadillo"), requireNamespace, logical(1),
          quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP legacy dCov gamma C++ oracle: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

fixture_path <- "fastkpc/tests/fixtures/legacy_dcov_gamma_oracle_v1.rds"
assert_true(file.exists(fixture_path),
            "legacy dCov gamma oracle fixture should exist")
cases <- readRDS(fixture_path)
assert_true(length(cases) >= 6L,
            "legacy dCov gamma oracle fixture should include representative cases")

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

  label <- meta$case_id
  assert_true(abs(cpp$p.value - meta$p.value) <= tol,
              paste(label, "C++ oracle p.value should match legacy oracle"))
  assert_true(abs(cpp$nV2 - meta$nV2) <= tol,
              paste(label, "C++ oracle nV2 should match legacy oracle"))
  assert_true(abs(cpp$mean - meta$nV2Mean) <= tol,
              paste(label, "C++ oracle mean should match legacy oracle"))
  assert_true(abs(cpp$variance - meta$nV2Variance) <= tol,
              paste(label, "C++ oracle variance should match legacy oracle"))
  assert_true(identical(cpp$p.value >= meta$alpha, meta$delete_edge),
              paste(label, "C++ oracle decision should match legacy oracle"))
  assert_true(all(c("distance_ms", "lowrank_ms", "lowrank_eig_ms",
                    "lowrank_select_ms", "lowrank_center_ms",
                    "lowrank_unaccounted_ms", "statistic_ms",
                    "moment_ms", "pgamma_ms", "total_ms") %in%
                    names(cpp$diagnostics)),
              paste(label, "C++ oracle should expose stage timings"))
  lowrank_parts <- cpp$diagnostics$lowrank_eig_ms +
    cpp$diagnostics$lowrank_select_ms +
    cpp$diagnostics$lowrank_center_ms +
    cpp$diagnostics$lowrank_unaccounted_ms
  assert_true(abs(lowrank_parts - cpp$diagnostics$lowrank_ms) <= 1e-6,
              paste(label, "C++ oracle lowrank timing should be accounted"))
}

cat("PASS legacy dCov gamma C++ oracle parity\n")
