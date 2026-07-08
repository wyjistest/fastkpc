source("fastkpc/R/native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

missing <- c("Rcpp", "RcppArmadillo")[
  !vapply(c("Rcpp", "RcppArmadillo"), requireNamespace, logical(1),
          quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP legacy dCov gamma C++ batch oracle: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

fixture_path <- "fastkpc/tests/fixtures/legacy_dcov_gamma_oracle_v1.rds"
assert_true(file.exists(fixture_path),
            "legacy dCov gamma oracle fixture should exist")
cases <- readRDS(fixture_path)
num_col <- unique(vapply(cases, function(case) case$meta$numCol, integer(1)))
index <- unique(vapply(cases, function(case) case$meta$index, numeric(1)))
assert_true(length(num_col) == 1L && length(index) == 1L,
            "batch oracle fixture should use one numCol/index group")

x <- do.call(cbind, lapply(cases, function(case) case$residuals$rx))
y <- do.call(cbind, lapply(cases, function(case) case$residuals$ry))
batch <- fastkpc_legacy_dcov_gamma_cpp_oracle_batch(
  x, y, numCol = num_col, index = index
)
scalar <- lapply(cases, function(case) {
  fastkpc_legacy_dcov_gamma_cpp_oracle(
    case$residuals$rx,
    case$residuals$ry,
    numCol = case$meta$numCol,
    index = case$meta$index
  )
})

scalar_tol <- 1e-12
legacy_tol <- 1e-8
expected_p <- vapply(cases, function(case) case$meta$p.value, numeric(1))
expected_nV2 <- vapply(cases, function(case) case$meta$nV2, numeric(1))
expected_mean <- vapply(cases, function(case) case$meta$nV2Mean, numeric(1))
expected_variance <- vapply(cases, function(case) case$meta$nV2Variance,
                            numeric(1))
expected_decision <- vapply(cases, function(case) {
  case$meta$p.value >= case$meta$alpha
}, logical(1))

scalar_p <- vapply(scalar, function(result) result$p.value, numeric(1))
scalar_nV2 <- vapply(scalar, function(result) result$nV2, numeric(1))
scalar_mean <- vapply(scalar, function(result) result$mean, numeric(1))
scalar_variance <- vapply(scalar, function(result) result$variance, numeric(1))

assert_true(max(abs(batch$p.value - scalar_p)) <= scalar_tol,
            "batched C++ oracle p.values should match scalar C++ oracle")
assert_true(max(abs(batch$nV2 - scalar_nV2)) <= scalar_tol,
            "batched C++ oracle nV2 should match scalar C++ oracle")
assert_true(max(abs(batch$mean - scalar_mean)) <= scalar_tol,
            "batched C++ oracle mean should match scalar C++ oracle")
assert_true(max(abs(batch$variance - scalar_variance)) <= scalar_tol,
            "batched C++ oracle variance should match scalar C++ oracle")
assert_true(max(abs(batch$p.value - expected_p)) <= legacy_tol,
            "batched C++ oracle p.values should match legacy oracle")
assert_true(max(abs(batch$nV2 - expected_nV2)) <= legacy_tol,
            "batched C++ oracle nV2 should match legacy oracle")
assert_true(max(abs(batch$mean - expected_mean)) <= legacy_tol,
            "batched C++ oracle mean should match legacy oracle")
assert_true(max(abs(batch$variance - expected_variance)) <= legacy_tol,
            "batched C++ oracle variance should match legacy oracle")
assert_true(identical(batch$p.value >= 0.1, expected_decision),
            "batched C++ oracle decisions should match legacy oracle")
assert_true(identical(as.integer(batch$diagnostics$batch_count),
                      as.integer(ncol(x))),
            "batched C++ oracle should report batch count")
assert_true(identical(as.integer(batch$diagnostics$n), as.integer(nrow(x))),
            "batched C++ oracle should report n")
assert_true(batch$diagnostics$total_ms >= 0,
            "batched C++ oracle should report elapsed time")
batch_diag_fields <- c(
  "input_ms", "distance_ms", "lowrank_ms", "lowrank_eig_ms",
  "lowrank_select_ms", "lowrank_center_ms", "lowrank_unaccounted_ms",
  "lowrank_full_eig_count", "lowrank_spectra_count",
  "lowrank_spectra_converged_count", "lowrank_spectra_failed_count",
  "lowrank_spectra_fallback_full_eig_count", "statistic_ms", "moment_ms",
  "pgamma_ms", "accounted_ms", "scalar_total_ms", "wrapper_overhead_ms",
  "batch_overhead_ms", "unaccounted_ms", "workspace_reuse_enabled",
  "distance_workspace_reuse_count", "statistic_moment_workspace_reuse_count",
  "lowrank_output_workspace_reuse_count",
  "lowrank_eig_workspace_reuse_count", "column_copy_count"
)
missing_batch_diag <- setdiff(batch_diag_fields, names(batch$diagnostics))
assert_true(length(missing_batch_diag) == 0L,
            paste("batched C++ oracle diagnostics missing",
                  missing_batch_diag[[1L]]))
assert_true(batch$diagnostics$input_ms >= 0 &&
              batch$diagnostics$distance_ms > 0 &&
              batch$diagnostics$lowrank_ms > 0 &&
              batch$diagnostics$statistic_ms >= 0 &&
              batch$diagnostics$moment_ms > 0 &&
              batch$diagnostics$pgamma_ms >= 0,
            "batched C++ oracle should aggregate stage timings")
assert_true(as.integer(batch$diagnostics$lowrank_full_eig_count) +
              as.integer(batch$diagnostics$lowrank_spectra_count) ==
              2L * ncol(x),
            "batched C++ oracle should aggregate two lowrank solves per pair")
accounted_parts <- batch$diagnostics$input_ms +
  batch$diagnostics$distance_ms +
  batch$diagnostics$lowrank_ms +
  batch$diagnostics$statistic_ms +
  batch$diagnostics$moment_ms +
  batch$diagnostics$pgamma_ms
assert_true(abs(accounted_parts - batch$diagnostics$accounted_ms) <= 1e-6,
            "batched C++ oracle accounted time should sum stage timings")
assert_true(batch$diagnostics$scalar_total_ms >=
              batch$diagnostics$accounted_ms,
            "batched C++ oracle should report scalar total time")
assert_true(batch$diagnostics$wrapper_overhead_ms >= 0,
            "batched C++ oracle should report wrapper overhead time")
assert_true(batch$diagnostics$batch_overhead_ms >= 0,
            "batched C++ oracle should report total batch overhead time")
assert_true(abs(
  batch$diagnostics$total_ms -
    batch$diagnostics$scalar_total_ms -
    batch$diagnostics$wrapper_overhead_ms
) <= 1e-6,
"batched C++ oracle total time should decompose into scalar and wrapper time")
assert_true(isTRUE(batch$diagnostics$workspace_reuse_enabled),
            "batched C++ oracle should report reusable workspace mode")
assert_true(identical(
  as.integer(batch$diagnostics$distance_workspace_reuse_count),
  2L * ncol(x)
), "batched C++ oracle should reuse x/y distance workspaces per pair")
assert_true(identical(
  as.integer(batch$diagnostics$statistic_moment_workspace_reuse_count),
  3L * ncol(x)
), "batched C++ oracle should reuse statistic/moment cross workspace per pair")
assert_true(identical(
  as.integer(batch$diagnostics$lowrank_output_workspace_reuse_count),
  2L * ncol(x)
), "batched C++ oracle should reuse x/y lowrank output workspaces per pair")
assert_true(identical(
  as.integer(batch$diagnostics$lowrank_eig_workspace_reuse_count),
  2L * ncol(x)
), "batched C++ oracle should reuse x/y lowrank eig workspaces per pair")
assert_true(identical(as.integer(batch$diagnostics$column_copy_count), 0L),
            "batched C++ oracle should avoid per-column Rcpp vector copies")

cat("PASS legacy dCov gamma C++ batch oracle parity\n")
