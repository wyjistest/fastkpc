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

cat("PASS legacy dCov gamma C++ batch oracle parity\n")
