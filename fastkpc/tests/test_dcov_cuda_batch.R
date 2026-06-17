source("fastkpc/R/dcov_exact.R")
source("fastkpc/R/cuda_native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

assert_error <- function(expr, pattern) {
  msg <- tryCatch({
    force(expr)
    NA_character_
  }, error = function(e) conditionMessage(e))
  if (is.na(msg)) stop("Expected error matching: ", pattern, call. = FALSE)
  if (!grepl(pattern, msg, fixed = TRUE)) {
    stop("Error message did not match. Got: ", msg, call. = FALSE)
  }
}

check_column_matches_cpu <- function(x, y, index = 1, legacy_index = TRUE, tol = 1e-10) {
  cuda <- fast_dcov_batch_cuda(x, y, index = index, legacy_index = legacy_index)
  xmat <- if (is.matrix(x)) x else matrix(as.numeric(x), ncol = 1)
  ymat <- if (is.matrix(y)) y else matrix(as.numeric(y), ncol = 1)
  for (k in seq_len(ncol(xmat))) {
    cpu <- dcov_gamma_exact(xmat[, k], ymat[, k], index = index,
                            legacy_index = legacy_index)
    assert_true(abs(cuda$p.value[k] - cpu$p.value) < tol,
                sprintf("p.value mismatch for column %d", k))
    assert_true(abs(cuda$nV2[k] - unname(cpu$statistic)) < tol,
                sprintf("nV2 mismatch for column %d", k))
    assert_true(abs(cuda$mean[k] - unname(cpu$estimates[2])) < tol,
                sprintf("mean mismatch for column %d", k))
    assert_true(abs(cuda$variance[k] - unname(cpu$estimates[3])) < tol,
                sprintf("variance mismatch for column %d", k))
  }
}

build_fastkpc_cuda_native(rebuild = TRUE)

set.seed(21)
x <- rnorm(80)
y <- 0.2 * x + rnorm(80)
check_column_matches_cpu(x, y)

set.seed(22)
xb <- matrix(rnorm(120 * 7), 120, 7)
yb <- xb * rep(seq(0.05, 0.35, length.out = 7), each = 120) +
  matrix(rnorm(120 * 7), 120, 7)
check_column_matches_cpu(xb, yb)

set.seed(23)
large_x <- matrix(rnorm(300 * 64), 300, 64)
large_y <- matrix(rnorm(300 * 64), 300, 64)
large <- fast_dcov_batch_cuda(large_x, large_y)
assert_true(length(large$p.value) == 64, "batch output should have 64 p-values")
assert_true(all(is.finite(large$p.value)), "batch p-values should be finite")
assert_true(all(large$p.value >= 0 & large$p.value <= 1),
            "batch p-values should be in [0, 1]")

set.seed(2301)
wide_batch <- 70000L
wide_x <- matrix(rnorm(12 * wide_batch), 12, wide_batch)
wide_y <- matrix(rnorm(12 * wide_batch), 12, wide_batch)
wide <- fast_dcov_batch_cuda(wide_x, wide_y)
assert_true(length(wide$p.value) == wide_batch,
            "wide batch output should keep all p-values")
assert_true(all(is.finite(wide$p.value)),
            "wide batch p-values should be finite")
for (k in c(1L, 35000L, wide_batch)) {
  cpu <- dcov_gamma_exact(wide_x[, k], wide_y[, k])
  assert_true(abs(wide$p.value[k] - cpu$p.value) < 1e-10,
              sprintf("wide batch p.value mismatch for column %d", k))
}

set.seed(24)
xi <- matrix(rnorm(90 * 3), 90, 3)
yi <- matrix(rnorm(90 * 3), 90, 3)
legacy <- fast_dcov_batch_cuda(xi, yi, index = 1.5, legacy_index = TRUE)
semantic <- fast_dcov_batch_cuda(xi, yi, index = 1.5, legacy_index = FALSE)
assert_true(max(abs(legacy$nV2 - semantic$nV2)) > 1e-8,
            "legacy_index should change nV2 when index != 1")
check_column_matches_cpu(xi, yi, index = 1.5, legacy_index = FALSE)

assert_error(fast_dcov_batch_cuda(1:5, 1:5), "gamma approximation requires n > 5")
bad <- matrix(rnorm(80), 40, 2)
bad[3, 1] <- Inf
assert_error(fast_dcov_batch_cuda(bad, bad), "Data contains missing or infinite values")

cat("test_dcov_cuda_batch.R: PASS\n")
