source("fastkpc/R/dcov_exact.R")
source("gpu-dcov/dcov_gamma_gpu.R")

relerr <- function(a, b) {
  abs(a - b) / pmax(abs(b), .Machine$double.eps)
}

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

assert_error <- function(expr, pattern) {
  msg <- tryCatch({
    force(expr)
    NA_character_
  }, error = function(e) conditionMessage(e))
  if (is.na(msg)) {
    stop("Expected error matching: ", pattern, call. = FALSE)
  }
  if (!grepl(pattern, msg, fixed = TRUE)) {
    stop("Error message did not match. Got: ", msg, call. = FALSE)
  }
}

check_against_gpu_scalars <- function(n) {
  set.seed(1000 + n)
  x <- rnorm(n)
  y <- 0.3 * x^2 + rnorm(n)

  exact <- dcov_gamma_exact(x, y)
  gpu <- dcov.gpu.stats(x, y)
  Sab <- gpu[1]
  Saa <- gpu[2]
  Sbb <- gpu[3]
  sumK <- gpu[4]
  sumL <- gpu[5]

  expected_nV2 <- Sab / n
  expected_mean <- (sumK / n^2) * (sumL / n^2)
  expected_var <- 2 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3) *
    Saa * Sbb / n^2

  assert_true(relerr(unname(exact$statistic), expected_nV2) < 1e-10,
              paste("nV2 mismatch for n", n))
  assert_true(relerr(unname(exact$estimates[2]), expected_mean) < 1e-10,
              paste("mean mismatch for n", n))
  assert_true(relerr(unname(exact$estimates[3]), expected_var) < 1e-10,
              paste("variance mismatch for n", n))
}

dcov.gpu.warmup()

check_against_gpu_scalars(300)
check_against_gpu_scalars(1000)

assert_error(
  dcov_gamma_exact(1:5, 1:5),
  "gamma approximation requires n > 5"
)

set.seed(42)
x <- rnorm(80)
y <- rnorm(80)
legacy <- dcov_gamma_exact(x, y, index = 1.5, legacy_index = TRUE)
semantic <- dcov_gamma_exact(x, y, index = 1.5, legacy_index = FALSE)
assert_true(abs(unname(legacy$statistic) - unname(semantic$statistic)) > 1e-8,
            "legacy_index should change statistic when index != 1")

assert_true(is.finite(semantic$p.value), "p.value should be finite")
assert_true(semantic$p.value >= 0 && semantic$p.value <= 1,
            "p.value should be in [0, 1]")

cat("test_dcov_exact.R: PASS\n")
