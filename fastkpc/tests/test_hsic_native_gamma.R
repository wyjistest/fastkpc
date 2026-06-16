source("fastkpc/R/native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

set.seed(211)
n <- 48
x <- seq(-2, 2, length.out = n)
y_dep <- sin(x) + rnorm(n, sd = 0.05)
y_ind <- sample(y_dep)

build_fastkpc_native(rebuild = TRUE)

dep <- fast_hsic_gamma_cpp(x, y_dep, sig = 1)
ind <- fast_hsic_gamma_cpp(x, y_ind, sig = 1)
repeat_dep <- fast_hsic_gamma_cpp(x, y_dep, sig = 1)

assert_true(is.list(dep), "HSIC gamma result should be a list")
assert_true(is.finite(dep$statistic), "HSIC statistic should be finite")
assert_true(is.finite(dep$p.value), "HSIC p-value should be finite")
assert_true(dep$p.value >= 0 && dep$p.value <= 1,
            "HSIC p-value should be in [0, 1]")
assert_true(dep$statistic > ind$statistic,
            "dependent fixture should have larger HSIC statistic")
assert_true(abs(dep$statistic - repeat_dep$statistic) < 1e-12,
            "HSIC gamma should repeat exactly")
assert_true(all(c("hsic", "mean", "variance", "shape", "scale") %in%
                  names(dep$diagnostics)),
            "HSIC gamma diagnostics should include gamma fields")

cat("test_hsic_native_gamma.R: PASS\n")
