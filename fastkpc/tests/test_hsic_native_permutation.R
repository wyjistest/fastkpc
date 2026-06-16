source("fastkpc/R/native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

set.seed(212)
n <- 40
x <- seq(-1, 1, length.out = n)
y <- x^2 + rnorm(n, sd = 0.04)

build_fastkpc_native(rebuild = TRUE)

a <- fast_hsic_perm_cpp(x, y, sig = 1, replicates = 30L, seed = 99L,
                        include_observed = TRUE)
b <- fast_hsic_perm_cpp(x, y, sig = 1, replicates = 30L, seed = 99L,
                        include_observed = TRUE)
c <- fast_hsic_perm_cpp(x, y, sig = 1, replicates = 30L, seed = 100L,
                        include_observed = TRUE)

assert_true(is.finite(a$p.value), "HSIC permutation p-value should be finite")
assert_true(a$p.value >= 0 && a$p.value <= 1,
            "HSIC permutation p-value should be in [0, 1]")
assert_true(length(a$replicates) == 30L,
            "HSIC permutation should return requested replicates")
assert_true(identical(a$replicates, b$replicates),
            "HSIC permutation fixed seed should repeat exactly")
assert_true(!identical(a$replicates, c$replicates),
            "HSIC permutation different seed should change replicate order")
assert_true(a$diagnostics$replicates == 30L,
            "HSIC permutation diagnostics should record replicate count")

cat("test_hsic_native_permutation.R: PASS\n")
