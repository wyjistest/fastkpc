source("fastkpc/R/native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

set.seed(31)
n <- 90
z1 <- rnorm(n)
z2 <- rnorm(n)
data <- cbind(
  x1 = z1 + rnorm(n, sd = 0.2),
  x2 = z1 - z2 + rnorm(n, sd = 0.2),
  x3 = z2 + rnorm(n, sd = 0.2),
  x4 = z1 * z2 + rnorm(n, sd = 0.2),
  x5 = rnorm(n)
)

self <- fast_residual_cache_selftest(data)

assert_true(self$key_order_invariant, "conditioning-set order should not change cache key")
assert_true(self$target_distinct, "different target variables should have distinct keys")
assert_true(self$params_distinct, "different backend params should have distinct keys")

enabled <- self$enabled_stats
assert_true(enabled$enabled, "enabled stats should report enabled TRUE")
assert_true(enabled$requests == 2L, "enabled cache should record two requests")
assert_true(enabled$hits == 1L, "enabled cache should record one hit")
assert_true(enabled$misses == 1L, "enabled cache should record one miss")
assert_true(enabled$computations == 1L, "enabled cache should compute once")
assert_true(enabled$stored_vectors == 1L, "enabled cache should store one vector")
assert_true(enabled$stored_values == n, "enabled cache should store n values")
assert_true(enabled$backend_name == "linear", "enabled cache backend should be linear")

disabled <- self$disabled_stats
assert_true(!disabled$enabled, "disabled stats should report enabled FALSE")
assert_true(disabled$requests == 2L, "disabled cache should record two requests")
assert_true(disabled$hits == 0L, "disabled cache should record no hits")
assert_true(disabled$misses == 0L, "disabled cache should record no misses")
assert_true(disabled$computations == 2L, "disabled cache should compute twice")
assert_true(disabled$stored_vectors == 0L, "disabled cache should store no vectors")

assert_true(self$max_abs_residual_diff < 1e-12,
            "cached residual values should match direct residualize_lm")

cat("test_residual_cache_core.R: PASS\n")
