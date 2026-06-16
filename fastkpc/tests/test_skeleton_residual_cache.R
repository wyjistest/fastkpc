source("fastkpc/R/native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

set_key <- function(values) paste(sort(as.integer(values)), collapse = ",")

compare_sepsets_exact <- function(a, b) {
  for (i in seq_along(a)) {
    for (j in seq_along(a[[i]])) {
      if (!identical(set_key(a[[i]][[j]]), set_key(b[[i]][[j]]))) {
        return(FALSE)
      }
    }
  }
  TRUE
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

alpha <- 0.2
max_ord <- 2

plain <- fast_skeleton_cpp(data, alpha = alpha, max_conditioning_size = max_ord)
uncached <- fast_skeleton_cpp_cached(data, alpha = alpha,
                                     max_conditioning_size = max_ord,
                                     residual_cache = FALSE)
cached <- fast_skeleton_cpp_cached(data, alpha = alpha,
                                   max_conditioning_size = max_ord,
                                   residual_cache = TRUE)

assert_true(identical(uncached$adjacency, plain$adjacency),
            "uncached cached-wrapper adjacency should match fast_skeleton_cpp")
assert_true(max(abs(uncached$pMax - plain$pMax)) < 1e-10,
            "uncached cached-wrapper pMax should match fast_skeleton_cpp")
assert_true(compare_sepsets_exact(uncached$sepsets, plain$sepsets),
            "uncached cached-wrapper sepsets should match fast_skeleton_cpp")
assert_true(identical(uncached$n.edgetests, plain$n.edgetests),
            "uncached cached-wrapper n.edgetests should match fast_skeleton_cpp")

assert_true(identical(cached$adjacency, uncached$adjacency),
            "cached CPU adjacency should match uncached")
assert_true(max(abs(cached$pMax - uncached$pMax)) < 1e-10,
            "cached CPU pMax should match uncached")
assert_true(compare_sepsets_exact(cached$sepsets, uncached$sepsets),
            "cached CPU sepsets should match uncached")
assert_true(identical(cached$n.edgetests, uncached$n.edgetests),
            "cached CPU n.edgetests should match uncached")

assert_true(cached$residual_cache$enabled, "cached run should report enabled TRUE")
assert_true(cached$residual_cache$hits > 0, "cached run should report cache hits")
assert_true(cached$residual_cache$computations < cached$residual_cache$requests,
            "cached run should compute fewer residuals than requests")
assert_true(cached$residual_backend == "linear", "residual backend should be linear")

assert_true(!uncached$residual_cache$enabled, "uncached run should report enabled FALSE")
assert_true(uncached$residual_cache$hits == 0L, "uncached run should report no hits")

cat("test_skeleton_residual_cache.R: PASS\n")
