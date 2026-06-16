source("fastkpc/R/native.R")
source("fastkpc/R/diff_report.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

max_abs_diff <- function(a, b) {
  max(abs(as.numeric(a) - as.numeric(b)))
}

set.seed(51)
n <- 120
z1 <- runif(n, -2, 2)
z2 <- runif(n, -2, 2)
data <- cbind(
  x1 = sin(z1) + rnorm(n, sd = 0.2),
  x2 = cos(z1) + rnorm(n, sd = 0.2),
  x3 = sin(z2) + rnorm(n, sd = 0.2),
  x4 = z1 * z2 + rnorm(n, sd = 0.2),
  x5 = rnorm(n)
)
alpha <- 0.2
max_ord <- 2

linear_backend <- fast_skeleton_cpp_backend(
  data, alpha, max_ord, residual_backend = "linear", residual_cache = TRUE
)
linear_cached <- fast_skeleton_cpp_cached(
  data, alpha, max_ord, residual_cache = TRUE
)

assert_true(identical(linear_backend$adjacency, linear_cached$adjacency),
            "linear backend adjacency should match cached linear wrapper")
assert_true(max_abs_diff(linear_backend$pMax, linear_cached$pMax) < 1e-10,
            "linear backend pMax should match cached linear wrapper")

fastspline_a <- fast_skeleton_cpp_backend(
  data, alpha, max_ord, residual_backend = "fastSpline", residual_cache = TRUE
)
fastspline_b <- fast_skeleton_cpp_backend(
  data, alpha, max_ord, residual_backend = "fastSpline", residual_cache = TRUE
)

assert_true(fastspline_a$backend == "cpu", "fastSpline CPU result backend should be cpu")
assert_true(fastspline_a$residual_backend == "fastSpline",
            "fastSpline CPU result should record residual backend")
assert_true(is.character(fastspline_a$residual_backend_params),
            "fastSpline CPU result should include backend params")
assert_true(fastspline_a$residual_cache$hits > 0,
            "fastSpline CPU cache should have hits")
assert_true(fastspline_a$residual_cache$computations < fastspline_a$residual_cache$requests,
            "fastSpline CPU cache computations should be lower than requests")

adj <- fastspline_a$adjacency
assert_true(identical(adj, t(adj)), "fastSpline adjacency should be symmetric")
assert_true(!any(diag(adj)), "fastSpline adjacency diagonal should be FALSE")

pmax <- fastspline_a$pMax
assert_true(max_abs_diff(pmax, t(pmax)) < 1e-12,
            "fastSpline pMax should be symmetric")
assert_true(max_abs_diff(diag(pmax), rep(1, ncol(pmax))) < 1e-12,
            "fastSpline pMax diagonal should be one")

assert_true(identical(fastspline_a$adjacency, fastspline_b$adjacency),
            "fastSpline repeated adjacency should be identical")
assert_true(identical(fastspline_a$sepsets, fastspline_b$sepsets),
            "fastSpline repeated sepsets should be identical")
assert_true(identical(fastspline_a$n.edgetests, fastspline_b$n.edgetests),
            "fastSpline repeated n.edgetests should be identical")
assert_true(max_abs_diff(fastspline_a$pMax, fastspline_b$pMax) < 1e-12,
            "fastSpline repeated pMax should match")

diff <- summarize_graph_diff(linear_backend, fastspline_a)
assert_true(is.list(diff$adjacency), "graph diff should include adjacency section")
assert_true(is.list(diff$pMax), "graph diff should include pMax section")
assert_true(is.list(diff$sepsets), "graph diff should include sepsets section")
assert_true(is.list(diff$n_edgetests), "graph diff should include n_edgetests section")

cat("test_skeleton_fastspline_cpu.R: PASS\n")
