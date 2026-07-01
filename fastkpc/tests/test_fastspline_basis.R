source("fastkpc/R/native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

set.seed(41)
n <- 80
z1 <- sort(runif(n, -2, 2))
z2 <- rnorm(n)
constant <- rep(3, n)
data <- cbind(z1 = z1, z2 = z2, constant = constant)

self <- fastspline_basis_selftest(data)

one_d <- self$one_d
assert_true(one_d$nrow == n, "one_d nrow should match input n")
assert_true(one_d$ncol >= 6, "one_d should have at least six columns")
assert_true(one_d$row_sums_close_to_one, "one_d non-intercept rows should sum to one")
assert_true(one_d$finite, "one_d design and penalty should be finite")
assert_true(one_d$penalty_dim == one_d$ncol, "one_d penalty dimension should match design columns")
assert_true(one_d$penalty_symmetric, "one_d penalty should be symmetric")

two_d <- self$two_d
assert_true(two_d$nrow == n, "two_d nrow should match input n")
assert_true(two_d$ncol > one_d$ncol, "two_d tensor design should be wider than one_d")
assert_true(two_d$finite, "two_d design and penalty should be finite")
assert_true(two_d$penalty_dim == two_d$ncol, "two_d penalty dimension should match design columns")
assert_true(two_d$penalty_symmetric, "two_d penalty should be symmetric")

additive <- self$additive
assert_true(additive$nrow == n, "additive nrow should match input n")
assert_true(additive$ncol > one_d$ncol, "additive design should be wider than one_d")
assert_true(additive$finite, "additive design and penalty should be finite")
assert_true(additive$penalty_dim == additive$ncol, "additive penalty dimension should match design columns")
assert_true(additive$penalty_symmetric, "additive penalty should be symmetric")

constant <- self$constant
assert_true(constant$finite, "constant input design and penalty should be finite")
assert_true(constant$non_intercept_cols_all_zero_or_constant,
            "constant input non-intercept columns should be zero or constant")

basis_source <- readLines("fastkpc/src/fastspline_basis.cpp", warn = FALSE)
assert_true(!any(grepl("std::pow", basis_source, fixed = TRUE)),
            "cubic fastSpline basis evaluation should avoid std::pow")

cat("test_fastspline_basis.R: PASS\n")
