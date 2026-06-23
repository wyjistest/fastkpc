source("fastkpc/R/kpc_tprs_residual_cpp.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_equal <- function(actual, expected, message) {
  if (!identical(actual, expected)) {
    fail(paste0(message, ": expected ", paste(expected, collapse = ","),
                " got ", paste(actual, collapse = ",")))
  }
}

set.seed(62302)
n <- 64L
t <- seq(-1, 1, length.out = n)
S1 <- matrix(c(t, t[seq_len(8L)]), ncol = 1)
setup1 <- kpc_tprs_residual_cpp_setup(S1)
assert_equal(setup1$backend_family, "kpcTprsResidualCPP",
             "1D backend family")
assert_equal(setup1$null_space_rank, 2L, "1D null-space rank")
assert_true(is.matrix(setup1$X), "1D setup matrix")
assert_equal(nrow(setup1$X), nrow(S1), "1D setup rows")
assert_equal(dim(setup1$penalty), rep(ncol(setup1$X), 2L),
             "1D penalty dimension")
assert_true(max(abs(setup1$penalty - t(setup1$penalty))) < 1e-12,
            "1D penalty symmetric")

S2 <- cbind(
  s1 = stats::runif(n, -1.5, 1.5),
  s2 = stats::runif(n, -0.8, 0.8)
)
theta <- pi / 5
rotation <- matrix(c(cos(theta), -sin(theta), sin(theta), cos(theta)), 2, 2)
S2_rot <- S2 %*% rotation
setup2 <- kpc_tprs_residual_cpp_setup(S2, k = 9L)
setup2_rot <- kpc_tprs_residual_cpp_setup(S2_rot, k = 9L)

assert_equal(setup2$null_space_rank, 3L, "2D null-space rank")
assert_equal(setup2$smooth_geometry, "joint-isotropic",
             "2D smooth geometry")
assert_equal(setup2$radial_basis, "r^2 log(r)", "2D radial basis")
assert_equal(dim(setup2$penalty), rep(ncol(setup2$X), 2L),
             "2D penalty dimension")
assert_true(max(abs(setup2$penalty - t(setup2$penalty))) < 1e-12,
            "2D penalty symmetric")
assert_true(max(abs(setup2$penalty - setup2_rot$penalty)) < 1e-10,
            "2D isotropic knot penalty should be rotation invariant")

bad <- tryCatch(
  kpc_tprs_residual_cpp_setup(cbind(S2, s3 = stats::rnorm(n))),
  error = function(e) e
)
assert_true(inherits(bad, "error"), "3D setup must fail closed")
assert_true(grepl("|S| = 1 or 2", conditionMessage(bad), fixed = TRUE),
            "3D setup error should name supported conditioning size")

cat("PASS kpcTprsResidualCPP setup geometry\n")
