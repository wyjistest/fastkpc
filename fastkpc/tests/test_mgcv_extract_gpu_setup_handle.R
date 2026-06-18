source("fastkpc/R/mgcv_extract_oracle.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_equal_num <- function(actual, expected, message, tol = 1e-10) {
  if (max(abs(as.numeric(actual) - as.numeric(expected))) > tol) {
    fail(paste0(message, ": max diff ",
                max(abs(as.numeric(actual) - as.numeric(expected)))))
  }
}

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP mgcv extract GPU setup handle: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

set.seed(242)
n <- 52
s1 <- stats::runif(n, -2, 2)
y <- sin(s1) + stats::rnorm(n, sd = 0.04)
data <- data.frame(y = y, s1 = s1)
formula <- y ~ s(s1, k = 8, bs = "tp")
sp <- 0.35

setup <- fastkpc_mgcv_extract_setup(
  formula = formula,
  data = data,
  sp = sp,
  target = 1L,
  S = 2L,
  k = 8L,
  bs = "tp"
)
handle <- fastkpc_mgcv_extract_gpu_setup_handle(setup)

assert_true(identical(handle$backend_family, "mgcvExtractGPU"),
            "handle backend should identify mgcvExtractGPU")
assert_true(identical(handle$mode, "fixed-sp-setup-handle"),
            "handle mode should identify setup handle")
assert_true(identical(handle$device_resident, FALSE),
            "R-level v1 handle should not claim device residency")
assert_true(identical(handle$setup_fingerprint$fingerprint,
                      setup$setup_fingerprint$fingerprint),
            "handle should preserve setup fingerprint")
assert_true(is.matrix(handle$X_null), "handle should expose null-space X")
assert_true(is.matrix(handle$penalty_null),
            "handle should expose null-space penalty")
assert_true(is.matrix(handle$XtX_null),
            "handle should expose null-space crossproduct")
assert_true(is.matrix(handle$Z), "handle should expose constraint nullspace")
assert_true(nrow(handle$X_null) == nrow(setup$X),
            "X_null should keep observation count")
assert_true(ncol(handle$X_null) == ncol(handle$Z),
            "X_null columns should match nullspace columns")
assert_equal_num(handle$X_null, setup$X %*% handle$Z,
                 "X_null should equal X %*% Z")

P <- fastkpc_assemble_penalty(
  p = ncol(setup$X),
  S = setup$S,
  off = setup$off,
  sp = setup$sp,
  H = setup$H
)
assert_equal_num(handle$penalty, P, "full penalty should match assembly")
assert_equal_num(handle$penalty_null, crossprod(handle$Z, P %*% handle$Z),
                 "null penalty should match transformed penalty")
assert_equal_num(handle$XtX_null, crossprod(handle$X_null),
                 "XtX_null should match crossproduct")
assert_true(identical(handle$diagnostics$penalty_count, length(setup$S)),
            "diagnostics should record penalty count")
assert_true(identical(handle$diagnostics$coefficient_dim, ncol(setup$X)),
            "diagnostics should record coefficient dimension")
assert_true(identical(handle$diagnostics$null_dim, ncol(handle$Z)),
            "diagnostics should record null dimension")
assert_true("constraint_rank" %in% names(handle$diagnostics),
            "diagnostics should include constraint rank")

cat("PASS mgcv extract GPU setup handle\n")
