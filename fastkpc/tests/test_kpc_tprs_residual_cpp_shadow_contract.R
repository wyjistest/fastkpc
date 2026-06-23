source("fastkpc/R/kpc_tprs_residual_cpp.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_equal <- function(actual, expected, message) {
  if (!identical(actual, expected)) {
    fail(paste0(message, ": expected ", paste(expected, collapse = ","),
                " got ", paste(actual, collapse = ",")))
  }
}

set.seed(62301)
n <- 80L
s1 <- stats::runif(n, -2, 2)
s2 <- stats::runif(n, -1, 1)
y <- sin(s1) + 0.25 * s2 + stats::rnorm(n, sd = 0.05)
S <- cbind(s1 = s1, s2 = s2)

candidate <- fastkpc_kpc_tprs_residual_cpp_candidate(y, S)
assert_equal(candidate$backend_family, "kpcTprsResidualCPP",
             "candidate backend family")
assert_equal(candidate$mode, "shadow-candidate-setup-only",
             "candidate mode")
assert_true(isFALSE(candidate$authoritative),
            "candidate must not be authoritative")
assert_equal(candidate$family, "gaussian_identity", "candidate family")
assert_equal(candidate$smooth_class, "tp", "candidate smooth class")
assert_equal(candidate$smooth_geometry, "joint-isotropic",
             "candidate must declare joint isotropic geometry")
assert_equal(candidate$conditioning_size, 2L, "conditioning size")
assert_true(is.matrix(candidate$X), "candidate must expose a setup matrix")
assert_equal(nrow(candidate$X), n, "candidate X rows")
assert_true(ncol(candidate$X) > candidate$null_space_rank,
            "candidate basis must include penalized columns")
assert_true(is.matrix(candidate$penalty), "candidate must expose penalty")
assert_equal(dim(candidate$penalty), rep(ncol(candidate$X), 2L),
             "candidate penalty dimensions")
assert_true(is.matrix(candidate$constraint),
            "candidate must expose centering constraint")
assert_equal(ncol(candidate$constraint), ncol(candidate$X),
             "constraint width")
assert_true(nrow(candidate$constraint) >= 1L,
            "candidate must include identifiability constraint")
assert_true(length(candidate$residuals) == 0L,
            "setup-only candidate must not manufacture residuals")
assert_true(is.na(candidate$selected_sp),
            "setup-only candidate must not manufacture selected sp")
assert_true(is.na(candidate$score),
            "setup-only candidate must not manufacture score")
assert_true(nchar(candidate$setup_fingerprint) > 0L,
            "candidate setup fingerprint required")

shadow <- fastkpc_kpc_tprs_residual_cpp_shadow(
  y = y,
  S = S,
  oracle = function(y, S) {
    list(
      residuals = y - mean(y),
      fitted = rep(mean(y), length(y)),
      coefficients = mean(y),
      edf = 1,
      selected_sp = 1,
      score = 0,
      basis_rank = 1L,
      null_space_rank = 1L,
      setup_fingerprint = "oracle-fixture",
      diagnostics = list(source = "oracle")
    )
  }
)
assert_true(isTRUE(shadow$oracle_authoritative),
            "shadow wrapper must keep oracle authoritative")
assert_equal(shadow$p_used_source, "oracle", "p used source")
assert_equal(shadow$decision_source, "oracle", "decision source")
assert_equal(shadow$oracle$setup_fingerprint, "oracle-fixture",
             "oracle result preserved")
assert_equal(shadow$candidate$backend_family, "kpcTprsResidualCPP",
             "candidate result included")

nonfinite <- tryCatch(
  fastkpc_kpc_tprs_residual_cpp_candidate(c(y[-1], Inf), S),
  error = function(e) e
)
assert_true(inherits(nonfinite, "error"), "non-finite y must fail closed")
assert_true(grepl("finite numeric", conditionMessage(nonfinite), fixed = TRUE),
            "non-finite y error should name finite numeric contract")

too_large <- tryCatch(
  fastkpc_kpc_tprs_residual_cpp_candidate(y, cbind(S, s3 = stats::rnorm(n))),
  error = function(e) e
)
assert_true(inherits(too_large, "error"), "|S| > 2 must fail closed")
assert_true(grepl("|S| = 1 or 2", conditionMessage(too_large), fixed = TRUE),
            "unsupported size error should name contract")

cat("PASS kpcTprsResidualCPP shadow contract\n")
