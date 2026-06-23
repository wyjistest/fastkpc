source("fastkpc/R/kpc_tprs_residual_cpp.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_equal <- function(actual, expected, message) {
  if (!identical(actual, expected)) {
    fail(paste0(message, ": expected ", paste(expected, collapse = ","),
                " got ", paste(actual, collapse = ",")))
  }
}

set.seed(62303)
n <- 72L
S1 <- matrix(stats::runif(n, -2, 2), ncol = 1)
S2 <- cbind(
  s1 = stats::runif(n, -2, 2),
  s2 = stats::runif(n, -1, 1)
)

setup1 <- kpc_tprs_residual_cpp_setup(S1)
setup2 <- kpc_tprs_residual_cpp_setup(S2)

assert_equal(setup1$basis_rank, 10L, "1D default basis dimension")
assert_equal(setup1$null_space_rank, 2L, "1D null-space rank")
assert_equal(setup1$penalized_rank, 8L, "1D penalized rank")
assert_equal(setup1$k_def, 8L, "1D mgcv default k.def")
assert_equal(ncol(setup1$raw$radial), 8L, "1D radial basis columns")
assert_equal(ncol(setup1$absorbed$X), setup1$effective_rank,
             "1D absorbed basis width")

assert_equal(setup2$basis_rank, 30L, "2D default basis dimension")
assert_equal(setup2$null_space_rank, 3L, "2D null-space rank")
assert_equal(setup2$penalized_rank, 27L, "2D penalized rank")
assert_equal(setup2$k_def, 27L, "2D mgcv default k.def")
assert_equal(ncol(setup2$raw$radial), 27L, "2D radial basis columns")
assert_equal(ncol(setup2$absorbed$X), setup2$effective_rank,
             "2D absorbed basis width")

assert_true(is.matrix(setup2$raw$shifted_covariates),
            "raw shifted covariates should be exposed")
assert_true(is.matrix(setup2$raw$unique_locations),
            "raw unique locations should be exposed")
assert_true(is.matrix(setup2$raw$radial_kernel_block),
            "raw radial kernel block should be exposed")
assert_true(is.matrix(setup2$raw$polynomial),
            "raw polynomial block should be exposed")
assert_true(is.matrix(setup2$raw$penalty),
            "raw penalty should be exposed")
assert_true(is.matrix(setup2$raw$constraint),
            "raw constraint should be exposed")
assert_true(is.matrix(setup2$absorbed$Z),
            "constraint null-space transform should be exposed")
assert_true(is.matrix(setup2$absorbed$penalty),
            "absorbed penalty should be exposed")
assert_equal(nrow(setup2$absorbed$penalty), ncol(setup2$absorbed$X),
             "absorbed penalty row count")
assert_equal(ncol(setup2$absorbed$penalty), ncol(setup2$absorbed$X),
             "absorbed penalty column count")

too_many_unique <- cbind(seq_len(2001L), seq_len(2001L) / 7)
err <- tryCatch(kpc_tprs_residual_cpp_setup(too_many_unique),
                error = function(e) e)
assert_true(inherits(err, "error"), "unique locations > 2000 should fail closed")
assert_true(grepl("unique conditioning locations <= 2000",
                  conditionMessage(err), fixed = TRUE),
            "unique-location error should name frozen contract")

cat("PASS kpcTprsResidualCPP default contract\n")
