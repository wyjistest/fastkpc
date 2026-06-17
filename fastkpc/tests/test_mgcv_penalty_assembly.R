source("fastkpc/R/mgcv_compat_contract.R")
source("fastkpc/R/mgcv_extract_oracle.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_equal_num <- function(actual, expected, message, tol = 1e-12) {
  if (max(abs(actual - expected)) > tol) {
    fail(paste0(message, ": max diff ", max(abs(actual - expected))))
  }
}

S <- list(
  matrix(c(2, 0, 0, 3), 2, 2),
  matrix(5, 1, 1)
)
H <- diag(c(0.1, 0.2, 0.3, 0.4))
P <- fastkpc_assemble_penalty(
  p = 4L,
  S = S,
  off = c(2L, 4L),
  sp = c(10, 2),
  H = H
)

expected <- H
expected[2:3, 2:3] <- expected[2:3, 2:3] + 10 * S[[1]]
expected[4, 4] <- expected[4, 4] + 2 * S[[2]][1, 1]
assert_equal_num(P, expected, "assembled penalty")

bad_length <- tryCatch(
  fastkpc_assemble_penalty(p = 4L, S = S, off = c(2L), sp = c(10, 2)),
  error = function(e) e
)
assert_true(inherits(bad_length, "error"), "off length mismatch must fail")
assert_true(grepl("length\\(S\\).*length\\(off\\).*length\\(sp\\)",
                  conditionMessage(bad_length)),
            "length mismatch message must mention S/off/sp")

bad_bounds <- tryCatch(
  fastkpc_assemble_penalty(p = 3L, S = S, off = c(2L, 4L), sp = c(10, 2)),
  error = function(e) e
)
assert_true(inherits(bad_bounds, "error"), "out-of-bounds penalty must fail")
assert_true(grepl("outside coefficient dimension", conditionMessage(bad_bounds)),
            "bounds error must mention coefficient dimension")

bad_square <- tryCatch(
  fastkpc_assemble_penalty(p = 4L, S = list(matrix(1, 2, 1)), off = 1L, sp = 1),
  error = function(e) e
)
assert_true(inherits(bad_square, "error"), "non-square penalty must fail")
assert_true(grepl("square", conditionMessage(bad_square)),
            "non-square error must mention square")

cat("PASS mgcv penalty assembly\n")
