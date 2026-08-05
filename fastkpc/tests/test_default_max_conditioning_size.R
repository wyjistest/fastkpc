source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_identical <- function(actual, expected, message) {
  if (!identical(actual, expected)) fail(message)
}
assert_error <- function(expr, pattern, message) {
  value <- tryCatch({
    force(expr)
    NULL
  }, error = identity)
  if (!inherits(value, "error") ||
      !grepl(pattern, conditionMessage(value), fixed = TRUE)) {
    fail(message)
  }
}

assert_identical(
  fastkpc_resolve_max_conditioning_size(Inf, 48L), 46L,
  "Inf should resolve to p-2"
)
assert_identical(
  fastkpc_resolve_max_conditioning_size(100, 48L), 46L,
  "finite limits above p-2 should resolve to p-2"
)
assert_identical(
  fastkpc_resolve_max_conditioning_size(7.9, 48L), 7L,
  "finite non-integer limits should match pcalg level reachability"
)
assert_identical(
  fastkpc_resolve_max_conditioning_size(0, 2L), 0L,
  "p=2 should have a zero conditioning limit"
)
assert_identical(
  eval(formals(precision_run_skeleton_full_cuda_native)$max_conditioning_size),
  Inf,
  "the native full-CUDA facade should default to kpcalg m.max=Inf semantics"
)
assert_error(
  fastkpc_resolve_max_conditioning_size(-1, 48L),
  "max_conditioning_size must be", "negative limits must fail closed"
)
assert_error(
  fastkpc_resolve_max_conditioning_size(NA_real_, 48L),
  "max_conditioning_size must be", "NA limits must fail closed"
)
assert_error(
  fastkpc_resolve_max_conditioning_size(c(1, 2), 48L),
  "max_conditioning_size must be", "vector limits must fail closed"
)

cat("PASS default max conditioning size resolution\n")
