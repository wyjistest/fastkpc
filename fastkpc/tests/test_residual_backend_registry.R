source("fastkpc/R/native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

backends <- list_residual_backends()
assert_true(identical(backends, c("linear", "fastSpline")),
            "registered residual backends should be linear and fastSpline")

self <- fast_residual_backend_selftest()

assert_true(self$linear_matches_direct,
            "linear backend residuals should match existing direct linear residuals")
assert_true(self$fastspline_differs_from_linear,
            "fastSpline residuals should differ from linear residuals on nonlinear data")
assert_true(self$key_separates_backend,
            "residual cache key should separate linear and fastSpline backends")
assert_true(self$key_separates_fastspline_params,
            "residual cache key should separate fastSpline params")
assert_true(self$fastspline_cache_stats$enabled,
            "fastSpline cache stats should report enabled TRUE")
assert_true(self$fastspline_cache_stats$requests == 2L,
            "fastSpline cache selftest should make two requests")
assert_true(self$fastspline_cache_stats$hits == 1L,
            "fastSpline repeated request should hit cache")
assert_true(self$fastspline_cache_stats$computations < self$fastspline_cache_stats$requests,
            "fastSpline cache computations should be lower than requests")
assert_true(self$fastspline_cache_stats$backend_name == "fastSpline",
            "fastSpline cache backend name should be recorded")

unknown_error <- tryCatch({
  fast_residual_backend_unknown_selftest()
  ""
}, error = function(e) conditionMessage(e))
assert_true(grepl("Unknown residual backend", unknown_error, fixed = TRUE),
            "unknown backend should raise a clear error")

cat("test_residual_backend_registry.R: PASS\n")
