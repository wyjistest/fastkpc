source("fastkpc/R/native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

self <- regrvonps_native_selftest()

required <- c(
  "dependent_linear_rejects",
  "smooth_fastspline_accepts",
  "parents_in_conditioning",
  "cache_hits_repeated",
  "pvalue_count_correct",
  "unknown_backend_error",
  "empty_S_safe"
)

missing <- setdiff(required, names(self))
assert_true(length(missing) == 0L,
            paste("regrvonps native selftest missing fields:",
                  paste(missing, collapse = ", ")))

for (name in required) {
  assert_true(self[[name]], paste(name, "should be TRUE"))
}

cat("test_regrvonps_native.R: PASS\n")
