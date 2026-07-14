source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
expect_error_contains <- function(expr, text) {
  error <- tryCatch(force(expr), error = identity)
  assert_true(
    inherits(error, "error") &&
      grepl(text, conditionMessage(error), fixed = TRUE),
    paste0("expected error containing '", text, "'")
  )
  invisible(error)
}

required_helpers <- c(
  "fastkpc_full_cuda_fixed_sp_phase3a_relative_l2_diff",
  "fastkpc_full_cuda_fixed_sp_phase3a_validate_parity"
)
missing_helpers <- required_helpers[!vapply(
  required_helpers, exists, logical(1L), mode = "function", inherits = TRUE
)]
assert_true(
  length(missing_helpers) == 0L,
  paste("missing Phase 3A iteration helpers:",
        paste(missing_helpers, collapse = ", "))
)

actual <- c(1e-150, -2e-150)
reference <- c(0, 0)
numerator <- sqrt(sum((actual - reference)^2))
frozen_expected <- numerator / 1e-300
old_result <- numerator / max(
  sqrt(sum(reference^2)), .Machine$double.xmin
)
observed <- fastkpc_full_cuda_fixed_sp_phase3a_relative_l2_diff(
  actual, reference
)
assert_true(
  isTRUE(all.equal(observed, frozen_expected, tolerance = 1e-15)),
  "relative-L2 uses the frozen 1e-300 denominator floor"
)
assert_true(
  !isTRUE(all.equal(observed, old_result, tolerance = 1e-15)),
  "relative-L2 must not use .Machine$double.xmin"
)

safe_count <- 172L
stable_count <- 98L
good_records <- data.frame(
  planned_route = c(
    rep("CHOLESKY_BATCHED", safe_count),
    rep("AUGMENTED_QR", stable_count)
  ),
  solver_status = c(
    rep("OK_CHOLESKY_SINGLE", safe_count),
    rep("ERR_STABLE_PATH_NOT_IMPLEMENTED", stable_count)
  ),
  residual_max_abs_diff = c(rep(1e-12, safe_count), rep(NA_real_, stable_count)),
  residual_relative_l2_diff = c(
    rep(1e-12, safe_count), rep(NA_real_, stable_count)
  ),
  fitted_max_abs_diff = c(rep(1e-12, safe_count), rep(NA_real_, stable_count)),
  fitted_relative_l2_diff = c(
    rep(1e-12, safe_count), rep(NA_real_, stable_count)
  ),
  stringsAsFactors = FALSE
)

benchmark_call_count <- 0L
validate_then_benchmark <- function(records) {
  parity <- fastkpc_full_cuda_fixed_sp_phase3a_validate_parity(records)
  benchmark_call_count <<- benchmark_call_count + 1L
  parity
}

bad_status <- good_records
bad_status$solver_status[[safe_count + 1L]] <- "OK_QR_SINGLE"
expect_error_contains(
  validate_then_benchmark(bad_status),
  "Phase 3A iteration status/numerical parity failed"
)
assert_true(
  benchmark_call_count == 0L,
  "bad stable status fails before benchmark work"
)

bad_error <- good_records
bad_error$residual_relative_l2_diff[[1L]] <- 1e-7
expect_error_contains(
  validate_then_benchmark(bad_error),
  "Phase 3A iteration status/numerical parity failed"
)
assert_true(
  benchmark_call_count == 0L,
  "bad numerical parity fails before benchmark work"
)

parity <- validate_then_benchmark(good_records)
assert_true(
  benchmark_call_count == 1L &&
    identical(names(parity), c("safe", "stable")) &&
    sum(parity$safe) == safe_count && sum(parity$stable) == stable_count,
  "valid parity reaches benchmark work with frozen target partitions"
)

runner_body <- paste(
  deparse(body(fastkpc_run_full_cuda_fixed_sp_phase3a_iteration)),
  collapse = "\n"
)
parity_position <- regexpr(
  "fastkpc_full_cuda_fixed_sp_phase3a_validate_parity",
  runner_body, fixed = TRUE
)[[1L]]
benchmark_setup_position <- regexpr(
  "safe_descriptors <- list()",
  runner_body, fixed = TRUE
)[[1L]]
warmup_position <- regexpr(
  "persistent_warmup <- run_persistent_corpus",
  runner_body, fixed = TRUE
)[[1L]]
assert_true(
  parity_position > 0L && benchmark_setup_position > parity_position &&
    warmup_position > benchmark_setup_position,
  "production parity validation precedes benchmark setup and warmup"
)

cat("PASS Phase 3A iteration helper contracts\n")
