source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

set.seed(20260729L)
n <- 80L
residuals <- cbind(
  rnorm(n),
  sin(seq_len(n) / 7) + rnorm(n, sd = 0.2),
  cos(seq_len(n) / 11) + rnorm(n, sd = 0.1),
  seq_len(n) / n + rnorm(n, sd = 0.3),
  rnorm(n)^3
)
storage.mode(residuals) <- "double"
left <- c(1L, 1L, 2L, 3L, 5L, 2L)
right <- c(2L, 3L, 3L, 4L, 1L, 5L)
num_col <- 8L

old_mode <- Sys.getenv(
  "FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK", unset = NA_character_
)
on.exit({
  if (is.na(old_mode)) {
    Sys.unsetenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK")
  } else {
    Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = old_mode)
  }
}, add = TRUE)
Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = "spectra")

reference <- lapply(seq_along(left), function(index) {
  fastkpc_cuda_legacy_dcov_gamma_cpp_oracle(
    residuals[, left[[index]]], residuals[, right[[index]]],
    numCol = num_col, index = 1
  )
})
candidate <- legacy_dcov_gamma_cpp_component_cache_batch_native(
  residuals, left, right, numCol = num_col, index = 1
)

fields <- c("p.value", "nV2", "mean", "variance")
errors <- vapply(fields, function(field) {
  expected <- vapply(reference, `[[`, numeric(1L), field)
  max(abs(as.numeric(candidate[[field]]) - expected))
}, numeric(1L))
reference_decision <- vapply(
  reference, function(value) value$p.value > 0.1, logical(1L)
)
candidate_decision <- candidate$p.value > 0.1
diagnostics <- candidate$diagnostics
used_components <- length(unique(c(left, right)))

assert_true(
  max(errors) <= 1e-12 &&
    identical(candidate_decision, reference_decision),
  "legacy dCov component-cache numerical semantics drifted"
)
assert_true(
  diagnostics$n == n && diagnostics$pair_count == length(left) &&
    diagnostics$component_count == used_components &&
    diagnostics$component_request_count == 2L * length(left) &&
    diagnostics$component_cache_miss_count == used_components &&
    diagnostics$component_cache_hit_count ==
      2L * length(left) - used_components &&
    identical(diagnostics$lowrank_mode, "spectra") &&
    diagnostics$lowrank_spectra_count == used_components &&
    diagnostics$lowrank_spectra_converged_count == used_components &&
    diagnostics$lowrank_spectra_failed_count == 0L &&
    diagnostics$lowrank_spectra_fallback_full_eig_count == 0L,
  "legacy dCov component-cache accounting is malformed"
)

print(data.frame(field = names(errors), max_absolute_error = errors),
      row.names = FALSE, digits = 17)
cat("PASS legacy dCov gamma C++ component cache parity\n")
