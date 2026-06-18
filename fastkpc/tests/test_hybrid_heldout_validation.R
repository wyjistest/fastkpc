source("fastkpc/R/hybrid_heldout_validation.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

cal <- data.frame(
  tau = c(log(1.5), log(2), log(3)),
  decision_flip_rate_primary = c(0.12, 0.12, 0.12),
  decision_flip_rate_hybrid = c(0.08, 0.04, 0.04),
  skeleton_shd_primary = c(5, 5, 5),
  skeleton_shd_hybrid = c(4, 2, 2),
  sepset_mismatch_hybrid = c(0.2, 0.1, 0.1),
  wanpdag_mismatch_hybrid = c(4, 2, 2),
  verification_rate = c(0.05, 0.12, 0.25),
  runtime_ratio = c(1.1, 1.3, 1.8)
)
held <- data.frame(
  tau = log(2),
  decision_flip_rate_primary = 0.10,
  decision_flip_rate_hybrid = 0.05,
  skeleton_shd_primary = 4,
  skeleton_shd_hybrid = 2,
  sepset_mismatch_primary = 0.2,
  sepset_mismatch_hybrid = 0.1,
  wanpdag_mismatch_primary = 3,
  wanpdag_mismatch_hybrid = 1,
  verification_rate = 0.12,
  runtime_ratio = 1.35
)

result <- fastkpc_validate_hybrid_tau_heldout(
  calibration = cal,
  heldout = held,
  max_runtime_ratio = 2
)
assert_true(result$selected_tau == log(2), "held-out validation should keep log(2)")
assert_true(result$heldout_pass, "held-out graph metrics should pass")
assert_true(grepl("experimental", result$recommendation, fixed = TRUE),
            "recommendation should remain experimental")

cat("PASS hybrid heldout validation\n")
