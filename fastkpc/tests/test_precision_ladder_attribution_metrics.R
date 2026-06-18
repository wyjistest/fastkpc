source("fastkpc/R/mgcv_compat_contract.R")
source("fastkpc/R/mgcv_extract_validation.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_close <- function(value, expected, tolerance, message) {
  if (!isTRUE(abs(value - expected) <= tolerance)) {
    fail(paste0(message, ": got ", value, " expected ", expected))
  }
}

metrics <- fastkpc_empty_compatibility_campaign_metrics()

required_residual <- c(
  "basis_projection_floor",
  "current_lambda_residual_rel_l2",
  "oracle_lambda_residual_rel_l2",
  "oracle_lambda_improvement",
  "current_lambda_log_p_drift",
  "oracle_lambda_log_p_drift",
  "current_lambda_decision_flip",
  "oracle_lambda_decision_flip"
)
missing_residual <- setdiff(required_residual, names(metrics$residual))
assert_true(length(missing_residual) == 0L,
            paste("missing residual attribution fields:",
                  paste(missing_residual, collapse = ", ")))

required_ci <- c(
  "kernel_bandwidth_legacy",
  "kernel_bandwidth_candidate",
  "test_stat_legacy",
  "test_stat_candidate",
  "p_frozen_config",
  "p_native_config",
  "decision_flip_frozen",
  "decision_flip_native"
)
missing_ci <- setdiff(required_ci, names(metrics$ci))
assert_true(length(missing_ci) == 0L,
            paste("missing frozen/native CI fields:",
                  paste(missing_ci, collapse = ", ")))

B <- cbind(1, seq_len(5))
representable <- as.numeric(B %*% c(2, -0.25))
floor0 <- fastkpc_basis_projection_floor(representable, B)
assert_close(floor0, 0, 1e-12,
             "projection floor should be zero for representable fitted values")

nonrepresentable <- c(1, -1, 2, -2, 3)
floor1 <- fastkpc_basis_projection_floor(nonrepresentable, B)
assert_true(is.finite(floor1) && floor1 > 0.1,
            "projection floor should be positive for nonrepresentable fits")

legacy_res <- c(1, -1, 0.5, -0.25)
current_res <- c(1.2, -0.9, 0.4, -0.2)
oracle_res <- c(1.02, -0.99, 0.5, -0.25)
gap <- fastkpc_oracle_lambda_gap(
  legacy_residual = legacy_res,
  current_residual = current_res,
  oracle_residual = oracle_res,
  current_lambda = 0.1,
  oracle_lambda = 0.03,
  p_legacy = 0.05,
  p_current = 0.10,
  p_oracle = 0.06,
  alpha = 0.05
)
assert_true(gap$oracle_lambda_residual_rel_l2 <
              gap$current_lambda_residual_rel_l2,
            "oracle residual error should be lower in the fixture")
assert_true(gap$oracle_lambda_improvement > 0,
            "oracle improvement should be positive when oracle error is lower")
assert_true(gap$current_lambda_decision_flip,
            "current lambda should flip the alpha decision in the fixture")
assert_true(gap$oracle_lambda_decision_flip,
            "oracle lambda p=0.06 should still flip legacy p=0.05 at alpha")

row <- fastkpc_make_ci_compatibility_row(
  canonical_test_order_id = 9L,
  x = 1L,
  y = 3L,
  S = 2L,
  conditioning_level = 1L,
  p_legacy = 0.04,
  p_backend = 0.08,
  alpha = 0.05,
  backend_used = "fastSplineCUDA",
  fallback_triggered = TRUE,
  verifier_backend = "mgcvExtractGPUFixedSP",
  kernel_bandwidth_legacy = 1.5,
  kernel_bandwidth_candidate = 1.7,
  test_stat_legacy = 2.1,
  test_stat_candidate = 2.4,
  p_frozen_config = 0.07,
  p_native_config = 0.08
)
assert_true(row$decision_flip_frozen,
            "frozen p=0.07 should flip legacy p=0.04 at alpha")
assert_true(row$decision_flip_native,
            "native p=0.08 should flip legacy p=0.04 at alpha")
assert_close(row$kernel_bandwidth_candidate, 1.7, 1e-12,
             "candidate bandwidth should be recorded")

cap <- fastkpc_fastspline_cuda_capabilities()
assert_true(identical(cap$backend, "fastSplineCUDA"),
            "capability backend should name fastSplineCUDA")
assert_true(identical(cap$role, "frozen approximate baseline"),
            "fastSplineCUDA should be declared frozen approximate baseline")
assert_true(isTRUE(cap$supported$true_batched_cusolver),
            "fastSplineCUDA capability should record cuSOLVER batching")
assert_true(isFALSE(cap$claims$mgcv_equivalent),
            "fastSplineCUDA must not claim mgcv equivalence")

cat("PASS precision ladder attribution metrics\n")
