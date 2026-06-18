source("fastkpc/R/failure_injection_scenarios.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

cases <- fastkpc_precision_failure_cases()
required_cases <- c("unsupported_mgcv_version", "unsupported_R_version",
                    "cuda_unavailable", "setup_fingerprint_mismatch",
                    "nan_primary_p", "verifier_failure")
assert_true(all(required_cases %in% names(cases)), "missing failure cases")

results <- fastkpc_run_precision_failure_injection(cases)
assert_true(all(results$compatibility_action %in% c("fallback", "warn-and-run")),
            "failure cases must not silently run unsupported GPU")
assert_true(all(nzchar(results$fallback_reason)),
            "failure cases must expose fallback reason")
assert_true(all(results$canonical_replay_preserved),
            "failure cases must preserve canonical replay")
assert_true(all(c("backend_requested", "backend_used", "p_source_used") %in%
                  names(results)),
            "failure diagnostics must expose backend and p-value source")

cat("PASS precision failure injection\n")
