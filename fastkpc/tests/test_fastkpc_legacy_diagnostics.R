source("fastkpc/R/validation_campaign.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

campaign <- run_fastkpc_validation_campaign(
  seeds = c(41),
  n_values = c(70),
  scenarios = c("chain"),
  engines = c("cpu"),
  residual_backends = c("fastSpline"),
  legacy = TRUE
)

required <- c("scenario", "seed", "n", "available", "reason_if_unavailable",
              "native_engine", "native_residual_backend", "pdag_exact",
              "directed_added", "directed_removed", "undirected_added",
              "undirected_removed", "max_abs_pdag_diff", "status")
assert_true(all(required %in% names(campaign$legacy)),
            "legacy table should have required columns")

if (!all(campaign$legacy$available)) {
  assert_true(all(nzchar(campaign$legacy$reason_if_unavailable)),
              "unavailable legacy rows should have reasons")
  assert_true(any(grepl("pcalg|graph", campaign$legacy$reason_if_unavailable)),
              "missing package reason should mention pcalg or graph")
} else {
  assert_true(all(campaign$legacy$status == "ok"), "available legacy rows should be ok")
  assert_true(all(is.finite(campaign$legacy$max_abs_pdag_diff)),
              "available legacy rows should have finite diff")
}

disabled <- run_fastkpc_validation_campaign(
  seeds = c(41),
  n_values = c(70),
  scenarios = c("chain"),
  engines = c("cpu"),
  residual_backends = c("fastSpline"),
  legacy = FALSE
)
assert_true(all(disabled$legacy$available == FALSE), "legacy disabled should be unavailable")
assert_true(all(disabled$legacy$reason_if_unavailable == "legacy disabled"),
            "legacy disabled reason should be explicit")

cat("test_fastkpc_legacy_diagnostics.R: PASS\n")
