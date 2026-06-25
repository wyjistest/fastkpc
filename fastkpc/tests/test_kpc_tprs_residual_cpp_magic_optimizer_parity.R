source("fastkpc/R/kpc_tprs_residual_cpp.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_has_names <- function(x, required, message) {
  missing <- setdiff(required, names(x))
  if (length(missing) > 0L) {
    fail(paste0(message, ": missing ", paste(missing, collapse = ", ")))
  }
}

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP kpcTprsResidualCPP magic optimizer parity: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

real_path <- "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"
if (!file.exists(real_path)) {
  cat("SKIP kpcTprsResidualCPP magic optimizer parity: real fixture unavailable\n")
  quit(save = "no", status = 0)
}

data <- readRDS(real_path)

low_basin <- fastkpc_kpc_tprs_magic_parity_diagnostics(
  y = data[, 30L],
  S = data[, 9L, drop = FALSE],
  label = "real-target30-S9"
)
high_basin <- fastkpc_kpc_tprs_magic_parity_diagnostics(
  y = data[, 34L],
  S = data[, 6L, drop = FALSE],
  label = "real-target34-S6"
)

required <- c(
  "label", "backend_family", "mode", "authoritative",
  "oracle_raw_sp", "mapped_lambda", "local_lambda", "global_lambda",
  "oracle_edf", "mapped_edf", "local_edf", "global_edf",
  "oracle_score", "mapped_score", "local_score", "global_score",
  "mapped_residual_rel_l2", "local_residual_rel_l2",
  "global_residual_rel_l2", "local_bracket_lower_lambda",
  "local_bracket_upper_lambda", "local_pre_boundary_lambda",
  "local_magic_boundary_probe", "local_magic_boundary_steps",
  "local_magic_boundary_direction", "local_contains_mapped_lambda",
  "global_score_lower_than_local", "basin_label"
)
assert_has_names(low_basin, required, "low-basin diagnostics")
assert_has_names(high_basin, required, "high-basin diagnostics")

assert_true(identical(low_basin$backend_family, "kpcTprsResidualCPP"),
            "low-basin backend family")
assert_true(identical(low_basin$mode, "magic-optimizer-parity-diagnostics"),
            "low-basin diagnostics mode")
assert_true(isFALSE(low_basin$authoritative),
            "magic diagnostics must be non-authoritative")

assert_true(low_basin$mapped_residual_rel_l2 < 1e-10,
            "mapped lambda should reproduce mgcv residuals for target 30")
assert_true(low_basin$local_residual_rel_l2 < 1e-6,
            "magic-compatible local optimizer should fix real target 30 drift")
assert_true(!isTRUE(low_basin$local_contains_mapped_lambda),
            "target 30 initial local bracket should miss mgcv mapped low-lambda basin")
assert_true(isTRUE(low_basin$local_magic_boundary_probe),
            "target 30 should require the mgcv magic boundary probe")
assert_true(low_basin$local_magic_boundary_direction < 0L,
            "target 30 boundary probe should move toward lower lambda")
assert_true(low_basin$global_score_lower_than_local,
            "target 30 global scan should find the lower-score basin")
assert_true(identical(low_basin$basin_label, "mgcv-local-basin-nonglobal"),
            "target 30 basin classification")

assert_true(high_basin$mapped_residual_rel_l2 < 1e-10,
            "mapped lambda should reproduce mgcv residuals for target 34")
assert_true(high_basin$local_residual_rel_l2 < 1e-6,
            "current local optimizer should stay in mgcv high-lambda basin")
assert_true(!isTRUE(high_basin$local_magic_boundary_probe),
            "target 34 should not be pulled into the global low-lambda basin")
assert_true(high_basin$global_score_lower_than_local,
            "target 34 global scan should expose a lower-score non-mgcv basin")
assert_true(identical(high_basin$basin_label, "mgcv-local-basin-nonglobal"),
            "target 34 basin classification")

cat("PASS kpcTprsResidualCPP magic optimizer parity diagnostics\n")
