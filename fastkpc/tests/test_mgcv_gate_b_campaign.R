source("fastkpc/R/mgcv_gate_b_campaign.R")

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP mgcv not installed\n")
  quit(save = "no", status = 0)
}

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

out_dir <- tempfile("fastkpc-gate-b-campaign-")
dir.create(out_dir, recursive = TRUE)

campaign <- fastkpc_run_mgcv_gate_b_campaign(
  seeds = c(11, 12),
  n_values = c(80),
  sp_grid = c("selected", "small", "medium", "large"),
  output_dir = out_dir
)

required <- c(
  "scenario_id", "seed", "n", "S_size", "formula_class", "sp_source",
  "sp", "edf_reference", "rank_setup", "constraint_rank", "penalty_rank",
  "coef_rel_l2", "fitted_rel_l2", "residual_rel_l2",
  "max_abs_residual_diff", "condition_number_proxy",
  "pass_gate_b", "warning_message"
)
missing <- setdiff(required, names(campaign$fixed_sp))
assert_true(length(missing) == 0L,
            paste("missing campaign fields:", paste(missing, collapse = ", ")))
assert_true(nrow(campaign$fixed_sp) > 0L, "campaign should produce rows")
assert_true(all(campaign$fixed_sp$pass_gate_b), "basic campaign rows should pass Gate B")
assert_true(file.exists(file.path(out_dir, "mgcv_gate_b_fixed_sp_campaign.csv")),
            "fixed-sp campaign CSV should be written")

read_back <- utils::read.csv(file.path(out_dir, "mgcv_gate_b_fixed_sp_campaign.csv"))
assert_true(nrow(read_back) == nrow(campaign$fixed_sp),
            "CSV row count should match in-memory campaign")

cat("PASS mgcv Gate B campaign\n")
