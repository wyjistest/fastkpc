source("fastkpc/R/mgcv_compat_contract.R")
source("fastkpc/R/mgcv_extract_validation.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

out_dir <- tempfile("fastkpc-precision-ladder-")
campaign <- fastkpc_run_precision_ladder_attribution_campaign(
  output_dir = out_dir,
  seeds = c(5L, 6L),
  n_values = c(40L),
  alpha = 0.05
)

expected_files <- c(
  "mgcv_residual_compatibility.csv",
  "mgcv_ci_compatibility.csv",
  "mgcv_graph_compatibility.csv",
  "fastspline_cuda_capabilities.csv",
  "precision_ladder_attribution_summary.csv"
)
missing_files <- expected_files[!file.exists(file.path(out_dir, expected_files))]
assert_true(length(missing_files) == 0L,
            paste("missing campaign files:", paste(missing_files, collapse = ", ")))

assert_true(nrow(campaign$residual) == 2L,
            "campaign should produce one residual row per seed/n scenario")
assert_true(nrow(campaign$ci) == 2L,
            "campaign should produce one CI row per seed/n scenario")
assert_true(nrow(campaign$graph) == 2L,
            "campaign should produce one graph row per seed/n scenario")

assert_true(all(is.finite(campaign$residual$basis_projection_floor)),
            "basis projection floor should be finite")
assert_true(all(campaign$residual$oracle_lambda_residual_rel_l2 <=
                  campaign$residual$current_lambda_residual_rel_l2),
            "oracle residual error should not exceed current residual error")
assert_true(all(campaign$ci$decision_flip_native == campaign$ci$decision_flip),
            "native decision flip should mirror backend p decision in this campaign")
assert_true(any(campaign$summary$mean_oracle_lambda_improvement > 0),
            "summary should report positive oracle-lambda improvement")

cap <- utils::read.csv(file.path(out_dir, "fastspline_cuda_capabilities.csv"),
                       stringsAsFactors = FALSE)
assert_true(identical(cap$backend[1], "fastSplineCUDA"),
            "capability CSV should identify fastSplineCUDA")
assert_true(identical(cap$mgcv_equivalent[1], FALSE),
            "capability CSV should record non-equivalence to mgcv")

cat("PASS precision ladder attribution campaign\n")
