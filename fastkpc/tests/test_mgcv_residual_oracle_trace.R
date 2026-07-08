source("fastkpc/R/mgcv_residual_oracle_trace.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

missing <- c("mgcv", "RSpectra")[
  !vapply(c("mgcv", "RSpectra"), requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP mgcv residual oracle trace: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

pmax <- matrix(0, nrow = 7L, ncol = 7L)
pmax[1L, 2L] <- pmax[2L, 1L] <- 0.1004
pmax[1L, 6L] <- pmax[6L, 1L] <- 0.10003
pmax[2L, 7L] <- pmax[7L, 2L] <- 0.1008
fake_result <- list(skeleton = list(
  pMax = pmax,
  per.level.log = list(
    list(),
    list(list(x = 1L, y = 2L, S_xy = 3L, S_yx = NULL)),
    list(list(x = 1L, y = 6L, S_xy = c(3L, 5L), S_yx = NULL)),
    list(list(x = 2L, y = 7L, S_xy = c(3L, 4L, 5L), S_yx = NULL))
  )
))
selected <- fastkpc_mgcv_oracle_cases_from_skeleton_result(
  fake_result, alpha = 0.1
)
assert_true(nrow(selected) >= 3L,
            "skeleton selector should emit representative oracle cases")
assert_true(all(c("source", "source_level", "source_pmax",
                  "source_distance_to_alpha") %in% names(selected)),
            "skeleton selector should preserve source diagnostics")
assert_true(any(selected$S == "3"),
            "skeleton selector should include |S|=1 deletion")
assert_true(any(selected$S == "3,5"),
            "skeleton selector should include hot level-2 deletion")
assert_true(any(selected$S == "3,4,5"),
            "skeleton selector should include late sparse deletion")
assert_true(any(grepl("near-alpha", selected$role, fixed = TRUE)),
            "skeleton selector should label near-alpha cases")

set.seed(83531)
n <- 72L
z1 <- stats::runif(n, -2, 2)
z2 <- stats::rnorm(n)
z3 <- stats::runif(n, -1, 1)
z4 <- stats::rnorm(n)
data <- cbind(
  x1 = sin(z1) + stats::rnorm(n, sd = 0.06),
  x2 = cos(z1) + 0.2 * z2 + stats::rnorm(n, sd = 0.06),
  x3 = z1 + stats::rnorm(n, sd = 0.04),
  x4 = z2 + 0.1 * z1,
  x5 = z3 + stats::rnorm(n, sd = 0.08),
  x6 = z1 * z3 + stats::rnorm(n, sd = 0.12),
  x7 = z4 + stats::rnorm(n, sd = 0.07)
)

cases <- data.frame(
  case_id = c("unit_s1", "hot_level_2", "late_sparse_gt2"),
  x = c(1L, 1L, 2L),
  y = c(2L, 6L, 7L),
  S = c("3", "3,5", "3,4,5"),
  role = c("|S|=1", "hot level-2", "late sparse |S|>2"),
  stringsAsFactors = FALSE
)

out_dir <- tempfile("fastkpc-mgcv-residual-oracle-")
artifact <- fastkpc_run_mgcv_residual_oracle_trace(
  data = data,
  cases = cases,
  alpha = 0.1,
  output_dir = out_dir
)

assert_true(file.exists(file.path(out_dir, "summary.csv")),
            "summary.csv should be written")
assert_true(file.exists(file.path(out_dir, "cases.csv")),
            "cases.csv should be written")
assert_true(file.exists(file.path(out_dir, "residuals.rds")),
            "residuals.rds should be written")
assert_true(file.exists(file.path(out_dir, "summary.md")),
            "summary.md should be written")

required_case_fields <- c(
  "case_id", "role", "target_x", "target_y", "S_key", "S_size",
  "formula_route", "formula_x", "formula_y", "regrxons_parameters",
  "runtime_ms", "dcov_p_value", "decision_at_alpha",
  "dcov_alpha", "dcov_log_alpha_distance",
  "conditioning_rank", "conditioning_rank_deficient",
  "conditioning_condition_kappa", "near_constant_column_count",
  "target_near_constant_count",
  "mgcv_family_x", "mgcv_family_y", "mgcv_link_x", "mgcv_link_y",
  "mgcv_converged_x", "mgcv_converged_y",
  "smooth_count_x", "smooth_count_y",
  "smooth_labels_x", "smooth_labels_y",
  "lpmatrix_ncol_x", "lpmatrix_ncol_y",
  "lpmatrix_rank_x", "lpmatrix_rank_y",
  "residual_x_hash", "residual_y_hash", "residual_length",
  "source", "source_level", "source_pmax", "source_distance_to_alpha",
  "mgcv_version", "R_version"
)
missing_case_fields <- setdiff(required_case_fields, names(artifact$cases))
assert_true(length(missing_case_fields) == 0L,
            paste("oracle cases missing", missing_case_fields[[1L]]))

assert_true(nrow(artifact$cases) == nrow(cases),
            "oracle trace should preserve requested case count")
assert_true(all(c(1L, 2L, 3L) %in% artifact$cases$S_size),
            "oracle trace should cover |S| 1, 2, and >2")
assert_true(identical(
  artifact$cases$formula_route,
  c("full_smooth", "full_smooth", "additive_smooth")
), "oracle trace should record the legacy formula route")
assert_true(all(is.finite(artifact$cases$dcov_p_value)),
            "oracle trace should record finite downstream dCov p-values")
assert_true(all(artifact$cases$dcov_alpha == 0.1),
            "oracle trace should record the downstream alpha")
assert_true(all(is.finite(artifact$cases$dcov_log_alpha_distance)),
            "oracle trace should record finite log-alpha decision distance")
assert_true(all(artifact$cases$residual_length == nrow(data)),
            "oracle trace should store residual vectors with data row count")
assert_true(all(artifact$cases$conditioning_rank <= artifact$cases$S_size),
            "oracle trace should record conditioning rank diagnostics")
assert_true(any(artifact$cases$smooth_count_x > 0L),
            "oracle trace should record mgcv smooth metadata")
assert_true(all(artifact$cases$lpmatrix_ncol_x > 0L),
            "oracle trace should record lpmatrix metadata")

residuals <- readRDS(file.path(out_dir, "residuals.rds"))
assert_true(length(residuals) == nrow(cases),
            "residual RDS should contain one entry per case")
assert_true(all(vapply(residuals, function(entry) {
  is.numeric(entry$residual_x) &&
    is.numeric(entry$residual_y) &&
    length(entry$residual_x) == nrow(data) &&
    length(entry$residual_y) == nrow(data)
}, logical(1))), "residual RDS should contain numeric residual vectors")

summary <- utils::read.csv(file.path(out_dir, "summary.csv"),
                           stringsAsFactors = FALSE)
required_summary_fields <- c(
  "rank_deficient_case_count",
  "near_constant_case_count",
  "max_conditioning_condition_kappa",
  "min_dcov_log_alpha_distance"
)
missing_summary_fields <- setdiff(required_summary_fields, names(summary))
assert_true(length(missing_summary_fields) == 0L,
            paste("oracle summary missing", missing_summary_fields[[1L]]))
assert_true(summary$case_count[[1L]] == nrow(cases),
            "summary should count oracle cases")
assert_true(summary$error_count[[1L]] == 0L,
            "summary should report no oracle errors")
assert_true(summary$min_dcov_log_alpha_distance[[1L]] >= 0,
            "summary should report non-negative log-alpha distance")

cat("PASS mgcv residual oracle trace\n")
