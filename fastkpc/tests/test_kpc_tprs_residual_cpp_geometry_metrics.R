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
  cat("SKIP kpcTprsResidualCPP geometry metrics: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

set.seed(62305)
n <- 70L
s1 <- stats::runif(n, -2, 2)
s2 <- stats::runif(n, -1, 1)
y <- sin(s1) + 0.2 * s2 + stats::rnorm(n, sd = 0.05)

metrics <- fastkpc_kpc_tprs_geometry_parity(
  y = y,
  S = cbind(s1 = s1, s2 = s2)
)
assert_has_names(metrics,
                 c("null_space_rank_match", "effective_rank_match",
                   "constraint_rank_candidate", "constraint_rank_oracle",
                   "projector_distance", "penalty_spectrum_distance",
                   "transform_metrics"),
                 "geometry metrics")
assert_true(isTRUE(metrics$null_space_rank_match),
            "null-space rank should match frozen contract")
assert_true(isTRUE(metrics$effective_rank_match),
            "effective rank should match frozen contract")
assert_true(is.finite(metrics$projector_distance),
            "projector distance should be finite")
assert_true(is.finite(metrics$penalty_spectrum_distance),
            "penalty spectrum distance should be finite")
assert_true(is.data.frame(metrics$transform_metrics),
            "transform metrics should be a data frame")
assert_true(all(c("transform", "candidate_projector_distance",
                  "candidate_penalty_spectrum_distance") %in%
                  names(metrics$transform_metrics)),
            "transform metric columns")
assert_true(all(c("translation", "scale", "rotation",
                  "duplicate_rows", "row_permutation") %in%
                  metrics$transform_metrics$transform),
            "expected transform rows")

cat("PASS kpcTprsResidualCPP geometry metrics\n")
