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
  cat("SKIP kpcTprsResidualCPP shadow campaign Gate D: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

set.seed(62315)
n <- 80L
z <- stats::runif(n, -2, 2)
x <- sin(z) + stats::rnorm(n, sd = 0.04)
y <- cos(z) + stats::rnorm(n, sd = 0.04)
w <- 0.5 * x - 0.25 * y + stats::rnorm(n, sd = 0.08)
data <- cbind(x = x, y = y, z = z, w = w)

campaign <- fastkpc_kpc_tprs_shadow_campaign(
  data = data,
  alpha = 0.05,
  max_conditioning_size = 1L
)

assert_has_names(
  campaign,
  c("backend_family", "mode", "authoritative", "oracle", "candidate",
    "trace", "agreement", "diagnostics"),
  "shadow campaign"
)
assert_true(identical(campaign$backend_family, "kpcTprsResidualCPP"),
            "backend family")
assert_true(identical(campaign$mode, "shadow-graph-campaign"),
            "campaign mode")
assert_true(isFALSE(campaign$authoritative),
            "campaign must be non-authoritative")
assert_true(isTRUE(campaign$oracle_authoritative),
            "oracle must remain authoritative")
assert_true(identical(campaign$p_used_source, "oracle"),
            "p_used source should remain oracle")
assert_true(identical(campaign$decision_source, "oracle"),
            "decision source should remain oracle")

assert_has_names(campaign$oracle,
                 c("adjacency", "sepsets", "pMax", "n.edgetests"),
                 "oracle graph")
assert_has_names(campaign$candidate,
                 c("adjacency", "sepsets", "pMax", "n.edgetests"),
                 "candidate graph")
assert_true(is.matrix(campaign$oracle$adjacency), "oracle adjacency matrix")
assert_true(is.matrix(campaign$candidate$adjacency), "candidate adjacency matrix")
assert_true(identical(dim(campaign$oracle$adjacency), dim(campaign$candidate$adjacency)),
            "adjacency dimensions")

assert_true(is.data.frame(campaign$trace) && nrow(campaign$trace) > 0L,
            "trace rows")
assert_has_names(
  campaign$trace,
  c("canonical_test_order_id", "conditioning_level", "x", "y", "S_key",
    "oracle_p", "candidate_p", "oracle_delete", "candidate_delete",
    "decision_flip", "candidate_mode", "candidate_selected_sp",
    "candidate_score", "candidate_edf"),
  "trace schema"
)
assert_true(all(!is.na(campaign$trace$canonical_test_order_id)),
            "canonical ids are present")
assert_true(identical(campaign$trace$canonical_test_order_id,
                      seq_len(nrow(campaign$trace))),
            "canonical order should be replay order")
assert_true(any(nzchar(campaign$trace$S_key)),
            "campaign should include conditional tests")
conditional <- campaign$trace[nzchar(campaign$trace$S_key), , drop = FALSE]
assert_true(nrow(conditional) > 0L, "conditional trace rows")
assert_true(all(conditional$candidate_mode == "continuous-gcv-candidate-shadow"),
            "conditional candidate rows should use continuous GCV")

assert_has_names(
  campaign$agreement,
  c("adjacency_identical", "skeleton_shd", "sepset_mismatch_rate",
    "pmax_max_abs_diff", "n_edgetests_identical", "decision_flip_count",
    "max_log_p_abs_diff"),
  "agreement summary"
)
assert_true(isTRUE(campaign$agreement$adjacency_identical),
            "shadow candidate adjacency should match oracle")
assert_true(campaign$agreement$skeleton_shd == 0L,
            "shadow candidate skeleton SHD")
assert_true(campaign$agreement$sepset_mismatch_rate == 0,
            "shadow candidate sepset mismatch")
assert_true(campaign$agreement$pmax_max_abs_diff < 1e-4,
            "shadow candidate pMax drift")
assert_true(isTRUE(campaign$agreement$n_edgetests_identical),
            "shadow candidate n.edgetests")
assert_true(campaign$agreement$decision_flip_count == 0L,
            "shadow candidate decision flips")

cat("PASS kpcTprsResidualCPP shadow campaign Gate D\n")
