source("fastkpc/R/kpc_tprs_residual_cpp.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP kpcTprsResidualCPP GCV local basin: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

make_scenario <- function(seed, n) {
  set.seed(seed)
  s <- stats::runif(n, -2, 2)
  z <- s + stats::rnorm(n, sd = 0.02)
  q <- 0.5 * sin(s) - 0.25 * cos(s) + stats::rnorm(n, sd = 0.08)
  cbind(
    x = sin(s) + stats::rnorm(n, sd = 0.04),
    y = cos(s) + stats::rnorm(n, sd = 0.04),
    z = z,
    q = q
  )
}

data <- make_scenario(62331L, 72L)
S <- data[, 1L, drop = FALSE]

oracle_x <- fastkpc_kpc_tprs_mgcv_gcv_oracle_fit(data[, 2L], S)
oracle_y <- fastkpc_kpc_tprs_mgcv_gcv_oracle_fit(data[, 4L], S)
candidate_x <- fastkpc_kpc_tprs_gcv_candidate(data[, 2L], S)
candidate_y <- fastkpc_kpc_tprs_gcv_candidate(data[, 4L], S)

oracle_p <- dcov_gamma_exact(oracle_x$residuals, oracle_y$residuals)$p.value
candidate_p <- dcov_gamma_exact(candidate_x$residuals,
                                candidate_y$residuals)$p.value

assert_true(abs(candidate_p - oracle_p) < 1e-5,
            paste("candidate p-value should stay in the mgcv-compatible basin;",
                  "oracle", oracle_p, "candidate", candidate_p))
assert_true(fastkpc_kpc_tprs_rel_l2(candidate_y$residuals,
                                    oracle_y$residuals) < 1e-4,
            "candidate residuals should stay near the mgcv-magic trajectory")
assert_true(abs(log(candidate_y$selected_sp / 0.031655121)) < 2e-3,
            "candidate should select the mgcv-compatible local basin")

global_y <- fastkpc_kpc_tprs_gcv_candidate(
  data[, 4L], S, selection = "global-grid")
assert_true(global_y$score < candidate_y$score,
            "global-grid diagnostic should still expose the lower GCV basin")
assert_true(global_y$selected_sp < candidate_y$selected_sp / 10,
            "global-grid diagnostic should select the undersmoothed basin")

cat("PASS kpcTprsResidualCPP GCV local basin\n")
