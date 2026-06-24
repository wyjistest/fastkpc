source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/kpc_tprs_residual_cpp_qualification.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP kpcTprsResidualCPP switch qualification Gate E: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

make_scenario <- function(seed, n, scale = 1, duplicate = FALSE,
                          near_collinear = FALSE) {
  set.seed(seed)
  s <- stats::runif(n, -2, 2) * scale
  z <- s + stats::rnorm(n, sd = 0.02 * scale)
  if (isTRUE(near_collinear)) {
    q <- z + stats::rnorm(n, sd = 1e-3 * max(1, scale))
  } else {
    q <- 0.5 * sin(s / max(1, scale)) -
      0.25 * cos(s / max(1, scale)) + stats::rnorm(n, sd = 0.08)
  }
  data <- cbind(
    x = sin(s / max(1, scale)) + stats::rnorm(n, sd = 0.04),
    y = cos(s / max(1, scale)) + stats::rnorm(n, sd = 0.04),
    z = z,
    q = q
  )
  if (isTRUE(duplicate)) {
    rows <- seq_len(min(8L, nrow(data)))
    data <- rbind(data, data[rows, , drop = FALSE])
  }
  data
}

scenarios <- list(
  seed_1_n72 = make_scenario(62330, 72L),
  seed_2_scaled = make_scenario(62331, 80L, scale = 25),
  duplicates = make_scenario(62332, 64L, duplicate = TRUE),
  near_collinear = make_scenario(62333, 76L, near_collinear = TRUE),
  small_n = make_scenario(62334, 42L)
)

campaign <- fastkpc_kpc_tprs_switch_qualification_campaign(
  scenarios = scenarios,
  alpha = 0.05,
  max_conditioning_size = 2L
)

assert_true(is.data.frame(campaign$summary), "summary should be a data frame")
assert_true(nrow(campaign$summary) == length(scenarios),
            "summary should include every scenario")
assert_true(all(campaign$summary$passed),
            paste(campaign$summary$scenario[!campaign$summary$passed],
                  collapse = ", "))
assert_true(all(campaign$summary$adjacency_identical),
            "all scenario adjacencies should match mgcv reference")
assert_true(all(campaign$summary$n_edgetests_identical),
            "all n.edgetests should match mgcv reference")
assert_true(max(campaign$summary$pmax_max_abs_diff, na.rm = TRUE) <=
              fastkpc_kpc_tprs_pmax_abs_tol(),
            "pMax drift should remain small")
assert_true(any(campaign$summary$conditional_kpc_rows > 0L),
            "campaign should exercise conditional kpcTprsResidualCPP rows")
assert_true(any(campaign$summary$two_d_kpc_rows > 0L),
            "campaign should exercise |S|=2 kpcTprsResidualCPP rows")
assert_true(all(campaign$summary$mgcv_fallback_available),
            "mgcv fallback must remain available")

unsupported <- tryCatch(
  fastkpc_execute_ci_kpc_tprs_residual_cpp(
    data = scenarios[[1L]], x = 1L, y = 2L, S = 1:3,
    ci_method = "dcc.gamma", index = 1, legacy_index = TRUE,
    hsic_params = list(), permutation_params = list(),
    route = list(setup_fingerprint = "unsupported"), role = "primary"
  ),
  error = function(e) e
)
assert_true(inherits(unsupported, "error"),
            "unsupported |S| > 2 should fail closed")
assert_true(grepl("1 <= |S| <= 2", conditionMessage(unsupported), fixed = TRUE),
            "unsupported failure should name supported conditioning range")

cat("PASS kpcTprsResidualCPP switch qualification Gate E\n")
