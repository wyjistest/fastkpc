source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
`%||%` <- function(x, y) if (is.null(x)) y else x

missing <- c("graph", "mgcv", "pcalg", "Rcpp", "RcppArmadillo", "RSpectra")[
  !vapply(c("graph", "mgcv", "pcalg", "Rcpp", "RcppArmadillo", "RSpectra"),
          requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP legacy dCov gamma C++ shadow route: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

old_shadow <- Sys.getenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_SHADOW",
                         unset = NA_character_)
on.exit({
  if (is.na(old_shadow)) {
    Sys.unsetenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_SHADOW")
  } else {
    Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_CPP_SHADOW = old_shadow)
  }
}, add = TRUE)

set.seed(9124)
n <- 54L
z2 <- stats::runif(n, -2, 2)
z <- stats::runif(n, -2, 2)
data <- cbind(
  x1 = sin(z) + stats::rnorm(n, sd = 0.15),
  x2 = cos(z) + stats::rnorm(n, sd = 0.15),
  x3 = z,
  x4 = z2,
  x5 = stats::rnorm(n)
)

run_compatible <- function() {
  fast_kpc(
    data,
    alpha = 0.08,
    max_conditioning_size = 3,
    engine = "cuda",
    precision = "compatible",
    graph_stage = "skeleton",
    ci_method = "dcc.gamma",
    precision_trace_level = "summary",
    benchmark = TRUE
  )
}

Sys.unsetenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_SHADOW")
baseline <- run_compatible()
baseline_summary <- baseline$skeleton$scheduler_diagnostics$summary

assert_true(identical(as.integer(
  baseline_summary$legacy_dcov_cpp_shadow_count %||% 0L), 0L),
  "default compatible route should not run legacy dCov C++ shadow")

Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_CPP_SHADOW = "1")
shadow <- run_compatible()
shadow_summary <- shadow$skeleton$scheduler_diagnostics$summary

assert_true(identical(shadow$skeleton$adjacency, baseline$skeleton$adjacency),
            "legacy dCov C++ shadow should not change skeleton adjacency")
assert_true(identical(shadow$skeleton$n.edgetests,
                      baseline$skeleton$n.edgetests),
            "legacy dCov C++ shadow should not change n.edgetests")
assert_true(shadow_summary$legacy_dcov_gamma_count > 0L,
            "shadow scenario should execute legacy dCov calls")
assert_true(identical(as.integer(shadow_summary$legacy_dcov_cpp_shadow_count),
                      as.integer(shadow_summary$legacy_dcov_gamma_count)),
            "shadow count should match legacy dCov call count")
assert_true(identical(
  as.integer(shadow_summary$legacy_dcov_cpp_shadow_decision_flip_count), 0L),
  "legacy dCov C++ shadow should not flip deletion decisions")
assert_true(identical(
  as.integer(shadow_summary$legacy_dcov_cpp_shadow_error_count), 0L),
  "legacy dCov C++ shadow should not report errors")
assert_true(shadow_summary$legacy_dcov_cpp_shadow_max_p_diff <= 1e-8,
            "legacy dCov C++ shadow p-value drift should stay within oracle tolerance")
assert_true(shadow_summary$legacy_dcov_cpp_shadow_max_nV2_diff <= 1e-8,
            "legacy dCov C++ shadow nV2 drift should stay within oracle tolerance")
assert_true(shadow_summary$legacy_dcov_cpp_shadow_ms > 0,
            "legacy dCov C++ shadow should report elapsed time")

cat("PASS legacy dCov gamma C++ shadow route\n")
