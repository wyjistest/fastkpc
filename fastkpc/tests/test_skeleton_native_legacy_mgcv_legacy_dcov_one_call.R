source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

missing <- c("mgcv", "RSpectra")[
  !vapply(c("mgcv", "RSpectra"), requireNamespace, logical(1),
          quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP skeleton native legacy mgcv legacy dCov one-call: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = "spectra")

compare_sepsets <- function(left, right) {
  if (length(left) != length(right)) return(FALSE)
  for (i in seq_along(left)) {
    if (length(left[[i]]) != length(right[[i]])) return(FALSE)
    for (j in seq_along(left[[i]])) {
      lhs <- sort(as.integer(left[[i]][[j]]))
      rhs <- sort(as.integer(right[[i]][[j]]))
      if (!identical(lhs, rhs)) return(FALSE)
    }
  }
  TRUE
}

set.seed(5303)
n <- 72L
z <- stats::runif(n, -2, 2)
data <- cbind(
  x1 = sin(z) + stats::rnorm(n, sd = 0.06),
  x2 = cos(z) + stats::rnorm(n, sd = 0.06),
  x3 = z + stats::rnorm(n, sd = 0.06),
  x4 = z^2 + stats::rnorm(n, sd = 0.06)
)
alpha <- 0.05
index <- 1
numCol <- floor(n / 10)
max_conditioning_size <- 1L

provider_counts <- new.env(parent = emptyenv())
provider_counts$level_calls <- 0L
provider_counts$request_count <- 0L
explicit <- precision_run_skeleton_residual_provider_legacy_dcov_native(
  data = data,
  alpha = alpha,
  max_conditioning_size = max_conditioning_size,
  residual_provider = fastkpc_legacy_mgcv_residual_provider(
    data = data,
    counter_env = provider_counts
  ),
  index = index,
  numCol = numCol,
  trace_level = "full"
)

one_call <- precision_run_skeleton_legacy_mgcv_legacy_dcov_native(
  data = data,
  alpha = alpha,
  max_conditioning_size = max_conditioning_size,
  index = index,
  numCol = numCol,
  trace_level = "full"
)

assert_true(identical(one_call$adjacency, explicit$adjacency),
            "one-call legacy mgcv legacy dCov adjacency should match explicit provider")
assert_true(max(abs(one_call$pMax - explicit$pMax)) < 1e-12,
            "one-call legacy mgcv legacy dCov pMax should match explicit provider")
assert_true(identical(as.integer(one_call$n.edgetests),
                      as.integer(explicit$n.edgetests)),
            "one-call legacy mgcv legacy dCov n.edgetests should match explicit provider")
assert_true(compare_sepsets(one_call$sepsets, explicit$sepsets),
            "one-call legacy mgcv legacy dCov sepsets should match explicit provider")
assert_true(provider_counts$level_calls > 0L,
            "explicit provider should be exercised for conditional levels")
assert_true(identical(as.integer(one_call$summary$residual_provider_request_count),
                      as.integer(explicit$summary$residual_provider_request_count)),
            "one-call wrapper should preserve residual request count")
assert_true(identical(as.integer(one_call$summary$legacy_dcov_native_count),
                      as.integer(explicit$summary$legacy_dcov_native_count)),
            "one-call wrapper should preserve legacy dCov task count")
assert_true(identical(one_call$summary$ci_backend,
                      "native-legacy-dcov.gamma"),
            "one-call wrapper should keep legacy dCov backend")
assert_true(identical(one_call$summary$residual_backend,
                      "provider-legacy-mgcv"),
            "one-call wrapper should keep legacy mgcv residual authority")
assert_true(identical(one_call$summary$entrypoint,
                      "legacy-mgcv-legacy-dcov-native"),
            "one-call wrapper should record its entrypoint")

cat("PASS skeleton native legacy mgcv legacy dCov one-call\n")
