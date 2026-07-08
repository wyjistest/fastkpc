source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_REAL_SUBSET_TESTS", unset = ""), "1")) {
  cat("SKIP skeleton native legacy mgcv legacy dCov real subset: set FASTKPC_RUN_REAL_SUBSET_TESTS=1\n")
  quit(save = "no", status = 0)
}

missing <- c("mgcv", "RSpectra")[
  !vapply(c("mgcv", "RSpectra"), requireNamespace, logical(1),
          quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP skeleton native legacy mgcv legacy dCov real subset: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

real_path <- "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"
if (!file.exists(real_path)) {
  cat("SKIP skeleton native legacy mgcv legacy dCov real subset: real fixture unavailable\n")
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

pmax_max_abs_diff <- function(left, right) {
  finite <- is.finite(left) & is.finite(right)
  if (!any(finite)) return(0)
  max(abs(left[finite] - right[finite]))
}

data <- readRDS(real_path)
hot_cols <- c(1L, 2L, 3L, 4L, 5L, 6L, 9L, 12L)
data <- as.matrix(data[, hot_cols, drop = FALSE])
storage.mode(data) <- "double"

alpha <- 0.1
index <- 1
numCol <- floor(nrow(data) / 10)
max_conditioning_size <- 2L

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
  trace_level = "summary"
)

one_call <- precision_run_skeleton_legacy_mgcv_legacy_dcov_native(
  data = data,
  alpha = alpha,
  max_conditioning_size = max_conditioning_size,
  index = index,
  numCol = numCol,
  trace_level = "summary"
)

assert_true(identical(one_call$adjacency, explicit$adjacency),
            "real subset one-call adjacency should match explicit provider")
assert_true(pmax_max_abs_diff(one_call$pMax, explicit$pMax) < 1e-12,
            "real subset one-call pMax should match explicit provider")
assert_true(identical(as.integer(one_call$n.edgetests),
                      as.integer(explicit$n.edgetests)),
            "real subset one-call n.edgetests should match explicit provider")
assert_true(compare_sepsets(one_call$sepsets, explicit$sepsets),
            "real subset one-call sepsets should match explicit provider")
assert_true(provider_counts$request_count > 100L,
            "real subset should exercise nontrivial residual-provider traffic")
assert_true(identical(as.integer(one_call$summary$residual_provider_request_count),
                      as.integer(provider_counts$request_count)),
            "real subset one-call should preserve residual request count")
assert_true(identical(as.integer(one_call$summary$legacy_dcov_native_count),
                      as.integer(explicit$summary$legacy_dcov_native_count)),
            "real subset one-call should preserve legacy dCov task count")
assert_true(identical(one_call$summary$entrypoint,
                      "legacy-mgcv-legacy-dcov-native"),
            "real subset one-call should record its entrypoint")
assert_true(isTRUE(one_call$summary$residual_provider_hidden),
            "real subset one-call should report hidden residual-provider seam")

cat("PASS skeleton native legacy mgcv legacy dCov real subset\n")
