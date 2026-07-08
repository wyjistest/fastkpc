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
route <- Sys.getenv(
  "FASTKPC_NATIVE_LEGACY_ONE_CALL_REAL_SUBSET_ROUTE",
  unset = "wrapper"
)
assert_true(route %in% c("wrapper", "facade"),
            "real subset route must be wrapper or facade")

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

real_data <- readRDS(real_path)
scenarios <- list(
  hot8 = list(
    cols = c(1L, 2L, 3L, 4L, 5L, 6L, 9L, 12L),
    min_requests = 100L
  ),
  hot12 = list(
    cols = c(1L, 2L, 3L, 4L, 5L, 6L, 9L, 12L,
             15L, 16L, 17L, 18L),
    min_requests = 300L
  )
)

requested <- Sys.getenv(
  "FASTKPC_NATIVE_LEGACY_ONE_CALL_REAL_SUBSET_CASES",
  unset = ""
)
if (nzchar(requested)) {
  requested_names <- trimws(strsplit(requested, ",", fixed = TRUE)[[1L]])
  requested_names <- requested_names[nzchar(requested_names)]
  unknown <- setdiff(requested_names, names(scenarios))
  assert_true(length(unknown) == 0L,
              paste("unknown real subset scenario(s):",
                    paste(unknown, collapse = ",")))
  scenarios <- scenarios[requested_names]
}
scenario_names <- names(scenarios)
assert_true(length(scenario_names) > 0L,
            "real subset gate should select at least one scenario")
if (!nzchar(requested)) {
  assert_true("hot12" %in% scenario_names,
              "real subset gate should include the hot12 scenario")
}

run_scenario <- function(name, scenario) {
  data <- as.matrix(real_data[, scenario$cols, drop = FALSE])
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

  one_call <- if (identical(route, "facade")) {
    fastkpc_compatible_cuda_skeleton(
      data = data,
      alpha = alpha,
      labels = colnames(data),
      options = list(
        max_conditioning_size = max_conditioning_size,
        index = index,
        numCol = numCol,
        trace_level = "summary",
        dcov_batch = "level"
      )
    )
  } else {
    precision_run_skeleton_legacy_mgcv_legacy_dcov_native(
      data = data,
      alpha = alpha,
      max_conditioning_size = max_conditioning_size,
      index = index,
      numCol = numCol,
      trace_level = "summary"
    )
  }

  prefix <- paste0("real subset ", name, " one-call")
  assert_true(identical(unname(one_call$adjacency), unname(explicit$adjacency)),
              paste(prefix, "adjacency should match explicit provider"))
  assert_true(pmax_max_abs_diff(unname(one_call$pMax),
                                unname(explicit$pMax)) < 1e-12,
              paste(prefix, "pMax should match explicit provider"))
  assert_true(identical(as.integer(one_call$n.edgetests),
                        as.integer(explicit$n.edgetests)),
              paste(prefix, "n.edgetests should match explicit provider"))
  assert_true(compare_sepsets(one_call$sepsets, explicit$sepsets),
              paste(prefix, "sepsets should match explicit provider"))
  assert_true(provider_counts$request_count > scenario$min_requests,
              paste(prefix, "should exercise nontrivial residual-provider traffic"))
  assert_true(identical(as.integer(one_call$summary$residual_provider_request_count),
                        as.integer(provider_counts$request_count)),
              paste(prefix, "should preserve residual request count"))
  assert_true(identical(as.integer(one_call$summary$legacy_dcov_native_count),
                        as.integer(explicit$summary$legacy_dcov_native_count)),
              paste(prefix, "should preserve legacy dCov task count"))
  assert_true(identical(one_call$summary$entrypoint,
                        "legacy-mgcv-legacy-dcov-native"),
              paste(prefix, "should record its entrypoint"))
  assert_true(isTRUE(one_call$summary$residual_provider_hidden),
              paste(prefix, "should report hidden residual-provider seam"))
  if (identical(route, "facade")) {
    assert_true(isTRUE(one_call$summary$compatible_cuda_facade),
                paste(prefix, "facade route should report compatible CUDA facade"))
    assert_true(identical(one_call$summary$compatible_cuda_route,
                          "legacy-mgcv-provider-native-legacy-dcov"),
                paste(prefix, "facade route should record compatible CUDA route"))
    assert_true(identical(one_call$summary$compatible_cuda_residual_authority,
                          "legacy-mgcv-regrXonS-provider"),
                paste(prefix, "facade route should record residual authority"))
    assert_true(identical(one_call$summary$compatible_cuda_ci_authority,
                          "native-legacy-dcov.gamma"),
                paste(prefix, "facade route should record CI authority"))
    assert_true(isTRUE(one_call$summary$legacy_dcov_native_batch_enabled),
                paste(prefix, "facade route should pass dcov_batch='level'"))
    assert_true(identical(rownames(one_call$adjacency), colnames(data)) &&
                  identical(colnames(one_call$adjacency), colnames(data)),
                paste(prefix, "facade route should label adjacency"))
    assert_true(identical(rownames(one_call$pMax), colnames(data)) &&
                  identical(colnames(one_call$pMax), colnames(data)),
                paste(prefix, "facade route should label pMax"))
  }

  cat(sprintf(
    "PASS skeleton native legacy mgcv legacy dCov real subset %s route=%s p=%d requests=%d\n",
    name,
    route,
    ncol(data),
    as.integer(provider_counts$request_count)
  ))
}

for (name in scenario_names) {
  run_scenario(name, scenarios[[name]])
}

cat("PASS skeleton native legacy mgcv legacy dCov real subset scenarios:",
    paste(scenario_names, collapse = ","), "\n")
