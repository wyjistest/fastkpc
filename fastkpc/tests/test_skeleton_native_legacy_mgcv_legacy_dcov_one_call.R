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
old_native_batch <- Sys.getenv("FASTKPC_NATIVE_LEGACY_DCOV_BATCH",
                               unset = NA_character_)
old_mgcv_backend <- Sys.getenv(c(
  "FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND",
  "FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_NATIVE_S_SIZE_LIMIT",
  "FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_CONDITION_THRESHOLD"
), unset = NA_character_)
on.exit({
  if (is.na(old_native_batch)) {
    Sys.unsetenv("FASTKPC_NATIVE_LEGACY_DCOV_BATCH")
  } else {
    Sys.setenv(FASTKPC_NATIVE_LEGACY_DCOV_BATCH = old_native_batch)
  }
  for (name in names(old_mgcv_backend)) {
    if (is.na(old_mgcv_backend[[name]])) {
      Sys.unsetenv(name)
    } else {
      do.call(Sys.setenv, stats::setNames(list(old_mgcv_backend[[name]]), name))
    }
  }
}, add = TRUE)
Sys.unsetenv("FASTKPC_NATIVE_LEGACY_DCOV_BATCH")
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND")
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_NATIVE_S_SIZE_LIMIT")
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_CONDITION_THRESHOLD")

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

one_call_batch <- precision_run_skeleton_legacy_mgcv_legacy_dcov_native(
  data = data,
  alpha = alpha,
  max_conditioning_size = max_conditioning_size,
  index = index,
  numCol = numCol,
  trace_level = "full",
  dcov_batch = "level"
)

Sys.setenv(
  FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND = "cpp_guarded",
  FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_NATIVE_S_SIZE_LIMIT = "1",
  FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_CONDITION_THRESHOLD = "1e300"
)
one_call_cpp_residual <- precision_run_skeleton_legacy_mgcv_legacy_dcov_native(
  data = data,
  alpha = alpha,
  max_conditioning_size = max_conditioning_size,
  index = index,
  numCol = numCol,
  trace_level = "full",
  dcov_batch = "level"
)
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND")
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_NATIVE_S_SIZE_LIMIT")
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_CONDITION_THRESHOLD")

facade <- fastkpc_compatible_cuda_skeleton(
  data = data,
  alpha = alpha,
  labels = colnames(data),
  options = list(
    max_conditioning_size = max_conditioning_size,
    index = index,
    numCol = numCol,
    trace_level = "full",
    dcov_batch = "level"
  )
)

Sys.setenv(FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND = "caller-sentinel")
facade_cpp_residual <- fastkpc_compatible_cuda_skeleton(
  data = data,
  alpha = alpha,
  labels = colnames(data),
  options = list(
    max_conditioning_size = max_conditioning_size,
    index = index,
    numCol = numCol,
    trace_level = "full",
    dcov_batch = "level",
    mgcv_residual_backend = "cpp_guarded",
    mgcv_residual_backend_native_s_size_limit = 1,
    mgcv_residual_backend_condition_threshold = 1e300
  )
)
assert_true(identical(Sys.getenv("FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND",
                                 unset = ""),
                      "caller-sentinel"),
            "compatible CUDA facade should restore caller mgcv residual backend env")
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND")

Sys.setenv(FASTKPC_NATIVE_LEGACY_DCOV_BATCH = "level")
one_call_none <- precision_run_skeleton_legacy_mgcv_legacy_dcov_native(
  data = data,
  alpha = alpha,
  max_conditioning_size = max_conditioning_size,
  index = index,
  numCol = numCol,
  trace_level = "summary",
  dcov_batch = "none"
)
assert_true(identical(Sys.getenv("FASTKPC_NATIVE_LEGACY_DCOV_BATCH", unset = ""),
                      "level"),
            "one-call dcov_batch='none' should restore caller batch env")
Sys.unsetenv("FASTKPC_NATIVE_LEGACY_DCOV_BATCH")

assert_true(identical(one_call$adjacency, explicit$adjacency),
            "one-call legacy mgcv legacy dCov adjacency should match explicit provider")
assert_true(identical(one_call_batch$adjacency, explicit$adjacency),
            "one-call level-batched legacy dCov adjacency should match explicit provider")
assert_true(identical(one_call_cpp_residual$adjacency, explicit$adjacency),
            "one-call guarded C++ residual backend adjacency should match explicit provider")
assert_true(identical(unname(facade$adjacency), unname(explicit$adjacency)),
            "compatible CUDA facade adjacency should match explicit provider")
assert_true(identical(unname(facade_cpp_residual$adjacency),
                      unname(explicit$adjacency)),
            "compatible CUDA facade guarded C++ residual adjacency should match explicit provider")
assert_true(identical(one_call_none$adjacency, explicit$adjacency),
            "one-call dcov_batch='none' adjacency should match explicit provider")
assert_true(max(abs(one_call$pMax - explicit$pMax)) < 1e-12,
            "one-call legacy mgcv legacy dCov pMax should match explicit provider")
assert_true(max(abs(one_call_batch$pMax - explicit$pMax)) < 1e-12,
            "one-call level-batched legacy dCov pMax should match explicit provider")
assert_true(max(abs(one_call_cpp_residual$pMax - explicit$pMax)) < 1e-8,
            "one-call guarded C++ residual backend pMax should match explicit provider")
assert_true(max(abs(unname(facade$pMax) - unname(explicit$pMax))) < 1e-12,
            "compatible CUDA facade pMax should match explicit provider")
assert_true(max(abs(unname(facade_cpp_residual$pMax) -
                    unname(explicit$pMax))) < 1e-8,
            "compatible CUDA facade guarded C++ residual pMax should match explicit provider")
assert_true(identical(as.integer(one_call$n.edgetests),
                      as.integer(explicit$n.edgetests)),
            "one-call legacy mgcv legacy dCov n.edgetests should match explicit provider")
assert_true(identical(as.integer(one_call_batch$n.edgetests),
                      as.integer(explicit$n.edgetests)),
            "one-call level-batched legacy dCov n.edgetests should match explicit provider")
assert_true(identical(as.integer(one_call_cpp_residual$n.edgetests),
                      as.integer(explicit$n.edgetests)),
            "one-call guarded C++ residual backend n.edgetests should match explicit provider")
assert_true(identical(as.integer(facade$n.edgetests),
                      as.integer(explicit$n.edgetests)),
            "compatible CUDA facade n.edgetests should match explicit provider")
assert_true(identical(as.integer(facade_cpp_residual$n.edgetests),
                      as.integer(explicit$n.edgetests)),
            "compatible CUDA facade guarded C++ residual n.edgetests should match explicit provider")
assert_true(compare_sepsets(one_call$sepsets, explicit$sepsets),
            "one-call legacy mgcv legacy dCov sepsets should match explicit provider")
assert_true(compare_sepsets(one_call_batch$sepsets, explicit$sepsets),
            "one-call level-batched legacy dCov sepsets should match explicit provider")
assert_true(compare_sepsets(one_call_cpp_residual$sepsets, explicit$sepsets),
            "one-call guarded C++ residual backend sepsets should match explicit provider")
assert_true(compare_sepsets(facade$sepsets, explicit$sepsets),
            "compatible CUDA facade sepsets should match explicit provider")
assert_true(compare_sepsets(facade_cpp_residual$sepsets, explicit$sepsets),
            "compatible CUDA facade guarded C++ residual sepsets should match explicit provider")
assert_true(provider_counts$level_calls > 0L,
            "explicit provider should be exercised for conditional levels")
assert_true(identical(as.integer(one_call$summary$residual_provider_request_count),
                      as.integer(explicit$summary$residual_provider_request_count)),
            "one-call wrapper should preserve residual request count")
assert_true(identical(as.integer(one_call_batch$summary$residual_provider_request_count),
                      as.integer(explicit$summary$residual_provider_request_count)),
            "one-call level-batched wrapper should preserve residual request count")
assert_true(identical(as.integer(one_call_cpp_residual$summary$residual_provider_request_count),
                      as.integer(explicit$summary$residual_provider_request_count)),
            "one-call guarded C++ residual backend should preserve residual request count")
assert_true(identical(as.integer(one_call$summary$legacy_dcov_native_count),
                      as.integer(explicit$summary$legacy_dcov_native_count)),
            "one-call wrapper should preserve legacy dCov task count")
assert_true(identical(as.integer(one_call_batch$summary$legacy_dcov_native_count),
                      as.integer(explicit$summary$legacy_dcov_native_count)),
            "one-call level-batched wrapper should preserve legacy dCov task count")
assert_true(identical(one_call$summary$ci_backend,
                      "native-legacy-dcov.gamma"),
            "one-call wrapper should keep legacy dCov backend")
assert_true(identical(one_call$summary$residual_backend,
                      "provider-legacy-mgcv"),
            "one-call wrapper should keep legacy mgcv residual authority")
assert_true(identical(one_call$summary$residual_provider_contract,
                      "level-residual-matrix-v1"),
            "one-call wrapper should record residual provider contract")
assert_true(identical(one_call$summary$residual_provider_response_mode,
                      "list"),
            "one-call wrapper should use structured residual provider response")
assert_true(identical(one_call$summary$residual_provider_response_backend,
                      "legacy-mgcv-regrXonS-level-batch"),
            "one-call wrapper should record hidden residual batch provider backend")
assert_true(identical(one_call_cpp_residual$summary$residual_provider_response_backend,
                      "legacy-mgcv-cpp-guarded-level-batch"),
            "one-call guarded C++ residual backend should record structured provider backend")
assert_true(identical(one_call_cpp_residual$summary$residual_provider_mgcv_backend,
                      "cpp_guarded"),
            "one-call guarded C++ residual backend should record selected mgcv backend")
assert_true(isTRUE(one_call_cpp_residual$summary$residual_provider_mgcv_cpp_backend_enabled),
            "one-call guarded C++ residual backend should report enabled")
assert_true(identical(as.integer(one_call_cpp_residual$summary$residual_provider_mgcv_cpp_backend_count),
                      as.integer(one_call_cpp_residual$summary$residual_provider_request_count)),
            "one-call guarded C++ residual backend should cover every provider request")
assert_true(one_call_cpp_residual$summary$residual_provider_mgcv_cpp_backend_native_count > 0L,
            "one-call guarded C++ residual backend should use native fixed-sp replay")
assert_true(identical(as.integer(one_call_cpp_residual$summary$residual_provider_mgcv_cpp_backend_fallback_count),
                      0L),
            "one-call guarded C++ residual backend should avoid fallback in the smoke envelope")
assert_true(identical(as.integer(one_call_cpp_residual$summary$residual_provider_mgcv_cpp_backend_error_count),
                      0L),
            "one-call guarded C++ residual backend should not report errors")
assert_true(identical(as.integer(one_call$summary$residual_provider_batch_count),
                      as.integer(one_call$summary$residual_provider_level_count)),
            "one-call wrapper should count one residual batch per conditional level")
assert_true(identical(as.integer(one_call$summary$residual_provider_matrix_cell_count),
                      as.integer(nrow(data) * one_call$summary$residual_provider_request_count)),
            "one-call wrapper should report residual provider matrix cell payload")
provider_timing_fields <- c(
  "residual_provider_call_ms",
  "residual_provider_matrix_copy_ms",
  "residual_provider_total_ms",
  "legacy_dcov_native_batch_materialize_ms",
  "legacy_dcov_native_batch_call_ms"
)
missing_provider_timing <- setdiff(provider_timing_fields,
                                   names(one_call_batch$summary))
assert_true(length(missing_provider_timing) == 0L,
            paste("one-call level-batched summary missing provider seam timing",
                  missing_provider_timing[[1L]]))
assert_true(one_call_batch$summary$residual_provider_call_ms >= 0,
            "one-call wrapper should time residual provider calls")
assert_true(one_call_batch$summary$residual_provider_matrix_copy_ms >= 0,
            "one-call wrapper should time residual provider matrix copies")
assert_true(one_call_batch$summary$residual_provider_total_ms >=
              one_call_batch$summary$residual_provider_call_ms,
            "one-call wrapper residual provider total should include call time")
assert_true(one_call_batch$summary$legacy_dcov_native_batch_materialize_ms >= 0,
            "one-call level-batched wrapper should time dCov batch materialization")
assert_true(one_call_batch$summary$legacy_dcov_native_batch_call_ms >= 0,
            "one-call level-batched wrapper should time dCov batch calls")
level_timing_fields <- c(
  "residual_provider_request_count",
  "residual_provider_call_ms",
  "residual_provider_matrix_copy_ms",
  "residual_provider_total_ms",
  "legacy_dcov_native_materialize_ms",
  "legacy_dcov_native_call_ms"
)
missing_level_timing <- setdiff(level_timing_fields, names(one_call_batch$levels))
assert_true(length(missing_level_timing) == 0L,
            paste("one-call level rows missing provider seam timing",
                  missing_level_timing[[1L]]))
assert_true(sum(as.integer(one_call_batch$levels$residual_provider_request_count)) ==
              as.integer(one_call_batch$summary$residual_provider_request_count),
            "one-call level rows should sum residual provider requests")
assert_true(sum(one_call_batch$levels$residual_provider_call_ms) >= 0,
            "one-call level rows should expose residual provider call timing")
assert_true(identical(one_call$summary$entrypoint,
                      "legacy-mgcv-legacy-dcov-native"),
            "one-call wrapper should record its entrypoint")
assert_true(!isTRUE(one_call$summary$legacy_dcov_native_batch_enabled),
            "one-call default should not enable native dCov batch without env")
assert_true(isTRUE(one_call_batch$summary$legacy_dcov_native_batch_enabled),
            "one-call dcov_batch='level' should enable native legacy dCov batching")
assert_true(isTRUE(facade$summary$legacy_dcov_native_batch_enabled),
            "compatible CUDA facade should pass dcov_batch option through")
assert_true(isTRUE(facade$summary$compatible_cuda_facade),
            "compatible CUDA facade should identify the R-facing facade")
assert_true(isTRUE(facade_cpp_residual$summary$compatible_cuda_facade),
            "compatible CUDA facade guarded C++ residual route should identify the facade")
assert_true(identical(facade$summary$compatible_cuda_entrypoint,
                      "fastkpc-compatible-cuda-skeleton"),
            "compatible CUDA facade should record the proposed API entrypoint")
assert_true(identical(facade$summary$compatible_cuda_route,
                      "legacy-mgcv-provider-native-legacy-dcov"),
            "compatible CUDA facade should record the experimental route")
assert_true(identical(facade$summary$compatible_cuda_residual_authority,
                      "legacy-mgcv-regrXonS-provider"),
            "compatible CUDA facade should record residual authority")
assert_true(identical(facade_cpp_residual$summary$compatible_cuda_residual_authority,
                      "legacy-mgcv-cpp-guarded-provider"),
            "compatible CUDA facade should record guarded C++ residual authority")
assert_true(identical(facade_cpp_residual$summary$residual_provider_response_backend,
                      "legacy-mgcv-cpp-guarded-level-batch"),
            "compatible CUDA facade should record guarded C++ provider backend")
assert_true(identical(facade_cpp_residual$summary$residual_provider_mgcv_backend,
                      "cpp_guarded"),
            "compatible CUDA facade should record selected guarded mgcv backend")
assert_true(identical(as.numeric(facade_cpp_residual$summary$compatible_cuda_mgcv_residual_backend_native_s_size_limit),
                      1),
            "compatible CUDA facade should record guarded residual native S-size limit")
assert_true(identical(as.numeric(facade_cpp_residual$summary$compatible_cuda_mgcv_residual_backend_condition_threshold),
                      1e300),
            "compatible CUDA facade should record guarded residual condition threshold")
assert_true(isTRUE(facade_cpp_residual$summary$residual_provider_mgcv_cpp_backend_enabled),
            "compatible CUDA facade should enable guarded C++ residual backend")
assert_true(identical(as.integer(facade_cpp_residual$summary$residual_provider_mgcv_cpp_backend_count),
                      as.integer(facade_cpp_residual$summary$residual_provider_request_count)),
            "compatible CUDA facade guarded C++ residual backend should cover provider requests")
assert_true(facade_cpp_residual$summary$residual_provider_mgcv_cpp_backend_native_count > 0L,
            "compatible CUDA facade guarded C++ residual backend should use native replay")
assert_true(identical(facade$summary$compatible_cuda_ci_authority,
                      "native-legacy-dcov.gamma"),
            "compatible CUDA facade should record CI authority")
assert_true(identical(facade$summary$native_entrypoint,
                      "legacy-mgcv-legacy-dcov-native"),
            "compatible CUDA facade should preserve native entrypoint metadata")
assert_true(identical(rownames(facade$adjacency), colnames(data)) &&
              identical(colnames(facade$adjacency), colnames(data)),
            "compatible CUDA facade should apply labels to adjacency")
assert_true(identical(rownames(facade$pMax), colnames(data)) &&
              identical(colnames(facade$pMax), colnames(data)),
            "compatible CUDA facade should apply labels to pMax")
assert_true(!isTRUE(one_call_none$summary$legacy_dcov_native_batch_enabled),
            "one-call dcov_batch='none' should disable native dCov batch during call")
assert_true(identical(as.integer(one_call_batch$summary$legacy_dcov_native_batch_pair_count),
                      as.integer(one_call_batch$summary$legacy_dcov_native_count)),
            "one-call level-batched wrapper should report all native dCov tasks as batch pairs")
assert_true(identical(as.integer(one_call_batch$summary$legacy_dcov_native_batch_column_materialize_count),
                      2L * as.integer(one_call_batch$summary$legacy_dcov_native_batch_pair_count)),
            "one-call level-batched wrapper should report host batch matrix materialization")
assert_true(!nzchar(Sys.getenv("FASTKPC_NATIVE_LEGACY_DCOV_BATCH", unset = "")),
            "one-call dcov_batch argument should not leak FASTKPC_NATIVE_LEGACY_DCOV_BATCH")

cat("PASS skeleton native legacy mgcv legacy dCov one-call\n")
