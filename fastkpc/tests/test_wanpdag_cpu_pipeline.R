source("fastkpc/R/native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

fixed_data <- function(n = 120) {
  z <- seq(-pi, pi, length.out = n)
  cbind(
    x1 = z,
    x2 = sin(z) + 0.03 * cos(19 * z),
    x3 = cos(0.5 * z),
    x4 = z^2 + 0.02 * sin(11 * z)
  )
}

data <- fixed_data()

result <- fast_kpc_wanpdag_cpp(
  data,
  alpha = 0.12,
  max_conditioning_size = 1L,
  residual_backend = "fastSpline",
  residual_cache = TRUE
)

assert_true(is.list(result$skeleton), "result should contain skeleton")
assert_true(is.list(result$orientation), "result should contain orientation")
assert_true(is.integer(result$orientation$pdag),
            "orientation$pdag should be an integer matrix")
assert_true(identical(dim(result$orientation$pdag), c(ncol(data), ncol(data))),
            "orientation$pdag should be square with p columns")
assert_true(identical(result$orientation$residual_backend, "fastSpline"),
            "orientation residual backend should be fastSpline")
assert_true(result$orientation$residual_cache$requests >=
              result$orientation$residual_cache$computations,
            "orientation cache stats should be recorded")

oriented <- fast_orient_wanpdag_cpp(
  result$skeleton,
  data,
  residual_backend = "fastSpline",
  residual_cache = TRUE,
  alpha = 0.12
)
assert_true(identical(oriented$pdag, result$orientation$pdag),
            "orienting an existing skeleton should match full CPU pipeline")

linear_result <- fast_kpc_wanpdag_cpp(
  data,
  alpha = 0.12,
  max_conditioning_size = 1L,
  residual_backend = "linear",
  residual_cache = TRUE
)
assert_true(identical(linear_result$orientation$residual_backend, "linear"),
            "linear residual backend should be accepted")

no_collider <- fast_kpc_wanpdag_cpp(
  data,
  alpha = 0.12,
  max_conditioning_size = 1L,
  residual_backend = "fastSpline",
  orient_collider = FALSE
)
assert_true(no_collider$orientation$counts$collider == 0L,
            "disabling colliders should force collider count to zero")

no_rules <- fast_kpc_wanpdag_cpp(
  data,
  alpha = 0.12,
  max_conditioning_size = 1L,
  residual_backend = "fastSpline",
  rules = c(FALSE, FALSE, FALSE)
)
assert_true(no_rules$orientation$counts$rule1 == 0L &&
              no_rules$orientation$counts$rule2 == 0L &&
              no_rules$orientation$counts$rule3 == 0L,
            "disabling rules should force rule counts to zero")

repeat_result <- fast_kpc_wanpdag_cpp(
  data,
  alpha = 0.12,
  max_conditioning_size = 1L,
  residual_backend = "fastSpline",
  residual_cache = TRUE
)
assert_true(identical(repeat_result$orientation$pdag, result$orientation$pdag),
            "repeated CPU WAN-PDAG runs should be deterministic")

cat("test_wanpdag_cpu_pipeline.R: PASS\n")
