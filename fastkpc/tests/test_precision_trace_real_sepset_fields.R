source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP precision trace real sepset fields: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

set.seed(995)
n <- 50
s <- seq(-1, 1, length.out = n)
data <- cbind(
  x = sin(s) + rnorm(n, sd = 0.04),
  y = cos(s) + rnorm(n, sd = 0.04),
  z = s
)

caps <- list(
  R_version = "4.5.0",
  mgcv_version = "1.9-4",
  cuda_available = TRUE,
  mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
)

result <- fast_kpc(
  data,
  alpha = 0.05,
  max_conditioning_size = 1,
  engine = "cpu",
  precision = "compatible",
  graph_stage = "skeleton",
  runtime_capabilities = caps,
  precision_trace_level = "full"
)

trace <- result$diagnostics$precision_trace
required <- c("x", "y", "S_key", "conditioning_target_side",
              "sepset_recorded")
missing <- setdiff(required, names(trace))
assert_true(length(missing) == 0L,
            paste("trace missing real scheduler fields:",
                  paste(missing, collapse = ", ")))

deleted <- trace[trace$edge_deleted %in% TRUE, , drop = FALSE]
assert_true(nrow(deleted) > 0L,
            "scenario should produce at least one deletion")
assert_true(all(deleted$sepset_recorded == deleted$S_key),
            "deleted rows should record sepset directly from S_key")
assert_true(all(is.finite(trace$x) & is.finite(trace$y)),
            "trace should include numeric endpoint columns")
assert_true(all(trace$conditioning_target_side %in% c("x", "y")),
            "trace should record which endpoint provided the conditioning side")

cat("PASS precision trace real sepset fields\n")
