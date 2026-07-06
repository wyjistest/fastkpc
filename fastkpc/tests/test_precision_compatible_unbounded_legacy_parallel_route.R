source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

missing <- c("graph", "mgcv", "pcalg", "RSpectra")[
  !vapply(c("graph", "mgcv", "pcalg", "RSpectra"), requireNamespace,
          logical(1), quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP precision compatible unbounded legacy parallel route: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

set.seed(7144)
data <- matrix(stats::rnorm(64 * 5), 64, 5)
colnames(data) <- paste0("V", seq_len(ncol(data)))

result <- fast_kpc(
  data,
  alpha = 0.05,
  max_conditioning_size = 3,
  engine = "cuda",
  precision = "compatible",
  graph_stage = "skeleton",
  ci_method = "dcc.gamma",
  precision_trace_level = "summary",
  benchmark = TRUE
)

assert_true(identical(result$skeleton$scheduler, "legacy-parallel"),
            "default compatible unbounded skeleton should use legacy parallel scheduler")
assert_true(identical(result$skeleton$residual_backend, "legacy-mgcv"),
            "legacy parallel compatible skeleton should report legacy mgcv backend")
assert_true(identical(result$skeleton$ci_backend, "legacy-dcov.gamma"),
            "legacy parallel compatible skeleton should report legacy dCov backend")

cat("PASS precision compatible unbounded legacy parallel route\n")
