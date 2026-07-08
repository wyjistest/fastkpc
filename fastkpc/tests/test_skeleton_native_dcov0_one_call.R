source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/dcov_exact.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

set.seed(3107)
n <- 72L
z <- stats::runif(n, -2, 2)
data <- cbind(
  x1 = z + stats::rnorm(n, sd = 0.08),
  x2 = z^2 + stats::rnorm(n, sd = 0.08),
  x3 = stats::rnorm(n),
  x4 = stats::rnorm(n)
)
alpha <- 0.10

tasks <- fastkpc_batched_precision_make_layer_plan(
  {
    adjacency <- matrix(TRUE, ncol(data), ncol(data))
    diag(adjacency) <- FALSE
    adjacency
  },
  level = 0L
)
pvalues <- vapply(tasks, function(task) {
  dcov_gamma_exact(data[, task$x], data[, task$y],
                   index = 1, legacy_index = TRUE)$p.value
}, numeric(1L))
reference <- precision_replay_layer_native(
  adjacency = {
    adjacency <- matrix(TRUE, ncol(data), ncol(data))
    diag(adjacency) <- FALSE
    adjacency
  },
  edge_x = vapply(tasks, `[[`, integer(1L), "edge_x"),
  edge_y = vapply(tasks, `[[`, integer(1L), "edge_y"),
  x = vapply(tasks, `[[`, integer(1L), "x"),
  y = vapply(tasks, `[[`, integer(1L), "y"),
  conditioning_sets = lapply(tasks, `[[`, "S"),
  p_values = pvalues,
  alpha = alpha,
  trace_level = "full"
)

native <- precision_run_skeleton_dcov0_native(
  data = data,
  alpha = alpha,
  index = 1,
  legacy_index = TRUE,
  trace_level = "full"
)

assert_true(identical(native$adjacency, reference$adjacency),
            "native dCov0 one-call adjacency should match R exact dCov replay")
assert_true(max(abs(native$pMax - reference$pMax)) < 1e-10,
            "native dCov0 one-call pMax should match R exact dCov replay")
assert_true(identical(as.integer(native$n.edgetests),
                      as.integer(reference$n.edgetests)),
            "native dCov0 one-call n.edgetests should match R exact dCov replay")
assert_true(identical(as.integer(native$summary$dcov_native_count),
                      length(tasks)),
            "native dCov0 summary should count all planned native dCov tasks")
assert_true(identical(as.integer(native$summary$levels), 1L),
            "native dCov0 summary should run exactly one level")
assert_true(any(native$tasks$native_edge_deleted),
            "native dCov0 gate should exercise at least one deletion")
assert_true(any(native$tasks$native_edge_ignored),
            "native dCov0 gate should exercise ignored post-delete rows")
assert_true(all(native$tasks$conditioning_size == 0L),
            "native dCov0 gate should only emit unconditional tasks")

cat("PASS skeleton native dCov0 one-call\n")
