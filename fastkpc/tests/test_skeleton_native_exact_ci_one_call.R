source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/dcov_exact.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

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

exact_ci_r <- function(data, x, y, S, index = 1, legacy_index = TRUE) {
  if (length(S) == 0L) {
    return(dcov_gamma_exact(data[, x], data[, y],
                            index = index,
                            legacy_index = legacy_index)$p.value)
  }
  residuals <- stats::lm(
    data[, c(x, y)] ~ as.matrix(data[, S, drop = FALSE])
  )$residuals
  dcov_gamma_exact(residuals[, 1L], residuals[, 2L],
                   index = index, legacy_index = legacy_index)$p.value
}

reference_exact_skeleton <- function(data, alpha, max_conditioning_size,
                                     index = 1, legacy_index = TRUE) {
  p <- ncol(data)
  adjacency <- matrix(TRUE, p, p)
  diag(adjacency) <- FALSE
  pmax <- matrix(-Inf, p, p)
  diag(pmax) <- 1
  sepsets <- replicate(p, replicate(p, integer(), simplify = FALSE),
                       simplify = FALSE)
  n_edge_tests <- integer()
  all_tasks <- list()

  for (level in seq.int(0L, max_conditioning_size)) {
    tasks <- fastkpc_batched_precision_make_layer_plan(adjacency, level)
    pvalues <- vapply(tasks, function(task) {
      exact_ci_r(data, task$x, task$y, task$S,
                 index = index, legacy_index = legacy_index)
    }, numeric(1L))
    replay <- precision_replay_layer_native(
      adjacency = adjacency,
      edge_x = vapply(tasks, `[[`, integer(1L), "edge_x"),
      edge_y = vapply(tasks, `[[`, integer(1L), "edge_y"),
      x = vapply(tasks, `[[`, integer(1L), "x"),
      y = vapply(tasks, `[[`, integer(1L), "y"),
      conditioning_sets = lapply(tasks, `[[`, "S"),
      p_values = pvalues,
      alpha = alpha,
      pmax = pmax,
      trace_level = "summary"
    )
    adjacency <- replay$adjacency
    pmax <- replay$pMax
    for (entry in replay$per.level.log) {
      x <- as.integer(entry$x)
      y <- as.integer(entry$y)
      sepsets[[x]][[y]] <- as.integer(entry$S)
      sepsets[[y]][[x]] <- as.integer(entry$S)
    }
    n_edge_tests <- c(n_edge_tests,
                      as.integer(replay$summary$tests_replayed))
    all_tasks <- c(all_tasks, tasks)
  }
  list(adjacency = adjacency, pMax = pmax, sepsets = sepsets,
       n.edgetests = n_edge_tests, task_count = length(all_tasks))
}

set.seed(8429)
n <- 76L
z <- stats::runif(n, -2, 2)
data <- cbind(
  x1 = z + stats::rnorm(n, sd = 0.04),
  x2 = z^2 + stats::rnorm(n, sd = 0.04),
  x3 = sin(z) + stats::rnorm(n, sd = 0.04),
  x4 = cos(z) + stats::rnorm(n, sd = 0.04)
)
alpha <- 0.05
max_conditioning_size <- 1L

reference <- reference_exact_skeleton(
  data, alpha = alpha, max_conditioning_size = max_conditioning_size,
  index = 1, legacy_index = TRUE
)
native <- precision_run_skeleton_exact_ci_native(
  data = data,
  alpha = alpha,
  max_conditioning_size = max_conditioning_size,
  index = 1,
  legacy_index = TRUE,
  trace_level = "full"
)

assert_true(identical(native$adjacency, reference$adjacency),
            "native exact-CI skeleton adjacency should match R reference")
assert_true(max(abs(native$pMax - reference$pMax)) < 1e-10,
            "native exact-CI skeleton pMax should match R reference")
assert_true(identical(as.integer(native$n.edgetests),
                      as.integer(reference$n.edgetests)),
            "native exact-CI skeleton n.edgetests should match R reference")
assert_true(compare_sepsets(native$sepsets, reference$sepsets),
            "native exact-CI skeleton sepsets should match R reference")
assert_true(identical(as.integer(native$summary$ci_native_count),
                      as.integer(reference$task_count)),
            "native exact-CI summary should count planned native CI tasks")
assert_true(as.integer(native$summary$residual_native_count) > 0L,
            "native exact-CI summary should count conditional residualizations")
assert_true(any(native$tasks$conditioning_size > 0L),
            "native exact-CI trace should include conditional tasks")
assert_true(identical(as.integer(native$summary$levels),
                      max_conditioning_size + 1L),
            "native exact-CI summary should record level count")

cat("PASS skeleton native exact-CI one-call\n")
