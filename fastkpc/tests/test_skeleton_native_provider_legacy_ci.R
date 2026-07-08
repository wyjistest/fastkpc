source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

missing <- c("mgcv", "RSpectra")[
  !vapply(c("mgcv", "RSpectra"), requireNamespace, logical(1),
          quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP skeleton native provider legacy CI: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

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

set.seed(9421)
n <- 64L
z <- stats::runif(n, -2, 2)
data <- cbind(
  x1 = sin(z) + stats::rnorm(n, sd = 0.08),
  x2 = cos(z) + stats::rnorm(n, sd = 0.08),
  x3 = z + stats::rnorm(n, sd = 0.08),
  x4 = z^2 + stats::rnorm(n, sd = 0.08)
)

make_provider <- function(counter_env) {
  force(counter_env)
  function(tasks, level) {
    counter_env$level_calls <- counter_env$level_calls + 1L
    counter_env$task_count <- counter_env$task_count + nrow(tasks)
    required <- c("task_index", "edge_x", "edge_y", "x", "y",
                  "conditioning_sets", "S_key", "conditioning_size")
    missing_fields <- setdiff(required, names(tasks))
    if (length(missing_fields) > 0L) {
      stop("provider task table missing fields: ",
           paste(missing_fields, collapse = ","), call. = FALSE)
    }
    vapply(seq_len(nrow(tasks)), function(i) {
      S <- as.integer(tasks$conditioning_sets[[i]])
      if (length(S) > 0L) counter_env$conditional_calls <-
        counter_env$conditional_calls + 1L
      route <- list(
        primary_backend = "legacy-mgcv",
        setup_fingerprint = paste0("legacy-mgcv:S:",
                                   fastkpc_precision_S_key(S))
      )
      fit <- fastkpc_execute_ci_legacy_mgcv(
        data = data,
        x = as.integer(tasks$x[[i]]),
        y = as.integer(tasks$y[[i]]),
        S = S,
        ci_method = "dcc.gamma",
        index = 1,
        legacy_index = TRUE,
        hsic_params = list(),
        permutation_params = list(),
        route = route,
        role = "primary"
      )
      as.numeric(fit$p.value)
    }, numeric(1L))
  }
}

replay_reference <- function(provider, alpha, max_conditioning_size) {
  p <- ncol(data)
  adjacency <- matrix(TRUE, p, p)
  diag(adjacency) <- FALSE
  pmax <- matrix(-Inf, p, p)
  diag(pmax) <- 1
  sepsets <- replicate(p, replicate(p, integer(), simplify = FALSE),
                       simplify = FALSE)
  n_edge_tests <- integer()
  total_tasks <- 0L

  for (level in seq.int(0L, max_conditioning_size)) {
    tasks <- fastkpc_batched_precision_make_layer_plan(adjacency, level)
    total_tasks <- total_tasks + length(tasks)
    task_table <- data.frame(
      task_index = seq_along(tasks),
      edge_x = vapply(tasks, `[[`, integer(1L), "edge_x"),
      edge_y = vapply(tasks, `[[`, integer(1L), "edge_y"),
      x = vapply(tasks, `[[`, integer(1L), "x"),
      y = vapply(tasks, `[[`, integer(1L), "y"),
      S_key = vapply(tasks, `[[`, character(1L), "S_key"),
      conditioning_size =
        vapply(tasks, function(task) length(task$S), integer(1L)),
      stringsAsFactors = FALSE
    )
    task_table$conditioning_sets <- I(lapply(tasks, `[[`, "S"))
    pvalues <- provider(task_table, level)
    replay <- precision_replay_layer_native(
      adjacency = adjacency,
      edge_x = task_table$edge_x,
      edge_y = task_table$edge_y,
      x = task_table$x,
      y = task_table$y,
      conditioning_sets = task_table$conditioning_sets,
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
  }
  list(adjacency = adjacency, pMax = pmax, sepsets = sepsets,
       n.edgetests = n_edge_tests, total_tasks = total_tasks)
}

alpha <- 0.05
max_conditioning_size <- 1L

ref_counts <- new.env(parent = emptyenv())
ref_counts$level_calls <- 0L
ref_counts$task_count <- 0L
ref_counts$conditional_calls <- 0L
reference <- replay_reference(make_provider(ref_counts), alpha,
                              max_conditioning_size)

native_counts <- new.env(parent = emptyenv())
native_counts$level_calls <- 0L
native_counts$task_count <- 0L
native_counts$conditional_calls <- 0L
native <- precision_run_skeleton_provider_native(
  p = ncol(data),
  alpha = alpha,
  max_conditioning_size = max_conditioning_size,
  provider = make_provider(native_counts),
  trace_level = "full"
)

assert_true(identical(native$adjacency, reference$adjacency),
            "native provider skeleton adjacency should match R replay")
assert_true(max(abs(native$pMax - reference$pMax)) < 1e-12,
            "native provider skeleton pMax should match R replay")
assert_true(identical(as.integer(native$n.edgetests),
                      as.integer(reference$n.edgetests)),
            "native provider skeleton n.edgetests should match R replay")
assert_true(compare_sepsets(native$sepsets, reference$sepsets),
            "native provider skeleton sepsets should match R replay")
assert_true(native_counts$level_calls == max_conditioning_size + 1L,
            "native provider should be called once per level")
assert_true(native_counts$task_count == reference$total_tasks,
            "native provider should receive every planned task")
assert_true(native_counts$conditional_calls > 0L,
            "native provider gate should exercise conditional legacy CI tasks")
assert_true(any(native$tasks$conditioning_size > 0L),
            "native provider trace should include conditional tasks")

cat("PASS skeleton native provider legacy CI\n")
