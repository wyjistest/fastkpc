source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/dcov_exact.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

missing <- c("mgcv", "RSpectra")[
  !vapply(c("mgcv", "RSpectra"), requireNamespace, logical(1),
          quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP skeleton native residual provider exact dCov: missing",
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

set.seed(5301)
n <- 68L
z <- stats::runif(n, -2, 2)
data <- cbind(
  x1 = sin(z) + stats::rnorm(n, sd = 0.06),
  x2 = cos(z) + stats::rnorm(n, sd = 0.06),
  x3 = z + stats::rnorm(n, sd = 0.06),
  x4 = z^2 + stats::rnorm(n, sd = 0.06)
)
alpha <- 0.05
max_conditioning_size <- 1L

make_residual_provider <- function(counter_env) {
  force(counter_env)
  function(requests, level) {
    counter_env$level_calls <- counter_env$level_calls + 1L
    counter_env$request_count <- counter_env$request_count + nrow(requests)
    required <- c("request_index", "target", "conditioning_sets",
                  "S_key", "conditioning_size")
    missing_fields <- setdiff(required, names(requests))
    if (length(missing_fields) > 0L) {
      stop("residual provider request table missing fields: ",
           paste(missing_fields, collapse = ","), call. = FALSE)
    }
    out <- matrix(NA_real_, nrow(data), nrow(requests))
    for (i in seq_len(nrow(requests))) {
      S <- as.integer(requests$conditioning_sets[[i]])
      assert_true(length(S) > 0L,
                  "residual provider should only receive conditional requests")
      out[, i] <- fastkpc_legacy_mgcv_residual(
        data = data,
        target = as.integer(requests$target[[i]]),
        S = S
      )
    }
    out
  }
}

reference_with_residual_provider <- function(provider) {
  p <- ncol(data)
  adjacency <- matrix(TRUE, p, p)
  diag(adjacency) <- FALSE
  pmax <- matrix(-Inf, p, p)
  diag(pmax) <- 1
  sepsets <- replicate(p, replicate(p, integer(), simplify = FALSE),
                       simplify = FALSE)
  n_edge_tests <- integer()
  task_count <- 0L
  residual_requests <- 0L

  for (level in seq.int(0L, max_conditioning_size)) {
    tasks <- fastkpc_batched_precision_make_layer_plan(adjacency, level)
    task_count <- task_count + length(tasks)
    residuals <- new.env(parent = emptyenv())
    requests <- list()
    seen <- new.env(parent = emptyenv())
    for (task in tasks) {
      if (length(task$S) == 0L) next
      for (target in c(task$x, task$y)) {
        key <- fastkpc_batched_precision_residual_key(target, task$S)
        if (exists(key, envir = seen, inherits = FALSE)) next
        assign(key, TRUE, envir = seen)
        requests[[length(requests) + 1L]] <- list(
          target = as.integer(target),
          S = as.integer(task$S),
          S_key = task$S_key
        )
      }
    }
    if (length(requests) > 0L) {
      request_table <- data.frame(
        request_index = seq_along(requests),
        target = vapply(requests, `[[`, integer(1L), "target"),
        S_key = vapply(requests, `[[`, character(1L), "S_key"),
        conditioning_size =
          vapply(requests, function(request) length(request$S), integer(1L)),
        stringsAsFactors = FALSE
      )
      request_table$conditioning_sets <- I(lapply(requests, `[[`, "S"))
      residual_matrix <- provider(request_table, level)
      residual_requests <- residual_requests + ncol(residual_matrix)
      for (i in seq_along(requests)) {
        key <- fastkpc_batched_precision_residual_key(
          requests[[i]]$target, requests[[i]]$S
        )
        assign(key, as.numeric(residual_matrix[, i]),
               envir = residuals)
      }
    }

    pvalues <- vapply(tasks, function(task) {
      if (length(task$S) == 0L) {
        rx <- data[, task$x]
        ry <- data[, task$y]
      } else {
        rx <- get(fastkpc_batched_precision_residual_key(task$x, task$S),
                  envir = residuals, inherits = FALSE)
        ry <- get(fastkpc_batched_precision_residual_key(task$y, task$S),
                  envir = residuals, inherits = FALSE)
      }
      dcov_gamma_exact(rx, ry, index = 1, legacy_index = TRUE)$p.value
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
  }
  list(adjacency = adjacency, pMax = pmax, sepsets = sepsets,
       n.edgetests = n_edge_tests, task_count = task_count,
       residual_requests = residual_requests)
}

ref_counts <- new.env(parent = emptyenv())
ref_counts$level_calls <- 0L
ref_counts$request_count <- 0L
reference <- reference_with_residual_provider(
  make_residual_provider(ref_counts)
)

native_counts <- new.env(parent = emptyenv())
native_counts$level_calls <- 0L
native_counts$request_count <- 0L
native <- precision_run_skeleton_residual_provider_native(
  data = data,
  alpha = alpha,
  max_conditioning_size = max_conditioning_size,
  residual_provider = make_residual_provider(native_counts),
  index = 1,
  legacy_index = TRUE,
  trace_level = "full"
)

assert_true(identical(native$adjacency, reference$adjacency),
            "native residual-provider skeleton adjacency should match R reference")
assert_true(max(abs(native$pMax - reference$pMax)) < 1e-10,
            "native residual-provider skeleton pMax should match R reference")
assert_true(identical(as.integer(native$n.edgetests),
                      as.integer(reference$n.edgetests)),
            "native residual-provider skeleton n.edgetests should match R reference")
assert_true(compare_sepsets(native$sepsets, reference$sepsets),
            "native residual-provider skeleton sepsets should match R reference")
assert_true(native_counts$level_calls > 0L,
            "native residual provider should be called for conditional levels")
assert_true(native_counts$request_count == reference$residual_requests,
            "native residual provider should receive the unique residual requests")
assert_true(identical(as.integer(native$summary$residual_provider_request_count),
                      as.integer(reference$residual_requests)),
            "native summary should count residual provider requests")
assert_true(identical(as.integer(native$summary$ci_native_count),
                      as.integer(reference$task_count)),
            "native summary should count planned native dCov tasks")
assert_true(any(native$tasks$conditioning_size > 0L),
            "native residual-provider trace should include conditional tasks")
assert_true(identical(native$summary$residual_backend,
                      "provider-legacy-mgcv"),
            "native summary should record provider residual backend")

cat("PASS skeleton native residual-provider exact dCov\n")
