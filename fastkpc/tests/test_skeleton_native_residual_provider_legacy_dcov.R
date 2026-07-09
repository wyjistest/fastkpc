source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

missing <- c("mgcv", "RSpectra")[
  !vapply(c("mgcv", "RSpectra"), requireNamespace, logical(1),
          quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP skeleton native residual provider legacy dCov: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = "spectra")
old_native_batch <- Sys.getenv("FASTKPC_NATIVE_LEGACY_DCOV_BATCH",
                               unset = NA_character_)
on.exit({
  if (is.na(old_native_batch)) {
    Sys.unsetenv("FASTKPC_NATIVE_LEGACY_DCOV_BATCH")
  } else {
    Sys.setenv(FASTKPC_NATIVE_LEGACY_DCOV_BATCH = old_native_batch)
  }
}, add = TRUE)
Sys.setenv(FASTKPC_NATIVE_LEGACY_DCOV_BATCH = "level")

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

legacy_dcov_p <- function(x, y, numCol, index) {
  fastkpc_legacy_dcov_gamma_cpp_oracle(
    x = x,
    y = y,
    numCol = numCol,
    index = index
  )$p.value
}

set.seed(5302)
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
      legacy_dcov_p(rx, ry, numCol = numCol, index = index)
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
native <- precision_run_skeleton_residual_provider_legacy_dcov_native(
  data = data,
  alpha = alpha,
  max_conditioning_size = max_conditioning_size,
  residual_provider = make_residual_provider(native_counts),
  index = index,
  numCol = numCol,
  trace_level = "full"
)

assert_true(identical(native$adjacency, reference$adjacency),
            "native residual-provider legacy dCov adjacency should match R reference")
assert_true(max(abs(native$pMax - reference$pMax)) < 1e-10,
            "native residual-provider legacy dCov pMax should match R reference")
assert_true(identical(as.integer(native$n.edgetests),
                      as.integer(reference$n.edgetests)),
            "native residual-provider legacy dCov n.edgetests should match R reference")
assert_true(compare_sepsets(native$sepsets, reference$sepsets),
            "native residual-provider legacy dCov sepsets should match R reference")
assert_true(native_counts$level_calls > 0L,
            "native residual provider should be called for conditional levels")
assert_true(native_counts$request_count == reference$residual_requests,
            "native residual provider should receive the unique residual requests")
assert_true(identical(as.integer(native$summary$residual_provider_request_count),
                      as.integer(reference$residual_requests)),
            "native summary should count residual provider requests")
assert_true(identical(native$summary$residual_provider_contract,
                      "level-residual-matrix-v1"),
            "native summary should record residual provider contract")
assert_true(identical(native$summary$residual_provider_response_mode,
                      "matrix"),
            "native summary should record matrix residual provider response mode")
assert_true(identical(native$summary$residual_provider_response_backend,
                      "matrix-provider"),
            "native summary should record matrix residual provider backend")
assert_true(identical(as.integer(native$summary$residual_provider_batch_count),
                      as.integer(native$summary$residual_provider_level_count)),
            "native summary should count one residual provider batch per conditional level")
assert_true(native$summary$residual_provider_batch_max_requests >= 1L,
            "native summary should record residual provider max batch requests")
assert_true(native$summary$residual_provider_batch_mean_requests >= 1,
            "native summary should record residual provider mean batch requests")
assert_true(identical(as.integer(native$summary$residual_provider_matrix_cell_count),
                      as.integer(nrow(data) * reference$residual_requests)),
            "native summary should record residual provider matrix cell payload")
assert_true(identical(as.integer(native$summary$ci_native_count),
                      as.integer(reference$task_count)),
            "native summary should count planned native legacy dCov tasks")
assert_true(identical(as.integer(native$summary$legacy_dcov_native_count),
                      as.integer(reference$task_count)),
            "native summary should count legacy dCov native tasks")
assert_true(any(native$tasks$conditioning_size > 0L),
            "native residual-provider legacy dCov trace should include conditional tasks")
assert_true(identical(native$summary$ci_backend,
                      "native-legacy-dcov.gamma"),
            "native summary should record legacy dCov backend")
assert_true(identical(native$summary$residual_backend,
                      "provider-legacy-mgcv"),
            "native summary should record provider residual backend")
assert_true(isTRUE(native$summary$legacy_dcov_native_batch_enabled),
            "native legacy dCov should report level batch mode")
expected_batch_count <- sum(as.integer(native$levels$tasks_planned) > 0L)
assert_true(identical(as.integer(native$summary$legacy_dcov_native_batch_count),
                      as.integer(expected_batch_count)),
            "native legacy dCov should batch each nonempty level")
assert_true(identical(as.integer(native$summary$legacy_dcov_native_batch_pair_count),
                      as.integer(native$summary$legacy_dcov_native_count)),
            "native legacy dCov batch pair count should match native task count")
assert_true(identical(as.integer(native$summary$legacy_dcov_native_batch_workspace_reuse_count),
                      as.integer(native$summary$legacy_dcov_native_batch_count)),
            "native legacy dCov should report workspace reuse per native batch")
assert_true(identical(as.integer(native$summary$legacy_dcov_native_batch_distance_workspace_reuse_count),
                      2L * as.integer(native$summary$legacy_dcov_native_batch_pair_count)),
            "native legacy dCov should report distance workspace reuse per pair")
assert_true(identical(as.integer(native$summary$legacy_dcov_native_batch_statistic_moment_workspace_reuse_count),
                      3L * as.integer(native$summary$legacy_dcov_native_batch_pair_count)),
            "native legacy dCov should report statistic/moment workspace reuse per pair")
assert_true(identical(as.integer(native$summary$legacy_dcov_native_batch_lowrank_output_workspace_reuse_count),
                      2L * as.integer(native$summary$legacy_dcov_native_batch_pair_count)),
            "native legacy dCov should report lowrank output workspace reuse per pair")
assert_true(identical(as.integer(native$summary$legacy_dcov_native_batch_lowrank_eig_workspace_reuse_count),
                      2L * as.integer(native$summary$legacy_dcov_native_batch_pair_count)),
            "native legacy dCov should report lowrank eig workspace reuse per pair")
assert_true(identical(as.integer(native$summary$legacy_dcov_native_batch_oracle_column_copy_count),
                      0L),
            "native legacy dCov batch oracle should avoid per-column internal copies")
assert_true(isTRUE(native$summary$legacy_dcov_native_batch_direct_input_enabled),
            "native legacy dCov batch should use direct residual/data column inputs")
assert_true(identical(as.integer(native$summary$legacy_dcov_native_batch_column_materialize_count),
                      0L),
            "native legacy dCov should avoid C++ skeleton matrix materialization columns")
assert_true(native$summary$legacy_dcov_native_batch_ms > 0,
            "native legacy dCov should report batch elapsed time")

cat("PASS skeleton native residual-provider legacy dCov\n")
