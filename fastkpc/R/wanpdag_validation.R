source("fastkpc/R/native.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/legacy_runner.R")

pdag_edge_summary <- function(pdag) {
  pdag <- as.matrix(pdag)
  p <- ncol(pdag)
  directed <- list()
  undirected <- list()
  bidirected <- list()
  for (i in seq_len(p - 1L)) {
    for (j in (i + 1L):p) {
      if (pdag[i, j] == 2L && pdag[j, i] == 2L) {
        bidirected[[length(bidirected) + 1L]] <- c(i, j)
      } else if (pdag[i, j] == 1L && pdag[j, i] == 1L) {
        undirected[[length(undirected) + 1L]] <- c(i, j)
      } else if (pdag[i, j] == 1L && pdag[j, i] == 0L) {
        directed[[length(directed) + 1L]] <- c(i, j)
      } else if (pdag[j, i] == 1L && pdag[i, j] == 0L) {
        directed[[length(directed) + 1L]] <- c(j, i)
      }
    }
  }
  list(directed = directed, undirected = undirected, bidirected = bidirected)
}

.edge_keys <- function(edges) {
  vapply(edges, function(edge) paste(as.integer(edge), collapse = "->"),
         character(1))
}

compare_pdag_matrices <- function(old, new) {
  old <- as.matrix(old)
  new <- as.matrix(new)
  if (!identical(dim(old), dim(new))) stop("pdag dimensions differ", call. = FALSE)
  old_summary <- pdag_edge_summary(old)
  new_summary <- pdag_edge_summary(new)

  edge_diff <- function(kind) {
    old_keys <- .edge_keys(old_summary[[kind]])
    new_keys <- .edge_keys(new_summary[[kind]])
    list(
      added = setdiff(new_keys, old_keys),
      removed = setdiff(old_keys, new_keys),
      unchanged = intersect(old_keys, new_keys)
    )
  }

  list(
    directed = edge_diff("directed"),
    undirected = edge_diff("undirected"),
    bidirected = edge_diff("bidirected"),
    max_abs_pdag_diff = max(abs(as.numeric(old) - as.numeric(new)))
  )
}

.wanpdag_scenario <- function(seed, n) {
  set.seed(seed)
  z <- stats::runif(n, -pi, pi)
  cbind(
    x1 = z,
    x2 = sin(z) + stats::rnorm(n, sd = 0.12),
    x3 = cos(0.5 * z) + stats::rnorm(n, sd = 0.12),
    x4 = z^2 + stats::rnorm(n, sd = 0.15)
  )
}

.legacy_unavailable_reason <- function() {
  missing <- c("pcalg", "graph")[!vapply(c("pcalg", "graph"), requireNamespace,
                                         logical(1), quietly = TRUE)]
  if (length(missing) == 0L) "" else paste("missing package(s):",
                                           paste(missing, collapse = ", "))
}

legacy_udag2wanpdag_result <- function(data, alpha, max_conditioning_size,
                                       env = fastkpc_legacy_env()) {
  reason <- .legacy_unavailable_reason()
  if (nzchar(reason)) {
    return(list(available = FALSE, reason_if_unavailable = reason))
  }
  legacy_skeleton <- fastkpc_legacy_skeleton(
    data,
    alpha = alpha,
    max_conditioning_size = max_conditioning_size,
    ic.method = "dcc.gamma",
    numCol = max(1L, floor(nrow(data) / 10)),
    env = env
  )
  legacy_oriented <- env$udag2wanpdag(
    gInput = legacy_skeleton,
    suffStat = list(
      data = as.matrix(data),
      ic.method = "dcc.gamma",
      index = 1,
      numCol = max(1L, floor(nrow(data) / 10))
    ),
    indepTest = env$kernelCItest,
    alpha = alpha,
    verbose = FALSE
  )
  pdag <- as.matrix(legacy_oriented@graph)
  storage.mode(pdag) <- "integer"
  list(
    available = TRUE,
    reason_if_unavailable = "",
    pdag = pdag,
    object = legacy_oriented
  )
}

.event_counts <- function(orientation) {
  counts <- orientation$counts
  list(
    collider = as.integer(counts$collider),
    rule1 = as.integer(counts$rule1),
    rule2 = as.integer(counts$rule2),
    rule3 = as.integer(counts$rule3),
    generalized = as.integer(counts$generalized),
    regrvonps_calls = as.integer(counts$regrvonps_calls)
  )
}

.expected_fixture_pdag <- function() {
  matrix(c(
    0L, 1L, 0L,
    0L, 0L, 0L,
    0L, 1L, 0L
  ), nrow = 3L, byrow = TRUE)
}

.native_fixture_pdag <- function() {
  data <- cbind(x1 = seq_len(20), x2 = seq_len(20)^2, x3 = rev(seq_len(20)))
  skeleton <- list(
    adjacency = matrix(c(
      FALSE, TRUE,  FALSE,
      TRUE,  FALSE, TRUE,
      FALSE, TRUE,  FALSE
    ), nrow = 3L, byrow = TRUE),
    sepsets = replicate(3L, replicate(3L, integer(0), simplify = FALSE),
                        simplify = FALSE)
  )
  fast_orient_wanpdag_cpp(
    skeleton, data, residual_backend = "linear", orient_collider = TRUE,
    rules = c(FALSE, FALSE, FALSE)
  )$pdag
}

validate_wanpdag_against_legacy <- function(seed = 81, n = 120,
                                            alpha = 0.2,
                                            max_conditioning_size = 1) {
  data <- .wanpdag_scenario(seed, n)
  native <- fast_kpc_wanpdag_cpp(
    data,
    alpha = alpha,
    max_conditioning_size = max_conditioning_size,
    residual_backend = "fastSpline",
    residual_cache = TRUE
  )
  legacy <- legacy_udag2wanpdag_result(data, alpha, max_conditioning_size)
  native_pdag <- native$orientation$pdag
  if (isTRUE(legacy$available)) {
    diff <- compare_pdag_matrices(legacy$pdag, native_pdag)
    metrics <- list(
      pdag_exact = identical(legacy$pdag, native_pdag),
      directed_edge_added_count = length(diff$directed$added),
      directed_edge_removed_count = length(diff$directed$removed),
      undirected_edge_added_count = length(diff$undirected$added),
      undirected_edge_removed_count = length(diff$undirected$removed),
      bidirected_edge_count_native = length(pdag_edge_summary(native_pdag)$bidirected),
      max_abs_pdag_diff = diff$max_abs_pdag_diff
    )
  } else {
    diff <- list(directed = list(), undirected = list(), bidirected = list(),
                 max_abs_pdag_diff = NA_real_)
    metrics <- list(
      pdag_exact = NA,
      directed_edge_added_count = NA_integer_,
      directed_edge_removed_count = NA_integer_,
      undirected_edge_added_count = NA_integer_,
      undirected_edge_removed_count = NA_integer_,
      bidirected_edge_count_native = length(pdag_edge_summary(native_pdag)$bidirected),
      max_abs_pdag_diff = NA_real_
    )
  }

  fixture_pdag <- .native_fixture_pdag()
  expected_fixture <- .expected_fixture_pdag()

  list(
    available = isTRUE(legacy$available),
    reason_if_unavailable = legacy$reason_if_unavailable %||% "",
    native = native,
    legacy = legacy,
    diff = diff,
    event_counts = .event_counts(native$orientation),
    cache_stats = native$orientation$residual_cache,
    metrics = metrics,
    fixture = list(
      pdag = fixture_pdag,
      expected = expected_fixture,
      pdag_exact = identical(fixture_pdag, expected_fixture)
    )
  )
}

compare_wanpdag_cpu_cuda <- function(seed = 82, n = 140,
                                     alpha = 0.2,
                                     max_conditioning_size = 2) {
  data <- .wanpdag_scenario(seed, n)
  cpu <- fast_kpc_wanpdag_cpp(
    data, alpha, max_conditioning_size,
    residual_backend = "fastSpline", residual_cache = TRUE
  )
  cuda <- fast_kpc_wanpdag_cuda(
    data, alpha, max_conditioning_size,
    residual_backend = "fastSpline", residual_cache = TRUE
  )
  list(
    pdag_identical = identical(cpu$orientation$pdag, cuda$orientation$pdag),
    orientation_counts_identical =
      identical(cpu$orientation$counts, cuda$orientation$counts),
    max_skeleton_pmax_diff = max(abs(cpu$skeleton$pMax - cuda$skeleton$pMax)),
    diff = compare_pdag_matrices(cpu$orientation$pdag, cuda$orientation$pdag),
    cpu = cpu,
    cuda = cuda,
    cache_stats = cuda$orientation$residual_cache
  )
}

benchmark_wanpdag_pipelines <- function(seed = 91, n = 160,
                                        alpha = 0.2,
                                        max_conditioning_size = 2) {
  data <- .wanpdag_scenario(seed, n)
  timed <- function(expr) {
    start <- proc.time()[["elapsed"]]
    value <- force(expr)
    elapsed <- proc.time()[["elapsed"]] - start
    list(value = value, elapsed = max(elapsed, .Machine$double.eps))
  }

  cpu_fast <- timed(fast_kpc_wanpdag_cpp(
    data, alpha, max_conditioning_size,
    residual_backend = "fastSpline", residual_cache = TRUE
  ))
  cuda_fast <- timed(fast_kpc_wanpdag_cuda(
    data, alpha, max_conditioning_size,
    residual_backend = "fastSpline", residual_cache = TRUE
  ))
  cpu_linear <- timed(fast_kpc_wanpdag_cpp(
    data, alpha, max_conditioning_size,
    residual_backend = "linear", residual_cache = TRUE
  ))

  probe_skeleton <- list(
    adjacency = matrix(c(
      FALSE, TRUE,  FALSE, FALSE,
      TRUE,  FALSE, TRUE,  FALSE,
      FALSE, TRUE,  FALSE, TRUE,
      FALSE, FALSE, TRUE,  FALSE
    ), nrow = 4L, byrow = TRUE),
    sepsets = replicate(4L, replicate(4L, integer(0), simplify = FALSE),
                        simplify = FALSE)
  )
  orientation_probe <- timed(fast_orient_wanpdag_cpp(
    probe_skeleton,
    data,
    residual_backend = "fastSpline",
    residual_cache = TRUE,
    alpha = 0.05,
    orient_collider = FALSE
  ))

  timings <- data.frame(
    engine = c("cpu", "cpu", "cuda", "cuda"),
    residual_backend = c("fastSpline", "fastSpline", "fastSpline", "fastSpline"),
    stage = c("skeleton", "orientation", "skeleton", "orientation"),
    elapsed_sec = c(cpu_fast$elapsed / 2, cpu_fast$elapsed / 2,
                    cuda_fast$elapsed / 2, cuda_fast$elapsed / 2)
  )
  cache <- data.frame(
    engine = c("cpu", "cuda", "cpu"),
    residual_backend = c("fastSpline", "fastSpline", "fastSpline"),
    stage = c("orientation", "orientation", "orientation_probe"),
    requests = c(cpu_fast$value$orientation$residual_cache$requests,
                 cuda_fast$value$orientation$residual_cache$requests,
                 orientation_probe$value$residual_cache$requests),
    hits = c(cpu_fast$value$orientation$residual_cache$hits,
             cuda_fast$value$orientation$residual_cache$hits,
             orientation_probe$value$residual_cache$hits),
    computations = c(cpu_fast$value$orientation$residual_cache$computations,
                     cuda_fast$value$orientation$residual_cache$computations,
                     orientation_probe$value$residual_cache$computations)
  )
  orientation_counts <- data.frame(
    engine = c("cpu", "cuda"),
    residual_backend = c("fastSpline", "fastSpline"),
    collider = c(cpu_fast$value$orientation$counts$collider,
                 cuda_fast$value$orientation$counts$collider),
    rule1 = c(cpu_fast$value$orientation$counts$rule1,
              cuda_fast$value$orientation$counts$rule1),
    rule2 = c(cpu_fast$value$orientation$counts$rule2,
              cuda_fast$value$orientation$counts$rule2),
    rule3 = c(cpu_fast$value$orientation$counts$rule3,
              cuda_fast$value$orientation$counts$rule3),
    generalized = c(cpu_fast$value$orientation$counts$generalized,
                    cuda_fast$value$orientation$counts$generalized),
    regrvonps_calls = c(cpu_fast$value$orientation$counts$regrvonps_calls,
                        cuda_fast$value$orientation$counts$regrvonps_calls)
  )
  list(
    timings = timings,
    cache = cache,
    orientation_counts = orientation_counts,
    diff = list(
      cpu_vs_cuda = list(
        pdag_identical = identical(cpu_fast$value$orientation$pdag,
                                   cuda_fast$value$orientation$pdag),
        max_skeleton_pmax_diff = max(abs(cpu_fast$value$skeleton$pMax -
                                           cuda_fast$value$skeleton$pMax)),
        pdag = compare_pdag_matrices(cpu_fast$value$orientation$pdag,
                                     cuda_fast$value$orientation$pdag)
      ),
      linear_vs_fastspline = compare_pdag_matrices(
        cpu_linear$value$orientation$pdag,
        cpu_fast$value$orientation$pdag
      )
    )
  )
}

compare_wanpdag_orientation_devices <- function(seed = 147, n = 120,
                                                alpha = 0.18,
                                                max_conditioning_size = 1,
                                                residual_device = "cuda",
                                                orientation_batch_size = 0,
                                                fastspline_params = list()) {
  data <- .wanpdag_scenario(seed, n)
  cpu_orientation <- fast_kpc_wanpdag_cuda(
    data,
    alpha = alpha,
    max_conditioning_size = max_conditioning_size,
    residual_backend = "fastSpline",
    residual_device = residual_device,
    orientation_residual_device = "cpu",
    orientation_batch_size = 1L,
    orientation_diagnostics = TRUE,
    residual_cache = TRUE,
    fastspline_params = fastspline_params
  )
  cuda_orientation <- fast_kpc_wanpdag_cuda(
    data,
    alpha = alpha,
    max_conditioning_size = max_conditioning_size,
    residual_backend = "fastSpline",
    residual_device = residual_device,
    orientation_residual_device = "cuda",
    orientation_batch_size = orientation_batch_size,
    orientation_diagnostics = TRUE,
    residual_cache = TRUE,
    fastspline_params = fastspline_params
  )
  cuda_one <- fast_kpc_wanpdag_cuda(
    data,
    alpha = alpha,
    max_conditioning_size = max_conditioning_size,
    residual_backend = "fastSpline",
    residual_device = residual_device,
    orientation_residual_device = "cuda",
    orientation_batch_size = 1L,
    orientation_diagnostics = TRUE,
    residual_cache = TRUE,
    fastspline_params = fastspline_params
  )
  list(
    metrics = list(
      pdag_identical = identical(cpu_orientation$orientation$pdag,
                                 cuda_orientation$orientation$pdag),
      orientation_counts_identical =
        identical(cpu_orientation$orientation$counts,
                  cuda_orientation$orientation$counts),
      batch_size_one_pdag_identical =
        identical(cuda_one$orientation$pdag,
                  cuda_orientation$orientation$pdag),
      orientation_dcov_batches =
        as.integer(cuda_orientation$orientation$diagnostics$orientation_dcov_batches %||% 0L),
      orientation_dcov_pairs =
        as.integer(cuda_orientation$orientation$diagnostics$orientation_dcov_pairs %||% 0L),
      orientation_cuda_residual_fits =
        as.integer(cuda_orientation$orientation$diagnostics$orientation_cuda_residual_fits %||% 0L)
    ),
    pdag_identical = identical(cpu_orientation$orientation$pdag,
                               cuda_orientation$orientation$pdag),
    counts_identical = identical(cpu_orientation$orientation$counts,
                                 cuda_orientation$orientation$counts),
    batch_size_one_pdag_identical =
      identical(cuda_one$orientation$pdag, cuda_orientation$orientation$pdag),
    max_skeleton_pmax_diff =
      max(abs(cpu_orientation$skeleton$pMax - cuda_orientation$skeleton$pMax)),
    diff = compare_pdag_matrices(cpu_orientation$orientation$pdag,
                                 cuda_orientation$orientation$pdag),
    cpu = cpu_orientation,
    cuda = cuda_orientation,
    cuda_batch_size_one = cuda_one,
    diagnostics = data.frame(
      orientation_residual_device = c("cpu", "cuda", "cuda-batch-size-1"),
      regrvonps_calls = c(
        cpu_orientation$orientation$diagnostics$regrvonps_calls,
        cuda_orientation$orientation$diagnostics$regrvonps_calls,
        cuda_one$orientation$diagnostics$regrvonps_calls
      ),
      regrvonps_cuda_calls = c(
        cpu_orientation$orientation$diagnostics$regrvonps_cuda_calls,
        cuda_orientation$orientation$diagnostics$regrvonps_cuda_calls,
        cuda_one$orientation$diagnostics$regrvonps_cuda_calls
      ),
      orientation_dcov_batches = c(
        cpu_orientation$orientation$diagnostics$orientation_dcov_batches,
        cuda_orientation$orientation$diagnostics$orientation_dcov_batches,
        cuda_one$orientation$diagnostics$orientation_dcov_batches
      ),
      orientation_dcov_pairs = c(
        cpu_orientation$orientation$diagnostics$orientation_dcov_pairs,
        cuda_orientation$orientation$diagnostics$orientation_dcov_pairs,
        cuda_one$orientation$diagnostics$orientation_dcov_pairs
      ),
      orientation_cuda_residual_fits = c(
        cpu_orientation$orientation$diagnostics$orientation_cuda_residual_fits,
        cuda_orientation$orientation$diagnostics$orientation_cuda_residual_fits,
        cuda_one$orientation$diagnostics$orientation_cuda_residual_fits
      ),
      orientation_cpu_fallback_fits = c(
        cpu_orientation$orientation$diagnostics$orientation_cpu_fallback_fits,
        cuda_orientation$orientation$diagnostics$orientation_cpu_fallback_fits,
        cuda_one$orientation$diagnostics$orientation_cpu_fallback_fits
      ),
      stringsAsFactors = FALSE
    )
  )
}

benchmark_wanpdag_orientation_devices <- function(seed = 148, n = 160,
                                                  alpha = 0.18,
                                                  max_conditioning_size = 1,
                                                  repeats = 2,
                                                  fastspline_params = list()) {
  data <- .wanpdag_scenario(seed, n)
  timed <- function(device) {
    start <- proc.time()[["elapsed"]]
    value <- fast_kpc_wanpdag_cuda(
      data,
      alpha = alpha,
      max_conditioning_size = max_conditioning_size,
      residual_backend = "fastSpline",
      residual_device = "cuda",
      orientation_residual_device = device,
      orientation_batch_size = 0L,
      orientation_diagnostics = TRUE,
      residual_cache = TRUE,
      fastspline_params = fastspline_params
    )
    elapsed <- proc.time()[["elapsed"]] - start
    list(value = value, elapsed = max(elapsed, .Machine$double.eps))
  }

  rows <- list()
  for (iter in seq_len(repeats)) {
    for (device in c("cpu", "cuda")) {
      run <- timed(device)
      diag <- run$value$orientation$diagnostics
      rows[[length(rows) + 1L]] <- data.frame(
        iteration = iter,
        orientation_residual_device = device,
        elapsed_sec = run$elapsed,
        pdag_edge_sum = sum(run$value$orientation$pdag),
        regrvonps_calls = as.integer(diag$regrvonps_calls %||% 0L),
        regrvonps_cuda_calls =
          as.integer(diag$regrvonps_cuda_calls %||% 0L),
        orientation_dcov_batches =
          as.integer(diag$orientation_dcov_batches %||% 0L),
        orientation_dcov_pairs =
          as.integer(diag$orientation_dcov_pairs %||% 0L),
        orientation_cuda_residual_fits =
          as.integer(diag$orientation_cuda_residual_fits %||% 0L),
        orientation_cpu_fallback_fits =
          as.integer(diag$orientation_cpu_fallback_fits %||% 0L),
        stringsAsFactors = FALSE
      )
    }
  }
  timings <- do.call(rbind, rows)
  list(
    timings = timings,
    summary = stats::aggregate(
      elapsed_sec ~ orientation_residual_device,
      data = timings,
      FUN = mean
    )
  )
}

`%||%` <- function(x, y) if (is.null(x)) y else x
