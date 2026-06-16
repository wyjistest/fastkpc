source("fastkpc/R/fast_kpc.R")

fastkpc_hsic_cuda_fixture <- function(seed, n) {
  set.seed(seed)
  x <- seq(-2, 2, length.out = n)
  y <- sin(x) + rnorm(n, sd = 0.06)
  list(x = x, y = y)
}

fastkpc_elapsed_value <- function(expr) {
  start <- proc.time()[["elapsed"]]
  value <- force(expr)
  elapsed <- proc.time()[["elapsed"]] - start
  list(value = value, elapsed_sec = max(elapsed, .Machine$double.eps))
}

validate_hsic_cuda_gamma_kernel <- function(seed = 401, n = 128, sig = 1) {
  fixture <- fastkpc_hsic_cuda_fixture(seed, n)
  cpu <- fast_hsic_gamma_cpp(fixture$x, fixture$y, sig = sig)
  load_fastkpc_cuda_native()
  cuda <- fast_hsic_gamma_cuda(fixture$x, fixture$y, sig = sig)
  list(
    metrics = list(
      statistic_abs_diff = abs(cpu$statistic - cuda$statistic),
      pvalue_abs_diff = abs(cpu$p.value - cuda$p.value),
      ci_backend = cuda$backend,
      cuda_batches = as.integer(cuda$diagnostics$batches %||% 1L),
      cuda_pairs = as.integer(cuda$diagnostics$pairs %||% 1L),
      bytes_allocated = as.numeric(cuda$diagnostics$bytes_allocated %||% 0)
    ),
    cpu = cpu,
    cuda = cuda
  )
}

validate_hsic_cuda_permutation_kernel <- function(seed = 402, n = 96,
                                                  replicates = 50) {
  fixture <- fastkpc_hsic_cuda_fixture(seed, n)
  load_fastkpc_cuda_native()
  first <- fast_hsic_perm_cuda(fixture$x, fixture$y, sig = 1,
                               replicates = as.integer(replicates),
                               seed = as.integer(seed),
                               include_observed = TRUE)
  second <- fast_hsic_perm_cuda(fixture$x, fixture$y, sig = 1,
                                replicates = as.integer(replicates),
                                seed = as.integer(seed),
                                include_observed = TRUE)
  replicate_diff <- max(abs(as.numeric(first$replicates) -
                              as.numeric(second$replicates)))
  list(
    metrics = list(
      fixed_seed_repeats = replicate_diff < 1e-12 &&
        identical(first$p.value, second$p.value),
      replicate_max_abs_diff = replicate_diff,
      pvalue_abs_diff = abs(first$p.value - second$p.value),
      ci_backend = first$backend,
      cuda_batches = as.integer(first$diagnostics$batches %||% 1L),
      cuda_pairs = as.integer(first$diagnostics$pairs %||% 1L),
      bytes_allocated = as.numeric(first$diagnostics$bytes_allocated %||% 0)
    ),
    first = first,
    second = second
  )
}

compare_hsic_cuda_cpu_skeleton <- function(seed = 403, n = 128) {
  set.seed(seed)
  z <- runif(n, -2, 2)
  data <- cbind(
    x1 = sin(z) + rnorm(n, sd = 0.07),
    x2 = z + rnorm(n, sd = 0.07),
    x3 = z^2 + rnorm(n, sd = 0.07),
    x4 = rnorm(n)
  )
  cpu <- fast_kpc(data, alpha = 0.2, max_conditioning_size = 1L,
                  engine = "cpu", graph_stage = "skeleton",
                  residual_backend = "linear", ci_method = "hsic.gamma",
                  hsic_params = list(sig = 1))
  cuda <- fast_kpc(data, alpha = 0.2, max_conditioning_size = 1L,
                   engine = "cuda", graph_stage = "skeleton",
                   residual_backend = "linear", residual_device = "cuda",
                   scheduler = "legacy", ci_method = "hsic.gamma",
                   hsic_params = list(sig = 1))
  pmax_diff <- max(abs(as.numeric(cpu$skeleton$pMax) -
                         as.numeric(cuda$skeleton$pMax)))
  diag <- cuda$skeleton$ci_diagnostics %||% list()
  list(
    metrics = list(
      adjacency_identical = identical(cpu$skeleton$adjacency,
                                      cuda$skeleton$adjacency),
      max_abs_pmax_diff = pmax_diff,
      ci_backend = cuda$skeleton$ci_backend,
      cuda_batches = as.integer(diag$ci_hsic_cuda_batches %||% 0L),
      cuda_pairs = as.integer(diag$ci_hsic_cuda_pairs %||% 0L)
    ),
    cpu = cpu,
    cuda = cuda
  )
}

benchmark_hsic_cuda_backends <- function(seed = 404,
                                         n_values = c(64, 128, 256),
                                         methods = c("hsic.gamma", "hsic.perm"),
                                         repeats = 3) {
  rows <- list()
  for (n in n_values) {
    fixture <- fastkpc_hsic_cuda_fixture(seed + as.integer(n), n)
    for (method in methods) {
      for (repeat_id in seq_len(repeats)) {
        cpu <- fastkpc_elapsed_value({
          if (method == "hsic.gamma") {
            fast_hsic_gamma_cpp(fixture$x, fixture$y, sig = 1)
          } else {
            fast_hsic_perm_cpp(fixture$x, fixture$y, sig = 1,
                               replicates = 30L, seed = seed,
                               include_observed = TRUE)
          }
        })
        rows[[length(rows) + 1L]] <- data.frame(
          n = as.integer(n),
          ci_method = method,
          backend = "cpu",
          repeat_id = as.integer(repeat_id),
          elapsed_sec = cpu$elapsed_sec,
          ci_backend = "native-cpu",
          cuda_batches = 0L,
          cuda_pairs = 0L,
          bytes_allocated = 0,
          stringsAsFactors = FALSE
        )

        cuda <- fastkpc_elapsed_value({
          if (method == "hsic.gamma") {
            fast_hsic_gamma_cuda(fixture$x, fixture$y, sig = 1)
          } else {
            fast_hsic_perm_cuda(fixture$x, fixture$y, sig = 1,
                                replicates = 30L, seed = seed,
                                include_observed = TRUE)
          }
        })
        cuda_diag <- cuda$value$diagnostics %||% list()
        rows[[length(rows) + 1L]] <- data.frame(
          n = as.integer(n),
          ci_method = method,
          backend = "cuda",
          repeat_id = as.integer(repeat_id),
          elapsed_sec = cuda$elapsed_sec,
          ci_backend = cuda$value$backend %||% "cuda-hsic",
          cuda_batches = as.integer(cuda_diag$batches %||% 1L),
          cuda_pairs = as.integer(cuda_diag$pairs %||% 1L),
          bytes_allocated = as.numeric(cuda_diag$bytes_allocated %||% 0),
          stringsAsFactors = FALSE
        )
      }
    }
  }
  timings <- do.call(rbind, rows)
  means <- aggregate(elapsed_sec ~ n + ci_method + backend, timings, mean)
  cpu <- means[means$backend == "cpu", c("n", "ci_method", "elapsed_sec")]
  cuda <- means[means$backend == "cuda", c("n", "ci_method", "elapsed_sec")]
  names(cpu)[names(cpu) == "elapsed_sec"] <- "cpu_elapsed_sec"
  names(cuda)[names(cuda) == "elapsed_sec"] <- "cuda_elapsed_sec"
  summary <- merge(cpu, cuda, by = c("n", "ci_method"), all = TRUE,
                   sort = FALSE)
  summary$speedup <- summary$cpu_elapsed_sec / summary$cuda_elapsed_sec
  list(timings = timings, summary = summary)
}
