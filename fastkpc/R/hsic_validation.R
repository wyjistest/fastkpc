source("fastkpc/R/fast_kpc.R")

fastkpc_hsic_fixture <- function(seed, n) {
  set.seed(seed)
  x <- seq(-2, 2, length.out = n)
  y <- sin(x) + rnorm(n, sd = 0.08)
  list(x = x, y = y)
}

validate_hsic_native_gamma <- function(seed = 221, n = 60, sig = 1) {
  fixture <- fastkpc_hsic_fixture(seed, n)
  native <- fast_hsic_gamma_cpp(fixture$x, fixture$y, sig = sig)
  list(
    metrics = list(
      native_p_value = native$p.value,
      native_statistic = native$statistic,
      finite = is.finite(native$p.value) && is.finite(native$statistic)
    ),
    native = native
  )
}

validate_hsic_native_permutation <- function(seed = 222, n = 50,
                                             sig = 1, replicates = 50L) {
  fixture <- fastkpc_hsic_fixture(seed, n)
  a <- fast_hsic_perm_cpp(fixture$x, fixture$y, sig = sig,
                          replicates = replicates, seed = seed,
                          include_observed = TRUE)
  b <- fast_hsic_perm_cpp(fixture$x, fixture$y, sig = sig,
                          replicates = replicates, seed = seed,
                          include_observed = TRUE)
  list(
    metrics = list(
      p_value = a$p.value,
      replicates_identical = identical(a$replicates, b$replicates),
      replicate_count = length(a$replicates)
    ),
    first = a,
    second = b
  )
}

compare_ci_methods <- function(seed = 223, n = 70,
                               methods = c("dcc.gamma", "hsic.gamma")) {
  set.seed(seed)
  z <- runif(n, -2, 2)
  data <- cbind(
    x1 = sin(z) + rnorm(n, sd = 0.08),
    x2 = cos(z) + rnorm(n, sd = 0.08),
    x3 = rnorm(n)
  )
  runs <- lapply(methods, function(method) {
    fast_kpc(data, alpha = 0.2, max_conditioning_size = 1,
             engine = "cpu", graph_stage = "skeleton",
             residual_backend = "linear", ci_method = method)
  })
  names(runs) <- methods
  list(methods = methods, runs = runs)
}

compare_hsic_cpu_cuda_resolution <- function(seed = 224, n = 90) {
  set.seed(seed)
  z <- runif(n, -2, 2)
  data <- cbind(
    x1 = sin(z) + rnorm(n, sd = 0.08),
    x2 = cos(z) + rnorm(n, sd = 0.08),
    x3 = rnorm(n)
  )
  cpu <- fast_kpc(data, alpha = 0.2, max_conditioning_size = 1,
                  engine = "cpu", graph_stage = "skeleton",
                  residual_backend = "linear", ci_method = "hsic.gamma")
  cuda <- fast_kpc(data, alpha = 0.2, max_conditioning_size = 1,
                   engine = "cuda", graph_stage = "skeleton",
                   residual_backend = "linear", residual_device = "cuda",
                   ci_method = "hsic.gamma")
  list(
    metrics = list(
      adjacency_identical = identical(cpu$skeleton$adjacency,
                                      cuda$skeleton$adjacency),
      max_abs_pmax_diff = max(abs(as.numeric(cpu$skeleton$pMax) -
                                    as.numeric(cuda$skeleton$pMax))),
      pmax_close = max(abs(as.numeric(cpu$skeleton$pMax) -
                             as.numeric(cuda$skeleton$pMax))) < 1e-12,
      cuda_ci_backend = cuda$skeleton$ci_backend
    ),
    cpu = cpu,
    cuda = cuda
  )
}

benchmark_hsic_backends <- function(seed = 225, n = 120, repeats = 2) {
  fixture <- fastkpc_hsic_fixture(seed, n)
  rows <- list()
  for (method in c("hsic.gamma", "hsic.perm")) {
    for (i in seq_len(repeats)) {
      start <- proc.time()[["elapsed"]]
      if (method == "hsic.gamma") {
        fast_hsic_gamma_cpp(fixture$x, fixture$y, sig = 1)
      } else {
        fast_hsic_perm_cpp(fixture$x, fixture$y, sig = 1,
                           replicates = 30L, seed = seed,
                           include_observed = TRUE)
      }
      elapsed <- proc.time()[["elapsed"]] - start
      rows[[length(rows) + 1L]] <- data.frame(
        ci_method = method,
        repeat_id = i,
        elapsed_sec = max(elapsed, .Machine$double.eps),
        stringsAsFactors = FALSE
      )
    }
  }
  timings <- do.call(rbind, rows)
  summary <- aggregate(elapsed_sec ~ ci_method, timings, mean)
  names(summary)[names(summary) == "elapsed_sec"] <- "mean_elapsed_sec"
  list(timings = timings, summary = summary)
}
