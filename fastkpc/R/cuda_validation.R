validate_cuda_dcov_batch <- function(n = 300, batch = 16, index = 1,
                                     legacy_index = TRUE, seed = 1) {
  source("fastkpc/R/dcov_exact.R")
  source("fastkpc/R/cuda_native.R")

  set.seed(seed)
  x <- matrix(stats::rnorm(n * batch), n, batch)
  y <- x * rep(seq(0.05, 0.25, length.out = batch), each = n) +
    matrix(stats::rnorm(n * batch), n, batch)
  cuda <- fast_dcov_batch_cuda(x, y, index = index, legacy_index = legacy_index)

  cpu_p <- numeric(batch)
  cpu_nV2 <- numeric(batch)
  for (k in seq_len(batch)) {
    cpu <- dcov_gamma_exact(x[, k], y[, k], index = index,
                            legacy_index = legacy_index)
    cpu_p[k] <- cpu$p.value
    cpu_nV2[k] <- unname(cpu$statistic)
  }

  list(
    max_abs_p_diff = max(abs(cuda$p.value - cpu_p)),
    max_abs_nV2_diff = max(abs(cuda$nV2 - cpu_nV2)),
    all_p_values_finite = all(is.finite(cuda$p.value)),
    all_p_values_in_unit_interval = all(cuda$p.value >= 0 & cuda$p.value <= 1)
  )
}

validate_cuda_skeleton_scenario <- function(seed = 4, n = 80, alpha = 0.2,
                                            max_conditioning_size = 1) {
  source("fastkpc/R/native.R")
  source("fastkpc/R/cuda_native.R")
  source("fastkpc/R/diff_report.R")
  source("fastkpc/R/legacy_runner.R")

  scenario <- fastkpc_fixed_scenario(seed = seed, n = n)
  cpu <- fast_skeleton_cpp(scenario$data, alpha = alpha,
                           max_conditioning_size = max_conditioning_size)
  cuda <- fast_skeleton_cuda(scenario$data, alpha = alpha,
                             max_conditioning_size = max_conditioning_size)
  diff <- summarize_graph_diff(cpu, cuda)

  sepsets_identical <- diff$sepsets$differing_count == 0
  list(
    diff = diff,
    max_abs_pmax_diff = max(abs(cuda$pMax - cpu$pMax)),
    adjacency_identical = identical(cuda$adjacency, cpu$adjacency),
    sepsets_identical = sepsets_identical,
    n_edgetests_identical = identical(cuda$n.edgetests, cpu$n.edgetests)
  )
}
