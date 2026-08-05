source("fastkpc/R/fast_kpc.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

set.seed(1211)
n <- 56L
z <- runif(n, -2, 2)
data <- cbind(
  x1 = sin(z) + rnorm(n, sd = 0.1),
  x2 = z + rnorm(n, sd = 0.1),
  x3 = z^2 + rnorm(n, sd = 0.1),
  x4 = rnorm(n)
)
methods <- c("dcc.perm", "hsic.gamma", "hsic.perm")

load_fastkpc_cuda_native(rebuild = FALSE)

for (engine in c("cpu", "cuda")) {
  for (method in methods) {
    result <- fast_kpc(
      data,
      alpha = 0.2,
      max_conditioning_size = 1L,
      engine = engine,
      precision = "hybrid",
      tau = 0,
      graph_stage = "skeleton",
      ci_method = method,
      hsic_params = list(sig = 1),
      permutation_params = list(
        replicates = 13L,
        seed = 707L,
        include_observed = TRUE
      ),
      precision_trace_level = "full"
    )
    diagnostics <- result$skeleton$ci_diagnostics

    assert_true(result$skeleton$ci_method == method,
                paste(engine, method, "precision route lost ci_method"))
    assert_true(result$skeleton$n.edgetests[[2L]] > 0L,
                paste(engine, method, "did not execute conditional tests"))
    assert_true(nrow(result$diagnostics$precision_trace) ==
                  sum(result$skeleton$n.edgetests),
                paste(engine, method, "precision trace is incomplete"))

    if (method == "dcc.perm") {
      assert_true(diagnostics$ci_dcc_perm_tests > 0L &&
                    diagnostics$ci_dcc_permutation_replicates ==
                      diagnostics$ci_dcc_perm_tests * 13L,
                  paste(engine, "dcc.perm diagnostics are incomplete"))
    } else if (method == "hsic.gamma") {
      assert_true(diagnostics$ci_hsic_gamma_tests > 0L,
                  paste(engine, "hsic.gamma diagnostics are incomplete"))
    } else {
      assert_true(diagnostics$ci_hsic_perm_tests > 0L &&
                    diagnostics$ci_hsic_permutation_replicates ==
                      diagnostics$ci_hsic_perm_tests * 13L,
                  paste(engine, "hsic.perm diagnostics are incomplete"))
    }

    if (engine == "cuda") {
      assert_true(grepl("fastSplineCUDA", result$skeleton$residual_backend,
                        fixed = TRUE),
                  paste(method, "CUDA precision residual route was not used"))
      assert_true(result$skeleton$ci_backend == "native-cpu",
                  paste(method, "precision CI backend receipt is not truthful"))
    }
  }
}

cat("test_precision_ci_methods_actual.R: PASS\n")
