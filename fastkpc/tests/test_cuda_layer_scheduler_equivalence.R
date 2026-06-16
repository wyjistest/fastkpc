source("fastkpc/R/native.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/legacy_runner.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

set_key <- function(values) paste(sort(as.integer(values)), collapse = ",")

compare_sepsets_exact <- function(a, b) {
  for (i in seq_along(a)) {
    for (j in seq_along(a[[i]])) {
      if (!identical(set_key(a[[i]][[j]]), set_key(b[[i]][[j]]))) return(FALSE)
    }
  }
  TRUE
}

check_case <- function(data, alpha, max_ord, label) {
  cpu <- fast_skeleton_cpp_backend(
    data, alpha, max_ord,
    residual_backend = "linear",
    residual_cache = TRUE
  )
  legacy <- fast_skeleton_cuda_backend(
    data, alpha, max_ord,
    residual_backend = "linear",
    scheduler = "legacy",
    residual_cache = TRUE
  )
  layer <- fast_skeleton_cuda_backend(
    data, alpha, max_ord,
    residual_backend = "linear",
    scheduler = "layer",
    residual_cache = TRUE
  )
  layer_one <- fast_skeleton_cuda_backend(
    data, alpha, max_ord,
    residual_backend = "linear",
    scheduler = "layer",
    batch_size = 1,
    residual_cache = TRUE
  )

  assert_true(identical(layer$adjacency, legacy$adjacency),
              paste(label, "layer adjacency should match legacy"))
  assert_true(identical(layer$n.edgetests, legacy$n.edgetests),
              paste(label, "layer n.edgetests should match legacy"))
  assert_true(compare_sepsets_exact(layer$sepsets, legacy$sepsets),
              paste(label, "layer sepsets should match legacy"))
  assert_true(max(abs(layer$pMax - legacy$pMax)) < 1e-8,
              paste(label, "layer pMax should match legacy"))

  assert_true(identical(layer$adjacency, cpu$adjacency),
              paste(label, "layer adjacency should match CPU"))
  assert_true(identical(layer$n.edgetests, cpu$n.edgetests),
              paste(label, "layer n.edgetests should match CPU"))
  assert_true(compare_sepsets_exact(layer$sepsets, cpu$sepsets),
              paste(label, "layer sepsets should match CPU"))
  assert_true(max(abs(layer$pMax - cpu$pMax)) < 1e-8,
              paste(label, "layer pMax should match CPU"))

  assert_true(identical(layer_one$adjacency, layer$adjacency),
              paste(label, "batch_size=1 adjacency should match auto"))
  assert_true(max(abs(layer_one$pMax - layer$pMax)) < 1e-8,
              paste(label, "batch_size=1 pMax should match auto"))
  assert_true(layer$scheduler == "layer", paste(label, "scheduler should be layer"))
}

build_fastkpc_native(rebuild = TRUE)
build_fastkpc_cuda_native(rebuild = TRUE)

scenario <- fastkpc_fixed_scenario()
check_case(scenario$data, scenario$alpha, scenario$max_conditioning_size,
           "fixed scenario")

set.seed(211)
n <- 110
z1 <- runif(n, -2, 2)
z2 <- runif(n, -2, 2)
data <- cbind(
  x1 = sin(z1) + rnorm(n, sd = 0.15),
  x2 = cos(z1) + rnorm(n, sd = 0.15),
  x3 = z1 * z2 + rnorm(n, sd = 0.15),
  x4 = sin(z2) + rnorm(n, sd = 0.15),
  x5 = rnorm(n)
)
check_case(data, alpha = 0.2, max_ord = 2, "nonlinear generated")

cat("test_cuda_layer_scheduler_equivalence.R: PASS\n")
