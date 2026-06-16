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
      if (!identical(set_key(a[[i]][[j]]), set_key(b[[i]][[j]]))) {
        return(FALSE)
      }
    }
  }
  TRUE
}

build_fastkpc_native(rebuild = TRUE)
build_fastkpc_cuda_native(rebuild = TRUE)

scenario <- fastkpc_fixed_scenario()
data <- scenario$data

cpu0 <- fast_skeleton_cpp(data, alpha = scenario$alpha, max_conditioning_size = 0)
cuda0 <- fast_skeleton_cuda(data, alpha = scenario$alpha, max_conditioning_size = 0)
assert_true(identical(cuda0$adjacency, cpu0$adjacency),
            "CUDA skeleton adjacency should match CPU for ord 0")
assert_true(max(abs(cuda0$pMax - cpu0$pMax)) < 1e-8,
            "CUDA skeleton pMax should match CPU for ord 0")
assert_true(compare_sepsets_exact(cuda0$sepsets, cpu0$sepsets),
            "CUDA skeleton sepsets should match CPU for ord 0")
assert_true(identical(cuda0$n.edgetests, cpu0$n.edgetests),
            "CUDA skeleton n.edgetests should match CPU for ord 0")
assert_true(cuda0$backend == "cuda", "CUDA skeleton should identify its backend")

cpu1 <- fast_skeleton_cpp(data, alpha = scenario$alpha,
                          max_conditioning_size = scenario$max_conditioning_size)
cuda_auto <- fast_skeleton_cuda(data, alpha = scenario$alpha,
                                max_conditioning_size = scenario$max_conditioning_size,
                                batch_size = 0)
cuda_one <- fast_skeleton_cuda(data, alpha = scenario$alpha,
                               max_conditioning_size = scenario$max_conditioning_size,
                               batch_size = 1)

assert_true(identical(cuda_auto$adjacency, cpu1$adjacency),
            "CUDA skeleton adjacency should match CPU for ord 1")
assert_true(max(abs(cuda_auto$pMax - cpu1$pMax)) < 1e-8,
            "CUDA skeleton pMax should match CPU for ord 1")
assert_true(compare_sepsets_exact(cuda_auto$sepsets, cpu1$sepsets),
            "CUDA skeleton sepsets should match CPU for ord 1")
assert_true(identical(cuda_auto$n.edgetests, cpu1$n.edgetests),
            "CUDA skeleton n.edgetests should match CPU for ord 1")

assert_true(length(cuda_auto$per.level.log) == length(cpu1$per.level.log),
            "CUDA skeleton should have one deletion log per level")
for (level in seq_along(cpu1$per.level.log)) {
  assert_true(length(cuda_auto$per.level.log[[level]]) ==
                length(cpu1$per.level.log[[level]]),
              sprintf("deletion log length mismatch at level %d", level))
}

assert_true(identical(cuda_one$adjacency, cuda_auto$adjacency),
            "batch_size = 1 should match auto adjacency")
assert_true(max(abs(cuda_one$pMax - cuda_auto$pMax)) < 1e-8,
            "batch_size = 1 should match auto pMax")
assert_true(compare_sepsets_exact(cuda_one$sepsets, cuda_auto$sepsets),
            "batch_size = 1 should match auto sepsets")
assert_true(identical(cuda_one$n.edgetests, cuda_auto$n.edgetests),
            "batch_size = 1 should match auto n.edgetests")

cat("test_skeleton_cuda_batch.R: PASS\n")
