# Fresh-process retained-stack A/B. Timing excludes artifact serialization.
# Rscript .../run_pinned_optimizer_ab.R 0|1 subset22|canonical OUTPUT [BASELINE]
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) %in% c(3L, 4L), args[[1L]] %in% c("0", "1"),
          args[[2L]] %in% c("subset22", "canonical"), !file.exists(args[[3L]]))
if (args[[1L]] == "1" && system2("grep", c(
  "-aFq", "--", shQuote("FASTKPC_MULTI_PENALTY_PINNED_IO"),
  shQuote("fastkpc/build/fastkpc_cuda.so")
)) != 0L) {
  stop("Pinned prototype is retired; apply the archived research patch and rebuild first.",
       call. = FALSE)
}
Sys.setenv(
  FASTKPC_MULTI_PENALTY_PINNED_IO = args[[1L]],
  FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES = "11811160064",
  FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS = "1",
  FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES = "8"
)
source("fastkpc/tools/research/exact_speedup_20261020/pinned_optimizer_gates.R")
source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
stopifnot(fastkpc_cuda_available(), requireNamespace("digest", quietly = TRUE))
data <- as.matrix(readRDS(
  "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"
))
if (args[[2L]] == "subset22") data <- data[, seq_len(22L), drop = FALSE]
invisible(full_cuda_ci_one_call_cache_control_native("configure", 262144L))
invisible(full_cuda_ci_one_call_cache_control_native("configure_target", 131072L))
invisible(full_cuda_ci_one_call_cache_control_native("reset"))
set.seed(20261020L)
seed_before <- .Random.seed
resources_before <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
native_sha256 <- digest::digest(file = "fastkpc/build/fastkpc_cuda.so", algo = "sha256")
runtime_environment <- as.list(Sys.getenv(c(
  "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_MAX_CONNECTIONS", "CUDA_DEVICE_MAX_COPY_CONNECTIONS",
  "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
  "FASTKPC_MULTI_PENALTY_PINNED_IO",
  "FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES",
  "FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS",
  "FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES"
), unset = ""))
started <- proc.time()[["elapsed"]]
result <- fastkpc_compatible_cuda_skeleton(data, alpha = 0.1, options = list(
  route = "full_cuda", compatible_cuda_strict = TRUE,
  max_conditioning_size = if (args[[2L]] == "canonical") Inf else 3L,
  index = 1, numCol = 35L, trace_level = "logical", ci_method = "dcc.gamma"
))
elapsed_sec <- unname(proc.time()[["elapsed"]] - started)
seed_after <- .Random.seed
resources_after <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
receipt <- list(
  schema = "exact-speedup-pinned-optimizer-ab-v1", workload = args[[2L]],
  pinned = args[[1L]], native_sha256 = native_sha256, elapsed_sec = elapsed_sec,
  runtime_environment = runtime_environment,
  seed_before = seed_before, seed_after = seed_after, gates = NULL,
  oracle_errors = NULL,
  resources_before = resources_before, resources_after = resources_after,
  result = result
)
saveRDS(receipt, paste0(args[[3L]], ".pending"), compress = FALSE)
receipt <- qualify_pinned_optimizer_receipt(
  receipt, if (length(args) == 4L) args[[4L]] else NULL
)
saveRDS(receipt, args[[3L]], compress = FALSE)
gates <- receipt$gates
oracle_errors <- receipt$oracle_errors
summary <- result$summary
cat(sprintf("pinned=%s workload=%s elapsed_sec=%.3f native=%s\n",
            args[[1L]], args[[2L]], elapsed_sec, native_sha256))
print(gates)
if (!is.null(oracle_errors)) print(oracle_errors)
for (nm in c("cuda_optimizer_host_ms", "cuda_multi_penalty_optimizer_host_ms",
             "cuda_dcov_host_ms", "cuda_dcov_full_eig_ms", "cuda_residual_solve_host_ms",
             "native_setup_ms", "physical_tests_evaluated")) {
  cat(nm, "=", format(summary[[nm]]), "\n")
}
stopifnot(all(unlist(gates)))
