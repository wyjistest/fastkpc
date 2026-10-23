# Extend actual-cohort coverage to every depth reached by the 22-column input.
# BASELINE='-' runs an uninstrumented reference first; capture runs must name it.
args <- commandArgs(TRUE)
stopifnot(length(args) == 2L)
out <- args[[1L]]
baseline <- args[[2L]]
capturing <- baseline != "-"
stopifnot(!file.exists(file.path(out, "receipt.rds")))
dir.create(out, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out, "raw"), showWarnings = FALSE)
if (capturing) {
  stopifnot(identical(normalizePath(Sys.getenv("FASTKPC_COHORT_CAPTURE_DIR")),
                      normalizePath(file.path(out, "raw"))))
} else stopifnot(!nzchar(Sys.getenv("FASTKPC_COHORT_CAPTURE_DIR")))
source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
source("fastkpc/tools/research/exact_speedup_20261020/pinned_optimizer_gates.R")
file_sha <- function(p) digest::digest(file = p, algo = "sha256", serialize = FALSE)
native <- "fastkpc/build/fastkpc_cuda.so"
native_sha <- file_sha(native)
stopifnot(identical(native_sha,
  "fe8286f7e6c6114ef66e040b16b693f5abcb23333dc9c0fd767c042c509c3c50"))
data_path <- "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"
data <- as.matrix(readRDS(data_path))[, seq_len(22L), drop = FALSE]
storage.mode(data) <- "double"
invisible(full_cuda_ci_one_call_cache_control_native("configure", 262144L))
invisible(full_cuda_ci_one_call_cache_control_native("configure_target", 131072L))
invisible(full_cuda_ci_one_call_cache_control_native("reset"))
set.seed(20261020L)
seed_before <- .Random.seed
resources_before <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
started <- proc.time()[["elapsed"]]
result <- fastkpc_compatible_cuda_skeleton(data, alpha = .1, options = list(
  route = "full_cuda", compatible_cuda_strict = TRUE, max_conditioning_size = Inf,
  index = 1, numCol = 35L, trace_level = "logical", ci_method = "dcc.gamma"))
receipt <- list(schema = "production-cohort-frozen-capture-inf-v1",
  workload = "subset22-default-inf", native_sha256 = native_sha,
  elapsed_sec = unname(proc.time()[["elapsed"]] - started),
  seed_before = seed_before, seed_after = .Random.seed,
  resources_before = resources_before,
  resources_after = fastkpc_full_cuda_phase35_vertical_resource_snapshot(), result = result)
receipt <- qualify_pinned_optimizer_receipt(receipt, if (capturing) baseline else NULL)
stopifnot(all(unlist(receipt$gates)), identical(file_sha(native), native_sha))
counts <- c(prefill = length(list.files(file.path(out, "raw"), "^prefill-.*json$")),
            finalize = length(list.files(file.path(out, "raw"), "^finalize-.*json$")))
if (capturing) stopifnot(
  counts[["prefill"]] == result$summary$exact_primary_fixed_reuse_store_batch_count,
  counts[["finalize"]] == result$summary$exact_primary_fixed_reuse_hit_batch_count,
  all(counts > 0L))
saveRDS(receipt, file.path(out, "receipt.rds"), compress = FALSE)
write.csv(result$tasks, file.path(out, "tasks.csv"), row.names = FALSE)
con <- file(file.path(out, "dataset.f64"), "wb")
writeBin(as.double(data), con, size = 8L, endian = "little")
close(con)
meta <- list(schema = receipt$schema, native_sha256 = native_sha,
  script_sha256 = file_sha("fastkpc/tools/research/production_cohort_frozen_20261021/run_default_inf.R"),
  receipt_sha256 = file_sha(file.path(out, "receipt.rds")),
  baseline_sha256 = if (capturing) file_sha(baseline) else NULL,
  data_source_sha256 = file_sha(data_path), dataset_sha256 = file_sha(file.path(out, "dataset.f64")),
  n = nrow(data), p = ncol(data), max_level = max(result$tasks$level),
  counts = as.list(counts), tasks = nrow(result$tasks), gates = receipt$gates,
  resource_before = receipt$resources_before, resource_after = receipt$resources_after,
  elapsed_diagnostic_only = receipt$elapsed_sec,
  shadow_policy = "explicit diagnostic D2H outside production counters; no CPU numerical producer",
  environment = as.list(Sys.getenv(c("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_MAX_CONNECTIONS",
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "LD_PRELOAD"))))
jsonlite::write_json(meta, file.path(out, "capture.json"), pretty = TRUE,
                     auto_unbox = TRUE, digits = NA)
print(receipt$gates)
print(counts)
cat("PASS default-Inf; tasks", nrow(result$tasks), "max level", max(result$tasks$level), "\n")
