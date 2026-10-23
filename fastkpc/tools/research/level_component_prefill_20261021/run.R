# Same-binary off/on qualification, with fresh per-process semantic caches.
# Numerical production stays on CUDA; CPU only compares receipts and identities.
args <- commandArgs(TRUE)
stopifnot(length(args) %in% c(3L, 4L), args[[2L]] %in% c("off", "on"))
out <- args[[1L]]
mode <- args[[2L]]
baseline_path <- args[[3L]]
maximum <- if (length(args) == 4L) as.numeric(args[[4L]]) else Inf
stopifnot(!file.exists(file.path(out, "receipt.rds")))
dir.create(out, recursive=TRUE, showWarnings=FALSE)
Sys.setenv(FASTKPC_DCC_GAMMA_LEVEL_COMPONENT_PREFILL=if (mode == "on") "1" else "0",
           FASTKPC_DIAG_LEVEL_COMPONENT_ORDER="1")
source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
source("fastkpc/tools/research/exact_speedup_20261020/pinned_optimizer_gates.R")
sha <- function(p) digest::digest(file=p, algo="sha256", serialize=FALSE)
native <- "fastkpc/build/fastkpc_cuda.so"
native_sha <- sha(native)
dataset <- "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"
data <- as.matrix(readRDS(dataset))[, seq_len(22L), drop=FALSE]
storage.mode(data) <- "double"
invisible(full_cuda_ci_one_call_cache_control_native("configure", 262144L))
invisible(full_cuda_ci_one_call_cache_control_native("configure_target", 131072L))
invisible(full_cuda_ci_one_call_cache_control_native("reset"))
set.seed(20261020L)
seed <- .Random.seed
before <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
started <- proc.time()[["elapsed"]]
result <- fastkpc_compatible_cuda_skeleton(data, alpha=.1, options=list(
  route="full_cuda", compatible_cuda_strict=TRUE, max_conditioning_size=maximum,
  index=1, numCol=35L, trace_level="logical", ci_method="dcc.gamma"))
elapsed <- unname(proc.time()[["elapsed"]] - started)
receipt <- list(schema="level-component-prefill-subset22-v1", mode=mode,
  workload=if (is.infinite(maximum)) "subset22-default-inf" else paste0("subset22-level", maximum),
  native_sha256=native_sha, dataset_sha256=sha(dataset), elapsed_sec=elapsed,
  seed_before=seed, seed_after=.Random.seed, resources_before=before,
  resources_after=fastkpc_full_cuda_phase35_vertical_resource_snapshot(), result=result,
  shadow_policy=if (nzchar(Sys.getenv("FASTKPC_LSCP_PAYLOAD_LOG")))
    "external diagnostic four-field D2H and CPU byte comparison; not timing evidence" else "none")
saveRDS(receipt, file.path(out, "pending.rds"), compress=FALSE)
receipt <- qualify_pinned_optimizer_receipt(receipt, if (baseline_path == "-") NULL else baseline_path)
s <- result$summary
receipt$gates$native_unchanged <- identical(sha(native), native_sha)
receipt$gates$authority_gate <- isTRUE(s$authority_gate_pass)
receipt$gates$requested_mode <- identical(s$level_component_prefill_enabled, mode == "on")
if (mode == "on") {
  reference <- readRDS(baseline_path)
  receipt$gates$same_native <- identical(native_sha, reference$native_sha256)
  receipt$gates$physical_groups_exact <- identical(s$physical_group_sequence_sha256,
    reference$result$summary$physical_group_sequence_sha256)
  receipt$gates$gram_keys_and_order_exact <- identical(s$gram_resolution_sequence_sha256,
    reference$result$summary$gram_resolution_sequence_sha256)
  receipt$gates$no_consumer_finalization <- s$exact_primary_fixed_reuse_hit_target_count == 0
  rows <- s$level_component_prefill
  receipt$gates$level_ledgers <- length(rows) > 0L && all(vapply(rows, function(r)
    r$planned == r$produced && r$produced == r$residual_targets &&
    r$resolved + r$pending_at_end == r$planned, logical(1L)))
  if (is.infinite(maximum)) receipt$gates$full_coverage <-
    nrow(result$tasks) == 24264L && s$level_component_produced == 7270 &&
    s$level_component_resolved == 6632 &&
    sum(vapply(rows, `[[`, numeric(1L), "resolve_batches")) == 2592
}
saveRDS(receipt, file.path(out, "receipt.rds"), compress=FALSE)
sources <- c("fastkpc/src/full_cuda_ci_one_call.cpp", "fastkpc/src/cuda/full_cuda_ci_vertical.cu",
             "fastkpc/src/cuda/full_cuda_ci_vertical.hpp")
jsonlite::write_json(list(schema=receipt$schema, mode=mode, native_sha256=native_sha,
  receipt_sha256=sha(file.path(out,"receipt.rds")), source_sha256=as.list(setNames(vapply(sources,sha,""),sources)),
  baseline_sha256=if (baseline_path == "-") NULL else sha(baseline_path),
  runner_sha256=sha("fastkpc/tools/research/level_component_prefill_20261021/run.R"),
  elapsed_sec=elapsed, tasks=nrow(result$tasks), gates=receipt$gates,
  level_component_prefill=s$level_component_prefill,
  physical_group_sequence_sha256=s$physical_group_sequence_sha256,
  gram_resolution_sequence_sha256=s$gram_resolution_sequence_sha256,
  performance_evidence=if (receipt$shadow_policy == "none")
    "single development run; no promotion" else "diagnostic only; no timing claim",
  shadow_policy=receipt$shadow_policy,
  resources_before=before, resources_after=receipt$resources_after),
  file.path(out,"report.json"),pretty=TRUE,auto_unbox=TRUE,digits=NA)
print(receipt$gates)
cat("elapsed", elapsed, "seconds; logical CI",nrow(result$tasks),"\n")
stopifnot(all(unlist(receipt$gates)))
