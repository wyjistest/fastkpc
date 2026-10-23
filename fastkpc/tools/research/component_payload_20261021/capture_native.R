# Actual accepted-native component-cache commits; diagnostic copies excluded
# from performance claims and explicitly disclosed in the receipt.
args <- commandArgs(TRUE)
stopifnot(length(args) == 1L)
out <- args[[1L]]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
source("fastkpc/tools/research/exact_speedup_20261020/pinned_optimizer_gates.R")
native <- normalizePath("fastkpc/build/fastkpc_cuda.so")
file_sha <- function(p) digest::digest(file=p,algo="sha256",serialize=FALSE)
stopifnot(identical(file_sha(native),
  "fe8286f7e6c6114ef66e040b16b693f5abcb23333dc9c0fd767c042c509c3c50"))
data <- as.matrix(readRDS("fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"))[,1:22,drop=FALSE]
storage.mode(data) <- "double"
invisible(full_cuda_ci_one_call_cache_control_native("configure",262144L))
invisible(full_cuda_ci_one_call_cache_control_native("configure_target",131072L))
invisible(full_cuda_ci_one_call_cache_control_native("reset"))
set.seed(20261020L)
seed <- .Random.seed
before <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
start <- proc.time()[["elapsed"]]
result <- fastkpc_compatible_cuda_skeleton(data,alpha=.1,options=list(
  route="full_cuda",compatible_cuda_strict=TRUE,max_conditioning_size=Inf,
  index=1,numCol=35L,trace_level="logical",ci_method="dcc.gamma"))
receipt <- list(schema="component-payload-native-capture-v1",workload="subset22-default-inf",
  native_sha256=file_sha(native),elapsed_sec=unname(proc.time()[["elapsed"]]-start),
  seed_before=seed,seed_after=.Random.seed,resources_before=before,
  resources_after=fastkpc_full_cuda_phase35_vertical_resource_snapshot(),result=result,
  shadow_policy="external diagnostic cache-payload/residual D2H; not timing evidence")
receipt <- qualify_pinned_optimizer_receipt(receipt,
  "fastkpc/artifacts/full_cuda_ci/production_cohort_frozen_20261021/baseline-inf/receipt.rds")
stopifnot(all(unlist(receipt$gates)))
saveRDS(receipt,file.path(out,"receipt.rds"),compress=FALSE)
keys <- readLines(Sys.getenv("FASTKPC_PAYLOAD_CAPTURE_KEYS"))
stopifnot(length(keys)==6632L,
          length(list.files(file.path(out,"raw"),pattern="\\.json$"))==length(keys))
jsonlite::write_json(list(schema=receipt$schema,native_sha256=receipt$native_sha256,
  receipt_sha256=file_sha(file.path(out,"receipt.rds")),
  runner_sha256=file_sha("fastkpc/tools/research/component_payload_20261021/capture_native.R"),
  interposer_sha256=file_sha(Sys.getenv("LD_PRELOAD")),
  gates=receipt$gates,keys=length(keys),elapsed_diagnostic_only=receipt$elapsed_sec),
  file.path(out,"capture.json"),pretty=TRUE,auto_unbox=TRUE,digits=NA)
print(receipt$gates)
cat("PASS actual native capture; tasks",nrow(result$tasks),"\n")
