# Read-only diagnostic interception; original numerical results are returned intact.
args <- commandArgs(TRUE)
stopifnot(length(args)==1L)
out <- args[[1L]]
dir.create(file.path(out,"raw"),recursive=TRUE,showWarnings=FALSE)
source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
source("fastkpc/tools/research/exact_speedup_20261020/pinned_optimizer_gates.R")
file_sha <- function(p) digest::digest(file=p,algo="sha256",serialize=FALSE)
native <- normalizePath("fastkpc/build/fastkpc_cuda.so")
stopifnot(identical(file_sha(native),"fe8286f7e6c6114ef66e040b16b693f5abcb23333dc9c0fd767c042c509c3c50"))
data <- as.matrix(readRDS("fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"))[,1:22,drop=FALSE]
storage.mode(data) <- "double"
invisible(full_cuda_ci_one_call_cache_control_native("configure",262144L))
invisible(full_cuda_ci_one_call_cache_control_native("configure_target",131072L))
invisible(full_cuda_ci_one_call_cache_control_native("reset"))
set.seed(20261020L)
seed <- .Random.seed
before <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
started <- proc.time()[["elapsed"]]
result <- fastkpc_compatible_cuda_skeleton(data,alpha=.1,options=list(
 route="full_cuda",compatible_cuda_strict=TRUE,max_conditioning_size=Inf,
 index=1,numCol=35L,trace_level="logical",ci_method="dcc.gamma"))
receipt <- list(schema="singleton-pvalue-native-capture-v1",workload="subset22-default-inf",
 native_sha256=file_sha(native),elapsed_sec=unname(proc.time()[["elapsed"]]-started),
 seed_before=seed,seed_after=.Random.seed,resources_before=before,
 resources_after=fastkpc_full_cuda_phase35_vertical_resource_snapshot(),result=result,
 shadow_policy="external diagnostic payload/pair-cross D2H; no performance claim")
receipt <- qualify_pinned_optimizer_receipt(receipt,
 "fastkpc/artifacts/full_cuda_ci/component_payload_20261021/native/receipt.rds")
stopifnot(all(unlist(receipt$gates)),result$summary$result_cache_hit_count==0,
 result$summary$logical_tests_consumed==24264,result$summary$physical_tests_evaluated==24264)
saveRDS(receipt,file.path(out,"receipt.rds"),compress=FALSE)
tasks <- result$tasks
con <- file(file.path(out,"task-pvalues.f64"),"wb")
writeBin(as.double(tasks$p_used),con,size=8L,endian="little"); close(con)
tasks$p_used <- NULL
write.table(tasks,file.path(out,"tasks.tsv"),sep="\t",quote=FALSE,row.names=FALSE,na="")
jsonlite::write_json(list(schema=receipt$schema,native_sha256=receipt$native_sha256,
 receipt_sha256=file_sha(file.path(out,"receipt.rds")),gates=receipt$gates,
 runner_sha256=file_sha("fastkpc/tools/research/singleton_pvalue_20261021/capture_native.R"),
 interposer_sha256=file_sha(Sys.getenv("LD_PRELOAD")),tasks=24264,
 dataset_key=result$summary$dataset_key,
 physical_pairs=result$summary$physical_tests_evaluated,result_cache_hits=0,
 elapsed_diagnostic_only=receipt$elapsed_sec),file.path(out,"capture.json"),
 pretty=TRUE,auto_unbox=TRUE,digits=NA)
print(receipt$gates)
cat("PASS singleton pair capture\n")
