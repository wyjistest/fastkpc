# Archive the exact successful source closure; do not overwrite older campaigns.
args <- commandArgs(TRUE); stopifnot(length(args)==2L)
w <- args[[1L]]; out <- args[[2L]]; stopifnot(!dir.exists(out))
sha <- function(p) digest::digest(file=p,algo="sha256",serialize=FALSE)
native <- "fastkpc/build/fastkpc_cuda.so"; native_sha <- sha(native)
fixture <- Sys.getenv("FASTKPC_LSCP_LIFECYCLE_FIXTURE",
  "/tmp/fastkpc-component-payload-20261021/native/raw/e526b295f52461bf3b30f515bdc61f58ee5741304e63124bdea93462423d3208.residual.f64")
stopifnot(identical(sha(fixture),
  "bd87f2b641ca623b5d4e65a6d421d351b46874f94f6608cf9bd6dc347bb7fa63"))
runs <- c("subset-off","subset-on","canonical-off","canonical-on","probe-off","probe-on","audit-off","audit-on")
receipts <- setNames(lapply(runs,function(nm) readRDS(file.path(w,nm,"receipt.rds"))),runs)
stopifnot(all(vapply(receipts,function(r) identical(r$native_sha256,native_sha)&&
  all(unlist(r$gates)),logical(1L))))
attribution <- jsonlite::read_json(file.path(w,"attribution.json"),simplifyVector=TRUE)
stopifnot(identical(attribution$status,"PASS"),attribution$components==65694L,
  attribution$unattributed_ci==0L,identical(attribution$native_sha256,native_sha),
  isTRUE(attribution$physical_to_logical_p_bits_exact),
  isTRUE(attribution$pmax_maximum_donors_verified))
sources <- c("fastkpc/src/full_cuda_ci_one_call.cpp","fastkpc/src/cuda/full_cuda_ci_vertical.cu",
             "fastkpc/src/cuda/full_cuda_ci_vertical.hpp")
source_sha <- setNames(vapply(sources,sha,""),sources)
for(nm in runs) {
  report <- jsonlite::read_json(file.path(w,nm,"report.json"),simplifyVector=FALSE)
  stopifnot(identical(unlist(report$source_sha256),source_sha))
}
for(log in c("configuration.log","cache-regression.log","empty-off.log","empty-on.log",
             "observer-smoke.log"))
  stopifnot(any(grepl("PASS",readLines(file.path(w,log)),fixed=TRUE)))
memcheck <- readLines(file.path(w,"lifecycle-memcheck.log"))
stopifnot(any(grepl("ERROR SUMMARY: 0 errors",memcheck,fixed=TRUE)),
  any(grepl("0 bytes leaked",memcheck,fixed=TRUE)),
  any(grepl("PASS caller device restored",memcheck,fixed=TRUE)))
before <- receipts[["canonical-off"]]; after <- receipts[["canonical-on"]]
stage_row <- function(r) {
  s<-r$result$summary; rows<-s$level_component_prefill
  get_sum <- function(name) sum(vapply(rows,`[[`,numeric(1),name))/1000
  data.frame(mode=r$mode,elapsed=r$elapsed_sec,optimizer=s$cuda_optimizer_host_ms/1000,
    consumer_dcov=s$cuda_dcov_host_ms/1000,stage_a=get_sum("stage_a_host_ms"),
    stage_b=get_sum("stage_b_host_ms"),consumer_eig=s$cuda_dcov_full_eig_ms/1000,
    stage_b_eig=get_sum("eig_cuda_ms"),total_eig=s$cuda_dcov_full_eig_ms/1000+get_sum("eig_cuda_ms"),
    prepared_build=s$cuda_multi_penalty_prepared_build_ms/1000,
    prepared_build_count=s$cuda_multi_penalty_prepared_build_count,
    native_setup=s$native_setup_ms/1000)
}
stages <- rbind(stage_row(before),stage_row(after))
dir.create(out,recursive=TRUE)
write.csv(stages,file.path(out,"timing.csv"),row.names=FALSE)
for(nm in runs) {
  dir.create(file.path(out,nm))
  stopifnot(system2("gzip",c("-n","-c",shQuote(file.path(w,nm,"receipt.rds"))),
    stdout=file.path(out,nm,"receipt.rds.gz"))==0L)
  stopifnot(file.copy(file.path(w,nm,"report.json"),file.path(out,nm)))
}
for(nm in c("probe-off","probe-on","audit-off","audit-on")) {
  for(p in list.files(file.path(w,nm),pattern="tsv$",full.names=TRUE))
    stopifnot(system2("gzip",c("-n","-c",shQuote(p)),
      stdout=file.path(out,nm,paste0(basename(p),".gz")))==0L)
  stopifnot(system2("tar",c("-C",shQuote(file.path(w,nm)),"-czf",
    shQuote(file.path(out,nm,"wide-raw.tar.gz")),"raw"))==0L)
}
logs <- list.files(w,pattern="\\.(log|csv|json)$",full.names=TRUE)
stopifnot(all(file.copy(logs,out)))
for(p in c("attributed-ci.rds","empty-off.rds","empty-on.rds"))
  stopifnot(file.copy(file.path(w,p),out))
for(p in c("audit_payload.so","empty_layer.so","device_lifecycle"))
  stopifnot(file.copy(file.path(w,p),out))
stopifnot(file.copy(fixture,file.path(out,"device_lifecycle.residual.f64")))
dir.create(file.path(out,"sources"))
stopifnot(all(file.copy(sources,file.path(out,"sources"))))
stopifnot(system2("gzip",c("-n","-c",shQuote(native)),
  stdout=file.path(out,"fastkpc_cuda.so.gz"))==0L)
stopifnot(system2("git",c("diff","--",sources),stdout=file.path(out,"runtime.patch"))==0L)
tools <- c(list.files("fastkpc/tools/research/level_component_prefill_canonical_20261021",full.names=TRUE),
  "fastkpc/tools/research/level_component_prefill_20261021/run.R",
  "fastkpc/tools/research/level_component_prefill_20261021/cache_lifecycle.cu",
  "fastkpc/tests/test_full_cuda_ci_level_component_prefill.R",
  "fastkpc/tests/test_full_cuda_ci_legacy_component_cache.R",
  "fastkpc/tools/research/exact_speedup_20261020/pinned_optimizer_gates.R",
  "fastkpc/tools/with_phase10_exact_development_env.sh")
dir.create(file.path(out,"tools"))
for(p in tools) {
  dest <- file.path(out,"tools",p); dir.create(dirname(dest),recursive=TRUE,showWarnings=FALSE)
  stopifnot(file.copy(p,dest))
}
closure <- list(schema="level-component-prefill-canonical-closure-v1",
  status="PASS_CANONICAL_OPT_IN_DENSE_PROTOTYPE",native_sha256=native_sha,
  source_sha256=as.list(source_sha),subset_ci=24264L,canonical_ci=240498L,
  oracle_errors=as.list(after$oracle_errors),attribution=attribution[c("components","original_wide_cohorts",
    "original_wide_first_miss_targets","component_difference_counts","changed_ci","unattributed_ci",
    "max_ci_difference","near_alpha_changed","near_alpha_max_difference")],
  timing=list(off_sec=before$elapsed_sec,on_sec=after$elapsed_sec,
    observed_reduction=1-after$elapsed_sec/before$elapsed_sec,repetitions_per_mode=1L),
  checks=list(empty_layer_off_on="PASS",caller_device_restoration="PASS",
    source_shared_memcheck_errors=0L,source_shared_leaked_bytes=0L,
    observer_smoke_components=4148L,physical_to_logical_p_bits_exact=TRUE,
    pmax_maximum_donors_verified=TRUE),
  evidence_class="source-closure development; one off/on timing pair and separate diagnostic observations",
  source_build="incremental: one-call, CUDA vertical, R API; retained other accepted objects",
  production_default=FALSE,structured_integration="NOT_RUN",phase10="ACTIVE",
  candidate="NOT_FROZEN",holdout="SEALED",recommended=FALSE)
jsonlite::write_json(closure,file.path(out,"closure.json"),pretty=TRUE,auto_unbox=TRUE,digits=NA)
inputs <- c("fastkpc/artifacts/full_cuda_ci/exact_speedup_20261020/canonical-queues32.rds",
  "fastkpc/artifacts/full_cuda_ci/level_component_prefill_20261021/final-off/receipt.rds",
  "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds",
  list.files("fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2",pattern="rds$",full.names=TRUE))
files <- list.files(out,recursive=TRUE,full.names=TRUE)
manifest <- list(schema="canonical-dense-prefill-manifest-v1",status="PASS",native_sha256=native_sha,
  files=setNames(lapply(files,function(p) list(bytes=unname(file.info(p)$size),sha256=sha(p))),
    substring(files,nchar(out)+2L)),inputs=as.list(setNames(vapply(inputs,sha,""),inputs)))
jsonlite::write_json(manifest,file.path(out,"manifest.json"),pretty=TRUE,auto_unbox=TRUE,digits=NA)
cat("PASS archive",out,"\nmanifest",sha(file.path(out,"manifest.json")),"\n")
