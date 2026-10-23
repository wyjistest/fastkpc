source("fastkpc/tools/research/guarded_qr_20261020/common.R")
args <- commandArgs(TRUE); work <- args[[1L]]
out <- args[[2L]]
stopifnot(!dir.exists(out))
fixture <- readRDS(file.path(work,"fixtures.rds"))
freeze <- readRDS(file.path(work,"frozen_guard.rds"))
capacity <- readRDS(file.path(work,"capacity_reference.rds"))
stopifnot(nrow(capacity$rows)==length(fixture$fixtures),
  all(unlist(capacity$rows[,4:ncol(capacity$rows)])))
rows <- do.call(rbind,lapply(c("calibration","validation"),function(split) {
  read.csv(file.path(work,paste0(split,".csv")))[,names(read.csv(file.path(work,"calibration.csv")))]
}))
paths <- c(list.files(file.path(work,"calibration"),pattern="[.]rds$",full.names=TRUE),
  list.files(file.path(work,"validation"),pattern="[.]rds$",full.names=TRUE))
details <- lapply(paths,readRDS)
details <- details[match(rows$id,vapply(details,function(d)d$summary$id,""))]
stopifnot(length(details)==length(fixture$fixtures),!anyDuplicated(rows$id))
flags <- lapply(details,function(d)guard_flags(d$features,freeze$scale))
rows$guard_flag <- vapply(flags,any,TRUE)
rows$flagged_targets <- vapply(flags,sum,0L)
rows$guarded_kernel_cost_ms <- rows$qr_ms + ifelse(rows$guard_flag,rows$exact_ms,0)
without_mixed <- vapply(details,function(d) {
  d$features$mixed_svd <- FALSE
  any(guard_flags(d$features,freeze$scale))
},TRUE)
metrics <- function(r) list(cohorts=nrow(r),targets=2L*nrow(r),
  raw_p_failures=sum(r$p_error>1e-10),raw_contract_failure_cohorts=sum(r$hard_failure),
  raw_safety_failure_cohorts=sum(r$safety_failure),max_p_error=max(r$p_error),
  worst_p_cohort=r$id[which.max(r$p_error)],decision_flips=sum(r$decision_flip),
  flagged_cohorts=sum(r$guard_flag),flagged_targets=sum(r$flagged_targets),
  missed_p_failures=sum(r$p_error>1e-10 & !r$guard_flag),
  missed_contract_failure_cohorts=sum(r$hard_failure & !r$guard_flag),
  missed_safety_failure_cohorts=sum(r$safety_failure & !r$guard_flag),
  exact_kernel_ms=sum(r$exact_ms),qr_kernel_ms=sum(r$qr_ms),
  raw_kernel_speedup=sum(r$exact_ms)/sum(r$qr_ms),
  replay_kernel_ms=sum(r$exact_ms[r$guard_flag]),
  cost_weighted_replay_fraction=sum(r$exact_ms[r$guard_flag])/sum(r$exact_ms),
  guarded_additive_kernel_cost_ms=sum(r$guarded_kernel_cost_ms),
  guarded_additive_kernel_speedup=sum(r$exact_ms)/sum(r$guarded_kernel_cost_ms))
point_errors <- do.call(rbind,unlist(lapply(details,function(d)lapply(d$same_points,function(x)x$error)),recursive=FALSE))
no_branch <- do.call(rbind,lapply(details,function(d) {
  data.frame(id=d$summary$id,target=1:2,
    same_decisions=vapply(d$divergence,function(x)is.na(x$first_decision_difference),TRUE),
    selected_sp_error=apply(abs(d$exact$log_sp-d$qr$log_sp),2,max),
    residual_error=d$target_gates$residual_max,
    residual_relative=d$target_gates$residual_rel)
}))
q <- split(rows,rows$q)
summary <- list(schema="guarded-qr-narrow-diagnostic-summary-v1",
  created_at=as.character(Sys.time()),producer=fixture$producer,
  environment=list(R=R.version.string,mgcv=as.character(packageVersion("mgcv")),
    Rcpp=as.character(packageVersion("Rcpp")),
    gpu=system2("nvidia-smi",c("--query-gpu=index,name,uuid,driver_version","--format=csv,noheader"),stdout=TRUE),
    variables=as.list(Sys.getenv(c("CUDA_VISIBLE_DEVICES","CUDA_DEVICE_MAX_CONNECTIONS",
      "OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")))),
  production_native_sha256=fixture$native_sha256,
  probe_native_sha256=file_sha(file.path(work,"guarded_qr_probe.so")),
  fixture_sha256=file_sha(file.path(work,"fixtures.rds")),
  frozen_guard_sha256=file_sha(file.path(work,"frozen_guard.rds")),
  scope="captured two-target optimizer; production exact fixed and GPU full eig; no production integration",
  shapes=sort(unique(rows$q)),guard_scale=freeze$scale,
  calibration=metrics(rows[rows$split=="calibration",]),
  validation=metrics(rows[rows$split=="validation",]),all=metrics(rows),
  by_q=lapply(q,metrics),by_dataset=lapply(split(rows,rows$dataset),metrics),
  same_sp=list(points=nrow(point_errors),max_absolute_errors=apply(point_errors,2,max)),
  same_discrete_decisions=list(targets=sum(no_branch$same_decisions),
    max_selected_sp_error=max(no_branch$selected_sp_error[no_branch$same_decisions],0),
    max_residual_error=max(no_branch$residual_error[no_branch$same_decisions],0),
    cohorts_with_same_decisions_but_p_failure=rows$id[rows$decision_divergences==0L & rows$p_error>1e-10]),
  fixed_threshold_mixed_svd_ablation=list(
    calibration_flags_changed=sum(rows$guard_flag[rows$split=="calibration"]!=without_mixed[rows$split=="calibration"]),
    validation_flags_changed=sum(rows$guard_flag[rows$split=="validation"]!=without_mixed[rows$split=="validation"]),
    note="post-calibration diagnostic only; frozen guard is unchanged"),
  reference_bitwise=all(rows$reference_bitwise),deterministic=all(rows$deterministic),
  production_resource_counters_closed=all(rows$resources_closed),
  production_capacity_reference=list(cohorts=nrow(capacity$rows),
    all_bitwise=TRUE,buckets=unique(capacity$rows$bucket)),
  production_runtime_source_or_defaults_modified=FALSE,phase10="ACTIVE",promotion="NOT_PROMOTED",
  candidate_frozen=FALSE,holdout="SEALED_NOT_RELEASED",recommended=FALSE,
  timing_limit="sum of three-run per-cohort kernel medians; excludes guard/queue/downstream cost; conservative whole-cohort replay; not full runtime timing",
  reference_limit="exact GPU producer reference, not a new full mgcv authority campaign; empirical guards are not certified bounds")
dir.create(out,recursive=TRUE)
for(name in c("fixtures.rds","frozen_guard.rds","build.json","guarded_qr_probe.so",
  "calibration.csv","calibration_grid.csv","validation.csv",
  "capacity_reference.rds","capacity_reference.csv")) {
  stopifnot(file.copy(file.path(work,name),file.path(out,name),overwrite=FALSE))
}
for(split in c("calibration","validation")) {
  dir.create(file.path(out,split))
  selected<-list.files(file.path(work,split),pattern="[.]rds$",full.names=TRUE)
  stopifnot(all(file.copy(selected,file.path(out,split),overwrite=FALSE)))
}
write.csv(rows,file.path(out,"cohorts.csv"),row.names=FALSE)
write.csv(no_branch,file.path(out,"trajectory_identity.csv"),row.names=FALSE)
jsonlite::write_json(summary,file.path(out,"summary.json"),pretty=TRUE,auto_unbox=TRUE,digits=17,na="null")
source_dir <- "fastkpc/tools/research/guarded_qr_20261020"
source_paths <- list.files(source_dir,full.names=TRUE)
dependencies <- c("fastkpc/src/cuda/mgcv_multi_penalty_gcv.cu",
  "fastkpc/src/cuda/mgcv_multi_penalty_gcv.hpp",
  "fastkpc/src/cuda/lapack_312_small_dgesdd.cuh",
  "fastkpc/src/cuda/lapack_312_small_dsyevr.cuh",
  list.files("fastkpc/src/cuda/third_party",pattern="[.](inc|cuh)$",full.names=TRUE,recursive=TRUE))
artifact_paths <- list.files(out,full.names=TRUE,recursive=TRUE)
manifest <- list(schema="guarded-qr-source-evidence-manifest-v1",
  files=setNames(lapply(artifact_paths,file_sha),substring(artifact_paths,nchar(out)+2L)),
  source_files=setNames(lapply(source_paths,file_sha),source_paths),
  source_dependencies=setNames(lapply(dependencies,file_sha),dependencies),
  production_source_sha256=file_sha("fastkpc/src/cuda/mgcv_multi_penalty_gcv.cu"),
  production_native_sha256=file_sha("fastkpc/build/fastkpc_cuda.so"))
stopifnot(identical(manifest$production_native_sha256,fixture$native_sha256))
jsonlite::write_json(manifest,file.path(out,"manifest.json"),pretty=TRUE,auto_unbox=TRUE)
print(summary[c("calibration","validation","same_sp","same_discrete_decisions")])
cat("manifest SHA256",file_sha(file.path(out,"manifest.json")),"\n")
