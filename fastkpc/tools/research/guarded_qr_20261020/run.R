source("fastkpc/tools/research/guarded_qr_20261020/common.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
args <- commandArgs(TRUE)
work <- args[[1L]]; split <- args[[2L]]
stopifnot(split %in% c("smoke", "calibration", "validation"))
fixture_path <- file.path(work,"fixtures.rds")
fixtures_receipt <- readRDS(fixture_path)
native_path <- "fastkpc/build/fastkpc_cuda.so"
stopifnot(identical(file_sha(native_path),fixtures_receipt$native_sha256))
freeze_path <- file.path(work,"frozen_guard.rds")
if (split=="validation") {
  freeze <- readRDS(freeze_path)
  stopifnot(identical(freeze$fixture_sha256,file_sha(fixture_path)),
    identical(freeze$guard_code_sha256,file_sha("fastkpc/tools/research/guarded_qr_20261020/common.R")))
}
# The legacy PreparedS diagnostic DTO schema accepts only data_p=48. It does
# not use that field in numerical computation: X, n, q and penalty arrays are
# explicit. Preserve the real input p in the fixture/receipt and use the old
# protocol constant solely for this bridge, as the existing real-data tools do.
# This adapter has no access to guard flags or numerical comparison outcomes.
make_dto_protocol <- make_dto
make_dto <- function(setup,key,data_p) {
  dto <- make_dto_protocol(setup,key,data_p)
  dto$data_p <- 48L
  dto
}
library_path <- file.path(work,"guarded_qr_probe.so")
dyn.load(library_path)
symbol <- getNativeSymbolInfo("C_guarded_qr_probe",PACKAGE="guarded_qr_probe")
probe <- function(p,Y,exact) .Call(symbol,p,Y,exact,3L)
chosen <- Filter(function(f) f$split==if(split=="smoke")"calibration" else split,
  fixtures_receipt$fixtures)
if (split=="smoke") chosen <- head(chosen,1L)
dir.create(file.path(work,split),showWarnings=FALSE)

production_exact <- function(p,Y) {
  handle <- fastkpc_full_cuda_phase6_prepared_create(p,2L)
  on.exit(full_cuda_ci_multi_penalty_gcv_prepared_free_native(handle))
  Sys.setenv(FASTKPC_TEST_MULTI_PENALTY_EXACT_ONLY="1")
  on.exit(Sys.unsetenv("FASTKPC_TEST_MULTI_PENALTY_EXACT_ONLY"),add=TRUE)
  result <- fastkpc_full_cuda_phase6_optimize_prepared(handle,Y,c("target-1","target-2"))
  full_cuda_ci_multi_penalty_gcv_residual_free_native(result$residual)
  result$optimization
}

same_points <- function(p,Y,exact,quick) {
  m <- length(p$initial_log_sp)
  response <- full_cuda_ci_multi_penalty_gcv_response_transcript_for_test(
    Y,p$magic_qr_packed,p$magic_tau)
  rows <- list()
  for(t in 1:2) {
    tr <- quick$trace[[t]]
    ids <- unique(round(seq(1,nrow(tr),length.out=4L)))
    pts <- cbind(initial=p$initial_log_sp,exact_selected=exact$log_sp[,t],
      qr_selected=quick$log_sp[,t], t(tr[ids,13+seq_len(m),drop=FALSE]))
    pts <- pts[,!duplicated(apply(pts,2,sha)),drop=FALSE]
    for(j in seq_len(ncol(pts))) {
      eval <- function(force) full_cuda_ci_multi_penalty_gcv_objective_transcript_for_test(
        p,response$y0[,t],response$squared_norm[t],pts[,j],force_stable_svd=force)
      a<-eval(TRUE); b<-eval(FALSE)
      rows[[length(rows)+1L]] <- list(target=t,log_sp=pts[,j],
        exact=list(route=a$decomposition_route,rank=a$numerical_rank,info=a$solver_info,
          score=a$score,edf=a$edf,rss=a$rss,gradient=a$gradient,hessian=a$hessian),
        qr=list(route=b$decomposition_route,rank=b$numerical_rank,info=b$solver_info,
          score=b$score,gradient=b$gradient,hessian=b$hessian),
        error=c(score=abs(a$score-b$score),gradient=max(abs(a$gradient-b$gradient)),
          hessian=max(abs(a$hessian-b$hessian)),coefficients=max(abs(a$coefficients-b$coefficients))),
        edf_error=abs(a$edf-b$edf))
    }
  }
  rows
}

summaries <- details <- list()
for (i in seq_along(chosen)) {
  f <- chosen[[i]]
  path <- file.path(work,split,paste0(f$id,".rds"))
  if (file.exists(path)) {
    detail <- readRDS(path)
    stopifnot(identical(detail$fixture_hash,sha(f)),
      identical(detail$probe_sha256,file_sha(library_path)))
    summaries[[i]] <- detail$summary; details[[i]] <- detail
    next
  }
  p <- fastkpc_full_cuda_phase6_prepare(f$setup)
  before <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
  # Alternate ordering; each call does one traced warmup then three timed,
  # untraced full-trajectory GPU runs. No compilation overlaps these timings.
  if (i %% 2L) { exact<-probe(p,f$Y,TRUE); quick<-probe(p,f$Y,FALSE) }
  else { quick<-probe(p,f$Y,FALSE); exact<-probe(p,f$Y,TRUE) }
  stopifnot(exact$deterministic,quick$deterministic,all(exact$status==0L))
  prod <- production_exact(p,f$Y)
  stopifnot(identical(bits(exact$log_sp),bits(prod$selected_log_sp)),
    prod$diagnostics$cuda_exact_primary_target_count==2L,
    prod$diagnostics$cuda_stability_replay_target_count==0L)
  a <- full_finalize(p,f,exact$log_sp)
  b <- full_finalize(p,f,quick$log_sp)
  points <- same_points(p,f$Y,exact,quick)
  selected_exact <- lapply(1:2,function(t) {
    matches <- Filter(function(x)x$target==t && identical(bits(x$log_sp),bits(quick$log_sp[,t])),points)
    stopifnot(length(matches)==1L)
    matches[[1L]]$exact
  })
  # Independent exact finalization also verifies the probe's refreshed state.
  stopifnot(identical(bits(exact$coefficients),bits(a$shadow$coefficients)))
  gate <- data.frame(target=1:2,
    residual_max=vapply(1:2,function(t)max(abs(a$shadow$residuals[,t]-b$shadow$residuals[,t])),0),
    residual_rel=vapply(1:2,function(t)rel_l2(b$shadow$residuals[,t],a$shadow$residuals[,t]),0),
    fitted_max=vapply(1:2,function(t)max(abs(a$shadow$fitted[,t]-b$shadow$fitted[,t])),0),
    fitted_rel=vapply(1:2,function(t)rel_l2(b$shadow$fitted[,t],a$shadow$fitted[,t]),0),
    edf_error=abs(exact$edf-vapply(selected_exact,`[[`,0,"edf")),
    score_error=abs(exact$score-vapply(selected_exact,`[[`,0,"score")),
    rss_error=abs(exact$rss-vapply(selected_exact,`[[`,0,"rss")),
    rank_equal=a$info$numerical_rank==b$info$numerical_rank)
  target_pass <- with(gate,residual_max<=1e-7 & residual_rel<=1e-8 &
    fitted_max<=1e-7 & fitted_rel<=1e-8 & edf_error<=1e-8 & rank_equal &
    score_error<=1e-8 & score_error/pmax(abs(exact$score),1e-300)<=1e-8 &
    rss_error<=1e-8 & rss_error/pmax(abs(exact$rss),1e-300)<=1e-8)
  pv_a <- a$p$records$p_value; pv_b <- b$p$records$p_value
  pv_error <- max(abs(pv_a-pv_b))
  num_a <- a$p$numerical; num_b <- b$p$numerical
  statistic_error <- max(abs(num_a$statistic-num_b$statistic))
  moment_error <- max(abs(unlist(num_a[c("mean","variance")])-unlist(num_b[c("mean","variance")])))
  statistic_pass <- statistic_error<=1e-9 &&
    all(abs(num_a$statistic-num_b$statistic)/pmax(abs(num_a$statistic),1e-300)<=1e-8)
  moment_pass <- moment_error<=1e-10 &&
    all(abs(unlist(num_a[c("mean","variance")])-unlist(num_b[c("mean","variance")])) /
      pmax(abs(unlist(num_a[c("mean","variance")])),1e-300)<=1e-8)
  flip <- any((pv_a>=0.1)!=(pv_b>=0.1))
  features <- guard_features(quick,f$setup)
  divergence <- lapply(1:2,function(t) {
    x<-exact$trace[[t]];y<-quick$trace[[t]];n<-min(nrow(x),nrow(y))
    # Route differs by design; compare only discrete optimizer decisions.
    z<-which(rowSums(x[seq_len(n),1:4,drop=FALSE]!=y[seq_len(n),1:4,drop=FALSE])>0)
    list(first_decision_difference=if(length(z))z[1L] else if(nrow(x)!=nrow(y))n+1L else NA_integer_,
      first_log_sp_difference=which(rowSums(x[seq_len(n),-(1:13),drop=FALSE]!=
        y[seq_len(n),-(1:13),drop=FALSE])>0)[1L])
  })
  after <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
  fields <- c("live_device_allocations","live_device_bytes","live_streams","live_events")
  stopifnot(identical(before[fields],after[fields]))
  summary <- data.frame(id=f$id,dataset=f$dataset,split=f$split,
    n=nrow(f$Y),q=ncol(p$X),penalties=length(p$initial_log_sp),
    p_exact=pv_a,p_qr=pv_b,p_error=pv_error,decision_flip=flip,
    target_gate_failures=sum(!target_pass),max_residual_error=max(gate$residual_max),
    statistic_error=statistic_error,moment_error=moment_error,
    hard_failure=!(all(target_pass)&&pv_error<=1e-10&&!flip&&statistic_pass&&moment_pass),
    safety_failure=!(all(target_pass)&&pv_error<=1e-11&&!flip&&statistic_pass&&moment_pass),
    exact_ms=median(exact$device_ms),qr_ms=median(quick$device_ms),
    exact_host_ms=median(exact$host_ms),qr_host_ms=median(quick$host_ms),
    qr_evaluations=sum(quick$qr_calls),svd_evaluations=sum(quick$svd_calls),
    decision_divergences=sum(vapply(divergence,function(x)!is.na(x$first_decision_difference),TRUE)),
    reference_bitwise=TRUE,deterministic=TRUE,resources_closed=TRUE)
  detail <- list(schema="guarded-qr-cohort-diagnostic-v1",fixture_hash=sha(f),
    actual_dataset_p=f$data_p,legacy_descriptor_data_p=48L,
    probe_sha256=file_sha(library_path),native_sha256=file_sha(native_path),
    summary=summary,features=features,exact=exact,qr=quick,
    production=prod,final_exact=a,final_qr=b,target_gates=gate,
    same_points=points,divergence=divergence,resources_before=before,resources_after=after)
  saveRDS(detail,path,compress=FALSE)
  summaries[[i]]<-summary;details[[i]]<-detail
  cat(sprintf("%s %d/%d q=%d p_error=%.3e target_fail=%d exact=%.2fms QR=%.2fms\n",
    split,i,length(chosen),summary$q,pv_error,sum(!target_pass),summary$exact_ms,summary$qr_ms))
  flush.console()
}
summary <- do.call(rbind,summaries)
write.csv(summary,file.path(work,paste0(split,".csv")),row.names=FALSE)
if (split=="calibration") {
  grid <- do.call(rbind,lapply(fixtures_receipt$guard_grid,function(scale) {
    flags<-vapply(details,function(d)any(guard_flags(d$features,scale)),TRUE)
    data.frame(scale=scale,flagged_cohorts=sum(flags),
      false_negative_hard=sum(summary$hard_failure & !flags),
      false_negative_safety=sum(summary$safety_failure & !flags),
      paired_cost_ms=sum(summary$qr_ms)+sum(summary$exact_ms[flags]),
      baseline_ms=sum(summary$exact_ms))
  }))
  write.csv(grid,file.path(work,"calibration_grid.csv"),row.names=FALSE)
  eligible<-which(grid$false_negative_safety==0L)
  stopifnot(length(eligible)>0L)
  best<-eligible[which.min(grid$paired_cost_ms[eligible])]
  freeze<-list(schema="guarded-qr-frozen-empirical-guard-v1",scale=grid$scale[best],
    created_at=as.character(Sys.time()),fixture_sha256=file_sha(fixture_path),
    calibration_csv_sha256=file_sha(file.path(work,"calibration.csv")),
    guard_code_sha256=file_sha("fastkpc/tools/research/guarded_qr_20261020/common.R"),
    grid=grid,numerical_gate="existing full contract; p safety margin 1e-11 against exact GPU reference",
    timing_boundary="two-target optimizer kernel only; guard dispatch and downstream work excluded")
  if(file.exists(freeze_path)) stop("refusing to overwrite frozen guard")
  saveRDS(freeze,freeze_path,compress=FALSE)
  print(grid)
}
if(split=="validation") {
  flags<-vapply(details,function(d)any(guard_flags(d$features,freeze$scale)),TRUE)
  summary$guard_flag<-flags
  write.csv(summary,file.path(work,"validation.csv"),row.names=FALSE)
  cat(sprintf("Frozen scale=%g: flagged=%d/%d, missed hard=%d, missed safety=%d, paired kernel cost=%.3fms vs exact %.3fms\n",
    freeze$scale,sum(flags),length(flags),sum(summary$hard_failure & !flags),
    sum(summary$safety_failure & !flags),sum(summary$qr_ms)+sum(summary$exact_ms[flags]),sum(summary$exact_ms)))
}
stopifnot(identical(file_sha(native_path),fixtures_receipt$native_sha256))
