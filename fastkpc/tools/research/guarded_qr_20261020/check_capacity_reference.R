# Qualify the isolated base80 probe against production capacity dispatch too.
# This is untimed and runs after the calibration/validation measurements.
source("fastkpc/tools/research/guarded_qr_20261020/common.R")
args <- commandArgs(TRUE); work <- args[[1L]]
out <- file.path(work,"capacity_reference.rds")
stopifnot(!file.exists(out))
fixtures <- readRDS(file.path(work,"fixtures.rds"))$fixtures
Sys.setenv(FASTKPC_TEST_MULTI_PENALTY_EXACT_ONLY="1")
on.exit(Sys.unsetenv("FASTKPC_TEST_MULTI_PENALTY_EXACT_ONLY"))
load_fastkpc_cuda_native()
rows <- list()
for(i in seq_along(fixtures)) {
  f <- fixtures[[i]]
  detail <- readRDS(file.path(work,f$split,paste0(f$id,".rds")))
  p <- fastkpc_full_cuda_phase6_prepare(f$setup)
  ref <- .Call("C_full_cuda_ci_multi_penalty_capacity_qualify",
    p$X,p$magic_qr_packed,as.double(p$magic_tau),p$magic_r,as.integer(p$magic_pivot),
    p$penalty_roots,p$penalty_matrices,as.integer(p$penalty_ranks),
    as.double(p$initial_log_sp),f$Y,NULL,PACKAGE="fastkpc_cuda")
  r <- data.frame(id=f$id,q=ncol(p$X),bucket=ref$capacity_bucket,
    sp_bits=identical(bits(detail$exact$log_sp),bits(ref$selected_log_sp)),
    coefficient_bits=identical(bits(detail$final_exact$shadow$coefficients),bits(ref$fixed_coefficients)),
    fitted_bits=identical(bits(detail$final_exact$shadow$fitted),bits(ref$fixed_fitted)),
    residual_bits=identical(bits(detail$final_exact$shadow$residuals),bits(ref$fixed_residuals)),
    rank_exact=identical(as.integer(detail$final_exact$info$numerical_rank),as.integer(ref$fixed_numerical_rank)),
    info_zero=all(ref$fixed_solver_info==0L),route_svd=all(ref$fixed_decomposition_route==2L),
    primary_exact=ref$diagnostics$cuda_exact_primary_target_count==2L,
    cpu_zero=all(unlist(ref$diagnostics[c("cpu_objective_count","cpu_optimizer_count","cpu_multi_penalty_solve_count","fallback_count")])==0),
    slots_released=ref$fixed_residual_slot_released && ref$optimizer_residual_slot_released)
  stopifnot(all(unlist(r[,4:ncol(r)])))
  rows[[i]] <- r
  cat(sprintf("capacity reference %d/%d: %s PASS\n",i,length(fixtures),r$bucket))
}
rows<-do.call(rbind,rows)
saveRDS(list(schema="guarded-qr-production-capacity-reference-v1",rows=rows,
  native_sha256=file_sha("fastkpc/build/fastkpc_cuda.so")),out,compress=FALSE)
write.csv(rows,file.path(work,"capacity_reference.csv"),row.names=FALSE)
