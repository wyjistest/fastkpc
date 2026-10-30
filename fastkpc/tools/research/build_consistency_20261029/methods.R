# GPU-only cross-native regression. No fresh CPU numerical oracle is invoked.
args <- commandArgs(TRUE)
stopifnot(length(args)==3L)
out <- args[[1L]]; native <- normalizePath(args[[2L]],mustWork=TRUE)
reference <- if(args[[3L]]=="-") NULL else readRDS(args[[3L]])
source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
.fastkpc_cuda_so <- local({path<-native;function()path})
stopifnot(fastkpc_cuda_available(), !file.exists(file.path(out,"receipt.rds")))
Sys.setenv(FASTKPC_STRICT_METHOD_DEFERRED_SVD_SUBMISSION="1",
  FASTKPC_STRICT_METHOD_PERMUTATION_GPU_OVERLAP="1",
  FASTKPC_STRICT_PERMUTATION_RESIDUAL_ROUTE="stable-svd")
sha <- function(p)digest::digest(file=p,algo="sha256",serialize=FALSE)
native_sha <- sha(native)
eq <- function(a,b)identical(a,b,num.eq=FALSE,single.NA=FALSE)
resources <- function() {
  fixed <- .Call("C_fixed_sp_cuda_test_resource_snapshot",PACKAGE="fastkpc_cuda")
  fixed <- fixed[grep("_active_count$",names(fixed))]
  vertical <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
  list(fixed=fixed,vertical=vertical[c("live_device_allocations","live_device_bytes","live_streams","live_events")])
}
set.seed(9127L)
n <- 56L; z <- runif(n,-2,2)
nonlinear <- cbind(x1=sin(z)+rnorm(n,sd=.08),x2=cos(z)+rnorm(n,sd=.08),
  x3=z+rnorm(n,sd=.08),x4=z^2+rnorm(n,sd=.08),x5=z^3+rnorm(n,sd=.08))
set.seed(9231L)
common <- rnorm(80L)
dense <- sapply(seq_len(6L),function(i)common+.4*rnorm(80L))
colnames(dense) <- paste0("x",seq_len(6L))
real <- as.matrix(readRDS("fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"))[,1:6,drop=FALSE]
storage.mode(real) <- "double"
cases <- list(nonlinear0=list(data=nonlinear,level=0L),nonlinear1=list(data=nonlinear,level=1L),
  nonlinear2=list(data=nonlinear,level=2L),dense3=list(data=dense,level=3L),
  real351x6=list(data=real,level=3L))
authority <- c("r_callback_count","legacy_mgcv_fit_count","legacy_mgcv_setup_count",
  "cpu_residual_solve_count","cpu_dcov_component_count","cpu_dcov_eigen_or_lowrank_count",
  "cpu_dcov_pair_stat_count","cpu_gamma_pvalue_count","cpu_spectra_count",
  "residual_d2h_bytes","component_d2h_bytes","unknown_fallback_count","approximate_backend_count")
task_fields <- c("level","task_index","x","y","S_key","canonical_test_order_id",
  "native_edge_deleted","native_edge_ignored")
results <- list(); reports <- list()
for(case in names(cases)) for(method in c("dcc.perm","hsic.gamma","hsic.perm")) {
  id <- paste(case,method,sep="/"); input <- cases[[case]]
  invisible(full_cuda_ci_one_call_cache_control_native("reset"))
  before <- resources(); stopifnot(all(unlist(before)==0))
  set.seed(707L); seed_before <- .Random.seed
  started <- proc.time()[["elapsed"]]
  value <- precision_run_skeleton_full_cuda_native(data=input$data,alpha=.1,
    max_conditioning_size=input$level,index=1,numCol=35L,trace_level="logical",
    compatible_cuda_strict=TRUE,ci_method=method,hsic_params=list(sig=1),
    permutation_params=list(replicates=100L,seed=707L,include_observed=TRUE))
  elapsed <- proc.time()[["elapsed"]]-started
  seed_after <- .Random.seed; after <- resources(); s <- value$summary
  gates <- list(resources_zero=all(unlist(after)==0) && eq(before,after),
    authority_zero=all(vapply(authority,function(k)!is.null(s[[k]]) && s[[k]]==0,logical(1))),
    authority_gate=isTRUE(s$authority_gate_pass),
    p_finite=all(is.finite(value$tasks$p_used)),
    fresh_result_cache=s$result_cache_dataset_warm_start_entries==0,
    fresh_target_cache=s$target_cache_dataset_warm_start_entries==0)
  permutation <- method!="hsic.gamma"
  gates$rng_receipts <- if(permutation)
    s$method_permutation_rng_receipt_count==s$frontier_batch_count &&
    s$method_permutation_rng_exact_receipt_count==s$frontier_batch_count else
    s$method_permutation_rng_receipt_count==0 && eq(seed_before,seed_after)
  status_names <- grep("route|status|rank|solver_info|rng_.*sha256",names(s),value=TRUE)
  status_names <- status_names[!grepl("_ms$",status_names)]
  if(!is.null(reference)) {
    old <- reference$results[[id]]; stopifnot(!is.null(old))
    gates$tasks <- eq(value$tasks[task_fields],old$result$tasks[task_fields])
    gates$p_bits <- eq(value$tasks$p_used,old$result$tasks$p_used)
    gates$pmax_bits <- eq(value$pMax,old$result$pMax)
    gates$adjacency <- eq(value$adjacency,old$result$adjacency)
    gates$sepsets <- eq(value$sepsets,old$result$sepsets)
    gates$counts <- eq(value$n.edgetests,old$result$n.edgetests)
    gates$route_status <- eq(s[status_names],old$result$summary[status_names])
    gates$rng_endpoint <- eq(seed_after,old$seed_after) && eq(seed_before,old$seed_before)
  }
  results[[id]] <- list(result=value,seed_before=seed_before,seed_after=seed_after,
    resources_before=before,resources_after=after,gates=gates,elapsed_sec=elapsed)
  reports[[id]] <- list(case=case,method=method,logical_ci=nrow(value$tasks),
    levels=sort(unique(value$tasks$level)),elapsed_sec=elapsed,gates=gates,
    route_status=s[status_names],maximum_p_difference=if(is.null(reference))NA_real_ else
      if(eq(value$tasks[task_fields],old$result$tasks[task_fields]))max(abs(value$tasks$p_used-old$result$tasks$p_used)) else Inf)
  saveRDS(list(native_sha256=native_sha,results=results),file.path(out,"pending.rds"),compress=FALSE)
  print(reports[[id]])
  stopifnot(all(unlist(gates)))
}
stopifnot(sha(native)==native_sha)
receipt <- list(status="PASS",native_sha256=native_sha,results=results,
  reference_sha256=if(is.null(reference))NULL else sha(args[[3L]]))
saveRDS(receipt,file.path(out,"receipt.rds"),compress=FALSE)
jsonlite::write_json(list(status="PASS",native_sha256=native_sha,reports=reports,
  receipt_sha256=sha(file.path(out,"receipt.rds")),reference_sha256=receipt$reference_sha256,
  scope="GPU cross-native regression on five fixtures; not a new CPU-authority qualification"),
  file.path(out,"report.json"),pretty=TRUE,auto_unbox=TRUE,digits=NA)
cat("PASS three-method cross-native p/status/graph/RNG/resource regression\n")
