# Offline authority/resource audit of the observed production prefix.
args<-commandArgs(TRUE);stopifnot(length(args)==1L)
x<-readRDS(file.path(args[[1L]],'capture.rds'));s<-x$result$summary
fields<-c('r_callback_count','legacy_mgcv_fit_count','legacy_mgcv_setup_count',
 'cpu_residual_solve_count','cpu_dcov_component_count','cpu_dcov_eigen_or_lowrank_count',
 'cpu_dcov_pair_stat_count','cpu_gamma_pvalue_count','cpu_spectra_count',
 'unknown_fallback_count','approximate_backend_count')
gates<-list(capture=all(unlist(x$gates)),
 cpu_authority_fallback_zero=all(vapply(fields,function(k)!is.null(s[[k]])&&s[[k]]==0,logical(1))))
stopifnot(all(unlist(gates)))
cat(jsonlite::toJSON(list(gates=gates,counters=s[fields],ci_count=nrow(x$result$tasks)),
 auto_unbox=TRUE,pretty=TRUE))
