# Convert a completed full-gate receipt for existing read-only trace analysis.
args <- commandArgs(TRUE)
stopifnot(length(args)==2L,!file.exists(file.path(args[[2]],"profile-result.json")))
receipt <- readRDS(file.path(args[[1]],"receipt.rds"))
report <- jsonlite::read_json(file.path(args[[1]],"report.json"),simplifyVector=TRUE)
hook <- jsonlite::read_json(file.path(args[[1]],"hook.json"),simplifyVector=TRUE)
stopifnot(all(unlist(report$gates)),all(unlist(hook$gates)),!hook$capture)
r <- receipt$result
stopifnot(length(r$tasks$p_used)==240498L,nrow(r$summary$prefill_batches)>0L)
jsonlite::write_json(list(elapsed=report$elapsed_sec,native_sha256=report$native_sha256,
  bitwise_vs_reference=report$gates$p_value_bitwise_exact,levels=r$levels,
  prefill_batches=r$summary$prefill_batches,
  evidence_class="CUPTI diagnostic only; no timing claim"),
  file.path(args[[2]],"profile-result.json"),pretty=TRUE,auto_unbox=TRUE,digits=NA)
