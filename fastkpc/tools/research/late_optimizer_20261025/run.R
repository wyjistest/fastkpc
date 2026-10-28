# Full original canonical qualification, with explicit research instrumentation.
args <- commandArgs(TRUE)
stopifnot(length(args)==4L,args[[2L]]=="1")
hook <- Sys.getenv("FASTKPC_LATE_OBSERVER")
stopifnot(nzchar(hook),Sys.getenv("FASTKPC_LATE_L8_PRELOAD") %in% c("0","1"))
dyn.load(hook)
hook_before <- .Call("late_optimizer_stats")
stopifnot(hook_before$registered_functions==5,hook_before$attribute_calls==0,
          hook_before$l8_prepared==0)
original <- "fastkpc/tools/research/device_properties_cache_20261025/run.R"
script_lines <- readLines(original,warn=FALSE)
old_label <- "GPU1 isolated incremental source-closure development; not full rebuild, clean producer or promotion"
new_label <- "GPU1-exclusive late-optimizer research interposer; host sharing recorded separately; not isolated runtime qualification or promotion"
stopifnot(sum(grepl(old_label,script_lines,fixed=TRUE))==1L)
script_lines <- sub(old_label,new_label,script_lines,fixed=TRUE)
eval(parse(text=script_lines),envir=.GlobalEnv)
hook_after <- .Call("late_optimizer_stats")
preload_on <- Sys.getenv("FASTKPC_LATE_L8_PRELOAD")=="1"
capture_on <- nzchar(Sys.getenv("FASTKPC_LATE_CAPTURE"))
hook_gates <- list(registry=hook_after$registered_functions==5,
  nine_l8_setups=hook_after$l8_prepared==9,errors=hook_after$errors==0,
  requested_mode=identical(hook_after$enabled,as.double(preload_on)),
  preload_count=hook_after$attribute_calls==if(preload_on)5 else 0,
  preload_once=hook_after$preload_batches==if(preload_on)1 else 0,
  capture_count=hook_after$capture_windows==if(capture_on)5 else 0,
  no_timing_shadow=capture_on||hook_after$capture_diagnostic_d2h_bytes==0)
if(preload_on)hook_gates$headroom <- min(hook_after$free_before,hook_after$free_after)>=2*1024^3 &&
  hook_after$free_before-hook_after$free_after<=256*1024^2
jsonlite::write_json(list(schema="late-optimizer-hook-v1",before=hook_before,after=hook_after,
  gates=hook_gates,capture=capture_on,preload=preload_on,receipt_sha256=sha(file.path(out,"receipt.rds")),
  diagnostic_capture_d2h_bytes=hook_after$capture_diagnostic_d2h_bytes),
  file.path(out,"hook.json"),pretty=TRUE,auto_unbox=TRUE,digits=NA)
stopifnot(all(unlist(hook_gates)))
print(hook_after)
