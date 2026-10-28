dyn.load(Sys.getenv("FASTKPC_L3_STACK_LIBRARY"))
failure <- tryCatch({
  source(Sys.getenv("FASTKPC_L3_ORIGINAL_RUNNER"), local=.GlobalEnv)
  NULL
}, error=function(e)e)
closure <- .Call("fastkpc_l3_stack_finish")
closure$error <- if(is.null(failure)) NULL else conditionMessage(failure)
jsonlite::write_json(closure, file.path(Sys.getenv("FASTKPC_L3_PROBE_OUT"),"stack-closure.json"),
  pretty=TRUE,auto_unbox=TRUE,digits=NA)
print(closure)
if(!is.null(failure)) stop(conditionMessage(failure),call.=FALSE)
stopifnot(isTRUE(closure$resources_closed),closure$observation_errors==0,
  closure$l3_launches==4065,closure$l3_requests==4065,
  closure$optimizer_setups_bitwise_exact==4065,closure$optimizer_targets_bitwise_exact==54878)
