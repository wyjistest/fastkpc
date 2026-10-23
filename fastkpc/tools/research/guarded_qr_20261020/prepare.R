source("fastkpc/tools/research/guarded_qr_20261020/common.R")
args <- commandArgs(TRUE)
out <- args[[1L]]
if (file.exists(out)) stop("fixture output exists")
canonical <- readRDS("fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds")
fixtures <- list()
add <- function(id, dataset, setup, Y, targets, S, split, data_p, source) {
  if (ncol(setup$X) > 80L) return(invisible(NULL))
  fixtures[[length(fixtures)+1L]] <<- list(id = id, dataset = dataset,
    setup = make_setup(setup), Y = unname(Y), targets = targets, S = S,
    split = split, data_p = data_p, source = source)
}
# Reuse the declared historical hard-case list, not its test execution.
test_path <- "fastkpc/tests/test_full_cuda_ci_multi_penalty_cuda_optimizer.R"
expressions <- parse(test_path)
case_assignment <- Filter(function(x) is.call(x) && identical(x[[1L]], as.name("<-")) &&
  identical(x[[2L]], as.name("cases")), as.list(expressions))
stopifnot(length(case_assignment) == 1L)
eval(case_assignment[[1L]])
for (i in seq_len(nrow(cases))) {
  c <- cases[i,]
  path <- sprintf("fastkpc/artifacts/full_cuda_ci/prepared_s_contract_v1/shards/shard_%d.rds", c$shard_id)
  shard <- readRDS(path); setup <- shard$prepared_s_setups[[c$prepared_s_key_sha256]]
  states <- shard$target_states
  w <- if (is.na(c$residual_key_sha256)) integer() else match(c$residual_key_sha256, states$residual_key_sha256)
  chosen <- head(unique(c(w,which(states$prepared_s_key_sha256 == c$prepared_s_key_sha256))),2L)
  stopifnot(length(chosen)==2L, !anyNA(chosen))
  targets <- states$target[chosen]
  add(paste0("formal-",c$label), "canonical", setup, canonical[,targets,drop=FALSE],
    targets, setup$sorted_S, if (i %% 2L) "calibration" else "validation", ncol(canonical),
    list(path=path, sha256=file_sha(path), key=c$prepared_s_key_sha256))
}

paths <- c(canonical = "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds",
  `20230331145508` = "causalDiscovery/20230331145508/combined_data_shuffle-causalDiscoveryInput.RData",
  `20240102191119` = "causalDiscovery/20240102191119/GSE99254-482col-causalDiscoveryInput.RData",
  `20241215091815` = "causalDiscovery/20241215091815/logged-lung-exhausted-allData-causalDiscoveryInput.RData")
for (dataset in names(paths)) {
  if (dataset == "canonical") {
    data <- canonical
    task_path <- "/tmp/fastkpc-exact-speedup-20261020/canonical-queues32.rds"
    tasks <- readRDS(task_path)$result$tasks
  } else {
    env <- new.env(); names <- load(paths[[dataset]], env)
    stopifnot(length(names)==1L); data <- as.matrix(env[[names]])
    task_path <- sprintf("causalDiscovery/check_20260808/results/%s_dcc.gamma.rds",dataset)
    tasks <- readRDS(task_path)$result$skeleton$tasks
  }
  storage.mode(data) <- "double"
  # Fixed task identities only: the old real-data p-values are invalid and
  # are deliberately never used as numerical references or for selection.
  for (level in 3:8) {
    candidates <- tasks[tasks$level==level,]
    candidates <- candidates[!duplicated(candidates$S_key),]
    if (!nrow(candidates)) next
    ids <- unique(as.integer(round(seq(1,nrow(candidates),length.out=2L))))
    for (j in seq_along(ids)) {
      row <- candidates[ids[j],]
      S <- as.integer(strsplit(row$S_key,"|",fixed=TRUE)[[1L]])
      targets <- as.integer(c(row$x,row$y))
      native <- full_cuda_ci_native_setup_native(data[,S,drop=FALSE])
      add(sprintf("%s-L%d-%d",dataset,level,j), dataset, native,
        data[,targets,drop=FALSE], targets, S,
        if (dataset=="canonical" && j==1L) "calibration" else "validation",
        ncol(data), list(path=paths[[dataset]], sha256=file_sha(paths[[dataset]]),
          tasks=task_path, task_sha256=file_sha(task_path),
          canonical_test_order_id=row$canonical_test_order_id))
    }
  }
}
ids <- vapply(fixtures, `[[`, "", "id")
stopifnot(!anyDuplicated(ids))
groups <- vapply(fixtures,function(f)paste(f$dataset,paste(f$S,collapse=",")),"")
splits <- vapply(fixtures,`[[`,"","split")
stopifnot(all(vapply(split(splits,groups),function(x)length(unique(x))==1L,TRUE)))
receipt <- list(schema="guarded-qr-captured-fixtures-v1", created_at=as.character(Sys.time()),
  producer=system2("git",c("rev-parse","HEAD"),stdout=TRUE),
  native_sha256=file_sha("fastkpc/build/fastkpc_cuda.so"),
  guard_grid=c(0,1/1024,1/64,1/4,1,4,16,64,1024,Inf),
  policy="calibrate only declared calibration cohorts; freeze before independent development validation; no sealed holdout",
  fixtures=fixtures)
saveRDS(receipt,out,compress=FALSE)
print(table(vapply(fixtures,`[[`,"","split"),vapply(fixtures,function(x)ncol(x$setup$X),0L)))
cat("fixture SHA256",file_sha(out),"\n")
