# Reuse already qualified exact-fixed GPU shadows, authenticated by the prior
# manifest and the unchanged production native. This does not replay QR.
source("fastkpc/tools/research/guarded_qr_20261020/common.R")
args <- commandArgs(TRUE)
out <- file.path(args[[1L]], "witnesses")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
old <- "fastkpc/artifacts/full_cuda_ci/guarded_qr_20261020"
manifest <- jsonlite::read_json(file.path(old,"manifest.json"), simplifyVector=TRUE)
checked <- function(rel) {
  path <- file.path(old,rel)
  stopifnot(identical(file_sha(path),manifest$files[[rel]]))
  readRDS(path)
}
fixture <- checked("fixtures.rds")
capacity <- checked("capacity_reference.rds")
stopifnot(all(unlist(capacity$rows[,4:ncol(capacity$rows)])),
  identical(fixture$native_sha256,file_sha("fastkpc/build/fastkpc_cuda.so")))
parts <- list(); sources <- list()
for (f in fixture$fixtures) {
  rel <- file.path(f$split,paste0(f$id,".rds"))
  d <- checked(rel)
  stopifnot(identical(d$fixture_hash,sha(f)),
    identical(d$native_sha256,fixture$native_sha256),
    all(d$final_exact$info$solver_info==0),
    all(d$final_exact$info$decomposition_route==2))
  a <- d$final_exact
  set.seed(20261020)
  oracle <- kpcalg::dcov.gamma(a$shadow$residuals[,1],a$shadow$residuals[,2],numCol=35L)
  row <- data.frame(id=f$id,dataset=f$dataset,n=nrow(f$Y),q=ncol(f$setup$X),
    penalties=length(f$setup$penalty_blocks),
    gpu_p=a$p$records$p_value,gpu_statistic=a$p$numerical$statistic,
    gpu_mean=a$p$numerical$mean,gpu_variance=a$p$numerical$variance,
    authority_p=oracle$p.value,statistic=unname(oracle$estimates[1]),
    mean=unname(oracle$estimates[2]),variance=unname(oracle$estimates[3]))
  parts[[length(parts)+1L]] <- list(row=row,residuals=a$shadow$residuals)
  sources[[rel]] <- manifest$files[[rel]]
}
rows <- do.call(rbind,lapply(parts,`[[`,"row"))
for (n in sort(unique(rows$n))) {
  selected <- which(rows$n==n)
  values <- do.call(cbind,lapply(parts[selected],`[[`,"residuals"))
  binary <- file.path(out,paste0(n,".f64"))
  con <- file(binary,"wb"); writeBin(as.double(values),con,size=8L,endian="little"); close(con)
  write.csv(rows[selected,],file.path(out,paste0(n,".csv")),row.names=FALSE)
}
jsonlite::write_json(list(schema="structured-exact-witness-export-v1",pairs=nrow(rows),
  source_manifest_sha256=file_sha(file.path(old,"manifest.json")),sources=sources,
  native_sha256=fixture$native_sha256,n=sort(unique(rows$n)),q=sort(unique(rows$q)),
  note="Previously captured exact-fixed shadows; all 72 cohorts have independent production-capacity bit parity",
  gpu_vs_kpcalg_max_p_error=max(abs(rows$gpu_p-rows$authority_p))),
  file.path(out,"manifest.json"),pretty=TRUE,auto_unbox=TRUE,digits=NA)
print(table(rows$dataset,rows$n))
