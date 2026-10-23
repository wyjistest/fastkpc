# Close only the final native/source identity, after all gates have passed.
args <- commandArgs(TRUE)
stopifnot(length(args)==2L)
work <- args[[1L]]
out <- args[[2L]]
stopifnot(!dir.exists(out))
sha <- function(p) digest::digest(file=p,algo="sha256",serialize=FALSE)
native <- "fastkpc/build/fastkpc_cuda.so"
native_sha <- sha(native)
runs <- c("final-off", "final-on", "final-payload")
receipts <- lapply(runs,function(nm) readRDS(file.path(work,nm,"receipt.rds")))
stopifnot(all(vapply(receipts,function(r) identical(r$native_sha256,native_sha) &&
  all(unlist(r$gates)),logical(1L))))
payload <- read.table(file.path(work,"final-payload-audit.log"),stringsAsFactors=FALSE)
stopifnot(nrow(payload)==6632L,length(unique(payload[[1L]]))==6632L,all(payload[[2L]]=="PASS"))
reference_dir <- Sys.getenv("FASTKPC_LSCP_REFERENCE")
reference_manifest <- "fastkpc/artifacts/full_cuda_ci/component_payload_20261021/native/raw-manifest.json"
stopifnot(dir.exists(reference_dir))
reference_files <- jsonlite::read_json(reference_manifest,simplifyVector=FALSE)
for (key in payload[[1L]]) {
  nm <- paste0(key,".f64"); p <- file.path(reference_dir,nm)
  stopifnot(identical(sha(p),reference_files[[nm]]$sha256),
            unname(file.info(p)$size)==reference_files[[nm]]$bytes)
}
stopifnot(any(grepl("ERROR SUMMARY: 0 errors",readLines(file.path(work,"final-memcheck.log")),fixed=TRUE)),
          any(grepl("0 bytes leaked",readLines(file.path(work,"final-memcheck.log")),fixed=TRUE)),
          any(grepl("PASS",readLines(file.path(work,"final-configuration-test.log")),fixed=TRUE)),
          any(grepl("PASS",readLines(file.path(work,"final-cache-regression.log")),fixed=TRUE)))
sources <- c("fastkpc/src/full_cuda_ci_one_call.cpp", "fastkpc/src/cuda/full_cuda_ci_vertical.cu",
             "fastkpc/src/cuda/full_cuda_ci_vertical.hpp")
for (nm in runs) {
  report <- jsonlite::read_json(file.path(work,nm,"report.json"),simplifyVector=FALSE)
  stopifnot(identical(unlist(report$source_sha256),setNames(vapply(sources,sha,""),sources)))
}
dir.create(out,recursive=TRUE)
stopifnot(file.copy(reference_manifest,file.path(out,"reference-payload-manifest.json")))
for (nm in runs) {
  dest <- file.path(out,nm); dir.create(dest)
  stopifnot(all(file.copy(file.path(work,nm,c("receipt.rds","report.json")),dest)))
}
for (nm in c("final-off.log","final-on.log","final-payload.log","final-payload-audit.log",
             "final-memcheck.log","final-configuration-test.log","final-cache-regression.log",
             "build-final.log","final-lifecycle-build.log","audit_payload.so","cache_lifecycle_final"))
  stopifnot(file.copy(file.path(work,nm),out))
dir.create(file.path(out,"sources"))
stopifnot(all(file.copy(sources,file.path(out,"sources"))))
stopifnot(system2("gzip",c("-n","-c",shQuote(native)),
  stdout=file.path(out,"fastkpc_cuda.so.gz")) == 0L)
system2("git",c("diff","--","fastkpc/src/full_cuda_ci_one_call.cpp",
  "fastkpc/src/cuda/full_cuda_ci_vertical.cu","fastkpc/src/cuda/full_cuda_ci_vertical.hpp"),
  stdout=file.path(out,"runtime.patch"))
before <- receipts[[1L]]$elapsed_sec; after <- receipts[[2L]]$elapsed_sec
decision <- list(schema="level-component-prefill-closure-v1",status="PASS_SUBSET22_OPT_IN_PROTOTYPE",
  native_sha256=native_sha,accepted_native_sha256="fe8286f7e6c6114ef66e040b16b693f5abcb23333dc9c0fd767c042c509c3c50",
  logical_ci=24264L,produced=7270L,resolved=6632L,unconsumed=638L,payload_bitwise_exact=6632L,
  verified_reference_payload_files=6632L,
  off_elapsed_sec=before,on_elapsed_sec=after,observed_reduction=1-after/before,
  evidence_class="single source-closure development comparison; not promotion",
  production_default=FALSE,canonical_qualification="NOT_RUN",phase10="ACTIVE",
  candidate="NOT_FROZEN",holdout="SEALED",recommended=FALSE)
jsonlite::write_json(decision,file.path(out,"closure.json"),pretty=TRUE,auto_unbox=TRUE,digits=NA)
tools <- list.files("fastkpc/tools/research/level_component_prefill_20261021",full.names=TRUE)
tools <- c(tools,"fastkpc/tests/test_full_cuda_ci_level_component_prefill.R")
files <- list.files(out,recursive=TRUE,full.names=TRUE)
manifest <- list(schema="level-component-prefill-manifest-v1",status="PASS",native_sha256=native_sha,
  files=setNames(lapply(files,function(p) list(bytes=unname(file.info(p)$size),sha256=sha(p))),
                 substring(files,nchar(out)+2L)),
  tools=as.list(setNames(vapply(tools,sha,""),tools)),
  inputs=list(accepted_subset_receipt_sha256=sha("fastkpc/artifacts/full_cuda_ci/production_cohort_frozen_20261021/baseline-inf/receipt.rds"),
              accepted_payload_manifest_sha256=sha("fastkpc/artifacts/full_cuda_ci/component_payload_20261021/native/raw-manifest.json")))
jsonlite::write_json(manifest,file.path(out,"manifest.json"),pretty=TRUE,auto_unbox=TRUE,digits=NA)
cat("PASS archive",out,"\nmanifest",sha(file.path(out,"manifest.json")),"\n")
