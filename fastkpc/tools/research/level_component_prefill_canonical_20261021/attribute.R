# Offline joins of exact component identities, actual GPU payload hashes and CI.
args <- commandArgs(TRUE); stopifnot(length(args)==1L)
w <- args[[1L]]
read_table <- function(p) read.delim(p,stringsAsFactors=FALSE,colClasses="character",check.names=FALSE)
a <- read_table(file.path(w,"audit-off","components.tsv"))
b <- read_table(file.path(w,"audit-on","components.tsv"))
c <- read_table(file.path(w,"audit-on","comparison.tsv"))
pa <- read_table(file.path(w,"audit-off","pairs.tsv"))
pb <- read_table(file.path(w,"audit-on","pairs.tsv"))
off <- readRDS(file.path(w,"canonical-off","receipt.rds"))
on <- readRDS(file.path(w,"canonical-on","receipt.rds"))
stopifnot(nrow(a)==65694L,nrow(b)==65694L,nrow(c)==65694L,
  !anyDuplicated(a$key),!anyDuplicated(b$key),!anyDuplicated(c$key),
  setequal(a$key,b$key),setequal(a$key,c$key),
  identical(pa[names(pa)!="p_bits"],pb[names(pb)!="p_bits"]))
b <- b[match(a$key,b$key),]; c <- c[match(a$key,c$key),]
identity_fields <- c("key","request","prepared","q","info","route","rank","y","coefficient")
stopifnot(identical(a[identity_fields],b[identity_fields]))
stopifnot(all(as.integer(b$width)>=1L & as.integer(b$width)<=16L))
wide <- as.integer(a$width)>=17L
fields <- c("fitted","residual","vectors","selected_values","total","self_moment")
different <- setNames(lapply(fields,function(f) a[[f]]!=b[[f]]),fields)
payload_changed <- Reduce(`|`,different[c("vectors","selected_values","total","self_moment")])
stopifnot(length(unique(a$request[wide]))==54L,
  all(!Reduce(`|`,different)[!wide]),all(!payload_changed|different$residual),
  all(as.integer(c$residual_changed)==as.integer(different$residual)),
  all(as.integer(c$vectors_changed)==as.integer(different$vectors)))
data <- as.matrix(readRDS("fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"))
y_sha <- vapply(seq_len(ncol(data)),function(i) digest::digest(
  writeBin(as.double(data[,i]),raw(),size=8),algo="sha256",serialize=FALSE),"")
target <- match(a$y,y_sha); stopifnot(!anyNA(target),!anyDuplicated(y_sha))
tasks <- off$result$tasks
s_keys <- unique(tasks$S_key[tasks$level>=3])
prepared_sha <- setNames(vapply(s_keys,function(s) digest::digest(paste0(
  "schema=full-cuda-ci-prepared-s-key-v1\ndataset=",off$result$summary$dataset_key,
  "\nS=",s,"\nformula_route=additive-smooth\nsemantic=mgcv-1.9-1-tprs-native-setup-v1\n"),
  algo="sha256",serialize=FALSE),""),s_keys)
component_s <- s_keys[match(a$prepared,prepared_sha)]
stopifnot(!anyNA(component_s))
key_target <- setNames(target,a$key)
key_s <- setNames(component_s,a$key)
left <- unname(key_target[pa$left]); right <- unname(key_target[pa$right])
pair_s <- unname(key_s[pa$left])
stopifnot(!anyNA(left),!anyNA(right),identical(pair_s,unname(key_s[pa$right])))
pair_identity <- paste(pair_s,left,right,sep="/")
stopifnot(!anyDuplicated(pair_identity),all(pa$prepared==unname(prepared_sha[pair_s])))
canonical_identity <- paste(tasks$S_key,tasks$x,tasks$y,sep="/")
at <- match(canonical_identity,pair_identity)
stopifnot(!anyNA(at[tasks$level>=3]))
# Bind the observed physical pair output to the canonical logical result, not
# only to its (S,x,y) lookup. The C++ observer prints the IEEE-754 uint64 in hex.
p_bits <- function(x) {
  bytes <- matrix(as.integer(writeBin(as.double(x),raw(),size=8,endian="little")),nrow=8)
  apply(bytes[8:1,,drop=FALSE],2L,function(b) paste(sprintf("%02x",b),collapse=""))
}
observed <- which(tasks$level>=3)
stopifnot(identical(unname(p_bits(tasks$p_used[observed])),pa$p_bits[at[observed]]),
  identical(unname(p_bits(on$result$tasks$p_used[observed])),pb$p_bits[at[observed]]))
different_p <- colSums(matrix(writeBin(as.double(on$result$tasks$p_used),raw(),size=8)!=
  writeBin(as.double(tasks$p_used),raw(),size=8),nrow=8))>0
stopifnot(!anyNA(at[different_p]))
changed_keys <- a$key[payload_changed]
affected <- !is.na(at) & (pa$left[at]%in%changed_keys|pa$right[at]%in%changed_keys)
stopifnot(all(!different_p|affected))
changed <- on$result$tasks[different_p,,drop=FALSE]
changed$baseline_p <- tasks$p_used[different_p]
changed$absolute_difference <- abs(changed$p_used-changed$baseline_p)
changed$left_component <- pa$left[at[different_p]]
changed$right_component <- pa$right[at[different_p]]
changed$left_changed <- changed$left_component%in%changed_keys
changed$right_changed <- changed$right_component%in%changed_keys
changed$left_baseline_width <- a$width[match(changed$left_component,a$key)]
changed$right_baseline_width <- a$width[match(changed$right_component,a$key)]
write.csv(changed,file.path(w,"attributed-ci.csv"),row.names=FALSE)
saveRDS(changed,file.path(w,"attributed-ci.rds"),compress=FALSE)
materialization <- data.frame(key=a$key,S_key=component_s,target=target,
  original_width=as.integer(a$width),new_width=as.integer(b$width),wide=wide,
  as.data.frame(different),stringsAsFactors=FALSE)
write.csv(materialization,file.path(w,"component-attribution.csv"),row.names=FALSE)
finite <- is.finite(on$result$pMax)&is.finite(off$result$pMax)
pmax_changes <- which(finite & on$result$pMax!=off$result$pMax,arr.ind=TRUE)
edge_key <- function(x,y) paste(pmin(x,y),pmax(x,y),sep="|")
changed_edges <- edge_key(changed$x,changed$y)
stopifnot(all(edge_key(pmax_changes[,1],pmax_changes[,2])%in%changed_edges))
edge_identity <- edge_key(tasks$edge_x,tasks$edge_y)
pmax_attribution <- lapply(unique(edge_key(pmax_changes[,1],pmax_changes[,2])),function(edge) {
  indices <- which(edge_identity==edge)
  i <- indices[which.max(tasks$p_used[indices])]
  j <- indices[which.max(on$result$tasks$p_used[indices])]
  x <- tasks$edge_x[i]; y <- tasks$edge_y[i]
  stopifnot(identical(off$result$pMax[x,y],tasks$p_used[i]),
    identical(on$result$pMax[x,y],on$result$tasks$p_used[j]),
    any(different_p[c(i,j)] & affected[c(i,j)]))
  data.frame(edge=edge,x=x,y=y,off_task_row=i,on_task_row=j,
    off_pmax=off$result$pMax[x,y],on_pmax=on$result$pMax[x,y],
    off_task_changed_payload=affected[i],on_task_changed_payload=affected[j])
})
pmax_attribution <- do.call(rbind,pmax_attribution)
write.csv(pmax_attribution,file.path(w,"pmax-attribution.csv"),row.names=FALSE)
errors <- vapply(c[grepl("error$",names(c))],function(x) max(as.numeric(x)),0)
near <- tasks$p_used>=.05 & tasks$p_used<=.15
sha <- function(p) digest::digest(file=p,algo="sha256",serialize=FALSE)
report <- list(schema="canonical-dense-prefill-attribution-v1",status="PASS",
  native_sha256=on$native_sha256,components=nrow(a),original_wide_cohorts=54L,
  original_wide_first_miss_targets=sum(wide),unaffected_components=sum(!wide),
  component_difference_counts=as.list(vapply(different,sum,0L)),max_materialization_errors=as.list(errors),
  ci_count=nrow(tasks),changed_ci=sum(different_p),unattributed_ci=sum(different_p&!affected),
  observed_physical_pairs=nrow(pa),physical_to_logical_p_bits_exact=TRUE,
  max_ci_difference=max(abs(on$result$tasks$p_used-tasks$p_used)),
  near_alpha_changed=sum(different_p&near),near_alpha_max_difference=
    max(abs(on$result$tasks$p_used[near]-tasks$p_used[near])),
  changed_finite_pmax=nrow(pmax_changes),changed_pmax_edges=nrow(pmax_attribution),
  pmax_maximum_donors_verified=TRUE,oracle_errors=as.list(on$oracle_errors),
  diagnostic_only=TRUE,production_cpu_authority=0,
  input_sha256=setNames(lapply(c("audit-off/components.tsv","audit-on/components.tsv",
    "audit-off/pairs.tsv","audit-on/pairs.tsv","audit-on/comparison.tsv",
    "canonical-off/receipt.rds","canonical-on/receipt.rds"),function(p) sha(file.path(w,p))),
    c("off_components","on_components","off_pairs","on_pairs","comparison","off_receipt","on_receipt")))
jsonlite::write_json(report,file.path(w,"attribution.json"),pretty=TRUE,auto_unbox=TRUE,digits=NA)
print(report[c("status","original_wide_first_miss_targets","component_difference_counts",
  "max_materialization_errors","changed_ci","unattributed_ci","max_ci_difference","oracle_errors")])
