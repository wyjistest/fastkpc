# Reuse the repository's authoritative audit expressions without re-running GPU.
# Rscript .../audit_saved_canonical.R RECEIPT NEW_JSON_OUTPUT
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L, !file.exists(args[[2L]]),
          requireNamespace("jsonlite", quietly = TRUE),
          requireNamespace("digest", quietly = TRUE))
receipt <- readRDS(args[[1L]])
stopifnot(identical(receipt$workload, "canonical"))
audit_source <- "fastkpc/tools/run_phase10_pvalue_contract_audit.R"
expressions <- parse(file = audit_source)
assignment_name <- function(x) {
  if (is.call(x) && identical(x[[1L]], as.name("<-")) && is.symbol(x[[2L]])) {
    as.character(x[[2L]])
  } else ""
}
names <- vapply(expressions, assignment_name, character(1L))
first <- which(names == "tasks")
last <- which(names == "native_path") - 1L
stopifnot(length(first) == 1L, length(last) == 1L, first < last)
state <- new.env(parent = globalenv())
for (function_name in c("fail", "canonical_deletion_sepsets", "numeric_comparison")) {
  index <- which(names == function_name)
  stopifnot(length(index) == 1L)
  eval(expressions[[index]], state)
}
oracle_dir <- "fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2"
state$result <- receipt$result
state$elapsed_sec <- receipt$elapsed_sec
state$logical <- readRDS(file.path(oracle_dir, "logical_ci_trace.rds"))
state$reference_pmax <- readRDS(file.path(oracle_dir, "pmax.rds"))
state$reference_adjacency <- readRDS(file.path(oracle_dir, "adjacency.rds"))
resource_fields <- c("live_device_allocations", "live_device_bytes", "live_streams", "live_events")
state$vertical_resource_closed <- identical(receipt$resources_before[resource_fields],
                                           receipt$resources_after[resource_fields])
# This exact interval contains task/CI/pMax, authority, route, exact-primary,
# coefficient-reuse, cache-accounting, graph/sepset and resource checks only.
for (index in seq.int(first, last)) eval(expressions[[index]], state)
output <- list(
  schema = "saved-canonical-full-audit-v1",
  receipt_sha256 = digest::digest(file = args[[1L]], algo = "sha256", serialize = FALSE),
  audit_source_sha256 = digest::digest(file = audit_source, algo = "sha256", serialize = FALSE),
  native_sha256 = receipt$native_sha256,
  audit = state$audit
)
jsonlite::write_json(output, args[[2L]], pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat(if (isTRUE(state$audit$pass)) "PASS" else "FAIL", args[[1L]],
    "CI", state$audit$ci$comparable_count,
    "max_error", format(state$audit$ci$maximum_absolute_error, digits = 17), "\n")
stopifnot(isTRUE(state$audit$pass))
