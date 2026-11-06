source("fastkpc/tools/research/numcol_20261106/common.R")
args <- commandArgs(TRUE)
stopifnot(length(args) == 3L, !file.exists(args[[3L]]))
native <- args[[1L]]; reference <- readRDS(args[[2L]]); out <- args[[3L]]
dir.create(out)
.fastkpc_cuda_so <- local({ path <- native; function() path })
rank_mode("on")
stopifnot(fastkpc_cuda_available())
invisible(full_cuda_ci_one_call_cache_control_native("reset"))
data <- rank_data("public500")
stopifnot(reference$numCol == 50L,
  identical(reference$input_sha256, digest::digest(data, algo = "sha256")))
set.seed(707L)
candidate <- fastkpc_compatible_cuda_wanpdag(data, options = list(
  ci_method = "dcc.gamma", trace_level = "logical"))
rng <- .Random.seed
oracle <- fastkpc_full_cuda_phase10_development_oracle(reference$result)
comparison <- fastkpc_full_cuda_compare_candidate_skeleton(oracle, candidate$skeleton)
trace <- fastkpc_full_cuda_normalize_logical_trace(candidate$skeleton)
ci <- numeric_check(trace$p_value, oracle$logical_trace$p_value)
pmax <- numeric_check(candidate$skeleton$pMax, oracle$reference$pMax)
suff_stat <- function(data) list(data = data, ic.method = "dcc.gamma", index = 1,
  numCol = as.integer(floor(nrow(data) / 10)))
set.seed(707L)
expected <- orientation_reference(fastkpc_skeleton_as_pc_algo(oracle$reference, data), data)
gates <- c(list(skeleton = isTRUE(comparison$summary$pass), ci = ci$pass, pmax = pmax$pass,
  default_rank = candidate$skeleton$summary$numCol == 50L,
  rng = identical(rng, expected$rng_end), resources = !is.null(rank_resources())),
  compare_orientation(candidate$orientation, expected))
saveRDS(list(candidate = candidate, reference = expected), file.path(out, "receipt.rds"), compress = FALSE)
jsonlite::write_json(list(status = if (all(vapply(gates, isTRUE, logical(1L)))) "PASS" else "FAIL",
  gates = gates, ci = ci, pmax = pmax, ci_count = nrow(trace),
  native_sha256 = digest::digest(file = native, algo = "sha256", serialize = FALSE)),
  file.path(out, "report.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
print(gates); stopifnot(all(vapply(gates, isTRUE, logical(1L))))
cat("PASS public 500x50 WAN-PDAG with omitted numCol\n")
