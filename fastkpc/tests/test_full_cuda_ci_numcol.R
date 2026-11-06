# Requires a working CUDA native; the qualification runner treats absence as FAIL.
source("fastkpc/tools/research/numcol_20261106/common.R")
stopifnot(fastkpc_cuda_available())
rank_mode("off")
data <- rank_data("dense140")
run_rank <- function(k) {
  cat("RUN rank", k, "\n"); flush.console()
  precision_run_skeleton_full_cuda_native(data, .1,
    max_conditioning_size = 2L, numCol = k, trace_level = "logical")
}
invisible(full_cuda_ci_one_call_cache_control_native("reset"))
bad <- list(0L, -1L, nrow(data), nrow(data) + 1L, NA_integer_, NaN, Inf,
  50.7, "50", TRUE, integer(), c(35L, 50L))
set.seed(707L); seed <- .Random.seed
for (value in bad) {
  for (direct in c(FALSE, TRUE)) {
    before <- rank_resources()[rank_live]
    err <- tryCatch({
      if (direct) .Call("C_full_cuda_ci_one_call_skeleton", data, .1, 0L, 1,
        value, "logical", TRUE, PACKAGE = "fastkpc_cuda") else run_rank(value)
      NULL
    }, error = identity)
    stopifnot(inherits(err, "error"), grepl("numCol", conditionMessage(err)),
      identical(before, rank_resources()[rank_live]), identical(seed, .Random.seed))
  }
}
a <- run_rank(7L)
b <- run_rank(50L)
c <- run_rank(65L)
warm <- run_rank(7L)
stopifnot(a$summary$numCol == 7L, b$summary$numCol == 50L, c$summary$numCol == 65L,
  a$summary$result_cache_hit_count == 0,
  b$summary$result_cache_hit_count == 0, c$summary$result_cache_hit_count == 0,
  warm$summary$result_cache_hit_count == warm$summary$logical_tests_consumed,
  warm$summary$physical_tests_evaluated == 0L,
  warm$summary$legacy_component_eig_active_lane_peak == 0L,
  identical(a$tasks, warm$tasks), identical(a$pMax, warm$pMax))
for (item in list(list(k = 50L, result = b), list(k = 65L, result = c))) {
  invisible(full_cuda_ci_one_call_cache_control_native("reset"))
  cold <- run_rank(item$k)
  stopifnot(identical(cold$tasks, item$result$tasks), identical(cold$pMax, item$result$pMax),
    identical(cold$adjacency, item$result$adjacency))
}
invisible(rank_resources())
cat("PASS numCol validation: 24 rejects, rank 7/50/65 cache separation and warm replay\n")
