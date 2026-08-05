source("fastkpc/R/native.R")
source("fastkpc/R/legacy_runner.R")

args <- commandArgs(trailingOnly = TRUE)
arg_or_default <- function(index, default) {
  if (length(args) >= index && nzchar(args[[index]])) args[[index]] else default
}

candidate_path <- arg_or_default(1L, "")
method <- arg_or_default(2L, "")
output_path <- arg_or_default(3L, "")
data_path <- arg_or_default(
  4L,
  "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"
)
seed <- as.integer(arg_or_default(5L, "707"))

fail <- function(message) stop(message, call. = FALSE)
require_true <- function(value, message) if (!isTRUE(value)) fail(message)
parse_conditioning_set <- function(key) {
  if (!nzchar(key)) return(integer())
  as.integer(strsplit(key, "|", fixed = TRUE)[[1L]])
}
conditioning_order_key <- function(key) {
  values <- parse_conditioning_set(key)
  if (!length(values)) return("")
  paste(sprintf("%010d", values), collapse = "|")
}

require_true(method %in% c("dcc.perm", "hsic.perm"),
             "method must be dcc.perm or hsic.perm")
require_true(file.exists(candidate_path), "candidate result is missing")
require_true(file.exists(data_path), "canonical data is missing")
require_true(nzchar(output_path) && !file.exists(output_path),
             "boundary output path is invalid or already exists")
require_true(!is.na(seed) && seed >= 0L, "seed must be non-negative")

build_fastkpc_native()
payload <- readRDS(candidate_path)
candidate <- if (is.list(payload) && !is.null(payload$result)) {
  payload$result
} else {
  payload
}
tasks <- candidate$tasks
data <- as.matrix(readRDS(data_path))
storage.mode(data) <- "double"
require_true(
  identical(dim(data), c(351L, 48L)) && is.data.frame(tasks) &&
    nrow(tasks) > 0L && identical(candidate$summary$ci_method, method) &&
    !any(tasks$native_edge_ignored),
  "candidate permutation trace contract changed"
)

replicates <- 100L
grid_counts <- round(as.numeric(tasks$p_used) * (replicates + 1L))
require_true(
  all(abs(as.numeric(tasks$p_used) * (replicates + 1L) - grid_counts) <=
        1e-12),
  "candidate p-values are not on the expected permutation grid"
)
selected_indices <- which(grid_counts %in% c(10, 11))
require_true(length(selected_indices) > 0L,
             "candidate has no alpha-boundary permutation tasks")

# The one-call scheduler consumes R RNG in pcalg task order, while the
# published trace is replayed in LayerPlan order. Reconstruct the physical
# order before advancing the legacy RNG stream.
conditioning_order_keys <- vapply(
  as.character(tasks$S_key), conditioning_order_key, character(1L)
)
physical_order <- order(
  tasks$level, tasks$x, tasks$y, conditioning_order_keys, method = "radix"
)
physical_rank <- integer(nrow(tasks))
physical_rank[physical_order] <- seq_len(nrow(tasks))
edge_groups <- interaction(
  tasks$level, tasks$edge_x, tasks$edge_y, drop = TRUE, lex.order = TRUE
)
edge_rank_sequences <- split(physical_rank, edge_groups)
require_true(
  all(vapply(edge_rank_sequences, function(ranks) {
    length(ranks) < 2L || all(diff(ranks) > 0L)
  }, logical(1L))),
  "candidate trace cannot reconstruct the physical permutation order"
)
selected_execution_indices <- physical_order[physical_order %in%
                                                selected_indices]

env <- fastkpc_legacy_env()
suff_stat <- list(
  data = data,
  ic.method = method,
  index = 1,
  numCol = 35L,
  sig = 1,
  p = replicates
)
cpu_p_values <- numeric(length(selected_indices))
started <- proc.time()
previous_rank <- 0L
set.seed(seed)
for (execution_position in seq_along(selected_execution_indices)) {
  task_index <- selected_execution_indices[[execution_position]]
  task_rank <- physical_rank[[task_index]]
  skipped <- task_rank - previous_rank - 1L
  if (skipped > 0L) {
    fastkpc_consume_legacy_permutation_tasks_export(
      method, nrow(data), replicates, skipped
    )
  }
  task <- tasks[task_index, , drop = FALSE]
  selected_position <- match(task_index, selected_indices)
  cpu_p_values[[selected_position]] <- env$kernelCItest(
    as.integer(task$x), as.integer(task$y),
    parse_conditioning_set(as.character(task$S_key)), suff_stat
  )
  previous_rank <- task_rank
  if (execution_position %% 25L == 0L ||
      execution_position == length(selected_execution_indices)) {
    cat(sprintf(
      paste0("boundary oracle progress: method=%s checked=%d/%d ",
             "task=%d physical_rank=%d\n"),
      method, execution_position, length(selected_execution_indices),
      task_index, task_rank
    ))
  }
}
elapsed <- unname((proc.time() - started)[["elapsed"]])

candidate_p_values <- as.numeric(tasks$p_used[selected_indices])
differences <- abs(candidate_p_values - cpu_p_values)
candidate_decisions <- candidate_p_values >= 0.1
cpu_decisions <- cpu_p_values >= 0.1
decision_flip_count <- sum(candidate_decisions != cpu_decisions)
receipt <- list(
  schema_version = "fastkpc-strict-permutation-boundary-oracle-v2",
  candidate_path = normalizePath(candidate_path),
  data_path = normalizePath(data_path),
  method = method,
  seed = seed,
  replicates = replicates,
  alpha = 0.1,
  task_count = nrow(tasks),
  selected_task_count = length(selected_indices),
  selected_task_indices = selected_indices,
  selected_task_physical_ranks = physical_rank[selected_indices],
  selected_task_execution_order = selected_execution_indices,
  selected_tasks = tasks[selected_indices, c(
    "level", "x", "y", "S_key", "p_used", "native_edge_deleted"
  ), drop = FALSE],
  cpu_p_values = cpu_p_values,
  differences = differences,
  max_abs_p_diff = max(differences),
  p_value_mismatch_count = sum(differences != 0),
  decision_flip_count = decision_flip_count,
  elapsed_sec = elapsed
)
receipt$pass <- identical(receipt$decision_flip_count, 0L) &&
  identical(receipt$p_value_mismatch_count, 0L)

temporary <- tempfile(".strict-permutation-boundary-",
                      tmpdir = dirname(output_path))
on.exit(unlink(temporary, force = TRUE), add = TRUE)
saveRDS(receipt, temporary, compress = "xz")
require_true(file.rename(temporary, output_path),
             "failed to publish boundary oracle")
cat(sprintf(
  paste0(
    "strict permutation boundary oracle: method=%s tasks=%d ",
    "max_abs_p_diff=%.17g flips=%d elapsed=%.3f pass=%s\n"
  ),
  method, length(selected_indices), max(differences), decision_flip_count,
  elapsed, receipt$pass
))
require_true(receipt$pass, "strict permutation boundary oracle gate failed")
