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
requested_workers <- as.integer(arg_or_default(
  6L, parallel::detectCores(logical = TRUE)
))
requested_shards <- as.integer(arg_or_default(7L, "200"))
tolerance <- as.numeric(arg_or_default(8L, "0"))

fail <- function(message) stop(message, call. = FALSE)
require_true <- function(value, message) if (!isTRUE(value)) fail(message)
elapsed_seconds <- function(started) {
  unname((proc.time() - started)[["elapsed"]])
}
parse_conditioning_set <- function(key) {
  if (!nzchar(key)) return(integer())
  as.integer(strsplit(key, "|", fixed = TRUE)[[1L]])
}
conditioning_order_key <- function(key) {
  values <- parse_conditioning_set(key)
  if (!length(values)) return("")
  paste(sprintf("%010d", values), collapse = "|")
}
split_round_robin <- function(count, workers) {
  if (count == 0L) return(list())
  split(seq_len(count), ((seq_len(count) - 1L) %% workers) + 1L)
}
split_contiguous <- function(values, shard_count) {
  count <- length(values)
  shard_count <- min(count, shard_count)
  boundaries <- floor(seq(0, count, length.out = shard_count + 1L))
  lapply(seq_len(shard_count), function(shard) {
    values[seq.int(boundaries[[shard]] + 1L, boundaries[[shard + 1L]])]
  })
}
normalize_sepsets <- function(value) {
  unname(lapply(unname(value), function(row) {
    unname(lapply(unname(row), function(entry) sort(as.integer(entry))))
  }))
}
atomic_save_rds <- function(value, path) {
  temporary <- tempfile(".permutation-oracle-", tmpdir = dirname(path))
  on.exit(unlink(temporary, force = TRUE), add = TRUE)
  saveRDS(value, temporary, compress = "xz")
  require_true(file.rename(temporary, path),
               paste0("failed to publish ", basename(path)))
}

require_true(method %in% c("dcc.perm", "hsic.perm"),
             "method must be dcc.perm or hsic.perm")
require_true(file.exists(candidate_path), "candidate result is missing")
require_true(file.exists(data_path), "canonical data is missing")
require_true(nzchar(output_path) && !file.exists(output_path),
             "oracle output path is invalid or already exists")
require_true(!is.na(seed) && seed >= 0L, "seed must be non-negative")
require_true(!is.na(requested_workers) && requested_workers >= 1L,
             "worker count must be positive")
require_true(!is.na(requested_shards) && requested_shards >= 1L,
             "shard count must be positive")
require_true(is.finite(tolerance) && tolerance >= 0,
             "tolerance must be finite and non-negative")

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
required_task_fields <- c(
  "level", "task_index", "edge_x", "edge_y", "x", "y", "S_key",
  "p_used", "native_edge_deleted", "native_edge_ignored"
)
require_true(
  identical(dim(data), c(351L, 48L)) && is.data.frame(tasks) &&
    nrow(tasks) > 0L && all(required_task_fields %in% names(tasks)) &&
    identical(candidate$summary$ci_method, method) &&
    identical(as.integer(candidate$summary$permutation_seed), seed) &&
    identical(as.integer(candidate$summary$permutation_replicates), 100L) &&
    isTRUE(candidate$summary$permutation_include_observed) &&
    !any(tasks$native_edge_ignored),
  "candidate permutation trace contract changed"
)

task_count <- nrow(tasks)
task_keys <- paste(
  as.integer(tasks$level), as.integer(tasks$x), as.integer(tasks$y),
  as.character(tasks$S_key), sep = "\037"
)
require_true(!anyDuplicated(task_keys),
             "candidate task identity is not unique")
candidate_md5 <- unname(tools::md5sum(candidate_path))
data_md5 <- unname(tools::md5sum(data_path))

# The trace is published in LayerPlan order, but the legacy R RNG is consumed
# in pcalg order. Every edge contributes a monotone prefix, so sorting the
# consumed union by the pcalg comparator reconstructs the physical order.
conditioning_order_keys <- vapply(
  as.character(tasks$S_key), conditioning_order_key, character(1L)
)
physical_order <- order(
  tasks$level, tasks$x, tasks$y, conditioning_order_keys, method = "radix"
)
physical_rank <- integer(task_count)
physical_rank[physical_order] <- seq_len(task_count)
edge_groups <- interaction(
  tasks$level, tasks$edge_x, tasks$edge_y, drop = TRUE, lex.order = TRUE
)
edge_rank_sequences <- split(physical_rank, edge_groups)
require_true(
  all(vapply(edge_rank_sequences, function(ranks) {
    length(ranks) < 2L || all(diff(ranks) > 0L)
  }, logical(1L))),
  "candidate trace cannot reconstruct physical permutation order"
)

task_shards <- split_contiguous(physical_order, requested_shards)
shard_count <- length(task_shards)
shard_dir <- paste0(output_path, ".shards")
dir.create(shard_dir, recursive = TRUE, showWarnings = FALSE)
require_true(dir.exists(shard_dir), "failed to create oracle shard directory")
shard_paths <- file.path(
  shard_dir, sprintf("shard-%04d-of-%04d.rds", seq_len(shard_count), shard_count)
)

# Record the exact R RNG state on both sides of every contiguous shard. This
# permits independent workers without changing the global permutation stream.
rng_started <- proc.time()
rng_kind <- RNGkind()
set.seed(seed)
rng_start_states <- vector("list", shard_count)
rng_end_states <- vector("list", shard_count)
for (shard in seq_len(shard_count)) {
  rng_start_states[[shard]] <- .Random.seed
  fastkpc_consume_legacy_permutation_tasks_export(
    method, nrow(data), 100L, length(task_shards[[shard]])
  )
  rng_end_states[[shard]] <- .Random.seed
}
rng_plan_elapsed <- elapsed_seconds(rng_started)

validate_shard <- function(shard, path) {
  if (!file.exists(path)) return(NULL)
  value <- tryCatch(readRDS(path), error = function(error) NULL)
  if (is.null(value)) return(NULL)
  expected_tasks <- task_shards[[shard]]
  valid <- is.list(value) &&
    identical(value$schema_version,
              "fastkpc-strict-permutation-trace-shard-v1") &&
    identical(value$method, method) && identical(value$seed, seed) &&
    identical(value$candidate_md5, candidate_md5) &&
    identical(value$data_md5, data_md5) &&
    identical(value$shard, shard) &&
    identical(value$shard_count, shard_count) &&
    identical(value$task_indices, expected_tasks) &&
    identical(value$task_keys, task_keys[expected_tasks]) &&
    identical(value$rng_start_state, rng_start_states[[shard]]) &&
    identical(value$rng_end_state, rng_end_states[[shard]]) &&
    isTRUE(value$rng_state_identical) &&
    length(value$cpu_p_values) == length(expected_tasks) &&
    all(is.finite(value$cpu_p_values))
  if (valid) value else NULL
}

existing_shards <- Map(validate_shard, seq_len(shard_count), shard_paths)
missing_shards <- which(vapply(existing_shards, is.null, logical(1L)))
invalid_existing_paths <- shard_paths[
  missing_shards[file.exists(shard_paths[missing_shards])]
]
if (length(invalid_existing_paths)) {
  require_true(all(unlink(invalid_existing_paths, force = TRUE) == 0L),
               "failed to remove invalid oracle shards")
}

residual_elapsed <- 0
residual_worker_seconds <- numeric()
pvalue_elapsed <- 0
pvalue_worker_seconds <- numeric()
workers_used <- 0L
if (length(missing_shards) > 0L) {
  target_values <- c(as.integer(tasks$x), as.integer(tasks$y))
  conditioning_values <- rep(as.character(tasks$S_key), 2L)
  residual_keys_all <- paste(target_values, conditioning_values, sep = "\037")
  first_occurrence <- !duplicated(residual_keys_all)
  residual_keys <- residual_keys_all[first_occurrence]
  residual_targets <- target_values[first_occurrence]
  residual_conditioning <- conditioning_values[first_occurrence]
  residual_count <- length(residual_keys)
  residual_index <- setNames(seq_len(residual_count), residual_keys)
  left_index <- unname(residual_index[paste(
    as.integer(tasks$x), as.character(tasks$S_key), sep = "\037"
  )])
  right_index <- unname(residual_index[paste(
    as.integer(tasks$y), as.character(tasks$S_key), sep = "\037"
  )])
  require_true(!anyNA(left_index) && !anyNA(right_index),
               "residual identity mapping is incomplete")

  env <- fastkpc_legacy_env()
  residual_workers <- min(requested_workers, residual_count)
  residual_started <- proc.time()
  residual_chunks <- parallel::mclapply(
    split_round_robin(residual_count, residual_workers),
    function(indices) {
      values <- matrix(NA_real_, nrow = nrow(data), ncol = length(indices))
      started <- proc.time()
      for (position in seq_along(indices)) {
        request <- indices[[position]]
        target <- residual_targets[[request]]
        conditioning_set <- parse_conditioning_set(
          residual_conditioning[[request]]
        )
        values[, position] <- if (length(conditioning_set) == 0L) {
          data[, target]
        } else {
          as.numeric(env$regrXonS(
            data[, target], data[, conditioning_set, drop = FALSE]
          )[, 1L])
        }
      }
      list(
        indices = indices,
        values = values,
        elapsed_sec = elapsed_seconds(started)
      )
    },
    mc.cores = residual_workers,
    mc.set.seed = FALSE,
    mc.cleanup = TRUE,
    mc.allow.recursive = FALSE,
    mc.preschedule = TRUE
  )
  residuals <- matrix(NA_real_, nrow = nrow(data), ncol = residual_count)
  for (chunk in residual_chunks) residuals[, chunk$indices] <- chunk$values
  require_true(all(is.finite(residuals)),
               "CPU oracle produced a non-finite residual")
  residual_elapsed <- elapsed_seconds(residual_started)
  residual_worker_seconds <- vapply(
    residual_chunks, `[[`, numeric(1L), "elapsed_sec"
  )
  rm(residual_chunks)
  invisible(gc())

  workers_used <- min(requested_workers, length(missing_shards))
  pvalue_started <- proc.time()
  shard_results <- parallel::mclapply(
    missing_shards,
    function(shard) {
      indices <- task_shards[[shard]]
      assign(".Random.seed", rng_start_states[[shard]], envir = .GlobalEnv)
      values <- numeric(length(indices))
      started <- proc.time()
      for (position in seq_along(indices)) {
        task <- indices[[position]]
        left <- residuals[, left_index[[task]]]
        right <- residuals[, right_index[[task]]]
        values[[position]] <- if (method == "dcc.perm") {
          unname(env$dcov.test(
            x = left, y = right, index = 1, R = 100L
          )$p.value)
        } else {
          unname(env$hsic.perm(
            x = left, y = right, sig = 1, p = 100L, numCol = 35L
          )$p.value)
        }
      }
      final_state <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
      result <- list(
        schema_version = "fastkpc-strict-permutation-trace-shard-v1",
        method = method,
        seed = seed,
        candidate_md5 = candidate_md5,
        data_md5 = data_md5,
        shard = shard,
        shard_count = shard_count,
        task_indices = indices,
        task_keys = task_keys[indices],
        physical_ranks = physical_rank[indices],
        rng_start_state = rng_start_states[[shard]],
        rng_end_state = rng_end_states[[shard]],
        rng_state_identical = identical(final_state, rng_end_states[[shard]]),
        cpu_p_values = values,
        elapsed_sec = elapsed_seconds(started)
      )
      atomic_save_rds(result, shard_paths[[shard]])
      cat(sprintf(
        "permutation oracle shard complete: method=%s shard=%d/%d tasks=%d elapsed=%.3f\n",
        method, shard, shard_count, length(indices), result$elapsed_sec
      ))
      result
    },
    mc.cores = workers_used,
    mc.set.seed = FALSE,
    mc.cleanup = TRUE,
    mc.allow.recursive = FALSE,
    mc.preschedule = FALSE
  )
  require_true(
    length(shard_results) == length(missing_shards) &&
      all(vapply(shard_results, function(value) {
        is.list(value) && isTRUE(value$rng_state_identical)
      }, logical(1L))),
    "permutation oracle worker or RNG-state gate failed"
  )
  pvalue_elapsed <- elapsed_seconds(pvalue_started)
  pvalue_worker_seconds <- vapply(
    shard_results, `[[`, numeric(1L), "elapsed_sec"
  )
}

completed_shards <- Map(validate_shard, seq_len(shard_count), shard_paths)
require_true(!any(vapply(completed_shards, is.null, logical(1L))),
             "permutation oracle shard set is incomplete")
require_true(all(vapply(completed_shards, function(value) {
  isTRUE(value$rng_state_identical)
}, logical(1L))), "permutation oracle RNG state mismatch")

cpu_p_values <- rep(NA_real_, task_count)
for (shard in completed_shards) {
  cpu_p_values[shard$task_indices] <- shard$cpu_p_values
}
require_true(all(is.finite(cpu_p_values)),
             "permutation oracle p-value merge is incomplete")

candidate_p_values <- as.numeric(tasks$p_used)
differences <- abs(candidate_p_values - cpu_p_values)
candidate_decisions <- candidate_p_values >= 0.1
cpu_decisions <- cpu_p_values >= 0.1
decision_flip_count <- sum(candidate_decisions != cpu_decisions)
deletion_trace_identical <- identical(
  as.logical(cpu_decisions), as.logical(tasks$native_edge_deleted)
)

p <- ncol(data)
cpu_adjacency <- matrix(TRUE, nrow = p, ncol = p)
diag(cpu_adjacency) <- FALSE
cpu_pmax <- matrix(-Inf, nrow = p, ncol = p)
diag(cpu_pmax) <- 1
cpu_sepsets <- lapply(seq_len(p), function(unused) vector("list", p))
for (task in seq_len(task_count)) {
  x <- as.integer(tasks$x[[task]])
  y <- as.integer(tasks$y[[task]])
  p_value <- cpu_p_values[[task]]
  if (p_value > cpu_pmax[x, y]) {
    cpu_pmax[x, y] <- p_value
    cpu_pmax[y, x] <- p_value
  }
  if (cpu_decisions[[task]]) {
    conditioning_set <- parse_conditioning_set(
      as.character(tasks$S_key[[task]])
    )
    cpu_adjacency[x, y] <- cpu_adjacency[y, x] <- FALSE
    cpu_sepsets[[x]][[y]] <- conditioning_set
    cpu_sepsets[[y]][[x]] <- conditioning_set
  }
}

candidate_adjacency <- candidate$adjacency != 0
shd <- sum(abs(
  as.integer(cpu_adjacency) - as.integer(candidate_adjacency)
)) / 2L
pmax_difference <- max(abs(cpu_pmax - unname(candidate$pMax)))
level_counts <- as.integer(table(factor(
  as.integer(tasks$level), levels = as.integer(candidate$levels$level)
)))
p_mismatch_indices <- which(differences > tolerance)
decision_flip_indices <- which(candidate_decisions != cpu_decisions)

receipt <- list(
  schema_version = "fastkpc-strict-permutation-trace-oracle-v1",
  candidate_path = normalizePath(candidate_path),
  candidate_md5 = candidate_md5,
  data_path = normalizePath(data_path),
  data_md5 = data_md5,
  method = method,
  n = nrow(data),
  p = ncol(data),
  alpha = 0.1,
  seed = seed,
  replicates = 100L,
  numCol = 35L,
  tolerance = tolerance,
  rng_kind = rng_kind,
  rng_plan_elapsed_sec = rng_plan_elapsed,
  requested_workers = requested_workers,
  workers_used = workers_used,
  shard_count = shard_count,
  reused_shard_count = shard_count - length(missing_shards),
  computed_shard_count = length(missing_shards),
  residual_elapsed_sec = residual_elapsed,
  residual_worker_sec = if (length(residual_worker_seconds)) {
    summary(residual_worker_seconds)
  } else {
    numeric()
  },
  pvalue_elapsed_sec = pvalue_elapsed,
  pvalue_worker_sec = if (length(pvalue_worker_seconds)) {
    summary(pvalue_worker_seconds)
  } else {
    numeric()
  },
  task_count = task_count,
  level_counts = level_counts,
  max_abs_p_diff = max(differences),
  p_value_mismatch_count = sum(differences > tolerance),
  first_p_mismatch = if (length(p_mismatch_indices)) {
    tasks[p_mismatch_indices[[1L]],
          c("level", "x", "y", "S_key", "p_used"),
          drop = FALSE]
  } else {
    NULL
  },
  decision_flip_count = decision_flip_count,
  first_decision_flip = if (length(decision_flip_indices)) {
    tasks[decision_flip_indices[[1L]],
          c("level", "x", "y", "S_key", "p_used",
            "native_edge_deleted"), drop = FALSE]
  } else {
    NULL
  },
  deletion_trace_identical = deletion_trace_identical,
  adjacency_identical = identical(
    unname(cpu_adjacency), unname(candidate_adjacency)
  ),
  SHD = as.integer(shd),
  sepsets_identical = identical(
    normalize_sepsets(cpu_sepsets), normalize_sepsets(candidate$sepsets)
  ),
  pmax_within_tolerance = is.finite(pmax_difference) &&
    pmax_difference <= tolerance,
  max_abs_pmax_diff = pmax_difference,
  n_edgetests_identical = identical(
    level_counts, as.integer(candidate$n.edgetests)
  ),
  cpu_p_values = cpu_p_values,
  differences = differences
)
receipt$pass <- identical(receipt$p_value_mismatch_count, 0L) &&
  identical(receipt$decision_flip_count, 0L) &&
  isTRUE(receipt$deletion_trace_identical) &&
  identical(receipt$SHD, 0L) && isTRUE(receipt$adjacency_identical) &&
  isTRUE(receipt$sepsets_identical) &&
  isTRUE(receipt$pmax_within_tolerance) &&
  isTRUE(receipt$n_edgetests_identical)

atomic_save_rds(receipt, output_path)
cat(sprintf(
  paste0(
    "strict permutation trace oracle: method=%s tasks=%d shards=%d ",
    "max_abs_p_diff=%.17g mismatches=%d flips=%d SHD=%d pass=%s\n"
  ),
  method, task_count, shard_count, receipt$max_abs_p_diff,
  receipt$p_value_mismatch_count, decision_flip_count, receipt$SHD,
  receipt$pass
))
require_true(receipt$pass, "strict permutation trace oracle gate failed")
