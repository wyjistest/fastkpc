source("fastkpc/R/legacy_runner.R")

args <- commandArgs(trailingOnly = TRUE)
arg_or_default <- function(index, default) {
  if (length(args) >= index && nzchar(args[[index]])) args[[index]] else default
}

candidate_path <- arg_or_default(
  1L, "/tmp/fastkpc-strict-hsic-gamma-351x48-inf.rds"
)
data_path <- arg_or_default(
  2L,
  "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"
)
output_path <- arg_or_default(
  3L, "/tmp/fastkpc-strict-hsic-gamma-351x48-inf-oracle.rds"
)
workers <- as.integer(arg_or_default(4L, parallel::detectCores(logical = TRUE)))
tolerance <- as.numeric(arg_or_default(5L, "1e-10"))
reuse_oracle_path <- arg_or_default(6L, "")

fail <- function(message) stop(message, call. = FALSE)
require_true <- function(value, message) if (!isTRUE(value)) fail(message)
elapsed_seconds <- function(started) {
  unname((proc.time() - started)[["elapsed"]])
}
split_indices <- function(count, worker_count) {
  if (count == 0L) return(list())
  split(seq_len(count), ((seq_len(count) - 1L) %% worker_count) + 1L)
}
parse_conditioning_set <- function(key) {
  if (!nzchar(key)) return(integer())
  as.integer(strsplit(key, "|", fixed = TRUE)[[1L]])
}
normalize_sepsets <- function(value) {
  unname(lapply(unname(value), function(row) {
    unname(lapply(unname(row), function(entry) sort(as.integer(entry))))
  }))
}

require_true(file.exists(candidate_path), "candidate result is missing")
require_true(file.exists(data_path), "canonical data is missing")
require_true(!file.exists(output_path), "oracle output already exists")
require_true(!is.na(workers) && workers >= 1L,
             "worker count must be a positive integer")
require_true(is.finite(tolerance) && tolerance >= 0,
             "p-value tolerance must be finite and non-negative")
if (nzchar(reuse_oracle_path)) {
  require_true(file.exists(reuse_oracle_path), "reuse oracle is missing")
}

payload <- readRDS(candidate_path)
candidate <- if (is.list(payload) && !is.null(payload$result)) {
  payload$result
} else {
  payload
}
data <- as.matrix(readRDS(data_path))
storage.mode(data) <- "double"
tasks <- candidate$tasks
require_true(
  is.matrix(data) && identical(dim(data), c(351L, 48L)) &&
    is.data.frame(tasks) && nrow(tasks) > 0L &&
    identical(candidate$summary$ci_method, "hsic.gamma") &&
    identical(as.integer(candidate$levels$level), seq.int(0L, 8L)),
  "candidate is not the strict 351x48 default-Inf hsic.gamma run"
)
required_task_fields <- c(
  "level", "x", "y", "S_key", "p_used", "native_edge_deleted",
  "native_edge_ignored"
)
require_true(all(required_task_fields %in% names(tasks)),
             "candidate logical trace is incomplete")
require_true(!any(tasks$native_edge_ignored),
             "candidate trace contains ignored physical rows")

task_count <- nrow(tasks)
task_keys <- paste(
  as.integer(tasks$level), as.integer(tasks$x), as.integer(tasks$y),
  as.character(tasks$S_key), sep = "\037"
)
require_true(!anyDuplicated(task_keys),
             "candidate task identity is not unique")

cpu_p_values <- rep(NA_real_, task_count)
reuse_hit_count <- 0L
if (nzchar(reuse_oracle_path)) {
  reuse_receipt <- readRDS(reuse_oracle_path)
  require_true(
    is.list(reuse_receipt) &&
      identical(as.integer(reuse_receipt$n), nrow(data)) &&
      identical(as.integer(reuse_receipt$p), ncol(data)) &&
      identical(as.numeric(reuse_receipt$alpha), 0.1) &&
      identical(as.numeric(reuse_receipt$sig), 1) &&
      identical(as.integer(reuse_receipt$numCol), 35L) &&
      length(reuse_receipt$cpu_p_values) == reuse_receipt$task_count &&
      file.exists(reuse_receipt$candidate_path),
    "reuse oracle contract is incompatible"
  )
  reuse_payload <- readRDS(reuse_receipt$candidate_path)
  reuse_candidate <- if (is.list(reuse_payload) &&
                         !is.null(reuse_payload$result)) {
    reuse_payload$result
  } else {
    reuse_payload
  }
  reuse_tasks <- reuse_candidate$tasks
  reuse_task_keys <- paste(
    as.integer(reuse_tasks$level), as.integer(reuse_tasks$x),
    as.integer(reuse_tasks$y), as.character(reuse_tasks$S_key),
    sep = "\037"
  )
  require_true(
    is.data.frame(reuse_tasks) && !anyDuplicated(reuse_task_keys) &&
      length(reuse_task_keys) == length(reuse_receipt$cpu_p_values) &&
      all(is.finite(reuse_receipt$cpu_p_values)),
    "reuse oracle task payload is invalid"
  )
  reuse_positions <- match(task_keys, reuse_task_keys)
  reuse_hits <- !is.na(reuse_positions)
  cpu_p_values[reuse_hits] <- as.numeric(
    reuse_receipt$cpu_p_values[reuse_positions[reuse_hits]]
  )
  reuse_hit_count <- sum(reuse_hits)
}

missing_tasks <- which(!is.finite(cpu_p_values))
missing_task_count <- length(missing_tasks)
all_target_values <- c(as.integer(tasks$x), as.integer(tasks$y))
all_conditioning_values <- rep(as.character(tasks$S_key), 2L)
unique_residual_count <- length(unique(paste(
  all_target_values, all_conditioning_values, sep = "\037"
)))
target_values <- c(
  as.integer(tasks$x[missing_tasks]), as.integer(tasks$y[missing_tasks])
)
conditioning_values <- rep(as.character(tasks$S_key[missing_tasks]), 2L)
residual_keys_all <- paste(target_values, conditioning_values, sep = "\037")
first_occurrence <- !duplicated(residual_keys_all)
residual_keys <- residual_keys_all[first_occurrence]
residual_targets <- target_values[first_occurrence]
residual_conditioning <- conditioning_values[first_occurrence]
residual_count <- length(residual_keys)
residual_index <- setNames(seq_len(residual_count), residual_keys)
left_index <- unname(residual_index[paste(
  as.integer(tasks$x[missing_tasks]),
  as.character(tasks$S_key[missing_tasks]), sep = "\037"
)])
right_index <- unname(residual_index[paste(
  as.integer(tasks$y[missing_tasks]),
  as.character(tasks$S_key[missing_tasks]), sep = "\037"
)])
require_true(!anyNA(left_index) && !anyNA(right_index),
             "residual identity mapping is incomplete")

requested_workers <- workers
workers <- min(workers, residual_count, missing_task_count)
env <- fastkpc_legacy_env()
residual_elapsed <- 0
residual_worker_seconds <- numeric()
pvalue_elapsed <- 0
pvalue_worker_seconds <- numeric()
if (missing_task_count > 0L) {
  residual_started <- proc.time()
  residual_chunks <- parallel::mclapply(
    split_indices(residual_count, workers),
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
    mc.cores = workers,
    mc.set.seed = FALSE,
    mc.cleanup = TRUE,
    mc.allow.recursive = FALSE,
    mc.preschedule = TRUE
  )
  residuals <- matrix(NA_real_, nrow = nrow(data), ncol = residual_count)
  for (chunk in residual_chunks) {
    residuals[, chunk$indices] <- chunk$values
  }
  require_true(all(is.finite(residuals)),
               "CPU oracle produced a non-finite residual")
  residual_elapsed <- elapsed_seconds(residual_started)
  residual_worker_seconds <- vapply(
    residual_chunks, `[[`, numeric(1L), "elapsed_sec"
  )
  rm(residual_chunks)
  invisible(gc())

  pvalue_started <- proc.time()
  pvalue_chunks <- parallel::mclapply(
    split_indices(missing_task_count, workers),
    function(indices) {
      values <- numeric(length(indices))
      started <- proc.time()
      for (position in seq_along(indices)) {
        task <- indices[[position]]
        values[[position]] <- unname(env$hsic.gamma(
          residuals[, left_index[[task]]],
          residuals[, right_index[[task]]],
          sig = 1, numCol = 35L
        )$p.value)
      }
      list(
        indices = indices,
        values = values,
        elapsed_sec = elapsed_seconds(started)
      )
    },
    mc.cores = workers,
    mc.set.seed = FALSE,
    mc.cleanup = TRUE,
    mc.allow.recursive = FALSE,
    mc.preschedule = TRUE
  )
  for (chunk in pvalue_chunks) {
    cpu_p_values[missing_tasks[chunk$indices]] <- chunk$values
  }
  pvalue_elapsed <- elapsed_seconds(pvalue_started)
  pvalue_worker_seconds <- vapply(
    pvalue_chunks, `[[`, numeric(1L), "elapsed_sec"
  )
  rm(pvalue_chunks)
  invisible(gc())
}
require_true(all(is.finite(cpu_p_values)),
             "CPU oracle produced a non-finite HSIC p-value")

candidate_p_values <- as.numeric(tasks$p_used)
differences <- abs(candidate_p_values - cpu_p_values)
candidate_decisions <- candidate_p_values >= 0.1
cpu_decisions <- cpu_p_values >= 0.1
decision_flip_count <- sum(candidate_decisions != cpu_decisions)

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
    S <- parse_conditioning_set(as.character(tasks$S_key[[task]]))
    cpu_adjacency[x, y] <- cpu_adjacency[y, x] <- FALSE
    cpu_sepsets[[x]][[y]] <- S
    cpu_sepsets[[y]][[x]] <- S
  }
}

candidate_adjacency <- candidate$adjacency != 0
shd <- sum(abs(
  as.integer(cpu_adjacency) - as.integer(candidate_adjacency)
)) / 2L
pmax_difference <- max(abs(cpu_pmax - unname(candidate$pMax)))
sepsets_identical <- identical(
  normalize_sepsets(cpu_sepsets), normalize_sepsets(candidate$sepsets)
)
level_counts <- as.integer(table(
  factor(as.integer(tasks$level), levels = as.integer(candidate$levels$level))
))
n_edgetests_identical <- identical(
  level_counts, as.integer(candidate$n.edgetests)
)

quantile_probabilities <- c(0, 0.5, 0.9, 0.99, 0.999, 1)
receipt <- list(
  schema_version = "fastkpc-strict-hsic-gamma-trace-oracle-v2",
  candidate_path = normalizePath(candidate_path),
  data_path = normalizePath(data_path),
  n = nrow(data),
  p = ncol(data),
  alpha = 0.1,
  sig = 1,
  numCol = 35L,
  workers = requested_workers,
  workers_used = workers,
  tolerance = tolerance,
  reuse_oracle_path = if (nzchar(reuse_oracle_path)) {
    normalizePath(reuse_oracle_path)
  } else {
    ""
  },
  reused_task_count = reuse_hit_count,
  computed_task_count = missing_task_count,
  level_counts = level_counts,
  task_count = task_count,
  unique_residual_count = unique_residual_count,
  computed_unique_residual_count = residual_count,
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
  max_abs_p_diff = max(differences),
  p_diff_quantiles = stats::quantile(
    differences, probabilities = quantile_probabilities, names = TRUE
  ),
  p_values_within_tolerance = all(differences <= tolerance),
  decision_flip_count = decision_flip_count,
  adjacency_identical = identical(
    unname(cpu_adjacency), unname(candidate_adjacency)
  ),
  SHD = as.integer(shd),
  sepsets_identical = sepsets_identical,
  pmax_within_tolerance = is.finite(pmax_difference) &&
    pmax_difference <= tolerance,
  max_abs_pmax_diff = pmax_difference,
  n_edgetests_identical = n_edgetests_identical,
  cpu_p_values = cpu_p_values,
  differences = differences
)
receipt$pass <- isTRUE(receipt$p_values_within_tolerance) &&
  identical(receipt$decision_flip_count, 0L) &&
  identical(receipt$SHD, 0L) && isTRUE(receipt$adjacency_identical) &&
  isTRUE(receipt$sepsets_identical) &&
  isTRUE(receipt$pmax_within_tolerance) &&
  isTRUE(receipt$n_edgetests_identical)

temporary <- tempfile(".strict-hsic-gamma-oracle-", tmpdir = dirname(output_path))
on.exit(unlink(temporary, force = TRUE), add = TRUE)
saveRDS(receipt, temporary, compress = "xz")
require_true(file.rename(temporary, output_path),
             "failed to publish CPU trace oracle")

cat(
  "strict hsic.gamma trace oracle: tasks=", task_count,
  "; reused_tasks=", reuse_hit_count,
  "; computed_tasks=", missing_task_count,
  "; unique_residuals=", unique_residual_count,
  "; computed_unique_residuals=", residual_count,
  "; residual_sec=", format(residual_elapsed, digits = 8L),
  "; hsic_sec=", format(pvalue_elapsed, digits = 8L),
  "; max_abs_p_diff=", format(max(differences), digits = 17L),
  "; flips=", decision_flip_count,
  "; SHD=", shd,
  "; pass=", receipt$pass, "\n", sep = ""
)
require_true(receipt$pass, "strict hsic.gamma trace oracle gate failed")
