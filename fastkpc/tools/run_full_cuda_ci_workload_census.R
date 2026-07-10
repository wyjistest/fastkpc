source("fastkpc/R/full_cuda_ci_workload_census.R")

env_value <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (nzchar(value)) value else default
}

env_integer <- function(name, default, minimum = 0L) {
  value <- suppressWarnings(as.integer(env_value(name, as.character(default))))
  if (length(value) != 1L || is.na(value) || value < minimum) {
    stop(name, " must be an integer >= ", minimum, call. = FALSE)
  }
  value
}

env_flag <- function(name, default = TRUE) {
  value <- tolower(env_value(name, if (default) "1" else "0"))
  if (!value %in% c("0", "1", "false", "true", "no", "yes")) {
    stop(name, " must be a boolean flag", call. = FALSE)
  }
  value %in% c("1", "true", "yes")
}

mode <- tolower(env_value("FASTKPC_FULL_CUDA_CENSUS_MODE", "metadata"))
if (!mode %in% c("structural", "parity", "metadata")) {
  stop("FASTKPC_FULL_CUDA_CENSUS_MODE must be structural, parity, or metadata",
       call. = FALSE)
}
oracle_dir <- env_value(
  "FASTKPC_FULL_CUDA_CENSUS_ORACLE_DIR",
  "fastkpc/artifacts/full_cuda_ci/oracle_351x48_v1"
)
data_path <- env_value(
  "FASTKPC_FULL_CUDA_CENSUS_DATA_PATH",
  paste0("fastkpc/artifacts/kpc_tprs_real_zhu/",
         "cancer_RD-causalDiscoveryInput.rds")
)
output_dir <- env_value(
  "FASTKPC_FULL_CUDA_CENSUS_OUTPUT_DIR",
  "fastkpc/artifacts/full_cuda_ci/workload_census_351x48_v1"
)
max_keys <- env_integer("FASTKPC_FULL_CUDA_CENSUS_MAX_KEYS", 0L, 0L)
requested_workers <- env_integer(
  "FASTKPC_FULL_CUDA_CENSUS_WORKERS", 1L, 1L
)
shard_count <- env_integer(
  "FASTKPC_FULL_CUDA_CENSUS_SHARD_COUNT", 64L, 1L
)
resume <- env_flag("FASTKPC_FULL_CUDA_CENSUS_RESUME", TRUE)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
paths <- fastkpc_full_cuda_census_artifact_paths(output_dir)
started <- proc.time()[["elapsed"]]
timings <- list()
timed <- function(stage, expression) {
  stage_started <- proc.time()[["elapsed"]]
  value <- force(expression)
  timings[[length(timings) + 1L]] <<- data.frame(
    stage = stage,
    elapsed_sec = proc.time()[["elapsed"]] - stage_started,
    stringsAsFactors = FALSE
  )
  value
}

gate_error <- function(message) {
  structure(
    list(message = message, call = NULL),
    class = c("fastkpc_full_cuda_census_gate_error", "error", "condition")
  )
}

command_lines <- c(
  paste0("FASTKPC_FULL_CUDA_CENSUS_ORACLE_DIR=", oracle_dir),
  paste0("FASTKPC_FULL_CUDA_CENSUS_DATA_PATH=", data_path),
  paste0("FASTKPC_FULL_CUDA_CENSUS_OUTPUT_DIR=", output_dir),
  paste0("FASTKPC_FULL_CUDA_CENSUS_MODE=", mode),
  paste0("FASTKPC_FULL_CUDA_CENSUS_MAX_KEYS=", max_keys),
  paste0("FASTKPC_FULL_CUDA_CENSUS_WORKERS=", requested_workers),
  paste0("FASTKPC_FULL_CUDA_CENSUS_SHARD_COUNT=", shard_count),
  paste0("FASTKPC_FULL_CUDA_CENSUS_RESUME=", as.integer(resume)),
  "Rscript fastkpc/tools/run_full_cuda_ci_workload_census.R"
)

run <- function() {
  inputs <- timed("load_inputs", fastkpc_full_cuda_census_load_inputs(
    oracle_dir = oracle_dir,
    data_path = data_path
  ))
  structural <- timed("structural_census", {
    value <- fastkpc_full_cuda_census_structural(inputs)
    fastkpc_full_cuda_census_validate_structural(value, canonical = TRUE)
    value
  })
  canonical_requests <- structural$residual_requests
  if (max_keys > nrow(canonical_requests)) {
    stop("FASTKPC_FULL_CUDA_CENSUS_MAX_KEYS exceeds canonical key count",
         call. = FALSE)
  }
  selected_count <- if (max_keys == 0L) nrow(canonical_requests) else max_keys
  selected_requests <- canonical_requests[seq_len(selected_count),
                                          , drop = FALSE]
  rownames(selected_requests) <- NULL

  parity_cases <- NULL
  parity_results <- NULL
  if (mode %in% c("parity", "metadata")) {
    parity_cases <- timed(
      "parity_case_selection",
      fastkpc_full_cuda_census_parity_cases(inputs)
    )
    parity_results <- timed("legacy_layout_parity", do.call(rbind, lapply(
      parity_cases, fastkpc_full_cuda_census_parity_case
    )))
    if (!all(parity_results$pass)) {
      stop("legacy layout parity failed", call. = FALSE)
    }
  }

  merged <- NULL
  actual_workers <- 1L
  executed_key_count <- 0L
  written_shard_count <- 0L
  reused_shard_count <- 0L
  context <- fastkpc_full_cuda_census_build_shard_context(
    inputs, structural, selected_requests
  )
  if (mode == "metadata") {
    assigned <- fastkpc_full_cuda_census_assign_shards(
      selected_requests, shard_count
    )
    if (!resume && dir.exists(paths$shards_dir)) {
      unlink(paths$shards_dir, recursive = TRUE, force = TRUE)
    }
    dir.create(paths$shards_dir, recursive = TRUE, showWarnings = FALSE)
    actual_workers <- if (.Platform$OS.type == "unix" &&
                          requested_workers > 1L) {
      min(requested_workers, shard_count, max(1L, selected_count))
    } else {
      1L
    }
    shard_ids <- 0:(shard_count - 1L)
    run_one <- function(shard_id) {
      fastkpc_full_cuda_census_run_shard(
        assigned_requests = assigned,
        shard_id = shard_id,
        context = context,
        output_dir = paths$shards_dir
      )
    }
    shard_results <- timed("metadata_shards", {
      if (actual_workers > 1L) {
        parallel::mclapply(
          shard_ids, run_one, mc.cores = actual_workers,
          mc.preschedule = FALSE
        )
      } else {
        lapply(shard_ids, run_one)
      }
    })
    shard_errors <- vapply(shard_results, inherits, logical(1L),
                           what = "try-error")
    if (any(shard_errors)) {
      stop("one or more metadata shards failed: ",
           paste(as.character(shard_results[shard_errors]), collapse = "; "),
           call. = FALSE)
    }
    shard_status <- vapply(shard_results, `[[`, character(1L), "status")
    written_shard_count <- sum(shard_status == "written")
    reused_shard_count <- sum(shard_status == "reused")
    executed_key_count <- sum(vapply(
      shard_results[shard_status == "written"],
      function(value) length(value$payload$request_keys), integer(1L)
    ))
    merged <- timed("merge_shards", fastkpc_full_cuda_census_merge_shards(
      requests = selected_requests,
      shard_count = shard_count,
      context = context,
      shard_dir = paths$shards_dir
    ))
  }

  stage_timing <- if (length(timings) == 0L) data.frame() else
    do.call(rbind, timings)
  elapsed_sec <- proc.time()[["elapsed"]] - started
  artifact <- timed("write_artifact", {
    fastkpc_full_cuda_census_write_artifact(
      output_dir = output_dir,
      oracle_dir = oracle_dir,
      data_path = data_path,
      inputs = inputs,
      structural = structural,
      selected_requests = selected_requests,
      context = context,
      mode = mode,
      requested_workers = requested_workers,
      actual_workers = actual_workers,
      shard_count = shard_count,
      resume = resume,
      parity_cases = parity_cases,
      parity_results = parity_results,
      merged = merged,
      stage_timing = stage_timing,
      elapsed_sec = elapsed_sec,
      command_lines = command_lines,
      executed_key_count = executed_key_count,
      written_shard_count = written_shard_count,
      reused_shard_count = reused_shard_count
    )
  })
  if (!isTRUE(artifact$summary$pass)) {
    stop(gate_error("workload census artifact failed internal gates"))
  }
  cat(sprintf(
    "PASS full CUDA CI workload census mode=%s scope=%s keys=%d elapsed=%.3f sec\n",
    mode, artifact$summary$run_scope,
    artifact$summary$selected_key_count, artifact$summary$elapsed_sec
  ))
  invisible(artifact)
}

error <- tryCatch({
  run()
  NULL
}, error = identity)

if (!is.null(error)) {
  failure <- list(
    pass = FALSE,
    mode = mode,
    phase1_complete = FALSE,
    error = conditionMessage(error),
    elapsed_sec = proc.time()[["elapsed"]] - started
  )
  preserve_gate_artifact <- inherits(
    error, "fastkpc_full_cuda_census_gate_error"
  ) && file.exists(paths$summary_json)
  if (!preserve_gate_artifact) {
    try(fastkpc_full_cuda_write_json(failure, paths$summary_json), silent = TRUE)
    try(writeLines(c(
      "# Full CUDA CI Workload Census",
      "",
      "- pass: FALSE",
      paste0("- error: ", conditionMessage(error))
    ), paths$summary_md), silent = TRUE)
  }
  message(conditionMessage(error))
  quit(save = "no", status = 1L)
}
