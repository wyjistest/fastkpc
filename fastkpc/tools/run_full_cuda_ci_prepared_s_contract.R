started <- proc.time()[["elapsed"]]
stage <- "load_module"
timings <- list()

raw_env_value <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (nzchar(value)) value else default
}

output_dir <- raw_env_value(
  "FASTKPC_FULL_CUDA_PREPARED_S_OUTPUT_DIR",
  "fastkpc/artifacts/full_cuda_ci/prepared_s_contract_v1"
)

timed <- function(stage_name, expression) {
  stage <<- stage_name
  stage_started <- proc.time()[["elapsed"]]
  value <- force(expression)
  timings[[length(timings) + 1L]] <<- data.frame(
    stage = stage_name,
    elapsed_seconds = as.numeric(
      proc.time()[["elapsed"]] - stage_started
    ),
    stringsAsFactors = FALSE
  )
  value
}

env_path <- function(name, default) {
  value <- raw_env_value(name, default)
  if (!is.character(value) || length(value) != 1L || is.na(value) ||
      !nzchar(value)) {
    stop(name, " must be one nonempty path", call. = FALSE)
  }
  value
}

env_bare_integer <- function(name, default, minimum) {
  text <- raw_env_value(name, as.character(default))
  if (!grepl("^(0|[1-9][0-9]*)$", text)) {
    stop(name, " must be a bare integer", call. = FALSE)
  }
  numeric_value <- suppressWarnings(as.numeric(text))
  if (!is.finite(numeric_value) || numeric_value > .Machine$integer.max ||
      numeric_value < minimum) {
    stop(
      name, " must be a bare integer >= ", minimum,
      call. = FALSE
    )
  }
  as.integer(numeric_value)
}

env_bare_flag <- function(name, default) {
  text <- raw_env_value(name, if (isTRUE(default)) "1" else "0")
  if (!text %in% c("0", "1")) {
    stop(name, " must be the bare boolean 0 or 1", call. = FALSE)
  }
  identical(text, "1")
}

env_enum <- function(name, default, allowed) {
  value <- raw_env_value(name, default)
  if (!value %in% allowed) {
    stop(
      name, " must be one of: ", paste(allowed, collapse = ","),
      call. = FALSE
    )
  }
  value
}

run <- function() {
  stage <<- "load_module"
  source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")

  stage <<- "parse_environment"
  if (length(commandArgs(trailingOnly = TRUE)) > 0L) {
    stop("Prepared-S runner accepts environment variables only",
         call. = FALSE)
  }
  census_dir <- env_path(
    "FASTKPC_FULL_CUDA_PREPARED_S_CENSUS_DIR",
    "fastkpc/artifacts/full_cuda_ci/workload_census_351x48_v1"
  )
  data_path <- env_path(
    "FASTKPC_FULL_CUDA_PREPARED_S_DATA_PATH",
    paste0(
      "fastkpc/artifacts/kpc_tprs_real_zhu/",
      "cancer_RD-causalDiscoveryInput.rds"
    )
  )
  output_dir <<- env_path(
    "FASTKPC_FULL_CUDA_PREPARED_S_OUTPUT_DIR",
    "fastkpc/artifacts/full_cuda_ci/prepared_s_contract_v1"
  )
  max_groups <- env_bare_integer(
    "FASTKPC_FULL_CUDA_PREPARED_S_MAX_GROUPS", 0L, 0L
  )
  parity_scope <- env_enum(
    "FASTKPC_FULL_CUDA_PREPARED_S_PARITY_SCOPE",
    "qualification",
    c("none", "iteration", "qualification")
  )
  requested_workers <- env_bare_integer(
    "FASTKPC_FULL_CUDA_PREPARED_S_WORKERS", 1L, 1L
  )
  shard_count <- env_bare_integer(
    "FASTKPC_FULL_CUDA_PREPARED_S_SHARD_COUNT", 64L, 1L
  )
  resume <- env_bare_flag(
    "FASTKPC_FULL_CUDA_PREPARED_S_RESUME", TRUE
  )
  if (requested_workers > 1L && .Platform$OS.type != "unix") {
    stop(
      "FASTKPC_FULL_CUDA_PREPARED_S_WORKERS > 1 requires Unix",
      call. = FALSE
    )
  }

  command_lines <- c(
    paste0(
      "FASTKPC_FULL_CUDA_PREPARED_S_CENSUS_DIR=", census_dir
    ),
    paste0("FASTKPC_FULL_CUDA_PREPARED_S_DATA_PATH=", data_path),
    paste0(
      "FASTKPC_FULL_CUDA_PREPARED_S_OUTPUT_DIR=", output_dir
    ),
    paste0(
      "FASTKPC_FULL_CUDA_PREPARED_S_MAX_GROUPS=", max_groups
    ),
    paste0(
      "FASTKPC_FULL_CUDA_PREPARED_S_PARITY_SCOPE=", parity_scope
    ),
    paste0(
      "FASTKPC_FULL_CUDA_PREPARED_S_WORKERS=", requested_workers
    ),
    paste0(
      "FASTKPC_FULL_CUDA_PREPARED_S_SHARD_COUNT=", shard_count
    ),
    paste0(
      "FASTKPC_FULL_CUDA_PREPARED_S_RESUME=", as.integer(resume)
    ),
    "Rscript fastkpc/tools/run_full_cuda_ci_prepared_s_contract.R"
  )

  inputs <- timed(
    "load_inputs",
    fastkpc_full_cuda_prepared_s_load_inputs(
      census_dir = census_dir,
      data_path = data_path
    )
  )
  selections <- timed("select_corpora", {
    iteration <- fastkpc_full_cuda_select_prepared_s_iteration_subset(
      inputs
    )
    qualification <-
      fastkpc_full_cuda_select_prepared_s_qualification_subset(inputs)
    selected_setup_index <-
      fastkpc_full_cuda_prepared_s_select_setup_corpus(
        inputs = inputs,
        max_groups = max_groups,
        parity_scope = parity_scope,
        iteration = iteration,
        qualification = qualification
      )
    list(
      iteration = iteration,
      qualification = qualification,
      selected_setup_index = selected_setup_index
    )
  })
  plan <- timed(
    "build_shard_plan",
    .fastkpc_full_cuda_prepared_s_selected_shard_plan(
      inputs = inputs,
      selected_setup_index = selections$selected_setup_index,
      shard_count = shard_count
    )
  )
  artifact_paths <- fastkpc_full_cuda_prepared_s_artifact_paths(
    output_dir
  )
  dir.create(
    artifact_paths$shards_dir,
    recursive = TRUE,
    showWarnings = FALSE
  )
  if (!resume) {
    final_shards <- list.files(
      artifact_paths$shards_dir,
      pattern = "^shard_[0-9]+\\.(rds|summary\\.json)$",
      full.names = TRUE
    )
    if (length(final_shards) > 0L) {
      stop(
        "Prepared-S resume is disabled and a final shard exists",
        call. = FALSE
      )
    }
  }

  actual_workers <- if (requested_workers > 1L) {
    min(requested_workers, shard_count)
  } else {
    1L
  }
  shard_ids <- 0:(shard_count - 1L)
  run_one_shard <- function(shard_id) {
    .fastkpc_full_cuda_run_selected_prepared_s_shard(
      plan = plan,
      shard_id = shard_id,
      output_dir = artifact_paths$shards_dir,
      resume = resume
    )
  }
  shard_results <- timed("run_shards", {
    if (actual_workers > 1L) {
      parallel::mclapply(
        shard_ids,
        run_one_shard,
        mc.cores = actual_workers,
        mc.preschedule = FALSE,
        mc.set.seed = FALSE
      )
    } else {
      lapply(shard_ids, run_one_shard)
    }
  })
  shard_errors <- vapply(shard_results, function(value) {
    inherits(value, "try-error") || inherits(value, "error")
  }, logical(1L))
  if (any(shard_errors)) {
    stop(
      "one or more Prepared-S shards failed: ",
      paste(as.character(shard_results[shard_errors]), collapse = "; "),
      call. = FALSE
    )
  }
  shard_status <- vapply(
    shard_results, `[[`, character(1L), "status"
  )
  written_shard_count <- as.integer(sum(shard_status == "written"))
  reused_shard_count <- as.integer(sum(shard_status == "reused"))
  executed_group_count <- as.integer(sum(vapply(
    shard_results[shard_status == "written"],
    function(value) length(value$payload$ordered_setup_keys),
    integer(1L)
  )))
  reused_group_count <- as.integer(sum(vapply(
    shard_results[shard_status == "reused"],
    function(value) length(value$payload$ordered_setup_keys),
    integer(1L)
  )))

  merged <- timed(
    "merge_shards",
    .fastkpc_full_cuda_merge_selected_prepared_s_shards(
      plan = plan,
      shard_dir = artifact_paths$shards_dir
    )
  )
  merged_group_ids <- vapply(
    merged$prepared_s_setups,
    `[[`,
    character(1L),
    "same_S_group_id"
  )
  if (anyNA(merged_group_ids) || anyDuplicated(merged_group_ids)) {
    stop("merged Prepared-S setup group lineage mismatch",
         call. = FALSE)
  }
  prepared_by_group <- setNames(
    unname(merged$prepared_s_setups), merged_group_ids
  )

  reusable_parity <- timed(
    "validate_reusable_parity",
    fastkpc_full_cuda_prepared_s_reusable_parity_artifact(
      output_dir = output_dir,
      inputs = inputs,
      plan = plan,
      iteration = selections$iteration,
      qualification = selections$qualification,
      merged = merged,
      parity_scope = parity_scope
    )
  )
  parity_evidence_reused <- !is.null(reusable_parity)
  if (parity_evidence_reused) {
    setup_semantic_parity <- reusable_parity$setup_semantic_parity
    target_parity <- reusable_parity$target_parity
    dcov_parity <- reusable_parity$dcov_parity
  } else if (identical(parity_scope, "none")) {
    setup_semantic_parity <-
      fastkpc_full_cuda_prepared_s_empty_setup_semantic_parity()
    target_parity <- list(
      rows = fastkpc_full_cuda_prepared_s_empty_target_parity(),
      residuals = new.env(hash = TRUE, parent = emptyenv())
    )
    dcov_parity <- list(
      rows = fastkpc_full_cuda_prepared_s_empty_dcov_parity()
    )
  } else {
    scope_selection <-
      fastkpc_full_cuda_prepared_s_selection_for_scope(
        inputs, parity_scope
      )
    scope_group_ids <- as.character(
      scope_selection$setup_groups$same_S_group_id
    )
    if (!all(scope_group_ids %in% names(prepared_by_group))) {
      stop(
        "selected Prepared-S corpus is missing parity setup groups",
        call. = FALSE
      )
    }
    scope_prepared <- prepared_by_group[scope_group_ids]
    target_parity <- timed(
      paste0(parity_scope, "_target_parity"),
      .fastkpc_full_cuda_run_prepared_s_target_parity_core(
        inputs = inputs,
        prepared_by_group = scope_prepared,
        target_keys = scope_selection$target_keys,
        scope = parity_scope,
        authenticated_target_states = merged$target_states
      )
    )
    setup_semantic_parity <- timed(
      paste0(parity_scope, "_setup_semantic_parity"),
      fastkpc_full_cuda_prepared_s_setup_semantic_parity(
        inputs = inputs,
        prepared_by_group = scope_prepared,
        target_parity_rows = target_parity$rows,
        scope = parity_scope
      )
    )
    dcov_parity <- timed(
      paste0(parity_scope, "_dcov_parity"),
      fastkpc_full_cuda_run_prepared_s_dcov_parity(
        inputs = inputs,
        logical_tests = scope_selection$logical_tests,
        residuals = target_parity$residuals,
        scope = parity_scope
      )
    )
  }

  environment_lines <- c(
    paste0("R_version=", inputs$manifest$R_version),
    paste0("mgcv_version=", inputs$manifest$mgcv_version),
    paste0("source_commit=", plan$context$source_commit),
    paste0("BLAS_identity=", plan$context$BLAS_identity),
    paste0("LAPACK_identity=", plan$context$LAPACK_identity),
    paste0("BLAS_thread_count=", plan$context$BLAS_thread_count),
    paste0("requested_workers=", requested_workers),
    paste0("actual_workers=", actual_workers),
    paste0("shard_count=", shard_count),
    paste0("resume=", as.integer(resume)),
    paste0("parity_scope=", parity_scope),
    paste0("max_groups=", max_groups),
    "",
    capture.output(sessionInfo())
  )
  stage_timing <- if (length(timings) == 0L) {
    data.frame(
      stage = character(),
      elapsed_seconds = numeric(),
      stringsAsFactors = FALSE
    )
  } else {
    do.call(rbind, timings)
  }
  rownames(stage_timing) <- NULL
  stage <<- "publish_artifact"
  elapsed_before_publication <- proc.time()[["elapsed"]] - started
  artifact <- fastkpc_full_cuda_write_prepared_s_artifact(
    output_dir = output_dir,
    census_dir = census_dir,
    data_path = data_path,
    inputs = inputs,
    plan = plan,
    iteration = selections$iteration,
    qualification = selections$qualification,
    merged = merged,
    setup_semantic_parity = setup_semantic_parity,
    target_parity = target_parity,
    dcov_parity = dcov_parity,
    parity_scope = parity_scope,
    max_groups = max_groups,
    requested_workers = requested_workers,
    actual_workers = actual_workers,
    resume = resume,
    stage_timing = stage_timing,
    elapsed_seconds = elapsed_before_publication,
    command_lines = command_lines,
    environment_lines = environment_lines,
    executed_group_count = executed_group_count,
    reused_group_count = reused_group_count,
    written_shard_count = written_shard_count,
    reused_shard_count = reused_shard_count,
    parity_evidence_reused = parity_evidence_reused
  )
  cat(sprintf(
    paste0(
      "PASS full CUDA CI Prepared-S contract scope=%s groups=%d ",
      "targets=%d written_shards=%d reused_shards=%d elapsed=%.3f sec\n"
    ),
    artifact$summary$run_scope,
    artifact$summary$selected_group_count,
    artifact$summary$target_state_count,
    artifact$summary$written_shard_count,
    artifact$summary$reused_shard_count,
    artifact$summary$elapsed_seconds
  ))
  invisible(artifact)
}

error <- tryCatch({
  run()
  NULL
}, error = identity)

if (!is.null(error)) {
  elapsed <- proc.time()[["elapsed"]] - started
  published <- tryCatch({
    if (exists(
          "fastkpc_full_cuda_prepared_s_write_failure_summary",
          mode = "function",
          inherits = TRUE
        )) {
      fastkpc_full_cuda_prepared_s_write_failure_summary(
        output_dir = output_dir,
        stage = stage,
        error = error,
        elapsed_seconds = elapsed
      )
    } else {
      dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
      if (!requireNamespace("jsonlite", quietly = TRUE)) {
        stop("jsonlite is unavailable for failure publication",
             call. = FALSE)
      }
      path <- file.path(output_dir, "summary.json")
      jsonlite::write_json(
        list(
          pass = FALSE,
          phase2_complete = FALSE,
          stage = stage,
          error_class = as.character(class(error)[[1L]]),
          error_message = conditionMessage(error),
          elapsed_seconds = as.numeric(elapsed)
        ),
        path,
        auto_unbox = TRUE,
        pretty = TRUE,
        null = "null",
        na = "null",
        digits = NA
      )
      path
    }
  }, error = identity)
  if (inherits(published, "error")) {
    message(
      conditionMessage(error),
      " (failure summary publication also failed: ",
      conditionMessage(published),
      ")"
    )
  } else {
    message(conditionMessage(error))
  }
  quit(save = "no", status = 1L)
}
