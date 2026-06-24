source("fastkpc/R/fast_kpc.R")

`%||%` <- function(x, y) if (is.null(x)) y else x

fastkpc_kpc_tprs_qualification_caps <- function(enable_kpc = FALSE) {
  caps <- list(
    R_version = "4.5.0",
    mgcv_version = "1.9-4",
    cuda_available = FALSE,
    mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
    spectral_gcv_version = "single-penalty-spectral-gcv-v1",
    setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
  )
  if (isTRUE(enable_kpc)) {
    caps$kpcTprsResidualCPP_supported <- TRUE
    caps$kpcTprsResidualCPP_backend_version <- "kpcTprsResidualCPP-v1"
  }
  caps
}

fastkpc_kpc_tprs_qualification_default_scenarios <- function() {
  make_scenario <- function(seed, n, scale = 1, duplicate = FALSE,
                            near_collinear = FALSE) {
    set.seed(seed)
    s <- stats::runif(n, -2, 2) * scale
    z <- s + stats::rnorm(n, sd = 0.02 * scale)
    if (isTRUE(near_collinear)) {
      q <- z + stats::rnorm(n, sd = 1e-3 * max(1, scale))
    } else {
      q <- 0.5 * sin(s / max(1, scale)) -
        0.25 * cos(s / max(1, scale)) + stats::rnorm(n, sd = 0.08)
    }
    data <- cbind(
      x = sin(s / max(1, scale)) + stats::rnorm(n, sd = 0.04),
      y = cos(s / max(1, scale)) + stats::rnorm(n, sd = 0.04),
      z = z,
      q = q
    )
    if (isTRUE(duplicate)) {
      rows <- seq_len(min(8L, nrow(data)))
      data <- rbind(data, data[rows, , drop = FALSE])
    }
    data
  }
  list(
    list(
      scenario_id = "seed-1-n72",
      n = 72L,
      seed = 62330L,
      generator = function(n, seed) {
        make_scenario(seed, n)
      }
    ),
    list(
      scenario_id = "seed-2-scaled",
      n = 80L,
      seed = 62331L,
      generator = function(n, seed) {
        make_scenario(seed, n, scale = 25)
      }
    ),
    list(
      scenario_id = "duplicates",
      n = 64L,
      seed = 62332L,
      generator = function(n, seed) {
        make_scenario(seed, n, duplicate = TRUE)
      }
    ),
    list(
      scenario_id = "near-collinear",
      n = 76L,
      seed = 62333L,
      generator = function(n, seed) {
        make_scenario(seed, n, near_collinear = TRUE)
      }
    ),
    list(
      scenario_id = "small-n",
      n = 42L,
      seed = 62334L,
      generator = function(n, seed) {
        make_scenario(seed, n)
      }
    )
  )
}

fastkpc_kpc_tprs_qualification_real_scenario <- function(path) {
  if (is.null(path) || !nzchar(path)) return(NULL)
  if (!file.exists(path)) {
    stop("real data path does not exist: ", path, call. = FALSE)
  }
  ext <- tolower(tools::file_ext(path))
  raw <- if (identical(ext, "rds")) {
    readRDS(path)
  } else {
    utils::read.csv(path, check.names = FALSE)
  }
  data <- as.data.frame(raw)
  numeric <- vapply(data, is.numeric, logical(1L))
  data <- data[, numeric, drop = FALSE]
  if (ncol(data) < 3L) {
    stop("real data must contain at least three numeric columns",
         call. = FALSE)
  }
  matrix_data <- as.matrix(data)
  keep <- stats::complete.cases(matrix_data) &
    apply(matrix_data, 1L, function(row) all(is.finite(row)))
  matrix_data <- matrix_data[keep, , drop = FALSE]
  if (nrow(matrix_data) < 20L) {
    stop("real data has fewer than 20 finite complete rows", call. = FALSE)
  }
  list(
    scenario_id = paste0("real-", tools::file_path_sans_ext(basename(path))),
    n = nrow(matrix_data),
    seed = 0L,
    generator = function(n, seed) {
      matrix_data[seq_len(min(n, nrow(matrix_data))), , drop = FALSE]
    }
  )
}

fastkpc_kpc_tprs_qualification_scenarios <- function(real_data_path = "") {
  scenarios <- fastkpc_kpc_tprs_qualification_default_scenarios()
  real <- fastkpc_kpc_tprs_qualification_real_scenario(real_data_path)
  if (is.null(real)) scenarios else c(scenarios, list(real))
}

fastkpc_kpc_tprs_count_sepset_mismatch <- function(left, right) {
  p <- length(left)
  if (p == 0L) return(0)
  mismatches <- 0L
  total <- 0L
  for (i in seq_len(p)) {
    for (j in seq_len(p)) {
      if (i == j) next
      total <- total + 1L
      li <- sort(as.integer(left[[i]][[j]]))
      ri <- sort(as.integer(right[[i]][[j]]))
      if (!identical(li, ri)) mismatches <- mismatches + 1L
    }
  }
  mismatches / max(1L, total)
}

fastkpc_kpc_tprs_restore_repeat_name <- function(data) {
  if (is.data.frame(data)) {
    names(data)[names(data) == "repeat."] <- "repeat"
  }
  data
}

fastkpc_kpc_tprs_trace_summary_row <- function(result, scenario_id, repeat_id,
                                               mode) {
  trace <- result$diagnostics$precision_trace
  if (!is.data.frame(trace)) trace <- data.frame()
  conditional <- if (nrow(trace) == 0L) {
    trace
  } else {
    trace[nzchar(trace$S_key), , drop = FALSE]
  }
  two_d <- if (nrow(conditional) == 0L) {
    conditional
  } else {
    conditional[grepl("|", conditional$S_key, fixed = TRUE), , drop = FALSE]
  }
  data.frame(
    scenario_id = scenario_id,
    `repeat` = as.integer(repeat_id),
    mode = mode,
    total_rows = as.integer(nrow(trace)),
    conditional_rows = as.integer(nrow(conditional)),
    kpc_rows = as.integer(sum(
      conditional$backend_executed == "kpcTprsResidualCPP", na.rm = TRUE)),
    two_d_kpc_rows = as.integer(sum(
      two_d$backend_executed == "kpcTprsResidualCPP", na.rm = TRUE)),
    mgcv_rows = as.integer(sum(
      conditional$backend_executed %in% c("mgcvExtractCPU", "mgcvExtractGPU"),
      na.rm = TRUE)),
    fallback_rows = as.integer(sum(
      conditional$fallback_triggered %in% TRUE, na.rm = TRUE)),
    legacy_rows = as.integer(sum(
      conditional$backend_executed == "legacy-mgcv", na.rm = TRUE)),
    stringsAsFactors = FALSE
  )
}

fastkpc_kpc_tprs_run_mode <- function(data, scenario_id, repeat_id, mode,
                                      alpha, max_conditioning_size) {
  caps <- fastkpc_kpc_tprs_qualification_caps(
    enable_kpc = identical(mode, "candidate_kpc"))
  start <- proc.time()[["elapsed"]]
  safe <- tryCatch({
    value <- fast_kpc(
      data,
      alpha = alpha,
      max_conditioning_size = max_conditioning_size,
      engine = "cpu",
      precision = "compatible",
      graph_stage = "skeleton",
      runtime_capabilities = caps
    )
    list(status = "ok", value = value, error_message = "")
  }, error = function(e) {
    list(status = "error", value = NULL, error_message = conditionMessage(e))
  })
  elapsed <- max(proc.time()[["elapsed"]] - start, 0)
  result <- safe$value
  trace <- if (!is.null(result)) result$diagnostics$precision_trace else data.frame()
  conditional <- if (is.data.frame(trace) && nrow(trace) > 0L) {
    trace[nzchar(trace$S_key), , drop = FALSE]
  } else {
    data.frame()
  }
  data.frame(
    scenario_id = scenario_id,
    `repeat` = as.integer(repeat_id),
    mode = mode,
    status = safe$status,
    wall_time_sec = as.numeric(elapsed),
    n = as.integer(nrow(data)),
    p = as.integer(ncol(data)),
    backend_planned = result$config$backend_planned %||% NA_character_,
    backend_executed = result$config$backend_executed %||% NA_character_,
    conditional_tests = as.integer(nrow(conditional)),
    error_message = safe$error_message,
    stringsAsFactors = FALSE
  )
}

fastkpc_kpc_tprs_graph_agreement_row <- function(reference, candidate,
                                                 scenario_id, repeat_id) {
  if (is.null(reference) || is.null(candidate)) {
    return(data.frame(
      scenario_id = scenario_id,
      `repeat` = as.integer(repeat_id),
      adjacency_identical = FALSE,
      n_edgetests_identical = FALSE,
      pmax_max_abs_diff = NA_real_,
      first_sepset_mismatch_rate = NA_real_,
      all_sepset_mismatch_rate = NA_real_,
      passed = FALSE,
      stringsAsFactors = FALSE
    ))
  }
  pmax_diff <- abs(candidate$skeleton$pMax - reference$skeleton$pMax)
  pmax_diff <- pmax_diff[upper.tri(pmax_diff)]
  sepset_mismatch <- fastkpc_kpc_tprs_count_sepset_mismatch(
    reference$skeleton$sepsets, candidate$skeleton$sepsets)
  pmax_max <- if (length(pmax_diff) == 0L) 0 else max(pmax_diff, na.rm = TRUE)
  adjacency_identical <- identical(candidate$skeleton$adjacency,
                                   reference$skeleton$adjacency)
  n_edgetests_identical <- identical(candidate$skeleton$n.edgetests,
                                     reference$skeleton$n.edgetests)
  data.frame(
    scenario_id = scenario_id,
    `repeat` = as.integer(repeat_id),
    adjacency_identical = adjacency_identical,
    n_edgetests_identical = n_edgetests_identical,
    pmax_max_abs_diff = pmax_max,
    first_sepset_mismatch_rate = sepset_mismatch,
    all_sepset_mismatch_rate = sepset_mismatch,
    passed = isTRUE(adjacency_identical) &&
      isTRUE(n_edgetests_identical) &&
      is.finite(pmax_max) && pmax_max < 1e-4 &&
      is.finite(sepset_mismatch) && sepset_mismatch == 0,
    stringsAsFactors = FALSE
  )
}

fastkpc_kpc_tprs_with_mgcv_forbidden <- function(expr) {
  if (!requireNamespace("mgcv", quietly = TRUE)) {
    return(list(value = NULL, forbidden_calls = NA_integer_,
                error = "mgcv unavailable"))
  }
  ns <- asNamespace("mgcv")
  calls <- new.env(parent = emptyenv())
  calls$count <- 0L
  blocker <- function(...) {
    calls$count <- calls$count + 1L
    stop("forbidden mgcv oracle call", call. = FALSE)
  }
  traced <- character()
  for (name in c("gam", "smoothCon", "magic")) {
    if (exists(name, envir = ns, inherits = FALSE)) {
      trace(name, tracer = blocker, print = FALSE, where = ns)
      traced <- c(traced, name)
    }
  }
  on.exit({
    for (name in rev(traced)) {
      try(untrace(name, where = ns), silent = TRUE)
    }
  }, add = TRUE)
  value <- tryCatch(force(expr), error = function(e) e)
  list(value = value, forbidden_calls = calls$count, error = "")
}

fastkpc_kpc_tprs_no_oracle_row <- function(data, alpha,
                                           max_conditioning_size) {
  guard <- fastkpc_kpc_tprs_with_mgcv_forbidden({
    fast_kpc(
      data,
      alpha = alpha,
      max_conditioning_size = max_conditioning_size,
      engine = "cpu",
      precision = "compatible",
      graph_stage = "skeleton",
      runtime_capabilities =
        fastkpc_kpc_tprs_qualification_caps(enable_kpc = TRUE)
    )
  })
  ok_result <- !inherits(guard$value, "error")
  trace <- if (ok_result) guard$value$diagnostics$precision_trace else data.frame()
  conditional <- if (is.data.frame(trace) && nrow(trace) > 0L) {
    trace[nzchar(trace$S_key), , drop = FALSE]
  } else {
    data.frame()
  }
  kpc_rows <- if (nrow(conditional) == 0L) {
    0L
  } else {
    sum(conditional$backend_executed == "kpcTprsResidualCPP", na.rm = TRUE)
  }
  failure <- if (ok_result) "" else conditionMessage(guard$value)
  data.frame(
    passed = ok_result && identical(guard$forbidden_calls, 0L) && kpc_rows > 0L,
    forbidden_calls = as.integer(guard$forbidden_calls),
    conditional_kpc_rows = as.integer(kpc_rows),
    failure_reason = failure,
    stringsAsFactors = FALSE
  )
}

fastkpc_run_kpc_tprs_residual_cpp_qualification <- function(
    output_dir = file.path("fastkpc", "artifacts",
                           "kpc_tprs_residual_cpp_qualification"),
    scenarios = NULL,
    repeats = 3L,
    alpha = 0.05,
    max_conditioning_size = 2L,
    real_data_path = Sys.getenv("FASTKPC_KPC_TPRS_REAL_DATA", ""),
    no_oracle_check = TRUE) {
  if (is.null(scenarios)) {
    scenarios <- fastkpc_kpc_tprs_qualification_scenarios(real_data_path)
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  run_rows <- list()
  graph_rows <- list()
  trace_rows <- list()
  details <- list()

  for (scenario in scenarios) {
    scenario_id <- scenario$scenario_id
    n <- as.integer(scenario$n %||% 64L)
    seed <- as.integer(scenario$seed %||% 62410L)
    for (repeat_id in seq_len(as.integer(repeats))) {
      data <- as.matrix(scenario$generator(n, seed + repeat_id - 1L))
      storage.mode(data) <- "double"
      if (is.null(colnames(data))) {
        colnames(data) <- paste0("V", seq_len(ncol(data)))
      }
      ref_row <- fastkpc_kpc_tprs_run_mode(
        data, scenario_id, repeat_id, "reference_mgcv",
        alpha, max_conditioning_size)
      reference <- if (identical(ref_row$status, "ok")) {
        fast_kpc(
          data,
          alpha = alpha,
          max_conditioning_size = max_conditioning_size,
          engine = "cpu",
          precision = "compatible",
          graph_stage = "skeleton",
          runtime_capabilities =
            fastkpc_kpc_tprs_qualification_caps(enable_kpc = FALSE)
        )
      } else {
        NULL
      }
      cand_start <- proc.time()[["elapsed"]]
      candidate_safe <- tryCatch({
        value <- fast_kpc(
          data,
          alpha = alpha,
          max_conditioning_size = max_conditioning_size,
          engine = "cpu",
          precision = "compatible",
          graph_stage = "skeleton",
          runtime_capabilities =
            fastkpc_kpc_tprs_qualification_caps(enable_kpc = TRUE)
        )
        list(status = "ok", value = value, error_message = "")
      }, error = function(e) {
        list(status = "error", value = NULL, error_message = conditionMessage(e))
      })
      cand_elapsed <- max(proc.time()[["elapsed"]] - cand_start, 0)
      candidate <- candidate_safe$value
      cand_trace <- if (!is.null(candidate)) {
        candidate$diagnostics$precision_trace
      } else {
        data.frame()
      }
      cand_conditional <- if (is.data.frame(cand_trace) && nrow(cand_trace) > 0L) {
        cand_trace[nzchar(cand_trace$S_key), , drop = FALSE]
      } else {
        data.frame()
      }
      cand_row <- data.frame(
        scenario_id = scenario_id,
        `repeat` = as.integer(repeat_id),
        mode = "candidate_kpc",
        status = candidate_safe$status,
        wall_time_sec = as.numeric(cand_elapsed),
        n = as.integer(nrow(data)),
        p = as.integer(ncol(data)),
        backend_planned = candidate$config$backend_planned %||% NA_character_,
        backend_executed = candidate$config$backend_executed %||% NA_character_,
        conditional_tests = as.integer(nrow(cand_conditional)),
        error_message = candidate_safe$error_message,
        stringsAsFactors = FALSE
      )
      run_rows[[length(run_rows) + 1L]] <- ref_row
      run_rows[[length(run_rows) + 1L]] <- cand_row
      if (!is.null(reference)) {
        trace_rows[[length(trace_rows) + 1L]] <-
          fastkpc_kpc_tprs_trace_summary_row(
            reference, scenario_id, repeat_id, "reference_mgcv")
      }
      if (!is.null(candidate)) {
        trace_rows[[length(trace_rows) + 1L]] <-
          fastkpc_kpc_tprs_trace_summary_row(
            candidate, scenario_id, repeat_id, "candidate_kpc")
      }
      graph_rows[[length(graph_rows) + 1L]] <-
        fastkpc_kpc_tprs_graph_agreement_row(
          reference, candidate, scenario_id, repeat_id)
      details[[paste(scenario_id, repeat_id, sep = "::")]] <-
        list(reference = reference, candidate = candidate)
    }
  }

  runs <- if (length(run_rows) == 0L) data.frame() else do.call(rbind, run_rows)
  runs <- fastkpc_kpc_tprs_restore_repeat_name(runs)
  graph_agreement <- if (length(graph_rows) == 0L) {
    data.frame()
  } else {
    do.call(rbind, graph_rows)
  }
  graph_agreement <- fastkpc_kpc_tprs_restore_repeat_name(graph_agreement)
  trace_summary <- if (length(trace_rows) == 0L) {
    data.frame()
  } else {
    do.call(rbind, trace_rows)
  }
  trace_summary <- fastkpc_kpc_tprs_restore_repeat_name(trace_summary)
  no_oracle <- if (isTRUE(no_oracle_check) && length(scenarios) > 0L) {
    scenario <- scenarios[[1L]]
    data <- as.matrix(scenario$generator(as.integer(scenario$n), scenario$seed))
    storage.mode(data) <- "double"
    fastkpc_kpc_tprs_no_oracle_row(data, alpha, max_conditioning_size)
  } else {
    data.frame(
      passed = NA,
      forbidden_calls = NA_integer_,
      conditional_kpc_rows = NA_integer_,
      failure_reason = "not-run",
      stringsAsFactors = FALSE
    )
  }
  kpc_rows <- if (nrow(trace_summary) == 0L) {
    0L
  } else {
    sum(trace_summary$kpc_rows, na.rm = TRUE)
  }
  two_d_rows <- if (nrow(trace_summary) == 0L) {
    0L
  } else {
    sum(trace_summary$two_d_kpc_rows, na.rm = TRUE)
  }
  fallback_rows <- if (nrow(trace_summary) == 0L) {
    0L
  } else {
    sum(trace_summary$fallback_rows, na.rm = TRUE)
  }
  graph_passed <- nrow(graph_agreement) > 0L &&
    all(graph_agreement$passed %in% TRUE)
  no_oracle_passed <- !isTRUE(no_oracle_check) || isTRUE(no_oracle$passed[[1L]])
  passed <- isTRUE(graph_passed) && kpc_rows > 0L && two_d_rows > 0L &&
    fallback_rows == 0L && isTRUE(no_oracle_passed)
  failure_reason <- if (passed) {
    ""
  } else {
    paste(c(
      if (!isTRUE(graph_passed)) "graph-agreement-failed" else NULL,
      if (kpc_rows <= 0L) "no-kpc-rows" else NULL,
      if (two_d_rows <= 0L) "no-2d-kpc-rows" else NULL,
      if (fallback_rows > 0L) "fallback-used" else NULL,
      if (!isTRUE(no_oracle_passed)) "no-oracle-failed" else NULL
    ), collapse = ";")
  }
  qualification_summary <- data.frame(
    scenarios = as.integer(length(scenarios)),
    repeats = as.integer(repeats),
    run_rows = as.integer(nrow(runs)),
    graph_rows = as.integer(nrow(graph_agreement)),
    kpc_rows = as.integer(kpc_rows),
    two_d_kpc_rows = as.integer(two_d_rows),
    fallback_rows = as.integer(fallback_rows),
    max_pmax_abs_diff = if (nrow(graph_agreement) == 0L) {
      NA_real_
    } else {
      max(graph_agreement$pmax_max_abs_diff, na.rm = TRUE)
    },
    passed = passed,
    failure_reason = failure_reason,
    stringsAsFactors = FALSE
  )
  summary <- list(
    passed = isTRUE(passed),
    scenarios = as.integer(length(scenarios)),
    repeats = as.integer(repeats),
    output_dir = output_dir,
    recommendation = if (isTRUE(passed)) {
      "kpcTprsResidualCPP remains qualified as experimental opt-in for |S|<=2"
    } else {
      "keep kpcTprsResidualCPP experimental and inspect qualification artifacts"
    }
  )
  paths <- list(
    runs = file.path(output_dir, "runs.csv"),
    graph_agreement = file.path(output_dir, "graph_agreement.csv"),
    trace_summary = file.path(output_dir, "trace_summary.csv"),
    qualification_summary = file.path(output_dir, "qualification_summary.csv"),
    no_oracle = file.path(output_dir, "no_oracle.csv"),
    summary_md = file.path(output_dir, "summary.md")
  )
  utils::write.csv(runs, paths$runs, row.names = FALSE)
  utils::write.csv(graph_agreement, paths$graph_agreement, row.names = FALSE)
  utils::write.csv(trace_summary, paths$trace_summary, row.names = FALSE)
  utils::write.csv(qualification_summary, paths$qualification_summary,
                   row.names = FALSE)
  utils::write.csv(no_oracle, paths$no_oracle, row.names = FALSE)
  writeLines(c(
    "# kpcTprsResidualCPP Qualification",
    "",
    paste0("- passed: ", passed),
    paste0("- scenarios: ", length(scenarios)),
    paste0("- repeats: ", as.integer(repeats)),
    paste0("- kpc rows: ", kpc_rows),
    paste0("- |S|=2 kpc rows: ", two_d_rows),
    paste0("- fallback rows: ", fallback_rows),
    paste0("- recommendation: ", summary$recommendation)
  ), paths$summary_md)
  list(
    runs = runs,
    graph_agreement = graph_agreement,
    trace_summary = trace_summary,
    qualification_summary = qualification_summary,
    no_oracle = no_oracle,
    summary = summary,
    paths = paths,
    output_dir = output_dir,
    details = details
  )
}
