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

fastkpc_kpc_tprs_cuda_available <- function() {
  exists("fastkpc_cuda_available", mode = "function") &&
    isTRUE(tryCatch(fastkpc_cuda_available(), error = function(e) FALSE))
}

fastkpc_kpc_tprs_mode_spec <- function(mode) {
  mode <- match.arg(
    mode,
    c("reference_mgcv", "candidate_kpc", "legacy_mgcv",
      "mgcv_extract_gpu")
  )
  caps <- fastkpc_kpc_tprs_qualification_caps(
    enable_kpc = identical(mode, "candidate_kpc"))
  executors <- NULL
  engine <- "cpu"
  skip_reason <- ""
  if (identical(mode, "legacy_mgcv")) {
    executors <- fastkpc_default_precision_executors()
    executors$mgcvExtractCPUGCVBridge <- fastkpc_execute_ci_legacy_mgcv
  } else if (identical(mode, "mgcv_extract_gpu")) {
    engine <- "cuda"
    caps$cuda_available <- TRUE
    if (!fastkpc_kpc_tprs_cuda_available()) {
      skip_reason <- "CUDA runtime unavailable"
    }
  }
  list(
    engine = engine,
    precision = "compatible",
    runtime_capabilities = caps,
    precision_executors = executors,
    skip_reason = skip_reason
  )
}

fastkpc_kpc_tprs_run_mode_result <- function(data, scenario_id, repeat_id, mode,
                                             alpha, max_conditioning_size) {
  spec <- fastkpc_kpc_tprs_mode_spec(mode)
  if (nzchar(spec$skip_reason)) {
    row <- data.frame(
      scenario_id = scenario_id,
      `repeat` = as.integer(repeat_id),
      mode = mode,
      status = "skipped",
      wall_time_sec = NA_real_,
      n = as.integer(nrow(data)),
      p = as.integer(ncol(data)),
      backend_planned = NA_character_,
      backend_executed = NA_character_,
      conditional_tests = NA_integer_,
      error_message = spec$skip_reason,
      stringsAsFactors = FALSE
    )
    return(list(row = row, result = NULL))
  }
  start <- proc.time()[["elapsed"]]
  safe <- tryCatch({
    value <- fast_kpc(
      data,
      alpha = alpha,
      max_conditioning_size = max_conditioning_size,
      engine = spec$engine,
      precision = spec$precision,
      graph_stage = "skeleton",
      runtime_capabilities = spec$runtime_capabilities,
      precision_executors = spec$precision_executors
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
  row <- data.frame(
    scenario_id = scenario_id,
    `repeat` = as.integer(repeat_id),
    mode = mode,
    status = safe$status,
    wall_time_sec = as.numeric(elapsed),
    n = as.integer(nrow(data)),
    p = as.integer(ncol(data)),
    backend_planned = if (is.null(result)) NA_character_ else
      result$config$backend_planned %||% NA_character_,
    backend_executed = if (is.null(result)) NA_character_ else
      result$config$backend_executed %||% NA_character_,
    conditional_tests = as.integer(nrow(conditional)),
    error_message = safe$error_message,
    stringsAsFactors = FALSE
  )
  list(row = row, result = result)
}

fastkpc_kpc_tprs_run_mode <- function(data, scenario_id, repeat_id, mode,
                                      alpha, max_conditioning_size) {
  fastkpc_kpc_tprs_run_mode_result(
    data, scenario_id, repeat_id, mode, alpha, max_conditioning_size
  )$row
}

fastkpc_kpc_tprs_backend_comparison_row <- function(
    reference, candidate, reference_run, candidate_run,
    scenario_id, repeat_id, reference_mode, candidate_mode) {
  graph <- fastkpc_kpc_tprs_graph_agreement_row(
    reference, candidate, scenario_id, repeat_id)
  reference_status <- if (is.null(reference_run)) "missing" else
    as.character(reference_run$status[[1L]])
  candidate_status <- if (is.null(candidate_run)) "missing" else
    as.character(candidate_run$status[[1L]])
  reference_time <- if (is.null(reference_run)) NA_real_ else
    as.numeric(reference_run$wall_time_sec[[1L]])
  candidate_time <- if (is.null(candidate_run)) NA_real_ else
    as.numeric(candidate_run$wall_time_sec[[1L]])
  runtime_ratio <- if (is.finite(reference_time) && reference_time > 0 &&
                       is.finite(candidate_time)) {
    candidate_time / reference_time
  } else {
    NA_real_
  }
  data.frame(
    scenario_id = scenario_id,
    `repeat` = as.integer(repeat_id),
    reference_mode = reference_mode,
    candidate_mode = candidate_mode,
    promotion_gate = candidate_mode %in%
      c("mgcv_extract_cpu", "candidate_kpc", "legacy_mgcv"),
    reference_status = reference_status,
    candidate_status = candidate_status,
    reference_wall_time_sec = reference_time,
    candidate_wall_time_sec = candidate_time,
    runtime_ratio = runtime_ratio,
    runtime_delta_sec = if (is.finite(reference_time) &&
                            is.finite(candidate_time)) {
      candidate_time - reference_time
    } else {
      NA_real_
    },
    adjacency_identical = graph$adjacency_identical[[1L]],
    n_edgetests_identical = graph$n_edgetests_identical[[1L]],
    pmax_max_abs_diff = graph$pmax_max_abs_diff[[1L]],
    first_sepset_mismatch_rate = graph$first_sepset_mismatch_rate[[1L]],
    all_sepset_mismatch_rate = graph$all_sepset_mismatch_rate[[1L]],
    reference_backend_executed =
      if (is.null(reference_run)) NA_character_ else
        as.character(reference_run$backend_executed[[1L]]),
    candidate_backend_executed =
      if (is.null(candidate_run)) NA_character_ else
        as.character(candidate_run$backend_executed[[1L]]),
    passed = identical(reference_status, "ok") &&
      identical(candidate_status, "ok") &&
      isTRUE(graph$passed[[1L]]),
    stringsAsFactors = FALSE
  )
}

fastkpc_kpc_tprs_pvalue_drift_rows <- function(
    reference, candidate, scenario_id, repeat_id, reference_mode,
    candidate_mode, top_n = 5L) {
  empty <- data.frame(
    scenario_id = character(),
    `repeat` = integer(),
    reference_mode = character(),
    candidate_mode = character(),
    canonical_test_order_id = integer(),
    x = integer(),
    y = integer(),
    S_key = character(),
    p_used_reference = numeric(),
    p_used_candidate = numeric(),
    abs_p_used_diff = numeric(),
    p_raw_reference = numeric(),
    p_raw_candidate = numeric(),
    reference_backend_executed = character(),
    candidate_backend_executed = character(),
    stringsAsFactors = FALSE
  )
  if (is.null(reference) || is.null(candidate)) return(empty)
  ref_trace <- reference$diagnostics$precision_trace
  cand_trace <- candidate$diagnostics$precision_trace
  if (!is.data.frame(ref_trace) || !is.data.frame(cand_trace) ||
      nrow(ref_trace) == 0L || nrow(cand_trace) == 0L) {
    return(empty)
  }
  keys <- c("canonical_test_order_id", "x", "y", "S_key")
  required <- c(keys, "p_used", "p_raw", "backend_executed")
  if (!all(required %in% names(ref_trace)) ||
      !all(required %in% names(cand_trace))) {
    return(empty)
  }
  merged <- merge(
    ref_trace[, required, drop = FALSE],
    cand_trace[, required, drop = FALSE],
    by = keys,
    suffixes = c("_reference", "_candidate")
  )
  if (nrow(merged) == 0L) return(empty)
  merged$abs_p_used_diff <- abs(
    as.numeric(merged$p_used_candidate) -
      as.numeric(merged$p_used_reference)
  )
  merged <- merged[order(-merged$abs_p_used_diff,
                         merged$canonical_test_order_id), , drop = FALSE]
  keep <- seq_len(min(as.integer(top_n), nrow(merged)))
  merged <- merged[keep, , drop = FALSE]
  data.frame(
    scenario_id = scenario_id,
    `repeat` = as.integer(repeat_id),
    reference_mode = reference_mode,
    candidate_mode = candidate_mode,
    canonical_test_order_id =
      as.integer(merged$canonical_test_order_id),
    x = as.integer(merged$x),
    y = as.integer(merged$y),
    S_key = as.character(merged$S_key),
    p_used_reference = as.numeric(merged$p_used_reference),
    p_used_candidate = as.numeric(merged$p_used_candidate),
    abs_p_used_diff = as.numeric(merged$abs_p_used_diff),
    p_raw_reference = as.numeric(merged$p_raw_reference),
    p_raw_candidate = as.numeric(merged$p_raw_candidate),
    reference_backend_executed =
      as.character(merged$backend_executed_reference),
    candidate_backend_executed =
      as.character(merged$backend_executed_candidate),
    stringsAsFactors = FALSE
  )
}

fastkpc_kpc_tprs_promotion_summary <- function(qualification_summary,
                                               backend_comparison,
                                               no_oracle) {
  candidate_rows <- backend_comparison[
    backend_comparison$candidate_mode == "candidate_kpc", , drop = FALSE]
  legacy_rows <- backend_comparison[
    backend_comparison$candidate_mode == "legacy_mgcv", , drop = FALSE]
  comparison_passed <- nrow(candidate_rows) > 0L &&
    all(candidate_rows$passed %in% TRUE)
  no_oracle_passed <- nrow(no_oracle) > 0L && isTRUE(no_oracle$passed[[1L]])
  qualification_passed <- nrow(qualification_summary) > 0L &&
    isTRUE(qualification_summary$passed[[1L]])
  passed <- isTRUE(qualification_passed) && isTRUE(comparison_passed) &&
    isTRUE(no_oracle_passed)
  failure_reason <- if (passed) {
    ""
  } else {
    paste(c(
      if (!isTRUE(qualification_passed)) "qualification-failed" else NULL,
      if (!isTRUE(comparison_passed)) "candidate-comparison-failed" else NULL,
      if (!isTRUE(no_oracle_passed)) "no-oracle-failed" else NULL
    ), collapse = ";")
  }
  data.frame(
    passed = passed,
    candidate_rows = as.integer(nrow(candidate_rows)),
    candidate_median_runtime_ratio =
      if (nrow(candidate_rows) == 0L) NA_real_ else
        stats::median(candidate_rows$runtime_ratio, na.rm = TRUE),
    candidate_max_pmax_abs_diff =
      if (nrow(candidate_rows) == 0L) NA_real_ else
        max(candidate_rows$pmax_max_abs_diff, na.rm = TRUE),
    legacy_rows = as.integer(nrow(legacy_rows)),
    legacy_median_runtime_ratio =
      if (nrow(legacy_rows) == 0L) NA_real_ else
        stats::median(legacy_rows$runtime_ratio, na.rm = TRUE),
    no_oracle_forbidden_calls =
      if (nrow(no_oracle) == 0L) NA_integer_ else
        as.integer(no_oracle$forbidden_calls[[1L]]),
    recommendation = if (passed) {
      paste(
        "kpcTprsResidualCPP is qualified for continued experimental",
        "opt-in benchmarking on |S|<=2 real workloads"
      )
    } else {
      "keep kpcTprsResidualCPP experimental and inspect qualification artifacts"
    },
    failure_reason = failure_reason,
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
  backend_rows <- list()
  drift_rows <- list()
  trace_rows <- list()
  details <- list()
  modes <- c("reference_mgcv", "candidate_kpc", "legacy_mgcv")
  if (isTRUE(fastkpc_kpc_tprs_cuda_available())) {
    modes <- c(modes, "mgcv_extract_gpu")
  }

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
      mode_results <- list()
      for (mode in modes) {
        mode_result <- fastkpc_kpc_tprs_run_mode_result(
          data, scenario_id, repeat_id, mode, alpha, max_conditioning_size)
        mode_results[[mode]] <- mode_result
        run_rows[[length(run_rows) + 1L]] <- mode_result$row
        if (!is.null(mode_result$result)) {
          trace_rows[[length(trace_rows) + 1L]] <-
            fastkpc_kpc_tprs_trace_summary_row(
              mode_result$result, scenario_id, repeat_id, mode)
        }
      }
      reference <- mode_results$reference_mgcv$result
      candidate <- mode_results$candidate_kpc$result
      graph_rows[[length(graph_rows) + 1L]] <-
        fastkpc_kpc_tprs_graph_agreement_row(
          reference, candidate, scenario_id, repeat_id)
      backend_rows[[length(backend_rows) + 1L]] <-
        fastkpc_kpc_tprs_backend_comparison_row(
          reference = reference,
          candidate = reference,
          reference_run = mode_results$reference_mgcv$row,
          candidate_run = mode_results$reference_mgcv$row,
          scenario_id = scenario_id,
          repeat_id = repeat_id,
          reference_mode = "mgcv_extract_cpu",
          candidate_mode = "mgcv_extract_cpu")
      for (mode in setdiff(modes, "reference_mgcv")) {
        backend_rows[[length(backend_rows) + 1L]] <-
          fastkpc_kpc_tprs_backend_comparison_row(
            reference = reference,
            candidate = mode_results[[mode]]$result,
            reference_run = mode_results$reference_mgcv$row,
            candidate_run = mode_results[[mode]]$row,
            scenario_id = scenario_id,
            repeat_id = repeat_id,
            reference_mode = "mgcv_extract_cpu",
            candidate_mode = mode)
        drift_rows[[length(drift_rows) + 1L]] <-
          fastkpc_kpc_tprs_pvalue_drift_rows(
            reference = reference,
            candidate = mode_results[[mode]]$result,
            scenario_id = scenario_id,
            repeat_id = repeat_id,
            reference_mode = "mgcv_extract_cpu",
            candidate_mode = mode)
      }
      details[[paste(scenario_id, repeat_id, sep = "::")]] <-
        lapply(mode_results, `[[`, "result")
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
  backend_comparison <- if (length(backend_rows) == 0L) {
    data.frame()
  } else {
    do.call(rbind, backend_rows)
  }
  backend_comparison <-
    fastkpc_kpc_tprs_restore_repeat_name(backend_comparison)
  pvalue_drift <- if (length(drift_rows) == 0L) {
    data.frame()
  } else {
    do.call(rbind, drift_rows)
  }
  pvalue_drift <- fastkpc_kpc_tprs_restore_repeat_name(pvalue_drift)
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
  promotion_summary <- fastkpc_kpc_tprs_promotion_summary(
    qualification_summary = qualification_summary,
    backend_comparison = backend_comparison,
    no_oracle = no_oracle
  )
  summary <- list(
    passed = isTRUE(passed),
    scenarios = as.integer(length(scenarios)),
    repeats = as.integer(repeats),
    output_dir = output_dir,
    recommendation = promotion_summary$recommendation[[1L]]
  )
  paths <- list(
    runs = file.path(output_dir, "runs.csv"),
    graph_agreement = file.path(output_dir, "graph_agreement.csv"),
    trace_summary = file.path(output_dir, "trace_summary.csv"),
    backend_comparison = file.path(output_dir, "backend_comparison.csv"),
    pvalue_drift = file.path(output_dir, "pvalue_drift.csv"),
    qualification_summary = file.path(output_dir, "qualification_summary.csv"),
    promotion_summary = file.path(output_dir, "promotion_summary.csv"),
    no_oracle = file.path(output_dir, "no_oracle.csv"),
    summary_md = file.path(output_dir, "summary.md")
  )
  utils::write.csv(runs, paths$runs, row.names = FALSE)
  utils::write.csv(graph_agreement, paths$graph_agreement, row.names = FALSE)
  utils::write.csv(trace_summary, paths$trace_summary, row.names = FALSE)
  utils::write.csv(backend_comparison, paths$backend_comparison,
                   row.names = FALSE)
  utils::write.csv(pvalue_drift, paths$pvalue_drift, row.names = FALSE)
  utils::write.csv(qualification_summary, paths$qualification_summary,
                   row.names = FALSE)
  utils::write.csv(promotion_summary, paths$promotion_summary,
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
    paste0("- candidate median runtime ratio vs mgcvExtractCPU: ",
           signif(promotion_summary$candidate_median_runtime_ratio[[1L]], 4)),
    paste0("- candidate max trace p-value drift vs mgcvExtractCPU: ",
           if (nrow(pvalue_drift) == 0L) {
             NA_real_
           } else {
             signif(max(
               pvalue_drift$abs_p_used_diff[
                 pvalue_drift$candidate_mode == "candidate_kpc"
               ],
               na.rm = TRUE
             ), 4)
           }),
    paste0("- legacy median runtime ratio vs mgcvExtractCPU: ",
           signif(promotion_summary$legacy_median_runtime_ratio[[1L]], 4)),
    paste0("- no-oracle forbidden calls: ",
           promotion_summary$no_oracle_forbidden_calls[[1L]]),
    paste0("- recommendation: ", summary$recommendation)
  ), paths$summary_md)
  list(
    runs = runs,
    graph_agreement = graph_agreement,
    trace_summary = trace_summary,
    backend_comparison = backend_comparison,
    pvalue_drift = pvalue_drift,
    qualification_summary = qualification_summary,
    promotion_summary = promotion_summary,
    no_oracle = no_oracle,
    summary = summary,
    paths = paths,
    output_dir = output_dir,
    details = details
  )
}
