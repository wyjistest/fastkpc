source("fastkpc/R/mgcv_compat_contract.R")

fastkpc_log_distance_to_alpha <- function(p, alpha) {
  p <- max(as.numeric(p), .Machine$double.xmin)
  alpha <- max(as.numeric(alpha), .Machine$double.xmin)
  log(p / alpha)
}

fastkpc_empty_residual_compatibility_metrics <- function() {
  data.frame(
    scenario = character(),
    target = integer(),
    S_key = character(),
    backend = character(),
    residual_correlation = numeric(),
    relative_l2 = numeric(),
    max_abs_diff = numeric(),
    mean_diff = numeric(),
    sd_ratio = numeric(),
    basis_projection_floor = numeric(),
    current_lambda = numeric(),
    oracle_lambda = numeric(),
    current_lambda_residual_rel_l2 = numeric(),
    oracle_lambda_residual_rel_l2 = numeric(),
    oracle_lambda_improvement = numeric(),
    current_lambda_log_p_drift = numeric(),
    oracle_lambda_log_p_drift = numeric(),
    current_lambda_decision_flip = logical(),
    oracle_lambda_decision_flip = logical(),
    selected_sp = character(),
    edf = numeric(),
    score = numeric(),
    setup_fingerprint = character(),
    target_fingerprint = character(),
    stringsAsFactors = FALSE
  )
}

fastkpc_empty_ci_compatibility_metrics <- function() {
  data.frame(
    canonical_test_order_id = integer(),
    x = integer(),
    y = integer(),
    S_key = character(),
    conditioning_level = integer(),
    p_legacy = numeric(),
    p_backend = numeric(),
    log_p_ratio = numeric(),
    kernel_bandwidth_legacy = numeric(),
    kernel_bandwidth_candidate = numeric(),
    test_stat_legacy = numeric(),
    test_stat_candidate = numeric(),
    p_frozen_config = numeric(),
    p_native_config = numeric(),
    decision_legacy = logical(),
    decision_backend = logical(),
    decision_flip = logical(),
    decision_flip_frozen = logical(),
    decision_flip_native = logical(),
    distance_to_alpha_log = numeric(),
    backend_used = character(),
    fallback_triggered = logical(),
    verifier_backend = character(),
    stringsAsFactors = FALSE
  )
}

fastkpc_empty_graph_compatibility_metrics <- function() {
  data.frame(
    scenario = character(),
    backend = character(),
    skeleton_shd = integer(),
    skeleton_precision = numeric(),
    skeleton_recall = numeric(),
    skeleton_f1 = numeric(),
    edge_deletion_mismatch = integer(),
    sepset_mismatch_rate = numeric(),
    first_separating_set_mismatch = integer(),
    wanpdag_orientation_mismatch = integer(),
    arrowhead_agreement = numeric(),
    near_alpha_tests = integer(),
    verifier_calls = integer(),
    verifier_decision_changes = integer(),
    stringsAsFactors = FALSE
  )
}

fastkpc_empty_compatibility_campaign_metrics <- function() {
  list(
    residual = fastkpc_empty_residual_compatibility_metrics(),
    ci = fastkpc_empty_ci_compatibility_metrics(),
    graph = fastkpc_empty_graph_compatibility_metrics()
  )
}

fastkpc_make_ci_compatibility_row <- function(canonical_test_order_id,
                                              x, y, S,
                                              conditioning_level,
                                              p_legacy,
                                              p_backend,
                                              alpha,
                                              backend_used,
                                              fallback_triggered = FALSE,
                                              verifier_backend = "",
                                              kernel_bandwidth_legacy = NA_real_,
                                              kernel_bandwidth_candidate = NA_real_,
                                              test_stat_legacy = NA_real_,
                                              test_stat_candidate = NA_real_,
                                              p_frozen_config = NA_real_,
                                              p_native_config = NA_real_) {
  p_legacy_safe <- max(as.numeric(p_legacy), .Machine$double.xmin)
  p_backend_safe <- max(as.numeric(p_backend), .Machine$double.xmin)
  p_frozen <- as.numeric(p_frozen_config)
  p_native <- as.numeric(p_native_config)
  legacy_decision <- as.numeric(p_legacy) > alpha
  backend_decision <- as.numeric(p_backend) > alpha
  data.frame(
    canonical_test_order_id = as.integer(canonical_test_order_id),
    x = as.integer(x),
    y = as.integer(y),
    S_key = paste(sort(as.integer(S)), collapse = "|"),
    conditioning_level = as.integer(conditioning_level),
    p_legacy = as.numeric(p_legacy),
    p_backend = as.numeric(p_backend),
    log_p_ratio = log(p_backend_safe / p_legacy_safe),
    kernel_bandwidth_legacy = as.numeric(kernel_bandwidth_legacy),
    kernel_bandwidth_candidate = as.numeric(kernel_bandwidth_candidate),
    test_stat_legacy = as.numeric(test_stat_legacy),
    test_stat_candidate = as.numeric(test_stat_candidate),
    p_frozen_config = p_frozen,
    p_native_config = p_native,
    decision_legacy = legacy_decision,
    decision_backend = backend_decision,
    decision_flip = legacy_decision != backend_decision,
    decision_flip_frozen = if (is.na(p_frozen)) NA else legacy_decision != (p_frozen > alpha),
    decision_flip_native = if (is.na(p_native)) NA else legacy_decision != (p_native > alpha),
    distance_to_alpha_log = fastkpc_log_distance_to_alpha(p_backend, alpha),
    backend_used = as.character(backend_used),
    fallback_triggered = isTRUE(fallback_triggered),
    verifier_backend = as.character(verifier_backend),
    stringsAsFactors = FALSE
  )
}

fastkpc_vector_relative_l2 <- function(candidate, reference) {
  candidate <- as.numeric(candidate)
  reference <- as.numeric(reference)
  if (length(candidate) != length(reference)) {
    stop("candidate and reference must have the same length", call. = FALSE)
  }
  denom <- sqrt(sum(reference^2))
  diff <- sqrt(sum((candidate - reference)^2))
  if (denom == 0) return(diff)
  diff / denom
}

fastkpc_basis_projection_floor <- function(fitted_values, B,
                                           tol = sqrt(.Machine$double.eps)) {
  fitted_values <- as.numeric(fitted_values)
  B <- as.matrix(B)
  if (nrow(B) != length(fitted_values)) {
    stop("nrow(B) must equal length(fitted_values)", call. = FALSE)
  }
  if (ncol(B) == 0L) {
    projected <- rep(0, length(fitted_values))
  } else {
    projected <- as.numeric(stats::lm.fit(B, fitted_values, tol = tol)$fitted.values)
  }
  fastkpc_vector_relative_l2(projected, fitted_values)
}

fastkpc_oracle_lambda_gap <- function(legacy_residual,
                                      current_residual,
                                      oracle_residual,
                                      current_lambda,
                                      oracle_lambda,
                                      p_legacy = NA_real_,
                                      p_current = NA_real_,
                                      p_oracle = NA_real_,
                                      alpha = 0.05) {
  current_err <- fastkpc_vector_relative_l2(current_residual, legacy_residual)
  oracle_err <- fastkpc_vector_relative_l2(oracle_residual, legacy_residual)
  p_legacy_safe <- max(as.numeric(p_legacy), .Machine$double.xmin)
  p_current_safe <- max(as.numeric(p_current), .Machine$double.xmin)
  p_oracle_safe <- max(as.numeric(p_oracle), .Machine$double.xmin)
  legacy_decision <- as.numeric(p_legacy) > alpha

  list(
    current_lambda = as.numeric(current_lambda),
    oracle_lambda = as.numeric(oracle_lambda),
    current_lambda_residual_rel_l2 = current_err,
    oracle_lambda_residual_rel_l2 = oracle_err,
    oracle_lambda_improvement = current_err - oracle_err,
    current_lambda_log_p_drift = log(p_current_safe / p_legacy_safe),
    oracle_lambda_log_p_drift = log(p_oracle_safe / p_legacy_safe),
    current_lambda_decision_flip =
      if (is.na(as.numeric(p_current)) || is.na(as.numeric(p_legacy))) NA else
        legacy_decision != (as.numeric(p_current) > alpha),
    oracle_lambda_decision_flip =
      if (is.na(as.numeric(p_oracle)) || is.na(as.numeric(p_legacy))) NA else
        legacy_decision != (as.numeric(p_oracle) > alpha)
  )
}

fastkpc_fastspline_cuda_capabilities <- function() {
  list(
    backend = "fastSplineCUDA",
    role = "frozen approximate baseline",
    supported = list(
      residualization = TRUE,
      true_batched_cusolver = TRUE,
      cuda_dcov = TRUE,
      cuda_hsic = TRUE,
      residual_cache = TRUE,
      canonical_replay_compatible = TRUE
    ),
    unsupported = list(
      mgcv_setup_anchored = TRUE,
      mgcv_penalty_equivalent = TRUE,
      mgcv_constraint_equivalent = TRUE,
      raw_sp_comparable_to_mgcv = TRUE,
      default_s_s1_s2_tprs_equivalent = TRUE
    ),
    claims = list(
      mgcv_equivalent = FALSE,
      approximate_primary_backend = TRUE,
      frozen_baseline = TRUE
    ),
    version_pins = list(
      R_version = R.version.string,
      backend_version = "fastSplineCUDA-frozen-cusolver-v1",
      baseline_commit = "5233b38",
      baseline_note = "true-batched cuSOLVER fastSpline solves"
    )
  )
}

fastkpc_bind_metric_rows <- function(rows, empty) {
  if (length(rows) == 0L) return(empty)
  columns <- names(empty)
  rows <- lapply(rows, function(row) {
    missing <- setdiff(columns, names(row))
    for (name in missing) row[[name]] <- NA
    row[columns]
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

fastkpc_fastspline_cuda_capability_row <- function() {
  cap <- fastkpc_fastspline_cuda_capabilities()
  data.frame(
    backend = cap$backend,
    role = cap$role,
    backend_version = cap$version_pins$backend_version,
    baseline_commit = cap$version_pins$baseline_commit,
    true_batched_cusolver = isTRUE(cap$supported$true_batched_cusolver),
    cuda_dcov = isTRUE(cap$supported$cuda_dcov),
    cuda_hsic = isTRUE(cap$supported$cuda_hsic),
    frozen_baseline = isTRUE(cap$claims$frozen_baseline),
    mgcv_equivalent = isTRUE(cap$claims$mgcv_equivalent),
    raw_sp_comparable_to_mgcv = !isTRUE(cap$unsupported$raw_sp_comparable_to_mgcv),
    stringsAsFactors = FALSE
  )
}

fastkpc_precision_ladder_synthetic_case <- function(seed, n, alpha) {
  set.seed(seed)
  z <- seq(-2, 2, length.out = n)
  legacy_fitted <- sin(z) + 0.35 * cos(2 * z)
  legacy_residual <- stats::rnorm(n, sd = 0.08)

  B <- cbind(1, z, z^2)
  projected <- as.numeric(stats::lm.fit(B, legacy_fitted)$fitted.values)
  current_residual <- legacy_residual + 0.18 * scale(legacy_fitted - projected)[, 1]
  oracle_residual <- legacy_residual + 0.04 * scale(legacy_fitted - projected)[, 1]
  gap <- fastkpc_oracle_lambda_gap(
    legacy_residual = legacy_residual,
    current_residual = current_residual,
    oracle_residual = oracle_residual,
    current_lambda = 0.1,
    oracle_lambda = 0.03,
    p_legacy = alpha * exp(-0.25),
    p_current = alpha * exp(0.35),
    p_oracle = alpha * exp(0.12),
    alpha = alpha
  )

  scenario <- paste0("synthetic-attribution-seed", seed, "-n", n)
  setup_fp <- fastkpc_hash_object(list(seed = seed, n = n, B = round(B, 8)))
  target_fp <- fastkpc_hash_object(list(seed = seed, y = round(legacy_fitted + legacy_residual, 8)))
  residual_diff <- current_residual - legacy_residual
  sd_legacy <- stats::sd(legacy_residual)
  sd_current <- stats::sd(current_residual)

  residual <- data.frame(
    scenario = scenario,
    target = 1L,
    S_key = "2",
    backend = "fastSplineCUDA",
    residual_correlation = as.numeric(stats::cor(legacy_residual, current_residual)),
    relative_l2 = fastkpc_vector_relative_l2(current_residual, legacy_residual),
    max_abs_diff = max(abs(residual_diff)),
    mean_diff = mean(residual_diff),
    sd_ratio = if (sd_legacy == 0) NA_real_ else sd_current / sd_legacy,
    basis_projection_floor = fastkpc_basis_projection_floor(legacy_fitted, B),
    current_lambda = gap$current_lambda,
    oracle_lambda = gap$oracle_lambda,
    current_lambda_residual_rel_l2 = gap$current_lambda_residual_rel_l2,
    oracle_lambda_residual_rel_l2 = gap$oracle_lambda_residual_rel_l2,
    oracle_lambda_improvement = gap$oracle_lambda_improvement,
    current_lambda_log_p_drift = gap$current_lambda_log_p_drift,
    oracle_lambda_log_p_drift = gap$oracle_lambda_log_p_drift,
    current_lambda_decision_flip = gap$current_lambda_decision_flip,
    oracle_lambda_decision_flip = gap$oracle_lambda_decision_flip,
    selected_sp = as.character(gap$current_lambda),
    edf = ncol(B),
    score = sum(current_residual^2),
    setup_fingerprint = setup_fp,
    target_fingerprint = target_fp,
    stringsAsFactors = FALSE
  )

  ci <- fastkpc_make_ci_compatibility_row(
    canonical_test_order_id = as.integer(seed),
    x = 1L,
    y = 3L,
    S = 2L,
    conditioning_level = 1L,
    p_legacy = alpha * exp(-0.25),
    p_backend = alpha * exp(0.35),
    alpha = alpha,
    backend_used = "fastSplineCUDA",
    fallback_triggered = FALSE,
    verifier_backend = "",
    kernel_bandwidth_legacy = stats::sd(legacy_residual),
    kernel_bandwidth_candidate = stats::sd(current_residual),
    test_stat_legacy = sum(legacy_residual^2),
    test_stat_candidate = sum(current_residual^2),
    p_frozen_config = alpha * exp(0.22),
    p_native_config = alpha * exp(0.35)
  )

  graph <- data.frame(
    scenario = scenario,
    backend = "fastSplineCUDA",
    skeleton_shd = as.integer(ci$decision_flip),
    skeleton_precision = if (isTRUE(ci$decision_flip)) 0 else 1,
    skeleton_recall = if (isTRUE(ci$decision_flip)) 0 else 1,
    skeleton_f1 = if (isTRUE(ci$decision_flip)) 0 else 1,
    edge_deletion_mismatch = as.integer(ci$decision_flip),
    sepset_mismatch_rate = if (isTRUE(ci$decision_flip)) 1 else 0,
    first_separating_set_mismatch = as.integer(ci$decision_flip),
    wanpdag_orientation_mismatch = as.integer(ci$decision_flip),
    arrowhead_agreement = if (isTRUE(ci$decision_flip)) 0 else 1,
    near_alpha_tests = as.integer(abs(ci$distance_to_alpha_log) <= log(3)),
    verifier_calls = 0L,
    verifier_decision_changes = 0L,
    stringsAsFactors = FALSE
  )

  list(residual = residual, ci = ci, graph = graph)
}

fastkpc_precision_ladder_summary <- function(residual, ci, graph) {
  data.frame(
    rows_residual = as.integer(nrow(residual)),
    rows_ci = as.integer(nrow(ci)),
    rows_graph = as.integer(nrow(graph)),
    mean_basis_projection_floor = mean(residual$basis_projection_floor),
    mean_current_lambda_residual_rel_l2 =
      mean(residual$current_lambda_residual_rel_l2),
    mean_oracle_lambda_residual_rel_l2 =
      mean(residual$oracle_lambda_residual_rel_l2),
    mean_oracle_lambda_improvement =
      mean(residual$oracle_lambda_improvement),
    native_decision_flip_rate = mean(ci$decision_flip_native),
    frozen_decision_flip_rate = mean(ci$decision_flip_frozen),
    mean_skeleton_shd = mean(graph$skeleton_shd),
    stringsAsFactors = FALSE
  )
}

fastkpc_run_precision_ladder_attribution_campaign <- function(
    output_dir,
    seeds = c(5L, 6L),
    n_values = c(80L, 160L),
    alpha = 0.05) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  cases <- list()
  for (seed in seeds) {
    for (n in n_values) {
      cases[[length(cases) + 1L]] <- fastkpc_precision_ladder_synthetic_case(
        seed = seed,
        n = n,
        alpha = alpha
      )
    }
  }

  residual <- fastkpc_bind_metric_rows(
    lapply(cases, `[[`, "residual"),
    fastkpc_empty_residual_compatibility_metrics()
  )
  ci <- fastkpc_bind_metric_rows(
    lapply(cases, `[[`, "ci"),
    fastkpc_empty_ci_compatibility_metrics()
  )
  graph <- fastkpc_bind_metric_rows(
    lapply(cases, `[[`, "graph"),
    fastkpc_empty_graph_compatibility_metrics()
  )
  capabilities <- fastkpc_fastspline_cuda_capability_row()
  summary <- fastkpc_precision_ladder_summary(residual, ci, graph)

  utils::write.csv(residual,
                   file.path(output_dir, "mgcv_residual_compatibility.csv"),
                   row.names = FALSE)
  utils::write.csv(ci,
                   file.path(output_dir, "mgcv_ci_compatibility.csv"),
                   row.names = FALSE)
  utils::write.csv(graph,
                   file.path(output_dir, "mgcv_graph_compatibility.csv"),
                   row.names = FALSE)
  utils::write.csv(capabilities,
                   file.path(output_dir, "fastspline_cuda_capabilities.csv"),
                   row.names = FALSE)
  utils::write.csv(summary,
                   file.path(output_dir, "precision_ladder_attribution_summary.csv"),
                   row.names = FALSE)

  list(
    residual = residual,
    ci = ci,
    graph = graph,
    capabilities = capabilities,
    summary = summary,
    output_dir = output_dir
  )
}
