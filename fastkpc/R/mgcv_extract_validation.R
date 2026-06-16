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
    decision_legacy = logical(),
    decision_backend = logical(),
    decision_flip = logical(),
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
                                              verifier_backend = "") {
  p_legacy_safe <- max(as.numeric(p_legacy), .Machine$double.xmin)
  p_backend_safe <- max(as.numeric(p_backend), .Machine$double.xmin)
  data.frame(
    canonical_test_order_id = as.integer(canonical_test_order_id),
    x = as.integer(x),
    y = as.integer(y),
    S_key = paste(sort(as.integer(S)), collapse = "|"),
    conditioning_level = as.integer(conditioning_level),
    p_legacy = as.numeric(p_legacy),
    p_backend = as.numeric(p_backend),
    log_p_ratio = log(p_backend_safe / p_legacy_safe),
    decision_legacy = as.numeric(p_legacy) > alpha,
    decision_backend = as.numeric(p_backend) > alpha,
    decision_flip = (as.numeric(p_legacy) > alpha) != (as.numeric(p_backend) > alpha),
    distance_to_alpha_log = fastkpc_log_distance_to_alpha(p_backend, alpha),
    backend_used = as.character(backend_used),
    fallback_triggered = isTRUE(fallback_triggered),
    verifier_backend = as.character(verifier_backend),
    stringsAsFactors = FALSE
  )
}
