source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/mgcv_compat_contract.R")

fastkpc_full_cuda_census_parse_s <- function(S_key) {
  if (length(S_key) != 1L || is.na(S_key)) {
    stop("conditioning set key must be one non-missing string",
         call. = FALSE)
  }
  S_key <- as.character(S_key)
  if (!nzchar(S_key)) return(integer())
  values <- suppressWarnings(as.integer(
    strsplit(S_key, "|", fixed = TRUE)[[1L]]
  ))
  if (anyNA(values) ||
      any(values < 1L) ||
      !identical(values, sort(unique(values), method = "radix")) ||
      !identical(S_key, paste(values, collapse = "|"))) {
    stop("conditioning set key is not canonical sorted unique",
         call. = FALSE)
  }
  values
}

fastkpc_full_cuda_census_residual_payload <- function(
    target, S, formula_class, data_hash, n, p) {
  fields <- c(
    "schema_version=full-cuda-ci-residual-key-v1",
    paste0("dataset_sha256=", data_hash),
    paste0("n=", as.integer(n)),
    paste0("p=", as.integer(p)),
    paste0("target_index=", as.integer(target)),
    paste0("sorted_S=", paste(as.integer(S), collapse = ",")),
    paste0("formula_class=", formula_class),
    "family=gaussian",
    "link=identity",
    "method=GCV.Cp",
    "optimizer=mgcv-default",
    "gamma=1",
    "select=false",
    "scale=mgcv-default",
    "weights=none",
    "offset=none",
    "formula_semantics_version=kpcalg_regrXonS_v1",
    "mgcv_semantics_version=legacy-mgcv-gam-default-selection-v1"
  )
  paste0(paste(fields, collapse = "\n"), "\n")
}

fastkpc_full_cuda_census_same_s_payload <- function(
    S, formula_class, data_hash, n, p) {
  fields <- c(
    "schema_version=full-cuda-ci-same-s-key-v1",
    paste0("dataset_sha256=", data_hash),
    paste0("n=", as.integer(n)),
    paste0("p=", as.integer(p)),
    paste0("sorted_S=", paste(as.integer(S), collapse = ",")),
    paste0("formula_class=", formula_class),
    "family=gaussian",
    "link=identity",
    "method=GCV.Cp",
    "optimizer=mgcv-default",
    "gamma=1",
    "select=false",
    "scale=mgcv-default",
    "weights=none",
    "offset=none",
    "formula_semantics_version=kpcalg_regrXonS_v1",
    "mgcv_semantics_version=legacy-mgcv-gam-default-selection-v1"
  )
  paste0(paste(fields, collapse = "\n"), "\n")
}

fastkpc_full_cuda_census_request_identity <- function(
    target, S_key, formula_class) {
  paste(as.integer(target), as.character(S_key),
        as.character(formula_class), sep = "\t")
}

fastkpc_full_cuda_census_build_key_map <- function(
    target, S_key, formula_class, data_hash, n, p,
    hash_fun = fastkpc_full_cuda_census_hash_utf8) {
  identity <- fastkpc_full_cuda_census_request_identity(
    target, S_key, formula_class
  )
  keep <- !duplicated(identity)
  key_target <- as.integer(target[keep])
  key_S_key <- as.character(S_key[keep])
  key_formula <- as.character(formula_class[keep])
  key_S <- lapply(key_S_key, fastkpc_full_cuda_census_parse_s)
  payload <- mapply(
    fastkpc_full_cuda_census_residual_payload,
    target = key_target,
    S = key_S,
    formula_class = key_formula,
    MoreArgs = list(data_hash = data_hash, n = n, p = p),
    USE.NAMES = FALSE
  )
  hash <- vapply(payload, hash_fun, character(1L))

  same_identity <- paste(key_S_key, key_formula, sep = "\t")
  same_keep <- !duplicated(same_identity)
  same_S_key <- key_S_key[same_keep]
  same_formula <- key_formula[same_keep]
  same_S <- key_S[same_keep]
  same_payload <- mapply(
    fastkpc_full_cuda_census_same_s_payload,
    S = same_S,
    formula_class = same_formula,
    MoreArgs = list(data_hash = data_hash, n = n, p = p),
    USE.NAMES = FALSE
  )
  same_hash <- vapply(same_payload, hash_fun, character(1L))
  fastkpc_full_cuda_census_validate_key_mapping(payload, hash)
  fastkpc_full_cuda_census_validate_key_mapping(same_payload, same_hash)
  same_index <- match(same_identity, same_identity[same_keep])

  list(
    map = data.frame(
      request_identity = identity[keep],
      residual_key_payload = payload,
      residual_key_sha256 = hash,
      target = key_target,
      S_key = key_S_key,
      S_size = as.integer(lengths(key_S)),
      formula_class = key_formula,
      same_S_group_id = same_hash[same_index],
      stringsAsFactors = FALSE
    ),
    raw_hash = hash[match(identity, identity[keep])]
  )
}

fastkpc_full_cuda_census_residual_hash <- function(
    target, S, formula_class, data_hash, n, p) {
  fastkpc_full_cuda_census_hash_utf8(
    fastkpc_full_cuda_census_residual_payload(
      target = target,
      S = S,
      formula_class = formula_class,
      data_hash = data_hash,
      n = n,
      p = p
    )
  )
}

fastkpc_full_cuda_census_reference_decision <- function(p, alpha, deleted) {
  if (length(p) != 1L || !is.finite(p)) {
    stop("canonical reference p-value is non-finite", call. = FALSE)
  }
  independent <- p >= alpha
  if (!identical(independent, isTRUE(deleted))) {
    stop("reference alpha decision disagrees with deletes_edge",
         call. = FALSE)
  }
  if (independent) "independent" else "dependent"
}

fastkpc_full_cuda_census_deletion_key <- function(level, x, y, S_key) {
  paste(as.integer(level), pmin(as.integer(x), as.integer(y)),
        pmax(as.integer(x), as.integer(y)), as.character(S_key), sep = "|")
}

fastkpc_full_cuda_census_validate_deletions <- function(trace, deletions) {
  selected <- trace[trace$deletes_edge %in% TRUE, , drop = FALSE]
  selected_key <- fastkpc_full_cuda_census_deletion_key(
    selected$level, selected$x, selected$y, selected$S_key
  )
  deletion_key <- fastkpc_full_cuda_census_deletion_key(
    deletions$level, deletions$edge_x, deletions$edge_y, deletions$S_key
  )
  if (anyDuplicated(selected_key) || anyDuplicated(deletion_key) ||
      !identical(sort(selected_key, method = "radix"),
                 sort(deletion_key, method = "radix"))) {
    stop("logical deleting rows do not match deletion trace one to one",
         call. = FALSE)
  }
  invisible(TRUE)
}

fastkpc_full_cuda_census_logical_tests <- function(
    trace, deletions, alpha, data_hash, n, p) {
  trace <- as.data.frame(trace, stringsAsFactors = FALSE)
  deletions <- as.data.frame(deletions, stringsAsFactors = FALSE)
  required_trace <- c(
    "logical_sequence_id", "source_sequence_id", "source_task_index",
    "level", "x", "y", "S_key", "p_value", "deletes_edge"
  )
  missing_trace <- setdiff(required_trace, names(trace))
  if (length(missing_trace) > 0L) {
    stop("logical trace missing fields: ",
         paste(missing_trace, collapse = ","), call. = FALSE)
  }
  required_deletions <- c("level", "edge_x", "edge_y", "S_key")
  missing_deletions <- setdiff(required_deletions, names(deletions))
  if (length(missing_deletions) > 0L) {
    stop("deletion trace missing fields: ",
         paste(missing_deletions, collapse = ","), call. = FALSE)
  }
  alpha <- as.numeric(alpha)
  if (length(alpha) != 1L || !is.finite(alpha) || alpha <= 0) {
    stop("alpha must be one finite positive value", call. = FALSE)
  }

  S <- lapply(trace$S_key, fastkpc_full_cuda_census_parse_s)
  invisible(lapply(deletions$S_key, fastkpc_full_cuda_census_parse_s))
  S_size <- lengths(S)
  formula_class <- vapply(
    S, fastkpc_regrxons_formula_class, character(1L)
  )
  decisions <- mapply(
    fastkpc_full_cuda_census_reference_decision,
    p = as.numeric(trace$p_value),
    deleted = as.logical(trace$deletes_edge),
    MoreArgs = list(alpha = alpha),
    USE.NAMES = FALSE
  )
  fastkpc_full_cuda_census_validate_deletions(trace, deletions)

  conditional <- which(S_size > 0L)
  key_info <- fastkpc_full_cuda_census_build_key_map(
    target = c(trace$x[conditional], trace$y[conditional]),
    S_key = rep(trace$S_key[conditional], 2L),
    formula_class = rep(formula_class[conditional], 2L),
    data_hash = data_hash,
    n = n,
    p = p
  )
  conditional_count <- length(conditional)
  residual_key_x <- rep(NA_character_, nrow(trace))
  residual_key_y <- rep(NA_character_, nrow(trace))
  residual_key_x[conditional] <- key_info$raw_hash[
    seq_len(conditional_count)
  ]
  residual_key_y[conditional] <- key_info$raw_hash[
    conditional_count + seq_len(conditional_count)
  ]
  reference_p <- as.numeric(trace$p_value)
  signed_distance <- reference_p - alpha
  signed_log_ratio <- log(pmax(reference_p, .Machine$double.xmin) / alpha)

  out <- data.frame(
    logical_sequence_id = as.integer(trace$logical_sequence_id),
    source_sequence_id = as.integer(trace$source_sequence_id),
    source_task_index = as.integer(trace$source_task_index),
    level = as.integer(trace$level),
    x = as.integer(trace$x),
    y = as.integer(trace$y),
    S_key = as.character(trace$S_key),
    S_size = as.integer(S_size),
    formula_class = formula_class,
    reference_p_value = reference_p,
    alpha = rep(alpha, nrow(trace)),
    reference_decision = decisions,
    reference_independent = as.logical(trace$deletes_edge),
    deletes_edge = as.logical(trace$deletes_edge),
    selected_sepset = as.logical(trace$deletes_edge),
    signed_distance_from_alpha = signed_distance,
    absolute_distance_from_alpha = abs(signed_distance),
    signed_log_ratio_from_alpha = signed_log_ratio,
    absolute_log_distance_from_alpha = abs(signed_log_ratio),
    residual_key_x = residual_key_x,
    residual_key_y = residual_key_y,
    stringsAsFactors = FALSE
  )
  attr(out, "dataset_sha256") <- as.character(data_hash)
  attr(out, "data_n") <- as.integer(n)
  attr(out, "data_p") <- as.integer(p)
  attr(out, "p_floor") <- .Machine$double.xmin
  attr(out, "residual_key_map") <- key_info$map
  out
}

fastkpc_full_cuda_census_validate_key_mapping <- function(payload, hash) {
  mapping <- unique(data.frame(
    payload = as.character(payload),
    hash = as.character(hash),
    stringsAsFactors = FALSE
  ))
  if (anyDuplicated(mapping$hash)) {
    stop("residual key SHA-256 collision: one hash maps to multiple payloads",
         call. = FALSE)
  }
  if (anyDuplicated(mapping$payload)) {
    stop("residual key serialization error: one payload maps to multiple hashes",
         call. = FALSE)
  }
  invisible(TRUE)
}

fastkpc_full_cuda_census_residual_requests <- function(
    logical_tests, hash_fun = fastkpc_full_cuda_census_hash_utf8) {
  use_cached_map <- missing(hash_fun)
  logical_tests <- as.data.frame(logical_tests, stringsAsFactors = FALSE)
  required <- c(
    "logical_sequence_id", "level", "x", "y", "S_key", "S_size",
    "formula_class"
  )
  missing <- setdiff(required, names(logical_tests))
  if (length(missing) > 0L) {
    stop("logical census missing fields: ", paste(missing, collapse = ","),
         call. = FALSE)
  }
  data_hash <- attr(logical_tests, "dataset_sha256", exact = TRUE)
  n <- attr(logical_tests, "data_n", exact = TRUE)
  p <- attr(logical_tests, "data_p", exact = TRUE)
  if (is.null(data_hash) || is.null(n) || is.null(p)) {
    stop("logical census is missing canonical data identity attributes",
         call. = FALSE)
  }
  conditional <- logical_tests[logical_tests$S_size > 0L, , drop = FALSE]
  if (nrow(conditional) == 0L) {
    stop("logical census has no conditional residual requests", call. = FALSE)
  }

  raw <- rbind(
    data.frame(
      logical_sequence_id = conditional$logical_sequence_id,
      level = conditional$level,
      target = conditional$x,
      S_key = conditional$S_key,
      S_size = conditional$S_size,
      formula_class = conditional$formula_class,
      stringsAsFactors = FALSE
    ),
    data.frame(
      logical_sequence_id = conditional$logical_sequence_id,
      level = conditional$level,
      target = conditional$y,
      S_key = conditional$S_key,
      S_size = conditional$S_size,
      formula_class = conditional$formula_class,
      stringsAsFactors = FALSE
    )
  )
  raw_identity <- fastkpc_full_cuda_census_request_identity(
    raw$target, raw$S_key, raw$formula_class
  )
  key_map <- if (use_cached_map) {
    attr(logical_tests, "residual_key_map", exact = TRUE)
  } else {
    NULL
  }
  if (is.null(key_map)) {
    key_info <- fastkpc_full_cuda_census_build_key_map(
      target = raw$target,
      S_key = raw$S_key,
      formula_class = raw$formula_class,
      data_hash = data_hash,
      n = n,
      p = p,
      hash_fun = hash_fun
    )
    key_map <- key_info$map
  }
  map_index <- match(raw_identity, key_map$request_identity)
  if (anyNA(map_index)) {
    stop("logical requests do not map to the canonical residual key corpus",
         call. = FALSE)
  }
  raw$residual_key_sha256 <- key_map$residual_key_sha256[map_index]

  order_id <- order(raw$residual_key_sha256, raw$logical_sequence_id,
                    method = "radix")
  raw <- raw[order_id, , drop = FALSE]
  starts <- c(1L, which(raw$residual_key_sha256[-1L] !=
                          raw$residual_key_sha256[-nrow(raw)]) + 1L)
  ends <- c(starts[-1L] - 1L, nrow(raw))
  first <- raw[starts, , drop = FALSE]
  first_key <- key_map[
    match(first$residual_key_sha256, key_map$residual_key_sha256),
    , drop = FALSE
  ]
  result <- data.frame(
    residual_key_payload = first_key$residual_key_payload,
    residual_key_sha256 = first$residual_key_sha256,
    target = as.integer(first_key$target),
    S_key = first_key$S_key,
    S_size = as.integer(first_key$S_size),
    formula_class = first_key$formula_class,
    same_S_group_id = first_key$same_S_group_id,
    same_S_group_size = integer(length(starts)),
    request_multiplicity = as.integer(ends - starts + 1L),
    first_logical_sequence_id = vapply(
      seq_along(starts),
      function(i) min(raw$logical_sequence_id[starts[[i]]:ends[[i]]]),
      integer(1L)
    ),
    last_logical_sequence_id = vapply(
      seq_along(starts),
      function(i) max(raw$logical_sequence_id[starts[[i]]:ends[[i]]]),
      integer(1L)
    ),
    first_level = vapply(
      seq_along(starts),
      function(i) min(raw$level[starts[[i]]:ends[[i]]]),
      integer(1L)
    ),
    last_level = vapply(
      seq_along(starts),
      function(i) max(raw$level[starts[[i]]:ends[[i]]]),
      integer(1L)
    ),
    stringsAsFactors = FALSE
  )
  group_sizes <- table(result$same_S_group_id)
  result$same_S_group_size <- as.integer(
    group_sizes[result$same_S_group_id]
  )
  result <- result[
    order(result$residual_key_sha256, method = "radix"), , drop = FALSE
  ]
  rownames(result) <- NULL
  corpus_payload <- paste0(
    paste(result$residual_key_sha256, collapse = "\n"), "\n"
  )
  attr(result, "canonical_key_corpus_hash") <- hash_fun(corpus_payload)
  attr(result, "conditional_residual_request_count") <- nrow(raw)
  result
}

fastkpc_full_cuda_census_logical_fields <- function() {
  c(
    "logical_sequence_id", "source_sequence_id", "source_task_index",
    "level", "x", "y", "S_key", "S_size", "formula_class",
    "reference_p_value", "alpha", "reference_decision",
    "reference_independent", "deletes_edge", "selected_sepset",
    "signed_distance_from_alpha", "absolute_distance_from_alpha",
    "signed_log_ratio_from_alpha", "absolute_log_distance_from_alpha",
    "residual_key_x", "residual_key_y"
  )
}

fastkpc_full_cuda_census_logical_hash <- function(logical_tests) {
  data_hash <- attr(logical_tests, "dataset_sha256", exact = TRUE)
  data_n <- attr(logical_tests, "data_n", exact = TRUE)
  data_p <- attr(logical_tests, "data_p", exact = TRUE)
  p_floor <- attr(logical_tests, "p_floor", exact = TRUE)
  logical_tests <- as.data.frame(logical_tests, stringsAsFactors = FALSE)
  fields <- fastkpc_full_cuda_census_logical_fields()
  missing <- setdiff(fields, names(logical_tests))
  if (length(missing) > 0L || is.null(data_hash) || is.null(data_n) ||
      is.null(data_p) || is.null(p_floor)) {
    stop("logical census hash input is incomplete", call. = FALSE)
  }
  fastkpc_full_cuda_census_metadata_hash(list(
    schema_version = "full-cuda-ci-logical-census-v1",
    dataset_sha256 = as.character(data_hash),
    data_n = as.integer(data_n),
    data_p = as.integer(data_p),
    p_floor = as.numeric(p_floor),
    field_order = fields,
    columns = unname(lapply(fields, function(field) logical_tests[[field]]))
  ))
}

fastkpc_full_cuda_census_structural <- function(inputs) {
  required <- c(
    "data", "oracle", "oracle_input_hashes",
    "oracle_input_bundle_sha256", "oracle_inherited_graph_gate",
    "new_candidate_graph_gate"
  )
  missing <- setdiff(required, names(inputs))
  if (length(missing) > 0L) {
    stop("Phase 1 inputs missing fields: ", paste(missing, collapse = ","),
         call. = FALSE)
  }
  manifest <- inputs$oracle$manifest
  logical_tests <- fastkpc_full_cuda_census_logical_tests(
    trace = inputs$oracle$logical_trace,
    deletions = inputs$oracle$deletion_trace,
    alpha = as.numeric(manifest$alpha),
    data_hash = fastkpc_full_cuda_census_input_contract()$dataset_matrix_sha256,
    n = nrow(inputs$data),
    p = ncol(inputs$data)
  )
  residual_requests <- fastkpc_full_cuda_census_residual_requests(
    logical_tests
  )
  list(
    logical_tests = logical_tests,
    canonical_logical_census_hash =
      fastkpc_full_cuda_census_logical_hash(logical_tests),
    residual_requests = residual_requests,
    canonical_key_corpus_hash = attr(
      residual_requests, "canonical_key_corpus_hash", exact = TRUE
    ),
    conditional_residual_request_count = attr(
      residual_requests, "conditional_residual_request_count", exact = TRUE
    ),
    unique_conditional_S_count = length(unique(
      residual_requests$same_S_group_id
    )),
    oracle_input_hashes = inputs$oracle_input_hashes,
    oracle_input_bundle_sha256 = inputs$oracle_input_bundle_sha256,
    oracle_inherited_graph_gate = inputs$oracle_inherited_graph_gate,
    new_candidate_graph_gate = inputs$new_candidate_graph_gate,
    p_floor = attr(logical_tests, "p_floor", exact = TRUE),
    historical_route_metric = list(
      name = "s_affinity_executed_mgcv_fit_count",
      value = 273284L,
      metric_role = "historical_route_metric",
      hard_gate = FALSE,
      provenance_status = "no_hash_protected_route_trace"
    )
  )
}

fastkpc_full_cuda_census_validate_structural <- function(
    structural, canonical = FALSE) {
  required <- c(
    "logical_tests", "canonical_logical_census_hash", "residual_requests",
    "canonical_key_corpus_hash",
    "conditional_residual_request_count", "unique_conditional_S_count",
    "oracle_inherited_graph_gate", "new_candidate_graph_gate",
    "historical_route_metric"
  )
  missing <- setdiff(required, names(structural))
  if (length(missing) > 0L) {
    stop("structural census missing fields: ", paste(missing, collapse = ","),
         call. = FALSE)
  }
  logical_tests <- structural$logical_tests
  requests <- structural$residual_requests
  actual_logical_hash <- fastkpc_full_cuda_census_logical_hash(logical_tests)
  if (!identical(as.character(structural$canonical_logical_census_hash),
                 actual_logical_hash)) {
    stop("logical census hash mismatch", call. = FALSE)
  }
  if (!isTRUE(structural$oracle_inherited_graph_gate) ||
      !identical(structural$new_candidate_graph_gate, "NOT_APPLICABLE")) {
    stop("structural census inherited graph scope is invalid", call. = FALSE)
  }
  if (anyDuplicated(requests$residual_key_payload) ||
      anyDuplicated(requests$residual_key_sha256)) {
    stop("structural census contains duplicate residual keys", call. = FALSE)
  }
  if (!identical(sum(requests$request_multiplicity),
                 as.integer(structural$conditional_residual_request_count))) {
    stop("structural census request multiplicity is inconsistent",
         call. = FALSE)
  }
  if (isTRUE(structural$historical_route_metric$hard_gate) ||
      !identical(structural$historical_route_metric$metric_role,
                 "historical_route_metric")) {
    stop("historical S-affinity route metric is incorrectly gating",
         call. = FALSE)
  }
  if (!isTRUE(canonical)) return(TRUE)

  contract <- fastkpc_full_cuda_census_input_contract()
  gates <- c(
    nrow(logical_tests) == 240489L,
    sum(logical_tests$S_size > 0L) == 238276L,
    structural$conditional_residual_request_count == 476552L,
    nrow(requests) == 110617L,
    structural$unique_conditional_S_count == 8634L,
    identical(structural$canonical_logical_census_hash,
              contract$canonical_logical_census_hash),
    identical(structural$canonical_key_corpus_hash,
              contract$canonical_key_corpus_hash)
  )
  if (!all(gates)) {
    stop("canonical structural census hard gate failed", call. = FALSE)
  }
  TRUE
}

fastkpc_full_cuda_census_require_parity_namespaces <- function() {
  fastkpc_full_cuda_require_namespace("mgcv")
  fastkpc_full_cuda_require_namespace("kpcalg")
  invisible(TRUE)
}

fastkpc_full_cuda_census_capture_warnings <- function(expression) {
  warnings <- list()
  started <- proc.time()[["elapsed"]]
  value <- withCallingHandlers(
    force(expression),
    warning = function(condition) {
      warnings[[length(warnings) + 1L]] <<- list(
        class = class(condition),
        message = conditionMessage(condition)
      )
      invokeRestart("muffleWarning")
    }
  )
  list(
    value = value,
    warnings = warnings,
    elapsed_ms = (proc.time()[["elapsed"]] - started) * 1000
  )
}

fastkpc_full_cuda_census_formula_function <- function(S_size) {
  fastkpc_full_cuda_census_require_parity_namespaces()
  name <- if (as.integer(S_size) > 2L) {
    "frml.additive.smooth"
  } else {
    "frml.full.smooth"
  }
  getFromNamespace(name, "kpcalg")
}

fastkpc_full_cuda_census_pair_layout_fits <- function(X, S_data) {
  fastkpc_full_cuda_census_require_parity_namespaces()
  X <- as.matrix(X)
  S_data <- as.matrix(S_data)
  if (ncol(X) != 2L || nrow(X) != nrow(S_data) || ncol(S_data) < 1L) {
    stop("pair layout requires two targets and a nonempty conditioning set",
         call. = FALSE)
  }
  data <- data.frame(cbind(X, S_data))
  names(data) <- paste0("x", seq_len(ncol(data)))
  formula_fun <- fastkpc_full_cuda_census_formula_function(ncol(S_data))
  predictors <- (ncol(X) + 1L):(ncol(X) + ncol(S_data))
  lapply(seq_len(ncol(X)), function(i) {
    formula <- formula_fun(i, predictors)
    fastkpc_full_cuda_census_capture_warnings(
      mgcv::gam(formula, data = data)
    )
  })
}

fastkpc_full_cuda_census_single_target_fit <- function(y, S_data) {
  fastkpc_full_cuda_census_require_parity_namespaces()
  y <- as.numeric(y)
  S_data <- as.matrix(S_data)
  if (length(y) != nrow(S_data) || ncol(S_data) < 1L) {
    stop("single-target layout requires aligned nonempty conditioning data",
         call. = FALSE)
  }
  data <- data.frame(cbind(y, S_data))
  names(data) <- paste0("x", seq_len(ncol(data)))
  formula_fun <- fastkpc_full_cuda_census_formula_function(ncol(S_data))
  formula <- formula_fun(1L, 2L:(1L + ncol(S_data)))
  fastkpc_full_cuda_census_capture_warnings(
    mgcv::gam(formula, data = data)
  )
}

fastkpc_full_cuda_census_metadata_hash <- function(value) {
  normalize <- function(x) {
    if (is.numeric(x)) storage.mode(x) <- "double"
    names(x) <- NULL
    if (!is.null(dim(x))) dimnames(x) <- NULL
    if (is.list(x)) x <- lapply(x, normalize)
    x
  }
  fastkpc_full_cuda_census_hash_raw(
    serialize(normalize(value), NULL, version = 2)
  )
}

fastkpc_full_cuda_census_make_canonical_parity_case <- function(
    inputs, trace_row) {
  S <- fastkpc_full_cuda_census_parse_s(trace_row$S_key[[1L]])
  list(
    case_id = paste0("canonical-", trace_row$logical_sequence_id[[1L]]),
    source_type = "canonical",
    logical_sequence_id = as.integer(trace_row$logical_sequence_id[[1L]]),
    S_size = length(S),
    X = inputs$data[, c(trace_row$x[[1L]], trace_row$y[[1L]]), drop = FALSE],
    S_data = inputs$data[, S, drop = FALSE],
    alpha = as.numeric(inputs$oracle$manifest$alpha),
    index = as.integer(inputs$oracle$manifest$index),
    numCol = as.integer(inputs$oracle$manifest$numCol)
  )
}

fastkpc_full_cuda_census_synthetic_parity_cases <- function() {
  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_seed) old_seed <- get(".Random.seed", envir = .GlobalEnv)
  on.exit({
    if (had_seed) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv,
                      inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)
  set.seed(20260710)
  n <- 351L
  z <- stats::rnorm(n)
  list(
    list(
      case_id = "synthetic-rank-deficient",
      source_type = "synthetic",
      logical_sequence_id = NA_integer_,
      S_size = 2L,
      X = cbind(z + stats::rnorm(n) * 0.2,
                -z + stats::rnorm(n) * 0.2),
      S_data = cbind(z, 2 * z),
      alpha = 0.1,
      index = 1L,
      numCol = 35L
    ),
    list(
      case_id = "synthetic-near-constant",
      source_type = "synthetic",
      logical_sequence_id = NA_integer_,
      S_size = 1L,
      X = cbind(stats::rnorm(n), stats::rnorm(n)),
      S_data = cbind(1 + stats::rnorm(n) * 1e-10),
      alpha = 0.1,
      index = 1L,
      numCol = 35L
    )
  )
}

fastkpc_full_cuda_census_parity_cases <- function(inputs) {
  trace <- inputs$oracle$logical_trace
  conditional <- trace[trace$level > 0L, , drop = FALSE]
  S_size <- vapply(
    conditional$S_key,
    function(value) length(fastkpc_full_cuda_census_parse_s(value)),
    integer(1L)
  )
  p_floor <- .Machine$double.xmin
  distance <- abs(log(pmax(conditional$p_value, p_floor) /
                        as.numeric(inputs$oracle$manifest$alpha)))
  first_for_size <- function(size) {
    which(S_size == as.integer(size))[[1L]]
  }
  selected <- unique(c(
    first_for_size(1L),
    first_for_size(2L),
    first_for_size(3L),
    which(S_size == max(S_size))[[1L]],
    which.min(distance)
  ))
  canonical <- lapply(
    selected,
    function(index) fastkpc_full_cuda_census_make_canonical_parity_case(
      inputs, conditional[index, , drop = FALSE]
    )
  )
  c(canonical, fastkpc_full_cuda_census_synthetic_parity_cases())
}

fastkpc_full_cuda_census_fit_values <- function(captured) {
  fit <- captured$value
  selected_sp <- fit$sp
  list(
    residuals = as.numeric(stats::residuals(fit)),
    fitted = as.numeric(stats::fitted(fit)),
    selected_sp = as.numeric(selected_sp),
    selected_sp_names = names(selected_sp),
    GCV_Cp = as.numeric(fit$gcv.ubre),
    EDF = as.numeric(sum(fit$edf)),
    warning_count = length(captured$warnings),
    elapsed_ms = as.numeric(captured$elapsed_ms)
  )
}

fastkpc_full_cuda_census_max_abs_diff <- function(left, right) {
  left <- as.double(left)
  right <- as.double(right)
  if (length(left) != length(right)) return(Inf)
  if (identical(left, right) || length(left) == 0L) return(0)
  if (any(!is.finite(left)) || any(!is.finite(right))) return(Inf)
  max(abs(left - right))
}

fastkpc_full_cuda_census_sp_names <- function(fits) {
  paste(vapply(fits, function(fit) {
    value <- fit$selected_sp_names
    if (is.null(value) || length(value) == 0L) "<unnamed>" else
      paste(value, collapse = "|")
  }, character(1L)), collapse = ";")
}

fastkpc_full_cuda_census_parity_case <- function(case) {
  fastkpc_full_cuda_census_require_parity_namespaces()
  pair_captured <- fastkpc_full_cuda_census_pair_layout_fits(
    case$X, case$S_data
  )
  single_captured <- lapply(seq_len(ncol(case$X)), function(i) {
    fastkpc_full_cuda_census_single_target_fit(
      case$X[, i], case$S_data
    )
  })
  actual_regr <- fastkpc_full_cuda_census_capture_warnings(
    getFromNamespace("regrXonS", "kpcalg")(case$X, case$S_data)
  )
  pair <- lapply(pair_captured, fastkpc_full_cuda_census_fit_values)
  single <- lapply(single_captured, fastkpc_full_cuda_census_fit_values)
  pair_residuals <- do.call(cbind, lapply(pair, `[[`, "residuals"))
  single_residuals <- do.call(cbind, lapply(single, `[[`, "residuals"))
  pair_fitted <- do.call(cbind, lapply(pair, `[[`, "fitted"))
  single_fitted <- do.call(cbind, lapply(single, `[[`, "fitted"))
  regr_residuals <- as.matrix(actual_regr$value)
  dcov <- getFromNamespace("dcov.gamma", "kpcalg")
  p_regr <- as.numeric(dcov(
    regr_residuals[, 1L], regr_residuals[, 2L],
    index = case$index, numCol = case$numCol
  )$p.value)
  p_pair <- as.numeric(dcov(
    pair_residuals[, 1L], pair_residuals[, 2L],
    index = case$index, numCol = case$numCol
  )$p.value)
  p_single <- as.numeric(dcov(
    single_residuals[, 1L], single_residuals[, 2L],
    index = case$index, numCol = case$numCol
  )$p.value)

  regr_pair_identical <- identical(
    fastkpc_full_cuda_census_metadata_hash(regr_residuals),
    fastkpc_full_cuda_census_metadata_hash(pair_residuals)
  )
  regr_pair_abs_diff <- fastkpc_full_cuda_census_max_abs_diff(
    regr_residuals, pair_residuals
  )
  pair_single_identical <- identical(
    fastkpc_full_cuda_census_metadata_hash(pair_residuals),
    fastkpc_full_cuda_census_metadata_hash(single_residuals)
  )
  pair_single_abs_diff <- fastkpc_full_cuda_census_max_abs_diff(
    pair_residuals, single_residuals
  )
  fitted_identical <- identical(
    fastkpc_full_cuda_census_metadata_hash(pair_fitted),
    fastkpc_full_cuda_census_metadata_hash(single_fitted)
  )
  fitted_abs_diff <- fastkpc_full_cuda_census_max_abs_diff(
    pair_fitted, single_fitted
  )
  sp_identical <- all(vapply(seq_along(pair), function(i) {
    identical(pair[[i]]$selected_sp, single[[i]]$selected_sp)
  }, logical(1L)))
  sp_abs_diff <- max(vapply(seq_along(pair), function(i) {
    fastkpc_full_cuda_census_max_abs_diff(
      pair[[i]]$selected_sp, single[[i]]$selected_sp
    )
  }, numeric(1L)))
  gcv_identical <- all(vapply(seq_along(pair), function(i) {
    identical(pair[[i]]$GCV_Cp, single[[i]]$GCV_Cp)
  }, logical(1L)))
  gcv_abs_diff <- max(vapply(seq_along(pair), function(i) {
    fastkpc_full_cuda_census_max_abs_diff(
      pair[[i]]$GCV_Cp, single[[i]]$GCV_Cp
    )
  }, numeric(1L)))
  edf_identical <- all(vapply(seq_along(pair), function(i) {
    identical(pair[[i]]$EDF, single[[i]]$EDF)
  }, logical(1L)))
  edf_abs_diff <- max(vapply(seq_along(pair), function(i) {
    fastkpc_full_cuda_census_max_abs_diff(pair[[i]]$EDF, single[[i]]$EDF)
  }, numeric(1L)))
  p_identical <- identical(p_regr, p_pair) && identical(p_pair, p_single)
  decision_identical <- identical(p_regr >= case$alpha,
                                  p_single >= case$alpha)
  pass <- all(c(
    regr_pair_identical, pair_single_identical, fitted_identical,
    sp_identical, gcv_identical, edf_identical, p_identical,
    decision_identical
  ))

  data.frame(
    case_id = case$case_id,
    source_type = case$source_type,
    logical_sequence_id = as.integer(case$logical_sequence_id),
    S_size = as.integer(case$S_size),
    regr_pair_residual_hash_identical = regr_pair_identical,
    regr_pair_residual_max_abs_diff = regr_pair_abs_diff,
    pair_single_residual_hash_identical = pair_single_identical,
    pair_single_residual_max_abs_diff = pair_single_abs_diff,
    fitted_hash_identical = fitted_identical,
    fitted_max_abs_diff = fitted_abs_diff,
    selected_sp_identical = sp_identical,
    pair_selected_sp_names = fastkpc_full_cuda_census_sp_names(pair),
    single_selected_sp_names = fastkpc_full_cuda_census_sp_names(single),
    selected_sp_max_abs_diff = sp_abs_diff,
    GCV_Cp_identical = gcv_identical,
    GCV_Cp_max_abs_diff = gcv_abs_diff,
    EDF_identical = edf_identical,
    EDF_max_abs_diff = edf_abs_diff,
    dcov_p_value_identical = p_identical,
    dcov_p_value_abs_diff = max(abs(c(p_regr - p_pair,
                                      p_pair - p_single))),
    decision_identical = decision_identical,
    pair_warning_count = sum(vapply(pair, `[[`, integer(1L),
                                      "warning_count")),
    single_warning_count = sum(vapply(single, `[[`, integer(1L),
                                        "warning_count")),
    pair_fit_elapsed_ms = sum(vapply(pair, `[[`, numeric(1L),
                                      "elapsed_ms")),
    single_fit_elapsed_ms = sum(vapply(single, `[[`, numeric(1L),
                                        "elapsed_ms")),
    regr_elapsed_ms = as.numeric(actual_regr$elapsed_ms),
    pass = pass,
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_census_write_parity <- function(cases, results, output_dir) {
  if (!is.list(cases) || length(cases) == 0L ||
      !is.data.frame(results) || nrow(results) != length(cases)) {
    stop("parity artifact cases/results are inconsistent", call. = FALSE)
  }
  if (!"pass" %in% names(results) || anyNA(results$pass) ||
      !all(results$pass)) {
    stop("legacy layout parity failed", call. = FALSE)
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  paths <- list(
    cases_rds = file.path(output_dir, "legacy_layout_parity_cases.rds"),
    results_csv = file.path(output_dir,
                            "legacy_layout_parity_results.csv")
  )
  saveRDS(cases, paths$cases_rds, version = 2)
  utils::write.csv(results, paths$results_csv, row.names = FALSE)
  paths
}

fastkpc_full_cuda_census_risk_config <- function() {
  list(
    risk_schema_version = "full-cuda-ci-risk-v1",
    near_constant_threshold = sqrt(.Machine$double.eps),
    rank_estimator = "lapack-svd-v1",
    rank_tolerance_formula = paste0(
      "max(dim(A))*max(singular_values(A))*",
      ".Machine$double.eps"
    ),
    condition_estimator = "lapack-svd-ratio-v1",
    condition_buckets = c(
      "not_applicable_empty", "finite_lt_1e4",
      "finite_1e4_to_lt_1e8", "finite_1e8_to_lt_1e12",
      "finite_ge_1e12", "rank_deficient_inf", "nonfinite_unknown"
    ),
    high_condition_threshold = 1e12,
    near_alpha_tau = log(2),
    near_alpha_buckets = c(
      "exact_boundary", "le_1e_minus_12", "le_1e_minus_9",
      "le_1e_minus_6", "le_1e_minus_3", "le_log_1_01",
      "le_log_1_1", "le_log_2", "farther"
    ),
    p_floor = .Machine$double.xmin,
    nonfinite_policy = "preserve-and-classify-fail-if-unclassified-v1",
    selected_sp_policy = "mgcv-first-sp-order-values-and-names-v1",
    warning_capture_policy =
      "withCallingHandlers-class-message-order-muffle-v1",
    convergence_field_policy =
      "raw-fit-converged-outer.info-mgcv.conv-with-provenance-v1"
  )
}

fastkpc_full_cuda_census_condition_bucket <- function(condition, rank,
                                                       expected_rank) {
  expected_rank <- as.integer(expected_rank)
  if (is.na(condition) && identical(as.integer(rank), 0L) &&
      identical(expected_rank, 0L)) {
    return("not_applicable_empty")
  }
  if (is.na(condition) || is.na(rank)) return("nonfinite_unknown")
  if (rank < expected_rank || is.infinite(condition)) {
    return("rank_deficient_inf")
  }
  if (condition < 1e4) return("finite_lt_1e4")
  if (condition < 1e8) return("finite_1e4_to_lt_1e8")
  if (condition < 1e12) return("finite_1e8_to_lt_1e12")
  "finite_ge_1e12"
}

fastkpc_full_cuda_census_svd_diagnostics <- function(
    A, expected_rank = min(dim(as.matrix(A)))) {
  A <- as.matrix(A)
  storage.mode(A) <- "double"
  dimnames(A) <- NULL
  expected_rank <- as.integer(expected_rank)
  if (length(expected_rank) != 1L || is.na(expected_rank) ||
      expected_rank < 0L) {
    stop("expected_rank must be one nonnegative integer", call. = FALSE)
  }
  if (any(dim(A) == 0L)) {
    return(list(
      rank = 0L,
      condition = NA_real_,
      bucket = "not_applicable_empty",
      tolerance = NA_real_
    ))
  }
  if (any(!is.finite(A))) {
    return(list(
      rank = NA_integer_,
      condition = NA_real_,
      bucket = "nonfinite_unknown",
      tolerance = NA_real_
    ))
  }
  singular <- La.svd(A, nu = 0L, nv = 0L)$d
  if (any(!is.finite(singular))) {
    return(list(
      rank = NA_integer_,
      condition = NA_real_,
      bucket = "nonfinite_unknown",
      tolerance = NA_real_
    ))
  }
  smax <- max(singular)
  if (identical(smax, 0)) {
    return(list(
      rank = 0L,
      condition = Inf,
      bucket = "rank_deficient_inf",
      tolerance = 0
    ))
  }
  tolerance <- max(dim(A)) * smax * .Machine$double.eps
  rank <- sum(singular > tolerance)
  condition <- if (rank < expected_rank) Inf else smax / min(singular)
  list(
    rank = as.integer(rank),
    condition = condition,
    bucket = fastkpc_full_cuda_census_condition_bucket(
      condition, as.integer(rank), expected_rank
    ),
    tolerance = tolerance
  )
}

fastkpc_full_cuda_census_near_constant <- function(value) {
  value <- as.double(value)
  sd_value <- if (length(value) < 2L || any(!is.finite(value))) {
    NA_real_
  } else {
    sqrt(sum((value - mean(value))^2) / (length(value) - 1L))
  }
  list(
    sd = sd_value,
    near_constant = !is.finite(sd_value) ||
      sd_value <= sqrt(.Machine$double.eps)
  )
}

fastkpc_full_cuda_census_near_alpha_bucket <- function(distance) {
  distance <- as.numeric(distance)
  if (length(distance) != 1L || !is.finite(distance)) {
    return("nonfinite_unknown")
  }
  if (distance < 0) {
    stop("near-alpha distance must be nonnegative", call. = FALSE)
  }
  if (distance == 0) return("exact_boundary")
  limits <- c(1e-12, 1e-9, 1e-6, 1e-3,
              log(1.01), log(1.1), log(2))
  labels <- c(
    "le_1e_minus_12", "le_1e_minus_9", "le_1e_minus_6",
    "le_1e_minus_3", "le_log_1_01", "le_log_1_1", "le_log_2"
  )
  index <- which(distance <= limits)[1L]
  if (is.na(index)) "farther" else labels[[index]]
}

fastkpc_full_cuda_census_right_nullspace <- function(C) {
  C <- as.matrix(C)
  storage.mode(C) <- "double"
  dimnames(C) <- NULL
  coefficient_count <- ncol(C)
  if (nrow(C) == 0L) return(diag(coefficient_count))
  diagnostics <- fastkpc_full_cuda_census_svd_diagnostics(
    C, expected_rank = min(dim(C))
  )
  if (is.na(diagnostics$rank)) {
    stop("cannot construct nullspace for non-finite constraints",
         call. = FALSE)
  }
  rank <- diagnostics$rank
  if (rank >= coefficient_count) {
    return(matrix(numeric(), nrow = coefficient_count, ncol = 0L))
  }
  decomposition <- La.svd(C, nu = 0L, nv = coefficient_count)
  transpose_v <- decomposition$vt
  t(transpose_v[seq.int(rank + 1L, coefficient_count), , drop = FALSE])
}

fastkpc_full_cuda_census_setup_fingerprint <- function(
    same_S_group_id, n, formula_class, model_matrix_hash,
    penalty_hashes, penalty_offsets, constraint_hash, H_hash,
    weights_policy, offset_policy, rank_metadata_hash) {
  if (!grepl("^[0-9a-f]{64}$", same_S_group_id)) {
    stop("same-S group id must be one lowercase SHA-256", call. = FALSE)
  }
  if (!formula_class %in% c("full-smooth", "additive-smooth")) {
    stop("setup fingerprint requires a conditional formula class",
         call. = FALSE)
  }
  fields <- c(
    "schema_version=full-cuda-ci-same-s-setup-fingerprint-v1",
    paste0(
      "dataset_sha256=",
      fastkpc_full_cuda_census_input_contract()$dataset_matrix_sha256
    ),
    paste0("same_S_group_id=", same_S_group_id),
    paste0("n=", as.integer(n)),
    "canonical_input_p=48",
    paste0("formula_class=", formula_class),
    "family=gaussian",
    "link=identity",
    "method=GCV.Cp",
    "optimizer=mgcv-default",
    "gamma=1",
    "select=false",
    "scale=mgcv-default",
    paste0("model_matrix_hash=", model_matrix_hash),
    paste0("penalty_hashes=", paste(penalty_hashes, collapse = ",")),
    paste0("penalty_offsets=", paste(as.integer(penalty_offsets),
                                      collapse = ",")),
    paste0("constraint_hash=", constraint_hash),
    paste0("H_hash=", H_hash),
    paste0("weights_policy=", weights_policy),
    paste0("offset_policy=", offset_policy),
    paste0("rank_metadata_hash=", rank_metadata_hash)
  )
  payload <- paste0(paste(fields, collapse = "\n"), "\n")
  list(
    payload = payload,
    sha256 = fastkpc_full_cuda_census_hash_utf8(payload)
  )
}

fastkpc_full_cuda_census_request_row <- function(request_row) {
  request_row <- as.data.frame(request_row, stringsAsFactors = FALSE)
  required <- c(
    "residual_key_sha256", "target", "S_key", "S_size",
    "formula_class", "same_S_group_id", "shard_id"
  )
  missing <- setdiff(required, names(request_row))
  if (nrow(request_row) != 1L || length(missing) > 0L) {
    stop("metadata extraction requires one complete residual request row",
         call. = FALSE)
  }
  request_row
}

fastkpc_full_cuda_census_penalty_components <- function(fit,
                                                         coefficient_count) {
  blocks <- list()
  embedded <- list()
  offsets <- integer()
  ranks <- integer()
  dimensions <- character()
  sp_indices <- integer()
  smooth_classes <- character()
  basis_dimensions <- integer()
  for (smooth in fit$smooth) {
    smooth_class <- class(smooth)[[1L]]
    smooth_classes <- c(smooth_classes, smooth_class)
    basis_dimensions <- c(
      basis_dimensions,
      as.integer(if (is.null(smooth$bs.dim)) {
        smooth$last.para - smooth$first.para + 1L
      } else {
        smooth$bs.dim
      })
    )
    penalties <- smooth$S
    if (is.null(penalties) || length(penalties) == 0L) next
    coefficient_index <- seq.int(
      as.integer(smooth$first.para), as.integer(smooth$last.para)
    )
    if (is.null(smooth$first.sp) || is.null(smooth$last.sp)) {
      stop("mgcv smooth is missing penalty-order metadata", call. = FALSE)
    }
    smooth_sp_indices <- seq.int(smooth$first.sp, smooth$last.sp)
    if (length(smooth_sp_indices) != length(penalties)) {
      stop("mgcv smooth penalty order is inconsistent", call. = FALSE)
    }
    for (penalty_index in seq_along(penalties)) {
      penalty <- penalties[[penalty_index]]
      penalty <- as.matrix(penalty)
      storage.mode(penalty) <- "double"
      dimnames(penalty) <- NULL
      if (!identical(dim(penalty),
                     c(length(coefficient_index), length(coefficient_index)))) {
        stop("mgcv penalty block does not match coefficient range",
             call. = FALSE)
      }
      full <- matrix(0, coefficient_count, coefficient_count)
      full[coefficient_index, coefficient_index] <- penalty
      diagnostics <- fastkpc_full_cuda_census_svd_diagnostics(
        penalty, expected_rank = ncol(penalty)
      )
      blocks[[length(blocks) + 1L]] <- penalty
      embedded[[length(embedded) + 1L]] <- full
      offsets <- c(offsets, coefficient_index[[1L]])
      ranks <- c(ranks, diagnostics$rank)
      dimensions <- c(dimensions, paste(dim(penalty), collapse = "x"))
      sp_indices <- c(sp_indices, smooth_sp_indices[[penalty_index]])
    }
  }
  if (length(sp_indices) > 0L) {
    order_id <- order(sp_indices, method = "radix")
    if (!identical(sort(as.integer(sp_indices), method = "radix"),
                   seq_along(sp_indices))) {
      stop("mgcv penalty indices are not canonical contiguous order",
           call. = FALSE)
    }
    blocks <- blocks[order_id]
    embedded <- embedded[order_id]
    offsets <- offsets[order_id]
    ranks <- ranks[order_id]
    dimensions <- dimensions[order_id]
    sp_indices <- sp_indices[order_id]
  }
  list(
    blocks = blocks,
    embedded = embedded,
    offsets = as.integer(offsets),
    ranks = as.integer(ranks),
    dimensions = dimensions,
    sp_indices = as.integer(sp_indices),
    hashes = vapply(
      blocks, fastkpc_full_cuda_census_metadata_hash, character(1L)
    ),
    smooth_classes = smooth_classes,
    basis_dimensions = as.integer(basis_dimensions)
  )
}

fastkpc_full_cuda_census_constraint_matrix <- function(fit,
                                                        coefficient_count) {
  if (!is.null(fit$C) && length(fit$C) > 0L) {
    constraint <- as.matrix(fit$C)
    if (ncol(constraint) != coefficient_count) {
      stop("fit-level constraint dimension is inconsistent", call. = FALSE)
    }
    storage.mode(constraint) <- "double"
    dimnames(constraint) <- NULL
    return(constraint)
  }
  pieces <- list()
  for (smooth in fit$smooth) {
    if (is.null(smooth$C) || length(smooth$C) == 0L) next
    local <- as.matrix(smooth$C)
    coefficient_index <- seq.int(
      as.integer(smooth$first.para), as.integer(smooth$last.para)
    )
    if (ncol(local) != length(coefficient_index)) {
      stop("smooth constraint dimension is inconsistent", call. = FALSE)
    }
    full <- matrix(0, nrow(local), coefficient_count)
    full[, coefficient_index] <- local
    pieces[[length(pieces) + 1L]] <- full
  }
  if (length(pieces) == 0L) {
    return(matrix(numeric(), nrow = 0L, ncol = coefficient_count))
  }
  constraint <- do.call(rbind, pieces)
  storage.mode(constraint) <- "double"
  dimnames(constraint) <- NULL
  constraint
}

fastkpc_full_cuda_census_fit_policy <- function(value, neutral, label) {
  if (is.null(value) || length(value) == 0L || all(value == neutral)) {
    return("none")
  }
  paste0(label, ":", fastkpc_full_cuda_census_metadata_hash(value))
}

fastkpc_full_cuda_census_setup_components <- function(
    fit, request_row, data) {
  request_row <- fastkpc_full_cuda_census_request_row(request_row)
  data <- as.matrix(data)
  target <- as.integer(request_row$target[[1L]])
  S <- fastkpc_full_cuda_census_parse_s(request_row$S_key[[1L]])
  if (target < 1L || target > ncol(data) || length(S) == 0L ||
      any(S < 1L | S > ncol(data))) {
    stop("residual request indexes are outside canonical data",
         call. = FALSE)
  }
  model_matrix <- as.matrix(stats::predict(fit, type = "lpmatrix"))
  storage.mode(model_matrix) <- "double"
  dimnames(model_matrix) <- NULL
  coefficient_count <- ncol(model_matrix)
  model_diagnostics <- fastkpc_full_cuda_census_svd_diagnostics(
    model_matrix, expected_rank = coefficient_count
  )
  conditioning_data <- data[, S, drop = FALSE]
  conditioning_diagnostics <- fastkpc_full_cuda_census_svd_diagnostics(
    conditioning_data, expected_rank = ncol(conditioning_data)
  )
  near_constant_conditioning <- vapply(
    seq_len(ncol(conditioning_data)),
    function(index) fastkpc_full_cuda_census_near_constant(
      conditioning_data[, index]
    )$near_constant,
    logical(1L)
  )

  penalties <- fastkpc_full_cuda_census_penalty_components(
    fit, coefficient_count
  )
  P_unit <- matrix(0, coefficient_count, coefficient_count)
  if (length(penalties$embedded) > 0L) {
    for (penalty in penalties$embedded) P_unit <- P_unit + penalty
  }
  constraint <- fastkpc_full_cuda_census_constraint_matrix(
    fit, coefficient_count
  )
  constraint_diagnostics <- fastkpc_full_cuda_census_svd_diagnostics(
    constraint, expected_rank = min(dim(constraint))
  )
  Z <- fastkpc_full_cuda_census_right_nullspace(constraint)
  projected_penalty <- if (ncol(Z) == 0L) {
    matrix(numeric(), nrow = 0L, ncol = 0L)
  } else {
    crossprod(Z, P_unit %*% Z)
  }
  projected_penalty_diagnostics <- fastkpc_full_cuda_census_svd_diagnostics(
    projected_penalty, expected_rank = ncol(projected_penalty)
  )
  penalty_nullity <- ncol(Z) - projected_penalty_diagnostics$rank

  H_present <- !is.null(fit$H) && length(fit$H) > 0L
  H <- if (H_present) as.matrix(fit$H) else
    matrix(0, coefficient_count, coefficient_count)
  if (!identical(dim(H), c(coefficient_count, coefficient_count))) {
    stop("mgcv H dimension is inconsistent", call. = FALSE)
  }
  storage.mode(H) <- "double"
  dimnames(H) <- NULL
  weights <- if (!is.null(fit$prior.weights)) {
    as.numeric(fit$prior.weights)
  } else {
    as.numeric(fit$weights)
  }
  offset <- if (is.null(fit$offset)) numeric() else as.numeric(fit$offset)
  weights_policy <- fastkpc_full_cuda_census_fit_policy(
    weights, 1, "nonunit"
  )
  offset_policy <- fastkpc_full_cuda_census_fit_policy(
    offset, 0, "nonzero"
  )

  rank_metadata <- list(
    model_matrix_dimensions = as.integer(dim(model_matrix)),
    model_matrix_rank = model_diagnostics$rank,
    penalty_block_dimensions = penalties$dimensions,
    penalty_ranks = penalties$ranks,
    penalty_nullity = as.integer(penalty_nullity),
    constraint_dimensions = as.integer(dim(constraint)),
    constraint_rank = constraint_diagnostics$rank,
    constraint_nullspace_dimension = as.integer(ncol(Z)),
    conditioning_dimensions = as.integer(dim(conditioning_data)),
    conditioning_rank = conditioning_diagnostics$rank
  )
  model_matrix_hash <- fastkpc_full_cuda_census_metadata_hash(model_matrix)
  constraint_hash <- fastkpc_full_cuda_census_metadata_hash(constraint)
  H_hash <- if (H_present) {
    fastkpc_full_cuda_census_metadata_hash(H)
  } else {
    "NONE"
  }
  rank_metadata_hash <- fastkpc_full_cuda_census_metadata_hash(rank_metadata)
  fingerprint <- fastkpc_full_cuda_census_setup_fingerprint(
    same_S_group_id = request_row$same_S_group_id[[1L]],
    n = nrow(data),
    formula_class = request_row$formula_class[[1L]],
    model_matrix_hash = model_matrix_hash,
    penalty_hashes = penalties$hashes,
    penalty_offsets = penalties$offsets,
    constraint_hash = constraint_hash,
    H_hash = H_hash,
    weights_policy = weights_policy,
    offset_policy = offset_policy,
    rank_metadata_hash = rank_metadata_hash
  )
  row <- data.frame(
    same_S_group_id = request_row$same_S_group_id[[1L]],
    S_key = request_row$S_key[[1L]],
    S_size = as.integer(request_row$S_size[[1L]]),
    formula_class = request_row$formula_class[[1L]],
    representative_residual_key_sha256 =
      request_row$residual_key_sha256[[1L]],
    formula_semantics_version = "kpcalg_regrXonS_v1",
    model_matrix_nrow = as.integer(nrow(model_matrix)),
    model_matrix_ncol = as.integer(ncol(model_matrix)),
    model_matrix_hash = model_matrix_hash,
    model_matrix_rank = model_diagnostics$rank,
    model_matrix_condition = model_diagnostics$condition,
    penalty_count = as.integer(length(penalties$blocks)),
    penalty_block_dimensions = I(list(penalties$dimensions)),
    penalty_ranks = I(list(penalties$ranks)),
    penalty_offsets = I(list(penalties$offsets)),
    penalty_hashes = I(list(penalties$hashes)),
    penalty_nullity = as.integer(penalty_nullity),
    constraint_dimensions = I(list(as.integer(dim(constraint)))),
    constraint_rank = constraint_diagnostics$rank,
    constraint_nullspace_dimension = as.integer(ncol(Z)),
    constraint_hash = constraint_hash,
    H_dimensions = I(list(if (H_present) as.integer(dim(H)) else
                            integer())),
    H_hash = H_hash,
    weights_policy = weights_policy,
    offset_policy = offset_policy,
    smooth_classes = I(list(penalties$smooth_classes)),
    basis_dimensions = I(list(penalties$basis_dimensions)),
    conditioning_rank = conditioning_diagnostics$rank,
    conditioning_condition = conditioning_diagnostics$condition,
    near_constant_conditioning_count =
      as.integer(sum(near_constant_conditioning)),
    setup_fingerprint = fingerprint$sha256,
    mgcv_version = as.character(utils::packageVersion("mgcv")),
    R_version = R.version.string,
    stringsAsFactors = FALSE
  )
  list(
    row = row,
    model_matrix = model_matrix,
    model_diagnostics = model_diagnostics,
    conditioning_diagnostics = conditioning_diagnostics,
    penalties = penalties,
    P_unit = P_unit,
    constraint = constraint,
    constraint_diagnostics = constraint_diagnostics,
    Z = Z,
    H = H,
    weights = weights,
    fingerprint_payload = fingerprint$payload
  )
}

fastkpc_full_cuda_census_setup_observation <- function(
    fit, request_row, data) {
  fastkpc_full_cuda_census_setup_components(fit, request_row, data)$row
}

fastkpc_full_cuda_census_convergence_fields <- function(fit) {
  fields <- list()
  if (!is.null(fit$converged)) {
    fields$converged <- list(source = "fit$converged", value = fit$converged)
  }
  if (!is.null(fit$outer.info)) {
    fields$outer.info <- list(source = "fit$outer.info",
                              value = fit$outer.info)
  }
  if (!is.null(fit$mgcv.conv)) {
    fields$mgcv.conv <- list(source = "fit$mgcv.conv", value = fit$mgcv.conv)
  }
  fields
}

fastkpc_full_cuda_census_nonconverged_values <- function(
    converged = NULL, outer_info = NULL, mgcv_conv = NULL) {
  flags <- logical()
  if (!is.null(converged)) flags <- c(flags, !isTRUE(converged))
  if (!is.null(mgcv_conv$fully.converged)) {
    flags <- c(flags, !isTRUE(mgcv_conv$fully.converged))
  }
  outer_convergence <- outer_info$conv
  if (!is.null(outer_convergence)) {
    outer_failed <- if (is.logical(outer_convergence)) {
      !isTRUE(outer_convergence)
    } else if (is.numeric(outer_convergence)) {
      length(outer_convergence) != 1L || !is.finite(outer_convergence) ||
        outer_convergence != 0
    } else {
      value <- tolower(trimws(as.character(outer_convergence)))
      length(value) != 1L ||
        !value %in% c("full convergence", "converged",
                      "successful convergence")
    }
    flags <- c(flags, outer_failed)
  }
  any(flags)
}

fastkpc_full_cuda_census_nonconverged <- function(fit) {
  fastkpc_full_cuda_census_nonconverged_values(
    converged = fit$converged,
    outer_info = fit$outer.info,
    mgcv_conv = fit$mgcv.conv
  )
}

fastkpc_full_cuda_census_nonconverged_fields <- function(fields) {
  if (is.null(fields) || !is.list(fields)) return(TRUE)
  value <- function(name) {
    field <- fields[[name]]
    if (is.null(field) || !is.list(field)) NULL else field$value
  }
  fastkpc_full_cuda_census_nonconverged_values(
    converged = value("converged"),
    outer_info = value("outer.info"),
    mgcv_conv = value("mgcv.conv")
  )
}

fastkpc_full_cuda_census_target_error_row <- function(request_row, error) {
  data.frame(
    residual_key_sha256 = request_row$residual_key_sha256[[1L]],
    same_S_group_id = request_row$same_S_group_id[[1L]],
    setup_fingerprint = NA_character_,
    shard_id = as.integer(request_row$shard_id[[1L]]),
    target = as.integer(request_row$target[[1L]]),
    fit_status = "error",
    fit_error = conditionMessage(error),
    fit_time_ms = NA_real_,
    formula = NA_character_,
    method = "GCV.Cp",
    optimizer = "mgcv-default",
    family = "gaussian",
    link = "identity",
    selected_sp = I(list(numeric())),
    selected_sp_names = I(list(character())),
    selected_sp_hash = NA_character_,
    GCV_Cp_score = NA_real_,
    EDF = NA_real_,
    convergence_fields = I(list(list())),
    warning_classes = I(list(list())),
    warning_messages = I(list(character())),
    coefficient_rank = NA_integer_,
    coefficient_all_finite = FALSE,
    fitted_all_finite = FALSE,
    residual_all_finite = FALSE,
    penalized_system_condition_at_selected_sp = NA_real_,
    target_sd = NA_real_,
    target_near_constant = TRUE,
    coefficient_hash = NA_character_,
    fitted_hash = NA_character_,
    residual_hash = NA_character_,
    target_fit_fingerprint = NA_character_,
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_census_value_has_nonfinite <- function(value) {
  if (is.null(value)) return(FALSE)
  if (is.numeric(value)) return(any(!is.finite(value)))
  if (is.list(value)) {
    return(any(vapply(value,
                      fastkpc_full_cuda_census_value_has_nonfinite,
                      logical(1L))))
  }
  FALSE
}

fastkpc_full_cuda_census_nonfinite_rows <- function(table, fields) {
  table <- as.data.frame(table, stringsAsFactors = FALSE)
  missing <- setdiff(fields, names(table))
  if (length(missing) > 0L) {
    stop("metadata table missing non-finite scan fields: ",
         paste(missing, collapse = ","), call. = FALSE)
  }
  if (nrow(table) == 0L) return(logical())
  vapply(seq_len(nrow(table)), function(index) {
    any(vapply(fields, function(field) {
      fastkpc_full_cuda_census_value_has_nonfinite(table[[field]][[index]])
    }, logical(1L)))
  }, logical(1L))
}

fastkpc_full_cuda_census_target_risk_row <- function(
    request_row, setup, target_row, penalized_diagnostics,
    mgcv_warning, mgcv_nonconverged, nonfinite_metadata,
    risk_config) {
  rank_deficient <- setup$model_diagnostics$bucket ==
    "rank_deficient_inf" ||
    setup$conditioning_diagnostics$bucket == "rank_deficient_inf" ||
    penalized_diagnostics$bucket == "rank_deficient_inf"
  condition <- target_row$penalized_system_condition_at_selected_sp[[1L]]
  high_condition <- !is.na(condition) &&
    (is.infinite(condition) ||
       condition >= risk_config$high_condition_threshold)
  data.frame(
    case_type = "target_key",
    residual_key_sha256 = request_row$residual_key_sha256[[1L]],
    logical_sequence_id = NA_integer_,
    same_S_group_id = request_row$same_S_group_id[[1L]],
    high_condition = high_condition,
    rank_deficient = rank_deficient,
    near_constant_target = target_row$target_near_constant[[1L]],
    near_constant_conditioner =
      setup$row$near_constant_conditioning_count[[1L]] > 0L,
    multi_penalty = setup$row$penalty_count[[1L]] > 1L,
    near_alpha = FALSE,
    mgcv_warning = mgcv_warning,
    mgcv_nonconverged = mgcv_nonconverged,
    nonfinite_metadata = nonfinite_metadata,
    condition_bucket = penalized_diagnostics$bucket,
    near_alpha_bucket = NA_character_,
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_census_fit_key <- function(data, request_row,
                                              risk_config =
                                                fastkpc_full_cuda_census_risk_config()) {
  request_row <- fastkpc_full_cuda_census_request_row(request_row)
  data <- as.matrix(data)
  tryCatch({
    target <- as.integer(request_row$target[[1L]])
    S <- fastkpc_full_cuda_census_parse_s(request_row$S_key[[1L]])
    if (target < 1L || target > ncol(data) || any(S < 1L | S > ncol(data))) {
      stop("residual request indexes are outside canonical data",
           call. = FALSE)
    }
    captured <- fastkpc_full_cuda_census_single_target_fit(
      data[, target], data[, S, drop = FALSE]
    )
    fit <- captured$value
    setup <- fastkpc_full_cuda_census_setup_components(
      fit, request_row, data
    )
    selected_sp <- fit$sp
    selected_sp_values <- as.numeric(selected_sp)
    selected_sp_names <- names(selected_sp)
    if (length(selected_sp_values) != length(setup$penalties$embedded)) {
      stop("selected sp does not match mgcv penalty order", call. = FALSE)
    }
    P_selected <- matrix(0, ncol(setup$model_matrix),
                         ncol(setup$model_matrix))
    if (length(selected_sp_values) > 0L) {
      for (index in seq_along(selected_sp_values)) {
        P_selected <- P_selected +
          selected_sp_values[[index]] * setup$penalties$embedded[[index]]
      }
    }
    weighted_model <- setup$model_matrix * setup$weights
    normal_system <- crossprod(setup$model_matrix, weighted_model) +
      P_selected + setup$H
    A <- if (ncol(setup$Z) == 0L) {
      matrix(numeric(), nrow = 0L, ncol = 0L)
    } else {
      crossprod(setup$Z, normal_system %*% setup$Z)
    }
    penalized_diagnostics <- fastkpc_full_cuda_census_svd_diagnostics(
      A, expected_rank = ncol(A)
    )
    target_risk <- fastkpc_full_cuda_census_near_constant(data[, target])
    coefficients <- as.numeric(stats::coef(fit))
    fitted <- as.numeric(stats::fitted(fit))
    residuals <- as.numeric(stats::residuals(fit))
    coefficient_all_finite <- all(is.finite(coefficients))
    fitted_all_finite <- all(is.finite(fitted))
    residual_all_finite <- all(is.finite(residuals))
    selected_sp_hash <- fastkpc_full_cuda_census_metadata_hash(
      selected_sp_values
    )
    coefficient_hash <- fastkpc_full_cuda_census_metadata_hash(coefficients)
    fitted_hash <- fastkpc_full_cuda_census_metadata_hash(fitted)
    residual_hash <- fastkpc_full_cuda_census_metadata_hash(residuals)
    convergence_fields <- fastkpc_full_cuda_census_convergence_fields(fit)
    warning_classes <- lapply(captured$warnings, `[[`, "class")
    warning_messages <- vapply(
      captured$warnings, `[[`, character(1L), "message"
    )
    GCV_Cp_score <- as.numeric(fit$gcv.ubre)
    EDF <- as.numeric(sum(fit$edf))
    fingerprint <- fastkpc_full_cuda_census_metadata_hash(list(
      "full-cuda-ci-target-fit-fingerprint-v1",
      request_row$residual_key_sha256[[1L]],
      setup$row$setup_fingerprint[[1L]],
      target,
      selected_sp_values,
      selected_sp_names,
      GCV_Cp_score,
      EDF,
      convergence_fields,
      warning_classes,
      warning_messages,
      coefficient_all_finite,
      fitted_all_finite,
      residual_all_finite,
      coefficient_hash,
      fitted_hash,
      residual_hash,
      penalized_diagnostics$condition,
      target_risk$sd
    ))
    target_row <- data.frame(
      residual_key_sha256 = request_row$residual_key_sha256[[1L]],
      same_S_group_id = request_row$same_S_group_id[[1L]],
      setup_fingerprint = setup$row$setup_fingerprint[[1L]],
      shard_id = as.integer(request_row$shard_id[[1L]]),
      target = target,
      fit_status = "success",
      fit_error = "NONE",
      fit_time_ms = as.numeric(captured$elapsed_ms),
      formula = paste(deparse(stats::formula(fit), width.cutoff = 500L),
                      collapse = " "),
      method = "GCV.Cp",
      optimizer = "mgcv-default",
      family = fit$family$family,
      link = fit$family$link,
      selected_sp = I(list(selected_sp_values)),
      selected_sp_names = I(list(selected_sp_names)),
      selected_sp_hash = selected_sp_hash,
      GCV_Cp_score = GCV_Cp_score,
      EDF = EDF,
      convergence_fields = I(list(convergence_fields)),
      warning_classes = I(list(warning_classes)),
      warning_messages = I(list(warning_messages)),
      coefficient_rank = as.integer(fit$rank),
      coefficient_all_finite = coefficient_all_finite,
      fitted_all_finite = fitted_all_finite,
      residual_all_finite = residual_all_finite,
      penalized_system_condition_at_selected_sp =
        penalized_diagnostics$condition,
      target_sd = target_risk$sd,
      target_near_constant = target_risk$near_constant,
      coefficient_hash = coefficient_hash,
      fitted_hash = fitted_hash,
      residual_hash = residual_hash,
      target_fit_fingerprint = fingerprint,
      stringsAsFactors = FALSE
    )
    nonfinite_metadata <-
      fastkpc_full_cuda_census_nonfinite_rows(
        setup$row, fastkpc_full_cuda_census_setup_metadata_fields()
      )[[1L]] ||
      fastkpc_full_cuda_census_nonfinite_rows(
        target_row, fastkpc_full_cuda_census_target_metadata_fields()
      )[[1L]] || !coefficient_all_finite || !fitted_all_finite ||
      !residual_all_finite
    risk_row <- fastkpc_full_cuda_census_target_risk_row(
      request_row = request_row,
      setup = setup,
      target_row = target_row,
      penalized_diagnostics = penalized_diagnostics,
      mgcv_warning = length(captured$warnings) > 0L,
      mgcv_nonconverged = fastkpc_full_cuda_census_nonconverged(fit),
      nonfinite_metadata = nonfinite_metadata,
      risk_config = risk_config
    )
    list(
      setup_observation = setup$row,
      target_fit = target_row,
      risk_cases = risk_row
    )
  }, error = function(error) {
    target_row <- fastkpc_full_cuda_census_target_error_row(
      request_row, error
    )
    risk_row <- data.frame(
      case_type = "target_key",
      residual_key_sha256 = request_row$residual_key_sha256[[1L]],
      logical_sequence_id = NA_integer_,
      same_S_group_id = request_row$same_S_group_id[[1L]],
      high_condition = FALSE,
      rank_deficient = FALSE,
      near_constant_target = TRUE,
      near_constant_conditioner = FALSE,
      multi_penalty = FALSE,
      near_alpha = FALSE,
      mgcv_warning = FALSE,
      mgcv_nonconverged = FALSE,
      nonfinite_metadata = TRUE,
      condition_bucket = "nonfinite_unknown",
      near_alpha_bucket = NA_character_,
      stringsAsFactors = FALSE
    )
    list(setup_observation = NULL, target_fit = target_row,
         risk_cases = risk_row)
  })
}

fastkpc_full_cuda_census_invariant_value <- function(value) {
  if (is.list(value)) {
    vapply(value, fastkpc_full_cuda_census_metadata_hash, character(1L))
  } else {
    as.character(value)
  }
}

fastkpc_full_cuda_census_compress_setups <- function(observations) {
  observations <- as.data.frame(observations, stringsAsFactors = FALSE)
  required <- c(
    "same_S_group_id", "representative_residual_key_sha256",
    "model_matrix_hash", "penalty_hashes", "constraint_hash",
    "setup_fingerprint"
  )
  missing <- setdiff(required, names(observations))
  if (nrow(observations) == 0L || length(missing) > 0L) {
    stop("same-S setup observations are incomplete", call. = FALSE)
  }
  groups <- split(seq_len(nrow(observations)),
                  observations$same_S_group_id)
  rows <- lapply(groups, function(index) {
    group <- observations[index, , drop = FALSE]
    for (field in c("model_matrix_hash", "penalty_hashes",
                    "constraint_hash", "setup_fingerprint")) {
      values <- fastkpc_full_cuda_census_invariant_value(group[[field]])
      if (length(unique(values)) != 1L) {
        stop("same-S setup invariant violation", call. = FALSE)
      }
    }
    representative <- order(
      group$representative_residual_key_sha256, method = "radix"
    )[[1L]]
    row <- group[representative, , drop = FALSE]
    row$representative_residual_key_sha256 <- min(
      group$representative_residual_key_sha256
    )
    row
  })
  result <- do.call(rbind, rows)
  result <- result[order(result$same_S_group_id, method = "radix"),
                   , drop = FALSE]
  rownames(result) <- NULL
  result
}

fastkpc_full_cuda_census_field_present <- function(column) {
  if (is.list(column)) {
    return(vapply(column, function(value) !is.null(value), logical(1L)))
  }
  !is.na(column)
}

fastkpc_full_cuda_census_field_finite <- function(column) {
  if (is.numeric(column)) return(is.finite(column))
  if (is.list(column) && all(vapply(column, is.numeric, logical(1L)))) {
    return(vapply(column, function(value) all(is.finite(value)), logical(1L)))
  }
  NULL
}

fastkpc_full_cuda_census_setup_metadata_fields <- function() {
  c(
    "same_S_group_id", "S_key", "S_size", "formula_class",
    "representative_residual_key_sha256", "formula_semantics_version",
    "model_matrix_nrow", "model_matrix_ncol", "model_matrix_hash",
    "model_matrix_rank", "model_matrix_condition", "penalty_count",
    "penalty_block_dimensions", "penalty_ranks", "penalty_offsets",
    "penalty_hashes", "penalty_nullity", "constraint_dimensions",
    "constraint_rank", "constraint_nullspace_dimension", "constraint_hash",
    "H_dimensions", "H_hash", "weights_policy", "offset_policy",
    "smooth_classes", "basis_dimensions", "conditioning_rank",
    "conditioning_condition", "near_constant_conditioning_count",
    "setup_fingerprint", "mgcv_version", "R_version"
  )
}

fastkpc_full_cuda_census_target_metadata_fields <- function() {
  c(
    "residual_key_sha256", "same_S_group_id", "setup_fingerprint",
    "shard_id", "target", "fit_status", "fit_error", "fit_time_ms", "formula",
    "method", "optimizer", "family", "link", "selected_sp",
    "selected_sp_names", "selected_sp_hash", "GCV_Cp_score", "EDF",
    "convergence_fields", "warning_classes", "warning_messages",
    "coefficient_rank", "coefficient_all_finite", "fitted_all_finite",
    "residual_all_finite", "penalized_system_condition_at_selected_sp",
    "target_sd", "target_near_constant", "coefficient_hash",
    "fitted_hash", "residual_hash", "target_fit_fingerprint"
  )
}

fastkpc_full_cuda_census_field_coverage <- function(
    same_s_setup_metadata, target_fit_metadata) {
  tables <- list(
    same_s_setup_metadata = same_s_setup_metadata,
    target_fit_metadata = target_fit_metadata
  )
  required_fields <- list(
    same_s_setup_metadata =
      fastkpc_full_cuda_census_setup_metadata_fields(),
    target_fit_metadata =
      fastkpc_full_cuda_census_target_metadata_fields()
  )
  rows <- list()
  for (table_name in names(tables)) {
    table <- as.data.frame(tables[[table_name]], stringsAsFactors = FALSE)
    missing <- setdiff(required_fields[[table_name]], names(table))
    if (length(missing) > 0L) {
      stop("metadata table missing required fields: ",
           paste(missing, collapse = ","), call. = FALSE)
    }
    for (field in names(table)) {
      present <- fastkpc_full_cuda_census_field_present(table[[field]])
      finite <- fastkpc_full_cuda_census_field_finite(table[[field]])
      rows[[length(rows) + 1L]] <- data.frame(
        table = table_name,
        field = field,
        total = as.integer(nrow(table)),
        present = as.integer(sum(present)),
        finite = if (is.null(finite)) NA_integer_ else
          as.integer(sum(finite)),
        required = field %in% required_fields[[table_name]],
        coverage_ratio = if (nrow(table) == 0L) NA_real_ else
          sum(present) / nrow(table),
        stringsAsFactors = FALSE
      )
    }
  }
  do.call(rbind, rows)
}

fastkpc_full_cuda_census_risk_cases <- function(target_risks,
                                                 logical_tests = NULL) {
  target_risks <- as.data.frame(target_risks, stringsAsFactors = FALSE)
  flags <- c(
    "high_condition", "rank_deficient", "near_constant_target",
    "near_constant_conditioner", "multi_penalty", "near_alpha",
    "mgcv_warning", "mgcv_nonconverged", "nonfinite_metadata"
  )
  missing <- setdiff(flags, names(target_risks))
  if (length(missing) > 0L) {
    stop("target risk rows are incomplete", call. = FALSE)
  }
  selected <- if (nrow(target_risks) == 0L) {
    target_risks
  } else {
    target_risks[rowSums(target_risks[, flags, drop = FALSE]) > 0L,
                 , drop = FALSE]
  }
  if (!is.null(logical_tests)) {
    logical_tests <- as.data.frame(logical_tests, stringsAsFactors = FALSE)
    near <- is.finite(logical_tests$absolute_log_distance_from_alpha) &
      logical_tests$absolute_log_distance_from_alpha <= log(2)
    if (any(near)) {
      distances <- logical_tests$absolute_log_distance_from_alpha[near]
      logical_rows <- data.frame(
        case_type = "logical_test",
        residual_key_sha256 = NA_character_,
        logical_sequence_id = logical_tests$logical_sequence_id[near],
        same_S_group_id = NA_character_,
        high_condition = FALSE,
        rank_deficient = FALSE,
        near_constant_target = FALSE,
        near_constant_conditioner = FALSE,
        multi_penalty = FALSE,
        near_alpha = TRUE,
        mgcv_warning = FALSE,
        mgcv_nonconverged = FALSE,
        nonfinite_metadata = FALSE,
        condition_bucket = NA_character_,
        near_alpha_bucket = vapply(
          distances, fastkpc_full_cuda_census_near_alpha_bucket,
          character(1L)
        ),
        stringsAsFactors = FALSE
      )
      selected <- rbind(selected, logical_rows)
    }
  }
  rownames(selected) <- NULL
  selected
}

fastkpc_full_cuda_census_validate_shard_count <- function(shard_count) {
  shard_count <- as.integer(shard_count)
  if (length(shard_count) != 1L || is.na(shard_count) || shard_count < 1L) {
    stop("shard_count must be one positive integer", call. = FALSE)
  }
  shard_count
}

fastkpc_full_cuda_census_assign_shards <- function(requests, shard_count) {
  requests <- as.data.frame(requests, stringsAsFactors = FALSE)
  shard_count <- fastkpc_full_cuda_census_validate_shard_count(shard_count)
  if (!"residual_key_sha256" %in% names(requests) || nrow(requests) == 0L) {
    stop("shard assignment requires residual keys", call. = FALSE)
  }
  if (anyNA(requests$residual_key_sha256) ||
      anyDuplicated(requests$residual_key_sha256)) {
    stop("duplicate residual key in shard assignment", call. = FALSE)
  }
  order_id <- order(requests$residual_key_sha256, method = "radix")
  requests <- requests[order_id, , drop = FALSE]
  requests$sorted_rank <- seq_len(nrow(requests))
  requests$shard_id <- (requests$sorted_rank - 1L) %% shard_count
  rownames(requests) <- NULL
  attr(requests, "shard_count") <- shard_count
  requests
}

fastkpc_full_cuda_census_key_set_payload <- function(keys) {
  keys <- as.character(keys)
  if (length(keys) == 0L) return("")
  paste0(paste(keys, collapse = "\n"), "\n")
}

fastkpc_full_cuda_census_key_set_hash <- function(keys) {
  fastkpc_full_cuda_census_hash_utf8(
    fastkpc_full_cuda_census_key_set_payload(keys)
  )
}

fastkpc_full_cuda_census_blas_thread_count <- function() {
  variables <- c(
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
    "OMP_NUM_THREADS"
  )
  values <- Sys.getenv(variables, unset = "")
  values <- suppressWarnings(as.integer(values[nzchar(values)]))
  values <- unique(values[!is.na(values) & values > 0L])
  if (length(values) == 1L) return(values[[1L]])
  if (length(values) > 1L) {
    stop("conflicting BLAS thread-count environment", call. = FALSE)
  }
  blas <- as.character(sessionInfo()$BLAS)
  if (grepl("libRblas", blas, fixed = TRUE)) return(1L)
  NA_integer_
}

fastkpc_full_cuda_census_runtime_identity <- function() {
  session <- sessionInfo()
  software <- extSoftVersion()
  software_identity <- function(name) {
    if (name %in% names(software)) as.character(software[[name]]) else
      "unreported"
  }
  list(
    source_commit = fastkpc_full_cuda_source_commit(),
    R_version = R.version.string,
    mgcv_version = if (requireNamespace("mgcv", quietly = TRUE)) {
      as.character(utils::packageVersion("mgcv"))
    } else {
      "unavailable"
    },
    BLAS_identity = paste(
      as.character(session$BLAS), software_identity("BLAS"),
      sep = "|"
    ),
    LAPACK_identity = paste(
      as.character(session$LAPACK), software_identity("LAPACK"),
      sep = "|"
    ),
    BLAS_thread_count = fastkpc_full_cuda_census_blas_thread_count()
  )
}

fastkpc_full_cuda_census_manifest_context_fields <- function() {
  c(
    "canonical_key_corpus_hash", "canonical_logical_census_hash",
    "dataset_sha256",
    "oracle_input_bundle_sha256", "source_commit", "R_version",
    "mgcv_version", "BLAS_identity", "LAPACK_identity",
    "BLAS_thread_count", "formula_semantics_version",
    "mgcv_semantics_version", "risk_threshold_config_hash",
    "metadata_schema_version"
  )
}

fastkpc_full_cuda_census_validate_manifest_context <- function(context) {
  if (!is.list(context)) {
    stop("shard context must be a list", call. = FALSE)
  }
  required <- fastkpc_full_cuda_census_manifest_context_fields()
  missing <- setdiff(required, names(context))
  if (length(missing) > 0L) {
    stop("shard context missing fields: ", paste(missing, collapse = ","),
         call. = FALSE)
  }
  text_fields <- setdiff(required, "BLAS_thread_count")
  for (field in text_fields) {
    value <- context[[field]]
    if (length(value) != 1L || is.na(value) || !nzchar(as.character(value))) {
      stop("invalid shard context field: ", field, call. = FALSE)
    }
  }
  thread_count <- as.integer(context$BLAS_thread_count)
  if (length(thread_count) != 1L || is.na(thread_count) ||
      thread_count < 1L) {
    stop("invalid shard context field: BLAS_thread_count", call. = FALSE)
  }
  for (field in c("canonical_key_corpus_hash",
                  "canonical_logical_census_hash", "dataset_sha256",
                  "oracle_input_bundle_sha256",
                  "risk_threshold_config_hash")) {
    if (!grepl("^[0-9a-f]{64}$", context[[field]])) {
      stop("invalid shard context field: ", field, call. = FALSE)
    }
  }
  if (!grepl("^[0-9a-f]{40}$", context$source_commit)) {
    stop("invalid shard context field: source_commit", call. = FALSE)
  }
  if (is.null(context$risk_config)) {
    stop("shard context missing risk_config", call. = FALSE)
  }
  invisible(TRUE)
}

fastkpc_full_cuda_census_shard_manifest <- function(
    assigned_requests, shard_id, context) {
  shard_count <- attr(assigned_requests, "shard_count", exact = TRUE)
  assigned_requests <- as.data.frame(assigned_requests,
                                     stringsAsFactors = FALSE)
  fastkpc_full_cuda_census_validate_manifest_context(context)
  required <- c("residual_key_sha256", "shard_id")
  if (length(setdiff(required, names(assigned_requests))) > 0L) {
    stop("assigned request table is incomplete", call. = FALSE)
  }
  if (is.null(shard_count)) {
    stop("assigned request table is missing explicit shard_count lineage",
         call. = FALSE)
  }
  shard_count <- fastkpc_full_cuda_census_validate_shard_count(shard_count)
  actual_corpus_hash <- fastkpc_full_cuda_census_key_set_hash(
    sort(assigned_requests$residual_key_sha256, method = "radix")
  )
  if (!identical(actual_corpus_hash,
                 context$canonical_key_corpus_hash)) {
    stop("canonical key corpus hash mismatch", call. = FALSE)
  }
  actual_risk_hash <- fastkpc_full_cuda_census_metadata_hash(
    context$risk_config
  )
  if (!identical(actual_risk_hash,
                 context$risk_threshold_config_hash)) {
    stop("risk threshold config hash mismatch", call. = FALSE)
  }
  shard_id <- as.integer(shard_id)
  if (length(shard_id) != 1L || is.na(shard_id) || shard_id < 0L ||
      shard_id >= shard_count) {
    stop("shard_id is outside assigned shard range", call. = FALSE)
  }
  keys <- assigned_requests$residual_key_sha256[
    assigned_requests$shard_id == shard_id
  ]
  list(
    canonical_key_corpus_hash = context$canonical_key_corpus_hash,
    canonical_logical_census_hash = context$canonical_logical_census_hash,
    expected_key_count_for_shard = as.integer(length(keys)),
    expected_key_hash_for_shard =
      fastkpc_full_cuda_census_key_set_hash(keys),
    dataset_sha256 = context$dataset_sha256,
    oracle_input_bundle_sha256 = context$oracle_input_bundle_sha256,
    source_commit = context$source_commit,
    R_version = context$R_version,
    mgcv_version = context$mgcv_version,
    BLAS_identity = context$BLAS_identity,
    LAPACK_identity = context$LAPACK_identity,
    BLAS_thread_count = context$BLAS_thread_count,
    formula_semantics_version = context$formula_semantics_version,
    mgcv_semantics_version = context$mgcv_semantics_version,
    risk_threshold_config_hash = context$risk_threshold_config_hash,
    metadata_schema_version = context$metadata_schema_version,
    shard_count = as.integer(shard_count),
    shard_id = shard_id
  )
}

fastkpc_full_cuda_census_manifest_equal <- function(actual, expected) {
  is.list(actual) && is.list(expected) &&
    identical(names(actual), names(expected)) &&
    identical(fastkpc_full_cuda_census_metadata_hash(actual),
              fastkpc_full_cuda_census_metadata_hash(expected))
}

fastkpc_full_cuda_census_frame_hash <- function(value) {
  value <- as.data.frame(value, stringsAsFactors = FALSE)
  fields <- names(value)
  fastkpc_full_cuda_census_metadata_hash(list(
    schema_version = "full-cuda-ci-frame-hash-v1",
    row_count = as.integer(nrow(value)),
    field_order = fields,
    columns = unname(lapply(fields, function(field) value[[field]]))
  ))
}

fastkpc_full_cuda_census_shard_metadata_hashes <- function(payload) {
  setup_hash <- fastkpc_full_cuda_census_frame_hash(
    payload$setup_observations
  )
  target_hash <- fastkpc_full_cuda_census_frame_hash(payload$target_fits)
  risk_hash <- fastkpc_full_cuda_census_frame_hash(payload$target_risks)
  list(
    setup_observations_hash = setup_hash,
    target_fits_hash = target_hash,
    target_risks_hash = risk_hash,
    payload_hash = fastkpc_full_cuda_census_metadata_hash(list(
      schema_version = "full-cuda-ci-shard-payload-hash-v1",
      manifest_hash = fastkpc_full_cuda_census_metadata_hash(payload$manifest),
      request_key_hash = fastkpc_full_cuda_census_key_set_hash(
        payload$request_keys
      ),
      setup_observations_hash = setup_hash,
      target_fits_hash = target_hash,
      target_risks_hash = risk_hash
    ))
  )
}

fastkpc_full_cuda_census_shard_paths <- function(output_dir, shard_id) {
  shard_id <- as.integer(shard_id)
  list(
    rds = file.path(output_dir, paste0("shard_", shard_id, ".rds")),
    summary_json = file.path(
      output_dir, paste0("shard_", shard_id, ".summary.json")
    )
  )
}

fastkpc_full_cuda_census_validate_shard_payload <- function(
    payload, summary, expected_manifest = NULL) {
  if (!is.list(payload) || !is.list(summary) ||
      !identical(summary$status, "complete") ||
      is.null(payload$manifest) || is.null(summary$manifest)) {
    stop("completed shard payload is invalid", call. = FALSE)
  }
  if (!fastkpc_full_cuda_census_manifest_equal(payload$manifest,
                                                summary$manifest)) {
    stop("shard manifest mismatch", call. = FALSE)
  }
  if (!is.null(expected_manifest) &&
      !fastkpc_full_cuda_census_manifest_equal(payload$manifest,
                                                expected_manifest)) {
    stop("shard manifest mismatch", call. = FALSE)
  }
  manifest_hash <- fastkpc_full_cuda_census_metadata_hash(payload$manifest)
  if (!identical(as.character(summary$manifest_hash), manifest_hash)) {
    stop("shard manifest mismatch", call. = FALSE)
  }
  keys <- as.character(payload$request_keys)
  if (anyNA(keys) || anyDuplicated(keys)) {
    stop("duplicate residual key in shard payload", call. = FALSE)
  }
  if (length(keys) != payload$manifest$expected_key_count_for_shard ||
      !identical(fastkpc_full_cuda_census_key_set_hash(keys),
                 payload$manifest$expected_key_hash_for_shard)) {
    stop("shard request key set mismatch", call. = FALSE)
  }
  summary_counts <- c(
    request_key_count = length(keys),
    setup_observation_count = nrow(as.data.frame(
      payload$setup_observations, stringsAsFactors = FALSE
    )),
    target_fit_count = nrow(as.data.frame(
      payload$target_fits, stringsAsFactors = FALSE
    )),
    target_risk_count = nrow(as.data.frame(
      payload$target_risks, stringsAsFactors = FALSE
    ))
  )
  if (length(setdiff(names(summary_counts), names(summary))) > 0L ||
      !identical(as.integer(unlist(summary[names(summary_counts)],
                                    use.names = FALSE)),
                 as.integer(unname(summary_counts))) ||
      !identical(as.character(summary$request_key_hash),
                 fastkpc_full_cuda_census_key_set_hash(keys))) {
    stop("shard summary count mismatch", call. = FALSE)
  }
  target_fits <- as.data.frame(payload$target_fits,
                               stringsAsFactors = FALSE)
  setup_observations <- as.data.frame(payload$setup_observations,
                                      stringsAsFactors = FALSE)
  target_risks <- as.data.frame(payload$target_risks,
                                stringsAsFactors = FALSE)
  if (length(keys) == 0L) {
    if (nrow(target_fits) != 0L || nrow(setup_observations) != 0L ||
        nrow(target_risks) != 0L) {
      stop("shard target key set mismatch", call. = FALSE)
    }
  } else {
    required_target <- c("residual_key_sha256", "same_S_group_id",
                         "setup_fingerprint", "shard_id", "fit_status")
    if (length(setdiff(required_target, names(target_fits))) > 0L ||
        anyNA(target_fits$residual_key_sha256) ||
        anyDuplicated(target_fits$residual_key_sha256)) {
      stop("duplicate residual key in shard target fits", call. = FALSE)
    }
    if (!identical(sort(target_fits$residual_key_sha256, method = "radix"),
                   sort(keys, method = "radix"))) {
      stop("shard target key set mismatch", call. = FALSE)
    }
    if (anyNA(target_fits$shard_id) ||
        any(as.integer(target_fits$shard_id) !=
            as.integer(payload$manifest$shard_id))) {
      stop("shard target lineage mismatch", call. = FALSE)
    }

    required_risk <- c("case_type", "residual_key_sha256",
                       "same_S_group_id")
    if (length(setdiff(required_risk, names(target_risks))) > 0L ||
        anyNA(target_risks$residual_key_sha256) ||
        anyDuplicated(target_risks$residual_key_sha256) ||
        !identical(sort(target_risks$residual_key_sha256, method = "radix"),
                   sort(keys, method = "radix"))) {
      stop("shard risk key set mismatch", call. = FALSE)
    }
    risk_index <- match(target_fits$residual_key_sha256,
                        target_risks$residual_key_sha256)
    if (anyNA(risk_index) ||
        any(target_risks$case_type[risk_index] != "target_key") ||
        !identical(as.character(target_fits$same_S_group_id),
                   as.character(target_risks$same_S_group_id[risk_index]))) {
      stop("shard target/risk lineage mismatch", call. = FALSE)
    }

    status <- as.character(target_fits$fit_status)
    if (any(!status %in% c("success", "error"))) {
      stop("shard target fit status is invalid", call. = FALSE)
    }
    successful <- status == "success"
    successful_keys <- target_fits$residual_key_sha256[successful]
    if (length(successful_keys) == 0L) {
      if (nrow(setup_observations) != 0L) {
        stop("shard setup key set mismatch", call. = FALSE)
      }
    } else {
      required_setup <- c("representative_residual_key_sha256",
                          "same_S_group_id", "setup_fingerprint")
      if (length(setdiff(required_setup, names(setup_observations))) > 0L ||
          anyNA(setup_observations$representative_residual_key_sha256) ||
          anyDuplicated(
            setup_observations$representative_residual_key_sha256
          ) ||
          !identical(sort(
            setup_observations$representative_residual_key_sha256,
            method = "radix"
          ), sort(successful_keys, method = "radix"))) {
        stop("shard setup key set mismatch", call. = FALSE)
      }
      setup_index <- match(successful_keys,
                           setup_observations$representative_residual_key_sha256)
      successful_targets <- target_fits[successful, , drop = FALSE]
      if (anyNA(setup_index) ||
          !identical(as.character(successful_targets$same_S_group_id),
                     as.character(
                       setup_observations$same_S_group_id[setup_index]
                     )) ||
          !identical(as.character(successful_targets$setup_fingerprint),
                     as.character(
                       setup_observations$setup_fingerprint[setup_index]
                     ))) {
        stop("shard target/setup lineage mismatch", call. = FALSE)
      }
    }
    if (any(!successful & !is.na(target_fits$setup_fingerprint))) {
      stop("failed shard target retained setup metadata", call. = FALSE)
    }
  }
  actual_hashes <- fastkpc_full_cuda_census_shard_metadata_hashes(payload)
  required_hashes <- names(actual_hashes)
  if (!all(required_hashes %in% names(summary)) ||
      !identical(
        as.character(unlist(summary[required_hashes], use.names = FALSE)),
        as.character(unlist(actual_hashes, use.names = FALSE))
      )) {
    stop("shard payload hash mismatch", call. = FALSE)
  }
  invisible(TRUE)
}

fastkpc_full_cuda_census_read_shard <- function(
    paths, expected_manifest = NULL) {
  fastkpc_full_cuda_require_namespace("jsonlite")
  if (!file.exists(paths$rds) || !file.exists(paths$summary_json)) {
    stop("missing shard files", call. = FALSE)
  }
  payload <- readRDS(paths$rds)
  summary <- jsonlite::read_json(paths$summary_json, simplifyVector = TRUE)
  fastkpc_full_cuda_census_validate_shard_payload(
    payload, summary, expected_manifest
  )
  list(payload = payload, summary = summary)
}

fastkpc_full_cuda_census_atomic_write_shard <- function(
    payload, summary, paths) {
  fastkpc_full_cuda_require_namespace("jsonlite")
  output_dir <- dirname(paths$rds)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  rds_tmp <- tempfile(".shard-rds.tmp-", tmpdir = output_dir)
  json_tmp <- tempfile(".shard-json.tmp-", tmpdir = output_dir)
  published_rds <- FALSE
  on.exit({
    unlink(c(rds_tmp, json_tmp), force = TRUE)
    if (published_rds && !file.exists(paths$summary_json)) {
      unlink(paths$rds, force = TRUE)
    }
  }, add = TRUE)

  saveRDS(payload, rds_tmp, version = 2)
  fastkpc_full_cuda_write_json(summary, json_tmp)
  validation_payload <- readRDS(rds_tmp)
  validation_summary <- jsonlite::read_json(json_tmp, simplifyVector = TRUE)
  fastkpc_full_cuda_census_validate_shard_payload(
    validation_payload, validation_summary, payload$manifest
  )
  unlink(c(paths$rds, paths$summary_json), force = TRUE)
  if (!file.rename(rds_tmp, paths$rds)) {
    stop("failed to atomically publish shard RDS", call. = FALSE)
  }
  published_rds <- TRUE
  if (!file.rename(json_tmp, paths$summary_json)) {
    stop("failed to atomically publish shard completion JSON",
         call. = FALSE)
  }
  published_rds <- FALSE
  invisible(paths)
}

fastkpc_full_cuda_census_bind_rows <- function(rows) {
  rows <- Filter(Negate(is.null), rows)
  if (length(rows) == 0L) return(data.frame())
  result <- do.call(rbind, rows)
  rownames(result) <- NULL
  result
}

fastkpc_full_cuda_census_empty_frame <- function(fields) {
  as.data.frame(
    setNames(lapply(fields, function(field) logical()), fields),
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_census_validate_fit_result <- function(value, request_row) {
  request_row <- fastkpc_full_cuda_census_request_row(request_row)
  if (!is.list(value) || is.null(value$target_fit) ||
      is.null(value$risk_cases)) {
    stop("fit_fun returned an incomplete shard row", call. = FALSE)
  }
  target <- as.data.frame(value$target_fit, stringsAsFactors = FALSE)
  risk <- as.data.frame(value$risk_cases, stringsAsFactors = FALSE)
  setup <- if (is.null(value$setup_observation)) {
    data.frame()
  } else {
    as.data.frame(value$setup_observation, stringsAsFactors = FALSE)
  }
  required_target <- c("residual_key_sha256", "same_S_group_id",
                       "setup_fingerprint", "shard_id", "target",
                       "fit_status")
  required_risk <- c("case_type", "residual_key_sha256", "same_S_group_id")
  if (nrow(target) != 1L ||
      length(setdiff(required_target, names(target))) > 0L ||
      !identical(as.character(target$residual_key_sha256[[1L]]),
                 as.character(request_row$residual_key_sha256[[1L]]))) {
    stop("fit_fun returned the wrong residual key", call. = FALSE)
  }
  if (!identical(as.character(target$same_S_group_id[[1L]]),
                 as.character(request_row$same_S_group_id[[1L]])) ||
      !identical(as.integer(target$shard_id[[1L]]),
                 as.integer(request_row$shard_id[[1L]])) ||
      !identical(as.integer(target$target[[1L]]),
                 as.integer(request_row$target[[1L]]))) {
    stop("fit_fun target/request lineage mismatch", call. = FALSE)
  }
  if (nrow(risk) != 1L ||
      length(setdiff(required_risk, names(risk))) > 0L ||
      !identical(as.character(risk$residual_key_sha256[[1L]]),
                 as.character(request_row$residual_key_sha256[[1L]]))) {
    stop("fit_fun returned the wrong risk residual key", call. = FALSE)
  }
  if (!identical(as.character(risk$case_type[[1L]]), "target_key") ||
      !identical(as.character(risk$same_S_group_id[[1L]]),
                 as.character(request_row$same_S_group_id[[1L]]))) {
    stop("fit_fun target/risk lineage mismatch", call. = FALSE)
  }

  status <- as.character(target$fit_status[[1L]])
  if (identical(status, "success")) {
    required_setup <- c("representative_residual_key_sha256",
                        "same_S_group_id", "setup_fingerprint")
    if (nrow(setup) != 1L ||
        length(setdiff(required_setup, names(setup))) > 0L) {
      stop("successful fit is missing its setup observation", call. = FALSE)
    }
    if (!identical(
          as.character(setup$representative_residual_key_sha256[[1L]]),
          as.character(request_row$residual_key_sha256[[1L]])) ||
        !identical(as.character(setup$same_S_group_id[[1L]]),
                   as.character(request_row$same_S_group_id[[1L]])) ||
        !identical(as.character(setup$setup_fingerprint[[1L]]),
                   as.character(target$setup_fingerprint[[1L]]))) {
      stop("fit_fun target/setup lineage mismatch", call. = FALSE)
    }
  } else if (identical(status, "error")) {
    if (nrow(setup) != 0L || !is.na(target$setup_fingerprint[[1L]])) {
      stop("failed fit retained unexpected setup metadata", call. = FALSE)
    }
  } else {
    stop("fit_fun returned an unknown fit status", call. = FALSE)
  }
  invisible(TRUE)
}

fastkpc_full_cuda_census_run_shard <- function(
    assigned_requests, shard_id, context, output_dir,
    fit_fun = fastkpc_full_cuda_census_fit_key) {
  assigned_requests <- as.data.frame(assigned_requests,
                                     stringsAsFactors = FALSE)
  required_context <- c("data", "risk_config")
  missing_context <- setdiff(required_context, names(context))
  if (length(missing_context) > 0L) {
    stop("shard execution context missing data or risk config",
         call. = FALSE)
  }
  manifest <- fastkpc_full_cuda_census_shard_manifest(
    assigned_requests, shard_id, context
  )
  paths <- fastkpc_full_cuda_census_shard_paths(output_dir, shard_id)
  if (file.exists(paths$summary_json)) {
    if (!file.exists(paths$rds)) {
      stop("missing shard RDS for completed summary", call. = FALSE)
    }
    completed <- fastkpc_full_cuda_census_read_shard(paths, manifest)
    return(list(status = "reused", paths = paths,
                payload = completed$payload))
  }
  if (file.exists(paths$rds)) unlink(paths$rds, force = TRUE)

  shard_requests <- assigned_requests[
    assigned_requests$shard_id == as.integer(shard_id), , drop = FALSE
  ]
  shard_requests <- shard_requests[
    order(shard_requests$sorted_rank, method = "radix"), , drop = FALSE
  ]
  results <- lapply(seq_len(nrow(shard_requests)), function(index) {
    value <- fit_fun(
      data = context$data,
      request_row = shard_requests[index, , drop = FALSE],
      risk_config = context$risk_config
    )
    fastkpc_full_cuda_census_validate_fit_result(
      value, shard_requests[index, , drop = FALSE]
    )
    value
  })
  payload <- list(
    manifest = manifest,
    request_keys = shard_requests$residual_key_sha256,
    setup_observations = fastkpc_full_cuda_census_bind_rows(
      lapply(results, `[[`, "setup_observation")
    ),
    target_fits = fastkpc_full_cuda_census_bind_rows(
      lapply(results, `[[`, "target_fit")
    ),
    target_risks = fastkpc_full_cuda_census_bind_rows(
      lapply(results, `[[`, "risk_cases")
    )
  )
  summary <- list(
    status = "complete",
    manifest = manifest,
    manifest_hash = fastkpc_full_cuda_census_metadata_hash(manifest),
    request_key_count = as.integer(length(payload$request_keys)),
    request_key_hash = fastkpc_full_cuda_census_key_set_hash(
      payload$request_keys
    ),
    setup_observation_count = as.integer(nrow(payload$setup_observations)),
    target_fit_count = as.integer(nrow(payload$target_fits)),
    target_risk_count = as.integer(nrow(payload$target_risks))
  )
  summary <- c(summary, fastkpc_full_cuda_census_shard_metadata_hashes(payload))
  fastkpc_full_cuda_census_atomic_write_shard(payload, summary, paths)
  list(status = "written", paths = paths, payload = payload)
}

fastkpc_full_cuda_census_merge_shards <- function(
    requests, shard_count, context, shard_dir) {
  shard_count <- fastkpc_full_cuda_census_validate_shard_count(shard_count)
  assigned <- fastkpc_full_cuda_census_assign_shards(requests, shard_count)
  expected_ids <- 0:(shard_count - 1L)
  summary_files <- list.files(
    shard_dir, pattern = "^shard_[0-9]+\\.summary\\.json$",
    full.names = TRUE
  )
  file_ids <- suppressWarnings(as.integer(sub(
    "^shard_([0-9]+)\\.summary\\.json$", "\\1", basename(summary_files)
  )))
  if (length(summary_files) != shard_count || anyNA(file_ids) ||
      !identical(sort(file_ids), expected_ids)) {
    if (length(setdiff(expected_ids, file_ids)) > 0L) {
      stop("missing shard completion summary", call. = FALSE)
    }
    stop("unexpected shard completion summary", call. = FALSE)
  }

  loaded <- lapply(expected_ids, function(shard_id) {
    paths <- fastkpc_full_cuda_census_shard_paths(shard_dir, shard_id)
    fastkpc_full_cuda_census_read_shard(paths)
  })
  declared_ids <- vapply(loaded, function(value) {
    as.integer(value$payload$manifest$shard_id)
  }, integer(1L))
  if (anyDuplicated(declared_ids)) {
    stop("duplicate shard id in completed payloads", call. = FALSE)
  }
  if (!identical(sort(declared_ids), expected_ids)) {
    stop("missing shard id in completed payloads", call. = FALSE)
  }
  for (index in seq_along(loaded)) {
    shard_id <- expected_ids[[index]]
    expected_manifest <- fastkpc_full_cuda_census_shard_manifest(
      assigned, shard_id, context
    )
    fastkpc_full_cuda_census_validate_shard_payload(
      loaded[[index]]$payload, loaded[[index]]$summary, expected_manifest
    )
  }

  target_fits <- fastkpc_full_cuda_census_bind_rows(lapply(
    loaded, function(value) value$payload$target_fits
  ))
  if (anyDuplicated(target_fits$residual_key_sha256)) {
    stop("duplicate residual key across shard target fits", call. = FALSE)
  }
  expected_keys <- assigned$residual_key_sha256
  if (!identical(sort(target_fits$residual_key_sha256, method = "radix"),
                 expected_keys)) {
    stop("merged shard target key set mismatch", call. = FALSE)
  }
  target_fits <- target_fits[
    order(target_fits$residual_key_sha256, method = "radix"), , drop = FALSE
  ]
  rownames(target_fits) <- NULL
  request_index <- match(target_fits$residual_key_sha256,
                         assigned$residual_key_sha256)
  if (anyNA(request_index) ||
      !identical(as.character(target_fits$same_S_group_id),
                 as.character(assigned$same_S_group_id[request_index])) ||
      !identical(as.integer(target_fits$target),
                 as.integer(assigned$target[request_index])) ||
      !identical(as.integer(target_fits$shard_id),
                 as.integer(assigned$shard_id[request_index]))) {
    stop("merged target/request lineage mismatch", call. = FALSE)
  }

  setup_observations <- fastkpc_full_cuda_census_bind_rows(lapply(
    loaded, function(value) value$payload$setup_observations
  ))
  successful <- target_fits$fit_status == "success"
  successful_keys <- target_fits$residual_key_sha256[successful]
  if (length(successful_keys) == 0L) {
    if (nrow(setup_observations) != 0L) {
      stop("merged setup key set mismatch", call. = FALSE)
    }
  } else {
    if (!all(c("representative_residual_key_sha256", "same_S_group_id",
               "setup_fingerprint") %in% names(setup_observations)) ||
        anyNA(setup_observations$representative_residual_key_sha256) ||
        anyDuplicated(setup_observations$representative_residual_key_sha256) ||
        !identical(sort(
          setup_observations$representative_residual_key_sha256,
          method = "radix"
        ), sort(successful_keys, method = "radix"))) {
      stop("merged setup key set mismatch", call. = FALSE)
    }
    setup_observations <- setup_observations[order(
      setup_observations$representative_residual_key_sha256,
      method = "radix"
    ), , drop = FALSE]
    rownames(setup_observations) <- NULL
    setup_index <- match(successful_keys,
                         setup_observations$representative_residual_key_sha256)
    successful_targets <- target_fits[successful, , drop = FALSE]
    if (anyNA(setup_index) ||
        !identical(as.character(successful_targets$same_S_group_id),
                   as.character(
                     setup_observations$same_S_group_id[setup_index]
                   )) ||
        !identical(as.character(successful_targets$setup_fingerprint),
                   as.character(
                     setup_observations$setup_fingerprint[setup_index]
                   ))) {
      stop("merged target/setup lineage mismatch", call. = FALSE)
    }
  }
  same_s_setups <- if (nrow(setup_observations) == 0L) {
    fastkpc_full_cuda_census_empty_frame(
      fastkpc_full_cuda_census_setup_metadata_fields()
    )
  } else {
    fastkpc_full_cuda_census_compress_setups(setup_observations)
  }
  target_risks <- fastkpc_full_cuda_census_bind_rows(lapply(
    loaded, function(value) value$payload$target_risks
  ))
  if (!all(c("case_type", "residual_key_sha256", "same_S_group_id") %in%
           names(target_risks)) ||
      anyNA(target_risks$residual_key_sha256) ||
      anyDuplicated(target_risks$residual_key_sha256) ||
      !identical(sort(target_risks$residual_key_sha256, method = "radix"),
                 expected_keys)) {
    stop("merged target risk key set mismatch", call. = FALSE)
  }
  target_risks <- target_risks[
    order(target_risks$residual_key_sha256, method = "radix"), , drop = FALSE
  ]
  rownames(target_risks) <- NULL
  risk_index <- match(target_fits$residual_key_sha256,
                      target_risks$residual_key_sha256)
  if (anyNA(risk_index) ||
      any(target_risks$case_type[risk_index] != "target_key") ||
      !identical(as.character(target_fits$same_S_group_id),
                 as.character(target_risks$same_S_group_id[risk_index]))) {
    stop("merged target/risk lineage mismatch", call. = FALSE)
  }
  risk_cases <- fastkpc_full_cuda_census_risk_cases(
    target_risks = target_risks,
    logical_tests = context$logical_tests
  )
  coverage <- fastkpc_full_cuda_census_field_coverage(
    same_s_setup_metadata = same_s_setups,
    target_fit_metadata = target_fits
  )
  coverage_complete <- !any(
    coverage$required &
      (!is.finite(coverage$coverage_ratio) |
         coverage$coverage_ratio < 1)
  )
  authenticated_metadata_hashes <- list(
    setup_observation_metadata =
      fastkpc_full_cuda_census_frame_hash(setup_observations),
    same_s_setup_metadata = fastkpc_full_cuda_census_frame_hash(same_s_setups),
    target_fit_metadata = fastkpc_full_cuda_census_frame_hash(target_fits),
    target_risk_metadata = fastkpc_full_cuda_census_frame_hash(target_risks),
    risk_cases = fastkpc_full_cuda_census_frame_hash(risk_cases)
  )
  list(
    assigned_requests = assigned,
    setup_observation_metadata = setup_observations,
    same_s_setup_metadata = same_s_setups,
    target_fit_metadata = target_fits,
    target_risk_metadata = target_risks,
    risk_cases = risk_cases,
    field_coverage = coverage,
    authenticated_metadata_hashes = authenticated_metadata_hashes,
    required_field_coverage_complete = coverage_complete
  )
}

fastkpc_full_cuda_census_metadata_schema_version <- function() {
  "full-cuda-ci-metadata-v3"
}

fastkpc_full_cuda_census_hash_schema_versions <- function() {
  list(
    logical_census = "full-cuda-ci-logical-census-v1",
    residual_key = "full-cuda-ci-residual-key-v1",
    same_s_key = "full-cuda-ci-same-s-key-v1",
    setup_fingerprint = "full-cuda-ci-same-s-setup-fingerprint-v1",
    target_fit_fingerprint = "full-cuda-ci-target-fit-fingerprint-v1",
    frame_hash = "full-cuda-ci-frame-hash-v1",
    shard_payload_hash = "full-cuda-ci-shard-payload-hash-v1",
    metadata_numeric_hash = "portable-r-serialization-v2-sha256-v1",
    metadata_object_hash = "portable-r-serialization-v2-sha256-v1"
  )
}

fastkpc_full_cuda_census_artifact_paths <- function(output_dir) {
  list(
    manifest_json = file.path(output_dir, "manifest.json"),
    summary_json = file.path(output_dir, "summary.json"),
    summary_md = file.path(output_dir, "summary.md"),
    commands_txt = file.path(output_dir, "commands.txt"),
    environment_txt = file.path(output_dir, "environment.txt"),
    oracle_input_hashes_csv = file.path(output_dir,
                                        "oracle_input_hashes.csv"),
    graph_agreement_csv = file.path(output_dir, "graph_agreement.csv"),
    sepset_agreement_csv = file.path(output_dir, "sepset_agreement.csv"),
    n_edgetests_csv = file.path(output_dir, "n_edgetests.csv"),
    deletion_trace_csv = file.path(output_dir, "deletion_trace.csv"),
    first_divergence_json = file.path(output_dir, "first_divergence.json"),
    fallbacks_csv = file.path(output_dir, "fallbacks.csv"),
    stage_timing_csv = file.path(output_dir, "stage_timing.csv"),
    raw_runs_csv = file.path(output_dir, "raw_runs.csv"),
    logical_tests_rds = file.path(output_dir, "logical_ci_tests.rds"),
    logical_tests_csv = file.path(output_dir, "logical_ci_tests.csv"),
    residual_requests_rds = file.path(output_dir, "residual_requests.rds"),
    residual_requests_csv = file.path(output_dir, "residual_requests.csv"),
    legacy_layout_parity_cases_rds = file.path(
      output_dir, "legacy_layout_parity_cases.rds"
    ),
    legacy_layout_parity_results_csv = file.path(
      output_dir, "legacy_layout_parity_results.csv"
    ),
    same_s_setup_metadata_rds = file.path(
      output_dir, "same_s_setup_metadata.rds"
    ),
    same_s_setup_metadata_csv = file.path(
      output_dir, "same_s_setup_metadata.csv"
    ),
    target_fit_metadata_rds = file.path(
      output_dir, "target_fit_metadata.rds"
    ),
    target_fit_metadata_csv = file.path(
      output_dir, "target_fit_metadata.csv"
    ),
    risk_cases_rds = file.path(output_dir, "risk_cases.rds"),
    risk_cases_csv = file.path(output_dir, "risk_cases.csv"),
    field_coverage_csv = file.path(output_dir, "field_coverage.csv"),
    counts_by_s_size_csv = file.path(output_dir, "counts_by_s_size.csv"),
    counts_by_penalty_count_csv = file.path(
      output_dir, "counts_by_penalty_count.csv"
    ),
    counts_by_model_dimension_csv = file.path(
      output_dir, "counts_by_model_dimension.csv"
    ),
    counts_by_condition_bucket_csv = file.path(
      output_dir, "counts_by_condition_bucket.csv"
    ),
    same_s_group_distribution_csv = file.path(
      output_dir, "same_s_group_distribution.csv"
    ),
    near_alpha_tests_csv = file.path(output_dir, "near_alpha_tests.csv"),
    unsupported_envelope_csv = file.path(
      output_dir, "unsupported_envelope.csv"
    ),
    shards_dir = file.path(output_dir, "shards")
  )
}

fastkpc_full_cuda_census_csv_frame <- function(value) {
  fastkpc_full_cuda_require_namespace("jsonlite")
  value <- as.data.frame(value, stringsAsFactors = FALSE)
  for (field in names(value)) {
    if (is.list(value[[field]])) {
      value[[field]] <- vapply(value[[field]], function(cell) {
        as.character(jsonlite::toJSON(
          cell, auto_unbox = TRUE, null = "null", na = "null",
          digits = NA
        ))
      }, character(1L))
    }
  }
  value
}

fastkpc_full_cuda_census_write_rds_csv <- function(value, rds_path,
                                                    csv_path) {
  saveRDS(value, rds_path, version = 2)
  utils::write.csv(
    fastkpc_full_cuda_census_csv_frame(value),
    csv_path, row.names = FALSE
  )
  invisible(list(rds = rds_path, csv = csv_path))
}

fastkpc_full_cuda_census_build_shard_context <- function(
    inputs, structural, selected_requests,
    risk_config = fastkpc_full_cuda_census_risk_config()) {
  selected_requests <- as.data.frame(selected_requests,
                                     stringsAsFactors = FALSE)
  runtime <- fastkpc_full_cuda_census_runtime_identity()
  list(
    canonical_key_corpus_hash = fastkpc_full_cuda_census_key_set_hash(
      sort(selected_requests$residual_key_sha256, method = "radix")
    ),
    canonical_logical_census_hash =
      structural$canonical_logical_census_hash,
    dataset_sha256 =
      fastkpc_full_cuda_census_input_contract()$dataset_matrix_sha256,
    oracle_input_bundle_sha256 = inputs$oracle_input_bundle_sha256,
    source_commit = runtime$source_commit,
    R_version = runtime$R_version,
    mgcv_version = runtime$mgcv_version,
    BLAS_identity = runtime$BLAS_identity,
    LAPACK_identity = runtime$LAPACK_identity,
    BLAS_thread_count = runtime$BLAS_thread_count,
    formula_semantics_version = "kpcalg_regrXonS_v1",
    mgcv_semantics_version = "legacy-mgcv-gam-default-selection-v1",
    risk_threshold_config_hash =
      fastkpc_full_cuda_census_metadata_hash(risk_config),
    metadata_schema_version =
      fastkpc_full_cuda_census_metadata_schema_version(),
    data = inputs$data,
    risk_config = risk_config,
    logical_tests = structural$logical_tests
  )
}

fastkpc_full_cuda_census_counts_by_s_size <- function(requests) {
  groups <- split(seq_len(nrow(requests)), requests$S_size)
  do.call(rbind, lapply(names(groups), function(size) {
    index <- groups[[size]]
    data.frame(
      S_size = as.integer(size),
      unique_key_count = as.integer(length(index)),
      logical_request_count = as.integer(sum(
        requests$request_multiplicity[index]
      )),
      stringsAsFactors = FALSE
    )
  }))
}

fastkpc_full_cuda_census_count_table <- function(value, field) {
  if (length(value) == 0L) {
    return(data.frame(
      field = character(), value = character(), count = integer(),
      stringsAsFactors = FALSE
    ))
  }
  counts <- table(value, useNA = "ifany")
  data.frame(
    field = field,
    value = names(counts),
    count = as.integer(counts),
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_census_condition_counts <- function(setups) {
  if (nrow(setups) == 0L) {
    return(fastkpc_full_cuda_census_count_table(
      character(), "model_matrix_condition"
    ))
  }
  buckets <- mapply(
    fastkpc_full_cuda_census_condition_bucket,
    condition = setups$model_matrix_condition,
    rank = setups$model_matrix_rank,
    expected_rank = setups$model_matrix_ncol,
    USE.NAMES = FALSE
  )
  fastkpc_full_cuda_census_count_table(buckets, "model_matrix_condition")
}

fastkpc_full_cuda_census_same_s_distribution <- function(requests) {
  groups <- split(seq_len(nrow(requests)), requests$same_S_group_id)
  rows <- lapply(names(groups), function(group_id) {
    index <- groups[[group_id]]
    data.frame(
      same_S_group_id = group_id,
      S_key = requests$S_key[index[[1L]]],
      S_size = as.integer(requests$S_size[index[[1L]]]),
      unique_target_count = as.integer(length(index)),
      logical_request_count = as.integer(sum(
        requests$request_multiplicity[index]
      )),
      stringsAsFactors = FALSE
    )
  })
  result <- do.call(rbind, rows)
  result <- result[order(result$same_S_group_id, method = "radix"),
                   , drop = FALSE]
  rownames(result) <- NULL
  result
}

fastkpc_full_cuda_census_fit_time_distribution <- function(
    group, fit_time_ms, field) {
  if (length(group) == 0L) {
    result <- data.frame(
      value = character(), key_count = integer(), total_fit_ms = numeric(),
      mean_fit_ms = numeric(), median_fit_ms = numeric(),
      p95_fit_ms = numeric(), stringsAsFactors = FALSE
    )
    names(result)[[1L]] <- field
    return(result)
  }
  groups <- split(seq_along(group), group)
  result <- do.call(rbind, lapply(names(groups), function(value) {
    index <- groups[[value]]
    timings <- as.numeric(fit_time_ms[index])
    data.frame(
      value = value,
      key_count = as.integer(length(index)),
      total_fit_ms = sum(timings),
      mean_fit_ms = mean(timings),
      median_fit_ms = stats::median(timings),
      p95_fit_ms = as.numeric(stats::quantile(
        timings, probs = 0.95, names = FALSE, type = 7
      )),
      stringsAsFactors = FALSE
    )
  }))
  names(result)[[1L]] <- field
  if (all(grepl("^[0-9]+$", result[[field]]))) {
    result[[field]] <- as.integer(result[[field]])
    result <- result[order(result[[field]]), , drop = FALSE]
  }
  rownames(result) <- NULL
  result
}

fastkpc_full_cuda_census_fit_time_distributions <- function(
    target_fits, selected_requests, setups) {
  if (nrow(target_fits) == 0L) {
    return(list(
      by_s_size = fastkpc_full_cuda_census_fit_time_distribution(
        integer(), numeric(), "S_size"
      ),
      by_penalty_count = fastkpc_full_cuda_census_fit_time_distribution(
        integer(), numeric(), "penalty_count"
      )
    ))
  }
  request_index <- match(target_fits$residual_key_sha256,
                         selected_requests$residual_key_sha256)
  setup_index <- match(target_fits$same_S_group_id,
                       setups$same_S_group_id)
  if (anyNA(request_index)) {
    stop("fit-time distribution request join is incomplete", call. = FALSE)
  }
  penalty_count <- rep("unavailable", nrow(target_fits))
  penalty_count[!is.na(setup_index)] <- as.character(
    setups$penalty_count[setup_index[!is.na(setup_index)]]
  )
  list(
    by_s_size = fastkpc_full_cuda_census_fit_time_distribution(
      selected_requests$S_size[request_index], target_fits$fit_time_ms,
      "S_size"
    ),
    by_penalty_count = fastkpc_full_cuda_census_fit_time_distribution(
      penalty_count, target_fits$fit_time_ms,
      "penalty_count"
    )
  )
}

fastkpc_full_cuda_census_near_alpha_tests <- function(logical_tests) {
  near <- is.finite(logical_tests$absolute_log_distance_from_alpha) &
    logical_tests$absolute_log_distance_from_alpha <= log(2)
  result <- logical_tests[near, c(
    "logical_sequence_id", "source_sequence_id", "level", "x", "y",
    "S_key", "reference_p_value", "alpha", "reference_decision",
    "absolute_log_distance_from_alpha"
  ), drop = FALSE]
  result$near_alpha_bucket <- vapply(
    result$absolute_log_distance_from_alpha,
    fastkpc_full_cuda_census_near_alpha_bucket, character(1L)
  )
  result
}

fastkpc_full_cuda_census_expected_target_risks <- function(
    target, setup,
    risk_config = fastkpc_full_cuda_census_risk_config()) {
  target <- as.data.frame(target, stringsAsFactors = FALSE)
  setup <- as.data.frame(setup, stringsAsFactors = FALSE)
  required_target <- c(
    "residual_key_sha256", "same_S_group_id", "fit_status",
    "warning_classes", "convergence_fields", "target_near_constant",
    "penalized_system_condition_at_selected_sp",
    "coefficient_all_finite", "fitted_all_finite", "residual_all_finite"
  )
  required_setup <- c(
    "same_S_group_id", "S_size", "model_matrix_ncol", "model_matrix_rank",
    "model_matrix_condition", "penalty_count", "conditioning_rank",
    "conditioning_condition", "near_constant_conditioning_count"
  )
  if (!all(required_target %in% names(target)) ||
      !all(required_setup %in% names(setup)) ||
      any(target$fit_status != "success")) {
    stop("target risk semantic inputs are incomplete", call. = FALSE)
  }
  setup_index <- match(target$same_S_group_id, setup$same_S_group_id)
  if (anyNA(setup_index)) {
    stop("target risk setup join is incomplete", call. = FALSE)
  }
  setup_rows <- setup[setup_index, , drop = FALSE]
  condition <- as.numeric(
    target$penalized_system_condition_at_selected_sp
  )
  penalized_bucket <- vapply(condition, function(value) {
    if (is.na(value)) return("nonfinite_unknown")
    fastkpc_full_cuda_census_condition_bucket(
      condition = value,
      rank = if (is.infinite(value)) 0L else 1L,
      expected_rank = 1L
    )
  }, character(1L))
  model_bucket <- mapply(
    fastkpc_full_cuda_census_condition_bucket,
    condition = as.numeric(setup_rows$model_matrix_condition),
    rank = as.integer(setup_rows$model_matrix_rank),
    expected_rank = as.integer(setup_rows$model_matrix_ncol),
    USE.NAMES = FALSE
  )
  conditioning_bucket <- mapply(
    fastkpc_full_cuda_census_condition_bucket,
    condition = as.numeric(setup_rows$conditioning_condition),
    rank = as.integer(setup_rows$conditioning_rank),
    expected_rank = as.integer(setup_rows$S_size),
    USE.NAMES = FALSE
  )
  target_nonfinite <- fastkpc_full_cuda_census_nonfinite_rows(
    target, fastkpc_full_cuda_census_target_metadata_fields()
  ) | !(target$coefficient_all_finite %in% TRUE) |
    !(target$fitted_all_finite %in% TRUE) |
    !(target$residual_all_finite %in% TRUE)
  setup_nonfinite <- fastkpc_full_cuda_census_nonfinite_rows(
    setup_rows, fastkpc_full_cuda_census_setup_metadata_fields()
  )
  data.frame(
    case_type = "target_key",
    residual_key_sha256 = as.character(target$residual_key_sha256),
    logical_sequence_id = rep(NA_integer_, nrow(target)),
    same_S_group_id = as.character(target$same_S_group_id),
    high_condition = !is.na(condition) &
      (is.infinite(condition) |
         condition >= as.numeric(risk_config$high_condition_threshold)),
    rank_deficient = model_bucket == "rank_deficient_inf" |
      conditioning_bucket == "rank_deficient_inf" |
      penalized_bucket == "rank_deficient_inf",
    near_constant_target = target$target_near_constant %in% TRUE,
    near_constant_conditioner =
      as.integer(setup_rows$near_constant_conditioning_count) > 0L,
    multi_penalty = as.integer(setup_rows$penalty_count) > 1L,
    near_alpha = FALSE,
    mgcv_warning = vapply(
      target$warning_classes, function(value) length(value) > 0L, logical(1L)
    ),
    mgcv_nonconverged = vapply(
      target$convergence_fields,
      fastkpc_full_cuda_census_nonconverged_fields,
      logical(1L)
    ),
    nonfinite_metadata = target_nonfinite | setup_nonfinite,
    condition_bucket = penalized_bucket,
    near_alpha_bucket = NA_character_,
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_census_row_hashes <- function(value) {
  value <- as.data.frame(value, stringsAsFactors = FALSE)
  if (nrow(value) == 0L) return(character())
  vapply(seq_len(nrow(value)), function(index) {
    fastkpc_full_cuda_census_frame_hash(value[index, , drop = FALSE])
  }, character(1L))
}

fastkpc_full_cuda_census_metadata_gate <- function(
    merged, selected_requests,
    risk_config = fastkpc_full_cuda_census_risk_config()) {
  selected_requests <- as.data.frame(selected_requests,
                                     stringsAsFactors = FALSE)
  selected_requests <- selected_requests[order(
    selected_requests$residual_key_sha256, method = "radix"
  ), , drop = FALSE]
  rownames(selected_requests) <- NULL
  selected_keys <- selected_requests$residual_key_sha256
  target <- merged$target_fit_metadata
  setup <- merged$same_s_setup_metadata
  setup_observations <- merged$setup_observation_metadata
  target_risks <- merged$target_risk_metadata
  if (is.null(setup_observations)) setup_observations <- data.frame()
  if (is.null(target_risks)) target_risks <- data.frame()
  authenticated_tables <- list(
    setup_observation_metadata = setup_observations,
    same_s_setup_metadata = setup,
    target_fit_metadata = target,
    target_risk_metadata = target_risks,
    risk_cases = merged$risk_cases
  )
  actual_authenticated_hashes <- lapply(
    authenticated_tables, fastkpc_full_cuda_census_frame_hash
  )
  expected_authenticated_hashes <- merged$authenticated_metadata_hashes
  exact_authenticated_metadata <-
    is.list(expected_authenticated_hashes) &&
    identical(names(expected_authenticated_hashes),
              names(actual_authenticated_hashes)) &&
    identical(
      as.character(unlist(expected_authenticated_hashes, use.names = FALSE)),
      as.character(unlist(actual_authenticated_hashes, use.names = FALSE))
    )
  fit_status <- as.character(target$fit_status)
  fit_error_count <- sum(is.na(fit_status) | fit_status != "success")
  exact_key_set <- identical(target$residual_key_sha256, selected_keys)
  request_index <- match(target$residual_key_sha256,
                         selected_requests$residual_key_sha256)
  request_lineage_mismatch_count <- if (anyNA(request_index)) {
    nrow(target)
  } else {
    mismatch <- as.character(target$same_S_group_id) != as.character(
      selected_requests$same_S_group_id[request_index]
    ) | as.integer(target$target) != as.integer(
      selected_requests$target[request_index]
    )
    if (anyNA(mismatch)) nrow(target) else sum(mismatch)
  }
  exact_request_lineage <- request_lineage_mismatch_count == 0L

  expected_groups <- sort(unique(selected_requests$same_S_group_id),
                          method = "radix")
  actual_groups <- sort(unique(setup$same_S_group_id), method = "radix")
  expected_group_count <- length(expected_groups)
  exact_group_count <- nrow(setup) == expected_group_count &&
    !anyNA(setup$same_S_group_id) &&
    identical(actual_groups, expected_groups)
  required_coverage <- merged$field_coverage$coverage_ratio[
    merged$field_coverage$required
  ]
  coverage_complete <- length(required_coverage) > 0L &&
    all(is.finite(required_coverage) & required_coverage == 1)

  risk_keys <- if ("residual_key_sha256" %in% names(target_risks)) {
    as.character(target_risks$residual_key_sha256)
  } else character()
  target_risk_key_mismatch_count <- length(union(
    setdiff(selected_keys, risk_keys), setdiff(risk_keys, selected_keys)
  ))
  exact_target_risk_key_set <- target_risk_key_mismatch_count == 0L &&
    !anyNA(risk_keys) && !anyDuplicated(risk_keys)
  risk_index <- match(target$residual_key_sha256, risk_keys)
  risk_lineage_mismatch_count <- if (
      anyNA(risk_index) ||
      !all(c("case_type", "same_S_group_id") %in% names(target_risks))) {
    nrow(target)
  } else {
    mismatch <- target_risks$case_type[risk_index] != "target_key" |
      as.character(target$same_S_group_id) != as.character(
        target_risks$same_S_group_id[risk_index]
      )
    if (anyNA(mismatch)) nrow(target) else sum(mismatch)
  }
  exact_target_risk_lineage <- risk_lineage_mismatch_count == 0L
  risk_semantic_mismatch_count <- if (
      anyNA(risk_index) || !exact_target_risk_key_set) {
    nrow(target)
  } else {
    expected_risks <- tryCatch(
      fastkpc_full_cuda_census_expected_target_risks(
        target = target,
        setup = setup,
        risk_config = risk_config
      ),
      error = function(error) NULL
    )
    if (is.null(expected_risks)) {
      nrow(target)
    } else {
      actual_risks <- target_risks[risk_index, names(expected_risks),
                                     drop = FALSE]
      if (!identical(names(actual_risks), names(expected_risks))) {
        nrow(target)
      } else {
        sum(fastkpc_full_cuda_census_row_hashes(actual_risks) !=
              fastkpc_full_cuda_census_row_hashes(expected_risks))
      }
    }
  }
  exact_risk_semantics <- risk_semantic_mismatch_count == 0L

  successful <- !is.na(fit_status) & fit_status == "success"
  successful_keys <- target$residual_key_sha256[successful]
  setup_observation_keys <- if (
      "representative_residual_key_sha256" %in% names(setup_observations)) {
    as.character(setup_observations$representative_residual_key_sha256)
  } else character()
  setup_observation_key_mismatch_count <- length(union(
    setdiff(successful_keys, setup_observation_keys),
    setdiff(setup_observation_keys, successful_keys)
  ))
  exact_setup_observation_key_set <-
    setup_observation_key_mismatch_count == 0L &&
    !anyNA(setup_observation_keys) && !anyDuplicated(setup_observation_keys)
  observation_index <- match(successful_keys, setup_observation_keys)
  compressed_setup_index <- match(target$same_S_group_id[successful],
                                  setup$same_S_group_id)
  target_setup_lineage_mismatch_count <- if (
      anyNA(observation_index) || anyNA(compressed_setup_index) ||
      !all(c("same_S_group_id", "setup_fingerprint") %in%
           names(setup_observations))) {
    length(successful_keys)
  } else {
    mismatch <-
      as.character(target$same_S_group_id[successful]) != as.character(
        setup_observations$same_S_group_id[observation_index]
      ) |
      as.character(target$setup_fingerprint[successful]) != as.character(
        setup_observations$setup_fingerprint[observation_index]
      ) |
      as.character(target$setup_fingerprint[successful]) != as.character(
        setup$setup_fingerprint[compressed_setup_index]
      )
    if (anyNA(mismatch)) length(successful_keys) else sum(mismatch)
  }
  exact_target_setup_lineage <-
    target_setup_lineage_mismatch_count == 0L

  warning_keys <- target$residual_key_sha256[vapply(
    target$warning_classes, function(value) length(value) > 0L, logical(1L)
  )]
  classified_warning_keys <- if (
      all(c("residual_key_sha256", "mgcv_warning") %in%
          names(target_risks))) {
    target_risks$residual_key_sha256[target_risks$mgcv_warning %in% TRUE]
  } else character()
  unclassified_warning_count <- length(setdiff(
    warning_keys, classified_warning_keys
  ))
  misclassified_warning_count <- length(setdiff(
    classified_warning_keys, warning_keys
  ))
  exact_warning_classification <- unclassified_warning_count == 0L &&
    misclassified_warning_count == 0L

  target_nonfinite <- fastkpc_full_cuda_census_nonfinite_rows(
    target, fastkpc_full_cuda_census_target_metadata_fields()
  )
  output_finite_fields <- c("coefficient_all_finite", "fitted_all_finite",
                            "residual_all_finite")
  if (!all(output_finite_fields %in% names(target))) {
    target_nonfinite[] <- TRUE
  } else {
    target_nonfinite <- target_nonfinite |
      !(target$coefficient_all_finite %in% TRUE) |
      !(target$fitted_all_finite %in% TRUE) |
      !(target$residual_all_finite %in% TRUE)
  }
  setup_nonfinite <- fastkpc_full_cuda_census_nonfinite_rows(
    setup, fastkpc_full_cuda_census_setup_metadata_fields()
  )
  nonfinite_groups <- setup$same_S_group_id[setup_nonfinite]
  nonfinite_keys <- unique(c(
    target$residual_key_sha256[target_nonfinite],
    target$residual_key_sha256[
      target$same_S_group_id %in% nonfinite_groups
    ]
  ))
  classified_nonfinite_keys <- if (
      all(c("residual_key_sha256", "nonfinite_metadata") %in%
          names(target_risks))) {
    target_risks$residual_key_sha256[
      target_risks$nonfinite_metadata %in% TRUE
    ]
  } else character()
  unclassified_nonfinite_count <- length(setdiff(
    nonfinite_keys, classified_nonfinite_keys
  ))
  misclassified_nonfinite_count <- length(setdiff(
    classified_nonfinite_keys, nonfinite_keys
  ))
  exact_nonfinite_classification <- unclassified_nonfinite_count == 0L &&
    misclassified_nonfinite_count == 0L

  pass <- fit_error_count == 0L && exact_key_set && exact_group_count &&
    exact_request_lineage && exact_target_risk_key_set &&
    exact_target_risk_lineage && exact_risk_semantics &&
    exact_setup_observation_key_set &&
    exact_target_setup_lineage && coverage_complete &&
    exact_warning_classification && exact_nonfinite_classification &&
    exact_authenticated_metadata
  list(
    pass = pass,
    fit_error_count = as.integer(fit_error_count),
    exact_key_set = exact_key_set,
    exact_request_lineage = exact_request_lineage,
    request_lineage_mismatch_count =
      as.integer(request_lineage_mismatch_count),
    expected_group_count = as.integer(expected_group_count),
    actual_group_count = as.integer(nrow(setup)),
    exact_group_count = exact_group_count,
    exact_target_risk_key_set = exact_target_risk_key_set,
    target_risk_key_mismatch_count =
      as.integer(target_risk_key_mismatch_count),
    exact_target_risk_lineage = exact_target_risk_lineage,
    risk_lineage_mismatch_count =
      as.integer(risk_lineage_mismatch_count),
    exact_risk_semantics = exact_risk_semantics,
    risk_semantic_mismatch_count =
      as.integer(risk_semantic_mismatch_count),
    exact_setup_observation_key_set = exact_setup_observation_key_set,
    setup_observation_key_mismatch_count =
      as.integer(setup_observation_key_mismatch_count),
    exact_target_setup_lineage = exact_target_setup_lineage,
    target_setup_lineage_mismatch_count =
      as.integer(target_setup_lineage_mismatch_count),
    coverage_complete = coverage_complete,
    exact_authenticated_metadata = exact_authenticated_metadata,
    exact_warning_classification = exact_warning_classification,
    unclassified_warning_count = as.integer(unclassified_warning_count),
    misclassified_warning_count = as.integer(misclassified_warning_count),
    exact_nonfinite_classification = exact_nonfinite_classification,
    unclassified_nonfinite_count =
      as.integer(unclassified_nonfinite_count),
    misclassified_nonfinite_count =
      as.integer(misclassified_nonfinite_count)
  )
}

fastkpc_full_cuda_census_write_inherited_files <- function(
    oracle_dir, paths) {
  mapping <- c(
    graph_agreement_csv = "graph_agreement.csv",
    sepset_agreement_csv = "sepset_agreement.csv",
    n_edgetests_csv = "n_edgetests.csv",
    deletion_trace_csv = "deletion_trace.csv",
    first_divergence_json = "first_divergence.json",
    fallbacks_csv = "fallbacks.csv"
  )
  for (field in names(mapping)) {
    source <- file.path(oracle_dir, mapping[[field]])
    if (!file.copy(source, paths[[field]], overwrite = TRUE)) {
      stop("failed to copy inherited Phase 0 evidence: ", source,
           call. = FALSE)
    }
  }
  invisible(TRUE)
}

fastkpc_full_cuda_census_summary_markdown <- function(summary) {
  c(
    "# Full CUDA CI Workload Census",
    "",
    paste0("- pass: ", summary$pass),
    paste0("- run_scope: ", summary$run_scope),
    paste0("- phase1_complete: ", summary$phase1_complete),
    paste0("- selected_key_count: ", summary$selected_key_count),
    paste0("- canonical_key_count: ", summary$canonical_key_count),
    paste0("- same_S_setup_count: ", summary$same_S_setup_count),
    paste0("- fit_error_count: ", summary$fit_error_count),
    paste0("- elapsed_sec: ", format(summary$elapsed_sec, digits = 10)),
    paste0("- keys_per_sec: ", format(summary$keys_per_sec, digits = 10)),
    paste0("- estimated_full_elapsed_sec: ",
           format(summary$estimated_full_elapsed_sec, digits = 10)),
    paste0("- performance_estimate_source: ",
           summary$performance_estimate_source),
    paste0("- oracle_inherited_graph_gate: ",
           summary$oracle_inherited_graph_gate),
    paste0("- new_candidate_graph_gate: ",
           summary$new_candidate_graph_gate)
  )
}

fastkpc_full_cuda_census_write_artifact <- function(
    output_dir, oracle_dir, data_path, inputs, structural,
    selected_requests, context, mode, requested_workers, actual_workers,
    shard_count, resume, parity_cases, parity_results, merged,
    stage_timing, elapsed_sec, command_lines, executed_key_count = 0L,
    written_shard_count = 0L, reused_shard_count = 0L) {
  fastkpc_full_cuda_require_namespace("jsonlite")
  artifact_started <- proc.time()[["elapsed"]]
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  paths <- fastkpc_full_cuda_census_artifact_paths(output_dir)
  dir.create(paths$shards_dir, recursive = TRUE, showWarnings = FALSE)

  utils::write.csv(inputs$oracle_input_hashes,
                   paths$oracle_input_hashes_csv, row.names = FALSE)
  fastkpc_full_cuda_census_write_inherited_files(oracle_dir, paths)
  fastkpc_full_cuda_census_write_rds_csv(
    structural$logical_tests, paths$logical_tests_rds,
    paths$logical_tests_csv
  )
  fastkpc_full_cuda_census_write_rds_csv(
    structural$residual_requests, paths$residual_requests_rds,
    paths$residual_requests_csv
  )
  parity_available <- is.list(parity_cases) && length(parity_cases) > 0L &&
    is.data.frame(parity_results) && nrow(parity_results) > 0L
  if (parity_available) {
    fastkpc_full_cuda_census_write_parity(
      parity_cases, parity_results, output_dir
    )
  } else {
    saveRDS(list(), paths$legacy_layout_parity_cases_rds, version = 2)
    utils::write.csv(data.frame(),
                     paths$legacy_layout_parity_results_csv,
                     row.names = FALSE)
  }
  metadata_available <- is.list(merged) &&
    all(c("setup_observation_metadata", "same_s_setup_metadata",
          "target_fit_metadata", "target_risk_metadata", "risk_cases",
          "field_coverage", "authenticated_metadata_hashes") %in%
        names(merged))
  if (!metadata_available) {
    merged <- list(
      setup_observation_metadata = data.frame(),
      same_s_setup_metadata = data.frame(),
      target_fit_metadata = data.frame(),
      target_risk_metadata = data.frame(),
      risk_cases = data.frame(),
      field_coverage = data.frame(
        table = character(), field = character(), total = integer(),
        present = integer(), finite = integer(), required = logical(),
        coverage_ratio = numeric(), stringsAsFactors = FALSE
      )
    )
  }
  fastkpc_full_cuda_census_write_rds_csv(
    merged$same_s_setup_metadata,
    paths$same_s_setup_metadata_rds,
    paths$same_s_setup_metadata_csv
  )
  fastkpc_full_cuda_census_write_rds_csv(
    merged$target_fit_metadata,
    paths$target_fit_metadata_rds,
    paths$target_fit_metadata_csv
  )
  fastkpc_full_cuda_census_write_rds_csv(
    merged$risk_cases, paths$risk_cases_rds, paths$risk_cases_csv
  )
  utils::write.csv(merged$field_coverage, paths$field_coverage_csv,
                   row.names = FALSE)

  counts_by_s <- fastkpc_full_cuda_census_counts_by_s_size(
    structural$residual_requests
  )
  counts_by_penalty <- fastkpc_full_cuda_census_count_table(
    merged$same_s_setup_metadata$penalty_count, "penalty_count"
  )
  model_dimension <- paste0(
    merged$same_s_setup_metadata$model_matrix_nrow, "x",
    merged$same_s_setup_metadata$model_matrix_ncol
  )
  counts_by_model <- fastkpc_full_cuda_census_count_table(
    model_dimension, "model_matrix_dimension"
  )
  counts_by_condition <- fastkpc_full_cuda_census_condition_counts(
    merged$same_s_setup_metadata
  )
  same_s_distribution <- fastkpc_full_cuda_census_same_s_distribution(
    structural$residual_requests
  )
  same_s_size_distribution <- fastkpc_full_cuda_census_count_table(
    same_s_distribution$unique_target_count, "unique_target_count"
  )
  near_alpha <- fastkpc_full_cuda_census_near_alpha_tests(
    structural$logical_tests
  )
  near_alpha_bucket_counts <- fastkpc_full_cuda_census_count_table(
    near_alpha$near_alpha_bucket, "near_alpha_bucket"
  )
  fit_time_distributions <- fastkpc_full_cuda_census_fit_time_distributions(
    merged$target_fit_metadata, selected_requests,
    merged$same_s_setup_metadata
  )
  gate <- if (metadata_available) {
    fastkpc_full_cuda_census_metadata_gate(
      merged, selected_requests, risk_config = context$risk_config
    )
  } else {
    list(
      pass = TRUE,
      fit_error_count = 0L,
      exact_key_set = NA,
      exact_request_lineage = NA,
      request_lineage_mismatch_count = 0L,
      expected_group_count = 0L,
      actual_group_count = 0L,
      exact_group_count = NA,
      exact_target_risk_key_set = NA,
      target_risk_key_mismatch_count = 0L,
      exact_target_risk_lineage = NA,
      risk_lineage_mismatch_count = 0L,
      exact_risk_semantics = NA,
      risk_semantic_mismatch_count = 0L,
      exact_setup_observation_key_set = NA,
      setup_observation_key_mismatch_count = 0L,
      exact_target_setup_lineage = NA,
      target_setup_lineage_mismatch_count = 0L,
      coverage_complete = NA,
      exact_authenticated_metadata = NA,
      exact_warning_classification = NA,
      unclassified_warning_count = 0L,
      misclassified_warning_count = 0L,
      exact_nonfinite_classification = NA,
      unclassified_nonfinite_count = 0L,
      misclassified_nonfinite_count = 0L
    )
  }
  unsupported <- data.frame(
    metric = c(
      "fit_error_count", "request_lineage_mismatch_count",
      "target_risk_key_mismatch_count", "risk_lineage_mismatch_count",
      "risk_semantic_mismatch_count",
      "setup_observation_key_mismatch_count",
      "target_setup_lineage_mismatch_count",
      "metadata_authentication_mismatch_count",
      "unclassified_warning_count", "misclassified_warning_count",
      "unclassified_nonfinite_count", "misclassified_nonfinite_count",
      "unknown_fallback_count", "approximate_backend_count"
    ),
    count = c(
      gate$fit_error_count, gate$request_lineage_mismatch_count,
      gate$target_risk_key_mismatch_count,
      gate$risk_lineage_mismatch_count,
      gate$risk_semantic_mismatch_count,
      gate$setup_observation_key_mismatch_count,
      gate$target_setup_lineage_mismatch_count,
      as.integer(!isTRUE(gate$exact_authenticated_metadata)),
      gate$unclassified_warning_count, gate$misclassified_warning_count,
      gate$unclassified_nonfinite_count,
      gate$misclassified_nonfinite_count, 0L, 0L
    ),
    supported = rep(FALSE, 14L),
    stringsAsFactors = FALSE
  )
  utils::write.csv(counts_by_s, paths$counts_by_s_size_csv,
                   row.names = FALSE)
  utils::write.csv(counts_by_penalty, paths$counts_by_penalty_count_csv,
                   row.names = FALSE)
  utils::write.csv(counts_by_model, paths$counts_by_model_dimension_csv,
                   row.names = FALSE)
  utils::write.csv(counts_by_condition,
                   paths$counts_by_condition_bucket_csv, row.names = FALSE)
  utils::write.csv(same_s_distribution,
                   paths$same_s_group_distribution_csv, row.names = FALSE)
  utils::write.csv(near_alpha, paths$near_alpha_tests_csv, row.names = FALSE)
  utils::write.csv(unsupported, paths$unsupported_envelope_csv,
                   row.names = FALSE)
  artifact_elapsed <- proc.time()[["elapsed"]] - artifact_started
  stage_timing <- rbind(
    stage_timing,
    data.frame(stage = "write_artifact", elapsed_sec = artifact_elapsed,
               stringsAsFactors = FALSE)
  )
  utils::write.csv(stage_timing, paths$stage_timing_csv, row.names = FALSE)
  elapsed_sec <- elapsed_sec + artifact_elapsed

  canonical_key_count <- nrow(structural$residual_requests)
  selected_key_count <- nrow(selected_requests)
  run_scope <- if (selected_key_count == canonical_key_count) {
    "full"
  } else {
    "scaled_prefix"
  }
  parity_pass <- parity_available && all(parity_results$pass)
  phase1_complete <- identical(mode, "metadata") && gate$pass &&
    selected_key_count == 110617L &&
    nrow(merged$same_s_setup_metadata) == 8634L
  inherited_pass <- isTRUE(structural$oracle_inherited_graph_gate) &&
    identical(structural$new_candidate_graph_gate, "NOT_APPLICABLE")
  pass <- switch(
    mode,
    structural = inherited_pass,
    parity = inherited_pass && parity_pass,
    metadata = inherited_pass && parity_pass && gate$pass,
    FALSE
  )
  dataset_file_row <- inputs$oracle_input_hashes$logical_path ==
    "dataset/cancer_RD-causalDiscoveryInput.rds"
  raw_run <- data.frame(
    mode = mode,
    run_scope = run_scope,
    selected_key_count = as.integer(selected_key_count),
    requested_workers = as.integer(requested_workers),
    actual_workers = as.integer(actual_workers),
    shard_count = as.integer(shard_count),
    executed_key_count = as.integer(executed_key_count),
    written_shard_count = as.integer(written_shard_count),
    reused_shard_count = as.integer(reused_shard_count),
    resume = isTRUE(resume),
    elapsed_sec = as.numeric(elapsed_sec),
    pass = pass,
    stringsAsFactors = FALSE
  )
  utils::write.csv(raw_run, paths$raw_runs_csv, row.names = FALSE)

  previous_summary <- NULL
  if (as.integer(executed_key_count) == 0L &&
      file.exists(paths$summary_json)) {
    previous_summary <- tryCatch(
      jsonlite::read_json(paths$summary_json, simplifyVector = TRUE),
      error = function(error) NULL
    )
    if (!is.list(previous_summary) ||
        !identical(as.character(previous_summary$mode), mode) ||
        !identical(as.integer(previous_summary$selected_key_count),
                   as.integer(selected_key_count)) ||
        !isTRUE(previous_summary$pass)) {
      previous_summary <- NULL
    }
  }
  keys_per_sec <- if (as.integer(executed_key_count) > 0L &&
                      elapsed_sec > 0) {
    as.integer(executed_key_count) / elapsed_sec
  } else if (!is.null(previous_summary)) {
    as.numeric(previous_summary$keys_per_sec)
  } else {
    NA_real_
  }
  estimated_full <- if (as.integer(executed_key_count) == 0L &&
                        !is.null(previous_summary)) {
    as.numeric(previous_summary$estimated_full_elapsed_sec)
  } else if (is.finite(keys_per_sec) && keys_per_sec > 0) {
    canonical_key_count / keys_per_sec
  } else {
    NA_real_
  }
  performance_estimate_source <- if (as.integer(executed_key_count) > 0L) {
    "current_execution"
  } else if (!is.null(previous_summary)) {
    "preserved_previous_execution"
  } else {
    "unavailable"
  }
  performance_basis_elapsed_sec <- if (
      identical(performance_estimate_source, "current_execution")) {
    elapsed_sec
  } else if (!is.null(previous_summary) &&
             !is.null(previous_summary$performance_basis_elapsed_sec)) {
    as.numeric(previous_summary$performance_basis_elapsed_sec)
  } else {
    NA_real_
  }

  manifest <- list(
    schema_version = "full-cuda-ci-workload-census-artifact-v1",
    mode = mode,
    run_scope = run_scope,
    phase1_complete = phase1_complete,
    p_floor = structural$p_floor,
    risk_config = context$risk_config,
    risk_threshold_config_hash = context$risk_threshold_config_hash,
    metadata_schema_version = context$metadata_schema_version,
    hash_schema_versions = fastkpc_full_cuda_census_hash_schema_versions(),
    dataset_matrix_sha256 = context$dataset_sha256,
    dataset_file_sha256 = inputs$oracle_input_hashes$actual_sha256[
      dataset_file_row
    ][[1L]],
    dataset_path = data_path,
    oracle_dir = oracle_dir,
    phase0_source_commit =
      fastkpc_full_cuda_census_input_contract()$phase0_source_commit,
    oracle_input_bundle_sha256 = context$oracle_input_bundle_sha256,
    canonical_logical_census_hash =
      structural$canonical_logical_census_hash,
    canonical_key_corpus_hash = structural$canonical_key_corpus_hash,
    selected_key_corpus_hash = context$canonical_key_corpus_hash,
    source_commit = context$source_commit,
    R_version = context$R_version,
    mgcv_version = context$mgcv_version,
    BLAS_identity = context$BLAS_identity,
    LAPACK_identity = context$LAPACK_identity,
    BLAS_thread_count = context$BLAS_thread_count,
    formula_semantics_version = context$formula_semantics_version,
    mgcv_semantics_version = context$mgcv_semantics_version,
    requested_workers = as.integer(requested_workers),
    actual_workers = as.integer(actual_workers),
    shard_count = as.integer(shard_count),
    executed_key_count = as.integer(executed_key_count),
    written_shard_count = as.integer(written_shard_count),
    reused_shard_count = as.integer(reused_shard_count),
    resume = isTRUE(resume),
    selected_key_count = as.integer(selected_key_count),
    canonical_key_count = as.integer(canonical_key_count),
    oracle_inherited_graph_gate =
      structural$oracle_inherited_graph_gate,
    new_candidate_graph_gate = structural$new_candidate_graph_gate,
    parity_pass = parity_pass
  )
  if (metadata_available) {
    manifest$authenticated_metadata_hashes <-
      merged$authenticated_metadata_hashes
  }
  summary <- list(
    pass = pass,
    mode = mode,
    run_scope = run_scope,
    phase1_complete = phase1_complete,
    logical_test_count = as.integer(nrow(structural$logical_tests)),
    conditional_logical_test_count = as.integer(sum(
      structural$logical_tests$S_size > 0L
    )),
    conditional_residual_request_count = as.integer(
      structural$conditional_residual_request_count
    ),
    canonical_global_unique_conditional_target_s_count = as.integer(
      nrow(structural$residual_requests)
    ),
    unique_conditional_S_count = as.integer(
      structural$unique_conditional_S_count
    ),
    same_s_setup_metadata_rows = as.integer(
      nrow(merged$same_s_setup_metadata)
    ),
    target_fit_metadata_rows = as.integer(
      nrow(merged$target_fit_metadata)
    ),
    setup_observation_metadata_rows = as.integer(
      nrow(merged$setup_observation_metadata)
    ),
    target_risk_metadata_rows = as.integer(
      nrow(merged$target_risk_metadata)
    ),
    mgcv_fit_error_count = gate$fit_error_count,
    same_s_invariant_violation_count = 0L,
    required_field_coverage = if (metadata_available) {
      min(merged$field_coverage$coverage_ratio[
        merged$field_coverage$required
      ])
    } else {
      NA_real_
    },
    legacy_layout_parity_pass = parity_pass,
    canonical_logical_census_hash =
      structural$canonical_logical_census_hash,
    canonical_key_corpus_hash = structural$canonical_key_corpus_hash,
    selected_key_count = as.integer(selected_key_count),
    canonical_key_count = as.integer(canonical_key_count),
    same_S_setup_count = as.integer(nrow(merged$same_s_setup_metadata)),
    fit_error_count = gate$fit_error_count,
    required_field_coverage_complete = gate$coverage_complete,
    exact_selected_key_set = gate$exact_key_set,
    exact_target_request_lineage = gate$exact_request_lineage,
    exact_selected_same_S_group_count = gate$exact_group_count,
    exact_target_risk_key_set = gate$exact_target_risk_key_set,
    exact_target_risk_lineage = gate$exact_target_risk_lineage,
    exact_risk_semantics = gate$exact_risk_semantics,
    risk_semantic_mismatch_count = gate$risk_semantic_mismatch_count,
    exact_setup_observation_key_set =
      gate$exact_setup_observation_key_set,
    exact_target_setup_lineage = gate$exact_target_setup_lineage,
    exact_authenticated_metadata = gate$exact_authenticated_metadata,
    exact_warning_classification = gate$exact_warning_classification,
    unclassified_warning_count = gate$unclassified_warning_count,
    misclassified_warning_count = gate$misclassified_warning_count,
    exact_nonfinite_classification = gate$exact_nonfinite_classification,
    unclassified_nonfinite_count = gate$unclassified_nonfinite_count,
    misclassified_nonfinite_count = gate$misclassified_nonfinite_count,
    parity_pass = parity_pass,
    oracle_inherited_graph_gate =
      structural$oracle_inherited_graph_gate,
    new_candidate_graph_gate = structural$new_candidate_graph_gate,
    elapsed_sec = as.numeric(elapsed_sec),
    keys_per_sec = keys_per_sec,
    estimated_full_elapsed_sec = estimated_full,
    performance_estimate_source = performance_estimate_source,
    performance_basis_elapsed_sec = performance_basis_elapsed_sec,
    requested_workers = as.integer(requested_workers),
    actual_workers = as.integer(actual_workers),
    shard_count = as.integer(shard_count),
    executed_key_count = as.integer(executed_key_count),
    written_shard_count = as.integer(written_shard_count),
    reused_shard_count = as.integer(reused_shard_count),
    counts_by_s_size = counts_by_s,
    counts_by_penalty_count = counts_by_penalty,
    counts_by_model_dimension = counts_by_model,
    counts_by_condition_bucket = counts_by_condition,
    same_s_group_size_distribution = same_s_size_distribution,
    near_alpha_bucket_counts = near_alpha_bucket_counts,
    fit_time_by_s_size = fit_time_distributions$by_s_size,
    fit_time_by_penalty_count = fit_time_distributions$by_penalty_count
  )
  writeLines(command_lines, paths$commands_txt)
  writeLines(c(
    fastkpc_full_cuda_environment_lines(),
    paste0("CENSUS_BLAS_IDENTITY=", context$BLAS_identity),
    paste0("CENSUS_LAPACK_IDENTITY=", context$LAPACK_identity),
    paste0("CENSUS_BLAS_THREAD_COUNT=", context$BLAS_thread_count)
  ), paths$environment_txt)
  writeLines(fastkpc_full_cuda_census_summary_markdown(summary),
             paths$summary_md)
  fastkpc_full_cuda_write_json(manifest, paths$manifest_json)
  fastkpc_full_cuda_write_json(summary, paths$summary_json)
  list(paths = paths, manifest = manifest, summary = summary)
}
