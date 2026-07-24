if (!exists("fastkpc_full_cuda_census_logical_hash", mode = "function")) {
  source("fastkpc/R/full_cuda_ci_workload_census.R")
}

fastkpc_full_cuda_shadow_logical_contract <- function(logical_tests) {
  if (!is.data.frame(logical_tests) || nrow(logical_tests) == 0L) {
    stop("logical contract source must be a nonempty data frame",
         call. = FALSE)
  }
  list(
    schema_version = "full-cuda-ci-shadow-logical-corpus-contract-v1",
    logical_test_count = as.integer(nrow(logical_tests)),
    canonical_logical_census_hash =
      fastkpc_full_cuda_census_logical_hash(logical_tests)
  )
}

fastkpc_full_cuda_shadow_canonical_logical_contract <- function() {
  phase0_contract <- fastkpc_full_cuda_canonical_contract()
  phase1_contract <- fastkpc_full_cuda_census_input_contract()
  list(
    schema_version = "full-cuda-ci-shadow-logical-corpus-contract-v1",
    logical_test_count = as.integer(sum(phase0_contract$n_edgetests)),
    canonical_logical_census_hash =
      as.character(phase1_contract$canonical_logical_census_hash)
  )
}

fastkpc_full_cuda_shadow_validate_logical_contract <- function(
    logical_tests, expected_logical_contract) {
  required <- c(
    "schema_version", "logical_test_count",
    "canonical_logical_census_hash"
  )
  if (!is.list(expected_logical_contract) ||
      length(setdiff(required, names(expected_logical_contract))) > 0L ||
      !identical(
        expected_logical_contract$schema_version,
        "full-cuda-ci-shadow-logical-corpus-contract-v1"
      )) {
    stop("expected logical corpus identity contract is invalid",
         call. = FALSE)
  }
  expected_count <- expected_logical_contract$logical_test_count
  expected_hash <- expected_logical_contract$canonical_logical_census_hash
  if (!is.integer(expected_count) || length(expected_count) != 1L ||
      is.na(expected_count) || expected_count < 1L ||
      !is.character(expected_hash) || length(expected_hash) != 1L ||
      is.na(expected_hash) ||
      !grepl("^[0-9a-f]{64}$", expected_hash)) {
    stop("expected logical corpus identity contract is invalid",
         call. = FALSE)
  }
  actual_count <- as.integer(nrow(logical_tests))
  if (!identical(actual_count, expected_count)) {
    stop(
      "logical corpus identity count mismatch: expected ", expected_count,
      "; observed ", actual_count,
      call. = FALSE
    )
  }
  actual_hash <- fastkpc_full_cuda_census_logical_hash(logical_tests)
  if (!identical(actual_hash, expected_hash)) {
    stop(
      "logical corpus identity hash mismatch: expected ", expected_hash,
      "; observed ", actual_hash,
      call. = FALSE
    )
  }
  invisible(TRUE)
}

fastkpc_full_cuda_shadow_integer <- function(value, field, minimum = 1L) {
  if (!is.numeric(value)) {
    stop("logical_tests contains invalid ", field, call. = FALSE)
  }
  numeric_value <- suppressWarnings(as.numeric(value))
  integer_value <- suppressWarnings(as.integer(value))
  if (any(!is.finite(numeric_value)) || anyNA(integer_value) ||
      any(numeric_value != integer_value) || any(integer_value < minimum)) {
    stop("logical_tests contains invalid ", field, call. = FALSE)
  }
  integer_value
}

fastkpc_full_cuda_shadow_labels <- function(labels) {
  if (!is.character(labels) || length(labels) < 2L || anyNA(labels) ||
      any(!nzchar(labels)) || anyDuplicated(labels)) {
    stop("labels must be unique nonempty character values", call. = FALSE)
  }
  labels
}

fastkpc_full_cuda_replay_logical_ci <- function(
    logical_tests, candidate_p_value, labels,
    expected_logical_contract =
      fastkpc_full_cuda_shadow_canonical_logical_contract()) {
  if (!is.data.frame(logical_tests) || nrow(logical_tests) == 0L) {
    stop("logical_tests must be a nonempty data frame", call. = FALSE)
  }
  required <- c(
    "logical_sequence_id", "source_sequence_id", "source_task_index",
    "level", "x", "y", "S_key", "alpha", "reference_decision"
  )
  missing <- setdiff(required, names(logical_tests))
  if (length(missing) > 0L) {
    stop("logical_tests missing fields: ", paste(missing, collapse = ","),
         call. = FALSE)
  }
  labels <- fastkpc_full_cuda_shadow_labels(labels)
  row_count <- nrow(logical_tests)
  if (!is.numeric(candidate_p_value) ||
      length(candidate_p_value) != row_count) {
    stop("candidate_p_value must be numeric with one value per logical row",
         call. = FALSE)
  }
  candidate_p_value <- as.numeric(candidate_p_value)
  if (any(!is.finite(candidate_p_value))) {
    stop("candidate_p_value must contain only finite values", call. = FALSE)
  }

  logical_sequence_id <- fastkpc_full_cuda_shadow_integer(
    logical_tests$logical_sequence_id, "logical_sequence_id"
  )
  if (anyDuplicated(logical_sequence_id)) {
    stop("logical_tests contains duplicate logical_sequence_id",
         call. = FALSE)
  }
  if (!identical(logical_sequence_id, seq_len(row_count))) {
    stop("logical_tests must be in complete canonical logical_sequence_id order",
         call. = FALSE)
  }
  fastkpc_full_cuda_shadow_validate_logical_contract(
    logical_tests, expected_logical_contract
  )
  source_sequence_id <- fastkpc_full_cuda_shadow_integer(
    logical_tests$source_sequence_id, "source_sequence_id"
  )
  source_task_index <- fastkpc_full_cuda_shadow_integer(
    logical_tests$source_task_index, "source_task_index"
  )
  level <- fastkpc_full_cuda_shadow_integer(
    logical_tests$level, "level", minimum = 0L
  )
  x <- fastkpc_full_cuda_shadow_integer(logical_tests$x, "x")
  y <- fastkpc_full_cuda_shadow_integer(logical_tests$y, "y")
  node_count <- length(labels)
  if (any(x > node_count) || any(y > node_count) || any(x == y)) {
    stop("logical_tests contains invalid node indexes", call. = FALSE)
  }
  source_keys <- paste(level, source_task_index, sep = "|")
  if (anyDuplicated(source_keys)) {
    stop("logical_tests contains duplicate source task indexes", call. = FALSE)
  }
  S_key <- as.character(logical_tests$S_key)
  if (length(S_key) != row_count || anyNA(S_key)) {
    stop("logical_tests contains invalid S_key", call. = FALSE)
  }
  alpha <- suppressWarnings(as.numeric(logical_tests$alpha))
  if (length(alpha) != row_count || any(!is.finite(alpha))) {
    stop("logical_tests contains invalid alpha", call. = FALSE)
  }
  reference_decision <- as.character(logical_tests$reference_decision)
  if (length(reference_decision) != row_count || anyNA(reference_decision) ||
      any(!reference_decision %in% c("dependent", "independent"))) {
    stop("logical_tests contains invalid reference_decision", call. = FALSE)
  }

  candidate_independent <- candidate_p_value > alpha
  candidate_decision <- ifelse(
    candidate_independent, "independent", "dependent"
  )
  decision_flip <- candidate_decision != reference_decision
  deletes_edge <- rep(FALSE, row_count)

  adjacency <- matrix(
    TRUE, node_count, node_count, dimnames = list(labels, labels)
  )
  diag(adjacency) <- FALSE
  pMax <- matrix(0, node_count, node_count, dimnames = list(labels, labels))
  diag(pMax) <- 1
  sepsets <- lapply(seq_len(node_count), function(index) {
    row <- lapply(seq_len(node_count), function(column) integer())
    names(row) <- labels
    row
  })
  names(sepsets) <- labels

  for (row_index in seq_len(row_count)) {
    tested_x <- x[[row_index]]
    tested_y <- y[[row_index]]
    if (!isTRUE(adjacency[tested_x, tested_y])) next
    if (candidate_p_value[[row_index]] > pMax[tested_x, tested_y]) {
      pMax[tested_x, tested_y] <- candidate_p_value[[row_index]]
      pMax[tested_y, tested_x] <- candidate_p_value[[row_index]]
    }
    if (!candidate_independent[[row_index]]) next
    S <- fastkpc_full_cuda_parse_s_key(S_key[[row_index]])
    canonical_S_key <- fastkpc_full_cuda_s_key(S)
    if (!identical(canonical_S_key, S_key[[row_index]]) ||
        length(S) != level[[row_index]] || any(S < 1L) ||
        any(S > node_count) || any(S %in% c(tested_x, tested_y))) {
      stop("logical_tests contains invalid canonical S_key at row ",
           row_index, call. = FALSE)
    }
    adjacency[tested_x, tested_y] <- FALSE
    adjacency[tested_y, tested_x] <- FALSE
    deletes_edge[[row_index]] <- TRUE
    sepsets[[tested_x]][[tested_y]] <- S
  }

  logical_trace <- data.frame(
    logical_sequence_id = logical_sequence_id,
    source_sequence_id = source_sequence_id,
    source_task_index = source_task_index,
    level = level,
    x = x,
    y = y,
    S_key = S_key,
    p_value = candidate_p_value,
    candidate_p_value = candidate_p_value,
    alpha = alpha,
    reference_decision = reference_decision,
    candidate_independent = candidate_independent,
    candidate_decision = candidate_decision,
    decision_flip = decision_flip,
    deletes_edge = deletes_edge,
    stringsAsFactors = FALSE
  )
  tasks <- data.frame(
    logical_sequence_id = logical_sequence_id,
    canonical_test_order_id = source_sequence_id,
    source_sequence_id = source_sequence_id,
    task_index = source_task_index,
    source_task_index = source_task_index,
    level = level,
    edge_x = pmin(x, y),
    edge_y = pmax(x, y),
    x = x,
    y = y,
    S_key = S_key,
    p_candidate = candidate_p_value,
    candidate_independent = candidate_independent,
    candidate_decision = candidate_decision,
    decision_flip = decision_flip,
    native_edge_deleted = deletes_edge,
    native_edge_ignored = FALSE,
    stringsAsFactors = FALSE
  )
  n.edgetests <- as.integer(tabulate(
    level + 1L, nbins = max(level) + 1L
  ))
  skeleton <- list(
    adjacency = adjacency,
    sepsets = sepsets,
    pMax = pMax,
    n.edgetests = n.edgetests,
    tasks = tasks,
    summary = list(
      unknown_fallback_count = 0L,
      approximate_backend_count = 0L
    )
  )
  list(skeleton = skeleton, logical_trace = logical_trace)
}
