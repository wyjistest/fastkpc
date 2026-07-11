fail <- function(message) stop(message, call. = FALSE)

assert_true <- function(value, message) {
  if (!isTRUE(value)) fail(message)
}

assert_identical <- function(actual, expected, message) {
  if (!identical(actual, expected)) fail(message)
}

assert_error <- function(expression, pattern, message) {
  error <- tryCatch({
    force(expression)
    NULL
  }, error = identity)
  assert_true(
    inherits(error, "error") &&
      grepl(pattern, conditionMessage(error), fixed = TRUE),
    message
  )
}

find_forbidden_result_object <- function(value, path = "result") {
  if (is.function(value) || is.environment(value) ||
      typeof(value) == "externalptr" || inherits(value, "formula") ||
      inherits(value, "gam") || is.language(value)) {
    return(path)
  }
  if (!is.list(value)) return(character())
  value_names <- names(value)
  if (is.null(value_names)) value_names <- rep("", length(value))
  normalized <- tolower(gsub("[^a-z0-9]", "", value_names))
  forbidden <- normalized %in% c("g", "minimal", "formula", "gam", "smooth")
  hits <- if (any(forbidden)) {
    paste0(path, "$", value_names[forbidden])
  } else {
    character()
  }
  for (index in seq_along(value)) {
    child <- if (nzchar(value_names[[index]])) {
      paste0(path, "$", value_names[[index]])
    } else {
      paste0(path, "[[", index, "]]" )
    }
    hits <- c(hits, find_forbidden_result_object(value[[index]], child))
  }
  hits
}

source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP target retarget: mgcv is unavailable\n")
  quit(save = "no", status = 0L)
}

census_dir <- Sys.getenv(
  "FASTKPC_PHASE1_CENSUS_DIR",
  unset = "fastkpc/artifacts/full_cuda_ci/workload_census_351x48_v1"
)
data_path <- Sys.getenv(
  "FASTKPC_PHASE2_DATA_PATH",
  unset = paste0(
    "fastkpc/artifacts/kpc_tprs_real_zhu/",
    "cancer_RD-causalDiscoveryInput.rds"
  )
)
if (!dir.exists(census_dir) || !file.exists(data_path)) {
  cat("SKIP target retarget: authenticated artifact inputs are unavailable\n")
  quit(save = "no", status = 0L)
}

inputs <- fastkpc_full_cuda_prepared_s_load_inputs(
  census_dir = census_dir,
  data_path = data_path
)

select_target_retarget_cases <- function(inputs) {
  risks <- as.data.frame(inputs$target_risks, stringsAsFactors = FALSE)
  fits <- as.data.frame(
    inputs$target_fit_metadata, stringsAsFactors = FALSE
  )
  setups <- as.data.frame(
    inputs$same_s_setup_metadata, stringsAsFactors = FALSE
  )
  logical_tests <- as.data.frame(
    inputs$logical_tests, stringsAsFactors = FALSE
  )

  assert_true(
    !anyDuplicated(as.character(risks$residual_key_sha256)) &&
      !anyDuplicated(as.character(fits$residual_key_sha256)) &&
      !anyDuplicated(as.character(setups$same_S_group_id)),
    "selection metadata keys must be unique"
  )
  fit_index <- match(
    as.character(risks$residual_key_sha256),
    as.character(fits$residual_key_sha256)
  )
  setup_index <- match(
    as.character(risks$same_S_group_id),
    as.character(setups$same_S_group_id)
  )
  assert_true(
    !anyNA(fit_index) && !anyNA(setup_index),
    "selection metadata joins must be complete"
  )

  risks$target <- as.integer(fits$target[fit_index])
  risks$penalty_count <- as.integer(setups$penalty_count[setup_index])
  finite_sp <- vapply(fits$selected_sp[fit_index], function(sp) {
    is.numeric(sp) && length(sp) > 0L &&
      all(is.finite(sp)) && all(sp > 0)
  }, logical(1L))
  finite_output <-
    as.character(fits$fit_status[fit_index]) == "success" &
    fits$coefficient_all_finite[fit_index] &
    fits$fitted_all_finite[fit_index] &
    fits$residual_all_finite[fit_index] & finite_sp
  finite_output[is.na(finite_output)] <- FALSE

  ordinary <- finite_output &
    !risks$high_condition & !risks$rank_deficient &
    !risks$near_constant_target & !risks$near_constant_conditioner &
    !risks$near_alpha & !risks$mgcv_warning &
    !risks$mgcv_nonconverged & !risks$nonfinite_metadata
  ordinary[is.na(ordinary)] <- FALSE

  first_key <- function(index, label) {
    keys <- as.character(risks$residual_key_sha256[index])
    assert_true(length(keys) > 0L, paste(label, "must have a candidate"))
    sort(keys, method = "radix")[[1L]]
  }
  penalty_counts <- c(1L, 3L, 4L, 5L, 6L, 7L)
  assignments <- do.call(rbind, lapply(penalty_counts, function(count) {
    key <- first_key(
      which(ordinary & risks$penalty_count == count),
      paste("ordinary penalty", count)
    )
    data.frame(
      category = paste0("ordinary_penalty_", count),
      residual_key_sha256 = key,
      penalty_count = count,
      stringsAsFactors = FALSE
    )
  }))

  for (field in c("rank_deficient", "mgcv_nonconverged")) {
    key <- first_key(which(finite_output & risks[[field]]), field)
    risk_index <- match(key, as.character(risks$residual_key_sha256))
    assignments <- rbind(assignments, data.frame(
      category = field,
      residual_key_sha256 = key,
      penalty_count = as.integer(risks$penalty_count[[risk_index]]),
      stringsAsFactors = FALSE
    ))
  }

  eligible <- logical_tests$S_size > 0L &
    is.finite(logical_tests$reference_p_value) &
    is.finite(logical_tests$alpha) &
    is.finite(logical_tests$absolute_log_distance_from_alpha)
  eligible[is.na(eligible)] <- FALSE
  eligible_index <- which(eligible)
  assert_true(
    length(eligible_index) > 0L,
    "closest conditional alpha selection must have a candidate"
  )
  closest_order <- order(
    logical_tests$absolute_log_distance_from_alpha[eligible_index],
    as.character(logical_tests$residual_key_x[eligible_index]),
    as.character(logical_tests$residual_key_y[eligible_index]),
    method = "radix"
  )
  closest_index <- eligible_index[closest_order[[1L]]]
  closest_keys <- sort(c(
    as.character(logical_tests$residual_key_x[[closest_index]]),
    as.character(logical_tests$residual_key_y[[closest_index]])
  ), method = "radix")
  closest_key <- closest_keys[[1L]]
  closest_risk_index <- match(
    closest_key, as.character(risks$residual_key_sha256)
  )
  assert_true(
    !is.na(closest_risk_index),
    "closest conditional target must resolve to target metadata"
  )
  assignments <- rbind(assignments, data.frame(
    category = "closest_conditional_to_alpha",
    residual_key_sha256 = closest_key,
    penalty_count = as.integer(
      risks$penalty_count[[closest_risk_index]]
    ),
    stringsAsFactors = FALSE
  ))
  rownames(assignments) <- NULL

  list(
    assignments = assignments,
    selected_keys = sort(
      unique(as.character(assignments$residual_key_sha256)),
      method = "radix"
    ),
    closest_logical_test = logical_tests[closest_index, , drop = FALSE]
  )
}

selection <- select_target_retarget_cases(inputs)
selection_repeat <- select_target_retarget_cases(inputs)
assert_identical(
  selection, selection_repeat,
  "target retarget selection must be deterministic"
)

expected_assignments <- data.frame(
  category = c(
    "ordinary_penalty_1", "ordinary_penalty_3", "ordinary_penalty_4",
    "ordinary_penalty_5", "ordinary_penalty_6", "ordinary_penalty_7",
    "rank_deficient", "mgcv_nonconverged",
    "closest_conditional_to_alpha"
  ),
  residual_key_sha256 = c(
    "000488d2c33db0aed590e09ac31b28e7ec06bd691b5e1da0071833042e775075",
    "000a866181bc6813ddd399443e6f6e1f5d869ec874ff1c0f362c5ecaf0b2bec6",
    "0009bd5eb1aed567b7d5690f33c355af6b74512c95b018961ce0f7f9689468b9",
    "0050ba459d76ff24b78bfec6931f3167f48f98b6d51052c8b3e3d6d201b56407",
    "05325d3aade7e4b404c54609356bb166cb1c07c754f8e8b7807880ab4e67840d",
    "20df8c405c654d9453459a28d1d7ad11bf98253ace072f74701b8b5b42b65b36",
    "00354ddff81cd49307434189f6bba0fc009f59b6eba9e118bd34b468583cea69",
    "00275a73df144a70ca1a2059dde9cfbb9cbf907a1013e6a7c904c900cdc6a596",
    "83d35f57ab23371cf24f487b1b822266d8747d33316f42164e61f2f5934f44fe"
  ),
  penalty_count = c(1L, 3L, 4L, 5L, 6L, 7L, 5L, 1L, 3L),
  stringsAsFactors = FALSE
)
assert_identical(
  selection$assignments, expected_assignments,
  "target retarget category identities must be canonical"
)
assert_true(
  nrow(selection$assignments) == 9L &&
    length(selection$selected_keys) == 9L &&
    identical(
      selection$selected_keys,
      sort(unique(expected_assignments$residual_key_sha256), method = "radix")
    ),
  "selected target keys must be deterministically deduplicated"
)
assert_true(
  identical(
    as.integer(selection$assignments$penalty_count[seq_len(6L)]),
    c(1L, 3L, 4L, 5L, 6L, 7L)
  ),
  "ordinary cases must cover every requested penalty count"
)

rank_key <- selection$assignments$residual_key_sha256[
  selection$assignments$category == "rank_deficient"
]
nonconverged_key <- selection$assignments$residual_key_sha256[
  selection$assignments$category == "mgcv_nonconverged"
]
risk_key <- as.character(inputs$target_risks$residual_key_sha256)
assert_true(
  isTRUE(inputs$target_risks$rank_deficient[match(rank_key, risk_key)]) &&
    isTRUE(inputs$target_risks$mgcv_nonconverged[
      match(nonconverged_key, risk_key)
    ]),
  "rare-risk selections must cover rank deficiency and nonconvergence"
)
closest <- selection$closest_logical_test
assert_true(
  closest$S_size[[1L]] > 0L &&
    identical(
      closest$absolute_log_distance_from_alpha[[1L]],
      min(
        inputs$logical_tests$absolute_log_distance_from_alpha[
          inputs$logical_tests$S_size > 0L &
            is.finite(
              inputs$logical_tests$absolute_log_distance_from_alpha
            )
        ]
      )
    ) &&
    selection$assignments$residual_key_sha256[
      selection$assignments$category == "closest_conditional_to_alpha"
    ] %in% c(closest$residual_key_x[[1L]], closest$residual_key_y[[1L]]),
  "closest conditional alpha case must use the canonical closest test"
)

independent_reference_setup <- function(prepared, materialized) {
  layout <- fastkpc_full_cuda_prepared_s_layout(
    inputs$data, prepared$sorted_S
  )
  layout$data[[1L]] <- materialized$y
  fastkpc_mgcv_extract_setup(
    formula = layout$formula,
    data = layout$data,
    sp = as.numeric(materialized$row$selected_sp[[1L]]),
    target = materialized$row$target[[1L]],
    S = prepared$sorted_S
  )
}

explicit_constraint_case <- function(prepared, materialized, epsilon) {
  prepared <- unserialize(serialize(prepared, NULL, version = 2))
  materialized <- unserialize(serialize(materialized, NULL, version = 2))
  p <- ncol(prepared$X)
  assert_true(
    p >= 3L && is.numeric(epsilon) && length(epsilon) == 1L &&
      is.finite(epsilon) && epsilon >= 0,
    "explicit constraint fixture requires one finite epsilon and p >= 3"
  )
  constraint_row_1 <- c(1, numeric(p - 1L))
  constraint_row_2 <- constraint_row_1
  constraint_row_2[[2L]] <- epsilon
  constraint <- fastkpc_full_cuda_prepared_s_matrix(rbind(
    constraint_row_1, constraint_row_2
  ))
  constraint_nullspace <- fastkpc_full_cuda_prepared_s_matrix(
    fastkpc_constraint_nullspace(constraint, p)
  )
  prepared$constraint <- constraint
  prepared$constraint_mode <- "explicit"
  prepared$constraint_nullspace <- constraint_nullspace
  prepared$constraint_rank <-
    fastkpc_full_cuda_census_svd_diagnostics(
      constraint, expected_rank = nrow(constraint)
    )$rank
  prepared$constraint_nullspace_dimension <- as.integer(
    ncol(constraint_nullspace)
  )
  prepared$nullspace_gram_policy <- "explicit-nullspace-gram"
  X_weighted <- if (is.null(prepared$weights)) {
    prepared$X
  } else {
    prepared$X * sqrt(prepared$weights)
  }
  prepared$nullspace_gram_matrix <-
    fastkpc_full_cuda_prepared_s_matrix(
      crossprod(X_weighted %*% constraint_nullspace)
    )
  P_unit <- fastkpc_assemble_penalty(
    p = p,
    S = prepared$penalty_blocks,
    off = prepared$penalty_offsets,
    sp = rep(1, length(prepared$penalty_blocks)),
    H = NULL
  )
  projected_penalty <- fastkpc_full_cuda_prepared_s_matrix(
    crossprod(
      constraint_nullspace,
      P_unit %*% constraint_nullspace
    )
  )
  projected_rank <- fastkpc_full_cuda_census_svd_diagnostics(
    projected_penalty, expected_rank = ncol(projected_penalty)
  )$rank
  prepared$penalty_nullity <- as.integer(
    ncol(constraint_nullspace) - projected_rank
  )
  prepared$semantic_fingerprint <-
    fastkpc_full_cuda_prepared_s_semantic_fingerprint(prepared)
  prepared$representation_fingerprint <-
    fastkpc_full_cuda_prepared_s_representation_fingerprint(prepared)

  projected_rhs <- if (is.null(prepared$weights)) {
    as.numeric(crossprod(prepared$X, materialized$y))
  } else {
    as.numeric(crossprod(
      prepared$X, materialized$y * prepared$weights
    ))
  }
  materialized$row$projected_rhs[[1L]] <- projected_rhs
  materialized$row$nullspace_projected_rhs[[1L]] <-
    as.numeric(crossprod(constraint_nullspace, projected_rhs))
  materialized$row$target_state_fingerprint[[1L]] <-
    fastkpc_full_cuda_target_state_fingerprint(materialized$row)
  list(prepared = prepared, materialized = materialized)
}

assert_constraint_rejected_before_kernel <- function(
    prepared, materialized, epsilon, label) {
  case <- explicit_constraint_case(prepared, materialized, epsilon)
  kernel_name <- "fastkpc_mgcv_magic_kernel_fixed_sp_coefficients"
  original_kernel <- get(kernel_name, envir = .GlobalEnv, inherits = FALSE)
  kernel_reached <- FALSE
  assign(kernel_name, function(setup, sp = setup$sp, ...) {
    kernel_reached <<- TRUE
    numeric(ncol(setup$X))
  }, envir = .GlobalEnv)
  on.exit(assign(
    kernel_name, original_kernel, envir = .GlobalEnv
  ), add = TRUE)
  error <- tryCatch({
    fastkpc_mgcv_magic_fixed_sp_from_prepared(
      case$prepared, case$materialized
    )
    NULL
  }, error = identity)
  error_message <- if (inherits(error, "error")) {
    conditionMessage(error)
  } else {
    "NONE"
  }
  assert_true(
    inherits(error, "error") &&
      grepl(
        "PreparedSSetup constraint rows must be independent",
        error_message, fixed = TRUE
      ) && !kernel_reached,
    paste0(
      label, " Prepared-S constraint must fail before C_magic; ",
      "kernel_reached=", kernel_reached, "; error=", error_message
    )
  )
}

assert_redundant_constraint_rejected <- function(prepared, materialized) {
  assert_constraint_rejected_before_kernel(
    prepared, materialized, epsilon = 0, label = "redundant"
  )
}

assert_near_dependent_constraint_rejected <- function(
    prepared, materialized) {
  assert_constraint_rejected_before_kernel(
    prepared, materialized, epsilon = 1e-8, label = "near-dependent"
  )
}

expected_result_fields <- c(
  "backend_family", "mode", "solve_source", "authoritative",
  "coefficients", "fitted", "residuals", "sp",
  "prepared_s_key_sha256", "residual_key_sha256"
)
exact_parity_count <- 0L
angles <- numeric()
first_case <- NULL

for (selected_key in selection$selected_keys) {
  target_row <- inputs$target_fit_metadata[
    as.character(inputs$target_fit_metadata$residual_key_sha256) ==
      selected_key,
    , drop = FALSE
  ]
  assert_true(
    nrow(target_row) == 1L,
    "selected target key must resolve to one Phase 1 target row"
  )
  setup_row <- inputs$same_s_setup_metadata[
    as.character(inputs$same_s_setup_metadata$same_S_group_id) ==
      as.character(target_row$same_S_group_id[[1L]]),
    , drop = FALSE
  ]
  assert_true(
    nrow(setup_row) == 1L,
    "selected target key must resolve to one same-S setup row"
  )

  prepared <- fastkpc_full_cuda_build_prepared_s_setup(inputs, setup_row)
  states <- fastkpc_full_cuda_build_target_states(inputs, prepared)
  state <- states[
    as.character(states$residual_key_sha256) == selected_key,
    , drop = FALSE
  ]
  assert_true(
    nrow(state) == 1L,
    "selected target state must resolve by residual key"
  )
  materialized <- fastkpc_full_cuda_materialize_target_state(
    state_row = state,
    data = inputs$data,
    dataset_sha256 = inputs$dataset_sha256
  )
  prepared_before <- serialize(prepared, NULL, version = 2)
  target_before <- serialize(materialized, NULL, version = 2)
  solved <- fastkpc_mgcv_magic_fixed_sp_from_prepared(
    prepared_setup = prepared,
    target_state = materialized
  )

  assert_identical(
    names(solved), expected_result_fields,
    "Prepared fixed-sp result must expose only explicit fields"
  )
  assert_true(
    identical(solved$backend_family, "mgcvExtractCPU") &&
      identical(solved$mode, "prepared-s-fixed-sp-mgcv-reference") &&
      identical(solved$solve_source, "mgcv-C-magic-from-prepared-s") &&
      isTRUE(solved$authoritative),
    "Prepared fixed-sp result must expose canonical solver identity"
  )
  assert_true(
    is.numeric(solved$coefficients) &&
      length(solved$coefficients) == ncol(prepared$X) &&
      all(is.finite(solved$coefficients)) &&
      is.numeric(solved$fitted) &&
      length(solved$fitted) == nrow(prepared$X) &&
      all(is.finite(solved$fitted)) &&
      is.numeric(solved$residuals) &&
      length(solved$residuals) == nrow(prepared$X) &&
      all(is.finite(solved$residuals)),
    "Prepared fixed-sp outputs must be finite with canonical dimensions"
  )
  assert_true(
    identical(solved$sp, as.numeric(state$selected_sp[[1L]])) &&
      identical(
        solved$prepared_s_key_sha256, prepared$prepared_s_key_sha256
      ) &&
      identical(solved$residual_key_sha256, selected_key),
    "Prepared fixed-sp result must retain exact target and setup lineage"
  )
  assert_true(
    identical(serialize(prepared, NULL, version = 2), prepared_before) &&
      identical(serialize(materialized, NULL, version = 2), target_before),
    "Prepared fixed-sp solve must not mutate either input"
  )
  assert_true(
    length(find_forbidden_result_object(solved)) == 0L &&
      length(
        fastkpc_full_cuda_prepared_s_find_executable_objects(solved)
      ) == 0L,
    "Prepared fixed-sp result must not retain raw G or executable objects"
  )

  coefficient_hash_equal <- identical(
    fastkpc_full_cuda_census_metadata_hash(solved$coefficients),
    state$coefficient_hash[[1L]]
  )
  fitted_hash_equal <- identical(
    fastkpc_full_cuda_census_metadata_hash(solved$fitted),
    state$fitted_hash[[1L]]
  )
  residual_hash_equal <- identical(
    fastkpc_full_cuda_census_metadata_hash(solved$residuals),
    state$residual_hash[[1L]]
  )
  assert_true(
    coefficient_hash_equal && fitted_hash_equal && residual_hash_equal,
    "Prepared fixed-sp solve must reproduce Phase 1 hashes exactly"
  )
  exact_parity_count <- exact_parity_count + 1L

  reference <- independent_reference_setup(prepared, materialized)
  diagnostics <- fastkpc_full_cuda_compare_prepared_s_semantics(
    prepared_setup = prepared,
    reference_setup = reference,
    solved_result = solved,
    state_row = state
  )
  diagnostic_fields <- c(
    "model_matrix_hash_equal", "model_matrix_rank_equal",
    "max_column_space_principal_angle", "semantic_angle_tolerance",
    "constraint_rank_equal", "constraint_projector_max_abs_diff",
    "constraint_action_equal", "penalty_order_equal",
    "coefficient_hash_equal", "fitted_hash_equal",
    "residual_hash_equal", "exact_behavior_gate"
  )
  assert_true(
    all(diagnostic_fields %in% names(diagnostics)),
    "semantic comparator must expose required diagnostics"
  )
  assert_true(
    isTRUE(diagnostics$model_matrix_hash_equal) &&
      isTRUE(diagnostics$model_matrix_rank_equal) &&
      isTRUE(diagnostics$constraint_rank_equal) &&
      isTRUE(diagnostics$constraint_action_equal) &&
      isTRUE(diagnostics$penalty_order_equal) &&
      isTRUE(diagnostics$coefficient_hash_equal) &&
      isTRUE(diagnostics$fitted_hash_equal) &&
      isTRUE(diagnostics$residual_hash_equal) &&
      isTRUE(diagnostics$exact_behavior_gate) &&
      diagnostics$max_column_space_principal_angle <=
        diagnostics$semantic_angle_tolerance,
    "canonical Prepared-S semantic diagnostics must pass"
  )
  angles <- c(angles, diagnostics$max_column_space_principal_angle)

  if (is.null(first_case)) {
    first_case <- list(
      prepared = prepared,
      materialized = materialized,
      state = state,
      solved = solved,
      reference = reference
    )
    assert_redundant_constraint_rejected(prepared, materialized)
    assert_near_dependent_constraint_rejected(prepared, materialized)
  }
}

assert_true(
  exact_parity_count == length(selection$selected_keys),
  "every selected key must pass exact Phase 1 parity"
)

tampered_solved <- first_case$solved
tampered_solved$coefficients[[1L]] <-
  tampered_solved$coefficients[[1L]] + 1
tampered_diagnostics <- fastkpc_full_cuda_compare_prepared_s_semantics(
  prepared_setup = first_case$prepared,
  reference_setup = first_case$reference,
  solved_result = tampered_solved,
  state_row = first_case$state
)
assert_true(
  !isTRUE(tampered_diagnostics$coefficient_hash_equal) &&
    !isTRUE(tampered_diagnostics$exact_behavior_gate),
  "tampered solved output must fail the exact behavior gate"
)

refresh_state_fingerprint <- function(state_row) {
  state_row$target_state_fingerprint[[1L]] <-
    fastkpc_full_cuda_target_state_fingerprint(state_row)
  state_row
}

malformed_setup <- first_case$prepared
malformed_setup$representation_fingerprint <- NULL
assert_error(
  fastkpc_mgcv_magic_fixed_sp_from_prepared(
    malformed_setup, first_case$materialized
  ),
  "PreparedSSetup schema field mismatch",
  "adapter must reject a malformed PreparedSSetup schema"
)

reordered_setup <- first_case$prepared[rev(names(first_case$prepared))]
assert_error(
  fastkpc_mgcv_magic_fixed_sp_from_prepared(
    reordered_setup, first_case$materialized
  ),
  "PreparedSSetup schema field mismatch",
  "adapter must reject reordered PreparedSSetup fields"
)

wrong_fingerprint_setup <- first_case$prepared
wrong_fingerprint_setup$semantic_fingerprint <- strrep("0", 64L)
assert_error(
  fastkpc_mgcv_magic_fixed_sp_from_prepared(
    wrong_fingerprint_setup, first_case$materialized
  ),
  "PreparedSSetup semantic fingerprint mismatch",
  "adapter must reject a wrong PreparedSSetup fingerprint"
)

assert_error(
  fastkpc_mgcv_magic_fixed_sp_from_prepared(
    first_case$prepared, first_case$state
  ),
  "materialized TargetState",
  "adapter must reject a non-materialized TargetState"
)

wrong_y_hash <- first_case$materialized
wrong_y_hash$row$y_hash[[1L]] <- strrep("0", 64L)
wrong_y_hash$row <- refresh_state_fingerprint(wrong_y_hash$row)
assert_error(
  fastkpc_mgcv_magic_fixed_sp_from_prepared(
    first_case$prepared, wrong_y_hash
  ),
  "TargetState y hash mismatch",
  "adapter must reject a wrong materialized y hash"
)

wrong_key <- first_case$materialized
wrong_key$row$residual_key_sha256[[1L]] <- strrep("0", 64L)
wrong_key$row <- refresh_state_fingerprint(wrong_key$row)
assert_error(
  fastkpc_mgcv_magic_fixed_sp_from_prepared(
    first_case$prepared, wrong_key
  ),
  "TargetState residual key serialization mismatch",
  "adapter must reject a wrong residual key"
)

wrong_sp <- first_case$materialized
wrong_sp$row$selected_sp[[1L]][[1L]] <-
  wrong_sp$row$selected_sp[[1L]][[1L]] * 2
wrong_sp$row <- refresh_state_fingerprint(wrong_sp$row)
assert_error(
  fastkpc_mgcv_magic_fixed_sp_from_prepared(
    first_case$prepared, wrong_sp
  ),
  "TargetState selected sp hash mismatch",
  "adapter must reject a wrong selected sp"
)

nonpositive_sp <- first_case$materialized
nonpositive_sp$row$selected_sp[[1L]][[1L]] <- 0
nonpositive_sp$row$selected_sp_hash[[1L]] <-
  fastkpc_full_cuda_census_metadata_hash(
    nonpositive_sp$row$selected_sp[[1L]]
  )
nonpositive_sp$row <- refresh_state_fingerprint(nonpositive_sp$row)
assert_error(
  fastkpc_mgcv_magic_fixed_sp_from_prepared(
    first_case$prepared, nonpositive_sp
  ),
  "TargetState selected sp is invalid",
  "adapter must reject nonpositive selected sp"
)

nonfinite_y <- first_case$materialized
nonfinite_y$y[[1L]] <- Inf
assert_error(
  fastkpc_mgcv_magic_fixed_sp_from_prepared(
    first_case$prepared, nonfinite_y
  ),
  "TargetState y must be finite",
  "adapter must reject nonfinite materialized y"
)

short_y <- first_case$materialized
short_y$y <- short_y$y[-length(short_y$y)]
assert_error(
  fastkpc_mgcv_magic_fixed_sp_from_prepared(
    first_case$prepared, short_y
  ),
  "TargetState y length mismatch",
  "adapter must reject a materialized y with the wrong length"
)

cat(sprintf(
  paste0(
    "METRICS selected_categories=%d selected_keys=%d exact_parity=%d ",
    "max_angle=%.17g\n"
  ),
  nrow(selection$assignments), length(selection$selected_keys),
  exact_parity_count, max(angles)
))
cat("PASS response-free Prepared-S fixed-sp mgcv reference and diagnostics\n")
