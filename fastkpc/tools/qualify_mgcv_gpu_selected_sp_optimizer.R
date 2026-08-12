source("fastkpc/R/legacy_runner.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/native.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cuda.R")

args <- commandArgs(trailingOnly = TRUE)
input_path <- if (length(args) >= 1L) args[[1L]] else {
  "/tmp/fastkpc_causal_large_parity_20260810/residual_route_bakeoff.rds"
}
output_path <- if (length(args) >= 2L) args[[2L]] else {
  "/tmp/fastkpc_causal_large_parity_20260810/selected_sp_optimizer_v2.rds"
}
requested_indices <- if (length(args) >= 3L) {
  as.integer(strsplit(args[[3L]], ",", fixed = TRUE)[[1L]])
} else {
  c(seq_len(11L), 14L)
}
fixed_sp_finalize <- identical(
  Sys.getenv("FASTKPC_SELECTED_SP_FIXED_FINALIZE", unset = "0"), "1"
)
force_stability_replay_all <- identical(
  Sys.getenv("FASTKPC_FORCE_STABILITY_REPLAY_ALL", unset = "0"), "1"
)
primary_fixed_reuse_diagnostic <- identical(
  Sys.getenv(
    "FASTKPC_SELECTED_SP_PRIMARY_FIXED_REUSE_DIAGNOSTIC", unset = "0"
  ),
  "1"
)
if (primary_fixed_reuse_diagnostic && !fixed_sp_finalize) {
  stop(
    "primary/fixed reuse diagnostic requires fixed-SP finalization",
    call. = FALSE
  )
}

if (!file.exists(input_path)) stop("source receipt is missing", call. = FALSE)
if (file.exists(output_path)) stop("output already exists", call. = FALSE)
if (anyNA(requested_indices) || any(requested_indices <= 0L)) {
  stop("witness indices must be positive integers", call. = FALSE)
}

data_paths <- c(
  `20230331145508` =
    "causalDiscovery/20230331145508/combined_data_shuffle-causalDiscoveryInput.RData",
  `20240102191119` =
    "causalDiscovery/20240102191119/GSE99254-482col-causalDiscoveryInput.RData",
  `20241215091815` =
    "causalDiscovery/20241215091815/logged-lung-exhausted-allData-causalDiscoveryInput.RData"
)

prepared_setup_from_native <- function(native_setup) {
  penalty_count <- length(native_setup$penalty_blocks)
  list(
    formula_class = "additive-smooth",
    X = unname(native_setup$X),
    gram_matrix = unname(native_setup$gram_matrix),
    penalty_blocks = setNames(
      lapply(native_setup$penalty_blocks, unname),
      paste0("penalty_", seq_len(penalty_count))
    ),
    penalty_offsets = as.integer(native_setup$penalty_offsets),
    penalty_ranks = as.integer(native_setup$penalty_ranks),
    mgcv_penalty_rank_metadata = as.integer(native_setup$penalty_ranks),
    constraint = matrix(
      numeric(), nrow = 0L, ncol = ncol(native_setup$X)
    ),
    H = NULL,
    weights = NULL
  )
}

p_value <- function(residuals) {
  fastkpc_cuda_legacy_dcov_gamma_cpp_oracle(
    residuals[, 1L], residuals[, 2L], numCol = 35L, index = 1
  )$p.value
}

source_receipt <- readRDS(input_path)
if (any(requested_indices > length(source_receipt$details))) {
  stop("witness index exceeds source receipt", call. = FALSE)
}
load_fastkpc_cuda_native()
if (!fastkpc_cuda_available()) stop("CUDA is unavailable", call. = FALSE)

loaded_data <- new.env(parent = emptyenv())
rows <- list()
details <- list()
for (source_index in requested_indices) {
  source_detail <- source_receipt$details[[source_index]]
  dataset <- as.character(source_detail$request$dataset)
  if (!exists(dataset, loaded_data, inherits = FALSE)) {
    payload <- new.env(parent = emptyenv())
    object_names <- load(data_paths[[dataset]], envir = payload)
    if (length(object_names) != 1L) {
      stop("data payload must contain one object", call. = FALSE)
    }
    data <- as.matrix(payload[[object_names[[1L]]]])
    storage.mode(data) <- "double"
    assign(dataset, data, loaded_data)
  }
  data <- get(dataset, loaded_data, inherits = FALSE)
  Y <- unname(data[, source_detail$targets, drop = FALSE])
  native_setup <- full_cuda_ci_native_setup_native(
    data[, source_detail$conditioning_set, drop = FALSE]
  )
  q <- ncol(native_setup$X)
  capacity <- NULL
  if (q > 80L || primary_fixed_reuse_diagnostic) {
    if (!fixed_sp_finalize) next
    geometry <- full_cuda_ci_native_geometry_prepare_native(
      native_setup$X, native_setup$penalty_blocks,
      native_setup$penalty_offsets, native_setup$penalty_ranks
    )
    capacity <- full_cuda_ci_multi_penalty_capacity_qualify_native(
      native_setup, geometry, Y
    )
    if (primary_fixed_reuse_diagnostic) {
      if (!isTRUE(capacity$diagnostics$exact_primary_only) ||
          !isTRUE(capacity$optimizer_device_resident_before_release) ||
          !isTRUE(capacity$optimizer_residual_slot_released) ||
          isTRUE(capacity$optimizer_fixed_sp_evaluation)) {
        stop(
          "exact-primary residual ownership qualification failed",
          call. = FALSE
        )
      }
    }
    optimization <- list(
      selected_log_sp = capacity$selected_log_sp,
      coefficients = capacity$optimizer_coefficients,
      fitted = capacity$optimizer_fitted,
      residuals = capacity$optimizer_residuals,
      solver_info = capacity$optimizer_solver_info,
      decomposition_route = capacity$optimizer_decomposition_route,
      numerical_rank = capacity$optimizer_numerical_rank,
      diagnostics = capacity$diagnostics
    )
    finalization <- list(
      coefficients = capacity$fixed_coefficients,
      residuals = capacity$fixed_residuals,
      solver_info = capacity$fixed_solver_info,
      decomposition_route = capacity$fixed_decomposition_route,
      diagnostics = list(cpu_objective_count = 0L)
    )
    residuals <- capacity$fixed_residuals
  } else {
    prepared <- fastkpc_full_cuda_phase6_prepare(
      prepared_setup_from_native(native_setup)
    )
    optimization <- fastkpc_full_cuda_phase6_optimize_cuda(
      prepared, Y,
      force_stability_replay_all = force_stability_replay_all
    )
    finalization <- if (fixed_sp_finalize) {
      fastkpc_full_cuda_phase6_evaluate_cuda(
        prepared, Y, optimization$selected_log_sp
      )
    } else {
      NULL
    }
    residuals <- if (fixed_sp_finalize) {
      Y - prepared$X %*% finalization$coefficients
    } else {
      Y - prepared$X %*% optimization$coefficients
    }
  }
  candidate_sp <- exp(optimization$selected_log_sp)
  reference_sp <- unname(source_detail$legacy_sp)
  reference_residuals <- unname(source_detail$residuals$legacy_auto)
  candidate_p <- p_value(residuals)
  reference_p <- p_value(reference_residuals)
  optimizer_fixed_log_sp_bitwise <- if (primary_fixed_reuse_diagnostic) {
    identical(
      as.double(capacity$selected_log_sp), as.double(capacity$fixed_log_sp)
    )
  } else {
    NA
  }
  optimizer_fixed_solver_info_exact <- if (
    primary_fixed_reuse_diagnostic
  ) {
    identical(
      as.integer(capacity$optimizer_solver_info),
      as.integer(capacity$fixed_solver_info)
    )
  } else {
    NA
  }
  optimizer_fixed_route_exact <- if (primary_fixed_reuse_diagnostic) {
    identical(
      as.integer(capacity$optimizer_decomposition_route),
      as.integer(capacity$fixed_decomposition_route)
    )
  } else {
    NA
  }
  optimizer_fixed_rank_exact <- if (primary_fixed_reuse_diagnostic) {
    identical(
      as.integer(capacity$optimizer_numerical_rank),
      as.integer(capacity$fixed_numerical_rank)
    )
  } else {
    NA
  }
  optimizer_fixed_coefficient_bitwise <- if (
    primary_fixed_reuse_diagnostic
  ) {
    identical(
      as.double(capacity$optimizer_coefficients),
      as.double(capacity$fixed_coefficients)
    )
  } else {
    NA
  }
  optimizer_fixed_fitted_bitwise <- if (primary_fixed_reuse_diagnostic) {
    identical(
      as.double(capacity$optimizer_fitted),
      as.double(capacity$fixed_fitted)
    )
  } else {
    NA
  }
  optimizer_fixed_residual_bitwise <- if (primary_fixed_reuse_diagnostic) {
    identical(
      as.double(capacity$optimizer_residuals),
      as.double(capacity$fixed_residuals)
    )
  } else {
    NA
  }
  optimizer_p <- if (primary_fixed_reuse_diagnostic) {
    p_value(capacity$optimizer_residuals)
  } else {
    NA_real_
  }
  optimizer_fixed_p_bitwise <- if (primary_fixed_reuse_diagnostic) {
    identical(optimizer_p, candidate_p)
  } else {
    NA
  }
  row <- data.frame(
    source_index = source_index,
    dataset = dataset,
    position = as.integer(source_detail$request$position),
    q = q,
    penalty_count = nrow(reference_sp),
    capacity_bucket = if (is.null(capacity)) "base-q80" else {
      capacity$capacity_bucket
    },
    max_log_sp_error = max(abs(
      optimization$selected_log_sp - log(reference_sp)
    )),
    max_sp_error = max(abs(candidate_sp - reference_sp)),
    max_sp_relative_error = max(abs(
      candidate_sp - reference_sp
    ) / pmax(abs(reference_sp), .Machine$double.xmin)),
    max_residual_error = max(abs(residuals - reference_residuals)),
    reference_p = reference_p,
    candidate_p = candidate_p,
    p_value_error = candidate_p - reference_p,
    contract_passed = abs(candidate_p - reference_p) <= 1e-10,
    optimizer_fixed_log_sp_bitwise = optimizer_fixed_log_sp_bitwise,
    optimizer_fixed_solver_info_exact = optimizer_fixed_solver_info_exact,
    optimizer_fixed_route_exact = optimizer_fixed_route_exact,
    optimizer_fixed_rank_exact = optimizer_fixed_rank_exact,
    optimizer_fixed_coefficient_bitwise =
      optimizer_fixed_coefficient_bitwise,
    optimizer_fixed_fitted_bitwise = optimizer_fixed_fitted_bitwise,
    optimizer_fixed_residual_bitwise = optimizer_fixed_residual_bitwise,
    optimizer_fixed_p_bitwise = optimizer_fixed_p_bitwise,
    optimizer_fixed_max_coefficient_error = if (
      primary_fixed_reuse_diagnostic
    ) {
      max(abs(
        capacity$optimizer_coefficients - capacity$fixed_coefficients
      ))
    } else {
      NA_real_
    },
    optimizer_fixed_max_fitted_error = if (
      primary_fixed_reuse_diagnostic
    ) {
      max(abs(capacity$optimizer_fitted - capacity$fixed_fitted))
    } else {
      NA_real_
    },
    optimizer_fixed_max_residual_error = if (
      primary_fixed_reuse_diagnostic
    ) {
      max(abs(capacity$optimizer_residuals - capacity$fixed_residuals))
    } else {
      NA_real_
    },
    optimizer_fixed_p_error = optimizer_p - candidate_p,
    cpu_objective_count = optimization$diagnostics$cpu_objective_count,
    finalization_cpu_objective_count = if (fixed_sp_finalize) {
      finalization$diagnostics$cpu_objective_count
    } else {
      0L
    },
    fallback_count = optimization$diagnostics$fallback_count,
    fixed_stable_svd = if (!fixed_sp_finalize) {
      FALSE
    } else if (!is.null(capacity)) {
      all(finalization$solver_info == 0L) &&
        all(finalization$decomposition_route == 2L)
    } else {
      all(finalization$solver_info == 0L) &&
        finalization$diagnostics$cuda_stable_svd_evaluation_count == ncol(Y)
    },
    stringsAsFactors = FALSE
  )
  rows[[length(rows) + 1L]] <- row
  details[[as.character(source_index)]] <- list(
    optimization = optimization,
    finalization = finalization,
    candidate_sp = candidate_sp,
    reference_sp = reference_sp,
    residuals = residuals,
    reference_residuals = reference_residuals,
    capacity = capacity
  )
  cat(sprintf(
    "source=%d q=%d log-sp=% .3e residual=% .3e p=% .3e pass=%s\n",
    source_index, q, row$max_log_sp_error, row$max_residual_error,
    row$p_value_error, row$contract_passed
  ))
}

rows <- do.call(rbind, rows)
receipt <- list(
  schema_version = if (fixed_sp_finalize) {
    if (primary_fixed_reuse_diagnostic) {
      "mgcv-gpu-exact-primary-fixed-reuse-qualification-v1"
    } else {
      "mgcv-gpu-selected-sp-fixed-finalization-qualification-v2"
    }
  } else {
    "mgcv-gpu-selected-sp-optimizer-qualification-v2"
  },
  native_sha256 = unname(digest::digest(
    file = "fastkpc/build/fastkpc_cuda.so", algo = "sha256",
    serialize = FALSE
  )),
  source_receipt = normalizePath(input_path),
  fixed_sp_finalize = fixed_sp_finalize,
  force_stability_replay_all = force_stability_replay_all,
  primary_fixed_reuse_diagnostic = primary_fixed_reuse_diagnostic,
  rows = rows,
  details = details,
  summary = list(
    case_count = nrow(rows),
    contract_pass_count = sum(rows$contract_passed),
    fixed_stable_svd_count = sum(rows$fixed_stable_svd),
    max_log_sp_error = max(rows$max_log_sp_error),
    max_sp_relative_error = max(rows$max_sp_relative_error),
    max_residual_error = max(rows$max_residual_error),
    max_abs_p_value_error = max(abs(rows$p_value_error)),
    optimizer_fixed_identity_count = if (primary_fixed_reuse_diagnostic) {
      sum(
        rows$optimizer_fixed_log_sp_bitwise &
          rows$optimizer_fixed_solver_info_exact &
          rows$optimizer_fixed_route_exact &
          rows$optimizer_fixed_rank_exact &
          rows$optimizer_fixed_coefficient_bitwise &
          rows$optimizer_fixed_fitted_bitwise &
          rows$optimizer_fixed_residual_bitwise &
          rows$optimizer_fixed_p_bitwise
      )
    } else {
      NA_integer_
    },
    optimizer_fixed_max_coefficient_error = if (
      primary_fixed_reuse_diagnostic
    ) max(rows$optimizer_fixed_max_coefficient_error) else NA_real_,
    optimizer_fixed_max_fitted_error = if (
      primary_fixed_reuse_diagnostic
    ) max(rows$optimizer_fixed_max_fitted_error) else NA_real_,
    optimizer_fixed_max_residual_error = if (
      primary_fixed_reuse_diagnostic
    ) max(rows$optimizer_fixed_max_residual_error) else NA_real_,
    optimizer_fixed_max_abs_p_error = if (
      primary_fixed_reuse_diagnostic
    ) max(abs(rows$optimizer_fixed_p_error)) else NA_real_,
    cpu_authority_count = sum(
      rows$cpu_objective_count + rows$finalization_cpu_objective_count
    ),
    fallback_count = sum(rows$fallback_count)
  )
)
saveRDS(receipt, output_path, compress = "xz", version = 3L)
print(rows, digits = 17, row.names = FALSE)
print(receipt$summary, digits = 17)
