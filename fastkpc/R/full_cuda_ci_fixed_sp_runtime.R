fastkpc_full_cuda_fixed_sp_contract <- function() {
  list(
    schema_version = "full-cuda-ci-fixed-sp-runtime-v1",
    native_dto_schema_version = "full-cuda-ci-prepared-s-native-dto-v1",
    cholesky_condition_max = 1e8,
    svd_condition_min = 1e12,
    route_levels = c(
      "CHOLESKY_BATCHED", "AUGMENTED_QR", "AUGMENTED_SVD"
    ),
    target_status_levels = c(
      "OK_CHOLESKY_BATCHED", "OK_CHOLESKY_SINGLE",
      "OK_AUGMENTED_QR", "OK_AUGMENTED_SVD",
      "ERR_NONFINITE_INPUT", "ERR_SP_SHAPE_OR_ORDER",
      "ERR_ROUTE_METADATA", "ERR_STABLE_PATH_NOT_IMPLEMENTED",
      "ERR_QR_FAILED", "ERR_SVD_FAILED", "ERR_NONFINITE_OUTPUT",
      "ERR_INTERNAL_CUDA"
    ),
    canonical_capacities = list(
      n = 351L, null_dim = 64L, target_count = 47L,
      penalty_count = 7L, augmented_rows = 407L
    )
  )
}

fastkpc_full_cuda_fixed_sp_validate_bare_vector <- function(
    value, name, allowed_types) {
  if (!(typeof(value) %in% allowed_types) || is.object(value) ||
      !is.null(attributes(value))) {
    stop(name, " must be a bare ", paste(allowed_types, collapse = "/"),
         " vector", call. = FALSE)
  }
  invisible(value)
}

fastkpc_full_cuda_fixed_sp_route <- function(
    condition, coefficient_rank, null_dim, authenticated) {
  validate_numeric <- function(value, name) {
    fastkpc_full_cuda_fixed_sp_validate_bare_vector(
      value, name, c("integer", "double")
    )
  }
  validate_integer_metadata <- function(value, name) {
    validate_numeric(value, name)
    non_na <- !is.na(value)
    if (any(non_na & (!is.finite(value) | value < 0 |
                     value != floor(value) |
                     value > .Machine$integer.max))) {
      stop(name, " must contain finite nonnegative integer-valued values",
           call. = FALSE)
    }
    invisible(value)
  }

  validate_numeric(condition, "condition")
  if (any(is.finite(condition) & condition < 0)) {
    stop("condition must have nonnegative finite values", call. = FALSE)
  }
  validate_integer_metadata(coefficient_rank, "coefficient_rank")
  validate_integer_metadata(null_dim, "null_dim")
  fastkpc_full_cuda_fixed_sp_validate_bare_vector(
    authenticated, "authenticated", "logical"
  )

  lengths <- c(length(condition), length(coefficient_rank),
               length(null_dim), length(authenticated))
  n <- max(lengths)
  if (n == 0L || any(!(lengths %in% c(1L, n)))) {
    stop("inputs must be non-empty and scalar or common-length vectors",
         call. = FALSE)
  }

  condition <- rep_len(condition, n)
  coefficient_rank <- rep_len(as.integer(coefficient_rank), n)
  null_dim <- rep_len(as.integer(null_dim), n)
  authenticated <- rep_len(authenticated, n)
  contract <- fastkpc_full_cuda_fixed_sp_contract()
  out <- rep(contract$route_levels[[3L]], n)
  trusted <- !is.na(authenticated) & authenticated &
    is.finite(condition) & !is.na(coefficient_rank) & !is.na(null_dim) &
    coefficient_rank == null_dim
  out[trusted & condition < contract$cholesky_condition_max] <-
    contract$route_levels[[1L]]
  out[trusted & condition >= contract$cholesky_condition_max &
        condition < contract$svd_condition_min] <- contract$route_levels[[2L]]
  if (n == 1L) out[[1L]] else out
}
