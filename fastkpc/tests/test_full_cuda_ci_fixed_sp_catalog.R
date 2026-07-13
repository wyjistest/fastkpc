source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_error <- function(expression, message) {
  error <- tryCatch({ force(expression); NULL }, error = identity)
  assert_true(inherits(error, "error"), message)
}

phase1_dir <- file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "workload_census_351x48_v1"
)
targets <- readRDS(file.path(phase1_dir, "target_fit_metadata.rds"))
setups <- readRDS(file.path(phase1_dir, "same_s_setup_metadata.rds"))
setup_index <- match(targets$same_S_group_id, setups$same_S_group_id)
assert_true(!anyNA(setup_index), "canonical target/setup null-dim join")
target_null_dim <- as.integer(
  setups$constraint_nullspace_dimension[setup_index]
)
routes <- fastkpc_full_cuda_fixed_sp_route(
  condition = targets$penalized_system_condition_at_selected_sp,
  coefficient_rank = targets$coefficient_rank,
  null_dim = target_null_dim,
  authenticated = rep(TRUE, nrow(targets))
)

planned_route_counts <- table(factor(
  routes,
  levels = c("CHOLESKY_BATCHED", "AUGMENTED_QR", "AUGMENTED_SVD")
))
assert_true(identical(
  as.integer(planned_route_counts), c(73158L, 4210L, 33249L)
), "canonical Phase 3 planned route counts")
rank_deficient <- targets$coefficient_rank < target_null_dim
assert_true(sum(rank_deficient) == 1L &&
              all(routes[rank_deficient] == "AUGMENTED_SVD"),
            "canonical rank-deficient targets route to SVD")
nonfinite_condition <-
  !is.finite(targets$penalized_system_condition_at_selected_sp)
assert_true(sum(nonfinite_condition) == 1162L &&
              all(routes[nonfinite_condition] == "AUGMENTED_SVD"),
            "canonical nonfinite-condition targets route to SVD")

assert_true(identical(
  fastkpc_full_cuda_fixed_sp_route(
    condition = 1, coefficient_rank = 10L, null_dim = 10L,
    authenticated = FALSE
  ),
  "AUGMENTED_SVD"
), "finite unauthenticated conditions must route to SVD")
assert_true(identical(
  fastkpc_full_cuda_fixed_sp_route(
    condition = 1, coefficient_rank = 9L, null_dim = 10L,
    authenticated = TRUE
  ),
  "AUGMENTED_SVD"
), "finite rank-deficient targets must route to SVD")

assert_error(
  fastkpc_full_cuda_fixed_sp_route(
    condition = factor("1"), coefficient_rank = 10L, null_dim = 10L,
    authenticated = TRUE
  ), "factor condition must be rejected")
assert_error(
  fastkpc_full_cuda_fixed_sp_route(
    condition = as.Date("2020-01-01"), coefficient_rank = 10L,
    null_dim = 10L, authenticated = TRUE
  ), "Date condition must be rejected")
assert_error(
  fastkpc_full_cuda_fixed_sp_route(
    condition = matrix(1, nrow = 1L), coefficient_rank = 10L,
    null_dim = 10L, authenticated = TRUE
  ), "matrix condition must be rejected")
assert_error(
  fastkpc_full_cuda_fixed_sp_route(
    condition = "1", coefficient_rank = 10L, null_dim = 10L,
    authenticated = TRUE
  ), "character condition must be rejected")
assert_error(
  fastkpc_full_cuda_fixed_sp_route(
    condition = 1, coefficient_rank = 10.5, null_dim = 10L,
    authenticated = TRUE
  ), "fractional coefficient rank must be rejected")
assert_error(
  fastkpc_full_cuda_fixed_sp_route(
    condition = -1, coefficient_rank = 10L, null_dim = 10L,
    authenticated = TRUE
  ), "negative condition must be rejected")
assert_error(
  fastkpc_full_cuda_fixed_sp_route(
    condition = 1, coefficient_rank = -1L, null_dim = 10L,
    authenticated = TRUE
  ), "negative coefficient rank must be rejected")
assert_error(
  fastkpc_full_cuda_fixed_sp_route(
    condition = 1, coefficient_rank = 10L, null_dim = 10L,
    authenticated = 1
  ), "nonlogical authentication must be rejected")
assert_error(
  fastkpc_full_cuda_fixed_sp_route(
    condition = 1, coefficient_rank = 10L, null_dim = 10L,
    authenticated = matrix(TRUE, nrow = 1L)
  ), "matrix authentication must be rejected")
assert_error(
  fastkpc_full_cuda_fixed_sp_route(
    condition = c(1, 2), coefficient_rank = c(10L, 10L, 10L),
    null_dim = 10L, authenticated = TRUE
  ), "incompatible non-scalar lengths must be rejected")

contract <- fastkpc_full_cuda_fixed_sp_contract()
assert_true(identical(contract$schema_version,
                      "full-cuda-ci-fixed-sp-runtime-v1"),
            "runtime schema version")
assert_true(identical(contract$native_dto_schema_version,
                      "full-cuda-ci-prepared-s-native-dto-v1"),
            "native DTO schema version")
assert_true(identical(contract$cholesky_condition_max, 1e8),
            "Cholesky condition threshold")
assert_true(identical(contract$svd_condition_min, 1e12),
            "SVD condition threshold")
assert_true(identical(
  fastkpc_full_cuda_fixed_sp_route(
    condition = contract$cholesky_condition_max,
    coefficient_rank = 10L, null_dim = 10L, authenticated = TRUE
  ),
  "AUGMENTED_QR"
), "Cholesky threshold belongs to QR interval")
assert_true(identical(
  fastkpc_full_cuda_fixed_sp_route(
    condition = contract$svd_condition_min,
    coefficient_rank = 10L, null_dim = 10L, authenticated = TRUE
  ),
  "AUGMENTED_SVD"
), "SVD threshold belongs to SVD interval")
assert_true(identical(
  contract$route_levels,
  c("CHOLESKY_BATCHED", "AUGMENTED_QR", "AUGMENTED_SVD")
), "route levels")
assert_true(identical(
  contract$target_status_levels,
  c(
    "OK_CHOLESKY_BATCHED", "OK_CHOLESKY_SINGLE",
    "OK_AUGMENTED_QR", "OK_AUGMENTED_SVD",
    "ERR_NONFINITE_INPUT", "ERR_SP_SHAPE_OR_ORDER",
    "ERR_ROUTE_METADATA", "ERR_STABLE_PATH_NOT_IMPLEMENTED",
    "ERR_QR_FAILED", "ERR_SVD_FAILED", "ERR_NONFINITE_OUTPUT",
    "ERR_INTERNAL_CUDA"
  )
), "target status levels")
assert_true(identical(
  contract$canonical_capacities,
  list(
    n = 351L, null_dim = 64L, target_count = 47L,
    penalty_count = 7L, augmented_rows = 407L
  )
), "canonical capacities")

cat("PASS Phase 3 fixed-sp route contract\n")
