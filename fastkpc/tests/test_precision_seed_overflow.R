source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

ids <- c(1L, 2147L, 2148L, 100000L, 1000000L)
seen <- lapply(ids, function(id) {
  fastkpc_precision_ci_randomness(
    ci_method = "hsic.perm",
    permutation_params = list(
      replicates = 12L,
      seed = 404L,
      include_observed = TRUE
    ),
    canonical_test_order_id = id
  )
})

for (i in seq_along(ids)) {
  value <- seen[[i]]
  assert_true(length(value$permutation_seed_effective) == 1L,
              "effective seed should be scalar")
  assert_true(!is.na(value$permutation_seed_effective),
              paste("effective seed should not be NA for test id", ids[[i]]))
  assert_true(value$permutation_seed_effective >= 0L,
              paste("effective seed should be non-negative for test id", ids[[i]]))
  assert_true(nzchar(value$permutation_plan_spec_hash),
              "permutation plan spec hash should be populated")

  again <- fastkpc_precision_ci_randomness(
    ci_method = "hsic.perm",
    permutation_params = list(
      replicates = 12L,
      seed = 404L,
      include_observed = TRUE
    ),
    canonical_test_order_id = ids[[i]]
  )
  assert_true(identical(value$permutation_seed_effective,
                        again$permutation_seed_effective),
              "effective seed should be reproducible")
  assert_true(identical(value$permutation_plan_spec_hash,
                        again$permutation_plan_spec_hash),
              "permutation plan spec hash should be reproducible")
}

cat("PASS precision seed overflow\n")
