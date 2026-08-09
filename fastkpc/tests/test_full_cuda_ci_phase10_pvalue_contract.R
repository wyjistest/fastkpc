source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")
source("fastkpc/R/full_cuda_ci_phase9_artifact.R")
source("fastkpc/R/full_cuda_ci_phase10_campaign.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

logical <- data.frame(
  logical_sequence_id = 1:3,
  reference_p_value = c(0.01, 0.1, 0.8)
)
result <- list(
  tasks = data.frame(
    canonical_test_order_id = 1:3,
    p_used = logical$reference_p_value
  ),
  pMax = matrix(c(1, 0.1, 0.1, 1), 2L, 2L)
)
oracle <- list(reference = list(pMax = result$pMax))

exact <- fastkpc_full_cuda_phase10_pvalue_contract_gate(
  result, logical, oracle
)
assert_true(
  isTRUE(exact$pass) && exact$tolerance == 1e-10 &&
    exact$ci$mismatch_count == 0L && exact$pMax$mismatch_count == 0L,
  "exact Phase 10 p-value payload did not pass the numerical contract"
)

drifted_ci <- result
drifted_ci$tasks$p_used[[3L]] <-
  drifted_ci$tasks$p_used[[3L]] + 2e-10
ci_gate <- fastkpc_full_cuda_phase10_pvalue_contract_gate(
  drifted_ci, logical, oracle
)
assert_true(
  !isTRUE(ci_gate$pass) && ci_gate$ci$mismatch_count == 1L &&
    ci_gate$ci$maximum_absolute_error > 1e-10,
  "Phase 10 accepted a CI p-value outside the 1e-10 contract"
)

drifted_pmax <- result
drifted_pmax$pMax[[2L]] <- drifted_pmax$pMax[[2L]] + 2e-10
pmax_gate <- fastkpc_full_cuda_phase10_pvalue_contract_gate(
  drifted_pmax, logical, oracle
)
assert_true(
  !isTRUE(pmax_gate$pass) && pmax_gate$pMax$mismatch_count == 1L &&
    pmax_gate$pMax$maximum_absolute_error > 1e-10,
  "Phase 10 accepted pMax outside the 1e-10 contract"
)

reordered <- result
reordered$tasks$canonical_test_order_id <- c(2L, 1L, 3L)
identity_gate <- fastkpc_full_cuda_phase10_pvalue_contract_gate(
  reordered, logical, oracle
)
assert_true(
  !isTRUE(identity_gate$pass) && !identity_gate$logical_identity_exact,
  "Phase 10 p-value gate accepted a reordered logical identity"
)

cat("PASS Phase 10 all-CI p-value numerical contract gate\n")
