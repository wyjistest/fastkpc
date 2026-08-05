source("fastkpc/R/fast_kpc.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

set.seed(1207)
data <- matrix(rnorm(48L * 4L), 48L, 4L)
permutation_params <- list(
  replicates = 13L,
  seed = 606L,
  include_observed = TRUE
)

a <- fast_kpc(
  data,
  alpha = 0.2,
  max_conditioning_size = 1L,
  engine = "cpu",
  precision = "hybrid",
  tau = 0,
  graph_stage = "skeleton",
  ci_method = "dcc.perm",
  permutation_params = permutation_params,
  precision_trace_level = "full"
)
b <- fast_kpc(
  data,
  alpha = 0.2,
  max_conditioning_size = 1L,
  engine = "cpu",
  precision = "hybrid",
  tau = 0,
  graph_stage = "skeleton",
  ci_method = "dcc.perm",
  permutation_params = permutation_params,
  precision_trace_level = "full"
)

trace_a <- a$diagnostics$precision_trace
trace_b <- b$diagnostics$precision_trace
required <- c(
  "ci_randomness_id",
  "permutation_seed_effective",
  "permutation_plan_spec_hash",
  "permutation_plan_hash",
  "permutation_replicates"
)

assert_true(a$skeleton$ci_method == "dcc.perm",
            "precision skeleton should retain dcc.perm")
assert_true(a$skeleton$ci_diagnostics$ci_dcc_perm_tests > 0L,
            "precision skeleton should count dcc.perm tests")
assert_true(a$skeleton$ci_diagnostics$ci_dcc_permutation_replicates ==
              a$skeleton$ci_diagnostics$ci_dcc_perm_tests * 13L,
            "precision skeleton should account for dcc.perm replicates")
assert_true(all(required %in% names(trace_a)),
            "precision dcc.perm trace should expose its randomness contract")
assert_true(all(grepl("^dcc.perm:", trace_a$ci_randomness_id)),
            "precision dcc.perm trace should use method-specific identities")
assert_true(all(trace_a$permutation_replicates == 13L),
            "precision dcc.perm trace should retain replicate count")
assert_true(identical(trace_a$ci_randomness_id, trace_b$ci_randomness_id) &&
              identical(trace_a$permutation_plan_spec_hash,
                        trace_b$permutation_plan_spec_hash),
            "precision dcc.perm fixed seed plan should repeat exactly")
assert_true(identical(a$skeleton$adjacency, b$skeleton$adjacency) &&
              identical(a$skeleton$pMax, b$skeleton$pMax),
            "precision dcc.perm fixed seed graph should repeat exactly")

cat("test_precision_dcc_perm_trace_plan.R: PASS\n")
