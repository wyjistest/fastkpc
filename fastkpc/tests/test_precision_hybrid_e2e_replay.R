source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

set.seed(103)
data <- matrix(rnorm(90 * 6), 90, 6)

primary <- fast_kpc(
  data, alpha = 0.2, max_conditioning_size = 1,
  engine = "cpu", precision = "fast",
  graph_stage = "skeleton", seed = 103
)
hybrid <- fast_kpc(
  data, alpha = 0.2, max_conditioning_size = 1,
  engine = "cpu", precision = "hybrid", tau = log(2),
  graph_stage = "skeleton", seed = 103
)

assert_true(hybrid$config$precision == "hybrid",
            "config should record hybrid precision")
assert_true(isTRUE(hybrid$config$canonical_replay_required),
            "hybrid must require canonical replay")
assert_true(is.data.frame(hybrid$diagnostics$precision_trace),
            "hybrid result should include precision trace")
trace <- hybrid$diagnostics$precision_trace
required <- c("run_id", "canonical_test_order_id", "backend_requested",
              "backend_used", "p_source_used", "fallback_reason")
assert_true(all(required %in% names(trace)),
            "precision trace should expose p-value source and fallback")
assert_true(identical(hybrid$skeleton$adjacency, primary$skeleton$adjacency) ||
              all(order(trace$canonical_test_order_id) ==
                    seq_along(trace$canonical_test_order_id)),
            "hybrid trace must preserve canonical ordering")

cat("PASS precision hybrid e2e replay\n")
