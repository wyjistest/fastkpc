source("fastkpc/R/validation_scenarios.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

names <- fastkpc_scenario_names()
required <- c("chain", "fork", "collider", "independent", "additive")
assert_true(all(required %in% names), "required scenario names should exist")

for (scenario in required) {
  a <- generate_fastkpc_scenario(scenario = scenario, seed = 10, n = 80)
  b <- generate_fastkpc_scenario(scenario = scenario, seed = 10, n = 80)
  c <- generate_fastkpc_scenario(scenario = scenario, seed = 11, n = 80)
  assert_true(is.matrix(a$data), paste(scenario, "data should be matrix"))
  assert_true(nrow(a$data) == 80, paste(scenario, "nrow should match"))
  assert_true(ncol(a$data) >= 4, paste(scenario, "should have at least four variables"))
  assert_true(identical(a$data, b$data), paste(scenario, "same seed should reproduce"))
  assert_true(!identical(a$data, c$data), paste(scenario, "different seed should differ"))
  assert_true(is.matrix(a$truth$adjacency), paste(scenario, "truth adjacency should exist"))
  assert_true(identical(dim(a$truth$adjacency), c(ncol(a$data), ncol(a$data))),
              paste(scenario, "truth adjacency dimension should match"))
  assert_true(is.character(a$description), paste(scenario, "description should exist"))
}

err <- tryCatch(generate_fastkpc_scenario("not-a-scenario", seed = 1, n = 20),
                error = conditionMessage)
assert_true(grepl("Unknown fastkpc validation scenario", err),
            "unknown scenario should fail clearly")

cat("test_validation_scenarios.R: PASS\n")
