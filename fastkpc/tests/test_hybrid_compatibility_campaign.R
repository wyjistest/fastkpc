source("fastkpc/R/hybrid_compatibility_campaign.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

out_dir <- tempfile("fastkpc-hybrid-compat-")
dir.create(out_dir, recursive = TRUE)

campaign <- fastkpc_run_hybrid_compatibility_campaign(output_dir = out_dir)
files <- c(
  "mgcv_residual_compatibility.csv",
  "mgcv_ci_compatibility.csv",
  "mgcv_graph_compatibility.csv",
  "hybrid_near_alpha_diagnostics.csv"
)
for (file in files) {
  assert_true(file.exists(file.path(out_dir, file)),
              paste("missing artifact", file))
}
assert_true(is.data.frame(campaign$ci), "CI artifact should be data.frame")
assert_true(is.data.frame(campaign$graph), "graph artifact should be data.frame")
assert_true("decision_flip_rate" %in% names(campaign$summary),
            "summary should include decision flip rate")
assert_true("near_alpha_fraction" %in% names(campaign$summary),
            "summary should include near-alpha fraction")
assert_true("verifier_decision_changes" %in% names(campaign$summary),
            "summary should include verifier decision changes")

cat("PASS hybrid compatibility campaign\n")
