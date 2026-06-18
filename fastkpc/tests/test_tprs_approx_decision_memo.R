source("fastkpc/R/mgcv_extract_validation.R")
source("fastkpc/R/tprs_approx_decision_memo.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

out_dir <- tempfile("fastkpc-tprs-memo-")
campaign <- fastkpc_run_precision_ladder_attribution_campaign(
  output_dir = out_dir,
  seeds = c(5L, 6L),
  n_values = 40L,
  alpha = 0.05
)

memo <- fastkpc_write_tprs_approx_go_no_go_memo(
  attribution = campaign,
  output_dir = out_dir
)

memo_file <- file.path(out_dir, "tprs_approx_cuda_go_no_go_memo.md")
decision_file <- file.path(out_dir, "tprs_approx_cuda_decision.csv")

assert_true(file.exists(memo_file), "decision memo should exist")
assert_true(file.exists(decision_file), "machine-readable decision CSV should exist")
assert_true(identical(memo$decision, "defer"),
            "synthetic campaign should defer tprsApproxCUDA implementation")

text <- paste(readLines(memo_file, warn = FALSE), collapse = "\n")
required_phrases <- c(
  "tprsApproxCUDA Go/No-Go Memo",
  "Decision: defer",
  "basis projection floor",
  "oracle-lambda",
  "mgcvExtractGPU",
  "fastSplineCUDA",
  "Do not implement tprsApproxCUDA yet"
)
for (phrase in required_phrases) {
  assert_true(grepl(phrase, text, fixed = TRUE),
              paste("memo missing phrase:", phrase))
}

decision <- utils::read.csv(decision_file, stringsAsFactors = FALSE)
required_columns <- c(
  "decision", "mean_basis_projection_floor",
  "mean_oracle_lambda_improvement",
  "native_decision_flip_rate", "mean_skeleton_shd",
  "reason"
)
missing_columns <- setdiff(required_columns, names(decision))
assert_true(length(missing_columns) == 0L,
            paste("missing decision columns:",
                  paste(missing_columns, collapse = ", ")))
assert_true(identical(decision$decision[1], "defer"),
            "decision CSV should match memo")

cat("PASS tprsApproxCUDA decision memo\n")
