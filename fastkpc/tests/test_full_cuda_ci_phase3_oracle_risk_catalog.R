source("fastkpc/R/full_cuda_ci_phase3_artifacts.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) {
  if (!isTRUE(value)) fail(message)
}

keys <- c(strrep("1", 64L), strrep("2", 64L))
target_parity <- data.frame(
  residual_key_sha256 = keys,
  target = c(1L, 2L),
  stringsAsFactors = FALSE
)
risk_fields <- .fastkpc_full_cuda_phase3_oracle_risk_fields()
target_risks <- data.frame(
  residual_key_sha256 = keys,
  stringsAsFactors = FALSE
)
for (field in risk_fields) target_risks[[field]] <- FALSE
target_risks$high_condition[[1L]] <- TRUE

catalog <- list(inputs = list(
  target_fit_metadata = data.frame(
    residual_key_sha256 = keys,
    target = c(1L, 2L),
    stringsAsFactors = FALSE
  ),
  target_risks = target_risks
))

risk_cases <- .fastkpc_full_cuda_phase3_oracle_risk_cases(
  target_parity, catalog = catalog
)
assert_true(
  nrow(risk_cases) == 1L &&
    identical(risk_cases$residual_key_sha256, keys[[1L]]) &&
    identical(risk_cases$high_condition, TRUE),
  "catalog risk cases use the authenticated full target-risk table"
)

cat("full CUDA CI Phase 3 catalog risk selector test: PASS\n")
