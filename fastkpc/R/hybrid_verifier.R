fastkpc_hybrid_policy <- function(enabled = TRUE,
                                  alpha = 0.05,
                                  tau = log(3),
                                  primary = "fastSplineCUDA",
                                  verifier = "mgcvExtractCPU",
                                  always_verify_nan = TRUE,
                                  always_verify_boundary = TRUE) {
  list(
    enabled = isTRUE(enabled),
    alpha = as.numeric(alpha),
    tau = as.numeric(tau),
    primary = as.character(primary),
    verifier = as.character(verifier),
    always_verify_nan = isTRUE(always_verify_nan),
    always_verify_boundary = isTRUE(always_verify_boundary)
  )
}

fastkpc_near_alpha <- function(p, policy) {
  if (!isTRUE(policy$enabled)) return(FALSE)
  if (!is.finite(p)) return(isTRUE(policy$always_verify_nan))
  p <- max(as.numeric(p), .Machine$double.xmin)
  alpha <- max(as.numeric(policy$alpha), .Machine$double.xmin)
  abs(log(p / alpha)) <= policy$tau + 1e-12
}

fastkpc_apply_hybrid_policy <- function(test_rows, policy) {
  out <- as.data.frame(test_rows, stringsAsFactors = FALSE)
  out$near_alpha_triggered <- vapply(out$primary_p, fastkpc_near_alpha,
                                    logical(1), policy = policy)
  has_verifier <- "verifier_p" %in% names(out) & is.finite(out$verifier_p)
  use_verifier <- out$near_alpha_triggered & has_verifier
  out$p_used <- out$primary_p
  out$p_used[use_verifier] <- out$verifier_p[use_verifier]
  out$p_source_used <- policy$primary
  out$p_source_used[use_verifier] <- policy$verifier
  out$decision_before_verify <- out$primary_p > policy$alpha
  out$decision_after_verify <- out$p_used > policy$alpha
  out$verification_reason <- ""
  out$verification_reason[out$near_alpha_triggered] <- "near-alpha"
  out
}
