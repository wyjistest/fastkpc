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

fastkpc_apply_hybrid_verifier <- function(primary_rows, verifier_rows, policy) {
  primary <- as.data.frame(primary_rows, stringsAsFactors = FALSE)
  verifier <- as.data.frame(verifier_rows, stringsAsFactors = FALSE)
  required_primary <- c("canonical_test_order_id", "x", "y", "S_key", "primary_p")
  missing_primary <- setdiff(required_primary, names(primary))
  if (length(missing_primary) > 0L) {
    stop("primary rows missing fields: ", paste(missing_primary, collapse = ", "),
         call. = FALSE)
  }
  if (nrow(verifier) == 0L) {
    verifier <- data.frame(
      canonical_test_order_id = integer(),
      verifier_p = numeric(),
      verifier_backend = character(),
      stringsAsFactors = FALSE
    )
  }
  required_verifier <- c("canonical_test_order_id", "verifier_p")
  missing_verifier <- setdiff(required_verifier, names(verifier))
  if (length(missing_verifier) > 0L) {
    stop("verifier rows missing fields: ", paste(missing_verifier, collapse = ", "),
         call. = FALSE)
  }

  verifier_idx <- match(primary$canonical_test_order_id,
                        verifier$canonical_test_order_id)
  primary$verifier_p <- NA_real_
  has_match <- !is.na(verifier_idx)
  primary$verifier_p[has_match] <- verifier$verifier_p[verifier_idx[has_match]]
  primary$verifier_backend <- ""
  if ("verifier_backend" %in% names(verifier)) {
    primary$verifier_backend[has_match] <-
      verifier$verifier_backend[verifier_idx[has_match]]
  }

  resolved <- fastkpc_apply_hybrid_policy(primary, policy)
  used_verifier <- resolved$p_source_used == policy$verifier
  resolved$verifier_backend[used_verifier & !nzchar(resolved$verifier_backend)] <-
    policy$verifier
  resolved
}

fastkpc_parse_S_key <- function(S_key) {
  if (is.null(S_key) || !nzchar(S_key)) return(integer())
  as.integer(strsplit(as.character(S_key), "\\|", fixed = FALSE)[[1]])
}

fastkpc_replay_canonical_ci_decisions <- function(test_rows, alpha, p,
                                                  initial_adjacency = NULL) {
  rows <- as.data.frame(test_rows, stringsAsFactors = FALSE)
  rows <- rows[order(rows$canonical_test_order_id), , drop = FALSE]
  if (is.null(initial_adjacency)) {
    adjacency <- matrix(TRUE, p, p)
    diag(adjacency) <- FALSE
  } else {
    adjacency <- as.matrix(initial_adjacency)
    storage.mode(adjacency) <- "logical"
  }
  sepsets <- replicate(p, replicate(p, integer(), simplify = FALSE),
                       simplify = FALSE)
  rows$edge_deleted <- FALSE
  rows$edge_already_deleted <- FALSE
  rows$sepset_recorded <- ""

  edge_done <- new.env(parent = emptyenv())
  for (i in seq_len(nrow(rows))) {
    x <- as.integer(rows$x[i])
    y <- as.integer(rows$y[i])
    key <- paste(sort(c(x, y)), collapse = "-")
    if (isTRUE(edge_done[[key]]) || !isTRUE(adjacency[x, y])) {
      rows$edge_already_deleted[i] <- TRUE
      next
    }
    pval <- rows$p_used[i]
    if (!is.finite(pval)) pval <- 0
    if (pval >= alpha) {
      S <- fastkpc_parse_S_key(rows$S_key[i])
      adjacency[x, y] <- FALSE
      adjacency[y, x] <- FALSE
      sepsets[[x]][[y]] <- S
      sepsets[[y]][[x]] <- S
      rows$edge_deleted[i] <- TRUE
      rows$sepset_recorded[i] <- rows$S_key[i]
      edge_done[[key]] <- TRUE
    }
  }
  list(adjacency = adjacency, sepsets = sepsets, diagnostics = rows)
}
