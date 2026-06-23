source("fastkpc/R/mgcv_compat_contract.R")
source("fastkpc/R/native.R")

fastkpc_kpc_tprs_stop <- function(message) {
  stop(paste0("kpcTprsResidualCPP unsupported input: ", message),
       call. = FALSE)
}

fastkpc_kpc_tprs_validate_input <- function(y, S) {
  y <- as.numeric(y)
  if (length(y) == 0L || any(!is.finite(y))) {
    fastkpc_kpc_tprs_stop("y must be finite numeric")
  }
  S <- as.matrix(S)
  storage.mode(S) <- "double"
  if (nrow(S) != length(y)) {
    fastkpc_kpc_tprs_stop("y and S must have the same number of rows")
  }
  if (ncol(S) < 1L || ncol(S) > 2L) {
    fastkpc_kpc_tprs_stop("|S| = 1 or 2 is required")
  }
  if (any(!is.finite(S))) {
    fastkpc_kpc_tprs_stop("S must be finite numeric")
  }
  list(y = y, S = S)
}

fastkpc_kpc_tprs_setup_fingerprint <- function(S, setup) {
  fastkpc_hash_object(list(
    backend = "kpcTprsResidualCPP",
    schema = "setup-shadow-v1",
    n = nrow(S),
    d = ncol(S),
    input_hash = fastkpc_hash_object(round(as.numeric(S), digits = 14)),
    basis_rank = setup$basis_rank,
    null_space_rank = setup$null_space_rank,
    effective_rank = setup$effective_rank,
    basis_dim = dim(setup$X),
    penalty_hash = fastkpc_hash_object(round(as.numeric(setup$penalty),
                                             digits = 14)),
    constraint_hash = fastkpc_hash_object(round(as.numeric(setup$constraint),
                                                digits = 14))
  ))
}

fastkpc_kpc_tprs_residual_cpp_candidate <- function(
    y, S, k = NA_integer_, tol = sqrt(.Machine$double.eps)) {
  input <- fastkpc_kpc_tprs_validate_input(y, S)
  setup <- kpc_tprs_residual_cpp_setup(
    input$S,
    as.integer(if (is.na(k)) 0L else k),
    as.numeric(tol)
  )
  fingerprint <- fastkpc_kpc_tprs_setup_fingerprint(input$S, setup)

  list(
    backend_family = "kpcTprsResidualCPP",
    mode = "shadow-candidate-setup-only",
    authoritative = FALSE,
    family = "gaussian_identity",
    smooth_class = "tp",
    smooth_geometry = "joint-isotropic",
    residual_output_only = TRUE,
    conditioning_size = as.integer(ncol(input$S)),
    X = setup$X,
    penalty = setup$penalty,
    constraint = setup$constraint,
    coefficients = numeric(),
    fitted = numeric(),
    residuals = numeric(),
    edf = NA_real_,
    selected_sp = NA_real_,
    score = NA_real_,
    basis_rank = as.integer(setup$basis_rank),
    null_space_rank = as.integer(setup$null_space_rank),
    effective_rank = as.integer(setup$effective_rank),
    setup_fingerprint = fingerprint,
    diagnostics = list(
      schema_version = "kpcTprsResidualCPP-shadow-v1",
      setup_only = TRUE,
      native_backend = setup$backend_family,
      radial_basis = setup$radial_basis,
      polynomial_basis = setup$polynomial_basis,
      k = as.integer(setup$k),
      tol = as.numeric(tol),
      does_not_drive_graph_decisions = TRUE
    )
  )
}

fastkpc_kpc_tprs_residual_cpp_shadow <- function(y, S, oracle, ...) {
  if (!is.function(oracle)) {
    stop("oracle must be a function", call. = FALSE)
  }
  oracle_result <- oracle(y, S, ...)
  candidate_result <- tryCatch(
    fastkpc_kpc_tprs_residual_cpp_candidate(y = y, S = S),
    error = function(e) {
      list(
        backend_family = "kpcTprsResidualCPP",
        mode = "shadow-candidate-failed-closed",
        authoritative = FALSE,
        residuals = numeric(),
        fitted = numeric(),
        coefficients = numeric(),
        edf = NA_real_,
        selected_sp = NA_real_,
        score = NA_real_,
        basis_rank = NA_integer_,
        null_space_rank = NA_integer_,
        setup_fingerprint = NA_character_,
        diagnostics = list(
          schema_version = "kpcTprsResidualCPP-shadow-v1",
          error = conditionMessage(e),
          does_not_drive_graph_decisions = TRUE
        )
      )
    }
  )

  list(
    oracle = oracle_result,
    candidate = candidate_result,
    oracle_authoritative = TRUE,
    p_used_source = "oracle",
    decision_source = "oracle",
    graph_decision_changed = FALSE,
    diagnostics = list(
      schema_version = "kpcTprsResidualCPP-shadow-v1",
      candidate_mode = candidate_result$mode,
      candidate_available = !identical(candidate_result$mode,
                                       "shadow-candidate-failed-closed")
    )
  )
}
