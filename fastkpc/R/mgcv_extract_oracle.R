source("fastkpc/R/mgcv_compat_contract.R")

fastkpc_require_mgcv <- function() {
  if (!requireNamespace("mgcv", quietly = TRUE)) {
    stop("mgcv is required for mgcvExtractCPU", call. = FALSE)
  }
  invisible(TRUE)
}

fastkpc_mgcv_hash_numeric <- function(x) {
  fastkpc_hash_object(round(as.numeric(x), digits = 14))
}

fastkpc_mgcv_penalty_hashes <- function(fit) {
  hashes <- character()
  smooths <- fit$smooth
  for (i in seq_along(smooths)) {
    penalties <- smooths[[i]]$S
    for (j in seq_along(penalties)) {
      hashes <- c(hashes, fastkpc_hash_object(penalties[[j]]))
    }
  }
  hashes
}

fastkpc_mgcv_selected_sp <- function(fit, fallback = NULL) {
  if (!is.null(fit$sp) && length(fit$sp) > 0L) return(fit$sp)
  if (!is.null(fit$full.sp) && length(fit$full.sp) > 0L) return(fit$full.sp)
  fallback
}

# Gate B only proves that the extraction bridge can reproduce residuals when
# mgcv supplies setup/sp semantics. It does not prove that fastkpc can construct
# mgcv's basis, penalties, constraints, rank behavior, or optimizer independently.
fastkpc_mgcv_extract_fixed_sp <- function(formula, data, sp,
                                          method = "GCV.Cp",
                                          target = 1L,
                                          S = integer(),
                                          k = NA_integer_,
                                          bs = "tp") {
  fastkpc_require_mgcv()
  data <- as.data.frame(data)

  fit <- mgcv::gam(
    formula = formula,
    data = data,
    sp = sp,
    method = method,
    fit = TRUE
  )

  residuals <- as.numeric(stats::residuals(fit))
  fitted <- as.numeric(stats::fitted(fit))
  selected_sp <- fastkpc_mgcv_selected_sp(fit, fallback = sp)
  response <- if (!is.null(fit$y)) fit$y else model.response(stats::model.frame(fit))
  lpmatrix_hash <- tryCatch(
    fastkpc_hash_object(round(stats::predict(fit, type = "lpmatrix"), digits = 14)),
    error = function(e) ""
  )
  sem <- fastkpc_regrxons_semantics(S = S, target = target,
                                    n = length(residuals), p = ncol(data))
  setup <- fastkpc_setup_fingerprint(
    sem,
    mgcv_version = as.character(utils::packageVersion("mgcv")),
    backend_family = "mgcvExtractCPU",
    backend_version = "fixed-sp-v1",
    k = k,
    bs = bs,
    method = method,
    model_matrix_hash = lpmatrix_hash,
    penalty_hashes = fastkpc_mgcv_penalty_hashes(fit),
    constraint_hash = fastkpc_hash_object(lapply(fit$smooth, `[[`, "C")),
    rank_metadata = paste0("rank=", fit$rank)
  )
  target_fp <- fastkpc_target_fingerprint(
    target = target,
    y_hash = fastkpc_mgcv_hash_numeric(response),
    sp_input = sp,
    sp_output = selected_sp,
    selected_sp = selected_sp,
    score = if (!is.null(fit$gcv.ubre)) as.numeric(fit$gcv.ubre) else NA_real_,
    edf = if (!is.null(fit$edf)) sum(fit$edf) else NA_real_,
    rank_if_target_specific = fit$rank,
    residual_hash = fastkpc_mgcv_hash_numeric(residuals),
    fitted_hash = fastkpc_mgcv_hash_numeric(fitted)
  )

  list(
    backend_family = "mgcvExtractCPU",
    mode = "fixed-sp",
    formula = formula,
    method = method,
    sp = selected_sp,
    residuals = residuals,
    fitted = fitted,
    score = if (!is.null(fit$gcv.ubre)) as.numeric(fit$gcv.ubre) else NA_real_,
    edf = if (!is.null(fit$edf)) sum(fit$edf) else NA_real_,
    rank = fit$rank,
    setup_fingerprint = setup,
    target_fingerprint = target_fp,
    mgcv_version = as.character(utils::packageVersion("mgcv"))
  )
}

# mgcvExtractGCVBridge may call mgcv to select smoothing parameters. A future
# mgcvPortGCVPrototype must be validated separately and should not inherit this
# bridge's strict parity gate until optimizer details are implemented.
fastkpc_mgcv_extract_gcv_bridge <- function(formula, data,
                                            method = "GCV.Cp",
                                            target = 1L,
                                            S = integer(),
                                            k = NA_integer_,
                                            bs = "tp") {
  fastkpc_require_mgcv()
  data <- as.data.frame(data)
  legacy <- mgcv::gam(formula = formula, data = data, method = method)
  fixed <- fastkpc_mgcv_extract_fixed_sp(
    formula = formula,
    data = data,
    sp = legacy$sp,
    method = method,
    target = target,
    S = S,
    k = k,
    bs = bs
  )
  fixed$mode <- "gcv-bridge"
  fixed$legacy_score <- if (!is.null(legacy$gcv.ubre)) as.numeric(legacy$gcv.ubre) else NA_real_
  fixed$legacy_edf <- if (!is.null(legacy$edf)) sum(legacy$edf) else NA_real_
  fixed$legacy_rank <- legacy$rank
  fixed
}

fastkpc_mgcv_extract_batch <- function(Y, S_data, S,
                                       target_ids = seq_len(ncol(Y)),
                                       formula_class = NULL,
                                       method = "GCV.Cp") {
  fastkpc_require_mgcv()
  Y <- as.matrix(Y)
  S_data <- as.data.frame(S_data)
  if (is.null(colnames(Y))) colnames(Y) <- paste0("target", seq_len(ncol(Y)))
  if (is.null(colnames(S_data))) colnames(S_data) <- paste0("s", seq_len(ncol(S_data)))
  if (is.null(formula_class)) formula_class <- fastkpc_regrxons_formula_class(S)

  n <- nrow(Y)
  q <- ncol(Y)
  residuals <- matrix(NA_real_, n, q)
  fitted <- matrix(NA_real_, n, q)
  sp <- vector("list", q)
  score <- rep(NA_real_, q)
  edf <- rep(NA_real_, q)
  ranks <- rep(NA_integer_, q)
  target_fps <- vector("list", q)

  rhs <- if (identical(formula_class, "additive-smooth")) {
    paste(sprintf("s(%s)", colnames(S_data)), collapse = " + ")
  } else {
    sprintf("s(%s)", paste(colnames(S_data), collapse = ", "))
  }

  sem <- fastkpc_regrxons_semantics(S = S, target = target_ids[1],
                                    n = n, p = q + ncol(S_data))
  setup_fp <- fastkpc_setup_fingerprint(
    sem,
    mgcv_version = as.character(utils::packageVersion("mgcv")),
    backend_family = "mgcvExtractCPU",
    backend_version = "batch-gcv-bridge-v1",
    method = method,
    model_matrix_hash = fastkpc_hash_object(list(S_data = S_data, rhs = rhs))
  )

  for (j in seq_len(q)) {
    local_data <- cbind(data.frame(.target = Y[, j]), S_data)
    form <- stats::as.formula(paste(".target ~", rhs))
    fit <- fastkpc_mgcv_extract_gcv_bridge(
      formula = form,
      data = local_data,
      method = method,
      target = target_ids[j],
      S = S
    )
    residuals[, j] <- fit$residuals
    fitted[, j] <- fit$fitted
    sp[[j]] <- fit$sp
    score[j] <- fit$score
    edf[j] <- fit$edf
    ranks[j] <- fit$rank
    target_fps[[j]] <- fit$target_fingerprint
  }

  colnames(residuals) <- colnames(Y)
  colnames(fitted) <- colnames(Y)
  list(
    backend_family = "mgcvExtractCPU",
    mode = "batch-gcv-bridge",
    residuals = residuals,
    fitted = fitted,
    sp = sp,
    score = score,
    edf = edf,
    rank = ranks,
    setup_fingerprint = setup_fp,
    target_fingerprints = target_fps,
    formula_class = formula_class,
    method = method
  )
}
