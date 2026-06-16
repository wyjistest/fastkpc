fastkpc_regrxons_formula_class <- function(S) {
  S <- as.integer(S)
  if (length(S) == 0L) return("direct-ci")
  if (length(S) <= 2L) return("full-smooth")
  "additive-smooth"
}

fastkpc_regrxons_semantics <- function(S, target, n, p,
                                       compatibility_mode = "kpcalg_regrXonS_v1") {
  S <- as.integer(S)
  list(
    compatibility_mode = compatibility_mode,
    target = as.integer(target),
    formula_class = fastkpc_regrxons_formula_class(S),
    conditioning_variable_order_used_in_formula = S,
    conditioning_set_as_set = sort(unique(S)),
    n = as.integer(n),
    p = as.integer(p),
    family = "gaussian_identity",
    output = "residuals_only"
  )
}

fastkpc_hash_object <- function(x) {
  path <- tempfile("fastkpc-hash-")
  on.exit(unlink(path), add = TRUE)
  con <- file(path, open = "wb")
  writeBin(serialize(x, NULL, version = 2), con)
  close(con)
  unname(as.character(tools::md5sum(path)))
}

fastkpc_collapse_key <- function(x) {
  if (length(x) == 0L) return("")
  paste(as.character(x), collapse = "|")
}

fastkpc_setup_fingerprint <- function(semantics,
                                      R_version = as.character(getRversion()),
                                      mgcv_version = if (requireNamespace("mgcv", quietly = TRUE)) {
                                        as.character(utils::packageVersion("mgcv"))
                                      } else {
                                        "unavailable"
                                      },
                                      fastkpc_version_or_source_hash = "",
                                      backend_family = "mgcvExtractCPU",
                                      backend_version = "v1",
                                      k = NA_integer_,
                                      bs = "tp",
                                      method = "GCV.Cp",
                                      optimizer = NA_character_,
                                      gamma = 1,
                                      select = FALSE,
                                      scale_setting = "mgcv-default",
                                      na_action = "na.fail",
                                      weights_policy = "none",
                                      intercept_policy = "mgcv-default",
                                      model_matrix_hash = "",
                                      penalty_hashes = character(),
                                      constraint_hash = "",
                                      rank_metadata = "",
                                      setup_warning_classes = character()) {
  fields <- list(
    R_version = R_version,
    mgcv_version = mgcv_version,
    fastkpc_version_or_source_hash = fastkpc_version_or_source_hash,
    kpcalg_compatibility_mode = semantics$compatibility_mode,
    backend_family = backend_family,
    backend_version = backend_version,
    formula_class = semantics$formula_class,
    conditioning_set_as_set = fastkpc_collapse_key(semantics$conditioning_set_as_set),
    conditioning_variable_order_used_in_formula =
      fastkpc_collapse_key(semantics$conditioning_variable_order_used_in_formula),
    n = semantics$n,
    input_p = semantics$p,
    k = k,
    bs = bs,
    method = method,
    optimizer = optimizer,
    gamma = gamma,
    select = select,
    family = semantics$family,
    scale_setting = scale_setting,
    na_action = na_action,
    weights_policy = weights_policy,
    intercept_policy = intercept_policy,
    model_matrix_hash = model_matrix_hash,
    penalty_hashes = fastkpc_collapse_key(penalty_hashes),
    constraint_hash = constraint_hash,
    rank_metadata = rank_metadata,
    setup_warning_classes = fastkpc_collapse_key(setup_warning_classes)
  )
  list(fields = fields, fingerprint = fastkpc_hash_object(fields))
}

fastkpc_target_fingerprint <- function(target, y_hash, sp_input = NULL,
                                       sp_output = NULL, selected_sp = NULL,
                                       score = NA_real_, edf = NA_real_,
                                       rank_if_target_specific = NA,
                                       target_warning_classes = character(),
                                       residual_hash = "",
                                       fitted_hash = "") {
  fields <- list(
    target_variable_id = as.integer(target),
    target_vector_hash = y_hash,
    sp_input = sp_input,
    sp_output = sp_output,
    selected_sp = selected_sp,
    score = score,
    edf = edf,
    rank_if_target_specific = rank_if_target_specific,
    target_warning_classes = fastkpc_collapse_key(target_warning_classes),
    residual_hash = residual_hash,
    fitted_hash = fitted_hash
  )
  list(fields = fields, fingerprint = fastkpc_hash_object(fields))
}
