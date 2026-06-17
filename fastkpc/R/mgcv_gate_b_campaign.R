source("fastkpc/R/mgcv_extract_oracle.R")

fastkpc_gate_b_formula <- function(S_size) {
  S_size <- as.integer(S_size)
  if (S_size == 1L) {
    stats::as.formula("y ~ s(s1)")
  } else if (S_size == 2L) {
    stats::as.formula("y ~ s(s1, s2)")
  } else {
    stats::as.formula(
      paste("y ~", paste(sprintf("s(s%d)", seq_len(S_size)), collapse = " + "))
    )
  }
}

fastkpc_gate_b_formula_class <- function(S_size) {
  if (as.integer(S_size) <= 2L) "full-smooth" else "additive-smooth"
}

fastkpc_gate_b_scenario_data <- function(seed, n, S_size, scenario_id) {
  set.seed(seed)
  z <- stats::runif(n, -2, 2)
  S <- matrix(stats::rnorm(n * S_size), n, S_size)
  if (identical(scenario_id, "mild_collinearity") && S_size >= 2L) {
    S[, 2] <- S[, 1] + stats::rnorm(n, sd = 0.03)
  }
  if (identical(scenario_id, "near_constant")) {
    S[, 1] <- 0.001 * stats::rnorm(n)
  }
  if (identical(scenario_id, "tied_values")) {
    S[, 1] <- round(S[, 1], digits = 1)
  }
  colnames(S) <- paste0("s", seq_len(S_size))
  y <- sin(z) + rowSums(S[, seq_len(S_size), drop = FALSE]) / max(1, S_size) +
    stats::rnorm(n, sd = 0.08)
  data.frame(y = y, S, check.names = FALSE)
}

fastkpc_gate_b_sp_value <- function(source, selected_sp, length_out) {
  if (identical(source, "selected")) return(selected_sp)
  value <- switch(
    source,
    small = 1e-4,
    medium = 1,
    large = 1e4,
    stop("unknown sp source: ", source, call. = FALSE)
  )
  rep(value, length_out)
}

fastkpc_matrix_rank <- function(x, tol = sqrt(.Machine$double.eps)) {
  if (is.null(x) || length(x) == 0L) return(0L)
  qr(as.matrix(x), tol = tol)$rank
}

fastkpc_condition_proxy <- function(A) {
  values <- tryCatch(svd(A, nu = 0, nv = 0)$d, error = function(e) numeric())
  values <- values[is.finite(values) & values > .Machine$double.eps]
  if (length(values) == 0L) return(Inf)
  max(values) / min(values)
}

fastkpc_gate_b_row <- function(scenario_id, seed, n, S_size, sp_source) {
  data <- fastkpc_gate_b_scenario_data(seed, n, S_size, scenario_id)
  formula <- fastkpc_gate_b_formula(S_size)
  selected <- mgcv::gam(formula, data = data, method = "GCV.Cp")
  sp <- fastkpc_gate_b_sp_value(sp_source, selected$sp, length(selected$sp))
  ref <- fastkpc_mgcv_gam_fixed_sp_reference(
    formula = formula,
    data = data,
    sp = sp,
    method = "GCV.Cp",
    target = 1L,
    S = seq.int(2L, length.out = S_size)
  )
  solved <- fastkpc_mgcv_extract_fixed_sp_solve(
    formula = formula,
    data = data,
    sp = sp,
    method = "GCV.Cp",
    target = 1L,
    S = seq.int(2L, length.out = S_size)
  )
  setup <- solved$setup
  P <- fastkpc_assemble_penalty(ncol(setup$X), setup$S, setup$off, setup$sp, setup$H)
  coef_rel <- fastkpc_relative_l2_diff(solved$coefficients, ref$coefficients)
  fitted_rel <- fastkpc_relative_l2_diff(solved$fitted, ref$fitted)
  residual_rel <- fastkpc_relative_l2_diff(solved$residuals, ref$residuals)
  max_abs_res <- max(abs(solved$residuals - ref$residuals))
  pass <- is.finite(fitted_rel) && is.finite(residual_rel) &&
    fitted_rel <= 1e-5 && residual_rel <= 1e-5 && max_abs_res <= 1e-5
  data.frame(
    scenario_id = scenario_id,
    seed = as.integer(seed),
    n = as.integer(n),
    S_size = as.integer(S_size),
    formula_class = fastkpc_gate_b_formula_class(S_size),
    sp_source = sp_source,
    sp = paste(signif(sp, 8), collapse = "|"),
    edf_reference = as.numeric(ref$edf),
    rank_setup = as.integer(setup$rank[1]),
    constraint_rank = fastkpc_matrix_rank(setup$C),
    penalty_rank = fastkpc_matrix_rank(P),
    coef_rel_l2 = coef_rel,
    fitted_rel_l2 = fitted_rel,
    residual_rel_l2 = residual_rel,
    max_abs_residual_diff = max_abs_res,
    condition_number_proxy = fastkpc_condition_proxy(crossprod(setup$X) + P),
    pass_gate_b = pass,
    warning_message = "",
    stringsAsFactors = FALSE
  )
}

fastkpc_run_mgcv_gate_b_campaign <- function(
    seeds = c(11, 12, 13),
    n_values = c(80, 200, 500),
    S_sizes = c(1L, 2L, 3L, 4L),
    scenarios = c("baseline", "mild_collinearity", "near_constant", "tied_values"),
    sp_grid = c("selected", "small", "medium", "large"),
    output_dir = NULL) {
  if (!requireNamespace("mgcv", quietly = TRUE)) {
    stop("mgcv is required for Gate B campaign", call. = FALSE)
  }
  rows <- list()
  for (scenario in scenarios) {
    for (seed in seeds) {
      for (n in n_values) {
        for (S_size in S_sizes) {
          for (sp_source in sp_grid) {
            row <- tryCatch(
              fastkpc_gate_b_row(scenario, seed, n, S_size, sp_source),
              error = function(e) data.frame(
                scenario_id = scenario,
                seed = as.integer(seed),
                n = as.integer(n),
                S_size = as.integer(S_size),
                formula_class = fastkpc_gate_b_formula_class(S_size),
                sp_source = sp_source,
                sp = "",
                edf_reference = NA_real_,
                rank_setup = NA_integer_,
                constraint_rank = NA_integer_,
                penalty_rank = NA_integer_,
                coef_rel_l2 = NA_real_,
                fitted_rel_l2 = NA_real_,
                residual_rel_l2 = NA_real_,
                max_abs_residual_diff = NA_real_,
                condition_number_proxy = NA_real_,
                pass_gate_b = FALSE,
                warning_message = conditionMessage(e),
                stringsAsFactors = FALSE
              )
            )
            rows[[length(rows) + 1L]] <- row
          }
        }
      }
    }
  }
  fixed_sp <- do.call(rbind, rows)
  if (!is.null(output_dir)) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    utils::write.csv(
      fixed_sp,
      file.path(output_dir, "mgcv_gate_b_fixed_sp_campaign.csv"),
      row.names = FALSE
    )
  }
  list(fixed_sp = fixed_sp, output_dir = output_dir)
}
