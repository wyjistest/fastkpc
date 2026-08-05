source("fastkpc/R/cuda_native.R")
source("fastkpc/R/legacy_runner.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS", unset = "0"), "1")) {
  cat("SKIP default-Inf level-8 CUDA qualification\n")
  quit(save = "no", status = 0L)
}
if (!requireNamespace("jsonlite", quietly = TRUE) ||
    !isTRUE(tryCatch(fastkpc_cuda_available(), error = function(error) FALSE))) {
  cat("SKIP default-Inf level-8 CUDA qualification: dependency unavailable\n")
  quit(save = "no", status = 0L)
}

bits_to_double <- function(value) {
  bytes <- substring(value, seq.int(1L, 15L, 2L), seq.int(2L, 16L, 2L))
  readBin(as.raw(strtoi(bytes, base = 16L)), "double", n = 1L,
          size = 8L, endian = "little")
}

fixture <- jsonlite::read_json(
  "fastkpc/tests/fixtures/default_inf_level8_oracle_v1.json",
  simplifyVector = TRUE
)
data <- as.matrix(readRDS(file.path(
  "fastkpc", "artifacts", "kpc_tprs_real_zhu",
  "cancer_RD-causalDiscoveryInput.rds"
)))
storage.mode(data) <- "double"
assert_true(
  identical(fixture$schema_version,
            "fastkpc-default-inf-level8-oracle-v1") &&
    identical(fixture$level, 8L) && length(fixture$y) == 9L &&
    length(fixture$S_key) == 9L &&
    length(fixture$p_value_bits_le) == 9L,
  "default-Inf level-8 fixture is malformed"
)
oracle_p <- vapply(fixture$p_value_bits_le, bits_to_double, numeric(1L))
env <- fastkpc_legacy_env()
control <- list(
  convergence_tolerance = 1e-7,
  max_step_halving = 25L,
  max_iterations = 400L,
  max_newton_step = 5,
  boundary_probe_step = 2,
  max_boundary_probes = 5L,
  rank_tolerance = sqrt(.Machine$double.eps),
  keep_transcript = TRUE
)

rows <- lapply(seq_along(fixture$y), function(task_index) {
  S <- as.integer(strsplit(fixture$S_key[[task_index]], "\\|", fixed = FALSE)[[1L]])
  targets <- c(fixture$x, fixture$y[[task_index]])
  setup <- full_cuda_ci_native_setup_native(data[, S, drop = FALSE])
  assert_true(
    identical(dim(setup$X), c(351L, 73L)) &&
      length(setup$penalty_blocks) == 8L &&
      identical(as.integer(setup$penalty_ranks), rep.int(8L, 8L)),
    paste0("level-8 native setup shape changed at task ", task_index)
  )
  geometry <- full_cuda_ci_native_geometry_prepare_native(
    setup$X, setup$penalty_blocks, setup$penalty_offsets,
    setup$penalty_ranks
  )
  Y <- data[, targets, drop = FALSE]
  candidate <- full_cuda_ci_multi_penalty_gcv_optimize_cuda_native(
    setup$X, Y, geometry$magic_qr_packed, geometry$magic_tau,
    geometry$magic_r, geometry$magic_pivot, geometry$penalty_roots,
    geometry$penalty_matrices, setup$penalty_ranks,
    geometry$initial_log_sp, control = control
  )
  references <- lapply(seq_len(2L), function(target_index) {
    full_cuda_ci_multi_penalty_gcv_optimize_cpp_native(
      setup$X, Y[, target_index], setup$penalty_blocks,
      setup$penalty_offsets, setup$penalty_ranks, control = control
    )
  })
  reference_integer <- function(name) {
    as.integer(vapply(references, `[[`, integer(1L), name))
  }
  assert_true(
    all(candidate$optimizer_status == 0L) &&
      identical(as.integer(candidate$optimizer_iterations),
                reference_integer("optimizer_iterations")) &&
      identical(as.integer(candidate$score_calls),
                reference_integer("score_calls")) &&
      identical(as.integer(candidate$objective_calls),
                reference_integer("objective_calls")) &&
      identical(as.integer(candidate$step_halving_count),
                reference_integer("step_halving_count")) &&
      identical(as.integer(candidate$boundary_probe_count),
                reference_integer("boundary_probe_count")) &&
      identical(as.integer(candidate$boundary_accepted_count),
                reference_integer("boundary_accepted_count")) &&
      identical(candidate$boundary_status,
                do.call(cbind, lapply(references, `[[`, "boundary_status"))),
    paste0("level-8 CUDA optimizer trajectory changed at task ", task_index)
  )
  reference_log_sp <- do.call(cbind, lapply(
    references, `[[`, "selected_log_sp"
  ))
  max_log_sp_error <- max(abs(
    candidate$selected_log_sp - reference_log_sp
  ))
  assert_true(
    is.finite(max_log_sp_error) && max_log_sp_error <= 1e-6,
    paste0("level-8 selected log-sp exceeded the qualified tolerance at task ",
           task_index)
  )

  gam_data <- data.frame(cbind(Y, data[, S, drop = FALSE]))
  colnames(gam_data) <- paste0("x", seq_len(ncol(gam_data)))
  residuals <- sapply(seq_len(2L), function(target_index) {
    fit <- env$gam(
      env$frml.additive.smooth(target_index, 3:10),
      data = gam_data,
      sp = exp(candidate$selected_log_sp[, target_index])
    )
    as.numeric(fit$residuals)
  })
  p_value <- unname(env$dcov.gamma(
    residuals[, 1L], residuals[, 2L], index = 1, numCol = 35L
  )$p.value)
  assert_true(
    identical(p_value, oracle_p[[task_index]]) && p_value < fixture$alpha,
    paste0("level-8 final p-value is not bitwise identical at task ",
           task_index)
  )
  data.frame(
    task_index = task_index,
    y = targets[[2L]],
    max_log_sp_error = max_log_sp_error,
    p_value = p_value,
    stringsAsFactors = FALSE
  )
})

rows <- do.call(rbind, rows)
cat(
  "PASS default-Inf level-8 CUDA qualification; tasks=", nrow(rows),
  "; max_log_sp_error=", format(max(rows$max_log_sp_error), digits = 7L),
  "; p_values_bitwise=TRUE\n", sep = ""
)
