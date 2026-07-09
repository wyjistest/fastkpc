source("fastkpc/R/native.R")
source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP legacy dCov CUDA lowrank gamma parity: FASTKPC_RUN_CUDA_TESTS != 1\n")
  quit(save = "no", status = 0)
}

missing <- c("Rcpp", "RcppArmadillo", "RcppEigen", "RSpectra")[
  !vapply(c("Rcpp", "RcppArmadillo", "RcppEigen", "RSpectra"),
          requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP legacy dCov CUDA lowrank gamma parity: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

fixture_path <- "fastkpc/tests/fixtures/legacy_dcov_gamma_oracle_v1.rds"
if (!file.exists(fixture_path)) {
  cat("SKIP legacy dCov CUDA lowrank gamma parity: fixture unavailable\n")
  quit(save = "no", status = 0)
}

build_fastkpc_cuda_native(rebuild = TRUE)
if (!fastkpc_cuda_available()) {
  cat("SKIP legacy dCov CUDA lowrank gamma parity: CUDA unavailable\n")
  quit(save = "no", status = 0)
}

old_lowrank <- Sys.getenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK",
                          unset = NA_character_)
on.exit({
  if (is.na(old_lowrank)) {
    Sys.unsetenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK")
  } else {
    Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = old_lowrank)
  }
}, add = TRUE)

Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = "spectra")

cases <- readRDS(fixture_path)
fixture_case <- cases[[1L]]
rx <- fixture_case$residuals$rx[seq_len(96L)]
ry <- fixture_case$residuals$ry[seq_len(96L)]
num_col <- 8L

oracle <- fastkpc_legacy_dcov_gamma_cpp_oracle(
  rx, ry, numCol = num_col, index = 1
)
cuda <- legacy_dcov_spectra_matvec_cuda_lowrank_gamma(
  rx, ry, numCol = num_col, index = 1
)

assert_true(identical(cuda$backend,
                      "cuda-dense-sym-matvec-spectra-lowrank-gamma"),
            "CUDA lowrank gamma helper should report backend")
assert_true(isTRUE(cuda$converged_x) && isTRUE(cuda$converged_y),
            "CUDA lowrank gamma helper should converge for fixture residuals")
assert_true(abs(as.numeric(cuda$p.value) - as.numeric(oracle$p.value)) < 1e-10,
            "CUDA lowrank gamma p.value should match C++ Spectra oracle")
assert_true(abs(as.numeric(cuda$nV2) - as.numeric(oracle$nV2)) < 1e-7,
            "CUDA lowrank gamma nV2 should match C++ Spectra oracle")
assert_true(abs(as.numeric(cuda$mean) - as.numeric(oracle$mean)) < 1e-10,
            "CUDA lowrank gamma mean should match C++ Spectra oracle")
assert_true(abs(as.numeric(cuda$variance) - as.numeric(oracle$variance)) <
              1e-10,
            "CUDA lowrank gamma variance should match C++ Spectra oracle")
assert_true(abs(as.numeric(cuda$statistic) -
                  as.numeric(oracle$statistic)) < 1e-7,
            "CUDA lowrank gamma statistic should match C++ Spectra oracle")
assert_true(as.integer(cuda$spectra_matvec_count) > 0L,
            "CUDA lowrank gamma helper should report Spectra matvecs")
assert_true(identical(as.numeric(cuda$matrix_h2d_ms_during_compute), 0),
            "CUDA lowrank gamma helper should not re-upload during compute")

batch_case_indices <- seq_len(3L)
batch_x <- do.call(cbind, lapply(batch_case_indices, function(case_index) {
  cases[[case_index]]$residuals$rx[seq_len(96L)]
}))
batch_y <- do.call(cbind, lapply(batch_case_indices, function(case_index) {
  cases[[case_index]]$residuals$ry[seq_len(96L)]
}))
batch_oracles <- lapply(batch_case_indices, function(case_index) {
  fastkpc_legacy_dcov_gamma_cpp_oracle(
    cases[[case_index]]$residuals$rx[seq_len(96L)],
    cases[[case_index]]$residuals$ry[seq_len(96L)],
    numCol = num_col,
    index = 1
  )
})
cuda_batch <- legacy_dcov_spectra_matvec_cuda_lowrank_gamma_batch(
  batch_x, batch_y, numCol = num_col, index = 1
)
assert_true(identical(cuda_batch$backend,
                      "cuda-dense-sym-matvec-spectra-lowrank-gamma-batch"),
            "CUDA lowrank gamma batch helper should report backend")
assert_true(identical(as.integer(cuda_batch$diagnostics$batch_count),
                      length(batch_case_indices)),
            "CUDA lowrank gamma batch helper should report batch size")
assert_true(identical(as.integer(cuda_batch$diagnostics$converged_count),
                      2L * length(batch_case_indices)),
            "CUDA lowrank gamma batch helper should converge both solves per pair")
batch_p_diff <- abs(as.numeric(cuda_batch$p.value) -
                      vapply(batch_oracles, function(item) {
                        as.numeric(item$p.value)
                      }, numeric(1L)))
batch_nV2_diff <- abs(as.numeric(cuda_batch$nV2) -
                        vapply(batch_oracles, function(item) {
                          as.numeric(item$nV2)
                        }, numeric(1L)))
assert_true(max(batch_p_diff) < 1e-10,
            "CUDA lowrank gamma batch p.values should match C++ Spectra oracle")
assert_true(max(batch_nV2_diff) < 1e-7,
            "CUDA lowrank gamma batch nV2 values should match C++ Spectra oracle")
assert_true(as.integer(cuda_batch$diagnostics$spectra_matvec_count) > 0L,
            "CUDA lowrank gamma batch helper should report Spectra matvecs")
assert_true(identical(
  as.numeric(cuda_batch$diagnostics$matrix_h2d_ms_during_compute), 0),
  "CUDA lowrank gamma batch helper should not re-upload during compute")

duplicate_x <- cbind(batch_x[, 1L], batch_x[, 1L], batch_x[, 2L],
                     batch_x[, 1L])
duplicate_y <- cbind(batch_y[, 1L], batch_y[, 1L], batch_y[, 2L],
                     batch_y[, 1L])
duplicate_oracles <- list(batch_oracles[[1L]], batch_oracles[[1L]],
                          batch_oracles[[2L]], batch_oracles[[1L]])
cuda_duplicate_batch <- legacy_dcov_spectra_matvec_cuda_lowrank_gamma_batch(
  duplicate_x, duplicate_y, numCol = num_col, index = 1
)
duplicate_p_diff <- abs(as.numeric(cuda_duplicate_batch$p.value) -
                          vapply(duplicate_oracles, function(item) {
                            as.numeric(item$p.value)
                          }, numeric(1L)))
duplicate_nV2_diff <- abs(as.numeric(cuda_duplicate_batch$nV2) -
                            vapply(duplicate_oracles, function(item) {
                              as.numeric(item$nV2)
                            }, numeric(1L)))
assert_true(max(duplicate_p_diff) < 1e-10,
            "CUDA lowrank duplicate batch p.values should match oracle")
assert_true(max(duplicate_nV2_diff) < 1e-7,
            "CUDA lowrank duplicate batch nV2 values should match oracle")
assert_true(identical(
  as.integer(cuda_duplicate_batch$diagnostics$component_cache_lookup_count),
  2L * ncol(duplicate_x)),
  "CUDA lowrank duplicate batch should look up both components per pair")
assert_true(
  as.integer(cuda_duplicate_batch$diagnostics$component_cache_hit_count) > 0L,
  "CUDA lowrank duplicate batch should hit repeated residual components"
)
assert_true(
  as.integer(cuda_duplicate_batch$diagnostics$component_cache_miss_count) <
    2L * ncol(duplicate_x),
  "CUDA lowrank duplicate batch should avoid repeated component solves"
)
assert_true(identical(
  as.integer(cuda_duplicate_batch$diagnostics$component_cache_entry_count),
  as.integer(cuda_duplicate_batch$diagnostics$component_cache_miss_count)),
  "CUDA lowrank duplicate batch should count one entry per component miss")

grid_rows <- lapply(seq_len(6L), function(case_index) {
  item <- cases[[case_index]]
  x <- item$residuals$rx[seq_len(96L)]
  y <- item$residuals$ry[seq_len(96L)]
  ref <- fastkpc_legacy_dcov_gamma_cpp_oracle(
    x, y, numCol = num_col, index = 1
  )
  got <- legacy_dcov_spectra_matvec_cuda_lowrank_gamma(
    x, y, numCol = num_col, index = 1
  )
  data.frame(
    case_index = case_index,
    p_value_abs_diff = abs(as.numeric(got$p.value) -
                             as.numeric(ref$p.value)),
    nV2_abs_diff = abs(as.numeric(got$nV2) - as.numeric(ref$nV2)),
    mean_abs_diff = abs(as.numeric(got$mean) - as.numeric(ref$mean)),
    variance_abs_diff = abs(as.numeric(got$variance) -
                              as.numeric(ref$variance)),
    converged = isTRUE(got$converged_x) && isTRUE(got$converged_y),
    matvec_count = as.integer(got$spectra_matvec_count)
  )
})
grid <- do.call(rbind, grid_rows)

assert_true(all(grid$converged),
            "CUDA lowrank gamma helper should converge on fixture grid")
assert_true(max(grid$p_value_abs_diff) < 1e-10,
            "CUDA lowrank gamma fixture-grid p.values should match oracle")
assert_true(max(grid$nV2_abs_diff) < 1e-7,
            "CUDA lowrank gamma fixture-grid statistics should match oracle")
assert_true(max(grid$mean_abs_diff) < 1e-10,
            "CUDA lowrank gamma fixture-grid means should match oracle")
assert_true(max(grid$variance_abs_diff) < 1e-10,
            "CUDA lowrank gamma fixture-grid variances should match oracle")
assert_true(all(grid$matvec_count > 0L),
            "CUDA lowrank gamma fixture-grid should report matvecs")

cat(sprintf(
  "PASS legacy dCov CUDA lowrank gamma parity cases=%d max_p_diff=%.3g max_nV2_diff=%.3g\n",
  nrow(grid),
  max(grid$p_value_abs_diff),
  max(grid$nV2_abs_diff)
))
