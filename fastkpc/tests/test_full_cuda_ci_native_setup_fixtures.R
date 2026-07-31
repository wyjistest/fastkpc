source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cpp.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

fixture_rows <- data.frame(
  label = c("one-dimensional", "joint-two-dimensional", "additive-three",
            "additive-seven"),
  shard_id = c(53L, 0L, 1L, 1L),
  prepared_s_key_sha256 = c(
    "038625cf986592962b731c79bf4c53c0415c820c7175a56b374a1f4ede5d5e55",
    "000bf94226b34186828cfa30c400753eb19ca2ff99409573df21ac06da2a72be",
    "001245052f571033286b2dc7526c24dbe5ec5c221660c094a8b9f052376b91da",
    "05b18f09b303d481e84ec14c6a3d11da92db2fcc0f9f213aa8319015f7369ae4"
  ),
  expected_columns = c(10L, 30L, 28L, 64L),
  expected_penalties = c(1L, 1L, 3L, 7L),
  stringsAsFactors = FALSE
)

data <- readRDS(file.path(
  "fastkpc", "artifacts", "kpc_tprs_real_zhu",
  "cancer_RD-causalDiscoveryInput.rds"
))
data <- as.matrix(data)
storage.mode(data) <- "double"

results <- lapply(seq_len(nrow(fixture_rows)), function(index) {
  fixture <- fixture_rows[index, , drop = FALSE]
  shard <- readRDS(file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1",
    "shards", paste0("shard_", fixture$shard_id, ".rds")
  ))
  oracle <- shard$prepared_s_setups[[fixture$prepared_s_key_sha256]]
  assert_true(!is.null(oracle), paste0("missing fixture: ", fixture$label))

  conditioning <- data[, oracle$sorted_S, drop = FALSE]
  candidate <- full_cuda_ci_native_setup_native(conditioning)
  repeated <- full_cuda_ci_native_setup_native(conditioning)

  assert_true(
    identical(candidate$schema_version, "full-cuda-ci-native-setup-v1") &&
      identical(candidate$semantic_version,
                "mgcv-1.9-1-tprs-native-setup-v1") &&
      is.character(candidate$semantic_fingerprint) &&
      length(candidate$semantic_fingerprint) == 1L &&
      grepl("^[0-9a-f]{64}$", candidate$semantic_fingerprint) &&
      identical(candidate$semantic_fingerprint,
                repeated$semantic_fingerprint),
    paste0("native setup identity mismatch: ", fixture$label)
  )
  assert_true(
    identical(candidate$formula_class, oracle$formula_class) &&
      identical(candidate$n, as.integer(nrow(data))) &&
      identical(candidate$S_size, as.integer(length(oracle$sorted_S))) &&
      identical(dim(candidate$X), dim(oracle$X)) &&
      ncol(candidate$X) == fixture$expected_columns &&
      length(candidate$penalty_blocks) == fixture$expected_penalties &&
      identical(candidate$penalty_offsets, oracle$penalty_offsets) &&
      identical(candidate$penalty_ranks,
                oracle$mgcv_penalty_rank_metadata) &&
      identical(candidate$basis_dimensions, oracle$basis_dimensions) &&
      identical(candidate$null_space_dimensions,
                oracle$smooth_null_space_dimensions),
    paste0("native setup structure mismatch: ", fixture$label)
  )
  assert_true(
    identical(candidate$diagnostics$legacy_mgcv_setup_count, 0L) &&
      identical(candidate$diagnostics$r_callback_count, 0L) &&
      identical(candidate$diagnostics$unsupported_count, 0L),
    paste0("native setup authority mismatch: ", fixture$label)
  )

  x_max_abs_error <- max(abs(candidate$X - oracle$X))
  penalty_max_abs_error <- max(vapply(
    seq_along(candidate$penalty_blocks),
    function(penalty_index) max(abs(
      candidate$penalty_blocks[[penalty_index]] -
        oracle$penalty_blocks[[penalty_index]]
    )),
    numeric(1L)
  ))
  assert_true(
    is.finite(x_max_abs_error) && x_max_abs_error <= 2e-11,
    paste0("native setup coordinate drift: ", fixture$label,
           " max_abs=", format(x_max_abs_error, digits = 17L))
  )
  assert_true(
    is.finite(penalty_max_abs_error) && penalty_max_abs_error <= 2e-10,
    paste0("native setup penalty drift: ", fixture$label,
           " max_abs=", format(penalty_max_abs_error, digits = 17L))
  )

  geometry <- full_cuda_ci_native_geometry_prepare_native(
    candidate$X, candidate$penalty_blocks, candidate$penalty_offsets,
    candidate$penalty_ranks
  )
  oracle_qr <- qr(oracle$X, LAPACK = TRUE)
  oracle_mroot <- get("mroot", envir = asNamespace("mgcv"))
  oracle_roots <- lapply(seq_along(oracle$penalty_blocks), function(block) {
    local_root <- oracle_mroot(
      oracle$penalty_blocks[[block]],
      rank = oracle$mgcv_penalty_rank_metadata[[block]], method = "chol"
    )
    full_root <- matrix(
      0, ncol(oracle$X), oracle$mgcv_penalty_rank_metadata[[block]]
    )
    rows <- oracle$penalty_offsets[[block]] + seq_len(nrow(local_root)) - 1L
    full_root[rows, ] <- local_root
    full_root[oracle_qr$pivot, , drop = FALSE]
  })
  oracle_penalty_matrices <- lapply(oracle_roots, tcrossprod)
  oracle_initial_sp <- get("initial.sp", envir = asNamespace("mgcv"))(
    oracle$X, oracle$penalty_blocks, oracle$penalty_offsets
  )
  assert_true(
    identical(geometry$schema_version, "full-cuda-ci-native-geometry-v1") &&
      identical(geometry$magic_q, qr.Q(oracle_qr, complete = FALSE)) &&
      identical(geometry$magic_qr_packed, unclass(oracle_qr)$qr) &&
      identical(geometry$magic_tau, as.numeric(oracle_qr$qraux)) &&
      identical(geometry$magic_r, qr.R(oracle_qr, complete = FALSE)) &&
      identical(geometry$magic_pivot, as.integer(oracle_qr$pivot)) &&
      all(vapply(seq_along(oracle_roots), function(block) {
        identical(geometry$penalty_roots[[block]], oracle_roots[[block]]) &&
          identical(
            geometry$penalty_matrices[[block]],
            oracle_penalty_matrices[[block]]
          )
      }, logical(1L))) &&
      identical(geometry$initial_sp, as.numeric(oracle_initial_sp)) &&
      identical(geometry$diagnostics$legacy_mgcv_mroot_count, 0L) &&
      identical(geometry$diagnostics$legacy_mgcv_initial_sp_count, 0L) &&
      identical(geometry$diagnostics$r_qr_count, 0L),
    paste0("native setup geometry drift: ", fixture$label)
  )

  list(
    fixture = fixture,
    oracle = oracle,
    candidate = candidate,
    geometry = geometry,
    x_max_abs_error = x_max_abs_error,
    penalty_max_abs_error = penalty_max_abs_error
  )
})

for (result_index in c(3L, 4L)) {
  result <- results[[result_index]]
  target <- if (result_index == 4L) 13L else {
    setdiff(seq_len(ncol(data)), result$oracle$sorted_S)[[1L]]
  }
  y <- as.numeric(data[, target])
  native_setup <- result$oracle
  native_setup$X <- result$candidate$X
  native_setup$penalty_blocks <- result$candidate$penalty_blocks
  native_setup$penalty_offsets <- result$candidate$penalty_offsets
  native_setup$penalty_ranks <- result$candidate$penalty_ranks
  native_setup$mgcv_penalty_rank_metadata <- result$candidate$penalty_ranks

  oracle_fit <- fastkpc_full_cuda_phase5_optimize_cpp(result$oracle, y)
  native_fit <- fastkpc_full_cuda_phase5_optimize_cpp(native_setup, y)
  fitted_max_abs_error <- max(abs(native_fit$fitted - oracle_fit$fitted))
  residual_max_abs_error <- max(abs(
    native_fit$residuals - oracle_fit$residuals
  ))
  log_sp_max_abs_error <- max(abs(
    native_fit$selected_log_sp - oracle_fit$selected_log_sp
  ))
  assert_true(
    fitted_max_abs_error <= 2e-8 && residual_max_abs_error <= 2e-8 &&
      log_sp_max_abs_error <= 2e-6,
    paste0(
      "native selected-fit drift: ", result$fixture$label,
      " target=", target,
      " fitted=", format(fitted_max_abs_error, digits = 17L),
      " residual=", format(residual_max_abs_error, digits = 17L),
      " log_sp=", format(log_sp_max_abs_error, digits = 17L)
    )
  )
}

cat("PASS Phase 7 native setup representative fixtures\n")
