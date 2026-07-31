fastkpc_full_cuda_phase10_development_require <- function(
    condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
  invisible(TRUE)
}

fastkpc_full_cuda_phase10_development_500x50_path <- function() {
  file.path(
    "fastkpc", "tests", "fixtures",
    "full_cuda_ci_development_500x50_v1.rds"
  )
}

fastkpc_full_cuda_phase10_development_500x50_data <- function() {
  old_kind <- RNGkind()
  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_seed) old_seed <- get(".Random.seed", envir = .GlobalEnv)
  on.exit({
    do.call(RNGkind, as.list(old_kind))
    if (had_seed) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)
  RNGkind("Mersenne-Twister", "Inversion", "Rejection")
  set.seed(1050050L)
  n <- 500L
  p <- 50L
  noise <- matrix(stats::rnorm(n * p), nrow = n, ncol = p)
  data <- noise
  for (start in seq.int(1L, p, by = 5L)) {
    data[, start + 1L] <-
      0.65 * data[, start] + sqrt(1 - 0.65^2) * noise[, start + 1L]
    data[, start + 2L] <-
      0.55 * data[, start] + sqrt(1 - 0.55^2) * noise[, start + 2L]
    data[, start + 3L] <-
      0.50 * data[, start + 1L] +
      sqrt(1 - 0.50^2) * noise[, start + 3L]
    data[, start + 4L] <-
      0.45 * data[, start + 2L] +
      sqrt(1 - 0.45^2) * noise[, start + 4L]
  }
  colnames(data) <- sprintf("dev500_v%02d", seq_len(p))
  storage.mode(data) <- "double"
  data
}

fastkpc_full_cuda_phase10_development_oracle <- function(result) {
  fastkpc_full_cuda_phase10_development_require(
    fastkpc_full_cuda_is_skeleton(result),
    "Phase 10 public 500x50 CPU oracle is not a skeleton"
  )
  list(
    reference = list(
      adjacency = result$adjacency,
      sepsets = result$sepsets,
      n.edgetests = as.integer(result$n.edgetests),
      pMax = result$pMax
    ),
    deletion_trace = fastkpc_full_cuda_normalize_deletion_trace(result),
    logical_trace = fastkpc_full_cuda_normalize_logical_trace(result)
  )
}

fastkpc_full_cuda_phase10_build_development_500x50 <- function() {
  data <- fastkpc_full_cuda_phase10_development_500x50_data()
  expected_environment <- c(
    FASTKPC_LEGACY_DCOV_GAMMA_BACKEND = "cpp",
    FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = "spectra",
    FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE = "1",
    FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY = "s",
    FASTKPC_NATIVE_LEGACY_MGCV_PROVIDER_CORES = "20",
    FASTKPC_NATIVE_LEGACY_DCOV_BATCH = "round",
    FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH_THREADS = "20"
  )
  old <- Sys.getenv(names(expected_environment), unset = NA_character_)
  on.exit(fastkpc_restore_env_vars(old), add = TRUE)
  do.call(Sys.setenv, as.list(expected_environment))
  captured <- fastkpc_full_cuda_phase10_campaign_timed_call(
    fastkpc_compatible_cuda_skeleton(
      data = data,
      alpha = 0.1,
      labels = colnames(data),
      options = list(
        route = "legacy", compatible_cuda_strict = TRUE,
        max_conditioning_size = 7L, index = 1, numCol = 35L,
        trace_level = "logical", dcov_batch = "round",
        mgcv_residual_backend = "r"
      )
    )
  )
  result <- captured$value
  oracle <- fastkpc_full_cuda_phase10_development_oracle(result)
  list(
    schema_version = "full-cuda-ci-development-500x50-v1",
    fixture_id = "public-development-500x50-v1",
    generator = list(
      RNGkind = c("Mersenne-Twister", "Inversion", "Rejection"),
      seed = 1050050L,
      block_count = 10L,
      block_size = 5L,
      source_sha256 = fastkpc_full_cuda_census_file_hash(
        "fastkpc/R/full_cuda_ci_phase10_development_500x50.R"
      )
    ),
    configuration = list(
      n = 500L, p = 50L, alpha = "0.1", index = 1L, num_col = 35L,
      maximum_conditioning_size = 7L,
      model = "Gaussian-identity-unweighted-zero-offset",
      oracle_route = "legacy-mgcv-provider-native-legacy-dcov-20-core"
    ),
    data = data,
    data_sha256 = unname(digest::digest(
      data, algo = "sha256", serialize = TRUE
    )),
    oracle = oracle,
    oracle_sha256 = unname(digest::digest(
      oracle, algo = "sha256", serialize = TRUE
    )),
    baseline_elapsed_sec = captured$elapsed_sec,
    baseline_summary = result$summary,
    promotion_holdout_claim = FALSE,
    pass = TRUE
  )
}

fastkpc_full_cuda_phase10_validate_development_500x50 <- function(
    artifact) {
  generated <- fastkpc_full_cuda_phase10_development_500x50_data()
  clean <- is.list(artifact) && identical(
    artifact$schema_version, "full-cuda-ci-development-500x50-v1"
  ) && identical(artifact$fixture_id, "public-development-500x50-v1") &&
    identical(artifact$configuration$n, 500L) &&
    identical(artifact$configuration$p, 50L) &&
    identical(artifact$configuration$maximum_conditioning_size, 7L) &&
    identical(dim(artifact$data), c(500L, 50L)) &&
    identical(artifact$data, generated) && all(is.finite(artifact$data)) &&
    identical(
      artifact$data_sha256,
      unname(digest::digest(artifact$data, algo = "sha256", serialize = TRUE))
    ) && identical(
      artifact$generator$source_sha256,
      fastkpc_full_cuda_census_file_hash(
        "fastkpc/R/full_cuda_ci_phase10_development_500x50.R"
      )
    ) && is.list(artifact$oracle) &&
    fastkpc_full_cuda_is_skeleton(artifact$oracle$reference) &&
    nrow(artifact$oracle$logical_trace) ==
      sum(artifact$oracle$reference$n.edgetests) &&
    all(is.finite(artifact$oracle$logical_trace$p_value)) &&
    identical(
      artifact$oracle_sha256,
      unname(digest::digest(
        artifact$oracle, algo = "sha256", serialize = TRUE
      ))
    ) && is.finite(artifact$baseline_elapsed_sec) &&
    artifact$baseline_elapsed_sec > 0 &&
    !isTRUE(artifact$promotion_holdout_claim) && isTRUE(artifact$pass)
  fastkpc_full_cuda_phase10_development_require(
    clean, "Phase 10 public 500x50 development artifact is invalid"
  )
  invisible(artifact)
}

fastkpc_full_cuda_phase10_load_development_500x50 <- function(
    path = fastkpc_full_cuda_phase10_development_500x50_path()) {
  fastkpc_full_cuda_phase10_development_require(
    file.exists(path) && !dir.exists(path),
    "Phase 10 public 500x50 development artifact is missing"
  )
  artifact <- readRDS(path)
  fastkpc_full_cuda_phase10_validate_development_500x50(artifact)
  artifact
}
