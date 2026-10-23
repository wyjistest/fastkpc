suppressPackageStartupMessages(library(Rcpp))
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/native.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cuda.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")
source("fastkpc/R/full_cuda_ci_phase35_bakeoff.R")

sha <- function(x) digest::digest(x, algo = "sha256")
file_sha <- function(path) digest::digest(file = path, algo = "sha256", serialize = FALSE)
bits <- function(x) writeBin(as.double(x), raw(), size = 8L)
rel_l2 <- function(a, b) sqrt(sum((a-b)^2)) / max(sqrt(sum(b^2)), 1e-300)

make_setup <- function(native) {
  list(formula_class = "additive-smooth", X = unname(native$X),
    gram_matrix = unname(native$gram_matrix),
    penalty_blocks = setNames(lapply(native$penalty_blocks, unname),
      paste0("penalty_", seq_along(native$penalty_blocks))),
    penalty_offsets = as.integer(native$penalty_offsets),
    penalty_ranks = as.integer(native$penalty_ranks),
    mgcv_penalty_rank_metadata = as.integer(native$penalty_ranks),
    constraint = matrix(numeric(), 0L, ncol(native$X)), H = NULL, weights = NULL)
}

make_dto <- function(setup, key, data_p) {
  m <- length(setup$penalty_blocks); q <- ncol(setup$X)
  # Diagnostic geometry identities bind the actual arrays; these are not
  # canonical production residual keys and are never stored in its caches.
  lineage <- setNames(rep(list(sha(setup)), 7L), c("dataset_sha256",
    "prepared_s_key_sha256", "same_S_group_id", "phase1_setup_fingerprint",
    "provider_fingerprint", "semantic_fingerprint", "representation_fingerprint"))
  lineage$prepared_s_key_sha256 <- key
  c(list(schema_version = "full-cuda-ci-prepared-s-native-dto-v1"), lineage,
    list(prepared_s_setup_schema_version = "full-cuda-ci-prepared-s-setup-v1",
      native_dto_schema_version = "full-cuda-ci-prepared-s-native-dto-v1",
      data_p = as.integer(data_p), n = as.integer(nrow(setup$X)),
      coefficient_dim = as.integer(q), null_dim = as.integer(q),
      penalty_count = as.integer(m), X = unname(setup$X),
      constraint_mode = "identity", constraint_nullspace = NULL,
      gram_matrix = unname(setup$gram_matrix), nullspace_gram_matrix = NULL,
      penalty_blocks = setup$penalty_blocks,
      penalty_offsets_zero_based = as.integer(setup$penalty_offsets - 1L),
      penalty_ranks = setup$penalty_ranks,
      penalty_sp_indices_zero_based = as.integer(seq_len(m)-1L),
      penalty_sp_labels = paste0("sp", seq_len(m)), H = NULL,
      weights_policy = "none-or-unit", offset_policy = "none-or-zero"))
}

full_finalize <- function(prepared, fixture, selected) {
  key <- sha(list(fixture$id, fixture$setup))
  keys <- vapply(seq_len(2L), function(t) sha(list(key, t, bits(selected[,t]))), "")
  dto <- make_dto(fixture$setup, key, fixture$data_p)
  multi <- fastkpc_full_cuda_phase6_prepared_create(prepared, 2L)
  runtime <- fixed_sp_cuda_runtime_create(0L)
  handle <- token <- NULL
  on.exit({
    if (!is.null(token)) full_cuda_ci_multi_penalty_gcv_residual_free_native(token)
    if (!is.null(handle)) fixed_sp_cuda_prepared_free(handle)
    fixed_sp_cuda_runtime_free(runtime)
    full_cuda_ci_multi_penalty_gcv_prepared_free_native(multi)
  })
  fixed_sp_cuda_runtime_reserve(runtime, dto$n, dto$null_dim, 2L,
    dto$penalty_count, dto$n + sum(dto$penalty_ranks))
  handle <- fixed_sp_cuda_prepared_create(runtime, dto)
  token <- full_cuda_ci_multi_penalty_gcv_evaluate_fixed_batch_native(
    multi, fixture$Y, selected, keys)
  info <- full_cuda_ci_multi_penalty_gcv_residual_info_native(token)
  stopifnot(isTRUE(info$fixed_sp_evaluation), all(info$solver_info == 0L),
    all(info$decomposition_route == 2L))
  shadow <- full_cuda_ci_multi_penalty_gcv_residual_shadow_native(token)
  request <- fastkpc_full_cuda_phase35_legacy_eig_batch_request(
    key, keys, 1.0, 1L, 2L, 0.1, 2L, 35L)
  # This diagnostic bridge accepts the fixed_sp_evaluation metadata and
  # consumes its device view. The bridge's returned optimizer route label
  # is historical; the authoritative producer metadata is retained above.
  p <- fastkpc_full_cuda_phase35_legacy_eig_batch_from_optimizer_residual(
    handle, token, request)
  authority <- c("cpu_dcov_component_count", "cpu_dcov_eigen_count",
    "cpu_dcov_pair_statistic_count", "cpu_gamma_p_value_count")
  stopifnot(all(vapply(authority,function(k)identical(as.integer(p$diagnostics[[k]]),0L),TRUE)),
    p$diagnostics$cuda_full_eig_count==2L,
    p$diagnostics$residual_d2h_bytes==0,
    p$diagnostics$component_d2h_bytes==0)
  list(info = info, shadow = shadow, p = p)
}

trace_names <- function(m) c("kind", "iteration", "index", "decision", "route",
  "rank", "aggregate_rank", "score", "condition", "delta", "noise",
  "margin1", "margin2", paste0("sp", seq_len(m)))

guard_features <- function(result, setup) {
  m <- length(setup$penalty_blocks); q <- ncol(setup$X)
  minimum <- function(x) if (length(x)) min(x) else Inf
  do.call(rbind, lapply(1:2, function(t) {
    a <- result$trace[[t]]; colnames(a) <- trace_names(m)
    take <- function(k, f) a[a[,"kind"] %in% k, f]
    data.frame(score_margin = minimum(abs(take(c(1,5), "delta")) /
        pmax(take(c(1,5), "noise"), 1e-300)),
      convergence_margin = minimum(abs(c(take(2,"margin1"),take(2,"margin2")))),
      hessian_margin = minimum(abs(take(3,"margin1"))),
      newton_margin = minimum(abs(take(4,"margin1"))),
      max_condition = max(a[,"condition"]),
      rank_risk = any(a[,"rank"] != q),
      root_risk = any(a[,"aggregate_rank"] != sum(setup$penalty_ranks)),
      status_risk = result$status[t] != 0L || result$info[t] != 0L,
      mixed_svd = result$svd_calls[t] > 0L)
  }))
}

guard_flags <- function(features, scale) {
  # Empirical margins, NOT a certified bound. Grid fixed before any outcomes.
  risk <- features$rank_risk | features$root_risk | features$status_risk |
    features$mixed_svd
  margins <- features[,c("score_margin","convergence_margin","hessian_margin",
    "newton_margin","max_condition")]
  risk <- risk | apply(margins, 1L, function(x) any(is.na(x)))
  risk | features$score_margin <= 32 * scale |
    features$convergence_margin <= 1e-4 * scale |
    features$hessian_margin <= 1e-8 * scale |
    features$newton_margin <= 1e-4 * scale |
    features$max_condition >= 1e6 / max(scale, 1e-12)
}
