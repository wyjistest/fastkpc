source("fastkpc/R/cuda_native.R")
source("fastkpc/R/native.R")

args <- commandArgs(trailingOnly = TRUE)
input_path <- if (length(args) >= 1L) args[[1L]] else {
  "/tmp/fastkpc_causal_large_parity_20260810/residual_route_bakeoff.rds"
}
output_path <- if (length(args) >= 2L) args[[2L]] else {
  "/tmp/fastkpc_fixed_sp_cuda_certificate_qualification.rds"
}
if (!file.exists(input_path)) stop("source receipt is missing", call. = FALSE)
if (file.exists(output_path)) stop("output already exists", call. = FALSE)

data_paths <- c(
  `20230331145508` =
    "causalDiscovery/20230331145508/combined_data_shuffle-causalDiscoveryInput.RData",
  `20240102191119` =
    "causalDiscovery/20240102191119/GSE99254-482col-causalDiscoveryInput.RData",
  `20241215091815` =
    "causalDiscovery/20241215091815/logged-lung-exhausted-allData-causalDiscoveryInput.RData"
)

resource_snapshot <- function() {
  .Call("C_fixed_sp_cuda_test_resource_snapshot", PACKAGE = "fastkpc_cuda")
}

make_test_dto <- function(native_setup, data, conditioning_set) {
  identity <- digest::digest(
    list(conditioning_set, native_setup$semantic_fingerprint),
    algo = "sha256"
  )
  blocks <- native_setup$penalty_blocks
  names(blocks) <- paste0("penalty_", seq_along(blocks))
  penalty_count <- length(blocks)
  list(
    schema_version = "full-cuda-ci-prepared-s-native-dto-v1",
    dataset_sha256 = digest::digest(data, algo = "sha256"),
    prepared_s_key_sha256 = identity,
    same_S_group_id = identity,
    phase1_setup_fingerprint = identity,
    provider_fingerprint = identity,
    semantic_fingerprint = native_setup$semantic_fingerprint,
    representation_fingerprint = identity,
    prepared_s_setup_schema_version = "full-cuda-ci-prepared-s-setup-v1",
    native_dto_schema_version = "full-cuda-ci-prepared-s-native-dto-v1",
    data_p = 48L,
    n = as.integer(native_setup$n),
    coefficient_dim = as.integer(ncol(native_setup$X)),
    null_dim = as.integer(ncol(native_setup$X)),
    penalty_count = as.integer(penalty_count),
    X = unname(native_setup$X),
    constraint_mode = "identity",
    constraint_nullspace = NULL,
    gram_matrix = unname(native_setup$gram_matrix),
    nullspace_gram_matrix = NULL,
    penalty_blocks = blocks,
    penalty_offsets_zero_based =
      as.integer(native_setup$penalty_offsets - 1L),
    penalty_ranks = as.integer(native_setup$penalty_ranks),
    penalty_sp_indices_zero_based =
      as.integer(seq_len(penalty_count) - 1L),
    penalty_sp_labels = paste0("penalty_", seq_len(penalty_count)),
    H = NULL,
    weights_policy = "none-or-unit",
    offset_policy = "none-or-zero"
  )
}

p_value <- function(residuals) {
  fastkpc_cuda_legacy_dcov_gamma_cpp_oracle(
    residuals[, 1L], residuals[, 2L], numCol = 35L, index = 1
  )$p.value
}

load_fastkpc_cuda_native()
if (!fastkpc_cuda_available()) stop("CUDA is unavailable", call. = FALSE)
source_receipt <- readRDS(input_path)
loaded_data <- new.env(parent = emptyenv())
resource_before <- resource_snapshot()
rows <- vector("list", length(source_receipt$details))
details <- vector("list", length(source_receipt$details))

for (source_index in seq_along(source_receipt$details)) {
  source_detail <- source_receipt$details[[source_index]]
  dataset <- as.character(source_detail$request$dataset)
  if (!exists(dataset, loaded_data, inherits = FALSE)) {
    payload <- new.env(parent = emptyenv())
    object_names <- load(data_paths[[dataset]], envir = payload)
    if (length(object_names) != 1L) {
      stop("data payload must contain one object", call. = FALSE)
    }
    data <- as.matrix(payload[[object_names[[1L]]]])
    storage.mode(data) <- "double"
    assign(dataset, data, loaded_data)
  }
  data <- get(dataset, loaded_data, inherits = FALSE)
  native_setup <- full_cuda_ci_native_setup_native(
    data[, source_detail$conditioning_set, drop = FALSE]
  )
  dto <- make_test_dto(
    native_setup, data, source_detail$conditioning_set
  )
  Y <- unname(data[, source_detail$targets, drop = FALSE])
  SP <- unname(source_detail$gpu_sp)
  target_keys <- vapply(seq_len(ncol(Y)), function(target) {
    digest::digest(
      list(dto$prepared_s_key_sha256, target, SP[, target]),
      algo = "sha256"
    )
  }, character(1L))

  runtime <- fixed_sp_cuda_runtime_create(0L)
  handle <- NULL
  token <- NULL
  cleanup <- function() {
    if (!is.null(token)) {
      try(fixed_sp_cuda_residual_release(token), silent = TRUE)
      try(fixed_sp_cuda_residual_free(token), silent = TRUE)
      token <<- NULL
    }
    if (!is.null(handle)) {
      try(fixed_sp_cuda_prepared_free(handle), silent = TRUE)
      handle <<- NULL
    }
    if (!is.null(runtime)) {
      try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE)
      runtime <<- NULL
    }
  }
  case_result <- tryCatch({
    fixed_sp_cuda_runtime_reserve(
      runtime, nrow(Y), dto$null_dim, ncol(Y), nrow(SP),
      nrow(Y) + dto$null_dim
    )
    handle <- fixed_sp_cuda_prepared_create(runtime, dto)

    token <- fixed_sp_cuda_solve_batch(
      handle, Y, SP, source_detail$planned_routes, target_keys,
      outputs = "residuals"
    )
    default_info <- fixed_sp_cuda_residual_info(token)
    default_shadow <- fixed_sp_cuda_materialize_shadow(token, "residuals")
    fixed_sp_cuda_residual_release(token)
    fixed_sp_cuda_residual_free(token)
    token <- NULL

    token <- fixed_sp_cuda_solve_certified_batch_for_test(
      handle, Y, SP, source_detail$planned_routes, target_keys,
      outputs = "residuals", refinement_max_iterations = 4L,
      backward_error_tolerance = 1e-15, fail_closed = FALSE
    )
    certified_info <- fixed_sp_cuda_residual_info(token)
    certified_shadow <- fixed_sp_cuda_materialize_shadow(token, "residuals")
    fixed_sp_cuda_residual_release(token)
    fixed_sp_cuda_residual_free(token)
    token <- NULL

    token <- fixed_sp_cuda_solve_certified_batch_for_test(
      handle, Y, SP, source_detail$planned_routes, target_keys,
      outputs = "residuals", refinement_max_iterations = 4L,
      backward_error_tolerance = 1e-15, fail_closed = TRUE
    )
    fail_closed_info <- fixed_sp_cuda_residual_info(token)
    fail_closed_materialization <- tryCatch(
      fixed_sp_cuda_materialize_shadow(token, "residuals"),
      error = identity
    )
    fail_closed_after <- fixed_sp_cuda_residual_info(token)
    fixed_sp_cuda_residual_release(token)
    fixed_sp_cuda_residual_free(token)
    token <- NULL

    reference <- unname(source_detail$residuals$native_gpu_sp)
    default_expected <- unname(source_detail$residuals$gpu)
    certificate_passed <- all(
      certified_info$residual_certificate_passed
    )
    fixed_p_error <- p_value(certified_shadow$residuals) -
      p_value(reference)
    fail_closed_expected <- !certificate_passed
    fail_closed_error <- inherits(fail_closed_materialization, "error")
    fail_closed_observed <- identical(
      fail_closed_error, fail_closed_expected
    )
    failed_shadow_suppressed <- !fail_closed_expected ||
      (fail_closed_error &&
       fail_closed_after$shadow_materialize_call_count == 0L &&
       fail_closed_after$shadow_materialize_target_count == 0L &&
       fail_closed_after$shadow_d2h_bytes == 0)
    mgcv_p_contract_passed <- abs(fixed_p_error) <= 1e-10
    certificate_semantic_false_positive <-
      certificate_passed && !mgcv_p_contract_passed
    list(
      row = data.frame(
        source_index = source_index,
        dataset = dataset,
        position = as.integer(source_detail$request$position),
        level = as.integer(source_detail$request$level),
        q = dto$null_dim,
        route = paste(source_detail$planned_routes, collapse = ","),
        default_bitwise_exact = identical(
          as.double(default_shadow$residuals),
          as.double(default_expected)
        ),
        initial_kkt_max = max(
          certified_info$initial_kkt_backward_error
        ),
        final_kkt_max = max(certified_info$final_kkt_backward_error),
        certificate_passed = certificate_passed,
        refinement_solver_info_zero = all(
          certified_info$refinement_solver_info == 0L
        ),
        max_residual_error = max(abs(
          certified_shadow$residuals - reference
        )),
        fixed_p_error = fixed_p_error,
        mgcv_p_contract_passed = mgcv_p_contract_passed,
        certificate_semantic_false_positive =
          certificate_semantic_false_positive,
        fail_closed_expected = fail_closed_expected,
        fail_closed_observed = fail_closed_observed,
        failed_shadow_suppressed = failed_shadow_suppressed,
        default_p_error = p_value(default_shadow$residuals) -
          p_value(reference),
        cpu_fallback_count = certified_info$cpu_fallback_count,
        unknown_fallback_count = certified_info$unknown_fallback_count,
        stringsAsFactors = FALSE
      ),
      detail = list(
        request = source_detail$request,
        default_info = default_info,
        certified_info = certified_info,
        fail_closed_info = fail_closed_info,
        fail_closed_after = fail_closed_after,
        fail_closed_error = if (fail_closed_error) {
          conditionMessage(fail_closed_materialization)
        } else {
          NULL
        },
        default_residuals = default_shadow$residuals,
        certified_residuals = certified_shadow$residuals,
        reference_residuals = reference
      )
    )
  }, error = function(error) {
    list(error = conditionMessage(error))
  })
  cleanup()
  if (!is.null(case_result$error)) {
    stop(
      "fixed-sp certificate case ", source_index, " failed: ",
      case_result$error, call. = FALSE
    )
  }
  rows[[source_index]] <- case_result$row
  details[[source_index]] <- case_result$detail
  print(case_result$row, digits = 17, row.names = FALSE)
}

rows <- do.call(rbind, rows)
invisible(gc())
resource_after <- resource_snapshot()
active_fields <- grep("_active_count$", names(resource_before), value = TRUE)
receipt <- list(
  schema_version = "mgcv-fixed-sp-cuda-kkt-diagnostic-v2",
  native_sha256 = unname(digest::digest(
    file = "fastkpc/build/fastkpc_cuda.so", algo = "sha256",
    serialize = FALSE
  )),
  source_receipt = normalizePath(input_path),
  rows = rows,
  details = details,
  resource_before = resource_before,
  resource_after = resource_after,
  summary = list(
    case_count = nrow(rows),
    default_bitwise_exact_count = sum(rows$default_bitwise_exact),
    certificate_pass_count = sum(rows$certificate_passed),
    certificate_failure_count = sum(!rows$certificate_passed),
    fixed_p_contract_count = sum(rows$mgcv_p_contract_passed),
    certificate_semantic_false_positive_count = sum(
      rows$certificate_semantic_false_positive
    ),
    fail_closed_expected_count = sum(rows$fail_closed_expected),
    fail_closed_observed_count = sum(
      rows$fail_closed_expected & rows$fail_closed_observed
    ),
    fail_closed_gate = all(
      rows$fail_closed_observed & rows$failed_shadow_suppressed
    ),
    max_initial_kkt = max(rows$initial_kkt_max),
    max_final_kkt = max(rows$final_kkt_max),
    max_residual_error = max(rows$max_residual_error),
    max_abs_fixed_p_error = max(abs(rows$fixed_p_error)),
    max_abs_default_p_error = max(abs(rows$default_p_error)),
    cpu_fallback_count = sum(rows$cpu_fallback_count),
    unknown_fallback_count = sum(rows$unknown_fallback_count),
    resources_released = all(vapply(active_fields, function(field) {
      identical(resource_after[[field]], resource_before[[field]])
    }, logical(1L))) &&
      identical(resource_after$cleanup_error_count,
                resource_before$cleanup_error_count)
  ),
  route_decision =
    "STOP_KKT_CERTIFICATE_NOT_SUFFICIENT_FOR_MGCV_SEMANTICS"
)
receipt$diagnostic_pass <-
  isTRUE(receipt$summary$fail_closed_gate) &&
  receipt$summary$certificate_failure_count > 0L &&
  receipt$summary$certificate_semantic_false_positive_count > 0L &&
  receipt$summary$cpu_fallback_count == 0L &&
  receipt$summary$unknown_fallback_count == 0L &&
  isTRUE(receipt$summary$resources_released)
saveRDS(receipt, output_path, compress = "xz", version = 3L)
print(receipt$summary, digits = 17)
cat("route decision:", receipt$route_decision, "\n")
cat("diagnostic pass:", receipt$diagnostic_pass, "\n")
