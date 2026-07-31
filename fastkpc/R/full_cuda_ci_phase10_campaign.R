fastkpc_full_cuda_phase10_campaign_require <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
  invisible(TRUE)
}

fastkpc_full_cuda_phase10_campaign_json_equivalent <- function(left, right) {
  options <- list(
    auto_unbox = TRUE, null = "null", na = "null", digits = NA
  )
  identical(
    do.call(jsonlite::toJSON, c(list(x = left), options)),
    do.call(jsonlite::toJSON, c(list(x = right), options))
  )
}

fastkpc_full_cuda_phase10_campaign_artifact_dir <- function(
    root = file.path("fastkpc", "artifacts", "full_cuda_ci")) {
  file.path(root, "promotion_351x48_v1")
}

fastkpc_full_cuda_phase10_campaign_staging_dir <- function(
    root = file.path("fastkpc", "artifacts", "full_cuda_ci")) {
  file.path(root, "phase10_promotion_staging_v1")
}

fastkpc_full_cuda_phase10_campaign_required_files <- function() {
  c(
    "manifest.json", "summary.json", "summary.md", "commands.txt",
    "environment.txt", "graph_agreement.csv", "sepset_agreement.csv",
    "n_edgetests.csv", "deletion_trace.csv", "first_divergence.json",
    "fallbacks.csv", "stage_timing.csv", "raw_runs.csv",
    "statistics.csv", "case_results.csv", "near_alpha_results.csv",
    "rank_condition_results.csv", "cache.csv", "machine_samples.csv",
    "campaign_order.csv", "evidence_inputs.csv", "source_closure.csv",
    "source_evidence.rds", "producer_identity.json",
    "backend_configuration.json", "build_recipe.json",
    "freeze.json", "validator_attestations.json", "execution_receipts.json",
    "adjacency.rds", "sepsets.rds", "pmax.rds", "logical_ci_trace.rds"
  )
}

fastkpc_full_cuda_phase10_campaign_source_paths <- function() {
  paths <- sort(unique(c(
    fastkpc_full_cuda_phase10_hardening_source_paths(),
    "fastkpc/R/full_cuda_ci_phase10_campaign.R",
    "fastkpc/tests/test_full_cuda_ci_phase10_campaign_artifact.R",
    "fastkpc/tests/test_full_cuda_ci_phase10_campaign_helpers.R",
    "fastkpc/tools/run_full_cuda_ci_phase10_campaign.R",
    "fastkpc/tools/run_full_cuda_ci_phase10_campaign.sh",
    "fastkpc/tools/run_full_cuda_ci_phase10_worker.R",
    "fastkpc/tools/run_full_cuda_ci_gate.sh"
  )), method = "radix")
  fastkpc_full_cuda_phase10_campaign_require(
    all(file.exists(paths) & !dir.exists(paths)) && !anyDuplicated(paths),
    "Phase 10 campaign source closure is incomplete"
  )
  paths
}

fastkpc_full_cuda_phase10_campaign_source_closure <- function() {
  paths <- fastkpc_full_cuda_phase10_campaign_source_paths()
  hashes <- setNames(as.list(vapply(
    paths, fastkpc_full_cuda_census_file_hash, character(1L)
  )), paths)
  list(
    table = data.frame(
      path = names(hashes),
      sha256 = unlist(hashes, use.names = FALSE),
      stringsAsFactors = FALSE
    ),
    hashes = hashes,
    sha256 = fastkpc_full_cuda_phase35_sha256_utf8(
      fastkpc_full_cuda_phase35_canonical_json(hashes)
    )
  )
}

fastkpc_full_cuda_phase10_campaign_paths <- function() {
  c(
    data = file.path(
      "fastkpc", "artifacts", "kpc_tprs_real_zhu",
      "cancer_RD-causalDiscoveryInput.rds"
    ),
    oracle_manifest = file.path(
      "fastkpc", "artifacts", "full_cuda_ci", "oracle_351x48_v1",
      "manifest.json"
    ),
    logical = file.path(
      "fastkpc", "artifacts", "full_cuda_ci",
      "workload_census_351x48_v1", "logical_ci_tests.rds"
    ),
    hardening_manifest = file.path(
      "fastkpc", "artifacts", "full_cuda_ci", "failure_injection_v1",
      "manifest.json"
    ),
    phase8_rank_condition = file.path(
      "fastkpc", "artifacts", "full_cuda_ci", "dcov_cuda_backend_v1",
      "rank_condition_results.csv"
    )
  )
}

fastkpc_full_cuda_phase10_campaign_backend_configuration <- function() {
  value <- list(
    schema_version = "full-cuda-ci-phase10-campaign-configuration-v1",
    candidate_route = "compatible.cuda/full_cuda-explicit",
    native_entrypoint = "compatible-cuda-full-skeleton-native-v1",
    scheduler = "cache-aware-frontier-4x-v1",
    compact_result_cache_capacity = 262144L,
    target_state_cache_capacity = 131072L,
    prepared_setup_cache_capacity = 64L,
    component_capacity = 47L,
    candidate_cold_repetitions = 5L,
    candidate_warm_repetitions = 5L,
    unmeasured_complete_warmups_per_warm_repetition = 1L,
    correct_baseline_repetitions = 5L,
    baseline_route = "legacy-mgcv-provider-native-legacy-dcov-20-core",
    baseline_dcov_batch = "round",
    baseline_dcov_low_rank = "spectra",
    cpu_affinity = "0-19",
    blas_threads = 1L,
    lapack_threads = 1L,
    openmp_threads = 1L,
    control_plane_threads = 1L,
    gpu_device_id = 0L,
    alpha = "0.1",
    index = 1L,
    num_col = 35L,
    maximum_conditioning_size = 7L,
    trace_level = "logical",
    precision = "float64",
    fmad = FALSE,
    fast_math = FALSE,
    explicit_route_only = TRUE,
    phase10_holdout_claim = FALSE
  )
  list(
    value = value,
    sha256 = fastkpc_full_cuda_phase35_sha256_utf8(
      fastkpc_full_cuda_phase35_canonical_json(value)
    )
  )
}

fastkpc_full_cuda_phase10_campaign_build_recipe <- function() {
  value <- list(
    schema_version = "full-cuda-ci-phase10-campaign-build-recipe-v1",
    build_script_sha256 = fastkpc_full_cuda_census_file_hash(
      "fastkpc/tools/build_cuda_native.sh"
    ),
    CXX17 = fastkpc_full_cuda_command_output(
      "R", c("CMD", "config", "CXX17")
    ),
    R_version = R.version.string,
    nvcc_path = "/usr/local/cuda/bin/nvcc",
    nvcc_version = fastkpc_full_cuda_command_output(
      "/usr/local/cuda/bin/nvcc", "--version"
    ),
    cuda_architecture = "sm_89",
    fmad = FALSE,
    fast_math = FALSE
  )
  list(
    value = value,
    sha256 = fastkpc_full_cuda_phase35_sha256_utf8(
      fastkpc_full_cuda_phase35_canonical_json(value)
    )
  )
}

fastkpc_full_cuda_phase10_campaign_statistics <- function(values) {
  values <- as.numeric(values)
  fastkpc_full_cuda_phase10_campaign_require(
    length(values) > 0L && all(is.finite(values) & values > 0),
    "Phase 10 campaign timing vector is invalid"
  )
  data.frame(
    repetitions = length(values),
    median_sec = stats::median(values),
    min_sec = min(values),
    max_sec = max(values),
    mad_sec = stats::mad(values, constant = 1),
    iqr_sec = stats::IQR(values),
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_phase10_campaign_command <- function(
    command, args = character()) {
  output <- tryCatch(
    suppressWarnings(system2(command, args, stdout = TRUE, stderr = TRUE)),
    error = function(error) structure(
      paste0("unavailable: ", conditionMessage(error)), status = 127L
    )
  )
  status <- attr(output, "status", exact = TRUE)
  list(
    status = if (is.null(status)) 0L else as.integer(status),
    output = as.character(output)
  )
}

fastkpc_full_cuda_phase10_campaign_gpu_rows <- function() {
  value <- fastkpc_full_cuda_phase10_campaign_command(
    "nvidia-smi",
    c(
      "--query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total,power.limit,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory,persistence_mode,compute_mode",
      "--format=csv,noheader,nounits"
    )
  )
  fastkpc_full_cuda_phase10_campaign_require(
    value$status == 0L && length(value$output) >= 1L,
    "Phase 10 campaign cannot query GPU identity"
  )
  rows <- lapply(value$output, function(line) trimws(strsplit(
    line, ",", fixed = TRUE
  )[[1L]]))
  fastkpc_full_cuda_phase10_campaign_require(
    all(vapply(rows, length, integer(1L)) == 13L),
    "Phase 10 GPU identity query shape changed"
  )
  do.call(rbind, lapply(rows, function(row) data.frame(
    index = as.integer(row[[1L]]),
    name = row[[2L]],
    uuid = row[[3L]],
    pci_bus_id = row[[4L]],
    driver_version = row[[5L]],
    memory_total_mib = as.numeric(row[[6L]]),
    power_limit_watts = as.numeric(row[[7L]]),
    temperature_c = as.numeric(row[[8L]]),
    power_draw_watts = as.numeric(row[[9L]]),
    sm_clock_mhz = as.numeric(row[[10L]]),
    memory_clock_mhz = as.numeric(row[[11L]]),
    persistence_mode = row[[12L]],
    compute_mode = row[[13L]],
    stringsAsFactors = FALSE
  )))
}

fastkpc_full_cuda_phase10_campaign_compute_processes <- function() {
  value <- fastkpc_full_cuda_phase10_campaign_command(
    "nvidia-smi",
    c(
      "--query-compute-apps=pid,gpu_uuid,used_memory",
      "--format=csv,noheader,nounits"
    )
  )
  fastkpc_full_cuda_phase10_campaign_require(
    value$status == 0L,
    "Phase 10 campaign cannot query concurrent GPU processes"
  )
  if (length(value$output) == 0L ||
      all(!nzchar(trimws(value$output)))) {
    return(data.frame(
      pid = integer(), gpu_uuid = character(), used_memory_mib = numeric(),
      stringsAsFactors = FALSE
    ))
  }
  rows <- lapply(value$output[nzchar(trimws(value$output))], function(line) {
    trimws(strsplit(line, ",", fixed = TRUE)[[1L]])
  })
  fastkpc_full_cuda_phase10_campaign_require(
    all(vapply(rows, length, integer(1L)) == 3L),
    "Phase 10 concurrent GPU process query shape changed"
  )
  do.call(rbind, lapply(rows, function(row) data.frame(
    pid = as.integer(row[[1L]]),
    gpu_uuid = row[[2L]],
    used_memory_mib = as.numeric(row[[3L]]),
    stringsAsFactors = FALSE
  )))
}

fastkpc_full_cuda_phase10_campaign_affinity <- function() {
  status <- readLines("/proc/self/status", warn = FALSE)
  line <- status[startsWith(status, "Cpus_allowed_list:")]
  fastkpc_full_cuda_phase10_campaign_require(
    length(line) == 1L,
    "Phase 10 campaign cannot read process CPU affinity"
  )
  trimws(sub("^[^:]+:", "", line))
}

fastkpc_full_cuda_phase10_campaign_governor <- function() {
  paths <- Sys.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
  if (length(paths) == 0L) return("unavailable")
  values <- sort(unique(unlist(lapply(paths, function(path) {
    trimws(readLines(path, warn = FALSE, n = 1L))
  }), use.names = FALSE)), method = "radix")
  paste(values, collapse = "|")
}

fastkpc_full_cuda_phase10_campaign_normalize_uuid <- function(value) {
  tolower(gsub("-", "", as.character(value), fixed = TRUE))
}

fastkpc_full_cuda_phase10_campaign_machine_snapshot <- function(
    require_idle_gpu = FALSE,
    contracts = fastkpc_full_cuda_phase35_load_contract_set()) {
  reference <- contracts$reference_machine_v1$payload
  gpu <- fastkpc_full_cuda_phase10_campaign_gpu_rows()
  selected <- gpu[gpu$index == 0L, , drop = FALSE]
  cpu_model_lines <- readLines("/proc/cpuinfo", warn = FALSE)
  cpu_model <- trimws(sub(
    "^[^:]+:", "", cpu_model_lines[startsWith(cpu_model_lines, "model name")][[1L]]
  ))
  meminfo <- readLines("/proc/meminfo", warn = FALSE)
  mem_kib <- as.numeric(sub(
    "[^0-9].*$", "", sub("^[^:]+:[[:space:]]*", "",
                           meminfo[startsWith(meminfo, "MemTotal:")][[1L]])
  ))
  compute <- fastkpc_full_cuda_phase10_campaign_compute_processes()
  env_names <- c(
    "CUDA_VISIBLE_DEVICES", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
    "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"
  )
  env_values <- Sys.getenv(env_names, unset = "")
  identity_gate <- nrow(selected) == 1L &&
    identical(Sys.info()[["nodename"]], reference$host$hostname) &&
    identical(R.version$arch, reference$host$architecture) &&
    identical(system("uname -r", intern = TRUE),
              reference$host$kernel_release) &&
    identical(mem_kib * 1024,
              as.numeric(reference$host$physical_memory_bytes)) &&
    identical(cpu_model, reference$cpu$model) &&
    identical(fastkpc_full_cuda_phase10_campaign_affinity(),
              reference$cpu$benchmark_affinity) &&
    identical(selected$name[[1L]], reference$gpu$model) &&
    identical(
      fastkpc_full_cuda_phase10_campaign_normalize_uuid(selected$uuid[[1L]]),
      fastkpc_full_cuda_phase10_campaign_normalize_uuid(reference$gpu$uuid)
    ) &&
    identical(tolower(selected$pci_bus_id[[1L]]),
              tolower(reference$gpu$pci_bus_id)) &&
    identical(selected$driver_version[[1L]], reference$gpu$driver_version) &&
    identical(
      as.numeric(selected$memory_total_mib[[1L]] * 1024^2),
      as.numeric(reference$gpu$total_memory_bytes)
    ) &&
    identical(selected$power_limit_watts[[1L]],
              as.numeric(reference$gpu$power_limit_watts)) &&
    identical(as.character(getRversion()), reference$software$R_version) &&
    identical(as.character(utils::packageVersion("mgcv")),
              reference$software$mgcv_version) &&
    identical(unname(env_values), c("0", "1", "1", "1", "1", "1"))
  idle_gate <- !isTRUE(require_idle_gpu) || nrow(compute) == 0L
  fastkpc_full_cuda_phase10_campaign_require(
    identity_gate && idle_gate,
    if (!identity_gate) {
      "Phase 10 reference machine or execution environment drifted"
    } else {
      "Phase 10 GPU idle precondition failed"
    }
  )
  list(
    schema_version = "full-cuda-ci-phase10-machine-snapshot-v1",
    captured_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
    hostname = Sys.info()[["nodename"]],
    architecture = R.version$arch,
    kernel_release = system("uname -r", intern = TRUE),
    physical_memory_bytes = mem_kib * 1024,
    cpu_model = cpu_model,
    affinity = fastkpc_full_cuda_phase10_campaign_affinity(),
    governor = fastkpc_full_cuda_phase10_campaign_governor(),
    gpu = selected,
    compute_processes = compute,
    environment = stats::setNames(as.list(unname(env_values)), env_names),
    identity_gate = identity_gate,
    idle_gate = idle_gate,
    pass = TRUE
  )
}

fastkpc_full_cuda_phase10_campaign_machine_identity <- function(snapshot) {
  list(
    schema_version = snapshot$schema_version,
    hostname = snapshot$hostname,
    architecture = snapshot$architecture,
    kernel_release = snapshot$kernel_release,
    physical_memory_bytes = snapshot$physical_memory_bytes,
    cpu_model = snapshot$cpu_model,
    affinity = snapshot$affinity,
    governor = snapshot$governor,
    gpu_name = snapshot$gpu$name[[1L]],
    gpu_uuid = snapshot$gpu$uuid[[1L]],
    gpu_pci_bus_id = snapshot$gpu$pci_bus_id[[1L]],
    driver_version = snapshot$gpu$driver_version[[1L]],
    gpu_memory_total_mib = snapshot$gpu$memory_total_mib[[1L]],
    gpu_power_limit_watts = snapshot$gpu$power_limit_watts[[1L]],
    environment = snapshot$environment
  )
}

fastkpc_full_cuda_phase10_campaign_order <- function() {
  rows <- lapply(seq_len(5L), function(repetition) data.frame(
    mode = "candidate_cold",
    repetition = repetition,
    stringsAsFactors = FALSE
  ))
  for (repetition in seq_len(5L)) {
    pair <- if (repetition %% 2L == 1L) {
      c("candidate_warm", "correct_baseline")
    } else {
      c("correct_baseline", "candidate_warm")
    }
    rows <- c(rows, lapply(pair, function(mode) data.frame(
      mode = mode,
      repetition = repetition,
      stringsAsFactors = FALSE
    )))
  }
  value <- do.call(rbind, rows)
  value$sequence <- seq_len(nrow(value))
  value <- value[, c("sequence", "mode", "repetition")]
  rownames(value) <- NULL
  value
}

fastkpc_full_cuda_phase10_campaign_order_records <- function() {
  value <- fastkpc_full_cuda_phase10_campaign_order()
  unname(lapply(seq_len(nrow(value)), function(index) list(
    sequence = as.integer(value$sequence[[index]]),
    mode = as.character(value$mode[[index]]),
    repetition = as.integer(value$repetition[[index]])
  )))
}

fastkpc_full_cuda_phase10_campaign_run_key <- function(mode, repetition) {
  fastkpc_full_cuda_phase10_campaign_require(
    mode %in% c("candidate_cold", "candidate_warm", "correct_baseline") &&
      length(repetition) == 1L && !is.na(repetition) &&
      repetition >= 1L && repetition <= 5L,
    "Phase 10 campaign run identity is invalid"
  )
  sprintf("%s-%02d", mode, as.integer(repetition))
}

fastkpc_full_cuda_phase10_campaign_run_path <- function(
    staging_dir, mode, repetition) {
  file.path(
    staging_dir, "runs",
    paste0(fastkpc_full_cuda_phase10_campaign_run_key(mode, repetition),
           ".rds")
  )
}

fastkpc_full_cuda_phase10_campaign_current_freeze <- function(
    machine_snapshot = fastkpc_full_cuda_phase10_campaign_machine_snapshot(
      require_idle_gpu = TRUE
    )) {
  paths <- fastkpc_full_cuda_phase10_campaign_paths()
  fastkpc_full_cuda_phase10_campaign_require(
    all(file.exists(paths) & !dir.exists(paths)),
    "Phase 10 campaign freeze input is missing"
  )
  hardening <- fastkpc_full_cuda_phase10_validate_hardening_artifact(
    dirname(paths[["hardening_manifest"]]), verify_current_sources = TRUE
  )
  source_closure <- fastkpc_full_cuda_phase10_campaign_source_closure()
  native_identity <- fastkpc_full_cuda_phase7_native_identity()
  backend <- fastkpc_full_cuda_phase10_campaign_backend_configuration()
  build <- fastkpc_full_cuda_phase10_campaign_build_recipe()
  contracts <- fastkpc_full_cuda_phase35_load_contract_set()
  contract_hashes <- setNames(lapply(
    contracts, `[[`, "sha256"
  ), names(contracts))
  core <- list(
    schema_version = "full-cuda-ci-phase10-campaign-freeze-v1",
    source_commit = fastkpc_full_cuda_source_commit(),
    source_closure_sha256 = source_closure$sha256,
    native_binary_path = native_identity$path,
    native_binary_sha256 = native_identity$sha256,
    backend_configuration_sha256 = backend$sha256,
    build_recipe_sha256 = build$sha256,
    contract_sha256 = contract_hashes,
    canonical_input_sha256 = setNames(lapply(
      paths, fastkpc_full_cuda_census_file_hash
    ), names(paths)),
    hardening_producer_identity_sha256 =
      hardening$producer$identity_sha256,
    hardening_source_closure_sha256 =
      hardening$producer$producer_source_closure_sha256,
    machine_identity =
      fastkpc_full_cuda_phase10_campaign_machine_identity(machine_snapshot),
    campaign_order = fastkpc_full_cuda_phase10_campaign_order_records(),
    candidate_cold_repetitions = 5L,
    candidate_warm_repetitions = 5L,
    correct_baseline_repetitions = 5L,
    holdout_state =
      contracts$promotion_holdout_manifest_v1$payload$state,
    holdout_opened = FALSE
  )
  core$freeze_identity_sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(core)
  )
  list(
    freeze = core,
    source_closure = source_closure,
    backend = backend,
    build = build,
    contracts = contracts,
    hardening = hardening,
    machine_snapshot = machine_snapshot
  )
}

fastkpc_full_cuda_phase10_campaign_validate_freeze <- function(
    freeze, verify_current = FALSE, require_idle_gpu = FALSE) {
  fastkpc_full_cuda_phase10_campaign_require(
    is.list(freeze) && identical(
      freeze$schema_version, "full-cuda-ci-phase10-campaign-freeze-v1"
    ) && grepl("^[0-9a-f]{64}$", freeze$source_closure_sha256) &&
      grepl("^[0-9a-f]{64}$", freeze$native_binary_sha256) &&
      grepl("^[0-9a-f]{64}$", freeze$backend_configuration_sha256) &&
      grepl("^[0-9a-f]{64}$", freeze$build_recipe_sha256) &&
      is.list(freeze$contract_sha256) &&
      is.list(freeze$canonical_input_sha256) &&
      is.list(freeze$campaign_order) &&
      identical(freeze$campaign_order,
                fastkpc_full_cuda_phase10_campaign_order_records()) &&
      freeze$candidate_cold_repetitions == 5L &&
      freeze$candidate_warm_repetitions == 5L &&
      freeze$correct_baseline_repetitions == 5L &&
      identical(freeze$holdout_state, "SEALED_NOT_RELEASED") &&
      !isTRUE(freeze$holdout_opened),
    "Phase 10 campaign freeze schema is malformed"
  )
  claimed <- freeze$freeze_identity_sha256
  core <- freeze
  core$freeze_identity_sha256 <- NULL
  expected <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(core)
  )
  fastkpc_full_cuda_phase10_campaign_require(
    identical(claimed, expected),
    "Phase 10 campaign freeze identity mismatch"
  )
  if (isTRUE(verify_current)) {
    current <- fastkpc_full_cuda_phase10_campaign_current_freeze(
      machine_snapshot = fastkpc_full_cuda_phase10_campaign_machine_snapshot(
        require_idle_gpu = require_idle_gpu
      )
    )$freeze
    fastkpc_full_cuda_phase10_campaign_require(
      identical(current, freeze),
      "Phase 10 campaign source, configuration, contract, or machine drifted"
    )
  }
  invisible(freeze)
}

fastkpc_full_cuda_phase10_campaign_prepare_staging <- function(
    staging_dir = fastkpc_full_cuda_phase10_campaign_staging_dir()) {
  dir.create(staging_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(staging_dir, "runs"), showWarnings = FALSE)
  freeze_path <- file.path(staging_dir, "freeze.rds")
  freeze_json_path <- file.path(staging_dir, "freeze.json")
  if (file.exists(freeze_path)) {
    freeze <- readRDS(freeze_path)
    fastkpc_full_cuda_phase10_campaign_validate_freeze(
      freeze, verify_current = TRUE, require_idle_gpu = TRUE
    )
    fastkpc_full_cuda_phase10_campaign_require(
      file.exists(freeze_json_path),
      "Phase 10 campaign JSON freeze receipt is missing"
    )
    return(invisible(freeze))
  }
  existing_runs <- list.files(
    file.path(staging_dir, "runs"), pattern = "[.]rds$", full.names = TRUE
  )
  fastkpc_full_cuda_phase10_campaign_require(
    length(existing_runs) == 0L,
    "Phase 10 campaign cannot create a new freeze over existing runs"
  )
  current <- fastkpc_full_cuda_phase10_campaign_current_freeze()
  saveRDS(current$freeze, freeze_path, compress = "xz")
  fastkpc_full_cuda_write_json(current$freeze, freeze_json_path)
  utils::write.csv(
    current$source_closure$table,
    file.path(staging_dir, "source_closure.csv"),
    row.names = FALSE, na = ""
  )
  fastkpc_full_cuda_write_json(
    current$backend$value,
    file.path(staging_dir, "backend_configuration.json")
  )
  fastkpc_full_cuda_write_json(
    current$build$value, file.path(staging_dir, "build_recipe.json")
  )
  utils::write.csv(
    fastkpc_full_cuda_phase10_campaign_order(),
    file.path(staging_dir, "campaign_order.csv"),
    row.names = FALSE, na = ""
  )
  invisible(current$freeze)
}

fastkpc_full_cuda_phase10_campaign_authority_zero_fields <- function() {
  c(
    "r_callback_count", "legacy_mgcv_fit_count",
    "legacy_mgcv_setup_count", "cpu_residual_solve_count",
    "cpu_dcov_component_count", "cpu_dcov_eigen_or_lowrank_count",
    "cpu_dcov_pair_stat_count", "cpu_gamma_pvalue_count",
    "cpu_spectra_count", "residual_d2h_bytes", "component_d2h_bytes",
    "unknown_fallback_count", "approximate_backend_count"
  )
}

fastkpc_full_cuda_phase10_validate_candidate_result <- function(
    result, boundary = c("cold", "warm"),
    oracle_dir = file.path(
      "fastkpc", "artifacts", "full_cuda_ci", "oracle_351x48_v1"
    ),
    logical_path = file.path(
      "fastkpc", "artifacts", "full_cuda_ci",
      "workload_census_351x48_v1", "logical_ci_tests.rds"
    )) {
  boundary <- match.arg(boundary)
  fastkpc_full_cuda_phase10_campaign_require(
    fastkpc_full_cuda_is_skeleton(result) && file.exists(logical_path),
    "Phase 10 candidate result or canonical trace is missing"
  )
  oracle <- fastkpc_load_full_cuda_ci_oracle(oracle_dir)
  comparison <- fastkpc_full_cuda_compare_candidate_skeleton(oracle, result)
  logical <- readRDS(logical_path)
  tasks <- result$tasks
  expected_n_edgetests <- c(
    2213L, 52659L, 125293L, 40694L, 13293L, 5422L, 835L, 80L
  )
  structural_fields_exact <- c(
    logical_id = identical(
      as.integer(tasks$canonical_test_order_id),
      as.integer(logical$logical_sequence_id)
    ),
    level = identical(as.integer(tasks$level), as.integer(logical$level)),
    task_index = identical(
      as.integer(tasks$task_index), as.integer(logical$source_task_index)
    ),
    x = identical(as.integer(tasks$x), as.integer(logical$x)),
    y = identical(as.integer(tasks$y), as.integer(logical$y)),
    S_key = identical(as.character(tasks$S_key), as.character(logical$S_key)),
    S_size = identical(
      as.integer(tasks$conditioning_size), as.integer(logical$S_size)
    ),
    deletion = identical(
      as.logical(tasks$native_edge_deleted), as.logical(logical$deletes_edge)
    )
  )
  decision_flip <- (tasks$p_used >= 0.1) != logical$reference_independent
  zero_fields <- fastkpc_full_cuda_phase10_campaign_authority_zero_fields()
  zero_values <- vapply(zero_fields, function(field) {
    value <- result$summary[[field]]
    if (is.null(value)) return(NA_real_)
    as.numeric(value)
  }, numeric(1L))
  summary <- result$summary
  common_gate <- isTRUE(comparison$summary$pass) &&
    identical(as.integer(result$n.edgetests), expected_n_edgetests) &&
    nrow(tasks) == 240489L && all(structural_fields_exact) &&
    !any(decision_flip) && all(is.finite(tasks$p_used)) &&
    all(!is.na(zero_values) & zero_values == 0) &&
    identical(summary$run_status, "ok") &&
    identical(summary$entrypoint,
              "compatible-cuda-full-skeleton-native-v1") &&
    identical(summary$compatible_cuda_route, "compatible.cuda") &&
    isTRUE(summary$compatible_cuda_strict) &&
    isTRUE(summary$authority_gate_pass) &&
    as.integer(summary$native_call_count) == 1L &&
    as.integer(summary$logical_tests_consumed) == 240489L &&
    as.integer(summary$speculative_tests_ignored) == 0L &&
    identical(summary$scheduler, "cache-aware-frontier-4x-v1") &&
    as.integer(summary$frontier_batch_count) == sum(result$levels$rounds) &&
    as.integer(summary$result_cache_capacity) == 262144L &&
    as.integer(summary$target_cache_capacity) == 131072L &&
    as.integer(summary$result_cache_request_count) == 240489L &&
    as.integer(summary$result_cache_request_count) ==
      as.integer(summary$result_cache_hit_count) +
        as.integer(summary$result_cache_miss_count) &&
    as.integer(summary$target_cache_request_count) ==
      as.integer(summary$target_cache_hit_count) +
        as.integer(summary$target_cache_miss_count) &&
    as.integer(summary$native_setup_cache_request_count) ==
      as.integer(summary$native_setup_cache_hit_count) +
        as.integer(summary$native_setup_cache_miss_count) &&
    is.finite(summary$elapsed_sec) && summary$elapsed_sec > 0
  boundary_gate <- if (identical(boundary, "cold")) {
    as.integer(summary$physical_tests_evaluated) == 241677L &&
      as.integer(summary$guarded_pair_count) == 1188L &&
      as.integer(summary$cuda_dcov_pair_count) == 241677L &&
      as.integer(summary$cuda_gamma_pvalue_count) == 241677L &&
      as.integer(summary$result_cache_warm_start_entries) == 0L &&
      as.integer(summary$result_cache_hit_count) == 0L &&
      as.integer(summary$result_cache_miss_count) == 240489L &&
      as.integer(summary$result_cache_insert_count) == 240489L &&
      as.integer(summary$result_cache_eviction_count) == 0L &&
      as.integer(summary$target_cache_warm_start_entries) == 0L &&
      as.integer(summary$target_cache_miss_count) == 110617L &&
      as.integer(summary$target_cache_insert_count) == 110617L &&
      as.integer(summary$target_cache_eviction_count) == 0L &&
      as.integer(summary$native_setup_count) ==
        as.integer(summary$native_setup_cache_miss_count) &&
      as.integer(summary$cuda_single_penalty_target_count) > 0L &&
      as.integer(summary$cuda_multi_penalty_target_count) > 0L &&
      as.integer(summary$physical_residual_fits) > 0L
  } else {
    as.integer(summary$physical_tests_evaluated) == 0L &&
      as.integer(summary$guarded_pair_count) == 0L &&
      as.integer(summary$cuda_dcov_pair_count) == 0L &&
      as.integer(summary$cuda_gamma_pvalue_count) == 0L &&
      as.integer(summary$result_cache_warm_start_entries) >= 240489L &&
      as.integer(summary$result_cache_hit_count) == 240489L &&
      as.integer(summary$result_cache_miss_count) == 0L &&
      as.integer(summary$result_cache_insert_count) == 0L &&
      as.integer(summary$result_cache_eviction_count) == 0L &&
      as.integer(summary$target_cache_warm_start_entries) == 110617L &&
      as.integer(summary$target_cache_request_count) == 0L &&
      as.integer(summary$native_setup_count) == 0L &&
      as.integer(summary$cuda_single_penalty_target_count) == 0L &&
      as.integer(summary$cuda_multi_penalty_target_count) == 0L &&
      as.integer(summary$physical_residual_fits) == 0L
  }
  fastkpc_full_cuda_phase10_campaign_require(
    common_gate && boundary_gate,
    paste0("Phase 10 ", boundary,
           " candidate correctness or authority gate failed")
  )
  list(
    schema_version = "full-cuda-ci-phase10-candidate-validation-v1",
    boundary = boundary,
    comparison_summary = comparison$summary,
    first_divergence = comparison$first_divergence,
    structural_fields_exact = structural_fields_exact,
    decision_flip_count = sum(decision_flip),
    authority_zero_values = zero_values,
    cache_gate = TRUE,
    pass = TRUE
  )
}

fastkpc_full_cuda_phase10_validate_baseline_result <- function(
    result,
    oracle_dir = file.path(
      "fastkpc", "artifacts", "full_cuda_ci", "oracle_351x48_v1"
    )) {
  fastkpc_full_cuda_phase10_campaign_require(
    fastkpc_full_cuda_is_skeleton(result),
    "Phase 10 correct baseline result is missing"
  )
  comparison <- fastkpc_full_cuda_compare_candidate_skeleton(
    fastkpc_load_full_cuda_ci_oracle(oracle_dir), result
  )
  summary <- result$summary
  clean <- isTRUE(comparison$summary$pass) &&
    identical(as.integer(result$n.edgetests), c(
      2213L, 52659L, 125293L, 40694L, 13293L, 5422L, 835L, 80L
    )) && nrow(result$tasks) == 240489L &&
    all(is.finite(result$tasks$p_used)) &&
    identical(summary$compatible_cuda_route,
              "legacy-mgcv-provider-native-legacy-dcov") &&
    identical(summary$compatible_cuda_residual_authority,
              "legacy-mgcv-regrXonS-provider") &&
    identical(summary$compatible_cuda_ci_authority,
              "native-legacy-dcov.gamma") &&
    isTRUE(summary$residual_provider_parallel_enabled) &&
    as.integer(summary$residual_provider_parallel_cores) == 20L &&
    isTRUE(summary$legacy_dcov_native_batch_enabled) &&
    identical(summary$legacy_dcov_native_batch_mode, "round") &&
    as.integer(summary$legacy_dcov_native_batch_pair_count) == 240489L &&
    as.integer(summary$unknown_fallback_count) == 0L &&
    as.integer(summary$approximate_backend_count) == 0L
  fastkpc_full_cuda_phase10_campaign_require(
    clean, "Phase 10 same-campaign correct baseline gate failed"
  )
  list(
    schema_version = "full-cuda-ci-phase10-baseline-validation-v1",
    comparison_summary = comparison$summary,
    first_divergence = comparison$first_divergence,
    pass = TRUE
  )
}

fastkpc_full_cuda_phase10_same_candidate_result <- function(left, right) {
  identical(left$adjacency, right$adjacency) &&
    identical(
      fastkpc_full_cuda_normalize_sepsets(left),
      fastkpc_full_cuda_normalize_sepsets(right)
    ) &&
    identical(as.integer(left$n.edgetests), as.integer(right$n.edgetests)) &&
    identical(left$tasks, right$tasks) && identical(left$pMax, right$pMax)
}

fastkpc_full_cuda_phase10_campaign_resource_snapshot <- function() {
  .Call("C_fixed_sp_cuda_test_resource_snapshot", PACKAGE = "fastkpc_cuda")
}

fastkpc_full_cuda_phase10_campaign_active_resources <- function(snapshot) {
  fields <- grep("_active_count$", names(snapshot), value = TRUE)
  stats::setNames(
    as.numeric(unlist(snapshot[fields], use.names = FALSE)), fields
  )
}

fastkpc_full_cuda_phase10_campaign_cache_control <- function(
    action, capacity = NULL) {
  invisible(full_cuda_ci_one_call_cache_control_native(action, capacity))
}

fastkpc_full_cuda_phase10_campaign_candidate_call <- function(data) {
  fastkpc_compatible_cuda_skeleton(
    data = data,
    alpha = 0.1,
    labels = colnames(data),
    options = list(
      route = "full_cuda",
      compatible_cuda_strict = TRUE,
      max_conditioning_size = 7L,
      index = 1,
      numCol = 35L,
      trace_level = "logical"
    )
  )
}

fastkpc_full_cuda_phase10_campaign_baseline_call <- function(data) {
  expected_env <- c(
    FASTKPC_LEGACY_DCOV_GAMMA_BACKEND = "cpp",
    FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = "spectra",
    FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE = "1",
    FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY = "s",
    FASTKPC_NATIVE_LEGACY_MGCV_PROVIDER_CORES = "20",
    FASTKPC_NATIVE_LEGACY_DCOV_BATCH = "round",
    FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH_THREADS = "20"
  )
  fastkpc_full_cuda_phase10_campaign_require(
    identical(Sys.getenv(names(expected_env), unset = ""), expected_env),
    "Phase 10 correct baseline environment drifted"
  )
  fastkpc_compatible_cuda_skeleton(
    data = data,
    alpha = 0.1,
    labels = colnames(data),
    options = list(
      route = "legacy",
      compatible_cuda_strict = TRUE,
      max_conditioning_size = 7L,
      index = 1,
      numCol = 35L,
      trace_level = "logical",
      dcov_batch = "round",
      mgcv_residual_backend = "r"
    )
  )
}

fastkpc_full_cuda_phase10_campaign_timed_call <- function(expression) {
  started <- Sys.time()
  timing <- system.time(value <- force(expression))
  ended <- Sys.time()
  list(
    started_utc = format(started, tz = "UTC", usetz = TRUE),
    ended_utc = format(ended, tz = "UTC", usetz = TRUE),
    timing = timing,
    elapsed_sec = as.numeric(timing[["elapsed"]]),
    value = value
  )
}

fastkpc_full_cuda_phase10_capture_campaign_run <- function(
    mode, repetition, freeze) {
  fastkpc_full_cuda_phase10_campaign_run_key(mode, repetition)
  fastkpc_full_cuda_phase10_campaign_validate_freeze(
    freeze, verify_current = TRUE, require_idle_gpu = TRUE
  )
  paths <- fastkpc_full_cuda_phase10_campaign_paths()
  data <- readRDS(paths[["data"]])
  fastkpc_full_cuda_phase10_campaign_require(
    is.matrix(data) && identical(dim(data), c(351L, 48L)) &&
      !is.null(colnames(data)) && all(is.finite(data)),
    "Phase 10 campaign canonical data is malformed"
  )
  machine_before <- fastkpc_full_cuda_phase10_campaign_machine_snapshot(
    require_idle_gpu = TRUE
  )
  resource_before <- fastkpc_full_cuda_phase10_campaign_resource_snapshot()
  warmup <- NULL
  if (mode %in% c("candidate_cold", "candidate_warm")) {
    fastkpc_full_cuda_phase10_campaign_cache_control("configure", 262144L)
    fastkpc_full_cuda_phase10_campaign_cache_control(
      "configure_target", 131072L
    )
    fastkpc_full_cuda_phase10_campaign_cache_control("reset")
  }
  if (identical(mode, "candidate_warm")) {
    captured_warmup <- fastkpc_full_cuda_phase10_campaign_timed_call(
      fastkpc_full_cuda_phase10_campaign_candidate_call(data)
    )
    warmup_validation <- fastkpc_full_cuda_phase10_validate_candidate_result(
      captured_warmup$value, boundary = "cold"
    )
    warmup <- list(
      started_utc = captured_warmup$started_utc,
      ended_utc = captured_warmup$ended_utc,
      timing = captured_warmup$timing,
      elapsed_sec = captured_warmup$elapsed_sec,
      result = captured_warmup$value,
      validation = warmup_validation,
      pass = TRUE
    )
  }
  captured <- if (identical(mode, "correct_baseline")) {
    fastkpc_full_cuda_phase10_campaign_timed_call(
      fastkpc_full_cuda_phase10_campaign_baseline_call(data)
    )
  } else {
    fastkpc_full_cuda_phase10_campaign_timed_call(
      fastkpc_full_cuda_phase10_campaign_candidate_call(data)
    )
  }
  validation <- if (identical(mode, "correct_baseline")) {
    fastkpc_full_cuda_phase10_validate_baseline_result(captured$value)
  } else {
    fastkpc_full_cuda_phase10_validate_candidate_result(
      captured$value,
      boundary = if (identical(mode, "candidate_warm")) "warm" else "cold"
    )
  }
  if (identical(mode, "candidate_warm")) {
    fastkpc_full_cuda_phase10_campaign_require(
      fastkpc_full_cuda_phase10_same_candidate_result(
        warmup$result, captured$value
      ),
      "Phase 10 measured warm result differs from its complete warm-up"
    )
  }
  cache_after <- full_cuda_ci_one_call_cache_control_native("info")
  resource_after <- fastkpc_full_cuda_phase10_campaign_resource_snapshot()
  machine_after <- fastkpc_full_cuda_phase10_campaign_machine_snapshot(
    require_idle_gpu = FALSE
  )
  fastkpc_full_cuda_phase10_campaign_require(
    identical(
      fastkpc_full_cuda_phase10_campaign_machine_identity(machine_before),
      freeze$machine_identity
    ) && identical(
      fastkpc_full_cuda_phase10_campaign_machine_identity(machine_after),
      freeze$machine_identity
    ) && identical(
      fastkpc_full_cuda_phase10_campaign_active_resources(resource_before),
      fastkpc_full_cuda_phase10_campaign_active_resources(resource_after)
    ) && all(
      fastkpc_full_cuda_phase10_campaign_active_resources(resource_after) == 0
    ),
    "Phase 10 campaign machine drift or tracked resource leak detected"
  )
  list(
    schema_version = "full-cuda-ci-phase10-campaign-run-evidence-v1",
    run_key = fastkpc_full_cuda_phase10_campaign_run_key(mode, repetition),
    mode = mode,
    repetition = as.integer(repetition),
    freeze_identity_sha256 = freeze$freeze_identity_sha256,
    source_closure_sha256 = freeze$source_closure_sha256,
    native_binary_sha256 = freeze$native_binary_sha256,
    started_utc = captured$started_utc,
    ended_utc = captured$ended_utc,
    timing = captured$timing,
    elapsed_sec = captured$elapsed_sec,
    result = captured$value,
    validation = validation,
    warmup = warmup,
    cache_after = cache_after,
    machine_before = machine_before,
    machine_after = machine_after,
    resource_before = resource_before,
    resource_after = resource_after,
    pass = TRUE
  )
}

fastkpc_full_cuda_phase10_validate_campaign_run <- function(
    evidence, freeze, verify_current = FALSE) {
  fastkpc_full_cuda_phase10_campaign_require(
    is.list(evidence) && identical(
      evidence$schema_version,
      "full-cuda-ci-phase10-campaign-run-evidence-v1"
    ) && evidence$mode %in%
      c("candidate_cold", "candidate_warm", "correct_baseline") &&
      identical(
        evidence$run_key,
        fastkpc_full_cuda_phase10_campaign_run_key(
          evidence$mode, evidence$repetition
        )
      ) && identical(evidence$freeze_identity_sha256,
                    freeze$freeze_identity_sha256) &&
      identical(evidence$source_closure_sha256,
                freeze$source_closure_sha256) &&
      identical(evidence$native_binary_sha256,
                freeze$native_binary_sha256) &&
      is.finite(evidence$elapsed_sec) && evidence$elapsed_sec > 0 &&
      fastkpc_full_cuda_is_skeleton(evidence$result) &&
      isTRUE(evidence$validation$pass) && isTRUE(evidence$pass),
    "Phase 10 campaign run evidence schema is malformed"
  )
  if (identical(evidence$mode, "correct_baseline")) {
    rebuilt_validation <-
      fastkpc_full_cuda_phase10_validate_baseline_result(evidence$result)
    fastkpc_full_cuda_phase10_campaign_require(
      identical(rebuilt_validation, evidence$validation) &&
        is.null(evidence$warmup),
      "Phase 10 correct baseline must not contain candidate warm-up evidence"
    )
  } else {
    boundary <- if (identical(evidence$mode, "candidate_warm")) {
      "warm"
    } else "cold"
    rebuilt_validation <- fastkpc_full_cuda_phase10_validate_candidate_result(
      evidence$result, boundary = boundary
    )
    fastkpc_full_cuda_phase10_campaign_require(
      identical(rebuilt_validation, evidence$validation),
      "Phase 10 campaign stored candidate validation drifted"
    )
    if (identical(boundary, "warm")) {
      fastkpc_full_cuda_phase10_campaign_require(
        is.list(evidence$warmup) && isTRUE(evidence$warmup$pass) &&
          is.finite(evidence$warmup$elapsed_sec) &&
          evidence$warmup$elapsed_sec > 0 &&
          isTRUE(evidence$warmup$validation$pass) &&
          fastkpc_full_cuda_phase10_same_candidate_result(
            evidence$warmup$result, evidence$result
          ),
        "Phase 10 warm run is missing a valid complete warm-up"
      )
      rebuilt_warmup_validation <-
        fastkpc_full_cuda_phase10_validate_candidate_result(
        evidence$warmup$result, boundary = "cold"
      )
      fastkpc_full_cuda_phase10_campaign_require(
        identical(
          rebuilt_warmup_validation, evidence$warmup$validation
        ),
        "Phase 10 campaign stored warm-up validation drifted"
      )
    } else {
      fastkpc_full_cuda_phase10_campaign_require(
        is.null(evidence$warmup),
        "Phase 10 cold run unexpectedly contains warm-up evidence"
      )
    }
  }
  fastkpc_full_cuda_phase10_campaign_require(
    identical(
      fastkpc_full_cuda_phase10_campaign_machine_identity(
        evidence$machine_before
      ), freeze$machine_identity
    ) && identical(
      fastkpc_full_cuda_phase10_campaign_machine_identity(
        evidence$machine_after
      ), freeze$machine_identity
    ) && isTRUE(evidence$machine_before$idle_gate) &&
      all(fastkpc_full_cuda_phase10_campaign_active_resources(
        evidence$resource_before
      ) == 0) && all(fastkpc_full_cuda_phase10_campaign_active_resources(
        evidence$resource_after
      ) == 0),
    "Phase 10 campaign run machine or resource evidence is invalid"
  )
  if (isTRUE(verify_current)) {
    fastkpc_full_cuda_phase10_campaign_validate_freeze(
      freeze, verify_current = TRUE, require_idle_gpu = FALSE
    )
  }
  invisible(evidence)
}

fastkpc_full_cuda_phase10_write_campaign_run <- function(evidence, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- tempfile(".phase10-run-", tmpdir = dirname(path))
  on.exit(unlink(temporary, force = TRUE), add = TRUE)
  saveRDS(evidence, temporary, compress = "xz")
  fastkpc_full_cuda_phase10_campaign_require(
    file.rename(temporary, path),
    "Phase 10 campaign run evidence could not be committed atomically"
  )
  invisible(path)
}

fastkpc_full_cuda_phase10_load_campaign_runs <- function(
    staging_dir, freeze, verify_current = FALSE) {
  order <- fastkpc_full_cuda_phase10_campaign_order()
  runs <- lapply(seq_len(nrow(order)), function(index) {
    path <- fastkpc_full_cuda_phase10_campaign_run_path(
      staging_dir, order$mode[[index]], order$repetition[[index]]
    )
    fastkpc_full_cuda_phase10_campaign_require(
      file.exists(path) && !dir.exists(path),
      paste0("Phase 10 campaign run is missing: ", basename(path))
    )
    evidence <- readRDS(path)
    fastkpc_full_cuda_phase10_validate_campaign_run(
      evidence, freeze, verify_current = verify_current
    )
    evidence$evidence_path <- normalizePath(
      path, winslash = "/", mustWork = TRUE
    )
    evidence$evidence_sha256 <- fastkpc_full_cuda_census_file_hash(path)
    evidence$sequence <- order$sequence[[index]]
    evidence
  })
  names(runs) <- vapply(runs, `[[`, character(1L), "run_key")
  runs
}

fastkpc_full_cuda_phase10_campaign_raw_rows <- function(runs) {
  do.call(rbind, lapply(runs, function(run) {
    source <- run$result$summary
    comparison <- run$validation$comparison_summary
    data.frame(
      sequence = as.integer(run$sequence),
      run_key = run$run_key,
      mode = run$mode,
      repetition = as.integer(run$repetition),
      route = if (identical(run$mode, "correct_baseline")) {
        "legacy-mgcv-provider-native-legacy-dcov-20-core"
      } else "compatible.cuda/full_cuda-explicit",
      elapsed_sec = as.numeric(run$elapsed_sec),
      native_elapsed_sec = if (is.null(source$elapsed_sec)) {
        NA_real_
      } else as.numeric(source$elapsed_sec),
      warmup_elapsed_sec = if (is.null(run$warmup)) NA_real_ else
        as.numeric(run$warmup$elapsed_sec),
      edge_count = as.integer(comparison$edge_count_candidate),
      SHD = as.integer(comparison$SHD),
      adjacency_identical = isTRUE(comparison$adjacency_identical),
      sepsets_identical = isTRUE(comparison$sepsets_identical),
      n_edgetests_identical = isTRUE(comparison$n_edgetests_identical),
      deletions_identical = isTRUE(comparison$deletions_identical),
      logical_ci_trace_identical =
        isTRUE(comparison$logical_ci_trace_identical),
      authority_gate_pass = if (identical(
        run$mode, "correct_baseline"
      )) NA else isTRUE(source$authority_gate_pass),
      result_cache_hits = if (is.null(source$result_cache_hit_count)) {
        NA_integer_
      } else as.integer(source$result_cache_hit_count),
      physical_tests_evaluated = if (is.null(
        source$physical_tests_evaluated
      )) NA_integer_ else as.integer(source$physical_tests_evaluated),
      evidence_sha256 = run$evidence_sha256,
      pass = isTRUE(run$pass) && isTRUE(run$validation$pass),
      stringsAsFactors = FALSE
    )
  }))
}

fastkpc_full_cuda_phase10_campaign_cache_rows <- function(runs) {
  candidate <- runs[vapply(runs, function(run) {
    !identical(run$mode, "correct_baseline")
  }, logical(1L))]
  do.call(rbind, lapply(candidate, function(run) {
    source <- run$result$summary
    data.frame(
      run_key = run$run_key,
      mode = run$mode,
      cache = c("compact-result", "target-state", "native-prepared-setup"),
      capacity = c(
        source$result_cache_capacity,
        source$target_cache_capacity,
        source$native_setup_cache_capacity
      ),
      warm_start_entries = c(
        source$result_cache_warm_start_entries,
        source$target_cache_warm_start_entries,
        0L
      ),
      requests = c(
        source$result_cache_request_count,
        source$target_cache_request_count,
        source$native_setup_cache_request_count
      ),
      hits = c(
        source$result_cache_hit_count,
        source$target_cache_hit_count,
        source$native_setup_cache_hit_count
      ),
      misses = c(
        source$result_cache_miss_count,
        source$target_cache_miss_count,
        source$native_setup_cache_miss_count
      ),
      evictions = c(
        source$result_cache_eviction_count,
        source$target_cache_eviction_count,
        source$native_setup_cache_eviction_count
      ),
      bounded = TRUE,
      stringsAsFactors = FALSE
    )
  }))
}

fastkpc_full_cuda_phase10_campaign_machine_rows <- function(runs) {
  do.call(rbind, lapply(runs, function(run) {
    do.call(rbind, lapply(c("before", "after"), function(stage) {
      snapshot <- run[[paste0("machine_", stage)]]
      gpu <- snapshot$gpu[1L, , drop = FALSE]
      data.frame(
        run_key = run$run_key,
        mode = run$mode,
        repetition = run$repetition,
        stage = stage,
        captured_at_utc = snapshot$captured_at_utc,
        governor = snapshot$governor,
        affinity = snapshot$affinity,
        gpu_temperature_c = gpu$temperature_c,
        gpu_power_draw_watts = gpu$power_draw_watts,
        gpu_sm_clock_mhz = gpu$sm_clock_mhz,
        gpu_memory_clock_mhz = gpu$memory_clock_mhz,
        concurrent_compute_process_count = nrow(snapshot$compute_processes),
        identity_gate = snapshot$identity_gate,
        idle_gate = snapshot$idle_gate,
        stringsAsFactors = FALSE
      )
    }))
  }))
}

fastkpc_full_cuda_phase10_campaign_stage_rows <- function(runs) {
  rows <- list()
  for (run in runs) {
    if (identical(run$mode, "correct_baseline")) next
    measured <- run$result$levels
    measured$run_key <- run$run_key
    measured$boundary <- "measured"
    rows[[length(rows) + 1L]] <- measured
    if (!is.null(run$warmup)) {
      warmup <- run$warmup$result$levels
      warmup$run_key <- run$run_key
      warmup$boundary <- "warmup"
      rows[[length(rows) + 1L]] <- warmup
    }
  }
  value <- do.call(rbind, rows)
  value[, c("run_key", "boundary", setdiff(
    names(value), c("run_key", "boundary")
  ))]
}

fastkpc_full_cuda_phase10_validate_campaign_runs <- function(runs, freeze) {
  order <- fastkpc_full_cuda_phase10_campaign_order()
  fastkpc_full_cuda_phase10_campaign_require(
    length(runs) == nrow(order) &&
      identical(
        unname(vapply(runs, `[[`, character(1L), "run_key")),
        vapply(seq_len(nrow(order)), function(index) {
          fastkpc_full_cuda_phase10_campaign_run_key(
            order$mode[[index]], order$repetition[[index]]
          )
        }, character(1L))
      ) &&
      all(vapply(runs, function(run) {
        isTRUE(run$pass) && isTRUE(run$validation$pass) &&
          identical(run$freeze_identity_sha256,
                    freeze$freeze_identity_sha256)
      }, logical(1L))),
    "Phase 10 campaign run set or order is incomplete"
  )
  cold <- runs[vapply(runs, function(run) {
    identical(run$mode, "candidate_cold")
  }, logical(1L))]
  warm <- runs[vapply(runs, function(run) {
    identical(run$mode, "candidate_warm")
  }, logical(1L))]
  baseline <- runs[vapply(runs, function(run) {
    identical(run$mode, "correct_baseline")
  }, logical(1L))]
  candidate_results <- c(
    lapply(cold, `[[`, "result"),
    lapply(warm, `[[`, "result"),
    lapply(warm, function(run) run$warmup$result)
  )
  reference <- candidate_results[[1L]]
  repeatability_gate <- all(vapply(candidate_results, function(result) {
    fastkpc_full_cuda_phase10_same_candidate_result(reference, result)
  }, logical(1L)))
  cold_statistics <- fastkpc_full_cuda_phase10_campaign_statistics(
    vapply(cold, `[[`, numeric(1L), "elapsed_sec")
  )
  warm_statistics <- fastkpc_full_cuda_phase10_campaign_statistics(
    vapply(warm, `[[`, numeric(1L), "elapsed_sec")
  )
  baseline_statistics <- fastkpc_full_cuda_phase10_campaign_statistics(
    vapply(baseline, `[[`, numeric(1L), "elapsed_sec")
  )
  ratio <- warm_statistics$median_sec[[1L]] /
    baseline_statistics$median_sec[[1L]]
  raw <- fastkpc_full_cuda_phase10_campaign_raw_rows(runs)
  gate <- length(cold) == 5L && length(warm) == 5L &&
    length(baseline) == 5L && all(raw$pass) &&
    all(raw$SHD == 0L) && all(raw$adjacency_identical) &&
    all(raw$sepsets_identical) && all(raw$n_edgetests_identical) &&
    all(raw$deletions_identical) && all(raw$logical_ci_trace_identical) &&
    all(raw$authority_gate_pass[raw$mode != "correct_baseline"]) &&
    repeatability_gate && warm_statistics$median_sec[[1L]] <= 120 &&
    ratio <= 0.80
  fastkpc_full_cuda_phase10_campaign_require(
    gate, "Phase 10 campaign correctness, repeatability, or performance failed"
  )
  list(
    schema_version = "full-cuda-ci-phase10-campaign-aggregate-v1",
    runs = runs,
    raw_runs = raw,
    cold_statistics = cold_statistics,
    warm_statistics = warm_statistics,
    baseline_statistics = baseline_statistics,
    candidate_to_baseline_ratio = ratio,
    repeatability_gate = repeatability_gate,
    absolute_performance_gate = warm_statistics$median_sec[[1L]] <= 120,
    relative_performance_gate = ratio <= 0.80,
    pass = TRUE
  )
}

fastkpc_full_cuda_phase10_campaign_statistics_table <- function(aggregate) {
  do.call(rbind, lapply(c("cold", "warm", "baseline"), function(boundary) {
    value <- aggregate[[paste0(boundary, "_statistics")]]
    value$boundary <- boundary
    value[, c("boundary", setdiff(names(value), "boundary"))]
  }))
}

fastkpc_full_cuda_phase10_campaign_cases <- function(
    result,
    logical_path = file.path(
      "fastkpc", "artifacts", "full_cuda_ci",
      "workload_census_351x48_v1", "logical_ci_tests.rds"
    )) {
  tasks <- result$tasks
  logical <- readRDS(logical_path)
  data.frame(
    logical_sequence_id = as.integer(tasks$canonical_test_order_id),
    level = as.integer(tasks$level),
    source_task_index = as.integer(tasks$task_index),
    x = as.integer(tasks$x),
    y = as.integer(tasks$y),
    S_key = as.character(tasks$S_key),
    alpha = 0.1,
    reference_p_value = as.numeric(logical$reference_p_value),
    candidate_p_value = as.numeric(tasks$p_used),
    absolute_p_value_error = abs(
      as.numeric(tasks$p_used) - as.numeric(logical$reference_p_value)
    ),
    absolute_log_distance_from_alpha = abs(log(
      pmax(as.numeric(logical$reference_p_value), .Machine$double.xmin) /
        0.1
    )),
    reference_independent = as.logical(logical$reference_independent),
    candidate_independent = as.logical(tasks$p_used >= 0.1),
    decision_flip = as.logical(
      (tasks$p_used >= 0.1) != logical$reference_independent
    ),
    deletes_edge = as.logical(tasks$native_edge_deleted),
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_phase10_campaign_evidence_inputs <- function(
    runs, staging_dir) {
  static <- fastkpc_full_cuda_phase10_campaign_paths()
  static_rows <- data.frame(
    input_kind = names(static),
    input_file = unname(static),
    sha256 = unname(vapply(
      static, fastkpc_full_cuda_census_file_hash, character(1L)
    )),
    stringsAsFactors = FALSE
  )
  run_rows <- do.call(rbind, lapply(runs, function(run) data.frame(
    input_kind = paste0("campaign_run_", run$run_key),
    input_file = run$evidence_path,
    sha256 = run$evidence_sha256,
    stringsAsFactors = FALSE
  )))
  freeze_paths <- c(
    freeze_rds = file.path(staging_dir, "freeze.rds"),
    freeze_json = file.path(staging_dir, "freeze.json")
  )
  freeze_rows <- data.frame(
    input_kind = names(freeze_paths),
    input_file = unname(freeze_paths),
    sha256 = unname(vapply(
      freeze_paths, fastkpc_full_cuda_census_file_hash, character(1L)
    )),
    stringsAsFactors = FALSE
  )
  rbind(static_rows, run_rows, freeze_rows)
}

fastkpc_full_cuda_phase10_campaign_producer <- function(
    aggregate, freeze, source_closure, native_identity, backend, build,
    contracts) {
  corpus_sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(list(
      dataset_sha256 = freeze$canonical_input_sha256$data,
      logical_sha256 = freeze$canonical_input_sha256$logical,
      logical_test_count = 240489L,
      campaign_run_sha256 = unname(lapply(
        aggregate$runs, `[[`, "evidence_sha256"
      ))
    ))
  )
  fastkpc_full_cuda_phase35_producer_identity(
    producer_source_closure_sha256 = source_closure$sha256,
    native_binary_sha256 = native_identity$sha256,
    route_semantic_version =
      "full-cuda-ci-phase10-canonical-promotion-campaign-v1",
    dataset_or_corpus_sha256 = corpus_sha256,
    oracle_sha256 = freeze$canonical_input_sha256$oracle_manifest,
    backend_configuration_sha256 = backend$sha256,
    build_recipe_sha256 = build$sha256,
    contracts = contracts
  )
}

fastkpc_full_cuda_phase10_campaign_summary <- function(
    aggregate, representative_comparison, representative_summary,
    producer, source_evidence_sha256, freeze, hardening, contracts) {
  cold <- aggregate$cold_statistics
  warm <- aggregate$warm_statistics
  baseline <- aggregate$baseline_statistics
  list(
    schema_version = "full-cuda-ci-phase10-campaign-summary-v1",
    run_status = "ok",
    timeout = FALSE,
    source_commit = freeze$source_commit,
    oracle_artifact =
      "fastkpc/artifacts/full_cuda_ci/oracle_351x48_v1",
    candidate_route = "compatible.cuda/full_cuda-explicit",
    baseline_route = "legacy-mgcv-provider-native-legacy-dcov-20-core",
    edge_count_reference =
      representative_comparison$summary$edge_count_reference,
    edge_count_candidate =
      representative_comparison$summary$edge_count_candidate,
    SHD = representative_comparison$summary$SHD,
    adjacency_identical =
      representative_comparison$summary$adjacency_identical,
    sepsets_identical =
      representative_comparison$summary$sepsets_identical,
    n_edgetests_identical =
      representative_comparison$summary$n_edgetests_identical,
    deletions_identical =
      representative_comparison$summary$deletions_identical,
    logical_ci_trace_identical =
      representative_comparison$summary$logical_ci_trace_identical,
    logical_tests_consumed = representative_summary$logical_tests_consumed,
    candidate_cold_repetitions = cold$repetitions[[1L]],
    candidate_warm_repetitions = warm$repetitions[[1L]],
    correct_baseline_repetitions = baseline$repetitions[[1L]],
    complete_warmup_repetitions = 5L,
    cold_median_sec = cold$median_sec[[1L]],
    cold_min_sec = cold$min_sec[[1L]],
    cold_max_sec = cold$max_sec[[1L]],
    cold_mad_sec = cold$mad_sec[[1L]],
    cold_iqr_sec = cold$iqr_sec[[1L]],
    warm_median_sec = warm$median_sec[[1L]],
    warm_min_sec = warm$min_sec[[1L]],
    warm_max_sec = warm$max_sec[[1L]],
    warm_mad_sec = warm$mad_sec[[1L]],
    warm_iqr_sec = warm$iqr_sec[[1L]],
    baseline_median_sec = baseline$median_sec[[1L]],
    baseline_min_sec = baseline$min_sec[[1L]],
    baseline_max_sec = baseline$max_sec[[1L]],
    baseline_mad_sec = baseline$mad_sec[[1L]],
    baseline_iqr_sec = baseline$iqr_sec[[1L]],
    candidate_to_baseline_ratio = aggregate$candidate_to_baseline_ratio,
    warm_absolute_limit_sec = 120,
    warm_relative_ratio_limit = "0.80",
    warm_stretch_limit_sec = 60,
    absolute_performance_gate = aggregate$absolute_performance_gate,
    relative_performance_gate = aggregate$relative_performance_gate,
    warm_stretch_gate = warm$median_sec[[1L]] <= 60,
    repeatability_gate = aggregate$repeatability_gate,
    every_run_correctness_gate = all(aggregate$raw_runs$pass),
    every_candidate_authority_gate = all(
      aggregate$raw_runs$authority_gate_pass[
        aggregate$raw_runs$mode != "correct_baseline"
      ]
    ),
    result_cache_capacity = representative_summary$result_cache_capacity,
    target_cache_capacity = representative_summary$target_cache_capacity,
    native_setup_cache_capacity =
      representative_summary$native_setup_cache_capacity,
    component_cache_capacity = representative_summary$component_cache_capacity,
    r_callback_count = 0L,
    legacy_mgcv_fit_count = 0L,
    legacy_mgcv_setup_count = 0L,
    cpu_residual_solve_count = 0L,
    cpu_dcov_component_count = 0L,
    cpu_dcov_eigen_or_lowrank_count = 0L,
    cpu_dcov_pair_stat_count = 0L,
    cpu_gamma_pvalue_count = 0L,
    cpu_spectra_count = 0L,
    residual_d2h_bytes = 0,
    component_d2h_bytes = 0,
    unknown_fallback_count = 0L,
    approximate_backend_count = 0L,
    hardening_artifact = fastkpc_full_cuda_phase10_hardening_artifact_dir(),
    hardening_producer_identity_sha256 =
      hardening$producer$identity_sha256,
    hardening_gate = isTRUE(hardening$summary$pass),
    holdout_state = "SEALED_NOT_RELEASED",
    holdout_gate = FALSE,
    phase10_canonical_campaign_claim = TRUE,
    phase10_promotion_claim = FALSE,
    recommended_route = FALSE,
    freeze_identity_sha256 = freeze$freeze_identity_sha256,
    architecture_contract_sha256 = contracts$architecture_contract_v1$sha256,
    numerical_contract_sha256 = contracts$numerical_contract_v1$sha256,
    artifact_identity_contract_sha256 =
      contracts$artifact_identity_contract_v1$sha256,
    reference_machine_contract_sha256 =
      contracts$reference_machine_v1$sha256,
    performance_budget_contract_sha256 =
      contracts$performance_budget_v1$sha256,
    source_evidence_sha256 = source_evidence_sha256,
    producer_identity_sha256 = producer$identity_sha256,
    source_closure_sha256 = producer$producer_source_closure_sha256,
    native_binary_sha256 = producer$native_binary_sha256,
    elapsed_sec = sum(aggregate$raw_runs$elapsed_sec) + sum(
      aggregate$raw_runs$warmup_elapsed_sec, na.rm = TRUE
    ),
    pass = TRUE
  )
}

fastkpc_full_cuda_phase10_campaign_validate_summary <- function(summary) {
  zero_fields <- fastkpc_full_cuda_phase10_campaign_authority_zero_fields()
  hash_fields <- c(
    "freeze_identity_sha256", "architecture_contract_sha256",
    "numerical_contract_sha256", "artifact_identity_contract_sha256",
    "reference_machine_contract_sha256",
    "performance_budget_contract_sha256", "source_evidence_sha256",
    "producer_identity_sha256", "source_closure_sha256",
    "native_binary_sha256", "hardening_producer_identity_sha256"
  )
  clean <- is.list(summary) && identical(
    summary$schema_version, "full-cuda-ci-phase10-campaign-summary-v1"
  ) && identical(summary$run_status, "ok") && !isTRUE(summary$timeout) &&
    summary$edge_count_reference == 110L &&
    summary$edge_count_candidate == 110L && summary$SHD == 0L &&
    isTRUE(summary$adjacency_identical) &&
    isTRUE(summary$sepsets_identical) &&
    isTRUE(summary$n_edgetests_identical) &&
    isTRUE(summary$deletions_identical) &&
    isTRUE(summary$logical_ci_trace_identical) &&
    summary$logical_tests_consumed == 240489L &&
    summary$candidate_cold_repetitions == 5L &&
    summary$candidate_warm_repetitions == 5L &&
    summary$correct_baseline_repetitions == 5L &&
    summary$complete_warmup_repetitions == 5L &&
    all(is.finite(c(
      summary$cold_median_sec, summary$cold_min_sec,
      summary$cold_max_sec, summary$cold_mad_sec, summary$cold_iqr_sec,
      summary$warm_median_sec, summary$warm_min_sec,
      summary$warm_max_sec, summary$warm_mad_sec, summary$warm_iqr_sec,
      summary$baseline_median_sec, summary$baseline_min_sec,
      summary$baseline_max_sec, summary$baseline_mad_sec,
      summary$baseline_iqr_sec, summary$candidate_to_baseline_ratio
    ))) && summary$warm_median_sec <= 120 &&
    summary$candidate_to_baseline_ratio <= 0.80 &&
    isTRUE(summary$absolute_performance_gate) &&
    isTRUE(summary$relative_performance_gate) &&
    isTRUE(summary$repeatability_gate) &&
    isTRUE(summary$every_run_correctness_gate) &&
    isTRUE(summary$every_candidate_authority_gate) &&
    summary$result_cache_capacity == 262144L &&
    summary$target_cache_capacity == 131072L &&
    summary$native_setup_cache_capacity == 64L &&
    summary$component_cache_capacity == 47L &&
    all(vapply(zero_fields, function(field) {
      identical(as.numeric(summary[[field]]), 0)
    }, logical(1L))) && isTRUE(summary$hardening_gate) &&
    identical(summary$holdout_state, "SEALED_NOT_RELEASED") &&
    !isTRUE(summary$holdout_gate) &&
    isTRUE(summary$phase10_canonical_campaign_claim) &&
    !isTRUE(summary$phase10_promotion_claim) &&
    !isTRUE(summary$recommended_route) &&
    is.finite(summary$elapsed_sec) && summary$elapsed_sec > 0 &&
    isTRUE(summary$pass) && all(vapply(hash_fields, function(field) {
      is.character(summary[[field]]) && length(summary[[field]]) == 1L &&
        grepl("^[0-9a-f]{64}$", summary[[field]])
    }, logical(1L)))
  fastkpc_full_cuda_phase10_campaign_require(
    clean, "Phase 10 canonical campaign summary gate failed"
  )
  invisible(TRUE)
}

fastkpc_full_cuda_phase10_campaign_manifest_fields <- function() {
  c(
    "schema_version", "artifact_kind", "claim_scope",
    "producer_semantic_envelope", "payload_manifest_sha256",
    "payload_file_sha256", "semantic_file_count",
    "validator_attestations_file", "volatile_receipt_file",
    "environment_file", "environment_file_sha256"
  )
}

fastkpc_full_cuda_phase10_campaign_read_table <- function(
    artifact_dir, name) {
  utils::read.csv(
    file.path(artifact_dir, name), stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

fastkpc_full_cuda_phase10_validate_campaign_payload <- function(
    artifact_dir, evidence, summary) {
  runs <- evidence$aggregate$runs
  freeze <- evidence$freeze
  warm <- runs[vapply(runs, function(run) {
    identical(run$mode, "candidate_warm")
  }, logical(1L))]
  fastkpc_full_cuda_phase10_campaign_require(
    length(warm) == 5L,
    "Phase 10 campaign payload is missing warm candidate evidence"
  )
  representative <- warm[[1L]]$result
  comparison <- fastkpc_full_cuda_compare_candidate_skeleton(
    fastkpc_load_full_cuda_ci_oracle(
      file.path(
        "fastkpc", "artifacts", "full_cuda_ci", "oracle_351x48_v1"
      )
    ), representative
  )
  graph <- fastkpc_full_cuda_phase10_campaign_read_table(
    artifact_dir, "graph_agreement.csv"
  )
  sepsets <- fastkpc_full_cuda_phase10_campaign_read_table(
    artifact_dir, "sepset_agreement.csv"
  )
  n_edgetests <- fastkpc_full_cuda_phase10_campaign_read_table(
    artifact_dir, "n_edgetests.csv"
  )
  deletions <- fastkpc_full_cuda_phase10_campaign_read_table(
    artifact_dir, "deletion_trace.csv"
  )
  first <- jsonlite::read_json(
    file.path(artifact_dir, "first_divergence.json"),
    simplifyVector = TRUE
  )
  fallbacks <- fastkpc_full_cuda_phase10_campaign_read_table(
    artifact_dir, "fallbacks.csv"
  )
  raw <- fastkpc_full_cuda_phase10_campaign_read_table(
    artifact_dir, "raw_runs.csv"
  )
  statistics <- fastkpc_full_cuda_phase10_campaign_read_table(
    artifact_dir, "statistics.csv"
  )
  cases <- fastkpc_full_cuda_phase10_campaign_read_table(
    artifact_dir, "case_results.csv"
  )
  near <- fastkpc_full_cuda_phase10_campaign_read_table(
    artifact_dir, "near_alpha_results.csv"
  )
  cache <- fastkpc_full_cuda_phase10_campaign_read_table(
    artifact_dir, "cache.csv"
  )
  machine <- fastkpc_full_cuda_phase10_campaign_read_table(
    artifact_dir, "machine_samples.csv"
  )
  order <- fastkpc_full_cuda_phase10_campaign_read_table(
    artifact_dir, "campaign_order.csv"
  )
  inputs <- fastkpc_full_cuda_phase10_campaign_read_table(
    artifact_dir, "evidence_inputs.csv"
  )
  expected_n_edgetests <- c(
    2213L, 52659L, 125293L, 40694L, 13293L, 5422L, 835L, 80L
  )
  graph_gate <- nrow(graph) == 1L &&
    graph$edge_count_reference[[1L]] == 110L &&
    graph$edge_count_candidate[[1L]] == 110L &&
    graph$SHD[[1L]] == 0L &&
    isTRUE(as.logical(graph$adjacency_identical[[1L]])) &&
    nrow(sepsets) == nrow(comparison$sepset_agreement) &&
    nrow(sepsets) > 0L && all(as.logical(sepsets$identical)) &&
    nrow(n_edgetests) == 8L &&
    identical(as.integer(n_edgetests$reference), expected_n_edgetests) &&
    identical(as.integer(n_edgetests$candidate), expected_n_edgetests) &&
    all(as.logical(n_edgetests$identical)) &&
    nrow(deletions) == nrow(comparison$candidate_deletions) &&
    !isTRUE(first$first_divergence_found) && nrow(fallbacks) == 3L &&
    sum(as.numeric(fallbacks$count)) == 0 &&
    !any(as.logical(fallbacks$accepted_for_phase10))
  fastkpc_full_cuda_phase10_campaign_require(
    graph_gate, "Phase 10 campaign graph payload is malformed"
  )

  expected_order <- fastkpc_full_cuda_phase10_campaign_order()
  order_gate <- nrow(order) == nrow(expected_order) &&
    identical(as.integer(order$sequence), expected_order$sequence) &&
    identical(as.character(order$mode), expected_order$mode) &&
    identical(as.integer(order$repetition), expected_order$repetition)
  expected_keys <- vapply(seq_len(nrow(expected_order)), function(index) {
    fastkpc_full_cuda_phase10_campaign_run_key(
      expected_order$mode[[index]], expected_order$repetition[[index]]
    )
  }, character(1L))
  raw_gate <- nrow(raw) == 15L &&
    identical(as.integer(raw$sequence), seq_len(15L)) &&
    identical(as.character(raw$run_key), expected_keys) &&
    identical(as.character(raw$mode), expected_order$mode) &&
    identical(as.integer(raw$repetition), expected_order$repetition) &&
    all(is.finite(raw$elapsed_sec) & raw$elapsed_sec > 0) &&
    all(as.integer(raw$edge_count) == 110L) &&
    all(as.integer(raw$SHD) == 0L) &&
    all(as.logical(raw$adjacency_identical)) &&
    all(as.logical(raw$sepsets_identical)) &&
    all(as.logical(raw$n_edgetests_identical)) &&
    all(as.logical(raw$deletions_identical)) &&
    all(as.logical(raw$logical_ci_trace_identical)) &&
    all(as.logical(raw$pass)) &&
    all(as.logical(raw$authority_gate_pass[
      raw$mode != "correct_baseline"
    ]))
  statistics_gate <- nrow(statistics) == 3L &&
    identical(as.character(statistics$boundary),
              c("cold", "warm", "baseline")) &&
    all(as.integer(statistics$repetitions) == 5L) &&
    all(is.finite(as.matrix(statistics[, c(
      "median_sec", "min_sec", "max_sec", "mad_sec", "iqr_sec"
    )]))) &&
    isTRUE(all.equal(
      as.numeric(statistics$median_sec),
      c(
        summary$cold_median_sec, summary$warm_median_sec,
        summary$baseline_median_sec
      ),
      tolerance = 1e-12, check.attributes = FALSE
    )) &&
    statistics$median_sec[[2L]] <= 120 &&
    statistics$median_sec[[2L]] / statistics$median_sec[[3L]] <= 0.80
  fastkpc_full_cuda_phase10_campaign_require(
    order_gate && raw_gate && statistics_gate,
    "Phase 10 campaign run or timing payload is malformed"
  )

  case_gate <- nrow(cases) == 240489L &&
    identical(as.integer(cases$logical_sequence_id), seq_len(240489L)) &&
    all(is.finite(cases$reference_p_value)) &&
    all(is.finite(cases$candidate_p_value)) &&
    !any(as.logical(cases$decision_flip)) &&
    all(as.logical(cases$reference_independent) ==
          as.logical(cases$candidate_independent)) &&
    nrow(near) == 1529L &&
    all(is.finite(near$absolute_log_distance_from_alpha) &
          near$absolute_log_distance_from_alpha <= log(2)) &&
    !any(as.logical(near$decision_flip))
  cache_gate <- nrow(cache) == 30L &&
    identical(sort(unique(as.character(cache$cache)), method = "radix"),
              c("compact-result", "native-prepared-setup", "target-state")) &&
    all(as.logical(cache$bounded)) &&
    all(cache$requests == cache$hits + cache$misses) &&
    all(cache$capacity > 0L) &&
    all(cache$capacity[cache$cache == "compact-result"] == 262144L) &&
    all(cache$capacity[cache$cache == "target-state"] == 131072L) &&
    all(cache$capacity[cache$cache == "native-prepared-setup"] == 64L)
  machine_gate <- nrow(machine) == 30L &&
    identical(sort(unique(as.character(machine$stage)), method = "radix"),
              c("after", "before")) &&
    all(as.logical(machine$identity_gate)) &&
    all(as.logical(machine$idle_gate)) &&
    all(machine$concurrent_compute_process_count[machine$stage == "before"] ==
          0L)
  fastkpc_full_cuda_phase10_campaign_require(
    case_gate && cache_gate && machine_gate,
    "Phase 10 campaign numerical, cache, or machine payload is malformed"
  )

  static_names <- names(fastkpc_full_cuda_phase10_campaign_paths())
  expected_input_kinds <- c(
    static_names,
    paste0("campaign_run_", expected_keys),
    "freeze_rds", "freeze_json"
  )
  input_gate <- nrow(inputs) == length(expected_input_kinds) &&
    identical(as.character(inputs$input_kind), expected_input_kinds) &&
    all(grepl("^[0-9a-f]{64}$", inputs$sha256)) &&
    all(vapply(seq_along(runs), function(index) {
      identical(
        as.character(inputs$sha256[
          inputs$input_kind == paste0("campaign_run_", runs[[index]]$run_key)
        ]),
        runs[[index]]$evidence_sha256
      )
    }, logical(1L)))
  rank_path <- file.path(artifact_dir, "rank_condition_results.csv")
  static_hash_gate <- all(vapply(static_names, function(name) {
    identical(
      as.character(inputs$sha256[inputs$input_kind == name]),
      freeze$canonical_input_sha256[[name]]
    )
  }, logical(1L))) && identical(
    fastkpc_full_cuda_census_file_hash(rank_path),
    freeze$canonical_input_sha256$phase8_rank_condition
  )
  fastkpc_full_cuda_phase10_campaign_require(
    input_gate && static_hash_gate &&
      file.info(file.path(artifact_dir, "summary.md"))$size[[1L]] > 0L &&
      file.info(file.path(artifact_dir, "commands.txt"))$size[[1L]] > 0L,
    "Phase 10 campaign supplemental payload is malformed"
  )

  saved_adjacency <- readRDS(file.path(artifact_dir, "adjacency.rds"))
  saved_sepsets <- readRDS(file.path(artifact_dir, "sepsets.rds"))
  saved_pmax <- readRDS(file.path(artifact_dir, "pmax.rds"))
  saved_logical <- readRDS(file.path(artifact_dir, "logical_ci_trace.rds"))
  rds_gate <- identical(saved_adjacency, comparison$candidate_adjacency) &&
    identical(saved_sepsets, representative$sepsets) &&
    identical(saved_pmax, representative$pMax) &&
    identical(saved_logical, comparison$candidate_logical)
  fastkpc_full_cuda_phase10_campaign_require(
    rds_gate, "Phase 10 campaign graph RDS payload drifted"
  )
  invisible(TRUE)
}

fastkpc_full_cuda_phase10_validate_campaign_artifact <- function(
    artifact_dir = fastkpc_full_cuda_phase10_campaign_artifact_dir(),
    verify_current_sources = FALSE) {
  artifact_dir <- normalizePath(
    artifact_dir, winslash = "/", mustWork = TRUE
  )
  required <- sort(
    fastkpc_full_cuda_phase10_campaign_required_files(), method = "radix"
  )
  actual <- sort(
    list.files(artifact_dir, all.files = FALSE, no.. = TRUE),
    method = "radix"
  )
  fastkpc_full_cuda_phase10_campaign_require(
    identical(actual, required),
    "Phase 10 campaign artifact standard file set is incomplete"
  )
  manifest <- jsonlite::read_json(
    file.path(artifact_dir, "manifest.json"), simplifyVector = FALSE
  )
  fastkpc_full_cuda_phase10_campaign_require(
    is.list(manifest) && identical(
      names(manifest), fastkpc_full_cuda_phase10_campaign_manifest_fields()
    ) && identical(
      manifest$schema_version,
      "full-cuda-ci-phase10-campaign-manifest-v1"
    ) && identical(manifest$artifact_kind, "promotion_351x48") &&
      identical(
        manifest$claim_scope,
        "phase10-canonical-performance-repeatability"
      ) && identical(
        manifest$validator_attestations_file,
        "validator_attestations.json"
      ) && identical(
        manifest$volatile_receipt_file, "execution_receipts.json"
      ) && identical(manifest$environment_file, "environment.txt"),
    "Phase 10 campaign artifact manifest schema mismatch"
  )
  excluded <- c(
    "manifest.json", "validator_attestations.json",
    "execution_receipts.json", "environment.txt", "commands.txt"
  )
  semantic <- sort(setdiff(required, excluded), method = "radix")
  payload_hashes <- manifest$payload_file_sha256
  fastkpc_full_cuda_phase10_campaign_require(
    is.list(payload_hashes) &&
      identical(sort(names(payload_hashes), method = "radix"), semantic),
    "Phase 10 campaign payload manifest is malformed"
  )
  actual_hashes <- setNames(lapply(names(payload_hashes), function(name) {
    fastkpc_full_cuda_census_file_hash(file.path(artifact_dir, name))
  }), names(payload_hashes))
  fastkpc_full_cuda_phase10_campaign_require(
    all(vapply(names(payload_hashes), function(name) {
      identical(payload_hashes[[name]], actual_hashes[[name]])
    }, logical(1L))),
    "Phase 10 campaign artifact payload identity mismatch"
  )
  payload_sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(actual_hashes)
  )
  fastkpc_full_cuda_phase10_campaign_require(
    identical(payload_sha256, manifest$payload_manifest_sha256) &&
      as.integer(manifest$semantic_file_count) == length(actual_hashes),
    "Phase 10 campaign payload manifest identity mismatch"
  )
  fastkpc_full_cuda_phase35_validate_identity_envelope(
    manifest$producer_semantic_envelope
  )
  producer <- jsonlite::read_json(
    file.path(artifact_dir, "producer_identity.json"), simplifyVector = FALSE
  )
  .fastkpc_full_cuda_phase35_validate_producer(producer)
  fastkpc_full_cuda_phase10_campaign_require(
    identical(
      fastkpc_full_cuda_phase35_canonical_json(producer),
      fastkpc_full_cuda_phase35_canonical_json(
        manifest$producer_semantic_envelope$producer
      )
    ) && identical(
      manifest$producer_semantic_envelope$payload_manifest_sha256,
      payload_sha256
    ),
    "Phase 10 campaign producer envelope mismatch"
  )
  source_closure <- fastkpc_full_cuda_phase10_campaign_read_table(
    artifact_dir, "source_closure.csv"
  )
  fastkpc_full_cuda_phase10_campaign_require(
    identical(names(source_closure), c("path", "sha256")) &&
      !anyDuplicated(source_closure$path) &&
      all(grepl("^[0-9a-f]{64}$", source_closure$sha256)),
    "Phase 10 campaign source closure table is malformed"
  )
  closure_hashes <- setNames(
    as.list(as.character(source_closure$sha256)),
    as.character(source_closure$path)
  )
  closure_sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(closure_hashes)
  )
  fastkpc_full_cuda_phase10_campaign_require(
    identical(closure_sha256, producer$producer_source_closure_sha256),
    "Phase 10 campaign producer source closure mismatch"
  )
  backend <- jsonlite::read_json(
    file.path(artifact_dir, "backend_configuration.json"),
    simplifyVector = FALSE
  )
  build <- jsonlite::read_json(
    file.path(artifact_dir, "build_recipe.json"), simplifyVector = FALSE
  )
  fastkpc_full_cuda_phase10_campaign_require(
    identical(
      fastkpc_full_cuda_phase35_sha256_utf8(
        fastkpc_full_cuda_phase35_canonical_json(backend)
      ), producer$backend_configuration_sha256
    ) && identical(
      fastkpc_full_cuda_phase35_sha256_utf8(
        fastkpc_full_cuda_phase35_canonical_json(build)
      ), producer$build_recipe_sha256
    ) && identical(
      producer$producer_source_closure_sha256,
      jsonlite::read_json(
        file.path(artifact_dir, "freeze.json"), simplifyVector = FALSE
      )$source_closure_sha256
    ),
    "Phase 10 campaign backend or build identity mismatch"
  )
  contracts <- fastkpc_full_cuda_phase35_load_contract_set()
  fastkpc_full_cuda_phase10_campaign_require(
    identical(
      fastkpc_full_cuda_phase35_canonical_json(
        producer$contract_snapshots
      ),
      fastkpc_full_cuda_phase35_canonical_json(
        fastkpc_full_cuda_phase35_contract_snapshots(contracts)
      )
    ),
    "Phase 10 campaign tracked contract snapshots drifted"
  )
  environment_sha256 <- fastkpc_full_cuda_census_file_hash(
    file.path(artifact_dir, manifest$environment_file)
  )
  fastkpc_full_cuda_phase10_campaign_require(
    identical(environment_sha256, manifest$environment_file_sha256),
    "Phase 10 campaign environment evidence hash mismatch"
  )
  attestations <- jsonlite::read_json(
    file.path(artifact_dir, manifest$validator_attestations_file),
    simplifyVector = FALSE
  )$attestations
  receipts <- jsonlite::read_json(
    file.path(artifact_dir, manifest$volatile_receipt_file),
    simplifyVector = FALSE
  )$execution_receipts
  fastkpc_full_cuda_phase10_campaign_require(
    is.list(attestations) && length(attestations) > 0L &&
      is.list(receipts) && length(receipts) > 0L,
    "Phase 10 campaign attestation or receipt is missing"
  )
  for (attestation in attestations) {
    .fastkpc_full_cuda_phase35_validate_attestation(attestation)
    fastkpc_full_cuda_phase10_campaign_require(
      identical(attestation$attested_producer_sha256,
                producer$identity_sha256) &&
        identical(attestation$environment_sha256, environment_sha256) &&
        identical(attestation$validator_source_closure_sha256,
                  closure_sha256) &&
        identical(attestation$validation_result, "PASS"),
      "Phase 10 campaign validator attestation mismatch"
    )
  }
  for (receipt in receipts) {
    .fastkpc_full_cuda_phase35_validate_receipt(receipt)
    fastkpc_full_cuda_phase10_campaign_require(
      identical(receipt$producer_sha256, producer$identity_sha256),
      "Phase 10 campaign execution receipt mismatch"
    )
  }

  summary <- jsonlite::read_json(
    file.path(artifact_dir, "summary.json"), simplifyVector = TRUE
  )
  fastkpc_full_cuda_phase10_campaign_validate_summary(summary)
  evidence_path <- file.path(artifact_dir, "source_evidence.rds")
  fastkpc_full_cuda_phase10_campaign_require(
    identical(summary$source_evidence_sha256,
              fastkpc_full_cuda_census_file_hash(evidence_path)) &&
      identical(summary$producer_identity_sha256, producer$identity_sha256) &&
      identical(summary$source_closure_sha256,
                producer$producer_source_closure_sha256) &&
      identical(summary$native_binary_sha256,
                producer$native_binary_sha256),
    "Phase 10 campaign summary identity linkage failed"
  )
  evidence <- readRDS(evidence_path)
  fastkpc_full_cuda_phase10_campaign_require(
    is.list(evidence) && identical(
      evidence$schema_version,
      "full-cuda-ci-phase10-campaign-publication-evidence-v1"
    ) && is.list(evidence$freeze) && is.list(evidence$aggregate) &&
      identical(evidence$hardening_manifest_sha256,
                evidence$freeze$canonical_input_sha256$hardening_manifest) &&
      isTRUE(evidence$pass),
    "Phase 10 campaign publication evidence is malformed"
  )
  fastkpc_full_cuda_phase10_campaign_validate_freeze(
    evidence$freeze, verify_current = FALSE
  )
  static <- fastkpc_full_cuda_phase10_campaign_paths()
  fastkpc_full_cuda_phase10_campaign_require(
    all(vapply(names(static), function(name) {
      identical(
        fastkpc_full_cuda_census_file_hash(static[[name]]),
        evidence$freeze$canonical_input_sha256[[name]]
      )
    }, logical(1L))) &&
      identical(producer$producer_source_closure_sha256,
                evidence$freeze$source_closure_sha256) &&
      identical(producer$native_binary_sha256,
                evidence$freeze$native_binary_sha256) &&
      identical(producer$backend_configuration_sha256,
                evidence$freeze$backend_configuration_sha256) &&
      identical(producer$build_recipe_sha256,
                evidence$freeze$build_recipe_sha256),
    "Phase 10 campaign frozen canonical input drifted"
  )
  for (run in evidence$aggregate$runs) {
    fastkpc_full_cuda_phase10_validate_campaign_run(
      run, evidence$freeze, verify_current = FALSE
    )
  }
  rebuilt <- fastkpc_full_cuda_phase10_validate_campaign_runs(
    evidence$aggregate$runs, evidence$freeze
  )
  fastkpc_full_cuda_phase10_campaign_require(
    identical(rebuilt$raw_runs, evidence$aggregate$raw_runs) &&
      identical(rebuilt$cold_statistics,
                evidence$aggregate$cold_statistics) &&
      identical(rebuilt$warm_statistics,
                evidence$aggregate$warm_statistics) &&
      identical(rebuilt$baseline_statistics,
                evidence$aggregate$baseline_statistics) &&
      identical(rebuilt$candidate_to_baseline_ratio,
                evidence$aggregate$candidate_to_baseline_ratio) &&
      isTRUE(rebuilt$pass),
    "Phase 10 campaign aggregate could not be independently rebuilt"
  )
  hardening <- fastkpc_full_cuda_phase10_validate_hardening_artifact(
    fastkpc_full_cuda_phase10_hardening_artifact_dir(),
    verify_current_sources = FALSE
  )
  fastkpc_full_cuda_phase10_campaign_require(
    identical(hardening$producer$identity_sha256,
              summary$hardening_producer_identity_sha256),
    "Phase 10 campaign hardening evidence linkage failed"
  )
  expected_producer <- fastkpc_full_cuda_phase10_campaign_producer(
    evidence$aggregate,
    evidence$freeze,
    list(sha256 = closure_sha256),
    list(sha256 = producer$native_binary_sha256),
    list(sha256 = producer$backend_configuration_sha256),
    list(sha256 = producer$build_recipe_sha256),
    contracts
  )
  warm_runs <- evidence$aggregate$runs[vapply(
    evidence$aggregate$runs,
    function(run) identical(run$mode, "candidate_warm"),
    logical(1L)
  )]
  representative <- warm_runs[[1L]]$result
  representative_comparison <- fastkpc_full_cuda_compare_candidate_skeleton(
    fastkpc_load_full_cuda_ci_oracle(
      file.path(
        "fastkpc", "artifacts", "full_cuda_ci", "oracle_351x48_v1"
      )
    ), representative
  )
  expected_summary <- fastkpc_full_cuda_phase10_campaign_summary(
    evidence$aggregate, representative_comparison,
    representative$summary, expected_producer,
    fastkpc_full_cuda_census_file_hash(evidence_path),
    evidence$freeze, hardening, contracts
  )
  fastkpc_full_cuda_phase10_campaign_require(
    identical(
      fastkpc_full_cuda_phase35_canonical_json(expected_producer),
      fastkpc_full_cuda_phase35_canonical_json(producer)
    ) && fastkpc_full_cuda_phase10_campaign_json_equivalent(
      expected_summary, summary
    ),
    "Phase 10 campaign producer or summary could not be rebuilt"
  )
  fastkpc_full_cuda_phase10_validate_campaign_payload(
    artifact_dir, evidence, summary
  )
  freeze_json <- jsonlite::read_json(
    file.path(artifact_dir, "freeze.json"), simplifyVector = FALSE
  )
  fastkpc_full_cuda_phase10_campaign_require(
    identical(
      fastkpc_full_cuda_phase35_canonical_json(freeze_json),
      fastkpc_full_cuda_phase35_canonical_json(evidence$freeze)
    ),
    "Phase 10 campaign JSON freeze receipt drifted"
  )
  if (isTRUE(verify_current_sources)) {
    current <- fastkpc_full_cuda_phase10_campaign_source_closure()
    fastkpc_full_cuda_phase10_campaign_require(
      identical(current$sha256, closure_sha256) &&
        identical(current$hashes, closure_hashes) &&
        identical(fastkpc_full_cuda_phase7_native_identity()$sha256,
                  producer$native_binary_sha256) &&
        identical(
          fastkpc_full_cuda_phase10_campaign_backend_configuration()$sha256,
          producer$backend_configuration_sha256
        ) && identical(
          fastkpc_full_cuda_phase10_campaign_build_recipe()$sha256,
          producer$build_recipe_sha256
        ),
      "Phase 10 campaign current source, binary, or configuration drifted"
    )
  }
  list(
    manifest = manifest, summary = summary, producer = producer,
    source_closure = source_closure, evidence = evidence
  )
}

fastkpc_full_cuda_phase10_campaign_write_table <- function(
    value, directory, name) {
  utils::write.csv(
    value, file.path(directory, name), row.names = FALSE, na = ""
  )
}

fastkpc_full_cuda_phase10_publish_campaign <- function(
    staging_dir = fastkpc_full_cuda_phase10_campaign_staging_dir(),
    output_dir = fastkpc_full_cuda_phase10_campaign_artifact_dir()) {
  freeze_path <- file.path(staging_dir, "freeze.rds")
  fastkpc_full_cuda_phase10_campaign_require(
    file.exists(freeze_path), "Phase 10 campaign freeze is missing"
  )
  freeze <- readRDS(freeze_path)
  fastkpc_full_cuda_phase10_campaign_validate_freeze(
    freeze, verify_current = TRUE, require_idle_gpu = TRUE
  )
  runs <- fastkpc_full_cuda_phase10_load_campaign_runs(
    staging_dir, freeze, verify_current = FALSE
  )
  aggregate <- fastkpc_full_cuda_phase10_validate_campaign_runs(runs, freeze)
  source_closure <- fastkpc_full_cuda_phase10_campaign_source_closure()
  native_identity <- fastkpc_full_cuda_phase7_native_identity()
  backend <- fastkpc_full_cuda_phase10_campaign_backend_configuration()
  build <- fastkpc_full_cuda_phase10_campaign_build_recipe()
  contracts <- fastkpc_full_cuda_phase35_load_contract_set()
  hardening <- fastkpc_full_cuda_phase10_validate_hardening_artifact(
    fastkpc_full_cuda_phase10_hardening_artifact_dir(),
    verify_current_sources = TRUE
  )
  fastkpc_full_cuda_phase10_campaign_require(
    identical(source_closure$sha256, freeze$source_closure_sha256) &&
      identical(native_identity$sha256, freeze$native_binary_sha256) &&
      identical(backend$sha256, freeze$backend_configuration_sha256) &&
      identical(build$sha256, freeze$build_recipe_sha256) &&
      identical(hardening$producer$identity_sha256,
                freeze$hardening_producer_identity_sha256),
    "Phase 10 campaign publication identity drifted after measurement"
  )
  warm_runs <- runs[vapply(runs, function(run) {
    identical(run$mode, "candidate_warm")
  }, logical(1L))]
  representative <- warm_runs[[1L]]$result
  oracle <- fastkpc_load_full_cuda_ci_oracle(
    "fastkpc/artifacts/full_cuda_ci/oracle_351x48_v1"
  )
  comparison <- fastkpc_full_cuda_compare_candidate_skeleton(
    oracle, representative
  )
  cases <- fastkpc_full_cuda_phase10_campaign_cases(representative)
  near <- cases[
    is.finite(cases$absolute_log_distance_from_alpha) &
      cases$absolute_log_distance_from_alpha <= log(2),
    , drop = FALSE
  ]
  producer <- fastkpc_full_cuda_phase10_campaign_producer(
    aggregate, freeze, source_closure, native_identity, backend, build,
    contracts
  )
  evidence <- list(
    schema_version = "full-cuda-ci-phase10-campaign-publication-evidence-v1",
    freeze = freeze,
    aggregate = aggregate,
    hardening_manifest_sha256 =
      freeze$canonical_input_sha256$hardening_manifest,
    pass = TRUE
  )

  parent <- dirname(output_dir)
  dir.create(parent, recursive = TRUE, showWarnings = FALSE)
  stage <- tempfile(".phase10-campaign-stage-", tmpdir = parent)
  dir.create(stage, recursive = TRUE)
  active <- TRUE
  on.exit({
    if (active && dir.exists(stage)) {
      unlink(stage, recursive = TRUE, force = TRUE)
    }
  }, add = TRUE)
  saveRDS(evidence, file.path(stage, "source_evidence.rds"), compress = "xz")
  evidence_sha256 <- fastkpc_full_cuda_census_file_hash(
    file.path(stage, "source_evidence.rds")
  )
  summary <- fastkpc_full_cuda_phase10_campaign_summary(
    aggregate, comparison, representative$summary, producer,
    evidence_sha256, freeze, hardening, contracts
  )
  fastkpc_full_cuda_phase10_campaign_validate_summary(summary)

  fastkpc_full_cuda_phase10_campaign_write_table(
    comparison$graph_agreement, stage, "graph_agreement.csv"
  )
  fastkpc_full_cuda_phase10_campaign_write_table(
    comparison$sepset_agreement, stage, "sepset_agreement.csv"
  )
  fastkpc_full_cuda_phase10_campaign_write_table(
    comparison$n_edgetests, stage, "n_edgetests.csv"
  )
  fastkpc_full_cuda_phase10_campaign_write_table(
    comparison$candidate_deletions, stage, "deletion_trace.csv"
  )
  fastkpc_full_cuda_write_json(
    comparison$first_divergence, file.path(stage, "first_divergence.json")
  )
  fastkpc_full_cuda_phase10_campaign_write_table(data.frame(
    fallback_class = c("unknown", "approximate", "cpu-numerical"),
    count = 0L,
    accepted_for_phase10 = FALSE,
    stringsAsFactors = FALSE
  ), stage, "fallbacks.csv")
  fastkpc_full_cuda_phase10_campaign_write_table(
    fastkpc_full_cuda_phase10_campaign_stage_rows(runs),
    stage, "stage_timing.csv"
  )
  fastkpc_full_cuda_phase10_campaign_write_table(
    aggregate$raw_runs, stage, "raw_runs.csv"
  )
  fastkpc_full_cuda_phase10_campaign_write_table(
    fastkpc_full_cuda_phase10_campaign_statistics_table(aggregate),
    stage, "statistics.csv"
  )
  fastkpc_full_cuda_phase10_campaign_write_table(
    cases, stage, "case_results.csv"
  )
  fastkpc_full_cuda_phase10_campaign_write_table(
    near, stage, "near_alpha_results.csv"
  )
  phase8_rank <- file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "dcov_cuda_backend_v1",
    "rank_condition_results.csv"
  )
  fastkpc_full_cuda_phase10_campaign_require(
    file.copy(
      phase8_rank, file.path(stage, "rank_condition_results.csv")
    ),
    "Phase 10 campaign rank/condition evidence could not be staged"
  )
  fastkpc_full_cuda_phase10_campaign_write_table(
    fastkpc_full_cuda_phase10_campaign_cache_rows(runs),
    stage, "cache.csv"
  )
  fastkpc_full_cuda_phase10_campaign_write_table(
    fastkpc_full_cuda_phase10_campaign_machine_rows(runs),
    stage, "machine_samples.csv"
  )
  fastkpc_full_cuda_phase10_campaign_write_table(
    fastkpc_full_cuda_phase10_campaign_order(),
    stage, "campaign_order.csv"
  )
  fastkpc_full_cuda_phase10_campaign_write_table(
    fastkpc_full_cuda_phase10_campaign_evidence_inputs(runs, staging_dir),
    stage, "evidence_inputs.csv"
  )
  fastkpc_full_cuda_phase10_campaign_write_table(
    source_closure$table, stage, "source_closure.csv"
  )
  saveRDS(comparison$candidate_adjacency, file.path(stage, "adjacency.rds"))
  saveRDS(representative$sepsets, file.path(stage, "sepsets.rds"))
  saveRDS(representative$pMax, file.path(stage, "pmax.rds"))
  saveRDS(
    comparison$candidate_logical, file.path(stage, "logical_ci_trace.rds")
  )
  fastkpc_full_cuda_write_json(summary, file.path(stage, "summary.json"))
  writeLines(c(
    "# Full CUDA CI Phase 10 canonical promotion campaign",
    "",
    paste0("- edge count: ", summary$edge_count_candidate, " / ",
           summary$edge_count_reference),
    paste0("- SHD: ", summary$SHD),
    paste0("- cold median seconds: ", summary$cold_median_sec),
    paste0("- warm median seconds: ", summary$warm_median_sec),
    paste0("- baseline median seconds: ", summary$baseline_median_sec),
    paste0("- candidate/baseline ratio: ",
           summary$candidate_to_baseline_ratio),
    paste0("- absolute gate: ", summary$absolute_performance_gate),
    paste0("- relative gate: ", summary$relative_performance_gate),
    paste0("- repeatability gate: ", summary$repeatability_gate),
    "- sealed holdout: pending",
    paste0("- canonical campaign pass: ", summary$pass)
  ), file.path(stage, "summary.md"), useBytes = TRUE)
  fastkpc_full_cuda_write_json(
    producer, file.path(stage, "producer_identity.json")
  )
  fastkpc_full_cuda_write_json(
    backend$value, file.path(stage, "backend_configuration.json")
  )
  fastkpc_full_cuda_write_json(
    build$value, file.path(stage, "build_recipe.json")
  )
  fastkpc_full_cuda_campaign_freeze_json <- file.path(staging_dir, "freeze.json")
  fastkpc_full_cuda_phase10_campaign_require(
    file.copy(
      fastkpc_full_cuda_campaign_freeze_json,
      file.path(stage, "freeze.json")
    ),
    "Phase 10 campaign freeze JSON could not be staged"
  )
  writeLines(c(
    "bash fastkpc/tools/build_cuda_native.sh",
    "bash fastkpc/tools/run_full_cuda_ci_phase10_campaign.sh",
    paste0("freeze_identity_sha256=", freeze$freeze_identity_sha256),
    paste0("source_evidence_sha256=", evidence_sha256),
    paste0("native_binary_sha256=", native_identity$sha256)
  ), file.path(stage, "commands.txt"), useBytes = TRUE)
  writeLines(c(
    fastkpc_full_cuda_environment_lines(),
    paste0("native_binary_path=", native_identity$path),
    paste0("native_binary_sha256=", native_identity$sha256),
    paste0("freeze_identity_sha256=", freeze$freeze_identity_sha256),
    paste0("cpu_governor=", freeze$machine_identity$governor),
    "phase10_route=compatible.cuda/full_cuda-explicit",
    "CUDA_VISIBLE_DEVICES=0", "OPENBLAS_NUM_THREADS=1",
    "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
    "BLIS_NUM_THREADS=1", "VECLIB_MAXIMUM_THREADS=1"
  ), file.path(stage, "environment.txt"), useBytes = TRUE)

  excluded <- c(
    "manifest.json", "validator_attestations.json",
    "execution_receipts.json", "environment.txt", "commands.txt"
  )
  semantic_files <- sort(setdiff(
    list.files(stage, all.files = FALSE, no.. = TRUE), excluded
  ), method = "radix")
  payload_hashes <- setNames(lapply(
    file.path(stage, semantic_files), fastkpc_full_cuda_census_file_hash
  ), semantic_files)
  payload_sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(payload_hashes)
  )
  envelope <- fastkpc_full_cuda_phase35_identity_envelope(
    producer, payload_sha256
  )
  environment_sha256 <- fastkpc_full_cuda_census_file_hash(
    file.path(stage, "environment.txt")
  )
  timestamp <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  attestation <- fastkpc_full_cuda_phase35_validator_attestation(
    producer = producer,
    validator_source_closure_sha256 = source_closure$sha256,
    validator_semantic_version = "full-cuda-ci-phase10-campaign-validator-v1",
    validator_contracts = contracts,
    validation_timestamp_utc = timestamp,
    environment_sha256 = environment_sha256,
    validation_result = "PASS"
  )
  info <- file.info(stage, extra_cols = TRUE)
  inode <- if ("ino" %in% names(info)) as.character(info$ino[[1L]]) else
    "unavailable"
  receipt <- fastkpc_full_cuda_phase35_execution_receipt(
    producer = producer,
    pid = as.integer(Sys.getpid()),
    session_id = paste0("R-", Sys.getpid(), "-", format(Sys.time(), "%s")),
    cuda_context_id = "cuda-device-0-phase10-campaign",
    artifact_path = file.path(
      normalizePath(parent, winslash = "/", mustWork = TRUE),
      basename(output_dir)
    ),
    artifact_inode = inode,
    staging_path = normalizePath(stage, winslash = "/", mustWork = TRUE),
    recorded_at_utc = timestamp
  )
  fastkpc_full_cuda_write_json(
    list(attestations = list(attestation)),
    file.path(stage, "validator_attestations.json")
  )
  fastkpc_full_cuda_write_json(
    list(execution_receipts = list(receipt)),
    file.path(stage, "execution_receipts.json")
  )
  manifest <- list(
    schema_version = "full-cuda-ci-phase10-campaign-manifest-v1",
    artifact_kind = "promotion_351x48",
    claim_scope = "phase10-canonical-performance-repeatability",
    producer_semantic_envelope = envelope,
    payload_manifest_sha256 = payload_sha256,
    payload_file_sha256 = payload_hashes,
    semantic_file_count = length(payload_hashes),
    validator_attestations_file = "validator_attestations.json",
    volatile_receipt_file = "execution_receipts.json",
    environment_file = "environment.txt",
    environment_file_sha256 = environment_sha256
  )
  fastkpc_full_cuda_write_json(manifest, file.path(stage, "manifest.json"))
  fastkpc_full_cuda_phase10_validate_campaign_artifact(
    stage, verify_current_sources = TRUE
  )

  backup <- NULL
  if (dir.exists(output_dir)) {
    backup <- tempfile(".phase10-campaign-backup-", tmpdir = parent)
    fastkpc_full_cuda_phase10_campaign_require(
      file.rename(output_dir, backup),
      "Phase 10 prior campaign artifact could not be staged"
    )
  }
  if (!file.rename(stage, output_dir)) {
    if (!is.null(backup)) file.rename(backup, output_dir)
    stop("Phase 10 campaign artifact publication failed", call. = FALSE)
  }
  active <- FALSE
  validated <- tryCatch(
    fastkpc_full_cuda_phase10_validate_campaign_artifact(
      output_dir, verify_current_sources = TRUE
    ), error = identity
  )
  if (inherits(validated, "error")) {
    failed <- tempfile(".phase10-campaign-failed-", tmpdir = parent)
    file.rename(output_dir, failed)
    if (!is.null(backup)) file.rename(backup, output_dir)
    stop(conditionMessage(validated), call. = FALSE)
  }
  if (!is.null(backup)) unlink(backup, recursive = TRUE, force = TRUE)
  validated
}
