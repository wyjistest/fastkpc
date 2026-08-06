source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!isTRUE(tryCatch(
  fastkpc_cuda_available(),
  error = function(error) FALSE
))) {
  cat("SKIP Phase 10 fixed-SP root cache: CUDA unavailable\n")
  quit(save = "no", status = 0L)
}

cache_environment <- "FASTKPC_PHASE10_FIXED_SP_ROOT_CACHE"
old_cache_environment <- Sys.getenv(cache_environment, unset = NA_character_)
on.exit({
  if (is.na(old_cache_environment)) {
    Sys.unsetenv(cache_environment)
  } else {
    do.call(Sys.setenv, setNames(list(old_cache_environment), cache_environment))
  }
}, add = TRUE)

resource_snapshot <- function() {
  .Call("C_fixed_sp_cuda_test_resource_snapshot", PACKAGE = "fastkpc_cuda")
}

normalize_sepsets <- function(value) {
  unname(lapply(unname(value), function(row) {
    unname(lapply(unname(row), function(entry) sort(as.integer(entry))))
  }))
}

same_result <- function(left, right) {
  identical(left$adjacency, right$adjacency) &&
    identical(normalize_sepsets(left$sepsets),
              normalize_sepsets(right$sepsets)) &&
    identical(left$pMax, right$pMax) &&
    identical(as.integer(left$n.edgetests), as.integer(right$n.edgetests)) &&
    identical(left$levels[, setdiff(names(left$levels), "elapsed_ms")],
              right$levels[, setdiff(names(right$levels), "elapsed_ms")]) &&
    identical(left$tasks, right$tasks)
}

run_candidate <- function(data) {
  full_cuda_ci_one_call_cache_control_native("reset")
  precision_run_skeleton_full_cuda_native(
    data = data,
    alpha = 0.1,
    max_conditioning_size = 3L,
    index = 1,
    numCol = 35L,
    trace_level = "logical",
    compatible_cuda_strict = TRUE
  )
}

set.seed(10091)
n <- 90L
p <- 8L
common <- stats::rnorm(n)
data <- sapply(seq_len(p), function(column) {
  sqrt(0.65) * common + sqrt(0.35) * stats::rnorm(n)
})
colnames(data) <- paste0("v", seq_len(p))

Sys.setenv(FASTKPC_PHASE10_FIXED_SP_ROOT_CACHE = "0")
baseline <- run_candidate(data)
assert_true(
  !isTRUE(baseline$summary$fixed_sp_root_cache_enabled) &&
    baseline$summary$fixed_sp_root_cache_lookup_count == 0 &&
    baseline$summary$fixed_sp_root_cache_entries == 0 &&
    baseline$summary$fixed_sp_root_cache_device_bytes == 0,
  "disabled fixed-SP root cache performed physical cache work"
)

before <- resource_snapshot()
Sys.setenv(FASTKPC_PHASE10_FIXED_SP_ROOT_CACHE = "1")
cached <- run_candidate(data)
after <- resource_snapshot()
summary <- cached$summary

assert_true(
  same_result(baseline, cached),
  "fixed-SP root cache changed strict one-call output bits"
)
assert_true(
  isTRUE(summary$fixed_sp_root_cache_enabled) &&
    summary$fixed_sp_root_cache_runtime_count >= 1 &&
    summary$fixed_sp_root_cache_capacity_entries_per_runtime == 4096 &&
    summary$fixed_sp_root_cache_capacity_entries_total ==
      4096 * summary$fixed_sp_root_cache_runtime_count &&
    summary$fixed_sp_root_cache_capacity_bytes_total ==
      256 * 1024^2 * summary$fixed_sp_root_cache_runtime_count,
  "fixed-SP root cache capacity receipt is malformed"
)
assert_true(
  summary$fixed_sp_root_cache_lookup_count > 0 &&
    summary$fixed_sp_root_cache_hit_count > 0 &&
    summary$fixed_sp_root_cache_miss_count > 0 &&
    summary$fixed_sp_root_cache_lookup_count ==
      summary$fixed_sp_root_cache_hit_count +
        summary$fixed_sp_root_cache_miss_count &&
    summary$fixed_sp_root_cache_miss_count ==
      summary$fixed_sp_root_cache_insert_count +
        summary$fixed_sp_root_cache_bypass_count &&
    summary$fixed_sp_root_cache_bypass_count == 0 &&
    summary$fixed_sp_root_cache_identity_rejection_count == 0,
  "fixed-SP root cache lookup receipt is not conserved"
)
assert_true(
  summary$fixed_sp_root_cache_entries ==
      summary$fixed_sp_root_cache_insert_count &&
    summary$fixed_sp_root_cache_peak_entries ==
      summary$fixed_sp_root_cache_entries &&
    summary$fixed_sp_root_cache_device_bytes > 0 &&
    summary$fixed_sp_root_cache_device_bytes ==
      summary$fixed_sp_root_cache_peak_device_bytes &&
    summary$fixed_sp_root_cache_insert_d2d_bytes ==
      summary$fixed_sp_root_cache_device_bytes &&
    summary$fixed_sp_root_cache_hit_d2d_bytes > 0,
  "fixed-SP root cache device-memory receipt is malformed"
)

active_fields <- grep("_active_count$", names(before), value = TRUE)
assert_true(
  length(active_fields) > 0L &&
    identical(as.numeric(unlist(before[active_fields], use.names = FALSE)),
              as.numeric(unlist(after[active_fields], use.names = FALSE))),
  "fixed-SP root cache left a live CUDA resource after one-call teardown"
)

runtime_source <- paste(
  readLines("fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu", warn = FALSE),
  collapse = "\n"
)
assert_true(
  grepl("penalty_root_cache_identity_matches(", runtime_source,
        fixed = TRUE) &&
    grepl("prepared penalty root cache identity is malformed",
          runtime_source, fixed = TRUE) &&
    grepl("prepared penalty root cache requires identity Z and null H",
          runtime_source, fixed = TRUE),
  "fixed-SP root cache identity or scope guard is missing"
)

cat(
  "PASS Phase 10 fixed-SP root cache; lookups=",
  summary$fixed_sp_root_cache_lookup_count,
  " hits=", summary$fixed_sp_root_cache_hit_count,
  " entries=", summary$fixed_sp_root_cache_entries,
  " bytes=", summary$fixed_sp_root_cache_device_bytes,
  "\n", sep = ""
)
