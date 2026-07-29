source("fastkpc/R/full_cuda_ci_phase5_artifacts.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

setup_keys <- sprintf("setup-%03d", 0:127)
shard_ids <- rep(0:63, each = 2L)
parts <- lapply(0:15, function(partition_id) {
  fastkpc_full_cuda_phase5_shadow_partition(
    setup_keys, shard_ids, partition_id, 16L
  )
})

assert_true(
  all(vapply(parts, function(value) {
    identical(
      value$assignment_strategy,
      "authenticated-shard-modulo-v1"
    ) && length(value$shard_ids) == 4L &&
      all(value$shard_ids %% 16L == value$partition_id) &&
      length(value$setup_keys) == 8L
  }, logical(1L))),
  "Phase 5 partitions must own complete authenticated shards"
)

observed_keys <- unlist(lapply(parts, `[[`, "setup_keys"), use.names = FALSE)
observed_shards <- unlist(lapply(parts, `[[`, "shard_ids"), use.names = FALSE)
assert_true(
  length(observed_keys) == length(setup_keys) &&
    !anyDuplicated(observed_keys) &&
    identical(sort(observed_keys), sort(setup_keys)) &&
    identical(sort(observed_shards), 0:63),
  "Phase 5 shard-aligned partitions must have exact corpus coverage"
)

unpartitioned <- fastkpc_full_cuda_phase5_shadow_partition(
  setup_keys, shard_ids
)
assert_true(
  identical(
    unpartitioned$assignment_strategy,
    "all-authenticated-shards-v1"
  ) && identical(unpartitioned$setup_keys, setup_keys) &&
    identical(unpartitioned$shard_ids, 0:63),
  "Phase 5 unpartitioned execution must retain every authenticated shard"
)

cat("PASS Phase 5 authenticated-shard partition coverage\n")
