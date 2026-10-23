# Component lifecycle replay, 2026-10-21

Offline structural replay of the dcc.gamma `(target, S)` component lifecycle
for a level-wide precompute design. It answers the first gate of
[the structured-eig real replay report](../../../docs/PHASE10_STRUCTURED_EIG_REAL_REPLAY_20261020.md):
how many components must be resident at once, and what fraction of
precomputed components is never consumed. Results are recorded in
[`PHASE10_COMPONENT_LIFECYCLE_REPLAY_20261021.md`](../../../docs/PHASE10_COMPONENT_LIFECYCLE_REPLAY_20261021.md).

Nothing here runs on the GPU, changes the native, or predicts wall time. The
input is the saved canonical receipt of the 32-connection development run
(`fastkpc/artifacts/full_cuda_ci/exact_speedup_20261020/canonical-queues32.rds`,
native `c4f599ef…`, 880.659 s). Run from the repository root.

```sh
Rscript -e '
rds <- "fastkpc/artifacts/full_cuda_ci/exact_speedup_20261020/canonical-queues32.rds"
x <- readRDS(rds)
out <- "fastkpc/artifacts/full_cuda_ci/component_lifecycle_20261021"
dir.create(out, showWarnings = FALSE, recursive = TRUE)
write.csv(x$result$tasks, file.path(out, "canonical_queues32_tasks.csv"), row.names = FALSE)
write.csv(x$result$levels, file.path(out, "canonical_queues32_levels.csv"), row.names = FALSE)
write.csv(x$result$legacy_component_wave_group_trace, file.path(out, "canonical_queues32_wave_groups.csv"), row.names = FALSE)
s <- x$result$summary
keep <- sapply(s, function(v) is.atomic(v) && length(v) == 1)
df <- data.frame(key = names(s)[keep], value = sapply(s[keep], function(v) format(v, digits = 17)), stringsAsFactors = FALSE)
top <- data.frame(key = c("receipt_path", "receipt_schema", "receipt_workload", "receipt_native_sha256", "receipt_elapsed_sec"),
                  value = c(rds, x$schema, x$workload, x$native_sha256, format(x$elapsed_sec, digits = 17)), stringsAsFactors = FALSE)
write.csv(rbind(top, df), file.path(out, "canonical_queues32_summary.csv"), row.names = FALSE)'

python3 -I fastkpc/tools/research/component_lifecycle_20261021/replay_component_lifecycle.py \
  fastkpc/artifacts/full_cuda_ci/component_lifecycle_20261021
```

`replay_component_lifecycle.py` rebuilds every layer plan with the native
`make_layer_plan` enumeration (`fastkpc/src/skeleton_task_scheduler.cpp`):
edge-major, x->y orientation before y->x, conditioning sets in lexicographic
combination order. It replays the recorded tests and deletions and refuses to
continue unless, for every level, planned/replayed/ignored/deletion counts,
each recorded task identity (`task_index`, orientation, `S_key`), and the
per-level unique component miss counts of the wave trace all match. The
component totals must also equal the receipt's prefill counters (132,926
planned and 110,635 consumed target keys for levels >= 1; 110,683 cache
inserts in all).

It then simulates, per level, production / consumption / release of
components under four semantics-preserving consumption orders (each edge's
tasks stay in plan order; only the interleaving across edges differs):

- `canonical`: the native plan order itself (one edge at a time);
- `round_robin`: round r takes task r of every still-active edge;
- `frontier_largest_bucket`: the production frontier heuristic without its
  prepared-context preference;
- `smallest_ready_s`: always the lexicographically smallest ready S.

Production policies: `demand` (produce when first needed, an oracle lower
bound), `batch_oracle_B` (batches of B in first-executed-use order, oracle),
`batch_static_B` (batches of B in first *planned* appearance order of that
consumption order, skipping components already known dead; implementable),
and `level` (everything before the first test). A component is released as
soon as every planned task referencing it has executed or became unreachable
through an edge deletion.

Feedback boundary: entries without a suffix apply each test's outcome
immediately after the test (`feedback = "task"`). The `round_robin` order
also carries `*_round_feedback` entries, which model a batched consumer that
prepares a whole round before any of its results exist: every component the
round needs is produced before the round starts, and all reference
resolutions, deletions and releases are applied only after the round
completes. Components used by an in-flight round are therefore never
released early. `speculative_s_major_640` records the structure of a fully
speculative S-major physical schedule for comparison only.

Output: `component_lifecycle_replay.json` in the artifact directory, with
the receipt's top-level `native_sha256`, the receipt file SHA-256, input CSV
SHA-256 values, cross-checks, a `capacity_requirements` block (reset-only
versus in-level reclaim entries and bytes per policy) and every per-level
policy result. The script refuses to run when the summary export lacks the
receipt's `receipt_native_sha256`. Top-level receipt rows carry a
`receipt_` prefix so they cannot collide with same-named summary fields.
