# L4 bounded refill: predeclared isolated experiment

Research only. No runtime integration, default change, new canonical elapsed,
promotion, real-data or final-PDAG qualification is authorized by this experiment.

## Identity and scope

Use runtime `33e2d3e`, native SHA
`2c8dd8af4ea021d8a784588717476f091a0132b9415067e4a3e115ca65bcd0ee`,
and the 34 original L4 captures (2,154 setups / 17,767 targets). All requests have
n=351, q=37, four penalties and total root rank 32. Actual target counts are
2,3,4,5,6,7,8,10,11,12,14,16,17,18. Do not pad, split or reorder cohorts.

W calls the production capacity multi optimizer with 32 workers and the 34
original windows. Only one window is resident. An observer preserves all three
device outputs (coefficients, fitted, residual) before releasing each output.
S uses 32 slots of capacity 18 and the same small64 implementation, with original
request order assigned by an atomic queue. Require every GPU function's normalized
SASS instruction words to match the source-bound production object.

Both variants save full outputs D2D into the same-size preallocated layer arena
and validate them after the timed layer. No shadow D2H between windows. W's
observer adds output retention inside the production batch call; explicitly report
that instrumentation. These are captured-layer timings, not canonical timings.
The full-lifecycle boundary charges packing, arena/pool/event creation/destruction,
request/result buffers, geometry upload, optimizer, output retention and lease
release. Fixture reading and delayed diagnostic shadow/comparison are excluded.

## Hard gates before timing

- Bitwise equality of all 14 floating vectors and all 19 integer vectors in the
  capture, including selected SP, optimizer trajectory counters, coefficients,
  fitted, residual, rank/route/info. Target identities/order exact. GPU numerical
  authority only; zero fallback/replay/terminal confirmation.
- Invalid/busy rebind does not mutate identity or generation. Released tokens
  remain stale after reuse; destroying an old token cannot release a new lease.
- Injected partial upload and partial detach poison/drain safely. Multi-worker
  cancellation drains all in-flight work and closes every lease/resource.
- Zero CUDA allocation/free, stream/event create/destroy, or BLAS create/destroy
  in the worker interval. All tracked resources close at each pass.
- Reduced checks cover first occurrence of all 14 capacities plus the layer
  endpoint. Ordinary checks include W, sequential S and parallel S. Memcheck
  covers sequential S at all capacities, small failure/lifetime/cancellation
  cases; full ordinary passes validate all captures. No sanitizer timing claim.
- Standalone 32-slot + full arena physical increment <= 1 GiB and at least
  2 GiB device headroom. Expected explicit application allocation is
  590,676,136 bytes. This does not qualify production-cache coexistence.

## Performance decision (declared before GPU execution)

GPU1, CPU affinity 0-19, CUDA connections 32, status wait enabled, BLAS/OMP/MKL
one thread. One complete W and S warmup, then six pairs alternating W/S and S/W.
GO for a separate opt-in runtime qualification only if every hard gate passes,
median S full-lifecycle / median W full-lifecycle <= 0.90, and all six paired
full-lifecycle differences favor S. Smaller or inconsistent improvement is
CONDITIONAL; near-flat/regression or a hard failure is STOP. Record work-only
timings too. No theoretical tail-model saving counts as measured gain.

GPU process telemetry and memory/power/clocks are retained. Stop only our own
process group if an unrelated GPU1 job appears. Any other-GPU activity is recorded.
Use immutable attempt prefixes and retain failed attempts. Build never overwrites
the production native. Commands:

```
python3 build.py ARTIFACT_DIR
python3 run.py ARTIFACT_DIR checks
python3 run.py ARTIFACT_DIR memcheck
python3 run.py ARTIFACT_DIR capacity
python3 run.py ARTIFACT_DIR campaign
python3 archive.py ARTIFACT_DIR
```
