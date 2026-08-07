# Strict method route-wait diagnostic

## Scope

This diagnostic attributes the host waits inside authoritative fixed-SP
submission without changing solver routes, floating-point work, method
ordering, RNG consumption, or cache policy. It does not implement deferred QR.

The method ticket/finalize layer still has zero intermediate host event waits.
The new opt-in instrumentation measures the existing Cholesky, QR, SVD, and
output-status waits inside `mgcv_fixed_sp_runtime` and propagates them into the
per-batch strict-method critical-path trace.

Enable it with:

```bash
FASTKPC_STRICT_METHOD_ROUTE_WAIT_DIAGNOSTICS=1
FASTKPC_STRICT_METHOD_CRITICAL_PATH_TRACE_CAPACITY=300000
```

The checkpoint fields are host boundary times. They include queued GPU work,
compact diagnostic D2H completion, and event wait time. They are not pure
kernel timings or pure D2H timings.

## Canonical development run

The diagnostic was run once on the canonical 351 x 48 `dcc.perm` workload:

```bash
FASTKPC_PHASE10_SETUP_OPTIMIZER_PIPELINE=1 \
FASTKPC_PHASE10_SETUP_OPTIMIZER_PRODUCER_DELAY_US=5000 \
FASTKPC_STRICT_METHOD_ROUTE_WAIT_DIAGNOSTICS=1 \
FASTKPC_STRICT_METHOD_CRITICAL_PATH_TRACE_CAPACITY=300000 \
Rscript fastkpc/tools/run_strict_ci_method_351x48_candidate.R \
  dcc.perm /tmp/fastkpc_dcc_perm_route_wait_diag_20260807.rds
```

Provenance:

```text
base HEAD                         38933cf
dataset SHA-256                   e03cbfafed3336f3da7725878e92af24556b5ebdc6227b94ea84469e62e94036
diagnostic native SHA-256         ccbf0f5a1c1fabe70edf2468acef5fc4e26290f8ab088d9543cb22e55dcd5abb
local diagnostic payload SHA-256  06e02957c7d6c5acc7fff5660d969de2e32d08c35588e40a3a36562e270f6dd6
trace rows / capacity / overflow  229675 / 300000 / 0
outer elapsed                     838.066 s
```

This is dirty-source diagnostic evidence: the native library was built from
the source changes documented by this commit before the commit existed. The
payload is local and ignored by Git. This is not clean promotion evidence.

The existing candidate runner passed all development gates:

```text
skeleton CI tests                 229675
maximum absolute p-value diff     0
decision flips                    0
graph / sepset / pMax mismatch    0
n.edgetests mismatch              0
final RNG mismatch                0
authority / fallback failures     0
```

## Route census

```text
total batches                     229675
residual-cache all-hit batches    134727
nonblocking preparation submits   186877
blocking preparation submits       42798

planned Cholesky batches            2220
  physical                            47
  cache all-hit                     2173

planned QR batches                173089
  physical QR solves               42751
  cache all-hit                   130338
  QR accepted targets             346178
  QR -> SVD targets                    0

planned direct-SVD batches         54366
  deferred physical SVD            52150
  cache all-hit                     2216

mixed planned-route batches            0
```

QR is therefore accepted for every canonical target in this run. The
reroute-tail risk is zero for this dataset, but the production implementation
must still preserve and test QR-to-SVD behavior.

## Host attribution

```text
preparation submit host boundary       459.534485 s
  blocking submits                      44.514251 s
  nonblocking submits                  415.020234 s

fixed-SP residual solve boundary       452.420016 s
QR checkpoint host waits                31.570152 s
output-status host waits                 0.071214 s
route-resolution CPU                     0.016044 s
output-status resolution CPU             0.015936 s
residual metadata resolution             0.193926 s
component submission host boundary        4.978679 s
component CUDA                           41.171309 s
finalization host boundary               59.062917 s
final compact-result wait                 0.957438 s
post-compact finalization                 1.213440 s
permutation generation + SHA            138.826090 s
```

QR checkpoint waits are 70.9% of the 44.514-second blocking-submit boundary,
but only 6.9% of the full 459.534-second submit boundary. The much larger
415.020-second nonblocking host boundary is not a QR checkpoint pool. It is
primarily the cost of submitting work for cache-hit and deferred-SVD cohorts.

For physical QR batches, the wait and permutation distributions in
milliseconds are:

```text
metric           p50       p90       p95       p99       max
QR wait       0.757910  0.765372  0.765497  0.765749  0.863375
permutation   0.597132  0.625288  0.651551  0.666017  0.910536
intersection  0.597021  0.624169  0.651299  0.665622  0.719571
```

## Deferred-QR opportunity

The implemented per-batch raw opportunity is:

```text
sum_i min(QR_checkpoint_wait_i, permutation_i) = 25.312005 s
```

Including the small output-status and route-resolution boundaries raises the
raw intersection to 25.316326 seconds.

The currently proposed deferred-QR shape resolves QR metadata after
permutation generation and only then submits component work. That moves
component work out of the current permutation overlap. A conservative
sequential model is therefore:

```text
raw QR/status intersection                    25.316326 s
- component/permutation overlap moved to tail  7.682454 s
= net model upper bound                       17.633872 s
```

This 17.634-second value is 2.10% of the observed 838.066-second outer run.
It is still an upper bound. It does not deduct concurrency slowdown, new
metadata/finalize overhead, failure-path machinery, or run-to-run variance.

## Decision

The diagnostic supports a conditional isolated deferred-QR prototype because:

- QR checkpoint waits dominate the genuinely blocking submit cohorts;
- the per-batch intersection is measurable;
- canonical QR acceptance is 100%;
- canonical QR-to-SVD reroutes are zero.

It does not support treating deferred QR as the main shared optimization:

- the raw QR opportunity is only 25.312 seconds;
- moving component work reduces the modeled upper bound to 17.634 seconds;
- the shared fixed-SP residual boundary remains about 452 seconds;
- the 415-second nonblocking submit boundary is outside the QR checkpoint
  opportunity.

Production integration remains gated on exact route/status/residual/p-value/RNG
parity and a repeatable full outer-time reduction beyond run variance. If an
isolated prototype does not preserve most of the 17.634-second modeled upper
bound, it should stop. Deeper stable-SVD residual work remains the shared
high-impact route for `hsic.gamma`, `dcc.perm`, and `hsic.perm`.

Phase 10 remains active. The candidate is not frozen, the recommended route is
unchanged, and the sealed holdout remains closed.
