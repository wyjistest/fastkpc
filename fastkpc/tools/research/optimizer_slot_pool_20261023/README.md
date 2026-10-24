# Bounded L5 prepared-slot experiment

Research only. The production native and its source closure remain unchanged.
The companion includes the exact production small64 implementation with hidden
symbols, exports only the slot API, and must reproduce every GPU instruction
word of the qualified small64 object. Both modes compare every output bit to
the frozen 955-setup / 6,816-target L5 capture.

R: qualified production native, 955 resident prepared objects, 32 workers,
original request order. S: 32 persistent slots with capacity 8, the same actual
target counts, original request order, and the same numerical code. Geometry
is uploaded on the owning stream at each rebind. Owned pageable input packets
remain alive until every stream drains; no pinned-I/O or prefetch mechanism is
introduced. The complete coefficients/fitted/residual payload is copied D2D
to a preallocated 40,786,944-byte arena for the full layer,
and a precreated blocking event confirms completion before releasing the old
lease. Old tokens remain retained until verification to test stale generations.
All diagnostic shadows are downloaded after the timed work. No output arena
or slot object is created/freed while workers execute.
Full-layer host packet packing is measured once and conservatively charged to
each S lifecycle estimate; R's geometry packing is already inside its prepare
boundary. The work-time comparison excludes both variants' initial preparation.

The output arena is bounded by this captured layer, not a generic production
capacity design. Production consumers may require only coefficients; this
experiment saves full outputs to verify overwrite safety and complete parity.
No sorting, >32 workers, Stage B overlap, new optimizer, or runtime integration.

Predeclared gates, before GPU execution:

- Complete bitwise identity, route/rank/info/trajectory and input order.
- GPU numerical authority only; no CPU solver or fallback.
- Busy/invalid rebind rejection without mutation, stale token rejection, old
  token destruction unable to release a newer lease, and cancellation cleanup.
- Reduced real-shape CUDA memcheck: zero errors/leaks. Full-layer ordinary
  replay separately covers all inputs and resource closure.
- No application-visible allocation/free, stream/event or cuBLAS handle
  mutation during the active worker interval; all such resources close.
- Standalone 32-slot plus full-output arena creation costs <=1 GiB of physical
  free memory and retains >=2 GiB headroom. This excludes joint qualification
  with the production component cache/arena and is not a deployment guarantee.
- One warmup each, then six paired R/S or S/R cycles. S median work time must
  be <=1.15× R and S including complete pool/arena construction and teardown
  must beat R including its lifecycle. Otherwise CONDITIONAL/STOP, even if
  correctness passes. P's older ~108-second reference is contextual only.

GPU1, CPU affinity 0–19, CUDA connections=32, status-wait=1, BLAS/OMP/MKL=1.
GPU0 external work is allowed; all evidence is shared-host development.
Profiles/models for other layers are screening tools, not substitutes for this
measured slot rebind/output transfer qualification. Phase 10 ACTIVE, candidate
NOT_FROZEN, holdout SEALED, recommended=false, runtime rolling NOT_YET_GO.

Use a fresh artifact directory:

```sh
python3 -I fastkpc/tools/research/optimizer_slot_pool_20261023/build.py OUTPUT
python3 -I fastkpc/tools/research/optimizer_slot_pool_20261023/run.py OUTPUT checks
python3 -I fastkpc/tools/research/optimizer_slot_pool_20261023/run.py OUTPUT memcheck
python3 -I fastkpc/tools/research/optimizer_slot_pool_20261023/run.py OUTPUT capacity
python3 -I fastkpc/tools/research/optimizer_slot_pool_20261023/run.py OUTPUT campaign
python3 -I fastkpc/tools/research/optimizer_slot_pool_20261023/review_audit.py OUTPUT
python3 -I fastkpc/tools/research/optimizer_slot_pool_20261023/archive.py OUTPUT
```

Completed: all 14 GPU functions reproduce the qualified instruction words.
Busy/invalid binding, generation/lease checks and cancellation pass; reduced
memcheck reports zero errors/leaks. Fifteen full-layer runs plus reduced checks
compare 102,376 target output instances bitwise. Standalone slot+output creation
costs 564 MiB physically (252.53 MiB application buffers); GPU1's sampled total
peak in that process is 2,536 MiB, a different measurement boundary. Six timed
R/S work medians are 31.808618/31.828590 seconds; lifecycle medians are
33.582750/31.932535 seconds, with all six S lifecycle wins. Result:
GO_L5_BOUNDED_SLOT_PROTOTYPE; joint runtime capacity and integration remain
unqualified. See [the result](../../../docs/PHASE10_OPTIMIZER_SLOT_POOL_20261023.md).
