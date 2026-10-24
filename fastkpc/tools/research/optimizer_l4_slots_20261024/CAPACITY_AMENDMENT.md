# Capacity protocol amendment, before any balanced timing

The original cold-context capacity attempt STOP is retained. Do not relabel its
exit code or claim the original protocol passed. Its gate ran after the full GPU
work but before shadow verification; it was not a successful qualification run.

A cold-context readout-only follow-up independently verifies all 17,767 target
outputs and closes all tracked resources. It measures 882 MiB after building the
32-slot pool plus output arena, 2,032 MiB after computation, and 1,156 MiB still
resident after all application allocations/streams/events/handles are destroyed.
Application allocation remains 590,676,136 bytes throughout computation.

A second fresh-process attribution run first executes the real production
optimizer on the captured 18-target request, verifies every output bit and
destroys its resources. That operation alone leaves 1,152 MiB resident. Replaying
all 2,154 setups / 17,767 targets through the unchanged slot prototype in that
context then measures a maximum **886 MiB** additional physical memory, all
outputs exact and every tracked resource closed. The exact driver/library
allocation responsible for retained context memory is not identified. These
measurements establish the common first-execution cost without assuming its cause.

The first gate mixed a common optimizer/context first-use cost with the bounded
pool's incremental cost. For a separate, amended standalone experiment, measure
the additional cost relative to a compute-warm production-native context. Keep
the same <=1 GiB incremental limit and >=2 GiB free limit. This change is declared
after memory attribution but **before any balanced performance comparison**.
The original README, protocol, failed attempt and both attribution runs remain
immutable. The original replay executable and numerical code remain unchanged.

The existing campaign first runs W and S full warmups. Every timed S pass still
enforces the same <=1 GiB guard both after construction and after computation.
Use one warmup per arm, six alternating W/S, S/W pairs, lifecycle median S/W
<=0.90 and six paired wins, with every original numerical/resource gate. No
post-hoc change to the performance threshold is allowed.

This qualifies only warm-context standalone capacity. Absolute process memory,
common cold-start cost and GPU telemetry remain reported. It does not qualify
coexistence with production component caches, downstream arenas or other contexts;
that is a separate mandatory gate before runtime integration. This amendment
does not remove the original L4 full-layer prepared-residency STOP.
