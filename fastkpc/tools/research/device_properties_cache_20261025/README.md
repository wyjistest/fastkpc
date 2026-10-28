# Device-properties cache qualification

Narrow, default-off experiment authorized after the 2026-10-25 slot-line review.
Only `FASTKPC_CUDA_DEVICE_PROPERTIES_CACHE` changes between modes. L4/L5 slots,
S2, status wait and the retained component stack stay enabled. No preloading,
L3/L6 scheduling or Stage B changes. Production numerical authority stays GPU.

All nine first-party properties-query sites use one process-wide implementation
in `cuda_status.cpp`, shared across all capacity translation units. Successful
queries alone are cached by CUDA logical device ID. Each process freezes the
0/1 setting at first use (unset/empty defaults to 0; other values fail). Device
visibility is fixed before CUDA initialization. Static metadata is not a device
health check; existing availability, count/current-device, set-device and
synchronization checks remain live. Failed misses return the original CUDA
status, retain caller error messages, and can be retried. The cache owns only
host metadata, with no CUDA resource or teardown operation. Third-party library
queries are outside the cache and its counters.

Predeclared gates (before measurements):

- Native concurrent first access, independent logical devices, failed queries
  and retry, invalid configuration, live availability/errors and metadata parity.
- All rebuilt CUDA device instruction words match the accepted parent objects.
- Full canonical 240,498 CI and pMax bits, task/deletion/skeleton/sepsets/count,
  RNG, physical cohorts, Gram order, optimizer counters and resource closure
  match the accepted L4-ON reference. CPU authority/fallback/large D2H stay zero.
- First-party cache ON issues one successful properties query for the one
  visible device per fresh process; OFF queries each request; failures zero.
- Six fresh-process, uninjected GPU1 runs: OFF1/ON1/ON2/OFF2/OFF3/ON3. Both GPUs
  otherwise idle, CPU affinity 0-19, one BLAS/OMP thread, same binary/sources.
  Persistence and offline authority comparison are outside elapsed time.
- Development GO requires all three paired wins and an outer median saving of
  at least max(2 seconds, 0.5% of OFF median, larger within-mode elapsed range).
  A positive median saving below this gate is CONDITIONAL, not a default change;
  no median improvement is STOP. API totals are not predicted wall-time savings.

Evidence is audited incremental source closure, not a clean producer/full rebuild
or promotion. Candidate NOT_FROZEN, Phase 10 ACTIVE, holdout SEALED and
recommended=false remain unchanged. Old evidence/tools are immutable.

Run from the repository root, passing a fresh artifact directory to each tool:
`build.py`, `checks.py`, `campaign.py`, then `archive.py`.
