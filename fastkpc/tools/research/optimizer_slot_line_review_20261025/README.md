# Optimizer slot-line review tools (2026-10-25)

Review diagnostics for [PHASE10_OPTIMIZER_SLOT_LINE_REVIEW_20261025.md](../../../docs/PHASE10_OPTIMIZER_SLOT_LINE_REVIEW_20261025.md).
Read-only with respect to production source, native and defaults. GPU1 only.

Artifacts: `fastkpc/artifacts/full_cuda_ci/optimizer_slot_line_review_20261025/` (local).

## Full-run trace of the production configuration

```sh
A=fastkpc/artifacts/full_cuda_ci/optimizer_slot_line_review_20261025
H=/usr/local/cuda-12.5/targets/x86_64-linux
g++ -O2 -std=c++17 -shared -fPIC ktrace.cpp -I$H/include -L$H/lib -lcupti \
  -Wl,-rpath,$H/lib -o $A/ktrace.so
sh trace.sh $A                       # one canonical run -> $A/ktrace-run/
python3 -I analyze.py $A/ktrace-run  # windows, spans, HOL late starts, slot gaps
python3 -I predict.py $A/ktrace-run  # request-order busy-time refill models, L8 launches
                                     # by correlation id, no-kernel gaps, device-property calls
python3 -I first_use.py $A/ktrace-run
```

Request order: window-mode streams are created in request order (consecutive stream IDs
per window, asserted in `predict.py`); slot levels use burst start (fetch) order. The
first version ordered window setups by first-kernel start; its outputs are kept as
`analysis-v1-start-order.json` / `predict-v1-start-order.json`.

`run_profile.R` runs the committed `4a1ff6b` configuration (S2, status-wait, L5 slots,
L4 workers32, L4 slots) on native `037b92f1…` and checks p-values, pMax, adjacency and
n.edgetests bitwise against Codex's L4-slot ON1 receipt. `ktrace.cpp` is a minimal CUPTI
injection library (`CUDA_INJECTION64_PATH`): concurrent-kernel and runtime-API activity
only, fixed binary records in `$KTRACE_DIR`.

`profile.sh` is the first attempt with nsys 2024.2.3 (`--trace=cuda`). It segfaulted in
nsys's bundled `libcupti.so.12.5` during L3 window 1; evidence is kept in `nsys-crash/`.

## Cache-ON comparison (device-properties cache review)

```sh
FASTKPC_CUDA_DEVICE_PROPERTIES_CACHE=1 FASTKPC_REVIEW_NATIVE_SHA256=994c43d3…20df9 \
  sh trace.sh $A ktrace-run-cache-on
python3 -I compare_windows.py $A/ktrace-run $A/ktrace-run-cache-on > $A/compare-off-vs-cache-on.json
```

`compare_windows.py` reports, per window-mode level, span vs. longest setup busy time
(time beyond the critical setup), late starts and channel spread, plus the L8 launch
pattern. See `docs/PHASE10_DEVICE_PROPERTIES_CACHE_REVIEW_20261025.md`.

## Probe

```sh
/usr/local/cuda-12.5/bin/nvcc -O2 -arch=sm_89 -o $A/lmem_growth_probe lmem_growth_probe.cu
CUDA_VISIBLE_DEVICES=1 $A/lmem_growth_probe PRESIZE SCENARIO   # PRESIZE 0|1, SCENARIO 0-5
CUDA_MODULE_LOADING=EAGER CUDA_VISIBLE_DEVICES=1 $A/lmem_growth_probe 0 3
```

Scenario 0: a larger-stack launch while a kernel runs. 1: same-function followers.
2/3/4: a not-yet-loaded function before each follower. 5: as 3 after
`cudaFuncGetAttributes` preload. Outputs: `lmem_growth_probe*.out`.

## Same-channel serialization (L6 runtime final review, 2026-10-27)

```sh
A=fastkpc/artifacts/full_cuda_ci
python3 -I same_channel.py $A/optimizer_slot_line_review_20261025/ktrace-run \
  $A/optimizer_slot_line_review_20261025/ktrace-run-cache-on \
  $A/late_optimizer_20261025/trace-off-trace $A/late_optimizer_20261025/trace-on-trace \
  > $A/optimizer_slot_line_review_20261025/same-channel.out
```

For every window-mode L6–L8 window it reports the critical (longest) persistent kernel,
the streams sharing its hardware channel, and the serial same-channel chain that ends last
(the window span). For every level, window or slot mode, it counts persistent-kernel starts
while another stream's persistent kernel runs on the same channel (CUPTI assigns the channel
per launch); late-start matches are compared with a shifted-time chance baseline. Use it on a
prototype trace to check whether slot mode still shares channels. `trace-off`/`trace-on` are
Codex's late-optimizer traces (same ktrace format). See
`docs/PHASE10_OPTIMIZER_L6_SLOT_RUNTIME_REVIEW_20261026.md`, final section.

```sh
python3 -I history.py $A/optimizer_slot_line_review_20261025/history-l7-l8.json
```

Inclusion list of historical L7/L8 optimizer times from every canonical `report.json`,
deduplicated by receipt SHA (continuation trees copy receipts). The rows mix natives,
configurations, traced and diagnostic runs: they show recurrence, not current bias or gain.
