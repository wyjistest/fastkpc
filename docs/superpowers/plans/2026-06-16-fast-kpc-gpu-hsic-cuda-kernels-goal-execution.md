# Fast kPC GPU HSIC CUDA Kernels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Replace the current HSIC CUDA CPU-resolution path with real opt-in GPU HSIC gamma and fixed-seed HSIC permutation CI execution for fastkpc skeleton and WAN-PDAG workflows, while preserving the existing default dCov gamma behavior and keeping `kpcalg/R` unchanged.

**Architecture:** Keep native CPU HSIC as the correctness oracle and fallback, but add CUDA kernels underneath the existing CI method abstraction. Implement GPU HSIC in staged layers: dense RBF Gram construction, centering/statistic reduction, batched pair execution, fixed-seed permutation on GPU, scheduler integration, WAN-PDAG orientation integration, diagnostics, validation, reports, and performance guardrails. `engine="cuda", ci_method="hsic.gamma"` and `engine="cuda", ci_method="hsic.perm"` must record a CUDA HSIC backend only when GPU kernels actually execute; otherwise diagnostics must explicitly record CPU fallback.

**Tech Stack:** R 4.4.1, Rcpp/RcppArmadillo, C++17, CUDA C++ with NVCC, existing `fastkpc/src/hsic_cpu.cpp`, `fastkpc/src/ci_method.cpp`, `fastkpc/src/skeleton_engine_cuda.cpp`, `fastkpc/src/cuda/dcov_batch_cuda.cu`, existing scheduler diagnostics, validation campaign/report/CLI tooling, and legacy reference behavior from `kpcalg/R/hsicgamma.R` and `kpcalg/R/hsicperm.R`.

---

## Goal Objective For Codex

Use this exact goal objective when starting a Codex goal:

```text
Build the next staged fast kPC backend slice from docs/superpowers/plans/2026-06-16-fast-kpc-gpu-hsic-cuda-kernels-goal-execution.md: implement real CUDA HSIC gamma and fixed-seed HSIC permutation kernels for fastkpc CI tests, integrate them into CUDA skeleton batching, scheduler diagnostics, WAN-PDAG orientation residual-vs-S tests, validation campaigns, reports, CLIs, benchmarks, and documentation, preserve default dcc.gamma behavior, retain native CPU HSIC fallback with explicit diagnostics, and keep kpcalg/R files unchanged.
```

Recommended token budget for the goal run: `900000`.

Do not mark the goal complete until every item in "Completion Criteria" is proven by current-state evidence. Mark the goal blocked only if the same local blocker repeats for three consecutive goal turns and no meaningful implementation or validation work remains possible.

## Baseline From Previous Goal

The previous goal implemented:

```text
Native CPU HSIC gamma
Native CPU HSIC permutation
ci_method = dcc.gamma / hsic.gamma / hsic.perm public API
CPU skeleton CI method dispatch
CUDA HSIC CPU-resolution with diagnostics
WAN-PDAG HSIC pipeline support through CPU evaluator
Validation campaign/report/CLI CI method fields
HSIC validation and benchmark helpers
README/report documentation
kpcalg/R MD5 unchanged
```

## Current Progress Notes

Status after the first implementation slice of this goal:

```text
Raw CUDA HSIC gamma wrapper exists and reports backend "cuda-hsic".
Raw CUDA HSIC fixed-seed permutation wrapper exists and reports backend "cuda-hsic".
CUDA skeleton hsic.gamma now executes real CUDA HSIC batches and reports ci_backend "cuda-hsic".
CUDA skeleton hsic.perm now executes real CUDA HSIC batches when permutation_params$seed is explicit.
CUDA skeleton hsic.perm without seed falls back to native-cpu with reason
"CUDA HSIC permutation requires explicit seed in this stage".
Layer scheduler HSIC CI batches are labeled kind = "hsic" and do not increment dcov_batches.
Existing public API and WAN-PDAG skeleton wrapper tests have been migrated away from the old CPU-resolution expectation.
kpcalg/R MD5 remains unchanged.
```

Current verified commands:

```bash
bash fastkpc/tools/clean_cuda_native.sh && bash fastkpc/tools/build_cuda_native.sh
Rscript fastkpc/tests/test_hsic_cuda_kernel_math.R
Rscript fastkpc/tests/test_hsic_cuda_permutation.R
Rscript fastkpc/tests/test_hsic_cuda_skeleton_backend.R
Rscript fastkpc/tests/test_hsic_cuda_scheduler_diagnostics.R
Rscript fastkpc/tests/test_hsic_cuda_skeleton_permutation.R
Rscript fastkpc/tests/test_hsic_skeleton_cuda_resolution.R
Rscript fastkpc/tests/test_hsic_native_gamma.R
Rscript fastkpc/tests/test_hsic_native_permutation.R
Rscript fastkpc/tests/test_hsic_skeleton_cpu.R
Rscript fastkpc/tests/test_fastkpc_ci_method_public_api.R
Rscript fastkpc/tests/test_hsic_wanpdag_pipeline.R
cd kpcalg && md5sum -c MD5 | rg '^R/'
```

Important remaining scope:

```text
WAN-PDAG orientation residual-vs-S tests still need native CUDA HSIC orientation integration.
Public API top-level config/diagnostics still need explicit cuda_hsic_requested/cuda_hsic_used fields.
Validation campaign/report/CLI CUDA HSIC artifacts are not complete.
CUDA HSIC benchmark helpers and memory/size fallback tests remain.
Completion Criteria below are not yet fully satisfied.
```

Important current behavior:

```text
engine="cpu", ci_method="hsic.gamma" -> native-cpu HSIC
engine="cpu", ci_method="hsic.perm" -> native-cpu HSIC permutation
engine="cuda", ci_method="dcc.gamma" -> current CUDA batched dCov path
engine="cuda", ci_method="hsic.gamma" -> native-cpu CPU-resolution reason
engine="cuda", ci_method="hsic.perm" -> native-cpu CPU-resolution reason
```

The next goal changes only the last two rows. They should become:

```text
engine="cuda", ci_method="hsic.gamma" -> cuda-hsic when supported
engine="cuda", ci_method="hsic.perm" -> cuda-hsic when fixed-seed GPU permutation is requested and supported
```

Fallback remains required:

```text
If CUDA is unavailable, memory is insufficient, sample size exceeds configured GPU limits,
or seed/RNG requirements cannot be satisfied, resolve to native-cpu and record the exact reason.
```

## Non-Negotiable Constraints

- Do not modify `kpcalg/R/*.R`.
- Do not change default `fast_kpc()` behavior. Default remains `ci_method="dcc.gamma"`.
- Do not change dCov CUDA kernels except where shared scheduling diagnostics need new CI method labels.
- Do not remove native CPU HSIC. It is the oracle and fallback path.
- Do not claim CUDA HSIC is used unless kernels actually executed.
- `ci_backend = "native-cpu"` always means the CI test was resolved to the CPU native implementation. It must never be described as a GPU kernel, CUDA HSIC, or partial GPU HSIC execution.
- `ci_backend = "cuda-hsic"` may be recorded only after `hsic_gamma_batch_cuda()` or `hsic_permutation_batch_cuda()` has executed successfully for at least one CI pair in the current workflow.
- Any `engine="cuda"` + HSIC path that cannot execute HSIC kernels must return `ci_backend = "native-cpu"` with a concrete `ci_backend_reason`; this is an explicit fallback, not a CUDA success state.
- Do not use R RNG on GPU. GPU permutation is supported only with deterministic explicit seeds in this goal.
- Do not require HSIC graph equality with dCov graph equality. Different CI methods may produce different graphs.
- Do not introduce multi-GPU scheduling.
- Do not replace exported `kpcalg::kpc()`.
- Do not initialize git if this workspace is not already a git repository.

## Backend Truthfulness Contract

This goal exists because the previous CUDA HSIC state was intentionally conservative:

```text
engine="cuda", ci_method="hsic.gamma" -> native-cpu
engine="cuda", ci_method="hsic.perm" -> native-cpu
```

That state is CPU fallback. It is correct as a diagnostic baseline, but it is not GPU HSIC.
The next goal must replace it with real CUDA execution where supported:

```text
raw fast_hsic_gamma_cuda() successful kernel execution -> backend "cuda-hsic"
raw fast_hsic_perm_cuda() successful kernel execution -> backend "cuda-hsic"
fast_skeleton_cuda_backend(..., ci_method="hsic.gamma") successful CI batches -> ci_backend "cuda-hsic"
fast_skeleton_cuda_backend(..., ci_method="hsic.perm", seed=<explicit>) successful CI batches -> ci_backend "cuda-hsic"
WAN-PDAG CUDA HSIC residual-vs-S orientation successful batches -> ci_backend "cuda-hsic"
```

Required anti-regression tests:

```text
1. A skeleton HSIC gamma CUDA test must fail if ci_backend is "native-cpu" after a successful CUDA build.
2. A skeleton HSIC gamma CUDA test must fail if ci_hsic_gamma_cuda_tests is zero.
3. A skeleton HSIC gamma CUDA test must fail if ci_hsic_cuda_batches is zero.
4. A fixed-seed skeleton HSIC permutation CUDA test must fail if ci_backend is "native-cpu".
5. A seedless skeleton HSIC permutation CUDA test must require native-cpu fallback with reason
   "CUDA HSIC permutation requires explicit seed in this stage".
6. A WAN-PDAG CUDA HSIC orientation test must fail if orientation diagnostics claim cuda-hsic
   without positive CUDA HSIC test and batch counters.
7. Raw CUDA wrappers must report "cuda-hsic"; raw wrappers must not call CPU HSIC and then
   relabel the result as CUDA.
```

Implementation guardrail:

```text
Do not set ci_backend, backend, or orientation ci_backend to "cuda-hsic" at API boundaries.
Set it only from successful CUDA HSIC batch result diagnostics, then propagate that value upward.
CPU fallback paths must originate from explicit resolution branches and carry a reason string.
```

## Design Targets

### Expanded Goal Targets

This goal should leave the next worker with more than a kernel implementation. It should
produce a truthful, inspectable CUDA HSIC backend across the fastkpc workflow surface.

Primary targets:

```text
1. Raw CUDA HSIC gamma wrapper executes CUDA kernels and reports backend "cuda-hsic".
2. Raw CUDA HSIC permutation wrapper executes GPU replicate reductions and reports backend "cuda-hsic".
3. CUDA skeleton hsic.gamma uses CUDA HSIC batches instead of CPU-resolution fallback.
4. CUDA skeleton hsic.perm uses CUDA HSIC batches only when an explicit seed is supplied.
5. Seedless CUDA hsic.perm remains native-cpu with an explicit fallback reason.
6. Layer scheduler records HSIC CI batch rows as kind = "hsic", not "dcov".
7. dCov CUDA behavior and default dcc.gamma behavior remain unchanged.
8. CPU HSIC remains the correctness oracle and fallback implementation.
9. Public API results expose enough diagnostics for users to distinguish cuda-hsic from native-cpu fallback.
10. Validation campaign and report artifacts preserve ci_method, ci_backend, fallback reason, and CUDA HSIC counters.
```

Additional implementation targets:

```text
1. Add per-result CUDA HSIC counters for tests, pairs, batches, fallback tests, memory, max_n, and max_batch_pairs.
2. Add scheduler diagnostics that make mixed residual/CI batching readable.
3. Add WAN-PDAG orientation CUDA HSIC support for residual-vs-S tests.
4. Add fixed-seed reproducibility checks for raw permutation and skeleton permutation.
5. Add memory/size fallback tests so oversized HSIC workloads do not crash the process.
6. Add CLI flags to request CUDA HSIC limits and print whether CUDA HSIC was actually used.
7. Add validation CSVs that isolate CUDA HSIC backend status and CPU fallbacks.
8. Add benchmark helpers that compare CPU HSIC and CUDA HSIC without brittle universal speedup gates.
9. Add docs explaining that native-cpu under engine="cuda" is fallback, not GPU execution.
10. Keep `kpcalg/R` MD5 unchanged through the entire goal.
```

Exit targets for this goal:

```text
1. No test can pass by relabeling CPU HSIC as cuda-hsic.
2. Every cuda-hsic success path has positive CUDA HSIC batch/test counters.
3. Every fallback path has backend native-cpu and a non-empty reason.
4. Raw, skeleton, scheduler, WAN-PDAG, public API, campaign/report, CLI, benchmark, and docs tests all pass.
5. A final MD5 check proves `kpcalg/R/*.R` was not modified.
```

### CI Backend Resolution

Final expected backend resolution:

```text
dcc.gamma + CPU engine -> native-cpu
dcc.gamma + CUDA engine -> cuda-dcov
hsic.gamma + CPU engine -> native-cpu
hsic.gamma + CUDA engine + CUDA available + within limits -> cuda-hsic
hsic.gamma + CUDA engine + unsupported condition -> native-cpu with reason
hsic.perm + CPU engine -> native-cpu
hsic.perm + CUDA engine + explicit seed + within limits -> cuda-hsic
hsic.perm + CUDA engine + seed NULL -> native-cpu with reason "CUDA HSIC permutation requires explicit seed in this stage"
hsic.perm + CUDA engine + unsupported condition -> native-cpu with reason
```

Required result fields:

```text
skeleton$ci_method
skeleton$ci_backend
skeleton$ci_backend_requested
skeleton$ci_backend_reason
skeleton$ci_diagnostics$ci_hsic_gamma_cuda_tests
skeleton$ci_diagnostics$ci_hsic_perm_cuda_tests
skeleton$ci_diagnostics$ci_hsic_cuda_batches
skeleton$ci_diagnostics$ci_hsic_cuda_pairs
skeleton$ci_diagnostics$ci_hsic_cuda_fallback_tests
skeleton$ci_diagnostics$ci_hsic_cuda_memory_bytes
skeleton$ci_diagnostics$ci_hsic_cuda_max_n
skeleton$ci_diagnostics$ci_hsic_cuda_max_batch_pairs
orientation$ci_method
orientation$ci_backend
orientation$ci_diagnostics$regrvonps_hsic_gamma_cuda_tests
orientation$ci_diagnostics$regrvonps_hsic_perm_cuda_tests
orientation$ci_diagnostics$regrvonps_hsic_cuda_batches
diagnostics$ci_method_available
diagnostics$cuda_hsic_available
diagnostics$cuda_hsic_reason
```

### Mathematical Contract

The GPU HSIC math must match native CPU dense HSIC, not legacy incomplete Cholesky exactly:

```text
sigma = 1 / sig
K_ij = exp(-sigma * (x_i - x_j)^2)
L_ij = exp(-sigma * (y_i - y_j)^2)
Kc = H K H
Lc = H L H
HSIC = sum(Kc * Lc) / n^2
```

Gamma approximation:

```text
mux = off-diagonal mean of K
muy = off-diagonal mean of L
mean = (1 + mux * muy - mux - muy) / n
variance = 2*(n-4)*(n-5)/(n*(n-1)*(n-2)*(n-3)) * sum(Kc^2) * sum(Lc^2) / n^4
shape = mean^2 / variance
scale = variance / mean
p.value = pgamma(HSIC, shape, scale, lower.tail = FALSE)
```

Permutation:

```text
observed = HSIC(Kc, Lc)
for each replicate r:
  permute Lc by the same row/column permutation
  statistic[r] = sum(Kc[i,j] * Lc[perm[i], perm[j]]) / n^2
if include_observed:
  p.value = mean(c(statistic, observed) >= observed)
else:
  p.value = mean(statistic >= observed)
```

Tolerance:

```text
GPU gamma statistic vs CPU statistic: absolute difference <= 1e-8 for n <= 256
GPU gamma p-value vs CPU p-value: absolute difference <= 1e-7 for n <= 256 when gamma parameters are valid
GPU permutation observed statistic vs CPU observed statistic: absolute difference <= 1e-8
GPU permutation fixed-seed replicate vector vs CPU fixed-seed vector: exact order equality is not required unless the same RNG algorithm is intentionally shared; p-value equality is required for GPU repeated runs with the same seed.
GPU fixed-seed repeated pMax: absolute difference <= 1e-12 within the same GPU implementation
```

## Files To Create Or Modify

Create:

```text
fastkpc/src/cuda/hsic_batch_cuda.hpp
fastkpc/src/cuda/hsic_batch_cuda.cu
fastkpc/src/hsic_batch_types.hpp
fastkpc/R/hsic_cuda_validation.R
fastkpc/tests/test_hsic_cuda_kernel_math.R
fastkpc/tests/test_hsic_cuda_permutation.R
fastkpc/tests/test_hsic_cuda_skeleton_backend.R
fastkpc/tests/test_hsic_cuda_scheduler_diagnostics.R
fastkpc/tests/test_hsic_cuda_skeleton_permutation.R
fastkpc/tests/test_hsic_cuda_wanpdag_orientation.R
fastkpc/tests/test_hsic_cuda_public_api.R
fastkpc/tests/test_hsic_cuda_campaign_report_cli.R
fastkpc/tests/test_hsic_cuda_docs_contract.R
fastkpc/tests/test_hsic_cuda_benchmark.R
```

Modify:

```text
fastkpc/src/hsic_cpu.hpp
fastkpc/src/hsic_cpu.cpp
fastkpc/src/ci_method.hpp
fastkpc/src/ci_method.cpp
fastkpc/src/fastkpc_types.hpp
fastkpc/src/orientation_types.hpp
fastkpc/src/skeleton_engine_cuda.cpp
fastkpc/src/regrvonps_device.cpp
fastkpc/src/r_api_cuda.cpp
fastkpc/R/cuda_native.R
fastkpc/R/fast_kpc.R
fastkpc/R/validation_campaign.R
fastkpc/R/report_writer.R
fastkpc/tools/build_cuda_native.sh
fastkpc/tools/run_fast_kpc.R
fastkpc/tools/run_validation_campaign.R
fastkpc/README.md
fastkpc/reports/README.md
```

Do not modify:

```text
kpcalg/R/dcovgamma.R
kpcalg/R/hsicgamma.R
kpcalg/R/hsicperm.R
kpcalg/R/hsictest.R
kpcalg/R/kernelCItest.R
kpcalg/R/kpc.R
```

## Implementation Phases

## Phase 0: Baseline Audit

Purpose: prove the current tree starts from the CPU-resolution baseline and that legacy files are unchanged.

- [x] Run:

```bash
test -f fastkpc/src/hsic_cpu.cpp
test -f fastkpc/src/ci_method.cpp
test -f fastkpc/tests/test_hsic_skeleton_cuda_resolution.R
cd kpcalg && md5sum -c MD5 | rg '^R/'
```

Expected:

```text
All test commands exit 0.
All kpcalg/R files report OK.
```

- [x] Run current HSIC CPU-resolution tests:

```bash
Rscript fastkpc/tests/test_hsic_native_gamma.R
Rscript fastkpc/tests/test_hsic_native_permutation.R
Rscript fastkpc/tests/test_hsic_skeleton_cuda_resolution.R
Rscript fastkpc/tests/test_hsic_wanpdag_pipeline.R
Rscript fastkpc/tests/test_fastkpc_ci_method_public_api.R
```

Expected:

```text
Every test prints PASS.
Historical baseline before this goal: CUDA HSIC tests reported native-cpu.
Current expected state after the first implementation slice: supported CUDA HSIC
skeleton tests report cuda-hsic; unsupported paths report native-cpu with reason.
```

## Phase 1: TDD Red Tests For Raw CUDA HSIC Kernels

Purpose: define GPU kernel API before implementation.

- [x] Create `fastkpc/tests/test_hsic_cuda_kernel_math.R`.

Required test content:

```r
source("fastkpc/R/native.R")
source("fastkpc/R/cuda_native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

set.seed(301)
n <- 64
x <- seq(-2, 2, length.out = n)
y <- sin(x) + rnorm(n, sd = 0.05)

cpu <- fast_hsic_gamma_cpp(x, y, sig = 1)
load_fastkpc_cuda_native(rebuild = TRUE)
gpu <- fast_hsic_gamma_cuda(x, y, sig = 1)

assert_true(gpu$backend == "cuda-hsic", "GPU HSIC gamma should report cuda-hsic")
assert_true(is.finite(gpu$statistic), "GPU HSIC statistic should be finite")
assert_true(abs(gpu$statistic - cpu$statistic) < 1e-8,
            "GPU HSIC statistic should match CPU dense HSIC")
assert_true(abs(gpu$p.value - cpu$p.value) < 1e-7,
            "GPU HSIC p-value should match CPU dense HSIC")
assert_true(gpu$diagnostics$n == n, "GPU HSIC diagnostics should record n")
assert_true(gpu$diagnostics$kernel == "rbf",
            "GPU HSIC diagnostics should record kernel")
assert_true(gpu$diagnostics$bytes_allocated > 0,
            "GPU HSIC diagnostics should record memory")

cat("test_hsic_cuda_kernel_math.R: PASS\n")
```

- [x] Run:

```bash
Rscript fastkpc/tests/test_hsic_cuda_kernel_math.R
```

Expected:

```text
FAIL because fast_hsic_gamma_cuda does not exist.
```

## Phase 2: TDD Red Tests For CUDA HSIC Permutation

Purpose: define deterministic GPU permutation behavior.

- [x] Create `fastkpc/tests/test_hsic_cuda_permutation.R`.

Required test content:

```r
source("fastkpc/R/native.R")
source("fastkpc/R/cuda_native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

set.seed(302)
n <- 56
x <- seq(-1.5, 1.5, length.out = n)
y <- x^2 + rnorm(n, sd = 0.03)

load_fastkpc_cuda_native(rebuild = TRUE)
a <- fast_hsic_perm_cuda(x, y, sig = 1, replicates = 40L, seed = 500L,
                         include_observed = TRUE)
b <- fast_hsic_perm_cuda(x, y, sig = 1, replicates = 40L, seed = 500L,
                         include_observed = TRUE)
c <- fast_hsic_perm_cuda(x, y, sig = 1, replicates = 40L, seed = 501L,
                         include_observed = TRUE)

assert_true(a$backend == "cuda-hsic", "GPU HSIC permutation should report cuda-hsic")
assert_true(is.finite(a$p.value), "GPU HSIC permutation p-value should be finite")
assert_true(a$p.value >= 0 && a$p.value <= 1,
            "GPU HSIC permutation p-value should be in [0, 1]")
assert_true(length(a$replicates) == 40L,
            "GPU HSIC permutation should return requested replicates")
assert_true(max(abs(as.numeric(a$replicates) - as.numeric(b$replicates))) < 1e-12,
            "GPU HSIC permutation fixed seed should repeat within tolerance")
assert_true(a$p.value == b$p.value,
            "GPU HSIC permutation fixed seed p-value should repeat exactly")
assert_true(!identical(a$replicates, c$replicates),
            "GPU HSIC permutation different seed should change replicate order")
assert_true(a$diagnostics$seed == 500L,
            "GPU HSIC permutation diagnostics should record seed")

cat("test_hsic_cuda_permutation.R: PASS\n")
```

- [x] Run:

```bash
Rscript fastkpc/tests/test_hsic_cuda_permutation.R
```

Expected:

```text
FAIL because fast_hsic_perm_cuda does not exist.
```

## Phase 3: CUDA HSIC Batch Types

Purpose: introduce a narrow C++/CUDA contract before writing kernels.

- [x] Create `fastkpc/src/hsic_batch_types.hpp`.

Required structs:

```cpp
#ifndef FASTKPC_HSIC_BATCH_TYPES_HPP
#define FASTKPC_HSIC_BATCH_TYPES_HPP

#include <string>
#include <vector>

struct HsicBatchOptions {
  double sig;
  int permutation_replicates;
  bool include_observed;
  bool has_seed;
  unsigned int seed;
  bool return_replicates;
  int max_n;
  int max_batch_pairs;
};

struct HsicBatchDiagnostics {
  std::string backend;
  std::string reason;
  int n;
  int pairs;
  int batches;
  int permutation_replicates;
  bool used_seed;
  unsigned int seed;
  std::size_t bytes_allocated;
  int cuda_blocks;
  int cuda_threads;
};

struct HsicBatchResult {
  std::vector<double> statistics;
  std::vector<double> p_values;
  std::vector<double> means;
  std::vector<double> variances;
  std::vector<double> shapes;
  std::vector<double> scales;
  std::vector<double> permutation_replicates;
  HsicBatchDiagnostics diagnostics;
};

HsicBatchOptions default_hsic_batch_options();

#endif
```

- [x] Modify `fastkpc/src/hsic_cpu.hpp` only if needed to share result conversion names. Do not move CPU math into CUDA files.

- [x] Run a compile-only check:

```bash
Rscript -e 'source("fastkpc/R/native.R"); build_fastkpc_native(rebuild=TRUE)'
```

Expected:

```text
Build succeeds.
```

## Phase 4: CUDA HSIC Gamma Kernels

Purpose: implement dense GPU HSIC gamma for one or more vector pairs.

- [x] Create `fastkpc/src/cuda/hsic_batch_cuda.hpp`.

Required exported C++ function signatures:

```cpp
#ifndef FASTKPC_CUDA_HSIC_BATCH_CUDA_HPP
#define FASTKPC_CUDA_HSIC_BATCH_CUDA_HPP

#include "hsic_batch_types.hpp"

HsicBatchResult hsic_gamma_batch_cuda(const double* x,
                                      const double* y,
                                      int n,
                                      int pairs,
                                      const HsicBatchOptions& options);

HsicBatchResult hsic_permutation_batch_cuda(const double* x,
                                            const double* y,
                                            int n,
                                            int pairs,
                                            const HsicBatchOptions& options);

bool hsic_cuda_available(std::string* reason);

#endif
```

- [x] Create `fastkpc/src/cuda/hsic_batch_cuda.cu`.

Implementation requirements:

```text
1. Accept column-major pair batches matching dcov_batch_cuda layout:
   x[pair * n + row], y[pair * n + row].
2. Allocate K, L, Kc, Lc per batch or per pair according to memory budget.
3. Compute RBF Gram matrices on GPU.
4. Compute row sums and grand sums on GPU.
5. Center kernels on GPU.
6. Reduce sum(Kc * Lc), sum(Kc^2), sum(Lc^2), off-diagonal sums.
7. Copy compact scalar results back to host.
8. Compute gamma p-values on host using `R::pgamma` for consistency with CPU.
9. Record backend `cuda-hsic`.
10. If allocation or CUDA call fails, throw a C++ exception with a specific message.
```

Kernel names:

```cpp
hsic_rbf_kernel
hsic_center_kernel
hsic_reduce_gamma_scalars_kernel
```

Memory guard:

```text
If n > options.max_n, throw "CUDA HSIC n exceeds configured max_n".
If pairs > options.max_batch_pairs, caller must split before calling.
If estimated bytes exceed available memory minus safety margin, throw
"CUDA HSIC memory estimate exceeds available device memory".
```

- [x] Modify `fastkpc/tools/build_cuda_native.sh`.

Add:

```sh
"$CXX" $COMMON_CXX -c "$ROOT/src/ci_method.cpp" -o "$BUILD/ci_method.o"
"$CXX" $COMMON_CXX -c "$ROOT/src/hsic_cpu.cpp" -o "$BUILD/hsic_cpu.o"
"$NVCC" -O3 -arch=sm_89 -Xcompiler -fPIC -std=c++17 \
  $COMMON_INC -c "$ROOT/src/cuda/hsic_batch_cuda.cu" \
  -o "$BUILD/hsic_batch_cuda.o"
```

Link:

```text
hsic_batch_cuda.o
hsic_cpu.o
ci_method.o
```

- [x] Run:

```bash
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
```

Expected:

```text
Build succeeds and writes fastkpc/build/fastkpc_cuda.so.
```

## Phase 5: R CUDA Wrappers For Raw HSIC

Purpose: expose raw CUDA HSIC gamma/permutation functions for direct tests.

- [x] Modify `fastkpc/src/r_api_cuda.cpp`.

Add `.Call` wrappers:

```cpp
extern "C" SEXP C_fast_hsic_gamma_cuda(SEXP xs, SEXP ys, SEXP sigs);
extern "C" SEXP C_fast_hsic_perm_cuda(SEXP xs, SEXP ys, SEXP sigs,
                                       SEXP replicatess, SEXP seeds,
                                       SEXP include_observeds);
```

The result list must include:

```text
method
backend
statistic
estimate
estimates
p.value
replicates
diagnostics
```

Diagnostics must include:

```text
n
pairs
backend
kernel
sig
bytes_allocated
cuda_blocks
cuda_threads
replicates
used_seed
seed
reason
```

- [x] Modify `fastkpc/R/cuda_native.R`.

Add:

```r
fast_hsic_gamma_cuda <- function(x, y, sig = 1) {
  load_fastkpc_cuda_native()
  .Call("C_fast_hsic_gamma_cuda", as.numeric(x), as.numeric(y),
        as.numeric(sig), PACKAGE = "fastkpc_cuda")
}

fast_hsic_perm_cuda <- function(x, y, sig = 1, replicates = 100L,
                                seed, include_observed = TRUE) {
  if (missing(seed) || is.null(seed)) {
    stop("CUDA HSIC permutation requires explicit seed in this stage",
         call. = FALSE)
  }
  load_fastkpc_cuda_native()
  .Call("C_fast_hsic_perm_cuda", as.numeric(x), as.numeric(y),
        as.numeric(sig), as.integer(replicates), as.integer(seed),
        isTRUE(include_observed), PACKAGE = "fastkpc_cuda")
}
```

- [x] Run:

```bash
Rscript fastkpc/tests/test_hsic_cuda_kernel_math.R
Rscript fastkpc/tests/test_hsic_cuda_permutation.R
```

Expected:

```text
Both tests print PASS.
```

## Phase 6: CUDA HSIC Batch CI Evaluator

Purpose: add HSIC CUDA evaluator under the skeleton scheduler, analogous to dCov batching.

- [x] Modify `fastkpc/src/ci_method.hpp`.

Add:

```cpp
enum class CiBackendKind {
  NativeCpu,
  CudaDcov,
  CudaHsic
};

struct CiBackendResolution {
  CiBackendKind backend;
  std::string backend_name;
  std::string reason;
  bool can_batch_on_cuda;
};
```

- [x] Modify `fastkpc/src/skeleton_engine_cuda.cpp`.

Replace current HSIC CPU-resolution branch:

```cpp
if (kind != CiMethodKind::DccGamma) {
  SkeletonResult result = run_skeleton_exact(...);
  result.ci_backend = "native-cpu";
  result.ci_backend_reason = "... not implemented ...";
  return result;
}
```

with real resolution:

```text
1. Resolve ci_method and permutation seed.
2. If hsic.gamma and CUDA HSIC available, use CUDA HSIC task batching.
3. If hsic.perm and explicit seed and CUDA HSIC available, use CUDA HSIC permutation batching.
4. Otherwise run CPU fallback and record reason.
```

- [x] Add `evaluate_tasks_hsic_cuda`.

Required behavior:

```text
Input: LayerCiTask vector, batch_size, HsicBatchOptions, residual cache
Output: p-values vector
For each batch:
  materialize x/y vectors using existing residual cache
  call hsic_gamma_batch_cuda or hsic_permutation_batch_cuda
  append scheduler batch diagnostic kind = "hsic"
  update ci diagnostics
```

- [x] Ensure replay semantics do not change:

```text
The layer plan may evaluate speculative tasks.
Only replayed tests update adjacency, sepsets, pMax, and n.edgetests.
```

## Phase 7: CUDA Skeleton HSIC Tests

Purpose: prove CUDA HSIC is actually used in skeleton runs.

- [x] Create `fastkpc/tests/test_hsic_cuda_skeleton_backend.R`.

Required assertions:

```text
1. fast_skeleton_cuda_backend(..., ci_method="hsic.gamma") returns ci_backend == "cuda-hsic".
2. ci_backend_reason is empty.
3. ci_hsic_gamma_cuda_tests > 0.
4. ci_hsic_cuda_batches > 0.
5. CPU HSIC gamma adjacency equals CUDA HSIC gamma adjacency on small deterministic fixture.
6. CPU HSIC gamma pMax is close to CUDA HSIC gamma pMax within 1e-7.
7. dcc.gamma CUDA skeleton behavior remains ci_backend == "cuda" or "cuda-dcov".
```

- [x] Create `fastkpc/tests/test_hsic_cuda_scheduler_diagnostics.R`.

Required assertions:

```text
1. scheduler_diagnostics$batches contains kind == "hsic" rows.
2. scheduler_diagnostics$summary$dcov_batches remains 0 for pure HSIC runs.
3. ci_diagnostics$ci_hsic_cuda_pairs equals or exceeds replayed tests.
4. layer and legacy scheduler replay outputs match for hsic.gamma.
5. residual cache hits still occur for conditional HSIC tests with residual_cache=TRUE.
```

- [x] Run:

```bash
Rscript fastkpc/tests/test_hsic_cuda_skeleton_backend.R
Rscript fastkpc/tests/test_hsic_cuda_scheduler_diagnostics.R
Rscript fastkpc/tests/test_hsic_skeleton_cpu.R
Rscript fastkpc/tests/test_hsic_skeleton_cuda_resolution.R
```

Expected:

```text
New tests print PASS.
Existing CPU-resolution test must be updated to expect cuda-hsic when supported and native-cpu fallback only for unsupported cases.
```

Current implementation evidence:

```text
Rscript fastkpc/tests/test_hsic_cuda_skeleton_backend.R -> PASS
Rscript fastkpc/tests/test_hsic_cuda_scheduler_diagnostics.R -> PASS
```

Remaining verification in this phase:

```text
Rscript fastkpc/tests/test_hsic_skeleton_cpu.R
Rscript fastkpc/tests/test_hsic_skeleton_cuda_resolution.R
```

## Phase 8: CUDA HSIC Permutation Scheduler Integration

Purpose: make `ci_method="hsic.perm"` use GPU when explicit seed is supplied.

- [x] Modify skeleton CUDA resolution rules:

```text
If permutation_params$seed is NULL:
  fallback to native-cpu
  reason = "CUDA HSIC permutation requires explicit seed in this stage"
If seed is present:
  use cuda-hsic
  use deterministic GPU permutation generator
```

- [x] Implement deterministic GPU permutation generation.

Allowed approach:

```text
Use a simple documented counter-based deterministic generator or host-generated
permutation table copied to GPU. For this goal, host-generated deterministic
permutations are acceptable if the HSIC replicate statistics are computed on GPU.
```

Preferred first implementation:

```text
Host generates integer permutation table using std::mt19937(seed + pair/replicate offset).
Copy permutation table to GPU.
GPU computes replicate HSIC statistics.
```

Reason:

```text
This avoids introducing cuRAND and keeps reproducibility auditable.
```

- [x] Create `fastkpc/tests/test_hsic_cuda_skeleton_permutation.R`.

Add assertions:

```text
1. repeated fast_skeleton_cuda_backend(..., ci_method="hsic.perm", seed fixed) pMax matches exactly.
2. fixed-seed CUDA HSIC permutation records ci_backend == "cuda-hsic".
3. fixed-seed CUDA HSIC permutation records positive CUDA HSIC test and batch counters.
4. missing seed records native-cpu fallback reason in skeleton CUDA API.
5. missing seed records positive CUDA HSIC fallback test counters.
```

- [x] Run:

```bash
Rscript fastkpc/tests/test_hsic_cuda_permutation.R
Rscript fastkpc/tests/test_hsic_cuda_skeleton_backend.R
Rscript fastkpc/tests/test_hsic_cuda_skeleton_permutation.R
```

Expected:

```text
All tests print PASS.
```

Current implementation evidence:

```text
Rscript fastkpc/tests/test_hsic_cuda_skeleton_permutation.R -> PASS
Rscript fastkpc/tests/test_hsic_cuda_permutation.R -> PASS in raw CUDA kernel phase
Rscript fastkpc/tests/test_hsic_cuda_skeleton_backend.R -> PASS
```

## Phase 9: WAN-PDAG Orientation CUDA HSIC

Purpose: make generalized orientation residual-vs-S tests use CUDA HSIC when requested and supported.

- [x] Modify `fastkpc/src/regrvonps_device.cpp`.

Current behavior should be:

```text
HSIC method -> regrvonps_native CPU evaluator
```

Change to:

```text
HSIC method + orientation_residual_device CUDA + supported seed/memory -> CUDA residual + CUDA HSIC
HSIC method + orientation_residual_device CPU but engine CUDA -> CPU residual + CUDA HSIC if vectors are materialized on host and copied to GPU
HSIC method unsupported -> native CPU fallback with reason
```

- [x] Add orientation diagnostics:

```text
orientation$ci_backend
orientation$ci_backend_reason
orientation$ci_diagnostics$regrvonps_hsic_gamma_cuda_tests
orientation$ci_diagnostics$regrvonps_hsic_perm_cuda_tests
orientation$ci_diagnostics$regrvonps_hsic_cuda_batches
orientation$ci_diagnostics$regrvonps_hsic_cuda_pairs
orientation$ci_diagnostics$regrvonps_hsic_cuda_fallback_tests
```

- [x] Create `fastkpc/tests/test_hsic_cuda_wanpdag_orientation.R`.

Required assertions:

```text
1. fast_kpc_wanpdag_cuda(..., ci_method="hsic.gamma") skeleton ci_backend == "cuda-hsic".
2. orientation ci_method == "hsic.gamma".
3. orientation ci_backend is "cuda-hsic" when generalized orientation tests are executed on CUDA.
4. orientation diagnostics record HSIC CUDA test counts when fixture exercises regrVonPS.
5. CPU and CUDA HSIC WAN-PDAG pdag match on small deterministic fixture within pMax tolerance.
6. hsic.perm fixed-seed WAN-PDAG CUDA repeated pdag matches exactly.
```

- [x] Run:

```bash
Rscript fastkpc/tests/test_hsic_cuda_wanpdag_orientation.R
Rscript fastkpc/tests/test_wanpdag_cuda_orientation_device.R
Rscript fastkpc/tests/test_regrvonps_cuda_orientation_device.R
```

Expected:

```text
Every test prints PASS.
```

Current implementation evidence:

```text
Rscript fastkpc/tests/test_hsic_cuda_wanpdag_orientation.R -> PASS
Rscript fastkpc/tests/test_wanpdag_cuda_orientation_device.R -> PASS
Rscript fastkpc/tests/test_regrvonps_cuda_orientation_device.R -> PASS after serial rerun
```

## Phase 10: Public API Diagnostics

Purpose: ensure users can tell whether HSIC ran on GPU.

- [x] Modify `fastkpc/R/fast_kpc.R`.

Required config fields:

```text
config$ci_method
config$ci_method_requested
config$ci_backend
config$ci_backend_requested
config$ci_backend_reason
config$cuda_hsic_requested
config$cuda_hsic_used
config$hsic_params
config$permutation_params
diagnostics$cuda_hsic_available
diagnostics$cuda_hsic_reason
```

- [x] Modify `print.fastkpc_result`.

Required output includes:

```text
ci_method: hsic.gamma
ci_backend: cuda-hsic
```

- [x] Create `fastkpc/tests/test_hsic_cuda_public_api.R`.

Required assertions:

```text
1. fast_kpc(..., engine="cuda", ci_method="hsic.gamma") config$ci_backend == "cuda-hsic".
2. skeleton$ci_backend == "cuda-hsic".
3. diagnostics$cuda_hsic_available is TRUE when kernels are available.
4. print output contains ci_method and ci_backend.
5. missing seed for hsic.perm records native-cpu fallback reason.
6. default fast_kpc() still records ci_method == "dcc.gamma".
```

- [x] Run:

```bash
Rscript fastkpc/tests/test_hsic_cuda_public_api.R
Rscript fastkpc/tests/test_fastkpc_ci_method_public_api.R
Rscript fastkpc/tests/test_fast_kpc_public_api.R
Rscript fastkpc/tests/test_fastkpc_result_contract.R
```

Expected:

```text
Every test prints PASS.
```

Current implementation evidence:

```text
Rscript fastkpc/tests/test_hsic_cuda_public_api.R -> PASS
Rscript fastkpc/tests/test_fastkpc_ci_method_public_api.R -> PASS
Rscript fastkpc/tests/test_fast_kpc_public_api.R -> PASS
Rscript fastkpc/tests/test_fastkpc_result_contract.R -> PASS
```

## Phase 11: Validation Campaign, Report, And CLI

Purpose: expose CUDA HSIC backend status in validation artifacts.

- [x] Modify `fastkpc/R/validation_campaign.R`.

Add columns:

```text
ci_backend
ci_backend_requested
ci_backend_reason
cuda_hsic_requested
cuda_hsic_used
ci_hsic_gamma_cuda_tests
ci_hsic_perm_cuda_tests
ci_hsic_cuda_batches
ci_hsic_cuda_pairs
ci_hsic_cuda_fallback_tests
```

Add artifacts:

```text
hsic_cuda_backend_diagnostics.csv
hsic_cuda_cpu_fallbacks.csv
hsic_cuda_perf.csv
```

- [x] Modify `fastkpc/R/report_writer.R`.

Summary must include sections:

```markdown
## HSIC CUDA Backend
## HSIC CUDA Fallbacks
## HSIC CUDA Performance
```

- [x] Modify `fastkpc/tools/run_fast_kpc.R`.

Add CLI options:

```text
--ci-backend auto,cpu,cuda
--hsic-cuda-max-n
--hsic-cuda-max-batch-pairs
--hsic-cuda-memory-fallback TRUE/FALSE
```

Output must print:

```text
ci_method=<value>
ci_backend=<value>
cuda_hsic_used=<TRUE/FALSE>
ci_hsic_cuda_batches=<value>
ci_hsic_cuda_pairs=<value>
```

- [x] Modify `fastkpc/tools/run_validation_campaign.R`.

Add:

```text
--ci-backend
--hsic-cuda-max-n
--hsic-cuda-max-batch-pairs
--hsic-cuda-memory-fallback
```

- [x] Create `fastkpc/tests/test_hsic_cuda_campaign_report_cli.R`.

Required assertions:

```text
1. campaign accepts ci_methods=c("hsic.gamma") and engines=c("cuda").
2. campaign$ci_method_diagnostics records ci_backend == "cuda-hsic".
3. campaign has hsic_cuda_backend_diagnostics.
4. report writes hsic_cuda_backend_diagnostics.csv.
5. summary markdown contains "HSIC CUDA Backend".
6. run_fast_kpc.R prints cuda_hsic_used=TRUE.
7. run_validation_campaign.R writes hsic_cuda_backend_diagnostics.csv.
```

- [x] Run:

```bash
Rscript fastkpc/tests/test_hsic_cuda_campaign_report_cli.R
```

Expected:

```text
Test prints PASS.
```

Current implementation evidence:

```text
Rscript fastkpc/tests/test_hsic_cuda_campaign_report_cli.R -> PASS
Rscript fastkpc/tests/test_ci_method_campaign_report_cli.R -> PASS
```

## Phase 12: Performance Benchmarks

Purpose: prove CUDA HSIC is materially useful without setting brittle speedup gates for all hardware.

- [x] Create or extend `fastkpc/R/hsic_cuda_validation.R`.

Required functions:

```r
validate_hsic_cuda_gamma_kernel <- function(seed = 401, n = 128, sig = 1)
validate_hsic_cuda_permutation_kernel <- function(seed = 402, n = 96,
                                                  replicates = 50)
compare_hsic_cuda_cpu_skeleton <- function(seed = 403, n = 128)
benchmark_hsic_cuda_backends <- function(seed = 404,
                                         n_values = c(64, 128, 256),
                                         methods = c("hsic.gamma", "hsic.perm"),
                                         repeats = 3)
```

Metrics:

```text
statistic_abs_diff
pvalue_abs_diff
adjacency_identical
max_abs_pmax_diff
cpu_elapsed_sec
cuda_elapsed_sec
speedup
ci_backend
cuda_batches
cuda_pairs
bytes_allocated
```

- [x] Create `fastkpc/tests/test_hsic_cuda_benchmark.R`.

Required assertions:

```text
1. validate_hsic_cuda_gamma_kernel() statistic_abs_diff < 1e-8.
2. validate_hsic_cuda_permutation_kernel() fixed seed repeats.
3. compare_hsic_cuda_cpu_skeleton() adjacency_identical TRUE.
4. benchmark_hsic_cuda_backends() returns rows for cpu and cuda.
5. benchmark_hsic_cuda_backends() includes speedup column.
6. No hard speedup threshold is enforced in tests.
```

- [x] Run:

```bash
Rscript fastkpc/tests/test_hsic_cuda_benchmark.R
```

Expected:

```text
Test prints PASS.
```

Current implementation evidence:

```text
Rscript fastkpc/tests/test_hsic_cuda_benchmark.R -> PASS
Rscript fastkpc/tests/test_hsic_benchmark.R -> PASS
```

## Phase 13: Memory And Fallback Tests

Purpose: avoid silent GPU OOM or fake CUDA reporting.

- [x] Add configurable limits:

```text
hsic_params$cuda_max_n
hsic_params$cuda_max_batch_pairs
hsic_params$cuda_memory_fallback
```

Defaults:

```text
cuda_max_n = 2048
cuda_max_batch_pairs = 64
cuda_memory_fallback = TRUE
```

- [x] Create fallback tests in `fastkpc/tests/test_hsic_cuda_skeleton_backend.R`.

Required assertions:

```text
1. hsic_params=list(cuda_max_n=10) with n=64 falls back to native-cpu.
2. fallback reason contains "n exceeds configured max_n".
3. ci_hsic_cuda_fallback_tests > 0.
4. cuda_memory_fallback=FALSE raises an error instead of falling back.
5. successful CUDA runs do not record fallback reason.
```

- [x] Run:

```bash
Rscript fastkpc/tests/test_hsic_cuda_skeleton_backend.R
```

Expected:

```text
Test prints PASS.
```

Current implementation evidence:

```text
bash fastkpc/tools/clean_cuda_native.sh && bash fastkpc/tools/build_cuda_native.sh -> PASS
Rscript fastkpc/tests/test_hsic_cuda_skeleton_backend.R -> PASS
Rscript fastkpc/tests/test_hsic_cuda_public_api.R -> PASS
Rscript fastkpc/tests/test_hsic_cuda_wanpdag_orientation.R -> PASS
```

## Phase 14: Documentation

Purpose: document exact GPU/CPU semantics and the fallback boundary.

- [x] Modify `fastkpc/README.md`.

Required text:

```text
CUDA HSIC kernels
cuda-hsic
native-cpu fallback
hsic.gamma
hsic.perm
explicit permutation seed
CUDA HSIC permutation requires explicit seed
ci_backend
cuda_hsic_used
hsic_cuda_backend_diagnostics.csv
hsic_cuda_cpu_fallbacks.csv
kpcalg/R/*.R files are not modified
```

- [x] Modify `fastkpc/reports/README.md`.

Add artifacts:

```text
hsic_cuda_backend_diagnostics.csv
hsic_cuda_cpu_fallbacks.csv
hsic_cuda_perf.csv
```

- [x] Create `fastkpc/tests/test_hsic_cuda_docs_contract.R`.

Required assertions:

```text
README mentions CUDA HSIC kernels.
README mentions cuda-hsic.
README mentions native-cpu fallback.
README mentions explicit permutation seed.
README mentions CUDA HSIC permutation requires explicit seed.
reports README mentions hsic_cuda_backend_diagnostics.csv.
reports README mentions hsic_cuda_cpu_fallbacks.csv.
```

- [x] Run:

```bash
Rscript fastkpc/tests/test_hsic_cuda_docs_contract.R
Rscript fastkpc/tests/test_hsic_docs_contract.R
Rscript fastkpc/tests/test_fastkpc_docs_contract.R
```

Expected:

```text
Every test prints PASS.
```

Current implementation evidence:

```text
Rscript fastkpc/tests/test_hsic_cuda_docs_contract.R -> PASS
Rscript fastkpc/tests/test_hsic_docs_contract.R -> PASS
Rscript fastkpc/tests/test_fastkpc_docs_contract.R -> PASS
```

## Phase 15: Full Verification Campaign

Purpose: prove CUDA HSIC kernels are complete and do not regress existing dCov/CUDA behavior.

- [x] Run clean builds:

```bash
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
Rscript -e 'source("fastkpc/R/native.R"); build_fastkpc_native(rebuild=TRUE)'
```

Expected:

```text
CUDA native build succeeds.
CPU native build succeeds.
```

- [x] Run new raw CUDA HSIC tests:

```bash
Rscript fastkpc/tests/test_hsic_cuda_kernel_math.R
Rscript fastkpc/tests/test_hsic_cuda_permutation.R
```

Expected:

```text
Both tests print PASS.
```

- [x] Run new CUDA HSIC integration tests:

```bash
Rscript fastkpc/tests/test_hsic_cuda_skeleton_backend.R
Rscript fastkpc/tests/test_hsic_cuda_scheduler_diagnostics.R
Rscript fastkpc/tests/test_hsic_cuda_wanpdag_orientation.R
Rscript fastkpc/tests/test_hsic_cuda_public_api.R
Rscript fastkpc/tests/test_hsic_cuda_campaign_report_cli.R
Rscript fastkpc/tests/test_hsic_cuda_docs_contract.R
Rscript fastkpc/tests/test_hsic_cuda_benchmark.R
```

Expected:

```text
Every test prints PASS.
```

- [x] Run existing HSIC CPU tests:

```bash
Rscript fastkpc/tests/test_hsic_native_gamma.R
Rscript fastkpc/tests/test_hsic_native_permutation.R
Rscript fastkpc/tests/test_hsic_skeleton_cpu.R
Rscript fastkpc/tests/test_hsic_skeleton_cuda_resolution.R
Rscript fastkpc/tests/test_hsic_wanpdag_pipeline.R
Rscript fastkpc/tests/test_fastkpc_ci_method_public_api.R
Rscript fastkpc/tests/test_ci_method_campaign_report_cli.R
Rscript fastkpc/tests/test_hsic_docs_contract.R
Rscript fastkpc/tests/test_hsic_benchmark.R
Rscript fastkpc/tests/test_fastkpc_reproducibility.R
```

Expected:

```text
Every test prints PASS.
```

- [x] Run existing CUDA dCov/residual/scheduler tests:

```bash
Rscript fastkpc/tests/test_dcov_cuda_batch.R
Rscript fastkpc/tests/test_skeleton_cuda_batch.R
Rscript fastkpc/tests/test_cuda_residual_device_skeleton.R
Rscript fastkpc/tests/test_cuda_fastspline_true_batch_contract.R
Rscript fastkpc/tests/test_cuda_fastspline_batch_grouping.R
Rscript fastkpc/tests/test_cuda_layer_scheduler_true_residual_batch.R
Rscript fastkpc/tests/test_cuda_layer_scheduler_residual_prefetch.R
Rscript fastkpc/tests/test_cuda_layer_scheduler_equivalence.R
Rscript fastkpc/tests/test_true_batched_fastspline_campaign_report_cli.R
```

Expected:

```text
Every test prints PASS.
```

- [x] Run existing public/WAN-PDAG tests:

```bash
Rscript fastkpc/tests/test_fast_kpc_public_api.R
Rscript fastkpc/tests/test_fastkpc_result_contract.R
Rscript fastkpc/tests/test_full_framework_smoke.R
Rscript fastkpc/tests/test_wanpdag_cpu_pipeline.R
Rscript fastkpc/tests/test_wanpdag_cuda_pipeline.R
Rscript fastkpc/tests/test_wanpdag_cuda_orientation_device.R
Rscript fastkpc/tests/test_regrvonps_cuda_orientation_device.R
Rscript fastkpc/tests/test_wanpdag_orientation_device_validation.R
```

Expected:

```text
Every test prints PASS.
```

- [x] Run compact CUDA HSIC campaign:

```bash
rm -rf fastkpc/reports/hsic_cuda_smoke
Rscript fastkpc/tools/run_validation_campaign.R \
  --engines cuda \
  --residual-backends fastSpline \
  --residual-devices cuda \
  --orientation-residual-devices cuda \
  --ci-methods hsic.gamma,hsic.perm \
  --permutation-seed 123 \
  --permutation-replicates 40 \
  --schedulers layer \
  --seeds 11 \
  --n-values 96 \
  --scenarios chain,additive \
  --legacy FALSE \
  --output-dir fastkpc/reports/hsic_cuda_smoke
Rscript -e 'x <- read.csv("fastkpc/reports/hsic_cuda_smoke/ci_method_diagnostics.csv"); print(x[, c("scenario", "ci_method", "ci_backend", "ci_tests")]); stopifnot(all(x$ci_backend == "cuda-hsic"), all(x$ci_tests > 0))'
```

Expected:

```text
Report command succeeds.
ci_method_diagnostics.csv contains cuda-hsic rows.
summary.md contains HSIC CUDA Backend section.
```

- [x] Run `kpcalg/R` MD5 audit:

```bash
cd kpcalg && md5sum -c MD5 | rg '^R/'
```

Expected:

```text
Every R/*.R file reports OK.
```

## Completion Criteria

The goal is complete only when all criteria are proven by fresh verification:

1. `fastkpc/src/cuda/hsic_batch_cuda.cu` exists and implements real GPU HSIC gamma scalar computation.
2. `fast_hsic_gamma_cuda()` returns `backend == "cuda-hsic"` on supported CUDA hardware.
3. GPU HSIC gamma statistic matches CPU dense HSIC within the required tolerance.
4. GPU HSIC gamma p-value matches CPU dense HSIC within the required tolerance.
5. `fast_hsic_perm_cuda()` exists and supports explicit fixed seeds.
6. GPU HSIC permutation repeated fixed-seed calls are deterministic.
7. Missing permutation seed does not silently use CUDA; it falls back or errors with explicit reason.
8. CUDA skeleton `ci_method="hsic.gamma"` uses `ci_backend == "cuda-hsic"` when supported.
9. CUDA skeleton `ci_method="hsic.perm"` uses `ci_backend == "cuda-hsic"` when explicit seed and limits are supported.
10. CUDA HSIC skeleton replay preserves stable graph semantics.
11. Scheduler diagnostics include HSIC batch rows and HSIC CUDA counts.
12. WAN-PDAG orientation can use CUDA HSIC for generalized residual-vs-S tests when supported.
13. Public `fast_kpc()` exposes `ci_backend`, `cuda_hsic_used`, and fallback reasons.
14. Validation campaign records CUDA HSIC backend diagnostics.
15. Reports write HSIC CUDA artifacts.
16. CLI supports CUDA HSIC controls and prints backend usage.
17. Documentation explains CUDA HSIC, CPU fallback, explicit seed requirement, and unchanged `kpcalg/R`.
18. Existing dCov CUDA tests still pass.
19. Existing CPU and public wrapper tests still pass.
20. `kpcalg/R` MD5 audit passes.

## Known Risks And Mitigations

### Risk: Dense Gram matrices use too much GPU memory

Mitigation:

```text
Use max_n and max_batch_pairs guards.
Estimate memory before allocation.
Fallback to CPU only when cuda_memory_fallback=TRUE.
Record bytes_allocated and fallback reason.
```

### Risk: GPU permutation reproducibility differs from CPU

Mitigation:

```text
Do not require CPU/GPU replicate order equality.
Require GPU/GPU fixed-seed deterministic equality.
Use CPU dense HSIC for observed statistic correctness.
Document RNG algorithm in source comments.
```

### Risk: Scheduler evaluates speculative HSIC tasks

Mitigation:

```text
Reuse existing layer replay model.
Track evaluated pairs separately from replayed tests.
Never update graph state before replay.
```

### Risk: Users misunderstand `engine="cuda"` fallback

Mitigation:

```text
Expose ci_backend and ci_backend_reason at skeleton, orientation, config, diagnostics,
CLI, report, and README levels.
```

### Risk: Kernel math differs from CPU centering

Mitigation:

```text
Test raw kernel math directly at n=16, 64, 128, 256.
Keep CPU dense HSIC as oracle.
Use double precision.
Compute gamma p-value on host using the same R math routine.
```

## Execution Notes

- Work in place unless an isolated worktree already exists. This workspace may not be a git repository.
- Use TDD: write failing tests first, confirm failure, implement minimal code, rerun.
- Keep native CPU HSIC code readable and independent; do not hide GPU fallback inside CPU functions.
- Prefer adding focused CUDA helper files over expanding `skeleton_engine_cuda.cpp` with kernel math.
- Keep generated build artifacts out of plan reasoning. `fastkpc/build/*.o`, `fastkpc/build/*.so`, and `fastkpc/src/*.o` may appear during local builds.
- Before final completion, re-run all commands in Phase 15 and report exact outcomes.
