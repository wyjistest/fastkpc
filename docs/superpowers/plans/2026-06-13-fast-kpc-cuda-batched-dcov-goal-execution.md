# Fast kPC CUDA Batched dCov Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in CUDA batched exact distance-covariance backend to the staged `fastkpc` C++ skeleton MVP, while preserving the CPU exact backend as the reference.

**Architecture:** Keep R as the facade and validation harness. Keep the stable skeleton scheduler in C++, but add a per-level speculative batch execution path that evaluates dCov tasks on CUDA and replays results in the original deterministic order. Keep residual regression on the current CPU linear MVP path for this goal; true residual caching and GPU spline residuals are later goals.

**Tech Stack:** R 4.4.1, Rcpp/RcppArmadillo, C++17, CUDA 12.5 via `/usr/local/cuda/bin/nvcc`, RTX 4090 `sm_89`, existing `gpu-dcov/` prototype as numeric reference.

---

## Goal Objective For Codex

Use this exact goal objective when starting a Codex goal:

```text
Build the next staged fast kPC backend slice from docs/superpowers/plans/2026-06-13-fast-kpc-cuda-batched-dcov-goal-execution.md: add an opt-in CUDA batched exact dCov backend and a CUDA-backed skeleton path, preserving CPU exact behavior as the reference and keeping legacy kpcalg/R files unchanged.
```

Recommended token budget for the goal run: `160000`.

Do not mark the goal complete until all completion criteria in Phase 6 are satisfied. Mark the goal blocked only if the same blocker prevents progress for three consecutive goal turns and cannot be resolved locally.

## Preconditions

The previous first-slice goal must already be complete.

Required current artifacts:

```text
fastkpc/R/dcov_exact.R
fastkpc/R/native.R
fastkpc/R/diff_report.R
fastkpc/R/legacy_runner.R
fastkpc/src/dcov_exact_cpu.cpp
fastkpc/src/skeleton_engine.cpp
fastkpc/src/rcpp_exports.cpp
fastkpc/tests/test_dcov_exact.R
fastkpc/tests/test_skeleton_mvp.R
fastkpc/tests/test_diff_report.R
gpu-dcov/validate.R
```

Required environment:

```text
R 4.4.1
Rcpp installed
RcppArmadillo installed
CUDA toolkit available at /usr/local/cuda/bin/nvcc
NVIDIA driver able to run CUDA kernels on RTX 4090 or compatible GPU
```

If CUDA is not available on this machine, stop and report the exact command that failed. Do not silently fall back to CPU and call the CUDA goal complete.

## Scope

In scope for this goal:

- Add a local CUDA native build path under `fastkpc/`.
- Add a CUDA batched exact dCov kernel that accepts many vector-pair tests in one native call.
- Add an R wrapper `fast_dcov_batch_cuda()` for direct batch dCov validation.
- Add an opt-in skeleton wrapper `fast_skeleton_cuda()` that uses CUDA batched dCov per conditioning level.
- Preserve CPU exact skeleton behavior as the reference for graph-level validation.
- Keep `kpcalg/R/*.R` unchanged.
- Document build, test, and behavioral limits in `fastkpc/README.md`.

Out of scope for this goal:

- Do not implement a CUDA GAM, fastSpline, or other residual backend.
- Do not implement residual caching.
- Do not migrate `udag2wanpdag()`.
- Do not replace exported `kpcalg::kpc()`.
- Do not support HSIC, permutation tests, or legacy `mgcv` residual equivalence.
- Do not require `pcalg`; it is not installed in the current environment.

## Design Contract

### CUDA dCov Batch API

The CUDA backend evaluates a batch of exact dCov gamma tests:

```text
Input:
  X: numeric matrix, n rows by b columns
  Y: numeric matrix, n rows by b columns
  index: scalar in [0, 2]
  legacy_index: boolean

Output:
  p.value: numeric vector length b
  nV2: numeric vector length b
  mean: numeric vector length b
  variance: numeric vector length b
  raw: b by 5 matrix with Sab, Saa, Sbb, sumK, sumL
```

Each column pair `X[, k]`, `Y[, k]` is one independence test.

Scaling must match the completed exact CPU backend:

```text
A = H K H
B = H L H
nV2 = sum(A * B) / n
nV2Mean = mean(K) * mean(L)
nV2Variance =
  2 * (n - 4) * (n - 5) / (n * (n - 1) * (n - 2) * (n - 3))
  * sum(A * A) * sum(B * B) / n^2
```

Index semantics:

```text
legacy_index = TRUE: ignore distance exponent and use raw distances, matching old kpcalg behavior.
legacy_index = FALSE: apply dist^index, matching documented dCov semantics.
```

Use FP64 for this goal. Mixed precision can be evaluated in a later performance goal after graph-level equivalence is stable.

### CUDA Skeleton Replay Semantics

The CUDA skeleton path must preserve the CPU exact MVP's observable behavior.

For each conditioning level:

1. Take the stable adjacency snapshot at the start of the level.
2. Enumerate candidate edge and conditioning-set tasks in the same order as the CPU skeleton.
3. Compute task p-values in CUDA batches.
4. Replay p-values sequentially in the original order.
5. For each edge, accept tests only until the first `p >= alpha` deletion event.
6. Ignore speculative p-values after an edge is already marked deleted for that level.
7. Count `n.edgetests` using only accepted replayed tests, not ignored speculative tests.
8. Apply deletions at the end of the level.

This allows aggressive GPU batching without changing adjacency, sepsets, pMax, or `n.edgetests`.

Conditional tests still use the existing CPU linear residualization MVP. For each conditional task, residual vectors are computed on CPU, packed into batch matrices, copied once for that CUDA batch, and evaluated on GPU. This is a staging step, not the final residual architecture.

## File Structure

Create these files:

- `fastkpc/R/cuda_native.R`  
  R wrappers for CUDA native build/load, `fast_dcov_batch_cuda()`, and `fast_skeleton_cuda()`.

- `fastkpc/R/cuda_validation.R`  
  Small reusable validation helpers for CPU-vs-CUDA dCov and graph comparisons.

- `fastkpc/src/cuda/dcov_batch_cuda.hpp`
- `fastkpc/src/cuda/dcov_batch_cuda.cu`  
  CUDA kernels and a C ABI function for batched exact dCov scalar reductions.

- `fastkpc/src/cuda/cuda_status.hpp`
- `fastkpc/src/cuda/cuda_status.cpp`  
  CUDA availability, device info, and error formatting helpers.

- `fastkpc/src/dcov_batch_types.hpp`  
  C++ structs for batch task metadata and batch dCov outputs.

- `fastkpc/src/skeleton_engine_cuda.hpp`
- `fastkpc/src/skeleton_engine_cuda.cpp`  
  Opt-in CUDA skeleton path using speculative per-level batching and deterministic replay.

- `fastkpc/src/r_api_cuda.cpp`  
  Manual `.Call` entry points for CUDA wrappers.

- `fastkpc/tools/build_cuda_native.sh`  
  Local CUDA build script producing `fastkpc/build/fastkpc_cuda.so`.

- `fastkpc/tools/clean_cuda_native.sh`  
  Removes CUDA build outputs.

- `fastkpc/tests/test_dcov_cuda_batch.R`
- `fastkpc/tests/test_skeleton_cuda_batch.R`
- `fastkpc/tests/test_cuda_build_contract.R`

Modify these files:

- `fastkpc/README.md`  
  Add CUDA build/test commands, scope notes, and CPU-vs-CUDA validation expectations.

Do not modify these files:

- `kpcalg/R/*.R`
- `gpu-dcov/*` except for reading as a numeric reference

## Phase 0: Baseline And CUDA Environment Audit

Purpose: prove the previous slice still works and CUDA can be used.

- [ ] Run:

```bash
pwd
find fastkpc -maxdepth 3 -type f | sort
find gpu-dcov -maxdepth 1 -type f | sort
find kpcalg/R -maxdepth 1 -type f | sort
```

Expected:

```text
Working directory is /data/wenyujianData/kpcalg.
fastkpc contains the completed first-slice files.
gpu-dcov contains dcov_gpu.cu, dcov_gamma_gpu.R, build.sh, and validate.R.
kpcalg/R contains the legacy R files.
```

- [ ] Run:

```bash
Rscript -e 'source("fastkpc/R/native.R"); build_fastkpc_native(rebuild=TRUE)'
Rscript fastkpc/tests/test_dcov_exact.R
Rscript fastkpc/tests/test_skeleton_mvp.R
Rscript fastkpc/tests/test_diff_report.R
Rscript gpu-dcov/validate.R
```

Expected:

```text
All first-slice tests pass.
gpu-dcov/validate.R ends with ALL CHECKS PASSED.
```

- [ ] Run:

```bash
/usr/local/cuda/bin/nvcc --version
nvidia-smi
```

Expected:

```text
nvcc is available.
nvidia-smi lists at least one CUDA-capable GPU.
```

If either command fails, stop and report the failure. Do not continue with CPU-only substitutions.

## Phase 1: CUDA Build Contract

Purpose: add a repeatable local build path for CUDA native code without disturbing the current CPU `sourceCpp()` flow.

- [ ] Create `fastkpc/tools/build_cuda_native.sh`.

Required behavior:

```text
Use /usr/local/cuda/bin/nvcc.
Compile CUDA sources with -O3 -arch=sm_89 -Xcompiler -fPIC.
Compile C++ sources with R CMD SHLIB-compatible flags and Rcpp include paths.
Link one shared object at fastkpc/build/fastkpc_cuda.so.
Link against cudart and R.
Create fastkpc/build if missing.
Exit nonzero on any compiler or linker failure.
Print the final shared object path.
Do not modify the existing CPU sourceCpp build path in fastkpc/R/native.R.
```

Rcpp include path requirement:

```bash
Rscript -e 'cat(Rcpp:::CxxFlags())'
```

Use this command or an equivalent `Rscript` call in the build script to locate
Rcpp headers. Do not hard-code a user library path.

The shared object must include:

```text
fastkpc/src/dcov_exact_cpu.cpp
fastkpc/src/skeleton_engine.cpp
fastkpc/src/skeleton_engine_cuda.cpp
fastkpc/src/cuda/cuda_status.cpp
fastkpc/src/cuda/dcov_batch_cuda.cu
fastkpc/src/r_api_cuda.cpp
```

- [ ] Create `fastkpc/tools/clean_cuda_native.sh`.

Required behavior:

```text
Remove fastkpc/build/*.o.
Remove fastkpc/build/fastkpc_cuda.so.
Leave source files unchanged.
Exit 0 if build files are already absent.
```

- [ ] Create `fastkpc/tests/test_cuda_build_contract.R`.

The test must check:

```text
1. fastkpc/R/cuda_native.R can be sourced.
2. build_fastkpc_cuda_native(rebuild = TRUE) produces fastkpc/build/fastkpc_cuda.so.
3. fastkpc_cuda_available() returns TRUE.
4. fastkpc_cuda_device_info() returns at least device_id, name, compute_capability, and total_global_mem.
5. The CPU wrappers from fastkpc/R/native.R still build and run after the CUDA build.
```

- [ ] Run:

```bash
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
Rscript fastkpc/tests/test_cuda_build_contract.R
```

Expected:

```text
The CUDA shared object is built.
test_cuda_build_contract.R prints PASS.
```

## Phase 2: Batched CUDA dCov Core

Purpose: implement the numeric CUDA batch backend and validate it against exact CPU results.

- [ ] Create `fastkpc/src/dcov_batch_types.hpp`.

Required C++ types:

```text
DcovBatchOptions:
  double index
  bool legacy_index

DcovBatchResult:
  std::vector<double> p_values
  std::vector<double> nV2
  std::vector<double> means
  std::vector<double> variances
  std::vector<double> raw_scalars
```

`raw_scalars` layout must be batch-major:

```text
raw_scalars[k * 5 + 0] = Sab
raw_scalars[k * 5 + 1] = Saa
raw_scalars[k * 5 + 2] = Sbb
raw_scalars[k * 5 + 3] = sumK
raw_scalars[k * 5 + 4] = sumL
```

- [ ] Create `fastkpc/src/cuda/dcov_batch_cuda.hpp` and `fastkpc/src/cuda/dcov_batch_cuda.cu`.

Required CUDA behavior:

```text
Input matrices are column-major n by batch.
Use one CUDA call to process the whole batch.
Do not materialize n by n distance matrices.
First pass computes row sums and total sums for K and L for every task.
Second pass computes Sab, Saa, and Sbb for every task.
Use FP64 accumulation.
Return CUDA errors to the caller, not Rf_error inside kernels.
Support legacy_index by treating effective index as 1 without pow().
Support semantic index by applying pow(distance, index).
```

Kernel layout requirement:

```text
rowsum grid includes a task dimension.
fused reduction grid includes a task dimension.
One native call may launch multiple kernels, but R must see one batch call.
```

- [ ] Create `.Call` wrapper in `fastkpc/src/r_api_cuda.cpp`.

Required exported entry point:

```text
C_fast_dcov_batch_cuda(SEXP xs, SEXP ys, SEXP indexs, SEXP legacy_indexs)
```

Registration requirement:

```text
Export the symbol with extern "C".
Include R_ext/Rdynload.h.
Register C_fast_dcov_batch_cuda in R_init_fastkpc_cuda().
Use R_useDynamicSymbols(dll, FALSE).
```

Input validation:

```text
xs and ys must be numeric matrices.
xs and ys must have identical dimensions.
n must be greater than 5.
all input values must be finite.
index outside [0, 2] warns and uses 1.
```

Return value:

```text
list(
  p.value = numeric(batch),
  nV2 = numeric(batch),
  mean = numeric(batch),
  variance = numeric(batch),
  raw = matrix(nrow = batch, ncol = 5)
)
```

- [ ] Create `fastkpc/R/cuda_native.R`.

Required R functions:

```text
build_fastkpc_cuda_native(rebuild = FALSE)
load_fastkpc_cuda_native(rebuild = FALSE)
fastkpc_cuda_available()
fastkpc_cuda_device_info()
fast_dcov_batch_cuda(x, y, index = 1, legacy_index = TRUE)
fast_skeleton_cuda(data, alpha, max_conditioning_size, index = 1, legacy_index = TRUE, batch_size = 0)
```

For `fast_dcov_batch_cuda()`:

```text
If x or y is a vector, coerce it to an n by 1 matrix.
If x or y is a matrix, preserve n rows and batch columns.
storage.mode must be double before .Call.
Return the list from C_fast_dcov_batch_cuda.
```

Loading requirement:

```text
load_fastkpc_cuda_native() must dyn.load("fastkpc/build/fastkpc_cuda.so") exactly once per R session.
If rebuild = TRUE, unload any previously loaded fastkpc_cuda.so before rebuilding when possible.
If the shared object is missing, call build_fastkpc_cuda_native(rebuild = TRUE).
```

- [ ] Create `fastkpc/tests/test_dcov_cuda_batch.R`.

The test must check:

```text
1. batch size 1 matches dcov_gamma_exact() within 1e-10 absolute p-value error.
2. batch size 7 matches dcov_gamma_exact() for every column within 1e-10 absolute p-value error.
3. batch size 64 at n = 300 returns finite p-values in [0, 1].
4. legacy_index = TRUE and legacy_index = FALSE differ when index = 1.5.
5. semantic index = 1.5 matches dcov_gamma_exact(..., legacy_index = FALSE).
6. invalid n <= 5 raises "gamma approximation requires n > 5".
7. non-finite inputs are rejected before launching CUDA.
```

- [ ] Run:

```bash
bash fastkpc/tools/build_cuda_native.sh
Rscript fastkpc/tests/test_dcov_cuda_batch.R
```

Expected:

```text
test_dcov_cuda_batch.R prints PASS.
```

## Phase 3: CUDA Skeleton Batch Engine

Purpose: add an opt-in CUDA-backed skeleton path while preserving CPU exact graph behavior.

- [ ] Create `fastkpc/src/skeleton_engine_cuda.hpp` and `fastkpc/src/skeleton_engine_cuda.cpp`.

Required behavior:

```text
Expose run_skeleton_cuda_batch(data, options, batch_size).
Use the same SkeletonOptions and SkeletonResult contracts as run_skeleton_exact().
Use stable level semantics.
Enumerate tasks from the level snapshot in exactly the CPU skeleton order.
For unconditional tasks, pack data columns directly into X and Y batch matrices.
For conditional tasks, use residualize_lm() from dcov_exact_cpu.cpp, then pack residual vectors.
Call the CUDA dCov batch backend for packed tasks.
Replay p-values sequentially in original task order.
Record pMax symmetrically.
Record sepsets symmetrically.
Apply deletions only at the end of the level.
Count n.edgetests after replay, matching the CPU exact MVP.
```

Required task metadata:

```text
edge_x
edge_y
orientation_x
orientation_y
conditioning_set
task_order
```

The replay logic must ignore speculative tasks for an edge after the first accepted deletion event for that edge in that level.

- [ ] Add `.Call` wrapper in `fastkpc/src/r_api_cuda.cpp`.

Required exported entry point:

```text
C_fast_skeleton_cuda(SEXP data, SEXP alphas, SEXP max_ords, SEXP indexs, SEXP legacy_indexs, SEXP batch_sizes)
```

Registration requirement:

```text
Register C_fast_skeleton_cuda in the same R_init_fastkpc_cuda() table as C_fast_dcov_batch_cuda.
```

Return value must match `fast_skeleton_cpp()`:

```text
list(
  adjacency = logical matrix,
  sepsets = nested list,
  pMax = numeric matrix,
  n.edgetests = integer vector,
  per.level.log = nested list,
  backend = "cuda"
)
```

- [ ] Implement `fast_skeleton_cuda()` in `fastkpc/R/cuda_native.R`.

Required behavior:

```text
Build and load CUDA native code on first use.
Coerce data to a double matrix.
Use batch_size = 0 to mean "auto".
Return the .Call result unchanged.
```

- [ ] Create `fastkpc/tests/test_skeleton_cuda_batch.R`.

The test must source `fastkpc/R/legacy_runner.R` for `fastkpc_fixed_scenario()`.

The test must check:

```text
1. On the fixed four-variable scenario from fastkpc/R/legacy_runner.R, CUDA skeleton and CPU skeleton have identical adjacency.
2. pMax differs by less than 1e-8 elementwise.
3. sepsets match after sorting each conditioning set.
4. n.edgetests is identical.
5. per.level.log has the same number of deletion entries per level.
6. max_conditioning_size = 0 works.
7. max_conditioning_size = 1 works.
8. batch_size = 1 produces the same graph as batch_size = 0.
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_skeleton_cuda_batch.R
```

Expected:

```text
test_skeleton_cuda_batch.R prints PASS.
```

## Phase 4: Graph Difference Validation Report

Purpose: make CPU-vs-CUDA graph behavior visible before treating the backend as usable.

- [ ] Create `fastkpc/R/cuda_validation.R`.

Required functions:

```text
validate_cuda_dcov_batch(n = 300, batch = 16, index = 1, legacy_index = TRUE, seed = 1)
validate_cuda_skeleton_scenario(seed = 4, n = 80, alpha = 0.2, max_conditioning_size = 1)
```

`validate_cuda_dcov_batch()` must return:

```text
list(
  max_abs_p_diff,
  max_abs_nV2_diff,
  all_p_values_finite,
  all_p_values_in_unit_interval
)
```

`validate_cuda_skeleton_scenario()` must return:

```text
list(
  diff = summarize_graph_diff(cpu_result, cuda_result),
  max_abs_pmax_diff,
  adjacency_identical,
  sepsets_identical,
  n_edgetests_identical
)
```

- [ ] Add a short validation command to `fastkpc/README.md`.

Required command:

```bash
Rscript -e 'source("fastkpc/R/native.R"); source("fastkpc/R/cuda_native.R"); source("fastkpc/R/cuda_validation.R"); print(validate_cuda_dcov_batch()); print(validate_cuda_skeleton_scenario())'
```

- [ ] Run the validation command.

Expected:

```text
adjacency_identical is TRUE.
sepsets_identical is TRUE.
n_edgetests_identical is TRUE.
max_abs_pmax_diff is less than 1e-8.
```

If a fixed scenario produces a graph difference because a p-value is numerically on the alpha boundary, do not hide it. Add a second fixed scenario that is not boundary-sensitive and document both results.

## Phase 5: Documentation And Build Hygiene

Purpose: leave the CUDA path usable by the next agent.

- [ ] Update `fastkpc/README.md`.

Required sections:

```text
CUDA Scope
CUDA Build
CUDA Tests
CPU-vs-CUDA Validation
Known Limits
```

Known limits must explicitly state:

```text
Conditional residualization still uses the CPU linear MVP path.
This goal does not implement mgcv equivalence.
This goal does not implement CUDA GAM or fastSpline.
This goal does not replace kpcalg::kpc().
The CUDA backend is opt-in through fastkpc/R/cuda_native.R.
```

- [ ] Confirm generated build artifacts are documented.

Required statement:

```text
fastkpc/build/*.o and fastkpc/build/fastkpc_cuda.so are local build artifacts.
They can be removed with bash fastkpc/tools/clean_cuda_native.sh and rebuilt.
```

- [ ] Run:

```bash
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
find fastkpc/build -maxdepth 1 -type f | sort
```

Expected:

```text
fastkpc/build/fastkpc_cuda.so exists after build.
Object files may also exist under fastkpc/build.
```

## Phase 6: Completion Criteria

The goal is complete only when all of these are true:

```text
1. Previous first-slice tests still pass:
   - fastkpc/tests/test_dcov_exact.R
   - fastkpc/tests/test_skeleton_mvp.R
   - fastkpc/tests/test_diff_report.R

2. gpu-dcov/validate.R still ends with ALL CHECKS PASSED.

3. CUDA native build works from a clean build directory:
   - bash fastkpc/tools/clean_cuda_native.sh
   - bash fastkpc/tools/build_cuda_native.sh

4. CUDA tests pass:
   - fastkpc/tests/test_cuda_build_contract.R
   - fastkpc/tests/test_dcov_cuda_batch.R
   - fastkpc/tests/test_skeleton_cuda_batch.R

5. CPU-vs-CUDA validation reports:
   - adjacency_identical TRUE
   - sepsets_identical TRUE
   - n_edgetests_identical TRUE
   - max_abs_pmax_diff < 1e-8

6. fastkpc/README.md documents CUDA build, tests, scope, artifacts, and known limits.

7. No kpcalg/R/*.R file has been modified.
```

Required final verification command:

```bash
set -e
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
Rscript -e 'source("fastkpc/R/native.R"); build_fastkpc_native(rebuild=TRUE)'
Rscript fastkpc/tests/test_dcov_exact.R
Rscript fastkpc/tests/test_skeleton_mvp.R
Rscript fastkpc/tests/test_diff_report.R
Rscript gpu-dcov/validate.R
Rscript fastkpc/tests/test_cuda_build_contract.R
Rscript fastkpc/tests/test_dcov_cuda_batch.R
Rscript fastkpc/tests/test_skeleton_cuda_batch.R
Rscript -e 'source("fastkpc/R/native.R"); source("fastkpc/R/cuda_native.R"); source("fastkpc/R/cuda_validation.R"); print(validate_cuda_dcov_batch()); print(validate_cuda_skeleton_scenario())'
cd kpcalg
md5sum -c MD5 | rg '^R/'
```

When marking this goal complete, report:

```text
The exact CUDA build command used.
The exact test commands run.
The pass/fail result of each test.
The CPU-vs-CUDA max p-value and pMax differences.
Whether any graph-level differences were observed.
The kpcalg/R MD5 result.
```

## Later Goals

Create separate goals after this CUDA dCov batch goal is complete.

### Later Goal B: Residual Cache

Objective:

```text
Add a residual cache keyed by target variable, conditioning set, residual backend, and backend parameters, initially using the current CPU linear MVP residual backend and then legacy mgcv as a validation backend.
```

### Later Goal C: fastSpline Residual Backend

Objective:

```text
Add a B-spline penalized least-squares residual backend as an explicit statistical replacement for mgcv, with graph-level validation against legacy mgcv.
```

### Later Goal D: WAN-PDAG Migration

Objective:

```text
Migrate kpcalg's udag2wanpdag generalized transitive orientation step to the C++ scheduler, including batched regrVonPS-style residual independence checks.
```

## Execution Rules For Codex

- Work in small commits if the workspace is a git repository. If it is not a git repository, do not initialize one unless the user asks.
- Prefer adding files under `fastkpc/` over modifying legacy files.
- Do not alter `kpcalg/R/*.R`.
- Do not delete or rewrite `gpu-dcov/`; use it as a numeric reference.
- Do not make CPU fallback count as CUDA completion.
- If CUDA build fails, fix the build system before adding skeleton CUDA behavior.
- If CUDA numeric tests fail, debug dCov scalars before debugging graph-level behavior.
- If graph-level CPU-vs-CUDA differences appear, use `fastkpc/R/diff_report.R` to expose them and determine whether the cause is numeric tolerance, replay order, speculative task handling, or a bug.
- Do not implement residual cache, fastSpline, GAM GPU, or WAN-PDAG migration in this goal.
- Do not replace exported `kpcalg::kpc()` in this goal.

## Final Handoff Prompt

After this plan is saved, the next Codex run can be started with:

```text
Create a goal with objective: "Build the next staged fast kPC backend slice from docs/superpowers/plans/2026-06-13-fast-kpc-cuda-batched-dcov-goal-execution.md: add an opt-in CUDA batched exact dCov backend and a CUDA-backed skeleton path, preserving CPU exact behavior as the reference and keeping legacy kpcalg/R files unchanged."
```

Then execute the plan with `superpowers:subagent-driven-development` or `superpowers:executing-plans`.
