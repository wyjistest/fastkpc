# Fast kPC C++/CUDA Backend Goal Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an incremental fast kPC backend that keeps the current R implementation as the legacy reference, first proving a C++ skeleton engine with exact distance-covariance tests, then adding CUDA batch execution and residual backends.

**Architecture:** Keep R as the facade and validation harness. Put graph scheduling, adjacency state, sepsets, pMax, and replay semantics in C++ CPU code. Put only bulk numeric kernels in CUDA, starting from exact distance covariance; do not port `mgcv` or directly patch `pcalg` as the primary architecture.

**Tech Stack:** R 4.4.1, kpcalg 1.0.1 source tree, pcalg 2.7-12 source as algorithm reference, Rcpp/RcppArmadillo for native integration, CUDA 12.5 for later batched kernels, existing `gpu-dcov/` prototype as numeric reference.

---

## Goal Objective For Codex

Use this exact goal objective when starting a Codex goal:

```text
Build a staged fast kPC backend for this kpcalg workspace. First create a validation baseline and C++ skeleton MVP using exact CPU distance covariance, preserving legacy R behavior as a reference and producing graph-level difference reports. Do not start CUDA batch kernels, fastSpline residuals, or WAN-PDAG migration until the C++ skeleton MVP is verified against the R baseline.
```

Recommended token budget for the first goal run: `120000`.

Do not mark the goal complete until Phase 1 and Phase 2 below are implemented, verified, and documented. Mark the goal blocked only if the same blocker prevents progress for three consecutive goal turns and cannot be resolved locally.

## Scope

This plan is for the first reliable slice of the rewrite, not the whole C++/CUDA system.

In scope for the first goal:

- Create a reproducible validation baseline for legacy R `kpcalg`.
- Add an exact CPU distance-covariance implementation suitable for testing.
- Add a C++ skeleton scheduler MVP, modeled after `pcalg::skeleton(method="stable.fast")`, but without R scalar `indepTest` callbacks in the core.
- Produce per-level and final graph difference reports.
- Keep all CUDA work behind a later phase gate.

Out of scope for the first goal:

- Do not implement CUDA batched dcov yet.
- Do not implement fastSpline or CUDA residual regression yet.
- Do not migrate `udag2wanpdag()` yet.
- Do not replace exported `kpc()` behavior.
- Do not support `hsic.gamma`, `hsic.perm`, `dcc.perm`, or `hsic.clust`.
- Do not try to replicate all `mgcv`, `pcalg`, or `kpcalg` semantics.

## Project Facts To Preserve

- `ic.method="dcc.gamma"` in `kpcalg/R/kernelCItest.R` dispatches to `dcov.gamma()`.
- Current `dcov.gamma()` uses `RSpectra::eigs()` as a low-rank approximation, but exact dCov gamma statistics can be computed from double-centered distance matrices without eigendecomposition.
- The exact statistic scaling to match kpcalg naming is:

```text
A = H K H
B = H L H
nV2 = sum(A * B) / n
dCov = sqrt(nV2 / n)
nV2Mean = mean(K) * mean(L)
nV2Variance =
  2 * (n - 4) * (n - 5) / (n * (n - 1) * (n - 2) * (n - 3))
  * sum(A * A) * sum(B * B) / n^2
```

- The original `index` argument is checked but not applied to `dist(x)` in `kpcalg/R/dcovgamma.R`. New exact code must expose this as a behavior choice:

```text
legacy_index = TRUE: preserve old kpcalg behavior and ignore distance exponent
legacy_index = FALSE: apply K = dist(x)^index and L = dist(y)^index
```

- `pcalg` already has a C++ skeleton implementation in `src/constraint.cpp`, reachable from R with `skeleton(..., method="stable.fast")`. It is useful as a reference, but its `IndepTest::test(u, v, S)` scalar callback shape is not the final GPU-friendly interface.
- Full `kpcalg::kpc()` also calls `udag2wanpdag()`, whose generalized transitive step calls `regrVonPS()`, which calls `regrXonS()` and then `indepTest()` again. This is a later migration phase.

## File Structure

Create or modify these files during the first goal.

- Create: `fastkpc/R/dcov_exact.R`  
  R exact distance-covariance reference implementation used by tests and validation scripts.

- Create: `fastkpc/R/legacy_runner.R`  
  Helpers to run legacy R `kernelCItest`, skeleton-level baselines, and small fixed scenarios.

- Create: `fastkpc/R/diff_report.R`  
  Functions that compare adjacency matrices, sepsets, pMax matrices, per-level deletion logs, and final graph summaries.

- Create: `fastkpc/src/fastkpc_types.hpp`  
  Small C++ structs for tasks, p-values, skeleton state, sepsets, and run logs.

- Create: `fastkpc/src/dcov_exact_cpu.hpp`
- Create: `fastkpc/src/dcov_exact_cpu.cpp`  
  C++ exact CPU distance-covariance p-value implementation. This is the oracle backend for the C++ skeleton MVP.

- Create: `fastkpc/src/skeleton_engine.hpp`
- Create: `fastkpc/src/skeleton_engine.cpp`  
  C++ skeleton scheduler with stable level semantics, adjacency state, conditioning-set enumeration, sepset recording, pMax recording, and n.edgetests.

- Create: `fastkpc/src/rcpp_exports.cpp`  
  Rcpp entry points for exact dcov and skeleton MVP.

- Create: `fastkpc/R/native.R`  
  R wrappers around the Rcpp entry points.

- Create: `fastkpc/tests/test_dcov_exact.R`
- Create: `fastkpc/tests/test_skeleton_mvp.R`
- Create: `fastkpc/tests/test_diff_report.R`  
  R tests for exact dCov, C++ skeleton behavior, and diff reporting.

- Create: `fastkpc/README.md`  
  Build, test, and scope notes for the fast backend.

- Modify only if needed: `kpcalg/R/kernelCItest.R`  
  Do not change this in the first goal unless adding an explicitly opt-in experimental path. The preferred first-goal integration is through `fastkpc/`.

## Phase Gates

Follow these phase gates strictly.

### Phase 0: Workspace And Baseline Inspection

Purpose: confirm the workspace state and avoid accidentally modifying legacy files.

- [ ] Run:

```bash
pwd
find . -maxdepth 3 -type f | sort
find gpu-dcov -maxdepth 1 -type f | sort
find kpcalg/R -maxdepth 1 -type f | sort
```

Expected:

```text
The working directory is /data/wenyujianData/kpcalg.
The repository contains kpcalg/, gpu-dcov/, kpcalg_1.0.1.tar.gz, and mgcv_1.9-4.tar.gz.
```

- [ ] Run:

```bash
Rscript -e 'cat("R ", as.character(getRversion()), "\n", sep=""); for (p in c("RSpectra","kernlab","mgcv","Rcpp","RcppArmadillo")) cat(p, ": ", requireNamespace(p, quietly=TRUE), "\n", sep="")'
```

Expected:

```text
R version is printed.
mgcv, Rcpp, and RcppArmadillo should be available before native work starts.
If Rcpp or RcppArmadillo is unavailable, stop and report the missing dependency.
```

- [ ] Run:

```bash
Rscript gpu-dcov/validate.R
```

Expected:

```text
The script ends with ALL CHECKS PASSED.
If it fails, do not proceed to C++ skeleton work; first document the numeric mismatch.
```

### Phase 1: Exact CPU Distance Covariance Baseline

Purpose: remove eigendecomposition from the statistical core and establish exact CPU behavior.

- [ ] Create `fastkpc/R/dcov_exact.R` with an R function named `dcov_gamma_exact()`.

Required behavior:

```text
Input: x, y, index = 1, legacy_index = TRUE.
Reject non-finite data.
Reject unequal sample sizes.
Reject n <= 5 with an explicit error: "gamma approximation requires n > 5".
If legacy_index is TRUE, do not apply dist^index.
If legacy_index is FALSE, apply dist^index to both distance matrices.
Use double-centering by row means and grand mean.
Use pgamma(..., lower.tail = FALSE).
Return an htest-shaped list with statistic, estimate, estimates, p.value, replicates, and data.name.
```

- [ ] Create `fastkpc/tests/test_dcov_exact.R`.

The test file must check:

```text
1. For index = 1, dcov_gamma_exact() matches gpu-dcov's exact scalar values within 1e-10 relative error on n = 300 and n = 1000.
2. n <= 5 raises "gamma approximation requires n > 5".
3. legacy_index = TRUE and legacy_index = FALSE differ when index = 1.5.
4. p.value is finite and in [0, 1].
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_dcov_exact.R
```

Expected:

```text
All exact dCov tests pass.
```

### Phase 2: C++ Skeleton MVP With CPU Exact CI

Purpose: prove the C++ control-flow engine before adding CUDA.

- [ ] Create `fastkpc/src/fastkpc_types.hpp`.

Required types:

```text
CiTask: x, y, conditioning_set.
CiResult: x, y, conditioning_set, p_value.
SkeletonOptions: alpha, max_conditioning_size, na_delete, stable.
SkeletonResult: adjacency, sepsets, pmax, n_edge_tests, per_level_log.
```

- [ ] Create `fastkpc/src/dcov_exact_cpu.hpp` and `fastkpc/src/dcov_exact_cpu.cpp`.

Required behavior:

```text
Compute exact dCov gamma p-values for two vectors.
Support legacy_index and semantic index modes.
Use no eigendecomposition.
Use no explicit H matrix.
Use lower-tail false gamma p-values.
Return NaN only if the numeric calculation is invalid after input checks.
```

- [ ] Create `fastkpc/src/skeleton_engine.hpp` and `fastkpc/src/skeleton_engine.cpp`.

Required behavior:

```text
Initialize a complete undirected graph except fixed gaps.
For each conditioning-set size, enumerate candidate edges using stable level semantics.
For each candidate, enumerate conditioning sets from the adjacency snapshot for that level.
Call the exact CPU CI backend.
Record pMax[x,y] and pMax[y,x].
When p >= alpha, mark the edge for deletion and record sepset.
Apply deletions at the end of the level, not immediately.
Record n.edgetests per level.
Return adjacency, sepsets, pMax, and a per-level deletion log.
```

- [ ] Create `fastkpc/src/rcpp_exports.cpp` and `fastkpc/R/native.R`.

Required R entry points:

```text
fast_dcov_exact_cpp(x, y, index = 1, legacy_index = TRUE)
fast_skeleton_cpp(data, alpha, max_conditioning_size, index = 1, legacy_index = TRUE)
```

The first implementation may assume `dcc.gamma.exact` only. Do not add `hsic` or permutation tests.

- [ ] Build the native code with a local command documented in `fastkpc/README.md`.

Expected:

```text
The build produces a loadable shared object for fastkpc native functions.
The build command must not modify kpcalg/R/*.R.
```

- [ ] Create `fastkpc/tests/test_skeleton_mvp.R`.

The test file must check:

```text
1. C++ exact dCov p-values match R exact dCov p-values on fixed vectors.
2. C++ skeleton returns symmetric adjacency with FALSE diagonal.
3. pMax is symmetric and has diagonal 1 or a documented diagonal value.
4. n.edgetests has one count per tested conditioning level.
5. On a fixed small synthetic dataset, C++ skeleton and an R exact skeleton reference produce the same adjacency and sepsets.
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_skeleton_mvp.R
```

Expected:

```text
All C++ skeleton MVP tests pass.
```

### Phase 3: Difference Reporting

Purpose: make behavior changes visible before adding performance features.

- [ ] Create `fastkpc/R/diff_report.R`.

Required reports:

```text
compare_adjacency(old, new): added_edges, removed_edges, unchanged_edges.
compare_pmax(old, new): max_abs_diff, mean_abs_diff, top_20_diffs.
compare_sepsets(old, new): matching_count, differing_count, differing_pairs.
summarize_graph_diff(old_result, new_result): single list combining adjacency, pMax, sepset, and n.edgetests differences.
```

- [ ] Create `fastkpc/tests/test_diff_report.R`.

The test file must check:

```text
1. Added and removed undirected edges are counted once, not twice.
2. pMax difference summary sorts largest differences first.
3. Sepset comparison reports the exact node pairs whose sepsets differ.
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_diff_report.R
```

Expected:

```text
All diff report tests pass.
```

### Phase 4: First Goal Completion Criteria

The first Codex goal is complete only when all of these are true:

```text
1. fastkpc/R/dcov_exact.R exists and passes tests.
2. fastkpc/src skeleton MVP exists and builds locally.
3. fastkpc/tests/test_dcov_exact.R passes.
4. fastkpc/tests/test_skeleton_mvp.R passes.
5. fastkpc/tests/test_diff_report.R passes.
6. fastkpc/README.md documents how to build and run tests.
7. No existing kpcalg/R/*.R file has been modified unless the user explicitly approved it.
```

When marking this goal complete, report:

```text
The exact command used to build native code.
The exact test commands run.
The pass/fail result of each test.
Any graph-level differences observed in fixed validation scenarios.
```

## Later Goals

Create separate goals after the first goal is complete.

### Later Goal A: CUDA Batched dCov

Objective:

```text
Replace the C++ skeleton MVP's exact CPU dCov backend with a CUDA batched exact dCov backend that keeps data resident on GPU and processes per-level CI tasks in batches.
```

Required preconditions:

```text
C++ skeleton MVP passes all tests.
gpu-dcov/validate.R passes.
The batch task format from Phase 2 is stable.
```

### Later Goal B: Residual Cache

Objective:

```text
Add a residual cache keyed by target variable, conditioning set, residual backend, and backend parameters, initially using legacy mgcv as the residual backend.
```

Required preconditions:

```text
C++ skeleton MVP exists.
Diff reports can expose graph changes caused by cached residual reuse.
```

### Later Goal C: fastSpline Residual Backend

Objective:

```text
Add a B-spline penalized least-squares residual backend as an explicit statistical replacement for mgcv, with graph-level validation against legacy mgcv.
```

Required preconditions:

```text
Residual cache exists.
Legacy mgcv residual backend exists as a validation reference.
```

### Later Goal D: WAN-PDAG Migration

Objective:

```text
Migrate kpcalg's udag2wanpdag generalized transitive orientation step to the C++ scheduler, including batched regrVonPS-style residual independence checks.
```

Required preconditions:

```text
C++ skeleton MVP is stable.
Residual backend has at least one validated fast option.
Diff reports include final graph comparisons.
```

## Execution Rules For Codex

- Work in small commits if the workspace is a git repository. If it is not a git repository, do not initialize one unless the user asks.
- Prefer adding files under `fastkpc/` over modifying legacy files.
- Do not delete or rewrite `gpu-dcov/`; use it as a numeric reference.
- Do not alter `kpcalg/R/*.R` during the first goal unless the user explicitly approves the integration path.
- If tests reveal differences from legacy R, do not hide them. Add them to the diff report and explain whether the difference comes from exact dCov replacing truncated `eigs`, index semantics, p-value tail handling, or a bug.
- If CUDA is tempting during Phase 1 or Phase 2, stop. CUDA starts only after the C++ skeleton MVP is verified.
- If a dependency is missing, report the exact missing package and the command that failed.
- If a native build fails, keep the error output and fix the build system before adding new features.

## Final Handoff Prompt

After this plan is saved, the next Codex run can be started with:

```text
Create a goal with objective: "Build the first staged fast kPC backend slice from docs/superpowers/plans/2026-06-13-fast-kpc-goal-execution.md: implement Phase 0 through Phase 4 only, keeping legacy kpcalg files unchanged unless explicitly approved."
```

Then execute the plan with `superpowers:subagent-driven-development` or `superpowers:executing-plans`.
