# Fast kPC WAN-PDAG Orientation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in native WAN-PDAG orientation stage to `fastkpc` that consumes the existing native skeleton results, reuses the fastSpline residual cache for generalized transitive orientation checks, and validates graph-level output against legacy `kpcalg::udag2wanpdag()` without modifying `kpcalg/R/*.R`.

**Architecture:** Keep the completed exact dCov, residual cache, fastSpline backend, CPU skeleton, and CUDA skeleton stable. Add a native orientation engine that represents partially directed graphs as integer adjacency matrices, ports the collider step and three orientation rules from legacy `udag2wanpdag()`, and implements the generalized transitive orientation loop with backend-aware residual independence checks. Expose this as a separate opt-in fastkpc API; do not replace `kpcalg::kpc()` in this goal.

**Tech Stack:** R 4.4.1, Rcpp/RcppArmadillo, C++17, CUDA 12.5 for existing skeleton/dCov paths, native fastSpline residual backend, exact dCov gamma p-values, legacy `kpcalg/R` loaders for validation, optional `pcalg`/`graph` packages for legacy object comparison.

---

## Goal Objective For Codex

Use this exact goal objective when starting a Codex goal:

```text
Build the next staged fast kPC backend slice from docs/superpowers/plans/2026-06-13-fast-kpc-wanpdag-orientation-goal-execution.md: add an opt-in native WAN-PDAG orientation stage that consumes fastkpc skeleton results, reuses fastSpline residual-cache logic for generalized transitive orientation checks, validates against legacy kpcalg::udag2wanpdag(), and keeps kpcalg/R files unchanged.
```

Recommended token budget for the goal run: `340000`.

This goal is intentionally larger than the fastSpline residual backend goal. It should run long enough to cover core algorithm migration, CPU/CUDA skeleton integration, validation, and documentation. Do not mark the goal complete until all completion criteria in Phase 11 are satisfied.

## Preconditions

The following previous stages must already be complete:

```text
exact CPU dCov backend
CUDA batched dCov backend
CPU and CUDA skeleton engines
residual cache
fastSpline residual backend
fastSpline validation and benchmark helpers
```

Required current files:

```text
fastkpc/R/native.R
fastkpc/R/cuda_native.R
fastkpc/R/diff_report.R
fastkpc/R/legacy_runner.R
fastkpc/R/fastspline_validation.R
fastkpc/src/fastkpc_types.hpp
fastkpc/src/residual_backend_registry.hpp
fastkpc/src/residual_cache.hpp
fastkpc/src/skeleton_engine.cpp
fastkpc/src/skeleton_engine_cuda.cpp
fastkpc/src/rcpp_exports.cpp
fastkpc/src/r_api_cuda.cpp
fastkpc/tools/build_cuda_native.sh
fastkpc/tests/test_skeleton_fastspline_cpu.R
fastkpc/tests/test_skeleton_fastspline_cuda.R
```

Required environment:

```text
R 4.4.1
Rcpp installed
RcppArmadillo installed
mgcv installed
CUDA toolkit available at /usr/local/cuda/bin/nvcc
NVIDIA driver able to run CUDA kernels
```

Optional but recommended validation packages:

```text
pcalg
graph
RSpectra
```

If `pcalg` or `graph` is unavailable, native WAN-PDAG implementation work may continue, but the goal cannot be marked complete unless the legacy object validation test records explicit package-unavailable diagnostics and all package-independent orientation tests pass. If CUDA is unavailable, the goal cannot be complete because CUDA skeleton to WAN-PDAG integration is in scope.

## Scope

In scope:

- Add a native orientation module that consumes a `SkeletonResult`.
- Represent partially directed graphs as integer matrices:

```text
0 = no edge
1/1 pair = undirected edge
1/0 pair = directed row -> column
2/2 pair = bidirected conflict marker when solve_confl = TRUE
```

- Port these parts of legacy `kpcalg/R/udag2wanpdag.R`:

```text
collider orientation step
orientConflictCollider behavior
checkImmor behavior
rule1
rule2
rule3
fixed-point rule application loop
generalized transitive orientation loop
```

- Add native `regrVonPS` equivalent using:

```text
selected residual_backend = "linear" or "fastSpline"
existing exact dCov gamma p-value
existing residual cache and backend params
parents(V) + S residualization set
```

- Add CPU and CUDA wrapper flows:

```text
fast_kpc_wanpdag_cpp()
fast_kpc_wanpdag_cuda()
fast_orient_wanpdag_cpp()
```

- Validate the native orientation engine against:

```text
hand-written small graph cases
legacy udag2wanpdag collider/rule behavior
legacy udag2wanpdag graph-level output where pcalg/graph are available
CPU skeleton + native WAN-PDAG vs CUDA skeleton + native WAN-PDAG
linear residual backend vs fastSpline residual backend with explicit diff reports
```

- Preserve existing fastkpc APIs and tests.
- Keep `kpcalg/R/*.R` unchanged.

Out of scope:

- Do not replace exported `kpcalg::kpc()`.
- Do not implement `pcAlgo` S4 object creation in native C++.
- Do not implement HSIC, permutation tests, or cluster tests.
- Do not implement CUDA residual kernels.
- Do not implement multi-GPU scheduling.
- Do not change legacy `kpcalg/R/*.R`.
- Do not require graph equality with legacy mgcv in every statistical scenario; report graph differences explicitly where smoothing differences are expected.

## Design Contract

### Orientation Matrix Contract

Native orientation code must use row-major `std::vector<int>` matrices with dimension `p`.

Required helpers:

```cpp
int pdag_get(const std::vector<int>& pdag, int p, int row, int col);
void pdag_set(std::vector<int>* pdag, int p, int row, int col, int value);
bool has_undirected_edge(const std::vector<int>& pdag, int p, int a, int b);
bool has_directed_edge(const std::vector<int>& pdag, int p, int a, int b);
bool has_any_edge(const std::vector<int>& pdag, int p, int a, int b);
```

Directed edge `a -> b` means:

```text
pdag[a,b] = 1
pdag[b,a] = 0
```

Undirected edge `a - b` means:

```text
pdag[a,b] = 1
pdag[b,a] = 1
```

Conflict/bidirected edge means:

```text
pdag[a,b] = 2
pdag[b,a] = 2
```

### Skeleton Input Contract

`SkeletonResult` already contains:

```text
adjacency
sepsets
pmax
n_edge_tests
per_level_log
residual cache stats
```

The orientation engine must consume:

```text
data matrix
SkeletonResult
SkeletonOptions
OrientationOptions
```

The initial `pdag` must be built from the undirected skeleton adjacency:

```text
if skeleton adjacency i,j is TRUE and i != j:
  pdag[i,j] = 1
  pdag[j,i] = 1
else:
  pdag[i,j] = 0
```

### Result Contract

Add:

```cpp
struct OrientationOptions {
  double alpha;
  bool verbose;
  bool solve_confl;
  bool orient_collider;
  bool rule1;
  bool rule2;
  bool rule3;
  bool residual_cache_enabled;
  std::string residual_backend_name;
  FastSplineParams fastspline_params;
  double index;
  bool legacy_index;
};

struct OrientationResult {
  std::vector<int> pdag;
  int p;
  std::vector<OrientationEvent> events;
  int collider_orientations;
  int rule1_orientations;
  int rule2_orientations;
  int rule3_orientations;
  int generalized_orientations;
  int regrvonps_calls;
  int residual_cache_requests;
  int residual_cache_hits;
  int residual_cache_computations;
  std::string residual_backend;
  std::string residual_backend_params;
};
```

`OrientationEvent` must record:

```text
phase
rule
x
y
z
S
p_value
accepted
message
```

Use `-1` for absent `z`, and use 0-based indices in C++ but convert to 1-based in R outputs.

### regrVonPS Native Contract

Legacy behavior:

```r
parentsV <- which(G[,V] == 1 & G[V,] == 0)
residV <- regrXonS(data[,V], data[,c(S, parentsV)])
for each W in S:
  pval <- indepTest(residV, data[,W])
return sum(pval < alpha)
```

Native behavior must match the same decision surface with exact dCov gamma:

```text
conditioning_for_residual = sorted unique union(S, parentsV)
residV = residual_backend(data, V, conditioning_for_residual)
for each W in S:
  pval = dcov_exact_pvalue(residV, data[,W], index, legacy_index)
reject_count = number of pval < alpha
```

Return diagnostics:

```text
reject_count
p_values
parents
conditioning_set
residual_cache_stats_delta
```

If `conditioning_for_residual` is empty, residualization must return centered target residuals with intercept behavior compatible with the selected backend. For this goal, `regrVonPS` is only called with non-empty `S`, but the helper must still handle empty sets without crashing.

### checkImmor Contract

Port legacy `checkImmor()`:

```text
udag = pmin(pdag + transpose(pdag), 1)
parentsV = nodes with pdag[parent,V] == 1 and pdag[V,parent] == 0
if length(S) > 1:
  test.dag1 = udag[S,S] + diag(length(S))
  if any non-adjacent pair inside S: return false
test.dag2 = udag[parentsV,S]
if sum(test.dag2) < length(S) * length(parentsV): return false
return true
```

When `parentsV` is empty, the second condition must be true.

### Legacy Compatibility Contract

For deterministic small graph fixtures, native and legacy orientation must match exactly:

```text
collider-only orientation
rule1-only orientation
rule2-only orientation
rule3-only orientation
fixed-point rules
generalized transitive loop on a small synthetic graph
```

For statistical graph scenarios, exact graph equality to legacy mgcv is not a blanket hard requirement. The validation report must include:

```text
adjacency directed-edge diff
undirected-edge diff
bidirected-edge diff
pdag matrix max absolute difference
orientation event counts
regrVonPS call count
cache stats
legacy availability diagnostics
```

CPU skeleton + native orientation and CUDA skeleton + native orientation must match exactly except for pMax tolerance inherited from skeleton:

```text
pdag identical
event counts identical
adjacency/pdag diff empty
max skeleton pMax diff < 1e-8
```

## File Structure

Create these files:

- `fastkpc/src/orientation_types.hpp`  
  Native orientation structs, event structs, result structs, option structs, and pdag constants.

- `fastkpc/src/orientation_matrix.hpp`
- `fastkpc/src/orientation_matrix.cpp`  
  Matrix accessors, edge predicates, directed/undirected setters, pdag diff helpers, and selftest helpers.

- `fastkpc/src/orientation_rules.hpp`
- `fastkpc/src/orientation_rules.cpp`  
  Collider step, conflict collider handling, `checkImmor`, rule1, rule2, rule3, and fixed-point rule loop.

- `fastkpc/src/regrvonps_native.hpp`
- `fastkpc/src/regrvonps_native.cpp`  
  Native residual independence checks for generalized transitive orientation using existing residual backend registry/cache and exact dCov.

- `fastkpc/src/wanpdag_engine.hpp`
- `fastkpc/src/wanpdag_engine.cpp`  
  End-to-end orientation engine that consumes skeleton output and runs collider/rules/generalized transitive loop.

- `fastkpc/R/wanpdag_validation.R`  
  R validation helpers, legacy comparison helpers, pdag diff helpers, benchmark/report helpers.

- `fastkpc/tests/test_orientation_matrix.R`
- `fastkpc/tests/test_orientation_rules.R`
- `fastkpc/tests/test_regrvonps_native.R`
- `fastkpc/tests/test_wanpdag_engine_core.R`
- `fastkpc/tests/test_wanpdag_cpu_pipeline.R`
- `fastkpc/tests/test_wanpdag_cuda_pipeline.R`
- `fastkpc/tests/test_wanpdag_legacy_validation.R`
- `fastkpc/tests/test_wanpdag_benchmark.R`
- `fastkpc/tests/test_wanpdag_docs_contract.R`

Modify these files:

- `fastkpc/src/fastkpc_types.hpp`  
  Do not move existing skeleton structs. Prefer keeping all orientation structs in `orientation_types.hpp`; modify this file only to add small shared metadata fields required by skeleton-to-orientation conversion.

- `fastkpc/src/rcpp_exports.cpp`  
  Add CPU orientation exports, test helpers, and result conversion helpers.

- `fastkpc/src/r_api_cuda.cpp`  
  Add CUDA skeleton-to-orientation wrapper that runs CUDA skeleton then native orientation.

- `fastkpc/R/native.R`  
  Add CPU orientation wrappers and selftest wrappers.

- `fastkpc/R/cuda_native.R`  
  Add CUDA WAN-PDAG pipeline wrapper.

- `fastkpc/tools/build_cuda_native.sh`  
  Compile/link orientation sources into `fastkpc_cuda.so`.

- `fastkpc/README.md`  
  Document WAN-PDAG scope, API, validation, benchmark, and known limits.

Do not modify:

- `kpcalg/R/*.R`
- `gpu-dcov/*` except for running validation

## Phase 0: Baseline Audit

Purpose: prove the completed fastSpline backend stage is green before adding orientation.

- [ ] Run:

```bash
pwd
find fastkpc -maxdepth 3 -type f | sort
Rscript -e 'cat("R ", as.character(getRversion()), "\n", sep=""); for (p in c("Rcpp","RcppArmadillo","mgcv","pcalg","graph","RSpectra")) cat(p, ": ", requireNamespace(p, quietly=TRUE), "\n", sep="")'
/usr/local/cuda/bin/nvcc --version
nvidia-smi
cd kpcalg
md5sum -c MD5 | rg '^R/'
```

Expected:

```text
Working directory is /data/wenyujianData/kpcalg.
Rcpp, RcppArmadillo, and mgcv are TRUE.
CUDA toolkit and at least one GPU are visible.
Every kpcalg/R MD5 line reports OK.
pcalg, graph, and RSpectra availability is recorded.
```

- [ ] Run current full fastSpline verification:

```bash
set -e
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
Rscript -e 'source("fastkpc/R/native.R"); build_fastkpc_native(rebuild=TRUE)'
Rscript fastkpc/tests/test_dcov_exact.R
Rscript fastkpc/tests/test_skeleton_mvp.R
Rscript fastkpc/tests/test_diff_report.R
Rscript fastkpc/tests/test_cuda_build_contract.R
Rscript fastkpc/tests/test_dcov_cuda_batch.R
Rscript fastkpc/tests/test_skeleton_cuda_batch.R
Rscript fastkpc/tests/test_residual_cache_core.R
Rscript fastkpc/tests/test_skeleton_residual_cache.R
Rscript fastkpc/tests/test_cuda_residual_cache.R
Rscript fastkpc/tests/test_fastspline_basis.R
Rscript fastkpc/tests/test_fastspline_solver.R
Rscript fastkpc/tests/test_residual_backend_registry.R
Rscript fastkpc/tests/test_skeleton_fastspline_cpu.R
Rscript fastkpc/tests/test_skeleton_fastspline_cuda.R
Rscript fastkpc/tests/test_fastspline_mgcv_validation.R
Rscript fastkpc/tests/test_fastspline_benchmark.R
```

Expected:

```text
All listed tests print PASS.
```

## Phase 1: Orientation Matrix Core

Purpose: establish a small, deterministic pdag matrix utility layer before porting rules.

- [ ] Create `fastkpc/tests/test_orientation_matrix.R`.

The test must call:

```r
orientation_matrix_selftest()
```

Required result fields:

```text
empty_has_no_edges
undirected_roundtrip
directed_roundtrip
conflict_roundtrip
edge_predicates_correct
from_skeleton_symmetric
diff_counts_correct
invalid_indices_rejected
```

Assertions:

```text
all fields are TRUE
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_orientation_matrix.R
```

Expected:

```text
The test fails because orientation_matrix_selftest() does not exist yet.
```

- [ ] Create `fastkpc/src/orientation_types.hpp`.

Required constants:

```cpp
static const int FASTKPC_EDGE_NONE = 0;
static const int FASTKPC_EDGE_PRESENT = 1;
static const int FASTKPC_EDGE_CONFLICT = 2;
```

Required structs:

```cpp
struct OrientationEvent {
  std::string phase;
  std::string rule;
  int x;
  int y;
  int z;
  std::vector<int> S;
  double p_value;
  bool accepted;
  std::string message;
};

struct OrientationOptions {
  double alpha;
  bool verbose;
  bool solve_confl;
  bool orient_collider;
  bool rule1;
  bool rule2;
  bool rule3;
  bool residual_cache_enabled;
  std::string residual_backend_name;
  FastSplineParams fastspline_params;
  double index;
  bool legacy_index;
};

struct OrientationResult {
  std::vector<int> pdag;
  int p;
  std::vector<OrientationEvent> events;
  int collider_orientations;
  int rule1_orientations;
  int rule2_orientations;
  int rule3_orientations;
  int generalized_orientations;
  int regrvonps_calls;
  int residual_cache_requests;
  int residual_cache_hits;
  int residual_cache_computations;
  std::string residual_backend;
  std::string residual_backend_params;
};
```

- [ ] Create `fastkpc/src/orientation_matrix.hpp` and `fastkpc/src/orientation_matrix.cpp`.

Required API:

```cpp
int pdag_index(int p, int row, int col);
int pdag_get(const std::vector<int>& pdag, int p, int row, int col);
void pdag_set(std::vector<int>* pdag, int p, int row, int col, int value);
bool has_any_edge(const std::vector<int>& pdag, int p, int a, int b);
bool has_undirected_edge(const std::vector<int>& pdag, int p, int a, int b);
bool has_directed_edge(const std::vector<int>& pdag, int p, int a, int b);
bool has_conflict_edge(const std::vector<int>& pdag, int p, int a, int b);
void set_no_edge(std::vector<int>* pdag, int p, int a, int b);
void set_undirected_edge(std::vector<int>* pdag, int p, int a, int b);
void set_directed_edge(std::vector<int>* pdag, int p, int from, int to);
void set_conflict_edge(std::vector<int>* pdag, int p, int a, int b);
std::vector<int> pdag_from_skeleton_adjacency(const std::vector<int>& adjacency, int p);
```

Index validation:

```text
row/col outside [0,p) throws "pdag index out of range".
pdag vector length != p*p throws "pdag dimension mismatch".
```

- [ ] Add `orientation_matrix_selftest_export()` to `fastkpc/src/rcpp_exports.cpp` and wrapper in `fastkpc/R/native.R`:

```r
orientation_matrix_selftest <- function() {
  build_fastkpc_native()
  orientation_matrix_selftest_export()
}
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_orientation_matrix.R
Rscript fastkpc/tests/test_fastspline_solver.R
```

Expected:

```text
Both tests print PASS.
```

## Phase 2: Collider And Orientation Rules

Purpose: port legacy collider orientation and rules 1-3 with deterministic fixtures.

- [ ] Create `fastkpc/tests/test_orientation_rules.R`.

Required helper:

```r
orientation_rules_selftest()
```

Required result fields:

```text
collider_orients_unshielded_triple
collider_respects_sepset
conflict_collider_marks_bidirected
check_immor_accepts_clique_S
check_immor_rejects_nonclique_S
check_immor_rejects_unconnected_parent
rule1_orients_chain_tail
rule2_orients_directed_chain
rule3_orients_double_parent_pattern
fixed_point_converges
rules_disabled_no_change
```

Assertions:

```text
all fields are TRUE
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_orientation_rules.R
```

Expected:

```text
The test fails because orientation_rules_selftest() does not exist yet.
```

- [ ] Create `fastkpc/src/orientation_rules.hpp` and `fastkpc/src/orientation_rules.cpp`.

Required API:

```cpp
bool check_immor(const std::vector<int>& pdag,
                 int p,
                 int V,
                 const std::vector<int>& S);

int orient_colliders(std::vector<int>* pdag,
                     int p,
                     const std::vector<std::vector<std::vector<int> > >& sepsets,
                     bool solve_confl,
                     const std::vector<int>& unf_vect,
                     std::vector<OrientationEvent>* events);

int apply_rule1(std::vector<int>* pdag,
                int p,
                bool solve_confl,
                const std::vector<int>& unf_vect,
                std::vector<OrientationEvent>* events);

int apply_rule2(std::vector<int>* pdag,
                int p,
                bool solve_confl,
                std::vector<OrientationEvent>* events);

int apply_rule3(std::vector<int>* pdag,
                int p,
                bool solve_confl,
                const std::vector<int>& unf_vect,
                std::vector<OrientationEvent>* events);

struct RuleApplicationCounts {
  int rule1;
  int rule2;
  int rule3;
};

RuleApplicationCounts apply_rules_until_converged(
  std::vector<int>* pdag,
  int p,
  const OrientationOptions& options,
  const std::vector<int>& unf_vect,
  std::vector<OrientationEvent>* events);
```

Implementation rules:

```text
Use legacy loop order: rows increasing, columns increasing, neighbor lists increasing.
Do not implement triple2numb fully in this goal; unf_vect can be accepted but empty-vector behavior must be correct.
When unf_vect is non-empty, add a diagnostic event with message "unfVect not implemented in native orientation" and skip conservative exclusions.
```

- [ ] Add `orientation_rules_selftest_export()` and wrapper:

```r
orientation_rules_selftest <- function() {
  build_fastkpc_native()
  orientation_rules_selftest_export()
}
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_orientation_rules.R
Rscript fastkpc/tests/test_orientation_matrix.R
```

Expected:

```text
Both tests print PASS.
```

## Phase 3: Native regrVonPS

Purpose: implement the residual independence decision used by generalized transitive orientation.

- [ ] Create `fastkpc/tests/test_regrvonps_native.R`.

Required helper:

```r
regrvonps_native_selftest()
```

Required test scenarios:

```text
1. linear backend returns reject_count > 0 on a dependent residual scenario.
2. fastSpline backend returns reject_count == 0 on y = sin(S) + independent noise after residualization.
3. parents(V) are included in the residual conditioning set.
4. residual cache stats have hits on repeated calls.
5. p_values length equals length(S).
6. unknown backend raises "Unknown residual backend".
7. empty S returns reject_count == 0 and no p_values.
```

Required result fields:

```text
dependent_linear_rejects
smooth_fastspline_accepts
parents_in_conditioning
cache_hits_repeated
pvalue_count_correct
unknown_backend_error
empty_S_safe
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_regrvonps_native.R
```

Expected:

```text
The test fails because regrvonps_native_selftest() does not exist yet.
```

- [ ] Create `fastkpc/src/regrvonps_native.hpp` and `fastkpc/src/regrvonps_native.cpp`.

Required structs:

```cpp
struct RegrVonPsResult {
  int reject_count;
  std::vector<double> p_values;
  std::vector<int> parents;
  std::vector<int> conditioning_set;
  int cache_requests_before;
  int cache_requests_after;
  int cache_hits_before;
  int cache_hits_after;
};
```

Required API:

```cpp
std::vector<int> parents_of(const std::vector<int>& pdag, int p, int V);
std::vector<int> sorted_unique_union(const std::vector<int>& a,
                                     const std::vector<int>& b);

RegrVonPsResult regrvonps_native(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& pdag,
  int p,
  int V,
  const std::vector<int>& S,
  const OrientationOptions& options,
  ResidualCache* residual_cache);
```

Implementation details:

```text
Use parents_of(pdag,p,V) where pdag[parent,V] == 1 and pdag[V,parent] == 0.
conditioning_set = sorted unique union(S, parents).
residV = residual_cache->get(data, V, conditioning_set).
For W in S, p_values push dcov_exact_pvalue(residV, data[,W], options.index, options.legacy_index).
reject_count = number of p_values < options.alpha.
If S is empty, do not call dCov and return reject_count = 0.
```

- [ ] Add `regrvonps_native_selftest_export()` and wrapper:

```r
regrvonps_native_selftest <- function() {
  build_fastkpc_native()
  regrvonps_native_selftest_export()
}
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_regrvonps_native.R
Rscript fastkpc/tests/test_residual_backend_registry.R
Rscript fastkpc/tests/test_fastspline_solver.R
```

Expected:

```text
All tests print PASS.
```

## Phase 4: WAN-PDAG Engine Core

Purpose: wire collider/rules/regrVonPS into one native orientation engine.

- [ ] Create `fastkpc/tests/test_wanpdag_engine_core.R`.

Required helper:

```r
wanpdag_engine_core_selftest()
```

Required result fields:

```text
empty_skeleton_returns_empty_pdag
collider_stage_count_correct
rules_stage_count_correct
generalized_stage_orients_expected_edge
event_log_has_one_based_indices_in_R
residual_backend_params_recorded
cache_stats_recorded
solve_confl_false_no_bidirected
rule_flags_disable_rules
```

Assertions:

```text
all fields are TRUE
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_wanpdag_engine_core.R
```

Expected:

```text
The test fails because wanpdag_engine_core_selftest() does not exist yet.
```

- [ ] Create `fastkpc/src/wanpdag_engine.hpp` and `fastkpc/src/wanpdag_engine.cpp`.

Required API:

```cpp
OrientationOptions default_orientation_options();

OrientationResult orient_wanpdag_native(
  const Rcpp::NumericMatrix& data,
  const SkeletonResult& skeleton,
  const OrientationOptions& options);
```

Engine sequence:

```text
1. Build initial pdag from skeleton adjacency.
2. If no edges, return immediately with pdag and zero counts.
3. If orient_collider, run orient_colliders().
4. Build undirected neighborhoods.
5. Run generalized transitive loop:
   s starts at 1.
   while max undirected neighborhood size >= s:
     for V in 0..p-1:
       if size condition matches legacy:
         enumerate subsets of undirected neighborhood of V of size s2.
         if check_immor accepts subset S:
           pval1 = regrvonps_native(V,S)
           if pval1.reject_count == 0:
             for each W in S:
               search W subsets containing V, as legacy does.
               reject orientation if W can also be regressed to independence.
             if accepted:
               orient S -> V.
               orient remaining undirected neighbors of V as V -> remaining.
               apply rules until converged.
               rebuild undirected neighborhoods.
               mark nbhd_updt[V] = true.
6. Return result counts and events.
```

Subset enumeration:

```text
Use increasing lexicographic combinations.
For a singleton neighborhood, subset matrix behavior must match legacy: one subset containing that one node.
```

- [ ] Add `wanpdag_engine_core_selftest_export()` and wrapper:

```r
wanpdag_engine_core_selftest <- function() {
  build_fastkpc_native()
  wanpdag_engine_core_selftest_export()
}
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_wanpdag_engine_core.R
Rscript fastkpc/tests/test_orientation_rules.R
Rscript fastkpc/tests/test_regrvonps_native.R
```

Expected:

```text
All tests print PASS.
```

## Phase 5: CPU WAN-PDAG Pipeline API

Purpose: expose native orientation as an opt-in CPU pipeline after skeleton.

- [ ] Create `fastkpc/tests/test_wanpdag_cpu_pipeline.R`.

Required wrappers:

```r
fast_orient_wanpdag_cpp(skeleton_result, data,
                        residual_backend = "fastSpline",
                        residual_cache = TRUE,
                        alpha = 0.2,
                        index = 1,
                        legacy_index = TRUE,
                        orient_collider = TRUE,
                        solve_confl = FALSE,
                        rules = c(TRUE, TRUE, TRUE),
                        fastspline_params = list())

fast_kpc_wanpdag_cpp(data, alpha, max_conditioning_size,
                     residual_backend = "fastSpline",
                     residual_cache = TRUE,
                     index = 1,
                     legacy_index = TRUE,
                     orient_collider = TRUE,
                     solve_confl = FALSE,
                     rules = c(TRUE, TRUE, TRUE),
                     fastspline_params = list())
```

Required tests:

```text
1. fast_kpc_wanpdag_cpp returns skeleton and orientation sections.
2. orientation$pdag is integer matrix with square dimension p.
3. orientation$residual_backend == "fastSpline".
4. orientation$residual_cache$hits > 0 on nonlinear fixed scenario.
5. fast_orient_wanpdag_cpp(skeleton, data) matches fast_kpc_wanpdag_cpp(data)$orientation.
6. residual_backend = "linear" remains accepted.
7. disabling orient_collider changes collider count to zero.
8. disabling all rules keeps rule counts zero.
9. repeated runs are deterministic.
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_wanpdag_cpu_pipeline.R
```

Expected:

```text
The test fails because fast_kpc_wanpdag_cpp() does not exist yet.
```

- [ ] Add Rcpp exports in `fastkpc/src/rcpp_exports.cpp`.

Required exports:

```cpp
Rcpp::List fast_orient_wanpdag_cpp_export(Rcpp::NumericMatrix data,
                                          Rcpp::LogicalMatrix adjacency,
                                          Rcpp::List sepsets,
                                          double alpha,
                                          double index,
                                          bool legacy_index,
                                          bool residual_cache,
                                          std::string residual_backend,
                                          Rcpp::List fastspline_params,
                                          bool orient_collider,
                                          bool solve_confl,
                                          Rcpp::LogicalVector rules);

Rcpp::List fast_kpc_wanpdag_cpp_export(Rcpp::NumericMatrix data,
                                       double alpha,
                                       int max_conditioning_size,
                                       double index,
                                       bool legacy_index,
                                       bool residual_cache,
                                       std::string residual_backend,
                                       Rcpp::List fastspline_params,
                                       bool orient_collider,
                                       bool solve_confl,
                                       Rcpp::LogicalVector rules);
```

Conversion helpers:

```text
orientation_result_to_list() returns pdag, events, counts, residual_backend, residual_backend_params, residual_cache.
sepsets_from_R_list() converts 1-based R sepsets to 0-based C++ sepsets.
rules vector length must be exactly 3; otherwise stop "rules must have length 3".
```

- [ ] Add R wrappers in `fastkpc/R/native.R`.

Required wrappers use matrix conversion and storage mode `"double"` for data.

- [ ] Run:

```bash
Rscript fastkpc/tests/test_wanpdag_cpu_pipeline.R
Rscript fastkpc/tests/test_wanpdag_engine_core.R
Rscript fastkpc/tests/test_skeleton_fastspline_cpu.R
```

Expected:

```text
All tests print PASS.
```

## Phase 6: CUDA Skeleton To WAN-PDAG Pipeline

Purpose: let CUDA skeleton output feed the same native orientation engine.

- [ ] Create `fastkpc/tests/test_wanpdag_cuda_pipeline.R`.

Required wrapper:

```r
fast_kpc_wanpdag_cuda(data, alpha, max_conditioning_size,
                      residual_backend = "fastSpline",
                      residual_cache = TRUE,
                      index = 1,
                      legacy_index = TRUE,
                      batch_size = 0,
                      orient_collider = TRUE,
                      solve_confl = FALSE,
                      rules = c(TRUE, TRUE, TRUE),
                      fastspline_params = list())
```

Required tests:

```text
1. CUDA pipeline returns skeleton backend "cuda".
2. CUDA pipeline orientation residual_backend == "fastSpline".
3. CPU pipeline pdag identical to CUDA pipeline pdag.
4. CPU pipeline orientation event counts equal CUDA pipeline event counts.
5. CPU skeleton pMax vs CUDA skeleton pMax max diff < 1e-8.
6. batch_size = 1 and batch_size = 0 produce identical pdag and event counts.
7. CUDA orientation cache stats show hits > 0.
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_wanpdag_cuda_pipeline.R
```

Expected:

```text
The test fails because fast_kpc_wanpdag_cuda() does not exist yet.
```

- [ ] Modify `fastkpc/src/r_api_cuda.cpp`.

Required `.Call` entry:

```cpp
C_fast_kpc_wanpdag_cuda(data, alpha, max_conditioning_size,
                        index, legacy_index, batch_size,
                        residual_cache, residual_backend,
                        fastspline_params, orient_collider,
                        solve_confl, rules)
```

Implementation:

```text
Run run_skeleton_cuda_batch(matrix, skeleton_options, batch_size).
Build OrientationOptions from the same residual backend options.
Run orient_wanpdag_native(matrix, skeleton_result, orientation_options).
Return `Rcpp::List::create(Rcpp::Named("skeleton") = skeleton_result_to_list(skeleton_result, matrix.ncol()), Rcpp::Named("orientation") = orientation_result_to_list(orientation_result))`.
Register method with arity 12.
```

- [ ] Modify `fastkpc/R/cuda_native.R`.

Add wrapper `fast_kpc_wanpdag_cuda()`.

- [ ] Modify `fastkpc/tools/build_cuda_native.sh`.

Compile/link:

```text
orientation_matrix.cpp
orientation_rules.cpp
regrvonps_native.cpp
wanpdag_engine.cpp
```

- [ ] Run:

```bash
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
Rscript fastkpc/tests/test_wanpdag_cuda_pipeline.R
Rscript fastkpc/tests/test_skeleton_fastspline_cuda.R
Rscript fastkpc/tests/test_cuda_residual_cache.R
```

Expected:

```text
All tests print PASS.
```

## Phase 7: Legacy WAN-PDAG Validation

Purpose: compare native orientation against legacy `kpcalg::udag2wanpdag()` where packages permit.

- [ ] Create `fastkpc/tests/test_wanpdag_legacy_validation.R`.

Required helper:

```r
validate_wanpdag_against_legacy(seed = 81, n = 120,
                                alpha = 0.2,
                                max_conditioning_size = 1)
```

Required return:

```text
available
reason_if_unavailable
native
legacy
diff
event_counts
cache_stats
metrics
```

Metrics fields:

```text
pdag_exact
directed_edge_added_count
directed_edge_removed_count
undirected_edge_added_count
undirected_edge_removed_count
bidirected_edge_count_native
max_abs_pdag_diff
```

Test requirements:

```text
1. If pcalg or graph is unavailable, available is FALSE and reason mentions missing package.
2. If available is TRUE, diff has directed/undirected/bidirected sections.
3. Native event_counts exists and all counts are non-negative integers.
4. Native cache stats has computations <= requests.
5. On a hand-written non-statistical fixture, native pdag equals legacy fixture pdag exactly.
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_wanpdag_legacy_validation.R
```

Expected:

```text
The test fails because validate_wanpdag_against_legacy() does not exist yet.
```

- [ ] Create `fastkpc/R/wanpdag_validation.R`.

Required functions:

```r
pdag_edge_summary <- function(pdag)
compare_pdag_matrices <- function(old, new)
legacy_udag2wanpdag_result <- function(data, alpha, max_conditioning_size, env = fastkpc_legacy_env())
validate_wanpdag_against_legacy <- function(seed = 81, n = 120, alpha = 0.2, max_conditioning_size = 1)
compare_wanpdag_cpu_cuda <- function(seed = 82, n = 140, alpha = 0.2, max_conditioning_size = 2)
```

Legacy wrapper behavior:

```text
If pcalg or graph is missing, return available = FALSE with reason.
Build legacy skeleton with fastkpc_legacy_skeleton().
Run env$udag2wanpdag().
Convert legacy graph object to pdag matrix.
Do not modify legacy env files.
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_wanpdag_legacy_validation.R
```

Expected:

```text
test_wanpdag_legacy_validation.R prints PASS.
```

## Phase 8: WAN-PDAG Benchmark And Reporting

Purpose: provide a repeatable small benchmark for the full skeleton + orientation pipeline.

- [ ] Create `fastkpc/tests/test_wanpdag_benchmark.R`.

Required helper:

```r
benchmark_wanpdag_pipelines(seed = 91, n = 160,
                            alpha = 0.2,
                            max_conditioning_size = 2)
```

Required return:

```text
timings data.frame(engine, residual_backend, stage, elapsed_sec)
cache data.frame(engine, residual_backend, stage, requests, hits, computations)
orientation_counts data.frame(engine, residual_backend, collider, rule1, rule2, rule3, generalized, regrvonps_calls)
diff list(cpu_vs_cuda, linear_vs_fastspline)
```

Test requirements:

```text
1. timings has rows for cpu/fastSpline/skeleton, cpu/fastSpline/orientation, cuda/fastSpline/skeleton, cuda/fastSpline/orientation.
2. elapsed_sec values are finite and positive.
3. cache has hits > 0 for fastSpline orientation.
4. CPU-vs-CUDA pdag identical.
5. CPU-vs-CUDA max skeleton pMax diff < 1e-8.
6. linear-vs-fastSpline diff object exists.
```

- [ ] Add helper to `fastkpc/R/wanpdag_validation.R`.

- [ ] Run:

```bash
Rscript fastkpc/tests/test_wanpdag_benchmark.R
```

Expected:

```text
test_wanpdag_benchmark.R prints PASS.
```

## Phase 9: Documentation Contract

Purpose: keep the new orientation stage usable and clearly scoped.

- [ ] Create `fastkpc/tests/test_wanpdag_docs_contract.R`.

Required checks:

```text
README contains "WAN-PDAG Orientation Scope"
README contains "WAN-PDAG API"
README contains "WAN-PDAG Validation"
README contains "WAN-PDAG Benchmark"
README contains "WAN-PDAG Known Limits"
README contains "fast_kpc_wanpdag_cpp"
README contains "fast_kpc_wanpdag_cuda"
README contains "kpcalg::kpc() is not replaced"
README contains "kpcalg/R/*.R files are not modified"
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_wanpdag_docs_contract.R
```

Expected:

```text
The test fails because README has not documented WAN-PDAG yet.
```

- [ ] Update `fastkpc/README.md`.

Required sections:

```text
WAN-PDAG Orientation Scope
WAN-PDAG API
WAN-PDAG Validation
WAN-PDAG Benchmark
WAN-PDAG Known Limits
```

Known limits must state:

```text
WAN-PDAG orientation is opt-in.
kpcalg::kpc() is not replaced.
kpcalg/R/*.R files are not modified.
HSIC and permutation tests are not implemented.
CUDA accelerates skeleton dCov batching only; orientation residuals remain CPU-side.
unfVect conservative/majority exclusions are accepted as input but not fully implemented in native orientation.
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_wanpdag_docs_contract.R
```

Expected:

```text
test_wanpdag_docs_contract.R prints PASS.
```

## Phase 10: Build Hygiene And API Search

Purpose: verify every new native/R/CUDA surface is wired.

- [ ] Run:

```bash
rg -n "orientation_matrix|orientation_rules|regrvonps_native|wanpdag_engine|fast_kpc_wanpdag|fast_orient_wanpdag|WAN-PDAG" fastkpc/src fastkpc/R fastkpc/tools fastkpc/tests fastkpc/README.md
```

Expected:

```text
Every new source file, wrapper, test, build script, and README has matching lines.
```

- [ ] Run:

```bash
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
Rscript -e 'source("fastkpc/R/native.R"); build_fastkpc_native(rebuild=TRUE)'
```

Expected:

```text
CUDA native build succeeds.
CPU sourceCpp build succeeds.
```

## Phase 11: Completion Criteria

The goal is complete only when all criteria are true:

```text
1. Existing fastkpc tests still pass:
   - fastkpc/tests/test_dcov_exact.R
   - fastkpc/tests/test_skeleton_mvp.R
   - fastkpc/tests/test_diff_report.R
   - fastkpc/tests/test_cuda_build_contract.R
   - fastkpc/tests/test_dcov_cuda_batch.R
   - fastkpc/tests/test_skeleton_cuda_batch.R
   - fastkpc/tests/test_residual_cache_core.R
   - fastkpc/tests/test_skeleton_residual_cache.R
   - fastkpc/tests/test_cuda_residual_cache.R
   - fastkpc/tests/test_fastspline_basis.R
   - fastkpc/tests/test_fastspline_solver.R
   - fastkpc/tests/test_residual_backend_registry.R
   - fastkpc/tests/test_skeleton_fastspline_cpu.R
   - fastkpc/tests/test_skeleton_fastspline_cuda.R
   - fastkpc/tests/test_fastspline_mgcv_validation.R
   - fastkpc/tests/test_fastspline_benchmark.R

2. New WAN-PDAG tests pass:
   - fastkpc/tests/test_orientation_matrix.R
   - fastkpc/tests/test_orientation_rules.R
   - fastkpc/tests/test_regrvonps_native.R
   - fastkpc/tests/test_wanpdag_engine_core.R
   - fastkpc/tests/test_wanpdag_cpu_pipeline.R
   - fastkpc/tests/test_wanpdag_cuda_pipeline.R
   - fastkpc/tests/test_wanpdag_legacy_validation.R
   - fastkpc/tests/test_wanpdag_benchmark.R
   - fastkpc/tests/test_wanpdag_docs_contract.R

3. CPU WAN-PDAG pipeline is deterministic:
   - repeated skeleton adjacency identical
   - repeated pdag identical
   - repeated orientation counts identical
   - repeated event log identical

4. CPU-vs-CUDA WAN-PDAG validation reports:
   - pdag_identical TRUE
   - orientation_counts_identical TRUE
   - max_skeleton_pmax_diff < 1e-8
   - orientation cache hits > 0

5. Legacy validation reports:
   - available TRUE with diff report, or available FALSE with explicit missing pcalg/graph reason
   - event counts are non-negative
   - cache computations <= requests
   - hand-written non-statistical fixture exactly matches expected legacy pdag

6. README documents WAN-PDAG scope, API, validation, benchmark, and known limits.

7. CUDA build script compiles and links all new orientation sources.

8. No kpcalg/R/*.R file has been modified.
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
Rscript fastkpc/tests/test_cuda_build_contract.R
Rscript fastkpc/tests/test_dcov_cuda_batch.R
Rscript fastkpc/tests/test_skeleton_cuda_batch.R
Rscript fastkpc/tests/test_residual_cache_core.R
Rscript fastkpc/tests/test_skeleton_residual_cache.R
Rscript fastkpc/tests/test_cuda_residual_cache.R
Rscript fastkpc/tests/test_fastspline_basis.R
Rscript fastkpc/tests/test_fastspline_solver.R
Rscript fastkpc/tests/test_residual_backend_registry.R
Rscript fastkpc/tests/test_skeleton_fastspline_cpu.R
Rscript fastkpc/tests/test_skeleton_fastspline_cuda.R
Rscript fastkpc/tests/test_fastspline_mgcv_validation.R
Rscript fastkpc/tests/test_fastspline_benchmark.R
Rscript fastkpc/tests/test_orientation_matrix.R
Rscript fastkpc/tests/test_orientation_rules.R
Rscript fastkpc/tests/test_regrvonps_native.R
Rscript fastkpc/tests/test_wanpdag_engine_core.R
Rscript fastkpc/tests/test_wanpdag_cpu_pipeline.R
Rscript fastkpc/tests/test_wanpdag_cuda_pipeline.R
Rscript fastkpc/tests/test_wanpdag_legacy_validation.R
Rscript fastkpc/tests/test_wanpdag_benchmark.R
Rscript fastkpc/tests/test_wanpdag_docs_contract.R
Rscript -e 'source("fastkpc/R/wanpdag_validation.R"); print(validate_wanpdag_against_legacy()); print(compare_wanpdag_cpu_cuda()); print(benchmark_wanpdag_pipelines())'
cd kpcalg
md5sum -c MD5 | rg '^R/'
```

When marking this goal complete, report:

```text
Exact build commands used.
Exact test commands run.
Pass/fail result for every existing and new test.
CPU WAN-PDAG deterministic validation.
CPU-vs-CUDA WAN-PDAG pdag diff and max skeleton pMax diff.
Legacy validation availability and diff metrics.
WAN-PDAG benchmark timing table.
Orientation cache stats.
kpcalg/R MD5 result.
```

## Later Goals

Create separate goals after this WAN-PDAG orientation goal is complete.

### Later Goal E: CUDA Residual Kernels

Objective:

```text
Move fastSpline basis evaluation and batched small-system solves to CUDA after the native WAN-PDAG pipeline and graph validation are stable.
```

### Later Goal F: Public kPC-Compatible Wrapper

Objective:

```text
Add an opt-in fastkpc wrapper that mirrors the high-level kpcalg::kpc() call shape while returning explicit fastkpc-native skeleton, orientation, validation, and benchmark sections without modifying kpcalg/R.
```

### Later Goal G: Larger Validation Campaign

Objective:

```text
Run a broader graph-level validation campaign across seeds, sample sizes, graph patterns, residual backends, and CPU/CUDA engines, then write a reproducible report under fastkpc/reports.
```

## Execution Rules For Codex

- Work in small commits if the workspace is a git repository. If it is not a git repository, do not initialize one unless the user asks.
- Use TDD: write each new test before implementing the corresponding code.
- Prefer new files under `fastkpc/` over modifying existing files unless the API needs to be wired.
- Do not alter `kpcalg/R/*.R`.
- Keep WAN-PDAG orientation opt-in.
- Preserve existing fastSpline skeleton behavior.
- Do not make legacy mgcv graph differences disappear by weakening tests; report them.
- Debug CPU-vs-CUDA differences in this order: skeleton pMax, skeleton sepsets, orientation input pdag, orientation event log, residual p-values.
- Do not implement CUDA residual kernels, HSIC, permutation tests, multi-GPU scheduling, or `kpcalg::kpc()` replacement in this goal.

## Final Handoff Prompt

After this plan is saved, the next Codex run can be started with:

```text
Create a goal with objective: "Build the next staged fast kPC backend slice from docs/superpowers/plans/2026-06-13-fast-kpc-wanpdag-orientation-goal-execution.md: add an opt-in native WAN-PDAG orientation stage that consumes fastkpc skeleton results, reuses fastSpline residual-cache logic for generalized transitive orientation checks, validates against legacy kpcalg::udag2wanpdag(), and keeps kpcalg/R files unchanged."
```

Then execute the plan with `superpowers:subagent-driven-development` or `superpowers:executing-plans`.
