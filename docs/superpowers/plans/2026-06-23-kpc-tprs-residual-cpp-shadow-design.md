# kpcTprsResidualCPP Shadow Design Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Design the `kpcTprsResidualCPP` shadow replacement for the restricted kPC single-smooth TPRS residual semantics currently covered by the version-pinned mgcv oracle.

**Architecture:** This is a design-only milestone. It produces a versioned design memo and acceptance plan for a future standalone C++ residual engine that runs beside `mgcvExtract` in shadow mode and never drives graph decisions until residual-, CI-, and graph-level gates pass.

**Tech Stack:** R, C++/Rcpp, existing fastkpc mgcvExtract oracle, existing C++ spectral scoring, existing CUDA fixed-sp batch solve, existing CUDA dCov, Markdown design docs.

---

## Scope

Create a design memo for `kpcTprsResidualCPP`, not implementation code.

The design memo must freeze:

- The name `kpcTprsResidualCPP`.
- The restricted input, supported, and unsupported surfaces.
- The shared oracle/candidate return contract.
- Shadow mode where mgcvExtract remains authoritative.
- Phases 1-5 from C++ TPRS setup through limited switch.
- Gate A through Gate E.
- Difficult numerical cases.
- Go / no-go criteria.
- Explicit non-goals.

The memo must use this exact positioning:

> `kpcTprsResidualCPP` is a standalone C++ residual engine for the restricted single-smooth TPRS semantics used by kPC. It is developed first as a shadow candidate against a version-pinned mgcv oracle and does not drive graph decisions until residual-, CI-, and graph-level compatibility gates pass.

The memo must not describe the project as:

- a C++ implementation of `mgcv::gam()`
- a generic GAM framework
- a drop-in replacement for mgcv
- a replacement for all of `kpcalg::regrXonS()`

## File Structure

- Create `docs/kpc-tprs-residual-cpp-shadow-design.md`
  - Versioned design memo that freezes the first `kpcTprsResidualCPP` contract, shadow architecture, phases, gates, hard cases, and go/no-go criteria.
- Do not modify `fastkpc/R/`, `fastkpc/src/`, `fastkpc/tests/`, or CUDA code in this design milestone.
- Do not add benchmark artifacts or generated CSVs.

## Task 1: Write The Design Memo

**Files:**
- Create: `docs/kpc-tprs-residual-cpp-shadow-design.md`

- [ ] **Step 1: Create the design memo**

Write `docs/kpc-tprs-residual-cpp-shadow-design.md` with these sections, in this order:

1. `# kpcTprsResidualCPP shadow design`
2. `## Positioning`
3. `## First-Version Contract`
4. `## Return Contract`
5. `## Shadow Architecture`
6. `## Implementation Phases`
7. `## Acceptance Metrics`
8. `## Gates`
9. `## Difficult Cases`
10. `## Go / No-Go`
11. `## Explicit Non-Goals`

- [ ] **Step 2: Verify memo was written**

Run:

```bash
test -s docs/kpc-tprs-residual-cpp-shadow-design.md
```

Expected: exit code 0.

- [ ] **Step 3: Check required language is present**

Run:

```bash
rg 'kpcTprsResidualCPP|shadow candidate|not a C\\+\\+ implementation of `mgcv::gam\\(\\)`|residual output only|\\|S\\| = 1 or 2|Gate A|Gate E|mgcv-compatible residual semantics' docs/kpc-tprs-residual-cpp-shadow-design.md
```

Expected: all required phrases are found.

- [ ] **Step 4: Commit**

Run:

```bash
git add docs/kpc-tprs-residual-cpp-shadow-design.md docs/superpowers/plans/2026-06-23-kpc-tprs-residual-cpp-shadow-design.md
git commit -m "docs: design kpc tprs residual shadow"
```

Expected: commit succeeds.

## Task 2: Verification

**Files:**
- Verify: `docs/kpc-tprs-residual-cpp-shadow-design.md`
- Verify: `docs/superpowers/plans/2026-06-23-kpc-tprs-residual-cpp-shadow-design.md`

- [ ] **Step 1: Confirm no production code changed**

Run:

```bash
git show --stat --oneline HEAD
```

Expected: only these docs appear:

```text
docs/kpc-tprs-residual-cpp-shadow-design.md
docs/superpowers/plans/2026-06-23-kpc-tprs-residual-cpp-shadow-design.md
```

- [ ] **Step 2: Confirm working tree is clean**

Run:

```bash
git status --short
```

Expected: no output.

## Self-Review

Spec coverage:

- Name is frozen as `kpcTprsResidualCPP`.
- The memo explicitly rejects generic GAM and mgcv clone scope.
- First-version supported and unsupported surfaces are listed.
- Shadow execution keeps mgcv oracle authoritative.
- Phases 1-5 are listed.
- Setup geometry, fixed-sp, GCV, graph, and switch gates are listed.
- Difficult numerical cases are listed.
- Go / no-go criteria are listed.

Placeholder scan:

- No placeholder instructions remain.

Type consistency:

- Return contract uses the same fields throughout.
- `|S| <= 2` is consistently treated as a single joint isotropic smooth and single-penalty subset.
