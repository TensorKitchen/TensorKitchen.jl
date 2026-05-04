# TensorKitchen Pipeline

This document explains how public APIs route into models, solvers, and result
converters.

## Public entry points

- `cpd(A, r; ...)` -> `CPDResult`
- `nncpd(A, r; ...)` -> `CPDResult`
- `btd(A, blocks, ranks; ...)` -> `BTDResult`
- `tucker(A, ranks; method=...)` -> `TuckerResult`
- `approx(...)` -> `ApproxResult` or auto-routed `CPDResult`/`BTDResult`

## Default behavior (quick reference)

- `cpd(A, r)`:
  - `init = :alswarm`
  - `solver = :rgd`
- `nncpd(A, r)`:
  - `init = :alswarm`
  - `solver = :rgd`
- `btd(A, blocks, ranks)`:
  - `init = :alswarm`
  - `warm_steps = 200`
  - `warm_init = BTDHOSVDMultistartInit(candidates=64, screening_steps=10, block_maxiter=12)`
  - `warm_rel_error_gate = nothing` (run manifold refinement by default; set e.g. `5e-2` to short-circuit on poor warm starts)
  - `solver = :rgd`
  - final BTD-ALS polish enabled by default for non-ALS solvers
  - `max_stagnation_restarts = 1` (retry with stronger multistart when ALS fit-change stalls at high rel-error)
- `tucker(A, ranks)`:
  - `method = :sthosvd`
- `approx(model::JoinModel)`:
  - `init = :alswarm`
  - `warm_steps = 500`
  - `solver = :rgd`

## Core execution architecture

Most optimization APIs share this core pattern:

1. Build a model (`JoinModel` + backend)
2. Call `_solve_model(...)`
3. Convert to a public result struct

`_solve_model` lives in `src/solvers/solve_dispatch.jl` and is the common
symbol-to-solver dispatch layer (`:rgd`, `:rcg`, `:lbfgs`, `:als`, `:btd_tsd`).

## API flows

### CPD (`cpd`, `nncpd`)

`cpd(A, r; ...)`:

1. Build `JoinModel(A, r; geometry=...)` with `CPDBackend`
2. Normalize/validate options (`solver`, `geometry`, `gradient_mode`, normalization policy)
3. Solve through `_solve_model(...)`
4. Optionally run nonnegative ALS polishing (for selected nonnegative paths)
5. Convert to `CPDResult`

Notes:

- `:als` means CP-ALS.
- Manifold solvers (`:rgd`, `:rgd_fixed`, `:rcg`, `:lbfgs`) share dispatch with other pipelines.

### BTD (`btd`)

`btd(A, blocks, ranks; ...)`:

1. Build a uniform Tucker family via `TuckerJoin(...)`
2. Wrap as `JoinModel` with `BTDBackend`
3. Choose effective initializer:
   - `solver == :als`: use requested init directly (default multistart)
   - `solver != :als`: use `BTDALSWarmStartInit(...)` so first-order methods start from a good BTD-ALS warm point
4. If the warm-start rel-error exceeds `warm_rel_error_gate`, return the warm BTD-ALS result directly
5. Otherwise solve through `_solve_model(...)`
6. If `solver != :als`, optionally polish with BTD-ALS (`btd_als_polish_maxiter`)
7. Convert to `BTDResult`

Polish step usefulness (brief):

- Usually helpful for a small final `rel_error` reduction after RGD converges near a good basin.
- Most useful for quality-focused runs (benchmarks, final fits).
- Can be skipped for speed-sensitive runs (`btd_als_polish_maxiter=0`) when small extra gains are not worth runtime.

BTD-specific initialization options:

- `:hosvd`: sequential block initialization on residual
- `:hosvd_multistart`: HOSVD subspace split candidates, optional screening ALS, keep lowest-cost candidate
- `:alswarm`: short BTD-ALS warm-start wrapper around base initializer

BTD-ALS stabilization behavior:

- Tracks per-iteration fit change (`|rel_t - rel_{t-1}|`)
- Detects stagnation when fit change is tiny but `rel_error` remains high
- Can restart from fresh multistart pool (`max_stagnation_restarts`)
- Reports true final Riemannian gradient norm (`grad_norm`) instead of a placeholder

### Tucker (`tucker`)

`tucker(A, ranks; method=...)` does not use `_solve_model`.
It dispatches directly to decomposition routines:

- `:sthosvd`
- `:thosvd` / `:hosvd`
- `:hooi`

## Generic `approx(...)` routing

`approx(manifolds, target; dispatch=:auto)` routes by manifold family:

- uniform `Manifolds.Segre` -> `cpd(...)`
- uniform `Manifolds.Tucker` matching target shape/rank -> `btd(...)`
- mixed or non-uniform family -> generic `JoinModel(...)` path -> `ApproxResult`

`dispatch=:cpd`, `:btd`, and `:generic` force behavior.

## Result types and post-processing

- `CPDResult`
- `BTDResult`
- `TuckerResult`
- `ApproxResult`

Common utilities:

- `reconstruct(result)`
- `rel_error(A, result)`

## File map

- API entry points: `src/api/approx.jl`, `src/api/cpd.jl`, `src/api/nncpd.jl`, `src/api/btd.jl`
- Routing helpers: `src/dispatch/approx_routing.jl`
- Solver dispatch core: `src/solvers/solve_dispatch.jl`
- BTD backend/init details: `src/btd/model.jl`, `src/solvers/btd_als.jl`
