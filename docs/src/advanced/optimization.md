# Advanced Optimization Methods

Optimization-based decompositions minimize a squared reconstruction objective
of the form

```math
f(\theta)
= \frac{1}{2}\left\|\mathcal A-\hat{\mathcal A}(\theta)\right\|_F^2,
```

where ``\theta`` denotes the decomposition parameters.

## Methods by decomposition

### CPD

- `:als` -- alternating least squares, a fast structured baseline.
- `:rgd` -- Riemannian gradient descent with line search; the default
  refinement method.
- `:rgd_fixed` -- Riemannian gradient descent with a fixed step, mainly for
  controlled experiments.
- `:rcg` -- Riemannian conjugate gradient.
- `:lbfgs` -- limited-memory Riemannian quasi-Newton refinement.
- `:lm` -- Levenberg--Marquardt using residual and Jacobian information.

### BTD

- `:als` -- alternating updates of the Tucker blocks.
- `:rgd` -- Riemannian gradient descent with line search; the default
  refinement method.
- `:rgd_fixed` -- fixed-step Riemannian gradient descent.
- `:rcg` -- Riemannian conjugate gradient.
- `:lbfgs` -- limited-memory Riemannian quasi-Newton refinement.
- `:btd_tsd` -- BTD-specific blockwise tangent-subspace descent.

BTD does not currently support `:lm`.

### Generic Join approximation

- `:rgd`
- `:rgd_fixed`
- `:rcg`
- `:lbfgs`
- `:lm`

Generic Join approximation does not provide ALS. When `approx` routes a uniform
Segre or Tucker collection to CPD or BTD, it inherits that specialized method
set.

### Tucker decomposition

Tucker uses direct decomposition routines rather than these optimization solver
objects:

- `tucker(...; method=:sthosvd)` -- sequentially truncated HOSVD.
- `tucker(...; method=:hooi)` -- iterative higher-order orthogonal iteration.
- `thosvd(...)` -- classical truncated HOSVD through its standalone function;
  `:thosvd` is not currently a `tucker` method symbol.

The high-level entry-point docstrings remain the source of truth for current
support and defaults: [`cpd`](@ref), [`btd`](@ref), [`approx`](@ref), and
[`tucker`](@ref).

## Initialization

CPD and BTD are nonconvex. Their default non-ALS paths use an ALS warm start so
the refinement solver begins from a structured fit. For reproducible comparisons,
set the relevant random seed or provide an explicit starting point.

## Stopping and diagnostics

Common controls include:

- `maxiter`: maximum number of refinement iterations;
- `tol`: stopping tolerance;
- `stepsize`: initial or fixed step size for applicable first-order methods;
- `verbose`: progress output.

Solver-specific constructor docstrings below list every advanced control and
its current default. When a solver is selected by symbol, pass those controls
as keywords to the high-level entry point. Passing a constructed solver object
is useful when the same configuration is reused.

Optimization-based result types provide:

```julia
rel_error(result)
iterations(result)
converged(result)
solver(result)
solver_info(result)
```

Use `rel_error(A, result)` when you want to recompute error from an explicit
reconstruction. Compare solver settings on held-out or application-relevant data
rather than selecting them from training error alone.

## Solver API and defaults

The constructor signatures and defaults below are inserted directly from source
docstrings:

```@docs
ALSSolver
RGDSolver
RGDFixedSolver
RCGSolver
LBFGSSolver
LMSolver
```

## Nonnegative CPD

`nncpd` handles nonnegative components through the CPD pipeline. Its standard
configuration is intended to hide geometry details, while advanced experiments
can compare alternative nonnegative geometries. The supported names and current
defaults are maintained in the [`nncpd`](@ref) docstring.
