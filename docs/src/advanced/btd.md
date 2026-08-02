# Advanced BTD Methods

A homogeneous block term decomposition with ``B`` Tucker blocks is

```math
\hat{\mathcal A}
= \sum_{b=1}^{B}
  \mathcal G_b \times_1 U_b^{(1)} \cdots \times_N U_b^{(N)},
```

where every block uses the same multilinear rank tuple in the current API.

## Initialization and refinement

BTD is nonconvex, so the initial blocks can materially affect the final fit.
Two complementary initialization ideas are useful:

- HOSVD multistart generates several structured candidates and keeps the best
  screened candidate.
- An ALS warm start improves the candidate before a manifold solver refines it.

The alternating path can be used directly, or as a warm start and optional
polishing step around a manifold refinement method. See the [`btd`](@ref) API
for the currently supported choices and defaults.

## Alternating block updates

For block ``b``, define the conceptual residual excluding that block:

```math
\mathcal R_b
= \mathcal A - \sum_{c\ne b}\mathcal X_c.
```

BTD-ALS updates the core and factors of ``\mathcal X_b`` while the other blocks
are fixed. An exact least-squares block minimization cannot increase the
objective. TensorKitchen's finite HOOI/ST-HOSVD block updates are approximate,
so strict monotonic decrease is not guaranteed for every configured update;
monitor `rel_error` and compare initializations.

## Initializer objects

Advanced users can configure the warm-start objects directly:

```julia
init = BTDHOSVDMultistartInit(
    24;
    screening_steps = 5,
    block_method = :hooi,
    block_maxiter = 10,
    seed = 0,
)

result = btd(A, blocks, ranks; solver = :als, init = init)
```

`BTDALSWarmStartInit` wraps a base initializer with a fixed number of initial
ALS steps before manifold refinement.

```@docs
BTDHOSVDMultistartInit
BTDALSWarmStartInit
```

## Choosing computational budgets

Initialization, warm-start, block-update, polishing, and restart budgets trade
runtime for additional opportunities to improve a nonconvex fit. Increase them
only after checking whether the extra work gives a meaningful reduction in
`rel_error(A, result)`. The exact keyword names and current defaults live in the
[`btd`](@ref) docstring rather than in this guide.

Interpret the main controls as follows:

- `warm_steps` sets the number of BTD-ALS warm-start iterations.
- `warm_rel_error_gate` is a failure cutoff: if the warm-start error is above
  the gate, the expensive manifold refinement is skipped. Use `nothing` to
  disable this short circuit.
- `block_method` chooses the Tucker update used inside BTD-ALS; `:hooi` performs
  iterative block refinement and `:sthosvd` performs one sequential pass.
- `btd_als_polish_maxiter` controls the optional final ALS polish; use `0` to
  disable it.
- `max_stagnation_restarts` limits retries when ALS fit change is small while
  the final error remains above `stagnation_rel_error`.

## High-level BTD API

The complete initializer, solver, warm-start, block-update, polishing, restart,
stopping, and diagnostic option list is inserted from the source docstring:

```@docs
btd
```

The lower-level ALS and BTD-specific tangent-subspace interfaces are documented
below for experiments that need direct solver control:

```@docs
fit_btd_als
BTDTSDSolver
```

## Storage considerations

The decomposition result stores compact cores and factors. Calling
`reconstruct(result)` creates the full approximation and can dominate memory for
large inputs. Inspect `blocks(result)`, `core(block)`, and `factors(block)` when a
dense reconstruction is not required.

See [Optimization methods](optimization.md) for solver comparison and result
diagnostics.
