# BTD Methods

A block term decomposition (BTD) is a sum of Tucker blocks:

```math
\hat{\mathcal A}
=\sum_{b=1}^{B}
  \mathcal G_b\times_1U_b^{(1)}\cdots\times_dU_b^{(d)}.
```

In the current API, all blocks use the same multilinear rank tuple.

## Alternating block updates

When updating block ``b``, the other blocks are held fixed. The target for that
update is the conceptual residual

```math
\mathcal R_b
=\mathcal A-\sum_{c\ne b}\mathcal X_c.
```

BTD-ALS fits a Tucker block to ``\mathcal R_b`` and then moves to the next
block. TensorKitchen can compute the needed projected contractions without
forming this full residual tensor.

## Initialization

TensorKitchen provides three initializer objects:

- `BTDHOSVDMultistartInit` builds several candidates from a materialized
  target and keeps the candidate with the smallest screened error.
- `BTDProjectedMultistartInit` builds and screens compact candidates through
  projected contractions, so it also works with lazy input.
- `BTDALSWarmStartInit` applies a fixed number of BTD-ALS steps to another
  initializer before manifold refinement.

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

For lazy input:

```julia
init = BTDProjectedMultistartInit(
    8;
    screening_steps = 2,
    block_maxiter = 3,
    seed = 0,
)

result = btd(
    counts,
    blocks,
    ranks;
    compute_type = Float32,
    materialize = false,
    solver = :als,
    init = init,
)
```

```@docs
BTDHOSVDMultistartInit
BTDProjectedMultistartInit
BTDALSWarmStartInit
```

## Main controls

- `warm_steps`: number of BTD-ALS iterations before manifold refinement.
- `warm_rel_error_gate`: skip manifold refinement when the warm-start error is
  above this value; use `nothing` to disable the gate.
- `block_method`: Tucker method used for a block update. `:hooi` performs
  repeated factor updates, while `:sthosvd` performs one sequential pass.
- `btd_als_polish_maxiter`: number of final ALS polishing iterations; use `0`
  to disable them.
- `max_stagnation_restarts`: maximum number of retries when ALS changes little
  but its error remains above `stagnation_rel_error`.

The [`btd`](@ref) docstring contains the full option list and current defaults.

## Storage

The projected path stores compact cores, factors, and contraction workspaces.
It does not need a full residual or a separate full reconstruction during block
updates. `reconstruct(result)` does create a tensor with the target dimensions.

Inspect the compact result without reconstruction using:

```julia
terms = blocks(result)
first_core = core(terms[1])
first_factors = factors(terms[1])
```

## API

```@docs
btd
fit_btd_als
BTDTSDSolver
```

See [Optimization methods](optimization.md) for the common solver names and
result accessors.
