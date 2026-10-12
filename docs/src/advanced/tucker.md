# Tucker Methods

For ``\mathcal A\in\mathbb R^{n_1\times\cdots\times n_d}``, a Tucker
approximation with multilinear rank ``(r_1,\ldots,r_d)`` is

```math
\hat{\mathcal A}
=\mathcal G\times_1U^{(1)}\cdots\times_dU^{(d)},
\qquad
U^{(k)}\in\mathbb R^{n_k\times r_k}.
```

The core ``\mathcal G`` stores the compressed coordinates. Each factor matrix
maps one compressed mode back to its original size.

## ST-HOSVD

Sequentially Truncated HOSVD processes one mode at a time. For mode ``k`` it:

1. finds the leading ``r_k`` left singular vectors of the current mode-``k``
   unfolding ``B_{(k)}``;
2. stores them in ``U^{(k)}``;
3. replaces the working tensor by
   ``\mathcal B\times_kU^{(k)\mathsf T}``.

The automatic rank-aware order processes modes in decreasing order of
``n_k/r_k``. This reduces the most strongly compressed mode first. An explicit
order can be supplied when needed:

```julia
result = tucker(
    A,
    ranks;
    method = :sthosvd,
    processing_order = [2, 3, 1],
)
```

When ranks are not supplied, `optimal_mode_order(dims)` uses the size-only
ordering from the ST-HOSVD algorithm.

## Randomized ST-HOSVD

For mode ``k``, the randomized backend uses a test matrix ``\Omega`` and forms

```math
Y=B_{(k)}\Omega,
\qquad
Q=\operatorname{orth}(Y).
```

The number of sketch columns is ``\ell=r_k+p``, where ``p`` is
`oversampling`. With ``q`` power iterations, the sketch becomes

```math
Y=\left(B_{(k)}B_{(k)}^\mathsf T\right)^qB_{(k)}\Omega.
```

TensorKitchen computes these products by tensor contractions. It does not
store the complete unfolding or test matrix. The algorithm sketches all
conceptual unfolding columns; it does not select a random subset of columns.

```julia
result = tucker(
    A,
    ranks;
    method = :sthosvd,
    svd_backend = :randomized,
    oversampling = 16,
    power_iterations = 1,
    block_columns = 65_536,
)
```

`block_columns` limits the number of conceptual unfolding columns processed at
once. It changes temporary memory use, not the requested Tucker ranks.

The implemented power loop orthonormalizes after each complete application of
``B_{(k)}B_{(k)}^\mathsf T``. It is not the fully stabilized variant that
orthonormalizes between both matrix products. Large values of
`power_iterations` can therefore lose small singular directions through
roundoff.

## HOOI

Higher-Order Orthogonal Iteration updates one factor matrix at a time. To
update mode ``k``, it projects the target along every other mode and computes
the leading ``r_k`` left singular vectors of the projected tensor.

```julia
result = tucker(
    A,
    ranks;
    method = :hooi,
    init = :sthosvd,
    maxiter = 50,
    tol = 1e-8,
)
```

```@docs
hooi
```

## Classical T-HOSVD

T-HOSVD computes every factor matrix from the original tensor and projects to
the core only after all factors have been found.

```@docs
thosvd
```

## Error measures

For every Tucker result, use

```julia
rel_error(A, result)
```

For exact ST-HOSVD, the stored singular values also give

```math
\left\|\mathcal A-\hat{\mathcal A}\right\|_F^2
=\sum_{k=1}^{d}\sum_{j>r_{p_k}}\sigma_{k,j}^2,
```

where ``p_k`` is the mode processed at step ``k``. `error_bound(result)`
computes this value. Randomized ST-HOSVD does not store all discarded singular
values, so it does not provide `error_bound`.

## API

```@docs
tucker
optimal_mode_order
processing_order
singular_values
sthosvd
error_bound
```

See [References](../references.md) for the ST-HOSVD and randomized
range-finding sources.
