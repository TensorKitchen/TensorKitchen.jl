# Tucker Methods for Advanced Usage

For an input tensor ``\mathcal A \in \mathbb R^{n_1\times\cdots\times n_d}``, a
Tucker approximation of multilinear rank ``(r_1,\ldots,r_d)`` is

```math
\hat{\mathcal A}
= \mathcal G \times_1 U^{(1)} \cdots \times_d U^{(d)},
\qquad U^{(k)} \in \mathbb R^{n_k\times r_k}.
```

## ST-HOSVD

Sequentially Truncated HOSVD processes one mode at a time. If ``\mathcal B`` is
the current working tensor, the update for mode ``k`` is:

1. obtain the leading ``r_k`` left singular vectors of the mode-``k`` unfolding
   ``B_{(k)}``;
2. store them as ``U^{(k)}``;
3. project the working tensor with ``\mathcal B \leftarrow
   \mathcal B\times_k U^{(k)\mathsf T}``.

The processing order can affect runtime and the final approximation. The
automatic rank-aware heuristic processes modes in decreasing order of
``n_k/r_k`` (equivalently, increasing ``r_k/n_k``), so the mode with the
strongest fractional compression reduces the working tensor first. This is a
storage-reduction heuristic, not a proof of optimal runtime or error.

When ranks are unavailable, `optimal_mode_order(dims)` instead follows the
size-only compact-SVD heuristic from the ST-HOSVD paper and processes modes in
increasing order of ``n_k``. Domain knowledge or benchmarking can still justify
an explicit `processing_order`.

```julia
result = tucker(
    A,
    ranks;
    method = :sthosvd,
    processing_order = [2, 3, 1],
)
```

The high-level dispatcher and current method choices are maintained with the
implementation:

```@docs
tucker
```

## Implicit randomized sketching

For a target rank ``r_k``, randomized ST-HOSVD uses a sketch size
``\ell=r_k+p``, where ``p`` is the oversampling parameter. With a Gaussian test
matrix ``\Omega``, it forms

```math
Y = B_{(k)}\Omega,
\qquad Q = \mathrm{orth}(Y).
```

Optional power iterations replace the basic sketch by

```math
Y = \left(B_{(k)}B_{(k)}^{\mathsf T}\right)^q B_{(k)}\Omega,
```

which can improve the subspace estimate when singular values decay slowly.
After constructing ``Q``, TensorKitchen forms the projected matrix conceptually
as

```math
C = Q^{\mathsf T}B_{(k)}.
```

It then diagonalizes the small Gram matrix
``CC^{\mathsf T}=R\Lambda R^{\mathsf T}`` and uses

```math
U^{(k)} = QR_{[:,1:r_k]},
\qquad
C_{\mathrm{new}} = R_{[:,1:r_k]}^{\mathsf T}C.
```

TensorKitchen evaluates these products through tensor contractions and generates
the Gaussian test matrix in bounded column blocks. It therefore does not retain
a full mode unfolding or a full ``\Omega``. This is a projection of all
conceptual unfolding columns, not random column selection.

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

Increasing `oversampling` or `power_iterations` can improve accuracy but also
increases computation. `block_columns` controls temporary sketch memory rather
than the target rank.

The implemented power loop is algebraically the randomized power method
``(B_{(k)}B_{(k)}^{\mathsf T})^qB_{(k)}\Omega`` and orthonormalizes after each
complete Gram application. It does not perform the intermediate
orthonormalization between every multiplication by ``B_{(k)}^{\mathsf T}`` and
``B_{(k)}`` used by the fully stabilized subspace-iteration variant. Large
`power_iterations` values can therefore lose weak singular directions through
roundoff; use small values and verify `rel_error(A, result)`.

## HOOI

Higher-Order Orthogonal Iteration alternates over the factor matrices. When
updating mode ``k``, it projects ``\mathcal A`` along all other modes and then
selects the leading ``r_k`` left singular vectors of that projected tensor.
Repeated sweeps seek a lower reconstruction error than the initial Tucker fit.

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

T-HOSVD computes every mode factor from the original tensor before projecting
to the core. Unlike ST-HOSVD, it does not shrink the working tensor between mode
factor computations. It is primarily useful as a reference algorithm.

```@docs
thosvd
```

## Error evaluation

Always available:

```julia
rel_error(A, result)
```

For an exact ST-HOSVD result, orthogonality of the sequential projection
residuals gives

```math
\left\|\mathcal A-\hat{\mathcal A}\right\|_F^2
= \sum_{k=1}^{d}\sum_{j>r_{p_k}}\sigma_{k,j}^2,
```

where ``p_k`` is the mode processed at step ``k`` and ``\sigma_{k,j}`` are the
singular values of that step's working unfolding. `error_bound(result)` evaluates
this quantity from stored spectra. Its exact-equality interpretation is specific
to exact ST-HOSVD; use reconstruction-based `rel_error` for HOOI and other Tucker
results. The randomized backend does not store complete discarded spectra and
therefore cannot use `error_bound`.

```@docs
optimal_mode_order
processing_order
singular_values
sthosvd
error_bound
```

See [References](../references.md) for ST-HOSVD and randomized range-finding
sources.
